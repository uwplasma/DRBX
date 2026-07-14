from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

from ..geometry import (
    LocalDomain3D,
    LocalFciGeometry3D,
)


_pytree_base = jax.tree_util.register_pytree_node_class


def _as_bool(value: object) -> bool:
    return bool(value)


@_pytree_base
@dataclass(frozen=True)
class SpmdGmresConfig:
    """Static configuration for native SPMD GMRES.

    The config is a PyTree with static auxiliary data so ``restart`` and
    ``maxiter`` can be used for fixed-size allocations inside ``shard_map`` and
    JIT-compiled code.
    """

    tol: float = 1.0e-6
    atol: float = 1.0e-6
    maxiter: int = 50
    restart: int = 50
    stagnation_iters: int = 20
    project_mean_zero: bool = False
    regularization_epsilon: float = 0.0
    check_finite: bool = True

    def __post_init__(self) -> None:
        if int(self.maxiter) <= 0:
            raise ValueError("SpmdGmresConfig.maxiter must be positive")
        if int(self.restart) <= 0:
            raise ValueError("SpmdGmresConfig.restart must be positive")
        if int(self.stagnation_iters) < 0:
            raise ValueError("SpmdGmresConfig.stagnation_iters must be non-negative")
        if float(self.tol) < 0.0 or float(self.atol) < 0.0:
            raise ValueError("SpmdGmresConfig tolerances must be non-negative")
        if float(self.regularization_epsilon) < 0.0:
            raise ValueError("SpmdGmresConfig.regularization_epsilon must be non-negative")
        object.__setattr__(self, "tol", float(self.tol))
        object.__setattr__(self, "atol", float(self.atol))
        object.__setattr__(self, "maxiter", int(self.maxiter))
        object.__setattr__(self, "restart", int(self.restart))
        object.__setattr__(self, "stagnation_iters", int(self.stagnation_iters))
        object.__setattr__(self, "project_mean_zero", _as_bool(self.project_mean_zero))
        object.__setattr__(
            self,
            "regularization_epsilon",
            float(self.regularization_epsilon),
        )
        object.__setattr__(self, "check_finite", _as_bool(self.check_finite))

    def tree_flatten(self):
        return (), (
            self.tol,
            self.atol,
            self.maxiter,
            self.restart,
            self.stagnation_iters,
            self.project_mean_zero,
            self.regularization_epsilon,
            self.check_finite,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        (
            tol,
            atol,
            maxiter,
            restart,
            stagnation_iters,
            project_mean_zero,
            regularization_epsilon,
            check_finite,
        ) = aux_data
        return cls(
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            restart=restart,
            stagnation_iters=stagnation_iters,
            project_mean_zero=project_mean_zero,
            regularization_epsilon=regularization_epsilon,
            check_finite=check_finite,
        )


@_pytree_base
@dataclass(frozen=True)
class SpmdGmresInfo:
    """Array-valued diagnostics returned by native SPMD GMRES."""

    num_steps: jnp.ndarray
    converged: jnp.ndarray
    failed: jnp.ndarray
    initial_residual_l2: jnp.ndarray
    final_residual_l2: jnp.ndarray
    final_residual_rel_l2: jnp.ndarray
    rhs_l2: jnp.ndarray
    projected_rhs_mean: jnp.ndarray
    projected_rhs_l2: jnp.ndarray
    phi_is_finite: jnp.ndarray
    rhs_is_finite: jnp.ndarray
    guess_is_finite: jnp.ndarray

    def __post_init__(self) -> None:
        pass

    def tree_flatten(self):
        children = (
            self.num_steps,
            self.converged,
            self.failed,
            self.initial_residual_l2,
            self.final_residual_l2,
            self.final_residual_rel_l2,
            self.rhs_l2,
            self.projected_rhs_mean,
            self.projected_rhs_l2,
            self.phi_is_finite,
            self.rhs_is_finite,
            self.guess_is_finite,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


def _mesh_axis_names(domain: LocalDomain3D) -> tuple[str, ...]:
    """Return configured collective axis names for a local domain."""

    if not isinstance(domain, LocalDomain3D):
        raise TypeError("domain must be a LocalDomain3D instance")
    return tuple(name for name in domain.mesh_axis_names if name is not None)


def _spmd_sum(value: jnp.ndarray, domain: LocalDomain3D) -> jnp.ndarray:
    """Sum a scalar over every configured SPMD mesh axis."""

    result = jnp.asarray(value)
    for axis_name in _mesh_axis_names(domain):
        result = lax.psum(result, axis_name=axis_name)
    return result


def _local_cell_volume_weights(geometry: LocalFciGeometry3D) -> jnp.ndarray:
    """Return owned-cell weights matching the global phi compatibility norm."""

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError("geometry must be a LocalFciGeometry3D instance")
    return (
        jnp.asarray(geometry.cell_metric.J_owned, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64)
    )


def _spmd_dot(
    x: jnp.ndarray,
    y: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
) -> jnp.ndarray:
    """Unweighted global dot product over owned-local vector shards."""

    del geometry
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    return _spmd_sum(jnp.sum(x * y), domain)


def _spmd_norm(
    x: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
) -> jnp.ndarray:
    """Unweighted global L2 norm over owned-local vector shards."""

    return jnp.sqrt(jnp.maximum(_spmd_dot(x, x, geometry, domain), 0.0))


def _spmd_weighted_mean(
    field: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
) -> jnp.ndarray:
    """Weighted global mean over owned-local cells."""

    values = jnp.asarray(field, dtype=jnp.float64)
    weights = _local_cell_volume_weights(geometry)
    numerator = _spmd_sum(jnp.sum(weights * values), domain)
    denominator = _spmd_sum(jnp.sum(weights), domain)
    return numerator / jnp.maximum(denominator, 1.0e-30)


def _spmd_weighted_l2(
    field: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
) -> jnp.ndarray:
    """Weighted global RMS norm over owned-local cells."""

    values = jnp.asarray(field, dtype=jnp.float64)
    weights = _local_cell_volume_weights(geometry)
    numerator = _spmd_sum(jnp.sum(weights * values * values), domain)
    denominator = _spmd_sum(jnp.sum(weights), domain)
    return jnp.sqrt(numerator / jnp.maximum(denominator, 1.0e-30))


def _spmd_remove_weighted_mean(
    field: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
) -> jnp.ndarray:
    """Remove the global weighted mean from an owned-local field."""

    values = jnp.asarray(field, dtype=jnp.float64)
    return values - _spmd_weighted_mean(values, geometry, domain)


def _spmd_all_finite(field: jnp.ndarray, domain: LocalDomain3D) -> jnp.ndarray:
    local = jnp.all(jnp.isfinite(field))
    return _spmd_sum(local.astype(jnp.int32), domain) == _spmd_sum(
        jnp.asarray(1, dtype=jnp.int32),
        domain,
    )


def _gmres_least_squares_solution(
    H: jnp.ndarray,
    beta: jnp.ndarray,
    step_index: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the small replicated least-squares problem for one GMRES step."""

    restart = int(H.shape[1])
    rows = jnp.arange(restart + 1)
    cols = jnp.arange(restart)
    active_rows = rows <= (step_index + 1)
    active_cols = cols <= step_index
    H_active = jnp.where(active_rows[:, None] & active_cols[None, :], H, 0.0)
    target = jnp.zeros((restart + 1,), dtype=H.dtype).at[0].set(beta)
    y = jnp.linalg.lstsq(H_active, target, rcond=None)[0]
    residual = jnp.linalg.norm(target - H_active @ y)
    return y, residual


def spmd_gmres_solve(
    apply_A: Callable[[jnp.ndarray], jnp.ndarray],
    rhs_owned: jnp.ndarray,
    guess_owned: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    config: SpmdGmresConfig = SpmdGmresConfig(),
) -> tuple[jnp.ndarray, SpmdGmresInfo]:
    """Solve ``A x = rhs`` with restarted GMRES inside an SPMD transform."""

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError("geometry must be a LocalFciGeometry3D instance")
    if not isinstance(domain, LocalDomain3D):
        raise TypeError("domain must be a LocalDomain3D instance")
    if domain.layout != geometry.layout:
        raise ValueError("domain and geometry must share the same HaloLayout3D")
    if not isinstance(config, SpmdGmresConfig):
        raise TypeError("config must be a SpmdGmresConfig instance")

    rhs = jnp.asarray(rhs_owned, dtype=jnp.float64)
    guess = jnp.asarray(guess_owned, dtype=jnp.float64)
    if rhs.shape != geometry.owned_shape:
        raise ValueError(f"rhs_owned must have shape {geometry.owned_shape}, got {rhs.shape}")
    if guess.shape != geometry.owned_shape:
        raise ValueError(
            f"guess_owned must have shape {geometry.owned_shape}, got {guess.shape}"
        )

    rhs_is_finite = _spmd_all_finite(rhs, domain)
    guess_is_finite = _spmd_all_finite(guess, domain)
    if config.project_mean_zero:
        rhs = _spmd_remove_weighted_mean(rhs, geometry, domain)
        guess = _spmd_remove_weighted_mean(guess, geometry, domain)
    projected_rhs_mean = _spmd_weighted_mean(rhs, geometry, domain)
    projected_rhs_l2 = _spmd_weighted_l2(rhs, geometry, domain)

    restart = int(config.restart)
    maxiter = int(config.maxiter)
    stagnation_iters = int(config.stagnation_iters)
    shape = geometry.owned_shape
    dtype = rhs.dtype

    rhs_l2 = _spmd_norm(rhs, geometry, domain)
    threshold = jnp.maximum(
        jnp.asarray(config.atol, dtype=dtype),
        jnp.asarray(config.tol, dtype=dtype) * rhs_l2,
    )

    x0 = guess
    r0 = rhs - apply_A(x0)
    beta0 = _spmd_norm(r0, geometry, domain)
    initial_residual = beta0
    initial_converged = beta0 <= threshold
    initial_failed = (
        (~jnp.isfinite(beta0))
        | (~rhs_is_finite)
        | (~guess_is_finite)
    )
    initial_failed = jnp.where(config.check_finite, initial_failed, False)

    def _outer_cond(carry):
        (
            _x,
            _residual,
            steps,
            converged,
            failed,
            _stale_count,
        ) = carry
        return (steps < maxiter) & (~converged) & (~failed)

    def _outer_body(carry):
        (
            x_base,
            residual,
            steps,
            converged,
            failed,
            stale_count,
        ) = carry

        r = rhs - apply_A(x_base)
        beta = _spmd_norm(r, geometry, domain)
        safe_beta = jnp.maximum(beta, 1.0e-30)
        v0 = jnp.where(beta > 0.0, r / safe_beta, jnp.zeros_like(r))
        V = jnp.zeros((restart + 1,) + shape, dtype=dtype)
        V = V.at[0].set(v0)
        H = jnp.zeros((restart + 1, restart), dtype=dtype)
        x_current = x_base
        residual_current = beta
        converged_current = converged | (beta <= threshold)
        failed_current = failed | (~jnp.isfinite(beta))
        remaining = maxiter - steps
        inner_limit = jnp.minimum(restart, remaining)

        def _inner_cond(inner_carry):
            (
                _V,
                _H,
                _x_current,
                _residual_current,
                converged_current,
                failed_current,
                _stale_count,
                steps_done,
                j,
            ) = inner_carry
            return (
                (j < restart)
                & (steps_done < inner_limit)
                & (~converged_current)
                & (~failed_current)
            )

        def _inner_body(inner_carry):
            (
                V,
                H,
                x_current,
                residual_current,
                converged_current,
                failed_current,
                stale_count,
                steps_done,
                j,
            ) = inner_carry

            vj = V[j]
            w0 = apply_A(vj)

            def _orthogonalize(i, orth_carry):
                w, H = orth_carry
                vi = V[i]
                hij_raw = _spmd_dot(w, vi, geometry, domain)
                hij = jnp.where(i <= j, hij_raw, 0.0)
                w = w - hij * vi
                H = H.at[i, j].set(hij)
                return w, H

            w, H = lax.fori_loop(0, restart, _orthogonalize, (w0, H))
            h_next_raw = _spmd_norm(w, geometry, domain)
            H = H.at[j + 1, j].set(h_next_raw)
            v_next = jnp.where(
                h_next_raw > 1.0e-30,
                w / jnp.maximum(h_next_raw, 1.0e-30),
                jnp.zeros_like(w),
            )
            V = V.at[j + 1].set(v_next)

            y, residual_estimate = _gmres_least_squares_solution(H, beta, j)
            candidate = x_base + jnp.tensordot(y, V[:-1], axes=((0,), (0,)))
            residual_next = residual_estimate
            improved = residual_next < residual_current * (1.0 - 1.0e-12)
            stale_next = jnp.where(improved, 0, stale_count + 1)
            converged_next = converged_current | (residual_next <= threshold)
            stagnated = (
                (stagnation_iters > 0)
                & (stale_next >= stagnation_iters)
                & (residual_next > threshold)
            )
            finite_ok = jnp.isfinite(residual_next) & jnp.all(jnp.isfinite(candidate))
            finite_failed = failed_current | (~finite_ok)
            finite_failed = jnp.where(config.check_finite, finite_failed, failed_current)
            failed_next = finite_failed | stagnated
            return (
                V,
                H,
                candidate,
                residual_next,
                converged_next,
                failed_next,
                stale_next,
                steps_done + jnp.asarray(1, dtype=jnp.int32),
                j + jnp.asarray(1, dtype=jnp.int32),
            )

        (
            _V,
            _H,
            x_current,
            residual_current,
            converged_current,
            failed_current,
            stale_count,
            steps_done,
            _j,
        ) = lax.while_loop(
            _inner_cond,
            _inner_body,
            (
                V,
                H,
                x_current,
                residual_current,
                converged_current,
                failed_current,
                stale_count,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )

        true_residual = _spmd_norm(rhs - apply_A(x_current), geometry, domain)
        converged_current = true_residual <= threshold
        finite_failed = failed_current | (~jnp.isfinite(true_residual))
        failed_current = jnp.where(config.check_finite, finite_failed, failed_current)
        return (
            x_current,
            true_residual,
            steps + steps_done,
            converged_current,
            failed_current,
            stale_count,
        )

    (
        phi,
        final_residual,
        num_steps,
        converged,
        failed,
        _stale_count,
    ) = lax.while_loop(
        _outer_cond,
        _outer_body,
        (
            x0,
            beta0,
            jnp.asarray(0, dtype=jnp.int32),
            initial_converged,
            initial_failed,
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )

    if config.project_mean_zero:
        phi = _spmd_remove_weighted_mean(phi, geometry, domain)
        final_residual = _spmd_norm(rhs - apply_A(phi), geometry, domain)
    phi_is_finite = _spmd_all_finite(phi, domain)
    failed = failed | (~converged)
    failed = failed | jnp.where(config.check_finite, ~phi_is_finite, False)
    info = SpmdGmresInfo(
        num_steps=num_steps,
        converged=converged,
        failed=failed,
        initial_residual_l2=initial_residual,
        final_residual_l2=final_residual,
        final_residual_rel_l2=final_residual / jnp.maximum(rhs_l2, 1.0e-30),
        rhs_l2=rhs_l2,
        projected_rhs_mean=projected_rhs_mean,
        projected_rhs_l2=projected_rhs_l2,
        phi_is_finite=phi_is_finite,
        rhs_is_finite=rhs_is_finite,
        guess_is_finite=guess_is_finite,
    )
    return phi, info

__all__ = [
    "SpmdGmresConfig",
    "SpmdGmresInfo",
    "_local_cell_volume_weights",
    "_mesh_axis_names",
    "_spmd_dot",
    "_spmd_norm",
    "_spmd_remove_weighted_mean",
    "_spmd_sum",
    "_spmd_weighted_mean",
    "spmd_gmres_solve",
]
