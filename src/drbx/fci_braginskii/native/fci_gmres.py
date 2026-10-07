"""Shard-compatible DRBX adapter for SOLVAX FGMRES."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
from solvax.krylov import gmres as solvax_gmres

from ..geometry.fci_geometry import (
    LocalDomain3D,
    LocalFciGeometry3D,
)


_pytree_base = jax.tree_util.register_pytree_node_class


@_pytree_base
@dataclass(frozen=True)
class SolvaxGmresConfig:
    """Static configuration for shard-compatible SOLVAX FGMRES.

    The config is a PyTree with static auxiliary data so ``restart`` and
    ``maxiter`` can be used for fixed-size allocations inside ``shard_map`` and
    JIT-compiled code.
    """

    tol: float = 1.0e-6
    atol: float = 1.0e-6
    maxiter: int = 50
    restart: int = 50
    acceptance_tol: float | None = None
    acceptance_atol: float | None = None
    project_mean_zero: bool = False
    regularization_epsilon: float = 0.0
    preconditioner: str = "none"
    residual_correction_steps: int = 0

    def __post_init__(self) -> None:
        if int(self.maxiter) <= 0:
            raise ValueError("SolvaxGmresConfig.maxiter must be positive")
        if int(self.restart) <= 0:
            raise ValueError("SolvaxGmresConfig.restart must be positive")
        if float(self.tol) < 0.0 or float(self.atol) < 0.0:
            raise ValueError("SolvaxGmresConfig tolerances must be non-negative")
        acceptance_tol = self.tol if self.acceptance_tol is None else self.acceptance_tol
        acceptance_atol = self.atol if self.acceptance_atol is None else self.acceptance_atol
        if float(acceptance_tol) < 0.0 or float(acceptance_atol) < 0.0:
            raise ValueError("SolvaxGmresConfig acceptance tolerances must be non-negative")
        if float(self.regularization_epsilon) < 0.0:
            raise ValueError("SolvaxGmresConfig.regularization_epsilon must be non-negative")
        if int(self.residual_correction_steps) < 0:
            raise ValueError(
                "SolvaxGmresConfig.residual_correction_steps must be non-negative"
            )
        if self.preconditioner not in (
            "none",
            "jacobi",
            "line-u",
            "line-v",
            "line-uv",
        ):
            raise ValueError(
                "SolvaxGmresConfig.preconditioner must be one of "
                "'none', 'jacobi', 'line-u', 'line-v', or 'line-uv'"
            )
        object.__setattr__(self, "tol", float(self.tol))
        object.__setattr__(self, "atol", float(self.atol))
        object.__setattr__(self, "maxiter", int(self.maxiter))
        object.__setattr__(self, "restart", int(self.restart))
        object.__setattr__(self, "acceptance_tol", float(acceptance_tol))
        object.__setattr__(self, "acceptance_atol", float(acceptance_atol))
        object.__setattr__(self, "project_mean_zero", bool(self.project_mean_zero))
        object.__setattr__(
            self,
            "regularization_epsilon",
            float(self.regularization_epsilon),
        )
        object.__setattr__(self, "preconditioner", str(self.preconditioner))
        object.__setattr__(
            self,
            "residual_correction_steps",
            int(self.residual_correction_steps),
        )

    def tree_flatten(self):
        return (), (
            self.tol,
            self.atol,
            self.maxiter,
            self.restart,
            self.acceptance_tol,
            self.acceptance_atol,
            self.project_mean_zero,
            self.regularization_epsilon,
            self.preconditioner,
            self.residual_correction_steps,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        (
            tol,
            atol,
            maxiter,
            restart,
            acceptance_tol,
            acceptance_atol,
            project_mean_zero,
            regularization_epsilon,
            preconditioner,
            residual_correction_steps,
        ) = aux_data
        return cls(
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            restart=restart,
            acceptance_tol=acceptance_tol,
            acceptance_atol=acceptance_atol,
            project_mean_zero=project_mean_zero,
            regularization_epsilon=regularization_epsilon,
            preconditioner=preconditioner,
            residual_correction_steps=residual_correction_steps,
        )


@_pytree_base
@dataclass(frozen=True)
class SolvaxGmresInfo:
    """Array-valued diagnostics returned by shard-compatible SOLVAX FGMRES."""

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
    # ``converged`` also accepts the relaxed acceptance tolerance; this flag
    # records whether the strict ``tol``/``atol`` target itself was met.
    strict_converged: jnp.ndarray

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
            self.strict_converged,
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
        jnp.asarray(
            geometry.cell_volume_geometry.volume,
            dtype=jnp.float64,
        )
        * jnp.asarray(
            geometry.cell_volume_geometry.volume_fraction,
            dtype=jnp.float64,
        )
        * jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64)
    )


def _normalize_active_cell_mask(
    active_cell_mask: jnp.ndarray | None,
    geometry: LocalFciGeometry3D,
) -> jnp.ndarray | None:
    if active_cell_mask is None:
        return None
    mask = jnp.asarray(active_cell_mask, dtype=bool)
    if mask.shape != geometry.owned_shape:
        raise ValueError(
            "active_cell_mask must have shape "
            f"{geometry.owned_shape}, got {mask.shape}"
        )
    return mask


def _mask_inactive_owned(
    values: jnp.ndarray,
    active_cell_mask: jnp.ndarray | None,
    *,
    inactive_value: float = 0.0,
) -> jnp.ndarray:
    values = jnp.asarray(values, dtype=jnp.float64)
    if active_cell_mask is None:
        return values
    return jnp.where(active_cell_mask, values, jnp.asarray(inactive_value, dtype=values.dtype))


def _spmd_dot(
    x: jnp.ndarray,
    y: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    active_cell_mask: jnp.ndarray | None = None,
    volume_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Global dot product over owned-local vector shards.

    ``volume_weights=None`` deliberately retains the historical Euclidean
    product.  Supplied weights are the unnormalised control-volume measure.
    """

    x = _mask_inactive_owned(x, active_cell_mask)
    y = _mask_inactive_owned(y, active_cell_mask)
    if volume_weights is None:
        integrand = x * y
    else:
        weights = jnp.asarray(volume_weights, dtype=jnp.float64)
        if weights.shape != geometry.owned_shape:
            raise ValueError(
                "volume_weights must have shape "
                f"{geometry.owned_shape}, got {weights.shape}"
            )
        weights = jnp.where(
            active_cell_mask if active_cell_mask is not None else True,
            weights,
            0.0,
        )
        integrand = weights * x * y
    return _spmd_sum(jnp.sum(integrand), domain)


def _spmd_norm(
    x: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    active_cell_mask: jnp.ndarray | None = None,
    volume_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Global Krylov norm over owned-local vector shards."""

    return jnp.sqrt(
        jnp.maximum(
            _spmd_dot(
                x, x, geometry, domain, active_cell_mask, volume_weights
            ),
            0.0,
        )
    )


def _spmd_volume_weights_valid(
    volume_weights: jnp.ndarray | None,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    active_cell_mask: jnp.ndarray | None,
) -> jnp.ndarray:
    """JAX-safe validity predicate for the supplied Krylov measure."""

    if volume_weights is None:
        return jnp.asarray(True)
    weights = jnp.asarray(volume_weights, dtype=jnp.float64)
    active = (
        jnp.ones(geometry.owned_shape, dtype=bool)
        if active_cell_mask is None
        else active_cell_mask
    )
    local = jnp.all((~active) | (jnp.isfinite(weights) & (weights > 0.0)))
    return _spmd_sum(local.astype(jnp.int32), domain) == _spmd_sum(
        jnp.asarray(1, dtype=jnp.int32), domain
    )


def _spmd_weighted_mean(
    field: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    active_cell_mask: jnp.ndarray | None = None,
    volume_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Weighted global mean over owned-local cells."""

    values = _mask_inactive_owned(field, active_cell_mask)
    weights = (
        _local_cell_volume_weights(geometry)
        if volume_weights is None
        else jnp.asarray(volume_weights, dtype=jnp.float64)
    )
    if weights.shape != geometry.owned_shape:
        raise ValueError(
            "volume_weights must have shape "
            f"{geometry.owned_shape}, got {weights.shape}"
        )
    if active_cell_mask is not None:
        weights = jnp.where(active_cell_mask, weights, 0.0)
    numerator = _spmd_sum(jnp.sum(weights * values), domain)
    denominator = _spmd_sum(jnp.sum(weights), domain)
    return numerator / jnp.maximum(denominator, 1.0e-30)


def _spmd_weighted_l2(
    field: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    active_cell_mask: jnp.ndarray | None = None,
    volume_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Weighted global RMS norm over owned-local cells."""

    values = _mask_inactive_owned(field, active_cell_mask)
    weights = (
        _local_cell_volume_weights(geometry)
        if volume_weights is None
        else jnp.asarray(volume_weights, dtype=jnp.float64)
    )
    if weights.shape != geometry.owned_shape:
        raise ValueError(
            "volume_weights must have shape "
            f"{geometry.owned_shape}, got {weights.shape}"
        )
    if active_cell_mask is not None:
        weights = jnp.where(active_cell_mask, weights, 0.0)
    numerator = _spmd_sum(jnp.sum(weights * values * values), domain)
    denominator = _spmd_sum(jnp.sum(weights), domain)
    return jnp.sqrt(numerator / jnp.maximum(denominator, 1.0e-30))


def _spmd_remove_weighted_mean(
    field: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    active_cell_mask: jnp.ndarray | None = None,
    volume_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Remove the global weighted mean from an owned-local field."""

    values = _mask_inactive_owned(field, active_cell_mask)
    result = values - _spmd_weighted_mean(
        values, geometry, domain, active_cell_mask, volume_weights
    )
    return _mask_inactive_owned(result, active_cell_mask)


def _spmd_all_finite(
    field: jnp.ndarray,
    domain: LocalDomain3D,
    active_cell_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    finite = jnp.isfinite(field)
    if active_cell_mask is not None:
        finite = finite | (~active_cell_mask)
    local = jnp.all(finite)
    return _spmd_sum(local.astype(jnp.int32), domain) == _spmd_sum(
        jnp.asarray(1, dtype=jnp.int32),
        domain,
    )


def gmres_cycle_budget(restart: int, maxiter: int) -> tuple[int, int, int]:
    """Split ``maxiter`` into full GMRES(``restart``) cycles plus a remainder.

    Returns ``(restart, full_cycles, tail_restart)`` with
    ``restart * full_cycles + tail_restart == maxiter``; the requested cycle
    length is kept (capped at ``maxiter``) instead of being reduced to a
    common divisor of the budget.
    """

    if restart < 1 or maxiter < 1:
        raise ValueError(
            f"GMRES restart and maxiter must be positive, got {restart} and {maxiter}"
        )
    restart = min(restart, maxiter)
    full_cycles, tail_restart = divmod(maxiter, restart)
    return restart, full_cycles, tail_restart


def solvax_gmres_solve(
    apply_A: Callable[[jnp.ndarray], jnp.ndarray],
    rhs_owned: jnp.ndarray,
    guess_owned: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    config: SolvaxGmresConfig = SolvaxGmresConfig(),
    active_cell_mask: jnp.ndarray | None = None,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    volume_weights: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, SolvaxGmresInfo]:
    """Solve ``A x = rhs`` with SOLVAX FGMRES inside an SPMD transform.

    SOLVAX's default inner product is local to one shard.  The custom inner
    product below retains DRBX's global owned-cell norm by reducing every
    Arnoldi product over the configured ``shard_map`` mesh axes.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError("geometry must be a LocalFciGeometry3D instance")
    if not isinstance(domain, LocalDomain3D):
        raise TypeError("domain must be a LocalDomain3D instance")
    if domain.layout != geometry.layout:
        raise ValueError("domain and geometry must share the same HaloLayout3D")
    if not isinstance(config, SolvaxGmresConfig):
        raise TypeError("config must be a SolvaxGmresConfig instance")

    rhs = jnp.asarray(rhs_owned, dtype=jnp.float64)
    guess = jnp.asarray(guess_owned, dtype=jnp.float64)
    if rhs.shape != geometry.owned_shape:
        raise ValueError(f"rhs_owned must have shape {geometry.owned_shape}, got {rhs.shape}")
    if guess.shape != geometry.owned_shape:
        raise ValueError(
            f"guess_owned must have shape {geometry.owned_shape}, got {guess.shape}"
        )

    active_mask = _normalize_active_cell_mask(active_cell_mask, geometry)
    if volume_weights is not None:
        volume_weights = jnp.asarray(volume_weights, dtype=jnp.float64)
        if volume_weights.shape != geometry.owned_shape:
            raise ValueError(
                "volume_weights must have shape "
                f"{geometry.owned_shape}, got {volume_weights.shape}"
            )
    volume_weights_valid = _spmd_volume_weights_valid(
        volume_weights, geometry, domain, active_mask
    )
    rhs_is_finite = _spmd_all_finite(rhs, domain, active_mask)
    guess_is_finite = _spmd_all_finite(guess, domain, active_mask)
    rhs = _mask_inactive_owned(rhs, active_mask)
    guess = _mask_inactive_owned(guess, active_mask)

    def masked_apply_A(values: jnp.ndarray) -> jnp.ndarray:
        return _mask_inactive_owned(
            apply_A(_mask_inactive_owned(values, active_mask)),
            active_mask,
        )

    if config.project_mean_zero:
        rhs = _spmd_remove_weighted_mean(
            rhs, geometry, domain, active_mask, volume_weights
        )
        guess = _spmd_remove_weighted_mean(
            guess, geometry, domain, active_mask, volume_weights
        )
    projected_rhs_mean = _spmd_weighted_mean(
        rhs, geometry, domain, active_mask, volume_weights
    )
    projected_rhs_l2 = _spmd_weighted_l2(
        rhs, geometry, domain, active_mask, volume_weights
    )

    restart, max_restarts, tail_restart = gmres_cycle_budget(
        int(config.restart), int(config.maxiter)
    )
    dtype = rhs.dtype

    rhs_l2 = _spmd_norm(rhs, geometry, domain, active_mask, volume_weights)
    threshold = jnp.maximum(
        jnp.asarray(config.atol, dtype=dtype),
        jnp.asarray(config.tol, dtype=dtype) * rhs_l2,
    )
    acceptance_threshold = jnp.maximum(
        jnp.asarray(config.acceptance_atol, dtype=dtype),
        jnp.asarray(config.acceptance_tol, dtype=dtype) * rhs_l2,
    )

    initial_residual = _spmd_norm(
        rhs - masked_apply_A(guess),
        geometry,
        domain,
        active_mask,
        volume_weights,
    )

    def global_inner_product(
        left: jnp.ndarray,
        right: jnp.ndarray,
    ) -> jnp.ndarray:
        return _spmd_dot(
            left,
            right,
            geometry,
            domain,
            active_mask,
            volume_weights,
        )

    effective_preconditioner = preconditioner
    if effective_preconditioner is not None:
        user_preconditioner = effective_preconditioner

        def effective_preconditioner(values: jnp.ndarray) -> jnp.ndarray:
            return _mask_inactive_owned(
                user_preconditioner(_mask_inactive_owned(values, active_mask)),
                active_mask,
            )

    def budgeted_gmres(
        target: jnp.ndarray, x0: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Full GMRES(restart) cycles, then one shorter cycle for the remainder
        # of the iteration budget; the second call returns immediately when
        # the first one has already converged.
        result = solvax_gmres(
            masked_apply_A,
            target,
            x0=x0,
            precond=effective_preconditioner,
            inner_product=global_inner_product,
            restart=restart,
            rtol=float(config.tol),
            atol=float(config.atol),
            max_restarts=max_restarts,
        )
        x = result.x
        iterations = jnp.asarray(result.iterations, dtype=jnp.int32)
        if tail_restart:
            tail = solvax_gmres(
                masked_apply_A,
                target,
                x0=x,
                precond=effective_preconditioner,
                inner_product=global_inner_product,
                restart=tail_restart,
                rtol=float(config.tol),
                atol=float(config.atol),
                max_restarts=1,
            )
            x = tail.x
            iterations = iterations + jnp.asarray(tail.iterations, dtype=jnp.int32)
        return x, iterations

    result_x, result_iterations = budgeted_gmres(rhs, guess)
    phi = _mask_inactive_owned(result_x, active_mask)
    if config.project_mean_zero:
        phi = _spmd_remove_weighted_mean(
            phi, geometry, domain, active_mask, volume_weights
        )
    final_residual = _spmd_norm(
        rhs - masked_apply_A(phi),
        geometry,
        domain,
        active_mask,
        volume_weights,
    )
    total_iterations = result_iterations

    # A long flexible Arnoldi cycle can report a small projected residual
    # while the independently recomputed physical residual remains above the
    # configured acceptance floor.  Only in that otherwise-rejected case,
    # solve the true residual equation from zero and add the correction.  This
    # is reliable residual replacement/iterative refinement: it changes the
    # linear solver trajectory, not the polarization operator or its target.
    for _ in range(int(config.residual_correction_steps)):
        needs_correction = (
            jnp.isfinite(final_residual)
            & rhs_is_finite
            & guess_is_finite
            & volume_weights_valid
            & (final_residual > acceptance_threshold)
        )

        def correct(
            operand: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            current_phi, current_residual_norm = operand
            correction_rhs = rhs - masked_apply_A(current_phi)
            correction_x, correction_iterations = budgeted_gmres(
                correction_rhs, jnp.zeros_like(current_phi)
            )
            candidate = _mask_inactive_owned(
                current_phi + correction_x,
                active_mask,
            )
            if config.project_mean_zero:
                candidate = _spmd_remove_weighted_mean(
                    candidate,
                    geometry,
                    domain,
                    active_mask,
                    volume_weights,
                )
            candidate_residual = _spmd_norm(
                rhs - masked_apply_A(candidate),
                geometry,
                domain,
                active_mask,
                volume_weights,
            )
            improved = jnp.isfinite(candidate_residual) & (
                candidate_residual < current_residual_norm
            )
            accepted_phi = jnp.where(improved, candidate, current_phi)
            accepted_residual = jnp.where(
                improved,
                candidate_residual,
                current_residual_norm,
            )
            return (
                accepted_phi,
                accepted_residual,
                correction_iterations,
            )

        def skip(
            operand: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            current_phi, current_residual_norm = operand
            return (
                current_phi,
                current_residual_norm,
                jnp.asarray(0, dtype=jnp.int32),
            )

        phi, final_residual, correction_iterations = lax.cond(
            needs_correction,
            correct,
            skip,
            (phi, final_residual),
        )
        total_iterations = total_iterations + correction_iterations
    phi_is_finite = _spmd_all_finite(phi, domain, active_mask)
    finite_failed = (
        (~jnp.isfinite(initial_residual))
        | (~jnp.isfinite(final_residual))
        | (~rhs_is_finite)
        | (~guess_is_finite)
        | (~phi_is_finite)
        | (~volume_weights_valid)
    )
    strict_converged = (~finite_failed) & (final_residual <= threshold)
    accepted = (~finite_failed) & (
        strict_converged | (final_residual <= acceptance_threshold)
    )
    failed = ~accepted
    info = SolvaxGmresInfo(
        num_steps=total_iterations,
        converged=accepted,
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
        strict_converged=strict_converged,
    )
    return phi, info


__all__ = [
    "SolvaxGmresConfig",
    "SolvaxGmresInfo",
    "_local_cell_volume_weights",
    "_mesh_axis_names",
    "_spmd_dot",
    "_spmd_norm",
    "_spmd_remove_weighted_mean",
    "_spmd_sum",
    "_spmd_weighted_mean",
    "solvax_gmres_solve",
]
