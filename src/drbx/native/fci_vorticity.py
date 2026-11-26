from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp

from ..geometry import MetricTensor3D
from .fci import conservative_perp_diffusion_xz


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FciVorticitySolveResult:
    potential: jnp.ndarray
    residual: jnp.ndarray
    residual_l2: jnp.ndarray
    iterations: int = field(metadata=dict(static=True))
    preconditioner: str | None = field(default=None, metadata=dict(static=True))


def apply_fci_vorticity_operator(
    potential: jnp.ndarray,
    density: jnp.ndarray,
    metric: MetricTensor3D,
    *,
    boussinesq: bool = True,
    regularization: float = 1.0e-9,
) -> jnp.ndarray:
    """Apply the positive perpendicular vorticity operator to ``phi``."""

    phi = _remove_mean(jnp.asarray(potential, dtype=jnp.float64), metric)
    n = jnp.asarray(density, dtype=jnp.float64)
    b2 = jnp.square(jnp.asarray(metric.Bxy, dtype=jnp.float64))
    coefficient = jnp.full_like(phi, jnp.mean(n / b2)) if boussinesq else n / jnp.maximum(b2, 1.0e-30)
    operator = -conservative_perp_diffusion_xz(phi, coefficient, metric)
    return _remove_mean(operator, metric) + jnp.asarray(regularization, dtype=jnp.float64) * phi


def solve_fci_vorticity_potential_cg(
    vorticity: jnp.ndarray,
    density: jnp.ndarray,
    metric: MetricTensor3D,
    *,
    iterations: int = 80,
    boussinesq: bool = True,
    regularization: float = 1.0e-9,
    preconditioner: str | None = None,
) -> FciVorticitySolveResult:
    """Solve the metric-weighted perpendicular vorticity inversion with CG."""

    normalized_preconditioner = _normalize_preconditioner_name(preconditioner)
    iters = int(iterations)
    potential, residual, residual_l2 = _solve_fci_vorticity_cg_kernel(
        vorticity,
        density,
        metric,
        regularization,
        iterations=iters,
        boussinesq=bool(boussinesq),
        use_jacobi=(normalized_preconditioner == "jacobi"),
    )
    return FciVorticitySolveResult(
        potential=potential,
        residual=residual,
        residual_l2=residual_l2,
        iterations=iters,
        preconditioner=normalized_preconditioner,
    )


def _vorticity_conductances(density: jnp.ndarray, metric: MetricTensor3D, *, boussinesq: bool):
    n = jnp.asarray(density, dtype=jnp.float64)
    b2 = jnp.square(jnp.asarray(metric.Bxy, dtype=jnp.float64))
    coef = jnp.full_like(n, jnp.mean(n / b2)) if boussinesq else n / jnp.maximum(b2, 1.0e-30)
    jac = jnp.asarray(metric.J, dtype=jnp.float64)
    dx = jnp.asarray(metric.dx, dtype=jnp.float64)
    dz = jnp.asarray(metric.dz, dtype=jnp.float64)
    kx = jac * coef * jnp.asarray(metric.g11, dtype=jnp.float64)
    kz = jac * coef * jnp.asarray(metric.g33, dtype=jnp.float64)
    cx = 0.5 * (kx[1:, :, :] + kx[:-1, :, :]) / jnp.maximum(0.5 * (dx[1:, :, :] + dx[:-1, :, :]), 1.0e-30)
    cz = 0.5 * (kz + jnp.roll(kz, -1, axis=2)) / jnp.maximum(0.5 * (dz + jnp.roll(dz, -1, axis=2)), 1.0e-30)
    dx_safe, dz_safe, jac_safe = jnp.maximum(dx, 1.0e-30), jnp.maximum(dz, 1.0e-30), jnp.maximum(jac, 1.0e-30)
    inv_jac = 1.0 / jac_safe
    return jac, jac / jnp.maximum(jnp.sum(jac), 1.0e-30), cx, cz, inv_jac / dx_safe, inv_jac / dz_safe


@partial(jax.jit, static_argnames=("iterations", "boussinesq", "use_jacobi"))
def _solve_fci_vorticity_cg_kernel(
    vorticity: jnp.ndarray,
    density: jnp.ndarray,
    metric: MetricTensor3D,
    regularization: float,
    *,
    iterations: int,
    boussinesq: bool,
    use_jacobi: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    jac, jac_w, cx, cz, inv_dx, inv_dz = _vorticity_conductances(density, metric, boussinesq=boussinesq)
    reg = jnp.asarray(regularization, dtype=jnp.float64)
    rhs = jnp.asarray(vorticity, dtype=jnp.float64)
    rhs = rhs - jnp.sum(jac_w * rhs)

    def apply_op(phi: jnp.ndarray) -> jnp.ndarray:
        phi = phi - jnp.sum(jac_w * phi)
        fx = cx * (phi[1:] - phi[:-1])
        div_x = jnp.concatenate((fx[:1], fx[1:] - fx[:-1], -fx[-1:]), axis=0) * inv_dx
        fz = cz * (jnp.roll(phi, -1, axis=2) - phi)
        op = -(div_x + (fz - jnp.roll(fz, 1, axis=2)) * inv_dz)
        return (op - jnp.sum(jac_w * op)) + reg * phi

    if use_jacobi:
        x_diag = jnp.concatenate((cx[:1], cx[1:] + cx[:-1], cx[-1:]), axis=0) * inv_dx
        inv_diag = 1.0 / jnp.maximum(x_diag + (cz + jnp.roll(cz, 1, axis=2)) * inv_dz + reg, 1.0e-30)
        precond = lambda r: (z := inv_diag * r) - jnp.sum(jac_w * z)  # noqa: E731
    else:
        precond = lambda r: r  # noqa: E731

    z0 = precond(rhs)
    rz0 = jnp.sum(jac * rhs * z0)

    def body(_index: int, carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]):
        x, r, p, rz = carry
        ap = apply_op(p)
        alpha = rz / jnp.maximum(jnp.sum(jac * p * ap), 1.0e-30)
        x_next = x + alpha * p
        r_next = r - alpha * ap
        z_next = precond(r_next)
        rz_next = jnp.sum(jac * r_next * z_next)
        return x_next, r_next, z_next + (rz_next / jnp.maximum(rz, 1.0e-30)) * p, rz_next

    potential, _, _, _ = jax.lax.fori_loop(0, iterations, body, (jnp.zeros_like(rhs), rhs, z0, rz0))
    potential = potential - jnp.sum(jac_w * potential)
    residual = rhs - apply_op(potential)
    residual_l2 = jnp.sqrt(jnp.sum(jac * residual * residual) / jnp.maximum(jnp.sum(jac * rhs * rhs), 1.0e-30))
    return potential, residual, residual_l2


def _normalize_preconditioner_name(preconditioner: str | None) -> str | None:
    if preconditioner is None:
        return None
    normalized = str(preconditioner).strip().lower().replace("-", "_")
    if normalized in {"", "none", "unpreconditioned"}:
        return None
    if normalized in {"jacobi", "diagonal", "diag"}:
        return "jacobi"
    raise ValueError(f"Unsupported FCI vorticity preconditioner {preconditioner!r}.")


def _fci_vorticity_jacobi_inverse_diagonal(
    density: jnp.ndarray,
    metric: MetricTensor3D,
    *,
    boussinesq: bool,
    regularization: float,
) -> jnp.ndarray:
    _jac, _jw, cx, cz, inv_dx, inv_dz = _vorticity_conductances(density, metric, boussinesq=boussinesq)
    diagonal = jnp.concatenate((cx[:1], cx[1:] + cx[:-1], cx[-1:]), axis=0) * inv_dx
    diagonal = diagonal + (cz + jnp.roll(cz, 1, axis=2)) * inv_dz + float(regularization)
    return 1.0 / jnp.maximum(diagonal, 1.0e-30)


def _remove_mean(value: jnp.ndarray, metric: MetricTensor3D) -> jnp.ndarray:
    weights = jnp.asarray(metric.J, dtype=jnp.float64)
    return value - jnp.sum(weights * value) / jnp.maximum(jnp.sum(weights), 1.0e-30)
