from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..geometry import MetricTensor3D
from .fci import conservative_perp_diffusion_xz


@dataclass(frozen=True)
class FciVorticitySolveResult:
    potential: jnp.ndarray
    residual: jnp.ndarray
    residual_l2: jnp.ndarray
    iterations: int
    preconditioner: str | None = None


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
    if boussinesq:
        coefficient = jnp.ones_like(phi) * jnp.mean(n / jnp.square(metric.Bxy))
    else:
        coefficient = n / jnp.maximum(jnp.square(metric.Bxy), 1.0e-30)
    operator = -conservative_perp_diffusion_xz(phi, coefficient, metric)
    return _remove_mean(operator, metric) + float(regularization) * phi


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

    rhs = _remove_mean(jnp.asarray(vorticity, dtype=jnp.float64), metric)
    x0 = jnp.zeros_like(rhs, dtype=jnp.float64)
    normalized_preconditioner = _normalize_preconditioner_name(preconditioner)

    def apply_operator(value: jnp.ndarray) -> jnp.ndarray:
        return apply_fci_vorticity_operator(
            value,
            density,
            metric,
            boussinesq=boussinesq,
            regularization=regularization,
        )

    r0 = rhs - apply_operator(x0)
    if normalized_preconditioner == "jacobi":
        inverse_diagonal = _fci_vorticity_jacobi_inverse_diagonal(
            density,
            metric,
            boussinesq=boussinesq,
            regularization=regularization,
        )

        def apply_preconditioner(residual: jnp.ndarray) -> jnp.ndarray:
            return _remove_mean(inverse_diagonal * residual, metric)

        z0 = apply_preconditioner(r0)
        p0 = z0
        rz0 = _inner(r0, z0, metric)

        def body(
            _index: int,
            carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            x, r, p, rz = carry
            ap = apply_operator(p)
            alpha = rz / jnp.maximum(_inner(p, ap, metric), 1.0e-30)
            x_next = _remove_mean(x + alpha * p, metric)
            r_next = r - alpha * ap
            z_next = apply_preconditioner(r_next)
            rz_next = _inner(r_next, z_next, metric)
            beta = rz_next / jnp.maximum(rz, 1.0e-30)
            p_next = z_next + beta * p
            return x_next, r_next, p_next, rz_next

        potential, residual, _direction, _rz = jax.lax.fori_loop(
            0,
            int(iterations),
            body,
            (x0, r0, p0, rz0),
        )
    else:
        p0 = r0

        def body(
            _index: int,
            carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            x, r, p = carry
            ap = apply_operator(p)
            rs = _inner(r, r, metric)
            alpha = rs / jnp.maximum(_inner(p, ap, metric), 1.0e-30)
            x_next = _remove_mean(x + alpha * p, metric)
            r_next = r - alpha * ap
            rs_next = _inner(r_next, r_next, metric)
            beta = rs_next / jnp.maximum(rs, 1.0e-30)
            p_next = r_next + beta * p
            return x_next, r_next, p_next

        potential, residual, _direction = jax.lax.fori_loop(
            0,
            int(iterations),
            body,
            (x0, r0, p0),
        )
    residual = rhs - apply_operator(potential)
    residual_l2 = jnp.sqrt(
        _inner(residual, residual, metric)
        / jnp.maximum(_inner(rhs, rhs, metric), 1.0e-30)
    )
    return FciVorticitySolveResult(
        potential=potential,
        residual=residual,
        residual_l2=residual_l2,
        iterations=int(iterations),
        preconditioner=normalized_preconditioner,
    )


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
    n = jnp.asarray(density, dtype=jnp.float64)
    if boussinesq:
        coefficient = jnp.ones_like(n) * jnp.mean(n / jnp.square(metric.Bxy))
    else:
        coefficient = n / jnp.maximum(jnp.square(metric.Bxy), 1.0e-30)
    jac = jnp.asarray(metric.J, dtype=jnp.float64)
    dx = jnp.asarray(metric.dx, dtype=jnp.float64)
    dz = jnp.asarray(metric.dz, dtype=jnp.float64)
    kx = jac * coefficient * jnp.asarray(metric.g11, dtype=jnp.float64)
    kz = jac * coefficient * jnp.asarray(metric.g33, dtype=jnp.float64)

    dx_face = 0.5 * (dx[1:, :, :] + dx[:-1, :, :])
    kx_face = 0.5 * (kx[1:, :, :] + kx[:-1, :, :])
    x_plus = jnp.zeros_like(n).at[:-1, :, :].set(
        kx_face / jnp.maximum(dx_face, 1.0e-30)
    )
    x_minus = jnp.zeros_like(n).at[1:, :, :].set(
        kx_face / jnp.maximum(dx_face, 1.0e-30)
    )

    dz_plus = 0.5 * (dz + jnp.roll(dz, -1, axis=2))
    kz_face = 0.5 * (kz + jnp.roll(kz, -1, axis=2))
    z_plus = kz_face / jnp.maximum(dz_plus, 1.0e-30)
    z_minus = jnp.roll(z_plus, 1, axis=2)
    diagonal = (
        (x_plus + x_minus) / jnp.maximum(dx, 1.0e-30)
        + (z_plus + z_minus) / jnp.maximum(dz, 1.0e-30)
    ) / jnp.maximum(jac, 1.0e-30)
    diagonal = diagonal + float(regularization)
    return 1.0 / jnp.maximum(diagonal, 1.0e-30)


def _inner(left: jnp.ndarray, right: jnp.ndarray, metric: MetricTensor3D) -> jnp.ndarray:
    return jnp.sum(jnp.asarray(metric.J, dtype=jnp.float64) * left * right)


def _remove_mean(value: jnp.ndarray, metric: MetricTensor3D) -> jnp.ndarray:
    weights = jnp.asarray(metric.J, dtype=jnp.float64)
    mean = jnp.sum(weights * value) / jnp.maximum(jnp.sum(weights), 1.0e-30)
    return value - mean
