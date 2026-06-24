from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..geometry import FciGeometry3D
from .fci import conservative_perp_diffusion_xy


@dataclass(frozen=True)
class FciVorticitySolveResult:
    potential: jnp.ndarray
    residual: jnp.ndarray
    residual_l2: jnp.ndarray
    iterations: int


def apply_fci_vorticity_operator(
    potential: jnp.ndarray,
    density: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    boussinesq: bool = True,
    regularization: float = 1.0e-9,
) -> jnp.ndarray:
    """Apply the positive perpendicular vorticity operator to ``phi``."""

    metric = geometry
    phi = _remove_mean(jnp.asarray(potential, dtype=jnp.float64), metric)
    n = jnp.asarray(density, dtype=jnp.float64)
    if boussinesq:
        coefficient = jnp.ones_like(phi) * jnp.mean(n / jnp.square(metric.Bmag))
    else:
        coefficient = n / jnp.maximum(jnp.square(metric.Bmag), 1.0e-30)
    operator = -conservative_perp_diffusion_xy(phi, coefficient, geometry)
    return _remove_mean(operator, metric) + float(regularization) * phi


def solve_fci_vorticity_potential_cg(
    vorticity: jnp.ndarray,
    density: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    iterations: int = 80,
    boussinesq: bool = True,
    regularization: float = 1.0e-9,
) -> FciVorticitySolveResult:
    """Solve the metric-weighted perpendicular vorticity inversion with CG."""

    metric = geometry
    rhs = _remove_mean(jnp.asarray(vorticity, dtype=jnp.float64), metric)
    x0 = jnp.zeros_like(rhs, dtype=jnp.float64)

    def apply_operator(value: jnp.ndarray) -> jnp.ndarray:
        return apply_fci_vorticity_operator(
            value,
            density,
            geometry,
            boussinesq=boussinesq,
            regularization=regularization,
        )

    r0 = rhs - apply_operator(x0)
    p0 = r0
    rs0 = _inner(r0, r0, metric)

    def body(_index: int, carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    potential, residual, _direction = jax.lax.fori_loop(0, int(iterations), body, (x0, r0, p0))
    residual = rhs - apply_operator(potential)
    residual_l2 = jnp.sqrt(_inner(residual, residual, metric) / jnp.maximum(_inner(rhs, rhs, metric), 1.0e-30))
    return FciVorticitySolveResult(
        potential=potential,
        residual=residual,
        residual_l2=residual_l2,
        iterations=int(iterations),
    )


def _inner(left: jnp.ndarray, right: jnp.ndarray, metric) -> jnp.ndarray:
    return jnp.sum(jnp.asarray(metric.J, dtype=jnp.float64) * left * right)


def _remove_mean(value: jnp.ndarray, metric) -> jnp.ndarray:
    weights = jnp.asarray(metric.J, dtype=jnp.float64)
    mean = jnp.sum(weights * value) / jnp.maximum(jnp.sum(weights), 1.0e-30)
    return value - mean
