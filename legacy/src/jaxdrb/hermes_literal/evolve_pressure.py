"""Literal pressure `transform_impl` bootstrap from Hermes."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .boundary_standard import apply_neumann_boundary_average_z
from .types import Field3DLayout


@dataclass(frozen=True)
class PressureTransformResult:
    pressure: jnp.ndarray
    temperature: jnp.ndarray


def _soft_floor(field: jnp.ndarray, floor: float) -> jnp.ndarray:
    arr = jnp.asarray(field, dtype=jnp.float64)
    floor_val = float(floor)
    if floor_val <= 0.0:
        return arr
    c = jnp.asarray(floor_val, dtype=jnp.float64)
    return 0.5 * (arr + jnp.sqrt(arr * arr + c * c))


def pressure_transform_impl(
    pressure: jnp.ndarray,
    density: jnp.ndarray,
    *,
    layout: Field3DLayout,
    density_floor: float,
    evolve_log: bool = False,
    neumann_boundary_average_z: bool = False,
    lower_x: bool = True,
    upper_x: bool = True,
) -> PressureTransformResult:
    """Mirror `EvolvePressure::transform_impl` for pressure and temperature."""

    p = (
        jnp.exp(jnp.asarray(pressure, dtype=jnp.float64))
        if evolve_log
        else jnp.asarray(pressure, dtype=jnp.float64)
    )
    if neumann_boundary_average_z:
        p = apply_neumann_boundary_average_z(
            p,
            layout=layout,
            lower_x=lower_x,
            upper_x=upper_x,
        )
    n = jnp.asarray(density, dtype=jnp.float64)
    p_floor = jnp.maximum(p, 0.0)
    temperature = p_floor / _soft_floor(n, density_floor)
    return PressureTransformResult(
        pressure=n * temperature,
        temperature=temperature,
    )
