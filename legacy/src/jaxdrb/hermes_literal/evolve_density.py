"""Literal density `transform_impl` bootstrap from Hermes."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .boundary_standard import apply_neumann_boundary_average_z
from .types import Field3DLayout


@dataclass(frozen=True)
class DensityTransformResult:
    density: jnp.ndarray


def density_transform_impl(
    density: jnp.ndarray,
    *,
    layout: Field3DLayout,
    evolve_log: bool = False,
    neumann_boundary_average_z: bool = False,
    lower_x: bool = True,
    upper_x: bool = True,
) -> DensityTransformResult:
    """Mirror `EvolveDensity::transform_impl` for the density field itself."""

    arr = (
        jnp.exp(jnp.asarray(density, dtype=jnp.float64))
        if evolve_log
        else jnp.asarray(density, dtype=jnp.float64)
    )
    if neumann_boundary_average_z:
        arr = apply_neumann_boundary_average_z(
            arr,
            layout=layout,
            lower_x=lower_x,
            upper_x=upper_x,
        )
    return DensityTransformResult(density=jnp.maximum(arr, 0.0))
