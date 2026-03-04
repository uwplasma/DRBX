from __future__ import annotations

import jax.numpy as jnp

from .ops import dpar_centered, ddx_centered


def vorticity_parallel_tendency(
    vpar_e: jnp.ndarray,
    vpar_i: jnp.ndarray,
    *,
    dz: float,
    coeff: float,
) -> jnp.ndarray:
    jpar = vpar_i - vpar_e
    return -float(coeff) * dpar_centered(jpar, float(dz))


def vorticity_curvature_tendency(
    pe: jnp.ndarray,
    bxcv: jnp.ndarray | None,
    *,
    dx: float,
    coeff: float,
) -> jnp.ndarray:
    if bxcv is None or float(coeff) == 0.0:
        return jnp.zeros_like(pe)
    return -float(coeff) * bxcv * ddx_centered(pe, float(dx))
