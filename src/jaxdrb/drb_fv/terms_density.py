from __future__ import annotations

import jax.numpy as jnp

from .flux_parallel import div_parallel_fv


def density_parallel_tendency(
    n: jnp.ndarray,
    vpar_e: jnp.ndarray,
    *,
    dz: float,
    limiter: str,
    n_floor: float,
) -> jnp.ndarray:
    n_eff = jnp.maximum(n, float(n_floor))
    return -div_parallel_fv(
        n_eff,
        vpar_e,
        dz=float(dz),
        limiter=limiter,
    )
