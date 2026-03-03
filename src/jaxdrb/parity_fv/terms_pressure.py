from __future__ import annotations

import jax.numpy as jnp

from .flux_parallel import div_parallel_fv
from .ops import dpar_centered


def pressure_parallel_tendencies(
    n: jnp.ndarray,
    Te: jnp.ndarray,
    vpar_e: jnp.ndarray,
    *,
    dn_parallel: jnp.ndarray,
    dz: float,
    limiter: str,
    n_floor: float,
    Te_floor: float,
    flux_coeff: float,
    work_coeff: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n_eff = jnp.maximum(n, float(n_floor))
    Te_eff = jnp.maximum(Te, float(Te_floor))
    pe = n_eff * Te_eff

    flux_term = -float(flux_coeff) * div_parallel_fv(
        pe,
        vpar_e,
        dz=float(dz),
        limiter=limiter,
    )
    work_term = float(work_coeff) * vpar_e * dpar_centered(pe, float(dz))
    dpe = flux_term + work_term
    dTe = (dpe - Te_eff * dn_parallel) / jnp.maximum(n_eff, float(n_floor))
    return dpe, dTe
