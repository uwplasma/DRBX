from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.state import DRBSystemState


def perp_bc_relaxation(
    params: DRBSystemParams, geom: GeometryAdapter, y: DRBSystemState
) -> DRBSystemState:
    if params.perp_bc_nu == 0.0 or not hasattr(geom, "enforce_bc_relaxation"):
        z = jnp.zeros_like(y.n)
        return DRBSystemState(
            n=z,
            omega=jnp.zeros_like(y.omega),
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    def relax(field: jnp.ndarray) -> jnp.ndarray:
        return geom.enforce_bc_relaxation(field, nu=float(params.perp_bc_nu))

    return DRBSystemState(
        n=relax(y.n),
        omega=relax(y.omega),
        vpar_e=relax(y.vpar_e),
        vpar_i=relax(y.vpar_i),
        Te=relax(y.Te),
        Ti=None if y.Ti is None else relax(y.Ti),
        psi=None if y.psi is None else relax(y.psi),
        N=None if y.N is None else relax(y.N),
    )
