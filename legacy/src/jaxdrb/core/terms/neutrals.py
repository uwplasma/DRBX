from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.core.closures.neutrals import rhs_neutral
from jaxdrb.core.state import DRBSystemState
from jaxdrb.operators.fd2d import enforce_bc_relaxation

from .context import TermContext
from .ops import laplacian, grid_of, is_2d


def neutrals_terms(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    if not ctx.neut_on or y.N is None:
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

    lap_N = laplacian(ctx.params, ctx.geom, y.N, ctx.bcs.n)
    adv_N = (
        -ctx.geom.bracket(ctx.phi, y.N, bc_phi=ctx.bcs.phi, bc_f=ctx.bcs.n)
        if ctx.params.nonlinear_on
        else jnp.zeros_like(y.N)
    )
    dN, dn_neut, dw_neut = rhs_neutral(
        N=y.N,
        n=y.n,
        omega=y.omega,
        dn0=ctx.params.neutrals,
        adv_N=adv_N,
        lap_N=lap_N,
    )

    if ctx.params.bc_enforce_nu != 0.0 and is_2d(ctx.geom):
        grid = grid_of(ctx.geom)
        if grid is not None:
            dN = dN + enforce_bc_relaxation(
                y.N,
                dx=grid.dx,
                dy=grid.dy,
                bc=ctx.bcs.n,
                nu=ctx.params.bc_enforce_nu,
            )

    return DRBSystemState(
        n=dn_neut,
        omega=dw_neut,
        vpar_e=jnp.zeros_like(y.vpar_e),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=jnp.zeros_like(y.Te),
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=dN,
    )
