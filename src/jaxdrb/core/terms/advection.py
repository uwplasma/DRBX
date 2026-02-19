from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState

from .context import TermContext


def exb_advection_terms(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    """Return ExB advection contributions for each evolved field."""

    if not ctx.params.nonlinear_on:
        z = jnp.zeros_like(y.n)
        return DRBSystemState(
            n=z,
            omega=z,
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    bc = ctx.bcs
    scale = ctx.nonlinear_scale
    phi = ctx.phi

    with jax.named_scope("bracket_terms"):
        adv_n = -ctx.geom.bracket(phi, ctx.n_phys, bc_phi=bc.phi, bc_f=bc.n) * scale
        adv_w = -ctx.geom.bracket(phi, y.omega, bc_phi=bc.phi, bc_f=bc.omega) * scale
        adv_ve = -ctx.geom.bracket(phi, y.vpar_e, bc_phi=bc.phi, bc_f=bc.vpar_e) * scale
        adv_vi = -ctx.geom.bracket(phi, y.vpar_i, bc_phi=bc.phi, bc_f=bc.vpar_i) * scale
        adv_Te = -ctx.geom.bracket(phi, ctx.Te_phys, bc_phi=bc.phi, bc_f=bc.Te) * scale
        adv_Ti = (
            -ctx.geom.bracket(phi, ctx.Ti, bc_phi=bc.phi, bc_f=bc.Ti) * scale
            if ctx.hot_on
            else jnp.zeros_like(ctx.Ti)
        )
        adv_psi = -ctx.geom.bracket(phi, ctx.psi, bc_phi=bc.phi, bc_f=bc.psi) * scale

    adv_N = None
    if ctx.neut_on and y.N is not None:
        with jax.named_scope("bracket_terms"):
            adv_N = -ctx.geom.bracket(phi, y.N, bc_phi=bc.phi, bc_f=bc.n)

    return DRBSystemState(
        n=adv_n,
        omega=adv_w,
        vpar_e=adv_ve,
        vpar_i=adv_vi,
        Te=adv_Te,
        Ti=adv_Ti if ctx.hot_on and y.Ti is not None else None,
        psi=adv_psi if y.psi is not None else None,
        N=adv_N if y.N is not None else None,
    )
