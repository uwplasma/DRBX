from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState

from .context import TermContext


def curvature_terms(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    if not ctx.params.curvature_on:
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

    tau_i = float(ctx.params.tau_i) if ctx.hot_on else 0.0

    with jax.named_scope("curvature"):
        C_phi = ctx.geom.curvature(ctx.phi)
        C_n = ctx.geom.curvature(ctx.n_phys)
        C_Te = ctx.geom.curvature(ctx.Te_phys)
        C_Ti = ctx.geom.curvature(ctx.Ti) if ctx.hot_on else jnp.zeros_like(ctx.Ti)
    C_p = (1.0 + tau_i) * C_n + C_Te + tau_i * C_Ti
    C_T = (2.0 / 3.0) * ((7.0 / 2.0) * C_Te + C_n - C_phi)
    Te_coeff = ctx.params.curvature_Te_coeff
    if Te_coeff is not None:
        C_T = float(Te_coeff) * C_T

    n_term = C_p - C_phi
    n_coeff = float(ctx.params.curvature_n_coeff)
    n_term = n_coeff * n_term

    return DRBSystemState(
        n=n_term,
        omega=C_p,
        vpar_e=jnp.zeros_like(y.vpar_e),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=C_T,
        Ti=(C_Ti if ctx.hot_on and y.Ti is not None else None),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )
