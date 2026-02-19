from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState

from .context import TermContext
from .ops import laplacian


class ParallelVars(eqx.Module):
    dpar_ve: jnp.ndarray
    dpar_vi: jnp.ndarray
    dpar_Te: jnp.ndarray
    dpar_Ti: jnp.ndarray
    dpar_j: jnp.ndarray
    dpar_psi: jnp.ndarray
    grad_par_phi_pe: jnp.ndarray
    jpar_total: jnp.ndarray


def parallel_vars(ctx: TermContext, y: DRBSystemState) -> ParallelVars:
    with jax.named_scope("parallel_dpar"):
        dpar_ve = ctx.geom.dpar(y.vpar_e, bc_kind="dirichlet")
        dpar_vi = ctx.geom.dpar(y.vpar_i, bc_kind="dirichlet")
        dpar_Te = ctx.geom.dpar(y.Te, bc_kind="neumann")
        dpar_Ti = (
            ctx.geom.dpar(ctx.Ti, bc_kind="neumann") if ctx.hot_on else jnp.zeros_like(ctx.Ti)
        )

    jpar_fluid = y.vpar_i - y.vpar_e
    jpar_em = (
        -laplacian(ctx.params, ctx.geom, ctx.psi, ctx.bcs.psi)
        if ctx.em_on
        else jnp.zeros_like(jpar_fluid)
    )
    jpar_total = jpar_fluid + jpar_em
    with jax.named_scope("parallel_current"):
        dpar_j = ctx.geom.dpar(jpar_total, bc_kind="dirichlet")

    with jax.named_scope("parallel_grad_phi_pe"):
        grad_par_phi_pe = ctx.geom.dpar(
            ctx.phi
            - ctx.n_phys
            - float(ctx.params.alpha_Te_ohm) * ctx.Te_phys
            - float(ctx.params.alpha_Ti_ohm) * ctx.Ti,
            bc_kind="dirichlet",
        )

    with jax.named_scope("parallel_dpar_psi"):
        dpar_psi = (
            ctx.geom.dpar(ctx.psi, bc_kind="dirichlet") if ctx.em_on else jnp.zeros_like(ctx.psi)
        )

    return ParallelVars(
        dpar_ve=dpar_ve,
        dpar_vi=dpar_vi,
        dpar_Te=dpar_Te,
        dpar_Ti=dpar_Ti,
        dpar_j=dpar_j,
        dpar_psi=dpar_psi,
        grad_par_phi_pe=grad_par_phi_pe,
        jpar_total=jpar_total,
    )


def parallel_conservative_terms(
    ctx: TermContext, y: DRBSystemState, par: ParallelVars
) -> DRBSystemState:
    tau_i = float(ctx.params.tau_i) if ctx.hot_on else 0.0
    vi_par_pressure = ctx.phi + tau_i * (ctx.n_phys + ctx.Ti)

    dn = -par.dpar_ve
    dTe = -(2.0 / 3.0) * par.dpar_ve
    dTi = -(2.0 / 3.0) * par.dpar_vi if ctx.hot_on else jnp.zeros_like(par.dpar_vi)

    vpar_e = par.grad_par_phi_pe / jnp.maximum(float(ctx.params.me_hat), 1e-12) - par.dpar_psi
    vpar_i = -ctx.geom.dpar(vi_par_pressure, bc_kind="dirichlet")

    psi = -par.grad_par_phi_pe if ctx.em_on else jnp.zeros_like(y.omega)

    return DRBSystemState(
        n=dn,
        omega=par.dpar_j,
        vpar_e=vpar_e,
        vpar_i=vpar_i,
        Te=dTe,
        Ti=dTi if ctx.hot_on and y.Ti is not None else None,
        psi=psi if y.psi is not None else None,
        N=None if y.N is None else None,
    )
