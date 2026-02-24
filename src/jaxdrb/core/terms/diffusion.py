from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState

from .context import TermContext
from .ops import laplacian, biharmonic
from .parallel import ParallelVars


def diffusion_terms(
    ctx: TermContext,
    y: DRBSystemState,
    par: ParallelVars,
) -> DRBSystemState:
    n_diff = ctx.n_phys if ctx.params.log_n else y.n
    Te_diff = ctx.Te_phys if ctx.params.log_Te else y.Te

    lap_n = laplacian(ctx.params, ctx.geom, n_diff, ctx.bcs.n)
    bih_n = biharmonic(ctx.params, ctx.geom, n_diff, ctx.bcs.n)
    lap_w = laplacian(ctx.params, ctx.geom, y.omega, ctx.bcs.omega)
    bih_w = biharmonic(ctx.params, ctx.geom, y.omega, ctx.bcs.omega)
    lap_Te = laplacian(ctx.params, ctx.geom, Te_diff, ctx.bcs.Te)
    bih_Te = biharmonic(ctx.params, ctx.geom, Te_diff, ctx.bcs.Te)
    lap_Ti = laplacian(ctx.params, ctx.geom, ctx.Ti, ctx.bcs.Ti)
    bih_Ti = biharmonic(ctx.params, ctx.geom, ctx.Ti, ctx.bcs.Ti)
    lap_psi = laplacian(ctx.params, ctx.geom, ctx.psi, ctx.bcs.psi)
    bih_psi = biharmonic(ctx.params, ctx.geom, ctx.psi, ctx.bcs.psi)

    if ctx.is_2d:
        omega_zonal = jnp.mean(y.omega, axis=1, keepdims=True) + jnp.zeros_like(y.omega)
    else:
        omega_zonal = jnp.zeros_like(y.omega)

    diss_n = (
        float(ctx.params.Dn) * lap_n
        - float(ctx.params.Dn4) * bih_n
        - float(ctx.params.mu_lin_n) * ctx.n_phys
    )
    diss_Te = (
        float(ctx.params.DTe) * lap_Te
        - float(ctx.params.DTe4) * bih_Te
        - float(ctx.params.mu_lin_Te) * ctx.Te_phys
    )
    diss_Ti = float(ctx.params.DTi) * lap_Ti - float(ctx.params.DTi4) * bih_Ti

    mu_zonal = (
        float(ctx.params.mu_zonal_omega) if bool(ctx.params.core_vorticity_damping_on) else 0.0
    )
    mu_lin = float(ctx.params.mu_lin_omega) if bool(ctx.params.core_vorticity_damping_on) else 0.0

    diss_w = (
        float(ctx.params.DOmega) * lap_w
        - float(ctx.params.DOmega4) * bih_w
        - mu_zonal * omega_zonal
        - mu_lin * y.omega
    )

    eta_eff = ctx.params.eta_par if ctx.params.eta_par != 0.0 else ctx.params.eta
    me = jnp.maximum(float(ctx.params.me_hat), 1e-8)
    diss_ve = -(float(eta_eff) / me) * (y.vpar_e - y.vpar_i)
    diss_ve = diss_ve - float(ctx.params.mu_lin_vpar_e) * y.vpar_e
    diss_vi = -float(ctx.params.mu_lin_vpar_i) * y.vpar_i
    # vpar sinks handled as dedicated SOL sink terms

    Ti_diss = (
        diss_Ti + ctx.params.chi_par * ctx.geom.dpar(par.dpar_Ti, bc_kind="neumann")
        if ctx.hot_on
        else jnp.zeros_like(ctx.Ti)
    )
    psi_diss = (
        -float(eta_eff) * par.jpar_total
        + float(ctx.params.Dpsi) * lap_psi
        - float(ctx.params.Dpsi4) * bih_psi
        + float(ctx.params.chi_par) * ctx.geom.dpar(par.dpar_psi, bc_kind="dirichlet")
        if ctx.em_on
        else jnp.zeros_like(ctx.psi)
    )

    return DRBSystemState(
        n=diss_n,
        omega=diss_w,
        vpar_e=diss_ve,
        vpar_i=diss_vi,
        Te=diss_Te,
        Ti=Ti_diss if y.Ti is not None else None,
        psi=psi_diss if y.psi is not None else None,
        N=None if y.N is None else jnp.zeros_like(y.N),
    )
