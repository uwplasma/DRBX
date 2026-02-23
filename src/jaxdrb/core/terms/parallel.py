from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState

from .context import TermContext
from .ops import laplacian


def _minmod(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    s = 0.5 * (jnp.sign(a) + jnp.sign(b))
    return s * jnp.minimum(jnp.abs(a), jnp.abs(b))


def _limited_slope(f: jnp.ndarray, limiter: str) -> jnp.ndarray:
    df = f[1:] - f[:-1]
    if limiter == "none":
        slope = jnp.zeros_like(f)
        return slope
    df_b = df[:-1]
    df_f = df[1:]
    if limiter == "mc":
        slope = _minmod(_minmod(2.0 * df_b, 2.0 * df_f), 0.5 * (df_b + df_f))
    else:
        slope = _minmod(df_b, df_f)
    slope_full = jnp.zeros_like(f)
    slope_full = slope_full.at[1:-1].set(slope)
    slope_full = slope_full.at[0].set(df[0])
    slope_full = slope_full.at[-1].set(df[-1])
    return slope_full


def _flux_divergence_open(f: jnp.ndarray, v: jnp.ndarray, dz: float, limiter: str) -> jnp.ndarray:
    slope_f = _limited_slope(f, limiter)
    slope_v = _limited_slope(v, limiter)
    f_L = f[:-1] + 0.5 * slope_f[:-1]
    f_R = f[1:] - 0.5 * slope_f[1:]
    v_L = v[:-1] + 0.5 * slope_v[:-1]
    v_R = v[1:] - 0.5 * slope_v[1:]
    amax = jnp.maximum(jnp.abs(v_L), jnp.abs(v_R))
    flux = 0.5 * (f_L * v_L + f_R * v_R) + 0.5 * amax * (f_L - f_R)

    div = jnp.zeros_like(f)
    div = div.at[1:-1].set((flux[1:] - flux[:-1]) / dz)
    div = div.at[0].set((flux[0] - f[0] * v[0]) / dz)
    div = div.at[-1].set((f[-1] * v[-1] - flux[-1]) / dz)
    return div


def _dpar_flux_conservative(ctx: TermContext, f: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    grid = getattr(ctx.geom, "grid", None)
    limiter = str(ctx.params.parallel_limiter).lower()
    if grid is not None and getattr(grid, "open_field_line", False):
        return _flux_divergence_open(f, v, float(grid.dz), limiter)
    return ctx.geom.dpar(f * v, bc_kind="dirichlet")


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
        dpar_Ti = ctx.geom.dpar(ctx.Ti, bc_kind="neumann") if ctx.hot_on else jnp.zeros_like(ctx.Ti)

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

    if bool(ctx.params.parallel_flux_conservative):
        n_eff = jnp.maximum(ctx.n_phys, float(ctx.params.n0_min))
        dn = -_dpar_flux_conservative(ctx, ctx.n_phys, y.vpar_e)
        pe = ctx.n_phys * ctx.Te_phys
        dp_e = -_dpar_flux_conservative(ctx, pe, y.vpar_e)
        dTe = (dp_e - ctx.Te_phys * dn) / n_eff
        if ctx.hot_on:
            pi = ctx.n_phys * ctx.Ti
            dp_i = -_dpar_flux_conservative(ctx, pi, y.vpar_i)
            dTi = (dp_i - ctx.Ti * dn) / n_eff
        else:
            dTi = jnp.zeros_like(par.dpar_vi)
    else:
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
