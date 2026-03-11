from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState
from jaxdrb.operators.fd2d import enforce_bc_relaxation

from .context import TermContext
from .fields import omega_from_phi
from .parallel import _flux_divergence_open


def _phi_boundary_relaxation_term(ctx: TermContext, phi: jnp.ndarray, nu_phi: float) -> jnp.ndarray:
    if nu_phi == 0.0:
        return jnp.zeros_like(phi)

    grid = getattr(ctx.geom, "grid", None)
    if grid is None or not hasattr(grid, "perp"):
        return jnp.zeros_like(phi)

    dx = float(grid.perp.dx)
    dy = float(grid.perp.dy)

    if phi.ndim == 2:
        return enforce_bc_relaxation(phi, dx=dx, dy=dy, bc=ctx.bcs.phi, nu=nu_phi)

    return jax.vmap(lambda p: enforce_bc_relaxation(p, dx=dx, dy=dy, bc=ctx.bcs.phi, nu=nu_phi))(
        phi
    )


def _soft_floor(x: jnp.ndarray, floor: float) -> jnp.ndarray:
    if floor <= 0.0:
        return x
    floor_val = jnp.asarray(float(floor))
    return 0.5 * (x + jnp.sqrt(x * x + floor_val * floor_val))


def _sound_speed(ctx: TermContext) -> jnp.ndarray:
    """Hermes-style sound speed sqrt(total_pressure / total_density)."""
    denom = 1.0 + float(ctx.params.me_hat)
    denom = max(denom, 1e-12)
    temp_floor = float(ctx.params.temperature_floor)
    Te_eff = _soft_floor(ctx.Te_phys, temp_floor)
    Ti_eff = _soft_floor(ctx.Ti, temp_floor) if ctx.hot_on else ctx.Ti
    total = Te_eff + Ti_eff
    return jnp.sqrt(jnp.maximum(total, 0.0) / denom)


def _div_par_lax(ctx: TermContext, f: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    grid = getattr(ctx.geom, "grid", None)
    if grid is None or not getattr(grid, "open_field_line", False):
        return jnp.zeros_like(f)
    limiter = str(ctx.params.parallel_limiter).lower()
    return _flux_divergence_open(f, jnp.zeros_like(f), float(grid.dz), limiter, wave=wave)


def extra_dissipation_terms(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    """Extra numerical dissipation (parallel phi/omega) and phi BC relaxation."""

    omega = jnp.zeros_like(y.omega)

    nu_phi_par = (
        float(ctx.params.phi_par_dissipation) if bool(ctx.params.phi_dissipation_on) else 0.0
    )
    if nu_phi_par != 0.0:
        with jax.named_scope("phi_par_dissipation"):
            model = str(ctx.params.phi_par_dissipation_model).lower()
            if model in ("lax", "lax_fv", "fv"):
                cs = _sound_speed(ctx)
                div = _div_par_lax(ctx, -ctx.phi, cs)
                omega = omega - nu_phi_par * div
            else:
                # d2par < 0 for Fourier modes, so +nu*d2par damps.
                omega = omega + nu_phi_par * ctx.geom.d2par(ctx.phi, bc_kind="dirichlet")

    nu_vort_par = float(ctx.params.vort_par_dissipation)
    if nu_vort_par != 0.0:
        with jax.named_scope("vort_par_dissipation"):
            omega = omega + nu_vort_par * ctx.geom.d2par(y.omega, bc_kind="dirichlet")

    if bool(ctx.params.phi_sheath_dissipation_on):
        grid = getattr(ctx.geom, "grid", None)
        if grid is not None and getattr(grid, "open_field_line", False) and ctx.phi.ndim == 3:
            if ctx.phi.shape[0] >= 2:
                phisheath_low = 0.5 * (ctx.phi[0] + ctx.phi[1])
                phisheath_high = 0.5 * (ctx.phi[-1] + ctx.phi[-2])
                diss = jnp.zeros_like(ctx.phi)
                diss = diss.at[0].set(jnp.minimum(phisheath_low, 0.0))
                diss = diss.at[-1].set(jnp.minimum(phisheath_high, 0.0))
                omega = omega + diss

    if bool(ctx.params.core_vorticity_damping_on):
        coeff = float(ctx.params.core_vorticity_damping_coeff)
        grid = getattr(ctx.geom, "grid", None)
        if coeff != 0.0 and grid is not None and y.omega.ndim == 3:
            vort_avg = jnp.mean(y.omega[:, 0, :], axis=0)
            damp = jnp.zeros_like(y.omega)
            damp = damp.at[:, 0, :].set(-coeff * vort_avg[None, :])
            omega = omega + damp

    if bool(ctx.params.phi_relax_in_rhs):
        base = float(ctx.params.bc_enforce_nu)
        val = ctx.params.bc_enforce_nu_phi
        nu_phi = base if val is None else float(val)
        if nu_phi != 0.0:
            with jax.named_scope("phi_bc_relaxation"):
                phi_rhs = _phi_boundary_relaxation_term(ctx, ctx.phi, nu_phi)
                omega = omega + omega_from_phi(
                    ctx.params,
                    ctx.geom,
                    phi_rhs,
                    ctx.n_phys,
                    ctx.bcs.phi,
                    Ti=ctx.Ti,
                    Te=ctx.Te_phys,
                )

    return DRBSystemState(
        n=jnp.zeros_like(y.n),
        omega=omega,
        vpar_e=jnp.zeros_like(y.vpar_e),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=jnp.zeros_like(y.Te),
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )
