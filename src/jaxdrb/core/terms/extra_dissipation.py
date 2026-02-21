from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState
from jaxdrb.operators.fd2d import enforce_bc_relaxation

from .context import TermContext
from .fields import omega_from_phi


def _phi_boundary_relaxation_term(
    ctx: TermContext, phi: jnp.ndarray, nu_phi: float
) -> jnp.ndarray:
    if nu_phi == 0.0:
        return jnp.zeros_like(phi)

    grid = getattr(ctx.geom, "grid", None)
    if grid is None or not hasattr(grid, "perp"):
        return jnp.zeros_like(phi)

    dx = float(grid.perp.dx)
    dy = float(grid.perp.dy)

    if phi.ndim == 2:
        return enforce_bc_relaxation(phi, dx=dx, dy=dy, bc=ctx.bcs.phi, nu=nu_phi)

    return jax.vmap(
        lambda p: enforce_bc_relaxation(p, dx=dx, dy=dy, bc=ctx.bcs.phi, nu=nu_phi)
    )(phi)


def extra_dissipation_terms(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    """Hermes-style extra numerical dissipation (parallel phi/omega) and phi BC relaxation."""

    omega = jnp.zeros_like(y.omega)

    nu_phi_par = float(ctx.params.phi_par_dissipation)
    if nu_phi_par != 0.0:
        with jax.named_scope("phi_par_dissipation"):
            omega = omega - nu_phi_par * ctx.geom.d2par(ctx.phi, bc_kind="dirichlet")

    nu_vort_par = float(ctx.params.vort_par_dissipation)
    if nu_vort_par != 0.0:
        with jax.named_scope("vort_par_dissipation"):
            omega = omega - nu_vort_par * ctx.geom.d2par(y.omega, bc_kind="dirichlet")

    if bool(ctx.params.phi_relax_in_rhs):
        base = float(ctx.params.bc_enforce_nu)
        val = ctx.params.bc_enforce_nu_phi
        nu_phi = base if val is None else float(val)
        if nu_phi != 0.0:
            with jax.named_scope("phi_bc_relaxation"):
                phi_rhs = _phi_boundary_relaxation_term(ctx, ctx.phi, nu_phi)
                omega = omega + omega_from_phi(ctx.params, ctx.geom, phi_rhs, ctx.n_phys, ctx.bcs.phi)

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
