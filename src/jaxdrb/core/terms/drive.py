from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState

from .context import TermContext
from .ops import ddx, ddy


def drive_terms(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    use_equilibrium = bool(getattr(ctx.params, "drive_from_equilibrium_on", False))
    if (
        not use_equilibrium
        and ctx.params.omega_n == 0.0
        and ctx.params.omega_Te == 0.0
        and (not ctx.hot_on or ctx.params.omega_Ti == 0.0)
    ):
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

    dphi_dy = (
        ddy(ctx.params, ctx.geom, ctx.phi, ctx.bcs.phi) if ctx.is_2d else ctx.geom.ddy(ctx.phi)
    )

    drive_mask = 1.0
    if ctx.params.sol_on and ctx.mask_open is not None:
        mode = str(ctx.params.omega_drive_mask).lower()
        if mode == "closed":
            drive_mask = ctx.mask_closed
        elif mode == "open":
            drive_mask = ctx.mask_open

    omega_n = float(ctx.params.omega_n)
    omega_Te = float(ctx.params.omega_Te)
    omega_Ti = float(ctx.params.omega_Ti)

    if use_equilibrium:
        mode = str(getattr(ctx.params, "drive_equilibrium_mode", "auto")).lower()
        use_sol = (
            mode in ("auto", "sol")
            and bool(getattr(ctx.params, "sol_on", False))
            and ctx.mask_closed is not None
        )
        n_eq = None
        Te_eq = None
        Ti_eq = None
        if use_sol:
            n_core = float(ctx.params.sol_n_core)
            n_sol = float(ctx.params.sol_n_sol)
            Te_core = float(ctx.params.sol_Te_core)
            Te_sol = float(ctx.params.sol_Te_sol)
            n_eq = n_sol + (n_core - n_sol) * ctx.mask_closed
            Te_eq = Te_sol + (Te_core - Te_sol) * ctx.mask_closed
            if ctx.hot_on:
                Ti_eq = float(ctx.params.tau_i) * Te_eq
        if n_eq is None and mode in ("auto", "constant"):
            n_eq_val = ctx.params.drive_equilibrium_n0
            if n_eq_val is None:
                n_eq_val = ctx.params.n0
            n_eq = jnp.asarray(n_eq_val)
            Te_eq_val = ctx.params.drive_equilibrium_Te0
            if Te_eq_val is not None:
                Te_eq = jnp.asarray(Te_eq_val)
            Ti_eq_val = ctx.params.drive_equilibrium_Ti0
            if Ti_eq_val is not None:
                Ti_eq = jnp.asarray(Ti_eq_val)
        if n_eq is not None:
            n_floor = float(ctx.params.n0_min)
            n_eq = jnp.asarray(n_eq)
            if n_eq.shape != y.n.shape:
                n_eq = jnp.broadcast_to(n_eq, y.n.shape)
            n_eq = jnp.maximum(n_eq, n_floor)
            omega_n = omega_n * (-ddx(ctx.params, ctx.geom, jnp.log(n_eq), ctx.bcs.n))
        if Te_eq is not None:
            Te_floor = max(float(ctx.params.sol_Te_floor), 1e-12)
            Te_eq = jnp.asarray(Te_eq)
            if Te_eq.shape != y.n.shape:
                Te_eq = jnp.broadcast_to(Te_eq, y.n.shape)
            Te_eq = jnp.maximum(Te_eq, Te_floor)
            omega_Te = omega_Te * (-ddx(ctx.params, ctx.geom, jnp.log(Te_eq), ctx.bcs.Te))
        if ctx.hot_on and Ti_eq is not None:
            Ti_floor = max(float(ctx.params.sol_Te_floor), 1e-12)
            Ti_eq = jnp.asarray(Ti_eq)
            if Ti_eq.shape != y.n.shape:
                Ti_eq = jnp.broadcast_to(Ti_eq, y.n.shape)
            Ti_eq = jnp.maximum(Ti_eq, Ti_floor)
            omega_Ti = omega_Ti * (-ddx(ctx.params, ctx.geom, jnp.log(Ti_eq), ctx.bcs.Ti))

    drive_n = -omega_n * dphi_dy * drive_mask
    drive_Te = -omega_Te * dphi_dy * drive_mask
    drive_Ti = -omega_Ti * dphi_dy * drive_mask

    return DRBSystemState(
        n=drive_n,
        omega=jnp.zeros_like(y.omega),
        vpar_e=jnp.zeros_like(y.vpar_e),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=drive_Te,
        Ti=(drive_Ti if ctx.hot_on and y.Ti is not None else None),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )
