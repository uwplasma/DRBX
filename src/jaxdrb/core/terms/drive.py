from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState

from .context import TermContext
from .ops import ddy


def drive_terms(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    if (
        ctx.params.omega_n == 0.0
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

    drive_n = -float(ctx.params.omega_n) * dphi_dy * drive_mask
    drive_Te = -float(ctx.params.omega_Te) * dphi_dy * drive_mask
    drive_Ti = -float(ctx.params.omega_Ti) * dphi_dy * drive_mask

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
