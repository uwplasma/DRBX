from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState

from .context import TermContext


def volume_source_terms(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    """Generic volumetric Gaussian sources."""

    if not bool(ctx.params.source_on):
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

    grid = getattr(ctx.geom, "grid", None)
    if grid is None:
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

    perp = getattr(grid, "perp", grid)
    x = perp.x
    ycoord = perp.y
    x0 = float(ctx.params.source_x0)
    y0 = float(ctx.params.source_y0)
    wx = max(float(ctx.params.source_width_x), 1e-12)
    wy = max(float(ctx.params.source_width_y), 1e-12)

    mode = str(ctx.params.source_x_mode).lower()
    if mode == "bout":
        x_min = jnp.min(x)
        x_max = jnp.max(x)
        denom = jnp.where((x_max - x_min) > 0.0, x_max - x_min, 1.0)
        x_use = (x - x_min) / denom
    else:
        x_use = x

    gx = jnp.exp(-(((x_use - x0) / wx) ** 2))
    profile = gx[:, None]
    if str(ctx.params.source_profile).lower() in ("gaussian_xy", "gaussian2d"):
        gy = jnp.exp(-(((ycoord - y0) / wy) ** 2))
        profile = profile * gy[None, :]

    if y.n.ndim == 3:
        profile = profile[None, :, :]

    src_n = float(ctx.params.source_n0) * profile
    src_Te = float(ctx.params.source_Te0) * profile

    return DRBSystemState(
        n=src_n,
        omega=jnp.zeros_like(y.omega),
        vpar_e=jnp.zeros_like(y.vpar_e),
        vpar_i=jnp.zeros_like(y.vpar_i),
        Te=src_Te,
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )
