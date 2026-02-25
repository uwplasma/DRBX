from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState

from .context import TermContext
from .ops import ddx, ddy


def _broadcast_to_shape(arr: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
    if arr.shape == shape:
        return arr
    if arr.ndim == 1:
        if len(shape) == 3 and arr.shape[0] == shape[0]:
            arr = arr[:, None, None]
        elif len(shape) == 2 and arr.shape[0] == shape[0]:
            arr = arr[:, None]
        elif len(shape) == 2 and arr.shape[0] == shape[1]:
            arr = arr[None, :]
    elif arr.ndim == 2 and len(shape) == 3 and arr.shape == shape[1:]:
        arr = arr[None, :, :]
    return jnp.broadcast_to(arr, shape)


def _curv_components(ctx: TermContext, shape: tuple[int, ...]) -> tuple[jnp.ndarray, jnp.ndarray]:
    if hasattr(ctx.geom, "curv_x") and hasattr(ctx.geom, "curv_y"):
        curv_x = _broadcast_to_shape(jnp.asarray(ctx.geom.curv_x), shape)
        curv_y = _broadcast_to_shape(jnp.asarray(ctx.geom.curv_y), shape)
        return curv_x, curv_y
    grid = getattr(ctx.geom, "grid", None)
    perp = getattr(grid, "perp", grid) if grid is not None else None
    if perp is not None and hasattr(perp, "x") and hasattr(perp, "y"):
        x = jnp.asarray(perp.x)
        y = jnp.asarray(perp.y)
        xx = x[:, None] if x.ndim == 1 else x
        yy = y[None, :] if y.ndim == 1 else y
        curv_x = ctx.geom.curvature(xx)
        curv_y = ctx.geom.curvature(yy)
        curv_x = _broadcast_to_shape(curv_x, shape)
        curv_y = _broadcast_to_shape(curv_y, shape)
        return curv_x, curv_y
    return jnp.zeros(shape), jnp.zeros(shape)


def _diamag_form(ctx: TermContext, shape: tuple[int, ...]) -> jnp.ndarray:
    form = getattr(ctx.params, "diamag_form", 1.0)
    profile = getattr(ctx.params, "diamag_form_profile", None)
    if profile:
        grid = getattr(ctx.geom, "grid", None)
        perp = getattr(grid, "perp", grid) if grid is not None else None
        if perp is not None and hasattr(perp, "x"):
            x = jnp.asarray(perp.x)
            x_min = jnp.min(x)
            x_max = jnp.max(x)
            denom = jnp.where((x_max - x_min) > 0.0, x_max - x_min, 1.0)
            x_norm = (x - x_min) / denom
            key = str(profile).lower()
            if key in ("x", "linear"):
                form = x_norm
            elif key in ("x*(1-x)", "x*(1-x_norm)", "x*(1 - x)", "x*(1 - x_norm)"):
                form = x_norm * (1.0 - x_norm)
            elif key in ("1-x", "1-x_norm", "1 - x", "1 - x_norm"):
                form = 1.0 - x_norm
            else:
                raise ValueError(f"Unknown diamag_form_profile: {profile}")
    form_arr = jnp.asarray(form, dtype=jnp.float64)
    return _broadcast_to_shape(form_arr, shape)


def _zero_normal_flux(
    vdx: jnp.ndarray, vdy: jnp.ndarray, grid, *, shape: tuple[int, ...]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if grid is None:
        return vdx, vdy
    if hasattr(grid, "perp"):
        nx = int(grid.perp.nx)
        ny = int(grid.perp.ny)
    else:
        nx = int(grid.nx)
        ny = int(grid.ny)
    if len(shape) == 3:
        vdx = vdx.at[:, 0, :].set(0.0)
        vdx = vdx.at[:, nx - 1, :].set(0.0)
        vdy = vdy.at[:, :, 0].set(0.0)
        vdy = vdy.at[:, :, ny - 1].set(0.0)
    else:
        vdx = vdx.at[0, :].set(0.0)
        vdx = vdx.at[nx - 1, :].set(0.0)
        vdy = vdy.at[:, 0].set(0.0)
        vdy = vdy.at[:, ny - 1].set(0.0)
    return vdx, vdy


def _diamag_flux(
    ctx: TermContext,
    *,
    f: jnp.ndarray,
    T: jnp.ndarray,
    q: float,
    bc,
    curv_x: jnp.ndarray,
    curv_y: jnp.ndarray,
    diamag_form: jnp.ndarray,
    bndry_flux: bool,
) -> jnp.ndarray:
    vdx = (T / q) * curv_x
    vdy = (T / q) * curv_y
    if not bndry_flux:
        vdx, vdy = _zero_normal_flux(vdx, vdy, getattr(ctx.geom, "grid", None), shape=f.shape)
    div_form = ddx(ctx.params, ctx.geom, f * vdx, bc) + ddy(ctx.params, ctx.geom, f * vdy, bc)
    grad_form = curv_x * ddx(ctx.params, ctx.geom, f * T / q, bc) + curv_y * ddy(
        ctx.params, ctx.geom, f * T / q, bc
    )
    return diamag_form * div_form + (1.0 - diamag_form) * grad_form


def diamagnetic_terms(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    if not bool(getattr(ctx.params, "diamagnetic_on", False)):
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

    shape = y.n.shape
    curv_x, curv_y = _curv_components(ctx, shape)
    if jnp.all(curv_x == 0.0) and jnp.all(curv_y == 0.0):
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

    diamag_form = _diamag_form(ctx, shape)
    bndry_flux = bool(getattr(ctx.params, "diamag_bndry_flux", True))
    n_eff = jnp.maximum(ctx.n_phys, float(ctx.params.n0_min))

    q_e = -1.0
    q_i = 1.0

    with jax.named_scope("diamagnetic"):
        dn = jnp.zeros_like(ctx.n_phys)
        model = str(getattr(ctx.params, "diamag_density_model", "electron")).lower()
        if model == "electron":
            dn = -_diamag_flux(
                ctx,
                f=ctx.n_phys,
                T=ctx.Te_phys,
                q=q_e,
                bc=ctx.bcs.n,
                curv_x=curv_x,
                curv_y=curv_y,
                diamag_form=diamag_form,
                bndry_flux=bndry_flux,
            )
        elif model == "ion":
            dn = -_diamag_flux(
                ctx,
                f=ctx.n_phys,
                T=ctx.Ti if ctx.hot_on else ctx.Te_phys,
                q=q_i,
                bc=ctx.bcs.n,
                curv_x=curv_x,
                curv_y=curv_y,
                diamag_form=diamag_form,
                bndry_flux=bndry_flux,
            )

        pe = ctx.n_phys * ctx.Te_phys
        dpe = -2.5 * _diamag_flux(
            ctx,
            f=pe,
            T=ctx.Te_phys,
            q=q_e,
            bc=ctx.bcs.Te,
            curv_x=curv_x,
            curv_y=curv_y,
            diamag_form=diamag_form,
            bndry_flux=bndry_flux,
        )
        dTe = (dpe - ctx.Te_phys * dn) / n_eff

        dvpar_e = jnp.zeros_like(y.vpar_e)
        dNv_e = -_diamag_flux(
            ctx,
            f=ctx.n_phys * y.vpar_e,
            T=ctx.Te_phys,
            q=q_e,
            bc=ctx.bcs.vpar_e,
            curv_x=curv_x,
            curv_y=curv_y,
            diamag_form=diamag_form,
            bndry_flux=bndry_flux,
        )
        dvpar_e = (dNv_e - y.vpar_e * dn) / n_eff

        dvpar_i = jnp.zeros_like(y.vpar_i)
        if ctx.hot_on:
            dNv_i = -_diamag_flux(
                ctx,
                f=ctx.n_phys * y.vpar_i,
                T=ctx.Ti,
                q=q_i,
                bc=ctx.bcs.vpar_i,
                curv_x=curv_x,
                curv_y=curv_y,
                diamag_form=diamag_form,
                bndry_flux=bndry_flux,
            )
            dvpar_i = (dNv_i - y.vpar_i * dn) / n_eff

        dTi = None
        if ctx.hot_on and y.Ti is not None:
            pi = ctx.n_phys * ctx.Ti
            dpi = -2.5 * _diamag_flux(
                ctx,
                f=pi,
                T=ctx.Ti,
                q=q_i,
                bc=ctx.bcs.Ti,
                curv_x=curv_x,
                curv_y=curv_y,
                diamag_form=diamag_form,
                bndry_flux=bndry_flux,
            )
            dTi = (dpi - ctx.Ti * dn) / n_eff

    return DRBSystemState(
        n=dn,
        omega=jnp.zeros_like(y.omega),
        vpar_e=dvpar_e,
        vpar_i=dvpar_i,
        Te=dTe,
        Ti=dTi if y.Ti is not None else None,
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )
