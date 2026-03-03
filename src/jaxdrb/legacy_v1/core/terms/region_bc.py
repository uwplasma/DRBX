from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.region_bcs import RegionBC
from jaxdrb.core.state import DRBSystemState, _state_zeros_like

from .context import TermContext
from .fields import log_rhs
from .ops import region_mask


def _collect_region_bcs(geom: GeometryAdapter) -> tuple[RegionBC, ...]:
    grid = getattr(geom, "grid", None)
    if grid is not None:
        bcs = getattr(grid, "region_bcs", None)
        if bcs:
            return tuple(bcs)
    bcs = getattr(geom, "region_bcs", None)
    if bcs:
        return tuple(bcs)
    return ()


def _apply_field_bc(
    geom: GeometryAdapter,
    field: jnp.ndarray,
    bc,
    mask: jnp.ndarray,
    *,
    use_dpar: bool,
) -> jnp.ndarray:
    if not bc.enabled():
        return jnp.zeros_like(field)
    kind = str(bc.kind).lower()
    nu = float(bc.nu)
    if kind in ("dirichlet", "value"):
        return -nu * mask * (field - float(bc.value))
    if kind in ("neumann", "gradient"):
        if not use_dpar:
            return jnp.zeros_like(field)
        grad = geom.dpar(field)
        return -nu * mask * (grad - float(bc.grad))
    return jnp.zeros_like(field)


def region_bc_relaxation(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    params = ctx.params
    geom = ctx.geom
    if not getattr(params, "region_bc_on", False):
        return _state_zeros_like(y)

    region_bcs = _collect_region_bcs(geom)
    if not region_bcs:
        return _state_zeros_like(y)

    out = _state_zeros_like(y)
    for region in region_bcs:
        mask = region_mask(geom, region.name, y.n.shape)
        if mask is None:
            continue
        mask = jnp.asarray(mask, dtype=jnp.float64)

        # n, Te use physical-space relaxation if log vars are enabled.
        n_term = _apply_field_bc(geom, ctx.n_phys, region.n, mask, use_dpar=True)
        Te_term = _apply_field_bc(geom, ctx.Te_phys, region.Te, mask, use_dpar=True)

        out = DRBSystemState(
            n=out.n + log_rhs(params, n_term, ctx.n_phys, ctx.n_floor, params.log_n),
            omega=out.omega + _apply_field_bc(geom, y.omega, region.omega, mask, use_dpar=True),
            vpar_e=out.vpar_e + _apply_field_bc(geom, y.vpar_e, region.vpar_e, mask, use_dpar=True),
            vpar_i=out.vpar_i + _apply_field_bc(geom, y.vpar_i, region.vpar_i, mask, use_dpar=True),
            Te=out.Te + log_rhs(params, Te_term, ctx.Te_phys, ctx.Te_floor, params.log_Te),
            Ti=(
                out.Ti
                if y.Ti is None
                else out.Ti + _apply_field_bc(geom, y.Ti, region.Ti, mask, use_dpar=True)
            ),
            psi=(
                out.psi
                if y.psi is None
                else out.psi + _apply_field_bc(geom, y.psi, region.psi, mask, use_dpar=True)
            ),
            N=out.N if y.N is None else out.N + jnp.zeros_like(y.N),
        )

    return out
