from __future__ import annotations

import equinox as eqx

from jaxdrb.bc import BC2D
from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams


class FieldBCs(eqx.Module):
    """Per-field 2D BCs resolved from params and geometry defaults."""

    n: BC2D
    omega: BC2D
    vpar_e: BC2D
    vpar_i: BC2D
    Te: BC2D
    Ti: BC2D
    psi: BC2D
    phi: BC2D


def _fallback_bc(params: DRBSystemParams, geom: GeometryAdapter) -> BC2D:
    grid = getattr(geom, "grid", None)
    if grid is not None and getattr(geom, "ndim", None) == 2:
        return grid.bc
    return params.perp_bc


def _bc_or(bc: BC2D | None, fallback: BC2D) -> BC2D:
    return bc if bc is not None else fallback


def resolve_bcs(params: DRBSystemParams, geom: GeometryAdapter) -> FieldBCs:
    fallback = _fallback_bc(params, geom)
    bc_n = _bc_or(params.bc_n, fallback)
    bc_omega = _bc_or(params.bc_omega, fallback)
    bc_vpar_e = _bc_or(params.bc_vpar_e, fallback)
    bc_vpar_i = _bc_or(params.bc_vpar_i, fallback)
    bc_Te = _bc_or(params.bc_Te, fallback)
    bc_Ti = _bc_or(params.bc_Ti, fallback)
    bc_psi = _bc_or(params.bc_psi, fallback)
    bc_phi = _bc_or(params.bc_phi, bc_omega)
    return FieldBCs(
        n=bc_n,
        omega=bc_omega,
        vpar_e=bc_vpar_e,
        vpar_i=bc_vpar_i,
        Te=bc_Te,
        Ti=bc_Ti,
        psi=bc_psi,
        phi=bc_phi,
    )


def is_periodic_bc(bc: BC2D) -> bool:
    return bc.kind_x == 0 and bc.kind_y == 0
