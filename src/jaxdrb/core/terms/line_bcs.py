from __future__ import annotations

from jaxdrb.core.bcs import bc_relaxation_1d
from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.state import DRBSystemState, _state_zeros_like


def line_bc_terms(params: DRBSystemParams, geom: GeometryAdapter, y: DRBSystemState) -> DRBSystemState:
    if not params.line_bcs.enabled:
        return _state_zeros_like(y)

    dl = float(getattr(geom, "geom", geom).dl) if hasattr(geom, "geom") else None
    if dl is None:
        return _state_zeros_like(y)

    bc = params.line_bcs
    return DRBSystemState(
        n=bc_relaxation_1d(y.n, bc=bc.n, dl=dl),
        omega=bc_relaxation_1d(y.omega, bc=bc.omega, dl=dl),
        vpar_e=bc_relaxation_1d(y.vpar_e, bc=bc.vpar_e, dl=dl),
        vpar_i=bc_relaxation_1d(y.vpar_i, bc=bc.vpar_i, dl=dl),
        Te=bc_relaxation_1d(y.Te, bc=bc.Te, dl=dl),
        Ti=None if y.Ti is None else bc_relaxation_1d(y.Ti, bc=bc.Ti, dl=dl),
        psi=None if y.psi is None else bc_relaxation_1d(y.psi, bc=bc.psi, dl=dl),
        N=None if y.N is None else bc_relaxation_1d(y.N, bc=bc.n, dl=dl),
    )
