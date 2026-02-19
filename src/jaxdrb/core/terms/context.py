from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.state import DRBSystemState

from .bcs import FieldBCs, resolve_bcs
from .fields import phys_n, phys_Te, phi_from_omega
from .ops import is_2d
from .sol import apply_sol_phi_bc, sol_masks


class TermContext(eqx.Module):
    """Shared derived quantities for term evaluation."""

    params: DRBSystemParams
    geom: GeometryAdapter
    bcs: FieldBCs
    n_phys: jnp.ndarray
    Te_phys: jnp.ndarray
    Ti: jnp.ndarray
    psi: jnp.ndarray
    phi: jnp.ndarray
    n_floor: float
    Te_floor: float
    hot_on: bool = eqx.field(static=True)
    em_on: bool = eqx.field(static=True)
    neut_on: bool = eqx.field(static=True)
    is_2d: bool = eqx.field(static=True)
    mask_closed: jnp.ndarray | None = None
    mask_open: jnp.ndarray | None = None
    nonlinear_scale: jnp.ndarray | float = 1.0


def build_context(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    y: DRBSystemState,
    *,
    phi_guess: jnp.ndarray | None = None,
) -> TermContext:
    hot_on = bool(params.hot_ion_on) and (y.Ti is not None)
    em_on = bool(params.em_on) and (y.psi is not None)
    neut_on = bool(params.neutrals_on) and (y.N is not None) and params.neutrals.enabled

    Ti = y.Ti if hot_on and y.Ti is not None else jnp.zeros_like(y.Te)
    psi = y.psi if em_on and y.psi is not None else jnp.zeros_like(y.Te)

    bcs = resolve_bcs(params, geom)

    n_phys = phys_n(params, y.n)
    Te_phys = phys_Te(params, y.Te)
    n_floor = float(params.sol_n_floor)
    Te_floor = float(params.sol_Te_floor)

    phi = phi_from_omega(params, geom, y.omega, n_phys, bcs.phi, phi_guess=phi_guess)
    if params.sol_on and params.sol_phi_bc_on:
        phi = apply_sol_phi_bc(params, geom, phi, Te_phys, bcs.phi)

    mask_closed = None
    mask_open = None
    nonlinear_scale = 1.0
    if params.sol_on:
        mask_closed, mask_open, nonlinear_scale = sol_masks(params, geom)

    return TermContext(
        params=params,
        geom=geom,
        bcs=bcs,
        n_phys=n_phys,
        Te_phys=Te_phys,
        Ti=Ti,
        psi=psi,
        phi=phi,
        n_floor=n_floor,
        Te_floor=Te_floor,
        hot_on=hot_on,
        em_on=em_on,
        neut_on=neut_on,
        is_2d=is_2d(geom),
        mask_closed=mask_closed,
        mask_open=mask_open,
        nonlinear_scale=nonlinear_scale,
    )
