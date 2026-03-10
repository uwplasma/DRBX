from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.core.geometry import GeometryAdapter
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.terms.fields import phys_n, phys_Te, phi_from_omega
from jaxdrb.core.terms.sol import apply_sol_phi_bc, sol_masks
from jaxdrb.hermes_literal.bcs import FieldBCs, resolve_bcs
from jaxdrb.hermes_literal.ops import is_2d
from jaxdrb.hermes_literal.species import prepare_reduced_species_state_global


class TermContext(eqx.Module):
    """Shared derived quantities for literal term evaluation."""

    params: DRBSystemParams
    geom: GeometryAdapter
    bcs: FieldBCs
    n_phys: jnp.ndarray
    Te_phys: jnp.ndarray
    Ti: jnp.ndarray
    psi: jnp.ndarray
    phi: jnp.ndarray
    n_prepared: jnp.ndarray
    Te_prepared: jnp.ndarray
    Ti_prepared: jnp.ndarray
    pe_prepared: jnp.ndarray
    pi_prepared: jnp.ndarray
    n_floor: float
    Te_floor: float
    hot_on: bool = eqx.field(static=True)
    em_on: bool = eqx.field(static=True)
    neut_on: bool = eqx.field(static=True)
    is_2d: bool = eqx.field(static=True)
    mask_closed: jnp.ndarray | None = None
    mask_open: jnp.ndarray | None = None
    nonlinear_scale: jnp.ndarray | float = 1.0
    phi_iters: jnp.ndarray | None = None


def build_context(
    params: DRBSystemParams,
    geom: GeometryAdapter,
    y: DRBSystemState,
    *,
    phi_guess: jnp.ndarray | None = None,
    return_phi_iters: bool = False,
    skip_phi: bool = False,
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
    use_mirror_species_state = (
        str(getattr(params, "exb_flux_scheme", "centered")).lower() == "hermes_mirror"
        or str(getattr(params, "parallel_flux_scheme", "rusanov")).lower() == "hermes_mirror"
    )

    if use_mirror_species_state:
        prepared = prepare_reduced_species_state_global(
            n_phys,
            Te_phys,
            Ti if hot_on else None,
            density_floor=max(float(params.n0_min), n_floor),
        )
        n_prepared = prepared.density
        Te_prepared = prepared.electron_temperature
        Ti_prepared = prepared.ion_temperature if hot_on else jnp.zeros_like(Ti)
        pe_prepared = prepared.electron_pressure
        pi_prepared = prepared.ion_pressure if hot_on else jnp.zeros_like(Ti)
    else:
        n_prepared = n_phys
        Te_prepared = Te_phys
        Ti_prepared = Ti
        pe_prepared = n_phys * Te_phys
        pi_prepared = n_phys * Ti

    if skip_phi:
        phi = jnp.zeros_like(y.omega)
        phi_iters = None
    else:
        if return_phi_iters:
            phi, phi_iters = phi_from_omega(
                params,
                geom,
                y.omega,
                n_phys,
                bcs.phi,
                Ti=Ti,
                Te=Te_phys,
                phi_guess=phi_guess,
                return_iters=True,
            )
        else:
            phi = phi_from_omega(
                params,
                geom,
                y.omega,
                n_phys,
                bcs.phi,
                Ti=Ti,
                Te=Te_phys,
                phi_guess=phi_guess,
            )
            phi_iters = None
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
        n_prepared=n_prepared,
        Te_prepared=Te_prepared,
        Ti_prepared=Ti_prepared,
        pe_prepared=pe_prepared,
        pi_prepared=pi_prepared,
        phi_iters=phi_iters,
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
