"""Literal Hermes ExB advection assembly for the strict Stage 1 path."""

from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState
from jaxdrb.hermes_literal.vorticity import full_omega_exb_advection


def exb_advection_terms(ctx, y: DRBSystemState) -> DRBSystemState:
    """Return strict-path ExB advection contributions using literal staging."""

    if not bool(ctx.params.nonlinear_on):
        z = jnp.zeros_like(y.n)
        return DRBSystemState(
            n=z,
            omega=z,
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=jnp.zeros_like(y.Te),
            Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    bc = ctx.bcs
    phi = ctx.phi
    scale = ctx.nonlinear_scale
    n_adv_field = ctx.n_prepared
    pe_adv_field = ctx.pe_prepared
    Te_adv_field = ctx.Te_prepared
    pi_adv_field = ctx.pi_prepared
    Ti_adv_field = ctx.Ti_prepared

    adv_n = (
        -ctx.geom.exb_flux_divergence(
            phi,
            n_adv_field,
            bc_phi=bc.phi,
            bc_adv=bc.n,
            positive=True,
        )
        * scale
    )
    adv_w = (
        -ctx.geom.exb_flux_divergence(
            phi,
            y.omega,
            bc_phi=bc.phi,
            bc_adv=bc.omega,
        )
        * scale
    )
    if not bool(getattr(ctx.params, "exb_advection_simplified", True)):
        adv_w = full_omega_exb_advection(ctx, y, phi=phi, scale=scale)
    n_eff = jnp.maximum(n_adv_field, float(ctx.params.n0_min))

    if bool(getattr(ctx.params, "exb_advect_conservative", True)):
        dNV_e = (
            -ctx.geom.exb_flux_divergence(
                phi,
                n_adv_field * y.vpar_e,
                bc_phi=bc.phi,
                bc_adv=bc.vpar_e,
                positive=True,
            )
            * scale
        )
        adv_ve = (dNV_e - y.vpar_e * adv_n) / n_eff

        dNV_i = (
            -ctx.geom.exb_flux_divergence(
                phi,
                n_adv_field * y.vpar_i,
                bc_phi=bc.phi,
                bc_adv=bc.vpar_i,
                positive=True,
            )
            * scale
        )
        adv_vi = (dNV_i - y.vpar_i * adv_n) / n_eff

        dP_e = (
            -ctx.geom.exb_flux_divergence(
                phi,
                pe_adv_field,
                bc_phi=bc.phi,
                bc_adv=bc.Te,
                positive=True,
            )
            * scale
        )
        adv_Te = (dP_e - Te_adv_field * adv_n) / n_eff

        if ctx.hot_on:
            dP_i = (
                -ctx.geom.exb_flux_divergence(
                    phi,
                    pi_adv_field,
                    bc_phi=bc.phi,
                    bc_adv=bc.Ti,
                    positive=True,
                )
                * scale
            )
            adv_Ti = (dP_i - Ti_adv_field * adv_n) / n_eff
        else:
            adv_Ti = jnp.zeros_like(ctx.Ti)
    else:
        adv_ve = (
            -ctx.geom.exb_flux_divergence(
                phi,
                y.vpar_e,
                bc_phi=bc.phi,
                bc_adv=bc.vpar_e,
            )
            * scale
        )
        adv_vi = (
            -ctx.geom.exb_flux_divergence(
                phi,
                y.vpar_i,
                bc_phi=bc.phi,
                bc_adv=bc.vpar_i,
            )
            * scale
        )
        adv_Te = (
            -ctx.geom.exb_flux_divergence(
                phi,
                Te_adv_field,
                bc_phi=bc.phi,
                bc_adv=bc.Te,
            )
            * scale
        )
        if ctx.hot_on:
            adv_Ti = (
                -ctx.geom.exb_flux_divergence(
                    phi,
                    Ti_adv_field,
                    bc_phi=bc.phi,
                    bc_adv=bc.Ti,
                )
                * scale
            )
        else:
            adv_Ti = jnp.zeros_like(ctx.Ti)

    if y.psi is not None:
        adv_psi = (
            -ctx.geom.exb_flux_divergence(
                phi,
                ctx.psi,
                bc_phi=bc.phi,
                bc_adv=bc.psi,
            )
            * scale
        )
    else:
        adv_psi = None

    return DRBSystemState(
        n=adv_n,
        omega=adv_w,
        vpar_e=adv_ve,
        vpar_i=adv_vi,
        Te=adv_Te,
        Ti=adv_Ti if y.Ti is not None else None,
        psi=adv_psi if y.psi is not None else None,
        N=None if y.N is None else None,
    )
