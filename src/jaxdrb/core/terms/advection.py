from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.state import DRBSystemState
from jaxdrb.legacy_hermes import full_omega_exb_advection as full_omega_exb_advection_mirror

from .context import TermContext
from .fields import _electron_pressure, _metric_div_coeff


def _pi_hat(ctx: TermContext) -> jnp.ndarray:
    if not bool(getattr(ctx.params, "diamagnetic_polarisation_on", False)):
        return jnp.zeros_like(ctx.n_phys)
    Abar = max(float(getattr(ctx.params, "average_atomic_mass", 1.0)), 1e-12)
    ion_pressure = ctx.n_phys * ctx.Ti
    electron_coeff = float(getattr(ctx.params, "me_hat", 0.0)) / Abar
    electron_pressure = _electron_pressure(ctx.params, ctx.n_phys, ctx.Te_phys)
    return ion_pressure - electron_coeff * electron_pressure


def _full_omega_exb_advection(
    ctx: TermContext,
    y: DRBSystemState,
    *,
    phi: jnp.ndarray,
    scale: jnp.ndarray | float,
) -> jnp.ndarray:
    bc = ctx.bcs
    Abar = float(getattr(ctx.params, "average_atomic_mass", 1.0))
    B = getattr(ctx.geom, "B", None)
    if B is None:
        invB2 = jnp.asarray(1.0, dtype=phi.dtype)
    else:
        invB2 = 1.0 / jnp.maximum(jnp.asarray(B, dtype=phi.dtype), 1e-12) ** 2
    invB2 = jnp.broadcast_to(invB2, phi.shape)
    pi_hat = _pi_hat(ctx)

    term = -ctx.geom.exb_flux_divergence(phi, 0.5 * y.omega, bc_phi=bc.phi, bc_adv=bc.omega)

    vE_dot_grad_pi = ctx.geom.bracket(phi, pi_hat, bc_phi=bc.phi, bc_f=bc.Te)
    coeff = 0.5 * Abar * invB2
    term = term - _metric_div_coeff(ctx.params, ctx.geom, vE_dot_grad_pi, coeff, bc.phi)

    bc_delp = bc.phi
    if bool(getattr(ctx.params, "poisson_invert_set", False)) and bc.phi.kind_x != 0:
        bc_delp = BC2D(
            kind_x=1,
            kind_y=bc.phi.kind_y,
            x_value=0.0,
            y_value=bc.phi.y_value,
            x_grad=0.0,
            y_grad=bc.phi.y_grad,
        )
    delp_phi = _metric_div_coeff(
        ctx.params,
        ctx.geom,
        phi,
        jnp.ones_like(phi),
        bc_delp,
    )
    delp_phi_2B2 = 0.5 * Abar * delp_phi * invB2
    term = term - ctx.geom.exb_flux_divergence(
        phi + pi_hat,
        delp_phi_2B2,
        bc_phi=bc.phi,
        bc_adv=bc.omega,
    )
    return term * scale


def exb_advection_terms(ctx: TermContext, y: DRBSystemState) -> DRBSystemState:
    """Return ExB advection contributions for each evolved field."""

    if not ctx.params.nonlinear_on:
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
    scale = ctx.nonlinear_scale
    phi = ctx.phi
    use_flux = str(ctx.params.exb_advection_form).lower() == "flux"
    use_flux = use_flux and getattr(ctx.geom, "jacobian", None) is not None
    use_hermes_mirror = str(getattr(ctx.params, "exb_flux_scheme", "centered")).lower() == (
        "hermes_mirror"
    )
    fields = [
        ctx.n_phys,
        y.omega,
        y.vpar_e,
        y.vpar_i,
        ctx.Te_phys,
    ]
    bc_fields = [bc.n, bc.omega, bc.vpar_e, bc.vpar_i, bc.Te]
    if ctx.hot_on:
        fields.append(ctx.Ti)
        bc_fields.append(bc.Ti)
    if y.psi is not None:
        fields.append(ctx.psi)
        bc_fields.append(bc.psi)

    if use_flux:
        with jax.named_scope("exb_flux_terms"):
            n_adv_field = ctx.n_prepared if use_hermes_mirror else ctx.n_phys
            pe_adv_field = ctx.pe_prepared if use_hermes_mirror else ctx.n_phys * ctx.Te_phys
            Te_adv_field = ctx.Te_prepared if use_hermes_mirror else ctx.Te_phys
            pi_adv_field = ctx.pi_prepared if use_hermes_mirror else ctx.n_phys * ctx.Ti
            Ti_adv_field = ctx.Ti_prepared if use_hermes_mirror else ctx.Ti

            adv_n = (
                -ctx.geom.exb_flux_divergence(
                    phi,
                    n_adv_field,
                    bc_phi=bc.phi,
                    bc_adv=bc_fields[0],
                    positive=True,
                )
                * scale
            )
            adv_w = (
                -ctx.geom.exb_flux_divergence(phi, y.omega, bc_phi=bc.phi, bc_adv=bc_fields[1])
                * scale
            )
            if not bool(getattr(ctx.params, "exb_advection_simplified", True)):
                if use_hermes_mirror:
                    adv_w = full_omega_exb_advection_mirror(ctx, y, phi=phi, scale=scale)
                else:
                    adv_w = _full_omega_exb_advection(ctx, y, phi=phi, scale=scale)
            use_cons = bool(ctx.params.exb_advect_conservative)
            n_eff = jnp.maximum(n_adv_field, float(ctx.params.n0_min))
            if use_cons:
                dNV_e = (
                    -ctx.geom.exb_flux_divergence(
                        phi,
                        n_adv_field * y.vpar_e,
                        bc_phi=bc.phi,
                        bc_adv=bc_fields[2],
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
                        bc_adv=bc_fields[3],
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
                        bc_adv=bc_fields[4],
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
                            bc_adv=bc_fields[5],
                            positive=True,
                        )
                        * scale
                    )
                    adv_Ti = (dP_i - Ti_adv_field * adv_n) / n_eff
                    idx = 6
                else:
                    adv_Ti = jnp.zeros_like(ctx.Ti)
                    idx = 5
            else:
                adv_ve = (
                    -ctx.geom.exb_flux_divergence(phi, y.vpar_e, bc_phi=bc.phi, bc_adv=bc_fields[2])
                    * scale
                )
                adv_vi = (
                    -ctx.geom.exb_flux_divergence(phi, y.vpar_i, bc_phi=bc.phi, bc_adv=bc_fields[3])
                    * scale
                )
                adv_Te = (
                    -ctx.geom.exb_flux_divergence(
                        phi, Te_adv_field, bc_phi=bc.phi, bc_adv=bc_fields[4]
                    )
                    * scale
                )
                if ctx.hot_on:
                    adv_Ti = (
                        -ctx.geom.exb_flux_divergence(
                            phi, Ti_adv_field, bc_phi=bc.phi, bc_adv=bc_fields[5]
                        )
                        * scale
                    )
                    idx = 6
                else:
                    adv_Ti = jnp.zeros_like(ctx.Ti)
                    idx = 5
            if y.psi is not None:
                adv_psi = (
                    -ctx.geom.exb_flux_divergence(
                        phi, ctx.psi, bc_phi=bc.phi, bc_adv=bc_fields[idx]
                    )
                    * scale
                )
            else:
                adv_psi = jnp.zeros_like(ctx.psi)
    else:
        with jax.named_scope("bracket_terms"):
            if hasattr(ctx.geom, "bracket_many"):
                brackets = ctx.geom.bracket_many(
                    phi, jnp.stack(fields), bc_phi=bc.phi, bc_f=bc_fields
                )
            else:
                brackets = jnp.stack(
                    [
                        ctx.geom.bracket(phi, f, bc_phi=bc.phi, bc_f=b)
                        for f, b in zip(fields, bc_fields)
                    ]
                )

        idx = 0
        adv_n = -brackets[idx] * scale
        idx += 1
        adv_w = -brackets[idx] * scale
        idx += 1
        adv_ve = -brackets[idx] * scale
        idx += 1
        adv_vi = -brackets[idx] * scale
        idx += 1
        adv_Te = -brackets[idx] * scale
        idx += 1
        if ctx.hot_on:
            adv_Ti = -brackets[idx] * scale
            idx += 1
        else:
            adv_Ti = jnp.zeros_like(ctx.Ti)
        if y.psi is not None:
            adv_psi = -brackets[idx] * scale
        else:
            adv_psi = jnp.zeros_like(ctx.psi)

    adv_N = None
    if ctx.neut_on and y.N is not None:
        with jax.named_scope("bracket_terms"):
            adv_N = -ctx.geom.bracket(phi, y.N, bc_phi=bc.phi, bc_f=bc.n)

    return DRBSystemState(
        n=adv_n,
        omega=adv_w,
        vpar_e=adv_ve,
        vpar_i=adv_vi,
        Te=adv_Te,
        Ti=adv_Ti if ctx.hot_on and y.Ti is not None else None,
        psi=adv_psi if y.psi is not None else None,
        N=adv_N if y.N is not None else None,
    )
