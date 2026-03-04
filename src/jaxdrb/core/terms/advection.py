from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState

from .context import TermContext


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
            adv_n = (
                -ctx.geom.exb_flux_divergence(
                    phi,
                    ctx.n_phys,
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
            use_cons = bool(ctx.params.exb_advect_conservative)
            n_eff = jnp.maximum(ctx.n_phys, float(ctx.params.n0_min))
            if use_cons:
                dNV_e = (
                    -ctx.geom.exb_flux_divergence(
                        phi,
                        ctx.n_phys * y.vpar_e,
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
                        ctx.n_phys * y.vpar_i,
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
                        ctx.n_phys * ctx.Te_phys,
                        bc_phi=bc.phi,
                        bc_adv=bc_fields[4],
                        positive=True,
                    )
                    * scale
                )
                adv_Te = (dP_e - ctx.Te_phys * adv_n) / n_eff
                if ctx.hot_on:
                    dP_i = (
                        -ctx.geom.exb_flux_divergence(
                            phi,
                            ctx.n_phys * ctx.Ti,
                            bc_phi=bc.phi,
                            bc_adv=bc_fields[5],
                            positive=True,
                        )
                        * scale
                    )
                    adv_Ti = (dP_i - ctx.Ti * adv_n) / n_eff
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
                        phi, ctx.Te_phys, bc_phi=bc.phi, bc_adv=bc_fields[4]
                    )
                    * scale
                )
                if ctx.hot_on:
                    adv_Ti = (
                        -ctx.geom.exb_flux_divergence(
                            phi, ctx.Ti, bc_phi=bc.phi, bc_adv=bc_fields[5]
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
