from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.core.state import CoreSplit, CoreState
from jaxdrb.models.bcs import bc_relaxation_1d
from jaxdrb.models.braginskii import (
    chi_par_Te as chi_par_Te_eff,
    chi_par_Ti as chi_par_Ti_eff,
    eta_parallel as eta_parallel_eff,
    nu_par_e as nu_par_e_eff,
    nu_par_i as nu_par_i_eff,
)
from jaxdrb.models.params import DRBParams
from jaxdrb.models.sheath import (
    apply_loizu_mpse_boundary_conditions,
    apply_loizu2012_mpse_full_linear_bc,
    apply_loizu2012_mpse_full_linear_bc_hot_ion,
    sheath_bc_rate,
    sheath_energy_losses,
    sheath_loss_rate,
)


class LineEquilibrium(eqx.Module):
    """Background profiles along the field line used by the RHS."""

    n0: jnp.ndarray
    Te0: jnp.ndarray

    @classmethod
    def constant(
        cls,
        nl: int,
        *,
        n0: float = 1.0,
        Te0: float = 1.0,
        dtype=jnp.float64,
    ) -> "LineEquilibrium":
        return cls(
            n0=jnp.full((nl,), float(n0), dtype=dtype),
            Te0=jnp.full((nl,), float(Te0), dtype=dtype),
        )


def phi_from_omega(
    omega: jnp.ndarray,
    kperp2: jnp.ndarray,
    *,
    params: DRBParams,
    eq: LineEquilibrium,
    n: jnp.ndarray | None = None,
) -> jnp.ndarray:
    k2 = jnp.maximum(kperp2, float(getattr(params, "kperp2_min", 1e-6)))
    if bool(getattr(params, "boussinesq", True)):
        return -omega / k2
    if n is None:
        raise ValueError("Non-Boussinesq polarization requires perturbation density.")
    n0 = jnp.asarray(eq.n0)
    n0_min = float(getattr(params, "n0_min", 1e-6))
    if bool(getattr(params, "non_boussinesq_perturbed_density_on", False)):
        n_eff = n0 + jnp.real(jnp.asarray(n))
    else:
        n_eff = n0
    n_eff = jnp.maximum(n_eff, n0_min)
    return -omega / (k2 * n_eff)


class CoreLineModel(eqx.Module):
    """Unified 1D (field-line) DRB RHS for cold/hot/EM variants."""

    params: DRBParams
    hot_ion_on: bool = eqx.field(static=True, default=False)
    em_on: bool = eqx.field(static=True, default=False)

    def rhs_decomposed(
        self,
        t: float,
        y: CoreState,
        geom,
        *,
        kx: float,
        ky: float,
        eq: LineEquilibrium | None = None,
    ) -> CoreSplit:
        _ = t
        if eq is None:
            eq = LineEquilibrium.constant(int(y.n.size), n0=1.0, Te0=1.0)

        k2 = geom.kperp2(kx, ky)
        phi = phi_from_omega(y.omega, k2, params=self.params, eq=eq, n=y.n)
        dpar = geom.dpar
        C = geom.curvature

        def d2par(f: jnp.ndarray) -> jnp.ndarray:
            return dpar(dpar(f))

        drive_n = -1j * ky * float(getattr(self.params, "omega_n", 0.0)) * phi
        drive_Te = -1j * ky * float(getattr(self.params, "omega_Te", 0.0)) * phi
        drive_Ti = -1j * ky * float(getattr(self.params, "omega_Ti", 0.0)) * phi

        tau_i = float(getattr(self.params, "tau_i", 0.0)) if self.hot_ion_on else 0.0
        p_tot = (1.0 + tau_i) * y.n + y.Te + tau_i * y.Ti

        if bool(getattr(self.params, "curvature_on", True)):
            C_phi = C(kx, ky, phi)
            C_p = C(kx, ky, p_tot)
            C_T = (2.0 / 3.0) * C(kx, ky, (7.0 / 2.0) * y.Te + y.n - phi)
        else:
            C_phi = jnp.zeros_like(phi)
            C_p = jnp.zeros_like(phi)
            C_T = jnp.zeros_like(phi)

        lap_n = -k2 * y.n
        lap_omega = -k2 * y.omega
        lap_Te = -k2 * y.Te
        lap_Ti = -k2 * y.Ti
        lap_psi = -k2 * y.psi

        grad_par_phi_pe = dpar(phi - y.n - float(getattr(self.params, "alpha_Te_ohm", 1.71)) * y.Te)
        eta_eff = jnp.maximum(eta_parallel_eff(self.params, eq, Te_state=y.Te), 1e-12)

        use_algebraic_ohm = float(getattr(self.params, "me_hat", 0.0)) == 0.0

        if self.em_on:
            jpar = k2 * y.psi
            vpar_e = y.vpar_i - jpar
            coef = 0.5 * float(getattr(self.params, "beta", 0.0)) + float(
                getattr(self.params, "me_hat", 0.0)
            ) * jnp.maximum(k2, float(getattr(self.params, "kperp2_min", 1e-6)))
            coef = jnp.maximum(coef, 1e-12)
            conservative = CoreState.from_optional(
                n=-dpar(vpar_e),
                omega=dpar(jpar),
                vpar_e=jnp.zeros_like(y.vpar_e),
                vpar_i=-dpar(phi),
                Te=-(2.0 / 3.0) * dpar(vpar_e),
                Ti=jnp.zeros_like(y.Ti),
                psi=-grad_par_phi_pe / coef,
            )
            source = CoreState.from_optional(
                n=drive_n + (C_p - C_phi),
                omega=C_p,
                vpar_e=jnp.zeros_like(y.vpar_e),
                vpar_i=jnp.zeros_like(y.vpar_i),
                Te=drive_Te + C_T,
                Ti=jnp.zeros_like(y.Ti),
                psi=jnp.zeros_like(y.psi),
            )
            dissipative = CoreState.from_optional(
                n=float(getattr(self.params, "Dn", 0.0)) * lap_n
                - float(getattr(self.params, "nu_sink_n", 0.0)) * y.n,
                omega=float(getattr(self.params, "DOmega", 0.0)) * lap_omega,
                vpar_e=jnp.zeros_like(y.vpar_e),
                vpar_i=nu_par_i_eff(self.params, eq, Te_state=y.Te) * d2par(y.vpar_i)
                - float(getattr(self.params, "nu_sink_vpar", 0.0)) * y.vpar_i,
                Te=float(getattr(self.params, "DTe", 0.0)) * lap_Te
                + chi_par_Te_eff(self.params, eq, Te_state=y.Te) * d2par(y.Te)
                - float(getattr(self.params, "nu_sink_Te", 0.0)) * y.Te,
                Ti=jnp.zeros_like(y.Ti),
                psi=(-eta_eff * jpar + float(getattr(self.params, "Dpsi", 0.0)) * lap_psi) / coef,
            )
            dissipative = self._apply_sheath_terms_em(dissipative, geom, eq, k2, phi, vpar_e, y)
        else:
            vpar_e_eff = jnp.where(
                use_algebraic_ohm, y.vpar_i + grad_par_phi_pe / eta_eff, y.vpar_e
            )
            jpar = y.vpar_i - vpar_e_eff
            conservative = CoreState.from_optional(
                n=-dpar(vpar_e_eff),
                omega=dpar(jpar),
                vpar_e=jnp.where(
                    use_algebraic_ohm,
                    jnp.zeros_like(y.vpar_e),
                    grad_par_phi_pe
                    / jnp.maximum(float(getattr(self.params, "me_hat", 0.0)), 1e-12),
                ),
                vpar_i=-dpar(phi + tau_i * (y.n + y.Ti)),
                Te=-(2.0 / 3.0) * dpar(vpar_e_eff),
                Ti=-(2.0 / 3.0) * dpar(y.vpar_i) if self.hot_ion_on else jnp.zeros_like(y.Ti),
                psi=jnp.zeros_like(y.psi),
            )
            source = CoreState.from_optional(
                n=drive_n + (C_p - C_phi),
                omega=C_p,
                vpar_e=jnp.zeros_like(y.vpar_e),
                vpar_i=jnp.zeros_like(y.vpar_i),
                Te=drive_Te + C_T,
                Ti=drive_Ti if self.hot_ion_on else jnp.zeros_like(y.Ti),
                psi=jnp.zeros_like(y.psi),
            )
            if use_algebraic_ohm:
                dvpar_e_eta = -eta_eff * (y.vpar_e - vpar_e_eff)
            else:
                dvpar_e_eta = -(eta_eff * (y.vpar_e - y.vpar_i)) / jnp.maximum(
                    float(getattr(self.params, "me_hat", 0.0)), 1e-12
                )
            DTi = float(getattr(self.params, "DTi", getattr(self.params, "DTe", 0.0)))
            dissipative = CoreState.from_optional(
                n=float(getattr(self.params, "Dn", 0.0)) * lap_n
                - float(getattr(self.params, "nu_sink_n", 0.0)) * y.n,
                omega=float(getattr(self.params, "DOmega", 0.0)) * lap_omega,
                vpar_e=dvpar_e_eta
                + nu_par_e_eff(self.params, eq, Te_state=y.Te) * d2par(y.vpar_e)
                - float(getattr(self.params, "nu_sink_vpar", 0.0)) * y.vpar_e,
                vpar_i=nu_par_i_eff(self.params, eq, Te_state=y.Te, Ti_state=y.Ti) * d2par(y.vpar_i)
                - float(getattr(self.params, "nu_sink_vpar", 0.0)) * y.vpar_i,
                Te=float(getattr(self.params, "DTe", 0.0)) * lap_Te
                + chi_par_Te_eff(self.params, eq, Te_state=y.Te) * d2par(y.Te)
                - float(getattr(self.params, "nu_sink_Te", 0.0)) * y.Te,
                Ti=DTi * lap_Ti
                + chi_par_Ti_eff(self.params, eq, Te_state=y.Te, Ti_state=y.Ti) * d2par(y.Ti)
                if self.hot_ion_on
                else jnp.zeros_like(y.Ti),
                psi=jnp.zeros_like(y.psi),
            )
            dissipative = self._apply_sheath_terms_es(dissipative, geom, eq, k2, phi, vpar_e_eff, y)

        if not self.em_on:
            dissipative = self._apply_line_bcs_es(dissipative, geom, y)
        else:
            dissipative = self._apply_line_bcs_em(dissipative, geom, y)

        return CoreSplit(conservative=conservative, source=source, dissipative=dissipative)

    def rhs(
        self,
        t: float,
        y: CoreState,
        geom,
        *,
        kx: float,
        ky: float,
        eq: LineEquilibrium | None = None,
    ) -> CoreState:
        split = self.rhs_decomposed(t, y, geom, kx=kx, ky=ky, eq=eq)
        if not bool(getattr(self.params, "operator_split_on", False)):
            return split.total()
        out = CoreState.zeros_like(y)
        if bool(getattr(self.params, "operator_conservative_on", True)):
            out = out.add(split.conservative)
        if bool(getattr(self.params, "operator_source_on", True)):
            out = out.add(split.source)
        if bool(getattr(self.params, "operator_dissipative_on", True)):
            out = out.add(split.dissipative)
        return out

    def _apply_sheath_terms_es(
        self,
        dissipative: CoreState,
        geom,
        eq: LineEquilibrium,
        k2: jnp.ndarray,
        phi: jnp.ndarray,
        vpar_e_eff: jnp.ndarray,
        y: CoreState,
    ) -> CoreState:
        if int(getattr(self.params, "sheath_bc_model", 0)) == 1:
            if self.hot_ion_on:
                dn_bc, domega_bc, dvpar_e_bc, dvpar_i_bc, dTe_bc, dTi_bc = (
                    apply_loizu2012_mpse_full_linear_bc_hot_ion(
                        params=self.params,
                        geom=geom,
                        eq=eq,
                        kperp2=k2,
                        phi=phi,
                        n=y.n,
                        omega=y.omega,
                        vpar_e=vpar_e_eff,
                        vpar_i=y.vpar_i,
                        Te=y.Te,
                        Ti=y.Ti,
                        dpar=geom.dpar,
                        d2par=lambda f: geom.dpar(geom.dpar(f)),
                    )
                )
                dissipative = dissipative.add(
                    CoreState.from_optional(
                        n=dn_bc,
                        omega=domega_bc,
                        vpar_e=dvpar_e_bc,
                        vpar_i=dvpar_i_bc,
                        Te=dTe_bc,
                        Ti=dTi_bc,
                    )
                )
            else:
                dn_bc, domega_bc, dvpar_e_bc, dvpar_i_bc, dTe_bc = (
                    apply_loizu2012_mpse_full_linear_bc(
                        params=self.params,
                        geom=geom,
                        eq=eq,
                        kperp2=k2,
                        phi=phi,
                        n=y.n,
                        omega=y.omega,
                        vpar_e=vpar_e_eff,
                        vpar_i=y.vpar_i,
                        Te=y.Te,
                        dpar=geom.dpar,
                        d2par=lambda f: geom.dpar(geom.dpar(f)),
                    )
                )
                dissipative = dissipative.add(
                    CoreState.from_optional(
                        n=dn_bc,
                        omega=domega_bc,
                        vpar_e=dvpar_e_bc,
                        vpar_i=dvpar_i_bc,
                        Te=dTe_bc,
                        Ti=jnp.zeros_like(y.Ti),
                    )
                )
        else:
            dvpar_e_sh, dvpar_i_sh = apply_loizu_mpse_boundary_conditions(
                params=self.params,
                geom=geom,
                eq=eq,
                phi=phi,
                vpar_e=vpar_e_eff,
                vpar_i=y.vpar_i,
                Te=y.Te,
                Ti=y.Ti if self.hot_ion_on else None,
            )
            dissipative = dissipative.add(
                CoreState.from_optional(
                    n=jnp.zeros_like(y.n),
                    omega=jnp.zeros_like(y.omega),
                    vpar_e=dvpar_e_sh,
                    vpar_i=dvpar_i_sh,
                    Te=jnp.zeros_like(y.Te),
                    Ti=jnp.zeros_like(y.Ti),
                )
            )

        dTe_sh, dTi_sh = sheath_energy_losses(
            params=self.params, geom=geom, Te=y.Te, Ti=y.Ti if self.hot_ion_on else None
        )
        dissipative = dissipative.add(
            CoreState.from_optional(
                n=jnp.zeros_like(y.n),
                omega=jnp.zeros_like(y.omega),
                vpar_e=jnp.zeros_like(y.vpar_e),
                vpar_i=jnp.zeros_like(y.vpar_i),
                Te=dTe_sh,
                Ti=jnp.zeros_like(y.Ti) if dTi_sh is None else dTi_sh,
            )
        )

        if bool(getattr(self.params, "sheath_end_damp_on", False)):
            bc = sheath_bc_rate(self.params, geom)
            if bc is not None:
                nu_bc, mask = bc
                dissipative = dissipative.add(
                    CoreState.from_optional(
                        n=-nu_bc * mask * y.n,
                        omega=-nu_bc * mask * y.omega,
                        vpar_e=-nu_bc * mask * y.vpar_e,
                        vpar_i=-nu_bc * mask * y.vpar_i,
                        Te=-nu_bc * mask * y.Te,
                        Ti=-nu_bc * mask * y.Ti,
                    )
                )

        nu_loss = sheath_loss_rate(self.params, geom)
        dissipative = dissipative.add(
            CoreState.from_optional(
                n=-nu_loss * y.n,
                omega=-nu_loss * y.omega,
                vpar_e=-nu_loss * y.vpar_e,
                vpar_i=-nu_loss * y.vpar_i,
                Te=-nu_loss * y.Te,
                Ti=-nu_loss * y.Ti,
            )
        )
        return dissipative

    def _apply_sheath_terms_em(
        self,
        dissipative: CoreState,
        geom,
        eq: LineEquilibrium,
        k2: jnp.ndarray,
        phi: jnp.ndarray,
        vpar_e: jnp.ndarray,
        y: CoreState,
    ) -> CoreState:
        if int(getattr(self.params, "sheath_bc_model", 0)) == 1:
            dn_bc, domega_bc, dvpar_e_bc, dvpar_i_bc, dTe_bc = apply_loizu2012_mpse_full_linear_bc(
                params=self.params,
                geom=geom,
                eq=eq,
                kperp2=k2,
                phi=phi,
                n=y.n,
                omega=y.omega,
                vpar_e=vpar_e,
                vpar_i=y.vpar_i,
                Te=y.Te,
                dpar=geom.dpar,
                d2par=lambda f: geom.dpar(geom.dpar(f)),
            )
            dissipative = dissipative.add(
                CoreState.from_optional(
                    n=dn_bc,
                    omega=domega_bc,
                    vpar_e=jnp.zeros_like(y.vpar_e),
                    vpar_i=dvpar_i_bc,
                    Te=dTe_bc,
                    psi=jnp.zeros_like(y.psi),
                )
            )
            djpar = dvpar_i_bc - dvpar_e_bc
        else:
            dvpar_e_sh, dvpar_i_sh = apply_loizu_mpse_boundary_conditions(
                params=self.params,
                geom=geom,
                eq=eq,
                phi=phi,
                vpar_e=vpar_e,
                vpar_i=y.vpar_i,
                Te=y.Te,
            )
            dissipative = dissipative.add(
                CoreState.from_optional(
                    n=jnp.zeros_like(y.n),
                    omega=jnp.zeros_like(y.omega),
                    vpar_e=jnp.zeros_like(y.vpar_e),
                    vpar_i=dvpar_i_sh,
                    Te=jnp.zeros_like(y.Te),
                    psi=jnp.zeros_like(y.psi),
                )
            )
            djpar = dvpar_i_sh - dvpar_e_sh

        k2_safe = jnp.maximum(k2, float(getattr(self.params, "kperp2_min", 1e-6)))
        dissipative = dissipative.add(
            CoreState.from_optional(
                n=jnp.zeros_like(y.n),
                omega=jnp.zeros_like(y.omega),
                vpar_e=jnp.zeros_like(y.vpar_e),
                vpar_i=jnp.zeros_like(y.vpar_i),
                Te=jnp.zeros_like(y.Te),
                psi=djpar / k2_safe,
            )
        )

        dTe_sh, _ = sheath_energy_losses(params=self.params, geom=geom, Te=y.Te)
        dissipative = dissipative.add(
            CoreState.from_optional(
                n=jnp.zeros_like(y.n),
                omega=jnp.zeros_like(y.omega),
                vpar_e=jnp.zeros_like(y.vpar_e),
                vpar_i=jnp.zeros_like(y.vpar_i),
                Te=dTe_sh,
                psi=jnp.zeros_like(y.psi),
            )
        )

        if bool(getattr(self.params, "sheath_end_damp_on", False)):
            bc = sheath_bc_rate(self.params, geom)
            if bc is not None:
                nu_bc, mask = bc
                dissipative = dissipative.add(
                    CoreState.from_optional(
                        n=-nu_bc * mask * y.n,
                        omega=-nu_bc * mask * y.omega,
                        vpar_e=jnp.zeros_like(y.vpar_e),
                        vpar_i=jnp.zeros_like(y.vpar_i),
                        Te=-nu_bc * mask * y.Te,
                        psi=-nu_bc * mask * y.psi,
                    )
                )

        nu_loss = sheath_loss_rate(self.params, geom)
        dissipative = dissipative.add(
            CoreState.from_optional(
                n=-nu_loss * y.n,
                omega=-nu_loss * y.omega,
                vpar_e=jnp.zeros_like(y.vpar_e),
                vpar_i=-nu_loss * y.vpar_i,
                Te=-nu_loss * y.Te,
                psi=-nu_loss * y.psi,
            )
        )
        return dissipative

    def _apply_line_bcs_es(self, dissipative: CoreState, geom, y: CoreState) -> CoreState:
        line_bcs = getattr(self.params, "line_bcs", None)
        if line_bcs is None or not line_bcs.enabled:
            return dissipative
        dl = float(getattr(geom, "dl", 1.0))
        return dissipative.add(
            CoreState.from_optional(
                n=bc_relaxation_1d(y.n, bc=line_bcs.n, dl=dl),
                omega=bc_relaxation_1d(y.omega, bc=line_bcs.omega, dl=dl),
                vpar_e=bc_relaxation_1d(y.vpar_e, bc=line_bcs.vpar_e, dl=dl),
                vpar_i=bc_relaxation_1d(y.vpar_i, bc=line_bcs.vpar_i, dl=dl),
                Te=bc_relaxation_1d(y.Te, bc=line_bcs.Te, dl=dl),
                Ti=bc_relaxation_1d(y.Ti, bc=line_bcs.Ti, dl=dl)
                if self.hot_ion_on
                else jnp.zeros_like(y.Ti),
            )
        )

    def _apply_line_bcs_em(self, dissipative: CoreState, geom, y: CoreState) -> CoreState:
        line_bcs = getattr(self.params, "line_bcs", None)
        if line_bcs is None or not line_bcs.enabled:
            return dissipative
        dl = float(getattr(geom, "dl", 1.0))
        return dissipative.add(
            CoreState.from_optional(
                n=bc_relaxation_1d(y.n, bc=line_bcs.n, dl=dl),
                omega=bc_relaxation_1d(y.omega, bc=line_bcs.omega, dl=dl),
                vpar_e=jnp.zeros_like(y.vpar_e),
                vpar_i=bc_relaxation_1d(y.vpar_i, bc=line_bcs.vpar_i, dl=dl),
                Te=bc_relaxation_1d(y.Te, bc=line_bcs.Te, dl=dl),
                psi=bc_relaxation_1d(y.psi, bc=line_bcs.psi, dl=dl),
            )
        )
