from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.operators.brackets import poisson_bracket_arakawa, poisson_bracket_centered

from .fd import ddx as ddx_fd
from .fd import ddy as ddy_fd
from .fd import enforce_bc_relaxation, inv_laplacian_cg, laplacian as laplacian_fd
from .grid import Grid2D
from .spectral import dealias, ddy, inv_laplacian, laplacian, poisson_bracket_spectral


class DRB2DHotIonParams(eqx.Module):
    """Hot-ion extension of the conservative 2D nonlinear DRB testbed."""

    # Background-gradient drives.
    omega_n: float = 0.0
    omega_Te: float = 0.0
    omega_Ti: float = 0.0

    # Parallel coupling modeled via constant k_par (optional).
    kpar: float = 0.0
    eta: float = 0.0
    me_hat: float = 0.2

    # Hot-ion parameters.
    tau_i: float = 1.0
    alpha_Te_ohm: float = 1.71
    alpha_Ti: float = 1.0

    # Curvature drive (simple slab interchange model).
    curvature_on: bool = False
    curvature_coeff: float = 0.0

    # Polarization closure (Boussinesq vs non-Boussinesq).
    boussinesq: bool = True
    n0: float = 1.0
    n0_min: float = 1e-6
    non_boussinesq_perturbed_density_on: bool = False

    # Dissipation.
    Dn: float = 0.0
    DOmega: float = 0.0
    DTe: float = 0.0
    DTi: float = 0.0

    # Numerical options.
    bracket: Literal["spectral", "arakawa", "centered"] = "arakawa"
    poisson: Literal["spectral", "cg_fd"] = "spectral"
    dealias_on: bool = True
    k2_min: float = 1e-12
    bc_enforce_nu: float = 0.0

    # Operator split toggles.
    operator_split_on: bool = False
    operator_conservative_on: bool = True
    operator_source_on: bool = True
    operator_dissipative_on: bool = True


class DRB2DHotIonState(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray
    vpar_e: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray
    Ti: jnp.ndarray


class DRB2DHotIonDecomposition(eqx.Module):
    conservative: DRB2DHotIonState
    source: DRB2DHotIonState
    dissipative: DRB2DHotIonState

    def total(self) -> DRB2DHotIonState:
        return _state_add(_state_add(self.conservative, self.source), self.dissipative)


def _state_add(a: DRB2DHotIonState, b: DRB2DHotIonState) -> DRB2DHotIonState:
    return DRB2DHotIonState(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        vpar_e=a.vpar_e + b.vpar_e,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
        Ti=a.Ti + b.Ti,
    )


def _state_zeros_like(y: DRB2DHotIonState) -> DRB2DHotIonState:
    z = jnp.zeros_like(y.n)
    return DRB2DHotIonState(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z, Ti=z)


class DRB2DHotIonModel(eqx.Module):
    params: DRB2DHotIonParams
    grid: Grid2D

    def phi_from_omega(self, omega: jnp.ndarray, n: jnp.ndarray | None = None) -> jnp.ndarray:
        if self.params.poisson == "spectral":
            if self.grid.bc.kind_x != 0 or self.grid.bc.kind_y != 0:
                raise ValueError("Spectral Poisson solve requires periodic BCs in x and y.")
            if self.params.boussinesq:
                return inv_laplacian(omega, self.grid.k2, k2_min=self.params.k2_min)
            if n is None:
                raise ValueError("Non-Boussinesq polarization requires density n.")
            n_eff = float(self.params.n0)
            if self.params.non_boussinesq_perturbed_density_on:
                n_eff = n_eff + jnp.real(jnp.asarray(n))
            n_eff = jnp.maximum(jnp.asarray(n_eff), float(self.params.n0_min))
            k2 = jnp.maximum(self.grid.k2, self.params.k2_min)
            return -omega / (k2 * n_eff)
        if not self.params.boussinesq:
            raise ValueError("Non-Boussinesq polarization currently requires spectral Poisson.")
        return inv_laplacian_cg(
            omega, dx=self.grid.dx, dy=self.grid.dy, bc=self.grid.bc, maxiter=300
        )

    def _bracket(self, phi: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        if self.params.bracket == "spectral":
            if self.grid.bc.kind_x != 0 or self.grid.bc.kind_y != 0:
                raise ValueError("Spectral bracket requires periodic BCs in x and y.")
            return poisson_bracket_spectral(
                phi,
                f,
                kx=self.grid.kx,
                ky=self.grid.ky,
                dealias_mask=self.grid.dealias_mask if self.params.dealias_on else None,
            )
        if self.params.bracket == "arakawa":
            if self.grid.bc.kind_x != 0 or self.grid.bc.kind_y != 0:
                raise ValueError("Arakawa bracket implementation currently assumes periodic BCs.")
            return poisson_bracket_arakawa(phi, f, self.grid.dx, self.grid.dy)
        if self.grid.bc.kind_x == 0 and self.grid.bc.kind_y == 0:
            j = poisson_bracket_centered(phi, f, self.grid.dx, self.grid.dy)
        else:
            dphi_dx = ddx_fd(phi, self.grid.dx, self.grid.bc)
            dphi_dy = ddy_fd(phi, self.grid.dy, self.grid.bc)
            df_dx = ddx_fd(f, self.grid.dx, self.grid.bc)
            df_dy = ddy_fd(f, self.grid.dy, self.grid.bc)
            j = dphi_dx * df_dy - dphi_dy * df_dx
        if self.params.dealias_on and self.grid.bc.kind_x == 0 and self.grid.bc.kind_y == 0:
            return dealias(j, self.grid.dealias_mask)
        return j

    def _dpar(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.params.kpar == 0.0:
            return jnp.zeros_like(f)
        return 1j * float(self.params.kpar) * f

    def _curvature(self, f: jnp.ndarray) -> jnp.ndarray:
        if not self.params.curvature_on or self.params.curvature_coeff == 0.0:
            return jnp.zeros_like(f)
        if (
            self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
            and self.params.poisson == "spectral"
        ):
            df_dy = ddy(f, self.grid.ky)
        else:
            df_dy = ddy_fd(f, self.grid.dy, self.grid.bc)
        return -float(self.params.curvature_coeff) * df_dy

    def rhs_decomposed(self, t: float, y: DRB2DHotIonState) -> DRB2DHotIonDecomposition:
        _ = t
        n = y.n
        omega = y.omega
        vpar_e = y.vpar_e
        vpar_i = y.vpar_i
        Te = y.Te
        Ti = y.Ti

        phi = self.phi_from_omega(omega, n=n)

        adv_n = -self._bracket(phi, n)
        adv_w = -self._bracket(phi, omega)
        adv_ve = -self._bracket(phi, vpar_e)
        adv_vi = -self._bracket(phi, vpar_i)
        adv_Te = -self._bracket(phi, Te)
        adv_Ti = -self._bracket(phi, Ti)

        grad_par_phi_pe = self._dpar(phi - n - float(self.params.alpha_Te_ohm) * Te)
        jpar = vpar_i - vpar_e
        tau_i = float(self.params.tau_i)

        p_tot = (1.0 + tau_i) * n + Te + tau_i * Ti
        C_phi = self._curvature(phi)
        C_p = self._curvature(p_tot)
        C_Te = (2.0 / 3.0) * self._curvature((7.0 / 2.0) * Te + n - phi)

        conservative = DRB2DHotIonState(
            n=adv_n - self._dpar(vpar_e),
            omega=adv_w + self._dpar(jpar),
            vpar_e=adv_ve + grad_par_phi_pe / jnp.maximum(float(self.params.me_hat), 1e-12),
            vpar_i=adv_vi - self._dpar(phi + tau_i * (n + Ti)),
            Te=adv_Te - (2.0 / 3.0) * self._dpar(vpar_e),
            Ti=adv_Ti - (2.0 / 3.0) * self._dpar(vpar_i),
        )

        drive_n = 0.0
        drive_Te = 0.0
        drive_Ti = 0.0
        if self.params.omega_n != 0.0 or self.params.omega_Te != 0.0 or self.params.omega_Ti != 0.0:
            if self.grid.bc.kind_y != 0:
                raise ValueError("Drive terms assume periodic y for spectral ky representation.")
            dphi_dy = ddy_fd(phi, self.grid.dy, self.grid.bc)
            drive_n = -float(self.params.omega_n) * dphi_dy
            drive_Te = -float(self.params.omega_Te) * dphi_dy
            drive_Ti = -float(self.params.omega_Ti) * dphi_dy

        source = DRB2DHotIonState(
            n=drive_n + (C_p - C_phi),
            omega=C_p,
            vpar_e=jnp.zeros_like(vpar_e),
            vpar_i=jnp.zeros_like(vpar_i),
            Te=drive_Te + C_Te,
            Ti=drive_Ti,
        )

        if (
            self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
            and self.params.poisson == "spectral"
        ):
            lap_n = laplacian(n, self.grid.k2)
            lap_w = laplacian(omega, self.grid.k2)
            lap_Te = laplacian(Te, self.grid.k2)
            lap_Ti = laplacian(Ti, self.grid.k2)
        else:
            lap_n = laplacian_fd(n, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_w = laplacian_fd(omega, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_Te = laplacian_fd(Te, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_Ti = laplacian_fd(Ti, self.grid.dx, self.grid.dy, self.grid.bc)

        dissipative = DRB2DHotIonState(
            n=float(self.params.Dn) * lap_n,
            omega=float(self.params.DOmega) * lap_w,
            vpar_e=-(float(self.params.eta) / jnp.maximum(float(self.params.me_hat), 1e-12))
            * (vpar_e - vpar_i),
            vpar_i=jnp.zeros_like(vpar_i),
            Te=float(self.params.DTe) * lap_Te,
            Ti=float(self.params.DTi) * lap_Ti,
        )

        if self.params.bc_enforce_nu != 0.0:
            dissipative = _state_add(
                dissipative,
                DRB2DHotIonState(
                    n=enforce_bc_relaxation(
                        n,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=self.grid.bc,
                        nu=self.params.bc_enforce_nu,
                    ),
                    omega=enforce_bc_relaxation(
                        omega,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=self.grid.bc,
                        nu=self.params.bc_enforce_nu,
                    ),
                    vpar_e=enforce_bc_relaxation(
                        vpar_e,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=self.grid.bc,
                        nu=self.params.bc_enforce_nu,
                    ),
                    vpar_i=enforce_bc_relaxation(
                        vpar_i,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=self.grid.bc,
                        nu=self.params.bc_enforce_nu,
                    ),
                    Te=enforce_bc_relaxation(
                        Te,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=self.grid.bc,
                        nu=self.params.bc_enforce_nu,
                    ),
                    Ti=enforce_bc_relaxation(
                        Ti,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=self.grid.bc,
                        nu=self.params.bc_enforce_nu,
                    ),
                ),
            )

        return DRB2DHotIonDecomposition(
            conservative=conservative, source=source, dissipative=dissipative
        )

    def rhs(self, t: float, y: DRB2DHotIonState) -> DRB2DHotIonState:
        split = self.rhs_decomposed(t, y)
        if not self.params.operator_split_on:
            return split.total()

        out = _state_zeros_like(y)
        if self.params.operator_conservative_on:
            out = _state_add(out, split.conservative)
        if self.params.operator_source_on:
            out = _state_add(out, split.source)
        if self.params.operator_dissipative_on:
            out = _state_add(out, split.dissipative)
        return out

    def energy(self, y: DRB2DHotIonState) -> jnp.ndarray:
        phi = self.phi_from_omega(y.omega, n=y.n)
        c_Te = 1.5 * float(self.params.alpha_Te_ohm)
        c_Ti = 1.5 * float(self.params.alpha_Ti)
        if not self.params.boussinesq:
            n_eff = float(self.params.n0)
            if self.params.non_boussinesq_perturbed_density_on:
                n_eff = n_eff + jnp.real(jnp.asarray(y.n))
            n_eff = jnp.maximum(jnp.asarray(n_eff), float(self.params.n0_min))
            phi_term = jnp.real(jnp.conj(phi) * phi) * self.grid.k2 * n_eff
        else:
            phi_term = jnp.real(jnp.conj(phi) * phi) * self.grid.k2
        return 0.5 * jnp.mean(
            jnp.real(jnp.conj(y.n) * y.n)
            + phi_term
            + float(self.params.me_hat) * jnp.real(jnp.conj(y.vpar_e) * y.vpar_e)
            + jnp.real(jnp.conj(y.vpar_i) * y.vpar_i)
            + c_Te * jnp.real(jnp.conj(y.Te) * y.Te)
            + c_Ti * jnp.real(jnp.conj(y.Ti) * y.Ti)
        )

    def energy_rate(self, y: DRB2DHotIonState, dy: DRB2DHotIonState) -> jnp.ndarray:
        if self.params.boussinesq:
            phi = self.phi_from_omega(y.omega, n=y.n)
            c_Te = 1.5 * float(self.params.alpha_Te_ohm)
            c_Ti = 1.5 * float(self.params.alpha_Ti)
            return jnp.mean(
                jnp.real(jnp.conj(y.n) * dy.n)
                - jnp.real(jnp.conj(phi) * dy.omega)
                + float(self.params.me_hat) * jnp.real(jnp.conj(y.vpar_e) * dy.vpar_e)
                + jnp.real(jnp.conj(y.vpar_i) * dy.vpar_i)
                + c_Te * jnp.real(jnp.conj(y.Te) * dy.Te)
                + c_Ti * jnp.real(jnp.conj(y.Ti) * dy.Ti)
            )

        eps = jnp.asarray(1.0e-7, dtype=jnp.float64)
        y_plus = DRB2DHotIonState(
            n=y.n + eps * dy.n,
            omega=y.omega + eps * dy.omega,
            vpar_e=y.vpar_e + eps * dy.vpar_e,
            vpar_i=y.vpar_i + eps * dy.vpar_i,
            Te=y.Te + eps * dy.Te,
            Ti=y.Ti + eps * dy.Ti,
        )
        y_minus = DRB2DHotIonState(
            n=y.n - eps * dy.n,
            omega=y.omega - eps * dy.omega,
            vpar_e=y.vpar_e - eps * dy.vpar_e,
            vpar_i=y.vpar_i - eps * dy.vpar_i,
            Te=y.Te - eps * dy.Te,
            Ti=y.Ti - eps * dy.Ti,
        )
        E_plus = self.energy(y_plus)
        E_minus = self.energy(y_minus)
        return (E_plus - E_minus) / (2.0 * eps)

    def energy_budget(self, y: DRB2DHotIonState) -> dict[str, jnp.ndarray]:
        """Return a term-by-term energy budget for hot-ion DRB2D."""

        n = y.n
        omega = y.omega
        vpar_e = y.vpar_e
        vpar_i = y.vpar_i
        Te = y.Te
        Ti = y.Ti
        phi = self.phi_from_omega(omega, n=n)

        adv_n = -self._bracket(phi, n)
        adv_w = -self._bracket(phi, omega)
        adv_ve = -self._bracket(phi, vpar_e)
        adv_vi = -self._bracket(phi, vpar_i)
        adv_Te = -self._bracket(phi, Te)
        adv_Ti = -self._bracket(phi, Ti)

        grad_par_phi_pe = self._dpar(phi - n - float(self.params.alpha_Te_ohm) * Te)
        jpar = vpar_i - vpar_e
        tau_i = float(self.params.tau_i)
        par_n = -self._dpar(vpar_e)
        par_w = self._dpar(jpar)
        par_ve = grad_par_phi_pe / jnp.maximum(float(self.params.me_hat), 1e-12)
        par_vi = -self._dpar(phi + tau_i * (n + Ti))
        par_Te = -(2.0 / 3.0) * self._dpar(vpar_e)
        par_Ti = -(2.0 / 3.0) * self._dpar(vpar_i)

        p_tot = (1.0 + tau_i) * n + Te + tau_i * Ti
        C_phi = self._curvature(phi)
        C_p = self._curvature(p_tot)
        C_Te = (2.0 / 3.0) * self._curvature((7.0 / 2.0) * Te + n - phi)

        drive_n = 0.0
        drive_Te = 0.0
        drive_Ti = 0.0
        if self.params.omega_n != 0.0 or self.params.omega_Te != 0.0 or self.params.omega_Ti != 0.0:
            dphi_dy = ddy_fd(phi, self.grid.dy, self.grid.bc)
            drive_n = -float(self.params.omega_n) * dphi_dy
            drive_Te = -float(self.params.omega_Te) * dphi_dy
            drive_Ti = -float(self.params.omega_Ti) * dphi_dy

        if (
            self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
            and self.params.poisson == "spectral"
        ):
            lap_n = laplacian(n, self.grid.k2)
            lap_w = laplacian(omega, self.grid.k2)
            lap_Te = laplacian(Te, self.grid.k2)
            lap_Ti = laplacian(Ti, self.grid.k2)
        else:
            lap_n = laplacian_fd(n, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_w = laplacian_fd(omega, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_Te = laplacian_fd(Te, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_Ti = laplacian_fd(Ti, self.grid.dx, self.grid.dy, self.grid.bc)

        diss_n = float(self.params.Dn) * lap_n
        diss_w = float(self.params.DOmega) * lap_w
        diss_ve = -(float(self.params.eta) / jnp.maximum(float(self.params.me_hat), 1e-12)) * (
            vpar_e - vpar_i
        )
        diss_vi = jnp.zeros_like(vpar_i)
        diss_Te = float(self.params.DTe) * lap_Te
        diss_Ti = float(self.params.DTi) * lap_Ti

        c_Te = 1.5 * float(self.params.alpha_Te_ohm)
        c_Ti = 1.5 * float(self.params.alpha_Ti)

        def edot(n_term, w_term, ve_term, vi_term, Te_term, Ti_term):
            return jnp.mean(
                jnp.real(jnp.conj(n) * n_term)
                - jnp.real(jnp.conj(phi) * w_term)
                + float(self.params.me_hat) * jnp.real(jnp.conj(vpar_e) * ve_term)
                + jnp.real(jnp.conj(vpar_i) * vi_term)
                + c_Te * jnp.real(jnp.conj(Te) * Te_term)
                + c_Ti * jnp.real(jnp.conj(Ti) * Ti_term)
            )

        E_dot_adv = edot(adv_n, adv_w, adv_ve, adv_vi, adv_Te, adv_Ti)
        E_dot_parallel = edot(par_n, par_w, par_ve, par_vi, par_Te, par_Ti)
        E_dot_curvature = edot(C_p - C_phi, C_p, 0.0, 0.0, C_Te, 0.0)
        E_dot_drive = edot(drive_n, 0.0, 0.0, 0.0, drive_Te, drive_Ti)
        E_dot_diss = edot(diss_n, diss_w, diss_ve, diss_vi, diss_Te, diss_Ti)
        E_dot_total = E_dot_adv + E_dot_parallel + E_dot_curvature + E_dot_drive + E_dot_diss

        return {
            "E_dot_adv": E_dot_adv,
            "E_dot_parallel": E_dot_parallel,
            "E_dot_curvature": E_dot_curvature,
            "E_dot_drive": E_dot_drive,
            "E_dot_diss": E_dot_diss,
            "E_dot_total": E_dot_total,
        }
