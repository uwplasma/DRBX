from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.core.nonlinear2d import Core2DModel, coerce_core2d_params
from jaxdrb.core.state import CoreState

from jaxdrb.bc import BC2D
from jaxdrb.operators.brackets import (
    poisson_bracket_arakawa,
    poisson_bracket_arakawa_fd,
    poisson_bracket_centered,
)

from .integrate import DiffraxSolverName, diffeqsolve as diffeqsolve_ode
from .fd import ddx as ddx_fd
from .fd import ddy as ddy_fd
from .fd import biharmonic as biharmonic_fd
from .fd import (
    enforce_bc_relaxation,
    inv_div_n_grad_cg,
    inv_laplacian_cg,
    inv_laplacian_fd_fft,
    laplacian as laplacian_fd,
)
from .grid import Grid2D
from .spectral import (
    biharmonic,
    dealias,
    ddx as ddx_spec,
    ddy as ddy_spec,
    inv_laplacian,
    laplacian,
    poisson_bracket_spectral,
    rfft2,
    irfft2,
)


class DRB2DEMParams(eqx.Module):
    """Electromagnetic extension of the conservative 2D nonlinear DRB testbed."""

    # Background-gradient drives.
    omega_n: float = 0.0
    omega_Te: float = 0.0

    # Parallel coupling modeled via constant k_par (optional).
    kpar: float = 0.0
    eta: float = 0.0
    me_hat: float = 0.2
    beta: float = 0.0
    Dpsi: float = 0.0

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

    # Hyperdiffusion (biharmonic), implemented as -D4 * ∇⁴.
    Dn4: float = 0.0
    DOmega4: float = 0.0
    DTe4: float = 0.0
    Dpsi4: float = 0.0

    # Optional drag on zonal (ky=0) vorticity: -mu * <omega>_y.
    mu_zonal_omega: float = 0.0

    # Numerical options.
    bracket: Literal["spectral", "arakawa", "centered"] = "arakawa"
    poisson: Literal["spectral", "cg_fd"] = "spectral"
    dealias_on: bool = True
    k2_min: float = 1e-12
    bc_enforce_nu: float = 0.0
    # Optional per-field BC overrides (None -> use Grid2D bc).
    bc_n: BC2D | None = None
    bc_omega: BC2D | None = None
    bc_vpar_i: BC2D | None = None
    bc_Te: BC2D | None = None
    bc_psi: BC2D | None = None
    bc_phi: BC2D | None = None
    # Non-Boussinesq variable-coefficient polarization solve settings.
    polarization_cg_maxiter: int = 400
    polarization_cg_tol: float = 1e-8
    polarization_cg_atol: float = 0.0

    # Thermal-force coefficient in Ohm's law.
    alpha_Te_ohm: float = 1.71

    # Operator split toggles.
    operator_split_on: bool = False
    operator_conservative_on: bool = True
    operator_source_on: bool = True
    operator_dissipative_on: bool = True


class DRB2DEMState(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray
    psi: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray


class DRB2DEMDecomposition(eqx.Module):
    conservative: DRB2DEMState
    source: DRB2DEMState
    dissipative: DRB2DEMState

    def total(self) -> DRB2DEMState:
        return _state_add(_state_add(self.conservative, self.source), self.dissipative)


def _state_add(a: DRB2DEMState, b: DRB2DEMState) -> DRB2DEMState:
    return DRB2DEMState(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        psi=a.psi + b.psi,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
    )


def _state_zeros_like(y: DRB2DEMState) -> DRB2DEMState:
    z = jnp.zeros_like(y.n)
    return DRB2DEMState(n=z, omega=z, psi=z, vpar_i=z, Te=z)


def _to_core_state(y: DRB2DEMState) -> CoreState:
    return CoreState.from_optional(
        n=y.n,
        omega=y.omega,
        vpar_e=jnp.zeros_like(y.n),
        vpar_i=y.vpar_i,
        Te=y.Te,
        psi=y.psi,
    )


def _from_core_state(y: CoreState) -> DRB2DEMState:
    return DRB2DEMState(
        n=y.n,
        omega=y.omega,
        psi=y.psi,
        vpar_i=y.vpar_i,
        Te=y.Te,
    )


class DRB2DEMModel(eqx.Module):
    params: DRB2DEMParams
    grid: Grid2D
    _core: Core2DModel = eqx.field(init=False, repr=False)

    def __post_init__(self):
        core_params = coerce_core2d_params(
            self.params,
            model_kind="drb",
            hot_ion_on=False,
            em_on=True,
        )
        object.__setattr__(self, "_core", Core2DModel(params=core_params, grid=self.grid))

    def _bc_or(self, bc: BC2D | None, fallback: BC2D | None = None) -> BC2D:
        if bc is not None:
            return bc
        if fallback is not None:
            return fallback
        return self.grid.bc

    def _bc_phi(self) -> BC2D:
        return self._bc_or(self.params.bc_phi, self._bc_or(self.params.bc_omega, self.grid.bc))

    def _is_periodic_bc(self, bc: BC2D) -> bool:
        return (
            bc.kind_x == 0
            and bc.kind_y == 0
            and self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
        )

    def _is_periodic_pair(self, bc_a: BC2D, bc_b: BC2D) -> bool:
        return self._is_periodic_bc(bc_a) and self._is_periodic_bc(bc_b)

    def phi_from_omega(self, omega: jnp.ndarray, n: jnp.ndarray | None = None) -> jnp.ndarray:
        bc_phi = self._bc_phi()
        if self.params.boussinesq:
            if self.params.poisson == "spectral":
                if not self._is_periodic_bc(bc_phi):
                    raise ValueError("Spectral Poisson solve requires periodic BCs in x and y.")
                return inv_laplacian(omega, self.grid.k2, k2_min=self.params.k2_min)
            if self.params.poisson == "cg_fd":
                try:
                    return inv_laplacian_fd_fft(
                        omega,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=bc_phi,
                    )
                except ValueError:
                    pass
            return inv_laplacian_cg(
                omega,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=bc_phi,
                maxiter=300,
                preconditioner="spectral" if self._is_periodic_bc(bc_phi) else "jacobi",
            )

        if n is None:
            raise ValueError("Non-Boussinesq polarization requires density n.")
        n_eff = float(self.params.n0)
        if self.params.non_boussinesq_perturbed_density_on:
            n_eff = n_eff + jnp.real(jnp.asarray(n))
        n_eff = jnp.maximum(jnp.asarray(n_eff), float(self.params.n0_min))
        precond = "spectral_jacobi" if self._is_periodic_bc(bc_phi) else "jacobi"
        return inv_div_n_grad_cg(
            omega,
            n_coeff=n_eff,
            dx=self.grid.dx,
            dy=self.grid.dy,
            bc=bc_phi,
            maxiter=int(self.params.polarization_cg_maxiter),
            tol=float(self.params.polarization_cg_tol),
            atol=float(self.params.polarization_cg_atol),
            preconditioner=precond,
        )

    def _bracket(
        self, phi: jnp.ndarray, f: jnp.ndarray, *, bc_phi: BC2D, bc_f: BC2D
    ) -> jnp.ndarray:
        periodic_pair = self._is_periodic_pair(bc_phi, bc_f)
        if self.params.bracket == "spectral":
            if not periodic_pair:
                raise ValueError("Spectral bracket requires periodic BCs in x and y.")
            return poisson_bracket_spectral(
                phi,
                f,
                kx=self.grid.kx,
                ky=self.grid.ky,
                dealias_mask=self.grid.dealias_mask if self.params.dealias_on else None,
            )
        if self.params.bracket == "arakawa":
            if periodic_pair:
                return poisson_bracket_arakawa(phi, f, self.grid.dx, self.grid.dy)
            return poisson_bracket_arakawa_fd(phi, f, self.grid.dx, self.grid.dy, bc_phi, bc_f)
        if periodic_pair:
            j = poisson_bracket_centered(phi, f, self.grid.dx, self.grid.dy)
        else:
            dphi_dx = ddx_fd(phi, self.grid.dx, bc_phi)
            dphi_dy = ddy_fd(phi, self.grid.dy, bc_phi)
            df_dx = ddx_fd(f, self.grid.dx, bc_f)
            df_dy = ddy_fd(f, self.grid.dy, bc_f)
            j = dphi_dx * df_dy - dphi_dy * df_dx
        if self.params.dealias_on and periodic_pair:
            return dealias(j, self.grid.dealias_mask)
        return j

    def _dpar(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.params.kpar == 0.0:
            return jnp.zeros_like(f)
        return 1j * float(self.params.kpar) * f

    def _curvature(self, f: jnp.ndarray, bc_f: BC2D) -> jnp.ndarray:
        if not self.params.curvature_on or self.params.curvature_coeff == 0.0:
            return jnp.zeros_like(f)
        if self._is_periodic_bc(bc_f) and self.params.poisson == "spectral":
            df_dy = ddy_spec(f, self.grid.ky)
        else:
            df_dy = ddy_fd(f, self.grid.dy, bc_f)
        return -float(self.params.curvature_coeff) * df_dy

    def _psi_rhs_from_terms(self, term_hat: jnp.ndarray) -> jnp.ndarray:
        coef = 0.5 * float(self.params.beta) + float(self.params.me_hat) * jnp.maximum(
            self.grid.k2, self.params.k2_min
        )
        coef = jnp.maximum(coef, 1e-12)
        psi_rhs_hat = term_hat / coef
        # For kpar>0 runs we evolve complex Fourier-mode amplitudes; preserve the complex
        # inverse FFT in that case. For kpar=0 the state is expected to be real-valued.
        return irfft2(psi_rhs_hat, real_output=(self.params.kpar == 0.0))

    def rhs_decomposed(self, t: float, y: DRB2DEMState) -> DRB2DEMDecomposition:
        core_state = _to_core_state(y)
        split = self._core.rhs_decomposed(t, core_state)
        return DRB2DEMDecomposition(
            conservative=_from_core_state(split.conservative),
            source=_from_core_state(split.source),
            dissipative=_from_core_state(split.dissipative),
        )

    def rhs(self, t: float, y: DRB2DEMState) -> DRB2DEMState:
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

    def energy(self, y: DRB2DEMState) -> jnp.ndarray:
        phi = self.phi_from_omega(y.omega, n=y.n)
        bc_phi = self._bc_phi()
        c_T = 1.5 * float(self.params.alpha_Te_ohm)
        jpar = -laplacian(y.psi, self.grid.k2)
        if (
            self.params.boussinesq
            and self._is_periodic_bc(bc_phi)
            and self.params.poisson == "spectral"
        ):
            phi_term = jnp.real(jnp.conj(phi) * phi) * self.grid.k2
        else:
            n_eff = 1.0
            if not self.params.boussinesq:
                n_eff = float(self.params.n0)
                if self.params.non_boussinesq_perturbed_density_on:
                    n_eff = n_eff + jnp.real(jnp.asarray(y.n))
                n_eff = jnp.maximum(jnp.asarray(n_eff), float(self.params.n0_min))
            if self._is_periodic_bc(bc_phi) and self.params.poisson == "spectral":
                gradphi_x = ddx_spec(phi, self.grid.kx)
                gradphi_y = ddy_spec(phi, self.grid.ky)
            else:
                gradphi_x = ddx_fd(phi, self.grid.dx, bc_phi)
                gradphi_y = ddy_fd(phi, self.grid.dy, bc_phi)
            phi_term = jnp.real(n_eff) * (
                jnp.real(jnp.conj(gradphi_x) * gradphi_x)
                + jnp.real(jnp.conj(gradphi_y) * gradphi_y)
            )
        psi_term = float(self.params.beta) * jnp.real(jnp.conj(y.psi) * jpar)
        return 0.5 * jnp.mean(
            jnp.real(jnp.conj(y.n) * y.n)
            + phi_term
            + jnp.real(jnp.conj(y.vpar_i) * y.vpar_i)
            + c_T * jnp.real(jnp.conj(y.Te) * y.Te)
            + psi_term
        )

    def energy_rate(self, y: DRB2DEMState, dy: DRB2DEMState) -> jnp.ndarray:
        if self.params.boussinesq:
            phi = self.phi_from_omega(y.omega, n=y.n)
            c_T = 1.5 * float(self.params.alpha_Te_ohm)
            jpar = -laplacian(y.psi, self.grid.k2)
            return jnp.mean(
                jnp.real(jnp.conj(y.n) * dy.n)
                - jnp.real(jnp.conj(phi) * dy.omega)
                + jnp.real(jnp.conj(y.vpar_i) * dy.vpar_i)
                + c_T * jnp.real(jnp.conj(y.Te) * dy.Te)
                + float(self.params.beta) * jnp.real(jnp.conj(jpar) * dy.psi)
            )

        eps = jnp.asarray(1.0e-7, dtype=jnp.float64)
        y_plus = DRB2DEMState(
            n=y.n + eps * dy.n,
            omega=y.omega + eps * dy.omega,
            psi=y.psi + eps * dy.psi,
            vpar_i=y.vpar_i + eps * dy.vpar_i,
            Te=y.Te + eps * dy.Te,
        )
        y_minus = DRB2DEMState(
            n=y.n - eps * dy.n,
            omega=y.omega - eps * dy.omega,
            psi=y.psi - eps * dy.psi,
            vpar_i=y.vpar_i - eps * dy.vpar_i,
            Te=y.Te - eps * dy.Te,
        )
        E_plus = self.energy(y_plus)
        E_minus = self.energy(y_minus)
        return (E_plus - E_minus) / (2.0 * eps)

    def energy_budget(self, y: DRB2DEMState) -> dict[str, jnp.ndarray]:
        """Return a term-by-term energy budget for EM DRB2D."""

        n = y.n
        omega = y.omega
        psi = y.psi
        vpar_i = y.vpar_i
        Te = y.Te
        phi = self.phi_from_omega(omega, n=n)
        bc_n = self._bc_or(self.params.bc_n)
        bc_omega = self._bc_or(self.params.bc_omega)
        bc_psi = self._bc_or(self.params.bc_psi)
        bc_vpar_i = self._bc_or(self.params.bc_vpar_i)
        bc_Te = self._bc_or(self.params.bc_Te)
        bc_phi = self._bc_phi()
        psi_hat = rfft2(psi)
        jpar = -laplacian(psi, self.grid.k2)
        vpar_e = vpar_i - jpar

        adv_n = -self._bracket(phi, n, bc_phi=bc_phi, bc_f=bc_n)
        adv_w = -self._bracket(phi, omega, bc_phi=bc_phi, bc_f=bc_omega)
        adv_psi = -self._bracket(phi, psi, bc_phi=bc_phi, bc_f=bc_psi)
        adv_vi = -self._bracket(phi, vpar_i, bc_phi=bc_phi, bc_f=bc_vpar_i)
        adv_Te = -self._bracket(phi, Te, bc_phi=bc_phi, bc_f=bc_Te)

        grad_par_phi_pe = self._dpar(phi - n - float(self.params.alpha_Te_ohm) * Te)
        grad_par_hat = rfft2(grad_par_phi_pe)
        jpar_hat = self.grid.k2 * psi_hat
        lap_psi_hat = -self.grid.k2 * psi_hat

        par_n = -self._dpar(vpar_e)
        par_w = self._dpar(jpar)
        par_psi = self._psi_rhs_from_terms(-grad_par_hat)
        par_vi = -self._dpar(phi)
        par_Te = -(2.0 / 3.0) * self._dpar(vpar_e)

        C_phi = self._curvature(phi, bc_phi)
        C_n = self._curvature(n, bc_n)
        C_Te = self._curvature(Te, bc_Te)
        C_p = C_n + C_Te
        C_Te = (2.0 / 3.0) * ((7.0 / 2.0) * C_Te + C_n - C_phi)

        drive_n = 0.0
        drive_Te = 0.0
        if self.params.omega_n != 0.0 or self.params.omega_Te != 0.0:
            dphi_dy = ddy_fd(phi, self.grid.dy, bc_phi)
            drive_n = -float(self.params.omega_n) * dphi_dy
            drive_Te = -float(self.params.omega_Te) * dphi_dy

        if self._is_periodic_bc(bc_n) and self.params.poisson == "spectral":
            lap_n = laplacian(n, self.grid.k2)
            bih_n = biharmonic(n, self.grid.k2)
        else:
            lap_n = laplacian_fd(n, self.grid.dx, self.grid.dy, bc_n)
            bih_n = biharmonic_fd(n, self.grid.dx, self.grid.dy, bc_n)
        if self._is_periodic_bc(bc_omega) and self.params.poisson == "spectral":
            lap_w = laplacian(omega, self.grid.k2)
            bih_w = biharmonic(omega, self.grid.k2)
        else:
            lap_w = laplacian_fd(omega, self.grid.dx, self.grid.dy, bc_omega)
            bih_w = biharmonic_fd(omega, self.grid.dx, self.grid.dy, bc_omega)
        if self._is_periodic_bc(bc_Te) and self.params.poisson == "spectral":
            lap_Te = laplacian(Te, self.grid.k2)
            bih_Te = biharmonic(Te, self.grid.k2)
        else:
            lap_Te = laplacian_fd(Te, self.grid.dx, self.grid.dy, bc_Te)
            bih_Te = biharmonic_fd(Te, self.grid.dx, self.grid.dy, bc_Te)

        omega_zonal = jnp.mean(omega, axis=1, keepdims=True) + jnp.zeros_like(omega)

        diss_psi_hat = (
            -float(self.params.eta) * jpar_hat
            + float(self.params.Dpsi) * lap_psi_hat
            - float(self.params.Dpsi4) * (self.grid.k2**2) * psi_hat
        )
        diss_psi = self._psi_rhs_from_terms(diss_psi_hat)

        diss_n = float(self.params.Dn) * lap_n - float(self.params.Dn4) * bih_n
        diss_w = (
            float(self.params.DOmega) * lap_w
            - float(self.params.DOmega4) * bih_w
            - float(self.params.mu_zonal_omega) * omega_zonal
        )
        diss_vi = jnp.zeros_like(vpar_i)
        diss_Te = float(self.params.DTe) * lap_Te - float(self.params.DTe4) * bih_Te

        if self.params.bc_enforce_nu != 0.0:
            diss_n = diss_n + enforce_bc_relaxation(
                n, dx=self.grid.dx, dy=self.grid.dy, bc=bc_n, nu=self.params.bc_enforce_nu
            )
            diss_w = diss_w + enforce_bc_relaxation(
                omega,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=bc_omega,
                nu=self.params.bc_enforce_nu,
            )
            diss_psi = diss_psi + enforce_bc_relaxation(
                psi,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=bc_psi,
                nu=self.params.bc_enforce_nu,
            )
            diss_vi = diss_vi + enforce_bc_relaxation(
                vpar_i,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=bc_vpar_i,
                nu=self.params.bc_enforce_nu,
            )
            diss_Te = diss_Te + enforce_bc_relaxation(
                Te, dx=self.grid.dx, dy=self.grid.dy, bc=bc_Te, nu=self.params.bc_enforce_nu
            )

        c_T = 1.5 * float(self.params.alpha_Te_ohm)

        def edot(n_term, w_term, psi_term, vi_term, Te_term):
            return jnp.mean(
                jnp.real(jnp.conj(n) * n_term)
                - jnp.real(jnp.conj(phi) * w_term)
                + jnp.real(jnp.conj(vpar_i) * vi_term)
                + c_T * jnp.real(jnp.conj(Te) * Te_term)
                + float(self.params.beta) * jnp.real(jnp.conj(jpar) * psi_term)
            )

        E_dot_adv = edot(adv_n, adv_w, adv_psi, adv_vi, adv_Te)
        E_dot_parallel = edot(par_n, par_w, par_psi, par_vi, par_Te)
        E_dot_curvature = edot(C_p - C_phi, C_p, 0.0, 0.0, C_Te)
        E_dot_drive = edot(drive_n, 0.0, 0.0, 0.0, drive_Te)
        E_dot_diss = edot(diss_n, diss_w, diss_psi, diss_vi, diss_Te)
        E_dot_total = E_dot_adv + E_dot_parallel + E_dot_curvature + E_dot_drive + E_dot_diss

        return {
            "E_dot_adv": E_dot_adv,
            "E_dot_parallel": E_dot_parallel,
            "E_dot_curvature": E_dot_curvature,
            "E_dot_drive": E_dot_drive,
            "E_dot_diss": E_dot_diss,
            "E_dot_total": E_dot_total,
        }

    def diffeqsolve(
        self,
        *,
        y0: DRB2DEMState,
        t0: float,
        t1: float,
        dt0: float,
        save_ts: jnp.ndarray | None = None,
        solver: DiffraxSolverName = "tsit5",
        adaptive: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        max_steps: int = 200_000,
        progress: bool | None = None,
    ):
        return diffeqsolve_ode(
            self.rhs,
            y0=y0,
            t0=t0,
            t1=t1,
            dt0=dt0,
            save_ts=save_ts,
            solver=solver,
            adaptive=adaptive,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            progress=progress,
        )
