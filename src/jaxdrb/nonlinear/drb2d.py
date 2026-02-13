from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.operators.brackets import poisson_bracket_arakawa, poisson_bracket_centered

from .integrate import DiffraxSolverName, diffeqsolve as diffeqsolve_ode
from .grid import Grid2D
from .fd import ddx as ddx_fd
from .fd import ddy as ddy_fd
from .fd import biharmonic as biharmonic_fd
from .fd import laplacian as laplacian_fd
from .fd import enforce_bc_relaxation, inv_div_n_grad_cg, inv_laplacian_cg
from .spectral import (
    biharmonic,
    dealias,
    ddx as ddx_spec,
    ddy as ddy_spec,
    inv_laplacian,
    laplacian,
    poisson_bracket_spectral,
)
from .neutrals import NeutralParams, rhs_neutral


class DRB2DParams(eqx.Module):
    """Conservative 2D nonlinear DRB testbed (periodic in x/y).

    This is a minimal nonlinear DRB model used to validate conservative operators
    and prepare the full nonlinear field-line system.
    """

    # Background-gradient drives (optional).
    omega_n: float = 0.0
    omega_Te: float = 0.0

    # Parallel coupling modeled via constant k_par (optional).
    kpar: float = 0.0
    eta: float = 0.0
    me_hat: float = 0.2

    # Curvature drive (simple slab interchange model).
    curvature_on: bool = False
    curvature_coeff: float = 0.0

    # Polarization closure (Boussinesq vs non-Boussinesq).
    boussinesq: bool = True
    n0: float = 1.0
    n0_min: float = 1e-6
    n0_max: float | None = None
    non_boussinesq_perturbed_density_on: bool = False

    # Dissipation.
    Dn: float = 0.0
    DOmega: float = 0.0
    DTe: float = 0.0

    # Hyperdiffusion (biharmonic), implemented as -D4 * ∇⁴. Useful for keeping coarse-grid
    # nonlinear movies visually turbulent without excessive Laplacian diffusion.
    Dn4: float = 0.0
    DOmega4: float = 0.0
    DTe4: float = 0.0

    # Optional drag on zonal (ky=0) vorticity, implemented as -mu * <omega>_y.
    # This is a common numerical control knob in 2D drift-wave/interchange turbulence to
    # prevent long-time condensation into a purely zonal/banded state.
    mu_zonal_omega: float = 0.0

    # Optional linear damping terms, implemented as -mu * f.
    #
    # In 2D perpendicular-box testbeds there is no explicit parallel coordinate; these
    # terms act as a simple surrogate for parallel losses / large-scale friction to help
    # reach statistically steady turbulence in long-time runs.
    mu_lin_n: float = 0.0
    mu_lin_omega: float = 0.0
    mu_lin_vpar_e: float = 0.0
    mu_lin_vpar_i: float = 0.0
    mu_lin_Te: float = 0.0

    # Numerical options.
    bracket: Literal["spectral", "arakawa", "centered"] = "arakawa"
    poisson: Literal["spectral", "cg_fd"] = "spectral"
    dealias_on: bool = True
    k2_min: float = 1e-12
    bc_enforce_nu: float = 0.0
    # Non-Boussinesq variable-coefficient polarization solve settings.
    polarization_cg_maxiter: int = 400
    polarization_cg_tol: float = 1e-8
    polarization_cg_atol: float = 0.0
    polarization_preconditioner: str = "auto"
    polarization_precond_shift: float = 1e-12

    # Thermal-force coefficient in Ohm's law.
    alpha_Te_ohm: float = 1.71

    # Operator split toggles (shared pattern).
    operator_split_on: bool = False
    operator_conservative_on: bool = True
    operator_source_on: bool = True
    operator_dissipative_on: bool = True

    # Optional neutral coupling (plasma-neutral exchange).
    neutrals: NeutralParams = NeutralParams()

    # Optional SOL-like closed→open radial setup (LCFS at fixed x = x_s).
    sol_on: bool = False
    sol_xs: float = 0.0
    sol_width: float = 0.05
    sol_n_core: float = 1.0
    sol_n_sol: float = 0.2
    sol_Te_core: float = 1.0
    sol_Te_sol: float = 0.2
    sol_relax_core: float = 0.2
    sol_relax_open: float = 0.6
    sol_sink_open_n: float = 0.0
    sol_sink_open_Te: float = 0.0
    sol_sink_open_omega: float = 0.0
    sol_sink_open_vpar: float = 0.0


class DRB2DState(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray
    vpar_e: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray
    N: jnp.ndarray | None = None


class DRB2DDecomposition(eqx.Module):
    conservative: DRB2DState
    source: DRB2DState
    dissipative: DRB2DState

    def total(self) -> DRB2DState:
        return _state_add(_state_add(self.conservative, self.source), self.dissipative)


def _state_add(a: DRB2DState, b: DRB2DState) -> DRB2DState:
    if a.N is None and b.N is None:
        N = None
    elif a.N is None:
        N = b.N
    elif b.N is None:
        N = a.N
    else:
        N = a.N + b.N
    return DRB2DState(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        vpar_e=a.vpar_e + b.vpar_e,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
        N=N,
    )


def _state_zeros_like(y: DRB2DState) -> DRB2DState:
    z = jnp.zeros_like(y.n)
    N = None if y.N is None else jnp.zeros_like(y.N)
    return DRB2DState(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z, N=N)


class DRB2DModel(eqx.Module):
    params: DRB2DParams
    grid: Grid2D

    def phi_from_omega(self, omega: jnp.ndarray, n: jnp.ndarray | None = None) -> jnp.ndarray:
        if self.params.boussinesq:
            if self.params.poisson == "spectral":
                if self.grid.bc.kind_x != 0 or self.grid.bc.kind_y != 0:
                    raise ValueError("Spectral Poisson solve requires periodic BCs in x and y.")
                return inv_laplacian(omega, self.grid.k2, k2_min=self.params.k2_min)
            return inv_laplacian_cg(
                omega,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=self.grid.bc,
                maxiter=300,
                preconditioner="spectral" if self.grid.bc.kind_x == 0 else "jacobi",
            )

        if n is None:
            raise ValueError("Non-Boussinesq polarization requires density n.")
        n_eff = float(self.params.n0)
        if self.params.non_boussinesq_perturbed_density_on:
            n_eff = n_eff + jnp.real(jnp.asarray(n))
        n_eff = jnp.maximum(jnp.asarray(n_eff), float(self.params.n0_min))
        if self.params.n0_max is not None:
            n_eff = jnp.minimum(n_eff, float(self.params.n0_max))
        if self.params.polarization_preconditioner == "auto":
            precond = (
                "spectral_jacobi"
                if (self.grid.bc.kind_x == 0 and self.grid.bc.kind_y == 0)
                else "jacobi"
            )
        else:
            precond = self.params.polarization_preconditioner
        return inv_div_n_grad_cg(
            omega,
            n_coeff=n_eff,
            dx=self.grid.dx,
            dy=self.grid.dy,
            bc=self.grid.bc,
            maxiter=int(self.params.polarization_cg_maxiter),
            tol=float(self.params.polarization_cg_tol),
            atol=float(self.params.polarization_cg_atol),
            preconditioner=precond,
            preconditioner_shift=float(self.params.polarization_precond_shift),
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
            df_dy = ddy_spec(f, self.grid.ky)
        else:
            df_dy = ddy_fd(f, self.grid.dy, self.grid.bc)
        return -float(self.params.curvature_coeff) * df_dy

    def _sol_masks(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        if not self.params.sol_on:
            z = jnp.zeros((self.grid.nx, self.grid.ny), dtype=self.grid.x.dtype)
            return z, 1.0 - z
        xs = float(self.params.sol_xs)
        width = max(float(self.params.sol_width), 1e-6)
        x = self.grid.x[:, None]
        mask_open = 0.5 * (1.0 + jnp.tanh((x - xs) / width))
        mask_closed = 1.0 - mask_open
        return mask_closed, mask_open

    def rhs_decomposed(self, t: float, y: DRB2DState) -> DRB2DDecomposition:
        _ = t
        n = y.n
        omega = y.omega
        vpar_e = y.vpar_e
        vpar_i = y.vpar_i
        Te = y.Te

        phi = self.phi_from_omega(omega, n=n)

        # Nonlinear ExB advection (conservative on periodic grids with Arakawa bracket).
        adv_n = -self._bracket(phi, n)
        adv_w = -self._bracket(phi, omega)
        adv_ve = -self._bracket(phi, vpar_e)
        adv_vi = -self._bracket(phi, vpar_i)
        adv_Te = -self._bracket(phi, Te)

        # Parallel couplings (modeled by k_par).
        grad_par_phi_pe = self._dpar(phi - n - float(self.params.alpha_Te_ohm) * Te)
        jpar = vpar_i - vpar_e

        C_phi = self._curvature(phi)
        C_p = self._curvature(n + Te)
        C_T = (2.0 / 3.0) * self._curvature((7.0 / 2.0) * Te + n - phi)

        conservative = DRB2DState(
            n=adv_n - self._dpar(vpar_e),
            omega=adv_w + self._dpar(jpar),
            vpar_e=adv_ve + grad_par_phi_pe / jnp.maximum(float(self.params.me_hat), 1e-12),
            vpar_i=adv_vi - self._dpar(phi),
            Te=adv_Te - (2.0 / 3.0) * self._dpar(vpar_e),
        )

        # Background drives (simple -ky omega_n phi, -ky omega_Te phi in Fourier-y).
        drive_n = 0.0
        drive_Te = 0.0
        if self.params.omega_n != 0.0 or self.params.omega_Te != 0.0:
            if self.grid.bc.kind_y != 0:
                raise ValueError("Drive terms assume periodic y for spectral ky representation.")
            dphi_dy = ddy_fd(phi, self.grid.dy, self.grid.bc)
            drive_n = -float(self.params.omega_n) * dphi_dy
            drive_Te = -float(self.params.omega_Te) * dphi_dy

        source = DRB2DState(
            n=drive_n + (C_p - C_phi),
            omega=C_p,
            vpar_e=jnp.zeros_like(vpar_e),
            vpar_i=jnp.zeros_like(vpar_i),
            Te=drive_Te + C_T,
        )

        # Closed→open SOL setup: relax n, Te toward prescribed profiles and apply
        # open-side damping terms. This provides a simple LCFS model in a periodic box.
        sol_sink_n = 0.0
        sol_sink_Te = 0.0
        sol_sink_omega = 0.0
        sol_sink_vpar = 0.0
        if self.params.sol_on:
            mask_closed, mask_open = self._sol_masks()
            n_eq = float(self.params.sol_n_sol) + (
                float(self.params.sol_n_core) - float(self.params.sol_n_sol)
            ) * mask_closed
            Te_eq = float(self.params.sol_Te_sol) + (
                float(self.params.sol_Te_core) - float(self.params.sol_Te_sol)
            ) * mask_closed
            relax = float(self.params.sol_relax_core) * mask_closed + float(
                self.params.sol_relax_open
            ) * mask_open
            sol_source_n = relax * (n_eq - n)
            sol_source_Te = relax * (Te_eq - Te)
            source = _state_add(
                source,
                DRB2DState(
                    n=sol_source_n,
                    omega=jnp.zeros_like(omega),
                    vpar_e=jnp.zeros_like(vpar_e),
                    vpar_i=jnp.zeros_like(vpar_i),
                    Te=sol_source_Te,
                ),
            )
            sol_sink_n = -float(self.params.sol_sink_open_n) * mask_open * n
            sol_sink_Te = -float(self.params.sol_sink_open_Te) * mask_open * Te
            sol_sink_omega = -float(self.params.sol_sink_open_omega) * mask_open * omega
            sol_sink_vpar = -float(self.params.sol_sink_open_vpar) * mask_open

        if (
            self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
            and self.params.poisson == "spectral"
        ):
            lap_n = laplacian(n, self.grid.k2)
            lap_w = laplacian(omega, self.grid.k2)
            lap_Te = laplacian(Te, self.grid.k2)
            bih_n = biharmonic(n, self.grid.k2)
            bih_w = biharmonic(omega, self.grid.k2)
            bih_Te = biharmonic(Te, self.grid.k2)
        else:
            lap_n = laplacian_fd(n, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_w = laplacian_fd(omega, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_Te = laplacian_fd(Te, self.grid.dx, self.grid.dy, self.grid.bc)
            bih_n = biharmonic_fd(n, self.grid.dx, self.grid.dy, self.grid.bc)
            bih_w = biharmonic_fd(omega, self.grid.dx, self.grid.dy, self.grid.bc)
            bih_Te = biharmonic_fd(Te, self.grid.dx, self.grid.dy, self.grid.bc)

        omega_zonal = jnp.mean(omega, axis=1, keepdims=True) + jnp.zeros_like(omega)

        dissipative = DRB2DState(
            n=float(self.params.Dn) * lap_n
            - float(self.params.Dn4) * bih_n
            - float(self.params.mu_lin_n) * n
            + sol_sink_n,
            omega=float(self.params.DOmega) * lap_w
            - float(self.params.DOmega4) * bih_w
            - float(self.params.mu_zonal_omega) * omega_zonal
            - float(self.params.mu_lin_omega) * omega
            + sol_sink_omega,
            vpar_e=-(float(self.params.eta) / jnp.maximum(float(self.params.me_hat), 1e-12))
            * (vpar_e - vpar_i)
            - float(self.params.mu_lin_vpar_e) * vpar_e
            + sol_sink_vpar * vpar_e,
            vpar_i=-float(self.params.mu_lin_vpar_i) * vpar_i + sol_sink_vpar * vpar_i,
            Te=float(self.params.DTe) * lap_Te
            - float(self.params.DTe4) * bih_Te
            - float(self.params.mu_lin_Te) * Te
            + sol_sink_Te,
            N=None,
        )

        if self.params.bc_enforce_nu != 0.0:
            dissipative = _state_add(
                dissipative,
                DRB2DState(
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
                    N=None,
                ),
            )

        if y.N is not None and self.params.neutrals.enabled:
            if (
                self.grid.bc.kind_x == 0
                and self.grid.bc.kind_y == 0
                and self.params.poisson == "spectral"
            ):
                lap_N = laplacian(y.N, self.grid.k2)
            else:
                lap_N = laplacian_fd(y.N, self.grid.dx, self.grid.dy, self.grid.bc)
            adv_N = -self._bracket(phi, y.N)
            dN, dn_from_neutrals, dw_from_neutrals = rhs_neutral(
                N=y.N,
                n=n,
                omega=omega,
                dn0=self.params.neutrals,
                adv_N=adv_N,
                lap_N=lap_N,
            )
            if self.params.bc_enforce_nu != 0.0:
                dN = dN + enforce_bc_relaxation(
                    y.N,
                    dx=self.grid.dx,
                    dy=self.grid.dy,
                    bc=self.grid.bc,
                    nu=self.params.bc_enforce_nu,
                )
            source = _state_add(
                source,
                DRB2DState(
                    n=dn_from_neutrals,
                    omega=dw_from_neutrals,
                    vpar_e=jnp.zeros_like(vpar_e),
                    vpar_i=jnp.zeros_like(vpar_i),
                    Te=jnp.zeros_like(Te),
                    N=dN,
                ),
            )

        return DRB2DDecomposition(conservative=conservative, source=source, dissipative=dissipative)

    def rhs(self, t: float, y: DRB2DState) -> DRB2DState:
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

    def energy(self, y: DRB2DState) -> jnp.ndarray:
        phi = self.phi_from_omega(y.omega, n=y.n)
        c_T = 1.5 * float(self.params.alpha_Te_ohm)
        if not self.params.boussinesq:
            n_eff = float(self.params.n0)
            if self.params.non_boussinesq_perturbed_density_on:
                n_eff = n_eff + jnp.real(jnp.asarray(y.n))
            n_eff = jnp.maximum(jnp.asarray(n_eff), float(self.params.n0_min))
            if (
                self.grid.bc.kind_x == 0
                and self.grid.bc.kind_y == 0
                and self.params.poisson == "spectral"
            ):
                gradphi_x = ddx_spec(phi, self.grid.kx)
                gradphi_y = ddy_spec(phi, self.grid.ky)
            else:
                gradphi_x = ddx_fd(phi, self.grid.dx, self.grid.bc)
                gradphi_y = ddy_fd(phi, self.grid.dy, self.grid.bc)
            phi_term = jnp.real(n_eff) * (
                jnp.real(jnp.conj(gradphi_x) * gradphi_x)
                + jnp.real(jnp.conj(gradphi_y) * gradphi_y)
            )
        else:
            phi_term = jnp.real(jnp.conj(phi) * phi) * self.grid.k2
        return 0.5 * jnp.mean(
            jnp.real(jnp.conj(y.n) * y.n)
            + phi_term
            + float(self.params.me_hat) * jnp.real(jnp.conj(y.vpar_e) * y.vpar_e)
            + jnp.real(jnp.conj(y.vpar_i) * y.vpar_i)
            + c_T * jnp.real(jnp.conj(y.Te) * y.Te)
        )

    def energy_rate(self, y: DRB2DState, dy: DRB2DState) -> jnp.ndarray:
        if self.params.boussinesq:
            phi = self.phi_from_omega(y.omega, n=y.n)
            c_T = 1.5 * float(self.params.alpha_Te_ohm)
            return jnp.mean(
                jnp.real(jnp.conj(y.n) * dy.n)
                - jnp.real(jnp.conj(phi) * dy.omega)
                + float(self.params.me_hat) * jnp.real(jnp.conj(y.vpar_e) * dy.vpar_e)
                + jnp.real(jnp.conj(y.vpar_i) * dy.vpar_i)
                + c_T * jnp.real(jnp.conj(y.Te) * dy.Te)
            )

        # Non-Boussinesq polarization makes E(y) depend on y implicitly through
        # the variable-coefficient SPD solve for phi. Use JVP to obtain a robust
        # directional derivative dE/dt = <∂E/∂y, dy/dt> without tuning a finite
        # difference step size.
        _, edot = jax.jvp(self.energy, (y,), (dy,))
        return edot

    def energy_budget(self, y: DRB2DState) -> dict[str, jnp.ndarray]:
        """Return a term-by-term energy budget for DRB2D.

        Uses the discrete identity:
          dE/dt = < n*dn - phi*dOmega + me_hat*vpar_e*dvpar_e + vpar_i*dvpar_i
                    + 1.5*alpha_Te*T_e*dT_e >
        """
        n = y.n
        omega = y.omega
        vpar_e = y.vpar_e
        vpar_i = y.vpar_i
        Te = y.Te

        phi = self.phi_from_omega(omega, n=n)

        # Nonlinear ExB advection.
        adv_n = -self._bracket(phi, n)
        adv_w = -self._bracket(phi, omega)
        adv_ve = -self._bracket(phi, vpar_e)
        adv_vi = -self._bracket(phi, vpar_i)
        adv_Te = -self._bracket(phi, Te)

        # Parallel couplings (k_par model).
        grad_par_phi_pe = self._dpar(phi - n - float(self.params.alpha_Te_ohm) * Te)
        jpar = vpar_i - vpar_e
        par_n = -self._dpar(vpar_e)
        par_w = self._dpar(jpar)
        par_ve = grad_par_phi_pe / jnp.maximum(float(self.params.me_hat), 1e-12)
        par_vi = -self._dpar(phi)
        par_Te = -(2.0 / 3.0) * self._dpar(vpar_e)

        # Curvature and drives.
        C_phi = self._curvature(phi)
        C_p = self._curvature(n + Te)
        C_T = (2.0 / 3.0) * self._curvature((7.0 / 2.0) * Te + n - phi)

        drive_n = 0.0
        drive_Te = 0.0
        if self.params.omega_n != 0.0 or self.params.omega_Te != 0.0:
            dphi_dy = ddy_fd(phi, self.grid.dy, self.grid.bc)
            drive_n = -float(self.params.omega_n) * dphi_dy
            drive_Te = -float(self.params.omega_Te) * dphi_dy

        sol_source_n = 0.0
        sol_source_Te = 0.0
        sol_sink_n = 0.0
        sol_sink_Te = 0.0
        sol_sink_omega = 0.0
        sol_sink_vpar = 0.0
        if self.params.sol_on:
            mask_closed, mask_open = self._sol_masks()
            n_eq = float(self.params.sol_n_sol) + (
                float(self.params.sol_n_core) - float(self.params.sol_n_sol)
            ) * mask_closed
            Te_eq = float(self.params.sol_Te_sol) + (
                float(self.params.sol_Te_core) - float(self.params.sol_Te_sol)
            ) * mask_closed
            relax = float(self.params.sol_relax_core) * mask_closed + float(
                self.params.sol_relax_open
            ) * mask_open
            sol_source_n = relax * (n_eq - n)
            sol_source_Te = relax * (Te_eq - Te)
            sol_sink_n = -float(self.params.sol_sink_open_n) * mask_open * n
            sol_sink_Te = -float(self.params.sol_sink_open_Te) * mask_open * Te
            sol_sink_omega = -float(self.params.sol_sink_open_omega) * mask_open * omega
            sol_sink_vpar = -float(self.params.sol_sink_open_vpar) * mask_open

        # Dissipation.
        if (
            self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
            and self.params.poisson == "spectral"
        ):
            lap_n = laplacian(n, self.grid.k2)
            lap_w = laplacian(omega, self.grid.k2)
            lap_Te = laplacian(Te, self.grid.k2)
            bih_n = biharmonic(n, self.grid.k2)
            bih_w = biharmonic(omega, self.grid.k2)
            bih_Te = biharmonic(Te, self.grid.k2)
        else:
            lap_n = laplacian_fd(n, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_w = laplacian_fd(omega, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_Te = laplacian_fd(Te, self.grid.dx, self.grid.dy, self.grid.bc)
            bih_n = biharmonic_fd(n, self.grid.dx, self.grid.dy, self.grid.bc)
            bih_w = biharmonic_fd(omega, self.grid.dx, self.grid.dy, self.grid.bc)
            bih_Te = biharmonic_fd(Te, self.grid.dx, self.grid.dy, self.grid.bc)

        omega_zonal = jnp.mean(omega, axis=1, keepdims=True) + jnp.zeros_like(omega)

        diss_n = (
            float(self.params.Dn) * lap_n
            - float(self.params.Dn4) * bih_n
            + sol_sink_n
        )
        diss_w = (
            float(self.params.DOmega) * lap_w
            - float(self.params.DOmega4) * bih_w
            - float(self.params.mu_zonal_omega) * omega_zonal
            + sol_sink_omega
        )
        diss_ve = -(float(self.params.eta) / jnp.maximum(float(self.params.me_hat), 1e-12)) * (
            vpar_e - vpar_i
        )
        diss_ve = diss_ve + sol_sink_vpar * vpar_e
        diss_vi = sol_sink_vpar * vpar_i
        diss_Te = (
            float(self.params.DTe) * lap_Te
            - float(self.params.DTe4) * bih_Te
            + sol_sink_Te
        )

        if self.params.bc_enforce_nu != 0.0:
            diss_n = diss_n + enforce_bc_relaxation(
                n, dx=self.grid.dx, dy=self.grid.dy, bc=self.grid.bc, nu=self.params.bc_enforce_nu
            )
            diss_w = diss_w + enforce_bc_relaxation(
                omega,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=self.grid.bc,
                nu=self.params.bc_enforce_nu,
            )
            diss_ve = diss_ve + enforce_bc_relaxation(
                vpar_e,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=self.grid.bc,
                nu=self.params.bc_enforce_nu,
            )
            diss_vi = diss_vi + enforce_bc_relaxation(
                vpar_i,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=self.grid.bc,
                nu=self.params.bc_enforce_nu,
            )
            diss_Te = diss_Te + enforce_bc_relaxation(
                Te, dx=self.grid.dx, dy=self.grid.dy, bc=self.grid.bc, nu=self.params.bc_enforce_nu
            )

        edot_adv = self.energy_rate(
            y, DRB2DState(n=adv_n, omega=adv_w, vpar_e=adv_ve, vpar_i=adv_vi, Te=adv_Te)
        )
        edot_par = self.energy_rate(
            y, DRB2DState(n=par_n, omega=par_w, vpar_e=par_ve, vpar_i=par_vi, Te=par_Te)
        )
        edot_curv = self.energy_rate(
            y,
            DRB2DState(
                n=C_p - C_phi,
                omega=C_p,
                vpar_e=jnp.zeros_like(vpar_e),
                vpar_i=jnp.zeros_like(vpar_i),
                Te=C_T,
            ),
        )
        edot_drive = self.energy_rate(
            y,
            DRB2DState(
                n=drive_n,
                omega=jnp.zeros_like(omega),
                vpar_e=jnp.zeros_like(vpar_e),
                vpar_i=jnp.zeros_like(vpar_i),
                Te=drive_Te,
            ),
        )
        if self.params.sol_on:
            edot_sol_relax = self.energy_rate(
                y,
                DRB2DState(
                    n=sol_source_n,
                    omega=jnp.zeros_like(omega),
                    vpar_e=jnp.zeros_like(vpar_e),
                    vpar_i=jnp.zeros_like(vpar_i),
                    Te=sol_source_Te,
                ),
            )
            edot_sol_sink = self.energy_rate(
                y,
                DRB2DState(
                    n=sol_sink_n,
                    omega=sol_sink_omega,
                    vpar_e=sol_sink_vpar * vpar_e,
                    vpar_i=sol_sink_vpar * vpar_i,
                    Te=sol_sink_Te,
                ),
            )
        edot_diss = self.energy_rate(
            y,
            DRB2DState(n=diss_n, omega=diss_w, vpar_e=diss_ve, vpar_i=diss_vi, Te=diss_Te),
        )

        out = {
            "E_dot_adv": edot_adv,
            "E_dot_parallel": edot_par,
            "E_dot_curvature": edot_curv,
            "E_dot_drive": edot_drive,
            "E_dot_diss": edot_diss,
        }
        if self.params.sol_on:
            out["E_dot_sol_relax"] = edot_sol_relax
            out["E_dot_sol_sink"] = edot_sol_sink
        if y.N is not None and self.params.neutrals.enabled:
            if (
                self.grid.bc.kind_x == 0
                and self.grid.bc.kind_y == 0
                and self.params.poisson == "spectral"
            ):
                lap_N = laplacian(y.N, self.grid.k2)
            else:
                lap_N = laplacian_fd(y.N, self.grid.dx, self.grid.dy, self.grid.bc)
            adv_N = -self._bracket(phi, y.N)
            dN, dn_from_neutrals, dw_from_neutrals = rhs_neutral(
                N=y.N,
                n=n,
                omega=omega,
                dn0=self.params.neutrals,
                adv_N=adv_N,
                lap_N=lap_N,
            )
            if self.params.bc_enforce_nu != 0.0:
                dN = dN + enforce_bc_relaxation(
                    y.N,
                    dx=self.grid.dx,
                    dy=self.grid.dy,
                    bc=self.grid.bc,
                    nu=self.params.bc_enforce_nu,
                )
            edot_neutrals = self.energy_rate(
                y,
                DRB2DState(
                    n=dn_from_neutrals,
                    omega=dw_from_neutrals,
                    vpar_e=jnp.zeros_like(vpar_e),
                    vpar_i=jnp.zeros_like(vpar_i),
                    Te=jnp.zeros_like(Te),
                    N=dN,
                ),
            )
            out["E_dot_neutrals"] = edot_neutrals

        out["E_dot_total"] = (
            out["E_dot_adv"]
            + out["E_dot_parallel"]
            + out["E_dot_curvature"]
            + out["E_dot_drive"]
            + out["E_dot_diss"]
            + out.get("E_dot_sol_relax", 0.0)
            + out.get("E_dot_sol_sink", 0.0)
            + out.get("E_dot_neutrals", 0.0)
        )
        return out

    def diffeqsolve(
        self,
        *,
        y0: DRB2DState,
        t0: float,
        t1: float,
        dt0: float,
        save_ts: jnp.ndarray | None = None,
        solver: DiffraxSolverName = "tsit5",
        adaptive: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        max_steps: int = 200_000,
        progress: bool = False,
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
