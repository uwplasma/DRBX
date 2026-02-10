from __future__ import annotations

from typing import Literal

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp

from jaxdrb.operators.brackets import poisson_bracket_arakawa, poisson_bracket_centered

from .grid import Grid2D
from .fd import ddx as ddx_fd
from .fd import ddy as ddy_fd
from .fd import laplacian as laplacian_fd
from .fd import enforce_bc_relaxation, inv_laplacian_cg
from .spectral import dealias, inv_laplacian, laplacian, poisson_bracket_spectral


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

    # Dissipation.
    Dn: float = 0.0
    DOmega: float = 0.0
    DTe: float = 0.0

    # Numerical options.
    bracket: Literal["spectral", "arakawa", "centered"] = "arakawa"
    poisson: Literal["spectral", "cg_fd"] = "spectral"
    dealias_on: bool = True
    k2_min: float = 1e-12
    bc_enforce_nu: float = 0.0

    # Thermal-force coefficient in Ohm's law.
    alpha_Te_ohm: float = 1.71

    # Operator split toggles (shared pattern).
    operator_split_on: bool = False
    operator_conservative_on: bool = True
    operator_source_on: bool = True
    operator_dissipative_on: bool = True


class DRB2DState(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray
    vpar_e: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray


class DRB2DDecomposition(eqx.Module):
    conservative: DRB2DState
    source: DRB2DState
    dissipative: DRB2DState

    def total(self) -> DRB2DState:
        return _state_add(_state_add(self.conservative, self.source), self.dissipative)


def _state_add(a: DRB2DState, b: DRB2DState) -> DRB2DState:
    return DRB2DState(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        vpar_e=a.vpar_e + b.vpar_e,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
    )


def _state_zeros_like(y: DRB2DState) -> DRB2DState:
    z = jnp.zeros_like(y.n)
    return DRB2DState(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z)


class DRB2DModel(eqx.Module):
    params: DRB2DParams
    grid: Grid2D

    def phi_from_omega(self, omega: jnp.ndarray) -> jnp.ndarray:
        if self.params.poisson == "spectral":
            if self.grid.bc.kind_x != 0 or self.grid.bc.kind_y != 0:
                raise ValueError("Spectral Poisson solve requires periodic BCs in x and y.")
            return inv_laplacian(omega, self.grid.k2, k2_min=self.params.k2_min)
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

    def rhs_decomposed(self, t: float, y: DRB2DState) -> DRB2DDecomposition:
        _ = t
        n = y.n
        omega = y.omega
        vpar_e = y.vpar_e
        vpar_i = y.vpar_i
        Te = y.Te

        phi = self.phi_from_omega(omega)

        # Nonlinear ExB advection (conservative on periodic grids with Arakawa bracket).
        adv_n = -self._bracket(phi, n)
        adv_w = -self._bracket(phi, omega)
        adv_ve = -self._bracket(phi, vpar_e)
        adv_vi = -self._bracket(phi, vpar_i)
        adv_Te = -self._bracket(phi, Te)

        # Parallel couplings (modeled by k_par).
        grad_par_phi_pe = self._dpar(phi - n - float(self.params.alpha_Te_ohm) * Te)
        jpar = vpar_i - vpar_e

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
            n=drive_n,
            omega=jnp.zeros_like(omega),
            vpar_e=jnp.zeros_like(vpar_e),
            vpar_i=jnp.zeros_like(vpar_i),
            Te=drive_Te,
        )

        if (
            self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
            and self.params.poisson == "spectral"
        ):
            lap_n = laplacian(n, self.grid.k2)
            lap_w = laplacian(omega, self.grid.k2)
            lap_Te = laplacian(Te, self.grid.k2)
        else:
            lap_n = laplacian_fd(n, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_w = laplacian_fd(omega, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_Te = laplacian_fd(Te, self.grid.dx, self.grid.dy, self.grid.bc)

        dissipative = DRB2DState(
            n=float(self.params.Dn) * lap_n,
            omega=float(self.params.DOmega) * lap_w,
            vpar_e=-(float(self.params.eta) / jnp.maximum(float(self.params.me_hat), 1e-12))
            * (vpar_e - vpar_i),
            vpar_i=jnp.zeros_like(vpar_i),
            Te=float(self.params.DTe) * lap_Te,
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
        phi = self.phi_from_omega(y.omega)
        c_T = 1.5 * float(self.params.alpha_Te_ohm)
        return 0.5 * jnp.mean(
            jnp.real(jnp.conj(y.n) * y.n)
            + jnp.real(jnp.conj(phi) * phi) * self.grid.k2
            + float(self.params.me_hat) * jnp.real(jnp.conj(y.vpar_e) * y.vpar_e)
            + jnp.real(jnp.conj(y.vpar_i) * y.vpar_i)
            + c_T * jnp.real(jnp.conj(y.Te) * y.Te)
        )

    def diffeqsolve(
        self,
        *,
        y0: DRB2DState,
        t0: float,
        t1: float,
        dt0: float,
        save_ts: jnp.ndarray | None = None,
    ):
        term = dfx.ODETerm(lambda t, y, args: self.rhs(t, y))
        solver = dfx.Tsit5()
        stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-8)
        saveat = dfx.SaveAt(ts=save_ts) if save_ts is not None else dfx.SaveAt(t1=True)
        return dfx.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=200_000,
        )
