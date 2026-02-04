from __future__ import annotations

from typing import Literal

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.operators.brackets import poisson_bracket_arakawa, poisson_bracket_centered

from .grid import Grid2D
from .neutrals import NeutralParams, rhs_neutral
from .fd import biharmonic as biharmonic_fd
from .fd import ddx as ddx_fd
from .fd import ddy as ddy_fd
from .fd import laplacian as laplacian_fd
from .fd import enforce_bc_relaxation, inv_laplacian_cg
from .spectral import biharmonic, ddy, dealias, inv_laplacian, laplacian, poisson_bracket_spectral


class HW2DParams(eqx.Module):
    """Hasegawa–Wakatani-like 2D nonlinear drift-wave testbed."""

    # Background-gradient drive (proxy for R/Ln).
    kappa: float = 1.0

    # Parallel coupling (adiabaticity / resistive coupling).
    alpha: float = 1.0

    # Dissipation.
    Dn: float = 1e-3
    DOmega: float = 1e-3
    nu4_n: float = 0.0  # hyperdiffusion coefficient for n: adds -nu4_n ∇⁴ n
    nu4_omega: float = 0.0  # hyperdiffusion coefficient for omega: adds -nu4_omega ∇⁴ omega

    # Numerical options.
    bracket: Literal["spectral", "arakawa", "centered"] = "arakawa"
    poisson: Literal["spectral", "cg_fd"] = "spectral"
    dealias_on: bool = True
    k2_min: float = 1e-12
    bc_enforce_nu: float = 0.0  # boundary relaxation rate for non-periodic BCs

    # Optional "modified HW" coupling: apply α(φ-n) only to non-zonal components (ky≠0),
    # avoiding unphysical damping of zonal flows.
    alpha_nonzonal_only: bool = False

    # Optional neutral coupling.
    neutrals: NeutralParams = NeutralParams()


class HW2DState(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray
    N: jnp.ndarray | None = None


class HW2DModel(eqx.Module):
    params: HW2DParams
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
            # Arakawa's Jacobian is designed to conserve quadratic invariants on periodic grids.
            # Applying an FFT filter to it can break these conservation properties, so we return it
            # as-is.
            return poisson_bracket_arakawa(phi, f, self.grid.dx, self.grid.dy)
        if self.grid.bc.kind_x == 0 and self.grid.bc.kind_y == 0:
            j = poisson_bracket_centered(phi, f, self.grid.dx, self.grid.dy)
        else:
            dphi_dx = ddx_fd(phi, self.grid.dx, self.grid.bc)
            dphi_dy = ddy_fd(phi, self.grid.dy, self.grid.bc)
            df_dx = ddx_fd(f, self.grid.dx, self.grid.bc)
            df_dy = ddy_fd(f, self.grid.dy, self.grid.bc)
            j = dphi_dx * df_dy - dphi_dy * df_dx
        return dealias(j, self.grid.dealias_mask) if self.params.dealias_on else j

    def rhs(self, t: float, y: HW2DState) -> HW2DState:
        _ = t
        n = y.n
        omega = y.omega

        phi = self.phi_from_omega(omega)

        # Main nonlinear advection.
        adv_n = self._bracket(phi, n)
        adv_w = self._bracket(phi, omega)

        # Background-gradient drive (E×B drift across background gradient).
        # Standard HW form uses -kappa ∂y phi in the density equation.
        if (
            self.params.bracket == "spectral"
            and self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
        ):
            dphi_dy = ddy(phi, self.grid.ky)
            dn_dy = ddy(n, self.grid.ky)
        else:
            dphi_dy = ddy_fd(phi, self.grid.dy, self.grid.bc)
            dn_dy = ddy_fd(n, self.grid.dy, self.grid.bc)

        drive_n = -self.params.kappa * dphi_dy
        drive_w = -self.params.kappa * dn_dy

        # Resistive/adiabatic coupling.
        couple = self.params.alpha * (phi - n)
        if self.params.alpha_nonzonal_only:
            couple = couple - jnp.mean(couple, axis=1, keepdims=True)

        if (
            self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
            and self.params.poisson == "spectral"
        ):
            lap_n = laplacian(n, self.grid.k2)
            lap_w = laplacian(omega, self.grid.k2)
        else:
            lap_n = laplacian_fd(n, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_w = laplacian_fd(omega, self.grid.dx, self.grid.dy, self.grid.bc)

        dn = -adv_n + drive_n + couple + self.params.Dn * lap_n
        dw = -adv_w + drive_w + couple + self.params.DOmega * lap_w

        # Optional hyperdiffusion (biharmonic). This is commonly used in HW studies to control
        # the enstrophy cascade with minimal impact on large scales.
        if self.params.nu4_n != 0.0 or self.params.nu4_omega != 0.0:
            if (
                self.grid.bc.kind_x == 0
                and self.grid.bc.kind_y == 0
                and self.params.poisson == "spectral"
            ):
                dn = dn - self.params.nu4_n * biharmonic(n, self.grid.k2)
                dw = dw - self.params.nu4_omega * biharmonic(omega, self.grid.k2)
            else:
                dn = dn - self.params.nu4_n * biharmonic_fd(
                    n, self.grid.dx, self.grid.dy, self.grid.bc
                )
                dw = dw - self.params.nu4_omega * biharmonic_fd(
                    omega, self.grid.dx, self.grid.dy, self.grid.bc
                )

        # Optional boundary enforcement (useful for non-periodic BC experiments).
        if self.params.bc_enforce_nu != 0.0:
            dn = dn + enforce_bc_relaxation(
                n, dx=self.grid.dx, dy=self.grid.dy, bc=self.grid.bc, nu=self.params.bc_enforce_nu
            )
            dw = dw + enforce_bc_relaxation(
                omega,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=self.grid.bc,
                nu=self.params.bc_enforce_nu,
            )

        if y.N is None:
            return HW2DState(n=dn, omega=dw, N=None)

        # Neutral coupling (optional).
        adv_N = self._bracket(phi, y.N)
        if (
            self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
            and self.params.poisson == "spectral"
        ):
            lap_N = laplacian(y.N, self.grid.k2)
        else:
            lap_N = laplacian_fd(y.N, self.grid.dx, self.grid.dy, self.grid.bc)
        dN, dn_from_neutrals = rhs_neutral(
            N=y.N,
            n=n,
            dn0=self.params.neutrals,
            adv_N=adv_N,
            lap_N=lap_N,
        )
        if self.params.bc_enforce_nu != 0.0:
            dN = dN + enforce_bc_relaxation(
                y.N, dx=self.grid.dx, dy=self.grid.dy, bc=self.grid.bc, nu=self.params.bc_enforce_nu
            )
        return HW2DState(n=dn + dn_from_neutrals, omega=dw, N=dN)

    def diffeqsolve(
        self,
        *,
        y0: HW2DState,
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

    def diagnostics(self, y: HW2DState) -> dict[str, jnp.ndarray]:
        """Compute basic integral diagnostics."""

        phi = self.phi_from_omega(y.omega)

        # Energy-like quantity: 0.5 ∫ (n^2 + |∇phi|^2) dA
        if (
            self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
            and self.params.poisson == "spectral"
        ):
            from .spectral import ddx as ddx_spec
            from .spectral import ddy as ddy_spec

            gradphi_x = ddx_spec(phi, self.grid.kx)
            gradphi_y = ddy_spec(phi, self.grid.ky)
        else:
            gradphi_x = ddx_fd(phi, self.grid.dx, self.grid.bc)
            gradphi_y = ddy_fd(phi, self.grid.dy, self.grid.bc)
        E = 0.5 * jnp.mean(y.n**2 + gradphi_x**2 + gradphi_y**2)

        Z = 0.5 * jnp.mean(y.omega**2)  # enstrophy-like
        out = {"E": E, "Z": Z}
        if y.N is not None:
            out["Nbar"] = jnp.mean(y.N)
        return out

    def energy_budget(self, y: HW2DState) -> dict[str, jnp.ndarray]:
        """Compute a discrete energy budget for HW2D.

        We use the standard energy functional (Camargo et al. 1995):

          E = 1/2 ⟨ n^2 + |∇φ|^2 ⟩

        and the periodic-domain identity:

          d/dt (1/2⟨|∇φ|^2⟩) = -⟨ φ ∂t ω ⟩,

        so that:

          Ė = ⟨ n ∂t n - φ ∂t ω ⟩.

        This lets us attribute contributions from each term in the RHS. In the continuous system,
        the Poisson-bracket advection terms are energy-conserving; this is a useful numerical check.
        """

        n = y.n
        omega = y.omega
        phi = self.phi_from_omega(omega)

        adv_n = self._bracket(phi, n)
        adv_w = self._bracket(phi, omega)

        if (
            self.params.bracket == "spectral"
            and self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
        ):
            dphi_dy = ddy(phi, self.grid.ky)
            dn_dy = ddy(n, self.grid.ky)
        else:
            dphi_dy = ddy_fd(phi, self.grid.dy, self.grid.bc)
            dn_dy = ddy_fd(n, self.grid.dy, self.grid.bc)

        drive_n = -self.params.kappa * dphi_dy
        drive_w = -self.params.kappa * dn_dy

        couple = self.params.alpha * (phi - n)
        if self.params.alpha_nonzonal_only:
            couple = couple - jnp.mean(couple, axis=1, keepdims=True)

        if (
            self.grid.bc.kind_x == 0
            and self.grid.bc.kind_y == 0
            and self.params.poisson == "spectral"
        ):
            lap_n = laplacian(n, self.grid.k2)
            lap_w = laplacian(omega, self.grid.k2)
            bih_n = biharmonic(n, self.grid.k2)
            bih_w = biharmonic(omega, self.grid.k2)
        else:
            lap_n = laplacian_fd(n, self.grid.dx, self.grid.dy, self.grid.bc)
            lap_w = laplacian_fd(omega, self.grid.dx, self.grid.dy, self.grid.bc)
            bih_n = biharmonic_fd(n, self.grid.dx, self.grid.dy, self.grid.bc)
            bih_w = biharmonic_fd(omega, self.grid.dx, self.grid.dy, self.grid.bc)

        dn_adv = -adv_n
        dw_adv = -adv_w
        dn_drive = drive_n
        dw_drive = drive_w
        dn_couple = couple
        dw_couple = couple
        dn_diff = self.params.Dn * lap_n
        dw_diff = self.params.DOmega * lap_w
        dn_hyper = -self.params.nu4_n * bih_n
        dw_hyper = -self.params.nu4_omega * bih_w

        def edot(dn_term, dw_term):
            return jnp.mean(n * dn_term - phi * dw_term)

        out = {
            "E_dot_adv": edot(dn_adv, dw_adv),
            "E_dot_drive": edot(dn_drive, dw_drive),
            "E_dot_couple": edot(dn_couple, dw_couple),
            "E_dot_diff": edot(dn_diff, dw_diff),
            "E_dot_hyper": edot(dn_hyper, dw_hyper),
        }
        out["E_dot_total"] = (
            out["E_dot_adv"]
            + out["E_dot_drive"]
            + out["E_dot_couple"]
            + out["E_dot_diff"]
            + out["E_dot_hyper"]
        )
        return out


@eqx.filter_jit
def hw2d_random_ic(key, grid: Grid2D, *, amp: float = 1e-2, include_neutrals: bool = False):
    n0 = amp * jax.random.normal(key, (grid.nx, grid.ny))
    omega0 = amp * jax.random.normal(jax.random.split(key, 2)[1], (grid.nx, grid.ny))
    N0 = None
    if include_neutrals:
        N0 = jnp.ones((grid.nx, grid.ny))
    return HW2DState(n=n0, omega=omega0, N=N0)
