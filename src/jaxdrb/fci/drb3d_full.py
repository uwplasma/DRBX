from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.nonlinear.fd import (
    ddx as ddx_fd,
    ddy as ddy_fd,
    enforce_bc_relaxation,
    inv_div_n_grad_cg,
    inv_laplacian_cg,
    laplacian as lap_fd,
)
from jaxdrb.nonlinear.fv import ddx as ddx_fv
from jaxdrb.nonlinear.fv import ddy as ddy_fv
from jaxdrb.nonlinear.fv import laplacian as lap_fv
from jaxdrb.nonlinear.spectral import ddx as ddx_spec
from jaxdrb.nonlinear.spectral import ddy as ddy_spec
from jaxdrb.nonlinear.spectral import inv_laplacian, laplacian as lap_spec
from jaxdrb.operators.brackets import poisson_bracket_arakawa, poisson_bracket_centered

from .grid import FCISlabGrid
from .parallel import parallel_derivative_centered_3d, parallel_derivative_target_aware_3d


class FCIDRB3DFullParams(eqx.Module):
    """Conservative 3D DRB state on FCI planes (cold-ion baseline).

    State variables are `(n, omega, vpar_e, vpar_i, Te)` on `(nz, nx, ny)` planes.
    """

    omega_n: float = 0.0
    omega_Te: float = 0.0
    kappa: float = 0.0

    alpha: float = 0.0
    eta_par: float = 0.0
    me_hat: float = 1.0

    Dn: float = 0.0
    DOmega: float = 0.0
    Dvpar: float = 0.0
    DTe: float = 0.0
    chi_par: float = 0.0

    bracket: Literal["arakawa", "centered"] = "arakawa"
    perp_operator: Literal["spectral", "fd", "fv"] = "spectral"
    perp_bc: BC2D = eqx.field(default_factory=BC2D.periodic)
    perp_bc_nu: float = 0.0

    use_target_aware_dpar: bool = True
    target_scheme: str = "appendix_b"

    boussinesq: bool = True
    non_boussinesq_perturbed_density_on: bool = True
    n0: float = 1.0
    n0_min: float = 1e-6
    poisson_preconditioner: Literal["spectral", "jacobi", "none"] = "spectral"
    poisson_maxiter: int = 400
    poisson_tol: float = 1e-10
    k2_min: float = 1e-12

    sheath_on: bool = False
    sheath_nu_mom: float = 0.0
    sheath_nu_particle: float = 0.0
    sheath_nu_energy: float = 0.0
    sheath_gamma_e: float = 3.5

    operator_split_on: bool = False
    operator_conservative_on: bool = True
    operator_source_on: bool = True
    operator_dissipative_on: bool = True


class FCIDRB3DFullState(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray
    vpar_e: jnp.ndarray
    vpar_i: jnp.ndarray
    Te: jnp.ndarray

    @classmethod
    def zeros(cls, shape: tuple[int, int, int], dtype=jnp.float64) -> "FCIDRB3DFullState":
        z = jnp.zeros(shape, dtype=dtype)
        return cls(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z)


class FCIDRB3DFullSplit(eqx.Module):
    conservative: FCIDRB3DFullState
    source: FCIDRB3DFullState
    dissipative: FCIDRB3DFullState

    def total(self) -> FCIDRB3DFullState:
        return _state_add(_state_add(self.conservative, self.source), self.dissipative)


def _state_add(a: FCIDRB3DFullState, b: FCIDRB3DFullState) -> FCIDRB3DFullState:
    return FCIDRB3DFullState(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        vpar_e=a.vpar_e + b.vpar_e,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
    )


def _state_zeros_like(y: FCIDRB3DFullState) -> FCIDRB3DFullState:
    z = jnp.zeros_like(y.n)
    return FCIDRB3DFullState(n=z, omega=z, vpar_e=z, vpar_i=z, Te=z)


class FCIDRB3DFullModel(eqx.Module):
    """3D conservative DRB milestone model on FCI planes with sheath budgets."""

    params: FCIDRB3DFullParams
    grid: FCISlabGrid

    @property
    def _kx(self) -> jnp.ndarray:
        kx_1d = jnp.asarray(2.0 * jnp.pi * jnp.fft.fftfreq(self.grid.nx, d=self.grid.dx))
        ky_1d = jnp.asarray(2.0 * jnp.pi * jnp.fft.fftfreq(self.grid.ny, d=self.grid.dy))
        kx, _ = jnp.meshgrid(kx_1d, ky_1d, indexing="ij")
        return kx

    @property
    def _ky(self) -> jnp.ndarray:
        kx_1d = jnp.asarray(2.0 * jnp.pi * jnp.fft.fftfreq(self.grid.nx, d=self.grid.dx))
        ky_1d = jnp.asarray(2.0 * jnp.pi * jnp.fft.fftfreq(self.grid.ny, d=self.grid.dy))
        _, ky = jnp.meshgrid(kx_1d, ky_1d, indexing="ij")
        return ky

    @property
    def _k2(self) -> jnp.ndarray:
        return self._kx**2 + self._ky**2

    def _ddx(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.params.perp_operator == "spectral" and self.params.perp_bc.kind_x == 0:
            return jax.vmap(lambda p: ddx_spec(p, self._kx))(f)
        if self.params.perp_operator == "fv":
            return jax.vmap(lambda p: ddx_fv(p, self.grid.dx, self.params.perp_bc))(f)
        return jax.vmap(lambda p: ddx_fd(p, self.grid.dx, self.params.perp_bc))(f)

    def _ddy(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.params.perp_operator == "spectral" and self.params.perp_bc.kind_y == 0:
            return jax.vmap(lambda p: ddy_spec(p, self._ky))(f)
        if self.params.perp_operator == "fv":
            return jax.vmap(lambda p: ddy_fv(p, self.grid.dy, self.params.perp_bc))(f)
        return jax.vmap(lambda p: ddy_fd(p, self.grid.dy, self.params.perp_bc))(f)

    def _lap(self, f: jnp.ndarray) -> jnp.ndarray:
        if (
            self.params.perp_operator == "spectral"
            and self.params.perp_bc.kind_x == 0
            and self.params.perp_bc.kind_y == 0
        ):
            return lap_spec(f, self._k2)
        if self.params.perp_operator == "fv":
            return jax.vmap(lambda p: lap_fv(p, self.grid.dx, self.grid.dy, self.params.perp_bc))(f)
        return jax.vmap(lambda p: lap_fd(p, self.grid.dx, self.grid.dy, self.params.perp_bc))(f)

    def _bracket_plane(self, phi2d: jnp.ndarray, f2d: jnp.ndarray) -> jnp.ndarray:
        if self.params.bracket == "arakawa":
            return poisson_bracket_arakawa(phi2d, f2d, self.grid.dx, self.grid.dy)
        return poisson_bracket_centered(phi2d, f2d, self.grid.dx, self.grid.dy)

    def _bracket(self, phi: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self._bracket_plane)(phi, f)

    def _phi_from_omega(self, omega: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
        if self.params.boussinesq:
            if (
                self.params.perp_operator == "spectral"
                and self.params.perp_bc.kind_x == 0
                and self.params.perp_bc.kind_y == 0
            ):
                return inv_laplacian(omega, self._k2, k2_min=self.params.k2_min)

            def solve_plane(rhs):
                return inv_laplacian_cg(
                    rhs,
                    dx=self.grid.dx,
                    dy=self.grid.dy,
                    bc=self.params.perp_bc,
                    maxiter=int(self.params.poisson_maxiter),
                    tol=float(self.params.poisson_tol),
                    preconditioner=str(self.params.poisson_preconditioner),
                )

            return jax.vmap(solve_plane)(omega)

        n_eff = jnp.asarray(float(self.params.n0), dtype=omega.dtype)
        if self.params.non_boussinesq_perturbed_density_on:
            n_eff = n_eff + jnp.asarray(n)
        n_eff = jnp.maximum(n_eff, jnp.asarray(float(self.params.n0_min), dtype=omega.dtype))

        def solve_plane(rhs, nc):
            return inv_div_n_grad_cg(
                rhs,
                n_coeff=nc,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=self.params.perp_bc,
                maxiter=int(self.params.poisson_maxiter),
                tol=float(self.params.poisson_tol),
                preconditioner=str(self.params.poisson_preconditioner),
            )

        return jax.vmap(solve_plane)(omega, n_eff)

    def _dpar(self, f: jnp.ndarray, *, bc_kind: Literal["dirichlet", "neumann"]) -> jnp.ndarray:
        if (
            self.grid.open_field_line
            and self.params.use_target_aware_dpar
            and self.grid.cell_centered
        ):
            if bc_kind == "dirichlet":
                from jaxdrb.bc import BC1D

                bc = BC1D.dirichlet(left=0.0, right=0.0, nu=0.0)
            else:
                from jaxdrb.bc import BC1D

                bc = BC1D.neumann(left=0.0, right=0.0, nu=0.0)
            return parallel_derivative_target_aware_3d(
                f,
                map_fwd=self.grid.map_fwd,
                map_bwd=self.grid.map_bwd,
                open_field_line=True,
                bc=bc,
                target_scheme=self.params.target_scheme,
            )
        return parallel_derivative_centered_3d(
            f,
            map_fwd=self.grid.map_fwd,
            map_bwd=self.grid.map_bwd,
            open_field_line=self.grid.open_field_line,
        )

    def _sheath_split(self, y: FCIDRB3DFullState, phi: jnp.ndarray) -> FCIDRB3DFullState:
        if not self.params.sheath_on:
            return _state_zeros_like(y)
        mask = self.grid.sheath_mask
        sign = self.grid.sheath_sign

        dve = jnp.zeros_like(y.vpar_e)
        dvi = jnp.zeros_like(y.vpar_i)
        dn = jnp.zeros_like(y.n)
        domega = jnp.zeros_like(y.omega)
        dTe = jnp.zeros_like(y.Te)

        if self.params.sheath_nu_mom != 0.0:
            vpar_i_target = sign * 0.5 * y.Te
            vpar_e_target = sign * (0.5 * y.Te - phi)
            dvi = dvi - self.params.sheath_nu_mom * mask * (y.vpar_i - vpar_i_target)
            dve = dve - self.params.sheath_nu_mom * mask * (y.vpar_e - vpar_e_target)

        if self.params.sheath_nu_particle != 0.0:
            dn = dn - self.params.sheath_nu_particle * mask * y.n
            domega = domega - self.params.sheath_nu_particle * mask * y.omega

        if self.params.sheath_nu_energy != 0.0:
            dTe = dTe - self.params.sheath_nu_energy * self.params.sheath_gamma_e * mask * y.Te

        return FCIDRB3DFullState(n=dn, omega=domega, vpar_e=dve, vpar_i=dvi, Te=dTe)

    def rhs_decomposed(self, t: float, y: FCIDRB3DFullState) -> FCIDRB3DFullSplit:
        _ = t
        phi = self._phi_from_omega(y.omega, y.n)

        dpar_ve = self._dpar(y.vpar_e, bc_kind="dirichlet")
        dpar_vi = self._dpar(y.vpar_i, bc_kind="dirichlet")
        dpar_Te = self._dpar(y.Te, bc_kind="neumann")
        dpar_phi = self._dpar(phi, bc_kind="dirichlet")
        dpar_j = self._dpar(y.vpar_i - y.vpar_e, bc_kind="dirichlet")
        dpar_ohm = self._dpar(phi - y.n - y.Te, bc_kind="dirichlet")

        conservative = FCIDRB3DFullState(
            n=-self._bracket(phi, y.n) - dpar_ve,
            omega=-self._bracket(phi, y.omega) + dpar_j,
            vpar_e=-self._bracket(phi, y.vpar_e) + dpar_ohm,
            vpar_i=-self._bracket(phi, y.vpar_i) - dpar_phi,
            Te=-self._bracket(phi, y.Te) - (2.0 / 3.0) * dpar_ve,
        )

        curv_n = self.params.kappa * self._ddy(y.n + y.Te)
        curv_phi = self.params.kappa * self._ddy(phi)
        curv_T = (2.0 / 3.0) * self.params.kappa * self._ddy(3.5 * y.Te + y.n - phi)
        source = FCIDRB3DFullState(
            n=-self.params.omega_n * self._ddy(phi) + curv_n - curv_phi,
            omega=curv_n,
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=-self.params.omega_Te * self._ddy(phi) + curv_T,
        )

        couple = self.params.alpha * (phi - y.n)
        me = jnp.maximum(float(self.params.me_hat), 1e-8)
        dissipative = FCIDRB3DFullState(
            n=couple + self.params.Dn * self._lap(y.n),
            omega=couple + self.params.DOmega * self._lap(y.omega),
            vpar_e=-(self.params.eta_par / me) * (y.vpar_e - y.vpar_i)
            + self.params.Dvpar * self._lap(y.vpar_e)
            + self.params.chi_par * self._dpar(dpar_ve, bc_kind="dirichlet"),
            vpar_i=self.params.Dvpar * self._lap(y.vpar_i)
            + self.params.chi_par * self._dpar(dpar_vi, bc_kind="dirichlet"),
            Te=self.params.DTe * self._lap(y.Te)
            + self.params.chi_par * self._dpar(dpar_Te, bc_kind="neumann"),
        )

        if self.params.perp_bc_nu != 0.0:

            def relax(field: jnp.ndarray) -> jnp.ndarray:
                return jax.vmap(
                    lambda p: enforce_bc_relaxation(
                        p,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=self.params.perp_bc,
                        nu=float(self.params.perp_bc_nu),
                    )
                )(field)

            dissipative = _state_add(
                dissipative,
                FCIDRB3DFullState(
                    n=relax(y.n),
                    omega=relax(y.omega),
                    vpar_e=relax(y.vpar_e),
                    vpar_i=relax(y.vpar_i),
                    Te=relax(y.Te),
                ),
            )

        dissipative = _state_add(dissipative, self._sheath_split(y, phi))

        return FCIDRB3DFullSplit(
            conservative=conservative,
            source=source,
            dissipative=dissipative,
        )

    def rhs(self, t: float, y: FCIDRB3DFullState) -> FCIDRB3DFullState:
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

    def energy(self, y: FCIDRB3DFullState) -> jnp.ndarray:
        phi = self._phi_from_omega(y.omega, y.n)
        if self.params.boussinesq:
            phi_term = -phi * y.omega
        else:
            gradx = self._ddx(phi)
            grady = self._ddy(phi)
            phi_term = gradx**2 + grady**2
        return 0.5 * jnp.mean(y.n**2 + phi_term + y.vpar_e**2 + y.vpar_i**2 + 1.5 * y.Te**2)

    def particle_content(self, y: FCIDRB3DFullState) -> jnp.ndarray:
        return jnp.mean(y.n)

    def current_content(self, y: FCIDRB3DFullState) -> jnp.ndarray:
        return jnp.mean(y.vpar_i - y.vpar_e)

    def particle_rate(self, dy: FCIDRB3DFullState) -> jnp.ndarray:
        return jnp.mean(dy.n)

    def energy_rate(self, y: FCIDRB3DFullState, dy: FCIDRB3DFullState) -> jnp.ndarray:
        if self.params.boussinesq:
            phi = self._phi_from_omega(y.omega, y.n)
            dphi = self._phi_from_omega(dy.omega, dy.n)
            return jnp.mean(
                y.n * dy.n
                - 0.5 * (dphi * y.omega + phi * dy.omega)
                + y.vpar_e * dy.vpar_e
                + y.vpar_i * dy.vpar_i
                + 1.5 * y.Te * dy.Te
            )

        eps = jnp.asarray(1.0e-7, dtype=jnp.float64)
        y_plus = FCIDRB3DFullState(
            n=y.n + eps * dy.n,
            omega=y.omega + eps * dy.omega,
            vpar_e=y.vpar_e + eps * dy.vpar_e,
            vpar_i=y.vpar_i + eps * dy.vpar_i,
            Te=y.Te + eps * dy.Te,
        )
        y_minus = FCIDRB3DFullState(
            n=y.n - eps * dy.n,
            omega=y.omega - eps * dy.omega,
            vpar_e=y.vpar_e - eps * dy.vpar_e,
            vpar_i=y.vpar_i - eps * dy.vpar_i,
            Te=y.Te - eps * dy.Te,
        )
        return (self.energy(y_plus) - self.energy(y_minus)) / (2.0 * eps)

    def sheath_budget_rates(self, y: FCIDRB3DFullState) -> tuple[jnp.ndarray, jnp.ndarray]:
        phi = self._phi_from_omega(y.omega, y.n)
        dy_sh = self._sheath_split(y, phi)
        return self.particle_rate(dy_sh), self.energy_rate(y, dy_sh)
