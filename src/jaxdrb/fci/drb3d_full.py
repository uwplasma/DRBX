from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.nonlinear.neutrals import NeutralParams, rhs_neutral
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
    alpha_Te_ohm: float = 1.0
    alpha_Ti_ohm: float = 0.0

    Dn: float = 0.0
    DOmega: float = 0.0
    Dvpar: float = 0.0
    DTe: float = 0.0
    chi_par: float = 0.0
    DTi: float = 0.0
    Dpsi: float = 0.0

    hot_ion_on: bool = False
    tau_i: float = 1.0
    omega_Ti: float = 0.0

    em_on: bool = False
    beta: float = 0.0

    neutrals_on: bool = False
    neutrals: NeutralParams = eqx.field(default_factory=NeutralParams)

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
    sheath_gamma_i: float = 3.5
    sheath_delta: float = 0.0
    sheath_cos2: float = 1.0
    sheath_bc_model: Literal["simple", "loizu_linear"] = "simple"

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
    Ti: jnp.ndarray | None = None
    psi: jnp.ndarray | None = None
    N: jnp.ndarray | None = None

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int, int],
        dtype=jnp.float64,
        *,
        hot_ion: bool = False,
        em: bool = False,
        neutrals: bool = False,
    ) -> "FCIDRB3DFullState":
        z = jnp.zeros(shape, dtype=dtype)
        return cls(
            n=z,
            omega=z,
            vpar_e=z,
            vpar_i=z,
            Te=z,
            Ti=z if hot_ion else None,
            psi=z if em else None,
            N=z if neutrals else None,
        )


class FCIDRB3DFullSplit(eqx.Module):
    conservative: FCIDRB3DFullState
    source: FCIDRB3DFullState
    dissipative: FCIDRB3DFullState

    def total(self) -> FCIDRB3DFullState:
        return _state_add(_state_add(self.conservative, self.source), self.dissipative)


def _state_add(a: FCIDRB3DFullState, b: FCIDRB3DFullState) -> FCIDRB3DFullState:
    def _opt_add(x: jnp.ndarray | None, y: jnp.ndarray | None) -> jnp.ndarray | None:
        if x is None and y is None:
            return None
        if x is None:
            return y
        if y is None:
            return x
        return x + y

    return FCIDRB3DFullState(
        n=a.n + b.n,
        omega=a.omega + b.omega,
        vpar_e=a.vpar_e + b.vpar_e,
        vpar_i=a.vpar_i + b.vpar_i,
        Te=a.Te + b.Te,
        Ti=_opt_add(a.Ti, b.Ti),
        psi=_opt_add(a.psi, b.psi),
        N=_opt_add(a.N, b.N),
    )


def _state_zeros_like(y: FCIDRB3DFullState) -> FCIDRB3DFullState:
    z = jnp.zeros_like(y.n)
    return FCIDRB3DFullState(
        n=z,
        omega=z,
        vpar_e=z,
        vpar_i=z,
        Te=z,
        Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
        psi=None if y.psi is None else jnp.zeros_like(y.psi),
        N=None if y.N is None else jnp.zeros_like(y.N),
    )


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

    def _psi_from_current(self, jpar: jnp.ndarray) -> jnp.ndarray:
        rhs = -jpar
        if (
            self.params.perp_operator == "spectral"
            and self.params.perp_bc.kind_x == 0
            and self.params.perp_bc.kind_y == 0
        ):
            return inv_laplacian(rhs, self._k2, k2_min=self.params.k2_min)

        def solve_plane(rhs2d):
            return inv_laplacian_cg(
                rhs2d,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=self.params.perp_bc,
                maxiter=int(self.params.poisson_maxiter),
                tol=float(self.params.poisson_tol),
                preconditioner=str(self.params.poisson_preconditioner),
            )

        return jax.vmap(solve_plane)(rhs)

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

    def _sheath_mask_sign(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        if hasattr(self.grid, "sheath_mask") and hasattr(self.grid, "sheath_sign"):
            mask = jnp.asarray(self.grid.sheath_mask, dtype=jnp.float64)
            sign = jnp.asarray(self.grid.sheath_sign, dtype=jnp.float64)
            if mask.shape != sign.shape:
                sign = jnp.broadcast_to(sign, mask.shape)
            return mask, sign

        hit_fwd = getattr(self.grid.map_fwd, "hit", None)
        hit_bwd = getattr(self.grid.map_bwd, "hit", None)
        shape = (self.grid.nz, self.grid.nx, self.grid.ny)
        if hit_fwd is None or hit_bwd is None:
            return jnp.zeros(shape), jnp.zeros(shape)
        hf = jnp.asarray(hit_fwd, dtype=jnp.float64)
        hb = jnp.asarray(hit_bwd, dtype=jnp.float64)
        if hf.ndim == 2:
            hf = hf[None, ...]
        if hb.ndim == 2:
            hb = hb[None, ...]
        hf = jnp.broadcast_to(hf, shape)
        hb = jnp.broadcast_to(hb, shape)
        return jnp.clip(hf + hb, 0.0, 1.0), hf - hb

    def _sheath_split_simple(
        self, y: FCIDRB3DFullState, phi: jnp.ndarray, mask: jnp.ndarray, sign: jnp.ndarray
    ) -> FCIDRB3DFullState:
        dve = jnp.zeros_like(y.vpar_e)
        dvi = jnp.zeros_like(y.vpar_i)
        dn = jnp.zeros_like(y.n)
        domega = jnp.zeros_like(y.omega)
        dTe = jnp.zeros_like(y.Te)
        dTi = None if y.Ti is None else jnp.zeros_like(y.Ti)
        dpsi = None if y.psi is None else jnp.zeros_like(y.psi)

        if self.params.sheath_nu_mom != 0.0:
            hot_on = bool(self.params.hot_ion_on) and (y.Ti is not None)
            tau_i = float(self.params.tau_i) if hot_on else 0.0
            cs0 = jnp.sqrt(1.0 + tau_i)
            dcs = (
                0.5
                * (y.Te + (y.Ti if hot_on and y.Ti is not None else 0.0))
                / jnp.maximum(cs0, 1e-12)
            )
            vpar_i_target = sign * (1.0 - float(self.params.sheath_delta)) * dcs
            vpar_e_target = sign * (dcs - phi)
            dvi = dvi - self.params.sheath_nu_mom * mask * (y.vpar_i - vpar_i_target)
            dve = dve - self.params.sheath_nu_mom * mask * (y.vpar_e - vpar_e_target)

        if self.params.sheath_nu_particle != 0.0:
            dn = dn - self.params.sheath_nu_particle * mask * y.n
            domega = domega - self.params.sheath_nu_particle * mask * y.omega

        if self.params.sheath_nu_energy != 0.0:
            dTe = dTe - self.params.sheath_nu_energy * self.params.sheath_gamma_e * mask * y.Te
            if dTi is not None:
                dTi = dTi - self.params.sheath_nu_energy * self.params.sheath_gamma_i * mask * y.Ti

        if dpsi is not None and self.params.em_on:
            dj_sh = dvi - dve
            dpsi = self._psi_from_current(dj_sh)

        return FCIDRB3DFullState(
            n=dn,
            omega=domega,
            vpar_e=dve,
            vpar_i=dvi,
            Te=dTe,
            Ti=dTi,
            psi=dpsi if y.psi is not None else None,
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    def _sheath_split_loizu_linear(
        self, y: FCIDRB3DFullState, phi: jnp.ndarray, mask: jnp.ndarray, sign: jnp.ndarray
    ) -> FCIDRB3DFullState:
        nz = int(self.grid.nz)
        if nz < 5:
            return self._sheath_split_simple(y, phi, mask, sign)

        nu_m = float(self.params.sheath_nu_mom)
        nu_p = float(self.params.sheath_nu_particle)
        nu_e = float(self.params.sheath_nu_energy)

        dn = jnp.zeros_like(y.n)
        domega = jnp.zeros_like(y.omega)
        dve = jnp.zeros_like(y.vpar_e)
        dvi = jnp.zeros_like(y.vpar_i)
        dTe = jnp.zeros_like(y.Te)
        dTi = None if y.Ti is None else jnp.zeros_like(y.Ti)
        dpsi = None if y.psi is None else jnp.zeros_like(y.psi)

        hot_on = bool(self.params.hot_ion_on) and (y.Ti is not None)
        tau_i = float(self.params.tau_i) if hot_on else 0.0
        cs0 = jnp.sqrt(1.0 + tau_i)
        inv_cs0 = 1.0 / jnp.maximum(cs0, 1e-12)
        delta = float(self.params.sheath_delta)
        cos2 = jnp.maximum(float(self.params.sheath_cos2), 1e-8)

        left = 0
        right = nz - 1
        mask_l = jnp.asarray(mask[left], dtype=y.n.dtype)
        mask_r = jnp.asarray(mask[right], dtype=y.n.dtype)
        sign_l = jnp.where(sign[left] != 0.0, sign[left], -1.0)
        sign_r = jnp.where(sign[right] != 0.0, sign[right], +1.0)

        Ti_arr = y.Ti if hot_on and y.Ti is not None else jnp.zeros_like(y.Te)
        dcs_l = 0.5 * inv_cs0 * (y.Te[left] + Ti_arr[left])
        dcs_r = 0.5 * inv_cs0 * (y.Te[right] + Ti_arr[right])

        vi_bc_l = sign_l * (1.0 - delta) * dcs_l
        vi_bc_r = sign_r * (1.0 - delta) * dcs_r

        phi_bc_l = phi[1] + cs0 * (y.vpar_i[1] - vi_bc_l)
        phi_bc_r = phi[-2] + cs0 * (y.vpar_i[-2] - vi_bc_r)

        n_bc_l = y.n[1] + inv_cs0 * (y.vpar_i[1] - vi_bc_l)
        n_bc_r = y.n[-2] + inv_cs0 * (y.vpar_i[-2] - vi_bc_r)

        dcs_bc_l = 0.5 * inv_cs0 * ((y.Te[1]) + (Ti_arr[1]))
        dcs_bc_r = 0.5 * inv_cs0 * ((y.Te[-2]) + (Ti_arr[-2]))
        ve_bc_l = sign_l * (dcs_bc_l - phi_bc_l)
        ve_bc_r = sign_r * (dcs_bc_r - phi_bc_r)

        phi_target = phi
        phi_target = phi_target.at[left].set(phi_bc_l)
        phi_target = phi_target.at[right].set(phi_bc_r)
        omega_from_phi = self._lap(phi_target)
        omega_bc_l = omega_from_phi[left]
        omega_bc_r = omega_from_phi[right]

        dl = jnp.maximum(jnp.asarray(jnp.mean(jnp.diff(self.grid.l)), dtype=y.n.dtype), 1e-8)
        dl2 = dl * dl
        v2_target_l = -omega_bc_l / (cos2 * cs0)
        v2_target_r = -omega_bc_r / (cos2 * cs0)
        vi_adj_l = (2.0 * vi_bc_l + 4.0 * y.vpar_i[2] - y.vpar_i[3] - dl2 * v2_target_l) / 5.0
        vi_adj_r = (2.0 * vi_bc_r + 4.0 * y.vpar_i[-3] - y.vpar_i[-4] - dl2 * v2_target_r) / 5.0

        if nu_m != 0.0:
            dvi = dvi.at[left].add(-nu_m * mask_l * (y.vpar_i[left] - vi_bc_l))
            dvi = dvi.at[right].add(-nu_m * mask_r * (y.vpar_i[right] - vi_bc_r))
            dvi = dvi.at[1].add(-nu_m * mask_l * (y.vpar_i[1] - vi_adj_l))
            dvi = dvi.at[-2].add(-nu_m * mask_r * (y.vpar_i[-2] - vi_adj_r))

            dve = dve.at[left].add(-nu_m * mask_l * (y.vpar_e[left] - ve_bc_l))
            dve = dve.at[right].add(-nu_m * mask_r * (y.vpar_e[right] - ve_bc_r))

        if nu_p != 0.0:
            dn = dn.at[left].add(-nu_p * mask_l * (y.n[left] - n_bc_l))
            dn = dn.at[right].add(-nu_p * mask_r * (y.n[right] - n_bc_r))
            domega = domega.at[left].add(-nu_p * mask_l * (y.omega[left] - omega_bc_l))
            domega = domega.at[right].add(-nu_p * mask_r * (y.omega[right] - omega_bc_r))

        if nu_e != 0.0:
            Te_bc_l = y.Te[1]
            Te_bc_r = y.Te[-2]
            dTe = dTe.at[left].add(
                -nu_e * mask_l * (y.Te[left] - Te_bc_l)
                - nu_e * self.params.sheath_gamma_e * mask_l * y.Te[left]
            )
            dTe = dTe.at[right].add(
                -nu_e * mask_r * (y.Te[right] - Te_bc_r)
                - nu_e * self.params.sheath_gamma_e * mask_r * y.Te[right]
            )
            if dTi is not None and y.Ti is not None:
                Ti_bc_l = y.Ti[1]
                Ti_bc_r = y.Ti[-2]
                dTi = dTi.at[left].add(
                    -nu_e * mask_l * (y.Ti[left] - Ti_bc_l)
                    - nu_e * self.params.sheath_gamma_i * mask_l * y.Ti[left]
                )
                dTi = dTi.at[right].add(
                    -nu_e * mask_r * (y.Ti[right] - Ti_bc_r)
                    - nu_e * self.params.sheath_gamma_i * mask_r * y.Ti[right]
                )

        if dpsi is not None and self.params.em_on:
            dj_sh = dvi - dve
            dpsi = self._psi_from_current(dj_sh)

        return FCIDRB3DFullState(
            n=dn,
            omega=domega,
            vpar_e=dve,
            vpar_i=dvi,
            Te=dTe,
            Ti=dTi,
            psi=dpsi if y.psi is not None else None,
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

    def _sheath_split(self, y: FCIDRB3DFullState, phi: jnp.ndarray) -> FCIDRB3DFullState:
        if not self.params.sheath_on:
            return _state_zeros_like(y)
        mask, sign = self._sheath_mask_sign()
        if self.params.sheath_bc_model == "loizu_linear":
            return self._sheath_split_loizu_linear(y, phi, mask, sign)
        return self._sheath_split_simple(y, phi, mask, sign)

    def rhs_decomposed(self, t: float, y: FCIDRB3DFullState) -> FCIDRB3DFullSplit:
        _ = t
        hot_on = bool(self.params.hot_ion_on) and (y.Ti is not None)
        em_on = bool(self.params.em_on) and (y.psi is not None)
        neut_on = (
            bool(self.params.neutrals_on) and (y.N is not None) and self.params.neutrals.enabled
        )

        Ti = y.Ti if hot_on and y.Ti is not None else jnp.zeros_like(y.Te)
        psi = y.psi if em_on and y.psi is not None else jnp.zeros_like(y.Te)
        phi = self._phi_from_omega(y.omega, y.n)

        dpar_ve = self._dpar(y.vpar_e, bc_kind="dirichlet")
        dpar_vi = self._dpar(y.vpar_i, bc_kind="dirichlet")
        dpar_Te = self._dpar(y.Te, bc_kind="neumann")
        dpar_Ti = self._dpar(Ti, bc_kind="neumann") if hot_on else jnp.zeros_like(Ti)
        jpar_fluid = y.vpar_i - y.vpar_e
        jpar_em = -self._lap(psi) if em_on else jnp.zeros_like(jpar_fluid)
        jpar_total = jpar_fluid + jpar_em
        dpar_j = self._dpar(jpar_total, bc_kind="dirichlet")
        dpar_ohm = self._dpar(
            phi
            - y.n
            - float(self.params.alpha_Te_ohm) * y.Te
            - float(self.params.alpha_Ti_ohm) * Ti,
            bc_kind="dirichlet",
        )
        dpar_psi = self._dpar(psi, bc_kind="dirichlet") if em_on else jnp.zeros_like(psi)

        tau_i = float(self.params.tau_i) if hot_on else 0.0
        vi_par_pressure = phi + tau_i * (y.n + Ti)
        Ti_conservative = -self._bracket(phi, Ti) - (2.0 / 3.0) * dpar_vi
        psi_conservative = -self._bracket(phi, psi) - dpar_ohm

        conservative = FCIDRB3DFullState(
            n=-self._bracket(phi, y.n) - dpar_ve,
            omega=-self._bracket(phi, y.omega) + dpar_j,
            vpar_e=-self._bracket(phi, y.vpar_e) + dpar_ohm - dpar_psi,
            vpar_i=-self._bracket(phi, y.vpar_i) - self._dpar(vi_par_pressure, bc_kind="dirichlet"),
            Te=-self._bracket(phi, y.Te) - (2.0 / 3.0) * dpar_ve,
            Ti=(Ti_conservative if hot_on else jnp.zeros_like(y.Ti)) if y.Ti is not None else None,
            psi=(
                (psi_conservative if em_on else jnp.zeros_like(y.psi))
                if y.psi is not None
                else None
            ),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

        p_tot = (1.0 + tau_i) * y.n + y.Te + tau_i * Ti
        curv_n = self.params.kappa * self._ddy(p_tot)
        curv_phi = self.params.kappa * self._ddy(phi)
        curv_T = (2.0 / 3.0) * self.params.kappa * self._ddy(3.5 * y.Te + y.n - phi)
        source = FCIDRB3DFullState(
            n=-self.params.omega_n * self._ddy(phi) + curv_n - curv_phi,
            omega=curv_n,
            vpar_e=jnp.zeros_like(y.vpar_e),
            vpar_i=jnp.zeros_like(y.vpar_i),
            Te=-self.params.omega_Te * self._ddy(phi) + curv_T,
            Ti=(
                (-float(self.params.omega_Ti) * self._ddy(phi) if hot_on else jnp.zeros_like(y.Ti))
                if y.Ti is not None
                else None
            ),
            psi=None if y.psi is None else jnp.zeros_like(y.psi),
            N=None if y.N is None else jnp.zeros_like(y.N),
        )

        couple = self.params.alpha * (phi - y.n)
        me = jnp.maximum(float(self.params.me_hat), 1e-8)
        Ti_diss = (
            self.params.DTi * self._lap(Ti)
            + self.params.chi_par * self._dpar(dpar_Ti, bc_kind="neumann")
            if hot_on
            else None
        )
        psi_diss = (
            -self.params.eta_par * jpar_total
            + self.params.Dpsi * self._lap(psi)
            + self.params.chi_par * self._dpar(dpar_psi, bc_kind="dirichlet")
            if em_on
            else None
        )
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
            Ti=(Ti_diss if hot_on else jnp.zeros_like(y.Ti)) if y.Ti is not None else None,
            psi=(psi_diss if em_on else jnp.zeros_like(y.psi)) if y.psi is not None else None,
            N=None if y.N is None else jnp.zeros_like(y.N),
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
                    Ti=None if y.Ti is None else relax(y.Ti),
                    psi=None if y.psi is None else relax(y.psi),
                    N=None if y.N is None else relax(y.N),
                ),
            )

        if neut_on and y.N is not None:
            lap_N = self._lap(y.N)
            adv_N = -self._bracket(phi, y.N)
            dN, dn_neut, dw_neut = rhs_neutral(
                N=y.N,
                n=y.n,
                omega=y.omega,
                dn0=self.params.neutrals,
                adv_N=adv_N,
                lap_N=lap_N,
            )
            source = _state_add(
                source,
                FCIDRB3DFullState(
                    n=dn_neut,
                    omega=dw_neut,
                    vpar_e=jnp.zeros_like(y.vpar_e),
                    vpar_i=jnp.zeros_like(y.vpar_i),
                    Te=jnp.zeros_like(y.Te),
                    Ti=None if y.Ti is None else jnp.zeros_like(y.Ti),
                    psi=None if y.psi is None else jnp.zeros_like(y.psi),
                    N=dN,
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
        E = 0.5 * jnp.mean(y.n**2 + phi_term + y.vpar_e**2 + y.vpar_i**2 + 1.5 * y.Te**2)
        if y.Ti is not None and self.params.hot_ion_on:
            E = E + 0.5 * jnp.mean(1.5 * float(self.params.tau_i) * y.Ti**2)
        if y.psi is not None and self.params.em_on:
            gradx_psi = self._ddx(y.psi)
            grady_psi = self._ddy(y.psi)
            E = E + 0.5 * jnp.mean(0.5 * float(self.params.beta) * (gradx_psi**2 + grady_psi**2))
        return E

    def particle_content(self, y: FCIDRB3DFullState) -> jnp.ndarray:
        return jnp.mean(y.n)

    def total_particle_content(self, y: FCIDRB3DFullState) -> jnp.ndarray:
        if y.N is None:
            return jnp.mean(y.n)
        return jnp.mean(y.n + y.N)

    def current_content(self, y: FCIDRB3DFullState) -> jnp.ndarray:
        if y.psi is not None and self.params.em_on:
            return jnp.mean(y.vpar_i - y.vpar_e - self._lap(y.psi))
        return jnp.mean(y.vpar_i - y.vpar_e)

    def particle_rate(self, dy: FCIDRB3DFullState) -> jnp.ndarray:
        return jnp.mean(dy.n)

    def total_particle_rate(self, dy: FCIDRB3DFullState) -> jnp.ndarray:
        if dy.N is None:
            return self.particle_rate(dy)
        return jnp.mean(dy.n + dy.N)

    def advective_particle_rate(self, y: FCIDRB3DFullState) -> jnp.ndarray:
        """Mean particle-content rate from the ExB advection term ``-[phi, n]``."""
        phi = self._phi_from_omega(y.omega, y.n)
        return jnp.mean(-self._bracket(phi, y.n))

    def parallel_particle_rate(self, y: FCIDRB3DFullState) -> jnp.ndarray:
        """Mean particle-content rate from parallel compression ``-∂|| v_{||e}``."""
        dpar_ve = self._dpar(y.vpar_e, bc_kind="dirichlet")
        return jnp.mean(-dpar_ve)

    def particle_budget_terms(self, y: FCIDRB3DFullState) -> dict[str, jnp.ndarray]:
        """Particle budget decomposition for diagnostics/gates on open-field-line runs."""
        split = self.rhs_decomposed(0.0, y)
        phi = self._phi_from_omega(y.omega, y.n)
        dy_sh = self._sheath_split(y, phi)
        total = self.total_particle_rate(split.total())
        advective = self.advective_particle_rate(y)
        parallel = self.parallel_particle_rate(y)
        sheath = self.total_particle_rate(dy_sh)
        neutral = (
            jnp.mean(split.source.n + split.source.N)
            if split.source.N is not None
            else jnp.mean(split.source.n)
        )
        other = total - (advective + parallel + neutral + sheath)
        return {
            "total": total,
            "advective": advective,
            "parallel": parallel,
            "neutral": neutral,
            "sheath": sheath,
            "other": other,
        }

    def energy_rate(self, y: FCIDRB3DFullState, dy: FCIDRB3DFullState) -> jnp.ndarray:
        if self.params.boussinesq:
            phi = self._phi_from_omega(y.omega, y.n)
            dphi = self._phi_from_omega(dy.omega, dy.n)
            out = (
                y.n * dy.n
                - 0.5 * (dphi * y.omega + phi * dy.omega)
                + y.vpar_e * dy.vpar_e
                + y.vpar_i * dy.vpar_i
                + 1.5 * y.Te * dy.Te
            )
            if y.Ti is not None and dy.Ti is not None and self.params.hot_ion_on:
                out = out + 1.5 * float(self.params.tau_i) * y.Ti * dy.Ti
            if y.psi is not None and dy.psi is not None and self.params.em_on:
                gradx_psi = self._ddx(y.psi)
                grady_psi = self._ddy(y.psi)
                dgradx_psi = self._ddx(dy.psi)
                dgrady_psi = self._ddy(dy.psi)
                out = out + 0.5 * float(self.params.beta) * (
                    gradx_psi * dgradx_psi + grady_psi * dgrady_psi
                )
            return jnp.mean(out)

        eps = jnp.asarray(1.0e-7, dtype=jnp.float64)
        y_plus = FCIDRB3DFullState(
            n=y.n + eps * dy.n,
            omega=y.omega + eps * dy.omega,
            vpar_e=y.vpar_e + eps * dy.vpar_e,
            vpar_i=y.vpar_i + eps * dy.vpar_i,
            Te=y.Te + eps * dy.Te,
            Ti=None if y.Ti is None or dy.Ti is None else y.Ti + eps * dy.Ti,
            psi=None if y.psi is None or dy.psi is None else y.psi + eps * dy.psi,
            N=None if y.N is None or dy.N is None else y.N + eps * dy.N,
        )
        y_minus = FCIDRB3DFullState(
            n=y.n - eps * dy.n,
            omega=y.omega - eps * dy.omega,
            vpar_e=y.vpar_e - eps * dy.vpar_e,
            vpar_i=y.vpar_i - eps * dy.vpar_i,
            Te=y.Te - eps * dy.Te,
            Ti=None if y.Ti is None or dy.Ti is None else y.Ti - eps * dy.Ti,
            psi=None if y.psi is None or dy.psi is None else y.psi - eps * dy.psi,
            N=None if y.N is None or dy.N is None else y.N - eps * dy.N,
        )
        return (self.energy(y_plus) - self.energy(y_minus)) / (2.0 * eps)

    def energy_budget_terms(self, y: FCIDRB3DFullState) -> dict[str, jnp.ndarray]:
        """Energy-rate decomposition from the split RHS for diagnostics/gates."""
        split = self.rhs_decomposed(0.0, y)
        phi = self._phi_from_omega(y.omega, y.n)
        dy_sh = self._sheath_split(y, phi)
        total = self.energy_rate(y, split.total())
        conservative = self.energy_rate(y, split.conservative)
        source = self.energy_rate(y, split.source)
        dissipative = self.energy_rate(y, split.dissipative)
        sheath = self.energy_rate(y, dy_sh)
        dissipative_other = dissipative - sheath
        residual = total - (conservative + source + dissipative_other + sheath)
        return {
            "total": total,
            "conservative": conservative,
            "source": source,
            "dissipative_other": dissipative_other,
            "sheath": sheath,
            "residual": residual,
        }

    def sheath_budget_rates(self, y: FCIDRB3DFullState) -> tuple[jnp.ndarray, jnp.ndarray]:
        phi = self._phi_from_omega(y.omega, y.n)
        dy_sh = self._sheath_split(y, phi)
        return self.particle_rate(dy_sh), self.energy_rate(y, dy_sh)
