from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.spectral import ddx as ddx_spec
from jaxdrb.nonlinear.spectral import ddy as ddy_spec
from jaxdrb.nonlinear.spectral import inv_laplacian, laplacian
from jaxdrb.operators.brackets import poisson_bracket_arakawa, poisson_bracket_centered

from .grid import FCISlabGrid
from .parallel import parallel_derivative_centered_3d


class FCIDRB3DParams(eqx.Module):
    """Minimal 3D DRB-like slab operator for FCI validation."""

    kappa: float = 0.0
    alpha: float = 0.0
    kpar: float = 0.0

    Dn: float = 0.0
    DOmega: float = 0.0

    bracket: Literal["arakawa", "centered"] = "arakawa"
    poisson: Literal["spectral", "fd_cg"] = "spectral"
    dealias_on: bool = False
    k2_min: float = 1e-12

    sheath_nu: float = 0.0


class FCIDRB3DState(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray


class FCIDRB3DModel(eqx.Module):
    params: FCIDRB3DParams
    grid: FCISlabGrid

    def _phi_from_omega(self, omega: jnp.ndarray) -> jnp.ndarray:
        if self.params.poisson != "spectral":
            raise NotImplementedError("FCI DRB3D currently supports spectral Poisson only.")
        k2 = self._k2
        return inv_laplacian(omega, k2, k2_min=self.params.k2_min)

    @property
    def _kx(self) -> jnp.ndarray:
        kx = 2.0 * jnp.pi * jnp.fft.fftfreq(self.grid.nx, d=self.grid.dx)
        return jnp.asarray(kx)

    @property
    def _ky(self) -> jnp.ndarray:
        ky = 2.0 * jnp.pi * jnp.fft.fftfreq(self.grid.ny, d=self.grid.dy)
        return jnp.asarray(ky)

    @property
    def _k2(self) -> jnp.ndarray:
        kx = self._kx
        ky = self._ky
        kx2, ky2 = jnp.meshgrid(kx, ky, indexing="ij")
        return kx2**2 + ky2**2

    def _bracket(self, phi: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        if self.params.bracket == "arakawa":
            return poisson_bracket_arakawa(phi, f, self.grid.dx, self.grid.dy)
        return poisson_bracket_centered(phi, f, self.grid.dx, self.grid.dy)

    def rhs(self, t: float, y: FCIDRB3DState) -> FCIDRB3DState:
        _ = t
        if self.params.poisson != "spectral":
            raise NotImplementedError("FCI DRB3D currently supports spectral Poisson only.")

        n = y.n
        omega = y.omega

        phi = self._phi_from_omega(omega)

        def plane_bracket(plane_phi, plane_f):
            return self._bracket(plane_phi, plane_f)

        adv_n = jax.vmap(plane_bracket, in_axes=(0, 0))(phi, n)
        adv_w = jax.vmap(plane_bracket, in_axes=(0, 0))(phi, omega)

        dphi_dy = ddy_spec(phi, self._ky)
        dn_dy = ddy_spec(n, self._ky)
        lap_n = laplacian(n, self._k2)
        lap_w = laplacian(omega, self._k2)

        drive_n = -self.params.kappa * dphi_dy
        drive_w = -self.params.kappa * dn_dy

        couple = self.params.alpha * (phi - n)

        dn = -adv_n + drive_n + couple + self.params.Dn * lap_n
        dw = -adv_w + drive_w + couple + self.params.DOmega * lap_w

        if self.params.kpar != 0.0:
            dpar_n = parallel_derivative_centered_3d(
                n,
                map_fwd=self.grid.map_fwd,
                map_bwd=self.grid.map_bwd,
                open_field_line=self.grid.open_field_line,
            )
            dpar_w = parallel_derivative_centered_3d(
                omega,
                map_fwd=self.grid.map_fwd,
                map_bwd=self.grid.map_bwd,
                open_field_line=self.grid.open_field_line,
            )
            dn = dn + self.params.kpar * dpar_n
            dw = dw + self.params.kpar * dpar_w

        if self.params.sheath_nu != 0.0:
            dn = dn - self.params.sheath_nu * self.grid.sheath_mask * n
            dw = dw - self.params.sheath_nu * self.grid.sheath_mask * omega

        return FCIDRB3DState(n=dn, omega=dw)

    def energy(self, y: FCIDRB3DState) -> jnp.ndarray:
        phi = self._phi_from_omega(y.omega)
        gradx = ddx_spec(phi, self._kx)
        grady = ddy_spec(phi, self._ky)
        return 0.5 * jnp.mean(y.n**2 + gradx**2 + grady**2)

    def energy_rate(self, y: FCIDRB3DState, dy: FCIDRB3DState) -> jnp.ndarray:
        phi = self._phi_from_omega(y.omega)
        dphi = self._phi_from_omega(dy.omega)
        gradx = ddx_spec(phi, self._kx)
        grady = ddy_spec(phi, self._ky)
        dgradx = ddx_spec(dphi, self._kx)
        dgrady = ddy_spec(dphi, self._ky)
        return jnp.mean(y.n * dy.n + gradx * dgradx + grady * dgrady)

    def mass_rate(self, dy: FCIDRB3DState) -> jnp.ndarray:
        return jnp.mean(dy.n)
