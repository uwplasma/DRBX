from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.nonlinear.fd import ddx as ddx_fd
from jaxdrb.nonlinear.fd import ddy as ddy_fd
from jaxdrb.nonlinear.fd import inv_div_n_grad_cg, inv_laplacian_cg
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
    boussinesq: bool = True
    non_boussinesq_perturbed_density_on: bool = True
    n0: float = 1.0
    n0_min: float = 1e-6
    poisson_preconditioner: Literal["spectral", "jacobi", "none"] = "spectral"
    poisson_maxiter: int = 400
    poisson_tol: float = 1e-10
    dealias_on: bool = False
    k2_min: float = 1e-12

    sheath_nu: float = 0.0


class FCIDRB3DState(eqx.Module):
    n: jnp.ndarray
    omega: jnp.ndarray


class FCIDRB3DModel(eqx.Module):
    params: FCIDRB3DParams
    grid: FCISlabGrid

    @property
    def _bc_perp(self) -> BC2D:
        return BC2D.periodic()

    def _phi_from_omega(self, omega: jnp.ndarray, *, n: jnp.ndarray | None = None) -> jnp.ndarray:
        if self.params.boussinesq:
            if self.params.poisson == "spectral":
                k2 = self._k2
                return inv_laplacian(omega, k2, k2_min=self.params.k2_min)

            def solve_plane(rhs):
                return inv_laplacian_cg(
                    rhs,
                    dx=self.grid.dx,
                    dy=self.grid.dy,
                    bc=self._bc_perp,
                    maxiter=int(self.params.poisson_maxiter),
                    tol=float(self.params.poisson_tol),
                    preconditioner=str(self.params.poisson_preconditioner),
                )

            return jax.vmap(solve_plane)(omega)

        if n is None:
            raise ValueError("Non-Boussinesq polarization requires density n.")
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
                bc=self._bc_perp,
                maxiter=int(self.params.poisson_maxiter),
                tol=float(self.params.poisson_tol),
                preconditioner=str(self.params.poisson_preconditioner),
            )

        return jax.vmap(solve_plane)(omega, n_eff)

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

    def _bracket(self, phi: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        if self.params.bracket == "arakawa":
            return poisson_bracket_arakawa(phi, f, self.grid.dx, self.grid.dy)
        return poisson_bracket_centered(phi, f, self.grid.dx, self.grid.dy)

    def rhs(self, t: float, y: FCIDRB3DState) -> FCIDRB3DState:
        _ = t

        n = y.n
        omega = y.omega

        phi = self._phi_from_omega(omega, n=n)

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
        phi = self._phi_from_omega(y.omega, n=y.n)
        if self.params.poisson == "spectral":
            gradx = ddx_spec(phi, self._kx)
            grady = ddy_spec(phi, self._ky)
        else:
            gradx = jax.vmap(lambda p: ddx_fd(p, self.grid.dx, self._bc_perp))(phi)
            grady = jax.vmap(lambda p: ddy_fd(p, self.grid.dy, self._bc_perp))(phi)

        if self.params.boussinesq:
            phi_term = gradx**2 + grady**2
        else:
            n_eff = jnp.asarray(float(self.params.n0), dtype=y.n.dtype)
            if self.params.non_boussinesq_perturbed_density_on:
                n_eff = n_eff + jnp.asarray(y.n)
            n_eff = jnp.maximum(n_eff, jnp.asarray(float(self.params.n0_min), dtype=y.n.dtype))
            phi_term = n_eff * (gradx**2 + grady**2)
        return 0.5 * jnp.mean(y.n**2 + phi_term)

    def energy_rate(self, y: FCIDRB3DState, dy: FCIDRB3DState) -> jnp.ndarray:
        if not self.params.boussinesq:
            eps = jnp.asarray(1.0e-7, dtype=jnp.float64)
            y_plus = FCIDRB3DState(n=y.n + eps * dy.n, omega=y.omega + eps * dy.omega)
            y_minus = FCIDRB3DState(n=y.n - eps * dy.n, omega=y.omega - eps * dy.omega)
            return (self.energy(y_plus) - self.energy(y_minus)) / (2.0 * eps)

        phi = self._phi_from_omega(y.omega, n=y.n)
        dphi = self._phi_from_omega(dy.omega, n=dy.n)
        if self.params.poisson == "spectral":
            gradx = ddx_spec(phi, self._kx)
            grady = ddy_spec(phi, self._ky)
            dgradx = ddx_spec(dphi, self._kx)
            dgrady = ddy_spec(dphi, self._ky)
        else:
            gradx = jax.vmap(lambda p: ddx_fd(p, self.grid.dx, self._bc_perp))(phi)
            grady = jax.vmap(lambda p: ddy_fd(p, self.grid.dy, self._bc_perp))(phi)
            dgradx = jax.vmap(lambda p: ddx_fd(p, self.grid.dx, self._bc_perp))(dphi)
            dgrady = jax.vmap(lambda p: ddy_fd(p, self.grid.dy, self._bc_perp))(dphi)
        return jnp.mean(y.n * dy.n + gradx * dgradx + grady * dgrady)

    def mass_rate(self, dy: FCIDRB3DState) -> jnp.ndarray:
        return jnp.mean(dy.n)
