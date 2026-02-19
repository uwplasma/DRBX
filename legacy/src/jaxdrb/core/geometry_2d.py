from __future__ import annotations

from typing import ClassVar, Literal

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.geometry import GeometryBase
from jaxdrb.core.operators import build_perp_operator_bundle, PerpOperatorBundle
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.nonlinear.fd import (
    ddx as ddx_fd,
    ddy as ddy_fd,
    enforce_bc_relaxation,
    inv_div_n_grad_cg,
    inv_laplacian_cg,
    inv_laplacian_fd_fft,
    inv_laplacian_mixed_fft,
)
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.spectral import (
    ddx as ddx_spec,
    ddy as ddy_spec,
    dealias,
    inv_laplacian,
    poisson_bracket_spectral,
)
from jaxdrb.operators.brackets import (
    poisson_bracket_arakawa,
    poisson_bracket_arakawa_fd,
    poisson_bracket_centered,
)


class Geometry2DAdapter(GeometryBase):
    """Geometry adapter for 2D perpendicular grids."""

    grid: Grid2D
    params: DRBSystemParams
    name: ClassVar[str] = "2d"
    ndim: ClassVar[int] = 2
    perp_ops: PerpOperatorBundle = eqx.field(init=False)
    _scheme: Literal["spectral", "fd"] = eqx.field(init=False)

    def __post_init__(self):
        scheme = self.params.perp_operator
        if scheme == "spectral" and not self._is_periodic(self.grid.bc):
            scheme = "fd"
        ops = build_perp_operator_bundle(
            scheme=scheme,
            bracket=self.params.bracket,
            nx=self.grid.nx,
            ny=self.grid.ny,
            dx=self.grid.dx,
            dy=self.grid.dy,
            dealias_on=self.params.dealias_on,
            bracket_zero_mean=self.params.bracket_zero_mean,
            bc=self.grid.bc,
        )
        object.__setattr__(self, "perp_ops", ops)
        object.__setattr__(self, "_scheme", scheme)

    def shape(self) -> tuple[int, int]:
        return int(self.grid.nx), int(self.grid.ny)

    def _is_periodic(self, bc: BC2D) -> bool:
        return bc.kind_x == 0 and bc.kind_y == 0

    def ddx(self, f: jnp.ndarray) -> jnp.ndarray:
        return self.perp_ops.ddx(f)

    def ddy(self, f: jnp.ndarray) -> jnp.ndarray:
        return self.perp_ops.ddy(f)

    def laplacian(self, f: jnp.ndarray) -> jnp.ndarray:
        return self.perp_ops.laplacian(f)

    def biharmonic(self, f: jnp.ndarray) -> jnp.ndarray:
        return self.perp_ops.biharmonic(f)

    def inv_laplacian(self, f: jnp.ndarray) -> jnp.ndarray:
        bc_phi = self.grid.bc
        if self.params.poisson == "spectral":
            if not self._is_periodic(bc_phi):
                raise ValueError("Spectral Poisson solve requires periodic BCs in x and y.")
            return inv_laplacian(f, self.perp_ops.k2, k2_min=self.params.k2_min)
        if self.params.poisson == "mixed_fft":
            return inv_laplacian_mixed_fft(
                f,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=bc_phi,
                gauge_epsilon=self.params.poisson_gauge_epsilon,
            )
        precond = self.params.poisson_preconditioner
        if precond == "auto":
            precond = "spectral"
        if precond == "spectral" and not self._is_periodic(bc_phi):
            precond = "jacobi"
        if self.params.poisson == "cg_fd":
            try:
                return inv_laplacian_fd_fft(
                    f,
                    dx=self.grid.dx,
                    dy=self.grid.dy,
                    bc=bc_phi,
                    gauge_epsilon=self.params.poisson_gauge_epsilon,
                )
            except ValueError:
                pass
        return inv_laplacian_cg(
            f,
            dx=self.grid.dx,
            dy=self.grid.dy,
            bc=bc_phi,
            maxiter=int(self.params.poisson_cg_maxiter),
            tol=float(self.params.poisson_cg_tol),
            atol=float(self.params.poisson_cg_atol),
            preconditioner=str(precond),
            k2_precond=self.perp_ops.k2 if str(precond) == "spectral" else None,
            gauge_epsilon=self.params.poisson_gauge_epsilon,
        )

    def inv_div_n_grad(self, n_eff: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        bc_phi = self.grid.bc
        precond = self.params.polarization_preconditioner
        if precond == "auto":
            precond = "spectral_jacobi"
        return inv_div_n_grad_cg(
            f,
            n_coeff=n_eff,
            dx=self.grid.dx,
            dy=self.grid.dy,
            bc=bc_phi,
            maxiter=int(self.params.polarization_cg_maxiter),
            tol=float(self.params.polarization_cg_tol),
            atol=float(self.params.polarization_cg_atol),
            preconditioner=precond,
            preconditioner_shift=float(self.params.polarization_precond_shift),
        )

    def bracket(
        self,
        phi: jnp.ndarray,
        f: jnp.ndarray,
        *,
        bc_phi: BC2D | None = None,
        bc_f: BC2D | None = None,
    ) -> jnp.ndarray:
        if bc_phi is None:
            bc_phi = self.grid.bc
        if bc_f is None:
            bc_f = self.grid.bc

        periodic_pair = self._is_periodic(bc_phi) and self._is_periodic(bc_f)
        scale = float(self.params.exb_scale)

        if self.params.bracket == "spectral":
            if not periodic_pair:
                raise ValueError("Spectral bracket requires periodic BCs in x and y.")
            return poisson_bracket_spectral(
                phi,
                f,
                kx=self.perp_ops.kx,
                ky=self.perp_ops.ky,
                dealias_mask=self.grid.dealias_mask if self.params.dealias_on else None,
            )

        if self.params.bracket == "arakawa":
            if periodic_pair:
                j = poisson_bracket_arakawa(phi, f, self.grid.dx, self.grid.dy)
            else:
                j = poisson_bracket_arakawa_fd(phi, f, self.grid.dx, self.grid.dy, bc_phi, bc_f)
                if self.params.bracket_zero_mean:
                    j = j - jnp.mean(j)
            if self.params.dealias_on and periodic_pair:
                return scale * dealias(j, self.grid.dealias_mask)
            return scale * j

        if periodic_pair:
            j = poisson_bracket_centered(phi, f, self.grid.dx, self.grid.dy)
        else:
            dphi_dx = ddx_fd(phi, self.grid.dx, bc_phi)
            dphi_dy = ddy_fd(phi, self.grid.dy, bc_phi)
            df_dx = ddx_fd(f, self.grid.dx, bc_f)
            df_dy = ddy_fd(f, self.grid.dy, bc_f)
            j = dphi_dx * df_dy - dphi_dy * df_dx
        if self.params.bracket_zero_mean and not periodic_pair:
            j = j - jnp.mean(j)
        if self.params.dealias_on and periodic_pair:
            return scale * dealias(j, self.grid.dealias_mask)
        return scale * j

    def dpar(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray:
        if self.params.kpar == 0.0:
            return jnp.zeros_like(f)
        return 1j * float(self.params.kpar) * f

    def d2par(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray:
        return self.dpar(self.dpar(f, bc_kind=bc_kind), bc_kind=bc_kind)

    def curvature(self, f: jnp.ndarray) -> jnp.ndarray:
        if not self.params.curvature_on or self.params.curvature_coeff == 0.0:
            return jnp.zeros_like(f)
        coeff = float(self.params.curvature_coeff)
        model = str(self.params.curvature_model).lower()
        bc = self.grid.bc
        if model in ("tokamak", "salpha", "sin", "sinusoidal"):
            theta_scale = self.params.curvature_theta_scale
            if theta_scale is None or float(theta_scale) <= 0.0:
                theta_scale = float(self.grid.Ly) / (2.0 * jnp.pi)
            theta = self.grid.y[None, :] / float(theta_scale)
            df_dx = ddx_fd(f, self.grid.dx, bc)
            df_dy = ddy_fd(f, self.grid.dy, bc)
            curv = jnp.sin(theta) * df_dx + jnp.cos(theta) * df_dy
        else:
            if self._is_periodic(bc) and self.params.poisson == "spectral":
                df_dy = ddy_spec(f, self.perp_ops.ky)
            else:
                df_dy = ddy_fd(f, self.grid.dy, bc)
            curv = df_dy
        curv = -coeff * curv
        scale = self.params.curvature_scale
        if scale is None:
            scale = float(self.params.exb_scale)
        if float(scale) != 1.0:
            curv = curv * float(scale)
        return curv

    def apply_bc(self, f: jnp.ndarray, bc: BC2D) -> jnp.ndarray:
        if float(self.params.bc_enforce_nu) == 0.0:
            return f
        return enforce_bc_relaxation(
            f,
            dx=self.grid.dx,
            dy=self.grid.dy,
            bc=bc,
            nu=float(self.params.bc_enforce_nu),
        )

    def enforce_bc_relaxation(self, f: jnp.ndarray, *, nu: float) -> jnp.ndarray:
        return enforce_bc_relaxation(
            f,
            dx=self.grid.dx,
            dy=self.grid.dy,
            bc=self.grid.bc,
            nu=float(nu),
        )
