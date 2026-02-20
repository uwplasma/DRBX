from __future__ import annotations

from typing import ClassVar, Literal

import equinox as eqx
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.geometry import GeometryBase
from jaxdrb.core.operators import build_perp_operator_bundle, PerpOperatorBundle
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.operators.fd2d import (
    ddx as ddx_fd,
    ddy as ddy_fd,
    enforce_bc_relaxation,
    build_fd_fft_eigs,
    build_div_n_grad_preconditioner,
    build_laplacian_preconditioner,
    inv_div_n_grad_cg,
    inv_laplacian_cg,
    inv_laplacian_fd_fft,
    inv_laplacian_mixed_fft,
)
from jaxdrb.geometry.plane import Grid2D
from jaxdrb.operators.spectral2d import (
    ddx as ddx_spec,
    ddy as ddy_spec,
    dealias,
    inv_laplacian,
    poisson_bracket_spectral,
    poisson_bracket_spectral_multi,
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
    poisson_preconditioner_fn: object = eqx.field(init=False, static=True, default=None)
    polarization_preconditioner_fn: object = eqx.field(init=False, static=True, default=None)
    poisson_fd_fft_eigs: tuple[jnp.ndarray, jnp.ndarray] | None = eqx.field(
        init=False, default=None
    )

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

        bc = self.grid.bc
        precond = self.params.poisson_preconditioner
        if precond == "auto":
            precond = "spectral" if self._is_periodic(bc) else "fd_fft"
        if precond == "spectral" and not self._is_periodic(bc):
            precond = "fd_fft"
        shape = (self.grid.nx, self.grid.ny)
        if bc.kind_x == 1 and bc.kind_y == 1:
            shape = (self.grid.nx - 2, self.grid.ny - 2)
        poisson_precond = build_laplacian_preconditioner(
            shape=shape,
            dx=self.grid.dx,
            dy=self.grid.dy,
            bc=self.grid.bc,
            preconditioner=str(precond),
            k2_precond=self.perp_ops.k2 if str(precond) == "spectral" else None,
            gauge_epsilon=self.params.poisson_gauge_epsilon,
        )
        object.__setattr__(self, "poisson_preconditioner_fn", poisson_precond)

        pol_precond = None
        if not bool(self.params.non_boussinesq_perturbed_density_on):
            pol = self.params.polarization_preconditioner
            if pol == "auto":
                pol = "spectral_jacobi"
            n_eff = max(float(self.params.n0), float(self.params.n0_min))
            if bc.kind_x == 1 and bc.kind_y == 1:
                n_coeff = jnp.full((self.grid.nx - 2, self.grid.ny - 2), n_eff)
            else:
                n_coeff = jnp.full((self.grid.nx, self.grid.ny), n_eff)
            pol_precond = build_div_n_grad_preconditioner(
                n_coeff=n_coeff,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=bc,
                preconditioner=str(pol),
                preconditioner_shift=float(self.params.polarization_precond_shift),
                n_floor=float(self.params.n0_min),
            )
        object.__setattr__(self, "polarization_preconditioner_fn", pol_precond)

        try:
            eigs = build_fd_fft_eigs(
                nx=self.grid.nx,
                ny=self.grid.ny,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=bc,
                dtype=jnp.float64,
            )
        except Exception:
            eigs = None
        object.__setattr__(self, "poisson_fd_fft_eigs", eigs)

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

    def inv_laplacian(self, f: jnp.ndarray, *, x0: jnp.ndarray | None = None) -> jnp.ndarray:
        bc_phi = self.grid.bc
        poisson = self.params.poisson
        if (
            self.params.poisson_force_fd_fft_when_nonperiodic
            and poisson == "spectral"
            and not self._is_periodic(bc_phi)
        ):
            poisson = "cg_fd"
        if poisson == "spectral":
            if not self._is_periodic(bc_phi):
                raise ValueError("Spectral Poisson solve requires periodic BCs in x and y.")
            return inv_laplacian(f, self.perp_ops.k2, k2_min=self.params.k2_min)
        if poisson == "mixed_fft":
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
        if poisson == "cg_fd":
            try:
                lam_x, lam_y = (None, None)
                if self.poisson_fd_fft_eigs is not None:
                    lam_x, lam_y = self.poisson_fd_fft_eigs
                return inv_laplacian_fd_fft(
                    f,
                    dx=self.grid.dx,
                    dy=self.grid.dy,
                    bc=bc_phi,
                    gauge_epsilon=self.params.poisson_gauge_epsilon,
                    lam_x=lam_x,
                    lam_y=lam_y,
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
            preconditioner_fn=self.poisson_preconditioner_fn,
            x0=x0,
        )

    def inv_div_n_grad(
        self, n_eff: jnp.ndarray, f: jnp.ndarray, *, x0: jnp.ndarray | None = None
    ) -> jnp.ndarray:
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
            preconditioner_fn=self.polarization_preconditioner_fn,
            x0=x0,
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

    def bracket_many(
        self,
        phi: jnp.ndarray,
        fields: jnp.ndarray,
        *,
        bc_phi: BC2D | None = None,
        bc_f: list[BC2D | None] | None = None,
    ) -> jnp.ndarray:
        if bc_phi is None:
            bc_phi = self.grid.bc
        if bc_f is None:
            bc_f = [self.grid.bc] * fields.shape[0]

        periodic_pair = self._is_periodic(bc_phi)
        if self.params.bracket == "spectral":
            if not periodic_pair:
                raise ValueError("Spectral bracket requires periodic BCs in x and y.")
            out = poisson_bracket_spectral_multi(
                phi,
                fields,
                kx=self.perp_ops.kx,
                ky=self.perp_ops.ky,
                dealias_mask=self.grid.dealias_mask if self.params.dealias_on else None,
            )
            return float(self.params.exb_scale) * out

        out = []
        for i in range(fields.shape[0]):
            out.append(self.bracket(phi, fields[i], bc_phi=bc_phi, bc_f=bc_f[i]))
        return jnp.stack(out)

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
