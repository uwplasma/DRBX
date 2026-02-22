from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.bc import BC1D, BC2D
from jaxdrb.core.geometry import GeometryBase
from jaxdrb.core.operators import build_perp_operator_bundle, PerpOperatorBundle
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.fci.parallel import (
    parallel_derivative_centered_3d,
    parallel_derivative_target_aware_3d,
)
from jaxdrb.operators.fd2d import (
    enforce_bc_relaxation,
    build_fd_fft_eigs,
    build_div_n_grad_preconditioner,
    build_laplacian_preconditioner,
    inv_div_n_grad_cg,
    inv_laplacian_cg,
    inv_laplacian_fd_fft,
)
from jaxdrb.operators.spectral2d import inv_laplacian as inv_laplacian_spec


class FCIGeometryAdapter(GeometryBase):
    """Geometry adapter for FCI slab grids (nz, nx, ny)."""

    grid: FCISlabGrid
    params: DRBSystemParams
    perp_ops: PerpOperatorBundle = eqx.field(init=False)
    poisson_preconditioner_fn: object = eqx.field(init=False, static=True, default=None)
    polarization_preconditioner_fn: object = eqx.field(init=False, static=True, default=None)
    poisson_fd_fft_eigs: tuple[jnp.ndarray, jnp.ndarray] | None = eqx.field(
        init=False, default=None
    )
    name: ClassVar[str] = "fci"
    ndim: ClassVar[int] = 3

    def __post_init__(self):
        ops = build_perp_operator_bundle(
            scheme=self.params.perp_operator,
            bracket=self.params.bracket,
            nx=self.grid.nx,
            ny=self.grid.ny,
            dx=self.grid.dx,
            dy=self.grid.dy,
            dealias_on=self.params.dealias_on,
            bracket_zero_mean=self.params.bracket_zero_mean,
            bc=self.params.perp_bc,
        )
        object.__setattr__(self, "perp_ops", ops)

        bc = self.params.perp_bc
        precond = self.params.poisson_preconditioner
        if precond == "auto":
            precond = "spectral" if (bc.kind_x == 0 and bc.kind_y == 0) else "jacobi"
        if precond == "spectral" and not (bc.kind_x == 0 and bc.kind_y == 0):
            precond = "jacobi"
        shape = (self.grid.nx, self.grid.ny)
        if bc.kind_x == 1 and bc.kind_y == 1:
            shape = (self.grid.nx - 2, self.grid.ny - 2)
        poisson_precond = build_laplacian_preconditioner(
            shape=shape,
            dx=self.grid.dx,
            dy=self.grid.dy,
            bc=bc,
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

    def shape(self) -> tuple[int, int, int]:
        return int(self.grid.nz), int(self.grid.nx), int(self.grid.ny)

    def _vmap_plane(self, op, f: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(op)(f)

    def ddx(self, f: jnp.ndarray) -> jnp.ndarray:
        return self._vmap_plane(self.perp_ops.ddx, f)

    def ddy(self, f: jnp.ndarray) -> jnp.ndarray:
        return self._vmap_plane(self.perp_ops.ddy, f)

    def laplacian(self, f: jnp.ndarray) -> jnp.ndarray:
        return self._vmap_plane(self.perp_ops.laplacian, f)

    def biharmonic(self, f: jnp.ndarray) -> jnp.ndarray:
        return self._vmap_plane(self.perp_ops.biharmonic, f)

    def inv_laplacian(self, f: jnp.ndarray, *, x0: jnp.ndarray | None = None) -> jnp.ndarray:
        if (
            self.params.perp_operator == "spectral"
            and self.params.perp_bc.kind_x == 0
            and self.params.perp_bc.kind_y == 0
        ):
            return self._vmap_plane(
                lambda rhs: inv_laplacian_spec(rhs, self.perp_ops.k2, k2_min=self.params.k2_min),
                f,
            )

        def solve(rhs, guess=None):
            precond = self.params.poisson_preconditioner
            if precond == "auto":
                precond = "spectral"
            poisson = self.params.poisson
            if (
                self.params.poisson_force_fd_fft_when_nonperiodic
                and poisson == "spectral"
                and not (self.params.perp_bc.kind_x == 0 and self.params.perp_bc.kind_y == 0)
            ):
                poisson = "cg_fd"
            if poisson == "cg_fd":
                try:
                    lam_x, lam_y = (None, None)
                    if self.poisson_fd_fft_eigs is not None:
                        lam_x, lam_y = self.poisson_fd_fft_eigs
                    return inv_laplacian_fd_fft(
                        rhs,
                        dx=self.grid.dx,
                        dy=self.grid.dy,
                        bc=self.params.perp_bc,
                        gauge_epsilon=self.params.poisson_gauge_epsilon,
                        lam_x=lam_x,
                        lam_y=lam_y,
                    )
                except ValueError:
                    pass
            return inv_laplacian_cg(
                rhs,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=self.params.perp_bc,
                maxiter=int(self.params.poisson_maxiter),
                tol=float(self.params.poisson_tol),
                preconditioner=str(precond),
                k2_precond=(self.perp_ops.k2 if str(precond) == "spectral" else None),
                preconditioner_fn=self.poisson_preconditioner_fn,
                x0=guess,
            )

        if x0 is None:
            return self._vmap_plane(lambda rhs: solve(rhs, None), f)
        return jax.vmap(solve)(f, x0)

    def inv_div_n_grad(
        self, n_eff: jnp.ndarray, f: jnp.ndarray, *, x0: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        def solve(rhs, nc, guess=None):
            precond = self.params.poisson_preconditioner
            if precond == "auto":
                precond = "spectral"
            return inv_div_n_grad_cg(
                rhs,
                n_coeff=nc,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=self.params.perp_bc,
                maxiter=int(self.params.poisson_maxiter),
                tol=float(self.params.poisson_tol),
                preconditioner=str(precond),
                preconditioner_fn=self.polarization_preconditioner_fn,
                x0=guess,
            )

        if x0 is None:
            return jax.vmap(lambda rhs, nc: solve(rhs, nc, None))(f, n_eff)
        return jax.vmap(solve)(f, n_eff, x0)

    def bracket(
        self,
        phi: jnp.ndarray,
        f: jnp.ndarray,
        *,
        bc_phi: BC2D | None = None,
        bc_f: BC2D | None = None,
    ) -> jnp.ndarray:
        return jax.vmap(lambda p, g: self.perp_ops.bracket_op(p, g, bc_phi=bc_phi, bc_f=bc_f))(
            phi, f
        )

    def dpar(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray:
        if (
            self.grid.open_field_line
            and self.params.use_target_aware_dpar
            and self.grid.cell_centered
        ):
            if bc_kind == "dirichlet":
                bc = BC1D.dirichlet(left=0.0, right=0.0, nu=0.0)
            else:
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

    def d2par(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray:
        return self.dpar(self.dpar(f, bc_kind=bc_kind), bc_kind=bc_kind)

    def curvature(self, f: jnp.ndarray) -> jnp.ndarray:
        kappa = self.kappa_profile()
        return kappa * self.ddy(f)

    def kappa_profile(self) -> jnp.ndarray | float:
        kappa = float(self.params.kappa)
        mode = str(self.params.kappa_profile).lower()
        if mode == "cosine":
            Ly = float(self.grid.dy) * float(self.grid.ny)
            y = self.grid.y0 + self.grid.dy * (jnp.arange(self.grid.ny) + 0.5)
            theta = (2.0 * jnp.pi) * (y / max(Ly, 1e-8)) - jnp.pi
            theta0 = float(self.params.kappa_theta0)
            return kappa * jnp.cos(theta - theta0)[None, None, :]
        return kappa

    def sheath_mask_sign(self) -> tuple[jnp.ndarray, jnp.ndarray]:
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

    def enforce_bc_relaxation(self, f: jnp.ndarray, *, nu: float) -> jnp.ndarray:
        return jax.vmap(
            lambda p: enforce_bc_relaxation(
                p,
                dx=self.grid.dx,
                dy=self.grid.dy,
                bc=self.params.perp_bc,
                nu=float(nu),
            )
        )(f)
