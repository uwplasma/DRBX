from __future__ import annotations

from typing import ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.bc import BC2D
from jaxdrb.core.geometry import GeometryBase
from jaxdrb.core.geometry_logb import salpha_logb_coefficients
from jaxdrb.core.operators import PerpOperatorBundle, build_perp_operator_bundle
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.geometry.plane import Grid2D
from jaxdrb.operators.fd2d import (
    enforce_bc_relaxation,
    build_div_n_grad_preconditioner,
    build_laplacian_preconditioner,
    inv_div_n_grad_cg,
    inv_laplacian_cg,
    inv_laplacian_fd_fft,
)
from jaxdrb.operators.spectral2d import inv_laplacian as inv_laplacian_spec


class FieldAlignedGrid(eqx.Module):
    """Structured field-aligned grid (perp plane + parallel coordinate)."""

    perp: Grid2D
    z: jnp.ndarray
    dz: float = eqx.field(static=True)
    open_field_line: bool = eqx.field(static=True)
    sheath_mask: jnp.ndarray
    sheath_sign: jnp.ndarray
    region_masks: dict[str, jnp.ndarray] | None = eqx.field(default=None)
    region_bcs: tuple[object, ...] | None = eqx.field(default=None)

    @classmethod
    def make(
        cls,
        *,
        nx: int,
        ny: int,
        nz: int,
        Lx: float,
        Ly: float,
        Lz: float,
        bc_x: str,
        bc_y: str,
        dealias: bool,
        open_field_line: bool,
        bc_value_x: float = 0.0,
        bc_value_y: float = 0.0,
        bc_grad_x: float = 0.0,
        bc_grad_y: float = 0.0,
        region_masks: dict[str, jnp.ndarray] | None = None,
        region_bcs: tuple[object, ...] | None = None,
    ) -> "FieldAlignedGrid":
        perp = Grid2D.make(
            nx=nx,
            ny=ny,
            Lx=Lx,
            Ly=Ly,
            dealias=dealias,
            bc_x=bc_x,
            bc_y=bc_y,
            bc_value_x=bc_value_x,
            bc_value_y=bc_value_y,
            bc_grad_x=bc_grad_x,
            bc_grad_y=bc_grad_y,
        )

        if open_field_line:
            z = jnp.linspace(-0.5 * Lz, 0.5 * Lz, nz, endpoint=True)
            dz = float(Lz / max(nz - 1, 1))
        else:
            z = jnp.linspace(-0.5 * Lz, 0.5 * Lz, nz, endpoint=False)
            dz = float(Lz / max(nz, 1))

        sheath_mask = jnp.zeros((nz, 1, 1), dtype=jnp.float64)
        sheath_sign = jnp.zeros((nz, 1, 1), dtype=jnp.float64)
        if open_field_line and nz >= 2:
            sheath_mask = sheath_mask.at[0].set(1.0)
            sheath_mask = sheath_mask.at[-1].set(1.0)
            sheath_sign = sheath_sign.at[0].set(-1.0)
            sheath_sign = sheath_sign.at[-1].set(1.0)

        return cls(
            perp=perp,
            z=z,
            dz=dz,
            open_field_line=open_field_line,
            sheath_mask=sheath_mask,
            sheath_sign=sheath_sign,
            region_masks=region_masks,
            region_bcs=region_bcs,
        )

    @classmethod
    def from_z(
        cls,
        *,
        perp: Grid2D,
        z: jnp.ndarray,
        open_field_line: bool,
        sheath_mask: jnp.ndarray | None = None,
        sheath_sign: jnp.ndarray | None = None,
        region_masks: dict[str, jnp.ndarray] | None = None,
        region_bcs: tuple[object, ...] | None = None,
    ) -> "FieldAlignedGrid":
        z_arr = jnp.asarray(z, dtype=jnp.float64)
        if z_arr.ndim != 1:
            raise ValueError("FieldAlignedGrid.from_z expects a 1D z array.")
        dz = float(jnp.mean(jnp.diff(z_arr))) if z_arr.size > 1 else 1.0

        nz = int(z_arr.size)
        if sheath_mask is None or sheath_sign is None:
            sheath_mask = jnp.zeros((nz, 1, 1), dtype=jnp.float64)
            sheath_sign = jnp.zeros((nz, 1, 1), dtype=jnp.float64)
            if open_field_line and nz >= 2:
                sheath_mask = sheath_mask.at[0].set(1.0)
                sheath_mask = sheath_mask.at[-1].set(1.0)
                sheath_sign = sheath_sign.at[0].set(-1.0)
                sheath_sign = sheath_sign.at[-1].set(1.0)
        else:
            sheath_mask = jnp.asarray(sheath_mask, dtype=jnp.float64)
            sheath_sign = jnp.asarray(sheath_sign, dtype=jnp.float64)
            if sheath_mask.ndim == 1:
                sheath_mask = sheath_mask[:, None, None]
            if sheath_sign.ndim == 1:
                sheath_sign = sheath_sign[:, None, None]

        return cls(
            perp=perp,
            z=z_arr,
            dz=dz,
            open_field_line=bool(open_field_line),
            sheath_mask=sheath_mask,
            sheath_sign=sheath_sign,
            region_masks=region_masks,
            region_bcs=region_bcs,
        )


class FieldAlignedGeometryAdapter(GeometryBase):
    """Generic 3D field-aligned geometry adapter with vector curvature."""

    grid: FieldAlignedGrid
    params: DRBSystemParams
    curv_x: jnp.ndarray
    curv_y: jnp.ndarray
    dpar_factor: jnp.ndarray
    B: jnp.ndarray
    perp_ops: PerpOperatorBundle = eqx.field(init=False)
    poisson_preconditioner_fn: object = eqx.field(init=False, static=True, default=None)
    polarization_preconditioner_fn: object = eqx.field(init=False, static=True, default=None)

    name: ClassVar[str] = "field_aligned"
    ndim: ClassVar[int] = 3

    def __post_init__(self):
        ops = build_perp_operator_bundle(
            scheme=self.params.perp_operator,
            bracket=self.params.bracket,
            nx=self.grid.perp.nx,
            ny=self.grid.perp.ny,
            dx=self.grid.perp.dx,
            dy=self.grid.perp.dy,
            dealias_on=self.params.dealias_on,
            bracket_zero_mean=self.params.bracket_zero_mean,
            bc=self.grid.perp.bc,
        )
        object.__setattr__(self, "perp_ops", ops)

        bc = self.grid.perp.bc
        precond = self.params.poisson_preconditioner
        if precond == "auto":
            precond = "spectral"
        if precond == "spectral" and not (bc.kind_x == 0 and bc.kind_y == 0):
            precond = "jacobi"
        shape = (self.grid.perp.nx, self.grid.perp.ny)
        if bc.kind_x == 1 and bc.kind_y == 1:
            shape = (self.grid.perp.nx - 2, self.grid.perp.ny - 2)
        poisson_precond = build_laplacian_preconditioner(
            shape=shape,
            dx=self.grid.perp.dx,
            dy=self.grid.perp.dy,
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
                n_coeff = jnp.full((self.grid.perp.nx - 2, self.grid.perp.ny - 2), n_eff)
            else:
                n_coeff = jnp.full((self.grid.perp.nx, self.grid.perp.ny), n_eff)
            pol_precond = build_div_n_grad_preconditioner(
                n_coeff=n_coeff,
                dx=self.grid.perp.dx,
                dy=self.grid.perp.dy,
                bc=bc,
                preconditioner=str(pol),
                preconditioner_shift=float(self.params.polarization_precond_shift),
                n_floor=float(self.params.n0_min),
            )
        object.__setattr__(self, "polarization_preconditioner_fn", pol_precond)

        z = self.grid.z

        def _broadcast(arr, name: str) -> jnp.ndarray:
            arr = jnp.asarray(arr, dtype=jnp.float64)
            if arr.ndim == 0:
                return jnp.full_like(z, arr)
            if arr.shape != z.shape:
                raise ValueError(f"{name} must be scalar or have shape {z.shape}.")
            return arr

        object.__setattr__(self, "curv_x", _broadcast(self.curv_x, "curv_x"))
        object.__setattr__(self, "curv_y", _broadcast(self.curv_y, "curv_y"))
        object.__setattr__(self, "dpar_factor", _broadcast(self.dpar_factor, "dpar_factor"))
        object.__setattr__(self, "B", _broadcast(self.B, "B"))

    @classmethod
    def from_coefficients(
        cls,
        *,
        params: DRBSystemParams,
        grid: FieldAlignedGrid,
        curv_x: jnp.ndarray | float,
        curv_y: jnp.ndarray | float,
        dpar_factor: jnp.ndarray | float,
        B: jnp.ndarray | float = 1.0,
    ) -> "FieldAlignedGeometryAdapter":
        return cls(
            grid=grid,
            params=params,
            curv_x=jnp.asarray(curv_x),
            curv_y=jnp.asarray(curv_y),
            dpar_factor=jnp.asarray(dpar_factor),
            B=jnp.asarray(B),
        )

    @classmethod
    def from_npz(
        cls,
        *,
        path: str,
        params: DRBSystemParams,
        grid: FieldAlignedGrid,
    ) -> "FieldAlignedGeometryAdapter":
        data = np.load(path)
        curv_x = data["curv_x"]
        curv_y = data["curv_y"]
        dpar_factor = data["dpar_factor"]
        B = data["B"] if "B" in data else 1.0
        return cls.from_coefficients(
            params=params,
            grid=grid,
            curv_x=curv_x,
            curv_y=curv_y,
            dpar_factor=dpar_factor,
            B=B,
        )

    @classmethod
    def make_salpha(
        cls,
        *,
        params: DRBSystemParams,
        nx: int,
        ny: int,
        nz: int,
        Lx: float,
        Ly: float,
        Lz: float,
        bc_x: str = "periodic",
        bc_y: str = "periodic",
        dealias: bool = True,
        open_field_line: bool = False,
        shat: float = 0.796,
        alpha: float = 0.0,
        q: float = 1.4,
        R0: float = 1.0,
        epsilon: float = 0.18,
        r0: float | None = None,
        curvature0: float | None = None,
        b_min: float = 0.05,
        theta_scale: float | None = None,
        curvature_model: Literal["vector_xy", "ky_only", "logB"] = "vector_xy",
        B0: float | None = None,
        epsilon_x_grad: float | None = None,
        theta_ballooning_on: bool = False,
        theta_ballooning_r: float | None = None,
        linear_shear_on: bool = False,
        bc_value_x: float = 0.0,
        bc_value_y: float = 0.0,
        bc_grad_x: float = 0.0,
        bc_grad_y: float = 0.0,
    ) -> "FieldAlignedGeometryAdapter":
        if curvature0 is None:
            curvature0 = float(epsilon)
        grid = FieldAlignedGrid.make(
            nx=nx,
            ny=ny,
            nz=nz,
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            bc_x=bc_x,
            bc_y=bc_y,
            dealias=dealias,
            open_field_line=open_field_line,
            bc_value_x=bc_value_x,
            bc_value_y=bc_value_y,
            bc_grad_x=bc_grad_x,
            bc_grad_y=bc_grad_y,
        )

        theta_scale_val = theta_scale
        if theta_scale_val is None or float(theta_scale_val) <= 0.0:
            theta_scale_val = float(grid.z[-1] - grid.z[0]) / (2.0 * jnp.pi)
            theta_scale_val = max(theta_scale_val, 1e-8)
        theta = grid.z / float(theta_scale_val)

        model = str(curvature_model).lower()
        if model in ("logb", "logb_curvature", "logb_bracket"):
            curv_x, curv_y, dpar_factor, B = salpha_logb_coefficients(
                theta,
                epsilon=float(epsilon),
                q=float(q),
                shat=float(shat),
                R0=float(R0),
                r0=r0,
                theta_scale=float(theta_scale_val),
                B0=B0,
                epsilon_x_grad=epsilon_x_grad,
                theta_ballooning_on=theta_ballooning_on,
                theta_ballooning_r=theta_ballooning_r,
                linear_shear_on=linear_shear_on,
            )
        else:
            B = 1.0 / jnp.maximum(1.0 + epsilon * jnp.cos(theta), b_min)
            if model == "ky_only":
                curv_x = jnp.zeros_like(theta)
                curv_y = curvature0 * jnp.cos(theta) * B
            else:
                curv_x = curvature0 * jnp.sin(theta) * B
                curv_y = curvature0 * jnp.cos(theta) * B
            dpar_factor = jnp.ones_like(theta) * (float(theta_scale_val) / max(q * R0, 1e-8))

        return cls.from_coefficients(
            params=params,
            grid=grid,
            curv_x=curv_x,
            curv_y=curv_y,
            dpar_factor=dpar_factor,
            B=B,
        )

    def shape(self) -> tuple[int, int, int]:
        return int(self.grid.z.size), int(self.grid.perp.nx), int(self.grid.perp.ny)

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

    def inv_laplacian(self, f: jnp.ndarray) -> jnp.ndarray:
        if (
            self.params.perp_operator == "spectral"
            and self.grid.perp.bc.kind_x == 0
            and self.grid.perp.bc.kind_y == 0
        ):
            return self._vmap_plane(
                lambda rhs: inv_laplacian_spec(rhs, self.perp_ops.k2, k2_min=self.params.k2_min),
                f,
            )

        def solve(rhs):
            precond = self.params.poisson_preconditioner
            if precond == "auto":
                precond = "spectral"
            poisson = self.params.poisson
            if (
                self.params.poisson_force_fd_fft_when_nonperiodic
                and poisson == "spectral"
                and not (self.grid.perp.bc.kind_x == 0 and self.grid.perp.bc.kind_y == 0)
            ):
                poisson = "cg_fd"
            if poisson == "cg_fd":
                try:
                    return inv_laplacian_fd_fft(
                        rhs,
                        dx=self.grid.perp.dx,
                        dy=self.grid.perp.dy,
                        bc=self.grid.perp.bc,
                        gauge_epsilon=self.params.poisson_gauge_epsilon,
                    )
                except ValueError:
                    pass
            return inv_laplacian_cg(
                rhs,
                dx=self.grid.perp.dx,
                dy=self.grid.perp.dy,
                bc=self.grid.perp.bc,
                maxiter=int(self.params.poisson_maxiter),
                tol=float(self.params.poisson_tol),
                preconditioner=str(precond),
                k2_precond=(
                    self.perp_ops.k2 if str(precond) == "spectral" else None
                ),
                preconditioner_fn=self.poisson_preconditioner_fn,
            )

        return self._vmap_plane(solve, f)

    def inv_div_n_grad(self, n_eff: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        def solve(rhs, nc):
            precond = self.params.poisson_preconditioner
            if precond == "auto":
                precond = "spectral"
            return inv_div_n_grad_cg(
                rhs,
                n_coeff=nc,
                dx=self.grid.perp.dx,
                dy=self.grid.perp.dy,
                bc=self.grid.perp.bc,
                maxiter=int(self.params.poisson_maxiter),
                tol=float(self.params.poisson_tol),
                preconditioner=str(precond),
                preconditioner_fn=self.polarization_preconditioner_fn,
            )

        return jax.vmap(solve)(f, n_eff)

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

    def _dpar_periodic(self, f: jnp.ndarray) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * self.grid.dz)

    def _dpar_open(self, f: jnp.ndarray) -> jnp.ndarray:
        face = 0.5 * (f[1:] + f[:-1])
        df = jnp.zeros_like(f)
        df = df.at[1:-1].set((face[1:] - face[:-1]) / self.grid.dz)
        df = df.at[0].set((face[0] - f[0]) / self.grid.dz)
        df = df.at[-1].set((f[-1] - face[-1]) / self.grid.dz)
        return df

    def dpar(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray:
        _ = bc_kind
        df = self._dpar_open(f) if self.grid.open_field_line else self._dpar_periodic(f)
        return df * self.dpar_factor[:, None, None]

    def d2par(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray:
        return self.dpar(self.dpar(f, bc_kind=bc_kind), bc_kind=bc_kind)

    def curvature(self, f: jnp.ndarray) -> jnp.ndarray:
        if not self.params.curvature_on or self.params.curvature_coeff == 0.0:
            return jnp.zeros_like(f)
        curv = (self.curv_x[:, None, None] * self.ddx(f)) + (
            self.curv_y[:, None, None] * self.ddy(f)
        )
        coeff = float(self.params.curvature_coeff)
        if coeff != 1.0:
            curv = curv * coeff
        scale = self.params.curvature_scale
        if scale is None:
            scale = float(self.params.exb_scale)
        if float(scale) != 1.0:
            curv = curv * float(scale)
        return curv

    def kappa_profile(self) -> jnp.ndarray | float:
        kappa = float(self.params.kappa)
        mode = str(self.params.kappa_profile).lower()
        if mode == "cosine":
            Ly = float(self.grid.perp.dy) * float(self.grid.perp.ny)
            y = self.grid.perp.y
            theta = (2.0 * jnp.pi) * (y / max(Ly, 1e-8)) - jnp.pi
            theta0 = float(self.params.kappa_theta0)
            return kappa * jnp.cos(theta - theta0)[None, None, :]
        return kappa

    def sheath_mask_sign(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        shape = (int(self.grid.z.size), int(self.grid.perp.nx), int(self.grid.perp.ny))
        mask = jnp.broadcast_to(self.grid.sheath_mask, shape)
        sign = jnp.broadcast_to(self.grid.sheath_sign, shape)
        return mask, sign

    def enforce_bc_relaxation(self, f: jnp.ndarray, *, nu: float) -> jnp.ndarray:
        return jax.vmap(
            lambda p: enforce_bc_relaxation(
                p,
                dx=self.grid.perp.dx,
                dy=self.grid.perp.dy,
                bc=self.grid.perp.bc,
                nu=float(nu),
            )
        )(f)
