from __future__ import annotations

from typing import ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.hermes_mirror import div_n_bxgrad_f_b_xppm as div_n_bxgrad_f_b_xppm_mirror
from jaxdrb.bc import BC2D
from jaxdrb.core.geometry import GeometryBase
from jaxdrb.core.geometry_logb import salpha_logb_coefficients
from jaxdrb.core.operators import PerpOperatorBundle, build_perp_operator_bundle
from jaxdrb.core.params import DRBSystemParams
from jaxdrb.geometry.plane import Grid2D
from jaxdrb.operators.fd2d import (
    enforce_bc_relaxation,
    build_fd_fft_eigs,
    build_div_n_grad_preconditioner,
    build_laplacian_preconditioner,
    metric_laplacian,
    inv_metric_laplacian_cg,
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
    z_shift: jnp.ndarray | None = None
    curv_par: jnp.ndarray | None = None
    jacobian: jnp.ndarray | None = None
    gpar: jnp.ndarray | None = None
    gxx: jnp.ndarray | None = None
    gxy: jnp.ndarray | None = None
    gyy: jnp.ndarray | None = None
    G1: jnp.ndarray | None = None
    G3: jnp.ndarray | None = None
    d1_dx: jnp.ndarray | None = None
    g23: jnp.ndarray | None = None
    g_22: jnp.ndarray | None = None
    g_23: jnp.ndarray | None = None
    metric_dx: jnp.ndarray | None = None
    metric_dy: jnp.ndarray | None = None
    metric_dz: jnp.ndarray | None = None
    shift_idx: jnp.ndarray | None = eqx.field(init=False, default=None)
    dpar_factor_const: bool = eqx.field(init=False, static=True, default=False)
    dpar_factor_scalar: float | None = eqx.field(init=False, static=True, default=None)
    perp_ops: PerpOperatorBundle = eqx.field(init=False)
    poisson_preconditioner_fn: object = eqx.field(init=False, static=True, default=None)
    polarization_preconditioner_fn: object = eqx.field(init=False, static=True, default=None)
    poisson_fd_fft_eigs: tuple[jnp.ndarray, jnp.ndarray] | None = eqx.field(
        init=False, default=None
    )

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
            precond = "spectral" if (bc.kind_x == 0 and bc.kind_y == 0) else "jacobi"
        if precond == "spectral" and not (bc.kind_x == 0 and bc.kind_y == 0):
            precond = "jacobi"
        shape = (self.grid.perp.nx, self.grid.perp.ny)
        if bc.kind_x == 1 and bc.kind_y == 1:
            shape = (self.grid.perp.nx - 2, self.grid.perp.ny - 2)
        poisson_precond = build_laplacian_preconditioner(
            shape=shape,
            dx=self.grid.perp.dx,
            dy=self.grid.perp.dy,
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

        try:
            eigs = build_fd_fft_eigs(
                nx=self.grid.perp.nx,
                ny=self.grid.perp.ny,
                dx=self.grid.perp.dx,
                dy=self.grid.perp.dy,
                bc=bc,
                dtype=jnp.float64,
            )
        except Exception:
            eigs = None
        object.__setattr__(self, "poisson_fd_fft_eigs", eigs)

        nz = int(self.grid.z.size)
        nx = int(self.grid.perp.nx)
        ny = int(self.grid.perp.ny)
        full_shape = (nz, nx, ny)

        def _as_field3(arr, name: str) -> jnp.ndarray:
            arr = jnp.asarray(arr, dtype=jnp.float64)
            if arr.ndim == 0:
                return jnp.full(full_shape, arr, dtype=jnp.float64)
            if arr.ndim == 1:
                if arr.shape[0] != nz:
                    raise ValueError(f"{name} 1D input must have shape ({nz},).")
                return jnp.broadcast_to(arr[:, None, None], full_shape)
            if arr.ndim == 2:
                if arr.shape == (nx, ny):
                    return jnp.broadcast_to(arr[None, :, :], full_shape)
                if arr.shape == (nz, nx):
                    return jnp.broadcast_to(arr[:, :, None], full_shape)
                if arr.shape == (nx, nz):
                    return jnp.broadcast_to(jnp.swapaxes(arr, 0, 1)[:, :, None], full_shape)
                raise ValueError(
                    f"{name} 2D input must be one of (nx, ny)={(nx, ny)}, "
                    f"(nz, nx)={(nz, nx)}, or (nx, nz)={(nx, nz)}."
                )
            if arr.ndim == 3 and arr.shape == full_shape:
                return arr
            raise ValueError(f"{name} has unsupported shape {arr.shape}; expected {full_shape}.")

        object.__setattr__(self, "curv_x", _as_field3(self.curv_x, "curv_x"))
        object.__setattr__(self, "curv_y", _as_field3(self.curv_y, "curv_y"))
        if self.curv_par is not None:
            object.__setattr__(self, "curv_par", _as_field3(self.curv_par, "curv_par"))
        dpar_scaled = _as_field3(self.dpar_factor, "dpar_factor") * float(
            self.params.dpar_factor_scale
        )
        object.__setattr__(self, "dpar_factor", dpar_scaled)
        object.__setattr__(self, "B", _as_field3(self.B, "B"))
        if self.jacobian is not None:
            object.__setattr__(self, "jacobian", _as_field3(self.jacobian, "jacobian"))
        if self.gpar is not None:
            object.__setattr__(self, "gpar", _as_field3(self.gpar, "gpar"))
        if self.G1 is not None:
            object.__setattr__(self, "G1", _as_field3(self.G1, "G1"))
        if self.G3 is not None:
            object.__setattr__(self, "G3", _as_field3(self.G3, "G3"))
        if self.d1_dx is not None:
            object.__setattr__(self, "d1_dx", _as_field3(self.d1_dx, "d1_dx"))
        if self.metric_dx is not None:
            object.__setattr__(self, "metric_dx", _as_field3(self.metric_dx, "metric_dx"))
        if self.metric_dy is not None:
            object.__setattr__(self, "metric_dy", _as_field3(self.metric_dy, "metric_dy"))
        if self.metric_dz is not None:
            object.__setattr__(self, "metric_dz", _as_field3(self.metric_dz, "metric_dz"))

        gxx = self.gxx
        gxy = self.gxy
        gyy = self.gyy
        if gxx is not None:
            gxx = jnp.asarray(gxx, dtype=jnp.float64)
            if gxx.ndim not in (0, 1, 2):
                raise ValueError("gxx must be scalar, 1D (nz), or 2D (nx, ny).")
            if gxx.ndim == 1 and gxx.shape[0] != self.grid.z.size:
                raise ValueError("gxx 1D metric must have length nz.")
            if gxx.ndim == 2:
                nz = int(self.grid.z.size)
                nx = int(self.grid.perp.nx)
                ny = int(self.grid.perp.ny)
                if gxx.shape == (nx, ny):
                    pass
                elif gxx.shape == (nz, nx):
                    pass
                elif gxx.shape == (nx, nz):
                    gxx = jnp.swapaxes(gxx, 0, 1)
                else:
                    raise ValueError(
                        "gxx must have shape (nx, ny), (nz, nx), or (nx, nz) to use metric Poisson."
                    )
        if gxy is not None:
            gxy = jnp.asarray(gxy, dtype=jnp.float64)
            if gxy.ndim not in (0, 1, 2):
                raise ValueError("gxy must be scalar, 1D (nz), or 2D (nx, ny).")
            if gxy.ndim == 1 and gxy.shape[0] != self.grid.z.size:
                raise ValueError("gxy 1D metric must have length nz.")
            if gxy.ndim == 2:
                nz = int(self.grid.z.size)
                nx = int(self.grid.perp.nx)
                ny = int(self.grid.perp.ny)
                if gxy.shape == (nx, ny):
                    pass
                elif gxy.shape == (nz, nx):
                    pass
                elif gxy.shape == (nx, nz):
                    gxy = jnp.swapaxes(gxy, 0, 1)
                else:
                    raise ValueError(
                        "gxy must have shape (nx, ny), (nz, nx), or (nx, nz) to use metric Poisson."
                    )
        if gyy is not None:
            gyy = jnp.asarray(gyy, dtype=jnp.float64)
            if gyy.ndim not in (0, 1, 2):
                raise ValueError("gyy must be scalar, 1D (nz), or 2D (nx, ny).")
            if gyy.ndim == 1 and gyy.shape[0] != self.grid.z.size:
                raise ValueError("gyy 1D metric must have length nz.")
            if gyy.ndim == 2:
                nz = int(self.grid.z.size)
                nx = int(self.grid.perp.nx)
                ny = int(self.grid.perp.ny)
                if gyy.shape == (nx, ny):
                    pass
                elif gyy.shape == (nz, nx):
                    pass
                elif gyy.shape == (nx, nz):
                    gyy = jnp.swapaxes(gyy, 0, 1)
                else:
                    raise ValueError(
                        "gyy must have shape (nx, ny), (nz, nx), or (nx, nz) to use metric Poisson."
                    )
        object.__setattr__(self, "gxx", gxx)
        object.__setattr__(self, "gxy", gxy)
        object.__setattr__(self, "gyy", gyy)

        shift_idx = None
        if self.z_shift is not None:
            z_shift = jnp.asarray(self.z_shift, dtype=jnp.float64)
            nz = int(self.grid.z.size)
            nx = int(self.grid.perp.nx)
            if z_shift.ndim == 0:
                z_shift = jnp.full((nz, nx), z_shift, dtype=jnp.float64)
            elif z_shift.ndim == 1:
                if z_shift.shape[0] == nz:
                    z_shift = jnp.broadcast_to(z_shift[:, None], (nz, nx))
                elif z_shift.shape[0] == nx:
                    z_shift = jnp.broadcast_to(z_shift[None, :], (nz, nx))
                else:
                    raise ValueError("z_shift 1D input must have length nz or nx.")
            elif z_shift.ndim == 2:
                if z_shift.shape == (nz, nx):
                    pass
                elif z_shift.shape == (nx, nz):
                    z_shift = jnp.swapaxes(z_shift, 0, 1)
                else:
                    raise ValueError("z_shift 2D input must have shape (nz, nx) or (nx, nz).")
            else:
                raise ValueError("z_shift must be scalar, 1D, or 2D.")
            shift_idx = z_shift / float(self.grid.perp.dy)
        object.__setattr__(self, "shift_idx", shift_idx)

        dpar_arr = np.asarray(self.dpar_factor)
        dpar_const = bool(np.max(dpar_arr) - np.min(dpar_arr) <= 1e-12)
        dpar_scalar = float(dpar_arr.flat[0]) if dpar_const else None
        object.__setattr__(self, "dpar_factor_const", dpar_const)
        object.__setattr__(self, "dpar_factor_scalar", dpar_scalar)

    @classmethod
    def from_coefficients(
        cls,
        *,
        params: DRBSystemParams,
        grid: FieldAlignedGrid,
        curv_x: jnp.ndarray | float,
        curv_y: jnp.ndarray | float,
        curv_par: jnp.ndarray | float | None = None,
        dpar_factor: jnp.ndarray | float,
        B: jnp.ndarray | float = 1.0,
        z_shift: jnp.ndarray | float | None = None,
        jacobian: jnp.ndarray | None = None,
        gpar: jnp.ndarray | None = None,
        gxx: jnp.ndarray | None = None,
        gxy: jnp.ndarray | None = None,
        gyy: jnp.ndarray | None = None,
        G1: jnp.ndarray | None = None,
        G3: jnp.ndarray | None = None,
        d1_dx: jnp.ndarray | None = None,
        g23: jnp.ndarray | None = None,
        g_22: jnp.ndarray | None = None,
        g_23: jnp.ndarray | None = None,
        metric_dx: jnp.ndarray | None = None,
        metric_dy: jnp.ndarray | None = None,
        metric_dz: jnp.ndarray | None = None,
    ) -> "FieldAlignedGeometryAdapter":
        return cls(
            grid=grid,
            params=params,
            curv_x=jnp.asarray(curv_x),
            curv_y=jnp.asarray(curv_y),
            curv_par=None if curv_par is None else jnp.asarray(curv_par),
            dpar_factor=jnp.asarray(dpar_factor),
            B=jnp.asarray(B),
            z_shift=None if z_shift is None else jnp.asarray(z_shift),
            jacobian=jacobian,
            gpar=gpar,
            gxx=gxx,
            gxy=gxy,
            gyy=gyy,
            G1=G1,
            G3=G3,
            d1_dx=d1_dx,
            g23=g23,
            g_22=g_22,
            g_23=g_23,
            metric_dx=metric_dx,
            metric_dy=metric_dy,
            metric_dz=metric_dz,
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
        curv_par = data["curv_par"] if "curv_par" in data else None
        dpar_factor = data["dpar_factor"]
        B = data["B"] if "B" in data else 1.0
        z_shift = data["z_shift"] if "z_shift" in data else None
        jacobian = data["J"] if "J" in data else None
        gpar = data["gpar"] if "gpar" in data else None
        gxx = data["gxx"] if "gxx" in data else None
        gxy = data["gxy"] if "gxy" in data else None
        gyy = data["gyy"] if "gyy" in data else None
        G1 = data["G1"] if "G1" in data else None
        G3 = data["G3"] if "G3" in data else None
        d1_dx = data["d1_dx"] if "d1_dx" in data else None
        g23 = data["g23"] if "g23" in data else None
        g_22 = data["g_22"] if "g_22" in data else None
        g_23 = data["g_23"] if "g_23" in data else None
        metric_dx = (
            data["metric_dx"] if "metric_dx" in data else (data["dx"] if "dx" in data else None)
        )
        metric_dy = (
            data["metric_dy"] if "metric_dy" in data else (data["dy"] if "dy" in data else None)
        )
        metric_dz = (
            data["metric_dz"] if "metric_dz" in data else (data["dz"] if "dz" in data else None)
        )
        return cls.from_coefficients(
            params=params,
            grid=grid,
            curv_x=curv_x,
            curv_y=curv_y,
            curv_par=curv_par,
            dpar_factor=dpar_factor,
            B=B,
            z_shift=z_shift,
            jacobian=jacobian,
            gpar=gpar,
            gxx=gxx,
            gxy=gxy,
            gyy=gyy,
            G1=G1,
            G3=G3,
            d1_dx=d1_dx,
            g23=g23,
            g_22=g_22,
            g_23=g_23,
            metric_dx=metric_dx,
            metric_dy=metric_dy,
            metric_dz=metric_dz,
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
        mode = str(getattr(self.params, "parallel_z_mode", "vmap")).lower()
        if mode == "scan":
            return jax.lax.map(op, f)
        return jax.vmap(op)(f)

    def ddx(self, f: jnp.ndarray) -> jnp.ndarray:
        return self._vmap_plane(self.perp_ops.ddx, f)

    def ddy(self, f: jnp.ndarray) -> jnp.ndarray:
        return self._vmap_plane(self.perp_ops.ddy, f)

    def laplacian(self, f: jnp.ndarray) -> jnp.ndarray:
        return self._vmap_plane(self.perp_ops.laplacian, f)

    def metric_available(self) -> bool:
        return (self.gxx is not None) and (self.gxy is not None) and (self.gyy is not None)

    def laplacian_metric(self, f: jnp.ndarray) -> jnp.ndarray:
        if not self.metric_available():
            return self.laplacian(f)
        gxx = jnp.asarray(self.gxx)
        gxy = jnp.asarray(self.gxy)
        gyy = jnp.asarray(self.gyy)
        jac = None if self.jacobian is None else jnp.asarray(self.jacobian)
        nx = self.grid.perp.nx
        ny = self.grid.perp.ny
        nz = int(self.grid.z.size)

        def _plane_metric(u, gx, gxy_loc, gy, jloc):
            gxx_plane = gx if gx.ndim == 2 else jnp.full((nx, ny), gx)
            gxy_plane = gxy_loc if gxy_loc.ndim == 2 else jnp.full((nx, ny), gxy_loc)
            gyy_plane = gy if gy.ndim == 2 else jnp.full((nx, ny), gy)
            j_plane = None
            if jloc is not None:
                j_plane = jloc if jloc.ndim == 2 else jnp.full((nx, ny), jloc)
            return metric_laplacian(
                u,
                gxx_plane,
                gxy_plane,
                gyy_plane,
                self.grid.perp.dx,
                self.grid.perp.dy,
                self.grid.perp.bc,
                jacobian=j_plane,
            )

        if gxx.ndim == 1:
            gxy_arr = gxy if gxy.ndim == 1 else jnp.full_like(gxx, gxy)
            gyy_arr = gyy if gyy.ndim == 1 else jnp.full_like(gxx, gyy)
            jac_arr = None
            if jac is not None:
                if jac.ndim in (1, 3):
                    jac_arr = jac
                else:
                    jac_arr = jnp.full_like(gxx, jac)
            else:
                jac_arr = jnp.full_like(gxx, jnp.nan)
            return jax.vmap(_plane_metric)(f, gxx, gxy_arr, gyy_arr, jac_arr)

        if gxx.ndim == 2 and gxx.shape == (nz, nx):

            def _plane_zx(u, gx, gxy_loc, gy, jloc):
                gxx_plane = jnp.broadcast_to(gx[:, None], (nx, ny))
                gxy_plane = jnp.broadcast_to(gxy_loc[:, None], (nx, ny))
                gyy_plane = jnp.broadcast_to(gy[:, None], (nx, ny))
                j_plane = None
                if jloc is not None:
                    j_plane = jloc if jloc.ndim == 2 else jnp.full((nx, ny), jloc)
                return metric_laplacian(
                    u,
                    gxx_plane,
                    gxy_plane,
                    gyy_plane,
                    self.grid.perp.dx,
                    self.grid.perp.dy,
                    self.grid.perp.bc,
                    jacobian=j_plane,
                )

            gxy_arr = gxy if gxy.ndim == 2 else jnp.full_like(gxx, gxy)
            gyy_arr = gyy if gyy.ndim == 2 else jnp.full_like(gxx, gyy)
            jac_arr = jac
            return jax.vmap(_plane_zx)(f, gxx, gxy_arr, gyy_arr, jac_arr)

        if jac is not None and jac.ndim == 3:
            return jax.vmap(lambda u, jloc: _plane_metric(u, gxx, gxy, gyy, jloc))(f, jac)
        return self._vmap_plane(lambda u: _plane_metric(u, gxx, gxy, gyy, jac), f)

    def inv_laplacian_metric(self, f: jnp.ndarray, *, x0: jnp.ndarray | None = None) -> jnp.ndarray:
        if not self.metric_available():
            return self.inv_laplacian(f, x0=x0)

        gxx = jnp.asarray(self.gxx)
        gxy = jnp.asarray(self.gxy)
        gyy = jnp.asarray(self.gyy)
        jac = None if self.jacobian is None else jnp.asarray(self.jacobian)
        nx = self.grid.perp.nx
        ny = self.grid.perp.ny
        nz = int(self.grid.z.size)

        def _plane_metric(u, gx, gxy_loc, gy, jloc):
            gxx_plane = gx if gx.ndim == 2 else jnp.full((nx, ny), gx)
            gxy_plane = gxy_loc if gxy_loc.ndim == 2 else jnp.full((nx, ny), gxy_loc)
            gyy_plane = gy if gy.ndim == 2 else jnp.full((nx, ny), gy)
            j_plane = None
            if jloc is not None:
                j_plane = jloc if jloc.ndim == 2 else jnp.full((nx, ny), jloc)
            return gxx_plane, gxy_plane, gyy_plane, j_plane

        def solve(rhs, guess=None):
            precond = self.params.poisson_preconditioner
            if precond == "auto":
                precond = "spectral"
            gxx_plane, gxy_plane, gyy_plane, j_plane = _plane_metric(rhs, gxx, gxy, gyy, jac)
            return inv_metric_laplacian_cg(
                rhs,
                gxx=gxx_plane,
                gxy=gxy_plane,
                gyy=gyy_plane,
                jacobian=j_plane,
                dx=self.grid.perp.dx,
                dy=self.grid.perp.dy,
                bc=self.grid.perp.bc,
                maxiter=int(self.params.poisson_maxiter),
                tol=float(self.params.poisson_tol),
                atol=float(self.params.poisson_cg_atol),
                preconditioner=str(precond),
                k2_precond=self.perp_ops.k2 if str(precond) == "spectral" else None,
                gauge_epsilon=self.params.poisson_gauge_epsilon,
                preconditioner_fn=self.poisson_preconditioner_fn,
                x0=guess,
            )

        if gxx.ndim == 1:

            def solve_z(rhs, gx, gxy_loc, gy, jloc, guess=None):
                gxx_plane = jnp.full((nx, ny), gx)
                gxy_plane = jnp.full((nx, ny), gxy_loc)
                gyy_plane = jnp.full((nx, ny), gy)
                j_plane = None
                if jloc is not None:
                    j_plane = jloc if getattr(jloc, "ndim", 0) == 2 else jnp.full((nx, ny), jloc)
                precond = self.params.poisson_preconditioner
                if precond == "auto":
                    precond = "spectral"
                return inv_metric_laplacian_cg(
                    rhs,
                    gxx=gxx_plane,
                    gxy=gxy_plane,
                    gyy=gyy_plane,
                    jacobian=j_plane,
                    dx=self.grid.perp.dx,
                    dy=self.grid.perp.dy,
                    bc=self.grid.perp.bc,
                    maxiter=int(self.params.poisson_maxiter),
                    tol=float(self.params.poisson_tol),
                    atol=float(self.params.poisson_cg_atol),
                    preconditioner=str(precond),
                    k2_precond=self.perp_ops.k2 if str(precond) == "spectral" else None,
                    gauge_epsilon=self.params.poisson_gauge_epsilon,
                    preconditioner_fn=self.poisson_preconditioner_fn,
                    x0=guess,
                )

            gxy_arr = gxy if gxy.ndim == 1 else jnp.full_like(gxx, gxy)
            gyy_arr = gyy if gyy.ndim == 1 else jnp.full_like(gxx, gyy)
            if jac is not None:
                if jac.ndim in (1, 3):
                    jac_arr = jac
                else:
                    jac_arr = jnp.full_like(gxx, jac)
            else:
                jac_arr = jnp.full_like(gxx, jnp.nan)
            if x0 is None:
                return jax.vmap(
                    lambda rhs, gx, gxy_loc, gy, jloc: solve_z(rhs, gx, gxy_loc, gy, jloc, None)
                )(f, gxx, gxy_arr, gyy_arr, jac_arr)
            return jax.vmap(solve_z)(f, gxx, gxy_arr, gyy_arr, jac_arr, x0)

        if gxx.ndim == 2 and gxx.shape == (nz, nx):

            def solve_zx(rhs, gx, gxy_loc, gy, jloc, guess=None):
                gxx_plane = jnp.broadcast_to(gx[:, None], (nx, ny))
                gxy_plane = jnp.broadcast_to(gxy_loc[:, None], (nx, ny))
                gyy_plane = jnp.broadcast_to(gy[:, None], (nx, ny))
                j_plane = None
                if jloc is not None:
                    j_plane = jloc if getattr(jloc, "ndim", 0) == 2 else jnp.full((nx, ny), jloc)
                precond = self.params.poisson_preconditioner
                if precond == "auto":
                    precond = "spectral"
                return inv_metric_laplacian_cg(
                    rhs,
                    gxx=gxx_plane,
                    gxy=gxy_plane,
                    gyy=gyy_plane,
                    jacobian=j_plane,
                    dx=self.grid.perp.dx,
                    dy=self.grid.perp.dy,
                    bc=self.grid.perp.bc,
                    maxiter=int(self.params.poisson_maxiter),
                    tol=float(self.params.poisson_tol),
                    atol=float(self.params.poisson_cg_atol),
                    preconditioner=str(precond),
                    k2_precond=self.perp_ops.k2 if str(precond) == "spectral" else None,
                    gauge_epsilon=self.params.poisson_gauge_epsilon,
                    preconditioner_fn=self.poisson_preconditioner_fn,
                    x0=guess,
                )

            gxy_arr = gxy if gxy.ndim == 2 else jnp.full_like(gxx, gxy)
            gyy_arr = gyy if gyy.ndim == 2 else jnp.full_like(gxx, gyy)
            if x0 is None:
                return jax.vmap(
                    lambda rhs, gx, gxy_loc, gy, jloc: solve_zx(rhs, gx, gxy_loc, gy, jloc, None)
                )(f, gxx, gxy_arr, gyy_arr, jac)
            return jax.vmap(solve_zx)(f, gxx, gxy_arr, gyy_arr, jac, x0)

        if x0 is None:
            return self._vmap_plane(lambda rhs: solve(rhs, None), f)
        return jax.vmap(solve)(f, x0)

    def biharmonic(self, f: jnp.ndarray) -> jnp.ndarray:
        return self._vmap_plane(self.perp_ops.biharmonic, f)

    def inv_laplacian(self, f: jnp.ndarray, *, x0: jnp.ndarray | None = None) -> jnp.ndarray:
        if (
            self.params.perp_operator == "spectral"
            and self.grid.perp.bc.kind_x == 0
            and self.grid.perp.bc.kind_y == 0
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
                and not (self.grid.perp.bc.kind_x == 0 and self.grid.perp.bc.kind_y == 0)
            ):
                poisson = "cg_fd"
            if poisson == "cg_fd":
                try:
                    lam_x, lam_y = (None, None)
                    if self.poisson_fd_fft_eigs is not None:
                        lam_x, lam_y = self.poisson_fd_fft_eigs
                    return inv_laplacian_fd_fft(
                        rhs,
                        dx=self.grid.perp.dx,
                        dy=self.grid.perp.dy,
                        bc=self.grid.perp.bc,
                        gauge_epsilon=self.params.poisson_gauge_epsilon,
                        lam_x=lam_x,
                        lam_y=lam_y,
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
        # Fast path when n_eff is constant -> scaled Laplacian solve.
        if jnp.ndim(n_eff) == 0:
            return self.inv_laplacian(f / n_eff, x0=None if x0 is None else x0 / n_eff)

        def solve(rhs, nc, guess=None):
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
                x0=guess,
            )

        def _is_constant(arr: jnp.ndarray) -> jnp.ndarray:
            return jnp.max(arr) - jnp.min(arr) < 1e-12

        is_const = _is_constant(jnp.asarray(n_eff))

        def _solve_const(args):
            rhs, nc, guess = args
            n0 = jnp.asarray(nc)[0, 0, 0] if nc.ndim == 3 else jnp.asarray(nc)[0, 0]
            guess = None if guess is None else guess / n0
            return self.inv_laplacian(rhs / n0, x0=guess)

        def _solve_var(args):
            rhs, nc, guess = args
            if guess is None:
                return jax.vmap(lambda rhs_i, nc_i: solve(rhs_i, nc_i, None))(rhs, nc)
            return jax.vmap(solve)(rhs, nc, guess)

        return jax.lax.cond(
            is_const,
            _solve_const,
            _solve_var,
            (f, n_eff, x0),
        )

    def bracket(
        self,
        phi: jnp.ndarray,
        f: jnp.ndarray,
        *,
        bc_phi: BC2D | None = None,
        bc_f: BC2D | None = None,
    ) -> jnp.ndarray:
        exb_y_scale = float(self.params.exb_y_scale)
        return jax.vmap(
            lambda p, g: self.perp_ops.bracket_op(
                p, g, bc_phi=bc_phi, bc_f=bc_f, exb_y_scale=exb_y_scale
            )
        )(phi, f)

    def bracket_many(
        self,
        phi: jnp.ndarray,
        fields: jnp.ndarray,
        *,
        bc_phi: BC2D | None = None,
        bc_f: list[BC2D | None] | None = None,
    ) -> jnp.ndarray:
        if bc_f is None:
            bc_f = [None] * fields.shape[0]

        def _plane(phi_plane, field_plane):
            return self.perp_ops.bracket_many(
                phi_plane,
                field_plane,
                bc_phi=bc_phi,
                bc_f=bc_f,
                exb_y_scale=self.params.exb_y_scale,
            )

        return jax.vmap(_plane, in_axes=(0, 1), out_axes=1)(phi, fields)

    def exb_flux_divergence(
        self,
        phi: jnp.ndarray,
        adv: jnp.ndarray,
        *,
        bc_phi: BC2D | None = None,
        bc_adv: BC2D | None = None,
        positive: bool = False,
    ) -> jnp.ndarray:
        if self.jacobian is None:
            raise ValueError("ExB flux divergence requires jacobian data.")
        bc_phi = self.grid.perp.bc if bc_phi is None else bc_phi
        bc_adv = self.grid.perp.bc if bc_adv is None else bc_adv
        exb_y_scale = float(self.params.exb_y_scale)
        neumann_avg_y = bool(getattr(self.params, "neumann_boundary_average_y", False))
        phi_eff = jnp.asarray(phi)
        adv_eff = jnp.asarray(adv)

        def _shift(arr: jnp.ndarray, offset: int, axis: int, *, periodic: bool) -> jnp.ndarray:
            if periodic:
                return jnp.roll(arr, int(offset), axis=axis)
            n = arr.shape[axis]
            idx = jnp.clip(jnp.arange(n) + int(offset), 0, n - 1)
            return jnp.take(arr, idx, axis=axis)

        def _fromm_face(
            arr: jnp.ndarray,
            vel: jnp.ndarray,
            axis: int,
            *,
            periodic: bool,
            positive: bool = False,
        ) -> jnp.ndarray:
            a_i = arr
            a_ip1 = _shift(arr, +1, axis, periodic=periodic)
            a_im1 = _shift(arr, -1, axis, periodic=periodic)
            a_ip2 = _shift(arr, +2, axis, periodic=periodic)
            upwind_pos = a_i + 0.25 * (a_ip1 - a_im1)
            upwind_neg = a_ip1 - 0.25 * (a_ip2 - a_i)
            face_val = jnp.where(vel > 0.0, upwind_pos, upwind_neg)
            if positive:
                face_val = jnp.maximum(face_val, 0.0)
            return face_val

        def _minmod3(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
            same_sign = (a * b > 0.0) & (a * c > 0.0)
            mag = jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c)))
            return jnp.where(same_sign, jnp.sign(a) * mag, 0.0)

        def _mc_lr(
            arr: jnp.ndarray,
            *,
            axis: int,
            periodic: bool,
            kind: int | None = None,
            value: float = 0.0,
            grad: float = 0.0,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            if axis == 1 and (not periodic):
                left_ghost = _x_ghost(
                    arr,
                    side="left",
                    kind=int(kind or 0),
                    value=float(value),
                    grad=float(grad),
                    order=1,
                )
                right_ghost = _x_ghost(
                    arr,
                    side="right",
                    kind=int(kind or 0),
                    value=float(value),
                    grad=float(grad),
                    order=1,
                )
                a_c = arr
                a_m = jnp.concatenate([left_ghost[:, None, :], arr[:, :-1, :]], axis=1)
                a_p = jnp.concatenate([arr[:, 1:, :], right_ghost[:, None, :]], axis=1)
            else:
                a_c = arr
                a_m = _shift(arr, -1, axis=axis, periodic=periodic)
                a_p = _shift(arr, +1, axis=axis, periodic=periodic)
            slope = _minmod3(2.0 * (a_p - a_c), 0.5 * (a_p - a_m), 2.0 * (a_c - a_m))
            left = a_c - 0.5 * slope
            right = a_c + 0.5 * slope
            return left, right

        def _x_ghost(
            arr: jnp.ndarray,
            *,
            side: str,
            kind: int,
            value: float,
            grad: float,
            order: int = 1,
        ) -> jnp.ndarray:
            if side == "left":
                edge = arr[:, 0, :]
                near = arr[:, 1, :] if arr.shape[1] > 1 else edge
                if neumann_avg_y and kind == 2:
                    edge = jnp.mean(edge, axis=-1, keepdims=True)
                    edge = jnp.broadcast_to(edge, arr[:, 0, :].shape)
                if kind == 1:
                    if order == 1:
                        return 2.0 * float(value) - edge
                    return 2.0 * float(value) - near
                if kind == 2:
                    if order == 1:
                        return edge - float(grad) * float(self.grid.perp.dx)
                    return near - 3.0 * float(grad) * float(self.grid.perp.dx)
                return arr[:, -1, :]
            edge = arr[:, -1, :]
            near = arr[:, -2, :] if arr.shape[1] > 1 else edge
            if neumann_avg_y and kind == 2:
                edge = jnp.mean(edge, axis=-1, keepdims=True)
                edge = jnp.broadcast_to(edge, arr[:, -1, :].shape)
            if kind == 1:
                if order == 1:
                    return 2.0 * float(value) - edge
                return 2.0 * float(value) - near
            if kind == 2:
                if order == 1:
                    return edge + float(grad) * float(self.grid.perp.dx)
                return near + 3.0 * float(grad) * float(self.grid.perp.dx)
            return arr[:, 0, :]

        def _fromm_x_boundary_flux(
            adv_arr: jnp.ndarray,
            vel: jnp.ndarray,
            *,
            side: str,
            kind: int,
            value: float,
            grad: float,
            positive: bool = False,
        ) -> jnp.ndarray:
            if side == "left":
                a0 = adv_arr[:, 0, :]
                a1 = adv_arr[:, 1, :] if adv_arr.shape[1] > 1 else a0
                ag1 = _x_ghost(adv_arr, side="left", kind=kind, value=value, grad=grad, order=1)
                ag2 = _x_ghost(adv_arr, side="left", kind=kind, value=value, grad=grad, order=2)
                outflow = a0 - 0.25 * (a1 - ag1)
                inflow = ag1 + 0.25 * (a0 - ag2)
                face_val = jnp.where(vel < 0.0, outflow, inflow)
            else:
                a0 = adv_arr[:, -1, :]
                am1 = adv_arr[:, -2, :] if adv_arr.shape[1] > 1 else a0
                ag1 = _x_ghost(adv_arr, side="right", kind=kind, value=value, grad=grad, order=1)
                ag2 = _x_ghost(adv_arr, side="right", kind=kind, value=value, grad=grad, order=2)
                outflow = a0 + 0.25 * (ag1 - am1)
                inflow = ag1 - 0.25 * (ag2 - a0)
                face_val = jnp.where(vel > 0.0, outflow, inflow)
            if positive:
                face_val = jnp.maximum(face_val, 0.0)
            return vel * face_val

        exb_flux_scheme = str(getattr(self.params, "exb_flux_scheme", "centered")).lower()
        if exb_flux_scheme == "hermes_mirror":
            if self.gxx is None or self.g23 is None or self.B is None or self.z_shift is None:
                raise ValueError(
                    "Hermes mirror ExB flux requires gxx, g23, B/Bxy, and z_shift coefficients."
                )
            if self.metric_dx is None or self.metric_dy is None or self.metric_dz is None:
                raise ValueError("Hermes mirror ExB flux requires dx, dy, and dz metric fields.")
            zlength = float(np.asarray(self.metric_dz).mean()) * float(self.grid.perp.ny)
            poloidal_x_scale = float(getattr(self.params, "exb_poloidal_x_scale", 1.0))
            poloidal_y_scale = float(getattr(self.params, "exb_poloidal_y_scale", 1.0))
            poloidal_scale = float(getattr(self.params, "exb_poloidal_scale", 1.0))
            return div_n_bxgrad_f_b_xppm_mirror(
                adv_eff,
                phi_eff,
                jacobian=self.jacobian,
                dx=self.metric_dx,
                dy=self.metric_dy,
                dz=self.metric_dz,
                g11=self.gxx,
                g23=self.g23,
                bxy=self.B,
                z_shift=self.z_shift,
                zlength=zlength,
                bc_phi=bc_phi,
                bc_adv=bc_adv,
                bndry_flux=bool(getattr(self.params, "exb_bndry_flux", True)),
                poloidal=bool(getattr(self.params, "exb_poloidal_flows", False)),
                positive=positive,
                interp=str(getattr(self.params, "parallel_shift_interp", "spectral")),
                neumann_boundary_average_z=bool(
                    getattr(self.params, "neumann_boundary_average_y", False)
                ),
                use_mc=True,
                periodic_parallel=not bool(self.grid.open_field_line),
                periodic_binormal=int(getattr(bc_adv, "kind_y", 0)) == 0,
                lower_boundary_open=bool(self.grid.open_field_line),
                upper_boundary_open=bool(self.grid.open_field_line),
                poisson_invert_set=bool(getattr(self.params, "poisson_invert_set", False)),
                parallel_edge_block=int(
                    getattr(self.params, "hermes_mirror_parallel_edge_block", 0)
                ),
                poloidal_scale=poloidal_scale,
                poloidal_x_scale=poloidal_x_scale,
                poloidal_y_scale=poloidal_y_scale,
            )
        if exb_flux_scheme in ("hermes_fromm", "hermes_xppm"):
            # Hermes/BOUT Div_n_bxGrad_f_B_XPPM-style X-Z flux:
            # corner-interpolated stream function + upwinded face reconstruction.
            J = jnp.asarray(self.jacobian, dtype=jnp.float64)
            if J.ndim == 2:
                J = jnp.broadcast_to(J[None, :, :], phi_eff.shape)
            if self.metric_dx is None:
                dx = jnp.full_like(J, float(self.grid.perp.dx))
            else:
                dx = jnp.asarray(self.metric_dx, dtype=jnp.float64)
                if dx.ndim == 2:
                    dx = jnp.broadcast_to(dx[None, :, :], phi_eff.shape)
            if self.metric_dz is None:
                dz = jnp.full_like(J, float(self.grid.perp.dy))
            else:
                dz = jnp.asarray(self.metric_dz, dtype=jnp.float64)
                if dz.ndim == 2:
                    dz = jnp.broadcast_to(dz[None, :, :], phi_eff.shape)
            periodic_x = int(getattr(bc_adv, "kind_x", 0)) == 0
            periodic_z = int(getattr(bc_adv, "kind_y", 0)) == 0
            use_mc = exb_flux_scheme == "hermes_xppm"
            bndry_flux = bool(getattr(self.params, "exb_bndry_flux", True))

            if periodic_x:
                f_xm = _shift(phi_eff, -1, axis=1, periodic=True)
                f_xp = _shift(phi_eff, +1, axis=1, periodic=True)
                J_xm = _shift(J, -1, axis=1, periodic=True)
                J_xp = _shift(J, +1, axis=1, periodic=True)
            else:
                phi_kind = int(getattr(bc_phi, "kind_x", 0))
                phi_val = float(getattr(bc_phi, "x_value", 0.0))
                phi_grad = float(getattr(bc_phi, "x_grad", 0.0))
                phi_left_ghost = _x_ghost(
                    phi_eff, side="left", kind=phi_kind, value=phi_val, grad=phi_grad, order=1
                )
                phi_right_ghost = _x_ghost(
                    phi_eff, side="right", kind=phi_kind, value=phi_val, grad=phi_grad, order=1
                )
                f_xm = jnp.concatenate([phi_left_ghost[:, None, :], phi_eff[:, :-1, :]], axis=1)
                f_xp = jnp.concatenate([phi_eff[:, 1:, :], phi_right_ghost[:, None, :]], axis=1)
                if J.shape[1] > 1:
                    J_left_ghost = 2.0 * J[:, 0, :] - J[:, 1, :]
                    J_right_ghost = 2.0 * J[:, -1, :] - J[:, -2, :]
                else:
                    J_left_ghost = J[:, 0, :]
                    J_right_ghost = J[:, -1, :]
                J_xm = jnp.concatenate([J_left_ghost[:, None, :], J[:, :-1, :]], axis=1)
                J_xp = jnp.concatenate([J[:, 1:, :], J_right_ghost[:, None, :]], axis=1)
            f_zm = _shift(phi_eff, -1, axis=2, periodic=periodic_z)
            f_zp = _shift(phi_eff, +1, axis=2, periodic=periodic_z)
            f_xm_zm = _shift(f_xm, -1, axis=2, periodic=periodic_z)
            f_xm_zp = _shift(f_xm, +1, axis=2, periodic=periodic_z)
            f_xp_zm = _shift(f_xp, -1, axis=2, periodic=periodic_z)
            f_xp_zp = _shift(f_xp, +1, axis=2, periodic=periodic_z)

            fmm = 0.25 * (phi_eff + f_xm + f_zm + f_xm_zm)
            fmp = 0.25 * (phi_eff + f_xm + f_zp + f_xm_zp)
            fpp = 0.25 * (phi_eff + f_xp + f_zp + f_xp_zp)
            fpm = 0.25 * (phi_eff + f_xp + f_zm + f_xp_zm)

            v_u = J * (fmp - fpp) / jnp.maximum(dx, 1e-30)
            v_r = 0.5 * (J + J_xp) * (fpp - fpm) / jnp.maximum(dz, 1e-30)
            if use_mc:
                kind = int(getattr(bc_adv, "kind_x", 0))
                val = float(getattr(bc_adv, "x_value", 0.0))
                grad = float(getattr(bc_adv, "x_grad", 0.0))
                left_x, right_x = _mc_lr(
                    adv_eff,
                    axis=1,
                    periodic=periodic_x,
                    kind=kind,
                    value=val,
                    grad=grad,
                )
                right_state = right_x
                left_state_next = _shift(left_x, +1, axis=1, periodic=periodic_x)
                if positive:
                    right_state = jnp.maximum(right_state, 0.0)
                    left_state_next = jnp.maximum(left_state_next, 0.0)
                flux_r = jnp.where(v_r > 0.0, v_r * right_state, v_r * left_state_next)

                if periodic_x:
                    flux_l = _shift(flux_r, -1, axis=1, periodic=periodic_x)
                else:
                    v_l = 0.5 * (J + J_xm) * (fmp - fmm) / jnp.maximum(dz, 1e-30)
                    left_ghost = _x_ghost(
                        adv_eff, side="left", kind=kind, value=val, grad=grad, order=1
                    )
                    right_ghost = _x_ghost(
                        adv_eff, side="right", kind=kind, value=val, grad=grad, order=1
                    )
                    left_in = 0.5 * (left_ghost + adv_eff[:, 0, :])
                    right_in = 0.5 * (right_ghost + adv_eff[:, -1, :])
                    left_out = left_x[:, 0, :]
                    right_out = right_x[:, -1, :]
                    if positive:
                        left_in = jnp.maximum(left_in, 0.0)
                        right_in = jnp.maximum(right_in, 0.0)
                        left_out = jnp.maximum(left_out, 0.0)
                        right_out = jnp.maximum(right_out, 0.0)
                    flux_left_b = jnp.where(
                        v_l[:, 0, :] < 0.0,
                        v_l[:, 0, :] * left_out,
                        v_l[:, 0, :] * left_in,
                    )
                    flux_right_b = jnp.where(
                        v_r[:, -1, :] > 0.0,
                        v_r[:, -1, :] * right_out,
                        v_r[:, -1, :] * right_in,
                    )
                    if not bndry_flux:
                        flux_left_b = jnp.where(v_l[:, 0, :] < 0.0, v_l[:, 0, :] * left_out, 0.0)
                        flux_right_b = jnp.where(
                            v_r[:, -1, :] > 0.0, v_r[:, -1, :] * right_out, 0.0
                        )
                    flux_r = flux_r.at[:, -1, :].set(flux_right_b)
                    flux_l = jnp.concatenate([flux_left_b[:, None, :], flux_r[:, :-1, :]], axis=1)

                left_z, right_z = _mc_lr(adv_eff, axis=2, periodic=periodic_z)
                up_state = right_z
                down_state = _shift(left_z, +1, axis=2, periodic=periodic_z)
                if positive:
                    up_state = jnp.maximum(up_state, 0.0)
                    down_state = jnp.maximum(down_state, 0.0)
                flux_u = jnp.where(v_u > 0.0, v_u * up_state, v_u * down_state)
                flux_d = _shift(flux_u, -1, axis=2, periodic=periodic_z)
            else:
                n_face_r = _fromm_face(adv_eff, v_r, axis=1, periodic=periodic_x, positive=positive)
                flux_r = v_r * n_face_r
                if periodic_x:
                    flux_l = _shift(flux_r, -1, axis=1, periodic=periodic_x)
                else:
                    kind = int(getattr(bc_adv, "kind_x", 0))
                    val = float(getattr(bc_adv, "x_value", 0.0))
                    grad = float(getattr(bc_adv, "x_grad", 0.0))
                    v_l = 0.5 * (J + J_xm) * (fmp - fmm) / jnp.maximum(dz, 1e-30)
                    flux_left_b = _fromm_x_boundary_flux(
                        adv_eff,
                        v_l[:, 0, :],
                        side="left",
                        kind=kind,
                        value=val,
                        grad=grad,
                        positive=positive,
                    )
                    flux_right_b = _fromm_x_boundary_flux(
                        adv_eff,
                        v_r[:, -1, :],
                        side="right",
                        kind=kind,
                        value=val,
                        grad=grad,
                        positive=positive,
                    )
                    flux_r = flux_r.at[:, -1, :].set(flux_right_b)
                    flux_l = jnp.concatenate([flux_left_b[:, None, :], flux_r[:, :-1, :]], axis=1)

                n_face_u = _fromm_face(adv_eff, v_u, axis=2, periodic=periodic_z, positive=positive)
                flux_u = v_u * n_face_u
                flux_d = _shift(flux_u, -1, axis=2, periodic=periodic_z)

            out = (flux_r - flux_l) / (jnp.maximum(J, 1e-30) * jnp.maximum(dx, 1e-30)) + (
                flux_u - flux_d
            ) / (jnp.maximum(J, 1e-30) * jnp.maximum(dz, 1e-30))
        else:

            def _plane(phi_plane, adv_plane, jac_plane):
                return self.perp_ops.exb_flux_divergence_centered(
                    phi_plane,
                    adv_plane,
                    dx=self.grid.perp.dx,
                    dy=self.grid.perp.dy,
                    jacobian=jac_plane,
                    bc_phi=bc_phi,
                    bc_adv=bc_adv,
                    exb_y_scale=exb_y_scale,
                )

            out = jax.vmap(_plane)(phi_eff, adv_eff, self.jacobian)

        # Optional metric-coupled X-Y ExB contribution used in field-aligned
        # BOUT/Hermes coordinates (poloidal flow term).
        if not bool(getattr(self.params, "exb_poloidal_flows", False)):
            return out
        if self.g23 is None or self.gxx is None or self.B is None:
            return out

        nz = int(self.grid.z.size)
        nx = int(self.grid.perp.nx)
        ny = int(self.grid.perp.ny)
        full_shape = (nz, nx, ny)

        def _as_field3(arr: jnp.ndarray) -> jnp.ndarray:
            arr = jnp.asarray(arr, dtype=jnp.float64)
            if arr.ndim == 0:
                return jnp.full(full_shape, arr, dtype=jnp.float64)
            if arr.ndim == 1:
                if arr.shape[0] != nz:
                    raise ValueError(f"metric 1D input must have shape ({nz},), got {arr.shape}")
                return jnp.broadcast_to(arr[:, None, None], full_shape)
            if arr.ndim == 2:
                if arr.shape == (nx, ny):
                    return jnp.broadcast_to(arr[None, :, :], full_shape)
                if arr.shape == (nz, nx):
                    return jnp.broadcast_to(arr[:, :, None], full_shape)
                if arr.shape == (nx, nz):
                    return jnp.broadcast_to(jnp.swapaxes(arr, 0, 1)[:, :, None], full_shape)
                raise ValueError("metric 2D input must have shape (nx, ny), (nz, nx), or (nx, nz).")
            if arr.ndim == 3 and arr.shape == full_shape:
                return arr
            raise ValueError(f"Unsupported metric shape {arr.shape}, expected {full_shape}.")

        use_shift = self.params.parallel_transform == "shifted" and self.shift_idx is not None
        J = _as_field3(self.jacobian)
        B = _as_field3(self.B)
        gxx = _as_field3(self.gxx)
        g23 = _as_field3(self.g23)
        dx_metric = (
            _as_field3(self.metric_dx)
            if self.metric_dx is not None
            else jnp.full(full_shape, float(self.grid.perp.dx), dtype=jnp.float64)
        )
        dy_metric = (
            _as_field3(self.metric_dy)
            if self.metric_dy is not None
            else jnp.full(full_shape, float(self.grid.dz), dtype=jnp.float64)
        )
        coeff = gxx * g23 / jnp.maximum(B * B, 1e-30)

        ddy_scheme = str(getattr(self.params, "exb_poloidal_ddy_scheme", "face")).lower()

        def _shift_clip(arr: jnp.ndarray, offset: int, axis: int) -> jnp.ndarray:
            n = arr.shape[axis]
            idx = jnp.clip(jnp.arange(n) + int(offset), 0, n - 1)
            return jnp.take(arr, idx, axis=axis)

        def _linear_edge_extrap(arr: jnp.ndarray, axis: int, *, lower: bool) -> jnp.ndarray:
            n = arr.shape[axis]
            if lower:
                edge = jnp.take(arr, 0, axis=axis)
                if n <= 1:
                    return edge
                near = jnp.take(arr, 1, axis=axis)
            else:
                edge = jnp.take(arr, n - 1, axis=axis)
                if n <= 1:
                    return edge
                near = jnp.take(arr, n - 2, axis=axis)
            return 2.0 * edge - near

        def _fromm_face_clip(
            arr: jnp.ndarray, vel: jnp.ndarray, axis: int, *, positive: bool = False
        ) -> jnp.ndarray:
            a_i = arr
            a_ip1 = _shift_clip(arr, +1, axis)
            a_im1 = _shift_clip(arr, -1, axis)
            a_ip2 = _shift_clip(arr, +2, axis)
            upwind_pos = a_i + 0.25 * (a_ip1 - a_im1)
            upwind_neg = a_ip1 - 0.25 * (a_ip2 - a_i)
            face_val = jnp.where(vel > 0.0, upwind_pos, upwind_neg)
            if positive:
                face_val = jnp.maximum(face_val, 0.0)
            return face_val

        def _fromm_parallel_boundary_flux(
            adv_arr: jnp.ndarray,
            vel: jnp.ndarray,
            *,
            side: str,
            positive: bool = False,
        ) -> jnp.ndarray:
            if side == "low":
                a0 = adv_arr[0]
                a1 = adv_arr[1] if adv_arr.shape[0] > 1 else a0
                ag = a0
                outflow = a0 - 0.25 * (a1 - ag)
                inflow = 0.5 * (ag + a0)
                face_val = jnp.where(vel < 0.0, outflow, inflow)
            else:
                a0 = adv_arr[-1]
                am1 = adv_arr[-2] if adv_arr.shape[0] > 1 else a0
                ag = a0
                outflow = a0 + 0.25 * (ag - am1)
                inflow = 0.5 * (ag + a0)
                face_val = jnp.where(vel > 0.0, outflow, inflow)
            if positive:
                face_val = jnp.maximum(face_val, 0.0)
            return vel * face_val

        # Hermes/BOUT poloidal ExB term has separate X and Y fluxes.
        # X-flux is evaluated in lab coordinates; Y-flux is evaluated in
        # field-aligned coordinates with shifted-metric transform.
        # Hermes/BOUT X-flux uses DDY(f) in lab coordinates (no
        # to/from-field-aligned transform here). Only the Y-flux branch
        # is evaluated in field-aligned coordinates.
        if self.grid.open_field_line:
            if ddy_scheme == "c2":
                dphi_dy = (
                    self._ddy_open_c2_metric(phi_eff, dy_metric)
                    if self.metric_dy is not None
                    else self._ddy_open_c2(phi_eff)
                )
            else:
                dphi_dy = (
                    self._dpar_open_metric(phi_eff, dy_metric)
                    if self.metric_dy is not None
                    else self._dpar_open(phi_eff)
                )
        else:
            if ddy_scheme == "c2":
                dphi_dy = (
                    self._ddy_periodic_c2_metric(phi_eff, dy_metric)
                    if self.metric_dy is not None
                    else self._ddy_periodic_c2(phi_eff)
                )
            else:
                dphi_dy = (
                    self._dpar_periodic_metric(phi_eff, dy_metric)
                    if self.metric_dy is not None
                    else self._dpar_periodic(phi_eff)
                )
        copy_x_grad = bool(getattr(self.params, "exb_copy_grad_x_boundary", True))
        if copy_x_grad and int(getattr(bc_phi, "kind_x", 0)) != 0:
            dphi_dy = dphi_dy.at[:, 0, :].set(dphi_dy[:, 1, :])
            dphi_dy = dphi_dy.at[:, -1, :].set(dphi_dy[:, -2, :])
        coeff_dphi_dy = coeff * dphi_dy
        coeff_dphi_dy_ip1 = _shift_clip(coeff_dphi_dy, +1, axis=1)
        J_ip1 = _shift_clip(J, +1, axis=1)
        vx_face = 0.5 * (J + J_ip1) * 0.5 * (coeff_dphi_dy + coeff_dphi_dy_ip1)
        n_face_x = _fromm_face_clip(adv_eff, vx_face, axis=1, positive=positive)
        flux_x = vx_face * n_face_x
        periodic_x = int(getattr(bc_adv, "kind_x", 0)) == 0
        if periodic_x:
            flux_l = _shift_clip(flux_x, -1, axis=1)
        else:
            kind = int(getattr(bc_adv, "kind_x", 0))
            val = float(getattr(bc_adv, "x_value", 0.0))
            grad = float(getattr(bc_adv, "x_grad", 0.0))
            coeff_left = coeff_dphi_dy[:, 0, :]
            coeff_right = coeff_dphi_dy[:, -1, :]
            coeff_left_ghost = _linear_edge_extrap(coeff_dphi_dy, axis=1, lower=True)
            coeff_right_ghost = _linear_edge_extrap(coeff_dphi_dy, axis=1, lower=False)
            J_left = J[:, 0, :]
            J_right = J[:, -1, :]
            J_left_ghost = _linear_edge_extrap(J, axis=1, lower=True)
            J_right_ghost = _linear_edge_extrap(J, axis=1, lower=False)
            vx_left = 0.5 * (J_left + J_left_ghost) * 0.5 * (coeff_left + coeff_left_ghost)
            vx_right = 0.5 * (J_right + J_right_ghost) * 0.5 * (coeff_right + coeff_right_ghost)
            flux_left_b = _fromm_x_boundary_flux(
                adv_eff, vx_left, side="left", kind=kind, value=val, grad=grad, positive=positive
            )
            flux_right_b = _fromm_x_boundary_flux(
                adv_eff,
                vx_right,
                side="right",
                kind=kind,
                value=val,
                grad=grad,
                positive=positive,
            )
            flux_x = flux_x.at[:, -1, :].set(flux_right_b)
            flux_l = jnp.concatenate([flux_left_b[:, None, :], flux_x[:, :-1, :]], axis=1)
        div_x = (flux_x - flux_l) / (jnp.maximum(J, 1e-30) * jnp.maximum(dx_metric, 1e-30))

        if self.metric_dx is None:
            dphi_dx = jax.vmap(lambda p: self.perp_ops.ddx(p))(phi_eff)
        else:
            dphi_dx = self._ddx_metric_c2(phi_eff, dx_metric)
        if copy_x_grad and int(getattr(bc_phi, "kind_x", 0)) != 0:
            dphi_dx = dphi_dx.at[:, 0, :].set(dphi_dx[:, 1, :])
            dphi_dx = dphi_dx.at[:, -1, :].set(dphi_dx[:, -2, :])
        if use_shift:
            dphi_dx_fa = self.to_field_aligned(dphi_dx)
            adv_fa = self.to_field_aligned(adv_eff)
            J_fa = self.to_field_aligned(J)
            coeff_fa = self.to_field_aligned(coeff)
            dy_fa = self.to_field_aligned(dy_metric)
        else:
            dphi_dx_fa = dphi_dx
            adv_fa = adv_eff
            J_fa = J
            coeff_fa = coeff
            dy_fa = dy_metric

        coeff_dphi_dx_fa = coeff_fa * dphi_dx_fa
        coeff_dphi_dx_jp1 = _shift_clip(coeff_dphi_dx_fa, +1, axis=0)
        J_jp1 = _shift_clip(J_fa, +1, axis=0)
        vy_up = -0.5 * (J_fa + J_jp1) * 0.5 * (coeff_dphi_dx_fa + coeff_dphi_dx_jp1)
        n_face_y = _fromm_face_clip(adv_fa, vy_up, axis=0, positive=positive)
        flux_up = vy_up * n_face_y

        coeff_low = coeff_dphi_dx_fa[0]
        coeff_high = coeff_dphi_dx_fa[-1]
        J_low = J_fa[0]
        J_high = J_fa[-1]
        vy_low = -(J_low * coeff_low)
        vy_high = -(J_high * coeff_high)
        if self.grid.open_field_line:
            vy_low = jnp.minimum(vy_low, 0.0)
            vy_high = jnp.maximum(vy_high, 0.0)
        flux_low_b = _fromm_parallel_boundary_flux(adv_fa, vy_low, side="low", positive=positive)
        flux_high_b = _fromm_parallel_boundary_flux(adv_fa, vy_high, side="high", positive=positive)
        flux_up = flux_up.at[-1].set(flux_high_b)
        flux_down = jnp.concatenate([flux_low_b[None, ...], flux_up[:-1]], axis=0)

        div_y_fa = (flux_up - flux_down) / (jnp.maximum(J_fa, 1e-30) * jnp.maximum(dy_fa, 1e-30))
        div_y = self.from_field_aligned(div_y_fa) if use_shift else div_y_fa

        poloidal_scale = float(getattr(self.params, "exb_poloidal_scale", 1.0))
        poloidal_x_scale = float(getattr(self.params, "exb_poloidal_x_scale", 1.0))
        poloidal_y_scale = float(getattr(self.params, "exb_poloidal_y_scale", 1.0))
        poloidal = poloidal_scale * (poloidal_x_scale * div_x + poloidal_y_scale * div_y)
        return out + poloidal

    def _dpar_periodic(self, f: jnp.ndarray) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * self.grid.dz)

    def _ddy_periodic_c2(self, f: jnp.ndarray) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * self.grid.dz)

    def _ddy_periodic_c2_metric(self, f: jnp.ndarray, ds: jnp.ndarray) -> jnp.ndarray:
        num = jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)
        den = jnp.roll(ds, -1, axis=0) + jnp.roll(ds, 1, axis=0)
        return num / jnp.maximum(den, 1e-30)

    def _ddy_open_c2(self, f: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros_like(f)
        if f.shape[0] <= 1:
            return out
        if f.shape[0] == 2:
            out = out.at[0].set((f[1] - f[0]) / self.grid.dz)
            out = out.at[1].set((f[1] - f[0]) / self.grid.dz)
            return out
        out = out.at[1:-1].set((f[2:] - f[:-2]) / (2.0 * self.grid.dz))
        out = out.at[0].set((-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * self.grid.dz))
        out = out.at[-1].set((3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * self.grid.dz))
        return out

    def _ddy_open_c2_metric(self, f: jnp.ndarray, ds: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros_like(f)
        if f.shape[0] <= 1:
            return out
        if f.shape[0] == 2:
            den = 0.5 * (ds[0] + ds[1])
            slope = (f[1] - f[0]) / jnp.maximum(den, 1e-30)
            out = out.at[0].set(slope)
            out = out.at[1].set(slope)
            return out
        out = out.at[1:-1].set((f[2:] - f[:-2]) / jnp.maximum(2.0 * ds[1:-1], 1e-30))
        out = out.at[0].set((-3.0 * f[0] + 4.0 * f[1] - f[2]) / jnp.maximum(2.0 * ds[0], 1e-30))
        out = out.at[-1].set((3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / jnp.maximum(2.0 * ds[-1], 1e-30))
        return out

    def _dpar_open(self, f: jnp.ndarray) -> jnp.ndarray:
        face = 0.5 * (f[1:] + f[:-1])
        df = jnp.zeros_like(f)
        df = df.at[1:-1].set((face[1:] - face[:-1]) / self.grid.dz)
        df = df.at[0].set((face[0] - f[0]) / self.grid.dz)
        df = df.at[-1].set((f[-1] - face[-1]) / self.grid.dz)
        return df

    def _dpar_periodic_metric(self, f: jnp.ndarray, ds: jnp.ndarray) -> jnp.ndarray:
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / jnp.maximum(2.0 * ds, 1e-30)

    def _dpar_open_metric(self, f: jnp.ndarray, ds: jnp.ndarray) -> jnp.ndarray:
        face = 0.5 * (f[1:] + f[:-1])
        df = jnp.zeros_like(f)
        df = df.at[1:-1].set((face[1:] - face[:-1]) / jnp.maximum(ds[1:-1], 1e-30))
        df = df.at[0].set((face[0] - f[0]) / jnp.maximum(ds[0], 1e-30))
        df = df.at[-1].set((f[-1] - face[-1]) / jnp.maximum(ds[-1], 1e-30))
        return df

    def _ddx_metric_c2(self, f: jnp.ndarray, dx: jnp.ndarray) -> jnp.ndarray:
        xp = jnp.take(f, jnp.clip(jnp.arange(f.shape[1]) + 1, 0, f.shape[1] - 1), axis=1)
        xm = jnp.take(f, jnp.clip(jnp.arange(f.shape[1]) - 1, 0, f.shape[1] - 1), axis=1)
        out = (xp - xm) / jnp.maximum(2.0 * dx, 1e-30)
        if f.shape[1] > 1:
            out = out.at[:, 0, :].set((f[:, 1, :] - f[:, 0, :]) / jnp.maximum(dx[:, 0, :], 1e-30))
            out = out.at[:, -1, :].set(
                (f[:, -1, :] - f[:, -2, :]) / jnp.maximum(dx[:, -1, :], 1e-30)
            )
        return out

    def _dpar_limited(self, f: jnp.ndarray, limiter: str) -> jnp.ndarray:
        df = f[1:] - f[:-1]
        df_b = df[:-1]
        df_f = df[1:]

        def _minmod(a, b):
            s = 0.5 * (jnp.sign(a) + jnp.sign(b))
            return s * jnp.minimum(jnp.abs(a), jnp.abs(b))

        if limiter == "mc":
            slope = _minmod(_minmod(2.0 * df_b, 2.0 * df_f), 0.5 * (df_b + df_f))
        else:
            slope = _minmod(df_b, df_f)

        slope_full = jnp.zeros_like(f)
        slope_full = slope_full.at[1:-1].set(slope)
        slope_full = slope_full.at[0].set(df[0])
        slope_full = slope_full.at[-1].set(df[-1])
        return slope_full / self.grid.dz

    def dpar(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray:
        _ = bc_kind
        limiter = str(self.params.parallel_limiter).lower()
        if self.params.parallel_transform == "shifted" and self.shift_idx is not None:
            f = self.to_field_aligned(f)
        if limiter != "none" and self.grid.open_field_line:
            df = self._dpar_limited(f, limiter)
        else:
            df = self._dpar_open(f) if self.grid.open_field_line else self._dpar_periodic(f)
        if self.params.parallel_transform == "shifted" and self.shift_idx is not None:
            df = self.from_field_aligned(df)
        return df * self.dpar_factor * float(self.params.parallel_sign)

    def div_par(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray:
        """Conservative parallel divergence: (1/J) d/dz (J f)."""
        _ = bc_kind
        if self.jacobian is None:
            return self.dpar(f, bc_kind=bc_kind)
        use_shift = self.params.parallel_transform == "shifted" and self.shift_idx is not None
        if use_shift:
            f = self.to_field_aligned(f)
        J = jnp.asarray(self.jacobian)
        if J.ndim == 1:
            J = J[:, None, None]
        elif J.ndim == 2:
            J = J[None, :, :]
        limiter = str(self.params.parallel_limiter).lower()
        flux = f * J
        if limiter != "none" and self.grid.open_field_line:
            df = self._dpar_limited(flux, limiter)
        else:
            df = self._dpar_open(flux) if self.grid.open_field_line else self._dpar_periodic(flux)
        out = (df * self.dpar_factor * float(self.params.parallel_sign)) / jnp.maximum(J, 1e-30)
        if use_shift:
            out = self.from_field_aligned(out)
        return out

    def _shift_binormal(
        self,
        f: jnp.ndarray,
        sign: float,
        *,
        exclude_x_boundaries: bool = False,
    ) -> jnp.ndarray:
        if self.shift_idx is None:
            return f
        ny = f.shape[-1]
        shift = float(sign) * self.shift_idx
        interp = str(getattr(self.params, "parallel_shift_interp", "linear")).lower()
        if interp == "spectral":
            k = jnp.fft.fftfreq(ny, d=1.0)
            phase = jnp.exp((2.0j * jnp.pi) * k[None, None, :] * shift[..., None])
            f_hat = jnp.fft.fft(f, axis=-1)
            shifted = jnp.real(jnp.fft.ifft(f_hat * phase, axis=-1))
        else:
            y = jnp.arange(ny, dtype=jnp.float64)
            y_src = (y[None, None, :] + shift[..., None]) % float(ny)
            y0 = jnp.floor(y_src).astype(jnp.int32)
            y1 = (y0 + 1) % ny
            frac = y_src - y0
            f0 = jnp.take_along_axis(f, y0, axis=-1)
            f1 = jnp.take_along_axis(f, y1, axis=-1)
            shifted = (1.0 - frac) * f0 + frac * f1
        if exclude_x_boundaries and f.shape[1] > 1:
            shifted = shifted.at[:, 0, :].set(f[:, 0, :])
            shifted = shifted.at[:, -1, :].set(f[:, -1, :])
        return shifted

    def to_field_aligned(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.shift_idx is None:
            return f
        return self._shift_binormal(f, 1.0)

    def from_field_aligned(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.shift_idx is None:
            return f
        return self._shift_binormal(f, -1.0)

    def to_field_aligned_nox(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.shift_idx is None:
            return f
        return self._shift_binormal(f, 1.0, exclude_x_boundaries=True)

    def from_field_aligned_nox(self, f: jnp.ndarray) -> jnp.ndarray:
        if self.shift_idx is None:
            return f
        return self._shift_binormal(f, -1.0, exclude_x_boundaries=True)

    def d2par(self, f: jnp.ndarray, *, bc_kind: str | None = None) -> jnp.ndarray:
        return self.dpar(self.dpar(f, bc_kind=bc_kind), bc_kind=bc_kind)

    def curvature(self, f: jnp.ndarray) -> jnp.ndarray:
        if not self.params.curvature_on or self.params.curvature_coeff == 0.0:
            return jnp.zeros_like(f)
        curv = (self.curv_x * self.ddx(f)) + (self.curv_y * self.ddy(f))
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
