"""Literal Hermes/BOUT finite-volume diffusion operators."""

from __future__ import annotations

import jax.numpy as jnp

from .boundary import apply_free_o2_field3d
from .exb import (
    _as_field_aligned_metric2d,
    _as_runtime_metric2d,
    _from_field_aligned_local,
    _pad_runtime_field,
    _pad_runtime_metric,
    _shift,
    _to_field_aligned_local,
)
from .types import FieldAlignedLocalLayout


def div_a_grad_perp_local(
    a: jnp.ndarray,
    f: jnp.ndarray,
    *,
    jacobian: jnp.ndarray | float,
    dx: jnp.ndarray | float,
    dy: jnp.ndarray | float,
    dz: jnp.ndarray | float,
    g11: jnp.ndarray | float,
    g23: jnp.ndarray | float,
    g_22: jnp.ndarray | float,
    g_23: jnp.ndarray | float,
    g33: jnp.ndarray | float,
    bxy: jnp.ndarray | float,
    z_shift: jnp.ndarray | float,
    zlength: float,
    layout: FieldAlignedLocalLayout,
    interp: str = "spectral",
    periodic_binormal: bool = True,
) -> jnp.ndarray:
    """Mirror `FV::Div_a_Grad_perp` on local guard-inclusive fields.

    Source of truth:
    `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/fv_ops.cxx`
    """

    a_arr = jnp.asarray(a, dtype=jnp.float64)
    f_arr = jnp.asarray(f, dtype=jnp.float64)
    if a_arr.shape != f_arr.shape or a_arr.ndim != 3:
        raise ValueError(
            f"Local mirror Div_a_Grad_perp expects matching `(npar, nx, nbinorm)` fields, got {a_arr.shape} and {f_arr.shape}."
        )
    layout.validate(tuple(int(v) for v in a_arr.shape))
    npar, nx, nbinorm = (int(v) for v in a_arr.shape)

    def _metric(name: str, arr: jnp.ndarray | float) -> jnp.ndarray:
        return jnp.broadcast_to(
            _as_field_aligned_metric2d(arr, npar=npar, nx=nx, nbinorm=nbinorm, name=name)[
                :, :, None
            ],
            a_arr.shape,
        )

    J = _metric("jacobian", jacobian)
    dx_arr = _metric("dx", dx)
    dy_arr = _metric("dy", dy)
    dz_arr = _metric("dz", dz)
    g11_arr = _metric("g11", g11)
    g23_arr = _metric("g23", g23)
    g_23_arr = _metric("g_23", g_23)
    g33_arr = _metric("g33", g33)
    bxy_arr = _metric("bxy", bxy)

    p_idx = jnp.arange(npar)
    x_face_idx = jnp.arange(nx - 1)
    x_cell_idx = jnp.arange(nx)

    x_face_mask = (
        (p_idx[:, None, None] >= int(layout.pstart))
        & (p_idx[:, None, None] <= int(layout.pend))
        & (x_face_idx[None, :, None] >= int(layout.xstart - 1))
        & (x_face_idx[None, :, None] <= int(layout.xend))
    )
    x_flux = (
        0.5
        * (a_arr[:, :-1, :] + a_arr[:, 1:, :])
        * (J[:, :-1, :] * g11_arr[:, :-1, :] + J[:, 1:, :] * g11_arr[:, 1:, :])
        * (f_arr[:, 1:, :] - f_arr[:, :-1, :])
        / jnp.maximum(dx_arr[:, :-1, :] + dx_arr[:, 1:, :], 1e-30)
    )
    x_flux = jnp.where(x_face_mask, x_flux, 0.0)
    result_x = jnp.zeros_like(f_arr)
    result_x = result_x.at[:, :-1, :].add(
        x_flux / jnp.maximum(dx_arr[:, :-1, :] * J[:, :-1, :], 1e-30)
    )
    result_x = result_x.at[:, 1:, :].add(
        -x_flux / jnp.maximum(dx_arr[:, 1:, :] * J[:, 1:, :], 1e-30)
    )

    def _to_aligned(arr: jnp.ndarray) -> jnp.ndarray:
        return _to_field_aligned_local(
            arr,
            z_shift=z_shift,
            zlength=zlength,
            open_field_line=layout.open_field_line,
            interp=interp,
        )

    a_fa = _to_aligned(a_arr)
    f_fa = _to_aligned(f_arr)
    J_fa = _to_aligned(J)
    dy_fa = _to_aligned(dy_arr)
    dz_fa = _to_aligned(dz_arr)
    g23_fa = _to_aligned(g23_arr)
    g_23_fa = _to_aligned(g_23_arr)
    g33_fa = _to_aligned(g33_arr)
    bxy_fa = _to_aligned(bxy_arr)

    a_c = a_fa[1:-1, :, :]
    a_up = a_fa[2:, :, :]
    a_down = a_fa[:-2, :, :]
    f_c = f_fa[1:-1, :, :]
    f_up = f_fa[2:, :, :]
    f_down = f_fa[:-2, :, :]
    J_c = J_fa[1:-1, :, :]
    J_up = J_fa[2:, :, :]
    J_down = J_fa[:-2, :, :]
    dy_c = dy_fa[1:-1, :, :]
    dy_up = dy_fa[2:, :, :]
    dy_down = dy_fa[:-2, :, :]
    dz_c = dz_fa[1:-1, :, :]
    dz_up = dz_fa[2:, :, :]
    dz_down = dz_fa[:-2, :, :]
    g23_c = g23_fa[1:-1, :, :]
    g23_up = g23_fa[2:, :, :]
    g23_down = g23_fa[:-2, :, :]
    g_23_c = g_23_fa[1:-1, :, :]
    g_23_up = g_23_fa[2:, :, :]
    g_23_down = g_23_fa[:-2, :, :]
    g33_c = g33_fa[1:-1, :, :]
    bxy_c = bxy_fa[1:-1, :, :]
    bxy_up = bxy_fa[2:, :, :]
    bxy_down = bxy_fa[:-2, :, :]

    f_c_kp = _shift(f_c, +1, axis=2, periodic=periodic_binormal)
    f_c_km = _shift(f_c, -1, axis=2, periodic=periodic_binormal)
    f_up_kp = _shift(f_up, +1, axis=2, periodic=periodic_binormal)
    f_up_km = _shift(f_up, -1, axis=2, periodic=periodic_binormal)
    f_down_kp = _shift(f_down, +1, axis=2, periodic=periodic_binormal)
    f_down_km = _shift(f_down, -1, axis=2, periodic=periodic_binormal)
    a_c_kp = _shift(a_c, +1, axis=2, periodic=periodic_binormal)
    J_c_kp = _shift(J_c, +1, axis=2, periodic=periodic_binormal)
    g33_c_kp = _shift(g33_c, +1, axis=2, periodic=periodic_binormal)
    f_up_self = f_up
    f_down_self = f_down

    coef_up = 0.5 * (
        g_23_c / jnp.maximum((J_c * bxy_c) ** 2, 1e-30)
        + g_23_up / jnp.maximum((J_up * bxy_up) ** 2, 1e-30)
    )
    dfdz_up = 0.5 * (f_c_kp - f_c_km + f_up_kp - f_up_km) / jnp.maximum(dz_c + dz_up, 1e-30)
    dfdy_up = 2.0 * (f_up - f_c) / jnp.maximum(dy_up + dy_c, 1e-30)
    y_flux_up = 0.25 * (a_c + a_up) * (J_c * g23_c + J_up * g23_up) * (dfdz_up - coef_up * dfdy_up)

    coef_down = 0.5 * (
        g_23_c / jnp.maximum((J_c * bxy_c) ** 2, 1e-30)
        + g_23_down / jnp.maximum((J_down * bxy_down) ** 2, 1e-30)
    )
    dfdz_down = 0.5 * (f_c_kp - f_c_km + f_down_kp - f_down_km) / jnp.maximum(dz_c + dz_down, 1e-30)
    dfdy_down = 2.0 * (f_c - f_down) / jnp.maximum(dy_c + dy_down, 1e-30)
    y_flux_down = (
        0.25
        * (a_c + a_down)
        * (J_c * g23_c + J_down * g23_down)
        * (dfdz_down - coef_down * dfdy_down)
    )

    coef_z = (
        g_23_c
        / jnp.maximum(dy_up + 2.0 * dy_c + dy_down, 1e-30)
        / jnp.maximum((J_c * bxy_c) ** 2, 1e-30)
    )
    z_flux = (
        0.25
        * (a_c + a_c_kp)
        * (J_c * g33_c + J_c_kp * g33_c_kp)
        * (
            (f_c_kp - f_c) / jnp.maximum(dz_c, 1e-30)
            - coef_z * (f_up_self + f_up_kp - f_down_self - f_down_kp)
        )
    )

    p_mid_idx = jnp.arange(1, npar - 1)
    yz_mask = (
        (p_mid_idx[:, None, None] >= int(layout.pstart))
        & (p_mid_idx[:, None, None] <= int(layout.pend))
        & (x_cell_idx[None, :, None] >= int(layout.xstart))
        & (x_cell_idx[None, :, None] <= int(layout.xend))
    )
    yz_mid = (
        y_flux_up / jnp.maximum(dy_c * J_c, 1e-30)
        - y_flux_down / jnp.maximum(dy_c * J_c, 1e-30)
        + z_flux / jnp.maximum(J_c * dz_c, 1e-30)
        - _shift(z_flux, +1, axis=2, periodic=periodic_binormal) / jnp.maximum(J_c * dz_c, 1e-30)
    )
    yz_mid = jnp.where(yz_mask, yz_mid, 0.0)

    yz_fa = jnp.zeros_like(f_fa)
    yz_fa = yz_fa.at[1:-1, :, :].set(yz_mid)
    yz_unaligned = _from_field_aligned_local(
        yz_fa,
        z_shift=z_shift,
        zlength=zlength,
        open_field_line=layout.open_field_line,
        interp=interp,
    )
    return result_x + yz_unaligned


def div_a_grad_perp(
    a: jnp.ndarray,
    f: jnp.ndarray,
    *,
    jacobian: jnp.ndarray | float,
    dx: jnp.ndarray | float,
    dy: jnp.ndarray | float,
    dz: jnp.ndarray | float,
    g11: jnp.ndarray | float,
    g23: jnp.ndarray | float,
    g_22: jnp.ndarray | float,
    g_23: jnp.ndarray | float,
    g33: jnp.ndarray | float,
    bxy: jnp.ndarray | float,
    z_shift: jnp.ndarray | float,
    zlength: float,
    bc_kind_x: int = 2,
    bc_value_x: float = 0.0,
    bc_grad_x: float = 0.0,
    coeff_bc_kind_x: int = 2,
    coeff_bc_value_x: float = 0.0,
    coeff_bc_grad_x: float = 0.0,
    interp: str = "spectral",
    periodic_parallel: bool = False,
    periodic_binormal: bool = True,
    lower_boundary_open: bool = True,
    upper_boundary_open: bool = True,
    apply_free_o2: bool = False,
) -> jnp.ndarray:
    """Runtime mirror wrapper for `FV::Div_a_Grad_perp`.

    The active Stage 1 path uses the single-device `(nz, nx, ny)` storage
    contract. This helper reconstructs the local guard-inclusive arrays,
    optionally applies literal `free_o2` boundary updates to the differentiated
    field, evaluates the local mirror operator, and slices the physical cells
    back out.
    """

    a_arr = jnp.asarray(a, dtype=jnp.float64)
    f_arr = jnp.asarray(f, dtype=jnp.float64)
    if a_arr.shape != f_arr.shape or a_arr.ndim != 3:
        raise ValueError(
            f"Runtime mirror Div_a_Grad_perp expects matching `(nz, nx, ny)` fields, got {a_arr.shape} and {f_arr.shape}."
        )
    nz, nx, ny = (int(v) for v in a_arr.shape)
    layout = FieldAlignedLocalLayout(
        pstart=2,
        pend=nz + 1,
        xstart=2,
        xend=nx + 1,
        open_field_line=not periodic_parallel,
    )
    interior = (
        slice(layout.pstart, layout.pend + 1),
        slice(layout.xstart, layout.xend + 1),
        slice(None),
    )

    dx2d = _as_runtime_metric2d(dx, nz=nz, nx=nx, ny=ny, name="dx")
    dy2d = _as_runtime_metric2d(dy, nz=nz, nx=nx, ny=ny, name="dy")
    dz2d = _as_runtime_metric2d(dz, nz=nz, nx=nx, ny=ny, name="dz")
    J2d = _as_runtime_metric2d(jacobian, nz=nz, nx=nx, ny=ny, name="jacobian")
    g11_2d = _as_runtime_metric2d(g11, nz=nz, nx=nx, ny=ny, name="g11")
    g23_2d = _as_runtime_metric2d(g23, nz=nz, nx=nx, ny=ny, name="g23")
    g_22_2d = _as_runtime_metric2d(g_22, nz=nz, nx=nx, ny=ny, name="g_22")
    g_23_2d = _as_runtime_metric2d(g_23, nz=nz, nx=nx, ny=ny, name="g_23")
    g33_2d = _as_runtime_metric2d(g33, nz=nz, nx=nx, ny=ny, name="g33")
    bxy_2d = _as_runtime_metric2d(bxy, nz=nz, nx=nx, ny=ny, name="bxy")
    zshift_2d = _as_runtime_metric2d(z_shift, nz=nz, nx=nx, ny=ny, name="z_shift")

    a_local = _pad_runtime_field(
        a_arr,
        dx=dx2d,
        dy=dy2d,
        bc_kind_x=int(coeff_bc_kind_x),
        bc_value_x=coeff_bc_value_x,
        bc_grad_x=coeff_bc_grad_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    f_local = _pad_runtime_field(
        f_arr,
        dx=dx2d,
        dy=dy2d,
        bc_kind_x=int(bc_kind_x),
        bc_value_x=bc_value_x,
        bc_grad_x=bc_grad_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    if apply_free_o2:
        f_local = apply_free_o2_field3d(
            f_local,
            axis=1,
            interior_start=layout.xstart,
            interior_end=layout.xend,
            guard_width=layout.x_guards,
        )
        if not periodic_parallel:
            f_local = apply_free_o2_field3d(
                f_local,
                axis=0,
                interior_start=layout.pstart,
                interior_end=layout.pend,
                guard_width=layout.p_guards,
                apply_lower=lower_boundary_open,
                apply_upper=upper_boundary_open,
            )

    J_local = _pad_runtime_metric(
        J2d,
        periodic_x=int(bc_kind_x) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    dx_local = _pad_runtime_metric(
        dx2d,
        periodic_x=int(bc_kind_x) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    dy_local = _pad_runtime_metric(
        dy2d,
        periodic_x=int(bc_kind_x) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    dz_local = _pad_runtime_metric(
        dz2d,
        periodic_x=int(bc_kind_x) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g11_local = _pad_runtime_metric(
        g11_2d,
        periodic_x=int(bc_kind_x) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g23_local = _pad_runtime_metric(
        g23_2d,
        periodic_x=int(bc_kind_x) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g_22_local = _pad_runtime_metric(
        g_22_2d,
        periodic_x=int(bc_kind_x) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g_23_local = _pad_runtime_metric(
        g_23_2d,
        periodic_x=int(bc_kind_x) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g33_local = _pad_runtime_metric(
        g33_2d,
        periodic_x=int(bc_kind_x) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    bxy_local = _pad_runtime_metric(
        bxy_2d,
        periodic_x=int(bc_kind_x) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    zshift_local = _pad_runtime_metric(
        zshift_2d,
        periodic_x=int(bc_kind_x) == 0,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )

    result_local = div_a_grad_perp_local(
        a_local,
        f_local,
        jacobian=J_local,
        dx=dx_local,
        dy=dy_local,
        dz=dz_local,
        g11=g11_local,
        g23=g23_local,
        g_22=g_22_local,
        g_23=g_23_local,
        g33=g33_local,
        bxy=bxy_local,
        z_shift=zshift_local,
        zlength=zlength,
        layout=layout,
        interp=interp,
        periodic_binormal=periodic_binormal,
    )
    return result_local[interior]
