"""Literal Hermes/BOUT perpendicular Laplacian for strict parity work."""

from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.bc import BC2D

from .exb import _as_runtime_metric2d, _pad_runtime_field, _pad_runtime_metric


def _shift_binormal(arr: jnp.ndarray, offset: int, *, periodic: bool) -> jnp.ndarray:
    if periodic:
        return jnp.roll(arr, int(offset), axis=2)
    n = arr.shape[2]
    idx = jnp.clip(jnp.arange(n) + int(offset), 0, n - 1)
    return jnp.take(arr, idx, axis=2)


def _ddx_metric_from_padded(metric_local: jnp.ndarray, dx_local: jnp.ndarray) -> jnp.ndarray:
    return (
        0.5
        * (metric_local[2:-2, 3:-1] - metric_local[2:-2, 1:-3])
        / jnp.maximum(
            dx_local[2:-2, 2:-2],
            1e-30,
        )
    )


def _ddy_metric_from_padded(metric_local: jnp.ndarray, dy_local: jnp.ndarray) -> jnp.ndarray:
    return (
        0.5
        * (metric_local[3:-1, 2:-2] - metric_local[1:-3, 2:-2])
        / jnp.maximum(
            dy_local[2:-2, 2:-2],
            1e-30,
        )
    )


def _runtime_coeff_2d(
    arr: jnp.ndarray | float | None,
    *,
    nz: int,
    nx: int,
    ny: int,
    name: str,
    fill: float = 0.0,
) -> jnp.ndarray:
    if arr is None:
        return jnp.full((nz, nx), float(fill), dtype=jnp.float64)
    return _as_runtime_metric2d(arr, nz=nz, nx=nx, ny=ny, name=name)


def derive_delp2_coefficients(
    *,
    geom,
    nz: int,
    nx: int,
    ny: int,
    periodic_x: bool,
    periodic_parallel: bool,
    lower_boundary_open: bool,
    upper_boundary_open: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return literal `G1`, `G3`, and `d1_dx` coefficient planes.

    Source of truth:
    `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/coordinates.cxx`

    `G1` and `G3` are loaded directly from geometry if present. Otherwise this
    falls back to the source formulas using the runtime metric planes:

    `G1 = (DDX(J*g11) + DDY(J*g12) + DDZ(J*g13)) / J`
    `G3 = (DDX(J*g13) + DDY(J*g23) + DDZ(J*g33)) / J`

    In the current strict axisymmetric path, the perpendicular metric
    coefficients are 2D planes `(npar, nx)` with no binormal dependence, so the
    `DDZ(...)` terms vanish and `g12` is zero.
    """

    j2d = _runtime_coeff_2d(getattr(geom, "jacobian", None), nz=nz, nx=nx, ny=ny, name="J")
    dx2d = _runtime_coeff_2d(getattr(geom, "metric_dx", None), nz=nz, nx=nx, ny=ny, name="dx")
    dy2d = _runtime_coeff_2d(getattr(geom, "metric_dy", None), nz=nz, nx=nx, ny=ny, name="dy")
    g11_2d = _runtime_coeff_2d(getattr(geom, "gxx", None), nz=nz, nx=nx, ny=ny, name="g11")
    g13_2d = _runtime_coeff_2d(getattr(geom, "gxy", None), nz=nz, nx=nx, ny=ny, name="g13")
    g23_2d = _runtime_coeff_2d(getattr(geom, "g23", None), nz=nz, nx=nx, ny=ny, name="g23")

    dx_local = _pad_runtime_metric(
        dx2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    dy_local = _pad_runtime_metric(
        dy2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )

    g1_src = getattr(geom, "G1", None)
    if g1_src is not None:
        g1_2d = _runtime_coeff_2d(g1_src, nz=nz, nx=nx, ny=ny, name="G1")
    else:
        tmp_local = _pad_runtime_metric(
            j2d * g11_2d,
            periodic_x=periodic_x,
            periodic_parallel=periodic_parallel,
            lower_boundary_open=lower_boundary_open,
            upper_boundary_open=upper_boundary_open,
        )
        g1_2d = _ddx_metric_from_padded(tmp_local, dx_local) / jnp.maximum(j2d, 1e-30)

    g3_src = getattr(geom, "G3", None)
    if g3_src is not None:
        g3_2d = _runtime_coeff_2d(g3_src, nz=nz, nx=nx, ny=ny, name="G3")
    else:
        g13_local = _pad_runtime_metric(
            j2d * g13_2d,
            periodic_x=periodic_x,
            periodic_parallel=periodic_parallel,
            lower_boundary_open=lower_boundary_open,
            upper_boundary_open=upper_boundary_open,
        )
        g23_local = _pad_runtime_metric(
            j2d * g23_2d,
            periodic_x=periodic_x,
            periodic_parallel=periodic_parallel,
            lower_boundary_open=lower_boundary_open,
            upper_boundary_open=upper_boundary_open,
        )
        g3_num = _ddx_metric_from_padded(g13_local, dx_local) + _ddy_metric_from_padded(
            g23_local,
            dy_local,
        )
        g3_2d = g3_num / jnp.maximum(j2d, 1e-30)

    d1_dx_src = getattr(geom, "d1_dx", None)
    if d1_dx_src is not None:
        d1_dx_2d = _runtime_coeff_2d(d1_dx_src, nz=nz, nx=nx, ny=ny, name="d1_dx")
    else:
        inv_dx_local = _pad_runtime_metric(
            1.0 / jnp.maximum(dx2d, 1e-30),
            periodic_x=periodic_x,
            periodic_parallel=periodic_parallel,
            lower_boundary_open=lower_boundary_open,
            upper_boundary_open=upper_boundary_open,
        )
        d1_dx_2d = _ddx_metric_from_padded(inv_dx_local, dx_local)

    return g1_2d, g3_2d, d1_dx_2d


def delp2_runtime(
    field: jnp.ndarray,
    *,
    geom,
    bc_field: BC2D,
) -> jnp.ndarray:
    """Mirror `Coordinates::Delp2(Field3D)` on runtime `(npar, nx, nbinorm)` arrays.

    Source of truth:
    - `/Users/rogerio/local/hermes-3/external/BOUT-dev/src/mesh/coordinates.cxx`
    - `/Users/rogerio/local/hermes-3/external/BOUT-dev/include/bout/single_index_ops.hxx`
    """

    f_arr = jnp.asarray(field, dtype=jnp.float64)
    if f_arr.ndim != 3:
        raise ValueError(
            f"Hermes mirror Delp2 expects `(npar, nx, nbinorm)` arrays, got {f_arr.shape}."
        )
    nz, nx, ny = (int(v) for v in f_arr.shape)

    dx2d = _runtime_coeff_2d(getattr(geom, "metric_dx", None), nz=nz, nx=nx, ny=ny, name="dx")
    dy2d = _runtime_coeff_2d(getattr(geom, "metric_dy", None), nz=nz, nx=nx, ny=ny, name="dy")
    dz2d = _runtime_coeff_2d(getattr(geom, "metric_dz", None), nz=nz, nx=nx, ny=ny, name="dz")
    g11_2d = _runtime_coeff_2d(getattr(geom, "gxx", None), nz=nz, nx=nx, ny=ny, name="g11")
    g13_2d = _runtime_coeff_2d(getattr(geom, "gxy", None), nz=nz, nx=nx, ny=ny, name="g13")
    g33_2d = _runtime_coeff_2d(getattr(geom, "gyy", None), nz=nz, nx=nx, ny=ny, name="g33")

    periodic_x = int(getattr(bc_field, "kind_x", 0)) == 0
    periodic_parallel = not bool(getattr(getattr(geom, "grid", None), "open_field_line", False))
    lower_boundary_open = bool(getattr(getattr(geom, "grid", None), "open_field_line", False))
    upper_boundary_open = lower_boundary_open
    periodic_binormal = int(getattr(bc_field, "kind_y", 0)) == 0

    g1_2d, g3_2d, d1_dx_2d = derive_delp2_coefficients(
        geom=geom,
        nz=nz,
        nx=nx,
        ny=ny,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )

    f_local = _pad_runtime_field(
        f_arr,
        dx=dx2d,
        dy=dy2d,
        bc_kind_x=int(getattr(bc_field, "kind_x", 0)),
        bc_value_x=getattr(bc_field, "x_value", 0.0),
        bc_grad_x=getattr(bc_field, "x_grad", 0.0),
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    dx_local = _pad_runtime_metric(
        dx2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    dz_local = _pad_runtime_metric(
        dz2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g11_local = _pad_runtime_metric(
        g11_2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g13_local = _pad_runtime_metric(
        g13_2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g33_local = _pad_runtime_metric(
        g33_2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g1_local = _pad_runtime_metric(
        g1_2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    g3_local = _pad_runtime_metric(
        g3_2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )
    d1_dx_local = _pad_runtime_metric(
        d1_dx_2d,
        periodic_x=periodic_x,
        periodic_parallel=periodic_parallel,
        lower_boundary_open=lower_boundary_open,
        upper_boundary_open=upper_boundary_open,
    )

    f_c = f_local[2:-2, 2:-2, :]
    f_xp = f_local[2:-2, 3:-1, :]
    f_xm = f_local[2:-2, 1:-3, :]
    f_zp = _shift_binormal(f_c, +1, periodic=periodic_binormal)
    f_zm = _shift_binormal(f_c, -1, periodic=periodic_binormal)
    f_zp_xp = _shift_binormal(f_xp, +1, periodic=periodic_binormal)
    f_zp_xm = _shift_binormal(f_xm, +1, periodic=periodic_binormal)
    f_zm_xp = _shift_binormal(f_xp, -1, periodic=periodic_binormal)
    f_zm_xm = _shift_binormal(f_xm, -1, periodic=periodic_binormal)

    dx = dx_local[2:-2, 2:-2][:, :, None]
    dz = dz_local[2:-2, 2:-2][:, :, None]
    g11 = g11_local[2:-2, 2:-2][:, :, None]
    g13 = g13_local[2:-2, 2:-2][:, :, None]
    g33 = g33_local[2:-2, 2:-2][:, :, None]
    g1 = g1_local[2:-2, 2:-2][:, :, None]
    g3 = g3_local[2:-2, 2:-2][:, :, None]
    d1_dx = d1_dx_local[2:-2, 2:-2][:, :, None]

    ddx = 0.5 * (f_xp - f_xm) / jnp.maximum(dx, 1e-30)
    ddz = 0.5 * (f_zp - f_zm) / jnp.maximum(dz, 1e-30)
    d2dx2 = (f_xp - (2.0 * f_c) + f_xm) / jnp.maximum(dx * dx, 1e-30)
    d2dz2 = (f_zp - (2.0 * f_c) + f_zm) / jnp.maximum(dz * dz, 1e-30)
    d2dxdz = ((f_zp_xp - f_zp_xm) - (f_zm_xp - f_zm_xm)) / jnp.maximum(
        4.0 * dx * dz,
        1e-30,
    )

    return (g1 + (d1_dx * g11)) * ddx + g3 * ddz + g11 * d2dx2 + g33 * d2dz2 + (2.0 * g13 * d2dxdz)
