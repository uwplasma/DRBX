from __future__ import annotations

from dataclasses import dataclass, field
import os

import numpy as np

_NUMPY_DEBUG = os.environ.get("JAX_DRB_FCI_NUMPY_DEBUG", "").lower() in {"1", "true", "yes"}
if _NUMPY_DEBUG:
    jax = None
    jnp = np
else:
    import jax
    import jax.numpy as jnp


if not _NUMPY_DEBUG:
    _pytree_base = jax.tree_util.register_pytree_node_class
else:
    def _pytree_base(cls):
        return cls


@_pytree_base
@dataclass(frozen=True)
class FciGeometry3D:
    """Self-contained 3D FCI geometry payload on the logical `(x, y, z)` grid.

    The object stores the full logical grid coordinates together with the
    traced field-line maps and the metric tensor components directly.
    Native FCI conventions are `(x, y, z) = (radial, poloidal, toroidal)`.
    """

    logical_grid: jnp.ndarray

    forward_x: jnp.ndarray
    forward_y: jnp.ndarray
    backward_x: jnp.ndarray
    backward_y: jnp.ndarray
    forward_length: jnp.ndarray
    backward_length: jnp.ndarray
    forward_boundary: jnp.ndarray
    backward_boundary: jnp.ndarray

    dx: jnp.ndarray
    dy: jnp.ndarray
    dz: jnp.ndarray
    J: jnp.ndarray
    B_contravariant: jnp.ndarray
    Bmag: jnp.ndarray = field(init=False)
    g11: jnp.ndarray
    g22: jnp.ndarray
    g33: jnp.ndarray
    g12: jnp.ndarray
    g13: jnp.ndarray
    g23: jnp.ndarray
    g_11: jnp.ndarray
    g_22: jnp.ndarray
    g_33: jnp.ndarray
    g_12: jnp.ndarray
    g_13: jnp.ndarray
    g_23: jnp.ndarray

    def __post_init__(self) -> None:
        logical_grid = jnp.asarray(self.logical_grid, dtype=jnp.float64)
        if logical_grid.ndim != 4 or logical_grid.shape[-1] != 3:
            raise ValueError(f"logical_grid must have shape (nx, ny, nz, 3), got {logical_grid.shape}")

        shape = tuple(int(value) for value in logical_grid.shape[:3])
        object.__setattr__(self, "logical_grid", logical_grid)

        for name in (
            "forward_x",
            "forward_y",
            "backward_x",
            "backward_y",
            "forward_length",
            "backward_length",
            "forward_boundary",
            "backward_boundary",
            "dx",
            "dy",
            "dz",
            "J",
            "g11",
            "g22",
            "g33",
            "g12",
            "g13",
            "g23",
            "g_11",
            "g_22",
            "g_33",
            "g_12",
            "g_13",
            "g_23",
        ):
            value = jnp.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            object.__setattr__(self, name, value)

        b = jnp.asarray(self.B_contravariant, dtype=jnp.float64)
        if b.shape != shape + (3,):
            raise ValueError(f"B_contravariant must have shape {shape + (3,)}, got {b.shape}")
        object.__setattr__(self, "B_contravariant", b)

        g_cov = jnp.stack(
            [
                jnp.stack([jnp.asarray(self.g_11, dtype=jnp.float64), jnp.asarray(self.g_12, dtype=jnp.float64), jnp.asarray(self.g_13, dtype=jnp.float64)], axis=-1),
                jnp.stack([jnp.asarray(self.g_12, dtype=jnp.float64), jnp.asarray(self.g_22, dtype=jnp.float64), jnp.asarray(self.g_23, dtype=jnp.float64)], axis=-1),
                jnp.stack([jnp.asarray(self.g_13, dtype=jnp.float64), jnp.asarray(self.g_23, dtype=jnp.float64), jnp.asarray(self.g_33, dtype=jnp.float64)], axis=-1),
            ],
            axis=-2,
        )
        bmag_sq = jnp.einsum("...i,...ij,...j->...", b, g_cov, b)
        object.__setattr__(self, "Bmag", jnp.sqrt(jnp.maximum(bmag_sq, 0.0)))

    def tree_flatten(self):
        children = (
            self.logical_grid,
            self.forward_x,
            self.forward_y,
            self.backward_x,
            self.backward_y,
            self.forward_length,
            self.backward_length,
            self.forward_boundary,
            self.backward_boundary,
            self.dx,
            self.dy,
            self.dz,
            self.J,
            self.B_contravariant,
            self.g11,
            self.g22,
            self.g33,
            self.g12,
            self.g13,
            self.g23,
            self.g_11,
            self.g_22,
            self.g_33,
            self.g_12,
            self.g_13,
            self.g_23,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (
            logical_grid,
            forward_x,
            forward_y,
            backward_x,
            backward_y,
            forward_length,
            backward_length,
            forward_boundary,
            backward_boundary,
            dx,
            dy,
            dz,
            J,
            B_contravariant,
            g11,
            g22,
            g33,
            g12,
            g13,
            g23,
            g_11,
            g_22,
            g_33,
            g_12,
            g_13,
            g_23,
        ) = children
        return cls(
            logical_grid=logical_grid,
            forward_x=forward_x,
            forward_y=forward_y,
            backward_x=backward_x,
            backward_y=backward_y,
            forward_length=forward_length,
            backward_length=backward_length,
            forward_boundary=forward_boundary,
            backward_boundary=backward_boundary,
            dx=dx,
            dy=dy,
            dz=dz,
            J=J,
            B_contravariant=B_contravariant,
            g11=g11,
            g22=g22,
            g33=g33,
            g12=g12,
            g13=g13,
            g23=g23,
            g_11=g_11,
            g_22=g_22,
            g_33=g_33,
            g_12=g_12,
            g_13=g_13,
            g_23=g_23,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.J.shape)

    @property
    def logical_axis_vectors(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return logical_axis_vectors_from_grid(self.logical_grid)

    @property
    def x_axis(self) -> jnp.ndarray:
        return self.logical_axis_vectors[0]

    @property
    def y_axis(self) -> jnp.ndarray:
        return self.logical_axis_vectors[1]

    @property
    def z_axis(self) -> jnp.ndarray:
        return self.logical_axis_vectors[2]


def logical_axis_vectors_from_grid(logical_grid: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract the 1D logical coordinate vectors from a structured grid."""

    grid = jnp.asarray(logical_grid, dtype=jnp.float64)
    if grid.ndim != 4 or grid.shape[-1] != 3:
        raise ValueError(f"logical_grid must have shape (nx, ny, nz, 3), got {grid.shape}")
    return grid[:, 0, 0, 0], grid[0, :, 0, 1], grid[0, 0, :, 2]


def logical_grid_from_axis_vectors(
    x_axis: jnp.ndarray,
    y_axis: jnp.ndarray,
    z_axis: jnp.ndarray,
) -> jnp.ndarray:
    """Build a structured logical grid from 1D logical axis vectors."""

    x = jnp.asarray(x_axis, dtype=jnp.float64)
    y = jnp.asarray(y_axis, dtype=jnp.float64)
    z = jnp.asarray(z_axis, dtype=jnp.float64)
    xx = jnp.broadcast_to(x[:, None, None], (x.size, y.size, z.size))
    yy = jnp.broadcast_to(y[None, :, None], (x.size, y.size, z.size))
    zz = jnp.broadcast_to(z[None, None, :], (x.size, y.size, z.size))
    return jnp.stack((xx, yy, zz), axis=-1)


def interpolate_B_contravariant(
    geometry: FciGeometry3D,
    points: jnp.ndarray,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    boundary_value: float = jnp.nan,
) -> jnp.ndarray:
    """Trilinearly interpolate `B_contravariant` at logical-space points."""

    return _interpolate_B_contravariant_on_grid(
        geometry.logical_grid,
        geometry.B_contravariant,
        points,
        periodic_axes=periodic_axes,
        boundary_value=boundary_value,
    )


def rk4_step(
    geometry: FciGeometry3D,
    point: jnp.ndarray,
    step: float,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    min_abs_bz: float = 1.0e-30,
    boundary_value: float = jnp.nan,
) -> jnp.ndarray:
    """Advance a logical-space field line by one fixed RK4 step in toroidal `z`."""

    return _rk4_step_on_grid(
        geometry.logical_grid,
        geometry.B_contravariant,
        point,
        step,
        periodic_axes=periodic_axes,
        min_abs_bz=min_abs_bz,
        boundary_value=boundary_value,
    )


def build_fci_maps_from_b_contravariant(
    logical_grid: jnp.ndarray,
    B_contravariant: jnp.ndarray,
    Bmag: jnp.ndarray,
    *,
    substeps: int = 8,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    min_abs_bz: float = 1.0e-30,
    boundary_value: float = jnp.nan,
) -> dict[str, jnp.ndarray | float]:
    """Trace one toroidal step from each logical point and return map fields."""

    logical_grid = jnp.asarray(logical_grid, dtype=jnp.float64)
    if logical_grid.ndim != 4 or logical_grid.shape[-1] != 3:
        raise ValueError(f"logical_grid must have shape (nx, ny, nz, 3), got {logical_grid.shape}")

    shape = tuple(int(value) for value in logical_grid.shape[:3])
    b = jnp.asarray(B_contravariant, dtype=jnp.float64)
    if b.shape != shape + (3,):
        raise ValueError(f"B_contravariant must have shape {shape + (3,)}, got {b.shape}")
    bmag = jnp.asarray(Bmag, dtype=jnp.float64)
    if bmag.shape != shape:
        raise ValueError(f"Bmag must have shape {shape}, got {bmag.shape}")

    x_axis, y_axis, z_axis = logical_axis_vectors_from_grid(logical_grid)
    nz = int(z_axis.size)
    if nz < 1:
        raise ValueError("logical_grid must contain at least one toroidal plane.")

    forward_x_planes = []
    forward_y_planes = []
    backward_x_planes = []
    backward_y_planes = []
    forward_length_planes = []
    backward_length_planes = []
    forward_boundary_planes = []
    backward_boundary_planes = []
    dz_planes = []

    for k in range(nz):
        current_plane = logical_grid[:, :, k, :].reshape((-1, 3))
        forward_step = _plane_step(z_axis, k, periodic=bool(periodic_axes[2]), direction=1)
        backward_step = _plane_step(z_axis, k, periodic=bool(periodic_axes[2]), direction=-1)

        forward_points, forward_length, forward_boundary = _trace_fieldline_to_plane(
            logical_grid,
            b,
            bmag,
            current_plane,
            step=forward_step,
            substeps=substeps,
            periodic_axes=periodic_axes,
            min_abs_bz=min_abs_bz,
            boundary_value=boundary_value,
        )
        backward_points, backward_length, backward_boundary = _trace_fieldline_to_plane(
            logical_grid,
            b,
            bmag,
            current_plane,
            step=backward_step,
            substeps=substeps,
            periodic_axes=periodic_axes,
            min_abs_bz=min_abs_bz,
            boundary_value=boundary_value,
        )

        forward_x_planes.append(_logical_coordinate_to_index(x_axis, forward_points[..., 0], periodic=bool(periodic_axes[0])))
        forward_y_planes.append(_logical_coordinate_to_index(y_axis, forward_points[..., 1], periodic=bool(periodic_axes[1])))
        backward_x_planes.append(_logical_coordinate_to_index(x_axis, backward_points[..., 0], periodic=bool(periodic_axes[0])))
        backward_y_planes.append(_logical_coordinate_to_index(y_axis, backward_points[..., 1], periodic=bool(periodic_axes[1])))
        forward_length_planes.append(forward_length)
        backward_length_planes.append(backward_length)
        forward_boundary_planes.append(forward_boundary)
        backward_boundary_planes.append(backward_boundary)
        dz_planes.append(jnp.full((shape[0], shape[1]), abs(float(forward_step)), dtype=jnp.float64))

    return {
        "forward_x": jnp.stack(forward_x_planes, axis=-1).reshape(shape),
        "forward_y": jnp.stack(forward_y_planes, axis=-1).reshape(shape),
        "backward_x": jnp.stack(backward_x_planes, axis=-1).reshape(shape),
        "backward_y": jnp.stack(backward_y_planes, axis=-1).reshape(shape),
        "forward_length": jnp.stack(forward_length_planes, axis=-1).reshape(shape),
        "backward_length": jnp.stack(backward_length_planes, axis=-1).reshape(shape),
        "forward_boundary": jnp.stack(forward_boundary_planes, axis=-1).reshape(shape),
        "backward_boundary": jnp.stack(backward_boundary_planes, axis=-1).reshape(shape),
        "dz": jnp.stack(dz_planes, axis=-1).reshape(shape),
    }


def logical_grid_from_axis_vectors_np(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> np.ndarray:
    """NumPy debug version of `logical_grid_from_axis_vectors`."""

    x = np.asarray(x_axis, dtype=np.float64)
    y = np.asarray(y_axis, dtype=np.float64)
    z = np.asarray(z_axis, dtype=np.float64)
    xx = np.broadcast_to(x[:, None, None], (x.size, y.size, z.size))
    yy = np.broadcast_to(y[None, :, None], (x.size, y.size, z.size))
    zz = np.broadcast_to(z[None, None, :], (x.size, y.size, z.size))
    return np.stack((xx, yy, zz), axis=-1)


def build_fci_maps_from_b_contravariant_np(
    logical_grid: np.ndarray,
    B_contravariant: np.ndarray,
    Bmag: np.ndarray,
    *,
    substeps: int = 8,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    min_abs_bz: float = 1.0e-30,
    boundary_value: float = np.nan,
) -> dict[str, np.ndarray]:
    """NumPy debug path for tracing one toroidal step from each logical point."""

    grid = np.asarray(logical_grid, dtype=np.float64)
    if grid.ndim != 4 or grid.shape[-1] != 3:
        raise ValueError(f"logical_grid must have shape (nx, ny, nz, 3), got {grid.shape}")
    shape = tuple(int(value) for value in grid.shape[:3])
    b = np.asarray(B_contravariant, dtype=np.float64)
    if b.shape != shape + (3,):
        raise ValueError(f"B_contravariant must have shape {shape + (3,)}, got {b.shape}")
    bmag = np.asarray(Bmag, dtype=np.float64)
    if bmag.shape != shape:
        raise ValueError(f"Bmag must have shape {shape}, got {bmag.shape}")

    x_axis, y_axis, z_axis = _logical_axis_vectors_from_grid_np(grid)
    forward_x_planes = []
    forward_y_planes = []
    backward_x_planes = []
    backward_y_planes = []
    forward_length_planes = []
    backward_length_planes = []
    forward_boundary_planes = []
    backward_boundary_planes = []
    dz_planes = []

    for k in range(int(z_axis.size)):
        seeds = grid[:, :, k, :].reshape((-1, 3))
        forward_step = _plane_step_np(z_axis, k, periodic=bool(periodic_axes[2]), direction=1)
        backward_step = _plane_step_np(z_axis, k, periodic=bool(periodic_axes[2]), direction=-1)
        forward_points, forward_length, forward_boundary = _trace_fieldline_to_plane_np(
            grid,
            b,
            bmag,
            seeds,
            step=forward_step,
            substeps=substeps,
            periodic_axes=periodic_axes,
            min_abs_bz=min_abs_bz,
            boundary_value=boundary_value,
        )
        backward_points, backward_length, backward_boundary = _trace_fieldline_to_plane_np(
            grid,
            b,
            bmag,
            seeds,
            step=backward_step,
            substeps=substeps,
            periodic_axes=periodic_axes,
            min_abs_bz=min_abs_bz,
            boundary_value=boundary_value,
        )
        forward_x_planes.append(_logical_coordinate_to_index_np(x_axis, forward_points[:, 0], periodic=bool(periodic_axes[0])))
        forward_y_planes.append(_logical_coordinate_to_index_np(y_axis, forward_points[:, 1], periodic=bool(periodic_axes[1])))
        backward_x_planes.append(_logical_coordinate_to_index_np(x_axis, backward_points[:, 0], periodic=bool(periodic_axes[0])))
        backward_y_planes.append(_logical_coordinate_to_index_np(y_axis, backward_points[:, 1], periodic=bool(periodic_axes[1])))
        forward_length_planes.append(forward_length)
        backward_length_planes.append(backward_length)
        forward_boundary_planes.append(forward_boundary)
        backward_boundary_planes.append(backward_boundary)
        dz_planes.append(np.full((shape[0], shape[1]), abs(float(forward_step)), dtype=np.float64))

    return {
        "forward_x": np.stack(forward_x_planes, axis=-1).reshape(shape),
        "forward_y": np.stack(forward_y_planes, axis=-1).reshape(shape),
        "backward_x": np.stack(backward_x_planes, axis=-1).reshape(shape),
        "backward_y": np.stack(backward_y_planes, axis=-1).reshape(shape),
        "forward_length": np.stack(forward_length_planes, axis=-1).reshape(shape),
        "backward_length": np.stack(backward_length_planes, axis=-1).reshape(shape),
        "forward_boundary": np.stack(forward_boundary_planes, axis=-1).reshape(shape),
        "backward_boundary": np.stack(backward_boundary_planes, axis=-1).reshape(shape),
        "dz": np.stack(dz_planes, axis=-1).reshape(shape),
    }


def _logical_axis_vectors_from_grid_np(logical_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.asarray(logical_grid, dtype=np.float64)
    return grid[:, 0, 0, 0], grid[0, :, 0, 1], grid[0, 0, :, 2]


def _interpolate_B_contravariant_on_grid_np(
    logical_grid: np.ndarray,
    B_contravariant: np.ndarray,
    points: np.ndarray,
    *,
    periodic_axes: tuple[bool, bool, bool],
    boundary_value: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    squeeze = pts.shape == (3,)
    if squeeze:
        pts = pts[None, :]
    x_axis, y_axis, z_axis = _logical_axis_vectors_from_grid_np(logical_grid)
    return_value = []
    for component in range(3):
        return_value.append(
            _interpolate_scalar_on_logical_grid_np(
                B_contravariant[..., component],
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                x_axis=x_axis,
                y_axis=y_axis,
                z_axis=z_axis,
                periodic_axes=periodic_axes,
                boundary_value=boundary_value,
            )
        )
    result = np.stack(return_value, axis=-1)
    return result[0] if squeeze else result


def _interpolate_scalar_on_logical_grid_np(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    periodic_axes: tuple[bool, bool, bool],
    boundary_value: float,
) -> np.ndarray:
    x0, x1, wx, valid_x = _bracket_axis_np(x_axis, x, periodic=bool(periodic_axes[0]))
    y0, y1, wy, valid_y = _bracket_axis_np(y_axis, y, periodic=bool(periodic_axes[1]))
    z0, z1, wz, valid_z = _bracket_axis_np(z_axis, z, periodic=bool(periodic_axes[2]))
    valid = valid_x & valid_y & valid_z
    interpolated = _trilinear_sample_np(values, x0, x1, wx, y0, y1, wy, z0, z1, wz)
    return np.where(valid, interpolated, np.asarray(boundary_value, dtype=np.float64))


def _bracket_axis_np(axis: np.ndarray, values: np.ndarray, *, periodic: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if axis.size < 2:
        raise ValueError("Each logical axis must contain at least two points for interpolation.")
    if periodic:
        spacing = axis[1] - axis[0]
        period = (axis[-1] - axis[0]) + spacing
        wrapped = np.mod(values - axis[0], period) + axis[0]
        upper = np.searchsorted(axis, wrapped, side="right")
        lower = np.clip(upper - 1, 0, axis.size - 1)
        next_index = np.mod(lower + 1, axis.size)
        lower_coord = axis[lower]
        upper_coord = np.where(lower == axis.size - 1, axis[0] + period, axis[next_index])
        weight = (wrapped - lower_coord) / (upper_coord - lower_coord)
        return lower, next_index, np.clip(weight, 0.0, 1.0), np.isfinite(values)
    upper = np.searchsorted(axis, values, side="right")
    lower = np.clip(upper - 1, 0, axis.size - 2)
    upper = lower + 1
    lower_coord = axis[lower]
    upper_coord = axis[upper]
    weight = (values - lower_coord) / (upper_coord - lower_coord)
    valid = (values >= axis[0]) & (values <= axis[-1]) & np.isfinite(values)
    return lower, upper, np.clip(weight, 0.0, 1.0), valid


def _trilinear_sample_np(values, x0, x1, wx, y0, y1, wy, z0, z1, wz):
    c000 = values[x0, y0, z0]
    c100 = values[x1, y0, z0]
    c010 = values[x0, y1, z0]
    c110 = values[x1, y1, z0]
    c001 = values[x0, y0, z1]
    c101 = values[x1, y0, z1]
    c011 = values[x0, y1, z1]
    c111 = values[x1, y1, z1]
    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return c0 * (1.0 - wz) + c1 * wz


def _trace_fieldline_to_plane_np(
    logical_grid: np.ndarray,
    B_contravariant: np.ndarray,
    Bmag: np.ndarray,
    seed_points: np.ndarray,
    *,
    step: float,
    substeps: int,
    periodic_axes: tuple[bool, bool, bool],
    min_abs_bz: float,
    boundary_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.asarray(seed_points, dtype=np.float64).copy()
    x_axis, y_axis, z_axis = _logical_axis_vectors_from_grid_np(logical_grid)
    step_size = float(step) / float(max(int(substeps), 1))
    length = np.zeros(state.shape[0], dtype=np.float64)
    alive = np.ones(state.shape[0], dtype=bool)

    for _ in range(max(int(substeps), 1)):
        b0 = _interpolate_B_contravariant_on_grid_np(
            logical_grid,
            B_contravariant,
            state,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )
        bmag0 = _interpolate_scalar_on_logical_grid_np(
            Bmag,
            state[:, 0],
            state[:, 1],
            state[:, 2],
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )
        next_state = np.stack(
            [
                _rk4_step_on_grid_np(
                    logical_grid,
                    B_contravariant,
                    point,
                    step_size,
                    periodic_axes=periodic_axes,
                    min_abs_bz=min_abs_bz,
                    boundary_value=boundary_value,
                )
                for point in state
            ],
            axis=0,
        )
        b1 = _interpolate_B_contravariant_on_grid_np(
            logical_grid,
            B_contravariant,
            next_state,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )
        bmag1 = _interpolate_scalar_on_logical_grid_np(
            Bmag,
            next_state[:, 0],
            next_state[:, 1],
            next_state[:, 2],
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )
        finite = np.all(np.isfinite(next_state), axis=-1)
        increment = 0.5 * abs(step_size) * (
            _speed_np(b0, bmag0, min_abs_bz) + _speed_np(b1, bmag1, min_abs_bz)
        )
        length = length + np.where(alive & finite, increment, 0.0)
        state = np.where((alive & finite)[:, None], next_state, state)
        alive = alive & finite
    return state, length, ~alive


def _speed_np(sampled_b: np.ndarray, sampled_bmag: np.ndarray, min_abs_bz: float) -> np.ndarray:
    bz = sampled_b[..., 2]
    safe_bz = np.where(np.abs(bz) < float(min_abs_bz), np.where(bz < 0.0, -1.0, 1.0) * float(min_abs_bz), bz)
    return np.asarray(sampled_bmag, dtype=np.float64) / np.maximum(np.abs(safe_bz), 1.0e-30)


def _rk4_step_on_grid_np(
    logical_grid: np.ndarray,
    B_contravariant: np.ndarray,
    point: np.ndarray,
    step: float,
    *,
    periodic_axes: tuple[bool, bool, bool],
    min_abs_bz: float,
    boundary_value: float,
) -> np.ndarray:
    state = np.asarray(point, dtype=np.float64)
    h = float(step)

    def rhs(value: np.ndarray) -> np.ndarray:
        b = _interpolate_B_contravariant_on_grid_np(
            logical_grid,
            B_contravariant,
            value,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )
        bz = b[2]
        safe_bz = np.where(abs(bz) < float(min_abs_bz), (-1.0 if bz < 0.0 else 1.0) * float(min_abs_bz), bz)
        return np.asarray((b[0] / safe_bz, b[1] / safe_bz, 1.0), dtype=np.float64)

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * h * k1)
    k3 = rhs(state + 0.5 * h * k2)
    k4 = rhs(state + h * k3)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _plane_step_np(z_axis: np.ndarray, index: int, *, periodic: bool, direction: int) -> float:
    z_axis = np.asarray(z_axis, dtype=np.float64)
    if z_axis.size < 2:
        return float(direction)
    if direction >= 0:
        if index < z_axis.size - 1:
            return float(z_axis[index + 1] - z_axis[index])
        if periodic:
            span = (z_axis[-1] - z_axis[0]) + (z_axis[1] - z_axis[0])
            return float((z_axis[0] + span) - z_axis[-1])
        return float(z_axis[-1] - z_axis[-2])
    if index > 0:
        return float(-(z_axis[index] - z_axis[index - 1]))
    if periodic:
        span = (z_axis[-1] - z_axis[0]) + (z_axis[1] - z_axis[0])
        return float(z_axis[-1] - span - z_axis[0])
    return float(-(z_axis[1] - z_axis[0]))


def _logical_coordinate_to_index_np(axis: np.ndarray, values: np.ndarray, *, periodic: bool) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if axis.size < 2:
        return np.zeros_like(values, dtype=np.float64)
    if periodic:
        spacing = axis[1] - axis[0]
        period = (axis[-1] - axis[0]) + spacing
        wrapped = np.mod(values - axis[0], period) + axis[0]
        upper = np.searchsorted(axis, wrapped, side="right")
        lower = np.clip(upper - 1, 0, axis.size - 1)
        next_index = np.mod(lower + 1, axis.size)
        lower_coord = axis[lower]
        upper_coord = np.where(lower == axis.size - 1, axis[0] + period, axis[next_index])
        weight = (wrapped - lower_coord) / (upper_coord - lower_coord)
        return lower.astype(np.float64) + np.clip(weight, 0.0, 1.0)
    upper = np.searchsorted(axis, values, side="right")
    lower = np.clip(upper - 1, 0, axis.size - 2)
    upper = lower + 1
    lower_coord = axis[lower]
    upper_coord = axis[upper]
    weight = (values - lower_coord) / (upper_coord - lower_coord)
    return lower.astype(np.float64) + np.clip(weight, 0.0, 1.0)


def _interpolate_scalar_on_logical_grid(
    values: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    *,
    x_axis: jnp.ndarray,
    y_axis: jnp.ndarray,
    z_axis: jnp.ndarray,
    periodic_axes: tuple[bool, bool, bool],
    boundary_value: float,
) -> jnp.ndarray:
    values = jnp.asarray(values, dtype=jnp.float64)
    x0, x1, wx, valid_x = _bracket_axis(x_axis, x, periodic=bool(periodic_axes[0]))
    y0, y1, wy, valid_y = _bracket_axis(y_axis, y, periodic=bool(periodic_axes[1]))
    z0, z1, wz, valid_z = _bracket_axis(z_axis, z, periodic=bool(periodic_axes[2]))
    valid = valid_x & valid_y & valid_z
    interpolated = _trilinear_sample(values, x0, x1, wx, y0, y1, wy, z0, z1, wz)
    return jnp.where(valid, interpolated, jnp.asarray(boundary_value, dtype=jnp.float64))


def _interpolate_B_contravariant_on_grid(
    logical_grid: jnp.ndarray,
    B_contravariant: jnp.ndarray,
    points: jnp.ndarray,
    *,
    periodic_axes: tuple[bool, bool, bool],
    boundary_value: float,
) -> jnp.ndarray:
    sampled_points = jnp.asarray(points, dtype=jnp.float64)
    if sampled_points.shape == (3,):
        sampled_points = sampled_points[None, :]
        squeeze = True
    else:
        if sampled_points.ndim < 1 or sampled_points.shape[-1] != 3:
            raise ValueError(f"points must have shape (3,) or (..., 3), got {sampled_points.shape}")
        squeeze = False

    x_axis, y_axis, z_axis = logical_axis_vectors_from_grid(logical_grid)
    samples = []
    for component in range(3):
        samples.append(
            _interpolate_scalar_on_logical_grid(
                B_contravariant[..., component],
                sampled_points[..., 0],
                sampled_points[..., 1],
                sampled_points[..., 2],
                x_axis=x_axis,
                y_axis=y_axis,
                z_axis=z_axis,
                periodic_axes=periodic_axes,
                boundary_value=boundary_value,
            )
        )
    result = jnp.stack(samples, axis=-1)
    return result[0] if squeeze else result


def _bracket_axis(
    axis: jnp.ndarray,
    values: jnp.ndarray,
    *,
    periodic: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    axis = jnp.asarray(axis, dtype=jnp.float64)
    values = jnp.asarray(values, dtype=jnp.float64)
    if axis.size < 2:
        raise ValueError("Each logical axis must contain at least two points for interpolation.")

    if periodic:
        spacing = axis[1] - axis[0]
        period = (axis[-1] - axis[0]) + spacing
        wrapped = jnp.mod(values - axis[0], period) + axis[0]
        upper = jnp.searchsorted(axis, wrapped, side="right")
        lower = jnp.clip(upper - 1, 0, int(axis.size) - 1)
        next_index = jnp.mod(lower + 1, int(axis.size))
        lower_coord = axis[lower]
        upper_coord = jnp.where(lower == int(axis.size) - 1, axis[0] + period, axis[next_index])
        weight = (wrapped - lower_coord) / (upper_coord - lower_coord)
        valid = jnp.isfinite(values)
        return lower, next_index, jnp.clip(weight, 0.0, 1.0), valid

    upper = jnp.searchsorted(axis, values, side="right")
    lower = jnp.clip(upper - 1, 0, int(axis.size) - 2)
    upper = lower + 1
    lower_coord = axis[lower]
    upper_coord = axis[upper]
    weight = (values - lower_coord) / (upper_coord - lower_coord)
    valid = (values >= axis[0]) & (values <= axis[-1]) & jnp.isfinite(values)
    return lower, upper, jnp.clip(weight, 0.0, 1.0), valid


def _trilinear_sample(
    values: jnp.ndarray,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    wx: jnp.ndarray,
    y0: jnp.ndarray,
    y1: jnp.ndarray,
    wy: jnp.ndarray,
    z0: jnp.ndarray,
    z1: jnp.ndarray,
    wz: jnp.ndarray,
) -> jnp.ndarray:
    c000 = values[x0, y0, z0]
    c100 = values[x1, y0, z0]
    c010 = values[x0, y1, z0]
    c110 = values[x1, y1, z0]
    c001 = values[x0, y0, z1]
    c101 = values[x1, y0, z1]
    c011 = values[x0, y1, z1]
    c111 = values[x1, y1, z1]
    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return c0 * (1.0 - wz) + c1 * wz


def _trace_fieldline_to_plane(
    logical_grid: jnp.ndarray,
    B_contravariant: jnp.ndarray,
    Bmag: jnp.ndarray,
    seed_points: jnp.ndarray,
    *,
    step: float,
    substeps: int,
    periodic_axes: tuple[bool, bool, bool],
    min_abs_bz: float,
    boundary_value: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Trace a batch of seeds across one toroidal plane and accumulate arclength."""

    points = jnp.asarray(seed_points, dtype=jnp.float64)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"seed_points must have shape (n, 3), got {points.shape}")

    x_axis, y_axis, z_axis = logical_axis_vectors_from_grid(logical_grid)
    nseed = int(points.shape[0])
    step_size = float(step) / float(max(int(substeps), 1))
    length = jnp.zeros(nseed, dtype=jnp.float64)
    alive = jnp.ones(nseed, dtype=bool)
    state = points

    def _speed(sampled_b: jnp.ndarray, sampled_bmag: jnp.ndarray) -> jnp.ndarray:
        bz = sampled_b[..., 2]
        safe_bz = jnp.where(
            jnp.abs(bz) < float(min_abs_bz),
            jnp.where(bz < 0.0, -1.0, 1.0) * float(min_abs_bz),
            bz,
        )
        return jnp.asarray(sampled_bmag, dtype=jnp.float64) / jnp.maximum(jnp.abs(safe_bz), 1.0e-30)

    for _ in range(max(int(substeps), 1)):
        b0 = jax.vmap(
            lambda point: _interpolate_B_contravariant_on_grid(
                logical_grid,
                B_contravariant,
                point,
                periodic_axes=periodic_axes,
                boundary_value=boundary_value,
            )
        )(state)
        bmag0 = _interpolate_scalar_on_logical_grid(
            Bmag,
            state[:, 0],
            state[:, 1],
            state[:, 2],
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )
        next_state = jax.vmap(
            lambda point: _rk4_step_on_grid(
                logical_grid,
                B_contravariant,
                point,
                step_size,
                periodic_axes=periodic_axes,
                min_abs_bz=min_abs_bz,
                boundary_value=boundary_value,
            )
        )(state)
        bmag1 = _interpolate_scalar_on_logical_grid(
            Bmag,
            next_state[:, 0],
            next_state[:, 1],
            next_state[:, 2],
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )
        b1 = jax.vmap(
            lambda point: _interpolate_B_contravariant_on_grid(
                logical_grid,
                B_contravariant,
                point,
                periodic_axes=periodic_axes,
                boundary_value=boundary_value,
            )
        )(next_state)
        finite = jnp.all(jnp.isfinite(next_state), axis=-1)
        increment = 0.5 * abs(step_size) * (_speed(b0, bmag0) + _speed(b1, bmag1))
        increment = jnp.where(alive & finite, increment, 0.0)
        length = length + increment
        state = jnp.where((alive & finite)[..., None], next_state, state)
        alive = alive & finite

    return state, length, ~alive


def _plane_step(z_axis: jnp.ndarray, index: int, *, periodic: bool, direction: int) -> float:
    """Return the signed logical `z` step from a plane to its neighbor."""

    z_axis = jnp.asarray(z_axis, dtype=jnp.float64)
    if z_axis.size < 2:
        return float(direction)

    if direction >= 0:
        if index < int(z_axis.size) - 1:
            return float(z_axis[index + 1] - z_axis[index])
        if periodic:
            span = (z_axis[-1] - z_axis[0]) + (z_axis[1] - z_axis[0])
            return float((z_axis[0] + span) - z_axis[-1])
        return float(z_axis[-1] - z_axis[-2])

    if index > 0:
        return float(-(z_axis[index] - z_axis[index - 1]))
    if periodic:
        span = (z_axis[-1] - z_axis[0]) + (z_axis[1] - z_axis[0])
        return float(z_axis[-1] - span - z_axis[0])
    return float(-(z_axis[1] - z_axis[0]))


def _rk4_step_on_grid(
    logical_grid: jnp.ndarray,
    B_contravariant: jnp.ndarray,
    point: jnp.ndarray,
    step: float,
    *,
    periodic_axes: tuple[bool, bool, bool],
    min_abs_bz: float,
    boundary_value: float,
) -> jnp.ndarray:
    state = jnp.asarray(point, dtype=jnp.float64)
    if state.shape != (3,):
        raise ValueError(f"point must have shape (3,), got {state.shape}")
    h = float(step)

    def rhs(value: jnp.ndarray) -> jnp.ndarray:
        b = _interpolate_B_contravariant_on_grid(
            logical_grid,
            B_contravariant,
            value,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )
        bz = jnp.asarray(b[2], dtype=jnp.float64)
        safe_bz = jnp.where(jnp.abs(bz) < float(min_abs_bz), jnp.where(bz < 0.0, -1.0, 1.0) * float(min_abs_bz), bz)
        return jnp.stack((b[0] / safe_bz, b[1] / safe_bz, jnp.array(1.0, dtype=jnp.float64)))

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * h * k1)
    k3 = rhs(state + 0.5 * h * k2)
    k4 = rhs(state + h * k3)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _logical_coordinate_to_index(
    axis: jnp.ndarray,
    values: jnp.ndarray,
    *,
    periodic: bool,
) -> jnp.ndarray:
    """Convert logical coordinate values to fractional grid indices."""

    axis = jnp.asarray(axis, dtype=jnp.float64)
    values = jnp.asarray(values, dtype=jnp.float64)
    if axis.size < 2:
        return jnp.zeros_like(values, dtype=jnp.float64)

    if periodic:
        spacing = axis[1] - axis[0]
        period = (axis[-1] - axis[0]) + spacing
        wrapped = jnp.mod(values - axis[0], period) + axis[0]
        upper = jnp.searchsorted(axis, wrapped, side="right")
        lower = jnp.clip(upper - 1, 0, int(axis.size) - 1)
        next_index = jnp.mod(lower + 1, int(axis.size))
        lower_coord = axis[lower]
        upper_coord = jnp.where(lower == int(axis.size) - 1, axis[0] + period, axis[next_index])
        weight = (wrapped - lower_coord) / (upper_coord - lower_coord)
        return jnp.asarray(lower, dtype=jnp.float64) + jnp.clip(weight, 0.0, 1.0)

    upper = jnp.searchsorted(axis, values, side="right")
    lower = jnp.clip(upper - 1, 0, int(axis.size) - 2)
    upper = lower + 1
    lower_coord = axis[lower]
    upper_coord = axis[upper]
    weight = (values - lower_coord) / (upper_coord - lower_coord)
    return jnp.asarray(lower, dtype=jnp.float64) + jnp.clip(weight, 0.0, 1.0)


def logical_b_contravariant_from_geometry(geometry: FciGeometry3D) -> jnp.ndarray:
    """Return a logical contravariant unit-field direction from geometry."""

    dz = jnp.asarray(geometry.dz, dtype=jnp.float64)
    forward_disp = jnp.stack(
        (
            jnp.asarray(geometry.forward_x, dtype=jnp.float64) - jnp.broadcast_to(
                jnp.arange(geometry.shape[0], dtype=jnp.float64)[:, None, None], geometry.shape
            ),
            jnp.asarray(geometry.forward_y, dtype=jnp.float64) - jnp.broadcast_to(
                jnp.arange(geometry.shape[1], dtype=jnp.float64)[None, :, None], geometry.shape
            ),
            jnp.ones(geometry.shape, dtype=jnp.float64) * dz,
        ),
        axis=-1,
    )
    backward_disp = jnp.stack(
        (
            jnp.broadcast_to(jnp.arange(geometry.shape[0], dtype=jnp.float64)[:, None, None], geometry.shape)
            - jnp.asarray(geometry.backward_x, dtype=jnp.float64),
            jnp.broadcast_to(jnp.arange(geometry.shape[1], dtype=jnp.float64)[None, :, None], geometry.shape)
            - jnp.asarray(geometry.backward_y, dtype=jnp.float64),
            jnp.ones(geometry.shape, dtype=jnp.float64) * dz,
        ),
        axis=-1,
    )
    forward_length = jnp.maximum(jnp.asarray(geometry.forward_length, dtype=jnp.float64)[..., None], 1.0e-30)
    backward_length = jnp.maximum(jnp.asarray(geometry.backward_length, dtype=jnp.float64)[..., None], 1.0e-30)
    b = 0.5 * (forward_disp / forward_length + backward_disp / backward_length)
    norm = jnp.linalg.norm(b, axis=-1, keepdims=True)
    return b / jnp.maximum(norm, 1.0e-30)


def logical_b_contravariant_from_traced_maps(
    *,
    forward_x: jnp.ndarray,
    forward_y: jnp.ndarray,
    backward_x: jnp.ndarray,
    backward_y: jnp.ndarray,
    forward_length: jnp.ndarray,
    backward_length: jnp.ndarray,
    dz: jnp.ndarray | float,
) -> jnp.ndarray:
    """Estimate the logical contravariant unit-field direction from traced maps."""

    forward_x = jnp.asarray(forward_x, dtype=jnp.float64)
    forward_y = jnp.asarray(forward_y, dtype=jnp.float64)
    backward_x = jnp.asarray(backward_x, dtype=jnp.float64)
    backward_y = jnp.asarray(backward_y, dtype=jnp.float64)
    shape = forward_x.shape
    x = jnp.broadcast_to(jnp.arange(shape[0], dtype=jnp.float64)[:, None, None], shape)
    y = jnp.broadcast_to(jnp.arange(shape[1], dtype=jnp.float64)[None, :, None], shape)
    dz = jnp.asarray(dz, dtype=jnp.float64)
    forward_disp = jnp.stack((forward_x - x, forward_y - y, jnp.ones(shape, dtype=jnp.float64) * dz), axis=-1)
    backward_disp = jnp.stack((x - backward_x, y - backward_y, jnp.ones(shape, dtype=jnp.float64) * dz), axis=-1)
    forward_length = jnp.maximum(jnp.asarray(forward_length, dtype=jnp.float64)[..., None], 1.0e-30)
    backward_length = jnp.maximum(jnp.asarray(backward_length, dtype=jnp.float64)[..., None], 1.0e-30)
    b = 0.5 * (forward_disp / forward_length + backward_disp / backward_length)
    norm = jnp.linalg.norm(b, axis=-1, keepdims=True)
    return b / jnp.maximum(norm, 1.0e-30)


def metric_inverse_residual(geometry: FciGeometry3D) -> jnp.ndarray:
    """Return `max(abs(g^ik g_kj - delta^i_j))` over the grid."""

    contravariant = jnp.stack(
        [
            jnp.stack([geometry.g11, geometry.g12, geometry.g13], axis=-1),
            jnp.stack([geometry.g12, geometry.g22, geometry.g23], axis=-1),
            jnp.stack([geometry.g13, geometry.g23, geometry.g33], axis=-1),
        ],
        axis=-2,
    )
    covariant = jnp.stack(
        [
            jnp.stack([geometry.g_11, geometry.g_12, geometry.g_13], axis=-1),
            jnp.stack([geometry.g_12, geometry.g_22, geometry.g_23], axis=-1),
            jnp.stack([geometry.g_13, geometry.g_23, geometry.g_33], axis=-1),
        ],
        axis=-2,
    )
    product = jnp.einsum("...ik,...kj->...ij", contravariant, covariant)
    identity = jnp.eye(3, dtype=product.dtype)
    return jnp.max(jnp.abs(product - identity))


def build_metric_report(geometry: FciGeometry3D) -> dict[str, object]:
    """Build finite/positive/inverse-consistency diagnostics for a geometry payload."""

    report: dict[str, object] = {
        "shape": list(geometry.shape),
        "inverse_residual_linf": float(metric_inverse_residual(geometry)),
        "fields": {},
    }
    for name in (
        "J",
        "Bmag",
        "B_contravariant",
        "g11",
        "g22",
        "g33",
        "g12",
        "g13",
        "g23",
        "g_11",
        "g_22",
        "g_33",
        "g_12",
        "g_13",
        "g_23",
    ):
        values = np.asarray(getattr(geometry, name), dtype=np.float64)
        report["fields"][name] = {
            "finite": bool(np.all(np.isfinite(values))),
            "minimum": float(np.nanmin(values)),
            "maximum": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
    report["passed"] = (
        bool(report["fields"]["J"]["finite"])
        and bool(report["fields"]["Bmag"]["finite"])
        and bool(report["fields"]["B_contravariant"]["finite"])
        and float(report["fields"]["J"]["minimum"]) > 0.0
        and float(report["fields"]["Bmag"]["minimum"]) > 0.0
        and float(report["inverse_residual_linf"]) < 1.0e-8
    )
    return report
