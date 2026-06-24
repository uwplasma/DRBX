from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

try:  # Optional external solver backend.
    import lineax as lx
except ImportError:  # pragma: no cover - depends on local optional install
    lx = None

_NUMPY_DEBUG = False
_pytree_base = jax.tree_util.register_pytree_node_class

try:
    from ..geometry import FciGeometry3D
except ImportError:  # Allows direct NumPy-debug imports from tests.
    from fci_geometry import FciGeometry3D


@_pytree_base
@dataclass(frozen=True)
class FciAxisBoundaryCondition:
    """Boundary condition for one logical coordinate axis.

    ``kind`` is static pytree metadata and must be either ``"dirichlet"`` or
    ``"neumann_flux"``. Boundary payloads are dynamic pytree leaves:
    ``lower_value``/``upper_value`` are used for Dirichlet axes, while
    ``lower_flux``/``upper_flux`` are used for Neumann-flux axes. Payloads may
    be scalars or arrays broadcastable to the boundary plane.
    """

    kind: str
    lower_value: object = 0.0
    upper_value: object = 0.0
    lower_flux: object = 0.0
    upper_flux: object = 0.0

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in {"dirichlet", "neumann_flux"}:
            raise ValueError("FciAxisBoundaryCondition.kind must be 'dirichlet' or 'neumann_flux'")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "lower_value", jnp.asarray(self.lower_value, dtype=jnp.float64))
        object.__setattr__(self, "upper_value", jnp.asarray(self.upper_value, dtype=jnp.float64))
        object.__setattr__(self, "lower_flux", jnp.asarray(self.lower_flux, dtype=jnp.float64))
        object.__setattr__(self, "upper_flux", jnp.asarray(self.upper_flux, dtype=jnp.float64))

    @classmethod
    def dirichlet(cls, *, lower_value, upper_value) -> "FciAxisBoundaryCondition":
        return cls(kind="dirichlet", lower_value=lower_value, upper_value=upper_value)

    @classmethod
    def neumann_flux(cls, *, lower_flux, upper_flux) -> "FciAxisBoundaryCondition":
        return cls(kind="neumann_flux", lower_flux=lower_flux, upper_flux=upper_flux)

    def tree_flatten(self):
        children = (self.lower_value, self.upper_value, self.lower_flux, self.upper_flux)
        return children, self.kind

    @classmethod
    def tree_unflatten(cls, kind, children):
        lower_value, upper_value, lower_flux, upper_flux = children
        return cls(
            kind=kind,
            lower_value=lower_value,
            upper_value=upper_value,
            lower_flux=lower_flux,
            upper_flux=upper_flux,
        )


@_pytree_base
@dataclass(frozen=True)
class FciBoundaryCondition:
    """Boundary conditions on logical ``(x, y, z)`` coordinates.

    Naming convention:
    - ``periodic_axes`` is a length-3 boolean tuple in ``(x, y, z)`` order.
    - Every non-periodic coordinate must provide its matching axis entry:
      ``x``, ``y``, or ``z``.
    - Each supplied axis entry must be an ``FciAxisBoundaryCondition``.
    - Periodic coordinate entries should be left as ``None``.
    - ``target_mean_phi`` fixes the gauge for all-Neumann inversions.
    """

    periodic_axes: tuple[bool, bool, bool]
    x: FciAxisBoundaryCondition | None = None
    y: FciAxisBoundaryCondition | None = None
    z: FciAxisBoundaryCondition | None = None
    target_mean_phi: object = 0.0

    def __post_init__(self) -> None:
        if len(self.periodic_axes) != 3:
            raise ValueError("FciBoundaryCondition.periodic_axes must have length 3")
        periodic_axes = tuple(bool(value) for value in self.periodic_axes)
        object.__setattr__(self, "periodic_axes", periodic_axes)
        for axis, axis_bc in enumerate((self.x, self.y, self.z)):
            name = _axis_name(axis)
            if periodic_axes[axis]:
                if axis_bc is not None:
                    raise ValueError(f"periodic coordinate {name!r} must not define a boundary condition")
                continue
            if axis_bc is None:
                raise ValueError(f"non-periodic coordinate {name!r} requires a boundary condition")
            if not isinstance(axis_bc, FciAxisBoundaryCondition):
                raise TypeError(f"coordinate {name!r} boundary condition must be FciAxisBoundaryCondition")
        object.__setattr__(self, "target_mean_phi", jnp.asarray(self.target_mean_phi, dtype=jnp.float64))

    def tree_flatten(self):
        children = (self.x, self.y, self.z, self.target_mean_phi)
        return children, self.periodic_axes

    @classmethod
    def tree_unflatten(cls, periodic_axes, children):
        x, y, z, target_mean_phi = children
        return cls(periodic_axes=periodic_axes, x=x, y=y, z=z, target_mean_phi=target_mean_phi)


def _interpolate_fci_plane(
    field: jnp.ndarray,
    x_prime: jnp.ndarray,
    y_prime: jnp.ndarray,
    *,
    z_offset: int,
    boundary_value: float = 0.0,
) -> jnp.ndarray:
    """Interpolate a field on the next or previous toroidal plane."""

    values = jnp.asarray(field, dtype=jnp.float64)
    nx, ny, nz = values.shape
    z = (jnp.arange(nz, dtype=jnp.int32)[None, None, :] + int(z_offset)) % nz
    z = jnp.broadcast_to(z, values.shape)

    x = jnp.asarray(x_prime, dtype=jnp.float64)
    y = jnp.mod(jnp.asarray(y_prime, dtype=jnp.float64), float(ny))
    valid = (x >= 0.0) & (x <= float(nx - 1))
    x_clipped = jnp.clip(x, 0.0, float(nx - 1))
    x0 = jnp.floor(x_clipped).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, nx - 1)
    y0 = jnp.floor(y).astype(jnp.int32) % ny
    y1 = (y0 + 1) % ny
    wx = x_clipped - x0.astype(jnp.float64)
    wy = y - jnp.floor(y)

    f00 = values[x0, y0, z]
    f10 = values[x1, y0, z]
    f01 = values[x0, y1, z]
    f11 = values[x1, y1, z]
    interpolated = (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f10
        + (1.0 - wx) * wy * f01
        + wx * wy * f11
    )
    return jnp.where(valid, interpolated, jnp.asarray(boundary_value, dtype=jnp.float64))


def grad_parallel_op(field: jnp.ndarray, geometry: FciGeometry3D, *, boundary_value: float = 0.0) -> jnp.ndarray:
    """Return the centered FCI parallel gradient per toroidal arclength."""

    up = _interpolate_fci_plane(
        field,
        geometry.forward_x,
        geometry.forward_y,
        z_offset=1,
        boundary_value=boundary_value,
    )
    down = _interpolate_fci_plane(
        field,
        geometry.backward_x,
        geometry.backward_y,
        z_offset=-1,
        boundary_value=boundary_value,
    )
    forward_length = jnp.asarray(geometry.forward_length, dtype=jnp.float64)
    backward_length = jnp.asarray(geometry.backward_length, dtype=jnp.float64)
    segment_length = forward_length + backward_length
    return (up - down) / jnp.maximum(segment_length, 1.0e-30)


def debug_only_grad_parallel_op(
    field: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return a direct `b^j * df/dx^j` parallel gradient for comparison/debugging."""

    values = jnp.asarray(field, dtype=jnp.float64)
    dfdx = _first_derivative_3d(values, geometry.dx, axis=0, periodic=periodic_axes[0])
    dfdy = _first_derivative_3d(values, geometry.dy, axis=1, periodic=periodic_axes[1])
    dfdz = _first_derivative_3d(values, geometry.dz, axis=2, periodic=periodic_axes[2])
    df = jnp.stack((dfdx, dfdy, dfdz), axis=-1)

    b = jnp.asarray(geometry.B_contravariant, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.Bmag, dtype=jnp.float64), float(b_floor))
    b_unit = b / bmag[..., None]
    return jnp.einsum("...j,...j->...", b_unit, df)


def grad_parallel_op_np(field: np.ndarray, geometry: FciGeometry3D, *, boundary_value: float = 0.0) -> np.ndarray:
    """NumPy debug path for the centered FCI parallel gradient."""

    up = _interpolate_fci_plane_np(
        field,
        geometry.forward_x,
        geometry.forward_y,
        z_offset=1,
        boundary_value=boundary_value,
    )
    down = _interpolate_fci_plane_np(
        field,
        geometry.backward_x,
        geometry.backward_y,
        z_offset=-1,
        boundary_value=boundary_value,
    )
    segment_length = np.asarray(geometry.forward_length, dtype=np.float64) + np.asarray(geometry.backward_length, dtype=np.float64)
    return (up - down) / np.maximum(segment_length, 1.0e-30)


def debug_only_grad_parallel_op_np(
    field: np.ndarray,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
) -> np.ndarray:
    """NumPy debug path for the direct `b^j * df/dx^j` parallel gradient."""

    values = np.asarray(field, dtype=np.float64)
    dfdx = _first_derivative_3d_np(values, geometry.dx, axis=0, periodic=periodic_axes[0])
    dfdy = _first_derivative_3d_np(values, geometry.dy, axis=1, periodic=periodic_axes[1])
    dfdz = _first_derivative_3d_np(values, geometry.dz, axis=2, periodic=periodic_axes[2])
    df = np.stack((dfdx, dfdy, dfdz), axis=-1)

    b = np.asarray(geometry.B_contravariant, dtype=np.float64)
    bmag = np.maximum(np.asarray(geometry.Bmag, dtype=np.float64), float(b_floor))
    b_unit = b / bmag[..., None]
    return np.einsum("...j,...j->...", b_unit, df)


def grad_perp_op_np(
    field: np.ndarray,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
) -> np.ndarray:
    """NumPy debug path for the projected perpendicular gradient."""

    values = np.asarray(field, dtype=np.float64)
    dfdx = _first_derivative_3d_np(values, geometry.dx, axis=0, periodic=periodic_axes[0])
    dfdy = _first_derivative_3d_np(values, geometry.dy, axis=1, periodic=periodic_axes[1])
    dfdz = _first_derivative_3d_np(values, geometry.dz, axis=2, periodic=periodic_axes[2])
    df = np.stack((dfdx, dfdy, dfdz), axis=-1)

    g = np.stack(
        [
            np.stack([geometry.g11, geometry.g12, geometry.g13], axis=-1),
            np.stack([geometry.g12, geometry.g22, geometry.g23], axis=-1),
            np.stack([geometry.g13, geometry.g23, geometry.g33], axis=-1),
        ],
        axis=-2,
    )
    b = np.asarray(geometry.B_contravariant, dtype=np.float64)
    bmag = np.maximum(np.asarray(geometry.Bmag, dtype=np.float64), float(b_floor))
    b_unit = b / bmag[..., None]
    projector = g - np.einsum("...i,...j->...ij", b_unit, b_unit)
    return np.einsum("...ij,...j->...i", projector, df)


def poisson_bracket_op_np(
    f: np.ndarray,
    g: np.ndarray,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> np.ndarray:
    """NumPy debug path for the logical Poisson bracket."""

    f_values = np.asarray(f, dtype=np.float64)
    g_values = np.asarray(g, dtype=np.float64)
    df = np.stack(
        (
            _first_derivative_3d_np(f_values, geometry.dx, axis=0, periodic=periodic_axes[0]),
            _first_derivative_3d_np(f_values, geometry.dy, axis=1, periodic=periodic_axes[1]),
            _first_derivative_3d_np(f_values, geometry.dz, axis=2, periodic=periodic_axes[2]),
        ),
        axis=-1,
    )
    dg = np.stack(
        (
            _first_derivative_3d_np(g_values, geometry.dx, axis=0, periodic=periodic_axes[0]),
            _first_derivative_3d_np(g_values, geometry.dy, axis=1, periodic=periodic_axes[1]),
            _first_derivative_3d_np(g_values, geometry.dz, axis=2, periodic=periodic_axes[2]),
        ),
        axis=-1,
    )

    g_cov = np.stack(
        [
            np.stack([geometry.g_11, geometry.g_12, geometry.g_13], axis=-1),
            np.stack([geometry.g_12, geometry.g_22, geometry.g_23], axis=-1),
            np.stack([geometry.g_13, geometry.g_23, geometry.g_33], axis=-1),
        ],
        axis=-2,
    )
    b = np.asarray(geometry.B_contravariant, dtype=np.float64)
    bmag = np.maximum(np.asarray(geometry.Bmag, dtype=np.float64), float(b_floor))
    b_unit = b / bmag[..., None]
    b_covariant = np.einsum("...ij,...j->...i", g_cov, b_unit)
    cross = np.cross(df, dg)
    return np.sum(b_covariant * cross, axis=-1) / np.maximum(np.asarray(geometry.J, dtype=np.float64), float(jacobian_floor))


def _interpolate_fci_plane_np(
    field: np.ndarray,
    x_prime: np.ndarray,
    y_prime: np.ndarray,
    *,
    z_offset: int,
    boundary_value: float = 0.0,
) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    nx, ny, nz = values.shape
    z = (np.arange(nz, dtype=np.int32)[None, None, :] + int(z_offset)) % nz
    z = np.broadcast_to(z, values.shape)

    x = np.asarray(x_prime, dtype=np.float64)
    y = np.mod(np.asarray(y_prime, dtype=np.float64), float(ny))
    valid = (x >= 0.0) & (x <= float(nx - 1))
    x_clipped = np.clip(x, 0.0, float(nx - 1))
    x0 = np.floor(x_clipped).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y0 = np.floor(y).astype(np.int32) % ny
    y1 = (y0 + 1) % ny
    wx = x_clipped - x0.astype(np.float64)
    wy = y - np.floor(y)

    f00 = values[x0, y0, z]
    f10 = values[x1, y0, z]
    f01 = values[x0, y1, z]
    f11 = values[x1, y1, z]
    interpolated = (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f10
        + (1.0 - wx) * wy * f01
        + wx * wy * f11
    )
    return np.where(valid, interpolated, np.asarray(boundary_value, dtype=np.float64))


def _first_derivative_3d_np(values: np.ndarray, spacing: np.ndarray | float, *, axis: int, periodic: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    if periodic:
        return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (
            np.roll(spacing, -1, axis=axis) + np.roll(spacing, 1, axis=axis)
        )

    derivative = np.zeros_like(values, dtype=np.float64)
    if values.shape[axis] >= 3:
        interior = [slice(None)] * values.ndim
        interior[axis] = slice(1, -1)
        forward = np.roll(values, -1, axis=axis)
        backward = np.roll(values, 1, axis=axis)
        denom = np.roll(spacing, -1, axis=axis) + np.roll(spacing, 1, axis=axis)
        derivative[tuple(interior)] = (forward[tuple(interior)] - backward[tuple(interior)]) / denom[tuple(interior)]

    if values.shape[axis] >= 2:
        lower = [slice(None)] * values.ndim
        upper = [slice(None)] * values.ndim
        edge = [slice(None)] * values.ndim
        lower[axis] = 0
        upper[axis] = 1
        edge[axis] = 2 if values.shape[axis] > 2 else 1
        derivative[tuple(lower)] = (
            -3.0 * values[tuple(lower)]
            + 4.0 * values[tuple(upper)]
            - values[tuple(edge)]
        ) / (spacing[tuple(lower)] + spacing[tuple(upper)])
        lower[axis] = -1
        upper[axis] = -2
        edge[axis] = -3 if values.shape[axis] > 2 else -2
        derivative[tuple(lower)] = (
            3.0 * values[tuple(lower)]
            - 4.0 * values[tuple(upper)]
            + values[tuple(edge)]
        ) / (spacing[tuple(lower)] + spacing[tuple(upper)])
    return derivative


def grad_perp_op(
    field: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the projected perpendicular gradient in logical contravariant components."""

    values = jnp.asarray(field, dtype=jnp.float64)
    dfdx = _first_derivative_3d(values, geometry.dx, axis=0, periodic=periodic_axes[0])
    dfdy = _first_derivative_3d(values, geometry.dy, axis=1, periodic=periodic_axes[1])
    dfdz = _first_derivative_3d(values, geometry.dz, axis=2, periodic=periodic_axes[2])
    df = jnp.stack((dfdx, dfdy, dfdz), axis=-1)

    g = jnp.stack(
        [
            jnp.stack([geometry.g11, geometry.g12, geometry.g13], axis=-1),
            jnp.stack([geometry.g12, geometry.g22, geometry.g23], axis=-1),
            jnp.stack([geometry.g13, geometry.g23, geometry.g33], axis=-1),
        ],
        axis=-2,
    )
    b = jnp.asarray(geometry.B_contravariant, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.Bmag, dtype=jnp.float64), float(b_floor))
    b_unit = b / bmag[..., None]
    projector = g - jnp.einsum("...i,...j->...ij", b_unit, b_unit)
    return jnp.einsum("...ij,...j->...i", projector, df)


def poisson_bracket_op(
    f: jnp.ndarray,
    g: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the logical Poisson bracket using the covariant Levi-Civita form."""

    metric = geometry
    f_values = jnp.asarray(f, dtype=jnp.float64)
    g_values = jnp.asarray(g, dtype=jnp.float64)
    df = jnp.stack(
        (
            _first_derivative_3d(f_values, geometry.dx, axis=0, periodic=periodic_axes[0]),
            _first_derivative_3d(f_values, geometry.dy, axis=1, periodic=periodic_axes[1]),
            _first_derivative_3d(f_values, geometry.dz, axis=2, periodic=periodic_axes[2]),
        ),
        axis=-1,
    )
    dg = jnp.stack(
        (
            _first_derivative_3d(g_values, geometry.dx, axis=0, periodic=periodic_axes[0]),
            _first_derivative_3d(g_values, geometry.dy, axis=1, periodic=periodic_axes[1]),
            _first_derivative_3d(g_values, geometry.dz, axis=2, periodic=periodic_axes[2]),
        ),
        axis=-1,
    )

    g_cov = jnp.stack(
        [
            jnp.stack([geometry.g_11, geometry.g_12, geometry.g_13], axis=-1),
            jnp.stack([geometry.g_12, geometry.g_22, geometry.g_23], axis=-1),
            jnp.stack([geometry.g_13, geometry.g_23, geometry.g_33], axis=-1),
        ],
        axis=-2,
    )
    b = jnp.asarray(geometry.B_contravariant, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.Bmag, dtype=jnp.float64), float(b_floor))
    b_unit = b / bmag[..., None]
    b_covariant = jnp.einsum("...ij,...j->...i", g_cov, b_unit)
    cross = jnp.cross(df, dg)
    return jnp.sum(b_covariant * cross, axis=-1) / jnp.maximum(
        jnp.asarray(metric.J, dtype=jnp.float64),
        float(jacobian_floor),
    )


def curvature_op(
    field: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the curvature operator applied to a scalar field."""

    metric = geometry
    values = jnp.asarray(field, dtype=jnp.float64)
    dfdx = _first_derivative_3d(values, geometry.dx, axis=0, periodic=periodic_axes[0])
    dfdy = _first_derivative_3d(values, geometry.dy, axis=1, periodic=periodic_axes[1])
    dfdz = _first_derivative_3d(values, geometry.dz, axis=2, periodic=periodic_axes[2])
    grad_f = jnp.stack((dfdx, dfdy, dfdz), axis=-1)

    g_cov = jnp.stack(
        [
            jnp.stack([metric.g_11, metric.g_12, metric.g_13], axis=-1),
            jnp.stack([metric.g_12, metric.g_22, metric.g_23], axis=-1),
            jnp.stack([metric.g_13, metric.g_23, metric.g_33], axis=-1),
        ],
        axis=-2,
    )
    b = jnp.asarray(metric.B_contravariant, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(metric.Bmag, dtype=jnp.float64), float(b_floor))
    b_unit = b / bmag[..., None]
    covariant_field = jnp.einsum("...ij,...j->...i", g_cov, b_unit) / bmag[..., None]

    dcov_dx = jnp.stack(
        [
            _first_derivative_3d(covariant_field[..., 0], geometry.dx, axis=0, periodic=periodic_axes[0]),
            _first_derivative_3d(covariant_field[..., 1], geometry.dx, axis=0, periodic=periodic_axes[0]),
            _first_derivative_3d(covariant_field[..., 2], geometry.dx, axis=0, periodic=periodic_axes[0]),
        ],
        axis=-1,
    )
    dcov_dy = jnp.stack(
        [
            _first_derivative_3d(covariant_field[..., 0], geometry.dy, axis=1, periodic=periodic_axes[1]),
            _first_derivative_3d(covariant_field[..., 1], geometry.dy, axis=1, periodic=periodic_axes[1]),
            _first_derivative_3d(covariant_field[..., 2], geometry.dy, axis=1, periodic=periodic_axes[1]),
        ],
        axis=-1,
    )
    dcov_dz = jnp.stack(
        [
            _first_derivative_3d(covariant_field[..., 0], geometry.dz, axis=2, periodic=periodic_axes[2]),
            _first_derivative_3d(covariant_field[..., 1], geometry.dz, axis=2, periodic=periodic_axes[2]),
            _first_derivative_3d(covariant_field[..., 2], geometry.dz, axis=2, periodic=periodic_axes[2]),
        ],
        axis=-1,
    )

    curl = jnp.stack(
        (
            dcov_dy[..., 2] - dcov_dz[..., 1],
            dcov_dz[..., 0] - dcov_dx[..., 2],
            dcov_dx[..., 1] - dcov_dy[..., 0],
        ),
        axis=-1,
    )
    coefficient = bmag / (2.0 * jnp.maximum(jnp.asarray(metric.J, dtype=jnp.float64), float(jacobian_floor)))
    return coefficient * jnp.sum(curl * grad_f, axis=-1)


def curvature_op_np(
    field: np.ndarray,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> np.ndarray:
    """NumPy debug path for the curvature operator."""

    metric = geometry
    values = np.asarray(field, dtype=np.float64)
    dfdx = _first_derivative_3d_np(values, geometry.dx, axis=0, periodic=periodic_axes[0])
    dfdy = _first_derivative_3d_np(values, geometry.dy, axis=1, periodic=periodic_axes[1])
    dfdz = _first_derivative_3d_np(values, geometry.dz, axis=2, periodic=periodic_axes[2])
    grad_f = np.stack((dfdx, dfdy, dfdz), axis=-1)

    g_cov = np.stack(
        [
            np.stack([metric.g_11, metric.g_12, metric.g_13], axis=-1),
            np.stack([metric.g_12, metric.g_22, metric.g_23], axis=-1),
            np.stack([metric.g_13, metric.g_23, metric.g_33], axis=-1),
        ],
        axis=-2,
    )
    b = np.asarray(metric.B_contravariant, dtype=np.float64)
    bmag = np.maximum(np.asarray(metric.Bmag, dtype=np.float64), float(b_floor))
    b_unit = b / bmag[..., None]
    covariant_field = np.einsum("...ij,...j->...i", g_cov, b_unit) / bmag[..., None]

    dcov_dx = np.stack(
        [
            _first_derivative_3d_np(covariant_field[..., 0], geometry.dx, axis=0, periodic=periodic_axes[0]),
            _first_derivative_3d_np(covariant_field[..., 1], geometry.dx, axis=0, periodic=periodic_axes[0]),
            _first_derivative_3d_np(covariant_field[..., 2], geometry.dx, axis=0, periodic=periodic_axes[0]),
        ],
        axis=-1,
    )
    dcov_dy = np.stack(
        [
            _first_derivative_3d_np(covariant_field[..., 0], geometry.dy, axis=1, periodic=periodic_axes[1]),
            _first_derivative_3d_np(covariant_field[..., 1], geometry.dy, axis=1, periodic=periodic_axes[1]),
            _first_derivative_3d_np(covariant_field[..., 2], geometry.dy, axis=1, periodic=periodic_axes[1]),
        ],
        axis=-1,
    )
    dcov_dz = np.stack(
        [
            _first_derivative_3d_np(covariant_field[..., 0], geometry.dz, axis=2, periodic=periodic_axes[2]),
            _first_derivative_3d_np(covariant_field[..., 1], geometry.dz, axis=2, periodic=periodic_axes[2]),
            _first_derivative_3d_np(covariant_field[..., 2], geometry.dz, axis=2, periodic=periodic_axes[2]),
        ],
        axis=-1,
    )

    curl = np.stack(
        (
            dcov_dy[..., 2] - dcov_dz[..., 1],
            dcov_dz[..., 0] - dcov_dx[..., 2],
            dcov_dx[..., 1] - dcov_dy[..., 0],
        ),
        axis=-1,
    )
    coefficient = bmag / (2.0 * np.maximum(np.asarray(metric.J, dtype=np.float64), float(jacobian_floor)))
    return coefficient * np.sum(curl * grad_f, axis=-1)


def _face_average_3d_np(values: np.ndarray, *, axis: int, periodic: bool) -> np.ndarray:
    face = 0.5 * (values + np.roll(values, -1, axis=axis))
    if periodic:
        return face
    last = [slice(None), slice(None), slice(None)]
    last[axis] = -1
    face = face.copy()
    face[tuple(last)] = values[tuple(last)]
    return face


def _axis_index_nd(axis: int, index: int, ndim: int) -> tuple[object, ...]:
    slices: list[object] = [slice(None)] * ndim
    slices[axis] = index
    return tuple(slices)


def _axis_slice_nd(axis: int, start: int | None, stop: int | None, ndim: int) -> tuple[object, ...]:
    slices: list[object] = [slice(None)] * ndim
    slices[axis] = slice(start, stop)
    return tuple(slices)


def _face_interpolate_3d_order4_np(values: np.ndarray, *, axis: int, periodic: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if periodic:
        return (-np.roll(values, 1, axis=axis) + 9.0 * values + 9.0 * np.roll(values, -1, axis=axis) - np.roll(values, -2, axis=axis)) / 16.0

    if values.shape[axis] < 4:
        raise ValueError("Fourth-order face interpolation requires at least 4 points along the selected axis")
    face = np.zeros_like(values, dtype=np.float64)
    ndim = values.ndim
    face[_axis_index_nd(axis, 0, ndim)] = (
        5.0 * values[_axis_index_nd(axis, 0, ndim)]
        + 15.0 * values[_axis_index_nd(axis, 1, ndim)]
        - 5.0 * values[_axis_index_nd(axis, 2, ndim)]
        + values[_axis_index_nd(axis, 3, ndim)]
    ) / 16.0
    face[_axis_slice_nd(axis, 1, -2, ndim)] = (
        -values[_axis_slice_nd(axis, None, -3, ndim)]
        + 9.0 * values[_axis_slice_nd(axis, 1, -2, ndim)]
        + 9.0 * values[_axis_slice_nd(axis, 2, -1, ndim)]
        - values[_axis_slice_nd(axis, 3, None, ndim)]
    ) / 16.0
    face[_axis_index_nd(axis, -2, ndim)] = (
        values[_axis_index_nd(axis, -4, ndim)]
        - 5.0 * values[_axis_index_nd(axis, -3, ndim)]
        + 15.0 * values[_axis_index_nd(axis, -2, ndim)]
        + 5.0 * values[_axis_index_nd(axis, -1, ndim)]
    ) / 16.0
    face[_axis_index_nd(axis, -1, ndim)] = values[_axis_index_nd(axis, -1, ndim)]
    return face


def _face_derivative_3d_order4_np(values: np.ndarray, spacing: np.ndarray | float, *, axis: int, periodic: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    h = np.asarray(spacing, dtype=np.float64)
    if h.ndim == 0:
        h = np.ones_like(values) * h
    h = np.maximum(h, 1.0e-30)
    if periodic:
        return (np.roll(values, 1, axis=axis) - 27.0 * values + 27.0 * np.roll(values, -1, axis=axis) - np.roll(values, -2, axis=axis)) / (24.0 * h)

    if values.shape[axis] < 4:
        raise ValueError("Fourth-order face derivative requires at least 4 points along the selected axis")
    derivative = np.zeros_like(values, dtype=np.float64)
    ndim = values.ndim
    derivative[_axis_index_nd(axis, 0, ndim)] = (
        -23.0 * values[_axis_index_nd(axis, 0, ndim)]
        + 21.0 * values[_axis_index_nd(axis, 1, ndim)]
        + 3.0 * values[_axis_index_nd(axis, 2, ndim)]
        - values[_axis_index_nd(axis, 3, ndim)]
    ) / (24.0 * h[_axis_index_nd(axis, 0, ndim)])
    derivative[_axis_slice_nd(axis, 1, -2, ndim)] = (
        values[_axis_slice_nd(axis, None, -3, ndim)]
        - 27.0 * values[_axis_slice_nd(axis, 1, -2, ndim)]
        + 27.0 * values[_axis_slice_nd(axis, 2, -1, ndim)]
        - values[_axis_slice_nd(axis, 3, None, ndim)]
    ) / (24.0 * h[_axis_slice_nd(axis, 1, -2, ndim)])
    derivative[_axis_index_nd(axis, -2, ndim)] = (
        values[_axis_index_nd(axis, -4, ndim)]
        - 3.0 * values[_axis_index_nd(axis, -3, ndim)]
        - 21.0 * values[_axis_index_nd(axis, -2, ndim)]
        + 23.0 * values[_axis_index_nd(axis, -1, ndim)]
    ) / (24.0 * h[_axis_index_nd(axis, -2, ndim)])
    return derivative


def _face_forward_difference_3d_np(
    values: np.ndarray,
    spacing: np.ndarray | float,
    *,
    axis: int,
    periodic: bool,
) -> np.ndarray:
    h = np.asarray(spacing, dtype=np.float64)
    if h.ndim == 0:
        h = np.ones_like(values) * h
    face_h = _face_average_3d_np(h, axis=axis, periodic=periodic)
    difference = (np.roll(values, -1, axis=axis) - values) / np.maximum(face_h, 1.0e-30)
    if periodic:
        return difference
    last = [slice(None), slice(None), slice(None)]
    penultimate = [slice(None), slice(None), slice(None)]
    antepenultimate = [slice(None), slice(None), slice(None)]
    last[axis] = -1
    penultimate[axis] = -2
    antepenultimate[axis] = -3
    backward = (3.0 * values[tuple(last)] - 4.0 * values[tuple(penultimate)] + values[tuple(antepenultimate)]) / np.maximum(
        2.0 * face_h[tuple(last)],
        1.0e-30,
    )
    difference = difference.copy()
    difference[tuple(last)] = backward
    return difference


@partial(jax.jit, static_argnames=("b_floor", "jacobian_floor"))
def perp_laplacian_op(
    field: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    bc: FciBoundaryCondition,
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the conservative perpendicular Laplacian.

    Boundary conditions for ``bc`` use ``FciBoundaryCondition`` with strict
    logical-coordinate entries ``x``, ``y``, and ``z``. Each non-periodic axis
    must provide an ``FciAxisBoundaryCondition``.
    """

    periodic_axes = _bc_periodic_axes(bc)
    axis_bcs = tuple(_bc_axis_spec(bc, axis, periodic_axes=periodic_axes) for axis in range(3))
    values = jnp.broadcast_to(jnp.asarray(field, dtype=jnp.float64), geometry.shape)
    g = jnp.stack(
        [
            jnp.stack([geometry.g11, geometry.g12, geometry.g13], axis=-1),
            jnp.stack([geometry.g12, geometry.g22, geometry.g23], axis=-1),
            jnp.stack([geometry.g13, geometry.g23, geometry.g33], axis=-1),
        ],
        axis=-2,
    )
    b = jnp.asarray(geometry.B_contravariant, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(geometry.Bmag, dtype=jnp.float64), float(b_floor))
    b_unit = b / bmag[..., None]
    projector = g - jnp.einsum("...i,...j->...ij", b_unit, b_unit)

    jac = jnp.asarray(geometry.J, dtype=jnp.float64)
    dfdx_node = _first_derivative_3d(values, geometry.dx, axis=0, periodic=periodic_axes[0])
    dfdy_node = _first_derivative_3d(values, geometry.dy, axis=1, periodic=periodic_axes[1])
    dfdz_node = _first_derivative_3d(values, geometry.dz, axis=2, periodic=periodic_axes[2])

    x_face_grad = jnp.stack(
        (
            _face_derivative_3d_order4(values, geometry.dx, axis=0, periodic=periodic_axes[0]),
            _face_interpolate_3d_order4(dfdy_node, axis=0, periodic=periodic_axes[0]),
            _face_interpolate_3d_order4(dfdz_node, axis=0, periodic=periodic_axes[0]),
        ),
        axis=-1,
    )
    y_face_grad = jnp.stack(
        (
            _face_interpolate_3d_order4(dfdx_node, axis=1, periodic=periodic_axes[1]),
            _face_derivative_3d_order4(values, geometry.dy, axis=1, periodic=periodic_axes[1]),
            _face_interpolate_3d_order4(dfdz_node, axis=1, periodic=periodic_axes[1]),
        ),
        axis=-1,
    )
    z_face_grad = jnp.stack(
        (
            _face_interpolate_3d_order4(dfdx_node, axis=2, periodic=periodic_axes[2]),
            _face_interpolate_3d_order4(dfdy_node, axis=2, periodic=periodic_axes[2]),
            _face_derivative_3d_order4(values, geometry.dz, axis=2, periodic=periodic_axes[2]),
        ),
        axis=-1,
    )

    x_face_projector = _face_interpolate_3d_order4(projector, axis=0, periodic=periodic_axes[0])
    y_face_projector = _face_interpolate_3d_order4(projector, axis=1, periodic=periodic_axes[1])
    z_face_projector = _face_interpolate_3d_order4(projector, axis=2, periodic=periodic_axes[2])
    x_face_jac = _face_interpolate_3d_order4(jac, axis=0, periodic=periodic_axes[0])
    y_face_jac = _face_interpolate_3d_order4(jac, axis=1, periodic=periodic_axes[1])
    z_face_jac = _face_interpolate_3d_order4(jac, axis=2, periodic=periodic_axes[2])

    x_flux = x_face_jac * jnp.einsum("...j,...j->...", x_face_projector[..., 0, :], x_face_grad)
    y_flux = y_face_jac * jnp.einsum("...j,...j->...", y_face_projector[..., 1, :], y_face_grad)
    z_flux = z_face_jac * jnp.einsum("...j,...j->...", z_face_projector[..., 2, :], z_face_grad)

    x_lower, x_upper = _axis_boundary_fluxes(x_flux, axis=0, axis_bc=axis_bcs[0]) if not periodic_axes[0] else (0.0, 0.0)
    y_lower, y_upper = _axis_boundary_fluxes(y_flux, axis=1, axis_bc=axis_bcs[1]) if not periodic_axes[1] else (0.0, 0.0)
    z_lower, z_upper = _axis_boundary_fluxes(z_flux, axis=2, axis_bc=axis_bcs[2]) if not periodic_axes[2] else (0.0, 0.0)

    div_flux = (
        _axis_divergence_from_flux(x_flux, geometry.dx, axis=0, periodic=periodic_axes[0], lower_flux=x_lower, upper_flux=x_upper)
        + _axis_divergence_from_flux(y_flux, geometry.dy, axis=1, periodic=periodic_axes[1], lower_flux=y_lower, upper_flux=y_upper)
        + _axis_divergence_from_flux(z_flux, geometry.dz, axis=2, periodic=periodic_axes[2], lower_flux=z_lower, upper_flux=z_upper)
    )
    return div_flux / jnp.maximum(jac, float(jacobian_floor))


def perp_laplacian_op_np(
    field: np.ndarray,
    geometry: FciGeometry3D,
    *,
    bc: FciBoundaryCondition,
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> np.ndarray:
    """NumPy debug twin of ``perp_laplacian_op``.

    This mirrors the consolidated JAX implementation while using the NumPy
    derivative, interpolation, boundary-flux, and divergence helpers.
    """

    periodic_axes = _bc_periodic_axes(bc)
    axis_bcs = tuple(_bc_axis_spec(bc, axis, periodic_axes=periodic_axes) for axis in range(3))
    values = np.broadcast_to(np.asarray(field, dtype=np.float64), geometry.shape)
    g = np.stack(
        [
            np.stack([geometry.g11, geometry.g12, geometry.g13], axis=-1),
            np.stack([geometry.g12, geometry.g22, geometry.g23], axis=-1),
            np.stack([geometry.g13, geometry.g23, geometry.g33], axis=-1),
        ],
        axis=-2,
    )
    b = np.asarray(geometry.B_contravariant, dtype=np.float64)
    bmag = np.maximum(np.asarray(geometry.Bmag, dtype=np.float64), float(b_floor))
    b_unit = b / bmag[..., None]
    projector = g - np.einsum("...i,...j->...ij", b_unit, b_unit)

    jac = np.asarray(geometry.J, dtype=np.float64)
    dfdx_node = _first_derivative_3d_np(values, geometry.dx, axis=0, periodic=periodic_axes[0])
    dfdy_node = _first_derivative_3d_np(values, geometry.dy, axis=1, periodic=periodic_axes[1])
    dfdz_node = _first_derivative_3d_np(values, geometry.dz, axis=2, periodic=periodic_axes[2])

    x_face_grad = np.stack(
        (
            _face_derivative_3d_order4_np(values, geometry.dx, axis=0, periodic=periodic_axes[0]),
            _face_interpolate_3d_order4_np(dfdy_node, axis=0, periodic=periodic_axes[0]),
            _face_interpolate_3d_order4_np(dfdz_node, axis=0, periodic=periodic_axes[0]),
        ),
        axis=-1,
    )
    y_face_grad = np.stack(
        (
            _face_interpolate_3d_order4_np(dfdx_node, axis=1, periodic=periodic_axes[1]),
            _face_derivative_3d_order4_np(values, geometry.dy, axis=1, periodic=periodic_axes[1]),
            _face_interpolate_3d_order4_np(dfdz_node, axis=1, periodic=periodic_axes[1]),
        ),
        axis=-1,
    )
    z_face_grad = np.stack(
        (
            _face_interpolate_3d_order4_np(dfdx_node, axis=2, periodic=periodic_axes[2]),
            _face_interpolate_3d_order4_np(dfdy_node, axis=2, periodic=periodic_axes[2]),
            _face_derivative_3d_order4_np(values, geometry.dz, axis=2, periodic=periodic_axes[2]),
        ),
        axis=-1,
    )

    x_face_projector = _face_interpolate_3d_order4_np(projector, axis=0, periodic=periodic_axes[0])
    y_face_projector = _face_interpolate_3d_order4_np(projector, axis=1, periodic=periodic_axes[1])
    z_face_projector = _face_interpolate_3d_order4_np(projector, axis=2, periodic=periodic_axes[2])
    x_face_jac = _face_interpolate_3d_order4_np(jac, axis=0, periodic=periodic_axes[0])
    y_face_jac = _face_interpolate_3d_order4_np(jac, axis=1, periodic=periodic_axes[1])
    z_face_jac = _face_interpolate_3d_order4_np(jac, axis=2, periodic=periodic_axes[2])

    x_flux = x_face_jac * np.einsum("...j,...j->...", x_face_projector[..., 0, :], x_face_grad)
    y_flux = y_face_jac * np.einsum("...j,...j->...", y_face_projector[..., 1, :], y_face_grad)
    z_flux = z_face_jac * np.einsum("...j,...j->...", z_face_projector[..., 2, :], z_face_grad)

    x_lower, x_upper = _axis_boundary_fluxes_np(x_flux, axis=0, axis_bc=axis_bcs[0]) if not periodic_axes[0] else (0.0, 0.0)
    y_lower, y_upper = _axis_boundary_fluxes_np(y_flux, axis=1, axis_bc=axis_bcs[1]) if not periodic_axes[1] else (0.0, 0.0)
    z_lower, z_upper = _axis_boundary_fluxes_np(z_flux, axis=2, axis_bc=axis_bcs[2]) if not periodic_axes[2] else (0.0, 0.0)

    div_flux = (
        _axis_divergence_from_flux_np(x_flux, geometry.dx, axis=0, periodic=periodic_axes[0], lower_flux=x_lower, upper_flux=x_upper)
        + _axis_divergence_from_flux_np(y_flux, geometry.dy, axis=1, periodic=periodic_axes[1], lower_flux=y_lower, upper_flux=y_upper)
        + _axis_divergence_from_flux_np(z_flux, geometry.dz, axis=2, periodic=periodic_axes[2], lower_flux=z_lower, upper_flux=z_upper)
    )
    return div_flux / np.maximum(jac, float(jacobian_floor))


def _axis_name(axis: int) -> str:
    return ("x", "y", "z")[int(axis)]


def _bc_periodic_axes(bc) -> tuple[bool, bool, bool]:
    if not isinstance(bc, FciBoundaryCondition):
        raise TypeError("bc must be an FciBoundaryCondition")
    periodic_axes = bc.periodic_axes
    if len(periodic_axes) != 3:
        raise ValueError("bc.periodic_axes must have length 3")
    return tuple(bool(value) for value in periodic_axes)


def _bc_axis_spec(bc, axis: int, *, periodic_axes: tuple[bool, bool, bool]) -> object | None:
    if periodic_axes[axis]:
        return None
    axis_bc = (bc.x, bc.y, bc.z)[axis]
    if not isinstance(axis_bc, FciAxisBoundaryCondition):
        raise TypeError(f"bc.{_axis_name(axis)} must be an FciAxisBoundaryCondition")
    return axis_bc


def _homogeneous_axis_bc(axis_bc: object | None) -> object | None:
    if axis_bc is None:
        return None
    kind = _bc_kind(axis_bc)
    if kind == "dirichlet":
        return FciAxisBoundaryCondition.dirichlet(lower_value=0.0, upper_value=0.0)
    if kind == "neumann_flux":
        return FciAxisBoundaryCondition.neumann_flux(lower_flux=0.0, upper_flux=0.0)
    raise ValueError("boundary condition kind must be 'dirichlet' or 'neumann_flux'")


def _homogeneous_bc(bc: FciBoundaryCondition) -> FciBoundaryCondition:
    periodic_axes = _bc_periodic_axes(bc)
    return FciBoundaryCondition(
        periodic_axes=periodic_axes,
        x=_homogeneous_axis_bc(bc.x),
        y=_homogeneous_axis_bc(bc.y),
        z=_homogeneous_axis_bc(bc.z),
        target_mean_phi=0.0,
    )


def _bc_kind(bc) -> str:
    if not isinstance(bc, FciAxisBoundaryCondition):
        raise TypeError("axis boundary condition must be an FciAxisBoundaryCondition")
    kind = bc.kind
    if kind not in {"dirichlet", "neumann_flux"}:
        raise ValueError("boundary condition kind must be 'dirichlet' or 'neumann_flux'")
    return kind


def _broadcast_boundary_value(value, target_shape: tuple[int, int]) -> jnp.ndarray:
    array = jnp.asarray(value, dtype=jnp.float64)
    return jnp.broadcast_to(array, target_shape)


def _broadcast_axis_boundary_value(value, *, axis: int, field_shape: tuple[int, int, int]) -> jnp.ndarray:
    target_shape = tuple(field_shape[index] for index in range(3) if index != axis)
    return _broadcast_boundary_value(value, target_shape)


def _set_axis_plane(field: jnp.ndarray, *, axis: int, index: int, value: jnp.ndarray) -> jnp.ndarray:
    if _NUMPY_DEBUG:
        result = np.array(field, copy=True)
        result[_axis_index_nd(axis, index, result.ndim)] = np.asarray(value, dtype=np.float64)
        return result
    return field.at[_axis_index_nd(axis, index, field.ndim)].set(value)


def _apply_dirichlet_constraints(
    field: jnp.ndarray,
    *,
    axis_bcs: tuple[object | None, object | None, object | None],
    periodic_axes: tuple[bool, bool, bool],
) -> jnp.ndarray:
    constrained = jnp.asarray(field, dtype=jnp.float64)
    for axis, axis_bc in enumerate(axis_bcs):
        if periodic_axes[axis] or axis_bc is None:
            continue
        if _bc_kind(axis_bc) != "dirichlet":
            continue
        constrained = _set_axis_plane(
            constrained,
            axis=axis,
            index=0,
            value=_broadcast_axis_boundary_value(axis_bc.lower_value, axis=axis, field_shape=constrained.shape),
        )
        constrained = _set_axis_plane(
            constrained,
            axis=axis,
            index=-1,
            value=_broadcast_axis_boundary_value(axis_bc.upper_value, axis=axis, field_shape=constrained.shape),
        )
    return constrained


def _zero_dirichlet_boundary_residual(
    field: jnp.ndarray,
    *,
    axis_bcs: tuple[object | None, object | None, object | None],
    periodic_axes: tuple[bool, bool, bool],
) -> jnp.ndarray:
    residual = jnp.asarray(field, dtype=jnp.float64)
    for axis, axis_bc in enumerate(axis_bcs):
        if periodic_axes[axis] or axis_bc is None or _bc_kind(axis_bc) != "dirichlet":
            continue
        residual = _set_axis_plane(residual, axis=axis, index=0, value=jnp.zeros_like(residual[_axis_index_nd(axis, 0, residual.ndim)]))
        residual = _set_axis_plane(residual, axis=axis, index=-1, value=jnp.zeros_like(residual[_axis_index_nd(axis, -1, residual.ndim)]))
    return residual


def _axis_boundary_fluxes(
    flux: jnp.ndarray,
    *,
    axis: int,
    axis_bc: object,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    kind = _bc_kind(axis_bc)
    if kind == "neumann_flux":
        lower = _broadcast_axis_boundary_value(axis_bc.lower_flux, axis=axis, field_shape=flux.shape)
        upper = _broadcast_axis_boundary_value(axis_bc.upper_flux, axis=axis, field_shape=flux.shape)
        return lower, upper
    if kind == "dirichlet":
        if flux.shape[axis] < 4:
            raise ValueError("dirichlet boundary extrapolation requires at least 4 points along the selected axis")
        ndim = flux.ndim
        lower = (
            4.0 * flux[_axis_index_nd(axis, 0, ndim)]
            - 6.0 * flux[_axis_index_nd(axis, 1, ndim)]
            + 4.0 * flux[_axis_index_nd(axis, 2, ndim)]
            - flux[_axis_index_nd(axis, 3, ndim)]
        )
        upper = (
            4.0 * flux[_axis_index_nd(axis, -1, ndim)]
            - 6.0 * flux[_axis_index_nd(axis, -2, ndim)]
            + 4.0 * flux[_axis_index_nd(axis, -3, ndim)]
            - flux[_axis_index_nd(axis, -4, ndim)]
        )
        return lower, upper
    raise ValueError("boundary condition kind must be 'dirichlet' or 'neumann_flux'")


def _axis_divergence_from_flux(
    flux: jnp.ndarray,
    spacing: jnp.ndarray | float,
    *,
    axis: int,
    periodic: bool,
    lower_flux: jnp.ndarray | float | None = None,
    upper_flux: jnp.ndarray | float | None = None,
) -> jnp.ndarray:
    h = jnp.asarray(spacing, dtype=jnp.float64)
    if h.ndim == 0:
        h = jnp.ones_like(flux) * h
    h = jnp.maximum(h, 1.0e-30)

    if periodic:
        return (flux - jnp.roll(flux, 1, axis=axis)) / h

    ndim = flux.ndim
    if flux.shape[axis] < 4:
        raise ValueError("Open-axis divergence requires at least 4 points along the selected axis")
    div = jnp.zeros_like(flux)
    lower = jnp.asarray(lower_flux, dtype=jnp.float64)
    upper = jnp.asarray(upper_flux, dtype=jnp.float64)
    div = div.at[_axis_index_nd(axis, 0, ndim)].set(
        ((-8.0 / 3.0) * lower + 3.0 * flux[_axis_index_nd(axis, 0, ndim)] - (1.0 / 3.0) * flux[_axis_index_nd(axis, 1, ndim)])
        / h[_axis_index_nd(axis, 0, ndim)]
    )
    div = div.at[_axis_slice_nd(axis, 1, -1, ndim)].set(
        (
            flux[_axis_slice_nd(axis, 1, -1, ndim)]
            - flux[_axis_slice_nd(axis, None, -2, ndim)]
        )
        / h[_axis_slice_nd(axis, 1, -1, ndim)]
    )
    div = div.at[_axis_index_nd(axis, -1, ndim)].set(
        ((8.0 / 3.0) * upper - 3.0 * flux[_axis_index_nd(axis, -2, ndim)] + (1.0 / 3.0) * flux[_axis_index_nd(axis, -3, ndim)])
        / h[_axis_index_nd(axis, -1, ndim)]
    )
    return div


def _axis_boundary_fluxes_np(
    flux: np.ndarray,
    *,
    axis: int,
    axis_bc: object,
) -> tuple[np.ndarray, np.ndarray]:
    kind = _bc_kind(axis_bc)
    if kind == "neumann_flux":
        target_shape = tuple(flux.shape[index] for index in range(3) if index != axis)
        lower = np.broadcast_to(np.asarray(axis_bc.lower_flux, dtype=np.float64), target_shape)
        upper = np.broadcast_to(np.asarray(axis_bc.upper_flux, dtype=np.float64), target_shape)
        return lower, upper
    if kind == "dirichlet":
        if flux.shape[axis] < 4:
            raise ValueError("dirichlet boundary extrapolation requires at least 4 points along the selected axis")
        ndim = flux.ndim
        lower = (
            4.0 * flux[_axis_index_nd(axis, 0, ndim)]
            - 6.0 * flux[_axis_index_nd(axis, 1, ndim)]
            + 4.0 * flux[_axis_index_nd(axis, 2, ndim)]
            - flux[_axis_index_nd(axis, 3, ndim)]
        )
        upper = (
            4.0 * flux[_axis_index_nd(axis, -1, ndim)]
            - 6.0 * flux[_axis_index_nd(axis, -2, ndim)]
            + 4.0 * flux[_axis_index_nd(axis, -3, ndim)]
            - flux[_axis_index_nd(axis, -4, ndim)]
        )
        return lower, upper
    raise ValueError("boundary condition kind must be 'dirichlet' or 'neumann_flux'")


def _axis_divergence_from_flux_np(
    flux: np.ndarray,
    spacing: np.ndarray | float,
    *,
    axis: int,
    periodic: bool,
    lower_flux: np.ndarray | float | None = None,
    upper_flux: np.ndarray | float | None = None,
) -> np.ndarray:
    h = np.asarray(spacing, dtype=np.float64)
    if h.ndim == 0:
        h = np.ones_like(flux) * h
    h = np.maximum(h, 1.0e-30)

    if periodic:
        return (flux - np.roll(flux, 1, axis=axis)) / h

    ndim = flux.ndim
    if flux.shape[axis] < 4:
        raise ValueError("Open-axis divergence requires at least 4 points along the selected axis")
    div = np.zeros_like(flux, dtype=np.float64)
    lower = np.asarray(lower_flux, dtype=np.float64)
    upper = np.asarray(upper_flux, dtype=np.float64)
    div[_axis_index_nd(axis, 0, ndim)] = (
        ((-8.0 / 3.0) * lower + 3.0 * flux[_axis_index_nd(axis, 0, ndim)] - (1.0 / 3.0) * flux[_axis_index_nd(axis, 1, ndim)])
        / h[_axis_index_nd(axis, 0, ndim)]
    )
    div[_axis_slice_nd(axis, 1, -1, ndim)] = (
        (
            flux[_axis_slice_nd(axis, 1, -1, ndim)]
            - flux[_axis_slice_nd(axis, None, -2, ndim)]
        )
        / h[_axis_slice_nd(axis, 1, -1, ndim)]
    )
    div[_axis_index_nd(axis, -1, ndim)] = (
        ((8.0 / 3.0) * upper - 3.0 * flux[_axis_index_nd(axis, -2, ndim)] + (1.0 / 3.0) * flux[_axis_index_nd(axis, -3, ndim)])
        / h[_axis_index_nd(axis, -1, ndim)]
    )
    return div


def _weighted_mean(field: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    weights = jnp.asarray(geometry.J, dtype=jnp.float64)
    return jnp.sum(weights * field) / jnp.maximum(jnp.sum(weights), 1.0e-30)


def _remove_weighted_mean(field: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    values = jnp.asarray(field, dtype=jnp.float64)
    return values - _weighted_mean(values, geometry)


def _set_weighted_mean(field: jnp.ndarray, geometry: FciGeometry3D, target_mean: object) -> jnp.ndarray:
    values = jnp.asarray(field, dtype=jnp.float64)
    target = jnp.asarray(target_mean, dtype=jnp.float64)
    return values + (target - _weighted_mean(values, geometry))


def _project_dirichlet_values(field: jnp.ndarray, *, bc: FciBoundaryCondition) -> jnp.ndarray:
    values = jnp.asarray(field, dtype=jnp.float64)
    periodic_axes = _bc_periodic_axes(bc)
    axis_bcs = tuple(_bc_axis_spec(bc, axis, periodic_axes=periodic_axes) for axis in range(3))
    for axis, axis_bc in enumerate(axis_bcs):
        if periodic_axes[axis] or axis_bc is None or _bc_kind(axis_bc) != "dirichlet":
            continue
        values = _set_axis_plane(
            values,
            axis=axis,
            index=0,
            value=_broadcast_axis_boundary_value(axis_bc.lower_value, axis=axis, field_shape=values.shape),
        )
        values = _set_axis_plane(
            values,
            axis=axis,
            index=-1,
            value=_broadcast_axis_boundary_value(axis_bc.upper_value, axis=axis, field_shape=values.shape),
        )
    return values


def _impose_dirichlet_rhs(rhs: jnp.ndarray, *, bc: FciBoundaryCondition) -> jnp.ndarray:
    residual = jnp.asarray(rhs, dtype=jnp.float64)
    periodic_axes = _bc_periodic_axes(bc)
    axis_bcs = tuple(_bc_axis_spec(bc, axis, periodic_axes=periodic_axes) for axis in range(3))
    for axis, axis_bc in enumerate(axis_bcs):
        if periodic_axes[axis] or axis_bc is None or _bc_kind(axis_bc) != "dirichlet":
            continue
        residual = _set_axis_plane(
            residual,
            axis=axis,
            index=0,
            value=_broadcast_axis_boundary_value(axis_bc.lower_value, axis=axis, field_shape=residual.shape),
        )
        residual = _set_axis_plane(
            residual,
            axis=axis,
            index=-1,
            value=_broadcast_axis_boundary_value(axis_bc.upper_value, axis=axis, field_shape=residual.shape),
        )
    return residual


def _impose_dirichlet_rows(residual: jnp.ndarray, field: jnp.ndarray, *, bc: FciBoundaryCondition) -> jnp.ndarray:
    values = jnp.asarray(residual, dtype=jnp.float64)
    source = jnp.asarray(field, dtype=jnp.float64)
    periodic_axes = _bc_periodic_axes(bc)
    axis_bcs = tuple(_bc_axis_spec(bc, axis, periodic_axes=periodic_axes) for axis in range(3))
    for axis, axis_bc in enumerate(axis_bcs):
        if periodic_axes[axis] or axis_bc is None or _bc_kind(axis_bc) != "dirichlet":
            continue
        values = _set_axis_plane(values, axis=axis, index=0, value=source[_axis_index_nd(axis, 0, source.ndim)])
        values = _set_axis_plane(values, axis=axis, index=-1, value=source[_axis_index_nd(axis, -1, source.ndim)])
    return values


@partial(jax.jit, static_argnames=("tol", "maxiter", "precondition", "debug"))
def invert_perp_laplacian_cg(
    omega: jnp.ndarray,
    geometry: FciGeometry3D,
    bc: FciBoundaryCondition,
    phi_guess: jnp.ndarray | None = None,
    tol: float = 1.0e-6,
    maxiter: int = 50,
    precondition: bool = False,
    debug: bool = False,
) -> jnp.ndarray:
    """Invert the conservative perpendicular Laplacian with matrix-free CG.

    Boundary-condition contract:
    - ``bc`` must be an ``FciBoundaryCondition`` pytree.
    - ``bc.periodic_axes`` is a length-3 boolean tuple in ``(x, y, z)`` order.
    - Every non-periodic coordinate requires its strict coordinate entry:
      ``bc.x``, ``bc.y``, or ``bc.z``.
    - Each non-periodic coordinate entry is an ``FciAxisBoundaryCondition`` with
      ``kind`` equal to ``"dirichlet"`` or ``"neumann_flux"``.
    - Optional ``bc.target_mean_phi`` fixes the gauge for all-Neumann solves.
    - ``precondition`` is reserved for future solver acceleration and is ignored
      in this solver-only implementation.

    The solver enforces every non-periodic coordinate explicitly.
    """
    if lx is None:  # pragma: no cover - depends on optional local install
        raise ImportError("invert_perp_laplacian_cg requires the optional lineax package.")

    omega = jnp.asarray(omega, dtype=jnp.float64)
    if omega.ndim != 3:
        raise ValueError("omega must be a three-dimensional field")
    if omega.shape != geometry.shape:
        raise ValueError("omega.shape must match geometry.shape")

    periodic_axes = _bc_periodic_axes(bc)
    axis_bcs = tuple(_bc_axis_spec(bc, axis, periodic_axes=periodic_axes) for axis in range(3))
    has_dirichlet = any(axis_bc is not None and _bc_kind(axis_bc) == "dirichlet" for axis_bc in axis_bcs)

    if phi_guess is None:
        phi_guess_array = jnp.zeros_like(omega)
    else:
        phi_guess_array = jnp.asarray(phi_guess, dtype=jnp.float64)
        if phi_guess_array.shape != omega.shape:
            raise ValueError("phi_guess must match omega.shape")

    for axis, axis_bc in enumerate(axis_bcs):
        if periodic_axes[axis]:
            continue
        if axis_bc is None:
            raise ValueError(f"missing boundary condition entry for coordinate {_axis_name(axis)!r}")
        if omega.shape[axis] < 4:
            raise ValueError(
                f"invert_perp_laplacian_cg requires at least 4 points along coordinate {_axis_name(axis)!r}"
            )

    if has_dirichlet:
        rhs = _impose_dirichlet_rhs(-omega, bc=bc)
        u0 = _project_dirichlet_values(phi_guess_array, bc=bc)
    else:
        target_mean_phi = jnp.asarray(bc.target_mean_phi, dtype=jnp.float64)
        rhs = _remove_weighted_mean(-omega, geometry)
        u0 = _remove_weighted_mean(phi_guess_array, geometry)
        u0 = _set_weighted_mean(u0, geometry, target_mean_phi)

    structure = jax.ShapeDtypeStruct(omega.shape, omega.dtype)

    def apply_A(field: jnp.ndarray) -> jnp.ndarray:
        values = jnp.asarray(field, dtype=jnp.float64)
        if not has_dirichlet:
            values = _remove_weighted_mean(values, geometry)
        residual = -perp_laplacian_op(values, geometry, bc=bc)
        if has_dirichlet:
            return _impose_dirichlet_rows(residual, values, bc=bc)
        return _remove_weighted_mean(residual, geometry)

    operator = lx.FunctionLinearOperator(apply_A, structure)
    solver = lx.GMRES(
        rtol=float(tol),
        atol=0.0,
        restart=min(int(maxiter), 20),
        max_steps=int(maxiter),
    )
    solution = lx.linear_solve(operator, rhs, solver, options={"y0": u0}, throw=False)
    if solution.result != lx.RESULTS.successful:
        raise RuntimeError(f"invert_perp_laplacian_cg failed with status {solution.result!r}")

    result = jnp.asarray(solution.value, dtype=jnp.float64)
    if has_dirichlet:
        return _project_dirichlet_values(result, bc=bc)
    return _set_weighted_mean(_remove_weighted_mean(result, geometry), geometry, bc.target_mean_phi)


def _first_derivative_3d(
    values: jnp.ndarray,
    spacing: jnp.ndarray | float,
    *,
    axis: int,
    periodic: bool,
) -> jnp.ndarray:
    """Centered first derivative with periodic or second-order edge treatment."""

    h = jnp.asarray(spacing, dtype=jnp.float64)
    if h.ndim == 0:
        h = jnp.ones_like(values) * h
    centered = (jnp.roll(values, -1, axis=axis) - jnp.roll(values, 1, axis=axis)) / jnp.maximum(2.0 * h, 1.0e-30)
    if periodic:
        return centered

    first = _axis_index(axis, 0)
    second = _axis_index(axis, 1)
    third = _axis_index(axis, 2)
    last = _axis_index(axis, -1)
    penultimate = _axis_index(axis, -2)
    antepenultimate = _axis_index(axis, -3)
    forward = (-3.0 * values[first] + 4.0 * values[second] - values[third]) / jnp.maximum(2.0 * h[first], 1.0e-30)
    backward = (3.0 * values[last] - 4.0 * values[penultimate] + values[antepenultimate]) / jnp.maximum(2.0 * h[last], 1.0e-30)
    return centered.at[first].set(forward).at[last].set(backward)


def _face_average_3d(values: jnp.ndarray, *, axis: int, periodic: bool) -> jnp.ndarray:
    """Return values averaged onto the high face of each cell along one axis."""

    face = 0.5 * (values + jnp.roll(values, -1, axis=axis))
    if periodic:
        return face
    return face.at[_axis_index(axis, -1)].set(values[_axis_index(axis, -1)])


def _face_interpolate_3d_order4(values: jnp.ndarray, *, axis: int, periodic: bool) -> jnp.ndarray:
    """Fourth-order interpolation from nodes to high faces."""

    values = jnp.asarray(values, dtype=jnp.float64)
    if periodic:
        return (-jnp.roll(values, 1, axis=axis) + 9.0 * values + 9.0 * jnp.roll(values, -1, axis=axis) - jnp.roll(values, -2, axis=axis)) / 16.0

    if values.shape[axis] < 4:
        raise ValueError("Fourth-order face interpolation requires at least 4 points along the selected axis")
    ndim = values.ndim
    face = jnp.zeros_like(values)
    face = face.at[_axis_index_nd(axis, 0, ndim)].set(
        (
            5.0 * values[_axis_index_nd(axis, 0, ndim)]
            + 15.0 * values[_axis_index_nd(axis, 1, ndim)]
            - 5.0 * values[_axis_index_nd(axis, 2, ndim)]
            + values[_axis_index_nd(axis, 3, ndim)]
        )
        / 16.0
    )
    face = face.at[_axis_slice_nd(axis, 1, -2, ndim)].set(
        (
            -values[_axis_slice_nd(axis, None, -3, ndim)]
            + 9.0 * values[_axis_slice_nd(axis, 1, -2, ndim)]
            + 9.0 * values[_axis_slice_nd(axis, 2, -1, ndim)]
            - values[_axis_slice_nd(axis, 3, None, ndim)]
        )
        / 16.0
    )
    face = face.at[_axis_index_nd(axis, -2, ndim)].set(
        (
            values[_axis_index_nd(axis, -4, ndim)]
            - 5.0 * values[_axis_index_nd(axis, -3, ndim)]
            + 15.0 * values[_axis_index_nd(axis, -2, ndim)]
            + 5.0 * values[_axis_index_nd(axis, -1, ndim)]
        )
        / 16.0
    )
    return face.at[_axis_index_nd(axis, -1, ndim)].set(values[_axis_index_nd(axis, -1, ndim)])


def _face_derivative_3d_order4(
    values: jnp.ndarray,
    spacing: jnp.ndarray | float,
    *,
    axis: int,
    periodic: bool,
) -> jnp.ndarray:
    """Fourth-order first derivative from nodes to high faces."""

    values = jnp.asarray(values, dtype=jnp.float64)
    h = jnp.asarray(spacing, dtype=jnp.float64)
    if h.ndim == 0:
        h = jnp.ones_like(values) * h
    h = jnp.maximum(h, 1.0e-30)
    if periodic:
        return (jnp.roll(values, 1, axis=axis) - 27.0 * values + 27.0 * jnp.roll(values, -1, axis=axis) - jnp.roll(values, -2, axis=axis)) / (24.0 * h)

    if values.shape[axis] < 4:
        raise ValueError("Fourth-order face derivative requires at least 4 points along the selected axis")
    ndim = values.ndim
    derivative = jnp.zeros_like(values)
    derivative = derivative.at[_axis_index_nd(axis, 0, ndim)].set(
        (
            -23.0 * values[_axis_index_nd(axis, 0, ndim)]
            + 21.0 * values[_axis_index_nd(axis, 1, ndim)]
            + 3.0 * values[_axis_index_nd(axis, 2, ndim)]
            - values[_axis_index_nd(axis, 3, ndim)]
        )
        / (24.0 * h[_axis_index_nd(axis, 0, ndim)])
    )
    derivative = derivative.at[_axis_slice_nd(axis, 1, -2, ndim)].set(
        (
            values[_axis_slice_nd(axis, None, -3, ndim)]
            - 27.0 * values[_axis_slice_nd(axis, 1, -2, ndim)]
            + 27.0 * values[_axis_slice_nd(axis, 2, -1, ndim)]
            - values[_axis_slice_nd(axis, 3, None, ndim)]
        )
        / (24.0 * h[_axis_slice_nd(axis, 1, -2, ndim)])
    )
    return derivative.at[_axis_index_nd(axis, -2, ndim)].set(
        (
            values[_axis_index_nd(axis, -4, ndim)]
            - 3.0 * values[_axis_index_nd(axis, -3, ndim)]
            - 21.0 * values[_axis_index_nd(axis, -2, ndim)]
            + 23.0 * values[_axis_index_nd(axis, -1, ndim)]
        )
        / (24.0 * h[_axis_index_nd(axis, -2, ndim)])
    )


def _face_forward_difference_3d(
    values: jnp.ndarray,
    spacing: jnp.ndarray | float,
    *,
    axis: int,
    periodic: bool,
) -> jnp.ndarray:
    """Return a forward difference interpreted on the high face of each cell."""

    h = jnp.asarray(spacing, dtype=jnp.float64)
    if h.ndim == 0:
        h = jnp.ones_like(values) * h
    face_h = _face_average_3d(h, axis=axis, periodic=periodic)
    difference = (jnp.roll(values, -1, axis=axis) - values) / jnp.maximum(face_h, 1.0e-30)
    if periodic:
        return difference
    last = _axis_index(axis, -1)
    penultimate = _axis_index(axis, -2)
    antepenultimate = _axis_index(axis, -3)
    backward = (
        3.0 * values[last] - 4.0 * values[penultimate] + values[antepenultimate]
    ) / jnp.maximum(2.0 * face_h[last], 1.0e-30)
    return difference.at[last].set(backward)


def _axis_index(axis: int, index: int) -> tuple[object, object, object]:
    slices: list[object] = [slice(None), slice(None), slice(None)]
    slices[axis] = index
    return tuple(slices)
