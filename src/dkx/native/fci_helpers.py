from __future__ import annotations

from typing import Literal, Sequence

import jax
import jax.numpy as jnp

from ..geometry.fci_geometry import FciGeometry3D, HaloLayout3D, LocalDomain3D


def _as_float64_array(value: jnp.ndarray, name: str) -> jnp.ndarray:
    """Normalize a generic 3D array to float64."""

    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3D, got {array.shape}")
    return array


def _as_optional_boundary_plane(value: jnp.ndarray | float | None, name: str) -> jnp.ndarray | None:
    """Normalize a boundary-plane payload for the global/reference path."""

    if value is None:
        return None
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim not in (0, 2):
        raise ValueError(f"{name} must be scalar or 2D, got {array.shape}")
    return array


def _as_face_flux_array(value: jnp.ndarray, name: str) -> jnp.ndarray:
    """Normalize a face-aligned 3D array without deciding global vs local layout."""

    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3D, got {array.shape}")
    return array


def _as_int_face_array(value: jnp.ndarray, name: str) -> jnp.ndarray:
    """Normalize a 3D integer face array."""

    array = jnp.asarray(value, dtype=jnp.int32)
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3D, got {array.shape}")
    return array


def _as_bool_face_array(value: jnp.ndarray, name: str) -> jnp.ndarray:
    """Normalize a 3D boolean face array."""

    array = jnp.asarray(value, dtype=bool)
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3D, got {array.shape}")
    return array


def _as_int_stencil_array(value: jnp.ndarray, name: str) -> jnp.ndarray:
    """Normalize a 2D integer stencil array."""

    array = jnp.asarray(value, dtype=jnp.int32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {array.shape}")
    return array


def _as_weight_stencil_array(value: jnp.ndarray, name: str) -> jnp.ndarray:
    """Normalize a 2D floating-point stencil weight array."""

    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {array.shape}")
    return array


def _as_coordinate_derivative_weight_array(value: jnp.ndarray, name: str) -> jnp.ndarray:
    """Normalize the fixed coordinate-derivative weight tensor."""

    array = jnp.asarray(value, dtype=jnp.float64)
    if array.shape != (3, 2, 4):
        raise ValueError(f"{name} must have shape (3, 2, 4), got {array.shape}")
    return array


def _as_wall_face_array(value: jnp.ndarray | float, n_faces: int, name: str) -> jnp.ndarray:
    """Normalize a per-wall-face scalar payload."""

    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim == 0:
        return jnp.broadcast_to(array, (int(n_faces),))
    if array.ndim != 1 or array.shape != (int(n_faces),):
        raise ValueError(f"{name} must be scalar or shape {(int(n_faces),)}, got {array.shape}")
    return array


def _as_coordinate_face_tuple(
    value: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    geometry: FciGeometry3D,
    name: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Validate global face tuples against a global active-cell geometry."""

    if len(value) != 3:
        raise ValueError(f"{name} must be a tuple of three face arrays")
    x_shape = (geometry.shape[0] + 1, geometry.shape[1], geometry.shape[2])
    y_shape = (geometry.shape[0], geometry.shape[1] + 1, geometry.shape[2])
    z_shape = (geometry.shape[0], geometry.shape[1], geometry.shape[2] + 1)
    x_value = jnp.asarray(value[0], dtype=jnp.float64)
    y_value = jnp.asarray(value[1], dtype=jnp.float64)
    z_value = jnp.asarray(value[2], dtype=jnp.float64)
    if x_value.shape != x_shape or y_value.shape != y_shape or z_value.shape != z_shape:
        raise ValueError(
            f"{name} face arrays must have shapes x={x_shape}, y={y_shape}, z={z_shape}; "
            f"got x={x_value.shape}, y={y_value.shape}, z={z_value.shape}"
        )
    return x_value, y_value, z_value


def _normalize_axis_flags(
    value: Sequence[bool],
    name: str,
) -> tuple[bool, bool, bool]:
    """Normalize a 3-axis boolean flag tuple."""

    if len(value) != 3:
        raise ValueError(f"{name} must have length 3, got {value}")
    return tuple(bool(axis) for axis in value)


def _axis_regular_lower_x_face(values: jnp.ndarray) -> jnp.ndarray:
    """Global/reference helper for the lower-x axis-regular face estimate."""

    if values.ndim != 3:
        raise ValueError(f"axis-regular lower x reconstruction requires a 3D field, got {values.shape}")
    if values.shape[1] % 2:
        raise ValueError("axis-regular lower-rho mapping requires an even poloidal grid")
    half_turn = values.shape[1] // 2
    return 0.5 * (values[0] + jnp.roll(values[0], shift=-half_turn, axis=0))


def _axis_name(axis: int) -> str:
    axis = int(axis)
    if axis == 0:
        return "x"
    if axis == 1:
        return "y"
    if axis == 2:
        return "z"
    raise ValueError(f"axis must be 0, 1, or 2, got {axis}")


def _validate_axis(axis: int) -> int:
    axis = int(axis)
    if axis < 0 or axis > 2:
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    return axis


def _local_cell_halo_array(value: jnp.ndarray, layout: HaloLayout3D, name: str) -> jnp.ndarray:
    """Validate a local cell-centered halo array."""

    array = jnp.asarray(value, dtype=jnp.float64)
    if array.shape != layout.cell_halo_shape:
        raise ValueError(
            f"{name} must have shape {layout.cell_halo_shape}, got {array.shape}"
        )
    return array


def _local_owned_cell_array(value: jnp.ndarray, layout: HaloLayout3D, name: str) -> jnp.ndarray:
    """Validate a local owned-cell array."""

    array = jnp.asarray(value, dtype=jnp.float64)
    if array.shape != layout.owned_shape:
        raise ValueError(f"{name} must have shape {layout.owned_shape}, got {array.shape}")
    return array


def _local_face_halo_array(value: jnp.ndarray, layout: HaloLayout3D, axis: int, name: str) -> jnp.ndarray:
    """Validate a halo-padded local face array."""

    axis = _validate_axis(axis)
    array = jnp.asarray(value, dtype=jnp.float64)
    expected_shape = layout.face_halo_shape(axis)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    return array


def _local_control_face_array(value: jnp.ndarray, layout: HaloLayout3D, axis: int, name: str) -> jnp.ndarray:
    """Validate a local owned/control-face array."""

    axis = _validate_axis(axis)
    array = jnp.asarray(value, dtype=jnp.float64)
    expected_shape = layout.face_control_shape(axis)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    return array


def _local_coordinate_face_tuple(
    value: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    layout: HaloLayout3D,
    name: str,
    *,
    region: Literal["control", "halo_face"] = "control",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Validate a local coordinate-face tuple against a halo layout."""

    if len(value) != 3:
        raise ValueError(f"{name} must be a tuple of three face arrays")
    validators = (
        _local_control_face_array if region == "control" else _local_face_halo_array
    )
    x_value = validators(value[0], layout, 0, f"{name}.x")
    y_value = validators(value[1], layout, 1, f"{name}.y")
    z_value = validators(value[2], layout, 2, f"{name}.z")
    return x_value, y_value, z_value


def local_side_plane_shape(layout: HaloLayout3D, axis: int) -> tuple[int, int]:
    """Return the owned-cell side-plane shape for a local boundary payload."""

    axis = _validate_axis(axis)
    nx, ny, nz = layout.owned_shape
    if axis == 0:
        return ny, nz
    if axis == 1:
        return nx, nz
    return nx, ny


def _as_local_side_plane_array(
    value: jnp.ndarray | float,
    layout: HaloLayout3D,
    axis: int,
    name: str,
    *,
    dtype=jnp.float64,
) -> jnp.ndarray:
    """Validate a local side-plane payload used by ghost-fill boundary helpers."""

    axis = _validate_axis(axis)
    array = jnp.asarray(value, dtype=dtype)
    expected_shape = local_side_plane_shape(layout, axis)
    if array.ndim == 0:
        return jnp.broadcast_to(array, expected_shape)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    return array


def _as_local_wall_array(
    value: jnp.ndarray,
    max_wall_faces: int,
    trailing_shape: tuple[int, ...],
    name: str,
    *,
    dtype=jnp.float64,
) -> jnp.ndarray:
    """Validate a padded local wall payload with an arbitrary trailing shape."""

    array = jnp.asarray(value, dtype=dtype)
    expected_shape = (int(max_wall_faces),) + tuple(int(v) for v in trailing_shape)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    return array


def _as_local_wall_int_array(
    value: jnp.ndarray,
    max_wall_faces: int,
    name: str,
) -> jnp.ndarray:
    """Validate a padded local wall integer vector."""

    return _as_local_wall_array(value, max_wall_faces, (), name, dtype=jnp.int32)


def _as_local_wall_bool_array(
    value: jnp.ndarray,
    max_wall_faces: int,
    name: str,
) -> jnp.ndarray:
    """Validate a padded local wall boolean vector."""

    return _as_local_wall_array(value, max_wall_faces, (), name, dtype=bool)


def _as_local_wall_stencil_index_array(
    value: jnp.ndarray,
    max_wall_faces: int,
    stencil_width: int,
    name: str,
) -> jnp.ndarray:
    """Validate local wall stencil indices against the halo field."""

    array = jnp.asarray(value, dtype=jnp.int32)
    expected_shape = (int(max_wall_faces), int(stencil_width))
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    return array


def _as_local_wall_stencil_weight_array(
    value: jnp.ndarray,
    max_wall_faces: int,
    stencil_width: int,
    name: str,
) -> jnp.ndarray:
    """Validate local wall stencil weights."""

    array = jnp.asarray(value, dtype=jnp.float64)
    expected_shape = (int(max_wall_faces), int(stencil_width))
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    return array


def local_physical_side_active(
    domain: LocalDomain3D,
    axis: int,
    side: Literal["lower", "upper"],
) -> bool | jnp.ndarray:
    """Return runtime physical-side ownership for a local boundary payload.

    The result is a Python boolean for an undecomposed axis and a traced JAX
    boolean inside an SPMD context with a configured mesh axis. It is true
    only on the runtime global side whose side kind is ``SIDE_PHYSICAL``.
    """

    axis = _validate_axis(axis)
    if side not in ("lower", "upper"):
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")
    if side == "lower":
        return domain.runtime_has_physical_lower(axis)
    return domain.runtime_has_physical_upper(axis)


def _local_side_mask(
    domain: LocalDomain3D,
    layout: HaloLayout3D,
    axis: int,
    side: Literal["lower", "upper"],
) -> jnp.ndarray:
    """Build a boolean mask for the physical side-plane payload."""

    active = local_physical_side_active(domain, axis, side)
    shape = local_side_plane_shape(layout, axis)
    return jnp.broadcast_to(jnp.asarray(active, dtype=bool), shape)
