from __future__ import annotations

from typing import Literal, Sequence

import jax
import jax.numpy as jnp

from .fci_geometry import HaloLayout3D, LocalDomain3D

# Definitions identical to (and closure-equivalent with) the shared module;
# re-exported here so existing import paths keep working.
from ...native.fci_helpers import (  # noqa: F401
    _as_float64_array,
    _as_face_flux_array,
    _as_int_face_array,
    _as_bool_face_array,
    _as_coordinate_derivative_weight_array,
    _as_wall_face_array,
    _normalize_axis_flags,
    _axis_regular_lower_x_face,
    _axis_name,
    _validate_axis,
    _local_cell_halo_array,
    _local_owned_cell_array,
    _local_face_halo_array,
    _local_control_face_array,
    _local_coordinate_face_tuple,
    local_side_plane_shape,
    _as_local_side_plane_array,
    _as_local_wall_array,
    _as_local_wall_int_array,
    _as_local_wall_bool_array,
    _as_local_wall_stencil_index_array,
    _as_local_wall_stencil_weight_array,
    local_physical_side_active,
    _local_side_mask,
)


def _as_coordinate_face_tuple(
    value: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    geometry,
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

