from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from ..geometry.fci_geometry import HaloLayout3D, LocalDomain3D


def _as_float64_array(value: jnp.ndarray, name: str) -> jnp.ndarray:
    """Normalize a generic 3D array to float64."""

    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3D, got {array.shape}")
    return array


def _as_face_flux_array(value: jnp.ndarray, name: str) -> jnp.ndarray:
    """Normalize a face-aligned 3D array without deciding global vs local layout."""

    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3D, got {array.shape}")
    return array


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


def local_side_plane_shape(layout: HaloLayout3D, axis: int) -> tuple[int, int]:
    """Return the owned-cell side-plane shape for a local boundary payload."""

    axis = _validate_axis(axis)
    nx, ny, nz = layout.owned_shape
    if axis == 0:
        return ny, nz
    if axis == 1:
        return nx, nz
    return nx, ny


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
