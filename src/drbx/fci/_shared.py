"""Definitions that were byte-identical in both FCI code families.

``drbx.geometry.fci_geometry`` / ``drbx.native.fci_*`` and
``drbx.fci_braginskii.{geometry,native}`` import these from here instead of
keeping two copies (FCI unification, issue #41, PR 1).  Only definitions whose
module-level dependencies are also identical live here, so both families
resolve exactly the same objects as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # annotation-only names
    from ..geometry.fci_geometry import CellCenteredGrid3D, Grid1D, LocalDomain3D
    from ..native.fci_boundaries import LocalStencil1D


# Dependency kinds are shared by local and remote FCI dependency metadata.
FCI_DEP_INVALID = 0
FCI_DEP_FIELD_INTERIOR = 1
FCI_DEP_PHYSICAL_BOUNDARY = 2
FCI_DEP_CUT_WALL = 3

NeighborIndex3D = tuple[int, int, int]
OptionalNeighborIndex3D = NeighborIndex3D | None


# Meaning of a global lower/upper side in a ShardSpec3D. These values are
# metadata for later halo stages; HaloExchange3D only owns regular-neighbor
# and SIDE_SIMPLE_PERIODIC data.
SIDE_PHYSICAL = 1
SIDE_SIMPLE_PERIODIC = 2
SIDE_AXIS_REGULAR = 3
SIDE_TOPOLOGY_MAPPED = 4
SIDE_UNUSED = 5


_VALID_SIDE_KINDS = frozenset(
    {
        SIDE_PHYSICAL,
        SIDE_SIMPLE_PERIODIC,
        SIDE_AXIS_REGULAR,
        SIDE_TOPOLOGY_MAPPED,
        SIDE_UNUSED,
    }
)


def _physical_domain_valid_mask(
    grid: CellCenteredGrid3D,
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    *,
    periodic_axes: tuple[bool, bool, bool],
) -> jnp.ndarray:
    valid_x = jnp.isfinite(x) if periodic_axes[0] else (jnp.isfinite(x) & (x >= grid.x.faces[0]) & (x <= grid.x.faces[-1]))
    valid_y = jnp.isfinite(y) if periodic_axes[1] else (jnp.isfinite(y) & (y >= grid.y.faces[0]) & (y <= grid.y.faces[-1]))
    valid_z = jnp.isfinite(z) if periodic_axes[2] else (jnp.isfinite(z) & (z >= grid.z.faces[0]) & (z <= grid.z.faces[-1]))
    return valid_x & valid_y & valid_z


def _extend_axis_with_ghost_cells(grid_axis: Grid1D) -> jnp.ndarray:
    centers = grid_axis.centers
    faces = grid_axis.faces
    lower = 2.0 * faces[0] - centers[0]
    upper = 2.0 * faces[-1] - centers[-1]
    return jnp.concatenate([
        jnp.asarray([lower], dtype=jnp.float64),
        centers,
        jnp.asarray([upper], dtype=jnp.float64),
    ])


def _extend_values_x(values: jnp.ndarray, grid_axis: Grid1D) -> jnp.ndarray:
    centers = grid_axis.centers
    faces = grid_axis.faces
    xg_lower = 2.0 * faces[0] - centers[0]
    xg_upper = 2.0 * faces[-1] - centers[-1]

    if values.shape[0] == 1:
        lower = values[0:1, :, :]
        upper = values[-1:, :, :]
    else:
        lower_slope = (values[1, :, :] - values[0, :, :]) / (centers[1] - centers[0])
        upper_slope = (values[-1, :, :] - values[-2, :, :]) / (centers[-1] - centers[-2])

        lower = (values[0, :, :] + (xg_lower - centers[0]) * lower_slope)[None, :, :]
        upper = (values[-1, :, :] + (xg_upper - centers[-1]) * upper_slope)[None, :, :]

    return jnp.concatenate([lower, values, upper], axis=0)


def _extend_values_y(values: jnp.ndarray, grid_axis: Grid1D) -> jnp.ndarray:
    centers = grid_axis.centers
    faces = grid_axis.faces
    yg_lower = 2.0 * faces[0] - centers[0]
    yg_upper = 2.0 * faces[-1] - centers[-1]

    if values.shape[1] == 1:
        lower = values[:, 0:1, :]
        upper = values[:, -1:, :]
    else:
        lower_slope = (values[:, 1, :] - values[:, 0, :]) / (centers[1] - centers[0])
        upper_slope = (values[:, -1, :] - values[:, -2, :]) / (centers[-1] - centers[-2])

        lower = (values[:, 0, :] + (yg_lower - centers[0]) * lower_slope)[:, None, :]
        upper = (values[:, -1, :] + (yg_upper - centers[-1]) * upper_slope)[:, None, :]

    return jnp.concatenate([lower, values, upper], axis=1)


def _extend_values_z(values: jnp.ndarray, grid_axis: Grid1D) -> jnp.ndarray:
    centers = grid_axis.centers
    faces = grid_axis.faces
    zg_lower = 2.0 * faces[0] - centers[0]
    zg_upper = 2.0 * faces[-1] - centers[-1]

    if values.shape[2] == 1:
        lower = values[:, :, 0:1]
        upper = values[:, :, -1:]
    else:
        lower_slope = (values[:, :, 1] - values[:, :, 0]) / (centers[1] - centers[0])
        upper_slope = (values[:, :, -1] - values[:, :, -2]) / (centers[-1] - centers[-2])

        lower = (values[:, :, 0] + (zg_lower - centers[0]) * lower_slope)[:, :, None]
        upper = (values[:, :, -1] + (zg_upper - centers[-1]) * upper_slope)[:, :, None]

    return jnp.concatenate([lower, values, upper], axis=2)


_pytree_base = jax.tree_util.register_pytree_node_class


def _take_stencil_finite_difference(stencil: LocalStencil1D) -> jnp.ndarray:
    """Apply a reconstructed 1D derivative stencil.

    The stencil arrays may represent either global/reference cells or local
    owned cells. The output has the same shape as the stencil.
    """

    if stencil.center.ndim != 3:
        raise ValueError(
            f"stencil center must be 3D, got shape {stencil.center.shape}"
        )

    minus = jnp.asarray(stencil.minus, dtype=jnp.float64)
    center = jnp.asarray(stencil.center, dtype=jnp.float64)
    plus = jnp.asarray(stencil.plus, dtype=jnp.float64)

    c_minus = jnp.asarray(stencil.derivative_minus_weight, dtype=jnp.float64)
    c_center = jnp.asarray(stencil.derivative_center_weight, dtype=jnp.float64)
    c_plus = jnp.asarray(stencil.derivative_plus_weight, dtype=jnp.float64)

    return c_minus * minus + c_center * center + c_plus * plus


def _axis_slice_nd(axis: int, start: int | None, stop: int | None, ndim: int) -> tuple[object, ...]:
    slices: list[object] = [slice(None)] * ndim
    slices[axis] = slice(start, stop)
    return tuple(slices)


BC_NONE = 0
BC_DIRICHLET = 1
BC_NEUMANN = 2
BC_NORMALFLUX = 3
BC_NOFLUX = 4


def _validate_halo_spatial_prefix(
    field_halo: jnp.ndarray,
    domain: LocalDomain3D,
    *,
    name: str = "field_halo",
) -> jnp.ndarray:
    """Validate the spatial part of a scalar or trailing-axis halo field."""

    field_halo = jnp.asarray(field_halo)
    expected_shape = tuple(int(value) for value in domain.layout.cell_halo_shape)
    if field_halo.ndim < 3 or tuple(field_halo.shape[:3]) != expected_shape:
        raise ValueError(
            f"{name} leading shape must match domain.layout.cell_halo_shape; "
            f"got {field_halo.shape}, expected prefix {expected_shape}"
        )
    return field_halo


_MESH_AXIS_NAMES = ("x", "y", "z")


_METRIC_NAMES = (
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
)


_CELL_FIELD_NAMES = _METRIC_NAMES + (
    "Bmag",
    "B_contra_x",
    "B_contra_y",
    "B_contra_z",
    "dx",
    "dy",
    "dz",
)
