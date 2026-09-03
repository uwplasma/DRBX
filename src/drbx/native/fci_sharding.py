"""Multi-device sharded execution helpers for the FCI stack.

This module promotes the proven ``shard_map`` harness patterns from
``tests/test_fci_operators_domain_decomp.py`` into a small library API and
extends them from operator-level tests to a full two-field RHS + RK4 step:

- :func:`make_shard_mesh` builds the three-axis execution mesh.
- :func:`build_local_fci_geometries` converts a global :class:`FciGeometry3D`
  into the per-shard representation the ``shard_map`` kernel consumes: a
  ``LocalDomain3D`` plus a cell-shaped packed bundle containing cell geometry
  and the exact lower/upper face samples owned by each cell.  The bundle is
  partitioned with ``PartitionSpec("x", "y", "z")``.
- :func:`assemble_local_fci_geometry` runs inside ``shard_map`` and assembles
  a :class:`LocalFciGeometry3D` from one shard's owned geometry block using
  halo exchange, periodic topology filling, and the runtime shard index.
- :func:`make_sharded_2field_step` returns a jitted RK4 step for the reduced
  two-field model where every stage prepares state halos (exchange plus
  periodic topology fill) before evaluating the RHS on local geometry.

The reduced two-field local stencil path closes physical sides with one-sided
derivative stencils and consumes no face-BC payload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import replace
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..geometry import (
    FciGeometry3D,
    HaloLayout3D,
    LocalBFieldGeometry,
    LocalCellCenteredGrid3D,
    LocalCellVolumeGeometry3D,
    LocalDomain3D,
    LocalFaceBFieldGeometry,
    LocalFaceMetricGeometry,
    LocalFciDirectionMap,
    LocalFciGeometry3D,
    LocalFciLocalDependencyTable,
    LocalFciRemoteDependencyTable,
    LocalFciMaps3D,
    FCI_DEP_FIELD_INTERIOR,
    FCI_DEP_INVALID,
    FCI_DEP_PHYSICAL_BOUNDARY,
    LocalGrid1D,
    LocalMetricGeometry,
    LocalRegularFaceGeometry3D,
    LocalSpacing3D,
    SIDE_AXIS_REGULAR,
    SIDE_PHYSICAL,
    SIDE_SIMPLE_PERIODIC,
    ShardSpec3D,
    StencilBuilderContext,
    build_local_curvature_coefficients,
    build_local_direct_stencil_one_sided_physical_from_halo,
)
from .fci_2_field_rhs import (
    Fci2FieldRhsParameters,
    Fci2FieldState,
    compute_local_2field_rhs,
)
from .fci_halo import (
    HaloExchange3D,
    LocalPeriodicTopologyRule3D,
    PolarAxisRegularVectorRule3D,
    TopologyHaloFiller3D,
)
from .fci_model import inject_owned_field_to_halo, inject_owned_vector_field_to_halo
from .fci_time_integrator import Rk4Stepper


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
_FACE_FIELD_NAMES = _METRIC_NAMES + (
    "Bmag",
    "B_contra_x",
    "B_contra_y",
    "B_contra_z",
)
_FACE_PACKED_FIELD_NAMES = tuple(
    f"{axis_name}_{side}_{field_name}"
    for axis_name in ("x", "y", "z")
    for side in ("lower", "upper")
    for field_name in _FACE_FIELD_NAMES
)
_SHARDED_FIELD_NAMES = _CELL_FIELD_NAMES + _FACE_PACKED_FIELD_NAMES
_MAP_FIELD_NAMES = (
    "forward_x",
    "forward_y",
    "backward_x",
    "backward_y",
    "forward_endpoint_x",
    "forward_endpoint_y",
    "forward_endpoint_z",
    "backward_endpoint_x",
    "backward_endpoint_y",
    "backward_endpoint_z",
    "forward_endpoint_b_contra_x",
    "forward_endpoint_b_contra_y",
    "forward_endpoint_b_contra_z",
    "forward_endpoint_bmag",
    "backward_endpoint_b_contra_x",
    "backward_endpoint_b_contra_y",
    "backward_endpoint_b_contra_z",
    "backward_endpoint_bmag",
    "forward_length",
    "backward_length",
    "forward_boundary",
    "backward_boundary",
)

# Parity under the signed-radial extension
# q(-u, theta, eta) = parity * q(u, theta + pi, eta).  The signed Jacobian
# must be odd so midpoint lifting places J=0 exactly on the polar axis.
_AXIS_REGULAR_CELL_FIELD_PARITY = (
    -1.0,  # J
    +1.0, +1.0, +1.0, -1.0, -1.0, +1.0,  # g^ij
    +1.0, +1.0, +1.0, -1.0, -1.0, +1.0,  # g_ij
    +1.0,  # |B|
    -1.0, +1.0, +1.0,  # B^i
    +1.0, +1.0, +1.0,  # logical cell widths
)


def make_shard_mesh(shard_counts: tuple[int, int, int]) -> Mesh:
    """Build the ``("x", "y", "z")`` execution mesh for the requested layout."""

    shard_counts = tuple(int(value) for value in shard_counts)
    if len(shard_counts) != 3 or any(value <= 0 for value in shard_counts):
        raise ValueError(f"shard_counts must contain three positive integers, got {shard_counts}")

    ndevices = math.prod(shard_counts)
    devices = np.asarray(jax.devices()[:ndevices], dtype=object)
    if devices.size < ndevices:
        raise RuntimeError(
            f"shard_counts={shard_counts} requires {ndevices} devices, "
            f"but only {devices.size} are available"
        )
    return Mesh(devices.reshape(shard_counts), _MESH_AXIS_NAMES)


def _assert_shape_divisible_by_shards(
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
) -> None:
    """Require equal-sized local blocks on every mesh axis."""

    for axis, (size, count) in enumerate(zip(shape, shard_counts)):
        if int(size) % int(count):
            raise ValueError(
                f"global shape axis {axis} with size {size} is not divisible by "
                f"shard count {count}; shape={shape}, shard_counts={shard_counts}"
            )


@dataclass(frozen=True)
class _UniformAxisMeta:
    """Static uniform-axis coordinate metadata for one logical axis."""

    center0: float
    face0: float
    spacing: float


def _uniform_axis_meta(grid_axis, *, axis: int) -> _UniformAxisMeta:
    centers = np.asarray(grid_axis.centers, dtype=np.float64)
    faces = np.asarray(grid_axis.faces, dtype=np.float64)
    if centers.size < 2:
        raise ValueError(f"sharded axis {axis} requires at least two cells, got {centers.size}")
    spacing = float((centers[-1] - centers[0]) / (centers.size - 1))
    deltas = np.diff(centers)
    tolerance = 1.0e-12 * max(1.0, abs(spacing))
    if np.max(np.abs(deltas - spacing)) > tolerance:
        raise ValueError(
            "build_local_fci_geometries requires uniformly spaced grid axes; "
            f"axis {axis} center spacings deviate by "
            f"{float(np.max(np.abs(deltas - spacing))):.3e}"
        )
    return _UniformAxisMeta(
        center0=float(centers[0]),
        face0=float(faces[0]),
        spacing=spacing,
    )


@dataclass(frozen=True)
class ShardedFciGeometry3D:
    """Per-shard geometry description consumed by the ``shard_map`` kernel.

    ``cell_fields`` is a global
    ``(nx, ny, nz, len(_SHARDED_FIELD_NAMES))`` array.  Its leading channels
    are cell-centered geometry.  The remaining channels store, for each face
    family, the exact lower and upper face samples adjacent to every cell.
    This redundant sided-face packing keeps the first three dimensions
    cell-shaped and therefore equally shardable while preserving shared-face
    values evaluated by the global metric builder.  Local cell halos use the
    normal exchange/topology pipeline; exact owned faces are unpacked without
    interpolating singular tensor components.
    """

    domain: LocalDomain3D
    cell_fields: jnp.ndarray
    axis_meta: tuple[_UniformAxisMeta, _UniformAxisMeta, _UniformAxisMeta]
    map_fields: jnp.ndarray | None = None
    maps_valid: bool = False

    @property
    def global_shape(self) -> tuple[int, int, int]:
        return self.domain.shard_spec.global_shape

    @property
    def shard_counts(self) -> tuple[int, int, int]:
        return self.domain.shard_spec.shard_counts

    @property
    def halo_width(self) -> int:
        return self.domain.layout.halo_width


def build_local_fci_geometries(
    geometry: FciGeometry3D,
    shard_counts: tuple[int, int, int],
    *,
    halo_width: int = 1,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
) -> ShardedFciGeometry3D:
    """Convert a global FCI geometry into the shard-local kernel inputs."""

    if not isinstance(geometry, FciGeometry3D):
        raise TypeError(f"geometry must be an FciGeometry3D instance, got {type(geometry).__name__}")
    shard_counts = tuple(int(value) for value in shard_counts)
    global_shape = geometry.shape
    periodic_axes = tuple(bool(value) for value in periodic_axes)
    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    if len(periodic_axes) != 3 or len(axis_regular_axes) != 3:
        raise ValueError("periodic_axes and axis_regular_axes must have length 3")
    if any(
        periodic and axis_regular
        for periodic, axis_regular in zip(periodic_axes, axis_regular_axes)
    ):
        raise ValueError(
            "periodic_axes and axis_regular_axes cannot overlap on one axis; "
            f"got periodic_axes={periodic_axes}, "
            f"axis_regular_axes={axis_regular_axes}"
        )
    if axis_regular_axes[1] or axis_regular_axes[2]:
        raise NotImplementedError(
            "build_local_fci_geometries currently supports axis regularity "
            "only on the lower x face"
        )
    if axis_regular_axes[0]:
        if not periodic_axes[1]:
            raise ValueError("lower-x axis regularity requires periodic theta (axis 1)")
        if int(global_shape[1]) % 2:
            raise ValueError(
                "lower-x axis regularity requires an even global theta count"
            )
    _assert_shape_divisible_by_shards(global_shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count) for size, count in zip(global_shape, shard_counts)
    )
    halo_width = int(halo_width)
    for axis, (extent, count) in enumerate(zip(owned_shape, shard_counts)):
        if count > 1 and halo_width > extent:
            raise ValueError(
                "halo exchange requires halo_width no larger than the owned "
                f"extent on decomposed axis {axis}; got halo_width={halo_width}, "
                f"owned extent={extent}"
            )

    layout = HaloLayout3D(owned_shape, halo_width)
    side_kind_lower = tuple(
        SIDE_AXIS_REGULAR
        if axis_regular
        else SIDE_SIMPLE_PERIODIC
        if periodic
        else SIDE_PHYSICAL
        for periodic, axis_regular in zip(periodic_axes, axis_regular_axes)
    )
    side_kind_upper = tuple(
        SIDE_SIMPLE_PERIODIC if periodic else SIDE_PHYSICAL for periodic in periodic_axes
    )
    spec = ShardSpec3D(
        global_shape=global_shape,
        # Local-array coordinates; runtime boundary ownership is decided by
        # LocalDomain3D.runtime_* predicates inside shard_map.
        owned_start=(0, 0, 0),
        owned_stop=owned_shape,
        shard_index=(0, 0, 0),
        shard_counts=shard_counts,
        periodic_axes=periodic_axes,
        axis_regular_axes=axis_regular_axes,
        halo_width=halo_width,
        side_kind_lower=side_kind_lower,
        side_kind_upper=side_kind_upper,
    )
    domain = LocalDomain3D(shard_spec=spec, layout=layout, mesh_axis_names=_MESH_AXIS_NAMES)

    metric = geometry.cell_metric
    bfield = geometry.cell_bfield
    spacing = geometry.spacing
    channels = [getattr(metric, name) for name in _METRIC_NAMES]
    channels.append(bfield.Bmag)
    channels.extend(bfield.B_contra[..., component] for component in range(3))
    channels.extend((spacing.dx, spacing.dy, spacing.dz))
    for axis, (face_metric, face_bfield) in enumerate(
        zip(geometry.face_metric.axes, geometry.face_bfield.axes)
    ):
        face_channels = [getattr(face_metric, name) for name in _METRIC_NAMES]
        face_channels.append(face_bfield.Bmag)
        face_channels.extend(
            face_bfield.B_contra[..., component] for component in range(3)
        )
        for side in ("lower", "upper"):
            start = 0 if side == "lower" else 1
            stop = -1 if side == "lower" else None
            face_slice = [slice(None)] * 3
            face_slice[axis] = slice(start, stop)
            channels.extend(value[tuple(face_slice)] for value in face_channels)

    if len(channels) != len(_SHARDED_FIELD_NAMES):
        raise AssertionError(
            "internal sharded geometry channel count mismatch: "
            f"got {len(channels)}, expected {len(_SHARDED_FIELD_NAMES)}"
        )
    cell_fields = jnp.stack(
        [jnp.asarray(channel, dtype=jnp.float64) for channel in channels],
        axis=-1,
    )

    # Keep the global traced payload separate from metric channels.  It has
    # the same leading shape and can therefore be passed through shard_map
    # with the same PartitionSpec.  NaN maps are the explicit opt-out used by
    # the coordinate path; do not lower them into a seemingly active map.
    map_values = [getattr(geometry.maps, name) for name in _MAP_FIELD_NAMES]
    maps_valid = bool(np.all(np.isfinite(np.asarray(map_values, dtype=np.float64))))
    map_fields = (
        jnp.stack([jnp.asarray(value, dtype=jnp.float64) for value in map_values], axis=-1)
        if maps_valid
        else None
    )

    axis_meta = tuple(
        _uniform_axis_meta(grid_axis, axis=axis)
        for axis, grid_axis in enumerate((geometry.grid.x, geometry.grid.y, geometry.grid.z))
    )
    return ShardedFciGeometry3D(
        domain=domain,
        cell_fields=cell_fields,
        axis_meta=axis_meta,
        map_fields=map_fields,
        maps_valid=maps_valid,
    )


def _empty_local_fci_maps(layout: HaloLayout3D) -> LocalFciMaps3D:
    """Inactive local-halo-only FCI maps for models that use direct stencils."""

    empty = LocalFciLocalDependencyTable(
        target_flat=jnp.zeros((1,), dtype=jnp.int32),
        source_i=jnp.zeros((1,), dtype=jnp.int32),
        source_j=jnp.zeros((1,), dtype=jnp.int32),
        source_k=jnp.zeros((1,), dtype=jnp.int32),
        weight=jnp.zeros((1,), dtype=jnp.float64),
        active=jnp.zeros((1,), dtype=bool),
    )
    direction = LocalFciDirectionMap(
        layout=layout,
        local=empty,
        target_valid=jnp.ones(layout.owned_shape, dtype=bool),
        connection_length=jnp.ones(layout.owned_shape, dtype=jnp.float64),
    )
    return LocalFciMaps3D(
        layout=layout,
        forward=direction,
        backward=direction,
        mode="local_halo_only",
    )


def _lower_local_fci_maps(
    map_fields_owned: jnp.ndarray,
    *,
    domain: LocalDomain3D,
    axis_meta: tuple[_UniformAxisMeta, _UniformAxisMeta, _UniformAxisMeta],
) -> LocalFciMaps3D:
    """Lower global-map channels owned by one shard into bilinear rows.

    The map tracer stores fractional cell indices for ordinary endpoints and
    logical endpoint coordinates for boundary endpoints.  Ordinary endpoints
    use four second-order bilinear rows.  A physical endpoint uses the same
    row width, but duplicates the radial source index and marks all four rows
    with a target-specific ``FCI_DEP_PHYSICAL_BOUNDARY`` slot.  This preserves
    the endpoint interpolation geometry without choosing a wall value here.

    A lower radial axis endpoint is topological rather than physical: its
    source is reflected to the first radial ring and shifted by half a
    poloidal period.  The resulting rows remain field-interior rows and the
    target remains valid for an unequal-leg stencil.
    """

    layout = domain.layout
    owned_shape = layout.owned_shape
    expected_shape = owned_shape + (len(_MAP_FIELD_NAMES),)
    if tuple(map_fields_owned.shape) != expected_shape:
        raise ValueError(
            f"map_fields_owned must have shape {expected_shape}, got {map_fields_owned.shape}"
        )

    map_fields_owned = jnp.asarray(map_fields_owned, dtype=jnp.float64)
    channels = {
        name: map_fields_owned[..., index]
        for index, name in enumerate(_MAP_FIELD_NAMES)
    }
    global_nx, global_ny, global_nz = domain.shard_spec.global_shape
    local_nx, local_ny, local_nz = owned_shape
    h = layout.halo_width
    shard_i = jnp.asarray(domain.runtime_shard_id(0), dtype=jnp.int32)
    shard_j = jnp.asarray(domain.runtime_shard_id(1), dtype=jnp.int32)
    shard_k = jnp.asarray(domain.runtime_shard_id(2), dtype=jnp.int32)
    local_sizes = (local_nx, local_ny, local_nz)
    global_sizes = (global_nx, global_ny, global_nz)

    ii, jj, kk = jnp.meshgrid(
        jnp.arange(local_nx, dtype=jnp.int32),
        jnp.arange(local_ny, dtype=jnp.int32),
        jnp.arange(local_nz, dtype=jnp.int32),
        indexing="ij",
    )
    target = jnp.arange(math.prod(owned_shape), dtype=jnp.int32).reshape(owned_shape)
    # Eight rows are reserved per target.  The first four are the ordinary
    # x/y bilinear endpoint; the upper-z four are inactive except for a
    # physical endpoint, where eta is also interpolated.
    target_flat = jnp.repeat(target.reshape(-1), 8)

    def _periodic_index(index: jnp.ndarray, axis: int) -> jnp.ndarray:
        if domain.periodic_axes[axis]:
            return jnp.mod(index, global_sizes[axis])
        return jnp.clip(index, 0, global_sizes[axis] - 1)

    def _owner_local(index: jnp.ndarray, axis: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        index = _periodic_index(index, axis)
        owner = index // local_sizes[axis]
        local = h + index - owner * local_sizes[axis]
        return owner.astype(jnp.int32), local.astype(jnp.int32)

    def _direction(prefix: str, sign: int) -> LocalFciDirectionMap:
        raw_x = channels[f"{prefix}_x"]
        raw_y = channels[f"{prefix}_y"]
        endpoint_x = channels[f"{prefix}_endpoint_x"]
        endpoint_y = channels[f"{prefix}_endpoint_y"]
        endpoint_z = channels[f"{prefix}_endpoint_z"]
        boundary = channels[f"{prefix}_boundary"].astype(bool)

        x_meta, y_meta, z_meta = axis_meta
        x_endpoint_index = (endpoint_x - x_meta.center0) / x_meta.spacing
        y_endpoint_index = (endpoint_y - y_meta.center0) / y_meta.spacing
        z_endpoint_index = (endpoint_z - z_meta.center0) / z_meta.spacing
        x_fractional = jnp.where(boundary, x_endpoint_index, raw_x)
        y_fractional = jnp.where(boundary, y_endpoint_index, raw_y)

        axis_hit = (
            boundary
            & bool(domain.axis_regular_axes[0])
            & (endpoint_x <= x_meta.face0 + 0.5 * abs(x_meta.spacing))
        )
        physical_boundary = boundary & ~axis_hit

        # Signed-radius topology: q(-u, theta, eta) is represented by the
        # first radial ring at theta+pi.  This is only metadata lowering;
        # no mode fitting or wall-value choice is performed here.
        y_fractional = jnp.where(
            axis_hit,
            y_fractional + 0.5 * float(global_ny),
            y_fractional,
        )
        # The lower face is half a cell below the first cell center, so its
        # signed-radial bilinear pair is x0=-1 (axis ghost), x1=0 (first ring).
        x_fractional = jnp.where(axis_hit, -0.5, x_fractional)

        # Do not clamp ordinary radial endpoints at the axis.  A fractional
        # x in (-1/2, 0) means the lower radial ghost plus the first owned
        # radial cell; the axis topology halo supplies the ghost value.
        x0 = jnp.floor(x_fractional).astype(jnp.int32)
        x0 = jnp.where(
            physical_boundary,
            jnp.clip(jnp.rint(x_fractional), 0, global_nx - 1).astype(jnp.int32),
            x0,
        )
        x1 = x0 + 1
        wx = jnp.clip(x_fractional - x0, 0.0, 1.0)
        # A physical face lies halfway between its adjacent cell center and
        # ghost center.  Keep both legs with weight 1/2 so the prepared
        # operator-specific ghost/leg fill determines the wall trace.
        outer_boundary = physical_boundary & (
            endpoint_x >= x_meta.face0 + (float(global_nx) - 0.5) * x_meta.spacing
        )
        lower_boundary = physical_boundary & ~outer_boundary
        x0 = jnp.where(outer_boundary, global_nx - 1, x0)
        x1 = jnp.where(outer_boundary, global_nx, x1)
        x0 = jnp.where(lower_boundary, -1, x0)
        x1 = jnp.where(lower_boundary, 0, x1)
        wx = jnp.where(physical_boundary, 0.5, wx)

        y0 = jnp.floor(y_fractional).astype(jnp.int32)
        wy = y_fractional - y0
        y0 = _periodic_index(y0, 1)
        y1 = _periodic_index(y0 + 1, 1)
        wy = jnp.clip(wy, 0.0, 1.0)

        z0 = jnp.floor(z_endpoint_index).astype(jnp.int32)
        z0 = _periodic_index(z0, 2)
        z1 = _periodic_index(z0 + 1, 2)
        wz = jnp.clip(z_endpoint_index - jnp.floor(z_endpoint_index), 0.0, 1.0)
        z_index = jnp.rint(z_endpoint_index).astype(jnp.int32)
        z_index = _periodic_index(z_index, 2)
        z0 = jnp.where(physical_boundary, z0, z_index)
        z1 = jnp.where(physical_boundary, z1, z_index)
        wz = jnp.where(physical_boundary, wz, 0.0)

        source_global_i = jnp.stack(
            (x0, x1, x0, x1, x0, x1, x0, x1), axis=-1
        )
        source_global_j = jnp.stack(
            (y0, y0, y1, y1, y0, y0, y1, y1), axis=-1
        )
        source_global_k = jnp.stack(
            (z0, z0, z0, z0, z1, z1, z1, z1), axis=-1
        )
        weights = jnp.stack(
            (
                (1.0 - wx) * (1.0 - wy),
                wx * (1.0 - wy),
                (1.0 - wx) * wy,
                wx * wy,
                (1.0 - wx) * (1.0 - wy) * wz,
                wx * (1.0 - wy) * wz,
                (1.0 - wx) * wy * wz,
                wx * wy * wz,
            ),
            axis=-1,
        )
        weights = weights.at[..., :4].multiply(1.0 - wz[..., None])

        owner_x, owner_local_i = _owner_local(source_global_i, 0)
        owner_y, owner_local_j = _owner_local(source_global_j, 1)
        owner_z, owner_local_k = _owner_local(source_global_k, 2)

        axis_field = ~physical_boundary
        axis_ghost_x = axis_field[..., None] & bool(domain.axis_regular_axes[0]) & (source_global_i < 0)
        owner_x = jnp.where(axis_ghost_x, 0, owner_x)
        owner_local_i = jnp.where(axis_ghost_x, h - 1, owner_local_i)
        outer_ghost_x = source_global_i == global_nx
        owner_x = jnp.where(
            outer_ghost_x,
            int(domain.shard_spec.shard_counts[0]) - 1,
            owner_x,
        )
        owner_local_i = jnp.where(outer_ghost_x, h + local_nx, owner_local_i)
        lower_physical_ghost_x = lower_boundary[..., None] & (source_global_i < 0)
        owner_x = jnp.where(lower_physical_ghost_x, 0, owner_x)
        owner_local_i = jnp.where(lower_physical_ghost_x, h - 1, owner_local_i)
        # The second radial leg of the axis pair is the first owned cell.
        owner_x = jnp.where(
            bool(domain.axis_regular_axes[0]) & axis_field[..., None] & (source_global_i == 0),
            0,
            owner_x,
        )
        owner_linear = (
            owner_x
            + int(domain.shard_spec.shard_counts[0])
            * (owner_y + int(domain.shard_spec.shard_counts[1]) * owner_z)
        ).astype(jnp.int32)
        my_linear = (
            shard_i
            + int(domain.shard_spec.shard_counts[0])
            * (shard_j + int(domain.shard_spec.shard_counts[1]) * shard_k)
        ).astype(jnp.int32)
        same_shard = owner_linear == my_linear

        row_physical = jnp.broadcast_to(physical_boundary[..., None], weights.shape)
        row_number = jnp.arange(weights.shape[-1], dtype=jnp.int32)
        row_active = row_physical | (
            (~row_physical) & (row_number[None, None, None, :] < 4)
        )
        # Physical endpoints are read from the prepared owner-shard ghost
        # halo, just like ordinary field endpoints.  ``endpoint_kind`` and
        # ``value_slot`` preserve the later operator-aware wall metadata, but
        # are not used as a sampler substitution here.
        local_active = row_active & same_shard
        remote_active = row_active & ~same_shard
        dependency_kind = jnp.full(weights.shape, FCI_DEP_FIELD_INTERIOR, dtype=jnp.int32)
        value_slot = jnp.broadcast_to(target[..., None], weights.shape)

        source_i = h + source_global_i - owner_x * local_nx
        source_j = h + source_global_j - owner_y * local_ny
        source_k = h + source_global_k - owner_z * local_nz

        local = LocalFciLocalDependencyTable(
            target_flat=target_flat,
            source_i=source_i.reshape(-1),
            source_j=source_j.reshape(-1),
            source_k=source_k.reshape(-1),
            weight=weights.reshape(-1),
            active=local_active.reshape(-1),
            dependency_kind=dependency_kind.reshape(-1),
            value_slot=value_slot.reshape(-1),
        )
        remote = None
        if math.prod(domain.shard_spec.shard_counts) > 1:
            remote = LocalFciRemoteDependencyTable(
                target_flat=target_flat,
                weight=weights.reshape(-1),
                receive_slot=jnp.arange(target_flat.size, dtype=jnp.int32),
                active=remote_active.reshape(-1),
                request_active=remote_active.reshape(-1),
                request_dependency_kind=jnp.where(
                    remote_active.reshape(-1),
                    FCI_DEP_FIELD_INTERIOR,
                    FCI_DEP_INVALID,
                ).astype(jnp.int32),
                request_source_global_i=source_global_i.reshape(-1),
                request_source_global_j=source_global_j.reshape(-1),
                request_source_global_k=source_global_k.reshape(-1),
                request_source_shard_index=jnp.stack(
                    (owner_x, owner_y, owner_z), axis=-1
                ).reshape((-1, 3)),
                request_source_shard_linear=owner_linear.reshape(-1),
                request_source_owner_local_i=owner_local_i.reshape(-1),
                request_source_owner_local_j=owner_local_j.reshape(-1),
                request_source_owner_local_k=owner_local_k.reshape(-1),
                request_value_slot=jnp.where(
                    row_physical.reshape(-1),
                    value_slot.reshape(-1),
                    jnp.zeros(target_flat.shape, dtype=jnp.int32),
                ),
            )

        endpoint_kind = jnp.where(
            physical_boundary,
            FCI_DEP_PHYSICAL_BOUNDARY,
            FCI_DEP_FIELD_INTERIOR,
        ).astype(jnp.int32)
        return LocalFciDirectionMap(
            layout=layout,
            local=local,
            remote=remote,
            target_valid=jnp.ones(owned_shape, dtype=bool),
            connection_length=channels[f"{prefix}_length"],
            endpoint_kind=endpoint_kind,
            endpoint_b_contra_x=channels[f"{prefix}_endpoint_b_contra_x"],
            endpoint_b_contra_y=channels[f"{prefix}_endpoint_b_contra_y"],
            endpoint_b_contra_z=channels[f"{prefix}_endpoint_b_contra_z"],
            endpoint_bmag=channels[f"{prefix}_endpoint_bmag"],
        )

    return LocalFciMaps3D(
        layout=layout,
        forward=_direction("forward", +1),
        backward=_direction("backward", -1),
        mode=("remote_dependencies" if math.prod(domain.shard_spec.shard_counts) > 1 else "local_halo_only"),
    )


def _local_axis_grid(
    layout: HaloLayout3D,
    *,
    axis: int,
    meta: _UniformAxisMeta,
    shard_id,
) -> LocalGrid1D:
    local_size = layout.owned_shape[axis]
    h = layout.halo_width
    start = jnp.asarray(shard_id, dtype=jnp.int32) * local_size
    center_indices = start + jnp.arange(-h, local_size + h)
    face_indices = start + jnp.arange(-h, local_size + h + 1)
    return LocalGrid1D(
        layout=layout,
        axis=axis,
        centers_halo=meta.center0 + center_indices * meta.spacing,
        faces_halo=meta.face0 + face_indices * meta.spacing,
        owned_start_global=0,
        owned_stop_global=local_size,
    )


def _axis_slice(values: jnp.ndarray, axis: int, start: int | None, stop: int | None) -> jnp.ndarray:
    index = [slice(None)] * values.ndim
    index[axis] = slice(start, stop)
    return values[tuple(index)]


def _lift_cell_halo_to_faces(values: jnp.ndarray, *, axis: int) -> jnp.ndarray:
    """Midpoint-interpolate a halo-shaped cell array onto one face family."""

    lower = _axis_slice(values, axis, 0, 1)
    upper = _axis_slice(values, axis, values.shape[axis] - 1, None)
    interior = 0.5 * (_axis_slice(values, axis, 0, -1) + _axis_slice(values, axis, 1, None))
    return jnp.concatenate((lower, interior, upper), axis=axis)


def assemble_local_fci_geometry(
    sharded_geometry: ShardedFciGeometry3D,
    cell_fields_owned: jnp.ndarray,
    map_fields_owned: jnp.ndarray | None = None,
) -> LocalFciGeometry3D:
    """Assemble one shard's ``LocalFciGeometry3D`` inside ``shard_map``.

    The owned geometry block is injected into a halo-shaped array, exchanged
    across shard interfaces, and topology-filled on periodic and axis-regular
    sides. Physical-side geometry halos are left unfilled; the direct
    two-field operators consume owned geometry values and close physical field
    planes with one-sided stencils. Face-family halo values use midpoint
    interpolation only as inactive/communication padding. Every owned face is
    overwritten by the exact sided samples packed by
    :func:`build_local_fci_geometries`; singular metric components are never
    reconstructed by arithmetic averaging.  When ``map_fields_owned`` is
    supplied, the retained global FCI map channels are lowered into local
    bilinear/trilinear dependency tables.  Omitting it preserves the direct
    coordinate path and installs inactive maps.
    """

    domain = sharded_geometry.domain
    layout = domain.layout
    expected_shape = layout.owned_shape + (len(_SHARDED_FIELD_NAMES),)
    if tuple(cell_fields_owned.shape) != expected_shape:
        raise ValueError(
            f"cell_fields_owned must have shape {expected_shape}, got {cell_fields_owned.shape}"
        )

    cell_fields_owned = jnp.asarray(cell_fields_owned, dtype=jnp.float64)
    fields_halo = inject_owned_vector_field_to_halo(
        cell_fields_owned[..., : len(_CELL_FIELD_NAMES)],
        layout,
    )
    fields_halo = HaloExchange3D()(fields_halo, domain)
    fields_halo = TopologyHaloFiller3D(
        rules=(LocalPeriodicTopologyRule3D(),)
    )(fields_halo, domain)
    if domain.axis_regular_axes[0]:
        global_theta = int(domain.shard_spec.global_shape[1])
        local_theta = int(domain.layout.owned_shape[1])
        theta_shards = int(domain.shard_spec.shard_counts[1])
        shard_shift, local_shift = divmod(global_theta // 2, local_theta)
        fields_halo = PolarAxisRegularVectorRule3D(
            axis=0,
            side="lower",
            angular_axis=1,
            mesh_axis_name=domain.mesh_axis_names[1],
            source_shard_offset=-shard_shift,
            local_shift_cells=local_shift,
            component_transform=jnp.diag(
                jnp.asarray(_AXIS_REGULAR_CELL_FIELD_PARITY, dtype=jnp.float64)
            ),
        )(fields_halo, domain)
    channel = {
        name: fields_halo[..., index] for index, name in enumerate(_CELL_FIELD_NAMES)
    }
    packed_face_channel = {
        name: cell_fields_owned[..., len(_CELL_FIELD_NAMES) + index]
        for index, name in enumerate(_FACE_PACKED_FIELD_NAMES)
    }

    def _exact_owned_face(
        *,
        axis: int,
        axis_name: str,
        field_name: str,
    ) -> jnp.ndarray:
        lower = packed_face_channel[f"{axis_name}_lower_{field_name}"]
        upper = packed_face_channel[f"{axis_name}_upper_{field_name}"]
        last = [slice(None)] * 3
        last[axis] = slice(-1, None)
        return jnp.concatenate((lower, upper[tuple(last)]), axis=axis)

    def _exact_face_halo(
        cell_halo: jnp.ndarray,
        *,
        axis: int,
        axis_name: str,
        field_name: str,
    ) -> jnp.ndarray:
        lifted = _lift_cell_halo_to_faces(cell_halo, axis=axis)
        return lifted.at[layout.face_control_slices(axis)].set(
            _exact_owned_face(
                axis=axis,
                axis_name=axis_name,
                field_name=field_name,
            )
        )

    cell_metric = LocalMetricGeometry(
        layout=layout,
        location="cell",
        **{f"{name}_halo": channel[name] for name in _METRIC_NAMES},
    )
    face_locations = ("x_face", "y_face", "z_face")
    face_metric = LocalFaceMetricGeometry(
        layout=layout,
        **{
            axis_name: LocalMetricGeometry(
                layout=layout,
                location=face_locations[axis],
                **{
                    f"{name}_halo": _exact_face_halo(
                        channel[name],
                        axis=axis,
                        axis_name=axis_name,
                        field_name=name,
                    )
                    for name in _METRIC_NAMES
                },
            )
            for axis, axis_name in enumerate(("x", "y", "z"))
        },
    )

    B_contra_halo = jnp.stack(
        (channel["B_contra_x"], channel["B_contra_y"], channel["B_contra_z"]),
        axis=-1,
    )
    cell_bfield = LocalBFieldGeometry(
        layout=layout,
        B_contra_halo=B_contra_halo,
        Bmag_halo=channel["Bmag"],
        location="cell",
    )
    face_bfield = LocalFaceBFieldGeometry(
        layout=layout,
        **{
            axis_name: LocalBFieldGeometry(
                layout=layout,
                B_contra_halo=jnp.stack(
                    tuple(
                        _exact_face_halo(
                            B_contra_halo[..., component],
                            axis=axis,
                            axis_name=axis_name,
                            field_name=f"B_contra_{component_name}",
                        )
                        for component, component_name in enumerate(("x", "y", "z"))
                    ),
                    axis=-1,
                ),
                Bmag_halo=_exact_face_halo(
                    channel["Bmag"],
                    axis=axis,
                    axis_name=axis_name,
                    field_name="Bmag",
                ),
                location=face_locations[axis],
            )
            for axis, axis_name in enumerate(("x", "y", "z"))
        },
    )

    spacing = LocalSpacing3D(
        layout=layout,
        dx_halo=channel["dx"],
        dy_halo=channel["dy"],
        dz_halo=channel["dz"],
    )
    grid = LocalCellCenteredGrid3D(
        layout=layout,
        x=_local_axis_grid(layout, axis=0, meta=sharded_geometry.axis_meta[0], shard_id=domain.runtime_shard_id(0)),
        y=_local_axis_grid(layout, axis=1, meta=sharded_geometry.axis_meta[1], shard_id=domain.runtime_shard_id(1)),
        z=_local_axis_grid(layout, axis=2, meta=sharded_geometry.axis_meta[2], shard_id=domain.runtime_shard_id(2)),
    )

    face_shapes = tuple(layout.face_control_shape(axis) for axis in range(3))
    regular = LocalRegularFaceGeometry3D(
        layout=layout,
        x_area=jnp.ones(face_shapes[0]),
        y_area=jnp.ones(face_shapes[1]),
        z_area=jnp.ones(face_shapes[2]),
        x_area_fraction=jnp.ones(face_shapes[0]),
        y_area_fraction=jnp.ones(face_shapes[1]),
        z_area_fraction=jnp.ones(face_shapes[2]),
        x_open_mask=jnp.ones(face_shapes[0], dtype=bool),
        y_open_mask=jnp.ones(face_shapes[1], dtype=bool),
        z_open_mask=jnp.ones(face_shapes[2], dtype=bool),
    )
    maps = (
        _empty_local_fci_maps(layout)
        if map_fields_owned is None
        else _lower_local_fci_maps(
            map_fields_owned,
            domain=domain,
            axis_meta=sharded_geometry.axis_meta,
        )
    )
    return LocalFciGeometry3D(
        layout=layout,
        grid=grid,
        maps=maps,
        spacing=spacing,
        cell_metric=cell_metric,
        face_metric=face_metric,
        cell_bfield=cell_bfield,
        face_bfield=face_bfield,
        regular_face_geometry=regular,
        cell_volume_geometry=LocalCellVolumeGeometry3D(
            layout=layout,
            volume=jnp.asarray(cell_metric.J_owned, dtype=jnp.float64),
            volume_fraction=jnp.ones(layout.owned_shape),
        ),
    )


def assemble_single_device_local_fci_geometry(
    sharded_geometry: ShardedFciGeometry3D,
    cell_fields_owned: jnp.ndarray | None = None,
    map_fields_owned: jnp.ndarray | None = None,
) -> LocalFciGeometry3D:
    """Assemble a one-device local geometry outside ``shard_map``.

    This is the strict host-preprocessing counterpart of
    :func:`assemble_local_fci_geometry`.  It accepts only a geometry built
    with exactly one shard in every logical direction and reuses the same
    numerical assembly routine after replacing only the execution metadata
    with an undecomposed domain.  Consequently every runtime shard id is the
    static integer zero, while periodic and lower-axis-regular topology
    handling remains identical to the normal shard-map path.
    """
    if not isinstance(sharded_geometry, ShardedFciGeometry3D):
        raise TypeError(
            "sharded_geometry must be a ShardedFciGeometry3D instance, "
            f"got {type(sharded_geometry).__name__}"
        )
    if sharded_geometry.shard_counts != (1, 1, 1):
        raise ValueError(
            "assemble_single_device_local_fci_geometry requires shard_counts "
            f"exactly (1, 1, 1), got {sharded_geometry.shard_counts}"
        )
    if cell_fields_owned is None:
        cell_fields_owned = sharded_geometry.cell_fields
    if map_fields_owned is None and sharded_geometry.maps_valid:
        map_fields_owned = sharded_geometry.map_fields
    host_domain = replace(sharded_geometry.domain, mesh_axis_names=(None, None, None))
    host_geometry = replace(sharded_geometry, domain=host_domain)
    return assemble_local_fci_geometry(
        host_geometry,
        cell_fields_owned,
        map_fields_owned,
    )


def _make_prepared_local_stencil_builder(
    domain: LocalDomain3D,
    context: StencilBuilderContext,
) -> Callable[..., object]:
    """Wrap halo preparation plus the one-sided physical local stencil build.

    The returned builder receives shard-owned fields, injects them into halo arrays,
    exchanges shard-interface halos, topology-fills undecomposed periodic
    sides, and closes physical side planes with one-sided derivative
    stencils. Like the global direct path, it consumes no face-BC payload.
    """

    halo_exchange = HaloExchange3D()
    topology_filler = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),))

    def _build(
        field_owned: jnp.ndarray,
        geometry: LocalFciGeometry3D,
    ):
        field_halo = inject_owned_field_to_halo(
            jnp.asarray(field_owned, dtype=jnp.float64),
            domain.layout,
        )
        field_halo = halo_exchange(field_halo, domain)
        field_halo = topology_filler(field_halo, domain)
        return build_local_direct_stencil_one_sided_physical_from_halo(
            field_halo,
            geometry,
            context,
        )

    return _build


@dataclass(frozen=True)
class Sharded2FieldStepInfo:
    """Static sharding facts about a sharded two-field RK4 step."""

    mesh: Mesh
    partition_spec: P
    state_sharding: NamedSharding
    domain: LocalDomain3D
    geometry: ShardedFciGeometry3D


def make_sharded_2field_step(
    geometry: FciGeometry3D,
    shard_counts: tuple[int, int, int],
    parameters: Fci2FieldRhsParameters,
    boundary_conditions: dict[str, object] | None = None,
    *,
    dt: float,
    halo_width: int = 1,
) -> tuple[object, Sharded2FieldStepInfo]:
    """Build a jitted sharded RK4 step for the reduced two-field model.

    Returns ``(step_fn, info)`` where ``step_fn(state)`` advances a global
    :class:`Fci2FieldState` by one RK4 step under ``shard_map`` with in/out
    partition spec ``P("x", "y", "z")`` on every state field. Each of the
    four stage RHS evaluations prepares fresh state halos (exchange plus
    periodic topology fill) before building stencils.

    The local direct stencil path uses its regular one-sided physical closure.
    Field-specific boundary payloads are not supported by this reduced model.
    """

    shard_counts = tuple(int(value) for value in shard_counts)
    boundary_conditions = dict(boundary_conditions or {})
    if boundary_conditions:
        raise ValueError(
            "the local two-field path uses one-sided physical stencils and "
            "does not accept field-specific boundary_conditions"
        )

    mesh = make_shard_mesh(shard_counts)
    sharded_geometry = build_local_fci_geometries(geometry, shard_counts, halo_width=halo_width)
    domain = sharded_geometry.domain
    partition_spec = P(*_MESH_AXIS_NAMES)
    state_sharding = NamedSharding(mesh, partition_spec)

    cell_fields_sharded = jax.device_put(sharded_geometry.cell_fields, state_sharding)
    map_fields = sharded_geometry.map_fields
    if map_fields is None:
        map_fields = jnp.zeros(
            sharded_geometry.global_shape + (len(_MAP_FIELD_NAMES),),
            dtype=jnp.float64,
        )
    map_fields_sharded = jax.device_put(map_fields, state_sharding)

    def _assemble_with_maps(cell_fields_owned, map_fields_owned):
        return assemble_local_fci_geometry(
            sharded_geometry,
            cell_fields_owned,
            map_fields_owned if sharded_geometry.maps_valid else None,
        )

    curvature_sharded = jax.jit(
        jax.shard_map(
            lambda cell_fields_owned, map_fields_owned: build_local_curvature_coefficients(
                _assemble_with_maps(cell_fields_owned, map_fields_owned),
                domain,
                periodic_axes=domain.periodic_axes,
                axis_regular_axes=(False, False, False),
            ),
            mesh=mesh,
            in_specs=(partition_spec, partition_spec),
            out_specs=partition_spec,
            check_vma=False,
        )
    )(cell_fields_sharded, map_fields_sharded)
    timestep = jnp.asarray(dt, dtype=jnp.float64)

    def _kernel(
        density,
        v_parallel,
        density_background,
        curvature_owned,
        cell_fields_owned,
        map_fields_owned,
    ):
        local_geometry = _assemble_with_maps(cell_fields_owned, map_fields_owned)
        context = StencilBuilderContext(layout=domain.layout, domain=domain)
        stencil_builder = _make_prepared_local_stencil_builder(domain, context)
        state = Fci2FieldState(
            density=density,
            v_parallel=v_parallel,
            density_background=density_background,
        )

        def _rhs_fn(stage_state, stage_time, carry):
            del stage_time
            result = compute_local_2field_rhs(
                stage_state,
                geometry=local_geometry,
                stencil_builder=stencil_builder,
                parameters=parameters,
                curvature_coefficients=curvature_owned,
            )
            return result.rhs, carry, None

        step = Rk4Stepper(_rhs_fn)(
            state,
            time=0.0,
            timestep=timestep,
            carry=None,
        )
        next_state = step.state
        return next_state.density, next_state.v_parallel, next_state.density_background

    sharded_kernel = jax.jit(
        jax.shard_map(
            _kernel,
            mesh=mesh,
            in_specs=(partition_spec,) * 6,
            out_specs=(partition_spec,) * 3,
            check_vma=False,
        )
    )

    def step_fn(state: Fci2FieldState) -> Fci2FieldState:
        density, v_parallel, density_background = sharded_kernel(
            jax.device_put(jnp.asarray(state.density, dtype=jnp.float64), state_sharding),
            jax.device_put(jnp.asarray(state.v_parallel, dtype=jnp.float64), state_sharding),
            jax.device_put(jnp.asarray(state.density_background, dtype=jnp.float64), state_sharding),
            curvature_sharded,
            cell_fields_sharded,
            map_fields_sharded,
        )
        return Fci2FieldState(
            density=density,
            v_parallel=v_parallel,
            density_background=density_background,
        )

    info = Sharded2FieldStepInfo(
        mesh=mesh,
        partition_spec=partition_spec,
        state_sharding=state_sharding,
        domain=domain,
        geometry=sharded_geometry,
    )
    return step_fn, info


__all__ = [
    "Sharded2FieldStepInfo",
    "ShardedFciGeometry3D",
    "assemble_local_fci_geometry",
    "assemble_single_device_local_fci_geometry",
    "build_local_fci_geometries",
    "make_shard_mesh",
    "make_sharded_2field_step",
]
