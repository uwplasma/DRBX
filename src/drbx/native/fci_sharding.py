"""Multi-device sharded execution helpers for the FCI stack.

This module promotes the proven ``shard_map`` harness patterns from
``tests/test_fci_operators_domain_decomp.py`` into a small library API and
extends them from operator-level tests to a full two-field RHS + RK4 step:

- :func:`make_shard_mesh` builds the three-axis execution mesh.
- :func:`build_local_fci_geometries` converts a global :class:`FciGeometry3D`
  into the per-shard representation the ``shard_map`` kernel consumes: a
  ``LocalDomain3D`` plus a stacked bundle of cell-centered geometry fields
  that is partitioned with ``PartitionSpec("x", "y", "z")``.
- :func:`assemble_local_fci_geometry` runs inside ``shard_map`` and assembles
  a :class:`LocalFciGeometry3D` from one shard's owned geometry block using
  halo exchange, periodic topology filling, and the runtime shard index.
- :func:`make_sharded_2field_step` returns a jitted RK4 step for the reduced
  two-field model where every stage prepares state halos (exchange plus
  periodic topology fill) before evaluating the RHS on local geometry.

The reduced two-field direct stencil path closes physical sides with
one-sided derivative stencils and consumes no face-BC payload, matching the
global single-device path exactly; ``boundary_conditions`` entries are
forwarded verbatim to :func:`compute_2field_rhs` for both paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    LocalFciMaps3D,
    LocalGrid1D,
    LocalMetricGeometry,
    LocalRegularFaceGeometry3D,
    LocalSpacing3D,
    SIDE_PHYSICAL,
    SIDE_SIMPLE_PERIODIC,
    ShardSpec3D,
    StencilBuilderContext,
    build_curvature_coefficients,
    build_local_direct_stencil_one_sided_physical_from_halo,
)
from .fci_2_field_rhs import Fci2FieldRhsParameters, Fci2FieldState, compute_2field_rhs
from .fci_halo import HaloExchange3D, LocalPeriodicTopologyRule3D, TopologyHaloFiller3D
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

    ``cell_fields`` is a global ``(nx, ny, nz, len(_CELL_FIELD_NAMES))`` array
    of cell-centered geometry values. It is partitioned on the mesh with
    ``PartitionSpec("x", "y", "z")`` so each kernel instance receives its
    owned block; :func:`assemble_local_fci_geometry` fills the halos through
    the same exchange/topology pipeline used for state fields.
    """

    domain: LocalDomain3D
    cell_fields: jnp.ndarray
    axis_meta: tuple[_UniformAxisMeta, _UniformAxisMeta, _UniformAxisMeta]

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
) -> ShardedFciGeometry3D:
    """Convert a global FCI geometry into the shard-local kernel inputs."""

    if not isinstance(geometry, FciGeometry3D):
        raise TypeError(f"geometry must be an FciGeometry3D instance, got {type(geometry).__name__}")
    shard_counts = tuple(int(value) for value in shard_counts)
    global_shape = geometry.shape
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
    side_kinds = tuple(
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
        periodic_axes=tuple(bool(value) for value in periodic_axes),
        halo_width=halo_width,
        side_kind_lower=side_kinds,
        side_kind_upper=side_kinds,
    )
    domain = LocalDomain3D(shard_spec=spec, layout=layout, mesh_axis_names=_MESH_AXIS_NAMES)

    metric = geometry.cell_metric
    bfield = geometry.cell_bfield
    spacing = geometry.spacing
    channels = [getattr(metric, name) for name in _METRIC_NAMES]
    channels.append(bfield.Bmag)
    channels.extend(bfield.B_contra[..., component] for component in range(3))
    channels.extend((spacing.dx, spacing.dy, spacing.dz))
    cell_fields = jnp.stack(
        [jnp.asarray(channel, dtype=jnp.float64) for channel in channels],
        axis=-1,
    )

    axis_meta = tuple(
        _uniform_axis_meta(grid_axis, axis=axis)
        for axis, grid_axis in enumerate((geometry.grid.x, geometry.grid.y, geometry.grid.z))
    )
    return ShardedFciGeometry3D(domain=domain, cell_fields=cell_fields, axis_meta=axis_meta)


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
) -> LocalFciGeometry3D:
    """Assemble one shard's ``LocalFciGeometry3D`` inside ``shard_map``.

    The owned geometry block is injected into a halo-shaped array, exchanged
    across shard interfaces, and topology-filled on undecomposed periodic
    sides. Physical-side geometry halos are left unfilled; the two-field
    operators only consume owned geometry values, and physical field planes
    are closed with one-sided stencils. Face-family geometry is midpoint
    interpolation of the cell values and is unused by the direct two-field
    operators.
    """

    domain = sharded_geometry.domain
    layout = domain.layout
    expected_shape = layout.owned_shape + (len(_CELL_FIELD_NAMES),)
    if tuple(cell_fields_owned.shape) != expected_shape:
        raise ValueError(
            f"cell_fields_owned must have shape {expected_shape}, got {cell_fields_owned.shape}"
        )

    fields_halo = inject_owned_vector_field_to_halo(
        jnp.asarray(cell_fields_owned, dtype=jnp.float64),
        layout,
    )
    fields_halo = HaloExchange3D()(fields_halo, domain)
    fields_halo = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),))(fields_halo, domain)
    channel = {
        name: fields_halo[..., index] for index, name in enumerate(_CELL_FIELD_NAMES)
    }

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
                    f"{name}_halo": _lift_cell_halo_to_faces(channel[name], axis=axis)
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
                B_contra_halo=_lift_cell_halo_to_faces(B_contra_halo, axis=axis),
                Bmag_halo=_lift_cell_halo_to_faces(channel["Bmag"], axis=axis),
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
    return LocalFciGeometry3D(
        layout=layout,
        grid=grid,
        maps=_empty_local_fci_maps(layout),
        spacing=spacing,
        cell_metric=cell_metric,
        face_metric=face_metric,
        cell_bfield=cell_bfield,
        face_bfield=face_bfield,
        regular_face_geometry=regular,
        cell_volume_geometry=LocalCellVolumeGeometry3D(
            layout=layout,
            volume=jnp.ones(layout.owned_shape),
            volume_fraction=jnp.ones(layout.owned_shape),
        ),
    )


def _make_prepared_local_stencil_builder(
    domain: LocalDomain3D,
    context: StencilBuilderContext,
) -> Callable[..., object]:
    """Wrap halo preparation plus the one-sided physical local stencil build.

    The returned builder follows the ``compute_2field_rhs`` stencil-builder
    call convention ``(field, geometry, periodic_axes=..., face_bc=..., ...)``
    but receives shard-owned fields: it injects them into halo arrays,
    exchanges shard-interface halos, topology-fills undecomposed periodic
    sides, and closes physical side planes with one-sided derivative
    stencils. Like the global direct path, it consumes no face-BC payload.
    """

    halo_exchange = HaloExchange3D()
    topology_filler = TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),))

    def _build(
        field_owned: jnp.ndarray,
        geometry: LocalFciGeometry3D,
        _context: object | None = None,
        face_bc: object | None = None,
        cut_wall_geometry: object | None = None,
        cut_wall_bc: object | None = None,
        *,
        periodic_axes: object | None = None,
    ):
        del _context, face_bc, cut_wall_geometry, cut_wall_bc, periodic_axes
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

    # The prepared builder intentionally follows the current RHS-facing
    # keyword contract.  Wrapping it in the legacy three-argument
    # ``LocalStencilBuilder`` adapter would discard those boundary arguments.
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

    ``boundary_conditions`` maps field names (``"density"``, ``"phi"``,
    ``"v_parallel"``) to face-BC payloads forwarded to
    :func:`compute_2field_rhs`; the two-field direct stencil path consumes no
    face-BC payload, so ``None`` matches the single-device behavior.
    """

    shard_counts = tuple(int(value) for value in shard_counts)
    boundary_conditions = dict(boundary_conditions or {})
    known_fields = ("density", "phi", "v_parallel")
    unknown = sorted(set(boundary_conditions) - set(known_fields))
    if unknown:
        raise ValueError(f"boundary_conditions has unknown fields {unknown}; expected {known_fields}")
    face_bcs = {name: boundary_conditions.get(name) for name in known_fields}

    mesh = make_shard_mesh(shard_counts)
    sharded_geometry = build_local_fci_geometries(geometry, shard_counts, halo_width=halo_width)
    domain = sharded_geometry.domain
    partition_spec = P(*_MESH_AXIS_NAMES)
    state_sharding = NamedSharding(mesh, partition_spec)

    curvature_coefficients = build_curvature_coefficients(
        geometry,
        periodic_axes=domain.periodic_axes,
    )
    curvature_sharded = jax.device_put(curvature_coefficients, state_sharding)
    cell_fields_sharded = jax.device_put(sharded_geometry.cell_fields, state_sharding)
    timestep = jnp.asarray(dt, dtype=jnp.float64)

    def _kernel(density, v_parallel, density_background, curvature_owned, cell_fields_owned):
        local_geometry = assemble_local_fci_geometry(sharded_geometry, cell_fields_owned)
        context = StencilBuilderContext(layout=domain.layout, domain=domain)
        stencil_builder = _make_prepared_local_stencil_builder(domain, context)
        state = Fci2FieldState(
            density=density,
            v_parallel=v_parallel,
            density_background=density_background,
        )

        def _rhs_fn(stage_state, stage_time, carry):
            del stage_time
            result, _timings = compute_2field_rhs(
                stage_state,
                geometry=local_geometry,
                stencil_builder=stencil_builder,
                parameters=parameters,
                curvature_coefficients=curvature_owned,
                density_face_bc=face_bcs["density"],
                phi_face_bc=face_bcs["phi"],
                v_parallel_face_bc=face_bcs["v_parallel"],
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
            in_specs=(partition_spec,) * 5,
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
    "build_local_fci_geometries",
    "make_shard_mesh",
    "make_sharded_2field_step",
]
