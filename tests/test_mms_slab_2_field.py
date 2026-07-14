from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import time as time_module

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from jax_drb.geometry import (
    BFieldGeometry,
    CellCenteredGrid3D,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    FciGeometry3D,
    FciMaps3D,
    Grid1D,
    HaloLayout3D,
    LocalBFieldGeometry,
    LocalCellCenteredGrid3D,
    LocalCellVolumeGeometry3D,
    LocalDomain3D,
    LocalFciDirectionMap,
    LocalFciGeometry3D,
    LocalFciLocalDependencyTable,
    LocalFciMaps3D,
    LocalFaceBFieldGeometry,
    LocalFaceMetricGeometry,
    LocalGrid1D,
    LocalMetricGeometry,
    LocalRegularFaceGeometry3D,
    LocalSpacing3D,
    MetricGeometry,
    ShardSpec3D,
    SIDE_PHYSICAL,
    Spacing3D,
    StencilBuilderContext,
    build_local_stencil_from_field,
    logical_grid_from_axis_vectors,
)
from jax_drb.native import Rk4Stepper
from jax_drb.native.fci_2_field_rhs import (
    Fci2FieldRhsParameters,
    Fci2FieldState,
)
from jax_drb.native.fci_boundaries import (
    BC_DIRICHLET,
    LocalBoundaryData3D,
    LocalBoundaryFaceBC3D,
)
from jax_drb.native.fci_halo import (
    GhostFillWeights1D,
    HaloExchange3D,
    PhysicalGhostCellFiller3D,
    PreparedLocalState3D,
)
from jax_drb.native.fci_model import FciFieldBundle, inject_owned_state_to_halo
from jax_drb.native.fci_operators import (
    local_curvature_op,
    local_grad_parallel_op_direct,
    local_poisson_bracket_op,
)


A = 0.1
B = 0.1
B0 = 1.0
alpha = 0.2
omega = 2.0 * jnp.pi
rho_star = 1.0
tf = 0.1
num_steps = 50

_MESH_AXIS_NAMES = ("x", "y", "z")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _Slab2FieldFaceBCBundle(FciFieldBundle):
    density: LocalBoundaryFaceBC3D
    v_parallel: LocalBoundaryFaceBC3D
    density_background: LocalBoundaryFaceBC3D


def _resolution_step_count(
    resolution: int,
    *,
    base_resolution: int = 20,
    base_steps: int = num_steps,
) -> int:
    """Scale the timestep count like ``sqrt(resolution)`` relative to the base grid."""

    scale = np.sqrt(float(resolution) / float(base_resolution))
    return max(1, int(round(float(base_steps) * scale)))


def _format_progress_bar(
    completed: int,
    total: int,
    *,
    start_time: float,
    width: int = 28,
) -> str:
    fraction = 1.0 if total <= 0 else min(1.0, max(0.0, float(completed) / float(total)))
    filled = int(round(float(width) * fraction))
    elapsed = time_module.perf_counter() - start_time
    rate = float(completed) / elapsed if elapsed > 0.0 and completed > 0 else 0.0
    remaining = (float(total - completed) / rate) if rate > 0.0 else float("nan")
    eta_text = "--:--" if not np.isfinite(remaining) else _format_duration(remaining)
    return (
        f"[{'#' * filled}{'.' * (width - filled)}] "
        f"{completed:>4d}/{total:<4d} {100.0 * fraction:6.2f}% "
        f"elapsed={_format_duration(elapsed)} eta={eta_text}"
    )


def _format_duration(seconds: float) -> str:
    whole_seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _mms_background_density_from_coordinates(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 + alpha * jnp.cos(jnp.pi * x)


def _mms_phi_from_coordinates(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    time: float | jax.Array,
) -> jnp.ndarray:
    return (
        A
        * jnp.sin(jnp.pi * x)
        * jnp.cos(2.0 * jnp.pi * y)
        * jnp.sin(jnp.pi * z)
        * jnp.cos(omega * time)
    )


def _mms_density_from_coordinates(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    time: float | jax.Array,
) -> jnp.ndarray:
    return _mms_background_density_from_coordinates(x) * jnp.exp(
        _mms_phi_from_coordinates(x, y, z, time)
    )


def _mms_v_parallel_from_coordinates(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    time: float | jax.Array,
) -> jnp.ndarray:
    return (
        B
        * jnp.cos(jnp.pi * x)
        * jnp.sin(2.0 * jnp.pi * y)
        * jnp.cos(jnp.pi * z)
        * jnp.sin(omega * time)
    )


def build_slab_2field_geometry(nx: int, ny: int, nz: int) -> FciGeometry3D:
    """Build a simple Cartesian slab geometry with constant unit ``B`` along ``z``."""

    x_axis = jnp.linspace(0.0, 1.0, nx, dtype=jnp.float64)
    y_axis = jnp.linspace(0.0, 1.0, ny, dtype=jnp.float64)
    z_axis = jnp.linspace(0.0, 1.0, nz, dtype=jnp.float64)
    target_shape = (nx, ny, nz)
    grid = CellCenteredGrid3D(
        x=Grid1D.from_centers(x_axis),
        y=Grid1D.from_centers(y_axis),
        z=Grid1D.from_centers(z_axis),
    )
    ones = jnp.ones(target_shape, dtype=jnp.float64)
    zeros = jnp.zeros(target_shape, dtype=jnp.float64)
    b_contravariant = jnp.stack((zeros, zeros, ones), axis=-1)

    def _metric(shape: tuple[int, int, int]) -> MetricGeometry:
        ones_local = jnp.ones(shape, dtype=jnp.float64)
        zeros_local = jnp.zeros(shape, dtype=jnp.float64)
        return MetricGeometry(
            J=ones_local,
            g11=ones_local,
            g22=ones_local,
            g33=ones_local,
            g12=zeros_local,
            g13=zeros_local,
            g23=zeros_local,
            g_11=ones_local,
            g_22=ones_local,
            g_33=ones_local,
            g_12=zeros_local,
            g_13=zeros_local,
            g_23=zeros_local,
        )

    cell_metric = _metric(target_shape)
    face_metric = FaceMetricGeometry(
        x=_metric((nx + 1, ny, nz)),
        y=_metric((nx, ny + 1, nz)),
        z=_metric((nx, ny, nz + 1)),
    )
    cell_bfield = BFieldGeometry(B_contra=b_contravariant, Bmag=ones)
    face_b_contra_x = jnp.broadcast_to(
        jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
        (nx + 1, ny, nz, 3),
    )
    face_b_contra_y = jnp.broadcast_to(
        jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
        (nx, ny + 1, nz, 3),
    )
    face_b_contra_z = jnp.broadcast_to(
        jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
        (nx, ny, nz + 1, 3),
    )
    face_bfield = FaceBFieldGeometry(
        x=BFieldGeometry(
            B_contra=face_b_contra_x,
            Bmag=jnp.ones((nx + 1, ny, nz), dtype=jnp.float64),
        ),
        y=BFieldGeometry(
            B_contra=face_b_contra_y,
            Bmag=jnp.ones((nx, ny + 1, nz), dtype=jnp.float64),
        ),
        z=BFieldGeometry(
            B_contra=face_b_contra_z,
            Bmag=jnp.ones((nx, ny, nz + 1), dtype=jnp.float64),
        ),
    )
    maps = FciMaps3D(
        forward_x=jnp.zeros(target_shape, dtype=jnp.float64),
        forward_y=jnp.zeros(target_shape, dtype=jnp.float64),
        backward_x=jnp.zeros(target_shape, dtype=jnp.float64),
        backward_y=jnp.zeros(target_shape, dtype=jnp.float64),
        forward_endpoint_x=jnp.zeros(target_shape, dtype=jnp.float64),
        forward_endpoint_y=jnp.zeros(target_shape, dtype=jnp.float64),
        forward_endpoint_z=jnp.zeros(target_shape, dtype=jnp.float64),
        backward_endpoint_x=jnp.zeros(target_shape, dtype=jnp.float64),
        backward_endpoint_y=jnp.zeros(target_shape, dtype=jnp.float64),
        backward_endpoint_z=jnp.zeros(target_shape, dtype=jnp.float64),
        forward_length=jnp.ones(target_shape, dtype=jnp.float64),
        backward_length=jnp.ones(target_shape, dtype=jnp.float64),
        forward_boundary=jnp.zeros(target_shape, dtype=bool),
        backward_boundary=jnp.zeros(target_shape, dtype=bool),
    )
    spacing = Spacing3D(
        dx=jnp.broadcast_to(grid.x.widths[:, None, None], target_shape),
        dy=jnp.broadcast_to(grid.y.widths[None, :, None], target_shape),
        dz=jnp.broadcast_to(grid.z.widths[None, None, :], target_shape),
    )
    return FciGeometry3D(
        grid=grid,
        maps=maps,
        spacing=spacing,
        cell_metric=cell_metric,
        face_metric=face_metric,
        cell_bfield=cell_bfield,
        face_bfield=face_bfield,
    )

def _mms_coordinates(geometry: FciGeometry3D) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    logical_grid = logical_grid_from_axis_vectors(*geometry.grid.logical_axis_vectors)
    return logical_grid[..., 0], logical_grid[..., 1], logical_grid[..., 2]


def _mms_local_coordinates(
    geometry: LocalFciGeometry3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return jnp.meshgrid(
        geometry.grid.x.centers_halo,
        geometry.grid.y.centers_halo,
        geometry.grid.z.centers_halo,
        indexing="ij",
    )


def _mms_background_density(geometry: FciGeometry3D) -> jnp.ndarray:
    x, _, _ = _mms_coordinates(geometry)
    return _mms_background_density_from_coordinates(x)


def _mms_phi(geometry: FciGeometry3D, time: float | jax.Array) -> jnp.ndarray:
    x, y, z = _mms_coordinates(geometry)
    return _mms_phi_from_coordinates(x, y, z, time)


def _mms_density(geometry: FciGeometry3D, time: float | jax.Array) -> jnp.ndarray:
    x, y, z = _mms_coordinates(geometry)
    return _mms_density_from_coordinates(x, y, z, time)


def _mms_v_parallel(geometry: FciGeometry3D, time: float | jax.Array) -> jnp.ndarray:
    x, y, z = _mms_coordinates(geometry)
    return _mms_v_parallel_from_coordinates(x, y, z, time)


def _mms_local_background_density_from_coordinates(x: jnp.ndarray) -> jnp.ndarray:
    return _mms_background_density_from_coordinates(x)


def _mms_local_phi_from_coordinates(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    time: float | jax.Array,
) -> jnp.ndarray:
    return _mms_phi_from_coordinates(x, y, z, time)


def _mms_local_density_from_coordinates(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    time: float | jax.Array,
) -> jnp.ndarray:
    return _mms_density_from_coordinates(x, y, z, time)


def _mms_local_v_parallel_from_coordinates(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    time: float | jax.Array,
) -> jnp.ndarray:
    return _mms_v_parallel_from_coordinates(x, y, z, time)


def _mms_local_background_density(geometry: LocalFciGeometry3D) -> jnp.ndarray:
    x, _, _ = _mms_local_coordinates(geometry)
    return _mms_local_background_density_from_coordinates(x)


def _mms_local_phi(
    geometry: LocalFciGeometry3D,
    time: float | jax.Array,
) -> jnp.ndarray:
    x, y, z = _mms_local_coordinates(geometry)
    return _mms_local_phi_from_coordinates(x, y, z, time)


def _mms_local_density(
    geometry: LocalFciGeometry3D,
    time: float | jax.Array,
) -> jnp.ndarray:
    x, y, z = _mms_local_coordinates(geometry)
    return _mms_local_density_from_coordinates(x, y, z, time)


def _mms_local_v_parallel(
    geometry: LocalFciGeometry3D,
    time: float | jax.Array,
) -> jnp.ndarray:
    x, y, z = _mms_local_coordinates(geometry)
    return _mms_local_v_parallel_from_coordinates(x, y, z, time)


def _mms_density_source(
    geometry: FciGeometry3D,
    time: float | jax.Array,
    *,
    rho_star_value: float,
) -> jnp.ndarray:
    x, y, z = _mms_coordinates(geometry)
    sx = jnp.sin(jnp.pi * x)
    cx = jnp.cos(jnp.pi * x)
    sy = jnp.sin(2.0 * jnp.pi * y)
    cy = jnp.cos(2.0 * jnp.pi * y)
    sz = jnp.sin(jnp.pi * z)
    st = jnp.sin(omega * time)
    ct = jnp.cos(omega * time)
    density_background = _mms_background_density_from_coordinates(x)
    density = _mms_density_from_coordinates(x, y, z, time)
    return density * (
        -A * omega * sx * cy * sz * st
        - (2.0 * alpha * (jnp.pi**2) * A / (rho_star_value * B0 * density_background))
        * sx**2
        * sy
        * sz
        * ct
        - jnp.pi * B * cx * sy * sz * st
    )


def _mms_local_density_source(
    geometry: LocalFciGeometry3D,
    time: float | jax.Array,
    *,
    rho_star_value: float,
) -> jnp.ndarray:
    x, y, z = _mms_local_coordinates(geometry)
    return _mms_local_density_source_from_coordinates(
        x,
        y,
        z,
        time,
        rho_star_value=rho_star_value,
    )


def _mms_local_density_source_from_coordinates(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    time: float | jax.Array,
    *,
    rho_star_value: float | jax.Array,
) -> jnp.ndarray:
    sx = jnp.sin(jnp.pi * x)
    cx = jnp.cos(jnp.pi * x)
    sy = jnp.sin(2.0 * jnp.pi * y)
    cy = jnp.cos(2.0 * jnp.pi * y)
    sz = jnp.sin(jnp.pi * z)
    st = jnp.sin(omega * time)
    ct = jnp.cos(omega * time)
    density_background = _mms_background_density_from_coordinates(x)
    density = _mms_density_from_coordinates(x, y, z, time)
    return density * (
        -A * omega * sx * cy * sz * st
        - (2.0 * alpha * (jnp.pi**2) * A / (rho_star_value * B0 * density_background))
        * sx**2
        * sy
        * sz
        * ct
        - jnp.pi * B * cx * sy * sz * st
    )


def _mms_v_parallel_source(
    geometry: FciGeometry3D,
    time: float | jax.Array,
    *,
    rho_star_value: float,
) -> jnp.ndarray:
    x, y, z = _mms_coordinates(geometry)
    sx = jnp.sin(jnp.pi * x)
    cx = jnp.cos(jnp.pi * x)
    sy = jnp.sin(2.0 * jnp.pi * y)
    cy = jnp.cos(2.0 * jnp.pi * y)
    sz = jnp.sin(jnp.pi * z)
    st = jnp.sin(omega * time)
    ct = jnp.cos(omega * time)
    return B * omega * cx * sy * jnp.cos(jnp.pi * z) * ct + (
        2.0 * (jnp.pi**2) * A * B / (rho_star_value * B0)
    ) * sz * jnp.cos(jnp.pi * z) * ct * st * (cx**2 * cy**2 - sx**2 * sy**2)


def _mms_local_v_parallel_source(
    geometry: LocalFciGeometry3D,
    time: float | jax.Array,
    *,
    rho_star_value: float,
) -> jnp.ndarray:
    x, y, z = _mms_local_coordinates(geometry)
    return _mms_local_v_parallel_source_from_coordinates(
        x,
        y,
        z,
        time,
        rho_star_value=rho_star_value,
    )


def _mms_local_v_parallel_source_from_coordinates(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    time: float | jax.Array,
    *,
    rho_star_value: float | jax.Array,
) -> jnp.ndarray:
    sx = jnp.sin(jnp.pi * x)
    cx = jnp.cos(jnp.pi * x)
    sy = jnp.sin(2.0 * jnp.pi * y)
    cy = jnp.cos(2.0 * jnp.pi * y)
    sz = jnp.sin(jnp.pi * z)
    st = jnp.sin(omega * time)
    ct = jnp.cos(omega * time)
    return B * omega * cx * sy * jnp.cos(jnp.pi * z) * ct + (
        2.0 * (jnp.pi**2) * A * B / (rho_star_value * B0)
    ) * sz * jnp.cos(jnp.pi * z) * ct * st * (cx**2 * cy**2 - sx**2 * sy**2)


def _mms_exact_state(geometry: FciGeometry3D, time: float | jax.Array) -> Fci2FieldState:
    return Fci2FieldState(
        density=_mms_density(geometry, time),
        v_parallel=_mms_v_parallel(geometry, time),
        density_background=_mms_background_density(geometry),
    )


def _mms_local_exact_state(
    geometry: LocalFciGeometry3D,
    time: float | jax.Array,
) -> Fci2FieldState:
    x, y, z = _mms_local_coordinates(geometry)
    return _mms_local_exact_state_from_coordinates(x, y, z, time)


def _mms_local_exact_state_from_coordinates(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    time: float | jax.Array,
) -> Fci2FieldState:
    return Fci2FieldState(
        density=_mms_local_density_from_coordinates(x, y, z, time),
        v_parallel=_mms_local_v_parallel_from_coordinates(x, y, z, time),
        density_background=_mms_local_background_density_from_coordinates(x),
    )
def _assert_mms_slab_geometry(geometry: FciGeometry3D) -> None:
    assert geometry.shape == geometry.grid.shape
    assert jnp.allclose(geometry.grid.x.centers[0], 0.0)
    assert jnp.allclose(geometry.grid.y.centers[0], 0.0)
    assert jnp.allclose(geometry.grid.z.centers[0], 0.0)
    assert jnp.allclose(geometry.grid.x.centers[-1], 1.0)
    assert jnp.allclose(geometry.grid.y.centers[-1], 1.0)
    assert jnp.allclose(geometry.grid.z.centers[-1], 1.0)
    assert jnp.allclose(geometry.cell_bfield.Bmag, B0)
    assert jnp.allclose(geometry.cell_bfield.B_contra[..., 0], 0.0)
    assert jnp.allclose(geometry.cell_bfield.B_contra[..., 1], 0.0)
    assert jnp.allclose(geometry.cell_bfield.B_contra[..., 2], 1.0)
    assert jnp.allclose(geometry.cell_metric.g11, 1.0)
    assert jnp.allclose(geometry.cell_metric.g22, 1.0)
    assert jnp.allclose(geometry.cell_metric.g33, 1.0)
    assert jnp.allclose(geometry.maps.forward_boundary, False)
    assert jnp.allclose(geometry.maps.backward_boundary, False)


def assert_shape_divisible_by_shards(
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
) -> None:
    """Require equal-sized local blocks for the slab domain-decomp harness."""

    if len(shape) != 3 or len(shard_counts) != 3:
        raise ValueError(f"shape and shard_counts must have length three: {shape}, {shard_counts}")
    for axis, (size, count) in enumerate(zip(shape, shard_counts)):
        if int(size) <= 0 or int(count) <= 0:
            raise ValueError(f"shape and shard_counts must be positive, got {shape}, {shard_counts}")
        if int(size) % int(count):
            raise ValueError(
                f"global shape axis {axis} with size {size} is not divisible by "
                f"shard count {count}; shape={shape}, shard_counts={shard_counts}"
            )


def make_mesh_for_shard_counts(
    shard_counts: tuple[int, int, int],
    *,
    axis_names: tuple[str, str, str] = _MESH_AXIS_NAMES,
) -> Mesh:
    """Build the execution mesh requested by a domain-decomposition case."""

    shard_counts = tuple(int(value) for value in shard_counts)
    if len(shard_counts) != 3 or any(value <= 0 for value in shard_counts):
        raise ValueError(f"shard_counts must contain three positive integers, got {shard_counts}")
    if len(axis_names) != 3 or len(set(axis_names)) != 3:
        raise ValueError(f"axis_names must contain three distinct names, got {axis_names}")

    ndevices = math.prod(shard_counts)
    devices = np.asarray(jax.devices()[:ndevices], dtype=object)
    if devices.size < ndevices:
        raise RuntimeError(
            f"shard_counts={shard_counts} requires {ndevices} devices, "
            f"but only {devices.size} are available"
        )
    return Mesh(devices.reshape(shard_counts), axis_names)


def _state_partition_spec() -> Fci2FieldState:
    spec = P("x", "y", "z")
    return Fci2FieldState(
        density=spec,
        v_parallel=spec,
        density_background=spec,
    )


def _put_scalar_field_on_mesh(field: jnp.ndarray, mesh: Mesh) -> jax.Array:
    field = jnp.asarray(field)
    if field.ndim != 3:
        raise ValueError(f"scalar owned fields must be three-dimensional, got {field.shape}")
    sharding = NamedSharding(mesh, P("x", "y", "z"))
    return jax.device_put(field, sharding)


def _put_state_on_mesh(state: Fci2FieldState, mesh: Mesh) -> Fci2FieldState:
    return Fci2FieldState(
        density=_put_scalar_field_on_mesh(state.density, mesh),
        v_parallel=_put_scalar_field_on_mesh(state.v_parallel, mesh),
        density_background=_put_scalar_field_on_mesh(state.density_background, mesh),
    )


def _gather_state_from_mesh(state: Fci2FieldState) -> Fci2FieldState:
    return Fci2FieldState(
        density=jnp.asarray(jax.device_get(state.density), dtype=jnp.float64),
        v_parallel=jnp.asarray(jax.device_get(state.v_parallel), dtype=jnp.float64),
        density_background=jnp.asarray(
            jax.device_get(state.density_background),
            dtype=jnp.float64,
        ),
    )


def _build_local_empty_maps(layout: HaloLayout3D) -> LocalFciMaps3D:
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


def build_local_slab_2field_geometry(
    shape: tuple[int, int, int],
    halo_width: int,
    *,
    global_shape: tuple[int, int, int],
    shard_index: tuple[object, object, object] = (0, 0, 0),
) -> LocalFciGeometry3D:
    """Build halo-padded local slab geometry for one logical shard."""

    nx, ny, nz = shape
    global_nx, global_ny, global_nz = global_shape
    layout = HaloLayout3D(shape, halo_width)

    def axis_grid(global_size: int, local_size: int, axis: int, shard_id: object) -> LocalGrid1D:
        if global_size < 2:
            raise ValueError(
                f"slab local geometry requires at least two cells on axis {axis}, got {global_size}"
            )
        spacing = 1.0 / float(global_size - 1)
        start = jnp.asarray(shard_id, dtype=jnp.int32) * local_size
        center_indices = start + jnp.arange(
            -halo_width,
            local_size + halo_width,
            dtype=jnp.float64,
        )
        face_indices = start + jnp.arange(
            -halo_width,
            local_size + halo_width + 1,
            dtype=jnp.float64,
        )
        return LocalGrid1D(
            layout=layout,
            axis=axis,
            centers_halo=center_indices * spacing,
            faces_halo=(face_indices - 0.5) * spacing,
            owned_start_global=0,
            owned_stop_global=local_size,
        )

    grid = LocalCellCenteredGrid3D(
        layout=layout,
        x=axis_grid(global_nx, nx, 0, shard_index[0]),
        y=axis_grid(global_ny, ny, 1, shard_index[1]),
        z=axis_grid(global_nz, nz, 2, shard_index[2]),
    )

    dx = 1.0 / float(global_nx - 1)
    dy = 1.0 / float(global_ny - 1)
    dz = 1.0 / float(global_nz - 1)
    spacing = LocalSpacing3D(
        layout=layout,
        dx_halo=jnp.full(layout.cell_halo_shape, dx, dtype=jnp.float64),
        dy_halo=jnp.full(layout.cell_halo_shape, dy, dtype=jnp.float64),
        dz_halo=jnp.full(layout.cell_halo_shape, dz, dtype=jnp.float64),
    )

    def _metric(shape_value: tuple[int, int, int], location: str) -> LocalMetricGeometry:
        ones = jnp.ones(shape_value, dtype=jnp.float64)
        zeros = jnp.zeros(shape_value, dtype=jnp.float64)
        return LocalMetricGeometry(
            layout=layout,
            J_halo=ones,
            g11_halo=ones,
            g22_halo=ones,
            g33_halo=ones,
            g12_halo=zeros,
            g13_halo=zeros,
            g23_halo=zeros,
            g_11_halo=ones,
            g_22_halo=ones,
            g_33_halo=ones,
            g_12_halo=zeros,
            g_13_halo=zeros,
            g_23_halo=zeros,
            location=location,
        )

    def _bfield(shape_value: tuple[int, int, int], location: str) -> LocalBFieldGeometry:
        zeros = jnp.zeros(shape_value, dtype=jnp.float64)
        ones = jnp.ones(shape_value, dtype=jnp.float64)
        b_contravariant = jnp.stack((zeros, zeros, ones), axis=-1)
        return LocalBFieldGeometry(
            layout=layout,
            B_contra_halo=b_contravariant,
            Bmag_halo=ones,
            location=location,
        )

    cell_metric = _metric(layout.cell_halo_shape, "cell")
    face_metric = LocalFaceMetricGeometry(
        layout=layout,
        x=_metric(layout.face_halo_shape(0), "x_face"),
        y=_metric(layout.face_halo_shape(1), "y_face"),
        z=_metric(layout.face_halo_shape(2), "z_face"),
    )
    cell_bfield = _bfield(layout.cell_halo_shape, "cell")
    face_bfield = LocalFaceBFieldGeometry(
        layout=layout,
        x=_bfield(layout.face_halo_shape(0), "x_face"),
        y=_bfield(layout.face_halo_shape(1), "y_face"),
        z=_bfield(layout.face_halo_shape(2), "z_face"),
    )
    regular_face_geometry = LocalRegularFaceGeometry3D(
        layout=layout,
        x_area=jnp.ones(layout.face_control_shape(0), dtype=jnp.float64),
        y_area=jnp.ones(layout.face_control_shape(1), dtype=jnp.float64),
        z_area=jnp.ones(layout.face_control_shape(2), dtype=jnp.float64),
        x_area_fraction=jnp.ones(layout.face_control_shape(0), dtype=jnp.float64),
        y_area_fraction=jnp.ones(layout.face_control_shape(1), dtype=jnp.float64),
        z_area_fraction=jnp.ones(layout.face_control_shape(2), dtype=jnp.float64),
        x_open_mask=jnp.ones(layout.face_control_shape(0), dtype=bool),
        y_open_mask=jnp.ones(layout.face_control_shape(1), dtype=bool),
        z_open_mask=jnp.ones(layout.face_control_shape(2), dtype=bool),
    )
    return LocalFciGeometry3D(
        layout=layout,
        grid=grid,
        maps=_build_local_empty_maps(layout),
        spacing=spacing,
        cell_metric=cell_metric,
        face_metric=face_metric,
        cell_bfield=cell_bfield,
        face_bfield=face_bfield,
        regular_face_geometry=regular_face_geometry,
        cell_volume_geometry=LocalCellVolumeGeometry3D(
            layout=layout,
            volume=jnp.ones(shape, dtype=jnp.float64),
            volume_fraction=jnp.ones(shape, dtype=jnp.float64),
        ),
    )


def _build_local_domain(
    global_shape: tuple[int, int, int],
    halo_width: int,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    mesh_axis_names: tuple[str | None, str | None, str | None] = _MESH_AXIS_NAMES,
) -> LocalDomain3D:
    assert_shape_divisible_by_shards(global_shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(global_shape, shard_counts)
    )
    layout = HaloLayout3D(owned_shape, halo_width)
    return LocalDomain3D(
        layout=layout,
        shard_spec=ShardSpec3D(
            global_shape=global_shape,
            owned_start=(0, 0, 0),
            owned_stop=owned_shape,
            shard_index=(0, 0, 0),
            shard_counts=tuple(int(value) for value in shard_counts),
            periodic_axes=(False, False, False),
            halo_width=halo_width,
            side_kind_lower=(SIDE_PHYSICAL, SIDE_PHYSICAL, SIDE_PHYSICAL),
            side_kind_upper=(SIDE_PHYSICAL, SIDE_PHYSICAL, SIDE_PHYSICAL),
        ),
        mesh_axis_names=mesh_axis_names,
    )


def _build_ghost_filler(halo_width: int) -> PhysicalGhostCellFiller3D:
    weights = GhostFillWeights1D(
        owned_weights=jnp.full((halo_width, 1), -1.0, dtype=jnp.float64),
        bc_weights=jnp.full((halo_width,), 2.0, dtype=jnp.float64),
    )
    neutral = GhostFillWeights1D(
        owned_weights=jnp.ones((halo_width, 1), dtype=jnp.float64),
        bc_weights=jnp.zeros((halo_width,), dtype=jnp.float64),
    )
    return PhysicalGhostCellFiller3D(
        dirichlet=(weights, weights, weights),
        neumann_lower=(neutral, neutral, neutral),
        neumann_upper=(neutral, neutral, neutral),
    )


def _apply_local_owned_dirichlet_to_field(
    field_owned: jnp.ndarray,
    exact_owned: jnp.ndarray,
    domain: LocalDomain3D,
) -> jnp.ndarray:
    field_owned = jnp.asarray(field_owned, dtype=jnp.float64)
    exact_owned = jnp.asarray(exact_owned, dtype=jnp.float64)
    result = field_owned
    result = result.at[0, :, :].set(
        jnp.where(domain.runtime_has_physical_lower(0), exact_owned[0, :, :], result[0, :, :])
    )
    result = result.at[-1, :, :].set(
        jnp.where(domain.runtime_has_physical_upper(0), exact_owned[-1, :, :], result[-1, :, :])
    )
    result = result.at[:, 0, :].set(
        jnp.where(domain.runtime_has_physical_lower(1), exact_owned[:, 0, :], result[:, 0, :])
    )
    result = result.at[:, -1, :].set(
        jnp.where(domain.runtime_has_physical_upper(1), exact_owned[:, -1, :], result[:, -1, :])
    )
    result = result.at[:, :, 0].set(
        jnp.where(domain.runtime_has_physical_lower(2), exact_owned[:, :, 0], result[:, :, 0])
    )
    result = result.at[:, :, -1].set(
        jnp.where(domain.runtime_has_physical_upper(2), exact_owned[:, :, -1], result[:, :, -1])
    )
    return result


def _apply_local_owned_dirichlet_to_state(
    state_owned: Fci2FieldState,
    exact_owned: Fci2FieldState,
    domain: LocalDomain3D,
) -> Fci2FieldState:
    return Fci2FieldState(
        density=_apply_local_owned_dirichlet_to_field(
            state_owned.density,
            exact_owned.density,
            domain,
        ),
        v_parallel=_apply_local_owned_dirichlet_to_field(
            state_owned.v_parallel,
            exact_owned.v_parallel,
            domain,
        ),
        density_background=_apply_local_owned_dirichlet_to_field(
            state_owned.density_background,
            exact_owned.density_background,
            domain,
        ),
    )


def _build_local_dirichlet_face_bc(
    values_halo: jnp.ndarray,
    domain: LocalDomain3D,
) -> LocalBoundaryFaceBC3D:
    layout = domain.layout
    h = layout.halo_width
    nx, ny, nz = layout.owned_shape
    zeros_x = jnp.zeros(layout.face_control_shape(0), dtype=jnp.float64)
    zeros_y = jnp.zeros(layout.face_control_shape(1), dtype=jnp.float64)
    zeros_z = jnp.zeros(layout.face_control_shape(2), dtype=jnp.float64)
    mask_x = jnp.zeros(layout.face_control_shape(0), dtype=bool)
    mask_y = jnp.zeros(layout.face_control_shape(1), dtype=bool)
    mask_z = jnp.zeros(layout.face_control_shape(2), dtype=bool)
    kind_x = jnp.zeros(layout.face_control_shape(0), dtype=jnp.int32)
    kind_y = jnp.zeros(layout.face_control_shape(1), dtype=jnp.int32)
    kind_z = jnp.zeros(layout.face_control_shape(2), dtype=jnp.int32)

    lower_x = values_halo[h, h : h + ny, h : h + nz]
    upper_x = values_halo[h + nx - 1, h : h + ny, h : h + nz]
    lower_y = values_halo[h : h + nx, h, h : h + nz]
    upper_y = values_halo[h : h + nx, h + ny - 1, h : h + nz]
    lower_z = values_halo[h : h + nx, h : h + ny, h]
    upper_z = values_halo[h : h + nx, h : h + ny, h + nz - 1]

    value_x = zeros_x.at[0].set(lower_x).at[-1].set(upper_x)
    value_y = zeros_y.at[:, 0, :].set(lower_y).at[:, -1, :].set(upper_y)
    value_z = zeros_z.at[:, :, 0].set(lower_z).at[:, :, -1].set(upper_z)

    mask_x = mask_x.at[0].set(domain.runtime_has_physical_lower(0)).at[-1].set(
        domain.runtime_has_physical_upper(0)
    )
    mask_y = mask_y.at[:, 0, :].set(domain.runtime_has_physical_lower(1)).at[:, -1, :].set(
        domain.runtime_has_physical_upper(1)
    )
    mask_z = mask_z.at[:, :, 0].set(domain.runtime_has_physical_lower(2)).at[:, :, -1].set(
        domain.runtime_has_physical_upper(2)
    )
    kind_x = kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET)
    kind_y = kind_y.at[:, 0, :].set(BC_DIRICHLET).at[:, -1, :].set(BC_DIRICHLET)
    kind_z = kind_z.at[:, :, 0].set(BC_DIRICHLET).at[:, :, -1].set(BC_DIRICHLET)
    return LocalBoundaryFaceBC3D(
        kind_x=kind_x,
        kind_y=kind_y,
        kind_z=kind_z,
        value_x=value_x,
        value_y=value_y,
        value_z=value_z,
        mask_x=mask_x,
        mask_y=mask_y,
        mask_z=mask_z,
        layout=layout,
    )


def _prepare_local_slab_stage_state(
    state_owned: Fci2FieldState,
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    domain: LocalDomain3D,
    *,
    halo_exchange: HaloExchange3D,
    physical_ghost_filler: PhysicalGhostCellFiller3D,
    stage_time: float | jax.Array,
) -> PreparedLocalState3D:
    exact_stage_halo = _mms_local_exact_state_from_coordinates(
        *coordinates_halo,
        stage_time,
    )
    exact_owned = Fci2FieldState(
        density=exact_stage_halo.density[domain.layout.owned_slices_cell],
        v_parallel=exact_stage_halo.v_parallel[domain.layout.owned_slices_cell],
        density_background=exact_stage_halo.density_background[domain.layout.owned_slices_cell],
    )
    state_bc_owned = _apply_local_owned_dirichlet_to_state(
        state_owned,
        exact_owned,
        domain,
    )
    state_halo = inject_owned_state_to_halo(state_bc_owned, domain.layout)
    state_halo = Fci2FieldState(
        density=halo_exchange(state_halo.density, domain),
        v_parallel=halo_exchange(state_halo.v_parallel, domain),
        density_background=halo_exchange(state_halo.density_background, domain),
    )
    face_bc_bundle = _Slab2FieldFaceBCBundle(
        density=_build_local_dirichlet_face_bc(exact_stage_halo.density, domain),
        v_parallel=_build_local_dirichlet_face_bc(exact_stage_halo.v_parallel, domain),
        density_background=_build_local_dirichlet_face_bc(
            exact_stage_halo.density_background,
            domain,
        ),
    )
    prepared_state_halo = Fci2FieldState(
        density=physical_ghost_filler(state_halo.density, domain, face_bc_bundle.density),
        v_parallel=physical_ghost_filler(
            state_halo.v_parallel,
            domain,
            face_bc_bundle.v_parallel,
        ),
        density_background=physical_ghost_filler(
            state_halo.density_background,
            domain,
            face_bc_bundle.density_background,
        ),
    )
    return PreparedLocalState3D(
        state_halo=prepared_state_halo,
        boundary_data=LocalBoundaryData3D(face_bc=face_bc_bundle),
    )


@dataclass(frozen=True)
class LocalSlab2FieldRhs:
    geometry: LocalFciGeometry3D
    domain: LocalDomain3D
    coordinates_halo: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    halo_exchange: HaloExchange3D
    physical_ghost_filler: PhysicalGhostCellFiller3D
    parameters: Fci2FieldRhsParameters
    curvature_coefficients_owned: jnp.ndarray
    timing_enabled: bool = False

    def __call__(
        self,
        state_owned: Fci2FieldState,
        stage_time: float | jax.Array,
        carry: None,
    ) -> tuple[Fci2FieldState, None, jnp.ndarray]:
        del carry
        prepared_stage = _prepare_local_slab_stage_state(
            state_owned,
            self.coordinates_halo,
            self.domain,
            halo_exchange=self.halo_exchange,
            physical_ghost_filler=self.physical_ghost_filler,
            stage_time=stage_time,
        )
        density_halo = jnp.asarray(prepared_stage.state_halo.density, dtype=jnp.float64)
        v_parallel_halo = jnp.asarray(
            prepared_stage.state_halo.v_parallel,
            dtype=jnp.float64,
        )
        density_background_halo = jnp.asarray(
            prepared_stage.state_halo.density_background,
            dtype=jnp.float64,
        )
        phi_halo = jnp.log(
            jnp.maximum(density_halo, 1.0e-30)
            / jnp.maximum(density_background_halo, 1.0e-30)
        )
        context = StencilBuilderContext(layout=self.domain.layout, domain=self.domain)
        density_stencil = build_local_stencil_from_field(
            density_halo,
            self.geometry,
            context,
        )
        phi_stencil = build_local_stencil_from_field(
            phi_halo,
            self.geometry,
            context,
        )
        v_parallel_stencil = build_local_stencil_from_field(
            v_parallel_halo,
            self.geometry,
            context,
        )

        density_owned = density_halo[self.domain.layout.owned_slices_cell]
        magnetic_field = jnp.maximum(
            jnp.asarray(self.geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            1.0e-30,
        )
        rho_star_value = jnp.asarray(self.parameters.rho_star, dtype=jnp.float64)
        poisson_density = local_poisson_bracket_op(
            phi_stencil,
            density_stencil,
            self.geometry,
        )
        curvature_density = local_curvature_op(
            density_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        curvature_phi = local_curvature_op(
            phi_stencil,
            self.geometry,
            curvature_coefficients=self.curvature_coefficients_owned,
        )
        parallel_velocity_gradient = local_grad_parallel_op_direct(
            v_parallel_stencil,
            self.geometry,
        )
        poisson_v_parallel = local_poisson_bracket_op(
            phi_stencil,
            v_parallel_stencil,
            self.geometry,
        )

        density_rhs = (
            -(poisson_density / (rho_star_value * magnetic_field))
            + (2.0 / magnetic_field) * curvature_density
            - (2.0 * density_owned / magnetic_field) * curvature_phi
            - density_owned * parallel_velocity_gradient
        )
        v_parallel_rhs = -(poisson_v_parallel / (rho_star_value * magnetic_field))
        density_rhs = density_rhs + _mms_local_density_source_from_coordinates(
            *self.coordinates_halo,
            stage_time,
            rho_star_value=rho_star_value,
        )[self.domain.layout.owned_slices_cell]
        v_parallel_rhs = v_parallel_rhs + _mms_local_v_parallel_source_from_coordinates(
            *self.coordinates_halo,
            stage_time,
            rho_star_value=rho_star_value,
        )[self.domain.layout.owned_slices_cell]

        rhs = Fci2FieldState(
            density=jnp.asarray(density_rhs, dtype=jnp.float64),
            v_parallel=jnp.asarray(v_parallel_rhs, dtype=jnp.float64),
            density_background=jnp.zeros(self.domain.layout.owned_shape, dtype=jnp.float64),
        )
        aux = jnp.zeros((3,), dtype=jnp.float64)
        if self.timing_enabled:
            aux = aux + 0.0
        return rhs, None, aux


def simulate_mms_2field_slab(
    geometry: FciGeometry3D,
    *,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    timestep: float | None = None,
    final_time: float = tf,
    rho_star_value: float = rho_star,
    show_progress: bool = False,
) -> tuple[Fci2FieldState, dict[str, float]]:
    """Advance the slab MMS using a shard-map local RHS with ``Rk4Stepper``."""

    _assert_mms_slab_geometry(geometry)
    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(geometry.shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(geometry.shape, shard_counts)
    )
    domain = _build_local_domain(geometry.shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)
    parameters = Fci2FieldRhsParameters(rho_star=rho_star_value)
    curvature_coefficients_owned = jnp.zeros(owned_shape + (3,), dtype=jnp.float64)
    dt = float(final_time) / float(num_steps) if timestep is None else float(timestep)
    steps = int(round(float(final_time) / dt))
    dt = float(final_time) / float(steps)
    initial_state = _mms_exact_state(geometry, 0.0)
    total_runtime = 0.0
    wall_step_times: list[float] = []
    prebuilt_local_geometry = None
    prebuilt_coordinates_halo = None
    if shard_counts == (1, 1, 1):
        prebuilt_local_geometry = build_local_slab_2field_geometry(
            owned_shape,
            halo_width,
            global_shape=geometry.shape,
            shard_index=(0, 0, 0),
        )
        prebuilt_coordinates_halo = _mms_local_coordinates(prebuilt_local_geometry)

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        state = _put_state_on_mesh(initial_state, mesh)
        state_spec = _state_partition_spec()

        def kernel(
            state_owned: Fci2FieldState,
            step_time: jax.Array,
            step_timestep: jax.Array,
        ) -> Fci2FieldState:
            if prebuilt_local_geometry is None:
                shard_index = tuple(lax.axis_index(name) for name in _MESH_AXIS_NAMES)
                local_geometry = build_local_slab_2field_geometry(
                    owned_shape,
                    halo_width,
                    global_shape=geometry.shape,
                    shard_index=shard_index,
                )
                coordinates_halo = _mms_local_coordinates(local_geometry)
            else:
                local_geometry = prebuilt_local_geometry
                coordinates_halo = prebuilt_coordinates_halo
            rhs = LocalSlab2FieldRhs(
                geometry=local_geometry,
                domain=domain,
                coordinates_halo=coordinates_halo,
                halo_exchange=HaloExchange3D(),
                physical_ghost_filler=ghost_filler,
                parameters=parameters,
                curvature_coefficients_owned=curvature_coefficients_owned,
            )
            step_result = Rk4Stepper(rhs)(
                state_owned,
                time=step_time,
                timestep=step_timestep,
                carry=None,
            )
            next_exact_halo = _mms_local_exact_state_from_coordinates(
                *coordinates_halo,
                step_time + step_timestep,
            )
            next_exact_owned = Fci2FieldState(
                density=next_exact_halo.density[domain.layout.owned_slices_cell],
                v_parallel=next_exact_halo.v_parallel[domain.layout.owned_slices_cell],
                density_background=next_exact_halo.density_background[
                    domain.layout.owned_slices_cell
                ],
            )
            next_state = _apply_local_owned_dirichlet_to_state(
                step_result.state,
                next_exact_owned,
                domain,
            )
            return next_state

        mapped_step_kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(state_spec, P(), P()),
            out_specs=state_spec,
            check_rep=False,
        )
        step_kernel = jax.jit(mapped_step_kernel)

        time_value = 0.0
        progress_start = time_module.perf_counter()
        if show_progress:
            print(
                f"slab_2field RK4 progress: {_format_progress_bar(0, steps, start_time=progress_start)}",
                end="",
                flush=True,
            )

        for step_index in range(steps):
            step_start = time_module.perf_counter()
            state = step_kernel(
                state,
                jnp.asarray(time_value, dtype=jnp.float64),
                jnp.asarray(dt, dtype=jnp.float64),
            )
            jax.block_until_ready(state.density)
            elapsed = time_module.perf_counter() - step_start
            total_runtime += elapsed
            wall_step_times.append(elapsed)
            time_value += dt
            if show_progress:
                print(
                    "\r"
                    f"slab_2field RK4 progress: "
                    f"{_format_progress_bar(step_index + 1, steps, start_time=progress_start)}",
                    end="",
                    flush=True,
                )

        if show_progress:
            print()

        final_state = _gather_state_from_mesh(state)

    mean_step_runtime = total_runtime / float(steps) if steps else 0.0
    timing_summary = {
        "total_runtime": float(total_runtime),
        "avg_step_runtime": float(mean_step_runtime),
        "prep_time": 0.0,
        "stencil_time": 0.0,
        "operator_time": 0.0,
    }
    if wall_step_times:
        print(
            "slab_2field mean timings per RK step: "
            f"wall={np.mean(np.asarray(wall_step_times, dtype=np.float64)):.6e} s, "
            f"prep={timing_summary['prep_time'] / float(steps):.6e} s, "
            f"stencil={timing_summary['stencil_time'] / float(steps):.6e} s, "
            f"operator={timing_summary['operator_time'] / float(steps):.6e} s"
        )
    return final_state, timing_summary


def _combined_error_statistics(
    final_state: Fci2FieldState,
    geometry: FciGeometry3D,
    time: float,
) -> tuple[float, float, float]:
    exact = _mms_exact_state(geometry, time)
    error = jnp.concatenate(
        [
            jnp.ravel(jnp.abs(final_state.density - exact.density)),
            jnp.ravel(jnp.abs(final_state.v_parallel - exact.v_parallel)),
        ]
    )
    return (
        float(jnp.sqrt(jnp.mean(error**2))),
        float(jnp.median(error)),
        float(jnp.max(error)),
    )


def _estimate_convergence_order(
    resolutions: list[int],
    errors: list[float],
) -> float | None:
    if len(resolutions) < 2:
        return None
    log_resolutions = np.log(np.asarray(resolutions, dtype=np.float64))
    log_errors = np.log(np.asarray(errors, dtype=np.float64))
    slope, _intercept = np.polyfit(log_resolutions, log_errors, 1)
    return float(-slope)


def _save_convergence_plot(
    resolutions: list[int],
    l2_errors: list[float],
    linf_errors: list[float],
    *,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    plotted_resolutions = np.asarray(resolutions, dtype=np.int64)
    log_resolutions = np.log(plotted_resolutions.astype(np.float64))
    l2_log_errors = np.log(np.asarray(l2_errors, dtype=np.float64))
    linf_log_errors = np.log(np.asarray(linf_errors, dtype=np.float64))
    l2_slope, l2_intercept = np.polyfit(log_resolutions, l2_log_errors, 1)
    linf_slope, linf_intercept = np.polyfit(log_resolutions, linf_log_errors, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.loglog(plotted_resolutions, l2_errors, "o-", label=f"L2, order {-l2_slope:.2f}")
    ax.loglog(plotted_resolutions, linf_errors, "^-", label=f"Linf, order {-linf_slope:.2f}")
    ax.loglog(
        plotted_resolutions,
        np.exp(l2_intercept) * plotted_resolutions.astype(np.float64) ** l2_slope,
        "--",
        color=ax.lines[0].get_color(),
    )
    ax.loglog(
        plotted_resolutions,
        np.exp(linf_intercept) * plotted_resolutions.astype(np.float64) ** linf_slope,
        "--",
        color=ax.lines[1].get_color(),
    )
    ax.set_xlabel("resolution")
    ax.set_ylabel("absolute error")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_slab_2field_convergence(
    *,
    resolutions: list[int],
    shard_counts: tuple[int, int, int],
    halo_width: int,
    final_time: float = tf,
    base_steps: int = num_steps,
    plot: bool = False,
    plot_path: str | None = None,
    show_progress: bool = False,
) -> dict[str, object]:
    successful_resolutions: list[int] = []
    l2_errors: list[float] = []
    linf_errors: list[float] = []

    for resolution in resolutions:
        shape = (resolution, resolution, resolution)
        assert_shape_divisible_by_shards(shape, shard_counts)
        geometry = build_slab_2field_geometry(*shape)
        steps = _resolution_step_count(resolution, base_steps=base_steps)
        dt = float(final_time) / float(steps)
        try:
            final_state, timing_summary = simulate_mms_2field_slab(
                geometry,
                shard_counts=shard_counts,
                halo_width=halo_width,
                final_time=final_time,
                timestep=dt,
                rho_star_value=rho_star,
                show_progress=show_progress,
            )
            l2_error, median_error, linf_error = _combined_error_statistics(
                final_state,
                geometry,
                final_time,
            )
        except FloatingPointError as exc:
            print(
                f"WARNING: N={resolution} shard_counts={shard_counts} "
                f"failed with non-finite values: {exc}"
            )
            continue

        successful_resolutions.append(resolution)
        l2_errors.append(l2_error)
        linf_errors.append(linf_error)
        print(
            f"N={resolution}: shard_counts={shard_counts}, "
            f"steps={steps}, total_runtime={timing_summary['total_runtime']:.6e} s, "
            f"avg_step_runtime={timing_summary['avg_step_runtime']:.6e} s, "
            f"L2={l2_error:.6e}, median={median_error:.6e}, Linf={linf_error:.6e}"
        )

    l2_order = _estimate_convergence_order(successful_resolutions, l2_errors)
    linf_order = _estimate_convergence_order(successful_resolutions, linf_errors)
    if l2_order is not None:
        print(f"slab_2field L2 convergence order: {l2_order:.6f}")
    if linf_order is not None:
        print(f"slab_2field Linf convergence order: {linf_order:.6f}")

    if plot and successful_resolutions:
        _save_convergence_plot(
            successful_resolutions,
            l2_errors,
            linf_errors,
            title=f"2-field slab MMS convergence ({shard_counts})",
            output_path=Path(plot_path or "slab_2field_convergence.png"),
        )

    return {
        "resolutions": successful_resolutions,
        "l2_errors": l2_errors,
        "linf_errors": linf_errors,
        "l2_order": l2_order,
        "linf_order": linf_order,
    }


def test_build_local_slab_geometry_matches_layout_shapes() -> None:
    geometry = build_local_slab_2field_geometry(
        (4, 5, 6),
        2,
        global_shape=(4, 5, 6),
    )
    layout = geometry.layout
    assert geometry.cell_metric.J_halo.shape == layout.cell_halo_shape
    assert geometry.face_metric.x.J_halo.shape == layout.face_halo_shape(0)
    assert geometry.face_metric.y.J_halo.shape == layout.face_halo_shape(1)
    assert geometry.face_metric.z.J_halo.shape == layout.face_halo_shape(2)
    assert geometry.cell_bfield.Bmag_halo.shape == layout.cell_halo_shape
    assert geometry.regular_face_geometry.x_area.shape == layout.face_control_shape(0)


def test_assert_shape_divisible_by_shards_rejects_uneven_partition() -> None:
    import pytest

    with pytest.raises(ValueError, match="not divisible"):
        assert_shape_divisible_by_shards((10, 8, 8), (3, 1, 1))


def test_local_slab_rhs_returns_owned_fields_only() -> None:
    shape = (4, 4, 4)
    halo_width = 2
    global_geometry = build_slab_2field_geometry(*shape)
    local_geometry = build_local_slab_2field_geometry(
        shape,
        halo_width,
        global_shape=shape,
    )
    domain = _build_local_domain(
        shape,
        halo_width,
        (1, 1, 1),
        mesh_axis_names=(None, None, None),
    )
    rhs = LocalSlab2FieldRhs(
        geometry=local_geometry,
        domain=domain,
        coordinates_halo=_mms_local_coordinates(local_geometry),
        halo_exchange=HaloExchange3D(),
        physical_ghost_filler=_build_ghost_filler(halo_width),
        parameters=Fci2FieldRhsParameters(rho_star=rho_star),
        curvature_coefficients_owned=jnp.zeros(shape + (3,), dtype=jnp.float64),
    )
    initial_state = _mms_exact_state(global_geometry, 0.0)
    rhs_state, carry, aux = rhs(initial_state, 0.0, None)
    assert carry is None
    rhs_state.assert_field_shape(shape)
    assert aux.shape == (3,)


def main() -> None:
    parser = argparse.ArgumentParser(description="Slab 2-field MMS convergence harness")
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[40, 60, 120],
    )
    parser.add_argument(
        "--shard-counts",
        nargs=3,
        type=int,
        metavar=("PX", "PY", "PZ"),
        default=(1, 1, 1),
    )
    parser.add_argument("--halo-width", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=tf)
    parser.add_argument("--base-steps", type=int, default=num_steps)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    resolutions = [int(value) for value in args.resolutions]
    shard_counts = tuple(int(value) for value in args.shard_counts)
    run_slab_2field_convergence(
        resolutions=resolutions,
        shard_counts=shard_counts,
        halo_width=int(args.halo_width),
        final_time=float(args.final_time),
        base_steps=int(args.base_steps),
        plot=bool(args.plot),
        plot_path=args.plot_path,
        show_progress=bool(args.show_progress),
    )


if __name__ == "__main__":
    main()
