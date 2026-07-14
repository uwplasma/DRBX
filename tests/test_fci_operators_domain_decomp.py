"""Shard-map tests for local FCI operators using the shifted-torus MMS."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys
import time

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax_drb.geometry import (
    BFieldGeometry,
    CellCenteredGrid3D,
    FCI_DEP_CUT_WALL,
    FCI_DEP_FIELD_INTERIOR,
    FCI_DEP_INVALID,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    FciGeometry3D,
    FciMaps3D,
    HaloLayout3D,
    Grid1D,
    LocalBFieldGeometry,
    LocalCellCenteredGrid3D,
    LocalCellVolumeGeometry3D,
    LocalCoordinateStencilDependencyMap3D,
    LocalCoordinateStencilLocalDependencyTable,
    LocalCoordinateStencilRemoteDependencyTable,
    LocalDomain3D,
    LocalFciDirectionMap,
    LocalFciGeometry3D,
    LocalFciLocalDependencyTable,
    LocalFciMaps3D,
    LocalFciRemoteDependencyTable,
    LocalFaceBFieldGeometry,
    LocalFaceMetricGeometry,
    LocalGrid1D,
    LocalMetricGeometry,
    LocalRegularFaceGeometry3D,
    LocalSpacing3D,
    LocalStencilBuilder,
    MetricGeometry,
    Spacing3D,
    StencilBuilderContext,
    SIDE_PHYSICAL,
    SIDE_SIMPLE_PERIODIC,
    ShardSpec3D,
    build_curvature_coefficients,
    build_fci_maps_from_b_contravariant,
    build_local_curvature_coefficients,
    build_local_conservative_stencil_from_field,
    build_local_direct_stencil_one_sided_physical_from_halo,
    build_local_fci_stencil_from_field,
)
from jax_drb.native.fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    LocalBoundaryConditionBuilder,
    LocalBoundaryData3D,
    LocalBoundaryFaceBC3D,
    LocalBoundaryPreparation3D,
)
from jax_drb.native.fci_halo import (
    GhostFillWeights1D,
    HaloExchange3D,
    LocalStateAndBoundaryPreparer3D,
    LocalPeriodicTopologyRule3D,
    PhysicalGhostCellFiller3D,
    RemoteFciDependencyExchange,
    TopologyHaloFiller3D,
)
from jax_drb.native.fci_model import FciFieldBundle, FciModelState
from jax_drb.native.fci_operators import (
    local_curvature_op,
    local_grad_parallel_op_direct,
    local_grad_parallel_op_fci,
    local_grad_perp_op_direct,
    local_parallel_laplacian_conservative_op,
    local_parallel_laplacian_direct_op,
    local_perp_laplacian_conservative_op,
    local_perp_laplacian_local_op,
    local_poisson_bracket_op,
)
from jax_drb.geometry import build_local_stencil_from_field


IOTA = 1.1
A = 0.2
M = 1
N = 1
R0 = 3.0
ALPHA = 0.25
C_PHI = 3.0
RHO_MIN = 0.2


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ScalarFieldState(FciModelState):
    field: jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ScalarFaceBCBundle(FciFieldBundle):
    field: LocalBoundaryFaceBC3D


def make_mesh_for_shard_counts(
    shard_counts: tuple[int, int, int],
    *,
    axis_names: tuple[str, str, str] = ("x", "y", "z"),
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


def put_scalar_field_on_mesh(field: jnp.ndarray, mesh: Mesh) -> jax.Array:
    """Place a global scalar field with the standard three-axis sharding."""

    field = jnp.asarray(field)
    if field.ndim != 3:
        raise ValueError(f"scalar owned fields must be three-dimensional, got {field.shape}")
    sharding = NamedSharding(mesh, P("x", "y", "z"))
    return jax.device_put(field, sharding)


def assert_shape_divisible_by_shards(
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
) -> None:
    """Require equal-sized local blocks for the initial SPMD test harness."""

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


def _periodic_halo_coordinates(values: jnp.ndarray, period: float, halo_width: int) -> jnp.ndarray:
    """Extend a periodic one-dimensional coordinate vector by ``halo_width``."""

    lower = values[-halo_width:] - period
    upper = values[:halo_width] + period
    return jnp.concatenate((lower, values, upper))


def _physical_halo_coordinates(values: jnp.ndarray, halo_width: int) -> jnp.ndarray:
    """Extend a uniformly spaced physical coordinate vector at both ends."""

    spacing = values[1] - values[0]
    lower = values[0] - spacing * jnp.arange(halo_width, 0, -1, dtype=values.dtype)
    upper = values[-1] + spacing * jnp.arange(1, halo_width + 1, dtype=values.dtype)
    return jnp.concatenate((lower, values, upper))


def _shifted_torus_metric(
    layout: HaloLayout3D,
    location: str,
    rho: jnp.ndarray,
    theta: jnp.ndarray,
) -> LocalMetricGeometry:
    """Evaluate shifted-torus covariant/contravariant metric components."""

    rho, theta = jnp.broadcast_arrays(
        jnp.asarray(rho, dtype=jnp.float64),
        jnp.asarray(theta, dtype=jnp.float64),
    )
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = R0 + ALPHA * rho + rho * cos_theta
    Q = 1.0 + ALPHA * cos_theta

    zeros = jnp.zeros_like(R)
    return LocalMetricGeometry(
        layout=layout,
        J_halo=R * rho * Q,
        g11_halo=1.0 / Q**2,
        g22_halo=(1.0 + 2.0 * ALPHA * cos_theta + ALPHA**2) / (rho**2 * Q**2),
        g33_halo=1.0 / R**2,
        g12_halo=ALPHA * sin_theta / (rho * Q**2),
        g13_halo=zeros,
        g23_halo=zeros,
        g_11_halo=1.0 + 2.0 * ALPHA * cos_theta + ALPHA**2,
        g_22_halo=rho**2,
        g_33_halo=R**2,
        g_12_halo=-ALPHA * rho * sin_theta,
        g_13_halo=zeros,
        g_23_halo=zeros,
        location=location,
    )


def _shifted_torus_bfield(
    metric: LocalMetricGeometry,
    rho: jnp.ndarray,
) -> LocalBFieldGeometry:
    """Evaluate the shifted-torus contravariant magnetic field."""

    J = metric.J_halo
    B_contra = jnp.stack(
        (
            jnp.zeros_like(J),
            IOTA * C_PHI / J,
            C_PHI / J,
        ),
        axis=-1,
    )
    g_cov = jnp.stack(
        (
            jnp.stack((metric.g_11_halo, metric.g_12_halo, metric.g_13_halo), axis=-1),
            jnp.stack((metric.g_12_halo, metric.g_22_halo, metric.g_23_halo), axis=-1),
            jnp.stack((metric.g_13_halo, metric.g_23_halo, metric.g_33_halo), axis=-1),
        ),
        axis=-2,
    )
    bmag = jnp.sqrt(jnp.einsum("...i,...ij,...j->...", B_contra, g_cov, B_contra))
    return LocalBFieldGeometry(
        layout=metric.layout,
        B_contra_halo=B_contra,
        Bmag_halo=bmag,
        location=metric.location,
    )


def _empty_maps(layout: HaloLayout3D) -> LocalFciMaps3D:
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


def _build_local_fci_maps(
    layout: HaloLayout3D,
    *,
    global_shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
    shard_index: tuple[object, object, object],
    rho_owned: jnp.ndarray,
    theta_owned: jnp.ndarray,
) -> LocalFciMaps3D:
    """Build a fixed-width, local-halo-only FCI map for the torus MMS.

    The shifted-torus field line satisfies ``dtheta/dphi = IOTA``.  We trace
    one toroidal cell in each direction, interpolate linearly in theta, and
    use the exchanged/topology-filled z halo for the neighboring toroidal
    plane.  Every dependency therefore fits in the regular halo for the
    initial test case; no remote FCI gather is involved.
    """

    nx, ny, nz = layout.owned_shape
    global_nx, global_ny, global_nz = global_shape
    del global_nx
    h = layout.halo_width

    shard_y = jnp.asarray(shard_index[1], dtype=jnp.int32)
    shard_z = jnp.asarray(shard_index[2], dtype=jnp.int32)
    local_ny = global_ny // int(shard_counts[1])
    local_nz = global_nz // int(shard_counts[2])

    ii, jj, kk = jnp.meshgrid(
        jnp.arange(nx, dtype=jnp.int32),
        jnp.arange(ny, dtype=jnp.int32),
        jnp.arange(nz, dtype=jnp.int32),
        indexing="ij",
    )
    target_flat = jnp.arange(nx * ny * nz, dtype=jnp.int32)
    target_flat = jnp.repeat(target_flat, 2)

    def _local_periodic_index(global_index, shard_id, global_size, local_size):
        # Preserve negative indices on the lower global side so they map into
        # the lower halo. Only upper-wrap indices need to be folded back.
        wrapped = jnp.where(
            global_index >= global_size,
            global_index - global_size,
            global_index,
        )
        start = shard_id * local_size
        delta = wrapped - start
        # Select the periodic image adjacent to this shard. This keeps
        # ordinary interior indices unchanged while mapping a last-shard
        # wrap-around source into the upper halo rather than far outside the
        # local array.
        delta = jnp.where(delta < -h, delta + global_size, delta)
        delta = jnp.where(delta > local_size + h, delta - global_size, delta)
        return h + delta

    dphi = 2.0 * jnp.pi / float(global_nz)

    def _direction(sign: int) -> LocalFciDirectionMap:
        # One toroidal-cell trace changes theta by IOTA*dphi.  Express that
        # displacement in theta-cell units so the interpolation remains
        # resolution-aware.
        theta_shift_cells = float(sign) * IOTA * float(global_ny) / float(global_nz)
        global_j = shard_y * local_ny + jj
        continuous_j = global_j + theta_shift_cells
        lower_global_j = jnp.floor(continuous_j).astype(jnp.int32)
        fraction = continuous_j - lower_global_j
        upper_global_j = lower_global_j + 1

        source_j0 = _local_periodic_index(
            lower_global_j,
            shard_y,
            global_ny,
            local_ny,
        )
        source_j1 = _local_periodic_index(
            upper_global_j,
            shard_y,
            global_ny,
            local_ny,
        )

        global_k = shard_z * local_nz + kk
        source_k = _local_periodic_index(
            global_k + sign,
            shard_z,
            global_nz,
            local_nz,
        )

        source_i = h + ii
        source_i = jnp.stack((source_i, source_i), axis=-1).reshape(-1)
        source_j = jnp.stack((source_j0, source_j1), axis=-1).reshape(-1)
        source_k = jnp.stack((source_k, source_k), axis=-1).reshape(-1)
        weights = jnp.stack((1.0 - fraction, fraction), axis=-1).reshape(-1)

        # Midpoint quadrature for the physical arc length along the trace.
        # This preserves the second-order accuracy of the centered FCI
        # derivative when the shifted-torus metric varies with theta.
        theta_mid = theta_owned + 0.5 * float(sign) * IOTA * dphi
        R_mid = R0 + ALPHA * rho_owned + rho_owned * jnp.cos(theta_mid)
        D_mid = jnp.sqrt(IOTA**2 * rho_owned**2 + R_mid**2)
        connection_length = D_mid * dphi

        local = LocalFciLocalDependencyTable(
            target_flat=target_flat,
            source_i=source_i,
            source_j=source_j,
            source_k=source_k,
            weight=weights,
            active=jnp.ones_like(target_flat, dtype=bool),
        )
        return LocalFciDirectionMap(
            layout=layout,
            local=local,
            target_valid=jnp.ones(layout.owned_shape, dtype=bool),
            connection_length=connection_length,
        )

    rho_owned = jnp.asarray(rho_owned, dtype=jnp.float64)
    theta_owned = jnp.asarray(theta_owned, dtype=jnp.float64)
    return LocalFciMaps3D(
        layout=layout,
        forward=_direction(+1),
        backward=_direction(-1),
        mode="local_halo_only",
    )


def _build_global_shifted_torus_fci_maps(
    shape: tuple[int, int, int],
) -> dict[str, jnp.ndarray]:
    """Build the reference RK4-traced FCI maps on the global grid."""

    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    grid = CellCenteredGrid3D(
        x=Grid1D(
            centers=0.5 * (rho_faces[:-1] + rho_faces[1:]),
            faces=rho_faces,
        ),
        y=Grid1D(
            centers=0.5 * (theta_faces[:-1] + theta_faces[1:]),
            faces=theta_faces,
        ),
        z=Grid1D(
            centers=0.5 * (phi_faces[:-1] + phi_faces[1:]),
            faces=phi_faces,
        ),
    )

    rho = grid.x.centers[:, None, None]
    theta = grid.y.centers[None, :, None]
    R = R0 + ALPHA * rho + rho * jnp.cos(theta)
    J = R * rho * (1.0 + ALPHA * jnp.cos(theta))
    R = jnp.broadcast_to(R, shape)
    J = jnp.broadcast_to(J, shape)
    B_contra = jnp.stack(
        (
            jnp.zeros((nx, ny, nz), dtype=jnp.float64),
            IOTA * C_PHI / J,
            C_PHI / J,
        ),
        axis=-1,
    )
    Bmag = C_PHI / J * jnp.sqrt(IOTA**2 * rho**2 + R**2)
    return build_fci_maps_from_b_contravariant(
        grid,
        B_contra,
        Bmag,
        periodic_axes=(False, True, True),
    )


def _build_global_shifted_torus_geometry(
    shape: tuple[int, int, int],
) -> FciGeometry3D:
    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    grid = CellCenteredGrid3D(
        x=Grid1D(
            centers=0.5 * (rho_faces[:-1] + rho_faces[1:]),
            faces=rho_faces,
        ),
        y=Grid1D(
            centers=0.5 * (theta_faces[:-1] + theta_faces[1:]),
            faces=theta_faces,
        ),
        z=Grid1D(
            centers=0.5 * (phi_faces[:-1] + phi_faces[1:]),
            faces=phi_faces,
        ),
    )

    def _metric(rho: jnp.ndarray, theta: jnp.ndarray) -> MetricGeometry:
        rho, theta = jnp.broadcast_arrays(
            jnp.asarray(rho, dtype=jnp.float64),
            jnp.asarray(theta, dtype=jnp.float64),
        )
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        r_major = R0 + ALPHA * rho + rho * cos_theta
        q = 1.0 + ALPHA * cos_theta
        zeros = jnp.zeros_like(r_major)
        return MetricGeometry(
            J=r_major * rho * q,
            g11=1.0 / q**2,
            g22=(1.0 + 2.0 * ALPHA * cos_theta + ALPHA**2) / (rho**2 * q**2),
            g33=1.0 / r_major**2,
            g12=ALPHA * sin_theta / (rho * q**2),
            g13=zeros,
            g23=zeros,
            g_11=1.0 + 2.0 * ALPHA * cos_theta + ALPHA**2,
            g_22=rho**2,
            g_33=r_major**2,
            g_12=-ALPHA * rho * sin_theta,
            g_13=zeros,
            g_23=zeros,
        )

    def _bfield(metric: MetricGeometry) -> BFieldGeometry:
        b_contra = jnp.stack(
            (
                jnp.zeros_like(metric.J),
                IOTA * C_PHI / metric.J,
                C_PHI / metric.J,
            ),
            axis=-1,
        )
        bmag = jnp.sqrt(
            jnp.einsum("...i,...ij,...j->...", b_contra, metric.g_cov, b_contra)
        )
        return BFieldGeometry(B_contra=b_contra, Bmag=bmag)

    cell_z = jnp.zeros((1, 1, nz), dtype=jnp.float64)
    face_z = jnp.zeros((1, 1, nz + 1), dtype=jnp.float64)
    rho = grid.x.centers[:, None, None]
    theta = grid.y.centers[None, :, None]
    cell_metric = _metric(rho + cell_z, theta + cell_z)
    x_face_metric = _metric(grid.x.faces[:, None, None] + cell_z, theta + cell_z)
    y_face_metric = _metric(rho + cell_z, grid.y.faces[None, :, None] + cell_z)
    z_face_metric = _metric(rho + face_z, theta + face_z)

    zeros = jnp.zeros(shape, dtype=jnp.float64)
    ones = jnp.ones(shape, dtype=jnp.float64)
    maps = FciMaps3D(
        forward_x=zeros,
        forward_y=zeros,
        backward_x=zeros,
        backward_y=zeros,
        forward_endpoint_x=zeros,
        forward_endpoint_y=zeros,
        forward_endpoint_z=zeros,
        backward_endpoint_x=zeros,
        backward_endpoint_y=zeros,
        backward_endpoint_z=zeros,
        forward_length=ones,
        backward_length=ones,
        forward_boundary=jnp.zeros(shape, dtype=bool),
        backward_boundary=jnp.zeros(shape, dtype=bool),
    )
    return FciGeometry3D(
        grid=grid,
        maps=maps,
        spacing=Spacing3D(
            dx=jnp.broadcast_to(grid.x.widths[:, None, None], shape),
            dy=jnp.broadcast_to(grid.y.widths[None, :, None], shape),
            dz=jnp.broadcast_to(grid.z.widths[None, None, :], shape),
        ),
        cell_metric=cell_metric,
        face_metric=FaceMetricGeometry(
            x=x_face_metric,
            y=y_face_metric,
            z=z_face_metric,
        ),
        cell_bfield=_bfield(cell_metric),
        face_bfield=FaceBFieldGeometry(
            x=_bfield(x_face_metric),
            y=_bfield(y_face_metric),
            z=_bfield(z_face_metric),
        ),
    )


def _build_local_fci_maps_from_traced_fields(
    layout: HaloLayout3D,
    map_fields: dict[str, jnp.ndarray],
    *,
    global_shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int],
    shard_index: tuple[object, object, object],
) -> LocalFciMaps3D:
    """Convert sharded global FCI map rows into halo-local dependencies."""

    nx, ny, nz = layout.owned_shape
    global_nx, global_ny, global_nz = global_shape
    h = layout.halo_width
    shard_x = jnp.asarray(shard_index[0], dtype=jnp.int32)
    shard_y = jnp.asarray(shard_index[1], dtype=jnp.int32)
    shard_z = jnp.asarray(shard_index[2], dtype=jnp.int32)
    local_nx = global_nx // int(shard_counts[0])
    local_ny = global_ny // int(shard_counts[1])
    local_nz = global_nz // int(shard_counts[2])

    ii, jj, kk = jnp.meshgrid(
        jnp.arange(nx, dtype=jnp.int32),
        jnp.arange(ny, dtype=jnp.int32),
        jnp.arange(nz, dtype=jnp.int32),
        indexing="ij",
    )
    target_flat = jnp.repeat(
        jnp.arange(nx * ny * nz, dtype=jnp.int32),
        4,
    )

    def _local_index(global_index, shard_id, global_size, local_size, *, periodic):
        global_index = jnp.asarray(global_index, dtype=jnp.int32)
        if periodic:
            global_index = jnp.mod(global_index, global_size)
        delta = global_index - shard_id * local_size
        if periodic:
            # Keep periodic sources in the neighboring halo when they cross
            # a shard boundary. This selects the nearby periodic image.
            delta = jnp.where(delta < -h, delta + global_size, delta)
            delta = jnp.where(delta > local_size + h, delta - global_size, delta)
        return h + delta

    def _direction(prefix: str, sign: int) -> LocalFciDirectionMap:
        x_fractional = jnp.asarray(map_fields[f"{prefix}_x"], dtype=jnp.float64)
        y_fractional = jnp.asarray(map_fields[f"{prefix}_y"], dtype=jnp.float64)
        endpoint_z = jnp.asarray(
            map_fields[f"{prefix}_endpoint_z"],
            dtype=jnp.float64,
        )
        length = jnp.asarray(
            map_fields[f"{prefix}_length"],
            dtype=jnp.float64,
        )
        boundary = jnp.asarray(
            map_fields[f"{prefix}_boundary"],
            dtype=bool,
        )

        x0 = jnp.floor(x_fractional).astype(jnp.int32)
        wx = x_fractional - x0
        x0 = jnp.clip(x0, 0, global_nx - 2)
        x1 = x0 + 1

        y0 = jnp.floor(y_fractional).astype(jnp.int32)
        wy = y_fractional - y0
        y1 = y0 + 1

        source_i0 = _local_index(
            x0,
            shard_x,
            global_nx,
            local_nx,
            periodic=False,
        )
        source_i1 = _local_index(
            x1,
            shard_x,
            global_nx,
            local_nx,
            periodic=False,
        )
        source_j0 = _local_index(
            y0,
            shard_y,
            global_ny,
            local_ny,
            periodic=True,
        )
        source_j1 = _local_index(
            y1,
            shard_y,
            global_ny,
            local_ny,
            periodic=True,
        )

        dz = 2.0 * jnp.pi / float(global_nz)
        z0 = 0.5 * dz
        endpoint_index = jnp.rint((endpoint_z - z0) / dz).astype(jnp.int32)
        target_global_k = shard_z * local_nz + kk
        endpoint_index = jnp.mod(endpoint_index, global_nz)
        delta = endpoint_index - target_global_k
        # Preserve the signed periodic crossing so global k=0 backward and
        # global k=nz-1 forward always use the lower/upper halo respectively.
        if sign < 0:
            delta = jnp.where(delta > global_nz // 2, delta - global_nz, delta)
        else:
            delta = jnp.where(delta < -global_nz // 2, delta + global_nz, delta)
        source_k = h + target_global_k + delta - shard_z * local_nz

        source_i = jnp.stack(
            (source_i0, source_i1, source_i0, source_i1), axis=-1
        ).reshape(-1)
        source_j = jnp.stack(
            (source_j0, source_j0, source_j1, source_j1), axis=-1
        ).reshape(-1)
        source_k = jnp.broadcast_to(
            source_k[..., None],
            source_k.shape + (4,),
        ).reshape(-1)
        weights = jnp.stack(
            (
                (1.0 - wx) * (1.0 - wy),
                wx * (1.0 - wy),
                (1.0 - wx) * wy,
                wx * wy,
            ),
            axis=-1,
        ).reshape(-1)
        row_active = jnp.broadcast_to(
            ~boundary[..., None],
            boundary.shape + (4,),
        ).reshape(-1)

        source_global_i = jnp.stack((x0, x1, x0, x1), axis=-1).reshape(-1)
        source_global_j = jnp.mod(
            jnp.stack((y0, y0, y1, y1), axis=-1),
            global_ny,
        ).reshape(-1)
        source_global_k = jnp.broadcast_to(
            endpoint_index[..., None],
            endpoint_index.shape + (4,),
        ).reshape(-1)

        owner_x = source_global_i // local_nx
        owner_y = source_global_j // local_ny
        owner_z = source_global_k // local_nz
        owner_linear = (
            owner_z * (int(shard_counts[1]) * int(shard_counts[0]))
            + owner_y * int(shard_counts[0])
            + owner_x
        ).astype(jnp.int32)
        my_linear = (
            shard_z * (int(shard_counts[1]) * int(shard_counts[0]))
            + shard_y * int(shard_counts[0])
            + shard_x
        ).astype(jnp.int32)
        same_shard = owner_linear == my_linear
        local_active = row_active & same_shard
        remote_active = row_active & ~same_shard

        owner_local_i = h + source_global_i - owner_x * local_nx
        owner_local_j = h + source_global_j - owner_y * local_ny
        owner_local_k = h + source_global_k - owner_z * local_nz

        local = LocalFciLocalDependencyTable(
            target_flat=target_flat,
            source_i=source_i,
            source_j=source_j,
            source_k=source_k,
            weight=weights,
            active=local_active,
        )
        remote = None
        if math.prod(shard_counts) > 1:
            receive_slot = jnp.arange(target_flat.size, dtype=jnp.int32)
            remote = LocalFciRemoteDependencyTable(
                target_flat=target_flat,
                weight=weights,
                receive_slot=receive_slot,
                active=remote_active,
                request_active=remote_active,
                request_dependency_kind=jnp.where(
                    remote_active,
                    FCI_DEP_FIELD_INTERIOR,
                    FCI_DEP_INVALID,
                ),
                request_source_global_i=source_global_i,
                request_source_global_j=source_global_j,
                request_source_global_k=source_global_k,
                request_source_shard_index=jnp.stack(
                    (owner_x, owner_y, owner_z),
                    axis=-1,
                ),
                request_source_shard_linear=owner_linear,
                request_source_owner_local_i=owner_local_i,
                request_source_owner_local_j=owner_local_j,
                request_source_owner_local_k=owner_local_k,
                request_value_slot=jnp.zeros(target_flat.shape, dtype=jnp.int32),
            )
        return LocalFciDirectionMap(
            layout=layout,
            local=local,
            remote=remote,
            target_valid=~boundary,
            connection_length=length,
        )

    return LocalFciMaps3D(
        layout=layout,
        forward=_direction("forward", +1),
        backward=_direction("backward", -1),
        mode=(
            "remote_dependencies"
            if math.prod(shard_counts) > 1
            else "local_halo_only"
        ),
    )


def _build_local_geometry(
    shape: tuple[int, int, int],
    halo_width: int,
    *,
    global_shape: tuple[int, int, int],
    shard_index: tuple[object, object, object] = (0, 0, 0),
    construct_fci_maps: bool = False,
    traced_maps: LocalFciMaps3D | None = None,
) -> LocalFciGeometry3D:
    """Build halo-padded shifted-torus geometry for one local shard."""

    nx, ny, nz = shape
    global_nx, global_ny, global_nz = global_shape
    layout = HaloLayout3D(shape, halo_width)

    # LocalGrid1D metadata is local-array metadata. The coordinate arrays
    # themselves carry the global coordinate values through the runtime shard
    # index supplied by shard_map.
    def axis_grid(global_size, local_size, lower, upper, axis, periodic, shard_id):
        spacing = (upper - lower) / float(global_size)
        start = jnp.asarray(shard_id, dtype=jnp.int32) * local_size
        center_indices = start + jnp.arange(-halo_width, local_size + halo_width)
        face_indices = start + jnp.arange(-halo_width, local_size + halo_width + 1)
        centers_halo = lower + (center_indices + 0.5) * spacing
        faces_halo = lower + face_indices * spacing
        return LocalGrid1D(
            layout=layout,
            axis=axis,
            centers_halo=centers_halo,
            faces_halo=faces_halo,
            owned_start_global=0,
            owned_stop_global=local_size,
        )

    grid = LocalCellCenteredGrid3D(
        layout=layout,
        x=axis_grid(global_nx, nx, RHO_MIN, 1.0, 0, False, shard_index[0]),
        y=axis_grid(global_ny, ny, 0.0, 2.0 * jnp.pi, 1, True, shard_index[1]),
        z=axis_grid(global_nz, nz, 0.0, 2.0 * jnp.pi, 2, True, shard_index[2]),
    )

    rho = grid.x.centers_halo
    theta = grid.y.centers_halo
    phi = grid.z.centers_halo
    rho_3d, theta_3d, phi_3d = jnp.meshgrid(rho, theta, phi, indexing="ij")
    dr = (1.0 - RHO_MIN) / float(global_nx)
    dtheta = 2.0 * jnp.pi / float(global_ny)
    dphi = 2.0 * jnp.pi / float(global_nz)
    spacing = LocalSpacing3D(
        layout=layout,
        dx_halo=jnp.full(layout.cell_halo_shape, dr, dtype=jnp.float64),
        dy_halo=jnp.full(layout.cell_halo_shape, dtheta, dtype=jnp.float64),
        dz_halo=jnp.full(layout.cell_halo_shape, dphi, dtype=jnp.float64),
    )

    cell_metric = _shifted_torus_metric(layout, "cell", rho_3d, theta_3d)
    face_metric = LocalFaceMetricGeometry(
        layout=layout,
        x=_shifted_torus_metric(
            layout, "x_face",
            *jnp.meshgrid(grid.x.faces, grid.y.centers, grid.z.centers, indexing="ij")[:2],
        ),
        y=_shifted_torus_metric(
            layout, "y_face",
            *jnp.meshgrid(grid.x.centers, grid.y.faces, grid.z.centers, indexing="ij")[:2],
        ),
        z=_shifted_torus_metric(
            layout, "z_face",
            *jnp.meshgrid(grid.x.centers, grid.y.centers, grid.z.faces, indexing="ij")[:2],
        ),
    )
    cell_bfield = _shifted_torus_bfield(cell_metric, rho_3d)
    face_bfield = LocalFaceBFieldGeometry(
        layout=layout,
        x=_shifted_torus_bfield(face_metric.x, grid.x.faces[:, None, None]),
        y=_shifted_torus_bfield(face_metric.y, grid.x.centers[:, None, None]),
        z=_shifted_torus_bfield(face_metric.z, grid.x.centers[:, None, None]),
    )

    face_shapes = (
        layout.face_control_shape(0),
        layout.face_control_shape(1),
        layout.face_control_shape(2),
    )
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

    if construct_fci_maps:
        if traced_maps is None:
            raise ValueError("traced_maps is required when construct_fci_maps=True")
        maps = traced_maps
    else:
        maps = _empty_maps(layout)

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
            volume=cell_metric.J_owned,
            volume_fraction=jnp.ones(shape),
        ),
    )


def _build_domain(
    global_shape: tuple[int, int, int],
    halo_width: int,
    shard_counts: tuple[int, int, int] = (1, 1, 1),
) -> LocalDomain3D:
    assert_shape_divisible_by_shards(global_shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(global_shape, shard_counts)
    )
    layout = HaloLayout3D(owned_shape, halo_width)
    spec = ShardSpec3D(
        global_shape=global_shape,
        # These are local-array coordinates. Runtime boundary ownership is
        # determined by LocalDomain3D.runtime_* inside shard_map.
        owned_start=(0, 0, 0),
        owned_stop=owned_shape,
        shard_index=(0, 0, 0),
        shard_counts=tuple(shard_counts),
        periodic_axes=(False, True, True),
        halo_width=halo_width,
        side_kind_lower=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
        side_kind_upper=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
    )
    return LocalDomain3D(
        shard_spec=spec,
        layout=layout,
        mesh_axis_names=("x", "y", "z"),
    )


def _assert_local_stencil_equal(left, right) -> None:
    for axis_name in ("x", "y", "z"):
        left_axis = getattr(left, axis_name)
        right_axis = getattr(right, axis_name)
        for value_name in ("center", "minus", "plus", "dx_min", "dx_plus"):
            assert jnp.array_equal(
                getattr(left_axis, value_name),
                getattr(right_axis, value_name),
            )


def test_local_stencil_builder_preserves_behavior_with_empty_cut_wall_dependencies() -> None:
    layout = HaloLayout3D((1, 1, 1), 1)
    geometry = _build_local_geometry(
        layout.owned_shape,
        layout.halo_width,
        global_shape=layout.owned_shape,
    )
    field_halo = jnp.arange(27, dtype=jnp.float64).reshape(layout.cell_halo_shape)

    plain = build_local_stencil_from_field(
        field_halo,
        geometry,
        StencilBuilderContext(layout=layout),
    )
    with_empty_dependencies = build_local_stencil_from_field(
        field_halo,
        geometry,
        StencilBuilderContext(
            layout=layout,
            cut_wall_stencil_dependencies=LocalCoordinateStencilDependencyMap3D.empty(
                layout
            ),
        ),
    )

    _assert_local_stencil_equal(with_empty_dependencies, plain)


def test_local_stencil_builder_patches_local_coordinate_cut_wall_rows() -> None:
    layout = HaloLayout3D((1, 1, 1), 1)
    geometry = _build_local_geometry(
        layout.owned_shape,
        layout.halo_width,
        global_shape=layout.owned_shape,
    )
    field_halo = jnp.arange(27, dtype=jnp.float64).reshape(layout.cell_halo_shape)
    dependencies = LocalCoordinateStencilDependencyMap3D(
        layout=layout,
        local=LocalCoordinateStencilLocalDependencyTable(
            target_flat=jnp.array([0, 0], dtype=jnp.int32),
            axis=jnp.array([0, 1], dtype=jnp.int32),
            side=jnp.array([1, 0], dtype=jnp.int32),
            value_slot=jnp.array([0, 1], dtype=jnp.int32),
            distance=jnp.array([0.25, 0.5], dtype=jnp.float64),
            active=jnp.array([True, True]),
        ),
    )

    stencil = build_local_stencil_from_field(
        field_halo,
        geometry,
        StencilBuilderContext(
            layout=layout,
            cut_wall_stencil_dependencies=dependencies,
            cut_wall_values=jnp.array([7.0, -3.0], dtype=jnp.float64),
        ),
    )

    assert math.isclose(float(stencil.x.plus[0, 0, 0]), 7.0)
    assert math.isclose(float(stencil.x.dx_plus[0, 0, 0]), 0.25)
    assert math.isclose(
        float(stencil.x.minus[0, 0, 0]),
        float(field_halo[0, 1, 1]),
    )
    assert math.isclose(float(stencil.y.minus[0, 0, 0]), -3.0)
    assert math.isclose(float(stencil.y.dx_min[0, 0, 0]), 0.5)
    assert math.isclose(
        float(stencil.y.plus[0, 0, 0]),
        float(field_halo[1, 2, 1]),
    )


def test_local_stencil_builder_patches_remote_coordinate_cut_wall_rows() -> None:
    layout = HaloLayout3D((1, 1, 1), 1)
    geometry = _build_local_geometry(
        layout.owned_shape,
        layout.halo_width,
        global_shape=layout.owned_shape,
    )
    field_halo = jnp.arange(27, dtype=jnp.float64).reshape(layout.cell_halo_shape)
    remote = LocalCoordinateStencilRemoteDependencyTable(
        target_flat=jnp.array([0], dtype=jnp.int32),
        axis=jnp.array([1], dtype=jnp.int32),
        side=jnp.array([0], dtype=jnp.int32),
        receive_slot=jnp.array([0], dtype=jnp.int32),
        distance=jnp.array([0.5], dtype=jnp.float64),
        active=jnp.array([True]),
        request_active=jnp.zeros((1,), dtype=bool),
        request_dependency_kind=jnp.array([FCI_DEP_INVALID], dtype=jnp.int32),
        request_source_global_i=jnp.zeros((1,), dtype=jnp.int32),
        request_source_global_j=jnp.zeros((1,), dtype=jnp.int32),
        request_source_global_k=jnp.zeros((1,), dtype=jnp.int32),
        request_source_shard_index=jnp.zeros((1, 3), dtype=jnp.int32),
        request_source_shard_linear=jnp.zeros((1,), dtype=jnp.int32),
        request_source_owner_local_i=jnp.zeros((1,), dtype=jnp.int32),
        request_source_owner_local_j=jnp.zeros((1,), dtype=jnp.int32),
        request_source_owner_local_k=jnp.zeros((1,), dtype=jnp.int32),
        request_value_slot=jnp.zeros((1,), dtype=jnp.int32),
    )
    dependencies = LocalCoordinateStencilDependencyMap3D(
        layout=layout,
        local=LocalCoordinateStencilLocalDependencyTable.empty(),
        remote=remote,
    )

    try:
        build_local_stencil_from_field(
            field_halo,
            geometry,
            StencilBuilderContext(
                layout=layout,
                cut_wall_stencil_dependencies=dependencies,
            ),
        )
    except ValueError as exc:
        assert "cut_wall_stencil_remote_values" in str(exc)
    else:
        raise AssertionError("expected missing remote cut-wall values to raise")

    stencil = build_local_stencil_from_field(
        field_halo,
        geometry,
        StencilBuilderContext(
            layout=layout,
            cut_wall_stencil_dependencies=dependencies,
            cut_wall_stencil_remote_values=jnp.array([-4.0], dtype=jnp.float64),
        ),
    )

    assert math.isclose(float(stencil.y.minus[0, 0, 0]), -4.0)
    assert math.isclose(float(stencil.y.dx_min[0, 0, 0]), 0.5)
    assert math.isclose(
        float(stencil.y.plus[0, 0, 0]),
        float(field_halo[1, 2, 1]),
    )


def test_build_local_curvature_coefficients_matches_global_single_shard() -> None:
    shape = (8, 10, 12)
    halo_width = 2
    global_geometry = _build_global_shifted_torus_geometry(shape)
    local_geometry = _build_local_geometry(
        shape,
        halo_width,
        global_shape=shape,
    )
    domain = replace(
        _build_domain(shape, halo_width, (1, 1, 1)),
        mesh_axis_names=(None, None, None),
    )

    global_coefficients = build_curvature_coefficients(
        global_geometry,
        periodic_axes=(False, True, True),
    )
    local_coefficients = build_local_curvature_coefficients(
        local_geometry,
        domain,
        periodic_axes=(False, True, True),
    )

    assert local_coefficients.shape == shape + (3,)
    assert jnp.allclose(local_coefficients, global_coefficients, rtol=2.0e-12, atol=2.0e-12)


def test_build_local_curvature_coefficients_rejects_layout_mismatch() -> None:
    geometry = _build_local_geometry(
        (4, 4, 4),
        2,
        global_shape=(4, 4, 4),
    )
    domain = replace(
        _build_domain((6, 4, 4), 2, (1, 1, 1)),
        mesh_axis_names=(None, None, None),
    )

    try:
        build_local_curvature_coefficients(geometry, domain)
    except ValueError as exc:
        assert "same HaloLayout3D" in str(exc)
    else:
        raise AssertionError("expected layout mismatch to raise")


def test_build_local_curvature_coefficients_rejects_axis_regular_axes() -> None:
    shape = (4, 4, 4)
    halo_width = 2
    geometry = _build_local_geometry(
        shape,
        halo_width,
        global_shape=shape,
    )
    domain = replace(
        _build_domain(shape, halo_width, (1, 1, 1)),
        mesh_axis_names=(None, None, None),
    )

    try:
        build_local_curvature_coefficients(
            geometry,
            domain,
            axis_regular_axes=(True, False, False),
        )
    except NotImplementedError as exc:
        assert "axis_regular_axes" in str(exc)
    else:
        raise AssertionError("expected axis_regular_axes to raise")


def _mms_parallel_field(
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / A)
    return amplitude * jnp.cos(M * theta) * jnp.sin(N * phi)


def _mms_poisson_g_field(
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / A)
    return amplitude * jnp.cos((M + 1.0) * theta) * jnp.sin((N + 1.0) * phi)


def _mms_poisson_bracket_expected(
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / A)

    f_rho = (
        -(jnp.pi / A)
        * jnp.sin(jnp.pi * rho / A)
        * jnp.cos(M * theta)
        * jnp.sin(N * phi)
    )
    f_theta = -amplitude * M * jnp.sin(M * theta) * jnp.sin(N * phi)
    f_phi = amplitude * N * jnp.cos(M * theta) * jnp.cos(N * phi)

    g_m = M + 1.0
    g_n = N + 1.0
    g_rho = (
        -(jnp.pi / A)
        * jnp.sin(jnp.pi * rho / A)
        * jnp.cos(g_m * theta)
        * jnp.sin(g_n * phi)
    )
    g_theta = -amplitude * g_m * jnp.sin(g_m * theta) * jnp.sin(g_n * phi)
    g_phi = amplitude * g_n * jnp.cos(g_m * theta) * jnp.cos(g_n * phi)

    R = R0 + ALPHA * rho + rho * jnp.cos(theta)
    Q = 1.0 + ALPHA * jnp.cos(theta)
    J = rho * R * Q
    D = jnp.sqrt((IOTA**2) * rho**2 + R**2)

    return (
        1.0
        / (J * D)
        * (
            -ALPHA
            * IOTA
            * rho
            * jnp.sin(theta)
            * (f_theta * g_phi - f_phi * g_theta)
            + IOTA * rho**2 * (f_phi * g_rho - f_rho * g_phi)
            + R**2 * (f_rho * g_theta - f_theta * g_rho)
        )
    )


def _mms_parallel_radial_derivative(
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    return (
        -(jnp.pi / A)
        * jnp.sin(jnp.pi * rho / A)
        * jnp.cos(M * theta)
        * jnp.sin(N * phi)
    )


def _mms_curvature_expected(
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / A)
    f_rho = _mms_parallel_radial_derivative(rho, theta, phi)
    f_theta = -amplitude * M * jnp.sin(M * theta) * jnp.sin(N * phi)
    f_phi = amplitude * N * jnp.cos(M * theta) * jnp.cos(N * phi)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = R0 + rho * (ALPHA + cos_theta)
    Q = 1.0 + ALPHA * cos_theta
    J = rho * R * Q
    d2 = (IOTA**2) * rho**2 + R**2
    d = jnp.sqrt(d2)
    p = ALPHA + cos_theta
    e = rho * Q + ALPHA * R
    a_term = (IOTA**2) * rho + R * p

    k_rho = (
        1.0
        / (2.0 * J)
        * (
            -2.0 * rho * R * sin_theta / d
            + 2.0 * rho * R**3 * sin_theta / d**3
            - rho * R**2 * sin_theta * e / (d * J)
        )
    )
    k_theta = (
        -1.0
        / (2.0 * J)
        * (
            2.0 * R * p / d
            - 2.0 * R**2 * a_term / d**3
            + R**2 * Q * (R + rho * p) / (d * J)
        )
    )
    k_phi = (
        IOTA
        / (2.0 * J)
        * (
            rho * (2.0 + ALPHA * cos_theta) / d
            - 2.0 * rho**2 * a_term / d**3
            + 2.0 * ALPHA * rho**2 * R * sin_theta**2 / d**3
            + (
                rho**2 * Q * (R + rho * p)
                - ALPHA * rho**2 * sin_theta**2 * e
            )
            / (d * J)
        )
    )

    return k_rho * f_rho + k_theta * f_theta + k_phi * f_phi


def _mms_parallel_laplacian_expected(
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / A)
    dfd_theta = -amplitude * M * jnp.sin(M * theta) * jnp.sin(N * phi)
    dfd_phi = amplitude * N * jnp.cos(M * theta) * jnp.cos(N * phi)

    f_thetatheta = -(M**2) * amplitude * jnp.cos(M * theta) * jnp.sin(N * phi)
    f_thetaphi = -M * N * amplitude * jnp.sin(M * theta) * jnp.cos(N * phi)
    f_phiphi = -(N**2) * amplitude * jnp.cos(M * theta) * jnp.sin(N * phi)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    r_major = R0 + ALPHA * rho + rho * cos_theta
    q = 1.0 + ALPHA * cos_theta
    jacobian = r_major * rho * q
    d = jnp.sqrt(IOTA**2 * rho**2 + r_major**2)

    r_theta = -rho * sin_theta
    q_theta = -ALPHA * sin_theta
    jacobian_theta = rho * (r_theta * q + r_major * q_theta)
    d_theta = r_major * r_theta / d

    numerator = IOTA * dfd_theta + dfd_phi
    numerator_theta = IOTA * f_thetatheta + f_thetaphi
    numerator_phi = IOTA * f_thetaphi + f_phiphi

    grad_parallel_theta = numerator_theta / d - numerator * d_theta / d**2
    grad_parallel_phi = numerator_phi / d
    expected_direct = (IOTA * grad_parallel_theta + grad_parallel_phi) / d

    flux_coefficient = jacobian / d**2
    flux_coefficient_theta = (
        jacobian_theta / d**2
        - 2.0 * jacobian * d_theta / d**3
    )
    expected_conservative = (
        IOTA
        * (
            flux_coefficient_theta * numerator
            + flux_coefficient * numerator_theta
        )
        + flux_coefficient * numerator_phi
    ) / jacobian

    return expected_direct, expected_conservative


def _mms_grad_perp_expected(
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
    shape: tuple[int, int, int],
) -> jnp.ndarray:
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / A)
    dfd_rho = (
        -(jnp.pi / A)
        * jnp.sin(jnp.pi * rho / A)
        * jnp.cos(M * theta)
        * jnp.sin(N * phi)
    )
    dfd_theta = -amplitude * M * jnp.sin(M * theta) * jnp.sin(N * phi)
    dfd_phi = amplitude * N * jnp.cos(M * theta) * jnp.cos(N * phi)
    df = jnp.stack((dfd_rho, dfd_theta, dfd_phi), axis=-1)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = R0 + ALPHA * rho + rho * cos_theta
    Q = 1.0 + ALPHA * cos_theta

    rho_b = jnp.broadcast_to(rho, shape)
    R_b = jnp.broadcast_to(R, shape)
    Q_b = jnp.broadcast_to(Q, shape)
    sin_b = jnp.broadcast_to(sin_theta, shape)
    cos_b = jnp.broadcast_to(cos_theta, shape)
    zeros = jnp.zeros(shape, dtype=jnp.float64)

    g_contra = jnp.stack(
        (
            jnp.stack(
                (
                    1.0 / Q_b**2,
                    ALPHA * sin_b / (rho_b * Q_b**2),
                    zeros,
                ),
                axis=-1,
            ),
            jnp.stack(
                (
                    ALPHA * sin_b / (rho_b * Q_b**2),
                    (1.0 + 2.0 * ALPHA * cos_b + ALPHA**2)
                    / (rho_b**2 * Q_b**2),
                    zeros,
                ),
                axis=-1,
            ),
            jnp.stack((zeros, zeros, 1.0 / R_b**2), axis=-1),
        ),
        axis=-2,
    )

    D = jnp.sqrt(IOTA**2 * rho_b**2 + R_b**2)
    b_contra = jnp.stack((zeros, IOTA / D, 1.0 / D), axis=-1)
    projector = g_contra - jnp.einsum("...i,...j->...ij", b_contra, b_contra)
    return jnp.einsum("...ij,...j->...i", projector, df)


def _mms_perp_laplacian_local_expected(
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    Fr = 1.0 + jnp.cos(jnp.pi * rho / A)
    Fr_rho = -(jnp.pi / A) * jnp.sin(jnp.pi * rho / A)
    Fr_rhorho = -((jnp.pi / A) ** 2) * jnp.cos(jnp.pi * rho / A)

    f_rho = Fr_rho * jnp.cos(M * theta) * jnp.sin(N * phi)
    f_theta = -M * Fr * jnp.sin(M * theta) * jnp.sin(N * phi)
    f_phi = N * Fr * jnp.cos(M * theta) * jnp.cos(N * phi)
    f_rhorho = Fr_rhorho * jnp.cos(M * theta) * jnp.sin(N * phi)
    f_rhotheta = -M * Fr_rho * jnp.sin(M * theta) * jnp.sin(N * phi)
    f_thetatheta = -(M**2) * Fr * jnp.cos(M * theta) * jnp.sin(N * phi)
    f_thetaphi = -M * N * Fr * jnp.sin(M * theta) * jnp.cos(N * phi)
    f_phiphi = -(N**2) * Fr * jnp.cos(M * theta) * jnp.sin(N * phi)

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = R0 + rho * (ALPHA + cos_theta)
    Q = 1.0 + ALPHA * cos_theta
    J = rho * R * Q
    S = (IOTA**2) * rho**2 + R**2
    E = rho * Q + ALPHA * R
    metric_theta = 1.0 + 2.0 * ALPHA * cos_theta + ALPHA**2

    P_rhorho = 1.0 / Q**2
    P_rhotheta = ALPHA * sin_theta / (rho * Q**2)
    P_thetatheta = metric_theta / (rho**2 * Q**2) - (IOTA**2) / S
    P_thetaphi = -IOTA / S
    P_phiphi = 1.0 / R**2 - 1.0 / S

    C_rho = (1.0 / J) * (
        R + rho * cos_theta + (ALPHA**2) * R * sin_theta**2 / Q**2
    )
    C_theta = (sin_theta / J) * (
        -1.0
        + ALPHA * R * (ALPHA**2 - 1.0) / (rho * Q**2)
        + rho * (IOTA**2) * E / S
        - 2.0 * rho**2 * (IOTA**2) * R**2 * Q / S**2
    )
    C_phi = (rho * IOTA * sin_theta / J) * (
        E / S - 2.0 * rho * R**2 * Q / S**2
    )

    return (
        P_rhorho * f_rhorho
        + 2.0 * P_rhotheta * f_rhotheta
        + P_thetatheta * f_thetatheta
        + 2.0 * P_thetaphi * f_thetaphi
        + P_phiphi * f_phiphi
        + C_rho * f_rho
        + C_theta * f_theta
        + C_phi * f_phi
    )


def _build_physical_bc(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D | None = None,
) -> LocalBoundaryFaceBC3D:
    layout = geometry.layout
    bc = LocalBoundaryFaceBC3D.empty(layout)
    theta = geometry.grid.y.centers_owned[None, :, None]
    phi = geometry.grid.z.centers_owned[None, None, :]
    lower_value = _mms_parallel_field(jnp.asarray(RHO_MIN), theta, phi)[0]
    upper_value = _mms_parallel_field(jnp.asarray(1.0), theta, phi)[0]
    lower_active = True if domain is None else domain.runtime_has_physical_lower(0)
    upper_active = True if domain is None else domain.runtime_has_physical_upper(0)
    value_x = bc.value_x
    kind_x = bc.kind_x
    mask_x = bc.mask_x
    value_x = value_x.at[0].set(jnp.where(lower_active, lower_value, value_x[0]))
    kind_x = kind_x.at[0].set(jnp.where(lower_active, BC_DIRICHLET, kind_x[0]))
    mask_x = mask_x.at[0].set(jnp.where(lower_active, True, mask_x[0]))
    value_x = value_x.at[-1].set(jnp.where(upper_active, upper_value, value_x[-1]))
    kind_x = kind_x.at[-1].set(jnp.where(upper_active, BC_DIRICHLET, kind_x[-1]))
    mask_x = mask_x.at[-1].set(jnp.where(upper_active, True, mask_x[-1]))
    return replace(
        bc,
        kind_x=kind_x,
        value_x=value_x,
        mask_x=mask_x,
    )


def _build_radial_dirichlet_bc(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    field_fn,
) -> LocalBoundaryFaceBC3D:
    bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    theta = geometry.grid.y.centers_owned[None, :, None]
    phi = geometry.grid.z.centers_owned[None, None, :]
    lower_x = geometry.grid.x.faces_halo[geometry.layout.halo_width]
    upper_x = geometry.grid.x.faces_halo[
        geometry.layout.halo_width + geometry.layout.owned_shape[0]
    ]
    lower_value = field_fn(lower_x, theta, phi)[0]
    upper_value = field_fn(upper_x, theta, phi)[0]
    lower_active = domain.runtime_has_physical_lower(0)
    upper_active = domain.runtime_has_physical_upper(0)

    kind_x = bc.kind_x
    value_x = bc.value_x
    mask_x = bc.mask_x
    kind_x = kind_x.at[0].set(jnp.where(lower_active, BC_DIRICHLET, kind_x[0]))
    value_x = value_x.at[0].set(jnp.where(lower_active, lower_value, value_x[0]))
    mask_x = mask_x.at[0].set(jnp.where(lower_active, True, mask_x[0]))
    kind_x = kind_x.at[-1].set(jnp.where(upper_active, BC_DIRICHLET, kind_x[-1]))
    value_x = value_x.at[-1].set(jnp.where(upper_active, upper_value, value_x[-1]))
    mask_x = mask_x.at[-1].set(jnp.where(upper_active, True, mask_x[-1]))
    return replace(
        bc,
        kind_x=kind_x,
        value_x=value_x,
        mask_x=mask_x,
    )


def _build_radial_neumann_bc(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    lower_value: jnp.ndarray | float = 0.0,
    upper_value: jnp.ndarray | float = 0.0,
) -> LocalBoundaryFaceBC3D:
    bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    shape = (geometry.layout.owned_shape[1], geometry.layout.owned_shape[2])
    lower = jnp.broadcast_to(jnp.asarray(lower_value, dtype=jnp.float64), shape)
    upper = jnp.broadcast_to(jnp.asarray(upper_value, dtype=jnp.float64), shape)
    lower_active = domain.runtime_has_physical_lower(0)
    upper_active = domain.runtime_has_physical_upper(0)

    kind_x = bc.kind_x
    value_x = bc.value_x
    mask_x = bc.mask_x
    kind_x = kind_x.at[0].set(jnp.where(lower_active, BC_NEUMANN, kind_x[0]))
    value_x = value_x.at[0].set(jnp.where(lower_active, lower, value_x[0]))
    mask_x = mask_x.at[0].set(jnp.where(lower_active, True, mask_x[0]))
    kind_x = kind_x.at[-1].set(jnp.where(upper_active, BC_NEUMANN, kind_x[-1]))
    value_x = value_x.at[-1].set(jnp.where(upper_active, upper, value_x[-1]))
    mask_x = mask_x.at[-1].set(jnp.where(upper_active, True, mask_x[-1]))
    return replace(
        bc,
        kind_x=kind_x,
        value_x=value_x,
        mask_x=mask_x,
    )


def _build_perp_laplacian_conservative_bc(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
) -> LocalBoundaryFaceBC3D:
    bc = LocalBoundaryFaceBC3D.empty(geometry.layout)
    y_centers = jnp.asarray(geometry.grid.y.centers_owned, dtype=jnp.float64)
    z_centers = jnp.asarray(geometry.grid.z.centers_owned, dtype=jnp.float64)
    lower_neumann = (jnp.pi / float(A)) * jnp.sin(
        jnp.pi * geometry.grid.x.faces_halo[geometry.layout.halo_width] / float(A)
    )
    upper_neumann = -(jnp.pi / float(A)) * jnp.sin(
        jnp.pi
        * geometry.grid.x.faces_halo[
            geometry.layout.halo_width + geometry.layout.owned_shape[0]
        ]
        / float(A)
    )
    pattern = jnp.cos(float(M) * y_centers[:, None]) * jnp.sin(
        float(N) * z_centers[None, :]
    )
    lower_active = domain.runtime_has_physical_lower(0)
    upper_active = domain.runtime_has_physical_upper(0)

    kind_x = bc.kind_x
    value_x = bc.value_x
    mask_x = bc.mask_x
    kind_x = kind_x.at[0].set(jnp.where(lower_active, BC_NEUMANN, kind_x[0]))
    value_x = value_x.at[0].set(
        jnp.where(lower_active, lower_neumann * pattern, value_x[0])
    )
    mask_x = mask_x.at[0].set(jnp.where(lower_active, True, mask_x[0]))
    kind_x = kind_x.at[-1].set(jnp.where(upper_active, BC_NEUMANN, kind_x[-1]))
    value_x = value_x.at[-1].set(
        jnp.where(upper_active, upper_neumann * pattern, value_x[-1])
    )
    mask_x = mask_x.at[-1].set(jnp.where(upper_active, True, mask_x[-1]))
    return replace(
        bc,
        kind_x=kind_x,
        value_x=value_x,
        mask_x=mask_x,
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
        dirichlet=(weights, neutral, neutral),
        neumann_lower=(neutral, neutral, neutral),
        neumann_upper=(neutral, neutral, neutral),
    )


def _build_scalar_boundary_builder(
    face_bc: LocalBoundaryFaceBC3D,
) -> LocalBoundaryConditionBuilder:
    def prepare(
        state_halo_pre_bc,
        geometry,
        domain,
        cut_wall_geometry,
    ) -> LocalBoundaryPreparation3D:
        del state_halo_pre_bc, geometry, domain, cut_wall_geometry
        return LocalBoundaryPreparation3D(
            local_data=LocalBoundaryData3D(
                face_bc=_ScalarFaceBCBundle(field=face_bc),
            ),
        )

    def finalize(
        preparation,
        remote_values,
        state_halo_pre_bc,
        geometry,
        domain,
        cut_wall_geometry,
    ) -> LocalBoundaryData3D:
        del remote_values, state_halo_pre_bc, geometry, domain, cut_wall_geometry
        if preparation.local_data is None:
            raise ValueError("scalar boundary preparation must include local_data")
        return preparation.local_data

    return LocalBoundaryConditionBuilder(
        prepare_fn=prepare,
        finalize_fn=finalize,
    )


def _prepare_scalar_field_halo(
    field_owned: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    ghost_filler: PhysicalGhostCellFiller3D,
    face_bc: LocalBoundaryFaceBC3D | None = None,
) -> tuple[jnp.ndarray, LocalBoundaryData3D]:
    face_bc = _build_physical_bc(geometry, domain) if face_bc is None else face_bc
    preparer = LocalStateAndBoundaryPreparer3D(
        boundary_builder=_build_scalar_boundary_builder(face_bc),
        physical_ghost_filler=ghost_filler,
        halo_exchange=HaloExchange3D(),
        topology_filler=TopologyHaloFiller3D(
            rules=(LocalPeriodicTopologyRule3D(),),
        ),
    )
    prepared = preparer(
        _ScalarFieldState(field=field_owned),
        geometry,
        domain,
    )
    return prepared.state_halo.field, prepared.boundary_data


def _one_sided_physical_stencil_builder() -> LocalStencilBuilder:
    return LocalStencilBuilder(build_local_direct_stencil_one_sided_physical_from_halo)


def run_shard_map_grad_parallel_direct_case(
    *,
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
) -> dict[str, float]:
    """Run the shifted-torus grad-parallel MMS case for equal-sized shards."""

    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(shape, shard_counts)
    )

    domain = _build_domain(shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)

    # Build the same shifted-torus MMS field and exact parallel derivative as
    # the global operator test, then partition the global arrays on the mesh.
    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    phi = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]
    field = _mms_parallel_field(rho, theta, phi)
    R = R0 + ALPHA * rho + rho * jnp.cos(theta)
    J = R * rho * (1.0 + ALPHA * jnp.cos(theta))
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / A)
    dfd_theta = -amplitude * M * jnp.sin(M * theta) * jnp.sin(N * phi)
    dfd_phi = amplitude * N * jnp.cos(M * theta) * jnp.cos(N * phi)
    Bmag = C_PHI / J * jnp.sqrt(IOTA**2 * rho**2 + R**2)
    expected = ((IOTA * C_PHI / J) * dfd_theta + (C_PHI / J) * dfd_phi) / Bmag

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        field_sharded = put_scalar_field_on_mesh(field, mesh)

        def kernel(field_owned):
            # HaloExchange3D and the LocalDomain3D runtime_* predicates use
            # the surrounding mesh axis indices, so this kernel has the same
            # execution path for one or many shards.
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                owned_shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            field_halo, _boundary_data = _prepare_scalar_field_halo(
                field_owned,
                geometry,
                domain,
                ghost_filler=ghost_filler,
            )
            context = StencilBuilderContext(
                layout=domain.layout,
                domain=domain,
            )
            stencil = build_local_direct_stencil_one_sided_physical_from_halo(
                field_halo,
                geometry,
                context,
            )
            return local_grad_parallel_op_direct(stencil, geometry)

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=P("x", "y", "z"),
            check_rep=False,
        )
        actual = kernel(field_sharded)

    error = jnp.asarray(actual) - expected
    return _error_norm_summary(error)


def run_shard_map_parallel_laplacian_direct_case(
    *,
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
) -> dict[str, float]:
    """Run the shifted-torus chained direct parallel-Laplacian MMS case."""

    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(shape, shard_counts)
    )

    domain = _build_domain(shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)
    one_sided_builder = _one_sided_physical_stencil_builder()

    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    phi = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]
    field = _mms_parallel_field(rho, theta, phi)
    expected, _expected_conservative = _mms_parallel_laplacian_expected(
        rho,
        theta,
        phi,
    )

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        field_sharded = put_scalar_field_on_mesh(field, mesh)

        def kernel(field_owned):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                owned_shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            field_halo, _boundary_data = _prepare_scalar_field_halo(
                field_owned,
                geometry,
                domain,
                ghost_filler=ghost_filler,
            )
            context = StencilBuilderContext(
                layout=domain.layout,
                domain=domain,
            )
            return local_parallel_laplacian_direct_op(
                field_halo,
                geometry,
                domain,
                context=context,
                first_stencil_builder=one_sided_builder,
                intermediate_stencil_builder=one_sided_builder,
                halo_exchange=HaloExchange3D(),
                topology_filler=TopologyHaloFiller3D(
                    rules=(LocalPeriodicTopologyRule3D(),),
                ),
            )

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=P("x", "y", "z"),
            check_rep=False,
        )
        actual = kernel(field_sharded)

    error = jnp.asarray(actual) - expected
    return _error_norm_summary(error)


def run_shard_map_parallel_laplacian_conservative_case(
    *,
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
) -> dict[str, float]:
    """Run the shifted-torus conservative parallel-Laplacian MMS case."""

    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(shape, shard_counts)
    )

    domain = _build_domain(shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)

    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    phi = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]
    field = _mms_parallel_field(rho, theta, phi)
    _expected_direct, expected = _mms_parallel_laplacian_expected(
        rho,
        theta,
        phi,
    )

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        field_sharded = put_scalar_field_on_mesh(field, mesh)

        def kernel(field_owned):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                owned_shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            face_bc = _build_physical_bc(geometry, domain)
            field_halo, _boundary_data = _prepare_scalar_field_halo(
                field_owned,
                geometry,
                domain,
                ghost_filler=ghost_filler,
                face_bc=face_bc,
            )
            context = StencilBuilderContext(
                layout=domain.layout,
                domain=domain,
            )
            stencil = build_local_conservative_stencil_from_field(
                field_halo,
                geometry,
                context,
            )
            return local_parallel_laplacian_conservative_op(
                stencil,
                geometry,
                domain,
                face_bc=face_bc,
            )

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=P("x", "y", "z"),
            check_rep=False,
        )
        actual = kernel(field_sharded)

    error = jnp.asarray(actual) - expected
    return _error_norm_summary(error)


def run_shard_map_grad_perp_case(
    *,
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
) -> dict[str, float]:
    """Run the shifted-torus direct perpendicular-gradient MMS case."""

    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(shape, shard_counts)
    )

    domain = _build_domain(shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)

    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    phi = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]
    field = _mms_parallel_field(rho, theta, phi)
    expected = _mms_grad_perp_expected(rho, theta, phi, shape)

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        field_sharded = put_scalar_field_on_mesh(field, mesh)

        def kernel(field_owned):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                owned_shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            field_halo, _boundary_data = _prepare_scalar_field_halo(
                field_owned,
                geometry,
                domain,
                ghost_filler=ghost_filler,
            )
            context = StencilBuilderContext(
                layout=domain.layout,
                domain=domain,
            )
            stencil = build_local_direct_stencil_one_sided_physical_from_halo(
                field_halo,
                geometry,
                context,
            )
            return local_grad_perp_op_direct(stencil, geometry)

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=P("x", "y", "z"),
            check_rep=False,
        )
        actual = kernel(field_sharded)

    error = jnp.asarray(actual) - expected
    return _error_norm_summary(error)


def run_shard_map_poisson_bracket_case(
    *,
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
) -> dict[str, float]:
    """Run the shifted-torus logical Poisson-bracket MMS case."""

    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(shape, shard_counts)
    )

    domain = _build_domain(shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)
    one_sided_builder = _one_sided_physical_stencil_builder()

    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    phi = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]

    f_field = _mms_parallel_field(rho, theta, phi)
    g_field = _mms_poisson_g_field(rho, theta, phi)
    expected = _mms_poisson_bracket_expected(rho, theta, phi)

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        f_sharded = put_scalar_field_on_mesh(f_field, mesh)
        g_sharded = put_scalar_field_on_mesh(g_field, mesh)

        def kernel(f_owned, g_owned):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                owned_shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            f_face_bc = _build_radial_neumann_bc(geometry, domain)
            g_face_bc = _build_radial_dirichlet_bc(
                geometry,
                domain,
                _mms_poisson_g_field,
            )
            f_halo, _f_boundary_data = _prepare_scalar_field_halo(
                f_owned,
                geometry,
                domain,
                ghost_filler=ghost_filler,
                face_bc=f_face_bc,
            )
            g_halo, _g_boundary_data = _prepare_scalar_field_halo(
                g_owned,
                geometry,
                domain,
                ghost_filler=ghost_filler,
                face_bc=g_face_bc,
            )
            context = StencilBuilderContext(
                layout=domain.layout,
                domain=domain,
            )
            f_stencil = one_sided_builder(f_halo, geometry, context)
            g_stencil = one_sided_builder(g_halo, geometry, context)
            return local_poisson_bracket_op(f_stencil, g_stencil, geometry)

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"), P("x", "y", "z")),
            out_specs=P("x", "y", "z"),
            check_rep=False,
        )
        actual = kernel(f_sharded, g_sharded)

    error = jnp.asarray(actual) - expected
    return _error_norm_summary(error)


def run_shard_map_curvature_case(
    *,
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
) -> dict[str, float]:
    """Run the shifted-torus local curvature MMS case."""

    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(shape, shard_counts)
    )

    domain = _build_domain(shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)
    one_sided_builder = _one_sided_physical_stencil_builder()

    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    phi = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]

    field = _mms_parallel_field(rho, theta, phi)
    expected = _mms_curvature_expected(rho, theta, phi)

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        field_sharded = put_scalar_field_on_mesh(field, mesh)

        def kernel(field_owned):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                owned_shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            theta_owned = geometry.grid.y.centers_owned[None, :, None]
            phi_owned = geometry.grid.z.centers_owned[None, None, :]
            lower_x = geometry.grid.x.faces_halo[geometry.layout.halo_width]
            upper_x = geometry.grid.x.faces_halo[
                geometry.layout.halo_width + geometry.layout.owned_shape[0]
            ]
            face_bc = _build_radial_neumann_bc(
                geometry,
                domain,
                lower_value=_mms_parallel_radial_derivative(
                    lower_x,
                    theta_owned,
                    phi_owned,
                )[0],
                upper_value=_mms_parallel_radial_derivative(
                    upper_x,
                    theta_owned,
                    phi_owned,
                )[0],
            )
            field_halo, _boundary_data = _prepare_scalar_field_halo(
                field_owned,
                geometry,
                domain,
                ghost_filler=ghost_filler,
                face_bc=face_bc,
            )
            context = StencilBuilderContext(
                layout=domain.layout,
                domain=domain,
            )
            stencil = one_sided_builder(field_halo, geometry, context)
            curvature_coefficients = build_local_curvature_coefficients(
                geometry,
                domain,
                periodic_axes=(False, True, True),
            )
            return local_curvature_op(
                stencil,
                geometry,
                curvature_coefficients=curvature_coefficients,
            )

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=P("x", "y", "z"),
            check_rep=False,
        )
        actual = kernel(field_sharded)

    error = jnp.asarray(actual) - expected
    return _error_norm_summary(error)


def run_shard_map_perp_laplacian_local_case(
    *,
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
) -> dict[str, float]:
    """Run the shifted-torus pointwise local perpendicular-Laplacian MMS case."""

    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(shape, shard_counts)
    )

    domain = _build_domain(shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)
    one_sided_builder = _one_sided_physical_stencil_builder()

    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    phi = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]
    field = _mms_parallel_field(rho, theta, phi)
    expected = _mms_perp_laplacian_local_expected(rho, theta, phi)

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        field_sharded = put_scalar_field_on_mesh(field, mesh)

        def kernel(field_owned):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                owned_shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            field_halo, _boundary_data = _prepare_scalar_field_halo(
                field_owned,
                geometry,
                domain,
                ghost_filler=ghost_filler,
            )
            context = StencilBuilderContext(
                layout=domain.layout,
                domain=domain,
            )
            return local_perp_laplacian_local_op(
                field_halo,
                geometry,
                domain,
                context=context,
                field_stencil_builder=one_sided_builder,
                intermediate_stencil_builder=one_sided_builder,
                halo_exchange=HaloExchange3D(),
                topology_filler=TopologyHaloFiller3D(
                    rules=(LocalPeriodicTopologyRule3D(),),
                ),
            )

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=P("x", "y", "z"),
            check_rep=False,
        )
        actual = kernel(field_sharded)

    error = jnp.asarray(actual) - expected
    return _error_norm_summary(error)


def run_shard_map_perp_laplacian_conservative_case(
    *,
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
) -> dict[str, float]:
    """Run the shifted-torus conservative perpendicular-Laplacian MMS case."""

    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(shape, shard_counts)
    )

    domain = _build_domain(shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)

    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    phi = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]
    field = _mms_parallel_field(rho, theta, phi)
    expected = _mms_perp_laplacian_local_expected(rho, theta, phi)

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        field_sharded = put_scalar_field_on_mesh(field, mesh)

        def kernel(field_owned):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                owned_shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            field_halo, _boundary_data = _prepare_scalar_field_halo(
                field_owned,
                geometry,
                domain,
                ghost_filler=ghost_filler,
            )
            context = StencilBuilderContext(
                layout=domain.layout,
                domain=domain,
            )
            stencil = build_local_conservative_stencil_from_field(
                field_halo,
                geometry,
                context,
            )
            face_bc = _build_perp_laplacian_conservative_bc(geometry, domain)
            return local_perp_laplacian_conservative_op(
                stencil,
                geometry,
                domain,
                face_bc=face_bc,
            )

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=P("x", "y", "z"),
            check_rep=False,
        )
        actual = kernel(field_sharded)

    error = jnp.asarray(actual) - expected
    return _error_norm_summary(error)


def _single_fci_local_row(
    *,
    source: tuple[int, int, int],
    dependency_kind: int | None = None,
    value_slot: int = 0,
) -> LocalFciLocalDependencyTable:
    kwargs = {}
    if dependency_kind is not None:
        kwargs["dependency_kind"] = jnp.array([dependency_kind], dtype=jnp.int32)
        kwargs["value_slot"] = jnp.array([value_slot], dtype=jnp.int32)
    return LocalFciLocalDependencyTable(
        target_flat=jnp.array([0], dtype=jnp.int32),
        source_i=jnp.array([source[0]], dtype=jnp.int32),
        source_j=jnp.array([source[1]], dtype=jnp.int32),
        source_k=jnp.array([source[2]], dtype=jnp.int32),
        weight=jnp.ones((1,), dtype=jnp.float64),
        active=jnp.ones((1,), dtype=bool),
        **kwargs,
    )


def _empty_fci_local_rows() -> LocalFciLocalDependencyTable:
    return LocalFciLocalDependencyTable(
        target_flat=jnp.zeros((0,), dtype=jnp.int32),
        source_i=jnp.zeros((0,), dtype=jnp.int32),
        source_j=jnp.zeros((0,), dtype=jnp.int32),
        source_k=jnp.zeros((0,), dtype=jnp.int32),
        weight=jnp.zeros((0,), dtype=jnp.float64),
        active=jnp.zeros((0,), dtype=bool),
    )


def _single_fci_remote_row() -> LocalFciRemoteDependencyTable:
    return LocalFciRemoteDependencyTable(
        target_flat=jnp.array([0], dtype=jnp.int32),
        weight=jnp.ones((1,), dtype=jnp.float64),
        receive_slot=jnp.array([0], dtype=jnp.int32),
        active=jnp.ones((1,), dtype=bool),
        request_active=jnp.zeros((1,), dtype=bool),
        request_dependency_kind=jnp.array([FCI_DEP_INVALID], dtype=jnp.int32),
        request_source_global_i=jnp.zeros((1,), dtype=jnp.int32),
        request_source_global_j=jnp.zeros((1,), dtype=jnp.int32),
        request_source_global_k=jnp.zeros((1,), dtype=jnp.int32),
        request_source_shard_index=jnp.zeros((1, 3), dtype=jnp.int32),
        request_source_shard_linear=jnp.zeros((1,), dtype=jnp.int32),
        request_source_owner_local_i=jnp.zeros((1,), dtype=jnp.int32),
        request_source_owner_local_j=jnp.zeros((1,), dtype=jnp.int32),
        request_source_owner_local_k=jnp.zeros((1,), dtype=jnp.int32),
        request_value_slot=jnp.zeros((1,), dtype=jnp.int32),
    )


def test_local_fci_stencil_builder_uses_second_order_endpoint_stencil() -> None:
    layout = HaloLayout3D((1, 1, 1), 1)
    maps = LocalFciMaps3D(
        layout=layout,
        forward=LocalFciDirectionMap(
            layout=layout,
            local=_single_fci_local_row(source=(2, 1, 1)),
            connection_length=jnp.ones(layout.owned_shape, dtype=jnp.float64),
        ),
        backward=LocalFciDirectionMap(
            layout=layout,
            local=_single_fci_local_row(source=(0, 1, 1)),
            connection_length=jnp.ones(layout.owned_shape, dtype=jnp.float64),
        ),
        mode="local_halo_only",
    )
    geometry = _build_local_geometry(
        layout.owned_shape,
        layout.halo_width,
        global_shape=layout.owned_shape,
        construct_fci_maps=True,
        traced_maps=maps,
    )
    field_halo = jnp.zeros(layout.cell_halo_shape, dtype=jnp.float64)
    field_halo = field_halo.at[0, 1, 1].set(0.0)
    field_halo = field_halo.at[1, 1, 1].set(2.0)
    field_halo = field_halo.at[2, 1, 1].set(4.0)

    stencil = build_local_fci_stencil_from_field(
        field_halo,
        geometry,
        StencilBuilderContext(layout=layout),
    )

    assert jnp.allclose(stencil.minus, jnp.array([[[0.0]]]))
    assert jnp.allclose(stencil.center, jnp.array([[[2.0]]]))
    assert jnp.allclose(stencil.plus, jnp.array([[[4.0]]]))
    assert jnp.allclose(local_grad_parallel_op_fci(stencil, geometry), 2.0)


def test_local_fci_stencil_builder_uses_remote_and_cut_wall_slots() -> None:
    layout = HaloLayout3D((1, 1, 1), 1)
    maps = LocalFciMaps3D(
        layout=layout,
        forward=LocalFciDirectionMap(
            layout=layout,
            local=_empty_fci_local_rows(),
            remote=_single_fci_remote_row(),
            connection_length=jnp.ones(layout.owned_shape, dtype=jnp.float64),
        ),
        backward=LocalFciDirectionMap(
            layout=layout,
            local=_single_fci_local_row(
                source=(0, 0, 0),
                dependency_kind=FCI_DEP_CUT_WALL,
                value_slot=0,
            ),
            connection_length=jnp.ones(layout.owned_shape, dtype=jnp.float64),
        ),
        mode="remote_dependencies",
    )
    geometry = _build_local_geometry(
        layout.owned_shape,
        layout.halo_width,
        global_shape=layout.owned_shape,
        construct_fci_maps=True,
        traced_maps=maps,
    )
    field_halo = jnp.zeros(layout.cell_halo_shape, dtype=jnp.float64)

    stencil = build_local_fci_stencil_from_field(
        field_halo,
        geometry,
        StencilBuilderContext(layout=layout),
        forward_remote_values=jnp.array([7.0], dtype=jnp.float64),
        cut_wall_values=jnp.array([-3.0], dtype=jnp.float64),
    )

    assert jnp.allclose(stencil.minus, jnp.array([[[-3.0]]]))
    assert jnp.allclose(stencil.plus, jnp.array([[[7.0]]]))
    assert jnp.allclose(local_grad_parallel_op_fci(stencil, geometry), 5.0)


def run_shard_map_grad_parallel_fci_case(
    *,
    shape: tuple[int, int, int],
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
) -> dict[str, float]:
    """Run the shifted-torus local FCI grad-parallel MMS case."""

    shard_counts = tuple(int(value) for value in shard_counts)
    assert_shape_divisible_by_shards(shape, shard_counts)
    owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(shape, shard_counts)
    )

    domain = _build_domain(shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)

    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    phi = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]
    field = _mms_parallel_field(rho, theta, phi)

    R = R0 + ALPHA * rho + rho * jnp.cos(theta)
    D = jnp.sqrt(IOTA**2 * rho**2 + R**2)
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / A)
    dfd_theta = -amplitude * M * jnp.sin(M * theta) * jnp.sin(N * phi)
    dfd_phi = amplitude * N * jnp.cos(M * theta) * jnp.cos(N * phi)
    expected = (IOTA * dfd_theta + dfd_phi) / D

    traced_map_fields = _build_global_shifted_torus_fci_maps(shape)
    map_names = (
        "forward_x",
        "forward_y",
        "forward_endpoint_z",
        "forward_length",
        "forward_boundary",
        "backward_x",
        "backward_y",
        "backward_endpoint_z",
        "backward_length",
        "backward_boundary",
    )

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        field_sharded = put_scalar_field_on_mesh(field, mesh)
        map_sharding = NamedSharding(mesh, P("x", "y", "z"))
        maps_sharded = tuple(
            jax.device_put(traced_map_fields[name], map_sharding)
            for name in map_names
        )

        def kernel(field_owned, *map_owned):
            local_map_fields = dict(zip(map_names, map_owned))
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            local_maps = _build_local_fci_maps_from_traced_fields(
                domain.layout,
                local_map_fields,
                global_shape=shape,
                shard_counts=shard_counts,
                shard_index=shard_index,
            )
            geometry = _build_local_geometry(
                owned_shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
                construct_fci_maps=True,
                traced_maps=local_maps,
            )
            field_halo, _boundary_data = _prepare_scalar_field_halo(
                field_owned,
                geometry,
                domain,
                ghost_filler=ghost_filler,
            )
            context = StencilBuilderContext(layout=domain.layout, domain=domain)
            forward_remote_values = RemoteFciDependencyExchange()(
                field_halo=field_halo,
                direction=geometry.maps.forward,
                context=context,
                cut_wall_bc=None,
            )
            backward_remote_values = RemoteFciDependencyExchange()(
                field_halo=field_halo,
                direction=geometry.maps.backward,
                context=context,
                cut_wall_bc=None,
            )
            stencil = build_local_fci_stencil_from_field(
                field_halo,
                geometry,
                context,
                forward_remote_values=forward_remote_values,
                backward_remote_values=backward_remote_values,
            )
            return local_grad_parallel_op_fci(stencil, geometry)

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),) * (1 + len(map_names)),
            out_specs=P("x", "y", "z"),
            check_rep=False,
        )
        actual = kernel(field_sharded, *maps_sharded)

    error = jnp.asarray(actual) - expected
    return _error_norm_summary(error)


def test_single_shard_shifted_torus_grad_parallel_direct() -> None:
    result = run_shard_map_grad_parallel_direct_case(
        shape=(32, 32, 32),
        shard_counts=(1, 1, 1),
        halo_width=1,
    )
    assert result["error_l2"] < 1.5e-1
    assert result["error_linf"] < 1.5e-1


def test_single_shard_shifted_torus_grad_parallel_fci() -> None:
    result = run_shard_map_grad_parallel_fci_case(
        shape=(32, 32, 32),
        shard_counts=(1, 1, 1),
        halo_width=2,
    )
    assert result["error_l2"] < 1.5e-1
    assert result["error_linf"] < 1.5e-1


def test_single_shard_shifted_torus_grad_perp() -> None:
    result = run_shard_map_grad_perp_case(
        shape=(32, 32, 32),
        shard_counts=(1, 1, 1),
        halo_width=1,
    )
    assert result["error_l2"] < 2.0e-1
    assert result["error_linf"] < 1.5


def print_jax_runtime_info() -> None:
    import jax

    print("=" * 80)
    print("JAX runtime")
    print("=" * 80)
    print("default backend:", jax.default_backend())
    print("local_device_count:", jax.local_device_count())
    print("devices:")
    for i, device in enumerate(jax.devices()):
        print(f"  [{i}] {device}")
    print("=" * 80)


def estimate_orders(errors: list[float], resolutions: list[int]) -> list[float]:
    if len(errors) != len(resolutions):
        raise ValueError("errors and resolutions must have the same length")
    if len(errors) < 2:
        return []

    orders = []
    for i in range(1, len(errors)):
        e0 = errors[i - 1]
        e1 = errors[i]
        n0 = resolutions[i - 1]
        n1 = resolutions[i]
        orders.append(float(jnp.log(e0 / e1) / jnp.log(n1 / n0)))
    return orders


def _error_norm_summary(error: jnp.ndarray) -> dict[str, float]:
    error = jnp.asarray(error)
    radial_interior = error[1:-1, ...]
    return {
        "error_l2": float(jnp.sqrt(jnp.mean(error**2))),
        "error_l2_interior": float(jnp.sqrt(jnp.mean(radial_interior**2))),
        "error_linf": float(jnp.max(jnp.abs(error))),
    }


def _print_resolution_result(
    *,
    n: int,
    l2: float,
    l2_interior: float,
    linf: float,
    elapsed: float,
) -> None:
    print(
        f"N={n:4d}  "
        f"L2_full={l2:.6e}  "
        f"L2_int={l2_interior:.6e}  "
        f"Linf={linf:.6e}  "
        f"time={elapsed:.3f}s"
    )


def run_grad_parallel_convergence(
    *,
    resolutions: tuple[int, ...] = (20, 40, 80),
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    make_plot: bool = True,
    plot_path: str = "domain_decomp_grad_parallel_convergence.png",
) -> dict[str, object]:
    """Run shifted-torus MMS convergence through ``shard_map``."""

    if len(resolutions) == 0:
        raise ValueError("resolutions must contain at least one value")
    if any(int(n) < 3 for n in resolutions):
        raise ValueError("each resolution must be at least 3")

    l2_errors: list[float] = []
    l2_interior_errors: list[float] = []
    linf_errors: list[float] = []
    case_times: list[float] = []

    print()
    print("=" * 80)
    print("Shifted-torus shard_map grad_parallel_direct convergence")
    print("=" * 80)
    print(f"shard_counts = {shard_counts}")
    print(f"halo_width   = {halo_width}")
    print()

    for n in resolutions:
        start = time.perf_counter()
        result = run_shard_map_grad_parallel_direct_case(
            shape=(n, n, n),
            shard_counts=shard_counts,
            halo_width=halo_width,
        )
        elapsed = time.perf_counter() - start
        l2 = result["error_l2"]
        l2_interior = result["error_l2_interior"]
        linf = result["error_linf"]
        l2_errors.append(l2)
        l2_interior_errors.append(l2_interior)
        linf_errors.append(linf)
        case_times.append(elapsed)
        _print_resolution_result(
            n=n,
            l2=l2,
            l2_interior=l2_interior,
            linf=linf,
            elapsed=elapsed,
        )

    l2_orders = estimate_orders(l2_errors, list(resolutions))
    l2_interior_orders = estimate_orders(l2_interior_errors, list(resolutions))
    linf_orders = estimate_orders(linf_errors, list(resolutions))

    print()
    print("Estimated orders")
    print("-" * 80)
    for i in range(len(l2_orders)):
        print(
            f"N={resolutions[i]} -> {resolutions[i + 1]}: "
            f"L2_full order={l2_orders[i]:.3f}, "
            f"L2_int order={l2_interior_orders[i]:.3f}, "
            f"Linf order={linf_orders[i]:.3f}"
        )

    if make_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.loglog(resolutions, l2_errors, "o-", label="L2 full error")
        plt.loglog(resolutions, l2_interior_errors, "^-", label="L2 radial-interior error")
        plt.loglog(resolutions, linf_errors, "s-", label="Linf error")
        plt.xlabel("Resolution N")
        plt.ylabel("Error")
        plt.title("Shifted-torus shard_map grad_parallel_direct convergence")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print()
        print(f"Saved plot to: {plot_path}")

    return {
        "resolutions": resolutions,
        "l2_errors": l2_errors,
        "l2_interior_errors": l2_interior_errors,
        "linf_errors": linf_errors,
        "case_times": case_times,
        "l2_orders": l2_orders,
        "l2_interior_orders": l2_interior_orders,
        "linf_orders": linf_orders,
    }


def run_parallel_laplacian_direct_convergence(
    *,
    resolutions: tuple[int, ...] = (20, 40, 80),
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    make_plot: bool = True,
    plot_path: str = "domain_decomp_parallel_laplacian_direct_convergence.png",
) -> dict[str, object]:
    """Run shifted-torus MMS convergence for local direct parallel Laplacian."""

    if len(resolutions) == 0:
        raise ValueError("resolutions must contain at least one value")
    if any(int(n) < 3 for n in resolutions):
        raise ValueError("each resolution must be at least 3")

    l2_errors: list[float] = []
    l2_interior_errors: list[float] = []
    linf_errors: list[float] = []
    case_times: list[float] = []

    print()
    print("=" * 80)
    print("Shifted-torus shard_map parallel_laplacian_direct convergence")
    print("=" * 80)
    print(f"shard_counts = {shard_counts}")
    print(f"halo_width   = {halo_width}")
    print()

    for n in resolutions:
        start = time.perf_counter()
        result = run_shard_map_parallel_laplacian_direct_case(
            shape=(n, n, n),
            shard_counts=shard_counts,
            halo_width=halo_width,
        )
        elapsed = time.perf_counter() - start
        l2 = result["error_l2"]
        l2_interior = result["error_l2_interior"]
        linf = result["error_linf"]
        l2_errors.append(l2)
        l2_interior_errors.append(l2_interior)
        linf_errors.append(linf)
        case_times.append(elapsed)
        _print_resolution_result(
            n=n,
            l2=l2,
            l2_interior=l2_interior,
            linf=linf,
            elapsed=elapsed,
        )

    l2_orders = estimate_orders(l2_errors, list(resolutions))
    l2_interior_orders = estimate_orders(l2_interior_errors, list(resolutions))
    linf_orders = estimate_orders(linf_errors, list(resolutions))

    print()
    print("Estimated orders")
    print("-" * 80)
    for i in range(len(l2_orders)):
        print(
            f"N={resolutions[i]} -> {resolutions[i + 1]}: "
            f"L2_full order={l2_orders[i]:.3f}, "
            f"L2_int order={l2_interior_orders[i]:.3f}, "
            f"Linf order={linf_orders[i]:.3f}"
        )

    if make_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.loglog(resolutions, l2_errors, "o-", label="L2 full error")
        plt.loglog(resolutions, l2_interior_errors, "^-", label="L2 radial-interior error")
        plt.loglog(resolutions, linf_errors, "s-", label="Linf error")
        plt.xlabel("Resolution N")
        plt.ylabel("Error")
        plt.title("Shifted-torus shard_map parallel_laplacian_direct convergence")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print()
        print(f"Saved plot to: {plot_path}")

    return {
        "resolutions": resolutions,
        "l2_errors": l2_errors,
        "l2_interior_errors": l2_interior_errors,
        "linf_errors": linf_errors,
        "case_times": case_times,
        "l2_orders": l2_orders,
        "l2_interior_orders": l2_interior_orders,
        "linf_orders": linf_orders,
    }


def run_parallel_laplacian_conservative_convergence(
    *,
    resolutions: tuple[int, ...] = (20, 40, 80),
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    make_plot: bool = True,
    plot_path: str = "domain_decomp_parallel_laplacian_conservative_convergence.png",
) -> dict[str, object]:
    """Run shifted-torus MMS convergence for conservative parallel Laplacian."""

    if len(resolutions) == 0:
        raise ValueError("resolutions must contain at least one value")
    if any(int(n) < 3 for n in resolutions):
        raise ValueError("each resolution must be at least 3")

    l2_errors: list[float] = []
    l2_interior_errors: list[float] = []
    linf_errors: list[float] = []
    case_times: list[float] = []

    print()
    print("=" * 80)
    print("Shifted-torus shard_map parallel_laplacian_conservative convergence")
    print("=" * 80)
    print(f"shard_counts = {shard_counts}")
    print(f"halo_width   = {halo_width}")
    print()

    for n in resolutions:
        start = time.perf_counter()
        result = run_shard_map_parallel_laplacian_conservative_case(
            shape=(n, n, n),
            shard_counts=shard_counts,
            halo_width=halo_width,
        )
        elapsed = time.perf_counter() - start
        l2 = result["error_l2"]
        l2_interior = result["error_l2_interior"]
        linf = result["error_linf"]
        l2_errors.append(l2)
        l2_interior_errors.append(l2_interior)
        linf_errors.append(linf)
        case_times.append(elapsed)
        _print_resolution_result(
            n=n,
            l2=l2,
            l2_interior=l2_interior,
            linf=linf,
            elapsed=elapsed,
        )

    l2_orders = estimate_orders(l2_errors, list(resolutions))
    l2_interior_orders = estimate_orders(l2_interior_errors, list(resolutions))
    linf_orders = estimate_orders(linf_errors, list(resolutions))

    print()
    print("Estimated orders")
    print("-" * 80)
    for i in range(len(l2_orders)):
        print(
            f"N={resolutions[i]} -> {resolutions[i + 1]}: "
            f"L2_full order={l2_orders[i]:.3f}, "
            f"L2_int order={l2_interior_orders[i]:.3f}, "
            f"Linf order={linf_orders[i]:.3f}"
        )

    if make_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.loglog(resolutions, l2_errors, "o-", label="L2 full error")
        plt.loglog(resolutions, l2_interior_errors, "^-", label="L2 radial-interior error")
        plt.loglog(resolutions, linf_errors, "s-", label="Linf error")
        plt.xlabel("Resolution N")
        plt.ylabel("Error")
        plt.title("Shifted-torus shard_map parallel_laplacian_conservative convergence")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print()
        print(f"Saved plot to: {plot_path}")

    return {
        "resolutions": resolutions,
        "l2_errors": l2_errors,
        "l2_interior_errors": l2_interior_errors,
        "linf_errors": linf_errors,
        "case_times": case_times,
        "l2_orders": l2_orders,
        "l2_interior_orders": l2_interior_orders,
        "linf_orders": linf_orders,
    }


def run_grad_perp_convergence(
    *,
    resolutions: tuple[int, ...] = (20, 40, 80),
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 1,
    make_plot: bool = True,
    plot_path: str = "domain_decomp_grad_perp_convergence.png",
) -> dict[str, object]:
    """Run shifted-torus MMS convergence for local direct grad_perp."""

    if len(resolutions) == 0:
        raise ValueError("resolutions must contain at least one value")
    if any(int(n) < 3 for n in resolutions):
        raise ValueError("each resolution must be at least 3")

    l2_errors: list[float] = []
    l2_interior_errors: list[float] = []
    linf_errors: list[float] = []
    case_times: list[float] = []

    print()
    print("=" * 80)
    print("Shifted-torus shard_map grad_perp convergence")
    print("=" * 80)
    print(f"shard_counts = {shard_counts}")
    print(f"halo_width   = {halo_width}")
    print()

    for n in resolutions:
        start = time.perf_counter()
        result = run_shard_map_grad_perp_case(
            shape=(n, n, n),
            shard_counts=shard_counts,
            halo_width=halo_width,
        )
        elapsed = time.perf_counter() - start
        l2 = result["error_l2"]
        l2_interior = result["error_l2_interior"]
        linf = result["error_linf"]
        l2_errors.append(l2)
        l2_interior_errors.append(l2_interior)
        linf_errors.append(linf)
        case_times.append(elapsed)
        _print_resolution_result(
            n=n,
            l2=l2,
            l2_interior=l2_interior,
            linf=linf,
            elapsed=elapsed,
        )

    l2_orders = estimate_orders(l2_errors, list(resolutions))
    l2_interior_orders = estimate_orders(l2_interior_errors, list(resolutions))
    linf_orders = estimate_orders(linf_errors, list(resolutions))

    print()
    print("Estimated orders")
    print("-" * 80)
    for i in range(len(l2_orders)):
        print(
            f"N={resolutions[i]} -> {resolutions[i + 1]}: "
            f"L2_full order={l2_orders[i]:.3f}, "
            f"L2_int order={l2_interior_orders[i]:.3f}, "
            f"Linf order={linf_orders[i]:.3f}"
        )

    if make_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.loglog(resolutions, l2_errors, "o-", label="L2 full error")
        plt.loglog(resolutions, l2_interior_errors, "^-", label="L2 radial-interior error")
        plt.loglog(resolutions, linf_errors, "s-", label="Linf error")
        plt.xlabel("Resolution N")
        plt.ylabel("Vector-component error")
        plt.title("Shifted-torus shard_map grad_perp convergence")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print()
        print(f"Saved plot to: {plot_path}")

    return {
        "resolutions": resolutions,
        "l2_errors": l2_errors,
        "l2_interior_errors": l2_interior_errors,
        "linf_errors": linf_errors,
        "case_times": case_times,
        "l2_orders": l2_orders,
        "l2_interior_orders": l2_interior_orders,
        "linf_orders": linf_orders,
    }


def run_perp_laplacian_local_convergence(
    *,
    resolutions: tuple[int, ...] = (20, 40, 80),
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    make_plot: bool = True,
    plot_path: str = "domain_decomp_perp_laplacian_local_convergence.png",
) -> dict[str, object]:
    """Run shifted-torus MMS convergence for local pointwise perp_laplacian."""

    if len(resolutions) == 0:
        raise ValueError("resolutions must contain at least one value")
    if any(int(n) < 3 for n in resolutions):
        raise ValueError("each resolution must be at least 3")

    l2_errors: list[float] = []
    l2_interior_errors: list[float] = []
    linf_errors: list[float] = []
    case_times: list[float] = []

    print()
    print("=" * 80)
    print("Shifted-torus shard_map perp_laplacian_local convergence")
    print("=" * 80)
    print(f"shard_counts = {shard_counts}")
    print(f"halo_width   = {halo_width}")
    print()

    for n in resolutions:
        start = time.perf_counter()
        result = run_shard_map_perp_laplacian_local_case(
            shape=(n, n, n),
            shard_counts=shard_counts,
            halo_width=halo_width,
        )
        elapsed = time.perf_counter() - start
        l2 = result["error_l2"]
        l2_interior = result["error_l2_interior"]
        linf = result["error_linf"]
        l2_errors.append(l2)
        l2_interior_errors.append(l2_interior)
        linf_errors.append(linf)
        case_times.append(elapsed)
        _print_resolution_result(
            n=n,
            l2=l2,
            l2_interior=l2_interior,
            linf=linf,
            elapsed=elapsed,
        )

    l2_orders = estimate_orders(l2_errors, list(resolutions))
    l2_interior_orders = estimate_orders(l2_interior_errors, list(resolutions))
    linf_orders = estimate_orders(linf_errors, list(resolutions))

    print()
    print("Estimated orders")
    print("-" * 80)
    for i in range(len(l2_orders)):
        print(
            f"N={resolutions[i]} -> {resolutions[i + 1]}: "
            f"L2_full order={l2_orders[i]:.3f}, "
            f"L2_int order={l2_interior_orders[i]:.3f}, "
            f"Linf order={linf_orders[i]:.3f}"
        )

    if make_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.loglog(resolutions, l2_errors, "o-", label="L2 full error")
        plt.loglog(resolutions, l2_interior_errors, "^-", label="L2 radial-interior error")
        plt.loglog(resolutions, linf_errors, "s-", label="Linf error")
        plt.xlabel("Resolution N")
        plt.ylabel("Error")
        plt.title("Shifted-torus shard_map perp_laplacian_local convergence")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print()
        print(f"Saved plot to: {plot_path}")

    return {
        "resolutions": resolutions,
        "l2_errors": l2_errors,
        "l2_interior_errors": l2_interior_errors,
        "linf_errors": linf_errors,
        "case_times": case_times,
        "l2_orders": l2_orders,
        "l2_interior_orders": l2_interior_orders,
        "linf_orders": linf_orders,
    }


def run_poisson_bracket_convergence(
    *,
    resolutions: tuple[int, ...] = (20, 40, 80),
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    make_plot: bool = True,
    plot_path: str = "domain_decomp_poisson_bracket_convergence.png",
) -> dict[str, object]:
    """Run shifted-torus MMS convergence for the local Poisson bracket."""

    if len(resolutions) == 0:
        raise ValueError("resolutions must contain at least one value")
    if any(int(n) < 3 for n in resolutions):
        raise ValueError("each resolution must be at least 3")

    l2_errors: list[float] = []
    l2_interior_errors: list[float] = []
    linf_errors: list[float] = []
    case_times: list[float] = []

    print()
    print("=" * 80)
    print("Shifted-torus shard_map poisson_bracket convergence")
    print("=" * 80)
    print(f"shard_counts = {shard_counts}")
    print(f"halo_width   = {halo_width}")
    print()

    for n in resolutions:
        start = time.perf_counter()
        result = run_shard_map_poisson_bracket_case(
            shape=(n, n, n),
            shard_counts=shard_counts,
            halo_width=halo_width,
        )
        elapsed = time.perf_counter() - start
        l2 = result["error_l2"]
        l2_interior = result["error_l2_interior"]
        linf = result["error_linf"]
        l2_errors.append(l2)
        l2_interior_errors.append(l2_interior)
        linf_errors.append(linf)
        case_times.append(elapsed)
        _print_resolution_result(
            n=n,
            l2=l2,
            l2_interior=l2_interior,
            linf=linf,
            elapsed=elapsed,
        )

    l2_orders = estimate_orders(l2_errors, list(resolutions))
    l2_interior_orders = estimate_orders(l2_interior_errors, list(resolutions))
    linf_orders = estimate_orders(linf_errors, list(resolutions))

    print()
    print("Estimated orders")
    print("-" * 80)
    for i in range(len(l2_orders)):
        print(
            f"N={resolutions[i]} -> {resolutions[i + 1]}: "
            f"L2_full order={l2_orders[i]:.3f}, "
            f"L2_int order={l2_interior_orders[i]:.3f}, "
            f"Linf order={linf_orders[i]:.3f}"
        )

    if make_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.loglog(resolutions, l2_errors, "o-", label="L2 full error")
        plt.loglog(resolutions, l2_interior_errors, "^-", label="L2 radial-interior error")
        plt.loglog(resolutions, linf_errors, "s-", label="Linf error")
        plt.xlabel("Resolution N")
        plt.ylabel("Error")
        plt.title("Shifted-torus shard_map poisson_bracket convergence")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print()
        print(f"Saved plot to: {plot_path}")

    return {
        "resolutions": resolutions,
        "l2_errors": l2_errors,
        "l2_interior_errors": l2_interior_errors,
        "linf_errors": linf_errors,
        "case_times": case_times,
        "l2_orders": l2_orders,
        "l2_interior_orders": l2_interior_orders,
        "linf_orders": linf_orders,
    }


def run_curvature_convergence(
    *,
    resolutions: tuple[int, ...] = (20, 40, 80),
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    make_plot: bool = True,
    plot_path: str = "domain_decomp_curvature_convergence.png",
) -> dict[str, object]:
    """Run shifted-torus MMS convergence for the local curvature operator."""

    if len(resolutions) == 0:
        raise ValueError("resolutions must contain at least one value")
    if any(int(n) < 3 for n in resolutions):
        raise ValueError("each resolution must be at least 3")

    l2_errors: list[float] = []
    l2_interior_errors: list[float] = []
    linf_errors: list[float] = []
    case_times: list[float] = []

    print()
    print("=" * 80)
    print("Shifted-torus shard_map curvature convergence")
    print("=" * 80)
    print(f"shard_counts = {shard_counts}")
    print(f"halo_width   = {halo_width}")
    print()

    for n in resolutions:
        start = time.perf_counter()
        result = run_shard_map_curvature_case(
            shape=(n, n, n),
            shard_counts=shard_counts,
            halo_width=halo_width,
        )
        elapsed = time.perf_counter() - start
        l2 = result["error_l2"]
        l2_interior = result["error_l2_interior"]
        linf = result["error_linf"]
        l2_errors.append(l2)
        l2_interior_errors.append(l2_interior)
        linf_errors.append(linf)
        case_times.append(elapsed)
        _print_resolution_result(
            n=n,
            l2=l2,
            l2_interior=l2_interior,
            linf=linf,
            elapsed=elapsed,
        )

    l2_orders = estimate_orders(l2_errors, list(resolutions))
    l2_interior_orders = estimate_orders(l2_interior_errors, list(resolutions))
    linf_orders = estimate_orders(linf_errors, list(resolutions))

    print()
    print("Estimated orders")
    print("-" * 80)
    for i in range(len(l2_orders)):
        print(
            f"N={resolutions[i]} -> {resolutions[i + 1]}: "
            f"L2_full order={l2_orders[i]:.3f}, "
            f"L2_int order={l2_interior_orders[i]:.3f}, "
            f"Linf order={linf_orders[i]:.3f}"
        )

    if make_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.loglog(resolutions, l2_errors, "o-", label="L2 full error")
        plt.loglog(resolutions, l2_interior_errors, "^-", label="L2 radial-interior error")
        plt.loglog(resolutions, linf_errors, "s-", label="Linf error")
        plt.xlabel("Resolution N")
        plt.ylabel("Error")
        plt.title("Shifted-torus shard_map curvature convergence")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print()
        print(f"Saved plot to: {plot_path}")

    return {
        "resolutions": resolutions,
        "l2_errors": l2_errors,
        "l2_interior_errors": l2_interior_errors,
        "linf_errors": linf_errors,
        "case_times": case_times,
        "l2_orders": l2_orders,
        "l2_interior_orders": l2_interior_orders,
        "linf_orders": linf_orders,
    }


def run_perp_laplacian_conservative_convergence(
    *,
    resolutions: tuple[int, ...] = (20, 40, 80),
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    make_plot: bool = True,
    plot_path: str = "domain_decomp_perp_laplacian_conservative_convergence.png",
) -> dict[str, object]:
    """Run shifted-torus MMS convergence for the conservative perp Laplacian."""

    if len(resolutions) == 0:
        raise ValueError("resolutions must contain at least one value")
    if any(int(n) < 3 for n in resolutions):
        raise ValueError("each resolution must be at least 3")

    l2_errors: list[float] = []
    l2_interior_errors: list[float] = []
    linf_errors: list[float] = []
    case_times: list[float] = []

    print()
    print("=" * 80)
    print("Shifted-torus shard_map perp_laplacian_conservative convergence")
    print("=" * 80)
    print(f"shard_counts = {shard_counts}")
    print(f"halo_width   = {halo_width}")
    print()

    for n in resolutions:
        start = time.perf_counter()
        result = run_shard_map_perp_laplacian_conservative_case(
            shape=(n, n, n),
            shard_counts=shard_counts,
            halo_width=halo_width,
        )
        elapsed = time.perf_counter() - start
        l2 = result["error_l2"]
        l2_interior = result["error_l2_interior"]
        linf = result["error_linf"]
        l2_errors.append(l2)
        l2_interior_errors.append(l2_interior)
        linf_errors.append(linf)
        case_times.append(elapsed)
        _print_resolution_result(
            n=n,
            l2=l2,
            l2_interior=l2_interior,
            linf=linf,
            elapsed=elapsed,
        )

    l2_orders = estimate_orders(l2_errors, list(resolutions))
    l2_interior_orders = estimate_orders(l2_interior_errors, list(resolutions))
    linf_orders = estimate_orders(linf_errors, list(resolutions))

    print()
    print("Estimated orders")
    print("-" * 80)
    for i in range(len(l2_orders)):
        print(
            f"N={resolutions[i]} -> {resolutions[i + 1]}: "
            f"L2_full order={l2_orders[i]:.3f}, "
            f"L2_int order={l2_interior_orders[i]:.3f}, "
            f"Linf order={linf_orders[i]:.3f}"
        )

    if make_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.loglog(resolutions, l2_errors, "o-", label="L2 full error")
        plt.loglog(resolutions, l2_interior_errors, "^-", label="L2 radial-interior error")
        plt.loglog(resolutions, linf_errors, "s-", label="Linf error")
        plt.xlabel("Resolution N")
        plt.ylabel("Error")
        plt.title("Shifted-torus shard_map perp_laplacian_conservative convergence")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print()
        print(f"Saved plot to: {plot_path}")

    return {
        "resolutions": resolutions,
        "l2_errors": l2_errors,
        "l2_interior_errors": l2_interior_errors,
        "linf_errors": linf_errors,
        "case_times": case_times,
        "l2_orders": l2_orders,
        "l2_interior_orders": l2_interior_orders,
        "linf_orders": linf_orders,
    }


def run_grad_parallel_fci_convergence(
    *,
    resolutions: tuple[int, ...] = (20, 40, 80),
    shard_counts: tuple[int, int, int] = (1, 1, 1),
    halo_width: int = 2,
    make_plot: bool = True,
    plot_path: str = "domain_decomp_grad_parallel_fci_convergence.png",
) -> dict[str, object]:
    """Run shifted-torus MMS convergence for local FCI grad_parallel."""

    if len(resolutions) == 0:
        raise ValueError("resolutions must contain at least one value")
    if any(int(n) < 3 for n in resolutions):
        raise ValueError("each resolution must be at least 3")

    l2_errors: list[float] = []
    l2_interior_errors: list[float] = []
    linf_errors: list[float] = []
    case_times: list[float] = []

    print()
    print("=" * 80)
    print("Shifted-torus shard_map grad_parallel_fci convergence")
    print("=" * 80)
    print(f"shard_counts = {shard_counts}")
    print(f"halo_width   = {halo_width}")
    print()

    for n in resolutions:
        start = time.perf_counter()
        result = run_shard_map_grad_parallel_fci_case(
            shape=(n, n, n),
            shard_counts=shard_counts,
            halo_width=halo_width,
        )
        elapsed = time.perf_counter() - start
        l2 = result["error_l2"]
        l2_interior = result["error_l2_interior"]
        linf = result["error_linf"]
        l2_errors.append(l2)
        l2_interior_errors.append(l2_interior)
        linf_errors.append(linf)
        case_times.append(elapsed)
        _print_resolution_result(
            n=n,
            l2=l2,
            l2_interior=l2_interior,
            linf=linf,
            elapsed=elapsed,
        )

    l2_orders = estimate_orders(l2_errors, list(resolutions))
    l2_interior_orders = estimate_orders(l2_interior_errors, list(resolutions))
    linf_orders = estimate_orders(linf_errors, list(resolutions))

    print()
    print("Estimated orders")
    print("-" * 80)
    for i in range(len(l2_orders)):
        print(
            f"N={resolutions[i]} -> {resolutions[i + 1]}: "
            f"L2_full order={l2_orders[i]:.3f}, "
            f"L2_int order={l2_interior_orders[i]:.3f}, "
            f"Linf order={linf_orders[i]:.3f}"
        )

    if make_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.loglog(resolutions, l2_errors, "o-", label="L2 full error")
        plt.loglog(resolutions, l2_interior_errors, "^-", label="L2 radial-interior error")
        plt.loglog(resolutions, linf_errors, "s-", label="Linf error")
        plt.xlabel("Resolution N")
        plt.ylabel("Error")
        plt.title("Shifted-torus shard_map grad_parallel_fci convergence")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print()
        print(f"Saved plot to: {plot_path}")

    return {
        "resolutions": resolutions,
        "l2_errors": l2_errors,
        "l2_interior_errors": l2_interior_errors,
        "linf_errors": linf_errors,
        "case_times": case_times,
        "l2_orders": l2_orders,
        "l2_interior_orders": l2_interior_orders,
        "linf_orders": linf_orders,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Domain-decomp operator-level shard_map tests."
    )
    parser.add_argument(
        "--operator",
        type=str,
        default="grad_parallel_direct",
        choices=[
            "grad_parallel_direct",
            "grad_parallel_fci",
            "curvature",
            "grad_perp",
            "parallel_laplacian_conservative",
            "parallel_laplacian_direct",
            "perp_laplacian_conservative",
            "perp_laplacian_local",
            "poisson_bracket",
        ],
    )
    parser.add_argument(
        "--shard-counts",
        type=int,
        nargs=3,
        metavar=("PX", "PY", "PZ"),
        default=(1, 1, 1),
        help="Number of shards along x, y, and z (default: 1 1 1).",
    )
    parser.add_argument("--resolutions", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--halo-width", type=int, default=2)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot-path",
        type=str,
        default="domain_decomp_operator_convergence.png",
    )
    args = parser.parse_args()

    print_jax_runtime_info()
    if args.operator == "grad_parallel_direct":
        run_grad_parallel_convergence(
            resolutions=tuple(args.resolutions),
            shard_counts=tuple(args.shard_counts),
            halo_width=args.halo_width,
            make_plot=args.plot,
            plot_path=args.plot_path,
        )
    elif args.operator == "curvature":
        run_curvature_convergence(
            resolutions=tuple(args.resolutions),
            shard_counts=tuple(args.shard_counts),
            halo_width=args.halo_width,
            make_plot=args.plot,
            plot_path=args.plot_path,
        )
    elif args.operator == "grad_perp":
        run_grad_perp_convergence(
            resolutions=tuple(args.resolutions),
            shard_counts=tuple(args.shard_counts),
            halo_width=args.halo_width,
            make_plot=args.plot,
            plot_path=args.plot_path,
        )
    elif args.operator == "parallel_laplacian_direct":
        run_parallel_laplacian_direct_convergence(
            resolutions=tuple(args.resolutions),
            shard_counts=tuple(args.shard_counts),
            halo_width=args.halo_width,
            make_plot=args.plot,
            plot_path=args.plot_path,
        )
    elif args.operator == "parallel_laplacian_conservative":
        run_parallel_laplacian_conservative_convergence(
            resolutions=tuple(args.resolutions),
            shard_counts=tuple(args.shard_counts),
            halo_width=args.halo_width,
            make_plot=args.plot,
            plot_path=args.plot_path,
        )
    elif args.operator == "perp_laplacian_local":
        run_perp_laplacian_local_convergence(
            resolutions=tuple(args.resolutions),
            shard_counts=tuple(args.shard_counts),
            halo_width=args.halo_width,
            make_plot=args.plot,
            plot_path=args.plot_path,
        )
    elif args.operator == "perp_laplacian_conservative":
        run_perp_laplacian_conservative_convergence(
            resolutions=tuple(args.resolutions),
            shard_counts=tuple(args.shard_counts),
            halo_width=args.halo_width,
            make_plot=args.plot,
            plot_path=args.plot_path,
        )
    elif args.operator == "poisson_bracket":
        run_poisson_bracket_convergence(
            resolutions=tuple(args.resolutions),
            shard_counts=tuple(args.shard_counts),
            halo_width=args.halo_width,
            make_plot=args.plot,
            plot_path=args.plot_path,
        )
    elif args.operator == "grad_parallel_fci":
        run_grad_parallel_fci_convergence(
            resolutions=tuple(args.resolutions),
            shard_counts=tuple(args.shard_counts),
            halo_width=args.halo_width,
            make_plot=args.plot,
            plot_path=args.plot_path,
        )


if __name__ == "__main__":
    main()
