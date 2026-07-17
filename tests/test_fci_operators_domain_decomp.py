"""Shard-map tests for local FCI operators using the shifted-torus MMS."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys

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

from drbx.geometry import (
    CellCenteredGrid3D,
    HaloLayout3D,
    Grid1D,
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
    StencilBuilderContext,
    SIDE_PHYSICAL,
    SIDE_SIMPLE_PERIODIC,
    ShardSpec3D,
    build_fci_maps_from_b_contravariant,
)
from drbx.native.fci_boundaries import (
    BC_DIRICHLET,
    LocalBoundaryFaceBC3D,
    LocalStencil1D,
)
from drbx.native.fci_halo import (
    GhostFillWeights1D,
    HaloExchange3D,
    LocalPeriodicTopologyRule3D,
    PhysicalGhostCellFiller3D,
    TopologyHaloFiller3D,
)
from drbx.native.fci_model import inject_owned_field_to_halo
from drbx.native.fci_operators import (
    local_grad_parallel_op_direct,
    local_grad_parallel_op_fci,
)
from drbx.geometry import build_local_stencil_from_field


IOTA = 1.1
A = 0.2
M = 1
N = 1
R0 = 3.0
ALPHA = 0.25
C_PHI = 3.0
RHO_MIN = 0.2


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
        active = jnp.broadcast_to(~boundary[..., None], boundary.shape + (4,)).reshape(-1)

        local = LocalFciLocalDependencyTable(
            target_flat=target_flat,
            source_i=source_i,
            source_j=source_j,
            source_k=source_k,
            weight=weights,
            active=active,
        )
        return LocalFciDirectionMap(
            layout=layout,
            local=local,
            target_valid=~boundary,
            connection_length=length,
        )

    return LocalFciMaps3D(
        layout=layout,
        forward=_direction("forward", +1),
        backward=_direction("backward", -1),
        mode="local_halo_only",
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
            volume=jnp.ones(shape),
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


def _mms_parallel_field(
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    amplitude = 1.0 + jnp.cos(jnp.pi * rho / A)
    return amplitude * jnp.cos(M * theta) * jnp.sin(N * phi)


def _build_physical_bc(geometry: LocalFciGeometry3D) -> LocalBoundaryFaceBC3D:
    layout = geometry.layout
    bc = LocalBoundaryFaceBC3D.empty(layout)
    theta = geometry.grid.y.centers_owned[None, :, None]
    phi = geometry.grid.z.centers_owned[None, None, :]
    lower_value = _mms_parallel_field(jnp.asarray(RHO_MIN), theta, phi)[0]
    upper_value = _mms_parallel_field(jnp.asarray(1.0), theta, phi)[0]
    value_x = bc.value_x.at[0].set(lower_value)
    value_x = value_x.at[-1].set(upper_value)
    return replace(
        bc,
        kind_x=bc.kind_x.at[0].set(BC_DIRICHLET).at[-1].set(BC_DIRICHLET),
        value_x=value_x,
        mask_x=bc.mask_x.at[0].set(True).at[-1].set(True),
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
            face_bc = _build_physical_bc(geometry)
            field_halo = inject_owned_field_to_halo(field_owned, domain.layout)
            field_halo = HaloExchange3D()(field_halo, domain)
            field_halo = TopologyHaloFiller3D(
                rules=(LocalPeriodicTopologyRule3D(),)
            )(field_halo, domain)
            field_halo = ghost_filler(field_halo, domain, face_bc)
            context = StencilBuilderContext(
                layout=domain.layout,
                domain=domain,
            )
            stencil = build_local_stencil_from_field(
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
    return {
        "error_l2": float(jnp.sqrt(jnp.mean(error**2))),
        "error_linf": float(jnp.max(jnp.abs(error))),
    }


def _build_local_fci_stencil_from_halo(
    field_halo: jnp.ndarray,
    geometry: LocalFciGeometry3D,
) -> LocalStencil1D:
    """Evaluate the local-halo FCI dependency tables into a 1D stencil."""

    field_halo = jnp.asarray(field_halo, dtype=jnp.float64)
    if field_halo.shape != geometry.halo_shape:
        raise ValueError(
            f"field_halo must have shape {geometry.halo_shape}, got {field_halo.shape}"
        )

    layout = geometry.layout
    owned = field_halo[layout.owned_slices_cell]
    n_owned = int(np.prod(geometry.owned_shape))

    def _evaluate(direction) -> jnp.ndarray:
        table = direction.local
        samples = field_halo[table.source_i, table.source_j, table.source_k]
        values = jnp.zeros((n_owned,), dtype=jnp.float64)
        values = values.at[table.target_flat].add(
            jnp.where(table.active, table.weight * samples, 0.0)
        )
        return values.reshape(geometry.owned_shape)

    backward = _evaluate(geometry.maps.backward)
    forward = _evaluate(geometry.maps.forward)
    return LocalStencil1D(
        center=owned,
        minus=backward,
        plus=forward,
        dx_min=geometry.maps.backward.connection_length,
        dx_plus=geometry.maps.forward.connection_length,
    )


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
            face_bc = _build_physical_bc(geometry)
            field_halo = inject_owned_field_to_halo(field_owned, domain.layout)
            field_halo = HaloExchange3D()(field_halo, domain)
            field_halo = TopologyHaloFiller3D(
                rules=(LocalPeriodicTopologyRule3D(),)
            )(field_halo, domain)
            field_halo = ghost_filler(field_halo, domain, face_bc)
            stencil = _build_local_fci_stencil_from_halo(field_halo, geometry)
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
    return {
        "error_l2": float(jnp.sqrt(jnp.mean(error**2))),
        "error_linf": float(jnp.max(jnp.abs(error))),
    }


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
    linf_errors: list[float] = []

    print()
    print("=" * 80)
    print("Shifted-torus shard_map grad_parallel_direct convergence")
    print("=" * 80)
    print(f"shard_counts = {shard_counts}")
    print(f"halo_width   = {halo_width}")
    print()

    for n in resolutions:
        result = run_shard_map_grad_parallel_direct_case(
            shape=(n, n, n),
            shard_counts=shard_counts,
            halo_width=halo_width,
        )
        l2 = result["error_l2"]
        linf = result["error_linf"]
        l2_errors.append(l2)
        linf_errors.append(linf)
        print(f"N={n:4d}  L2={l2:.6e}  Linf={linf:.6e}")

    l2_orders = estimate_orders(l2_errors, list(resolutions))
    linf_orders = estimate_orders(linf_errors, list(resolutions))

    print()
    print("Estimated orders")
    print("-" * 80)
    for i in range(len(l2_orders)):
        print(
            f"N={resolutions[i]} -> {resolutions[i + 1]}: "
            f"L2 order={l2_orders[i]:.3f}, "
            f"Linf order={linf_orders[i]:.3f}"
        )

    if make_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.loglog(resolutions, l2_errors, "o-", label="L2 error")
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
        "linf_errors": linf_errors,
        "l2_orders": l2_orders,
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
    linf_errors: list[float] = []

    print()
    print("=" * 80)
    print("Shifted-torus shard_map grad_parallel_fci convergence")
    print("=" * 80)
    print(f"shard_counts = {shard_counts}")
    print(f"halo_width   = {halo_width}")
    print()

    for n in resolutions:
        result = run_shard_map_grad_parallel_fci_case(
            shape=(n, n, n),
            shard_counts=shard_counts,
            halo_width=halo_width,
        )
        l2 = result["error_l2"]
        linf = result["error_linf"]
        l2_errors.append(l2)
        linf_errors.append(linf)
        print(f"N={n:4d}  L2={l2:.6e}  Linf={linf:.6e}")

    l2_orders = estimate_orders(l2_errors, list(resolutions))
    linf_orders = estimate_orders(linf_errors, list(resolutions))

    print()
    print("Estimated orders")
    print("-" * 80)
    for i in range(len(l2_orders)):
        print(
            f"N={resolutions[i]} -> {resolutions[i + 1]}: "
            f"L2 order={l2_orders[i]:.3f}, "
            f"Linf order={linf_orders[i]:.3f}"
        )

    if make_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.loglog(resolutions, l2_errors, "o-", label="L2 error")
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
        "linf_errors": linf_errors,
        "l2_orders": l2_orders,
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
        choices=["grad_parallel_direct", "grad_parallel_fci"],
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
