from __future__ import annotations

from itertools import product

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from jax_drb.geometry import (
    HaloLayout3D,
    LocalBFieldGeometry,
    LocalCellCenteredGrid3D,
    LocalCellVolumeGeometry3D,
    LocalFaceBFieldGeometry,
    LocalFaceMetricGeometry,
    LocalFciGeometry3D,
    LocalGrid1D,
    LocalMetricGeometry,
    LocalRegularFaceGeometry3D,
    LocalSpacing3D,
)

from test_fci_operators_domain_decomp import (
    _build_domain as build_shifted_torus_local_domain,
    _empty_maps,
    assert_shape_divisible_by_shards,
    make_mesh_for_shard_counts,
)


MESH_AXIS_NAMES = ("x", "y", "z")
PERIODIC_AXES = (False, True, True)


def iter_shard_indices(
    shard_counts: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
    counts = tuple(int(value) for value in shard_counts)
    return tuple(product(*(range(count) for count in counts)))


def stack_local_shard_pytree(
    shard_counts: tuple[int, int, int],
    builder,
):
    shard_values = [builder(index) for index in iter_shard_indices(shard_counts)]
    if not shard_values:
        raise ValueError("shard_counts must contain at least one shard")

    def _stack(*leaves):
        arrays = [jnp.asarray(leaf) for leaf in leaves]
        stacked = jnp.stack(arrays, axis=0)
        return stacked.reshape(tuple(int(value) for value in shard_counts) + arrays[0].shape)

    return jax.tree_util.tree_map(_stack, *shard_values)


def expand_local_shard_pytree(pytree):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(leaf)[None, None, None, ...],
        pytree,
    )


def extract_local_shard_pytree(pytree):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(leaf)[0, 0, 0, ...],
        pytree,
    )


def local_shard_pytree_partition_spec(pytree):
    def _spec(leaf):
        array = jnp.asarray(leaf)
        trailing = max(0, int(array.ndim) - 3)
        return P(*MESH_AXIS_NAMES, *([None] * trailing))

    return jax.tree_util.tree_map(_spec, pytree)


def put_local_shard_pytree_on_mesh(pytree, mesh):
    def _put(leaf):
        array = jnp.asarray(leaf)
        trailing = max(0, int(array.ndim) - 3)
        return jax.device_put(
            array,
            NamedSharding(mesh, P(*MESH_AXIS_NAMES, *([None] * trailing))),
        )

    return jax.tree_util.tree_map(_put, pytree)


def _axis_grid(
    layout: HaloLayout3D,
    *,
    global_size: int,
    local_size: int,
    lower: float,
    upper: float,
    axis: int,
    shard_id: object,
    endpoint: bool,
    cell_centered: bool = False,
) -> LocalGrid1D:
    halo_width = int(layout.halo_width)
    if bool(endpoint) and not bool(cell_centered):
        spacing = (
            1.0
            if int(global_size) == 1
            else (float(upper) - float(lower)) / float(global_size - 1)
        )
    else:
        spacing = (float(upper) - float(lower)) / float(global_size)
    start = jnp.asarray(shard_id, dtype=jnp.int32) * int(local_size)
    center_indices = start + jnp.arange(-halo_width, local_size + halo_width)
    face_indices = start + jnp.arange(-halo_width, local_size + halo_width + 1)
    if cell_centered:
        centers_halo = float(lower) + (center_indices + 0.5) * spacing
        faces_halo = float(lower) + face_indices * spacing
    else:
        centers_halo = float(lower) + center_indices * spacing
        faces_halo = float(lower) + (face_indices - 0.5) * spacing
    return LocalGrid1D(
        layout=layout,
        axis=axis,
        centers_halo=centers_halo,
        faces_halo=faces_halo,
        owned_start_global=0,
        owned_stop_global=local_size,
    )


def _shifted_torus_metric(
    layout: HaloLayout3D,
    location: str,
    x: jnp.ndarray,
    theta: jnp.ndarray,
    *,
    x_min: float,
    x_max: float,
    r0: float,
    alpha_value: float,
    sigma: float,
) -> LocalMetricGeometry:
    x, theta = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(theta, dtype=jnp.float64),
    )
    x_mid = 0.5 * (float(x_min) + float(x_max))
    theta_shift = theta + float(sigma) * (x - x_mid)
    cos_theta = jnp.cos(theta_shift)
    sin_theta = jnp.sin(theta_shift)
    R = float(r0) + float(alpha_value) * x + x * cos_theta
    Q = 1.0 + float(alpha_value) * cos_theta
    zeros = jnp.zeros_like(R)
    return LocalMetricGeometry(
        layout=layout,
        J_halo=R * x * Q,
        g11_halo=1.0 / Q**2,
        g22_halo=(1.0 + 2.0 * float(alpha_value) * cos_theta + float(alpha_value) ** 2)
        / (x**2 * Q**2),
        g33_halo=1.0 / R**2,
        g12_halo=float(alpha_value) * sin_theta / (x * Q**2),
        g13_halo=zeros,
        g23_halo=zeros,
        g_11_halo=1.0 + 2.0 * float(alpha_value) * cos_theta + float(alpha_value) ** 2,
        g_22_halo=x**2,
        g_33_halo=R**2,
        g_12_halo=-float(alpha_value) * x * sin_theta,
        g_13_halo=zeros,
        g_23_halo=zeros,
        location=location,
    )


def _shifted_torus_bfield(
    layout: HaloLayout3D,
    metric: LocalMetricGeometry,
    *,
    iota: float,
    c_phi: float,
) -> LocalBFieldGeometry:
    jacobian = metric.J_halo
    B_contra = jnp.stack(
        (
            jnp.zeros_like(jacobian),
            float(iota) * float(c_phi) / jacobian,
            float(c_phi) / jacobian,
        ),
        axis=-1,
    )
    bmag = jnp.sqrt(
        jnp.einsum(
            "...i,...ij,...j->...",
            B_contra,
            metric.g_cov,
            B_contra,
        )
    )
    return LocalBFieldGeometry(
        layout=layout,
        B_contra_halo=B_contra,
        Bmag_halo=bmag,
        location=metric.location,
    )


def build_shifted_torus_local_geometry(
    shape: tuple[int, int, int],
    halo_width: int,
    *,
    global_shape: tuple[int, int, int],
    shard_index: tuple[object, object, object] = (0, 0, 0),
    construct_fci_maps: bool = False,
    traced_maps=None,
    x_min: float = 0.2,
    x_max: float = 1.0,
    r0: float = 3.0,
    alpha_value: float = 0.25,
    iota: float = 1.1,
    c_phi: float = 3.0,
    sigma: float = 0.0,
) -> LocalFciGeometry3D:
    """Build one shard's halo-padded shifted-torus MMS geometry."""

    if construct_fci_maps:
        if traced_maps is None:
            raise ValueError("traced_maps is required when construct_fci_maps=True")
        maps = traced_maps
    else:
        maps = None

    nx, ny, nz = tuple(int(value) for value in shape)
    global_nx, global_ny, global_nz = tuple(int(value) for value in global_shape)
    layout = HaloLayout3D((nx, ny, nz), int(halo_width))
    grid = LocalCellCenteredGrid3D(
        layout=layout,
        x=_axis_grid(
            layout,
            global_size=global_nx,
            local_size=nx,
            lower=x_min,
            upper=x_max,
            axis=0,
            shard_id=shard_index[0],
            endpoint=True,
            cell_centered=True,
        ),
        y=_axis_grid(
            layout,
            global_size=global_ny,
            local_size=ny,
            lower=0.0,
            upper=2.0 * jnp.pi,
            axis=1,
            shard_id=shard_index[1],
            endpoint=False,
        ),
        z=_axis_grid(
            layout,
            global_size=global_nz,
            local_size=nz,
            lower=0.0,
            upper=2.0 * jnp.pi,
            axis=2,
            shard_id=shard_index[2],
            endpoint=False,
        ),
    )
    x_3d, theta_3d, _zeta_3d = jnp.meshgrid(
        grid.x.centers_halo,
        grid.y.centers_halo,
        grid.z.centers_halo,
        indexing="ij",
    )
    dx = (float(x_max) - float(x_min)) / float(global_nx)
    dy = 2.0 * jnp.pi / float(global_ny)
    dz = 2.0 * jnp.pi / float(global_nz)
    spacing = LocalSpacing3D(
        layout=layout,
        dx_halo=jnp.full(layout.cell_halo_shape, dx, dtype=jnp.float64),
        dy_halo=jnp.full(layout.cell_halo_shape, dy, dtype=jnp.float64),
        dz_halo=jnp.full(layout.cell_halo_shape, dz, dtype=jnp.float64),
    )
    metric_kwargs = dict(
        x_min=x_min,
        x_max=x_max,
        r0=r0,
        alpha_value=alpha_value,
        sigma=sigma,
    )
    cell_metric = _shifted_torus_metric(layout, "cell", x_3d, theta_3d, **metric_kwargs)
    face_metric = LocalFaceMetricGeometry(
        layout=layout,
        x=_shifted_torus_metric(
            layout,
            "x_face",
            *jnp.meshgrid(grid.x.faces, grid.y.centers, grid.z.centers, indexing="ij")[:2],
            **metric_kwargs,
        ),
        y=_shifted_torus_metric(
            layout,
            "y_face",
            *jnp.meshgrid(grid.x.centers, grid.y.faces, grid.z.centers, indexing="ij")[:2],
            **metric_kwargs,
        ),
        z=_shifted_torus_metric(
            layout,
            "z_face",
            *jnp.meshgrid(grid.x.centers, grid.y.centers, grid.z.faces, indexing="ij")[:2],
            **metric_kwargs,
        ),
    )
    bfield_kwargs = dict(iota=iota, c_phi=c_phi)
    cell_bfield = _shifted_torus_bfield(layout, cell_metric, **bfield_kwargs)
    face_bfield = LocalFaceBFieldGeometry(
        layout=layout,
        x=_shifted_torus_bfield(layout, face_metric.x, **bfield_kwargs),
        y=_shifted_torus_bfield(layout, face_metric.y, **bfield_kwargs),
        z=_shifted_torus_bfield(layout, face_metric.z, **bfield_kwargs),
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
    return LocalFciGeometry3D(
        layout=layout,
        grid=grid,
        maps=_empty_maps(layout) if maps is None else maps,
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
