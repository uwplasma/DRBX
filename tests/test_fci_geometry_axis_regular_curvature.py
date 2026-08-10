"""Manufactured polar test for local axis-regular curvature coefficients."""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from drbx.geometry import (  # noqa: E402
    HaloLayout3D,
    LocalBFieldGeometry,
    LocalCellCenteredGrid3D,
    LocalCellVolumeGeometry3D,
    LocalDomain3D,
    LocalFaceBFieldGeometry,
    LocalFaceMetricGeometry,
    LocalFciGeometry3D,
    LocalGrid1D,
    LocalMetricGeometry,
    LocalRegularFaceGeometry3D,
    LocalSpacing3D,
    NeighborMap3D,
    SIDE_AXIS_REGULAR,
    SIDE_PHYSICAL,
    SIDE_SIMPLE_PERIODIC,
    ShardSpec3D,
    StencilBuilderContext,
    build_local_stencil_from_field,
    build_local_curvature_coefficients,
)
from test_fci_operators_domain_decomp import _empty_maps  # noqa: E402
from drbx.native.fci_operators import local_curvature_op  # noqa: E402


def _axis_grid(
    layout, axis, n, lower, upper, periodic, *, global_n=None, global_start=0
):
    del periodic
    h = layout.halo_width
    global_n = n if global_n is None else int(global_n)
    spacing = (upper - lower) / global_n
    indices = global_start + jnp.arange(-h, n + h, dtype=jnp.float64)
    centers = lower + (indices + 0.5) * spacing
    faces = lower + (
        global_start + jnp.arange(-h, n + h + 1, dtype=jnp.float64)
    ) * spacing
    return LocalGrid1D(
        layout=layout,
        axis=axis,
        centers_halo=centers,
        faces_halo=faces,
        owned_start_global=0,
        owned_stop_global=n,
    )


def _polar_metric(layout, location, rho, theta):
    rho, theta = jnp.broadcast_arrays(jnp.asarray(rho), jnp.asarray(theta))
    zeros = jnp.zeros_like(rho)
    rho2 = rho * rho
    rho2_safe = jnp.maximum(rho2, 1.0e-12)
    return LocalMetricGeometry(
        layout=layout,
        J_halo=rho,
        g11_halo=jnp.ones_like(rho),
        g22_halo=1.0 / rho2_safe,
        g33_halo=jnp.ones_like(rho),
        g12_halo=zeros,
        g13_halo=zeros,
        g23_halo=zeros,
        g_11_halo=jnp.ones_like(rho),
        g_22_halo=rho2,
        g_33_halo=jnp.ones_like(rho),
        g_12_halo=zeros,
        g_13_halo=zeros,
        g_23_halo=zeros,
        location=location,
    )


def _polar_bfield(metric, alpha):
    rho = jnp.sqrt(jnp.maximum(metric.g_22_halo, 0.0))
    B = jnp.stack((jnp.zeros_like(rho), alpha * jnp.ones_like(rho), jnp.ones_like(rho)), axis=-1)
    Bmag = jnp.sqrt(1.0 + alpha * alpha * rho * rho)
    return LocalBFieldGeometry(
        layout=metric.layout,
        B_contra_halo=B,
        Bmag_halo=Bmag,
        location=metric.location,
    )


def _build_polar_geometry(
    shape=(8, 32, 4),
    halo_width=2,
    alpha=0.35,
    *,
    global_shape=None,
    shard_counts=(1, 1, 1),
    coordinate_shard_index=(0, 0, 0),
    mesh_axis_names=(None, None, None),
):
    nx, ny, nz = shape
    global_shape = shape if global_shape is None else tuple(global_shape)
    starts = tuple(
        int(coordinate_shard_index[axis]) * int(shape[axis])
        for axis in range(3)
    )
    layout = HaloLayout3D(shape, halo_width)
    grid = LocalCellCenteredGrid3D(
        layout=layout,
        x=_axis_grid(layout, 0, nx, 0.0, 1.0, False, global_n=global_shape[0], global_start=starts[0]),
        y=_axis_grid(layout, 1, ny, 0.0, 2.0 * jnp.pi, True, global_n=global_shape[1], global_start=starts[1]),
        z=_axis_grid(layout, 2, nz, 0.0, 2.0 * jnp.pi, True, global_n=global_shape[2], global_start=starts[2]),
    )
    rho, theta, zeta = jnp.meshgrid(
        grid.x.centers, grid.y.centers, grid.z.centers, indexing="ij"
    )
    dr = 1.0 / global_shape[0]
    dtheta = 2.0 * jnp.pi / global_shape[1]
    dzeta = 2.0 * jnp.pi / global_shape[2]
    spacing = LocalSpacing3D(
        layout=layout,
        dx_halo=jnp.full(layout.cell_halo_shape, dr),
        dy_halo=jnp.full(layout.cell_halo_shape, dtheta),
        dz_halo=jnp.full(layout.cell_halo_shape, dzeta),
    )
    cell_metric = _polar_metric(layout, "cell", rho, theta)
    face_metric = LocalFaceMetricGeometry(
        layout=layout,
        x=_polar_metric(layout, "x_face", *jnp.meshgrid(grid.x.faces, grid.y.centers, grid.z.centers, indexing="ij")[:2]),
        y=_polar_metric(layout, "y_face", *jnp.meshgrid(grid.x.centers, grid.y.faces, grid.z.centers, indexing="ij")[:2]),
        z=_polar_metric(layout, "z_face", *jnp.meshgrid(grid.x.centers, grid.y.centers, grid.z.faces, indexing="ij")[:2]),
    )
    cell_bfield = _polar_bfield(cell_metric, alpha)
    face_bfield = LocalFaceBFieldGeometry(
        layout=layout,
        x=_polar_bfield(face_metric.x, alpha),
        y=_polar_bfield(face_metric.y, alpha),
        z=_polar_bfield(face_metric.z, alpha),
    )
    face_shapes = tuple(layout.face_control_shape(axis) for axis in range(3))
    regular = LocalRegularFaceGeometry3D(
        layout=layout,
        x_area=jnp.ones(face_shapes[0]), y_area=jnp.ones(face_shapes[1]), z_area=jnp.ones(face_shapes[2]),
        x_area_fraction=jnp.ones(face_shapes[0]), y_area_fraction=jnp.ones(face_shapes[1]), z_area_fraction=jnp.ones(face_shapes[2]),
        x_open_mask=jnp.ones(face_shapes[0], dtype=bool), y_open_mask=jnp.ones(face_shapes[1], dtype=bool), z_open_mask=jnp.ones(face_shapes[2], dtype=bool),
    )
    geometry = LocalFciGeometry3D(
        layout=layout,
        grid=grid,
        maps=_empty_maps(layout),
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
    domain = LocalDomain3D(
        shard_spec=ShardSpec3D(
            global_shape=global_shape,
            owned_start=(0, 0, 0),
            owned_stop=shape,
            shard_index=(0, 0, 0),
            shard_counts=shard_counts,
            periodic_axes=(False, True, True),
            axis_regular_axes=(True, False, False),
            halo_width=halo_width,
            side_kind_lower=(SIDE_AXIS_REGULAR, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
            side_kind_upper=(SIDE_PHYSICAL, SIDE_SIMPLE_PERIODIC, SIDE_SIMPLE_PERIODIC),
        ),
        layout=layout,
        neighbor_map=NeighborMap3D(minus=(None, None, None), plus=(None, None, None)),
        mesh_axis_names=mesh_axis_names,
    )
    return geometry, domain


def test_local_axis_regular_curvature_is_finite_and_matches_polar_limit():
    geometry, domain = _build_polar_geometry(shape=(32, 64, 4))
    coefficients = build_local_curvature_coefficients(
        geometry,
        domain,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )
    assert coefficients.shape == geometry.shape + (3,)
    assert bool(jnp.all(jnp.isfinite(coefficients)))

    # For A=B_cov/Bmag^2 and B=(0,alpha,1), the builder convention has
    # C^rho=0, C^theta=alpha^2/S^(3/2), and
    # C^zeta=alpha/S^(3/2), S=1+alpha^2*rho^2.
    rho = geometry.grid.x.centers_owned[:, None, None]
    alpha = 0.35
    S = 1.0 + alpha * alpha * rho * rho
    expected = jnp.stack(
        (
            jnp.zeros_like(rho),
            alpha * alpha / S ** 1.5 + jnp.zeros_like(rho),
            alpha / S ** 1.5 + jnp.zeros_like(rho),
        ),
        axis=-1,
    )
    # The first ring is the singular-coordinate ring; the Cartesian branch
    # should retain the same smooth finite limit as the interior. The next
    # few rings exercise the ordinary logical branch as well.
    assert bool(jnp.allclose(coefficients[0], expected[0], rtol=0.04, atol=0.01))
    assert bool(jnp.allclose(coefficients[2:8], expected[2:8], rtol=0.04, atol=0.01))
    assert bool(jnp.allclose(coefficients[8:], expected[8:], rtol=0.015, atol=0.004))


def test_local_axis_regular_curvature_direct_operator_matches_manufactured_field():
    geometry, domain = _build_polar_geometry(shape=(32, 64, 4))
    coefficients = build_local_curvature_coefficients(
        geometry,
        domain,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )

    rho, theta, zeta = jnp.meshgrid(
        geometry.grid.x.centers,
        geometry.grid.y.centers,
        geometry.grid.z.centers,
        indexing="ij",
    )
    field = (
        rho * jnp.cos(theta)
        + 0.2 * rho**2 * jnp.sin(2.0 * theta)
        + 0.3 * jnp.sin(zeta)
    )
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    stencil = build_local_stencil_from_field(field, geometry, context)
    result = local_curvature_op(
        stencil,
        geometry,
        curvature_coefficients=coefficients,
    )

    owned = geometry.layout.owned_slices_cell
    rho_owned = rho[owned]
    theta_owned = theta[owned]
    zeta_owned = zeta[owned]
    analytic_gradient = jnp.stack(
        (
            jnp.cos(theta_owned) + 0.4 * rho_owned * jnp.sin(2.0 * theta_owned),
            -rho_owned * jnp.sin(theta_owned)
            + 0.4 * rho_owned**2 * jnp.cos(2.0 * theta_owned),
            0.3 * jnp.cos(zeta_owned),
        ),
        axis=-1,
    )
    expected = jnp.einsum("...i,...i->...", coefficients, analytic_gradient)
    error = jnp.abs(result - expected)
    assert result.shape == geometry.owned_shape
    assert bool(jnp.all(jnp.isfinite(result)))
    assert float(jnp.max(error[0])) < 0.08
    assert float(jnp.max(error[1:])) < 0.08


@pytest.mark.parametrize(
    "periodic, axis, error",
    [
        ((True, True, True), (True, False, False), ValueError),
        ((False, False, True), (False, True, False), NotImplementedError),
    ],
)
def test_local_axis_regular_curvature_validation(periodic, axis, error):
    geometry, domain = _build_polar_geometry(shape=(4, 8, 4))
    with pytest.raises(error):
        build_local_curvature_coefficients(
            geometry, domain, periodic_axes=periodic, axis_regular_axes=axis
        )


def test_local_axis_regular_curvature_requires_even_poloidal_counts():
    geometry, domain = _build_polar_geometry(shape=(4, 7, 4))
    with pytest.raises(ValueError, match="even global poloidal"):
        build_local_curvature_coefficients(
            geometry,
            domain,
            periodic_axes=(False, True, True),
            axis_regular_axes=(True, False, False),
        )


def test_sharded_theta_requires_mesh_axis_name():
    geometry, domain = _build_polar_geometry(
        shape=(4, 2, 4),
        global_shape=(4, 6, 4),
        shard_counts=(1, 3, 1),
    )
    with pytest.raises(ValueError, match="theta mesh axis name"):
        build_local_curvature_coefficients(
            geometry,
            domain,
            periodic_axes=(False, True, True),
            axis_regular_axes=(True, False, False),
        )


@pytest.mark.skipif(
    jax.local_device_count() < 3,
    reason="requires three local devices for split theta half-turn assembly",
)
def test_three_theta_shards_axis_ring_matches_helical_field_target():
    local_shape = (16, 6, 4)
    global_shape = (16, 18, 4)
    shard_counts = (1, 3, 1)
    geometries = []
    domain = None
    for theta_shard in range(3):
        geometry, shard_domain = _build_polar_geometry(
            shape=local_shape,
            global_shape=global_shape,
            shard_counts=shard_counts,
            coordinate_shard_index=(0, theta_shard, 0),
            mesh_axis_names=(None, "theta", None),
        )
        geometries.append(geometry)
        if domain is None:
            domain = shard_domain

    template_leaves, geometry_tree = jax.tree_util.tree_flatten(geometries[0])
    geometry_leaves = [jax.tree_util.tree_flatten(item)[0] for item in geometries]
    array_indices = [
        index
        for index, value in enumerate(template_leaves)
        if isinstance(value, jax.Array)
    ]
    stacked_arrays = tuple(
        jnp.stack([leaves[index] for leaves in geometry_leaves])
        for index in array_indices
    )

    def _kernel(*array_values):
        leaves = list(template_leaves)
        for index, value in zip(array_indices, array_values):
            leaves[index] = value
        geometry = jax.tree_util.tree_unflatten(geometry_tree, leaves)
        return build_local_curvature_coefficients(
            geometry,
            domain,
            periodic_axes=(False, True, True),
            axis_regular_axes=(True, False, False),
        )

    coefficients = jax.pmap(
        _kernel,
        axis_name="theta",
    )(*stacked_arrays)

    alpha = 0.35
    for shard, geometry in enumerate(geometries):
        rho = geometry.grid.x.centers_owned[:, None, None]
        S = 1.0 + alpha * alpha * rho * rho
        expected = jnp.stack(
            (
                jnp.zeros_like(rho),
                alpha * alpha / S**1.5 + jnp.zeros_like(rho),
                alpha / S**1.5 + jnp.zeros_like(rho),
            ),
            axis=-1,
        )
        assert bool(jnp.all(jnp.isfinite(coefficients[shard])))
        assert bool(
            jnp.allclose(
                coefficients[shard, 0], expected[0], rtol=0.08, atol=0.02
            )
        )
        assert bool(
            jnp.allclose(
                coefficients[shard, 1:], expected[1:], rtol=0.06, atol=0.015
            )
        )
