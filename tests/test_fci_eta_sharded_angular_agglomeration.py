"""Focused tests for eta-only native angular-agglomeration lowering."""

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from drbx.geometry import StencilBuilderContext, build_shifted_torus_geometry
from drbx.geometry.fci_control_volumes import (
    build_polar_angular_agglomeration_geometry,
)
from drbx.native.fci_angular_agglomeration import (
    RLP_PACKED_FIELD_COUNT,
    assemble_local_polar_angular_agglomeration_geometry,
    build_sharded_polar_angular_agglomeration_payload,
    lower_polar_angular_agglomeration_geometry,
)
from drbx.native.fci_boundaries import (
    BC_NOFLUX,
    LocalBoundaryFaceBC3D,
    LocalControlVolumeBoundaryBC3D,
)
from drbx.native.fci_control_volume_operators import (
    CUBIC_MONOMIAL_EXPONENTS,
    control_volume_average_basis,
    monomial_basis,
)
from drbx.native.fci_operators import (
    LocalPerpLaplacianInverseSolver,
    _local_control_volume_integrated_divergence,
    build_local_control_volume_field_closure,
    build_local_control_volume_poisson_face_stencil,
    inject_owned_field_to_halo,
)
from drbx.native.fci_gmres import SolvaxGmresConfig
from drbx.native.fci_halo import (
    HaloExchange3D,
    LocalPeriodicTopologyRule3D,
    TopologyHaloFiller3D,
)
from drbx.native.fci_rlp_diffusion import apply_rlp_cell_average_prolongation
from drbx.native.fci_sharding import (
    assemble_local_fci_geometry,
    assemble_single_device_local_fci_geometry,
    build_local_fci_geometries,
    make_shard_mesh,
)


def _host(shape=(3, 8, 4)):
    u = np.linspace(0.0, 1.0, shape[0] + 1)
    theta = np.linspace(-np.pi, np.pi, shape[1] + 1)
    eta = np.linspace(-np.pi, np.pi, shape[2] + 1)
    return build_polar_angular_agglomeration_geometry(
        u,
        theta,
        eta,
        lambda points: np.maximum(np.asarray(points)[..., 0], 1.0e-14),
        quadrature_order=2,
        angular_group_size=(shape[1], 2, 1, 1)[: shape[0]],
    )


def test_eta_payload_is_global_cell_shaped_and_has_explicit_partition_spec():
    shape = (3, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 2), halo_width=1)
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host, fci.domain
    )
    assert packed.shape == shape + (RLP_PACKED_FIELD_COUNT,)
    assert descriptor.packed_cell_shape == packed.shape
    assert descriptor.cell_partition_spec == P("x", "y", "z", None)
    np.testing.assert_allclose(
        np.asarray(packed[..., 0]), host.raw_volume
    )


def test_eta_assembly_keeps_theta_owners_and_localizes_eta_owner_k():
    shape = (3, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=1)
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host, fci.domain
    )
    local_fci = assemble_single_device_local_fci_geometry(fci)
    lowered = assemble_local_polar_angular_agglomeration_geometry(
        descriptor, packed, local_fci
    )
    cells = lowered.cells
    assert tuple(cells.owner_k.shape) == shape
    np.testing.assert_array_equal(
        np.asarray(cells.owner_k), np.broadcast_to(np.arange(shape[2]), shape)
    )
    np.testing.assert_array_equal(
        np.asarray(cells.owner_j[:, :, 0]),
        np.asarray([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 4, 4, 6, 6],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ]),
    )
    assert not bool(np.any(np.asarray(cells.owner_is_remote)))


def test_moment_shared_transition_faces_are_canonical_and_reproduce_cubics():
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=2)
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host,
        fci.domain,
        compile_compact_transition_faces=True,
    )
    local_fci = assemble_single_device_local_fci_geometry(fci)
    lowered = assemble_local_polar_angular_agglomeration_geometry(
        descriptor, packed, local_fci
    )
    transition_count = int(
        np.count_nonzero(np.diff(np.asarray(host.angular_group_size)) != 0)
    )
    # Every radial q transition is represented by each theta/eta fine subface
    # exactly once.
    assert descriptor.compact_face_count == transition_count * shape[1] * shape[2]
    faces = lowered.irregular_faces
    assert np.all(np.asarray(faces.logical_axis) == 0)
    face_keys = np.stack(
        (
            np.asarray(faces.logical_face_i),
            np.asarray(faces.logical_face_j),
            np.asarray(faces.logical_face_k),
        ),
        axis=-1,
    )
    assert np.unique(face_keys, axis=0).shape[0] == faces.max_rows
    assert np.all(
        np.asarray(lowered.regular_faces.x_open_mask)[
            face_keys[:, 0], face_keys[:, 1], face_keys[:, 2]
        ]
    )

    cells = lowered.cells
    basis_average = control_volume_average_basis(
        np.asarray(cells.centroid),
        np.asarray(cells.second_moment),
        np.asarray(cells.third_moment),
        exponents=CUBIC_MONOMIAL_EXPONENTS,
    )
    coefficients = np.asarray(
        [
            -0.2 + 0.5 * index / (len(CUBIC_MONOMIAL_EXPONENTS) - 1)
            if power[2] == 0
            else 0.0
            for index, power in enumerate(CUBIC_MONOMIAL_EXPONENTS)
        ]
    )
    owner_values = basis_average @ coefficients
    owner_values = np.where(np.asarray(cells.is_active_owner), owner_values, 0.0)
    halo = inject_owned_field_to_halo(owner_values, local_fci.layout)
    closure = build_local_control_volume_field_closure(
        halo,
        lowered,
        LocalControlVolumeBoundaryBC3D.empty(max_rows=faces.max_rows),
        domain=fci.domain,
    )
    expected = monomial_basis(
        np.asarray(faces.quadrature_points),
        exponents=CUBIC_MONOMIAL_EXPONENTS,
    ) @ coefficients
    active = np.asarray(faces.quadrature_active)
    np.testing.assert_allclose(
        np.asarray(closure.face_value)[active],
        expected[active],
        rtol=2.0e-10,
        atol=2.0e-10,
    )

    compact_flux = np.linspace(-1.0, 1.0, faces.max_rows)
    zero_regular_flux = tuple(
        np.zeros(local_fci.layout.face_control_shape(axis))
        for axis in range(3)
    )
    divergence = _local_control_volume_integrated_divergence(
        zero_regular_flux,
        compact_flux,
        local_fci,
        fci.domain,
        lowered,
        volume_floor=1.0e-30,
    )
    integrated_total = np.sum(
        np.asarray(divergence) * np.asarray(cells.aggregate_volume)
    )
    assert abs(float(integrated_total)) < 2.0e-13


def test_eta_sharded_transition_payload_reports_per_shard_row_count():
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 2), halo_width=2)
    descriptor, _packed = build_sharded_polar_angular_agglomeration_payload(
        host,
        fci.domain,
        compile_compact_transition_faces=True,
    )
    transition_count = int(
        np.count_nonzero(np.diff(np.asarray(host.angular_group_size)) != 0)
    )
    assert descriptor.global_compact_face_count == (
        transition_count * shape[1] * shape[2]
    )
    assert descriptor.compact_face_count == (
        transition_count * shape[1] * (shape[2] // 2)
    )


def test_eta_assembly_is_callable_inside_shard_map():
    shape = (3, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=1)
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host, fci.domain
    )
    mesh = make_shard_mesh((1, 1, 1))
    fci_fields = jax.device_put(
        fci.cell_fields, NamedSharding(mesh, P("x", "y", "z", None))
    )
    rlp_fields = jax.device_put(
        packed, NamedSharding(mesh, descriptor.cell_partition_spec)
    )

    def kernel(fci_owned, rlp_owned):
        local_fci = assemble_local_fci_geometry(fci, fci_owned)
        local_rlp = assemble_local_polar_angular_agglomeration_geometry(
            descriptor, rlp_owned, local_fci
        )
        return local_rlp.cells.aggregate_volume

    mapped = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(
                P("x", "y", "z", None),
                descriptor.cell_partition_spec,
            ),
            out_specs=P("x", "y", "z"),
            check_vma=False,
        )
    )(fci_fields, rlp_fields)
    np.testing.assert_allclose(np.asarray(mapped), host.aggregate_chart_volume)


def test_eta_assembly_two_shard_map_matches_global_payload_when_available():
    if len(jax.devices()) < 2:
        pytest.skip("requires two JAX devices for a two-eta-shard execution test")
    shape = (3, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 2), halo_width=1)
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host, fci.domain
    )
    mesh = make_shard_mesh((1, 1, 2))
    spec = P("x", "y", "z", None)
    fci_fields = jax.device_put(fci.cell_fields, NamedSharding(mesh, spec))
    rlp_fields = jax.device_put(
        packed, NamedSharding(mesh, descriptor.cell_partition_spec)
    )

    def kernel(fci_owned, rlp_owned):
        local_fci = assemble_local_fci_geometry(fci, fci_owned)
        local_rlp = assemble_local_polar_angular_agglomeration_geometry(
            descriptor, rlp_owned, local_fci
        )
        return local_rlp.cells.aggregate_volume

    mapped = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(spec, descriptor.cell_partition_spec),
            out_specs=P("x", "y", "z"),
            check_vma=False,
        )
    )(fci_fields, rlp_fields)
    np.testing.assert_allclose(np.asarray(mapped), host.aggregate_chart_volume)


def test_eta_sharded_face_fit_matches_one_device_when_available():
    if len(jax.devices()) < 2:
        pytest.skip("requires two JAX devices for a two-eta-shard execution test")
    shape = (4, 8, 4)
    host = _host(shape)
    global_geometry = build_shifted_torus_geometry(
        shape, construct_fci_maps=False
    )
    ii, jj, kk = np.indices(shape)
    owner = np.where(
        np.asarray(host.topology.is_active_owner),
        0.2 * ii - 0.07 * jj + np.sin(2.0 * np.pi * (kk + 0.3) / shape[2]),
        0.0,
    )
    topology = TopologyHaloFiller3D(
        (LocalPeriodicTopologyRule3D((False, True, True)),)
    )

    one = build_local_fci_geometries(
        global_geometry, (1, 1, 1), halo_width=2
    )
    one_local = assemble_single_device_local_fci_geometry(one)
    one_rlp = lower_polar_angular_agglomeration_geometry(
        host, one_local, compile_direct_poisson_faces=True
    )
    one_domain = replace(one.domain, mesh_axis_names=(None, None, None))
    one_halo = inject_owned_field_to_halo(jnp.asarray(owner), one_local.layout)
    one_halo = topology(one_halo, one_domain)
    _, expected_left, expected_right = (
        build_local_control_volume_poisson_face_stencil(
            one_halo,
            one_local,
            one_domain,
            StencilBuilderContext(layout=one_domain.layout, domain=one_domain),
            one_rlp,
            LocalControlVolumeBoundaryBC3D.empty(
                max_rows=one_rlp.irregular_faces.max_rows
            ),
            owner_values_owned=jnp.asarray(owner),
            halo_exchange=HaloExchange3D(),
            topology_filler=topology,
        )
    )

    split = build_local_fci_geometries(
        global_geometry, (1, 1, 2), halo_width=2
    )
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host,
        split.domain,
        compile_compact_transition_faces=True,
    )
    mesh = make_shard_mesh((1, 1, 2))
    spec = P("x", "y", "z", None)

    def kernel(fci_owned, rlp_owned, owner_owned):
        local_fci = assemble_local_fci_geometry(split, fci_owned)
        local_rlp = assemble_local_polar_angular_agglomeration_geometry(
            descriptor, rlp_owned, local_fci
        )
        exchange = HaloExchange3D()
        field_halo = inject_owned_field_to_halo(owner_owned, local_fci.layout)
        field_halo = exchange(field_halo, split.domain)
        field_halo = topology(field_halo, split.domain)
        _, left, right = build_local_control_volume_poisson_face_stencil(
            field_halo,
            local_fci,
            split.domain,
            StencilBuilderContext(
                layout=split.domain.layout, domain=split.domain
            ),
            local_rlp,
            LocalControlVolumeBoundaryBC3D.empty(
                max_rows=descriptor.compact_face_count
            ),
            owner_values_owned=owner_owned,
            halo_exchange=exchange,
            topology_filler=topology,
        )
        return left.x, right.x

    got_left, got_right = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(spec, descriptor.cell_partition_spec, P("x", "y", "z")),
            out_specs=(P("x", "y", "z"), P("x", "y", "z")),
            check_vma=False,
        )
    )(
        jax.device_put(split.cell_fields, NamedSharding(mesh, spec)),
        jax.device_put(packed, NamedSharding(mesh, descriptor.cell_partition_spec)),
        jax.device_put(owner, NamedSharding(mesh, P("x", "y", "z"))),
    )
    transition_i = np.flatnonzero(
        np.diff(np.asarray(host.angular_group_size)) != 0
    ) + 1
    np.testing.assert_allclose(
        np.asarray(got_left)[transition_i],
        np.asarray(expected_left.x)[transition_i],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(got_right)[transition_i],
        np.asarray(expected_right.x)[transition_i],
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_eta_sharded_full_diffusion_matches_global_with_varying_weights_when_available():
    if len(jax.devices()) < 2:
        pytest.skip("requires two JAX devices for a two-eta-shard execution test")
    shape = (4, 8, 4)
    u = np.linspace(0.0, 1.0, shape[0] + 1)
    theta = np.linspace(-np.pi, np.pi, shape[1] + 1)
    eta = np.linspace(-np.pi, np.pi, shape[2] + 1)

    def varying_jacobian(points):
        points = np.asarray(points)
        return np.maximum(
            points[..., 0]
            * (1.0 + 0.17 * np.cos(points[..., 2]))
            * (1.0 + 0.08 * points[..., 0] * np.cos(points[..., 1])),
            1.0e-14,
        )

    host = build_polar_angular_agglomeration_geometry(
        u,
        theta,
        eta,
        varying_jacobian,
        quadrature_order=3,
        angular_group_size=(8, 2, 1, 1),
    )
    global_geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    owner = np.zeros(shape, dtype=np.float64)
    ii, jj, kk = np.indices(shape)
    active = np.asarray(host.topology.is_active_owner)
    owner[active] = (
        np.sin(2.0 * np.pi * (kk[active] + 0.3) / shape[2])
        + 0.2 * ii[active]
        - 0.07 * jj[active]
    )
    topology = TopologyHaloFiller3D(
        (LocalPeriodicTopologyRule3D((False, True, True)),)
    )

    def radial_noflux(layout):
        boundary = LocalBoundaryFaceBC3D.empty(layout)
        return replace(
            boundary,
            kind_x=boundary.kind_x.at[0].set(BC_NOFLUX).at[-1].set(BC_NOFLUX),
            mask_x=boundary.mask_x.at[0].set(True).at[-1].set(True),
        )

    one = build_local_fci_geometries(global_geometry, (1, 1, 1), halo_width=1)
    one_local = assemble_single_device_local_fci_geometry(one)
    one_rlp = lower_polar_angular_agglomeration_geometry(host, one_local)
    one_domain = replace(one.domain, mesh_axis_names=(None, None, None))
    one_solver = LocalPerpLaplacianInverseSolver(
        geometry=one_local,
        domain=one_domain,
        control_volume_geometry=one_rlp,
        control_volume_boundary_bc=LocalControlVolumeBoundaryBC3D.empty(),
        halo_exchange=HaloExchange3D(),
        topology_filler=topology,
        stencil_builder_context=StencilBuilderContext(
            layout=one_domain.layout, domain=one_domain
        ),
        config=SolvaxGmresConfig(regularization_epsilon=0.0),
    )
    face_bc = radial_noflux(one_domain.layout)
    expected = one_solver.apply_positive_operator(
        jnp.asarray(owner), face_bc=face_bc, project_mean_zero=False
    )

    split = build_local_fci_geometries(global_geometry, (1, 1, 2), halo_width=1)
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host, split.domain
    )
    mesh = make_shard_mesh((1, 1, 2))
    spec = P("x", "y", "z", None)
    fci_fields = jax.device_put(split.cell_fields, NamedSharding(mesh, spec))
    rlp_fields = jax.device_put(
        packed, NamedSharding(mesh, descriptor.cell_partition_spec)
    )
    owner_fields = jax.device_put(owner, NamedSharding(mesh, P("x", "y", "z")))

    def kernel(fci_owned, rlp_owned, owner_owned):
        local_fci = assemble_local_fci_geometry(split, fci_owned)
        local_rlp = assemble_local_polar_angular_agglomeration_geometry(
            descriptor, rlp_owned, local_fci
        )
        solver = LocalPerpLaplacianInverseSolver(
            geometry=local_fci,
            domain=split.domain,
            control_volume_geometry=local_rlp,
            control_volume_boundary_bc=LocalControlVolumeBoundaryBC3D.empty(),
            halo_exchange=HaloExchange3D(),
            topology_filler=topology,
            stencil_builder_context=StencilBuilderContext(
                layout=split.domain.layout, domain=split.domain
            ),
            config=SolvaxGmresConfig(regularization_epsilon=0.0),
        )
        return solver.apply_positive_operator(
            owner_owned,
            face_bc=radial_noflux(split.domain.layout),
            project_mean_zero=False,
        )

    got = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(spec, descriptor.cell_partition_spec, P("x", "y", "z")),
            out_specs=P("x", "y", "z"),
            check_vma=False,
        )
    )(fci_fields, rlp_fields, owner_fields)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=2e-11, atol=2e-11)

    # The host quadrature volume intentionally differs from midpoint J; exact
    # owner balance therefore also verifies the RLP-specific fine denominator.
    owner_volume = np.asarray(one_rlp.cells.aggregate_volume)
    np.testing.assert_allclose(
        np.sum(owner_volume * np.asarray(got)), 0.0, rtol=0.0, atol=2e-10
    )


def test_four_eta_shards_with_one_local_plane_apply_two_hop_reconstruction_when_available():
    if len(jax.devices()) < 4:
        pytest.skip("requires four JAX devices for one-plane eta shards")
    shape = (4, 8, 4)
    host = _host(shape)
    global_geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    owner = np.zeros(shape, dtype=np.float64)
    active = np.asarray(host.topology.is_active_owner)
    ii, jj, kk = np.indices(shape)
    owner[active] = 0.3 * ii[active] - 0.11 * jj[active] + np.sin(
        2.0 * np.pi * (kk[active] + 0.2) / shape[2]
    )
    one = build_local_fci_geometries(global_geometry, (1, 1, 1), halo_width=1)
    one_local = assemble_single_device_local_fci_geometry(one)
    one_rlp = lower_polar_angular_agglomeration_geometry(host, one_local)
    expected = apply_rlp_cell_average_prolongation(
        jnp.asarray(owner), one_rlp.diffusion_prolongation
    )

    split = build_local_fci_geometries(global_geometry, (1, 1, 4), halo_width=1)
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host, split.domain
    )
    mesh = make_shard_mesh((1, 1, 4))
    spec = P("x", "y", "z", None)

    def kernel(fci_owned, rlp_owned, owner_owned):
        local_fci = assemble_local_fci_geometry(split, fci_owned)
        local_rlp = assemble_local_polar_angular_agglomeration_geometry(
            descriptor, rlp_owned, local_fci
        )
        return apply_rlp_cell_average_prolongation(
            owner_owned,
            local_rlp.diffusion_prolongation,
            domain=split.domain,
        )

    got = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(spec, descriptor.cell_partition_spec, P("x", "y", "z")),
            out_specs=P("x", "y", "z"),
            check_vma=False,
        )
    )(
        jax.device_put(split.cell_fields, NamedSharding(mesh, spec)),
        jax.device_put(packed, NamedSharding(mesh, descriptor.cell_partition_spec)),
        jax.device_put(owner, NamedSharding(mesh, P("x", "y", "z"))),
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("shard_counts", [(2, 1, 1), (1, 2, 1)])
def test_eta_payload_rejects_radial_or_theta_sharding(shard_counts):
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, shard_counts, halo_width=1)
    with pytest.raises(ValueError, match="eta sharding only|x/theta sharding"):
        build_sharded_polar_angular_agglomeration_payload(host, fci.domain)
