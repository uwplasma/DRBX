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
    _fit_radial_curvature_face_weights,
    _fit_transition_face_value_weights,
    assemble_local_polar_angular_agglomeration_geometry,
    build_sharded_polar_angular_agglomeration_payload,
    lower_polar_angular_agglomeration_geometry,
)
from drbx.native.fci_boundaries import (
    BC_NOFLUX,
    LocalBoundaryFaceBC3D,
    LocalControlVolumeBoundaryBC3D,
    LocalRegularBoundaryMomentClosure3D,
)
from drbx.native.fci_control_volume_operators import (
    CUBIC_MONOMIAL_EXPONENTS,
    control_volume_average_basis,
    monomial_basis,
    monomial_exponents,
)
from drbx.native.fci_operators import (
    LocalPerpLaplacianInverseSolver,
    _local_control_volume_integrated_divergence,
    build_local_control_volume_field_closure,
    build_local_control_volume_poisson_face_stencil,
    build_local_radial_curvature_face_states,
    inject_owned_field_to_halo,
)
from drbx.native.fci_gmres import SolvaxGmresConfig
from drbx.native.fci_halo import (
    HaloLayout3D,
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


def test_transition_face_fit_falls_back_before_exceeding_weight_l1_limit():
    coordinates = (-1.0, -0.3, 0.4, 1.0)
    centroid = np.asarray(
        [
            (x, y, z)
            for x in coordinates
            for y in coordinates
            for z in coordinates
        ],
        dtype=np.float64,
    )
    second_moment = np.zeros((centroid.shape[0], 3, 3), dtype=np.float64)
    third_moment = np.zeros((centroid.shape[0], 3, 3, 3), dtype=np.float64)
    point = np.asarray(((1.5, 0.0, 0.0),), dtype=np.float64)
    common = {
        "origin": np.zeros((3,), dtype=np.float64),
        "scale": np.ones((3,), dtype=np.float64),
    }

    cubic_weights, cubic_order, *_ = _fit_transition_face_value_weights(
        centroid,
        second_moment,
        third_moment,
        point,
        **common,
    )
    bounded_weights, bounded_order, *_ = _fit_transition_face_value_weights(
        centroid,
        second_moment,
        third_moment,
        point,
        max_weight_l1=8.0,
        **common,
    )

    assert cubic_order == 3
    assert float(np.sum(np.abs(cubic_weights))) > 8.0
    assert bounded_order == 2
    assert float(np.sum(np.abs(bounded_weights))) <= 8.0

    with pytest.raises(ValueError, match="finite and positive"):
        _fit_transition_face_value_weights(
            centroid,
            second_moment,
            third_moment,
            point,
            max_weight_l1=0.0,
            **common,
        )


def test_radial_curvature_adaptive_fit_replaces_unstable_last_donor():
    centroid = np.asarray(
        [
            (x, y, z)
            for x in (-0.3, -0.1, 0.1, 0.3)
            for y in (-0.1, 0.1)
            for z in np.linspace(-0.4, 0.4, 8)
        ]
        + [(0.0, 0.6, 0.0)],
        dtype=np.float64,
    )
    second_moment = np.zeros((65, 3, 3), dtype=np.float64)
    second_moment[:, 1, 1] = 1.0e-6 * (
        1.0 + 0.2 * np.cos(np.arange(65) * 1.7)
    )
    third_moment = np.zeros((65, 3, 3, 3), dtype=np.float64)
    sort_order = np.argsort(np.sum(centroid**2, axis=1), kind="stable")
    centroid = centroid[sort_order]
    second_moment = second_moment[sort_order]
    third_moment = third_moment[sort_order]
    points = np.asarray(
        ((0.02, 0.02, 0.02), (-0.02, 0.02, -0.02),
         (0.02, -0.02, -0.02), (-0.02, -0.02, 0.02)),
        dtype=np.float64,
    )
    common = dict(
        origin=np.zeros(3),
        scale=np.ones(3),
        outward=np.asarray((1.0, 0.0, 0.0)),
        max_equations=64,
    )

    displacement = centroid[:64]
    distance2 = np.einsum("ni,ni->n", displacement, displacement)
    base_weight = 1.0 / np.maximum(distance2, 0.25)
    signed_normal = centroid[:64, 0]
    nearest_observation_weights = (
        base_weight,
        base_weight * np.where(signed_normal <= 0.0, 4.0, 1.0),
        base_weight * np.where(signed_normal >= 0.0, 4.0, 1.0),
    )
    nearest_fits = tuple(
        _fit_transition_face_value_weights(
            centroid[:64],
            second_moment[:64],
            third_moment[:64],
            points,
            origin=common["origin"],
            scale=common["scale"],
            observation_weight=observation_weight,
            max_weight_l1=8.0,
        )
        for observation_weight in nearest_observation_weights
    )
    assert all(fit[1] < 2 for fit in nearest_fits)

    selected, fits = _fit_radial_curvature_face_weights(
        centroid, second_moment, third_moment, points, **common
    )
    assert selected.size == 64
    assert selected[-1] == 64
    np.testing.assert_array_equal(selected[:-1], np.arange(63))
    assert all(fit[1] >= 2 for fit in fits)
    assert all(
        float(np.max(np.sum(np.abs(fit[0]), axis=-1))) <= 8.0
        for fit in fits
    )
    quadratic_exponents = monomial_exponents(2)
    selected_basis = control_volume_average_basis(
        centroid[selected],
        second_moment[selected],
        third_moment[selected],
        origin=common["origin"],
        scale=common["scale"],
        exponents=quadratic_exponents,
    )
    quadratic_target = monomial_basis(
        (points - common["origin"][None, :]) / common["scale"][None, :],
        exponents=quadratic_exponents,
    )
    for fit in fits:
        np.testing.assert_allclose(
            fit[0] @ selected_basis,
            quadratic_target,
            rtol=0.0,
            atol=1.0e-9,
        )

    good_selected, good_fits = _fit_radial_curvature_face_weights(
        centroid[selected],
        second_moment[selected],
        third_moment[selected],
        points,
        **common,
    )
    np.testing.assert_array_equal(good_selected, np.arange(64))
    assert all(fit[1] >= 2 for fit in good_fits)

    with pytest.raises(ValueError, match="quality exhaustion"):
        _fit_radial_curvature_face_weights(
            centroid[:64],
            second_moment[:64],
            third_moment[:64],
            points,
            **common,
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


def test_opt_in_regular_boundary_closure_reproduces_cubic_radial_moments():
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(
        geometry,
        (1, 1, 1),
        halo_width=1,
        axis_regular_axes=(True, False, False),
    )
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host,
        fci.domain,
        compile_regular_boundary_closure=True,
    )
    local_fci = assemble_single_device_local_fci_geometry(fci)
    lowered = assemble_local_polar_angular_agglomeration_geometry(
        descriptor, packed, local_fci
    )
    closure = lowered.regular_boundary_closure
    assert closure is not None
    assert not np.any(np.asarray(closure.x_valid[0]))
    assert np.all(np.asarray(closure.x_valid[-1]))
    assert not np.any(np.asarray(closure.y_valid))
    assert not np.any(np.asarray(closure.z_valid))

    wall = float(
        host.radial_centers[-1] + 0.5 * host.radial_widths[-1]
    )
    indices = np.asarray((shape[0] - 1, shape[0] - 2, shape[0] - 3))
    centroid = np.asarray(host.raw_radial_centroid)[indices]
    central_second = np.asarray(host.raw_radial_second_moment)[indices]
    central_third = np.asarray(host.raw_radial_third_moment)[indices]
    raw_moments = (
        np.ones_like(centroid),
        centroid,
        central_second + centroid**2,
        central_third + 3.0 * centroid * central_second + centroid**3,
    )
    face_weights = np.asarray(closure.x_face_weights[-1])
    owner_weights = np.asarray(closure.x_owner_weights[-1])
    first_centroid = centroid[0]
    for power, averages in enumerate(raw_moments):
        observations = np.concatenate(
            (
                np.full((1,) + averages.shape[1:], wall**power),
                averages,
            ),
            axis=0,
        )
        face_derivative = np.einsum(
            "...m,m...->...", face_weights, observations
        )
        owner_derivative = np.einsum(
            "...m,m...->...", owner_weights, observations
        )
        expected_face = 0.0 if power == 0 else power * wall ** (power - 1)
        expected_owner = (
            np.zeros_like(first_centroid)
            if power == 0
            else power * first_centroid ** (power - 1)
        )
        np.testing.assert_allclose(face_derivative, expected_face, atol=2e-11)
        np.testing.assert_allclose(owner_derivative, expected_owner, atol=2e-11)


def test_regular_boundary_closure_remains_absent_without_opt_in():
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=1)
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host, fci.domain
    )
    lowered = assemble_local_polar_angular_agglomeration_geometry(
        descriptor,
        packed,
        assemble_single_device_local_fci_geometry(fci),
    )
    assert lowered.regular_boundary_closure is None


def test_empty_regular_boundary_closure_is_legal_on_one_plane_eta_axis():
    closure = LocalRegularBoundaryMomentClosure3D.empty(
        HaloLayout3D((4, 8, 1), halo_width=1)
    )
    with pytest.raises(ValueError, match="at least three owned cells"):
        replace(
            closure,
            z_valid=closure.z_valid.at[-1].set(True),
        )


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


def test_preferred_direct_face_functionals_matches_legacy_one_device_lowering():
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=2)
    local_fci = assemble_single_device_local_fci_geometry(fci)
    legacy = lower_polar_angular_agglomeration_geometry(
        host, local_fci, compile_direct_poisson_faces=True
    )
    preferred = lower_polar_angular_agglomeration_geometry(
        host, local_fci, compile_direct_face_functionals=True
    )
    for name in (
        "global_face_id",
        "logical_axis",
        "logical_face_i",
        "logical_face_j",
        "logical_face_k",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(legacy.irregular_faces, name)),
            np.asarray(getattr(preferred.irregular_faces, name)),
        )
    for name in (
        "value_weights",
        "upwind_minus_value_weights",
        "upwind_plus_value_weights",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(legacy.face_functionals, name)),
            np.asarray(getattr(preferred.face_functionals, name)),
            rtol=0.0,
            atol=0.0,
        )


def test_preferred_direct_face_functionals_matches_legacy_sharded_payload():
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 2), halo_width=2)
    legacy, legacy_packed = build_sharded_polar_angular_agglomeration_payload(
        host, fci.domain, compile_compact_transition_faces=True
    )
    preferred, preferred_packed = build_sharded_polar_angular_agglomeration_payload(
        host, fci.domain, compile_direct_face_functionals=True
    )
    assert legacy.global_compact_face_count == preferred.global_compact_face_count
    assert legacy.compact_face_count == preferred.compact_face_count
    np.testing.assert_array_equal(np.asarray(legacy_packed), np.asarray(preferred_packed))


def test_direct_face_functionals_rejects_conflicting_legacy_switches():
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=1)
    local_fci = assemble_single_device_local_fci_geometry(fci)
    with pytest.raises(ValueError, match="disagrees"):
        lower_polar_angular_agglomeration_geometry(
            host,
            local_fci,
            compile_direct_poisson_faces=True,
            compile_direct_face_functionals=False,
        )
    split = build_local_fci_geometries(geometry, (1, 1, 2), halo_width=1)
    with pytest.raises(ValueError, match="disagrees"):
        build_sharded_polar_angular_agglomeration_payload(
            host,
            split.domain,
            compile_compact_transition_faces=True,
            compile_direct_face_functionals=False,
        )


def test_direct_face_band_radius_is_nonnegative_for_both_builders():
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=1)
    local_fci = assemble_single_device_local_fci_geometry(fci)
    with pytest.raises(ValueError, match="direct_face_band_radius"):
        lower_polar_angular_agglomeration_geometry(
            host,
            local_fci,
            compile_direct_face_functionals=True,
            direct_face_band_radius=-1,
        )
    split = build_local_fci_geometries(geometry, (1, 1, 2), halo_width=1)
    with pytest.raises(ValueError, match="direct_face_band_radius"):
        build_sharded_polar_angular_agglomeration_payload(
            host,
            split.domain,
            compile_direct_face_functionals=True,
            direct_face_band_radius=-1,
        )


def test_direct_face_band_radius_requires_direct_compilation():
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    one = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=1)
    with pytest.raises(ValueError, match="requires direct face compilation"):
        lower_polar_angular_agglomeration_geometry(
            host,
            assemble_single_device_local_fci_geometry(one),
            direct_face_band_radius=1,
        )
    split = build_local_fci_geometries(geometry, (1, 1, 2), halo_width=1)
    with pytest.raises(ValueError, match="requires direct face compilation"):
        build_sharded_polar_angular_agglomeration_payload(
            host,
            split.domain,
            direct_face_band_radius=1,
        )


def test_direct_face_band_radius_one_covers_neighboring_rows_one_device_and_eta_shards():
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    one = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=2)
    one_local = assemble_single_device_local_fci_geometry(one)
    lowered = lower_polar_angular_agglomeration_geometry(
        host,
        one_local,
        compile_direct_face_functionals=True,
        direct_face_band_radius=1,
    )
    faces = lowered.irregular_faces
    assert faces.max_rows == (shape[0] - 1) * shape[1] * shape[2]
    assert set(np.unique(np.asarray(faces.logical_face_i))) == {1, 2, 3}

    split = build_local_fci_geometries(geometry, (1, 1, 2), halo_width=2)
    descriptor, _packed = build_sharded_polar_angular_agglomeration_payload(
        host,
        split.domain,
        compile_direct_face_functionals=True,
        direct_face_band_radius=1,
    )
    assert descriptor.global_compact_face_count == faces.max_rows
    assert descriptor.compact_face_count == faces.max_rows // 2


def test_lean_radial_curvature_rows_match_full_generic_fit_and_shard_counts():
    shape = (4, 8, 8)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    one = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=2)
    one_local = assemble_single_device_local_fci_geometry(one)
    lowered = lower_polar_angular_agglomeration_geometry(
        host,
        one_local,
        compile_direct_face_functionals=True,
        direct_face_band_radius=shape[0],
        compile_radial_curvature_faces=True,
    )
    generic_faces = lowered.irregular_faces
    generic = lowered.face_functionals
    lean = lowered.radial_curvature_faces
    assert lean.max_rows == generic_faces.max_rows
    for name in ("logical_face_i", "logical_face_j", "logical_face_k"):
        np.testing.assert_array_equal(
            np.asarray(getattr(lean, name)),
            np.asarray(getattr(generic_faces, name)),
        )
    for generic_name, lean_name in (
        ("value_weights", "centered_weights"),
        ("upwind_minus_value_weights", "minus_weights"),
        ("upwind_plus_value_weights", "plus_weights"),
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(generic, generic_name))[:, 0],
            np.asarray(getattr(lean, lean_name)),
            rtol=0.0,
            atol=0.0,
        )
    for name in ("centered_weights", "minus_weights", "plus_weights"):
        row_l1 = np.sum(np.abs(np.asarray(getattr(lean, name))), axis=-1)
        assert float(np.max(row_l1)) <= 8.0
    lean_bytes = sum(np.asarray(value).nbytes for value in lean.tree_flatten()[0])
    generic_bytes = sum(
        np.asarray(value).nbytes
        for value in (
            *generic_faces.tree_flatten()[0],
            *generic.tree_flatten()[0],
        )
    )
    assert lean_bytes < 0.25 * generic_bytes

    split = build_local_fci_geometries(geometry, (1, 1, 2), halo_width=2)
    descriptor, _packed = build_sharded_polar_angular_agglomeration_payload(
        host,
        split.domain,
        compile_radial_curvature_faces=True,
    )
    assert descriptor.global_radial_curvature_face_count == lean.max_rows
    assert descriptor.radial_curvature_face_count == lean.max_rows // 2


def test_eta_sharded_lean_curvature_traces_match_one_device_across_eta_seam():
    if len(jax.devices()) < 2:
        pytest.skip("requires two JAX devices for a two-eta-shard execution test")
    shape = (4, 8, 8)
    host = _host(shape)
    global_geometry = build_shifted_torus_geometry(
        shape, construct_fci_maps=False
    )
    ii, jj, kk = np.indices(shape)
    active = np.asarray(host.topology.is_active_owner)
    owner = np.zeros(shape + (3,), dtype=np.float64)
    owner[..., 0] = 0.13 * ii + 0.07 * jj + np.sin(
        2.0 * np.pi * (kk + 0.37) / shape[2]
    )
    owner[..., 1] = -0.11 * ii + 0.05 * jj + np.cos(
        2.0 * np.pi * (kk + 0.19) / shape[2]
    )
    owner[..., 2] = 0.03 * ii - 0.17 * jj + np.sin(
        4.0 * np.pi * (kk + 0.23) / shape[2]
    )
    owner = np.where(active[..., None], owner, 0.0)
    topology = TopologyHaloFiller3D(
        (LocalPeriodicTopologyRule3D((False, True, True)),)
    )

    one = build_local_fci_geometries(global_geometry, (1, 1, 1), halo_width=2)
    one_local = assemble_single_device_local_fci_geometry(one)
    one_domain = replace(one.domain, mesh_axis_names=(None, None, None))
    one_rlp = lower_polar_angular_agglomeration_geometry(
        host, one_local, compile_radial_curvature_faces=True
    )
    expected = build_local_radial_curvature_face_states(
        jnp.asarray(owner),
        one_local,
        one_domain,
        one_rlp,
        halo_exchange=HaloExchange3D(),
        topology_filler=topology,
    )
    assert np.any(np.asarray(one_rlp.radial_curvature_faces.logical_face_k) == 0)
    assert np.any(np.asarray(one_rlp.radial_curvature_faces.logical_face_k) == shape[2] - 1)

    split = build_local_fci_geometries(global_geometry, (1, 1, 2), halo_width=2)
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host, split.domain, compile_radial_curvature_faces=True
    )
    mesh = make_shard_mesh((1, 1, 2))
    cell_spec = P("x", "y", "z", None)
    fci_fields = jax.device_put(split.cell_fields, NamedSharding(mesh, cell_spec))
    rlp_fields = jax.device_put(
        packed, NamedSharding(mesh, descriptor.cell_partition_spec)
    )
    owner_fields = jax.device_put(
        owner, NamedSharding(mesh, P("x", "y", "z", None))
    )

    def kernel(fci_owned, rlp_owned, owner_owned):
        local_fci = assemble_local_fci_geometry(split, fci_owned)
        local_rlp = assemble_local_polar_angular_agglomeration_geometry(
            descriptor, rlp_owned, local_fci
        )
        states = build_local_radial_curvature_face_states(
            owner_owned,
            local_fci,
            split.domain,
            local_rlp,
            halo_exchange=HaloExchange3D(),
            topology_filler=topology,
        )
        return states.centered, states.minus, states.plus, states.valid

    got = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(cell_spec, descriptor.cell_partition_spec, P("x", "y", "z", None)),
            out_specs=(P("z"), P("z"), P("z"), P("z")),
            check_vma=False,
        )
    )(fci_fields, rlp_fields, owner_fields)
    for actual, reference in zip(
        got,
        (expected.centered, expected.minus, expected.plus, expected.valid),
    ):
        if reference.dtype == bool:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
        else:
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(reference), rtol=2e-12, atol=2e-12
            )
    assert np.all(np.asarray(expected.valid))
    assert np.all(np.asarray(got[3]))


def test_direct_face_geometry_sampler_uses_logical_points_and_populates_metrics():
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    split = build_local_fci_geometries(geometry, (1, 1, 2), halo_width=1)
    seen = []

    def analytic_values(points):
        points = np.asarray(points)
        u = points[..., 0]
        diagonal = np.stack((1.0 + u, 2.0 + 0.5 * u, 3.0 + u), axis=-1)
        g_contra = np.zeros(points.shape[:-1] + (3, 3))
        g_contra[..., np.arange(3), np.arange(3)] = diagonal
        g_cov = np.zeros_like(g_contra)
        g_cov[..., np.arange(3), np.arange(3)] = 1.0 / diagonal
        b_contra = np.stack(
            (0.2 + 0.1 * u, 0.3 + 0.05 * points[..., 1], 1.5 + 0.1 * points[..., 2]),
            axis=-1,
        )
        return {
            "J": 2.0 + u,
            "g_contra": g_contra,
            "g_cov": g_cov,
            "B_contra": b_contra,
            "Bmag": 2.0 + 0.25 * u,
        }

    def sampler(points):
        seen.append(np.asarray(points).copy())
        return analytic_values(points)

    origins = (0.37, -0.21)
    descriptor, _packed = build_sharded_polar_angular_agglomeration_payload(
        host,
        split.domain,
        compile_direct_face_functionals=True,
        direct_face_geometry_sampler=sampler,
        direct_face_angular_origins=origins,
    )
    assert len(seen) == 1
    logical = seen[0]
    assert logical.shape == (descriptor.global_compact_face_count, 4, 3)
    compact = descriptor.compact_transition_payload
    faces = compact.faces
    assert not np.allclose(faces["quadrature_points"][:, 0], logical)

    sampled = analytic_values(logical)
    np.testing.assert_allclose(faces["J"][:, 0], sampled["J"])
    np.testing.assert_allclose(faces["g_contra"][:, 0], sampled["g_contra"])
    np.testing.assert_allclose(faces["g_cov"][:, 0], sampled["g_cov"])
    np.testing.assert_allclose(faces["B_contra"][:, 0], sampled["B_contra"])
    np.testing.assert_allclose(faces["Bmag"][:, 0], sampled["Bmag"])
    expected_projector = sampled["g_contra"] - np.einsum(
        "...i,...j->...ij",
        sampled["B_contra"] / sampled["Bmag"][..., None],
        sampled["B_contra"] / sampled["Bmag"][..., None],
    )
    np.testing.assert_allclose(faces["projector"][:, 0], expected_projector)

    gauss = 1.0 / np.sqrt(3.0)
    first_j = int(faces["logical_face_j"][0])
    first_k = int(faces["logical_face_k"][0])
    dtheta = host.theta_period / shape[1]
    deta = host.eta_period / shape[2]
    np.testing.assert_allclose(
        logical[0, :, 1], origins[0] + (first_j + 0.5) * dtheta
        + 0.5 * dtheta * np.asarray((-gauss, -gauss, gauss, gauss))
    )
    np.testing.assert_allclose(
        logical[0, :, 2], origins[1] + (first_k + 0.5) * deta
        + 0.5 * deta * np.asarray((-gauss, gauss, -gauss, gauss))
    )
    assert len(seen) == 1
    assert descriptor.compact_face_count == descriptor.global_compact_face_count // 2

    one = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=1)
    lowered = lower_polar_angular_agglomeration_geometry(
        host,
        assemble_single_device_local_fci_geometry(one),
        compile_direct_face_functionals=True,
        direct_face_geometry_sampler=sampler,
        direct_face_angular_origins=origins,
    )
    assert len(seen) == 2
    local_faces = lowered.irregular_faces
    np.testing.assert_allclose(np.asarray(local_faces.J)[:, 0], sampled["J"])
    np.testing.assert_allclose(
        np.asarray(local_faces.projector)[:, 0], expected_projector
    )


def test_direct_face_geometry_sampler_requires_direct_compilation():
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=1)
    with pytest.raises(ValueError, match="require compile_direct_face_functionals"):
        build_sharded_polar_angular_agglomeration_payload(
            host,
            fci.domain,
            direct_face_angular_origins=(0.0, 0.0),
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


def test_four_eta_shards_lower_regular_boundary_closure_with_one_local_plane_when_available():
    if len(jax.devices()) < 4:
        pytest.skip("requires four JAX devices for one-plane eta shards")
    shape = (4, 8, 4)
    host = _host(shape)
    global_geometry = build_shifted_torus_geometry(
        shape, construct_fci_maps=False
    )

    one = build_local_fci_geometries(
        global_geometry,
        (1, 1, 1),
        halo_width=1,
        axis_regular_axes=(True, False, False),
    )
    one_descriptor, one_packed = (
        build_sharded_polar_angular_agglomeration_payload(
            host,
            one.domain,
            compile_regular_boundary_closure=True,
        )
    )
    one_rlp = assemble_local_polar_angular_agglomeration_geometry(
        one_descriptor,
        one_packed,
        assemble_single_device_local_fci_geometry(one),
    )
    expected = one_rlp.regular_boundary_closure
    assert expected is not None

    split = build_local_fci_geometries(
        global_geometry,
        (1, 1, 4),
        halo_width=1,
        axis_regular_axes=(True, False, False),
    )
    descriptor, packed = build_sharded_polar_angular_agglomeration_payload(
        host,
        split.domain,
        compile_regular_boundary_closure=True,
    )
    mesh = make_shard_mesh((1, 1, 4))
    spec = P("x", "y", "z", None)

    def kernel(fci_owned, rlp_owned):
        local_fci = assemble_local_fci_geometry(split, fci_owned)
        local_rlp = assemble_local_polar_angular_agglomeration_geometry(
            descriptor, rlp_owned, local_fci
        )
        closure = local_rlp.regular_boundary_closure
        assert closure is not None
        return (
            closure.x_face_weights,
            closure.x_owner_weights,
            closure.x_valid,
        )

    got_face, got_owner, got_valid = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(spec, descriptor.cell_partition_spec),
            out_specs=(spec, spec, P("x", "y", "z")),
            check_vma=False,
        )
    )(
        jax.device_put(split.cell_fields, NamedSharding(mesh, spec)),
        jax.device_put(packed, NamedSharding(mesh, descriptor.cell_partition_spec)),
    )
    np.testing.assert_allclose(
        np.asarray(got_face), np.asarray(expected.x_face_weights), atol=2e-11
    )
    np.testing.assert_allclose(
        np.asarray(got_owner), np.asarray(expected.x_owner_weights), atol=2e-11
    )
    np.testing.assert_array_equal(
        np.asarray(got_valid), np.asarray(expected.x_valid)
    )


@pytest.mark.parametrize("shard_counts", [(2, 1, 1), (1, 2, 1)])
def test_eta_payload_rejects_radial_or_theta_sharding(shard_counts):
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, shard_counts, halo_width=1)
    with pytest.raises(ValueError, match="eta sharding only|x/theta sharding"):
        build_sharded_polar_angular_agglomeration_payload(host, fci.domain)
