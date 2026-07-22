"""Focused contracts for global agglomeration and direct moment functionals."""

from __future__ import annotations

import argparse
import time

import numpy as np

from drbx.geometry.fci_control_volumes import (
    build_global_control_volume_topology,
    combine_volume_moments,
    compile_local_control_volume_geometry,
)
from drbx.native.fci_boundaries import CV_RECONSTRUCTION_EQUATION_CELL
from drbx.native.fci_control_volume_operators import (
    CUBIC_MONOMIAL_EXPONENTS,
    cubic_control_volume_average_basis,
    cubic_dense_face_targets,
    cubic_monomial_basis,
    evaluate_local_face_functional,
    pack_local_face_functionals,
    precompute_local_face_functional,
)


def _unit_faces(shape: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
    return (
        np.ones((shape[0] + 1, shape[1], shape[2]), dtype=np.float64),
        np.ones((shape[0], shape[1] + 1, shape[2]), dtype=np.float64),
        np.ones((shape[0], shape[1], shape[2] + 1), dtype=np.float64),
    )


def _raw_geometry(shape: tuple[int, int, int]):
    coordinate = np.stack(
        np.meshgrid(
            *(np.arange(size, dtype=np.float64) + 0.5 for size in shape),
            indexing="ij",
        ),
        axis=-1,
    )
    volume = np.ones(shape, dtype=np.float64)
    second = np.zeros(shape + (3, 3), dtype=np.float64)
    second[..., 0, 0] = 1.0 / 12.0
    second[..., 1, 1] = 1.0 / 12.0
    second[..., 2, 2] = 1.0 / 12.0
    third = np.zeros(shape + (3, 3, 3), dtype=np.float64)
    fraction = np.ones(shape, dtype=np.float64)
    return volume, coordinate, second, third, fraction


def test_global_agglomeration_is_direct_and_conservative() -> None:
    shape = (4, 4, 4)
    volume, centroid, second, third, fraction = _raw_geometry(shape)
    source = (1, 1, 1)
    volume[source] = 0.2
    fraction[source] = 0.2
    topology = build_global_control_volume_topology(
        raw_volume=volume,
        raw_centroid=centroid,
        raw_second_moment=second,
        raw_third_moment=third,
        fluid_volume_fraction=fraction,
        face_open_measure=_unit_faces(shape),
    )
    owner = tuple(topology.owner_index[source])
    assert owner == (0, 1, 1)
    assert bool(topology.is_merge_source[source])
    assert not bool(topology.is_active_owner[source])
    assert bool(topology.is_active_owner[owner])
    assert topology.aggregate_id[source] == topology.aggregate_id[owner]
    assert np.isclose(
        np.sum(topology.aggregate_volume), np.sum(volume), rtol=0.0, atol=1.0e-14
    )
    expected = combine_volume_moments(
        np.asarray((volume[owner], volume[source])),
        np.asarray((centroid[owner], centroid[source])),
        np.asarray((second[owner], second[source])),
        np.asarray((third[owner], third[source])),
    )
    np.testing.assert_allclose(topology.aggregate_volume[owner], expected[0])
    np.testing.assert_allclose(topology.aggregate_centroid[owner], expected[1])
    np.testing.assert_allclose(topology.aggregate_second_moment[owner], expected[2])
    np.testing.assert_allclose(topology.aggregate_third_moment[owner], expected[3])
    assert not np.any(
        (topology.face_minus_aggregate_id >= 0)
        & (topology.face_minus_aggregate_id == topology.face_plus_aggregate_id)
    )
    assert topology.face_storage_index.shape == (
        topology.face_id.size,
        3,
    )


def test_global_topology_compiles_identically_across_shards() -> None:
    shape = (4, 4, 4)
    volume, centroid, second, third, fraction = _raw_geometry(shape)
    volume[1, 1, 1] = 0.2
    fraction[1, 1, 1] = 0.2
    topology = build_global_control_volume_topology(
        raw_volume=volume,
        raw_centroid=centroid,
        raw_second_moment=second,
        raw_third_moment=third,
        fluid_volume_fraction=fraction,
        face_open_measure=_unit_faces(shape),
    )
    whole = compile_local_control_volume_geometry(
        topology, shard_index=(0, 0, 0), shard_counts=(1, 1, 1)
    )
    lower = compile_local_control_volume_geometry(
        topology, shard_index=(0, 0, 0), shard_counts=(1, 1, 2)
    )
    upper = compile_local_control_volume_geometry(
        topology, shard_index=(0, 0, 1), shard_counts=(1, 1, 2)
    )
    assert set(np.unique(whole.local_aggregate_id)) == set(
        np.unique(np.concatenate((lower.local_aggregate_id.ravel(), upper.local_aggregate_id.ravel())))
    )
    assert set(whole.local_face_id) == set(
        np.concatenate((lower.local_face_id, upper.local_face_id))
    )
    for local in (whole, lower, upper):
        assert local.local_face_storage_index.shape == (
            local.local_face_id.size,
            3,
        )
        assert local.local_face_axis.shape == local.local_face_id.shape


def test_periodic_seam_is_one_global_interface_and_can_merge_across_it() -> None:
    shape = (4, 2, 2)
    volume, centroid, second, third, fraction = _raw_geometry(shape)
    source = (0, 0, 0)
    volume[source] = 0.2
    fraction[source] = 0.2
    faces = list(_unit_faces(shape))
    faces[0].fill(0.0)
    faces[0][0, 0, 0] = 2.0
    faces[0][-1, 0, 0] = 2.0
    topology = build_global_control_volume_topology(
        raw_volume=volume,
        raw_centroid=centroid,
        raw_second_moment=second,
        raw_third_moment=third,
        fluid_volume_fraction=fraction,
        face_open_measure=tuple(faces),
        periodic_axes=(True, False, False),
        coordinate_periods=(4.0, 1.0, 1.0),
    )
    assert tuple(topology.owner_index[source]) == (3, 0, 0)
    seam_owner_ids = {
        int(topology.aggregate_id[3, 0, 0]),
        int(topology.aggregate_id[0, 0, 0]),
    }
    seam_rows = np.flatnonzero(
        (topology.face_axis == 0)
        & np.isin(topology.face_minus_aggregate_id, list(seam_owner_ids))
        & np.isin(topology.face_plus_aggregate_id, list(seam_owner_ids))
    )
    # The source is merged into the high-x owner, so no aggregate interface
    # remains at this seam; importantly, no duplicate physical boundary faces
    # were emitted either.
    assert seam_rows.size == 0
    assert not np.any(
        (topology.face_axis == 0)
        & ((topology.face_minus_aggregate_id < 0) | (topology.face_plus_aggregate_id < 0))
    )


def test_cross_shard_aggregate_owner_is_explicit_in_local_metadata() -> None:
    shape = (2, 2, 4)
    volume, centroid, second, third, fraction = _raw_geometry(shape)
    source = (0, 0, 1)
    volume[source] = 0.2
    fraction[source] = 0.2
    faces = list(_unit_faces(shape))
    for axis in (0, 1):
        faces[axis].fill(0.0)
    faces[2].fill(0.0)
    faces[2][0, 0, 2] = 2.0
    topology = build_global_control_volume_topology(
        raw_volume=volume,
        raw_centroid=centroid,
        raw_second_moment=second,
        raw_third_moment=third,
        fluid_volume_fraction=fraction,
        face_open_measure=tuple(faces),
    )
    target = (0, 0, 2)
    assert tuple(topology.owner_index[source]) == target
    lower = compile_local_control_volume_geometry(
        topology,
        shard_index=(0, 0, 0),
        shard_counts=(1, 1, 2),
    )
    target_id = int(topology.aggregate_id[target])
    assert target_id in set(lower.remote_aggregate_id.tolist())
    assert int(lower.local_aggregate_id[source]) == target_id


def test_direct_cubic_face_functional_reproduces_monomials() -> None:
    rng = np.random.default_rng(7)
    points = rng.uniform(-0.8, 0.8, size=(32, 3))
    powers = tuple(
        (px, py, total - px - py)
        for total in range(4)
        for px in range(total, -1, -1)
        for py in range(total - px, -1, -1)
    )
    matrix = np.asarray(
        [
            [np.prod(point ** power) for power in powers]
            for point in points
        ],
        dtype=np.float64,
    )
    value_target = np.zeros((20,), dtype=np.float64)
    value_target[0] = 1.0
    gradient_target = np.zeros((3, 20), dtype=np.float64)
    for axis, power in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        gradient_target[axis, powers.index(power)] = 1.0
    functional = precompute_local_face_functional(
        matrix,
        equation_kind=np.full((32,), CV_RECONSTRUCTION_EQUATION_CELL),
        sample_reference=np.arange(32),
        value_target=value_target,
        gradient_target=gradient_target,
    )
    for column in range(20):
        value, gradient = evaluate_local_face_functional(
            functional,
            local_values=matrix[:, column],
        )
        np.testing.assert_allclose(value, value_target[column], atol=1.0e-11)
        np.testing.assert_allclose(
            gradient, gradient_target[:, column], atol=1.0e-11
        )
    packed = pack_local_face_functionals([functional])
    np.testing.assert_array_equal(packed.face_id, np.asarray((-1,)))
    np.testing.assert_allclose(packed.value_weights[0], functional.value_weights)


def test_weighted_direct_cubic_functional_preserves_reproduction() -> None:
    rng = np.random.default_rng(11)
    points = rng.uniform(-0.5, 0.5, size=(28, 3))
    powers = tuple(
        (px, py, total - px - py)
        for total in range(4)
        for px in range(total, -1, -1)
        for py in range(total - px, -1, -1)
    )
    matrix = np.asarray(
        [[np.prod(point ** power) for power in powers] for point in points],
        dtype=np.float64,
    )
    value_target = np.zeros((20,), dtype=np.float64)
    value_target[0] = 1.0
    gradient_target = np.zeros((3, 20), dtype=np.float64)
    gradient_target[0, powers.index((1, 0, 0))] = 1.0
    weights = np.linspace(0.25, 2.0, matrix.shape[0])
    functional = precompute_local_face_functional(
        matrix,
        equation_kind=np.full((matrix.shape[0],), CV_RECONSTRUCTION_EQUATION_CELL),
        sample_reference=np.arange(matrix.shape[0]),
        value_target=value_target,
        gradient_target=gradient_target,
        observation_weight=weights,
    )
    np.testing.assert_allclose(
        functional.value_weights @ matrix,
        value_target,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        functional.gradient_weights @ matrix,
        gradient_target,
        atol=1.0e-11,
    )


def test_cubic_control_volume_average_basis_matches_quadrature() -> None:
    # A uniform rectangular control volume has known central moments.  Use an
    # off-origin box to exercise the translated, scaled average formula.
    lower = np.asarray((-0.3, 0.2, -0.1))
    upper = np.asarray((0.5, 0.8, 0.7))
    centroid = 0.5 * (lower + upper)
    width = upper - lower
    second = np.diag(width**2 / 12.0)
    third = np.zeros((3, 3, 3))
    origin = np.asarray((0.1, -0.2, 0.3))
    scale = np.asarray((0.8, 0.6, 0.9))
    average = cubic_control_volume_average_basis(
        centroid,
        second,
        third,
        origin=origin,
        scale=scale,
    )
    nodes, weights = np.polynomial.legendre.leggauss(4)
    one_dimensional = [
        0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes
        for lo, hi in zip(lower, upper)
    ]
    grid = np.stack(np.meshgrid(*one_dimensional, indexing="ij"), axis=-1)
    quadrature_weight = np.einsum("i,j,k->ijk", weights, weights, weights)
    sampled = np.einsum(
        "ijk,ijkc->c",
        quadrature_weight,
        cubic_monomial_basis((grid - origin) / scale),
    ) / 8.0
    np.testing.assert_allclose(average, sampled, rtol=0.0, atol=1.0e-13)
    # The basis ordering itself is a stable, public numerical contract for
    # direct face-functionals.
    assert len(CUBIC_MONOMIAL_EXPONENTS) == 20


def test_dense_face_targets_match_the_structured_cubic_functional() -> None:
    centers = np.asarray(
        (
            (-0.5, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (1.5, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    second = np.broadcast_to(np.eye(3) / 12.0, (3, 3, 3)).copy()
    third = np.zeros((3, 3, 3, 3), dtype=np.float64)
    scalar_coefficients = np.asarray((0.5, 0.5, 0.0))
    gradient_coefficients = np.zeros((3, 3), dtype=np.float64)
    gradient_coefficients[0] = np.asarray((-1.0, 1.0, 0.0))
    value_target, gradient_target = cubic_dense_face_targets(
        centers,
        second,
        third,
        scalar_coefficients=scalar_coefficients,
        gradient_coefficients=gradient_coefficients,
    )
    basis = cubic_control_volume_average_basis(centers, second, third)
    np.testing.assert_allclose(value_target, scalar_coefficients @ basis)
    np.testing.assert_allclose(gradient_target, gradient_coefficients @ basis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-runtime-info", action="store_true")
    args = parser.parse_args()
    checks = (
        ("global_agglomeration_is_direct_and_conservative", test_global_agglomeration_is_direct_and_conservative),
        ("global_topology_compiles_identically_across_shards", test_global_topology_compiles_identically_across_shards),
        ("periodic_seam_is_one_global_interface_and_can_merge_across_it", test_periodic_seam_is_one_global_interface_and_can_merge_across_it),
        ("cross_shard_aggregate_owner_is_explicit_in_local_metadata", test_cross_shard_aggregate_owner_is_explicit_in_local_metadata),
        ("direct_cubic_face_functional_reproduces_monomials", test_direct_cubic_face_functional_reproduces_monomials),
        ("weighted_direct_cubic_functional_preserves_reproduction", test_weighted_direct_cubic_functional_preserves_reproduction),
        ("cubic_control_volume_average_basis_matches_quadrature", test_cubic_control_volume_average_basis_matches_quadrature),
        ("dense_face_targets_match_the_structured_cubic_functional", test_dense_face_targets_match_the_structured_cubic_functional),
    )
    print("=" * 80)
    print("FCI control-volume characterization checks")
    print("=" * 80)
    for name, check in checks:
        start = time.perf_counter()
        check()
        print(f"PASS {name} time={time.perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()
