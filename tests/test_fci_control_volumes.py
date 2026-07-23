"""Focused contracts for global agglomeration and direct moment functionals."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from drbx.geometry import HaloLayout3D, LocalControlVolumeCellGeometry3D
from drbx.native.fci_operators import expand_local_control_volume_owner_field

from drbx.geometry.fci_control_volumes import (
    build_global_control_volume_topology,
    combine_volume_moments,
    compile_local_control_volume_geometry,
    remote_owner_halo_coordinate,
)
from drbx.native.fci_boundaries import (CV_RECONSTRUCTION_EQUATION_CELL, CV_RECONSTRUCTION_EQUATION_DIRICHLET, CV_RECONSTRUCTION_EQUATION_REMOTE_CELL)
from drbx.native.fci_control_volume_operators import (
    CUBIC_MONOMIAL_EXPONENTS,
    LocalMomentFittedFaceFunctional3D,
    cubic_control_volume_average_basis,
    cubic_dense_face_targets,
    cubic_monomial_basis,
    evaluate_local_face_functional,
    pack_local_face_functionals,
    precompute_local_face_functional,
    cubic_projected_face_flux_target,
    cubic_parallel_face_flux_target,
    cubic_parallel_gradient_face_flux_target,
    evaluate_local_projected_face_flux,
    evaluate_local_parallel_face_flux,
    evaluate_local_parallel_gradient_face_flux,
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


def test_remote_owner_halo_coordinate_is_face_exact_and_periodic_aware() -> None:
    shape = (3, 4, 5)
    h = 1
    counts = (2, 3, 4)
    # Target in the lower neighbouring shard: remote local index n-1 maps to
    # the immediately adjacent lower halo, not to raw owner_local=n-1.
    lower = remote_owner_halo_coordinate(
        owner_local=np.array([2, 1, 3]), owner_shard=np.array([0, 1, 2]),
        local_shard=np.array([1, 1, 2]), owned_shape=shape, halo_width=h,
        shard_counts=counts, periodic_axes=(False, True, True),
    )
    np.testing.assert_array_equal(lower, np.array([h - 1, h + 1, h + 3]))
    upper = remote_owner_halo_coordinate(
        owner_local=np.array([0, 1, 3]), owner_shard=np.array([1, 1, 2]),
        local_shard=np.array([0, 1, 2]), owned_shape=shape, halo_width=h,
        shard_counts=counts, periodic_axes=(False, True, True),
    )
    np.testing.assert_array_equal(upper, np.array([h + shape[0], h + 1, h + 3]))
    # Periodic y wrap: local shard zero sees the final shard as its lower
    # neighbour, then maps its upper boundary owner to halo h-1.
    wrapped = remote_owner_halo_coordinate(
        owner_local=np.array([1, shape[1] - 1, 2]), owner_shard=np.array([0, 2, 1]),
        local_shard=np.array([0, 0, 1]), owned_shape=shape, halo_width=h,
        shard_counts=counts, periodic_axes=(False, True, True),
    )
    np.testing.assert_array_equal(wrapped, np.array([h + 1, h - 1, h + 2]))
    with pytest.raises(ValueError, match="exactly one directly adjacent"):
        remote_owner_halo_coordinate(
            owner_local=np.array([0, 0, 0]), owner_shard=np.array([1, 1, 2]),
            local_shard=np.array([0, 0, 2]), owned_shape=shape, halo_width=h,
            shard_counts=counts, periodic_axes=(False, True, True),
        )


def test_control_volume_aggregate_id_defaults_to_owner_identity() -> None:
    layout = HaloLayout3D((2, 2, 2), 1)
    volume, centroid, second, third, _ = _raw_geometry((2, 2, 2))
    cells = LocalControlVolumeCellGeometry3D.identity(
        layout, volume=jnp.asarray(volume), centroid=jnp.asarray(centroid),
        second_moment=jnp.asarray(second), third_moment=jnp.asarray(third),
    )
    expected = np.arange(8, dtype=np.int64).reshape((2, 2, 2))
    np.testing.assert_array_equal(np.asarray(cells.aggregate_id), expected)
    leaves, tree = jax.tree_util.tree_flatten(cells)
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    np.testing.assert_array_equal(np.asarray(restored.aggregate_id), expected)


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


@pytest.mark.parametrize("shard_counts", ((1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2)))
def test_global_faces_have_one_evaluator_row_and_exact_remote_target(
    shard_counts: tuple[int, int, int],
) -> None:
    shape = (4, 4, 4)
    volume, centroid, second, third, fraction = _raw_geometry(shape)
    volume[1, 1, 1] = 0.2
    fraction[1, 1, 1] = 0.2
    topology = build_global_control_volume_topology(
        raw_volume=volume, raw_centroid=centroid, raw_second_moment=second,
        raw_third_moment=third, fluid_volume_fraction=fraction,
        face_open_measure=_unit_faces(shape), periodic_axes=(False, True, True),
        coordinate_periods=(1.0, 4.0, 4.0),
    )
    compiled = [
        compile_local_control_volume_geometry(
            topology, shard_index=index, shard_counts=shard_counts,
        )
        for index in np.ndindex(*shard_counts)
    ]
    evaluator_ids = np.concatenate([local.local_face_id for local in compiled])
    np.testing.assert_array_equal(np.sort(evaluator_ids), topology.face_id)
    assert np.unique(evaluator_ids).size == topology.face_id.size
    extent = np.asarray(shape) // np.asarray(shard_counts)
    by_id = {int(face_id): row for row, face_id in enumerate(topology.face_id)}
    for local in compiled:
        for row, face_id in enumerate(local.local_face_id):
            global_row = by_id[int(face_id)]
            minus = int(topology.face_minus_aggregate_id[global_row])
            plus = int(topology.face_plus_aggregate_id[global_row])
            evaluator = minus if minus >= 0 else plus
            owner = np.asarray(np.unravel_index(evaluator, shape))
            expected_shard = owner // extent
            np.testing.assert_array_equal(local.local_face_evaluator_shard_index[row], expected_shard)
            np.testing.assert_array_equal(local.local_face_evaluator_owner_index[row], owner)
            expected_remote = -1
            if plus >= 0 and np.any(np.asarray(np.unravel_index(plus, shape)) // extent != expected_shard):
                expected_remote = plus
            assert int(local.local_face_remote_target_aggregate_id[row]) == expected_remote
    # The periodic high images never reappear as additional global IDs.
    for axis in (1, 2):
        seam = topology.face_storage_index[:, axis] == 0
        assert np.unique(topology.face_id[seam]).size == np.count_nonzero(seam)


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


def test_historical_direction_ordinal_breaks_geometric_target_ties() -> None:
    """The legacy ordinal order is x-, x+, y-, y+, z-, z+ (not target lexicographic order)."""
    shape = (3, 3, 1)
    volume, centroid, second, third, fraction = _raw_geometry(shape)
    source = (1, 1, 0)
    volume[source] = fraction[source] = 0.2
    faces = [np.zeros_like(face) for face in _unit_faces(shape)]
    # x+ and y- have equal measure and equal centroid distance.  Their target
    # coordinates sort as y- < x+, but historical direction ordinal chooses x+.
    faces[0][2, 1, 0] = 1.0
    faces[1][1, 1, 0] = 1.0
    topology = build_global_control_volume_topology(
        raw_volume=volume,
        raw_centroid=centroid,
        raw_second_moment=second,
        raw_third_moment=third,
        fluid_volume_fraction=fraction,
        face_open_measure=tuple(faces),
    )
    assert tuple(topology.owner_index[source]) == (2, 1, 0)


def test_positive_volume_floor_removes_tiny_cells_from_ownership_and_faces() -> None:
    shape = (3, 2, 2)
    volume, centroid, second, third, fraction = _raw_geometry(shape)
    tiny = (1, 0, 0)
    volume[tiny] = 1.0e-14
    fraction[tiny] = 0.1
    topology = build_global_control_volume_topology(
        raw_volume=volume,
        raw_centroid=centroid,
        raw_second_moment=second,
        raw_third_moment=third,
        fluid_volume_fraction=fraction,
        face_open_measure=_unit_faces(shape),
        positive_volume_floor=1.0e-12,
    )
    tiny_id = int(topology.aggregate_id[tiny])
    assert not bool(topology.is_merge_source[tiny])
    assert not bool(topology.is_active_owner[tiny])
    assert bool(topology.is_active_owner[0, 0, 0])
    assert bool(topology.is_active_owner[2, 1, 1])
    assert tiny_id not in set(topology.face_minus_aggregate_id.tolist())
    assert tiny_id not in set(topology.face_plus_aggregate_id.tolist())


def _cross_shard_topology_with_moments():
    """A cheap topology whose only merge crosses the j=2 shard boundary."""
    shape = (4, 4, 4)
    volume, centroid, second, third, fraction = _raw_geometry(shape)
    grid = np.indices(shape, dtype=np.float64)
    second[..., 0, 1] = second[..., 1, 0] = 0.01 * (1.0 + grid[0])
    third[..., 0, 1, 2] = 0.002 * (1.0 + grid[1])
    third[..., 1, 0, 2] = third[..., 2, 0, 1] = third[..., 0, 2, 1] = third[..., 0, 1, 2]
    source, target = (1, 1, 1), (1, 2, 1)
    volume[source] = fraction[source] = 0.2
    faces = [np.zeros_like(face) for face in _unit_faces(shape)]
    faces[1][1, 2, 1] = 3.0
    topology = build_global_control_volume_topology(
        raw_volume=volume,
        raw_centroid=centroid,
        raw_second_moment=second,
        raw_third_moment=third,
        fluid_volume_fraction=fraction,
        face_open_measure=tuple(faces),
    )
    return topology, volume, centroid, second, third, source, target


def test_raw_geometry_compilation_is_decomposition_invariant_with_full_moments() -> None:
    topology, volume, centroid, second, third, _, _ = _cross_shard_topology_with_moments()
    shape = topology.shape
    for counts in ((1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2)):
        owned = tuple(shape[axis] // counts[axis] for axis in range(3))
        stitched_id = np.empty(shape, dtype=np.int64)
        stitched_owner = np.empty(shape + (3,), dtype=np.int32)
        total_active_volume = 0.0
        for shard in np.ndindex(counts):
            local = compile_local_control_volume_geometry(
                topology, shard_index=shard, shard_counts=counts,
                raw_volume=volume, raw_centroid=centroid,
                raw_second_moment=second, raw_third_moment=third,
            )
            slices = tuple(slice(shard[a] * owned[a], (shard[a] + 1) * owned[a]) for a in range(3))
            stitched_id[slices] = local.local_aggregate_id
            stitched_owner[slices] = local.local_owner_index
            active = local.local_active_owner
            total_active_volume += float(np.sum(local.local_aggregate_volume[active]))
            for owner in np.argwhere(active):
                owner_t = tuple(owner)
                global_owner = tuple(owner[a] + slices[a].start for a in range(3))
                members = topology.aggregate_id == topology.aggregate_id[global_owner]
                assert int(local.local_member_count[owner_t]) == int(np.count_nonzero(members))
                assert int(local.local_received_source_count[owner_t]) == int(np.count_nonzero(members) - 1)
                np.testing.assert_array_equal(local.local_aggregate_volume[owner_t], topology.aggregate_volume[global_owner])
                np.testing.assert_array_equal(local.local_aggregate_centroid[owner_t], topology.aggregate_centroid[global_owner])
                np.testing.assert_array_equal(local.local_aggregate_second_moment[owner_t], topology.aggregate_second_moment[global_owner])
                np.testing.assert_array_equal(local.local_aggregate_third_moment[owner_t], topology.aggregate_third_moment[global_owner])
        np.testing.assert_array_equal(stitched_id, topology.aggregate_id)
        np.testing.assert_array_equal(stitched_owner, topology.owner_index)
        assert total_active_volume == float(np.sum(topology.aggregate_volume[topology.is_active_owner]))


def test_cross_shard_source_has_exact_remote_owner_metadata() -> None:
    topology, volume, centroid, second, third, source, target = _cross_shard_topology_with_moments()
    lower = compile_local_control_volume_geometry(
        topology, shard_index=(0, 0, 0), shard_counts=(1, 2, 1),
        raw_volume=volume, raw_centroid=centroid,
        raw_second_moment=second, raw_third_moment=third,
    )
    upper = compile_local_control_volume_geometry(
        topology, shard_index=(0, 1, 0), shard_counts=(1, 2, 1),
    )
    target_id = int(topology.aggregate_id[target])
    assert bool(lower.owner_is_remote[source])
    np.testing.assert_array_equal(lower.owner_shard_index[source], (0, 1, 0))
    np.testing.assert_array_equal(lower.owner_local_index[source], (1, 0, 1))
    assert target_id in set(lower.remote_aggregate_id.tolist())
    assert not bool(lower.local_active_owner[source])
    assert bool(upper.local_active_owner[1, 0, 1])
    assert int(np.count_nonzero(lower.local_active_owner)) + int(np.count_nonzero(upper.local_active_owner)) == int(np.count_nonzero(topology.is_active_owner))


def test_remote_owner_metadata_and_expansion_use_owner_halo() -> None:
    layout = HaloLayout3D((2, 2, 2), 1)
    volume = jnp.ones((2, 2, 2), dtype=jnp.float64)
    centroid = jnp.zeros((2, 2, 2, 3), dtype=jnp.float64)
    base = LocalControlVolumeCellGeometry3D.identity(layout, volume=volume, centroid=centroid)
    source = (0, 0, 0)
    cells = LocalControlVolumeCellGeometry3D(
        **{**base.__dict__, "is_merged_source": base.is_merged_source.at[source].set(True),
           "is_active_owner": base.is_active_owner.at[source].set(False),
           "aggregate_volume": base.aggregate_volume.at[source].set(0.0),
           "owner_is_remote": jnp.zeros((2,2,2), dtype=bool).at[source].set(True),
           "remote_owner_halo_i": jnp.zeros((2,2,2), dtype=jnp.int32).at[source].set(3),
           "remote_owner_halo_j": jnp.zeros((2,2,2), dtype=jnp.int32).at[source].set(1),
           "remote_owner_halo_k": jnp.zeros((2,2,2), dtype=jnp.int32).at[source].set(1)})
    values = jnp.arange(8, dtype=jnp.float64).reshape((2,2,2))
    with pytest.raises(ValueError, match="owner_values_halo"):
        expand_local_control_volume_owner_field(values, cells)
    halo = jnp.zeros(layout.cell_halo_shape, dtype=jnp.float64).at[3,1,1].set(37.0)
    expanded = expand_local_control_volume_owner_field(values, cells, owner_values_halo=halo)
    assert float(expanded[source]) == 37.0


def test_cubic_projected_flux_target_matches_explicit_monomial_quadrature() -> None:
    rng = np.random.default_rng(4)
    points = rng.normal(size=(2, 3, 3)); jacobian = 0.2 + rng.random((2,3))
    area = rng.normal(size=(2,3,3)); projector = rng.normal(size=(2,3,3,3))
    active = np.array([[True, False, True], [True, True, False]])
    origin = np.array([.3,-.7,1.1]); scale = np.array([.4,1.7,2.2])
    target = cubic_projected_face_flux_target(points, jacobian, area, projector, active, origin=origin, scale=scale)
    explicit = []
    xi = (points-origin)/scale
    for p in CUBIC_MONOMIAL_EXPONENTS:
        grad = np.zeros_like(points)
        for a in range(3):
            if p[a]:
                q=list(p); q[a]-=1
                grad[...,a]=p[a]*xi[...,0]**q[0]*xi[...,1]**q[1]*xi[...,2]**q[2]/scale[a]
        explicit.append(np.sum(np.where(active, jacobian*np.einsum('...i,...ij,...j->...',area,projector,grad),0)))
    np.testing.assert_allclose(target, explicit, atol=1e-12)


def test_cubic_parallel_flux_target_matches_explicit_monomial_quadrature() -> None:
    rng = np.random.default_rng(41)
    points = rng.normal(size=(2, 3, 3)); jacobian = 0.2 + rng.random((2, 3))
    area = rng.normal(size=(2, 3, 3)); b_contra = rng.normal(size=(2, 3, 3))
    bmag = rng.random((2, 3)); active = np.array([[True, False, True], [True, True, False]])
    origin = np.array([.3, -.7, 1.1]); scale = np.array([.4, 1.7, 2.2])
    target = cubic_parallel_face_flux_target(
        points, jacobian, area, b_contra, bmag, active,
        origin=origin, scale=scale, b_floor=.15,
    )
    xi = (points - origin) / scale
    explicit = []
    flux_scale = jacobian * np.einsum("...i,...i->...", area, b_contra / np.maximum(bmag, .15)[..., None])
    for power in CUBIC_MONOMIAL_EXPONENTS:
        phi = xi[..., 0] ** power[0] * xi[..., 1] ** power[1] * xi[..., 2] ** power[2]
        explicit.append(np.sum(np.where(active, flux_scale * phi, 0.0)))
    np.testing.assert_allclose(target, explicit, atol=1e-12)

    gradient_target = cubic_parallel_gradient_face_flux_target(
        points,
        jacobian,
        area,
        b_contra,
        bmag,
        active,
        origin=origin,
        scale=scale,
        b_floor=.15,
    )
    b = b_contra / np.maximum(bmag, .15)[..., None]
    expected_gradient_target = cubic_projected_face_flux_target(
        points,
        jacobian,
        area,
        np.einsum("...i,...j->...ij", b, b),
        active,
        origin=origin,
        scale=scale,
    )
    np.testing.assert_allclose(gradient_target, expected_gradient_target, atol=1e-12)


def test_projected_functional_reproduces_mixed_observations_and_packs() -> None:
    rng = np.random.default_rng(6)
    points = rng.normal(size=(20, 3))
    origin = np.array([0.2, -0.4, 0.7])
    scale = np.array([0.8, 1.7, 2.3])
    centroids = points[:14]
    second = np.broadcast_to(np.diag([.07, .11, .13]), (14, 3, 3)).copy()
    seed_third = np.full((3, 3, 3), 0.002)
    # Explicitly symmetrize a finite nonzero central M3 tensor.
    third_one = sum(
        np.transpose(seed_third, permutation)
        for permutation in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    ) / 6.0
    third = np.broadcast_to(third_one, (14, 3, 3, 3)).copy()
    # CELL/REMOTE are genuine finite-volume average rows; only Dirichlet is a trace.
    matrix = np.vstack((
        cubic_control_volume_average_basis(
            centroids, second, third, origin=origin, scale=scale,
        ),
        cubic_monomial_basis((points[14:] - origin) / scale),
    ))
    kinds=np.array([CV_RECONSTRUCTION_EQUATION_CELL]*7+[CV_RECONSTRUCTION_EQUATION_REMOTE_CELL]*7+[CV_RECONSTRUCTION_EQUATION_DIRICHLET]*6)
    refs=np.r_[np.arange(7),np.arange(7),np.arange(6)]
    target = cubic_projected_face_flux_target(
        points[:1], np.ones((1,)), np.array([[1., 2., -1.]]),
        np.array([[[.2, 1., 0.], [-.3, .1, .5], [.7, 0., .4]]]),
        np.array([True]), origin=origin, scale=scale,
    )
    parallel_target = cubic_parallel_face_flux_target(
        points[:1], np.ones((1,)), np.array([[1., 2., -1.]]),
        np.array([[.4, -.8, .6]]), np.array([.2]), np.array([True]),
        origin=origin, scale=scale, b_floor=.3,
    )
    parallel_gradient_target = cubic_projected_face_flux_target(
        points[:1], np.ones((1,)), np.array([[1., 2., -1.]]),
        np.array([[[.8, -.1, .2], [-.1, .5, .3], [.2, .3, .7]]]),
        np.array([True]), origin=origin, scale=scale,
    )
    f=precompute_local_face_functional(matrix,equation_kind=kinds,sample_reference=refs,value_target=np.zeros(20),gradient_target=np.zeros((3,20)),projected_flux_target=target,parallel_flux_target=parallel_target,parallel_gradient_flux_target=parallel_gradient_target,face_id=2,face_sign=-1,max_projected_flux_l1=1e9,max_parallel_flux_l1=1e9,max_parallel_gradient_flux_l1=1e9)
    for column in range(20):
        obs=matrix[:,column]
        got=evaluate_local_projected_face_flux(f,local_values=obs[:7],remote_values=obs[7:14],boundary_values=obs[14:])
        np.testing.assert_allclose(got,target[column],atol=1e-10)
        got_parallel=evaluate_local_parallel_face_flux(f,local_values=obs[:7],remote_values=obs[7:14],boundary_values=obs[14:])
        np.testing.assert_allclose(got_parallel,parallel_target[column],atol=1e-10)
        got_parallel_gradient=evaluate_local_parallel_gradient_face_flux(f,local_values=obs[:7],remote_values=obs[7:14],boundary_values=obs[14:])
        np.testing.assert_allclose(got_parallel_gradient,parallel_gradient_target[column],atol=1e-10)
    packed=pack_local_face_functionals([f])
    assert packed.face_id.dtype==np.int64 and packed.face_sign[0]==-1
    np.testing.assert_allclose(packed.projected_flux_weights[0],f.projected_flux_weights)
    np.testing.assert_allclose(packed.parallel_flux_weights[0],f.parallel_flux_weights)
    np.testing.assert_allclose(packed.parallel_gradient_flux_weights[0],f.parallel_gradient_flux_weights)
    assert packed.normalized_projected_weight_norm[0] == f.normalized_projected_weight_norm
    assert packed.normalized_parallel_weight_norm[0] == f.normalized_parallel_weight_norm
    assert packed.normalized_parallel_gradient_weight_norm[0] == f.normalized_parallel_gradient_weight_norm
    assert pack_local_face_functionals([]).projected_flux_weights.shape==(0,0)
    assert pack_local_face_functionals([]).parallel_flux_weights.shape==(0,0)
    assert pack_local_face_functionals([]).parallel_gradient_flux_weights.shape==(0,0)


def test_projected_functional_rejects_bad_rank_condition_and_weight_norm() -> None:
    points=np.stack(np.meshgrid(np.arange(3.),np.arange(3.),np.arange(3.),indexing='ij'),axis=-1).reshape((-1,3))[:20]
    # A repeated row is rank deficient.
    matrix=cubic_monomial_basis(points); kinds=np.full(20,CV_RECONSTRUCTION_EQUATION_CELL); refs=np.arange(20)
    with pytest.raises(ValueError, match='rank deficient'):
        precompute_local_face_functional(np.tile(matrix[0],(20,1)),equation_kind=kinds,sample_reference=refs,value_target=np.zeros(20),gradient_target=np.zeros((3,20)))
    # Full rank Vandermonde, rejected solely by an intentionally strict limit.
    rng=np.random.default_rng(10); full=cubic_monomial_basis(rng.normal(size=(20,3)))
    with pytest.raises(ValueError, match='ill conditioned'):
        precompute_local_face_functional(full,equation_kind=kinds,sample_reference=refs,value_target=np.zeros(20),gradient_target=np.zeros((3,20)),condition_limit=1.0)
    with pytest.raises(ValueError, match='projected-flux norm'):
        precompute_local_face_functional(full,equation_kind=kinds,sample_reference=refs,value_target=np.zeros(20),gradient_target=np.zeros((3,20)),projected_flux_target=np.ones(20),max_projected_flux_l1=1e-16)
    with pytest.raises(ValueError, match='parallel-flux norm'):
        precompute_local_face_functional(full,equation_kind=kinds,sample_reference=refs,value_target=np.zeros(20),gradient_target=np.zeros((3,20)),parallel_flux_target=np.ones(20),max_parallel_flux_l1=1e-16)
    with pytest.raises(ValueError, match='normalized parallel weight norm'):
        precompute_local_face_functional(full,equation_kind=kinds,sample_reference=refs,value_target=np.zeros(20),gradient_target=np.zeros((3,20)),parallel_flux_target=np.ones(20),max_normalized_parallel_weight_norm=1e-16)
    with pytest.raises(ValueError, match='parallel-gradient-flux norm'):
        precompute_local_face_functional(full,equation_kind=kinds,sample_reference=refs,value_target=np.zeros(20),gradient_target=np.zeros((3,20)),parallel_gradient_flux_target=np.ones(20),max_parallel_gradient_flux_l1=1e-16)
    # An omitted optional target is intentionally represented by zero weights.
    f=precompute_local_face_functional(full,equation_kind=kinds,sample_reference=refs,value_target=np.zeros(20),gradient_target=np.zeros((3,20)),max_derivative_l1=1e9)
    np.testing.assert_array_equal(f.projected_flux_weights,0.0)
    np.testing.assert_array_equal(f.parallel_flux_weights,0.0)
    np.testing.assert_array_equal(f.parallel_gradient_flux_weights,0.0)
    direct = LocalMomentFittedFaceFunctional3D(equation_kind=np.array([CV_RECONSTRUCTION_EQUATION_CELL]),sample_reference=np.array([0]),active=np.array([True]),value_weights=np.zeros(1),gradient_weights=np.zeros((3,1)),polynomial_order=3,rank=1,condition_number=1.,reproduction_residual=0.,normalized_weight_norm=0.)
    np.testing.assert_array_equal(direct.projected_flux_weights,0.0)
    np.testing.assert_array_equal(direct.parallel_flux_weights,0.0)
    np.testing.assert_array_equal(direct.parallel_gradient_flux_weights,0.0)


def test_face_functional_reference_padding_is_valid_only_when_inactive() -> None:
    kwargs = dict(
        equation_kind=np.array([CV_RECONSTRUCTION_EQUATION_CELL]),
        sample_reference=np.array([-1]),
        value_weights=np.zeros(1),
        gradient_weights=np.zeros((3, 1)),
        polynomial_order=3,
        rank=1,
        condition_number=1.0,
        reproduction_residual=0.0,
        normalized_weight_norm=0.0,
    )
    with pytest.raises(ValueError, match="nonnegative references"):
        LocalMomentFittedFaceFunctional3D(active=np.array([True]), **kwargs)
    padded = LocalMomentFittedFaceFunctional3D(active=np.array([False]), **kwargs)
    assert int(padded.sample_reference[0]) == -1


@pytest.mark.parametrize('bad_scale', [0.0, -1.0, [1.0, 2.0], [1.0, -2.0, 3.0]])
def test_projected_target_rejects_invalid_scale_and_input_data(bad_scale) -> None:
    points=np.zeros((2,3)); jac=np.ones(2); area=np.ones((2,3)); proj=np.ones((2,3,3)); active=np.ones(2,dtype=bool)
    with pytest.raises(ValueError): cubic_projected_face_flux_target(points,jac,area,proj,active,origin=np.zeros(3),scale=bad_scale)


def test_projected_target_rejects_shape_and_nonfinite_inputs() -> None:
    points=np.zeros((2,3)); jac=np.ones(2); area=np.ones((2,3)); proj=np.ones((2,3,3)); active=np.ones(2,dtype=bool)
    for args in [
        (points, np.ones(3), area, proj),
        (points, jac, np.ones((3,3)), proj),
        (points, jac, area, np.ones((2,3,2))),
        (np.array([[np.nan,0,0],[0,0,0]]),jac,area,proj),
        (points,np.array([np.inf,1]),area,proj),
        (points,jac,np.array([[np.inf,0,0],[0,0,0]]),proj),
        (points,jac,area,np.where(np.eye(3)[None,:,:].astype(bool),np.nan,proj)),
    ]:
        with pytest.raises(ValueError): cubic_projected_face_flux_target(*args,active,origin=np.zeros(3),scale=1.0)
    matrix=np.eye(20); kinds=np.full(20,CV_RECONSTRUCTION_EQUATION_CELL); refs=np.arange(20)
    with pytest.raises(ValueError, match='finite'):
        precompute_local_face_functional(matrix,equation_kind=kinds,sample_reference=refs,value_target=np.zeros(20),gradient_target=np.zeros((3,20)),projected_flux_target=np.r_[np.nan,np.zeros(19)])
    with pytest.raises(ValueError, match='finite'):
        precompute_local_face_functional(matrix,equation_kind=kinds,sample_reference=refs,value_target=np.zeros(20),gradient_target=np.zeros((3,20)),parallel_flux_target=np.r_[np.nan,np.zeros(19)])


def test_parallel_target_rejects_shape_nonfinite_and_invalid_floor() -> None:
    points=np.zeros((2,3)); jac=np.ones(2); area=np.ones((2,3)); b=np.ones((2,3)); bmag=np.ones(2); active=np.ones(2,dtype=bool)
    for args in [
        (points, np.ones(3), area, b, bmag),
        (points, jac, np.ones((3,3)), b, bmag),
        (points, jac, area, np.ones((2,2)), bmag),
        (points, jac, area, b, np.ones(3)),
        (np.array([[np.nan,0,0],[0,0,0]]), jac, area, b, bmag),
        (points, jac, area, np.array([[np.nan,0,0],[0,0,0]]), bmag),
        (points, jac, area, b, np.array([np.inf, 1.])),
    ]:
        with pytest.raises(ValueError):
            cubic_parallel_face_flux_target(*args, active, origin=np.zeros(3), scale=1.0)
    for bad_floor in (-1.0, 0.0, np.nan):
        with pytest.raises(ValueError, match='b_floor'):
            cubic_parallel_face_flux_target(points, jac, area, b, bmag, active, origin=np.zeros(3), scale=1.0, b_floor=bad_floor)


def test_periodic_seam_face_ids_are_unique_contiguous_and_canonical() -> None:
    shape = (4, 2, 2)
    volume, centroid, second, third, fraction = _raw_geometry(shape)
    faces = [np.zeros_like(face) for face in _unit_faces(shape)]
    faces[0][0, 0, 0] = faces[0][-1, 0, 0] = 2.0
    topology = build_global_control_volume_topology(
        raw_volume=volume, raw_centroid=centroid,
        raw_second_moment=second, raw_third_moment=third,
        fluid_volume_fraction=fraction, face_open_measure=tuple(faces),
        periodic_axes=(True, False, False), coordinate_periods=(4.0, 1.0, 1.0),
    )
    np.testing.assert_array_equal(topology.face_id, np.arange(topology.face_id.size))
    seam = np.flatnonzero((topology.face_axis == 0) & (topology.face_storage_index[:, 0] == 0))
    assert seam.size == 1
    assert not np.any((topology.face_axis == 0) & (topology.face_storage_index[:, 0] == shape[0]))


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
