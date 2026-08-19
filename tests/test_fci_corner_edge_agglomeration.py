from types import SimpleNamespace

import numpy as np
import pytest

from drbx.geometry.fci_corner_edge_agglomeration import build_corner_edge_agglomeration


def _axis_faces(shape, axis):
    face_shape = list(shape)
    face_shape[axis] += 1
    return SimpleNamespace(
        J=np.ones(face_shape),
    )


def _axis_bfield(shape, axis):
    face_shape = list(shape)
    face_shape[axis] += 1
    b = np.zeros(face_shape + [3])
    b[..., axis] = 1.0
    return SimpleNamespace(B_contra=b, Bmag=np.ones(face_shape))


def _geometry(*, seed_locations=((0, 1, 0), (9, 8, 1))):
    shape = (10, 10, 2)
    jacobian = np.ones(shape)
    # Each seed has exactly one low-volume, high-conductance-compatible
    # neighbour.  The two-cell volume is .9 of the mesh median.
    for i, j, k in seed_locations:
        jacobian[i, j, k] = 0.45
        jacobian[1 if i == 0 else 8, j, k] = 0.45
    b = np.full(shape + (3,), 0.1)
    for location in seed_locations:
        b[location] = (10.0, 0.0, 0.0)
    grid = SimpleNamespace(
        x=SimpleNamespace(centers=np.arange(shape[0], dtype=float)),
        y=SimpleNamespace(centers=np.arange(shape[1], dtype=float)),
        z=SimpleNamespace(centers=np.arange(shape[2], dtype=float)),
    )
    return SimpleNamespace(
        shape=shape,
        grid=grid,
        spacing=SimpleNamespace(dx=np.ones(shape), dy=np.ones(shape), dz=np.ones(shape)),
        cell_metric=SimpleNamespace(J=jacobian),
        cell_bfield=SimpleNamespace(B_contra=b, Bmag=np.ones(shape)),
        face_metric=SimpleNamespace(x=_axis_faces(shape, 0), y=_axis_faces(shape, 1), z=_axis_faces(shape, 2)),
        face_bfield=SimpleNamespace(x=_axis_bfield(shape, 0), y=_axis_bfield(shape, 1), z=_axis_bfield(shape, 2)),
    )


def _members(owner, canonical):
    return {tuple(index) for index in np.argwhere(np.all(owner == canonical, axis=-1))}


def _connected_in_plane(members):
    pending = {next(iter(members))}
    visited = set()
    while pending:
        cell = pending.pop()
        if cell in visited:
            continue
        visited.add(cell)
        i, j, k = cell
        pending.update(
            other for other in members
            if other not in visited and other[2] == k and abs(other[0] - i) + abs(other[1] - j) == 1
        )
    return visited == members


def test_corner_edge_owner_map_is_direct_connected_eta_local_and_deterministic():
    geometry = _geometry()
    first = build_corner_edge_agglomeration(geometry, rate_threshold=5.0)
    second = build_corner_edge_agglomeration(geometry, rate_threshold=5.0)
    topology = first.topology
    np.testing.assert_array_equal(topology.owner_index, second.topology.owner_index)
    owner_at_owner = topology.owner_index[tuple(np.moveaxis(topology.owner_index, -1, 0))]
    np.testing.assert_array_equal(owner_at_owner, topology.owner_index)
    assert np.all(topology.owner_index[..., 2] == np.indices(geometry.shape)[2])
    assert np.all(topology.is_merge_source[first.seed_mask] | topology.is_active_owner[first.seed_mask])

    for canonical in np.argwhere(topology.is_active_owner):
        members = _members(topology.owner_index, canonical)
        if len(members) == 1:
            continue
        assert len({cell[2] for cell in members}) == 1
        assert _connected_in_plane(members)
        volume = sum(first.raw_volume[cell] for cell in members)
        assert volume >= first.target_volume_lower
        assert first.projected_parallel_rate[tuple(canonical)] <= 5.0


def test_corner_edge_rejects_a_noncorner_cfl_seed():
    geometry = _geometry(seed_locations=((5, 5, 0),))
    with pytest.raises(ValueError, match="outside configured corner-attached edge strips"):
        build_corner_edge_agglomeration(geometry, rate_threshold=5.0)
