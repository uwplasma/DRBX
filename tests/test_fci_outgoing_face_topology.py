"""Focused invariants for owner-space outgoing FCI faces."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drbx.geometry import (
    HaloLayout3D,
    LocalFciDirectionMap,
    LocalFciLocalDependencyTable,
    LocalFciMaps3D,
    build_local_control_volume_cell_geometry,
)
from drbx.native.fci_operators import (
    build_local_outgoing_fci_face_topology,
    build_local_outgoing_fci_face_topology_from_geometry,
    prolong_local_outgoing_fci_face_owner_field,
    restrict_local_outgoing_fci_face_field,
)


def _topology():
    shape = (2, 3, 2)
    layout = HaloLayout3D(shape, halo_width=1)
    ii, jj, kk = np.indices(shape, dtype=np.int32)
    # The first two theta source edges share one cell/face owner, but retain
    # deliberately different FCI destinations and interpolation provenance.
    owner_i, owner_j, owner_k = ii.copy(), jj.copy(), kk.copy()
    owner_j[0, 1, :] = 0
    measure = 1.0 + ii + 2.0 * jj + 0.25 * kk
    destination_i = (ii + jj) % shape[0]
    destination_j = (2 * jj + kk) % shape[1]
    destination_k = (kk + 1) % 5  # global/remote eta endpoint identifier
    provenance = np.stack((0.25 + jj, 0.75 - 0.1 * ii), axis=-1)
    return build_local_outgoing_fci_face_topology(
        layout,
        edge_owner_i=owner_i,
        edge_owner_j=owner_j,
        edge_owner_k=owner_k,
        edge_measure=measure,
        edge_destination_i=destination_i,
        edge_destination_j=destination_j,
        edge_destination_k=destination_k,
        edge_interpolation_provenance=provenance,
    )


def test_face_owners_are_direct_eta_local_and_keep_fine_provenance():
    topology = _topology()
    oi, oj, ok = (np.asarray(topology.edge_owner_i), np.asarray(topology.edge_owner_j), np.asarray(topology.edge_owner_k))
    active = np.asarray(topology.edge_active)
    assert np.all(ok[active] == np.indices(topology.shape)[2][active])
    np.testing.assert_array_equal(oi[oi, oj, ok], oi)
    np.testing.assert_array_equal(oj[oi, oj, ok], oj)
    np.testing.assert_array_equal(ok[oi, oj, ok], ok)
    assert topology.is_active_owner[0, 0, 0]
    assert not topology.is_active_owner[0, 1, 0]
    # Grouped face storage does not discard the distinct fine outgoing edges.
    assert topology.edge_destination_j[0, 0, 0] != topology.edge_destination_j[0, 1, 0]
    assert topology.edge_interpolation_provenance[0, 0, 0, 0] != topology.edge_interpolation_provenance[0, 1, 0, 0]


def test_face_prolong_restrict_preserves_constants_and_is_weighted_adjoint():
    topology = _topology()
    rng = np.random.default_rng(4)
    owner = jnp.asarray(rng.normal(size=topology.shape)) * topology.is_active_owner
    fine = jnp.asarray(rng.normal(size=topology.shape))
    prolonged = prolong_local_outgoing_fci_face_owner_field(owner, topology)
    restricted = restrict_local_outgoing_fci_face_field(fine, topology)
    edge_inner = jnp.sum(topology.edge_measure * prolonged * fine)
    owner_inner = jnp.sum(topology.aggregate_measure * owner * restricted)
    np.testing.assert_allclose(edge_inner, owner_inner, rtol=2e-14, atol=2e-14)
    constant_owner = 3.25 * topology.is_active_owner
    np.testing.assert_allclose(
        prolong_local_outgoing_fci_face_owner_field(constant_owner, topology),
        3.25 * topology.edge_active,
    )
    np.testing.assert_allclose(
        restrict_local_outgoing_fci_face_field(3.25 * topology.edge_active, topology),
        constant_owner,
    )
    # Face aliases remain zero in owner storage after restriction.
    assert np.all(np.asarray(restricted)[~np.asarray(topology.is_active_owner)] == 0.0)


def test_face_owner_rejects_cross_eta_owner_map():
    shape = (1, 1, 2)
    ii, jj, kk = np.indices(shape, dtype=np.int32)
    with pytest.raises(ValueError, match="eta-plane local"):
        build_local_outgoing_fci_face_topology(
            HaloLayout3D(shape, 1),
            edge_owner_i=ii, edge_owner_j=jj, edge_owner_k=kk[:, :, ::-1],
            edge_measure=np.ones(shape),
            edge_destination_i=ii, edge_destination_j=jj, edge_destination_k=kk,
            edge_interpolation_provenance=np.zeros(shape + (2,)),
        )


def test_production_builder_uses_raw_volume_edge_mass_and_cell_owner_map():
    """The real local geometry/map path keeps volume, not area, as ``W_e``."""
    shape = (1, 2, 2)
    layout = HaloLayout3D(shape, 1)
    raw_volume = jnp.asarray([[[2.0, 3.0], [5.0, 7.0]]])
    source = jnp.asarray([[[False, False], [True, True]]])
    cells = build_local_control_volume_cell_geometry(
        layout,
        raw_volume=raw_volume,
        raw_centroid=jnp.zeros(shape + (3,), dtype=jnp.float64),
        raw_second_moment=jnp.zeros(shape + (3, 3), dtype=jnp.float64),
        source_active=source,
        target_i=jnp.zeros(shape, dtype=jnp.int32),
        target_j=jnp.zeros(shape, dtype=jnp.int32),
        target_k=jnp.broadcast_to(jnp.arange(shape[2], dtype=jnp.int32), shape),
    )
    empty_rows = LocalFciLocalDependencyTable(
        target_flat=jnp.zeros((0,), dtype=jnp.int32),
        source_i=jnp.zeros((0,), dtype=jnp.int32),
        source_j=jnp.zeros((0,), dtype=jnp.int32),
        source_k=jnp.zeros((0,), dtype=jnp.int32),
        weight=jnp.zeros((0,), dtype=jnp.float64),
        active=jnp.zeros((0,), dtype=bool),
    )
    direction = LocalFciDirectionMap(
        layout=layout, local=empty_rows,
        target_valid=jnp.ones(shape, dtype=bool),
        connection_length=2.0 * jnp.ones(shape, dtype=jnp.float64),
    )
    topology = build_local_outgoing_fci_face_topology_from_geometry(
        cells, LocalFciMaps3D(layout, direction, direction),
    )
    np.testing.assert_allclose(np.asarray(topology.edge_measure), raw_volume)
    np.testing.assert_allclose(
        np.asarray(topology.aggregate_measure),
        np.asarray([[[7.0, 10.0], [0.0, 0.0]]]),
    )


def test_production_builder_is_jittable_with_geometry_and_fci_maps():
    """The production path runs while local geometry is traced by shard_map."""
    shape = (1, 1, 2)
    layout = HaloLayout3D(shape, 1)
    cells = build_local_control_volume_cell_geometry(
        layout,
        raw_volume=jnp.asarray([[[2.0, 5.0]]]),
        raw_centroid=jnp.zeros(shape + (3,), dtype=jnp.float64),
        raw_second_moment=jnp.zeros(shape + (3, 3), dtype=jnp.float64),
    )
    empty_rows = LocalFciLocalDependencyTable(
        target_flat=jnp.zeros((0,), dtype=jnp.int32),
        source_i=jnp.zeros((0,), dtype=jnp.int32),
        source_j=jnp.zeros((0,), dtype=jnp.int32),
        source_k=jnp.zeros((0,), dtype=jnp.int32),
        weight=jnp.zeros((0,), dtype=jnp.float64),
        active=jnp.zeros((0,), dtype=bool),
    )
    direction = LocalFciDirectionMap(
        layout=layout, local=empty_rows,
        target_valid=jnp.ones(shape, dtype=bool),
        connection_length=jnp.ones(shape, dtype=jnp.float64),
    )
    maps = LocalFciMaps3D(layout, direction, direction)
    topology = jax.jit(
        lambda volume: build_local_outgoing_fci_face_topology_from_geometry(
            replace(cells, raw_volume=volume), maps,
        )
    )(cells.raw_volume)
    np.testing.assert_allclose(
        np.asarray(topology.edge_measure), np.asarray(cells.raw_volume)
    )


def _production_support_fixture(*, destination_theta, endpoint_kind=None, axis_row=False):
    """One local interpolation row per edge, with theta-only cell aliases."""
    shape = (1, 4, 1)
    layout = HaloLayout3D(shape, 1)
    cells = build_local_control_volume_cell_geometry(
        layout,
        raw_volume=jnp.ones(shape, dtype=jnp.float64),
        raw_centroid=jnp.zeros(shape + (3,), dtype=jnp.float64),
        raw_second_moment=jnp.zeros(shape + (3, 3), dtype=jnp.float64),
        source_active=jnp.asarray([[[False], [True], [False], [True]]]),
        target_i=jnp.zeros(shape, dtype=jnp.int32),
        target_j=jnp.asarray([0, 0, 2, 2], dtype=jnp.int32)[None, :, None],
        target_k=jnp.zeros(shape, dtype=jnp.int32),
    )
    nrows = int(np.prod(shape))
    halo = layout.halo_width
    source_i = jnp.full((nrows,), halo, dtype=jnp.int32)
    if axis_row:
        source_i = source_i.at[0].set(halo - 1)
    rows = LocalFciLocalDependencyTable(
        target_flat=jnp.arange(nrows, dtype=jnp.int32),
        source_i=source_i,
        source_j=jnp.asarray(destination_theta, dtype=jnp.int32) + halo,
        source_k=jnp.full((nrows,), halo, dtype=jnp.int32),
        weight=jnp.ones((nrows,), dtype=jnp.float64),
        active=jnp.ones((nrows,), dtype=bool),
    )
    direction = LocalFciDirectionMap(
        layout=layout, local=rows,
        target_valid=jnp.ones(shape, dtype=bool),
        connection_length=jnp.ones(shape, dtype=jnp.float64),
        endpoint_kind=(None if endpoint_kind is None else jnp.asarray(endpoint_kind, dtype=jnp.int32)[None, :, None]),
    )
    return cells, LocalFciMaps3D(layout, direction, direction)


def test_production_builder_splits_only_connectivity_or_endpoint_distinct_edges():
    # Cell aggregates are (0, 1) -> theta owner 0 and (2, 3) -> owner 2.
    same_cells = [0, 0, 2, 2]
    cells, maps = _production_support_fixture(destination_theta=same_cells)
    topology = build_local_outgoing_fci_face_topology_from_geometry(cells, maps)
    np.testing.assert_array_equal(np.asarray(topology.edge_owner_j)[0, :, 0], same_cells)

    # The second member of the first aggregate now reaches owner 2, creating
    # a separate agglomerated subface while the second aggregate stays merged.
    cells, maps = _production_support_fixture(destination_theta=[0, 2, 2, 2])
    split = build_local_outgoing_fci_face_topology_from_geometry(cells, maps)
    split_again = build_local_outgoing_fci_face_topology_from_geometry(cells, maps)
    split_jit = jax.jit(
        lambda volume: build_local_outgoing_fci_face_topology_from_geometry(
            replace(cells, raw_volume=volume), maps,
        )
    )(cells.raw_volume)
    np.testing.assert_array_equal(np.asarray(split.edge_owner_j)[0, :, 0], [0, 1, 2, 2])
    np.testing.assert_array_equal(split.edge_owner_j, split_again.edge_owner_j)
    np.testing.assert_array_equal(split.edge_owner_j, split_jit.edge_owner_j)
    assert split.ownership_policy == "coarse-endpoint-support-v1"
    assert np.any(np.asarray(split.edge_destination_support)[0, 0, 0] != np.asarray(split.edge_destination_support)[0, 1, 0])

    # Endpoint kind is part of the same signature even when support matches.
    cells, maps = _production_support_fixture(
        destination_theta=same_cells, endpoint_kind=[0, 1, 0, 0],
    )
    endpoint_split = build_local_outgoing_fci_face_topology_from_geometry(cells, maps)
    np.testing.assert_array_equal(np.asarray(endpoint_split.edge_owner_j)[0, :, 0], [0, 1, 2, 2])


def test_production_builder_normalizes_axis_rows_and_never_uses_inactive_canonical_slot():
    cells, maps = _production_support_fixture(destination_theta=[0, 0, 2, 2], axis_row=True)
    active = jnp.asarray([[[False], [True], [True], [True]]])
    topology = build_local_outgoing_fci_face_topology_from_geometry(
        cells, maps, edge_active=active,
    )
    # Signed radial source -1 is normalized to radial owner 0, so it has the
    # same support signature as the regular row.  The active row selects itself
    # rather than retaining inactive theta zero as its owner slot.
    np.testing.assert_array_equal(np.asarray(topology.edge_owner_j)[0, :, 0], [0, 1, 2, 2])
    assert not np.asarray(topology.is_active_owner)[0, 0, 0]
    assert np.asarray(topology.is_active_owner)[0, 1, 0]
