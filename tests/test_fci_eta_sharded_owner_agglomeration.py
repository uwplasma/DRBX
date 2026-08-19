"""Focused tests for arbitrary eta-plane-local owner-map lowering."""

from pathlib import Path
import sys

import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drbx.geometry import build_shifted_torus_geometry
from drbx.native.fci_owner_agglomeration import (
    CORNER_EDGE_PACKED_FIELD_COUNT,
    assemble_local_plane_local_owner_map_geometry,
    build_sharded_plane_local_owner_map_payload,
)
from drbx.native.fci_sharding import (
    assemble_local_fci_geometry,
    assemble_single_device_local_fci_geometry,
    build_local_fci_geometries,
    make_shard_mesh,
)


def _l_owner_map(shape):
    ii, jj, kk = np.indices(shape, dtype=np.int32)
    owner_i, owner_j = ii.copy(), jj.copy()
    # A genuine L-shaped three-cell aggregate in every eta plane.
    owner_i[1, 0, :] = 0
    owner_j[1, 0, :] = 0
    owner_i[1, 1, :] = 0
    owner_j[1, 1, :] = 0
    raw = 1.0 + 0.1 * ii + 0.01 * jj + 0.001 * kk
    aggregate = raw.copy()
    aggregate[0, 0, :] = raw[0, 0, :] + raw[1, 0, :] + raw[1, 1, :]
    return owner_i, owner_j, raw, aggregate


def test_one_device_l_shape_has_correct_owner_and_pr_metadata():
    shape = (4, 5, 2)
    owner_i, owner_j, raw, aggregate = _l_owner_map(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=1)
    descriptor, packed = build_sharded_plane_local_owner_map_payload(
        owner_i, owner_j, raw, aggregate, fci.domain
    )
    assert packed.shape == shape + (CORNER_EDGE_PACKED_FIELD_COUNT,)
    assert descriptor.packed_cell_shape == packed.shape
    assert descriptor.cell_partition_spec == P("x", "y", "z", None)
    local = assemble_local_plane_local_owner_map_geometry(
        descriptor, packed, assemble_single_device_local_fci_geometry(fci)
    )
    cells = local.cells
    assert local.angular_group_sizes is None
    assert not bool(np.any(np.asarray(cells.owner_is_remote)))
    np.testing.assert_array_equal(np.asarray(cells.owner_i), owner_i)
    np.testing.assert_array_equal(np.asarray(cells.owner_j), owner_j)
    np.testing.assert_array_equal(np.asarray(cells.owner_k), np.broadcast_to(np.arange(shape[2]), shape))
    # P injects through (0, 0, k); R receives all three storage cells there.
    assert bool(cells.is_active_owner[0, 0, 0])
    assert int(cells.member_count[0, 0, 0]) == 3
    assert int(cells.received_source_count[0, 0, 0]) == 2
    assert bool(cells.is_merged_source[1, 0, 0])
    assert bool(cells.is_merged_source[1, 1, 0])
    assert int(cells.member_count[1, 0, 0]) == 0
    np.testing.assert_allclose(np.asarray(cells.aggregate_volume), aggregate * np.asarray(cells.is_active_owner))


def test_payload_rejects_non_eta_sharding_and_dangling_owner():
    shape = (4, 6, 2)
    owner_i, owner_j, raw, aggregate = _l_owner_map(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    split = build_local_fci_geometries(geometry, (1, 2, 1), halo_width=1)
    with pytest.raises(ValueError, match="shard_counts"):
        build_sharded_plane_local_owner_map_payload(owner_i, owner_j, raw, aggregate, split.domain)
    owner_i[0, 0, :] = 1
    owner_j[0, 0, :] = 4
    one = build_local_fci_geometries(geometry, (1, 1, 1), halo_width=1)
    with pytest.raises(ValueError, match="self-owning"):
        build_sharded_plane_local_owner_map_payload(owner_i, owner_j, raw, aggregate, one.domain)


def test_assembly_can_run_inside_eta_shard_map_when_available():
    if len(jax.devices()) < 2:
        pytest.skip("requires two JAX devices for eta-shard test")
    shape = (4, 5, 4)
    owner_i, owner_j, raw, aggregate = _l_owner_map(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, (1, 1, 2), halo_width=1)
    descriptor, packed = build_sharded_plane_local_owner_map_payload(
        owner_i, owner_j, raw, aggregate, fci.domain
    )
    mesh = make_shard_mesh((1, 1, 2))
    fci_fields = jax.device_put(fci.cell_fields, NamedSharding(mesh, P("x", "y", "z", None)))
    packed_fields = jax.device_put(packed, NamedSharding(mesh, descriptor.cell_partition_spec))

    def kernel(fci_owned, owner_owned):
        local_fci = assemble_local_fci_geometry(fci, fci_owned)
        local = assemble_local_plane_local_owner_map_geometry(descriptor, owner_owned, local_fci)
        return local.cells.member_count

    got = jax.jit(jax.shard_map(
        kernel, mesh=mesh,
        in_specs=(P("x", "y", "z", None), descriptor.cell_partition_spec),
        out_specs=P("x", "y", "z"), check_vma=False,
    ))(fci_fields, packed_fields)
    expected = np.zeros(shape, dtype=np.int32)
    ii, jj, _ = np.indices(shape, dtype=np.int32)
    expected[(owner_i == ii) & (owner_j == jj)] = 1
    expected[0, 0, :] = 3
    np.testing.assert_array_equal(np.asarray(got), expected)
