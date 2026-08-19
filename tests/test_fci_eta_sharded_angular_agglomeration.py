"""Focused tests for eta-only native angular-agglomeration lowering."""

from pathlib import Path
import sys

import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from drbx.geometry import build_shifted_torus_geometry
from drbx.geometry.fci_control_volumes import (
    build_polar_angular_agglomeration_geometry,
)
from drbx.native.fci_angular_agglomeration import (
    RLP_PACKED_FIELD_COUNT,
    assemble_local_polar_angular_agglomeration_geometry,
    build_sharded_polar_angular_agglomeration_payload,
)
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


@pytest.mark.parametrize("shard_counts", [(2, 1, 1), (1, 2, 1)])
def test_eta_payload_rejects_radial_or_theta_sharding(shard_counts):
    shape = (4, 8, 4)
    host = _host(shape)
    geometry = build_shifted_torus_geometry(shape, construct_fci_maps=False)
    fci = build_local_fci_geometries(geometry, shard_counts, halo_width=1)
    with pytest.raises(ValueError, match="eta sharding only|x/theta sharding"):
        build_sharded_polar_angular_agglomeration_payload(host, fci.domain)
