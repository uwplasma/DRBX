"""Tests for strict host-side one-device FCI geometry assembly."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from drbx.geometry import build_shifted_torus_geometry  # noqa: E402
from drbx.native.fci_sharding import (  # noqa: E402
    assemble_local_fci_geometry,
    assemble_single_device_local_fci_geometry,
    build_local_fci_geometries,
    make_shard_mesh,
)


def _comparison_arrays(local):
    """Return the dense numerical geometry arrays, excluding static map rows."""
    arrays = []
    arrays.extend(jax.tree_util.tree_leaves(local.cell_metric))
    arrays.extend(jax.tree_util.tree_leaves(local.face_metric))
    arrays.extend(jax.tree_util.tree_leaves(local.cell_bfield))
    arrays.extend(jax.tree_util.tree_leaves(local.face_bfield))
    arrays.extend(jax.tree_util.tree_leaves(local.spacing))
    arrays.extend(jax.tree_util.tree_leaves(local.regular_face_geometry))
    arrays.extend(jax.tree_util.tree_leaves(local.cell_volume_geometry))
    return tuple(leaf for leaf in arrays if hasattr(leaf, "shape"))


def test_host_assembly_matches_one_device_shard_map_for_polar_axis_geometry():
    geometry = build_shifted_torus_geometry((4, 8, 3), construct_fci_maps=True)
    sharded = build_local_fci_geometries(
        geometry,
        (1, 1, 1),
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )
    host = assemble_single_device_local_fci_geometry(sharded)
    mesh = make_shard_mesh((1, 1, 1))
    spec = P("x", "y", "z")
    fields = jax.device_put(sharded.cell_fields, jax.sharding.NamedSharding(mesh, spec))

    def kernel(fields_owned):
        local = assemble_local_fci_geometry(sharded, fields_owned)
        return _comparison_arrays(local)

    mapped = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=spec,
            out_specs=tuple(P() for _ in _comparison_arrays(host)),
            check_vma=False,
        )
    )(fields)
    host_leaves = _comparison_arrays(host)
    mapped_leaves = tuple(mapped)
    assert len(host_leaves) == len(mapped_leaves)
    for expected, actual in zip(host_leaves, mapped_leaves):
        if isinstance(expected, (jax.Array, np.ndarray)) or hasattr(expected, "shape"):
            np.testing.assert_allclose(np.asarray(expected), np.asarray(actual), equal_nan=True)
        else:
            assert expected == actual


def test_host_assembly_rejects_multi_shard_geometry():
    geometry = build_shifted_torus_geometry((4, 8, 3), construct_fci_maps=False)
    sharded = build_local_fci_geometries(geometry, (2, 1, 1))
    with pytest.raises(ValueError, match=r"exactly \(1, 1, 1\)"):
        assemble_single_device_local_fci_geometry(sharded)
