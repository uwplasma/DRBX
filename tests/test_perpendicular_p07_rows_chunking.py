"""P09 bitwise performance changes of the P07 integrated rows and of the plan lowering (fast, synthetic world).

* the static conditioned flag: buckets without conditioned faces carry no boundary arrays and skip the lift, bitwise
  equal to applying the all-zero lift arrays;
* the chunked (``lax.map``) contraction with chunk-padded buckets equals the one-piece contraction bitwise, padded
  faces are no-ops, the padded plan still matches the sparse export;
* ``p07_action`` / ``p07_face_flux`` with ``columns`` equal the leading columns of the full call bitwise;
* weight arrays the operators never read are absent from the lowered plan, the sharded plan keeps the structure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native import fci_perpendicular_integrated_rows as IR
from drbx.native.fci_perpendicular_integrated_rows import (
    IntegratedFacePayload, apply_integrated_face_rows, pad_integrated_batch, padded_face_count)
from drbx.native.fci_perpendicular_p07_operator import p07_action, p07_face_flux
from drbx.native.fci_perpendicular_p07_sparse import apply_p07_sparse, export_p07_sparse
from drbx.native.fci_perpendicular_point_rows import BoundaryArrays
from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables
from drbx.native.fci_perpendicular_sharding import shard_perpendicular_plan
from drbx.stencils.operator_plan import plan_nbytes
from tests import perpendicular_sharding_case as sharding_case
from tests.perpendicular_synthetic import NF, Boundary, lower_world, make_world

OWNERS = [0, 1, 2, 40, 96, 99, 100, 114]
KINDS = ("dirichlet", "neumann", "dirichlet")
SMALL_CHUNK = 8


@pytest.fixture(scope="module")
def world():
    return make_world(owners=OWNERS)


@pytest.fixture(scope="module")
def plan(world):
    return lower_world(world)


@pytest.fixture(scope="module")
def chunked_plan(world):
    """The same plan lowered with a tiny chunk, so every bucket larger than 8 faces is padded."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IR, "FACE_CHUNK", SMALL_CHUNK)
        return lower_world(world)


@pytest.fixture(scope="module")
def bc(plan):
    boundary = Boundary()
    return boundary_data_from_callables(plan, boundary.dirichlet, boundary.normal)


@pytest.fixture(scope="module")
def fields(world):
    return np.random.default_rng(5).normal(size=(world.n_owners, NF))


def _payload(plan, batches=None):
    rows = plan.p07.rows
    return IntegratedFacePayload(rows.batches if batches is None else batches, rows.lower_owner, rows.upper_owner,
                                 plan.p07.owner_volume, rows.boundary_query_count, rows.face_count)


def _boundary(plan, bc):
    return BoundaryArrays(bc.dirichlet_value, bc.dirichlet_tangential)


def _same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


# --------------------------------------------------------------------------
# Static conditioned flag
# --------------------------------------------------------------------------

def test_unconditioned_buckets_drop_their_boundary_arrays(plan):
    batches = plan.p07.rows.batches
    flags = [b.boundary_donor_ids is not None for b in batches]
    assert any(flags) and not all(flags)                      # both kinds of buckets occur
    for b, lifted in zip(batches, flags):
        assert lifted == bool(np.any(b.conditioned))
        assert (b.tangential_ids is not None) == (b.tangential_weights is not None) == lifted
    leaves = jax.tree_util.tree_leaves(plan)                  # None is an empty pytree node, not a leaf
    assert all(hasattr(leaf, "shape") for leaf in leaves)


def test_skipping_the_lift_of_unconditioned_buckets_is_bitwise(plan, fields, bc):
    skipped = apply_integrated_face_rows(_payload(plan), fields, _boundary(plan, bc))
    legacy = []
    for b in plan.p07.rows.batches:
        if b.boundary_donor_ids is None:                      # the arrays the old lowering stored: zeros
            n, w = b.donor_ids.shape
            b = b._replace(boundary_donor_ids=np.zeros((n, w), np.int32), tangential_ids=np.zeros((n, 9), np.int32),
                           tangential_weights=np.zeros((n, 9, 2)))
        legacy.append(b)
    assert _same(skipped, apply_integrated_face_rows(_payload(plan, tuple(legacy)), fields, _boundary(plan, bc)))
    run = lambda p: jax.jit(lambda f, g: apply_integrated_face_rows(p, f, g))(fields, _boundary(plan, bc))
    assert _same(run(_payload(plan)), run(_payload(plan, tuple(legacy))))                 # and inside jit


# --------------------------------------------------------------------------
# Chunking with padding
# --------------------------------------------------------------------------

def test_padded_face_count_and_padding_are_no_ops():
    assert [padded_face_count(c, 8) for c in (0, 1, 8, 9, 16, 17)] == [0, 1, 8, 16, 16, 24]
    batch = IR.IntegratedFaceBatch(np.arange(10, dtype=np.int32), np.ones((10, 3), np.int32), np.ones((10, 3)),
                                   np.ones((10, 3), np.int32), np.ones((10, 9), np.int32), np.ones((10, 9, 2)),
                                   np.ones(10, bool))
    padded = pad_integrated_batch(batch, 99, chunk=8)
    assert len(padded.face_ids) == 16 and pad_integrated_batch(batch, 99, chunk=10) is batch
    np.testing.assert_array_equal(padded.face_ids[:10], batch.face_ids)
    assert np.all(padded.face_ids[10:] == 99) and np.all(padded.donor_ids[10:] == 0)
    assert not padded.weights[10:].any() and not padded.tangential_weights[10:].any() and not padded.conditioned[10:].any()


def test_chunked_contraction_with_padded_buckets_equals_the_one_piece_contraction(plan, chunked_plan, fields, bc):
    sizes = [len(b.face_ids) for b in chunked_plan.p07.rows.batches]
    assert any(s > SMALL_CHUNK for s in sizes) and all(s <= SMALL_CHUNK or s % SMALL_CHUNK == 0 for s in sizes)
    assert sizes != [len(b.face_ids) for b in plan.p07.rows.batches]            # padding happened
    padded = [b for b in chunked_plan.p07.rows.batches if np.any(b.face_ids >= chunked_plan.p07.rows.face_count)]
    assert padded and all(not b.weights[b.face_ids >= chunked_plan.p07.rows.face_count].any() for b in padded)
    reference = apply_integrated_face_rows(_payload(plan), fields, _boundary(plan, bc))      # buckets of < 1024 faces
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IR, "FACE_CHUNK", SMALL_CHUNK)
        chunked = apply_integrated_face_rows(_payload(chunked_plan), fields, _boundary(chunked_plan, bc))
        assert "while" in jax.jit(lambda f, g: apply_integrated_face_rows(_payload(chunked_plan), f, g)).lower(
            fields, _boundary(chunked_plan, bc)).as_text()                           # lax.map is really in the program
    assert _same(chunked, reference)
    assert _same(chunked, apply_integrated_face_rows(_payload(chunked_plan), fields, _boundary(chunked_plan, bc)))  # padded, one piece
    # the full operators agree as well, for mixed and uniform kinds
    for kinds in (KINDS, "dirichlet", "neumann"):
        assert _same(p07_face_flux(chunked_plan, fields, bc, kinds), p07_face_flux(plan, fields, bc, kinds))
        assert _same(p07_action(chunked_plan, fields, bc, kinds), p07_action(plan, fields, bc, kinds))


def test_sparse_export_ignores_the_padding_faces(plan, chunked_plan, fields, bc):
    for kind in ("dirichlet", "neumann"):
        a, b = export_p07_sparse(plan, kind), export_p07_sparse(chunked_plan, kind)
        for name in ("matrix", "dirichlet_value", "dirichlet_tangential", "neumann_normal"):
            assert (getattr(a, name) != getattr(b, name)).nnz == 0
        np.testing.assert_allclose(apply_p07_sparse(b, fields, bc), np.asarray(p07_action(chunked_plan, fields, bc, kind)),
                                   rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# Column restriction
# --------------------------------------------------------------------------

def test_p07_columns_are_the_leading_columns_of_the_full_call(plan, fields, bc):
    columns = 2                                          # bitwise for 2 of 3 columns (see the operator docstring)
    full_flux, full_action = p07_face_flux(plan, fields, bc, KINDS), p07_action(plan, fields, bc, KINDS)
    assert _same(p07_face_flux(plan, fields, bc, KINDS, columns=columns), np.asarray(full_flux)[:, :columns])
    assert _same(p07_action(plan, fields, bc, KINDS, columns=columns), np.asarray(full_action)[:, :columns])
    assert p07_action(plan, fields, bc, KINDS, columns=columns).shape == (fields.shape[0], columns)
    one = np.asarray(p07_action(plan, fields, bc, KINDS, columns=1))                      # rounding-level agreement only
    np.testing.assert_allclose(one, np.asarray(full_action)[:, :1], rtol=1e-12, atol=1e-12)
    assert _same(p07_action(plan, fields, bc, KINDS, columns=NF), full_action)           # all columns: the plain call
    with pytest.raises(ValueError):
        p07_action(plan, fields, bc, KINDS, columns=NF + 1)


# --------------------------------------------------------------------------
# Plan fields that no operator reads
# --------------------------------------------------------------------------

def test_never_read_weight_arrays_are_absent_from_the_plan(plan):
    p07n, side, common, cells = plan.p07.neumann, plan.faces.side_neumann, plan.faces.common_neumann, plan.cells.neumann
    assert p07n.value_weights is None and p07n.boundary_value_weights is None          # P07 contracts gradients only
    assert p07n.gradient_weights is not None and p07n.boundary_gradient_weights is not None
    assert side.gradient_weights is None and side.boundary_gradient_weights is None    # R3 side rows: values only
    assert side.value_weights is not None and side.boundary_value_weights is not None
    for rows in (common, cells):                                                       # kept (value and gradient)
        assert all(a is not None for a in jax.tree_util.tree_leaves(rows)) and len(jax.tree_util.tree_leaves(rows)) == 6
    payload = p07n.payload(0)
    assert payload.value_weights is None
    assert plan_nbytes(plan) == sum(np.asarray(a).nbytes for a in jax.tree_util.tree_leaves(plan))


def test_sharded_plan_keeps_the_dropped_fields_dropped_and_pads_chunk_multiples():
    n = sharding_case.N
    raw_to_owner = sharding_case.plane_structured_owners(n)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IR, "FACE_CHUNK", SMALL_CHUNK)
        world = make_world(seed=0, n=n, raw_to_owner=raw_to_owner)
        plan = lower_world(world)
        sharded = shard_perpendicular_plan(plan, raw_to_owner, n, 2)
    stacked = sharded.plan.p07
    assert stacked.neumann.value_weights is None and sharded.plan.faces.side_neumann.gradient_weights is None
    for g, b in zip(plan.p07.rows.batches, stacked.rows.batches):
        assert (g.boundary_donor_ids is None) == (b.boundary_donor_ids is None)
        size = b.face_ids.shape[1]
        assert size <= SMALL_CHUNK or size % SMALL_CHUNK == 0
    local_faces = stacked.rows.face_count
    kept = set()
    for s in range(2):                          # every global face is kept by a shard that owns one of its sides
        ids = np.concatenate([np.asarray(b.face_ids)[s] for b in stacked.rows.batches])
        real = ids[ids < local_faces]
        assert len(np.unique(real)) == len(real)                                 # padding aside, a face is in one bucket
        kept |= {int(x) for x in np.asarray(stacked.p07_id)[s][real]}
    assert kept == {int(x) for x in plan.p07.p07_id}
