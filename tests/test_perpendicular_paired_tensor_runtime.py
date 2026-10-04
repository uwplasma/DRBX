"""Paired tensor sources at run time (storage stage 2): the loader, the kernel and the eta sharding apply a symmetric
cell row ``1/2 (A + B)`` from the factors of its two parts, without ever materializing the merged CSR row.

Fast and synthetic, on the n = 12 toy grid of ``tests/test_perpendicular_paired_tensor_rows.py`` (coupled, ringwise
and singleton pairs, a mixed ring that stays CSR, a ringwise row without a mirror, boundary rows):

* ``decode_chunk_factored`` keeps pairs factored (A then B, ``TensorRows.pair_part``); the pair markers validate;
* ``lower_point_chunks`` emits *paired* batches (a ``TensorMirror`` with the index arrays of B) with the layout of the
  dense lowering of the same chunk -- slots, targets and eta offsets identical -- and agrees in the applied values and
  gradients to rounding; the mirror arrays resolve to the factors of B; the theta combos cover B;
* the kernel's paired apply is ``0.5 * (A + B)`` of its two parts, blocked or not, eager or jitted;
* ``shard_perpendicular_plan`` keeps, localizes, window-checks and pads the mirror arrays: every shard applies its
  owned cells to the single-device values.
"""
from __future__ import annotations

import dataclasses

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry.fci_perpendicular_reconstruction import PairedFactors, StructuredReconstruction
from drbx.native import fci_perpendicular_tensor_rows as kernel
from drbx.native.fci_perpendicular_sharding import (
    DEFAULT_HALO, PlaneLayout, _Ctx, _Shard, _select_sources, local_plan, plane_major_permutation,
    shard_perpendicular_plan)
from drbx.native.fci_perpendicular_point_rows import BoundaryArrays
from drbx.native.fci_perpendicular_source_rows import apply_source_rows, payload_nbytes
from drbx.native.fci_perpendicular_tensor_rows import TensorMirror
from drbx.stencils import artifact as art
from drbx.stencils import tensor_rows as tr_mod
from drbx.stencils.artifact import ENCODING_PAIRED, ENCODING_TENSOR, decode_chunk, encode_chunk
from drbx.stencils.loader import (
    FactoredChunk, LoaderGrid, RowSelection, iter_factored_point_chunks, lower_point_chunks, lower_point_files)
from drbx.stencils.operator_plan import CellPlan, PerpendicularPlan
from tests.test_perpendicular_paired_tensor_rows import KEYS, N, RINGS, _arrays, _case, _pack
from tests.test_perpendicular_symmetric_cell_rows import _cell, _context
from tests.test_stencils_tensor_loader import (
    TOL, _apply, _assert_same_layout, _boundary, _fields, _relative, _weight_scales)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def paired():
    case = _case("symmetric")
    data = encode_chunk("cells", case["chunk"], factors=case["factors"])
    grid = LoaderGrid.from_context(case["context"])
    item = FactoredChunk(*art.decode_chunk_factored("cells", data))               # pairs stay factored (the default)
    dense = decode_chunk("cells", data)
    return dict(case=case, data=data, grid=grid, item=item, dense=dense, encoding=_arrays(data)["src_encoding"],
                plan=lower_point_chunks([item], grid=grid), csr=lower_point_chunks([dense], grid=grid))


def _paired_batches(plan):
    return [b for b in plan.payload.tensor_batches if b.mirror is not None]


# ---------------------------------------------------------------------------
# the decoded view and the pair markers
# ---------------------------------------------------------------------------
def test_decode_keeps_pairs_factored_as_consecutive_a_b_sources(paired):
    case, encoding = paired["case"], paired["encoding"]
    stored, tr, is_tensor = art.decode_chunk_factored("cells", paired["data"])
    assert np.array_equal(is_tensor, encoding > 0) and (encoding == ENCODING_PAIRED).any() and (encoding == ENCODING_TENSOR).any()
    pairs, singles = int((encoding == ENCODING_PAIRED).sum()), int((encoding == ENCODING_TENSOR).sum())
    assert tr.n_sources == 2 * pairs + singles
    assert np.array_equal(tr.multiplicity, encoding[is_tensor]) and len(tr.logical_sources) == is_tensor.sum()
    assert np.array_equal(tr.pair_part, tr_mod.pair_part_of(encoding[is_tensor]))
    a = np.flatnonzero(tr.pair_part == 1)
    assert len(a) == pairs and np.all(tr.pair_part[a + 1] == 2) and np.array_equal(tr.family[a], tr.family[a + 1])
    # tensor sources (pairs too) store no donors; the tags and pointers are the full decode's
    full = decode_chunk("cells", paired["data"])
    assert np.array_equal(np.diff(stored.donor_ptr) == 0, is_tensor)
    for name in ("request", "entity_id", "quad_node", "family", "conditioned", "bc_variant", "radial_degree",
                 "target_point", "has_gradient", "source_ptr", "query_table"):
        assert getattr(stored, name).tobytes() == getattr(full, name).tobytes(), name
    # expanding the factors and merging the pairs gives the original rows bit for bit
    exp = tr_mod.merge_paired_expansion(tr_mod.expand_tensor_rows(tr), np.diff(tr.target_ptr), tr.has_gradient, tr.multiplicity)
    for k, s in enumerate(np.flatnonzero(is_tensor)):
        donor, value, gradient = exp.source_rows(k, 1, True)
        row = case["rows"][s]
        assert donor.tobytes() == row.donor_ids.tobytes() and value.tobytes() == row.value.tobytes()
        assert np.ascontiguousarray(gradient).tobytes() == np.ascontiguousarray(row.gradient).tobytes()
    # the first-stage view is still there for consumers that cannot apply a pair
    _, singles_only, flag = art.decode_chunk_factored("cells", paired["data"], expand_pairs=True)
    assert np.array_equal(flag, encoding == ENCODING_TENSOR) and not singles_only.pair_part.any()


def test_pair_markers_are_validated_and_taken_whole(paired):
    _, tr, _ = art.decode_chunk_factored("cells", paired["data"])
    tr.check_pairs()
    assert tr_mod.pair_part_of([1, 2, 1]).tolist() == [0, 1, 2, 0]
    assert tr_mod.pair_part_of([1, 1]).tolist() == [0, 0] and tr_mod.pair_part_of([2, 2]).tolist() == [1, 2, 1, 2]
    a = int(np.flatnonzero(tr.pair_part == 1)[0])
    s = int(np.flatnonzero(tr.pair_part == 0)[0])
    bad = {
        "a pair is an A part": dataclasses.replace(tr, pair_part=np.where(np.arange(tr.n_sources) == a + 1, 0, tr.pair_part).astype(np.int8)),
        "pair_part": dataclasses.replace(tr, pair_part=np.where(np.arange(tr.n_sources) == s, 3, tr.pair_part).astype(np.int8)),
        "differ in family": dataclasses.replace(tr, family=np.where(np.arange(tr.n_sources) == a + 1, 1 - tr.family[a], tr.family).astype(np.int8)),
        "differ in targets": dataclasses.replace(tr, has_gradient=np.where(np.arange(tr.n_sources) == a, ~tr.has_gradient[a], tr.has_gradient)),
        "pair_part ": dataclasses.replace(tr, pair_part=tr.pair_part[:-1]),
    }
    for message, rows in bad.items():
        with pytest.raises(ValueError, match=message.strip()):
            rows.check_pairs()
    # a pair is taken whole or not at all; the markers follow the sources
    sub = tr.take_sources([a, a + 1, s])
    assert sub.pair_part.tolist() == [1, 2, 0] and sub.multiplicity.tolist() == [2, 1]
    sub.check_pairs()
    with pytest.raises(ValueError, match="A part followed by its B part"):
        tr.take_sources([a])
    with pytest.raises(ValueError, match="A part followed by its B part"):
        tr.take_sources([a + 1, s])


# ---------------------------------------------------------------------------
# the loader: paired batches against the dense lowering of the same chunk
# ---------------------------------------------------------------------------
def test_paired_plan_has_the_dense_layout_and_applies_to_rounding(paired):
    plan, csr, dense = paired["plan"], paired["csr"], paired["dense"]
    _assert_same_layout(plan, csr)                      # slots, targets, eta offsets and halo of the merged rows
    batches = _paired_batches(plan)
    assert {b.family for b in batches} == {"singleton", "ringwise"}
    assert any(b.mirror is None for b in plan.payload.tensor_batches)             # the ringwise row without a mirror
    assert plan.payload.tensor_batches and not csr.payload.tensor_batches
    scale_v, scale_g = _weight_scales(dense)
    rows = np.flatnonzero(csr.targets.gradient_slot >= 0)
    slots = csr.targets.gradient_slot[rows]
    for nf, seed in ((1, 1), (3, 2), (5, 3)):
        fields = _fields(paired["case"]["context"], nf, seed)
        v_p, g_p = (np.asarray(x) for x in _apply(plan, fields))
        v_c, g_c = (np.asarray(x) for x in _apply(csr, fields))
        magnitude = np.abs(fields).max()
        assert _relative(v_p - v_c, np.maximum(scale_v, 1e-300)[:, None] * magnitude) <= TOL
        assert _relative(g_p[slots] - g_c[slots], np.maximum(scale_g[rows], 1e-300)[:, :, None] * magnitude) <= TOL


def test_paired_buckets_are_summarized_and_far_smaller_than_the_csr_rows(paired):
    plan, csr = paired["plan"], paired["csr"]
    assert payload_nbytes(plan.payload) == sum(s.payload_bytes for s in plan.bucket_summary) + plan.tensor_table_bytes
    kinds = {s.encoding for s in plan.bucket_summary}
    assert kinds >= {"csr", "tensor", "tensor_paired"}
    for s in plan.bucket_summary:
        parts = {"tensor": 1, "tensor_paired": 2}.get(s.encoding)
        if parts:
            assert s.width == parts * 112 and s.donor_entries == parts * 112 * s.targets
    paired_targets = sum(s.targets for s in plan.bucket_summary if s.encoding == "tensor_paired")
    assert paired_targets == int((paired["encoding"] == ENCODING_PAIRED).sum())
    assert payload_nbytes(plan.payload) < 0.7 * payload_nbytes(csr.payload)


def _resolve(part, row, tables, ringwise):
    """The factors a batch row reads through the global tables: ``part`` is the batch (A) or its ``mirror`` (B)."""
    out = dict(eta_index=tables.eta_index[part.t_eta[row]], eta_value=tables.eta_value[part.t_eta[row]],
               eta_derivative=tables.eta_derivative[part.t_eta[row]], radial=tables.radial[part.t_radial[row]])
    if ringwise:
        entry = part.t_ring[row]
        out.update(ring_owner=tables.ring_owner[entry], ring_value=tables.ring_value[entry],
                   ring_derivative=tables.ring_derivative[entry])
    else:
        out.update(layers=tables.layers[part.layer_id[row]], theta_index=tables.theta_index[part.t_theta[row]],
                   theta_value=tables.theta_value[part.t_theta[row]], theta_derivative=tables.theta_derivative[part.t_theta[row]])
    return out


def test_mirror_arrays_resolve_to_the_factors_of_part_b(paired):
    plan, factors = paired["plan"], paired["case"]["factors"]
    tables = plan.payload.tensor_tables
    value_slot = plan.targets.value_slot
    where = {}
    for batch in plan.payload.tensor_batches:
        for row, slot in enumerate(batch.value_slots):
            where[int(slot)] = (batch, row)
    checked = {"singleton": 0, "ringwise": 0}
    for s, f in enumerate(factors):
        if not isinstance(f, PairedFactors):
            continue
        batch, row = where[int(value_slot[s])]
        ringwise = batch.family == "ringwise"
        assert batch.family == f.a.family and batch.mirror is not None
        for part, expected in ((batch, f.a), (batch.mirror, f.b)):
            for name, array in _resolve(part, row, tables, ringwise).items():
                want = np.asarray(getattr(expected, name))
                want = want if name == "layers" else want[0]                  # one target per cell
                assert np.array_equal(np.asarray(array), want), (s, name)
        checked[batch.family] += 1
    assert min(checked.values()) >= 2
    # a single (unpaired) target has no mirror
    assert any(b.mirror is None for b in plan.payload.tensor_batches)


def test_theta_combos_cover_the_mirror_combinations(paired):
    plan = paired["plan"]
    tables = plan.payload.tensor_tables
    n = tables.n
    used, mirror_only = set(), set()
    for batch in plan.payload.tensor_batches:
        if batch.t_ring is not None:
            continue
        for part in (batch,) + (() if batch.mirror is None else (batch.mirror,)):
            layer = tables.layers[np.asarray(part.layer_id, dtype=np.int64)].astype(np.int64)
            row = np.asarray(part.t_theta, dtype=np.int64)[:, None]
            key = (row * 2 + (layer < 0)) * n + np.where(layer < 0, -layer - 1, layer)
            assert np.all(tables.combo_lookup[key] >= 0)
            assert np.array_equal(tables.combo_key[tables.combo_lookup[key]], key)
            used |= set(key.reshape(-1).tolist())
            if part is not batch:
                mirror_only |= set(key.reshape(-1).tolist())
    assert np.array_equal(tables.combo_key, np.array(sorted(used))) and mirror_only
    a_only = set()
    for batch in plan.payload.tensor_batches:
        if batch.t_ring is None:
            layer = tables.layers[np.asarray(batch.layer_id, dtype=np.int64)].astype(np.int64)
            a_only |= set(((np.asarray(batch.t_theta, dtype=np.int64)[:, None] * 2 + (layer < 0)) * n
                           + np.where(layer < 0, -layer - 1, layer)).reshape(-1).tolist())
    assert mirror_only - a_only                           # B reaches combinations no A part uses: they are listed too
    fields = np.random.default_rng(5).normal(size=(paired["grid"].n_owners, 2))
    context = kernel.prepare_fields(tables, plan.payload.tensor_batches, jnp.asarray(fields))
    tv = np.asarray(context.theta_value)
    raw = tables.raw_to_owner.reshape(n, n, n)
    for c in range(len(tables.combo_key)):
        a, s, r = tables.combo_key[c] // (2 * n), (tables.combo_key[c] // n) % 2, tables.combo_key[c] % n
        theta = (tables.theta_index[a] + s * (n // 2)) % n
        expect = np.einsum("j,jkf->kf", tables.theta_value[a], fields[raw[r, theta]])
        np.testing.assert_allclose(tv[c], expect, rtol=0, atol=1e-13 * np.abs(expect).max())


def test_tables_are_shared_between_the_two_parts_of_a_pair(paired):
    """A and B read the same deduplicated tables: the tables of the paired plan hold no more rows than the factors of
    both parts need, and nothing that is only a copy."""
    tables = paired["plan"].payload.tensor_tables
    for names in (("theta_index", "theta_value", "theta_derivative"), ("eta_index", "eta_value", "eta_derivative"),
                  ("radial",), ("layers",), ("ring_owner", "ring_value", "ring_derivative")):
        rows = np.concatenate([np.ascontiguousarray(getattr(tables, n)).reshape(len(getattr(tables, n)), -1)
                               .astype(np.float64 if getattr(tables, n).dtype.kind == "f" else np.int64)
                               .view(np.uint64) for n in names], axis=1)
        assert len(np.unique(rows, axis=0)) == len(rows), names


def test_pairs_with_several_targets_and_without_stored_gradient_lower_like_the_dense_chunk():
    """Sources of ``q = 3`` targets (jittered, one on the cell centre) with and without a stored gradient: B's targets
    follow A's, so every target finds its mirror; a value-only pair is a value-only paired batch."""
    context = _context(N, RINGS)
    S = StructuredReconstruction(context, cell_stencil="symmetric")
    rng = np.random.default_rng(5)
    rows, factors, keys = [], [], []
    for key in ((4, 3, 5), (4, 8, 1), (5, 6, 7), (8, 6, 11), (8, 2, 6), (9, 0, 0), (9, 5, 3)):
        centre = _cell(context, key)[0]
        points = centre + rng.uniform(-.2, .2, size=(3, 3)) * np.array([1 / N, 2 * np.pi / N, 2 * np.pi / N])
        points[0] = centre
        row, fac = S.rows_with_factors(key, points, "cell")
        assert isinstance(fac, PairedFactors)
        rows.append(row), factors.append(fac), keys.append(key)
    store = [True, False, True, True, False, False, True]
    data = encode_chunk("cells", _pack(rows, keys, store_gradient=store), factors=factors)
    grid = LoaderGrid.from_context(context)
    item = FactoredChunk(*art.decode_chunk_factored("cells", data))
    dense = decode_chunk("cells", data)
    assert (_arrays(data)["src_encoding"] == ENCODING_PAIRED).all()
    plan, csr = lower_point_chunks([item], grid=grid), lower_point_chunks([dense], grid=grid)
    _assert_same_layout(plan, csr)
    kinds = {(b.family, b.gradient_slots is not None) for b in _paired_batches(plan)}
    assert kinds == {(family, gradient) for family in ("singleton", "ringwise") for gradient in (True, False)}
    assert sum(len(b.value_slots) for b in _paired_batches(plan)) == 3 * len(keys)
    fields = _fields(context, 3)
    v_p, g_p = (np.asarray(x) for x in _apply(plan, fields))
    v_c, g_c = (np.asarray(x) for x in _apply(csr, fields))
    scale_v, scale_g = _weight_scales(dense)
    magnitude = np.abs(fields).max()
    assert _relative(v_p - v_c, np.maximum(scale_v, 1e-300)[:, None] * magnitude) <= TOL
    rows_g = np.flatnonzero(csr.targets.gradient_slot >= 0)
    slots = csr.targets.gradient_slot[rows_g]
    assert len(slots) == 3 * sum(store)
    assert _relative(g_p[slots] - g_c[slots], np.maximum(scale_g[rows_g], 1e-300)[:, :, None] * magnitude) <= TOL


def test_numpy_apply_merges_the_pairs_like_the_dense_rows(paired):
    """The NumPy reference contraction: a pair's two parts are two sources, ``merge_pairs`` makes it the mean."""
    case = paired["case"]
    _, tr, is_tensor = art.decode_chunk_factored("cells", paired["data"])
    fields = _fields(case["context"], 3)
    raw_v, raw_g = tr_mod.apply_tensor_rows_numpy(tr, fields)
    assert len(raw_v) == tr.n_targets and len(raw_g) == tr.n_targets                       # a row per part
    values, gradients = tr_mod.apply_tensor_rows_numpy(tr, fields, merge_pairs=True)
    sources = np.flatnonzero(is_tensor)
    assert len(values) == len(gradients) == len(sources)                                   # one target per cell
    for k, s in enumerate(sources):
        row = case["rows"][s]
        reference = row.value[0] @ fields[row.donor_ids]
        np.testing.assert_allclose(values[k], reference, rtol=0, atol=1e-12 * np.abs(reference).max())
        reference = np.einsum("ad,df->af", row.gradient[0], fields[row.donor_ids])
        np.testing.assert_allclose(gradients[k], reference, rtol=0, atol=1e-12 * np.abs(reference).max())
    only_v, no_g = tr_mod.apply_tensor_rows_numpy(tr, fields, gradients=False, merge_pairs=True)
    assert no_g is None and only_v.tobytes() == values.tobytes()
    no_v, only_g = tr_mod.apply_tensor_rows_numpy(tr, fields, values=False, merge_pairs=True)
    assert no_v is None and only_g.tobytes() == gradients.tobytes()


# ---------------------------------------------------------------------------
# the kernel
# ---------------------------------------------------------------------------
def _parts(batch):
    """The two single batches (A and B) a paired batch is the mean of."""
    m = batch.mirror
    a = dataclasses.replace(batch, mirror=None)
    b = dataclasses.replace(batch, mirror=None, t_eta=m.t_eta, t_radial=m.t_radial, layer_id=m.layer_id, t_theta=m.t_theta,
                            t_ring=m.t_ring)
    return a, b


def test_paired_apply_is_the_mean_of_its_two_parts(paired):
    plan = paired["plan"]
    tables = plan.payload.tensor_tables
    fields = jnp.asarray(_fields(paired["case"]["context"], 3))
    seen = set()
    for batch in _paired_batches(plan):
        a, b = _parts(batch)
        value, gradient = kernel.apply_tensor_batch(tables, batch, fields)
        (va, ga), (vb, gb) = kernel.apply_tensor_batch(tables, a, fields), kernel.apply_tensor_batch(tables, b, fields)
        scale = float(jnp.abs(va).max() + jnp.abs(vb).max())
        np.testing.assert_allclose(np.asarray(value), 0.5 * (np.asarray(va) + np.asarray(vb)), rtol=0, atol=1e-14 * scale)
        if gradient is not None:
            scale = float(jnp.abs(ga).max() + jnp.abs(gb).max())
            np.testing.assert_allclose(np.asarray(gradient), 0.5 * (np.asarray(ga) + np.asarray(gb)), rtol=0, atol=1e-14 * scale)
        seen.add((batch.family, gradient is not None))
    assert {"singleton", "ringwise"} <= {family for family, _ in seen}


def test_paired_block_processing_matches_the_single_block(paired, monkeypatch):
    plan = paired["plan"]
    tables = plan.payload.tensor_tables
    fields = jnp.asarray(_fields(paired["case"]["context"], 3))
    done = 0
    for batch in _paired_batches(plan):
        full = kernel.apply_tensor_batch(tables, batch, fields)
        monkeypatch.setattr(kernel, "ANGULAR_BLOCK_ELEMENTS", 32 * 3 * 3)         # 3 paired targets per block
        monkeypatch.setattr(kernel, "RING_BLOCK_ELEMENTS", 32 * 3 * 3)
        monkeypatch.setattr(kernel, "COMBO_BLOCK_ELEMENTS", 7 * 4 * 3 * 3)
        monkeypatch.setattr(kernel, "SINGLE_BLOCK_BYTES", 0)
        blocked = kernel.apply_tensor_batch(tables, batch, fields)
        monkeypatch.undo()
        assert len(batch.value_slots) > 3
        for x, y in zip(full, blocked):
            if x is not None:
                np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=0, atol=1e-14 * np.abs(np.asarray(x)).max())
        done += 1
    assert done >= 2


def test_paired_eager_equals_jit_and_jvp_is_the_applied_tangent(paired):
    plan, context = paired["plan"], paired["case"]["context"]
    fields = _fields(context, 4)
    boundary = _boundary(plan.boundary_points, 4)
    eager = _apply(plan, fields)
    jitted = jax.jit(apply_source_rows, static_argnames=("values", "gradients"))(plan.payload, fields, boundary)
    for a, b in zip(eager, jitted):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    tangent = _fields(context, 4, seed=8)
    _, jvp = jax.jvp(lambda f: _apply(plan, f), (jnp.asarray(fields),), (jnp.asarray(tangent),))
    zero = BoundaryArrays(np.zeros_like(boundary.values), np.zeros_like(boundary.tangential_gradients))
    linear = apply_source_rows(plan.payload, tangent, zero)
    for a, b in zip(jvp, linear):
        scale = float(np.abs(np.asarray(b)).max())
        assert float(np.abs(np.asarray(a) - np.asarray(b)).max()) / scale <= TOL
    v = _apply(plan, fields, gradients=False)
    g = _apply(plan, fields, values=False)
    assert v[1] is None and g[0] is None
    np.testing.assert_allclose(np.asarray(v[0]), np.asarray(eager[0]), rtol=0, atol=TOL * np.abs(np.asarray(eager[0])).max())
    np.testing.assert_allclose(np.asarray(g[1]), np.asarray(eager[1]), rtol=0, atol=TOL * np.abs(np.asarray(eager[1])).max())


# ---------------------------------------------------------------------------
# selections, chunks and files
# ---------------------------------------------------------------------------
def test_selection_of_paired_and_csr_sources(paired):
    case, plan_item, dense, grid = paired["case"], paired["item"], paired["dense"], paired["grid"]
    first = np.asarray(dense.source_ptr)[:-1]
    entity = np.asarray(dense.entity_id)[first]
    encoding = paired["encoding"]
    fields = _fields(case["context"])
    selects = {"pairs": RowSelection(entity_ids=entity[encoding == ENCODING_PAIRED]),
               "pairs+singles": RowSelection(entity_ids=entity[encoding > 0]),
               "a few pairs": RowSelection(entity_ids=entity[encoding == ENCODING_PAIRED][::3]),
               "csr only": RowSelection(entity_ids=entity[encoding == 0])}
    for name, select in selects.items():
        plan_t = lower_point_chunks([plan_item], grid=grid, select=select)
        plan_c = lower_point_chunks([dense], grid=grid, select=select)
        _assert_same_layout(plan_t, plan_c)
        v_t, g_t = (np.asarray(x) for x in _apply(plan_t, fields))
        v_c, g_c = (np.asarray(x) for x in _apply(plan_c, fields))
        np.testing.assert_allclose(v_t, v_c, rtol=0, atol=1e-12 * np.abs(v_c).max())
        np.testing.assert_allclose(g_t, g_c, rtol=0, atol=1e-12 * np.abs(g_c).max())
        assert bool(_paired_batches(plan_t)) == (name != "csr only"), name
    csr_only = lower_point_chunks([plan_item], grid=grid, select=selects["csr only"])
    assert csr_only.payload.tensor_tables is None and not csr_only.payload.tensor_batches


def test_tables_merge_across_chunks_and_a_chunk_order_does_not_matter(paired):
    case, grid = paired["case"], paired["grid"]
    context, rows, factors = case["context"], case["rows"], case["factors"]
    order = np.random.default_rng(4).permutation(len(KEYS))
    items, denses = [], []
    for part in np.array_split(order, 3):                       # three shuffled chunks, each its own artifact chunk
        chunk = _pack([rows[s] for s in part], [KEYS[s] for s in part])
        data = encode_chunk("cells", chunk, factors=[factors[s] for s in part])
        items.append(FactoredChunk(*art.decode_chunk_factored("cells", data)))
        denses.append(decode_chunk("cells", data))
    merged, reference = lower_point_chunks(items, grid=grid), lower_point_chunks(denses, grid=grid)
    _assert_same_layout(merged, reference)
    assert _paired_batches(merged)
    fields = _fields(context)
    for a, b in zip(_apply(merged, fields), _apply(reference, fields)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12 * np.abs(np.asarray(b)).max())
    # chunk by chunk, the tables of the merge are deduplicated by bit pattern
    tables = merged.payload.tensor_tables
    for names in (("theta_index", "theta_value", "theta_derivative"), ("eta_index", "eta_value", "eta_derivative"),
                  ("radial",), ("layers",), ("ring_owner", "ring_value", "ring_derivative")):
        rows_ = np.concatenate([np.ascontiguousarray(getattr(tables, n)).reshape(len(getattr(tables, n)), -1)
                                .astype(np.float64 if getattr(tables, n).dtype.kind == "f" else np.int64)
                                .view(np.uint64) for n in names], axis=1)
        assert len(np.unique(rows_, axis=0)) == len(rows_), names


def test_artifact_directory_and_files_lower_to_the_same_paired_plan(paired, tmp_path):
    case, grid = paired["case"], paired["grid"]
    identity = art.build_identity(component_hashes={"geometry": "pair"}, source_hashes={}, policy={"pair": 1})
    root = art.save_row_artifact(tmp_path, N, identity=identity, cells=[case["chunk"]], point_factors={"cells": [case["factors"]]})
    plan_dir = lower_point_chunks(iter_factored_point_chunks(tmp_path, N, identity, groups=("cells",)), grid=grid)
    plan_file = lower_point_files([root / "rows" / "cells_0.npz"], grid=grid)
    for plan in (plan_dir, plan_file):
        _assert_same_layout(plan, paired["plan"])
        assert _paired_batches(plan)
        for a, b in zip(jax.tree_util.tree_leaves(plan.payload), jax.tree_util.tree_leaves(paired["plan"].payload)):
            assert np.array_equal(np.asarray(a), np.asarray(b))


def test_corrupt_pair_markers_and_counts_are_rejected_by_the_loader(paired):
    grid, item = paired["grid"], paired["item"]
    tr = item.tensor_rows
    a = int(np.flatnonzero(tr.pair_part == 1)[0])
    unmarked = dataclasses.replace(tr, pair_part=np.where(np.arange(tr.n_sources) == a + 1, 0, tr.pair_part).astype(np.int8))
    with pytest.raises(ValueError, match="A part followed by its B part"):
        lower_point_chunks([FactoredChunk(item.chunk, unmarked, item.is_tensor)], grid=grid)
    flat = dataclasses.replace(tr, pair_part=np.zeros(tr.n_sources, dtype=np.int8))         # pairs read as singles: too many rows
    with pytest.raises(ValueError, match="tensor rows hold"):
        lower_point_chunks([FactoredChunk(item.chunk, flat, item.is_tensor)], grid=grid)
    far = dataclasses.replace(tr, layers=np.where(np.arange(tr.n_sources)[:, None] == a + 1, N + 5, tr.layers).astype(tr.layers.dtype))
    with pytest.raises(ValueError, match="radial layer outside the grid"):
        lower_point_chunks([FactoredChunk(item.chunk, far, item.is_tensor)], grid=grid)                    # B's layers are checked


# ---------------------------------------------------------------------------
# the eta sharding
# ---------------------------------------------------------------------------
def _layout(context):
    perm, inverse, m = plane_major_permutation(context.ro, context.n)
    return PlaneLayout(perm, inverse, m)


def test_selection_of_paired_batches_reproduces_the_global_apply_on_the_owned_cells(paired):
    case, plan = paired["case"], paired["plan"]
    context, payload = case["context"], plan.payload
    keys = np.array(KEYS)
    raw = np.ravel_multi_index(keys.T, (N,) * 3)
    owner = context.ro[raw]
    ref_v, ref_g = (np.asarray(x) for x in _apply(plan, _fields(context)))
    fields_global = _fields(context)
    boundary = _boundary(plan.boundary_points, fields_global.shape[1])
    for n_shards in (2, 3):
        ctx = _Ctx(_layout(context), N, n_shards, DEFAULT_HALO)
        for s in range(n_shards):
            sh = _Shard(ctx, s)
            owned = sh.owned[owner]
            if not owned.any():
                continue
            need_v = np.zeros(payload.n_targets, dtype=bool)
            need_g = np.zeros(payload.n_gradient_targets, dtype=bool)
            need_v[plan.targets.value_slot[owned]] = True
            need_g[plan.targets.gradient_slot[owned]] = True
            local, vmap, gmap = _select_sources(payload, need_v, need_g, sh, "paired")
            assert any(b.mirror is not None and len(b.value_slots) for b in local.tensor_batches)
            fields = np.concatenate([fields_global[sh.ext_old], np.zeros((1, fields_global.shape[1]))])
            v, g = (np.asarray(x) for x in apply_source_rows(local, fields, boundary))
            slots = np.flatnonzero(need_v)
            assert np.abs(v[vmap[slots]] - ref_v[slots]).max() <= 1e-13 * max(np.abs(ref_v).max(), 1.0)
            gslots = np.flatnonzero(need_g)
            assert np.abs(g[gmap[gslots]] - ref_g[gslots]).max() <= 1e-13 * max(np.abs(ref_g).max(), 1.0)


def _lowered(keys):
    """``(context, rows, factors, lowered plan)`` of the symmetric cell rows ``keys`` (paired batches stay factored)."""
    context = _context(N, RINGS)
    S = StructuredReconstruction(context, cell_stencil="symmetric")
    rows, factors = zip(*(S.rows_with_factors(key, _cell(context, key), "cell") for key in keys))
    data = encode_chunk("cells", _pack(rows, keys), factors=list(factors))
    plan = lower_point_chunks([FactoredChunk(*art.decode_chunk_factored("cells", data))], grid=LoaderGrid.from_context(context))
    return context, list(rows), list(factors), plan


def test_the_window_check_covers_part_b_of_a_pair():
    """Cells on the top owned plane ``k`` of a shard read the planes ``k-2..k+1`` through A (inside a halo of 1) but
    ``k-1..k+2`` through B (one plane beyond it): with B replaced by a copy of A the selection passes, with the real B
    it must raise (singleton and ringwise pairs both)."""
    keys = [(i, j, 5) for i in (4, 5, 7, 8) for j in (1, 6)]                      # ringwise and singleton pairs on plane 5
    context, _, factors, plan = _lowered(keys)
    assert all(isinstance(f, PairedFactors) for f in factors) and {b.family for b in _paired_batches(plan)} == {"singleton", "ringwise"}
    payload = plan.payload
    ctx = _Ctx(_layout(context), N, 4, 1)                             # 3 planes per shard, a halo of 1
    sh = _Shard(ctx, 1)                                                # owns planes 3..5, window 2..6
    need_v = np.ones(payload.n_targets, dtype=bool)
    need_g = np.ones(payload.n_gradient_targets, dtype=bool)
    with pytest.raises(ValueError, match="outside the extended window"):
        _select_sources(payload, need_v, need_g, sh, "paired")

    def mirror_is_a(b):
        return b if b.mirror is None else dataclasses.replace(
            b, mirror=TensorMirror(b.t_eta, b.t_radial, b.layer_id, b.t_theta, b.t_ring))

    a_only = dataclasses.replace(payload, tensor_batches=tuple(mirror_is_a(b) for b in payload.tensor_batches))
    local, _, _ = _select_sources(a_only, need_v, need_g, sh, "paired")        # part A alone is inside the window
    assert all(len(b.value_slots) for b in local.tensor_batches)
    # one shard further up the same cells are inside the window on both sides only with a halo of 2
    assert _select_sources(payload, need_v, need_g, _Shard(_Ctx(_layout(context), N, 4, 2), 1), "paired")[0].tensor_batches


def _cells_only_plan(context, keys, lowered):
    """A cells-only ``PerpendicularPlan`` of unconditioned symmetric cell rows (geometry arrays are placeholders)."""
    n_cells = len(keys)
    raw = np.ravel_multi_index(np.array(keys).T, (N,) * 3)
    cells = CellPlan(
        rows=lowered.payload, value_slot=lowered.targets.value_slot.astype(np.int32),
        gradient_slot=lowered.targets.gradient_slot.astype(np.int32), conditioned=np.zeros(n_cells, dtype=bool), neumann=None,
        neumann_cell=np.zeros(0, dtype=np.int32), raw_ids=raw.astype(np.int64), raw_owner=context.ro[raw].astype(np.int32),
        raw_volume=np.ones(n_cells), owner_volume=np.asarray(context.vol, dtype=np.float64), evolution_volume=np.ones(len(context.vol)),
        h=np.ones((n_cells, 3)), jac=np.ones(n_cells), B=np.ones(n_cells), K=np.zeros((n_cells, 3)), J=np.ones(n_cells),
        weight=np.ones(n_cells), evolution_weight=np.ones(n_cells))
    return PerpendicularPlan(cells, None, None, np.zeros((0, 3)), np.zeros((0, 3)), N)


@pytest.mark.parametrize("n_shards", (1, 2, 4))
def test_sharded_cell_plan_applies_the_owned_paired_cells_to_the_single_device_values(n_shards):
    # rings 3..9 on every eta plane: a ringwise single, ringwise pairs, the mixed ring (CSR) and singleton pairs
    keys = [(i, (5 * k + 1) % N, k) for i in (3, 4, 5, 6, 7, 8, 9) for k in range(N)]
    context, _, _, lowered = _lowered(keys)
    plan = _cells_only_plan(context, keys, lowered)
    assert _paired_batches(lowered) and any(b.mirror is None for b in lowered.payload.tensor_batches) and lowered.payload.batches
    sharded = shard_perpendicular_plan(plan, context.ro, N, n_shards, DEFAULT_HALO)
    fields = np.random.default_rng(6).normal(size=(len(context.vol), 3))
    ref_v, ref_g = (np.asarray(x) for x in apply_source_rows(lowered.payload, fields, None))
    ctx = _Ctx(sharded.layout, N, n_shards, DEFAULT_HALO)
    owner = np.asarray(plan.cells.raw_owner)
    padded = None
    for s in range(n_shards):
        sh = _Shard(ctx, s)
        local = local_plan(sharded, s)
        kept = np.flatnonzero(sh.owned[owner])
        rows_s = local.cells.rows
        padded = padded or [(b.mirror is not None, tuple(a.shape for a in jax.tree_util.tree_leaves(b)))
                            for b in rows_s.tensor_batches]
        assert [(b.mirror is not None, tuple(a.shape for a in jax.tree_util.tree_leaves(b)))
                for b in rows_s.tensor_batches] == padded                        # identical shapes on every shard
        fields_local = np.concatenate([fields[sh.ext_old], np.zeros((1, 3))])
        v, g = (np.asarray(x) for x in apply_source_rows(rows_s, fields_local, None))
        # local cell ``c`` of the kept cells (padded cells follow) is global cell ``kept[c]``
        vs = np.asarray(local.cells.value_slot)[:len(kept)]
        gs = np.asarray(local.cells.gradient_slot)[:len(kept)]
        assert np.abs(v[vs] - ref_v[plan.cells.value_slot[kept]]).max() <= 1e-13 * np.abs(ref_v).max()
        assert np.abs(g[gs] - ref_g[plan.cells.gradient_slot[kept]]).max() <= 1e-13 * np.abs(ref_g).max()
    assert sum(len(np.flatnonzero(_Shard(ctx, s).owned[owner])) for s in range(n_shards)) == len(keys)
