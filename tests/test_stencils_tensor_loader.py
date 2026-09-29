"""Runtime of tensor-encoded point sources (P08 step 2, task D2): loader lowering + JAX kernel.

Fast synthetic tests reuse the toy-grid constructions of ``tests/test_stencils_tensor_rows.py`` (every
factored family: singleton incl. wrapped radial layers and cross-plane aggregate owners, ringwise incl.
an aggregate owner spanning two planes, centered_radial, next to CSR-only sources in the same chunk).
Each chunk is lowered twice, from its factored decode (tensor sources never expanded) and from the
expanded CSR rows; the two plans must have identical slots/targets and agree in the applied values,
gradients and against ``apply_tensor_rows_numpy``. The real-geometry gate is in
``tests/test_stencils_tensor_loader_real.py`` (slow).
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.native import fci_perpendicular_tensor_rows as kernel
from drbx.native.fci_perpendicular_point_rows import BoundaryArrays
from drbx.native.fci_perpendicular_source_rows import apply_source_rows, payload_nbytes
from drbx.stencils import artifact as art
from drbx.stencils.loader import (
    FactoredChunk, LoaderGrid, RowSelection, iter_factored_point_chunks, lower_point_chunks,
    lower_point_files)
from drbx.stencils.tensor_rows import apply_tensor_rows_numpy
from tests.test_stencils_tensor_rows import (
    IDENTITY_PLAN, PLANE_MERGE, RING_MERGE, RING_PLAN, _context, _sources)

TOL = 1e-13
MAXIMA: dict[str, float] = {}


def _factored(chunk, factors):
    """(factored item, expanded CSR chunk, TensorRows, is_tensor) through the real v3 codec."""
    data = art.encode_chunk("faces", chunk, factors=factors)
    item = FactoredChunk(*art.decode_chunk_factored("faces", data))
    assert item.tensor_rows is not None and item.is_tensor.any()
    dense = art.decode_chunk("faces", data)
    return item, dense


@pytest.fixture(scope="module")
def cases():
    out = {}
    for name, merges, plan in (("identity", (), IDENTITY_PLAN), ("ring", RING_MERGE, RING_PLAN),
                               ("plane", PLANE_MERGE, RING_PLAN)):
        context = _context(merges)
        _, rows, factors, chunk = _sources(context, plan)
        item, dense = _factored(chunk, factors)
        out[name] = dict(context=context, grid=LoaderGrid.from_context(context), item=item, dense=dense,
                         chunk=chunk)
    return out


def _weight_scales(dense):
    """Per target: sum |w| of the value row and of each gradient component (0 without gradient)."""
    widths = np.diff(dense.donor_ptr)
    target = np.repeat(np.arange(len(widths)), widths)
    value = np.bincount(target, weights=np.abs(dense.value), minlength=len(widths))
    gradient = np.zeros((len(widths), 3))
    g_widths = np.diff(dense.gradient_ptr)
    g_target = np.repeat(np.arange(len(g_widths)), g_widths)
    for a in range(3):
        gradient[:, a] = np.bincount(g_target, weights=np.abs(dense.gradient[a]), minlength=len(widths))
    return value, gradient


def _relative(diff, scale):
    return float(np.max(np.abs(diff) / scale)) if diff.size else 0.0


def _boundary(points, nf):
    """Analytic prescribed g and tangential derivatives at the plan's boundary query points."""
    if not len(points):
        return None
    r, th, eta = points.T
    scale = 1 + 0.5 * np.arange(nf)
    values = np.sin(2 * r + th)[:, None] * np.cos(eta)[:, None] * scale
    tangent = np.stack((np.cos(th + eta)[:, None] * scale, np.sin(2 * eta - r)[:, None] * scale), axis=1)
    return BoundaryArrays(values, tangent)


def _apply(plan, fields, **kwargs):
    """``apply_source_rows`` with the analytic boundary arrays of the plan's own queries (if any)."""
    return apply_source_rows(plan.payload, fields, _boundary(plan.boundary_points, fields.shape[1]), **kwargs)


def _fields(context, nf=3, seed=3):
    return np.random.default_rng(seed).normal(size=(len(context.vol), nf))


def _tensor_slots(item):
    """Value slots of the tensor targets (chunk order); all sources are selected, slot == chunk target."""
    per_target = np.repeat(item.is_tensor, np.diff(item.chunk.source_ptr))
    return np.flatnonzero(per_target)


def _assert_same_layout(a, b):
    for x, y in zip(a.targets, b.targets):
        assert x.dtype == y.dtype and np.array_equal(x, y)
    for name in ("source_target_ptr", "source_eta_offset_min", "source_eta_offset_max"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.family_names == b.family_names
    assert (a.payload.n_targets, a.payload.n_gradient_targets, a.payload.n_boundary_queries) == \
           (b.payload.n_targets, b.payload.n_gradient_targets, b.payload.n_boundary_queries)


@pytest.mark.parametrize("name", ("identity", "ring", "plane"))
def test_tensor_lowering_matches_csr_lowering_and_numpy_reference(name, cases):
    case = cases[name]
    item, dense, grid = case["item"], case["dense"], case["grid"]
    plan_t = lower_point_chunks([item], grid=grid)
    plan_c = lower_point_chunks([dense], grid=grid)
    _assert_same_layout(plan_t, plan_c)                   # identical slots/targets/halo
    assert plan_t.payload.tensor_batches and not plan_c.payload.tensor_batches
    fields = _fields(case["context"])
    v_t, g_t = (np.asarray(x) for x in _apply(plan_t, fields))
    v_c, g_c = (np.asarray(x) for x in _apply(plan_c, fields))
    scale_v, scale_g = _weight_scales(dense)
    magnitude = np.abs(fields).max()
    err_v = _relative(v_t - v_c, np.maximum(scale_v, 1e-300)[:, None] * magnitude)
    rows = np.flatnonzero(plan_c.targets.gradient_slot >= 0)
    slots = plan_c.targets.gradient_slot[rows]
    err_g = _relative(g_t[slots] - g_c[slots], np.maximum(scale_g[rows], 1e-300)[:, :, None] * magnitude)
    # ... and against the NumPy reference contraction of the tensor targets
    tr = item.tensor_rows
    ref_v, ref_g = apply_tensor_rows_numpy(tr, fields)
    tslots = _tensor_slots(item)
    err_rv = _relative(v_t[tslots] - ref_v, np.maximum(scale_v[tslots], 1e-300)[:, None] * magnitude)
    tg = plan_t.targets.gradient_slot[tslots]
    has = tg >= 0
    err_rg = _relative(g_t[tg[has]] - ref_g, np.maximum(scale_g[tslots][has], 1e-300)[:, :, None] * magnitude)
    for key, err in (("value vs CSR", err_v), ("gradient vs CSR", err_g), ("value vs numpy", err_rv),
                     ("gradient vs numpy", err_rg)):
        MAXIMA[f"{name}: {key}"] = err
        assert err <= TOL, (name, key, err)
    assert len(tslots) and has.any()


def test_families_and_wrapped_layers_are_covered(cases):
    families = set()
    for case in cases.values():
        plan = lower_point_chunks([case["item"]], grid=case["grid"])
        families |= {s.family for s in plan.bucket_summary if s.encoding == "tensor"}
        assert plan.bucket_summary[-1].encoding == "tensor"
    assert families == {"singleton", "ringwise", "centered_radial"}
    tables = lower_point_chunks([cases["identity"]["item"]], grid=cases["identity"]["grid"]).payload.tensor_tables
    assert (tables.layers < 0).any()                        # wrapped radial layer across the axis


def test_theta_combos_are_exactly_the_used_ring_shift_layer_keys(cases):
    """``combo_key`` lists the (theta row, half-turn shift, radial layer) keys the singleton /
    centered_radial targets derive (a layer below the axis wraps to ``-l-1`` with a half-turn shift);
    ``combo_lookup`` finds every one of them, and nothing else is listed. Each precontracted table row equals
    the theta contraction of the raw cells it stands for."""
    for name in ("identity", "plane"):
        case = cases[name]
        plan = lower_point_chunks([case["item"]], grid=case["grid"])
        tables = plan.payload.tensor_tables
        n = tables.n
        used = set()
        for batch in plan.payload.tensor_batches:
            if batch.t_ring is not None:
                continue
            layer = tables.layers[np.asarray(batch.layer_id, dtype=np.int64)].astype(np.int64)
            row = np.asarray(batch.t_theta, dtype=np.int64)[:, None]
            key = (row * 2 + (layer < 0)) * n + np.where(layer < 0, -layer - 1, layer)
            assert np.all(tables.combo_lookup[key] >= 0)
            assert np.array_equal(tables.combo_key[tables.combo_lookup[key]], key)
            used |= set(key.reshape(-1).tolist())
        assert np.array_equal(tables.combo_key, np.array(sorted(used)))
        assert (((tables.combo_key // n) % 2) == 1).any() == (tables.layers < 0).any()      # shifted combos exist
        assert (tables.combo_lookup >= 0).sum() == len(tables.combo_key)
        fields = np.random.default_rng(5).normal(size=(case["grid"].n_owners, 2))
        context = kernel.prepare_fields(tables, plan.payload.tensor_batches, jnp.asarray(fields))
        tv = np.asarray(context.theta_value)
        raw = tables.raw_to_owner.reshape(n, n, n)
        for c in range(len(tables.combo_key)):
            a, s, r = tables.combo_key[c] // (2 * n), (tables.combo_key[c] // n) % 2, tables.combo_key[c] % n
            theta = (tables.theta_index[a] + s * (n // 2)) % n
            expect = np.einsum("j,jkf->kf", tables.theta_value[a], fields[raw[r, theta]])
            np.testing.assert_allclose(tv[c], expect, rtol=0, atol=1e-13 * np.abs(expect).max())


def test_ring_aggregate_owner_spans_two_planes_and_accumulates(cases):
    """Some ringwise target reaches one owner through several (layer, plane, slot) entries, so the
    contraction sums those donors (the CSR expansion accumulates them into one column)."""
    tr = cases["ring"]["item"].tensor_rows
    flat = tr.ring_owner[tr.t_ring].reshape(len(tr.t_ring), -1)
    assert any(len(np.unique(row)) < flat.shape[1] for row in flat)
    plan = lower_point_chunks([cases["ring"]["item"]], grid=cases["ring"]["grid"])
    assert any(b.family == "ringwise" for b in plan.payload.tensor_batches)


def test_mixed_chunk_slots_and_csr_buckets_unchanged(cases):
    case = cases["identity"]
    plan_t = lower_point_chunks([case["item"]], grid=case["grid"])
    plan_c = lower_point_chunks([case["dense"]], grid=case["grid"])
    csr_t = [s for s in plan_t.bucket_summary if s.encoding == "csr"]
    tensor_sources = int(case["item"].is_tensor.sum())
    assert 0 < tensor_sources < len(case["item"].is_tensor)
    assert sum(s.sources for s in csr_t) + tensor_sources == sum(s.sources for s in plan_c.bucket_summary)
    # the CSR buckets that exist in both plans are bit-identical arrays
    by_key = lambda plan: {(s.family, s.conditioned, s.has_gradient, s.nodes, s.width): b
                           for s, b in zip(plan.bucket_summary, plan.payload.batches)
                           if s.encoding == "csr"}
    only_csr_t, in_c = by_key(plan_t), by_key(plan_c)
    assert set(only_csr_t) <= set(in_c)
    for key, batch in only_csr_t.items():
        assert batch.donor_ids.shape[0] <= in_c[key].donor_ids.shape[0]
    # every tensor target has exactly one slot, disjoint from the CSR targets' rows
    slots = np.concatenate([np.asarray(b.value_slots).reshape(-1) for b in plan_t.payload.batches]
                           + [b.value_slots for b in plan_t.payload.tensor_batches])
    assert np.array_equal(np.sort(slots), np.arange(plan_t.payload.n_targets))


def test_eager_jit_bitwise_and_jvp_is_the_applied_tangent(cases):
    case = cases["identity"]
    plan = lower_point_chunks([case["item"]], grid=case["grid"])
    fields = _fields(case["context"], 4)
    eager = _apply(plan, fields)
    boundary = _boundary(plan.boundary_points, 4)
    jitted = jax.jit(apply_source_rows, static_argnames=("values", "gradients"))(plan.payload, fields, boundary)
    closed = jax.jit(lambda f: _apply(plan, f))(fields)
    for a, b in zip(eager, jitted):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    # A payload closed over as constants may be constant-folded by XLA: the tensor rows stay bitwise
    # equal; (the one-target CSR source of B's buckets can differ in the last bit -- also in the all-CSR plan).
    tensor_rows = np.concatenate([b.value_slots for b in plan.payload.tensor_batches])
    tensor_grad = np.concatenate([b.gradient_slots for b in plan.payload.tensor_batches
                                  if b.gradient_slots is not None])
    assert np.array_equal(np.asarray(eager[0])[tensor_rows], np.asarray(closed[0])[tensor_rows])
    assert np.array_equal(np.asarray(eager[1])[tensor_grad], np.asarray(closed[1])[tensor_grad])
    pure = lower_point_chunks([cases["ring"]["item"]], grid=cases["ring"]["grid"])
    ring_fields = _fields(cases["ring"]["context"], 4)
    ring_boundary = _boundary(pure.boundary_points, 4)
    for a, b in zip(apply_source_rows(pure.payload, ring_fields, ring_boundary),
                    jax.jit(lambda f: apply_source_rows(pure.payload, f, ring_boundary))(ring_fields)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    tangent = _fields(case["context"], 4, seed=8)
    _, jvp = jax.jvp(lambda f: _apply(plan, f), (jnp.asarray(fields),), (jnp.asarray(tangent),))
    zero = BoundaryArrays(np.zeros_like(boundary.values), np.zeros_like(boundary.tangential_gradients))
    linear = apply_source_rows(plan.payload, tangent, zero)
    for a, b in zip(jvp, linear):
        scale = float(np.abs(np.asarray(b)).max())
        err = float(np.abs(np.asarray(a) - np.asarray(b)).max()) / scale
        MAXIMA["jvp"] = max(MAXIMA.get("jvp", 0.0), err)
        assert err <= TOL
    # value-only / gradient-only entry points agree with the joint call to rounding
    v = _apply(plan, fields, gradients=False)
    g = _apply(plan, fields, values=False)
    assert v[1] is None and g[0] is None
    np.testing.assert_allclose(np.asarray(v[0]), np.asarray(eager[0]), rtol=0, atol=TOL * np.abs(np.asarray(eager[0])).max())
    np.testing.assert_allclose(np.asarray(g[1]), np.asarray(eager[1]), rtol=0, atol=TOL * np.abs(np.asarray(eager[1])).max())


@pytest.mark.parametrize("name", ("identity", "ring"))
def test_block_processing_matches_single_block(name, cases, monkeypatch):
    """The lax.map path (padded blocks) equals the one-block path (called op by op, not through the jit cache)."""
    case = cases[name]
    plan = lower_point_chunks([case["item"]], grid=case["grid"])
    fields = jnp.asarray(_fields(case["context"], 3))
    tables = plan.payload.tensor_tables
    for batch in plan.payload.tensor_batches[::2]:                       # keeps the eager compiles few
        full = kernel.apply_tensor_batch(tables, batch, fields)
        monkeypatch.setattr(kernel, "ANGULAR_BLOCK_ELEMENTS", 16 * 3 * 5)        # 5 targets per block
        monkeypatch.setattr(kernel, "COMBO_BLOCK_ELEMENTS", 7 * 4 * 3 * 3)       # 3 theta combos per block
        monkeypatch.setattr(kernel, "RING_BLOCK_ELEMENTS", 16 * 3 * 5)
        blocked = kernel.apply_tensor_batch(tables, batch, fields)
        monkeypatch.undo()
        assert full[0].shape == (len(batch.value_slots), 3)
        assert len(batch.value_slots) > 5
        for a, b in zip(full, blocked):
            if a is not None:
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-14 * np.abs(np.asarray(a)).max())


def test_selection_by_request_variant_and_entity(cases):
    case = cases["identity"]
    item, dense, grid = case["item"], case["dense"], case["grid"]
    fields = _fields(case["context"])
    first = np.asarray(dense.source_ptr)[:-1]
    entity = np.asarray(dense.entity_id)[first]
    tensor_entities = entity[item.is_tensor]
    selects = (RowSelection(requests=("R2",)), RowSelection(requests=("R1", "R3")),
               RowSelection(bc_variants=("",)), RowSelection(bc_variants=("D",)),
               RowSelection(entity_ids=tensor_entities[:2]), RowSelection(requests=("R2",), entity_ids=entity[:5]))
    saw_tensor = 0
    for select in selects:
        plan_t = lower_point_chunks([item], grid=grid, select=select)
        plan_c = lower_point_chunks([dense], grid=grid, select=select)
        _assert_same_layout(plan_t, plan_c)
        if plan_t.payload.n_targets == 0:
            continue
        saw_tensor += bool(plan_t.payload.tensor_batches)
        v_t, g_t = (np.asarray(x) for x in _apply(plan_t, fields))
        v_c, g_c = (np.asarray(x) for x in _apply(plan_c, fields))
        np.testing.assert_allclose(v_t, v_c, rtol=0, atol=1e-12 * np.abs(v_c).max())
        if g_c.size:
            np.testing.assert_allclose(g_t, g_c, rtol=0, atol=1e-12 * np.abs(g_c).max())
    assert saw_tensor >= 4
    empty = lower_point_chunks([item], grid=grid, select=RowSelection(entity_ids=np.array([10 ** 6])))
    assert empty.payload.n_targets == 0 and empty.payload.tensor_tables is None


def test_tables_are_merged_across_chunks_and_deduplicated_by_bit_pattern():
    context = _context()
    grid = LoaderGrid.from_context(context)
    items, denses = [], []
    for seed, plan in ((0, IDENTITY_PLAN[:6]), (1, IDENTITY_PLAN[6:]), (0, IDENTITY_PLAN[:6])):
        _, _, factors, chunk = _sources(context, plan, seed=seed)
        item, dense = _factored(chunk, factors)
        items.append(item), denses.append(dense)
    merged = lower_point_chunks(items, grid=grid)
    reference = lower_point_chunks(denses, grid=grid)
    _assert_same_layout(merged, reference)
    tables = merged.payload.tensor_tables
    for names in (("theta_index", "theta_value", "theta_derivative"), ("eta_index", "eta_value", "eta_derivative"),
                  ("radial",), ("layers",)):
        rows = np.concatenate([np.ascontiguousarray(getattr(tables, n)).reshape(len(getattr(tables, n)), -1)
                               .astype(np.float64 if getattr(tables, n).dtype.kind == "f" else np.int64)
                               .view(np.uint64) for n in names], axis=1)
        assert len(np.unique(rows, axis=0)) == len(rows), names
    # the rows of one chunk are a subset of the merged tables: lowering chunk 0 alone needs no more rows
    single = lower_point_chunks(items[:1], grid=grid).payload.tensor_tables
    assert len(single.eta_value) <= len(tables.eta_value) <= sum(
        len(i.tensor_rows.eta_value) for i in items)
    assert len(lower_point_chunks(items[:1] + items[2:], grid=grid).payload.tensor_tables.eta_value) \
        == len(single.eta_value)                              # chunk 2 repeats chunk 0's factors: nothing new
    fields = _fields(context)
    for a, b in zip(_apply(merged, fields), _apply(reference, fields)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12 * np.abs(np.asarray(b)).max())


def test_memory_accounting_covers_tensor_batches_and_tables(cases):
    for case in cases.values():
        plan = lower_point_chunks([case["item"]], grid=case["grid"])
        assert payload_nbytes(plan.payload) == sum(s.payload_bytes for s in plan.bucket_summary) \
            + plan.tensor_table_bytes
        for summary in plan.bucket_summary:
            if summary.encoding == "tensor":
                assert summary.width == 112 and summary.donor_entries == 112 * summary.targets
        assert plan.tensor_table_bytes > 0
    plan = lower_point_chunks([cases["ring"]["dense"]], grid=cases["ring"]["grid"])
    assert plan.tensor_table_bytes == 0 and all(s.encoding == "csr" for s in plan.bucket_summary)


def test_old_layout_bytes_needs_expanded_chunks(cases):
    from drbx.stencils.loader import old_point_layout_bytes
    case = cases["ring"]
    assert old_point_layout_bytes([case["dense"]]) > 0
    with pytest.raises(ValueError, match="expanded"):
        old_point_layout_bytes([case["item"]])


def test_files_and_verified_artifact_directory(cases, tmp_path):
    case = cases["identity"]
    item, dense, grid = case["item"], case["dense"], case["grid"]
    identity = art.build_identity(component_hashes={"geometry": "d2"}, source_hashes={}, policy={"d2": 1})
    _, _, factors, chunk = _sources(case["context"], IDENTITY_PLAN)
    root = art.save_row_artifact(tmp_path, 8, identity=identity, faces=[chunk], point_factors={"faces": [factors]})
    plan_mem = lower_point_chunks([item], grid=grid)
    plan_dir = lower_point_chunks(iter_factored_point_chunks(tmp_path, 8, identity), grid=grid)
    plan_files = lower_point_files([root / "rows" / "faces_0.npz"], grid=grid)
    plan_pair = lower_point_files([("faces", root / "rows" / "faces_0.npz")], grid=grid)
    for plan in (plan_dir, plan_files, plan_pair):
        _assert_same_layout(plan, plan_mem)
        assert plan.payload.tensor_batches
        for a, b in zip(jax.tree_util.tree_leaves(plan.payload), jax.tree_util.tree_leaves(plan_mem.payload)):
            assert np.array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError, match="identity mismatch"):
        list(iter_factored_point_chunks(tmp_path, 8, {**identity, "policy": {"d2": 2}}))
    with pytest.raises(ValueError, match="cannot tell"):
        lower_point_files([tmp_path / "other.npz"], grid=grid)


def test_invalid_tensor_input_is_rejected(cases):
    case = cases["identity"]
    item, grid = case["item"], case["grid"]
    tr = item.tensor_rows
    big = LoaderGrid.from_arrays(n=16, raw_to_owner=np.arange(16 ** 3), eta_centers=np.linspace(0, 1, 16))
    with pytest.raises(ValueError, match="tensor rows are for n=8"):
        lower_point_chunks([item], grid=big)
    import dataclasses
    bad_owner = dataclasses.replace(tr, raw_owner=(tr.raw_owner + 1) % grid.n_owners)
    with pytest.raises(ValueError, match="raw_to_owner entries disagree"):
        lower_point_chunks([FactoredChunk(item.chunk, bad_owner, item.is_tensor)], grid=grid)
    bad_family = dataclasses.replace(tr, family=(tr.family + 1) % 3)
    with pytest.raises(ValueError, match="family tag disagree"):
        lower_point_chunks([FactoredChunk(item.chunk, bad_family, item.is_tensor)], grid=grid)
    bad_gradient = dataclasses.replace(tr, has_gradient=~tr.has_gradient)
    with pytest.raises(ValueError, match="has_gradient"):
        lower_point_chunks([FactoredChunk(item.chunk, bad_gradient, item.is_tensor)], grid=grid)
    wrong = item.is_tensor.copy()
    wrong[np.flatnonzero(~wrong)[0]] = True                    # a CSR source flagged as tensor
    with pytest.raises(ValueError):
        lower_point_chunks([FactoredChunk(item.chunk, tr, wrong)], grid=grid)
    bad_theta = dataclasses.replace(tr, theta_index=tr.theta_index + 100)
    with pytest.raises(ValueError, match="theta plane id"):
        lower_point_chunks([FactoredChunk(item.chunk, bad_theta, item.is_tensor)], grid=grid)


def test_zz_report_maxima():
    """Printed with ``-s``; the assertions above already bound every entry by TOL."""
    for key in sorted(MAXIMA):
        print(f"[D2 synthetic] {key}: {MAXIMA[key]:.3e}")
    assert all(v <= TOL for v in MAXIMA.values())
