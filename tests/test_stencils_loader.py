"""Vectorized chunk loader vs the per-row reference lowerings (P08 step 2a, task B).

Fast tests use synthetic rows (aggregated owners that span eta planes, mixed
donor widths, several families, conditioned/value-only/gradient rows). Each row
kind is packed with ``drbx.stencils.artifact`` and lowered twice: by the loader
under test and by the per-row reference (``lower_point_rows`` etc.); the
applied results are matched per target key. The slow test replays the real
local N32 dev chunks against a direct NumPy CSR evaluation.
"""
from __future__ import annotations

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry.fci_perpendicular_integrated_rows import IntegratedFaceRow
from drbx.geometry.fci_perpendicular_neumann_trace import NeumannPointRows
from drbx.geometry.fci_perpendicular_reconstruction import PointRowContext, PointRows
from drbx.native.fci_perpendicular_integrated_rows import (
    apply_integrated_face_rows, lower_integrated_face_rows, scatter_integrated_face_flux)
from drbx.native.fci_perpendicular_neumann_rows import (
    apply_neumann_point_rows, lower_neumann_point_rows)
from drbx.native.fci_perpendicular_point_rows import (
    BoundaryArrays, apply_point_rows, lower_point_rows)
from drbx.native.fci_perpendicular_source_rows import (
    apply_source_rows, payload_nbytes, source_gradients, source_values)
from drbx.stencils import loader
from drbx.stencils.artifact import (
    BC_VARIANTS, REQUEST_KINDS, PointRowChunk, pack_integrated_rows, pack_neumann_rows,
    pack_point_rows)
from drbx.stencils.loader import (
    LoaderGrid, RowSelection, lower_integrated_chunks, lower_neumann_chunks,
    lower_point_chunks, old_point_layout_bytes)

N = 8
TOL = 1e-13
FAMILIES = ("singleton", "ringwise", "coupled_quartic", "boundary_transverse", "quartic_wall")
MAXIMA: dict[str, float] = {}


# --------------------------------------------------------------------------
# Synthetic geometry and rows
# --------------------------------------------------------------------------

def _context(n=N):
    """Owners aggregate three consecutive raw ids, so many span (and wrap) eta planes."""
    faces = (np.linspace(0, 1, n + 1), np.linspace(0, 2 * np.pi, n + 1),
             np.linspace(0, 2 * np.pi, n + 1))
    centers = tuple((axis[:-1] + axis[1:]) / 2 for axis in faces)
    raw_to_owner = np.arange(n ** 3) // 3
    raw_volume = np.ones(n ** 3)
    owner_volume = np.bincount(raw_to_owner, weights=raw_volume)
    return PointRowContext.from_arrays(
        faces=faces, centers=centers, raw_to_owner=raw_to_owner, raw_volume=raw_volume,
        owner_volume=owner_volume, owner_centroid_xy=np.zeros((len(owner_volume), 2)),
        eta_period=2 * np.pi, dr=1 / n, dtheta=2 * np.pi / n, deta=2 * np.pi / n)


def _random_points(rng, count):
    return np.column_stack((rng.uniform(0.05, 1, count), rng.uniform(0, 2 * np.pi, count),
                            rng.uniform(0, 2 * np.pi, count)))


def _point_specs(rng, context, count=60):
    pool = _random_points(rng, 40)
    n_owners = len(context.vol)
    specs = []
    for i in range(count):
        family = FAMILIES[int(rng.integers(len(FAMILIES)))]
        q = int(rng.choice((1, 4, 9)))
        d = int(rng.choice((0, 3, 16, 17, 33, 40)))
        conditioned = d > 0 and rng.random() < 0.45
        has_gradient = bool(rng.random() < 0.6)
        targets = _random_points(rng, q)
        if conditioned:
            pick = rng.random(q) < 0.5
            targets[pick] = pool[rng.integers(len(pool), size=int(pick.sum()))]
        donors = rng.choice(n_owners, size=d, replace=False).astype(np.int64)
        row = PointRows(
            donors, rng.normal(size=(q, d)), rng.normal(size=(q, 3, d)) * 3.0, conditioned,
            pool[rng.integers(len(pool), size=d)] if conditioned else np.empty((0, 3)),
            targets, {"family": family})
        specs.append((row, has_gradient))
    return specs


def _pack_points(specs, request, variants, entity0=0):
    rows = [row for row, _ in specs]
    return pack_point_rows(
        rows, request=request, entity_id=[entity0 + i for i in range(len(rows))],
        bc_variant=variants, radial_degree=[int(i % 5) for i in range(len(rows))],
        store_gradient=[gradient for _, gradient in specs])


def _boundary(points, nf):
    """Analytic prescribed g and tangential derivatives at the plan's own query points."""
    if not len(points):
        return None
    r, th, eta = points.T
    scale = 1 + 0.5 * np.arange(nf)
    values = np.sin(2 * r + th)[:, None] * np.cos(eta)[:, None] * scale
    tangent = np.stack((np.cos(th + eta)[:, None] * scale, np.sin(2 * eta - r)[:, None] * scale),
                       axis=1)
    return BoundaryArrays(values, tangent)


def _fields(rng, n_owners, nf=3):
    return rng.normal(size=(n_owners, nf))


def _key(request, entity, node, variant):
    return (int(request), int(entity), int(node), int(variant))


def _relative(diff, scale):
    return float(np.max(np.abs(diff) / scale)) if diff.size else 0.0


# --------------------------------------------------------------------------
# (1) point rows: loader + apply_source_rows vs lower_point_rows + apply_point_rows
# --------------------------------------------------------------------------

def _two_chunk_fixture(seed=0):
    rng = np.random.default_rng(seed)
    context = _context()
    specs1 = _point_specs(rng, context, 40)
    specs2 = _point_specs(rng, context, 30)
    variants1 = [BC_VARIANTS[i % 3] for i in range(len(specs1))]
    variants2 = [BC_VARIANTS[(i + 1) % 3] for i in range(len(specs2))]
    chunk1 = _pack_points(specs1, "R2", variants1)
    chunk2 = _pack_points(specs2, "R3", variants2, entity0=1000)
    return context, [chunk1, chunk2], (specs1, specs2), (variants1, variants2)


def _old_targets(specs_by_chunk, variants_by_chunk, requests):
    """Reference target list in packing order: keys, rows, keeps-gradient flags."""
    keys, rows, has_gradient = [], [], []
    for (specs, variants, request, entity0) in zip(specs_by_chunk, variants_by_chunk, requests,
                                                   (0, 1000)):
        for i, (row, gradient) in enumerate(specs):
            rows.append(row)
            for node in range(len(row.trace_target_points)):
                keys.append(_key(REQUEST_KINDS.index(request), entity0 + i, node,
                                 BC_VARIANTS.index(variants[i])))
                has_gradient.append(gradient)
    return keys, rows, np.array(has_gradient)


def _old_and_new_point_results(context, chunks, specs_by_chunk, variants_by_chunk, nf=3, seed=5):
    rng = np.random.default_rng(seed)
    fields = _fields(rng, len(context.vol), nf)
    grid = LoaderGrid.from_context(context)
    plan = lower_point_chunks(chunks, grid=grid)
    keys, rows, has_gradient = _old_targets(specs_by_chunk, variants_by_chunk, ("R2", "R3"))
    old = lower_point_rows(context, rows)
    old_boundary = _boundary(old.boundary_points, nf)
    old_value, old_gradient = apply_point_rows(old.payload, fields, old_boundary)
    new_boundary = _boundary(plan.boundary_points, nf)
    new_value, new_gradient = apply_source_rows(plan.payload, fields, new_boundary)
    return (plan, old, fields, keys, has_gradient, np.asarray(old_value), np.asarray(old_gradient),
            np.asarray(new_value), np.asarray(new_gradient), rows)


def _slot_lookup(plan):
    t = plan.targets
    return {_key(t.request[i], t.entity_id[i], t.quad_node[i], t.bc_variant[i]): i
            for i in range(len(t.entity_id))}


def test_point_loader_matches_reference_values_and_gradients():
    context, chunks, specs, variants = _two_chunk_fixture()
    (plan, old, fields, keys, has_gradient, old_v, old_g, new_v, new_g,
     rows) = _old_and_new_point_results(context, chunks, specs, variants)
    lookup = _slot_lookup(plan)
    assert len(lookup) == len(keys) == len(plan.targets.entity_id)
    order = np.array([lookup[k] for k in keys])
    assert new_v.shape == old_v.shape
    # Weight scale of each target: sum |w| times the largest applied magnitude.
    row_scale = np.array([np.abs(rows[s].value[j]).sum()
                          for s, row in enumerate(rows) for j in range(len(row.trace_target_points))])
    magnitude = max(np.abs(fields).max(), np.abs(old.boundary_points).max() if len(old.boundary_points) else 0,
                    1.0)
    err_v = _relative(new_v[order] - old_v, np.maximum(row_scale, 1.0)[:, None] * magnitude)
    grad_scale = np.array([np.abs(rows[s].gradient[j]).sum()
                           for s, row in enumerate(rows) for j in range(len(row.trace_target_points))])
    gslots = plan.targets.gradient_slot[order]
    assert np.array_equal(gslots >= 0, has_gradient)
    keep = np.flatnonzero(has_gradient)
    diff = new_g[gslots[keep]] - old_g[keep]
    err_g = _relative(diff, np.maximum(grad_scale[keep], 1.0)[:, None, None] * magnitude)
    MAXIMA["point values"], MAXIMA["point gradients"] = err_v, err_g
    assert err_v <= TOL and err_g <= TOL
    # Structure: bucket widths are multiples of 16, sources share slots, dtypes.
    assert plan.payload.n_targets == len(keys)
    assert plan.payload.n_gradient_targets == int(has_gradient.sum())
    for batch, summary in zip(plan.payload.batches, plan.bucket_summary):
        assert batch.donor_ids.dtype == np.int32 and batch.value.dtype == np.float64
        assert batch.value.shape[2] % 16 == 0
        assert (batch.gradient is None) == (not summary.has_gradient) == (batch.gradient_slots is None)
        assert (batch.donor_query is None) == (not summary.conditioned) == (batch.target_query is None)
        assert summary.payload_bytes == sum(a.nbytes for a in batch if a is not None)
        if batch.donor_query is not None:
            assert batch.donor_query.dtype == np.int32 and batch.target_query.dtype == np.int32
    assert {s.conditioned for s in plan.bucket_summary} == {True, False}
    assert {s.has_gradient for s in plan.bucket_summary} == {True, False}
    assert {s.width for s in plan.bucket_summary} >= {0, 16, 32, 48}
    assert set(plan.family_names) == set(FAMILIES)
    assert payload_nbytes(plan.payload) == sum(s.payload_bytes for s in plan.bucket_summary)


def test_boundary_queries_are_deduplicated_by_bit_pattern():
    context, chunks, specs, variants = _two_chunk_fixture(seed=3)
    plan = lower_point_chunks(chunks, grid=LoaderGrid.from_context(context))
    points = plan.boundary_points
    assert points.dtype == np.float64 and points.shape[1] == 3
    assert len(np.unique(points.view(np.dtype((np.void, 24))))) == len(points)
    expected = set()
    for chunk in chunks:
        conditioned_targets = chunk.conditioned
        expected |= {p.tobytes() for p in chunk.target_point[conditioned_targets]}
        for t in range(len(chunk.entity_id)):
            if chunk.conditioned[t]:
                sl = slice(chunk.donor_ptr[t], chunk.donor_ptr[t + 1])
                expected |= {p.tobytes() for p in chunk.query_table[chunk.donor_query[sl]]}
    assert {p.tobytes() for p in points} == expected
    # -0.0 and 0.0 are different bit patterns and stay distinct.
    unique, inverse = loader._unique_rows(np.array([[0.0, 1, 2], [-0.0, 1, 2], [0.0, 1, 2]]))
    assert len(unique) == 2 and list(inverse) == [0, 1, 0]


def test_point_loader_rejects_source_invariant_violations():
    context, chunks, _, _ = _two_chunk_fixture(seed=1)
    grid = LoaderGrid.from_context(context)
    chunk = chunks[0]
    src = int(np.flatnonzero(np.diff(chunk.source_ptr) > 1)[0])
    # a source whose second target has a different donor id
    t1 = int(chunk.source_ptr[src]) + 1
    if chunk.donor_ptr[t1 + 1] > chunk.donor_ptr[t1]:
        donor = chunk.donor.copy()
        donor[chunk.donor_ptr[t1]] = (donor[chunk.donor_ptr[t1]] + 1) % len(context.vol)
        bad = PointRowChunk(**{**chunk.__dict__, "donor": donor})
        with pytest.raises(ValueError, match="donor ids differ"):
            lower_point_chunks([bad], grid=grid)
    # a source whose targets disagree on a tag
    family = chunk.family.copy().astype("<U32")
    family[t1] = "different"
    bad = PointRowChunk(**{**chunk.__dict__, "family": family})
    with pytest.raises(ValueError, match="disagree on family"):
        lower_point_chunks([bad], grid=grid)
    conditioned_src = [s for s in range(len(chunk.source_ptr) - 1)
                       if chunk.conditioned[chunk.source_ptr[s]] and chunk.source_ptr[s + 1] - chunk.source_ptr[s] > 1][0]
    t1 = int(chunk.source_ptr[conditioned_src]) + 1
    query = chunk.donor_query.copy()
    query[chunk.donor_ptr[t1]] = (query[chunk.donor_ptr[t1]] + 1) % len(chunk.query_table)
    bad = PointRowChunk(**{**chunk.__dict__, "donor_query": query})
    with pytest.raises(ValueError, match="donor_query differs"):
        lower_point_chunks([bad], grid=grid)
    with pytest.raises(ValueError, match="invalid compact donor"):
        lower_point_chunks([chunk], grid=LoaderGrid.from_arrays(
            n=N, raw_to_owner=np.arange(N ** 3) // 8, eta_centers=context.centers[2]))


# --------------------------------------------------------------------------
# (3) eta offsets
# --------------------------------------------------------------------------

def test_point_eta_offset_min_max_match_reference_tuples():
    context, chunks, specs, variants = _two_chunk_fixture(seed=2)
    (plan, old, *_rest) = _old_and_new_point_results(context, chunks, specs, variants)
    # Old targets are in packing order, which is also the loader's order here.
    sources = plan.targets.source
    lo = np.zeros(len(plan.source_eta_offset_min), dtype=np.int64)
    hi = np.zeros_like(lo)
    seen = np.zeros(len(lo), dtype=bool)
    for t, offsets in enumerate(old.donor_eta_offset_sets):
        values = [k for donor in offsets for k in donor]
        s = int(sources[t])
        if not values:
            continue
        lo[s] = min(lo[s], min(values)) if seen[s] else min(values)
        hi[s] = max(hi[s], max(values)) if seen[s] else max(values)
        seen[s] = True
    np.testing.assert_array_equal(plan.source_eta_offset_min, lo)
    np.testing.assert_array_equal(plan.source_eta_offset_max, hi)
    assert plan.source_eta_offset_min.dtype == np.int32
    # The wrap is exercised: some offsets are negative and some positive.
    assert plan.source_eta_offset_min.min() < 0 < plan.source_eta_offset_max.max()
    assert np.array_equal(plan.source_target_ptr[1:] - plan.source_target_ptr[:-1],
                          np.bincount(sources))


def test_loader_grid_planes_match_owner_members():
    context = _context()
    grid = LoaderGrid.from_context(context)
    for owner in (0, 1, 2, 5, 60, len(context.vol) - 1):
        planes = np.unique(context.members(owner) % context.n)
        np.testing.assert_array_equal(grid.plane_index[grid.plane_ptr[owner]:grid.plane_ptr[owner + 1]], planes)


# --------------------------------------------------------------------------
# (4) eager vs jit, (5) jvp
# --------------------------------------------------------------------------

def test_apply_source_rows_eager_matches_jit_bitwise_and_jvp_is_linear():
    context, chunks, _, _ = _two_chunk_fixture(seed=4)
    plan = lower_point_chunks(chunks, grid=LoaderGrid.from_context(context))
    rng = np.random.default_rng(9)
    fields = _fields(rng, len(context.vol), 4)
    boundary = _boundary(plan.boundary_points, 4)
    eager = apply_source_rows(plan.payload, fields, boundary)
    jitted = jax.jit(apply_source_rows, static_argnames=("values", "gradients"))(
        plan.payload, fields, boundary)
    closed = jax.jit(lambda f, b: apply_source_rows(plan.payload, f, b))(fields, boundary)
    for a, b, c in zip(eager, jitted, closed):
        assert np.array_equal(np.asarray(a), np.asarray(b))
        assert np.array_equal(np.asarray(a), np.asarray(c))
    assert eager[0].shape == (plan.payload.n_targets, 4)
    assert eager[1].shape == (plan.payload.n_gradient_targets, 3, 4)
    tangent = _fields(rng, len(context.vol), 4)
    zero = BoundaryArrays(np.zeros_like(boundary.values), np.zeros_like(boundary.tangential_gradients))

    def action(f):
        return apply_source_rows(plan.payload, f, boundary)

    _, jvp = jax.jvp(action, (jnp.asarray(fields),), (jnp.asarray(tangent),))
    linear = apply_source_rows(plan.payload, tangent, zero)
    for a, b in zip(jvp, linear):
        scale = float(np.abs(np.asarray(b)).max())
        MAXIMA["jvp"] = max(MAXIMA.get("jvp", 0.0), float(np.abs(np.asarray(a) - np.asarray(b)).max()) / scale)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-13 * scale)
    # values-only / gradients-only entry points agree with the joint call
    # (a separately compiled program may fuse differently, so only equal to rounding)
    for part, joint in ((source_values(plan.payload, fields, boundary), eager[0]),
                        (source_gradients(plan.payload, fields, boundary), eager[1])):
        np.testing.assert_allclose(np.asarray(part), np.asarray(joint), rtol=0,
                                   atol=1e-13 * float(np.abs(np.asarray(joint)).max()))
    with pytest.raises(ValueError, match="boundary arrays"):
        apply_source_rows(plan.payload, fields)
    with pytest.raises(ValueError, match="incompatible shapes"):
        apply_source_rows(plan.payload, fields, BoundaryArrays(boundary.values[:-1], boundary.tangential_gradients))


# --------------------------------------------------------------------------
# (6) selection
# --------------------------------------------------------------------------

def test_selection_by_request_variant_and_entity():
    context, chunks, specs, variants = _two_chunk_fixture(seed=6)
    grid = LoaderGrid.from_context(context)
    full = lower_point_chunks(chunks, grid=grid)
    rng = np.random.default_rng(11)
    fields = _fields(rng, len(context.vol))
    full_v, full_g = apply_source_rows(full.payload, fields, _boundary(full.boundary_points, 3))
    full_lookup = _slot_lookup(full)
    r3, d_only = REQUEST_KINDS.index("R3"), BC_VARIANTS.index("D")

    def check(selection, predicate):
        plan = lower_point_chunks(chunks, grid=grid, select=selection)
        t = plan.targets
        expect = [i for i in range(len(full.targets.entity_id)) if predicate(full.targets, i)]
        assert len(t.entity_id) == len(expect) > 0
        assert np.array_equal(t.value_slot, np.arange(len(expect)))
        assert plan.payload.n_targets == len(expect)
        v, g = apply_source_rows(plan.payload, fields, _boundary(plan.boundary_points, 3))
        for i in range(len(t.entity_id)):
            j = full_lookup[_key(t.request[i], t.entity_id[i], t.quad_node[i], t.bc_variant[i])]
            np.testing.assert_allclose(np.asarray(v)[i], np.asarray(full_v)[j], rtol=0, atol=1e-13)
            assert (t.gradient_slot[i] >= 0) == (full.targets.gradient_slot[j] >= 0)
            if t.gradient_slot[i] >= 0:
                np.testing.assert_allclose(np.asarray(g)[t.gradient_slot[i]],
                                           np.asarray(full_g)[full.targets.gradient_slot[j]],
                                           rtol=0, atol=1e-12)
        assert len(plan.boundary_points) <= len(full.boundary_points)
        assert len(plan.source_eta_offset_min) == len(np.unique(t.source))
        return plan

    check(RowSelection(requests=("R3",)), lambda t, i: t.request[i] == r3)
    check(RowSelection(bc_variants=("D",)), lambda t, i: t.bc_variant[i] == d_only)
    check(RowSelection(requests=("R2",), bc_variants=("D", "N")),
          lambda t, i: t.request[i] == REQUEST_KINDS.index("R2") and t.bc_variant[i] != 0)
    entities = np.array([3, 7, 8, 1005, 1010])
    check(RowSelection(entity_ids=entities), lambda t, i: t.entity_id[i] in entities)
    empty = lower_point_chunks(chunks, grid=grid, select=RowSelection(entity_ids=np.array([-5])))
    assert empty.payload.batches == () and empty.payload.n_targets == 0 and len(empty.boundary_points) == 0
    with pytest.raises(ValueError, match="unsupported request"):
        RowSelection(requests=("R9",))


def test_streaming_generator_and_chunk_order_only_permute_slots():
    context, chunks, _, _ = _two_chunk_fixture(seed=7)
    grid = LoaderGrid.from_context(context)
    a = lower_point_chunks(chunks, grid=grid)
    b = lower_point_chunks((c for c in chunks), grid=grid)
    assert [s.payload_bytes for s in a.bucket_summary] == [s.payload_bytes for s in b.bucket_summary]
    np.testing.assert_array_equal(a.targets.entity_id, b.targets.entity_id)
    np.testing.assert_array_equal(a.boundary_points, b.boundary_points)


# --------------------------------------------------------------------------
# (2) integrated rows
# --------------------------------------------------------------------------

def _integrated_rows(rng, context, count=48):
    pool = _random_points(rng, 30)
    n_owners = len(context.vol)
    rows = []
    for i in range(count):
        d = int(rng.choice((0, 3, 16, 25, 40)))
        conditioned = d > 0 and rng.random() < 0.5
        weights = rng.normal(size=d)
        target = _random_points(rng, 9)
        if conditioned:
            target[::2] = pool[rng.integers(len(pool), size=len(target[::2]))]
        rows.append(IntegratedFaceRow(
            rng.choice(n_owners, size=d, replace=False).astype(np.int64), weights, conditioned,
            pool[rng.integers(len(pool), size=d)] if conditioned else np.empty((0, 3)),
            target, -weights if conditioned else np.empty(0),
            rng.normal(size=(9, 2)) if conditioned else np.empty((0, 2)),
            int(rng.choice((0, 1, 2, 4, 7)))))
    return rows


def test_integrated_loader_matches_reference_flux_and_scatter():
    context = _context()
    grid = LoaderGrid.from_context(context)
    rng = np.random.default_rng(21)
    rows = _integrated_rows(rng, context)
    endpoints = rng.integers(-1, len(context.vol), size=(len(rows), 2))
    chunks = [pack_integrated_rows(rows[:30], entity_id=np.arange(30)),
              pack_integrated_rows(rows[30:], entity_id=np.arange(30, len(rows)))]
    fields = _fields(rng, len(context.vol), 3)
    old = lower_integrated_face_rows(context, tuple(rows), endpoints)
    plan = lower_integrated_chunks(chunks, grid=grid, endpoints=endpoints, owner_volume=context.vol)
    old_flux = np.asarray(apply_integrated_face_rows(old.payload, fields, _boundary(old.boundary_points, 3)))
    new_flux = np.asarray(apply_integrated_face_rows(plan.payload, fields, _boundary(plan.boundary_points, 3)))
    scale = max(np.abs(fields).max(), 1.0) * np.array([max(np.abs(r.weights).sum(), 1.0) for r in rows])[:, None]
    err = _relative(new_flux - old_flux, scale)
    old_scatter = np.asarray(scatter_integrated_face_flux(old.payload, old_flux))
    new_scatter = np.asarray(scatter_integrated_face_flux(plan.payload, new_flux))
    err_scatter = _relative(new_scatter - old_scatter, max(np.abs(old_scatter).max(), 1.0))
    MAXIMA["integrated flux"], MAXIMA["integrated scatter"] = err, err_scatter
    assert err <= TOL and err_scatter <= TOL
    np.testing.assert_array_equal(plan.face_entity_id, np.arange(len(rows)))
    assert plan.payload.face_count == len(rows)
    assert plan.payload.batches[0].donor_ids.dtype == np.int32
    assert all(b.weights.shape[1] % 16 == 0 for b in plan.payload.batches)
    assert {s.conditioned for s in plan.bucket_summary} == {True, False}
    # eta offsets: min/max of the reference tuples (q3 node 4)
    for f, offsets in enumerate(old.donor_eta_offsets):
        values = [k for donor in offsets for k in donor]
        assert plan.eta_offset_min[f] == (min(values) if values else 0)
        assert plan.eta_offset_max[f] == (max(values) if values else 0)
    assert plan.eta_offset_min.min() < 0 < plan.eta_offset_max.max()
    # the qualified integrated kernel (not part of this task) is not bitwise eager/jit
    # stable, so the jitted flux is only compared to rounding
    boundary = _boundary(plan.boundary_points, 3)
    jitted = jax.jit(lambda f, b: apply_integrated_face_rows(plan.payload, f, b))(fields, boundary)
    np.testing.assert_allclose(np.asarray(jitted), new_flux, rtol=0, atol=1e-13 * np.abs(new_flux).max())
    # entity subset (also reorders nothing: output faces follow chunk order)
    subset = np.array([2, 9, 30, 41])
    sub = lower_integrated_chunks(chunks, grid=grid, endpoints=endpoints, owner_volume=context.vol,
                                  select=RowSelection(entity_ids=subset))
    np.testing.assert_array_equal(sub.face_entity_id, subset)
    sub_flux = np.asarray(apply_integrated_face_rows(sub.payload, fields, _boundary(sub.boundary_points, 3)))
    np.testing.assert_allclose(sub_flux, new_flux[subset], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(sub.payload.lower_owner, endpoints[subset, 0])
    assert lower_integrated_chunks(chunks, grid=grid, endpoints=endpoints, owner_volume=context.vol,
                                   select=RowSelection(requests=("R1",))).payload.face_count == 0
    # loading must equal -weights for conditioned rows
    conditioned_face = next(i for i, r in enumerate(rows) if r.boundary_conditioned)
    broken = pack_integrated_rows([rows[conditioned_face]], entity_id=[conditioned_face])
    broken.value_loading[0] += 1e-3
    with pytest.raises(ValueError, match="value loading"):
        lower_integrated_chunks([broken], grid=grid, endpoints=endpoints, owner_volume=context.vol)


# --------------------------------------------------------------------------
# (2) Neumann rows
# --------------------------------------------------------------------------

def _neumann_rows(rng, context, count=20):
    pool = _random_points(rng, 60)
    rows = []
    for i in range(count):
        d = int(rng.choice((5, 17, 40)))
        rows.append(NeumannPointRows(
            rng.choice(len(context.vol), size=d, replace=False).astype(np.int64),
            rng.normal(size=d), rng.normal(size=(3, d)), pool[rng.choice(60, size=28, replace=False)],
            rng.normal(size=28).astype(np.longdouble), rng.normal(size=(3, 28)).astype(np.longdouble),
            1.0 + i, 1e-12))
    return rows


def test_neumann_loader_matches_reference():
    context = _context()
    rng = np.random.default_rng(31)
    rows = _neumann_rows(rng, context)
    chunks = [pack_neumann_rows(rows[:8], entity_id=np.arange(8), request="R2", radial_degree=4),
              pack_neumann_rows(rows[8:], entity_id=np.arange(8, 20), request="R3", radial_degree=3)]
    old_payload, old_points = lower_neumann_point_rows(rows)
    plan = lower_neumann_chunks(chunks)
    assert plan.payload.donor_ids.shape == old_payload.donor_ids.shape
    assert plan.payload.boundary_value_weights.dtype == np.float64
    assert plan.payload.boundary_gradient_weights.dtype == np.float64
    assert plan.payload.donor_ids.dtype == np.int32 and plan.payload.boundary_ids.dtype == np.int32
    fields = _fields(rng, len(context.vol), 3)

    def g_at(points):
        return np.sin(3 * points[:, 0] + points[:, 1] - points[:, 2])[:, None] * (1 + np.arange(3))

    old_v, old_g = apply_neumann_point_rows(old_payload, fields, g_at(old_points))
    new_v, new_g = apply_neumann_point_rows(plan.payload, fields, g_at(plan.wall_points))
    scale = max(np.abs(fields).max(), 1.0) * np.array([max(np.abs(r.value).sum(), np.abs(r.gradient).sum(), 1.0)
                                                       for r in rows])
    err_v = _relative(np.asarray(new_v) - np.asarray(old_v), scale[:, None])
    err_g = _relative(np.asarray(new_g) - np.asarray(old_g), scale[:, None, None])
    MAXIMA["neumann value"], MAXIMA["neumann gradient"] = err_v, err_g
    assert err_v <= TOL and err_g <= TOL
    assert plan.payload.boundary_query_count == len(plan.wall_points) == old_payload.boundary_query_count
    assert len(np.unique(plan.wall_points.view(np.dtype((np.void, 24))))) == len(plan.wall_points)
    np.testing.assert_array_equal(plan.request, [REQUEST_KINDS.index("R2")] * 8 + [REQUEST_KINDS.index("R3")] * 12)
    np.testing.assert_array_equal(plan.radial_degree, [4] * 8 + [3] * 12)
    np.testing.assert_array_equal(plan.entity_id, np.arange(20))
    # selection by originating request / entity subset
    sub = lower_neumann_chunks(chunks, select=RowSelection(requests=("R3",)))
    np.testing.assert_array_equal(sub.entity_id, np.arange(8, 20))
    sub_v, _ = apply_neumann_point_rows(sub.payload, fields, g_at(sub.wall_points))
    np.testing.assert_allclose(np.asarray(sub_v), np.asarray(new_v)[8:], rtol=0, atol=1e-13)
    sub = lower_neumann_chunks(chunks, select=RowSelection(entity_ids=np.array([1, 9, 15])))
    assert list(sub.entity_id) == [1, 9, 15]
    assert lower_neumann_chunks(chunks, select=RowSelection(requests=("R1",))).payload.donor_ids.shape[0] == 0


def test_neumann_longdouble_weights_are_cast_to_float64_explicitly():
    context = _context()
    rng = np.random.default_rng(32)
    rows = _neumann_rows(rng, context, 4)
    chunk = pack_neumann_rows(rows, entity_id=np.arange(4), request="R1", radial_degree=4)
    plan = lower_neumann_chunks([chunk])
    np.testing.assert_array_equal(plan.payload.boundary_value_weights,
                                  np.asarray(chunk.boundary_value).astype(np.float64))
    np.testing.assert_array_equal(plan.payload.boundary_gradient_weights,
                                  np.asarray(chunk.boundary_gradient).astype(np.float64))


def test_old_layout_bytes_formula_matches_reference_arrays():
    context, chunks, specs, variants = _two_chunk_fixture(seed=8)
    keys, rows, _ = _old_targets(specs, variants, ("R2", "R3"))
    old = lower_point_rows(context, rows)
    reference = sum(sum(np.asarray(a).nbytes for a in batch) for batch in old.payload.batches)
    reference += old.payload.output_template.nbytes
    assert old_point_layout_bytes(chunks) == reference


def test_zz_report_equivalence_maxima():
    """Print the worst relative differences seen by the equivalence tests (run with -s)."""
    for name, value in sorted(MAXIMA.items()):
        print(f"max relative difference, {name}: {value:.3e}")
    assert all(value <= TOL for value in MAXIMA.values())


# --------------------------------------------------------------------------
# Slow: real local N32 dev chunks vs a direct NumPy CSR evaluation
# --------------------------------------------------------------------------

DEV_N32 = Path("/Users/yxie/Desktop/HSX drbx/work/p08_step1_campaign_dev_20260929T002907Z/artifact/N32")


def _csr_reference(chunk, fields, nf):
    """Per-target CSR sums straight from the chunk arrays (Dirichlet lift with the analytic g)."""
    n_t = len(chunk.entity_id)
    values = np.zeros((n_t, nf))
    gradients = np.zeros((n_t, 3, nf))
    widths = np.diff(chunk.donor_ptr)
    for a, b in loader._blocks(widths, 1 << 20):
        p0, p1 = int(chunk.donor_ptr[a]), int(chunk.donor_ptr[b])
        cond = np.repeat(chunk.conditioned[a:b], widths[a:b])
        lift = np.zeros((p1 - p0, nf))
        if cond.any():
            lift[cond] = _boundary(chunk.query_table[chunk.donor_query[p0:p1][cond]], nf).values
        data = fields[chunk.donor[p0:p1]] - lift
        rows = np.flatnonzero(widths[a:b] > 0)
        starts = (chunk.donor_ptr[a:b] - p0)[rows]
        if len(rows):
            values[a + rows] = np.add.reduceat(chunk.value[p0:p1, None] * data, starts, axis=0)
        target_cond = np.flatnonzero(chunk.conditioned[a:b])
        if len(target_cond):
            values[a + target_cond] += _boundary(chunk.target_point[a + target_cond], nf).values
        # gradients (targets that store one); donors are the same entries as the values
        g_rows = np.flatnonzero(chunk.has_gradient[a:b] & (widths[a:b] > 0))
        if len(g_rows):
            for r in g_rows:     # per-target reference loop; test code only
                t = a + int(r)
                s0, s1 = int(chunk.donor_ptr[t]) - p0, int(chunk.donor_ptr[t + 1]) - p0
                gs = int(chunk.gradient_ptr[t])
                gradients[t] = chunk.gradient[:, gs:gs + (s1 - s0)] @ data[s0:s1]
        for t in target_cond:
            if chunk.has_gradient[a + t]:
                gradients[a + t, 1:] += _boundary(chunk.target_point[a + t][None], nf).tangential_gradients[0]
    return values, gradients


@pytest.mark.slow
@pytest.mark.skipif(not DEV_N32.exists(), reason="local N32 dev chunks not available")
def test_real_dev_chunks_against_direct_csr(capsys):
    import time
    import tracemalloc
    from drbx.stencils.artifact import decode_chunk

    census = np.load(DEV_N32 / "census.npz")
    n = int(census["n"])
    raw_to_owner = -np.ones(n ** 3, dtype=np.int64)
    for raw, owner in ((census["raw_lo"], census["owner_lo"]), (census["raw_hi"], census["owner_hi"])):
        found = raw >= 0
        raw_to_owner[raw[found]] = owner[found]
    assert np.all(raw_to_owner >= 0)
    eta = np.load(DEV_N32 / "geometry.npz")["raw_points"][:n, 2]
    grid = LoaderGrid.from_arrays(n=n, raw_to_owner=raw_to_owner, eta_centers=eta)
    rng = np.random.default_rng(1)
    nf = 2
    fields = rng.normal(size=(grid.n_owners, nf))
    report = {}
    for group, name in (("cells", "cells_cells_00000"), ("faces", "faces_faces_00000")):
        chunk = decode_chunk(group, (DEV_N32 / "rows" / f"{name}.npz").read_bytes())
        tracemalloc.start()
        start = time.perf_counter()
        plan = lower_point_chunks([chunk], grid=grid)
        seconds = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert len(plan.targets.entity_id) == len(chunk.entity_id)
        np.testing.assert_array_equal(plan.targets.entity_id, chunk.entity_id)
        np.testing.assert_array_equal(plan.targets.quad_node, chunk.quad_node)
        np.testing.assert_array_equal(plan.targets.target_points, chunk.target_point)
        assert plan.payload.n_gradient_targets == int(chunk.has_gradient.sum())
        boundary = _boundary(plan.boundary_points, nf)
        values, gradients = apply_source_rows(plan.payload, fields, boundary)
        ref_v, ref_g = _csr_reference(chunk, fields, nf)
        widths = np.diff(chunk.donor_ptr)
        abs_sum = np.zeros(len(widths))
        nz = np.flatnonzero(widths)
        abs_sum[nz] = np.add.reduceat(np.abs(chunk.value), chunk.donor_ptr[:-1][nz])
        scale = np.maximum(abs_sum, 1.0) * np.abs(fields).max()
        err_v = _relative(np.asarray(values) - ref_v, scale[:, None])
        keep = np.flatnonzero(plan.targets.gradient_slot >= 0)
        g_abs = np.zeros(len(widths))
        gw = np.diff(chunk.gradient_ptr)
        gnz = np.flatnonzero(gw)
        g_abs[gnz] = np.add.reduceat(np.abs(chunk.gradient).sum(axis=0), chunk.gradient_ptr[:-1][gnz])
        err_g = _relative(np.asarray(gradients)[plan.targets.gradient_slot[keep]] - ref_g[keep],
                          np.maximum(g_abs[keep], 1.0)[:, None, None] * np.abs(fields).max())
        # sampled eta offsets vs the reference per-owner member-plane definition
        order = np.argsort(raw_to_owner, kind="stable")
        starts = np.r_[0, np.cumsum(np.bincount(raw_to_owner, minlength=grid.n_owners))]
        sources = np.linspace(0, len(plan.source_eta_offset_min) - 1, 150).astype(int)
        for s in sources:
            t0, t1 = plan.source_target_ptr[s], plan.source_target_ptr[s + 1]
            t = int(t0)
            donors = chunk.donor[chunk.donor_ptr[t]:chunk.donor_ptr[t + 1]]
            offsets = []
            for target in range(int(t0), int(t1)):
                plane = int(np.argmin(np.abs(eta - chunk.target_point[target, 2])))
                for oid in donors:
                    members = order[starts[oid]:starts[oid + 1]]
                    offsets += [int((int(k) - plane + n // 2) % n - n // 2) for k in np.unique(members % n)]
            lo, hi = (min(offsets), max(offsets)) if offsets else (0, 0)
            assert (plan.source_eta_offset_min[s], plan.source_eta_offset_max[s]) == (lo, hi)
        from drbx.native.fci_perpendicular_source_rows import payload_nbytes
        new_bytes = payload_nbytes(plan.payload)
        old_bytes = old_point_layout_bytes([chunk])
        report[group] = dict(targets=len(chunk.entity_id), sources=len(plan.source_eta_offset_min),
                             buckets=len(plan.payload.batches), conditioned=int(chunk.conditioned.sum()),
                             queries=len(plan.boundary_points), seconds=seconds, peak_mb=peak / 1e6,
                             chunk_mb=sum(getattr(chunk, f).nbytes for f in chunk.__dataclass_fields__
                                          if hasattr(getattr(chunk, f), "nbytes")) / 1e6,
                             new_mb=new_bytes / 1e6, old_mb=old_bytes / 1e6,
                             err_values=err_v, err_gradients=err_g,
                             eta=(int(plan.source_eta_offset_min.min()), int(plan.source_eta_offset_max.max())))
        assert err_v <= TOL and err_g <= TOL
        assert new_bytes < old_bytes
    # P07 integrated chunk: direct CSR (no conditioned faces locally, formula kept anyway)
    chunk = decode_chunk("p07", (DEV_N32 / "rows" / "p07_p07_00000.npz").read_bytes())
    endpoints = np.stack((census["owner_lo"], census["owner_hi"]), axis=1)
    p07_id = np.asarray(census["p07_id"])
    face_rows = np.full(int(p07_id.max()) + 1, -1, dtype=np.int64)
    face_rows[p07_id[p07_id >= 0]] = np.flatnonzero(p07_id >= 0)
    face_endpoints = np.where(face_rows[:, None] >= 0, endpoints[np.maximum(face_rows, 0)], -1)
    start = time.perf_counter()
    plan = lower_integrated_chunks([chunk], grid=grid, endpoints=face_endpoints,
                                   owner_volume=np.ones(grid.n_owners))
    p07_seconds = time.perf_counter() - start
    flux = np.asarray(apply_integrated_face_rows(plan.payload, fields, _boundary(plan.boundary_points, nf)))
    expected = np.zeros((len(chunk.entity_id), nf))
    for t in range(len(chunk.entity_id)):
        sl = slice(chunk.donor_ptr[t], chunk.donor_ptr[t + 1])
        expected[t] = chunk.weight[sl] @ fields[chunk.donor[sl]]
    err_p07 = _relative(flux - expected, max(np.abs(expected).max(), 1.0))
    assert err_p07 <= TOL and not chunk.conditioned.any()
    report["p07"] = dict(faces=len(chunk.entity_id), seconds=p07_seconds, err=err_p07,
                         eta=(int(plan.eta_offset_min.min()), int(plan.eta_offset_max.max())))
    with capsys.disabled():
        print("\nreal N32 dev chunks:")
        for name, item in report.items():
            print(" ", name, {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in item.items()})
