"""Real-geometry gate of the tensor runtime (P08 step 2, task D2; slow).

Takes the bounded real N32 build units of ``tests/test_stencils_tensor_rows_real.py`` (same fixture code,
covering every factored family and request next to CSR-only sources), encodes each once CSR-only and once
with tensor factors, lowers both (the tensor chunk through ``decode_chunk_factored``, never expanded),
applies both plans to random owner fields and compares every target (<= 1e-13 relative to the row weight
scale x field magnitude). Prints per unit: runtime payload bytes (tensor plan, all-CSR plan, B's old
layout, analytic), lowering time, and warm jitted apply time (median) on CPU.

``D2_UNIT_CACHE=<dir>`` caches the built units on disk between runs (development aid only).
"""
from __future__ import annotations

import pickle
import os
import statistics
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

from drbx.native.fci_perpendicular_source_rows import apply_source_rows, payload_nbytes
from drbx.stencils import artifact as art
from drbx.stencils.loader import (
    FactoredChunk, LoaderGrid, lower_point_chunks, old_point_layout_bytes)
from tests import test_stencils_tensor_rows_real as d1
from tests.test_stencils_tensor_loader import _boundary, _relative, _weight_scales

TOL = 1e-13
NF = 4
RUNS = 5


def _d1_units():
    fixture = d1.real_units
    build = getattr(fixture, "_get_wrapped_function", lambda: fixture.__wrapped__)()
    return build()


@pytest.fixture(scope="module")
def real_chunks():
    """``[(stage, index, csr_chunk, tensor_bytes)]`` and the ``LoaderGrid`` of the N32 geometry."""
    cache = os.environ.get("D2_UNIT_CACHE")
    path = Path(cache) / "d2_units.pkl" if cache else None
    if path is not None and path.is_file():
        with path.open("rb") as handle:
            return pickle.load(handle)
    built, tools = _d1_units()
    context = tools["context"]
    grid = LoaderGrid.from_context(context)
    units = []
    for (stage, index), rows in built.items():
        chunk = tools["build_artifact"]._pack_point_rows(rows)
        tensor = art.encode_chunk(stage, chunk, factors=[r.factors for r in rows])
        units.append((stage, index, chunk, tensor))
    result = (units, grid)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(result, handle)
    return result


def _timed(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def _apply_times(plan_a, plan_b, fields, boundary):
    """``(out_a, out_b, median_a, median_b)``: both plans jitted, warmed, then timed interleaved (a, b, a, b, ...)
    so a loaded machine affects both alike."""
    run_a, run_b = jax.jit(apply_source_rows), jax.jit(apply_source_rows)
    out_a = jax.block_until_ready(run_a(plan_a.payload, fields, boundary))     # compile + warm
    out_b = jax.block_until_ready(run_b(plan_b.payload, fields, boundary))
    times_a, times_b = [], []
    for _ in range(RUNS):
        start = time.perf_counter()
        jax.block_until_ready(run_a(plan_a.payload, fields, boundary))
        times_a.append(time.perf_counter() - start)
        start = time.perf_counter()
        jax.block_until_ready(run_b(plan_b.payload, fields, boundary))
        times_b.append(time.perf_counter() - start)
    return out_a, out_b, statistics.median(times_a), statistics.median(times_b)


@d1.needs_geometry
@pytest.mark.slow
def test_real_units_tensor_plan_matches_csr_plan(real_chunks, capsys):
    units, grid = real_chunks
    rng = np.random.default_rng(11)
    fields = rng.normal(size=(grid.n_owners, NF))
    table = []
    worst = {"value": 0.0, "gradient": 0.0}
    for stage, index, csr_chunk, tensor_bytes in units:
        item = FactoredChunk(*art.decode_chunk_factored(stage, tensor_bytes))
        assert item.is_tensor.any()
        plan_t, t_lower_tensor = _timed(lambda: lower_point_chunks([item], grid=grid))
        plan_c, t_lower_csr = _timed(lambda: lower_point_chunks([csr_chunk], grid=grid))
        # identical slots, targets and halo
        for a, b in zip(plan_t.targets, plan_c.targets):
            assert a.dtype == b.dtype and np.array_equal(a, b)
        for name in ("source_target_ptr", "source_eta_offset_min", "source_eta_offset_max"):
            assert np.array_equal(getattr(plan_t, name), getattr(plan_c, name)), name
        assert np.array_equal(plan_t.boundary_points, plan_c.boundary_points)
        boundary = _boundary(plan_c.boundary_points, NF)
        (v_t, g_t), (v_c, g_c), apply_tensor, apply_csr = _apply_times(plan_t, plan_c, fields, boundary)
        v_t, g_t, v_c, g_c = (np.asarray(x) for x in (v_t, g_t, v_c, g_c))
        scale_v, scale_g = _weight_scales(csr_chunk)
        magnitude = max(np.abs(fields).max(), np.abs(plan_c.boundary_points).max() if len(plan_c.boundary_points) else 0)
        err_v = _relative(v_t - v_c, np.maximum(scale_v, 1e-300)[:, None] * magnitude)
        rows = np.flatnonzero(plan_c.targets.gradient_slot >= 0)
        slots = plan_c.targets.gradient_slot[rows]
        err_g = _relative(g_t[slots] - g_c[slots], np.maximum(scale_g[rows], 1e-300)[:, :, None] * magnitude)
        worst["value"], worst["gradient"] = max(worst["value"], err_v), max(worst["gradient"], err_g)
        assert err_v <= TOL and err_g <= TOL, (stage, index, err_v, err_g)
        tensor_targets = int(np.repeat(item.is_tensor, np.diff(item.chunk.source_ptr)).sum())
        table.append((stage, index, len(item.is_tensor), int(item.is_tensor.sum()), tensor_targets,
                      len(csr_chunk.entity_id), payload_nbytes(plan_t.payload), payload_nbytes(plan_c.payload),
                      old_point_layout_bytes([csr_chunk]), t_lower_tensor, t_lower_csr, apply_tensor, apply_csr,
                      err_v, err_g, [(s.family, s.has_gradient, s.targets) for s in plan_t.bucket_summary
                                     if s.encoding == "tensor"]))
    with capsys.disabled():
        print(f"\n[D2 real N32 gate] {NF} fields, max rel. error value {worst['value']:.2e} gradient {worst['gradient']:.2e}, "
              f"load average {os.getloadavg()[0]:.1f}")
        print("  unit | sources (tensor) | targets (tensor) | payload MB: tensor / all-CSR / old layout | "
              "lowering s: tensor / CSR | apply s (jit, warm median of 5, interleaved): tensor / CSR | err v / g")
        for (stage, index, sources, tensor_sources, tensor_targets, targets, b_t, b_c, b_old, l_t, l_c, a_t, a_c,
             e_v, e_g, batches) in table:
            print(f"  {stage} {index}: {sources} ({tensor_sources}) | {targets} ({tensor_targets}) | "
                  f"{b_t / 1e6:.2f} / {b_c / 1e6:.2f} / {b_old / 1e6:.2f} | {l_t:.2f} / {l_c:.2f} | "
                  f"{a_t:.4f} / {a_c:.4f} | {e_v:.1e} / {e_g:.1e}")
            print(f"      tensor batches: {batches}")


@d1.needs_geometry
@pytest.mark.slow
def test_real_units_in_one_streamed_plan_share_tables(real_chunks, capsys):
    """All units through one ``lower_point_chunks`` call (a lazy generator, chunks decoded one at a time):
    same slots and values as the all-CSR plan, and the grid-global tables are smaller than the per-unit sum."""
    units, grid = real_chunks
    fields = np.random.default_rng(12).normal(size=(grid.n_owners, NF))

    def factored():
        for stage, _, _, tensor_bytes in units:
            yield FactoredChunk(*art.decode_chunk_factored(stage, tensor_bytes))

    plan_t, lower_t = _timed(lambda: lower_point_chunks(factored(), grid=grid))
    plan_c, lower_c = _timed(lambda: lower_point_chunks([u[2] for u in units], grid=grid))
    for a, b in zip(plan_t.targets, plan_c.targets):
        assert np.array_equal(a, b)
    assert np.array_equal(plan_t.source_eta_offset_min, plan_c.source_eta_offset_min)
    assert np.array_equal(plan_t.source_eta_offset_max, plan_c.source_eta_offset_max)
    boundary = _boundary(plan_c.boundary_points, NF)
    (v_t, g_t), (v_c, g_c), apply_t, apply_c = _apply_times(plan_t, plan_c, fields, boundary)
    scales = [_weight_scales(u[2]) for u in units]
    scale_v = np.concatenate([a for a, _ in scales])
    scale_g = np.concatenate([b for _, b in scales])
    magnitude = np.abs(fields).max()
    err_v = _relative(np.asarray(v_t) - np.asarray(v_c), np.maximum(scale_v, 1e-300)[:, None] * magnitude)
    rows = np.flatnonzero(plan_c.targets.gradient_slot >= 0)
    slots = plan_c.targets.gradient_slot[rows]
    err_g = _relative(np.asarray(g_t)[slots] - np.asarray(g_c)[slots],
                      np.maximum(scale_g[rows], 1e-300)[:, :, None] * magnitude)
    assert err_v <= TOL and err_g <= TOL, (err_v, err_g)
    per_unit = sum(lower_point_chunks([FactoredChunk(*art.decode_chunk_factored(stage, tb))],
                                      grid=grid).tensor_table_bytes for stage, _, _, tb in units)
    assert plan_t.tensor_table_bytes <= per_unit
    with capsys.disabled():
        print(f"\n[D2 real N32, all {len(units)} units in one plan] payload MB tensor {payload_nbytes(plan_t.payload) / 1e6:.2f} "
              f"(tables {plan_t.tensor_table_bytes / 1e6:.2f} MB, per-unit tables summed {per_unit / 1e6:.2f} MB) "
              f"vs all-CSR {payload_nbytes(plan_c.payload) / 1e6:.2f} MB; lowering {lower_t:.2f} s vs {lower_c:.2f} s; "
              f"apply {apply_t * 1e3:.1f} ms vs {apply_c * 1e3:.1f} ms (load {os.getloadavg()[0]:.1f}); "
              f"max rel. err value {err_v:.1e} gradient {err_g:.1e}")
