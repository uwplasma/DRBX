"""K2 qualification gates of the autodiff curvature ``K`` (P08 operator-change bundle, item 1).

Design: ``work/p08_bundle_autodiff_curvature_20260930/design.md`` section K2.  Every ``run_qk*`` function returns a
JSON-able dict (and writes ``<output_dir>/<name>.json`` when ``output_dir`` is given).

* **QK1** (:func:`run_qk1`), ``K`` itself at one grid: autodiff vs the frozen finite-difference ``K`` at the owner-closure
  raw midpoints and face q3 nodes, ~3000 random raw and face nodes, and (N32: all; N48/N64: a random sample) the wall q3
  nodes.  Relative difference (median / p99 / max), banded by ``u``, wall nodes separate (their finite-difference value is
  the frozen one-sided wall rule); a finite-difference step sweep at the 10 worst interior points; the divergence identity
  ``d_i(|J| K^i / B)`` for autodiff (autodiff of autodiff) and, for comparison, for the finite-difference ``K`` (central
  differences); per-point timing.
* **QK2** (:func:`run_qk2`), operator and reference on the bounded owner closure (P06N 14 variants, P06-legacy 4 fields):
  ``dN = N_ad - N_fd`` (q1 material / remainder / total and the q3 correction), ``dR = R_ad - R_fd``, and the new
  ``(N - R)_ad`` against ``(N - R)_fd`` per region; ``compare_to_oracle`` with autodiff on both sides.
* **QK3** (:func:`run_qk3`), the G3.3 combined RHS with autodiff on both sides against the saved finite-difference G3.3.
* **QK4** (:func:`run_qk4`), reproducibility: geometry arrays built in chunks of 4096 / 2048 / 1024 (raw and face, wall
  nodes included), bitwise comparison per array for fd, autodiff-sequential and autodiff-block ``K``; timings.

Everything runs single-process on the bounded closure (never a full-grid geometry stage).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from p_shared import owner_closure as oc                                   # noqa: E402
from p_shared import replay_units as ru                                    # noqa: E402
from p_shared import step3_gates as sg                                     # noqa: E402
from p_shared.curvature_reference import AutodiffCurvatureReference        # noqa: E402
from p_shared.provider import ScriptsGeometryProvider                      # noqa: E402
from p_shared.replay_support import DEFAULT_PATHS, build_environment       # noqa: E402
from p07_diffusion_global.numerics import quadrature                       # noqa: E402

WORKSPACE = _SCRIPTS.parents[1]                       # .../HSX drbx
GATES_DIR = WORKSPACE / "work/p08_bundle_autodiff_curvature_20260930/gates"
DEFAULT_SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
FD_G33_DIR = sg.GATES_DIR                             # saved finite-difference G3.3 reports (QK3)
SCHEMA = "drbx.p08-curvature-gates.v1"

U_BANDS = (("u<0.1", 0.0, 0.1), ("0.1<=u<=0.9", 0.1, 0.9), ("0.9<u<=0.97", 0.9, 0.97), ("u>0.97", 0.97, 1.0 + 1e-9))
SWEEP_STEPS = (8e-4, 4e-4, 2e-4, 1e-4, 5e-5, 2.5e-5)
WALL_ATOL = 8 * np.finfo(float).eps                   # the frozen _face_geometry wall test (atol, rtol = 0)

# QK2 gate thresholds
DN_OVER_NR_MAX = 1.0e-2           # max|dN| / max|(N-R)_fd| per term and grid (real variants)
NR_GROWTH_MAX = 1.10              # new region |N-R| may exceed the archived one by at most 10%
CONTROL_ABS_MAX = 1.0e-8          # p06n configuration.json constant_action_absolute_maximum
REGION_FLOOR = 1.0e-12            # absolute floor below which a region error counts as zero


# ---------------------------------------------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------------------------------------------
def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def _write(report: dict, output_dir, name: str) -> None:
    if output_dir is None:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / name).write_text(json.dumps(report, indent=2, allow_nan=False, default=_json_default))


def _num(x) -> Optional[float]:
    """A finite float or ``None`` (strict JSON has no NaN/inf)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _quantiles(rel: np.ndarray) -> dict:
    rel = np.asarray(rel, dtype=np.float64)
    if rel.size == 0:
        return {"n": 0, "median": None, "p99": None, "max": None}
    return {"n": int(rel.size), "median": _num(np.median(rel)), "p99": _num(np.quantile(rel, 0.99)),
            "max": _num(np.max(rel))}


def _banded(u: np.ndarray, rel: np.ndarray) -> dict:
    return {name: _quantiles(rel[(u >= lo) & (u < hi)]) for name, lo, hi in U_BANDS}


def _rel_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1.0e-300)


def _is_wall(u: np.ndarray) -> np.ndarray:
    return np.isclose(np.asarray(u), 1.0, rtol=0.0, atol=WALL_ATOL)


def _make_provider(sidecar, curvature: str, *, mode: Optional[str] = None, block: int = 256) -> ScriptsGeometryProvider:
    """A provider for ``curvature``; ``mode``/``block`` (autodiff only) pick the batching of the autodiff kernel
    (``None`` = the package default)."""
    provider = ScriptsGeometryProvider.from_sidecar(str(sidecar), verify_hashes=False, curvature="fd", face_quadrature="q3")
    if curvature == "fd":
        return provider
    kwargs = {} if mode is None else {"mode": mode}
    return ScriptsGeometryProvider(AutodiffCurvatureReference(provider.reference, block=block, **kwargs),
                                   curvature="autodiff", face_quadrature="q3")


def _closure_selection(env) -> dict:
    """The bounded owner closure exactly as :func:`p_shared.owner_closure.build_owner_rows` selects it (ids only)."""
    t, census = env.t, env.census
    owners = sorted(set(int(o) for o in oc.selection_fixture(t, census)["owners"]))
    owner_arr = np.asarray(owners, dtype=np.int64)
    raw_ids = np.flatnonzero(np.isin(t.ro, owner_arr)).astype(np.int64)
    incident = oc.incident_census_rows(census, owners)
    face_rows = np.intersect1d(incident, ru.face_row_selection(census))
    return {"owners": owners, "raw_ids": raw_ids, "face_rows": face_rows}


def _raw_nodes(t, raw_ids) -> np.ndarray:
    n = int(t.n)
    keys = np.array(np.unravel_index(np.asarray(raw_ids, dtype=np.int64), (n, n, n))).T.astype(np.int64)
    return quadrature(t.faces, keys, 1, face=False)[0].reshape(-1, 3)


def _face_nodes(t, census, rows) -> np.ndarray:
    keys = census.keys()[np.asarray(rows, dtype=np.int64)]
    return quadrature(t.faces, keys, 3, face=True)[0].reshape(-1, 3)


def _wall_nodes(t, count: Optional[int], rng) -> np.ndarray:
    """q3 nodes of radial faces at ``u = 1``: all of them (``count=None``) or a random ``count`` faces' worth."""
    n = int(t.n)
    j, k = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    keys = np.stack([np.zeros(n * n, np.int64), np.full(n * n, n, np.int64), j.ravel(), k.ravel()], axis=1)
    if count is not None:
        keys = keys[np.sort(rng.choice(len(keys), size=min(count, len(keys)), replace=False))]
    return quadrature(t.faces, keys, 3, face=True)[0].reshape(-1, 3)


# ---------------------------------------------------------------------------------------------------------------
# QK1 -- K itself
# ---------------------------------------------------------------------------------------------------------------
def _divergence_fd(ref, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``d_i (|J| K_fd^i / B)`` by the reference's own 4th-order central stencil, on the finite-difference ``K``.
    Returns ``(divergence (Q,), flux scale max_i |J K^i / B| (Q,))``."""
    def flux(x):
        metric = ref._metric(x)
        return np.abs(metric["J"])[:, None] * ref._curvature(x) / metric["B"][:, None]

    total = np.zeros(len(points))
    for axis in range(3):
        total += ref._derivative(lambda x, a=axis: flux(x)[:, a], points, axis)
    return total, np.max(np.abs(flux(points)), axis=1)


def run_qk1(n: int, *, input_root=WORKSPACE, sidecar_path=DEFAULT_SIDECAR, sample: int = 3000,
            wall_sample: Optional[int] = None, divergence_points: int = 600, fd_divergence_points: int = 150,
            seed: int = 0, output_dir=None) -> dict:
    """QK1 at grid ``n`` (module docstring).  ``wall_sample=None`` uses every wall q3 node (the N32 default of the
    design); a number uses that many random wall faces (x 9 nodes)."""
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar_path), curvature="autodiff", face_quadrature="q3")
    ref_ad, ref_fd = env.ref, env.ref.wrapped
    kernel = ref_ad.autodiff()
    t, census = env.t, env.census
    sel = _closure_selection(env)
    face_all = ru.face_row_selection(census)
    face_pick = np.sort(rng.choice(face_all, size=min(len(face_all), math.ceil(sample / 9)), replace=False))
    raw_pick = np.sort(rng.choice(len(t.pts), size=min(len(t.pts), sample), replace=False))
    if wall_sample is None:
        wall_count = None
    else:
        wall_count = max(1, wall_sample // 9)
    sets = {
        "closure_raw": ("raw", _raw_nodes(t, sel["raw_ids"])),
        "closure_face": ("face", _face_nodes(t, census, sel["face_rows"])),
        "random_raw": ("raw", _raw_nodes(t, raw_pick)),
        "random_face": ("face", _face_nodes(t, census, face_pick)),
        "wall_q3": ("face", _wall_nodes(t, wall_count, rng)),
    }
    import p06_structured_global.numerics as p06numerics

    kernel(sets["closure_raw"][1][:8])                        # compile outside the timings
    report_sets: dict = {}
    interior_pool: list = []
    timing: dict = {}
    for name, (kind, q) in sets.items():
        t0 = time.perf_counter(); K_ad = kernel(q); t_ad = time.perf_counter() - t0
        t0 = time.perf_counter()
        K_fd = ref_fd._curvature(q) if kind == "raw" else p06numerics._face_geometry(ref_fd, q)[2]
        t_fd = time.perf_counter() - t0
        rel = _rel_diff(K_ad, K_fd)
        u = q[:, 0]
        wall = _is_wall(u)
        entry = {"kind": kind, "nodes": int(len(q)), "wall_nodes": int(wall.sum()),
                 "all": _quantiles(rel), "interior": _quantiles(rel[~wall]), "wall": _quantiles(rel[wall]),
                 "interior_by_u": _banded(u[~wall], rel[~wall]), "wall_by_u": None,
                 "seconds_autodiff": t_ad, "seconds_fd": t_fd,
                 "us_per_point_autodiff": 1e6 * t_ad / len(q), "us_per_point_fd": 1e6 * t_fd / len(q)}
        med = float(np.median(np.linalg.norm(K_fd[~wall], axis=1))) if (~wall).any() else 1.0
        absdiff = np.linalg.norm(K_ad - K_fd, axis=1) / med
        entry["abs_diff_over_median_norm"] = {"interior": _quantiles(absdiff[~wall]), "wall": _quantiles(absdiff[wall])}
        report_sets[name] = entry
        timing[name] = (t_ad, t_fd, len(q))
        if (~wall).any():
            interior_pool.append((q[~wall], rel[~wall], K_ad[~wall]))
    # fd step sweep at the worst interior points
    pool_q = np.concatenate([p[0] for p in interior_pool]); pool_rel = np.concatenate([p[1] for p in interior_pool])
    pool_K = np.concatenate([p[2] for p in interior_pool])
    worst = np.argsort(pool_rel)[::-1][:10]
    original_step = ref_fd.finite_difference_step
    sweep_rows = []
    try:
        for idx in worst:
            errors = []
            for h in SWEEP_STEPS:
                ref_fd.finite_difference_step = h
                k = ref_fd._curvature(pool_q[idx:idx + 1])[0]
                errors.append(float(np.linalg.norm(k - pool_K[idx]) / np.linalg.norm(pool_K[idx])))
            best = min(errors)
            default_error = errors[SWEEP_STEPS.index(2e-4)]
            sweep_rows.append({"point": [float(x) for x in pool_q[idx]], "K_norm": float(np.linalg.norm(pool_K[idx])),
                               "rel_diff_at_default_step": float(pool_rel[idx]), "steps": list(SWEEP_STEPS),
                               "rel_error_vs_autodiff": errors, "best": best, "best_step": SWEEP_STEPS[int(np.argmin(errors))],
                               "improvement_best_over_default": (default_error / best) if best > 0 else None,
                               # converges onto autodiff: the error falls by >= 10x from the default step, or is
                               # already below 1e-8, at some step of the sweep
                               "converges": bool(best <= 0.1 * default_error or best <= 1.0e-8)})
    finally:
        ref_fd.finite_difference_step = original_step
    # divergence identity
    q_div = np.concatenate([sets["closure_raw"][1], sets["random_raw"][1]])
    q_div = q_div[rng.choice(len(q_div), size=min(divergence_points, len(q_div)), replace=False)]
    q_wall = sets["wall_q3"][1][rng.choice(len(sets["wall_q3"][1]), size=min(100, len(sets["wall_q3"][1])), replace=False)]
    h = float(original_step)

    def ad_div(q):
        terms = kernel.divergence_identity_terms(q)
        div = terms.sum(axis=-1)
        flux = np.max(np.abs(kernel.flux(q)), axis=1)
        return div, np.abs(terms).sum(axis=-1), flux

    div, term_scale, flux = ad_div(q_div)
    div_w, term_scale_w, flux_w = ad_div(q_wall)
    q_fd = q_div[:fd_divergence_points]
    div_fd, flux_fd = _divergence_fd(ref_fd, q_fd)
    div_ad_same, _, flux_same = ad_div(q_fd)
    divergence = {
        "step_h": h,
        "autodiff_raw": {"n": int(len(q_div)), "abs_over_sum_abs_terms": _quantiles(np.abs(div) / term_scale),
                         "abs_times_h_over_flux": _quantiles(np.abs(div) * h / flux)},
        "autodiff_wall": {"n": int(len(q_wall)), "abs_over_sum_abs_terms": _quantiles(np.abs(div_w) / term_scale_w),
                          "abs_times_h_over_flux": _quantiles(np.abs(div_w) * h / flux_w)},
        "fd_central_differences": {"n": int(len(q_fd)), "abs_times_h_over_flux": _quantiles(np.abs(div_fd) * h / flux_fd),
                                   "abs_over_flux": _quantiles(np.abs(div_fd) / flux_fd)},
        "autodiff_same_points": {"n": int(len(q_fd)), "abs_times_h_over_flux": _quantiles(np.abs(div_ad_same) * h / flux_same),
                                 "abs_over_flux": _quantiles(np.abs(div_ad_same) / flux_same)},
    }
    interior_all = [report_sets[k]["interior"] for k in ("closure_raw", "closure_face", "random_raw", "random_face")]
    report = {
        "schema": SCHEMA, "gate": "QK1", "n": int(n), "kernel_mode": kernel.mode, "kernel_block": kernel.block,
        "sets": report_sets, "fd_step_sweep_worst_interior": sweep_rows,
        "sweep_all_converge": bool(all(r["converges"] for r in sweep_rows)),
        "divergence_identity": divergence,
        "interior_median_max": max(x["median"] for x in interior_all if x["median"] is not None),
        "seconds": time.perf_counter() - started,
    }
    _write(report, output_dir, f"N{n}_qk1.json")
    return report


# ---------------------------------------------------------------------------------------------------------------
# QK4 -- reproducibility (geometry chunk size); decides the autodiff batching default
# ---------------------------------------------------------------------------------------------------------------
QK4_CHUNKS = (4096, 2048, 1024)


def _qk4_unit(env, raw_count: int, face_count: int, seed: int = 0):
    """A bounded unit: the outermost ``raw_count`` raw cells (a contiguous block of the last radial layers) and
    ``face_count`` census face rows of the outer layers that include every wall (u = 1) face row."""
    t, census = env.t, env.census
    n = int(t.n)
    rng = np.random.default_rng(seed)
    raw_ids = np.arange(n ** 3 - raw_count, n ** 3, dtype=np.int64)
    face_rows = ru.face_row_selection(census)
    keys = census.keys()[face_rows]
    wall = (keys[:, 0] == 0) & (keys[:, 1] == n)
    outer = (keys[:, 1] >= n - 3) & ~wall
    wall_rows = face_rows[wall]
    other = face_rows[outer]
    other = np.sort(rng.choice(other, size=max(0, min(len(other), face_count - len(wall_rows))), replace=False))
    rows = np.sort(np.concatenate([wall_rows, other]))
    raw_keys = np.array(np.unravel_index(raw_ids, (n, n, n))).T.astype(np.int64)
    return raw_keys, census.keys()[rows], int(len(wall_rows))


def _compare_arrays(reference, other) -> dict:
    out = {}
    for name in reference:
        a, b = np.asarray(reference[name]), np.asarray(other[name])
        equal = bool(np.array_equal(a, b, equal_nan=True))
        entry = {"bitwise": equal}
        if not equal:
            diff = np.abs(a - b)
            scale = max(float(np.max(np.abs(a))), 1e-300)
            entry.update(max_abs=_num(np.nanmax(diff)), max_rel_to_max=_num(np.nanmax(diff) / scale),
                         differing_entries=int(np.count_nonzero(diff > 0)))
        out[name] = entry
    return out


def run_qk4(n: int = 32, *, input_root=WORKSPACE, sidecar_path=DEFAULT_SIDECAR, raw_count: int = 4608,
            face_count: int = 4608, chunks: Sequence[int] = QK4_CHUNKS,
            configs: Sequence[str] = ("fd", "autodiff:sequential", "autodiff:block"), output_dir=None) -> dict:
    """QK4: geometry arrays of a bounded unit built with several geometry chunk sizes, compared bitwise per array
    against the first chunk size; the verdict for K is ``bitwise_K`` and, per config, the geometry-stage seconds."""
    started = time.perf_counter()
    env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar_path), curvature="fd", face_quadrature="q3")
    raw_keys, face_keys, n_wall_rows = _qk4_unit(env, raw_count, face_count)
    faces = env.ctx.faces
    raw_fields, face_fields = oc._RAW_GEOMETRY_FIELDS, oc._FACE_GEOMETRY_FIELDS
    K_fields = ("p06_raw_K", "p06_face_K")
    results: dict = {}
    for config in configs:
        curvature, _, mode = config.partition(":")
        provider = _make_provider(sidecar_path, curvature, mode=mode or None)
        if curvature == "autodiff":                              # compile outside the timings
            provider.reference.autodiff()(np.asarray([[0.5, 0.5, 0.5]]))
        built = {}
        seconds = {}
        for chunk in chunks:
            t0 = time.perf_counter()
            geom = oc._owner_geometry_arrays(provider, faces, raw_keys, face_keys, raw_chunk=chunk, face_chunk=chunk)
            seconds[chunk] = time.perf_counter() - t0
            built[chunk] = {f: np.array(getattr(geom, f)) for f in (*raw_fields, *face_fields)}
        base = chunks[0]
        comparisons = {str(c): _compare_arrays(built[base], built[c]) for c in chunks[1:]}
        bitwise_by_array = {f: all(comparisons[str(c)][f]["bitwise"] for c in chunks[1:]) for f in built[base]}
        results[config] = {
            "geometry_seconds_by_chunk": {str(c): s for c, s in seconds.items()},
            "us_per_raw_plus_face_node": {str(c): 1e6 * s / (len(raw_keys) + 9 * len(face_keys)) for c, s in seconds.items()},
            "bitwise_by_array": bitwise_by_array,
            "bitwise_K": bool(all(bitwise_by_array[f] for f in K_fields)),
            "bitwise_all": bool(all(bitwise_by_array.values())),
            "differences": {c: {f: e for f, e in comp.items() if not e["bitwise"]} for c, comp in comparisons.items()},
        }
        del built
        gc.collect()
    K_only = _qk4_k_timing(env, sidecar_path, raw_keys, face_keys)
    p07 = {}
    for config, r in results.items():
        p07[config] = r["bitwise_by_array"]["p07_raw_divergence"]
    report = {
        "schema": SCHEMA, "gate": "QK4", "n": int(n), "chunks": list(chunks),
        "unit": {"raw_cells": int(len(raw_keys)), "face_rows": int(len(face_keys)), "face_nodes": int(9 * len(face_keys)),
                 "wall_face_rows": n_wall_rows},
        "configs": results, "K_only_timing": K_only,
        "p07_raw_divergence_bitwise_independent_of_chunking": p07,
        "decision": {
            "block_bitwise_K": results.get("autodiff:block", {}).get("bitwise_K"),
            "sequential_bitwise_K": results.get("autodiff:sequential", {}).get("bitwise_K"),
            "fd_bitwise_K": results.get("fd", {}).get("bitwise_K"),
            "default_mode": ("block" if results.get("autodiff:block", {}).get("bitwise_K") else "sequential"),
        },
        "seconds": time.perf_counter() - started,
    }
    _write(report, output_dir, f"N{n}_qk4.json")
    return report


def _qk4_k_timing(env, sidecar_path, raw_keys, face_keys) -> dict:
    """K-only microseconds per point on the unit's nodes: fd, autodiff sequential, autodiff block."""
    t = env.t
    faces = env.ctx.faces
    q = np.concatenate([quadrature(faces, raw_keys, 1, face=False)[0].reshape(-1, 3),
                        quadrature(faces, face_keys, 3, face=True)[0].reshape(-1, 3)])
    q = q[~_is_wall(q[:, 0])][:20000]
    out = {"points": int(len(q))}
    ref_fd = _make_provider(sidecar_path, "fd").reference
    t0 = time.perf_counter(); ref_fd._curvature(q); out["us_per_point_fd"] = 1e6 * (time.perf_counter() - t0) / len(q)
    for mode in ("sequential", "block"):
        ref = _make_provider(sidecar_path, "autodiff", mode=mode).reference
        ref._curvature(q[:8])
        t0 = time.perf_counter(); ref._curvature(q); out[f"us_per_point_{mode}"] = 1e6 * (time.perf_counter() - t0) / len(q)
    return out


# ---------------------------------------------------------------------------------------------------------------
# QK2 -- operator and reference on the owner closure
# ---------------------------------------------------------------------------------------------------------------
def _legacy_reference(env, built, evolution_volume_dense, owner_arr, fields) -> dict:
    """P06-legacy ``reference_evolution:{field}:{term}`` at the closure owners recomputed with ``env.ref``
    (the frozen construction: ``sum_raw evolution_weight * _continuum_terms(exact) / evolution_volume``)."""
    import p06_structured_global.numerics as p06numerics
    from perpendicular_structured.reference_geometry import curvature_geometry
    from p_shared import campaign_fields as cf

    t, n = env.t, int(env.t.n)
    raw_ids = built["raw_ids"]
    raw_keys = np.array(np.unravel_index(raw_ids, (n, n, n))).T.astype(np.int64)
    points, weight = quadrature(t.faces, raw_keys, 1, face=False)
    points = points.reshape(-1, 3)
    prepared = curvature_geometry(env.ref, points)
    evolution_weight = weight.reshape(-1) * np.asarray(prepared.J) / np.maximum(np.asarray(prepared.B), 1.0e-30)
    owner_pos = np.searchsorted(owner_arr, t.ro[raw_ids])
    out = {}
    for field in fields:
        values, gradients = p06numerics._evaluate_fields(field, env.ref, points, cf.P06_LEGACY_TIME_VALUE)
        terms = p06numerics._continuum_terms(values, gradients, prepared)
        for ti, term in enumerate(("material", "remainder", "total")):
            acc = np.zeros((len(owner_arr), 4))
            np.add.at(acc, owner_pos, evolution_weight[:, None] * terms[ti])
            out[(field, term)] = acc / np.maximum(evolution_volume_dense, 1e-300)[:, None]
    return out


def _collect_closure(n: int, curvature: str, *, input_root, sidecar_path, paths, campaigns=("p06n", "p06_legacy")) -> dict:
    """Owner-closure host terms of P06N and P06-legacy (+ the oracle table) with ``curvature`` on both sides."""
    import p06_structured_global.numerics as p06numerics
    import p06n_field_derived_global.core as p06n_core

    started = time.perf_counter()
    env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar_path), curvature=curvature,
                            face_quadrature="q3")
    t = env.t
    owners = sorted(set(int(o) for o in oc.selection_fixture(t, env.census)["owners"]))
    owner_arr = np.asarray(owners, dtype=np.int64)
    provider = oc.load_provider_for_env(sidecar_path, curvature=curvature, face_quadrature="q3")
    built = oc.build_owner_rows(env, owners, provider=provider)
    oracle = ru._load_oracle_owner_values(env, dict(paths), campaigns)
    out = oc.assemble_owner_terms(env, built, campaigns, oracle)
    table = oc.compare_to_oracle(env, out, owners, dict(paths), campaigns)

    def dense(pair):
        return oc.owner_values_from_pairs(pair, owner_arr, len(t.vol))

    evolution = dense(out["cells"]["q1_evolution_volume"])
    safe = np.maximum(evolution, 1e-300)
    variants = list(p06n_core.VARIANT_NAMES)
    data = {"n": n, "curvature": curvature, "owners": owners, "owner_volume": np.asarray(t.vol)[owner_arr],
            "evolution_volume": evolution, "variants": variants, "table": table}
    for label in ("material", "remainder", "total", "R_material", "R_remainder", "R_total"):
        data[f"p06n_{label}"] = np.stack([dense(out["cells"][f"p06n_raw_{label}"][vi]) / safe[:, None]
                                          for vi in range(len(variants))])
    data["p06n_correction"] = np.stack([dense(out["faces"]["p06n_faces_correction"][vi]) / safe[:, None]
                                        for vi in range(len(variants))])
    fields = list(p06numerics.FIELD_NAMES)
    data["legacy_fields"] = fields
    for field in fields:
        for term in ("material", "remainder", "total"):
            data[f"legacy_{field}_{term}"] = dense(out["cells"]["p06legacy_raw_centered"][field][term]) / safe[:, None]
        data[f"legacy_{field}_correction"] = dense(out["faces"]["p06legacy_faces_correction"][field]) / safe[:, None]
    ref_terms = _legacy_reference(env, built, evolution, owner_arr, fields)
    for (field, term), value in ref_terms.items():
        data[f"legacy_{field}_R_{term}"] = value
    with np.load(Path(paths["p06_legacy"]) / f"N{n}.npz", allow_pickle=False) as z:
        for field in fields:
            for term in ("material", "remainder", "total"):
                data[f"legacy_{field}_Rsaved_{term}"] = np.array(z[f"reference_evolution:{field}:{term}"][owner_arr])
    masks = p06n_core.regional_masks(t)
    data["region_masks"] = {name: np.asarray(mask)[owner_arr] for name, mask in masks.items()}
    data["region_masks"]["all"] = np.ones(len(owner_arr), dtype=bool)
    data["seconds"] = time.perf_counter() - started
    return data


def _region_l2(error: np.ndarray, weight: np.ndarray, masks: dict) -> dict:
    """Owner-volume weighted L2 and max of ``error (O, c)`` per region (per component, ``(c,)`` lists)."""
    out = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        w = weight[mask]
        e = error[mask]
        out[name] = {"owners": int(mask.sum()), "l2": np.sqrt(np.sum(w[:, None] * e ** 2, axis=0) / w.sum()),
                     "max_abs": np.max(np.abs(e), axis=0)}
    return out


def _region_ratios(err_ad: np.ndarray, err_fd: np.ndarray, weight: np.ndarray, masks: dict, components: slice) -> dict:
    """Per region: worst over components of ``l2_ad / l2_fd`` (or, below the floor, ``1 + (l2_ad - l2_fd) / floor``)."""
    ad = _region_l2(err_ad[:, components], weight, masks)
    fd = _region_l2(err_fd[:, components], weight, masks)
    out = {}
    for name in fd:
        ratios = []
        for a, f in zip(ad[name]["l2"], fd[name]["l2"]):
            ratios.append(a / f if f > REGION_FLOOR else 1.0 + max(a - f, 0.0) / REGION_FLOOR)
        out[name] = {"owners": fd[name]["owners"], "l2_fd": [float(x) for x in fd[name]["l2"]],
                     "l2_ad": [float(x) for x in ad[name]["l2"]], "ratio_max": float(max(ratios))}
    return out


def _compare_closure(fd: dict, ad: dict) -> dict:
    """The QK2 tables from two :func:`_collect_closure` results."""
    weight = fd["owner_volume"]
    masks = fd["region_masks"]
    variants = fd["variants"]
    real = [i for i, v in enumerate(variants) if not v.startswith("control_")]
    control = [i for i, v in enumerate(variants) if v.startswith("control_")]

    def NR(d, i, candidate):
        material = d["p06n_material"][i] + (d["p06n_correction"][i] if candidate == "U" else 0.0)
        return {"material": material - d["p06n_R_material"][i], "remainder": d["p06n_remainder"][i] - d["p06n_R_remainder"][i],
                "total": material + d["p06n_remainder"][i] - d["p06n_R_total"][i]}

    # ---- deltas, real variants (per term over real variants, owners, components) ----
    nr_fd_max = {c: max(float(np.max(np.abs(NR(fd, i, c)["total"]))) for i in real) for c in ("centered", "U")}
    scale = nr_fd_max["U"]                                   # scale of the q3 correction: the U total error
    deltas = {}
    for term in ("material", "remainder", "total"):
        dN = max(float(np.max(np.abs(ad[f"p06n_{term}"][i] - fd[f"p06n_{term}"][i]))) for i in real)
        dR = max(float(np.max(np.abs(ad[f"p06n_R_{term}"][i] - fd[f"p06n_R_{term}"][i]))) for i in real)
        nr = max(float(np.max(np.abs(NR(fd, i, "centered")[term]))) for i in real)
        deltas[f"q1_{term}"] = {"max_dN": dN, "max_dR": dR, "max_abs_NR_fd": nr, "dN_over_NR": dN / nr, "dR_over_NR": dR / nr,
                                "pass": dN / nr <= DN_OVER_NR_MAX}
    dcorr = max(float(np.max(np.abs(ad["p06n_correction"][i] - fd["p06n_correction"][i]))) for i in real)
    deltas["q3_correction"] = {"max_dN": dcorr, "max_abs_correction_fd": max(float(np.max(np.abs(fd["p06n_correction"][i]))) for i in real),
                               "max_abs_NR_fd_U_total": scale, "dN_over_NR": dcorr / scale, "pass": dcorr / scale <= DN_OVER_NR_MAX}
    # ---- per variant table ----
    per_variant = {}
    for i, name in enumerate(variants):
        row = {"kind": "control" if i in control else "real"}
        for term in ("material", "remainder", "total"):
            row[f"max_dN_{term}"] = float(np.max(np.abs(ad[f"p06n_{term}"][i] - fd[f"p06n_{term}"][i])))
            row[f"max_dR_{term}"] = float(np.max(np.abs(ad[f"p06n_R_{term}"][i] - fd[f"p06n_R_{term}"][i])))
        row["max_dN_correction"] = float(np.max(np.abs(ad["p06n_correction"][i] - fd["p06n_correction"][i])))
        if i in control:
            row["max_abs_action_ad"] = float(max(np.max(np.abs(ad["p06n_total"][i])),
                                                 np.max(np.abs(ad["p06n_total"][i] + ad["p06n_correction"][i]))))
            row["max_abs_action_fd"] = float(max(np.max(np.abs(fd["p06n_total"][i])),
                                                 np.max(np.abs(fd["p06n_total"][i] + fd["p06n_correction"][i]))))
        per_variant[name] = row
    controls = {"variants": [variants[i] for i in control],
                "max_abs_action_ad": max(per_variant[variants[i]]["max_abs_action_ad"] for i in control),
                "max_abs_action_fd": max(per_variant[variants[i]]["max_abs_action_fd"] for i in control),
                "max_dN": max(max(per_variant[variants[i]][k] for k in per_variant[variants[i]]
                                  if k.startswith("max_dN")) for i in control),
                "abs_limit": CONTROL_ABS_MAX}
    controls["pass"] = bool(controls["max_abs_action_ad"] <= CONTROL_ABS_MAX and controls["max_dN"] <= CONTROL_ABS_MAX)
    # ---- (N - R) new vs archived, per region ----
    regions = {}
    worst_region_ratio = 0.0
    for cand in ("centered", "U"):
        regions[cand] = {}
        for term in ("material", "remainder", "total"):
            per_region: dict = {}
            for i in real:
                r = _region_ratios(NR(ad, i, cand)[term], NR(fd, i, cand)[term], weight, masks, slice(0, 3))
                for name, entry in r.items():
                    cur = per_region.get(name)
                    if cur is None or entry["ratio_max"] > cur["ratio_max"]:
                        per_region[name] = dict(entry, variant=variants[i])
            regions[cand][term] = per_region
            if term == "total":
                worst_region_ratio = max([worst_region_ratio] + [e["ratio_max"] for e in per_region.values()])
    # ---- P06-legacy ----
    legacy = {}
    for field in fd["legacy_fields"]:
        row = {}
        for term in ("material", "remainder", "total"):
            key = f"legacy_{field}_{term}"
            row[f"max_dN_{term}"] = float(np.max(np.abs(ad[key] - fd[key])))
            row[f"max_dR_{term}"] = float(np.max(np.abs(ad[f"legacy_{field}_R_{term}"] - fd[f"legacy_{field}_R_{term}"])))
        row["max_dN_correction"] = float(np.max(np.abs(ad[f"legacy_{field}_correction"] - fd[f"legacy_{field}_correction"])))
        row["R_fd_recompute_vs_saved_max_abs"] = float(max(np.max(np.abs(fd[f"legacy_{field}_R_{t}"] - fd[f"legacy_{field}_Rsaved_{t}"]))
                                                           for t in ("material", "remainder", "total")))

        def leg_nr(d, cand):
            material = d[f"legacy_{field}_material"] + (d[f"legacy_{field}_correction"] if cand == "U" else 0.0)
            return material + d[f"legacy_{field}_remainder"] - d[f"legacy_{field}_R_total"]

        row["max_abs_NR_fd_centered"] = float(np.max(np.abs(leg_nr(fd, "centered"))))
        row["max_abs_NR_fd_U"] = float(np.max(np.abs(leg_nr(fd, "U"))))
        row["regions"] = {cand: _region_ratios(leg_nr(ad, cand), leg_nr(fd, cand), weight, masks, slice(0, 4)) for cand in ("centered", "U")}
        row["region_ratio_max"] = max(e["ratio_max"] for cand in row["regions"].values() for e in cand.values())
        legacy[field] = row
    leg_scale = max(max(r["max_abs_NR_fd_centered"], r["max_abs_NR_fd_U"]) for r in legacy.values())
    leg_dN = max(max(r[k] for k in r if k.startswith("max_dN")) for r in legacy.values())
    legacy_gate = {"max_dN": leg_dN, "max_abs_NR_fd": leg_scale, "dN_over_NR": leg_dN / leg_scale,
                   "max_dR": max(max(r[k] for k in r if k.startswith("max_dR")) for r in legacy.values()),
                   "region_ratio_max": max(r["region_ratio_max"] for r in legacy.values())}
    legacy_gate["pass"] = bool(legacy_gate["dN_over_NR"] <= DN_OVER_NR_MAX and legacy_gate["region_ratio_max"] <= NR_GROWTH_MAX)
    # ---- oracle rows ----
    def summarize(table):
        failing = [{k: r[k] for k in ("campaign", "term", "max_abs", "ratio_to_oracle_NR")} for r in table if not r["pass"]]
        return {"rows": len(table), "passed": sum(bool(r["pass"]) for r in table), "failing": failing}

    oracle_rows = []
    for r_fd, r_ad in zip(fd["table"], ad["table"]):
        oracle_rows.append({"campaign": r_ad["campaign"], "term": r_ad["term"], "pass_fd": bool(r_fd["pass"]), "pass_ad": bool(r_ad["pass"]),
                            "max_abs_fd": r_fd["max_abs"], "max_abs_ad": r_ad["max_abs"],
                            "ratio_fd": r_fd["ratio_to_oracle_NR"], "ratio_ad": r_ad["ratio_to_oracle_NR"]})
    real_pass = bool(all(d["pass"] for d in deltas.values()) and worst_region_ratio <= NR_GROWTH_MAX)
    return {
        "owners": fd["owners"], "variants": variants,
        "p06n": {"deltas": deltas, "real_variant_count": len(real), "control": controls,
                 "nr_fd_max": nr_fd_max, "regions": regions, "worst_region_ratio_total": worst_region_ratio,
                 "per_variant": per_variant, "pass": bool(real_pass and controls["pass"])},
        "p06_legacy": {"fields": legacy, "gate": legacy_gate},
        "oracle": {"fd": summarize(fd["table"]), "autodiff": summarize(ad["table"]), "rows": oracle_rows},
        "pass": bool(real_pass and controls["pass"] and legacy_gate["pass"]),
    }


def run_qk2(n: int, *, input_root=WORKSPACE, sidecar_path=DEFAULT_SIDECAR, paths: Optional[dict] = None,
            output_dir=None) -> dict:
    """QK2 at grid ``n``: finite-difference and autodiff owner-closure host terms, their differences and the tables."""
    started = time.perf_counter()
    paths = dict(DEFAULT_PATHS if paths is None else paths)
    fd = _collect_closure(n, "fd", input_root=input_root, sidecar_path=sidecar_path, paths=paths)
    gc.collect()
    ad = _collect_closure(n, "autodiff", input_root=input_root, sidecar_path=sidecar_path, paths=paths)
    report = {"schema": SCHEMA, "gate": "QK2", "n": int(n), **_compare_closure(fd, ad),
              "seconds_collect": {"fd": fd["seconds"], "autodiff": ad["seconds"]}, "seconds": time.perf_counter() - started}
    _write(report, output_dir, f"N{n}_qk2.json")
    return report


# ---------------------------------------------------------------------------------------------------------------
# QK3 -- combined RHS (G3.3)
# ---------------------------------------------------------------------------------------------------------------
def _compare_g33(ad: dict, fd: dict) -> dict:
    rows = []
    for variant in ad["results"]:
        for field in ad["results"][variant]["fields"]:
            for term in sg.G33_TERMS:
                a = ad["results"][variant]["fields"][field][term]
                f = fd["results"][variant]["fields"][field][term]
                d_rel = None if a["rel_l2"] is None or f["rel_l2"] in (None, 0.0) else abs(a["rel_l2"] - f["rel_l2"]) / f["rel_l2"]
                rows.append({"variant": variant, "field": field, "term": term, "rel_l2_fd": f["rel_l2"], "rel_l2_ad": a["rel_l2"],
                             "rel_change_of_rel_l2": d_rel, "fit_scale_fd": f["fit_scale"], "fit_scale_ad": a["fit_scale"],
                             "d_fit_scale": (None if a["fit_scale"] is None or f["fit_scale"] is None
                                             else abs(a["fit_scale"] - f["fit_scale"]))})
    changes = [r["rel_change_of_rel_l2"] for r in rows if r["rel_change_of_rel_l2"] is not None]
    fits = [r["fit_scale_ad"] for r in rows if r["fit_scale_ad"] is not None]
    curv = [r for r in rows if r["term"] == "curvature"]
    return {"rows": rows, "max_rel_change_of_rel_l2": max(changes, default=0.0),
            "max_rel_change_of_rel_l2_curvature": max((r["rel_change_of_rel_l2"] for r in curv if r["rel_change_of_rel_l2"] is not None), default=0.0),
            "max_d_fit_scale": max((r["d_fit_scale"] for r in rows if r["d_fit_scale"] is not None), default=0.0),
            "fit_scale_range_ad": [min(fits), max(fits)] if fits else None,
            "max_rel_l2_fd": fd["max_rel_l2"], "max_rel_l2_ad": ad["max_rel_l2"]}


def run_qk3(n: int, *, input_root=WORKSPACE, sidecar_path=DEFAULT_SIDECAR, fd_dir=FD_G33_DIR, output_dir=None,
            report_out: Optional[dict] = None) -> dict:
    """QK3 at grid ``n``: ``run_g33`` with autodiff on both sides against the saved finite-difference ``N{n}_g33.json``.
    The autodiff G3.3 report itself is stored in ``report_out[n]`` when given (for :func:`qk3_orders`)."""
    started = time.perf_counter()
    setup = sg.build_setup(n, ("p06n",), input_root=input_root, sidecar_path=sidecar_path, curvature="autodiff", face_quadrature="q3")
    ad = sg.run_g33(n, setup=setup)
    del setup
    gc.collect()
    fd = json.loads((Path(fd_dir) / f"N{n}_g33.json").read_text())
    if report_out is not None:
        report_out[int(n)] = ad
    report = {"schema": SCHEMA, "gate": "QK3", "n": int(n), "curvature": ad["curvature"], **_compare_g33(ad, fd),
              "seconds": time.perf_counter() - started}
    _write(report, output_dir, f"N{n}_qk3.json")
    _write(ad, output_dir, f"N{n}_g33_autodiff.json")
    return report


def qk3_orders(reports: dict, fd_dir=FD_G33_DIR, *, output_dir=None) -> dict:
    """Observed orders of the autodiff G3.3 reports ``{n: report}`` against the saved finite-difference orders:
    per (variant, field, term) the ``rel_l2`` orders, their change, and the flags.  ``convergence_ok``: no new flag and
    every order at least the finite-difference order minus 0.05."""
    grids = sorted(int(n) for n in reports)
    ad = sg.observed_orders({n: reports[n] for n in grids})
    fd_reports = {n: json.loads((Path(fd_dir) / f"N{n}_g33.json").read_text()) for n in grids}
    fd = sg.observed_orders(fd_reports)
    rows = []
    ok = True
    for variant in ad:
        for field in ad[variant]:
            for term in ad[variant][field]:
                a, f = ad[variant][field][term], fd[variant][field][term]
                oa, of = a["order_rel_l2"], f["order_rel_l2"]
                new_flags = [x for x in a["flags"] if x not in f["flags"]]
                worse = any(x is not None and y is not None and x < y - 0.05 for x, y in zip(oa, of))
                ok = ok and not new_flags and not worse
                rows.append({"variant": variant, "field": field, "term": term, "order_fd": of, "order_ad": oa,
                             "d_order": [None if x is None or y is None else x - y for x, y in zip(oa, of)],
                             "flags_fd": f["flags"], "flags_ad": a["flags"], "new_flags": new_flags, "worse": worse})
    report = {"schema": SCHEMA, "gate": "QK3_orders", "grids": grids, "rows": rows, "convergence_ok": bool(ok),
              "max_abs_d_order": max((abs(x) for r in rows for x in r["d_order"] if x is not None), default=0.0)}
    _write(report, output_dir, "qk3_orders.json")
    return report


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------
def _summary_line(report: dict) -> str:
    gate = report["gate"]
    if gate == "QK1":
        parts = [f"{k}: med {v['interior']['median']:.1e} p99 {v['interior']['p99']:.1e} max {v['interior']['max']:.1e}"
                 for k, v in report["sets"].items() if v["interior"]["n"]]
        w = report["sets"]["wall_q3"]["wall"]
        return (f"QK1 N{report['n']}: " + " | ".join(parts) + f" | wall med {w['median']:.1e} max {w['max']:.1e}"
                f" | sweep converge {report['sweep_all_converge']}")
    if gate == "QK2":
        d = report["p06n"]["deltas"]
        return (f"QK2 N{report['n']}: pass={report['pass']} dN/NR " + ", ".join(f"{k} {v['dN_over_NR']:.1e}" for k, v in d.items())
                + f" | region ratio {report['p06n']['worst_region_ratio_total']:.6f} | legacy dN/NR {report['p06_legacy']['gate']['dN_over_NR']:.1e}"
                f" | oracle ad {report['oracle']['autodiff']['passed']}/{report['oracle']['autodiff']['rows']}")
    if gate == "QK3":
        return (f"QK3 N{report['n']}: max rel change of rel_l2 {report['max_rel_change_of_rel_l2']:.2e}, "
                f"fit range {report['fit_scale_range_ad']}, max rel_l2 fd {report['max_rel_l2_fd']:.6f} ad {report['max_rel_l2_ad']:.6f}")
    if gate == "QK4":
        return "QK4 N%d: %s" % (report["n"], json.dumps(report["decision"]))
    return gate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grids", default="32")
    parser.add_argument("--gates", default="qk4,qk1,qk2,qk3")
    parser.add_argument("--output-dir", default=str(GATES_DIR))
    parser.add_argument("--input-root", default=str(WORKSPACE))
    parser.add_argument("--sidecar", default=str(DEFAULT_SIDECAR))
    args = parser.parse_args(argv)
    gates = set(args.gates.split(","))
    grids = [int(g) for g in args.grids.split(",")]
    common = dict(input_root=args.input_root, sidecar_path=args.sidecar, output_dir=args.output_dir)
    ok = True
    if "qk4" in gates:
        report = run_qk4(grids[0], **common)
        print(_summary_line(report), flush=True)
    g33_reports: dict = {}
    for n in grids:
        if "qk1" in gates:
            report = run_qk1(n, wall_sample=None if n == 32 else 3000, **common)
            print(_summary_line(report), flush=True)
            ok = ok and report["sweep_all_converge"]
        if "qk2" in gates:
            report = run_qk2(n, **common)
            print(_summary_line(report), flush=True)
            ok = ok and report["pass"]
        if "qk3" in gates:
            report = run_qk3(n, report_out=g33_reports, **common)
            print(_summary_line(report), flush=True)
    if "qk3" in gates and len(g33_reports) >= 2:
        orders = qk3_orders(g33_reports, output_dir=args.output_dir)
        print(f"QK3 orders: convergence_ok={orders['convergence_ok']} max |d order| {orders['max_abs_d_order']:.2e}")
        ok = ok and orders["convergence_ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
