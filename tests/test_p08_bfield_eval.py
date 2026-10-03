"""Tests for ``scripts/p08_bfield_eval`` (spline vs compact-C3 B evaluator comparison on a region x knot-class sample).

Fast tests are synthetic (no HSX inputs, no geometry): knot classification, region codes, sampling, the metric / aggregation
code and the analysis (summary, flags, coefficient table, report).  One real smoke test (``slow``; skipped when the inputs are
missing) runs N32 with the spline evaluator and one owner per cell.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]              # .../DRBX
SCRIPTS = REPO / "scripts"
WORKSPACE = REPO.parent                                 # .../HSX drbx
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p08_bfield_eval import analyze, metrics as M, sampling as S    # noqa: E402

VARIANTS = ("main_phi_dirichlet", "main_phi_neumann")


# ---------------------------------------------------------------------------
# knot classification, regions, sampling
# ---------------------------------------------------------------------------
def test_classify_knot_plane_mid_and_unclassified():
    dphi = 0.1
    phi = np.array([0.0, 0.019, 0.021, 0.049, 0.031, 0.05, 0.069, 0.079, 0.099, 0.1])
    #  plane (< 0.2 dphi = 0.02 from a plane): 0.0, 0.019, 0.099 (0.001 away), 0.1 ;  mid (|frac - 0.5| < 0.2): 0.031, 0.049,
    #  0.05, 0.069 ; unclassified: 0.021, 0.079
    got = S.classify_knot(phi, 0.0, dphi)
    assert got.tolist() == [0, 0, -1, 1, 1, 1, 1, -1, 0, 0]


def test_classify_knot_is_periodic_and_uses_the_plane_offset():
    dphi, phi0 = 0.05, 0.013
    base = np.array([phi0, phi0 + 0.5 * dphi, phi0 + 0.25 * dphi])
    np.testing.assert_array_equal(S.classify_knot(base, phi0, dphi), [0, 1, -1])
    np.testing.assert_array_equal(S.classify_knot(base + 11 * dphi, phi0, dphi), [0, 1, -1])
    np.testing.assert_array_equal(S.classify_knot(base - 5 * dphi, phi0, dphi), [0, 1, -1])
    assert S.classify_knot(base[:0], phi0, dphi).shape == (0,)
    with pytest.raises(ValueError):
        S.classify_knot(base, 0.0, 0.0)


def test_plane_grid_checks_that_the_planes_tile_the_period():
    period, nphi = np.pi / 2, 90
    good = SimpleNamespace(phi=np.arange(nphi) * period / nphi, period=period)
    phi0, dphi = S.plane_grid(good)
    assert phi0 == 0.0 and dphi == pytest.approx(period / nphi)
    with pytest.raises(ValueError):
        S.plane_grid(SimpleNamespace(phi=np.arange(nphi) * period / 80, period=period))


def _fake_topology(n=4):
    """n^3 raw cells, owner = raw cell (identity) except the axis radius row (radius 0) aggregated into 2 owners."""
    ids = np.arange(n ** 3)
    ro = ids.copy()
    radius = ids // (n * n)
    ro[radius == 0] = 0                                  # all radius-0 cells share owner 0
    _u, ro = np.unique(ro, return_inverse=True)
    return SimpleNamespace(n=n, ro=ro, vol=np.ones(ro.max() + 1), rv=np.ones(n ** 3), pts=np.zeros((n ** 3, 3)))


def test_axis_core_mask_marks_owners_with_a_radius_zero_cell():
    t = _fake_topology()
    mask = S.axis_core_mask(t)
    assert mask.sum() == 1 and mask[0]


def test_region_codes_are_disjoint_with_priority_order():
    m = 8
    z = np.zeros(m, dtype=bool)

    def on(*idx):
        out = z.copy()
        out[list(idx)] = True
        return out

    masks = {"physical_wall": on(1), "transverse_last_two_layers": on(1, 2, 3), "transition": on(0, 3, 4),
             "aggregate": on(4, 5), "ordinary": ~on(0, 1, 4, 5)}
    code = S.region_codes(masks, on(0))
    names = [S.REGIONS[c] for c in code]
    assert names == ["axis_core", "physical_wall", "last_two_layers", "last_two_layers", "transition", "aggregate",
                     "ordinary", "ordinary"]
    assert (code >= 0).all()


def test_owner_centroid_phi_is_a_volume_weighted_circular_mean():
    t = SimpleNamespace(pts=np.stack([np.arange(4.0), np.zeros(4), np.zeros(4)], axis=1), ro=np.array([0, 0, 1, 1]), rv=np.array([1.0, 1.0, 1.0, 3.0]), vol=np.ones(2))
    angles = np.array([np.pi - 0.1, -np.pi + 0.1, 0.2, 0.4])      # owner 0 straddles the branch cut

    def position(points):
        a = angles[points[:, 0].astype(int)]                # the first logical coordinate carries the raw index
        return np.stack([np.cos(a), np.sin(a), np.zeros(len(a))], axis=1)

    phi = S.owner_centroid_phi(t, position, chunk=3)
    assert abs(abs(phi[0]) - np.pi) < 1e-12
    expected = np.arctan2(np.sin(0.2) + 3 * np.sin(0.4), np.cos(0.2) + 3 * np.cos(0.4))
    assert phi[1] == pytest.approx(expected)


def test_draw_sample_is_deterministic_without_replacement_and_counts_cells():
    rng = np.random.default_rng(3)
    region = rng.integers(0, len(S.REGIONS), size=400).astype(np.int8)
    knot = rng.integers(-1, 2, size=400).astype(np.int8)
    region[:3] = 0
    knot[:3] = 0                                           # a cell with very few members (axis_core|plane has <= a handful)
    a, counts_a = S.draw_sample(region, knot, 6, seed=0)
    b, counts_b = S.draw_sample(region, knot, 6, seed=0)
    c, _ = S.draw_sample(region, knot, 6, seed=1)
    np.testing.assert_array_equal(a, b)
    assert counts_a == counts_b and not np.array_equal(a, c)
    assert len(np.unique(a)) == len(a) and np.all(np.diff(a) > 0)
    assert (knot[a] >= 0).all()                            # unclassified owners are never drawn
    for name, cell in counts_a.items():
        r, k = name.split("|")
        pool = np.flatnonzero((region == S.REGIONS.index(r)) & (knot == S.KNOTS.index(k)))
        assert cell["available"] == len(pool) and cell["drawn"] == min(6, len(pool))
        assert np.isin(a, pool).sum() >= cell["drawn"]
    assert sum(c["drawn"] for c in counts_a.values()) == len(a)


# ---------------------------------------------------------------------------
# metrics and aggregation
# ---------------------------------------------------------------------------
def _arrays(m=40, seed=0, error=1e-3, rmid_error=5e-2):
    rng = np.random.default_rng(seed)
    V, F, T = len(VARIANTS), len(M.FIELDS), len(M.TERMS)
    R = rng.normal(size=(V, F, T, m)) + 2.0
    N = R * (1.0 + error)
    O = R[:, :, M.TERMS.index(M.DIFFUSION)]
    Rmid = O * (1.0 + rmid_error)
    region = (np.arange(m) % len(S.REGIONS)).astype(np.int8)
    knot = ((np.arange(m) // len(S.REGIONS)) % 2).astype(np.int8)
    return {"owners": np.arange(m) * 3, "volume": rng.uniform(0.5, 2.0, size=m), "region": region, "knot": knot,
            "phi": np.zeros(m), "N": N, "R": R, "Rmid": Rmid}


def test_term_metrics_and_observed_order_match_the_step3_gates_implementation():
    sg = pytest.importorskip("p_shared.step3_gates")
    rng = np.random.default_rng(1)
    vol, ref = rng.uniform(1, 2, 30), rng.normal(size=30)
    comb = ref + 0.01 * rng.normal(size=30)
    for kwargs in ({}, {"fallback_l2": 5.0, "fallback_max": 9.0}):
        assert M.term_metrics(comb, ref, vol, **kwargs) == sg.term_metrics(comb, ref, vol, **kwargs)
    tiny = np.full(30, 1e-20)
    assert M.term_metrics(comb, tiny, vol, fallback_l2=5.0, fallback_max=9.0) == sg.term_metrics(
        comb, tiny, vol, fallback_l2=5.0, fallback_max=9.0)
    assert M.observed_order(1e-2, 2.5e-3, 32, 64) == pytest.approx(sg.observed_order(1e-2, 2.5e-3, 32, 64)) == pytest.approx(2.0)
    assert M.observed_order(None, 1.0, 32, 64) is None and M.observed_order(0.0, 1.0, 32, 64) is None


def test_subset_masks_cover_region_by_knot_and_pool():
    arrays = _arrays()
    masks = M.subset_masks(arrays["region"], arrays["knot"])
    assert len(masks) == (len(S.REGIONS) + 1) * (len(S.KNOTS) + 1)
    assert masks["all|all"].all()
    np.testing.assert_array_equal(masks["all|plane"] | masks["all|mid"], masks["all|all"])
    assert not (masks["all|plane"] & masks["all|mid"]).any()
    for r in S.REGIONS:
        np.testing.assert_array_equal(masks[f"{r}|plane"] | masks[f"{r}|mid"], masks[f"{r}|all"])


def test_compute_metrics_comparisons_and_aggregation():
    arrays = _arrays(error=1e-3, rmid_error=5e-2)
    metrics = M.compute_metrics(arrays, VARIANTS)
    pooled = metrics["all|all"]
    assert pooled["n_owners"] == 40
    entry = pooled["variants"][VARIANTS[0]]["density"]
    assert set(entry["perpendicular_diffusion"]) == {"N-O", "O-R", "N-R"}
    assert set(entry["poisson_bracket"]) == {"N-R"} and set(entry["total"]) == {"N-R"}
    vol = arrays["volume"]
    R0 = arrays["R"][0, 0]
    t = M.TERMS.index("curvature")
    expected = M.owner_weighted_l2(arrays["N"][0, 0, t] - R0[t], vol)
    assert entry["curvature"]["N-R"]["l2"] == pytest.approx(expected)
    assert entry["curvature"]["N-R"]["rel_l2"] == pytest.approx(1e-3, rel=1e-9)
    # diffusion: N-O = 1e-3 |O|, O-R = 5e-2/(1.05) relative to R, N-R in between
    diff = entry["perpendicular_diffusion"]
    assert diff["N-O"]["rel_l2"] == pytest.approx(1e-3, rel=1e-9)
    assert diff["O-R"]["rel_l2"] == pytest.approx(0.05 / 1.05, rel=1e-9)
    assert diff["N-R"]["rel_l2"] == pytest.approx(abs(1.001 - 1.05) / 1.05, rel=1e-9)
    assert diff["O-R"]["fit_scale"] == pytest.approx(1 / 1.05)
    # a region subset only uses its owners
    ordinary = metrics["ordinary|all"]
    mask = arrays["region"] == S.REGIONS.index("ordinary")
    assert ordinary["n_owners"] == int(mask.sum())
    got = ordinary["variants"][VARIANTS[1]]["Te"]["curvature"]["N-R"]["l2"]
    want = M.owner_weighted_l2((arrays["N"][1, 1, t] - arrays["R"][1, 1, t])[mask], arrays["volume"][mask])
    assert got == pytest.approx(want)
    # restricting the arrays and recomputing equals masking inside compute_metrics
    keep = arrays["owners"] < 60
    again = M.compute_metrics(M.restrict(arrays, keep), VARIANTS)
    direct = M.compute_metrics(arrays, VARIANTS, keep=keep)
    assert again == direct


def test_compute_metrics_skips_empty_subsets_and_flags_degenerate_references():
    arrays = _arrays(m=12)
    arrays["region"][:] = S.REGIONS.index("ordinary")
    metrics = M.compute_metrics(arrays, VARIANTS)
    assert "axis_core|all" not in metrics and "ordinary|all" in metrics and "all|all" in metrics
    t = M.TERMS.index("poisson_bracket")
    arrays["R"][0, 0, t] = 1e-30
    arrays["N"][0, 0, t] = 1e-30 * 2
    degenerate = M.compute_metrics(arrays, VARIANTS)["all|all"]["variants"][VARIANTS[0]]["density"]["poisson_bracket"]["N-R"]
    assert degenerate["degenerate_reference"] and degenerate["fit_scale"] is None and degenerate["rel_l2"] < 1e-20


def test_sanity_and_diffusion_convention_check():
    arrays = _arrays()
    assert M.sanity(arrays, VARIANTS) == {"finite": True, "zero_references": []}
    ok = M.diffusion_convention_check(arrays, VARIANTS)
    assert ok["ok"] and all(e["ok"] for e in ok["entries"].values())
    flipped = dict(arrays, Rmid=-arrays["Rmid"])
    bad = M.diffusion_convention_check(flipped, VARIANTS)
    assert not bad["ok"] and all(e["fit_scale"] < 0 for e in bad["entries"].values())
    broken = dict(arrays, N=arrays["N"].copy())
    broken["N"][0, 0, 0, 0] = np.nan
    broken["R"] = arrays["R"].copy()
    broken["R"][1, 2, M.TERMS.index("curvature"), :] = 0.0
    result = M.sanity(broken, VARIANTS)
    assert result["finite"] is False and result["zero_references"] == [f"{VARIANTS[1]}/Ti/curvature"]


# ---------------------------------------------------------------------------
# analysis on synthetic runs
# ---------------------------------------------------------------------------
GEOM_SHAPES = {"p05_raw_h": (3,), "p05_raw_jacobian": (), "p06_raw_J": (), "p06_raw_B": (), "p06_raw_K": (3,),
               "p07_raw_tensor": (3, 3), "p07_raw_divergence": (3,)}
FACE_SHAPES = {"p05_face_h": (4, 3), "p05_face_jacobian": (4,), "p06_face_J": (4,), "p06_face_B": (4,), "p06_face_K": (4, 3),
               "p07_face_tensor": (9, 3, 3)}


def _geometry(all_owners, keep_owners, seed, shift):
    """Closure coefficients of ``keep_owners`` (the raw cells of those owners and the faces incident to them) out of one fixed
    base set for ``all_owners`` (the same ``seed`` gives both modes identical base arrays and points)."""
    rng = np.random.default_rng(seed)
    raw_ids = np.arange(3 * len(all_owners))
    raw_owner = np.repeat(all_owners, 3)
    face_rows = np.arange(2 * len(all_owners)) + 100
    lo, hi = np.repeat(all_owners, 2), np.roll(np.repeat(all_owners, 2), 1)
    full = {"geom__raw_points": rng.normal(size=(len(raw_ids), 3)), "geom__face_points": rng.normal(size=(len(face_rows), 4, 3)),
            "geom__p07_face_points": rng.normal(size=(len(face_rows), 9, 3))}
    for name, shape in {**GEOM_SHAPES, **FACE_SHAPES}.items():
        n = len(raw_ids) if "_raw_" in name else len(face_rows)
        base = rng.uniform(1.0, 2.0, size=(n, *shape))
        full[f"geom__{name}"] = base * (1.0 + shift) if name.endswith(("_B", "_K", "h", "tensor", "divergence")) else base
    keep_raw = np.isin(raw_owner, keep_owners)
    keep_face = np.isin(lo, keep_owners) | np.isin(hi, keep_owners)
    out = {"raw_ids": raw_ids[keep_raw], "raw_owner": raw_owner[keep_raw], "face_rows": face_rows[keep_face],
           "face_owner_lo": lo[keep_face], "face_owner_hi": hi[keep_face], "p07_rows": face_rows[keep_face][:3]}
    for key, value in full.items():
        out[key] = value[keep_raw] if ("_raw_" in key or key == "geom__raw_points") else value[keep_face]
    return out


def _write_run(folder, n, mode, *, scale, shift, owners=None, seed=0):
    arrays = _arrays(m=36, seed=seed, error=scale)
    owners = arrays["owners"] if owners is None else owners
    keep = np.isin(arrays["owners"], owners)
    arrays = M.restrict(arrays, keep)
    metrics = M.compute_metrics(arrays, VARIANTS)
    geometry = _geometry(_arrays(m=36, seed=0)["owners"], arrays["owners"], 11, shift if mode == "compact_c3" else 0.0)
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "arrays.npz", **{k: v for k, v in arrays.items()}, **geometry)
    drawn = [int(o) for o in _arrays(m=36, seed=0)["owners"]]
    results = {
        "schema": "x", "n": n, "mode": mode, "variants": list(VARIANTS), "fields": list(M.FIELDS), "terms": list(M.TERMS),
        "sample": {"drawn_owners": drawn, "owners": [int(o) for o in arrays["owners"]],
                   "cells": {f"{r}|{k}": {"available": 9, "drawn": 3, "kept": 3} for r in S.REGIONS for k in S.KNOTS}},
        "dropped_owners": {"owners": sorted(set(drawn) - set(int(o) for o in arrays["owners"])), "failures": {}},
        "metrics": metrics, "sanity": M.sanity(arrays, VARIANTS),
        "diffusion_convention_check": M.diffusion_convention_check(arrays, VARIANTS),
        "receipt": {"timings_seconds": {"total": 1.0}, "peak_rss_gib": 0.5}}
    (folder / "results.json").write_text(json.dumps(results))


def test_relative_change_pointwise_and_global():
    a = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])          # tensor-like: components = the last axis
    b = a * np.array([[1.0], [1.1], [1.0]])
    out = analyze.relative_change(a, b, 1)
    assert out["n_points"] == 3
    assert out["max_rel"] == pytest.approx(0.1)
    assert out["rms_rel"] == pytest.approx(np.sqrt(0.01 / 3))
    assert out["global_rel"] == pytest.approx(np.sqrt(0.04 / 3) / np.sqrt((1 + 4 + 25) / 3))
    assert analyze.relative_change(np.zeros((0, 3)), np.zeros((0, 3)), 1)["n_points"] == 0
    with pytest.raises(ValueError):
        analyze.relative_change(a, a[:2], 1)
    scalar = analyze.relative_change(np.array([2.0, 4.0]), np.array([2.0, 4.4]), 0)
    assert scalar["max_rel"] == pytest.approx(0.1)


def test_analyze_summary_orders_ratios_flags_and_report(tmp_path):
    # N32: C3 error equal to spline's; N48: spline error drops by 1.5^2, C3 by almost nothing -> ratio flag and an order drop
    for n, s_err, c_err in ((32, 1e-2, 1e-2), (48, 1e-2 / 1.5 ** 2, 1e-2 * 0.99)):
        _write_run(tmp_path / f"N{n}_spline", n, "spline", scale=s_err, shift=0.0)
        _write_run(tmp_path / f"N{n}_compact_c3", n, "compact_c3", scale=c_err, shift=2e-6)
    summary = analyze.build_summary(tmp_path)
    assert summary["grids"] == [32, 48] and summary["owner_sets"][32]["equal"]
    e = summary["series"][VARIANTS[0]]["density"]["curvature/N-R"]["all|all"]
    assert e["ratio"]["rel_l2"][32] == pytest.approx(1.0, rel=1e-9)
    assert e["ratio"]["rel_l2"][48] == pytest.approx(0.99 * 1.5 ** 2, rel=1e-9)
    assert e["order"]["spline"]["rel_l2"][0] == pytest.approx(2.0, rel=1e-9)
    assert e["order"]["compact_c3"]["rel_l2"][0] == pytest.approx(np.log(1 / 0.99) / np.log(1.5), rel=1e-6)
    kinds = {f["kind"] for f in summary["flags"]}
    assert kinds == {"ratio", "order_drop"}
    assert all(f["ratio"] > 1.5 and f["n"] == 48 for f in summary["flags"] if f["kind"] == "ratio")
    drop = next(f for f in summary["flags"] if f["kind"] == "order_drop")
    assert drop["grids"] == [32, 48] and drop["order_compact_c3"] < drop["order_spline"] - 0.5
    # coefficients: the shifted arrays change by 2e-6, the controls (jacobian, J) by exactly 0
    coeff = summary["coefficients"][32]
    assert coeff["p06_raw_B"]["all|all"]["rms_rel"] == pytest.approx(2e-6, rel=1e-6)
    assert coeff["p07_face_tensor"]["ordinary|plane"]["max_rel"] == pytest.approx(2e-6, rel=1e-6)
    assert coeff["p05_raw_jacobian"]["all|all"]["max_rel"] == 0.0 and coeff["p06_face_J"]["all|all"]["max_rel"] == 0.0
    report, summary_path = analyze.write_outputs(tmp_path, summary)
    text = report.read_text()
    assert "## Flags" in text and "perpendicular_diffusion/N-O" in text and "all|all" in text
    assert json.loads(summary_path.read_text())["grids"] == [32, 48]


def test_analyze_uses_the_common_owners_when_a_mode_dropped_some(tmp_path):
    full = _arrays(m=36, seed=0)["owners"]
    _write_run(tmp_path / "N32_spline", 32, "spline", scale=1e-3, shift=0.0)
    _write_run(tmp_path / "N32_compact_c3", 32, "compact_c3", scale=1e-3, shift=1e-6, owners=full[3:])
    summary = analyze.build_summary(tmp_path)
    assert summary["owner_sets"][32] == {"drawn": 36, "common": 33, "dropped": {"spline": [], "compact_c3": [int(o) for o in full[:3]]},
                                         "equal": False}
    counts = [summary["series"][VARIANTS[0]]["density"]["curvature/N-R"][s]["n_owners"][32] for s in ("all|all",)]
    assert counts == [33]
    # the recomputed spline metrics equal a direct computation on the common owners
    spline_arrays = analyze.load_run(tmp_path / "N32_spline")[1]
    direct = M.compute_metrics(M.restrict(spline_arrays, np.isin(spline_arrays["owners"], full[3:])), VARIANTS)
    got = summary["series"][VARIANTS[0]]["density"]["curvature/N-R"]["all|all"]["spline"]["l2"][32]
    assert got == pytest.approx(direct["all|all"]["variants"][VARIANTS[0]]["density"]["curvature"]["N-R"]["l2"])


def test_analyze_refuses_different_samples_or_points(tmp_path):
    _write_run(tmp_path / "N32_spline", 32, "spline", scale=1e-3, shift=0.0)
    _write_run(tmp_path / "N32_compact_c3", 32, "compact_c3", scale=1e-3, shift=1e-6)
    path = tmp_path / "N32_compact_c3" / "arrays.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    data["geom__raw_points"] = data["geom__raw_points"] + 1e-9
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="identical points"):
        analyze.build_summary(tmp_path)
    results = json.loads((tmp_path / "N32_compact_c3" / "results.json").read_text())
    results["sample"]["drawn_owners"] = results["sample"]["drawn_owners"][:-1]
    (tmp_path / "N32_compact_c3" / "results.json").write_text(json.dumps(results))
    with pytest.raises(ValueError, match="different owner samples"):
        analyze.build_summary(tmp_path)
    with pytest.raises(ValueError, match="no grid with both modes"):
        analyze.build_summary(tmp_path / "missing")


def test_references_module_and_build_setup_option_exist():
    from p08_bfield_eval import references
    assert references.FIELD_SLOTS == {"density": 0, "Te": 1, "Ti": 2, "vorticity": 3}
    sg = pytest.importorskip("p_shared.step3_gates")
    assert inspect.signature(sg.build_setup).parameters["owners"].default is None


# ---------------------------------------------------------------------------
# real smoke (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_real_smoke_n32_spline_one_owner_per_cell(tmp_path):
    pytest.importorskip("jax")
    from p_shared.replay_support import DEFAULT_SIDECAR
    needed = [Path(DEFAULT_SIDECAR), WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"]
    if not all(p.exists() for p in needed):
        pytest.skip("HSX inputs not available")
    from p08_bfield_eval import run
    from tests import p06n_owner_values_live as live
    with live.patched():                      # the frozen P06N owner_values file was removed: computed on the fly
        results = run.run(32, "spline", tmp_path, per_cell=1)
    folder = tmp_path / "N32_spline"
    assert (folder / "results.json").is_file() and (folder / "arrays.npz").is_file()
    assert results["sample"]["n_owners"] + len(results["dropped_owners"]["owners"]) == results["sample"]["n_owners_drawn"] == 12
    assert results["receipt"]["provenance_bfield_toroidal"] == "spline"
    assert results["sanity"] == {"finite": True, "zero_references": []}
    with np.load(folder / "arrays.npz") as z:
        assert z["N"].shape == (2, 4, 4, results["sample"]["n_owners"]) and np.isfinite(z["N"]).all()
        assert z["geom__p06_raw_B"].shape[0] == len(z["raw_ids"])
    pooled = results["metrics"]["all|all"]["variants"]["main_phi_dirichlet"]
    for field in M.FIELDS:
        for term, comparison in M.COMPARISONS:
            entry = pooled[field][term][comparison]
            assert entry["finite"] and entry["rel_l2"] is not None and entry["rel_l2"] < 1.0
    # O and R carry the same sign / coefficient convention
    for entry in results["diffusion_convention_check"]["entries"].values():
        assert 0.5 < entry["fit_scale"] < 1.5
