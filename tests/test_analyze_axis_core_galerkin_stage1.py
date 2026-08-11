"""Synthetic tests for the root-level Stage-1 snapshot comparison utility."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import analyze_axis_core_galerkin_stage1 as stage1  # noqa: E402


def _write(path: Path, scale: float, requested: float, actual: float, nonfinite: bool = False) -> None:
    shape = (7, 8, 3)
    theta = np.arange(shape[1])[None, :, None]
    eta = np.arange(shape[2])[None, None, :]
    base = np.ones(shape) * scale + 0.25 * np.cos(2 * np.pi * 4 * theta / shape[1]) * (np.arange(shape[0])[:, None, None] + 1)
    if nonfinite:
        base[5, 1, 0] = np.nan
    payload = {name: base.copy() for name in stage1.FIELDS}
    payload.update(time=np.asarray(actual), requested_snapshot_time=np.asarray(requested), step=np.asarray(1))
    payload["Ve_rhs_terms"] = np.stack([base, 2 * base])
    payload["run_metadata_json"] = np.asarray(json.dumps({"ve_term_names": ["first", "second"]}))
    np.savez(path, **payload)


def test_metrics_pair_terms_and_boundary_rings(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    _write(baseline / "b.snapshot_t1d000000000000em02.npz", 1.0, 1e-2, 1.0e-2)
    _write(candidate / "c.snapshot_t1d000000000000em02.npz", 2.0, 1e-2, 1.1e-2)
    report = stage1.compare(baseline, candidate, cutoff=4, boundary_rings=(3, 6))
    assert report["matched_count"] == 1
    item = report["snapshots"][0]
    assert item["candidate_over_baseline"]["fields"]["Ve"]["max_abs"] > 1.0
    assert item["baseline"]["fields"]["Ve"]["boundary_rings"]["6"] is not None
    assert set(item["candidate"]["Ve_rhs_terms"]) == {"first", "second"}
    assert item["candidate"]["fields"]["Ve"]["largest_adjacent_high_mode_jump"]["between_rings"] is not None


def test_missing_later_snapshot_and_first_nonfinite(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    _write(baseline / "b.snapshot_t1d000000000000em02.npz", 1.0, 1e-2, 1e-2)
    _write(baseline / "b.snapshot_t2d000000000000em02.npz", 1.0, 2e-2, 2e-2, nonfinite=True)
    _write(candidate / "c.snapshot_t1d000000000000em02.npz", 1.0, 1e-2, 1e-2)
    report = stage1.compare(baseline, candidate)
    assert report["matched_count"] == 1
    assert report["unmatched_baseline"]
    assert report["first_nonfinite"]["baseline"]["requested_time"] == 2e-2
    assert report["snapshots"][0]["baseline"]["fields"]["Ve"]["nonfinite_count"] == 0
