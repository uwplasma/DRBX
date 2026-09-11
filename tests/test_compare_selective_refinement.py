"""Focused tests for common-final-time selective refinement comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "work" / "boundary_load_audit"))
from compare_selective_refinement import compare_selective_refinement  # noqa: E402


FIELDS = ("density", "Te", "Ti", "Vi", "Ve", "vorticity")


def _write_pair(tmp_path: Path, *, initial_shift: float = 0.0):
    initial = np.full((6, 2), initial_shift)
    initial_phi = np.full(2, initial_shift)
    initial_gauge = np.asarray([initial_shift])
    arrays = {
        "initial_material": initial,
        "initial_phi_active": initial_phi,
        "initial_gauge": initial_gauge,
    }
    for prefix, value in (("minimal", 1.1), ("full", 1.0)):
        arrays[f"{prefix}_final_material"] = np.full((6, 2), value)
        arrays[f"{prefix}_final_phi_active"] = np.full(2, value)
        arrays[f"{prefix}_final_gauge"] = np.asarray([value])
    coarse_npz = tmp_path / "coarse.npz"
    np.savez(coarse_npz, **arrays)
    refined_arrays = dict(arrays)
    refined_arrays["minimal_final_material"] = np.full((6, 2), 1.3)
    refined_arrays["minimal_final_phi_active"] = np.full(2, 1.3)
    refined_arrays["minimal_final_gauge"] = np.asarray([1.3])
    refined_arrays["full_final_material"] = np.full((6, 2), 1.2)
    refined_arrays["full_final_phi_active"] = np.full(2, 1.2)
    refined_arrays["full_final_gauge"] = np.asarray([1.2])
    refined_npz = tmp_path / "refined.npz"
    np.savez(refined_npz, **refined_arrays)
    metadata = {"status": "completed", "dt": 0.1, "steps": 2, "cases": {"minimal": {}, "full": {}}}
    refined_metadata = {"status": "completed", "dt": 0.05, "steps": 4, "cases": {"minimal": {}, "full": {}}}
    coarse_json = tmp_path / "coarse.json"
    refined_json = tmp_path / "refined.json"
    coarse_json.write_text(json.dumps(metadata))
    refined_json.write_text(json.dumps(refined_metadata))
    return coarse_json, coarse_npz, refined_json, refined_npz


def test_common_final_time_comparison_reports_partition_temporal_and_ratios(tmp_path):
    paths = _write_pair(tmp_path)
    result = compare_selective_refinement(*paths)
    assert all(result["validation"].values())
    assert result["coarse_final_time"] == result["refined_final_time"] == 0.2
    assert result["partition_error"]["coarse"]["fields"]["density"]["increment_relative"] == pytest.approx(0.1)
    assert result["partition_error"]["refined"]["fields"]["density"]["increment_relative"] == pytest.approx(1.0 / 12.0)
    assert result["temporal_difference"]["full_coarse_vs_refined"]["fields"]["density"]["increment_relative"] == pytest.approx(1.0 / 6.0)
    assert result["partition_over_full_temporal"]["coarse"]["aggregate"]["partition_over_full_temporal"] == pytest.approx(0.5)
    assert result["max_values"]["full_temporal"]["max_increment_relative"] == pytest.approx(1.0 / 6.0)


def test_zero_increment_and_validation_flags_are_explicit(tmp_path):
    paths = _write_pair(tmp_path)
    coarse_json, coarse_npz, refined_json, refined_npz = paths
    with np.load(coarse_npz) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    arrays["minimal_final_material"] = arrays["full_final_material"].copy()
    arrays["minimal_final_phi_active"] = arrays["full_final_phi_active"].copy()
    arrays["minimal_final_gauge"] = arrays["full_final_gauge"].copy()
    np.savez(coarse_npz, **arrays)
    result = compare_selective_refinement(*paths)
    assert result["partition_error"]["coarse"]["fields"]["density"]["absolute_l2"] == 0.0

    refined_json.write_text(json.dumps({"status": "failed", "dt": 0.05, "steps": 4, "cases": {"minimal": {}, "full": {}}}))
    with pytest.raises(ValueError, match="status completed"):
        compare_selective_refinement(*paths)
