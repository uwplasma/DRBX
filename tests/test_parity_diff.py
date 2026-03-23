from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jax_drb.cli import main
from jax_drb.parity.compare import load_summary_json
from jax_drb.parity.diff import build_array_diff_report, compare_recycling_artifacts, format_array_diff_report, format_recycling_diff_report


_BASELINE_DIR = Path("/Users/rogerio/local/jax_drb/references/baselines/reference")


def test_array_diff_report_tracks_max_abs_diff_and_location() -> None:
    expected = {
        "field": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "scalar": np.array(1.0, dtype=np.float64),
    }
    actual = {
        "field": np.array([[1.0, 2.0], [3.5, 4.0]], dtype=np.float64),
        "scalar": np.array(1.25, dtype=np.float64),
    }

    report = build_array_diff_report(expected, actual, compare_variables=("field", "scalar"))

    assert report.compared_fields == ("field", "scalar")
    assert report.max_abs_diff == 0.5
    assert len(report.entries) == 2

    field_entry = next(entry for entry in report.entries if entry.field == "field")
    assert field_entry.max_abs_location == (1, 0)
    assert field_entry.expected_value == 3.0
    assert field_entry.actual_value == 3.5

    scalar_entry = next(entry for entry in report.entries if entry.field == "scalar")
    assert scalar_entry.max_abs_location == ()
    assert scalar_entry.max_abs_diff == 0.25


def test_array_diff_report_formats_locations() -> None:
    report = build_array_diff_report({"a": np.array([1.0, 2.0])}, {"a": np.array([1.0, 3.0])})
    text = format_array_diff_report(report)

    assert "a: max_abs_diff=1.00000000e+00" in text
    assert "@(1,)" in text


def test_recycling_summary_diff_report_localizes_worst_variable(tmp_path: Path) -> None:
    expected_path = _BASELINE_DIR / "recycling_1d_rhs.json"
    actual_path = tmp_path / "recycling_1d_rhs.json"
    payload = load_summary_json(expected_path)
    payload["variable_summaries"]["Nd"]["mean"] += 5.0e-2
    actual_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    report = compare_recycling_artifacts(expected_path, actual_path, artifact_kind="summary")

    assert not report.ok
    assert report.worst_field == "variable_summaries.Nd.mean"
    assert report.worst_variable == "Nd"
    assert report.worst_location is None

    text = format_recycling_diff_report(report)
    assert "summary: mismatch" in text
    assert "worst_variable: Nd" in text
    assert "variable_summaries.Nd.mean" in text


def test_compare_recycling_cli_reports_summary_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    expected_path = _BASELINE_DIR / "recycling_1d_rhs.json"
    actual_path = tmp_path / "recycling_1d_rhs.json"
    payload = load_summary_json(expected_path)
    payload["variable_summaries"]["Nd"]["mean"] += 5.0e-2
    actual_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    exit_code = main(["compare-recycling", str(expected_path), str(actual_path)])

    assert exit_code == 1
    text = capsys.readouterr().out
    assert "summary: mismatch" in text
    assert "worst_variable: Nd" in text
