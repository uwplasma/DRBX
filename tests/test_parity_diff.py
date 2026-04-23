from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jax_drb.cli import main
from jax_drb.parity.arrays import build_portable_array_payload, write_portable_array_payload
from jax_drb.parity.compare import load_summary_json
from jax_drb.parity.diff import (
    build_array_time_trace,
    build_array_diff_report,
    build_scaled_array_diff_entries,
    compare_recycling_artifacts,
    filter_scaled_array_diff_entries_to_band,
    format_array_diff_report,
    format_recycling_diff_report,
)


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


def test_array_diff_report_tracks_missing_shape_and_empty_fields() -> None:
    expected = {
        "missing_actual": np.asarray([1.0], dtype=np.float64),
        "shape_mismatch": np.asarray([1.0, 2.0], dtype=np.float64),
        "empty": np.asarray([], dtype=np.float64),
    }
    actual = {
        "missing_expected": np.asarray([1.0], dtype=np.float64),
        "shape_mismatch": np.asarray([[1.0, 2.0]], dtype=np.float64),
        "empty": np.asarray([], dtype=np.float64),
    }

    report = build_array_diff_report(
        expected,
        actual,
        compare_variables=("missing_expected", "missing_actual", "shape_mismatch", "empty"),
    )

    assert report.ok is False
    assert report.missing_expected_fields == ("missing_expected",)
    assert report.missing_actual_fields == ("missing_actual",)

    shape_entry = next(entry for entry in report.entries if entry.field == "shape_mismatch")
    assert shape_entry.max_abs_diff == float("inf")
    assert shape_entry.max_abs_location == ()
    assert np.isnan(shape_entry.expected_value)
    assert np.isnan(shape_entry.actual_value)

    empty_entry = next(entry for entry in report.entries if entry.field == "empty")
    assert empty_entry.max_abs_diff == 0.0
    assert empty_entry.expected_value == 0.0
    assert empty_entry.actual_value == 0.0

    text = format_array_diff_report(report)
    assert "missing expected fields: missing_expected" in text
    assert "missing actual fields: missing_actual" in text


def test_array_diff_report_ok_and_empty_format_for_identical_empty_inputs() -> None:
    report = build_array_diff_report({}, {})

    assert report.ok is True
    assert report.max_abs_diff == 0.0
    assert format_array_diff_report(report) == "comparison: ok"


def test_array_diff_report_formats_locations() -> None:
    report = build_array_diff_report({"a": np.array([1.0, 2.0])}, {"a": np.array([1.0, 3.0])})
    text = format_array_diff_report(report)

    assert "a: max_abs_diff=1.00000000e+00" in text
    assert "@(1,)" in text


def test_scaled_array_diff_entries_classify_small_denominator_fields() -> None:
    expected = {
        "large": np.array([100.0, 50.0], dtype=np.float64),
        "tiny": np.array([0.0, 1.0e-14], dtype=np.float64),
    }
    actual = {
        "large": np.array([100.2, 50.0], dtype=np.float64),
        "tiny": np.array([0.0, 2.0e-14], dtype=np.float64),
    }

    entries = build_scaled_array_diff_entries(expected, actual, compare_variables=("large", "tiny"), near_zero_atol=1.0e-12)

    large = next(entry for entry in entries if entry.field == "large")
    assert large.max_abs_diff == pytest.approx(2.0e-1)
    assert large.expected_abs_max == pytest.approx(1.0e2)
    assert large.relative_to_expected_max == pytest.approx(2.0e-3)
    assert large.near_zero_expected is False

    tiny = next(entry for entry in entries if entry.field == "tiny")
    assert tiny.max_abs_diff == pytest.approx(1.0e-14)
    assert tiny.expected_abs_max == pytest.approx(1.0e-14)
    assert tiny.relative_to_expected_max is None
    assert tiny.near_zero_expected is True


def test_scaled_array_diff_entries_handle_empty_and_shape_mismatched_arrays() -> None:
    entries = build_scaled_array_diff_entries(
        {"empty": np.asarray([], dtype=np.float64), "bad": np.asarray([1.0, 2.0])},
        {"empty": np.asarray([], dtype=np.float64), "bad": np.asarray([[1.0, 2.0]])},
        compare_variables=("empty", "bad"),
    )

    empty = next(entry for entry in entries if entry.field == "empty")
    assert empty.expected_abs_max == 0.0
    assert empty.actual_abs_max == 0.0
    assert empty.near_zero_expected is True
    assert empty.relative_to_expected_max is None

    bad = next(entry for entry in entries if entry.field == "bad")
    assert bad.max_abs_diff == float("inf")
    assert bad.relative_to_expected_max == float("inf")


def test_filter_scaled_array_diff_entries_to_band_keeps_y_edge_cells() -> None:
    expected = {
        "lower": np.zeros((2, 3, 4, 1), dtype=np.float64),
        "middle": np.zeros((2, 3, 4, 1), dtype=np.float64),
        "upper": np.zeros((2, 3, 4, 1), dtype=np.float64),
    }
    actual = {
        "lower": np.zeros((2, 3, 4, 1), dtype=np.float64),
        "middle": np.zeros((2, 3, 4, 1), dtype=np.float64),
        "upper": np.zeros((2, 3, 4, 1), dtype=np.float64),
    }
    actual["lower"][1, 0, 0, 0] = 1.0
    actual["middle"][1, 0, 1, 0] = 2.0
    actual["upper"][1, 0, 3, 0] = 3.0

    entries = build_scaled_array_diff_entries(expected, actual, compare_variables=("lower", "middle", "upper"))
    filtered = filter_scaled_array_diff_entries_to_band(entries, axis=2)

    assert tuple(entry.field for entry in filtered) == ("lower", "upper")


def test_filter_scaled_array_diff_entries_ignores_invalid_axes_and_scalar_locations() -> None:
    entries = build_scaled_array_diff_entries(
        {
            "scalar": np.asarray(1.0, dtype=np.float64),
            "field": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        },
        {
            "scalar": np.asarray(2.0, dtype=np.float64),
            "field": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        },
        compare_variables=("scalar", "field"),
    )

    assert filter_scaled_array_diff_entries_to_band(entries, axis=-1) == ()
    assert tuple(entry.field for entry in filter_scaled_array_diff_entries_to_band(entries, axis=0)) == ("field",)


def test_build_array_time_trace_tracks_spatial_cell_over_time() -> None:
    expected = {
        "field": np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.5, 2.5], [3.5, 4.5]],
                [[2.0, 3.0], [4.0, 5.0]],
            ],
            dtype=np.float64,
        )
    }
    actual = {
        "field": np.array(
            [
                [[1.0, 2.2], [3.0, 4.0]],
                [[1.5, 2.7], [3.5, 4.0]],
                [[2.0, 2.8], [4.0, 5.2]],
            ],
            dtype=np.float64,
        )
    }

    trace = build_array_time_trace(expected, actual, field="field", spatial_location=(0, 1))

    assert trace.field == "field"
    assert trace.spatial_location == (0, 1)
    assert trace.expected_series == pytest.approx((2.0, 2.5, 3.0))
    assert trace.actual_series == pytest.approx((2.2, 2.7, 2.8))
    assert trace.abs_diff_series == pytest.approx((0.2, 0.2, 0.2))


def test_build_array_time_trace_handles_scalars_and_rejects_bad_indices() -> None:
    scalar_trace = build_array_time_trace(
        {"field": np.asarray(1.0, dtype=np.float64)},
        {"field": np.asarray(1.25, dtype=np.float64)},
        field="field",
        spatial_location=(),
    )
    assert scalar_trace.spatial_location == ()
    assert scalar_trace.expected_series == (1.0,)
    assert scalar_trace.actual_series == (1.25,)
    assert scalar_trace.abs_diff_series == pytest.approx((0.25,))

    with pytest.raises(ValueError, match="shape mismatch"):
        build_array_time_trace(
            {"field": np.zeros((2, 2), dtype=np.float64)},
            {"field": np.zeros((2, 3), dtype=np.float64)},
            field="field",
            spatial_location=(0,),
        )
    with pytest.raises(ValueError, match="time_axis"):
        build_array_time_trace(
            {"field": np.zeros((2, 2), dtype=np.float64)},
            {"field": np.zeros((2, 2), dtype=np.float64)},
            field="field",
            spatial_location=(0,),
            time_axis=2,
        )
    with pytest.raises(ValueError, match="spatial_location"):
        build_array_time_trace(
            {"field": np.zeros((2, 2), dtype=np.float64)},
            {"field": np.zeros((2, 2), dtype=np.float64)},
            field="field",
            spatial_location=(0, 1),
        )
    with pytest.raises(ValueError, match="spatial index"):
        build_array_time_trace(
            {"field": np.zeros((2, 2), dtype=np.float64)},
            {"field": np.zeros((2, 2), dtype=np.float64)},
            field="field",
            spatial_location=(3,),
        )


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


def test_recycling_array_diff_report_localizes_worst_field_and_location(tmp_path: Path) -> None:
    expected_path = tmp_path / "expected.npz"
    actual_path = tmp_path / "actual.npz"
    expected_payload = build_portable_array_payload(
        case_name="toy",
        parity_mode="one_step",
        capability_tier="native_exact",
        compare_variables=("Ne", "Pe"),
        component_labels=("e:evolve_density",),
        dimensions={"t": 1, "x": 2},
        time_points=(0.0,),
        dataset_scalars={"Nnorm": 1.0},
        variables={
            "Ne": np.asarray([[1.0, 2.0]], dtype=np.float64),
            "Pe": np.asarray([[3.0, 4.0]], dtype=np.float64),
        },
    )
    actual_payload = build_portable_array_payload(
        case_name="toy",
        parity_mode="one_step",
        capability_tier="native_exact",
        compare_variables=("Ne", "Pe"),
        component_labels=("e:evolve_density",),
        dimensions={"t": 1, "x": 2},
        time_points=(0.0,),
        dataset_scalars={"Nnorm": 1.0},
        variables={
            "Ne": np.asarray([[1.0, 2.0]], dtype=np.float64),
            "Pe": np.asarray([[3.0, 5.5]], dtype=np.float64),
        },
    )
    write_portable_array_payload(expected_payload, expected_path)
    write_portable_array_payload(actual_payload, actual_path)

    report = compare_recycling_artifacts(expected_path, actual_path)

    assert report.ok is False
    assert report.artifact_kind == "arrays"
    assert report.worst_field == "Pe"
    assert report.worst_variable == "Pe"
    assert report.worst_location == (0, 1)
    assert report.max_abs_diff == pytest.approx(1.5)

    text = format_recycling_diff_report(report)
    assert "metadata: mismatch" in text
    assert "arrays:" in text
    assert "worst: Pe @ (0, 1)" in text


def test_recycling_array_diff_report_marks_metadata_only_mismatch(tmp_path: Path) -> None:
    expected_path = tmp_path / "expected.npz"
    actual_path = tmp_path / "actual.npz"
    expected_payload = build_portable_array_payload(
        case_name="toy",
        parity_mode="one_step",
        capability_tier="native_exact",
        compare_variables=("Ne",),
        component_labels=("e:evolve_density",),
        dimensions={"t": 1, "x": 2},
        time_points=(0.0,),
        dataset_scalars={"Nnorm": 1.0},
        variables={"Ne": np.asarray([[1.0, 2.0]], dtype=np.float64)},
    )
    actual_payload = {
        **expected_payload,
        "case_name": "toy_changed",
    }
    write_portable_array_payload(expected_payload, expected_path)
    write_portable_array_payload(actual_payload, actual_path)

    report = compare_recycling_artifacts(expected_path, actual_path)

    assert report.ok is False
    assert report.worst_field == "case_name"
    assert report.worst_variable is None
    assert report.worst_location is None


def test_recycling_artifact_kind_resolution_rejects_unsupported_or_mixed_inputs(tmp_path: Path) -> None:
    expected_json = tmp_path / "expected.json"
    actual_npz = tmp_path / "actual.npz"
    expected_json.write_text("{}", encoding="utf-8")
    np.savez_compressed(actual_npz, __metadata__="{}")

    with pytest.raises(ValueError, match="unsupported artifact_kind"):
        compare_recycling_artifacts(expected_json, expected_json, artifact_kind="bad")
    with pytest.raises(ValueError, match="cannot infer recycling artifact kind"):
        compare_recycling_artifacts(expected_json, actual_npz)


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
