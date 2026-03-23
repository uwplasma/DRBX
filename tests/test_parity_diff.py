from __future__ import annotations

import numpy as np

from jax_drb.parity.diff import build_array_diff_report, format_array_diff_report


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
