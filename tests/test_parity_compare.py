from __future__ import annotations

import pytest

from jax_drb.parity.compare import compare_summary_payloads


def _summary_payload() -> dict[str, object]:
    return {
        "case_name": "toy",
        "parity_mode": "one_rhs",
        "compare_variables": ["Ne"],
        "component_labels": ["e:evolve_density"],
        "dimensions": {"t": 1, "x": 3},
        "time_points": [0.0],
        "dataset_scalars": {"Nnorm": 1.0},
        "variable_summaries": {
            "Ne": {
                "name": "Ne",
                "dimensions": ["t", "x"],
                "shape": [1, 3],
                "minimum": 1.0,
                "maximum": 3.0,
                "mean": 2.0,
                "max_abs_delta_last_first": None,
            }
        },
    }


def test_compare_summary_payloads_accepts_matching_payloads() -> None:
    payload = _summary_payload()

    result = compare_summary_payloads(payload, payload)
    assert result.ok is True
    assert result.issues == ()


def test_compare_summary_payloads_reports_numeric_mismatches() -> None:
    expected = _summary_payload()
    actual = {
        **expected,
        "dataset_scalars": {"Nnorm": 2.0},
    }

    result = compare_summary_payloads(expected, actual)
    assert result.ok is False
    assert result.issues[0].field == "dataset_scalars.Nnorm"


def test_compare_summary_payloads_accepts_tuple_list_metadata_equivalence() -> None:
    expected = _summary_payload()
    actual = {
        **expected,
        "compare_variables": ("Ne",),
        "component_labels": ("e:evolve_density",),
        "variable_summaries": {
            "Ne": {
                **expected["variable_summaries"]["Ne"],
                "dimensions": ("t", "x"),
                "shape": (1, 3),
            }
        },
    }

    result = compare_summary_payloads(expected, actual)
    assert result.ok is True
    assert result.issues == ()


def test_compare_summary_payloads_tolerates_missing_capability_tier_in_older_baselines() -> None:
    expected = _summary_payload()
    actual = {
        **expected,
        "capability_tier": "native_exact",
    }

    result = compare_summary_payloads(expected, actual)
    assert result.ok is True
    assert result.issues == ()


def test_compare_summary_payloads_reports_metadata_and_capability_mismatches() -> None:
    expected = {
        **_summary_payload(),
        "capability_tier": "native_exact",
    }
    actual = {
        **expected,
        "case_name": "changed",
        "parity_mode": "short_window",
        "compare_variables": ["Pe"],
        "component_labels": ["e:evolve_pressure"],
        "dimensions": {"t": 2, "x": 3},
        "capability_tier": "native_operational",
    }

    result = compare_summary_payloads(expected, actual)
    fields = tuple(issue.field for issue in result.issues)

    assert result.ok is False
    assert "case_name" in fields
    assert "parity_mode" in fields
    assert "compare_variables" in fields
    assert "component_labels" in fields
    assert "dimensions" in fields
    assert "capability_tier" in fields


def test_compare_summary_payloads_reports_time_length_and_value_mismatches() -> None:
    expected = _summary_payload()

    length_mismatch = compare_summary_payloads(
        expected,
        {**expected, "time_points": [0.0, 1.0]},
    )
    value_mismatch = compare_summary_payloads(
        expected,
        {**expected, "time_points": [1.0e-3]},
        time_atol=1.0e-12,
    )

    assert length_mismatch.ok is False
    assert length_mismatch.issues[0].field == "time_points"
    assert value_mismatch.ok is False
    assert value_mismatch.issues[0].field == "time_points[0]"


def test_compare_summary_payloads_reports_missing_and_unexpected_dataset_scalars() -> None:
    expected = {
        **_summary_payload(),
        "dataset_scalars": {"Nnorm": 1.0, "Tnorm": 2.0},
    }
    actual = {
        **expected,
        "dataset_scalars": {"Nnorm": 1.0, "Bnorm": 3.0},
    }

    result = compare_summary_payloads(expected, actual)
    fields = tuple(issue.field for issue in result.issues)

    assert "dataset_scalars.Tnorm" in fields
    assert "dataset_scalars.Bnorm" in fields


def test_compare_summary_payloads_reports_variable_summary_structure_mismatches() -> None:
    expected = _summary_payload()
    actual = {
        **expected,
        "variable_summaries": {
            "Ne": {
                **expected["variable_summaries"]["Ne"],
                "name": "Pe",
                "dimensions": ["x", "t"],
                "shape": [3, 1],
            }
        },
    }

    result = compare_summary_payloads(expected, actual)
    fields = tuple(issue.field for issue in result.issues)

    assert "variable_summaries.Ne.name" in fields
    assert "variable_summaries.Ne.dimensions" in fields
    assert "variable_summaries.Ne.shape" in fields


@pytest.mark.parametrize("field", ["minimum", "maximum", "mean"])
def test_compare_summary_payloads_reports_variable_summary_numeric_mismatches(field: str) -> None:
    expected = _summary_payload()
    actual_summary = {
        **expected["variable_summaries"]["Ne"],
        field: expected["variable_summaries"]["Ne"][field] + 0.5,
    }
    actual = {
        **expected,
        "variable_summaries": {"Ne": actual_summary},
    }

    result = compare_summary_payloads(expected, actual)

    assert result.ok is False
    assert result.issues[0].field == f"variable_summaries.Ne.{field}"


def test_compare_summary_payloads_reports_delta_missing_and_numeric_mismatches() -> None:
    expected = _summary_payload()
    expected_with_delta = {
        **expected,
        "variable_summaries": {
            "Ne": {
                **expected["variable_summaries"]["Ne"],
                "max_abs_delta_last_first": 1.0,
            }
        },
    }

    missing_result = compare_summary_payloads(
        expected,
        {
            **expected,
            "variable_summaries": {
                "Ne": {
                    **expected["variable_summaries"]["Ne"],
                    "max_abs_delta_last_first": 1.0,
                }
            },
        },
    )
    numeric_result = compare_summary_payloads(
        expected_with_delta,
        {
            **expected_with_delta,
            "variable_summaries": {
                "Ne": {
                    **expected_with_delta["variable_summaries"]["Ne"],
                    "max_abs_delta_last_first": 1.5,
                }
            },
        },
    )

    assert missing_result.ok is False
    assert missing_result.issues[0].field == "variable_summaries.Ne.max_abs_delta_last_first"
    assert numeric_result.ok is False
    assert numeric_result.issues[0].field == "variable_summaries.Ne.max_abs_delta_last_first"


def test_compare_summary_payloads_reports_missing_and_unexpected_variables() -> None:
    expected = _summary_payload()
    actual = {
        **expected,
        "variable_summaries": {
            "Pe": {
                "name": "Pe",
                "dimensions": ["t", "x"],
                "shape": [1, 3],
                "minimum": 1.0,
                "maximum": 3.0,
                "mean": 2.0,
                "max_abs_delta_last_first": None,
            }
        },
    }

    result = compare_summary_payloads(expected, actual)
    fields = tuple(issue.field for issue in result.issues)

    assert "variable_summaries.Ne" in fields
    assert "variable_summaries.Pe" in fields
