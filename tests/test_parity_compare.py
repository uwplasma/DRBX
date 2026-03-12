from __future__ import annotations

from jax_drb.parity.compare import compare_summary_payloads


def test_compare_summary_payloads_accepts_matching_payloads() -> None:
    payload = {
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

    result = compare_summary_payloads(payload, payload)
    assert result.ok is True
    assert result.issues == ()


def test_compare_summary_payloads_reports_numeric_mismatches() -> None:
    expected = {
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
    actual = {
        **expected,
        "dataset_scalars": {"Nnorm": 2.0},
    }

    result = compare_summary_payloads(expected, actual)
    assert result.ok is False
    assert result.issues[0].field == "dataset_scalars.Nnorm"


def test_compare_summary_payloads_accepts_tuple_list_metadata_equivalence() -> None:
    expected = {
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
