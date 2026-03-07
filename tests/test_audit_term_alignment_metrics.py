from __future__ import annotations

from tools.audit_term_alignment import _compute_term_mismatch, _first_failing_term


def test_compute_term_mismatch_adds_array_metrics() -> None:
    jax_rows = [
        {
            "step": 0,
            "t": 0.01,
            "field": "Pe",
            "term": "advection",
            "rms": 1.0,
            "maxabs": 1.0,
        }
    ]
    hermes_rows = [
        {
            "step": 0,
            "t": 0.01,
            "field": "Pe",
            "term": "exb",
            "rms": 1.0,
            "maxabs": 1.0,
        }
    ]
    jax_arrays = {(0, "Pe", "advection"): [1.0, -1.0, 1.0, -1.0]}
    hermes_arrays = {(0, "Pe", "exb"): [1.0, 1.0, -1.0, -1.0]}

    rows = _compute_term_mismatch(
        jax_rows,
        hermes_rows,
        jax_term_fields=jax_arrays,
        hermes_term_fields=hermes_arrays,
        strict_axis=True,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["rel_diff"] == 0.0
    assert row["array_diff_rms"] > 0.0
    assert row["array_rel_diff"] > 0.0
    assert row["array_diff_maxabs"] == 2.0
    assert row["array_corr"] == 0.0


def test_first_failing_term_defaults_to_array_metric() -> None:
    rows = [
        {
            "step": 0,
            "field": "Pe",
            "term": "advection",
            "rel_diff": 0.4,
            "weighted_rel": 0.4,
            "array_rel_diff": 0.1,
            "weighted_array_rel": 0.1,
        },
        {
            "step": 0,
            "field": "Pe",
            "term": "parallel",
            "rel_diff": 0.0,
            "weighted_rel": 0.0,
            "array_rel_diff": 0.7,
            "weighted_array_rel": 0.7,
        },
    ]

    first, _ = _first_failing_term(rows, fields={"Pe"}, terms={"advection", "parallel"})

    assert first is not None
    assert first["term"] == "parallel"


def test_first_failing_term_can_rank_by_rms_metric() -> None:
    rows = [
        {
            "step": 0,
            "field": "Pe",
            "term": "advection",
            "rel_diff": 0.4,
            "weighted_rel": 0.4,
            "array_rel_diff": 0.1,
            "weighted_array_rel": 0.1,
        },
        {
            "step": 0,
            "field": "Pe",
            "term": "parallel",
            "rel_diff": 0.0,
            "weighted_rel": 0.0,
            "array_rel_diff": 0.7,
            "weighted_array_rel": 0.7,
        },
    ]

    first, _ = _first_failing_term(
        rows,
        fields={"Pe"},
        terms={"advection", "parallel"},
        ranking_metric="rms",
    )

    assert first is not None
    assert first["term"] == "advection"
