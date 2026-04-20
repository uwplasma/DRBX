from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jax_drb.validation.tokamak_native_selected_field import (
    _compare_native_selected_field_histories,
    create_native_tokamak_selected_field_package,
)


class _FakeResult:
    def __init__(self) -> None:
        self.variables = {
            "Ne": np.array(
                [
                    [[[1.1], [2.2]]],
                    [[[1.3], [2.4]]],
                ],
                dtype=np.float64,
            ),
            "Pe": np.array(
                [
                    [[[0.55], [0.65]]],
                    [[[0.75], [0.85]]],
                ],
                dtype=np.float64,
            ),
            "phi": np.array(
                [
                    [[[0.01], [0.02]]],
                    [[[0.03], [0.04]]],
                ],
                dtype=np.float64,
            ),
        }
        self.time_points = (0.0, 1.0)
        self.payload = {
            "capability_tier": "native_exact",
            "parity_mode": "short_window",
            "dimensions": {"t": 2, "x": 1, "y": 2, "z": 1},
            "time_points": [0.0, 1.0],
            "configured_nout": 1,
            "configured_timestep": 1.0,
            "component_labels": ["component_a", "component_b"],
            "dataset_scalars": {"Cs0": 2.0},
            "producer": "jax-drb",
        }


def test_create_native_tokamak_selected_field_package_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    baseline_dir = tmp_path / "reference_arrays"
    baseline_dir.mkdir()
    np.savez_compressed(
        baseline_dir / "tokamak_turbulence_short_window.npz",
        __metadata__=json.dumps(
            {
                "case_name": "tokamak_turbulence_short_window",
                "parity_mode": "short_window",
                "capability_tier": "native_exact",
                "producer": "external-reference",
                "compare_variables": ["Ne", "Pe", "phi"],
                "component_labels": ["component_a", "component_b"],
                "dimensions": {"t": 2, "x": 1, "y": 2, "z": 1},
                "time_points": [0.0, 1.0],
                "dataset_scalars": {"Cs0": 2.0},
                "variable_dimensions": {
                    "Ne": ["t", "x", "y", "z"],
                    "Pe": ["t", "x", "y", "z"],
                    "phi": ["t", "x", "y", "z"],
                },
                "effective_output_points": 2,
            },
            sort_keys=True,
        ),
        var__Ne=np.array([[[[1.0], [2.0]]], [[[1.2], [2.2]]]], dtype=np.float64),
        var__Pe=np.array([[[[0.5], [0.6]]], [[[0.7], [0.8]]]], dtype=np.float64),
        var__phi=np.array([[[[0.0], [0.01]]], [[[0.02], [0.03]]]], dtype=np.float64),
    )

    monkeypatch.setattr(
        "jax_drb.validation.tokamak_native_selected_field._REFERENCE_ARRAY_BASELINE_DIR",
        baseline_dir,
    )
    monkeypatch.setattr(
        "jax_drb.validation.tokamak_native_selected_field.run_curated_case",
        lambda case_name, reference_root: _FakeResult(),
    )

    artifacts = create_native_tokamak_selected_field_package(
        case_name="tokamak_turbulence_short_window",
        reference_root=tmp_path / "reference_root",
        output_root=tmp_path / "output",
    )

    assert artifacts.parity_json_path.exists()
    assert artifacts.parity_arrays_npz_path.exists()
    assert artifacts.parity_plot_png_path.exists()
    assert artifacts.comparison_json_path.exists()
    assert artifacts.comparison_plot_png_path.exists()
    assert artifacts.observable_report_json_path.exists()
    assert artifacts.runtime_report_json_path.exists()

    parity = json.loads(artifacts.parity_json_path.read_text(encoding="utf-8"))
    assert parity["case_name"] == "tokamak_turbulence_short_window"
    assert parity["variable_errors"]["Ne"]["max_abs_error"] > 0.0

    observable = json.loads(artifacts.observable_report_json_path.read_text(encoding="utf-8"))
    assert observable["benchmark_adapter"] == "native_tokamak_selected_field"
    assert observable["metadata"]["native_capability_tier"] == "native_exact"
    assert observable["metadata"]["reference_source"] == "committed_reference_arrays"

    runtime = json.loads(artifacts.runtime_report_json_path.read_text(encoding="utf-8"))
    assert runtime["case_name"] == "tokamak_turbulence_short_window"
    assert runtime["selected_fields"] == ["Ne", "Pe", "phi"]
    assert runtime["component_labels"] == ["component_a", "component_b"]

    comparison = json.loads(artifacts.comparison_json_path.read_text(encoding="utf-8"))
    assert comparison["reference_source"] == "committed_reference_arrays"
    assert comparison["reference_producer"] == "external-reference"
    assert comparison["comparison_histories"]["Ne"]["reference_domain_mean"] == [1.5, 1.7000000000000002]
    assert comparison["comparison_histories"]["Ne"]["native_domain_mean"] == [1.6500000000000001, 1.85]


def test_compare_native_tokamak_selected_field_histories_rejects_mismatched_time_points() -> None:
    expected = {"Ne": np.ones((2, 1, 1, 1), dtype=np.float64)}
    actual = {"Ne": np.ones((2, 1, 1, 1), dtype=np.float64)}
    try:
        _compare_native_selected_field_histories(
            case_name="tokamak_turbulence_one_step",
            expected_fields=expected,
            actual_fields=actual,
            expected_time_points=[0.0, 1.0],
            actual_time_points=(0.0, 2.0),
            field_names=("Ne",),
        )
    except ValueError as exc:
        assert "time points do not match" in str(exc)
    else:
        raise AssertionError("Expected mismatched time points to raise ValueError")


def test_compare_native_tokamak_selected_field_histories_rejects_missing_field() -> None:
    try:
        _compare_native_selected_field_histories(
            case_name="tokamak_turbulence_one_step",
            expected_fields={"Ne": np.ones((1, 1, 1, 1), dtype=np.float64)},
            actual_fields={"Pe": np.ones((1, 1, 1, 1), dtype=np.float64)},
            expected_time_points=[0.0],
            actual_time_points=(0.0,),
            field_names=("Ne",),
        )
    except KeyError as exc:
        assert "missing" in str(exc).lower()
    else:
        raise AssertionError("Expected missing selected field to raise KeyError")


def test_compare_native_tokamak_selected_field_histories_rejects_shape_mismatch() -> None:
    try:
        _compare_native_selected_field_histories(
            case_name="tokamak_turbulence_one_step",
            expected_fields={"Ne": np.ones((1, 1, 1, 1), dtype=np.float64)},
            actual_fields={"Ne": np.ones((1, 1, 2, 1), dtype=np.float64)},
            expected_time_points=[0.0],
            actual_time_points=(0.0,),
            field_names=("Ne",),
        )
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("Expected selected-field shape mismatch to raise ValueError")
