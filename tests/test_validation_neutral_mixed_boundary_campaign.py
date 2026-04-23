from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jax_drb.validation import (
    build_neutral_mixed_boundary_campaign_report,
    create_neutral_mixed_boundary_campaign_package,
    save_neutral_mixed_boundary_campaign_plot,
)
import jax_drb.validation.neutral_mixed_boundary_campaign as campaign_mod


def _sample_report() -> dict[str, object]:
    return {
        "case_name": "neutral_mixed_one_step",
        "reference_code": "hermes-3",
        "reference_root": "/tmp/reference",
        "reference_binary": "/tmp/reference/build/hermes-3",
        "reference_path": "/tmp/reference/BOUT.inp",
        "compare_variables": ["Nh", "Ph", "NVh"],
        "time_points": [0.1],
        "x_index": 5,
        "y_index": 3,
        "z_index": 5,
        "y_coordinate": [0, 1, 2, 3, 4, 5],
        "boundary_band_cells": 2,
        "native_elapsed_seconds": 2.0,
        "reference_elapsed_seconds": 1.0,
        "native_to_reference_runtime_ratio": 2.0,
        "profiles": {
            "Nh": {
                "reference_lineout": [1.0, 1.1, 1.2, 1.1, 1.0, 0.9],
                "native_lineout": [1.0, 1.12, 1.18, 1.09, 0.99, 0.88],
                "abs_diff_lineout": [0.0, 0.02, 0.02, 0.01, 0.01, 0.02],
                "max_abs_profile_y": [0.01, 0.02, 0.02, 0.01, 0.01, 0.02],
                "line_x": 5,
                "line_z": 5,
            },
            "Ph": {
                "reference_lineout": [0.1, 0.11, 0.12, 0.11, 0.1, 0.09],
                "native_lineout": [0.1, 0.111, 0.118, 0.109, 0.099, 0.091],
                "abs_diff_lineout": [0.0, 0.001, 0.002, 0.001, 0.001, 0.001],
                "max_abs_profile_y": [0.0, 0.001, 0.002, 0.001, 0.001, 0.001],
                "line_x": 5,
                "line_z": 5,
            },
            "NVh": {
                "reference_lineout": [0.0, 0.0, 0.01, 0.01, 0.0, 0.0],
                "native_lineout": [0.002, 0.003, 0.008, 0.009, 0.002, 0.003],
                "abs_diff_lineout": [0.002, 0.003, 0.002, 0.001, 0.002, 0.003],
                "max_abs_profile_y": [0.002, 0.003, 0.002, 0.001, 0.002, 0.003],
                "line_x": 5,
                "line_z": 5,
            },
        },
        "field_metrics": {
            "Nh": {
                "max_abs_error": 2.0e-2,
                "rms_error": 1.1e-2,
                "lineout_max_abs_error": 2.0e-2,
                "lower_boundary_max_abs_error": 2.0e-2,
                "upper_boundary_max_abs_error": 2.0e-2,
                "interior_max_abs_error": 2.0e-2,
            },
            "Ph": {
                "max_abs_error": 2.0e-3,
                "rms_error": 1.0e-3,
                "lineout_max_abs_error": 2.0e-3,
                "lower_boundary_max_abs_error": 1.0e-3,
                "upper_boundary_max_abs_error": 1.0e-3,
                "interior_max_abs_error": 2.0e-3,
            },
            "NVh": {
                "max_abs_error": 3.0e-3,
                "rms_error": 2.2e-3,
                "lineout_max_abs_error": 3.0e-3,
                "lower_boundary_max_abs_error": 3.0e-3,
                "upper_boundary_max_abs_error": 3.0e-3,
                "interior_max_abs_error": 2.0e-3,
            },
        },
        "worst_field": "Nh",
        "worst_max_abs_error": 2.0e-2,
        "notes": {
            "comparison_surface": "test",
            "plot_note": "test",
        },
    }


def test_save_neutral_mixed_boundary_campaign_plot_writes_png(tmp_path: Path) -> None:
    path = tmp_path / "campaign.png"
    save_neutral_mixed_boundary_campaign_plot(_sample_report(), path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_create_neutral_mixed_boundary_campaign_package_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(campaign_mod, "build_neutral_mixed_boundary_campaign_report", lambda **kwargs: _sample_report())

    artifacts = create_neutral_mixed_boundary_campaign_package(
        reference_root=tmp_path,
        output_root=tmp_path / "output",
    )

    assert artifacts.report_json_path.exists()
    assert artifacts.report_npz_path.exists()
    assert artifacts.report_plot_png_path.exists()
    payload = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert payload["reference_code"] == "hermes-3"
    assert payload["worst_field"] == "Nh"


def test_build_neutral_mixed_boundary_campaign_report_has_expected_schema(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(campaign_mod, "discover_reference_binary", lambda **kwargs: tmp_path / "build" / "hermes-3")
    monkeypatch.setattr(campaign_mod, "resolve_reference_case", lambda *args, **kwargs: (type("R", (), {"input_path": tmp_path / "BOUT.inp", "trim_x_guards": False, "trim_y_guards": False})(), None))
    monkeypatch.setattr(
        campaign_mod,
        "run_curated_case",
        lambda *args, **kwargs: type(
            "NativeResult",
            (),
            {
                "payload": {"time_points": [0.1], "variables": {}},
                "variables": {},
                "run_config": type("RunConfig", (), {"mesh": type("Mesh", (), {"mxg": 0, "myg": 0})()})(),
            },
        )(),
    )

    def _reference_execution(*args, **kwargs):
        workdir = kwargs["workdir"]
        (workdir / "BOUT.dmp.0.nc").write_text("placeholder", encoding="utf-8")
        return type(
            "ReferenceExecution",
            (),
            {
                "summary": type(
                    "Summary",
                    (),
                    {
                        "case_name": "neutral_mixed_one_step",
                        "parity_mode": "one_step",
                        "capability_tier": "reference_backed_native",
                        "compare_variables": ("Nh", "Ph", "NVh"),
                        "component_labels": (),
                        "overrides": {},
                        "nout": 1,
                        "timestep": 1.0,
                    },
                )(),
            },
        )()

    monkeypatch.setattr(campaign_mod, "run_reference_case", _reference_execution)
    monkeypatch.setattr(
        campaign_mod,
        "build_dataset_array_payload",
        lambda *args, **kwargs: {
            "time_points": [0.1],
            "variables": {
                "Nh": [[[[1.0], [1.1], [1.0]]]],
                "Ph": [[[[0.1], [0.11], [0.1]]]],
                "NVh": [[[[0.0], [0.01], [0.0]]]],
            },
        },
    )
    monkeypatch.setattr(
        campaign_mod,
        "build_array_payload_from_summary_payload",
        lambda *args, **kwargs: {
            "time_points": [0.1],
            "variables": {
                "Nh": np.asarray([[[[1.0], [1.12], [1.0]]]], dtype=float),
                "Ph": np.asarray([[[[0.1], [0.111], [0.1]]]], dtype=float),
                "NVh": np.asarray([[[[0.002], [0.008], [0.002]]]], dtype=float),
            },
        },
    )

    report = build_neutral_mixed_boundary_campaign_report(reference_root=tmp_path)

    assert report["case_name"] == "neutral_mixed_one_step"
    assert report["worst_field"] in {"Nh", "Ph", "NVh"}
    assert set(report["profiles"]) == {"Nh", "Ph", "NVh"}
