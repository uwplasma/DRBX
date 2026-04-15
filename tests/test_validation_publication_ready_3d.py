from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import create_publication_ready_3d_campaign_package


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_create_publication_ready_3d_campaign_package_writes_summary_and_plot(tmp_path: Path) -> None:
    tokamak_one_step_runtime = tmp_path / "tokamak_one_step_runtime.json"
    tokamak_one_step_parity = tmp_path / "tokamak_one_step_parity.json"
    tokamak_short_runtime = tmp_path / "tokamak_short_runtime.json"
    tokamak_short_parity = tmp_path / "tokamak_short_parity.json"
    traced_parity = tmp_path / "traced_parity.json"
    traced_source = tmp_path / "traced_source.json"
    stellarator_parity = tmp_path / "stellarator_parity.json"
    stellarator_source = tmp_path / "stellarator_source.json"
    convergence = tmp_path / "convergence.json"

    _write_json(
        tokamak_one_step_runtime,
        {
            "capability_tier": "native_exact",
            "case_name": "tokamak_one_step",
            "elapsed_seconds": 0.2,
            "selected_fields": ["Ne", "Pe", "phi"],
        },
    )
    _write_json(
        tokamak_one_step_parity,
        {
            "field_names": ["Ne", "Pe", "phi"],
            "variable_errors": {
                "Ne": {"relative_l2_error": 0.01},
                "Pe": {"relative_l2_error": 0.02},
                "phi": {"relative_l2_error": 0.03},
            },
        },
    )
    _write_json(
        tokamak_short_runtime,
        {
            "capability_tier": "native_exact",
            "case_name": "tokamak_short",
            "elapsed_seconds": 0.4,
            "selected_fields": ["Ne", "phi", "Vort"],
        },
    )
    _write_json(
        tokamak_short_parity,
        {
            "field_names": ["Ne", "phi", "Vort"],
            "variable_errors": {
                "Ne": {"relative_l2_error": 0.01},
                "phi": {"relative_l2_error": 0.005},
                "Vort": {"relative_l2_error": 0.02},
            },
        },
    )
    _write_json(
        traced_parity,
        {
            "field_names": ["g11", "g33"],
            "variable_errors": {
                "g11": {"relative_l2_error": 0.04},
                "g33": {"relative_l2_error": 0.15},
            },
        },
    )
    _write_json(
        traced_source,
        {
            "source_mode": "explicit_pair",
            "candidate_origin": "provided_external_input",
        },
    )
    _write_json(
        stellarator_parity,
        {
            "field_names": ["iota", "pressure", "toroidal_flux"],
            "variable_errors": {
                "iota": {"relative_l2_error": 0.01},
                "pressure": {"relative_l2_error": 0.03},
                "toroidal_flux": {"relative_l2_error": 0.02},
            },
        },
    )
    _write_json(
        stellarator_source,
        {
            "source_mode": "explicit_pair",
            "candidate_origin": "provided_external_input",
        },
    )
    _write_json(
        convergence,
        {
            "case": "fluid_1d_mms_convergence",
            "resolutions": [32, 64, 128],
            "observed_orders": [
                {"density_order": 2.1, "momentum_order": 1.9, "pressure_order": 2.0},
                {"density_order": 1.8, "momentum_order": 2.0, "pressure_order": 2.2},
            ],
        },
    )

    artifacts = create_publication_ready_3d_campaign_package(
        output_root=tmp_path / "output",
        tokamak_one_step_runtime_report=tokamak_one_step_runtime,
        tokamak_one_step_parity_json=tokamak_one_step_parity,
        tokamak_short_window_runtime_report=tokamak_short_runtime,
        tokamak_short_window_parity_json=tokamak_short_parity,
        traced_field_line_parity_json=traced_parity,
        traced_field_line_source_report=traced_source,
        stellarator_parity_json=stellarator_parity,
        stellarator_source_report=stellarator_source,
        convergence_report_json=convergence,
    )

    assert artifacts.summary_json_path.exists()
    assert artifacts.summary_plot_png_path.exists()

    summary = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert summary["case"] == "publication_ready_3d_campaign"
    assert len(summary["lane_summaries"]) == 4
    assert summary["campaign_status"]["native_non_tokamak_rungs"] == 0
    assert summary["convergence_summary"]["min_density_order"] == 1.8
