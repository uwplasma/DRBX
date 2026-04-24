from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import (
    build_hermes_offender_register_report,
    create_hermes_offender_register_package,
    save_hermes_offender_register_plot,
)


def _case(
    name: str,
    *,
    field: str,
    rel_l2: float,
    rel_max: float,
    abs_error: float,
    runtime_ratio: float,
    normalization_sensitive: bool = False,
) -> dict[str, object]:
    return {
        "case_name": name,
        "display_label": name,
        "family": "synthetic",
        "capability_tier": "native_exact",
        "parity_mode": "one_step",
        "native_elapsed_seconds": 2.0 * runtime_ratio,
        "reference_elapsed_seconds": 2.0,
        "native_to_reference_runtime_ratio": runtime_ratio,
        "worst_relative_l2_field": field,
        "worst_relative_l2_error": rel_l2,
        "worst_relative_rms_field": field,
        "worst_relative_rms_error": 0.5 * rel_l2,
        "worst_max_abs_field": field,
        "worst_max_abs_diff": abs_error,
        "worst_relative_to_expected_max": rel_max,
        "normalization_sensitive": normalization_sensitive,
        "exact_match": False,
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    live_path = tmp_path / "live.json"
    comparison_path = tmp_path / "comparison.json"
    live_path.write_text(
        json.dumps(
            {
                "reference_code": "hermes-3",
                "case_count": 2,
                "cases": [
                    _case(
                        "neutral_mixed_one_step",
                        field="NVh",
                        rel_l2=2.0,
                        rel_max=2.4,
                        abs_error=3.0e-3,
                        runtime_ratio=4.0,
                    ),
                    _case(
                        "tokamak_recycling_one_step",
                        field="NVd",
                        rel_l2=0.7,
                        rel_max=0.9,
                        abs_error=1.0e-8,
                        runtime_ratio=0.5,
                        normalization_sensitive=True,
                    ),
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    comparison_path.write_text(
        json.dumps(
            {
                "reference_code": "hermes-3",
                "lane_count": 1,
                "lanes": [
                    {
                        "lane_name": "stellarator_vmec_native_selected_field",
                        "geometry_family": "stellarator_vmec_3d",
                        "worst_relative_l2_field": "toroidal_flux",
                        "worst_relative_l2_error": 0.4,
                        "worst_max_abs_field": "toroidal_flux",
                        "worst_max_abs_error": 0.03,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return live_path, comparison_path


def test_build_hermes_offender_register_ranks_parity_runtime_and_memory(tmp_path: Path) -> None:
    live_path, comparison_path = _write_inputs(tmp_path)

    report = build_hermes_offender_register_report(
        live_rerun_json=live_path,
        comparison_summary_json=comparison_path,
    )

    assert report["reference_code"] == "hermes-3"
    assert report["case_count"] == 2
    assert report["parity_offenders"][0]["case_name"] == "neutral_mixed_one_step"
    assert report["parity_offenders"][0]["component_hint"] == "neutral mixed boundary and parallel momentum closure"
    assert report["runtime_offenders"][0]["case_name"] == "neutral_mixed_one_step"
    assert report["runtime_offenders"][0]["runtime_status"] == "native_slower"
    assert report["memory_offenders"][0]["memory_measurement_status"] == "not_measured_in_live_register"
    assert report["top_offenders"]["parity"]["rank"] == 1


def test_create_hermes_offender_register_package_writes_json_and_plot(tmp_path: Path) -> None:
    live_path, comparison_path = _write_inputs(tmp_path)

    artifacts = create_hermes_offender_register_package(
        output_root=tmp_path / "output",
        live_rerun_json=live_path,
        comparison_summary_json=comparison_path,
    )

    assert artifacts.report_json_path.exists()
    assert artifacts.report_plot_png_path.exists()
    payload = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert payload["parity_offenders"][0]["case_name"] == "neutral_mixed_one_step"


def test_save_hermes_offender_register_plot_handles_empty_report(tmp_path: Path) -> None:
    output = tmp_path / "empty.png"

    save_hermes_offender_register_plot(
        {"parity_offenders": [], "runtime_offenders": []},
        output,
    )

    assert output.exists()
    assert output.stat().st_size > 0
