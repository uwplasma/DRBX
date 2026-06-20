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
    native_peak_mib: float = 500.0,
    reference_peak_mib: float = 250.0,
    normalization_sensitive: bool = False,
) -> dict[str, object]:
    native_peak_bytes = int(native_peak_mib * 1024.0 * 1024.0)
    reference_peak_bytes = int(reference_peak_mib * 1024.0 * 1024.0)
    return {
        "case_name": name,
        "display_label": name,
        "family": "synthetic",
        "capability_tier": "native_exact",
        "parity_mode": "one_step",
        "native_elapsed_seconds": 2.0 * runtime_ratio,
        "reference_elapsed_seconds": 2.0,
        "native_to_reference_runtime_ratio": runtime_ratio,
        "native_memory_measurement_status": "sampled_process_tree_rss",
        "reference_memory_measurement_status": "sampled_process_tree_rss",
        "native_peak_rss_bytes": native_peak_bytes,
        "reference_peak_rss_bytes": reference_peak_bytes,
        "native_peak_rss_mebibytes": native_peak_mib,
        "reference_peak_rss_mebibytes": reference_peak_mib,
        "native_peak_rss_delta_bytes": native_peak_bytes // 4,
        "reference_peak_rss_delta_bytes": reference_peak_bytes // 4,
        "native_peak_rss_delta_mebibytes": native_peak_mib / 4.0,
        "reference_peak_rss_delta_mebibytes": reference_peak_mib / 4.0,
        "native_to_reference_peak_rss_ratio": native_peak_mib / reference_peak_mib,
        "native_to_reference_peak_rss_delta_ratio": native_peak_mib / reference_peak_mib,
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
                        native_peak_mib=600.0,
                        reference_peak_mib=200.0,
                    ),
                    _case(
                        "tokamak_recycling_one_step",
                        field="NVd",
                        rel_l2=0.7,
                        rel_max=0.9,
                        abs_error=1.0e-8,
                        runtime_ratio=0.5,
                        native_peak_mib=200.0,
                        reference_peak_mib=400.0,
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
                        "source_mode": "explicit_pair",
                        "candidate_origin": "provided_external_input",
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
    assert report["parity_offenders"][0]["component_hint"] == "neutral mixed target-band state history and boundary sequencing"
    assert "source formulas are already closed" in report["parity_offenders"][0]["recommended_next_action"]
    assert report["actionable_parity_offenders"][0]["case_name"] == (
        "neutral_mixed_one_step"
    )
    assert report["top_offenders"]["actionable_parity"]["case_name"] == (
        "neutral_mixed_one_step"
    )
    assert "/Users/" not in json.dumps(report["source_artifacts"], sort_keys=True)
    assert report["runtime_offenders"][0]["case_name"] == "neutral_mixed_one_step"
    assert report["runtime_offenders"][0]["runtime_status"] == "native_slower"
    assert report["memory_offenders"][0]["memory_measurement_status"] == "sampled_process_tree_rss"
    assert report["memory_offenders"][0]["native_to_reference_peak_rss_ratio"] == 3.0
    assert report["top_offenders"]["parity"]["rank"] == 1


def test_build_hermes_offender_register_filters_normalization_sensitive_actionables(
    tmp_path: Path,
) -> None:
    live_path = tmp_path / "live.json"
    comparison_path = tmp_path / "comparison.json"
    live_path.write_text(
        json.dumps(
            {
                "reference_code": "hermes-3",
                "case_count": 2,
                "cases": [
                    _case(
                        "near_zero_recycling_one_step",
                        field="NVd",
                        rel_l2=0.99,
                        rel_max=1.2,
                        abs_error=1.0e-12,
                        runtime_ratio=1.0,
                        normalization_sensitive=True,
                    ),
                    _case(
                        "recycling_dthe_one_step",
                        field="NVd",
                        rel_l2=0.05,
                        rel_max=0.06,
                        abs_error=1.0e-2,
                        runtime_ratio=3.0,
                    ),
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    comparison_path.write_text(
        json.dumps({"reference_code": "hermes-3", "lane_count": 0, "lanes": []}),
        encoding="utf-8",
    )

    report = build_hermes_offender_register_report(
        live_rerun_json=live_path,
        comparison_summary_json=comparison_path,
    )

    assert report["parity_offenders"][0]["case_name"] == "near_zero_recycling_one_step"
    assert report["parity_offenders"][0]["normalization_sensitive"] is True
    assert report["actionable_parity_offenders"][0]["case_name"] == (
        "recycling_dthe_one_step"
    )
    assert report["top_offenders"]["actionable_parity"]["case_name"] == (
        "recycling_dthe_one_step"
    )


def test_build_hermes_offender_register_filters_synthetic_preview_actionables(
    tmp_path: Path,
) -> None:
    live_path = tmp_path / "live.json"
    comparison_path = tmp_path / "comparison.json"
    live_path.write_text(
        json.dumps({"reference_code": "hermes-3", "case_count": 0, "cases": []}),
        encoding="utf-8",
    )
    comparison_path.write_text(
        json.dumps(
            {
                "reference_code": "hermes-3",
                "lane_count": 2,
                "lanes": [
                    {
                        "lane_name": "stellarator_vmec_native_selected_field",
                        "geometry_family": "stellarator_vmec_3d",
                        "worst_relative_l2_field": "toroidal_flux",
                        "worst_relative_l2_error": 0.5,
                        "worst_max_abs_field": "toroidal_flux",
                        "worst_max_abs_error": 0.04,
                        "source_mode": "synthetic_preview",
                        "candidate_origin": "synthetic_preview_pair",
                    },
                    {
                        "lane_name": "traced_field_line_native_selected_field",
                        "geometry_family": "traced_field_line_3d",
                        "worst_relative_l2_field": "g33",
                        "worst_relative_l2_error": 0.1,
                        "worst_max_abs_field": "g33",
                        "worst_max_abs_error": 0.2,
                        "source_mode": "explicit_pair",
                        "candidate_origin": "provided_external_input",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = build_hermes_offender_register_report(
        live_rerun_json=live_path,
        comparison_summary_json=comparison_path,
    )

    assert report["parity_offenders"][0]["case_name"] == (
        "stellarator_vmec_native_selected_field"
    )
    assert report["parity_offenders"][0]["diagnostic_preview"] is True
    assert report["actionable_parity_offenders"][0]["case_name"] == (
        "traced_field_line_native_selected_field"
    )
    assert "synthetic preview" in report["parity_offenders"][0]["recommended_next_action"]


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


def test_build_hermes_offender_register_falls_back_to_runtime_memory_proxy(tmp_path: Path) -> None:
    live_path, comparison_path = _write_inputs(tmp_path)
    live_payload = json.loads(live_path.read_text(encoding="utf-8"))
    for case in live_payload["cases"]:
        for key in tuple(case):
            if "rss" in key or "memory" in key:
                case.pop(key)
    live_path.write_text(json.dumps(live_payload), encoding="utf-8")

    report = build_hermes_offender_register_report(
        live_rerun_json=live_path,
        comparison_summary_json=comparison_path,
    )

    assert report["memory_offenders"][0]["memory_measurement_status"] == "not_measured_in_live_register"
    assert report["memory_offenders"][0]["case_name"] == "neutral_mixed_one_step"
