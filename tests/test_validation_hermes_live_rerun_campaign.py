from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import (
    HermesLiveRerunCaseSpec,
    build_hermes_live_rerun_campaign_report,
    create_hermes_live_rerun_campaign_package,
    save_hermes_live_rerun_campaign_plot,
)
import jax_drb.validation.hermes_live_rerun_campaign as campaign_mod


def _sample_case(
    case_name: str,
    *,
    runtime_ratio: float,
    rel_l2: float,
    rel_max: float,
    exact: bool,
    normalization_sensitive: bool = False,
) -> dict[str, object]:
    return {
        "case_name": case_name,
        "display_label": case_name.replace("_", "\n"),
        "family": "synthetic",
        "reference_path": f"{case_name}/BOUT.inp",
        "parity_mode": "one_step",
        "capability_tier": "reference_backed_native",
        "compare_variable_count": 3,
        "native_elapsed_seconds": 2.0 * runtime_ratio,
        "reference_elapsed_seconds": 2.0,
        "native_to_reference_runtime_ratio": runtime_ratio,
        "reference_to_native_speedup": 0.0 if runtime_ratio == 0.0 else 1.0 / runtime_ratio,
        "worst_relative_l2_field": "Ne",
        "worst_relative_l2_error": rel_l2,
        "worst_relative_rms_field": "Ne",
        "worst_relative_rms_error": 0.5 * rel_max,
        "worst_max_abs_field": "Ne",
        "worst_max_abs_diff": rel_max,
        "worst_relative_to_expected_max": rel_max,
        "normalization_sensitive": normalization_sensitive,
        "exact_match": exact,
    }


def _sample_report() -> dict[str, object]:
    cases = [
        _sample_case("fast_case", runtime_ratio=0.5, rel_l2=1.0e-5, rel_max=2.0e-5, exact=False),
        _sample_case("exact_case", runtime_ratio=1.2, rel_l2=0.0, rel_max=0.0, exact=True),
    ]
    return {
        "reference_code": "hermes-3",
        "reference_root": "/tmp/reference",
        "reference_binary": "/tmp/reference/build/hermes-3",
        "case_count": len(cases),
        "cases": cases,
        "summaries": {
            "exact_match_case_count": 1,
            "best_runtime_ratio_case": "fast_case",
            "best_runtime_ratio": 0.5,
            "worst_runtime_ratio_case": "exact_case",
            "worst_runtime_ratio": 1.2,
            "worst_relative_l2_case": "fast_case",
            "worst_relative_l2_error": 1.0e-5,
            "worst_relative_rms_case": "fast_case",
            "worst_relative_rms_error": 1.0e-5,
            "normalization_sensitive_case_count": 0,
            "normalization_sensitive_cases": [],
        },
        "notes": {
            "comparison_surface": "test",
            "runtime_note": "test",
            "three_d_status": "test",
        },
    }


def test_save_hermes_live_rerun_campaign_plot_writes_png(tmp_path: Path) -> None:
    path = tmp_path / "campaign.png"
    save_hermes_live_rerun_campaign_plot(_sample_report(), path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_create_hermes_live_rerun_campaign_package_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(campaign_mod, "build_hermes_live_rerun_campaign_report", lambda **kwargs: _sample_report())

    artifacts = create_hermes_live_rerun_campaign_package(
        reference_root=tmp_path,
        output_root=tmp_path / "output",
    )

    assert artifacts.report_json_path.exists()
    assert artifacts.report_npz_path.exists()
    assert artifacts.report_plot_png_path.exists()
    payload = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert payload["reference_code"] == "hermes-3"
    assert payload["case_count"] == 2


def test_build_hermes_live_rerun_campaign_report_aggregates_cases(monkeypatch, tmp_path: Path) -> None:
    specs = (
        HermesLiveRerunCaseSpec("case_a", "A", "family_a"),
        HermesLiveRerunCaseSpec("case_b", "B", "family_b"),
    )

    monkeypatch.setattr(campaign_mod, "discover_reference_binary", lambda **kwargs: tmp_path / "build" / "hermes-3")
    monkeypatch.setattr(
        campaign_mod,
        "_run_hermes_live_rerun_case",
        lambda spec, **kwargs: _sample_case(
            spec.case_name,
            runtime_ratio=0.75 if spec.case_name == "case_a" else 1.5,
            rel_l2=1.0e-4 if spec.case_name == "case_a" else 2.5e-4,
            rel_max=3.0e-4 if spec.case_name == "case_a" else 4.0e-4,
            exact=spec.case_name == "case_a",
            normalization_sensitive=spec.case_name == "case_b",
        ),
    )

    report = build_hermes_live_rerun_campaign_report(reference_root=tmp_path, case_specs=specs)

    assert report["reference_binary"] == str(tmp_path / "build" / "hermes-3")
    assert report["case_count"] == 2
    assert report["summaries"]["exact_match_case_count"] == 1
    assert report["summaries"]["worst_relative_l2_case"] == "case_b"
    assert report["summaries"]["worst_relative_rms_case"] == "case_b"
    assert report["summaries"]["normalization_sensitive_case_count"] == 1
    assert report["summaries"]["normalization_sensitive_cases"] == ["case_b"]
