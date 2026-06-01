from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import (
    build_tokamak_recycling_observable_campaign_report,
    create_tokamak_recycling_observable_campaign_package,
    save_tokamak_recycling_observable_campaign_plot,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_ARRAYS = (
    _REPO_ROOT
    / "references"
    / "baselines"
    / "reference_arrays"
    / "tokamak_recycling_dthe_one_step.npz"
)


def test_tokamak_recycling_observable_campaign_report_uses_profile_observables() -> None:
    report = build_tokamak_recycling_observable_campaign_report(
        reference_arrays_npz=_REFERENCE_ARRAYS,
        native_arrays_npz=_REFERENCE_ARRAYS,
    )

    assert report["case_name"] == "tokamak_recycling_dthe_one_step"
    assert "/Users/" not in report["native_source"]
    assert "local/hermes" not in report["native_source"]
    assert report["passed_metric_count"] == report["metric_count"]
    assert "solps_iter_tcv_x21" in report["literature_anchor"]
    profiles = report["profiles"]
    assert profiles["target_indices"] == {"lower": 0, "upper": 7}
    assert set(profiles["species"]) == {"d", "t", "he"}
    assert "target_flux_proxy" in profiles["species"]["d"]
    assert "neutral_parallel_density" in profiles["species"]["d"]
    assert "electron_temperature" in profiles


def test_tokamak_recycling_observable_campaign_package_writes_outputs(tmp_path: Path) -> None:
    artifacts = create_tokamak_recycling_observable_campaign_package(
        output_root=tmp_path / "artifacts",
        reference_arrays_npz=_REFERENCE_ARRAYS,
        native_arrays_npz=_REFERENCE_ARRAYS,
    )
    report = build_tokamak_recycling_observable_campaign_report(
        reference_arrays_npz=_REFERENCE_ARRAYS,
        native_arrays_npz=_REFERENCE_ARRAYS,
    )
    plot = save_tokamak_recycling_observable_campaign_plot(report, tmp_path / "plot.png")

    assert artifacts.report_json_path.exists()
    assert artifacts.report_npz_path.exists()
    assert artifacts.report_plot_png_path.exists()
    assert plot.exists()
    payload = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    assert "/Users/" not in json.dumps(payload, sort_keys=True)
    assert payload["observable_contract"]["target_density_profiles"].startswith("final")
