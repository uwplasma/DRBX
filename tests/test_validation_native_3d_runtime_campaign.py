from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import create_native_3d_runtime_campaign_package


def _write_runtime(path: Path, *, geometry_family: str, elapsed_seconds: float, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "geometry_family": geometry_family,
                "capability_tier": "native_exact_reduced",
                "elapsed_seconds": elapsed_seconds,
                "selected_fields": fields,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_create_native_3d_runtime_campaign_package_writes_summary_and_plot(tmp_path: Path) -> None:
    tokamak_one = tmp_path / "tokamak_one.json"
    tokamak_short = tmp_path / "tokamak_short.json"
    traced = tmp_path / "traced.json"
    stellarator = tmp_path / "stellarator.json"
    _write_runtime(tokamak_one, geometry_family="diverted_tokamak_3d", elapsed_seconds=0.2, fields=["Ne", "Pe", "phi"])
    _write_runtime(tokamak_short, geometry_family="diverted_tokamak_3d", elapsed_seconds=0.4, fields=["Ne", "phi", "Vort"])
    _write_runtime(traced, geometry_family="traced_field_line_3d", elapsed_seconds=0.05, fields=["g11", "g33"])
    _write_runtime(stellarator, geometry_family="stellarator_vmec_3d", elapsed_seconds=0.06, fields=["iota", "pressure", "toroidal_flux"])

    artifacts = create_native_3d_runtime_campaign_package(
        output_root=tmp_path / "output",
        tokamak_one_step_runtime_report=tokamak_one,
        tokamak_short_window_runtime_report=tokamak_short,
        traced_native_runtime_report=traced,
        stellarator_native_runtime_report=stellarator,
    )
    assert artifacts.summary_json_path.exists()
    assert artifacts.summary_plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["case"] == "native_3d_runtime_campaign"
    assert len(payload["native_lane_runtimes"]) == 4
    assert len(payload["scaling_sweeps"]) == 2
