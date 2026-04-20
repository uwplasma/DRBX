from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation.geometry_observables import (
    build_geometry_observable_report,
    line_group_from_report,
    profile_group_from_report,
    slice_group_from_report,
    write_geometry_observable_report,
)


def test_geometry_observable_helpers_build_family_reports(tmp_path: Path) -> None:
    profile_report = {
        "diagnostics": {
            "FHRP": {"Ne": {"coordinate_name": "R-Rsep", "coordinate_values": [0.0], "mean": [1.0]}},
        }
    }
    line_report = {
        "diagnostics": {
            "radial_midplane": {"J": {"coordinate_name": "s", "coordinate_values": [0.0], "mean": [1.0]}},
        }
    }
    slice_report = {"slice_name": "radial_index_planes", "coordinate_name": "s_index", "field_name": "g_33"}
    report = build_geometry_observable_report(
        geometry_family="demo_3d",
        benchmark_adapter="demo_adapter",
        observable_groups=(
            profile_group_from_report(profile_report, name="profiles", description="Demo profile bundle."),
            line_group_from_report(line_report, name="lineouts", description="Demo line bundle."),
            slice_group_from_report(slice_report, name="slices", description="Demo slice bundle."),
        ),
        metadata={"preview_mode": True},
    )
    path = write_geometry_observable_report(report, tmp_path / "observables.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["geometry_family"] == "demo_3d"
    assert payload["observable_groups"][0]["families"][0]["name"] == "FHRP"
    assert payload["observable_groups"][1]["families"][0]["kind"] == "lineout"
    assert payload["observable_groups"][2]["families"][0]["field_names"] == ["g_33"]
