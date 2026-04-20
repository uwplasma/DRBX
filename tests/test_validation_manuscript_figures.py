from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import create_manuscript_figure_package


def test_create_manuscript_figure_package_writes_expected_artifacts(tmp_path: Path) -> None:
    artifacts = create_manuscript_figure_package(output_root=tmp_path / "output")

    assert artifacts.manifest_json_path.exists()
    assert artifacts.architecture_png_path.exists()
    assert artifacts.equations_geometry_png_path.exists()
    assert artifacts.transient_panel_png_path.exists()

    payload = json.loads(artifacts.manifest_json_path.read_text(encoding="utf-8"))
    assert payload["case"] == "manuscript_figures"
    assert len(payload["figures"]) == 3
    assert {entry["name"] for entry in payload["figures"]} == {
        "architecture_validation_ladder",
        "equations_geometry_summary",
        "transient_validation_panel",
    }
