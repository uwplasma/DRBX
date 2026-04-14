from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import create_traced_field_line_scaffold_package


def test_traced_field_line_scaffold_preview_generates_artifacts(tmp_path: Path) -> None:
    artifacts = create_traced_field_line_scaffold_package(output_root=tmp_path / "output")
    for path in (
        artifacts.manifest_json_path,
        artifacts.input_report_json_path,
        artifacts.validation_contract_json_path,
        artifacts.metric_report_json_path,
        artifacts.metric_arrays_npz_path,
        artifacts.metric_plot_png_path,
    ):
        assert path.exists()

    manifest = json.loads(artifacts.manifest_json_path.read_text(encoding="utf-8"))
    assert manifest["geometry_family"] == "traced_field_line_3d"
    assert manifest["benchmark_adapter"] == "stellarator_traced_field_line_scaffold"
    assert manifest["preview_mode"] is True

    input_report = json.loads(artifacts.input_report_json_path.read_text(encoding="utf-8"))
    assert input_report["coordinate_system"] == "field_aligned"
    assert input_report["dimensions"]["ns"] == 24
    assert "Bmag" in input_report["declared_metric_fields"]

    contract = json.loads(artifacts.validation_contract_json_path.read_text(encoding="utf-8"))
    assert contract["metric_checks"][0] == "positive_jacobian"
    assert contract["promotion_gates"][-1] == "native_execution_bundle"

    metric_report = json.loads(artifacts.metric_report_json_path.read_text(encoding="utf-8"))
    assert metric_report["metric_fields"]["Bmag"]["finite"] is True
    assert metric_report["metric_fields"]["jacobian"]["minimum"] > 0.0
