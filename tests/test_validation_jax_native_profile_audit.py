from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import create_jax_native_profile_audit_package


def test_create_jax_native_profile_audit_package_writes_summary_plot_and_traces(
    tmp_path: Path,
) -> None:
    artifacts = create_jax_native_profile_audit_package(output_root=tmp_path / "artifacts")

    assert artifacts.summary_json_path.exists()
    assert artifacts.summary_plot_png_path.exists()

    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["case"] == "jax_native_profile_audit"
    lanes = {entry["lane_name"]: entry for entry in payload["lanes"]}
    assert set(lanes) == {
        "traced_field_line_native_selected_field",
        "stellarator_vmec_native_selected_field",
    }
    for entry in lanes.values():
        assert entry["compile_seconds"] >= 0.0
        assert entry["first_execute_seconds"] >= 0.0
        assert entry["warm_execute_seconds"] >= 0.0
        assert entry["trace_files"]
