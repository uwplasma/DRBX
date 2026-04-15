from __future__ import annotations

import json
from pathlib import Path

from jax_drb.validation import (
    compare_traced_field_line_selected_fields,
    create_traced_field_line_selected_field_parity_package,
)
from jax_drb.validation.traced_field_line_scaffold import _write_synthetic_mesh_spec


def test_compare_traced_field_line_selected_fields_is_zero_for_identical_specs(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    _write_synthetic_mesh_spec(reference)
    _write_synthetic_mesh_spec(candidate)
    result = compare_traced_field_line_selected_fields(
        reference_mesh_spec=reference,
        candidate_mesh_spec=candidate,
    )
    for error in result.variable_errors.values():
        assert error.max_abs_error == 0.0
        assert error.rms_error == 0.0


def test_create_traced_field_line_selected_field_parity_package_writes_artifacts(tmp_path: Path) -> None:
    artifacts = create_traced_field_line_selected_field_parity_package(
        reference_mesh_spec=None,
        candidate_mesh_spec=None,
        output_root=tmp_path / "output",
    )
    assert artifacts.parity_json_path.exists()
    assert artifacts.parity_arrays_npz_path.exists()
    assert artifacts.parity_plot_png_path.exists()
    assert artifacts.observable_report_json_path.exists()
    payload = json.loads(artifacts.parity_json_path.read_text(encoding="utf-8"))
    assert payload["field_names"] == ["jacobian", "g_11", "g_33"]
    observable = json.loads(artifacts.observable_report_json_path.read_text(encoding="utf-8"))
    assert observable["observable_groups"][0]["families"][0]["kind"] == "selected_field_parity"
