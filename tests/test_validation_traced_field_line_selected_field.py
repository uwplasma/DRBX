from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

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
    assert artifacts.source_report_json_path.exists()
    payload = json.loads(artifacts.parity_json_path.read_text(encoding="utf-8"))
    assert payload["field_names"] == ["jacobian", "g_11", "g_33"]
    observable = json.loads(artifacts.observable_report_json_path.read_text(encoding="utf-8"))
    assert observable["observable_groups"][0]["families"][0]["kind"] == "selected_field_parity"
    assert observable["metadata"]["source_mode"] == "synthetic_preview"
    source = json.loads(artifacts.source_report_json_path.read_text(encoding="utf-8"))
    assert source["candidate_origin"] == "synthetic_preview_pair"


def test_create_traced_field_line_selected_field_parity_package_derives_candidate_from_external_grid(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.fci.nc"
    with Dataset(reference, "w") as dataset:
        dataset.createDimension("x", 4)
        dataset.createDimension("y", 3)
        dataset.createDimension("z", 2)
        for name, scale in (("J", 1.0), ("g_11", 2.0), ("g_33", 3.0)):
            variable = dataset.createVariable(name, "f8", ("x", "y", "z"))
            values = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
            variable[:] = scale + values

    artifacts = create_traced_field_line_selected_field_parity_package(
        reference_mesh_spec=reference,
        candidate_mesh_spec=None,
        output_root=tmp_path / "output",
    )

    payload = json.loads(artifacts.parity_json_path.read_text(encoding="utf-8"))
    assert payload["field_names"] == ["J", "g_11", "g_33"]
    assert payload["variable_errors"]["J"]["max_abs_error"] > 0.0
    observable = json.loads(artifacts.observable_report_json_path.read_text(encoding="utf-8"))
    assert observable["metadata"]["source_mode"] == "external_explicit_pair"
    source = json.loads(artifacts.source_report_json_path.read_text(encoding="utf-8"))
    assert source["candidate_origin"] == "materialized_from_reference_input"


def test_create_traced_field_line_selected_field_parity_package_supports_explicit_external_pair(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.fci.nc"
    candidate = tmp_path / "candidate.fci.nc"
    for path, offset in ((reference, 0.0), (candidate, 0.25)):
        with Dataset(path, "w") as dataset:
            dataset.createDimension("x", 4)
            dataset.createDimension("y", 3)
            dataset.createDimension("z", 2)
            for name, scale in (("g11", 2.0), ("g33", 3.0)):
                variable = dataset.createVariable(name, "f8", ("x", "y", "z"))
                values = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
                variable[:] = scale + values + offset

    artifacts = create_traced_field_line_selected_field_parity_package(
        reference_mesh_spec=reference,
        candidate_mesh_spec=candidate,
        output_root=tmp_path / "output",
        field_names=("g11", "g33"),
    )

    payload = json.loads(artifacts.parity_json_path.read_text(encoding="utf-8"))
    assert payload["field_names"] == ["g11", "g33"]
    source = json.loads(artifacts.source_report_json_path.read_text(encoding="utf-8"))
    assert source["source_mode"] == "explicit_pair"
    assert source["candidate_origin"] == "provided_external_input"
