from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_drb.validation.traced_field_line_native_selected_field import (
    compare_native_traced_field_line_selected_fields,
    create_native_traced_field_line_selected_field_package,
)


def _write_metric_grid(path: Path, *, offset: float) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("x", 4)
        dataset.createDimension("y", 3)
        dataset.createDimension("z", 2)
        for name, scale in (("g11", 2.0), ("g33", 3.0)):
            variable = dataset.createVariable(name, "f8", ("x", "y", "z"))
            values = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
            variable[:] = scale + values + offset


def test_compare_native_traced_field_line_selected_fields_returns_zero_for_identical_pair(tmp_path: Path) -> None:
    reference = tmp_path / "reference.fci.nc"
    candidate = tmp_path / "candidate.fci.nc"
    _write_metric_grid(reference, offset=0.0)
    _write_metric_grid(candidate, offset=0.0)
    result, reference_profiles, candidate_profiles = compare_native_traced_field_line_selected_fields(
        reference_mesh_spec=reference,
        candidate_mesh_spec=candidate,
    )
    assert tuple(result.field_names) == ("g11", "g33")
    for error in result.variable_errors.values():
        assert error.max_abs_error == 0.0
        assert error.rms_error == 0.0
    np.testing.assert_allclose(reference_profiles["g11"], candidate_profiles["g11"])


def test_create_native_traced_field_line_selected_field_package_writes_artifacts(tmp_path: Path) -> None:
    reference = tmp_path / "reference.fci.nc"
    candidate = tmp_path / "candidate.fci.nc"
    _write_metric_grid(reference, offset=0.0)
    _write_metric_grid(candidate, offset=0.25)
    artifacts = create_native_traced_field_line_selected_field_package(
        reference_mesh_spec=reference,
        candidate_mesh_spec=candidate,
        output_root=tmp_path / "output",
    )
    assert artifacts.parity_json_path.exists()
    assert artifacts.parity_arrays_npz_path.exists()
    assert artifacts.parity_plot_png_path.exists()
    assert artifacts.comparison_json_path.exists()
    assert artifacts.comparison_plot_png_path.exists()
    assert artifacts.observable_report_json_path.exists()
    assert artifacts.runtime_report_json_path.exists()

    parity = json.loads(artifacts.parity_json_path.read_text(encoding="utf-8"))
    assert parity["field_names"] == ["g11", "g33"]
    assert parity["variable_errors"]["g11"]["max_abs_error"] > 0.0
    runtime = json.loads(artifacts.runtime_report_json_path.read_text(encoding="utf-8"))
    assert runtime["native_capability_tier"] == "native_exact_reduced"
    assert runtime["selected_fields"] == ["g11", "g33"]
