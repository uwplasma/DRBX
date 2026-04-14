from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

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


def test_traced_field_line_scaffold_reads_netcdf_fci_grid(tmp_path: Path) -> None:
    grid_path = tmp_path / "sample.fci.nc"
    with Dataset(grid_path, "w") as dataset:
        dataset.createDimension("x", 3)
        dataset.createDimension("y", 2)
        dataset.createDimension("z", 4)
        for name, values in {
            "Bxy": np.full((3, 2, 4), 1.5),
            "J": np.full((3, 2, 4), 0.9),
            "g11": np.full((3, 2, 4), 0.8),
            "g22": np.full((3, 2, 4), 1.2),
            "g33": np.full((3, 2, 4), 1.6),
        }.items():
            variable = dataset.createVariable(name, "f8", ("x", "y", "z"))
            variable[:] = values

    artifacts = create_traced_field_line_scaffold_package(
        output_root=tmp_path / "output",
        mesh_spec_path=grid_path,
    )

    manifest = json.loads(artifacts.manifest_json_path.read_text(encoding="utf-8"))
    assert manifest["source_format"] == "netcdf_fci_grid"
    assert manifest["preview_mode"] is False

    input_report = json.loads(artifacts.input_report_json_path.read_text(encoding="utf-8"))
    assert input_report["source_format"] == "netcdf_fci_grid"
    assert input_report["dimensions"] == {"ns": 3, "ntheta": 4, "nphi": 2}
    assert "g11" in input_report["declared_metric_fields"]

    metric_report = json.loads(artifacts.metric_report_json_path.read_text(encoding="utf-8"))
    assert metric_report["source_format"] == "netcdf_fci_grid"
    assert metric_report["metric_fields"]["Bxy"]["mean"] == 1.5
