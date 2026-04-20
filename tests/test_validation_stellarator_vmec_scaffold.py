from __future__ import annotations

import json
from pathlib import Path

from netCDF4 import Dataset
import numpy as np

from jax_drb.validation import create_stellarator_vmec_scaffold_package


def test_stellarator_vmec_scaffold_preview_generates_artifacts(tmp_path: Path) -> None:
    artifacts = create_stellarator_vmec_scaffold_package(output_root=tmp_path / "output")
    for path in (
        artifacts.manifest_json_path,
        artifacts.input_report_json_path,
        artifacts.validation_contract_json_path,
        artifacts.profile_report_json_path,
        artifacts.profile_arrays_npz_path,
        artifacts.profile_plot_png_path,
        artifacts.surface_report_json_path,
        artifacts.surface_arrays_npz_path,
        artifacts.surface_plot_png_path,
        artifacts.surface_gif_path,
        artifacts.observable_report_json_path,
    ):
        assert path.exists()

    manifest = json.loads(artifacts.manifest_json_path.read_text(encoding="utf-8"))
    assert manifest["geometry_family"] == "stellarator_vmec_3d"
    assert manifest["benchmark_adapter"] == "stellarator_vmec_scaffold"
    assert manifest["preview_mode"] is True
    assert manifest["source_format"] == "synthetic_vmec_wout"

    input_report = json.loads(artifacts.input_report_json_path.read_text(encoding="utf-8"))
    assert input_report["coordinate_system"] == "vmec_flux_coordinates"
    assert input_report["nfp"] == 5
    assert input_report["dimensions"]["selected_surfaces"] == 3

    contract = json.loads(artifacts.validation_contract_json_path.read_text(encoding="utf-8"))
    assert contract["promotion_gates"][-1] == "native_execution_bundle"
    assert contract["surface_checks"][0] == "finite_surface_coordinates"

    profile_report = json.loads(artifacts.profile_report_json_path.read_text(encoding="utf-8"))
    assert sorted(profile_report["diagnostics"]["radial_profiles"]) == ["iota", "pressure", "toroidal_flux"]

    surface_report = json.loads(artifacts.surface_report_json_path.read_text(encoding="utf-8"))
    assert surface_report["coordinate_name"] == "toroidal_angle"
    assert len(surface_report["frames"]) == 24
    assert len(surface_report["frames"][0]["surface_summaries"]) == 3

    observable_report = json.loads(artifacts.observable_report_json_path.read_text(encoding="utf-8"))
    assert observable_report["geometry_family"] == "stellarator_vmec_3d"
    assert observable_report["observable_groups"][0]["families"][0]["kind"] == "profile"
    assert observable_report["observable_groups"][1]["families"][0]["kind"] == "surface_cross_section"


def test_stellarator_vmec_scaffold_reads_vmec_wout_netcdf(tmp_path: Path) -> None:
    wout_path = tmp_path / "wout_test.nc"
    ns = 6
    mn_mode = 2
    with Dataset(wout_path, "w") as dataset:
        dataset.createDimension("ns", ns)
        dataset.createDimension("mn_mode", mn_mode)
        dataset.createVariable("iotaf", "f8", ("ns",))[:] = np.linspace(0.35, 0.55, ns)
        dataset.createVariable("presf", "f8", ("ns",))[:] = np.linspace(1000.0, 10.0, ns)
        dataset.createVariable("phi", "f8", ("ns",))[:] = np.linspace(0.0, 2.0, ns)
        dataset.createVariable("xm", "f8", ("mn_mode",))[:] = np.asarray([0.0, 1.0])
        dataset.createVariable("xn", "f8", ("mn_mode",))[:] = np.asarray([0.0, 0.0])
        rmnc = dataset.createVariable("rmnc", "f8", ("ns", "mn_mode"))
        zmns = dataset.createVariable("zmns", "f8", ("ns", "mn_mode"))
        rmnc[:] = np.column_stack((np.full(ns, 4.1), np.linspace(0.05, 0.4, ns)))
        zmns[:] = np.column_stack((np.zeros(ns), np.linspace(0.08, 0.55, ns)))
        dataset.createVariable("nfp", "i4")[:] = 4

    artifacts = create_stellarator_vmec_scaffold_package(
        output_root=tmp_path / "output",
        equilibrium_path=wout_path,
    )

    manifest = json.loads(artifacts.manifest_json_path.read_text(encoding="utf-8"))
    assert manifest["preview_mode"] is False
    assert manifest["source_format"] == "vmec_wout_netcdf"

    input_report = json.loads(artifacts.input_report_json_path.read_text(encoding="utf-8"))
    assert input_report["source_format"] == "vmec_wout_netcdf"
    assert input_report["dimensions"]["ns"] == ns
    assert input_report["nfp"] == 4

    surface_report = json.loads(artifacts.surface_report_json_path.read_text(encoding="utf-8"))
    assert len(surface_report["frames"]) == 24
    first_surface = surface_report["frames"][0]["surface_summaries"][0]
    assert first_surface["radial_extent"] > 0.0
    assert first_surface["vertical_extent"] > 0.0
