from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from dkx.validation.vmec_extender_edge_field_campaign import (
    build_vmec_extender_edge_field_campaign_report,
    create_vmec_extender_edge_field_campaign_package,
)


def _write_campaign_grid(path: Path) -> Path:
    nfp = 5
    phi_period = 2.0 * np.pi / float(nfp)
    R = np.asarray([1.0, 1.3, 1.7], dtype=np.float64)
    phi = np.linspace(0.0, phi_period, 5, endpoint=False, dtype=np.float64)
    Z = np.asarray([-0.4, 0.1, 0.6], dtype=np.float64)
    RR, PP, ZZ = np.meshgrid(R, phi, Z, indexing="ij")
    BR = R[:, None, None] + 2.0 * PP + 3.0 * ZZ
    Bphi = 2.0 + RR
    BZ = RR - PP + ZZ
    absB = np.sqrt(BR * BR + Bphi * Bphi + BZ * BZ)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("nR", R.size)
        dataset.createDimension("nphi", phi.size)
        dataset.createDimension("nZ", Z.size)
        dataset.createVariable("R", "f8", ("nR",))[:] = R
        dataset.createVariable("phi", "f8", ("nphi",))[:] = phi
        dataset.createVariable("Z", "f8", ("nZ",))[:] = Z
        dataset.createVariable("BR", "f8", ("nR", "nphi", "nZ"))[:] = BR
        dataset.createVariable("Bphi", "f8", ("nR", "nphi", "nZ"))[:] = Bphi
        dataset.createVariable("BZ", "f8", ("nR", "nphi", "nZ"))[:] = BZ
        dataset.createVariable("absB", "f8", ("nR", "nphi", "nZ"))[:] = absB
        dataset.setncattr("format", "extended_field")
        dataset.setncattr("coordinate_convention", "physical cylindrical (R, phi, Z)")
        dataset.setncattr("field_components", "BR,Bphi,BZ")
        dataset.setncattr("nfp", nfp)
        dataset.setncattr("source", "synthetic_campaign_grid")
        dataset.setncattr("src_nphi", 8)
        dataset.setncattr("src_ntheta", 8)
        dataset.setncattr("digits", 8)
        dataset.setncattr("branch", "internal")
        dataset.setncattr("units", "SI")
    return path


def test_vmec_extender_edge_field_campaign_report_passes_synthetic_grid(tmp_path: Path) -> None:
    report = build_vmec_extender_edge_field_campaign_report(
        field_grid_path=_write_campaign_grid(tmp_path / "vmec_extender_field.nc")
    )

    assert report["family"] == "vmec_extender_edge_field_campaign"
    assert report["source"] == "synthetic_campaign_grid"
    assert report["grid_shape"] == [3, 5, 3]
    assert report["nfp"] == 5
    assert report["metadata_passed"] is True
    assert report["passed"] is True
    assert report["node_interpolation_max_abs_error"] < 1.0e-12
    assert report["midpoint_interpolation_max_abs_error"] < 1.0e-12
    assert report["field_period_relative_l2"] < 1.0e-12
    assert report["fieldline_rhs_max_abs_error"] < 1.0e-12


def test_vmec_extender_edge_field_campaign_package_writes_artifacts(tmp_path: Path) -> None:
    artifacts = create_vmec_extender_edge_field_campaign_package(
        output_root=tmp_path / "artifacts",
        field_grid_path=_write_campaign_grid(tmp_path / "vmec_extender_field.nc"),
    )

    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    payload = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    arrays = np.load(artifacts.arrays_npz_path, allow_pickle=True)
    assert "metric_values" in arrays
    assert "node_points" in arrays
    assert arrays["node_interpolated_B"].shape == arrays["node_expected_B"].shape
