from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from drbx.geometry.vmec_extender_import import load_vmec_extender_grid_netcdf
from drbx.validation.vmec_extender_sol_smoke_campaign import (
    build_vmec_extender_sol_smoke_report,
    create_vmec_extender_sol_smoke_package,
    simulate_vmec_extender_scalar_sol_smoke,
)


def test_vmec_extender_sol_smoke_evolves_on_imported_toroidal_maps(tmp_path: Path) -> None:
    grid = load_vmec_extender_grid_netcdf(_write_toroidal_grid(tmp_path / "field.nc"))

    result = simulate_vmec_extender_scalar_sol_smoke(grid, frames=8, substeps_per_frame=2)
    report = build_vmec_extender_sol_smoke_report(grid, result)

    assert report["family"] == "vmec_extender_sol_smoke"
    assert report["passed"] is True
    assert report["final_min"] > 0.0
    assert report["final_max"] < 2.0
    assert report["final_rms_fluctuation"] > 1.0e-4
    assert report["fci_map_identity_max_abs_error"] < 1.0e-12
    assert report["parallel_mode_decay_relative_error"] < 1.0e-12
    assert result.history.shape == (8, 6, 10, 6)
    assert result.source.shape == (6, 10, 6)


def test_vmec_extender_sol_smoke_package_writes_public_artifacts(tmp_path: Path) -> None:
    artifacts = create_vmec_extender_sol_smoke_package(
        output_root=tmp_path / "artifacts",
        field_grid_path=_write_toroidal_grid(tmp_path / "field.nc"),
    )

    assert artifacts.summary_json_path.exists()
    assert artifacts.arrays_npz_path.exists()
    assert artifacts.plot_png_path.exists()
    report = json.loads(artifacts.summary_json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    arrays = np.load(artifacts.arrays_npz_path)
    assert arrays["history"].shape[1:] == (6, 10, 6)
    assert arrays["endpoint_mask"].dtype == np.bool_


def _write_toroidal_grid(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    R = np.linspace(1.15, 1.65, 6)
    nfp = 5
    phi_period = 2.0 * np.pi / nfp
    phi = np.linspace(0.0, phi_period, 10, endpoint=False)
    Z = np.linspace(-0.32, 0.32, 6)
    shape = (R.size, phi.size, Z.size)
    BR = np.zeros(shape, dtype=np.float64)
    Bphi = np.ones(shape, dtype=np.float64) * 2.1
    BZ = np.zeros(shape, dtype=np.float64)
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
        dataset.setncattr("coordinate_convention", "physical cylindrical R, phi, Z")
        dataset.setncattr("field_components", "BR,Bphi,BZ")
        dataset.setncattr("nfp", nfp)
        dataset.setncattr("phi_period", phi_period)
        dataset.setncattr("source", "synthetic_vmec_extender_sol_smoke_test")
        dataset.setncattr("branch", "internal")
        dataset.setncattr("units", "SI")
    return path
