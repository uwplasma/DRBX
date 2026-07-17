"""VMEC-extender field-grid import and compact SOL verification gate.

The script writes two tiny synthetic NetCDF field grids in the VMEC-extender
``extended_field`` format (cylindrical ``R``/``phi``/``Z`` axes with
``BR``/``Bphi``/``BZ`` components) so it runs on a fresh clone without any
external VMEC data, then feeds them to the two public campaign packages:

1. ``create_vmec_extender_edge_field_campaign_package`` -- the edge-field
   import/verification gate (grid parsing, field interpolation, FCI map QA);
2. ``create_vmec_extender_sol_smoke_package`` -- the compact toroidal SOL
   smoke gate on the imported field.

To import a real VMEC-extender file instead, point ``EDGE_GRID_PATH`` or
``SOL_GRID_PATH`` at it and drop the corresponding ``write_synthetic_*`` call.

It prints the summary/arrays/plot paths of both packages; artifacts land under
``docs/data/vmec_extender_edge_field_artifacts`` (relative to the current
working directory).

Run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/vmec-extender/imported_field.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from drbx.validation import (
    create_vmec_extender_edge_field_campaign_package,
    create_vmec_extender_sol_smoke_package,
)

# --- PARAMETERS ------------------------------------------------------------------
OUTPUT_ROOT = Path("docs/data/vmec_extender_edge_field_artifacts")  # artifact root (cwd-relative)
EDGE_GRID_PATH = OUTPUT_ROOT / "synthetic_vmec_extender_field.nc"
SOL_GRID_PATH = OUTPUT_ROOT / "synthetic_vmec_extender_toroidal_field.nc"


def write_synthetic_vmec_extender_grid(path: Path) -> Path:
    nfp = 5
    phi_period = 2.0 * np.pi / float(nfp)
    R = np.asarray([1.0, 1.3, 1.7], dtype=np.float64)
    phi = np.linspace(0.0, phi_period, 5, endpoint=False, dtype=np.float64)
    Z = np.asarray([-0.4, 0.1, 0.6], dtype=np.float64)
    RR, PP, ZZ = np.meshgrid(R, phi, Z, indexing="ij")
    BR = RR + 2.0 * PP + 3.0 * ZZ
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
        dataset.setncattr("source", "synthetic_vmec_extender_demo")
        dataset.setncattr("src_nphi", 8)
        dataset.setncattr("src_ntheta", 8)
        dataset.setncattr("digits", 8)
        dataset.setncattr("branch", "internal")
        dataset.setncattr("units", "SI")
    return path


def write_synthetic_toroidal_vmec_extender_grid(path: Path) -> Path:
    nfp = 5
    phi_period = 2.0 * np.pi / float(nfp)
    R = np.linspace(1.15, 1.65, 8, dtype=np.float64)
    phi = np.linspace(0.0, phi_period, 12, endpoint=False, dtype=np.float64)
    Z = np.linspace(-0.36, 0.36, 8, dtype=np.float64)
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
        dataset.setncattr("coordinate_convention", "physical cylindrical (R, phi, Z)")
        dataset.setncattr("field_components", "BR,Bphi,BZ")
        dataset.setncattr("nfp", nfp)
        dataset.setncattr("phi_period", phi_period)
        dataset.setncattr("source", "synthetic_vmec_extender_toroidal_sol_smoke_demo")
        dataset.setncattr("src_nphi", 12)
        dataset.setncattr("src_ntheta", 8)
        dataset.setncattr("digits", 8)
        dataset.setncattr("branch", "internal")
        dataset.setncattr("units", "SI")
    return path


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
write_synthetic_vmec_extender_grid(EDGE_GRID_PATH)
edge_artifacts = create_vmec_extender_edge_field_campaign_package(
    output_root=OUTPUT_ROOT,
    field_grid_path=EDGE_GRID_PATH,
)
write_synthetic_toroidal_vmec_extender_grid(SOL_GRID_PATH)
sol_artifacts = create_vmec_extender_sol_smoke_package(
    output_root=OUTPUT_ROOT,
    field_grid_path=SOL_GRID_PATH,
)

print(f"edge summary: {edge_artifacts.summary_json_path}")
print(f"edge arrays:  {edge_artifacts.arrays_npz_path}")
print(f"edge plot:    {edge_artifacts.plot_png_path}")
print(f"sol summary:  {sol_artifacts.summary_json_path}")
print(f"sol arrays:   {sol_artifacts.arrays_npz_path}")
print(f"sol plot:     {sol_artifacts.plot_png_path}")
