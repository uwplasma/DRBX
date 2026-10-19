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
``SOL_GRID_PATH`` at it and drop the corresponding ``write_extended_field_grid`` call.

It prints the summary/arrays/plot paths of both packages; artifacts land under
``docs/data/vmec_extender_edge_field_artifacts`` (relative to the current
working directory).

Run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/vmec-extender/imported_field.py
"""

from __future__ import annotations

import time
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
NFP = 5


def write_extended_field_grid(path: Path, R, phi, Z, BR, Bphi, BZ, **attributes) -> Path:
    """Write cylindrical (R, phi, Z) field arrays in the VMEC-extender ``extended_field`` NetCDF format."""

    with Dataset(path, "w") as dataset:
        for name, axis in (("R", R), ("phi", phi), ("Z", Z)):
            dataset.createDimension(f"n{name}", axis.size)
            dataset.createVariable(name, "f8", (f"n{name}",))[:] = axis
        absB = np.sqrt(BR * BR + Bphi * Bphi + BZ * BZ)
        for name, values in (("BR", BR), ("Bphi", Bphi), ("BZ", BZ), ("absB", absB)):
            dataset.createVariable(name, "f8", ("nR", "nphi", "nZ"))[:] = values
        dataset.setncatts({"format": "extended_field", "coordinate_convention": "physical cylindrical (R, phi, Z)",
                           "field_components": "BR,Bphi,BZ", "nfp": NFP, "src_ntheta": 8, "digits": 8,
                           "branch": "internal", "units": "SI", **attributes})
    return path


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
period = 2.0 * np.pi / NFP
# edge-field gate: a tiny 3x5x3 grid with a smooth, non-trivial linear field
R, phi, Z = np.array([1.0, 1.3, 1.7]), np.linspace(0.0, period, 5, endpoint=False), np.array([-0.4, 0.1, 0.6])
RR, PP, ZZ = np.meshgrid(R, phi, Z, indexing="ij")
write_extended_field_grid(EDGE_GRID_PATH, R, phi, Z, RR + 2.0 * PP + 3.0 * ZZ, 2.0 + RR, RR - PP + ZZ,
                          source="synthetic_vmec_extender_demo", src_nphi=8)
print(f"wrote synthetic edge grid {RR.shape}: {EDGE_GRID_PATH}")
start = time.perf_counter()
edge_artifacts = create_vmec_extender_edge_field_campaign_package(output_root=OUTPUT_ROOT, field_grid_path=EDGE_GRID_PATH)
print(f"edge-field gate done in {time.perf_counter() - start:.1f} s")
# SOL smoke gate: a purely toroidal 8x12x8 field
R, phi, Z = np.linspace(1.15, 1.65, 8), np.linspace(0.0, period, 12, endpoint=False), np.linspace(-0.36, 0.36, 8)
zeros = np.zeros((R.size, phi.size, Z.size))
write_extended_field_grid(SOL_GRID_PATH, R, phi, Z, zeros, zeros + 2.1, zeros, phi_period=period,
                          source="synthetic_vmec_extender_toroidal_sol_smoke_demo", src_nphi=12)
print(f"wrote synthetic toroidal grid {zeros.shape}: {SOL_GRID_PATH}")
start = time.perf_counter()
sol_artifacts = create_vmec_extender_sol_smoke_package(output_root=OUTPUT_ROOT, field_grid_path=SOL_GRID_PATH)
print(f"SOL smoke gate done in {time.perf_counter() - start:.1f} s")

print(f"edge summary: {edge_artifacts.summary_json_path}")
print(f"edge arrays:  {edge_artifacts.arrays_npz_path}")
print(f"edge plot:    {edge_artifacts.plot_png_path}")
print(f"sol summary:  {sol_artifacts.summary_json_path}")
print(f"sol arrays:   {sol_artifacts.arrays_npz_path}")
print(f"sol plot:     {sol_artifacts.plot_png_path}")
