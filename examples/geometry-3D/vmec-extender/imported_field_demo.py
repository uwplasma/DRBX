from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jax_drb.validation import create_vmec_extender_edge_field_campaign_package


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a small synthetic VMEC-extender-style field grid, import it "
            "through the JAXDRB edge-field contract, and write validation artifacts."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/vmec_extender_edge_field_artifacts"),
        help="Directory where JSON, NPZ, and PNG validation artifacts are written.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_root / "synthetic_vmec_extender_field.nc"
    write_synthetic_vmec_extender_grid(grid_path)
    artifacts = create_vmec_extender_edge_field_campaign_package(
        output_root=args.output_root,
        field_grid_path=grid_path,
    )
    print(f"summary: {artifacts.summary_json_path}")
    print(f"arrays:  {artifacts.arrays_npz_path}")
    print(f"plot:    {artifacts.plot_png_path}")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
