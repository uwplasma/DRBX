from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import create_essos_imported_fci_campaign_package


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trace a scaled VMEC QA seed grid with ESSOS, import the resulting field-line "
            "maps, and run JAXDRB FCI sheath/recycling and neutral validation gates."
        ),
    )
    parser.add_argument(
        "--coil-json",
        type=Path,
        default=None,
        help="Path to ESSOS_biot_savart_LandremanPaulQA.json. If omitted, JAX_DRB_ESSOS_ROOT is used.",
    )
    parser.add_argument(
        "--essos-root",
        type=Path,
        default=None,
        help="ESSOS checkout used only for the field-line import step.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/essos_imported_fci_artifacts"),
        help="Directory where JSON/NPZ/PNG validation artifacts are written.",
    )
    parser.add_argument("--nx", type=int, default=5, help="Number of VMEC-shaped radial grid points.")
    parser.add_argument("--ny", type=int, default=8, help="Number of toroidal planes.")
    parser.add_argument("--nz", type=int, default=20, help="Number of poloidal grid points.")
    parser.add_argument("--rho-min", type=float, default=0.12, help="Inner logical minor radius of the imported VMEC-shaped shell.")
    parser.add_argument("--rho-max", type=float, default=0.34, help="Outer logical minor radius of the imported VMEC-shaped shell.")
    parser.add_argument("--times-to-trace", type=int, default=360, help="ESSOS trace samples per seed.")
    parser.add_argument("--maxtime", type=float, default=80.0, help="ESSOS field-line integration time.")
    args = parser.parse_args()

    configure_jax_runtime(precision="float64")
    artifacts = create_essos_imported_fci_campaign_package(
        output_root=args.output_root,
        coil_json_path=args.coil_json,
        essos_root=args.essos_root,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        maxtime=args.maxtime,
        times_to_trace=args.times_to_trace,
    )
    print(f"wrote report: {artifacts.report_json_path}")
    print(f"wrote arrays: {artifacts.arrays_npz_path}")
    print(f"wrote plot: {artifacts.plot_png_path}")


if __name__ == "__main__":
    main()
