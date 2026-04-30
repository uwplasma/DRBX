from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import create_essos_imported_pytree_campaign_package


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trace an ESSOS scaled VMEC QA seed grid, convert it into FCI maps, and run "
            "the JAXDRB fixed-layout PyTree/JVP RHS gate on the imported maps."
        ),
    )
    parser.add_argument("--coil-json", type=Path, default=None, help="Optional ESSOS Landreman-Paul QA coil JSON path.")
    parser.add_argument(
        "--vmec-wout",
        type=Path,
        default=None,
        help="Optional Landreman-Paul QA VMEC wout path used by VMEC and hybrid map sources.",
    )
    parser.add_argument("--essos-root", type=Path, default=None, help="Optional ESSOS checkout root.")
    parser.add_argument(
        "--map-source",
        choices=("coil", "vmec", "hybrid"),
        default="coil",
        help="FCI map source: coil trace, VMEC-coordinate map, or VMEC map with coil target masks.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/essos_imported_pytree_artifacts"),
        help="Directory where JSON/NPZ/PNG validation artifacts are written.",
    )
    parser.add_argument("--case-label", default="essos_imported_pytree_campaign", help="Artifact filename stem.")
    parser.add_argument("--nx", type=int, default=4, help="Number of VMEC-shaped radial grid points.")
    parser.add_argument("--ny", type=int, default=6, help="Number of toroidal planes.")
    parser.add_argument("--nz", type=int, default=12, help="Number of poloidal grid points.")
    parser.add_argument("--rho-min", type=float, default=0.12, help="Inner logical minor radius of the imported VMEC-shaped shell.")
    parser.add_argument("--rho-max", type=float, default=0.34, help="Outer logical minor radius of the imported VMEC-shaped shell.")
    parser.add_argument("--times-to-trace", type=int, default=280, help="ESSOS trace samples per seed.")
    parser.add_argument("--maxtime", type=float, default=60.0, help="ESSOS field-line integration time.")
    parser.add_argument("--steps", type=int, default=5, help="Short fixed-layout PyTree transient steps.")
    args = parser.parse_args()

    configure_jax_runtime(precision="float64")
    artifacts = create_essos_imported_pytree_campaign_package(
        output_root=args.output_root,
        case_label=args.case_label,
        coil_json_path=args.coil_json,
        vmec_wout_path=args.vmec_wout,
        essos_root=args.essos_root,
        map_source=args.map_source,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        maxtime=args.maxtime,
        times_to_trace=args.times_to_trace,
        steps=args.steps,
    )
    print(f"wrote report: {artifacts.report_json_path}")
    print(f"wrote arrays: {artifacts.arrays_npz_path}")
    print(f"wrote plot: {artifacts.plot_png_path}")


if __name__ == "__main__":
    main()
