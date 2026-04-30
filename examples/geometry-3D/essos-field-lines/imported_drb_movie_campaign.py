from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import create_essos_imported_drb_movie_package


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trace an ESSOS Landreman-Paul QA coil field on scaled VMEC QA surfaces, import the field-line maps, "
            "and render a reduced JAXDRB DRB transient movie with sheath, recycling, "
            "and neutral closures."
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
        default=Path("docs/data/essos_imported_drb_movie_artifacts"),
        help="Directory where JSON/NPZ/PNG/GIF validation artifacts are written.",
    )
    parser.add_argument("--case-label", default="essos_imported_drb_movie_campaign", help="Artifact filename stem.")
    parser.add_argument("--nx", type=int, default=8, help="Number of VMEC-shaped radial grid points.")
    parser.add_argument("--ny", type=int, default=28, help="Number of toroidal planes.")
    parser.add_argument("--nz", type=int, default=80, help="Number of poloidal grid points.")
    parser.add_argument("--rho-min", type=float, default=0.20, help="Inner logical minor radius of the imported VMEC-shaped shell.")
    parser.add_argument("--rho-max", type=float, default=0.92, help="Outer logical minor radius of the imported VMEC-shaped shell.")
    parser.add_argument("--times-to-trace", type=int, default=720, help="ESSOS trace samples per seed.")
    parser.add_argument("--maxtime", type=float, default=135.0, help="ESSOS field-line integration time.")
    parser.add_argument("--frames", type=int, default=32, help="Number of saved movie frames.")
    parser.add_argument("--substeps-per-frame", type=int, default=6, help="JAXDRB substeps between movie frames.")
    parser.add_argument("--dt", type=float, default=1.2e-3, help="Explicit reduced transient substep.")
    args = parser.parse_args()

    configure_jax_runtime(precision="float64")
    artifacts = create_essos_imported_drb_movie_package(
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
        frames=args.frames,
        substeps_per_frame=args.substeps_per_frame,
        dt=args.dt,
    )
    print(f"wrote report: {artifacts.report_json_path}")
    print(f"wrote arrays: {artifacts.arrays_npz_path}")
    print(f"wrote snapshots: {artifacts.snapshot_png_path}")
    print(f"wrote diagnostics: {artifacts.diagnostics_png_path}")
    print(f"wrote poster: {artifacts.poster_png_path}")
    print(f"wrote movie: {artifacts.movie_gif_path}")


if __name__ == "__main__":
    main()
