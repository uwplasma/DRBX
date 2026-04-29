from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import create_essos_vmec_fieldline_surface_package


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trace ESSOS Landreman-Paul QA coil field lines from scaled VMEC seed surfaces "
            "and compare Poincare hits with the same VMEC surfaces."
        ),
    )
    parser.add_argument("--coil-json", type=Path, default=None, help="Optional ESSOS Landreman-Paul QA coil JSON path.")
    parser.add_argument("--vmec-wout", type=Path, default=None, help="Optional Landreman-Paul QA VMEC wout path.")
    parser.add_argument("--essos-root", type=Path, default=None, help="Optional ESSOS checkout root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/essos_vmec_fieldline_surface_artifacts"),
        help="Directory where JSON/NPZ/PNG validation artifacts are written.",
    )
    parser.add_argument("--rho-min", type=float, default=0.20, help="Innermost scaled VMEC seed radius.")
    parser.add_argument("--rho-max", type=float, default=0.92, help="Outermost scaled VMEC seed radius.")
    parser.add_argument("--n-surfaces", type=int, default=7, help="Number of VMEC surfaces to seed and compare.")
    parser.add_argument("--ntheta-surface", type=int, default=320, help="Poloidal samples on each reference surface.")
    parser.add_argument("--times-to-trace", type=int, default=4200, help="ESSOS trace samples per field line.")
    parser.add_argument("--maxtime", type=float, default=900.0, help="ESSOS field-line integration time.")
    args = parser.parse_args()

    configure_jax_runtime(precision="float64")
    artifacts = create_essos_vmec_fieldline_surface_package(
        output_root=args.output_root,
        coil_json_path=args.coil_json,
        vmec_wout_path=args.vmec_wout,
        essos_root=args.essos_root,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        n_surfaces=args.n_surfaces,
        ntheta_surface=args.ntheta_surface,
        times_to_trace=args.times_to_trace,
        maxtime=args.maxtime,
    )
    print(f"wrote report: {artifacts.report_json_path}")
    print(f"wrote arrays: {artifacts.arrays_npz_path}")
    print(f"wrote plot: {artifacts.plot_png_path}")


if __name__ == "__main__":
    main()
