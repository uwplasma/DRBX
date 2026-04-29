from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import create_essos_fieldline_import_package


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import ESSOS-produced Landreman-Paul QA fields and field lines into jax_drb artifacts.",
    )
    parser.add_argument(
        "--coil-json",
        type=Path,
        default=None,
        help="Path to ESSOS_biot_savart_LandremanPaulQA.json. If omitted, JAX_DRB_ESSOS_ROOT is used.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/essos_fieldline_import_artifacts"),
        help="Directory where JSON/NPZ/PNG import artifacts are written.",
    )
    parser.add_argument("--n-field-lines", type=int, default=8, help="Number of ESSOS field lines to trace.")
    parser.add_argument("--times-to-trace", type=int, default=6000, help="ESSOS trace samples per field line.")
    parser.add_argument("--maxtime", type=float, default=1000.0, help="ESSOS field-line integration time.")
    parser.add_argument("--r-min", type=float, default=1.21, help="Minimum seed major radius.")
    parser.add_argument("--r-max", type=float, default=1.40, help="Maximum seed major radius.")
    args = parser.parse_args()

    configure_jax_runtime(precision="float64")
    artifacts = create_essos_fieldline_import_package(
        output_root=args.output_root,
        coil_json_path=args.coil_json,
        r_min=args.r_min,
        r_max=args.r_max,
        n_field_lines=args.n_field_lines,
        maxtime=args.maxtime,
        times_to_trace=args.times_to_trace,
    )
    print(f"wrote report: {artifacts.report_json_path}")
    print(f"wrote arrays: {artifacts.arrays_npz_path}")
    print(f"wrote plot: {artifacts.plot_png_path}")


if __name__ == "__main__":
    main()
