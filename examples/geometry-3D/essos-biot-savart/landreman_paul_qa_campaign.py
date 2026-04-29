from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import create_essos_biot_savart_campaign_package


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the coil-field annular FCI turbulence campaign from an ESSOS Fourier-coil JSON file.",
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
        default=Path("docs/data/essos_biot_savart_landreman_paul_qa_artifacts"),
        help="Directory where JSON/NPZ/PNG/GIF campaign artifacts are written.",
    )
    parser.add_argument("--nx", type=int, default=14, help="Minor-radius cells for each annular campaign grid.")
    parser.add_argument("--ny", type=int, default=18, help="Toroidal planes for each annular campaign grid.")
    parser.add_argument("--nz", type=int, default=28, help="Poloidal cells for each annular campaign grid.")
    args = parser.parse_args()

    configure_jax_runtime(precision="float64")
    artifacts = create_essos_biot_savart_campaign_package(
        output_root=args.output_root,
        coil_json_path=args.coil_json,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
    )
    print(f"wrote report: {artifacts.report_json_path}")
    print(f"wrote arrays: {artifacts.arrays_npz_path}")
    print(f"wrote plot: {artifacts.plot_png_path}")
    print(f"wrote field-line plot: {artifacts.field_line_png_path}")
    print(f"wrote movie: {artifacts.movie_gif_path}")


if __name__ == "__main__":
    main()
