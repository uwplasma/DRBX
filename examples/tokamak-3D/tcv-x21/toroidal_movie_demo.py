from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_tcv_x21_toroidal_movie_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a toroidal 3D tokamak movie from the committed TCV-X21 scaffold arrays.")
    parser.add_argument(
        "--arrays",
        type=Path,
        default=Path("docs/data/tokamak_tcv_x21_scaffold_artifacts/data/tokamak_tcv_x21_scaffold_arrays.npz"),
        help="Input scaffold arrays NPZ.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/tokamak_tcv_x21_toroidal_movie_artifacts"),
        help="Output artifact directory.",
    )
    args = parser.parse_args()

    artifacts = create_tcv_x21_toroidal_movie_package(
        arrays_npz_path=args.arrays,
        output_root=args.output_root,
    )
    print(f"arrays: {artifacts.arrays_npz_path}")
    print(f"summary: {artifacts.summary_json_path}")
    print(f"poster: {artifacts.poster_png_path}")
    print(f"movie: {artifacts.movie_gif_path}")


if __name__ == "__main__":
    main()
