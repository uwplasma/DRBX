from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_autodiff_diffusion_uncertainty_package


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a publication-style uncertainty-quantification demonstration on the "
            "native differentiable diffusion lane."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_repo_root() / "docs" / "data" / "autodiff_diffusion_uncertainty_artifacts",
    )
    parser.add_argument("--sample-count", type=int, default=96)
    parser.add_argument("--random-seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = create_autodiff_diffusion_uncertainty_package(
        output_root=args.output_root,
        sample_count=args.sample_count,
        random_seed=args.random_seed,
    )
    print("== Autodiff Diffusion Uncertainty ==")
    print(f"  - analysis_json: {artifacts.analysis_json_path}")
    print(f"  - arrays_npz: {artifacts.arrays_npz_path}")
    print(f"  - plot_png: {artifacts.plot_png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
