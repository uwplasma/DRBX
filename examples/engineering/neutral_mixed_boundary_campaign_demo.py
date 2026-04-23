from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.reference.paths import default_reference_root
from jax_drb.validation import create_neutral_mixed_boundary_campaign_package


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Rerun the neutral mixed one-step Hermès comparison and write the "
            "JSON, NPZ, and publication-grade PNG artifacts."
        )
    )
    parser.add_argument("--reference-root", type=Path, default=default_reference_root())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "docs" / "data" / "neutral_mixed_boundary_campaign_artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_root = args.reference_root
    if reference_root is None:
        raise SystemExit("Set --reference-root or JAX_DRB_REFERENCE_ROOT to a Hermes/reference-suite checkout.")
    artifacts = create_neutral_mixed_boundary_campaign_package(
        reference_root=reference_root,
        output_root=args.output_root,
    )
    print(artifacts.report_json_path)
    print(artifacts.report_npz_path)
    print(artifacts.report_plot_png_path)


if __name__ == "__main__":
    main()
