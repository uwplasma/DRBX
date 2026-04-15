from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_publication_ready_3d_campaign_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the reviewer-facing 3D publication campaign bundle.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/publication_ready_3d_artifacts"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = create_publication_ready_3d_campaign_package(output_root=args.output_root)
    print("== Publication-Ready 3D Campaign ==")
    print(f"  - summary_json: {artifacts.summary_json_path}")
    print(f"  - summary_plot_png: {artifacts.summary_plot_png_path}")


if __name__ == "__main__":
    main()
