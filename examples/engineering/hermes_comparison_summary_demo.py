from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_hermes_comparison_summary_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/hermes_comparison_summary_artifacts"),
    )
    args = parser.parse_args()
    artifacts = create_hermes_comparison_summary_package(output_root=args.output_root)
    print("== Hermes Comparison Summary ==")
    print(f"  - summary_json: {artifacts.summary_json_path}")
    print(f"  - summary_plot_png: {artifacts.summary_plot_png_path}")


if __name__ == "__main__":
    main()
