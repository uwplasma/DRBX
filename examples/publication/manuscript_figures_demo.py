from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_manuscript_figure_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the remaining manuscript schematic and transient-panel figures.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/manuscript_figures_artifacts"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = create_manuscript_figure_package(output_root=args.output_root)
    print("== Manuscript Figures ==")
    print(f"  - manifest_json: {artifacts.manifest_json_path}")
    print(f"  - architecture_png: {artifacts.architecture_png_path}")
    print(f"  - equations_geometry_png: {artifacts.equations_geometry_png_path}")
    print(f"  - transient_panel_png: {artifacts.transient_panel_png_path}")


if __name__ == "__main__":
    main()
