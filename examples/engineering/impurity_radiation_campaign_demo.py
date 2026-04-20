from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_impurity_radiation_campaign_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/impurity_radiation_campaign_artifacts"),
    )
    args = parser.parse_args()

    artifacts = create_impurity_radiation_campaign_package(output_root=args.output_root)
    print("== Impurity / Radiation Campaign ==")
    print(f"  - summary_json: {artifacts.summary_json_path}")
    print(f"  - arrays_npz: {artifacts.arrays_npz_path}")
    print(f"  - plot_png: {artifacts.plot_png_path}")


if __name__ == "__main__":
    main()
