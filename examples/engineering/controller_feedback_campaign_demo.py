from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.reference.paths import default_reference_root
from jax_drb.validation import create_controller_feedback_campaign_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/controller_feedback_campaign_artifacts"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=default_reference_root(),
    )
    args = parser.parse_args()
    if args.reference_root is None:
        raise SystemExit("reference root is required for the controller feedback campaign")

    artifacts = create_controller_feedback_campaign_package(
        output_root=args.output_root,
        reference_root=args.reference_root,
    )
    print("== Controller Feedback Campaign ==")
    print(f"  - summary_json: {artifacts.summary_json_path}")
    print(f"  - arrays_npz: {artifacts.arrays_npz_path}")
    print(f"  - plot_png: {artifacts.plot_png_path}")


if __name__ == "__main__":
    main()
