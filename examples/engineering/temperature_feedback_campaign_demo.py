from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import create_temperature_feedback_campaign_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("docs/data/temperature_feedback_campaign_artifacts"),
    )
    parser.add_argument("--nout", type=int, default=4)
    parser.add_argument("--timestep", type=float, default=100.0)
    parser.add_argument("--ny", type=int, default=80)
    args = parser.parse_args()

    artifacts = create_temperature_feedback_campaign_package(
        output_root=args.output_root,
        nout=args.nout,
        timestep=args.timestep,
        ny=args.ny,
    )
    print("== Temperature Feedback Campaign ==")
    print(f"  - summary_json: {artifacts.summary_json_path}")
    print(f"  - arrays_npz: {artifacts.arrays_npz_path}")
    print(f"  - plot_png: {artifacts.plot_png_path}")


if __name__ == "__main__":
    main()
