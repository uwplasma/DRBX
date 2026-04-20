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
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--reference-binary", type=Path, default=None)
    parser.add_argument("--nout", type=int, default=4)
    parser.add_argument("--timestep", type=float, default=100.0)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--solver-type", type=str, default="cvode")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    artifacts = create_temperature_feedback_campaign_package(
        output_root=args.output_root,
        reference_root=args.reference_root,
        reference_binary=args.reference_binary,
        nout=args.nout,
        timestep=args.timestep,
        ny=args.ny,
        solver_type=args.solver_type,
        timeout_seconds=args.timeout_seconds,
    )
    print("== Temperature Feedback Campaign ==")
    print(f"  - summary_json: {artifacts.summary_json_path}")
    print(f"  - arrays_npz: {artifacts.arrays_npz_path}")
    print(f"  - plot_png: {artifacts.plot_png_path}")


if __name__ == "__main__":
    main()
