from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.native import run_curated_case
from jax_drb.parity.arrays import (
    build_array_payload_from_summary_payload,
    load_portable_array_payload,
    write_portable_array_payload,
)
from jax_drb.validation import create_blob2d_meeting_package


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a meeting-ready Blob2D package with movies and publication figures."
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/Users/rogerio/local/hermes-3"),
        help="Hermes-3 checkout used to run the native curated case.",
    )
    parser.add_argument(
        "--case-name",
        default="blob2d_short_window",
        help="Curated case name to run. Defaults to blob2d_short_window.",
    )
    parser.add_argument(
        "--arrays-in",
        type=Path,
        default=None,
        help="Use an existing portable array payload instead of running the curated case.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_repo_root() / "docs",
        help="Root directory where docs/images, docs/movies, and docs/data artifacts will be written.",
    )
    parser.add_argument(
        "--native-arrays-out",
        type=Path,
        default=None,
        help="Optional path for the native portable array payload. Defaults to docs/data/<case>_native.npz.",
    )
    parser.add_argument(
        "--reference-metrics-in",
        type=Path,
        default=_default_repo_root() / "references" / "baselines" / "reference_metrics" / "blob2d_short_window_metrics.json",
        help="Reference Blob2D metrics JSON for the parity figure.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Movie frame rate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = _default_repo_root()

    if args.arrays_in is not None:
        payload = load_portable_array_payload(args.arrays_in)
        native_arrays_path = args.arrays_in
    else:
        result = run_curated_case(args.case_name, reference_root=args.reference_root)
        payload = build_array_payload_from_summary_payload(result.payload, result.variables)
        native_arrays_path = (
            args.native_arrays_out
            if args.native_arrays_out is not None
            else args.output_root / "data" / f"{args.case_name}_native.npz"
        )
        write_portable_array_payload(payload, native_arrays_path)

    artifacts = create_blob2d_meeting_package(
        payload,
        output_root=args.output_root,
        native_arrays_path=native_arrays_path,
        reference_metrics_path=args.reference_metrics_in,
        case_label="blob2d_meeting",
        fps=args.fps,
    )

    print(f"repo_root: {repo_root}")
    print(f"native_arrays: {artifacts.native_arrays_path}")
    print(f"analysis_json: {artifacts.analysis_json_path}")
    print(f"parity_json: {artifacts.parity_json_path}")
    print(f"snapshots_png: {artifacts.snapshots_png_path}")
    print(f"parity_png: {artifacts.parity_png_path}")
    print(f"poster_png: {artifacts.poster_png_path}")
    print(f"movie_2d: {artifacts.movie_2d_path}")
    print(f"movie_3d: {artifacts.movie_3d_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
