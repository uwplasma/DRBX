from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.native import run_curated_case
from jax_drb.parity.arrays import (
    build_array_payload_from_summary_payload,
    load_portable_array_payload,
    write_portable_array_payload,
)
from jax_drb.validation import create_alfven_wave_meeting_package


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Generate a meeting-ready Alfven-wave package with movies and publication figures."
    )
    parser.add_argument("--reference-root", type=Path, default=Path("/Users/rogerio/local/hermes-3"))
    parser.add_argument("--case-name", default="alfven_wave_short_window")
    parser.add_argument("--field-variable", default="phi")
    parser.add_argument("--x-index", type=int, default=2)
    parser.add_argument("--arrays-in", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=repo_root / "docs")
    parser.add_argument("--native-arrays-out", type=Path, default=None)
    parser.add_argument(
        "--expected-arrays-in",
        type=Path,
        default=repo_root / "references" / "baselines" / "reference_arrays" / "alfven_wave_short_window.npz",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("/Users/rogerio/local/hermes-3/tests/integrated/alfven-wave/data/BOUT.inp"),
    )
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

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

    artifacts = create_alfven_wave_meeting_package(
        payload,
        input_file=args.input_file,
        expected_arrays_path=args.expected_arrays_in,
        native_arrays_path=native_arrays_path,
        output_root=args.output_root,
        field_variable=args.field_variable,
        x_index=args.x_index,
        case_label="alfven_wave_meeting",
        fps=args.fps,
    )

    print(f"native_arrays: {artifacts.native_arrays_path}")
    print(f"analysis_json: {artifacts.analysis_json_path}")
    print(f"parity_json: {artifacts.parity_json_path}")
    print(f"snapshots_png: {artifacts.snapshots_png_path}")
    print(f"diagnostics_png: {artifacts.diagnostics_png_path}")
    print(f"parity_png: {artifacts.parity_png_path}")
    print(f"poster_png: {artifacts.poster_png_path}")
    print(f"movie_2d: {artifacts.movie_2d_path}")
    print(f"movie_3d: {artifacts.movie_3d_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
