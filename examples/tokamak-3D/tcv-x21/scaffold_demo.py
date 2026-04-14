from __future__ import annotations

import argparse
from pathlib import Path

from jax_drb.validation import (
    DEFAULT_TCV_X21_CASE_NAME,
    create_tcv_x21_scaffold_package,
    resolve_tcv_x21_reference_case,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the first honest TCV-X21 3D tokamak scaffold package. "
            "The default mode uses a tiny synthetic preview workdir, while a local "
            "TCV-X21 reference checkout can be bound in through --reference-root."
        )
    )
    parser.add_argument("--reference-root", type=Path, default=_repo_root())
    parser.add_argument("--case-name", default=DEFAULT_TCV_X21_CASE_NAME)
    parser.add_argument("--field-name", default="phi")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_repo_root() / "docs" / "data" / "tokamak_tcv_x21_scaffold_artifacts",
    )
    parser.add_argument("--workdir-in", type=Path, default=None)
    parser.add_argument("--mesh-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frames-per-interval", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    resolved = resolve_tcv_x21_reference_case(args.reference_root, case_name=args.case_name)
    mesh_path = args.mesh_path
    if mesh_path is None and args.workdir_in is not None:
        inferred = args.workdir_in / "tokamak.nc"
        if inferred.exists():
            mesh_path = inferred

    if verbose:
        _print_section("TCV-X21 Scaffold")
        _print_kv(
            {
                "reference_root": args.reference_root,
                "case_name": args.case_name,
                "reference_input_path": resolved.input_path,
                "reference_exists": resolved.exists,
                "output_root": args.output_root,
                "workdir_in": args.workdir_in if args.workdir_in is not None else "<synthetic preview>",
                "mesh_path": mesh_path if mesh_path is not None else "<synthetic preview>",
                "field_name": args.field_name,
                "fps": args.fps,
                "frames_per_interval": args.frames_per_interval,
            }
        )

    artifacts = create_tcv_x21_scaffold_package(
        reference_root=args.reference_root,
        output_root=args.output_root,
        case_name=args.case_name,
        field_name=args.field_name,
        workdir_in=args.workdir_in,
        mesh_path=mesh_path,
        fps=args.fps,
        frames_per_interval=args.frames_per_interval,
    )

    if verbose:
        _print_section("Artifacts")
        _print_kv(
            {
                "manifest_json": artifacts.manifest_json_path,
                "input_report_json": artifacts.input_report_json_path,
                "validation_contract_json": artifacts.validation_contract_json_path,
                "observable_report_json": artifacts.observable_report_json_path,
                "profile_report_json": artifacts.profile_report_json_path,
                "profile_arrays_npz": artifacts.profile_arrays_npz_path,
                "profile_plot_png": artifacts.profile_plot_png_path,
                "arrays_npz": artifacts.arrays_npz_path,
                "analysis_json": artifacts.analysis_json_path,
                "snapshots_png": artifacts.snapshots_png_path,
                "poster_png": artifacts.poster_png_path,
                "movie_gif": artifacts.movie_gif_path,
            }
        )


def _print_section(title: str) -> None:
    print(f"\n== {title} ==")


def _print_kv(values: dict[str, object]) -> None:
    for key, value in values.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
