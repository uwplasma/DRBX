from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from jax_drb.reference.paths import default_reference_root
from typing import Any, Mapping

import numpy as np

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native import run_curated_case
from jax_drb.parity.arrays import (
    build_array_payload_from_summary_payload,
    load_portable_array_payload,
    write_portable_array_payload,
)
from jax_drb.parity.reference import resolve_reference_case
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.validation import create_alfven_wave_meeting_package


@dataclass(frozen=True)
class AlfvenMeetingSettings:
    """User-editable knobs for this native tutorial example."""

    reference_root: Path | None
    case_name: str
    field_variable: str
    x_index: int
    arrays_in: Path | None
    output_root: Path
    native_arrays_out: Path | None
    expected_arrays_in: Path
    input_file: Path
    fps: int
    verbose: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Generate a meeting-ready Alfven-wave package with verbose run logs, "
            "a saved .npz result payload, publication figures, and Matplotlib movies."
        )
    )
    parser.add_argument("--reference-root", type=Path, default=default_reference_root())
    parser.add_argument("--case-name", default="alfven_wave_short_window")
    parser.add_argument("--field-variable", default="phi")
    parser.add_argument("--x-index", type=int, default=2)
    parser.add_argument(
        "--arrays-in",
        type=Path,
        default=None,
        help="Read an existing .npz payload and only regenerate plots/movies.",
    )
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
        default=None,
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--quiet", action="store_true", help="Suppress the verbose tutorial log.")
    return parser.parse_args()


def build_demo_settings(args: argparse.Namespace) -> AlfvenMeetingSettings:
    reference_root = args.reference_root
    if args.input_file is None:
        if reference_root is None:
            raise SystemExit("Set --reference-root or JAX_DRB_REFERENCE_ROOT when --arrays-in is not used.")
        _, resolved_input = resolve_reference_case(args.case_name, reference_root=reference_root)
        input_file = resolved_input
    else:
        input_file = args.input_file
    return AlfvenMeetingSettings(
        reference_root=reference_root,
        case_name=args.case_name,
        field_variable=args.field_variable,
        x_index=args.x_index,
        arrays_in=args.arrays_in,
        output_root=args.output_root,
        native_arrays_out=args.native_arrays_out,
        expected_arrays_in=args.expected_arrays_in,
        input_file=input_file,
        fps=args.fps,
        verbose=not args.quiet,
    )


def describe_requested_case(settings: AlfvenMeetingSettings) -> None:
    """Print the curated case, selected variables, and resolved run setup."""
    if not settings.verbose:
        return

    _print_section("Requested Alfven-Wave Demo")
    _print_kv(
        {
            "case": settings.case_name,
            "reference_root": settings.reference_root,
            "input_file": settings.input_file,
            "field_variable": settings.field_variable,
            "x_index": settings.x_index,
            "fps": settings.fps,
            "arrays_in": settings.arrays_in if settings.arrays_in is not None else "<run native curated case>",
            "expected_arrays": settings.expected_arrays_in,
            "output_root": settings.output_root,
        }
    )

    if settings.arrays_in is None:
        case, input_path = resolve_reference_case(settings.case_name, reference_root=settings.reference_root)
        _print_section("Curated Benchmark-Case Metadata")
        _print_kv(
            {
                "stage": case.stage,
                "parity_mode": case.parity_mode,
                "reference_path": case.reference_path,
                "resolved_input": input_path,
                "compare_variables": ", ".join(case.compare_variables),
                "extra_overrides": ", ".join(case.extra_overrides) if case.extra_overrides else "<none>",
                "process_count": case.process_count,
                "rationale": case.rationale,
            }
        )

    config = load_bout_input(settings.input_file)
    describe_run_configuration(RunConfiguration.from_config(config))


def describe_run_configuration(run_config: RunConfiguration) -> None:
    """Show users where time steps, grid resolution, solver options, and physics components enter."""
    _print_section("Resolved Run Configuration")
    _print_kv(
        {
            "time:nout": run_config.time.nout,
            "time:timestep": run_config.time.timestep,
            "mesh:nx": run_config.mesh.nx,
            "mesh:ny": run_config.mesh.ny,
            "mesh:nz": run_config.mesh.nz,
            "mesh:mxg/myg": f"{run_config.mesh.mxg}/{run_config.mesh.myg}",
            "mesh:file": run_config.mesh.file or "<analytic mesh>",
            "parallel_transform": run_config.mesh.parallel_transform.type,
            "solver:type": run_config.solver.type or "<native default>",
            "solver:rtol": run_config.solver.rtol,
            "solver:atol": run_config.solver.atol,
            "solver:mxstep": run_config.solver.mxstep,
            "normalization": "present" if run_config.normalization is not None else "<none>",
            "components": ", ".join(component.label for component in run_config.components) or "<none>",
        }
    )
    if run_config.mesh.resolved_scalars:
        _print_mapping("mesh resolved scalars", run_config.mesh.resolved_scalars, limit=14)
    if run_config.model_scalars:
        _print_mapping("model resolved scalars", run_config.model_scalars, limit=14)


def run_or_load_arrays(settings: AlfvenMeetingSettings) -> tuple[dict[str, Any], Path]:
    """Run jax_drb or load a saved .npz payload, then return the arrays and their path."""
    if settings.arrays_in is not None:
        _print_step(settings, f"Loading saved array payload: {settings.arrays_in}")
        payload = load_portable_array_payload(settings.arrays_in)
        print_payload_summary(settings, payload)
        return payload, settings.arrays_in

    _print_step(settings, "Running native curated case through jax_drb")
    start = time.perf_counter()
    result = run_curated_case(settings.case_name, reference_root=settings.reference_root)
    elapsed = time.perf_counter() - start

    payload = build_array_payload_from_summary_payload(result.payload, result.variables)
    native_arrays_path = (
        settings.native_arrays_out
        if settings.native_arrays_out is not None
        else settings.output_root / "data" / f"{settings.case_name}_native.npz"
    )
    _print_step(settings, f"Writing complete native result payload: {native_arrays_path}")
    write_portable_array_payload(payload, native_arrays_path)
    if settings.verbose:
        _print_kv({"native_run_seconds": f"{elapsed:.3f}", "native_arrays": native_arrays_path})
    print_payload_summary(settings, payload)
    return payload, native_arrays_path


def print_payload_summary(settings: AlfvenMeetingSettings, payload: Mapping[str, Any]) -> None:
    """Print field shapes and basic ranges so the .npz can be sanity-checked without plotting."""
    if not settings.verbose:
        return

    _print_section("Saved Result Payload")
    _print_kv(
        {
            "case_name": payload.get("case_name"),
            "parity_mode": payload.get("parity_mode"),
            "producer": payload.get("producer"),
            "effective_output_points": payload.get("effective_output_points"),
            "time_points": _summarize_sequence(payload.get("time_points", [])),
        }
    )
    variables = payload.get("variables", {})
    for name, values in variables.items():
        array = np.asarray(values, dtype=np.float64)
        _print_kv(
            {
                f"{name}:shape": array.shape,
                f"{name}:min": f"{array.min():.8e}",
                f"{name}:max": f"{array.max():.8e}",
                f"{name}:mean": f"{array.mean():.8e}",
                f"{name}:rms": f"{np.sqrt(np.mean(array ** 2)):.8e}",
            }
        )


def create_plots_and_movies(settings: AlfvenMeetingSettings, payload: Mapping[str, Any], native_arrays_path: Path):
    """Create the Matplotlib 2D movie, 3D movie, snapshots, diagnostics, and parity plot."""
    _print_step(settings, "Creating Matplotlib figures and movies")
    start = time.perf_counter()
    artifacts = create_alfven_wave_meeting_package(
        payload,
        input_file=settings.input_file,
        expected_arrays_path=settings.expected_arrays_in,
        native_arrays_path=native_arrays_path,
        output_root=settings.output_root,
        field_variable=settings.field_variable,
        x_index=settings.x_index,
        case_label="alfven_wave_meeting",
        fps=settings.fps,
    )
    elapsed = time.perf_counter() - start
    if settings.verbose:
        _print_kv({"plot_and_movie_seconds": f"{elapsed:.3f}"})
    return artifacts


def print_artifact_report(artifacts) -> None:
    _print_section("Generated Artifacts")
    _print_kv(
        {
            "native_arrays": artifacts.native_arrays_path,
            "analysis_json": artifacts.analysis_json_path,
            "parity_json": artifacts.parity_json_path,
            "snapshots_png": artifacts.snapshots_png_path,
            "diagnostics_png": artifacts.diagnostics_png_path,
            "parity_png": artifacts.parity_png_path,
            "poster_png": artifacts.poster_png_path,
            "movie_2d": artifacts.movie_2d_path,
            "movie_3d": artifacts.movie_3d_path,
        }
    )


def main() -> int:
    settings = build_demo_settings(parse_args())
    describe_requested_case(settings)
    payload, native_arrays_path = run_or_load_arrays(settings)
    artifacts = create_plots_and_movies(settings, payload, native_arrays_path)
    print_artifact_report(artifacts)
    return 0


def _print_section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def _print_step(settings: AlfvenMeetingSettings, message: str) -> None:
    if settings.verbose:
        print(f"[jax_drb example] {message}")


def _print_kv(items: Mapping[str, object]) -> None:
    width = max((len(str(key)) for key in items), default=1)
    for key, value in items.items():
        print(f"  {key:<{width}} : {value}")


def _print_mapping(label: str, values: Mapping[str, object], *, limit: int) -> None:
    print(f"  {label}:")
    for index, (key, value) in enumerate(sorted(values.items())):
        if index >= limit:
            print(f"    ... {len(values) - limit} more")
            break
        print(f"    {key:<24} {value}")


def _summarize_sequence(values: object) -> str:
    sequence = list(values) if values is not None else []
    if not sequence:
        return "<none>"
    if len(sequence) <= 6:
        return ", ".join(f"{float(value):.6g}" for value in sequence)
    return (
        f"{float(sequence[0]):.6g}, {float(sequence[1]):.6g}, ... "
        f"{float(sequence[-2]):.6g}, {float(sequence[-1]):.6g} ({len(sequence)} values)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
