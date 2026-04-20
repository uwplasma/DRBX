from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from jax_drb.reference.paths import default_reference_root
from typing import Any, Mapping

import numpy as np

from jax_drb.native import run_curated_case
from jax_drb.parity.arrays import (
    build_array_payload_from_summary_payload,
    load_portable_array_payload,
    write_portable_array_payload,
)
from jax_drb.parity.reference import resolve_reference_case
from jax_drb.validation import (
    Blob2DMeetingArtifacts,
    analyze_blob2d_array_payload,
    create_blob2d_meeting_package,
    save_blob2d_heatmap_movie,
    save_blob2d_poster_frame,
    save_blob2d_snapshot_panel,
    save_blob2d_surface_movie,
    write_blob2d_analysis_json,
)


@dataclass(frozen=True)
class Blob2DMeetingSettings:
    """User-editable knobs for the Blob2D movie/plot example."""

    reference_root: Path | None
    case_name: str
    arrays_in: Path | None
    output_root: Path
    native_arrays_out: Path | None
    reference_metrics_in: Path
    density_variable: str
    background_density: float
    fps: int
    skip_parity: bool
    verbose: bool


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    repo_root = _default_repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Generate a meeting-ready Blob2D package with verbose logs, saved .npz data, "
            "Matplotlib 2D/3D movies, and summary figures."
        )
    )
    parser.add_argument("--reference-root", type=Path, default=default_reference_root())
    parser.add_argument("--case-name", default="blob2d_short_window")
    parser.add_argument(
        "--arrays-in",
        type=Path,
        default=None,
        help="Read an existing portable .npz array payload instead of running the expensive native case.",
    )
    parser.add_argument("--output-root", type=Path, default=repo_root / "docs")
    parser.add_argument("--native-arrays-out", type=Path, default=None)
    parser.add_argument(
        "--reference-metrics-in",
        type=Path,
        default=repo_root / "references" / "baselines" / "reference_metrics" / "blob2d_short_window_metrics.json",
    )
    parser.add_argument("--density-variable", default="Ne")
    parser.add_argument("--background-density", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Only analyze and visualize this payload. Useful when --arrays-in is not on the short-window timeline.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the verbose tutorial log.")
    return parser.parse_args()


def build_demo_settings(args: argparse.Namespace) -> Blob2DMeetingSettings:
    if args.arrays_in is None and args.reference_root is None:
        raise SystemExit("Set --reference-root or JAX_DRB_REFERENCE_ROOT when --arrays-in is not used.")
    return Blob2DMeetingSettings(
        reference_root=args.reference_root,
        case_name=args.case_name,
        arrays_in=args.arrays_in,
        output_root=args.output_root,
        native_arrays_out=args.native_arrays_out,
        reference_metrics_in=args.reference_metrics_in,
        density_variable=args.density_variable,
        background_density=args.background_density,
        fps=args.fps,
        skip_parity=args.skip_parity,
        verbose=not args.quiet,
    )


def describe_requested_case(settings: Blob2DMeetingSettings) -> None:
    """Print the case controls a user would typically edit first."""
    if not settings.verbose:
        return
    _print_section("Requested Blob2D Demo")
    _print_kv(
        {
            "case": settings.case_name,
            "reference_root": settings.reference_root,
            "arrays_in": settings.arrays_in if settings.arrays_in is not None else "<run native curated case>",
            "density_variable": settings.density_variable,
            "background_density": settings.background_density,
            "fps": settings.fps,
            "reference_metrics": settings.reference_metrics_in,
            "skip_parity": settings.skip_parity,
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


def run_or_load_arrays(settings: Blob2DMeetingSettings) -> tuple[dict[str, Any], Path]:
    """Run jax_drb or read a saved .npz result. Prefer --arrays-in for quick movie regeneration."""
    if settings.arrays_in is not None:
        _print_step(settings, f"Loading saved array payload: {settings.arrays_in}")
        payload = load_portable_array_payload(settings.arrays_in)
        print_payload_summary(settings, payload)
        return payload, settings.arrays_in

    _print_step(settings, "Running native curated Blob2D case through jax_drb")
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


def print_payload_summary(settings: Blob2DMeetingSettings, payload: Mapping[str, Any]) -> None:
    """Print the saved fields and basic density ranges before plotting."""
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
    for name, values in payload.get("variables", {}).items():
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


def create_plots_and_movies(settings: Blob2DMeetingSettings, payload: Mapping[str, Any], native_arrays_path: Path):
    """Create Blob2D snapshots, parity plot, 2D movie, and 3D surface movie from a saved .npz."""
    _print_step(settings, "Creating Matplotlib Blob2D figures and movies")
    start = time.perf_counter()
    if settings.skip_parity:
        artifacts = create_blob2d_visualization_package_without_parity(settings, payload, native_arrays_path)
    else:
        artifacts = create_blob2d_meeting_package(
            payload,
            output_root=settings.output_root,
            native_arrays_path=native_arrays_path,
            reference_metrics_path=settings.reference_metrics_in,
            density_variable=settings.density_variable,
            background_density=settings.background_density,
            case_label="blob2d_meeting",
            fps=settings.fps,
        )
    elapsed = time.perf_counter() - start
    if settings.verbose:
        _print_kv({"plot_and_movie_seconds": f"{elapsed:.3f}"})
    return artifacts


def create_blob2d_visualization_package_without_parity(
    settings: Blob2DMeetingSettings,
    payload: Mapping[str, Any],
    native_arrays_path: Path,
) -> Blob2DMeetingArtifacts:
    """Visualize any Blob2D payload, even if it does not share the short-window parity timeline."""
    images_dir = settings.output_root / "images"
    movies_dir = settings.output_root / "movies"
    data_dir = settings.output_root / "data"
    images_dir.mkdir(parents=True, exist_ok=True)
    movies_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    analysis = analyze_blob2d_array_payload(
        payload,
        density_variable=settings.density_variable,
        background_density=settings.background_density,
    )
    analysis_json_path = data_dir / "blob2d_meeting_analysis.json"
    parity_json_path = data_dir / "blob2d_meeting_parity_skipped.json"
    snapshots_png_path = images_dir / "blob2d_meeting_snapshots.png"
    parity_png_path = images_dir / "blob2d_meeting_parity_skipped.png"
    poster_png_path = images_dir / "blob2d_meeting_movie_poster.png"
    movie_2d_path = movies_dir / "blob2d_meeting_2d.mp4"
    movie_3d_path = movies_dir / "blob2d_meeting_3d.mp4"

    write_blob2d_analysis_json(analysis, analysis_json_path)
    parity_json_path.write_text(
        "{\n"
        '  "status": "skipped",\n'
        '  "reason": "This payload was visualized without requiring matching reference metrics."\n'
        "}\n",
        encoding="utf-8",
    )
    _save_parity_skipped_plot(parity_png_path)
    save_blob2d_snapshot_panel(
        payload,
        analysis=analysis,
        path=snapshots_png_path,
        density_variable=settings.density_variable,
        background_density=settings.background_density,
    )
    save_blob2d_poster_frame(
        payload,
        analysis=analysis,
        path=poster_png_path,
        density_variable=settings.density_variable,
        background_density=settings.background_density,
    )
    save_blob2d_heatmap_movie(
        payload,
        analysis=analysis,
        path=movie_2d_path,
        density_variable=settings.density_variable,
        background_density=settings.background_density,
        fps=settings.fps,
    )
    save_blob2d_surface_movie(
        payload,
        analysis=analysis,
        path=movie_3d_path,
        density_variable=settings.density_variable,
        background_density=settings.background_density,
        fps=settings.fps,
    )
    return Blob2DMeetingArtifacts(
        native_arrays_path=native_arrays_path,
        analysis_json_path=analysis_json_path,
        parity_json_path=parity_json_path,
        snapshots_png_path=snapshots_png_path,
        parity_png_path=parity_png_path,
        poster_png_path=poster_png_path,
        movie_2d_path=movie_2d_path,
        movie_3d_path=movie_3d_path,
    )


def _save_parity_skipped_plot(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.5, 2.6), constrained_layout=True)
    axis.text(
        0.5,
        0.5,
        "Parity plot skipped\nvisualization-only payload",
        ha="center",
        va="center",
        fontsize=13,
    )
    axis.set_axis_off()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def print_artifact_report(artifacts) -> None:
    _print_section("Generated Artifacts")
    _print_kv(
        {
            "native_arrays": artifacts.native_arrays_path,
            "analysis_json": artifacts.analysis_json_path,
            "parity_json": artifacts.parity_json_path,
            "snapshots_png": artifacts.snapshots_png_path,
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


def _print_step(settings: Blob2DMeetingSettings, message: str) -> None:
    if settings.verbose:
        print(f"[jax_drb example] {message}")


def _print_kv(items: Mapping[str, object]) -> None:
    width = max((len(str(key)) for key in items), default=1)
    for key, value in items.items():
        print(f"  {key:<{width}} : {value}")


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
