from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from jax_drb.parity.reference import run_reference_case
from jax_drb.reference.paths import default_reference_root
from jax_drb.validation import create_diverted_tokamak_movie_package


@dataclass(frozen=True)
class DivertedTokamakMovieSettings:
    reference_root: Path
    case_name: str
    field_name: str
    output_root: Path
    workdir_in: Path | None
    fps: int
    frames_per_interval: int
    verbose: bool


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REFERENCE_ROOT = default_reference_root()
CASE_NAME = "tokamak_turbulence_short_window"
FIELD_NAME = "phi"
OUTPUT_ROOT = _default_repo_root() / "docs" / "data" / "diverted_tokamak_turbulence_artifacts"
WORKDIR_IN: Path | None = None
FPS = 10
FRAMES_PER_INTERVAL = 10
VERBOSE = True


def describe_requested_case(settings: DivertedTokamakMovieSettings) -> None:
    if not settings.verbose:
        return
    _print_section("Requested Diverted Tokamak Demo")
    _print_kv(
        {
            "reference_root": settings.reference_root,
            "case_name": settings.case_name,
            "field_name": settings.field_name,
            "output_root": settings.output_root,
            "workdir_in": settings.workdir_in if settings.workdir_in is not None else "<run fresh benchmark case>",
            "fps": settings.fps,
            "frames_per_interval": settings.frames_per_interval,
        }
    )


def run_or_reuse_reference_case(settings: DivertedTokamakMovieSettings) -> tuple[Path, Path, bool]:
    if settings.workdir_in is not None:
        return settings.workdir_in, settings.workdir_in / "tokamak.nc", False
    _print_step(settings, f"Running curated benchmark case: {settings.case_name}")
    start = time.perf_counter()
    execution = run_reference_case(
        settings.case_name,
        reference_root=settings.reference_root,
        keep_workdir=True,
    )
    elapsed = time.perf_counter() - start
    workdir = Path(execution.summary.workdir)
    mesh_path = workdir / "tokamak.nc"
    if settings.verbose:
        _print_kv(
            {
                "benchmark_run_seconds": f"{elapsed:.3f}",
                "workdir": workdir,
                "mesh_path": mesh_path,
            }
        )
    return workdir, mesh_path, True


def create_plots_and_movie(
    settings: DivertedTokamakMovieSettings,
    *,
    workdir: Path,
    mesh_path: Path,
):
    _print_step(settings, "Creating diverted-tokamak figures and GIF")
    start = time.perf_counter()
    artifacts = create_diverted_tokamak_movie_package(
        workdir=workdir,
        mesh_path=mesh_path,
        output_root=settings.output_root,
        field_name=settings.field_name,
        case_label="diverted_tokamak_turbulence",
        fps=settings.fps,
        frames_per_interval=settings.frames_per_interval,
    )
    elapsed = time.perf_counter() - start
    if settings.verbose:
        _print_kv(
            {
                "plot_and_movie_seconds": f"{elapsed:.3f}",
                "arrays_npz": artifacts.arrays_npz_path,
                "analysis_json": artifacts.analysis_json_path,
                "snapshots_png": artifacts.snapshots_png_path,
                "poster_png": artifacts.poster_png_path,
                "movie_gif": artifacts.movie_gif_path,
            }
        )
    return artifacts


def maybe_cleanup_workdir(settings: DivertedTokamakMovieSettings, workdir: Path, *, created_here: bool) -> None:
    if not created_here:
        return
    _print_step(settings, f"Cleaning temporary reference workdir: {workdir}")
    shutil.rmtree(workdir, ignore_errors=True)


def _print_section(title: str) -> None:
    print(f"\n== {title} ==")


def _print_step(settings: DivertedTokamakMovieSettings, message: str) -> None:
    if settings.verbose:
        print(f"[tokamak-demo] {message}")


def _print_kv(values: dict[str, object]) -> None:
    for key, value in values.items():
        print(f"  - {key}: {value}")


settings = DivertedTokamakMovieSettings(
    reference_root=REFERENCE_ROOT,
    case_name=CASE_NAME,
    field_name=FIELD_NAME,
    output_root=OUTPUT_ROOT,
    workdir_in=WORKDIR_IN,
    fps=FPS,
    frames_per_interval=FRAMES_PER_INTERVAL,
    verbose=VERBOSE,
)

describe_requested_case(settings)
workdir, mesh_path, created_here = run_or_reuse_reference_case(settings)
try:
    artifacts = create_plots_and_movie(settings, workdir=workdir, mesh_path=mesh_path)
finally:
    maybe_cleanup_workdir(settings, workdir, created_here=created_here)

_print_section("Artifacts")
_print_kv(
    {
        "arrays_npz": artifacts.arrays_npz_path,
        "analysis_json": artifacts.analysis_json_path,
        "snapshots_png": artifacts.snapshots_png_path,
        "poster_png": artifacts.poster_png_path,
        "movie_gif": artifacts.movie_gif_path,
    }
)
