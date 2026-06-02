from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from jax_drb.parity.reference import run_reference_case
from jax_drb.reference.paths import require_reference_root
from jax_drb.runtime.artifacts import ensure_docs_media
from jax_drb.validation import (
    create_diverted_tokamak_movie_package,
    create_diverted_tokamak_movie_package_from_arrays,
)


@dataclass(frozen=True)
class DivertedTokamakMovieSettings:
    reference_root: Path | None
    reference_binary: Path | None
    case_name: str
    field_name: str
    output_root: Path
    workdir_in: Path | None
    mesh_path: Path | None
    release_arrays_path: Path
    use_release_arrays_if_available: bool
    fps: int
    frames_per_interval: int
    verbose: bool


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Keep the public example self-contained by default. The script regenerates the
# movie from release-backed JAXDRB arrays and only enters the external-reference
# path when a developer explicitly sets REFERENCE_ROOT or WORKDIR_IN below.
REFERENCE_ROOT: Path | None = None
REFERENCE_BINARY: Path | None = None
CASE_NAME = "tokamak_turbulence_short_window"
FIELD_NAME = "phi"
OUTPUT_ROOT = _default_repo_root() / "docs" / "data" / "diverted_tokamak_turbulence_artifacts"
WORKDIR_IN: Path | None = None
MESH_PATH: Path | None = None
RELEASE_ARRAYS_PATH = OUTPUT_ROOT / "data" / "diverted_tokamak_turbulence_arrays.npz"
USE_RELEASE_ARRAYS_IF_AVAILABLE = True
FPS = 10
FRAMES_PER_INTERVAL = 10
VERBOSE = True


def describe_requested_case(settings: DivertedTokamakMovieSettings) -> None:
    if not settings.verbose:
        return
    _print_section("Requested Diverted Tokamak Demo")
    _print_kv(
        {
            "reference_root": settings.reference_root
            if settings.reference_root is not None
            else "<not used in self-contained release-array mode>",
            "reference_binary": settings.reference_binary
            if settings.reference_binary is not None
            else "<not used in self-contained release-array mode>",
            "case_name": settings.case_name,
            "field_name": settings.field_name,
            "output_root": settings.output_root,
            "workdir_in": settings.workdir_in
            if settings.workdir_in is not None
            else "<not used in self-contained release-array mode>",
            "mesh_path": settings.mesh_path
            if settings.mesh_path is not None
            else "<loaded from release arrays in self-contained mode>",
            "release_arrays_path": settings.release_arrays_path,
            "use_release_arrays_if_available": settings.use_release_arrays_if_available,
            "fps": settings.fps,
            "frames_per_interval": settings.frames_per_interval,
        }
    )


def should_use_release_arrays(settings: DivertedTokamakMovieSettings) -> bool:
    if not settings.use_release_arrays_if_available or settings.workdir_in is not None:
        return False
    if settings.release_arrays_path.exists():
        return True
    if settings.reference_root is not None:
        return False
    _print_step(settings, "Release arrays are missing; trying private release artifact restore")
    try:
        ensure_docs_media(root=_default_repo_root())
    except Exception as error:
        raise FileNotFoundError(
            "Release arrays are missing. Run `gh auth login --hostname github.com` "
            "and `python scripts/fetch_example_artifacts.py --skip-baselines` "
            "before running this self-contained example."
        ) from error
    return settings.release_arrays_path.exists()


def create_plots_and_movie_from_release_arrays(settings: DivertedTokamakMovieSettings):
    _print_step(settings, "Creating diverted-tokamak figures and GIF from release arrays")
    start = time.perf_counter()
    artifacts = create_diverted_tokamak_movie_package_from_arrays(
        arrays_npz_path=settings.release_arrays_path,
        output_root=settings.output_root,
        case_label="diverted_tokamak_turbulence",
        field_name=settings.field_name,
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


def run_or_reuse_reference_case(
    settings: DivertedTokamakMovieSettings,
) -> tuple[Path, Path, bool]:
    if settings.workdir_in is not None:
        return settings.workdir_in, _resolve_mesh_path_for_workdir(settings), False
    reference_root = _require_reference_root(settings.reference_root)
    _print_step(settings, f"Running curated benchmark case: {settings.case_name}")
    start = time.perf_counter()
    execution = run_reference_case(
        settings.case_name,
        reference_root=reference_root,
        reference_binary=settings.reference_binary,
        keep_workdir=True,
    )
    elapsed = time.perf_counter() - start
    workdir = Path(execution.summary.workdir)
    mesh_path = _resolve_mesh_path_for_workdir(
        DivertedTokamakMovieSettings(
            reference_root=reference_root,
            reference_binary=settings.reference_binary,
            case_name=settings.case_name,
            field_name=settings.field_name,
            output_root=settings.output_root,
            workdir_in=workdir,
            mesh_path=settings.mesh_path,
            release_arrays_path=settings.release_arrays_path,
            use_release_arrays_if_available=settings.use_release_arrays_if_available,
            fps=settings.fps,
            frames_per_interval=settings.frames_per_interval,
            verbose=settings.verbose,
        )
    )
    if settings.verbose:
        _print_kv(
            {
                "benchmark_run_seconds": f"{elapsed:.3f}",
                "workdir": workdir,
                "mesh_path": mesh_path,
            }
        )
    return workdir, mesh_path, True


def _require_reference_root(reference_root: Path | None) -> Path:
    if reference_root is not None:
        return reference_root
    return require_reference_root()


def _resolve_mesh_path_for_workdir(settings: DivertedTokamakMovieSettings) -> Path:
    if settings.mesh_path is not None:
        return settings.mesh_path
    if settings.workdir_in is not None and (settings.workdir_in / "tokamak.nc").exists():
        return settings.workdir_in / "tokamak.nc"
    if settings.reference_root is not None:
        reference_mesh = settings.reference_root / "examples" / "tokamak-2D" / "tokamak.nc"
        if reference_mesh.exists():
            return reference_mesh
    raise FileNotFoundError(
        "Could not find tokamak.nc. Set MESH_PATH, keep a reference workdir containing tokamak.nc, "
        "or set REFERENCE_ROOT in this script to a reference-suite checkout with "
        "examples/tokamak-2D/tokamak.nc."
    )


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


def maybe_cleanup_workdir(
    settings: DivertedTokamakMovieSettings, workdir: Path, *, created_here: bool
) -> None:
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
    reference_binary=REFERENCE_BINARY,
    case_name=CASE_NAME,
    field_name=FIELD_NAME,
    output_root=OUTPUT_ROOT,
    workdir_in=WORKDIR_IN,
    mesh_path=MESH_PATH,
    release_arrays_path=RELEASE_ARRAYS_PATH,
    use_release_arrays_if_available=USE_RELEASE_ARRAYS_IF_AVAILABLE,
    fps=FPS,
    frames_per_interval=FRAMES_PER_INTERVAL,
    verbose=VERBOSE,
)

describe_requested_case(settings)
if should_use_release_arrays(settings):
    artifacts = create_plots_and_movie_from_release_arrays(settings)
else:
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
