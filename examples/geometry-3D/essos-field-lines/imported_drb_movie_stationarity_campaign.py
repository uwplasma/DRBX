from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dkx.runtime import configure_jax_runtime
from dkx.validation import create_essos_imported_drb_movie_stationarity_package


# SIMSOPT-style user parameters: edit these values, then run this file.
RUN_EXAMPLE = True

COIL_JSON_PATH: Path | None = None
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
MAP_SOURCE = "hybrid"

OUTPUT_ROOT = Path("docs/data/essos_imported_drb_movie_stationarity_jacobi_artifacts")
CASE_LABEL = "essos_imported_drb_movie_stationarity_jacobi"

NX = 16
NY = 96
NZ = 48
RHO_MIN = 0.20
RHO_MAX = 0.60
TIMES_TO_TRACE = 80
MAXTIME = 24.0
FRAMES = 12
SUBSTEPS_PER_FRAME = 3
DT = 2.0e-3
POTENTIAL_ITERATIONS = 3072
POTENTIAL_REGULARIZATION = 5.0
POTENTIAL_PRECONDITIONER = "jacobi"
TAIL_FRACTION = 0.50
RELATIVE_TOLERANCE = 0.35
MIN_FRAMES = 12
REQUIRE_STATIONARITY_READY = False


@dataclass(frozen=True)
class ImportedDrbMovieStationaritySettings:
    output_root: Path
    case_label: str
    coil_json_path: Path | None
    vmec_wout_path: Path | None
    essos_root: Path | None
    map_source: str
    nx: int
    ny: int
    nz: int
    rho_min: float
    rho_max: float
    maxtime: float
    times_to_trace: int
    frames: int
    substeps_per_frame: int
    dt: float
    potential_iterations: int
    potential_regularization: float
    potential_preconditioner: str | None
    tail_fraction: float
    relative_tolerance: float
    min_frames: int
    require_stationarity_ready: bool


def build_stationarity_settings(
    *,
    output_root: Path = OUTPUT_ROOT,
    case_label: str = CASE_LABEL,
    coil_json_path: Path | None = COIL_JSON_PATH,
    vmec_wout_path: Path | None = VMEC_WOUT_PATH,
    essos_root: Path | None = ESSOS_ROOT,
    map_source: str = MAP_SOURCE,
    nx: int = NX,
    ny: int = NY,
    nz: int = NZ,
    rho_min: float = RHO_MIN,
    rho_max: float = RHO_MAX,
    maxtime: float = MAXTIME,
    times_to_trace: int = TIMES_TO_TRACE,
    frames: int = FRAMES,
    substeps_per_frame: int = SUBSTEPS_PER_FRAME,
    dt: float = DT,
    potential_iterations: int = POTENTIAL_ITERATIONS,
    potential_regularization: float = POTENTIAL_REGULARIZATION,
    potential_preconditioner: str | None = POTENTIAL_PRECONDITIONER,
    tail_fraction: float = TAIL_FRACTION,
    relative_tolerance: float = RELATIVE_TOLERANCE,
    min_frames: int = MIN_FRAMES,
    require_stationarity_ready: bool = REQUIRE_STATIONARITY_READY,
) -> ImportedDrbMovieStationaritySettings:
    """Resolve a JSON-only long-window stationarity gate configuration."""

    return ImportedDrbMovieStationaritySettings(
        output_root=Path(output_root),
        case_label=str(case_label),
        coil_json_path=None if coil_json_path is None else Path(coil_json_path),
        vmec_wout_path=None if vmec_wout_path is None else Path(vmec_wout_path),
        essos_root=None if essos_root is None else Path(essos_root),
        map_source=str(map_source),
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        rho_min=float(rho_min),
        rho_max=float(rho_max),
        maxtime=float(maxtime),
        times_to_trace=int(times_to_trace),
        frames=int(frames),
        substeps_per_frame=int(substeps_per_frame),
        dt=float(dt),
        potential_iterations=int(potential_iterations),
        potential_regularization=float(potential_regularization),
        potential_preconditioner=potential_preconditioner,
        tail_fraction=float(tail_fraction),
        relative_tolerance=float(relative_tolerance),
        min_frames=int(min_frames),
        require_stationarity_ready=bool(require_stationarity_ready),
    )


def run_stationarity_campaign(
    settings: ImportedDrbMovieStationaritySettings,
) -> dict[str, object]:
    """Run the long-window gate and return the stationarity report."""

    configure_jax_runtime(precision="float64")
    artifacts = create_essos_imported_drb_movie_stationarity_package(
        output_root=settings.output_root,
        case_label=settings.case_label,
        coil_json_path=settings.coil_json_path,
        vmec_wout_path=settings.vmec_wout_path,
        essos_root=settings.essos_root,
        map_source=settings.map_source,
        nx=settings.nx,
        ny=settings.ny,
        nz=settings.nz,
        rho_min=settings.rho_min,
        rho_max=settings.rho_max,
        maxtime=settings.maxtime,
        times_to_trace=settings.times_to_trace,
        frames=settings.frames,
        substeps_per_frame=settings.substeps_per_frame,
        dt=settings.dt,
        potential_iterations=settings.potential_iterations,
        potential_regularization=settings.potential_regularization,
        potential_preconditioner=settings.potential_preconditioner,
        tail_fraction=settings.tail_fraction,
        relative_tolerance=settings.relative_tolerance,
        min_frames=settings.min_frames,
    )
    import json

    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    print(f"wrote movie stationarity report: {artifacts.report_json_path}")
    print(
        "movie stationarity evidence: "
        f"stationarity_passed={report['stationarity_passed']}, "
        f"publication_ready={report['publication_ready']}, "
        f"reasons={report['movie_promotion_rejection_reasons']}"
    )
    if settings.require_stationarity_ready and not bool(report["publication_ready"]):
        raise RuntimeError(
            "Imported-field movie stationarity gate failed: "
            f"{report['movie_promotion_rejection_reasons']}"
        )
    return report


if RUN_EXAMPLE:
    STATIONARITY_SETTINGS = build_stationarity_settings()
    STATIONARITY_REPORT = run_stationarity_campaign(STATIONARITY_SETTINGS)
