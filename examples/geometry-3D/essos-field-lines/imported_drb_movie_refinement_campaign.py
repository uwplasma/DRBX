from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import create_essos_imported_drb_movie_refinement_campaign_package


# SIMSOPT-style user parameters: edit these first, then run this file.
RUN_EXAMPLE = True

COIL_JSON_PATH: Path | None = None
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
MAP_SOURCE = "hybrid"

OUTPUT_ROOT = Path("docs/data/essos_imported_drb_movie_refinement_campaign_artifacts")
CASE_LABEL = "essos_imported_drb_movie_refinement_campaign"

# This default is intentionally compact and report-only. Increase these levels
# before using the output as publication evidence.
GRID_SHAPES = (
    (3, 4, 8),
    (4, 6, 12),
)
TIME_SHAPE: tuple[int, int, int] | None = None
TIME_DT_VALUES = (2.0e-3, 1.0e-3)

RHO_MIN = 0.20
RHO_MAX = 0.60
TIMES_TO_TRACE = 80
MAXTIME = 24.0
FRAMES = 4
SUBSTEPS_PER_FRAME = 2
GRID_DT = 2.0e-3
RELATIVE_TOLERANCE = 0.30
REQUIRE_PUBLICATION_READY = False


@dataclass(frozen=True)
class ImportedDrbMovieRefinementCampaignSettings:
    output_root: Path
    case_label: str
    coil_json_path: Path | None
    vmec_wout_path: Path | None
    essos_root: Path | None
    map_source: str
    grid_shapes: tuple[tuple[int, int, int], ...]
    time_shape: tuple[int, int, int] | None
    time_dt_values: tuple[float, ...]
    rho_min: float
    rho_max: float
    maxtime: float
    times_to_trace: int
    frames: int
    substeps_per_frame: int
    grid_dt: float
    relative_tolerance: float
    require_publication_ready: bool


def build_refinement_campaign_settings(
    *,
    output_root: Path = OUTPUT_ROOT,
    case_label: str = CASE_LABEL,
    coil_json_path: Path | None = COIL_JSON_PATH,
    vmec_wout_path: Path | None = VMEC_WOUT_PATH,
    essos_root: Path | None = ESSOS_ROOT,
    map_source: str = MAP_SOURCE,
    grid_shapes: tuple[tuple[int, int, int], ...] = GRID_SHAPES,
    time_shape: tuple[int, int, int] | None = TIME_SHAPE,
    time_dt_values: tuple[float, ...] = TIME_DT_VALUES,
    rho_min: float = RHO_MIN,
    rho_max: float = RHO_MAX,
    maxtime: float = MAXTIME,
    times_to_trace: int = TIMES_TO_TRACE,
    frames: int = FRAMES,
    substeps_per_frame: int = SUBSTEPS_PER_FRAME,
    grid_dt: float = GRID_DT,
    relative_tolerance: float = RELATIVE_TOLERANCE,
    require_publication_ready: bool = REQUIRE_PUBLICATION_READY,
) -> ImportedDrbMovieRefinementCampaignSettings:
    """Resolve top-level parameters for a report-only movie refinement run."""

    return ImportedDrbMovieRefinementCampaignSettings(
        output_root=Path(output_root),
        case_label=str(case_label),
        coil_json_path=None if coil_json_path is None else Path(coil_json_path),
        vmec_wout_path=None if vmec_wout_path is None else Path(vmec_wout_path),
        essos_root=None if essos_root is None else Path(essos_root),
        map_source=str(map_source),
        grid_shapes=tuple(tuple(int(value) for value in shape) for shape in grid_shapes),
        time_shape=(
            None if time_shape is None else tuple(int(value) for value in time_shape)
        ),
        time_dt_values=tuple(float(value) for value in time_dt_values),
        rho_min=float(rho_min),
        rho_max=float(rho_max),
        maxtime=float(maxtime),
        times_to_trace=int(times_to_trace),
        frames=int(frames),
        substeps_per_frame=int(substeps_per_frame),
        grid_dt=float(grid_dt),
        relative_tolerance=float(relative_tolerance),
        require_publication_ready=bool(require_publication_ready),
    )


def run_refinement_campaign(
    settings: ImportedDrbMovieRefinementCampaignSettings,
) -> dict[str, object]:
    """Run report-only grid/time transients and return the summary report."""

    configure_jax_runtime(precision="float64")
    artifacts = create_essos_imported_drb_movie_refinement_campaign_package(
        output_root=settings.output_root,
        case_label=settings.case_label,
        coil_json_path=settings.coil_json_path,
        vmec_wout_path=settings.vmec_wout_path,
        essos_root=settings.essos_root,
        map_source=settings.map_source,
        grid_shapes=settings.grid_shapes,
        time_shape=settings.time_shape,
        time_dt_values=settings.time_dt_values,
        rho_min=settings.rho_min,
        rho_max=settings.rho_max,
        maxtime=settings.maxtime,
        times_to_trace=settings.times_to_trace,
        frames=settings.frames,
        substeps_per_frame=settings.substeps_per_frame,
        grid_dt=settings.grid_dt,
        relative_tolerance=settings.relative_tolerance,
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    print(f"wrote movie refinement summary: {artifacts.report_json_path}")
    print(f"wrote grid reports: {[str(path) for path in artifacts.grid_report_json_paths]}")
    print(f"wrote time reports: {[str(path) for path in artifacts.time_report_json_paths]}")
    print(
        "movie refinement evidence: "
        f"publication_ready={report['publication_ready']}, "
        f"grid_passed={report['grid_refinement_passed']}, "
        f"time_passed={report['time_refinement_passed']}, "
        f"reasons={report['movie_promotion_rejection_reasons']}"
    )
    if settings.require_publication_ready and not bool(report["publication_ready"]):
        raise RuntimeError(
            "Imported-field movie refinement campaign failed: "
            f"{report['movie_promotion_rejection_reasons']}"
        )
    return report


if RUN_EXAMPLE:
    REFINEMENT_CAMPAIGN_SETTINGS = build_refinement_campaign_settings()
    REFINEMENT_CAMPAIGN = run_refinement_campaign(REFINEMENT_CAMPAIGN_SETTINGS)
