from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from jax_drb.validation import create_essos_imported_drb_movie_refinement_summary_package


# SIMSOPT-style user parameters: edit these first, then run this file.
RUN_EXAMPLE = True

OUTPUT_ROOT = Path("docs/data/essos_imported_drb_movie_refinement_artifacts")
CASE_LABEL = "essos_imported_drb_movie_refinement_summary"

# Replace these with two or more same-map-source movie reports after regenerating
# grid and timestep sweeps with imported_drb_movie_campaign.py. The default
# single-report lists intentionally document that the committed restored movie
# assets are not yet grid/time-refinement evidence.
GRID_REPORT_JSON_PATHS = (
    Path(
        "docs/data/essos_imported_drb_movie_hybrid_artifacts/data/"
        "essos_imported_drb_movie_hybrid_campaign.json"
    ),
)
TIME_REPORT_JSON_PATHS = GRID_REPORT_JSON_PATHS
RELATIVE_TOLERANCE = 0.30
REQUIRE_PUBLICATION_READY = False


@dataclass(frozen=True)
class ImportedDrbMovieRefinementSummarySettings:
    output_root: Path
    case_label: str
    grid_report_json_paths: tuple[Path, ...]
    time_report_json_paths: tuple[Path, ...]
    relative_tolerance: float
    require_publication_ready: bool


def build_refinement_summary_settings(
    *,
    output_root: Path = OUTPUT_ROOT,
    case_label: str = CASE_LABEL,
    grid_report_json_paths: tuple[Path, ...] = GRID_REPORT_JSON_PATHS,
    time_report_json_paths: tuple[Path, ...] = TIME_REPORT_JSON_PATHS,
    relative_tolerance: float = RELATIVE_TOLERANCE,
    require_publication_ready: bool = REQUIRE_PUBLICATION_READY,
) -> ImportedDrbMovieRefinementSummarySettings:
    """Resolve top-level user parameters for a movie refinement summary run."""

    return ImportedDrbMovieRefinementSummarySettings(
        output_root=Path(output_root),
        case_label=str(case_label),
        grid_report_json_paths=tuple(Path(path) for path in grid_report_json_paths),
        time_report_json_paths=tuple(Path(path) for path in time_report_json_paths),
        relative_tolerance=float(relative_tolerance),
        require_publication_ready=bool(require_publication_ready),
    )


def run_refinement_summary(settings: ImportedDrbMovieRefinementSummarySettings) -> dict[str, object]:
    """Build and optionally enforce a report-only movie grid/time refinement gate."""

    artifacts = create_essos_imported_drb_movie_refinement_summary_package(
        output_root=settings.output_root,
        case_label=settings.case_label,
        grid_report_json_paths=settings.grid_report_json_paths,
        time_report_json_paths=settings.time_report_json_paths,
        relative_tolerance=settings.relative_tolerance,
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    print(f"wrote movie refinement summary: {artifacts.report_json_path}")
    print(
        "movie refinement evidence: "
        f"publication_ready={report['publication_ready']}, "
        f"grid_passed={report['grid_refinement_passed']}, "
        f"time_passed={report['time_refinement_passed']}, "
        f"reasons={report['movie_promotion_rejection_reasons']}"
    )
    suggestion = report.get("next_campaign_suggestion", {})
    if suggestion:
        print(
            "suggested next campaign: "
            f"grid_shapes={suggestion.get('suggested_grid_shapes')}, "
            "effective_frame_dt_values="
            f"{suggestion.get('recommended_time_effective_frame_dt_values')}, "
            f"notes={suggestion.get('recommendation_notes')}"
        )
    if settings.require_publication_ready and not bool(report["publication_ready"]):
        raise RuntimeError(
            "Imported-field movie refinement gate failed: "
            f"{report['movie_promotion_rejection_reasons']}"
        )
    return report


if RUN_EXAMPLE:
    REFINEMENT_SUMMARY_SETTINGS = build_refinement_summary_settings()
    REFINEMENT_SUMMARY = run_refinement_summary(REFINEMENT_SUMMARY_SETTINGS)
