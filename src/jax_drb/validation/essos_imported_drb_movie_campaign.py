from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from ..geometry import (
    EssosImportedFciGeometry,
    build_essos_imported_fci_geometry,
    build_essos_vmec_scaled_qa_coordinates,
    resolve_essos_landreman_qa_wout,
)
from ..native.fci import conservative_perp_diffusion_xz, logical_exb_bracket_xz
from ..native.fci_drb_rhs import FciDrbRhsParameters, FciDrbState, compute_fci_drb_rhs
from ..native.fci_neutral import compute_fci_neutral_reaction_diffusion
from ..native.fci_sheath_recycling import compute_fci_sheath_recycling
from .essos_imported_pytree_campaign import initial_essos_imported_drb_state


@dataclass(frozen=True)
class EssosImportedDrbMovieArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    snapshot_png_path: Path
    diagnostics_png_path: Path
    poster_png_path: Path
    movie_gif_path: Path


@dataclass(frozen=True)
class EssosImportedDrbMovieRefinementArtifacts:
    report_json_path: Path


@dataclass(frozen=True)
class EssosImportedDrbMovieRefinementCampaignArtifacts:
    report_json_path: Path
    grid_report_json_paths: tuple[Path, ...]
    time_report_json_paths: tuple[Path, ...]


@dataclass(frozen=True)
class EssosImportedDrbMovieResult:
    geometry: EssosImportedFciGeometry
    report: dict[str, Any]
    arrays: dict[str, np.ndarray]


ESSOS_IMPORTED_DRB_MOVIE_REFINEMENT_METRICS = (
    "final_fluctuation_rms",
    "max_fluctuation_rms",
    "radial_flux_abs_mean",
    "radial_flux_rms",
    "low_mode_spectral_power_fraction",
    "spectral_centroid_poloidal_fraction",
    "spectral_centroid_toroidal_fraction",
    "spectral_edge_band_power_fraction",
    "final_potential_residual_l2",
)

ESSOS_IMPORTED_DRB_MOVIE_MAX_EDGE_BAND_FRACTION = 0.85
ESSOS_IMPORTED_DRB_MOVIE_REFINEMENT_METRIC_FLOORS = {
    "final_potential_residual_l2": 1.0e-10,
}
ESSOS_IMPORTED_DRB_MOVIE_REFINEMENT_DEFAULT_METRIC_FLOOR = 1.0e-12
ESSOS_IMPORTED_DRB_MOVIE_REFINEMENT_NEAR_TOLERANCE_FACTOR = 1.05
ESSOS_IMPORTED_DRB_MOVIE_DEFAULT_POTENTIAL_ITERATIONS = 768
ESSOS_IMPORTED_DRB_MOVIE_DEFAULT_POTENTIAL_REGULARIZATION = 5.0
ESSOS_IMPORTED_DRB_MOVIE_DEFAULT_POTENTIAL_PRECONDITIONER: str | None = None


def _strict_json_payload(value: Any) -> Any:
    """Return a JSON-standards-compliant payload with nonfinite values as null."""

    if isinstance(value, dict):
        return {str(key): _strict_json_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json_payload(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    return value


def _write_strict_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _strict_json_payload(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def classify_essos_imported_drb_movie_evidence(map_source: str) -> dict[str, Any]:
    """Classify whether an imported-field movie is publication evidence."""

    normalized = str(map_source).strip().lower()
    required_gates = [
        "connection_length_refinement_summary_promotion_ready",
        "movie_grid_refinement_passed",
        "movie_time_refinement_passed",
        "long_time_statistical_stationarity_or_convergence_passed",
    ]
    reasons = [
        "movie_grid_refinement_not_passed",
        "movie_time_refinement_not_passed",
        "long_time_statistical_stationarity_not_demonstrated",
    ]
    if normalized == "coil":
        evidence_role = "movie_showcase_pending_connection_grid_time_refinement"
        reasons.insert(0, "coil_connection_length_refinement_not_promotion_ready")
    elif normalized == "hybrid":
        evidence_role = "movie_showcase_connection_control_pending_grid_time_refinement"
    elif normalized == "vmec":
        evidence_role = "closed_field_movie_control_pending_open_sol_endpoint_evidence"
        reasons.insert(0, "closed_field_control_not_open_sol_endpoint_evidence")
    else:
        evidence_role = "movie_showcase_pending_validation"
        reasons.insert(0, f"unknown_map_source:{normalized}")
    return {
        "publication_ready": False,
        "movie_evidence_role": evidence_role,
        "movie_promotion_rejection_reasons": reasons,
        "required_publication_gates": required_gates,
    }


def create_essos_imported_drb_movie_refinement_summary_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_imported_drb_movie_refinement_summary",
    grid_report_json_paths: tuple[str | Path, ...] | list[str | Path] = (),
    time_report_json_paths: tuple[str | Path, ...] | list[str | Path] = (),
    relative_tolerance: float = 0.30,
) -> EssosImportedDrbMovieRefinementArtifacts:
    """Write a lightweight grid/time refinement summary from movie reports."""

    root = Path(output_root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    grid_reports = _load_movie_refinement_reports(grid_report_json_paths)
    time_reports = _load_movie_refinement_reports(time_report_json_paths)
    report = build_essos_imported_drb_movie_refinement_summary(
        grid_reports=grid_reports,
        time_reports=time_reports,
        relative_tolerance=relative_tolerance,
    )
    report_json_path = data_dir / f"{case_label}.json"
    _write_strict_json(report_json_path, report)
    return EssosImportedDrbMovieRefinementArtifacts(report_json_path=report_json_path)


def create_essos_imported_drb_movie_refinement_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_imported_drb_movie_refinement_campaign",
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    map_source: str = "hybrid",
    grid_shapes: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]] = (
        (3, 4, 8),
        (4, 6, 12),
    ),
    time_shape: tuple[int, int, int] | list[int] | None = None,
    time_dt_values: tuple[float, ...] | list[float] = (2.0e-3, 1.0e-3),
    rho_min: float = 0.20,
    rho_max: float = 0.60,
    maxtime: float = 24.0,
    times_to_trace: int = 80,
    frames: int = 4,
    substeps_per_frame: int = 2,
    grid_dt: float = 2.0e-3,
    relative_tolerance: float = 0.30,
    potential_iterations: int = ESSOS_IMPORTED_DRB_MOVIE_DEFAULT_POTENTIAL_ITERATIONS,
    potential_regularization: float = ESSOS_IMPORTED_DRB_MOVIE_DEFAULT_POTENTIAL_REGULARIZATION,
    potential_preconditioner: str | None = (
        ESSOS_IMPORTED_DRB_MOVIE_DEFAULT_POTENTIAL_PRECONDITIONER
    ),
    reuse_existing_reports: bool = False,
) -> EssosImportedDrbMovieRefinementCampaignArtifacts:
    """Run a lightweight report-only grid/time movie refinement campaign.

    The campaign intentionally writes JSON reports only. It does not save NPZ,
    PNG, or GIF artifacts, so it can be used to search for a publication-grade
    grid/time configuration before committing or release-hosting heavy media.
    If ``reuse_existing_reports`` is true, a report is reused only when its
    recorded grid, timestep, geometry, transient, and potential-solver metadata
    match the requested run.
    """

    root = Path(output_root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    normalized_grids = tuple(_normalize_movie_grid_shape(shape) for shape in grid_shapes)
    if len(normalized_grids) < 2:
        raise ValueError("grid_shapes must contain at least two grid levels.")
    normalized_time_shape = (
        _normalize_movie_grid_shape(time_shape)
        if time_shape is not None
        else normalized_grids[-1]
    )
    normalized_time_dt_values = tuple(float(value) for value in time_dt_values)
    if len(normalized_time_dt_values) < 2:
        raise ValueError("time_dt_values must contain at least two timestep values.")
    if any(value <= 0.0 for value in normalized_time_dt_values):
        raise ValueError("time_dt_values must be positive.")

    report_cache: dict[tuple[tuple[int, int, int], float], Path] = {}

    def run_report(shape: tuple[int, int, int], dt: float, role: str, index: int) -> Path:
        cache_key = (shape, float(dt))
        if cache_key in report_cache:
            return report_cache[cache_key]
        nx, ny, nz = shape
        report_json_path = data_dir / (
            f"{case_label}_{role}_{index:02d}_{nx}x{ny}x{nz}_dt={float(dt):.6g}.json"
        )
        if reuse_existing_reports and _imported_drb_movie_report_matches_request(
            report_json_path,
            case_label=case_label,
            role=role,
            index=index,
            shape=shape,
            dt=float(dt),
            map_source=map_source,
            rho_min=rho_min,
            rho_max=rho_max,
            maxtime=maxtime,
            times_to_trace=times_to_trace,
            frames=frames,
            substeps_per_frame=substeps_per_frame,
            potential_iterations=potential_iterations,
            potential_regularization=potential_regularization,
            potential_preconditioner=potential_preconditioner,
        ):
            report_cache[cache_key] = report_json_path
            return report_json_path
        result = build_essos_imported_drb_movie_campaign(
            coil_json_path=coil_json_path,
            vmec_wout_path=vmec_wout_path,
            essos_root=essos_root,
            map_source=map_source,
            nx=nx,
            ny=ny,
            nz=nz,
            rho_min=rho_min,
            rho_max=rho_max,
            maxtime=maxtime,
            times_to_trace=times_to_trace,
            frames=frames,
            substeps_per_frame=substeps_per_frame,
            dt=float(dt),
            potential_iterations=potential_iterations,
            potential_regularization=potential_regularization,
            potential_preconditioner=potential_preconditioner,
        )
        report = dict(result.report)
        report.update(
            {
                "refinement_campaign_role": role,
                "refinement_campaign_index": int(index),
                "refinement_campaign_case_label": str(case_label),
            }
        )
        _write_strict_json(report_json_path, report)
        report_cache[cache_key] = report_json_path
        return report_json_path

    grid_report_json_paths = tuple(
        run_report(shape, grid_dt, "grid", index)
        for index, shape in enumerate(normalized_grids)
    )
    time_report_json_paths = tuple(
        run_report(normalized_time_shape, dt, "time", index)
        for index, dt in enumerate(normalized_time_dt_values)
    )
    artifacts = create_essos_imported_drb_movie_refinement_summary_package(
        output_root=root,
        case_label=f"{case_label}_summary",
        grid_report_json_paths=grid_report_json_paths,
        time_report_json_paths=time_report_json_paths,
        relative_tolerance=relative_tolerance,
    )
    return EssosImportedDrbMovieRefinementCampaignArtifacts(
        report_json_path=artifacts.report_json_path,
        grid_report_json_paths=grid_report_json_paths,
        time_report_json_paths=time_report_json_paths,
    )


def _imported_drb_movie_report_matches_request(
    path: Path,
    *,
    case_label: str,
    role: str,
    index: int,
    shape: tuple[int, int, int],
    dt: float,
    map_source: str,
    rho_min: float,
    rho_max: float,
    maxtime: float,
    times_to_trace: int,
    frames: int,
    substeps_per_frame: int,
    potential_iterations: int,
    potential_regularization: float,
    potential_preconditioner: str | None,
) -> bool:
    """Return true when an existing report is valid for a requested rerun."""

    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    geometry = report.get("geometry")
    if not isinstance(geometry, dict):
        geometry = {}
    requested_preconditioner = _normalize_optional_report_string(
        potential_preconditioner
    )
    report_preconditioner = _normalize_optional_report_string(
        report.get("potential_preconditioner")
    )
    checks = (
        tuple(report.get("movie_physics_grid", ())) == tuple(shape),
        _same_optional_float(report.get("dt"), dt),
        _same_optional_int(report.get("frames"), frames),
        _same_optional_int(report.get("substeps_per_frame"), substeps_per_frame),
        str(report.get("map_source", "")).strip().lower()
        == str(map_source).strip().lower(),
        _same_optional_int(report.get("potential_iterations"), potential_iterations),
        _same_optional_float(
            report.get("potential_regularization"), potential_regularization
        ),
        report_preconditioner == requested_preconditioner,
        str(report.get("refinement_campaign_case_label", "")) == str(case_label),
        str(report.get("refinement_campaign_role", "")) == str(role),
        _same_optional_int(report.get("refinement_campaign_index"), index),
        _same_optional_int(geometry.get("nx"), shape[0]),
        _same_optional_int(geometry.get("ny"), shape[1]),
        _same_optional_int(geometry.get("nz"), shape[2]),
        _same_optional_float(geometry.get("rho_min"), rho_min),
        _same_optional_float(geometry.get("rho_max"), rho_max),
        _same_optional_float(geometry.get("maxtime"), maxtime),
        _same_optional_int(geometry.get("times_to_trace"), times_to_trace),
        str(geometry.get("map_source", report.get("map_source", ""))).strip().lower()
        == str(map_source).strip().lower(),
    )
    return all(checks)


def _normalize_movie_grid_shape(
    shape: tuple[int, int, int] | list[int] | None,
) -> tuple[int, int, int]:
    if not isinstance(shape, (tuple, list)) or len(shape) != 3:
        raise ValueError("movie grid shapes must be length-3 tuples.")
    normalized = tuple(int(value) for value in shape)
    if any(value <= 0 for value in normalized):
        raise ValueError("movie grid shapes must contain positive integers.")
    return normalized


def build_essos_imported_drb_movie_refinement_summary(
    *,
    grid_reports: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
    time_reports: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
    relative_tolerance: float = 0.30,
) -> dict[str, Any]:
    """Build grid/time refinement diagnostics from movie report dictionaries."""

    grid_diagnostics = build_essos_imported_drb_movie_refinement_diagnostics(
        grid_reports,
        refinement_axis="grid",
        relative_tolerance=relative_tolerance,
    )
    time_diagnostics = build_essos_imported_drb_movie_refinement_diagnostics(
        time_reports,
        refinement_axis="time",
        relative_tolerance=relative_tolerance,
    )
    grid_passed = bool(grid_diagnostics["passed"])
    time_passed = bool(time_diagnostics["passed"])
    report = {
        "diagnostic": "essos_imported_drb_movie_refinement_summary",
        "claim_scope": (
            "report-only grid/time refinement gate for imported-field DRB "
            "movies; this summary does not regenerate GIF or NPZ artifacts"
        ),
        "relative_tolerance": float(relative_tolerance),
        "grid_refinement_passed": grid_passed,
        "time_refinement_passed": time_passed,
        "publication_ready": bool(grid_passed and time_passed),
        "movie_promotion_rejection_reasons": _movie_refinement_rejection_reasons(
            grid_diagnostics,
            time_diagnostics,
        ),
        "grid_refinement_diagnostics": grid_diagnostics,
        "time_refinement_diagnostics": time_diagnostics,
    }
    report["next_campaign_suggestion"] = (
        build_essos_imported_drb_movie_refinement_next_campaign(report)
    )
    return report


def build_essos_imported_drb_movie_refinement_next_campaign(
    summary_report: dict[str, Any],
    *,
    max_total_cells: int | None = None,
) -> dict[str, Any]:
    """Suggest the next report-only movie refinement campaign settings.

    The suggestion is deterministic and deliberately conservative. It does not
    weaken the publication gate; it translates the dominant failed metrics into
    a concrete next grid/timestep candidate so high-resolution searches are
    auditable and repeatable.
    """

    grid_diagnostics = dict(summary_report.get("grid_refinement_diagnostics", {}))
    time_diagnostics = dict(summary_report.get("time_refinement_diagnostics", {}))
    current_grid = _movie_refinement_finest_grid(grid_diagnostics)
    grid_failed_metrics = _movie_refinement_failed_metric_names(grid_diagnostics)
    time_failed_metrics = _movie_refinement_failed_metric_names(time_diagnostics)
    grid_near_tolerance_metrics = _movie_refinement_near_tolerance_metric_names(
        grid_diagnostics
    )
    time_near_tolerance_metrics = _movie_refinement_near_tolerance_metric_names(
        time_diagnostics
    )
    current_potential_iterations = _movie_refinement_max_potential_iterations(
        grid_diagnostics,
        time_diagnostics,
    )
    potential_solve_action = _movie_refinement_potential_solve_action(
        grid_failed_metrics=grid_failed_metrics,
        time_failed_metrics=time_failed_metrics,
    )
    recommended_potential_iterations = (
        int(max(1, current_potential_iterations) * 2)
        if potential_solve_action != "no_potential_residual_blocker"
        and current_potential_iterations is not None
        else None
    )
    notes: list[str] = []
    grid_multiplier = [1.0, 1.0, 1.0]
    if current_grid is not None:
        if {"radial_flux_abs_mean", "radial_flux_rms"} & grid_failed_metrics:
            grid_multiplier = [
                max(grid_multiplier[0], 2.0),
                max(grid_multiplier[1], 1.5),
                max(grid_multiplier[2], 1.5),
            ]
            notes.append(
                "radial transport is grid-sensitive; refine radial and "
                "field-line-following transverse resolution together"
            )
        if (
            not bool(grid_diagnostics.get("spectral_resolution_passed", True))
            or "low_mode_spectral_power_fraction" in grid_failed_metrics
            or "spectral_edge_band_power_fraction" in grid_failed_metrics
        ):
            grid_multiplier = [
                max(grid_multiplier[0], 1.25),
                max(grid_multiplier[1], 2.0),
                max(grid_multiplier[2], 2.0),
            ]
            notes.append(
                "spectral content is too close to the grid edge; refine "
                "poloidal and toroidal movie-grid resolution before promotion"
            )
        if "spectral_centroid_poloidal_fraction" in grid_failed_metrics:
            grid_multiplier[1] = max(grid_multiplier[1], 2.0)
            notes.append("poloidal spectral centroid moves under refinement")
        if "spectral_centroid_toroidal_fraction" in grid_failed_metrics:
            grid_multiplier[2] = max(grid_multiplier[2], 2.0)
            notes.append("toroidal spectral centroid moves under refinement")
        if {"final_fluctuation_rms", "max_fluctuation_rms"} & grid_failed_metrics:
            grid_multiplier = [max(value, 1.5) for value in grid_multiplier]
            notes.append("scalar transient metrics are grid-sensitive")
        if "final_potential_residual_l2" in grid_failed_metrics:
            notes.append(
                "elliptic potential residual is grid-sensitive; rerun the same "
                "grid pair with a larger potential_iterations budget before "
                "escalating movie resolution solely because of the residual"
            )

    should_refine_grid = current_grid is not None and any(
        float(value) > 1.0 for value in grid_multiplier
    )
    suggested_next_grid = (
        _movie_refinement_scaled_grid(current_grid, tuple(grid_multiplier))
        if should_refine_grid
        else None
    )
    suggested_grid_shapes = (
        [list(current_grid), list(suggested_next_grid)]
        if current_grid is not None and suggested_next_grid is not None
        else []
    )
    current_cells = (
        int(np.prod(current_grid, dtype=np.int64)) if current_grid is not None else None
    )
    suggested_cells = (
        int(np.prod(suggested_next_grid, dtype=np.int64))
        if suggested_next_grid is not None
        else None
    )
    fits_budget = (
        None
        if max_total_cells is None or suggested_cells is None
        else bool(suggested_cells <= int(max_total_cells))
    )

    time_axis_values = [
        float(value) for value in time_diagnostics.get("axis_values", [])
    ]
    current_effective_frame_dt = (
        min(time_axis_values) if time_axis_values else None
    )
    recommended_time_values: list[float] = []
    time_action = "no_time_refinement_reports_available"
    if current_effective_frame_dt is not None:
        if bool(time_diagnostics.get("passed", False)):
            recommended_time_values = sorted(
                {
                    float(current_effective_frame_dt),
                    float(current_effective_frame_dt * 2.0),
                },
                reverse=True,
            )
            time_action = "reuse_current_timestep_pair_after_grid_change"
        elif not time_failed_metrics and not bool(
            time_diagnostics.get("spectral_resolution_passed", True)
        ):
            recommended_time_values = sorted(
                {float(value) for value in time_axis_values},
                reverse=True,
            )
            time_action = "fix_grid_resolution_before_reducing_timestep"
            notes.append(
                "time gate has no scalar-metric offender; fix spectral grid "
                "resolution before spending wall time on smaller timesteps"
            )
        else:
            recommended_time_values = sorted(
                {
                    float(current_effective_frame_dt),
                    float(current_effective_frame_dt * 0.5),
                },
                reverse=True,
            )
            time_action = "halve_effective_frame_dt_after_grid_candidate"
            notes.append("time-refinement scalar metrics are not stable")

    if current_grid is None:
        notes.append("at least two grid reports are required before suggesting a grid")
    if recommended_potential_iterations is not None:
        notes.append(
            "rerun the residual-blocked candidate with "
            f"potential_iterations={recommended_potential_iterations} before "
            "changing physics claims based on elliptic residual movement"
        )
    near_tolerance_metrics = grid_near_tolerance_metrics | time_near_tolerance_metrics
    if {"radial_flux_abs_mean", "radial_flux_rms"} & near_tolerance_metrics:
        notes.append(
            "radial-flux convergence is a near-tolerance miss; rerun the same "
            "grid with a longer transient or independent phase before treating "
            "the next larger grid as mandatory"
        )

    return {
        "diagnostic": "essos_imported_drb_movie_next_campaign_suggestion",
        "claim_scope": (
            "planning aid only; suggested settings are not validation evidence "
            "until the regenerated refinement summary passes"
        ),
        "publication_ready_current": bool(summary_report.get("publication_ready", False)),
        "grid_refinement_passed_current": bool(
            summary_report.get("grid_refinement_passed", False)
        ),
        "time_refinement_passed_current": bool(
            summary_report.get("time_refinement_passed", False)
        ),
        "dominant_grid_blockers": grid_diagnostics.get("dominant_failed_metrics", []),
        "dominant_time_blockers": time_diagnostics.get("dominant_failed_metrics", []),
        "near_tolerance_grid_blockers": grid_diagnostics.get(
            "near_tolerance_failed_metric_reports", []
        ),
        "near_tolerance_time_blockers": time_diagnostics.get(
            "near_tolerance_failed_metric_reports", []
        ),
        "current_finest_grid": list(current_grid) if current_grid is not None else None,
        "suggested_grid_multiplier": [float(value) for value in grid_multiplier],
        "suggested_next_grid": (
            list(suggested_next_grid) if suggested_next_grid is not None else None
        ),
        "suggested_grid_shapes": suggested_grid_shapes,
        "current_finest_grid_cell_count": current_cells,
        "suggested_next_grid_cell_count": suggested_cells,
        "max_total_cells": None if max_total_cells is None else int(max_total_cells),
        "suggested_grid_fits_cell_budget": fits_budget,
        "current_effective_frame_dt": current_effective_frame_dt,
        "recommended_time_effective_frame_dt_values": recommended_time_values,
        "time_refinement_action": time_action,
        "current_potential_iterations": current_potential_iterations,
        "recommended_potential_iterations": recommended_potential_iterations,
        "potential_solve_action": potential_solve_action,
        "recommendation_notes": notes,
    }


def build_essos_imported_drb_movie_refinement_diagnostics(
    reports: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    refinement_axis: str,
    relative_tolerance: float = 0.30,
) -> dict[str, Any]:
    """Compare scalar movie diagnostics across grid or timestep refinements."""

    normalized_axis = str(refinement_axis).strip().lower().replace("-", "_")
    if normalized_axis not in {"grid", "time"}:
        raise ValueError("refinement_axis must be either 'grid' or 'time'.")
    tolerance = float(relative_tolerance)
    if tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive.")
    report_list = [dict(report) for report in reports]
    base: dict[str, Any] = {
        "diagnostic": "essos_imported_drb_movie_refinement",
        "refinement_axis": normalized_axis,
        "report_count": len(report_list),
        "relative_tolerance": tolerance,
        "metric_keys": list(ESSOS_IMPORTED_DRB_MOVIE_REFINEMENT_METRICS),
        "pair_reports": [],
        "max_relative_metric_change": None,
        "axis_progression_passed": False,
        "all_reports_passed": False,
        "map_source_consistent": False,
        "spectral_resolution_passed": False,
        "spectral_resolution_reports": [],
        "failed_metric_reports": [],
        "near_tolerance_failed_metric_reports": [],
        "dominant_failed_metrics": [],
        "refinement_recommendations": [],
        "passed": False,
    }
    if len(report_list) < 2:
        return {
            **base,
            "reason": f"need_at_least_two_{normalized_axis}_reports",
        }

    sorted_reports = sorted(
        report_list,
        key=(
            _movie_grid_refinement_key
            if normalized_axis == "grid"
            else _movie_time_refinement_key
        ),
    )
    if normalized_axis == "time":
        sorted_reports = list(reversed(sorted_reports))
    map_sources = {str(report.get("map_source", "unknown")) for report in sorted_reports}
    labels = [_movie_report_label(report, index) for index, report in enumerate(sorted_reports)]
    axis_values = [
        _movie_grid_refinement_key(report)
        if normalized_axis == "grid"
        else _movie_time_refinement_key(report)
        for report in sorted_reports
    ]
    axis_progression_passed = all(
        current > previous
        for previous, current in zip(axis_values, axis_values[1:])
    )
    if normalized_axis == "time":
        axis_progression_passed = all(
            current < previous
            for previous, current in zip(axis_values, axis_values[1:])
        )

    pair_reports = [
        _build_movie_refinement_pair_report(
            coarse=coarse,
            fine=fine,
            coarse_label=coarse_label,
            fine_label=fine_label,
            relative_tolerance=tolerance,
        )
        for coarse, fine, coarse_label, fine_label in zip(
            sorted_reports,
            sorted_reports[1:],
            labels,
            labels[1:],
        )
    ]
    max_change_values = [
        float(pair["max_relative_metric_change"])
        for pair in pair_reports
        if pair["max_relative_metric_change"] is not None
    ]
    all_reports_passed = all(bool(report.get("passed")) for report in sorted_reports)
    map_source_consistent = len(map_sources) == 1
    spectral_resolution_reports = [
        _build_movie_spectral_resolution_report(report=report, label=label)
        for report, label in zip(sorted_reports, labels)
    ]
    spectral_resolution_passed = all(
        bool(report["passed"]) for report in spectral_resolution_reports
    )
    failed_metric_reports = _build_movie_failed_metric_reports(
        pair_reports=pair_reports,
        relative_tolerance=tolerance,
    )
    near_tolerance_failed_metric_reports = [
        report
        for report in failed_metric_reports
        if bool(report.get("near_tolerance"))
    ]
    refinement_recommendations = _movie_refinement_recommendations(
        refinement_axis=normalized_axis,
        failed_metric_reports=failed_metric_reports,
        near_tolerance_failed_metric_reports=near_tolerance_failed_metric_reports,
        spectral_resolution_passed=spectral_resolution_passed,
    )
    passed = bool(
        all_reports_passed
        and map_source_consistent
        and axis_progression_passed
        and spectral_resolution_passed
        and pair_reports
        and all(bool(pair["passed"]) for pair in pair_reports)
    )
    return {
        **base,
        "map_source": next(iter(map_sources)) if map_source_consistent else None,
        "map_sources": sorted(map_sources),
        "labels": labels,
        "axis_values": [float(value) for value in axis_values],
        "axis_progression_passed": bool(axis_progression_passed),
        "all_reports_passed": bool(all_reports_passed),
        "map_source_consistent": bool(map_source_consistent),
        "spectral_resolution_passed": bool(spectral_resolution_passed),
        "spectral_resolution_reports": spectral_resolution_reports,
        "pair_reports": pair_reports,
        "failed_metric_reports": failed_metric_reports,
        "near_tolerance_failed_metric_reports": near_tolerance_failed_metric_reports,
        "dominant_failed_metrics": failed_metric_reports[:5],
        "refinement_recommendations": refinement_recommendations,
        "max_relative_metric_change": (
            max(max_change_values) if max_change_values else None
        ),
        "passed": passed,
    }


def _load_movie_refinement_reports(paths: tuple[str | Path, ...] | list[str | Path]) -> tuple[dict[str, Any], ...]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        resolved = Path(path)
        reports.append(json.loads(resolved.read_text(encoding="utf-8")))
    return tuple(reports)


def _movie_refinement_rejection_reasons(
    grid_diagnostics: dict[str, Any],
    time_diagnostics: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not bool(grid_diagnostics["passed"]):
        reasons.append("movie_grid_refinement_not_passed")
        if "reason" in grid_diagnostics:
            reasons.append(str(grid_diagnostics["reason"]))
        if grid_diagnostics.get("spectral_resolution_reports") and not bool(
            grid_diagnostics.get("spectral_resolution_passed", False)
        ):
            reasons.append("movie_grid_spectral_resolution_not_passed")
            reasons.extend(_movie_spectral_resolution_rejection_reasons(grid_diagnostics))
    if not bool(time_diagnostics["passed"]):
        reasons.append("movie_time_refinement_not_passed")
        if "reason" in time_diagnostics:
            reasons.append(str(time_diagnostics["reason"]))
        if time_diagnostics.get("spectral_resolution_reports") and not bool(
            time_diagnostics.get("spectral_resolution_passed", False)
        ):
            reasons.append("movie_time_spectral_resolution_not_passed")
            reasons.extend(_movie_spectral_resolution_rejection_reasons(time_diagnostics))
    return reasons


def _movie_spectral_resolution_rejection_reasons(diagnostics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for report in diagnostics.get("spectral_resolution_reports", []):
        for reason in report.get("reasons", []):
            reason_text = str(reason)
            if reason_text not in reasons:
                reasons.append(reason_text)
    return reasons


def _build_movie_spectral_resolution_report(
    *,
    report: dict[str, Any],
    label: str,
    max_edge_band_fraction: float = ESSOS_IMPORTED_DRB_MOVIE_MAX_EDGE_BAND_FRACTION,
) -> dict[str, Any]:
    reasons: list[str] = []
    low_mode_window_covers_grid = bool(report.get("low_mode_window_covers_grid", False))
    if low_mode_window_covers_grid:
        reasons.append("low_mode_window_covers_grid")
    edge_fraction = _optional_float(report.get("spectral_edge_band_power_fraction"))
    if edge_fraction is None:
        reasons.append("missing_spectral_edge_band_power_fraction")
    elif edge_fraction > max_edge_band_fraction:
        reasons.append("spectral_edge_band_power_fraction_above_limit")
    poloidal_fraction = _optional_float(report.get("spectral_centroid_poloidal_fraction"))
    toroidal_fraction = _optional_float(report.get("spectral_centroid_toroidal_fraction"))
    if poloidal_fraction is None:
        reasons.append("missing_spectral_centroid_poloidal_fraction")
    elif not 0.0 <= poloidal_fraction <= 1.0:
        reasons.append("spectral_centroid_poloidal_fraction_out_of_bounds")
    if toroidal_fraction is None:
        reasons.append("missing_spectral_centroid_toroidal_fraction")
    elif not 0.0 <= toroidal_fraction <= 1.0:
        reasons.append("spectral_centroid_toroidal_fraction_out_of_bounds")
    return {
        "label": label,
        "low_mode_window_covers_grid": bool(low_mode_window_covers_grid),
        "spectral_edge_band_power_fraction": edge_fraction,
        "max_spectral_edge_band_power_fraction": float(max_edge_band_fraction),
        "spectral_centroid_poloidal_fraction": poloidal_fraction,
        "spectral_centroid_toroidal_fraction": toroidal_fraction,
        "reasons": reasons,
        "passed": not reasons,
    }


def _movie_grid_refinement_key(report: dict[str, Any]) -> float:
    grid = report.get("movie_physics_grid", ())
    if not isinstance(grid, (list, tuple)) or len(grid) != 3:
        return 0.0
    product = 1
    for value in grid:
        product *= int(value)
    return float(product)


def _movie_time_refinement_key(report: dict[str, Any]) -> float:
    return float(report.get("dt", 0.0)) * float(report.get("substeps_per_frame", 0.0))


def _movie_report_label(report: dict[str, Any], index: int) -> str:
    case = report.get("case")
    grid = report.get("movie_physics_grid")
    grid_label = ""
    if isinstance(grid, (list, tuple)) and len(grid) == 3:
        grid_label = f"{int(grid[0])}x{int(grid[1])}x{int(grid[2])}"
    frame_dt = _movie_time_refinement_key(report)
    if case:
        suffix = f"{grid_label}_frame_dt={frame_dt:g}" if grid_label else f"frame_dt={frame_dt:g}"
        return f"{case}:{suffix}"
    if grid_label:
        return f"report_{index}_{grid_label}_frame_dt={frame_dt:g}"
    return f"report_{index}"


def _build_movie_refinement_pair_report(
    *,
    coarse: dict[str, Any],
    fine: dict[str, Any],
    coarse_label: str,
    fine_label: str,
    relative_tolerance: float,
) -> dict[str, Any]:
    metric_reports: dict[str, dict[str, Any]] = {}
    relative_changes: list[float] = []
    for key in ESSOS_IMPORTED_DRB_MOVIE_REFINEMENT_METRICS:
        coarse_value = _optional_float(coarse.get(key))
        fine_value = _optional_float(fine.get(key))
        if coarse_value is None or fine_value is None:
            metric_reports[key] = {
                "coarse": coarse_value,
                "fine": fine_value,
                "relative_change": None,
                "sign_agreement": False,
                "passed": False,
            }
            continue
        denominator_floor = _movie_refinement_metric_floor(key)
        denominator = max(abs(fine_value), denominator_floor)
        relative_change = abs(coarse_value - fine_value) / denominator
        relative_changes.append(float(relative_change))
        metric_reports[key] = {
            "coarse": coarse_value,
            "fine": fine_value,
            "denominator_floor": float(denominator_floor),
            "relative_change": float(relative_change),
            "sign_agreement": True,
            "passed": bool(relative_change <= relative_tolerance),
        }
    max_change = max(relative_changes) if relative_changes else None
    metric_passed = all(bool(item["passed"]) for item in metric_reports.values())
    radial_flux_proxy_sign_agreement = _signed_metric_agrees(
        coarse.get("radial_flux_proxy"),
        fine.get("radial_flux_proxy"),
    )
    return {
        "coarse_label": coarse_label,
        "fine_label": fine_label,
        "coarse_grid": list(coarse.get("movie_physics_grid", [])),
        "fine_grid": list(fine.get("movie_physics_grid", [])),
        "coarse_effective_frame_dt": _movie_time_refinement_key(coarse),
        "fine_effective_frame_dt": _movie_time_refinement_key(fine),
        "coarse_potential_iterations": _optional_int(
            coarse.get("potential_iterations")
        ),
        "fine_potential_iterations": _optional_int(fine.get("potential_iterations")),
        "coarse_potential_preconditioner": coarse.get("potential_preconditioner"),
        "fine_potential_preconditioner": fine.get("potential_preconditioner"),
        "metric_reports": metric_reports,
        "max_relative_metric_change": max_change,
        "radial_flux_proxy_sign_agreement": bool(radial_flux_proxy_sign_agreement),
        "radial_flux_sign_passed": bool(radial_flux_proxy_sign_agreement),
        "passed": bool(metric_passed),
    }


def _build_movie_failed_metric_reports(
    *,
    pair_reports: list[dict[str, Any]],
    relative_tolerance: float,
) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pair_reports):
        for metric, report in pair.get("metric_reports", {}).items():
            if bool(report.get("passed")):
                continue
            relative_change = report.get("relative_change")
            near_tolerance = (
                relative_change is not None
                and float(relative_change)
                <= float(relative_tolerance)
                * ESSOS_IMPORTED_DRB_MOVIE_REFINEMENT_NEAR_TOLERANCE_FACTOR
            )
            failed.append(
                {
                    "pair_index": int(pair_index),
                    "coarse_label": pair.get("coarse_label"),
                    "fine_label": pair.get("fine_label"),
                    "metric": str(metric),
                    "coarse": report.get("coarse"),
                    "fine": report.get("fine"),
                    "denominator_floor": report.get("denominator_floor"),
                    "relative_change": relative_change,
                    "relative_tolerance": float(relative_tolerance),
                    "near_tolerance": bool(near_tolerance),
                    "near_tolerance_factor": float(
                        ESSOS_IMPORTED_DRB_MOVIE_REFINEMENT_NEAR_TOLERANCE_FACTOR
                    ),
                    "reason": _movie_refinement_metric_failure_reason(
                        metric=str(metric),
                        report=report,
                        relative_tolerance=relative_tolerance,
                    ),
                }
            )
    return sorted(
        failed,
        key=lambda item: (
            item["relative_change"] is None,
            -float(item["relative_change"] or 0.0),
            str(item["metric"]),
        ),
    )


def _movie_refinement_metric_failure_reason(
    *,
    metric: str,
    report: dict[str, Any],
    relative_tolerance: float,
) -> str:
    if report.get("relative_change") is None:
        return "missing_or_nonfinite_metric"
    if str(metric).startswith("spectral_") or str(metric) == "low_mode_spectral_power_fraction":
        return "spectral_content_not_grid_or_time_stable"
    if str(metric).startswith("radial_flux_"):
        return "radial_transport_not_grid_or_time_stable"
    if str(metric) == "final_potential_residual_l2":
        return "elliptic_residual_not_grid_or_time_stable"
    return f"relative_change_above_{float(relative_tolerance):g}"


def _movie_refinement_recommendations(
    *,
    refinement_axis: str,
    failed_metric_reports: list[dict[str, Any]],
    near_tolerance_failed_metric_reports: list[dict[str, Any]],
    spectral_resolution_passed: bool,
) -> list[str]:
    recommendations: list[str] = []
    failed_metrics = {str(report["metric"]) for report in failed_metric_reports}
    near_tolerance_metrics = {
        str(report["metric"]) for report in near_tolerance_failed_metric_reports
    }
    if not spectral_resolution_passed:
        recommendations.append(
            "Increase the physics/movie grid or reduce the resolved low-mode window "
            "before promoting the movie; the spectrum is too close to the grid edge."
        )
    radial_metrics = {"radial_flux_abs_mean", "radial_flux_rms"} & failed_metrics
    if radial_metrics and radial_metrics <= near_tolerance_metrics:
        recommendations.append(
            "Radial transport is only marginally above the convergence tolerance; "
            "repeat the same grid with a longer transient or independent phase "
            "before paying for a larger radial/toroidal refinement."
        )
    elif radial_metrics:
        recommendations.append(
            "Treat radial transport as unresolved: refine the radial grid and the "
            "field-line-following transverse grid, then rerun the same transient."
        )
    if "spectral_centroid_toroidal_fraction" in failed_metrics:
        recommendations.append(
            "Refine toroidal resolution and map sampling; the turbulent spectrum is "
            "moving in toroidal-mode space across refinement levels."
        )
    if "spectral_centroid_poloidal_fraction" in failed_metrics:
        recommendations.append(
            "Refine poloidal resolution and interpolation order; the turbulent "
            "spectrum is moving in poloidal-mode space across refinement levels."
        )
    if "final_potential_residual_l2" in failed_metrics:
        recommendations.append(
            "Check the elliptic potential solve tolerance and conditioning only after "
            "transport and spectral metrics are stable."
        )
    if not recommendations and failed_metric_reports:
        recommendations.append(
            f"Rerun the {refinement_axis} refinement with the dominant failed metric "
            "as the primary convergence observable."
        )
    return recommendations


def _movie_refinement_failed_metric_names(diagnostics: dict[str, Any]) -> set[str]:
    return {
        str(report.get("metric"))
        for report in diagnostics.get("failed_metric_reports", [])
        if report.get("metric") is not None
    }


def _movie_refinement_near_tolerance_metric_names(
    diagnostics: dict[str, Any],
) -> set[str]:
    return {
        str(report.get("metric"))
        for report in diagnostics.get("near_tolerance_failed_metric_reports", [])
        if report.get("metric") is not None
    }


def _movie_refinement_potential_solve_action(
    *,
    grid_failed_metrics: set[str],
    time_failed_metrics: set[str],
) -> str:
    residual_failed = "final_potential_residual_l2" in (
        set(grid_failed_metrics) | set(time_failed_metrics)
    )
    if not residual_failed:
        return "no_potential_residual_blocker"
    physics_metrics = (
        set(grid_failed_metrics) | set(time_failed_metrics)
    ) - {"final_potential_residual_l2"}
    if physics_metrics:
        return "check_potential_solver_after_primary_physics_metric_refinement"
    return "rerun_same_grid_time_pair_with_larger_potential_iterations"


def _movie_refinement_max_potential_iterations(
    *diagnostics_items: dict[str, Any],
) -> int | None:
    values: list[int] = []
    for diagnostics in diagnostics_items:
        for pair in diagnostics.get("pair_reports", []):
            for key in ("coarse_potential_iterations", "fine_potential_iterations"):
                value = _optional_int(pair.get(key))
                if value is not None:
                    values.append(value)
    if not values:
        return None
    return max(values)


def _movie_refinement_finest_grid(
    diagnostics: dict[str, Any],
) -> tuple[int, int, int] | None:
    grids: list[tuple[int, int, int]] = []
    for pair in diagnostics.get("pair_reports", []):
        for key in ("coarse_grid", "fine_grid"):
            grid = pair.get(key)
            if isinstance(grid, (list, tuple)) and len(grid) == 3:
                try:
                    grids.append(tuple(int(value) for value in grid))
                except (TypeError, ValueError):
                    continue
    if not grids:
        return None
    return max(grids, key=lambda grid: int(np.prod(grid, dtype=np.int64)))


def _movie_refinement_scaled_grid(
    grid: tuple[int, int, int],
    multipliers: tuple[float, float, float],
) -> tuple[int, int, int]:
    suggested = []
    for axis_size, multiplier in zip(grid, multipliers, strict=True):
        if float(multiplier) <= 1.0:
            scaled = int(axis_size)
        else:
            scaled = int(np.ceil(float(axis_size) * float(multiplier)))
            if scaled <= int(axis_size):
                scaled = int(axis_size) + 1
        suggested.append(scaled)
    # Keep the rFFT/toroidal direction even where possible for cleaner spectra.
    if suggested[2] % 2:
        suggested[2] += 1
    return tuple(suggested)


def _movie_refinement_metric_floor(key: str) -> float:
    return float(
        ESSOS_IMPORTED_DRB_MOVIE_REFINEMENT_METRIC_FLOORS.get(
            key,
            ESSOS_IMPORTED_DRB_MOVIE_REFINEMENT_DEFAULT_METRIC_FLOOR,
        )
    )


def _signed_metric_agrees(coarse_value: Any, fine_value: Any, *, floor: float = 1.0e-12) -> bool:
    coarse_float = _optional_float(coarse_value)
    fine_float = _optional_float(fine_value)
    if coarse_float is None or fine_float is None:
        return False
    if max(abs(coarse_float), abs(fine_float)) <= floor:
        return True
    return bool(np.sign(coarse_float) == np.sign(fine_float))


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _same_optional_float(value: Any, expected: float, *, rtol: float = 1.0e-12) -> bool:
    result = _optional_float(value)
    if result is None:
        return False
    return bool(np.isclose(result, float(expected), rtol=rtol, atol=rtol))


def _same_optional_int(value: Any, expected: int) -> bool:
    result = _optional_int(value)
    return result is not None and result == int(expected)


def _normalize_optional_report_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def create_essos_imported_drb_movie_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_imported_drb_movie_campaign",
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    map_source: str = "coil",
    nx: int = 8,
    ny: int = 28,
    nz: int = 80,
    rho_min: float = 0.20,
    rho_max: float = 0.92,
    maxtime: float = 135.0,
    times_to_trace: int = 720,
    frames: int = 32,
    substeps_per_frame: int = 6,
    dt: float = 1.2e-3,
    potential_iterations: int = ESSOS_IMPORTED_DRB_MOVIE_DEFAULT_POTENTIAL_ITERATIONS,
    potential_regularization: float = ESSOS_IMPORTED_DRB_MOVIE_DEFAULT_POTENTIAL_REGULARIZATION,
    potential_preconditioner: str | None = (
        ESSOS_IMPORTED_DRB_MOVIE_DEFAULT_POTENTIAL_PRECONDITIONER
    ),
) -> EssosImportedDrbMovieArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    movies_dir = root / "movies"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    movies_dir.mkdir(parents=True, exist_ok=True)

    result = build_essos_imported_drb_movie_campaign(
        coil_json_path=coil_json_path,
        vmec_wout_path=vmec_wout_path,
        essos_root=essos_root,
        map_source=map_source,
        nx=nx,
        ny=ny,
        nz=nz,
        rho_min=rho_min,
        rho_max=rho_max,
        maxtime=maxtime,
        times_to_trace=times_to_trace,
        frames=frames,
        substeps_per_frame=substeps_per_frame,
        dt=dt,
        potential_iterations=potential_iterations,
        potential_regularization=potential_regularization,
        potential_preconditioner=potential_preconditioner,
    )
    report = dict(result.report)
    report_json_path = data_dir / f"{case_label}.json"
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **result.arrays)
    snapshot_png_path = images_dir / f"{case_label}_snapshots.png"
    save_essos_imported_drb_snapshot_panel(result.geometry, result.arrays, snapshot_png_path)
    diagnostics_png_path = images_dir / f"{case_label}_diagnostics.png"
    save_essos_imported_drb_diagnostics_panel(result.geometry, result.report, result.arrays, diagnostics_png_path)
    poster_png_path = images_dir / f"{case_label}_poster.png"
    save_essos_imported_drb_3d_frame(
        result.geometry,
        result.arrays["density_fluctuation_history"][-1],
        float(result.arrays["time"][-1]),
        poster_png_path,
        vmax=float(result.arrays["movie_vmax"][0]),
    )
    movie_gif_path = movies_dir / f"{case_label}.gif"
    save_essos_imported_drb_3d_movie(result.geometry, result.arrays, movie_gif_path)
    report.update(_audit_movie_gif(movie_gif_path))
    _write_strict_json(report_json_path, report)
    return EssosImportedDrbMovieArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        snapshot_png_path=snapshot_png_path,
        diagnostics_png_path=diagnostics_png_path,
        poster_png_path=poster_png_path,
        movie_gif_path=movie_gif_path,
    )


def build_essos_imported_drb_movie_campaign(
    *,
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    map_source: str = "coil",
    nx: int = 8,
    ny: int = 28,
    nz: int = 80,
    rho_min: float = 0.20,
    rho_max: float = 0.92,
    maxtime: float = 135.0,
    times_to_trace: int = 720,
    frames: int = 32,
    substeps_per_frame: int = 6,
    dt: float = 1.2e-3,
    potential_iterations: int = ESSOS_IMPORTED_DRB_MOVIE_DEFAULT_POTENTIAL_ITERATIONS,
    potential_regularization: float = ESSOS_IMPORTED_DRB_MOVIE_DEFAULT_POTENTIAL_REGULARIZATION,
    potential_preconditioner: str | None = (
        ESSOS_IMPORTED_DRB_MOVIE_DEFAULT_POTENTIAL_PRECONDITIONER
    ),
) -> EssosImportedDrbMovieResult:
    geometry = build_essos_imported_fci_geometry(
        coil_json_path=coil_json_path,
        vmec_wout_path=vmec_wout_path,
        essos_root=essos_root,
        map_source=map_source,
        nx=nx,
        ny=ny,
        nz=nz,
        rho_min=rho_min,
        rho_max=rho_max,
        maxtime=maxtime,
        times_to_trace=times_to_trace,
    )
    parameters = FciDrbRhsParameters(
        recycling_fraction=0.965,
        recycled_neutral_energy=0.026,
        vorticity_diffusivity=3.5e-4,
        potential_iterations=int(potential_iterations),
        potential_regularization=float(potential_regularization),
        potential_preconditioner=potential_preconditioner,
    )
    run_movie = _build_essos_imported_movie_scan(
        geometry,
        parameters=parameters,
        frames=frames,
        substeps_per_frame=substeps_per_frame,
        dt=dt,
    )
    initial = _seed_movie_multimode_fluctuations(initial_essos_imported_drb_state(geometry, drive_scale=1.08), geometry)
    t0 = time.perf_counter()
    final_state, movie_history, diagnostics = run_movie(initial)
    _block_until_ready((final_state, movie_history, diagnostics))
    execute_seconds = time.perf_counter() - t0

    movie_history_np = np.asarray(movie_history, dtype=np.float64)
    diagnostics_np = np.asarray(diagnostics, dtype=np.float64)
    final_state_np = _state_to_numpy(final_state)
    final_sheath, final_neutral = _final_closure_diagnostics(geometry, final_state)
    report = _build_essos_imported_drb_movie_report(
        geometry=geometry,
        movie_history=movie_history_np,
        diagnostics=diagnostics_np,
        final_state=final_state_np,
        final_sheath=final_sheath,
        final_neutral=final_neutral,
        frames=frames,
        substeps_per_frame=substeps_per_frame,
        dt=dt,
        execute_seconds=execute_seconds,
        potential_iterations=int(parameters.potential_iterations),
        potential_regularization=float(parameters.potential_regularization),
        potential_preconditioner=parameters.potential_preconditioner,
    )
    arrays = _build_essos_imported_drb_movie_arrays(
        geometry=geometry,
        movie_history=movie_history_np,
        diagnostics=diagnostics_np,
        final_state=final_state_np,
        frame_dt=float(dt) * float(substeps_per_frame),
    )
    return EssosImportedDrbMovieResult(geometry=geometry, report=report, arrays=arrays)


def save_essos_imported_drb_snapshot_panel(
    geometry: EssosImportedFciGeometry,
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    history = np.asarray(arrays["density_fluctuation_history"], dtype=np.float64)
    time = np.asarray(arrays["time"], dtype=np.float64)
    major_radius, vertical = _major_radius_and_vertical(geometry)
    time_indices = np.asarray([0, history.shape[0] // 2, history.shape[0] - 1], dtype=int)
    toroidal_indices = np.linspace(0, geometry.shape[1] - 1, min(4, geometry.shape[1]), dtype=int)
    vmax = float(arrays["movie_vmax"][0])
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axes = plt.subplots(
        len(time_indices),
        len(toroidal_indices),
        figsize=(4.1 * len(toroidal_indices), 3.3 * len(time_indices)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    image = None
    for row, time_index in enumerate(time_indices):
        for col, toroidal_index in enumerate(toroidal_indices):
            axis = axes[row, col]
            image = axis.pcolormesh(
                major_radius[:, toroidal_index, :],
                vertical[:, toroidal_index, :],
                history[time_index, :, toroidal_index, :],
                shading="gouraud",
                cmap="coolwarm",
                norm=norm,
            )
            axis.plot(major_radius[0, toroidal_index, :], vertical[0, toroidal_index, :], color="white", lw=1.4)
            axis.plot(major_radius[-1, toroidal_index, :], vertical[-1, toroidal_index, :], color="0.20", lw=1.0)
            axis.set_aspect("equal", adjustable="box")
            phi_value = 2.0 * np.pi * toroidal_index / max(geometry.shape[1], 1)
            axis.set_title(rf"$t={time[time_index]:.3f}$, $\phi={phi_value:.2f}$")
            axis.set_xlabel("R")
            axis.set_ylabel("Z")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.76, label=r"$\tilde{n}_i/\langle n_i\rangle_\phi$")
    fig.suptitle(f"ESSOS-imported {_essos_imported_map_label(geometry)} DRB transient: density fluctuations on FCI planes", fontsize=15)
    fig.savefig(resolved, dpi=180)
    plt.close(fig)
    return resolved


def save_essos_imported_drb_diagnostics_panel(
    geometry: EssosImportedFciGeometry,
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = np.asarray(arrays["diagnostics"], dtype=np.float64)
    time = np.asarray(arrays["time"], dtype=np.float64)
    major_radius, vertical = _major_radius_and_vertical(geometry)
    toroidal_index = min(geometry.shape[1] // 3, geometry.shape[1] - 1)
    endpoint_count = np.asarray(arrays["endpoint_count_toroidal"], dtype=np.float64)
    spectrum = np.asarray(arrays["final_spectrum_log10"], dtype=np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.5), constrained_layout=True)
    density_image = axes[0, 0].pcolormesh(
        major_radius[:, toroidal_index, :],
        vertical[:, toroidal_index, :],
        arrays["final_ion_density"][:, toroidal_index, :],
        shading="gouraud",
        cmap="turbo",
    )
    axes[0, 0].set_title("final ion density")
    fig.colorbar(density_image, ax=axes[0, 0], label=r"$N_i$")

    neutral_image = axes[0, 1].pcolormesh(
        major_radius[:, toroidal_index, :],
        vertical[:, toroidal_index, :],
        arrays["final_neutral_density"][:, toroidal_index, :],
        shading="gouraud",
        cmap="magma",
    )
    axes[0, 1].set_title("final neutral density")
    fig.colorbar(neutral_image, ax=axes[0, 1], label=r"$N_n$")

    endpoint_image = axes[0, 2].imshow(endpoint_count.T, origin="lower", aspect="auto", cmap="inferno")
    axes[0, 2].set_title("imported endpoint count")
    axes[0, 2].set_xlabel("toroidal index")
    axes[0, 2].set_ylabel("poloidal index")
    fig.colorbar(endpoint_image, ax=axes[0, 2], label="target crossings")

    axes[1, 0].plot(time, diagnostics[:, 0], lw=2.2, label="fluctuation RMS")
    axes[1, 0].plot(time, diagnostics[:, 1], lw=2.0, label="mean ion density")
    axes[1, 0].plot(time, diagnostics[:, 2], lw=2.0, label="mean neutral density")
    axes[1, 0].set_title("global transient diagnostics")
    axes[1, 0].set_xlabel("normalized time")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    spectrum_image = axes[1, 1].imshow(spectrum.T, origin="lower", aspect="auto", cmap="viridis")
    axes[1, 1].set_title("final toroidal-poloidal spectrum")
    axes[1, 1].set_xlabel("toroidal mode index")
    axes[1, 1].set_ylabel("poloidal mode index")
    fig.colorbar(spectrum_image, ax=axes[1, 1], label=r"$\log_{10}$ power")

    radial = np.asarray(arrays["radial_coordinate"], dtype=np.float64)
    axes[1, 2].plot(radial, arrays["final_radial_flux_proxy"], lw=2.2, color="#005f73")
    axes[1, 2].axhline(0.0, lw=0.9, color="0.35")
    axes[1, 2].set_title("final radial flux proxy")
    axes[1, 2].set_xlabel(r"$\rho$")
    axes[1, 2].set_ylabel(r"$\langle \tilde{n}_i \tilde{v}_\rho\rangle$")
    axes[1, 2].grid(alpha=0.25)
    axes[1, 2].text(
        0.04,
        0.95,
        "\n".join(
            [
                f"RMS = {report['final_fluctuation_rms']:.2e}",
                f"endpoint frac. = {report['endpoint_fraction']:.2f}",
                f"sheath residual = {report['particle_recycling_relative_error']:.1e}",
                f"neutral residual = {report['neutral_particle_relative_error']:.1e}",
            ]
        ),
        transform=axes[1, 2].transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "0.82"},
    )

    for axis in (axes[0, 0], axes[0, 1]):
        axis.plot(major_radius[0, toroidal_index, :], vertical[0, toroidal_index, :], color="white", lw=1.4)
        axis.plot(major_radius[-1, toroidal_index, :], vertical[-1, toroidal_index, :], color="0.22", lw=1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("R")
        axis.set_ylabel("Z")
    fig.suptitle(f"ESSOS-imported {_essos_imported_map_label(geometry)} DRB transient: sheath, recycling, neutrals, and fluctuation gates", fontsize=15)
    fig.savefig(resolved, dpi=180)
    plt.close(fig)
    return resolved


def save_essos_imported_drb_3d_movie(
    geometry: EssosImportedFciGeometry,
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    history = np.asarray(arrays["density_fluctuation_history"], dtype=np.float64)
    time_values = np.asarray(arrays["time"], dtype=np.float64)
    frame_indices = np.linspace(0, history.shape[0] - 1, min(24, history.shape[0]), dtype=int)
    vmax = float(arrays["movie_vmax"][0])
    with tempfile.TemporaryDirectory(prefix="jax_drb_essos_drb_movie_") as temp_dir:
        frame_paths = []
        for local_index, frame_index in enumerate(frame_indices):
            frame_path = Path(temp_dir) / f"frame_{local_index:03d}.png"
            save_essos_imported_drb_3d_frame(
                geometry,
                history[frame_index],
                float(time_values[frame_index]),
                frame_path,
                vmax=vmax,
            )
            frame_paths.append(frame_path)
        first = Image.open(frame_paths[0]).convert("RGB").quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        images = [first]
        for frame_path in frame_paths[1:]:
            images.append(Image.open(frame_path).convert("RGB").quantize(palette=first, dither=Image.Dither.NONE))
        images[0].save(resolved, save_all=True, append_images=images[1:], duration=120, loop=0)
        for image in images:
            image.close()
    return resolved


def save_essos_imported_drb_3d_frame(
    geometry: EssosImportedFciGeometry,
    field: np.ndarray,
    time_value: float,
    path: str | Path,
    *,
    vmax: float | None = None,
) -> Path:
    if max(geometry.shape) >= 16:
        try:
            return _save_essos_imported_drb_3d_frame_pyvista(geometry, field, time_value, path, vmax=vmax)
        except Exception:
            pass
    return _save_essos_imported_drb_3d_frame_matplotlib(geometry, field, time_value, path, vmax=vmax)


def _build_essos_imported_movie_scan(
    geometry: EssosImportedFciGeometry,
    *,
    parameters: FciDrbRhsParameters,
    frames: int,
    substeps_per_frame: int,
    dt: float,
):
    radial = _normalized_minor_radius_jax(geometry)
    curvature_proxy = _magnetic_curvature_proxy_jax(geometry)
    source_envelope = jnp.exp(-jnp.square((radial - 0.30) / 0.20))
    neutral_puff_envelope = jnp.exp(-jnp.square((radial - 0.86) / 0.12))
    edge_sink_envelope = jnp.exp(-jnp.square((radial - 0.98) / 0.16))
    theta = geometry.poloidal_angle
    phi = geometry.toroidal_angle
    helical = jnp.sin(2.0 * theta - phi) + 0.35 * jnp.cos(3.0 * theta - 2.0 * phi)
    helical = helical / jnp.maximum(jnp.std(helical), 1.0e-12)
    fine_helical = (
        0.65 * jnp.sin(5.0 * theta - 3.0 * phi + 0.45 * curvature_proxy)
        + 0.45 * jnp.cos(7.0 * theta + 2.0 * phi)
        + 0.25 * jnp.sin(11.0 * theta - 5.0 * phi)
    )
    fine_helical = fine_helical / jnp.maximum(jnp.std(fine_helical), 1.0e-12)

    def step_state(state: FciDrbState, scalar_time: jax.Array) -> FciDrbState:
        result = compute_fci_drb_rhs(state, maps=geometry.maps, metric=geometry.metric, parameters=parameters)
        phi_field = result.potential
        pressure = state.ion_pressure + state.electron_pressure
        grad_pressure = _radial_derivative(pressure, geometry)
        fluctuation_drive = source_envelope * (
            1.0
            + 0.18 * jnp.sin(31.0 * scalar_time + helical)
            + 0.120 * jnp.sin(53.0 * scalar_time + fine_helical)
            + 0.070 * fine_helical
        )
        neutral_puff = neutral_puff_envelope * (
            1.0 + 0.16 * jnp.cos(17.0 * scalar_time - helical) + 0.055 * jnp.sin(37.0 * scalar_time + fine_helical)
        )
        ion_diffusion = conservative_perp_diffusion_xz(state.ion_density, 3.0e-4 * jnp.ones_like(state.ion_density), geometry.metric)
        electron_diffusion = conservative_perp_diffusion_xz(
            state.electron_density,
            3.0e-4 * jnp.ones_like(state.electron_density),
            geometry.metric,
        )
        ion_pressure_diffusion = conservative_perp_diffusion_xz(
            state.ion_pressure,
            2.5e-4 * jnp.ones_like(state.ion_pressure),
            geometry.metric,
        )
        electron_pressure_diffusion = conservative_perp_diffusion_xz(
            state.electron_pressure,
            2.5e-4 * jnp.ones_like(state.electron_pressure),
            geometry.metric,
        )
        ion_adv = _logical_exb_advection(phi_field, state.ion_density, geometry)
        electron_adv = _logical_exb_advection(phi_field, state.electron_density, geometry)
        neutral_adv = 0.35 * _logical_exb_advection(phi_field, state.neutral_density, geometry)
        pressure_adv = _logical_exb_advection(phi_field, pressure, geometry)
        vorticity_adv = _logical_exb_advection(phi_field, state.vorticity, geometry)
        edge_particle_sink = 0.030 * edge_sink_envelope * state.ion_density
        edge_energy_sink = 0.030 * edge_sink_envelope * pressure

        source_strength = 0.13
        neutral_puff_strength = 0.045
        rhs = FciDrbState(
            ion_density=(
                result.rhs.ion_density
                - 0.070 * ion_adv
                + ion_diffusion
                + source_strength * fluctuation_drive
                - edge_particle_sink
            ),
            electron_density=(
                result.rhs.electron_density
                - 0.070 * electron_adv
                + electron_diffusion
                + source_strength * fluctuation_drive
                - edge_particle_sink
                + 0.22 * (state.ion_density - state.electron_density)
            ),
            neutral_density=result.rhs.neutral_density - 0.022 * neutral_adv + neutral_puff_strength * neutral_puff,
            ion_pressure=(
                result.rhs.ion_pressure
                - 0.042 * pressure_adv
                + ion_pressure_diffusion
                + 0.026 * fluctuation_drive
                - 0.45 * edge_energy_sink
            ),
            electron_pressure=(
                result.rhs.electron_pressure
                - 0.042 * pressure_adv
                + electron_pressure_diffusion
                + 0.035 * fluctuation_drive
                - 0.55 * edge_energy_sink
            ),
            neutral_pressure=result.rhs.neutral_pressure + 0.012 * neutral_puff - 0.010 * neutral_adv,
            ion_momentum=result.rhs.ion_momentum - 0.018 * _logical_exb_advection(phi_field, state.ion_momentum, geometry),
            neutral_momentum=result.rhs.neutral_momentum - 0.010 * neutral_adv,
            vorticity=(
                result.rhs.vorticity
                - 0.050 * vorticity_adv
                + 0.034 * curvature_proxy * grad_pressure
                + 0.011 * curvature_proxy * fluctuation_drive
                + 0.010 * source_envelope * fine_helical
                - 0.028 * state.vorticity
            ),
        )
        return _clip_movie_state(_add_scaled_state(state, rhs, dt))

    def run(initial_state: FciDrbState) -> tuple[FciDrbState, jax.Array, jax.Array]:
        def frame_step(state: FciDrbState, frame_index: jax.Array) -> tuple[FciDrbState, tuple[jax.Array, jax.Array]]:
            def substep(local_index: int, carry: FciDrbState) -> FciDrbState:
                scalar_time = (frame_index * int(substeps_per_frame) + local_index) * float(dt)
                return step_state(carry, scalar_time)

            next_state = jax.lax.fori_loop(0, int(substeps_per_frame), substep, state)
            movie_field = _density_fluctuation(next_state.ion_density)
            result = compute_fci_drb_rhs(next_state, maps=geometry.maps, metric=geometry.metric, parameters=parameters)
            diagnostics = jnp.asarray(
                [
                    jnp.sqrt(jnp.mean(jnp.square(movie_field))),
                    jnp.mean(next_state.ion_density),
                    jnp.mean(next_state.neutral_density),
                    jnp.sqrt(jnp.mean(jnp.square(next_state.vorticity))),
                    result.potential_residual_l2,
                    jnp.min(next_state.ion_density),
                    jnp.min(next_state.neutral_density),
                ],
                dtype=jnp.float64,
            )
            return next_state, (movie_field, diagnostics)

        final_state, (movie_history, diagnostics) = jax.lax.scan(
            frame_step,
            initial_state,
            jnp.arange(int(frames), dtype=jnp.int32),
        )
        return final_state, movie_history, diagnostics

    return jax.jit(run)


def _build_essos_imported_drb_movie_report(
    *,
    geometry: EssosImportedFciGeometry,
    movie_history: np.ndarray,
    diagnostics: np.ndarray,
    final_state: dict[str, np.ndarray],
    final_sheath: dict[str, float],
    final_neutral: dict[str, float],
    frames: int,
    substeps_per_frame: int,
    dt: float,
    execute_seconds: float,
    potential_iterations: int,
    potential_regularization: float,
    potential_preconditioner: str | None,
) -> dict[str, Any]:
    final_fluctuation = movie_history[-1]
    spectrum = np.abs(np.fft.rfftn(final_fluctuation, axes=(1, 2))) ** 2
    total_power = float(np.sum(spectrum))
    low_mode_fraction = float(np.sum(spectrum[:, :4, :6]) / max(total_power, 1.0e-30))
    spectral_stats = _spectral_mode_statistics(spectrum)
    mode_power = np.mean(spectrum, axis=0)
    if mode_power.size:
        mode_power[0, 0] = 0.0
    peak_mode = np.unravel_index(int(np.argmax(mode_power)), mode_power.shape)
    endpoint_fraction = float(
        np.mean(np.asarray(geometry.maps.forward_boundary, dtype=bool) | np.asarray(geometry.maps.backward_boundary, dtype=bool))
    )
    map_source = str(geometry.metadata.get("map_source", "coil"))
    endpoint_gate = endpoint_fraction < 1.0e-12 if map_source == "vmec" else 0.05 < endpoint_fraction <= 1.0
    b_modulation_gate = 1.01 if map_source == "vmec" else 1.05
    if map_source == "coil":
        case = "essos_imported_qa_coil_drb_transient_movie"
        source = "ESSOS-imported Landreman-Paul QA coil FCI maps with JAXDRB fixed-layout DRB transient"
    elif map_source == "vmec":
        case = "essos_imported_qa_vmec_drb_transient_movie"
        source = "ESSOS-imported Landreman-Paul QA VMEC-coordinate FCI maps with JAXDRB fixed-layout DRB transient"
    else:
        case = "essos_imported_qa_hybrid_drb_transient_movie"
        source = "ESSOS-imported Landreman-Paul QA hybrid FCI maps with JAXDRB fixed-layout DRB transient"
    bmag = np.asarray(geometry.magnetic_field_magnitude, dtype=np.float64)
    finite = all(np.all(np.isfinite(value)) for value in [movie_history, diagnostics, *final_state.values()])
    min_density = float(min(np.min(final_state["ion_density"]), np.min(final_state["neutral_density"])))
    radial_flux = _radial_flux_proxy(movie_history, geometry)
    radial_flux_stats = _radial_flux_profile_statistics(radial_flux)
    report: dict[str, Any] = {
        "case": case,
        "source": source,
        "map_source": map_source,
        "claim_scope": (
            "movie-grade reduced DRB transient on a near-boundary VMEC-shaped physics grid; "
            "coil and hybrid map sources include open-field sheath/recycling endpoints, while "
            "the VMEC map source is a closed-field coordinate-map reference"
        ),
        **classify_essos_imported_drb_movie_evidence(map_source),
        "geometry": geometry.metadata,
        "movie_physics_grid": [int(value) for value in geometry.shape],
        "movie_render_coordinate_model": "raw_vmec_fourier_surface_registered_to_vmec_jax_plot",
        "frames": int(frames),
        "substeps_per_frame": int(substeps_per_frame),
        "dt": float(dt),
        "potential_solver": "fixed_iteration_metric_weighted_cg",
        "potential_iterations": int(potential_iterations),
        "potential_regularization": float(potential_regularization),
        "potential_preconditioner": potential_preconditioner,
        "execute_seconds": float(execute_seconds),
        "endpoint_fraction": endpoint_fraction,
        "magnetic_field_modulation": float(np.max(bmag) / max(float(np.min(bmag)), 1.0e-30)),
        "connection_length_mean": float(np.mean(np.asarray(geometry.connection_length, dtype=np.float64))),
        "final_min_density": min_density,
        "initial_fluctuation_rms": float(diagnostics[0, 0]),
        "final_fluctuation_rms": float(diagnostics[-1, 0]),
        "max_fluctuation_rms": float(np.max(diagnostics[:, 0])),
        "final_ion_density_mean": float(np.mean(final_state["ion_density"])),
        "final_neutral_density_mean": float(np.mean(final_state["neutral_density"])),
        "final_vorticity_rms": float(diagnostics[-1, 3]),
        "final_potential_residual_l2": float(diagnostics[-1, 4]),
        "radial_flux_proxy": radial_flux_stats["mean"],
        "radial_flux_abs_mean": radial_flux_stats["abs_mean"],
        "radial_flux_rms": radial_flux_stats["rms"],
        "radial_flux_peak_abs": radial_flux_stats["peak_abs"],
        "radial_flux_cancellation_ratio": radial_flux_stats["cancellation_ratio"],
        "radial_flux_positive_fraction": radial_flux_stats["positive_fraction"],
        "low_mode_spectral_power_fraction": low_mode_fraction,
        **spectral_stats,
        "dominant_poloidal_mode_index": int(peak_mode[1]),
        "dominant_toroidal_mode_index": int(peak_mode[0]),
        **final_sheath,
        **final_neutral,
    }
    report["passed"] = (
        finite
        and min_density > 0.0
        and endpoint_gate
        and report["magnetic_field_modulation"] > b_modulation_gate
        and report["final_fluctuation_rms"] > 1.0e-4
        and report["final_potential_residual_l2"] < 5.0
        and report["max_fluctuation_rms"] > report["initial_fluctuation_rms"] * 0.80
        and report["radial_flux_abs_mean"] > 1.0e-8
        and 0.0 < low_mode_fraction <= 1.0
        and report["particle_recycling_relative_error"] < 1.0e-10
        and report["current_balance_relative_error"] < 1.0e-10
        and report["neutral_particle_relative_error"] < 1.0e-10
        and report["neutral_momentum_relative_error"] < 1.0e-10
    )
    return report


def _build_essos_imported_drb_movie_arrays(
    *,
    geometry: EssosImportedFciGeometry,
    movie_history: np.ndarray,
    diagnostics: np.ndarray,
    final_state: dict[str, np.ndarray],
    frame_dt: float,
) -> dict[str, np.ndarray]:
    vmax = float(np.nanpercentile(np.abs(movie_history), 95.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    final_spectrum = np.mean(np.abs(np.fft.rfftn(movie_history[-1], axes=(1, 2))) ** 2, axis=0)
    final_spectrum[0, 0] = 0.0
    return {
        "density_fluctuation_history": movie_history.astype(np.float16),
        "diagnostics": diagnostics.astype(np.float32),
        "time": (np.arange(movie_history.shape[0], dtype=np.float64) * float(frame_dt)).astype(np.float64),
        "movie_vmax": np.asarray([vmax], dtype=np.float32),
        "x": np.asarray(geometry.coordinates_x, dtype=np.float32),
        "y": np.asarray(geometry.coordinates_y, dtype=np.float32),
        "z": np.asarray(geometry.coordinates_z, dtype=np.float32),
        "radial_coordinate": np.mean(np.asarray(geometry.minor_radius, dtype=np.float64), axis=(1, 2)).astype(np.float32),
        "magnetic_field_section": np.asarray(geometry.magnetic_field_magnitude[:, 0, :], dtype=np.float32),
        "endpoint_count_toroidal": (
            np.asarray(geometry.maps.forward_boundary, dtype=np.float64)
            + np.asarray(geometry.maps.backward_boundary, dtype=np.float64)
        ).sum(axis=0).astype(np.float32),
        "final_ion_density": final_state["ion_density"].astype(np.float32),
        "final_neutral_density": final_state["neutral_density"].astype(np.float32),
        "final_vorticity": final_state["vorticity"].astype(np.float32),
        "final_radial_flux_proxy": _radial_flux_proxy(movie_history, geometry).astype(np.float32),
        "final_spectrum_log10": np.log10(np.maximum(final_spectrum, 1.0e-18)).astype(np.float32),
    }


def _final_closure_diagnostics(
    geometry: EssosImportedFciGeometry,
    state: FciDrbState,
) -> tuple[dict[str, float], dict[str, float]]:
    ion_density = jnp.asarray(state.ion_density, dtype=jnp.float64)
    electron_density = jnp.asarray(state.electron_density, dtype=jnp.float64)
    ion_temperature = state.ion_pressure / jnp.maximum(ion_density, 1.0e-12)
    electron_temperature = state.electron_pressure / jnp.maximum(electron_density, 1.0e-12)
    sheath = compute_fci_sheath_recycling(
        ion_density,
        electron_temperature,
        ion_temperature,
        geometry.maps,
        recycling_fraction=0.965,
        recycled_neutral_energy=0.026,
    )
    neutral = compute_fci_neutral_reaction_diffusion(
        neutral_density=state.neutral_density,
        neutral_pressure=state.neutral_pressure,
        neutral_momentum=state.neutral_momentum,
        ion_density=state.ion_density,
        ion_pressure=state.ion_pressure,
        ion_momentum=state.ion_momentum,
        electron_density=state.electron_density,
        electron_pressure=state.electron_pressure,
        maps=geometry.maps,
        metric=geometry.metric,
    )
    sheath_report = {
        "total_particle_loss": float(sheath.total_ion_particle_loss),
        "total_target_heat_load": float(sheath.total_target_heat_load),
        "particle_recycling_relative_error": float(
            jnp.abs(sheath.particle_recycling_residual) / jnp.maximum(jnp.abs(sheath.total_recycled_particle_source), 1.0e-30)
        ),
        "current_balance_relative_error": float(
            jnp.abs(sheath.current_balance_residual) / jnp.maximum(jnp.abs(sheath.total_ion_particle_loss), 1.0e-30)
        ),
    }
    neutral_report = {
        "total_ionisation": float(jnp.sum(neutral.ionisation_rate)),
        "total_recombination": float(jnp.sum(neutral.recombination_rate)),
        "total_charge_exchange": float(jnp.sum(neutral.charge_exchange_rate)),
        "neutral_particle_relative_error": float(
            jnp.abs(neutral.total_particle_residual)
            / jnp.maximum(jnp.sum(jnp.abs(neutral.ion_density_source)), 1.0e-30)
        ),
        "neutral_momentum_relative_error": float(
            jnp.abs(neutral.total_momentum_residual)
            / jnp.maximum(jnp.sum(jnp.abs(neutral.ion_momentum_source)), 1.0e-30)
        ),
    }
    return sheath_report, neutral_report


def _save_essos_imported_drb_3d_frame_pyvista(
    geometry: EssosImportedFciGeometry,
    field: np.ndarray,
    time_value: float,
    path: str | Path,
    *,
    vmax: float | None,
) -> Path:
    import pyvista as pv

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    render = _build_movie_render_coordinates(geometry, raw_vmec_scale=True)
    x = render["x"]
    y = render["y"]
    z = render["z"]
    phi = render["phi"]
    theta = render["theta"]
    values = np.asarray(field, dtype=np.float64)
    value_limit = _movie_value_limit(values, vmax)
    scalar_name = "ion density fluctuation"
    nr, nphi, ntheta = x.shape
    phi_window = np.arange(0, max(3, nphi - max(1, nphi // 4)), dtype=int)
    theta_window = np.arange(0, ntheta, dtype=int)
    radial_window = np.arange(0, nr, dtype=int)
    outer_i = max(nr - 1, 0)
    middle_i = max(int(0.58 * (nr - 1)), 0)

    plotter = pv.Plotter(off_screen=True, window_size=(1280, 900))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")

    def add_surface(
        x_surface: np.ndarray,
        y_surface: np.ndarray,
        z_surface: np.ndarray,
        scalar_values: np.ndarray,
        *,
        opacity: float,
        show_scalar_bar: bool,
    ) -> None:
        mesh = pv.StructuredGrid(x_surface, y_surface, z_surface)
        mesh[scalar_name] = np.asarray(scalar_values, dtype=np.float64).ravel(order="F")
        plotter.add_mesh(
            mesh,
            scalars=scalar_name,
            cmap="coolwarm",
            clim=(-value_limit, value_limit),
            opacity=opacity,
            smooth_shading=True,
            show_edges=False,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={
                "title": scalar_name,
                "title_font_size": 18,
                "label_font_size": 14,
                "fmt": "%.2e",
                "shadow": False,
            },
        )

    for radial_index, opacity, show_bar in ((outer_i, 0.74, True), (middle_i, 0.46, False)):
        radial_fraction = float(radial_index) / max(float(nr - 1), 1.0)
        surface_phi = phi[np.ix_([radial_index], phi_window, theta_window)][0]
        surface_theta = theta[np.ix_([radial_index], phi_window, theta_window)][0]
        add_surface(
            x[np.ix_([radial_index], phi_window, theta_window)][0],
            y[np.ix_([radial_index], phi_window, theta_window)][0],
            z[np.ix_([radial_index], phi_window, theta_window)][0],
            _interpolate_movie_field_surface(
                values,
                radial_fraction=radial_fraction,
                phi=surface_phi,
                theta=surface_theta,
            ),
            opacity=opacity,
            show_scalar_bar=show_bar,
        )

    for cut_j in (max(1, nphi // 10), max(2, 3 * nphi // 5)):
        cut_phi = phi[np.ix_(radial_window, [cut_j], theta_window)][:, 0, :]
        cut_theta = theta[np.ix_(radial_window, [cut_j], theta_window)][:, 0, :]
        add_surface(
            x[np.ix_(radial_window, [cut_j], theta_window)][:, 0, :],
            y[np.ix_(radial_window, [cut_j], theta_window)][:, 0, :],
            z[np.ix_(radial_window, [cut_j], theta_window)][:, 0, :],
            _interpolate_movie_field_cut(
                values,
                radial_fractions=np.linspace(0.0, 1.0, nr)[:, None],
                phi=cut_phi,
                theta=cut_theta,
            ),
            opacity=0.95,
            show_scalar_bar=False,
        )

    _add_boundary_wire(plotter, x, y, z, radial_index=0, color="#333333", opacity=0.30)
    _add_boundary_wire(plotter, x, y, z, radial_index=outer_i, color="black", opacity=0.45)
    plotter.add_text(
        f"ESSOS-imported {_essos_imported_map_label(geometry)} DRB transient on Landreman-Paul QA VMEC surfaces\n"
        f"sheath + recycling + neutral closures, t = {time_value:.3f}",
        position=(32, 830),
        font_size=14,
        color="black",
    )
    plotter.add_text(
        "Fixed camera; opened toroidal sector exposes radial cuts and non-axisymmetric fluctuation structure",
        position="lower_left",
        font_size=11,
        color="black",
    )
    center = (float(np.nanmean(x)), float(np.nanmean(y)), float(np.nanmean(z)))
    radius = 1.55 * max(float(np.nanmax(x) - np.nanmin(x)), float(np.nanmax(y) - np.nanmin(y)))
    angle = np.deg2rad(-42.0)
    camera = (
        center[0] + radius * np.cos(angle),
        center[1] + radius * np.sin(angle),
        center[2] + 0.48 * radius,
    )
    plotter.camera_position = [camera, center, (0.0, 0.0, 1.0)]
    plotter.screenshot(str(resolved))
    plotter.close()
    return resolved


def _save_essos_imported_drb_3d_frame_matplotlib(
    geometry: EssosImportedFciGeometry,
    field: np.ndarray,
    time_value: float,
    path: str | Path,
    *,
    vmax: float | None,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(field, dtype=np.float64)
    value_limit = _movie_value_limit(values, vmax)
    norm = colors.TwoSlopeNorm(vmin=-value_limit, vcenter=0.0, vmax=value_limit)
    cmap = plt.get_cmap("coolwarm")
    render = _build_movie_render_coordinates(geometry)
    x = render["x"]
    y = render["y"]
    z = render["z"]
    phi = render["phi"]
    theta = render["theta"]
    nr, nphi, _ntheta = x.shape
    outer_i = nr - 1
    middle_i = max(int(0.54 * (nr - 1)), 0)
    phi_segments = ((0, int(0.54 * nphi)), (int(0.70 * nphi), nphi))
    cut_indices = (int(0.54 * nphi), int(0.70 * nphi))

    fig = plt.figure(figsize=(9.2, 7.4), constrained_layout=False)
    axis = fig.add_axes([0.00, 0.04, 0.84, 0.88], projection="3d")
    for start, stop in phi_segments:
        if stop - start < 3:
            continue
        segment = np.s_[start:stop, :]
        outer_values = _interpolate_movie_field_surface(
            values,
            radial_fraction=1.0,
            phi=phi[outer_i][segment],
            theta=theta[outer_i][segment],
        )
        axis.plot_surface(
            x[outer_i][segment],
            y[outer_i][segment],
            z[outer_i][segment],
            facecolors=cmap(norm(outer_values)),
            linewidth=0,
            antialiased=True,
            alpha=0.90,
            shade=False,
        )
        middle_values = _interpolate_movie_field_surface(
            values,
            radial_fraction=0.54,
            phi=phi[middle_i][segment],
            theta=theta[middle_i][segment],
        )
        axis.plot_surface(
            x[middle_i][segment],
            y[middle_i][segment],
            z[middle_i][segment],
            facecolors=cmap(norm(middle_values)),
            linewidth=0,
            antialiased=True,
            alpha=0.48,
            shade=False,
        )

    for cut_index in cut_indices:
        cut_index = int(np.clip(cut_index, 0, nphi - 1))
        cut_values = _interpolate_movie_field_cut(
            values,
            radial_fractions=np.linspace(0.0, 1.0, nr)[:, None],
            phi=phi[:, cut_index, :],
            theta=theta[:, cut_index, :],
        )
        axis.plot_surface(
            x[:, cut_index, :],
            y[:, cut_index, :],
            z[:, cut_index, :],
            facecolors=cmap(norm(cut_values)),
            linewidth=0,
            antialiased=True,
            alpha=0.98,
            shade=False,
        )

    _plot_movie_boundary_rings(axis, x, y, z, color="0.08", alpha=0.48)
    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar_axis = fig.add_axes([0.86, 0.18, 0.028, 0.62])
    fig.colorbar(scalar, cax=colorbar_axis, label="ion density fluctuation")
    fig.text(
        0.03,
        0.955,
        f"ESSOS-imported {_essos_imported_map_label(geometry)} DRB transient on Landreman-Paul QA VMEC surfaces",
        ha="left",
        va="top",
        fontsize=15,
    )
    fig.text(
        0.03,
        0.918,
        f"fixed camera, opened toroidal/radial sector, t = {time_value:.3f}",
        ha="left",
        va="top",
        fontsize=11,
    )
    fig.text(
        0.03,
        0.035,
        "The non-axisymmetric QA boundary is seeded from the VMEC Fourier surface; colors show ion-density fluctuations.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    axis.set_axis_off()
    axis.grid(False)
    axis.view_init(elev=21.0, azim=-49.0)
    extent = float(np.max(np.sqrt(x * x + y * y)))
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_zlim(float(np.min(z)) * 1.1, float(np.max(z)) * 1.1)
    try:
        axis.set_box_aspect((1.0, 1.0, 0.34), zoom=1.45)
    except TypeError:
        axis.set_box_aspect((1.0, 1.0, 0.34))
    fig.savefig(resolved, dpi=170, facecolor="white")
    plt.close(fig)
    return resolved


def _build_movie_render_coordinates_impl(
    geometry: EssosImportedFciGeometry,
    *,
    raw_vmec_scale: bool,
) -> dict[str, np.ndarray]:
    metadata = dict(geometry.metadata)
    if metadata.get("coordinate_model") == "scaled_vmec_fourier_flux_surfaces":
        try:
            wout_path = resolve_essos_landreman_qa_wout(essos_root=os.environ.get("JAX_DRB_ESSOS_ROOT"))
            if raw_vmec_scale:
                axis_major_radius = float(metadata.get("vmec_raw_axis_major_radius", metadata["axis_major_radius"]))
                axis_vertical = float(metadata.get("vmec_raw_axis_vertical", metadata["axis_vertical"]))
            else:
                axis_major_radius = float(metadata["axis_major_radius"])
                axis_vertical = float(metadata["axis_vertical"])
            return build_essos_vmec_scaled_qa_coordinates(
                wout_path,
                nx=max(int(geometry.shape[0]), 18),
                ny=max(4 * int(geometry.shape[1]), 96),
                nz=max(2 * int(geometry.shape[2]), 112),
                rho_min=float(metadata["rho_min"]),
                rho_max=float(metadata["rho_max"]),
                axis_major_radius=axis_major_radius,
                axis_vertical=axis_vertical,
            )
        except Exception:
            pass
    return {
        "x": np.asarray(geometry.coordinates_x, dtype=np.float64),
        "y": np.asarray(geometry.coordinates_y, dtype=np.float64),
        "z": np.asarray(geometry.coordinates_z, dtype=np.float64),
        "phi": np.asarray(geometry.toroidal_angle, dtype=np.float64),
        "theta": np.asarray(geometry.poloidal_angle, dtype=np.float64),
    }


def _build_movie_render_coordinates(geometry: EssosImportedFciGeometry, *, raw_vmec_scale: bool = False) -> dict[str, np.ndarray]:
    return _build_movie_render_coordinates_impl(geometry, raw_vmec_scale=raw_vmec_scale)


def _interpolate_movie_field_surface(
    values: np.ndarray,
    *,
    radial_fraction: float,
    phi: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    nx, ny, nz = values.shape
    radial = np.full_like(phi, float(radial_fraction) * float(nx - 1), dtype=np.float64)
    if radial_fraction >= 1.0:
        radial = np.full_like(phi, np.nextafter(float(nx - 1), 0.0), dtype=np.float64)
    coords = np.asarray(
        [
            radial,
            np.mod(phi, 2.0 * np.pi) / (2.0 * np.pi) * float(ny),
            np.mod(theta, 2.0 * np.pi) / (2.0 * np.pi) * float(nz),
        ],
        dtype=np.float64,
    )
    return map_coordinates(values, coords, order=1, mode="wrap")


def _interpolate_movie_field_cut(
    values: np.ndarray,
    *,
    radial_fractions: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    nx, ny, nz = values.shape
    radial = np.broadcast_to(radial_fractions * float(nx - 1), theta.shape)
    coords = np.asarray(
        [
            radial,
            np.mod(phi, 2.0 * np.pi) / (2.0 * np.pi) * float(ny),
            np.mod(theta, 2.0 * np.pi) / (2.0 * np.pi) * float(nz),
        ],
        dtype=np.float64,
    )
    return map_coordinates(values, coords, order=1, mode="wrap")


def _plot_movie_boundary_rings(
    axis: Any,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    color: str,
    alpha: float,
) -> None:
    for theta_index in np.linspace(0, x.shape[2] - 1, 7, dtype=int):
        axis.plot(x[-1, :, theta_index], y[-1, :, theta_index], z[-1, :, theta_index], color=color, alpha=alpha, lw=0.55)
    for phi_index in np.linspace(0, x.shape[1] - 1, 9, dtype=int):
        axis.plot(x[-1, phi_index, :], y[-1, phi_index, :], z[-1, phi_index, :], color=color, alpha=alpha, lw=0.55)


def _add_boundary_wire(plotter: Any, x: np.ndarray, y: np.ndarray, z: np.ndarray, *, radial_index: int, color: str, opacity: float) -> None:
    try:
        import pyvista as pv

        for theta_index in np.linspace(0, x.shape[2] - 1, min(5, x.shape[2]), dtype=int):
            points = np.column_stack([x[radial_index, :, theta_index], y[radial_index, :, theta_index], z[radial_index, :, theta_index]])
            points = np.vstack([points, points[:1]])
            line = pv.PolyData(points)
            line.lines = np.hstack([[points.shape[0]], np.arange(points.shape[0])])
            plotter.add_mesh(line, color=color, line_width=1.4, opacity=opacity)
    except Exception:
        return


def _state_to_numpy(state: FciDrbState) -> dict[str, np.ndarray]:
    return {
        "ion_density": np.asarray(state.ion_density, dtype=np.float64),
        "electron_density": np.asarray(state.electron_density, dtype=np.float64),
        "neutral_density": np.asarray(state.neutral_density, dtype=np.float64),
        "ion_pressure": np.asarray(state.ion_pressure, dtype=np.float64),
        "electron_pressure": np.asarray(state.electron_pressure, dtype=np.float64),
        "neutral_pressure": np.asarray(state.neutral_pressure, dtype=np.float64),
        "ion_momentum": np.asarray(state.ion_momentum, dtype=np.float64),
        "neutral_momentum": np.asarray(state.neutral_momentum, dtype=np.float64),
        "vorticity": np.asarray(state.vorticity, dtype=np.float64),
    }


def _density_fluctuation(ion_density: jax.Array) -> jax.Array:
    mean = jnp.mean(ion_density, axis=1, keepdims=True)
    return (ion_density - mean) / jnp.maximum(mean, 1.0e-12)


def _seed_movie_multimode_fluctuations(state: FciDrbState, geometry: EssosImportedFciGeometry) -> FciDrbState:
    radial = _normalized_minor_radius_jax(geometry)
    theta = geometry.poloidal_angle
    phi = geometry.toroidal_angle
    envelope = jnp.exp(-jnp.square((radial - 0.58) / 0.34))
    modes = (
        jnp.sin(3.0 * theta - 2.0 * phi)
        + 0.55 * jnp.cos(5.0 * theta + 3.0 * phi)
        + 0.35 * jnp.sin(8.0 * theta - 5.0 * phi)
        + 0.20 * jnp.cos(13.0 * theta + 4.0 * phi)
    )
    modes = modes / jnp.maximum(jnp.std(modes), 1.0e-12)
    perturbation = 0.038 * envelope * modes
    ion_density = jnp.maximum(state.ion_density * (1.0 + perturbation), 1.0e-6)
    electron_density = jnp.maximum(state.electron_density * (1.0 + 0.85 * perturbation), 1.0e-6)
    neutral_density = jnp.maximum(state.neutral_density * (1.0 + 0.35 * perturbation), 1.0e-8)
    return FciDrbState(
        ion_density=ion_density,
        electron_density=electron_density,
        neutral_density=neutral_density,
        ion_pressure=jnp.maximum(state.ion_pressure * (1.0 + 0.75 * perturbation), 1.0e-7 * ion_density),
        electron_pressure=jnp.maximum(state.electron_pressure * (1.0 + 0.65 * perturbation), 1.0e-7 * electron_density),
        neutral_pressure=jnp.maximum(state.neutral_pressure * (1.0 + 0.25 * perturbation), 1.0e-8 * neutral_density),
        ion_momentum=state.ion_momentum + 0.012 * ion_density * envelope * modes,
        neutral_momentum=state.neutral_momentum + 0.004 * neutral_density * envelope * modes,
        vorticity=state.vorticity + 0.018 * envelope * modes,
    )


def _logical_exb_advection(potential: jax.Array, field: jax.Array, geometry: EssosImportedFciGeometry) -> jax.Array:
    bracket = logical_exb_bracket_xz(potential, field, geometry.metric)
    scale = jnp.maximum(jnp.mean(jnp.abs(bracket)), 1.0e-8)
    return bracket / scale


def _radial_derivative(field: jax.Array, geometry: EssosImportedFciGeometry) -> jax.Array:
    values = jnp.asarray(field, dtype=jnp.float64)
    spacing = jnp.asarray(geometry.metric.dx, dtype=jnp.float64)
    centered = (jnp.roll(values, -1, axis=0) - jnp.roll(values, 1, axis=0)) / jnp.maximum(2.0 * spacing, 1.0e-30)
    first = (values[1, :, :] - values[0, :, :]) / jnp.maximum(spacing[0, :, :], 1.0e-30)
    last = (values[-1, :, :] - values[-2, :, :]) / jnp.maximum(spacing[-1, :, :], 1.0e-30)
    return centered.at[0, :, :].set(first).at[-1, :, :].set(last)


def _normalized_minor_radius_jax(geometry: EssosImportedFciGeometry) -> jax.Array:
    rho = jnp.asarray(geometry.minor_radius, dtype=jnp.float64)
    return (rho - jnp.min(rho)) / jnp.maximum(jnp.max(rho) - jnp.min(rho), 1.0e-12)


def _magnetic_curvature_proxy_jax(geometry: EssosImportedFciGeometry) -> jax.Array:
    bmag = jnp.asarray(geometry.magnetic_field_magnitude, dtype=jnp.float64)
    proxy = (bmag - jnp.mean(bmag, axis=1, keepdims=True)) / jnp.maximum(jnp.mean(bmag), 1.0e-12)
    return proxy / jnp.maximum(jnp.std(proxy), 1.0e-12)


def _clip_movie_state(state: FciDrbState) -> FciDrbState:
    ion_density = jnp.maximum(state.ion_density, 1.0e-6)
    electron_density = jnp.maximum(state.electron_density, 1.0e-6)
    neutral_density = jnp.maximum(state.neutral_density, 1.0e-8)
    return FciDrbState(
        ion_density=ion_density,
        electron_density=electron_density,
        neutral_density=neutral_density,
        ion_pressure=jnp.maximum(state.ion_pressure, 1.0e-7 * ion_density),
        electron_pressure=jnp.maximum(state.electron_pressure, 1.0e-7 * electron_density),
        neutral_pressure=jnp.maximum(state.neutral_pressure, 1.0e-8 * neutral_density),
        ion_momentum=jnp.clip(state.ion_momentum, -2.0, 2.0),
        neutral_momentum=jnp.clip(state.neutral_momentum, -1.0, 1.0),
        vorticity=jnp.clip(state.vorticity, -2.0, 2.0),
    )


def _add_scaled_state(state: FciDrbState, rhs: FciDrbState, scale: float) -> FciDrbState:
    return jax.tree_util.tree_map(lambda value, increment: value + float(scale) * increment, state, rhs)


def _radial_flux_proxy(movie_history: np.ndarray, geometry: EssosImportedFciGeometry) -> np.ndarray:
    final = np.asarray(movie_history[-1], dtype=np.float64)
    potential_proxy = np.roll(final, 2, axis=2)
    dz = np.asarray(geometry.metric.dz, dtype=np.float64)
    radial_velocity = -(np.roll(potential_proxy, -1, axis=2) - np.roll(potential_proxy, 1, axis=2)) / np.maximum(2.0 * dz, 1.0e-30)
    return np.mean(final * radial_velocity, axis=(1, 2))


def _radial_flux_profile_statistics(profile: np.ndarray) -> dict[str, float]:
    values = np.asarray(profile, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": 0.0,
            "abs_mean": 0.0,
            "rms": 0.0,
            "peak_abs": 0.0,
            "cancellation_ratio": 0.0,
            "positive_fraction": 0.0,
        }
    abs_values = np.abs(finite)
    mean = float(np.mean(finite))
    abs_mean = float(np.mean(abs_values))
    return {
        "mean": mean,
        "abs_mean": abs_mean,
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "peak_abs": float(np.max(abs_values)),
        "cancellation_ratio": float(abs(mean) / max(abs_mean, 1.0e-30)),
        "positive_fraction": float(np.mean(finite > 0.0)),
    }


def _spectral_mode_statistics(spectrum: np.ndarray) -> dict[str, Any]:
    mode_power = np.mean(np.asarray(spectrum, dtype=np.float64), axis=0)
    if mode_power.ndim != 2 or mode_power.size == 0:
        return {
            "spectral_poloidal_mode_count": 0,
            "spectral_toroidal_mode_count": 0,
            "spectral_centroid_poloidal_index": 0.0,
            "spectral_centroid_toroidal_index": 0.0,
            "spectral_edge_band_power_fraction": 0.0,
            "low_mode_window_covers_grid": False,
        }
    fluctuation_power = np.array(mode_power, copy=True)
    fluctuation_power[0, 0] = 0.0
    total = float(np.sum(fluctuation_power))
    poloidal_count, toroidal_count = fluctuation_power.shape
    poloidal_index, toroidal_index = np.indices(fluctuation_power.shape)
    if total <= 1.0e-30:
        poloidal_centroid = 0.0
        toroidal_centroid = 0.0
        edge_fraction = 0.0
    else:
        poloidal_centroid = float(np.sum(poloidal_index * fluctuation_power) / total)
        toroidal_centroid = float(np.sum(toroidal_index * fluctuation_power) / total)
        poloidal_edge_start = max(1, int(np.floor(0.75 * max(poloidal_count - 1, 1))))
        toroidal_edge_start = max(1, int(np.floor(0.75 * max(toroidal_count - 1, 1))))
        edge_mask = (poloidal_index >= poloidal_edge_start) | (
            toroidal_index >= toroidal_edge_start
        )
        edge_fraction = float(np.sum(fluctuation_power[edge_mask]) / total)
    return {
        "spectral_poloidal_mode_count": int(poloidal_count),
        "spectral_toroidal_mode_count": int(toroidal_count),
        "spectral_centroid_poloidal_index": poloidal_centroid,
        "spectral_centroid_toroidal_index": toroidal_centroid,
        "spectral_centroid_poloidal_fraction": float(
            poloidal_centroid / max(float(poloidal_count - 1), 1.0)
        ),
        "spectral_centroid_toroidal_fraction": float(
            toroidal_centroid / max(float(toroidal_count - 1), 1.0)
        ),
        "spectral_edge_band_power_fraction": edge_fraction,
        "low_mode_window_covers_grid": bool(poloidal_count <= 4 and toroidal_count <= 6),
    }


def _major_radius_and_vertical(geometry: EssosImportedFciGeometry) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(geometry.coordinates_x, dtype=np.float64)
    y = np.asarray(geometry.coordinates_y, dtype=np.float64)
    z = np.asarray(geometry.coordinates_z, dtype=np.float64)
    return np.sqrt(x * x + y * y), z


def _essos_imported_map_label(geometry: EssosImportedFciGeometry) -> str:
    map_source = str(geometry.metadata.get("map_source", "coil"))
    labels = {
        "coil": "QA-coil",
        "vmec": "QA VMEC-coordinate",
        "hybrid": "QA hybrid",
    }
    return labels.get(map_source, f"QA {map_source}")


def _movie_value_limit(values: np.ndarray, vmax: float | None) -> float:
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(values), 99.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        return 1.0
    return float(vmax)


def _audit_movie_gif(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return {"movie_audit_passed": False, "movie_audit_reason": "missing_gif"}
    frames: list[Image.Image] = []
    try:
        image = Image.open(resolved)
        frame_index = 0
        while True:
            frames.append(image.convert("RGB"))
            frame_index += 1
            image.seek(frame_index)
    except EOFError:
        pass
    except Exception as exc:
        return {"movie_audit_passed": False, "movie_audit_reason": type(exc).__name__}
    if len(frames) < 2:
        return {"movie_audit_passed": False, "movie_frame_count": len(frames), "movie_audit_reason": "too_few_frames"}
    from PIL import ImageChops, ImageStat

    white = Image.new("RGB", frames[0].size, "white")
    bboxes = [ImageChops.difference(frame, white).getbbox() for frame in frames]
    rms_values = []
    for left, right in zip(frames[:-1], frames[1:], strict=True):
        stat = ImageStat.Stat(ImageChops.difference(left, right))
        rms_values.append(float(np.sqrt(np.mean(np.square(stat.rms)))))
    unique_bbox_count = len({str(value) for value in bboxes})
    rms_array = np.asarray(rms_values, dtype=np.float64)
    return {
        "movie_audit_passed": bool(unique_bbox_count <= 3 and float(np.max(rms_array)) < 12.0),
        "movie_frame_count": len(frames),
        "movie_frame_size": [int(frames[0].size[0]), int(frames[0].size[1])],
        "movie_file_size_bytes": int(resolved.stat().st_size),
        "movie_bbox_unique_count": int(unique_bbox_count),
        "movie_frame_rms_min": float(np.min(rms_array)),
        "movie_frame_rms_median": float(np.median(rms_array)),
        "movie_frame_rms_max": float(np.max(rms_array)),
    }


def _block_until_ready(value: object) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
