from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .essos_imported_fci_campaign import (
    _IMPORTED_FCI_DIAGNOSTIC_SCHEMA,
    _IMPORTED_FCI_REQUIRED_REPORT_FIELDS,
)


_IMPORTED_DRB_MOVIE_REQUIRED_REPORT_FIELDS = (
    "case",
    "source",
    "map_source",
    "claim_scope",
    "publication_ready",
    "movie_evidence_role",
    "movie_promotion_rejection_reasons",
    "required_publication_gates",
    "geometry",
    "movie_physics_grid",
    "movie_render_coordinate_model",
    "frames",
    "substeps_per_frame",
    "dt",
    "execute_seconds",
    "endpoint_fraction",
    "magnetic_field_modulation",
    "connection_length_mean",
    "final_min_density",
    "initial_fluctuation_rms",
    "final_fluctuation_rms",
    "max_fluctuation_rms",
    "final_ion_density_mean",
    "final_neutral_density_mean",
    "final_vorticity_rms",
    "final_potential_residual_l2",
    "radial_flux_proxy",
    "radial_flux_abs_mean",
    "radial_flux_rms",
    "radial_flux_peak_abs",
    "radial_flux_cancellation_ratio",
    "radial_flux_positive_fraction",
    "low_mode_spectral_power_fraction",
    "spectral_poloidal_mode_count",
    "spectral_toroidal_mode_count",
    "spectral_centroid_poloidal_index",
    "spectral_centroid_toroidal_index",
    "spectral_centroid_poloidal_fraction",
    "spectral_centroid_toroidal_fraction",
    "spectral_edge_band_power_fraction",
    "low_mode_window_covers_grid",
    "dominant_poloidal_mode_index",
    "dominant_toroidal_mode_index",
    "total_target_heat_load",
    "total_particle_loss",
    "total_ionisation",
    "total_recombination",
    "total_charge_exchange",
    "particle_recycling_relative_error",
    "current_balance_relative_error",
    "neutral_particle_relative_error",
    "neutral_momentum_relative_error",
    "movie_audit_passed",
    "movie_frame_count",
    "movie_frame_size",
    "movie_file_size_bytes",
    "movie_bbox_unique_count",
    "movie_frame_rms_min",
    "movie_frame_rms_median",
    "movie_frame_rms_max",
    "passed",
)


def audit_essos_imported_artifact_report(
    report_json_path: str | Path,
    *,
    artifact_kind: str = "auto",
) -> dict[str, Any]:
    """Audit an imported-field validation report against the current schema.

    The audit is intentionally lightweight: it reads a committed JSON report and
    checks whether the report still contains the fields produced by the current
    validation code. It does not rerun ESSOS, VMEC, or DRBX simulations.
    """

    path = Path(report_json_path)
    base_report: dict[str, Any] = {
        "report_json_path": str(path),
        "artifact_kind": str(artifact_kind),
        "exists": path.exists(),
        "loaded": False,
        "schema_passed": False,
        "report_passed": None,
        "missing_report_fields": [],
        "missing_diagnostic_fields": {},
        "stale": True,
    }
    if not path.exists():
        base_report["reason"] = "missing_report_json"
        return base_report
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        base_report["reason"] = f"invalid_json:{exc.msg}"
        return base_report
    if not isinstance(report, Mapping):
        base_report["reason"] = "report_json_is_not_an_object"
        return base_report

    resolved_kind = _resolve_essos_imported_artifact_kind(report, artifact_kind)
    required_fields = _required_report_fields_for_kind(resolved_kind)
    diagnostic_schema = _diagnostic_schema_for_kind(resolved_kind)
    missing_fields = tuple(field for field in required_fields if field not in report)
    missing_diagnostics = _missing_nested_diagnostic_fields(report, diagnostic_schema)
    schema_passed = not missing_fields and not missing_diagnostics

    return {
        **base_report,
        "artifact_kind": resolved_kind,
        "loaded": True,
        "schema_passed": bool(schema_passed),
        "report_passed": report.get("passed"),
        "missing_report_fields": list(missing_fields),
        "missing_diagnostic_fields": {
            key: list(value) for key, value in missing_diagnostics.items()
        },
        "stale": not bool(schema_passed),
    }


def audit_essos_imported_artifact_reports(
    report_json_paths: Sequence[str | Path],
    *,
    artifact_kind: str = "auto",
) -> dict[str, Any]:
    """Audit a sequence of imported-field report JSON files."""

    reports = tuple(
        audit_essos_imported_artifact_report(path, artifact_kind=artifact_kind)
        for path in report_json_paths
    )
    stale_reports = tuple(item for item in reports if item["stale"])
    return {
        "report_count": len(reports),
        "stale_report_count": len(stale_reports),
        "schema_passed": not stale_reports,
        "reports": list(reports),
    }


def audit_hybrid_open_sol_promotion_evidence(
    *,
    fci_report_json_path: str | Path,
    stationarity_report_json_path: str | Path,
    refinement_summary_json_path: str | Path,
    media_manifest_json_path: str | Path,
) -> dict[str, Any]:
    """Audit the release-backed hybrid open-SOL promotion evidence bundle.

    The hybrid QA lane is promoted only when several independently generated
    reports agree: imported FCI/source accounting, long-window stationarity,
    grid/time refinement, and visual-QA/media provenance. This audit is a
    lightweight manifest-level gate; it intentionally does not rerun ESSOS,
    VMEC, DRBX transients, or media rendering.
    """

    paths = {
        "fci_source_profile": Path(fci_report_json_path),
        "stationarity": Path(stationarity_report_json_path),
        "grid_time_refinement": Path(refinement_summary_json_path),
        "media_manifest": Path(media_manifest_json_path),
    }
    loaded = {name: _load_json_mapping(path) for name, path in paths.items()}
    stage_reports = (
        _audit_hybrid_fci_stage(loaded["fci_source_profile"]),
        _audit_hybrid_stationarity_stage(loaded["stationarity"]),
        _audit_hybrid_refinement_stage(loaded["grid_time_refinement"]),
        _audit_hybrid_media_stage(loaded["media_manifest"]),
    )
    promotion_rejection_reasons = sorted(
        {
            reason
            for stage in stage_reports
            if not stage["passed"]
            for reason in stage["reasons"]
        }
    )
    promotion_ready = not promotion_rejection_reasons

    return {
        "diagnostic": "essos_hybrid_open_sol_promotion_evidence_audit",
        "claim_scope": (
            "report-only audit of hybrid VMEC/coil open-SOL FCI, source, "
            "stationarity, grid/time-refinement, media, and visual-QA evidence"
        ),
        "map_source": "hybrid",
        "evidence_paths": {name: str(path) for name, path in paths.items()},
        "stage_reports": list(stage_reports),
        "promotion_ready": bool(promotion_ready),
        "promotion_rejection_reasons": promotion_rejection_reasons,
    }


def _resolve_essos_imported_artifact_kind(
    report: Mapping[str, Any],
    artifact_kind: str,
) -> str:
    normalized = str(artifact_kind or "auto").strip().lower().replace("-", "_")
    if normalized in {"fci", "imported_fci"}:
        return "fci"
    if normalized in {"movie", "drb_movie", "imported_drb_movie"}:
        return "movie"
    if normalized != "auto":
        raise ValueError(f"Unsupported imported artifact kind {artifact_kind!r}.")
    if "movie_frame_count" in report or "movie_render_coordinate_model" in report:
        return "movie"
    return "fci"


def _required_report_fields_for_kind(kind: str) -> tuple[str, ...]:
    if kind == "fci":
        return tuple(_IMPORTED_FCI_REQUIRED_REPORT_FIELDS)
    if kind == "movie":
        return tuple(_IMPORTED_DRB_MOVIE_REQUIRED_REPORT_FIELDS)
    raise ValueError(f"Unsupported imported artifact kind {kind!r}.")


def _diagnostic_schema_for_kind(kind: str) -> Mapping[str, Sequence[str]]:
    if kind == "fci":
        return _IMPORTED_FCI_DIAGNOSTIC_SCHEMA
    if kind == "movie":
        return {}
    raise ValueError(f"Unsupported imported artifact kind {kind!r}.")


def _load_json_mapping(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "loaded": False,
        "data": None,
        "reasons": [],
    }
    if not path.exists():
        result["reasons"] = ["missing_json"]
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["reasons"] = [f"invalid_json:{exc.msg}"]
        return result
    if not isinstance(payload, Mapping):
        result["reasons"] = ["json_is_not_an_object"]
        return result
    result["loaded"] = True
    result["data"] = payload
    return result


def _audit_hybrid_fci_stage(loaded: Mapping[str, Any]) -> dict[str, Any]:
    report = _loaded_report_or_none(loaded)
    reasons = _loading_reasons("fci", loaded)
    if report is not None:
        if report.get("map_source") != "hybrid":
            reasons.append("fci_map_source_not_hybrid")
        if report.get("passed") is not True:
            reasons.append("fci_report_not_passed")
        if report.get("connection_length_resolution_passed") is not True:
            reasons.append("fci_connection_length_resolution_not_passed")
        if report.get("map_diagnostics_passed") is not True:
            reasons.append("fci_map_diagnostics_not_passed")
        if not _finite_greater_than(report.get("target_fraction"), 0.0):
            reasons.append("fci_target_fraction_not_positive")
        for key in (
            "particle_recycling_relative_error",
            "neutral_particle_relative_error",
            "current_balance_relative_error",
            "neutral_momentum_relative_error",
        ):
            if not _finite_abs_at_most(report.get(key), 1.0e-10):
                reasons.append(f"fci_{key}_above_tolerance")
    return _hybrid_stage_report(
        "hybrid_fci_source_profile",
        loaded,
        reasons,
        summary={
            "target_fraction": None if report is None else report.get("target_fraction"),
            "magnetic_field_modulation": None
            if report is None
            else report.get("magnetic_field_modulation"),
        },
    )


def _audit_hybrid_stationarity_stage(loaded: Mapping[str, Any]) -> dict[str, Any]:
    report = _loaded_report_or_none(loaded)
    reasons = _loading_reasons("stationarity", loaded)
    if report is not None:
        if report.get("map_source") != "hybrid":
            reasons.append("stationarity_map_source_not_hybrid")
        if report.get("publication_ready") is not True:
            reasons.append("stationarity_publication_ready_not_true")
        if report.get("stationarity_passed") is not True:
            reasons.append("stationarity_not_passed")
        if report.get("underlying_movie_passed") is not True:
            reasons.append("stationarity_underlying_movie_not_passed")
        if not _integer_at_least(report.get("frames"), 12):
            reasons.append("stationarity_too_few_frames")
        if report.get("potential_preconditioner") != "jacobi":
            reasons.append("stationarity_potential_preconditioner_not_jacobi")
        if not _finite_abs_below(report.get("potential_tail_max"), 1.0e-8):
            reasons.append("stationarity_potential_tail_too_large")
        if not _finite_greater_than(report.get("min_density_tail"), 0.0):
            reasons.append("stationarity_min_density_tail_not_positive")
    return _hybrid_stage_report(
        "hybrid_stationarity",
        loaded,
        reasons,
        summary={
            "frames": None if report is None else report.get("frames"),
            "potential_tail_max": None
            if report is None
            else report.get("potential_tail_max"),
        },
    )


def _audit_hybrid_refinement_stage(loaded: Mapping[str, Any]) -> dict[str, Any]:
    report = _loaded_report_or_none(loaded)
    reasons = _loading_reasons("refinement", loaded)
    if report is not None:
        if report.get("publication_ready") is not True:
            reasons.append("refinement_publication_ready_not_true")
        if report.get("grid_refinement_passed") is not True:
            reasons.append("grid_refinement_not_passed")
        if report.get("time_refinement_passed") is not True:
            reasons.append("time_refinement_not_passed")
        for key in ("grid_refinement_diagnostics", "time_refinement_diagnostics"):
            diagnostics = report.get(key)
            if not isinstance(diagnostics, Mapping):
                reasons.append(f"{key}_missing")
                continue
            prefix = "grid" if key.startswith("grid") else "time"
            if diagnostics.get("passed") is not True:
                reasons.append(f"{prefix}_refinement_diagnostics_not_passed")
            if diagnostics.get("all_reports_passed") is not True:
                reasons.append(f"{prefix}_refinement_reports_not_all_passed")
            if diagnostics.get("map_source") != "hybrid":
                reasons.append(f"{prefix}_refinement_map_source_not_hybrid")
            if diagnostics.get("map_source_consistent") is not True:
                reasons.append(f"{prefix}_refinement_map_source_not_consistent")
            if diagnostics.get("spectral_resolution_passed") is not True:
                reasons.append(f"{prefix}_refinement_spectral_resolution_not_passed")
            if not _finite_abs_at_most(
                diagnostics.get("max_relative_metric_change"),
                report.get("relative_tolerance", 0.3),
            ):
                reasons.append(f"{prefix}_refinement_metric_change_above_tolerance")
    return _hybrid_stage_report(
        "hybrid_grid_time_refinement",
        loaded,
        reasons,
        summary={
            "grid_max_relative_metric_change": _nested_get(
                report,
                "grid_refinement_diagnostics",
                "max_relative_metric_change",
            ),
            "time_max_relative_metric_change": _nested_get(
                report,
                "time_refinement_diagnostics",
                "max_relative_metric_change",
            ),
        },
    )


def _audit_hybrid_media_stage(loaded: Mapping[str, Any]) -> dict[str, Any]:
    report = _loaded_report_or_none(loaded)
    reasons = _loading_reasons("media", loaded)
    if report is not None:
        qa = report.get("qa")
        files = report.get("files")
        release_assets = report.get("release_assets")
        file_paths = (
            [str(item.get("path", "")) for item in files if isinstance(item, Mapping)]
            if isinstance(files, Sequence) and not isinstance(files, (str, bytes))
            else []
        )
        if report.get("map_source") != "hybrid":
            reasons.append("media_map_source_not_hybrid")
        if not isinstance(qa, Mapping):
            reasons.append("media_qa_missing")
        else:
            if qa.get("visual_qa") != "passed_local_frame_contact_sheet":
                reasons.append("media_visual_qa_not_passed")
            if qa.get("camera_stability") != "passed":
                reasons.append("media_camera_stability_not_passed")
            if qa.get("non_axisymmetric_geometry_visible") is not True:
                reasons.append("media_non_axisymmetric_geometry_not_visible")
            if qa.get("opened_radial_toroidal_sector_visible") is not True:
                reasons.append("media_opened_sector_not_visible")
        if not isinstance(release_assets, Sequence) or isinstance(
            release_assets,
            (str, bytes),
        ):
            reasons.append("media_release_assets_missing")
        else:
            urls = [str(url) for url in release_assets]
            if not urls:
                reasons.append("media_release_assets_empty")
            if any(
                not url.startswith(
                    "https://github.com/uwplasma/drbx/releases/download/"
                )
                for url in urls
            ):
                reasons.append("media_release_asset_url_not_project_release")
        required_file_fragments = {
            "gif": ".gif",
            "diagnostics": "diagnostics",
            "poster": "poster",
            "snapshots": "snapshots",
            "contact_sheet": "contact_sheet",
        }
        for label, fragment in required_file_fragments.items():
            if not any(fragment in path for path in file_paths):
                reasons.append(f"media_{label}_file_missing")
    return _hybrid_stage_report(
        "hybrid_media_manifest",
        loaded,
        reasons,
        summary={
            "file_count": None
            if report is None or not isinstance(report.get("files"), Sequence)
            else len(report.get("files", ())),
            "release_asset_count": None
            if report is None or not isinstance(report.get("release_assets"), Sequence)
            else len(report.get("release_assets", ())),
        },
    )


def _hybrid_stage_report(
    stage: str,
    loaded: Mapping[str, Any],
    reasons: Sequence[str],
    *,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "path": loaded["path"],
        "exists": bool(loaded["exists"]),
        "loaded": bool(loaded["loaded"]),
        "passed": not reasons,
        "reasons": list(reasons),
        "summary": dict(summary or {}),
    }


def _loaded_report_or_none(loaded: Mapping[str, Any]) -> Mapping[str, Any] | None:
    data = loaded.get("data")
    return data if isinstance(data, Mapping) else None


def _loading_reasons(prefix: str, loaded: Mapping[str, Any]) -> list[str]:
    return [f"{prefix}_{reason}" for reason in loaded.get("reasons", ())]


def _finite_abs_at_most(value: Any, limit: float) -> bool:
    return (
        _finite_number(value)
        and _finite_number(limit)
        and abs(float(value)) <= float(limit)
    )


def _finite_abs_below(value: Any, limit: float) -> bool:
    return (
        _finite_number(value)
        and _finite_number(limit)
        and abs(float(value)) < float(limit)
    )


def _finite_greater_than(value: Any, threshold: float) -> bool:
    return (
        _finite_number(value)
        and _finite_number(threshold)
        and float(value) > float(threshold)
    )


def _integer_at_least(value: Any, threshold: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= threshold


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nested_get(report: Mapping[str, Any] | None, parent: str, child: str) -> Any:
    if report is None:
        return None
    parent_value = report.get(parent)
    if not isinstance(parent_value, Mapping):
        return None
    return parent_value.get(child)


def _missing_nested_diagnostic_fields(
    report: Mapping[str, Any],
    diagnostic_schema: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    missing: dict[str, tuple[str, ...]] = {}
    for parent_key, child_keys in diagnostic_schema.items():
        parent = report.get(parent_key)
        if not isinstance(parent, Mapping):
            missing[parent_key] = tuple(child_keys)
            continue
        child_missing = tuple(child for child in child_keys if child not in parent)
        if child_missing:
            missing[parent_key] = child_missing
    return missing
