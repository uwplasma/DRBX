from __future__ import annotations

import json
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
    validation code. It does not rerun ESSOS, VMEC, or JAXDRB simulations.
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
