from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import (
    audit_hybrid_open_sol_promotion_evidence,
    build_essos_imported_fci_source_profile_gate,
    create_essos_imported_drb_movie_package,
    create_essos_imported_drb_movie_refinement_campaign_package,
    create_essos_imported_drb_movie_stationarity_package,
    create_essos_imported_fci_campaign_package,
    create_live_essos_imported_connection_length_refinement_package,
)
from jax_drb.validation.essos_imported_fci_campaign import (
    create_essos_imported_fci_dry_run_artifact_package,
)


# SIMSOPT-style user parameters: edit these values, then run this file.
RUN_EXAMPLE = True
REPO_ROOT = Path(__file__).resolve().parents[3]

# The default writes a self-contained promotion ledger. Set the live flags only
# when an ESSOS checkout plus Landreman-Paul QA coil and VMEC inputs are
# available. Hybrid maps use smooth VMEC coordinates with coil-derived endpoint
# masks and |B| modulation, so they are the planned open-SOL bridge when pure
# direct-coil maps remain too rough for promotion.
WRITE_DRY_RUN_CONTRACT = True
RUN_LIVE_FCI_GATE = False
RUN_LIVE_CONNECTION_REFINEMENT_GATE = False
RUN_LIVE_STATIONARITY_GATE = False
RUN_LIVE_MOVIE_REFINEMENT_GATE = False
RUN_LIVE_MEDIA_GATE = False
RUN_RELEASE_EVIDENCE_AUDIT = True
REQUIRE_PROMOTION_READY = False

OUTPUT_ROOT = Path("artifacts/essos_hybrid_open_sol")
CASE_LABEL = "essos_hybrid_open_sol"
COIL_JSON_PATH: Path | None = None
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
PRECISION = "float64"

MAP_SOURCE = "hybrid"
CONNECTION_QUANTITY = "parallel_step_per_toroidal_radian"

# Release-backed high-resolution hybrid evidence. These files are lightweight
# JSON manifests/reports committed with the docs; large media live in releases.
RELEASE_FCI_REPORT_JSON_PATH = (
    REPO_ROOT
    / "docs/data/essos_imported_fci_hybrid_artifacts/data/essos_imported_fci_hybrid_campaign.json"
)
RELEASE_STATIONARITY_REPORT_JSON_PATH = (
    REPO_ROOT
    / (
        "docs/data/essos_imported_drb_movie_stationarity_jacobi_artifacts/data/"
        "essos_imported_drb_movie_stationarity_jacobi.json"
    )
)
RELEASE_REFINEMENT_SUMMARY_JSON_PATH = (
    REPO_ROOT
    / (
        "docs/data/essos_imported_drb_movie_refinement_poloidal_96_jacobi_artifacts/data/"
        "essos_imported_drb_movie_refinement_poloidal_96_jacobi_summary.json"
    )
)
RELEASE_MEDIA_MANIFEST_JSON_PATH = (
    REPO_ROOT / "docs/data/essos_imported_drb_movie_stationarity_jacobi_media_manifest.json"
)

# FCI/open-endpoint validation gate.
FCI_NX = 5
FCI_NY = 8
FCI_NZ = 20
FCI_RHO_MIN = 0.12
FCI_RHO_MAX = 0.34
FCI_MAXTIME = 80.0
FCI_TIMES_TO_TRACE = 360
FCI_TRACE_TOLERANCE = 1.0e-8
REQUIRE_CONNECTION_RESOLUTION = True

# Live hybrid nested-grid refinement.
REFINEMENT_LEVEL_SHAPES = (
    (3, 4, 6),
    (6, 8, 12),
    (12, 16, 24),
)
REFINEMENT_MAXTIME = 40.0
REFINEMENT_TIMES_TO_TRACE = 160
REFINEMENT_CONVERGENCE_THRESHOLD = 0.10
REFINEMENT_LINF_THRESHOLD = 0.20
REFINEMENT_MINIMUM_OBSERVED_ORDER = 0.50
REFINEMENT_REQUIRE_OBSERVED_ORDER = True

# Report-only reduced transient gate.
STATIONARITY_PRESET = "quick"  # "quick" for workflow QA, "promotion" for media claims.
MOVIE_NX = 16
MOVIE_NY = 96
MOVIE_NZ = 48
MOVIE_RHO_MIN = 0.20
MOVIE_RHO_MAX = 0.60
MOVIE_MAXTIME = 24.0
MOVIE_TIMES_TO_TRACE = 80
MOVIE_FRAMES = 12
MOVIE_SUBSTEPS_PER_FRAME = 3
MOVIE_DT = 2.0e-3
MOVIE_POTENTIAL_ITERATIONS = 3072
MOVIE_POTENTIAL_REGULARIZATION = 5.0
MOVIE_POTENTIAL_PRECONDITIONER = "jacobi"
MOVIE_TAIL_FRACTION = 0.50
MOVIE_RELATIVE_TOLERANCE = 0.35
MOVIE_MIN_FRAMES = 12

# Fast report-only stationarity smoke gate. Passing this preset proves the
# workflow is wired correctly; it is not enough for README or publication media.
QUICK_STATIONARITY_NX = 4
QUICK_STATIONARITY_NY = 12
QUICK_STATIONARITY_NZ = 24
QUICK_STATIONARITY_RHO_MIN = 0.20
QUICK_STATIONARITY_RHO_MAX = 0.60
QUICK_STATIONARITY_MAXTIME = 18.0
QUICK_STATIONARITY_TIMES_TO_TRACE = 48
QUICK_STATIONARITY_FRAMES = 4
QUICK_STATIONARITY_SUBSTEPS_PER_FRAME = 1
QUICK_STATIONARITY_DT = 1.5e-3
QUICK_STATIONARITY_POTENTIAL_ITERATIONS = 384
QUICK_STATIONARITY_POTENTIAL_REGULARIZATION = 5.0
QUICK_STATIONARITY_POTENTIAL_PRECONDITIONER = "jacobi"
QUICK_STATIONARITY_TAIL_FRACTION = 0.50
QUICK_STATIONARITY_RELATIVE_TOLERANCE = 0.75
QUICK_STATIONARITY_MIN_FRAMES = 4

# Report-only grid/time refinement around the current promoted compact shape.
MOVIE_REFINEMENT_GRID_SHAPES = (
    (8, 12, 24),
    (16, 24, 48),
)
MOVIE_REFINEMENT_TIME_SHAPE = (16, 24, 48)
MOVIE_REFINEMENT_TIME_DT_VALUES = (2.0e-3, 1.0e-3)
MOVIE_REFINEMENT_FRAMES = 4
MOVIE_REFINEMENT_SUBSTEPS_PER_FRAME = 2
MOVIE_REFINEMENT_GRID_DT = 2.0e-3

# Optional GIF/PNG media stage.
MEDIA_NX = MOVIE_NX
MEDIA_NY = MOVIE_NY
MEDIA_NZ = MOVIE_NZ
MEDIA_RHO_MIN = MOVIE_RHO_MIN
MEDIA_RHO_MAX = MOVIE_RHO_MAX
MEDIA_MAXTIME = MOVIE_MAXTIME
MEDIA_TIMES_TO_TRACE = MOVIE_TIMES_TO_TRACE
MEDIA_FRAMES = MOVIE_FRAMES
MEDIA_SUBSTEPS_PER_FRAME = MOVIE_SUBSTEPS_PER_FRAME
MEDIA_DT = MOVIE_DT
MEDIA_POTENTIAL_ITERATIONS = MOVIE_POTENTIAL_ITERATIONS
MEDIA_POTENTIAL_REGULARIZATION = MOVIE_POTENTIAL_REGULARIZATION
MEDIA_POTENTIAL_PRECONDITIONER = MOVIE_POTENTIAL_PRECONDITIONER


@dataclass(frozen=True)
class HybridOpenSolSettings:
    output_root: Path
    case_label: str
    coil_json_path: Path | None
    vmec_wout_path: Path | None
    essos_root: Path | None
    precision: str
    write_dry_run_contract: bool
    run_live_fci_gate: bool
    run_live_connection_refinement_gate: bool
    run_live_stationarity_gate: bool
    run_live_movie_refinement_gate: bool
    run_live_media_gate: bool
    run_release_evidence_audit: bool
    release_fci_report_json_path: Path
    release_stationarity_report_json_path: Path
    release_refinement_summary_json_path: Path
    release_media_manifest_json_path: Path
    require_promotion_ready: bool
    stationarity_preset: str


def build_settings(
    *,
    output_root: Path = OUTPUT_ROOT,
    case_label: str = CASE_LABEL,
    coil_json_path: Path | None = COIL_JSON_PATH,
    vmec_wout_path: Path | None = VMEC_WOUT_PATH,
    essos_root: Path | None = ESSOS_ROOT,
    precision: str = PRECISION,
    write_dry_run_contract: bool = WRITE_DRY_RUN_CONTRACT,
    run_live_fci_gate: bool = RUN_LIVE_FCI_GATE,
    run_live_connection_refinement_gate: bool = RUN_LIVE_CONNECTION_REFINEMENT_GATE,
    run_live_stationarity_gate: bool = RUN_LIVE_STATIONARITY_GATE,
    run_live_movie_refinement_gate: bool = RUN_LIVE_MOVIE_REFINEMENT_GATE,
    run_live_media_gate: bool = RUN_LIVE_MEDIA_GATE,
    run_release_evidence_audit: bool = RUN_RELEASE_EVIDENCE_AUDIT,
    release_fci_report_json_path: Path = RELEASE_FCI_REPORT_JSON_PATH,
    release_stationarity_report_json_path: Path = RELEASE_STATIONARITY_REPORT_JSON_PATH,
    release_refinement_summary_json_path: Path = RELEASE_REFINEMENT_SUMMARY_JSON_PATH,
    release_media_manifest_json_path: Path = RELEASE_MEDIA_MANIFEST_JSON_PATH,
    require_promotion_ready: bool = REQUIRE_PROMOTION_READY,
    stationarity_preset: str = STATIONARITY_PRESET,
) -> HybridOpenSolSettings:
    """Resolve top-level hybrid open-SOL workflow settings."""

    normalized_stationarity_preset = str(stationarity_preset).strip().lower()
    if normalized_stationarity_preset not in {"quick", "promotion"}:
        raise ValueError(
            "stationarity_preset must be 'quick' or 'promotion'; got "
            f"{stationarity_preset!r}."
        )
    return HybridOpenSolSettings(
        output_root=Path(output_root),
        case_label=str(case_label),
        coil_json_path=None if coil_json_path is None else Path(coil_json_path),
        vmec_wout_path=None if vmec_wout_path is None else Path(vmec_wout_path),
        essos_root=None if essos_root is None else Path(essos_root),
        precision=str(precision),
        write_dry_run_contract=bool(write_dry_run_contract),
        run_live_fci_gate=bool(run_live_fci_gate),
        run_live_connection_refinement_gate=bool(run_live_connection_refinement_gate),
        run_live_stationarity_gate=bool(run_live_stationarity_gate),
        run_live_movie_refinement_gate=bool(run_live_movie_refinement_gate),
        run_live_media_gate=bool(run_live_media_gate),
        run_release_evidence_audit=bool(run_release_evidence_audit),
        release_fci_report_json_path=Path(release_fci_report_json_path),
        release_stationarity_report_json_path=Path(release_stationarity_report_json_path),
        release_refinement_summary_json_path=Path(release_refinement_summary_json_path),
        release_media_manifest_json_path=Path(release_media_manifest_json_path),
        require_promotion_ready=bool(require_promotion_ready),
        stationarity_preset=normalized_stationarity_preset,
    )


def _path_text(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_blocker(report: dict[str, Any]) -> dict[str, Any]:
    """Return a compact blocker record for a non-promoted workflow stage."""

    reasons = list(report.get("promotion_rejection_reasons", []))
    reasons.extend(report.get("movie_promotion_rejection_reasons", []))
    if not reasons:
        status = str(report.get("status", "unknown"))
        if status in {"skipped", "contract_only", "diagnostic"}:
            reasons.append(f"{status}_stage_not_live_promotion_evidence")
        else:
            reasons.append("stage_not_promotion_ready")
    return {
        "stage": report.get("stage"),
        "status": report.get("status"),
        "reasons": reasons,
        "next_action": report.get("next_action"),
    }


def write_workflow_summary(
    settings: HybridOpenSolSettings,
    stage_reports: list[dict[str, Any]],
) -> Path:
    """Write a compact promotion ledger for the hybrid open-SOL lane."""

    data_dir = settings.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_path = data_dir / f"{settings.case_label}_workflow_summary.json"
    non_live_statuses = {"skipped", "contract_only", "diagnostic", "release_evidence"}
    live_stage_reports = [
        report
        for report in stage_reports
        if report["status"] not in non_live_statuses
    ]
    promotion_stage_reports = [
        report
        for report in stage_reports
        if report["status"] != "diagnostic"
    ]
    promotion_ready = bool(promotion_stage_reports) and all(
        bool(report.get("promotion_ready", False)) for report in promotion_stage_reports
    )
    blocking_stage_records = [
        _stage_blocker(report)
        for report in stage_reports
        if report["status"] != "diagnostic"
        and not bool(report.get("promotion_ready", False))
    ]
    diagnostic_stage_records = [
        _stage_blocker(report)
        for report in stage_reports
        if report["status"] == "diagnostic"
        and not bool(report.get("promotion_ready", False))
    ]
    release_evidence_stage_records = [
        report
        for report in stage_reports
        if report["status"] == "release_evidence"
    ]
    promotion_rejection_reasons = [
        reason
        for record in blocking_stage_records
        for reason in record["reasons"]
    ]
    if not live_stage_reports:
        promotion_rejection_reasons.insert(0, "no_live_promotion_gates_ran")
    next_actions = [
        record["next_action"]
        for record in [*blocking_stage_records, *diagnostic_stage_records]
        if record.get("next_action")
    ]
    payload = {
        "diagnostic": "essos_hybrid_open_sol_workflow",
        "map_source": MAP_SOURCE,
        "connection_quantity": CONNECTION_QUANTITY,
        "claim_boundary": (
            "Hybrid VMEC/coil open-SOL workflow. VMEC supplies smooth map "
            "coordinates while coil traces supply endpoint masks and |B| "
            "modulation. Promotion requires live FCI/source/profile, "
            "connection-length, stationarity, grid/time, and visual-QA gates."
        ),
        "settings": {
            "output_root": _path_text(settings.output_root),
            "case_label": settings.case_label,
            "coil_json_path": _path_text(settings.coil_json_path),
            "vmec_wout_path": _path_text(settings.vmec_wout_path),
            "essos_root": _path_text(settings.essos_root),
            "precision": settings.precision,
            "write_dry_run_contract": settings.write_dry_run_contract,
            "run_live_fci_gate": settings.run_live_fci_gate,
            "run_live_connection_refinement_gate": settings.run_live_connection_refinement_gate,
            "run_live_stationarity_gate": settings.run_live_stationarity_gate,
            "run_live_movie_refinement_gate": settings.run_live_movie_refinement_gate,
            "run_live_media_gate": settings.run_live_media_gate,
            "run_release_evidence_audit": settings.run_release_evidence_audit,
            "release_fci_report_json_path": _path_text(settings.release_fci_report_json_path),
            "release_stationarity_report_json_path": _path_text(
                settings.release_stationarity_report_json_path
            ),
            "release_refinement_summary_json_path": _path_text(
                settings.release_refinement_summary_json_path
            ),
            "release_media_manifest_json_path": _path_text(
                settings.release_media_manifest_json_path
            ),
            "require_promotion_ready": settings.require_promotion_ready,
            "stationarity_preset": settings.stationarity_preset,
        },
        "stage_reports": stage_reports,
        "release_evidence_ready": any(
            bool(report.get("promotion_ready", False))
            for report in release_evidence_stage_records
        ),
        "release_evidence_stage_count": len(release_evidence_stage_records),
        "promotion_ready": promotion_ready,
        "promotion_rejection_reasons": sorted(set(promotion_rejection_reasons)),
        "promotion_blocking_stages": blocking_stage_records,
        "diagnostic_stages": diagnostic_stage_records,
        "next_actions": next_actions,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote hybrid open-SOL workflow summary: {summary_path}")
    if settings.require_promotion_ready and not promotion_ready:
        raise RuntimeError(
            "Hybrid open-SOL workflow is not promotion-ready: "
            f"{payload['promotion_rejection_reasons']}"
        )
    return summary_path


def run_release_evidence_audit(settings: HybridOpenSolSettings) -> dict[str, Any] | None:
    """Audit the committed release-backed high-resolution hybrid evidence."""

    if not settings.run_release_evidence_audit:
        return None

    audit = audit_hybrid_open_sol_promotion_evidence(
        fci_report_json_path=settings.release_fci_report_json_path,
        stationarity_report_json_path=settings.release_stationarity_report_json_path,
        refinement_summary_json_path=settings.release_refinement_summary_json_path,
        media_manifest_json_path=settings.release_media_manifest_json_path,
    )
    audit_path = settings.output_root / "data" / f"{settings.case_label}_release_evidence_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote hybrid release-evidence audit: {audit_path}")
    return {
        "stage": "hybrid_release_evidence_audit",
        "status": "release_evidence",
        "promotion_ready": bool(audit.get("promotion_ready", False)),
        "report_json_path": _path_text(audit_path),
        "audited_evidence_paths": audit.get("evidence_paths", {}),
        "audited_stage_reports": audit.get("stage_reports", []),
        "promotion_rejection_reasons": audit.get("promotion_rejection_reasons", []),
        "next_action": (
            "Regenerate or repair the high-resolution hybrid FCI/source, "
            "stationarity, grid/time-refinement, or media-manifest evidence."
        ),
    }


def run_fci_gate(settings: HybridOpenSolSettings) -> dict[str, Any]:
    """Run or describe the hybrid FCI endpoint/source validation gate."""

    if settings.write_dry_run_contract:
        dry_run = create_essos_imported_fci_dry_run_artifact_package(
            output_root=settings.output_root / "fci",
            case_label=f"{settings.case_label}_fci",
            coil_json_path=settings.coil_json_path,
            vmec_wout_path=settings.vmec_wout_path,
            essos_root=settings.essos_root,
            map_source=MAP_SOURCE,
            nx=FCI_NX,
            ny=FCI_NY,
            nz=FCI_NZ,
            rho_min=FCI_RHO_MIN,
            rho_max=FCI_RHO_MAX,
            maxtime=FCI_MAXTIME,
            times_to_trace=FCI_TIMES_TO_TRACE,
            trace_tolerance=FCI_TRACE_TOLERANCE,
            precision=settings.precision,
            require_connection_resolution=REQUIRE_CONNECTION_RESOLUTION,
        )
        print(f"wrote hybrid FCI dry-run contract: {dry_run.contract_json_path}")

    if not settings.run_live_fci_gate:
        return {
            "stage": "hybrid_fci_endpoint_source_gate",
            "status": "contract_only" if settings.write_dry_run_contract else "skipped",
            "promotion_ready": False,
            "next_action": "Set RUN_LIVE_FCI_GATE=True after ESSOS coil and VMEC geometry are available.",
        }

    configure_jax_runtime(precision=settings.precision)
    artifacts = create_essos_imported_fci_campaign_package(
        output_root=settings.output_root / "fci",
        case_label=f"{settings.case_label}_fci",
        coil_json_path=settings.coil_json_path,
        vmec_wout_path=settings.vmec_wout_path,
        essos_root=settings.essos_root,
        map_source=MAP_SOURCE,
        nx=FCI_NX,
        ny=FCI_NY,
        nz=FCI_NZ,
        rho_min=FCI_RHO_MIN,
        rho_max=FCI_RHO_MAX,
        maxtime=FCI_MAXTIME,
        times_to_trace=FCI_TIMES_TO_TRACE,
        trace_tolerance=FCI_TRACE_TOLERANCE,
        require_connection_resolution=REQUIRE_CONNECTION_RESOLUTION,
    )
    report = _read_json(artifacts.report_json_path)
    return {
        "stage": "hybrid_fci_endpoint_source_gate",
        "status": "ran",
        "report_json_path": _path_text(artifacts.report_json_path),
        "arrays_npz_path": _path_text(artifacts.arrays_npz_path),
        "plot_png_path": _path_text(artifacts.plot_png_path),
        "passed": bool(report.get("passed", False)),
        "promotion_ready": bool(
            report.get("passed", False)
            and report.get("connection_length_resolution_passed", False)
            and report.get("map_diagnostics_passed", False)
        ),
    }


def run_source_profile_gate(
    settings: HybridOpenSolSettings,
    fci_stage: dict[str, Any],
) -> dict[str, Any]:
    """Validate target/source/profile artifacts produced by the FCI gate."""

    if fci_stage.get("status") != "ran":
        return {
            "stage": "hybrid_source_profile_gate",
            "status": "contract_only" if settings.write_dry_run_contract else "skipped",
            "promotion_ready": False,
            "next_action": (
                "Set RUN_LIVE_FCI_GATE=True to generate target-label, heat-load, "
                "neutral-source, and radial-profile artifacts for the hybrid map."
            ),
        }

    report_path = Path(str(fci_stage["report_json_path"]))
    arrays_path = Path(str(fci_stage["arrays_npz_path"]))
    report = _read_json(report_path)
    with np.load(arrays_path) as arrays:
        gate = build_essos_imported_fci_source_profile_gate(report, arrays)

    gate_path = settings.output_root / "data" / f"{settings.case_label}_source_profile_gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "stage": "hybrid_source_profile_gate",
        "status": "ran",
        "report_json_path": _path_text(gate_path),
        "source_report_json_path": _path_text(report_path),
        "source_arrays_npz_path": _path_text(arrays_path),
        "source_plot_png_path": fci_stage.get("plot_png_path"),
        "passed": bool(gate.get("passed", False)),
        "promotion_ready": bool(gate.get("promotion_ready", False)),
        "evidence_role": gate.get("evidence_role"),
        "promotion_rejection_reasons": gate.get("promotion_rejection_reasons", []),
    }


def run_connection_refinement_gate(settings: HybridOpenSolSettings) -> dict[str, Any]:
    """Run or describe hybrid parallel-step refinement."""

    if not settings.run_live_connection_refinement_gate:
        return {
            "stage": "hybrid_parallel_step_refinement_gate",
            "status": "skipped",
            "promotion_ready": False,
            "next_action": (
                "Set RUN_LIVE_CONNECTION_REFINEMENT_GATE=True to test hybrid "
                "parallel-step-per-radian refinement."
            ),
        }

    artifacts = create_live_essos_imported_connection_length_refinement_package(
        output_root=settings.output_root / "connection_length",
        case_label=f"{settings.case_label}_parallel_step",
        coil_json_path=settings.coil_json_path,
        vmec_wout_path=settings.vmec_wout_path,
        essos_root=settings.essos_root,
        map_source=MAP_SOURCE,
        connection_quantity=CONNECTION_QUANTITY,
        level_shapes=REFINEMENT_LEVEL_SHAPES,
        rho_min=FCI_RHO_MIN,
        rho_max=FCI_RHO_MAX,
        maxtime=REFINEMENT_MAXTIME,
        times_to_trace=REFINEMENT_TIMES_TO_TRACE,
        trace_tolerance=FCI_TRACE_TOLERANCE,
        convergence_threshold=REFINEMENT_CONVERGENCE_THRESHOLD,
        linf_threshold=REFINEMENT_LINF_THRESHOLD,
        minimum_observed_order=REFINEMENT_MINIMUM_OBSERVED_ORDER,
        require_observed_order=REFINEMENT_REQUIRE_OBSERVED_ORDER,
    )
    report = _read_json(artifacts.report_json_path)
    return {
        "stage": "hybrid_parallel_step_refinement_gate",
        "status": "ran",
        "connection_quantity": CONNECTION_QUANTITY,
        "report_json_path": _path_text(artifacts.report_json_path),
        "arrays_npz_path": _path_text(artifacts.arrays_npz_path),
        "plot_png_path": _path_text(artifacts.plot_png_path),
        "passed": bool(report.get("passed", False)),
        "promotion_ready": bool(report.get("promotion_ready", report.get("passed", False))),
        "evidence_role": report.get("evidence_role"),
        "promotion_rejection_reasons": report.get("promotion_rejection_reasons", []),
    }


def _stationarity_parameter_bundle(settings: HybridOpenSolSettings) -> tuple[dict[str, Any], bool]:
    """Return stationarity parameters and whether they can promote media."""

    if settings.stationarity_preset == "promotion":
        return (
            {
                "nx": MOVIE_NX,
                "ny": MOVIE_NY,
                "nz": MOVIE_NZ,
                "rho_min": MOVIE_RHO_MIN,
                "rho_max": MOVIE_RHO_MAX,
                "maxtime": MOVIE_MAXTIME,
                "times_to_trace": MOVIE_TIMES_TO_TRACE,
                "frames": MOVIE_FRAMES,
                "substeps_per_frame": MOVIE_SUBSTEPS_PER_FRAME,
                "dt": MOVIE_DT,
                "potential_iterations": MOVIE_POTENTIAL_ITERATIONS,
                "potential_regularization": MOVIE_POTENTIAL_REGULARIZATION,
                "potential_preconditioner": MOVIE_POTENTIAL_PRECONDITIONER,
                "tail_fraction": MOVIE_TAIL_FRACTION,
                "relative_tolerance": MOVIE_RELATIVE_TOLERANCE,
                "min_frames": MOVIE_MIN_FRAMES,
            },
            True,
        )
    return (
        {
            "nx": QUICK_STATIONARITY_NX,
            "ny": QUICK_STATIONARITY_NY,
            "nz": QUICK_STATIONARITY_NZ,
            "rho_min": QUICK_STATIONARITY_RHO_MIN,
            "rho_max": QUICK_STATIONARITY_RHO_MAX,
            "maxtime": QUICK_STATIONARITY_MAXTIME,
            "times_to_trace": QUICK_STATIONARITY_TIMES_TO_TRACE,
            "frames": QUICK_STATIONARITY_FRAMES,
            "substeps_per_frame": QUICK_STATIONARITY_SUBSTEPS_PER_FRAME,
            "dt": QUICK_STATIONARITY_DT,
            "potential_iterations": QUICK_STATIONARITY_POTENTIAL_ITERATIONS,
            "potential_regularization": QUICK_STATIONARITY_POTENTIAL_REGULARIZATION,
            "potential_preconditioner": QUICK_STATIONARITY_POTENTIAL_PRECONDITIONER,
            "tail_fraction": QUICK_STATIONARITY_TAIL_FRACTION,
            "relative_tolerance": QUICK_STATIONARITY_RELATIVE_TOLERANCE,
            "min_frames": QUICK_STATIONARITY_MIN_FRAMES,
        },
        False,
    )


def run_stationarity_gate(settings: HybridOpenSolSettings) -> dict[str, Any]:
    """Run or describe the report-only hybrid reduced-transient gate."""

    if not settings.run_live_stationarity_gate:
        return {
            "stage": "hybrid_reduced_transient_stationarity_gate",
            "status": "skipped",
            "promotion_ready": False,
            "next_action": (
                "Set RUN_LIVE_STATIONARITY_GATE=True after FCI, source/profile, "
                "and connection-refinement gates pass for the same hybrid map."
            ),
        }

    configure_jax_runtime(precision=settings.precision)
    stationarity_parameters, promotion_capable = _stationarity_parameter_bundle(settings)
    artifacts = create_essos_imported_drb_movie_stationarity_package(
        output_root=settings.output_root / "stationarity",
        case_label=f"{settings.case_label}_stationarity",
        coil_json_path=settings.coil_json_path,
        vmec_wout_path=settings.vmec_wout_path,
        essos_root=settings.essos_root,
        map_source=MAP_SOURCE,
        **stationarity_parameters,
    )
    report = _read_json(artifacts.report_json_path)
    rejection_reasons = list(report.get("movie_promotion_rejection_reasons", []))
    if bool(report.get("publication_ready", False)) and not promotion_capable:
        rejection_reasons.append("quick_stationarity_preset_not_promotion_evidence")
    return {
        "stage": "hybrid_reduced_transient_stationarity_gate",
        "status": "ran",
        "stationarity_preset": settings.stationarity_preset,
        "report_json_path": _path_text(artifacts.report_json_path),
        "stationarity_passed": bool(report.get("stationarity_passed", False)),
        "promotion_ready": bool(report.get("publication_ready", False) and promotion_capable),
        "movie_promotion_rejection_reasons": rejection_reasons,
    }


def run_movie_refinement_gate(settings: HybridOpenSolSettings) -> dict[str, Any]:
    """Run or describe the report-only hybrid grid/time refinement gate."""

    if not settings.run_live_movie_refinement_gate:
        return {
            "stage": "hybrid_movie_grid_time_refinement_gate",
            "status": "skipped",
            "promotion_ready": False,
            "next_action": (
                "Set RUN_LIVE_MOVIE_REFINEMENT_GATE=True to run grid/time "
                "refinement before promoting hybrid media."
            ),
        }

    configure_jax_runtime(precision=settings.precision)
    artifacts = create_essos_imported_drb_movie_refinement_campaign_package(
        output_root=settings.output_root / "movie_refinement",
        case_label=f"{settings.case_label}_movie_refinement",
        coil_json_path=settings.coil_json_path,
        vmec_wout_path=settings.vmec_wout_path,
        essos_root=settings.essos_root,
        map_source=MAP_SOURCE,
        grid_shapes=MOVIE_REFINEMENT_GRID_SHAPES,
        time_shape=MOVIE_REFINEMENT_TIME_SHAPE,
        time_dt_values=MOVIE_REFINEMENT_TIME_DT_VALUES,
        rho_min=MOVIE_RHO_MIN,
        rho_max=MOVIE_RHO_MAX,
        maxtime=MOVIE_MAXTIME,
        times_to_trace=MOVIE_TIMES_TO_TRACE,
        frames=MOVIE_REFINEMENT_FRAMES,
        substeps_per_frame=MOVIE_REFINEMENT_SUBSTEPS_PER_FRAME,
        grid_dt=MOVIE_REFINEMENT_GRID_DT,
        relative_tolerance=MOVIE_RELATIVE_TOLERANCE,
        potential_iterations=MOVIE_POTENTIAL_ITERATIONS,
        potential_regularization=MOVIE_POTENTIAL_REGULARIZATION,
        potential_preconditioner=MOVIE_POTENTIAL_PRECONDITIONER,
        reuse_existing_reports=True,
    )
    report = _read_json(artifacts.report_json_path)
    return {
        "stage": "hybrid_movie_grid_time_refinement_gate",
        "status": "ran",
        "report_json_path": _path_text(artifacts.report_json_path),
        "grid_report_json_paths": [path.as_posix() for path in artifacts.grid_report_json_paths],
        "time_report_json_paths": [path.as_posix() for path in artifacts.time_report_json_paths],
        "grid_refinement_passed": bool(report.get("grid_refinement_passed", False)),
        "time_refinement_passed": bool(report.get("time_refinement_passed", False)),
        "promotion_ready": bool(report.get("publication_ready", False)),
        "movie_promotion_rejection_reasons": report.get("movie_promotion_rejection_reasons", []),
    }


def run_media_gate(settings: HybridOpenSolSettings) -> dict[str, Any]:
    """Run or describe the optional hybrid GIF/PNG media stage."""

    if not settings.run_live_media_gate:
        return {
            "stage": "hybrid_diagnostic_turbulence_media",
            "status": "skipped",
            "promotion_ready": False,
            "next_action": (
                "Set RUN_LIVE_MEDIA_GATE=True only after the FCI, source/profile, "
                "connection, stationarity, grid/time, and visual-QA gates are reviewed."
            ),
        }

    configure_jax_runtime(precision=settings.precision)
    artifacts = create_essos_imported_drb_movie_package(
        output_root=settings.output_root / "media",
        case_label=f"{settings.case_label}_diagnostic_media",
        coil_json_path=settings.coil_json_path,
        vmec_wout_path=settings.vmec_wout_path,
        essos_root=settings.essos_root,
        map_source=MAP_SOURCE,
        nx=MEDIA_NX,
        ny=MEDIA_NY,
        nz=MEDIA_NZ,
        rho_min=MEDIA_RHO_MIN,
        rho_max=MEDIA_RHO_MAX,
        maxtime=MEDIA_MAXTIME,
        times_to_trace=MEDIA_TIMES_TO_TRACE,
        frames=MEDIA_FRAMES,
        substeps_per_frame=MEDIA_SUBSTEPS_PER_FRAME,
        dt=MEDIA_DT,
        potential_iterations=MEDIA_POTENTIAL_ITERATIONS,
        potential_regularization=MEDIA_POTENTIAL_REGULARIZATION,
        potential_preconditioner=MEDIA_POTENTIAL_PRECONDITIONER,
    )
    report = _read_json(artifacts.report_json_path)
    return {
        "stage": "hybrid_diagnostic_turbulence_media",
        "status": "ran",
        "report_json_path": _path_text(artifacts.report_json_path),
        "arrays_npz_path": _path_text(artifacts.arrays_npz_path),
        "snapshot_png_path": _path_text(artifacts.snapshot_png_path),
        "diagnostics_png_path": _path_text(artifacts.diagnostics_png_path),
        "poster_png_path": _path_text(artifacts.poster_png_path),
        "movie_gif_path": _path_text(artifacts.movie_gif_path),
        "passed": bool(report.get("passed", False)),
        "movie_visual_qa_passed": bool(report.get("movie_visual_qa_passed", False)),
        "promotion_ready": bool(report.get("publication_ready", False)),
        "movie_evidence_role": report.get("movie_evidence_role"),
        "movie_promotion_rejection_reasons": report.get("movie_promotion_rejection_reasons", []),
    }


def run_hybrid_workflow(settings: HybridOpenSolSettings) -> Path:
    """Run the configured hybrid open-SOL workflow stages."""

    fci_stage = run_fci_gate(settings)
    stage_reports = [
        fci_stage,
        run_source_profile_gate(settings, fci_stage),
        run_connection_refinement_gate(settings),
        run_stationarity_gate(settings),
        run_movie_refinement_gate(settings),
        run_media_gate(settings),
    ]
    release_evidence_stage = run_release_evidence_audit(settings)
    if release_evidence_stage is not None:
        stage_reports.append(release_evidence_stage)
    return write_workflow_summary(settings, stage_reports)


if RUN_EXAMPLE:
    HYBRID_SETTINGS = build_settings()
    HYBRID_WORKFLOW_SUMMARY = run_hybrid_workflow(HYBRID_SETTINGS)
