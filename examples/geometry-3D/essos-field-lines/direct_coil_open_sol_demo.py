from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import (
    build_essos_imported_fci_source_profile_gate,
    create_essos_imported_drb_movie_package,
    create_essos_imported_drb_movie_stationarity_package,
    create_essos_imported_fci_campaign_package,
    create_live_essos_imported_connection_length_refinement_package,
    create_live_essos_imported_endpoint_label_refinement_package,
    save_essos_imported_fci_source_profile_gate_plot,
)
from jax_drb.validation.essos_imported_fci_campaign import (
    create_essos_imported_fci_dry_run_artifact_package,
)


# SIMSOPT-style user parameters: edit these values, then run this file.
RUN_EXAMPLE = True

# The default is a self-contained workflow contract. Set the live flags below
# only when an ESSOS checkout and the Landreman-Paul QA coil JSON are available.
WRITE_DRY_RUN_CONTRACT = True
RUN_LIVE_FCI_GATE = False
RUN_LIVE_CONNECTION_REFINEMENT_GATE = False
RUN_LIVE_ENDPOINT_LABEL_REFINEMENT_GATE = False
RUN_LIVE_COLLOCATED_ENDPOINT_LABEL_REFINEMENT_GATE = False
RUN_LIVE_STATIONARITY_GATE = False
RUN_LIVE_MEDIA_GATE = False
REQUIRE_PROMOTION_READY = False

OUTPUT_ROOT = Path("artifacts/essos_direct_coil_open_sol")
CASE_LABEL = "essos_direct_coil_open_sol"
COIL_JSON_PATH: Path | None = None
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None
PRECISION = "float64"

# Direct-coil open-field semantics. Do not change this to "hybrid" or "vmec"
# in this script; those are separate control/bridge examples in the plan.
MAP_SOURCE = "coil"
REFINEMENT_QUANTITIES = ("adjacent_step_length",)
DIAGNOSTIC_REFINEMENT_QUANTITIES = ("target_exit_length",)

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

# Live pure-coil nested-grid refinement. This is the promotion blocker for the
# direct-coil lane; weak observed order keeps the result as diagnostic evidence.
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
ENDPOINT_LABEL_MINIMUM_AGREEMENT_FRACTION = 0.90
ENDPOINT_LABEL_MINIMUM_ENDPOINT_AGREEMENT_FRACTION = 0.80
ENDPOINT_LABEL_MINIMUM_ENDPOINT_UNION_FRACTION = 0.01
COLLOCATED_ENDPOINT_LABEL_LEVEL_SHAPES = (
    (3, 5, 9),
    (7, 15, 27),
)

# Report-only reduced transient gate. This is not the final GIF-producing
# campaign; it checks whether the direct-coil settings are stable enough to
# justify a later media run.
MOVIE_NX = 8
MOVIE_NY = 28
MOVIE_NZ = 80
MOVIE_RHO_MIN = 0.20
MOVIE_RHO_MAX = 0.92
MOVIE_MAXTIME = 135.0
MOVIE_TIMES_TO_TRACE = 720
MOVIE_FRAMES = 32
MOVIE_SUBSTEPS_PER_FRAME = 6
MOVIE_DT = 1.2e-3
MOVIE_POTENTIAL_ITERATIONS = 3072
MOVIE_POTENTIAL_REGULARIZATION = 5.0
MOVIE_POTENTIAL_PRECONDITIONER = "jacobi"
MOVIE_TAIL_FRACTION = 0.50
MOVIE_RELATIVE_TOLERANCE = 0.35
MOVIE_MIN_FRAMES = 12

# Optional GIF/PNG media stage. This produces diagnostic artifacts from the
# same direct-coil map source. It is intentionally disabled by default because
# direct-coil media must stay out of README promotion until the geometry and
# refinement gates above are green.
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
class DirectCoilOpenSolSettings:
    output_root: Path
    case_label: str
    coil_json_path: Path | None
    vmec_wout_path: Path | None
    essos_root: Path | None
    precision: str
    write_dry_run_contract: bool
    run_live_fci_gate: bool
    run_live_connection_refinement_gate: bool
    run_live_endpoint_label_refinement_gate: bool
    run_live_collocated_endpoint_label_refinement_gate: bool
    run_live_stationarity_gate: bool
    run_live_media_gate: bool
    require_promotion_ready: bool


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
    run_live_endpoint_label_refinement_gate: bool = RUN_LIVE_ENDPOINT_LABEL_REFINEMENT_GATE,
    run_live_collocated_endpoint_label_refinement_gate: bool = RUN_LIVE_COLLOCATED_ENDPOINT_LABEL_REFINEMENT_GATE,
    run_live_stationarity_gate: bool = RUN_LIVE_STATIONARITY_GATE,
    run_live_media_gate: bool = RUN_LIVE_MEDIA_GATE,
    require_promotion_ready: bool = REQUIRE_PROMOTION_READY,
) -> DirectCoilOpenSolSettings:
    """Resolve top-level direct-coil open-SOL workflow settings."""

    return DirectCoilOpenSolSettings(
        output_root=Path(output_root),
        case_label=str(case_label),
        coil_json_path=None if coil_json_path is None else Path(coil_json_path),
        vmec_wout_path=None if vmec_wout_path is None else Path(vmec_wout_path),
        essos_root=None if essos_root is None else Path(essos_root),
        precision=str(precision),
        write_dry_run_contract=bool(write_dry_run_contract),
        run_live_fci_gate=bool(run_live_fci_gate),
        run_live_connection_refinement_gate=bool(run_live_connection_refinement_gate),
        run_live_endpoint_label_refinement_gate=bool(run_live_endpoint_label_refinement_gate),
        run_live_collocated_endpoint_label_refinement_gate=bool(
            run_live_collocated_endpoint_label_refinement_gate
        ),
        run_live_stationarity_gate=bool(run_live_stationarity_gate),
        run_live_media_gate=bool(run_live_media_gate),
        require_promotion_ready=bool(require_promotion_ready),
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
    settings: DirectCoilOpenSolSettings,
    stage_reports: list[dict[str, Any]],
) -> Path:
    """Write a compact promotion ledger for the direct-coil lane."""

    data_dir = settings.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_path = data_dir / f"{settings.case_label}_workflow_summary.json"
    live_stage_reports = [
        report
        for report in stage_reports
        if report["status"] not in {"skipped", "contract_only", "diagnostic"}
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
        "diagnostic": "essos_direct_coil_open_sol_workflow",
        "map_source": MAP_SOURCE,
        "connection_refinement_quantities": list(REFINEMENT_QUANTITIES),
        "diagnostic_connection_refinement_quantities": list(DIAGNOSTIC_REFINEMENT_QUANTITIES),
        "claim_boundary": (
            "Direct ESSOS coil-field open-SOL workflow. The default run writes a "
            "contract only; a promoted movie requires live FCI, connection-length, "
            "endpoint/source, and stationarity gates to pass."
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
            "run_live_endpoint_label_refinement_gate": settings.run_live_endpoint_label_refinement_gate,
            "run_live_collocated_endpoint_label_refinement_gate": (
                settings.run_live_collocated_endpoint_label_refinement_gate
            ),
            "run_live_stationarity_gate": settings.run_live_stationarity_gate,
            "run_live_media_gate": settings.run_live_media_gate,
            "require_promotion_ready": settings.require_promotion_ready,
        },
        "stage_reports": stage_reports,
        "promotion_ready": promotion_ready,
        "promotion_rejection_reasons": sorted(set(promotion_rejection_reasons)),
        "promotion_blocking_stages": blocking_stage_records,
        "diagnostic_stages": diagnostic_stage_records,
        "next_actions": next_actions,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote direct-coil workflow summary: {summary_path}")
    if settings.require_promotion_ready and not promotion_ready:
        raise RuntimeError(
            "Direct-coil open-SOL workflow is not promotion-ready: "
            f"{payload['promotion_rejection_reasons']}"
        )
    return summary_path


def run_fci_gate(settings: DirectCoilOpenSolSettings) -> dict[str, Any]:
    """Run or describe the direct-coil FCI endpoint/source validation gate."""

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
        print(f"wrote direct-coil FCI dry-run contract: {dry_run.contract_json_path}")

    if not settings.run_live_fci_gate:
        return {
            "stage": "direct_coil_fci_endpoint_source_gate",
            "status": "contract_only" if settings.write_dry_run_contract else "skipped",
            "promotion_ready": False,
            "next_action": "Set RUN_LIVE_FCI_GATE=True after ESSOS coil geometry is available.",
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
        "stage": "direct_coil_fci_endpoint_source_gate",
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
    settings: DirectCoilOpenSolSettings,
    fci_stage: dict[str, Any],
) -> dict[str, Any]:
    """Validate target/source/profile artifacts produced by the FCI gate."""

    if fci_stage.get("status") != "ran":
        return {
            "stage": "direct_coil_source_profile_gate",
            "status": "contract_only" if settings.write_dry_run_contract else "skipped",
            "promotion_ready": False,
            "next_action": (
                "Set RUN_LIVE_FCI_GATE=True to generate target-label, heat-load, "
                "neutral-source, and radial-profile artifacts for this map."
            ),
        }

    report_path = Path(str(fci_stage["report_json_path"]))
    arrays_path = Path(str(fci_stage["arrays_npz_path"]))
    report = _read_json(report_path)
    gate_path = settings.output_root / "data" / f"{settings.case_label}_source_profile_gate.json"
    gate_plot_path = settings.output_root / "images" / f"{settings.case_label}_source_profile_gate.png"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(arrays_path) as arrays:
        gate = build_essos_imported_fci_source_profile_gate(report, arrays)
        save_essos_imported_fci_source_profile_gate_plot(gate, arrays, gate_plot_path)
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "stage": "direct_coil_source_profile_gate",
        "status": "ran",
        "report_json_path": _path_text(gate_path),
        "plot_png_path": _path_text(gate_plot_path),
        "source_report_json_path": _path_text(report_path),
        "source_arrays_npz_path": _path_text(arrays_path),
        "source_plot_png_path": fci_stage.get("plot_png_path"),
        "passed": bool(gate.get("passed", False)),
        "promotion_ready": bool(gate.get("promotion_ready", False)),
        "evidence_role": gate.get("evidence_role"),
        "promotion_rejection_reasons": gate.get("promotion_rejection_reasons", []),
    }


def run_connection_refinement_gate(
    settings: DirectCoilOpenSolSettings,
    *,
    quantity: str,
    diagnostic: bool = False,
) -> dict[str, Any]:
    """Run or describe one pure-coil connection-length refinement blocker."""

    if not settings.run_live_connection_refinement_gate:
        return {
            "stage": f"direct_coil_{quantity}_refinement_gate",
            "status": "diagnostic" if diagnostic else "skipped",
            "promotion_ready": False,
            "next_action": (
                "Set RUN_LIVE_CONNECTION_REFINEMENT_GATE=True to test whether "
                f"pure-coil {quantity} maps have promotion-grade refinement evidence."
            ),
        }

    artifacts = create_live_essos_imported_connection_length_refinement_package(
        output_root=settings.output_root / "connection_length",
        case_label=f"{settings.case_label}_{quantity}",
        coil_json_path=settings.coil_json_path,
        vmec_wout_path=settings.vmec_wout_path,
        essos_root=settings.essos_root,
        map_source=MAP_SOURCE,
        connection_quantity=quantity,
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
        "stage": f"direct_coil_{quantity}_refinement_gate",
        "status": "diagnostic" if diagnostic else "ran",
        "connection_quantity": quantity,
        "report_json_path": _path_text(artifacts.report_json_path),
        "arrays_npz_path": _path_text(artifacts.arrays_npz_path),
        "plot_png_path": _path_text(artifacts.plot_png_path),
        "passed": bool(report.get("passed", False)),
        "promotion_ready": False if diagnostic else bool(report.get("promotion_ready", report.get("passed", False))),
        "evidence_role": report.get("evidence_role"),
        "diagnostic_only": bool(diagnostic),
        "promotion_rejection_reasons": report.get("promotion_rejection_reasons", []),
    }


def run_endpoint_label_refinement_gate(settings: DirectCoilOpenSolSettings) -> dict[str, Any]:
    """Run or describe categorical endpoint-label refinement for direct-coil maps."""

    if not settings.run_live_endpoint_label_refinement_gate:
        return {
            "stage": "direct_coil_endpoint_label_refinement_gate",
            "status": "skipped",
            "promotion_ready": False,
            "next_action": (
                "Set RUN_LIVE_ENDPOINT_LABEL_REFINEMENT_GATE=True to compare "
                "nested directional endpoint labels before promoting direct-coil media."
            ),
        }

    artifacts = create_live_essos_imported_endpoint_label_refinement_package(
        output_root=settings.output_root / "endpoint_labels",
        case_label=f"{settings.case_label}_endpoint_labels",
        coil_json_path=settings.coil_json_path,
        vmec_wout_path=settings.vmec_wout_path,
        essos_root=settings.essos_root,
        map_source=MAP_SOURCE,
        level_shapes=REFINEMENT_LEVEL_SHAPES,
        rho_min=FCI_RHO_MIN,
        rho_max=FCI_RHO_MAX,
        maxtime=REFINEMENT_MAXTIME,
        times_to_trace=REFINEMENT_TIMES_TO_TRACE,
        trace_tolerance=FCI_TRACE_TOLERANCE,
        minimum_agreement_fraction=ENDPOINT_LABEL_MINIMUM_AGREEMENT_FRACTION,
        minimum_endpoint_agreement_fraction=ENDPOINT_LABEL_MINIMUM_ENDPOINT_AGREEMENT_FRACTION,
        minimum_endpoint_union_fraction=ENDPOINT_LABEL_MINIMUM_ENDPOINT_UNION_FRACTION,
    )
    report = _read_json(artifacts.report_json_path)
    return {
        "stage": "direct_coil_endpoint_label_refinement_gate",
        "status": "ran",
        "report_json_path": _path_text(artifacts.report_json_path),
        "arrays_npz_path": _path_text(artifacts.arrays_npz_path),
        "plot_png_path": _path_text(artifacts.plot_png_path),
        "passed": bool(report.get("passed", False)),
        "promotion_ready": bool(report.get("promotion_ready", report.get("passed", False))),
        "evidence_role": report.get("evidence_role"),
        "promotion_rejection_reasons": report.get("promotion_rejection_reasons", []),
    }


def run_collocated_endpoint_label_refinement_gate(settings: DirectCoilOpenSolSettings) -> dict[str, Any]:
    """Run an odd-ratio endpoint-label diagnostic with collocated seed angles."""

    if not settings.run_live_collocated_endpoint_label_refinement_gate:
        return {
            "stage": "direct_coil_collocated_endpoint_label_refinement_gate",
            "status": "diagnostic",
            "promotion_ready": False,
            "next_action": (
                "Set RUN_LIVE_COLLOCATED_ENDPOINT_LABEL_REFINEMENT_GATE=True to "
                "test whether endpoint-label instability is caused by non-collocated "
                "even-ratio seed grids."
            ),
        }

    artifacts = create_live_essos_imported_endpoint_label_refinement_package(
        output_root=settings.output_root / "endpoint_labels_collocated",
        case_label=f"{settings.case_label}_endpoint_labels_collocated",
        coil_json_path=settings.coil_json_path,
        vmec_wout_path=settings.vmec_wout_path,
        essos_root=settings.essos_root,
        map_source=MAP_SOURCE,
        level_shapes=COLLOCATED_ENDPOINT_LABEL_LEVEL_SHAPES,
        rho_min=FCI_RHO_MIN,
        rho_max=FCI_RHO_MAX,
        maxtime=REFINEMENT_MAXTIME,
        times_to_trace=REFINEMENT_TIMES_TO_TRACE,
        trace_tolerance=FCI_TRACE_TOLERANCE,
        minimum_agreement_fraction=ENDPOINT_LABEL_MINIMUM_AGREEMENT_FRACTION,
        minimum_endpoint_agreement_fraction=ENDPOINT_LABEL_MINIMUM_ENDPOINT_AGREEMENT_FRACTION,
        minimum_endpoint_union_fraction=ENDPOINT_LABEL_MINIMUM_ENDPOINT_UNION_FRACTION,
        require_three_levels=False,
    )
    report = _read_json(artifacts.report_json_path)
    return {
        "stage": "direct_coil_collocated_endpoint_label_refinement_gate",
        "status": "diagnostic",
        "report_json_path": _path_text(artifacts.report_json_path),
        "arrays_npz_path": _path_text(artifacts.arrays_npz_path),
        "plot_png_path": _path_text(artifacts.plot_png_path),
        "passed": bool(report.get("passed", False)),
        "promotion_ready": False,
        "evidence_role": report.get("evidence_role"),
        "dominant_endpoint_instability_mode": report.get("diagnostics", {}).get(
            "dominant_endpoint_instability_mode"
        ),
        "dominant_direction_component_error": report.get("diagnostics", {}).get(
            "dominant_direction_component_error"
        ),
        "promotion_rejection_reasons": report.get("promotion_rejection_reasons", []),
    }


def run_stationarity_gate(settings: DirectCoilOpenSolSettings) -> dict[str, Any]:
    """Run or describe the report-only direct-coil reduced-transient gate."""

    if not settings.run_live_stationarity_gate:
        return {
            "stage": "direct_coil_reduced_transient_stationarity_gate",
            "status": "skipped",
            "promotion_ready": False,
            "next_action": (
                "Set RUN_LIVE_STATIONARITY_GATE=True only after the FCI and "
                "connection-length gates have passed for the same map source."
            ),
        }

    configure_jax_runtime(precision=settings.precision)
    artifacts = create_essos_imported_drb_movie_stationarity_package(
        output_root=settings.output_root / "stationarity",
        case_label=f"{settings.case_label}_stationarity",
        coil_json_path=settings.coil_json_path,
        vmec_wout_path=settings.vmec_wout_path,
        essos_root=settings.essos_root,
        map_source=MAP_SOURCE,
        nx=MOVIE_NX,
        ny=MOVIE_NY,
        nz=MOVIE_NZ,
        rho_min=MOVIE_RHO_MIN,
        rho_max=MOVIE_RHO_MAX,
        maxtime=MOVIE_MAXTIME,
        times_to_trace=MOVIE_TIMES_TO_TRACE,
        frames=MOVIE_FRAMES,
        substeps_per_frame=MOVIE_SUBSTEPS_PER_FRAME,
        dt=MOVIE_DT,
        potential_iterations=MOVIE_POTENTIAL_ITERATIONS,
        potential_regularization=MOVIE_POTENTIAL_REGULARIZATION,
        potential_preconditioner=MOVIE_POTENTIAL_PRECONDITIONER,
        tail_fraction=MOVIE_TAIL_FRACTION,
        relative_tolerance=MOVIE_RELATIVE_TOLERANCE,
        min_frames=MOVIE_MIN_FRAMES,
    )
    report = _read_json(artifacts.report_json_path)
    return {
        "stage": "direct_coil_reduced_transient_stationarity_gate",
        "status": "ran",
        "report_json_path": _path_text(artifacts.report_json_path),
        "stationarity_passed": bool(report.get("stationarity_passed", False)),
        "promotion_ready": bool(report.get("publication_ready", False)),
        "movie_promotion_rejection_reasons": report.get(
            "movie_promotion_rejection_reasons",
            [],
        ),
    }


def run_media_gate(settings: DirectCoilOpenSolSettings) -> dict[str, Any]:
    """Run or describe the diagnostic direct-coil GIF/PNG media stage."""

    if not settings.run_live_media_gate:
        return {
            "stage": "direct_coil_diagnostic_turbulence_media",
            "status": "skipped",
            "promotion_ready": False,
            "next_action": (
                "Set RUN_LIVE_MEDIA_GATE=True only after the live FCI, "
                "endpoint-label, connection-length, and stationarity gates "
                "have been reviewed for this direct-coil map source."
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
        "stage": "direct_coil_diagnostic_turbulence_media",
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
        "movie_promotion_rejection_reasons": report.get(
            "movie_promotion_rejection_reasons",
            [],
        ),
    }


def run_direct_coil_workflow(settings: DirectCoilOpenSolSettings) -> Path:
    """Run the configured direct-coil open-SOL workflow stages."""

    fci_stage = run_fci_gate(settings)
    stage_reports = [
        fci_stage,
        run_source_profile_gate(settings, fci_stage),
        run_endpoint_label_refinement_gate(settings),
        run_collocated_endpoint_label_refinement_gate(settings),
        *(
            run_connection_refinement_gate(settings, quantity=quantity)
            for quantity in REFINEMENT_QUANTITIES
        ),
        *(
            run_connection_refinement_gate(settings, quantity=quantity, diagnostic=True)
            for quantity in DIAGNOSTIC_REFINEMENT_QUANTITIES
        ),
        run_stationarity_gate(settings),
        run_media_gate(settings),
    ]
    return write_workflow_summary(settings, stage_reports)


if RUN_EXAMPLE:
    DIRECT_COIL_SETTINGS = build_settings()
    DIRECT_COIL_WORKFLOW_SUMMARY = run_direct_coil_workflow(DIRECT_COIL_SETTINGS)
