"""Open-SOL promotion workflow on imported Landreman-Paul QA geometry.

Two map lanes share one ledger (pick with ``MAP_SOURCE``):

- ``"hybrid"``: smooth VMEC coordinates with coil-derived endpoint masks and
  |B| modulation -- the open-SOL bridge while pure coil maps are too rough.
  Extra gates: grid/time movie refinement and an audit of the committed
  release-backed high-resolution evidence (FCI report, stationarity report,
  refinement summary, media manifest under ``docs/data``).
- ``"coil"``: the pure direct-coil lane. Extra gates: categorical endpoint-label
  refinement (plain, plus collocated odd-ratio and boundary-resolved
  diagnostics) and a diagnostic ``target_exit_length`` refinement.

Common gates: FCI endpoint/source validation, source/heat-load/profile gate,
connection-length refinement, reduced-transient stationarity, and diagnostic
media (GIF/PNG). By default every live gate is off and the run is
self-contained: it writes the FCI dry-run contract and the promotion ledger
``data/<CASE_LABEL>_workflow_summary.json`` recording which gates must pass
before any movie may be promoted. Live gates need an ESSOS checkout
(``DRBX_ESSOS_ROOT`` or ``ESSOS_ROOT``) and are enabled with the ``RUN_LIVE_*``
flags. Outputs land under ``output/open_sol_workflow/<MAP_SOURCE>``
(cwd-relative); every path is printed.

Run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/essos-field-lines/open_sol_workflow.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from drbx.runtime import configure_jax_runtime
from drbx.validation import (
    audit_hybrid_open_sol_promotion_evidence,
    build_essos_imported_fci_source_profile_gate,
    create_essos_imported_drb_movie_package,
    create_essos_imported_drb_movie_refinement_campaign_package,
    create_essos_imported_drb_movie_stationarity_package,
    create_essos_imported_fci_campaign_package,
    create_live_essos_imported_connection_length_refinement_package,
    create_live_essos_imported_endpoint_label_refinement_package,
    save_essos_imported_fci_source_profile_gate_plot,
)
from drbx.validation.essos_imported_fci_campaign import create_essos_imported_fci_dry_run_artifact_package

# --- PARAMETERS ------------------------------------------------------------------
MAP_SOURCE = "hybrid"  # "hybrid" or "coil"
WRITE_DRY_RUN_CONTRACT = True
RUN_LIVE_FCI_GATE = False  # also runs the source/profile gate on its output
RUN_LIVE_CONNECTION_REFINEMENT_GATE = False
RUN_LIVE_ENDPOINT_LABEL_GATES = False  # coil lane only
RUN_LIVE_STATIONARITY_GATE = False
RUN_LIVE_MOVIE_REFINEMENT_GATE = False  # hybrid lane only
RUN_LIVE_MEDIA_GATE = False
RUN_RELEASE_EVIDENCE_AUDIT = True  # hybrid lane only; reads docs/data JSONs
REQUIRE_PROMOTION_READY = False
STATIONARITY_PRESET = "quick"  # "quick" = workflow smoke (never promotes), "promotion" = movie settings

OUTPUT_ROOT = Path("output/open_sol_workflow") / MAP_SOURCE
CASE_LABEL = "essos_hybrid_open_sol" if MAP_SOURCE == "hybrid" else "essos_direct_coil_open_sol"
SOURCE = dict(coil_json_path=None, vmec_wout_path=None, essos_root=None, map_source=MAP_SOURCE)  # None -> ESSOS checkout
PRECISION = "float64"

FCI = dict(nx=5, ny=8, nz=20, rho_min=0.12, rho_max=0.34, maxtime=80.0, times_to_trace=360,
           trace_tolerance=1.0e-8, require_connection_resolution=True)
CONNECTION_QUANTITIES = ("parallel_step_per_toroidal_radian",) if MAP_SOURCE == "hybrid" else ("adjacent_step_length",)
DIAGNOSTIC_CONNECTION_QUANTITIES = () if MAP_SOURCE == "hybrid" else ("target_exit_length",)
REFINEMENT = dict(level_shapes=((3, 4, 6), (6, 8, 12), (12, 16, 24)), rho_min=0.12, rho_max=0.34, maxtime=40.0,
                  times_to_trace=160, trace_tolerance=1.0e-8)
CONNECTION_GATE = dict(convergence_threshold=0.10, linf_threshold=0.20, minimum_observed_order=0.50,
                       require_observed_order=True)
ENDPOINT_LABEL_GATE = dict(minimum_agreement_fraction=0.90, minimum_endpoint_agreement_fraction=0.80,
                           minimum_endpoint_union_fraction=0.01, minimum_boundary_excluded_valid_fraction=0.20)
COLLOCATED_ENDPOINT_LEVELS = ((3, 5, 9), (7, 15, 27))
BOUNDARY_RESOLVED_ENDPOINT_LEVELS = ((7, 15, 27), (11, 25, 45))
MOVIE = (  # reduced-transient / media settings per lane
    dict(nx=16, ny=96, nz=48, rho_min=0.20, rho_max=0.60, maxtime=24.0, times_to_trace=80, frames=12,
         substeps_per_frame=3, dt=2.0e-3)
    if MAP_SOURCE == "hybrid" else
    dict(nx=8, ny=28, nz=80, rho_min=0.20, rho_max=0.92, maxtime=135.0, times_to_trace=720, frames=32,
         substeps_per_frame=6, dt=1.2e-3)
)
POTENTIAL = dict(potential_iterations=3072, potential_regularization=5.0, potential_preconditioner="jacobi")
STATIONARITY_GATE = dict(tail_fraction=0.50, relative_tolerance=0.35, min_frames=12)
QUICK_STATIONARITY = dict(nx=4, ny=12, nz=24, rho_min=0.20, rho_max=0.60, maxtime=18.0, times_to_trace=48, frames=4,
                          substeps_per_frame=1, dt=1.5e-3, potential_iterations=384, tail_fraction=0.50,
                          relative_tolerance=0.75, min_frames=4)
MOVIE_REFINEMENT = dict(grid_shapes=((8, 12, 24), (16, 24, 48)), time_shape=(16, 24, 48), time_dt_values=(2.0e-3, 1.0e-3),
                        frames=4, substeps_per_frame=2, grid_dt=2.0e-3)
RELEASE_EVIDENCE = dict(
    fci_report_json_path=Path("docs/data/essos_imported_fci_hybrid_artifacts/data/essos_imported_fci_hybrid_campaign.json"),
    stationarity_report_json_path=Path("docs/data/essos_imported_drb_movie_stationarity_jacobi_artifacts/data/"
                                       "essos_imported_drb_movie_stationarity_jacobi.json"),
    refinement_summary_json_path=Path("docs/data/essos_imported_drb_movie_refinement_poloidal_96_jacobi_artifacts/data/"
                                      "essos_imported_drb_movie_refinement_poloidal_96_jacobi_summary.json"),
    media_manifest_json_path=Path("docs/data/essos_imported_drb_movie_stationarity_jacobi_media_manifest.json"),
)
# endpoint-label report fields copied into the ledger (top level, then report["diagnostics"])
ENDPOINT_LABEL_KEYS = (
    "dominant_endpoint_instability_mode", "dominant_direction_component_error", "target_boundary_projection_suspected",
    "minimum_boundary_excluded_valid_fraction_actual", "minimum_boundary_excluded_valid_fraction_required",
    "boundary_excluded_valid_fraction_passed", "boundary_excluded_agreement_passed",
    "boundary_excluded_endpoint_agreement_passed", "target_boundary_only_instability",
    "projection_neighborhood_supported", "conservative_projection_available",
    "minimum_conservative_projection_agreement_fraction_actual",
    "minimum_conservative_projection_endpoint_agreement_fraction_actual", "signed_target_transition_available",
    "dominant_signed_target_transition_mode", "dominant_signed_target_transition_shell_mode",
    "minimum_signed_target_transition_consistency_fraction_actual",
    "maximum_signed_target_projection_false_positive_fraction_actual",
    "minimum_signed_target_transition_shell_mismatch_coverage_fraction_actual",
    "minimum_signed_target_transition_shell_label_stability_fraction_actual",
    "minimum_signed_target_transition_shell_evidence_stability_fraction_actual",
    "maximum_signed_target_transition_bulk_mismatch_fraction_actual",
    "signed_target_transition_shell_refinement_supported", "signed_target_transition_shell_only_instability",
    "dominant_endpoint_boundary_localization", "projection_recommended_next_action",
)

# --- helpers -----------------------------------------------------------------------
LANE = "hybrid" if MAP_SOURCE == "hybrid" else "direct_coil"
NOT_LIVE = {"skipped", "contract_only", "diagnostic", "release_evidence"}


def stage(name, status, next_action=None, artifacts=None, report=None, ready=None, **extra):
    """One ledger record: status, artifact paths, and the report's verdict."""

    record = {"stage": f"{LANE}_{name}", "status": status, "promotion_ready": bool(ready)}
    if next_action:
        record["next_action"] = next_action
    for key, value in (vars(artifacts) if artifacts else {}).items():
        if key.endswith("_path") and value is not None:
            record[key] = Path(value).as_posix()
        elif key.endswith("_paths"):
            record[key] = [Path(item).as_posix() for item in value]
    for key in ("passed", "evidence_role", "promotion_rejection_reasons", "movie_promotion_rejection_reasons",
                "stationarity_passed", "grid_refinement_passed", "time_refinement_passed", "movie_visual_qa_passed",
                "movie_evidence_role"):
        if report and key in report:
            record[key] = report[key]
    record.update(extra)
    print(f"  {record['stage']}: status={status}, promotion_ready={record['promotion_ready']}")
    return record


def read(artifacts):
    return json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))


def blocker(record):
    reasons = list(record.get("promotion_rejection_reasons", [])) + list(record.get("movie_promotion_rejection_reasons", []))
    if not reasons:
        status = record["status"]
        reasons = [f"{status}_stage_not_live_promotion_evidence" if status in NOT_LIVE else "stage_not_promotion_ready"]
    return {"stage": record["stage"], "status": record["status"], "reasons": reasons, "next_action": record.get("next_action")}


def timed(label, function, **kwargs):
    print(f"  running {label}...")
    start = time.perf_counter()
    result = function(**kwargs)
    print(f"  {label} finished in {time.perf_counter() - start:.1f} s")
    return result


# --- run the gates ------------------------------------------------------------------
configure_jax_runtime(precision=PRECISION)
print(f"open-SOL workflow: lane={LANE}, map_source={MAP_SOURCE}, output_root={OUTPUT_ROOT}")
stages = []
fci_kwargs = dict(output_root=OUTPUT_ROOT / "fci", case_label=f"{CASE_LABEL}_fci", **SOURCE, **FCI)
if WRITE_DRY_RUN_CONTRACT:
    contract = create_essos_imported_fci_dry_run_artifact_package(**fci_kwargs, precision=PRECISION)
    print(f"  wrote FCI dry-run contract: {contract.contract_json_path}")

# FCI endpoint/source gate, then the source/heat-load/profile gate on its arrays
if RUN_LIVE_FCI_GATE:
    fci = timed("FCI gate", create_essos_imported_fci_campaign_package, **fci_kwargs)
    report = read(fci)
    stages.append(stage("fci_endpoint_source_gate", "ran", artifacts=fci, report=report, ready=report.get("passed")
                        and report.get("connection_length_resolution_passed") and report.get("map_diagnostics_passed")))
    gate_path = OUTPUT_ROOT / "data" / f"{CASE_LABEL}_source_profile_gate.json"
    gate_plot_path = OUTPUT_ROOT / "images" / f"{CASE_LABEL}_source_profile_gate.png"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(fci.arrays_npz_path) as arrays:
        gate = build_essos_imported_fci_source_profile_gate(report, arrays)
        save_essos_imported_fci_source_profile_gate_plot(gate, arrays, gate_plot_path)
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    stages.append(stage("source_profile_gate", "ran", report=gate, ready=gate.get("promotion_ready"),
                        report_json_path=gate_path.as_posix(), plot_png_path=gate_plot_path.as_posix(),
                        source_report_json_path=fci.report_json_path.as_posix()))
else:
    status = "contract_only" if WRITE_DRY_RUN_CONTRACT else "skipped"
    stages.append(stage("fci_endpoint_source_gate", status, "Set RUN_LIVE_FCI_GATE=True once ESSOS geometry is available."))
    stages.append(stage("source_profile_gate", status, "Set RUN_LIVE_FCI_GATE=True to generate target-label, heat-load, "
                        "neutral-source, and radial-profile artifacts for this map."))

# endpoint-label refinement (coil lane): one promotion gate plus two diagnostics
if MAP_SOURCE == "coil":
    for name, levels, diagnostic in (("endpoint_label_refinement_gate", REFINEMENT["level_shapes"], False),
                                     ("collocated_endpoint_label_refinement_gate", COLLOCATED_ENDPOINT_LEVELS, True),
                                     ("boundary_resolved_endpoint_label_refinement_gate", BOUNDARY_RESOLVED_ENDPOINT_LEVELS, True)):
        status = "diagnostic" if diagnostic else ("ran" if RUN_LIVE_ENDPOINT_LABEL_GATES else "skipped")
        if not RUN_LIVE_ENDPOINT_LABEL_GATES:
            stages.append(stage(name, status, f"Set RUN_LIVE_ENDPOINT_LABEL_GATES=True to compare nested endpoint labels on {levels}."))
            continue
        artifacts = timed(name, create_live_essos_imported_endpoint_label_refinement_package,
                          output_root=OUTPUT_ROOT / name, case_label=f"{CASE_LABEL}_{name}", **SOURCE,
                          **{**REFINEMENT, "level_shapes": levels}, **ENDPOINT_LABEL_GATE,
                          **({"require_three_levels": False} if diagnostic else {}))
        report = read(artifacts)
        diagnostics = report.get("diagnostics", {})
        pairs = diagnostics.get("pair_reports", [])
        extra = {key: report.get(key, diagnostics.get(key)) for key in ENDPOINT_LABEL_KEYS}
        for key in ("projection_neighborhood_mismatch_support_fraction", "projection_neighborhood_endpoint_mismatch_support_fraction"):
            values = [float(pair[key]) for pair in pairs if pair.get(key) is not None]
            extra[f"{key}_max"] = max(values) if values else None
        stages.append(stage(name, status, artifacts=artifacts, report=report,
                            ready=False if diagnostic else report.get("promotion_ready", report.get("passed")), **extra))

# connection-length refinement (diagnostic quantities never promote)
for quantity, diagnostic in [(q, False) for q in CONNECTION_QUANTITIES] + [(q, True) for q in DIAGNOSTIC_CONNECTION_QUANTITIES]:
    name = f"{quantity}_refinement_gate"
    if not RUN_LIVE_CONNECTION_REFINEMENT_GATE:
        stages.append(stage(name, "diagnostic" if diagnostic else "skipped",
                            f"Set RUN_LIVE_CONNECTION_REFINEMENT_GATE=True to test {quantity} refinement."))
        continue
    artifacts = timed(name, create_live_essos_imported_connection_length_refinement_package,
                      output_root=OUTPUT_ROOT / "connection_length", case_label=f"{CASE_LABEL}_{quantity}", **SOURCE,
                      connection_quantity=quantity, **REFINEMENT, **CONNECTION_GATE)
    report = read(artifacts)
    stages.append(stage(name, "diagnostic" if diagnostic else "ran", artifacts=artifacts, report=report,
                        ready=not diagnostic and report.get("promotion_ready", report.get("passed")),
                        connection_quantity=quantity))

# reduced-transient stationarity (the quick preset never promotes)
if RUN_LIVE_STATIONARITY_GATE:
    parameters = QUICK_STATIONARITY if STATIONARITY_PRESET == "quick" else {**MOVIE, **POTENTIAL, **STATIONARITY_GATE}
    parameters = {**POTENTIAL, **parameters}
    artifacts = timed("stationarity gate", create_essos_imported_drb_movie_stationarity_package,
                      output_root=OUTPUT_ROOT / "stationarity", case_label=f"{CASE_LABEL}_stationarity", **SOURCE, **parameters)
    report = read(artifacts)
    reasons = list(report.get("movie_promotion_rejection_reasons", []))
    if report.get("publication_ready") and STATIONARITY_PRESET == "quick":
        reasons.append("quick_stationarity_preset_not_promotion_evidence")
    stages.append(stage("reduced_transient_stationarity_gate", "ran", artifacts=artifacts,
                        report={**report, "movie_promotion_rejection_reasons": reasons},
                        ready=report.get("publication_ready") and STATIONARITY_PRESET != "quick",
                        stationarity_preset=STATIONARITY_PRESET))
else:
    stages.append(stage("reduced_transient_stationarity_gate", "skipped",
                        "Set RUN_LIVE_STATIONARITY_GATE=True after the FCI and connection-length gates pass."))

# grid/time movie refinement (hybrid lane)
if MAP_SOURCE == "hybrid":
    if RUN_LIVE_MOVIE_REFINEMENT_GATE:
        artifacts = timed("movie refinement gate", create_essos_imported_drb_movie_refinement_campaign_package,
                          output_root=OUTPUT_ROOT / "movie_refinement", case_label=f"{CASE_LABEL}_movie_refinement",
                          **SOURCE, **MOVIE_REFINEMENT, **POTENTIAL, rho_min=MOVIE["rho_min"], rho_max=MOVIE["rho_max"],
                          maxtime=MOVIE["maxtime"], times_to_trace=MOVIE["times_to_trace"],
                          relative_tolerance=STATIONARITY_GATE["relative_tolerance"], reuse_existing_reports=True)
        report = read(artifacts)
        stages.append(stage("movie_grid_time_refinement_gate", "ran", artifacts=artifacts, report=report,
                            ready=report.get("publication_ready")))
    else:
        stages.append(stage("movie_grid_time_refinement_gate", "skipped",
                            "Set RUN_LIVE_MOVIE_REFINEMENT_GATE=True before promoting hybrid media."))

# diagnostic GIF/PNG media
if RUN_LIVE_MEDIA_GATE:
    artifacts = timed("media gate", create_essos_imported_drb_movie_package, output_root=OUTPUT_ROOT / "media",
                      case_label=f"{CASE_LABEL}_diagnostic_media", **SOURCE, **MOVIE, **POTENTIAL)
    report = read(artifacts)
    stages.append(stage("diagnostic_turbulence_media", "ran", artifacts=artifacts, report=report,
                        ready=report.get("publication_ready")))
else:
    stages.append(stage("diagnostic_turbulence_media", "skipped",
                        "Set RUN_LIVE_MEDIA_GATE=True only after the live gates above have been reviewed."))

# release-backed evidence audit (hybrid lane)
if MAP_SOURCE == "hybrid" and RUN_RELEASE_EVIDENCE_AUDIT:
    audit = audit_hybrid_open_sol_promotion_evidence(**RELEASE_EVIDENCE)
    audit_path = OUTPUT_ROOT / "data" / f"{CASE_LABEL}_release_evidence_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  wrote release-evidence audit: {audit_path}")
    stages.append(stage("release_evidence_audit", "release_evidence",
                        "Regenerate or repair the high-resolution hybrid FCI/source, stationarity, "
                        "grid/time-refinement, or media-manifest evidence.", report=audit,
                        ready=audit.get("promotion_ready"), report_json_path=audit_path.as_posix(),
                        audited_evidence_paths=audit.get("evidence_paths", {}),
                        audited_stage_reports=audit.get("stage_reports", [])))

# --- promotion ledger -----------------------------------------------------------------
live = [record for record in stages if record["status"] not in NOT_LIVE]
promotable = [record for record in stages if record["status"] != "diagnostic"]
promotion_ready = bool(promotable) and all(record["promotion_ready"] for record in promotable)
blocking = [blocker(record) for record in promotable if not record["promotion_ready"]]
diagnostic = [blocker(record) for record in stages if record["status"] == "diagnostic" and not record["promotion_ready"]]
release_ready = any(record["promotion_ready"] for record in stages if record["status"] == "release_evidence")
reasons = sorted({reason for record in blocking for reason in record["reasons"]} | (set() if live else {"no_live_promotion_gates_ran"}))
if promotion_ready:
    closeout = "live_promotion_ready"
elif release_ready:
    closeout = "release_backed_compact_vacuum_bridge_ready" if not live else "release_backed_evidence_ready_live_gates_incomplete"
elif live:
    closeout = "live_evidence_incomplete"
else:
    closeout = "finalized_diagnostic_contract" if MAP_SOURCE == "coil" else "workflow_contract_or_live_evidence_incomplete"
ledger = {
    "diagnostic": f"essos_{LANE}_open_sol_workflow",
    "map_source": MAP_SOURCE,
    "connection_refinement_quantities": list(CONNECTION_QUANTITIES),
    "diagnostic_connection_refinement_quantities": list(DIAGNOSTIC_CONNECTION_QUANTITIES),
    "claim_boundary": (f"{LANE} open-SOL workflow. The default run writes a contract only; a promoted movie requires "
                       "live FCI/source/profile, connection-length, stationarity (and, for hybrid, grid/time and "
                       "visual-QA) gates to pass."),
    "settings": {key: value for key, value in globals().items() if key.startswith(("RUN_", "WRITE_", "REQUIRE_", "STATIONARITY_PRESET"))},
    "stage_reports": stages,
    "near_term_closeout_status": closeout,
    "release_evidence_ready": release_ready,
    "deferred_claims": [f"promoted_{LANE}_open_sol_movie", "predictive_device_scale_stellarator_turbulence",
                        "finite_beta_vmec_extender_open_sol", f"full_drb_physics_on_{LANE}_geometry"],
    "promotion_ready": promotion_ready,
    "promotion_rejection_reasons": reasons,
    "promotion_blocking_stages": blocking,
    "diagnostic_stages": diagnostic,
    "next_actions": [record["next_action"] for record in blocking + diagnostic if record.get("next_action")],
}
summary_path = OUTPUT_ROOT / "data" / f"{CASE_LABEL}_workflow_summary.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(ledger, indent=2, sort_keys=True, default=str), encoding="utf-8")
print(f"closeout status: {closeout}; promotion_ready={promotion_ready}; {len(live)} live gates ran")
print(f"rejection reasons: {reasons}")
print(f"wrote workflow summary: {summary_path}")
if REQUIRE_PROMOTION_READY and not promotion_ready:
    raise RuntimeError(f"{LANE} open-SOL workflow is not promotion-ready: {reasons}")
