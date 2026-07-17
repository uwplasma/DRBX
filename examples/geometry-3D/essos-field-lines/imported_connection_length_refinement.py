"""Nested connection-length refinement gate for imported field-line maps.

The script runs the promotion-blocking grid-refinement diagnostic for
connection-length-type quantities on imported FCI maps. The default run is
fully self-contained (``LIVE_IMPORT = False``): it exercises the manufactured
nested-grid campaign, checks RMS/Linf convergence and the observed order
against the thresholds below, prints the evidence summary, and fails loudly
(``require_refinement_gate_passed``) if the gate is not promotion-ready.

Set ``LIVE_IMPORT = True`` (with an ESSOS checkout and, per source, the
Landreman-Paul QA coil JSON / VMEC wout) to regenerate live refinement reports
for any of the ``"coil"``, ``"vmec"``, and ``"hybrid"`` map sources; the
grid-invariant refinement quantity per source is resolved automatically.

Artifacts (report JSON, arrays NPZ, plot PNG, sweep summary) land under
``docs/data/essos_imported_connection_length_refinement_artifacts`` (relative
to the current working directory) and every path is printed.

Edit the PARAMETERS constants below and run from the repository root:

    PYTHONPATH=src python examples/geometry-3D/essos-field-lines/imported_connection_length_refinement.py

(The trailing ``RUN_EXAMPLE`` block is what test loaders toggle off to import
the helper functions without running the campaigns.)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from jax_drb.validation import (
    create_essos_imported_connection_length_refinement_package,
    create_live_essos_imported_connection_length_refinement_package,
)


# --- PARAMETERS: edit these values, then run this file -----------------------------
RUN_EXAMPLE = True

# The default path is self-contained and does not require external coil or VMEC
# data. Set LIVE_IMPORT = True to regenerate live imported-field promotion
# reports from an ESSOS/VMEC checkout.
LIVE_IMPORT = False
MAP_SOURCES_TO_RUN = ("hybrid",)  # "coil", "vmec", and/or "hybrid" in live mode.
CONNECTION_QUANTITY = "auto"

OUTPUT_ROOT = Path("docs/data/essos_imported_connection_length_refinement_artifacts")
CASE_LABEL = "essos_imported_connection_length_refinement"
WRITE_SWEEP_SUMMARY = True
COIL_JSON_PATH: Path | None = None
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None

LEVEL_SHAPES = (
    (4, 6, 8),
    (8, 12, 16),
    (16, 24, 32),
)
LIVE_LEVEL_SHAPES = (
    (3, 4, 6),
    (6, 8, 12),
    (12, 16, 24),
)
RHO_MIN = 0.12
RHO_MAX = 0.34
MAXTIME = 40.0
TIMES_TO_TRACE = 160
TRACE_TOLERANCE = 1.0e-8

CONVERGENCE_THRESHOLD = 0.02
LINF_THRESHOLD = 0.05
LIVE_CONVERGENCE_THRESHOLD = 0.10
LIVE_LINF_THRESHOLD = 0.20
MINIMUM_OBSERVED_ORDER = 1.5
LIVE_MINIMUM_OBSERVED_ORDER = 0.5
REQUIRE_OBSERVED_ORDER = True
REQUIRE_PASS = True


LIVE_CONNECTION_QUANTITIES = {
    # Pure coil maps still need refinement work; this quantity targets the FCI
    # adjacent-plane map rather than endpoint wall-hit distance.
    "coil": "adjacent_step_length",
    # VMEC-coordinate adjacent steps scale with toroidal spacing, so the
    # grid-invariant quantity is the parallel step per toroidal radian.
    "vmec": "parallel_step_per_toroidal_radian",
    "hybrid": "parallel_step_per_toroidal_radian",
}


@dataclass(frozen=True)
class ConnectionLengthRefinementRunSettings:
    live_import: bool
    map_source: str
    connection_quantity: str
    output_root: Path
    case_label: str
    coil_json_path: Path | None
    vmec_wout_path: Path | None
    essos_root: Path | None
    level_shapes: tuple[tuple[int, int, int], ...]
    rho_min: float
    rho_max: float
    maxtime: float
    times_to_trace: int
    trace_tolerance: float
    convergence_threshold: float
    linf_threshold: float
    minimum_observed_order: float
    require_observed_order: bool
    require_pass: bool


def resolve_connection_quantity(map_source: str, requested: str = CONNECTION_QUANTITY) -> str:
    """Resolve the refinement quantity for a live imported map source."""

    normalized_source = str(map_source).strip().lower()
    normalized_quantity = str(requested).strip().lower().replace("-", "_")
    if normalized_quantity not in {"", "auto", "default"}:
        return normalized_quantity
    try:
        return LIVE_CONNECTION_QUANTITIES[normalized_source]
    except KeyError as exc:
        raise ValueError(f"Unsupported imported map_source={map_source!r}.") from exc


def build_run_settings(
    *,
    live_import: bool = LIVE_IMPORT,
    map_sources: tuple[str, ...] = MAP_SOURCES_TO_RUN,
    connection_quantity: str = CONNECTION_QUANTITY,
    output_root: Path = OUTPUT_ROOT,
    case_label: str = CASE_LABEL,
    coil_json_path: Path | None = COIL_JSON_PATH,
    vmec_wout_path: Path | None = VMEC_WOUT_PATH,
    essos_root: Path | None = ESSOS_ROOT,
    level_shapes: tuple[tuple[int, int, int], ...] = LEVEL_SHAPES,
    live_level_shapes: tuple[tuple[int, int, int], ...] = LIVE_LEVEL_SHAPES,
    rho_min: float = RHO_MIN,
    rho_max: float = RHO_MAX,
    maxtime: float = MAXTIME,
    times_to_trace: int = TIMES_TO_TRACE,
    trace_tolerance: float = TRACE_TOLERANCE,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    live_convergence_threshold: float = LIVE_CONVERGENCE_THRESHOLD,
    linf_threshold: float = LINF_THRESHOLD,
    live_linf_threshold: float = LIVE_LINF_THRESHOLD,
    minimum_observed_order: float = MINIMUM_OBSERVED_ORDER,
    live_minimum_observed_order: float = LIVE_MINIMUM_OBSERVED_ORDER,
    require_observed_order: bool = REQUIRE_OBSERVED_ORDER,
    require_pass: bool = REQUIRE_PASS,
) -> tuple[ConnectionLengthRefinementRunSettings, ...]:
    """Resolve top-level parameters into one refinement campaign per source."""

    if not bool(live_import):
        return (
            ConnectionLengthRefinementRunSettings(
                live_import=False,
                map_source="manufactured",
                connection_quantity="manufactured",
                output_root=Path(output_root),
                case_label=str(case_label),
                coil_json_path=None,
                vmec_wout_path=None,
                essos_root=None,
                level_shapes=tuple(tuple(int(value) for value in shape) for shape in level_shapes),
                rho_min=float(rho_min),
                rho_max=float(rho_max),
                maxtime=float(maxtime),
                times_to_trace=int(times_to_trace),
                trace_tolerance=float(trace_tolerance),
                convergence_threshold=float(convergence_threshold),
                linf_threshold=float(linf_threshold),
                minimum_observed_order=float(minimum_observed_order),
                require_observed_order=bool(require_observed_order),
                require_pass=bool(require_pass),
            ),
        )

    settings: list[ConnectionLengthRefinementRunSettings] = []
    for source in map_sources:
        normalized_source = str(source).strip().lower()
        resolved_quantity = resolve_connection_quantity(
            normalized_source,
            requested=connection_quantity,
        )
        settings.append(
            ConnectionLengthRefinementRunSettings(
                live_import=True,
                map_source=normalized_source,
                connection_quantity=resolved_quantity,
                output_root=Path(output_root),
                case_label=f"{case_label}_{normalized_source}_live",
                coil_json_path=coil_json_path,
                vmec_wout_path=vmec_wout_path,
                essos_root=essos_root,
                level_shapes=tuple(
                    tuple(int(value) for value in shape) for shape in live_level_shapes
                ),
                rho_min=float(rho_min),
                rho_max=float(rho_max),
                maxtime=float(maxtime),
                times_to_trace=int(times_to_trace),
                trace_tolerance=float(trace_tolerance),
                convergence_threshold=float(live_convergence_threshold),
                linf_threshold=float(live_linf_threshold),
                minimum_observed_order=float(live_minimum_observed_order),
                require_observed_order=bool(require_observed_order),
                require_pass=bool(require_pass),
            )
        )
    return tuple(settings)


def require_refinement_gate_passed(report_path: Path) -> None:
    """Fail the example when the generated refinement report is not promotable."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if bool(report.get("promotion_ready", report.get("passed"))):
        return
    diagnostics = report.get("diagnostics", {})
    raise RuntimeError(
        "Imported connection-length refinement gate failed: "
        f"finest_rms={report.get('finest_normalized_rms_error')!r}, "
        f"finest_linf={report.get('finest_normalized_linf_error')!r}, "
        f"minimum_observed_order={report.get('minimum_observed_order_actual')!r}, "
        f"observed_order_required={diagnostics.get('observed_order_required')!r}, "
        f"evidence_role={report.get('evidence_role')!r}, "
        f"promotion_rejection_reasons={report.get('promotion_rejection_reasons')!r}, "
        f"monotonic_rms={report.get('monotonic_rms_error_reduction')!r}, "
        f"monotonic_linf={report.get('monotonic_linf_error_reduction')!r}"
    )


def summarize_refinement_artifact(
    settings: ConnectionLengthRefinementRunSettings,
    report_path: Path,
    arrays_path: Path,
    plot_path: Path,
) -> dict[str, object]:
    """Extract the compact fields needed to compare refinement campaigns."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    diagnostics = report.get("diagnostics", {})
    return {
        "case_label": settings.case_label,
        "live_import": settings.live_import,
        "map_source": settings.map_source,
        "connection_quantity": settings.connection_quantity,
        "level_shapes": [list(shape) for shape in settings.level_shapes],
        "report_json_path": str(report_path),
        "arrays_npz_path": str(arrays_path),
        "plot_png_path": str(plot_path),
        "passed": bool(report.get("passed", False)),
        "promotion_ready": bool(report.get("promotion_ready", False)),
        "advisory_only": bool(report.get("advisory_only", False)),
        "evidence_role": str(report.get("evidence_role", "unknown")),
        "promotion_rejection_reasons": list(
            report.get("promotion_rejection_reasons", [])
        ),
        "finest_normalized_rms_error": report.get("finest_normalized_rms_error"),
        "finest_normalized_linf_error": report.get("finest_normalized_linf_error"),
        "minimum_observed_order_actual": report.get("minimum_observed_order_actual"),
        "minimum_finite_pair_fraction": report.get("minimum_finite_pair_fraction"),
        "monotonic_rms_error_reduction": bool(
            report.get("monotonic_rms_error_reduction", False)
        ),
        "monotonic_linf_error_reduction": bool(
            report.get("monotonic_linf_error_reduction", False)
        ),
        "observed_order_required": bool(diagnostics.get("observed_order_required", False)),
        "observed_order_available": bool(diagnostics.get("observed_order_available", False)),
    }


def run_refinement_campaign(settings: ConnectionLengthRefinementRunSettings) -> dict[str, object]:
    """Run one manufactured or live imported connection-length refinement gate."""

    if settings.live_import:
        artifacts = create_live_essos_imported_connection_length_refinement_package(
            output_root=settings.output_root,
            case_label=settings.case_label,
            coil_json_path=settings.coil_json_path,
            vmec_wout_path=settings.vmec_wout_path,
            essos_root=settings.essos_root,
            map_source=settings.map_source,
            connection_quantity=settings.connection_quantity,
            level_shapes=settings.level_shapes,
            rho_min=settings.rho_min,
            rho_max=settings.rho_max,
            maxtime=settings.maxtime,
            times_to_trace=settings.times_to_trace,
            trace_tolerance=settings.trace_tolerance,
            convergence_threshold=settings.convergence_threshold,
            linf_threshold=settings.linf_threshold,
            minimum_observed_order=settings.minimum_observed_order,
            require_observed_order=settings.require_observed_order,
        )
    else:
        artifacts = create_essos_imported_connection_length_refinement_package(
            output_root=settings.output_root,
            case_label=settings.case_label,
            level_shapes=settings.level_shapes,
            convergence_threshold=settings.convergence_threshold,
            linf_threshold=settings.linf_threshold,
            minimum_observed_order=settings.minimum_observed_order,
            require_observed_order=settings.require_observed_order,
        )

    print(
        "connection-length refinement: "
        f"live={settings.live_import}, "
        f"map_source={settings.map_source}, "
        f"quantity={settings.connection_quantity}, "
        f"levels={settings.level_shapes}"
    )
    print(f"wrote report: {artifacts.report_json_path}")
    print(f"wrote arrays: {artifacts.arrays_npz_path}")
    print(f"wrote plot:   {artifacts.plot_png_path}")
    entry = summarize_refinement_artifact(
        settings,
        artifacts.report_json_path,
        artifacts.arrays_npz_path,
        artifacts.plot_png_path,
    )
    print(
        "refinement evidence: "
        f"promotion_ready={entry['promotion_ready']}, "
        f"evidence_role={entry['evidence_role']}, "
        f"rms={entry['finest_normalized_rms_error']}, "
        f"linf={entry['finest_normalized_linf_error']}, "
        f"observed_order={entry['minimum_observed_order_actual']}"
    )
    if settings.require_pass:
        require_refinement_gate_passed(artifacts.report_json_path)
        print("connection-length refinement gate passed")
    return entry


def _summary_case_prefix(settings: tuple[ConnectionLengthRefinementRunSettings, ...]) -> str:
    if not settings:
        return CASE_LABEL
    label = settings[0].case_label
    for suffix in ("_coil_live", "_vmec_live", "_hybrid_live"):
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label


def write_refinement_sweep_summary(
    settings: tuple[ConnectionLengthRefinementRunSettings, ...],
    entries: list[dict[str, object]],
    *,
    summary_path: Path | None = None,
) -> Path:
    """Write one lightweight summary for a manufactured or live refinement sweep."""

    if not settings:
        raise ValueError("At least one refinement setting is required.")
    resolved_path = (
        Path(summary_path)
        if summary_path is not None
        else settings[0].output_root / "data" / f"{_summary_case_prefix(settings)}_summary.json"
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "diagnostic": "essos_imported_connection_length_refinement_sweep",
        "report_count": len(entries),
        "promotion_ready_count": sum(bool(entry["promotion_ready"]) for entry in entries),
        "advisory_count": sum(bool(entry["advisory_only"]) for entry in entries),
        "negative_control_count": sum(
            str(entry["evidence_role"]).startswith("negative_") for entry in entries
        ),
        "all_promotion_ready": all(bool(entry["promotion_ready"]) for entry in entries),
        "entries": entries,
    }
    resolved_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote sweep summary: {resolved_path}")
    return resolved_path


def run_resolved_campaigns(
    settings: tuple[ConnectionLengthRefinementRunSettings, ...],
) -> dict[str, object]:
    """Run all resolved connection-length refinement campaigns."""

    entries: list[dict[str, object]] = []
    for item in settings:
        entries.append(run_refinement_campaign(item))
    summary_path = None
    if WRITE_SWEEP_SUMMARY:
        summary_path = write_refinement_sweep_summary(settings, entries)
    return {
        "diagnostic": "essos_imported_connection_length_refinement_sweep",
        "report_count": len(entries),
        "promotion_ready_count": sum(bool(entry["promotion_ready"]) for entry in entries),
        "advisory_count": sum(bool(entry["advisory_only"]) for entry in entries),
        "negative_control_count": sum(
            str(entry["evidence_role"]).startswith("negative_") for entry in entries
        ),
        "all_promotion_ready": all(bool(entry["promotion_ready"]) for entry in entries),
        "summary_json_path": str(summary_path) if summary_path is not None else None,
        "entries": entries,
    }


if RUN_EXAMPLE:
    RESOLVED_RUN_SETTINGS = build_run_settings()
    run_resolved_campaigns(RESOLVED_RUN_SETTINGS)
