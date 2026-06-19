from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from jax_drb.validation import (
    create_essos_imported_connection_length_refinement_package,
    create_live_essos_imported_connection_length_refinement_package,
)


# SIMSOPT-style user parameters: edit these first, then run this file.
RUN_EXAMPLE = True

# The default path is self-contained and does not require external coil or VMEC
# data. Set LIVE_IMPORT = True to regenerate live imported-field promotion
# reports from an ESSOS/VMEC checkout.
LIVE_IMPORT = False
MAP_SOURCES_TO_RUN = ("hybrid",)  # "coil", "vmec", and/or "hybrid" in live mode.
CONNECTION_QUANTITY = "auto"

OUTPUT_ROOT = Path("docs/data/essos_imported_connection_length_refinement_artifacts")
CASE_LABEL = "essos_imported_connection_length_refinement"
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


def run_refinement_campaign(settings: ConnectionLengthRefinementRunSettings) -> None:
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
    if settings.require_pass:
        require_refinement_gate_passed(artifacts.report_json_path)
        print("connection-length refinement gate passed")


def run_resolved_campaigns(
    settings: tuple[ConnectionLengthRefinementRunSettings, ...],
) -> None:
    """Run all resolved connection-length refinement campaigns."""

    for item in settings:
        run_refinement_campaign(item)


if RUN_EXAMPLE:
    RESOLVED_RUN_SETTINGS = build_run_settings()
    run_resolved_campaigns(RESOLVED_RUN_SETTINGS)
