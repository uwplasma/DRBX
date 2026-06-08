from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation.essos_imported_fci_campaign import (
    create_essos_imported_fci_campaign_package,
    create_essos_imported_fci_dry_run_artifact_package,
)


# SIMSOPT-style user parameters: edit these first, then run this file.
RUN_EXAMPLE = True
DRY_RUN = True
WRITE_DRY_RUN_ARTIFACTS = False

MAP_SOURCES_TO_RUN = ("coil",)
OUTPUT_ROOT: Path | None = None
CASE_LABEL: str | None = None
COIL_JSON_PATH: Path | None = None
VMEC_WOUT_PATH: Path | None = None
ESSOS_ROOT: Path | None = None

NX = 5
NY = 8
NZ = 20
RHO_MIN = 0.12
RHO_MAX = 0.34
TIMES_TO_TRACE = 360
MAXTIME = 80.0
TRACE_TOLERANCE = 1.0e-8
PRECISION = "float64"


DEFAULT_OUTPUT_ROOTS = {
    "coil": Path("docs/data/essos_imported_fci_artifacts"),
    "vmec": Path("docs/data/essos_imported_fci_vmec_artifacts"),
    "hybrid": Path("docs/data/essos_imported_fci_hybrid_artifacts"),
}
DEFAULT_CASE_LABELS = {
    "coil": "essos_imported_fci_campaign",
    "vmec": "essos_imported_fci_vmec_campaign",
    "hybrid": "essos_imported_fci_hybrid_campaign",
}
MAP_SOURCES = tuple(DEFAULT_OUTPUT_ROOTS)


@dataclass(frozen=True)
class ImportedFciRunSettings:
    """Resolved settings for one imported-field FCI validation artifact."""

    map_source: str
    output_root: Path
    case_label: str
    coil_json_path: Path | None
    vmec_wout_path: Path | None
    essos_root: Path | None
    nx: int
    ny: int
    nz: int
    rho_min: float
    rho_max: float
    times_to_trace: int
    maxtime: float
    trace_tolerance: float
    precision: str


def build_run_settings(
    *,
    map_sources: tuple[str, ...] = MAP_SOURCES_TO_RUN,
    output_root: Path | None = OUTPUT_ROOT,
    case_label: str | None = CASE_LABEL,
    coil_json_path: Path | None = COIL_JSON_PATH,
    vmec_wout_path: Path | None = VMEC_WOUT_PATH,
    essos_root: Path | None = ESSOS_ROOT,
    nx: int = NX,
    ny: int = NY,
    nz: int = NZ,
    rho_min: float = RHO_MIN,
    rho_max: float = RHO_MAX,
    times_to_trace: int = TIMES_TO_TRACE,
    maxtime: float = MAXTIME,
    trace_tolerance: float = TRACE_TOLERANCE,
    precision: str = PRECISION,
) -> tuple[ImportedFciRunSettings, ...]:
    """Resolve top-level parameters into one setting object per map source."""

    requested_sources = tuple(map_sources)
    if not requested_sources:
        raise ValueError("map_sources must contain at least one source.")
    unknown = sorted(set(requested_sources) - set(MAP_SOURCES))
    if unknown:
        raise ValueError(f"Unknown imported FCI map sources: {unknown!r}.")
    if len(requested_sources) > 1 and (output_root is not None or case_label is not None):
        raise ValueError(
            "Use source-specific default output roots and case labels when running multiple map sources."
        )
    return tuple(
        ImportedFciRunSettings(
            map_source=source,
            output_root=(
                Path(output_root)
                if output_root is not None
                else DEFAULT_OUTPUT_ROOTS[source]
            ),
            case_label=case_label if case_label is not None else DEFAULT_CASE_LABELS[source],
            coil_json_path=coil_json_path,
            vmec_wout_path=vmec_wout_path,
            essos_root=essos_root,
            nx=int(nx),
            ny=int(ny),
            nz=int(nz),
            rho_min=float(rho_min),
            rho_max=float(rho_max),
            times_to_trace=int(times_to_trace),
            maxtime=float(maxtime),
            trace_tolerance=float(trace_tolerance),
            precision=str(precision),
        )
        for source in requested_sources
    )


def print_dry_run(settings: ImportedFciRunSettings) -> None:
    print(
        "dry-run imported FCI campaign: "
        f"map_source={settings.map_source}, "
        f"output_root={settings.output_root}, "
        f"case_label={settings.case_label}, "
        f"grid=({settings.nx}, {settings.ny}, {settings.nz}), "
        f"rho=[{settings.rho_min:g}, {settings.rho_max:g}], "
        f"maxtime={settings.maxtime:g}, "
        f"times_to_trace={settings.times_to_trace}, "
        f"precision={settings.precision}"
    )


def write_dry_run_artifact(settings: ImportedFciRunSettings) -> None:
    artifacts = create_essos_imported_fci_dry_run_artifact_package(
        output_root=settings.output_root,
        case_label=settings.case_label,
        coil_json_path=settings.coil_json_path,
        vmec_wout_path=settings.vmec_wout_path,
        essos_root=settings.essos_root,
        map_source=settings.map_source,
        nx=settings.nx,
        ny=settings.ny,
        nz=settings.nz,
        rho_min=settings.rho_min,
        rho_max=settings.rho_max,
        maxtime=settings.maxtime,
        times_to_trace=settings.times_to_trace,
        trace_tolerance=settings.trace_tolerance,
        precision=settings.precision,
    )
    print(f"wrote dry-run contract: {artifacts.contract_json_path}")


def run_campaign(settings: ImportedFciRunSettings) -> None:
    configure_jax_runtime(precision=settings.precision)
    artifacts = create_essos_imported_fci_campaign_package(
        output_root=settings.output_root,
        case_label=settings.case_label,
        coil_json_path=settings.coil_json_path,
        vmec_wout_path=settings.vmec_wout_path,
        essos_root=settings.essos_root,
        map_source=settings.map_source,
        nx=settings.nx,
        ny=settings.ny,
        nz=settings.nz,
        rho_min=settings.rho_min,
        rho_max=settings.rho_max,
        maxtime=settings.maxtime,
        times_to_trace=settings.times_to_trace,
        trace_tolerance=settings.trace_tolerance,
    )

    print(f"wrote report: {artifacts.report_json_path}")
    print(f"wrote arrays: {artifacts.arrays_npz_path}")
    print(f"wrote plot: {artifacts.plot_png_path}")


def run_resolved_campaigns(
    settings: tuple[ImportedFciRunSettings, ...],
    *,
    dry_run: bool = DRY_RUN,
    dry_run_artifacts: bool = WRITE_DRY_RUN_ARTIFACTS,
) -> None:
    """Run or dry-run all resolved imported-FCI artifact settings."""

    for item in settings:
        if dry_run:
            print_dry_run(item)
            if dry_run_artifacts:
                write_dry_run_artifact(item)
        else:
            run_campaign(item)


if RUN_EXAMPLE:
    RESOLVED_RUN_SETTINGS = build_run_settings()
    run_resolved_campaigns(RESOLVED_RUN_SETTINGS)
