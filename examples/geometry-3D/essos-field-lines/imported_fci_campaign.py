from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from jax_drb.runtime import configure_jax_runtime
from jax_drb.validation import create_essos_imported_fci_campaign_package

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the ESSOS-imported non-axisymmetric FCI SOL validation package. "
            "The command needs an ESSOS checkout unless --dry-run is used."
        )
    )
    parser.add_argument(
        "--map-source",
        choices=MAP_SOURCES,
        default="coil",
        help="Imported map semantics: coil endpoint maps, closed VMEC-coordinate maps, or hybrid VMEC maps with coil endpoint masks.",
    )
    parser.add_argument(
        "--all-map-sources",
        action="store_true",
        help="Run coil, VMEC-coordinate, and hybrid artifacts with their source-specific default output roots.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Artifact root for a single map source. Omit to use the docs/data default for that source.",
    )
    parser.add_argument(
        "--case-label",
        default=None,
        help="File stem for a single map-source artifact. Omit to use the docs/data default for that source.",
    )
    parser.add_argument("--coil-json-path", type=Path, default=None)
    parser.add_argument("--vmec-wout-path", type=Path, default=None)
    parser.add_argument("--essos-root", type=Path, default=None)
    parser.add_argument("--nx", type=int, default=5)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--rho-min", type=float, default=0.12)
    parser.add_argument("--rho-max", type=float, default=0.34)
    parser.add_argument("--times-to-trace", type=int, default=360)
    parser.add_argument("--maxtime", type=float, default=80.0)
    parser.add_argument("--trace-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved runs without importing ESSOS or writing artifacts.",
    )
    args = parser.parse_args(argv)
    if args.all_map_sources and (args.output_root is not None or args.case_label is not None):
        parser.error("--all-map-sources uses source-specific default output roots and case labels")
    return args


def build_run_settings(args: argparse.Namespace) -> tuple[ImportedFciRunSettings, ...]:
    sources = MAP_SOURCES if args.all_map_sources else (args.map_source,)
    return tuple(_settings_for_source(args, source) for source in sources)


def _settings_for_source(args: argparse.Namespace, map_source: str) -> ImportedFciRunSettings:
    return ImportedFciRunSettings(
        map_source=map_source,
        output_root=args.output_root if args.output_root is not None else DEFAULT_OUTPUT_ROOTS[map_source],
        case_label=args.case_label if args.case_label is not None else DEFAULT_CASE_LABELS[map_source],
        coil_json_path=args.coil_json_path,
        vmec_wout_path=args.vmec_wout_path,
        essos_root=args.essos_root,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        times_to_trace=args.times_to_trace,
        maxtime=args.maxtime,
        trace_tolerance=args.trace_tolerance,
        precision=args.precision,
    )


def _print_dry_run(settings: ImportedFciRunSettings) -> None:
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = build_run_settings(args)
    for item in settings:
        if args.dry_run:
            _print_dry_run(item)
        else:
            run_campaign(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
