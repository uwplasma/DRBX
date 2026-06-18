from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np

from ..geometry import build_essos_imported_fci_geometry
from ..native.fci_neutral import compute_fci_neutral_reaction_diffusion
from ..native.fci_sheath_recycling import compute_fci_sheath_recycling


@dataclass(frozen=True)
class EssosImportedFciCampaignArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


@dataclass(frozen=True)
class EssosImportedFciDryRunArtifacts:
    contract_json_path: Path


@dataclass(frozen=True)
class EssosImportedConnectionLengthRefinementArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


@dataclass(frozen=True)
class EssosImportedConnectionLengthLevels:
    levels: tuple[np.ndarray, ...]
    labels: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...]
    coordinates: tuple[dict[str, np.ndarray], ...] = ()
    quantity: str = "raw_connection_length"


_IMPORTED_FCI_ARRAY_KEYS = (
    "major_radius_section",
    "vertical_section",
    "magnetic_field_section",
    "endpoint_count_toroidal",
    "connection_toroidal",
    "heat_load_toroidal",
    "ionisation_toroidal",
    "radial_grid",
    "radial_profiles",
    "summary",
)
_IMPORTED_FCI_REQUIRED_REPORT_FIELDS = (
    "case",
    "source",
    "map_source",
    "geometry",
    "forward_boundary_fraction",
    "backward_boundary_fraction",
    "target_fraction",
    "magnetic_field_modulation",
    "connection_length_min",
    "connection_length_mean",
    "connection_length_max",
    "connection_length_diagnostics",
    "connection_length_resolution_diagnostics",
    "endpoint_length_diagnostics",
    "refinement_diagnostics",
    "consumed_map_diagnostics",
    "map_diagnostics_passed",
    "particle_recycling_relative_error",
    "current_balance_relative_error",
    "neutral_particle_relative_error",
    "neutral_momentum_relative_error",
    "neutral_diffusion_relative_integral",
    "passed",
)
_IMPORTED_FCI_DIAGNOSTIC_SCHEMA = {
    "connection_length_diagnostics": [
        "finite_fraction",
        "nonnegative_fraction",
        "min",
        "p05",
        "median",
        "p95",
        "max",
        "mean",
        "std",
        "coefficient_of_variation",
        "zero_fraction",
        "radial_mean_profile",
    ],
    "connection_length_resolution_diagnostics": [
        "finite_face_fraction",
        "normalized_face_jump_mean",
        "normalized_face_jump_p95",
        "normalized_face_jump_max",
        "underresolved_face_fraction",
        "minimum_cells_per_connection_scale",
        "radial_normalized_jump_p95",
        "toroidal_normalized_jump_p95",
        "poloidal_normalized_jump_p95",
        "advisory_threshold",
        "passed",
    ],
    "endpoint_length_diagnostics": [
        "endpoint_cell_count",
        "target_exit_finite_endpoint_fraction",
        "target_exit_nonnegative_finite_fraction",
        "adjacent_step_finite_nonendpoint_fraction",
        "passed",
    ],
    "refinement_diagnostics": [
        "shape",
        "cell_count",
        "dphi",
        "radial_points",
        "toroidal_planes",
        "poloidal_points",
        "forward_map_coordinate_finite_fraction",
        "backward_map_coordinate_finite_fraction",
        "mean_bidirectional_abs_radial_shift_cells",
        "mean_bidirectional_abs_poloidal_shift_cells",
        "p95_bidirectional_abs_poloidal_shift_cells",
    ],
    "consumed_map_diagnostics": [
        "expected_endpoint_count_sum",
        "consumed_endpoint_count_sum",
        "endpoint_count_linf_error",
        "endpoint_count_matches_boundary_masks",
        "target_cell_fraction",
        "boundary_cell_fraction",
        "orphan_endpoint_fraction",
        "unconsumed_boundary_fraction",
        "double_endpoint_fraction",
    ],
}


def create_essos_imported_fci_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_imported_fci_campaign",
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    map_source: str = "coil",
    nx: int = 5,
    ny: int = 8,
    nz: int = 20,
    rho_min: float = 0.12,
    rho_max: float = 0.34,
    maxtime: float = 80.0,
    times_to_trace: int = 360,
    trace_tolerance: float = 1.0e-8,
) -> EssosImportedFciCampaignArtifacts:
    """Write an imported-field-line FCI validation package."""

    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report, arrays = build_essos_imported_fci_campaign(
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
        trace_tolerance=trace_tolerance,
    )
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_essos_imported_fci_campaign_plot(report, arrays, plot_png_path)
    return EssosImportedFciCampaignArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def create_essos_imported_fci_dry_run_artifact_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_imported_fci_campaign",
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    map_source: str = "coil",
    nx: int = 5,
    ny: int = 8,
    nz: int = 20,
    rho_min: float = 0.12,
    rho_max: float = 0.34,
    maxtime: float = 80.0,
    times_to_trace: int = 360,
    trace_tolerance: float = 1.0e-8,
    precision: str = "float64",
) -> EssosImportedFciDryRunArtifacts:
    """Write a self-contained dry-run contract for the imported FCI campaign."""

    root = Path(output_root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    contract = build_essos_imported_fci_dry_run_artifact_contract(
        output_root=root,
        case_label=case_label,
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
        trace_tolerance=trace_tolerance,
        precision=precision,
    )
    contract_json_path = data_dir / f"{case_label}_dry_run_contract.json"
    contract_json_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    return EssosImportedFciDryRunArtifacts(contract_json_path=contract_json_path)


def create_essos_imported_connection_length_refinement_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_imported_connection_length_refinement",
    connection_levels: tuple[np.ndarray, ...] | list[np.ndarray] | None = None,
    coordinate_levels: tuple[dict[str, np.ndarray], ...] | list[dict[str, np.ndarray]] | None = None,
    labels: tuple[str, ...] | list[str] | None = None,
    level_shapes: tuple[tuple[int, int, int], ...] = (
        (4, 6, 8),
        (8, 12, 16),
        (16, 24, 32),
    ),
    convergence_threshold: float = 0.02,
    linf_threshold: float = 0.05,
    minimum_observed_order: float = 1.5,
    require_observed_order: bool = False,
) -> EssosImportedConnectionLengthRefinementArtifacts:
    """Write a self-contained nested-grid connection-length refinement gate.

    By default this uses a deterministic manufactured non-axisymmetric
    connection-length field. Live imported coil, VMEC, or hybrid campaigns can
    pass their own nested ``connection_levels`` and reuse the same diagnostics,
    plot, and JSON schema.
    """

    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report, arrays = build_essos_imported_connection_length_refinement_campaign(
        connection_levels=connection_levels,
        coordinate_levels=coordinate_levels,
        labels=labels,
        level_shapes=level_shapes,
        convergence_threshold=convergence_threshold,
        linf_threshold=linf_threshold,
        minimum_observed_order=minimum_observed_order,
        require_observed_order=require_observed_order,
    )
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_essos_imported_connection_length_refinement_plot(report, arrays, plot_png_path)
    return EssosImportedConnectionLengthRefinementArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def create_live_essos_imported_connection_length_refinement_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_imported_connection_length_refinement_live",
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    map_source: str = "hybrid",
    connection_quantity: str = "raw_connection_length",
    level_shapes: tuple[tuple[int, int, int], ...] = (
        (3, 4, 6),
        (6, 8, 12),
    ),
    rho_min: float = 0.12,
    rho_max: float = 0.34,
    maxtime: float = 40.0,
    times_to_trace: int = 160,
    trace_tolerance: float = 1.0e-8,
    convergence_threshold: float = 0.05,
    linf_threshold: float = 0.15,
    minimum_observed_order: float = 0.0,
    require_observed_order: bool = False,
) -> EssosImportedConnectionLengthRefinementArtifacts:
    """Write a live imported connection-length refinement gate.

    This wraps the manufactured refinement diagnostics with actual imported
    coil, VMEC, or hybrid geometry levels. The default thresholds are advisory
    because two live levels verify restriction consistency but cannot establish
    a robust observed order.
    """

    live_levels = build_live_essos_imported_connection_length_levels(
        coil_json_path=coil_json_path,
        vmec_wout_path=vmec_wout_path,
        essos_root=essos_root,
        map_source=map_source,
        connection_quantity=connection_quantity,
        level_shapes=level_shapes,
        rho_min=rho_min,
        rho_max=rho_max,
        maxtime=maxtime,
        times_to_trace=times_to_trace,
        trace_tolerance=trace_tolerance,
    )
    artifacts = create_essos_imported_connection_length_refinement_package(
        output_root=output_root,
        case_label=case_label,
        connection_levels=live_levels.levels,
        coordinate_levels=live_levels.coordinates,
        labels=live_levels.labels,
        convergence_threshold=convergence_threshold,
        linf_threshold=linf_threshold,
        minimum_observed_order=minimum_observed_order,
        require_observed_order=require_observed_order,
    )
    report = json.loads(artifacts.report_json_path.read_text(encoding="utf-8"))
    report["source"] = "live imported connection-length refinement gate"
    report["map_source"] = str(map_source)
    report["connection_quantity"] = live_levels.quantity
    report["geometry_metadata"] = list(live_levels.metadata)
    report["live_imported"] = True
    artifacts.report_json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    with np.load(artifacts.arrays_npz_path) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    save_essos_imported_connection_length_refinement_plot(
        report,
        arrays,
        artifacts.plot_png_path,
    )
    return artifacts


def build_live_essos_imported_connection_length_levels(
    *,
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    map_source: str = "hybrid",
    connection_quantity: str = "raw_connection_length",
    level_shapes: tuple[tuple[int, int, int], ...] = (
        (3, 4, 6),
        (6, 8, 12),
    ),
    rho_min: float = 0.12,
    rho_max: float = 0.34,
    maxtime: float = 40.0,
    times_to_trace: int = 160,
    trace_tolerance: float = 1.0e-8,
) -> EssosImportedConnectionLengthLevels:
    """Build nested live imported connection-length arrays for refinement tests."""

    if len(level_shapes) < 2:
        raise ValueError("At least two level_shapes are required for live refinement.")
    connection_quantity = _normalize_connection_refinement_quantity(connection_quantity)
    levels: list[np.ndarray] = []
    labels: list[str] = []
    metadata: list[dict[str, Any]] = []
    coordinate_levels: list[dict[str, np.ndarray]] = []
    for shape in level_shapes:
        if len(shape) != 3:
            raise ValueError(f"Each live refinement shape must be (nx, ny, nz), got {shape!r}.")
        nx, ny, nz = (int(value) for value in shape)
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
            trace_tolerance=trace_tolerance,
        )
        connection = _connection_level_for_refinement_quantity(
            geometry,
            quantity=connection_quantity,
        )
        if connection.shape != (nx, ny, nz):
            raise ValueError(
                "Imported connection-length shape mismatch: "
                f"expected {(nx, ny, nz)}, got {connection.shape}."
            )
        levels.append(connection)
        coordinates = _connection_length_geometry_coordinates(geometry)
        if coordinates:
            coordinates = {
                key: np.asarray(value, dtype=np.float64)
                for key, value in coordinates.items()
            }
        labels.append(f"{map_source}_{nx}x{ny}x{nz}")
        level_metadata = dict(geometry.metadata)
        level_metadata["connection_refinement_quantity"] = connection_quantity
        metadata.append(level_metadata)
        if coordinates:
            coordinate_levels.append(coordinates)
    coordinate_tuple: tuple[dict[str, np.ndarray], ...]
    if len(coordinate_levels) == len(levels):
        coordinate_tuple = tuple(coordinate_levels)
    else:
        coordinate_tuple = ()
    return EssosImportedConnectionLengthLevels(
        levels=tuple(levels),
        labels=tuple(labels),
        metadata=tuple(metadata),
        coordinates=coordinate_tuple,
        quantity=connection_quantity,
    )


def build_essos_imported_connection_length_refinement_campaign(
    *,
    connection_levels: tuple[np.ndarray, ...] | list[np.ndarray] | None = None,
    coordinate_levels: tuple[dict[str, np.ndarray], ...] | list[dict[str, np.ndarray]] | None = None,
    labels: tuple[str, ...] | list[str] | None = None,
    level_shapes: tuple[tuple[int, int, int], ...] = (
        (4, 6, 8),
        (8, 12, 16),
        (16, 24, 32),
    ),
    convergence_threshold: float = 0.02,
    linf_threshold: float = 0.05,
    minimum_observed_order: float = 1.5,
    require_observed_order: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build report and arrays for nested connection-length refinement."""

    if connection_levels is None:
        levels = _manufactured_connection_length_levels(level_shapes)
        level_labels = [f"manufactured_{shape[0]}x{shape[1]}x{shape[2]}" for shape in level_shapes]
        source = "manufactured non-axisymmetric connection-length refinement gate"
        manufactured = True
    else:
        levels = [np.asarray(level, dtype=np.float64) for level in connection_levels]
        level_labels = (
            [f"level_{index}" for index in range(len(levels))]
            if labels is None
            else [str(label) for label in labels]
        )
        source = "user-supplied imported connection-length refinement gate"
        manufactured = False

    diagnostics = build_essos_imported_connection_length_refinement_diagnostics(
        levels,
        labels=level_labels,
        coordinate_levels=coordinate_levels,
        convergence_threshold=convergence_threshold,
        linf_threshold=linf_threshold,
        minimum_observed_order=minimum_observed_order,
        require_observed_order=require_observed_order,
    )
    observed_orders = [
        item["observed_order"]
        for item in diagnostics["observed_orders"]
        if item["observed_order"] is not None
    ]
    last_pair = diagnostics["pair_reports"][-1]
    rms_reduction_factors = [
        factor
        for factor in diagnostics["rms_error_reduction_factors"]
        if factor is not None
    ]
    linf_reduction_factors = [
        factor
        for factor in diagnostics["linf_error_reduction_factors"]
        if factor is not None
    ]
    report = {
        "case": "essos_imported_connection_length_refinement",
        "source": source,
        "manufactured": manufactured,
        "diagnostics": diagnostics,
        "level_shapes": [[int(value) for value in level.shape] for level in levels],
        "finest_normalized_rms_error": last_pair["normalized_rms_error"],
        "finest_normalized_linf_error": last_pair["normalized_linf_error"],
        "minimum_observed_order_actual": min(observed_orders) if observed_orders else None,
        "observed_order_required": bool(require_observed_order),
        "minimum_rms_error_reduction_factor": (
            min(rms_reduction_factors) if rms_reduction_factors else None
        ),
        "minimum_linf_error_reduction_factor": (
            min(linf_reduction_factors) if linf_reduction_factors else None
        ),
        "monotonic_rms_error_reduction": bool(
            diagnostics["monotonic_rms_error_reduction"]
        ),
        "monotonic_linf_error_reduction": bool(
            diagnostics["monotonic_linf_error_reduction"]
        ),
        "passed": bool(diagnostics["passed"]),
    }
    arrays: dict[str, np.ndarray] = {
        "summary": np.asarray(
            [
                float(last_pair["normalized_rms_error"] or np.nan),
                float(last_pair["normalized_linf_error"] or np.nan),
                float(min(observed_orders) if observed_orders else np.nan),
            ],
            dtype=np.float64,
        ),
        "pair_normalized_rms_error": np.asarray(
            [pair["normalized_rms_error"] for pair in diagnostics["pair_reports"]],
            dtype=np.float64,
        ),
        "pair_normalized_linf_error": np.asarray(
            [pair["normalized_linf_error"] for pair in diagnostics["pair_reports"]],
            dtype=np.float64,
        ),
        "observed_order": np.asarray(observed_orders, dtype=np.float64),
        "rms_error_reduction_factor": np.asarray(
            rms_reduction_factors,
            dtype=np.float64,
        ),
        "linf_error_reduction_factor": np.asarray(
            linf_reduction_factors,
            dtype=np.float64,
        ),
    }
    for index, level in enumerate(levels):
        arrays[f"level_{index}"] = np.asarray(level, dtype=np.float64)
        arrays[f"level_{index}_radial_mean"] = np.mean(level, axis=(1, 2))
        arrays[f"level_{index}_toroidal_mean"] = np.mean(level, axis=0)
    coordinate_payloads = _normalize_connection_coordinate_levels(
        coordinate_levels,
        expected_shapes=[level.shape for level in levels],
    )
    for index, coordinates in enumerate(coordinate_payloads):
        for key, values in coordinates.items():
            arrays[f"level_{index}_{key}"] = np.asarray(values, dtype=np.float64)
    return report, arrays


def _manufactured_connection_length_levels(
    level_shapes: tuple[tuple[int, int, int], ...],
) -> list[np.ndarray]:
    if len(level_shapes) < 2:
        raise ValueError("At least two manufactured connection-length levels are required.")
    return [
        _manufactured_connection_length_level(tuple(int(value) for value in shape))
        for shape in level_shapes
    ]


def _manufactured_connection_length_level(shape: tuple[int, int, int]) -> np.ndarray:
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"Connection-length level shape must contain three positive values; got {shape!r}.")
    nx, ny, nz = shape
    radial = (np.arange(nx, dtype=np.float64) + 0.5) / float(nx)
    toroidal = 2.0 * np.pi * (np.arange(ny, dtype=np.float64) + 0.5) / float(ny)
    poloidal = 2.0 * np.pi * (np.arange(nz, dtype=np.float64) + 0.5) / float(nz)
    rho, phi, theta = np.meshgrid(radial, toroidal, poloidal, indexing="ij")
    connection = (
        10.0
        + 1.3 * rho
        + 0.8 * np.sin(2.0 * np.pi * rho) * np.cos(phi - 0.3)
        + 0.45 * np.cos(5.0 * phi - 2.0 * theta)
        + 0.25 * np.sin(theta + phi)
    )
    return np.asarray(connection, dtype=np.float64)


def build_essos_imported_fci_dry_run_artifact_contract(
    *,
    output_root: str | Path,
    case_label: str = "essos_imported_fci_campaign",
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    map_source: str = "coil",
    nx: int = 5,
    ny: int = 8,
    nz: int = 20,
    rho_min: float = 0.12,
    rho_max: float = 0.34,
    maxtime: float = 80.0,
    times_to_trace: int = 360,
    trace_tolerance: float = 1.0e-8,
    precision: str = "float64",
) -> dict[str, Any]:
    """Return the ESSOS-imported FCI artifact contract without importing ESSOS."""

    map_source = _normalize_imported_fci_map_source(map_source)
    _validate_imported_fci_grid(nx=nx, ny=ny, nz=nz)
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    return {
        "case": "essos_imported_vmec_qa_fci_dry_run_contract",
        "schema_version": 1,
        "self_contained": True,
        "execution_mode": "dry_run",
        "requires_essos_runtime": False,
        "live_run_requires_essos_runtime": True,
        "map_source": map_source,
        "case_label": str(case_label),
        "output_root": str(root),
        "precision": str(precision),
        "planned_artifacts": {
            "report_json": str(data_dir / f"{case_label}.json"),
            "arrays_npz": str(data_dir / f"{case_label}.npz"),
            "plot_png": str(images_dir / f"{case_label}.png"),
            "dry_run_contract_json": str(data_dir / f"{case_label}_dry_run_contract.json"),
        },
        "grid": {
            "shape": [int(nx), int(ny), int(nz)],
            "cell_count": int(nx) * int(ny) * int(nz),
            "nx": int(nx),
            "ny": int(ny),
            "nz": int(nz),
            "rho_min": float(rho_min),
            "rho_max": float(rho_max),
        },
        "trace": {
            "maxtime": float(maxtime),
            "times_to_trace": int(times_to_trace),
            "trace_tolerance": float(trace_tolerance),
        },
        "external_inputs": {
            "coil_json_path": _path_to_optional_string(coil_json_path),
            "vmec_wout_path": _path_to_optional_string(vmec_wout_path),
            "essos_root": _path_to_optional_string(essos_root),
            "not_read_in_dry_run": True,
        },
        "map_semantics": _imported_fci_map_semantics(map_source),
        "required_report_fields": list(_IMPORTED_FCI_REQUIRED_REPORT_FIELDS),
        "required_array_keys": list(_IMPORTED_FCI_ARRAY_KEYS),
        "diagnostic_schema": _IMPORTED_FCI_DIAGNOSTIC_SCHEMA,
        "acceptance_contract": _imported_fci_acceptance_contract(map_source),
        "passed": True,
    }


def build_essos_imported_fci_campaign(
    *,
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    map_source: str = "coil",
    nx: int = 5,
    ny: int = 8,
    nz: int = 20,
    rho_min: float = 0.12,
    rho_max: float = 0.34,
    maxtime: float = 80.0,
    times_to_trace: int = 360,
    trace_tolerance: float = 1.0e-8,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run sheath/recycling and neutral gates on ESSOS-imported FCI maps."""

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
        trace_tolerance=trace_tolerance,
    )
    rho = np.asarray(geometry.minor_radius, dtype=np.float64)
    theta = np.asarray(geometry.poloidal_angle, dtype=np.float64)
    phi = np.asarray(geometry.toroidal_angle, dtype=np.float64)
    bmag = np.asarray(geometry.magnetic_field_magnitude, dtype=np.float64)
    connection = np.asarray(geometry.connection_length, dtype=np.float64)

    normalized_rho = (rho - float(rho_min)) / max(float(rho_max - rho_min), 1.0e-30)
    normalized_b = bmag / max(float(np.mean(bmag)), 1.0e-30)
    density = 0.34 + 0.90 * np.exp(-((normalized_rho - 0.40) / 0.24) ** 2) * (
        1.0 + 0.08 * np.cos(theta - 2.0 * phi)
    )
    electron_temperature = 0.060 + 0.16 * (1.0 - normalized_rho) ** 1.35 + 0.012 * np.sin(theta + phi)
    ion_temperature = 0.055 + 0.11 * (1.0 - normalized_rho) ** 1.20 + 0.010 * np.cos(2.0 * theta - phi)
    density = np.maximum(density, 1.0e-5)
    electron_temperature = np.maximum(electron_temperature / np.sqrt(normalized_b), 5.0e-3)
    ion_temperature = np.maximum(ion_temperature / normalized_b**0.25, 5.0e-3)

    sheath = compute_fci_sheath_recycling(
        density,
        electron_temperature,
        ion_temperature,
        geometry.maps,
        recycling_fraction=0.965,
        electron_sheath_transmission=5.0,
        ion_sheath_transmission=3.5,
        recycled_neutral_energy=0.026,
    )

    neutral_density = 0.16 + 0.58 * np.exp(-((normalized_rho - 0.88) / 0.16) ** 2) * (
        1.0 + 0.12 * np.cos(2.0 * theta - phi)
    )
    neutral_temperature = 0.032 + 0.018 * normalized_rho
    ion_density = density
    electron_density = density * (1.0 + 0.01 * np.sin(theta - phi))
    neutral_pressure = neutral_density * neutral_temperature
    ion_pressure = ion_density * ion_temperature
    electron_pressure = electron_density * electron_temperature
    neutral_momentum = neutral_density * (0.020 * np.sin(theta - 2.0 * phi))
    ion_momentum = ion_density * (0.035 * np.cos(theta + phi))
    neutral = compute_fci_neutral_reaction_diffusion(
        neutral_density=neutral_density,
        neutral_pressure=neutral_pressure,
        neutral_momentum=neutral_momentum,
        ion_density=ion_density,
        ion_pressure=ion_pressure,
        ion_momentum=ion_momentum,
        electron_density=electron_density,
        electron_pressure=electron_pressure,
        maps=geometry.maps,
        metric=geometry.metric,
    )

    endpoint_count = np.asarray(sheath.masks.endpoint_count, dtype=np.float64)
    target_mask = endpoint_count > 0.0
    heat_load = np.asarray(sheath.target_heat_load, dtype=np.float64)
    particle_loss = np.asarray(sheath.ion_particle_loss, dtype=np.float64)
    ionisation = np.asarray(neutral.ionisation_rate, dtype=np.float64)
    recombination = np.asarray(neutral.recombination_rate, dtype=np.float64)
    charge_exchange = np.asarray(neutral.charge_exchange_rate, dtype=np.float64)
    neutral_diffusion = np.asarray(neutral.neutral_diffusion_source, dtype=np.float64)
    jacobian = np.asarray(geometry.metric.J, dtype=np.float64)

    forward_boundary_fraction = float(np.mean(np.asarray(geometry.maps.forward_boundary, dtype=bool)))
    backward_boundary_fraction = float(np.mean(np.asarray(geometry.maps.backward_boundary, dtype=bool)))
    target_fraction = float(np.mean(target_mask))
    total_particle_loss = float(np.asarray(sheath.total_ion_particle_loss))
    total_heat_load = float(np.asarray(sheath.total_target_heat_load))
    total_ionisation = float(np.sum(ionisation))
    total_recombination = float(np.sum(recombination))
    total_charge_exchange = float(np.sum(charge_exchange))
    particle_recycling_relative_error = float(
        abs(np.asarray(sheath.particle_recycling_residual)) / max(float(np.asarray(sheath.total_recycled_particle_source)), 1.0e-30)
    )
    current_balance_relative_error = float(
        abs(np.asarray(sheath.current_balance_residual)) / max(total_particle_loss, 1.0e-30)
    )
    neutral_particle_relative_error = float(
        abs(np.asarray(neutral.total_particle_residual)) / max(total_ionisation + total_recombination, 1.0e-30)
    )
    neutral_momentum_relative_error = float(
        abs(np.asarray(neutral.total_momentum_residual))
        / max(float(np.sum(np.abs(np.asarray(neutral.neutral_momentum_source + neutral.ion_momentum_source)))) + 1.0e-30, 1.0e-30)
    )
    neutral_diffusion_integral = float(np.sum(jacobian * neutral_diffusion))
    neutral_diffusion_relative_integral = abs(neutral_diffusion_integral) / max(
        float(np.sum(np.abs(jacobian * neutral_diffusion))),
        1.0e-30,
    )
    b_modulation = float(np.max(bmag) / max(float(np.min(bmag)), 1.0e-30))
    positive_heat = heat_load[heat_load > 0.0]
    heat_load_contrast = (
        float(np.percentile(positive_heat, 95.0) / max(np.percentile(positive_heat, 50.0), 1.0e-30))
        if positive_heat.size
        else 0.0
    )
    actual_map_source = str(geometry.metadata.get("map_source", "coil"))
    map_diagnostics = build_essos_imported_fci_map_diagnostics(
        maps=geometry.maps,
        connection_length=connection,
        adjacent_step_length=np.asarray(geometry.adjacent_step_length, dtype=np.float64)
        if geometry.adjacent_step_length is not None
        else None,
        target_exit_length=np.asarray(geometry.target_exit_length, dtype=np.float64)
        if geometry.target_exit_length is not None
        else None,
        forward_target_exit_length=np.asarray(geometry.forward_target_exit_length, dtype=np.float64)
        if geometry.forward_target_exit_length is not None
        else None,
        backward_target_exit_length=np.asarray(geometry.backward_target_exit_length, dtype=np.float64)
        if geometry.backward_target_exit_length is not None
        else None,
        endpoint_count=endpoint_count,
        map_source=actual_map_source,
    )

    report: dict[str, Any] = {
        "case": "essos_imported_vmec_qa_fci_sheath_neutral_gate",
        "source": "ESSOS-imported field-line maps with jax_drb FCI closures",
        "map_source": actual_map_source,
        "geometry": geometry.metadata,
        "forward_boundary_fraction": forward_boundary_fraction,
        "backward_boundary_fraction": backward_boundary_fraction,
        "target_fraction": target_fraction,
        "magnetic_field_min": float(np.min(bmag)),
        "magnetic_field_mean": float(np.mean(bmag)),
        "magnetic_field_max": float(np.max(bmag)),
        "magnetic_field_modulation": b_modulation,
        "connection_length_min": float(np.min(connection)),
        "connection_length_mean": float(np.mean(connection)),
        "connection_length_max": float(np.max(connection)),
        "connection_length_diagnostics": map_diagnostics["connection_length_diagnostics"],
        "connection_length_resolution_diagnostics": map_diagnostics["connection_length_resolution_diagnostics"],
        "endpoint_length_diagnostics": map_diagnostics["endpoint_length_diagnostics"],
        "refinement_diagnostics": map_diagnostics["refinement_diagnostics"],
        "consumed_map_diagnostics": map_diagnostics["consumed_map_diagnostics"],
        "map_diagnostics_passed": bool(map_diagnostics["passed"]),
        "total_particle_loss": total_particle_loss,
        "total_target_heat_load": total_heat_load,
        "particle_recycling_relative_error": particle_recycling_relative_error,
        "current_balance_relative_error": current_balance_relative_error,
        "total_ionisation": total_ionisation,
        "total_recombination": total_recombination,
        "total_charge_exchange": total_charge_exchange,
        "neutral_particle_relative_error": neutral_particle_relative_error,
        "neutral_momentum_relative_error": neutral_momentum_relative_error,
        "neutral_diffusion_relative_integral": neutral_diffusion_relative_integral,
        "heat_load_contrast": heat_load_contrast,
    }
    if actual_map_source == "vmec":
        report["passed"] = (
            forward_boundary_fraction < 1.0e-12
            and backward_boundary_fraction < 1.0e-12
            and target_fraction < 1.0e-12
            and b_modulation > 1.01
            and total_ionisation > 0.0
            and total_charge_exchange > 0.0
            and particle_recycling_relative_error < 1.0e-12
            and current_balance_relative_error < 1.0e-12
            and neutral_particle_relative_error < 1.0e-12
            and neutral_momentum_relative_error < 1.0e-12
            and neutral_diffusion_relative_integral < 5.0e-2
            and bool(map_diagnostics["passed"])
        )
    else:
        report["passed"] = (
            0.05 < forward_boundary_fraction < 0.95
            and 0.05 < backward_boundary_fraction < 0.95
            and 0.05 < target_fraction <= 1.0
            and b_modulation > 1.05
            and total_particle_loss > 0.0
            and total_heat_load > 0.0
            and total_ionisation > 0.0
            and total_charge_exchange > 0.0
            and particle_recycling_relative_error < 1.0e-12
            and current_balance_relative_error < 1.0e-12
            and neutral_particle_relative_error < 1.0e-12
            and neutral_momentum_relative_error < 1.0e-12
            and neutral_diffusion_relative_integral < 5.0e-2
            and heat_load_contrast > 1.01
            and bool(map_diagnostics["passed"])
        )

    major_radius = np.sqrt(np.asarray(geometry.coordinates_x, dtype=np.float64) ** 2 + np.asarray(geometry.coordinates_y, dtype=np.float64) ** 2)
    arrays = {
        "major_radius_section": major_radius[:, 0, :].astype(np.float32),
        "vertical_section": np.asarray(geometry.coordinates_z, dtype=np.float64)[:, 0, :].astype(np.float32),
        "magnetic_field_section": bmag[:, 0, :].astype(np.float32),
        "endpoint_count_toroidal": np.sum(endpoint_count, axis=0).astype(np.float32),
        "connection_toroidal": np.mean(connection, axis=0).astype(np.float32),
        "heat_load_toroidal": np.sum(heat_load, axis=0).astype(np.float32),
        "ionisation_toroidal": np.sum(ionisation, axis=0).astype(np.float32),
        "radial_grid": np.mean(rho, axis=(1, 2)).astype(np.float32),
        "radial_profiles": np.stack(
            [
                np.mean(bmag, axis=(1, 2)),
                np.mean(connection, axis=(1, 2)),
                np.sum(particle_loss, axis=(1, 2)),
                np.sum(ionisation, axis=(1, 2)),
            ],
            axis=1,
        ).astype(np.float32),
        "summary": np.asarray(
            [
                total_particle_loss,
                total_heat_load,
                total_ionisation,
                total_recombination,
                total_charge_exchange,
                neutral_diffusion_relative_integral,
            ],
            dtype=np.float32,
        ),
    }
    return report, arrays


def build_essos_imported_fci_map_diagnostics(
    *,
    maps: Any,
    connection_length: np.ndarray,
    adjacent_step_length: np.ndarray | None = None,
    target_exit_length: np.ndarray | None = None,
    forward_target_exit_length: np.ndarray | None = None,
    backward_target_exit_length: np.ndarray | None = None,
    endpoint_count: np.ndarray,
    map_source: str,
) -> dict[str, Any]:
    """Summarize imported-map health and whether sheath masks consumed it exactly."""

    map_source = _normalize_imported_fci_map_source(map_source)
    connection = np.asarray(connection_length, dtype=np.float64)
    forward_x = np.asarray(maps.forward_x, dtype=np.float64)
    forward_z = np.asarray(maps.forward_z, dtype=np.float64)
    backward_x = np.asarray(maps.backward_x, dtype=np.float64)
    backward_z = np.asarray(maps.backward_z, dtype=np.float64)
    forward_boundary = np.asarray(maps.forward_boundary, dtype=bool)
    backward_boundary = np.asarray(maps.backward_boundary, dtype=bool)
    endpoint = np.asarray(endpoint_count, dtype=np.float64)
    shape = tuple(int(value) for value in forward_x.shape)
    expected_shape = connection.shape
    if shape != expected_shape or endpoint.shape != expected_shape:
        raise ValueError(
            "Imported FCI diagnostics require map, connection-length, and endpoint-count arrays "
            f"with the same shape; got map={shape}, connection={expected_shape}, endpoint={endpoint.shape}."
        )

    nx, ny, nz = shape
    finite_connection = np.isfinite(connection)
    finite_connection_values = connection[finite_connection]
    nonnegative_connection = finite_connection & (connection >= 0.0)
    radial_mean_profile = [
        _optional_float(np.mean(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else None
        for values in connection.reshape((nx, -1))
    ]
    connection_diagnostics = {
        "finite_fraction": float(np.mean(finite_connection)),
        "nonnegative_fraction": float(np.mean(nonnegative_connection)),
        "min": _optional_percentile(finite_connection_values, 0.0),
        "p05": _optional_percentile(finite_connection_values, 5.0),
        "median": _optional_percentile(finite_connection_values, 50.0),
        "p95": _optional_percentile(finite_connection_values, 95.0),
        "max": _optional_percentile(finite_connection_values, 100.0),
        "mean": _optional_float(np.mean(finite_connection_values)) if finite_connection_values.size else None,
        "std": _optional_float(np.std(finite_connection_values)) if finite_connection_values.size else None,
        "coefficient_of_variation": _coefficient_of_variation(finite_connection_values),
        "zero_fraction": float(np.mean(finite_connection & (np.abs(connection) <= 1.0e-14))),
        "radial_mean_profile": radial_mean_profile,
    }
    connection_resolution_diagnostics = _connection_length_resolution_diagnostics(connection)

    x_index = np.broadcast_to(np.arange(nx, dtype=np.float64)[:, None, None], shape)
    z_index = np.broadcast_to(np.arange(nz, dtype=np.float64)[None, None, :], shape)
    forward_finite = np.isfinite(forward_x) & np.isfinite(forward_z)
    backward_finite = np.isfinite(backward_x) & np.isfinite(backward_z)
    forward_valid = forward_finite & ~forward_boundary
    backward_valid = backward_finite & ~backward_boundary
    forward_dx = forward_x - x_index
    backward_dx = backward_x - x_index
    forward_dz = _periodic_cell_delta(forward_z - z_index, float(nz))
    backward_dz = _periodic_cell_delta(backward_z - z_index, float(nz))
    bidirectional_abs_dx = np.concatenate(
        [
            np.abs(forward_dx[forward_valid]).reshape(-1),
            np.abs(backward_dx[backward_valid]).reshape(-1),
        ]
    )
    bidirectional_abs_dz = np.concatenate(
        [
            np.abs(forward_dz[forward_valid]).reshape(-1),
            np.abs(backward_dz[backward_valid]).reshape(-1),
        ]
    )
    refinement_diagnostics = {
        "shape": [int(nx), int(ny), int(nz)],
        "cell_count": int(np.prod(shape)),
        "dphi": float(maps.dphi),
        "radial_points": int(nx),
        "toroidal_planes": int(ny),
        "poloidal_points": int(nz),
        "forward_map_coordinate_finite_fraction": float(np.mean(forward_finite)),
        "backward_map_coordinate_finite_fraction": float(np.mean(backward_finite)),
        "forward_nonboundary_fraction": float(np.mean(~forward_boundary)),
        "backward_nonboundary_fraction": float(np.mean(~backward_boundary)),
        "mean_bidirectional_abs_radial_shift_cells": (
            _optional_float(np.mean(bidirectional_abs_dx)) if bidirectional_abs_dx.size else None
        ),
        "max_bidirectional_abs_radial_shift_cells": (
            _optional_float(np.max(bidirectional_abs_dx)) if bidirectional_abs_dx.size else None
        ),
        "mean_bidirectional_abs_poloidal_shift_cells": (
            _optional_float(np.mean(bidirectional_abs_dz)) if bidirectional_abs_dz.size else None
        ),
        "p95_bidirectional_abs_poloidal_shift_cells": _optional_percentile(bidirectional_abs_dz, 95.0),
        "max_bidirectional_abs_poloidal_shift_cells": (
            _optional_float(np.max(bidirectional_abs_dz)) if bidirectional_abs_dz.size else None
        ),
    }

    expected_endpoint = forward_boundary.astype(np.float64) + backward_boundary.astype(np.float64)
    endpoint_error = endpoint - expected_endpoint
    endpoint_linf_error = float(np.max(np.abs(endpoint_error))) if endpoint_error.size else 0.0
    expected_endpoint_count_sum = float(np.sum(expected_endpoint))
    consumed_endpoint_count_sum = float(np.sum(endpoint))
    boundary_cells = expected_endpoint > 0.0
    target_cells = endpoint > 0.0
    consumed_map_diagnostics = {
        "expected_endpoint_count_sum": expected_endpoint_count_sum,
        "consumed_endpoint_count_sum": consumed_endpoint_count_sum,
        "endpoint_count_linf_error": endpoint_linf_error,
        "endpoint_count_matches_boundary_masks": bool(endpoint_linf_error <= 1.0e-12),
        "target_cell_fraction": float(np.mean(target_cells)),
        "boundary_cell_fraction": float(np.mean(boundary_cells)),
        "orphan_endpoint_fraction": float(np.mean(target_cells & ~boundary_cells)),
        "unconsumed_boundary_fraction": float(np.mean(boundary_cells & ~target_cells)),
        "double_endpoint_fraction": float(np.mean(endpoint >= 2.0 - 1.0e-12)),
        "forward_boundary_fraction": float(np.mean(forward_boundary)),
        "backward_boundary_fraction": float(np.mean(backward_boundary)),
    }
    endpoint_length_diagnostics = _endpoint_length_diagnostics(
        map_source=map_source,
        expected_endpoint=expected_endpoint,
        forward_boundary=forward_boundary,
        backward_boundary=backward_boundary,
        adjacent_step_length=adjacent_step_length,
        target_exit_length=target_exit_length,
        forward_target_exit_length=forward_target_exit_length,
        backward_target_exit_length=backward_target_exit_length,
    )
    connection_passed = (
        connection_diagnostics["finite_fraction"] == 1.0
        and connection_diagnostics["nonnegative_fraction"] == 1.0
    )
    refinement_passed = (
        refinement_diagnostics["forward_map_coordinate_finite_fraction"] == 1.0
        and refinement_diagnostics["backward_map_coordinate_finite_fraction"] == 1.0
        and nx >= 2
        and ny >= 2
        and nz >= 4
    )
    if map_source == "vmec":
        consumed_map_passed = (
            consumed_map_diagnostics["endpoint_count_matches_boundary_masks"]
            and expected_endpoint_count_sum <= 1.0e-12
            and consumed_endpoint_count_sum <= 1.0e-12
        )
    else:
        consumed_map_passed = (
            consumed_map_diagnostics["endpoint_count_matches_boundary_masks"]
            and expected_endpoint_count_sum > 0.0
            and consumed_endpoint_count_sum > 0.0
        )
    return {
        "map_source": map_source,
        "connection_length_diagnostics": connection_diagnostics,
        "connection_length_resolution_diagnostics": connection_resolution_diagnostics,
        "endpoint_length_diagnostics": endpoint_length_diagnostics,
        "refinement_diagnostics": refinement_diagnostics,
        "consumed_map_diagnostics": consumed_map_diagnostics,
        "passed": bool(
            connection_passed
            and refinement_passed
            and consumed_map_passed
            and endpoint_length_diagnostics["passed"]
        ),
    }


def build_essos_imported_connection_length_refinement_diagnostics(
    connection_levels: tuple[np.ndarray, ...] | list[np.ndarray],
    labels: tuple[str, ...] | list[str] | None = None,
    *,
    coordinate_levels: tuple[dict[str, np.ndarray], ...] | list[dict[str, np.ndarray]] | None = None,
    convergence_threshold: float = 0.35,
    linf_threshold: float = 0.75,
    minimum_observed_order: float = 0.5,
    require_observed_order: bool = False,
) -> dict[str, Any]:
    """Compare nested imported connection-length grids by conservative restriction.

    The single-grid face-jump diagnostic is useful for QA, but a publication
    refinement claim needs repeated imported maps. By default this helper
    restricts each fine connection-length grid to the adjacent coarse grid by
    block averages. Live imported grids may also pass logical coordinates, in
    which case the fine level is interpolated at the coarse coordinates before
    errors and observed order are computed.
    """

    levels = [np.asarray(level, dtype=np.float64) for level in connection_levels]
    if len(levels) < 2:
        raise ValueError("Connection-length refinement diagnostics require at least two levels.")
    for index, level in enumerate(levels):
        if level.ndim != 3:
            raise ValueError(
                "Connection-length refinement levels must be three-dimensional; "
                f"level {index} has shape {level.shape}."
            )

    if labels is None:
        level_labels = [f"level_{index}" for index in range(len(levels))]
    else:
        level_labels = [str(label) for label in labels]
        if len(level_labels) != len(levels):
            raise ValueError("Connection-length refinement labels must match level count.")
    coordinate_payloads = _normalize_connection_coordinate_levels(
        coordinate_levels,
        expected_shapes=[level.shape for level in levels],
    )
    use_coordinate_restriction = bool(coordinate_payloads)
    restriction_method = (
        "coordinate_interpolation" if use_coordinate_restriction else "block_average"
    )

    pair_reports: list[dict[str, Any]] = []
    for index, (coarse, fine) in enumerate(zip(levels, levels[1:])):
        if use_coordinate_restriction:
            restricted = _sample_connection_length_at_coarse_coordinates(
                fine=fine,
                fine_coordinates=coordinate_payloads[index + 1],
                coarse_coordinates=coordinate_payloads[index],
            )
        else:
            restricted = _restrict_connection_length_to_coarse_grid(
                fine=fine,
                coarse_shape=coarse.shape,
            )
        finite_mask = np.isfinite(coarse) & np.isfinite(restricted)
        diff = coarse - restricted
        finite_diff = diff[finite_mask]
        scale = _connection_length_pair_scale(coarse, restricted)
        normalized = np.abs(finite_diff) / scale if finite_diff.size else np.asarray([], dtype=np.float64)
        ratio = min(
            fine.shape[axis] / coarse.shape[axis]
            for axis in range(3)
        )
        pair_reports.append(
            {
                "coarse_label": level_labels[index],
                "fine_label": level_labels[index + 1],
                "coarse_shape": [int(value) for value in coarse.shape],
                "fine_shape": [int(value) for value in fine.shape],
                "restriction_method": restriction_method,
                "refinement_ratio_min": float(ratio),
                "finite_fraction": float(np.mean(finite_mask)),
                "absolute_rms_error": (
                    float(np.sqrt(np.mean(np.square(finite_diff))))
                    if finite_diff.size
                    else None
                ),
                "absolute_linf_error": (
                    float(np.max(np.abs(finite_diff))) if finite_diff.size else None
                ),
                "normalized_rms_error": (
                    float(np.sqrt(np.mean(np.square(normalized))))
                    if normalized.size
                    else None
                ),
                "normalized_p95_error": _optional_percentile(normalized, 95.0),
                "normalized_linf_error": (
                    float(np.max(normalized)) if normalized.size else None
                ),
            }
        )

    observed_orders: list[dict[str, Any]] = []
    for previous, current in zip(pair_reports, pair_reports[1:]):
        previous_error = previous["normalized_rms_error"]
        current_error = current["normalized_rms_error"]
        ratio = current["refinement_ratio_min"]
        if (
            previous_error is None
            or current_error is None
            or previous_error <= 0.0
            or current_error <= 0.0
            or ratio <= 1.0
        ):
            order = None
        else:
            order = float(np.log(previous_error / current_error) / np.log(ratio))
        observed_orders.append(
            {
                "coarse_pair": f"{previous['coarse_label']}->{previous['fine_label']}",
                "fine_pair": f"{current['coarse_label']}->{current['fine_label']}",
                "observed_order": order,
            }
        )
    rms_error_values = [
        float(pair["normalized_rms_error"])
        for pair in pair_reports
        if pair["normalized_rms_error"] is not None
    ]
    linf_error_values = [
        float(pair["normalized_linf_error"])
        for pair in pair_reports
        if pair["normalized_linf_error"] is not None
    ]
    rms_reduction_factors = _successive_error_reduction_factors(rms_error_values)
    linf_reduction_factors = _successive_error_reduction_factors(linf_error_values)
    monotonic_rms_reduction = _errors_decrease_monotonically(rms_error_values)
    monotonic_linf_reduction = _errors_decrease_monotonically(linf_error_values)

    last_pair = pair_reports[-1]
    last_rms = last_pair["normalized_rms_error"]
    last_linf = last_pair["normalized_linf_error"]
    finite_pairs = all(pair["finite_fraction"] == 1.0 for pair in pair_reports)
    order_values = [
        float(item["observed_order"])
        for item in observed_orders
        if item["observed_order"] is not None
    ]
    has_required_order = bool(order_values) or not bool(require_observed_order)
    order_passed = has_required_order and (
        not order_values or min(order_values) >= float(minimum_observed_order)
    )
    error_passed = (
        last_rms is not None
        and last_linf is not None
        and float(last_rms) <= float(convergence_threshold)
        and float(last_linf) <= float(linf_threshold)
    )
    monotonic_passed = bool(monotonic_rms_reduction and monotonic_linf_reduction)
    return {
        "diagnostic": "essos_imported_connection_length_refinement",
        "restriction_method": restriction_method,
        "level_count": len(levels),
        "level_labels": level_labels,
        "pair_reports": pair_reports,
        "observed_orders": observed_orders,
        "rms_error_reduction_factors": rms_reduction_factors,
        "linf_error_reduction_factors": linf_reduction_factors,
        "monotonic_rms_error_reduction": bool(monotonic_rms_reduction),
        "monotonic_linf_error_reduction": bool(monotonic_linf_reduction),
        "convergence_threshold": float(convergence_threshold),
        "linf_threshold": float(linf_threshold),
        "minimum_observed_order": float(minimum_observed_order),
        "observed_order_required": bool(require_observed_order),
        "observed_order_available": bool(order_values),
        "passed": bool(finite_pairs and error_passed and order_passed and monotonic_passed),
    }


def _errors_decrease_monotonically(errors: list[float]) -> bool:
    if len(errors) < 2:
        return True
    return all(
        np.isfinite(previous)
        and np.isfinite(current)
        and current <= previous
        for previous, current in zip(errors, errors[1:])
    )


def _successive_error_reduction_factors(errors: list[float]) -> list[float | None]:
    factors: list[float | None] = []
    for previous, current in zip(errors, errors[1:]):
        if not np.isfinite(previous) or not np.isfinite(current) or current <= 0.0:
            factors.append(None)
        else:
            factors.append(float(previous / current))
    return factors


def save_essos_imported_connection_length_refinement_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Save a publication-style plot for nested connection-length refinement."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = report["diagnostics"]
    level_count = int(diagnostics["level_count"])
    coarsest = np.asarray(arrays["level_0"], dtype=np.float64)
    finest = np.asarray(arrays[f"level_{level_count - 1}"], dtype=np.float64)
    pair_indices = np.arange(len(diagnostics["pair_reports"]), dtype=np.float64)
    pair_labels = [
        f"{_format_grid_shape(pair['coarse_shape'])}\n-> {_format_grid_shape(pair['fine_shape'])}"
        for pair in diagnostics["pair_reports"]
    ]
    rms = np.asarray(arrays["pair_normalized_rms_error"], dtype=np.float64)
    linf = np.asarray(arrays["pair_normalized_linf_error"], dtype=np.float64)
    quantity_label = _connection_quantity_plot_label(
        str(report.get("connection_quantity", "raw_connection_length"))
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    coarse_image = axes[0, 0].imshow(
        np.mean(coarsest, axis=0).T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[0, 0].set_title(f"coarse mean {quantity_label}")
    axes[0, 0].set_xlabel("toroidal index")
    axes[0, 0].set_ylabel("poloidal index")
    fig.colorbar(coarse_image, ax=axes[0, 0], label=quantity_label)

    fine_image = axes[0, 1].imshow(
        np.mean(finest, axis=0).T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[0, 1].set_title(f"finest mean {quantity_label}")
    axes[0, 1].set_xlabel("toroidal index")
    axes[0, 1].set_ylabel("poloidal index")
    fig.colorbar(fine_image, ax=axes[0, 1], label=quantity_label)

    axes[1, 0].plot(pair_indices, rms, "o-", lw=2.0, label="RMS")
    axes[1, 0].plot(pair_indices, linf, "s--", lw=2.0, label="Linf")
    axes[1, 0].axhline(
        float(diagnostics["convergence_threshold"]),
        color="0.35",
        lw=1.0,
        ls=":",
        label="RMS threshold",
    )
    axes[1, 0].set_xticks(pair_indices, pair_labels)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("restricted fine-grid error")
    axes[1, 0].set_ylabel("normalized error")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False, fontsize=8)

    for index in range(level_count):
        radial = np.asarray(arrays[f"level_{index}_radial_mean"], dtype=np.float64)
        x = np.linspace(0.0, 1.0, radial.size)
        axes[1, 1].plot(x, radial, lw=1.8, label=diagnostics["level_labels"][index])
    axes[1, 1].set_title(f"radial mean {quantity_label}")
    axes[1, 1].set_xlabel("normalized radius")
    axes[1, 1].set_ylabel(quantity_label)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False, fontsize=7)

    order = report.get("minimum_observed_order_actual")
    order_text = "n/a" if order is None else f"{float(order):.2f}"
    fig.suptitle(
        f"Imported-field {quantity_label} refinement gate: "
        f"passed={report['passed']}, "
        f"finest RMS={report['finest_normalized_rms_error']:.2e}, "
        f"observed order={order_text}",
        fontsize=13,
    )
    fig.savefig(resolved, dpi=180)
    plt.close(fig)
    return resolved


def _connection_quantity_plot_label(quantity: str) -> str:
    normalized = _normalize_connection_refinement_quantity(quantity)
    if normalized == "parallel_step_per_toroidal_radian":
        return "parallel step length per radian"
    if normalized == "adjacent_step_length":
        return "adjacent step length"
    if normalized == "target_exit_length":
        return "target-exit length"
    return "connection length"


def _format_grid_shape(shape: list[int] | tuple[int, ...]) -> str:
    return "x".join(str(int(value)) for value in shape)


def save_essos_imported_fci_campaign_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=True)
    extent = [0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi]
    map_source = str(report.get("map_source", report.get("geometry", {}).get("map_source", "coil")))
    map_label = {
        "coil": "coil-traced open-field map",
        "vmec": "VMEC-coordinate closed-field map",
        "hybrid": "hybrid VMEC-coordinate map with coil endpoint masks",
    }.get(map_source, f"{map_source} map")

    section = axes[0, 0].tricontourf(
        arrays["major_radius_section"].ravel(),
        arrays["vertical_section"].ravel(),
        arrays["magnetic_field_section"].ravel(),
        levels=18,
        cmap="cividis",
    )
    axes[0, 0].scatter(
        arrays["major_radius_section"].ravel(),
        arrays["vertical_section"].ravel(),
        s=7,
        c="white",
        alpha=0.55,
        linewidths=0.0,
    )
    axes[0, 0].set_title("imported VMEC QA shell")
    axes[0, 0].set_xlabel("major radius")
    axes[0, 0].set_ylabel("vertical coordinate")
    axes[0, 0].set_aspect("equal", adjustable="box")
    fig.colorbar(section, ax=axes[0, 0], label="|B|")

    endpoint = axes[0, 1].imshow(
        arrays["endpoint_count_toroidal"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    axes[0, 1].set_title("endpoint count from map masks")
    axes[0, 1].set_xlabel("toroidal angle")
    axes[0, 1].set_ylabel("poloidal angle")
    fig.colorbar(endpoint, ax=axes[0, 1], label="open endpoints")

    connection = axes[0, 2].imshow(
        arrays["connection_toroidal"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    axes[0, 2].set_title("mean connection-length proxy")
    axes[0, 2].set_xlabel("toroidal angle")
    axes[0, 2].set_ylabel("poloidal angle")
    fig.colorbar(connection, ax=axes[0, 2], label="arc length")

    heat = axes[1, 0].imshow(
        arrays["heat_load_toroidal"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="inferno",
    )
    axes[1, 0].set_title("sheath heat-load response")
    axes[1, 0].set_xlabel("toroidal angle")
    axes[1, 0].set_ylabel("poloidal angle")
    fig.colorbar(heat, ax=axes[1, 0], label="normalized heat load")

    ionisation = axes[1, 1].imshow(
        arrays["ionisation_toroidal"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="plasma",
    )
    axes[1, 1].set_title("neutral ionisation response")
    axes[1, 1].set_xlabel("toroidal angle")
    axes[1, 1].set_ylabel("poloidal angle")
    fig.colorbar(ionisation, ax=axes[1, 1], label="normalized source")

    radial_grid = arrays["radial_grid"]
    radial_profiles = arrays["radial_profiles"]
    labels = ["|B|", "connection", "particle loss", "ionisation"]
    colors = ["#005f73", "#9b2226", "#ee9b00", "#0a9396"]
    for index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        profile = radial_profiles[:, index]
        axes[1, 2].plot(
            radial_grid,
            profile / max(float(np.max(np.abs(profile))), 1.0e-30),
            lw=2.1,
            color=color,
            label=label,
        )
    axes[1, 2].set_title("normalized radial diagnostics")
    axes[1, 2].set_xlabel("minor radius")
    axes[1, 2].set_ylim(-0.05, 1.08)
    axes[1, 2].grid(alpha=0.25)
    axes[1, 2].legend(frameon=False, fontsize=8)
    axes[1, 2].text(
        0.03,
        0.96,
        "\n".join(
            [
                f"target fraction = {report['target_fraction']:.2f}",
                f"|B|max/min = {report['magnetic_field_modulation']:.2f}",
                f"neutral balance = {report['neutral_particle_relative_error']:.1e}",
            ]
        ),
        transform=axes[1, 2].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "0.8"},
    )
    fig.suptitle(
        f"Imported non-axisymmetric FCI gate ({map_label}): JAXDRB sheath and neutral closures",
        fontsize=14,
    )
    fig.savefig(resolved, dpi=190)
    plt.close(fig)
    return resolved


def _normalize_imported_fci_map_source(map_source: str) -> str:
    normalized = str(map_source).strip().lower().replace("-", "_")
    aliases = {
        "essos": "coil",
        "essos_coil": "coil",
        "coil_map": "coil",
        "vmec_map": "vmec",
        "hybrid_map": "hybrid",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"coil", "vmec", "hybrid"}:
        raise ValueError("map_source must be one of 'coil', 'vmec', or 'hybrid'")
    return normalized


def _validate_imported_fci_grid(*, nx: int, ny: int, nz: int) -> None:
    if int(nx) < 2 or int(ny) < 2 or int(nz) < 4:
        raise ValueError("ESSOS imported FCI dry-run contract requires nx >= 2, ny >= 2, and nz >= 4")


def _path_to_optional_string(path: str | Path | None) -> str | None:
    return None if path is None else str(Path(path))


def _imported_fci_map_semantics(map_source: str) -> dict[str, Any]:
    descriptions = {
        "coil": {
            "map_coordinates": "ESSOS Biot-Savart coil-traced adjacent-plane endpoints",
            "endpoint_masks": "ESSOS coil-trace exits and radial-edge hits",
            "connection_length": "coil-trace exit length when available, otherwise adjacent-plane arc length",
            "expected_target_behavior": "open-field sheath/recycling endpoints are present",
        },
        "vmec": {
            "map_coordinates": "VMEC-coordinate RK4 adjacent-plane map at fixed flux surface",
            "endpoint_masks": "closed-field control with endpoint masks disabled",
            "connection_length": "bidirectional VMEC-coordinate adjacent-plane arc length",
            "expected_target_behavior": "zero sheath target endpoints",
        },
        "hybrid": {
            "map_coordinates": "VMEC-coordinate RK4 adjacent-plane map at fixed flux surface",
            "endpoint_masks": "ESSOS coil-trace exits and radial-edge hits",
            "connection_length": "coil-trace exit length when available, otherwise adjacent-plane arc length",
            "expected_target_behavior": "open-field sheath/recycling endpoints consumed on VMEC map coordinates",
        },
    }
    return descriptions[_normalize_imported_fci_map_source(map_source)]


def _imported_fci_acceptance_contract(map_source: str) -> dict[str, list[str]]:
    map_source = _normalize_imported_fci_map_source(map_source)
    source_specific = {
        "coil": [
            "forward and backward boundary masks are both nonzero",
            "sheath endpoint counts exactly match forward plus backward map boundary masks",
            "target heat load and particle loss are positive",
        ],
        "vmec": [
            "forward and backward boundary masks are zero",
            "sheath endpoint counts are zero",
            "neutral and metric-diffusion diagnostics remain finite on the closed map",
        ],
        "hybrid": [
            "VMEC-coordinate maps remain finite while coil-derived endpoint masks are nonzero",
            "sheath endpoint counts exactly match forward plus backward coil boundary masks",
            "target heat load and particle loss are positive on the consumed hybrid map",
        ],
    }
    return {
        "common": [
            "connection-length diagnostics are finite and nonnegative",
            (
                "single-grid connection-length resolution diagnostics report "
                "grid-scale roughness as an advisory pre-refinement check"
            ),
            "map coordinate diagnostics are finite at the declared grid refinement",
            "particle recycling, current balance, neutral particle, and neutral momentum residuals close",
            "report JSON contains every required diagnostic field and arrays NPZ contains every required key",
        ],
        "source_specific": source_specific[map_source],
    }


def _periodic_cell_delta(delta: np.ndarray, period: float) -> np.ndarray:
    if period <= 0.0:
        return delta
    return np.mod(delta + 0.5 * period, period) - 0.5 * period


def _endpoint_length_diagnostics(
    *,
    map_source: str,
    expected_endpoint: np.ndarray,
    forward_boundary: np.ndarray,
    backward_boundary: np.ndarray,
    adjacent_step_length: np.ndarray | None,
    target_exit_length: np.ndarray | None,
    forward_target_exit_length: np.ndarray | None,
    backward_target_exit_length: np.ndarray | None,
) -> dict[str, Any]:
    """Summarize wall-hit and adjacent-step lengths separately."""

    endpoint_mask = np.asarray(expected_endpoint, dtype=np.float64) > 0.0
    nonendpoint_mask = ~endpoint_mask
    endpoint_count = int(np.sum(endpoint_mask))
    nonendpoint_count = int(np.sum(nonendpoint_mask))
    target_exit = _optional_array_like(target_exit_length, expected_endpoint.shape)
    forward_exit = _optional_array_like(forward_target_exit_length, expected_endpoint.shape)
    backward_exit = _optional_array_like(backward_target_exit_length, expected_endpoint.shape)
    adjacent = _optional_array_like(adjacent_step_length, expected_endpoint.shape)

    target_finite = np.isfinite(target_exit)
    target_values = target_exit[target_finite]
    forward_finite = np.isfinite(forward_exit)
    backward_finite = np.isfinite(backward_exit)
    adjacent_finite = np.isfinite(adjacent)
    target_nonnegative = target_finite & (target_exit >= 0.0)
    adjacent_nonnegative = adjacent_finite & (adjacent >= 0.0)
    source = _normalize_imported_fci_map_source(map_source)
    if source == "vmec":
        passed = bool(endpoint_count == 0 and _fraction(adjacent_finite, nonendpoint_mask) == 1.0)
    else:
        passed = bool(
            endpoint_count > 0
            and _fraction(target_finite, endpoint_mask) > 0.0
            and _fraction(target_nonnegative, target_finite) == 1.0
            and _fraction(adjacent_nonnegative, adjacent_finite) == 1.0
        )
    return {
        "endpoint_cell_count": endpoint_count,
        "nonendpoint_cell_count": nonendpoint_count,
        "target_exit_finite_fraction": float(np.mean(target_finite)),
        "target_exit_finite_endpoint_fraction": _fraction(target_finite, endpoint_mask),
        "target_exit_finite_nonendpoint_fraction": _fraction(target_finite, nonendpoint_mask),
        "target_exit_nonnegative_finite_fraction": _fraction(target_nonnegative, target_finite),
        "target_exit_min": _optional_percentile(target_values, 0.0),
        "target_exit_median": _optional_percentile(target_values, 50.0),
        "target_exit_max": _optional_percentile(target_values, 100.0),
        "forward_exit_finite_forward_boundary_fraction": _fraction(
            forward_finite,
            np.asarray(forward_boundary, dtype=bool),
        ),
        "backward_exit_finite_backward_boundary_fraction": _fraction(
            backward_finite,
            np.asarray(backward_boundary, dtype=bool),
        ),
        "adjacent_step_finite_fraction": float(np.mean(adjacent_finite)),
        "adjacent_step_finite_nonendpoint_fraction": _fraction(adjacent_finite, nonendpoint_mask),
        "adjacent_step_nonnegative_finite_fraction": _fraction(
            adjacent_nonnegative,
            adjacent_finite,
        ),
        "passed": passed,
    }


def _optional_array_like(values: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if values is None:
        return np.full(shape, np.nan, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(
            f"Endpoint-length diagnostic array shape mismatch: expected {shape}, got {array.shape}."
        )
    return array


def _fraction(mask: np.ndarray, where: np.ndarray) -> float | None:
    where_bool = np.asarray(where, dtype=bool)
    if not np.any(where_bool):
        return None
    return float(np.mean(np.asarray(mask, dtype=bool)[where_bool]))


def _connection_length_resolution_diagnostics(
    connection: np.ndarray,
    *,
    advisory_threshold: float = 0.5,
) -> dict[str, Any]:
    """Estimate whether a single imported connection-length grid is resolved.

    This is intentionally an advisory single-grid diagnostic. A production
    refinement claim still needs repeated imported runs at multiple grids.
    """

    values = np.asarray(connection, dtype=np.float64)
    radial_jumps = _normalized_connection_neighbor_jumps(values, axis=0, periodic=False)
    toroidal_jumps = _normalized_connection_neighbor_jumps(values, axis=1, periodic=True)
    poloidal_jumps = _normalized_connection_neighbor_jumps(values, axis=2, periodic=True)
    all_jumps = np.concatenate([radial_jumps, toroidal_jumps, poloidal_jumps])
    finite_jumps = all_jumps[np.isfinite(all_jumps)]
    threshold = float(advisory_threshold)
    if finite_jumps.size == 0:
        return {
            "finite_face_fraction": 0.0,
            "normalized_face_jump_mean": None,
            "normalized_face_jump_p95": None,
            "normalized_face_jump_max": None,
            "underresolved_face_fraction": 1.0,
            "minimum_cells_per_connection_scale": None,
            "radial_normalized_jump_p95": _optional_percentile(radial_jumps, 95.0),
            "toroidal_normalized_jump_p95": _optional_percentile(toroidal_jumps, 95.0),
            "poloidal_normalized_jump_p95": _optional_percentile(poloidal_jumps, 95.0),
            "advisory_threshold": threshold,
            "passed": False,
        }

    p95 = float(np.percentile(finite_jumps, 95.0))
    finite_face_fraction = float(finite_jumps.size / max(all_jumps.size, 1))
    return {
        "finite_face_fraction": finite_face_fraction,
        "normalized_face_jump_mean": float(np.mean(finite_jumps)),
        "normalized_face_jump_p95": p95,
        "normalized_face_jump_max": float(np.max(finite_jumps)),
        "underresolved_face_fraction": float(np.mean(finite_jumps > threshold)),
        "minimum_cells_per_connection_scale": float(1.0 / max(p95, 1.0e-30)),
        "radial_normalized_jump_p95": _optional_percentile(radial_jumps, 95.0),
        "toroidal_normalized_jump_p95": _optional_percentile(toroidal_jumps, 95.0),
        "poloidal_normalized_jump_p95": _optional_percentile(poloidal_jumps, 95.0),
        "advisory_threshold": threshold,
        "passed": bool(finite_face_fraction == 1.0 and p95 <= threshold),
    }


def _connection_length_geometry_coordinates(geometry: Any) -> dict[str, np.ndarray]:
    coordinates: dict[str, np.ndarray] = {}
    for key, attribute in (
        ("minor_radius", "minor_radius"),
        ("toroidal_angle", "toroidal_angle"),
        ("poloidal_angle", "poloidal_angle"),
    ):
        if hasattr(geometry, attribute):
            coordinates[key] = np.asarray(getattr(geometry, attribute), dtype=np.float64)
    return coordinates if len(coordinates) == 3 else {}


def _normalize_connection_refinement_quantity(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "raw": "raw_connection_length",
        "connection_length": "raw_connection_length",
        "raw_connection_length": "raw_connection_length",
        "adjacent_step": "adjacent_step_length",
        "adjacent_step_length": "adjacent_step_length",
        "adjacent_plane_step": "adjacent_step_length",
        "adjacent_plane_step_length": "adjacent_step_length",
        "target_exit": "target_exit_length",
        "target_exit_length": "target_exit_length",
        "exit_length": "target_exit_length",
        "wall_hit_length": "target_exit_length",
        "per_radian": "parallel_step_per_toroidal_radian",
        "per_toroidal_radian": "parallel_step_per_toroidal_radian",
        "parallel_step_per_radian": "parallel_step_per_toroidal_radian",
        "parallel_step_per_toroidal_radian": "parallel_step_per_toroidal_radian",
    }
    if normalized not in aliases:
        raise ValueError(
            "connection_quantity must be 'raw_connection_length', "
            "'adjacent_step_length', 'target_exit_length', or "
            "'parallel_step_per_toroidal_radian'."
        )
    return aliases[normalized]


def _connection_level_for_refinement_quantity(
    geometry: Any,
    *,
    quantity: str,
) -> np.ndarray:
    connection = np.asarray(geometry.connection_length, dtype=np.float64)
    if quantity == "raw_connection_length":
        return connection
    if quantity == "adjacent_step_length":
        if not hasattr(geometry, "adjacent_step_length") or geometry.adjacent_step_length is None:
            raise ValueError("adjacent_step_length requires geometry.adjacent_step_length.")
        return np.asarray(geometry.adjacent_step_length, dtype=np.float64)
    if quantity == "target_exit_length":
        if not hasattr(geometry, "target_exit_length") or geometry.target_exit_length is None:
            raise ValueError("target_exit_length requires geometry.target_exit_length.")
        return np.asarray(geometry.target_exit_length, dtype=np.float64)
    if quantity == "parallel_step_per_toroidal_radian":
        if not hasattr(geometry, "maps") or not hasattr(geometry.maps, "dphi"):
            raise ValueError(
                "parallel_step_per_toroidal_radian requires geometry.maps.dphi."
            )
        dphi = abs(float(geometry.maps.dphi))
        if dphi <= 0.0:
            raise ValueError("geometry.maps.dphi must be positive for refinement.")
        if hasattr(geometry, "adjacent_step_length") and geometry.adjacent_step_length is not None:
            connection = np.asarray(geometry.adjacent_step_length, dtype=np.float64)
        return connection / dphi
    raise ValueError(f"Unsupported connection refinement quantity {quantity!r}.")


def _normalize_connection_coordinate_levels(
    coordinate_levels: tuple[dict[str, np.ndarray], ...] | list[dict[str, np.ndarray]] | None,
    *,
    expected_shapes: list[tuple[int, ...]],
) -> tuple[dict[str, np.ndarray], ...]:
    if coordinate_levels is None:
        return ()
    if len(coordinate_levels) != len(expected_shapes):
        raise ValueError("Coordinate level count must match connection level count.")
    normalized: list[dict[str, np.ndarray]] = []
    for index, (coordinates, expected_shape) in enumerate(zip(coordinate_levels, expected_shapes)):
        payload: dict[str, np.ndarray] = {}
        for key in ("minor_radius", "toroidal_angle", "poloidal_angle"):
            if key not in coordinates:
                raise ValueError(f"Coordinate level {index} is missing {key!r}.")
            values = np.asarray(coordinates[key], dtype=np.float64)
            if values.shape != expected_shape:
                raise ValueError(
                    f"Coordinate level {index} key {key!r} has shape {values.shape}; "
                    f"expected {expected_shape}."
                )
            payload[key] = values
        normalized.append(payload)
    return tuple(normalized)


def _sample_connection_length_at_coarse_coordinates(
    *,
    fine: np.ndarray,
    fine_coordinates: dict[str, np.ndarray],
    coarse_coordinates: dict[str, np.ndarray],
) -> np.ndarray:
    fine = np.asarray(fine, dtype=np.float64)
    fine_rho = _coordinate_axis(fine_coordinates["minor_radius"], axis=0)
    fine_phi = _coordinate_axis(fine_coordinates["toroidal_angle"], axis=1)
    fine_theta = _coordinate_axis(fine_coordinates["poloidal_angle"], axis=2)
    coarse_rho = np.asarray(coarse_coordinates["minor_radius"], dtype=np.float64)
    coarse_phi = np.asarray(coarse_coordinates["toroidal_angle"], dtype=np.float64)
    coarse_theta = np.asarray(coarse_coordinates["poloidal_angle"], dtype=np.float64)

    x0, x1, wx = _linear_axis_indices(fine_rho, coarse_rho)
    y0, y1, wy = _periodic_axis_indices(fine_phi, coarse_phi, period=2.0 * np.pi)
    z0, z1, wz = _periodic_axis_indices(fine_theta, coarse_theta, period=2.0 * np.pi)

    c000 = fine[x0, y0, z0]
    c100 = fine[x1, y0, z0]
    c010 = fine[x0, y1, z0]
    c110 = fine[x1, y1, z0]
    c001 = fine[x0, y0, z1]
    c101 = fine[x1, y0, z1]
    c011 = fine[x0, y1, z1]
    c111 = fine[x1, y1, z1]
    c00 = (1.0 - wx) * c000 + wx * c100
    c10 = (1.0 - wx) * c010 + wx * c110
    c01 = (1.0 - wx) * c001 + wx * c101
    c11 = (1.0 - wx) * c011 + wx * c111
    c0 = (1.0 - wy) * c00 + wy * c10
    c1 = (1.0 - wy) * c01 + wy * c11
    return (1.0 - wz) * c0 + wz * c1


def _coordinate_axis(values: np.ndarray, *, axis: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.ndim != 3:
        raise ValueError(f"Coordinate arrays must be 1D or 3D, got shape {array.shape}.")
    if axis == 0:
        return np.asarray(array[:, 0, 0], dtype=np.float64)
    if axis == 1:
        return np.asarray(array[0, :, 0], dtype=np.float64)
    if axis == 2:
        return np.asarray(array[0, 0, :], dtype=np.float64)
    raise ValueError(f"Unsupported coordinate axis {axis}.")


def _linear_axis_indices(axis_values: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis_values = np.asarray(axis_values, dtype=np.float64)
    if axis_values.ndim != 1 or axis_values.size == 0:
        raise ValueError("Linear interpolation axis must be a non-empty 1D array.")
    if axis_values.size == 1:
        zeros = np.zeros_like(targets, dtype=int)
        weights = np.zeros_like(targets, dtype=np.float64)
        return zeros, zeros, weights
    if np.any(np.diff(axis_values) <= 0.0):
        raise ValueError("Linear interpolation axis must be strictly increasing.")
    coordinate = np.interp(
        np.asarray(targets, dtype=np.float64),
        axis_values,
        np.arange(axis_values.size, dtype=np.float64),
        left=np.nan,
        right=np.nan,
    )
    x0 = np.floor(coordinate).astype(int)
    x0 = np.clip(x0, 0, axis_values.size - 1)
    x1 = np.clip(x0 + 1, 0, axis_values.size - 1)
    weights = np.where(np.isfinite(coordinate), coordinate - x0.astype(np.float64), np.nan)
    weights = np.where(x0 == x1, 0.0, weights)
    return x0, x1, weights


def _periodic_axis_indices(
    axis_values: np.ndarray,
    targets: np.ndarray,
    *,
    period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis_values = np.asarray(axis_values, dtype=np.float64)
    if axis_values.ndim != 1 or axis_values.size == 0:
        raise ValueError("Periodic interpolation axis must be a non-empty 1D array.")
    if axis_values.size == 1:
        zeros = np.zeros_like(targets, dtype=int)
        weights = np.zeros_like(targets, dtype=np.float64)
        return zeros, zeros, weights
    spacing = float(period) / float(axis_values.size)
    coordinate = np.mod(np.asarray(targets, dtype=np.float64) - axis_values[0], float(period))
    coordinate = coordinate / max(spacing, 1.0e-30)
    x0 = np.floor(coordinate).astype(int) % axis_values.size
    x1 = (x0 + 1) % axis_values.size
    weights = coordinate - np.floor(coordinate)
    return x0, x1, np.asarray(weights, dtype=np.float64)


def _restrict_connection_length_to_coarse_grid(
    *,
    fine: np.ndarray,
    coarse_shape: tuple[int, int, int],
) -> np.ndarray:
    if len(coarse_shape) != 3:
        raise ValueError(f"Coarse connection-length shape must be 3D; got {coarse_shape}.")
    fine_shape = tuple(int(value) for value in fine.shape)
    ratios = []
    for coarse_size, fine_size in zip(coarse_shape, fine_shape):
        if coarse_size <= 0 or fine_size <= 0 or fine_size % coarse_size != 0:
            raise ValueError(
                "Connection-length refinement levels must be nested by integer "
                f"ratios; coarse={coarse_shape}, fine={fine_shape}."
            )
        ratios.append(fine_size // coarse_size)
    nx, ny, nz = coarse_shape
    rx, ry, rz = ratios
    return fine.reshape(nx, rx, ny, ry, nz, rz).mean(axis=(1, 3, 5))


def _connection_length_pair_scale(coarse: np.ndarray, restricted: np.ndarray) -> float:
    values = np.concatenate(
        [
            np.asarray(coarse, dtype=np.float64).reshape(-1),
            np.asarray(restricted, dtype=np.float64).reshape(-1),
        ]
    )
    finite_values = np.abs(values[np.isfinite(values)])
    if finite_values.size == 0:
        return 1.0
    return max(float(np.median(finite_values)), 1.0e-30)


def _normalized_connection_neighbor_jumps(values: np.ndarray, *, axis: int, periodic: bool) -> np.ndarray:
    if values.shape[axis] < 2:
        return np.asarray([], dtype=np.float64)
    if periodic:
        left = values
        right = np.roll(values, -1, axis=axis)
    else:
        left = np.take(values, indices=range(values.shape[axis] - 1), axis=axis)
        right = np.take(values, indices=range(1, values.shape[axis]), axis=axis)
    scale = 0.5 * (np.abs(left) + np.abs(right))
    floor = _connection_length_scale_floor(values)
    jumps = np.abs(right - left) / np.maximum(scale, floor)
    return jumps.reshape(-1)


def _connection_length_scale_floor(values: np.ndarray) -> float:
    finite_values = np.abs(np.asarray(values, dtype=np.float64))
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 1.0e-30
    return max(float(np.median(finite_values)) * 1.0e-12, 1.0e-30)


def _optional_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return None
    return float(np.percentile(finite_values, percentile))


def _optional_float(value: float | np.floating[Any]) -> float:
    return float(value)


def _coefficient_of_variation(values: np.ndarray) -> float | None:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return None
    mean = float(np.mean(finite_values))
    if abs(mean) <= 1.0e-30:
        return None
    return float(np.std(finite_values) / abs(mean))
