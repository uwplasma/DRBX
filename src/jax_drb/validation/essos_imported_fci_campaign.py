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
        "refinement_diagnostics": refinement_diagnostics,
        "consumed_map_diagnostics": consumed_map_diagnostics,
        "passed": bool(connection_passed and refinement_passed and consumed_map_passed),
    }


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
