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


def save_essos_imported_fci_campaign_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=True)
    extent = [0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi]

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
    axes[0, 1].set_title("traced endpoint count")
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
        "Imported non-axisymmetric FCI gate: external field-line maps with JAXDRB sheath and neutral closures",
        fontsize=14,
    )
    fig.savefig(resolved, dpi=190)
    plt.close(fig)
    return resolved
