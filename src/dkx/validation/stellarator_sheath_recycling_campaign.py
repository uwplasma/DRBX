from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np

from ..geometry import build_synthetic_stellarator_geometry
from ..native.fci_sheath_recycling import compute_fci_sheath_recycling


@dataclass(frozen=True)
class StellaratorSheathRecyclingCampaignArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_stellarator_sheath_recycling_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "stellarator_sheath_recycling_campaign",
    nx: int = 36,
    ny: int = 32,
    nz: int = 64,
) -> StellaratorSheathRecyclingCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report, arrays = build_stellarator_sheath_recycling_campaign(nx=nx, ny=ny, nz=nz)
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_stellarator_sheath_recycling_plot(report, arrays, plot_png_path)
    return StellaratorSheathRecyclingCampaignArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_stellarator_sheath_recycling_campaign(
    *,
    nx: int = 36,
    ny: int = 32,
    nz: int = 64,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    geometry = build_synthetic_stellarator_geometry(nx=nx, ny=ny, nz=nz)
    radial = np.asarray(geometry.radial, dtype=np.float64)
    theta = np.asarray(geometry.poloidal_angle, dtype=np.float64)
    phi = np.asarray(geometry.toroidal_angle, dtype=np.float64)
    connection = np.asarray(geometry.connection_length, dtype=np.float64)

    density = (
        0.28
        + 1.10 * np.exp(-((radial - 0.70) / 0.19) ** 2)
        + 0.15 * np.exp(-((radial - 0.91) / 0.06) ** 2) * (1.0 + np.cos(2.0 * theta - 5.0 * phi))
    )
    electron_temperature = (
        0.055
        + 0.18 * (1.0 - radial) ** 1.25
        + 0.018 * np.cos(theta - 5.0 * phi)
        + 0.010 * np.sin(3.0 * theta + 4.0 * phi)
    )
    ion_temperature = (
        0.050
        + 0.13 * (1.0 - radial) ** 1.10
        + 0.015 * np.sin(2.0 * theta - 5.0 * phi + 0.35)
    )
    density = np.maximum(density, 1.0e-4)
    electron_temperature = np.maximum(electron_temperature, 5.0e-3)
    ion_temperature = np.maximum(ion_temperature, 5.0e-3)

    result = compute_fci_sheath_recycling(
        density,
        electron_temperature,
        ion_temperature,
        geometry.maps,
        recycling_fraction=0.97,
        electron_sheath_transmission=5.0,
        ion_sheath_transmission=3.5,
        recycled_neutral_energy=0.025,
    )

    target_mask = np.asarray(result.masks.active, dtype=bool)
    endpoint_count = np.asarray(result.masks.endpoint_count, dtype=np.float64)
    heat_load = np.asarray(result.target_heat_load, dtype=np.float64)
    particle_loss = np.asarray(result.ion_particle_loss, dtype=np.float64)
    recycled_source = np.asarray(result.recycled_particle_source, dtype=np.float64)
    neutral_energy = np.asarray(result.recycled_neutral_energy_source, dtype=np.float64)
    current_residual = np.asarray(result.current_residual, dtype=np.float64)

    total_particle_loss = float(np.asarray(result.total_ion_particle_loss))
    total_recycled_particle_source = float(np.asarray(result.total_recycled_particle_source))
    total_heat_load = float(np.asarray(result.total_target_heat_load))
    total_recycled_neutral_energy = float(np.asarray(result.total_recycled_neutral_energy))
    particle_recycling_relative_error = float(
        abs(np.asarray(result.particle_recycling_residual)) / max(total_recycled_particle_source, 1.0e-30)
    )
    neutral_energy_relative_error = float(
        abs(np.asarray(result.neutral_energy_recycling_residual)) / max(total_recycled_neutral_energy, 1.0e-30)
    )
    current_balance_relative_error = float(
        abs(np.asarray(result.current_balance_residual)) / max(total_particle_loss, 1.0e-30)
    )
    target_fraction = float(np.mean(target_mask))
    positive_heat = heat_load[heat_load > 0.0]
    heat_load_contrast = float(np.percentile(positive_heat, 99.0) / max(np.percentile(positive_heat, 50.0), 1.0e-30))
    short_connection_mask = connection <= np.percentile(connection, 35.0)
    short_connection_heat_fraction = float(
        np.sum(heat_load[short_connection_mask]) / max(np.sum(heat_load), 1.0e-30)
    )
    heat_connection_correlation = _finite_corrcoef(
        np.ravel(heat_load[target_mask]),
        np.ravel(1.0 / np.maximum(connection[target_mask], 1.0e-12)),
    )
    endpoint_heat_fraction = float(np.sum(heat_load[target_mask]) / max(np.sum(heat_load), 1.0e-30))
    max_current_residual = float(np.max(np.abs(current_residual)))

    report: dict[str, Any] = {
        "case": "non_axisymmetric_fci_sheath_recycling",
        "geometry": geometry.metadata,
        "model": {
            "recycling_fraction": 0.97,
            "electron_sheath_transmission": 5.0,
            "ion_sheath_transmission": 3.5,
            "recycled_neutral_energy": 0.025,
            "sound_speed": "sqrt((Te + Ti) / mi)",
            "ion_particle_loss": "endpoint_count * n * sound_speed",
            "electron_particle_loss": "ion_particle_loss by zero-current reconstruction",
            "electron_heat_loss": "gamma_e * ion_particle_loss * Te",
            "ion_heat_loss": "gamma_i * ion_particle_loss * Ti",
            "recycled_particle_source": "recycling_fraction * ion_particle_loss",
        },
        "target_fraction": target_fraction,
        "total_particle_loss": total_particle_loss,
        "total_recycled_particle_source": total_recycled_particle_source,
        "total_heat_load": total_heat_load,
        "total_recycled_neutral_energy": total_recycled_neutral_energy,
        "particle_recycling_relative_error": particle_recycling_relative_error,
        "neutral_energy_relative_error": neutral_energy_relative_error,
        "current_balance_relative_error": current_balance_relative_error,
        "max_current_residual": max_current_residual,
        "heat_load_contrast": heat_load_contrast,
        "short_connection_heat_fraction": short_connection_heat_fraction,
        "heat_inverse_connection_correlation": heat_connection_correlation,
        "endpoint_heat_fraction": endpoint_heat_fraction,
    }
    report["passed"] = (
        0.002 < target_fraction < 0.35
        and total_particle_loss > 0.0
        and total_heat_load > 0.0
        and particle_recycling_relative_error < 1.0e-12
        and neutral_energy_relative_error < 1.0e-12
        and current_balance_relative_error < 1.0e-12
        and max_current_residual < 1.0e-12
        and heat_load_contrast > 1.05
        and short_connection_heat_fraction > 0.05
        and endpoint_heat_fraction > 0.999999
    )

    heat_toroidal = np.sum(heat_load, axis=0)
    loss_toroidal = np.sum(particle_loss, axis=0)
    source_toroidal = np.sum(recycled_source, axis=0)
    mask_toroidal = np.sum(endpoint_count, axis=0)
    connection_heat_weighted = np.sum(connection * heat_load, axis=0) / np.maximum(np.sum(heat_load, axis=0), 1.0e-30)
    radial_profile = np.stack(
        [
            np.mean(density, axis=(1, 2)),
            np.sum(particle_loss, axis=(1, 2)),
            np.sum(heat_load, axis=(1, 2)),
            np.sum(recycled_source, axis=(1, 2)),
        ],
        axis=1,
    )
    toroidal_profile = np.stack(
        [
            np.sum(particle_loss, axis=(0, 2)),
            np.sum(heat_load, axis=(0, 2)),
            np.sum(recycled_source, axis=(0, 2)),
        ],
        axis=1,
    )
    summary_bars = np.asarray(
        [
            total_particle_loss,
            total_recycled_particle_source,
            total_heat_load,
            total_recycled_neutral_energy,
        ],
        dtype=np.float64,
    )
    arrays = {
        "target_mask_toroidal": mask_toroidal.astype(np.float32),
        "particle_loss_toroidal": loss_toroidal.astype(np.float32),
        "heat_load_toroidal": heat_toroidal.astype(np.float32),
        "recycled_source_toroidal": source_toroidal.astype(np.float32),
        "connection_toroidal_heat_weighted": connection_heat_weighted.astype(np.float32),
        "radial_grid": np.mean(radial, axis=(1, 2)).astype(np.float32),
        "radial_profile": radial_profile.astype(np.float32),
        "toroidal_angle": np.mean(phi, axis=(0, 2)).astype(np.float32),
        "toroidal_profile": toroidal_profile.astype(np.float32),
        "summary_bars": summary_bars.astype(np.float32),
    }
    return report, arrays


def save_stellarator_sheath_recycling_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.2), constrained_layout=True)
    extent = [0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi]

    loss_image = axes[0, 0].imshow(
        arrays["particle_loss_toroidal"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    axes[0, 0].set_title("integrated Bohm particle loss")
    axes[0, 0].set_xlabel("toroidal angle")
    axes[0, 0].set_ylabel("poloidal angle")
    fig.colorbar(loss_image, ax=axes[0, 0], label="normalized particle flux")

    connection_image = axes[0, 1].imshow(
        arrays["connection_toroidal_heat_weighted"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    axes[0, 1].set_title("heat-weighted connection-length proxy")
    axes[0, 1].set_xlabel("toroidal angle")
    axes[0, 1].set_ylabel("poloidal angle")
    fig.colorbar(connection_image, ax=axes[0, 1], label="length proxy")

    heat_image = axes[0, 2].imshow(
        arrays["heat_load_toroidal"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="inferno",
    )
    axes[0, 2].set_title("integrated target heat load")
    axes[0, 2].set_xlabel("toroidal angle")
    axes[0, 2].set_ylabel("poloidal angle")
    fig.colorbar(heat_image, ax=axes[0, 2], label="normalized heat load")

    radial_grid = arrays["radial_grid"]
    radial_profile = arrays["radial_profile"]
    normalized_density = radial_profile[:, 0] / max(float(np.max(radial_profile[:, 0])), 1.0e-30)
    normalized_loss = radial_profile[:, 1] / max(float(np.max(radial_profile[:, 1])), 1.0e-30)
    normalized_heat = radial_profile[:, 2] / max(float(np.max(radial_profile[:, 2])), 1.0e-30)
    normalized_source = radial_profile[:, 3] / max(float(np.max(radial_profile[:, 3])), 1.0e-30)
    axes[1, 0].plot(radial_grid, normalized_density, color="#005f73", lw=2.2, label="mean density")
    axes[1, 0].plot(radial_grid, normalized_loss, color="#9b2226", lw=2.2, label="particle loss")
    axes[1, 0].plot(radial_grid, normalized_heat, color="#ee9b00", lw=2.2, label="heat load")
    axes[1, 0].plot(radial_grid, normalized_source, color="#0a9396", lw=2.2, label="recycled source")
    axes[1, 0].set_title("normalized radial profiles")
    axes[1, 0].set_xlabel("normalized radius")
    axes[1, 0].set_ylim(-0.05, 1.08)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False, fontsize=8)

    toroidal_angle = arrays["toroidal_angle"]
    toroidal_profile = arrays["toroidal_profile"]
    axes[1, 1].plot(toroidal_angle, toroidal_profile[:, 0], color="#9b2226", lw=2.0, label="particle loss")
    axes[1, 1].plot(toroidal_angle, toroidal_profile[:, 1], color="#ee9b00", lw=2.0, label="heat load")
    axes[1, 1].plot(toroidal_angle, toroidal_profile[:, 2], color="#0a9396", lw=2.0, label="recycled source")
    axes[1, 1].set_title("toroidal modulation of target response")
    axes[1, 1].set_xlabel("toroidal angle")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False, fontsize=8)

    summary = arrays["summary_bars"]
    labels = ["ion loss", "neutral source", "heat load", "neutral energy"]
    axes[1, 2].bar(np.arange(summary.size), summary, color=["#9b2226", "#0a9396", "#ee9b00", "#005f73"])
    axes[1, 2].set_xticks(np.arange(summary.size), labels, rotation=18, ha="right")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_title("integrated balance diagnostics")
    axes[1, 2].grid(axis="y", alpha=0.25)
    axes[1, 2].text(
        0.03,
        0.96,
        "\n".join(
            [
                f"particle balance error = {report['particle_recycling_relative_error']:.1e}",
                f"current balance error = {report['current_balance_relative_error']:.1e}",
                f"short-L heat fraction = {report['short_connection_heat_fraction']:.2f}",
            ]
        ),
        transform=axes[1, 2].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )

    fig.suptitle(
        "Non-axisymmetric FCI sheath/recycling gate: "
        "Bohm loss, zero-current closure, and exact recycled-source accounting",
        fontsize=14,
    )
    fig.savefig(resolved, dpi=190)
    plt.close(fig)
    return resolved


def _finite_corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(mask)) < 3:
        return 0.0
    left_valid = left[mask]
    right_valid = right[mask]
    if float(np.std(left_valid)) <= 0.0 or float(np.std(right_valid)) <= 0.0:
        return 0.0
    return float(np.corrcoef(left_valid, right_valid)[0, 1])
