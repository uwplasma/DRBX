from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np

from ..geometry import build_synthetic_stellarator_geometry
from ..native.fci_neutral import compute_fci_neutral_reaction_diffusion


@dataclass(frozen=True)
class StellaratorNeutralPhysicsCampaignArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_stellarator_neutral_physics_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "stellarator_neutral_physics_campaign",
    nx: int = 34,
    ny: int = 30,
    nz: int = 60,
) -> StellaratorNeutralPhysicsCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    report, arrays = build_stellarator_neutral_physics_campaign(nx=nx, ny=ny, nz=nz)
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_stellarator_neutral_physics_plot(report, arrays, plot_png_path)
    return StellaratorNeutralPhysicsCampaignArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_stellarator_neutral_physics_campaign(
    *,
    nx: int = 34,
    ny: int = 30,
    nz: int = 60,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    geometry = build_synthetic_stellarator_geometry(nx=nx, ny=ny, nz=nz)
    radial = np.asarray(geometry.radial, dtype=np.float64)
    theta = np.asarray(geometry.poloidal_angle, dtype=np.float64)
    phi = np.asarray(geometry.toroidal_angle, dtype=np.float64)

    neutral_density = 0.12 + 0.62 * np.exp(-((radial - 0.93) / 0.08) ** 2) * (
        1.0 + 0.18 * np.cos(2.0 * theta - 5.0 * phi)
    )
    ion_density = 0.35 + 1.0 * np.exp(-((radial - 0.68) / 0.20) ** 2)
    electron_density = ion_density * (1.0 + 0.015 * np.cos(theta - 5.0 * phi))
    neutral_temperature = 0.035 + 0.020 * radial
    ion_temperature = 0.075 + 0.12 * (1.0 - radial)
    electron_temperature = 0.095 + 0.16 * (1.0 - radial)
    neutral_pressure = neutral_density * neutral_temperature
    ion_pressure = ion_density * ion_temperature
    electron_pressure = electron_density * electron_temperature
    neutral_velocity = 0.025 * np.sin(theta - 5.0 * phi)
    ion_velocity = 0.040 * np.cos(2.0 * theta - 5.0 * phi)
    neutral_momentum = neutral_density * neutral_velocity
    ion_momentum = ion_density * ion_velocity

    result = compute_fci_neutral_reaction_diffusion(
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

    ionisation = np.asarray(result.ionisation_rate, dtype=np.float64)
    recombination = np.asarray(result.recombination_rate, dtype=np.float64)
    charge_exchange = np.asarray(result.charge_exchange_rate, dtype=np.float64)
    neutral_diffusion = np.asarray(result.neutral_diffusion_source, dtype=np.float64)
    neutral_source = np.asarray(result.neutral_density_source, dtype=np.float64)
    ion_source = np.asarray(result.ion_density_source, dtype=np.float64)
    momentum_pair = np.asarray(result.neutral_momentum_source + result.ion_momentum_source, dtype=np.float64)

    total_ionisation = float(np.sum(ionisation))
    total_recombination = float(np.sum(recombination))
    total_charge_exchange = float(np.sum(charge_exchange))
    particle_residual = float(np.asarray(result.total_particle_residual))
    momentum_residual = float(np.asarray(result.total_momentum_residual))
    particle_relative_error = abs(particle_residual) / max(total_ionisation + total_recombination, 1.0e-30)
    momentum_relative_error = abs(momentum_residual) / max(float(np.sum(np.abs(momentum_pair))) + 1.0e-30, 1.0e-30)
    jacobian = np.asarray(geometry.metric.J, dtype=np.float64)
    diffusion_integral = float(np.sum(jacobian * neutral_diffusion))
    diffusion_relative_integral = abs(diffusion_integral) / max(float(np.sum(np.abs(jacobian * neutral_diffusion))), 1.0e-30)
    cx_to_ionisation_ratio = total_charge_exchange / max(total_ionisation, 1.0e-30)

    report: dict[str, Any] = {
        "case": "non_axisymmetric_neutral_diffusion_reaction_gate",
        "geometry": geometry.metadata,
        "total_ionisation": total_ionisation,
        "total_recombination": total_recombination,
        "total_charge_exchange": total_charge_exchange,
        "cx_to_ionisation_ratio": cx_to_ionisation_ratio,
        "particle_reaction_residual": particle_residual,
        "particle_reaction_relative_error": particle_relative_error,
        "momentum_reaction_residual": momentum_residual,
        "momentum_reaction_relative_error": momentum_relative_error,
        "neutral_diffusion_integral": diffusion_integral,
        "neutral_diffusion_relative_integral": diffusion_relative_integral,
        "max_ionisation_rate": float(np.max(ionisation)),
        "max_charge_exchange_rate": float(np.max(charge_exchange)),
    }
    report["passed"] = (
        total_ionisation > 0.0
        and total_recombination > 0.0
        and total_charge_exchange > 0.0
        and particle_relative_error < 1.0e-12
        and momentum_relative_error < 1.0e-12
        and diffusion_relative_integral < 2.0e-2
        and cx_to_ionisation_ratio > 0.05
    )
    arrays = {
        "ionisation_toroidal": np.sum(ionisation, axis=0).astype(np.float32),
        "recombination_toroidal": np.sum(recombination, axis=0).astype(np.float32),
        "charge_exchange_toroidal": np.sum(charge_exchange, axis=0).astype(np.float32),
        "neutral_source_toroidal": np.sum(neutral_source, axis=0).astype(np.float32),
        "ion_source_toroidal": np.sum(ion_source, axis=0).astype(np.float32),
        "neutral_density_slice": neutral_density[:, 0, :].astype(np.float32),
        "radial_grid": np.mean(radial, axis=(1, 2)).astype(np.float32),
        "radial_rates": np.stack(
            [
                np.mean(neutral_density, axis=(1, 2)),
                np.sum(ionisation, axis=(1, 2)),
                np.sum(recombination, axis=(1, 2)),
                np.sum(charge_exchange, axis=(1, 2)),
            ],
            axis=1,
        ).astype(np.float32),
        "summary": np.asarray(
            [
                total_ionisation,
                total_recombination,
                total_charge_exchange,
                abs(particle_residual),
                abs(momentum_residual),
            ],
            dtype=np.float32,
        ),
    }
    return report, arrays


def save_stellarator_neutral_physics_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.8), constrained_layout=True)
    extent = [0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi]

    image0 = axes[0, 0].imshow(arrays["ionisation_toroidal"].T, origin="lower", aspect="auto", extent=extent, cmap="inferno")
    axes[0, 0].set_title("ionisation source")
    axes[0, 0].set_xlabel("toroidal angle")
    axes[0, 0].set_ylabel("poloidal angle")
    fig.colorbar(image0, ax=axes[0, 0])

    image1 = axes[0, 1].imshow(
        arrays["charge_exchange_toroidal"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    axes[0, 1].set_title("charge-exchange rate")
    axes[0, 1].set_xlabel("toroidal angle")
    axes[0, 1].set_ylabel("poloidal angle")
    fig.colorbar(image1, ax=axes[0, 1])

    signed = arrays["ion_source_toroidal"].T + arrays["neutral_source_toroidal"].T
    vmax = float(np.max(np.abs(signed)))
    image2 = axes[0, 2].imshow(
        signed,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[0, 2].set_title("local plasma + neutral source")
    axes[0, 2].set_xlabel("toroidal angle")
    axes[0, 2].set_ylabel("poloidal angle")
    fig.colorbar(image2, ax=axes[0, 2])

    radial_grid = arrays["radial_grid"]
    radial_rates = arrays["radial_rates"]
    axes[1, 0].plot(radial_grid, radial_rates[:, 0] / np.max(radial_rates[:, 0]), lw=2.0, label="neutral density")
    axes[1, 0].plot(radial_grid, radial_rates[:, 1] / np.max(radial_rates[:, 1]), lw=2.0, label="ionisation")
    axes[1, 0].plot(radial_grid, radial_rates[:, 2] / np.max(radial_rates[:, 2]), lw=2.0, label="recombination")
    axes[1, 0].plot(radial_grid, radial_rates[:, 3] / np.max(radial_rates[:, 3]), lw=2.0, label="charge exchange")
    axes[1, 0].set_title("normalized radial reaction profiles")
    axes[1, 0].set_xlabel("normalized radius")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False, fontsize=8)

    image3 = axes[1, 1].imshow(arrays["neutral_density_slice"], origin="lower", aspect="auto", cmap="viridis")
    axes[1, 1].set_title("neutral density at one toroidal plane")
    axes[1, 1].set_xlabel("poloidal index")
    axes[1, 1].set_ylabel("radial index")
    fig.colorbar(image3, ax=axes[1, 1])

    summary = arrays["summary"]
    labels = ["ion", "rec", "CX", "particle err", "mom err"]
    axes[1, 2].bar(np.arange(summary.size), summary, color=["#9b2226", "#005f73", "#ee9b00", "#94d2bd", "#0a9396"])
    axes[1, 2].set_xticks(np.arange(summary.size), labels, rotation=18, ha="right")
    axes[1, 2].set_yscale("log")
    axes[1, 2].grid(axis="y", alpha=0.25)
    axes[1, 2].set_title("integrated reaction balance")
    axes[1, 2].text(
        0.03,
        0.96,
        "\n".join(
            [
                f"particle rel. err = {report['particle_reaction_relative_error']:.1e}",
                f"momentum rel. err = {report['momentum_reaction_relative_error']:.1e}",
                f"CX / ionisation = {report['cx_to_ionisation_ratio']:.2f}",
            ]
        ),
        transform=axes[1, 2].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )
    fig.suptitle(
        "Non-axisymmetric neutral gate: diffusion plus ionisation, recombination, and charge exchange",
        fontsize=14,
    )
    fig.savefig(resolved, dpi=190)
    plt.close(fig)
    return resolved
