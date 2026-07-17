from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import jax.numpy as jnp
import numpy as np

from ..geometry import build_synthetic_stellarator_geometry
from ..native.fci_vorticity import apply_fci_vorticity_operator, solve_fci_vorticity_potential_cg


@dataclass(frozen=True)
class StellaratorVorticityCampaignArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_stellarator_vorticity_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "stellarator_vorticity_campaign",
    nx: int = 28,
    ny: int = 26,
    nz: int = 52,
) -> StellaratorVorticityCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    report, arrays = build_stellarator_vorticity_campaign(nx=nx, ny=ny, nz=nz)
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_stellarator_vorticity_plot(report, arrays, plot_png_path)
    return StellaratorVorticityCampaignArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_stellarator_vorticity_campaign(
    *,
    nx: int = 28,
    ny: int = 26,
    nz: int = 52,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    geometry = build_synthetic_stellarator_geometry(nx=nx, ny=ny, nz=nz)
    phi_exact = _manufactured_potential(geometry)
    density = _manufactured_density(geometry)
    boussinesq_vorticity = apply_fci_vorticity_operator(
        phi_exact,
        density,
        geometry.metric,
        boussinesq=True,
    )
    non_boussinesq_vorticity = apply_fci_vorticity_operator(
        phi_exact,
        density,
        geometry.metric,
        boussinesq=False,
    )
    boussinesq_solve = solve_fci_vorticity_potential_cg(
        boussinesq_vorticity,
        density,
        geometry.metric,
        iterations=600,
        boussinesq=True,
    )
    non_boussinesq_solve = solve_fci_vorticity_potential_cg(
        non_boussinesq_vorticity,
        density,
        geometry.metric,
        iterations=600,
        boussinesq=False,
    )
    phi_centered = _remove_weighted_mean(np.asarray(phi_exact), np.asarray(geometry.metric.J))
    boussinesq_phi = np.asarray(boussinesq_solve.potential, dtype=np.float64)
    non_boussinesq_phi = np.asarray(non_boussinesq_solve.potential, dtype=np.float64)
    boussinesq_error = boussinesq_phi - phi_centered
    non_boussinesq_error = non_boussinesq_phi - phi_centered
    phi_norm = max(np.sqrt(np.mean(phi_centered * phi_centered)), 1.0e-30)
    boussinesq_relative_l2_error = float(
        np.sqrt(np.mean(boussinesq_error * boussinesq_error)) / phi_norm
    )
    non_boussinesq_relative_l2_error = float(
        np.sqrt(np.mean(non_boussinesq_error * non_boussinesq_error)) / phi_norm
    )
    boussinesq_residual_l2 = float(np.asarray(boussinesq_solve.residual_l2))
    non_boussinesq_residual_l2 = float(np.asarray(non_boussinesq_solve.residual_l2))
    boussinesq_vorticity_np = np.asarray(boussinesq_vorticity, dtype=np.float64)
    non_boussinesq_vorticity_np = np.asarray(non_boussinesq_vorticity, dtype=np.float64)
    vorticity_difference = non_boussinesq_vorticity_np - boussinesq_vorticity_np
    operator_difference_relative_l2 = float(
        np.sqrt(np.mean(vorticity_difference * vorticity_difference))
        / max(np.sqrt(np.mean(boussinesq_vorticity_np * boussinesq_vorticity_np)), 1.0e-30)
    )
    coefficient = np.asarray(density / jnp.maximum(jnp.square(geometry.metric.Bxy), 1.0e-30))
    coefficient_contrast = float(np.max(coefficient) / max(float(np.min(coefficient)), 1.0e-30))
    constant_coefficient_density = 1.7 * jnp.square(geometry.metric.Bxy)
    constant_boussinesq_operator = apply_fci_vorticity_operator(
        phi_exact,
        constant_coefficient_density,
        geometry.metric,
        boussinesq=True,
    )
    constant_non_boussinesq_operator = apply_fci_vorticity_operator(
        phi_exact,
        constant_coefficient_density,
        geometry.metric,
        boussinesq=False,
    )
    constant_coefficient_operator_linf = float(
        np.max(
            np.abs(
                np.asarray(constant_boussinesq_operator)
                - np.asarray(constant_non_boussinesq_operator)
            )
        )
    )
    exb_radial = _radial_exb_proxy(non_boussinesq_phi, geometry)
    report: dict[str, Any] = {
        "case": "non_axisymmetric_boussinesq_non_boussinesq_vorticity_inversion",
        "geometry": geometry.metadata,
        "iterations": int(non_boussinesq_solve.iterations),
        "boussinesq_relative_l2_potential_error": boussinesq_relative_l2_error,
        "non_boussinesq_relative_l2_potential_error": non_boussinesq_relative_l2_error,
        "relative_l2_potential_error": non_boussinesq_relative_l2_error,
        "boussinesq_relative_residual_l2": boussinesq_residual_l2,
        "non_boussinesq_relative_residual_l2": non_boussinesq_residual_l2,
        "relative_residual_l2": non_boussinesq_residual_l2,
        "operator_difference_relative_l2": operator_difference_relative_l2,
        "constant_coefficient_operator_linf": constant_coefficient_operator_linf,
        "density_min": float(np.min(np.asarray(density))),
        "density_max": float(np.max(np.asarray(density))),
        "density_over_b_squared_contrast": coefficient_contrast,
        "radial_exb_proxy_rms": float(np.sqrt(np.mean(exb_radial * exb_radial))),
        "potential_rms": float(np.sqrt(np.mean(non_boussinesq_phi * non_boussinesq_phi))),
        "boussinesq_vorticity_rms": float(
            np.sqrt(np.mean(boussinesq_vorticity_np * boussinesq_vorticity_np))
        ),
        "non_boussinesq_vorticity_rms": float(
            np.sqrt(np.mean(non_boussinesq_vorticity_np * non_boussinesq_vorticity_np))
        ),
        "vorticity_rms": float(
            np.sqrt(np.mean(non_boussinesq_vorticity_np * non_boussinesq_vorticity_np))
        ),
    }
    report["passed"] = (
        boussinesq_relative_l2_error < 2.5e-2
        and non_boussinesq_relative_l2_error < 2.5e-2
        and boussinesq_residual_l2 < 5.0e-3
        and non_boussinesq_residual_l2 < 5.0e-3
        and operator_difference_relative_l2 > 5.0e-2
        and constant_coefficient_operator_linf < 1.0e-8
        and report["radial_exb_proxy_rms"] > 1.0e-4
    )
    arrays = {
        "phi_exact_slice": phi_centered[:, 0, :].astype(np.float32),
        "phi_solved_slice": non_boussinesq_phi[:, 0, :].astype(np.float32),
        "boussinesq_phi_solved_slice": boussinesq_phi[:, 0, :].astype(np.float32),
        "non_boussinesq_phi_solved_slice": non_boussinesq_phi[:, 0, :].astype(np.float32),
        "phi_error_slice": non_boussinesq_error[:, 0, :].astype(np.float32),
        "boussinesq_phi_error_slice": boussinesq_error[:, 0, :].astype(np.float32),
        "non_boussinesq_phi_error_slice": non_boussinesq_error[:, 0, :].astype(np.float32),
        "vorticity_slice": non_boussinesq_vorticity_np[:, 0, :].astype(np.float32),
        "boussinesq_vorticity_slice": boussinesq_vorticity_np[:, 0, :].astype(np.float32),
        "non_boussinesq_vorticity_slice": non_boussinesq_vorticity_np[:, 0, :].astype(np.float32),
        "vorticity_difference_slice": vorticity_difference[:, 0, :].astype(np.float32),
        "density_over_b_squared_slice": coefficient[:, 0, :].astype(np.float32),
        "radial_exb_slice": exb_radial[:, 0, :].astype(np.float32),
        "summary": np.asarray(
            [
                boussinesq_relative_l2_error,
                non_boussinesq_relative_l2_error,
                boussinesq_residual_l2,
                non_boussinesq_residual_l2,
                operator_difference_relative_l2,
                constant_coefficient_operator_linf,
                report["radial_exb_proxy_rms"],
                report["potential_rms"],
            ],
            dtype=np.float32,
        ),
    }
    return report, arrays


def save_stellarator_vorticity_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(16.2, 12.2), constrained_layout=True)
    image0 = axes[0, 0].imshow(arrays["phi_exact_slice"], origin="lower", aspect="auto", cmap="viridis")
    axes[0, 0].set_title("manufactured potential")
    fig.colorbar(image0, ax=axes[0, 0])
    image1 = axes[0, 1].imshow(arrays["boussinesq_phi_solved_slice"], origin="lower", aspect="auto", cmap="viridis")
    axes[0, 1].set_title("Boussinesq reconstructed potential")
    fig.colorbar(image1, ax=axes[0, 1])
    image2 = axes[0, 2].imshow(arrays["non_boussinesq_phi_solved_slice"], origin="lower", aspect="auto", cmap="viridis")
    axes[0, 2].set_title("non-Boussinesq reconstructed potential")
    fig.colorbar(image2, ax=axes[0, 2])
    image3 = axes[1, 0].imshow(arrays["boussinesq_vorticity_slice"], origin="lower", aspect="auto", cmap="magma")
    axes[1, 0].set_title("Boussinesq vorticity")
    fig.colorbar(image3, ax=axes[1, 0])
    image4 = axes[1, 1].imshow(arrays["non_boussinesq_vorticity_slice"], origin="lower", aspect="auto", cmap="magma")
    axes[1, 1].set_title("non-Boussinesq vorticity")
    fig.colorbar(image4, ax=axes[1, 1])
    vmax_vort = float(np.max(np.abs(arrays["vorticity_difference_slice"])))
    image5 = axes[1, 2].imshow(
        arrays["vorticity_difference_slice"],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax_vort,
        vmax=vmax_vort,
    )
    axes[1, 2].set_title("non-Boussinesq minus Boussinesq")
    fig.colorbar(image5, ax=axes[1, 2])
    vmax_bouss = float(np.max(np.abs(arrays["boussinesq_phi_error_slice"])))
    image6 = axes[2, 0].imshow(
        arrays["boussinesq_phi_error_slice"],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax_bouss,
        vmax=vmax_bouss,
    )
    axes[2, 0].set_title("Boussinesq potential error")
    fig.colorbar(image6, ax=axes[2, 0])
    vmax_non_bouss = float(np.max(np.abs(arrays["non_boussinesq_phi_error_slice"])))
    image7 = axes[2, 1].imshow(
        arrays["non_boussinesq_phi_error_slice"],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax_non_bouss,
        vmax=vmax_non_bouss,
    )
    axes[2, 1].set_title("non-Boussinesq potential error")
    fig.colorbar(image7, ax=axes[2, 1])
    labels = [
        "Bq err",
        "non-Bq err",
        "Bq res",
        "non-Bq res",
        "op diff",
        "const eq",
        "ExB rms",
        "phi rms",
    ]
    colors = ["#9b2226", "#bb3e03", "#0a9396", "#005f73", "#ee9b00", "#94d2bd", "#ca6702", "#001219"]
    axes[2, 2].bar(np.arange(len(labels)), arrays["summary"], color=colors)
    axes[2, 2].set_xticks(np.arange(len(labels)), labels, rotation=28, ha="right")
    axes[2, 2].set_yscale("log")
    axes[2, 2].grid(axis="y", alpha=0.25)
    axes[2, 2].set_title("model and inversion metrics")
    for axis in axes.ravel()[:8]:
        axis.set_xlabel("poloidal index")
        axis.set_ylabel("radial index")
    fig.suptitle(
        "Non-axisymmetric vorticity gate: Boussinesq vs non-Boussinesq perpendicular polarization",
        fontsize=14,
    )
    fig.savefig(resolved, dpi=190)
    plt.close(fig)
    return resolved


def _manufactured_potential(geometry: object) -> jnp.ndarray:
    return (
        jnp.sin(jnp.pi * geometry.radial) * jnp.cos(2.0 * geometry.poloidal_angle - 5.0 * geometry.toroidal_angle)
        + 0.20 * geometry.radial * jnp.sin(3.0 * geometry.poloidal_angle + 2.0 * geometry.toroidal_angle)
    )


def _manufactured_density(geometry: object) -> jnp.ndarray:
    density = (
        1.0
        + 0.35 * geometry.radial
        + 0.12 * jnp.cos(geometry.poloidal_angle - 2.0 * geometry.toroidal_angle)
        + 0.08 * jnp.cos(3.0 * geometry.toroidal_angle)
    )
    return jnp.maximum(density, 0.1)


def _remove_weighted_mean(value: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return value - float(np.sum(weights * value) / np.sum(weights))


def _radial_exb_proxy(phi: np.ndarray, geometry: object) -> np.ndarray:
    dz = float(2.0 * np.pi / geometry.shape[2])
    return -(np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2.0 * dz)
