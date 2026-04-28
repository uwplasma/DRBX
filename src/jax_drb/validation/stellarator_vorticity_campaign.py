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
    density = 1.0 + 0.25 * geometry.radial + 0.04 * jnp.cos(2.0 * geometry.poloidal_angle - 5.0 * geometry.toroidal_angle)
    vorticity = apply_fci_vorticity_operator(phi_exact, density, geometry.metric)
    solve = solve_fci_vorticity_potential_cg(vorticity, density, geometry.metric, iterations=600)
    phi_centered = _remove_weighted_mean(np.asarray(phi_exact), np.asarray(geometry.metric.J))
    phi_solved = np.asarray(solve.potential, dtype=np.float64)
    error = phi_solved - phi_centered
    relative_l2_error = float(np.sqrt(np.mean(error * error)) / max(np.sqrt(np.mean(phi_centered * phi_centered)), 1.0e-30))
    residual_l2 = float(np.asarray(solve.residual_l2))
    exb_radial = _radial_exb_proxy(phi_solved, geometry)
    report: dict[str, Any] = {
        "case": "non_axisymmetric_metric_weighted_vorticity_inversion",
        "geometry": geometry.metadata,
        "iterations": int(solve.iterations),
        "relative_l2_potential_error": relative_l2_error,
        "relative_residual_l2": residual_l2,
        "radial_exb_proxy_rms": float(np.sqrt(np.mean(exb_radial * exb_radial))),
        "potential_rms": float(np.sqrt(np.mean(phi_solved * phi_solved))),
        "vorticity_rms": float(np.sqrt(np.mean(np.asarray(vorticity) * np.asarray(vorticity)))),
    }
    report["passed"] = relative_l2_error < 2.5e-2 and residual_l2 < 5.0e-3 and report["radial_exb_proxy_rms"] > 1.0e-4
    arrays = {
        "phi_exact_slice": phi_centered[:, 0, :].astype(np.float32),
        "phi_solved_slice": phi_solved[:, 0, :].astype(np.float32),
        "phi_error_slice": error[:, 0, :].astype(np.float32),
        "vorticity_slice": np.asarray(vorticity[:, 0, :], dtype=np.float32),
        "radial_exb_slice": exb_radial[:, 0, :].astype(np.float32),
        "summary": np.asarray(
            [relative_l2_error, residual_l2, report["radial_exb_proxy_rms"], report["potential_rms"]],
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
    fig, axes = plt.subplots(2, 3, figsize=(15.3, 8.6), constrained_layout=True)
    image0 = axes[0, 0].imshow(arrays["phi_exact_slice"], origin="lower", aspect="auto", cmap="viridis")
    axes[0, 0].set_title("manufactured potential")
    fig.colorbar(image0, ax=axes[0, 0])
    image1 = axes[0, 1].imshow(arrays["phi_solved_slice"], origin="lower", aspect="auto", cmap="viridis")
    axes[0, 1].set_title("CG reconstructed potential")
    fig.colorbar(image1, ax=axes[0, 1])
    vmax = float(np.max(np.abs(arrays["phi_error_slice"])))
    image2 = axes[0, 2].imshow(arrays["phi_error_slice"], origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title("potential reconstruction error")
    fig.colorbar(image2, ax=axes[0, 2])
    image3 = axes[1, 0].imshow(arrays["vorticity_slice"], origin="lower", aspect="auto", cmap="magma")
    axes[1, 0].set_title("metric-weighted vorticity")
    fig.colorbar(image3, ax=axes[1, 0])
    image4 = axes[1, 1].imshow(arrays["radial_exb_slice"], origin="lower", aspect="auto", cmap="coolwarm")
    axes[1, 1].set_title("radial E x B proxy")
    fig.colorbar(image4, ax=axes[1, 1])
    labels = ["phi L2 err", "residual", "ExB rms", "phi rms"]
    axes[1, 2].bar(np.arange(4), arrays["summary"], color=["#9b2226", "#0a9396", "#ee9b00", "#005f73"])
    axes[1, 2].set_xticks(np.arange(4), labels, rotation=18, ha="right")
    axes[1, 2].set_yscale("log")
    axes[1, 2].grid(axis="y", alpha=0.25)
    axes[1, 2].set_title("inversion metrics")
    for axis in axes.ravel()[:5]:
        axis.set_xlabel("poloidal index")
        axis.set_ylabel("radial index")
    fig.suptitle(
        "Non-axisymmetric vorticity/potential gate: metric-weighted perpendicular inversion and E x B proxy",
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


def _remove_weighted_mean(value: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return value - float(np.sum(weights * value) / np.sum(weights))


def _radial_exb_proxy(phi: np.ndarray, geometry: object) -> np.ndarray:
    dz = float(2.0 * np.pi / geometry.shape[2])
    return -(np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2.0 * dz)
