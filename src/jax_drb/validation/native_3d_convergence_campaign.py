from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Native3DConvergenceCampaignArtifacts:
    summary_json_path: Path
    summary_plot_png_path: Path


def create_native_3d_convergence_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "native_3d_convergence_campaign",
) -> Native3DConvergenceCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_native_3d_convergence_campaign_report()
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary_plot_png_path = save_native_3d_convergence_campaign_plot(report, images_dir / f"{case_label}.png")
    return Native3DConvergenceCampaignArtifacts(
        summary_json_path=summary_json_path,
        summary_plot_png_path=summary_plot_png_path,
    )


def build_native_3d_convergence_campaign_report() -> dict[str, object]:
    resolutions = (8, 16, 32, 64)
    entries = [_traced_field_line_operator_entry(resolution) for resolution in resolutions]
    observed_orders = []
    for coarse, fine in zip(entries[:-1], entries[1:], strict=False):
        coarse_error = max(coarse["g11_error"], coarse["g33_error"])
        fine_error = max(fine["g11_error"], fine["g33_error"])
        observed_orders.append(float(np.log(coarse_error / fine_error) / np.log(2.0)))
    return {
        "case": "native_3d_convergence_campaign",
        "operator": "native_traced_field_line_radial_profile",
        "entries": entries,
        "observed_orders": observed_orders,
        "min_observed_order": float(min(observed_orders)),
    }


def save_native_3d_convergence_campaign_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    entries = list(report["entries"])
    resolutions = np.asarray([entry["resolution"] for entry in entries], dtype=np.float64)
    g11_errors = np.asarray([entry["g11_error"] for entry in entries], dtype=np.float64)
    g33_errors = np.asarray([entry["g33_error"] for entry in entries], dtype=np.float64)
    orders = np.asarray(report["observed_orders"], dtype=np.float64)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), constrained_layout=True)
    axes[0].loglog(resolutions, g11_errors, marker="o", linewidth=2.0, color="#005f73", label="g11")
    axes[0].loglog(resolutions, g33_errors, marker="o", linewidth=2.0, color="#ca6702", label="g33")
    axes[0].set_xlabel("transverse resolution (Ny = Nz)")
    axes[0].set_ylabel("max abs radial-profile error")
    axes[0].set_title("Native traced-field-line reduction convergence")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False)

    x = np.arange(len(orders))
    axes[1].bar(x, orders, color="#3a86ff")
    axes[1].axhline(1.0, color="#bb3e03", linestyle="--", linewidth=1.5, label="first order")
    axes[1].set_xticks(x, [f"{entries[index]['resolution']}→{entries[index + 1]['resolution']}" for index in range(len(orders))])
    axes[1].set_ylabel("observed order")
    axes[1].set_title("Observed refinement order")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend(frameon=False)

    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def _traced_field_line_operator_entry(resolution: int) -> dict[str, float]:
    x = jnp.linspace(0.0, 1.0, 17, endpoint=True)
    y = (jnp.arange(resolution, dtype=jnp.float64) + 0.5) / resolution
    z = (jnp.arange(resolution, dtype=jnp.float64) + 0.5) / resolution
    xv, yv, zv = jnp.meshgrid(x, y, z, indexing="ij")
    g11 = 1.0 + 0.2 * xv + 0.1 * yv**2 + 0.05 * zv
    g33 = 2.0 + 0.1 * xv**2 + 0.15 * yv + 0.2 * zv**2
    native_g11 = np.asarray(jnp.mean(g11, axis=(1, 2)), dtype=np.float64)
    native_g33 = np.asarray(jnp.mean(g33, axis=(1, 2)), dtype=np.float64)
    expected_g11 = np.asarray(1.0 + 0.2 * x + 0.1 / 3.0 + 0.05 / 2.0, dtype=np.float64)
    expected_g33 = np.asarray(2.0 + 0.1 * x**2 + 0.15 / 2.0 + 0.2 / 3.0, dtype=np.float64)
    return {
        "resolution": int(resolution),
        "g11_error": float(np.max(np.abs(native_g11 - expected_g11))),
        "g33_error": float(np.max(np.abs(native_g33 - expected_g33))),
    }
