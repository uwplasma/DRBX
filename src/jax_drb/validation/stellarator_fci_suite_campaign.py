from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np

from ..geometry import build_synthetic_stellarator_geometry
from .stellarator_fci_geometry_campaign import build_stellarator_fci_geometry_report


@dataclass(frozen=True)
class StellaratorFciSuiteCampaignArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_stellarator_fci_suite_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "stellarator_fci_suite_campaign",
    nx: int = 30,
    ny: int = 28,
    nz: int = 56,
) -> StellaratorFciSuiteCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report, arrays = build_stellarator_fci_suite_campaign(nx=nx, ny=ny, nz=nz)
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_stellarator_fci_suite_plot(report, arrays, plot_png_path)
    return StellaratorFciSuiteCampaignArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_stellarator_fci_suite_campaign(
    *,
    nx: int = 30,
    ny: int = 28,
    nz: int = 56,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    configs = _suite_configs()
    reports: list[dict[str, Any]] = []
    row_labels: list[str] = []
    cross_r: list[np.ndarray] = []
    cross_z: list[np.ndarray] = []
    bmag_slices: list[np.ndarray] = []
    connection_slices: list[np.ndarray] = []
    curvature_map_slices: list[np.ndarray] = []
    summary_rows: list[list[float]] = []
    for config in configs:
        geometry = build_synthetic_stellarator_geometry(nx=nx, ny=ny, nz=nz, **config["parameters"])
        report = build_stellarator_fci_geometry_report(geometry)
        reports.append(report)
        row_labels.append(str(config["label"]))
        x = np.asarray(geometry.coordinates_x)
        y = np.asarray(geometry.coordinates_y)
        z = np.asarray(geometry.coordinates_z)
        r_major = np.sqrt(x * x + y * y)
        toroidal_index = max(1, ny // 6)
        cross_r.append(r_major[:, toroidal_index, :].astype(np.float32))
        cross_z.append(z[:, toroidal_index, :].astype(np.float32))
        bmag_slices.append(np.asarray(geometry.metric.Bxy[:, 0, :], dtype=np.float32))
        connection_slices.append(np.asarray(geometry.connection_length[:, 0, :], dtype=np.float32))
        radial_shift = np.asarray(geometry.maps.forward_x) - np.arange(
            geometry.maps.shape[0],
            dtype=np.float64,
        )[:, None, None]
        curvature_map_slices.append(np.asarray(geometry.curvature[:, 0, :] * radial_shift[:, 0, :], dtype=np.float32))
        map_report = report["map_diagnostics"]
        summary_rows.append(
            [
                float(report["magnetic_field"]["mirror_ratio"]),
                float(report["connection_length"]["mean"]),
                float(report["connection_length"]["std"]),
                float(report["curvature"]["std"]),
                float(map_report["radial_shift_linf_cells"]),
                float(map_report["forward_boundary_fraction"] + map_report["backward_boundary_fraction"]),
            ]
        )
    summary = np.asarray(summary_rows, dtype=np.float64)
    report = {
        "case": "stellarator_fci_multi_configuration_suite",
        "configuration_labels": row_labels,
        "configurations": reports,
        "summary_columns": [
            "mirror_ratio",
            "connection_length_mean",
            "connection_length_std",
            "curvature_std",
            "radial_shift_linf_cells",
            "two_sided_boundary_fraction",
        ],
        "summary": summary_rows,
        "passed": all(bool(item["passed"]) for item in reports)
        and float(np.min(summary[:, 1])) > 0.0
        and float(np.max(summary[:, 4])) > 0.0
        and float(np.max(summary[:, 5])) < 0.20,
    }
    arrays = {
        "labels": np.asarray(row_labels),
        "cross_r": np.asarray(cross_r, dtype=np.float32),
        "cross_z": np.asarray(cross_z, dtype=np.float32),
        "bmag_slices": np.asarray(bmag_slices, dtype=np.float32),
        "connection_slices": np.asarray(connection_slices, dtype=np.float32),
        "curvature_map_slices": np.asarray(curvature_map_slices, dtype=np.float32),
        "summary": summary.astype(np.float32),
    }
    return report, arrays


def save_stellarator_fci_suite_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(label) for label in arrays["labels"]]
    nrows = len(labels)
    fig, axes = plt.subplots(
        nrows + 1,
        4,
        figsize=(15.5, 11.2),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.72]},
    )
    bmag_vmin = float(np.min(arrays["bmag_slices"]))
    bmag_vmax = float(np.max(arrays["bmag_slices"]))
    connection_vmin = float(np.min(arrays["connection_slices"]))
    connection_vmax = float(np.max(arrays["connection_slices"]))
    curvature_vmax = float(np.max(np.abs(arrays["curvature_map_slices"])))
    for row, label in enumerate(labels):
        cross_r = arrays["cross_r"][row]
        cross_z = arrays["cross_z"][row]
        for surface in (max(1, cross_r.shape[0] // 4), max(2, cross_r.shape[0] // 2), cross_r.shape[0] - 2):
            axes[row, 0].plot(cross_r[surface], cross_z[surface], linewidth=1.2)
        if row == 0:
            axes[row, 0].set_title("sampled cross-sections")
        axes[row, 0].set_xlabel("major radius" if row == nrows - 1 else "")
        axes[row, 0].set_ylabel(f"{label}\nvertical")
        axes[row, 0].set_aspect("equal", adjustable="box")

        image0 = axes[row, 1].imshow(
            arrays["bmag_slices"][row],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=bmag_vmin,
            vmax=bmag_vmax,
        )
        if row == 0:
            axes[row, 1].set_title("magnetic-field strength")
        axes[row, 1].set_xlabel("poloidal index" if row == nrows - 1 else "")
        axes[row, 1].set_ylabel("radial index")

        image1 = axes[row, 2].imshow(
            arrays["connection_slices"][row],
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=connection_vmin,
            vmax=connection_vmax,
        )
        if row == 0:
            axes[row, 2].set_title("connection-length proxy")
        axes[row, 2].set_xlabel("poloidal index" if row == nrows - 1 else "")
        axes[row, 2].set_ylabel("radial index")

        image2 = axes[row, 3].imshow(
            arrays["curvature_map_slices"][row],
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-curvature_vmax,
            vmax=curvature_vmax,
        )
        if row == 0:
            axes[row, 3].set_title("curvature x radial map shift")
        axes[row, 3].set_xlabel("poloidal index" if row == nrows - 1 else "")
        axes[row, 3].set_ylabel("radial index")

    fig.colorbar(image0, ax=axes[:nrows, 1], label="normalized |B|", shrink=0.72)
    fig.colorbar(image1, ax=axes[:nrows, 2], label="length proxy", shrink=0.72)
    fig.colorbar(image2, ax=axes[:nrows, 3], label="signed proxy", shrink=0.72)

    summary = arrays["summary"]
    x_locations = np.arange(nrows)
    bar_axes = axes[nrows]
    for axis in bar_axes:
        axis.set_xticks(x_locations, labels, rotation=18, ha="right", fontsize=8)
    bar_axes[0].bar(x_locations, summary[:, 0], color="#005f73")
    bar_axes[0].set_title("mirror ratio")
    bar_axes[1].bar(x_locations, summary[:, 1], color="#9b2226")
    bar_axes[1].set_title("mean connection length")
    bar_axes[2].bar(x_locations, summary[:, 3], color="#ca6702")
    bar_axes[2].set_title("curvature RMS")
    bar_axes[3].bar(x_locations, summary[:, 4], color="#0a9396")
    bar_axes[3].set_title("max radial map shift")
    for axis in bar_axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Multi-configuration 3D stellarator field-line-map geometry suite: "
        f"{len(labels)} passing metric/map gates",
        fontsize=14,
    )
    fig.savefig(resolved, dpi=180)
    plt.close(fig)
    return resolved


def _suite_configs() -> list[dict[str, Any]]:
    return [
        {
            "label": "baseline island SOL",
            "parameters": {
                "elongation": 1.45,
                "field_periods": 5,
                "island_mode": 2,
                "island_amplitude": 0.030,
                "mirror_amplitude": 0.16,
                "iota_axis": 0.38,
                "iota_edge": 0.58,
            },
        },
        {
            "label": "strong island map",
            "parameters": {
                "elongation": 1.35,
                "field_periods": 5,
                "island_mode": 2,
                "island_amplitude": 0.055,
                "mirror_amplitude": 0.18,
                "iota_axis": 0.34,
                "iota_edge": 0.64,
            },
        },
        {
            "label": "high mirror shear",
            "parameters": {
                "elongation": 1.65,
                "field_periods": 4,
                "island_mode": 3,
                "island_amplitude": 0.026,
                "mirror_amplitude": 0.24,
                "iota_axis": 0.30,
                "iota_edge": 0.70,
            },
        },
    ]
