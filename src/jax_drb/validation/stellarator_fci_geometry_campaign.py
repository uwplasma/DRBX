from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from ..geometry import SyntheticStellaratorGeometry, build_metric_report, build_synthetic_stellarator_geometry


@dataclass(frozen=True)
class StellaratorFciGeometryCampaignArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_stellarator_fci_geometry_campaign_package(
    *,
    output_root: str | Path,
    case_label: str = "stellarator_fci_geometry_campaign",
    nx: int = 36,
    ny: int = 32,
    nz: int = 64,
) -> StellaratorFciGeometryCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    geometry = build_synthetic_stellarator_geometry(nx=nx, ny=ny, nz=nz)
    report = build_stellarator_fci_geometry_report(geometry)
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    _write_geometry_arrays(geometry, arrays_npz_path)
    plot_png_path = images_dir / f"{case_label}.png"
    save_stellarator_fci_geometry_plot(geometry, report, plot_png_path)
    return StellaratorFciGeometryCampaignArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_stellarator_fci_geometry_report(geometry: SyntheticStellaratorGeometry) -> dict[str, object]:
    metric_report = build_metric_report(geometry)
    maps = geometry
    forward_dx = np.asarray(maps.forward_x) - np.arange(maps.shape[0], dtype=np.float64)[:, None, None]
    forward_y0 = np.arange(maps.shape[1], dtype=np.float64)[None, :, None]
    forward_dy = np.mod(
        np.asarray(maps.forward_y) - forward_y0 + maps.shape[1] / 2.0,
        maps.shape[1],
    ) - maps.shape[1] / 2.0
    connection = np.asarray(geometry.connection_length, dtype=np.float64)
    curvature = np.asarray(geometry.curvature, dtype=np.float64)
    bmag = np.asarray(geometry.Bmag, dtype=np.float64)
    iota = np.asarray(geometry.iota, dtype=np.float64)
    report = {
        "case": "non_axisymmetric_fci_geometry",
        "geometry": geometry.metadata,
        "metric": metric_report,
        "map_diagnostics": {
            "forward_boundary_fraction": float(np.mean(np.asarray(maps.forward_boundary))),
            "backward_boundary_fraction": float(np.mean(np.asarray(maps.backward_boundary))),
            "radial_shift_linf_cells": float(np.max(np.abs(forward_dx))),
            "poloidal_shift_mean_cells": float(np.mean(forward_dy)),
            "poloidal_shift_std_cells": float(np.std(forward_dy)),
        },
        "connection_length": {
            "minimum": float(np.min(connection)),
            "maximum": float(np.max(connection)),
            "mean": float(np.mean(connection)),
            "std": float(np.std(connection)),
            "outer_midplane_mean": float(np.mean(connection[-4:, :, :])),
        },
        "curvature": {
            "minimum": float(np.min(curvature)),
            "maximum": float(np.max(curvature)),
            "mean": float(np.mean(curvature)),
            "std": float(np.std(curvature)),
        },
        "magnetic_field": {
            "minimum": float(np.min(bmag)),
            "maximum": float(np.max(bmag)),
            "mirror_ratio": float((np.max(bmag) - np.min(bmag)) / np.mean(bmag)),
        },
        "rotational_transform": {
            "axis_proxy": float(np.mean(iota[0, :, :])),
            "edge_proxy": float(np.mean(iota[-1, :, :])),
        },
    }
    report["passed"] = (
        bool(metric_report["passed"])
        and report["magnetic_field"]["minimum"] > 0.0
        and report["connection_length"]["minimum"] > 0.0
        and report["map_diagnostics"]["radial_shift_linf_cells"] > 0.0
    )
    return report


def save_stellarator_fci_geometry_plot(
    geometry: SyntheticStellaratorGeometry,
    report: dict[str, object],
    path: str | Path,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(geometry.coordinates_x)
    y = np.asarray(geometry.coordinates_y)
    z = np.asarray(geometry.coordinates_z)
    r_major = np.sqrt(x * x + y * y)
    bmag = np.asarray(geometry.Bmag)
    connection = np.asarray(geometry.connection_length)
    curvature = np.asarray(geometry.curvature)
    maps = geometry
    radial_shift = np.asarray(maps.forward_x) - np.arange(maps.shape[0], dtype=np.float64)[:, None, None]
    theta = np.asarray(geometry.poloidal_angle[:, 0, :])
    radial = np.asarray(geometry.radial[:, 0, :])

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)
    toroidal_indices = [0, max(1, maps.shape[1] // 8), max(2, maps.shape[1] // 4), max(3, 3 * maps.shape[1] // 8)]
    colors = ["#005f73", "#0a9396", "#ee9b00", "#9b2226"]
    for index, color in zip(toroidal_indices, colors, strict=False):
        for surface in (maps.shape[0] // 4, maps.shape[0] // 2, maps.shape[0] - 2):
            axes[0, 0].plot(
                r_major[surface, index, :],
                z[surface, index, :],
                color=color,
                linewidth=1.3,
                alpha=0.75,
            )
    axes[0, 0].set_title("Rotating cross-sections and sampled flux surfaces")
    axes[0, 0].set_xlabel("major radius")
    axes[0, 0].set_ylabel("vertical coordinate")
    axes[0, 0].set_aspect("equal", adjustable="box")

    image0 = axes[0, 1].pcolormesh(
        theta,
        radial,
        bmag[:, 0, :],
        shading="auto",
        cmap="viridis",
    )
    axes[0, 1].set_title("Magnetic-field strength at one toroidal plane")
    axes[0, 1].set_xlabel("poloidal angle")
    axes[0, 1].set_ylabel("normalized radius")
    fig.colorbar(image0, ax=axes[0, 1], label="normalized |B|")

    image1 = axes[1, 0].pcolormesh(theta, radial, connection[:, 0, :], shading="auto", cmap="magma")
    axes[1, 0].set_title("Field-line connection-length proxy")
    axes[1, 0].set_xlabel("poloidal angle")
    axes[1, 0].set_ylabel("normalized radius")
    fig.colorbar(image1, ax=axes[1, 0], label="length proxy")

    image2 = axes[1, 1].pcolormesh(
        theta,
        radial,
        curvature[:, 0, :] * radial_shift[:, 0, :],
        shading="auto",
        cmap="coolwarm",
    )
    axes[1, 1].set_title("Curvature-weighted radial map displacement")
    axes[1, 1].set_xlabel("poloidal angle")
    axes[1, 1].set_ylabel("normalized radius")
    fig.colorbar(image2, ax=axes[1, 1], label="signed proxy")
    fig.suptitle(
        "3D non-axisymmetric geometry gate: "
        f"metric residual {float(report['metric']['inverse_residual_linf']):.1e}, "
        f"mirror ratio {float(report['magnetic_field']['mirror_ratio']):.2f}",
        fontsize=12,
    )
    fig.savefig(resolved, dpi=180)
    plt.close(fig)
    return resolved


def _write_geometry_arrays(geometry: SyntheticStellaratorGeometry, path: Path) -> Path:
    np.savez_compressed(
        path,
        x=np.asarray(geometry.coordinates_x, dtype=np.float32),
        y=np.asarray(geometry.coordinates_y, dtype=np.float32),
        z=np.asarray(geometry.coordinates_z, dtype=np.float32),
        radial=np.asarray(geometry.radial, dtype=np.float32),
        toroidal_angle=np.asarray(geometry.toroidal_angle, dtype=np.float32),
        poloidal_angle=np.asarray(geometry.poloidal_angle, dtype=np.float32),
        B_contravariant=np.asarray(geometry.B_contravariant, dtype=np.float32),
        J=np.asarray(geometry.J, dtype=np.float32),
        g_22=np.asarray(geometry.g_22, dtype=np.float32),
        curvature=np.asarray(geometry.curvature, dtype=np.float32),
        connection_length=np.asarray(geometry.connection_length, dtype=np.float32),
        forward_x=np.asarray(geometry.forward_x, dtype=np.float32),
        forward_y=np.asarray(geometry.forward_y, dtype=np.float32),
        backward_x=np.asarray(geometry.backward_x, dtype=np.float32),
        backward_y=np.asarray(geometry.backward_y, dtype=np.float32),
    )
    return path
