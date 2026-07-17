from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from ..geometry import (
    EssosFieldLineBundle,
    resolve_essos_landreman_qa_json,
    save_essos_field_line_bundle_npz,
    trace_essos_coil_field_lines,
)


@dataclass(frozen=True)
class EssosFieldLineImportArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_essos_fieldline_import_package(
    *,
    output_root: str | Path,
    coil_json_path: str | Path | None = None,
    case_label: str = "essos_landreman_paul_qa_fieldline_import",
    r_min: float = 1.21,
    r_max: float = 1.40,
    n_field_lines: int = 8,
    maxtime: float = 1000.0,
    times_to_trace: int = 6000,
    trace_tolerance: float = 1.0e-8,
) -> EssosFieldLineImportArtifacts:
    """Export ESSOS-produced fields and field lines into a `drbx` artifact."""

    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    resolved_coil_json = resolve_essos_landreman_qa_json(coil_json_path)
    bundle = trace_essos_coil_field_lines(
        coil_json_path=resolved_coil_json,
        r_min=r_min,
        r_max=r_max,
        n_field_lines=n_field_lines,
        maxtime=maxtime,
        times_to_trace=times_to_trace,
        trace_tolerance=trace_tolerance,
    )
    report = build_essos_fieldline_import_report(bundle)
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    arrays_npz_path = save_essos_field_line_bundle_npz(bundle, data_dir / f"{case_label}.npz")
    plot_png_path = save_essos_fieldline_import_plot(bundle, images_dir / f"{case_label}.png")
    return EssosFieldLineImportArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_essos_fieldline_import_report(bundle: EssosFieldLineBundle) -> dict[str, object]:
    """Build a sanitized report for an ESSOS field-line import bundle."""

    b_norm = np.linalg.norm(bundle.field_sample_b_xyz, axis=1)
    radius = np.sqrt(bundle.trajectories_xyz[:, :, 0] ** 2 + bundle.trajectories_xyz[:, :, 1] ** 2)
    radial_span = np.ptp(radius, axis=1)
    z_span = np.ptp(bundle.trajectories_xyz[:, :, 2], axis=1)
    report: dict[str, object] = {
        "case": "essos_landreman_paul_qa_fieldline_import",
        "source": "ESSOS external field and field-line import",
        "metadata": bundle.metadata,
        "n_field_lines": bundle.n_field_lines,
        "n_times": bundle.n_times,
        "poincare_point_count": bundle.poincare_point_count,
        "field_sample_count": int(bundle.field_sample_xyz.shape[0]),
        "coil_count": int(bundle.coil_gamma_xyz.shape[0]),
        "coil_segments": int(bundle.coil_gamma_xyz.shape[1]),
        "field_magnitude_min": float(np.min(b_norm)),
        "field_magnitude_mean": float(np.mean(b_norm)),
        "field_magnitude_max": float(np.max(b_norm)),
        "radial_span_mean": float(np.mean(radial_span)),
        "radial_span_max": float(np.max(radial_span)),
        "vertical_span_mean": float(np.mean(z_span)),
        "vertical_span_max": float(np.max(z_span)),
    }
    report["passed"] = bool(
        bundle.n_field_lines > 0
        and bundle.n_times > 10
        and bundle.poincare_point_count > 0
        and np.all(np.isfinite(bundle.trajectories_xyz))
        and np.all(np.isfinite(bundle.field_sample_b_xyz))
        and float(np.max(b_norm)) > float(np.min(b_norm)) > 0.0
    )
    return report


def save_essos_fieldline_import_plot(bundle: EssosFieldLineBundle, path: str | Path) -> Path:
    """Save a QA plot of ESSOS-imported coils, field lines, field samples, and Poincare data."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.0, 5.6), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axp = fig.add_subplot(1, 2, 2)

    for coil in bundle.coil_gamma_xyz:
        ax3d.plot(coil[:, 0], coil[:, 1], coil[:, 2], color="0.50", alpha=0.20, lw=0.6)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, bundle.n_field_lines))
    stride = max(1, bundle.n_times // 1500)
    for index, trajectory in enumerate(bundle.trajectories_xyz):
        ax3d.plot(
            trajectory[::stride, 0],
            trajectory[::stride, 1],
            trajectory[::stride, 2],
            color=colors[index],
            lw=0.8,
            alpha=0.88,
        )
    ax3d.scatter(
        bundle.initial_xyz[:, 0],
        bundle.initial_xyz[:, 1],
        bundle.initial_xyz[:, 2],
        s=18.0,
        color="black",
        depthshade=False,
        label="field-line seeds",
    )
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("ESSOS-exported coil field lines")
    ax3d.legend(frameon=False, loc="upper left")
    _set_equal_3d(ax3d, bundle.trajectories_xyz.reshape((-1, 3)))

    if bundle.poincare_point_count:
        scatter = axp.scatter(
            bundle.poincare_r,
            bundle.poincare_z,
            c=bundle.poincare_line_index,
            cmap="viridis",
            s=2.5,
            alpha=0.78,
            linewidths=0.0,
        )
        colorbar = fig.colorbar(scatter, ax=axp, fraction=0.045, pad=0.03)
        colorbar.set_label("field-line seed")
    axp.scatter(
        np.sqrt(bundle.initial_xyz[:, 0] ** 2 + bundle.initial_xyz[:, 1] ** 2),
        bundle.initial_xyz[:, 2],
        s=18.0,
        color="black",
        label="seeds",
        zorder=3,
    )
    axp.set_xlabel("R")
    axp.set_ylabel("Z")
    axp.set_aspect("equal", adjustable="box")
    axp.grid(alpha=0.28)
    axp.set_title("ESSOS Poincare sections imported into drbx")
    axp.legend(frameon=False, loc="upper right")

    fig.suptitle("External ESSOS field-line import for DRBX geometry/FCI workflows", fontsize=13)
    fig.savefig(resolved, dpi=190)
    plt.close(fig)
    return resolved


def _set_equal_3d(axis: plt.Axes, points: np.ndarray) -> None:
    center = np.mean(points, axis=0)
    span = float(np.max(np.ptp(points, axis=0)))
    span = max(span, 1.0e-12)
    half = 0.5 * span
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)
