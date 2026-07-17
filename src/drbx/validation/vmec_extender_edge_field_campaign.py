from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from ..geometry.vmec_extender_import import (
    VmecExtenderGrid,
    interpolate_vmec_extender_B_cyl,
    load_vmec_extender_grid_netcdf,
    vmec_extender_fieldline_rhs_RZ_phi,
)
from .publication_plotting import annotate_bars, save_publication_figure, style_axis


@dataclass(frozen=True)
class VmecExtenderEdgeFieldCampaignArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_vmec_extender_edge_field_campaign_package(
    *,
    output_root: str | Path,
    field_grid_path: str | Path,
    case_label: str = "vmec_extender_edge_field_campaign",
    strict_metadata: bool = True,
) -> VmecExtenderEdgeFieldCampaignArtifacts:
    """Run the imported-field verification campaign and write public artifacts."""

    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_vmec_extender_edge_field_campaign_report(
        field_grid_path=field_grid_path,
        strict_metadata=strict_metadata,
    )
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(
        arrays_npz_path,
        metric_names=np.asarray(
            [
                "node_interpolation_max_abs_error",
                "midpoint_interpolation_max_abs_error",
                "field_period_relative_l2",
                "fieldline_rhs_max_abs_error",
                "absB_consistency_max_abs_error",
            ],
            dtype=object,
        ),
        metric_values=np.asarray(
            [
                report["node_interpolation_max_abs_error"],
                report["midpoint_interpolation_max_abs_error"],
                report["field_period_relative_l2"],
                report["fieldline_rhs_max_abs_error"],
                report["absB_consistency_max_abs_error"],
            ],
            dtype=np.float64,
        ),
        node_points=np.asarray(report["node_sample_points"], dtype=np.float64),
        node_expected_B=np.asarray(report["node_sample_expected_B"], dtype=np.float64),
        node_interpolated_B=np.asarray(report["node_sample_interpolated_B"], dtype=np.float64),
    )

    plot_png_path = save_vmec_extender_edge_field_campaign_plot(report, images_dir / f"{case_label}.png")
    return VmecExtenderEdgeFieldCampaignArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_vmec_extender_edge_field_campaign_report(
    *,
    field_grid_path: str | Path,
    strict_metadata: bool = True,
) -> dict[str, object]:
    """Build a numerical verification report for an imported edge-field grid."""

    grid = load_vmec_extender_grid_netcdf(field_grid_path, strict_metadata=strict_metadata)
    node_points = _node_points(grid)
    node_expected = _field_stack(grid)
    node_interpolated = np.asarray(interpolate_vmec_extender_B_cyl(grid, node_points), dtype=np.float64)
    node_error = float(np.max(np.abs(node_interpolated - node_expected)))

    midpoint_points, midpoint_expected = _midpoint_trilinear_reference(grid)
    midpoint_interpolated = np.asarray(interpolate_vmec_extender_B_cyl(grid, midpoint_points), dtype=np.float64)
    midpoint_error = float(np.max(np.abs(midpoint_interpolated - midpoint_expected))) if midpoint_expected.size else 0.0

    period_points = _periodicity_probe_points(grid)
    period_base = np.asarray(interpolate_vmec_extender_B_cyl(grid, period_points), dtype=np.float64)
    shifted_points = period_points.copy()
    shifted_points[..., 1] = shifted_points[..., 1] + float(grid.phi_period)
    period_shifted = np.asarray(interpolate_vmec_extender_B_cyl(grid, shifted_points), dtype=np.float64)
    period_relative_l2 = _relative_l2(period_base, period_shifted)

    rhs_points = midpoint_points.reshape((-1, 3))[: max(1, min(32, midpoint_points.reshape((-1, 3)).shape[0]))]
    rhs_B = np.asarray(interpolate_vmec_extender_B_cyl(grid, rhs_points), dtype=np.float64)
    rhs_expected = np.stack(
        (
            rhs_points[:, 0] * rhs_B[:, 0] / rhs_B[:, 1],
            rhs_points[:, 0] * rhs_B[:, 2] / rhs_B[:, 1],
        ),
        axis=-1,
    )
    rhs_actual = np.asarray(vmec_extender_fieldline_rhs_RZ_phi(grid, rhs_points), dtype=np.float64)
    rhs_error = float(np.max(np.abs(rhs_actual - rhs_expected)))

    absB_error = float(grid.metadata.get("absB_consistency_max_abs_error", np.nan))
    passed = (
        node_error <= 1.0e-10
        and midpoint_error <= 1.0e-10
        and period_relative_l2 <= 1.0e-12
        and rhs_error <= 1.0e-10
        and absB_error <= 1.0e-8
    )
    sample_count = min(8, node_points.reshape((-1, 3)).shape[0])
    return {
        "family": "vmec_extender_edge_field_campaign",
        "case": Path(field_grid_path).stem,
        "source": str(grid.metadata.get("source", "unknown")),
        "grid_shape": [int(value) for value in grid.shape],
        "nfp": int(grid.nfp),
        "phi_period": float(grid.phi_period),
        "metadata_passed": True,
        "node_interpolation_max_abs_error": node_error,
        "midpoint_interpolation_max_abs_error": midpoint_error,
        "field_period_relative_l2": period_relative_l2,
        "fieldline_rhs_max_abs_error": rhs_error,
        "absB_consistency_max_abs_error": absB_error,
        "passed": bool(passed),
        "notes": (
            "The campaign verifies the gridded VMEC-extender import contract, "
            "physical-phi wrapping, trilinear interpolation, and field-line RHS."
        ),
        "node_sample_points": node_points.reshape((-1, 3))[:sample_count].tolist(),
        "node_sample_expected_B": node_expected.reshape((-1, 3))[:sample_count].tolist(),
        "node_sample_interpolated_B": node_interpolated.reshape((-1, 3))[:sample_count].tolist(),
    }


def save_vmec_extender_edge_field_campaign_plot(report: dict[str, object], path: str | Path) -> Path:
    """Save a compact publication-style summary plot for the import campaign."""

    metrics = np.asarray(
        [
            report["node_interpolation_max_abs_error"],
            report["midpoint_interpolation_max_abs_error"],
            report["field_period_relative_l2"],
            report["fieldline_rhs_max_abs_error"],
            report["absB_consistency_max_abs_error"],
        ],
        dtype=np.float64,
    )
    labels = [
        "node\ninterp",
        "midpoint\ninterp",
        "field-period\nwrap",
        "field-line\nRHS",
        "|B|\nclosure",
    ]
    floor = np.maximum(metrics, 1.0e-16)
    colors = ["#0a9396" if bool(report["passed"]) else "#bb3e03"] * len(metrics)
    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    x = np.arange(len(metrics), dtype=np.float64)
    axis.bar(x, floor, color=colors, width=0.66, alpha=0.92)
    axis.set_xticks(x, labels)
    style_axis(
        axis,
        title="VMEC-extender edge-field import",
        ylabel="error metric",
        yscale="log",
    )
    axis.tick_params(axis="x", labelsize=8.5)
    axis.tick_params(axis="y", labelsize=8.5)
    axis.set_title("VMEC-extender edge-field import", fontsize=11.0, fontweight="semibold", pad=7.0)
    axis.set_ylabel("error metric", fontsize=9.5)
    annotate_bars(axis, x, floor, fmt="{:.1e}", fontsize=7.5)
    axis.text(
        0.98,
        0.95,
        f"shape={tuple(report['grid_shape'])}\nnfp={report['nfp']}, period={float(report['phi_period']):.4f}",
        transform=axis.transAxes,
        va="top",
        ha="right",
        fontsize=8.0,
        linespacing=1.25,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d9d9d9", "linewidth": 0.6},
    )
    save_publication_figure(figure, path)
    return Path(path)


def _node_points(grid: VmecExtenderGrid) -> np.ndarray:
    R, phi, Z = np.meshgrid(np.asarray(grid.R), np.asarray(grid.phi), np.asarray(grid.Z), indexing="ij")
    return np.stack((R, phi, Z), axis=-1)


def _field_stack(grid: VmecExtenderGrid) -> np.ndarray:
    return np.stack(
        (
            np.asarray(grid.BR, dtype=np.float64),
            np.asarray(grid.Bphi, dtype=np.float64),
            np.asarray(grid.BZ, dtype=np.float64),
        ),
        axis=-1,
    )


def _midpoint_trilinear_reference(grid: VmecExtenderGrid) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(grid.R, dtype=np.float64)
    phi = np.asarray(grid.phi, dtype=np.float64)
    Z = np.asarray(grid.Z, dtype=np.float64)
    if min(R.size, phi.size, Z.size) < 2:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    R_mid = 0.5 * (R[:-1] + R[1:])
    phi_mid = 0.5 * (phi[:-1] + phi[1:])
    Z_mid = 0.5 * (Z[:-1] + Z[1:])
    RR, PP, ZZ = np.meshgrid(R_mid, phi_mid, Z_mid, indexing="ij")
    points = np.stack((RR, PP, ZZ), axis=-1)
    fields = _field_stack(grid)
    expected = 0.125 * (
        fields[:-1, :-1, :-1, :]
        + fields[1:, :-1, :-1, :]
        + fields[:-1, 1:, :-1, :]
        + fields[1:, 1:, :-1, :]
        + fields[:-1, :-1, 1:, :]
        + fields[1:, :-1, 1:, :]
        + fields[:-1, 1:, 1:, :]
        + fields[1:, 1:, 1:, :]
    )
    return points, expected


def _periodicity_probe_points(grid: VmecExtenderGrid) -> np.ndarray:
    R = np.asarray(grid.R, dtype=np.float64)
    phi = np.asarray(grid.phi, dtype=np.float64)
    Z = np.asarray(grid.Z, dtype=np.float64)
    R_probe = 0.5 * (R[:-1] + R[1:])
    phi_probe = phi[: max(1, min(3, phi.size))]
    Z_probe = 0.5 * (Z[:-1] + Z[1:])
    RR, PP, ZZ = np.meshgrid(R_probe, phi_probe, Z_probe, indexing="ij")
    return np.stack((RR, PP, ZZ), axis=-1)


def _relative_l2(expected: np.ndarray, actual: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.ravel(actual - expected)))
    denominator = float(max(np.linalg.norm(np.ravel(expected)), np.finfo(np.float64).tiny))
    return numerator / denominator
