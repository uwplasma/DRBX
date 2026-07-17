from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np

from ..geometry import (
    build_essos_vmec_scaled_qa_coordinates,
    load_essos_coil_field_axis,
    load_essos_vmec_field_axis,
    resolve_essos_landreman_qa_json,
    resolve_essos_landreman_qa_wout,
    trace_essos_coil_initial_conditions,
    trace_essos_vmec_initial_conditions,
)
from .publication_plotting import save_publication_figure, style_axis


@dataclass(frozen=True)
class EssosVmecFieldlineSurfaceArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


@dataclass(frozen=True)
class EssosVmecFieldlineSurfaceResult:
    report: dict[str, Any]
    arrays: dict[str, np.ndarray]


def create_essos_vmec_fieldline_surface_package(
    *,
    output_root: str | Path,
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    case_label: str = "essos_vmec_fieldline_surface_campaign",
    rho_min: float = 0.20,
    rho_max: float = 0.92,
    n_surfaces: int = 7,
    ntheta_surface: int = 320,
    maxtime: float = 900.0,
    times_to_trace: int = 4200,
    trace_tolerance: float = 1.0e-8,
    sections: tuple[float, ...] = (0.0, float(np.pi / 2.0), float(np.pi), float(3.0 * np.pi / 2.0)),
    field_source: str = "coil",
) -> EssosVmecFieldlineSurfaceArtifacts:
    """Trace external field lines and compare Poincare hits with scaled VMEC surfaces."""

    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    result = build_essos_vmec_fieldline_surface_campaign(
        coil_json_path=coil_json_path,
        vmec_wout_path=vmec_wout_path,
        essos_root=essos_root,
        rho_min=rho_min,
        rho_max=rho_max,
        n_surfaces=n_surfaces,
        ntheta_surface=ntheta_surface,
        maxtime=maxtime,
        times_to_trace=times_to_trace,
        trace_tolerance=trace_tolerance,
        sections=sections,
        field_source=field_source,
    )
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(result.report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **result.arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_essos_vmec_fieldline_surface_plot(result.report, result.arrays, plot_png_path)
    return EssosVmecFieldlineSurfaceArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_essos_vmec_fieldline_surface_campaign(
    *,
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    rho_min: float = 0.20,
    rho_max: float = 0.92,
    n_surfaces: int = 7,
    ntheta_surface: int = 320,
    maxtime: float = 900.0,
    times_to_trace: int = 4200,
    trace_tolerance: float = 1.0e-8,
    sections: tuple[float, ...] = (0.0, float(np.pi / 2.0), float(np.pi), float(3.0 * np.pi / 2.0)),
    field_source: str = "coil",
) -> EssosVmecFieldlineSurfaceResult:
    """Build the independent field-line/VMEC-surface registration gate."""

    if n_surfaces < 2:
        raise ValueError("field-line surface validation requires at least two surfaces")
    if ntheta_surface < 16:
        raise ValueError("field-line surface validation requires ntheta_surface >= 16")
    if not sections:
        raise ValueError("at least one Poincare section is required")

    field_source = field_source.lower().replace("-", "_")
    if field_source not in {"coil", "vmec"}:
        raise ValueError("field_source must be either 'coil' or 'vmec'")

    resolved_coil_json = resolve_essos_landreman_qa_json(coil_json_path, essos_root=essos_root)
    resolved_wout = resolve_essos_landreman_qa_wout(vmec_wout_path, essos_root=essos_root)
    axis_major_radius, axis_vertical = load_essos_coil_field_axis(coil_json_path=resolved_coil_json, essos_root=essos_root)
    vmec_axis_major_radius, vmec_axis_vertical = load_essos_vmec_field_axis(vmec_wout_path=resolved_wout, essos_root=essos_root)
    sections_array = np.mod(np.asarray(sections, dtype=np.float64), 2.0 * np.pi)

    seed_coordinates = build_essos_vmec_scaled_qa_coordinates(
        resolved_wout,
        nx=int(n_surfaces),
        ny=1,
        nz=max(int(ntheta_surface), 32),
        rho_min=float(rho_min),
        rho_max=float(rho_max),
        axis_major_radius=axis_major_radius,
        axis_vertical=axis_vertical,
    )
    surface_nphi = max(32, 4 * len(sections_array))
    surface_coordinates_full = build_essos_vmec_scaled_qa_coordinates(
        resolved_wout,
        nx=int(n_surfaces),
        ny=surface_nphi,
        nz=int(ntheta_surface),
        rho_min=float(rho_min),
        rho_max=float(rho_max),
        axis_major_radius=axis_major_radius,
        axis_vertical=axis_vertical,
    )
    section_indices = np.mod(
        np.rint(sections_array / (2.0 * np.pi) * float(surface_nphi)).astype(int),
        surface_nphi,
    )
    surface_major = np.asarray(surface_coordinates_full["major"][:, section_indices, :], dtype=np.float64)
    surface_vertical = np.asarray(surface_coordinates_full["vertical"][:, section_indices, :], dtype=np.float64)
    surface_phi = np.asarray(surface_coordinates_full["phi"][:, section_indices, :], dtype=np.float64)
    surface_theta = np.asarray(surface_coordinates_full["theta"][:, section_indices, :], dtype=np.float64)
    initial_xyz = np.stack(
        [
            seed_coordinates["x"][:, 0, 0],
            seed_coordinates["y"][:, 0, 0],
            seed_coordinates["z"][:, 0, 0],
        ],
        axis=-1,
    )
    current_sign = 1.0
    trajectories_stp = np.empty((0, 0, 3), dtype=np.float64)
    fieldline_s_drift_max = 0.0
    if field_source == "vmec":
        initial_stp = np.stack(
            [
                np.asarray(seed_coordinates["s_1d"], dtype=np.float64),
                np.zeros(int(n_surfaces), dtype=np.float64),
                np.zeros(int(n_surfaces), dtype=np.float64),
            ],
            axis=-1,
        )
        trajectories_stp = trace_essos_vmec_initial_conditions(
            initial_stp,
            vmec_wout_path=resolved_wout,
            essos_root=essos_root,
            maxtime=maxtime,
            times_to_trace=times_to_trace,
            trace_tolerance=trace_tolerance,
        )
        trajectories = _vmec_coordinate_hits_to_xyz_placeholder(trajectories_stp, initial_xyz)
        hits = _extract_vmec_coordinate_poincare_hits(
            trajectories_stp,
            sections_array,
            surface_major=surface_major,
            surface_vertical=surface_vertical,
            surface_theta=surface_theta,
        )
        fieldline_s_drift_max = float(np.max(np.abs(trajectories_stp[:, :, 0] - initial_stp[:, None, 0])))
    else:
        trajectories = trace_essos_coil_initial_conditions(
            initial_xyz,
            coil_json_path=resolved_coil_json,
            essos_root=essos_root,
            current_sign=1.0,
            maxtime=maxtime,
            times_to_trace=times_to_trace,
            trace_tolerance=trace_tolerance,
        )
        if _median_toroidal_advance(trajectories) < 0.0:
            current_sign = -1.0
            trajectories = trace_essos_coil_initial_conditions(
                initial_xyz,
                coil_json_path=resolved_coil_json,
                essos_root=essos_root,
                current_sign=current_sign,
                maxtime=maxtime,
                times_to_trace=times_to_trace,
                trace_tolerance=trace_tolerance,
            )
        hits = _extract_poincare_hits(trajectories, sections_array)
    same_surface_distance, nearest_surface_distance = _distance_to_reference_surfaces(
        hits,
        surface_major=surface_major,
        surface_vertical=surface_vertical,
    )
    reference_extent = _reference_minor_extent(
        surface_major[-1],
        surface_vertical[-1],
    )
    normalized_same = same_surface_distance / max(reference_extent, 1.0e-30)
    normalized_nearest = nearest_surface_distance / max(reference_extent, 1.0e-30)
    finite = (
        np.all(np.isfinite(trajectories))
        and np.all(np.isfinite(trajectories_stp))
        and np.all(np.isfinite(same_surface_distance))
        and np.all(np.isfinite(nearest_surface_distance))
    )
    point_count = int(hits["r"].size)
    same_p95 = _percentile_or_inf(normalized_same, 95.0)
    nearest_p95 = _percentile_or_inf(normalized_nearest, 95.0)
    nonaxisymmetric_rms = float(surface_coordinates_full["metadata"]["surface_nonaxisymmetric_major_rms"])
    if field_source == "vmec":
        source = "ESSOS VMEC field-line tracing compared with Landreman-Paul QA VMEC Fourier surfaces"
        claim_scope = (
            "surface-preserving VMEC-field Poincare diagnostic for the Landreman-Paul QA "
            "Fourier boundary used by the movie renderer; distances are evaluated on the "
            "scaled physics-grid copy"
        )
        same_threshold = 5.0e-2
        nearest_threshold = 5.0e-2
        s_drift_threshold = 1.0e-7
    else:
        source = "ESSOS coil field-line tracing compared with scaled Landreman-Paul QA VMEC surfaces"
        claim_scope = (
            "independent coil-field Poincare diagnostic for VMEC geometry registration; "
            "the strict closed-surface match flag is reported separately because the "
            "imported coil field can leave the scaled VMEC seed shell"
        )
        same_threshold = 0.35
        nearest_threshold = 0.20
        s_drift_threshold = float("inf")
    report: dict[str, Any] = {
        "case": "essos_vmec_fieldline_surface_campaign",
        "source": source,
        "claim_scope": claim_scope,
        "field_source": field_source,
        "coil_json_file": resolved_coil_json.name,
        "vmec_wout_file": resolved_wout.name,
        "coordinate_model": "scaled_vmec_fourier_flux_surfaces",
        "n_surfaces": int(n_surfaces),
        "ntheta_surface": int(ntheta_surface),
        "sections": [float(value) for value in sections_array],
        "maxtime": float(maxtime),
        "times_to_trace": int(times_to_trace),
        "trace_tolerance": float(trace_tolerance),
        "current_sign": float(current_sign),
        "poincare_point_count": point_count,
        "points_per_surface_mean": float(point_count / max(int(n_surfaces), 1)),
        "axis_major_radius": float(axis_major_radius),
        "axis_vertical": float(axis_vertical),
        "vmec_raw_axis_major_radius": float(vmec_axis_major_radius),
        "vmec_raw_axis_vertical": float(vmec_axis_vertical),
        "rho_min": float(rho_min),
        "rho_max": float(rho_max),
        "fieldline_s_drift_max": fieldline_s_drift_max,
        "reference_minor_extent": float(reference_extent),
        "surface_nonaxisymmetric_major_rms": nonaxisymmetric_rms,
        "same_surface_distance_median": _median_or_inf(same_surface_distance),
        "same_surface_distance_p95": _percentile_or_inf(same_surface_distance, 95.0),
        "same_surface_distance_max": _max_or_inf(same_surface_distance),
        "same_surface_distance_normalized_median": _median_or_inf(normalized_same),
        "same_surface_distance_normalized_p95": same_p95,
        "nearest_surface_distance_normalized_p95": nearest_p95,
        "nearest_surface_distance_normalized_max": _max_or_inf(normalized_nearest),
        "fieldline_radial_span_mean": _line_span_mean(hits["r"], hits["line_index"], int(n_surfaces)),
        "fieldline_vertical_span_mean": _line_span_mean(hits["z"], hits["line_index"], int(n_surfaces)),
    }
    report["fieldline_surface_match_passed"] = bool(
        finite
        and point_count >= max(2 * int(n_surfaces), 1)
        and nonaxisymmetric_rms > 5.0e-2
        and same_p95 < same_threshold
        and nearest_p95 < nearest_threshold
        and fieldline_s_drift_max < s_drift_threshold
    )
    report["passed"] = bool(finite and point_count >= max(2 * int(n_surfaces), 1) and nonaxisymmetric_rms > 5.0e-2)
    arrays = {
        "initial_xyz": initial_xyz.astype(np.float32),
        "trajectories_xyz": trajectories.astype(np.float32),
        "trajectories_stp": trajectories_stp.astype(np.float32),
        "surface_major": surface_major.astype(np.float32),
        "surface_vertical": surface_vertical.astype(np.float32),
        "surface_phi": surface_phi.astype(np.float32),
        "surface_theta": surface_theta.astype(np.float32),
        "rho_values": np.asarray(seed_coordinates["rho_1d"], dtype=np.float32),
        "sections": sections_array.astype(np.float32),
        "poincare_r": hits["r"].astype(np.float32),
        "poincare_z": hits["z"].astype(np.float32),
        "poincare_time_index": hits["time_index"].astype(np.float32),
        "poincare_section_index": hits["section_index"].astype(np.int32),
        "poincare_line_index": hits["line_index"].astype(np.int32),
        "same_surface_distance": same_surface_distance.astype(np.float32),
        "nearest_surface_distance": nearest_surface_distance.astype(np.float32),
        "same_surface_distance_normalized": normalized_same.astype(np.float32),
        "nearest_surface_distance_normalized": normalized_nearest.astype(np.float32),
    }
    return EssosVmecFieldlineSurfaceResult(report=report, arrays=arrays)


def save_essos_vmec_fieldline_surface_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Save the field-line/surface registration figure used to QA the movie geometry."""

    surface_major = np.asarray(arrays["surface_major"], dtype=np.float64)
    surface_vertical = np.asarray(arrays["surface_vertical"], dtype=np.float64)
    sections = np.asarray(arrays["sections"], dtype=np.float64)
    hit_r = np.asarray(arrays["poincare_r"], dtype=np.float64)
    hit_z = np.asarray(arrays["poincare_z"], dtype=np.float64)
    hit_section = np.asarray(arrays["poincare_section_index"], dtype=np.int32)
    hit_line = np.asarray(arrays["poincare_line_index"], dtype=np.int32)
    normalized_same = np.asarray(arrays["same_surface_distance_normalized"], dtype=np.float64)

    panel_count = min(len(sections), 4)
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.4), constrained_layout=True)
    flat_axes = axes.ravel()
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=0, vmax=max(surface_major.shape[0] - 1, 1))
    for section_index in range(panel_count):
        axis = flat_axes[section_index]
        for surface_index in range(surface_major.shape[0]):
            color = "0.82" if surface_index < surface_major.shape[0] - 1 else "black"
            width = 0.9 if surface_index < surface_major.shape[0] - 1 else 1.5
            axis.plot(
                surface_major[surface_index, section_index, :],
                surface_vertical[surface_index, section_index, :],
                color=color,
                lw=width,
                alpha=0.92,
            )
        mask = hit_section == section_index
        if np.any(mask):
            axis.scatter(
                hit_r[mask],
                hit_z[mask],
                c=hit_line[mask],
                s=5.0,
                cmap=cmap,
                norm=norm,
                linewidths=0.0,
                alpha=0.78,
            )
        axis.set_aspect("equal", adjustable="box")
        style_axis(
            axis,
            title=rf"Poincare section $\phi={sections[section_index]:.2f}$",
            xlabel="R",
            ylabel="Z",
            grid="both",
        )
    for empty_index in range(panel_count, 4):
        flat_axes[empty_index].axis("off")

    hist_axis = flat_axes[4]
    hist_axis.hist(normalized_same[np.isfinite(normalized_same)], bins=24, color="#0A6E7D", alpha=0.82)
    hist_axis.axvline(report["same_surface_distance_normalized_p95"], color="#CA6702", lw=2.0, label="95th percentile")
    style_axis(
        hist_axis,
        title="distance to seeded VMEC surface",
        xlabel="distance / outer-surface minor extent",
        ylabel="count",
        grid="y",
    )
    hist_axis.legend(frameon=False, fontsize=9)

    text_axis = flat_axes[5]
    text_axis.axis("off")
    text_axis.text(
        0.02,
        0.95,
        "\n".join(
            [
                "Independent geometry registration",
                f"points: {report['poincare_point_count']}",
                f"non-axisymmetric RMS: {report['surface_nonaxisymmetric_major_rms']:.3f}",
                f"same-surface p95: {report['same_surface_distance_normalized_p95']:.3f}",
                f"nearest-surface p95: {report['nearest_surface_distance_normalized_p95']:.3f}",
                f"radial span mean: {report['fieldline_radial_span_mean']:.3e}",
                f"vertical span mean: {report['fieldline_vertical_span_mean']:.3e}",
                f"passed: {report['passed']}",
            ]
        ),
        transform=text_axis.transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.96},
    )
    title_prefix = "VMEC-field" if report.get("field_source") == "vmec" else "QA-coil"
    fig.suptitle(
        f"{title_prefix} field-line Poincare hits over scaled Landreman-Paul VMEC surfaces",
        fontsize=15,
        fontweight="semibold",
    )
    save_publication_figure(fig, path)
    return Path(path)


def _extract_poincare_hits(trajectories_xyz: np.ndarray, sections: np.ndarray) -> dict[str, np.ndarray]:
    r_values: list[float] = []
    z_values: list[float] = []
    time_values: list[float] = []
    section_values: list[int] = []
    line_values: list[int] = []
    period = 2.0 * np.pi
    for line_index, trajectory in enumerate(np.asarray(trajectories_xyz, dtype=np.float64)):
        major = np.sqrt(trajectory[:, 0] ** 2 + trajectory[:, 1] ** 2)
        vertical = trajectory[:, 2]
        phi = np.unwrap(np.arctan2(trajectory[:, 1], trajectory[:, 0]))
        if not np.all(np.isfinite(phi)) or phi.size < 3:
            continue
        if phi[-1] < phi[0]:
            phi = phi[::-1]
            major = major[::-1]
            vertical = vertical[::-1]
        phi_min = float(phi[0])
        phi_max = float(phi[-1])
        for section_index, section in enumerate(sections):
            k_min = int(np.floor((phi_min - float(section)) / period)) - 1
            k_max = int(np.ceil((phi_max - float(section)) / period)) + 1
            for k_value in range(k_min, k_max + 1):
                target = float(section) + period * float(k_value)
                if target <= phi_min + 1.0e-10 or target >= phi_max - 1.0e-10:
                    continue
                r_values.append(float(np.interp(target, phi, major)))
                z_values.append(float(np.interp(target, phi, vertical)))
                time_values.append(float(np.interp(target, phi, np.arange(phi.size, dtype=np.float64))))
                section_values.append(int(section_index))
                line_values.append(int(line_index))
    return {
        "r": np.asarray(r_values, dtype=np.float64),
        "z": np.asarray(z_values, dtype=np.float64),
        "time_index": np.asarray(time_values, dtype=np.float64),
        "section_index": np.asarray(section_values, dtype=np.int32),
        "line_index": np.asarray(line_values, dtype=np.int32),
    }


def _vmec_coordinate_hits_to_xyz_placeholder(trajectories_stp: np.ndarray, initial_xyz: np.ndarray) -> np.ndarray:
    """Return finite Cartesian placeholders for VMEC-coordinate field-line traces.

    The validation plot and distance metrics for VMEC-coordinate traces are
    computed directly from the VMEC coordinates and the sampled surface table.
    A finite Cartesian array is still stored so the artifact schema remains
    compatible with the coil-trace package.
    """

    stp = np.asarray(trajectories_stp, dtype=np.float64)
    seeds = np.asarray(initial_xyz, dtype=np.float64)
    return np.broadcast_to(seeds[:, None, :], stp.shape).copy()


def _extract_vmec_coordinate_poincare_hits(
    trajectories_stp: np.ndarray,
    sections: np.ndarray,
    *,
    surface_major: np.ndarray,
    surface_vertical: np.ndarray,
    surface_theta: np.ndarray,
) -> dict[str, np.ndarray]:
    r_values: list[float] = []
    z_values: list[float] = []
    time_values: list[float] = []
    section_values: list[int] = []
    line_values: list[int] = []
    theta_values: list[float] = []
    s_values: list[float] = []
    period = 2.0 * np.pi
    for line_index, trajectory in enumerate(np.asarray(trajectories_stp, dtype=np.float64)):
        if trajectory.shape[0] < 3:
            continue
        s_coord = trajectory[:, 0]
        theta = trajectory[:, 1]
        phi = np.unwrap(trajectory[:, 2])
        if not np.all(np.isfinite(phi)):
            continue
        sample_index = np.arange(phi.size, dtype=np.float64)
        if phi[-1] < phi[0]:
            s_coord = s_coord[::-1]
            theta = theta[::-1]
            phi = phi[::-1]
            sample_index = sample_index[::-1]
        phi_min = float(phi[0])
        phi_max = float(phi[-1])
        for section_index, section in enumerate(sections):
            k_min = int(np.floor((phi_min - float(section)) / period)) - 1
            k_max = int(np.ceil((phi_max - float(section)) / period)) + 1
            for k_value in range(k_min, k_max + 1):
                target = float(section) + period * float(k_value)
                if target <= phi_min + 1.0e-10 or target >= phi_max - 1.0e-10:
                    continue
                theta_value = float(np.interp(target, phi, theta))
                r_value, z_value = _interpolate_surface_section_at_theta(
                    theta_value,
                    surface_major[int(line_index), int(section_index), :],
                    surface_vertical[int(line_index), int(section_index), :],
                    surface_theta[int(line_index), int(section_index), :],
                )
                r_values.append(r_value)
                z_values.append(z_value)
                time_values.append(float(np.interp(target, phi, sample_index)))
                section_values.append(int(section_index))
                line_values.append(int(line_index))
                theta_values.append(theta_value)
                s_values.append(float(np.interp(target, phi, s_coord)))
    return {
        "r": np.asarray(r_values, dtype=np.float64),
        "z": np.asarray(z_values, dtype=np.float64),
        "time_index": np.asarray(time_values, dtype=np.float64),
        "section_index": np.asarray(section_values, dtype=np.int32),
        "line_index": np.asarray(line_values, dtype=np.int32),
        "theta": np.asarray(theta_values, dtype=np.float64),
        "s": np.asarray(s_values, dtype=np.float64),
    }


def _interpolate_surface_section_at_theta(
    theta_value: float,
    major: np.ndarray,
    vertical: np.ndarray,
    theta: np.ndarray,
) -> tuple[float, float]:
    theta_mod = float(np.mod(theta_value, 2.0 * np.pi))
    theta_curve = np.mod(np.asarray(theta, dtype=np.float64), 2.0 * np.pi)
    order = np.argsort(theta_curve)
    theta_sorted = theta_curve[order]
    major_sorted = np.asarray(major, dtype=np.float64)[order]
    vertical_sorted = np.asarray(vertical, dtype=np.float64)[order]
    theta_ext = np.concatenate([theta_sorted, theta_sorted[:1] + 2.0 * np.pi])
    major_ext = np.concatenate([major_sorted, major_sorted[:1]])
    vertical_ext = np.concatenate([vertical_sorted, vertical_sorted[:1]])
    return float(np.interp(theta_mod, theta_ext, major_ext)), float(np.interp(theta_mod, theta_ext, vertical_ext))


def _distance_to_reference_surfaces(
    hits: dict[str, np.ndarray],
    *,
    surface_major: np.ndarray,
    surface_vertical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    same_distance = np.empty(hits["r"].shape, dtype=np.float64)
    nearest_distance = np.empty_like(same_distance)
    for index, (r_value, z_value, section_index, line_index) in enumerate(
        zip(hits["r"], hits["z"], hits["section_index"], hits["line_index"], strict=True)
    ):
        section = int(np.clip(section_index, 0, surface_major.shape[1] - 1))
        line = int(np.clip(line_index, 0, surface_major.shape[0] - 1))
        same_curve_distance = np.sqrt(
            (surface_major[line, section, :] - float(r_value)) ** 2
            + (surface_vertical[line, section, :] - float(z_value)) ** 2
        )
        all_curve_distance = np.sqrt(
            (surface_major[:, section, :] - float(r_value)) ** 2
            + (surface_vertical[:, section, :] - float(z_value)) ** 2
        )
        same_distance[index] = float(np.min(same_curve_distance))
        nearest_distance[index] = float(np.min(all_curve_distance))
    return same_distance, nearest_distance


def _reference_minor_extent(surface_major: np.ndarray, surface_vertical: np.ndarray) -> float:
    center_major = np.mean(surface_major, axis=1, keepdims=True)
    radius = np.sqrt((surface_major - center_major) ** 2 + surface_vertical * surface_vertical)
    return float(np.sqrt(np.mean(radius * radius)))


def _median_toroidal_advance(trajectories_xyz: np.ndarray) -> float:
    phi = np.unwrap(np.arctan2(trajectories_xyz[:, :, 1], trajectories_xyz[:, :, 0]), axis=1)
    return float(np.nanmedian(phi[:, -1] - phi[:, 0]))


def _line_span_mean(values: np.ndarray, line_indices: np.ndarray, n_lines: int) -> float:
    spans: list[float] = []
    values_array = np.asarray(values, dtype=np.float64)
    line_array = np.asarray(line_indices, dtype=np.int32)
    for line_index in range(int(n_lines)):
        selected = values_array[line_array == line_index]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            spans.append(float(np.ptp(selected)))
    return float(np.mean(spans)) if spans else 0.0


def _median_or_inf(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("inf")


def _percentile_or_inf(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float("inf")


def _max_or_inf(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else float("inf")
