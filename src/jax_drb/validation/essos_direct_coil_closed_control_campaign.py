from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np

from ..geometry import (
    build_essos_vmec_scaled_qa_coordinates,
    load_essos_coil_field_axis,
    resolve_essos_landreman_qa_json,
    resolve_essos_landreman_qa_wout,
    trace_essos_coil_initial_conditions,
)
from .publication_plotting import save_publication_figure, style_axis


@dataclass(frozen=True)
class EssosDirectCoilClosedControlArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


@dataclass(frozen=True)
class EssosDirectCoilClosedControlResult:
    report: dict[str, Any]
    arrays: dict[str, np.ndarray]


def create_essos_direct_coil_closed_control_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_direct_coil_closed_control",
    use_live_essos: bool = False,
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    trajectories_xyz: np.ndarray | None = None,
    initial_xyz: np.ndarray | None = None,
    times: np.ndarray | None = None,
    coil_gamma_xyz: np.ndarray | None = None,
    rho_min: float = 0.20,
    rho_max: float = 0.82,
    n_radial_seeds: int = 5,
    n_poloidal_seeds: int = 4,
    maxtime: float = 900.0,
    times_to_trace: int = 4200,
    trace_tolerance: float = 1.0e-8,
    poincare_sections: tuple[float, ...] = (0.0, float(np.pi / 2.0), float(np.pi), float(3.0 * np.pi / 2.0)),
    closed_return_tolerance: float = 3.0e-2,
    near_closed_return_tolerance: float = 1.5e-1,
    minimum_closed_or_near_fraction: float = 0.20,
) -> EssosDirectCoilClosedControlArtifacts:
    """Write the direct-coil closed/near-closed field-line control artifact.

    The default mode is self-contained and uses manufactured non-axisymmetric
    traces. Set ``use_live_essos=True`` to trace the Landreman-Paul QA direct
    coil field with ESSOS.
    """

    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    result = build_essos_direct_coil_closed_control_campaign(
        use_live_essos=use_live_essos,
        coil_json_path=coil_json_path,
        vmec_wout_path=vmec_wout_path,
        essos_root=essos_root,
        trajectories_xyz=trajectories_xyz,
        initial_xyz=initial_xyz,
        times=times,
        coil_gamma_xyz=coil_gamma_xyz,
        rho_min=rho_min,
        rho_max=rho_max,
        n_radial_seeds=n_radial_seeds,
        n_poloidal_seeds=n_poloidal_seeds,
        maxtime=maxtime,
        times_to_trace=times_to_trace,
        trace_tolerance=trace_tolerance,
        poincare_sections=poincare_sections,
        closed_return_tolerance=closed_return_tolerance,
        near_closed_return_tolerance=near_closed_return_tolerance,
        minimum_closed_or_near_fraction=minimum_closed_or_near_fraction,
    )
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(result.report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **result.arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_essos_direct_coil_closed_control_plot(result.report, result.arrays, plot_png_path)
    return EssosDirectCoilClosedControlArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_essos_direct_coil_closed_control_campaign(
    *,
    use_live_essos: bool = False,
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    trajectories_xyz: np.ndarray | None = None,
    initial_xyz: np.ndarray | None = None,
    times: np.ndarray | None = None,
    coil_gamma_xyz: np.ndarray | None = None,
    rho_min: float = 0.20,
    rho_max: float = 0.82,
    n_radial_seeds: int = 5,
    n_poloidal_seeds: int = 4,
    maxtime: float = 900.0,
    times_to_trace: int = 4200,
    trace_tolerance: float = 1.0e-8,
    poincare_sections: tuple[float, ...] = (0.0, float(np.pi / 2.0), float(np.pi), float(3.0 * np.pi / 2.0)),
    closed_return_tolerance: float = 3.0e-2,
    near_closed_return_tolerance: float = 1.5e-1,
    minimum_closed_or_near_fraction: float = 0.20,
) -> EssosDirectCoilClosedControlResult:
    """Build the direct-coil closed/near-closed control report and arrays."""

    if use_live_essos and trajectories_xyz is not None:
        raise ValueError("Provide either live ESSOS inputs or trajectories_xyz, not both.")
    if use_live_essos:
        inputs = _build_live_essos_direct_coil_closed_control_inputs(
            coil_json_path=coil_json_path,
            vmec_wout_path=vmec_wout_path,
            essos_root=essos_root,
            rho_min=rho_min,
            rho_max=rho_max,
            n_radial_seeds=n_radial_seeds,
            n_poloidal_seeds=n_poloidal_seeds,
            maxtime=maxtime,
            times_to_trace=times_to_trace,
            trace_tolerance=trace_tolerance,
        )
    elif trajectories_xyz is None:
        inputs = _manufactured_direct_coil_closed_control_inputs(
            n_radial_seeds=n_radial_seeds,
            n_poloidal_seeds=n_poloidal_seeds,
            times_to_trace=max(256, int(times_to_trace // 8)),
        )
    else:
        if initial_xyz is None:
            raise ValueError("initial_xyz is required when trajectories_xyz is supplied.")
        trajectories = np.asarray(trajectories_xyz, dtype=np.float64)
        times_array = (
            np.arange(trajectories.shape[1], dtype=np.float64)
            if times is None
            else np.asarray(times, dtype=np.float64)
        )
        inputs = {
            "source": "user supplied direct-coil closed-control traces",
            "live_essos": False,
            "trajectories_xyz": trajectories,
            "initial_xyz": np.asarray(initial_xyz, dtype=np.float64),
            "times": times_array,
            "coil_gamma_xyz": (
                np.empty((0, 0, 3), dtype=np.float64)
                if coil_gamma_xyz is None
                else np.asarray(coil_gamma_xyz, dtype=np.float64)
            ),
            "metadata": {},
        }
    return _build_closed_control_from_inputs(
        inputs,
        poincare_sections=poincare_sections,
        closed_return_tolerance=closed_return_tolerance,
        near_closed_return_tolerance=near_closed_return_tolerance,
        minimum_closed_or_near_fraction=minimum_closed_or_near_fraction,
    )


def save_essos_direct_coil_closed_control_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Save the direct-coil closed/near-closed control QA figure."""

    trajectories = np.asarray(arrays["trajectories_xyz"], dtype=np.float64)
    initial_xyz = np.asarray(arrays["initial_xyz"], dtype=np.float64)
    coil_gamma = np.asarray(arrays.get("coil_gamma_xyz", np.empty((0, 0, 3))), dtype=np.float64)
    line_label = np.asarray(arrays["line_classification"], dtype=np.int32)
    return_distance = np.asarray(arrays["line_return_distance_normalized"], dtype=np.float64)
    poincare_r = np.asarray(arrays["poincare_r"], dtype=np.float64)
    poincare_z = np.asarray(arrays["poincare_z"], dtype=np.float64)
    poincare_line = np.asarray(arrays["poincare_line_index"], dtype=np.int32)
    sections = np.asarray(arrays["poincare_sections"], dtype=np.float64)

    fig = plt.figure(figsize=(14.2, 8.4), constrained_layout=True)
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    axp = fig.add_subplot(2, 2, 2)
    axh = fig.add_subplot(2, 2, 3)
    axt = fig.add_subplot(2, 2, 4)

    for coil in coil_gamma[:24]:
        if coil.size:
            ax3d.plot(coil[:, 0], coil[:, 1], coil[:, 2], color="0.62", lw=0.45, alpha=0.22)
    colors = _line_label_colors(line_label)
    stride = max(1, trajectories.shape[1] // 1000)
    for index, trajectory in enumerate(trajectories):
        ax3d.plot(
            trajectory[::stride, 0],
            trajectory[::stride, 1],
            trajectory[::stride, 2],
            color=colors[index],
            lw=0.9,
            alpha=0.84,
        )
    ax3d.scatter(initial_xyz[:, 0], initial_xyz[:, 1], initial_xyz[:, 2], s=12.0, color="black", depthshade=False)
    ax3d.set_title("direct-coil trace control")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    _set_equal_3d(ax3d, trajectories.reshape((-1, 3)))

    if poincare_r.size:
        scatter = axp.scatter(
            poincare_r,
            poincare_z,
            c=poincare_line,
            s=4.0,
            cmap="viridis",
            linewidths=0.0,
            alpha=0.78,
        )
        colorbar = fig.colorbar(scatter, ax=axp, fraction=0.046, pad=0.03)
        colorbar.set_label("seed index")
    axp.scatter(
        np.sqrt(initial_xyz[:, 0] ** 2 + initial_xyz[:, 1] ** 2),
        initial_xyz[:, 2],
        s=12.0,
        color="black",
        label="seeds",
        zorder=4,
    )
    section_text = ", ".join(f"{value:.2f}" for value in sections[:4])
    style_axis(
        axp,
        title=f"Poincare sections ({section_text})",
        xlabel="R",
        ylabel="Z",
        grid="both",
    )
    axp.set_aspect("equal", adjustable="box")
    axp.legend(frameon=False, fontsize=8)

    finite_return = return_distance[np.isfinite(return_distance)]
    if finite_return.size:
        axh.hist(finite_return, bins=24, color="#0A6E7D", alpha=0.82)
    axh.axvline(report["closed_return_tolerance"], color="#2A9D8F", lw=2.0, label="closed")
    axh.axvline(report["near_closed_return_tolerance"], color="#CA6702", lw=2.0, label="near-closed")
    style_axis(
        axh,
        title="minimum same-section return distance",
        xlabel="distance / reference minor extent",
        ylabel="seed count",
        grid="y",
    )
    axh.legend(frameon=False, fontsize=9)

    axt.axis("off")
    axt.text(
        0.02,
        0.95,
        "\n".join(
            [
                "Closed/near-closed diagnostic control",
                f"source: {report['source_mode']}",
                f"seeds: {report['n_field_lines']}",
                f"Poincare points: {report['poincare_point_count']}",
                f"turns mean: {report['toroidal_turns_mean']:.2f}",
                f"return p50: {report['return_distance_normalized_median']:.3e}",
                f"return p95: {report['return_distance_normalized_p95']:.3e}",
                f"closed fraction: {report['closed_fraction']:.2f}",
                f"near-closed fraction: {report['near_closed_fraction']:.2f}",
                f"closed-control passed: {report['closed_control_passed']}",
                "No target, sheath, or recycling semantics are applied.",
            ]
        ),
        transform=axt.transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.96},
    )
    fig.suptitle("ESSOS direct-coil closed/near-closed field-line control", fontsize=15, fontweight="semibold")
    save_publication_figure(fig, path)
    return Path(path)


def _build_live_essos_direct_coil_closed_control_inputs(
    *,
    coil_json_path: str | Path | None,
    vmec_wout_path: str | Path | None,
    essos_root: str | Path | None,
    rho_min: float,
    rho_max: float,
    n_radial_seeds: int,
    n_poloidal_seeds: int,
    maxtime: float,
    times_to_trace: int,
    trace_tolerance: float,
) -> dict[str, Any]:
    resolved_coil_json = resolve_essos_landreman_qa_json(coil_json_path, essos_root=essos_root)
    resolved_wout = resolve_essos_landreman_qa_wout(vmec_wout_path, essos_root=essos_root)
    axis_major_radius, axis_vertical = load_essos_coil_field_axis(
        coil_json_path=resolved_coil_json,
        essos_root=essos_root,
    )
    coordinates = build_essos_vmec_scaled_qa_coordinates(
        resolved_wout,
        nx=int(n_radial_seeds),
        ny=1,
        nz=int(n_poloidal_seeds),
        rho_min=float(rho_min),
        rho_max=float(rho_max),
        axis_major_radius=axis_major_radius,
        axis_vertical=axis_vertical,
    )
    initial_xyz = np.stack(
        [
            coordinates["x"].reshape(-1),
            coordinates["y"].reshape(-1),
            coordinates["z"].reshape(-1),
        ],
        axis=-1,
    )
    trajectories = trace_essos_coil_initial_conditions(
        initial_xyz,
        coil_json_path=resolved_coil_json,
        essos_root=essos_root,
        current_sign=1.0,
        maxtime=float(maxtime),
        times_to_trace=int(times_to_trace),
        trace_tolerance=float(trace_tolerance),
    )
    current_sign = 1.0
    if _median_toroidal_advance(trajectories) < 0.0:
        current_sign = -1.0
        trajectories = trace_essos_coil_initial_conditions(
            initial_xyz,
            coil_json_path=resolved_coil_json,
            essos_root=essos_root,
            current_sign=current_sign,
            maxtime=float(maxtime),
            times_to_trace=int(times_to_trace),
            trace_tolerance=float(trace_tolerance),
        )
    return {
        "source": "ESSOS direct Biot-Savart coil field-line trace",
        "live_essos": True,
        "trajectories_xyz": trajectories,
        "initial_xyz": initial_xyz,
        "times": np.linspace(0.0, float(maxtime), int(trajectories.shape[1]), dtype=np.float64),
        "coil_gamma_xyz": np.empty((0, 0, 3), dtype=np.float64),
        "metadata": {
            "coil_json_file": resolved_coil_json.name,
            "vmec_wout_file": resolved_wout.name,
            "rho_min": float(rho_min),
            "rho_max": float(rho_max),
            "n_radial_seeds": int(n_radial_seeds),
            "n_poloidal_seeds": int(n_poloidal_seeds),
            "maxtime": float(maxtime),
            "times_to_trace": int(times_to_trace),
            "trace_tolerance": float(trace_tolerance),
            "current_sign": float(current_sign),
        },
    }


def _build_closed_control_from_inputs(
    inputs: dict[str, Any],
    *,
    poincare_sections: tuple[float, ...],
    closed_return_tolerance: float,
    near_closed_return_tolerance: float,
    minimum_closed_or_near_fraction: float,
) -> EssosDirectCoilClosedControlResult:
    trajectories = np.asarray(inputs["trajectories_xyz"], dtype=np.float64)
    initial_xyz = np.asarray(inputs["initial_xyz"], dtype=np.float64)
    times = np.asarray(inputs["times"], dtype=np.float64)
    coil_gamma_xyz = np.asarray(inputs["coil_gamma_xyz"], dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
        raise ValueError("trajectories_xyz must have shape (n_field_lines, n_times, 3).")
    if initial_xyz.shape != (trajectories.shape[0], 3):
        raise ValueError("initial_xyz must have shape (n_field_lines, 3).")
    if times.shape != (trajectories.shape[1],):
        raise ValueError("times must have shape (n_times,).")
    if closed_return_tolerance < 0.0 or near_closed_return_tolerance < closed_return_tolerance:
        raise ValueError("Require 0 <= closed_return_tolerance <= near_closed_return_tolerance.")
    sections = np.mod(np.asarray(poincare_sections, dtype=np.float64), 2.0 * np.pi)
    if sections.size == 0:
        raise ValueError("At least one Poincare section is required.")

    line_length = _line_arc_lengths(trajectories)
    toroidal_advance = _line_toroidal_advances(trajectories)
    toroidal_turns = np.abs(toroidal_advance) / (2.0 * np.pi)
    radial = np.sqrt(trajectories[:, :, 0] ** 2 + trajectories[:, :, 1] ** 2)
    vertical = trajectories[:, :, 2]
    radial_span = np.ptp(radial, axis=1)
    vertical_span = np.ptp(vertical, axis=1)
    reference_extent = _reference_minor_extent(initial_xyz, radial_span, vertical_span)
    return_distances, return_time_indices, return_points = _minimum_same_section_return_distances(
        trajectories,
        initial_xyz,
    )
    normalized_return = return_distances / max(reference_extent, 1.0e-30)
    line_classification = _classify_return_distances(
        normalized_return,
        closed_return_tolerance=float(closed_return_tolerance),
        near_closed_return_tolerance=float(near_closed_return_tolerance),
    )
    poincare = _extract_poincare_hits(trajectories, sections)

    finite = bool(
        np.all(np.isfinite(trajectories))
        and np.all(np.isfinite(initial_xyz))
        and np.all(np.isfinite(line_length))
        and np.all(np.isfinite(toroidal_turns))
    )
    closed_fraction = float(np.mean(line_classification == 0)) if line_classification.size else 0.0
    near_closed_fraction = float(np.mean(line_classification == 1)) if line_classification.size else 0.0
    open_like_fraction = float(np.mean(line_classification == 2)) if line_classification.size else 0.0
    no_return_fraction = float(np.mean(line_classification == 3)) if line_classification.size else 0.0
    closed_or_near_fraction = closed_fraction + near_closed_fraction
    closed_control_passed = bool(
        finite
        and trajectories.shape[0] > 0
        and trajectories.shape[1] > 16
        and poincare["r"].size >= max(trajectories.shape[0], 1)
        and closed_or_near_fraction >= float(minimum_closed_or_near_fraction)
    )
    report: dict[str, Any] = {
        "case": "essos_direct_coil_closed_control",
        "source": inputs["source"],
        "source_mode": "live_essos" if inputs["live_essos"] else "self_contained_contract",
        "claim_scope": (
            "Direct-coil closed/near-closed field-line diagnostic. This report "
            "classifies return-map behavior and deliberately does not apply "
            "open-SOL target, sheath, recycling, or neutral-source semantics."
        ),
        "metadata": dict(inputs.get("metadata", {})),
        "n_field_lines": int(trajectories.shape[0]),
        "n_times": int(trajectories.shape[1]),
        "poincare_sections": [float(value) for value in sections],
        "poincare_point_count": int(poincare["r"].size),
        "reference_minor_extent": float(reference_extent),
        "line_length_mean": float(np.mean(line_length)),
        "line_length_max": float(np.max(line_length)),
        "toroidal_turns_mean": float(np.mean(toroidal_turns)),
        "toroidal_turns_max": float(np.max(toroidal_turns)),
        "radial_span_mean": float(np.mean(radial_span)),
        "vertical_span_mean": float(np.mean(vertical_span)),
        "return_distance_median": _finite_percentile(return_distances, 50.0),
        "return_distance_p95": _finite_percentile(return_distances, 95.0),
        "return_distance_normalized_median": _finite_percentile(normalized_return, 50.0),
        "return_distance_normalized_p95": _finite_percentile(normalized_return, 95.0),
        "return_distance_normalized_min": _finite_min(normalized_return),
        "closed_return_tolerance": float(closed_return_tolerance),
        "near_closed_return_tolerance": float(near_closed_return_tolerance),
        "minimum_closed_or_near_fraction": float(minimum_closed_or_near_fraction),
        "closed_fraction": closed_fraction,
        "near_closed_fraction": near_closed_fraction,
        "open_like_fraction": open_like_fraction,
        "no_return_fraction": no_return_fraction,
        "closed_or_near_fraction": closed_or_near_fraction,
        "target_semantics_applied": False,
        "sheath_recycling_semantics_applied": False,
        "closed_control_passed": closed_control_passed,
        "passed": bool(finite and trajectories.shape[0] > 0 and trajectories.shape[1] > 16),
        "promotion_ready": closed_control_passed,
        "promotion_rejection_reasons": (
            []
            if closed_control_passed
            else [
                reason
                for reason, active in (
                    ("not_enough_poincare_points", poincare["r"].size < max(trajectories.shape[0], 1)),
                    ("closed_or_near_fraction_below_threshold", closed_or_near_fraction < float(minimum_closed_or_near_fraction)),
                    ("nonfinite_trace_values", not finite),
                )
                if active
            ]
        ),
    }
    arrays = {
        "trajectories_xyz": trajectories.astype(np.float32),
        "initial_xyz": initial_xyz.astype(np.float32),
        "times": times.astype(np.float64),
        "coil_gamma_xyz": coil_gamma_xyz.astype(np.float32),
        "line_length": line_length.astype(np.float64),
        "line_toroidal_turns": toroidal_turns.astype(np.float64),
        "line_radial_span": radial_span.astype(np.float64),
        "line_vertical_span": vertical_span.astype(np.float64),
        "line_return_distance": return_distances.astype(np.float64),
        "line_return_distance_normalized": normalized_return.astype(np.float64),
        "line_return_time_index": return_time_indices.astype(np.float64),
        "line_return_point_xyz": return_points.astype(np.float32),
        "line_classification": line_classification.astype(np.int32),
        "poincare_sections": sections.astype(np.float64),
        "poincare_r": poincare["r"].astype(np.float32),
        "poincare_z": poincare["z"].astype(np.float32),
        "poincare_time_index": poincare["time_index"].astype(np.float64),
        "poincare_section_index": poincare["section_index"].astype(np.int32),
        "poincare_line_index": poincare["line_index"].astype(np.int32),
        "summary": np.asarray(
            [
                report["closed_fraction"],
                report["near_closed_fraction"],
                report["open_like_fraction"],
                report["return_distance_normalized_p95"],
                float(report["closed_control_passed"]),
            ],
            dtype=np.float64,
        ),
    }
    return EssosDirectCoilClosedControlResult(report=report, arrays=arrays)


def _manufactured_direct_coil_closed_control_inputs(
    *,
    n_radial_seeds: int,
    n_poloidal_seeds: int,
    times_to_trace: int,
) -> dict[str, Any]:
    n_radial = max(2, int(n_radial_seeds))
    n_poloidal = max(2, int(n_poloidal_seeds))
    times = np.linspace(0.0, 2.0 * np.pi * 8.0, int(times_to_trace), dtype=np.float64)
    radii = np.linspace(0.12, 0.34, n_radial, dtype=np.float64)
    theta0 = np.linspace(0.0, 2.0 * np.pi, n_poloidal, endpoint=False, dtype=np.float64)
    trajectories: list[np.ndarray] = []
    initial: list[np.ndarray] = []
    for i, radius in enumerate(radii):
        for j, theta_seed in enumerate(theta0):
            iota = 0.50 if (i + j) % 3 == 0 else 0.50 + 0.004 * (i + 1)
            phase = theta_seed + iota * times
            major = 1.55 + radius * np.cos(phase) + 0.035 * np.cos(5.0 * times + 0.4 * j)
            vertical = 0.02 + 0.82 * radius * np.sin(phase) * (1.0 + 0.08 * np.sin(5.0 * times))
            x = major * np.cos(times)
            y = major * np.sin(times)
            line = np.stack([x, y, vertical], axis=-1)
            trajectories.append(line)
            initial.append(line[0])
    coil_phi = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=True)
    coils = []
    for shift in np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False):
        radius = 1.55 + 0.42 * np.cos(coil_phi + shift)
        coils.append(
            np.stack(
                [
                    radius * np.cos(coil_phi),
                    radius * np.sin(coil_phi),
                    0.34 * np.sin(coil_phi + shift),
                ],
                axis=-1,
            )
        )
    return {
        "source": "manufactured non-axisymmetric direct-coil closed-control contract",
        "live_essos": False,
        "trajectories_xyz": np.asarray(trajectories, dtype=np.float64),
        "initial_xyz": np.asarray(initial, dtype=np.float64),
        "times": times,
        "coil_gamma_xyz": np.asarray(coils, dtype=np.float64),
        "metadata": {
            "manufactured": True,
            "n_radial_seeds": n_radial,
            "n_poloidal_seeds": n_poloidal,
            "field_period_modulation": 5,
        },
    }


def _line_arc_lengths(trajectories: np.ndarray) -> np.ndarray:
    diffs = np.diff(trajectories, axis=1)
    return np.sum(np.linalg.norm(diffs, axis=-1), axis=1)


def _line_toroidal_advances(trajectories: np.ndarray) -> np.ndarray:
    phi = np.unwrap(np.arctan2(trajectories[:, :, 1], trajectories[:, :, 0]), axis=1)
    return phi[:, -1] - phi[:, 0]


def _median_toroidal_advance(trajectories: np.ndarray) -> float:
    return float(np.nanmedian(_line_toroidal_advances(trajectories)))


def _reference_minor_extent(initial_xyz: np.ndarray, radial_span: np.ndarray, vertical_span: np.ndarray) -> float:
    initial_major = np.sqrt(initial_xyz[:, 0] ** 2 + initial_xyz[:, 1] ** 2)
    initial_vertical = initial_xyz[:, 2]
    candidates = [
        float(np.ptp(initial_major)),
        float(np.ptp(initial_vertical)),
        float(np.nanmedian(radial_span + vertical_span)),
    ]
    return max([value for value in candidates if np.isfinite(value)] + [1.0e-6])


def _minimum_same_section_return_distances(
    trajectories: np.ndarray,
    initial_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_lines = trajectories.shape[0]
    distances = np.full(n_lines, np.inf, dtype=np.float64)
    time_indices = np.full(n_lines, np.nan, dtype=np.float64)
    points = np.full((n_lines, 3), np.nan, dtype=np.float64)
    period = 2.0 * np.pi
    for line_index, trajectory in enumerate(trajectories):
        phi = np.unwrap(np.arctan2(trajectory[:, 1], trajectory[:, 0]))
        if phi.size < 3 or not np.all(np.isfinite(phi)):
            continue
        direction = 1.0 if phi[-1] >= phi[0] else -1.0
        if direction < 0.0:
            phi = phi[::-1]
            trajectory = trajectory[::-1]
        phi0 = float(phi[0])
        phi_min = float(phi[0])
        phi_max = float(phi[-1])
        max_turn = int(np.floor((phi_max - phi0) / period))
        best_distance = np.inf
        best_index = np.nan
        best_point = np.full(3, np.nan, dtype=np.float64)
        for turn in range(1, max_turn + 1):
            target = phi0 + period * float(turn)
            if target <= phi_min + 1.0e-10 or target >= phi_max - 1.0e-10:
                continue
            point = np.array(
                [
                    np.interp(target, phi, trajectory[:, 0]),
                    np.interp(target, phi, trajectory[:, 1]),
                    np.interp(target, phi, trajectory[:, 2]),
                ],
                dtype=np.float64,
            )
            distance = float(np.linalg.norm(point - initial_xyz[line_index]))
            if distance < best_distance:
                best_distance = distance
                best_index = float(np.interp(target, phi, np.arange(phi.size, dtype=np.float64)))
                best_point = point
        distances[line_index] = best_distance
        time_indices[line_index] = best_index
        points[line_index] = best_point
    return distances, time_indices, points


def _classify_return_distances(
    normalized_return: np.ndarray,
    *,
    closed_return_tolerance: float,
    near_closed_return_tolerance: float,
) -> np.ndarray:
    labels = np.full(normalized_return.shape, 3, dtype=np.int32)
    finite = np.isfinite(normalized_return)
    labels[finite] = 2
    labels[finite & (normalized_return <= near_closed_return_tolerance)] = 1
    labels[finite & (normalized_return <= closed_return_tolerance)] = 0
    return labels


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


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("inf")
    return float(np.percentile(finite, float(percentile)))


def _finite_min(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("inf")
    return float(np.min(finite))


def _line_label_colors(labels: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [0.10, 0.49, 0.47, 1.0],
            [0.85, 0.37, 0.01, 1.0],
            [0.29, 0.35, 0.55, 1.0],
            [0.55, 0.55, 0.55, 1.0],
        ],
        dtype=np.float64,
    )
    return palette[np.clip(labels.astype(int), 0, len(palette) - 1)]


def _set_equal_3d(axis: plt.Axes, points: np.ndarray) -> None:
    finite = np.asarray(points, dtype=np.float64)
    finite = finite[np.all(np.isfinite(finite), axis=1)]
    if finite.size == 0:
        return
    center = np.mean(finite, axis=0)
    span = float(np.max(np.ptp(finite, axis=0)))
    span = max(span, 1.0e-12)
    half = 0.5 * span
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)
