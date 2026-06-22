from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

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
class EssosDirectCoilClosedControlRefinementArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


@dataclass(frozen=True)
class EssosDirectCoilClosedControlTransientArtifacts:
    report_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path
    movie_gif_path: Path | None


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
            "source_mode": "provided_trace_bundle",
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


def create_essos_direct_coil_closed_control_refinement_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_direct_coil_closed_control_refinement",
    use_live_essos: bool = False,
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    level_settings: tuple[tuple[int, int, int], ...] = (
        (3, 3, 256),
        (5, 4, 512),
        (7, 6, 768),
    ),
    rho_min: float = 0.20,
    rho_max: float = 0.82,
    maxtime: float = 900.0,
    trace_tolerance: float = 1.0e-8,
    poincare_sections: tuple[float, ...] = (0.0, float(np.pi / 2.0), float(np.pi), float(3.0 * np.pi / 2.0)),
    closed_return_tolerance: float = 3.0e-2,
    near_closed_return_tolerance: float = 1.5e-1,
    minimum_closed_or_near_fraction: float = 0.20,
    maximum_closed_or_near_fraction_spread: float = 5.0e-2,
    maximum_class_fraction_spread: float = 2.5e-1,
    minimum_poincare_points_per_line: float = 4.0,
) -> EssosDirectCoilClosedControlRefinementArtifacts:
    """Write the direct-coil closed-control refinement artifact.

    The refinement gate reruns the same closed/near-closed diagnostic at
    progressively larger seed and trace samples. It is intentionally summary
    based: closed-field controls are promoted by stable return-map
    classification and bounded same-section return distance, not by open-SOL
    target-to-target connection-length semantics.
    """

    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report, arrays = build_essos_direct_coil_closed_control_refinement_campaign(
        use_live_essos=use_live_essos,
        coil_json_path=coil_json_path,
        vmec_wout_path=vmec_wout_path,
        essos_root=essos_root,
        level_settings=level_settings,
        rho_min=rho_min,
        rho_max=rho_max,
        maxtime=maxtime,
        trace_tolerance=trace_tolerance,
        poincare_sections=poincare_sections,
        closed_return_tolerance=closed_return_tolerance,
        near_closed_return_tolerance=near_closed_return_tolerance,
        minimum_closed_or_near_fraction=minimum_closed_or_near_fraction,
        maximum_closed_or_near_fraction_spread=maximum_closed_or_near_fraction_spread,
        maximum_class_fraction_spread=maximum_class_fraction_spread,
        minimum_poincare_points_per_line=minimum_poincare_points_per_line,
    )
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_essos_direct_coil_closed_control_refinement_plot(report, arrays, plot_png_path)
    return EssosDirectCoilClosedControlRefinementArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_essos_direct_coil_closed_control_refinement_campaign(
    *,
    use_live_essos: bool = False,
    coil_json_path: str | Path | None = None,
    vmec_wout_path: str | Path | None = None,
    essos_root: str | Path | None = None,
    level_settings: tuple[tuple[int, int, int], ...] = (
        (3, 3, 256),
        (5, 4, 512),
        (7, 6, 768),
    ),
    rho_min: float = 0.20,
    rho_max: float = 0.82,
    maxtime: float = 900.0,
    trace_tolerance: float = 1.0e-8,
    poincare_sections: tuple[float, ...] = (0.0, float(np.pi / 2.0), float(np.pi), float(3.0 * np.pi / 2.0)),
    closed_return_tolerance: float = 3.0e-2,
    near_closed_return_tolerance: float = 1.5e-1,
    minimum_closed_or_near_fraction: float = 0.20,
    maximum_closed_or_near_fraction_spread: float = 5.0e-2,
    maximum_class_fraction_spread: float = 2.5e-1,
    minimum_poincare_points_per_line: float = 4.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the direct-coil closed-control refinement report and arrays."""

    if len(level_settings) < 2:
        raise ValueError("Closed-control refinement requires at least two levels.")
    level_reports: list[dict[str, Any]] = []
    level_summary: list[list[float]] = []
    for level_index, (n_radial, n_poloidal, trace_samples) in enumerate(level_settings):
        if int(n_radial) < 2 or int(n_poloidal) < 2 or int(trace_samples) < 32:
            raise ValueError(
                "Each closed-control refinement level must use at least two "
                "radial seeds, two poloidal seeds, and 32 trace samples."
            )
        if use_live_essos:
            result = build_essos_direct_coil_closed_control_campaign(
                use_live_essos=True,
                coil_json_path=coil_json_path,
                vmec_wout_path=vmec_wout_path,
                essos_root=essos_root,
                rho_min=rho_min,
                rho_max=rho_max,
                n_radial_seeds=int(n_radial),
                n_poloidal_seeds=int(n_poloidal),
                maxtime=maxtime,
                times_to_trace=int(trace_samples),
                trace_tolerance=trace_tolerance,
                poincare_sections=poincare_sections,
                closed_return_tolerance=closed_return_tolerance,
                near_closed_return_tolerance=near_closed_return_tolerance,
                minimum_closed_or_near_fraction=minimum_closed_or_near_fraction,
            )
        else:
            inputs = _manufactured_direct_coil_closed_control_inputs(
                n_radial_seeds=int(n_radial),
                n_poloidal_seeds=int(n_poloidal),
                times_to_trace=int(trace_samples),
            )
            result = _build_closed_control_from_inputs(
                inputs,
                poincare_sections=poincare_sections,
                closed_return_tolerance=closed_return_tolerance,
                near_closed_return_tolerance=near_closed_return_tolerance,
                minimum_closed_or_near_fraction=minimum_closed_or_near_fraction,
            )
        level_report = dict(result.report)
        level_report["level_index"] = int(level_index)
        level_report["level_settings"] = {
            "n_radial_seeds": int(n_radial),
            "n_poloidal_seeds": int(n_poloidal),
            "times_to_trace": int(trace_samples),
        }
        level_reports.append(level_report)
        level_summary.append(
            [
                float(level_index),
                float(level_report["n_field_lines"]),
                float(level_report["n_times"]),
                float(level_report["toroidal_turns_mean"]),
                float(level_report["poincare_point_count"]),
                float(level_report["return_distance_normalized_p95"]),
                float(level_report["closed_fraction"]),
                float(level_report["near_closed_fraction"]),
                float(level_report["open_like_fraction"]),
                float(level_report["no_return_fraction"]),
                float(level_report["closed_or_near_fraction"]),
                float(level_report["closed_control_passed"]),
            ]
        )

    diagnostics = build_essos_direct_coil_closed_control_refinement_diagnostics(
        level_reports,
        minimum_closed_or_near_fraction=minimum_closed_or_near_fraction,
        maximum_closed_or_near_fraction_spread=maximum_closed_or_near_fraction_spread,
        maximum_class_fraction_spread=maximum_class_fraction_spread,
        minimum_poincare_points_per_line=minimum_poincare_points_per_line,
    )
    report: dict[str, Any] = {
        "case": "essos_direct_coil_closed_control_refinement",
        "source": (
            "live ESSOS direct-coil closed-control refinement"
            if use_live_essos
            else "self-contained manufactured direct-coil closed-control refinement"
        ),
        "claim_scope": (
            "Closed/near-closed direct-coil refinement gate. It checks stability "
            "of return-map classification and bounded same-section return "
            "distance across larger seed/time samples, while forbidding open-SOL "
            "target, sheath, recycling, and neutral semantics."
        ),
        "level_settings": [
            {
                "n_radial_seeds": int(n_radial),
                "n_poloidal_seeds": int(n_poloidal),
                "times_to_trace": int(trace_samples),
            }
            for n_radial, n_poloidal, trace_samples in level_settings
        ],
        "level_reports": level_reports,
        **diagnostics,
    }
    arrays = {
        "level_summary": np.asarray(level_summary, dtype=np.float64),
        "level_summary_columns": np.asarray(
            [
                "level_index",
                "n_field_lines",
                "n_times",
                "toroidal_turns_mean",
                "poincare_point_count",
                "return_distance_normalized_p95",
                "closed_fraction",
                "near_closed_fraction",
                "open_like_fraction",
                "no_return_fraction",
                "closed_or_near_fraction",
                "closed_control_passed",
            ],
            dtype="U48",
        ),
    }
    return report, arrays


def build_essos_direct_coil_closed_control_refinement_diagnostics(
    level_reports: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    minimum_closed_or_near_fraction: float = 0.20,
    maximum_closed_or_near_fraction_spread: float = 5.0e-2,
    maximum_class_fraction_spread: float = 2.5e-1,
    minimum_poincare_points_per_line: float = 4.0,
) -> dict[str, Any]:
    """Summarize whether closed-control return-map evidence is stable."""

    if len(level_reports) < 2:
        raise ValueError("Closed-control refinement diagnostics require at least two reports.")
    closed_or_near = np.asarray(
        [float(report.get("closed_or_near_fraction", np.nan)) for report in level_reports],
        dtype=np.float64,
    )
    closed = np.asarray(
        [float(report.get("closed_fraction", np.nan)) for report in level_reports],
        dtype=np.float64,
    )
    near = np.asarray(
        [float(report.get("near_closed_fraction", np.nan)) for report in level_reports],
        dtype=np.float64,
    )
    open_like = np.asarray(
        [float(report.get("open_like_fraction", np.nan)) for report in level_reports],
        dtype=np.float64,
    )
    no_return = np.asarray(
        [float(report.get("no_return_fraction", np.nan)) for report in level_reports],
        dtype=np.float64,
    )
    return_p95 = np.asarray(
        [float(report.get("return_distance_normalized_p95", np.inf)) for report in level_reports],
        dtype=np.float64,
    )
    near_tolerance = np.asarray(
        [float(report.get("near_closed_return_tolerance", np.nan)) for report in level_reports],
        dtype=np.float64,
    )
    poincare_density = np.asarray(
        [
            float(report.get("poincare_point_count", 0.0)) / max(float(report.get("n_field_lines", 0.0)), 1.0)
            for report in level_reports
        ],
        dtype=np.float64,
    )
    all_closed_controls_passed = all(bool(report.get("closed_control_passed", False)) for report in level_reports)
    target_semantics_absent = all(not bool(report.get("target_semantics_applied", True)) for report in level_reports)
    sheath_semantics_absent = all(
        not bool(report.get("sheath_recycling_semantics_applied", True)) for report in level_reports
    )
    finite = bool(
        np.all(np.isfinite(closed_or_near))
        and np.all(np.isfinite(closed))
        and np.all(np.isfinite(near))
        and np.all(np.isfinite(open_like))
        and np.all(np.isfinite(no_return))
        and np.all(np.isfinite(return_p95))
        and np.all(np.isfinite(near_tolerance))
        and np.all(np.isfinite(poincare_density))
    )
    closed_or_near_min = _finite_min(closed_or_near)
    closed_or_near_spread = _finite_range(closed_or_near)
    closed_fraction_spread = _finite_range(closed)
    near_fraction_spread = _finite_range(near)
    open_or_no_return_max = _finite_max(open_like + no_return)
    return_p95_max = _finite_max(return_p95)
    near_tolerance_max = _finite_max(near_tolerance)
    poincare_density_min = _finite_min(poincare_density)

    closed_or_near_floor_passed = bool(closed_or_near_min >= float(minimum_closed_or_near_fraction))
    closed_or_near_stability_passed = bool(
        closed_or_near_spread <= float(maximum_closed_or_near_fraction_spread)
    )
    class_fraction_stability_passed = bool(
        closed_fraction_spread <= float(maximum_class_fraction_spread)
        and near_fraction_spread <= float(maximum_class_fraction_spread)
    )
    return_distance_bound_passed = bool(return_p95_max <= near_tolerance_max)
    poincare_density_passed = bool(poincare_density_min >= float(minimum_poincare_points_per_line))
    open_fraction_passed = bool(open_or_no_return_max <= 1.0 - float(minimum_closed_or_near_fraction))

    rejection_reasons = [
        reason
        for reason, active in (
            ("nonfinite_refinement_metrics", not finite),
            ("not_all_level_closed_controls_passed", not all_closed_controls_passed),
            ("target_semantics_present", not target_semantics_absent),
            ("sheath_recycling_semantics_present", not sheath_semantics_absent),
            ("closed_or_near_fraction_below_threshold", not closed_or_near_floor_passed),
            ("closed_or_near_fraction_unstable", not closed_or_near_stability_passed),
            ("closed_near_split_unstable", not class_fraction_stability_passed),
            ("return_distance_p95_exceeds_near_closed_tolerance", not return_distance_bound_passed),
            ("insufficient_poincare_points_per_line", not poincare_density_passed),
            ("open_or_no_return_fraction_too_large", not open_fraction_passed),
        )
        if active
    ]
    promotion_ready = bool(not rejection_reasons)
    return {
        "diagnostic": "essos_direct_coil_closed_control_refinement",
        "n_levels": int(len(level_reports)),
        "finite_refinement_metrics": finite,
        "all_level_closed_controls_passed": bool(all_closed_controls_passed),
        "target_semantics_absent": bool(target_semantics_absent),
        "sheath_recycling_semantics_absent": bool(sheath_semantics_absent),
        "minimum_closed_or_near_fraction": float(minimum_closed_or_near_fraction),
        "maximum_closed_or_near_fraction_spread": float(maximum_closed_or_near_fraction_spread),
        "maximum_class_fraction_spread": float(maximum_class_fraction_spread),
        "minimum_poincare_points_per_line": float(minimum_poincare_points_per_line),
        "closed_or_near_fraction_min": float(closed_or_near_min),
        "closed_or_near_fraction_spread": float(closed_or_near_spread),
        "closed_fraction_spread": float(closed_fraction_spread),
        "near_closed_fraction_spread": float(near_fraction_spread),
        "open_or_no_return_fraction_max": float(open_or_no_return_max),
        "return_distance_normalized_p95_max": float(return_p95_max),
        "near_closed_return_tolerance_max": float(near_tolerance_max),
        "poincare_points_per_line_min": float(poincare_density_min),
        "closed_or_near_floor_passed": closed_or_near_floor_passed,
        "closed_or_near_stability_passed": closed_or_near_stability_passed,
        "class_fraction_stability_passed": class_fraction_stability_passed,
        "return_distance_bound_passed": return_distance_bound_passed,
        "poincare_density_passed": poincare_density_passed,
        "open_fraction_passed": open_fraction_passed,
        "passed": promotion_ready,
        "promotion_ready": promotion_ready,
        "evidence_role": (
            "closed_control_refinement_gate_passed"
            if promotion_ready
            else "closed_control_refinement_gate_failed"
        ),
        "promotion_rejection_reasons": rejection_reasons,
    }


def create_essos_direct_coil_closed_control_transient_package(
    *,
    output_root: str | Path,
    case_label: str = "essos_direct_coil_closed_control_transient",
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
    frames: int = 12,
    substeps_per_frame: int = 4,
    dt: float = 2.0e-2,
    samples_per_line: int = 192,
    parallel_diffusivity: float = 2.5e-2,
    advection_strength: float = 6.0e-2,
    drive_strength: float = 5.0e-2,
    write_movie: bool = True,
) -> EssosDirectCoilClosedControlTransientArtifacts:
    """Write a reduced closed-trace transient on direct-coil field lines.

    This is a closed/near-closed control media gate. It deliberately omits
    target, sheath, recycling, and neutral-loss semantics; open-SOL physics
    must use the direct-coil open-field or hybrid workflows with endpoint
    masks.
    """

    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    movies_dir = root / "movies"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if write_movie:
        movies_dir.mkdir(parents=True, exist_ok=True)

    control = build_essos_direct_coil_closed_control_campaign(
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
    report, arrays = build_essos_direct_coil_closed_control_transient_campaign(
        control.report,
        control.arrays,
        frames=frames,
        substeps_per_frame=substeps_per_frame,
        dt=dt,
        samples_per_line=samples_per_line,
        parallel_diffusivity=parallel_diffusivity,
        advection_strength=advection_strength,
        drive_strength=drive_strength,
    )
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(arrays_npz_path, **arrays)
    plot_png_path = images_dir / f"{case_label}.png"
    save_essos_direct_coil_closed_control_transient_plot(report, arrays, plot_png_path)
    movie_gif_path: Path | None = None
    if write_movie:
        movie_gif_path = movies_dir / f"{case_label}.gif"
        save_essos_direct_coil_closed_control_transient_movie(report, arrays, movie_gif_path)
    return EssosDirectCoilClosedControlTransientArtifacts(
        report_json_path=report_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
        movie_gif_path=movie_gif_path,
    )


def build_essos_direct_coil_closed_control_transient_campaign(
    control_report: dict[str, Any],
    control_arrays: dict[str, np.ndarray],
    *,
    frames: int = 12,
    substeps_per_frame: int = 4,
    dt: float = 2.0e-2,
    samples_per_line: int = 192,
    parallel_diffusivity: float = 2.5e-2,
    advection_strength: float = 6.0e-2,
    drive_strength: float = 5.0e-2,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run a compact periodic scalar transient on closed direct-coil traces."""

    trajectories = np.asarray(control_arrays["trajectories_xyz"], dtype=np.float64)
    line_label = np.asarray(control_arrays["line_classification"], dtype=np.int32)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
        raise ValueError("trajectories_xyz must have shape (n_field_lines, n_times, 3).")
    if int(frames) < 2 or int(substeps_per_frame) < 1:
        raise ValueError("The closed-control transient needs at least two frames and one substep.")
    n_lines, n_trace_times, _ = trajectories.shape
    if n_lines == 0 or n_trace_times < 16:
        raise ValueError("The closed-control transient needs nonempty traces with at least 16 samples.")

    sample_count = int(np.clip(int(samples_per_line), 16, n_trace_times))
    sample_indices = np.linspace(0, n_trace_times - 1, sample_count, dtype=int)
    sampled = trajectories[:, sample_indices, :]
    major = np.sqrt(sampled[:, :, 0] ** 2 + sampled[:, :, 1] ** 2)
    vertical = sampled[:, :, 2]
    phi = np.unwrap(np.arctan2(sampled[:, :, 1], sampled[:, :, 0]), axis=1)
    phase = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False, dtype=np.float64)
    normalized_major = _normalize_array(major)
    normalized_vertical = _normalize_array(vertical)
    closed_weight = np.where(line_label <= 1, 1.0, 0.45).reshape((-1, 1))

    density = 1.0 + 0.045 * closed_weight * (
        np.sin(phase[None, :] + 0.7 * np.arange(n_lines, dtype=np.float64)[:, None])
        + 0.35 * np.cos(2.0 * phase[None, :] - 0.5 * normalized_major)
        + 0.20 * normalized_vertical
    )
    density = np.maximum(density, 1.0e-6)
    history: list[np.ndarray] = []
    rms_history: list[float] = []
    mass_history: list[float] = []
    line_profile_history: list[np.ndarray] = []
    grad_history: list[float] = []

    def record(state: np.ndarray) -> None:
        line_mean = np.mean(state, axis=1)
        fluctuation = state - line_mean[:, None]
        history.append(fluctuation.astype(np.float32))
        rms_history.append(float(np.sqrt(np.mean(fluctuation**2))))
        mass_history.append(float(np.sum(state)))
        line_profile_history.append(line_mean.astype(np.float64))
        grad_history.append(float(np.sqrt(np.mean(_periodic_gradient(state) ** 2))))

    record(density)
    total_substeps = int(frames) * int(substeps_per_frame)
    for frame_index in range(int(frames)):
        for local_index in range(int(substeps_per_frame)):
            step_index = frame_index * int(substeps_per_frame) + local_index
            scalar_time = float(step_index) / max(float(total_substeps - 1), 1.0)
            drive = _closed_trace_drive(
                phase,
                normalized_major,
                normalized_vertical,
                phi,
                scalar_time=scalar_time,
            )
            rhs = (
                float(parallel_diffusivity) * _periodic_laplacian(density)
                - float(advection_strength) * _periodic_gradient(density)
                + float(drive_strength) * drive
            )
            rhs = rhs - np.mean(rhs, axis=1, keepdims=True)
            density = np.maximum(density + float(dt) * rhs, 1.0e-8)
        record(density)

    density_history = np.asarray(history, dtype=np.float32)
    line_profiles = np.asarray(line_profile_history, dtype=np.float64)
    rms = np.asarray(rms_history, dtype=np.float64)
    mass = np.asarray(mass_history, dtype=np.float64)
    grad = np.asarray(grad_history, dtype=np.float64)
    time = np.arange(int(frames) + 1, dtype=np.float64) * float(substeps_per_frame) * float(dt)
    mass_relative_drift = float(
        np.max(np.abs(mass - mass[0])) / max(abs(float(mass[0])), 1.0e-30)
    )
    temporal_rms_change = float(np.max(rms) - np.min(rms))
    finite = bool(
        np.all(np.isfinite(density_history))
        and np.all(np.isfinite(line_profiles))
        and np.all(np.isfinite(mass))
        and np.all(np.isfinite(rms))
    )
    movie_vmax = float(np.nanpercentile(np.abs(density_history), 97.0))
    if not np.isfinite(movie_vmax) or movie_vmax <= 0.0:
        movie_vmax = 1.0e-3
    closed_control_passed = bool(control_report.get("closed_control_passed", False))
    no_open_semantics = bool(
        not control_report.get("target_semantics_applied", True)
        and not control_report.get("sheath_recycling_semantics_applied", True)
    )
    transient_ready = bool(
        finite
        and closed_control_passed
        and no_open_semantics
        and np.min(density) > 0.0
        and mass_relative_drift < 5.0e-3
        and rms[-1] > 1.0e-5
        and temporal_rms_change > 1.0e-7
    )
    report: dict[str, Any] = {
        "case": "essos_direct_coil_closed_control_transient",
        "source": "JAXDRB reduced periodic scalar transient on direct-coil closed/near-closed traces",
        "source_mode": control_report.get("source_mode", "unknown"),
        "claim_scope": (
            "Direct-coil closed/near-closed reduced transient. The model is a "
            "periodic line-following scalar advection-diffusion control on the "
            "validated trace bundle; it deliberately has no target endpoints, "
            "sheath losses, recycling source, or neutral-loss semantics."
        ),
        "control_report_summary": {
            "closed_control_passed": closed_control_passed,
            "closed_fraction": float(control_report.get("closed_fraction", 0.0)),
            "near_closed_fraction": float(control_report.get("near_closed_fraction", 0.0)),
            "open_like_fraction": float(control_report.get("open_like_fraction", 0.0)),
            "no_return_fraction": float(control_report.get("no_return_fraction", 0.0)),
            "return_distance_normalized_p95": float(
                control_report.get("return_distance_normalized_p95", np.inf)
            ),
        },
        "target_semantics_applied": False,
        "sheath_recycling_semantics_applied": False,
        "neutral_loss_semantics_applied": False,
        "open_sol_publication_ready": False,
        "open_sol_rejection_reason": "closed_direct_coil_trace_has_no_endpoint_sheath_recycling_or_neutral_loss_semantics",
        "frames": int(frames),
        "substeps_per_frame": int(substeps_per_frame),
        "dt": float(dt),
        "samples_per_line": int(sample_count),
        "n_field_lines": int(n_lines),
        "parallel_diffusivity": float(parallel_diffusivity),
        "advection_strength": float(advection_strength),
        "drive_strength": float(drive_strength),
        "finite_transient": finite,
        "no_open_sol_semantics": no_open_semantics,
        "initial_fluctuation_rms": float(rms[0]),
        "final_fluctuation_rms": float(rms[-1]),
        "max_fluctuation_rms": float(np.max(rms)),
        "temporal_rms_change": temporal_rms_change,
        "mass_relative_drift": mass_relative_drift,
        "final_parallel_gradient_rms": float(grad[-1]),
        "final_min_density": float(np.min(density)),
        "final_max_density": float(np.max(density)),
        "fixed_camera": True,
        "fixed_color_limits": True,
        "movie_visual_qa_passed": transient_ready,
        "closed_control_media_ready": transient_ready,
        "passed": transient_ready,
        "promotion_ready": transient_ready,
        "promotion_rejection_reasons": [
            reason
            for reason, active in (
                ("base_closed_control_failed", not closed_control_passed),
                ("open_sol_semantics_present", not no_open_semantics),
                ("nonfinite_transient", not finite),
                ("mass_drift_exceeds_threshold", mass_relative_drift >= 5.0e-3),
                ("fluctuation_too_small", rms[-1] <= 1.0e-5),
                ("temporal_variation_too_small", temporal_rms_change <= 1.0e-7),
            )
            if active
        ],
    }
    arrays = {
        "time": time.astype(np.float64),
        "density_fluctuation_history": density_history,
        "line_profile_history": line_profiles.astype(np.float32),
        "fluctuation_rms_history": rms.astype(np.float64),
        "mass_history": mass.astype(np.float64),
        "parallel_gradient_rms_history": grad.astype(np.float64),
        "sampled_trajectories_xyz": sampled.astype(np.float32),
        "sampled_major_radius": major.astype(np.float32),
        "sampled_vertical": vertical.astype(np.float32),
        "sampled_phi": phi.astype(np.float32),
        "line_classification": line_label.astype(np.int32),
        "movie_vmax": np.asarray([movie_vmax], dtype=np.float64),
        "summary": np.asarray(
            [
                report["final_fluctuation_rms"],
                report["mass_relative_drift"],
                report["temporal_rms_change"],
                float(report["closed_control_media_ready"]),
            ],
            dtype=np.float64,
        ),
    }
    return report, arrays


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


def save_essos_direct_coil_closed_control_refinement_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Save the direct-coil closed-control refinement QA figure."""

    summary = np.asarray(arrays["level_summary"], dtype=np.float64)
    columns = [str(value) for value in np.asarray(arrays["level_summary_columns"])]
    column_index = {name: index for index, name in enumerate(columns)}
    levels = summary[:, column_index["level_index"]]
    closed = summary[:, column_index["closed_fraction"]]
    near = summary[:, column_index["near_closed_fraction"]]
    open_like = summary[:, column_index["open_like_fraction"]]
    no_return = summary[:, column_index["no_return_fraction"]]
    return_p95 = summary[:, column_index["return_distance_normalized_p95"]]
    poincare_density = summary[:, column_index["poincare_point_count"]] / np.maximum(
        summary[:, column_index["n_field_lines"]],
        1.0,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.0), constrained_layout=True)
    axf, axr, axp, axt = axes.ravel()
    width = 0.72
    axf.bar(levels, closed, width=width, color="#0A6E7D", label="closed")
    axf.bar(levels, near, bottom=closed, width=width, color="#CA6702", label="near-closed")
    axf.bar(levels, open_like, bottom=closed + near, width=width, color="#577590", label="open-like")
    axf.bar(levels, no_return, bottom=closed + near + open_like, width=width, color="0.65", label="no return")
    axf.axhline(report["minimum_closed_or_near_fraction"], color="black", lw=1.0, ls="--")
    style_axis(
        axf,
        title="return-map classification by refinement level",
        xlabel="refinement level",
        ylabel="fraction",
        grid="y",
    )
    axf.set_ylim(0.0, 1.04)
    axf.legend(frameon=False, fontsize=8)

    axr.plot(levels, return_p95, marker="o", color="#BB3E03", lw=2.0)
    axr.axhline(report["near_closed_return_tolerance_max"], color="black", lw=1.0, ls="--")
    style_axis(
        axr,
        title="same-section return distance",
        xlabel="refinement level",
        ylabel="p95 normalized distance",
        grid="both",
    )

    axp.plot(levels, poincare_density, marker="s", color="#2A9D8F", lw=2.0)
    axp.axhline(report["minimum_poincare_points_per_line"], color="black", lw=1.0, ls="--")
    style_axis(
        axp,
        title="Poincare sampling density",
        xlabel="refinement level",
        ylabel="points per seed line",
        grid="both",
    )

    axt.axis("off")
    axt.text(
        0.02,
        0.95,
        "\n".join(
            [
                "Closed-control refinement gate",
                f"levels: {report['n_levels']}",
                f"closed/near min: {report['closed_or_near_fraction_min']:.3f}",
                f"closed/near spread: {report['closed_or_near_fraction_spread']:.3e}",
                f"return p95 max: {report['return_distance_normalized_p95_max']:.3e}",
                f"Poincare/line min: {report['poincare_points_per_line_min']:.1f}",
                f"target semantics absent: {report['target_semantics_absent']}",
                f"sheath/recycling absent: {report['sheath_recycling_semantics_absent']}",
                f"promotion ready: {report['promotion_ready']}",
            ]
        ),
        transform=axt.transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.96},
    )
    fig.suptitle("ESSOS direct-coil closed-control refinement", fontsize=15, fontweight="semibold")
    save_publication_figure(fig, path)
    return Path(path)


def save_essos_direct_coil_closed_control_transient_plot(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Save the direct-coil closed-trace transient QA figure."""

    history = np.asarray(arrays["density_fluctuation_history"], dtype=np.float64)
    sampled = np.asarray(arrays["sampled_trajectories_xyz"], dtype=np.float64)
    major = np.asarray(arrays["sampled_major_radius"], dtype=np.float64)
    vertical = np.asarray(arrays["sampled_vertical"], dtype=np.float64)
    line_label = np.asarray(arrays["line_classification"], dtype=np.int32)
    time = np.asarray(arrays["time"], dtype=np.float64)
    rms = np.asarray(arrays["fluctuation_rms_history"], dtype=np.float64)
    mass = np.asarray(arrays["mass_history"], dtype=np.float64)
    line_profiles = np.asarray(arrays["line_profile_history"], dtype=np.float64)
    vmax = float(np.asarray(arrays["movie_vmax"], dtype=np.float64)[0])
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(16.0, 9.0), constrained_layout=True)
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    axrz = fig.add_subplot(2, 3, 2)
    axline = fig.add_subplot(2, 3, 3)
    axhist = fig.add_subplot(2, 3, 4)
    axmass = fig.add_subplot(2, 3, 5)
    axtext = fig.add_subplot(2, 3, 6)

    final = history[-1]
    stride = max(1, sampled.shape[1] // 96)
    scatter = ax3d.scatter(
        sampled[:, ::stride, 0].reshape(-1),
        sampled[:, ::stride, 1].reshape(-1),
        sampled[:, ::stride, 2].reshape(-1),
        c=final[:, ::stride].reshape(-1),
        s=4.0,
        cmap="coolwarm",
        norm=norm,
        linewidths=0.0,
        alpha=0.86,
    )
    ax3d.set_title("closed direct-coil trace transient")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    _set_equal_3d(ax3d, sampled.reshape((-1, 3)))
    fig.colorbar(scatter, ax=ax3d, fraction=0.045, pad=0.02, label=r"$\tilde{n}$")

    rz = axrz.scatter(
        major.reshape(-1),
        vertical.reshape(-1),
        c=final.reshape(-1),
        s=4.5,
        cmap="coolwarm",
        norm=norm,
        linewidths=0.0,
        alpha=0.82,
    )
    axrz.set_aspect("equal", adjustable="box")
    style_axis(axrz, title="final R-Z fluctuation projection", xlabel="R", ylabel="Z", grid="both")
    fig.colorbar(rz, ax=axrz, fraction=0.046, pad=0.03, label=r"$\tilde{n}$")

    line_coordinate = np.linspace(0.0, 1.0, final.shape[1], endpoint=False)
    for index in range(min(final.shape[0], 10)):
        color = _line_label_colors(line_label)[index]
        axline.plot(line_coordinate, final[index], lw=1.2, alpha=0.8, color=color)
    style_axis(axline, title="final line-following fluctuations", xlabel="normalized line coordinate", ylabel=r"$\tilde{n}$")

    axhist.hist(final.reshape(-1), bins=32, color="#0A6E7D", alpha=0.82)
    style_axis(axhist, title="final fluctuation distribution", xlabel=r"$\tilde{n}$", ylabel="sample count", grid="y")

    axmass.plot(time, rms, lw=2.0, color="#005f73", label="fluctuation RMS")
    mass_drift = (mass - mass[0]) / max(abs(float(mass[0])), 1.0e-30)
    axmass.plot(time, mass_drift, lw=1.8, color="#bb3e03", label="relative mass drift")
    axmass.plot(time, np.std(line_profiles, axis=1), lw=1.5, color="#2A9D8F", label="line-mean spread")
    axmass.legend(frameon=False, fontsize=8)
    style_axis(axmass, title="closed-trace scalar controls", xlabel="time", grid="both")

    summary = report["control_report_summary"]
    axtext.axis("off")
    axtext.text(
        0.02,
        0.96,
        "\n".join(
            [
                "Direct-coil closed/near-closed transient",
                f"source: {report['source_mode']}",
                f"lines: {report['n_field_lines']}",
                f"samples/line: {report['samples_per_line']}",
                f"closed fraction: {summary['closed_fraction']:.2f}",
                f"near-closed fraction: {summary['near_closed_fraction']:.2f}",
                f"return p95: {summary['return_distance_normalized_p95']:.2e}",
                f"final RMS: {report['final_fluctuation_rms']:.2e}",
                f"mass drift: {report['mass_relative_drift']:.2e}",
                f"media ready: {report['closed_control_media_ready']}",
                "No target, sheath, recycling, or neutral-loss terms.",
            ]
        ),
        transform=axtext.transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.96},
    )
    fig.suptitle("ESSOS direct-coil closed-control reduced transient", fontsize=15, fontweight="semibold")
    save_publication_figure(fig, path)
    return Path(path)


def save_essos_direct_coil_closed_control_transient_movie(
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Save a fixed-camera GIF for the direct-coil closed-trace transient."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    history = np.asarray(arrays["density_fluctuation_history"], dtype=np.float64)
    sampled = np.asarray(arrays["sampled_trajectories_xyz"], dtype=np.float64)
    time = np.asarray(arrays["time"], dtype=np.float64)
    vmax = float(np.asarray(arrays["movie_vmax"], dtype=np.float64)[0])
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    frame_indices = np.linspace(0, history.shape[0] - 1, min(18, history.shape[0]), dtype=int)
    stride = max(1, sampled.shape[1] // 120)
    points = sampled[:, ::stride, :].reshape((-1, 3))
    with tempfile.TemporaryDirectory(prefix="jax_drb_direct_coil_closed_movie_") as temp_dir:
        frame_paths: list[Path] = []
        for local_index, frame_index in enumerate(frame_indices):
            frame_path = Path(temp_dir) / f"frame_{local_index:03d}.png"
            fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
            axis = fig.add_subplot(1, 1, 1, projection="3d")
            color_values = history[frame_index, :, ::stride].reshape(-1)
            image = axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=color_values,
                s=5.0,
                cmap="coolwarm",
                norm=norm,
                linewidths=0.0,
                alpha=0.88,
            )
            _set_equal_3d(axis, sampled.reshape((-1, 3)))
            axis.view_init(elev=22.0, azim=42.0)
            axis.set_xlabel("X")
            axis.set_ylabel("Y")
            axis.set_zlabel("Z")
            axis.set_title(
                "Closed direct-coil trace: periodic density fluctuation\n"
                f"t={time[frame_index]:.3f}, no target/sheath/recycling losses",
                fontsize=11,
            )
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02, label=r"$\tilde{n}$")
            fig.savefig(frame_path, dpi=145, facecolor="white")
            plt.close(fig)
            frame_paths.append(frame_path)
        first = Image.open(frame_paths[0]).convert("RGB").quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        images = [first]
        for frame_path in frame_paths[1:]:
            images.append(Image.open(frame_path).convert("RGB").quantize(palette=first, dither=Image.Dither.NONE))
        images[0].save(resolved, save_all=True, append_images=images[1:], duration=130, loop=0)
        for image in images:
            image.close()
    return resolved


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
        "source_mode": str(
            inputs.get(
                "source_mode",
                "live_essos" if inputs["live_essos"] else "self_contained_contract",
            )
        ),
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


def _normalize_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    center = np.nanmean(array)
    scale = np.nanstd(array)
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    return (array - center) / scale


def _periodic_gradient(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return 0.5 * (np.roll(array, -1, axis=1) - np.roll(array, 1, axis=1))


def _periodic_laplacian(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.roll(array, -1, axis=1) - 2.0 * array + np.roll(array, 1, axis=1)


def _closed_trace_drive(
    phase: np.ndarray,
    normalized_major: np.ndarray,
    normalized_vertical: np.ndarray,
    phi: np.ndarray,
    *,
    scalar_time: float,
) -> np.ndarray:
    angle = 2.0 * np.pi * float(scalar_time)
    drive = (
        np.sin(2.0 * phase[None, :] - 0.18 * phi + 1.4 * angle)
        + 0.32 * np.cos(3.0 * phase[None, :] + 0.24 * normalized_major - 0.9 * angle)
        + 0.18 * normalized_vertical
    )
    return drive - np.mean(drive, axis=1, keepdims=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    return value


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


def _finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("-inf")
    return float(np.max(finite))


def _finite_range(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("inf")
    return float(np.max(finite) - np.min(finite))


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
