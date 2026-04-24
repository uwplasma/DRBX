from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from matplotlib import pyplot as plt
import numpy as np

from ..solver import (
    build_locality_sparsity,
    build_modulo_color_groups,
    build_sparse_difference_quotient_jacobian,
    prepare_sparse_difference_quotient_plan,
    solve_sparse_newton_system,
)
from .publication_plotting import annotate_bars, save_publication_figure, style_axis


@dataclass(frozen=True)
class ImplicitSolverProfileAuditArtifacts:
    report_json_path: Path
    report_plot_png_path: Path


def create_implicit_solver_profile_audit_package(
    *,
    output_root: str | Path,
    case_label: str = "implicit_solver_profile_audit",
) -> ImplicitSolverProfileAuditArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_implicit_solver_profile_audit_report()
    report_json_path = data_dir / f"{case_label}.json"
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report_plot_png_path = save_implicit_solver_profile_audit_plot(
        report,
        images_dir / f"{case_label}.png",
    )
    return ImplicitSolverProfileAuditArtifacts(
        report_json_path=report_json_path,
        report_plot_png_path=report_plot_png_path,
    )


def build_implicit_solver_profile_audit_report(
    *,
    active_shape: tuple[int, int, int] = (12, 1, 16),
    field_count: int = 3,
    repeats: int = 3,
) -> dict[str, object]:
    sparsity = build_locality_sparsity(
        active_shape,
        field_count=field_count,
        radii=(1, 0, 1),
        periodic_axes=(2,),
    )
    color_groups = build_modulo_color_groups(
        active_shape,
        field_count=field_count,
        color_periods=(2, 1, 4),
    )
    plan_started_at = perf_counter()
    plan = prepare_sparse_difference_quotient_plan(
        sparsity=sparsity,
        color_groups=color_groups,
    )
    plan_build_seconds = perf_counter() - plan_started_at
    state = np.linspace(0.25, 1.05, field_count * int(np.prod(active_shape)), dtype=np.float64)

    def residual(vector: np.ndarray) -> np.ndarray:
        fields = vector.reshape((field_count,) + active_shape)
        outputs: list[np.ndarray] = []
        for index in range(field_count):
            own = fields[index]
            left = np.roll(own, 1, axis=2)
            right = np.roll(own, -1, axis=2)
            coupled = fields[(index + 1) % field_count]
            outputs.append(np.sin(own) + 0.04 * (left - 2.0 * own + right) - 0.03 * coupled)
        return np.concatenate([output.ravel() for output in outputs])

    unplanned_seconds = _time_jacobian_builds(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        repeats=repeats,
        parallel_workers=1,
    )
    planned_serial_seconds = _time_jacobian_builds(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        repeats=repeats,
        parallel_workers=1,
        difference_plan=plan,
    )
    planned_parallel_seconds = _time_jacobian_builds(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        repeats=repeats,
        parallel_workers=2,
        difference_plan=plan,
    )
    reference_jacobian = build_sparse_difference_quotient_jacobian(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        parallel_workers=1,
    )
    planned_jacobian = build_sparse_difference_quotient_jacobian(
        residual,
        state,
        sparsity=sparsity,
        color_groups=color_groups,
        difference_plan=plan,
        parallel_workers=2,
    )
    max_jacobian_abs_diff = float(np.max(np.abs((planned_jacobian - reference_jacobian).toarray())))

    target = 0.85 + 0.05 * np.sin(np.linspace(0.0, np.pi, state.size, dtype=np.float64))

    def diagonal_residual(vector: np.ndarray) -> np.ndarray:
        return vector * vector - target * target

    diagonal_sparsity = build_locality_sparsity((state.size,), field_count=1, radii=(0,))
    diagonal_color_groups = build_modulo_color_groups((state.size,), field_count=1, color_periods=(8,))
    solved, info = solve_sparse_newton_system(
        diagonal_residual,
        np.maximum(state, 0.5),
        active_shape=(state.size,),
        sparsity=diagonal_sparsity,
        color_groups=diagonal_color_groups,
        residual_tolerance=1.0e-10,
        step_tolerance=1.0e-12,
        max_nonlinear_iterations=8,
        linear_restart=20,
        linear_maxiter=50,
        linear_rtol=1.0e-10,
        prefer_direct_linear_solve=True,
    )

    return {
        "case": "implicit_solver_profile_audit",
        "active_shape": list(active_shape),
        "field_count": int(field_count),
        "state_size": int(state.size),
        "sparsity_nnz": int(sparsity.nnz),
        "color_group_count": int(len(color_groups)),
        "repeats": int(repeats),
        "plan_build_seconds": float(plan_build_seconds),
        "jacobian_build_seconds": {
            "unplanned_serial_mean": float(np.mean(unplanned_seconds)),
            "planned_serial_mean": float(np.mean(planned_serial_seconds)),
            "planned_parallel_mean": float(np.mean(planned_parallel_seconds)),
        },
        "jacobian_build_samples": {
            "unplanned_serial": unplanned_seconds,
            "planned_serial": planned_serial_seconds,
            "planned_parallel": planned_parallel_seconds,
        },
        "speedups": {
            "planned_vs_unplanned_serial": _safe_ratio(np.mean(unplanned_seconds), np.mean(planned_serial_seconds)),
            "parallel_vs_planned_serial": _safe_ratio(np.mean(planned_serial_seconds), np.mean(planned_parallel_seconds)),
        },
        "max_jacobian_abs_diff": max_jacobian_abs_diff,
        "newton": {
            "residual_inf_norm": float(info.residual_inf_norm),
            "solution_max_abs_error": float(np.max(np.abs(solved - target))),
            "nonlinear_iterations": int(info.nonlinear_iterations),
            "linear_iterations": int(info.linear_iterations),
            "residual_evaluation_count": int(info.residual_evaluation_count),
            "residual_evaluation_seconds": float(info.residual_evaluation_seconds),
            "jacobian_refresh_count": int(info.jacobian_refresh_count),
            "jacobian_assembly_seconds": float(info.jacobian_assembly_seconds),
            "linear_solve_seconds": float(info.linear_solve_seconds),
            "line_search_seconds": float(info.line_search_seconds),
            "fallback_used": bool(info.fallback_used),
        },
        "notes": {
            "numerical_role": (
                "This audit verifies that the precomputed color-plan path is algebraically identical "
                "to the original sparse finite-difference Jacobian path and records the phase timings "
                "needed to interpret heavy recycling implicit solves."
            ),
            "paper_role": (
                "Use this as the methods/performance support figure for sparse finite-difference "
                "Jacobian assembly before showing full Hermes-backed recycling runtime comparisons."
            ),
        },
    }


def save_implicit_solver_profile_audit_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    timings = report["jacobian_build_seconds"]
    newton = report["newton"]
    labels = ["unplanned\nserial", "planned\nserial", "planned\n2 threads"]
    values = np.asarray(
        [
            float(timings["unplanned_serial_mean"]),
            float(timings["planned_serial_mean"]),
            float(timings["planned_parallel_mean"]),
        ],
        dtype=np.float64,
    )
    phase_labels = ["residual", "Jacobian", "linear solve", "line search"]
    phase_values = np.asarray(
        [
            float(newton["residual_evaluation_seconds"]),
            float(newton["jacobian_assembly_seconds"]),
            float(newton["linear_solve_seconds"]),
            float(newton["line_search_seconds"]),
        ],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x, np.maximum(values, 1.0e-12), color=["#6c757d", "#0a9396", "#ee9b00"])
    axes[0].set_xticks(x, labels)
    style_axis(
        axes[0],
        title="Colored finite-difference Jacobian assembly",
        ylabel="mean seconds",
        yscale="log",
        grid="y",
    )
    annotate_bars(axes[0], x, np.maximum(values, 1.0e-12), fmt="{:.2e}", fontsize=8.6)

    phase_x = np.arange(len(phase_labels))
    axes[1].bar(phase_x, np.maximum(phase_values, 1.0e-12), color="#005f73")
    axes[1].set_xticks(phase_x, phase_labels, rotation=20, ha="right")
    style_axis(
        axes[1],
        title="Sparse Newton phase timing on a diagonal nonlinear solve",
        ylabel="seconds",
        yscale="log",
        grid="y",
    )
    annotate_bars(axes[1], phase_x, np.maximum(phase_values, 1.0e-12), fmt="{:.2e}", fontsize=8.6)
    axes[1].text(
        0.03,
        0.96,
        (
            f"residual inf-norm: {float(newton['residual_inf_norm']):.1e}\n"
            f"solution max error: {float(newton['solution_max_abs_error']):.1e}\n"
            f"Jacobian refreshes: {int(newton['jacobian_refresh_count'])}"
        ),
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "#ced4da", "alpha": 0.92},
    )
    figure.suptitle(
        "Implicit solver audit: sparse finite-difference Jacobian plan and phase diagnostics",
        fontsize=12.8,
        fontweight="semibold",
    )
    save_publication_figure(figure, target)
    return target


def _time_jacobian_builds(
    residual,
    state: np.ndarray,
    *,
    sparsity,
    color_groups: tuple[tuple[int, ...], ...],
    repeats: int,
    parallel_workers: int,
    difference_plan=None,
) -> list[float]:
    samples: list[float] = []
    for _ in range(max(1, int(repeats))):
        started_at = perf_counter()
        build_sparse_difference_quotient_jacobian(
            residual,
            state,
            sparsity=sparsity,
            color_groups=color_groups,
            difference_plan=difference_plan,
            parallel_workers=parallel_workers,
        )
        samples.append(float(perf_counter() - started_at))
    return samples


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)
