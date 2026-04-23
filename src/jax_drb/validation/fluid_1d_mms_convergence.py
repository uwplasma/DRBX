from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from .publication_plotting import annotate_bars, save_publication_figure, style_axis

from ..config.boutinp import parse_bout_input
from ..native.fluid_1d import advance_mms_history, evaluate_field_option
from ..native.mesh import build_structured_mesh
from ..native.metrics import build_structured_metrics
from ..runtime.run_config import RunConfiguration


_MMS_TEMPLATE = """
nout = 50
timestep = {timestep}

MXG = 0

[mesh]
nx = 1
ny = {ny}
nz = 1
Ly = 10
dy = Ly / ny
J = 1

[solver]
mxstep = 10000
rtol = 1e-7
mms = true

[model]
components = i
normalise_metric = false
Nnorm = 1e18
Bnorm = 1
Tnorm = 5

[i]
type = evolve_density, evolve_pressure, evolve_momentum
charge = 1.0
AA = 2.0
thermal_conduction = false

[Ni]
solution = 1 - 0.1*sin(t - 2.0*y)
source = -0.1*cos(t - 2.0*y) + 0.0628318530717959*cos(2*t + y)

[Pi]
solution = 0.1*cos(t + 3.0*y) + 1
source = (0.0628318530717959*cos(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.0125663706143592*sin(2*t + y)*cos(t - 2.0*y)/(1 - 0.1*sin(t - 2.0*y))^2)*(0.0666666666666667*cos(t + 3.0*y) + 0.666666666666667) - 0.1*sin(t + 3.0*y) + 0.0628318530717959*(0.1*cos(t + 3.0*y) + 1)*cos(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.0188495559215388*sin(t + 3.0*y)*sin(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.0125663706143592*(0.1*cos(t + 3.0*y) + 1)*sin(2*t + y)*cos(t - 2.0*y)/(1 - 0.1*sin(t - 2.0*y))^2

[NVi]
solution = 0.2*sin(2*t + y)
source = -0.188495559215388*sin(t + 3.0*y) + 0.4*cos(2*t + y) + 0.0251327412287183*sin(2*t + y)*cos(2*t + y)/(1 - 0.1*sin(t - 2.0*y)) - 0.00251327412287184*sin(2*t + y)^2*cos(t - 2.0*y)/(1 - 0.1*sin(t - 2.0*y))^2
"""


@dataclass(frozen=True)
class Fluid1DMmsConvergenceArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    summary_plot_png_path: Path


def create_fluid_1d_mms_convergence_package(
    *,
    output_root: str | Path,
    case_label: str = "fluid_1d_mms_convergence",
    resolutions: tuple[int, ...] = (32, 64, 128),
    timestep: float = 0.05,
    steps: int = 2,
    substeps: int = 20,
) -> Fluid1DMmsConvergenceArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    report = build_fluid_1d_mms_convergence_report(
        resolutions=resolutions,
        timestep=timestep,
        steps=steps,
        substeps=substeps,
    )
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(
        arrays_npz_path,
        resolutions=np.asarray(report["resolutions"], dtype=np.int64),
        density_l2=np.asarray([entry["errors"]["density_l2"] for entry in report["runs"]], dtype=np.float64),
        pressure_l2=np.asarray([entry["errors"]["pressure_l2"] for entry in report["runs"]], dtype=np.float64),
        momentum_l2=np.asarray([entry["errors"]["momentum_l2"] for entry in report["runs"]], dtype=np.float64),
        density_order=np.asarray([entry["density_order"] for entry in report["observed_orders"]], dtype=np.float64),
        pressure_order=np.asarray([entry["pressure_order"] for entry in report["observed_orders"]], dtype=np.float64),
        momentum_order=np.asarray([entry["momentum_order"] for entry in report["observed_orders"]], dtype=np.float64),
    )

    summary_plot_png_path = save_fluid_1d_mms_convergence_plot(report, images_dir / f"{case_label}.png")
    return Fluid1DMmsConvergenceArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        summary_plot_png_path=summary_plot_png_path,
    )


def build_mms_config(*, ny: int, timestep: float) -> str:
    return _MMS_TEMPLATE.format(ny=int(ny), timestep=float(timestep))


def l2_error(numerical: np.ndarray, exact: np.ndarray, *, start: int, end: int) -> float:
    interior_numerical = np.asarray(numerical[:, start : end + 1, :], dtype=np.float64)
    interior_exact = np.asarray(exact[:, start : end + 1, :], dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(interior_numerical - interior_exact))))


def observed_order(coarse_error: float, fine_error: float) -> float:
    if coarse_error <= 0.0 or fine_error <= 0.0:
        return float("inf")
    return float(np.log(coarse_error / fine_error) / np.log(2.0))


def run_mms_resolution(*, ny: int, timestep: float, steps: int, substeps: int) -> dict[str, object]:
    config = parse_bout_input(build_mms_config(ny=ny, timestep=timestep))
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    history = advance_mms_history(
        config,
        section="i",
        mesh=mesh,
        metrics=metrics,
        atomic_mass=2.0,
        timestep=timestep,
        steps=steps,
        substeps=substeps,
    )
    final_time = float(steps * timestep)
    exact_density = evaluate_field_option(config, "Ni", "solution", mesh=mesh, time=final_time)
    exact_pressure = evaluate_field_option(config, "Pi", "solution", mesh=mesh, time=final_time)
    exact_momentum = evaluate_field_option(config, "NVi", "solution", mesh=mesh, time=final_time)

    density_error = l2_error(np.asarray(history.density_history[-1]), np.asarray(exact_density), start=mesh.ystart, end=mesh.yend)
    pressure_error = l2_error(np.asarray(history.pressure_history[-1]), np.asarray(exact_pressure), start=mesh.ystart, end=mesh.yend)
    momentum_error = l2_error(np.asarray(history.momentum_history[-1]), np.asarray(exact_momentum), start=mesh.ystart, end=mesh.yend)

    return {
        "ny": int(ny),
        "timestep": float(timestep),
        "steps": int(steps),
        "substeps": int(substeps),
        "final_time": final_time,
        "errors": {
            "density_l2": density_error,
            "pressure_l2": pressure_error,
            "momentum_l2": momentum_error,
        },
    }


def build_fluid_1d_mms_convergence_report(
    *,
    resolutions: tuple[int, ...] = (32, 64, 128),
    timestep: float = 0.05,
    steps: int = 2,
    substeps: int = 20,
) -> dict[str, object]:
    runs = [run_mms_resolution(ny=ny, timestep=timestep, steps=steps, substeps=substeps) for ny in resolutions]
    orders: list[dict[str, float | int]] = []
    for coarse, fine in zip(runs[:-1], runs[1:], strict=False):
        coarse_errors = coarse["errors"]
        fine_errors = fine["errors"]
        orders.append(
            {
                "from_ny": int(coarse["ny"]),
                "to_ny": int(fine["ny"]),
                "density_order": observed_order(float(coarse_errors["density_l2"]), float(fine_errors["density_l2"])),
                "pressure_order": observed_order(float(coarse_errors["pressure_l2"]), float(fine_errors["pressure_l2"])),
                "momentum_order": observed_order(float(coarse_errors["momentum_l2"]), float(fine_errors["momentum_l2"])),
            }
        )
    return {
        "family": "manufactured_solution_convergence",
        "case": "fluid_1d_mms_convergence",
        "operator_family": "fluid_1d_density_pressure_momentum",
        "literature_anchor": "Hermes-3 and broader verification literature use refinement studies and observed-order plots to separate operator correctness from benchmark validation.",
        "resolutions": [int(ny) for ny in resolutions],
        "timestep": float(timestep),
        "steps": int(steps),
        "substeps": int(substeps),
        "runs": runs,
        "observed_orders": orders,
        "min_observed_order": {
            "density": float(min(entry["density_order"] for entry in orders)),
            "pressure": float(min(entry["pressure_order"] for entry in orders)),
            "momentum": float(min(entry["momentum_order"] for entry in orders)),
        },
    }


def save_fluid_1d_mms_convergence_plot(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    runs = list(report["runs"])
    resolutions = np.asarray([entry["ny"] for entry in runs], dtype=np.float64)
    density_error = np.asarray([entry["errors"]["density_l2"] for entry in runs], dtype=np.float64)
    pressure_error = np.asarray([entry["errors"]["pressure_l2"] for entry in runs], dtype=np.float64)
    momentum_error = np.asarray([entry["errors"]["momentum_l2"] for entry in runs], dtype=np.float64)
    orders = list(report["observed_orders"])

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    axes[0].loglog(resolutions, density_error, marker="o", linewidth=2.0, color="#005f73", label="density")
    axes[0].loglog(resolutions, pressure_error, marker="o", linewidth=2.0, color="#ca6702", label="pressure")
    axes[0].loglog(resolutions, momentum_error, marker="o", linewidth=2.0, color="#3a86ff", label="momentum")
    style_axis(
        axes[0],
        title="Fluid 1D MMS refinement errors",
        xlabel="interior Ny resolution",
        ylabel="L2 error",
        xscale="log",
        yscale="log",
        grid="both",
    )
    axes[0].legend(frameon=False)

    order_labels = [f"{entry['from_ny']}→{entry['to_ny']}" for entry in orders]
    x = np.arange(len(order_labels))
    width = 0.25
    axes[1].bar(x - width, [entry["density_order"] for entry in orders], width=width, color="#005f73", label="density")
    axes[1].bar(x, [entry["pressure_order"] for entry in orders], width=width, color="#ca6702", label="pressure")
    axes[1].bar(x + width, [entry["momentum_order"] for entry in orders], width=width, color="#3a86ff", label="momentum")
    axes[1].axhline(2.0, color="#bb3e03", linestyle="--", linewidth=1.5, label="second order")
    axes[1].set_xticks(x, order_labels)
    style_axis(axes[1], title="Observed MMS refinement order", ylabel="observed order")
    axes[1].legend(frameon=False)
    annotate_bars(axes[1], x - width, np.asarray([entry["density_order"] for entry in orders], dtype=np.float64), fmt="{:.2f}", fontsize=8.5)
    annotate_bars(axes[1], x, np.asarray([entry["pressure_order"] for entry in orders], dtype=np.float64), fmt="{:.2f}", fontsize=8.5)
    annotate_bars(axes[1], x + width, np.asarray([entry["momentum_order"] for entry in orders], dtype=np.float64), fmt="{:.2f}", fontsize=8.5)
    figure.suptitle("Fluid 1D manufactured-solution convergence audit", fontsize=14.0, fontweight="semibold")
    save_publication_figure(figure, target)
    return target
