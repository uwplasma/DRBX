from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
from netCDF4 import Dataset
import numpy as np

from ..config.boutinp import NumericResolver, load_bout_input
from ..native.mesh import StructuredMesh, build_structured_mesh
from ..native.metrics import build_structured_metrics
from ..native.recycling_1d import advance_recycling_1d_implicit_history, compute_recycling_1d_rhs
from ..native.units import resolved_dataset_scalars
from ..parity.reference import run_reference_case
from ..runtime.run_config import RunConfiguration


@dataclass(frozen=True)
class ControllerFeedbackMetric:
    name: str
    max_abs_diff: float
    target: float
    passed: bool
    notes: str


@dataclass(frozen=True)
class ControllerFeedbackCampaignArtifacts:
    summary_json_path: Path
    arrays_npz_path: Path
    plot_png_path: Path


def create_controller_feedback_campaign_package(
    *,
    output_root: str | Path,
    reference_root: str | Path,
    reference_binary: str | Path | None = None,
    case_label: str = "controller_feedback_campaign",
    case_name: str = "recycling_1d_one_step",
    timestep: float = 25.0,
    steps: int = 4,
) -> ControllerFeedbackCampaignArtifacts:
    root = Path(output_root)
    data_dir = root / "data"
    images_dir = root / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_controller_feedback_campaign(
        reference_root=reference_root,
        reference_binary=reference_binary,
        case_name=case_name,
        timestep=timestep,
        steps=steps,
    )
    summary_payload = {
        "family": "controller_feedback",
        "case": case_name,
        "steps": int(steps),
        "timestep": float(timestep),
        "metric_count": len(metrics),
        "passed_metric_count": sum(1 for metric in metrics if metric.passed),
        "metrics": [
            {
                "name": metric.name,
                "max_abs_diff": float(metric.max_abs_diff),
                "target": float(metric.target),
                "passed": bool(metric.passed),
                "notes": metric.notes,
            }
            for metric in metrics
        ],
    }
    summary_json_path = data_dir / f"{case_label}.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    arrays_npz_path = data_dir / f"{case_label}.npz"
    np.savez_compressed(
        arrays_npz_path,
        metric_values=np.asarray([metric.max_abs_diff for metric in metrics], dtype=np.float64),
        metric_targets=np.asarray([metric.target for metric in metrics], dtype=np.float64),
        metric_pass=np.asarray([1.0 if metric.passed else 0.0 for metric in metrics], dtype=np.float64),
    )

    plot_png_path = images_dir / f"{case_label}.png"
    _save_controller_feedback_plot(metrics, plot_png_path)
    return ControllerFeedbackCampaignArtifacts(
        summary_json_path=summary_json_path,
        arrays_npz_path=arrays_npz_path,
        plot_png_path=plot_png_path,
    )


def build_controller_feedback_campaign(
    *,
    reference_root: str | Path,
    reference_binary: str | Path | None = None,
    case_name: str = "recycling_1d_one_step",
    timestep: float = 25.0,
    steps: int = 4,
) -> tuple[ControllerFeedbackMetric, ...]:
    reference_root_path = Path(reference_root)
    series = _build_controller_series_report(
        case_name=case_name,
        reference_root=reference_root_path,
        reference_binary=Path(reference_binary) if reference_binary is not None else None,
        timestep=timestep,
        steps=steps,
    )
    targets = {
        "d+_density_error_integral": 3.0e-3,
        "density_feedback_src_mult_d+": 5.0e-4,
        "density_feedback_src_p_d+": 5.0e-4,
        "density_feedback_src_i_d+": 2.0e-6,
        "Sd_target_recycle": 2.0e-4,
        "Sd+_feedback": 1.0e-12,
    }
    notes = {
        "d+_density_error_integral": "Accepted-step controller integral history stays close to the reference dense-history restart integral.",
        "density_feedback_src_mult_d+": "Controller source multiplier tracks the reference one-step controller surface.",
        "density_feedback_src_p_d+": "Controller proportional term remains close to the reference dense-history controller state.",
        "density_feedback_src_i_d+": "Controller integral term remains close to the reference dense-history controller state.",
        "Sd_target_recycle": "Target recycling source history stays bounded on the same controlled one-step history.",
        "Sd+_feedback": "Density-feedback source stays exactly zero when the configured source shape is zero.",
    }
    metrics: list[ControllerFeedbackMetric] = []
    for name in (
        "d+_density_error_integral",
        "density_feedback_src_mult_d+",
        "density_feedback_src_p_d+",
        "density_feedback_src_i_d+",
        "Sd_target_recycle",
        "Sd+_feedback",
    ):
        value = float(series[name])
        target = float(targets[name])
        metrics.append(
            ControllerFeedbackMetric(
                name=name,
                max_abs_diff=value,
                target=target,
                passed=value <= target,
                notes=notes[name],
            )
        )
    return tuple(metrics)


def _build_controller_series_report(
    *,
    case_name: str,
    reference_root: Path,
    reference_binary: Path | None,
    timestep: float,
    steps: int,
) -> dict[str, float]:
    input_path = _case_input_path(case_name, reference_root)
    config = load_bout_input(input_path)
    controller_species = _controller_species(config)
    controller_gains = _controller_gains(config, controller_species)
    recycled_species = _recycling_target_species(config)

    with tempfile.TemporaryDirectory(prefix=f"jaxdrb-{case_name}-controller-campaign-") as temp_dir:
        reference_result = run_reference_case(
            case_name,
            reference_root=reference_root,
            reference_binary=reference_binary,
            workdir=Path(temp_dir),
            extra_overrides=(f"nout={steps}", f"timestep={timestep:g}"),
            keep_workdir=True,
        )
        reference_dump = Path(reference_result.summary.workdir) / "BOUT.dmp.0.nc"
        with Dataset(reference_dump) as dataset:
            reference_scalar_series: dict[str, np.ndarray] = {}
            reference_target_cell_series: dict[str, np.ndarray] = {}
            for species in controller_species:
                for prefix in ("density_feedback_src_mult_", "density_feedback_src_p_", "density_feedback_src_i_"):
                    name = f"{prefix}{species}"
                    if name in dataset.variables:
                        reference_scalar_series[name] = _extract_scalar_series(dataset, name)
                integral_name = f"density_feedback_src_i_{species}"
                if integral_name in reference_scalar_series:
                    reference_scalar_series[f"{species}_density_error_integral"] = _controller_integral_series_from_term(
                        reference_scalar_series[integral_name],
                        controller_gain=controller_gains[species],
                    )
                source_name = f"S{species}_feedback"
                if source_name in dataset.variables:
                    reference_target_cell_series[source_name] = np.asarray(dataset.variables[source_name][:], dtype=np.float64)
            for species in recycled_species:
                source_name = f"S{species}_target_recycle"
                if source_name in dataset.variables:
                    reference_target_cell_series[source_name] = np.asarray(dataset.variables[source_name][:], dtype=np.float64)

        _, native_diagnostics, _, mesh = _native_history_and_diagnostics(
            case_name=case_name,
            reference_root=reference_root,
            timestep=timestep,
            steps=steps,
        )

    summary: dict[str, float] = {}
    for name, values in reference_scalar_series.items():
        if name in native_diagnostics:
            native = np.asarray(native_diagnostics[name], dtype=np.float64).reshape(-1)
        elif name.endswith("_density_error_integral"):
            species = name.removesuffix("_density_error_integral")
            integral_name = f"density_feedback_src_i_{species}"
            native = _controller_integral_series_from_term(
                np.asarray(native_diagnostics[integral_name], dtype=np.float64).reshape(-1),
                controller_gain=controller_gains[species],
            )
        else:
            raise KeyError(name)
        reference = np.asarray(values, dtype=np.float64).reshape(-1)
        summary[name] = float(np.max(np.abs(native - reference)))
    for name, values in reference_target_cell_series.items():
        native = _extract_field_series_at_target_cell(np.asarray(native_diagnostics[name], dtype=np.float64), mesh)
        reference = _extract_field_series_at_target_cell(np.asarray(values, dtype=np.float64), mesh)
        summary[name] = float(np.max(np.abs(native - reference)))
    return summary


def _case_input_path(case_name: str, reference_root: Path) -> Path:
    if case_name == "recycling_1d_one_step":
        return reference_root / "tests/integrated/1D-recycling/data/BOUT.inp"
    raise ValueError(f"unsupported controller campaign case: {case_name}")


def _controller_species(config) -> tuple[str, ...]:
    names: list[str] = []
    for name in config.section_names():
        if not config.has_option(name, "type"):
            continue
        raw_types = config.parsed(name, "type")
        if isinstance(raw_types, tuple):
            component_types = tuple(str(part).strip(" ()") for part in raw_types)
        else:
            component_types = tuple(part.strip(" ()") for part in str(raw_types).split(","))
        if "upstream_density_feedback" in component_types:
            names.append(name)
    return tuple(sorted(names))


def _controller_gains(config, species_names: tuple[str, ...]) -> dict[str, float]:
    resolver = NumericResolver(config)
    return {
        name: float(resolver.resolve(name, "density_controller_i")) if config.has_option(name, "density_controller_i") else 1.0e-3
        for name in species_names
    }


def _recycling_target_species(config) -> tuple[str, ...]:
    names: list[str] = []
    for name in config.section_names():
        if config.has_option(name, "target_recycle") and bool(config.parsed(name, "target_recycle")):
            recycle_as = str(config.parsed(name, "recycle_as")) if config.has_option(name, "recycle_as") else ""
            if recycle_as:
                names.append(recycle_as)
    return tuple(sorted(set(names)))


def _controller_integral_series_from_term(integral_term: np.ndarray, *, controller_gain: float) -> np.ndarray:
    values = np.asarray(integral_term, dtype=np.float64)
    if controller_gain == 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return values / float(controller_gain)


def _extract_scalar_series(dataset: Dataset, variable_name: str) -> np.ndarray:
    return np.asarray(dataset.variables[variable_name][:], dtype=np.float64).reshape(-1)


def _extract_field_series_at_target_cell(
    values: np.ndarray,
    mesh: StructuredMesh,
    *,
    x_index: int = 0,
    y_offset: int = 0,
    z_index: int = 0,
    target_edge: str = "upper",
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4:
        raise ValueError(f"expected a 4D time series, got shape {array.shape}")
    x_global = mesh.xstart + int(x_index)
    if target_edge == "upper":
        y_global = mesh.yend - int(y_offset)
    else:
        y_global = mesh.ystart + int(y_offset)
    return np.asarray(array[:, x_global, y_global, int(z_index)], dtype=np.float64)


def _native_history_and_diagnostics(
    *,
    case_name: str,
    reference_root: Path,
    timestep: float,
    steps: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, StructuredMesh]:
    input_path = _case_input_path(case_name, reference_root)
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)
    history = advance_recycling_1d_implicit_history(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        timestep=timestep,
        steps=steps,
        solver_mode="continuation",
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=30,
    )
    diagnostics: dict[str, list[np.ndarray]] = {}
    for step in range(steps + 1):
        field_overrides = {
            name: np.asarray(values[step], dtype=np.float64)
            for name, values in history.variable_history.items()
            if values.ndim == 4
        }
        feedback_integrals = {
            name: float(values[step])
            for name, values in history.feedback_integral_history.items()
        }
        rhs = compute_recycling_1d_rhs(
            config,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            field_overrides=field_overrides,
            feedback_integrals=feedback_integrals,
        )
        for name, values in rhs.variables.items():
            diagnostics.setdefault(name, []).append(
                np.asarray(values[0] if values.ndim == 4 and values.shape[0] == 1 else values, dtype=np.float64)
            )
    stacked_diagnostics = {name: np.stack(values, axis=0) for name, values in diagnostics.items()}
    time_points = np.asarray([index * timestep for index in range(steps + 1)], dtype=np.float64)
    return history.variable_history, stacked_diagnostics, time_points, mesh


def _save_controller_feedback_plot(metrics: tuple[ControllerFeedbackMetric, ...], path: Path) -> None:
    labels = [metric.name.replace("_", "\n") for metric in metrics]
    values = [metric.max_abs_diff for metric in metrics]
    targets = [metric.target for metric in metrics]
    colors = ["#0a9396" if metric.passed else "#bb3e03" for metric in metrics]
    figure, axis = plt.subplots(figsize=(12.5, 6.5), constrained_layout=True)
    x = np.arange(len(metrics))
    axis.bar(x, values, color=colors, alpha=0.92, label="max |native - reference|")
    axis.plot(x, targets, color="#3a86ff", marker="o", linewidth=2.0, label="gate")
    axis.set_xticks(x, labels)
    axis.set_ylabel("max absolute difference")
    axis.set_title("Controller-feedback dense-history gate")
    axis.grid(alpha=0.25, axis="y")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)
