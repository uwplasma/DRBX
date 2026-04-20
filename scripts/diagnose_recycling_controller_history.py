#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from jax_drb.config.boutinp import NumericResolver, load_bout_input
from jax_drb.native.mesh import StructuredMesh, build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import advance_recycling_1d_implicit_history, compute_recycling_1d_rhs
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.reference import run_reference_case
from jax_drb.runtime.run_config import RunConfiguration


@dataclass(frozen=True)
class ControllerSeriesSummary:
    name: str
    max_abs_diff: float
    worst_step: int
    worst_time: float
    native_value: float
    reference_value: float


def controller_integral_series_from_term(
    integral_term: np.ndarray,
    *,
    controller_gain: float,
) -> np.ndarray:
    values = np.asarray(integral_term, dtype=np.float64)
    if controller_gain == 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return values / float(controller_gain)


def extract_scalar_series(dataset: Dataset, variable_name: str) -> np.ndarray:
    return np.asarray(dataset.variables[variable_name][:], dtype=np.float64).reshape(-1)


def extract_field_series_at_target_cell(
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


def build_series_summary(
    *,
    name: str,
    reference: np.ndarray,
    native: np.ndarray,
    time_points: np.ndarray,
) -> ControllerSeriesSummary:
    reference_values = np.asarray(reference, dtype=np.float64)
    native_values = np.asarray(native, dtype=np.float64)
    delta = np.abs(native_values - reference_values)
    worst_step = int(np.nanargmax(delta))
    return ControllerSeriesSummary(
        name=name,
        max_abs_diff=float(delta[worst_step]),
        worst_step=worst_step,
        worst_time=float(time_points[worst_step]),
        native_value=float(native_values[worst_step]),
        reference_value=float(reference_values[worst_step]),
    )


def _case_input_path(case_name: str, reference_root: Path) -> Path:
    if case_name == "recycling_1d_one_step":
        return reference_root / "tests/integrated/1D-recycling/data/BOUT.inp"
    if case_name == "recycling_dthe_one_step":
        return reference_root / "tests/integrated/1D-recycling-dthe/data/BOUT.inp"
    raise ValueError(f"unsupported case: {case_name}")


def _solver_mode_for_case(case_name: str) -> str:
    return "continuation" if case_name == "recycling_1d_one_step" else "bdf"


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
        solver_mode=_solver_mode_for_case(case_name),
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
            diagnostics.setdefault(name, []).append(np.asarray(values[0] if values.ndim == 4 and values.shape[0] == 1 else values, dtype=np.float64))
    stacked_diagnostics = {
        name: np.stack(values, axis=0)
        for name, values in diagnostics.items()
    }
    time_points = np.asarray([index * timestep for index in range(steps + 1)], dtype=np.float64)
    return history.variable_history, stacked_diagnostics, time_points, mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare controller integral/source history and target recycling source history against Hermes."
    )
    parser.add_argument("--case", default="recycling_1d_one_step", choices=("recycling_1d_one_step", "recycling_dthe_one_step"))
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--reference-binary", type=Path)
    parser.add_argument("--reference-workdir", type=Path)
    parser.add_argument("--timestep", type=float, default=25.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--x-index", type=int, default=0)
    parser.add_argument("--y-offset", type=int, default=0)
    parser.add_argument("--z-index", type=int, default=0)
    parser.add_argument("--target-edge", choices=("upper", "lower"), default="upper")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    input_path = _case_input_path(args.case, args.reference_root)
    config = load_bout_input(input_path)
    controller_species = _controller_species(config)
    controller_gains = _controller_gains(config, controller_species)
    recycled_species = _recycling_target_species(config)

    def _run_dense_reference() -> Path:
        workdir = Path(tempfile.mkdtemp(prefix=f"jaxdrb-{args.case}-controller-history-"))
        reference_result = run_reference_case(
            args.case,
            reference_root=args.reference_root,
            reference_binary=args.reference_binary,
            workdir=workdir,
            extra_overrides=(f"nout={args.steps}", f"timestep={args.timestep:g}"),
            keep_workdir=True,
        )
        return Path(reference_result.summary.workdir)

    reference_workdir = _run_dense_reference() if args.reference_workdir is None else args.reference_workdir

    reference_dump = reference_workdir / "BOUT.dmp.0.nc"
    with Dataset(reference_dump) as dataset:
        time_points = np.asarray(dataset.variables["t_array"][:], dtype=np.float64)
        reference_scalar_series: dict[str, np.ndarray] = {}
        reference_target_cell_series: dict[str, np.ndarray] = {}
        for species in controller_species:
            for prefix in ("density_feedback_src_mult_", "density_feedback_src_p_", "density_feedback_src_i_"):
                name = f"{prefix}{species}"
                if name in dataset.variables:
                    reference_scalar_series[name] = extract_scalar_series(dataset, name)
            integral_name = f"density_feedback_src_i_{species}"
            if integral_name in reference_scalar_series:
                reference_scalar_series[f"{species}_density_error_integral"] = controller_integral_series_from_term(
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

    _, native_diagnostics, native_time, mesh = _native_history_and_diagnostics(
        case_name=args.case,
        reference_root=args.reference_root,
        timestep=float(args.timestep),
        steps=int(args.steps),
    )
    if time_points.shape != native_time.shape:
        reference_workdir = _run_dense_reference()
        reference_dump = reference_workdir / "BOUT.dmp.0.nc"
        with Dataset(reference_dump) as dataset:
            time_points = np.asarray(dataset.variables["t_array"][:], dtype=np.float64)
            reference_scalar_series.clear()
            reference_target_cell_series.clear()
            for species in controller_species:
                for prefix in ("density_feedback_src_mult_", "density_feedback_src_p_", "density_feedback_src_i_"):
                    name = f"{prefix}{species}"
                    if name in dataset.variables:
                        reference_scalar_series[name] = extract_scalar_series(dataset, name)
                integral_name = f"density_feedback_src_i_{species}"
                if integral_name in reference_scalar_series:
                    reference_scalar_series[f"{species}_density_error_integral"] = controller_integral_series_from_term(
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
        if time_points.shape != native_time.shape:
            raise ValueError(f"time axis mismatch: reference {time_points.shape}, native {native_time.shape}")

    summaries: list[ControllerSeriesSummary] = []
    for name, reference_values in reference_scalar_series.items():
        if name == "density_feedback_src_i_d+":
            pass
        native_values: np.ndarray | None = None
        if name.endswith("_density_error_integral"):
            species = name.removesuffix("_density_error_integral")
            if species in native_diagnostics:
                native_values = np.asarray(native_diagnostics[species], dtype=np.float64)
        elif name in native_diagnostics:
            native_values = np.asarray(native_diagnostics[name], dtype=np.float64).reshape(-1)
        elif name.startswith("density_feedback_src_"):
            native_values = np.asarray(native_diagnostics.get(name), dtype=np.float64).reshape(-1) if name in native_diagnostics else None
        if native_values is None:
            if name.endswith("_density_error_integral"):
                species = name.removesuffix("_density_error_integral")
                values = native_diagnostics.get(f"density_feedback_src_i_{species}")
                if values is not None:
                    native_values = controller_integral_series_from_term(
                        np.asarray(values, dtype=np.float64).reshape(-1),
                        controller_gain=controller_gains[species],
                    )
        if native_values is None:
            continue
        summaries.append(build_series_summary(name=name, reference=reference_values, native=native_values, time_points=time_points))

    for name, reference_values in reference_target_cell_series.items():
        if name not in native_diagnostics:
            continue
        reference_series = extract_field_series_at_target_cell(
            reference_values,
            mesh,
            x_index=args.x_index,
            y_offset=args.y_offset,
            z_index=args.z_index,
            target_edge=args.target_edge,
        )
        native_series = extract_field_series_at_target_cell(
            native_diagnostics[name],
            mesh,
            x_index=args.x_index,
            y_offset=args.y_offset,
            z_index=args.z_index,
            target_edge=args.target_edge,
        )
        summaries.append(build_series_summary(name=name, reference=reference_series, native=native_series, time_points=time_points))

    summaries.sort(key=lambda item: item.max_abs_diff, reverse=True)

    print(f"case={args.case}")
    print(f"reference_workdir={reference_workdir}")
    for summary in summaries:
        print(
            f"{summary.name}: max_abs={summary.max_abs_diff:.8e} "
            f"worst_step={summary.worst_step} time={summary.worst_time:.8e} "
            f"native={summary.native_value:.8e} reference={summary.reference_value:.8e}"
        )

    if args.json_out is not None:
        payload: dict[str, Any] = {
            "case": args.case,
            "reference_workdir": str(reference_workdir),
            "series": [asdict(summary) for summary in summaries],
        }
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
