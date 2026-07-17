"""Slim native runner for the proved deck models.

This module supersedes the historical ``native.runner`` recycling/reference
lane.  It keeps only the accuracy-tested execution branches that back the
``dkx run`` command:

* single-component ``evolve_density`` (one-rhs),
* anomalous diffusion (:mod:`dkx.native.transport`),
* periodic fluid MMS (:mod:`dkx.native.fluid_1d`),
* electrostatic vorticity (:mod:`dkx.native.vorticity`).

It imports only kept modules and carries slim, dependency-free copies of the
portable-summary / portable-array helpers so the CLI can write run artifacts
without the removed ``parity`` package.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping as ABCMapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver, load_bout_input
from ..runtime import runtime_numpy_dtype
from ..runtime.output import RestartBundle, build_run_event, print_run_event
from ..runtime.run_config import RunConfiguration
from .expression import ArrayExpressionEvaluator
from .fluid_1d import Fluid1DState, advance_mms_history, compute_mms_rhs, initialize_mms_state
from .mesh import (
    StructuredMesh,
    apply_field_boundaries,
    broadcast_to_field_shape,
    build_structured_mesh,
)
from .metrics import StructuredMetrics, build_structured_metrics
from .transport import advance_anomalous_diffusion_history
from .units import resolved_dataset_scalars
from .vorticity import (
    advance_vorticity_history,
    apply_vorticity_boundaries,
    build_vorticity_operator,
    compute_vorticity_rhs,
)


# --------------------------------------------------------------------------- #
# Result / restart containers (formerly native.runner_state)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NativeRunResult:
    payload: Mapping[str, Any]
    variables: Mapping[str, Any]
    time_points: tuple[float, ...]
    run_config: RunConfiguration
    mesh: StructuredMesh
    metrics: StructuredMetrics
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeExecutionResult:
    time_points: tuple[float, ...]
    variables: Mapping[str, Any]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __iter__(self):
        yield self.time_points
        yield self.variables


def coerce_native_execution_result(result: object) -> NativeExecutionResult:
    if isinstance(result, NativeExecutionResult):
        return result
    time_points, variables = result  # type: ignore[misc]
    return NativeExecutionResult(time_points=tuple(time_points), variables=variables)


@dataclass(frozen=True)
class NativeRestartState:
    time_offset: float
    completed_steps: int
    configured_timestep: float
    variables: Mapping[str, np.ndarray]


def _restart_variable_names(run_config: RunConfiguration) -> tuple[str, ...]:
    implementations = tuple(component.implementation for component in run_config.components)
    sections = tuple(dict.fromkeys(component.section for component in run_config.components))
    if implementations == ("evolve_density", "evolve_pressure", "anomalous_diffusion") and len(sections) == 1:
        section = sections[0]
        return (f"N{section}", f"P{section}")
    if implementations == ("evolve_density", "evolve_pressure", "evolve_momentum") and len(sections) == 1:
        section = sections[0]
        return (f"N{section}", f"P{section}", f"NV{section}")
    if implementations == ("vorticity",):
        return ("Vort",)
    return ()


def build_restart_state(result: NativeRunResult, *, parity_mode: str) -> RestartBundle | None:
    names = _restart_variable_names(result.run_config)
    if not names:
        return None
    dtype = runtime_numpy_dtype()
    final_state = {
        name: np.asarray(result.variables[name][-1], dtype=dtype)
        for name in names
        if name in result.variables
    }
    if tuple(final_state) != names:
        return None
    return RestartBundle(
        case_name=str(result.payload.get("case_name", "run")),
        parity_mode=parity_mode,
        component_labels=tuple(request.label for request in result.run_config.components),
        current_time=float(result.time_points[-1]) if result.time_points else 0.0,
        completed_steps=max(len(result.time_points) - 1, 0),
        configured_timestep=float(result.run_config.time.timestep),
        state_variables=final_state,
    )


# --------------------------------------------------------------------------- #
# Portable summary / array payload helpers (formerly parity.portable / arrays)
# --------------------------------------------------------------------------- #
def _summarize_array(name: str, data: np.ndarray, dimension_names: tuple[str, ...]) -> dict[str, Any]:
    delta = None
    if data.ndim >= 1 and data.shape[0] >= 2:
        delta = float(np.max(np.abs(data[-1] - data[0])))
    if len(dimension_names) == data.ndim:
        dimensions = dimension_names
    else:
        dimensions = tuple(["t", *[f"dim_{index}" for index in range(1, data.ndim)]])
    return {
        "name": name,
        "dimensions": list(dimensions),
        "shape": [int(value) for value in data.shape],
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
        "mean": float(np.mean(data)),
        "max_abs_delta_last_first": delta,
    }


def build_portable_summary_payload(
    *,
    case_name: str,
    parity_mode: str,
    capability_tier: str,
    compare_variables: tuple[str, ...],
    component_labels: tuple[str, ...],
    dimensions: Mapping[str, int],
    time_points: tuple[float, ...],
    dataset_scalars: Mapping[str, float],
    variables: Mapping[str, Any],
    overrides: tuple[str, ...] = (),
    configured_nout: int | None = None,
    configured_timestep: float | None = None,
    producer: str = "dkx",
) -> dict[str, Any]:
    summary_dimensions = tuple(dimensions)
    summaries = {
        name: _summarize_array(name, np.asarray(variables[name], dtype=np.float64), summary_dimensions)
        for name in compare_variables
        if name in variables
    }
    payload: dict[str, Any] = {
        "case_name": case_name,
        "parity_mode": parity_mode,
        "capability_tier": capability_tier,
        "producer": producer,
        "overrides": list(overrides),
        "compare_variables": list(compare_variables),
        "component_labels": list(component_labels),
        "dimensions": dict(dimensions),
        "time_points": list(time_points),
        "dataset_scalars": {key: float(value) for key, value in dataset_scalars.items()},
        "variable_summaries": summaries,
        "effective_output_points": len(time_points),
    }
    if configured_nout is not None:
        payload["configured_nout"] = configured_nout
    if configured_timestep is not None:
        payload["configured_timestep"] = configured_timestep
    return payload


def write_portable_summary_payload(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return target


def build_portable_array_payload(
    *,
    case_name: str,
    parity_mode: str,
    capability_tier: str,
    compare_variables: tuple[str, ...],
    component_labels: tuple[str, ...],
    dimensions: Mapping[str, int],
    time_points: tuple[float, ...],
    dataset_scalars: Mapping[str, float],
    variables: Mapping[str, Any],
    variable_dimensions: Mapping[str, tuple[str, ...]] | None = None,
    overrides: tuple[str, ...] = (),
    configured_nout: int | None = None,
    configured_timestep: float | None = None,
    producer: str = "dkx",
) -> dict[str, Any]:
    summary_dimensions = tuple(dimensions)
    payload_variables: dict[str, np.ndarray] = {}
    payload_variable_dimensions: dict[str, list[str]] = {}
    for name in compare_variables:
        if name not in variables:
            continue
        array = np.asarray(variables[name], dtype=np.float64)
        payload_variables[name] = array
        if variable_dimensions is not None and name in variable_dimensions:
            dims = variable_dimensions[name]
        elif len(summary_dimensions) == array.ndim:
            dims = summary_dimensions
        else:
            dims = tuple(["t", *[f"dim_{index}" for index in range(1, array.ndim)]])
        payload_variable_dimensions[name] = list(dims)

    payload: dict[str, Any] = {
        "case_name": case_name,
        "parity_mode": parity_mode,
        "capability_tier": capability_tier,
        "producer": producer,
        "overrides": list(overrides),
        "compare_variables": list(compare_variables),
        "component_labels": list(component_labels),
        "dimensions": dict(dimensions),
        "time_points": list(time_points),
        "dataset_scalars": {key: float(value) for key, value in dataset_scalars.items()},
        "variable_dimensions": payload_variable_dimensions,
        "variables": payload_variables,
        "effective_output_points": len(time_points),
    }
    if configured_nout is not None:
        payload["configured_nout"] = configured_nout
    if configured_timestep is not None:
        payload["configured_timestep"] = configured_timestep
    return payload


def write_portable_array_payload(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in payload.items() if key != "variables"}
    arrays = {
        f"var__{name}": np.asarray(value, dtype=np.float64)
        for name, value in payload.get("variables", {}).items()
    }
    np.savez_compressed(target, __metadata__=json.dumps(metadata, sort_keys=True), **arrays)
    return target


def load_portable_array_payload(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["__metadata__"]))
        variables = {
            key.removeprefix("var__"): np.asarray(payload[key], dtype=np.float64)
            for key in payload.files
            if key.startswith("var__")
        }
    metadata["variables"] = variables
    return metadata


# --------------------------------------------------------------------------- #
# Execution helpers
# --------------------------------------------------------------------------- #
def _effective_output_steps(parity_mode: str, *, configured_nout: int) -> int:
    if parity_mode == "one_rhs":
        return 0
    if parity_mode == "one_step":
        return 1
    return configured_nout


def _default_overrides(parity_mode: str) -> tuple[str, ...]:
    if parity_mode == "one_rhs":
        return ("nout=0",)
    if parity_mode == "one_step":
        return ("nout=1",)
    return ()


def _prepare_compare_variables(
    variables: Mapping[str, Any],
    mesh: StructuredMesh,
    *,
    trim_x_guards: bool,
    trim_y_guards: bool,
) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for name, value in variables.items():
        array = np.asarray(value, dtype=np.float64)
        if trim_x_guards and array.ndim >= 4 and array.shape[1] > 2 * mesh.mxg:
            array = array[:, mesh.mxg : -mesh.mxg, ...]
        if trim_y_guards and array.ndim >= 4 and array.shape[2] > 2 * mesh.myg:
            array = array[:, :, mesh.myg : -mesh.myg, ...]
        prepared[name] = array
    return prepared


# --------------------------------------------------------------------------- #
# Public run entry points
# --------------------------------------------------------------------------- #
def run_input_case(
    input_path: str | Path,
    *,
    case_name: str | None = None,
    parity_mode: str = "manual",
    compare_variables: tuple[str, ...] = (),
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
    verbose: bool = False,
    verbosity: str = "detailed",
    event_logger: Callable[[ABCMapping[str, Any]], None] | None = None,
) -> NativeRunResult:
    config = load_bout_input(input_path)
    resolved_case_name = case_name or Path(input_path).stem
    return run_config_case(
        config,
        case_name=resolved_case_name,
        parity_mode=parity_mode,
        compare_variables=compare_variables,
        restart_state=restart_state,
        output_steps=output_steps,
        verbose=verbose,
        verbosity=verbosity,
        event_logger=event_logger,
    )


def run_config_case(
    config: BoutConfig,
    *,
    case_name: str,
    parity_mode: str,
    compare_variables: tuple[str, ...] = (),
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
    verbose: bool = False,
    verbosity: str = "detailed",
    event_logger: Callable[[ABCMapping[str, Any]], None] | None = None,
) -> NativeRunResult:
    event_sink = event_logger
    if event_sink is None and verbose:
        event_sink = lambda event: print_run_event(event, verbosity=verbosity)

    def emit(stage: str, message: str, **details: Any) -> None:
        if event_sink is None:
            return
        event_sink(build_run_event(stage=stage, message=message, details=details or None))

    run_config = RunConfiguration.from_config(config)
    emit(
        "configuration",
        "Resolved native run configuration",
        case_name=case_name,
        parity_mode=parity_mode,
        capability_tier="native_exact",
        nout=run_config.time.nout,
        timestep=run_config.time.timestep,
        components=",".join(component.label for component in run_config.components),
        restart=restart_state is not None,
    )
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    emit(
        "mesh",
        "Built structured mesh and metrics",
        nx=mesh.nx,
        ny=mesh.local_ny,
        nz=mesh.nz,
        mxg=mesh.mxg,
        myg=mesh.myg,
        file=run_config.mesh.file or "<analytic mesh>",
    )
    execution = coerce_native_execution_result(
        _execute_supported_case(
            config,
            run_config,
            mesh,
            metrics,
            parity_mode=parity_mode,
            restart_state=restart_state,
            output_steps=output_steps,
        )
    )
    time_points = execution.time_points
    variables = execution.variables
    emit(
        "run",
        "Native solver completed",
        stored_states=len(time_points),
        compare_variables=",".join(compare_variables or tuple(variables)),
    )
    compare_names = compare_variables or tuple(variables)
    trimmed_variables = _prepare_compare_variables(
        variables,
        mesh,
        trim_x_guards=False,
        trim_y_guards=False,
    )
    dataset_scalars = resolved_dataset_scalars(run_config)
    payload = build_portable_summary_payload(
        case_name=case_name,
        parity_mode=parity_mode,
        capability_tier="native_exact",
        compare_variables=compare_names,
        component_labels=tuple(component.label for component in run_config.components),
        dimensions={"t": len(time_points), "x": mesh.nx, "y": mesh.local_ny, "z": mesh.nz},
        time_points=time_points,
        dataset_scalars=dataset_scalars,
        variables={name: np.asarray(value, dtype=np.float64) for name, value in trimmed_variables.items()},
        overrides=_default_overrides(parity_mode),
        configured_nout=run_config.time.nout,
        configured_timestep=run_config.time.timestep,
        producer="dkx",
    )
    emit(
        "summary",
        "Prepared portable summary payload",
        variables=",".join(sorted(trimmed_variables)),
        time_start=time_points[0] if time_points else 0.0,
        time_end=time_points[-1] if time_points else 0.0,
    )
    return NativeRunResult(
        payload=payload,
        variables=trimmed_variables,
        time_points=time_points,
        run_config=run_config,
        mesh=mesh,
        metrics=metrics,
        diagnostics=execution.diagnostics,
    )


def _execute_supported_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    *,
    parity_mode: str,
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
) -> NativeExecutionResult | tuple[tuple[float, ...], dict[str, Any]]:
    implementations = tuple(component.implementation for component in run_config.components)
    if len(run_config.components) == 1 and implementations == ("evolve_density",):
        component = run_config.components[0]
        variable_name = f"N{component.section}"
        if parity_mode != "one_rhs":
            raise NotImplementedError(
                "Native single-component density execution currently supports one_rhs parity only."
            )
        field_values = _initialize_species_field(config, variable_name, mesh)
        return (0.0,), {variable_name: field_values[None, ...]}

    if _is_supported_diffusion_case(run_config):
        return _execute_diffusion_case(
            config,
            run_config,
            mesh,
            metrics,
            parity_mode=parity_mode,
            restart_state=restart_state,
            output_steps=output_steps,
        )

    if _is_supported_periodic_fluid_mms_case(config, run_config, mesh, metrics):
        return _execute_periodic_fluid_mms_case(
            config,
            run_config,
            mesh,
            metrics,
            parity_mode=parity_mode,
            restart_state=restart_state,
            output_steps=output_steps,
        )

    if _is_supported_electrostatic_vorticity_case(config, run_config, mesh, metrics):
        return _execute_electrostatic_vorticity_case(
            config,
            run_config,
            mesh,
            metrics,
            parity_mode=parity_mode,
            restart_state=restart_state,
            output_steps=output_steps,
        )

    raise NotImplementedError(
        "Native execution is not implemented for the configured component set: "
        + ", ".join(component.label for component in run_config.components)
    )


def _execute_diffusion_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    *,
    parity_mode: str,
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    dtype = runtime_numpy_dtype()
    section = run_config.components[0].section
    density_name = f"N{section}"
    pressure_name = f"P{section}"
    if restart_state is None:
        density = _initialize_species_field(config, density_name, mesh)
        pressure = _initialize_species_field(config, pressure_name, mesh)
        time_offset = 0.0
    else:
        density = np.asarray(restart_state.variables[density_name], dtype=dtype)
        pressure = np.asarray(restart_state.variables[pressure_name], dtype=dtype)
        time_offset = restart_state.time_offset
    if not np.allclose(np.asarray(density), np.asarray(pressure), rtol=1e-12, atol=1e-12):
        raise NotImplementedError(
            "Native anomalous diffusion currently requires identical density and pressure initial states."
        )
    if not np.allclose(np.asarray(metrics.g23), 0.0, rtol=1e-12, atol=1e-12):
        raise NotImplementedError("Native anomalous diffusion currently supports g23 = 0 structured metrics only.")
    density_boundary = _field_boundary_kind(config, density_name)
    pressure_boundary = _field_boundary_kind(config, pressure_name)

    if bool(config.parsed(section, "thermal_conduction")) if config.has_option(section, "thermal_conduction") else True:
        raise NotImplementedError("Native one-step anomalous diffusion currently requires thermal_conduction = false.")

    resolver = NumericResolver(config)
    scalars = resolved_dataset_scalars(run_config)
    anomalous_D = resolver.resolve(section, "anomalous_D") / (scalars["rho_s0"] * scalars["rho_s0"] * scalars["Omega_ci"])
    if config.has_option(section, "anomalous_chi") and abs(resolver.resolve(section, "anomalous_chi")) > 0.0:
        raise NotImplementedError("Native one-step anomalous diffusion does not yet support anomalous_chi.")
    if config.has_option(section, "anomalous_nu") and abs(resolver.resolve(section, "anomalous_nu")) > 0.0:
        raise NotImplementedError("Native one-step anomalous diffusion does not yet support anomalous_nu.")

    steps = output_steps if output_steps is not None else _effective_output_steps(parity_mode, configured_nout=run_config.time.nout)
    history = advance_anomalous_diffusion_history(
        density,
        pressure,
        mesh=mesh,
        metrics=metrics,
        anomalous_D=anomalous_D,
        density_boundary=density_boundary,
        pressure_boundary=pressure_boundary,
        timestep=run_config.time.timestep,
        steps=steps,
    )
    time_points = tuple(time_offset + run_config.time.timestep * index for index in range(steps + 1))
    return time_points, {
        density_name: np.asarray(history.density_history, dtype=dtype),
        pressure_name: np.asarray(history.pressure_history, dtype=dtype),
    }


def _initialize_species_field(config: BoutConfig, variable_name: str, mesh: StructuredMesh) -> Any:
    if not config.has_section(variable_name):
        raise KeyError(f"Missing initial condition section for {variable_name}.")
    if config.has_option(variable_name, "function"):
        option_name = "function"
    elif config.has_option(variable_name, "solution"):
        option_name = "solution"
    else:
        raise KeyError(f"Missing initial condition function or solution for {variable_name}.")
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    field_values = broadcast_to_field_shape(
        evaluator.evaluate(config.raw(variable_name, option_name), current_section=variable_name),
        mesh,
    )
    field_values = apply_field_boundaries(field_values, mesh, x_boundary=_field_boundary_kind(config, variable_name))
    return field_values


def _field_boundary_kind(config: BoutConfig, variable_name: str) -> str:
    if config.has_option(variable_name, "bndry_all"):
        return str(config.parsed(variable_name, "bndry_all")).strip().lower()
    return "dirichlet_zero"


def _is_supported_diffusion_case(run_config: RunConfiguration) -> bool:
    implementations = tuple(component.implementation for component in run_config.components)
    if implementations != ("evolve_density", "evolve_pressure", "anomalous_diffusion"):
        return False
    sections = {component.section for component in run_config.components}
    return len(sections) == 1


def _execute_periodic_fluid_mms_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    *,
    parity_mode: str,
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    section = run_config.components[0].section
    atomic_mass = float(config.parsed(section, "AA"))
    if parity_mode == "one_rhs":
        state = initialize_mms_state(config, section=section, mesh=mesh)
        rhs = compute_mms_rhs(config, state, section=section, mesh=mesh, metrics=metrics, atomic_mass=atomic_mass, time=0.0)
        return (0.0,), {
            f"ddt(N{section})": np.asarray(rhs.density[None, ...], dtype=np.float64),
            f"ddt(P{section})": np.asarray(rhs.pressure[None, ...], dtype=np.float64),
            f"ddt(NV{section})": np.asarray(rhs.momentum[None, ...], dtype=np.float64),
        }

    steps = output_steps if output_steps is not None else _effective_output_steps(parity_mode, configured_nout=run_config.time.nout)
    initial_state = (
        None
        if restart_state is None
        else Fluid1DState(
            density=jnp.asarray(restart_state.variables[f"N{section}"], dtype=jnp.float64),
            pressure=jnp.asarray(restart_state.variables[f"P{section}"], dtype=jnp.float64),
            momentum=jnp.asarray(restart_state.variables[f"NV{section}"], dtype=jnp.float64),
        )
    )
    time_offset = 0.0 if restart_state is None else restart_state.time_offset
    history = advance_mms_history(
        config,
        section=section,
        mesh=mesh,
        metrics=metrics,
        atomic_mass=atomic_mass,
        timestep=run_config.time.timestep,
        steps=steps,
        substeps=20,
        initial_state=initial_state,
        start_time=time_offset,
    )
    time_points = tuple(time_offset + run_config.time.timestep * index for index in range(steps + 1))
    return time_points, {
        f"N{section}": np.asarray(history.density_history, dtype=np.float64),
        f"P{section}": np.asarray(history.pressure_history, dtype=np.float64),
        f"NV{section}": np.asarray(history.momentum_history, dtype=np.float64),
    }


def _execute_electrostatic_vorticity_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    *,
    parity_mode: str,
    restart_state: NativeRestartState | None = None,
    output_steps: int | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    initial = (
        apply_vorticity_boundaries(_initialize_species_field(config, "Vort", mesh), mesh)
        if restart_state is None
        else apply_vorticity_boundaries(np.asarray(restart_state.variables["Vort"], dtype=np.float64), mesh)
    )
    time_offset = 0.0 if restart_state is None else restart_state.time_offset
    average_atomic_mass = float(config.parsed("vorticity", "average_atomic_mass")) if config.has_option("vorticity", "average_atomic_mass") else 2.0
    operator = build_vorticity_operator(mesh=mesh, metrics=metrics, average_atomic_mass=average_atomic_mass)

    if parity_mode == "one_rhs":
        rhs = compute_vorticity_rhs(initial, mesh=mesh, metrics=metrics, operator=operator)
        return (0.0,), {"ddt(Vort)": np.asarray(rhs.vorticity[None, ...], dtype=np.float64)}

    steps = output_steps if output_steps is not None else _effective_output_steps(parity_mode, configured_nout=run_config.time.nout)
    history = advance_vorticity_history(
        initial,
        mesh=mesh,
        metrics=metrics,
        operator=operator,
        timestep=run_config.time.timestep,
        steps=steps,
        start_time=time_offset,
        rtol=1e-6,
        atol=1e-8,
        mxstep=20000,
    )
    time_points = tuple(time_offset + run_config.time.timestep * index for index in range(steps + 1))
    return time_points, {
        "Vort": np.asarray(history.vorticity_history, dtype=np.float64),
        "phi": np.asarray(history.potential_history, dtype=np.float64),
    }


def _is_supported_periodic_fluid_mms_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> bool:
    implementations = tuple(component.implementation for component in run_config.components)
    if implementations != ("evolve_density", "evolve_pressure", "evolve_momentum"):
        return False
    if len({component.section for component in run_config.components}) != 1:
        return False
    section = run_config.components[0].section
    if run_config.mesh.nx != 1 or run_config.mesh.nz != 1:
        return False
    if run_config.solver.mms is not True:
        return False
    if bool(config.parsed(section, "thermal_conduction")) if config.has_option(section, "thermal_conduction") else True:
        return False
    if bool(config.parsed(section, "p_div_v")) if config.has_option(section, "p_div_v") else False:
        return False
    if not _uniform_identity_parallel_metric(mesh, metrics=metrics):
        return False
    return all(
        config.has_section(name) and config.has_option(name, "solution") and config.has_option(name, "source")
        for name in (f"N{section}", f"P{section}", f"NV{section}")
    )


def _is_supported_electrostatic_vorticity_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> bool:
    if tuple(component.implementation for component in run_config.components) != ("vorticity",):
        return False
    if run_config.mesh.ny != 1 or run_config.mesh.myg != 0:
        return False
    if mesh.mxg != 2:
        return False
    if not config.has_section("Vort") or not config.has_option("Vort", "function"):
        return False
    option_defaults = {
        "diamagnetic": False,
        "diamagnetic_polarisation": False,
        "bndry_flux": False,
        "poloidal_flows": False,
        "split_n0": False,
        "phi_dissipation": False,
        "vort_dissipation": False,
        "collisional_friction": False,
        "phi_boundary_relax": False,
        "phi_sheath_dissipation": False,
        "damp_core_vorticity": False,
    }
    for key, expected in option_defaults.items():
        value = bool(config.parsed("vorticity", key)) if config.has_option("vorticity", key) else (True if key == "phi_dissipation" else False)
        if value != expected:
            return False
    exb_advection = bool(config.parsed("vorticity", "exb_advection")) if config.has_option("vorticity", "exb_advection") else True
    exb_advection_simplified = (
        bool(config.parsed("vorticity", "exb_advection_simplified"))
        if config.has_option("vorticity", "exb_advection_simplified")
        else True
    )
    if not exb_advection or not exb_advection_simplified:
        return False
    if not np.allclose(np.asarray(metrics.g23), 0.0, rtol=1e-12, atol=1e-12):
        return False
    return True


def _uniform_identity_parallel_metric(mesh: StructuredMesh, *, metrics: StructuredMetrics) -> bool:
    if not np.allclose(np.asarray(metrics.J), 1.0, rtol=1e-12, atol=1e-12):
        return False
    if not np.allclose(np.asarray(metrics.g22), 1.0, rtol=1e-12, atol=1e-12):
        return False
    if not np.allclose(np.asarray(metrics.g23), 0.0, rtol=1e-12, atol=1e-12):
        return False
    dy = np.asarray(metrics.dy[:, mesh.ystart : mesh.yend + 1, :], dtype=np.float64)
    return np.allclose(dy, dy[:, :1, :], rtol=1e-12, atol=1e-12)
