from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver, load_bout_input
from ..parity.portable import build_portable_summary_payload
from ..parity.reference import make_default_overrides
from ..reference.cases import ReferenceCase
from ..runtime.run_config import RunConfiguration
from .expression import ArrayExpressionEvaluator
from .metrics import StructuredMetrics, build_structured_metrics
from .mesh import (
    StructuredMesh,
    apply_field_boundaries,
    broadcast_to_field_shape,
    build_structured_mesh,
)
from .transport import advance_anomalous_diffusion_one_step
from .units import resolved_dataset_scalars


@dataclass(frozen=True)
class NativeRunResult:
    payload: Mapping[str, Any]
    variables: Mapping[str, Any]
    time_points: tuple[float, ...]
    run_config: RunConfiguration
    mesh: StructuredMesh
    metrics: StructuredMetrics


def run_curated_case(
    case_name: str,
    *,
    reference_root: str | Path,
    manifest_path: str | Path | None = None,
) -> NativeRunResult:
    from ..parity.reference import resolve_reference_case

    case, input_path = resolve_reference_case(case_name, reference_root=reference_root, manifest_path=manifest_path)
    return run_input_case(
        input_path,
        case_name=case.name,
        parity_mode=case.parity_mode,
        compare_variables=case.compare_variables,
        reference_case=case,
    )


def run_input_case(
    input_path: str | Path,
    *,
    case_name: str | None = None,
    parity_mode: str = "manual",
    compare_variables: tuple[str, ...] = (),
    reference_case: ReferenceCase | None = None,
) -> NativeRunResult:
    config = load_bout_input(input_path)
    return run_config_case(
        config,
        case_name=case_name or Path(input_path).stem,
        parity_mode=parity_mode,
        compare_variables=compare_variables,
        reference_case=reference_case,
    )


def run_config_case(
    config: BoutConfig,
    *,
    case_name: str,
    parity_mode: str,
    compare_variables: tuple[str, ...] = (),
    reference_case: ReferenceCase | None = None,
) -> NativeRunResult:
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    time_points, variables = _execute_supported_case(config, run_config, mesh, metrics)
    compare_names = compare_variables or tuple(variables)
    dataset_scalars = resolved_dataset_scalars(run_config)
    payload = build_portable_summary_payload(
        case_name=case_name,
        parity_mode=parity_mode,
        compare_variables=compare_names,
        component_labels=tuple(component.label for component in run_config.components),
        dimensions={"t": len(time_points), "x": mesh.nx, "y": mesh.local_ny, "z": mesh.nz},
        time_points=time_points,
        dataset_scalars=dataset_scalars,
        variables={name: np.asarray(value, dtype=np.float64) for name, value in variables.items()},
        overrides=make_default_overrides(parity_mode),
        configured_nout=run_config.time.nout,
        configured_timestep=run_config.time.timestep,
        producer="jax-drb",
    )
    return NativeRunResult(
        payload=payload,
        variables=variables,
        time_points=time_points,
        run_config=run_config,
        mesh=mesh,
        metrics=metrics,
    )


def _execute_supported_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    implementations = tuple(component.implementation for component in run_config.components)
    if len(run_config.components) == 1 and implementations == ("evolve_density",):
        component = run_config.components[0]
        variable_name = f"N{component.section}"
        field = _initialize_species_field(config, variable_name, mesh)
        return (0.0,), {variable_name: field[None, ...]}

    if _is_supported_diffusion_case(run_config):
        return _execute_diffusion_case(config, run_config, mesh, metrics)

    raise NotImplementedError(
        "Native execution is not implemented for the configured component set: "
        + ", ".join(component.label for component in run_config.components)
    )


def _execute_diffusion_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    section = run_config.components[0].section
    density_name = f"N{section}"
    pressure_name = f"P{section}"
    density = _initialize_species_field(config, density_name, mesh)
    pressure = _initialize_species_field(config, pressure_name, mesh)
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

    stepped = advance_anomalous_diffusion_one_step(
        density,
        pressure,
        mesh=mesh,
        metrics=metrics,
        anomalous_D=anomalous_D,
        density_boundary=density_boundary,
        pressure_boundary=pressure_boundary,
        timestep=run_config.time.timestep,
    )
    time_points = (0.0, run_config.time.timestep)
    return time_points, {
        density_name: np.asarray(np.stack((density, stepped.density), axis=0), dtype=np.float64),
        pressure_name: np.asarray(np.stack((pressure, stepped.pressure), axis=0), dtype=np.float64),
    }


def _initialize_species_field(config: BoutConfig, variable_name: str, mesh: StructuredMesh) -> Any:
    if not config.has_section(variable_name) or not config.has_option(variable_name, "function"):
        raise KeyError(f"Missing initial condition function for {variable_name}.")
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    field = broadcast_to_field_shape(
        evaluator.evaluate(config.raw(variable_name, "function"), current_section=variable_name),
        mesh,
    )
    field = apply_field_boundaries(field, mesh, x_boundary=_field_boundary_kind(config, variable_name))
    return field


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
