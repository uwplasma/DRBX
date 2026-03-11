from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver, load_bout_input
from ..parity.portable import build_portable_summary_payload
from ..parity.reference import make_default_overrides, merge_overrides
from ..reference.cases import ReferenceCase
from ..runtime.run_config import RunConfiguration
from .expression import ArrayExpressionEvaluator
from .fluid_1d import advance_mms_history, compute_mms_rhs, initialize_mms_state
from .metrics import StructuredMetrics, build_structured_metrics
from .mesh import (
    StructuredMesh,
    apply_field_boundaries,
    broadcast_to_field_shape,
    build_structured_mesh,
)
from .transport import advance_anomalous_diffusion_history
from .units import resolved_dataset_scalars
from .vorticity import advance_vorticity_history, apply_vorticity_boundaries, build_vorticity_operator, compute_vorticity_rhs


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
    time_points, variables = _execute_supported_case(config, run_config, mesh, metrics, parity_mode=parity_mode)
    compare_names = compare_variables or tuple(variables)
    trimmed_variables = _prepare_compare_variables(
        variables,
        mesh,
        trim_y_guards=reference_case.trim_y_guards if reference_case is not None else False,
    )
    dataset_scalars = resolved_dataset_scalars(run_config)
    payload = build_portable_summary_payload(
        case_name=case_name,
        parity_mode=parity_mode,
        compare_variables=compare_names,
        component_labels=tuple(component.label for component in run_config.components),
        dimensions={"t": len(time_points), "x": mesh.nx, "y": mesh.local_ny, "z": mesh.nz},
        time_points=time_points,
        dataset_scalars=dataset_scalars,
        variables={name: np.asarray(value, dtype=np.float64) for name, value in trimmed_variables.items()},
        overrides=_effective_overrides(parity_mode, reference_case=reference_case),
        configured_nout=run_config.time.nout,
        configured_timestep=run_config.time.timestep,
        producer="jax-drb",
    )
    return NativeRunResult(
        payload=payload,
        variables=trimmed_variables,
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
    *,
    parity_mode: str,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    implementations = tuple(component.implementation for component in run_config.components)
    if len(run_config.components) == 1 and implementations == ("evolve_density",):
        component = run_config.components[0]
        variable_name = f"N{component.section}"
        if parity_mode != "one_rhs":
            raise NotImplementedError("Native single-component density execution currently supports one_rhs parity only.")
        field = _initialize_species_field(config, variable_name, mesh)
        return (0.0,), {variable_name: field[None, ...]}

    if _is_supported_diffusion_case(run_config):
        return _execute_diffusion_case(config, run_config, mesh, metrics, parity_mode=parity_mode)

    if _is_supported_periodic_fluid_mms_case(config, run_config, mesh, metrics):
        return _execute_periodic_fluid_mms_case(config, run_config, mesh, metrics, parity_mode=parity_mode)

    if _is_supported_electrostatic_vorticity_case(config, run_config, mesh, metrics):
        return _execute_electrostatic_vorticity_case(config, run_config, mesh, metrics, parity_mode=parity_mode)

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
) -> tuple[tuple[float, ...], dict[str, Any]]:
    section = run_config.components[0].section
    density_name = f"N{section}"
    pressure_name = f"P{section}"
    density = _initialize_species_field(config, density_name, mesh)
    pressure = _initialize_species_field(config, pressure_name, mesh)
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

    steps = _effective_output_steps(parity_mode, configured_nout=run_config.time.nout)
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
    time_points = tuple(run_config.time.timestep * index for index in range(steps + 1))
    return time_points, {
        density_name: np.asarray(history.density_history, dtype=np.float64),
        pressure_name: np.asarray(history.pressure_history, dtype=np.float64),
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
    field = broadcast_to_field_shape(
        evaluator.evaluate(config.raw(variable_name, option_name), current_section=variable_name),
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


def _execute_periodic_fluid_mms_case(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    *,
    parity_mode: str,
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

    steps = _effective_output_steps(parity_mode, configured_nout=run_config.time.nout)
    history = advance_mms_history(
        config,
        section=section,
        mesh=mesh,
        metrics=metrics,
        atomic_mass=atomic_mass,
        timestep=run_config.time.timestep,
        steps=steps,
        substeps=20,
    )
    time_points = tuple(run_config.time.timestep * index for index in range(steps + 1))
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
) -> tuple[tuple[float, ...], dict[str, Any]]:
    initial = apply_vorticity_boundaries(_initialize_species_field(config, "Vort", mesh), mesh)
    average_atomic_mass = float(config.parsed("vorticity", "average_atomic_mass")) if config.has_option("vorticity", "average_atomic_mass") else 2.0
    operator = build_vorticity_operator(mesh=mesh, metrics=metrics, average_atomic_mass=average_atomic_mass)

    if parity_mode == "one_rhs":
        rhs = compute_vorticity_rhs(initial, mesh=mesh, metrics=metrics, operator=operator)
        return (0.0,), {"ddt(Vort)": np.asarray(rhs.vorticity[None, ...], dtype=np.float64)}

    steps = _effective_output_steps(parity_mode, configured_nout=run_config.time.nout)
    history = advance_vorticity_history(
        initial,
        mesh=mesh,
        metrics=metrics,
        operator=operator,
        timestep=run_config.time.timestep,
        steps=steps,
        rtol=1e-6,
        atol=1e-8,
        mxstep=20000,
    )
    time_points = tuple(run_config.time.timestep * index for index in range(steps + 1))
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


def _prepare_compare_variables(
    variables: Mapping[str, Any],
    mesh: StructuredMesh,
    *,
    trim_y_guards: bool,
) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for name, value in variables.items():
        array = np.asarray(value, dtype=np.float64)
        if trim_y_guards and array.ndim >= 4 and array.shape[2] > 2 * mesh.myg:
            array = array[:, :, mesh.myg : -mesh.myg, ...]
        prepared[name] = array
    return prepared


def _effective_overrides(parity_mode: str, *, reference_case: ReferenceCase | None) -> tuple[str, ...]:
    case_overrides = reference_case.extra_overrides if reference_case is not None else ()
    return merge_overrides(make_default_overrides(parity_mode), case_overrides)


def _effective_output_steps(parity_mode: str, *, configured_nout: int) -> int:
    if parity_mode == "one_rhs":
        return 0
    if parity_mode == "one_step":
        return 1
    return configured_nout
