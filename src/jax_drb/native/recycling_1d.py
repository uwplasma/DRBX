from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from importlib import resources
import math
import re

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver
from ..solver import (
    ImplicitStepInfo,
    backward_euler_residual,
    bdf2_residual,
    build_locality_sparsity,
    build_modulo_color_groups,
    build_sparse_difference_quotient_jacobian,
    pack_active_fields,
    solve_matrix_free_newton_system,
    solve_sparse_newton_system,
    unpack_active_fields,
)
from .expression import ArrayExpressionEvaluator
from .mesh import StructuredMesh, broadcast_to_field_shape
from .metrics import StructuredMetrics
from .neutral_mixed import _div_par_fvv_open, _div_par_k_grad_par_open, _div_par_mod_open, _grad_par_open
from .open_field import (
    apply_noflow_flow_guards,
    apply_noflow_scalar_guards,
    apply_parallel_electric_force,
    compute_electron_force_balance,
    compute_target_recycling_sources,
    limit_free,
)


@dataclass(frozen=True)
class OpenFieldSpecies:
    name: str
    density: np.ndarray
    pressure: np.ndarray
    momentum: np.ndarray
    charge: float
    atomic_mass: float
    density_floor: float
    has_pressure: bool
    has_momentum: bool
    noflow_lower_y: bool
    noflow_upper_y: bool
    target_recycle: bool
    recycle_as: str | None
    target_recycle_multiplier: float
    target_recycle_energy: float
    target_fast_recycle_fraction: float
    target_fast_recycle_energy_factor: float

    @property
    def density_name(self) -> str:
        return f"N{self.name}"

    @property
    def pressure_name(self) -> str:
        return f"P{self.name}"

    @property
    def momentum_name(self) -> str:
        return f"NV{self.name}"


@dataclass(frozen=True)
class Recycling1DRhsResult:
    variables: dict[str, np.ndarray]
    feedback_integral_rhs: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Recycling1DHistoryResult:
    variable_history: dict[str, np.ndarray]
    feedback_integral_history: dict[str, np.ndarray]


@dataclass(frozen=True)
class Recycling1DImplicitStepInfo:
    residual_inf_norm: float
    active_size: int
    nonlinear_iterations: int
    linear_iterations: int


@dataclass(frozen=True)
class _DensityFeedbackController:
    species_name: str
    density_upstream: float
    density_controller_p: float
    density_controller_i: float
    density_integral_positive: bool
    density_source_positive: bool
    density_source_shape: np.ndarray
    diagnose: bool


@dataclass(frozen=True)
class _RecyclingRuntimeModel:
    species_templates: dict[str, OpenFieldSpecies]
    controllers: dict[str, _DensityFeedbackController]
    explicit_pressure_sources: dict[str, np.ndarray]
    field_names: tuple[str, ...]
    feedback_names: tuple[str, ...]


_AMJUEL_FILENAMES = {
    ("d", "iz"): "iz_AMJUEL_H.x_2.1.5.json",
    ("d", "rec"): "rec_AMJUEL_H.x_2.1.8.json",
    ("t", "iz"): "iz_AMJUEL_H.x_2.1.5.json",
    ("t", "rec"): "rec_AMJUEL_H.x_2.1.8.json",
    ("he", "iz"): "iz_AMJUEL_H.x_2.3.9a.json",
    ("he", "rec"): "rec_AMJUEL_H.x_2.3.13a.json",
}


def compute_recycling_1d_rhs(
    config: BoutConfig,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    field_overrides: dict[str, np.ndarray] | None = None,
    feedback_integrals: dict[str, float] | None = None,
    apply_sheath_boundaries: bool = True,
) -> Recycling1DRhsResult:
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=field_overrides,
    )
    fields = _build_recycling_state_fields(runtime_model, field_overrides=field_overrides)
    species = _override_species_fields(runtime_model.species_templates, fields=fields, mesh=mesh)
    return _compute_recycling_1d_rhs_from_species(
        config,
        species=species,
        controllers=runtime_model.controllers,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        feedback_integrals=feedback_integrals,
        explicit_pressure_sources=runtime_model.explicit_pressure_sources,
        apply_sheath_boundaries=apply_sheath_boundaries,
    )


def _compute_recycling_1d_rhs_from_species(
    config: BoutConfig,
    *,
    species: dict[str, OpenFieldSpecies],
    controllers: dict[str, _DensityFeedbackController],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    feedback_integrals: dict[str, float] | None,
    feedback_previous_errors: dict[str, float] | None = None,
    feedback_timestep: float | None = None,
    explicit_pressure_sources: dict[str, np.ndarray] | None = None,
    apply_sheath_boundaries: bool = True,
) -> Recycling1DRhsResult:
    pressure_sources = explicit_pressure_sources or {}
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    neutrals = tuple(sp for sp in species.values() if sp.charge == 0.0)
    electron = species["e"]
    electron_density = _electron_density(ions)

    density_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    energy_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    momentum_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    diagnostics: dict[str, np.ndarray] = {}

    reaction_terms = _reaction_sources(
        config,
        species=species,
        electron_density=electron_density,
        dataset_scalars=dataset_scalars,
    )
    for name, value in reaction_terms.density_source.items():
        density_source[name] = density_source[name] + value
    for name, value in reaction_terms.energy_source.items():
        energy_source[name] = energy_source[name] + value
    for name, value in reaction_terms.momentum_source.items():
        momentum_source[name] = momentum_source[name] + value
    diagnostics.update(reaction_terms.diagnostics)

    prepared, ion_boundary, electron_boundary = _prepare_open_field_states(
        species,
        mesh=mesh,
        metrics=metrics,
        apply_sheath_boundaries=apply_sheath_boundaries,
    )
    ion_velocity = ion_boundary.velocity
    for name, value in ion_boundary.energy_source.items():
        energy_source[name] = energy_source[name] + value

    collision_terms = _apply_collision_closure(
        config,
        species,
        prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
    )
    for name, value in collision_terms.energy_source.items():
        energy_source[name] = energy_source[name] + value
    for name, value in collision_terms.momentum_source.items():
        momentum_source[name] = momentum_source[name] + value
    diagnostics.update(collision_terms.diagnostics)

    neutral_diffusion_terms = _apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
    )
    for name, value in neutral_diffusion_terms.density_source.items():
        density_source[name] = density_source[name] + value
    for name, value in neutral_diffusion_terms.energy_source.items():
        energy_source[name] = energy_source[name] + value
    for name, value in neutral_diffusion_terms.momentum_source.items():
        momentum_source[name] = momentum_source[name] + value
    diagnostics.update(neutral_diffusion_terms.diagnostics)

    energy_source["e"] = energy_source["e"] + electron_boundary.energy_source

    recycling_terms = _target_recycling_sources(
        ions=ions,
        prepared=prepared,
        neutrals=neutrals,
        ion_velocity=ion_velocity,
        mesh=mesh,
        metrics=metrics,
    )
    for name, value in recycling_terms.density_source.items():
        density_source[name] = density_source[name] + value
    for name, value in recycling_terms.energy_source.items():
        energy_source[name] = energy_source[name] + value
    diagnostics.update(recycling_terms.diagnostics)

    feedback_terms = _apply_upstream_density_feedback(
        species,
        prepared,
        controllers=controllers,
        mesh=mesh,
        feedback_integrals=feedback_integrals,
        feedback_previous_errors=feedback_previous_errors,
        feedback_timestep=feedback_timestep,
    )
    for name, value in feedback_terms.density_source.items():
        density_source[name] = density_source[name] + value
    for name, value in feedback_terms.energy_source.items():
        energy_source[name] = energy_source[name] + value
    diagnostics.update(feedback_terms.diagnostics)

    electron_force_density = -_grad_par_electron_force_balance_open(
        electron_boundary.pressure,
        mesh=mesh,
        metrics=metrics,
    )
    electron_force_density = electron_force_density + momentum_source["e"]
    electron_epar = electron_force_density / np.maximum(electron_density, 1.0e-5)
    for ion in ions:
        momentum_source[ion.name] = momentum_source[ion.name] + np.asarray(
            apply_parallel_electric_force(
                ion.density,
                charge=ion.charge,
                epar=electron_epar,
            ),
            dtype=np.float64,
        )
    diagnostics["Epar"] = np.asarray(electron_epar, dtype=np.float64)
    diagnostics["Ve"] = np.asarray(electron_boundary.velocity, dtype=np.float64)

    variables: dict[str, np.ndarray] = {}
    variables[electron.density_name] = prepared["e"].density[None, ...]

    for ion in ions:
        ion_state = prepared[ion.name]
        temperature = ion_state.temperature
        fastest_wave = np.sqrt(np.maximum(temperature, 0.0) / ion.atomic_mass)
        density_rhs = density_source[ion.name] - _div_par_mod_open(
            ion_state.density,
            ion_velocity[ion.name],
            fastest_wave,
            mesh=mesh,
            metrics=metrics,
        )
        pressure_rhs = np.asarray(
            pressure_sources.get(
                ion.name,
                _explicit_pressure_source(config, ion.name, mesh=mesh, dataset_scalars=dataset_scalars),
            ),
            dtype=np.float64,
        )
        pressure_rhs = pressure_rhs - (5.0 / 3.0) * _div_par_mod_open(
            ion_state.pressure,
            ion_velocity[ion.name],
            fastest_wave,
            mesh=mesh,
            metrics=metrics,
        )
        pressure_rhs = pressure_rhs + (2.0 / 3.0) * ion_velocity[ion.name] * _grad_par_open(
            ion_state.pressure,
            mesh=mesh,
            metrics=metrics,
        )
        pressure_rhs = pressure_rhs + (2.0 / 3.0) * energy_source[ion.name]
        momentum_rhs = -ion.atomic_mass * _div_par_fvv_open(
            _soft_floor(ion_state.density, ion.density_floor),
            ion_velocity[ion.name],
            fastest_wave,
            mesh=mesh,
            metrics=metrics,
            fix_flux=False,
        )
        momentum_rhs = momentum_rhs - _grad_par_open(ion_state.pressure, mesh=mesh, metrics=metrics)
        momentum_rhs = momentum_rhs + momentum_source[ion.name]
        momentum_rhs = momentum_rhs + ion_state.momentum_error

        variables[ion.density_name] = ion_state.density[None, ...]
        variables[ion.pressure_name] = ion_state.pressure[None, ...]
        variables[ion.momentum_name] = ion_state.momentum[None, ...]
        variables[f"SNV{ion.name}"] = momentum_source[ion.name][None, ...]
        variables[f"ddt({ion.density_name})"] = density_rhs[None, ...]
        variables[f"ddt({ion.pressure_name})"] = pressure_rhs[None, ...]
        variables[f"ddt({ion.momentum_name})"] = momentum_rhs[None, ...]

    electron_velocity = _electron_zero_current_velocity(
        ions,
        prepared=prepared,
        ion_velocity=ion_velocity,
        electron_density=prepared["e"].density,
    )
    electron_fastest_wave = np.sqrt(np.maximum(prepared["e"].temperature, 0.0) / electron.atomic_mass)
    electron_pressure_rhs = np.asarray(
        pressure_sources.get(
            "e",
            _explicit_pressure_source(config, "e", mesh=mesh, dataset_scalars=dataset_scalars),
        ),
        dtype=np.float64,
    )
    electron_pressure_rhs = electron_pressure_rhs - (5.0 / 3.0) * _div_par_mod_open(
        electron_boundary.pressure,
        electron_velocity,
        electron_fastest_wave,
        mesh=mesh,
        metrics=metrics,
    )
    electron_pressure_rhs = electron_pressure_rhs + (2.0 / 3.0) * electron_velocity * _grad_par_open(
        electron_boundary.pressure,
        mesh=mesh,
        metrics=metrics,
    )
    electron_pressure_rhs = electron_pressure_rhs + (2.0 / 3.0) * energy_source["e"]
    variables[electron.pressure_name] = electron_boundary.pressure[None, ...]
    variables[f"ddt({electron.pressure_name})"] = electron_pressure_rhs[None, ...]

    for neutral in neutrals:
        neutral_state = prepared[neutral.name]
        temperature = neutral_state.temperature
        fastest_wave = np.sqrt(np.maximum(temperature, 0.0) / neutral.atomic_mass)
        density_rhs = density_source[neutral.name] - _div_par_mod_open(
            neutral_state.density,
            neutral_state.velocity,
            fastest_wave,
            mesh=mesh,
            metrics=metrics,
        )
        pressure_rhs = -(5.0 / 3.0) * _div_par_mod_open(
            neutral_state.pressure,
            neutral_state.velocity,
            fastest_wave,
            mesh=mesh,
            metrics=metrics,
        )
        pressure_rhs = pressure_rhs + (2.0 / 3.0) * neutral_state.velocity * _grad_par_open(
            neutral_state.pressure,
            mesh=mesh,
            metrics=metrics,
        )
        pressure_rhs = pressure_rhs + (2.0 / 3.0) * energy_source[neutral.name]
        momentum_rhs = -neutral.atomic_mass * _div_par_fvv_open(
            _soft_floor(neutral_state.density, neutral.density_floor),
            neutral_state.velocity,
            fastest_wave,
            mesh=mesh,
            metrics=metrics,
            fix_flux=False,
        )
        momentum_rhs = momentum_rhs - _grad_par_open(neutral_state.pressure, mesh=mesh, metrics=metrics)
        momentum_rhs = momentum_rhs + momentum_source[neutral.name]
        momentum_rhs = momentum_rhs + neutral_state.momentum_error
        variables[neutral.density_name] = neutral_state.density[None, ...]
        variables[neutral.pressure_name] = neutral_state.pressure[None, ...]
        variables[neutral.momentum_name] = neutral_state.momentum[None, ...]
        variables[f"SNV{neutral.name}"] = momentum_source[neutral.name][None, ...]
        variables[f"ddt({neutral.density_name})"] = density_rhs[None, ...]
        variables[f"ddt({neutral.pressure_name})"] = pressure_rhs[None, ...]
        variables[f"ddt({neutral.momentum_name})"] = momentum_rhs[None, ...]

    for name, value in diagnostics.items():
        variables[name] = value[None, ...]

    return Recycling1DRhsResult(
        variables=variables,
        feedback_integral_rhs=feedback_terms.feedback_integral_rhs,
    )


def _initialize_species(
    config: BoutConfig,
    *,
    mesh: StructuredMesh,
    dataset_scalars: dict[str, float] | None = None,
    field_overrides: dict[str, np.ndarray] | None = None,
) -> dict[str, OpenFieldSpecies]:
    resolver = NumericResolver(config)
    overrides = field_overrides or {}
    scalars = dataset_scalars or {}
    model_species = []
    for section in config.sections:
        if section == "e":
            model_species.append(section)
            continue
        if not config.has_option(section, "type"):
            continue
        type_values = config.parsed(section, "type")
        if isinstance(type_values, tuple) and any(str(item).startswith("evolve_") or str(item) in {"quasineutral", "neutral_mixed"} for item in type_values):
            model_species.append(section)

    species: dict[str, OpenFieldSpecies] = {}
    for name in model_species:
        density_name = f"N{name}"
        pressure_name = f"P{name}"
        momentum_name = f"NV{name}"
        density = np.asarray(overrides[density_name], dtype=np.float64, copy=True) if density_name in overrides else (
            _evaluate_field_option(config, density_name, mesh=mesh) if config.has_section(density_name) else None
        )
        if name == "e":
            if density is None:
                density = None
            pressure = np.asarray(overrides[pressure_name], dtype=np.float64, copy=True) if pressure_name in overrides else _evaluate_field_option(config, pressure_name, mesh=mesh)
            momentum = np.zeros_like(pressure, dtype=np.float64)
        else:
            if density is None:
                raise KeyError(f"Missing density section for {name}.")
            pressure = np.asarray(overrides[pressure_name], dtype=np.float64, copy=True) if pressure_name in overrides else (
                _evaluate_field_option(config, pressure_name, mesh=mesh) if config.has_section(pressure_name) else density.copy()
            )
            momentum = np.asarray(overrides[momentum_name], dtype=np.float64, copy=True) if momentum_name in overrides else (
                _evaluate_field_option(config, momentum_name, mesh=mesh) if config.has_section(momentum_name) else np.zeros_like(density, dtype=np.float64)
            )

        type_values = config.parsed(name, "type")
        components = tuple(str(item) for item in (type_values if isinstance(type_values, tuple) else (type_values,)))
        noflow = "noflow_boundary" in components
        species[name] = OpenFieldSpecies(
            name=name,
            density=np.array(density if density is not None else pressure, dtype=np.float64, copy=True),
            pressure=np.array(pressure, dtype=np.float64, copy=True),
            momentum=np.array(momentum, dtype=np.float64, copy=True),
            charge=float(resolver.resolve(name, "charge")) if config.has_option(name, "charge") else (-1.0 if name == "e" else 0.0),
            atomic_mass=float(resolver.resolve(name, "AA")) if config.has_option(name, "AA") else (1.0 / 1836.0),
            density_floor=float(resolver.resolve(name, "density_floor")) if config.has_option(name, "density_floor") else 1.0e-7,
            has_pressure="evolve_pressure" in components or name == "e" or "neutral_mixed" in components,
            has_momentum="evolve_momentum" in components or "neutral_mixed" in components,
            noflow_lower_y=bool(config.parsed(name, "noflow_lower_y")) if config.has_option(name, "noflow_lower_y") else noflow,
            noflow_upper_y=bool(config.parsed(name, "noflow_upper_y")) if config.has_option(name, "noflow_upper_y") else noflow,
            target_recycle=bool(config.parsed(name, "target_recycle")) if config.has_option(name, "target_recycle") else False,
            recycle_as=str(config.parsed(name, "recycle_as")) if config.has_option(name, "recycle_as") else None,
            target_recycle_multiplier=float(resolver.resolve(name, "target_recycle_multiplier")) if config.has_option(name, "target_recycle_multiplier") else 0.0,
            target_recycle_energy=(
                float(resolver.resolve(name, "target_recycle_energy")) / float(scalars.get("Tnorm", 1.0))
                if config.has_option(name, "target_recycle_energy")
                else 0.0
            ),
            target_fast_recycle_fraction=float(resolver.resolve(name, "target_fast_recycle_fraction")) if config.has_option(name, "target_fast_recycle_fraction") else 0.0,
            target_fast_recycle_energy_factor=float(resolver.resolve(name, "target_fast_recycle_energy_factor")) if config.has_option(name, "target_fast_recycle_energy_factor") else 0.0,
        )

    for name, sp in tuple(species.items()):
        density = sp.density
        pressure = sp.pressure
        momentum = sp.momentum
        if sp.noflow_lower_y and mesh.has_lower_y_target:
            density = np.asarray(apply_noflow_scalar_guards(density, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64)
            pressure = np.asarray(apply_noflow_scalar_guards(pressure, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64)
            momentum = np.asarray(apply_noflow_flow_guards(momentum, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64)
        if sp.noflow_upper_y and mesh.has_upper_y_target:
            density = np.asarray(apply_noflow_scalar_guards(density, mesh=mesh, lower_y=False, upper_y=True), dtype=np.float64)
            pressure = np.asarray(apply_noflow_scalar_guards(pressure, mesh=mesh, lower_y=False, upper_y=True), dtype=np.float64)
            momentum = np.asarray(apply_noflow_flow_guards(momentum, mesh=mesh, lower_y=False, upper_y=True), dtype=np.float64)
        species[name] = OpenFieldSpecies(**{**sp.__dict__, "density": density, "pressure": pressure, "momentum": momentum})
    return species


def _build_recycling_runtime_model(
    config: BoutConfig,
    *,
    mesh: StructuredMesh,
    dataset_scalars: dict[str, float],
    field_overrides: dict[str, np.ndarray] | None = None,
) -> _RecyclingRuntimeModel:
    species_templates = _initialize_species(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=field_overrides,
    )
    controllers = _load_density_feedback_controllers(
        config,
        species=species_templates,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
    )
    field_names = _recycling_evolving_variable_names(species_templates)
    return _RecyclingRuntimeModel(
        species_templates=species_templates,
        controllers=controllers,
        explicit_pressure_sources=_load_explicit_pressure_sources(
            config,
            species_templates=species_templates,
            mesh=mesh,
            dataset_scalars=dataset_scalars,
        ),
        field_names=field_names,
        feedback_names=tuple(sorted(controllers)),
    )


def _load_explicit_pressure_sources(
    config: BoutConfig,
    *,
    species_templates: dict[str, OpenFieldSpecies],
    mesh: StructuredMesh,
    dataset_scalars: dict[str, float],
) -> dict[str, np.ndarray]:
    return {
        name: _explicit_pressure_source(config, name, mesh=mesh, dataset_scalars=dataset_scalars)
        for name in species_templates
        if species_templates[name].has_pressure or name == "e"
    }


def _build_recycling_state_fields(
    runtime_model: _RecyclingRuntimeModel,
    *,
    field_overrides: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    overrides = field_overrides or {}
    fields = _recycling_field_templates(runtime_model.species_templates, field_names=runtime_model.field_names)
    for name, value in overrides.items():
        if name in fields:
            fields[name] = np.asarray(value, dtype=np.float64, copy=True)
    return fields


def _override_species_fields(
    species_templates: dict[str, OpenFieldSpecies],
    *,
    fields: dict[str, np.ndarray],
    mesh: StructuredMesh,
) -> dict[str, OpenFieldSpecies]:
    species: dict[str, OpenFieldSpecies] = {}
    for name, template in species_templates.items():
        density = np.asarray(fields.get(template.density_name, template.density), dtype=np.float64, copy=True)
        pressure = np.asarray(fields.get(template.pressure_name, template.pressure), dtype=np.float64, copy=True)
        momentum = np.asarray(fields.get(template.momentum_name, template.momentum), dtype=np.float64, copy=True)
        if template.noflow_lower_y and mesh.has_lower_y_target:
            density = np.asarray(apply_noflow_scalar_guards(density, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64)
            pressure = np.asarray(apply_noflow_scalar_guards(pressure, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64)
            momentum = np.asarray(apply_noflow_flow_guards(momentum, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64)
        if template.noflow_upper_y and mesh.has_upper_y_target:
            density = np.asarray(apply_noflow_scalar_guards(density, mesh=mesh, lower_y=False, upper_y=True), dtype=np.float64)
            pressure = np.asarray(apply_noflow_scalar_guards(pressure, mesh=mesh, lower_y=False, upper_y=True), dtype=np.float64)
            momentum = np.asarray(apply_noflow_flow_guards(momentum, mesh=mesh, lower_y=False, upper_y=True), dtype=np.float64)
        species[name] = OpenFieldSpecies(
            **{
                **template.__dict__,
                "density": density,
                "pressure": pressure,
                "momentum": momentum,
            }
        )
    return species


def _evaluate_field_option(config: BoutConfig, variable_name: str, *, mesh: StructuredMesh) -> np.ndarray:
    raw_value = config.raw(variable_name, "function") if config.has_option(variable_name, "function") else config.raw(variable_name, "solution")
    resolved_reference = _try_literal_reference(config, raw_value)
    if resolved_reference is not None:
        return _evaluate_field_value(config, resolved_reference[0], mesh=mesh, option_name=resolved_reference[1])
    return _evaluate_field_value(config, variable_name, mesh=mesh, option_name="function" if config.has_option(variable_name, "function") else "solution")


def _evaluate_field_value(
    config: BoutConfig,
    variable_name: str,
    *,
    mesh: StructuredMesh,
    option_name: str,
) -> np.ndarray:
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    field = broadcast_to_field_shape(evaluator.resolve_option(variable_name, option_name), mesh)
    return np.asarray(field, dtype=np.float64)


def _explicit_pressure_source(
    config: BoutConfig,
    species_name: str,
    *,
    mesh: StructuredMesh,
    dataset_scalars: dict[str, float],
) -> np.ndarray:
    section = f"P{species_name}"
    if not config.has_section(section) or not config.has_option(section, "source"):
        return np.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    raw_value = config.raw(section, "source")
    resolved_reference = _try_literal_reference(config, raw_value)
    if resolved_reference is not None:
        field = _evaluate_field_value(config, resolved_reference[0], mesh=mesh, option_name=resolved_reference[1])
    else:
        evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
        field = broadcast_to_field_shape(evaluator.resolve_option(section, "source"), mesh)
    source_normalisation = 1.60218e-19 * dataset_scalars["Nnorm"] * dataset_scalars["Tnorm"] * dataset_scalars["Omega_ci"]
    return np.asarray(field, dtype=np.float64) / source_normalisation


def _try_literal_reference(config: BoutConfig, raw_value: str) -> tuple[str, str] | None:
    value = raw_value.strip()
    if not (value.startswith("`") and value.endswith("`")):
        return None
    reference = value[1:-1]
    if ":" not in reference:
        return None
    section, key = reference.split(":", 1)
    if not config.has_section(section) or not config.has_option(section, key):
        return None
    return section, key


def _load_density_feedback_controllers(
    config: BoutConfig,
    *,
    species: dict[str, OpenFieldSpecies],
    mesh: StructuredMesh,
    dataset_scalars: dict[str, float],
) -> dict[str, _DensityFeedbackController]:
    resolver = NumericResolver(config)
    nnorm = float(dataset_scalars["Nnorm"])
    omega_ci = float(dataset_scalars["Omega_ci"])
    controllers: dict[str, _DensityFeedbackController] = {}
    for name, sp in species.items():
        if name == "e" or sp.charge <= 0.0 or not config.has_option(name, "type"):
            continue
        type_values = config.parsed(name, "type")
        components = tuple(str(item).strip() for item in (type_values if isinstance(type_values, tuple) else (type_values,)))
        if "upstream_density_feedback" not in components:
            continue
        density_section = f"N{name}"
        if config.has_option(density_section, "source_shape"):
            raw_value = config.raw(density_section, "source_shape")
            resolved_reference = _try_literal_reference(config, raw_value)
            if resolved_reference is not None:
                source_shape = _evaluate_field_value(
                    config,
                    resolved_reference[0],
                    mesh=mesh,
                    option_name=resolved_reference[1],
                )
            else:
                evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
                source_shape = broadcast_to_field_shape(
                    evaluator.resolve_option(density_section, "source_shape"),
                    mesh,
                )
            source_shape = np.asarray(source_shape, dtype=np.float64) / (nnorm * omega_ci)
        else:
            source_shape = np.zeros_like(sp.density, dtype=np.float64)
        controllers[name] = _DensityFeedbackController(
            species_name=name,
            density_upstream=float(resolver.resolve(name, "density_upstream")) / nnorm,
            density_controller_p=float(resolver.resolve(name, "density_controller_p")) if config.has_option(name, "density_controller_p") else 1.0e-2,
            density_controller_i=float(resolver.resolve(name, "density_controller_i")) if config.has_option(name, "density_controller_i") else 1.0e-3,
            density_integral_positive=bool(config.parsed(name, "density_integral_positive")) if config.has_option(name, "density_integral_positive") else False,
            density_source_positive=bool(config.parsed(name, "density_source_positive")) if config.has_option(name, "density_source_positive") else True,
            density_source_shape=source_shape,
            diagnose=bool(config.parsed(name, "diagnose")) if config.has_option(name, "diagnose") else False,
        )
    return controllers


def _electron_density(ions: tuple[OpenFieldSpecies, ...]) -> np.ndarray:
    density = np.zeros_like(ions[0].density, dtype=np.float64)
    for ion in ions:
        density = density + ion.charge * ion.density
    return density


def _raw_species_velocity(species: OpenFieldSpecies) -> np.ndarray:
    return np.asarray(
        species.momentum / np.maximum(species.atomic_mass * species.density, 1.0e-8),
        dtype=np.float64,
    )


def _safe_temperature(pressure: np.ndarray, density: np.ndarray, density_floor: float = 1.0e-8) -> np.ndarray:
    pressure_floor = np.maximum(np.asarray(pressure, dtype=np.float64), 0.0)
    return pressure_floor / _soft_floor(np.asarray(density, dtype=np.float64), density_floor)


def _soft_floor(value: np.ndarray, minimum: float) -> np.ndarray:
    value_array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    minimum_value = float(minimum)
    return value_array + minimum_value * np.exp(-value_array / minimum_value)


@dataclass(frozen=True)
class _ReactionTerms:
    density_source: dict[str, np.ndarray]
    energy_source: dict[str, np.ndarray]
    momentum_source: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]


@dataclass(frozen=True)
class _PreparedSpeciesState:
    density: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    velocity: np.ndarray
    momentum: np.ndarray
    momentum_error: np.ndarray


@dataclass(frozen=True)
class _CollisionClosureTerms:
    energy_source: dict[str, np.ndarray]
    momentum_source: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]


@dataclass(frozen=True)
class _NeutralParallelDiffusionTerms:
    density_source: dict[str, np.ndarray]
    energy_source: dict[str, np.ndarray]
    momentum_source: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]


@dataclass(frozen=True)
class _DensityFeedbackTerms:
    density_source: dict[str, np.ndarray]
    energy_source: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]
    feedback_integral_rhs: dict[str, float]


_QE = 1.602176634e-19
_EPS0 = 8.8541878128e-12
_MP = 1.67262192369e-27
_ME = 9.1093837015e-31


def _reaction_sources(
    config: BoutConfig,
    *,
    species: dict[str, OpenFieldSpecies],
    electron_density: np.ndarray,
    dataset_scalars: dict[str, float],
) -> _ReactionTerms:
    density_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    energy_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    momentum_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    diagnostics: dict[str, np.ndarray] = {}

    if not config.has_section("reactions") or not config.has_option("reactions", "type"):
        return _ReactionTerms(density_source=density_source, energy_source=energy_source, momentum_source=momentum_source, diagnostics=diagnostics)

    reactions = config.parsed("reactions", "type")
    for reaction in (reactions if isinstance(reactions, tuple) else (reactions,)):
        tokens = tuple(part.strip() for part in str(reaction).split("->"))
        if len(tokens) != 2:
            continue
        lhs = tuple(part.strip() for part in re.split(r"\s+\+\s+", tokens[0].strip()))
        rhs = tuple(part.strip() for part in re.split(r"\s+\+\s+", tokens[1].strip()))

        if len(lhs) == 2 and lhs[1] == "e" and len(rhs) == 2 and rhs[1] == "2e":
            atom = lhs[0]
            ion = rhs[0]
            result = _amjuel_ionisation(atom, ion, species=species, electron_density=electron_density, dataset_scalars=dataset_scalars)
            _accumulate_terms(result, density_source, energy_source, momentum_source, diagnostics)
        elif len(lhs) == 2 and lhs[1] == "e" and len(rhs) == 1:
            ion = lhs[0]
            atom = rhs[0]
            result = _amjuel_recombination(atom, ion, species=species, electron_density=electron_density, dataset_scalars=dataset_scalars)
            _accumulate_terms(result, density_source, energy_source, momentum_source, diagnostics)
        elif _is_charge_exchange_reaction(lhs, rhs):
            atom1 = lhs[0]
            ion1 = lhs[1]
            ion2 = rhs[0]
            atom2 = rhs[1]
            result = _charge_exchange(atom1, ion1, atom2, ion2, species=species, dataset_scalars=dataset_scalars)
            _accumulate_terms(result, density_source, energy_source, momentum_source, diagnostics)
    return _ReactionTerms(density_source=density_source, energy_source=energy_source, momentum_source=momentum_source, diagnostics=diagnostics)


def _is_charge_exchange_reaction(lhs: tuple[str, ...], rhs: tuple[str, ...]) -> bool:
    if len(lhs) != 2 or len(rhs) != 2:
        return False
    atom1, ion1 = lhs
    ion2, atom2 = rhs
    if not ion1.endswith("+") or not ion2.endswith("+"):
        return False
    return ion2[:-1] == atom1 and ion1[:-1] == atom2


def _accumulate_terms(
    result: _ReactionTerms,
    density_source: dict[str, np.ndarray],
    energy_source: dict[str, np.ndarray],
    momentum_source: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
) -> None:
    for name, value in result.density_source.items():
        density_source[name] = density_source[name] + value
    for name, value in result.energy_source.items():
        energy_source[name] = energy_source[name] + value
    for name, value in result.momentum_source.items():
        momentum_source[name] = momentum_source[name] + value
    diagnostics.update(result.diagnostics)


def _prepare_species_state(
    species: OpenFieldSpecies,
    *,
    mesh: StructuredMesh,
) -> _PreparedSpeciesState:
    density = species.density.copy()
    pressure = species.pressure.copy()
    temperature = _safe_temperature(pressure, density, species.density_floor)
    limited_density = _soft_floor(density, species.density_floor)
    velocity = species.momentum / np.maximum(species.atomic_mass * limited_density, 1.0e-8)
    momentum = species.atomic_mass * density * velocity

    if species.noflow_lower_y or species.noflow_upper_y:
        density = np.array(
            apply_noflow_scalar_guards(
                density,
                mesh=mesh,
                lower_y=species.noflow_lower_y,
                upper_y=species.noflow_upper_y,
            ),
            dtype=np.float64,
            copy=True,
        )
        pressure = np.array(
            apply_noflow_scalar_guards(
                pressure,
                mesh=mesh,
                lower_y=species.noflow_lower_y,
                upper_y=species.noflow_upper_y,
            ),
            dtype=np.float64,
            copy=True,
        )
        temperature = np.array(
            apply_noflow_scalar_guards(
                temperature,
                mesh=mesh,
                lower_y=species.noflow_lower_y,
                upper_y=species.noflow_upper_y,
            ),
            dtype=np.float64,
            copy=True,
        )
        velocity = np.array(
            apply_noflow_flow_guards(
                velocity,
                mesh=mesh,
                lower_y=species.noflow_lower_y,
                upper_y=species.noflow_upper_y,
            ),
            dtype=np.float64,
            copy=True,
        )
    momentum = np.asarray(species.atomic_mass * density * velocity, dtype=np.float64)
    momentum_error = np.asarray(momentum - species.momentum, dtype=np.float64)
    return _PreparedSpeciesState(
        density=density,
        pressure=pressure,
        temperature=temperature,
        velocity=velocity,
        momentum=momentum,
        momentum_error=momentum_error,
    )


def _prepare_open_field_states(
    species: dict[str, OpenFieldSpecies],
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    apply_sheath_boundaries: bool = True,
) -> tuple[dict[str, _PreparedSpeciesState], _IonBoundaryResult, _ElectronBoundaryResult]:
    prepared = {name: _prepare_species_state(sp, mesh=mesh) for name, sp in species.items()}
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    electron_density = np.zeros_like(species["e"].density, dtype=np.float64)
    for ion in ions:
        electron_density = electron_density + ion.charge * prepared[ion.name].density

    electron_velocity = _electron_zero_current_velocity(
        ions,
        prepared=prepared,
        ion_velocity={ion.name: prepared[ion.name].velocity for ion in ions},
        electron_density=electron_density,
    )

    if apply_sheath_boundaries:
        electron_boundary = _apply_electron_sheath_boundary(
            electron_pressure=prepared["e"].pressure,
            electron_density=electron_density,
            electron_velocity=electron_velocity,
            electron_mass=species["e"].atomic_mass,
            electron_density_floor=species["e"].density_floor,
            ion_velocity={ion.name: prepared[ion.name].velocity for ion in ions},
            ions=ions,
            prepared_ions=prepared,
            mesh=mesh,
            metrics=metrics,
        )
    else:
        electron_boundary = _ElectronBoundaryResult(
            density=np.asarray(prepared["e"].density, dtype=np.float64),
            temperature=np.asarray(prepared["e"].temperature, dtype=np.float64),
            pressure=np.asarray(prepared["e"].pressure, dtype=np.float64),
            velocity=np.asarray(electron_velocity, dtype=np.float64),
            momentum=np.asarray(species["e"].atomic_mass * prepared["e"].density * electron_velocity, dtype=np.float64),
            energy_source=np.zeros_like(prepared["e"].density, dtype=np.float64),
        )
    prepared["e"] = _PreparedSpeciesState(
        density=electron_boundary.density,
        pressure=electron_boundary.pressure,
        temperature=electron_boundary.temperature,
        velocity=np.asarray(
            electron_boundary.momentum
            / np.maximum(
                species["e"].atomic_mass
                * _soft_floor(electron_boundary.density, species["e"].density_floor),
                1.0e-8,
            ),
            dtype=np.float64,
        ),
        momentum=np.asarray(electron_boundary.momentum, dtype=np.float64),
        momentum_error=np.asarray(
            species["e"].atomic_mass
            * electron_boundary.density
            * (
                electron_boundary.momentum
                / np.maximum(
                    species["e"].atomic_mass
                    * _soft_floor(electron_boundary.density, species["e"].density_floor),
                    1.0e-8,
                )
            )
            - electron_boundary.momentum,
            dtype=np.float64,
        ),
    )

    if apply_sheath_boundaries:
        ion_boundary = _apply_ion_sheath_boundary(
            ions,
            electron_pressure=prepared["e"].pressure,
            electron_density=prepared["e"].density,
            electron_density_floor=species["e"].density_floor,
            mesh=mesh,
            metrics=metrics,
        )
    else:
        ion_boundary = _IonBoundaryResult(
            density={ion.name: np.asarray(prepared[ion.name].density, dtype=np.float64) for ion in ions},
            pressure={ion.name: np.asarray(prepared[ion.name].pressure, dtype=np.float64) for ion in ions},
            temperature={ion.name: np.asarray(prepared[ion.name].temperature, dtype=np.float64) for ion in ions},
            velocity={ion.name: np.asarray(prepared[ion.name].velocity, dtype=np.float64) for ion in ions},
            momentum={ion.name: np.asarray(prepared[ion.name].momentum, dtype=np.float64) for ion in ions},
            energy_source={ion.name: np.zeros_like(prepared[ion.name].density, dtype=np.float64) for ion in ions},
        )
    for ion in ions:
        prepared[ion.name] = _PreparedSpeciesState(
            density=ion_boundary.density[ion.name],
            pressure=ion_boundary.pressure[ion.name],
            temperature=ion_boundary.temperature[ion.name],
            velocity=np.asarray(
                ion_boundary.momentum[ion.name]
                / np.maximum(
                    ion.atomic_mass * _soft_floor(ion_boundary.density[ion.name], ion.density_floor),
                    1.0e-8,
                ),
                dtype=np.float64,
            ),
            momentum=np.asarray(ion_boundary.momentum[ion.name], dtype=np.float64),
            momentum_error=np.asarray(
                ion.atomic_mass
                * ion_boundary.density[ion.name]
                * (
                    ion_boundary.momentum[ion.name]
                    / np.maximum(
                        ion.atomic_mass * _soft_floor(ion_boundary.density[ion.name], ion.density_floor),
                        1.0e-8,
                    )
                )
                - ion_boundary.momentum[ion.name],
                dtype=np.float64,
            ),
        )
    return prepared, ion_boundary, electron_boundary


def _apply_upstream_density_feedback(
    species: dict[str, OpenFieldSpecies],
    prepared: dict[str, _PreparedSpeciesState],
    *,
    controllers: dict[str, _DensityFeedbackController],
    mesh: StructuredMesh,
    feedback_integrals: dict[str, float] | None,
    feedback_previous_errors: dict[str, float] | None = None,
    feedback_timestep: float | None = None,
) -> _DensityFeedbackTerms:
    density_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    energy_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    diagnostics: dict[str, np.ndarray] = {}
    integral_rhs: dict[str, float] = {}
    integrals = feedback_integrals or {}

    for name, controller in controllers.items():
        upstream_density = float(prepared[name].density[mesh.xstart, mesh.ystart, 0])
        error = controller.density_upstream - upstream_density
        stored_integral = float(integrals.get(name, 0.0))
        integrated_error = stored_integral
        if feedback_timestep is not None:
            previous_error = error if feedback_previous_errors is None else float(feedback_previous_errors.get(name, error))
            integrated_error = stored_integral + float(feedback_timestep) * 0.5 * (error + previous_error)
        if controller.density_integral_positive and integrated_error < 0.0:
            integrated_error = 0.0
        proportional_term = controller.density_controller_p * error
        integral_term = controller.density_controller_i * integrated_error
        source_multiplier = proportional_term + integral_term
        if controller.density_source_positive and source_multiplier < 0.0:
            source_multiplier = 0.0
        source = source_multiplier * controller.density_source_shape
        density_source[name] = density_source[name] + source
        velocity = np.asarray(prepared[name].velocity, dtype=np.float64)
        energy_source[name] = energy_source[name] + 0.5 * species[name].atomic_mass * np.square(velocity) * source
        diagnostics[f"S{name}_feedback"] = np.asarray(source, dtype=np.float64)
        diagnostics[f"density_feedback_src_mult_{name}"] = np.asarray(source_multiplier, dtype=np.float64)
        diagnostics[f"density_feedback_src_p_{name}"] = np.asarray(proportional_term, dtype=np.float64)
        diagnostics[f"density_feedback_src_i_{name}"] = np.asarray(integral_term, dtype=np.float64)
        diagnostics[f"density_feedback_src_shape_{name}"] = np.asarray(controller.density_source_shape, dtype=np.float64)
        integral_rhs[name] = error

    return _DensityFeedbackTerms(
        density_source=density_source,
        energy_source=energy_source,
        diagnostics=diagnostics,
        feedback_integral_rhs=integral_rhs,
    )


def _configured_component_names(config: BoutConfig) -> tuple[str, ...]:
    for section in ("model", "hermes"):
        if config.has_section(section) and config.has_option(section, "components"):
            values = config.parsed(section, "components")
            if isinstance(values, tuple):
                return tuple(str(value).strip() for value in values)
            return (str(values).strip(),)
    return ()


def _compute_collision_frequencies(
    config: BoutConfig,
    species: dict[str, OpenFieldSpecies],
    prepared: dict[str, _PreparedSpeciesState],
    *,
    dataset_scalars: dict[str, float],
) -> dict[tuple[str, str], np.ndarray]:
    collision_rates: dict[tuple[str, str], np.ndarray] = {}
    nnorm = float(dataset_scalars["Nnorm"])
    tnorm = float(dataset_scalars["Tnorm"])
    rho_s0 = float(dataset_scalars["rho_s0"])
    omega_ci = float(dataset_scalars["Omega_ci"])
    electron_ion = bool(config.parsed("braginskii_collisions", "electron_ion")) if config.has_option("braginskii_collisions", "electron_ion") else True
    electron_electron = bool(config.parsed("braginskii_collisions", "electron_electron")) if config.has_option("braginskii_collisions", "electron_electron") else True
    electron_neutral = bool(config.parsed("braginskii_collisions", "electron_neutral")) if config.has_option("braginskii_collisions", "electron_neutral") else False
    ion_ion = bool(config.parsed("braginskii_collisions", "ion_ion")) if config.has_option("braginskii_collisions", "ion_ion") else True
    ion_neutral = bool(config.parsed("braginskii_collisions", "ion_neutral")) if config.has_option("braginskii_collisions", "ion_neutral") else False
    neutral_neutral = bool(config.parsed("braginskii_collisions", "neutral_neutral")) if config.has_option("braginskii_collisions", "neutral_neutral") else True

    electron = species["e"]
    electron_state = prepared["e"]
    te_ev = electron_state.temperature * tnorm
    ne_m3 = electron_state.density * nnorm

    if electron_electron:
        te_limited = np.maximum(te_ev, 0.1)
        ne_limited = np.maximum(ne_m3, 1.0e10)
        log_te = np.log(te_limited)
        coulomb_log = (
            30.4
            - 0.5 * np.log(ne_limited)
            + 1.25 * log_te
            - np.sqrt(1.0e-5 + np.square(log_te - 2.0) / 16.0)
        )
        v1sq = 2.0 * te_limited * _QE / _ME
        nu_ee = (
            (_QE**4)
            * np.maximum(ne_m3, 0.0)
            * np.maximum(coulomb_log, 1.0)
            * 2.0
            / (3.0 * np.power(math.pi * 2.0 * v1sq, 1.5) * ((_EPS0 * _ME) ** 2))
        )
        collision_rates[("e", "e")] = np.asarray(nu_ee / omega_ci, dtype=np.float64)

    for species_name, sp in species.items():
        if not electron_ion or species_name == "e" or sp.charge <= 0.0:
            continue
        state = prepared[species_name]
        ti_ev = state.temperature * tnorm
        ni_m3 = state.density * nnorm
        zi = sp.charge
        ai = sp.atomic_mass
        me_mi = _ME / (_MP * ai)

        te_limited = np.maximum(te_ev, 0.1)
        ti_limited = np.maximum(ti_ev, 0.1)
        ne_limited = np.maximum(ne_m3, 1.0e10)
        ni_limited = np.maximum(ni_m3, 1.0e10)
        mask_very_low = (te_ev < 0.1) | (ni_m3 < 1.0e10) | (ne_m3 < 1.0e10)
        mask_low_te = te_ev < (ti_ev * me_mi)
        mask_mid_te = te_ev < (math.exp(2.0) * zi * zi)
        coulomb_log = np.where(
            mask_very_low,
            10.0,
            np.where(
                mask_low_te,
                23.0 - 0.5 * np.log(ni_limited) + 1.5 * np.log(ti_limited) - np.log((zi * zi) * ai),
                np.where(
                    mask_mid_te,
                    30.0 - 0.5 * np.log(ne_limited) - np.log(zi) + 1.5 * np.log(te_limited),
                    31.0 - 0.5 * np.log(ne_limited) + np.log(te_limited),
                ),
            ),
        )
        vesq = 2.0 * te_limited * _QE / _ME
        visq = 2.0 * ti_limited * _QE / (_MP * ai)
        nu_ei = (
            (((_QE * _QE) * zi) ** 2)
            * np.maximum(ni_m3, 0.0)
            * np.maximum(coulomb_log, 1.0)
            * (1.0 + me_mi)
            / (3.0 * np.power(math.pi * (vesq + visq), 1.5) * ((_EPS0 * _ME) ** 2))
        )
        nu_ei = np.asarray(nu_ei / omega_ci, dtype=np.float64)
        collision_rates[("e", species_name)] = nu_ei
        collision_rates[(species_name, "e")] = (
            nu_ei
            * (electron.atomic_mass / sp.atomic_mass)
            * prepared["e"].density
            / np.maximum(state.density, 1.0e-5)
        )

    for neutral_name, neutral_species in species.items():
        if not electron_neutral or neutral_name == "e" or neutral_species.charge != 0.0:
            continue
        neutral_state = prepared[neutral_name]
        vth_e = np.sqrt((_MP / _ME) * np.maximum(prepared["e"].temperature, 0.0))
        nu_en = vth_e * nnorm * neutral_state.density * 5.0e-19 * rho_s0
        nu_en = np.asarray(nu_en, dtype=np.float64)
        collision_rates[("e", neutral_name)] = nu_en
        collision_rates[(neutral_name, "e")] = (
            nu_en
            * (electron.atomic_mass / neutral_species.atomic_mass)
            * prepared["e"].density
            / np.maximum(neutral_state.density, 1.0e-5)
        )

    names = tuple(sorted(name for name in species if name != "e"))

    def collide(name1: str, name2: str, nu_12: np.ndarray) -> None:
        first_species = species[name1]
        second_species = species[name2]
        first_state = prepared[name1]
        second_state = prepared[name2]
        nu_12 = np.asarray(nu_12, dtype=np.float64)
        collision_rates[(name1, name2)] = nu_12
        if name1 == name2:
            return
        collision_rates[(name2, name1)] = (
            nu_12
            * (first_species.atomic_mass / second_species.atomic_mass)
            * first_state.density
            / np.maximum(second_state.density, 1.0e-5)
        )

    for index, first_name in enumerate(names):
        first_species = species[first_name]
        first_state = prepared[first_name]
        t1_ev = first_state.temperature * tnorm
        n1_m3 = first_state.density * nnorm
        first_charged = first_species.charge != 0.0

        for second_name in names[index:]:
            second_species = species[second_name]
            second_state = prepared[second_name]
            t2_ev = second_state.temperature * tnorm
            n2_m3 = second_state.density * nnorm
            second_charged = second_species.charge != 0.0

            if first_charged:
                if second_charged:
                    if not ion_ion:
                        continue
                    z1 = first_species.charge
                    z2 = second_species.charge
                    a1 = first_species.atomic_mass
                    a2 = second_species.atomic_mass
                    m1 = a1 * _MP
                    m2 = a2 * _MP
                    t1_limited = np.maximum(t1_ev, 0.1)
                    t2_limited = np.maximum(t2_ev, 0.1)
                    n1_limited = np.maximum(n1_m3, 1.0e10)
                    n2_limited = np.maximum(n2_m3, 1.0e10)
                    coulomb_log = 29.91 - np.log(
                        ((z1 * z2 * (a1 + a2)) / (a1 * t2_limited + a2 * t1_limited))
                        * np.sqrt(n1_limited * (z1 * z1) / t1_limited + n2_limited * (z2 * z2) / t2_limited)
                    )
                    v1sq = 2.0 * t1_limited * _QE / m1
                    v2sq = 2.0 * t2_limited * _QE / m2
                    nu_12 = (
                        (((z1 * _QE) * (z2 * _QE)) ** 2)
                        * n2_limited
                        * np.maximum(coulomb_log, 1.0)
                        * (1.0 + (m1 / m2))
                        / (3.0 * np.power(math.pi * (v1sq + v2sq), 1.5) * ((_EPS0 * m1) ** 2))
                    )
                    collide(first_name, second_name, nu_12 / omega_ci)
                else:
                    if not ion_neutral:
                        continue
                    vrel = np.sqrt(
                        np.maximum(
                            first_state.temperature / first_species.atomic_mass
                            + second_state.temperature / second_species.atomic_mass,
                            0.0,
                        )
                    )
                    collide(first_name, second_name, vrel * second_state.density * nnorm * 5.0e-19 * rho_s0)
            else:
                if second_charged:
                    if not ion_neutral:
                        continue
                    vrel = np.sqrt(
                        np.maximum(
                            first_state.temperature / first_species.atomic_mass
                            + second_state.temperature / second_species.atomic_mass,
                            0.0,
                        )
                    )
                    collide(first_name, second_name, vrel * second_state.density * nnorm * 5.0e-19 * rho_s0)
                else:
                    if not neutral_neutral:
                        continue
                    vrel = np.sqrt(
                        np.maximum(
                            first_state.temperature / first_species.atomic_mass
                            + second_state.temperature / second_species.atomic_mass,
                            0.0,
                        )
                    )
                    collide(first_name, second_name, vrel * second_state.density * nnorm * (math.pi * (2.8e-10**2)) * rho_s0)
    return collision_rates


def _momentum_coefficient(name1: str, charge1: float, name2: str, charge2: float) -> float:
    def coefficient(charge: float) -> float:
        if charge == 1.0:
            return 0.51
        if charge == 2.0:
            return 0.44
        if charge == 3.0:
            return 0.40
        return 0.38

    if name1 == "e":
        return coefficient(charge2)
    if name2 == "e":
        return coefficient(charge1)
    return 1.0


def _thermal_force_enabled(config: BoutConfig, option_name: str, default: bool) -> bool:
    if not config.has_section("braginskii_thermal_force") or not config.has_option("braginskii_thermal_force", option_name):
        return default
    return bool(config.parsed("braginskii_thermal_force", option_name))


def _ion_thermal_force_pair(
    species1_name: str,
    species2_name: str,
    *,
    species: dict[str, OpenFieldSpecies],
    prepared: dict[str, _PreparedSpeciesState],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    override_mass_restrictions: bool,
) -> tuple[str, str, np.ndarray] | None:
    species1 = species[species1_name]
    species2 = species[species2_name]
    if species1_name == "e" or species2_name == "e":
        return None
    if species1.charge == 0.0 or species2.charge == 0.0:
        return None

    if species1.atomic_mass < 4.0 and species2.atomic_mass > 10.0:
        light_name, heavy_name = species1_name, species2_name
    elif species1.atomic_mass > 10.0 and species2.atomic_mass < 4.0:
        light_name, heavy_name = species2_name, species1_name
    elif override_mass_restrictions:
        if species1.atomic_mass < species2.atomic_mass:
            light_name, heavy_name = species1_name, species2_name
        else:
            light_name, heavy_name = species2_name, species1_name
    else:
        return None

    light = species[light_name]
    heavy = species[heavy_name]
    if heavy.charge == 0.0:
        return None

    mu = heavy.atomic_mass / (light.atomic_mass + heavy.atomic_mass)
    beta = (
        3.0
        * (
            mu
            + 5.0
            * np.sqrt(2.0)
            * (heavy.charge**2)
            * (1.1 * (mu ** 2.5) - 0.35 * (mu ** 1.5))
            - 1.0
        )
        / (2.6 - 2.0 * mu + 5.4 * (mu**2))
    )
    heavy_force = prepared[heavy_name].density * beta * _grad_par_open(
        prepared[light_name].temperature,
        mesh=mesh,
        metrics=metrics,
    )
    return light_name, heavy_name, np.asarray(heavy_force, dtype=np.float64)


def _apply_collision_closure(
    config: BoutConfig,
    species: dict[str, OpenFieldSpecies],
    prepared: dict[str, _PreparedSpeciesState],
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
) -> _CollisionClosureTerms:
    configured_components = set(_configured_component_names(config))
    energy_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    momentum_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    diagnostics: dict[str, np.ndarray] = {}
    collision_rates = _compute_collision_frequencies(config, species, prepared, dataset_scalars=dataset_scalars)
    cx_rates = _charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=dataset_scalars,
    )

    names = tuple(species)
    for index, first_name in enumerate(names):
        first_species = species[first_name]
        first_state = prepared[first_name]
        for second_name in names[index + 1 :]:
            rate_key = (first_name, second_name)
            if rate_key not in collision_rates:
                continue
            second_species = species[second_name]
            second_state = prepared[second_name]
            nu_12 = collision_rates[rate_key]
            a1 = first_species.atomic_mass
            a2 = second_species.atomic_mass

            if "braginskii_friction" in configured_components:
                coeff = _momentum_coefficient(first_name, first_species.charge, second_name, second_species.charge)
                friction = (
                    coeff
                    * a1
                    * nu_12
                    * first_state.density
                    * (second_state.velocity - first_state.velocity)
                )
                momentum_source[first_name] = momentum_source[first_name] + friction
                momentum_source[second_name] = momentum_source[second_name] - friction
                diagnostics[f"F{first_name}{second_name}_coll"] = np.asarray(friction, dtype=np.float64)
                diagnostics[f"F{second_name}{first_name}_coll"] = np.asarray(-friction, dtype=np.float64)

                if first_species.has_pressure or second_species.has_pressure:
                    velocity_delta = second_state.velocity - first_state.velocity
                    first_heating = (a2 / (a1 + a2)) * velocity_delta * friction
                    second_heating = (a1 / (a1 + a2)) * velocity_delta * friction
                    energy_source[first_name] = energy_source[first_name] + first_heating
                    energy_source[second_name] = energy_source[second_name] + second_heating
                    diagnostics[f"E{first_name}{second_name}_coll_friction"] = np.asarray(first_heating, dtype=np.float64)
                    diagnostics[f"E{second_name}{first_name}_coll_friction"] = np.asarray(second_heating, dtype=np.float64)

            if "braginskii_heat_exchange" in configured_components and (first_species.has_pressure or second_species.has_pressure):
                heat_exchange = 3.0 * (a1 / (a1 + a2)) * nu_12 * first_state.density * (
                    second_state.temperature - first_state.temperature
                )
                energy_source[first_name] = energy_source[first_name] + heat_exchange
                energy_source[second_name] = energy_source[second_name] - heat_exchange

    if "braginskii_thermal_force" in configured_components and _thermal_force_enabled(config, "electron_ion", True):
        electron_temperature_gradient = _grad_par_open(prepared["e"].temperature, mesh=mesh, metrics=metrics)
        for name, sp in species.items():
            if name == "e" or sp.charge <= 0.0:
                continue
            ion_force = prepared[name].density * (0.71 * (sp.charge**2)) * electron_temperature_gradient
            momentum_source[name] = momentum_source[name] + ion_force
            momentum_source["e"] = momentum_source["e"] - ion_force

    if "braginskii_thermal_force" in configured_components and _thermal_force_enabled(config, "ion_ion", True):
        ion_names = tuple(name for name, sp in species.items() if name != "e" and sp.charge != 0.0)
        override_mass_restrictions = _thermal_force_enabled(
            config,
            "override_ion_mass_restrictions",
            False,
        )
        for index, first_name in enumerate(ion_names):
            for second_name in ion_names[index + 1 :]:
                pair = _ion_thermal_force_pair(
                    first_name,
                    second_name,
                    species=species,
                    prepared=prepared,
                    mesh=mesh,
                    metrics=metrics,
                    override_mass_restrictions=override_mass_restrictions,
                )
                if pair is None:
                    continue
                light_name, heavy_name, heavy_force = pair
                momentum_source[heavy_name] = momentum_source[heavy_name] + heavy_force
                momentum_source[light_name] = momentum_source[light_name] - heavy_force

    if "braginskii_ion_viscosity" in configured_components:
        for name, sp in species.items():
            if name == "e" or sp.charge == 0.0:
                continue
            total_collisionality = np.zeros_like(prepared[name].density, dtype=np.float64)
            for other_name in species:
                rate = collision_rates.get((name, other_name))
                if rate is not None:
                    total_collisionality = total_collisionality + rate
            if name in cx_rates:
                total_collisionality = total_collisionality + cx_rates[name]
            total_collisionality = np.maximum(total_collisionality, 1.0e-12)
            tau = 1.0 / total_collisionality
            eta = 1.28 * prepared[name].pressure * tau
            viscosity_source = _div_par_parallel_ion_viscosity_open(
                eta,
                prepared[name].velocity,
                mesh=mesh,
                metrics=metrics,
            )
            momentum_source[name] = momentum_source[name] + viscosity_source
            energy_source[name] = energy_source[name] - prepared[name].velocity * viscosity_source
            diagnostics[f"DivPiPar_{name}"] = np.asarray(viscosity_source, dtype=np.float64)

    if "braginskii_conduction" in configured_components:
        for name, sp in species.items():
            if not sp.has_pressure:
                continue
            if config.has_option(name, "thermal_conduction") and not bool(config.parsed(name, "thermal_conduction")):
                continue
            tau = _conduction_collision_time(
                config,
                species=species,
                prepared=prepared,
                collision_rates=collision_rates,
                cx_rates=cx_rates,
                species_name=name,
            )
            kappa_coefficient = _conduction_kappa_coefficient(config, sp)
            temperature = np.asarray(prepared[name].temperature, dtype=np.float64)
            pressure = np.maximum(np.asarray(prepared[name].pressure, dtype=np.float64), 0.0)
            kappa_par = kappa_coefficient * pressure * tau / sp.atomic_mass
            if mesh.myg > 0:
                kappa_par = np.asarray(
                    apply_noflow_scalar_guards(kappa_par, mesh=mesh, lower_y=True, upper_y=True),
                    dtype=np.float64,
                )
            energy_source[name] = energy_source[name] + _div_par_k_grad_par_open(
                kappa_par,
                temperature,
                mesh=mesh,
                metrics=metrics,
                boundary_flux=False,
            )

    return _CollisionClosureTerms(energy_source=energy_source, momentum_source=momentum_source, diagnostics=diagnostics)


def _div_par_parallel_ion_viscosity_open(
    eta: np.ndarray,
    velocity: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    bxy = np.maximum(np.asarray(metrics.Bxy, dtype=np.float64), 1.0e-12)
    sqrt_b = np.sqrt(bxy)
    return sqrt_b * _div_par_k_grad_par_open(
        eta / bxy,
        sqrt_b * np.asarray(velocity, dtype=np.float64),
        mesh=mesh,
        metrics=metrics,
        boundary_flux=True,
    )


def _apply_neutral_parallel_diffusion(
    config: BoutConfig,
    *,
    species: dict[str, OpenFieldSpecies],
    prepared: dict[str, _PreparedSpeciesState],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
) -> _NeutralParallelDiffusionTerms:
    density_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    energy_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    momentum_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    diagnostics: dict[str, np.ndarray] = {}

    if "neutral_parallel_diffusion" not in set(_configured_component_names(config)):
        return _NeutralParallelDiffusionTerms(
            density_source=density_source,
            energy_source=energy_source,
            momentum_source=momentum_source,
            diagnostics=diagnostics,
        )

    section = "neutral_parallel_diffusion"
    dneut = float(config.parsed(section, "dneut")) if config.has_option(section, "dneut") else 0.0
    if dneut <= 0.0:
        return _NeutralParallelDiffusionTerms(
            density_source=density_source,
            energy_source=energy_source,
            momentum_source=momentum_source,
            diagnostics=diagnostics,
        )

    diffusion_mode = (
        str(config.parsed(section, "diffusion_collisions_mode")).strip().lower()
        if config.has_option(section, "diffusion_collisions_mode")
        else "afn"
    )
    equation_fix = bool(config.parsed(section, "equation_fix")) if config.has_option(section, "equation_fix") else True
    perpendicular_conduction = (
        bool(config.parsed(section, "perpendicular_conduction"))
        if config.has_option(section, "perpendicular_conduction")
        else True
    )
    perpendicular_viscosity = (
        bool(config.parsed(section, "perpendicular_viscosity"))
        if config.has_option(section, "perpendicular_viscosity")
        else True
    )
    diagnose = bool(config.parsed(section, "diagnose")) if config.has_option(section, "diagnose") else False

    collision_rates = _compute_collision_frequencies(config, species, prepared, dataset_scalars=dataset_scalars)
    ionisation_rates = _neutral_ionisation_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=dataset_scalars,
    )
    charge_exchange_rates = _neutral_charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=dataset_scalars,
    )

    advection_factor = 2.5 if equation_fix else 1.5
    kappa_factor = 2.5 if equation_fix else 1.0

    for name, sp in species.items():
        if name == "e" or sp.charge != 0.0:
            continue

        nu = np.zeros_like(prepared[name].density, dtype=np.float64)
        if diffusion_mode == "afn":
            if name in ionisation_rates:
                nu = nu + ionisation_rates[name]
            if name in charge_exchange_rates:
                nu = nu + charge_exchange_rates[name]
        elif diffusion_mode == "multispecies":
            for other_name in species:
                rate = collision_rates.get((name, other_name))
                if rate is not None:
                    nu = nu + rate
            if name in charge_exchange_rates:
                nu = nu + charge_exchange_rates[name]
        else:
            raise NotImplementedError(
                f"Unsupported neutral_parallel_diffusion diffusion_collisions_mode={diffusion_mode!r}."
            )

        density = np.asarray(prepared[name].density, dtype=np.float64)
        pressure = np.asarray(prepared[name].pressure, dtype=np.float64)
        temperature = np.asarray(prepared[name].temperature, dtype=np.float64)
        momentum = np.asarray(prepared[name].momentum, dtype=np.float64)
        velocity = np.asarray(prepared[name].velocity, dtype=np.float64)

        diffusion = dneut * temperature / np.maximum(sp.atomic_mass * nu, 1.0e-10)
        diffusion = _apply_open_field_dirichlet_scalar_guards(
            diffusion,
            mesh=mesh,
            lower_y=sp.noflow_lower_y,
            upper_y=sp.noflow_upper_y,
        )
        log_pressure = np.log(np.maximum(pressure, 1.0e-7))
        log_pressure = _apply_open_field_neumann_scalar_guards(
            log_pressure,
            mesh=mesh,
            lower_y=sp.noflow_lower_y,
            upper_y=sp.noflow_upper_y,
        )

        density_rhs = _div_par_k_grad_par_open(
            diffusion * density,
            log_pressure,
            mesh=mesh,
            metrics=metrics,
            boundary_flux=False,
        )
        density_source[name] = density_source[name] + density_rhs

        energy_rhs = _div_par_k_grad_par_open(
            diffusion * advection_factor * pressure,
            log_pressure,
            mesh=mesh,
            metrics=metrics,
            boundary_flux=False,
        )
        conductivity = kappa_factor * density * diffusion
        conductivity = _apply_open_field_neumann_scalar_guards(
            conductivity,
            mesh=mesh,
            lower_y=sp.noflow_lower_y,
            upper_y=sp.noflow_upper_y,
        )
        if perpendicular_conduction:
            energy_rhs = energy_rhs + _div_par_k_grad_par_open(
                conductivity,
                temperature,
                mesh=mesh,
                metrics=metrics,
                boundary_flux=False,
            )
        energy_source[name] = energy_source[name] + energy_rhs

        momentum_rhs = np.zeros_like(density_rhs, dtype=np.float64)
        if sp.has_momentum and perpendicular_viscosity:
            eta_n = (2.0 / 5.0) * conductivity
            momentum_rhs = _div_par_k_grad_par_open(
                diffusion * momentum,
                log_pressure,
                mesh=mesh,
                metrics=metrics,
                boundary_flux=False,
            )
            momentum_rhs = momentum_rhs + _div_par_k_grad_par_open(
                eta_n,
                velocity,
                mesh=mesh,
                metrics=metrics,
                boundary_flux=True,
            )
            momentum_source[name] = momentum_source[name] + momentum_rhs

        if diagnose:
            diagnostics[f"D{name}_Dpar"] = np.asarray(diffusion, dtype=np.float64)
            diagnostics[f"S{name}_Dpar"] = np.asarray(density_rhs, dtype=np.float64)
            diagnostics[f"E{name}_Dpar"] = np.asarray(energy_rhs, dtype=np.float64)
            if sp.has_momentum and perpendicular_viscosity:
                diagnostics[f"F{name}_Dpar"] = np.asarray(momentum_rhs, dtype=np.float64)

    return _NeutralParallelDiffusionTerms(
        density_source=density_source,
        energy_source=energy_source,
        momentum_source=momentum_source,
        diagnostics=diagnostics,
    )


def _apply_open_field_neumann_scalar_guards(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    lower_y: bool,
    upper_y: bool,
) -> np.ndarray:
    return np.asarray(
        apply_noflow_scalar_guards(field, mesh=mesh, lower_y=lower_y, upper_y=upper_y),
        dtype=np.float64,
    )


def _apply_open_field_dirichlet_scalar_guards(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    lower_y: bool,
    upper_y: bool,
) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64, copy=True)
    if mesh.myg <= 0:
        return result
    if lower_y:
        result[:, mesh.ystart - 1, :] = -result[:, mesh.ystart, :]
    if upper_y:
        result[:, mesh.yend + 1, :] = -result[:, mesh.yend, :]
    return result


def _conduction_kappa_coefficient(config: BoutConfig, species: OpenFieldSpecies) -> float:
    if config.has_option(species.name, "kappa_coefficient"):
        return float(NumericResolver(config).resolve(species.name, "kappa_coefficient"))
    if species.charge < 0.0:
        return 3.16 / math.sqrt(2.0)
    if species.charge == 0.0:
        return 2.5
    return 3.9


def _conduction_collision_time(
    config: BoutConfig,
    *,
    species: dict[str, OpenFieldSpecies],
    prepared: dict[str, _PreparedSpeciesState],
    collision_rates: dict[tuple[str, str], np.ndarray],
    cx_rates: dict[str, np.ndarray],
    species_name: str,
) -> np.ndarray:
    species_type = "electron" if species_name == "e" else ("neutral" if species[species_name].charge == 0.0 else "ion")
    mode = str(config.parsed(species_name, "conduction_collisions_mode")).strip().lower() if config.has_option(species_name, "conduction_collisions_mode") else "multispecies"
    total = np.zeros_like(prepared[species_name].density, dtype=np.float64)

    if mode == "braginskii":
        if species_type == "electron":
            rate = collision_rates.get((species_name, species_name))
            if rate is not None:
                total = total + rate
        elif species_type == "ion":
            rate = collision_rates.get((species_name, species_name))
            if rate is not None:
                total = total + rate
        else:
            raise NotImplementedError("Neutral conduction_collisions_mode='braginskii' is not supported.")
    elif mode == "multispecies":
        for other_name in species:
            rate = collision_rates.get((species_name, other_name))
            if rate is not None:
                total = total + rate
        if species_name in cx_rates:
            total = total + cx_rates[species_name]
    elif mode == "afn":
        if species_type != "neutral":
            raise NotImplementedError("Conduction_collisions_mode='afn' is only supported for neutrals.")
        for other_name, other_species in species.items():
            if other_species.charge > 0.0:
                cx_rate = collision_rates.get((species_name, other_name))
                if cx_rate is not None:
                    total = total + cx_rate
        if species_name in cx_rates:
            total = total + cx_rates[species_name]
    else:
        raise NotImplementedError(f"Unsupported conduction_collisions_mode {mode!r} for {species_name}.")

    return 1.0 / np.maximum(total, 1.0e-10)


def _neutral_ionisation_collision_rates(
    config: BoutConfig,
    *,
    species: dict[str, OpenFieldSpecies],
    prepared: dict[str, _PreparedSpeciesState],
    dataset_scalars: dict[str, float],
) -> dict[str, np.ndarray]:
    if not config.has_section("reactions") or not config.has_option("reactions", "type"):
        return {}
    totals: dict[str, np.ndarray] = {}
    electron_density = _electron_density(tuple(sp for sp in species.values() if sp.charge > 0.0))
    electron_temperature = _safe_temperature(
        species["e"].pressure,
        electron_density,
        species["e"].density_floor,
    )
    reactions = config.parsed("reactions", "type")
    for reaction in (reactions if isinstance(reactions, tuple) else (reactions,)):
        tokens = tuple(part.strip() for part in str(reaction).split("->"))
        if len(tokens) != 2:
            continue
        lhs = tuple(part.strip() for part in re.split(r"\s+\+\s+", tokens[0].strip()))
        rhs = tuple(part.strip() for part in re.split(r"\s+\+\s+", tokens[1].strip()))
        if not (len(lhs) == 2 and lhs[1] == "e" and len(rhs) == 2 and rhs[1] == "2e"):
            continue
        atom_name = lhs[0]
        if atom_name not in species or species[atom_name].charge != 0.0:
            continue
        sigma_v_coeffs, _, _ = _load_amjuel_rate(atom_name, "iz")
        sigma_v = _eval_amjuel_fit(
            np.asarray(electron_temperature, dtype=np.float64) * dataset_scalars["Tnorm"],
            np.asarray(electron_density, dtype=np.float64) * dataset_scalars["Nnorm"],
            sigma_v_coeffs,
        )
        totals[atom_name] = np.asarray(
            np.asarray(electron_density, dtype=np.float64)
            * sigma_v
            * (dataset_scalars["Nnorm"] / dataset_scalars["Omega_ci"]),
            dtype=np.float64,
        )
    return totals


def _neutral_charge_exchange_collision_rates(
    config: BoutConfig,
    *,
    species: dict[str, OpenFieldSpecies],
    prepared: dict[str, _PreparedSpeciesState],
    dataset_scalars: dict[str, float],
) -> dict[str, np.ndarray]:
    if not config.has_section("reactions") or not config.has_option("reactions", "type"):
        return {}
    totals: dict[str, np.ndarray] = {}
    reactions = config.parsed("reactions", "type")
    for reaction in (reactions if isinstance(reactions, tuple) else (reactions,)):
        tokens = tuple(part.strip() for part in str(reaction).split("->"))
        if len(tokens) != 2:
            continue
        lhs = tuple(part.strip() for part in re.split(r"\s+\+\s+", tokens[0].strip()))
        rhs = tuple(part.strip() for part in re.split(r"\s+\+\s+", tokens[1].strip()))
        if not _is_charge_exchange_reaction(lhs, rhs):
            continue
        atom_name = lhs[0]
        ion_name = lhs[1]
        if atom_name not in species or ion_name not in species:
            continue
        atom = species[atom_name]
        ion = species[ion_name]
        atom_temperature = prepared[atom_name].temperature
        ion_temperature = prepared[ion_name].temperature
        teff = np.clip(
            (atom_temperature / atom.atomic_mass + ion_temperature / ion.atomic_mass) * dataset_scalars["Tnorm"],
            0.01,
            10000.0,
        )
        sigma_v = _hydrogen_cx_sigmav(teff, dataset_scalars)
        totals[atom_name] = np.asarray(
            totals.get(atom_name, 0.0) + prepared[ion_name].density * sigma_v,
            dtype=np.float64,
        )
    return totals


def _amjuel_ionisation(
    atom_name: str,
    ion_name: str,
    *,
    species: dict[str, OpenFieldSpecies],
    electron_density: np.ndarray,
    dataset_scalars: dict[str, float],
) -> _ReactionTerms:
    atom = species[atom_name]
    ion = species[ion_name]
    electron_pressure = species["e"].pressure
    electron_temperature = _safe_temperature(
        electron_pressure,
        electron_density,
        species["e"].density_floor,
    )
    atom_temperature = _safe_temperature(atom.pressure, atom.density, atom.density_floor)
    sigma_v, sigma_v_E, electron_heating = _load_amjuel_rate(atom_name, "iz")
    rate = _amjuel_reaction_rate(atom.density, electron_density, electron_temperature, sigma_v, dataset_scalars)
    radiation = _amjuel_energy_loss(atom.density, electron_density, electron_temperature, sigma_v_E, electron_heating, rate, dataset_scalars)

    density_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    energy_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    momentum_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}

    density_source[atom_name] -= rate
    density_source[ion_name] += rate
    atom_velocity = atom.momentum / np.maximum(atom.atomic_mass * atom.density, 1.0e-8)
    ion_momentum = rate * atom.atomic_mass * atom_velocity
    momentum_source[atom_name] -= ion_momentum
    momentum_source[ion_name] += ion_momentum
    energy_source[atom_name] -= 1.5 * rate * atom_temperature
    energy_source[ion_name] += 1.5 * rate * atom_temperature
    energy_source["e"] -= radiation
    diagnostics = {
        f"S{ion_name}_iz": rate,
        f"F{ion_name}_iz": ion_momentum,
        f"E{ion_name}_iz": 1.5 * rate * atom_temperature,
        f"R{ion_name}_ex": -radiation,
    }
    return _ReactionTerms(density_source=density_source, energy_source=energy_source, momentum_source=momentum_source, diagnostics=diagnostics)


def _amjuel_recombination(
    atom_name: str,
    ion_name: str,
    *,
    species: dict[str, OpenFieldSpecies],
    electron_density: np.ndarray,
    dataset_scalars: dict[str, float],
) -> _ReactionTerms:
    atom = species[atom_name]
    ion = species[ion_name]
    electron_pressure = species["e"].pressure
    electron_temperature = _safe_temperature(
        electron_pressure,
        electron_density,
        species["e"].density_floor,
    )
    ion_temperature = _safe_temperature(ion.pressure, ion.density, ion.density_floor)
    sigma_v, sigma_v_E, electron_heating = _load_amjuel_rate(atom_name, "rec")
    rate = _amjuel_reaction_rate(ion.density, electron_density, electron_temperature, sigma_v, dataset_scalars)
    radiation = _amjuel_energy_loss(ion.density, electron_density, electron_temperature, sigma_v_E, electron_heating, rate, dataset_scalars)

    density_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    energy_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    momentum_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}

    density_source[ion_name] -= rate
    density_source[atom_name] += rate
    ion_velocity = ion.momentum / np.maximum(ion.atomic_mass * ion.density, 1.0e-8)
    ion_momentum = rate * ion.atomic_mass * ion_velocity
    momentum_source[ion_name] -= ion_momentum
    momentum_source[atom_name] += ion_momentum
    energy_source[ion_name] -= 1.5 * rate * ion_temperature
    energy_source[atom_name] += 1.5 * rate * ion_temperature
    energy_source["e"] -= radiation
    diagnostics = {
        f"S{ion_name}_rec": -rate,
        f"F{ion_name}_rec": -ion_momentum,
        f"E{ion_name}_rec": -(1.5 * rate * ion_temperature),
        f"R{ion_name}_rec": -radiation,
    }
    return _ReactionTerms(density_source=density_source, energy_source=energy_source, momentum_source=momentum_source, diagnostics=diagnostics)


def _charge_exchange(
    atom1_name: str,
    ion1_name: str,
    atom2_name: str,
    ion2_name: str,
    *,
    species: dict[str, OpenFieldSpecies],
    dataset_scalars: dict[str, float],
) -> _ReactionTerms:
    atom1 = species[atom1_name]
    ion1 = species[ion1_name]
    atom2 = species[atom2_name]
    ion2 = species[ion2_name]
    atom_temperature = _safe_temperature(atom1.pressure, atom1.density, atom1.density_floor)
    ion_temperature = _safe_temperature(ion1.pressure, ion1.density, ion1.density_floor)
    teff = np.clip((atom_temperature / atom1.atomic_mass + ion_temperature / ion1.atomic_mass) * dataset_scalars["Tnorm"], 0.01, 10000.0)
    sigmav = _hydrogen_cx_sigmav(teff, dataset_scalars)
    rate = atom1.density * ion1.density * sigmav

    density_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    energy_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    momentum_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}

    atom_velocity = atom1.momentum / np.maximum(atom1.atomic_mass * atom1.density, 1.0e-8)
    ion_velocity = ion1.momentum / np.maximum(ion1.atomic_mass * ion1.density, 1.0e-8)
    atom2_velocity = atom2.momentum / np.maximum(atom2.atomic_mass * atom2.density, 1.0e-8)
    ion2_velocity = ion2.momentum / np.maximum(ion2.atomic_mass * ion2.density, 1.0e-8)

    if atom1_name != atom2_name or ion1_name != ion2_name:
        density_source[atom1_name] -= rate
        density_source[ion2_name] += rate
        density_source[ion1_name] -= rate
        density_source[atom2_name] += rate

    atom_momentum = rate * atom1.atomic_mass * atom_velocity
    ion_momentum = rate * ion1.atomic_mass * ion_velocity
    momentum_source[atom1_name] -= atom_momentum
    momentum_source[ion2_name] += atom_momentum
    momentum_source[ion1_name] -= ion_momentum
    momentum_source[atom2_name] += ion_momentum

    atom_energy = 1.5 * rate * atom_temperature
    ion_energy = 1.5 * rate * ion_temperature
    energy_source[atom1_name] -= atom_energy
    energy_source[ion2_name] += atom_energy
    energy_source[ion1_name] -= ion_energy
    energy_source[atom2_name] += ion_energy
    energy_source[ion2_name] += 0.5 * atom1.atomic_mass * rate * np.square(ion2_velocity - atom_velocity)
    energy_source[atom2_name] += 0.5 * ion1.atomic_mass * rate * np.square(atom2_velocity - ion_velocity)

    diag_suffix = f"{atom1_name}{ion1_name}_cx"
    if atom1_name == atom2_name and ion1_name == ion2_name:
        diagnostics = {
            f"E{diag_suffix}": ion_energy - atom_energy,
            f"F{diag_suffix}": ion_momentum - atom_momentum,
            f"K{diag_suffix}": ion1.density * sigmav,
        }
    else:
        diagnostics = {
            f"S{diag_suffix}": -rate,
            f"F{diag_suffix}": -atom_momentum,
            f"F{ion1_name}{atom1_name}_cx": -ion_momentum,
            f"E{diag_suffix}": -atom_energy,
            f"E{ion1_name}{atom1_name}_cx": -ion_energy,
            f"K{diag_suffix}": ion1.density * sigmav,
        }
    return _ReactionTerms(density_source=density_source, energy_source=energy_source, momentum_source=momentum_source, diagnostics=diagnostics)


def _charge_exchange_collision_rates(
    config: BoutConfig,
    *,
    species: dict[str, OpenFieldSpecies],
    prepared: dict[str, _PreparedSpeciesState],
    dataset_scalars: dict[str, float],
) -> dict[str, np.ndarray]:
    if not config.has_section("reactions") or not config.has_option("reactions", "type"):
        return {}
    totals: dict[str, np.ndarray] = {}
    reactions = config.parsed("reactions", "type")
    for reaction in (reactions if isinstance(reactions, tuple) else (reactions,)):
        tokens = tuple(part.strip() for part in str(reaction).split("->"))
        if len(tokens) != 2:
            continue
        lhs = tuple(part.strip() for part in re.split(r"\s+\+\s+", tokens[0].strip()))
        rhs = tuple(part.strip() for part in re.split(r"\s+\+\s+", tokens[1].strip()))
        if not _is_charge_exchange_reaction(lhs, rhs):
            continue
        atom1_name = lhs[0]
        ion1_name = lhs[1]
        if atom1_name not in species or ion1_name not in species:
            continue
        atom1 = species[atom1_name]
        ion1 = species[ion1_name]
        atom_temperature = prepared[atom1_name].temperature
        ion_temperature = prepared[ion1_name].temperature
        teff = np.clip(
            (atom_temperature / atom1.atomic_mass + ion_temperature / ion1.atomic_mass) * dataset_scalars["Tnorm"],
            0.01,
            10000.0,
        )
        sigmav = _hydrogen_cx_sigmav(teff, dataset_scalars)
        atom_rate = prepared[ion1_name].density * sigmav
        ion_rate = prepared[atom1_name].density * sigmav
        if atom1_name in totals:
            totals[atom1_name] = totals[atom1_name] + atom_rate
        else:
            totals[atom1_name] = np.asarray(atom_rate, dtype=np.float64)
        if ion1_name in totals:
            totals[ion1_name] = totals[ion1_name] + ion_rate
        else:
            totals[ion1_name] = np.asarray(ion_rate, dtype=np.float64)
    return totals


def _amjuel_reaction_rate(
    heavy_density: np.ndarray,
    electron_density: np.ndarray,
    electron_temperature: np.ndarray,
    sigma_v_coeffs: np.ndarray,
    dataset_scalars: dict[str, float],
) -> np.ndarray:
    sigma_v = _eval_amjuel_fit(
        np.asarray(electron_temperature, dtype=np.float64) * dataset_scalars["Tnorm"],
        np.asarray(electron_density, dtype=np.float64) * dataset_scalars["Nnorm"],
        sigma_v_coeffs,
    )
    return np.asarray(heavy_density, dtype=np.float64) * np.asarray(electron_density, dtype=np.float64) * sigma_v * (
        dataset_scalars["Nnorm"] / dataset_scalars["Omega_ci"]
    )


def _amjuel_energy_loss(
    heavy_density: np.ndarray,
    electron_density: np.ndarray,
    electron_temperature: np.ndarray,
    sigma_v_E_coeffs: np.ndarray,
    electron_heating: float,
    reaction_rate: np.ndarray,
    dataset_scalars: dict[str, float],
) -> np.ndarray:
    sigma_v_E = _eval_amjuel_fit(
        np.asarray(electron_temperature, dtype=np.float64) * dataset_scalars["Tnorm"],
        np.asarray(electron_density, dtype=np.float64) * dataset_scalars["Nnorm"],
        sigma_v_E_coeffs,
    )
    energy_loss = (
        np.asarray(heavy_density, dtype=np.float64)
        * np.asarray(electron_density, dtype=np.float64)
        * sigma_v_E
        * dataset_scalars["Nnorm"]
        / (dataset_scalars["Tnorm"] * dataset_scalars["Omega_ci"])
    )
    return energy_loss - (electron_heating / dataset_scalars["Tnorm"]) * reaction_rate


@lru_cache(maxsize=None)
def _load_amjuel_rate(species_name: str, reaction_kind: str) -> tuple[np.ndarray, np.ndarray, float]:
    filename = _AMJUEL_FILENAMES[(species_name, reaction_kind)]
    payload = json.loads(resources.files("jax_drb.data.atomic_rates").joinpath(filename).read_text(encoding="utf-8"))
    return (
        np.asarray(payload["sigma_v_coeffs"], dtype=np.float64),
        np.asarray(payload["sigma_v_E_coeffs"], dtype=np.float64),
        float(payload["electron_heating"]),
    )


def _eval_amjuel_fit(temperature_ev: np.ndarray, density_m3: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    temperature = np.clip(np.asarray(temperature_ev, dtype=np.float64), 0.1, 1.0e4)
    density = np.clip(np.asarray(density_m3, dtype=np.float64), 1.0e14, 1.0e22)
    logn = np.log(density / 1.0e14)
    logt = np.log(temperature)
    result = np.zeros_like(logt, dtype=np.float64)
    logt_power = np.ones_like(logt, dtype=np.float64)
    for row in coeffs:
        logn_power = np.ones_like(logn, dtype=np.float64)
        for coefficient in row:
            result = result + coefficient * logn_power * logt_power
            logn_power = logn_power * logn
        logt_power = logt_power * logt
    return np.exp(result) * 1.0e-6


def _hydrogen_cx_sigmav(teff_ev: np.ndarray, dataset_scalars: dict[str, float]) -> np.ndarray:
    lnT = np.log(np.asarray(teff_ev, dtype=np.float64))
    ln_sigma_v = -18.5028
    lnT_power = lnT.copy()
    for coefficient in (0.3708409, 7.949876e-3, -6.143769e-4, -4.698969e-4, -4.096807e-4, 1.440382e-4, -1.514243e-5, 5.122435e-7):
        ln_sigma_v = ln_sigma_v + coefficient * lnT_power
        lnT_power = lnT_power * lnT
    return np.exp(ln_sigma_v) * (1.0e-6 * dataset_scalars["Nnorm"] / dataset_scalars["Omega_ci"])


@dataclass(frozen=True)
class _IonBoundaryResult:
    density: dict[str, np.ndarray]
    pressure: dict[str, np.ndarray]
    temperature: dict[str, np.ndarray]
    velocity: dict[str, np.ndarray]
    momentum: dict[str, np.ndarray]
    energy_source: dict[str, np.ndarray]


def _apply_ion_sheath_boundary(
    ions: tuple[OpenFieldSpecies, ...],
    *,
    electron_pressure: np.ndarray,
    electron_density: np.ndarray,
    electron_density_floor: float,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> _IonBoundaryResult:
    te = _safe_temperature(electron_pressure, electron_density, electron_density_floor)
    g22 = np.asarray(metrics.g_22, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    boundary_density: dict[str, np.ndarray] = {}
    boundary_pressure: dict[str, np.ndarray] = {}
    boundary_temperature: dict[str, np.ndarray] = {}
    velocity: dict[str, np.ndarray] = {}
    momentum: dict[str, np.ndarray] = {}
    energy_source: dict[str, np.ndarray] = {ion.name: np.zeros_like(ion.density, dtype=np.float64) for ion in ions}

    for ion in ions:
        density = ion.density.copy()
        pressure = ion.pressure.copy()
        temperature = _safe_temperature(pressure, density, ion.density_floor)
        vel = ion.momentum / np.maximum(
            ion.atomic_mass * _soft_floor(density, ion.density_floor),
            1.0e-8,
        )
        if ion.noflow_lower_y:
            density = np.array(apply_noflow_scalar_guards(density, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64, copy=True)
            temperature = np.array(apply_noflow_scalar_guards(temperature, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64, copy=True)
            pressure = np.array(apply_noflow_scalar_guards(pressure, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64, copy=True)
            vel = np.array(apply_noflow_flow_guards(vel, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64, copy=True)

        momentum_field = ion.momentum.copy()
        if mesh.has_upper_y_target:
            j = mesh.yend
            jp = j + 1
            jm = j - 1
            ni_i = density[:, j, :]
            ni_m = density[:, jm, :]
            density[:, jp, :] = np.asarray(limit_free(ni_m, ni_i, 0), dtype=np.float64)
            temperature[:, jp, :] = np.asarray(limit_free(temperature[:, jm, :], temperature[:, j, :], 0), dtype=np.float64)
            pressure[:, jp, :] = np.asarray(limit_free(pressure[:, jm, :], pressure[:, j, :], 0), dtype=np.float64)

            nisheath = 0.5 * (density[:, jp, :] + density[:, j, :])
            nesheath = 0.5 * (electron_density[:, jp, :] + electron_density[:, j, :]) if jp < electron_density.shape[1] else 0.5 * (electron_density[:, jm, :] + electron_density[:, j, :])
            tesheath = np.maximum(0.5 * (te[:, jp, :] + te[:, j, :]) if jp < te.shape[1] else 0.5 * (te[:, jm, :] + te[:, j, :]), 1.0e-5)
            tisheath = np.maximum(0.5 * (temperature[:, jp, :] + temperature[:, j, :]), 1.0e-5)
            s_i = np.clip(nisheath / np.maximum(nesheath, 1.0e-10), 0.0, 1.0)
            grad_ne = electron_density[:, j, :] - nesheath
            grad_ni = density[:, j, :] - nisheath
            mask = np.abs(grad_ni) < 1.0e-3
            grad_ne = np.where(mask, 1.0e-3, grad_ne)
            grad_ni = np.where(mask, 1.0e-3, grad_ni)
            c_i_sq = np.clip(((5.0 / 3.0) * tisheath + ion.charge * s_i * tesheath * grad_ne / grad_ni) / ion.atomic_mass, 0.0, 100.0)
            gamma_i = 2.5 + 0.5 * ion.atomic_mass * c_i_sq / tisheath
            visheath = np.sqrt(c_i_sq)
            vel[:, jp, :] = 2.0 * visheath - vel[:, j, :]
            momentum_field[:, jp, :] = 2.0 * ion.atomic_mass * nisheath * visheath - momentum_field[:, j, :]

            q = ((gamma_i - 1.0 - 1.0 / ((5.0 / 3.0) - 1.0)) * tisheath - 0.5 * c_i_sq * ion.atomic_mass) * nisheath * visheath
            q = np.maximum(q, 0.0)
            flux = q * (J[:, j, :] + J[:, jp, :]) / (np.sqrt(g22[:, j, :]) + np.sqrt(g22[:, jp, :]))
            power = flux / (dy[:, j, :] * J[:, j, :])
            energy_source[ion.name][:, j, :] -= power

        if mesh.has_lower_y_target:
            j = mesh.ystart
            jm = j - 1
            jp = j + 1
            ni_i = density[:, j, :]
            ni_p = density[:, jp, :]
            density[:, jm, :] = np.asarray(limit_free(ni_p, ni_i, 0), dtype=np.float64)
            temperature[:, jm, :] = np.asarray(limit_free(temperature[:, jp, :], temperature[:, j, :], 0), dtype=np.float64)
            pressure[:, jm, :] = np.asarray(limit_free(pressure[:, jp, :], pressure[:, j, :], 0), dtype=np.float64)

            nisheath = 0.5 * (density[:, jm, :] + density[:, j, :])
            nesheath = 0.5 * (electron_density[:, jm, :] + electron_density[:, j, :])
            tesheath = np.maximum(0.5 * (te[:, jm, :] + te[:, j, :]), 1.0e-5)
            tisheath = np.maximum(0.5 * (temperature[:, jm, :] + temperature[:, j, :]), 1.0e-5)
            s_i = np.clip(nisheath / np.maximum(nesheath, 1.0e-10), 0.0, 1.0)
            grad_ne = electron_density[:, j, :] - nesheath
            grad_ni = density[:, j, :] - nisheath
            mask = np.abs(grad_ni) < 1.0e-3
            grad_ne = np.where(mask, 1.0e-3, grad_ne)
            grad_ni = np.where(mask, 1.0e-3, grad_ni)
            c_i_sq = np.clip(((5.0 / 3.0) * tisheath + ion.charge * s_i * tesheath * grad_ne / grad_ni) / ion.atomic_mass, 0.0, 100.0)
            gamma_i = 2.5 + 0.5 * ion.atomic_mass * c_i_sq / tisheath
            visheath = np.sqrt(c_i_sq)
            vel[:, jm, :] = -2.0 * visheath - vel[:, j, :]
            momentum_field[:, jm, :] = -2.0 * ion.atomic_mass * nisheath * visheath - momentum_field[:, j, :]

            q = ((gamma_i - 1.0 - 1.0 / ((5.0 / 3.0) - 1.0)) * tisheath - 0.5 * c_i_sq * ion.atomic_mass) * nisheath * visheath
            q = np.maximum(q, 0.0)
            flux = q * (J[:, j, :] + J[:, jm, :]) / (np.sqrt(g22[:, j, :]) + np.sqrt(g22[:, jm, :]))
            power = flux / (dy[:, j, :] * J[:, j, :])
            energy_source[ion.name][:, j, :] -= power

        boundary_density[ion.name] = density
        boundary_pressure[ion.name] = pressure
        boundary_temperature[ion.name] = temperature
        velocity[ion.name] = vel
        momentum[ion.name] = momentum_field
    return _IonBoundaryResult(
        density=boundary_density,
        pressure=boundary_pressure,
        temperature=boundary_temperature,
        velocity=velocity,
        momentum=momentum,
        energy_source=energy_source,
    )


@dataclass(frozen=True)
class _ElectronBoundaryResult:
    density: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    velocity: np.ndarray
    momentum: np.ndarray
    energy_source: np.ndarray


def _apply_electron_sheath_boundary(
    *,
    electron_pressure: np.ndarray,
    electron_density: np.ndarray,
    electron_velocity: np.ndarray,
    electron_mass: float,
    electron_density_floor: float,
    ion_velocity: dict[str, np.ndarray],
    ions: tuple[OpenFieldSpecies, ...],
    prepared_ions: dict[str, _PreparedSpeciesState],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> _ElectronBoundaryResult:
    density = np.array(apply_noflow_scalar_guards(electron_density, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64, copy=True)
    pressure = np.array(apply_noflow_scalar_guards(electron_pressure, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64, copy=True)
    temperature = _safe_temperature(pressure, density, electron_density_floor)
    velocity = np.array(
        apply_noflow_flow_guards(
            np.asarray(electron_velocity, dtype=np.float64),
            mesh=mesh,
            lower_y=True,
            upper_y=False,
        ),
        dtype=np.float64,
        copy=True,
    )
    momentum = np.asarray(electron_mass * density * velocity, dtype=np.float64)
    me = 1.0 / 1836.0
    g22 = np.asarray(metrics.g_22, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    phi = np.zeros_like(density, dtype=np.float64)
    energy_source = np.zeros_like(density, dtype=np.float64)

    if mesh.has_upper_y_target:
        j = mesh.yend
        jp = j + 1
        jm = j - 1
        density[:, jp, :] = np.asarray(limit_free(density[:, jm, :], density[:, j, :], 0), dtype=np.float64)
        temperature[:, jp, :] = np.asarray(limit_free(temperature[:, jm, :], temperature[:, j, :], 0), dtype=np.float64)
        pressure[:, jp, :] = np.asarray(limit_free(pressure[:, jm, :], pressure[:, j, :], 0), dtype=np.float64)

        ion_sum = np.zeros_like(density[:, j, :], dtype=np.float64)
        for ion in ions:
            ion_state = prepared_ions[ion.name]
            ti = np.asarray(ion_state.temperature, dtype=np.float64)
            ni = np.asarray(ion_state.density, dtype=np.float64)
            s_i = np.clip(0.5 * (3.0 * ni[:, j, :] / np.maximum(density[:, j, :], 1.0e-12) - ni[:, jm, :] / np.maximum(density[:, jm, :], 1.0e-12)), 0.0, 1.0)
            s_i = np.where(np.isfinite(s_i), s_i, 1.0)
            grad_ne = density[:, j, :] - density[:, jm, :]
            grad_ni = ni[:, j, :] - ni[:, jm, :]
            mask = np.abs(grad_ni) < 2.0e-3
            grad_ne = np.where(mask, 2.0e-3, grad_ne)
            grad_ni = np.where(mask, 2.0e-3, grad_ni)
            c_i_sq = np.clip(((5.0 / 3.0) * ti[:, j, :] + ion.charge * s_i * temperature[:, j, :] * grad_ne / grad_ni) / ion.atomic_mass, 0.0, 100.0)
            ion_sum = ion_sum + s_i * ion.charge * np.sqrt(c_i_sq)

        valid = temperature[:, j, :] > 0.0
        safe_temperature = np.maximum(temperature[:, j, :], 1.0e-12)
        log_argument = np.sqrt(safe_temperature / (me * (2.0 * math.pi))) / np.maximum(ion_sum, 1.0e-12)
        phi[:, j, :] = np.where(valid, safe_temperature * np.log(np.maximum(log_argument, 1.0e-12)), 0.0)
        phi[:, jp, :] = phi[:, j, :]
        phi[:, j - 1, :] = phi[:, j, :]

        phisheath = np.maximum(0.5 * (phi[:, jp, :] + phi[:, j, :]), 0.0)
        tesheath = 0.5 * (temperature[:, jp, :] + temperature[:, j, :])
        nesheath = 0.5 * (density[:, jp, :] + density[:, j, :])
        gamma_e = np.maximum(2.0 + phisheath / np.maximum(tesheath, 1.0e-5), 0.0)
        vesheath = np.where(
            tesheath < 1.0e-10,
            0.0,
            np.sqrt(tesheath / (2.0 * math.pi * me)) * np.exp(-phisheath / np.maximum(tesheath, 1.0e-12)),
        )
        velocity[:, jp, :] = 2.0 * vesheath - velocity[:, j, :]
        momentum[:, jp, :] = 2.0 * electron_mass * nesheath * vesheath - momentum[:, j, :]
        q = ((gamma_e - 1.0 - 1.0 / ((5.0 / 3.0) - 1.0)) * tesheath - 0.5 * me * np.square(vesheath)) * nesheath * vesheath
        q = np.maximum(q, 0.0)
        flux = q * (J[:, j, :] + J[:, jp, :]) / (np.sqrt(g22[:, j, :]) + np.sqrt(g22[:, jp, :]))
        power = flux / (dy[:, j, :] * J[:, j, :])
        energy_source[:, j, :] -= power

    if mesh.has_lower_y_target:
        j = mesh.ystart
        jm = j - 1
        jp = j + 1
        density[:, jm, :] = np.asarray(limit_free(density[:, jp, :], density[:, j, :], 0), dtype=np.float64)
        temperature[:, jm, :] = np.asarray(limit_free(temperature[:, jp, :], temperature[:, j, :], 0), dtype=np.float64)
        pressure[:, jm, :] = np.asarray(limit_free(pressure[:, jp, :], pressure[:, j, :], 0), dtype=np.float64)

        ion_sum = np.zeros_like(density[:, j, :], dtype=np.float64)
        for ion in ions:
            ion_state = prepared_ions[ion.name]
            ti = np.asarray(ion_state.temperature, dtype=np.float64)
            ni = np.asarray(ion_state.density, dtype=np.float64)
            s_i = np.clip(0.5 * (3.0 * ni[:, j, :] / np.maximum(density[:, j, :], 1.0e-12) - ni[:, jp, :] / np.maximum(density[:, jp, :], 1.0e-12)), 0.0, 1.0)
            s_i = np.where(np.isfinite(s_i), s_i, 1.0)
            grad_ne = density[:, j, :] - density[:, jp, :]
            grad_ni = ni[:, j, :] - ni[:, jp, :]
            mask = np.abs(grad_ni) < 2.0e-3
            grad_ne = np.where(mask, 2.0e-3, grad_ne)
            grad_ni = np.where(mask, 2.0e-3, grad_ni)
            c_i_sq = np.clip(((5.0 / 3.0) * ti[:, j, :] + ion.charge * s_i * temperature[:, j, :] * grad_ne / grad_ni) / ion.atomic_mass, 0.0, 100.0)
            ion_sum = ion_sum + s_i * ion.charge * np.sqrt(c_i_sq)

        valid = temperature[:, j, :] > 0.0
        safe_temperature = np.maximum(temperature[:, j, :], 1.0e-12)
        log_argument = np.sqrt(safe_temperature / (me * (2.0 * math.pi))) / np.maximum(ion_sum, 1.0e-12)
        phi[:, j, :] = np.where(valid, safe_temperature * np.log(np.maximum(log_argument, 1.0e-12)), 0.0)
        phi[:, jm, :] = phi[:, j, :]
        phi[:, j + 1, :] = phi[:, j, :]

        phisheath = np.maximum(0.5 * (phi[:, jm, :] + phi[:, j, :]), 0.0)
        tesheath = 0.5 * (temperature[:, jm, :] + temperature[:, j, :])
        nesheath = 0.5 * (density[:, jm, :] + density[:, j, :])
        gamma_e = np.maximum(2.0 + phisheath / np.maximum(tesheath, 1.0e-5), 0.0)
        vesheath = np.where(
            tesheath < 1.0e-10,
            0.0,
            np.sqrt(tesheath / (2.0 * math.pi * me)) * np.exp(-phisheath / np.maximum(tesheath, 1.0e-12)),
        )
        velocity[:, jm, :] = -2.0 * vesheath - velocity[:, j, :]
        momentum[:, jm, :] = -2.0 * electron_mass * nesheath * vesheath - momentum[:, j, :]
        q = ((gamma_e - 1.0 - 1.0 / ((5.0 / 3.0) - 1.0)) * tesheath - 0.5 * me * np.square(vesheath)) * nesheath * vesheath
        q = np.maximum(q, 0.0)
        flux = q * (J[:, j, :] + J[:, jm, :]) / (np.sqrt(g22[:, j, :]) + np.sqrt(g22[:, jm, :]))
        power = flux / (dy[:, j, :] * J[:, j, :])
        energy_source[:, j, :] -= power
    return _ElectronBoundaryResult(
        density=density,
        temperature=temperature,
        pressure=pressure,
        velocity=velocity,
        momentum=momentum,
        energy_source=energy_source,
    )


@dataclass(frozen=True)
class _RecyclingTerms:
    density_source: dict[str, np.ndarray]
    energy_source: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]


def _target_recycling_sources(
    *,
    ions: tuple[OpenFieldSpecies, ...],
    prepared: dict[str, _PreparedSpeciesState],
    neutrals: tuple[OpenFieldSpecies, ...],
    ion_velocity: dict[str, np.ndarray],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> _RecyclingTerms:
    neutral_lookup = {sp.name: sp for sp in neutrals}
    density_source = {sp.name: np.zeros_like(sp.density, dtype=np.float64) for sp in (*ions, *neutrals)}
    energy_source = {sp.name: np.zeros_like(sp.density, dtype=np.float64) for sp in (*ions, *neutrals)}
    diagnostics: dict[str, np.ndarray] = {}

    for ion in ions:
        if not ion.target_recycle or ion.recycle_as is None or ion.recycle_as not in neutral_lookup:
            continue
        neutral = neutral_lookup[ion.recycle_as]
        ion_state = prepared[ion.name]
        result = compute_target_recycling_sources(
            ion_state.density,
            ion_velocity[ion.name],
            ion_state.temperature,
            mesh=mesh,
            J=np.asarray(metrics.J, dtype=np.float64),
            dy=np.asarray(metrics.dy, dtype=np.float64),
            dx=np.asarray(metrics.dx, dtype=np.float64),
            dz=np.asarray(metrics.dz, dtype=np.float64),
            g_22=np.asarray(metrics.g_22, dtype=np.float64),
            target_multiplier=ion.target_recycle_multiplier,
            target_energy=ion.target_recycle_energy,
            gamma_i=0.0,
            target_fast_recycle_fraction=ion.target_fast_recycle_fraction,
            target_fast_recycle_energy_factor=ion.target_fast_recycle_energy_factor,
            lower_y=mesh.has_lower_y_target,
            upper_y=mesh.has_upper_y_target,
        )
        density_source[neutral.name] = density_source[neutral.name] + np.asarray(result.density_source, dtype=np.float64)
        energy_source[neutral.name] = energy_source[neutral.name] + np.asarray(result.energy_source, dtype=np.float64)
        diagnostics[f"S{neutral.name}_target_recycle"] = np.asarray(result.target_density_source, dtype=np.float64)
        diagnostics[f"E{neutral.name}_target_recycle"] = np.asarray(result.target_energy_source, dtype=np.float64)

    return _RecyclingTerms(density_source=density_source, energy_source=energy_source, diagnostics=diagnostics)


def _electron_zero_current_velocity(
    ions: tuple[OpenFieldSpecies, ...],
    *,
    prepared: dict[str, _PreparedSpeciesState],
    ion_velocity: dict[str, np.ndarray],
    electron_density: np.ndarray,
) -> np.ndarray:
    current = np.zeros_like(electron_density, dtype=np.float64)
    for ion in ions:
        current = current + ion.charge * prepared[ion.name].density * ion_velocity[ion.name]
    return current / np.maximum(electron_density, 1.0e-5)


def _grad_par_electron_force_balance_open(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    """Match BOUT's Grad_par/DDY stencil used by electron_force_balance."""
    result = np.zeros_like(field, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    g_22 = np.asarray(metrics.g_22, dtype=np.float64)

    for i in range(mesh.xstart, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                result[i, j, k] = (
                    0.5
                    * (field[i, j + 1, k] - field[i, j - 1, k])
                    / (dy[i, j, k] * np.sqrt(g_22[i, j, k]))
                )
    return result


def advance_recycling_1d_implicit_history(
    config: BoutConfig,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    timestep: float,
    steps: int,
    solver_mode: str = "bdf",
    residual_tolerance: float = 1.0e-8,
    max_nonlinear_iterations: int = 20,
) -> Recycling1DHistoryResult:
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
    )
    field_names = runtime_model.field_names
    feedback_names = runtime_model.feedback_names
    fields = _recycling_field_templates(runtime_model.species_templates, field_names=field_names)
    integrals = {name: 0.0 for name in feedback_names}

    if solver_mode == "continuation":
        return _advance_recycling_1d_continuation_history(
            config,
            fields,
            runtime_model=runtime_model,
            feedback_integrals=integrals,
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            timestep=timestep,
            steps=steps,
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )

    if solver_mode == "adaptive_be":
        return _advance_recycling_1d_adaptive_be_history(
            config,
            fields,
            runtime_model=runtime_model,
            feedback_integrals=integrals,
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            timestep=timestep,
            steps=steps,
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )

    if solver_mode == "adaptive_bdf":
        return _advance_recycling_1d_adaptive_bdf_history(
            config,
            fields,
            runtime_model=runtime_model,
            feedback_integrals=integrals,
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            timestep=timestep,
            steps=steps,
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )

    if solver_mode == "bdf":
        return _advance_recycling_1d_bdf_history(
            config,
            fields,
            runtime_model=runtime_model,
            feedback_integrals=integrals,
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            timestep=timestep,
            steps=steps,
        )

    variable_history = {name: [np.asarray(fields[name], dtype=np.float64)] for name in field_names}
    feedback_history = {name: [np.asarray(0.0, dtype=np.float64)] for name in feedback_names}

    for _ in range(steps):
        fields, integrals, _ = advance_recycling_1d_backward_euler_step(
            config,
            fields,
            runtime_model=runtime_model,
            feedback_integrals=integrals,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            timestep=timestep,
            solver_mode=solver_mode,
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
        for name in field_names:
            variable_history[name].append(np.asarray(fields[name], dtype=np.float64))
        for name in feedback_names:
            feedback_history[name].append(np.asarray(integrals[name], dtype=np.float64))

    return Recycling1DHistoryResult(
        variable_history={name: np.stack(history, axis=0) for name, history in variable_history.items()},
        feedback_integral_history={name: np.stack(history, axis=0) for name, history in feedback_history.items()},
    )


def _advance_recycling_1d_continuation_history(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel,
    feedback_integrals: dict[str, float],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    timestep: float,
    steps: int,
    residual_tolerance: float,
    max_nonlinear_iterations: int,
) -> Recycling1DHistoryResult:
    current_fields = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
    current_integrals = {name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names}
    variable_history = {name: [np.asarray(current_fields[name], dtype=np.float64)] for name in field_names}
    feedback_history = {name: [np.asarray(current_integrals[name], dtype=np.float64)] for name in feedback_names}
    suggested_dt = _initial_recycling_continuation_dt(runtime_model, timestep=timestep)

    for _ in range(steps):
        current_fields, current_integrals, suggested_dt = _advance_recycling_1d_output_interval(
            config,
            current_fields,
            runtime_model=runtime_model,
            feedback_integrals=current_integrals,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            output_timestep=timestep,
            suggested_dt=suggested_dt,
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
        for name in field_names:
            variable_history[name].append(np.asarray(current_fields[name], dtype=np.float64))
        for name in feedback_names:
            feedback_history[name].append(np.asarray(current_integrals[name], dtype=np.float64))

    return Recycling1DHistoryResult(
        variable_history={name: np.stack(history, axis=0) for name, history in variable_history.items()},
        feedback_integral_history={name: np.stack(history, axis=0) for name, history in feedback_history.items()},
    )


def _advance_recycling_1d_adaptive_be_history(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel,
    feedback_integrals: dict[str, float],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    timestep: float,
    steps: int,
    residual_tolerance: float,
    max_nonlinear_iterations: int,
) -> Recycling1DHistoryResult:
    current_fields = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
    current_integrals = {name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names}
    variable_history = {name: [np.asarray(current_fields[name], dtype=np.float64)] for name in field_names}
    feedback_history = {name: [np.asarray(current_integrals[name], dtype=np.float64)] for name in feedback_names}
    suggested_dt = min(float(timestep), 10.0 if len(field_names) > 10 else 5.0)

    for _ in range(steps):
        current_fields, current_integrals, suggested_dt = _advance_recycling_1d_adaptive_be_interval(
            config,
            current_fields,
            runtime_model=runtime_model,
            feedback_integrals=current_integrals,
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            output_timestep=timestep,
            suggested_dt=suggested_dt,
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
        for name in field_names:
            variable_history[name].append(np.asarray(current_fields[name], dtype=np.float64))
        for name in feedback_names:
            feedback_history[name].append(np.asarray(current_integrals[name], dtype=np.float64))

    return Recycling1DHistoryResult(
        variable_history={name: np.stack(history, axis=0) for name, history in variable_history.items()},
        feedback_integral_history={name: np.stack(history, axis=0) for name, history in feedback_history.items()},
    )


def _advance_recycling_1d_adaptive_bdf_history(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel,
    feedback_integrals: dict[str, float],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    timestep: float,
    steps: int,
    residual_tolerance: float,
    max_nonlinear_iterations: int,
) -> Recycling1DHistoryResult:
    current_fields = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
    current_integrals = {name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names}
    previous_fields: dict[str, np.ndarray] | None = None
    previous_integrals: dict[str, float] | None = None
    previous_dt: float | None = None
    variable_history = {name: [np.asarray(current_fields[name], dtype=np.float64)] for name in field_names}
    feedback_history = {name: [np.asarray(current_integrals[name], dtype=np.float64)] for name in feedback_names}
    suggested_dt = _initial_recycling_continuation_dt(runtime_model, timestep=timestep)

    for _ in range(steps):
        (
            current_fields,
            current_integrals,
            previous_fields,
            previous_integrals,
            previous_dt,
            suggested_dt,
        ) = _advance_recycling_1d_adaptive_bdf_interval(
            config,
            current_fields,
            runtime_model=runtime_model,
            feedback_integrals=current_integrals,
            previous_fields=previous_fields,
            previous_integrals=previous_integrals,
            previous_dt=previous_dt,
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            output_timestep=timestep,
            suggested_dt=suggested_dt,
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
        for name in field_names:
            variable_history[name].append(np.asarray(current_fields[name], dtype=np.float64))
        for name in feedback_names:
            feedback_history[name].append(np.asarray(current_integrals[name], dtype=np.float64))

    return Recycling1DHistoryResult(
        variable_history={name: np.stack(history, axis=0) for name, history in variable_history.items()},
        feedback_integral_history={name: np.stack(history, axis=0) for name, history in feedback_history.items()},
    )


def _advance_recycling_1d_adaptive_bdf_interval(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel,
    feedback_integrals: dict[str, float],
    previous_fields: dict[str, np.ndarray] | None,
    previous_integrals: dict[str, float] | None,
    previous_dt: float | None,
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    output_timestep: float,
    suggested_dt: float,
    residual_tolerance: float,
    max_nonlinear_iterations: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, np.ndarray] | None,
    dict[str, float] | None,
    float | None,
    float,
]:
    relative_tolerance = float(config.parsed("solver", "rtol")) if config.has_option("solver", "rtol") else 1.0e-6
    absolute_tolerance = float(config.parsed("solver", "atol")) if config.has_option("solver", "atol") else 1.0e-9
    remaining = float(output_timestep)
    minimum_dt = max(float(output_timestep) / 8192.0, 0.25)
    dt = min(float(suggested_dt), remaining)
    current_fields = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
    current_integrals = {name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names}
    prev_fields = None if previous_fields is None else {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in previous_fields.items()}
    prev_integrals = None if previous_integrals is None else {name: float(previous_integrals.get(name, 0.0)) for name in feedback_names}
    prev_dt = previous_dt

    while remaining > 1.0e-12:
        dt = min(dt, remaining)
        use_bdf2 = (
            prev_fields is not None
            and prev_integrals is not None
            and prev_dt is not None
            and abs(float(prev_dt) - float(dt)) <= 1.0e-12
        )
        if not use_bdf2:
            candidate_fields, candidate_integrals, error_ratio = _advance_recycling_1d_startup_step(
                config,
                current_fields,
                runtime_model=runtime_model,
                feedback_integrals=current_integrals,
                field_names=field_names,
                feedback_names=feedback_names,
                mesh=mesh,
                metrics=metrics,
                dataset_scalars=dataset_scalars,
                timestep=dt,
                residual_tolerance=residual_tolerance,
                max_nonlinear_iterations=max_nonlinear_iterations,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        else:
            be_fields, be_integrals, _ = advance_recycling_1d_backward_euler_step(
                config,
                current_fields,
                runtime_model=runtime_model,
                feedback_integrals=current_integrals,
                mesh=mesh,
                metrics=metrics,
                dataset_scalars=dataset_scalars,
                timestep=dt,
                solver_mode="sparse",
                residual_tolerance=residual_tolerance,
                max_nonlinear_iterations=max_nonlinear_iterations,
            )
            bdf_fields, bdf_integrals, _ = advance_recycling_1d_bdf2_step(
                config,
                current_fields,
                prev_fields,
                runtime_model=runtime_model,
                feedback_integrals=current_integrals,
                previous_feedback_integrals=prev_integrals,
                mesh=mesh,
                metrics=metrics,
                dataset_scalars=dataset_scalars,
                timestep=dt,
                solver_mode="sparse",
                residual_tolerance=residual_tolerance,
                max_nonlinear_iterations=max_nonlinear_iterations,
            )
            error_ratio = _recycling_state_error_ratio(
                be_fields,
                be_integrals,
                bdf_fields,
                bdf_integrals,
                field_names=field_names,
                feedback_names=feedback_names,
                mesh=mesh,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            error_ratio /= 3.0
            if np.isfinite(error_ratio) and error_ratio <= 1.0:
                candidate_fields = bdf_fields
                candidate_integrals = bdf_integrals
            else:
                candidate_fields = be_fields
                candidate_integrals = be_integrals
        order = 2 if use_bdf2 else 1

        if np.isfinite(error_ratio) and error_ratio <= 1.0:
            prev_fields = current_fields
            prev_integrals = current_integrals
            current_fields = candidate_fields
            current_integrals = candidate_integrals
            prev_dt = dt
            remaining -= dt
            next_dt = _choose_recycling_next_dt(
                dt,
                error_ratio=error_ratio,
                order=order,
                remaining=remaining,
                minimum_dt=minimum_dt,
            )
            if abs(next_dt - dt) > 1.0e-12:
                prev_fields = None
                prev_integrals = None
                prev_dt = None
            dt = next_dt
            continue

        if dt <= minimum_dt:
            prev_fields = current_fields
            prev_integrals = current_integrals
            prev_dt = dt
            current_fields = candidate_fields
            current_integrals = candidate_integrals
            remaining -= dt
            continue

        dt = max(0.5 * dt, minimum_dt)
        prev_fields = None
        prev_integrals = None
        prev_dt = None

    return current_fields, current_integrals, prev_fields, prev_integrals, prev_dt, max(min(dt, float(output_timestep)), minimum_dt)


def _choose_recycling_next_dt(
    current_dt: float,
    *,
    error_ratio: float,
    order: int,
    remaining: float,
    minimum_dt: float,
) -> float:
    if remaining <= 1.0e-12:
        return current_dt
    if not np.isfinite(error_ratio) or error_ratio <= 0.0:
        factor = 2.0
    else:
        factor = 0.9 * error_ratio ** (-1.0 / float(order + 1))
    factor = min(max(factor, 0.5), 2.0)
    return max(min(current_dt * factor, remaining), minimum_dt)


def _advance_recycling_1d_startup_step(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel,
    feedback_integrals: dict[str, float],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    timestep: float,
    residual_tolerance: float,
    max_nonlinear_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    full_fields, full_integrals, _ = advance_recycling_1d_backward_euler_step(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        timestep=timestep,
        solver_mode="sparse",
        residual_tolerance=residual_tolerance,
        max_nonlinear_iterations=max_nonlinear_iterations,
    )
    half_fields, half_integrals, _ = advance_recycling_1d_backward_euler_step(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        timestep=0.5 * timestep,
        solver_mode="sparse",
        residual_tolerance=residual_tolerance,
        max_nonlinear_iterations=max_nonlinear_iterations,
    )
    half_fields, half_integrals, _ = advance_recycling_1d_backward_euler_step(
        config,
        half_fields,
        runtime_model=runtime_model,
        feedback_integrals=half_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        timestep=0.5 * timestep,
        solver_mode="sparse",
        residual_tolerance=residual_tolerance,
        max_nonlinear_iterations=max_nonlinear_iterations,
    )
    error_ratio = _recycling_state_error_ratio(
        full_fields,
        full_integrals,
        half_fields,
        half_integrals,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    return half_fields, half_integrals, error_ratio


def _advance_recycling_1d_adaptive_be_interval(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel,
    feedback_integrals: dict[str, float],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    output_timestep: float,
    suggested_dt: float,
    residual_tolerance: float,
    max_nonlinear_iterations: int,
) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    relative_tolerance = float(config.parsed("solver", "rtol")) if config.has_option("solver", "rtol") else 1.0e-6
    absolute_tolerance = float(config.parsed("solver", "atol")) if config.has_option("solver", "atol") else 1.0e-9
    remaining = float(output_timestep)
    minimum_dt = max(float(output_timestep) / 8192.0, 0.25)
    dt = min(float(suggested_dt), remaining)
    current_fields = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
    current_integrals = {name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names}

    while remaining > 1.0e-12:
        dt = min(dt, remaining)
        full_fields, full_integrals, _ = advance_recycling_1d_backward_euler_step(
            config,
            current_fields,
            runtime_model=runtime_model,
            feedback_integrals=current_integrals,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            timestep=dt,
            solver_mode="sparse",
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
        half_fields, half_integrals, _ = advance_recycling_1d_backward_euler_step(
            config,
            current_fields,
            runtime_model=runtime_model,
            feedback_integrals=current_integrals,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            timestep=0.5 * dt,
            solver_mode="sparse",
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
        half_fields, half_integrals, _ = advance_recycling_1d_backward_euler_step(
            config,
            half_fields,
            runtime_model=runtime_model,
            feedback_integrals=half_integrals,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            timestep=0.5 * dt,
            solver_mode="sparse",
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
        error_ratio = _recycling_state_error_ratio(
            full_fields,
            full_integrals,
            half_fields,
            half_integrals,
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        if np.isfinite(error_ratio) and error_ratio <= 1.0:
            current_fields = half_fields
            current_integrals = half_integrals
            remaining -= dt
            if error_ratio < 0.1:
                dt = min(2.0 * dt, remaining if remaining > 1.0e-12 else dt)
            continue
        if dt <= minimum_dt:
            current_fields = half_fields
            current_integrals = half_integrals
            remaining -= dt
            continue
        dt = max(0.5 * dt, minimum_dt)

    return current_fields, current_integrals, max(min(dt, float(output_timestep)), minimum_dt)


def _recycling_state_error_ratio(
    full_fields: dict[str, np.ndarray],
    full_integrals: dict[str, float],
    half_fields: dict[str, np.ndarray],
    half_integrals: dict[str, float],
    *,
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> float:
    active_slices = _recycling_active_domain_slices(mesh)
    squared_terms: list[np.ndarray] = []
    for name in field_names:
        full = np.asarray(full_fields[name], dtype=np.float64)[active_slices]
        half = np.asarray(half_fields[name], dtype=np.float64)[active_slices]
        scale = float(absolute_tolerance) + float(relative_tolerance) * np.maximum(np.abs(full), np.abs(half))
        squared_terms.append(np.square((half - full) / scale).ravel())
    for name in feedback_names:
        full = float(full_integrals.get(name, 0.0))
        half = float(half_integrals.get(name, 0.0))
        scale = float(absolute_tolerance) + float(relative_tolerance) * max(abs(full), abs(half), 1.0)
        squared_terms.append(np.asarray([(half - full) / scale], dtype=np.float64))
    if not squared_terms:
        return 0.0
    combined = np.concatenate(squared_terms)
    return float(np.sqrt(np.mean(combined * combined)))


def _initial_recycling_continuation_dt(
    runtime_model: _RecyclingRuntimeModel,
    *,
    timestep: float,
) -> float:
    base_dt = 25.0 if len(runtime_model.field_names) > 10 else 100.0
    return min(float(timestep), base_dt)


def _advance_recycling_1d_output_interval(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel,
    feedback_integrals: dict[str, float],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    output_timestep: float,
    suggested_dt: float,
    residual_tolerance: float,
    max_nonlinear_iterations: int,
) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    trial_dt = min(float(suggested_dt), float(output_timestep))
    minimum_dt = max(float(output_timestep) / 4096.0, 1.0)
    acceptance_residual = max(1.0e4 * residual_tolerance, 5.0e-3)

    while True:
        step_count = max(1, int(math.ceil(float(output_timestep) / trial_dt)))
        step_dt = float(output_timestep) / float(step_count)
        start_fields = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
        start_integrals = {name: float(feedback_integrals.get(name, 0.0)) for name in runtime_model.feedback_names}
        current_fields = start_fields
        current_integrals = start_integrals
        previous_fields: dict[str, np.ndarray] | None = None
        previous_integrals: dict[str, float] | None = None
        failed = False
        last_info: Recycling1DImplicitStepInfo | None = None

        for _ in range(step_count):
            if previous_fields is None or previous_integrals is None:
                next_fields, next_integrals, info = advance_recycling_1d_backward_euler_step(
                    config,
                    current_fields,
                    runtime_model=runtime_model,
                    feedback_integrals=current_integrals,
                    mesh=mesh,
                    metrics=metrics,
                    dataset_scalars=dataset_scalars,
                    timestep=step_dt,
                    solver_mode="sparse",
                    residual_tolerance=residual_tolerance,
                    max_nonlinear_iterations=max_nonlinear_iterations,
                )
            else:
                next_fields, next_integrals, info = advance_recycling_1d_bdf2_step(
                    config,
                    current_fields,
                    previous_fields,
                    runtime_model=runtime_model,
                    feedback_integrals=current_integrals,
                    previous_feedback_integrals=previous_integrals,
                    mesh=mesh,
                    metrics=metrics,
                    dataset_scalars=dataset_scalars,
                    timestep=step_dt,
                    solver_mode="sparse",
                    residual_tolerance=residual_tolerance,
                    max_nonlinear_iterations=max_nonlinear_iterations,
                )

            last_info = info
            if not np.isfinite(info.residual_inf_norm) or info.residual_inf_norm > acceptance_residual:
                failed = True
                break

            previous_fields = current_fields
            previous_integrals = current_integrals
            current_fields = next_fields
            current_integrals = next_integrals

        if not failed:
            return current_fields, current_integrals, step_dt
        if step_dt <= minimum_dt:
            raise RuntimeError(
                f"Recycling continuation interval failed at dt={step_dt:g}; residual={float('nan') if last_info is None else last_info.residual_inf_norm:g}"
            )
        trial_dt = 0.5 * step_dt


def advance_recycling_1d_backward_euler_step(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel | None = None,
    feedback_integrals: dict[str, float],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    timestep: float,
    solver_mode: str = "sparse",
    residual_tolerance: float = 1.0e-8,
    max_nonlinear_iterations: int = 20,
) -> tuple[dict[str, np.ndarray], dict[str, float], Recycling1DImplicitStepInfo]:
    runtime_model = runtime_model or _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
    )
    field_names = runtime_model.field_names
    packed_feedback_names: tuple[str, ...] = ()
    previous_feedback_errors = _current_feedback_errors(fields, controllers=runtime_model.controllers, mesh=mesh)
    packed_previous = _pack_recycling_active_state(
        fields,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
    )

    def residual(packed_state: np.ndarray) -> np.ndarray:
        state_fields, state_integrals = _unpack_recycling_active_state(
            packed_state,
            field_templates=fields,
            feedback_integrals=feedback_integrals,
            field_names=field_names,
            feedback_names=packed_feedback_names,
            mesh=mesh,
        )
        rhs = _compute_recycling_1d_packed_rhs(
            config,
            state_fields,
            sanitize_fields=False,
            feedback_integrals=feedback_integrals,
            feedback_previous_errors=previous_feedback_errors,
            feedback_timestep=timestep,
            field_names=field_names,
            feedback_names=(),
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            runtime_model=runtime_model,
        )
        return _build_recycling_mixed_be_residual(
            packed_state,
            packed_previous,
            rhs_fields=rhs,
            feedback_integrals=state_integrals,
            previous_feedback_integrals=feedback_integrals,
            current_feedback_errors=previous_feedback_errors,
            previous_feedback_errors=previous_feedback_errors,
            field_names=field_names,
            feedback_names=packed_feedback_names,
            timestep=timestep,
        )

    if solver_mode == "sparse":
        solved, info = solve_sparse_newton_system(
            residual,
            packed_previous,
            active_shape=(packed_previous.size,),
            sparsity=_build_recycling_residual_sparsity(
                active_shape=_recycling_active_shape(mesh),
                field_count=len(field_names),
                controller_count=len(packed_feedback_names),
            ),
            color_groups=_build_recycling_color_groups(
                active_shape=_recycling_active_shape(mesh),
                field_count=len(field_names),
                controller_count=len(packed_feedback_names),
            ),
            residual_tolerance=residual_tolerance,
            step_tolerance=1.0e-11,
            max_nonlinear_iterations=max_nonlinear_iterations,
            linear_restart=20,
            linear_maxiter=300,
            linear_rtol=1.0e-8,
            prefer_direct_linear_solve=True,
            jacobian_refresh_frequency=3 if len(field_names) > 10 else 1,
        )
    elif solver_mode == "matrix_free":
        solved, info = solve_matrix_free_newton_system(
            residual,
            packed_previous,
            active_shape=(packed_previous.size,),
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
    else:
        raise ValueError(f"Unsupported recycling implicit solver_mode={solver_mode!r}.")
    next_fields, next_integrals = _unpack_recycling_active_state(
        solved,
        field_templates=fields,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
    )
    sanitized_fields = _sanitize_recycling_fields(config, next_fields)
    sanitized_integrals = _advance_feedback_integrals(
        sanitized_fields,
        controllers=runtime_model.controllers,
        feedback_integrals=feedback_integrals,
        feedback_previous_errors=previous_feedback_errors,
        mesh=mesh,
        timestep=timestep,
    )
    return sanitized_fields, sanitized_integrals, _as_recycling_step_info(info)


def advance_recycling_1d_bdf2_step(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    previous_fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel | None = None,
    feedback_integrals: dict[str, float],
    previous_feedback_integrals: dict[str, float],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    timestep: float,
    solver_mode: str = "sparse",
    residual_tolerance: float = 1.0e-8,
    max_nonlinear_iterations: int = 20,
) -> tuple[dict[str, np.ndarray], dict[str, float], Recycling1DImplicitStepInfo]:
    runtime_model = runtime_model or _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
    )
    field_names = runtime_model.field_names
    packed_feedback_names: tuple[str, ...] = ()
    previous_feedback_errors = _current_feedback_errors(fields, controllers=runtime_model.controllers, mesh=mesh)
    packed_previous = _pack_recycling_active_state(
        fields,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
    )
    packed_previous_previous = _pack_recycling_active_state(
        previous_fields,
        feedback_integrals=previous_feedback_integrals,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
    )

    def residual(packed_state: np.ndarray) -> np.ndarray:
        state_fields, state_integrals = _unpack_recycling_active_state(
            packed_state,
            field_templates=fields,
            feedback_integrals=feedback_integrals,
            field_names=field_names,
            feedback_names=packed_feedback_names,
            mesh=mesh,
        )
        rhs = _compute_recycling_1d_packed_rhs(
            config,
            state_fields,
            sanitize_fields=False,
            feedback_integrals=feedback_integrals,
            feedback_previous_errors=previous_feedback_errors,
            feedback_timestep=timestep,
            field_names=field_names,
            feedback_names=(),
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            runtime_model=runtime_model,
        )
        return _build_recycling_mixed_bdf2_residual(
            packed_state,
            packed_previous,
            packed_previous_previous,
            rhs_fields=rhs,
            feedback_integrals=state_integrals,
            previous_feedback_integrals=feedback_integrals,
            previous_previous_feedback_integrals=previous_feedback_integrals,
            current_feedback_errors=previous_feedback_errors,
            previous_feedback_errors=previous_feedback_errors,
            field_names=field_names,
            feedback_names=packed_feedback_names,
            timestep=timestep,
        )

    if solver_mode == "sparse":
        solved, info = solve_sparse_newton_system(
            residual,
            packed_previous,
            active_shape=(packed_previous.size,),
            sparsity=_build_recycling_residual_sparsity(
                active_shape=_recycling_active_shape(mesh),
                field_count=len(field_names),
                controller_count=len(packed_feedback_names),
            ),
            color_groups=_build_recycling_color_groups(
                active_shape=_recycling_active_shape(mesh),
                field_count=len(field_names),
                controller_count=len(packed_feedback_names),
            ),
            residual_tolerance=residual_tolerance,
            step_tolerance=1.0e-11,
            max_nonlinear_iterations=max_nonlinear_iterations,
            linear_restart=20,
            linear_maxiter=300,
            linear_rtol=1.0e-8,
            prefer_direct_linear_solve=True,
            jacobian_refresh_frequency=3 if len(field_names) > 10 else 1,
        )
    elif solver_mode == "matrix_free":
        solved, info = solve_matrix_free_newton_system(
            residual,
            packed_previous,
            active_shape=(packed_previous.size,),
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
    else:
        raise ValueError(f"Unsupported recycling implicit solver_mode={solver_mode!r}.")
    next_fields, next_integrals = _unpack_recycling_active_state(
        solved,
        field_templates=fields,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
    )
    sanitized_fields = _sanitize_recycling_fields(config, next_fields)
    sanitized_integrals = _advance_feedback_integrals(
        sanitized_fields,
        controllers=runtime_model.controllers,
        feedback_integrals=feedback_integrals,
        feedback_previous_errors=previous_feedback_errors,
        mesh=mesh,
        timestep=timestep,
    )
    return sanitized_fields, sanitized_integrals, _as_recycling_step_info(info)


def _advance_recycling_1d_bdf_history(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel,
    feedback_integrals: dict[str, float],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    timestep: float,
    steps: int,
) -> Recycling1DHistoryResult:
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("BDF recycling stepping requires scipy.") from exc

    active_shape = _recycling_active_shape(mesh)
    sparsity = _build_recycling_residual_sparsity(
        active_shape=active_shape,
        field_count=len(field_names),
        controller_count=len(feedback_names),
    )
    color_groups = _build_recycling_color_groups(
        active_shape=active_shape,
        field_count=len(field_names),
        controller_count=len(feedback_names),
    )
    current_fields = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
    current_integrals = {name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names}
    total_time = float(timestep) * float(steps)
    relative_tolerance = float(config.parsed("solver", "rtol")) if config.has_option("solver", "rtol") else 1.0e-6
    absolute_tolerance = float(config.parsed("solver", "atol")) if config.has_option("solver", "atol") else 1.0e-9
    output_times = np.linspace(0.0, total_time, steps + 1, dtype=np.float64)
    y0 = _pack_recycling_active_state(
        current_fields,
        feedback_integrals=current_integrals,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
    )

    def rhs(_time: float, packed_state: np.ndarray) -> np.ndarray:
        state_fields, state_integrals = _unpack_recycling_active_state(
            packed_state,
            field_templates=current_fields,
            feedback_integrals=current_integrals,
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
        )
        return _compute_recycling_1d_packed_rhs(
            config,
            state_fields,
            sanitize_fields=True,
            feedback_integrals=state_integrals,
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            runtime_model=runtime_model,
        )

    def jacobian(_time: float, packed_state: np.ndarray):
        rhs_value = rhs(_time, packed_state)
        return build_sparse_difference_quotient_jacobian(
            lambda state: rhs(_time, state),
            packed_state,
            base_residual=rhs_value,
            sparsity=sparsity,
            color_groups=color_groups,
        )

    solution = solve_ivp(
        rhs,
        (0.0, total_time),
        y0,
        method="BDF",
        jac=jacobian,
        t_eval=output_times,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        jac_sparsity=sparsity,
        max_step=min(float(timestep), 25.0),
    )
    if not solution.success:
        raise RuntimeError(f"Adaptive recycling BDF step failed: {solution.message}")

    variable_history = {name: [] for name in field_names}
    feedback_history = {name: [] for name in feedback_names}
    for column in range(solution.y.shape[1]):
        sample_fields, sample_integrals = _unpack_recycling_active_state(
            solution.y[:, column],
            field_templates=current_fields,
            feedback_integrals=current_integrals,
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
        )
        sample_fields = _sanitize_recycling_fields(config, sample_fields)
        sample_integrals = _sanitize_feedback_integrals(sample_integrals, controllers=runtime_model.controllers)
        for name in field_names:
            variable_history[name].append(np.asarray(sample_fields[name], dtype=np.float64))
        for name in feedback_names:
            feedback_history[name].append(np.asarray(sample_integrals[name], dtype=np.float64))

    return Recycling1DHistoryResult(
        variable_history={name: np.stack(history, axis=0) for name, history in variable_history.items()},
        feedback_integral_history={name: np.stack(history, axis=0) for name, history in feedback_history.items()},
    )


def _compute_recycling_1d_packed_rhs(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel | None = None,
    sanitize_fields: bool = True,
    feedback_integrals: dict[str, float],
    feedback_previous_errors: dict[str, float] | None = None,
    feedback_timestep: float | None = None,
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
) -> np.ndarray:
    runtime_model = runtime_model or _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
    )
    sanitized_fields = _sanitize_recycling_fields(config, fields) if sanitize_fields else {
        name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()
    }
    species = _override_species_fields(runtime_model.species_templates, fields=sanitized_fields, mesh=mesh)
    result = _compute_recycling_1d_rhs_from_species(
        config,
        species=species,
        controllers=runtime_model.controllers,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        feedback_integrals=feedback_integrals,
        feedback_previous_errors=feedback_previous_errors,
        feedback_timestep=feedback_timestep,
        explicit_pressure_sources=runtime_model.explicit_pressure_sources,
    )
    active_slices = _recycling_active_domain_slices(mesh)
    pieces = [
        np.asarray(result.variables[f"ddt({name})"][0][active_slices], dtype=np.float64).ravel()
        for name in field_names
    ]
    pieces.extend(np.asarray(result.feedback_integral_rhs.get(name, 0.0), dtype=np.float64).reshape(1) for name in feedback_names)
    return np.concatenate(pieces) if pieces else np.array([], dtype=np.float64)


def _pack_recycling_active_state(
    fields: dict[str, np.ndarray],
    *,
    feedback_integrals: dict[str, float],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
) -> np.ndarray:
    field_block = pack_active_fields(
        tuple(np.asarray(fields[name], dtype=np.float64) for name in field_names),
        active_slices=_recycling_active_domain_slices(mesh),
    )
    if not feedback_names:
        return field_block
    scalar_block = np.asarray([feedback_integrals.get(name, 0.0) for name in feedback_names], dtype=np.float64)
    return np.concatenate([field_block, scalar_block])


def _unpack_recycling_active_state(
    packed: np.ndarray,
    *,
    field_templates: dict[str, np.ndarray],
    feedback_integrals: dict[str, float],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    packed_array = np.asarray(packed, dtype=np.float64)
    field_size = _recycling_active_field_size(mesh) * len(field_names)
    field_block = packed_array[:field_size]
    scalar_block = packed_array[field_size:]
    unpacked_fields = unpack_active_fields(
        field_block,
        templates=tuple(np.asarray(field_templates[name], dtype=np.float64) for name in field_names),
        active_slices=_recycling_active_domain_slices(mesh),
    )
    restored_fields = {name: np.asarray(value, dtype=np.float64) for name, value in zip(field_names, unpacked_fields, strict=True)}
    restored_integrals = {name: float(value) for name, value in feedback_integrals.items()}
    for index, name in enumerate(feedback_names):
        restored_integrals[name] = float(scalar_block[index])
    return restored_fields, restored_integrals


def _sanitize_recycling_fields(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    sanitized = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
    resolver = NumericResolver(config)
    ion_density_names = sorted(name for name in sanitized if name.startswith("N") and not name.startswith("NV") and name != "Ne")
    electron_density = np.zeros_like(sanitized[ion_density_names[0]], dtype=np.float64) if ion_density_names else None
    if electron_density is not None:
        for density_name in ion_density_names:
            species_name = density_name[1:]
            if species_name == "e":
                continue
            charge = float(resolver.resolve(species_name, "charge")) if config.has_option(species_name, "charge") else 0.0
            if charge > 0.0:
                electron_density = electron_density + charge * np.maximum(sanitized[density_name], 1.0e-12)
    for name in list(sanitized):
        if name.startswith("N") and not name.startswith("NV") and name != "Ne":
            species_name = name[1:]
            density_floor = float(resolver.resolve(species_name, "density_floor")) if config.has_option(species_name, "density_floor") else 1.0e-7
            sanitized[name] = np.maximum(sanitized[name], density_floor)
        elif name.startswith("P"):
            species_name = name[1:]
            if config.has_option(species_name, "temperature_floor"):
                temperature_floor = float(resolver.resolve(species_name, "temperature_floor"))
            elif species_name == "e":
                temperature_floor = 0.1
            else:
                charge = float(resolver.resolve(species_name, "charge")) if config.has_option(species_name, "charge") else 0.0
                temperature_floor = 0.1 if charge != 0.0 else 0.0
            if species_name == "e" and electron_density is not None:
                sanitized[name] = np.maximum(sanitized[name], temperature_floor * np.maximum(electron_density, 1.0e-7))
            else:
                density_name = f"N{species_name}"
                density = sanitized.get(density_name)
                floor_density = np.maximum(density, 1.0e-7) if density is not None else 1.0e-7
                sanitized[name] = np.maximum(sanitized[name], temperature_floor * floor_density)
    return sanitized


def _predict_recycling_fields_from_rhs(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel,
    feedback_integrals: dict[str, float],
    field_names: tuple[str, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    timestep: float,
) -> dict[str, np.ndarray]:
    sanitized_fields = _sanitize_recycling_fields(config, fields)
    species = _override_species_fields(runtime_model.species_templates, fields=sanitized_fields, mesh=mesh)
    result = _compute_recycling_1d_rhs_from_species(
        config,
        species=species,
        controllers=runtime_model.controllers,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        feedback_integrals=feedback_integrals,
        explicit_pressure_sources=runtime_model.explicit_pressure_sources,
    )
    predicted = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in sanitized_fields.items()}
    for name in field_names:
        rhs_name = f"ddt({name})"
        if rhs_name not in result.variables:
            continue
        predicted[name] = np.asarray(
            sanitized_fields[name] + float(timestep) * np.asarray(result.variables[rhs_name][0], dtype=np.float64),
            dtype=np.float64,
        )
    return _sanitize_recycling_fields(config, predicted)


def _current_feedback_errors(
    fields: dict[str, np.ndarray],
    *,
    controllers: dict[str, _DensityFeedbackController],
    mesh: StructuredMesh,
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for name, controller in controllers.items():
        density_name = f"N{name}"
        if density_name not in fields:
            continue
        upstream_density = float(np.asarray(fields[density_name], dtype=np.float64)[mesh.xstart, mesh.ystart, 0])
        errors[name] = controller.density_upstream - upstream_density
    return errors


def _advance_feedback_integrals(
    fields: dict[str, np.ndarray],
    *,
    controllers: dict[str, _DensityFeedbackController],
    feedback_integrals: dict[str, float],
    feedback_previous_errors: dict[str, float],
    mesh: StructuredMesh,
    timestep: float,
) -> dict[str, float]:
    updated = {name: float(value) for name, value in feedback_integrals.items()}
    current_errors = _current_feedback_errors(fields, controllers=controllers, mesh=mesh)
    for name, controller in controllers.items():
        current_error = float(current_errors.get(name, 0.0))
        previous_error = float(feedback_previous_errors.get(name, current_error))
        integral = float(feedback_integrals.get(name, 0.0)) + float(timestep) * 0.5 * (current_error + previous_error)
        if controller.density_integral_positive and integral < 0.0:
            integral = 0.0
        updated[name] = integral
    return updated


def _advance_feedback_integrals_from_predictor(
    *,
    controllers: dict[str, _DensityFeedbackController],
    feedback_integrals: dict[str, float],
    feedback_previous_errors: dict[str, float],
    predictor_feedback_errors: dict[str, float],
    timestep: float,
) -> dict[str, float]:
    updated = {name: float(value) for name, value in feedback_integrals.items()}
    for name, controller in controllers.items():
        previous_error = float(feedback_previous_errors.get(name, 0.0))
        predictor_error = float(predictor_feedback_errors.get(name, previous_error))
        integral = float(feedback_integrals.get(name, 0.0)) + float(timestep) * 0.5 * (predictor_error + previous_error)
        if controller.density_integral_positive and integral < 0.0:
            integral = 0.0
        updated[name] = integral
    return updated


def _sanitize_feedback_integrals(
    feedback_integrals: dict[str, float],
    *,
    controllers: dict[str, _DensityFeedbackController],
) -> dict[str, float]:
    sanitized = {name: float(value) for name, value in feedback_integrals.items()}
    for name, controller in controllers.items():
        integral = float(sanitized.get(name, 0.0))
        if controller.density_integral_positive and integral < 0.0:
            integral = 0.0
        sanitized[name] = integral
    return sanitized


def _feedback_integral_vector(
    feedback_integrals: dict[str, float],
    *,
    feedback_names: tuple[str, ...],
) -> np.ndarray:
    return np.asarray([float(feedback_integrals.get(name, 0.0)) for name in feedback_names], dtype=np.float64)


def _feedback_error_vector(
    feedback_errors: dict[str, float],
    *,
    feedback_names: tuple[str, ...],
) -> np.ndarray:
    return np.asarray([float(feedback_errors.get(name, 0.0)) for name in feedback_names], dtype=np.float64)


def _build_recycling_mixed_be_residual(
    packed_state: np.ndarray,
    previous_packed_state: np.ndarray,
    *,
    rhs_fields: np.ndarray,
    feedback_integrals: dict[str, float],
    previous_feedback_integrals: dict[str, float],
    current_feedback_errors: dict[str, float],
    previous_feedback_errors: dict[str, float],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    timestep: float,
) -> np.ndarray:
    field_size = np.asarray(previous_packed_state, dtype=np.float64).size - len(feedback_names)
    field_block = backward_euler_residual(
        np.asarray(packed_state, dtype=np.float64)[:field_size],
        np.asarray(previous_packed_state, dtype=np.float64)[:field_size],
        rhs_fields,
        timestep=timestep,
    )
    controller_block = backward_euler_residual(
        _feedback_integral_vector(feedback_integrals, feedback_names=feedback_names),
        _feedback_integral_vector(previous_feedback_integrals, feedback_names=feedback_names),
        _feedback_error_vector(current_feedback_errors, feedback_names=feedback_names),
        timestep=timestep,
    )
    return np.concatenate([field_block, controller_block]) if feedback_names else field_block


def _build_recycling_mixed_bdf2_residual(
    packed_state: np.ndarray,
    previous_packed_state: np.ndarray,
    previous_previous_packed_state: np.ndarray,
    *,
    rhs_fields: np.ndarray,
    feedback_integrals: dict[str, float],
    previous_feedback_integrals: dict[str, float],
    previous_previous_feedback_integrals: dict[str, float],
    current_feedback_errors: dict[str, float],
    previous_feedback_errors: dict[str, float],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    timestep: float,
) -> np.ndarray:
    field_size = np.asarray(previous_packed_state, dtype=np.float64).size - len(feedback_names)
    field_block = bdf2_residual(
        np.asarray(packed_state, dtype=np.float64)[:field_size],
        np.asarray(previous_packed_state, dtype=np.float64)[:field_size],
        np.asarray(previous_previous_packed_state, dtype=np.float64)[:field_size],
        rhs_fields,
        timestep=timestep,
    )
    controller_block = bdf2_residual(
        _feedback_integral_vector(feedback_integrals, feedback_names=feedback_names),
        _feedback_integral_vector(previous_feedback_integrals, feedback_names=feedback_names),
        _feedback_integral_vector(previous_previous_feedback_integrals, feedback_names=feedback_names),
        _feedback_error_vector(current_feedback_errors, feedback_names=feedback_names),
        timestep=timestep,
    )
    return np.concatenate([field_block, controller_block]) if feedback_names else field_block


def _recycling_evolving_variable_names(species: dict[str, OpenFieldSpecies]) -> tuple[str, ...]:
    names: list[str] = []
    for name, sp in species.items():
        if name == "e":
            names.append("Pe")
            continue
        names.extend((sp.density_name, sp.pressure_name, sp.momentum_name))
    return tuple(names)


def _recycling_field_templates(
    species: dict[str, OpenFieldSpecies],
    *,
    field_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for name in field_names:
        if name == "Pe":
            templates[name] = np.asarray(species["e"].pressure, dtype=np.float64)
            continue
        species_name = name[1:] if name.startswith("N") else name[1:]
        if name.startswith("NV"):
            species_name = name[2:]
        sp = species[species_name]
        if name.startswith("N") and not name.startswith("NV"):
            templates[name] = np.asarray(sp.density, dtype=np.float64)
        elif name.startswith("P"):
            templates[name] = np.asarray(sp.pressure, dtype=np.float64)
        else:
            templates[name] = np.asarray(sp.momentum, dtype=np.float64)
    return templates


def _recycling_active_domain_slices(mesh: StructuredMesh) -> tuple[slice, slice, slice]:
    return (
        slice(mesh.xstart, mesh.xend + 1),
        slice(mesh.ystart, mesh.yend + 1),
        slice(None),
    )


def _recycling_active_field_size(mesh: StructuredMesh) -> int:
    return int(np.prod(_recycling_active_shape(mesh)))


def _recycling_active_shape(mesh: StructuredMesh) -> tuple[int, int, int]:
    active_slices = _recycling_active_domain_slices(mesh)
    return tuple(
        len(range(*active_slice.indices(axis_extent)))
        for active_slice, axis_extent in zip(active_slices, (mesh.nx, mesh.local_ny, mesh.nz), strict=True)
    )


@lru_cache(maxsize=None)
def _build_recycling_residual_sparsity(
    *,
    active_shape: tuple[int, int, int],
    field_count: int,
    controller_count: int,
):
    try:
        from scipy.sparse import lil_matrix
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Sparse recycling stepping requires scipy.") from exc

    field_sparsity = build_locality_sparsity(
        active_shape,
        field_count=field_count,
        radii=(0, 2, 0),
        periodic_axes=(),
    )
    field_size = field_sparsity.shape[0]
    total_size = field_size + controller_count
    sparsity = lil_matrix((total_size, total_size), dtype=bool)
    sparsity[:field_size, :field_size] = field_sparsity
    if controller_count > 0:
        sparsity[:field_size, field_size:total_size] = True
        sparsity[field_size:total_size, :field_size] = True
        sparsity[field_size:total_size, field_size:total_size] = True
    return sparsity.tocsr()


@lru_cache(maxsize=None)
def _build_recycling_color_groups(
    *,
    active_shape: tuple[int, int, int],
    field_count: int,
    controller_count: int,
) -> tuple[tuple[int, ...], ...]:
    field_groups = build_modulo_color_groups(
        active_shape,
        field_count=field_count,
        color_periods=(1, min(5, active_shape[1]), 1),
    )
    field_size = field_count * int(np.prod(active_shape))
    controller_groups = tuple((field_size + index,) for index in range(controller_count))
    return field_groups + controller_groups


def _as_recycling_step_info(info: ImplicitStepInfo) -> Recycling1DImplicitStepInfo:
    return Recycling1DImplicitStepInfo(
        residual_inf_norm=info.residual_inf_norm,
        active_size=int(np.prod(info.active_shape)),
        nonlinear_iterations=info.nonlinear_iterations,
        linear_iterations=info.linear_iterations,
    )
    pressure_sources = explicit_pressure_sources or {}
