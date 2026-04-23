from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
import re
from typing import Any, Callable, Mapping

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
from .mesh import StructuredMesh
from .metrics import StructuredMetrics
from .neutral_mixed import (
    _div_a_grad_perp_flows,
    _div_par_fvv_open,
    _div_par_k_grad_par_open,
    _div_par_mod_open,
    _grad_par_open,
)
from .open_field import (
    apply_noflow_flow_guards,
    apply_noflow_scalar_guards,
    apply_parallel_electric_force,
    compute_electron_force_balance,
    compute_target_recycling_sources,
    limit_free,
)
from .recycling_boundaries import (
    apply_neutral_target_density_guards as _apply_neutral_target_density_guards,
    apply_open_field_dirichlet_scalar_guards as _apply_open_field_dirichlet_scalar_guards,
    apply_open_field_neumann_scalar_guards as _apply_open_field_neumann_scalar_guards,
)
from .recycling_atomic import (
    AMJUEL_FILENAMES as _AMJUEL_FILENAMES,
    OPENADAS_FILENAMES as _OPENADAS_FILENAMES,
    amjuel_energy_loss as _amjuel_energy_loss,
    amjuel_reaction_rate as _amjuel_reaction_rate,
    charge_exchange_rate_multiplier as _charge_exchange_rate_multiplier,
    eval_amjuel_fit as _eval_amjuel_fit,
    eval_openadas_rate as _eval_openadas_rate,
    hydrogen_cx_sigmav as _hydrogen_cx_sigmav,
    load_amjuel_rate as _load_amjuel_rate,
    load_openadas_rate as _load_openadas_rate,
    openadas_energy_loss as _openadas_energy_loss,
    openadas_reaction_rate as _openadas_reaction_rate,
)
from .recycling_anomalous_diffusion import (
    AnomalousDiffusionTerms as _AnomalousDiffusionTerms,
    apply_anomalous_diffusion as _apply_anomalous_diffusion,
    div_a_grad_perp_upwind_flows_nz1 as _div_a_grad_perp_upwind_flows_nz1,
)
from .recycling_collisions import (
    IonParallelViscosityInputs as _IonParallelViscosityInputs,
    compute_collision_frequencies as _compute_collision_frequencies,
    electron_density as _electron_density,
    ion_parallel_viscosity_inputs as _ion_parallel_viscosity_inputs,
    prepared_electron_density as _prepared_electron_density,
)
from .recycling_collision_closure import (
    CollisionClosureTerms as _CollisionClosureTerms,
    apply_collision_closure as _apply_collision_closure,
    conduction_collision_time as _conduction_collision_time,
    conduction_kappa_coefficient as _conduction_kappa_coefficient,
    div_par_parallel_ion_viscosity_open as _div_par_parallel_ion_viscosity_open,
    ion_thermal_force_pair as _ion_thermal_force_pair,
    momentum_coefficient as _momentum_coefficient,
    parallel_ion_viscous_stress_open as _parallel_ion_viscous_stress_open,
    thermal_force_enabled as _thermal_force_enabled,
)
from .recycling_neutral_diffusion import (
    NeutralParallelDiffusionTerms as _NeutralParallelDiffusionTerms,
    apply_neutral_parallel_diffusion as _apply_neutral_parallel_diffusion,
    configured_component_names as _configured_component_names,
)
from .recycling_reactions import (
    ReactionTerms as _ReactionTerms,
    accumulate_terms as _accumulate_terms,
    amjuel_ionisation as _amjuel_ionisation,
    amjuel_recombination as _amjuel_recombination,
    charge_exchange as _charge_exchange,
    charge_exchange_collision_rates as _charge_exchange_collision_rates,
    is_charge_exchange_reaction as _is_charge_exchange_reaction,
    neutral_charge_exchange_collision_rates as _neutral_charge_exchange_collision_rates,
    neutral_ionisation_collision_rates as _neutral_ionisation_collision_rates,
    reaction_sources as _reaction_sources,
)
from .recycling_rhs_terms import (
    ElectronPressureRhsTerms,
    IonRhsTerms,
    assemble_electron_pressure_rhs_terms as _assemble_electron_pressure_rhs_terms,
    assemble_ion_rhs_terms as _assemble_ion_rhs_terms,
)
from .recycling_setup import (
    DensityFeedbackController as _DensityFeedbackController,
    OpenFieldSpecies,
    RecyclingRuntimeModel as _RecyclingRuntimeModel,
    build_recycling_runtime_model as _build_recycling_runtime_model,
    evaluate_field_option as _evaluate_field_option,
    evaluate_field_value as _evaluate_field_value,
    evaluate_option_field as _evaluate_option_field,
    explicit_pressure_source as _explicit_pressure_source,
    initialize_species as _initialize_species,
    load_density_feedback_controllers as _load_density_feedback_controllers,
    load_explicit_pressure_sources as _load_explicit_pressure_sources,
    override_species_fields as _override_species_fields,
    resolve_species_numeric_option as _resolve_species_numeric_option,
    try_literal_reference as _try_literal_reference,
)
from .recycling_state import (
    PreparedSpeciesState as _PreparedSpeciesState,
    axisymmetric_profile as _axisymmetric_profile,
    merge_target_guard_cells as _merge_target_guard_cells,
    prepare_species_state as _prepare_species_state,
    raw_species_velocity as _raw_species_velocity,
    safe_temperature as _safe_temperature,
    soft_floor as _soft_floor,
)
from .recycling_sanitize import sanitize_recycling_fields as _sanitize_recycling_fields
from .recycling_targets import (
    RecyclingTerms as _RecyclingTerms,
    electron_zero_current_velocity as _electron_zero_current_velocity,
    grad_par_electron_force_balance_open as _grad_par_electron_force_balance_open,
    target_recycling_sources as _target_recycling_sources,
)
from .recycling_fields import (
    build_recycling_state_fields as _build_recycling_state_fields,
    recycling_evolving_variable_names as _recycling_evolving_variable_names,
    recycling_field_templates as _recycling_field_templates,
)
from .recycling_feedback import (
    advance_feedback_integrals as _advance_feedback_integrals,
    advance_feedback_integrals_from_predictor as _advance_feedback_integrals_from_predictor,
    current_feedback_errors as _current_feedback_errors,
    feedback_error_vector as _feedback_error_vector,
    feedback_integral_vector as _feedback_integral_vector,
    sanitize_feedback_integrals as _sanitize_feedback_integrals,
)
from .recycling_layout import (
    RecyclingPackedStateLayout as _RecyclingPackedStateLayout,
    build_recycling_packed_state_layout as _build_recycling_packed_state_layout,
    pack_recycling_active_state as _pack_recycling_active_state,
    recycling_active_domain_slices as _recycling_active_domain_slices,
    recycling_active_field_size as _recycling_active_field_size,
    recycling_active_shape as _recycling_active_shape,
    unpack_recycling_active_state as _unpack_recycling_active_state,
)

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


RecyclingProgressCallback = Callable[[Mapping[str, Any]], None]

@dataclass(frozen=True)
class _SimpleSheathSettings:
    gamma_e: float
    gamma_i: float
    secondary_electron_coef: float
    sheath_ion_polytropic: float
    lower_y: bool
    upper_y: bool
    no_flow: bool
    density_boundary_mode: float
    pressure_boundary_mode: float
    temperature_boundary_mode: float
    wall_potential: np.ndarray


@dataclass(frozen=True)
class _FullSheathSettings:
    secondary_electron_coef: float
    sin_alpha: np.ndarray
    lower_y: bool
    upper_y: bool
    wall_potential: np.ndarray
    floor_potential: bool

def compute_recycling_1d_rhs(
    config: BoutConfig,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    field_overrides: dict[str, np.ndarray] | None = None,
    field_template_overrides: dict[str, np.ndarray] | None = None,
    feedback_integrals: dict[str, float] | None = None,
    apply_sheath_boundaries: bool = True,
    preserve_dump_target_state: bool = False,
    preserve_dump_ion_target_state_only: bool = False,
    density_source_overrides: dict[str, np.ndarray] | None = None,
    pressure_source_overrides: dict[str, np.ndarray] | None = None,
    momentum_source_overrides: dict[str, np.ndarray] | None = None,
) -> Recycling1DRhsResult:
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=field_overrides,
        field_template_overrides=field_template_overrides,
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
        preserve_dump_target_state=preserve_dump_target_state,
        preserve_dump_ion_target_state_only=preserve_dump_ion_target_state_only,
        density_source_overrides=density_source_overrides,
        pressure_source_overrides=pressure_source_overrides,
        momentum_source_overrides=momentum_source_overrides,
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
    preserve_dump_target_state: bool = False,
    preserve_dump_ion_target_state_only: bool = False,
    density_source_overrides: dict[str, np.ndarray] | None = None,
    pressure_source_overrides: dict[str, np.ndarray] | None = None,
    momentum_source_overrides: dict[str, np.ndarray] | None = None,
) -> Recycling1DRhsResult:
    pressure_sources = explicit_pressure_sources or {}
    if pressure_source_overrides:
        pressure_sources = {
            **pressure_sources,
            **{name: np.asarray(value, dtype=np.float64) for name, value in pressure_source_overrides.items()},
        }
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

    anomalous_terms = _apply_anomalous_diffusion(
        config,
        species=species,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
    )
    for name, value in anomalous_terms.density_source.items():
        density_source[name] = density_source[name] + value
    for name, value in anomalous_terms.energy_source.items():
        energy_source[name] = energy_source[name] + value
    for name, value in anomalous_terms.momentum_source.items():
        momentum_source[name] = momentum_source[name] + value
    diagnostics.update(anomalous_terms.diagnostics)

    prepared, ion_boundary, electron_boundary = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        apply_sheath_boundaries=apply_sheath_boundaries,
        preserve_dump_target_state=preserve_dump_target_state,
        preserve_dump_ion_target_state_only=preserve_dump_ion_target_state_only,
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

    simple_sheath_settings = _load_simple_sheath_settings(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
    )
    recycling_terms = _target_recycling_sources(
        ions=ions,
        prepared=prepared,
        neutrals=neutrals,
        ion_velocity=ion_velocity,
        mesh=mesh,
        metrics=metrics,
        gamma_i=0.0 if simple_sheath_settings is None else simple_sheath_settings.gamma_i,
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

    if density_source_overrides:
        for name, value in density_source_overrides.items():
            if name in density_source:
                density_source[name] = np.asarray(value, dtype=np.float64)
    if momentum_source_overrides:
        for name, value in momentum_source_overrides.items():
            if name in momentum_source:
                momentum_source[name] = np.asarray(value, dtype=np.float64)

    electron_force_density = -_grad_par_electron_force_balance_open(
        electron_boundary.pressure,
        mesh=mesh,
        metrics=metrics,
    )
    electron_force_density = electron_force_density + momentum_source["e"]
    electron_parallel_density = np.maximum(
        np.asarray(electron_boundary.density, dtype=np.float64),
        1.0e-5,
    )
    electron_epar = electron_force_density / electron_parallel_density
    for ion in ions:
        momentum_source[ion.name] = momentum_source[ion.name] + np.asarray(
            apply_parallel_electric_force(
                prepared[ion.name].density,
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
        ion_rhs_terms = _assemble_ion_rhs_terms(
            density_source=np.asarray(density_source[ion.name], dtype=np.float64),
            explicit_pressure_source=np.asarray(
                pressure_sources.get(
                    ion.name,
                    _explicit_pressure_source(config, ion.name, mesh=mesh, dataset_scalars=dataset_scalars),
                ),
                dtype=np.float64,
            ),
            momentum_source=np.asarray(momentum_source[ion.name], dtype=np.float64),
            atomic_mass=ion.atomic_mass,
            density_floor=ion.density_floor,
            ion_state=ion_state,
            ion_velocity=np.asarray(ion_velocity[ion.name], dtype=np.float64),
            fastest_wave=np.asarray(fastest_wave, dtype=np.float64),
            mesh=mesh,
            metrics=metrics,
            energy_source=np.asarray(energy_source[ion.name], dtype=np.float64),
        )

        variables[ion.density_name] = ion_state.density[None, ...]
        variables[ion.pressure_name] = ion_state.pressure[None, ...]
        variables[ion.momentum_name] = ion_state.momentum[None, ...]
        variables[f"SNV{ion.name}"] = momentum_source[ion.name][None, ...]
        variables[f"ddt({ion.density_name})"] = ion_rhs_terms.density_total[None, ...]
        variables[f"ddt({ion.pressure_name})"] = ion_rhs_terms.pressure_total[None, ...]
        variables[f"ddt({ion.momentum_name})"] = ion_rhs_terms.momentum_total[None, ...]

    electron_velocity = np.asarray(electron_boundary.velocity, dtype=np.float64)
    electron_fastest_wave = np.sqrt(np.maximum(prepared["e"].temperature, 0.0) / electron.atomic_mass)
    electron_pressure_rhs_terms = _assemble_electron_pressure_rhs_terms(
        explicit_pressure_source=np.asarray(
            pressure_sources.get(
                "e",
                _explicit_pressure_source(config, "e", mesh=mesh, dataset_scalars=dataset_scalars),
            ),
            dtype=np.float64,
        ),
        electron_pressure=electron_boundary.pressure,
        electron_velocity=electron_velocity,
        electron_fastest_wave=electron_fastest_wave,
        electron_energy_source=energy_source["e"],
        mesh=mesh,
        metrics=metrics,
    )
    variables[electron.pressure_name] = electron_boundary.pressure[None, ...]
    variables[f"ddt({electron.pressure_name})"] = electron_pressure_rhs_terms.total[None, ...]

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
        pressure_rhs = np.asarray(
            pressure_sources.get(
                neutral.name,
                _explicit_pressure_source(config, neutral.name, mesh=mesh, dataset_scalars=dataset_scalars),
            ),
            dtype=np.float64,
        )
        pressure_rhs = pressure_rhs - (5.0 / 3.0) * _div_par_mod_open(
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


def _load_simple_sheath_settings(
    config: BoutConfig,
    *,
    mesh: StructuredMesh,
    dataset_scalars: dict[str, float],
) -> _SimpleSheathSettings | None:
    if not config.has_section("sheath_boundary_simple"):
        return None
    resolver = NumericResolver(config)
    tnorm = float(dataset_scalars.get("Tnorm", 1.0))
    wall_potential = (
        _evaluate_option_field(config, "sheath_boundary_simple", "wall_potential", mesh=mesh) / tnorm
        if config.has_option("sheath_boundary_simple", "wall_potential")
        else np.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    )
    return _SimpleSheathSettings(
        gamma_e=float(resolver.resolve("sheath_boundary_simple", "gamma_e"))
        if config.has_option("sheath_boundary_simple", "gamma_e")
        else 3.5,
        gamma_i=float(resolver.resolve("sheath_boundary_simple", "gamma_i"))
        if config.has_option("sheath_boundary_simple", "gamma_i")
        else 3.5,
        secondary_electron_coef=float(resolver.resolve("sheath_boundary_simple", "secondary_electron_coef"))
        if config.has_option("sheath_boundary_simple", "secondary_electron_coef")
        else 0.0,
        sheath_ion_polytropic=float(resolver.resolve("sheath_boundary_simple", "sheath_ion_polytropic"))
        if config.has_option("sheath_boundary_simple", "sheath_ion_polytropic")
        else 1.0,
        lower_y=bool(config.parsed("sheath_boundary_simple", "lower_y"))
        if config.has_option("sheath_boundary_simple", "lower_y")
        else True,
        upper_y=bool(config.parsed("sheath_boundary_simple", "upper_y"))
        if config.has_option("sheath_boundary_simple", "upper_y")
        else True,
        no_flow=bool(config.parsed("sheath_boundary_simple", "no_flow"))
        if config.has_option("sheath_boundary_simple", "no_flow")
        else False,
        density_boundary_mode=float(resolver.resolve("sheath_boundary_simple", "density_boundary_mode"))
        if config.has_option("sheath_boundary_simple", "density_boundary_mode")
        else 1.0,
        pressure_boundary_mode=float(resolver.resolve("sheath_boundary_simple", "pressure_boundary_mode"))
        if config.has_option("sheath_boundary_simple", "pressure_boundary_mode")
        else 1.0,
        temperature_boundary_mode=float(resolver.resolve("sheath_boundary_simple", "temperature_boundary_mode"))
        if config.has_option("sheath_boundary_simple", "temperature_boundary_mode")
        else 1.0,
        wall_potential=np.asarray(wall_potential, dtype=np.float64),
    )


def _load_full_sheath_settings(
    config: BoutConfig,
    *,
    mesh: StructuredMesh,
    dataset_scalars: dict[str, float],
) -> _FullSheathSettings:
    resolver = NumericResolver(config)
    section = "sheath_boundary"
    tnorm = float(dataset_scalars.get("Tnorm", 1.0))
    wall_potential = (
        _evaluate_option_field(config, section, "wall_potential", mesh=mesh) / tnorm
        if config.has_section(section) and config.has_option(section, "wall_potential")
        else np.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    )
    sin_alpha = (
        _evaluate_option_field(config, section, "sin_alpha", mesh=mesh)
        if config.has_section(section) and config.has_option(section, "sin_alpha")
        else np.ones((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    )
    return _FullSheathSettings(
        secondary_electron_coef=float(resolver.resolve(section, "secondary_electron_coef"))
        if config.has_section(section) and config.has_option(section, "secondary_electron_coef")
        else 0.0,
        sin_alpha=np.asarray(sin_alpha, dtype=np.float64),
        lower_y=bool(config.parsed(section, "lower_y"))
        if config.has_section(section) and config.has_option(section, "lower_y")
        else True,
        upper_y=bool(config.parsed(section, "upper_y"))
        if config.has_section(section) and config.has_option(section, "upper_y")
        else True,
        wall_potential=np.asarray(wall_potential, dtype=np.float64),
        floor_potential=bool(config.parsed(section, "floor_potential"))
        if config.has_section(section) and config.has_option(section, "floor_potential")
        else True,
    )


@dataclass(frozen=True)
class _DensityFeedbackTerms:
    density_source: dict[str, np.ndarray]
    energy_source: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]
    feedback_integral_rhs: dict[str, float]

def _prepare_open_field_states(
    species: dict[str, OpenFieldSpecies],
    *,
    config: BoutConfig,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    apply_sheath_boundaries: bool = True,
    preserve_dump_target_state: bool = False,
    preserve_dump_ion_target_state_only: bool = False,
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
        simple_sheath_settings = _load_simple_sheath_settings(
            config,
            mesh=mesh,
            dataset_scalars=dataset_scalars,
        )
        full_sheath_settings = _load_full_sheath_settings(
            config,
            mesh=mesh,
            dataset_scalars=dataset_scalars,
        )
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
            simple_settings=simple_sheath_settings,
            full_settings=full_sheath_settings,
        )
        if preserve_dump_target_state and not preserve_dump_ion_target_state_only:
            electron_boundary = _ElectronBoundaryResult(
                density=_merge_target_guard_cells(prepared["e"].density, electron_boundary.density, mesh=mesh),
                temperature=_merge_target_guard_cells(prepared["e"].temperature, electron_boundary.temperature, mesh=mesh),
                pressure=_merge_target_guard_cells(prepared["e"].pressure, electron_boundary.pressure, mesh=mesh),
                velocity=_merge_target_guard_cells(electron_velocity, electron_boundary.velocity, mesh=mesh),
                momentum=_merge_target_guard_cells(
                    species["e"].atomic_mass * prepared["e"].density * electron_velocity,
                    electron_boundary.momentum,
                    mesh=mesh,
                ),
                energy_source=np.asarray(electron_boundary.energy_source, dtype=np.float64),
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
        ion_boundary_state = _apply_ion_sheath_boundary(
            ions,
            electron_pressure=prepared["e"].pressure,
            electron_density=prepared["e"].density,
            electron_density_floor=species["e"].density_floor,
            mesh=mesh,
            metrics=metrics,
            simple_settings=simple_sheath_settings,
            full_settings=full_sheath_settings,
        )
    else:
        ion_boundary_state = None

    if apply_sheath_boundaries and ion_boundary_state is not None:
        if preserve_dump_target_state:
            ion_boundary = _IonBoundaryResult(
                density={
                    ion.name: np.asarray(
                        _merge_target_guard_cells(
                            prepared[ion.name].density,
                            ion_boundary_state.density[ion.name],
                            mesh=mesh,
                        ),
                        dtype=np.float64,
                    )
                    for ion in ions
                },
                pressure={
                    ion.name: np.asarray(
                        _merge_target_guard_cells(
                            prepared[ion.name].pressure,
                            ion_boundary_state.pressure[ion.name],
                            mesh=mesh,
                        ),
                        dtype=np.float64,
                    )
                    for ion in ions
                },
                temperature={
                    ion.name: np.asarray(
                        _merge_target_guard_cells(
                            prepared[ion.name].temperature,
                            ion_boundary_state.temperature[ion.name],
                            mesh=mesh,
                        ),
                        dtype=np.float64,
                    )
                    for ion in ions
                },
                velocity={
                    ion.name: np.asarray(
                        _merge_target_guard_cells(
                            prepared[ion.name].velocity,
                            ion_boundary_state.velocity[ion.name],
                            mesh=mesh,
                        ),
                        dtype=np.float64,
                    )
                    for ion in ions
                },
                momentum={
                    ion.name: np.asarray(
                        _merge_target_guard_cells(
                            prepared[ion.name].momentum,
                            ion_boundary_state.momentum[ion.name],
                            mesh=mesh,
                        ),
                        dtype=np.float64,
                    )
                    for ion in ions
                },
                energy_source={
                    ion.name: np.asarray(
                        np.zeros_like(ion_boundary_state.energy_source[ion.name], dtype=np.float64)
                        if preserve_dump_ion_target_state_only
                        else ion_boundary_state.energy_source[ion.name],
                        dtype=np.float64,
                    )
                    for ion in ions
                },
            )
        else:
            ion_boundary = ion_boundary_state
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
    simple_settings: _SimpleSheathSettings | None = None,
    full_settings: _FullSheathSettings | None = None,
) -> _IonBoundaryResult:
    te = _safe_temperature(electron_pressure, electron_density, electron_density_floor)
    dx = np.asarray(metrics.dx, dtype=np.float64)
    dz = np.asarray(metrics.dz, dtype=np.float64)
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
        lower_y_enabled = (
            simple_settings.lower_y
            if simple_settings is not None
            else True if full_settings is None else full_settings.lower_y
        )
        upper_y_enabled = (
            simple_settings.upper_y
            if simple_settings is not None
            else True if full_settings is None else full_settings.upper_y
        )

        if mesh.has_upper_y_target and upper_y_enabled:
            j = mesh.yend
            jp = j + 1
            jm = j - 1
            ni_i = density[:, j, :]
            ni_m = density[:, jm, :]
            density[:, jp, :] = np.asarray(
                limit_free(ni_m, ni_i, 0.0 if simple_settings is None else simple_settings.density_boundary_mode),
                dtype=np.float64,
            )
            temperature[:, jp, :] = np.asarray(
                limit_free(
                    temperature[:, jm, :],
                    temperature[:, j, :],
                    0.0 if simple_settings is None else simple_settings.temperature_boundary_mode,
                ),
                dtype=np.float64,
            )
            pressure[:, jp, :] = np.asarray(
                limit_free(
                    pressure[:, jm, :],
                    pressure[:, j, :],
                    0.0 if simple_settings is None else simple_settings.pressure_boundary_mode,
                ),
                dtype=np.float64,
            )

            nisheath = 0.5 * (density[:, jp, :] + density[:, j, :])
            tesheath = np.maximum(0.5 * (te[:, jp, :] + te[:, j, :]) if jp < te.shape[1] else 0.5 * (te[:, jm, :] + te[:, j, :]), 1.0e-5)
            tisheath = np.maximum(0.5 * (temperature[:, jp, :] + temperature[:, j, :]), 1.0e-5)
            if simple_settings is not None:
                c_i_sq = np.maximum(
                    (simple_settings.sheath_ion_polytropic * tisheath + ion.charge * tesheath) / ion.atomic_mass,
                    0.0,
                )
                visheath = np.maximum(vel[:, j, :], np.sqrt(c_i_sq))
                if simple_settings.no_flow:
                    visheath = np.zeros_like(visheath)
                vel[:, jp, :] = 2.0 * visheath - vel[:, j, :]
                momentum_field[:, jp, :] = 2.0 * ion.atomic_mass * nisheath * visheath - momentum_field[:, j, :]
                q = simple_settings.gamma_i * tisheath * nisheath * visheath
                q = q - (2.5 * tisheath + 0.5 * ion.atomic_mass * np.square(visheath)) * nisheath * visheath
                da = (
                    (J[:, j, :] + J[:, jp, :]) / (np.sqrt(g22[:, j, :]) + np.sqrt(g22[:, jp, :]))
                    * 0.5
                    * (dx[:, j, :] + dx[:, jp, :])
                    * 0.5
                    * (dz[:, j, :] + dz[:, jp, :])
                )
                dv = dx[:, j, :] * dy[:, j, :] * dz[:, j, :] * J[:, j, :]
                power = q * da / np.maximum(dv, 1.0e-30)
                energy_source[ion.name][:, j, :] -= power
            else:
                nesheath = (
                    0.5 * (electron_density[:, jp, :] + electron_density[:, j, :])
                    if jp < electron_density.shape[1]
                    else 0.5 * (electron_density[:, jm, :] + electron_density[:, j, :])
                )
                s_i = np.clip(nisheath / np.maximum(nesheath, 1.0e-10), 0.0, 1.0)
                grad_ne = electron_density[:, j, :] - nesheath
                grad_ni = density[:, j, :] - nisheath
                mask = np.abs(grad_ni) < 1.0e-3
                grad_ne = np.where(mask, 1.0e-3, grad_ne)
                grad_ni = np.where(mask, 1.0e-3, grad_ni)
                c_i_sq = np.clip(
                    ((5.0 / 3.0) * tisheath + ion.charge * s_i * tesheath * grad_ne / grad_ni) / ion.atomic_mass,
                    0.0,
                    100.0,
                )
                gamma_i = 2.5 + 0.5 * ion.atomic_mass * c_i_sq / tisheath
                visheath = np.sqrt(c_i_sq)
                vel[:, jp, :] = 2.0 * visheath - vel[:, j, :]
                momentum_field[:, jp, :] = 2.0 * ion.atomic_mass * nisheath * visheath - momentum_field[:, j, :]
                q = ((gamma_i - 1.0 - 1.0 / ((5.0 / 3.0) - 1.0)) * tisheath - 0.5 * c_i_sq * ion.atomic_mass) * nisheath * visheath
                q = np.maximum(q, 0.0)
                flux = q * (J[:, j, :] + J[:, jp, :]) / (np.sqrt(g22[:, j, :]) + np.sqrt(g22[:, jp, :]))
                power = flux / (dy[:, j, :] * J[:, j, :])
                energy_source[ion.name][:, j, :] -= power

        if mesh.has_lower_y_target and lower_y_enabled:
            j = mesh.ystart
            jm = j - 1
            jp = j + 1
            ni_i = density[:, j, :]
            ni_p = density[:, jp, :]
            density[:, jm, :] = np.asarray(
                limit_free(ni_p, ni_i, 0.0 if simple_settings is None else simple_settings.density_boundary_mode),
                dtype=np.float64,
            )
            temperature[:, jm, :] = np.asarray(
                limit_free(
                    temperature[:, jp, :],
                    temperature[:, j, :],
                    0.0 if simple_settings is None else simple_settings.temperature_boundary_mode,
                ),
                dtype=np.float64,
            )
            pressure[:, jm, :] = np.asarray(
                limit_free(
                    pressure[:, jp, :],
                    pressure[:, j, :],
                    0.0 if simple_settings is None else simple_settings.pressure_boundary_mode,
                ),
                dtype=np.float64,
            )

            nisheath = 0.5 * (density[:, jm, :] + density[:, j, :])
            tesheath = np.maximum(0.5 * (te[:, jm, :] + te[:, j, :]), 1.0e-5)
            tisheath = np.maximum(0.5 * (temperature[:, jm, :] + temperature[:, j, :]), 1.0e-5)
            if simple_settings is not None:
                c_i_sq = np.maximum(
                    (simple_settings.sheath_ion_polytropic * tisheath + ion.charge * tesheath) / ion.atomic_mass,
                    0.0,
                )
                visheath = np.minimum(vel[:, j, :], -np.sqrt(c_i_sq))
                if simple_settings.no_flow:
                    visheath = np.zeros_like(visheath)
                vel[:, jm, :] = 2.0 * visheath - vel[:, j, :]
                momentum_field[:, jm, :] = 2.0 * ion.atomic_mass * nisheath * visheath - momentum_field[:, j, :]
                q = simple_settings.gamma_i * tisheath * nisheath * visheath
                q = q - (2.5 * tisheath + 0.5 * ion.atomic_mass * np.square(visheath)) * nisheath * visheath
                da = (
                    (J[:, j, :] + J[:, jm, :]) / (np.sqrt(g22[:, j, :]) + np.sqrt(g22[:, jm, :]))
                    * 0.5
                    * (dx[:, j, :] + dx[:, jm, :])
                    * 0.5
                    * (dz[:, j, :] + dz[:, jm, :])
                )
                dv = dx[:, j, :] * dy[:, j, :] * dz[:, j, :] * J[:, j, :]
                power = q * da / np.maximum(dv, 1.0e-30)
                energy_source[ion.name][:, j, :] += power
            else:
                nesheath = 0.5 * (electron_density[:, jm, :] + electron_density[:, j, :])
                s_i = np.clip(nisheath / np.maximum(nesheath, 1.0e-10), 0.0, 1.0)
                grad_ne = electron_density[:, j, :] - nesheath
                grad_ni = density[:, j, :] - nisheath
                mask = np.abs(grad_ni) < 1.0e-3
                grad_ne = np.where(mask, 1.0e-3, grad_ne)
                grad_ni = np.where(mask, 1.0e-3, grad_ni)
                c_i_sq = np.clip(
                    ((5.0 / 3.0) * tisheath + ion.charge * s_i * tesheath * grad_ne / grad_ni) / ion.atomic_mass,
                    0.0,
                    100.0,
                )
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
    simple_settings: _SimpleSheathSettings | None = None,
    full_settings: _FullSheathSettings | None = None,
) -> _ElectronBoundaryResult:
    if simple_settings is not None:
        return _apply_electron_simple_sheath_boundary(
            electron_pressure=electron_pressure,
            electron_density=electron_density,
            electron_velocity=electron_velocity,
            electron_mass=electron_mass,
            electron_density_floor=electron_density_floor,
            ion_velocity=ion_velocity,
            ions=ions,
            prepared_ions=prepared_ions,
            mesh=mesh,
            metrics=metrics,
            settings=simple_settings,
        )
    settings = full_settings or _FullSheathSettings(
        secondary_electron_coef=0.0,
        sin_alpha=np.ones_like(electron_density, dtype=np.float64),
        lower_y=True,
        upper_y=True,
        wall_potential=np.zeros_like(electron_density, dtype=np.float64),
        floor_potential=True,
    )
    density = np.array(
        apply_noflow_scalar_guards(
            electron_density,
            mesh=mesh,
            lower_y=settings.lower_y,
            upper_y=settings.upper_y,
        ),
        dtype=np.float64,
        copy=True,
    )
    pressure = np.array(
        apply_noflow_scalar_guards(
            electron_pressure,
            mesh=mesh,
            lower_y=settings.lower_y,
            upper_y=settings.upper_y,
        ),
        dtype=np.float64,
        copy=True,
    )
    temperature = _safe_temperature(pressure, density, electron_density_floor)
    velocity = np.array(
        apply_noflow_flow_guards(
            np.asarray(electron_velocity, dtype=np.float64),
            mesh=mesh,
            lower_y=settings.lower_y,
            upper_y=settings.upper_y,
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
    secondary_coef = float(settings.secondary_electron_coef)
    sin_alpha = np.asarray(settings.sin_alpha, dtype=np.float64)
    wall_potential = np.asarray(settings.wall_potential, dtype=np.float64)
    electron_adiabatic = 5.0 / 3.0

    def _full_ion_sum_at_boundary(*, j: int, neighbor: int) -> np.ndarray:
        total = np.zeros_like(density[:, j, :], dtype=np.float64)
        for ion in ions:
            ion_state = prepared_ions[ion.name]
            ti = np.asarray(ion_state.temperature, dtype=np.float64)
            ni = np.asarray(ion_state.density, dtype=np.float64)
            s_i = np.clip(
                0.5
                * (
                    3.0 * ni[:, j, :] / np.maximum(density[:, j, :], 1.0e-12)
                    - ni[:, neighbor, :] / np.maximum(density[:, neighbor, :], 1.0e-12)
                ),
                0.0,
                1.0,
            )
            s_i = np.where(np.isfinite(s_i), s_i, 1.0)
            grad_ne = density[:, neighbor, :] - density[:, j, :]
            grad_ni = ni[:, neighbor, :] - ni[:, j, :]
            use_floor = np.abs(grad_ni) < 2.0e-3
            grad_ne = np.where(use_floor, 2.0e-3, grad_ne)
            grad_ni = np.where(use_floor, 2.0e-3, grad_ni)
            c_i_sq = np.clip(
                ((5.0 / 3.0) * ti[:, j, :] + ion.charge * s_i * temperature[:, j, :] * grad_ne / grad_ni)
                / ion.atomic_mass,
                0.0,
                100.0,
            )
            total = total + s_i * ion.charge * sin_alpha[:, neighbor, :] * np.sqrt(c_i_sq)
        return total

    def _set_zero_current_phi(*, j: int, neighbor: int, ghost: int) -> None:
        valid = temperature[:, j, :] > 0.0
        safe_temperature = np.maximum(temperature[:, j, :], 1.0e-12)
        ion_sum = np.maximum(_full_ion_sum_at_boundary(j=j, neighbor=neighbor), 1.0e-12)
        phi[:, j, :] = np.where(
            valid,
            safe_temperature
            * np.log(
                np.maximum(
                    np.sqrt(safe_temperature / (me * (2.0 * math.pi))) * (1.0 - secondary_coef) / ion_sum,
                    1.0e-12,
                )
            ),
            0.0,
        )
        phi[:, j, :] = phi[:, j, :] + wall_potential[:, j, :]
        phi[:, neighbor, :] = phi[:, j, :]
        phi[:, ghost, :] = phi[:, j, :]

    if mesh.has_upper_y_target and settings.upper_y:
        j = mesh.yend
        jp = j + 1
        jm = j - 1
        density[:, jp, :] = np.asarray(limit_free(density[:, jm, :], density[:, j, :], 0), dtype=np.float64)
        temperature[:, jp, :] = np.asarray(limit_free(temperature[:, jm, :], temperature[:, j, :], 0), dtype=np.float64)
        pressure[:, jp, :] = np.asarray(limit_free(pressure[:, jm, :], pressure[:, j, :], 0), dtype=np.float64)
        _set_zero_current_phi(j=j, neighbor=jm, ghost=jp)
        phi[:, jp, :] = 2.0 * phi[:, j, :] - phi[:, jm, :]
        phi_wall = wall_potential[:, j, :]
        phisheath_raw = 0.5 * (phi[:, jp, :] + phi[:, j, :])
        phisheath = np.maximum(phisheath_raw, phi_wall) if settings.floor_potential else phisheath_raw
        tesheath = 0.5 * (temperature[:, jp, :] + temperature[:, j, :])
        nesheath = 0.5 * (density[:, jp, :] + density[:, j, :])
        gamma_e = np.maximum(
            2.0 / (1.0 - secondary_coef) + (phisheath - phi_wall) / np.maximum(tesheath, 1.0e-5),
            0.0,
        )
        vesheath = np.where(
            tesheath < 1.0e-10,
            0.0,
            np.sqrt(tesheath / (2.0 * math.pi * me))
            * (1.0 - secondary_coef)
            * np.exp(-(phisheath - phi_wall) / np.maximum(tesheath, 1.0e-12)),
        )
        velocity[:, jp, :] = 2.0 * vesheath - velocity[:, j, :]
        momentum[:, jp, :] = 2.0 * electron_mass * nesheath * vesheath - momentum[:, j, :]
        q = ((gamma_e - 1.0 - 1.0 / (electron_adiabatic - 1.0)) * tesheath - 0.5 * me * np.square(vesheath)) * nesheath * vesheath
        q = np.maximum(q, 0.0)
        flux = q * (J[:, j, :] + J[:, jp, :]) / (np.sqrt(g22[:, j, :]) + np.sqrt(g22[:, jp, :]))
        power = flux / (dy[:, j, :] * J[:, j, :])
        energy_source[:, j, :] -= power

    if mesh.has_lower_y_target and settings.lower_y:
        j = mesh.ystart
        jm = j - 1
        jp = j + 1
        density[:, jm, :] = np.asarray(limit_free(density[:, jp, :], density[:, j, :], 0), dtype=np.float64)
        temperature[:, jm, :] = np.asarray(limit_free(temperature[:, jp, :], temperature[:, j, :], 0), dtype=np.float64)
        pressure[:, jm, :] = np.asarray(limit_free(pressure[:, jp, :], pressure[:, j, :], 0), dtype=np.float64)
        _set_zero_current_phi(j=j, neighbor=jp, ghost=jm)
        phi[:, jm, :] = 2.0 * phi[:, j, :] - phi[:, jp, :]
        phi_wall = wall_potential[:, j, :]
        phisheath_raw = 0.5 * (phi[:, jm, :] + phi[:, j, :])
        phisheath = np.maximum(phisheath_raw, phi_wall) if settings.floor_potential else phisheath_raw
        tesheath = 0.5 * (temperature[:, jm, :] + temperature[:, j, :])
        nesheath = 0.5 * (density[:, jm, :] + density[:, j, :])
        gamma_e = np.maximum(
            2.0 / (1.0 - secondary_coef) + (phisheath - phi_wall) / np.maximum(tesheath, 1.0e-5),
            0.0,
        )
        vesheath = np.where(
            tesheath < 1.0e-10,
            0.0,
            -np.sqrt(tesheath / (2.0 * math.pi * me))
            * (1.0 - secondary_coef)
            * np.exp(-(phisheath - phi_wall) / np.maximum(tesheath, 1.0e-12)),
        )
        velocity[:, jm, :] = 2.0 * vesheath - velocity[:, j, :]
        momentum[:, jm, :] = 2.0 * electron_mass * nesheath * vesheath - momentum[:, j, :]
        q = ((gamma_e - 1.0 - 1.0 / (electron_adiabatic - 1.0)) * tesheath - 0.5 * me * np.square(vesheath)) * nesheath * vesheath
        q = np.minimum(q, 0.0)
        flux = q * (J[:, j, :] + J[:, jm, :]) / (np.sqrt(g22[:, j, :]) + np.sqrt(g22[:, jm, :]))
        power = flux / (dy[:, j, :] * J[:, j, :])
        energy_source[:, j, :] += power
    return _ElectronBoundaryResult(
        density=density,
        temperature=temperature,
        pressure=pressure,
        velocity=velocity,
        momentum=momentum,
        energy_source=energy_source,
    )


def _apply_electron_simple_sheath_boundary(
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
    settings: _SimpleSheathSettings,
) -> _ElectronBoundaryResult:
    density = np.array(
        apply_noflow_scalar_guards(electron_density, mesh=mesh, lower_y=settings.lower_y, upper_y=settings.upper_y),
        dtype=np.float64,
        copy=True,
    )
    pressure = np.array(
        apply_noflow_scalar_guards(electron_pressure, mesh=mesh, lower_y=settings.lower_y, upper_y=settings.upper_y),
        dtype=np.float64,
        copy=True,
    )
    temperature = _safe_temperature(pressure, density, electron_density_floor)
    velocity = np.array(
        apply_noflow_flow_guards(
            np.asarray(electron_velocity, dtype=np.float64),
            mesh=mesh,
            lower_y=settings.lower_y,
            upper_y=settings.upper_y,
        ),
        dtype=np.float64,
        copy=True,
    )
    momentum = np.asarray(electron_mass * density * velocity, dtype=np.float64)
    energy_source = np.zeros_like(density, dtype=np.float64)
    g22 = np.asarray(metrics.g_22, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    dx = np.asarray(metrics.dx, dtype=np.float64)
    dz = np.asarray(metrics.dz, dtype=np.float64)
    J = np.asarray(metrics.J, dtype=np.float64)
    secondary_coef = float(settings.secondary_electron_coef)
    ion_polytropic = float(settings.sheath_ion_polytropic)
    gamma_e = float(settings.gamma_e)
    wall_potential = np.asarray(settings.wall_potential, dtype=np.float64)

    def _ion_sum_at_boundary(j: int, guard: int, *, sign: float) -> np.ndarray:
        total = np.zeros_like(density[:, j, :], dtype=np.float64)
        for ion in ions:
            ion_state = prepared_ions[ion.name]
            ni = np.asarray(ion_state.density, dtype=np.float64)
            ti = np.asarray(ion_state.temperature, dtype=np.float64)
            vi = np.asarray(ion_velocity[ion.name], dtype=np.float64)
            ni_guard = np.asarray(
                limit_free(ni[:, guard, :], ni[:, j, :], settings.density_boundary_mode),
                dtype=np.float64,
            )
            ti_guard = np.asarray(
                limit_free(ti[:, guard, :], ti[:, j, :], settings.temperature_boundary_mode),
                dtype=np.float64,
            )
            te_guard = np.asarray(
                limit_free(temperature[:, guard, :], temperature[:, j, :], settings.temperature_boundary_mode),
                dtype=np.float64,
            )
            nisheath = 0.5 * (ni_guard + ni[:, j, :])
            tesheath = np.maximum(0.5 * (te_guard + temperature[:, j, :]), 1.0e-5)
            tisheath = np.maximum(0.5 * (ti_guard + ti[:, j, :]), 1.0e-5)
            c_i_sq = (ion_polytropic * tisheath + ion.charge * tesheath) / ion.atomic_mass
            c_i_sq = np.maximum(c_i_sq, 0.0)
            if sign < 0.0:
                visheath = np.minimum(vi[:, j, :], -np.sqrt(c_i_sq))
                total = total - ion.charge * nisheath * visheath
            else:
                visheath = np.maximum(vi[:, j, :], np.sqrt(c_i_sq))
                total = total + ion.charge * nisheath * visheath
        return total

    if settings.lower_y and mesh.has_lower_y_target:
        j = mesh.ystart
        guard = j + 1
        ghost = j - 1
        density[:, ghost, :] = np.asarray(
            limit_free(density[:, guard, :], density[:, j, :], settings.density_boundary_mode),
            dtype=np.float64,
        )
        temperature[:, ghost, :] = np.asarray(
            limit_free(temperature[:, guard, :], temperature[:, j, :], settings.temperature_boundary_mode),
            dtype=np.float64,
        )
        pressure[:, ghost, :] = np.asarray(
            limit_free(pressure[:, guard, :], pressure[:, j, :], settings.pressure_boundary_mode),
            dtype=np.float64,
        )
        nesheath = 0.5 * (density[:, ghost, :] + density[:, j, :])
        tesheath = np.maximum(0.5 * (temperature[:, ghost, :] + temperature[:, j, :]), 1.0e-5)
        ion_sum = np.maximum(_ion_sum_at_boundary(j, guard, sign=-1.0), 1.0e-5)
        phi_boundary = tesheath * np.log(
            np.sqrt(tesheath / (electron_mass * (2.0 * math.pi)))
            * (1.0 - secondary_coef)
            * np.maximum(nesheath, 1.0e-5)
            / ion_sum
        )
        phi_boundary = phi_boundary + wall_potential[:, j, :]
        phisheath = np.maximum(phi_boundary, wall_potential[:, j, :])
        vesheath = -np.sqrt(tesheath / (2.0 * math.pi * electron_mass)) * (1.0 - secondary_coef) * np.exp(
            -(phisheath - wall_potential[:, j, :]) / np.maximum(tesheath, 1.0e-5)
        )
        q = gamma_e * tesheath * nesheath * vesheath
        if settings.no_flow:
            vesheath = 0.0
        velocity[:, ghost, :] = 2.0 * vesheath - velocity[:, j, :]
        momentum[:, ghost, :] = 2.0 * electron_mass * nesheath * vesheath - momentum[:, j, :]
        q = q - (2.5 * tesheath + 0.5 * electron_mass * np.square(vesheath)) * nesheath * vesheath
        area = (
            (J[:, j, :] + J[:, ghost, :])
            / (np.sqrt(g22[:, j, :]) + np.sqrt(g22[:, ghost, :]))
            * 0.5
            * (dx[:, j, :] + dx[:, ghost, :])
            * 0.5
            * (dz[:, j, :] + dz[:, ghost, :])
        )
        volume = dx[:, j, :] * dy[:, j, :] * dz[:, j, :] * J[:, j, :]
        energy_source[:, j, :] = energy_source[:, j, :] + (q * area) / volume

    if settings.upper_y and mesh.has_upper_y_target:
        j = mesh.yend
        guard = j - 1
        ghost = j + 1
        density[:, ghost, :] = np.asarray(
            limit_free(density[:, guard, :], density[:, j, :], settings.density_boundary_mode),
            dtype=np.float64,
        )
        temperature[:, ghost, :] = np.asarray(
            limit_free(temperature[:, guard, :], temperature[:, j, :], settings.temperature_boundary_mode),
            dtype=np.float64,
        )
        pressure[:, ghost, :] = np.asarray(
            limit_free(pressure[:, guard, :], pressure[:, j, :], settings.pressure_boundary_mode),
            dtype=np.float64,
        )
        nesheath = 0.5 * (density[:, ghost, :] + density[:, j, :])
        tesheath = np.maximum(0.5 * (temperature[:, ghost, :] + temperature[:, j, :]), 1.0e-5)
        ion_sum = np.maximum(_ion_sum_at_boundary(j, guard, sign=1.0), 1.0e-5)
        phi_boundary = tesheath * np.log(
            np.sqrt(tesheath / (electron_mass * (2.0 * math.pi)))
            * (1.0 - secondary_coef)
            * np.maximum(nesheath, 1.0e-5)
            / ion_sum
        )
        phi_boundary = phi_boundary + wall_potential[:, j, :]
        phisheath = np.maximum(phi_boundary, wall_potential[:, j, :])
        vesheath = np.sqrt(tesheath / (2.0 * math.pi * electron_mass)) * (1.0 - secondary_coef) * np.exp(
            -(phisheath - wall_potential[:, j, :]) / np.maximum(tesheath, 1.0e-5)
        )
        q = gamma_e * tesheath * nesheath * vesheath
        if settings.no_flow:
            vesheath = 0.0
        velocity[:, ghost, :] = 2.0 * vesheath - velocity[:, j, :]
        momentum[:, ghost, :] = 2.0 * electron_mass * nesheath * vesheath - momentum[:, j, :]
        q = q - (2.5 * tesheath + 0.5 * electron_mass * np.square(vesheath)) * nesheath * vesheath
        area = (
            (J[:, j, :] + J[:, ghost, :])
            / (np.sqrt(g22[:, j, :]) + np.sqrt(g22[:, ghost, :]))
            * 0.5
            * (dx[:, j, :] + dx[:, ghost, :])
            * 0.5
            * (dz[:, j, :] + dz[:, ghost, :])
        )
        volume = dx[:, j, :] * dy[:, j, :] * dz[:, j, :] * J[:, j, :]
        energy_source[:, j, :] = energy_source[:, j, :] - (q * area) / volume

    return _ElectronBoundaryResult(
        density=density,
        temperature=temperature,
        pressure=pressure,
        velocity=velocity,
        momentum=momentum,
        energy_source=energy_source,
    )


def advance_recycling_1d_implicit_history(
    config: BoutConfig,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    timestep: float,
    steps: int,
    initial_fields: dict[str, np.ndarray] | None = None,
    field_template_overrides: dict[str, np.ndarray] | None = None,
    initial_feedback_integrals: dict[str, float] | None = None,
    density_source_overrides: dict[str, np.ndarray] | None = None,
    pressure_source_overrides: dict[str, np.ndarray] | None = None,
    momentum_source_overrides: dict[str, np.ndarray] | None = None,
    preserve_dump_target_state: bool = False,
    preserve_dump_ion_target_state_only: bool = False,
    solver_mode: str = "bdf",
    residual_tolerance: float = 1.0e-8,
    max_nonlinear_iterations: int = 20,
    progress_callback: RecyclingProgressCallback | None = None,
) -> Recycling1DHistoryResult:
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
        field_overrides=initial_fields,
        field_template_overrides=field_template_overrides,
        density_source_overrides=density_source_overrides,
        pressure_source_overrides=pressure_source_overrides,
        momentum_source_overrides=momentum_source_overrides,
        preserve_dump_target_state=preserve_dump_target_state,
        preserve_dump_ion_target_state_only=preserve_dump_ion_target_state_only,
    )
    field_names = runtime_model.field_names
    feedback_names = runtime_model.feedback_names
    fields = _build_recycling_state_fields(runtime_model, field_overrides=initial_fields)
    integrals = {name: float((initial_feedback_integrals or {}).get(name, 0.0)) for name in feedback_names}

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
            progress_callback=progress_callback,
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
            progress_callback=progress_callback,
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
            progress_callback=progress_callback,
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
            progress_callback=progress_callback,
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
        if progress_callback is not None:
            progress_callback(
                {
                    "interval_index": len(next(iter(variable_history.values()))) - 1,
                    "steps": steps,
                    "solver_mode": solver_mode,
                    "accepted_dt": float(timestep),
                    "stored_states": len(next(iter(variable_history.values()))),
                }
            )

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
    progress_callback: RecyclingProgressCallback | None = None,
) -> Recycling1DHistoryResult:
    current_fields = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
    current_integrals = {name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names}
    variable_history = {name: [np.asarray(current_fields[name], dtype=np.float64)] for name in field_names}
    feedback_history = {name: [np.asarray(current_integrals[name], dtype=np.float64)] for name in feedback_names}
    suggested_dt = _initial_recycling_continuation_dt(runtime_model, timestep=timestep)

    for interval_index in range(steps):
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
            startup_warmup=(interval_index == 0),
        )
        for name in field_names:
            variable_history[name].append(np.asarray(current_fields[name], dtype=np.float64))
        for name in feedback_names:
            feedback_history[name].append(np.asarray(current_integrals[name], dtype=np.float64))
        if progress_callback is not None:
            progress_callback(
                {
                    "interval_index": interval_index + 1,
                    "steps": steps,
                    "solver_mode": "continuation",
                    "accepted_dt": float(suggested_dt),
                    "stored_states": len(next(iter(variable_history.values()))),
                }
            )

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
    progress_callback: RecyclingProgressCallback | None = None,
) -> Recycling1DHistoryResult:
    current_fields = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
    current_integrals = {name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names}
    variable_history = {name: [np.asarray(current_fields[name], dtype=np.float64)] for name in field_names}
    feedback_history = {name: [np.asarray(current_integrals[name], dtype=np.float64)] for name in feedback_names}
    suggested_dt = min(float(timestep), 10.0 if len(field_names) > 10 else 5.0)

    for interval_index in range(steps):
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
        if progress_callback is not None:
            progress_callback(
                {
                    "interval_index": interval_index + 1,
                    "steps": steps,
                    "solver_mode": "adaptive_be",
                    "accepted_dt": float(suggested_dt),
                    "stored_states": len(next(iter(variable_history.values()))),
                }
            )

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
    progress_callback: RecyclingProgressCallback | None = None,
) -> Recycling1DHistoryResult:
    current_fields = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
    current_integrals = {name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names}
    previous_fields: dict[str, np.ndarray] | None = None
    previous_integrals: dict[str, float] | None = None
    previous_dt: float | None = None
    variable_history = {name: [np.asarray(current_fields[name], dtype=np.float64)] for name in field_names}
    feedback_history = {name: [np.asarray(current_integrals[name], dtype=np.float64)] for name in feedback_names}
    suggested_dt = _initial_recycling_continuation_dt(runtime_model, timestep=timestep)

    for interval_index in range(steps):
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
        if progress_callback is not None:
            progress_callback(
                {
                    "interval_index": interval_index + 1,
                    "steps": steps,
                    "solver_mode": "adaptive_bdf",
                    "accepted_dt": float(suggested_dt),
                    "stored_states": len(next(iter(variable_history.values()))),
                }
            )

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
    startup_warmup: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    trial_dt = min(float(suggested_dt), float(output_timestep))
    minimum_dt = max(float(output_timestep) / 4096.0, 1.0)
    acceptance_residual = max(1.0e4 * residual_tolerance, 5.0e-3)
    startup_window = (
        min(25.0, float(output_timestep))
        if startup_warmup
        else 0.0
    )
    startup_dt = min(6.25, startup_window) if startup_window > 0.0 else 0.0
    remaining = float(output_timestep)
    elapsed = 0.0
    current_fields = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
    current_integrals = {name: float(feedback_integrals.get(name, 0.0)) for name in runtime_model.feedback_names}
    previous_fields: dict[str, np.ndarray] | None = None
    previous_integrals: dict[str, float] | None = None
    last_info: Recycling1DImplicitStepInfo | None = None
    last_dt = trial_dt

    while remaining > 1.0e-12:
        if startup_window > elapsed + 1.0e-12:
            step_dt = min(startup_dt, startup_window - elapsed, remaining)
        else:
            step_dt = min(trial_dt, remaining)

        while True:
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
                    evolve_feedback_integrals=True,
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
                    evolve_feedback_integrals=True,
                )

            last_info = info
            if np.isfinite(info.residual_inf_norm) and info.residual_inf_norm <= acceptance_residual:
                break
            if step_dt <= minimum_dt:
                raise RuntimeError(
                    f"Recycling continuation interval failed at dt={step_dt:g}; residual={float('nan') if last_info is None else last_info.residual_inf_norm:g}"
                )
            step_dt = 0.5 * step_dt

        previous_fields = current_fields
        previous_integrals = current_integrals
        current_fields = next_fields
        current_integrals = next_integrals
        remaining -= step_dt
        elapsed += step_dt
        last_dt = step_dt

    return current_fields, current_integrals, last_dt


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
    evolve_feedback_integrals: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, float], Recycling1DImplicitStepInfo]:
    runtime_model = runtime_model or _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
    )
    field_names = runtime_model.field_names
    packed_feedback_names = runtime_model.feedback_names if evolve_feedback_integrals else ()
    layout = _build_recycling_packed_state_layout(
        fields=fields,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
    )
    previous_feedback_errors = _current_feedback_errors(fields, controllers=runtime_model.controllers, mesh=mesh)
    packed_previous = _pack_recycling_active_state(
        fields,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
        layout=layout,
    )
    packed_initial_guess = _predict_recycling_packed_state(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        feedback_previous_errors=previous_feedback_errors,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        timestep=timestep,
        layout=layout,
    )

    def residual(packed_state: np.ndarray) -> np.ndarray:
        state_fields, state_integrals = _unpack_recycling_active_state(
            packed_state,
            field_templates=fields,
            feedback_integrals=feedback_integrals,
            field_names=field_names,
            feedback_names=packed_feedback_names,
            mesh=mesh,
            layout=layout,
        )
        rhs = _compute_recycling_1d_packed_rhs(
            config,
            state_fields,
            sanitize_fields=False,
            feedback_integrals=state_integrals,
            feedback_previous_errors=previous_feedback_errors,
            # When controller integrals are part of the implicit state, the source
            # path should consume that state directly rather than applying a second
            # trapezoid predictor to the same integral.
            feedback_timestep=None if packed_feedback_names else timestep,
            field_names=field_names,
            feedback_names=packed_feedback_names,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            runtime_model=runtime_model,
            layout=layout,
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
            packed_initial_guess,
            active_shape=(packed_previous.size,),
            sparsity=_build_recycling_residual_sparsity(
                active_shape=layout.active_shape,
                field_count=len(field_names),
                controller_count=len(packed_feedback_names),
            ),
            color_groups=_build_recycling_color_groups(
                active_shape=layout.active_shape,
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
        layout=layout,
    )
    sanitized_fields = _sanitize_recycling_fields(config, next_fields)
    if evolve_feedback_integrals:
        sanitized_integrals = _sanitize_feedback_integrals(next_integrals, controllers=runtime_model.controllers)
    else:
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
    evolve_feedback_integrals: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, float], Recycling1DImplicitStepInfo]:
    runtime_model = runtime_model or _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
    )
    field_names = runtime_model.field_names
    packed_feedback_names = runtime_model.feedback_names if evolve_feedback_integrals else ()
    layout = _build_recycling_packed_state_layout(
        fields=fields,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
    )
    previous_feedback_errors = _current_feedback_errors(fields, controllers=runtime_model.controllers, mesh=mesh)
    packed_previous = _pack_recycling_active_state(
        fields,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
        layout=layout,
    )
    packed_previous_previous = _pack_recycling_active_state(
        previous_fields,
        feedback_integrals=previous_feedback_integrals,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
        layout=layout,
    )
    packed_initial_guess = _predict_recycling_packed_state(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        feedback_previous_errors=previous_feedback_errors,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        timestep=timestep,
        layout=layout,
    )

    def residual(packed_state: np.ndarray) -> np.ndarray:
        state_fields, state_integrals = _unpack_recycling_active_state(
            packed_state,
            field_templates=fields,
            feedback_integrals=feedback_integrals,
            field_names=field_names,
            feedback_names=packed_feedback_names,
            mesh=mesh,
            layout=layout,
        )
        rhs = _compute_recycling_1d_packed_rhs(
            config,
            state_fields,
            sanitize_fields=False,
            feedback_integrals=state_integrals,
            feedback_previous_errors=previous_feedback_errors,
            # When controller integrals are part of the implicit state, the source
            # path should consume that state directly rather than applying a second
            # trapezoid predictor to the same integral.
            feedback_timestep=None if packed_feedback_names else timestep,
            field_names=field_names,
            feedback_names=packed_feedback_names,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            runtime_model=runtime_model,
            layout=layout,
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
            packed_initial_guess,
            active_shape=(packed_previous.size,),
            sparsity=_build_recycling_residual_sparsity(
                active_shape=layout.active_shape,
                field_count=len(field_names),
                controller_count=len(packed_feedback_names),
            ),
            color_groups=_build_recycling_color_groups(
                active_shape=layout.active_shape,
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
        layout=layout,
    )
    sanitized_fields = _sanitize_recycling_fields(config, next_fields)
    if evolve_feedback_integrals:
        sanitized_integrals = _sanitize_feedback_integrals(next_integrals, controllers=runtime_model.controllers)
    else:
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
    progress_callback: RecyclingProgressCallback | None = None,
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
    layout = _build_recycling_packed_state_layout(
        fields=fields,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
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
        layout=layout,
    )
    sparsity_csc = sparsity.tocsc()

    def rhs(_time: float, packed_state: np.ndarray) -> np.ndarray:
        state_fields, state_integrals = _unpack_recycling_active_state(
            packed_state,
            field_templates=current_fields,
            feedback_integrals=current_integrals,
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
            layout=layout,
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
            layout=layout,
        )

    def jacobian(_time: float, packed_state: np.ndarray):
        rhs_value = rhs(_time, packed_state)
        return build_sparse_difference_quotient_jacobian(
            lambda state: rhs(_time, state),
            packed_state,
            base_residual=rhs_value,
            sparsity=sparsity,
            color_groups=color_groups,
            sparsity_csc=sparsity_csc,
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
            layout=layout,
        )
        sample_fields = _sanitize_recycling_fields(config, sample_fields)
        sample_integrals = _sanitize_feedback_integrals(sample_integrals, controllers=runtime_model.controllers)
        for name in field_names:
            variable_history[name].append(np.asarray(sample_fields[name], dtype=np.float64))
        for name in feedback_names:
            feedback_history[name].append(np.asarray(sample_integrals[name], dtype=np.float64))
        if progress_callback is not None and column > 0:
            progress_callback(
                {
                    "interval_index": column,
                    "steps": steps,
                    "solver_mode": "bdf",
                    "accepted_dt": float(timestep),
                    "stored_states": column + 1,
                }
            )

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
    layout: _RecyclingPackedStateLayout | None = None,
) -> np.ndarray:
    runtime_model = runtime_model or _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
    )
    sanitized_fields = _sanitize_recycling_fields(config, fields) if sanitize_fields else {
        name: np.asarray(value, dtype=np.float64) for name, value in fields.items()
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
        density_source_overrides=runtime_model.density_source_overrides,
        pressure_source_overrides=runtime_model.pressure_source_overrides,
        momentum_source_overrides=runtime_model.momentum_source_overrides,
        preserve_dump_target_state=runtime_model.preserve_dump_target_state,
        preserve_dump_ion_target_state_only=runtime_model.preserve_dump_ion_target_state_only,
    )
    active_slices = layout.active_slices if layout is not None else _recycling_active_domain_slices(mesh)
    pieces = [
        np.asarray(result.variables[f"ddt({name})"][0][active_slices], dtype=np.float64).ravel()
        for name in field_names
    ]
    pieces.extend(np.asarray(result.feedback_integral_rhs.get(name, 0.0), dtype=np.float64).reshape(1) for name in feedback_names)
    return np.concatenate(pieces) if pieces else np.array([], dtype=np.float64)


def _predict_recycling_packed_state(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel,
    feedback_integrals: dict[str, float],
    feedback_previous_errors: dict[str, float] | None,
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    timestep: float,
    layout: _RecyclingPackedStateLayout | None = None,
) -> np.ndarray:
    packed_previous = _pack_recycling_active_state(
        fields,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
        layout=layout,
    )
    rhs = _compute_recycling_1d_packed_rhs(
        config,
        fields,
        sanitize_fields=False,
        feedback_integrals=feedback_integrals,
        feedback_previous_errors=feedback_previous_errors,
        feedback_timestep=None if feedback_names else timestep,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        runtime_model=runtime_model,
        layout=layout,
    )
    predicted = np.asarray(packed_previous, dtype=np.float64) + float(timestep) * np.asarray(rhs, dtype=np.float64)
    predicted_fields, predicted_integrals = _unpack_recycling_active_state(
        predicted,
        field_templates=fields,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
        layout=layout,
    )
    sanitized_fields = _sanitize_recycling_fields(config, predicted_fields)
    sanitized_integrals = _sanitize_feedback_integrals(predicted_integrals, controllers=runtime_model.controllers)
    return _pack_recycling_active_state(
        sanitized_fields,
        feedback_integrals=sanitized_integrals,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
        layout=layout,
    )


def _pack_recycling_active_state(
    fields: dict[str, np.ndarray],
    *,
    feedback_integrals: dict[str, float],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    layout: _RecyclingPackedStateLayout | None = None,
) -> np.ndarray:
    field_block = pack_active_fields(
        tuple(np.asarray(fields[name], dtype=np.float64) for name in field_names),
        active_slices=(layout.active_slices if layout is not None else _recycling_active_domain_slices(mesh)),
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
    layout: _RecyclingPackedStateLayout | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    packed_array = np.asarray(packed, dtype=np.float64)
    field_size = layout.field_size if layout is not None else (_recycling_active_field_size(mesh) * len(field_names))
    field_block = packed_array[:field_size]
    scalar_block = packed_array[field_size:]
    unpacked_fields = unpack_active_fields(
        field_block,
        templates=(
            layout.field_templates
            if layout is not None
            else tuple(np.asarray(field_templates[name], dtype=np.float64) for name in field_names)
        ),
        active_slices=(layout.active_slices if layout is not None else _recycling_active_domain_slices(mesh)),
    )
    restored_fields = {name: value for name, value in zip(field_names, unpacked_fields, strict=True)}
    restored_integrals = {name: float(value) for name, value in feedback_integrals.items()}
    for index, name in enumerate(feedback_names):
        restored_integrals[name] = float(scalar_block[index])
    return restored_fields, restored_integrals


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
    rhs_array = np.asarray(rhs_fields, dtype=np.float64)
    field_rhs = rhs_array[:field_size]
    field_block = backward_euler_residual(
        np.asarray(packed_state, dtype=np.float64)[:field_size],
        np.asarray(previous_packed_state, dtype=np.float64)[:field_size],
        field_rhs,
        timestep=timestep,
    )
    controller_rhs = rhs_array[field_size:]
    controller_block = backward_euler_residual(
        _feedback_integral_vector(feedback_integrals, feedback_names=feedback_names),
        _feedback_integral_vector(previous_feedback_integrals, feedback_names=feedback_names),
        controller_rhs if feedback_names else _feedback_error_vector(current_feedback_errors, feedback_names=feedback_names),
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
    rhs_array = np.asarray(rhs_fields, dtype=np.float64)
    field_rhs = rhs_array[:field_size]
    field_block = bdf2_residual(
        np.asarray(packed_state, dtype=np.float64)[:field_size],
        np.asarray(previous_packed_state, dtype=np.float64)[:field_size],
        np.asarray(previous_previous_packed_state, dtype=np.float64)[:field_size],
        field_rhs,
        timestep=timestep,
    )
    controller_rhs = rhs_array[field_size:]
    controller_block = bdf2_residual(
        _feedback_integral_vector(feedback_integrals, feedback_names=feedback_names),
        _feedback_integral_vector(previous_feedback_integrals, feedback_names=feedback_names),
        _feedback_integral_vector(previous_previous_feedback_integrals, feedback_names=feedback_names),
        controller_rhs if feedback_names else _feedback_error_vector(current_feedback_errors, feedback_names=feedback_names),
        timestep=timestep,
    )
    return np.concatenate([field_block, controller_block]) if feedback_names else field_block


@lru_cache(maxsize=None)
def _build_recycling_residual_sparsity(
    *,
    active_shape: tuple[int, int, int],
    field_count: int,
    controller_count: int,
):
    try:
        from scipy.sparse import bmat, csr_matrix
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Sparse recycling stepping requires scipy.") from exc

    field_sparsity = build_locality_sparsity(
        active_shape,
        field_count=field_count,
        radii=(0, 2, 0),
        periodic_axes=(),
    )
    field_size = field_sparsity.shape[0]
    if controller_count <= 0:
        return field_sparsity.tocsr()

    field_to_controller = csr_matrix(np.ones((field_size, controller_count), dtype=bool))
    controller_to_field = csr_matrix(np.ones((controller_count, field_size), dtype=bool))
    controller_block = csr_matrix(np.ones((controller_count, controller_count), dtype=bool))
    return bmat(
        [
            [field_sparsity, field_to_controller],
            [controller_to_field, controller_block],
        ],
        format="csr",
        dtype=bool,
    )


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
