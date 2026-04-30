from __future__ import annotations

from functools import lru_cache
import math
import os
import re
import time

import jax.numpy as jnp
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
    prepare_sparse_difference_quotient_plan,
    solve_jax_linearized_newton_system,
    solve_matrix_free_newton_system,
    solve_sparse_newton_system,
    unpack_active_fields,
)
from .mesh import StructuredMesh
from .array_backend import use_jax_backend
from .metrics import StructuredMetrics
from .neutral_mixed import (
    _div_a_grad_perp_flows,
    _div_par_fvv_open,
    _div_par_k_grad_par_open,
    _div_par_mod_open,
    _grad_par_open,
)
from .open_field import (
    TargetBoundaryGeometry,
    apply_noflow_flow_guards,
    apply_noflow_scalar_guards,
    compute_full_ion_zero_current_sum_term,
    compute_electron_force_balance,
    compute_full_electron_sheath_boundary,
    compute_full_ion_sheath_boundary,
    compute_simple_ion_sheath_boundary,
    compute_zero_current_electron_sheath_potential,
    compute_target_recycling_sources,
    limit_free,
    prepare_electron_sheath_state,
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
    assemble_electron_parallel_force_terms as _assemble_electron_parallel_force_terms,
    assemble_electron_pressure_rhs_terms as _assemble_electron_pressure_rhs_terms,
    assemble_ion_rhs_terms as _assemble_ion_rhs_terms,
    assemble_neutral_rhs_terms as _assemble_neutral_rhs_terms,
)
from .recycling_source_accumulation import (
    add_species_sources as _add_species_sources,
    apply_species_source_overrides as _apply_species_source_overrides,
    zero_species_sources as _zero_species_sources,
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
from .recycling_fixed_residual import (
    build_fixed_bdf2_residual as _build_fixed_bdf2_residual,
    build_fixed_backward_euler_residual as _build_fixed_backward_euler_residual,
    build_fixed_host_rhs_bridge as _build_fixed_host_rhs_bridge,
    pack_fixed_state as _pack_fixed_state,
    unpack_fixed_state as _unpack_fixed_state,
)
from .recycling_progress import build_recycling_progress_details as _build_recycling_progress_details
from .recycling_1d_state import (
    DensityFeedbackTerms as _DensityFeedbackTerms,
    ElectronBoundaryResult as _ElectronBoundaryResult,
    FullSheathSettings as _FullSheathSettings,
    IonBoundaryResult as _IonBoundaryResult,
    Recycling1DHistoryResult,
    Recycling1DImplicitStepInfo,
    Recycling1DRhsResult,
    RecyclingProgressCallback,
    SimpleSheathSettings as _SimpleSheathSettings,
)

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
    pressure_source_overrides_are_total: bool = False,
    momentum_source_overrides: dict[str, np.ndarray] | None = None,
) -> Recycling1DRhsResult:
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        metrics=metrics,
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
        pressure_source_overrides_are_total=pressure_source_overrides_are_total,
        momentum_source_overrides=momentum_source_overrides,
        lower_target_geometry=runtime_model.lower_target_geometry,
        upper_target_geometry=runtime_model.upper_target_geometry,
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
    include_reaction_diagnostics: bool = True,
    density_source_overrides: dict[str, np.ndarray] | None = None,
    pressure_source_overrides: dict[str, np.ndarray] | None = None,
    pressure_source_overrides_are_total: bool = False,
    momentum_source_overrides: dict[str, np.ndarray] | None = None,
    lower_target_geometry: TargetBoundaryGeometry | None = None,
    upper_target_geometry: TargetBoundaryGeometry | None = None,
) -> Recycling1DRhsResult:
    pressure_sources = explicit_pressure_sources or {}
    pressure_source_override_names = (
        frozenset(pressure_source_overrides or ())
        if pressure_source_overrides_are_total
        else frozenset()
    )
    if pressure_source_overrides:
        use_jax_overrides = use_jax_backend(*pressure_source_overrides.values())
        pressure_source_array = jnp.asarray if use_jax_overrides else np.asarray
        pressure_source_dtype = jnp.float64 if use_jax_overrides else np.float64
        pressure_sources = {
            **pressure_sources,
            **{
                name: pressure_source_array(value, dtype=pressure_source_dtype)
                for name, value in pressure_source_overrides.items()
            },
        }
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    neutrals = tuple(sp for sp in species.values() if sp.charge == 0.0)
    electron = species["e"]
    electron_density = _electron_density(ions)

    density_source = _zero_species_sources(species)
    energy_source = _zero_species_sources(species)
    momentum_source = _zero_species_sources(species)
    diagnostics: dict[str, np.ndarray] = {}

    reaction_terms = _reaction_sources(
        config,
        species=species,
        electron_density=electron_density,
        dataset_scalars=dataset_scalars,
        include_diagnostics=include_reaction_diagnostics,
    )
    _add_species_sources(density_source, reaction_terms.density_source)
    _add_species_sources(energy_source, reaction_terms.energy_source)
    _add_species_sources(momentum_source, reaction_terms.momentum_source)
    diagnostics.update(reaction_terms.diagnostics)

    anomalous_terms = _apply_anomalous_diffusion(
        config,
        species=species,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
    )
    _add_species_sources(density_source, anomalous_terms.density_source)
    _add_species_sources(energy_source, anomalous_terms.energy_source)
    _add_species_sources(momentum_source, anomalous_terms.momentum_source)
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
    _add_species_sources(energy_source, ion_boundary.energy_source)

    collision_rates = _compute_collision_frequencies(
        config,
        species,
        prepared,
        dataset_scalars=dataset_scalars,
    )
    charge_exchange_rates = _charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=dataset_scalars,
    )
    neutral_ionisation_rates = _neutral_ionisation_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=dataset_scalars,
    )

    collision_terms = _apply_collision_closure(
        config,
        species,
        prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        collision_rates=collision_rates,
        cx_rates=charge_exchange_rates,
    )
    _add_species_sources(energy_source, collision_terms.energy_source)
    _add_species_sources(momentum_source, collision_terms.momentum_source)
    diagnostics.update(collision_terms.diagnostics)

    neutral_diffusion_terms = _apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        collision_rates=collision_rates,
        ionisation_rates=neutral_ionisation_rates,
        charge_exchange_rates=charge_exchange_rates,
    )
    _add_species_sources(density_source, neutral_diffusion_terms.density_source)
    _add_species_sources(energy_source, neutral_diffusion_terms.energy_source)
    _add_species_sources(momentum_source, neutral_diffusion_terms.momentum_source)
    diagnostics.update(neutral_diffusion_terms.diagnostics)

    _add_species_sources(energy_source, {"e": electron_boundary.energy_source})

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
        lower_geometry=lower_target_geometry,
        upper_geometry=upper_target_geometry,
    )
    _add_species_sources(density_source, recycling_terms.density_source)
    _add_species_sources(energy_source, recycling_terms.energy_source)
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
    _add_species_sources(density_source, feedback_terms.density_source)
    _add_species_sources(energy_source, feedback_terms.energy_source)
    diagnostics.update(feedback_terms.diagnostics)

    _apply_species_source_overrides(density_source, density_source_overrides)
    _apply_species_source_overrides(momentum_source, momentum_source_overrides)

    electron_force_terms = _assemble_electron_parallel_force_terms(
        electron_pressure=electron_boundary.pressure,
        electron_density=electron_boundary.density,
        electron_momentum_source=momentum_source["e"],
        ion_density={ion.name: prepared[ion.name].density for ion in ions},
        ion_charge={ion.name: ion.charge for ion in ions},
        ion_momentum_source={ion.name: momentum_source[ion.name] for ion in ions},
        mesh=mesh,
        metrics=metrics,
    )
    for ion in ions:
        momentum_source[ion.name] = electron_force_terms.ion_momentum_source[ion.name]
    electron_epar = electron_force_terms.epar
    use_jax_rhs = use_jax_backend(
        electron_boundary.velocity,
        electron_force_terms.epar,
        *(prepared[name].temperature for name in prepared),
    )
    rhs_array = jnp.asarray if use_jax_rhs else np.asarray
    rhs_dtype = jnp.float64 if use_jax_rhs else np.float64
    rhs_maximum = jnp.maximum if use_jax_rhs else np.maximum
    rhs_sqrt = jnp.sqrt if use_jax_rhs else np.sqrt
    diagnostics["Epar"] = rhs_array(electron_force_terms.epar, dtype=rhs_dtype)
    diagnostics["Ve"] = rhs_array(electron_boundary.velocity, dtype=rhs_dtype)

    variables: dict[str, np.ndarray] = {}
    variables[electron.density_name] = prepared["e"].density[None, ...]

    for ion in ions:
        ion_state = prepared[ion.name]
        temperature = ion_state.temperature
        fastest_wave = rhs_sqrt(rhs_maximum(temperature, 0.0) / ion.atomic_mass)
        explicit_pressure_source = pressure_sources.get(ion.name)
        if explicit_pressure_source is None:
            explicit_pressure_source = _explicit_pressure_source(
                config,
                ion.name,
                mesh=mesh,
                dataset_scalars=dataset_scalars,
            )
        ion_rhs_terms = _assemble_ion_rhs_terms(
            density_source=density_source[ion.name],
            explicit_pressure_source=explicit_pressure_source,
            momentum_source=momentum_source[ion.name],
            atomic_mass=ion.atomic_mass,
            density_floor=ion.density_floor,
            ion_state=ion_state,
            ion_velocity=ion_velocity[ion.name],
            fastest_wave=fastest_wave,
            mesh=mesh,
            metrics=metrics,
            energy_source=energy_source[ion.name],
        )

        variables[ion.density_name] = ion_state.density[None, ...]
        variables[ion.pressure_name] = ion_state.pressure[None, ...]
        variables[ion.momentum_name] = ion_state.momentum[None, ...]
        variables[f"SNV{ion.name}"] = momentum_source[ion.name][None, ...]
        variables[f"ddt({ion.density_name})"] = ion_rhs_terms.density_total[None, ...]
        variables[f"ddt({ion.pressure_name})"] = ion_rhs_terms.pressure_total[None, ...]
        variables[f"ddt({ion.momentum_name})"] = ion_rhs_terms.momentum_total[None, ...]

    electron_velocity = rhs_array(electron_boundary.velocity, dtype=rhs_dtype)
    electron_fastest_wave = rhs_sqrt(rhs_maximum(prepared["e"].temperature, 0.0) / electron.atomic_mass)
    electron_explicit_pressure_source = pressure_sources.get("e")
    if electron_explicit_pressure_source is None:
        electron_explicit_pressure_source = _explicit_pressure_source(
            config,
            "e",
            mesh=mesh,
            dataset_scalars=dataset_scalars,
        )
    electron_pressure_rhs_terms = _assemble_electron_pressure_rhs_terms(
        explicit_pressure_source=electron_explicit_pressure_source,
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
        fastest_wave = rhs_sqrt(rhs_maximum(temperature, 0.0) / neutral.atomic_mass)
        neutral_explicit_pressure_source = pressure_sources.get(neutral.name)
        if neutral_explicit_pressure_source is None:
            neutral_explicit_pressure_source = _explicit_pressure_source(
                config,
                neutral.name,
                mesh=mesh,
                dataset_scalars=dataset_scalars,
            )
        neutral_rhs_terms = _assemble_neutral_rhs_terms(
            density_source=density_source[neutral.name],
            explicit_pressure_source=neutral_explicit_pressure_source,
            momentum_source=momentum_source[neutral.name],
            atomic_mass=neutral.atomic_mass,
            density_floor=neutral.density_floor,
            neutral_state=neutral_state,
            neutral_velocity=neutral_state.velocity,
            fastest_wave=fastest_wave,
            mesh=mesh,
            metrics=metrics,
            energy_source=energy_source[neutral.name],
            include_energy_source=neutral.name not in pressure_source_override_names,
        )
        variables[neutral.density_name] = neutral_state.density[None, ...]
        variables[neutral.pressure_name] = neutral_state.pressure[None, ...]
        variables[neutral.momentum_name] = neutral_state.momentum[None, ...]
        variables[f"SNV{neutral.name}"] = momentum_source[neutral.name][None, ...]
        variables[f"ddt({neutral.density_name})"] = neutral_rhs_terms.density_total[None, ...]
        variables[f"ddt({neutral.pressure_name})"] = neutral_rhs_terms.pressure_total[None, ...]
        variables[f"ddt({neutral.momentum_name})"] = neutral_rhs_terms.momentum_total[None, ...]

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
    use_jax_electron_density = use_jax_backend(
        species["e"].density,
        *(prepared[ion.name].density for ion in ions),
    )
    electron_density_array = jnp.asarray if use_jax_electron_density else np.asarray
    electron_density_dtype = jnp.float64 if use_jax_electron_density else np.float64
    electron_density = (
        jnp.zeros_like(electron_density_array(species["e"].density, dtype=electron_density_dtype))
        if use_jax_electron_density
        else np.zeros_like(electron_density_array(species["e"].density, dtype=electron_density_dtype))
    )
    for ion in ions:
        electron_density = electron_density + ion.charge * electron_density_array(
            prepared[ion.name].density,
            dtype=electron_density_dtype,
        )

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
        use_jax_electron_boundary = use_jax_backend(
            prepared["e"].density,
            prepared["e"].temperature,
            prepared["e"].pressure,
            electron_velocity,
        )
        electron_boundary_array = jnp.asarray if use_jax_electron_boundary else np.asarray
        electron_boundary_dtype = jnp.float64 if use_jax_electron_boundary else np.float64
        electron_boundary_density = electron_boundary_array(prepared["e"].density, dtype=electron_boundary_dtype)
        electron_boundary_temperature = electron_boundary_array(prepared["e"].temperature, dtype=electron_boundary_dtype)
        electron_boundary_pressure = electron_boundary_array(prepared["e"].pressure, dtype=electron_boundary_dtype)
        electron_boundary_velocity = electron_boundary_array(electron_velocity, dtype=electron_boundary_dtype)
        electron_boundary_momentum = electron_boundary_array(
            species["e"].atomic_mass * prepared["e"].density * electron_velocity,
            dtype=electron_boundary_dtype,
        )
        electron_zero_source = (
            jnp.zeros_like(electron_boundary_density, dtype=jnp.float64)
            if use_jax_electron_boundary
            else np.zeros_like(electron_boundary_density, dtype=np.float64)
        )
        electron_boundary = _ElectronBoundaryResult(
            density=electron_boundary_density,
            temperature=electron_boundary_temperature,
            pressure=electron_boundary_pressure,
            velocity=electron_boundary_velocity,
            momentum=electron_boundary_momentum,
            energy_source=electron_zero_source,
        )
    use_jax_electron_prepared = use_jax_backend(
        electron_boundary.density,
        electron_boundary.pressure,
        electron_boundary.momentum,
    )
    electron_prepared_array = jnp.asarray if use_jax_electron_prepared else np.asarray
    electron_prepared_dtype = jnp.float64 if use_jax_electron_prepared else np.float64
    electron_prepared_maximum = jnp.maximum if use_jax_electron_prepared else np.maximum
    electron_boundary_density = electron_prepared_array(electron_boundary.density, dtype=electron_prepared_dtype)
    electron_boundary_momentum = electron_prepared_array(electron_boundary.momentum, dtype=electron_prepared_dtype)
    electron_reconstructed_velocity = electron_boundary_momentum / electron_prepared_maximum(
        species["e"].atomic_mass * _soft_floor(electron_boundary_density, species["e"].density_floor),
        1.0e-8,
    )
    prepared["e"] = _PreparedSpeciesState(
        density=electron_boundary.density,
        pressure=electron_boundary.pressure,
        temperature=electron_boundary.temperature,
        velocity=electron_prepared_array(electron_reconstructed_velocity, dtype=electron_prepared_dtype),
        momentum=electron_boundary_momentum,
        momentum_error=electron_prepared_array(
            species["e"].atomic_mass
            * electron_boundary_density
            * electron_reconstructed_velocity
            - electron_boundary_momentum,
            dtype=electron_prepared_dtype,
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
        use_jax_ion_boundary = use_jax_backend(*(prepared[ion.name].density for ion in ions))
        ion_boundary_array = jnp.asarray if use_jax_ion_boundary else np.asarray
        ion_boundary_dtype = jnp.float64 if use_jax_ion_boundary else np.float64
        ion_boundary = _IonBoundaryResult(
            density={ion.name: ion_boundary_array(prepared[ion.name].density, dtype=ion_boundary_dtype) for ion in ions},
            pressure={ion.name: ion_boundary_array(prepared[ion.name].pressure, dtype=ion_boundary_dtype) for ion in ions},
            temperature={
                ion.name: ion_boundary_array(prepared[ion.name].temperature, dtype=ion_boundary_dtype)
                for ion in ions
            },
            velocity={ion.name: ion_boundary_array(prepared[ion.name].velocity, dtype=ion_boundary_dtype) for ion in ions},
            momentum={ion.name: ion_boundary_array(prepared[ion.name].momentum, dtype=ion_boundary_dtype) for ion in ions},
            energy_source={
                ion.name: (
                    jnp.zeros_like(ion_boundary_array(prepared[ion.name].density, dtype=ion_boundary_dtype))
                    if use_jax_ion_boundary
                    else np.zeros_like(ion_boundary_array(prepared[ion.name].density, dtype=ion_boundary_dtype))
                )
                for ion in ions
            },
        )
    for ion in ions:
        use_jax_ion_prepared = use_jax_backend(
            ion_boundary.density[ion.name],
            ion_boundary.momentum[ion.name],
        )
        ion_prepared_array = jnp.asarray if use_jax_ion_prepared else np.asarray
        ion_prepared_dtype = jnp.float64 if use_jax_ion_prepared else np.float64
        ion_prepared_maximum = jnp.maximum if use_jax_ion_prepared else np.maximum
        ion_boundary_density = ion_prepared_array(ion_boundary.density[ion.name], dtype=ion_prepared_dtype)
        ion_boundary_momentum = ion_prepared_array(ion_boundary.momentum[ion.name], dtype=ion_prepared_dtype)
        ion_reconstructed_velocity = ion_boundary_momentum / ion_prepared_maximum(
            ion.atomic_mass * _soft_floor(ion_boundary_density, ion.density_floor),
            1.0e-8,
        )
        prepared[ion.name] = _PreparedSpeciesState(
            density=ion_boundary.density[ion.name],
            pressure=ion_boundary.pressure[ion.name],
            temperature=ion_boundary.temperature[ion.name],
            velocity=ion_prepared_array(ion_reconstructed_velocity, dtype=ion_prepared_dtype),
            momentum=ion_boundary_momentum,
            momentum_error=ion_prepared_array(
                ion.atomic_mass * ion_boundary_density * ion_reconstructed_velocity - ion_boundary_momentum,
                dtype=ion_prepared_dtype,
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
    use_jax = use_jax_backend(
        *(sp.density for sp in species.values()),
        *(state.density for state in prepared.values()),
        *((feedback_integrals or {}).values()),
    )
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    zeros_like = jnp.zeros_like if use_jax else np.zeros_like
    maximum = jnp.maximum if use_jax else np.maximum
    square = jnp.square if use_jax else np.square
    density_source = {name: zeros_like(sp.density, dtype=dtype) for name, sp in species.items()}
    energy_source = {name: zeros_like(sp.density, dtype=dtype) for name, sp in species.items()}
    diagnostics: dict[str, np.ndarray] = {}
    integral_rhs: dict[str, float] = {}
    integrals = feedback_integrals or {}

    for name, controller in controllers.items():
        upstream_density = array(prepared[name].density, dtype=dtype)[mesh.xstart, mesh.ystart, 0]
        error = controller.density_upstream - upstream_density
        stored_integral = array(integrals.get(name, 0.0), dtype=dtype)
        integrated_error = stored_integral
        if feedback_timestep is not None:
            previous_error = error if feedback_previous_errors is None else array(feedback_previous_errors.get(name, error), dtype=dtype)
            integrated_error = stored_integral + float(feedback_timestep) * 0.5 * (error + previous_error)
        if controller.density_integral_positive:
            integrated_error = maximum(integrated_error, 0.0)
        proportional_term = controller.density_controller_p * error
        integral_term = controller.density_controller_i * integrated_error
        source_multiplier = proportional_term + integral_term
        if controller.density_source_positive:
            source_multiplier = maximum(source_multiplier, 0.0)
        source = source_multiplier * array(controller.density_source_shape, dtype=dtype)
        density_source[name] += source
        velocity = array(prepared[name].velocity, dtype=dtype)
        energy_source[name] += 0.5 * species[name].atomic_mass * square(velocity) * source
        diagnostics[f"S{name}_feedback"] = array(source, dtype=dtype)
        diagnostics[f"density_feedback_src_mult_{name}"] = array(source_multiplier, dtype=dtype)
        diagnostics[f"density_feedback_src_p_{name}"] = array(proportional_term, dtype=dtype)
        diagnostics[f"density_feedback_src_i_{name}"] = array(integral_term, dtype=dtype)
        diagnostics[f"density_feedback_src_shape_{name}"] = array(controller.density_source_shape, dtype=dtype)
        integral_rhs[name] = error

    return _DensityFeedbackTerms(
        density_source=density_source,
        energy_source=energy_source,
        diagnostics=diagnostics,
        feedback_integral_rhs=integral_rhs,
    )


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
    use_jax = use_jax_backend(
        electron_pressure,
        electron_density,
        te,
        *(ion.density for ion in ions),
        *(ion.pressure for ion in ions),
        *(ion.momentum for ion in ions),
    )
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    zeros_like = jnp.zeros_like if use_jax else np.zeros_like
    maximum = jnp.maximum if use_jax else np.maximum
    sqrt = jnp.sqrt if use_jax else np.sqrt
    dx = array(metrics.dx, dtype=dtype)
    dz = array(metrics.dz, dtype=dtype)
    g22 = array(metrics.g_22, dtype=dtype)
    dy = array(metrics.dy, dtype=dtype)
    J = array(metrics.J, dtype=dtype)
    te = array(te, dtype=dtype)
    boundary_density: dict[str, np.ndarray] = {}
    boundary_pressure: dict[str, np.ndarray] = {}
    boundary_temperature: dict[str, np.ndarray] = {}
    velocity: dict[str, np.ndarray] = {}
    momentum: dict[str, np.ndarray] = {}
    energy_source: dict[str, np.ndarray] = {ion.name: zeros_like(ion.density, dtype=dtype) for ion in ions}

    def _copy_field(value):
        field = array(value, dtype=dtype)
        return field if use_jax else np.array(field, dtype=np.float64, copy=True)

    def _set_y(field, j: int, value):
        if use_jax:
            return field.at[:, j, :].set(array(value, dtype=dtype))
        field[:, j, :] = np.asarray(value, dtype=np.float64)
        return field

    def _add_y(field, j: int, value):
        if use_jax:
            return field.at[:, j, :].add(array(value, dtype=dtype))
        field[:, j, :] += np.asarray(value, dtype=np.float64)
        return field

    for ion in ions:
        density = _copy_field(ion.density)
        pressure = _copy_field(ion.pressure)
        temperature = _safe_temperature(pressure, density, ion.density_floor)
        vel = array(ion.momentum, dtype=dtype) / maximum(
            ion.atomic_mass * _soft_floor(density, ion.density_floor),
            1.0e-8,
        )
        if ion.noflow_lower_y:
            density = _copy_field(apply_noflow_scalar_guards(density, mesh=mesh, lower_y=True, upper_y=False))
            temperature = _copy_field(apply_noflow_scalar_guards(temperature, mesh=mesh, lower_y=True, upper_y=False))
            pressure = _copy_field(apply_noflow_scalar_guards(pressure, mesh=mesh, lower_y=True, upper_y=False))
            vel = _copy_field(apply_noflow_flow_guards(vel, mesh=mesh, lower_y=True, upper_y=False))

        momentum_field = _copy_field(ion.momentum)
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
            density = _set_y(
                density,
                jp,
                limit_free(ni_m, ni_i, 0.0 if simple_settings is None else simple_settings.density_boundary_mode),
            )
            temperature = _set_y(
                temperature,
                jp,
                limit_free(
                    temperature[:, jm, :],
                    temperature[:, j, :],
                    0.0 if simple_settings is None else simple_settings.temperature_boundary_mode,
                ),
            )
            pressure = _set_y(
                pressure,
                jp,
                limit_free(
                    pressure[:, jm, :],
                    pressure[:, j, :],
                    0.0 if simple_settings is None else simple_settings.pressure_boundary_mode,
                ),
            )

            nisheath = 0.5 * (density[:, jp, :] + density[:, j, :])
            tesheath = maximum(0.5 * (te[:, jp, :] + te[:, j, :]) if jp < te.shape[1] else 0.5 * (te[:, jm, :] + te[:, j, :]), 1.0e-5)
            tisheath = maximum(0.5 * (temperature[:, jp, :] + temperature[:, j, :]), 1.0e-5)
            if simple_settings is not None:
                da = (
                    (J[:, j, :] + J[:, jp, :]) / (sqrt(g22[:, j, :]) + sqrt(g22[:, jp, :]))
                    * 0.5
                    * (dx[:, j, :] + dx[:, jp, :])
                    * 0.5
                    * (dz[:, j, :] + dz[:, jp, :])
                )
                dv = dx[:, j, :] * dy[:, j, :] * dz[:, j, :] * J[:, j, :]
                sheath = compute_simple_ion_sheath_boundary(
                    sheath_density=nisheath,
                    sheath_temperature=tisheath,
                    electron_sheath_temperature=tesheath,
                    interior_velocity=vel[:, j, :],
                    interior_momentum=momentum_field[:, j, :],
                    atomic_mass=ion.atomic_mass,
                    charge=ion.charge,
                    gamma_i=simple_settings.gamma_i,
                    sheath_ion_polytropic=simple_settings.sheath_ion_polytropic,
                    direction=1.0,
                    no_flow=simple_settings.no_flow,
                    source_scale=da / maximum(dv, 1.0e-30),
                )
                vel = _set_y(vel, jp, sheath.guard_velocity)
                momentum_field = _set_y(momentum_field, jp, sheath.guard_momentum)
                energy_source[ion.name] = _add_y(energy_source[ion.name], j, sheath.energy_source_delta)
            else:
                nesheath = (
                    0.5 * (electron_density[:, jp, :] + electron_density[:, j, :])
                    if jp < electron_density.shape[1]
                    else 0.5 * (electron_density[:, jm, :] + electron_density[:, j, :])
                )
                grad_ne = electron_density[:, j, :] - nesheath
                grad_ni = density[:, j, :] - nisheath
                source_scale = (J[:, j, :] + J[:, jp, :]) / (sqrt(g22[:, j, :]) + sqrt(g22[:, jp, :])) / (
                    dy[:, j, :] * J[:, j, :]
                )
                sheath = compute_full_ion_sheath_boundary(
                    sheath_density=nisheath,
                    sheath_temperature=tisheath,
                    electron_sheath_density=nesheath,
                    electron_sheath_temperature=tesheath,
                    electron_density_gradient=grad_ne,
                    ion_density_gradient=grad_ni,
                    interior_velocity=vel[:, j, :],
                    interior_momentum=momentum_field[:, j, :],
                    atomic_mass=ion.atomic_mass,
                    charge=ion.charge,
                    direction=1.0,
                    source_scale=source_scale,
                )
                vel = _set_y(vel, jp, sheath.guard_velocity)
                momentum_field = _set_y(momentum_field, jp, sheath.guard_momentum)
                energy_source[ion.name] = _add_y(energy_source[ion.name], j, sheath.energy_source_delta)

        if mesh.has_lower_y_target and lower_y_enabled:
            j = mesh.ystart
            jm = j - 1
            jp = j + 1
            ni_i = density[:, j, :]
            ni_p = density[:, jp, :]
            density = _set_y(
                density,
                jm,
                limit_free(ni_p, ni_i, 0.0 if simple_settings is None else simple_settings.density_boundary_mode),
            )
            temperature = _set_y(
                temperature,
                jm,
                limit_free(
                    temperature[:, jp, :],
                    temperature[:, j, :],
                    0.0 if simple_settings is None else simple_settings.temperature_boundary_mode,
                ),
            )
            pressure = _set_y(
                pressure,
                jm,
                limit_free(
                    pressure[:, jp, :],
                    pressure[:, j, :],
                    0.0 if simple_settings is None else simple_settings.pressure_boundary_mode,
                ),
            )

            nisheath = 0.5 * (density[:, jm, :] + density[:, j, :])
            tesheath = maximum(0.5 * (te[:, jm, :] + te[:, j, :]), 1.0e-5)
            tisheath = maximum(0.5 * (temperature[:, jm, :] + temperature[:, j, :]), 1.0e-5)
            if simple_settings is not None:
                da = (
                    (J[:, j, :] + J[:, jm, :]) / (sqrt(g22[:, j, :]) + sqrt(g22[:, jm, :]))
                    * 0.5
                    * (dx[:, j, :] + dx[:, jm, :])
                    * 0.5
                    * (dz[:, j, :] + dz[:, jm, :])
                )
                dv = dx[:, j, :] * dy[:, j, :] * dz[:, j, :] * J[:, j, :]
                sheath = compute_simple_ion_sheath_boundary(
                    sheath_density=nisheath,
                    sheath_temperature=tisheath,
                    electron_sheath_temperature=tesheath,
                    interior_velocity=vel[:, j, :],
                    interior_momentum=momentum_field[:, j, :],
                    atomic_mass=ion.atomic_mass,
                    charge=ion.charge,
                    gamma_i=simple_settings.gamma_i,
                    sheath_ion_polytropic=simple_settings.sheath_ion_polytropic,
                    direction=-1.0,
                    no_flow=simple_settings.no_flow,
                    source_scale=da / maximum(dv, 1.0e-30),
                )
                vel = _set_y(vel, jm, sheath.guard_velocity)
                momentum_field = _set_y(momentum_field, jm, sheath.guard_momentum)
                energy_source[ion.name] = _add_y(energy_source[ion.name], j, sheath.energy_source_delta)
            else:
                nesheath = 0.5 * (electron_density[:, jm, :] + electron_density[:, j, :])
                grad_ne = electron_density[:, j, :] - nesheath
                grad_ni = density[:, j, :] - nisheath
                source_scale = (J[:, j, :] + J[:, jm, :]) / (sqrt(g22[:, j, :]) + sqrt(g22[:, jm, :])) / (
                    dy[:, j, :] * J[:, j, :]
                )
                sheath = compute_full_ion_sheath_boundary(
                    sheath_density=nisheath,
                    sheath_temperature=tisheath,
                    electron_sheath_density=nesheath,
                    electron_sheath_temperature=tesheath,
                    electron_density_gradient=grad_ne,
                    ion_density_gradient=grad_ni,
                    interior_velocity=vel[:, j, :],
                    interior_momentum=momentum_field[:, j, :],
                    atomic_mass=ion.atomic_mass,
                    charge=ion.charge,
                    direction=-1.0,
                    source_scale=source_scale,
                )
                vel = _set_y(vel, jm, sheath.guard_velocity)
                momentum_field = _set_y(momentum_field, jm, sheath.guard_momentum)
                energy_source[ion.name] = _add_y(energy_source[ion.name], j, sheath.energy_source_delta)

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
    prepared_electron = prepare_electron_sheath_state(
        electron_pressure=electron_pressure,
        electron_density=electron_density,
        electron_velocity=electron_velocity,
        electron_mass=electron_mass,
        electron_density_floor=electron_density_floor,
        mesh=mesh,
        lower_y=settings.lower_y,
        upper_y=settings.upper_y,
    )
    use_jax = use_jax_backend(
        prepared_electron.density,
        prepared_electron.pressure,
        prepared_electron.temperature,
        prepared_electron.velocity,
        prepared_electron.momentum,
    )
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    zeros_like = jnp.zeros_like if use_jax else np.zeros_like
    sqrt = jnp.sqrt if use_jax else np.sqrt
    density = array(prepared_electron.density, dtype=dtype)
    pressure = array(prepared_electron.pressure, dtype=dtype)
    temperature = array(prepared_electron.temperature, dtype=dtype)
    velocity = array(prepared_electron.velocity, dtype=dtype)
    momentum = array(prepared_electron.momentum, dtype=dtype)
    if not use_jax:
        density = np.array(density, dtype=np.float64, copy=True)
        pressure = np.array(pressure, dtype=np.float64, copy=True)
        temperature = np.array(temperature, dtype=np.float64, copy=True)
        velocity = np.array(velocity, dtype=np.float64, copy=True)
        momentum = np.array(momentum, dtype=np.float64, copy=True)
    me = 1.0 / 1836.0
    g22 = array(metrics.g_22, dtype=dtype)
    dy = array(metrics.dy, dtype=dtype)
    J = array(metrics.J, dtype=dtype)
    phi = zeros_like(density, dtype=dtype)
    energy_source = zeros_like(density, dtype=dtype)
    secondary_coef = float(settings.secondary_electron_coef)
    sin_alpha = array(settings.sin_alpha, dtype=dtype)
    wall_potential = array(settings.wall_potential, dtype=dtype)
    electron_adiabatic = 5.0 / 3.0

    def _set_y(field, j: int, value):
        if use_jax:
            return field.at[:, j, :].set(array(value, dtype=dtype))
        field[:, j, :] = np.asarray(value, dtype=np.float64)
        return field

    def _add_y(field, j: int, value):
        if use_jax:
            return field.at[:, j, :].add(array(value, dtype=dtype))
        field[:, j, :] += np.asarray(value, dtype=np.float64)
        return field

    def _full_ion_sum_at_boundary(*, j: int, neighbor: int) -> np.ndarray:
        total = zeros_like(density[:, j, :], dtype=dtype)
        for ion in ions:
            ion_state = prepared_ions[ion.name]
            ti = array(ion_state.temperature, dtype=dtype)
            ni = array(ion_state.density, dtype=dtype)
            total = total + array(
                compute_full_ion_zero_current_sum_term(
                    ion_density_boundary=ni[:, j, :],
                    ion_density_neighbor=ni[:, neighbor, :],
                    electron_density_boundary=density[:, j, :],
                    electron_density_neighbor=density[:, neighbor, :],
                    ion_temperature_boundary=ti[:, j, :],
                    electron_temperature_boundary=temperature[:, j, :],
                    electron_density_gradient=density[:, neighbor, :] - density[:, j, :],
                    ion_density_gradient=ni[:, neighbor, :] - ni[:, j, :],
                    sin_alpha_neighbor=sin_alpha[:, neighbor, :],
                    atomic_mass=ion.atomic_mass,
                    charge=ion.charge,
                ),
                dtype=dtype,
            )
        return total

    def _set_zero_current_phi(phi_field, *, j: int, neighbor: int, ghost: int):
        boundary_phi = array(
            compute_zero_current_electron_sheath_potential(
                electron_temperature_boundary=temperature[:, j, :],
                ion_current_sum=_full_ion_sum_at_boundary(j=j, neighbor=neighbor),
                wall_potential_boundary=wall_potential[:, j, :],
                electron_thermal_mass=me,
                secondary_electron_coef=secondary_coef,
            ),
            dtype=dtype,
        )
        phi_field = _set_y(phi_field, j, boundary_phi)
        phi_field = _set_y(phi_field, neighbor, boundary_phi)
        phi_field = _set_y(phi_field, ghost, boundary_phi)
        return phi_field

    if mesh.has_upper_y_target and settings.upper_y:
        j = mesh.yend
        jp = j + 1
        jm = j - 1
        density = _set_y(density, jp, limit_free(density[:, jm, :], density[:, j, :], 0))
        temperature = _set_y(temperature, jp, limit_free(temperature[:, jm, :], temperature[:, j, :], 0))
        pressure = _set_y(pressure, jp, limit_free(pressure[:, jm, :], pressure[:, j, :], 0))
        phi = _set_zero_current_phi(phi, j=j, neighbor=jm, ghost=jp)
        phi = _set_y(phi, jp, 2.0 * phi[:, j, :] - phi[:, jm, :])
        phi_wall = wall_potential[:, j, :]
        phisheath_raw = 0.5 * (phi[:, jp, :] + phi[:, j, :])
        tesheath = 0.5 * (temperature[:, jp, :] + temperature[:, j, :])
        nesheath = 0.5 * (density[:, jp, :] + density[:, j, :])
        source_scale = (J[:, j, :] + J[:, jp, :]) / (sqrt(g22[:, j, :]) + sqrt(g22[:, jp, :])) / (
            dy[:, j, :] * J[:, j, :]
        )
        sheath = compute_full_electron_sheath_boundary(
            sheath_density=nesheath,
            sheath_temperature=tesheath,
            sheath_potential_raw=phisheath_raw,
            wall_potential=phi_wall,
            interior_velocity=velocity[:, j, :],
            interior_momentum=momentum[:, j, :],
            electron_mass=electron_mass,
            electron_thermal_mass=me,
            secondary_electron_coef=secondary_coef,
            electron_adiabatic=electron_adiabatic,
            direction=1.0,
            floor_potential=settings.floor_potential,
            source_scale=source_scale,
        )
        velocity = _set_y(velocity, jp, sheath.guard_velocity)
        momentum = _set_y(momentum, jp, sheath.guard_momentum)
        energy_source = _add_y(energy_source, j, sheath.energy_source_delta)

    if mesh.has_lower_y_target and settings.lower_y:
        j = mesh.ystart
        jm = j - 1
        jp = j + 1
        density = _set_y(density, jm, limit_free(density[:, jp, :], density[:, j, :], 0))
        temperature = _set_y(temperature, jm, limit_free(temperature[:, jp, :], temperature[:, j, :], 0))
        pressure = _set_y(pressure, jm, limit_free(pressure[:, jp, :], pressure[:, j, :], 0))
        phi = _set_zero_current_phi(phi, j=j, neighbor=jp, ghost=jm)
        phi = _set_y(phi, jm, 2.0 * phi[:, j, :] - phi[:, jp, :])
        phi_wall = wall_potential[:, j, :]
        phisheath_raw = 0.5 * (phi[:, jm, :] + phi[:, j, :])
        tesheath = 0.5 * (temperature[:, jm, :] + temperature[:, j, :])
        nesheath = 0.5 * (density[:, jm, :] + density[:, j, :])
        source_scale = (J[:, j, :] + J[:, jm, :]) / (sqrt(g22[:, j, :]) + sqrt(g22[:, jm, :])) / (
            dy[:, j, :] * J[:, j, :]
        )
        sheath = compute_full_electron_sheath_boundary(
            sheath_density=nesheath,
            sheath_temperature=tesheath,
            sheath_potential_raw=phisheath_raw,
            wall_potential=phi_wall,
            interior_velocity=velocity[:, j, :],
            interior_momentum=momentum[:, j, :],
            electron_mass=electron_mass,
            electron_thermal_mass=me,
            secondary_electron_coef=secondary_coef,
            electron_adiabatic=electron_adiabatic,
            direction=-1.0,
            floor_potential=settings.floor_potential,
            source_scale=source_scale,
        )
        velocity = _set_y(velocity, jm, sheath.guard_velocity)
        momentum = _set_y(momentum, jm, sheath.guard_momentum)
        energy_source = _add_y(energy_source, j, sheath.energy_source_delta)
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
        metrics=metrics,
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

    if solver_mode in {
        "adaptive_bdf",
        "adaptive_bdf_sparse_jvp",
        "adaptive_bdf_jax_linearized",
        "adaptive_bdf_jax_linearized_lineax",
    }:
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
            step_solver_mode=_adaptive_bdf_step_solver_mode(solver_mode),
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
    run_started_at = time.perf_counter()
    interval_started_at = run_started_at

    for interval_index in range(steps):
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
            details, interval_started_at = _build_recycling_progress_details(
                interval_index=interval_index + 1,
                steps=steps,
                solver_mode=solver_mode,
                accepted_dt=float(timestep),
                stored_states=len(next(iter(variable_history.values()))),
                output_timestep=timestep,
                run_started_at=run_started_at,
                interval_started_at=interval_started_at,
            )
            progress_callback(details)

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
    run_started_at = time.perf_counter()
    interval_started_at = run_started_at

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
            details, interval_started_at = _build_recycling_progress_details(
                interval_index=interval_index + 1,
                steps=steps,
                solver_mode="continuation",
                accepted_dt=float(suggested_dt),
                stored_states=len(next(iter(variable_history.values()))),
                output_timestep=timestep,
                run_started_at=run_started_at,
                interval_started_at=interval_started_at,
            )
            progress_callback(details)

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
    run_started_at = time.perf_counter()
    interval_started_at = run_started_at

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
            details, interval_started_at = _build_recycling_progress_details(
                interval_index=interval_index + 1,
                steps=steps,
                solver_mode="adaptive_be",
                accepted_dt=float(suggested_dt),
                stored_states=len(next(iter(variable_history.values()))),
                output_timestep=timestep,
                run_started_at=run_started_at,
                interval_started_at=interval_started_at,
            )
            progress_callback(details)

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
    step_solver_mode: str = "sparse",
) -> Recycling1DHistoryResult:
    current_fields = {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()}
    current_integrals = {name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names}
    previous_fields: dict[str, np.ndarray] | None = None
    previous_integrals: dict[str, float] | None = None
    previous_dt: float | None = None
    variable_history = {name: [np.asarray(current_fields[name], dtype=np.float64)] for name in field_names}
    feedback_history = {name: [np.asarray(current_integrals[name], dtype=np.float64)] for name in feedback_names}
    suggested_dt = _initial_recycling_continuation_dt(runtime_model, timestep=timestep)
    run_started_at = time.perf_counter()
    interval_started_at = run_started_at

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
            step_solver_mode=step_solver_mode,
        )
        for name in field_names:
            variable_history[name].append(np.asarray(current_fields[name], dtype=np.float64))
        for name in feedback_names:
            feedback_history[name].append(np.asarray(current_integrals[name], dtype=np.float64))
        if progress_callback is not None:
            details, interval_started_at = _build_recycling_progress_details(
                interval_index=interval_index + 1,
                steps=steps,
                solver_mode="adaptive_bdf",
                accepted_dt=float(suggested_dt),
                stored_states=len(next(iter(variable_history.values()))),
                output_timestep=timestep,
                run_started_at=run_started_at,
                interval_started_at=interval_started_at,
            )
            progress_callback(details)

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
    step_solver_mode: str = "sparse",
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
                step_solver_mode=step_solver_mode,
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
                solver_mode=step_solver_mode,
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
                solver_mode=step_solver_mode,
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


def _adaptive_bdf_step_solver_mode(history_solver_mode: str) -> str:
    if history_solver_mode == "adaptive_bdf":
        return "sparse"
    if history_solver_mode == "adaptive_bdf_sparse_jvp":
        return "sparse_jvp"
    if history_solver_mode == "adaptive_bdf_jax_linearized":
        return "jax_linearized"
    if history_solver_mode == "adaptive_bdf_jax_linearized_lineax":
        return "jax_linearized_lineax"
    raise ValueError(f"Unsupported adaptive BDF solver mode {history_solver_mode!r}.")


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
    step_solver_mode: str = "sparse",
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
        solver_mode=step_solver_mode,
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
        solver_mode=step_solver_mode,
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
        solver_mode=step_solver_mode,
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
        squared_terms.append(((half - full) / scale).ravel())
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
    minimum_dt = max(float(output_timestep) / 4096.0, 1.0e-6)
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
        metrics=metrics,
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

    def packed_rhs(state_fields: dict[str, object], state_integrals: dict[str, object]) -> object:
        return _compute_recycling_1d_packed_rhs(
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

    fixed_rhs = _build_fixed_host_rhs_bridge(
        packed_rhs,
        layout=layout,
        base_feedback_integrals=feedback_integrals,
    )
    fixed_residual = _build_fixed_backward_euler_residual(
        fixed_rhs,
        layout=layout,
        previous_packed_state=packed_previous,
        timestep=timestep,
    )

    def residual(packed_state: object) -> object:
        return fixed_residual(packed_state)

    if solver_mode in {"sparse", "sparse_jvp"}:
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
            jacobian_mode="jvp" if solver_mode == "sparse_jvp" else _resolve_recycling_sparse_jacobian_mode(),
            jvp_batch_size=_resolve_recycling_jvp_batch_size(),
        )
    elif solver_mode == "matrix_free":
        solved, info = solve_matrix_free_newton_system(
            residual,
            packed_previous,
            active_shape=(packed_previous.size,),
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
    elif solver_mode in {"jax_linearized", "jax_linearized_lineax"}:
        solved, info = solve_jax_linearized_newton_system(
            residual,
            packed_initial_guess,
            active_shape=(packed_previous.size,),
            residual_tolerance=residual_tolerance,
            step_tolerance=1.0e-11,
            max_nonlinear_iterations=max_nonlinear_iterations,
            linear_restart=20,
            linear_maxiter=20,
            linear_solver_backend=(
                "lineax_gmres"
                if solver_mode == "jax_linearized_lineax"
                else _resolve_recycling_jax_linear_solver_backend()
            ),
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
        metrics=metrics,
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

    def packed_rhs(state_fields: dict[str, object], state_integrals: dict[str, object]) -> object:
        return _compute_recycling_1d_packed_rhs(
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

    fixed_rhs = _build_fixed_host_rhs_bridge(
        packed_rhs,
        layout=layout,
        base_feedback_integrals=feedback_integrals,
    )
    fixed_residual = _build_fixed_bdf2_residual(
        fixed_rhs,
        layout=layout,
        previous_packed_state=packed_previous,
        previous_previous_packed_state=packed_previous_previous,
        timestep=timestep,
    )

    def residual(packed_state: object) -> object:
        return fixed_residual(packed_state)

    if solver_mode in {"sparse", "sparse_jvp"}:
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
            jacobian_mode="jvp" if solver_mode == "sparse_jvp" else _resolve_recycling_sparse_jacobian_mode(),
            jvp_batch_size=_resolve_recycling_jvp_batch_size(),
        )
    elif solver_mode == "matrix_free":
        solved, info = solve_matrix_free_newton_system(
            residual,
            packed_previous,
            active_shape=(packed_previous.size,),
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
    elif solver_mode in {"jax_linearized", "jax_linearized_lineax"}:
        solved, info = solve_jax_linearized_newton_system(
            residual,
            packed_initial_guess,
            active_shape=(packed_previous.size,),
            residual_tolerance=residual_tolerance,
            step_tolerance=1.0e-11,
            max_nonlinear_iterations=max_nonlinear_iterations,
            linear_restart=20,
            linear_maxiter=20,
            linear_solver_backend=(
                "lineax_gmres"
                if solver_mode == "jax_linearized_lineax"
                else _resolve_recycling_jax_linear_solver_backend()
            ),
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
    difference_plan = prepare_sparse_difference_quotient_plan(
        sparsity=sparsity,
        color_groups=color_groups,
        sparsity_csc=sparsity_csc,
    )
    rhs_cache_time: float | None = None
    rhs_cache_state: np.ndarray | None = None
    rhs_cache_value: np.ndarray | None = None
    rhs_evaluation_count = 0
    rhs_cache_hit_count = 0
    jacobian_callback_count = 0
    jacobian_parallel_workers = _resolve_recycling_bdf_jacobian_parallel_workers()

    def packed_rhs(state_fields: dict[str, object], state_integrals: dict[str, object]) -> object:
        use_jax_state = use_jax_backend(
            *(state_fields[name] for name in state_fields),
            *(state_integrals[name] for name in state_integrals),
        )
        return _compute_recycling_1d_packed_rhs(
            config,
            state_fields,
            sanitize_fields=not use_jax_state,
            feedback_integrals=state_integrals,
            field_names=field_names,
            feedback_names=feedback_names,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            runtime_model=runtime_model,
            layout=layout,
        )

    fixed_rhs = _build_fixed_host_rhs_bridge(
        packed_rhs,
        layout=layout,
        base_feedback_integrals=current_integrals,
    )

    def _evaluate_rhs(_time: float, packed_state: np.ndarray) -> np.ndarray:
        nonlocal rhs_evaluation_count
        rhs_evaluation_count += 1
        fixed_state = _unpack_fixed_state(packed_state, layout=layout)
        return np.asarray(_pack_fixed_state(fixed_rhs(fixed_state)), dtype=np.float64)

    def rhs(_time: float, packed_state: np.ndarray) -> np.ndarray:
        nonlocal rhs_cache_hit_count, rhs_cache_time, rhs_cache_state, rhs_cache_value
        packed_array = np.asarray(packed_state, dtype=np.float64)
        if (
            rhs_cache_time is not None
            and rhs_cache_value is not None
            and rhs_cache_state is not None
            and float(_time) == rhs_cache_time
            and packed_array.shape == rhs_cache_state.shape
            and np.array_equal(packed_array, rhs_cache_state)
        ):
            rhs_cache_hit_count += 1
            return np.array(rhs_cache_value, dtype=np.float64, copy=True)
        value = np.asarray(_evaluate_rhs(_time, packed_array), dtype=np.float64)
        rhs_cache_time = float(_time)
        rhs_cache_state = np.array(packed_array, dtype=np.float64, copy=True)
        rhs_cache_value = np.array(value, dtype=np.float64, copy=True)
        return value

    def jacobian(_time: float, packed_state: np.ndarray):
        nonlocal jacobian_callback_count
        jacobian_callback_count += 1
        rhs_value = rhs(_time, packed_state)
        return build_sparse_difference_quotient_jacobian(
            lambda state: _evaluate_rhs(_time, state),
            packed_state,
            base_residual=rhs_value,
            sparsity=sparsity,
            color_groups=color_groups,
            sparsity_csc=sparsity_csc,
            difference_plan=difference_plan,
            parallel_workers=jacobian_parallel_workers,
        )

    run_started_at = time.perf_counter()
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
    solve_finished_at = time.perf_counter()
    average_interval_seconds = (
        max(solve_finished_at - run_started_at, 0.0) / float(max(steps, 1))
        if steps > 0
        else 0.0
    )

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
            details, _ = _build_recycling_progress_details(
                interval_index=column,
                steps=steps,
                solver_mode="bdf",
                accepted_dt=float(timestep),
                stored_states=column + 1,
                output_timestep=timestep,
                run_started_at=run_started_at,
                interval_started_at=solve_finished_at - average_interval_seconds,
                now=solve_finished_at,
                live_progress=False,
            )
            progress_callback(details)

    return Recycling1DHistoryResult(
        variable_history={name: np.stack(history, axis=0) for name, history in variable_history.items()},
        feedback_integral_history={name: np.stack(history, axis=0) for name, history in feedback_history.items()},
        diagnostics={
            "bdf_rhs_evaluation_count": int(rhs_evaluation_count),
            "bdf_rhs_cache_hit_count": int(rhs_cache_hit_count),
            "bdf_jacobian_callback_count": int(jacobian_callback_count),
            "bdf_jacobian_parallel_workers": int(jacobian_parallel_workers),
        },
    )


def _resolve_recycling_bdf_jacobian_parallel_workers() -> int:
    env_value = os.environ.get("JAX_DRB_FD_JACOBIAN_THREADS")
    if env_value is None:
        return 1
    try:
        return max(1, int(env_value))
    except ValueError:
        return 1


def _resolve_recycling_sparse_jacobian_mode() -> str:
    env_value = os.environ.get("JAX_DRB_RECYCLING_JACOBIAN_MODE", "fd").strip().lower()
    aliases = {
        "finite_difference": "fd",
        "finite-difference": "fd",
        "difference_quotient": "fd",
        "difference-quotient": "fd",
        "jvp": "jvp",
        "autodiff": "jvp",
        "jax": "jvp",
    }
    resolved = aliases.get(env_value, env_value)
    return resolved if resolved in {"fd", "jvp"} else "fd"


def _resolve_recycling_jvp_batch_size() -> int | None:
    env_value = os.environ.get("JAX_DRB_RECYCLING_JVP_BATCH_SIZE")
    if env_value is None or not env_value.strip():
        return None
    try:
        return max(1, int(env_value))
    except ValueError:
        return None


def _resolve_recycling_jax_linear_solver_backend() -> str:
    env_value = os.environ.get("JAX_DRB_RECYCLING_JAX_LINEAR_SOLVER", "jax_gmres").strip().lower()
    aliases = {
        "jax": "jax_gmres",
        "jax_scipy": "jax_gmres",
        "gmres": "jax_gmres",
        "jax_gmres": "jax_gmres",
        "lineax": "lineax_gmres",
        "lineax_gmres": "lineax_gmres",
    }
    return aliases.get(env_value, "jax_gmres")


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
        metrics=metrics,
        dataset_scalars=dataset_scalars,
    )
    if sanitize_fields:
        sanitized_fields = _sanitize_recycling_fields(config, fields)
    elif use_jax_backend(*(fields[name] for name in fields)):
        sanitized_fields = {name: jnp.asarray(value, dtype=jnp.float64) for name, value in fields.items()}
    else:
        sanitized_fields = {name: np.asarray(value, dtype=np.float64) for name, value in fields.items()}
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
        lower_target_geometry=runtime_model.lower_target_geometry,
        upper_target_geometry=runtime_model.upper_target_geometry,
        preserve_dump_target_state=runtime_model.preserve_dump_target_state,
        preserve_dump_ion_target_state_only=runtime_model.preserve_dump_ion_target_state_only,
        include_reaction_diagnostics=False,
    )
    active_slices = layout.active_slices if layout is not None else _recycling_active_domain_slices(mesh)
    use_jax_result = use_jax_backend(*(result.variables[f"ddt({name})"] for name in field_names))
    rhs_array = jnp.asarray if use_jax_result else np.asarray
    rhs_dtype = jnp.float64 if use_jax_result else np.float64
    concatenate = jnp.concatenate if use_jax_result else np.concatenate
    empty = jnp.array([], dtype=jnp.float64) if use_jax_result else np.array([], dtype=np.float64)
    pieces = [
        rhs_array(result.variables[f"ddt({name})"][0][active_slices], dtype=rhs_dtype).ravel()
        for name in field_names
    ]
    pieces.extend(rhs_array(result.feedback_integral_rhs.get(name, 0.0), dtype=rhs_dtype).reshape(1) for name in feedback_names)
    return concatenate(pieces) if pieces else empty


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
        tuple(fields[name] for name in field_names),
        active_slices=(layout.active_slices if layout is not None else _recycling_active_domain_slices(mesh)),
    )
    if not feedback_names:
        return field_block
    if use_jax_backend(field_block, *(feedback_integrals.get(name, 0.0) for name in feedback_names)):
        scalar_block = jnp.asarray([feedback_integrals.get(name, 0.0) for name in feedback_names], dtype=jnp.float64)
        return jnp.concatenate([field_block, scalar_block])
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
    use_jax = use_jax_backend(packed, *(field_templates[name] for name in field_names))
    packed_array = jnp.asarray(packed, dtype=jnp.float64) if use_jax else np.asarray(packed, dtype=np.float64)
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
    restored_integrals = {name: value if use_jax_backend(value) else float(value) for name, value in feedback_integrals.items()}
    for index, name in enumerate(feedback_names):
        restored_integrals[name] = scalar_block[index] if use_jax else float(scalar_block[index])
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
        lower_target_geometry=runtime_model.lower_target_geometry,
        upper_target_geometry=runtime_model.upper_target_geometry,
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
    use_jax = use_jax_backend(packed_state, previous_packed_state, rhs_fields)
    previous_array = jnp.asarray(previous_packed_state, dtype=jnp.float64) if use_jax else np.asarray(previous_packed_state, dtype=np.float64)
    packed_array = jnp.asarray(packed_state, dtype=jnp.float64) if use_jax else np.asarray(packed_state, dtype=np.float64)
    rhs_array = jnp.asarray(rhs_fields, dtype=jnp.float64) if use_jax else np.asarray(rhs_fields, dtype=np.float64)
    field_size = previous_array.size - len(feedback_names)
    field_rhs = rhs_array[:field_size]
    field_block = backward_euler_residual(
        packed_array[:field_size],
        previous_array[:field_size],
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
    if not feedback_names:
        return field_block
    return jnp.concatenate([field_block, controller_block]) if use_jax else np.concatenate([field_block, controller_block])


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
    use_jax = use_jax_backend(packed_state, previous_packed_state, previous_previous_packed_state, rhs_fields)
    previous_array = jnp.asarray(previous_packed_state, dtype=jnp.float64) if use_jax else np.asarray(previous_packed_state, dtype=np.float64)
    previous_previous_array = (
        jnp.asarray(previous_previous_packed_state, dtype=jnp.float64)
        if use_jax
        else np.asarray(previous_previous_packed_state, dtype=np.float64)
    )
    packed_array = jnp.asarray(packed_state, dtype=jnp.float64) if use_jax else np.asarray(packed_state, dtype=np.float64)
    rhs_array = jnp.asarray(rhs_fields, dtype=jnp.float64) if use_jax else np.asarray(rhs_fields, dtype=np.float64)
    field_size = previous_array.size - len(feedback_names)
    field_rhs = rhs_array[:field_size]
    field_block = bdf2_residual(
        packed_array[:field_size],
        previous_array[:field_size],
        previous_previous_array[:field_size],
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
    if not feedback_names:
        return field_block
    return jnp.concatenate([field_block, controller_block]) if use_jax else np.concatenate([field_block, controller_block])


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
        diagnostics={
            "residual_evaluation_count": int(getattr(info, "residual_evaluation_count", 0)),
            "residual_evaluation_seconds": float(getattr(info, "residual_evaluation_seconds", 0.0)),
            "jacobian_refresh_count": int(getattr(info, "jacobian_refresh_count", 0)),
            "jacobian_assembly_seconds": float(getattr(info, "jacobian_assembly_seconds", 0.0)),
            "linear_solve_seconds": float(getattr(info, "linear_solve_seconds", 0.0)),
            "line_search_seconds": float(getattr(info, "line_search_seconds", 0.0)),
            "fallback_used": bool(getattr(info, "fallback_used", False)),
            "jacobian_mode": str(getattr(info, "jacobian_mode", "")),
        },
    )
    pressure_sources = explicit_pressure_sources or {}
