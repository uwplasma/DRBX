from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
import re
import time
from typing import Callable

import jax.numpy as jnp
import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver
from ..solver import (
    ImplicitStepInfo,
    SparseJvpWorkspace,
    backward_euler_residual,
    bdf2_residual,
    build_locality_sparsity,
    build_modulo_color_groups,
    build_sparse_difference_quotient_jacobian,
    build_sparse_jvp_jacobian,
    pack_active_fields,
    prepare_sparse_difference_quotient_plan,
    prepare_sparse_jvp_direction_batches,
    prepare_sparse_jvp_workspace,
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
    RecyclingFixedState as _RecyclingFixedState,
    build_fixed_array_rhs as _build_fixed_array_rhs,
    build_fixed_bdf2_residual as _build_fixed_bdf2_residual,
    build_fixed_backward_euler_residual as _build_fixed_backward_euler_residual,
    build_fixed_host_rhs_bridge as _build_fixed_host_rhs_bridge,
    fixed_state_to_feedback_integrals as _fixed_state_to_feedback_integrals,
    fixed_state_to_full_fields as _fixed_state_to_full_fields,
    pack_fixed_state as _pack_fixed_state,
    unpack_fixed_state as _unpack_fixed_state,
)
from .recycling_progress import (
    build_recycling_progress_details as _build_recycling_progress_details,
)
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


@dataclass(frozen=True)
class Recycling1DBackwardEulerResidualContext:
    """Fixed-layout backward-Euler residual and metadata for recycling gates."""

    residual: Callable[[object], object]
    packed_previous_state: np.ndarray
    packed_initial_guess: np.ndarray
    layout: _RecyclingPackedStateLayout
    runtime_model: _RecyclingRuntimeModel
    field_names: tuple[str, ...]
    feedback_names: tuple[str, ...]
    feedback_previous_errors: dict[str, float]


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
    fields = _build_recycling_state_fields(
        runtime_model, field_overrides=field_overrides
    )
    species = _override_species_fields(
        runtime_model.species_templates, fields=fields, mesh=mesh
    )
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
        gamma_i=0.0
        if simple_sheath_settings is None
        else simple_sheath_settings.gamma_i,
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
    electron_fastest_wave = rhs_sqrt(
        rhs_maximum(prepared["e"].temperature, 0.0) / electron.atomic_mass
    )
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
    variables[f"ddt({electron.pressure_name})"] = electron_pressure_rhs_terms.total[
        None, ...
    ]

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
        variables[f"ddt({neutral.density_name})"] = neutral_rhs_terms.density_total[
            None, ...
        ]
        variables[f"ddt({neutral.pressure_name})"] = neutral_rhs_terms.pressure_total[
            None, ...
        ]
        variables[f"ddt({neutral.momentum_name})"] = neutral_rhs_terms.momentum_total[
            None, ...
        ]

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
        _evaluate_option_field(
            config, "sheath_boundary_simple", "wall_potential", mesh=mesh
        )
        / tnorm
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
        secondary_electron_coef=float(
            resolver.resolve("sheath_boundary_simple", "secondary_electron_coef")
        )
        if config.has_option("sheath_boundary_simple", "secondary_electron_coef")
        else 0.0,
        sheath_ion_polytropic=float(
            resolver.resolve("sheath_boundary_simple", "sheath_ion_polytropic")
        )
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
        density_boundary_mode=float(
            resolver.resolve("sheath_boundary_simple", "density_boundary_mode")
        )
        if config.has_option("sheath_boundary_simple", "density_boundary_mode")
        else 1.0,
        pressure_boundary_mode=float(
            resolver.resolve("sheath_boundary_simple", "pressure_boundary_mode")
        )
        if config.has_option("sheath_boundary_simple", "pressure_boundary_mode")
        else 1.0,
        temperature_boundary_mode=float(
            resolver.resolve("sheath_boundary_simple", "temperature_boundary_mode")
        )
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
        secondary_electron_coef=float(
            resolver.resolve(section, "secondary_electron_coef")
        )
        if config.has_section(section)
        and config.has_option(section, "secondary_electron_coef")
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
) -> tuple[
    dict[str, _PreparedSpeciesState], _IonBoundaryResult, _ElectronBoundaryResult
]:
    prepared = {
        name: _prepare_species_state(sp, mesh=mesh) for name, sp in species.items()
    }
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    use_jax_electron_density = use_jax_backend(
        species["e"].density,
        *(prepared[ion.name].density for ion in ions),
    )
    electron_density_array = jnp.asarray if use_jax_electron_density else np.asarray
    electron_density_dtype = jnp.float64 if use_jax_electron_density else np.float64
    electron_density = (
        jnp.zeros_like(
            electron_density_array(species["e"].density, dtype=electron_density_dtype)
        )
        if use_jax_electron_density
        else np.zeros_like(
            electron_density_array(species["e"].density, dtype=electron_density_dtype)
        )
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
                density=_merge_target_guard_cells(
                    prepared["e"].density, electron_boundary.density, mesh=mesh
                ),
                temperature=_merge_target_guard_cells(
                    prepared["e"].temperature, electron_boundary.temperature, mesh=mesh
                ),
                pressure=_merge_target_guard_cells(
                    prepared["e"].pressure, electron_boundary.pressure, mesh=mesh
                ),
                velocity=_merge_target_guard_cells(
                    electron_velocity, electron_boundary.velocity, mesh=mesh
                ),
                momentum=_merge_target_guard_cells(
                    species["e"].atomic_mass
                    * prepared["e"].density
                    * electron_velocity,
                    electron_boundary.momentum,
                    mesh=mesh,
                ),
                energy_source=np.asarray(
                    electron_boundary.energy_source, dtype=np.float64
                ),
            )
    else:
        use_jax_electron_boundary = use_jax_backend(
            prepared["e"].density,
            prepared["e"].temperature,
            prepared["e"].pressure,
            electron_velocity,
        )
        electron_boundary_array = (
            jnp.asarray if use_jax_electron_boundary else np.asarray
        )
        electron_boundary_dtype = (
            jnp.float64 if use_jax_electron_boundary else np.float64
        )
        electron_boundary_density = electron_boundary_array(
            prepared["e"].density, dtype=electron_boundary_dtype
        )
        electron_boundary_temperature = electron_boundary_array(
            prepared["e"].temperature, dtype=electron_boundary_dtype
        )
        electron_boundary_pressure = electron_boundary_array(
            prepared["e"].pressure, dtype=electron_boundary_dtype
        )
        electron_boundary_velocity = electron_boundary_array(
            electron_velocity, dtype=electron_boundary_dtype
        )
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
    electron_boundary_density = electron_prepared_array(
        electron_boundary.density, dtype=electron_prepared_dtype
    )
    electron_boundary_momentum = electron_prepared_array(
        electron_boundary.momentum, dtype=electron_prepared_dtype
    )
    electron_reconstructed_velocity = (
        electron_boundary_momentum
        / electron_prepared_maximum(
            species["e"].atomic_mass
            * _soft_floor(electron_boundary_density, species["e"].density_floor),
            1.0e-8,
        )
    )
    prepared["e"] = _PreparedSpeciesState(
        density=electron_boundary.density,
        pressure=electron_boundary.pressure,
        temperature=electron_boundary.temperature,
        velocity=electron_prepared_array(
            electron_reconstructed_velocity, dtype=electron_prepared_dtype
        ),
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
                        np.zeros_like(
                            ion_boundary_state.energy_source[ion.name], dtype=np.float64
                        )
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
        use_jax_ion_boundary = use_jax_backend(
            *(prepared[ion.name].density for ion in ions)
        )
        ion_boundary_array = jnp.asarray if use_jax_ion_boundary else np.asarray
        ion_boundary_dtype = jnp.float64 if use_jax_ion_boundary else np.float64
        ion_boundary = _IonBoundaryResult(
            density={
                ion.name: ion_boundary_array(
                    prepared[ion.name].density, dtype=ion_boundary_dtype
                )
                for ion in ions
            },
            pressure={
                ion.name: ion_boundary_array(
                    prepared[ion.name].pressure, dtype=ion_boundary_dtype
                )
                for ion in ions
            },
            temperature={
                ion.name: ion_boundary_array(
                    prepared[ion.name].temperature, dtype=ion_boundary_dtype
                )
                for ion in ions
            },
            velocity={
                ion.name: ion_boundary_array(
                    prepared[ion.name].velocity, dtype=ion_boundary_dtype
                )
                for ion in ions
            },
            momentum={
                ion.name: ion_boundary_array(
                    prepared[ion.name].momentum, dtype=ion_boundary_dtype
                )
                for ion in ions
            },
            energy_source={
                ion.name: (
                    jnp.zeros_like(
                        ion_boundary_array(
                            prepared[ion.name].density, dtype=ion_boundary_dtype
                        )
                    )
                    if use_jax_ion_boundary
                    else np.zeros_like(
                        ion_boundary_array(
                            prepared[ion.name].density, dtype=ion_boundary_dtype
                        )
                    )
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
        ion_boundary_density = ion_prepared_array(
            ion_boundary.density[ion.name], dtype=ion_prepared_dtype
        )
        ion_boundary_momentum = ion_prepared_array(
            ion_boundary.momentum[ion.name], dtype=ion_prepared_dtype
        )
        ion_reconstructed_velocity = ion_boundary_momentum / ion_prepared_maximum(
            ion.atomic_mass * _soft_floor(ion_boundary_density, ion.density_floor),
            1.0e-8,
        )
        prepared[ion.name] = _PreparedSpeciesState(
            density=ion_boundary.density[ion.name],
            pressure=ion_boundary.pressure[ion.name],
            temperature=ion_boundary.temperature[ion.name],
            velocity=ion_prepared_array(
                ion_reconstructed_velocity, dtype=ion_prepared_dtype
            ),
            momentum=ion_boundary_momentum,
            momentum_error=ion_prepared_array(
                ion.atomic_mass * ion_boundary_density * ion_reconstructed_velocity
                - ion_boundary_momentum,
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
    density_source = {
        name: zeros_like(sp.density, dtype=dtype) for name, sp in species.items()
    }
    energy_source = {
        name: zeros_like(sp.density, dtype=dtype) for name, sp in species.items()
    }
    diagnostics: dict[str, np.ndarray] = {}
    integral_rhs: dict[str, float] = {}
    integrals = feedback_integrals or {}

    for name, controller in controllers.items():
        upstream_density = array(prepared[name].density, dtype=dtype)[
            mesh.xstart, mesh.ystart, 0
        ]
        error = controller.density_upstream - upstream_density
        stored_integral = array(integrals.get(name, 0.0), dtype=dtype)
        integrated_error = stored_integral
        if feedback_timestep is not None:
            previous_error = (
                error
                if feedback_previous_errors is None
                else array(feedback_previous_errors.get(name, error), dtype=dtype)
            )
            integrated_error = stored_integral + float(feedback_timestep) * 0.5 * (
                error + previous_error
            )
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
        energy_source[name] += (
            0.5 * species[name].atomic_mass * square(velocity) * source
        )
        diagnostics[f"S{name}_feedback"] = array(source, dtype=dtype)
        diagnostics[f"density_feedback_src_mult_{name}"] = array(
            source_multiplier, dtype=dtype
        )
        diagnostics[f"density_feedback_src_p_{name}"] = array(
            proportional_term, dtype=dtype
        )
        diagnostics[f"density_feedback_src_i_{name}"] = array(
            integral_term, dtype=dtype
        )
        diagnostics[f"density_feedback_src_shape_{name}"] = array(
            controller.density_source_shape, dtype=dtype
        )
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
    energy_source: dict[str, np.ndarray] = {
        ion.name: zeros_like(ion.density, dtype=dtype) for ion in ions
    }

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
            density = _copy_field(
                apply_noflow_scalar_guards(
                    density, mesh=mesh, lower_y=True, upper_y=False
                )
            )
            temperature = _copy_field(
                apply_noflow_scalar_guards(
                    temperature, mesh=mesh, lower_y=True, upper_y=False
                )
            )
            pressure = _copy_field(
                apply_noflow_scalar_guards(
                    pressure, mesh=mesh, lower_y=True, upper_y=False
                )
            )
            vel = _copy_field(
                apply_noflow_flow_guards(vel, mesh=mesh, lower_y=True, upper_y=False)
            )

        momentum_field = _copy_field(ion.momentum)
        lower_y_enabled = (
            simple_settings.lower_y
            if simple_settings is not None
            else True
            if full_settings is None
            else full_settings.lower_y
        )
        upper_y_enabled = (
            simple_settings.upper_y
            if simple_settings is not None
            else True
            if full_settings is None
            else full_settings.upper_y
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
                limit_free(
                    ni_m,
                    ni_i,
                    0.0
                    if simple_settings is None
                    else simple_settings.density_boundary_mode,
                ),
            )
            temperature = _set_y(
                temperature,
                jp,
                limit_free(
                    temperature[:, jm, :],
                    temperature[:, j, :],
                    0.0
                    if simple_settings is None
                    else simple_settings.temperature_boundary_mode,
                ),
            )
            pressure = _set_y(
                pressure,
                jp,
                limit_free(
                    pressure[:, jm, :],
                    pressure[:, j, :],
                    0.0
                    if simple_settings is None
                    else simple_settings.pressure_boundary_mode,
                ),
            )

            nisheath = 0.5 * (density[:, jp, :] + density[:, j, :])
            tesheath = maximum(
                0.5 * (te[:, jp, :] + te[:, j, :])
                if jp < te.shape[1]
                else 0.5 * (te[:, jm, :] + te[:, j, :]),
                1.0e-5,
            )
            tisheath = maximum(
                0.5 * (temperature[:, jp, :] + temperature[:, j, :]), 1.0e-5
            )
            if simple_settings is not None:
                da = (
                    (J[:, j, :] + J[:, jp, :])
                    / (sqrt(g22[:, j, :]) + sqrt(g22[:, jp, :]))
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
                energy_source[ion.name] = _add_y(
                    energy_source[ion.name], j, sheath.energy_source_delta
                )
            else:
                nesheath = (
                    0.5 * (electron_density[:, jp, :] + electron_density[:, j, :])
                    if jp < electron_density.shape[1]
                    else 0.5 * (electron_density[:, jm, :] + electron_density[:, j, :])
                )
                grad_ne = electron_density[:, j, :] - nesheath
                grad_ni = density[:, j, :] - nisheath
                source_scale = (
                    (J[:, j, :] + J[:, jp, :])
                    / (sqrt(g22[:, j, :]) + sqrt(g22[:, jp, :]))
                    / (dy[:, j, :] * J[:, j, :])
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
                energy_source[ion.name] = _add_y(
                    energy_source[ion.name], j, sheath.energy_source_delta
                )

        if mesh.has_lower_y_target and lower_y_enabled:
            j = mesh.ystart
            jm = j - 1
            jp = j + 1
            ni_i = density[:, j, :]
            ni_p = density[:, jp, :]
            density = _set_y(
                density,
                jm,
                limit_free(
                    ni_p,
                    ni_i,
                    0.0
                    if simple_settings is None
                    else simple_settings.density_boundary_mode,
                ),
            )
            temperature = _set_y(
                temperature,
                jm,
                limit_free(
                    temperature[:, jp, :],
                    temperature[:, j, :],
                    0.0
                    if simple_settings is None
                    else simple_settings.temperature_boundary_mode,
                ),
            )
            pressure = _set_y(
                pressure,
                jm,
                limit_free(
                    pressure[:, jp, :],
                    pressure[:, j, :],
                    0.0
                    if simple_settings is None
                    else simple_settings.pressure_boundary_mode,
                ),
            )

            nisheath = 0.5 * (density[:, jm, :] + density[:, j, :])
            tesheath = maximum(0.5 * (te[:, jm, :] + te[:, j, :]), 1.0e-5)
            tisheath = maximum(
                0.5 * (temperature[:, jm, :] + temperature[:, j, :]), 1.0e-5
            )
            if simple_settings is not None:
                da = (
                    (J[:, j, :] + J[:, jm, :])
                    / (sqrt(g22[:, j, :]) + sqrt(g22[:, jm, :]))
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
                energy_source[ion.name] = _add_y(
                    energy_source[ion.name], j, sheath.energy_source_delta
                )
            else:
                nesheath = 0.5 * (
                    electron_density[:, jm, :] + electron_density[:, j, :]
                )
                grad_ne = electron_density[:, j, :] - nesheath
                grad_ni = density[:, j, :] - nisheath
                source_scale = (
                    (J[:, j, :] + J[:, jm, :])
                    / (sqrt(g22[:, j, :]) + sqrt(g22[:, jm, :]))
                    / (dy[:, j, :] * J[:, j, :])
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
                energy_source[ion.name] = _add_y(
                    energy_source[ion.name], j, sheath.energy_source_delta
                )

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
                    electron_density_gradient=density[:, neighbor, :]
                    - density[:, j, :],
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
        density = _set_y(
            density, jp, limit_free(density[:, jm, :], density[:, j, :], 0)
        )
        temperature = _set_y(
            temperature, jp, limit_free(temperature[:, jm, :], temperature[:, j, :], 0)
        )
        pressure = _set_y(
            pressure, jp, limit_free(pressure[:, jm, :], pressure[:, j, :], 0)
        )
        phi = _set_zero_current_phi(phi, j=j, neighbor=jm, ghost=jp)
        phi = _set_y(phi, jp, 2.0 * phi[:, j, :] - phi[:, jm, :])
        phi_wall = wall_potential[:, j, :]
        phisheath_raw = 0.5 * (phi[:, jp, :] + phi[:, j, :])
        tesheath = 0.5 * (temperature[:, jp, :] + temperature[:, j, :])
        nesheath = 0.5 * (density[:, jp, :] + density[:, j, :])
        source_scale = (
            (J[:, j, :] + J[:, jp, :])
            / (sqrt(g22[:, j, :]) + sqrt(g22[:, jp, :]))
            / (dy[:, j, :] * J[:, j, :])
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
        density = _set_y(
            density, jm, limit_free(density[:, jp, :], density[:, j, :], 0)
        )
        temperature = _set_y(
            temperature, jm, limit_free(temperature[:, jp, :], temperature[:, j, :], 0)
        )
        pressure = _set_y(
            pressure, jm, limit_free(pressure[:, jp, :], pressure[:, j, :], 0)
        )
        phi = _set_zero_current_phi(phi, j=j, neighbor=jp, ghost=jm)
        phi = _set_y(phi, jm, 2.0 * phi[:, j, :] - phi[:, jp, :])
        phi_wall = wall_potential[:, j, :]
        phisheath_raw = 0.5 * (phi[:, jm, :] + phi[:, j, :])
        tesheath = 0.5 * (temperature[:, jm, :] + temperature[:, j, :])
        nesheath = 0.5 * (density[:, jm, :] + density[:, j, :])
        source_scale = (
            (J[:, j, :] + J[:, jm, :])
            / (sqrt(g22[:, j, :]) + sqrt(g22[:, jm, :]))
            / (dy[:, j, :] * J[:, j, :])
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
                limit_free(
                    ni[:, guard, :], ni[:, j, :], settings.density_boundary_mode
                ),
                dtype=np.float64,
            )
            ti_guard = np.asarray(
                limit_free(
                    ti[:, guard, :], ti[:, j, :], settings.temperature_boundary_mode
                ),
                dtype=np.float64,
            )
            te_guard = np.asarray(
                limit_free(
                    temperature[:, guard, :],
                    temperature[:, j, :],
                    settings.temperature_boundary_mode,
                ),
                dtype=np.float64,
            )
            nisheath = 0.5 * (ni_guard + ni[:, j, :])
            tesheath = np.maximum(0.5 * (te_guard + temperature[:, j, :]), 1.0e-5)
            tisheath = np.maximum(0.5 * (ti_guard + ti[:, j, :]), 1.0e-5)
            c_i_sq = (
                ion_polytropic * tisheath + ion.charge * tesheath
            ) / ion.atomic_mass
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
            limit_free(
                density[:, guard, :], density[:, j, :], settings.density_boundary_mode
            ),
            dtype=np.float64,
        )
        temperature[:, ghost, :] = np.asarray(
            limit_free(
                temperature[:, guard, :],
                temperature[:, j, :],
                settings.temperature_boundary_mode,
            ),
            dtype=np.float64,
        )
        pressure[:, ghost, :] = np.asarray(
            limit_free(
                pressure[:, guard, :],
                pressure[:, j, :],
                settings.pressure_boundary_mode,
            ),
            dtype=np.float64,
        )
        nesheath = 0.5 * (density[:, ghost, :] + density[:, j, :])
        tesheath = np.maximum(
            0.5 * (temperature[:, ghost, :] + temperature[:, j, :]), 1.0e-5
        )
        ion_sum = np.maximum(_ion_sum_at_boundary(j, guard, sign=-1.0), 1.0e-5)
        phi_boundary = tesheath * np.log(
            np.sqrt(tesheath / (electron_mass * (2.0 * math.pi)))
            * (1.0 - secondary_coef)
            * np.maximum(nesheath, 1.0e-5)
            / ion_sum
        )
        phi_boundary = phi_boundary + wall_potential[:, j, :]
        phisheath = np.maximum(phi_boundary, wall_potential[:, j, :])
        vesheath = (
            -np.sqrt(tesheath / (2.0 * math.pi * electron_mass))
            * (1.0 - secondary_coef)
            * np.exp(
                -(phisheath - wall_potential[:, j, :]) / np.maximum(tesheath, 1.0e-5)
            )
        )
        q = gamma_e * tesheath * nesheath * vesheath
        if settings.no_flow:
            vesheath = 0.0
        velocity[:, ghost, :] = 2.0 * vesheath - velocity[:, j, :]
        momentum[:, ghost, :] = (
            2.0 * electron_mass * nesheath * vesheath - momentum[:, j, :]
        )
        q = (
            q
            - (2.5 * tesheath + 0.5 * electron_mass * np.square(vesheath))
            * nesheath
            * vesheath
        )
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
            limit_free(
                density[:, guard, :], density[:, j, :], settings.density_boundary_mode
            ),
            dtype=np.float64,
        )
        temperature[:, ghost, :] = np.asarray(
            limit_free(
                temperature[:, guard, :],
                temperature[:, j, :],
                settings.temperature_boundary_mode,
            ),
            dtype=np.float64,
        )
        pressure[:, ghost, :] = np.asarray(
            limit_free(
                pressure[:, guard, :],
                pressure[:, j, :],
                settings.pressure_boundary_mode,
            ),
            dtype=np.float64,
        )
        nesheath = 0.5 * (density[:, ghost, :] + density[:, j, :])
        tesheath = np.maximum(
            0.5 * (temperature[:, ghost, :] + temperature[:, j, :]), 1.0e-5
        )
        ion_sum = np.maximum(_ion_sum_at_boundary(j, guard, sign=1.0), 1.0e-5)
        phi_boundary = tesheath * np.log(
            np.sqrt(tesheath / (electron_mass * (2.0 * math.pi)))
            * (1.0 - secondary_coef)
            * np.maximum(nesheath, 1.0e-5)
            / ion_sum
        )
        phi_boundary = phi_boundary + wall_potential[:, j, :]
        phisheath = np.maximum(phi_boundary, wall_potential[:, j, :])
        vesheath = (
            np.sqrt(tesheath / (2.0 * math.pi * electron_mass))
            * (1.0 - secondary_coef)
            * np.exp(
                -(phisheath - wall_potential[:, j, :]) / np.maximum(tesheath, 1.0e-5)
            )
        )
        q = gamma_e * tesheath * nesheath * vesheath
        if settings.no_flow:
            vesheath = 0.0
        velocity[:, ghost, :] = 2.0 * vesheath - velocity[:, j, :]
        momentum[:, ghost, :] = (
            2.0 * electron_mass * nesheath * vesheath - momentum[:, j, :]
        )
        q = (
            q
            - (2.5 * tesheath + 0.5 * electron_mass * np.square(vesheath))
            * nesheath
            * vesheath
        )
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
    fields = _build_recycling_state_fields(
        runtime_model, field_overrides=initial_fields
    )
    integrals = {
        name: float((initial_feedback_integrals or {}).get(name, 0.0))
        for name in feedback_names
    }

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
        "adaptive_bdf_active_array_jax_linearized",
        "adaptive_bdf_active_array_jax_linearized_lineax",
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

    if solver_mode in {"bdf", "bdf_fixed_full_field_jvp", "bdf_active_array_jvp"}:
        rhs_backend = {
            "bdf": "host_bridge",
            "bdf_fixed_full_field_jvp": "fixed_full_field_array",
            "bdf_active_array_jvp": "active_array",
        }[solver_mode]
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
            jacobian_mode="jvp" if solver_mode != "bdf" else None,
            rhs_backend=rhs_backend,
            solver_mode_label=solver_mode,
        )

    fixed_bdf2_step_modes = {
        "fixed_bdf2_jax_linearized": "jax_linearized",
        "fixed_bdf2_jax_linearized_lineax": "jax_linearized_lineax",
        "fixed_bdf2_active_array_jax_linearized": "active_array_jax_linearized",
        "fixed_bdf2_active_array_jax_linearized_lineax": (
            "active_array_jax_linearized_lineax"
        ),
    }
    if solver_mode in fixed_bdf2_step_modes:
        return _advance_recycling_1d_fixed_bdf2_history(
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
            step_solver_mode=fixed_bdf2_step_modes[solver_mode],
            solver_mode_label=solver_mode,
        )

    variable_history = {
        name: [np.asarray(fields[name], dtype=np.float64)] for name in field_names
    }
    feedback_history = {
        name: [np.asarray(0.0, dtype=np.float64)] for name in feedback_names
    }
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
        variable_history={
            name: np.stack(history, axis=0)
            for name, history in variable_history.items()
        },
        feedback_integral_history={
            name: np.stack(history, axis=0)
            for name, history in feedback_history.items()
        },
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
    current_fields = {
        name: np.asarray(value, dtype=np.float64, copy=True)
        for name, value in fields.items()
    }
    current_integrals = {
        name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names
    }
    variable_history = {
        name: [np.asarray(current_fields[name], dtype=np.float64)]
        for name in field_names
    }
    feedback_history = {
        name: [np.asarray(current_integrals[name], dtype=np.float64)]
        for name in feedback_names
    }
    suggested_dt = _initial_recycling_continuation_dt(runtime_model, timestep=timestep)
    run_started_at = time.perf_counter()
    interval_started_at = run_started_at

    for interval_index in range(steps):
        current_fields, current_integrals, suggested_dt = (
            _advance_recycling_1d_output_interval(
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
        )
        for name in field_names:
            variable_history[name].append(
                np.asarray(current_fields[name], dtype=np.float64)
            )
        for name in feedback_names:
            feedback_history[name].append(
                np.asarray(current_integrals[name], dtype=np.float64)
            )
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
        variable_history={
            name: np.stack(history, axis=0)
            for name, history in variable_history.items()
        },
        feedback_integral_history={
            name: np.stack(history, axis=0)
            for name, history in feedback_history.items()
        },
    )


def _advance_recycling_1d_fixed_bdf2_history(
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
    step_solver_mode: str = "jax_linearized",
    solver_mode_label: str = "fixed_bdf2_jax_linearized",
) -> Recycling1DHistoryResult:
    """Advance output windows with fixed-layout BE startup and BDF2 steps.

    This is an opt-in promotion lane for the JAX-transformable recycling
    residual. It deliberately avoids the SciPy ``solve_ivp`` full-output seam:
    each output interval is solved by the same fixed-layout implicit stepper
    used by the one-step JAX-linearized gates, with controller integrals packed
    into the residual state.
    """

    current_fields = {
        name: np.asarray(value, dtype=np.float64, copy=True)
        for name, value in fields.items()
    }
    current_integrals = {
        name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names
    }
    previous_fields: dict[str, np.ndarray] | None = None
    previous_integrals: dict[str, float] | None = None
    previous_timestep: float | None = None
    max_internal_timestep = _resolve_recycling_fixed_bdf2_max_internal_timestep(config)
    variable_history = {
        name: [np.asarray(current_fields[name], dtype=np.float64)]
        for name in field_names
    }
    feedback_history = {
        name: [np.asarray(current_integrals[name], dtype=np.float64)]
        for name in feedback_names
    }
    diagnostics: dict[str, float | int | bool | str | None] = {
        "fixed_bdf2_solver_mode": str(solver_mode_label),
        "fixed_bdf2_step_solver_mode": str(step_solver_mode),
        "fixed_bdf2_startup_steps": 0,
        "fixed_bdf2_bdf2_steps": 0,
        "fixed_bdf2_fixed_full_field_rhs_steps": 0,
        "fixed_bdf2_jax_linearized_action_steps": 0,
        "fixed_bdf2_lineax_action_steps": 0,
        "fixed_bdf2_max_residual_inf_norm": 0.0,
        "fixed_bdf2_total_nonlinear_iterations": 0,
        "fixed_bdf2_total_linear_iterations": 0,
        "fixed_bdf2_total_residual_evaluation_count": 0,
        "fixed_bdf2_total_jacobian_refresh_count": 0,
        "fixed_bdf2_total_linear_solve_seconds": 0.0,
        "fixed_bdf2_total_residual_evaluation_seconds": 0.0,
        "fixed_bdf2_active_array_rhs_steps": 0,
        "fixed_bdf2_residual_jitted_steps": 0,
        "fixed_bdf2_unconverged_solver_steps": 0,
        "fixed_bdf2_unknown_convergence_solver_steps": 0,
        "fixed_bdf2_linear_solver_failed_steps": 0,
        "fixed_bdf2_evolve_feedback_integrals": True,
        "fixed_bdf2_internal_substeps": 0,
        "fixed_bdf2_max_output_substeps": 1,
        "fixed_bdf2_max_internal_timestep": None
        if max_internal_timestep is None
        else float(max_internal_timestep),
    }
    run_started_at = time.perf_counter()
    interval_started_at = run_started_at

    def record_step(info: Recycling1DImplicitStepInfo, *, method: str) -> None:
        diagnostics["fixed_bdf2_max_residual_inf_norm"] = max(
            float(diagnostics["fixed_bdf2_max_residual_inf_norm"]),
            float(info.residual_inf_norm),
        )
        diagnostics["fixed_bdf2_total_nonlinear_iterations"] = int(
            diagnostics["fixed_bdf2_total_nonlinear_iterations"]
        ) + int(info.nonlinear_iterations)
        diagnostics["fixed_bdf2_total_linear_iterations"] = int(
            diagnostics["fixed_bdf2_total_linear_iterations"]
        ) + int(info.linear_iterations)
        diagnostics["fixed_bdf2_total_residual_evaluation_count"] = int(
            diagnostics["fixed_bdf2_total_residual_evaluation_count"]
        ) + int(info.diagnostics.get("residual_evaluation_count", 0))
        diagnostics["fixed_bdf2_total_jacobian_refresh_count"] = int(
            diagnostics["fixed_bdf2_total_jacobian_refresh_count"]
        ) + int(info.diagnostics.get("jacobian_refresh_count", 0))
        diagnostics["fixed_bdf2_total_linear_solve_seconds"] = float(
            diagnostics["fixed_bdf2_total_linear_solve_seconds"]
        ) + float(info.diagnostics.get("linear_solve_seconds", 0.0))
        diagnostics["fixed_bdf2_total_residual_evaluation_seconds"] = float(
            diagnostics["fixed_bdf2_total_residual_evaluation_seconds"]
        ) + float(info.diagnostics.get("residual_evaluation_seconds", 0.0))
        if info.diagnostics.get("rhs_backend") == "fixed_full_field_array":
            diagnostics["fixed_bdf2_fixed_full_field_rhs_steps"] = (
                int(diagnostics["fixed_bdf2_fixed_full_field_rhs_steps"]) + 1
            )
        if info.diagnostics.get("rhs_backend") == "active_array":
            diagnostics["fixed_bdf2_active_array_rhs_steps"] = (
                int(diagnostics["fixed_bdf2_active_array_rhs_steps"]) + 1
            )
        step_solver_mode_name = str(info.diagnostics.get("solver_mode", ""))
        if "jax_linearized" in step_solver_mode_name:
            diagnostics["fixed_bdf2_jax_linearized_action_steps"] = (
                int(diagnostics["fixed_bdf2_jax_linearized_action_steps"]) + 1
            )
        if "lineax" in step_solver_mode_name:
            diagnostics["fixed_bdf2_lineax_action_steps"] = (
                int(diagnostics["fixed_bdf2_lineax_action_steps"]) + 1
            )
        if bool(info.diagnostics.get("residual_jitted", False)):
            diagnostics["fixed_bdf2_residual_jitted_steps"] = (
                int(diagnostics["fixed_bdf2_residual_jitted_steps"]) + 1
            )
        converged = info.diagnostics.get("converged")
        if converged is False:
            diagnostics["fixed_bdf2_unconverged_solver_steps"] = (
                int(diagnostics["fixed_bdf2_unconverged_solver_steps"]) + 1
            )
        elif converged is None:
            diagnostics["fixed_bdf2_unknown_convergence_solver_steps"] = (
                int(diagnostics["fixed_bdf2_unknown_convergence_solver_steps"]) + 1
            )
        if info.diagnostics.get("linear_solver_success") is False:
            diagnostics["fixed_bdf2_linear_solver_failed_steps"] = (
                int(diagnostics["fixed_bdf2_linear_solver_failed_steps"]) + 1
            )
        key = (
            "fixed_bdf2_startup_steps"
            if method == "backward_euler"
            else "fixed_bdf2_bdf2_steps"
        )
        diagnostics[key] = int(diagnostics[key]) + 1

    for interval_index in range(steps):
        output_substeps = 1
        if max_internal_timestep is not None:
            output_substeps = max(
                1, int(np.ceil(float(timestep) / float(max_internal_timestep)))
            )
        internal_timestep = float(timestep) / float(output_substeps)
        diagnostics["fixed_bdf2_max_output_substeps"] = max(
            int(diagnostics["fixed_bdf2_max_output_substeps"]), output_substeps
        )

        for _ in range(output_substeps):
            if (
                previous_fields is None
                or previous_integrals is None
                or previous_timestep is None
            ):
                next_fields, next_integrals, info = (
                    advance_recycling_1d_backward_euler_step(
                        config,
                        current_fields,
                        runtime_model=runtime_model,
                        feedback_integrals=current_integrals,
                        mesh=mesh,
                        metrics=metrics,
                        dataset_scalars=dataset_scalars,
                        timestep=internal_timestep,
                        solver_mode=step_solver_mode,
                        residual_tolerance=residual_tolerance,
                        max_nonlinear_iterations=max_nonlinear_iterations,
                        evolve_feedback_integrals=True,
                    )
                )
                record_step(info, method="backward_euler")
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
                    timestep=internal_timestep,
                    previous_timestep=previous_timestep,
                    solver_mode=step_solver_mode,
                    residual_tolerance=residual_tolerance,
                    max_nonlinear_iterations=max_nonlinear_iterations,
                    evolve_feedback_integrals=True,
                )
                record_step(info, method="bdf2")
            diagnostics["fixed_bdf2_internal_substeps"] = (
                int(diagnostics["fixed_bdf2_internal_substeps"]) + 1
            )
            previous_fields = current_fields
            previous_integrals = current_integrals
            previous_timestep = internal_timestep
            current_fields = next_fields
            current_integrals = next_integrals
        for name in field_names:
            variable_history[name].append(
                np.asarray(current_fields[name], dtype=np.float64)
            )
        for name in feedback_names:
            feedback_history[name].append(
                np.asarray(current_integrals[name], dtype=np.float64)
            )
        if progress_callback is not None:
            details, interval_started_at = _build_recycling_progress_details(
                interval_index=interval_index + 1,
                steps=steps,
                solver_mode=solver_mode_label,
                accepted_dt=float(internal_timestep),
                stored_states=len(next(iter(variable_history.values()))),
                output_timestep=timestep,
                run_started_at=run_started_at,
                interval_started_at=interval_started_at,
            )
            progress_callback(details)

    return Recycling1DHistoryResult(
        variable_history={
            name: np.stack(history, axis=0)
            for name, history in variable_history.items()
        },
        feedback_integral_history={
            name: np.stack(history, axis=0)
            for name, history in feedback_history.items()
        },
        diagnostics=diagnostics,
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
    current_fields = {
        name: np.asarray(value, dtype=np.float64, copy=True)
        for name, value in fields.items()
    }
    current_integrals = {
        name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names
    }
    variable_history = {
        name: [np.asarray(current_fields[name], dtype=np.float64)]
        for name in field_names
    }
    feedback_history = {
        name: [np.asarray(current_integrals[name], dtype=np.float64)]
        for name in feedback_names
    }
    suggested_dt = min(float(timestep), 10.0 if len(field_names) > 10 else 5.0)
    run_started_at = time.perf_counter()
    interval_started_at = run_started_at

    for interval_index in range(steps):
        current_fields, current_integrals, suggested_dt = (
            _advance_recycling_1d_adaptive_be_interval(
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
        )
        for name in field_names:
            variable_history[name].append(
                np.asarray(current_fields[name], dtype=np.float64)
            )
        for name in feedback_names:
            feedback_history[name].append(
                np.asarray(current_integrals[name], dtype=np.float64)
            )
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
        variable_history={
            name: np.stack(history, axis=0)
            for name, history in variable_history.items()
        },
        feedback_integral_history={
            name: np.stack(history, axis=0)
            for name, history in feedback_history.items()
        },
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
    current_fields = {
        name: np.asarray(value, dtype=np.float64, copy=True)
        for name, value in fields.items()
    }
    current_integrals = {
        name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names
    }
    previous_fields: dict[str, np.ndarray] | None = None
    previous_integrals: dict[str, float] | None = None
    previous_dt: float | None = None
    variable_history = {
        name: [np.asarray(current_fields[name], dtype=np.float64)]
        for name in field_names
    }
    feedback_history = {
        name: [np.asarray(current_integrals[name], dtype=np.float64)]
        for name in feedback_names
    }
    suggested_dt = _initial_recycling_adaptive_bdf_dt(
        config,
        runtime_model,
        timestep=timestep,
        step_solver_mode=step_solver_mode,
    )
    interval_stats = _new_adaptive_bdf_interval_stats(step_solver_mode)
    interval_stats["adaptive_bdf_interval_count"] = 0
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
            step_stats,
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
        _accumulate_adaptive_bdf_interval_stats(interval_stats, step_stats)
        interval_stats["adaptive_bdf_interval_count"] = (
            int(interval_stats["adaptive_bdf_interval_count"]) + 1
        )
        for name in field_names:
            variable_history[name].append(
                np.asarray(current_fields[name], dtype=np.float64)
            )
        for name in feedback_names:
            feedback_history[name].append(
                np.asarray(current_integrals[name], dtype=np.float64)
            )
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
        variable_history={
            name: np.stack(history, axis=0)
            for name, history in variable_history.items()
        },
        feedback_integral_history={
            name: np.stack(history, axis=0)
            for name, history in feedback_history.items()
        },
        diagnostics=interval_stats,
    )


def _new_adaptive_bdf_interval_stats(
    step_solver_mode: str,
) -> dict[str, float | int | str | None]:
    return {
        "adaptive_bdf_accepted_steps": 0,
        "adaptive_bdf_rejected_steps": 0,
        "adaptive_bdf_reused_history_after_rejection": 0,
        "adaptive_bdf_minimum_dt_fallbacks": 0,
        "adaptive_bdf_startup_trials": 0,
        "adaptive_bdf_bdf2_trials": 0,
        "adaptive_bdf_bdf2_accepted_steps": 0,
        "adaptive_bdf_trial_solver_steps": 0,
        "adaptive_bdf_unconverged_solver_steps": 0,
        "adaptive_bdf_unknown_convergence_solver_steps": 0,
        "adaptive_bdf_fixed_full_field_rhs_solver_steps": 0,
        "adaptive_bdf_active_array_rhs_solver_steps": 0,
        "adaptive_bdf_host_bridge_rhs_solver_steps": 0,
        "adaptive_bdf_sparse_jvp_jacobian_solver_steps": 0,
        "adaptive_bdf_sparse_jvp_workspace_reuses": 0,
        "adaptive_bdf_fd_jacobian_solver_steps": 0,
        "adaptive_bdf_jax_linearized_action_solver_steps": 0,
        "adaptive_bdf_bicgstab_action_solver_steps": 0,
        "adaptive_bdf_lineax_action_solver_steps": 0,
        "adaptive_bdf_residual_evaluation_count": 0,
        "adaptive_bdf_jacobian_refresh_count": 0,
        "adaptive_bdf_linear_iterations": 0,
        "adaptive_bdf_linear_solver_failed_steps": 0,
        "adaptive_bdf_unknown_linear_solver_steps": 0,
        "adaptive_bdf_jvp_jacobian_batch_count": 0,
        "adaptive_bdf_jvp_jacobian_prebuilt_direction_batch_uses": 0,
        "adaptive_bdf_interval_wall_seconds": 0.0,
        "adaptive_bdf_startup_trial_seconds": 0.0,
        "adaptive_bdf_backward_euler_trial_seconds": 0.0,
        "adaptive_bdf_bdf2_trial_seconds": 0.0,
        "adaptive_bdf_error_estimator_seconds": 0.0,
        "adaptive_bdf_residual_evaluation_seconds": 0.0,
        "adaptive_bdf_jacobian_assembly_seconds": 0.0,
        "adaptive_bdf_linear_solve_seconds": 0.0,
        "adaptive_bdf_line_search_seconds": 0.0,
        "adaptive_bdf_jvp_jacobian_total_seconds": 0.0,
        "adaptive_bdf_jvp_jacobian_linearize_seconds": 0.0,
        "adaptive_bdf_jvp_jacobian_tangent_build_seconds": 0.0,
        "adaptive_bdf_jvp_jacobian_push_seconds": 0.0,
        "adaptive_bdf_jvp_jacobian_device_execute_seconds": 0.0,
        "adaptive_bdf_jvp_jacobian_host_transfer_seconds": 0.0,
        "adaptive_bdf_jvp_jacobian_sparse_assembly_seconds": 0.0,
        "adaptive_bdf_min_accepted_dt": None,
        "adaptive_bdf_max_accepted_dt": None,
        "adaptive_bdf_last_error_ratio": None,
        "adaptive_bdf_max_error_ratio": None,
        "adaptive_bdf_last_accepted_error_ratio": None,
        "adaptive_bdf_max_accepted_error_ratio": None,
        "adaptive_bdf_step_solver_mode": str(step_solver_mode),
    }


def _add_adaptive_bdf_elapsed(
    stats: dict[str, float | int | str | None],
    key: str,
    started_at: float,
) -> None:
    elapsed = max(0.0, float(time.perf_counter()) - float(started_at))
    stats[key] = float(stats.get(key, 0.0) or 0.0) + elapsed


def _adaptive_bdf_trace_path() -> str | None:
    value = os.environ.get("JAX_DRB_RECYCLING_ADAPTIVE_BDF_TRACE_JSONL")
    if value is None or not value.strip():
        return None
    return os.path.expanduser(value)


def _json_ready_adaptive_bdf_trace_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_ready_adaptive_bdf_trace_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready_adaptive_bdf_trace_value(item) for item in value]
    return str(value)


def _write_adaptive_bdf_trace_record(
    *,
    event: str,
    trial_kind: str,
    dt: float,
    use_bdf2: bool,
    step_solver_mode: str,
    info: Recycling1DImplicitStepInfo | object | None = None,
    elapsed_seconds: float | None = None,
    error_ratio: float | None = None,
    error_contributors: dict[str, object] | None = None,
) -> None:
    trace_path = _adaptive_bdf_trace_path()
    if trace_path is None:
        return
    record: dict[str, object] = {
        "event": str(event),
        "trial_kind": str(trial_kind),
        "dt": float(dt),
        "use_bdf2": bool(use_bdf2),
        "step_solver_mode": str(step_solver_mode),
        "time": time.time(),
    }
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = float(elapsed_seconds)
    if error_ratio is not None:
        record["error_ratio"] = float(error_ratio) if np.isfinite(error_ratio) else None
    if error_contributors is not None:
        record["error_contributors"] = error_contributors
    if info is not None:
        residual_inf_norm = float(getattr(info, "residual_inf_norm", np.nan))
        record["residual_inf_norm"] = (
            residual_inf_norm if np.isfinite(residual_inf_norm) else None
        )
        record["nonlinear_iterations"] = int(getattr(info, "nonlinear_iterations", 0))
        record["linear_iterations"] = int(getattr(info, "linear_iterations", 0))
        diagnostics = getattr(info, "diagnostics", None)
        if isinstance(diagnostics, dict):
            for key in (
                "rhs_backend",
                "jacobian_mode",
                "converged",
                "residual_evaluation_count",
                "residual_evaluation_seconds",
                "jacobian_refresh_count",
                "jacobian_assembly_seconds",
                "linear_solve_seconds",
                "line_search_seconds",
                "linear_solver_backend",
                "linear_solver_status",
                "linear_solver_success",
                "linear_solver_reported_iterations",
                "jvp_direction_batch_count",
                "jvp_direction_build_seconds",
                "jvp_jacobian_total_seconds",
                "jvp_jacobian_linearize_seconds",
                "jvp_jacobian_tangent_build_seconds",
                "jvp_jacobian_push_seconds",
                "jvp_jacobian_device_execute_seconds",
                "jvp_jacobian_host_transfer_seconds",
                "jvp_jacobian_sparse_assembly_seconds",
                "jvp_jacobian_batch_count",
                "jvp_jacobian_prebuilt_direction_batch_uses",
                "jvp_direction_workspace_reuses",
            ):
                if key in diagnostics:
                    record[key] = diagnostics[key]
    parent = os.path.dirname(trace_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(trace_path, "a", encoding="utf-8") as trace_file:
        trace_file.write(
            json.dumps(_json_ready_adaptive_bdf_trace_value(record), sort_keys=True)
            + "\n"
        )
        trace_file.flush()


def _record_adaptive_bdf_error_ratio(
    stats: dict[str, float | int | str | None],
    error_ratio: float,
) -> None:
    finite_error_ratio = float(error_ratio) if np.isfinite(error_ratio) else None
    stats["adaptive_bdf_last_error_ratio"] = finite_error_ratio
    if finite_error_ratio is None:
        return
    previous_max = stats["adaptive_bdf_max_error_ratio"]
    if previous_max is None:
        stats["adaptive_bdf_max_error_ratio"] = finite_error_ratio
    else:
        stats["adaptive_bdf_max_error_ratio"] = max(
            float(previous_max), finite_error_ratio
        )


def _record_adaptive_bdf_accepted_dt(
    stats: dict[str, float | int | str | None],
    dt: float,
) -> None:
    accepted_dt = float(dt)
    min_dt = stats["adaptive_bdf_min_accepted_dt"]
    max_dt = stats["adaptive_bdf_max_accepted_dt"]
    stats["adaptive_bdf_min_accepted_dt"] = (
        accepted_dt if min_dt is None else min(float(min_dt), accepted_dt)
    )
    stats["adaptive_bdf_max_accepted_dt"] = (
        accepted_dt if max_dt is None else max(float(max_dt), accepted_dt)
    )


def _record_adaptive_bdf_accepted_error_ratio(
    stats: dict[str, float | int | str | None],
    error_ratio: float,
) -> None:
    finite_error_ratio = float(error_ratio) if np.isfinite(error_ratio) else None
    stats["adaptive_bdf_last_accepted_error_ratio"] = finite_error_ratio
    if finite_error_ratio is None:
        return
    previous_max = stats["adaptive_bdf_max_accepted_error_ratio"]
    if previous_max is None:
        stats["adaptive_bdf_max_accepted_error_ratio"] = finite_error_ratio
    else:
        stats["adaptive_bdf_max_accepted_error_ratio"] = max(
            float(previous_max), finite_error_ratio
        )


def _adaptive_bdf_error_contributors_if_tracing(
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
    field_absolute_tolerance_floors: dict[str, float] | None = None,
) -> dict[str, object] | None:
    if _adaptive_bdf_trace_path() is None:
        return None
    return _recycling_state_error_contributors(
        full_fields,
        full_integrals,
        half_fields,
        half_integrals,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        field_absolute_tolerance_floors=field_absolute_tolerance_floors,
    )


def _scale_adaptive_bdf_error_contributors(
    contributors: dict[str, object] | None,
    scale: float,
) -> dict[str, object] | None:
    if contributors is None:
        return None

    def _scaled_entry(entry: dict[str, object]) -> dict[str, object]:
        scaled = dict(entry)
        for key in ("rms_ratio", "max_abs_ratio", "mean_abs_ratio"):
            value = float(scaled[key])
            scaled[key] = value * float(scale) if np.isfinite(value) else value
        for key in ("rms_difference", "max_abs_difference"):
            if key not in scaled:
                continue
            value = float(scaled[key])
            scaled[key] = value * float(scale) if np.isfinite(value) else value
        squared = float(scaled["squared_error_sum"])
        scaled["squared_error_sum"] = (
            squared * float(scale) * float(scale) if np.isfinite(squared) else squared
        )
        return scaled

    fields = [_scaled_entry(dict(item)) for item in contributors.get("fields", [])]  # type: ignore[union-attr]
    feedback = [_scaled_entry(dict(item)) for item in contributors.get("feedback", [])]  # type: ignore[union-attr]
    all_contributors = [*fields, *feedback]
    dominant = max(
        all_contributors, key=lambda item: float(item["rms_ratio"]), default=None
    )
    overall = float(contributors.get("overall_ratio", 0.0))  # type: ignore[union-attr]
    scaled_overall = overall * float(scale) if np.isfinite(overall) else overall
    return {
        "overall_ratio": scaled_overall,
        "component_count": int(contributors.get("component_count", 0)),  # type: ignore[union-attr]
        "dominant": dominant,
        "fields": fields,
        "feedback": feedback,
    }


def _record_adaptive_bdf_step_solver_info(
    stats: dict[str, float | int | str | None],
    info: Recycling1DImplicitStepInfo | object,
) -> None:
    stats["adaptive_bdf_trial_solver_steps"] = (
        int(stats["adaptive_bdf_trial_solver_steps"]) + 1
    )
    diagnostics = getattr(info, "diagnostics", None)
    converged = (
        diagnostics.get("converged")
        if isinstance(diagnostics, dict)
        else getattr(info, "converged", None)
    )
    if converged is False:
        stats["adaptive_bdf_unconverged_solver_steps"] = (
            int(stats["adaptive_bdf_unconverged_solver_steps"]) + 1
        )
    elif converged is None:
        stats["adaptive_bdf_unknown_convergence_solver_steps"] = (
            int(stats["adaptive_bdf_unknown_convergence_solver_steps"]) + 1
        )
    if not isinstance(diagnostics, dict):
        return
    for source_key, destination_key in (
        ("residual_evaluation_count", "adaptive_bdf_residual_evaluation_count"),
        ("jacobian_refresh_count", "adaptive_bdf_jacobian_refresh_count"),
        ("jvp_direction_workspace_reuses", "adaptive_bdf_sparse_jvp_workspace_reuses"),
        ("jvp_jacobian_batch_count", "adaptive_bdf_jvp_jacobian_batch_count"),
        (
            "jvp_jacobian_prebuilt_direction_batch_uses",
            "adaptive_bdf_jvp_jacobian_prebuilt_direction_batch_uses",
        ),
    ):
        stats[destination_key] = int(stats[destination_key]) + int(
            diagnostics.get(source_key, 0) or 0
        )
    stats["adaptive_bdf_linear_iterations"] = int(
        stats["adaptive_bdf_linear_iterations"]
    ) + int(getattr(info, "linear_iterations", 0) or 0)
    if diagnostics.get("linear_solver_success") is False:
        stats["adaptive_bdf_linear_solver_failed_steps"] = (
            int(stats["adaptive_bdf_linear_solver_failed_steps"]) + 1
        )
    elif "linear_solver_success" in diagnostics and diagnostics.get(
        "linear_solver_success"
    ) is None:
        stats["adaptive_bdf_unknown_linear_solver_steps"] = (
            int(stats["adaptive_bdf_unknown_linear_solver_steps"]) + 1
        )
    for source_key, destination_key in (
        ("residual_evaluation_seconds", "adaptive_bdf_residual_evaluation_seconds"),
        ("jacobian_assembly_seconds", "adaptive_bdf_jacobian_assembly_seconds"),
        ("linear_solve_seconds", "adaptive_bdf_linear_solve_seconds"),
        ("line_search_seconds", "adaptive_bdf_line_search_seconds"),
        ("jvp_jacobian_total_seconds", "adaptive_bdf_jvp_jacobian_total_seconds"),
        (
            "jvp_jacobian_linearize_seconds",
            "adaptive_bdf_jvp_jacobian_linearize_seconds",
        ),
        (
            "jvp_jacobian_tangent_build_seconds",
            "adaptive_bdf_jvp_jacobian_tangent_build_seconds",
        ),
        ("jvp_jacobian_push_seconds", "adaptive_bdf_jvp_jacobian_push_seconds"),
        (
            "jvp_jacobian_device_execute_seconds",
            "adaptive_bdf_jvp_jacobian_device_execute_seconds",
        ),
        (
            "jvp_jacobian_host_transfer_seconds",
            "adaptive_bdf_jvp_jacobian_host_transfer_seconds",
        ),
        (
            "jvp_jacobian_sparse_assembly_seconds",
            "adaptive_bdf_jvp_jacobian_sparse_assembly_seconds",
        ),
    ):
        stats[destination_key] = float(stats[destination_key]) + float(
            diagnostics.get(source_key, 0.0) or 0.0
        )
    rhs_backend = str(diagnostics.get("rhs_backend", ""))
    if rhs_backend == "fixed_full_field_array":
        stats["adaptive_bdf_fixed_full_field_rhs_solver_steps"] = (
            int(stats["adaptive_bdf_fixed_full_field_rhs_solver_steps"]) + 1
        )
    elif rhs_backend == "active_array":
        stats["adaptive_bdf_active_array_rhs_solver_steps"] = (
            int(stats["adaptive_bdf_active_array_rhs_solver_steps"]) + 1
        )
    elif rhs_backend == "host_bridge":
        stats["adaptive_bdf_host_bridge_rhs_solver_steps"] = (
            int(stats["adaptive_bdf_host_bridge_rhs_solver_steps"]) + 1
        )
    jacobian_mode = str(diagnostics.get("jacobian_mode", ""))
    if jacobian_mode == "jvp":
        stats["adaptive_bdf_sparse_jvp_jacobian_solver_steps"] = (
            int(stats["adaptive_bdf_sparse_jvp_jacobian_solver_steps"]) + 1
        )
    elif jacobian_mode == "fd":
        stats["adaptive_bdf_fd_jacobian_solver_steps"] = (
            int(stats["adaptive_bdf_fd_jacobian_solver_steps"]) + 1
        )
    elif jacobian_mode.startswith("jax_linearized:"):
        stats["adaptive_bdf_jax_linearized_action_solver_steps"] = (
            int(stats["adaptive_bdf_jax_linearized_action_solver_steps"]) + 1
        )
        if "lineax" in jacobian_mode:
            stats["adaptive_bdf_lineax_action_solver_steps"] = (
                int(stats["adaptive_bdf_lineax_action_solver_steps"]) + 1
            )
        if "bicgstab" in jacobian_mode:
            stats["adaptive_bdf_bicgstab_action_solver_steps"] = (
                int(stats["adaptive_bdf_bicgstab_action_solver_steps"]) + 1
            )


def _accumulate_adaptive_bdf_interval_stats(
    total: dict[str, float | int | str | None],
    step: dict[str, float | int | str | None],
) -> None:
    count_keys = (
        "adaptive_bdf_accepted_steps",
        "adaptive_bdf_rejected_steps",
        "adaptive_bdf_reused_history_after_rejection",
        "adaptive_bdf_minimum_dt_fallbacks",
        "adaptive_bdf_startup_trials",
        "adaptive_bdf_bdf2_trials",
        "adaptive_bdf_bdf2_accepted_steps",
        "adaptive_bdf_trial_solver_steps",
        "adaptive_bdf_unconverged_solver_steps",
        "adaptive_bdf_unknown_convergence_solver_steps",
        "adaptive_bdf_fixed_full_field_rhs_solver_steps",
        "adaptive_bdf_active_array_rhs_solver_steps",
        "adaptive_bdf_host_bridge_rhs_solver_steps",
        "adaptive_bdf_sparse_jvp_jacobian_solver_steps",
        "adaptive_bdf_sparse_jvp_workspace_reuses",
        "adaptive_bdf_fd_jacobian_solver_steps",
        "adaptive_bdf_jax_linearized_action_solver_steps",
        "adaptive_bdf_bicgstab_action_solver_steps",
        "adaptive_bdf_lineax_action_solver_steps",
        "adaptive_bdf_residual_evaluation_count",
        "adaptive_bdf_jacobian_refresh_count",
        "adaptive_bdf_linear_iterations",
        "adaptive_bdf_linear_solver_failed_steps",
        "adaptive_bdf_unknown_linear_solver_steps",
        "adaptive_bdf_jvp_jacobian_batch_count",
        "adaptive_bdf_jvp_jacobian_prebuilt_direction_batch_uses",
    )
    for key in count_keys:
        total[key] = int(total[key]) + int(step.get(key, 0))

    elapsed_keys = (
        "adaptive_bdf_interval_wall_seconds",
        "adaptive_bdf_startup_trial_seconds",
        "adaptive_bdf_backward_euler_trial_seconds",
        "adaptive_bdf_bdf2_trial_seconds",
        "adaptive_bdf_error_estimator_seconds",
        "adaptive_bdf_residual_evaluation_seconds",
        "adaptive_bdf_jacobian_assembly_seconds",
        "adaptive_bdf_linear_solve_seconds",
        "adaptive_bdf_line_search_seconds",
        "adaptive_bdf_jvp_jacobian_total_seconds",
        "adaptive_bdf_jvp_jacobian_linearize_seconds",
        "adaptive_bdf_jvp_jacobian_tangent_build_seconds",
        "adaptive_bdf_jvp_jacobian_push_seconds",
        "adaptive_bdf_jvp_jacobian_device_execute_seconds",
        "adaptive_bdf_jvp_jacobian_host_transfer_seconds",
        "adaptive_bdf_jvp_jacobian_sparse_assembly_seconds",
    )
    for key in elapsed_keys:
        total[key] = float(total.get(key, 0.0) or 0.0) + float(
            step.get(key, 0.0) or 0.0
        )

    for key, reducer in (
        ("adaptive_bdf_min_accepted_dt", min),
        ("adaptive_bdf_max_accepted_dt", max),
        ("adaptive_bdf_max_error_ratio", max),
        ("adaptive_bdf_max_accepted_error_ratio", max),
    ):
        value = step.get(key)
        if value is None:
            continue
        current = total.get(key)
        total[key] = (
            float(value) if current is None else reducer(float(current), float(value))
        )

    if step.get("adaptive_bdf_last_error_ratio") is not None:
        total["adaptive_bdf_last_error_ratio"] = float(
            step["adaptive_bdf_last_error_ratio"]
        )
    if step.get("adaptive_bdf_last_accepted_error_ratio") is not None:
        total["adaptive_bdf_last_accepted_error_ratio"] = float(
            step["adaptive_bdf_last_accepted_error_ratio"]
        )
    total["adaptive_bdf_step_solver_mode"] = str(
        step.get(
            "adaptive_bdf_step_solver_mode", total["adaptive_bdf_step_solver_mode"]
        )
    )


def _adaptive_bdf_minimum_dt(output_timestep: float) -> float:
    output_dt = float(output_timestep)
    if not np.isfinite(output_dt) or output_dt <= 0.0:
        raise ValueError("adaptive BDF output_timestep must be positive and finite.")
    full_window_floor = 0.25
    interval_relative_floor = output_dt / 64.0
    hard_relative_floor = output_dt / 8192.0
    return min(
        output_dt,
        max(hard_relative_floor, min(full_window_floor, interval_relative_floor)),
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
    dict[str, float | int | str | None],
]:
    relative_tolerance = (
        float(config.parsed("solver", "rtol"))
        if config.has_option("solver", "rtol")
        else 1.0e-6
    )
    absolute_tolerance = (
        float(config.parsed("solver", "atol"))
        if config.has_option("solver", "atol")
        else 1.0e-9
    )
    remaining = float(output_timestep)
    minimum_dt = _adaptive_bdf_minimum_dt(output_timestep)
    dt = min(float(suggested_dt), remaining)
    current_fields = {
        name: np.asarray(value, dtype=np.float64, copy=True)
        for name, value in fields.items()
    }
    current_integrals = {
        name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names
    }
    prev_fields = (
        None
        if previous_fields is None
        else {
            name: np.asarray(value, dtype=np.float64, copy=True)
            for name, value in previous_fields.items()
        }
    )
    prev_integrals = (
        None
        if previous_integrals is None
        else {name: float(previous_integrals.get(name, 0.0)) for name in feedback_names}
    )
    prev_dt = previous_dt
    stats = _new_adaptive_bdf_interval_stats(step_solver_mode)
    interval_started_at = time.perf_counter()
    field_absolute_tolerance_floors = _resolve_recycling_adaptive_bdf_field_atol_floors(
        config, field_names
    )
    reuse_rejected_bdf2_history = (
        _resolve_recycling_adaptive_bdf_reuse_rejected_history(
            config, step_solver_mode=step_solver_mode
        )
    )
    sparse_jvp_workspace = (
        _build_recycling_sparse_jvp_workspace(
            field_names=field_names,
            packed_feedback_names=(),
            mesh=mesh,
            jvp_batch_size=_resolve_recycling_jvp_batch_size(),
        )
        if step_solver_mode == "sparse_jvp"
        else None
    )

    while remaining > 1.0e-12:
        dt = min(dt, remaining)
        use_bdf2 = (
            prev_fields is not None
            and prev_integrals is not None
            and prev_dt is not None
            and float(prev_dt) > 0.0
        )
        if not use_bdf2:
            stats["adaptive_bdf_startup_trials"] = (
                int(stats["adaptive_bdf_startup_trials"]) + 1
            )
            startup_started_at = time.perf_counter()
            candidate_fields, candidate_integrals, error_ratio = (
                _advance_recycling_1d_startup_step(
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
                    field_absolute_tolerance_floors=field_absolute_tolerance_floors,
                    step_solver_mode=step_solver_mode,
                    stats=stats,
                    sparse_jvp_workspace=sparse_jvp_workspace,
                )
            )
            _add_adaptive_bdf_elapsed(
                stats, "adaptive_bdf_startup_trial_seconds", startup_started_at
            )
        else:
            stats["adaptive_bdf_bdf2_trials"] = (
                int(stats["adaptive_bdf_bdf2_trials"]) + 1
            )
            be_started_at = time.perf_counter()
            _write_adaptive_bdf_trace_record(
                event="start",
                trial_kind="bdf2_backward_euler_predictor",
                dt=dt,
                use_bdf2=True,
                step_solver_mode=step_solver_mode,
            )
            be_fields, be_integrals, be_info = advance_recycling_1d_backward_euler_step(
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
                sparse_jvp_workspace=sparse_jvp_workspace,
            )
            be_elapsed = max(0.0, float(time.perf_counter()) - float(be_started_at))
            stats["adaptive_bdf_backward_euler_trial_seconds"] = (
                float(stats["adaptive_bdf_backward_euler_trial_seconds"]) + be_elapsed
            )
            _record_adaptive_bdf_step_solver_info(stats, be_info)
            _write_adaptive_bdf_trace_record(
                event="end",
                trial_kind="bdf2_backward_euler_predictor",
                dt=dt,
                use_bdf2=True,
                step_solver_mode=step_solver_mode,
                info=be_info,
                elapsed_seconds=be_elapsed,
            )
            bdf2_started_at = time.perf_counter()
            _write_adaptive_bdf_trace_record(
                event="start",
                trial_kind="bdf2_corrector",
                dt=dt,
                use_bdf2=True,
                step_solver_mode=step_solver_mode,
            )
            bdf_fields, bdf_integrals, bdf_info = advance_recycling_1d_bdf2_step(
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
                previous_timestep=float(prev_dt),
                solver_mode=step_solver_mode,
                residual_tolerance=residual_tolerance,
                max_nonlinear_iterations=max_nonlinear_iterations,
                sparse_jvp_workspace=sparse_jvp_workspace,
            )
            bdf2_elapsed = max(0.0, float(time.perf_counter()) - float(bdf2_started_at))
            stats["adaptive_bdf_bdf2_trial_seconds"] = (
                float(stats["adaptive_bdf_bdf2_trial_seconds"]) + bdf2_elapsed
            )
            _record_adaptive_bdf_step_solver_info(stats, bdf_info)
            _write_adaptive_bdf_trace_record(
                event="end",
                trial_kind="bdf2_corrector",
                dt=dt,
                use_bdf2=True,
                step_solver_mode=step_solver_mode,
                info=bdf_info,
                elapsed_seconds=bdf2_elapsed,
            )
            error_started_at = time.perf_counter()
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
                field_absolute_tolerance_floors=field_absolute_tolerance_floors,
            )
            error_contributors = _adaptive_bdf_error_contributors_if_tracing(
                be_fields,
                be_integrals,
                bdf_fields,
                bdf_integrals,
                field_names=field_names,
                feedback_names=feedback_names,
                mesh=mesh,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                field_absolute_tolerance_floors=field_absolute_tolerance_floors,
            )
            _add_adaptive_bdf_elapsed(
                stats, "adaptive_bdf_error_estimator_seconds", error_started_at
            )
            error_ratio /= 3.0
            error_contributors = _scale_adaptive_bdf_error_contributors(
                error_contributors, 1.0 / 3.0
            )
            _write_adaptive_bdf_trace_record(
                event="error_estimate",
                trial_kind="bdf2_embedded_difference",
                dt=dt,
                use_bdf2=True,
                step_solver_mode=step_solver_mode,
                error_ratio=float(error_ratio),
                error_contributors=error_contributors,
            )
            if (
                np.isfinite(error_ratio)
                and error_ratio <= _ADAPTIVE_BDF_ACCEPTANCE_ERROR_RATIO
            ):
                candidate_fields = bdf_fields
                candidate_integrals = bdf_integrals
            else:
                candidate_fields = be_fields
                candidate_integrals = be_integrals
        order = 2 if use_bdf2 else 1
        _record_adaptive_bdf_error_ratio(stats, float(error_ratio))

        if (
            np.isfinite(error_ratio)
            and error_ratio <= _ADAPTIVE_BDF_ACCEPTANCE_ERROR_RATIO
        ):
            stats["adaptive_bdf_accepted_steps"] = (
                int(stats["adaptive_bdf_accepted_steps"]) + 1
            )
            if use_bdf2:
                stats["adaptive_bdf_bdf2_accepted_steps"] = (
                    int(stats["adaptive_bdf_bdf2_accepted_steps"]) + 1
                )
            _record_adaptive_bdf_accepted_dt(stats, dt)
            _record_adaptive_bdf_accepted_error_ratio(stats, float(error_ratio))
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
            dt = next_dt
            continue

        if dt <= minimum_dt:
            stats["adaptive_bdf_accepted_steps"] = (
                int(stats["adaptive_bdf_accepted_steps"]) + 1
            )
            stats["adaptive_bdf_minimum_dt_fallbacks"] = (
                int(stats["adaptive_bdf_minimum_dt_fallbacks"]) + 1
            )
            _record_adaptive_bdf_accepted_dt(stats, dt)
            _record_adaptive_bdf_accepted_error_ratio(stats, float(error_ratio))
            prev_fields = current_fields
            prev_integrals = current_integrals
            prev_dt = dt
            current_fields = candidate_fields
            current_integrals = candidate_integrals
            remaining -= dt
            continue

        stats["adaptive_bdf_rejected_steps"] = (
            int(stats["adaptive_bdf_rejected_steps"]) + 1
        )
        dt = max(0.5 * dt, minimum_dt)
        if use_bdf2 and reuse_rejected_bdf2_history:
            stats["adaptive_bdf_reused_history_after_rejection"] = (
                int(stats["adaptive_bdf_reused_history_after_rejection"]) + 1
            )
        else:
            prev_fields = None
            prev_integrals = None
            prev_dt = None

    _add_adaptive_bdf_elapsed(
        stats, "adaptive_bdf_interval_wall_seconds", interval_started_at
    )
    return (
        current_fields,
        current_integrals,
        prev_fields,
        prev_integrals,
        prev_dt,
        max(min(dt, float(output_timestep)), minimum_dt),
        stats,
    )


_ADAPTIVE_RECYCLING_TIMESTEP_SAFETY = 0.85
_ADAPTIVE_BDF_ACCEPTANCE_ERROR_RATIO = 0.95


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
        factor = _ADAPTIVE_RECYCLING_TIMESTEP_SAFETY * error_ratio ** (
            -1.0 / float(order + 1)
        )
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
    if history_solver_mode == "adaptive_bdf_active_array_jax_linearized":
        return "active_array_jax_linearized"
    if history_solver_mode == "adaptive_bdf_active_array_jax_linearized_lineax":
        return "active_array_jax_linearized_lineax"
    raise ValueError(f"Unsupported adaptive BDF solver mode {history_solver_mode!r}.")


def _recycling_solver_rhs_backend(solver_mode: str) -> str:
    if solver_mode in {"sparse_jvp", "jax_linearized", "jax_linearized_lineax"}:
        return "fixed_full_field_array"
    if solver_mode in {
        "active_array_jax_linearized",
        "active_array_jax_linearized_lineax",
    }:
        return "active_array"
    return "host_bridge"


def _recycling_solver_uses_fixed_full_field_rhs(solver_mode: str) -> bool:
    return _recycling_solver_rhs_backend(solver_mode) == "fixed_full_field_array"


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
    field_absolute_tolerance_floors: dict[str, float] | None = None,
    step_solver_mode: str = "sparse",
    stats: dict[str, float | int | str | None] | None = None,
    sparse_jvp_workspace: SparseJvpWorkspace | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    full_started_at = time.perf_counter()
    _write_adaptive_bdf_trace_record(
        event="start",
        trial_kind="startup_full_backward_euler",
        dt=timestep,
        use_bdf2=False,
        step_solver_mode=step_solver_mode,
    )
    full_fields, full_integrals, full_info = advance_recycling_1d_backward_euler_step(
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
        sparse_jvp_workspace=sparse_jvp_workspace,
    )
    full_elapsed = max(0.0, float(time.perf_counter()) - float(full_started_at))
    if stats is not None:
        stats["adaptive_bdf_backward_euler_trial_seconds"] = (
            float(stats["adaptive_bdf_backward_euler_trial_seconds"]) + full_elapsed
        )
        _record_adaptive_bdf_step_solver_info(stats, full_info)
    _write_adaptive_bdf_trace_record(
        event="end",
        trial_kind="startup_full_backward_euler",
        dt=timestep,
        use_bdf2=False,
        step_solver_mode=step_solver_mode,
        info=full_info,
        elapsed_seconds=full_elapsed,
    )
    first_half_started_at = time.perf_counter()
    _write_adaptive_bdf_trace_record(
        event="start",
        trial_kind="startup_first_half_backward_euler",
        dt=0.5 * timestep,
        use_bdf2=False,
        step_solver_mode=step_solver_mode,
    )
    half_fields, half_integrals, first_half_info = (
        advance_recycling_1d_backward_euler_step(
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
            sparse_jvp_workspace=sparse_jvp_workspace,
        )
    )
    first_half_elapsed = max(
        0.0, float(time.perf_counter()) - float(first_half_started_at)
    )
    if stats is not None:
        stats["adaptive_bdf_backward_euler_trial_seconds"] = (
            float(stats["adaptive_bdf_backward_euler_trial_seconds"])
            + first_half_elapsed
        )
        _record_adaptive_bdf_step_solver_info(stats, first_half_info)
    _write_adaptive_bdf_trace_record(
        event="end",
        trial_kind="startup_first_half_backward_euler",
        dt=0.5 * timestep,
        use_bdf2=False,
        step_solver_mode=step_solver_mode,
        info=first_half_info,
        elapsed_seconds=first_half_elapsed,
    )
    second_half_started_at = time.perf_counter()
    _write_adaptive_bdf_trace_record(
        event="start",
        trial_kind="startup_second_half_backward_euler",
        dt=0.5 * timestep,
        use_bdf2=False,
        step_solver_mode=step_solver_mode,
    )
    half_fields, half_integrals, second_half_info = (
        advance_recycling_1d_backward_euler_step(
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
            sparse_jvp_workspace=sparse_jvp_workspace,
        )
    )
    second_half_elapsed = max(
        0.0, float(time.perf_counter()) - float(second_half_started_at)
    )
    if stats is not None:
        stats["adaptive_bdf_backward_euler_trial_seconds"] = (
            float(stats["adaptive_bdf_backward_euler_trial_seconds"])
            + second_half_elapsed
        )
        _record_adaptive_bdf_step_solver_info(stats, second_half_info)
    _write_adaptive_bdf_trace_record(
        event="end",
        trial_kind="startup_second_half_backward_euler",
        dt=0.5 * timestep,
        use_bdf2=False,
        step_solver_mode=step_solver_mode,
        info=second_half_info,
        elapsed_seconds=second_half_elapsed,
    )
    error_started_at = time.perf_counter()
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
        field_absolute_tolerance_floors=field_absolute_tolerance_floors,
    )
    error_contributors = _adaptive_bdf_error_contributors_if_tracing(
        full_fields,
        full_integrals,
        half_fields,
        half_integrals,
        field_names=field_names,
        feedback_names=feedback_names,
        mesh=mesh,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        field_absolute_tolerance_floors=field_absolute_tolerance_floors,
    )
    if stats is not None:
        _add_adaptive_bdf_elapsed(
            stats, "adaptive_bdf_error_estimator_seconds", error_started_at
        )
    _write_adaptive_bdf_trace_record(
        event="error_estimate",
        trial_kind="startup_embedded_difference",
        dt=timestep,
        use_bdf2=False,
        step_solver_mode=step_solver_mode,
        error_ratio=float(error_ratio),
        error_contributors=error_contributors,
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
    relative_tolerance = (
        float(config.parsed("solver", "rtol"))
        if config.has_option("solver", "rtol")
        else 1.0e-6
    )
    absolute_tolerance = (
        float(config.parsed("solver", "atol"))
        if config.has_option("solver", "atol")
        else 1.0e-9
    )
    remaining = float(output_timestep)
    minimum_dt = max(float(output_timestep) / 8192.0, 0.25)
    dt = min(float(suggested_dt), remaining)
    current_fields = {
        name: np.asarray(value, dtype=np.float64, copy=True)
        for name, value in fields.items()
    }
    current_integrals = {
        name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names
    }

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

    return (
        current_fields,
        current_integrals,
        max(min(dt, float(output_timestep)), minimum_dt),
    )


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
    field_absolute_tolerance_floors: dict[str, float] | None = None,
) -> float:
    active_slices = _recycling_active_domain_slices(mesh)
    squared_terms: list[np.ndarray] = []
    field_absolute_tolerance_floors = field_absolute_tolerance_floors or {}
    for name in field_names:
        full = np.asarray(full_fields[name], dtype=np.float64)[active_slices]
        half = np.asarray(half_fields[name], dtype=np.float64)[active_slices]
        field_absolute_tolerance = max(
            float(absolute_tolerance),
            float(field_absolute_tolerance_floors.get(name, 0.0)),
        )
        scale = field_absolute_tolerance + float(relative_tolerance) * np.maximum(
            np.abs(full), np.abs(half)
        )
        squared_terms.append(((half - full) / scale).ravel())
    for name in feedback_names:
        full = float(full_integrals.get(name, 0.0))
        half = float(half_integrals.get(name, 0.0))
        scale = float(absolute_tolerance) + float(relative_tolerance) * max(
            abs(full), abs(half), 1.0
        )
        squared_terms.append(np.asarray([(half - full) / scale], dtype=np.float64))
    if not squared_terms:
        return 0.0
    combined = np.concatenate(squared_terms)
    return float(np.sqrt(np.mean(combined * combined)))


def _recycling_state_error_contributors(
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
    field_absolute_tolerance_floors: dict[str, float] | None = None,
) -> dict[str, object]:
    active_slices = _recycling_active_domain_slices(mesh)
    field_absolute_tolerance_floors = field_absolute_tolerance_floors or {}
    field_contributors: list[dict[str, object]] = []
    feedback_contributors: list[dict[str, object]] = []
    squared_sum = 0.0
    component_count = 0

    for name in field_names:
        full = np.asarray(full_fields[name], dtype=np.float64)[active_slices]
        half = np.asarray(half_fields[name], dtype=np.float64)[active_slices]
        field_absolute_tolerance = max(
            float(absolute_tolerance),
            float(field_absolute_tolerance_floors.get(name, 0.0)),
        )
        scale = field_absolute_tolerance + float(relative_tolerance) * np.maximum(
            np.abs(full), np.abs(half)
        )
        normalized = np.asarray((half - full) / scale, dtype=np.float64).ravel()
        difference = np.asarray(half - full, dtype=np.float64).ravel()
        scale_flat = np.asarray(scale, dtype=np.float64).ravel()
        finite_mask = np.isfinite(normalized)
        finite = normalized[finite_mask]
        finite_difference = difference[np.isfinite(difference)]
        finite_scale = scale_flat[np.isfinite(scale_flat)]
        count = int(normalized.size)
        nonfinite_count = int(count - finite.size)
        if finite_difference.size:
            rms_difference = float(
                np.sqrt(np.mean(finite_difference * finite_difference))
            )
            max_abs_difference = float(np.max(np.abs(finite_difference)))
        else:
            rms_difference = 0.0
            max_abs_difference = 0.0
        if finite_scale.size:
            min_scale = float(np.min(finite_scale))
            mean_scale = float(np.mean(finite_scale))
            max_scale = float(np.max(finite_scale))
        else:
            min_scale = math.inf
            mean_scale = math.inf
            max_scale = math.inf
        if count and nonfinite_count:
            rms = math.inf
            max_abs = math.inf
            mean_abs = math.inf
            squared = math.inf
            squared_sum = math.inf
            component_count += count
        elif count:
            abs_values = np.abs(finite)
            rms = float(np.sqrt(np.mean(finite * finite)))
            max_abs = float(np.max(abs_values))
            mean_abs = float(np.mean(abs_values))
            squared = float(np.sum(finite * finite))
            squared_sum += squared
            component_count += count
        else:
            rms = 0.0
            max_abs = 0.0
            mean_abs = 0.0
            squared = 0.0
            nonfinite_count = 0
        field_contributors.append(
            {
                "name": name,
                "component_count": count,
                "nonfinite_count": nonfinite_count,
                "rms_ratio": rms,
                "max_abs_ratio": max_abs,
                "mean_abs_ratio": mean_abs,
                "rms_difference": rms_difference,
                "max_abs_difference": max_abs_difference,
                "min_scale": min_scale,
                "mean_scale": mean_scale,
                "max_scale": max_scale,
                "squared_error_sum": squared,
            }
        )

    for name in feedback_names:
        full = float(full_integrals.get(name, 0.0))
        half = float(half_integrals.get(name, 0.0))
        scale = float(absolute_tolerance) + float(relative_tolerance) * max(
            abs(full), abs(half), 1.0
        )
        difference = half - full
        normalized = float((half - full) / scale)
        if np.isfinite(normalized):
            squared = normalized * normalized
            squared_sum += float(squared)
            component_count += 1
            abs_value = abs(normalized)
        else:
            squared = math.inf
            squared_sum = math.inf
            component_count += 1
            abs_value = math.inf
        feedback_contributors.append(
            {
                "name": name,
                "component_count": 1,
                "nonfinite_count": 0 if np.isfinite(normalized) else 1,
                "rms_ratio": abs_value,
                "max_abs_ratio": abs_value,
                "mean_abs_ratio": abs_value,
                "rms_difference": abs(difference)
                if np.isfinite(difference)
                else math.inf,
                "max_abs_difference": abs(difference)
                if np.isfinite(difference)
                else math.inf,
                "min_scale": scale if np.isfinite(scale) else math.inf,
                "mean_scale": scale if np.isfinite(scale) else math.inf,
                "max_scale": scale if np.isfinite(scale) else math.inf,
                "squared_error_sum": float(squared),
            }
        )

    all_contributors = [*field_contributors, *feedback_contributors]
    dominant = max(
        all_contributors, key=lambda item: float(item["rms_ratio"]), default=None
    )
    overall = float(np.sqrt(squared_sum / component_count)) if component_count else 0.0
    return {
        "overall_ratio": overall,
        "component_count": int(component_count),
        "dominant": dominant,
        "fields": field_contributors,
        "feedback": feedback_contributors,
    }


def _initial_recycling_continuation_dt(
    runtime_model: _RecyclingRuntimeModel,
    *,
    timestep: float,
) -> float:
    base_dt = 25.0 if len(runtime_model.field_names) > 10 else 100.0
    return min(float(timestep), base_dt)


def _initial_recycling_adaptive_bdf_dt(
    config: BoutConfig,
    runtime_model: _RecyclingRuntimeModel,
    *,
    timestep: float,
    step_solver_mode: str = "sparse",
) -> float:
    for section_name in ("runtime", "jax_drb"):
        if not config.has_option(section_name, "recycling_adaptive_bdf_initial_dt"):
            continue
        try:
            configured = float(
                config.parsed(section_name, "recycling_adaptive_bdf_initial_dt")
            )
        except Exception:
            continue
        if np.isfinite(configured) and configured > 0.0:
            return min(float(timestep), configured)
    base_dt = _initial_recycling_continuation_dt(runtime_model, timestep=timestep)
    if "jax_linearized" in str(step_solver_mode):
        return min(base_dt, max(float(timestep) / 16.0, _adaptive_bdf_minimum_dt(timestep)))
    return base_dt


def _build_recycling_sparse_jvp_workspace(
    *,
    field_names: tuple[str, ...],
    packed_feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    jvp_batch_size: int | None,
) -> SparseJvpWorkspace:
    active_shape = _recycling_active_shape(mesh)
    sparsity = _build_recycling_residual_sparsity(
        active_shape=active_shape,
        field_count=len(field_names),
        controller_count=len(packed_feedback_names),
    )
    color_groups = _build_recycling_color_groups(
        active_shape=active_shape,
        field_count=len(field_names),
        controller_count=len(packed_feedback_names),
    )
    return prepare_sparse_jvp_workspace(
        sparsity=sparsity,
        color_groups=color_groups,
        state_shape=(int(sparsity.shape[0]),),
        dtype=np.float64,
        batch_size=jvp_batch_size,
    )


def build_recycling_1d_backward_euler_residual_context(
    config: BoutConfig,
    fields: dict[str, np.ndarray],
    *,
    runtime_model: _RecyclingRuntimeModel | None = None,
    feedback_integrals: dict[str, float],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    timestep: float,
    evolve_feedback_integrals: bool = False,
    rhs_backend: str = "host_bridge",
) -> Recycling1DBackwardEulerResidualContext:
    """Build the fixed-layout BE residual used by implicit recycling solves.

    The returned residual is the promoted differentiability seam for recycling:
    it packs the active domain into a static vector, reconstructs fixed-layout
    fields for the currently validated RHS, and remains compatible with
    ``jax.jit``, ``jax.vmap`` and ``jax.jvp`` when the selected RHS terms are
    backend-preserving.
    """

    runtime_model = runtime_model or _build_recycling_runtime_model(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
    )
    field_names = runtime_model.field_names
    packed_feedback_names = (
        runtime_model.feedback_names if evolve_feedback_integrals else ()
    )
    layout = _build_recycling_packed_state_layout(
        fields=fields,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
    )
    previous_feedback_errors = _current_feedback_errors(
        fields, controllers=runtime_model.controllers, mesh=mesh
    )
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

    feedback_timestep = None if packed_feedback_names else timestep
    if rhs_backend == "active_array":
        fixed_rhs = _build_active_array_recycling_rhs(
            config,
            runtime_model=runtime_model,
            layout=layout,
            base_feedback_integrals=feedback_integrals,
            feedback_previous_errors=previous_feedback_errors,
            feedback_timestep=feedback_timestep,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
        )
    elif rhs_backend == "fixed_full_field_array":
        fixed_rhs = _build_fixed_full_field_recycling_rhs(
            config,
            runtime_model=runtime_model,
            layout=layout,
            base_feedback_integrals=feedback_integrals,
            feedback_previous_errors=previous_feedback_errors,
            feedback_timestep=feedback_timestep,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
        )
    elif rhs_backend == "host_bridge":

        def packed_rhs(
            state_fields: dict[str, object], state_integrals: dict[str, object]
        ) -> object:
            return _compute_recycling_1d_packed_rhs(
                config,
                state_fields,
                sanitize_fields=False,
                feedback_integrals=state_integrals,
                feedback_previous_errors=previous_feedback_errors,
                # When controller integrals are part of the implicit state, the
                # source path should consume that state directly rather than
                # applying a second trapezoid predictor to the same integral.
                feedback_timestep=feedback_timestep,
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
    else:
        raise ValueError(
            f"Unsupported recycling fixed residual rhs_backend={rhs_backend!r}."
        )
    fixed_residual = _build_fixed_backward_euler_residual(
        fixed_rhs,
        layout=layout,
        previous_packed_state=packed_previous,
        timestep=timestep,
    )

    def residual(packed_state: object) -> object:
        return fixed_residual(packed_state)

    return Recycling1DBackwardEulerResidualContext(
        residual=residual,
        packed_previous_state=np.asarray(packed_previous, dtype=np.float64),
        packed_initial_guess=np.asarray(packed_initial_guess, dtype=np.float64),
        layout=layout,
        runtime_model=runtime_model,
        field_names=tuple(field_names),
        feedback_names=tuple(packed_feedback_names),
        feedback_previous_errors=previous_feedback_errors,
    )


def build_recycling_1d_bdf2_residual_context(
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
    previous_timestep: float | None = None,
    evolve_feedback_integrals: bool = False,
    rhs_backend: str = "host_bridge",
) -> Recycling1DBackwardEulerResidualContext:
    """Build the fixed-layout variable-step BDF2 residual for recycling.

    This is the BDF2 counterpart to
    ``build_recycling_1d_backward_euler_residual_context``. Keeping it as a
    first-class context makes the adaptive BDF promotion path testable with
    ``jax.jvp``/``jax.linearize`` before any production default is changed.
    """

    runtime_model = runtime_model or _build_recycling_runtime_model(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
    )
    field_names = runtime_model.field_names
    packed_feedback_names = (
        runtime_model.feedback_names if evolve_feedback_integrals else ()
    )
    layout = _build_recycling_packed_state_layout(
        fields=fields,
        field_names=field_names,
        feedback_names=packed_feedback_names,
        mesh=mesh,
    )
    previous_feedback_errors = _current_feedback_errors(
        fields, controllers=runtime_model.controllers, mesh=mesh
    )
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

    feedback_timestep = None if packed_feedback_names else timestep
    if rhs_backend == "active_array":
        fixed_rhs = _build_active_array_recycling_rhs(
            config,
            runtime_model=runtime_model,
            layout=layout,
            base_feedback_integrals=feedback_integrals,
            feedback_previous_errors=previous_feedback_errors,
            feedback_timestep=feedback_timestep,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
        )
    elif rhs_backend == "fixed_full_field_array":
        fixed_rhs = _build_fixed_full_field_recycling_rhs(
            config,
            runtime_model=runtime_model,
            layout=layout,
            base_feedback_integrals=feedback_integrals,
            feedback_previous_errors=previous_feedback_errors,
            feedback_timestep=feedback_timestep,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
        )
    elif rhs_backend == "host_bridge":

        def packed_rhs(
            state_fields: dict[str, object], state_integrals: dict[str, object]
        ) -> object:
            return _compute_recycling_1d_packed_rhs(
                config,
                state_fields,
                sanitize_fields=False,
                feedback_integrals=state_integrals,
                feedback_previous_errors=previous_feedback_errors,
                # When controller integrals are part of the implicit state, the
                # source path should consume that state directly rather than
                # applying a second trapezoid predictor to the same integral.
                feedback_timestep=feedback_timestep,
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
    else:
        raise ValueError(
            f"Unsupported recycling fixed residual rhs_backend={rhs_backend!r}."
        )

    fixed_residual = _build_fixed_bdf2_residual(
        fixed_rhs,
        layout=layout,
        previous_packed_state=packed_previous,
        previous_previous_packed_state=packed_previous_previous,
        timestep=timestep,
        previous_timestep=previous_timestep,
    )

    def residual(packed_state: object) -> object:
        return fixed_residual(packed_state)

    return Recycling1DBackwardEulerResidualContext(
        residual=residual,
        packed_previous_state=np.asarray(packed_previous, dtype=np.float64),
        packed_initial_guess=np.asarray(packed_initial_guess, dtype=np.float64),
        layout=layout,
        runtime_model=runtime_model,
        field_names=tuple(field_names),
        feedback_names=tuple(packed_feedback_names),
        feedback_previous_errors=previous_feedback_errors,
    )


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
    startup_window = min(25.0, float(output_timestep)) if startup_warmup else 0.0
    startup_dt = min(6.25, startup_window) if startup_window > 0.0 else 0.0
    remaining = float(output_timestep)
    elapsed = 0.0
    current_fields = {
        name: np.asarray(value, dtype=np.float64, copy=True)
        for name, value in fields.items()
    }
    current_integrals = {
        name: float(feedback_integrals.get(name, 0.0))
        for name in runtime_model.feedback_names
    }
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
                next_fields, next_integrals, info = (
                    advance_recycling_1d_backward_euler_step(
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
            if (
                np.isfinite(info.residual_inf_norm)
                and info.residual_inf_norm <= acceptance_residual
            ):
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
    sparse_jvp_workspace: SparseJvpWorkspace | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float], Recycling1DImplicitStepInfo]:
    rhs_backend = _recycling_solver_rhs_backend(solver_mode)
    context = build_recycling_1d_backward_euler_residual_context(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        timestep=timestep,
        evolve_feedback_integrals=evolve_feedback_integrals,
        rhs_backend=rhs_backend,
    )
    runtime_model = context.runtime_model
    field_names = context.field_names
    packed_feedback_names = context.feedback_names
    layout = context.layout
    previous_feedback_errors = context.feedback_previous_errors
    packed_previous = context.packed_previous_state
    packed_initial_guess = context.packed_initial_guess
    residual = context.residual

    if solver_mode in {"sparse", "sparse_jvp"}:
        workspace_compatible = (
            solver_mode == "sparse_jvp"
            and sparse_jvp_workspace is not None
            and tuple(sparse_jvp_workspace.sparsity_shape)
            == (int(packed_previous.size), int(packed_previous.size))
        )
        if workspace_compatible:
            sparsity = sparse_jvp_workspace.sparsity
            color_groups = sparse_jvp_workspace.color_groups
        else:
            sparsity = _build_recycling_residual_sparsity(
                active_shape=layout.active_shape,
                field_count=len(field_names),
                controller_count=len(packed_feedback_names),
            )
            color_groups = _build_recycling_color_groups(
                active_shape=layout.active_shape,
                field_count=len(field_names),
                controller_count=len(packed_feedback_names),
            )
        solved, info = solve_sparse_newton_system(
            residual,
            packed_initial_guess,
            active_shape=(packed_previous.size,),
            sparsity=sparsity,
            color_groups=color_groups,
            residual_tolerance=residual_tolerance,
            step_tolerance=1.0e-11,
            max_nonlinear_iterations=max_nonlinear_iterations,
            linear_restart=20,
            linear_maxiter=300,
            linear_rtol=1.0e-8,
            prefer_direct_linear_solve=True,
            jacobian_refresh_frequency=3 if len(field_names) > 10 else 1,
            jacobian_mode="jvp"
            if solver_mode == "sparse_jvp"
            else _resolve_recycling_sparse_jacobian_mode(),
            jvp_batch_size=_resolve_recycling_jvp_batch_size(),
            sparse_jvp_workspace=sparse_jvp_workspace if workspace_compatible else None,
        )
    elif solver_mode == "matrix_free":
        solved, info = solve_matrix_free_newton_system(
            residual,
            packed_initial_guess,
            active_shape=(packed_previous.size,),
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
    elif solver_mode in {
        "jax_linearized",
        "jax_linearized_lineax",
        "active_array_jax_linearized",
        "active_array_jax_linearized_lineax",
    }:
        linear_restart, linear_maxiter = _resolve_recycling_jax_linear_solver_controls(
            config
        )
        jit_residual = _resolve_recycling_jax_linear_jit_residual(config)
        linear_backend = (
            "lineax_gmres"
            if solver_mode.endswith("_lineax")
            else _resolve_recycling_jax_linear_solver_backend()
        )
        preconditioner_name = (
            _resolve_recycling_jax_linear_preconditioner_name(config)
            if linear_backend == "jax_gmres"
            else None
        )
        solved, info = solve_jax_linearized_newton_system(
            residual,
            packed_initial_guess,
            active_shape=(packed_previous.size,),
            residual_tolerance=residual_tolerance,
            step_tolerance=1.0e-11,
            max_nonlinear_iterations=max_nonlinear_iterations,
            linear_restart=linear_restart,
            linear_maxiter=linear_maxiter,
            linear_solver_backend=linear_backend,
            linear_preconditioner=_build_recycling_jax_linear_preconditioner(
                packed_initial_guess,
                name=preconditioner_name,
                layout=layout,
            ),
            linear_preconditioner_name=preconditioner_name,
            jit_residual=jit_residual,
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
        sanitized_integrals = _sanitize_feedback_integrals(
            next_integrals, controllers=runtime_model.controllers
        )
    else:
        sanitized_integrals = _advance_feedback_integrals(
            sanitized_fields,
            controllers=runtime_model.controllers,
            feedback_integrals=feedback_integrals,
            feedback_previous_errors=previous_feedback_errors,
            mesh=mesh,
            timestep=timestep,
        )
    return (
        sanitized_fields,
        sanitized_integrals,
        _as_recycling_step_info(
            info,
            solver_mode=solver_mode,
            rhs_backend=rhs_backend,
            step_method="backward_euler",
        ),
    )


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
    previous_timestep: float | None = None,
    solver_mode: str = "sparse",
    residual_tolerance: float = 1.0e-8,
    max_nonlinear_iterations: int = 20,
    evolve_feedback_integrals: bool = False,
    sparse_jvp_workspace: SparseJvpWorkspace | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float], Recycling1DImplicitStepInfo]:
    rhs_backend = _recycling_solver_rhs_backend(solver_mode)
    context = build_recycling_1d_bdf2_residual_context(
        config,
        fields,
        previous_fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        previous_feedback_integrals=previous_feedback_integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        timestep=timestep,
        previous_timestep=previous_timestep,
        evolve_feedback_integrals=evolve_feedback_integrals,
        rhs_backend=rhs_backend,
    )
    runtime_model = context.runtime_model
    field_names = context.field_names
    packed_feedback_names = context.feedback_names
    layout = context.layout
    previous_feedback_errors = context.feedback_previous_errors
    packed_previous = context.packed_previous_state
    packed_initial_guess = context.packed_initial_guess
    residual = context.residual

    if solver_mode in {"sparse", "sparse_jvp"}:
        workspace_compatible = (
            solver_mode == "sparse_jvp"
            and sparse_jvp_workspace is not None
            and tuple(sparse_jvp_workspace.sparsity_shape)
            == (int(packed_previous.size), int(packed_previous.size))
        )
        if workspace_compatible:
            sparsity = sparse_jvp_workspace.sparsity
            color_groups = sparse_jvp_workspace.color_groups
        else:
            sparsity = _build_recycling_residual_sparsity(
                active_shape=layout.active_shape,
                field_count=len(field_names),
                controller_count=len(packed_feedback_names),
            )
            color_groups = _build_recycling_color_groups(
                active_shape=layout.active_shape,
                field_count=len(field_names),
                controller_count=len(packed_feedback_names),
            )
        solved, info = solve_sparse_newton_system(
            residual,
            packed_initial_guess,
            active_shape=(packed_previous.size,),
            sparsity=sparsity,
            color_groups=color_groups,
            residual_tolerance=residual_tolerance,
            step_tolerance=1.0e-11,
            max_nonlinear_iterations=max_nonlinear_iterations,
            linear_restart=20,
            linear_maxiter=300,
            linear_rtol=1.0e-8,
            prefer_direct_linear_solve=True,
            jacobian_refresh_frequency=3 if len(field_names) > 10 else 1,
            jacobian_mode="jvp"
            if solver_mode == "sparse_jvp"
            else _resolve_recycling_sparse_jacobian_mode(),
            jvp_batch_size=_resolve_recycling_jvp_batch_size(),
            sparse_jvp_workspace=sparse_jvp_workspace if workspace_compatible else None,
        )
    elif solver_mode == "matrix_free":
        solved, info = solve_matrix_free_newton_system(
            residual,
            packed_initial_guess,
            active_shape=(packed_previous.size,),
            residual_tolerance=residual_tolerance,
            max_nonlinear_iterations=max_nonlinear_iterations,
        )
    elif solver_mode in {
        "jax_linearized",
        "jax_linearized_lineax",
        "active_array_jax_linearized",
        "active_array_jax_linearized_lineax",
    }:
        linear_restart, linear_maxiter = _resolve_recycling_jax_linear_solver_controls(
            config
        )
        jit_residual = _resolve_recycling_jax_linear_jit_residual(config)
        linear_backend = (
            "lineax_gmres"
            if solver_mode.endswith("_lineax")
            else _resolve_recycling_jax_linear_solver_backend()
        )
        preconditioner_name = (
            _resolve_recycling_jax_linear_preconditioner_name(config)
            if linear_backend == "jax_gmres"
            else None
        )
        solved, info = solve_jax_linearized_newton_system(
            residual,
            packed_initial_guess,
            active_shape=(packed_previous.size,),
            residual_tolerance=residual_tolerance,
            step_tolerance=1.0e-11,
            max_nonlinear_iterations=max_nonlinear_iterations,
            linear_restart=linear_restart,
            linear_maxiter=linear_maxiter,
            linear_solver_backend=linear_backend,
            linear_preconditioner=_build_recycling_jax_linear_preconditioner(
                packed_initial_guess,
                name=preconditioner_name,
                layout=layout,
            ),
            linear_preconditioner_name=preconditioner_name,
            jit_residual=jit_residual,
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
        sanitized_integrals = _sanitize_feedback_integrals(
            next_integrals, controllers=runtime_model.controllers
        )
    else:
        sanitized_integrals = _advance_feedback_integrals(
            sanitized_fields,
            controllers=runtime_model.controllers,
            feedback_integrals=feedback_integrals,
            feedback_previous_errors=previous_feedback_errors,
            mesh=mesh,
            timestep=timestep,
        )
    return (
        sanitized_fields,
        sanitized_integrals,
        _as_recycling_step_info(
            info,
            solver_mode=solver_mode,
            rhs_backend=rhs_backend,
            step_method="bdf2",
        ),
    )


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
    jacobian_mode: str | None = None,
    rhs_backend: str = "host_bridge",
    solver_mode_label: str = "bdf",
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
    current_fields = {
        name: np.asarray(value, dtype=np.float64, copy=True)
        for name, value in fields.items()
    }
    current_integrals = {
        name: float(feedback_integrals.get(name, 0.0)) for name in feedback_names
    }
    total_time = float(timestep) * float(steps)
    relative_tolerance = (
        float(config.parsed("solver", "rtol"))
        if config.has_option("solver", "rtol")
        else 1.0e-6
    )
    absolute_tolerance = (
        float(config.parsed("solver", "atol"))
        if config.has_option("solver", "atol")
        else 1.0e-9
    )
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
    jacobian_callback_seconds = 0.0
    jacobian_base_rhs_evaluation_count = 0
    jvp_rhs_evaluation_count = 0
    jvp_jacobian_linearize_seconds = 0.0
    jvp_jacobian_tangent_build_seconds = 0.0
    jvp_jacobian_push_seconds = 0.0
    jvp_jacobian_device_execute_seconds = 0.0
    jvp_jacobian_host_transfer_seconds = 0.0
    jvp_jacobian_sparse_assembly_seconds = 0.0
    jvp_jacobian_total_seconds = 0.0
    jvp_jacobian_batch_count = 0
    jvp_jacobian_prebuilt_direction_batch_uses = 0
    rhs_callback_seconds = 0.0
    rhs_evaluation_seconds = 0.0
    rhs_object_evaluation_seconds = 0.0
    rhs_numpy_conversion_seconds = 0.0
    if rhs_backend not in {"host_bridge", "fixed_full_field_array", "active_array"}:
        raise ValueError(f"Unsupported recycling BDF rhs_backend={rhs_backend!r}.")
    resolved_jacobian_mode = (
        _resolve_recycling_bdf_jacobian_mode()
        if jacobian_mode is None
        else str(jacobian_mode).strip().lower()
    )
    bdf_jacobian_mode = (
        resolved_jacobian_mode if resolved_jacobian_mode in {"fd", "jvp"} else "fd"
    )
    bdf_jvp_batch_size = _resolve_recycling_jvp_batch_size()
    jacobian_parallel_workers = _resolve_recycling_bdf_jacobian_parallel_workers()
    jvp_direction_batches = (
        prepare_sparse_jvp_direction_batches(
            difference_plan=difference_plan,
            state_shape=tuple(y0.shape),
            dtype=np.float64,
            batch_size=bdf_jvp_batch_size,
        )
        if bdf_jacobian_mode == "jvp"
        else None
    )

    def packed_rhs(
        state_fields: dict[str, object], state_integrals: dict[str, object]
    ) -> object:
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

    if rhs_backend == "active_array":
        fixed_rhs = _build_active_array_recycling_rhs(
            config,
            runtime_model=runtime_model,
            layout=layout,
            base_feedback_integrals=current_integrals,
            feedback_previous_errors=None,
            feedback_timestep=None if feedback_names else timestep,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
        )
    elif rhs_backend == "fixed_full_field_array":
        fixed_rhs = _build_fixed_full_field_recycling_rhs(
            config,
            runtime_model=runtime_model,
            layout=layout,
            base_feedback_integrals=current_integrals,
            feedback_previous_errors=None,
            feedback_timestep=None if feedback_names else timestep,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
        )
    else:
        fixed_rhs = _build_fixed_host_rhs_bridge(
            packed_rhs,
            layout=layout,
            base_feedback_integrals=current_integrals,
        )

    def _evaluate_rhs_object(packed_state: object) -> object:
        nonlocal rhs_object_evaluation_seconds
        rhs_object_started_at = time.perf_counter()
        try:
            fixed_state = _unpack_fixed_state(packed_state, layout=layout)
            return _pack_fixed_state(fixed_rhs(fixed_state))
        finally:
            rhs_object_evaluation_seconds += time.perf_counter() - rhs_object_started_at

    def _evaluate_rhs(_time: float, packed_state: np.ndarray) -> np.ndarray:
        nonlocal \
            rhs_evaluation_count, \
            rhs_evaluation_seconds, \
            rhs_numpy_conversion_seconds
        rhs_evaluation_started_at = time.perf_counter()
        rhs_evaluation_count += 1
        try:
            rhs_object = _evaluate_rhs_object(packed_state)
            numpy_conversion_started_at = time.perf_counter()
            try:
                return np.asarray(rhs_object, dtype=np.float64)
            finally:
                rhs_numpy_conversion_seconds += (
                    time.perf_counter() - numpy_conversion_started_at
                )
        finally:
            rhs_evaluation_seconds += time.perf_counter() - rhs_evaluation_started_at

    def rhs(_time: float, packed_state: np.ndarray) -> np.ndarray:
        nonlocal \
            rhs_cache_hit_count, \
            rhs_cache_time, \
            rhs_cache_state, \
            rhs_cache_value, \
            rhs_callback_seconds
        rhs_callback_started_at = time.perf_counter()
        try:
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
            value = _evaluate_rhs(_time, packed_array)
            rhs_cache_time = float(_time)
            rhs_cache_state = np.array(packed_array, dtype=np.float64, copy=True)
            rhs_cache_value = np.array(value, dtype=np.float64, copy=True)
            return value
        finally:
            rhs_callback_seconds += time.perf_counter() - rhs_callback_started_at

    def jacobian(_time: float, packed_state: np.ndarray):
        nonlocal \
            jacobian_base_rhs_evaluation_count, \
            jacobian_callback_count, \
            jacobian_callback_seconds
        nonlocal jvp_jacobian_batch_count, jvp_jacobian_linearize_seconds
        nonlocal \
            jvp_jacobian_device_execute_seconds, \
            jvp_jacobian_host_transfer_seconds, \
            jvp_jacobian_push_seconds
        nonlocal \
            jvp_jacobian_sparse_assembly_seconds, \
            jvp_jacobian_tangent_build_seconds
        nonlocal \
            jvp_jacobian_prebuilt_direction_batch_uses, \
            jvp_jacobian_total_seconds, \
            jvp_rhs_evaluation_count
        jacobian_started_at = time.perf_counter()
        jacobian_callback_count += 1
        try:
            if bdf_jacobian_mode == "jvp":

                def evaluate_jvp_rhs(state: object) -> object:
                    nonlocal jvp_rhs_evaluation_count
                    jvp_rhs_evaluation_count += 1
                    return _evaluate_rhs_object(state)

                def record_jvp_timing(timing: dict[str, float | int]) -> None:
                    nonlocal jvp_jacobian_batch_count, jvp_jacobian_linearize_seconds
                    nonlocal \
                        jvp_jacobian_device_execute_seconds, \
                        jvp_jacobian_host_transfer_seconds
                    nonlocal \
                        jvp_jacobian_push_seconds, \
                        jvp_jacobian_sparse_assembly_seconds
                    nonlocal \
                        jvp_jacobian_tangent_build_seconds, \
                        jvp_jacobian_total_seconds
                    nonlocal jvp_jacobian_prebuilt_direction_batch_uses
                    jvp_jacobian_total_seconds += float(
                        timing.get("total_seconds", 0.0)
                    )
                    jvp_jacobian_linearize_seconds += float(
                        timing.get("linearize_seconds", 0.0)
                    )
                    jvp_jacobian_tangent_build_seconds += float(
                        timing.get("tangent_build_seconds", 0.0)
                    )
                    jvp_jacobian_push_seconds += float(timing.get("push_seconds", 0.0))
                    jvp_jacobian_device_execute_seconds += float(
                        timing.get("device_execute_seconds", 0.0)
                    )
                    jvp_jacobian_host_transfer_seconds += float(
                        timing.get("host_transfer_seconds", 0.0)
                    )
                    jvp_jacobian_sparse_assembly_seconds += float(
                        timing.get("sparse_assembly_seconds", 0.0)
                    )
                    jvp_jacobian_batch_count += int(timing.get("batch_count", 0))
                    jvp_jacobian_prebuilt_direction_batch_uses += int(
                        timing.get("prebuilt_direction_batches", 0)
                    )

                return build_sparse_jvp_jacobian(
                    evaluate_jvp_rhs,
                    packed_state,
                    sparsity=sparsity,
                    color_groups=color_groups,
                    sparsity_csc=sparsity_csc,
                    difference_plan=difference_plan,
                    batch_size=bdf_jvp_batch_size,
                    direction_batches=jvp_direction_batches,
                    timing_callback=record_jvp_timing,
                )
            jacobian_base_rhs_evaluation_count += 1
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
        finally:
            jacobian_callback_seconds += time.perf_counter() - jacobian_started_at

    run_started_at = time.perf_counter()
    max_step = min(float(timestep), 25.0)
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
        max_step=max_step,
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
        sample_integrals = _sanitize_feedback_integrals(
            sample_integrals, controllers=runtime_model.controllers
        )
        for name in field_names:
            variable_history[name].append(
                np.asarray(sample_fields[name], dtype=np.float64)
            )
        for name in feedback_names:
            feedback_history[name].append(
                np.asarray(sample_integrals[name], dtype=np.float64)
            )
        if progress_callback is not None and column > 0:
            details, _ = _build_recycling_progress_details(
                interval_index=column,
                steps=steps,
                solver_mode=solver_mode_label,
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
        variable_history={
            name: np.stack(history, axis=0)
            for name, history in variable_history.items()
        },
        feedback_integral_history={
            name: np.stack(history, axis=0)
            for name, history in feedback_history.items()
        },
        diagnostics={
            "bdf_rhs_evaluation_count": int(rhs_evaluation_count),
            "bdf_rhs_cache_hit_count": int(rhs_cache_hit_count),
            "bdf_rhs_callback_seconds": float(rhs_callback_seconds),
            "bdf_rhs_evaluation_seconds": float(rhs_evaluation_seconds),
            "bdf_rhs_object_evaluation_seconds": float(rhs_object_evaluation_seconds),
            "bdf_rhs_numpy_conversion_seconds": float(rhs_numpy_conversion_seconds),
            "bdf_jacobian_callback_count": int(jacobian_callback_count),
            "bdf_jacobian_callback_seconds": float(jacobian_callback_seconds),
            "bdf_jacobian_base_rhs_evaluation_count": int(
                jacobian_base_rhs_evaluation_count
            ),
            "bdf_jvp_rhs_evaluation_count": int(jvp_rhs_evaluation_count),
            "bdf_jvp_jacobian_batch_count": int(jvp_jacobian_batch_count),
            "bdf_jvp_jacobian_prebuilt_direction_batch_uses": int(
                jvp_jacobian_prebuilt_direction_batch_uses
            ),
            "bdf_jvp_jacobian_linearize_seconds": float(jvp_jacobian_linearize_seconds),
            "bdf_jvp_jacobian_push_seconds": float(jvp_jacobian_push_seconds),
            "bdf_jvp_jacobian_device_execute_seconds": float(
                jvp_jacobian_device_execute_seconds
            ),
            "bdf_jvp_jacobian_host_transfer_seconds": float(
                jvp_jacobian_host_transfer_seconds
            ),
            "bdf_jvp_jacobian_sparse_assembly_seconds": float(
                jvp_jacobian_sparse_assembly_seconds
            ),
            "bdf_jvp_jacobian_tangent_build_seconds": float(
                jvp_jacobian_tangent_build_seconds
            ),
            "bdf_jvp_jacobian_total_seconds": float(jvp_jacobian_total_seconds),
            "bdf_jacobian_mode": bdf_jacobian_mode,
            "bdf_rhs_backend": str(rhs_backend),
            "bdf_jvp_batch_size": None
            if bdf_jvp_batch_size is None
            else int(bdf_jvp_batch_size),
            "bdf_jvp_direction_batch_count": 0
            if jvp_direction_batches is None
            else int(len(jvp_direction_batches)),
            "bdf_jacobian_parallel_workers": int(jacobian_parallel_workers),
            "bdf_solve_seconds": float(max(solve_finished_at - run_started_at, 0.0)),
            "bdf_active_size": int(y0.size),
            "bdf_sparse_nnz": int(sparsity.nnz),
            "bdf_color_group_count": int(len(color_groups)),
            "bdf_max_step": float(max_step),
            "bdf_scipy_status": int(getattr(solution, "status", 0)),
            "bdf_scipy_nfev": int(getattr(solution, "nfev", rhs_evaluation_count)),
            "bdf_scipy_njev": int(getattr(solution, "njev", jacobian_callback_count)),
            "bdf_scipy_nlu": int(getattr(solution, "nlu", 0)),
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


def _resolve_recycling_bdf_jacobian_mode() -> str:
    env_value = (
        os.environ.get(
            "JAX_DRB_RECYCLING_BDF_JACOBIAN_MODE",
            os.environ.get("JAX_DRB_RECYCLING_JACOBIAN_MODE", "fd"),
        )
        .strip()
        .lower()
    )
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


def _resolve_recycling_fixed_bdf2_max_internal_timestep(
    config: BoutConfig | None = None,
) -> float | None:
    option_name = "recycling_fixed_bdf2_max_internal_timestep"
    if config is not None:
        for section_name in ("runtime", "jax_drb"):
            if not config.has_option(section_name, option_name):
                continue
            try:
                configured = float(config.parsed(section_name, option_name))
            except (TypeError, ValueError):
                return None
            return configured if np.isfinite(configured) and configured > 0.0 else None
    env_value = os.environ.get("JAX_DRB_RECYCLING_FIXED_BDF2_MAX_INTERNAL_TIMESTEP")
    if env_value is None or not env_value.strip():
        return None
    try:
        configured = float(env_value)
    except ValueError:
        return None
    return configured if np.isfinite(configured) and configured > 0.0 else None


def _resolve_recycling_adaptive_bdf_component_atol_floor(
    config: BoutConfig | None,
    *,
    option_name: str,
    env_name: str,
) -> float | None:
    if config is not None:
        for section_name in ("runtime", "jax_drb"):
            if not config.has_option(section_name, option_name):
                continue
            try:
                configured = float(config.parsed(section_name, option_name))
            except (TypeError, ValueError):
                return None
            return configured if np.isfinite(configured) and configured > 0.0 else None
    env_value = os.environ.get(env_name)
    if env_value is None or not env_value.strip():
        return None
    try:
        configured = float(env_value)
    except ValueError:
        return None
    return configured if np.isfinite(configured) and configured > 0.0 else None


def _resolve_recycling_adaptive_bdf_momentum_atol_floor(
    config: BoutConfig | None = None,
) -> float | None:
    return _resolve_recycling_adaptive_bdf_component_atol_floor(
        config,
        option_name="recycling_adaptive_bdf_momentum_atol_floor",
        env_name="JAX_DRB_RECYCLING_ADAPTIVE_BDF_MOMENTUM_ATOL_FLOOR",
    )


def _resolve_recycling_adaptive_bdf_density_atol_floor(
    config: BoutConfig | None = None,
) -> float | None:
    return _resolve_recycling_adaptive_bdf_component_atol_floor(
        config,
        option_name="recycling_adaptive_bdf_density_atol_floor",
        env_name="JAX_DRB_RECYCLING_ADAPTIVE_BDF_DENSITY_ATOL_FLOOR",
    )


def _resolve_recycling_adaptive_bdf_pressure_atol_floor(
    config: BoutConfig | None = None,
) -> float | None:
    return _resolve_recycling_adaptive_bdf_component_atol_floor(
        config,
        option_name="recycling_adaptive_bdf_pressure_atol_floor",
        env_name="JAX_DRB_RECYCLING_ADAPTIVE_BDF_PRESSURE_ATOL_FLOOR",
    )


def _resolve_recycling_adaptive_bdf_field_atol_floors(
    config: BoutConfig | None,
    field_names: tuple[str, ...],
) -> dict[str, float]:
    momentum_floor = _resolve_recycling_adaptive_bdf_momentum_atol_floor(config)
    density_floor = _resolve_recycling_adaptive_bdf_density_atol_floor(config)
    pressure_floor = _resolve_recycling_adaptive_bdf_pressure_atol_floor(config)
    floors: dict[str, float] = {}
    for name in field_names:
        if momentum_floor is not None and name.startswith("NV"):
            floors[name] = float(momentum_floor)
        elif pressure_floor is not None and name.startswith("P"):
            floors[name] = float(pressure_floor)
        elif density_floor is not None and name.startswith("N"):
            floors[name] = float(density_floor)
    return floors


def _resolve_recycling_adaptive_bdf_reuse_rejected_history(
    config: BoutConfig | None = None,
    *,
    step_solver_mode: str = "sparse",
) -> bool:
    return _resolve_bool_runtime_option(
        config,
        option_name="recycling_adaptive_bdf_reuse_rejected_history",
        env_name="JAX_DRB_RECYCLING_ADAPTIVE_BDF_REUSE_REJECTED_HISTORY",
        default="jax_linearized" in str(step_solver_mode),
    )


def _resolve_recycling_jax_linear_solver_backend() -> str:
    env_value = (
        os.environ.get("JAX_DRB_RECYCLING_JAX_LINEAR_SOLVER", "jax_gmres")
        .strip()
        .lower()
    )
    aliases = {
        "jax": "jax_gmres",
        "jax_scipy": "jax_gmres",
        "gmres": "jax_gmres",
        "jax_gmres": "jax_gmres",
        "bicgstab": "jax_bicgstab",
        "jax_bicgstab": "jax_bicgstab",
        "lineax": "lineax_gmres",
        "lineax_gmres": "lineax_gmres",
    }
    return aliases.get(env_value, "jax_gmres")


def _resolve_recycling_jax_linear_preconditioner_name(
    config: BoutConfig | None = None,
) -> str | None:
    """Resolve the opt-in JAX GMRES left-preconditioner for recycling solves."""

    option_name = "recycling_jax_linear_preconditioner"
    raw_value: str | None = None
    if config is not None:
        for section_name in ("runtime", "jax_drb"):
            if not config.has_option(section_name, option_name):
                continue
            raw_value = str(config.parsed(section_name, option_name))
            break
    if raw_value is None:
        raw_value = os.environ.get("JAX_DRB_RECYCLING_JAX_LINEAR_PRECONDITIONER")
    if raw_value is None or not raw_value.strip():
        return None
    normalized = raw_value.strip().lower().replace("-", "_")
    aliases = {
        "0": None,
        "false": None,
        "no": None,
        "none": None,
        "off": None,
        "state": "state_scale",
        "state_scale": "state_scale",
        "row_scale": "state_scale",
        "jacobi": "state_scale",
        "field": "field_scale",
        "field_scale": "field_scale",
        "field_rms": "field_scale",
        "block": "field_scale",
        "block_scale": "field_scale",
    }
    return aliases.get(normalized)


def _build_recycling_jax_linear_preconditioner(
    packed_initial_guess: object,
    *,
    name: str | None,
    layout: _RecyclingPackedStateLayout | None = None,
) -> Callable[[object], object] | None:
    if name is None:
        return None
    if name not in {"state_scale", "field_scale"}:
        raise ValueError(f"Unsupported recycling JAX preconditioner {name!r}.")
    packed_guess = jnp.asarray(packed_initial_guess, dtype=jnp.float64)
    if name == "state_scale":
        scale = jnp.maximum(jnp.abs(packed_guess), 1.0)
    else:
        if layout is None:
            raise ValueError(
                "layout is required for recycling field_scale preconditioner."
            )
        scale = _recycling_field_scale_preconditioner_vector(
            packed_guess,
            layout=layout,
        )

    def preconditioner(vector: object) -> object:
        return jnp.asarray(vector, dtype=jnp.float64) / scale

    return preconditioner


def _recycling_field_scale_preconditioner_vector(
    packed_initial_guess: object,
    *,
    layout: _RecyclingPackedStateLayout,
) -> object:
    """Return conservative block row scales for packed fixed-layout residuals."""

    packed_guess = jnp.asarray(packed_initial_guess, dtype=jnp.float64)
    active_cell_count = int(np.prod(tuple(layout.active_shape), dtype=np.int64))
    field_scales: list[object] = []
    offset = 0
    for _field_name in layout.field_names:
        block = packed_guess[offset : offset + active_cell_count]
        if active_cell_count:
            block_scale = jnp.sqrt(jnp.mean(jnp.square(block)))
            block_scale = jnp.maximum(block_scale, 1.0)
            field_scales.append(jnp.full_like(block, block_scale))
        offset += active_cell_count
    feedback_count = len(tuple(layout.feedback_names))
    if feedback_count:
        feedback_block = packed_guess[offset : offset + feedback_count]
        field_scales.append(jnp.maximum(jnp.abs(feedback_block), 1.0))
    if not field_scales:
        return jnp.ones_like(packed_guess)
    return jnp.concatenate(tuple(field_scales))


def _resolve_positive_int_runtime_option(
    config: BoutConfig | None,
    *,
    option_name: str,
    env_name: str,
    default: int,
) -> int:
    if config is not None:
        for section_name in ("runtime", "jax_drb"):
            if not config.has_option(section_name, option_name):
                continue
            try:
                value = int(config.parsed(section_name, option_name))
            except (TypeError, ValueError):
                continue
            return max(1, value)
    env_value = os.environ.get(env_name)
    if env_value is not None and env_value.strip():
        try:
            return max(1, int(env_value))
        except ValueError:
            return int(default)
    return int(default)


def _resolve_bool_runtime_option(
    config: BoutConfig | None,
    *,
    option_name: str,
    env_name: str,
    default: bool = False,
) -> bool:
    truthy = {"1", "true", "yes", "on"}
    falsey = {"0", "false", "no", "off"}
    if config is not None:
        for section_name in ("runtime", "jax_drb"):
            if not config.has_option(section_name, option_name):
                continue
            value = str(config.parsed(section_name, option_name)).strip().lower()
            if value in truthy:
                return True
            if value in falsey:
                return False
    env_value = os.environ.get(env_name)
    if env_value is not None and env_value.strip():
        value = env_value.strip().lower()
        if value in truthy:
            return True
        if value in falsey:
            return False
    return bool(default)


def _resolve_recycling_jax_linear_solver_controls(
    config: BoutConfig | None = None,
) -> tuple[int, int]:
    return (
        _resolve_positive_int_runtime_option(
            config,
            option_name="recycling_jax_linear_restart",
            env_name="JAX_DRB_RECYCLING_JAX_LINEAR_RESTART",
            default=20,
        ),
        _resolve_positive_int_runtime_option(
            config,
            option_name="recycling_jax_linear_maxiter",
            env_name="JAX_DRB_RECYCLING_JAX_LINEAR_MAXITER",
            default=20,
        ),
    )


def _resolve_recycling_jax_linear_jit_residual(
    config: BoutConfig | None = None,
) -> bool:
    return _resolve_bool_runtime_option(
        config,
        option_name="recycling_jax_linear_jit_residual",
        env_name="JAX_DRB_RECYCLING_JAX_LINEAR_JIT_RESIDUAL",
        default=False,
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
        metrics=metrics,
        dataset_scalars=dataset_scalars,
    )
    if sanitize_fields:
        sanitized_fields = _sanitize_recycling_fields(config, fields)
    elif use_jax_backend(*(fields[name] for name in fields)):
        sanitized_fields = {
            name: jnp.asarray(value, dtype=jnp.float64)
            for name, value in fields.items()
        }
    else:
        sanitized_fields = {
            name: np.asarray(value, dtype=np.float64) for name, value in fields.items()
        }
    species = _override_species_fields(
        runtime_model.species_templates, fields=sanitized_fields, mesh=mesh
    )
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
    active_slices = (
        layout.active_slices
        if layout is not None
        else _recycling_active_domain_slices(mesh)
    )
    use_jax_result = use_jax_backend(
        *(result.variables[f"ddt({name})"] for name in field_names)
    )
    rhs_array = jnp.asarray if use_jax_result else np.asarray
    rhs_dtype = jnp.float64 if use_jax_result else np.float64
    concatenate = jnp.concatenate if use_jax_result else np.concatenate
    empty = (
        jnp.array([], dtype=jnp.float64)
        if use_jax_result
        else np.array([], dtype=np.float64)
    )
    pieces = [
        rhs_array(
            result.variables[f"ddt({name})"][0][active_slices], dtype=rhs_dtype
        ).ravel()
        for name in field_names
    ]
    pieces.extend(
        rhs_array(result.feedback_integral_rhs.get(name, 0.0), dtype=rhs_dtype).reshape(
            1
        )
        for name in feedback_names
    )
    return concatenate(pieces) if pieces else empty


def _build_fixed_full_field_recycling_rhs(
    config: BoutConfig,
    *,
    runtime_model: _RecyclingRuntimeModel,
    layout: _RecyclingPackedStateLayout,
    base_feedback_integrals: dict[str, object],
    feedback_previous_errors: dict[str, float] | None,
    feedback_timestep: float | None,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
) -> Callable[[_RecyclingFixedState], _RecyclingFixedState]:
    """Build a fixed-layout RHS that returns active arrays without repacking."""

    def rhs(state: _RecyclingFixedState) -> _RecyclingFixedState:
        full_fields = _fixed_state_to_full_fields(state, layout=layout)
        state_integrals = _fixed_state_to_feedback_integrals(
            state,
            layout=layout,
            base_feedback_integrals=base_feedback_integrals,
        )
        if use_jax_backend(*(full_fields[name] for name in full_fields)):
            state_fields = {
                name: jnp.asarray(value, dtype=jnp.float64)
                for name, value in full_fields.items()
            }
        else:
            state_fields = {
                name: np.asarray(value, dtype=np.float64)
                for name, value in full_fields.items()
            }
        species = _override_species_fields(
            runtime_model.species_templates, fields=state_fields, mesh=mesh
        )
        result = _compute_recycling_1d_rhs_from_species(
            config,
            species=species,
            controllers=runtime_model.controllers,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=dataset_scalars,
            feedback_integrals=state_integrals,
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
        field_rhs = tuple(
            result.variables[f"ddt({name})"][0][layout.active_slices]
            for name in layout.field_names
        )
        feedback_rhs_values = tuple(
            result.feedback_integral_rhs.get(name, 0.0)
            for name in layout.feedback_names
        )
        use_jax_result = use_jax_backend(
            *field_rhs, *feedback_rhs_values, state.feedback_values
        )
        if use_jax_result:
            feedback_rhs = (
                jnp.asarray(feedback_rhs_values, dtype=jnp.float64)
                if feedback_rhs_values
                else jnp.asarray([], dtype=jnp.float64)
            )
            return _RecyclingFixedState(
                field_values=tuple(
                    jnp.asarray(value, dtype=jnp.float64) for value in field_rhs
                ),
                feedback_values=feedback_rhs,
            )
        feedback_rhs_np = (
            np.asarray(feedback_rhs_values, dtype=np.float64)
            if feedback_rhs_values
            else np.asarray([], dtype=np.float64)
        )
        return _RecyclingFixedState(
            field_values=tuple(
                np.asarray(value, dtype=np.float64) for value in field_rhs
            ),
            feedback_values=feedback_rhs_np,
        )

    return rhs


def _build_active_array_recycling_rhs(
    config: BoutConfig,
    *,
    runtime_model: _RecyclingRuntimeModel,
    layout: _RecyclingPackedStateLayout,
    base_feedback_integrals: dict[str, object],
    feedback_previous_errors: dict[str, float] | None,
    feedback_timestep: float | None,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
) -> Callable[[_RecyclingFixedState], _RecyclingFixedState]:
    """Build the opt-in active-array RHS migration seam.

    This backend keeps the validated full-field kernel as the oracle while
    routing through ``build_fixed_array_rhs``. It gives promoted solver tests a
    stable active-array surface before individual sheath, neutral, collision,
    and recycling source terms are moved off guard-cell reconstruction.
    """

    full_field_rhs = _build_fixed_full_field_recycling_rhs(
        config,
        runtime_model=runtime_model,
        layout=layout,
        base_feedback_integrals=base_feedback_integrals,
        feedback_previous_errors=feedback_previous_errors,
        feedback_timestep=feedback_timestep,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
    )

    def field_rhs(
        active_fields: dict[str, object], feedback_values: object
    ) -> dict[str, object]:
        state = _RecyclingFixedState(
            field_values=tuple(active_fields[name] for name in layout.field_names),
            feedback_values=jnp.asarray(feedback_values, dtype=jnp.float64),
        )
        rhs_state = full_field_rhs(state)
        return {
            name: value
            for name, value in zip(
                layout.field_names, rhs_state.field_values, strict=True
            )
        }

    def feedback_rhs(
        active_fields: dict[str, object], feedback_values: object
    ) -> object:
        state = _RecyclingFixedState(
            field_values=tuple(active_fields[name] for name in layout.field_names),
            feedback_values=jnp.asarray(feedback_values, dtype=jnp.float64),
        )
        return full_field_rhs(state).feedback_values

    return _build_fixed_array_rhs(
        field_rhs,
        layout=layout,
        feedback_rhs_function=feedback_rhs if layout.feedback_names else None,
    )


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
    predicted = np.asarray(packed_previous, dtype=np.float64) + float(
        timestep
    ) * np.asarray(rhs, dtype=np.float64)
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
    sanitized_integrals = _sanitize_feedback_integrals(
        predicted_integrals, controllers=runtime_model.controllers
    )
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
        active_slices=(
            layout.active_slices
            if layout is not None
            else _recycling_active_domain_slices(mesh)
        ),
    )
    if not feedback_names:
        return field_block
    if use_jax_backend(
        field_block, *(feedback_integrals.get(name, 0.0) for name in feedback_names)
    ):
        scalar_block = jnp.asarray(
            [feedback_integrals.get(name, 0.0) for name in feedback_names],
            dtype=jnp.float64,
        )
        return jnp.concatenate([field_block, scalar_block])
    scalar_block = np.asarray(
        [feedback_integrals.get(name, 0.0) for name in feedback_names], dtype=np.float64
    )
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
    packed_array = (
        jnp.asarray(packed, dtype=jnp.float64)
        if use_jax
        else np.asarray(packed, dtype=np.float64)
    )
    field_size = (
        layout.field_size
        if layout is not None
        else (_recycling_active_field_size(mesh) * len(field_names))
    )
    field_block = packed_array[:field_size]
    scalar_block = packed_array[field_size:]
    unpacked_fields = unpack_active_fields(
        field_block,
        templates=(
            layout.field_templates
            if layout is not None
            else tuple(
                np.asarray(field_templates[name], dtype=np.float64)
                for name in field_names
            )
        ),
        active_slices=(
            layout.active_slices
            if layout is not None
            else _recycling_active_domain_slices(mesh)
        ),
    )
    restored_fields = {
        name: value for name, value in zip(field_names, unpacked_fields, strict=True)
    }
    restored_integrals = {
        name: value if use_jax_backend(value) else float(value)
        for name, value in feedback_integrals.items()
    }
    for index, name in enumerate(feedback_names):
        restored_integrals[name] = (
            scalar_block[index] if use_jax else float(scalar_block[index])
        )
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
    species = _override_species_fields(
        runtime_model.species_templates, fields=sanitized_fields, mesh=mesh
    )
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
    predicted = {
        name: np.asarray(value, dtype=np.float64, copy=True)
        for name, value in sanitized_fields.items()
    }
    for name in field_names:
        rhs_name = f"ddt({name})"
        if rhs_name not in result.variables:
            continue
        predicted[name] = np.asarray(
            sanitized_fields[name]
            + float(timestep)
            * np.asarray(result.variables[rhs_name][0], dtype=np.float64),
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
    previous_array = (
        jnp.asarray(previous_packed_state, dtype=jnp.float64)
        if use_jax
        else np.asarray(previous_packed_state, dtype=np.float64)
    )
    packed_array = (
        jnp.asarray(packed_state, dtype=jnp.float64)
        if use_jax
        else np.asarray(packed_state, dtype=np.float64)
    )
    rhs_array = (
        jnp.asarray(rhs_fields, dtype=jnp.float64)
        if use_jax
        else np.asarray(rhs_fields, dtype=np.float64)
    )
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
        _feedback_integral_vector(
            previous_feedback_integrals, feedback_names=feedback_names
        ),
        controller_rhs
        if feedback_names
        else _feedback_error_vector(
            current_feedback_errors, feedback_names=feedback_names
        ),
        timestep=timestep,
    )
    if not feedback_names:
        return field_block
    return (
        jnp.concatenate([field_block, controller_block])
        if use_jax
        else np.concatenate([field_block, controller_block])
    )


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
    use_jax = use_jax_backend(
        packed_state, previous_packed_state, previous_previous_packed_state, rhs_fields
    )
    previous_array = (
        jnp.asarray(previous_packed_state, dtype=jnp.float64)
        if use_jax
        else np.asarray(previous_packed_state, dtype=np.float64)
    )
    previous_previous_array = (
        jnp.asarray(previous_previous_packed_state, dtype=jnp.float64)
        if use_jax
        else np.asarray(previous_previous_packed_state, dtype=np.float64)
    )
    packed_array = (
        jnp.asarray(packed_state, dtype=jnp.float64)
        if use_jax
        else np.asarray(packed_state, dtype=np.float64)
    )
    rhs_array = (
        jnp.asarray(rhs_fields, dtype=jnp.float64)
        if use_jax
        else np.asarray(rhs_fields, dtype=np.float64)
    )
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
        _feedback_integral_vector(
            previous_feedback_integrals, feedback_names=feedback_names
        ),
        _feedback_integral_vector(
            previous_previous_feedback_integrals, feedback_names=feedback_names
        ),
        controller_rhs
        if feedback_names
        else _feedback_error_vector(
            current_feedback_errors, feedback_names=feedback_names
        ),
        timestep=timestep,
    )
    if not feedback_names:
        return field_block
    return (
        jnp.concatenate([field_block, controller_block])
        if use_jax
        else np.concatenate([field_block, controller_block])
    )


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

    field_to_controller = csr_matrix(
        np.ones((field_size, controller_count), dtype=bool)
    )
    controller_to_field = csr_matrix(
        np.ones((controller_count, field_size), dtype=bool)
    )
    controller_block = csr_matrix(
        np.ones((controller_count, controller_count), dtype=bool)
    )
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
    controller_groups = tuple(
        (field_size + index,) for index in range(controller_count)
    )
    return field_groups + controller_groups


def _as_recycling_step_info(
    info: ImplicitStepInfo,
    *,
    solver_mode: str | None = None,
    rhs_backend: str | None = None,
    step_method: str | None = None,
) -> Recycling1DImplicitStepInfo:
    diagnostics: dict[str, float | int | bool | str | None] = {
        "residual_evaluation_count": int(getattr(info, "residual_evaluation_count", 0)),
        "residual_evaluation_seconds": float(
            getattr(info, "residual_evaluation_seconds", 0.0)
        ),
        "jacobian_refresh_count": int(getattr(info, "jacobian_refresh_count", 0)),
        "jacobian_assembly_seconds": float(
            getattr(info, "jacobian_assembly_seconds", 0.0)
        ),
        "linear_solve_seconds": float(getattr(info, "linear_solve_seconds", 0.0)),
        "line_search_seconds": float(getattr(info, "line_search_seconds", 0.0)),
        "fallback_used": bool(getattr(info, "fallback_used", False)),
        "jacobian_mode": str(getattr(info, "jacobian_mode", "")),
        "converged": getattr(info, "converged", None),
        "linear_solver_backend": getattr(info, "linear_solver_backend", None),
        "linear_solver_status": getattr(info, "linear_solver_status", None),
        "linear_solver_success": getattr(info, "linear_solver_success", None),
        "linear_solver_reported_iterations": getattr(
            info, "linear_solver_reported_iterations", None
        ),
        "linear_preconditioner": getattr(info, "linear_preconditioner", None),
        "jvp_direction_batch_count": int(getattr(info, "jvp_direction_batch_count", 0)),
        "jvp_direction_build_seconds": float(
            getattr(info, "jvp_direction_build_seconds", 0.0)
        ),
        "jvp_jacobian_total_seconds": float(
            getattr(info, "jvp_jacobian_total_seconds", 0.0)
        ),
        "jvp_jacobian_linearize_seconds": float(
            getattr(info, "jvp_jacobian_linearize_seconds", 0.0)
        ),
        "jvp_jacobian_tangent_build_seconds": float(
            getattr(info, "jvp_jacobian_tangent_build_seconds", 0.0)
        ),
        "jvp_jacobian_push_seconds": float(
            getattr(info, "jvp_jacobian_push_seconds", 0.0)
        ),
        "jvp_jacobian_device_execute_seconds": float(
            getattr(info, "jvp_jacobian_device_execute_seconds", 0.0)
        ),
        "jvp_jacobian_host_transfer_seconds": float(
            getattr(info, "jvp_jacobian_host_transfer_seconds", 0.0)
        ),
        "jvp_jacobian_sparse_assembly_seconds": float(
            getattr(info, "jvp_jacobian_sparse_assembly_seconds", 0.0)
        ),
        "jvp_jacobian_batch_count": int(getattr(info, "jvp_jacobian_batch_count", 0)),
        "jvp_jacobian_prebuilt_direction_batch_uses": int(
            getattr(info, "jvp_jacobian_prebuilt_direction_batch_uses", 0)
        ),
        "jvp_direction_workspace_reuses": int(
            getattr(info, "jvp_direction_workspace_reuses", 0)
        ),
        "residual_jitted": bool(getattr(info, "residual_jitted", False)),
    }
    if solver_mode is not None:
        diagnostics["solver_mode"] = str(solver_mode)
    if rhs_backend is not None:
        diagnostics["rhs_backend"] = str(rhs_backend)
    if step_method is not None:
        diagnostics["step_method"] = str(step_method)
    return Recycling1DImplicitStepInfo(
        residual_inf_norm=info.residual_inf_norm,
        active_size=int(np.prod(info.active_shape)),
        nonlinear_iterations=info.nonlinear_iterations,
        linear_iterations=info.linear_iterations,
        diagnostics=diagnostics,
    )
