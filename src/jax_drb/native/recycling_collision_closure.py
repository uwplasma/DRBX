from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import jax.numpy as jnp
import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver
from .array_backend import use_jax_backend
from .mesh import StructuredMesh
from .metrics import StructuredMetrics
from .neutral_mixed import _div_par_k_grad_par_open, _grad_par_open
from .open_field import apply_noflow_scalar_guards
from .recycling_collisions import (
    compute_collision_frequencies,
    ion_parallel_viscosity_inputs,
)
from .recycling_layout import RecyclingPackedStateLayout
from .recycling_neutral_diffusion import configured_component_names
from .recycling_reactions import charge_exchange_collision_rates
from .recycling_setup import OpenFieldSpecies
from .recycling_state import (
    PreparedSpeciesState,
    safe_temperature as _safe_temperature,
    soft_floor as _soft_floor,
)


@dataclass(frozen=True)
class CollisionClosureTerms:
    energy_source: dict[str, np.ndarray]
    momentum_source: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]


def fixed_layout_collision_friction_heat_exchange_from_active_fields(
    config: BoutConfig,
    *,
    active_fields: Mapping[str, np.ndarray],
    species: Mapping[str, OpenFieldSpecies],
    collision_rates: Mapping[tuple[str, str], np.ndarray],
) -> CollisionClosureTerms:
    """Return pointwise collision friction and heat-exchange active sources.

    This is the fixed-layout active-array counterpart to the local part of
    :func:`apply_collision_closure`. It deliberately excludes gradient terms
    such as thermal forces, ion viscosity, and conduction, which require
    stencil-specific active-grid ports.
    """

    configured_components = set(configured_component_names(config))
    use_jax = use_jax_backend(
        *(active_fields.values()),
        *(collision_rates.values()),
    )
    prepared = _prepared_active_collision_states(active_fields, species, use_jax=use_jax)
    energy_source = {
        name: (
            jnp.zeros_like(jnp.asarray(state.density, dtype=jnp.float64))
            if use_jax
            else np.zeros_like(np.asarray(state.density, dtype=np.float64))
        )
        for name, state in prepared.items()
    }
    momentum_source = {
        name: (
            jnp.zeros_like(jnp.asarray(state.density, dtype=jnp.float64))
            if use_jax
            else np.zeros_like(np.asarray(state.density, dtype=np.float64))
        )
        for name, state in prepared.items()
    }
    diagnostics: dict[str, np.ndarray] = {}

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
            a1 = float(first_species.atomic_mass)
            a2 = float(second_species.atomic_mass)

            if "braginskii_friction" in configured_components:
                coeff = momentum_coefficient(
                    first_name,
                    float(first_species.charge),
                    second_name,
                    float(second_species.charge),
                )
                friction = (
                    coeff
                    * a1
                    * nu_12
                    * first_state.density
                    * (second_state.velocity - first_state.velocity)
                )
                momentum_source[first_name] = momentum_source[first_name] + friction
                momentum_source[second_name] = momentum_source[second_name] - friction
                diagnostics[f"F{first_name}{second_name}_coll"] = friction
                diagnostics[f"F{second_name}{first_name}_coll"] = -friction

                if first_species.has_pressure or second_species.has_pressure:
                    velocity_delta = second_state.velocity - first_state.velocity
                    first_heating = (a2 / (a1 + a2)) * velocity_delta * friction
                    second_heating = (a1 / (a1 + a2)) * velocity_delta * friction
                    energy_source[first_name] = energy_source[first_name] + first_heating
                    energy_source[second_name] = energy_source[second_name] + second_heating
                    diagnostics[f"E{first_name}{second_name}_coll_friction"] = first_heating
                    diagnostics[f"E{second_name}{first_name}_coll_friction"] = second_heating

            if "braginskii_heat_exchange" in configured_components and (
                first_species.has_pressure or second_species.has_pressure
            ):
                heat_exchange = (
                    3.0
                    * (a1 / (a1 + a2))
                    * nu_12
                    * first_state.density
                    * (second_state.temperature - first_state.temperature)
                )
                energy_source[first_name] = energy_source[first_name] + heat_exchange
                energy_source[second_name] = energy_source[second_name] - heat_exchange

    return CollisionClosureTerms(
        energy_source=energy_source,
        momentum_source=momentum_source,
        diagnostics=diagnostics,
    )


def fixed_layout_collision_friction_heat_exchange_field_rhs_from_active_fields(
    config: BoutConfig,
    *,
    active_fields: Mapping[str, np.ndarray],
    species: Mapping[str, OpenFieldSpecies],
    collision_rates: Mapping[tuple[str, str], np.ndarray],
) -> dict[str, np.ndarray]:
    """Map active pointwise collision sources into field-RHS contributions."""

    terms = fixed_layout_collision_friction_heat_exchange_from_active_fields(
        config,
        active_fields=active_fields,
        species=species,
        collision_rates=collision_rates,
    )
    field_rhs: dict[str, np.ndarray] = {}
    for name, sp in species.items():
        if sp.has_pressure:
            field_rhs[sp.pressure_name] = (2.0 / 3.0) * terms.energy_source[name]
        if sp.has_momentum:
            field_rhs[sp.momentum_name] = terms.momentum_source[name]
    return field_rhs


def fixed_layout_collision_transport_field_rhs_from_prepared(
    config: BoutConfig,
    *,
    species: Mapping[str, OpenFieldSpecies],
    prepared: Mapping[str, PreparedSpeciesState],
    layout: RecyclingPackedStateLayout,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    collision_rates: Mapping[tuple[str, str], np.ndarray] | None = None,
    charge_exchange_rates: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Return active field-RHS blocks for gradient collision closures.

    Pointwise friction and heat exchange already have an active-field helper.
    This helper covers the remaining transport-like collision closures from
    :func:`apply_collision_closure`: electron/ion thermal forces, ion-ion
    thermal forces, parallel ion viscosity, and parallel thermal conduction.
    It deliberately consumes sheath/no-flow-prepared full fields while those
    boundary-preparation kernels are still being migrated, but returns only the
    active fixed-layout field RHS needed by the promoted residual seam.
    """

    configured_components = set(configured_component_names(config))
    use_jax = use_jax_backend(
        *(state.density for state in prepared.values()),
        *(state.temperature for state in prepared.values()),
        *(state.velocity for state in prepared.values()),
        *((rate for rate in collision_rates.values()) if collision_rates is not None else ()),
        *((rate for rate in charge_exchange_rates.values()) if charge_exchange_rates is not None else ()),
    )
    zero_like = jnp.zeros_like if use_jax else np.zeros_like
    as_array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    energy_source = {
        name: zero_like(as_array(state.density, dtype=dtype), dtype=dtype)
        for name, state in prepared.items()
    }
    momentum_source = {
        name: zero_like(as_array(state.density, dtype=dtype), dtype=dtype)
        for name, state in prepared.items()
    }
    collision_rate_dict = (
        compute_collision_frequencies(
            config,
            dict(species),
            dict(prepared),
            dataset_scalars=dataset_scalars,
        )
        if collision_rates is None
        else dict(collision_rates)
    )
    cx_rate_dict = (
        charge_exchange_collision_rates(
            config,
            species=dict(species),
            prepared=dict(prepared),
            dataset_scalars=dataset_scalars,
        )
        if charge_exchange_rates is None
        else dict(charge_exchange_rates)
    )

    if (
        "braginskii_thermal_force" in configured_components
        and thermal_force_enabled(config, "electron_ion", True)
    ):
        electron_temperature_gradient = _grad_par_open(
            prepared["e"].temperature,
            mesh=mesh,
            metrics=metrics,
        )
        for name, sp in species.items():
            if name == "e" or sp.charge <= 0.0:
                continue
            ion_force = (
                prepared[name].density
                * (0.71 * (float(sp.charge) ** 2))
                * electron_temperature_gradient
            )
            momentum_source[name] = momentum_source[name] + ion_force
            momentum_source["e"] = momentum_source["e"] - ion_force

    if (
        "braginskii_thermal_force" in configured_components
        and thermal_force_enabled(config, "ion_ion", True)
    ):
        ion_names = tuple(
            name
            for name, sp in species.items()
            if name != "e" and sp.charge != 0.0
        )
        override_mass_restrictions = thermal_force_enabled(
            config,
            "override_ion_mass_restrictions",
            False,
        )
        for index, first_name in enumerate(ion_names):
            for second_name in ion_names[index + 1 :]:
                pair = ion_thermal_force_pair(
                    first_name,
                    second_name,
                    species=dict(species),
                    prepared=dict(prepared),
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
            viscosity_inputs = ion_parallel_viscosity_inputs(
                species_name=name,
                species=dict(species),
                prepared=dict(prepared),
                collision_rates=collision_rate_dict,
                cx_rates=cx_rate_dict,
            )
            viscosity_source = div_par_parallel_ion_viscosity_open(
                viscosity_inputs.eta,
                prepared[name].velocity,
                mesh=mesh,
                metrics=metrics,
            )
            momentum_source[name] = momentum_source[name] + viscosity_source
            energy_source[name] = (
                energy_source[name] - prepared[name].velocity * viscosity_source
            )

    if "braginskii_conduction" in configured_components:
        for name, sp in species.items():
            if not sp.has_pressure:
                continue
            if config.has_option(name, "thermal_conduction") and not bool(
                config.parsed(name, "thermal_conduction")
            ):
                continue
            tau = conduction_collision_time(
                config,
                species=dict(species),
                prepared=dict(prepared),
                collision_rates=collision_rate_dict,
                cx_rates=cx_rate_dict,
                species_name=name,
            )
            kappa_coefficient = conduction_kappa_coefficient(config, sp)
            if use_jax:
                temperature = jnp.asarray(prepared[name].temperature, dtype=jnp.float64)
                pressure = jnp.maximum(
                    jnp.asarray(prepared[name].pressure, dtype=jnp.float64),
                    0.0,
                )
            else:
                temperature = np.asarray(prepared[name].temperature, dtype=np.float64)
                pressure = np.maximum(
                    np.asarray(prepared[name].pressure, dtype=np.float64),
                    0.0,
                )
            kappa_par = kappa_coefficient * pressure * tau / float(sp.atomic_mass)
            if mesh.myg > 0:
                kappa_par = apply_noflow_scalar_guards(
                    kappa_par,
                    mesh=mesh,
                    lower_y=True,
                    upper_y=True,
                )
                if not use_jax:
                    kappa_par = np.asarray(kappa_par, dtype=np.float64)
            energy_source[name] = energy_source[name] + _div_par_k_grad_par_open(
                kappa_par,
                temperature,
                mesh=mesh,
                metrics=metrics,
                boundary_flux=False,
            )

    field_rhs: dict[str, np.ndarray] = {}
    for name, sp in species.items():
        if sp.has_pressure and sp.pressure_name in layout.field_names:
            field_rhs[sp.pressure_name] = (
                (2.0 / 3.0) * energy_source[name]
            )[layout.active_slices]
        if sp.has_momentum and sp.momentum_name in layout.field_names:
            field_rhs[sp.momentum_name] = momentum_source[name][layout.active_slices]
    return field_rhs


def _prepared_active_collision_states(
    active_fields: Mapping[str, np.ndarray],
    species: Mapping[str, OpenFieldSpecies],
    *,
    use_jax: bool,
) -> dict[str, PreparedSpeciesState]:
    prepared: dict[str, PreparedSpeciesState] = {}

    def _array(value):
        return (
            jnp.asarray(value, dtype=jnp.float64)
            if use_jax
            else np.asarray(value, dtype=np.float64)
        )

    ion_names = tuple(name for name, sp in species.items() if sp.charge > 0.0)
    missing = []
    for name, sp in species.items():
        if name != "e" and sp.density_name not in active_fields:
            missing.append(sp.density_name)
        if sp.pressure_name not in active_fields:
            missing.append(sp.pressure_name)
        if sp.has_momentum and name != "e" and sp.momentum_name not in active_fields:
            missing.append(sp.momentum_name)
    if missing:
        missing_text = ", ".join(sorted(set(missing)))
        raise KeyError(f"Missing active collision fields: {missing_text}")

    density_cache: dict[str, np.ndarray] = {}
    for name, sp in species.items():
        if sp.density_name in active_fields:
            density_cache[name] = _array(active_fields[sp.density_name])
        elif name == "e" and ion_names:
            density_cache[name] = _active_electron_density(
                active_fields,
                species=species,
                ion_names=ion_names,
                use_jax=use_jax,
            )

    for name, sp in species.items():
        density = density_cache[name]
        pressure = _array(active_fields[sp.pressure_name])
        temperature = _safe_temperature(pressure, density, float(sp.density_floor))
        if sp.momentum_name in active_fields:
            momentum = _array(active_fields[sp.momentum_name])
        else:
            momentum = (
                jnp.zeros_like(jnp.asarray(density, dtype=jnp.float64))
                if use_jax
                else np.zeros_like(np.asarray(density, dtype=np.float64))
            )
        limited_density = _soft_floor(density, float(sp.density_floor))
        if use_jax:
            velocity = momentum / jnp.maximum(
                float(sp.atomic_mass) * limited_density,
                1.0e-8,
            )
        else:
            velocity = momentum / np.maximum(
                float(sp.atomic_mass) * limited_density,
                1.0e-8,
            )
        prepared[name] = PreparedSpeciesState(
            density=density,
            pressure=pressure,
            temperature=temperature,
            velocity=velocity,
            momentum=momentum,
            momentum_error=(
                jnp.zeros_like(jnp.asarray(momentum, dtype=jnp.float64))
                if use_jax
                else np.zeros_like(np.asarray(momentum, dtype=np.float64))
            ),
        )
    return prepared


def _active_electron_density(
    active_fields: Mapping[str, np.ndarray],
    *,
    species: Mapping[str, OpenFieldSpecies],
    ion_names: tuple[str, ...],
    use_jax: bool,
) -> np.ndarray:
    first = active_fields[species[ion_names[0]].density_name]
    result = (
        jnp.zeros_like(jnp.asarray(first, dtype=jnp.float64))
        if use_jax
        else np.zeros_like(np.asarray(first, dtype=np.float64))
    )
    for name in ion_names:
        density = active_fields[species[name].density_name]
        if use_jax:
            result = result + float(species[name].charge) * jnp.asarray(
                density,
                dtype=jnp.float64,
            )
        else:
            result = result + float(species[name].charge) * np.asarray(
                density,
                dtype=np.float64,
            )
    return result


def momentum_coefficient(name1: str, charge1: float, name2: str, charge2: float) -> float:
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


def thermal_force_enabled(config: BoutConfig, option_name: str, default: bool) -> bool:
    if not config.has_section("braginskii_thermal_force") or not config.has_option("braginskii_thermal_force", option_name):
        return default
    return bool(config.parsed("braginskii_thermal_force", option_name))


def ion_thermal_force_pair(
    species1_name: str,
    species2_name: str,
    *,
    species: dict[str, OpenFieldSpecies],
    prepared: dict[str, PreparedSpeciesState],
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
    return (
        light_name,
        heavy_name,
        heavy_force if use_jax_backend(heavy_force) else np.asarray(heavy_force, dtype=np.float64),
    )


def parallel_ion_viscous_stress_open(
    pressure: np.ndarray,
    tau: np.ndarray,
    velocity: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    bounce_factor: np.ndarray | None = None,
) -> np.ndarray:
    use_jax = use_jax_backend(pressure, tau, velocity, bounce_factor)
    if use_jax:
        bxy = jnp.maximum(jnp.asarray(metrics.Bxy, dtype=jnp.float64), 1.0e-12)
        pressure_array = jnp.asarray(pressure, dtype=jnp.float64)
        tau_array = jnp.asarray(tau, dtype=jnp.float64)
        velocity_array = jnp.asarray(velocity, dtype=jnp.float64)
        grad_par_logb = _grad_par_open(jnp.log(bxy), mesh=mesh, metrics=metrics)
        effective_bounce_factor = (
            jnp.ones_like(pressure_array, dtype=jnp.float64)
            if bounce_factor is None
            else jnp.asarray(bounce_factor, dtype=jnp.float64)
        )
        return -0.96 * pressure_array * tau_array * effective_bounce_factor * (
            2.0 * _grad_par_open(velocity_array, mesh=mesh, metrics=metrics)
            + velocity_array * grad_par_logb
        )

    bxy = np.maximum(np.asarray(metrics.Bxy, dtype=np.float64), 1.0e-12)
    grad_par_logb = _grad_par_open(np.log(bxy), mesh=mesh, metrics=metrics)
    effective_bounce_factor = (
        np.ones_like(np.asarray(pressure, dtype=np.float64), dtype=np.float64)
        if bounce_factor is None
        else np.asarray(bounce_factor, dtype=np.float64)
    )
    return (
        -0.96
        * np.asarray(pressure, dtype=np.float64)
        * np.asarray(tau, dtype=np.float64)
        * effective_bounce_factor
        * (
            2.0 * _grad_par_open(np.asarray(velocity, dtype=np.float64), mesh=mesh, metrics=metrics)
            + np.asarray(velocity, dtype=np.float64) * grad_par_logb
        )
    )


def div_par_parallel_ion_viscosity_open(
    eta: np.ndarray,
    velocity: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    if use_jax_backend(eta, velocity):
        bxy = jnp.maximum(jnp.asarray(metrics.Bxy, dtype=jnp.float64), 1.0e-12)
        sqrt_b = jnp.sqrt(bxy)
        return sqrt_b * _div_par_k_grad_par_open(
            jnp.asarray(eta, dtype=jnp.float64) / bxy,
            sqrt_b * jnp.asarray(velocity, dtype=jnp.float64),
            mesh=mesh,
            metrics=metrics,
            boundary_flux=True,
        )

    bxy = np.maximum(np.asarray(metrics.Bxy, dtype=np.float64), 1.0e-12)
    sqrt_b = np.sqrt(bxy)
    return sqrt_b * _div_par_k_grad_par_open(
        eta / bxy,
        sqrt_b * np.asarray(velocity, dtype=np.float64),
        mesh=mesh,
        metrics=metrics,
        boundary_flux=True,
    )


def conduction_kappa_coefficient(config: BoutConfig, species: OpenFieldSpecies) -> float:
    if config.has_option(species.name, "kappa_coefficient"):
        return float(NumericResolver(config).resolve(species.name, "kappa_coefficient"))
    if species.charge < 0.0:
        return 3.16 / math.sqrt(2.0)
    if species.charge == 0.0:
        return 2.5
    return 3.9


def conduction_collision_time(
    config: BoutConfig,
    *,
    species: dict[str, OpenFieldSpecies],
    prepared: dict[str, PreparedSpeciesState],
    collision_rates: dict[tuple[str, str], np.ndarray],
    cx_rates: dict[str, np.ndarray],
    species_name: str,
) -> np.ndarray:
    species_type = "electron" if species_name == "e" else ("neutral" if species[species_name].charge == 0.0 else "ion")
    mode = str(config.parsed(species_name, "conduction_collisions_mode")).strip().lower() if config.has_option(species_name, "conduction_collisions_mode") else "multispecies"
    use_jax = use_jax_backend(
        prepared[species_name].density,
        *(rate for rate in collision_rates.values()),
        *(rate for rate in cx_rates.values()),
    )
    total = (
        jnp.zeros_like(jnp.asarray(prepared[species_name].density, dtype=jnp.float64), dtype=jnp.float64)
        if use_jax
        else np.zeros_like(prepared[species_name].density, dtype=np.float64)
    )

    if mode == "braginskii":
        if species_type == "electron":
            rate = collision_rates.get((species_name, species_name))
            if rate is not None:
                total += rate
        elif species_type == "ion":
            rate = collision_rates.get((species_name, species_name))
            if rate is not None:
                total += rate
        else:
            raise NotImplementedError("Neutral conduction_collisions_mode='braginskii' is not supported.")
    elif mode == "multispecies":
        for other_name in species:
            rate = collision_rates.get((species_name, other_name))
            if rate is not None:
                total += rate
        if species_name in cx_rates:
            total += cx_rates[species_name]
    elif mode == "afn":
        if species_type != "neutral":
            raise NotImplementedError("Conduction_collisions_mode='afn' is only supported for neutrals.")
        for other_name, other_species in species.items():
            if other_species.charge == 0.0:
                continue
            rate = collision_rates.get((species_name, other_name))
            if rate is not None:
                total += rate
        if species_name in cx_rates:
            total += cx_rates[species_name]
    else:
        raise NotImplementedError(f"Unsupported conduction_collisions_mode={mode!r}.")

    if use_jax:
        return 1.0 / jnp.maximum(total, 1.0e-30)
    return 1.0 / np.maximum(total, 1.0e-30)


def apply_collision_closure(
    config: BoutConfig,
    species: dict[str, OpenFieldSpecies],
    prepared: dict[str, PreparedSpeciesState],
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    collision_rates: dict[tuple[str, str], np.ndarray] | None = None,
    cx_rates: dict[str, np.ndarray] | None = None,
) -> CollisionClosureTerms:
    configured_components = set(configured_component_names(config))
    use_jax = use_jax_backend(
        *(state.density for state in prepared.values()),
        *(state.temperature for state in prepared.values()),
        *(state.velocity for state in prepared.values()),
        *((rate for rate in collision_rates.values()) if collision_rates is not None else ()),
        *((rate for rate in cx_rates.values()) if cx_rates is not None else ()),
    )
    energy_source = {
        name: (
            jnp.zeros_like(jnp.asarray(sp.density, dtype=jnp.float64), dtype=jnp.float64)
            if use_jax
            else np.zeros_like(sp.density, dtype=np.float64)
        )
        for name, sp in species.items()
    }
    momentum_source = {
        name: (
            jnp.zeros_like(jnp.asarray(sp.density, dtype=jnp.float64), dtype=jnp.float64)
            if use_jax
            else np.zeros_like(sp.density, dtype=np.float64)
        )
        for name, sp in species.items()
    }
    diagnostics: dict[str, np.ndarray] = {}
    collision_rates = (
        compute_collision_frequencies(config, species, prepared, dataset_scalars=dataset_scalars)
        if collision_rates is None
        else collision_rates
    )
    cx_rates = (
        charge_exchange_collision_rates(
            config,
            species=species,
            prepared=prepared,
            dataset_scalars=dataset_scalars,
        )
        if cx_rates is None
        else cx_rates
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
                coeff = momentum_coefficient(first_name, first_species.charge, second_name, second_species.charge)
                friction = (
                    coeff
                    * a1
                    * nu_12
                    * first_state.density
                    * (second_state.velocity - first_state.velocity)
                )
                momentum_source[first_name] += friction
                momentum_source[second_name] -= friction
                diagnostics[f"F{first_name}{second_name}_coll"] = friction if use_jax else np.asarray(friction, dtype=np.float64)
                diagnostics[f"F{second_name}{first_name}_coll"] = -friction if use_jax else np.asarray(-friction, dtype=np.float64)

                if first_species.has_pressure or second_species.has_pressure:
                    velocity_delta = second_state.velocity - first_state.velocity
                    first_heating = (a2 / (a1 + a2)) * velocity_delta * friction
                    second_heating = (a1 / (a1 + a2)) * velocity_delta * friction
                    energy_source[first_name] += first_heating
                    energy_source[second_name] += second_heating
                    diagnostics[f"E{first_name}{second_name}_coll_friction"] = (
                        first_heating if use_jax else np.asarray(first_heating, dtype=np.float64)
                    )
                    diagnostics[f"E{second_name}{first_name}_coll_friction"] = (
                        second_heating if use_jax else np.asarray(second_heating, dtype=np.float64)
                    )

            if "braginskii_heat_exchange" in configured_components and (first_species.has_pressure or second_species.has_pressure):
                heat_exchange = 3.0 * (a1 / (a1 + a2)) * nu_12 * first_state.density * (
                    second_state.temperature - first_state.temperature
                )
                energy_source[first_name] += heat_exchange
                energy_source[second_name] -= heat_exchange
                diagnostics[f"E{first_name}{second_name}_coll_heat_exchange"] = (
                    heat_exchange if use_jax else np.asarray(heat_exchange, dtype=np.float64)
                )
                diagnostics[f"E{second_name}{first_name}_coll_heat_exchange"] = (
                    -heat_exchange if use_jax else np.asarray(-heat_exchange, dtype=np.float64)
                )

    if "braginskii_thermal_force" in configured_components and thermal_force_enabled(config, "electron_ion", True):
        electron_temperature_gradient = _grad_par_open(prepared["e"].temperature, mesh=mesh, metrics=metrics)
        for name, sp in species.items():
            if name == "e" or sp.charge <= 0.0:
                continue
            ion_force = prepared[name].density * (0.71 * (sp.charge**2)) * electron_temperature_gradient
            momentum_source[name] += ion_force
            momentum_source["e"] -= ion_force

    if "braginskii_thermal_force" in configured_components and thermal_force_enabled(config, "ion_ion", True):
        ion_names = tuple(name for name, sp in species.items() if name != "e" and sp.charge != 0.0)
        override_mass_restrictions = thermal_force_enabled(
            config,
            "override_ion_mass_restrictions",
            False,
        )
        for index, first_name in enumerate(ion_names):
            for second_name in ion_names[index + 1 :]:
                pair = ion_thermal_force_pair(
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
                momentum_source[heavy_name] += heavy_force
                momentum_source[light_name] -= heavy_force

    if "braginskii_ion_viscosity" in configured_components:
        for name, sp in species.items():
            if name == "e" or sp.charge == 0.0:
                continue
            viscosity_inputs = ion_parallel_viscosity_inputs(
                species_name=name,
                species=species,
                prepared=prepared,
                collision_rates=collision_rates,
                cx_rates=cx_rates,
            )
            viscosity_source = div_par_parallel_ion_viscosity_open(
                viscosity_inputs.eta,
                prepared[name].velocity,
                mesh=mesh,
                metrics=metrics,
            )
            momentum_source[name] += viscosity_source
            viscosity_energy = -prepared[name].velocity * viscosity_source
            energy_source[name] += viscosity_energy
            diagnostics[f"DivPiPar_{name}"] = viscosity_source if use_jax else np.asarray(viscosity_source, dtype=np.float64)
            diagnostics[f"E{name}_viscosity"] = (
                viscosity_energy if use_jax else np.asarray(viscosity_energy, dtype=np.float64)
            )

    if "braginskii_conduction" in configured_components:
        for name, sp in species.items():
            if not sp.has_pressure:
                continue
            if config.has_option(name, "thermal_conduction") and not bool(config.parsed(name, "thermal_conduction")):
                continue
            tau = conduction_collision_time(
                config,
                species=species,
                prepared=prepared,
                collision_rates=collision_rates,
                cx_rates=cx_rates,
                species_name=name,
            )
            kappa_coefficient = conduction_kappa_coefficient(config, sp)
            if use_jax:
                temperature = jnp.asarray(prepared[name].temperature, dtype=jnp.float64)
                pressure = jnp.maximum(jnp.asarray(prepared[name].pressure, dtype=jnp.float64), 0.0)
            else:
                temperature = np.asarray(prepared[name].temperature, dtype=np.float64)
                pressure = np.maximum(np.asarray(prepared[name].pressure, dtype=np.float64), 0.0)
            kappa_par = kappa_coefficient * pressure * tau / sp.atomic_mass
            if mesh.myg > 0:
                kappa_par = apply_noflow_scalar_guards(kappa_par, mesh=mesh, lower_y=True, upper_y=True)
                if not use_jax:
                    kappa_par = np.asarray(kappa_par, dtype=np.float64)
            conduction_energy = _div_par_k_grad_par_open(
                kappa_par,
                temperature,
                mesh=mesh,
                metrics=metrics,
                boundary_flux=False,
            )
            energy_source[name] += conduction_energy
            diagnostics[f"E{name}_conduction"] = (
                conduction_energy if use_jax else np.asarray(conduction_energy, dtype=np.float64)
            )

    return CollisionClosureTerms(
        energy_source=energy_source,
        momentum_source=momentum_source,
        diagnostics=diagnostics,
    )
