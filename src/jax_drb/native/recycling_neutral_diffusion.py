from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from ..config.boutinp import BoutConfig
from .array_backend import use_jax_backend
from .metrics import StructuredMetrics
from .mesh import StructuredMesh
from .neutral_mixed import _div_par_k_grad_par_open
from .recycling_boundaries import (
    apply_open_field_dirichlet_scalar_guards,
    apply_open_field_neumann_scalar_guards,
)
from .recycling_collisions import compute_collision_frequencies
from .recycling_reactions import (
    neutral_charge_exchange_collision_rates,
    neutral_ionisation_collision_rates,
)
from .recycling_setup import OpenFieldSpecies
from .recycling_state import PreparedSpeciesState


@dataclass(frozen=True)
class NeutralParallelDiffusionTerms:
    density_source: dict[str, np.ndarray]
    energy_source: dict[str, np.ndarray]
    momentum_source: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]


def configured_component_names(config: BoutConfig) -> tuple[str, ...]:
    for section in ("model", "hermes"):
        if config.has_section(section) and config.has_option(section, "components"):
            values = config.parsed(section, "components")
            if isinstance(values, tuple):
                return tuple(str(value).strip() for value in values)
            return (str(values).strip(),)
    return ()


def apply_neutral_parallel_diffusion(
    config: BoutConfig,
    *,
    species: dict[str, OpenFieldSpecies],
    prepared: dict[str, PreparedSpeciesState],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    collision_rates: dict[tuple[str, str], np.ndarray] | None = None,
    ionisation_rates: dict[str, np.ndarray] | None = None,
    charge_exchange_rates: dict[str, np.ndarray] | None = None,
) -> NeutralParallelDiffusionTerms:
    use_jax = use_jax_backend(
        *(state.density for state in prepared.values()),
        *(state.pressure for state in prepared.values()),
        *(state.temperature for state in prepared.values()),
        *((rate for rate in collision_rates.values()) if collision_rates is not None else ()),
        *((rate for rate in ionisation_rates.values()) if ionisation_rates is not None else ()),
        *((rate for rate in charge_exchange_rates.values()) if charge_exchange_rates is not None else ()),
        metrics.dy,
        metrics.J,
        metrics.g_22,
    )
    density_source = {
        name: (
            jnp.zeros_like(jnp.asarray(sp.density, dtype=jnp.float64), dtype=jnp.float64)
            if use_jax
            else np.zeros_like(sp.density, dtype=np.float64)
        )
        for name, sp in species.items()
    }
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

    if "neutral_parallel_diffusion" not in set(configured_component_names(config)):
        return NeutralParallelDiffusionTerms(
            density_source=density_source,
            energy_source=energy_source,
            momentum_source=momentum_source,
            diagnostics=diagnostics,
        )

    section = "neutral_parallel_diffusion"
    dneut = float(config.parsed(section, "dneut")) if config.has_option(section, "dneut") else 0.0
    if dneut <= 0.0:
        return NeutralParallelDiffusionTerms(
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

    collision_rates = (
        compute_collision_frequencies(config, species, prepared, dataset_scalars=dataset_scalars)
        if collision_rates is None
        else collision_rates
    )
    ionisation_rates = (
        neutral_ionisation_collision_rates(
            config,
            species=species,
            prepared=prepared,
            dataset_scalars=dataset_scalars,
        )
        if ionisation_rates is None
        else ionisation_rates
    )
    charge_exchange_rates = (
        neutral_charge_exchange_collision_rates(
            config,
            species=species,
            prepared=prepared,
            dataset_scalars=dataset_scalars,
        )
        if charge_exchange_rates is None
        else charge_exchange_rates
    )

    advection_factor = 2.5 if equation_fix else 1.5
    kappa_factor = 2.5 if equation_fix else 1.0

    for name, sp in species.items():
        if name == "e" or sp.charge != 0.0:
            continue

        nu = (
            jnp.zeros_like(jnp.asarray(prepared[name].density, dtype=jnp.float64), dtype=jnp.float64)
            if use_jax
            else np.zeros_like(prepared[name].density, dtype=np.float64)
        )
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

        if use_jax:
            density = jnp.asarray(prepared[name].density, dtype=jnp.float64)
            pressure = jnp.asarray(prepared[name].pressure, dtype=jnp.float64)
            temperature = jnp.asarray(prepared[name].temperature, dtype=jnp.float64)
            momentum = jnp.asarray(prepared[name].momentum, dtype=jnp.float64)
            velocity = jnp.asarray(prepared[name].velocity, dtype=jnp.float64)
            diffusion = dneut * temperature / jnp.maximum(sp.atomic_mass * nu, 1.0e-10)
        else:
            density = np.asarray(prepared[name].density, dtype=np.float64)
            pressure = np.asarray(prepared[name].pressure, dtype=np.float64)
            temperature = np.asarray(prepared[name].temperature, dtype=np.float64)
            momentum = np.asarray(prepared[name].momentum, dtype=np.float64)
            velocity = np.asarray(prepared[name].velocity, dtype=np.float64)
            diffusion = dneut * temperature / np.maximum(sp.atomic_mass * nu, 1.0e-10)

        diffusion = apply_open_field_dirichlet_scalar_guards(
            diffusion,
            mesh=mesh,
            lower_y=sp.noflow_lower_y,
            upper_y=sp.noflow_upper_y,
        )
        log_pressure = jnp.log(jnp.maximum(pressure, 1.0e-7)) if use_jax else np.log(np.maximum(pressure, 1.0e-7))
        log_pressure = apply_open_field_neumann_scalar_guards(
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
        conductivity = apply_open_field_neumann_scalar_guards(
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

        momentum_rhs = (
            jnp.zeros_like(jnp.asarray(density_rhs, dtype=jnp.float64), dtype=jnp.float64)
            if use_jax
            else np.zeros_like(density_rhs, dtype=np.float64)
        )
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
            diagnostics[f"D{name}_Dpar"] = diffusion if use_jax else np.asarray(diffusion, dtype=np.float64)
            diagnostics[f"S{name}_Dpar"] = density_rhs if use_jax else np.asarray(density_rhs, dtype=np.float64)
            diagnostics[f"E{name}_Dpar"] = energy_rhs if use_jax else np.asarray(energy_rhs, dtype=np.float64)
            if sp.has_momentum and perpendicular_viscosity:
                diagnostics[f"F{name}_Dpar"] = momentum_rhs if use_jax else np.asarray(momentum_rhs, dtype=np.float64)

    return NeutralParallelDiffusionTerms(
        density_source=density_source,
        energy_source=energy_source,
        momentum_source=momentum_source,
        diagnostics=diagnostics,
    )
