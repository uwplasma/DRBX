from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from importlib import resources
import math
import re

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver
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
) -> Recycling1DRhsResult:
    species = _initialize_species(config, mesh=mesh)
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
    energy_source["e"] = energy_source["e"] + electron_boundary.energy_source

    recycling_terms = _target_recycling_sources(
        ions=ions,
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

    electron_force = compute_electron_force_balance(
        electron_boundary.pressure,
        prepared["e"].density,
        mesh=mesh,
        dy=jnp.asarray(metrics.dy, dtype=jnp.float64),
        electron_momentum_source=momentum_source["e"],
    )
    for ion in ions:
        momentum_source[ion.name] = momentum_source[ion.name] + np.asarray(
            apply_parallel_electric_force(
                jnp.asarray(prepared[ion.name].density, dtype=jnp.float64),
                charge=ion.charge,
                epar=electron_force.epar,
            ),
            dtype=np.float64,
        )

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
        pressure_rhs = _explicit_pressure_source(config, ion.name, mesh=mesh, dataset_scalars=dataset_scalars)
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
        # The reference momentum operator includes the boundary-cell contribution from
        # the guard-side FV sweep. Re-using the neutral helper needs a signed factor to
        # recover the same last-cell sheath flux on this 1D open-field branch.
        momentum_rhs = 2.0 * ion.atomic_mass * _div_par_fvv_open(
            np.maximum(ion_state.density, 1.0e-7),
            ion_velocity[ion.name],
            fastest_wave,
            mesh=mesh,
            metrics=metrics,
        )
        momentum_rhs = momentum_rhs - _grad_par_open(ion_state.pressure, mesh=mesh, metrics=metrics)
        momentum_rhs = momentum_rhs + momentum_source[ion.name]

        variables[ion.density_name] = ion_state.density[None, ...]
        variables[ion.pressure_name] = ion_state.pressure[None, ...]
        variables[ion.momentum_name] = ion_state.momentum[None, ...]
        variables[f"ddt({ion.density_name})"] = density_rhs[None, ...]
        variables[f"ddt({ion.pressure_name})"] = pressure_rhs[None, ...]
        variables[f"ddt({ion.momentum_name})"] = momentum_rhs[None, ...]

    electron_velocity = _electron_zero_current_velocity(ions, ion_velocity=ion_velocity, electron_density=prepared["e"].density)
    electron_fastest_wave = np.sqrt(np.maximum(prepared["e"].temperature, 0.0) / electron.atomic_mass)
    electron_pressure_rhs = _explicit_pressure_source(config, "e", mesh=mesh, dataset_scalars=dataset_scalars)
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
        variables[neutral.density_name] = neutral_state.density[None, ...]
        variables[neutral.pressure_name] = neutral_state.pressure[None, ...]
        variables[neutral.momentum_name] = neutral_state.momentum[None, ...]
        variables[f"ddt({neutral.density_name})"] = density_source[neutral.name][None, ...]
        variables[f"ddt({neutral.pressure_name})"] = ((2.0 / 3.0) * energy_source[neutral.name])[None, ...]
        variables[f"ddt({neutral.momentum_name})"] = momentum_source[neutral.name][None, ...]

    for name, value in diagnostics.items():
        variables[name] = value[None, ...]

    return Recycling1DRhsResult(variables=variables)


def _initialize_species(config: BoutConfig, *, mesh: StructuredMesh) -> dict[str, OpenFieldSpecies]:
    resolver = NumericResolver(config)
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
        density = _evaluate_field_option(config, f"N{name}", mesh=mesh) if config.has_section(f"N{name}") else None
        if name == "e":
            if density is None:
                density = None
            pressure = _evaluate_field_option(config, "Pe", mesh=mesh)
            momentum = np.zeros_like(pressure, dtype=np.float64)
        else:
            if density is None:
                raise KeyError(f"Missing density section for {name}.")
            pressure = _evaluate_field_option(config, f"P{name}", mesh=mesh) if config.has_section(f"P{name}") else density.copy()
            momentum = _evaluate_field_option(config, f"NV{name}", mesh=mesh) if config.has_section(f"NV{name}") else np.zeros_like(density, dtype=np.float64)

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
            has_pressure="evolve_pressure" in components or name == "e" or "neutral_mixed" in components,
            has_momentum="evolve_momentum" in components or "neutral_mixed" in components,
            noflow_lower_y=bool(config.parsed(name, "noflow_lower_y")) if config.has_option(name, "noflow_lower_y") else noflow,
            noflow_upper_y=bool(config.parsed(name, "noflow_upper_y")) if config.has_option(name, "noflow_upper_y") else noflow,
            target_recycle=bool(config.parsed(name, "target_recycle")) if config.has_option(name, "target_recycle") else False,
            recycle_as=str(config.parsed(name, "recycle_as")) if config.has_option(name, "recycle_as") else None,
            target_recycle_multiplier=float(resolver.resolve(name, "target_recycle_multiplier")) if config.has_option(name, "target_recycle_multiplier") else 0.0,
            target_recycle_energy=float(resolver.resolve(name, "target_recycle_energy")) if config.has_option(name, "target_recycle_energy") else 0.0,
            target_fast_recycle_fraction=float(resolver.resolve(name, "target_fast_recycle_fraction")) if config.has_option(name, "target_fast_recycle_fraction") else 0.0,
            target_fast_recycle_energy_factor=float(resolver.resolve(name, "target_fast_recycle_energy_factor")) if config.has_option(name, "target_fast_recycle_energy_factor") else 0.0,
        )

    for name, sp in tuple(species.items()):
        density = sp.density
        pressure = sp.pressure
        momentum = sp.momentum
        if sp.noflow_lower_y:
            density = np.asarray(apply_noflow_scalar_guards(density, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64)
            pressure = np.asarray(apply_noflow_scalar_guards(pressure, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64)
            momentum = np.asarray(apply_noflow_flow_guards(momentum, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64)
        if sp.noflow_upper_y:
            density = np.asarray(apply_noflow_scalar_guards(density, mesh=mesh, lower_y=False, upper_y=True), dtype=np.float64)
            pressure = np.asarray(apply_noflow_scalar_guards(pressure, mesh=mesh, lower_y=False, upper_y=True), dtype=np.float64)
            momentum = np.asarray(apply_noflow_flow_guards(momentum, mesh=mesh, lower_y=False, upper_y=True), dtype=np.float64)
        species[name] = OpenFieldSpecies(**{**sp.__dict__, "density": density, "pressure": pressure, "momentum": momentum})
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


def _electron_density(ions: tuple[OpenFieldSpecies, ...]) -> np.ndarray:
    density = np.zeros_like(ions[0].density, dtype=np.float64)
    for ion in ions:
        density = density + ion.charge * ion.density
    return density


def _safe_temperature(pressure: np.ndarray, density: np.ndarray, density_floor: float = 1.0e-8) -> np.ndarray:
    return np.asarray(pressure, dtype=np.float64) / np.maximum(np.asarray(density, dtype=np.float64), density_floor)


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


@dataclass(frozen=True)
class _CollisionClosureTerms:
    energy_source: dict[str, np.ndarray]
    momentum_source: dict[str, np.ndarray]


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
        elif len(lhs) == 2 and len(rhs) == 2 and lhs[0] == rhs[1] and lhs[1] == rhs[0]:
            atom1 = lhs[0]
            ion1 = lhs[1]
            ion2 = rhs[0]
            atom2 = rhs[1]
            result = _charge_exchange(atom1, ion1, atom2, ion2, species=species, dataset_scalars=dataset_scalars)
            _accumulate_terms(result, density_source, energy_source, momentum_source, diagnostics)
    return _ReactionTerms(density_source=density_source, energy_source=energy_source, momentum_source=momentum_source, diagnostics=diagnostics)


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
    temperature = _safe_temperature(pressure, density)
    velocity = species.momentum / np.maximum(species.atomic_mass * density, 1.0e-8)
    momentum = species.momentum.copy()

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
    return _PreparedSpeciesState(
        density=density,
        pressure=pressure,
        temperature=temperature,
        velocity=velocity,
        momentum=momentum,
    )


def _prepare_open_field_states(
    species: dict[str, OpenFieldSpecies],
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> tuple[dict[str, _PreparedSpeciesState], _IonBoundaryResult, _ElectronBoundaryResult]:
    prepared = {name: _prepare_species_state(sp, mesh=mesh) for name, sp in species.items()}
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    electron_density = np.zeros_like(species["e"].density, dtype=np.float64)
    for ion in ions:
        electron_density = electron_density + ion.charge * prepared[ion.name].density

    ion_boundary = _apply_ion_sheath_boundary(
        ions,
        electron_pressure=prepared["e"].pressure,
        electron_density=electron_density,
        mesh=mesh,
        metrics=metrics,
    )
    for ion in ions:
        prepared[ion.name] = _PreparedSpeciesState(
            density=ion_boundary.density[ion.name],
            pressure=ion_boundary.pressure[ion.name],
            temperature=ion_boundary.temperature[ion.name],
            velocity=ion_boundary.velocity[ion.name],
            momentum=ion_boundary.momentum[ion.name],
        )

    electron_density = np.zeros_like(species["e"].density, dtype=np.float64)
    for ion in ions:
        electron_density = electron_density + ion.charge * prepared[ion.name].density

    electron_boundary = _apply_electron_sheath_boundary(
        electron_pressure=prepared["e"].pressure,
        electron_density=electron_density,
        ion_velocity={ion.name: prepared[ion.name].velocity for ion in ions},
        ions=ions,
        mesh=mesh,
        metrics=metrics,
    )
    prepared["e"] = _PreparedSpeciesState(
        density=electron_boundary.density,
        pressure=electron_boundary.pressure,
        temperature=electron_boundary.temperature,
        velocity=np.zeros_like(electron_boundary.density, dtype=np.float64),
        momentum=np.zeros_like(electron_boundary.density, dtype=np.float64),
    )
    return prepared, ion_boundary, electron_boundary


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
    electron_neutral = bool(config.parsed("braginskii_collisions", "electron_neutral")) if config.has_option("braginskii_collisions", "electron_neutral") else False
    ion_ion = bool(config.parsed("braginskii_collisions", "ion_ion")) if config.has_option("braginskii_collisions", "ion_ion") else True
    ion_neutral = bool(config.parsed("braginskii_collisions", "ion_neutral")) if config.has_option("braginskii_collisions", "ion_neutral") else False
    neutral_neutral = bool(config.parsed("braginskii_collisions", "neutral_neutral")) if config.has_option("braginskii_collisions", "neutral_neutral") else True

    electron = species["e"]
    electron_state = prepared["e"]
    te_ev = electron_state.temperature * tnorm
    ne_m3 = electron_state.density * nnorm

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

    charged_names = [name for name, sp in species.items() if name != "e" and sp.charge != 0.0]
    neutral_names = [name for name, sp in species.items() if name != "e" and sp.charge == 0.0]

    for charged_name in charged_names:
        charged_species = species[charged_name]
        charged_state = prepared[charged_name]
        temperature_ev = charged_state.temperature * tnorm
        density_m3 = charged_state.density * nnorm
        charge = charged_species.charge
        atomic_mass = charged_species.atomic_mass
        mass = atomic_mass * _MP
        temperature_limited = np.maximum(temperature_ev, 0.1)
        density_limited = np.maximum(density_m3, 1.0e10)
        coulomb_log = 29.91 - np.log(
            ((charge * charge * (2.0 * atomic_mass)) / (2.0 * atomic_mass * temperature_limited))
            * np.sqrt(2.0 * density_limited * (charge * charge) / temperature_limited)
        )
        v_sq = 2.0 * temperature_limited * _QE / mass
        nu_self = (
            (((charge * _QE) * (charge * _QE)) ** 2)
            * density_limited
            * np.maximum(coulomb_log, 1.0)
            * 2.0
            / (3.0 * np.power(math.pi * (2.0 * v_sq), 1.5) * ((_EPS0 * mass) ** 2))
        )
        collision_rates[(charged_name, charged_name)] = np.asarray(nu_self / omega_ci, dtype=np.float64)

    for charged_name in charged_names:
        charged_species = species[charged_name]
        charged_state = prepared[charged_name]
        for neutral_name in neutral_names:
            neutral_species = species[neutral_name]
            neutral_state = prepared[neutral_name]
            vrel = np.sqrt(
                charged_state.temperature / charged_species.atomic_mass
                + neutral_state.temperature / neutral_species.atomic_mass
            )
        if ion_neutral:
            nu_cn = vrel * neutral_state.density * nnorm * 5.0e-19 * rho_s0
            nu_cn = np.asarray(nu_cn, dtype=np.float64)
            collision_rates[(charged_name, neutral_name)] = nu_cn
            collision_rates[(neutral_name, charged_name)] = (
                nu_cn
                * (charged_species.atomic_mass / neutral_species.atomic_mass)
                * charged_state.density
                / np.maximum(neutral_state.density, 1.0e-5)
            )

    for index, first_name in enumerate(charged_names):
        if not ion_ion:
            break
        first_species = species[first_name]
        first_state = prepared[first_name]
        t1_ev = first_state.temperature * tnorm
        n1_m3 = first_state.density * nnorm
        for second_name in charged_names[index + 1 :]:
            second_species = species[second_name]
            second_state = prepared[second_name]
            t2_ev = second_state.temperature * tnorm
            n2_m3 = second_state.density * nnorm
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
            nu_12 = np.asarray(nu_12 / omega_ci, dtype=np.float64)
            collision_rates[(first_name, second_name)] = nu_12
            collision_rates[(second_name, first_name)] = (
                nu_12
                * (first_species.atomic_mass / second_species.atomic_mass)
                * first_state.density
                / np.maximum(second_state.density, 1.0e-5)
            )

    for index, first_name in enumerate(neutral_names):
        if not neutral_neutral:
            break
        first_species = species[first_name]
        first_state = prepared[first_name]
        for second_name in neutral_names[index + 1 :]:
            second_species = species[second_name]
            second_state = prepared[second_name]
            vrel = np.sqrt(
                first_state.temperature / first_species.atomic_mass
                + second_state.temperature / second_species.atomic_mass
            )
            nu_12 = vrel * second_state.density * nnorm * (math.pi * (2.8e-10**2)) * rho_s0
            nu_12 = np.asarray(nu_12, dtype=np.float64)
            collision_rates[(first_name, second_name)] = nu_12
            collision_rates[(second_name, first_name)] = (
                nu_12
                * (first_species.atomic_mass / second_species.atomic_mass)
                * first_state.density
                / np.maximum(second_state.density, 1.0e-5)
            )
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

                if first_species.has_pressure or second_species.has_pressure:
                    velocity_delta = second_state.velocity - first_state.velocity
                    energy_source[first_name] = energy_source[first_name] + (
                        (a2 / (a1 + a2)) * velocity_delta * friction
                    )
                    energy_source[second_name] = energy_source[second_name] + (
                        (a1 / (a1 + a2)) * velocity_delta * friction
                    )

            if "braginskii_heat_exchange" in configured_components and (first_species.has_pressure or second_species.has_pressure):
                heat_exchange = 3.0 * (a1 / (a1 + a2)) * nu_12 * first_state.density * (
                    second_state.temperature - first_state.temperature
                )
                energy_source[first_name] = energy_source[first_name] + heat_exchange
                energy_source[second_name] = energy_source[second_name] - heat_exchange

    if "braginskii_thermal_force" in configured_components:
        electron_temperature_gradient = _grad_par_open(prepared["e"].temperature, mesh=mesh, metrics=metrics)
        for name, sp in species.items():
            if name == "e" or sp.charge <= 0.0:
                continue
            ion_force = prepared[name].density * (0.71 * (sp.charge**2)) * electron_temperature_gradient
            momentum_source[name] = momentum_source[name] + ion_force
            momentum_source["e"] = momentum_source["e"] - ion_force

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
            viscosity_source = _div_par_k_grad_par_open(
                eta,
                prepared[name].velocity,
                mesh=mesh,
                metrics=metrics,
                boundary_flux=True,
            )
            momentum_source[name] = momentum_source[name] + viscosity_source
            energy_source[name] = energy_source[name] - prepared[name].velocity * viscosity_source

    return _CollisionClosureTerms(energy_source=energy_source, momentum_source=momentum_source)


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
    electron_temperature = _safe_temperature(electron_pressure, electron_density)
    atom_temperature = _safe_temperature(atom.pressure, atom.density)
    sigma_v, sigma_v_E, electron_heating = _load_amjuel_rate(atom_name, "iz")
    rate = _amjuel_reaction_rate(atom.density, electron_density, electron_temperature, sigma_v, dataset_scalars)
    radiation = _amjuel_energy_loss(atom.density, electron_density, electron_temperature, sigma_v_E, electron_heating, rate, dataset_scalars)

    density_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    energy_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    momentum_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}

    density_source[atom_name] -= rate
    density_source[ion_name] += rate
    energy_source[atom_name] -= 1.5 * rate * atom_temperature
    energy_source[ion_name] += 1.5 * rate * atom_temperature
    energy_source["e"] -= radiation
    diagnostics = {
        f"S{ion_name}_iz": rate,
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
    electron_temperature = _safe_temperature(electron_pressure, electron_density)
    ion_temperature = _safe_temperature(ion.pressure, ion.density)
    sigma_v, sigma_v_E, electron_heating = _load_amjuel_rate(atom_name, "rec")
    rate = _amjuel_reaction_rate(ion.density, electron_density, electron_temperature, sigma_v, dataset_scalars)
    radiation = _amjuel_energy_loss(ion.density, electron_density, electron_temperature, sigma_v_E, electron_heating, rate, dataset_scalars)

    density_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    energy_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    momentum_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}

    density_source[ion_name] -= rate
    density_source[atom_name] += rate
    energy_source[ion_name] -= 1.5 * rate * ion_temperature
    energy_source[atom_name] += 1.5 * rate * ion_temperature
    energy_source["e"] -= radiation
    diagnostics = {
        f"S{ion_name}_rec": -rate,
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
    atom_temperature = _safe_temperature(atom1.pressure, atom1.density)
    ion_temperature = _safe_temperature(ion1.pressure, ion1.density)
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
            f"F{diag_suffix}": np.zeros_like(rate, dtype=np.float64),
            f"K{diag_suffix}": ion1.density * sigmav,
        }
    else:
        diagnostics = {
            f"S{diag_suffix}": -rate,
            f"E{diag_suffix}": -atom_energy,
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
        if not (len(lhs) == 2 and len(rhs) == 2 and lhs[0] == rhs[1] and lhs[1] == rhs[0]):
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
        ion_rate = prepared[atom1_name].density * sigmav
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
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> _IonBoundaryResult:
    te = _safe_temperature(electron_pressure, electron_density)
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
        temperature = _safe_temperature(pressure, density)
        vel = ion.momentum / np.maximum(ion.atomic_mass * density, 1.0e-8)
        if ion.noflow_lower_y:
            density = np.array(apply_noflow_scalar_guards(density, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64, copy=True)
            temperature = np.array(apply_noflow_scalar_guards(temperature, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64, copy=True)
            pressure = np.array(apply_noflow_scalar_guards(pressure, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64, copy=True)
            vel = np.array(apply_noflow_flow_guards(vel, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64, copy=True)

        j = mesh.yend
        jp = j + 1
        jm = j - 1
        ni_i = density[:, j, :]
        ni_m = density[:, jm, :]
        ne_i = electron_density[:, j, :]
        ne_m = electron_density[:, jm, :]
        density[:, jp, :] = np.asarray(limit_free(jnp.asarray(ni_m), jnp.asarray(ni_i), 0), dtype=np.float64)
        temperature[:, jp, :] = np.asarray(limit_free(jnp.asarray(temperature[:, jm, :]), jnp.asarray(temperature[:, j, :]), 0), dtype=np.float64)
        pressure[:, jp, :] = np.asarray(limit_free(jnp.asarray(pressure[:, jm, :]), jnp.asarray(pressure[:, j, :]), 0), dtype=np.float64)

        nisheath = 0.5 * (density[:, jp, :] + density[:, j, :])
        nesheath = 0.5 * (electron_density[:, jp, :] + electron_density[:, j, :]) if jp < electron_density.shape[1] else 0.5 * (electron_density[:, j, :] + electron_density[:, j, :])
        if jp >= electron_density.shape[1]:
            nesheath = 0.5 * (electron_density[:, jm, :] + electron_density[:, j, :])
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
        momentum_field = ion.momentum.copy()
        momentum_field[:, jp, :] = 2.0 * ion.atomic_mass * nisheath * visheath - momentum_field[:, j, :]

        q = ((gamma_i - 1.0 - 1.0 / ((5.0 / 3.0) - 1.0)) * tisheath - 0.5 * c_i_sq * ion.atomic_mass) * nisheath * visheath
        q = np.maximum(q, 0.0)
        flux = q * (J[:, j, :] + J[:, jp, :]) / (np.sqrt(g22[:, j, :]) + np.sqrt(g22[:, jp, :]))
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
    energy_source: np.ndarray


def _apply_electron_sheath_boundary(
    *,
    electron_pressure: np.ndarray,
    electron_density: np.ndarray,
    ion_velocity: dict[str, np.ndarray],
    ions: tuple[OpenFieldSpecies, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> _ElectronBoundaryResult:
    density = np.array(apply_noflow_scalar_guards(electron_density, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64, copy=True)
    pressure = np.array(apply_noflow_scalar_guards(electron_pressure, mesh=mesh, lower_y=True, upper_y=False), dtype=np.float64, copy=True)
    temperature = _safe_temperature(pressure, density)

    j = mesh.yend
    jp = j + 1
    jm = j - 1
    density[:, jp, :] = np.asarray(limit_free(jnp.asarray(density[:, jm, :]), jnp.asarray(density[:, j, :]), 0), dtype=np.float64)
    temperature[:, jp, :] = np.asarray(limit_free(jnp.asarray(temperature[:, jm, :]), jnp.asarray(temperature[:, j, :]), 0), dtype=np.float64)
    pressure[:, jp, :] = np.asarray(limit_free(jnp.asarray(pressure[:, jm, :]), jnp.asarray(pressure[:, j, :]), 0), dtype=np.float64)

    ion_sum = np.zeros_like(density[:, j, :], dtype=np.float64)
    for ion in ions:
        ti = _safe_temperature(ion.pressure, ion.density)
        ni = ion.density
        s_i = np.clip(0.5 * (3.0 * ni[:, j, :] / np.maximum(density[:, j, :], 1.0e-12) - ni[:, jm, :] / np.maximum(density[:, jm, :], 1.0e-12)), 0.0, 1.0)
        s_i = np.where(np.isfinite(s_i), s_i, 1.0)
        grad_ne = density[:, j, :] - density[:, jm, :]
        grad_ni = ni[:, j, :] - ni[:, jm, :]
        mask = np.abs(grad_ni) < 2.0e-3
        grad_ne = np.where(mask, 2.0e-3, grad_ne)
        grad_ni = np.where(mask, 2.0e-3, grad_ni)
        c_i_sq = np.clip(((5.0 / 3.0) * ti[:, j, :] + ion.charge * s_i * temperature[:, j, :] * grad_ne / grad_ni) / ion.atomic_mass, 0.0, 100.0)
        ion_sum = ion_sum + s_i * ion.charge * np.sqrt(c_i_sq)

    me = 1.0 / 1836.0
    phi = np.zeros_like(density, dtype=np.float64)
    valid = temperature[:, j, :] > 0.0
    phi[:, j, :] = np.where(valid, temperature[:, j, :] * np.log(np.sqrt(temperature[:, j, :] / (me * (2.0 * math.pi))) / np.maximum(ion_sum, 1.0e-12)), 0.0)
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
    q = ((gamma_e - 1.0 - 1.0 / ((5.0 / 3.0) - 1.0)) * tesheath - 0.5 * me * np.square(vesheath)) * nesheath * vesheath
    q = np.maximum(q, 0.0)
    flux = q * (np.asarray(metrics.J)[:, j, :] + np.asarray(metrics.J)[:, jp, :]) / (
        np.sqrt(np.asarray(metrics.g_22)[:, j, :]) + np.sqrt(np.asarray(metrics.g_22)[:, jp, :])
    )
    power = flux / (np.asarray(metrics.dy)[:, j, :] * np.asarray(metrics.J)[:, j, :])
    energy_source = np.zeros_like(density, dtype=np.float64)
    energy_source[:, j, :] -= power
    return _ElectronBoundaryResult(
        density=density,
        temperature=temperature,
        pressure=pressure,
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
        result = compute_target_recycling_sources(
            jnp.asarray(ion.density, dtype=jnp.float64),
            jnp.asarray(ion_velocity[ion.name], dtype=jnp.float64),
            jnp.asarray(_safe_temperature(ion.pressure, ion.density), dtype=jnp.float64),
            mesh=mesh,
            J=jnp.asarray(metrics.J, dtype=jnp.float64),
            dy=jnp.asarray(metrics.dy, dtype=jnp.float64),
            dx=jnp.asarray(metrics.dx, dtype=jnp.float64),
            dz=jnp.asarray(metrics.dz, dtype=jnp.float64),
            g_22=jnp.asarray(metrics.g_22, dtype=jnp.float64),
            target_multiplier=ion.target_recycle_multiplier,
            target_energy=ion.target_recycle_energy,
            gamma_i=0.0,
            target_fast_recycle_fraction=ion.target_fast_recycle_fraction,
            target_fast_recycle_energy_factor=ion.target_fast_recycle_energy_factor,
            lower_y=False,
            upper_y=True,
        )
        density_source[neutral.name] = density_source[neutral.name] + np.asarray(result.density_source, dtype=np.float64)
        energy_source[neutral.name] = energy_source[neutral.name] + np.asarray(result.energy_source, dtype=np.float64)
        diagnostics[f"S{neutral.name}_target_recycle"] = np.asarray(result.target_density_source, dtype=np.float64)
        diagnostics[f"E{neutral.name}_target_recycle"] = np.asarray(result.target_energy_source, dtype=np.float64)

    return _RecyclingTerms(density_source=density_source, energy_source=energy_source, diagnostics=diagnostics)


def _electron_zero_current_velocity(
    ions: tuple[OpenFieldSpecies, ...],
    *,
    ion_velocity: dict[str, np.ndarray],
    electron_density: np.ndarray,
) -> np.ndarray:
    current = np.zeros_like(electron_density, dtype=np.float64)
    for ion in ions:
        current = current + ion.charge * ion.density * ion_velocity[ion.name]
    return current / np.maximum(electron_density, 1.0e-5)
