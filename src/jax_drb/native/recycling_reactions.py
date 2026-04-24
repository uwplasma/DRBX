from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from ..config.boutinp import BoutConfig
from .recycling_atomic import (
    OPENADAS_FILENAMES,
    amjuel_reaction_rate_and_energy_loss,
    charge_exchange_rate_multiplier,
    eval_amjuel_fit,
    eval_openadas_rate,
    hydrogen_cx_sigmav,
    load_amjuel_rate,
    load_openadas_rate,
    openadas_energy_loss,
    openadas_reaction_rate,
)


@dataclass(frozen=True)
class ReactionTerms:
    density_source: dict[str, np.ndarray]
    energy_source: dict[str, np.ndarray]
    momentum_source: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]


def _soft_floor(value: np.ndarray, minimum: float) -> np.ndarray:
    value_array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    minimum_value = float(minimum)
    return value_array + minimum_value * np.exp(-value_array / minimum_value)


def _safe_temperature(pressure: np.ndarray, density: np.ndarray, density_floor: float = 1.0e-8) -> np.ndarray:
    pressure_floor = np.maximum(np.asarray(pressure, dtype=np.float64), 0.0)
    return pressure_floor / _soft_floor(np.asarray(density, dtype=np.float64), density_floor)


def is_charge_exchange_reaction(lhs: tuple[str, ...], rhs: tuple[str, ...]) -> bool:
    if len(lhs) != 2 or len(rhs) != 2:
        return False
    atom1, ion1 = lhs
    ion2, atom2 = rhs
    if not ion1.endswith("+") or not ion2.endswith("+"):
        return False
    return ion2[:-1] == atom1 and ion1[:-1] == atom2


def accumulate_terms(
    result: ReactionTerms,
    density_source: dict[str, np.ndarray],
    energy_source: dict[str, np.ndarray],
    momentum_source: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
) -> None:
    for name, value in result.density_source.items():
        density_source[name] += value
    for name, value in result.energy_source.items():
        energy_source[name] += value
    for name, value in result.momentum_source.items():
        momentum_source[name] += value
    diagnostics.update(result.diagnostics)


def _initialize_terms(
    species: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    density_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    energy_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    momentum_source = {name: np.zeros_like(sp.density, dtype=np.float64) for name, sp in species.items()}
    diagnostics: dict[str, np.ndarray] = {}
    return density_source, energy_source, momentum_source, diagnostics


def _accumulate_amjuel_ionisation(
    atom_name: str,
    ion_name: str,
    *,
    species: dict[str, Any],
    electron_density: np.ndarray,
    dataset_scalars: dict[str, float],
    density_source: dict[str, np.ndarray],
    energy_source: dict[str, np.ndarray],
    momentum_source: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
) -> None:
    atom = species[atom_name]
    electron_pressure = species["e"].pressure
    electron_temperature = _safe_temperature(electron_pressure, electron_density, species["e"].density_floor)
    atom_temperature = _safe_temperature(atom.pressure, atom.density, atom.density_floor)
    if (atom_name, "iz") in OPENADAS_FILENAMES:
        rate = openadas_reaction_rate(atom.density, electron_density, electron_temperature, atom_name, "iz", dataset_scalars)
        radiation = openadas_energy_loss(
            atom.density,
            electron_density,
            electron_temperature,
            atom_name,
            "iz",
            reaction_rate=rate,
            dataset_scalars=dataset_scalars,
        )
    else:
        sigma_v, sigma_v_E, electron_heating = load_amjuel_rate(atom_name, "iz")
        rate, radiation = amjuel_reaction_rate_and_energy_loss(
            atom.density,
            electron_density,
            electron_temperature,
            sigma_v,
            sigma_v_E,
            electron_heating,
            dataset_scalars,
        )
    atom_velocity = atom.momentum / np.maximum(atom.atomic_mass * atom.density, 1.0e-8)
    ion_momentum = rate * atom.atomic_mass * atom_velocity
    density_source[atom_name] -= rate
    density_source[ion_name] += rate
    momentum_source[atom_name] -= ion_momentum
    momentum_source[ion_name] += ion_momentum
    atom_energy = 1.5 * rate * atom_temperature
    energy_source[atom_name] -= atom_energy
    energy_source[ion_name] += atom_energy
    energy_source["e"] -= radiation
    diagnostics[f"S{ion_name}_iz"] = rate
    diagnostics[f"F{ion_name}_iz"] = ion_momentum
    diagnostics[f"E{ion_name}_iz"] = atom_energy
    diagnostics[f"R{ion_name}_ex"] = -radiation


def _accumulate_amjuel_recombination(
    atom_name: str,
    ion_name: str,
    *,
    species: dict[str, Any],
    electron_density: np.ndarray,
    dataset_scalars: dict[str, float],
    density_source: dict[str, np.ndarray],
    energy_source: dict[str, np.ndarray],
    momentum_source: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
) -> None:
    atom = species[atom_name]
    ion = species[ion_name]
    electron_pressure = species["e"].pressure
    electron_temperature = _safe_temperature(electron_pressure, electron_density, species["e"].density_floor)
    ion_temperature = _safe_temperature(ion.pressure, ion.density, ion.density_floor)
    if (atom_name, "rec") in OPENADAS_FILENAMES:
        rate = openadas_reaction_rate(ion.density, electron_density, electron_temperature, atom_name, "rec", dataset_scalars)
        radiation = openadas_energy_loss(
            ion.density,
            electron_density,
            electron_temperature,
            atom_name,
            "rec",
            reaction_rate=rate,
            dataset_scalars=dataset_scalars,
        )
    else:
        sigma_v, sigma_v_E, electron_heating = load_amjuel_rate(atom_name, "rec")
        rate, radiation = amjuel_reaction_rate_and_energy_loss(
            ion.density,
            electron_density,
            electron_temperature,
            sigma_v,
            sigma_v_E,
            electron_heating,
            dataset_scalars,
        )
    ion_velocity = ion.momentum / np.maximum(ion.atomic_mass * ion.density, 1.0e-8)
    ion_momentum = rate * ion.atomic_mass * ion_velocity
    density_source[ion_name] -= rate
    density_source[atom_name] += rate
    momentum_source[ion_name] -= ion_momentum
    momentum_source[atom_name] += ion_momentum
    ion_energy = 1.5 * rate * ion_temperature
    energy_source[ion_name] -= ion_energy
    energy_source[atom_name] += ion_energy
    energy_source["e"] -= radiation
    diagnostics[f"S{ion_name}_rec"] = -rate
    diagnostics[f"F{ion_name}_rec"] = -ion_momentum
    diagnostics[f"E{ion_name}_rec"] = -ion_energy
    diagnostics[f"R{ion_name}_rec"] = -radiation


def _accumulate_charge_exchange(
    atom1_name: str,
    ion1_name: str,
    atom2_name: str,
    ion2_name: str,
    *,
    config: BoutConfig,
    species: dict[str, Any],
    dataset_scalars: dict[str, float],
    density_source: dict[str, np.ndarray],
    energy_source: dict[str, np.ndarray],
    momentum_source: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
) -> None:
    atom1 = species[atom1_name]
    ion1 = species[ion1_name]
    atom2 = species[atom2_name]
    ion2 = species[ion2_name]
    atom_temperature = _safe_temperature(atom1.pressure, atom1.density, atom1.density_floor)
    ion_temperature = _safe_temperature(ion1.pressure, ion1.density, ion1.density_floor)
    teff = np.clip((atom_temperature / atom1.atomic_mass + ion_temperature / ion1.atomic_mass) * dataset_scalars["Tnorm"], 0.01, 10000.0)
    sigmav = hydrogen_cx_sigmav(teff, dataset_scalars) * charge_exchange_rate_multiplier(config, atom_name=atom1_name)
    rate = atom1.density * ion1.density * sigmav
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
        diagnostics[f"E{diag_suffix}"] = ion_energy - atom_energy
        diagnostics[f"F{diag_suffix}"] = ion_momentum - atom_momentum
        diagnostics[f"K{diag_suffix}"] = ion1.density * sigmav
    else:
        diagnostics[f"S{diag_suffix}"] = -rate
        diagnostics[f"F{diag_suffix}"] = -atom_momentum
        diagnostics[f"F{ion1_name}{atom1_name}_cx"] = -ion_momentum
        diagnostics[f"E{diag_suffix}"] = -atom_energy
        diagnostics[f"E{ion1_name}{atom1_name}_cx"] = -ion_energy
        diagnostics[f"K{diag_suffix}"] = ion1.density * sigmav


def reaction_sources(
    config: BoutConfig,
    *,
    species: dict[str, Any],
    electron_density: np.ndarray,
    dataset_scalars: dict[str, float],
) -> ReactionTerms:
    density_source, energy_source, momentum_source, diagnostics = _initialize_terms(species)

    if not config.has_section("reactions") or not config.has_option("reactions", "type"):
        return ReactionTerms(density_source=density_source, energy_source=energy_source, momentum_source=momentum_source, diagnostics=diagnostics)

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
            _accumulate_amjuel_ionisation(
                atom,
                ion,
                species=species,
                electron_density=electron_density,
                dataset_scalars=dataset_scalars,
                density_source=density_source,
                energy_source=energy_source,
                momentum_source=momentum_source,
                diagnostics=diagnostics,
            )
        elif len(lhs) == 2 and lhs[1] == "e" and len(rhs) == 1:
            ion = lhs[0]
            atom = rhs[0]
            _accumulate_amjuel_recombination(
                atom,
                ion,
                species=species,
                electron_density=electron_density,
                dataset_scalars=dataset_scalars,
                density_source=density_source,
                energy_source=energy_source,
                momentum_source=momentum_source,
                diagnostics=diagnostics,
            )
        elif is_charge_exchange_reaction(lhs, rhs):
            atom1 = lhs[0]
            ion1 = lhs[1]
            ion2 = rhs[0]
            atom2 = rhs[1]
            _accumulate_charge_exchange(
                atom1,
                ion1,
                atom2,
                ion2,
                config=config,
                species=species,
                dataset_scalars=dataset_scalars,
                density_source=density_source,
                energy_source=energy_source,
                momentum_source=momentum_source,
                diagnostics=diagnostics,
            )
    return ReactionTerms(density_source=density_source, energy_source=energy_source, momentum_source=momentum_source, diagnostics=diagnostics)


def neutral_ionisation_collision_rates(
    config: BoutConfig,
    *,
    species: dict[str, Any],
    prepared: dict[str, Any],
    dataset_scalars: dict[str, float],
) -> dict[str, np.ndarray]:
    if not config.has_section("reactions") or not config.has_option("reactions", "type"):
        return {}
    totals: dict[str, np.ndarray] = {}
    electron_density = np.zeros_like(prepared[next(name for name, sp in species.items() if sp.charge > 0.0)].density, dtype=np.float64)
    for sp in species.values():
        if sp.charge > 0.0:
            electron_density += sp.charge * np.asarray(prepared[sp.name].density, dtype=np.float64)
    electron_temperature = _safe_temperature(species["e"].pressure, electron_density, species["e"].density_floor)
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
        if (atom_name, "iz") in OPENADAS_FILENAMES:
            rate_coeff, _, log_temperature, log_density, _ = load_openadas_rate(atom_name, "iz")
            sigma_v = eval_openadas_rate(
                np.asarray(electron_temperature, dtype=np.float64) * dataset_scalars["Tnorm"],
                np.asarray(electron_density, dtype=np.float64) * dataset_scalars["Nnorm"],
                rate_coeff,
                log_temperature=log_temperature,
                log_density=log_density,
            )
        else:
            sigma_v_coeffs, _, _ = load_amjuel_rate(atom_name, "iz")
            sigma_v = eval_amjuel_fit(
                np.asarray(electron_temperature, dtype=np.float64) * dataset_scalars["Tnorm"],
                np.asarray(electron_density, dtype=np.float64) * dataset_scalars["Nnorm"],
                sigma_v_coeffs,
            )
        totals[atom_name] = np.asarray(
            np.asarray(electron_density, dtype=np.float64) * sigma_v * (dataset_scalars["Nnorm"] / dataset_scalars["Omega_ci"]),
            dtype=np.float64,
        )
    return totals


def neutral_charge_exchange_collision_rates(
    config: BoutConfig,
    *,
    species: dict[str, Any],
    prepared: dict[str, Any],
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
        if not is_charge_exchange_reaction(lhs, rhs):
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
        sigma_v = hydrogen_cx_sigmav(teff, dataset_scalars)
        if atom_name in totals:
            totals[atom_name] += prepared[ion_name].density * sigma_v
        else:
            totals[atom_name] = np.asarray(prepared[ion_name].density * sigma_v, dtype=np.float64)
    return totals


def amjuel_ionisation(
    atom_name: str,
    ion_name: str,
    *,
    species: dict[str, Any],
    electron_density: np.ndarray,
    dataset_scalars: dict[str, float],
) -> ReactionTerms:
    density_source, energy_source, momentum_source, diagnostics = _initialize_terms(species)
    _accumulate_amjuel_ionisation(
        atom_name,
        ion_name,
        species=species,
        electron_density=electron_density,
        dataset_scalars=dataset_scalars,
        density_source=density_source,
        energy_source=energy_source,
        momentum_source=momentum_source,
        diagnostics=diagnostics,
    )
    return ReactionTerms(density_source=density_source, energy_source=energy_source, momentum_source=momentum_source, diagnostics=diagnostics)


def amjuel_recombination(
    atom_name: str,
    ion_name: str,
    *,
    species: dict[str, Any],
    electron_density: np.ndarray,
    dataset_scalars: dict[str, float],
) -> ReactionTerms:
    density_source, energy_source, momentum_source, diagnostics = _initialize_terms(species)
    _accumulate_amjuel_recombination(
        atom_name,
        ion_name,
        species=species,
        electron_density=electron_density,
        dataset_scalars=dataset_scalars,
        density_source=density_source,
        energy_source=energy_source,
        momentum_source=momentum_source,
        diagnostics=diagnostics,
    )
    return ReactionTerms(density_source=density_source, energy_source=energy_source, momentum_source=momentum_source, diagnostics=diagnostics)


def charge_exchange(
    atom1_name: str,
    ion1_name: str,
    atom2_name: str,
    ion2_name: str,
    *,
    config: BoutConfig,
    species: dict[str, Any],
    dataset_scalars: dict[str, float],
) -> ReactionTerms:
    density_source, energy_source, momentum_source, diagnostics = _initialize_terms(species)
    _accumulate_charge_exchange(
        atom1_name,
        ion1_name,
        atom2_name,
        ion2_name,
        config=config,
        species=species,
        dataset_scalars=dataset_scalars,
        density_source=density_source,
        energy_source=energy_source,
        momentum_source=momentum_source,
        diagnostics=diagnostics,
    )
    return ReactionTerms(density_source=density_source, energy_source=energy_source, momentum_source=momentum_source, diagnostics=diagnostics)


def charge_exchange_collision_rates(
    config: BoutConfig,
    *,
    species: dict[str, Any],
    prepared: dict[str, Any],
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
        if not is_charge_exchange_reaction(lhs, rhs):
            continue
        atom1_name = lhs[0]
        ion1_name = lhs[1]
        if atom1_name not in species or ion1_name not in species:
            continue
        atom1 = species[atom1_name]
        ion1 = species[ion1_name]
        atom_temperature = prepared[atom1_name].temperature
        ion_temperature = prepared[ion1_name].temperature
        teff = np.clip((atom_temperature / atom1.atomic_mass + ion_temperature / ion1.atomic_mass) * dataset_scalars["Tnorm"], 0.01, 10000.0)
        sigmav = hydrogen_cx_sigmav(teff, dataset_scalars) * charge_exchange_rate_multiplier(config, atom_name=atom1_name)
        atom_rate = prepared[ion1_name].density * sigmav
        ion_rate = prepared[atom1_name].density * sigmav
        if atom1_name in totals:
            totals[atom1_name] += atom_rate
        else:
            totals[atom1_name] = np.asarray(atom_rate, dtype=np.float64)
        if ion1_name in totals:
            totals[ion1_name] += ion_rate
        else:
            totals[ion1_name] = np.asarray(ion_rate, dtype=np.float64)
    return totals
