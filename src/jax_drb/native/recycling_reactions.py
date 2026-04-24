from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from ..config.boutinp import BoutConfig
from .recycling_atomic import (
    OPENADAS_FILENAMES,
    amjuel_log_inputs,
    amjuel_reaction_rate_and_energy_loss,
    charge_exchange_rate_multiplier,
    eval_amjuel_fit_from_logs,
    eval_openadas_rate,
    hydrogen_cx_sigmav,
    load_amjuel_rate,
    load_openadas_rate,
    openadas_energy_loss,
    openadas_reaction_rate,
)


def _use_jax_backend(*values: object) -> bool:
    for value in values:
        if value is None:
            continue
        module = type(value).__module__
        if module.startswith("jax") or module.startswith("jaxlib"):
            return True
    return False


@dataclass(frozen=True)
class ReactionTerms:
    density_source: dict[str, np.ndarray]
    energy_source: dict[str, np.ndarray]
    momentum_source: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]


class FixedLayoutHydrogenReactionSources(NamedTuple):
    atom_density_source: np.ndarray
    ion_density_source: np.ndarray
    electron_density_source: np.ndarray
    atom_energy_source: np.ndarray
    ion_energy_source: np.ndarray
    electron_energy_source: np.ndarray
    atom_momentum_source: np.ndarray
    ion_momentum_source: np.ndarray
    electron_momentum_source: np.ndarray
    ionisation_rate: np.ndarray
    recombination_rate: np.ndarray
    charge_exchange_rate: np.ndarray
    ionisation_radiation: np.ndarray
    recombination_radiation: np.ndarray


class FixedLayoutDtheReactionSources(NamedTuple):
    neutral_density_source: np.ndarray
    ion_density_source: np.ndarray
    electron_density_source: np.ndarray
    neutral_energy_source: np.ndarray
    ion_energy_source: np.ndarray
    electron_energy_source: np.ndarray
    neutral_momentum_source: np.ndarray
    ion_momentum_source: np.ndarray
    electron_momentum_source: np.ndarray
    ionisation_rate: np.ndarray
    recombination_rate: np.ndarray
    charge_exchange_rate: np.ndarray
    ionisation_radiation: np.ndarray
    recombination_radiation: np.ndarray


def _soft_floor(value: np.ndarray, minimum: float) -> np.ndarray:
    if _use_jax_backend(value):
        value_array = jnp.maximum(jnp.asarray(value, dtype=jnp.float64), 0.0)
        minimum_value = float(minimum)
        return value_array + minimum_value * jnp.exp(-value_array / minimum_value)

    value_array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    minimum_value = float(minimum)
    return value_array + minimum_value * np.exp(-value_array / minimum_value)


def _safe_temperature(pressure: np.ndarray, density: np.ndarray, density_floor: float = 1.0e-8) -> np.ndarray:
    if _use_jax_backend(pressure, density):
        pressure_floor = jnp.maximum(jnp.asarray(pressure, dtype=jnp.float64), 0.0)
        return pressure_floor / _soft_floor(jnp.asarray(density, dtype=jnp.float64), density_floor)

    pressure_floor = np.maximum(np.asarray(pressure, dtype=np.float64), 0.0)
    return pressure_floor / _soft_floor(np.asarray(density, dtype=np.float64), density_floor)


def _zeros_like(value: np.ndarray) -> np.ndarray:
    if _use_jax_backend(value):
        return jnp.zeros_like(jnp.asarray(value, dtype=jnp.float64))
    return np.zeros_like(value, dtype=np.float64)


def _safe_velocity(momentum: np.ndarray, density: np.ndarray, atomic_mass: float) -> np.ndarray:
    if _use_jax_backend(momentum, density):
        return jnp.asarray(momentum, dtype=jnp.float64) / jnp.maximum(
            float(atomic_mass) * jnp.asarray(density, dtype=jnp.float64),
            1.0e-8,
        )
    return np.asarray(momentum, dtype=np.float64) / np.maximum(
        float(atomic_mass) * np.asarray(density, dtype=np.float64),
        1.0e-8,
    )


def _clip(value: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if _use_jax_backend(value):
        return jnp.clip(jnp.asarray(value, dtype=jnp.float64), float(lower), float(upper))
    return np.clip(np.asarray(value, dtype=np.float64), float(lower), float(upper))


def _square(value: np.ndarray) -> np.ndarray:
    if _use_jax_backend(value):
        return jnp.square(value)
    return np.square(value)


def fixed_layout_hydrogen_reaction_sources(
    *,
    atom_density: np.ndarray,
    atom_pressure: np.ndarray,
    atom_momentum: np.ndarray,
    ion_density: np.ndarray,
    ion_pressure: np.ndarray,
    ion_momentum: np.ndarray,
    electron_density: np.ndarray,
    electron_pressure: np.ndarray,
    dataset_scalars: dict[str, float],
    atom_name: str = "d",
    atom_mass: float = 2.0,
    ion_mass: float = 2.0,
    atom_density_floor: float = 1.0e-8,
    ion_density_floor: float = 1.0e-8,
    electron_density_floor: float = 1.0e-8,
    cx_multiplier: float = 1.0,
) -> FixedLayoutHydrogenReactionSources:
    """Return array-only H-isotope ionisation, recombination, and CX sources.

    This is the first fixed-layout bridge toward a JAX-native recycling
    residual. It intentionally covers the common same-isotope hydrogenic
    reaction block and mirrors the existing dictionary-oriented implementation.
    """

    electron_temperature = _safe_temperature(electron_pressure, electron_density, electron_density_floor)
    atom_temperature = _safe_temperature(atom_pressure, atom_density, atom_density_floor)
    ion_temperature = _safe_temperature(ion_pressure, ion_density, ion_density_floor)
    atom_velocity = _safe_velocity(atom_momentum, atom_density, atom_mass)
    ion_velocity = _safe_velocity(ion_momentum, ion_density, ion_mass)

    iz_sigma_v, iz_sigma_v_E, iz_electron_heating = load_amjuel_rate(atom_name, "iz")
    ionisation_rate, ionisation_radiation = amjuel_reaction_rate_and_energy_loss(
        atom_density,
        electron_density,
        electron_temperature,
        iz_sigma_v,
        iz_sigma_v_E,
        iz_electron_heating,
        dataset_scalars,
    )

    rec_sigma_v, rec_sigma_v_E, rec_electron_heating = load_amjuel_rate(atom_name, "rec")
    recombination_rate, recombination_radiation = amjuel_reaction_rate_and_energy_loss(
        ion_density,
        electron_density,
        electron_temperature,
        rec_sigma_v,
        rec_sigma_v_E,
        rec_electron_heating,
        dataset_scalars,
    )

    teff = _clip(
        (atom_temperature / atom_mass + ion_temperature / ion_mass) * dataset_scalars["Tnorm"],
        0.01,
        10000.0,
    )
    charge_exchange_rate = (
        atom_density
        * ion_density
        * hydrogen_cx_sigmav(teff, dataset_scalars)
        * float(cx_multiplier)
    )

    ionisation_momentum = ionisation_rate * atom_mass * atom_velocity
    recombination_momentum = recombination_rate * ion_mass * ion_velocity
    cx_atom_momentum = charge_exchange_rate * atom_mass * atom_velocity
    cx_ion_momentum = charge_exchange_rate * ion_mass * ion_velocity

    ionisation_atom_energy = 1.5 * ionisation_rate * atom_temperature
    recombination_ion_energy = 1.5 * recombination_rate * ion_temperature
    cx_atom_energy = 1.5 * charge_exchange_rate * atom_temperature
    cx_ion_energy = 1.5 * charge_exchange_rate * ion_temperature
    velocity_delta = ion_velocity - atom_velocity
    cx_ion_kinetic = 0.5 * atom_mass * charge_exchange_rate * _square(velocity_delta)
    cx_atom_kinetic = 0.5 * ion_mass * charge_exchange_rate * _square(-velocity_delta)

    zero = _zeros_like(atom_density)
    atom_density_source = -ionisation_rate + recombination_rate
    ion_density_source = ionisation_rate - recombination_rate
    atom_momentum_source = -ionisation_momentum + recombination_momentum - cx_atom_momentum + cx_ion_momentum
    ion_momentum_source = ionisation_momentum - recombination_momentum + cx_atom_momentum - cx_ion_momentum
    atom_energy_source = -ionisation_atom_energy + recombination_ion_energy - cx_atom_energy + cx_ion_energy + cx_atom_kinetic
    ion_energy_source = ionisation_atom_energy - recombination_ion_energy + cx_atom_energy - cx_ion_energy + cx_ion_kinetic
    electron_energy_source = -ionisation_radiation - recombination_radiation

    return FixedLayoutHydrogenReactionSources(
        atom_density_source=atom_density_source,
        ion_density_source=ion_density_source,
        electron_density_source=zero,
        atom_energy_source=atom_energy_source,
        ion_energy_source=ion_energy_source,
        electron_energy_source=electron_energy_source,
        atom_momentum_source=atom_momentum_source,
        ion_momentum_source=ion_momentum_source,
        electron_momentum_source=zero,
        ionisation_rate=ionisation_rate,
        recombination_rate=recombination_rate,
        charge_exchange_rate=charge_exchange_rate,
        ionisation_radiation=ionisation_radiation,
        recombination_radiation=recombination_radiation,
    )


def fixed_layout_dthe_reaction_sources(
    *,
    neutral_density: np.ndarray,
    neutral_pressure: np.ndarray,
    neutral_momentum: np.ndarray,
    ion_density: np.ndarray,
    ion_pressure: np.ndarray,
    ion_momentum: np.ndarray,
    electron_density: np.ndarray,
    electron_pressure: np.ndarray,
    dataset_scalars: dict[str, float],
    atom_names: tuple[str, str, str] = ("d", "t", "he"),
    atom_masses: tuple[float, float, float] = (2.0, 3.0, 4.0),
    ion_masses: tuple[float, float, float] = (2.0, 3.0, 4.0),
    cx_multipliers: tuple[float, float, float] = (1.0, 1.0, 1.0),
    neutral_density_floor: float = 1.0e-8,
    neutral_density_floors: tuple[float, float, float] | None = None,
    ion_density_floor: float = 1.0e-8,
    ion_density_floors: tuple[float, float, float] | None = None,
    electron_density_floor: float = 1.0e-8,
) -> FixedLayoutDtheReactionSources:
    """Return fixed-layout D/T/He ionisation, recombination, and D/T CX sources.

    The returned arrays use species axis order `(d, t, he)` by default. Helium
    participates in ionisation and recombination on the current Hermès D/T/He
    lane; hydrogenic charge exchange is included for D-D, T-T, D-T, and T-D.
    """

    use_jax = _use_jax_backend(neutral_density, neutral_pressure, neutral_momentum, ion_density, ion_pressure, ion_momentum)
    neutral_density = jnp.asarray(neutral_density, dtype=jnp.float64) if use_jax else np.asarray(neutral_density, dtype=np.float64)
    neutral_pressure = jnp.asarray(neutral_pressure, dtype=jnp.float64) if use_jax else np.asarray(neutral_pressure, dtype=np.float64)
    neutral_momentum = jnp.asarray(neutral_momentum, dtype=jnp.float64) if use_jax else np.asarray(neutral_momentum, dtype=np.float64)
    ion_density = jnp.asarray(ion_density, dtype=jnp.float64) if use_jax else np.asarray(ion_density, dtype=np.float64)
    ion_pressure = jnp.asarray(ion_pressure, dtype=jnp.float64) if use_jax else np.asarray(ion_pressure, dtype=np.float64)
    ion_momentum = jnp.asarray(ion_momentum, dtype=jnp.float64) if use_jax else np.asarray(ion_momentum, dtype=np.float64)

    zero_neutral = _zeros_like(neutral_density)
    zero_field = _zeros_like(electron_density)
    neutral_density_source = zero_neutral
    ion_density_source = _zeros_like(ion_density)
    neutral_energy_source = _zeros_like(neutral_density)
    ion_energy_source = _zeros_like(ion_density)
    neutral_momentum_source = _zeros_like(neutral_density)
    ion_momentum_source = _zeros_like(ion_density)
    electron_energy_source = zero_field
    ionisation_rate = _zeros_like(neutral_density)
    recombination_rate = _zeros_like(ion_density)
    ionisation_radiation = _zeros_like(neutral_density)
    recombination_radiation = _zeros_like(ion_density)
    charge_exchange_rate = _zeros_like(_expand_cx_shape(neutral_density, use_jax=use_jax))

    electron_temperature = _safe_temperature(electron_pressure, electron_density, electron_density_floor)
    amjuel_log_temperature, amjuel_log_density = amjuel_log_inputs(
        electron_temperature * dataset_scalars["Tnorm"],
        electron_density * dataset_scalars["Nnorm"],
    )
    amjuel_fit_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float]] = {}
    neutral_floor_values = neutral_density_floors or tuple(float(neutral_density_floor) for _ in atom_names)
    ion_floor_values = ion_density_floors or tuple(float(ion_density_floor) for _ in atom_names)
    neutral_temperature = tuple(
        _safe_temperature(neutral_pressure[index], neutral_density[index], neutral_floor_values[index])
        for index in range(len(atom_names))
    )
    ion_temperature = tuple(
        _safe_temperature(ion_pressure[index], ion_density[index], ion_floor_values[index])
        for index in range(len(atom_names))
    )
    neutral_velocity = tuple(
        _safe_velocity(neutral_momentum[index], neutral_density[index], atom_masses[index])
        for index in range(len(atom_names))
    )
    ion_velocity = tuple(
        _safe_velocity(ion_momentum[index], ion_density[index], ion_masses[index])
        for index in range(len(atom_names))
    )

    for index, atom_name in enumerate(atom_names):
        iz_rate, iz_radiation = _fixed_layout_paired_rate_and_radiation(
            atom_name,
            "iz",
            neutral_density[index],
            electron_density,
            electron_temperature,
            dataset_scalars,
            amjuel_log_temperature=amjuel_log_temperature,
            amjuel_log_density=amjuel_log_density,
            amjuel_fit_cache=amjuel_fit_cache,
        )
        rec_rate, rec_radiation = _fixed_layout_paired_rate_and_radiation(
            atom_name,
            "rec",
            ion_density[index],
            electron_density,
            electron_temperature,
            dataset_scalars,
            amjuel_log_temperature=amjuel_log_temperature,
            amjuel_log_density=amjuel_log_density,
            amjuel_fit_cache=amjuel_fit_cache,
        )
        iz_momentum = iz_rate * atom_masses[index] * neutral_velocity[index]
        rec_momentum = rec_rate * ion_masses[index] * ion_velocity[index]
        iz_energy = 1.5 * iz_rate * neutral_temperature[index]
        rec_energy = 1.5 * rec_rate * ion_temperature[index]

        neutral_density_source = _axis_add(neutral_density_source, index, -iz_rate + rec_rate, use_jax=use_jax)
        ion_density_source = _axis_add(ion_density_source, index, iz_rate - rec_rate, use_jax=use_jax)
        neutral_momentum_source = _axis_add(neutral_momentum_source, index, -iz_momentum + rec_momentum, use_jax=use_jax)
        ion_momentum_source = _axis_add(ion_momentum_source, index, iz_momentum - rec_momentum, use_jax=use_jax)
        neutral_energy_source = _axis_add(neutral_energy_source, index, -iz_energy + rec_energy, use_jax=use_jax)
        ion_energy_source = _axis_add(ion_energy_source, index, iz_energy - rec_energy, use_jax=use_jax)
        electron_energy_source = electron_energy_source - iz_radiation - rec_radiation
        ionisation_rate = _axis_add(ionisation_rate, index, iz_rate, use_jax=use_jax)
        recombination_rate = _axis_add(recombination_rate, index, rec_rate, use_jax=use_jax)
        ionisation_radiation = _axis_add(ionisation_radiation, index, iz_radiation, use_jax=use_jax)
        recombination_radiation = _axis_add(recombination_radiation, index, rec_radiation, use_jax=use_jax)

    for atom_index, ion_index in ((0, 0), (1, 1), (0, 1), (1, 0)):
        rate = _fixed_layout_cx_rate(
            neutral_density[atom_index],
            ion_density[ion_index],
            neutral_temperature[atom_index],
            ion_temperature[ion_index],
            atom_mass=atom_masses[atom_index],
            ion_mass=ion_masses[ion_index],
            dataset_scalars=dataset_scalars,
            multiplier=float(cx_multipliers[atom_index]),
        )
        charge_exchange_rate = _axis2_add(charge_exchange_rate, atom_index, ion_index, rate, use_jax=use_jax)
        atom_velocity = neutral_velocity[atom_index]
        ion1_velocity = ion_velocity[ion_index]
        atom2_velocity = neutral_velocity[ion_index]
        ion2_velocity = ion_velocity[atom_index]
        atom_momentum = rate * atom_masses[atom_index] * atom_velocity
        ion1_momentum = rate * ion_masses[ion_index] * ion1_velocity
        atom_energy = 1.5 * rate * neutral_temperature[atom_index]
        ion1_energy = 1.5 * rate * ion_temperature[ion_index]

        if atom_index != ion_index:
            neutral_density_source = _axis_add(neutral_density_source, atom_index, -rate, use_jax=use_jax)
            ion_density_source = _axis_add(ion_density_source, atom_index, rate, use_jax=use_jax)
            ion_density_source = _axis_add(ion_density_source, ion_index, -rate, use_jax=use_jax)
            neutral_density_source = _axis_add(neutral_density_source, ion_index, rate, use_jax=use_jax)
        neutral_momentum_source = _axis_add(neutral_momentum_source, atom_index, -atom_momentum, use_jax=use_jax)
        ion_momentum_source = _axis_add(ion_momentum_source, atom_index, atom_momentum, use_jax=use_jax)
        ion_momentum_source = _axis_add(ion_momentum_source, ion_index, -ion1_momentum, use_jax=use_jax)
        neutral_momentum_source = _axis_add(neutral_momentum_source, ion_index, ion1_momentum, use_jax=use_jax)
        neutral_energy_source = _axis_add(neutral_energy_source, atom_index, -atom_energy, use_jax=use_jax)
        ion_energy_source = _axis_add(ion_energy_source, atom_index, atom_energy, use_jax=use_jax)
        ion_energy_source = _axis_add(
            ion_energy_source,
            atom_index,
            0.5 * atom_masses[atom_index] * rate * _square(ion2_velocity - atom_velocity),
            use_jax=use_jax,
        )
        ion_energy_source = _axis_add(ion_energy_source, ion_index, -ion1_energy, use_jax=use_jax)
        neutral_energy_source = _axis_add(neutral_energy_source, ion_index, ion1_energy, use_jax=use_jax)
        neutral_energy_source = _axis_add(
            neutral_energy_source,
            ion_index,
            0.5 * ion_masses[ion_index] * rate * _square(atom2_velocity - ion1_velocity),
            use_jax=use_jax,
        )

    return FixedLayoutDtheReactionSources(
        neutral_density_source=neutral_density_source,
        ion_density_source=ion_density_source,
        electron_density_source=zero_field,
        neutral_energy_source=neutral_energy_source,
        ion_energy_source=ion_energy_source,
        electron_energy_source=electron_energy_source,
        neutral_momentum_source=neutral_momentum_source,
        ion_momentum_source=ion_momentum_source,
        electron_momentum_source=zero_field,
        ionisation_rate=ionisation_rate,
        recombination_rate=recombination_rate,
        charge_exchange_rate=charge_exchange_rate,
        ionisation_radiation=ionisation_radiation,
        recombination_radiation=recombination_radiation,
    )


def _fixed_layout_paired_rate_and_radiation(
    atom_name: str,
    reaction_type: str,
    heavy_density: np.ndarray,
    electron_density: np.ndarray,
    electron_temperature: np.ndarray,
    dataset_scalars: dict[str, float],
    *,
    amjuel_log_temperature: np.ndarray,
    amjuel_log_density: np.ndarray,
    amjuel_fit_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray]:
    if (atom_name, reaction_type) in OPENADAS_FILENAMES:
        return _paired_rate_and_radiation(
            atom_name,
            reaction_type,
            heavy_density,
            electron_density,
            electron_temperature,
            dataset_scalars,
        )

    # D and T use the same packaged hydrogenic AMJUEL tables.
    table_atom_name = "d" if atom_name in {"d", "t"} else atom_name
    cache_key = (table_atom_name, reaction_type)
    if cache_key not in amjuel_fit_cache:
        sigma_v_coeffs, sigma_v_E_coeffs, electron_heating = load_amjuel_rate(table_atom_name, reaction_type)
        sigma_v = eval_amjuel_fit_from_logs(amjuel_log_temperature, amjuel_log_density, sigma_v_coeffs)
        sigma_v_E = eval_amjuel_fit_from_logs(amjuel_log_temperature, amjuel_log_density, sigma_v_E_coeffs)
        amjuel_fit_cache[cache_key] = (sigma_v, sigma_v_E, electron_heating)
    sigma_v, sigma_v_E, electron_heating = amjuel_fit_cache[cache_key]
    return _amjuel_rate_and_radiation_from_fit_values(
        heavy_density,
        electron_density,
        sigma_v,
        sigma_v_E,
        electron_heating,
        dataset_scalars,
    )


def _amjuel_rate_and_radiation_from_fit_values(
    heavy_density: np.ndarray,
    electron_density: np.ndarray,
    sigma_v: np.ndarray,
    sigma_v_E: np.ndarray,
    electron_heating: float,
    dataset_scalars: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    if _use_jax_backend(heavy_density, electron_density, sigma_v, sigma_v_E):
        heavy = jnp.asarray(heavy_density, dtype=jnp.float64)
        electrons = jnp.asarray(electron_density, dtype=jnp.float64)
        rate = heavy * electrons * jnp.asarray(sigma_v, dtype=jnp.float64) * (
            dataset_scalars["Nnorm"] / dataset_scalars["Omega_ci"]
        )
        energy_loss = (
            heavy
            * electrons
            * jnp.asarray(sigma_v_E, dtype=jnp.float64)
            * dataset_scalars["Nnorm"]
            / (dataset_scalars["Tnorm"] * dataset_scalars["Omega_ci"])
        )
        return rate, energy_loss - (electron_heating / dataset_scalars["Tnorm"]) * rate

    heavy = np.asarray(heavy_density, dtype=np.float64)
    electrons = np.asarray(electron_density, dtype=np.float64)
    rate = (
        heavy
        * electrons
        * np.asarray(sigma_v, dtype=np.float64)
        * (dataset_scalars["Nnorm"] / dataset_scalars["Omega_ci"])
    )
    energy_loss = (
        heavy
        * electrons
        * np.asarray(sigma_v_E, dtype=np.float64)
        * dataset_scalars["Nnorm"]
        / (dataset_scalars["Tnorm"] * dataset_scalars["Omega_ci"])
    )
    return rate, energy_loss - (electron_heating / dataset_scalars["Tnorm"]) * rate


def _paired_rate_and_radiation(
    atom_name: str,
    reaction_type: str,
    heavy_density: np.ndarray,
    electron_density: np.ndarray,
    electron_temperature: np.ndarray,
    dataset_scalars: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    if (atom_name, reaction_type) in OPENADAS_FILENAMES:
        rate = openadas_reaction_rate(
            heavy_density,
            electron_density,
            electron_temperature,
            atom_name,
            reaction_type,
            dataset_scalars,
        )
        radiation = openadas_energy_loss(
            heavy_density,
            electron_density,
            electron_temperature,
            atom_name,
            reaction_type,
            reaction_rate=rate,
            dataset_scalars=dataset_scalars,
        )
        return rate, radiation
    sigma_v, sigma_v_E, electron_heating = load_amjuel_rate(atom_name, reaction_type)
    return amjuel_reaction_rate_and_energy_loss(
        heavy_density,
        electron_density,
        electron_temperature,
        sigma_v,
        sigma_v_E,
        electron_heating,
        dataset_scalars,
    )


def _fixed_layout_cx_rate(
    atom_density: np.ndarray,
    ion_density: np.ndarray,
    atom_temperature: np.ndarray,
    ion_temperature: np.ndarray,
    *,
    atom_mass: float,
    ion_mass: float,
    dataset_scalars: dict[str, float],
    multiplier: float = 1.0,
) -> np.ndarray:
    teff = _clip(
        (atom_temperature / atom_mass + ion_temperature / ion_mass) * dataset_scalars["Tnorm"],
        0.01,
        10000.0,
    )
    return atom_density * ion_density * hydrogen_cx_sigmav(teff, dataset_scalars) * float(multiplier)


def _expand_cx_shape(neutral_density: np.ndarray, *, use_jax: bool) -> np.ndarray:
    shape = (neutral_density.shape[0], neutral_density.shape[0], *neutral_density.shape[1:])
    return jnp.zeros(shape, dtype=jnp.float64) if use_jax else np.zeros(shape, dtype=np.float64)


def _axis_add(array: np.ndarray, index: int, value: np.ndarray, *, use_jax: bool) -> np.ndarray:
    if use_jax:
        return array.at[index].add(value)
    array[index] += value
    return array


def _axis2_add(array: np.ndarray, index0: int, index1: int, value: np.ndarray, *, use_jax: bool) -> np.ndarray:
    if use_jax:
        return array.at[index0, index1].add(value)
    array[index0, index1] += value
    return array


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
    density_source = {name: _zeros_like(sp.density) for name, sp in species.items()}
    energy_source = {name: _zeros_like(sp.density) for name, sp in species.items()}
    momentum_source = {name: _zeros_like(sp.density) for name, sp in species.items()}
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
    atom_velocity = _safe_velocity(atom.momentum, atom.density, atom.atomic_mass)
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
    ion_velocity = _safe_velocity(ion.momentum, ion.density, ion.atomic_mass)
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
    teff = _clip(
        (atom_temperature / atom1.atomic_mass + ion_temperature / ion1.atomic_mass) * dataset_scalars["Tnorm"],
        0.01,
        10000.0,
    )
    sigmav = hydrogen_cx_sigmav(teff, dataset_scalars) * charge_exchange_rate_multiplier(config, atom_name=atom1_name)
    rate = atom1.density * ion1.density * sigmav
    atom_velocity = _safe_velocity(atom1.momentum, atom1.density, atom1.atomic_mass)
    ion_velocity = _safe_velocity(ion1.momentum, ion1.density, ion1.atomic_mass)
    atom2_velocity = _safe_velocity(atom2.momentum, atom2.density, atom2.atomic_mass)
    ion2_velocity = _safe_velocity(ion2.momentum, ion2.density, ion2.atomic_mass)
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
    energy_source[ion2_name] += 0.5 * atom1.atomic_mass * rate * _square(ion2_velocity - atom_velocity)
    energy_source[atom2_name] += 0.5 * ion1.atomic_mass * rate * _square(atom2_velocity - ion_velocity)
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
    include_diagnostics: bool = True,
) -> ReactionTerms:
    if not include_diagnostics and _can_use_fixed_layout_dthe_reaction_sources(config, species):
        return _fixed_layout_dthe_reaction_terms(
            config,
            species=species,
            electron_density=electron_density,
            dataset_scalars=dataset_scalars,
        )

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
    if not include_diagnostics:
        diagnostics = {}
    return ReactionTerms(density_source=density_source, energy_source=energy_source, momentum_source=momentum_source, diagnostics=diagnostics)


def _can_use_fixed_layout_dthe_reaction_sources(config: BoutConfig, species: dict[str, Any]) -> bool:
    if not config.has_section("reactions") or not config.has_option("reactions", "type"):
        return False
    required_species = {"d", "d+", "t", "t+", "he", "he+", "e"}
    if not required_species.issubset(species):
        return False
    reactions = config.parsed("reactions", "type")
    reaction_tuple = tuple(
        str(reaction).strip()
        for reaction in (reactions if isinstance(reactions, tuple) else (reactions,))
    )
    expected_reactions = {
        "d + e -> d+ + 2e",
        "d+ + e -> d",
        "d + d+ -> d+ + d",
        "t + e -> t+ + 2e",
        "t+ + e -> t",
        "t + t+ -> t+ + t",
        "d + t+ -> d+ + t",
        "t + d+ -> t+ + d",
        "he + e -> he+ + 2e",
        "he+ + e -> he",
    }
    return set(reaction_tuple) == expected_reactions


def _fixed_layout_dthe_reaction_terms(
    config: BoutConfig,
    *,
    species: dict[str, Any],
    electron_density: np.ndarray,
    dataset_scalars: dict[str, float],
) -> ReactionTerms:
    atom_names = ("d", "t", "he")
    ion_names = ("d+", "t+", "he+")
    fixed = fixed_layout_dthe_reaction_sources(
        neutral_density=_stack_species_axis(species, atom_names, "density"),
        neutral_pressure=_stack_species_axis(species, atom_names, "pressure"),
        neutral_momentum=_stack_species_axis(species, atom_names, "momentum"),
        ion_density=_stack_species_axis(species, ion_names, "density"),
        ion_pressure=_stack_species_axis(species, ion_names, "pressure"),
        ion_momentum=_stack_species_axis(species, ion_names, "momentum"),
        electron_density=electron_density,
        electron_pressure=species["e"].pressure,
        dataset_scalars=dataset_scalars,
        atom_names=atom_names,
        atom_masses=tuple(float(species[name].atomic_mass) for name in atom_names),
        ion_masses=tuple(float(species[name].atomic_mass) for name in ion_names),
        cx_multipliers=tuple(charge_exchange_rate_multiplier(config, atom_name=name) for name in atom_names),
        neutral_density_floors=tuple(float(species[name].density_floor) for name in atom_names),
        ion_density_floors=tuple(float(species[name].density_floor) for name in ion_names),
        electron_density_floor=float(species["e"].density_floor),
    )
    density_source = {name: _zeros_like(sp.density) for name, sp in species.items()}
    energy_source = {name: _zeros_like(sp.density) for name, sp in species.items()}
    momentum_source = {name: _zeros_like(sp.density) for name, sp in species.items()}
    for index, name in enumerate(atom_names):
        density_source[name] = fixed.neutral_density_source[index]
        energy_source[name] = fixed.neutral_energy_source[index]
        momentum_source[name] = fixed.neutral_momentum_source[index]
    for index, name in enumerate(ion_names):
        density_source[name] = fixed.ion_density_source[index]
        energy_source[name] = fixed.ion_energy_source[index]
        momentum_source[name] = fixed.ion_momentum_source[index]
    density_source["e"] = fixed.electron_density_source
    energy_source["e"] = fixed.electron_energy_source
    momentum_source["e"] = fixed.electron_momentum_source
    return ReactionTerms(
        density_source=density_source,
        energy_source=energy_source,
        momentum_source=momentum_source,
        diagnostics={},
    )


def _stack_species_axis(species: dict[str, Any], names: tuple[str, ...], field_name: str) -> np.ndarray:
    arrays = [getattr(species[name], field_name) for name in names]
    if _use_jax_backend(*arrays):
        return jnp.stack([jnp.asarray(array, dtype=jnp.float64) for array in arrays], axis=0)
    return np.stack([np.asarray(array, dtype=np.float64) for array in arrays], axis=0)


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
    electron_temperature_physical = np.asarray(electron_temperature, dtype=np.float64) * dataset_scalars["Tnorm"]
    electron_density_physical = np.asarray(electron_density, dtype=np.float64) * dataset_scalars["Nnorm"]
    amjuel_log_temperature, amjuel_log_density = amjuel_log_inputs(electron_temperature_physical, electron_density_physical)
    amjuel_sigma_cache: dict[str, np.ndarray] = {}
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
                electron_temperature_physical,
                electron_density_physical,
                rate_coeff,
                log_temperature=log_temperature,
                log_density=log_density,
            )
        else:
            # D and T use the same packaged hydrogenic AMJUEL ionisation table.
            table_atom_name = "d" if atom_name in {"d", "t"} else atom_name
            if table_atom_name not in amjuel_sigma_cache:
                sigma_v_coeffs, _, _ = load_amjuel_rate(table_atom_name, "iz")
                amjuel_sigma_cache[table_atom_name] = eval_amjuel_fit_from_logs(
                    amjuel_log_temperature,
                    amjuel_log_density,
                    sigma_v_coeffs,
                )
            sigma_v = amjuel_sigma_cache[table_atom_name]
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
