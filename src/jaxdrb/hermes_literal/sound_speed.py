"""Literal translation of Hermes `sound_speed.cxx`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jax.numpy as jnp

from .state import LiteralSpeciesState


@dataclass(frozen=True)
class SoundSpeedResult:
    sound_speed: jnp.ndarray
    fastest_wave: jnp.ndarray


def _soft_floor(field: jnp.ndarray, floor: float) -> jnp.ndarray:
    arr = jnp.asarray(field, dtype=jnp.float64)
    floor_val = float(floor)
    if floor_val <= 0.0:
        return arr
    c = jnp.asarray(floor_val, dtype=jnp.float64)
    return 0.5 * (arr + jnp.sqrt(arr * arr + c * c))


def compute_fastest_wave(
    species: Iterable[LiteralSpeciesState],
    *,
    electron_dynamics: bool = True,
    alfven_wave: bool = False,
    beta_norm: float = 0.0,
    Bxy: jnp.ndarray | None = None,
    temperature_floor: float = 0.0,
    fastest_wave_factor: float = 1.0,
) -> SoundSpeedResult:
    """Mirror `SoundSpeed::transform_impl`.

    Source:
    - `/Users/rogerio/local/hermes-3/src/sound_speed.cxx`
    - `/Users/rogerio/local/hermes-3/include/sound_speed.hxx`
    """

    species_list = tuple(species)
    template = None
    for sp in species_list:
        if sp.pressure is not None:
            template = sp.pressure
            break
        if sp.temperature is not None:
            template = sp.temperature
            break
        if sp.density is not None:
            template = sp.density
            break
    if template is None:
        raise ValueError("At least one species field is required to compute sound speed.")

    total_pressure = jnp.zeros_like(jnp.asarray(template, dtype=jnp.float64))
    total_density = jnp.zeros_like(total_pressure)
    fastest_wave = jnp.zeros_like(total_pressure)

    for index, sp in enumerate(species_list):
        if sp.pressure is not None:
            total_pressure = total_pressure + jnp.asarray(sp.pressure, dtype=jnp.float64)

        is_electron = (sp.charge is not None) and float(sp.charge) < 0.0 and index == 0
        include_species_speed = electron_dynamics or (not is_electron)

        if sp.AA is not None and sp.density is not None and include_species_speed:
            total_density = total_density + jnp.asarray(sp.density, dtype=jnp.float64) * float(
                sp.AA
            )

        if sp.AA is not None and sp.temperature is not None and include_species_speed:
            sound = jnp.sqrt(_soft_floor(sp.temperature, temperature_floor) / float(sp.AA))
            fastest_wave = jnp.maximum(fastest_wave, sound)

    total_density = _soft_floor(total_density, 1e-10)
    sound_speed = jnp.sqrt(total_pressure / total_density)
    fastest_wave = jnp.maximum(fastest_wave, sound_speed)

    if alfven_wave:
        if Bxy is None:
            raise ValueError("Bxy is required when alfven_wave=True.")
        alfven = float(beta_norm) * jnp.asarray(Bxy, dtype=jnp.float64) / jnp.sqrt(total_density)
        fastest_wave = jnp.maximum(fastest_wave, alfven)

    return SoundSpeedResult(
        sound_speed=sound_speed,
        fastest_wave=fastest_wave * float(fastest_wave_factor),
    )
