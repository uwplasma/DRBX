from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver

VACUUM_PERMEABILITY = 4.0e-7 * np.pi


@dataclass(frozen=True)
class ChargedSpeciesMetadata:
    section: str
    charge: float
    atomic_mass: float

    @property
    def current_factor(self) -> float:
        return self.charge / self.atomic_mass

    @property
    def alpha_factor(self) -> float:
        return (self.charge * self.charge) / self.atomic_mass


def compute_beta_em(*, Nnorm: float, Tnorm: float, Bnorm: float) -> float:
    return float(VACUUM_PERMEABILITY * 1.602176634e-19 * Tnorm * Nnorm / (Bnorm * Bnorm))


def extract_charged_species_metadata(config: BoutConfig) -> tuple[ChargedSpeciesMetadata, ...]:
    resolver = NumericResolver(config)
    species: list[ChargedSpeciesMetadata] = []
    for section in config.section_names():
        if not config.has_option(section, "charge") or not config.has_option(section, "AA"):
            continue
        charge = float(resolver.resolve(section, "charge"))
        if abs(charge) < 1.0e-12:
            continue
        species.append(
            ChargedSpeciesMetadata(
                section=section,
                charge=charge,
                atomic_mass=float(resolver.resolve(section, "AA")),
            )
        )
    return tuple(species)


def compute_parallel_current_density(
    momentum_fields: Mapping[str, np.ndarray],
    species_metadata: tuple[ChargedSpeciesMetadata, ...],
) -> np.ndarray:
    first = next(iter(momentum_fields.values()))
    current = np.zeros_like(np.asarray(first, dtype=np.float64), dtype=np.float64)
    for species in species_metadata:
        name = f"NV{species.section}"
        if name not in momentum_fields:
            continue
        current = current + species.current_factor * np.asarray(momentum_fields[name], dtype=np.float64)
    return current


def compute_alpha_em(
    density_fields: Mapping[str, np.ndarray],
    species_metadata: tuple[ChargedSpeciesMetadata, ...],
    *,
    density_floor: float = 1.0e-5,
) -> np.ndarray:
    first = next(iter(density_fields.values()))
    alpha = np.zeros_like(np.asarray(first, dtype=np.float64), dtype=np.float64)
    for species in species_metadata:
        name = f"N{species.section}"
        if name not in density_fields:
            continue
        density = np.asarray(density_fields[name], dtype=np.float64)
        alpha = alpha + species.alpha_factor * np.maximum(density, density_floor)
    return alpha


def apply_canonical_momentum_correction(
    *,
    density: np.ndarray,
    momentum: np.ndarray,
    velocity: np.ndarray,
    apar: np.ndarray,
    charge: float,
    atomic_mass: float,
    density_floor: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray]:
    density_array = np.asarray(density, dtype=np.float64)
    apar_array = np.asarray(apar, dtype=np.float64)
    corrected_momentum = np.asarray(momentum, dtype=np.float64) - charge * density_array * apar_array
    corrected_velocity = np.asarray(velocity, dtype=np.float64) - (
        (charge / atomic_mass) * density_array * apar_array / np.maximum(density_array, density_floor)
    )
    return corrected_momentum, corrected_velocity


def compute_apar_flutter(apar: np.ndarray, *, axis: int = 1) -> np.ndarray:
    apar_array = np.asarray(apar, dtype=np.float64)
    return apar_array - np.mean(apar_array, axis=axis, keepdims=True)
