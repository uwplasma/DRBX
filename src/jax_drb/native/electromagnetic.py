from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver
from .mesh import StructuredMesh, apply_neumann_x_guards
from .metrics import StructuredMetrics

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


def solve_slab_neumann_apar(
    current_density: np.ndarray,
    *,
    density_fields: Mapping[str, np.ndarray],
    species_metadata: tuple[ChargedSpeciesMetadata, ...],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    beta_em: float,
    density_floor: float = 1.0e-5,
) -> np.ndarray:
    current = np.asarray(current_density, dtype=np.float64)
    if current.shape != (mesh.nx, mesh.local_ny, mesh.nz):
        raise ValueError("current_density must match the full structured field shape.")
    if mesh.xstart != mesh.xend:
        raise NotImplementedError("Native slab Apar currently requires a single interior radial cell.")

    alpha = compute_alpha_em(density_fields, species_metadata, density_floor=density_floor)
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    interior_current = current[mesh.xstart, y_slice, :]
    alpha_core = np.asarray(alpha[mesh.xstart, y_slice, :], dtype=np.float64)
    g33_core = np.asarray(metrics.g33[mesh.xstart, y_slice, :], dtype=np.float64)
    dz_core = np.asarray(metrics.dz[mesh.xstart, y_slice, :], dtype=np.float64)

    alpha_row = alpha_core[:, 0]
    g33_row = g33_core[:, 0]
    dz_row = dz_core[:, 0]
    if not np.allclose(alpha_core, alpha_row[:, None], rtol=1.0e-12, atol=1.0e-12):
        raise NotImplementedError("Native slab Apar currently requires alpha_em uniform along z.")
    if not np.allclose(g33_core, g33_row[:, None], rtol=1.0e-12, atol=1.0e-12):
        raise NotImplementedError("Native slab Apar currently requires g33 uniform along z.")
    if not np.allclose(dz_core, dz_row[:, None], rtol=1.0e-12, atol=1.0e-12):
        raise NotImplementedError("Native slab Apar currently requires dz uniform along z.")

    wave_numbers = (2.0 * np.pi * np.arange(mesh.nz // 2 + 1, dtype=np.float64)[None, :]) / (
        dz_row[:, None] * float(mesh.nz)
    )
    rhs_hat = np.fft.rfft((-float(beta_em)) * interior_current, axis=-1)
    denominator = -(wave_numbers * wave_numbers) * g33_row[:, None] - (
        float(beta_em) * alpha_row[:, None]
    )
    interior_apar = np.fft.irfft(rhs_hat / denominator, n=mesh.nz, axis=-1)

    full = np.zeros_like(current, dtype=np.float64)
    full[mesh.xstart, y_slice, :] = interior_apar
    full = np.array(apply_neumann_x_guards(full, mesh), dtype=np.float64, copy=True)
    for offset in range(mesh.myg):
        full[:, mesh.ystart - 1 - offset, :] = full[:, mesh.yend - offset, :]
        full[:, mesh.yend + 1 + offset, :] = full[:, mesh.ystart + offset, :]
    return full
