from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import StructuredMesh
from .open_field import apply_noflow_flow_guards, apply_noflow_scalar_guards
from .recycling_boundaries import apply_neutral_target_density_guards
from .recycling_setup import OpenFieldSpecies


@dataclass(frozen=True)
class PreparedSpeciesState:
    density: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    velocity: np.ndarray
    momentum: np.ndarray
    momentum_error: np.ndarray


def merge_target_guard_cells(base: np.ndarray, boundary: np.ndarray, *, mesh: StructuredMesh) -> np.ndarray:
    merged = np.asarray(base, dtype=np.float64, copy=True)
    boundary_array = np.asarray(boundary, dtype=np.float64)
    if mesh.myg <= 0:
        return merged
    if mesh.has_lower_y_target:
        merged[:, mesh.ystart - 1, :] = boundary_array[:, mesh.ystart - 1, :]
    if mesh.has_upper_y_target:
        merged[:, mesh.yend + 1, :] = boundary_array[:, mesh.yend + 1, :]
    return merged


def raw_species_velocity(species: OpenFieldSpecies) -> np.ndarray:
    return np.asarray(
        species.momentum / np.maximum(species.atomic_mass * species.density, 1.0e-8),
        dtype=np.float64,
    )


def soft_floor(value: np.ndarray, minimum: float) -> np.ndarray:
    value_array = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    minimum_value = float(minimum)
    return value_array + minimum_value * np.exp(-value_array / minimum_value)


def safe_temperature(pressure: np.ndarray, density: np.ndarray, density_floor: float = 1.0e-8) -> np.ndarray:
    pressure_floor = np.maximum(np.asarray(pressure, dtype=np.float64), 0.0)
    return pressure_floor / soft_floor(np.asarray(density, dtype=np.float64), density_floor)


def axisymmetric_profile(field: np.ndarray) -> np.ndarray:
    field_array = np.asarray(field, dtype=np.float64)
    mean = np.mean(field_array, axis=2, keepdims=True)
    return np.repeat(mean, field_array.shape[2], axis=2)


def prepare_species_state(
    species: OpenFieldSpecies,
    *,
    mesh: StructuredMesh,
) -> PreparedSpeciesState:
    density = species.density.copy()
    pressure = species.pressure.copy()
    temperature = safe_temperature(pressure, density, species.density_floor)
    limited_density = soft_floor(density, species.density_floor)
    momentum = np.asarray(species.momentum, dtype=np.float64, copy=True)
    velocity = momentum / np.maximum(species.atomic_mass * limited_density, 1.0e-8)

    if species.charge == 0.0 and (mesh.has_lower_y_target or mesh.has_upper_y_target):
        density = apply_neutral_target_density_guards(
            density,
            mesh=mesh,
            lower_y=mesh.has_lower_y_target,
            upper_y=mesh.has_upper_y_target,
        )
        pressure = np.array(
            apply_noflow_scalar_guards(
                pressure,
                mesh=mesh,
                lower_y=mesh.has_lower_y_target,
                upper_y=mesh.has_upper_y_target,
            ),
            dtype=np.float64,
            copy=True,
        )
        momentum = np.array(
            apply_noflow_flow_guards(
                momentum,
                mesh=mesh,
                lower_y=mesh.has_lower_y_target,
                upper_y=mesh.has_upper_y_target,
            ),
            dtype=np.float64,
            copy=True,
        )
        temperature = safe_temperature(pressure, density, species.density_floor)
        velocity = momentum / np.maximum(
            species.atomic_mass * soft_floor(density, species.density_floor),
            1.0e-8,
        )

    effective_noflow_lower_y = bool(species.noflow_lower_y and mesh.has_lower_y_target)
    effective_noflow_upper_y = bool(species.noflow_upper_y and mesh.has_upper_y_target)
    if effective_noflow_lower_y or effective_noflow_upper_y:
        density = np.array(
            apply_noflow_scalar_guards(
                density,
                mesh=mesh,
                lower_y=effective_noflow_lower_y,
                upper_y=effective_noflow_upper_y,
            ),
            dtype=np.float64,
            copy=True,
        )
        pressure = np.array(
            apply_noflow_scalar_guards(
                pressure,
                mesh=mesh,
                lower_y=effective_noflow_lower_y,
                upper_y=effective_noflow_upper_y,
            ),
            dtype=np.float64,
            copy=True,
        )
        temperature = np.array(
            apply_noflow_scalar_guards(
                temperature,
                mesh=mesh,
                lower_y=effective_noflow_lower_y,
                upper_y=effective_noflow_upper_y,
            ),
            dtype=np.float64,
            copy=True,
        )
        velocity = np.array(
            apply_noflow_flow_guards(
                velocity,
                mesh=mesh,
                lower_y=effective_noflow_lower_y,
                upper_y=effective_noflow_upper_y,
            ),
            dtype=np.float64,
            copy=True,
        )
    momentum = np.asarray(species.atomic_mass * density * velocity, dtype=np.float64)
    momentum_error = np.asarray(momentum - species.momentum, dtype=np.float64)
    return PreparedSpeciesState(
        density=density,
        pressure=pressure,
        temperature=temperature,
        velocity=velocity,
        momentum=momentum,
        momentum_error=momentum_error,
    )
