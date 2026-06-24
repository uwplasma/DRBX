from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..geometry import FciGeometry3D


@dataclass(frozen=True)
class FciTargetMasks:
    """Open-field endpoint masks derived from traced FCI plane intersections."""

    forward: jnp.ndarray
    backward: jnp.ndarray
    endpoint_count: jnp.ndarray
    active: jnp.ndarray


@dataclass(frozen=True)
class FciSheathRecyclingResult:
    """JAX-native sheath-loss and target-recycling diagnostic fields."""

    masks: FciTargetMasks
    sound_speed: jnp.ndarray
    forward_ion_particle_flux: jnp.ndarray
    backward_ion_particle_flux: jnp.ndarray
    ion_particle_loss: jnp.ndarray
    electron_particle_loss: jnp.ndarray
    electron_heat_loss: jnp.ndarray
    ion_heat_loss: jnp.ndarray
    target_heat_load: jnp.ndarray
    recycled_particle_source: jnp.ndarray
    recycled_neutral_energy_source: jnp.ndarray
    plasma_particle_sink: jnp.ndarray
    plasma_energy_sink: jnp.ndarray
    current_residual: jnp.ndarray
    total_ion_particle_loss: jnp.ndarray
    total_electron_particle_loss: jnp.ndarray
    total_recycled_particle_source: jnp.ndarray
    total_target_heat_load: jnp.ndarray
    total_recycled_neutral_energy: jnp.ndarray
    particle_recycling_residual: jnp.ndarray
    neutral_energy_recycling_residual: jnp.ndarray
    current_balance_residual: jnp.ndarray


def build_fci_target_masks(geometry: FciGeometry3D) -> FciTargetMasks:
    """Build endpoint masks from forward/backward field-line map exits."""

    forward = jnp.asarray(geometry.forward_boundary, dtype=jnp.float64)
    backward = jnp.asarray(geometry.backward_boundary, dtype=jnp.float64)
    endpoint_count = forward + backward
    active = endpoint_count > 0.0
    return FciTargetMasks(
        forward=forward,
        backward=backward,
        endpoint_count=endpoint_count,
        active=active,
    )


def fci_bohm_sound_speed(
    electron_temperature: jnp.ndarray,
    ion_temperature: jnp.ndarray,
    *,
    ion_mass: float = 1.0,
    temperature_floor: float = 1.0e-12,
) -> jnp.ndarray:
    """Return the normalized sheath-entry sound speed."""

    te = jnp.maximum(jnp.asarray(electron_temperature, dtype=jnp.float64), float(temperature_floor))
    ti = jnp.maximum(jnp.asarray(ion_temperature, dtype=jnp.float64), float(temperature_floor))
    return jnp.sqrt((te + ti) / float(ion_mass))


def compute_fci_sheath_recycling(
    density: jnp.ndarray,
    electron_temperature: jnp.ndarray,
    ion_temperature: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    recycling_fraction: float = 0.98,
    electron_sheath_transmission: float = 5.0,
    ion_sheath_transmission: float = 3.5,
    recycled_neutral_energy: float = 0.03,
    ion_mass: float = 1.0,
    density_floor: float = 1.0e-12,
    temperature_floor: float = 1.0e-12,
) -> FciSheathRecyclingResult:
    """Evaluate target loss and recycled neutral sources on FCI endpoints.

    The closure is the first non-axisymmetric target gate. It applies a
    normalized Bohm flux on every traced endpoint and closes exact accounting
    identities for recycled particle and neutral-energy sources. The electron
    particle flux is reconstructed from zero-current balance for singly charged
    ions, so the current residual is an explicit regression field.
    """

    masks = build_fci_target_masks(geometry)
    n = jnp.maximum(jnp.asarray(density, dtype=jnp.float64), float(density_floor))
    te = jnp.maximum(jnp.asarray(electron_temperature, dtype=jnp.float64), float(temperature_floor))
    ti = jnp.maximum(jnp.asarray(ion_temperature, dtype=jnp.float64), float(temperature_floor))
    cs = fci_bohm_sound_speed(
        te,
        ti,
        ion_mass=ion_mass,
        temperature_floor=temperature_floor,
    )

    forward_flux = masks.forward * n * cs
    backward_flux = masks.backward * n * cs
    ion_loss = forward_flux + backward_flux
    electron_loss = ion_loss
    electron_heat_loss = float(electron_sheath_transmission) * ion_loss * te
    ion_heat_loss = float(ion_sheath_transmission) * ion_loss * ti
    target_heat_load = electron_heat_loss + ion_heat_loss
    recycled_particle_source = float(recycling_fraction) * ion_loss
    recycled_neutral_energy_source = float(recycled_neutral_energy) * recycled_particle_source
    plasma_particle_sink = -ion_loss
    plasma_energy_sink = -target_heat_load
    current_residual = electron_loss - ion_loss

    total_ion_loss = jnp.sum(ion_loss)
    total_electron_loss = jnp.sum(electron_loss)
    total_recycled_source = jnp.sum(recycled_particle_source)
    total_heat_load = jnp.sum(target_heat_load)
    total_neutral_energy = jnp.sum(recycled_neutral_energy_source)
    particle_recycling_residual = total_recycled_source - float(recycling_fraction) * total_ion_loss
    neutral_energy_recycling_residual = total_neutral_energy - float(recycled_neutral_energy) * total_recycled_source
    current_balance_residual = total_electron_loss - total_ion_loss

    return FciSheathRecyclingResult(
        masks=masks,
        sound_speed=cs,
        forward_ion_particle_flux=forward_flux,
        backward_ion_particle_flux=backward_flux,
        ion_particle_loss=ion_loss,
        electron_particle_loss=electron_loss,
        electron_heat_loss=electron_heat_loss,
        ion_heat_loss=ion_heat_loss,
        target_heat_load=target_heat_load,
        recycled_particle_source=recycled_particle_source,
        recycled_neutral_energy_source=recycled_neutral_energy_source,
        plasma_particle_sink=plasma_particle_sink,
        plasma_energy_sink=plasma_energy_sink,
        current_residual=current_residual,
        total_ion_particle_loss=total_ion_loss,
        total_electron_particle_loss=total_electron_loss,
        total_recycled_particle_source=total_recycled_source,
        total_target_heat_load=total_heat_load,
        total_recycled_neutral_energy=total_neutral_energy,
        particle_recycling_residual=particle_recycling_residual,
        neutral_energy_recycling_residual=neutral_energy_recycling_residual,
        current_balance_residual=current_balance_residual,
    )


def fci_sheath_recycling_field_rhs(
    fields: dict[str, jnp.ndarray],
    geometry: FciGeometry3D,
    *,
    ion_density_name: str = "Ni",
    electron_density_name: str = "Ne",
    neutral_density_name: str = "Nn",
    ion_pressure_name: str = "Pi",
    electron_pressure_name: str = "Pe",
    neutral_pressure_name: str = "Pn",
    recycling_fraction: float = 0.98,
    electron_sheath_transmission: float = 5.0,
    ion_sheath_transmission: float = 3.5,
    recycled_neutral_energy: float = 0.03,
    ion_mass: float = 1.0,
    density_floor: float = 1.0e-12,
    temperature_floor: float = 1.0e-12,
) -> dict[str, jnp.ndarray]:
    """Return fixed-field RHS arrays for FCI sheath and recycling sources."""

    ion_density = jnp.asarray(fields[ion_density_name], dtype=jnp.float64)
    electron_density = jnp.asarray(fields.get(electron_density_name, ion_density), dtype=jnp.float64)
    ion_pressure = jnp.asarray(fields[ion_pressure_name], dtype=jnp.float64)
    electron_pressure = jnp.asarray(fields[electron_pressure_name], dtype=jnp.float64)
    ion_temperature = ion_pressure / jnp.maximum(ion_density, float(density_floor))
    electron_temperature = electron_pressure / jnp.maximum(electron_density, float(density_floor))
    result = compute_fci_sheath_recycling(
        ion_density,
        electron_temperature,
        ion_temperature,
        geometry,
        recycling_fraction=recycling_fraction,
        electron_sheath_transmission=electron_sheath_transmission,
        ion_sheath_transmission=ion_sheath_transmission,
        recycled_neutral_energy=recycled_neutral_energy,
        ion_mass=ion_mass,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
    )
    rhs = {
        ion_density_name: -result.ion_particle_loss,
        electron_density_name: -result.electron_particle_loss,
        neutral_density_name: result.recycled_particle_source,
        ion_pressure_name: -result.ion_heat_loss,
        electron_pressure_name: -result.electron_heat_loss,
        neutral_pressure_name: result.recycled_neutral_energy_source,
    }
    return {name: value for name, value in rhs.items() if name in fields}
