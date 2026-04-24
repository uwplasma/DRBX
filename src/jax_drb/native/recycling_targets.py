from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from .array_backend import use_jax_backend
from .mesh import StructuredMesh
from .metrics import StructuredMetrics
from .open_field import TargetBoundaryGeometry, compute_target_recycling_sources
from .recycling_setup import OpenFieldSpecies
from .recycling_state import PreparedSpeciesState


@dataclass(frozen=True)
class RecyclingTerms:
    density_source: dict[str, np.ndarray]
    energy_source: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]


def target_recycling_sources(
    *,
    ions: tuple[OpenFieldSpecies, ...],
    prepared: dict[str, PreparedSpeciesState],
    neutrals: tuple[OpenFieldSpecies, ...],
    ion_velocity: dict[str, np.ndarray],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    gamma_i: float,
    lower_geometry: TargetBoundaryGeometry | None = None,
    upper_geometry: TargetBoundaryGeometry | None = None,
) -> RecyclingTerms:
    neutral_lookup = {sp.name: sp for sp in neutrals}
    use_jax = use_jax_backend(
        *(prepared[sp.name].density for sp in (*ions, *neutrals) if sp.name in prepared),
        *(ion_velocity.get(sp.name) for sp in ions),
        metrics.J,
        metrics.dy,
        metrics.dx,
        metrics.dz,
        metrics.g_22,
    )
    density_source = {
        sp.name: (
            jnp.zeros_like(jnp.asarray(sp.density, dtype=jnp.float64), dtype=jnp.float64)
            if use_jax
            else np.zeros_like(sp.density, dtype=np.float64)
        )
        for sp in (*ions, *neutrals)
    }
    energy_source = {
        sp.name: (
            jnp.zeros_like(jnp.asarray(sp.density, dtype=jnp.float64), dtype=jnp.float64)
            if use_jax
            else np.zeros_like(sp.density, dtype=np.float64)
        )
        for sp in (*ions, *neutrals)
    }
    diagnostics: dict[str, np.ndarray] = {}

    for ion in ions:
        if not ion.target_recycle or ion.recycle_as is None or ion.recycle_as not in neutral_lookup:
            continue
        neutral = neutral_lookup[ion.recycle_as]
        ion_state = prepared[ion.name]
        result = compute_target_recycling_sources(
            ion_state.density,
            ion_velocity[ion.name],
            ion_state.temperature,
            mesh=mesh,
            J=np.asarray(metrics.J, dtype=np.float64),
            dy=np.asarray(metrics.dy, dtype=np.float64),
            dx=np.asarray(metrics.dx, dtype=np.float64),
            dz=np.asarray(metrics.dz, dtype=np.float64),
            g_22=np.asarray(metrics.g_22, dtype=np.float64),
            target_multiplier=ion.target_recycle_multiplier,
            target_energy=ion.target_recycle_energy,
            gamma_i=gamma_i,
            target_fast_recycle_fraction=ion.target_fast_recycle_fraction,
            target_fast_recycle_energy_factor=ion.target_fast_recycle_energy_factor,
            lower_y=mesh.has_lower_y_target,
            upper_y=mesh.has_upper_y_target,
            lower_geometry=lower_geometry,
            upper_geometry=upper_geometry,
        )
        density_source[neutral.name] += result.density_source if use_jax else np.asarray(result.density_source, dtype=np.float64)
        energy_source[neutral.name] += result.energy_source if use_jax else np.asarray(result.energy_source, dtype=np.float64)
        diagnostics[f"S{neutral.name}_target_recycle"] = (
            result.target_density_source if use_jax else np.asarray(result.target_density_source, dtype=np.float64)
        )
        diagnostics[f"E{neutral.name}_target_recycle"] = (
            result.target_energy_source if use_jax else np.asarray(result.target_energy_source, dtype=np.float64)
        )

    return RecyclingTerms(density_source=density_source, energy_source=energy_source, diagnostics=diagnostics)


def electron_zero_current_velocity(
    ions: tuple[OpenFieldSpecies, ...],
    *,
    prepared: dict[str, PreparedSpeciesState],
    ion_velocity: dict[str, np.ndarray],
    electron_density: np.ndarray,
) -> np.ndarray:
    use_jax = use_jax_backend(electron_density, *(prepared[ion.name].density for ion in ions), *(ion_velocity[ion.name] for ion in ions))
    current = (
        jnp.zeros_like(jnp.asarray(electron_density, dtype=jnp.float64), dtype=jnp.float64)
        if use_jax
        else np.zeros_like(electron_density, dtype=np.float64)
    )
    for ion in ions:
        current += ion.charge * prepared[ion.name].density * ion_velocity[ion.name]
    if use_jax:
        return current / jnp.maximum(jnp.asarray(electron_density, dtype=jnp.float64), 1.0e-5)
    return current / np.maximum(electron_density, 1.0e-5)


def grad_par_electron_force_balance_open(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    """Match the open-field electron-force-balance centered stencil."""
    result = np.zeros_like(field, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    g_22 = np.asarray(metrics.g_22, dtype=np.float64)

    for i in range(mesh.xstart, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                result[i, j, k] = (
                    0.5
                    * (field[i, j + 1, k] - field[i, j - 1, k])
                    / (dy[i, j, k] * np.sqrt(g_22[i, j, k]))
                )
    return result
