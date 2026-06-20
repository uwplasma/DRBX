from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from .array_backend import use_jax_backend
from .mesh import StructuredMesh
from .metrics import StructuredMetrics
from .open_field import TargetBoundaryGeometry, compute_target_recycling_sources
from .recycling_layout import RecyclingPackedStateLayout, recycling_layout_field_name_set
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


def fixed_layout_target_recycling_field_rhs(
    *,
    ions: tuple[OpenFieldSpecies, ...],
    prepared: dict[str, PreparedSpeciesState],
    neutrals: tuple[OpenFieldSpecies, ...],
    ion_velocity: dict[str, np.ndarray],
    layout: RecyclingPackedStateLayout,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    gamma_i: float,
    lower_geometry: TargetBoundaryGeometry | None = None,
    upper_geometry: TargetBoundaryGeometry | None = None,
) -> dict[str, np.ndarray]:
    """Map prepared target-recycling sources into active fixed-layout RHS terms."""

    terms = target_recycling_sources(
        ions=ions,
        prepared=prepared,
        neutrals=neutrals,
        ion_velocity=ion_velocity,
        mesh=mesh,
        metrics=metrics,
        gamma_i=gamma_i,
        lower_geometry=lower_geometry,
        upper_geometry=upper_geometry,
    )
    layout_fields = recycling_layout_field_name_set(layout)
    active_slices = layout.active_slices
    field_rhs: dict[str, np.ndarray] = {}
    for neutral in neutrals:
        if neutral.density_name in layout_fields:
            field_rhs[neutral.density_name] = terms.density_source[neutral.name][
                active_slices
            ]
        if neutral.has_pressure and neutral.pressure_name in layout_fields:
            field_rhs[neutral.pressure_name] = (
                (2.0 / 3.0) * terms.energy_source[neutral.name][active_slices]
            )
    return field_rhs


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
    if use_jax_backend(field):
        field_array = jnp.asarray(field, dtype=jnp.float64)
        result = jnp.zeros_like(field_array, dtype=jnp.float64)
        dy = jnp.asarray(metrics.dy, dtype=jnp.float64)
        g_22 = jnp.asarray(metrics.g_22, dtype=jnp.float64)
        x_slice = slice(mesh.xstart, mesh.xend + 1)
        y_slice = slice(mesh.ystart, mesh.yend + 1)
        active_gradient = (
            0.5
            * (
                field_array[x_slice, mesh.ystart + 1 : mesh.yend + 2, :]
                - field_array[x_slice, mesh.ystart - 1 : mesh.yend, :]
            )
            / (dy[x_slice, y_slice, :] * jnp.sqrt(g_22[x_slice, y_slice, :]))
        )
        return result.at[x_slice, y_slice, :].set(active_gradient)

    result = np.zeros_like(field, dtype=np.float64)
    dy = np.asarray(metrics.dy, dtype=np.float64)
    g_22 = np.asarray(metrics.g_22, dtype=np.float64)
    x_slice = slice(mesh.xstart, mesh.xend + 1)
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    result[x_slice, y_slice, :] = (
        0.5
        * (
            np.asarray(field[x_slice, mesh.ystart + 1 : mesh.yend + 2, :], dtype=np.float64)
            - np.asarray(field[x_slice, mesh.ystart - 1 : mesh.yend, :], dtype=np.float64)
        )
        / (dy[x_slice, y_slice, :] * np.sqrt(g_22[x_slice, y_slice, :]))
    )
    return result


def grad_par_electron_force_balance_open_active(
    field: np.ndarray,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
) -> np.ndarray:
    """Return the active-domain slice of electron force-balance gradient."""

    if use_jax_backend(field):
        field_array = jnp.asarray(field, dtype=jnp.float64)
        dy = jnp.asarray(metrics.dy, dtype=jnp.float64)
        g_22 = jnp.asarray(metrics.g_22, dtype=jnp.float64)
        x_slice = slice(mesh.xstart, mesh.xend + 1)
        y_slice = slice(mesh.ystart, mesh.yend + 1)
        return (
            0.5
            * (
                field_array[x_slice, mesh.ystart + 1 : mesh.yend + 2, :]
                - field_array[x_slice, mesh.ystart - 1 : mesh.yend, :]
            )
            / (dy[x_slice, y_slice, :] * jnp.sqrt(g_22[x_slice, y_slice, :]))
        )

    dy = np.asarray(metrics.dy, dtype=np.float64)
    g_22 = np.asarray(metrics.g_22, dtype=np.float64)
    x_slice = slice(mesh.xstart, mesh.xend + 1)
    y_slice = slice(mesh.ystart, mesh.yend + 1)
    return (
        0.5
        * (
            np.asarray(
                field[x_slice, mesh.ystart + 1 : mesh.yend + 2, :],
                dtype=np.float64,
            )
            - np.asarray(
                field[x_slice, mesh.ystart - 1 : mesh.yend, :],
                dtype=np.float64,
            )
        )
        / (dy[x_slice, y_slice, :] * np.sqrt(g_22[x_slice, y_slice, :]))
    )
