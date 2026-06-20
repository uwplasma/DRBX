from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import jax.numpy as jnp

from ..config.boutinp import BoutConfig
from .array_backend import use_jax_backend
from .metrics import StructuredMetrics
from .mesh import StructuredMesh
from .open_field import TargetBoundaryGeometry
from .recycling_collision_closure import (
    fixed_layout_collision_friction_heat_exchange_field_rhs_from_active_fields,
)
from .recycling_layout import RecyclingPackedStateLayout
from .recycling_neutral_diffusion import (
    fixed_layout_neutral_parallel_diffusion_field_rhs_from_active_fields,
)
from .recycling_reactions import fixed_layout_dthe_reaction_field_rhs_from_active_fields
from .recycling_rhs_terms import (
    assemble_electron_parallel_force_terms,
    assemble_electron_pressure_rhs_terms,
    assemble_ion_rhs_terms,
    assemble_neutral_rhs_terms,
)
from .recycling_setup import OpenFieldSpecies
from .recycling_state import PreparedSpeciesState
from .recycling_targets import fixed_layout_target_recycling_field_rhs
from .safe_math import sqrt_nonnegative


def add_field_rhs_contribution(
    accumulated: dict[str, np.ndarray],
    contribution: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Add one active-layout field-RHS contribution dictionary in place."""

    for name, value in contribution.items():
        accumulated[name] = accumulated[name] + value if name in accumulated else value
    return accumulated


def fixed_layout_recycling_source_field_rhs_from_active_fields(
    config: BoutConfig,
    *,
    active_fields: Mapping[str, np.ndarray],
    layout: RecyclingPackedStateLayout,
    species: Mapping[str, OpenFieldSpecies],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: dict[str, float],
    collision_rates: Mapping[tuple[str, str], np.ndarray] | None = None,
    ionisation_rates: Mapping[str, np.ndarray] | None = None,
    charge_exchange_rates: Mapping[str, np.ndarray] | None = None,
    species_templates: Mapping[str, OpenFieldSpecies] | None = None,
    prepared: Mapping[str, PreparedSpeciesState] | None = None,
    ion_velocity: Mapping[str, np.ndarray] | None = None,
    gamma_i: float = 0.0,
    lower_geometry: TargetBoundaryGeometry | None = None,
    upper_geometry: TargetBoundaryGeometry | None = None,
    include_dthe_reactions: bool = False,
    include_pointwise_collisions: bool = False,
    include_neutral_parallel_diffusion: bool = False,
    include_target_recycling: bool = False,
) -> dict[str, np.ndarray]:
    """Compose promoted source terms into one active fixed-layout field RHS.

    This is a staging seam, not a complete recycling RHS: it sums source terms
    that already have active-layout parity gates while the remaining advective,
    force-balance, conduction, viscosity, feedback, and full sheath assembly
    terms are ported separately.
    """

    field_rhs: dict[str, np.ndarray] = {}
    if include_dthe_reactions:
        add_field_rhs_contribution(
            field_rhs,
            fixed_layout_dthe_reaction_field_rhs_from_active_fields(
                config,
                active_fields=active_fields,
                species=species,
                dataset_scalars=dataset_scalars,
            ),
        )

    if include_pointwise_collisions:
        if collision_rates is None:
            raise ValueError(
                "collision_rates are required when include_pointwise_collisions=True."
            )
        add_field_rhs_contribution(
            field_rhs,
            fixed_layout_collision_friction_heat_exchange_field_rhs_from_active_fields(
                config,
                active_fields=active_fields,
                species=species,
                collision_rates=_active_rate_slices(collision_rates, layout=layout),
            ),
        )

    if include_neutral_parallel_diffusion:
        add_field_rhs_contribution(
            field_rhs,
            fixed_layout_neutral_parallel_diffusion_field_rhs_from_active_fields(
                config,
                active_fields=active_fields,
                layout=layout,
                species_templates=(
                    species if species_templates is None else species_templates
                ),
                mesh=mesh,
                metrics=metrics,
                dataset_scalars=dataset_scalars,
                collision_rates=_full_rate_dict_or_none(collision_rates, layout=layout),
                ionisation_rates=_full_rate_dict_or_none(
                    ionisation_rates, layout=layout
                ),
                charge_exchange_rates=_full_rate_dict_or_none(
                    charge_exchange_rates, layout=layout
                ),
            ),
        )

    if include_target_recycling:
        if prepared is None or ion_velocity is None:
            raise ValueError(
                "prepared and ion_velocity are required when "
                "include_target_recycling=True."
            )
        ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
        neutrals = tuple(
            sp for sp in species.values() if sp.charge == 0.0 and sp.name != "e"
        )
        add_field_rhs_contribution(
            field_rhs,
            fixed_layout_target_recycling_field_rhs(
                ions=ions,
                prepared=dict(prepared),
                neutrals=neutrals,
                ion_velocity=dict(ion_velocity),
                layout=layout,
                mesh=mesh,
                metrics=metrics,
                gamma_i=gamma_i,
                lower_geometry=lower_geometry,
                upper_geometry=upper_geometry,
            ),
        )

    return field_rhs


def assemble_fixed_layout_recycling_field_rhs_from_sources(
    *,
    source_field_rhs: Mapping[str, np.ndarray],
    layout: RecyclingPackedStateLayout,
    species: Mapping[str, OpenFieldSpecies],
    prepared: Mapping[str, PreparedSpeciesState],
    ion_velocity: Mapping[str, np.ndarray],
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    explicit_pressure_sources: Mapping[str, np.ndarray] | None = None,
    pressure_source_override_names: tuple[str, ...] = (),
    defer_active_source_scatter: bool = True,
) -> dict[str, np.ndarray]:
    """Insert active source blocks into the existing open-field RHS assembly.

    ``source_field_rhs`` is keyed by evolved field names. Density and momentum
    entries are inserted directly, while pressure entries are converted back to
    internal energy sources using ``Q = 3 P_rhs / 2`` because the pressure
    equation receives ``(2/3) Q``.

    By default additive source entries are kept on the active layout until the
    final active RHS has been assembled.  This avoids repeated full-field
    zero/scatter/slice operations in the JAX-linearized residual path while
    preserving the same full-field transport and force-balance stencils.  The
    function falls back to full scattering when an electron momentum source is
    present, because that term changes the parallel electric field before ion
    momentum sources are assembled.
    """

    unknown_fields = set(source_field_rhs) - set(layout.field_names)
    if unknown_fields:
        unknown = ", ".join(repr(name) for name in sorted(unknown_fields))
        raise ValueError(f"Source RHS returned unknown layout entries: {unknown}.")

    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    neutrals = tuple(sp for sp in species.values() if sp.charge == 0.0)
    if "e" not in species or "e" not in prepared:
        raise KeyError("Electron species 'e' is required for recycling RHS assembly.")

    use_jax = use_jax_backend(
        *(source_field_rhs.values()),
        *(state.density for state in prepared.values()),
        *(state.pressure for state in prepared.values()),
        *(state.velocity for state in prepared.values()),
    )
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    defer_sources = bool(defer_active_source_scatter) and _can_defer_source_scatter(
        source_field_rhs,
        species=species,
    )
    scattered_source_rhs: Mapping[str, np.ndarray] = (
        {} if defer_sources else source_field_rhs
    )

    density_source: dict[str, np.ndarray] = {}
    energy_source: dict[str, np.ndarray] = {}
    momentum_source: dict[str, np.ndarray] = {}
    for name, sp in species.items():
        template = prepared[name].density
        density_source[name] = _full_source_from_field_rhs(
            scattered_source_rhs.get(sp.density_name),
            template=template,
            layout=layout,
            use_jax=use_jax,
        )
        pressure_rhs = _full_source_from_field_rhs(
            scattered_source_rhs.get(sp.pressure_name),
            template=template,
            layout=layout,
            use_jax=use_jax,
        )
        energy_source[name] = 1.5 * pressure_rhs
        momentum_source[name] = _full_source_from_field_rhs(
            scattered_source_rhs.get(sp.momentum_name),
            template=template,
            layout=layout,
            use_jax=use_jax,
        )

    electron_force_terms = assemble_electron_parallel_force_terms(
        electron_pressure=prepared["e"].pressure,
        electron_density=prepared["e"].density,
        electron_momentum_source=momentum_source["e"],
        ion_density={ion.name: prepared[ion.name].density for ion in ions},
        ion_charge={ion.name: ion.charge for ion in ions},
        ion_momentum_source={ion.name: momentum_source[ion.name] for ion in ions},
        mesh=mesh,
        metrics=metrics,
    )
    for ion in ions:
        momentum_source[ion.name] = electron_force_terms.ion_momentum_source[ion.name]

    pressure_sources = dict(explicit_pressure_sources or {})
    pressure_override_names = frozenset(pressure_source_override_names)
    assembled: dict[str, np.ndarray] = {}

    for ion in ions:
        state = prepared[ion.name]
        fastest_wave = sqrt_nonnegative(array(state.temperature, dtype=dtype) / ion.atomic_mass)
        terms = assemble_ion_rhs_terms(
            density_source=density_source[ion.name],
            explicit_pressure_source=_full_pressure_source(
                pressure_sources.get(ion.name),
                template=state.density,
                use_jax=use_jax,
            ),
            momentum_source=momentum_source[ion.name],
            atomic_mass=ion.atomic_mass,
            density_floor=ion.density_floor,
            ion_state=state,
            ion_velocity=ion_velocity[ion.name],
            fastest_wave=fastest_wave,
            mesh=mesh,
            metrics=metrics,
            energy_source=energy_source[ion.name],
        )
        _set_active_if_layout_field(assembled, ion.density_name, terms.density_total, layout)
        _set_active_if_layout_field(assembled, ion.pressure_name, terms.pressure_total, layout)
        _set_active_if_layout_field(assembled, ion.momentum_name, terms.momentum_total, layout)

    electron_state = prepared["e"]
    electron_fastest_wave = sqrt_nonnegative(
        array(electron_state.temperature, dtype=dtype) / species["e"].atomic_mass
    )
    electron_terms = assemble_electron_pressure_rhs_terms(
        explicit_pressure_source=_full_pressure_source(
            pressure_sources.get("e"),
            template=electron_state.density,
            use_jax=use_jax,
        ),
        electron_pressure=electron_state.pressure,
        electron_velocity=electron_state.velocity,
        electron_fastest_wave=electron_fastest_wave,
        electron_energy_source=energy_source["e"],
        mesh=mesh,
        metrics=metrics,
    )
    _set_active_if_layout_field(assembled, species["e"].pressure_name, electron_terms.total, layout)

    for neutral in neutrals:
        state = prepared[neutral.name]
        fastest_wave = sqrt_nonnegative(array(state.temperature, dtype=dtype) / neutral.atomic_mass)
        terms = assemble_neutral_rhs_terms(
            density_source=density_source[neutral.name],
            explicit_pressure_source=_full_pressure_source(
                pressure_sources.get(neutral.name),
                template=state.density,
                use_jax=use_jax,
            ),
            momentum_source=momentum_source[neutral.name],
            atomic_mass=neutral.atomic_mass,
            density_floor=neutral.density_floor,
            neutral_state=state,
            neutral_velocity=state.velocity,
            fastest_wave=fastest_wave,
            mesh=mesh,
            metrics=metrics,
            energy_source=energy_source[neutral.name],
            include_energy_source=neutral.name not in pressure_override_names,
        )
        _set_active_if_layout_field(assembled, neutral.density_name, terms.density_total, layout)
        _set_active_if_layout_field(assembled, neutral.pressure_name, terms.pressure_total, layout)
        _set_active_if_layout_field(assembled, neutral.momentum_name, terms.momentum_total, layout)

    if defer_sources:
        _add_deferred_active_source_rhs(
            assembled,
            source_field_rhs,
            layout=layout,
            species=species,
            use_jax=use_jax,
            pressure_source_override_names=tuple(pressure_override_names),
        )

    return assembled


def _can_defer_source_scatter(
    source_field_rhs: Mapping[str, np.ndarray],
    *,
    species: Mapping[str, OpenFieldSpecies],
) -> bool:
    """Return whether active sources can be added after full stencil assembly."""

    electron = species.get("e")
    if electron is None:
        return True
    return electron.momentum_name not in source_field_rhs


def _add_deferred_active_source_rhs(
    assembled: dict[str, np.ndarray],
    source_field_rhs: Mapping[str, np.ndarray],
    *,
    layout: RecyclingPackedStateLayout,
    species: Mapping[str, OpenFieldSpecies],
    use_jax: bool,
    pressure_source_override_names: tuple[str, ...] = (),
) -> None:
    pressure_overrides = frozenset(pressure_source_override_names)
    for species_name, sp in species.items():
        _add_deferred_field_source(
            assembled,
            sp.density_name,
            source_field_rhs.get(sp.density_name),
            layout=layout,
            use_jax=use_jax,
        )
        if not (sp.charge == 0.0 and species_name in pressure_overrides):
            _add_deferred_field_source(
                assembled,
                sp.pressure_name,
                source_field_rhs.get(sp.pressure_name),
                layout=layout,
                use_jax=use_jax,
            )
        _add_deferred_field_source(
            assembled,
            sp.momentum_name,
            source_field_rhs.get(sp.momentum_name),
            layout=layout,
            use_jax=use_jax,
        )


def _add_deferred_field_source(
    assembled: dict[str, np.ndarray],
    field_name: str,
    value: np.ndarray | None,
    *,
    layout: RecyclingPackedStateLayout,
    use_jax: bool,
) -> None:
    if value is None or field_name not in layout.field_names:
        return
    source = _active_source_from_field_rhs(value, layout=layout, use_jax=use_jax)
    assembled[field_name] = (
        source if field_name not in assembled else assembled[field_name] + source
    )


def _active_rate_slices(
    rates: Mapping[tuple[str, str], np.ndarray],
    *,
    layout: RecyclingPackedStateLayout,
) -> dict[tuple[str, str], np.ndarray]:
    active_rates: dict[tuple[str, str], np.ndarray] = {}
    for key, value in rates.items():
        active_rates[key] = (
            value
            if tuple(value.shape) == tuple(layout.active_shape)
            else value[layout.active_slices]
        )
    return active_rates


def _full_rate_dict_or_none(
    rates: Mapping[object, np.ndarray] | None,
    *,
    layout: RecyclingPackedStateLayout,
) -> dict[object, np.ndarray] | None:
    if rates is None:
        return None
    full_shape = tuple(layout.field_templates[0].shape) if layout.field_templates else ()
    if not all(tuple(value.shape) == full_shape for value in rates.values()):
        return None
    return dict(rates)


def _active_source_from_field_rhs(
    value: np.ndarray,
    *,
    layout: RecyclingPackedStateLayout,
    use_jax: bool,
) -> np.ndarray:
    array = jnp.asarray if use_jax else np.asarray
    dtype = jnp.float64 if use_jax else np.float64
    source = array(value, dtype=dtype)
    if tuple(source.shape) == tuple(layout.active_shape):
        return source
    full_shape = tuple(layout.field_templates[0].shape)
    if tuple(source.shape) == full_shape:
        return source[layout.active_slices]
    raise ValueError(
        f"Source RHS shape {tuple(source.shape)} does not match active shape "
        f"{tuple(layout.active_shape)} or full shape {full_shape}."
    )


def _full_source_from_field_rhs(
    value: np.ndarray | None,
    *,
    template: np.ndarray,
    layout: RecyclingPackedStateLayout,
    use_jax: bool,
) -> np.ndarray:
    if use_jax:
        full = jnp.zeros_like(jnp.asarray(template, dtype=jnp.float64))
        if value is None:
            return full
        source = jnp.asarray(value, dtype=jnp.float64)
        if tuple(source.shape) == tuple(layout.active_shape):
            return full.at[layout.active_slices].set(source)
        if tuple(source.shape) == tuple(full.shape):
            return source
    else:
        full = np.zeros_like(np.asarray(template, dtype=np.float64), dtype=np.float64)
        if value is None:
            return full
        source = np.asarray(value, dtype=np.float64)
        if tuple(source.shape) == tuple(layout.active_shape):
            full[layout.active_slices] = source
            return full
        if tuple(source.shape) == tuple(full.shape):
            return source
    raise ValueError(
        f"Source RHS shape {tuple(source.shape)} does not match active shape "
        f"{tuple(layout.active_shape)} or full shape {tuple(full.shape)}."
    )


def _full_pressure_source(
    value: np.ndarray | None,
    *,
    template: np.ndarray,
    use_jax: bool,
) -> np.ndarray:
    if use_jax:
        return (
            jnp.zeros_like(jnp.asarray(template, dtype=jnp.float64))
            if value is None
            else jnp.asarray(value, dtype=jnp.float64)
        )
    return (
        np.zeros_like(np.asarray(template, dtype=np.float64), dtype=np.float64)
        if value is None
        else np.asarray(value, dtype=np.float64)
    )


def _set_active_if_layout_field(
    assembled: dict[str, np.ndarray],
    field_name: str,
    value: np.ndarray,
    layout: RecyclingPackedStateLayout,
) -> None:
    if field_name in layout.field_names:
        assembled[field_name] = value[layout.active_slices]
