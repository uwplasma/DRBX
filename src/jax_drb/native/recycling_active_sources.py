from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..config.boutinp import BoutConfig
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
from .recycling_setup import OpenFieldSpecies
from .recycling_state import PreparedSpeciesState
from .recycling_targets import fixed_layout_target_recycling_field_rhs


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
