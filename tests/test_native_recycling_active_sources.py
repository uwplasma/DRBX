from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import (
    _build_recycling_runtime_model,
    _build_recycling_state_fields,
    _prepare_open_field_states,
)
from jax_drb.native.recycling_active_sources import (
    add_field_rhs_contribution,
    fixed_layout_recycling_source_field_rhs_from_active_fields,
)
from jax_drb.native.recycling_collision_closure import (
    fixed_layout_collision_friction_heat_exchange_field_rhs_from_active_fields,
)
from jax_drb.native.recycling_collisions import compute_collision_frequencies
from jax_drb.native.recycling_fixed_residual import (
    RecyclingFixedState,
    build_fixed_array_rhs,
    fixed_state_from_fields,
)
from jax_drb.native.recycling_layout import build_recycling_packed_state_layout
from jax_drb.native.recycling_neutral_diffusion import (
    fixed_layout_neutral_parallel_diffusion_field_rhs_from_active_fields,
)
from jax_drb.native.recycling_reactions import (
    fixed_layout_dthe_reaction_field_rhs_from_active_fields,
    neutral_charge_exchange_collision_rates,
    neutral_ionisation_collision_rates,
)
from jax_drb.native.recycling_setup import build_species_field_overrider
from jax_drb.native.recycling_targets import fixed_layout_target_recycling_field_rhs
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars


_REFERENCE_ROOT = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "reference-root"
    / "tests"
    / "integrated"
)
_DTHE_INPUT = _REFERENCE_ROOT / "1D-recycling-dthe" / "data" / "BOUT.inp"
_HYDROGEN_INPUT = _REFERENCE_ROOT / "1D-recycling" / "data" / "BOUT.inp"


def _build_context(input_path: Path):
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    fields = _build_recycling_state_fields(runtime_model)
    species = build_species_field_overrider(runtime_model.species_templates, mesh=mesh)(
        fields
    )
    prepared, ion_boundary, _electron_boundary = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    layout = build_recycling_packed_state_layout(
        fields=fields,
        field_names=runtime_model.field_names,
        feedback_names=runtime_model.feedback_names,
        mesh=mesh,
    )
    state = fixed_state_from_fields(
        fields,
        feedback_integrals={name: 0.0 for name in runtime_model.feedback_names},
        layout=layout,
    )
    active_fields = {
        name: value for name, value in zip(layout.field_names, state.field_values, strict=True)
    }
    return (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        species,
        prepared,
        ion_boundary,
        layout,
        state,
        active_fields,
    )


def _rate_inputs(config, species, prepared, scalars):
    collision_rates = compute_collision_frequencies(
        config,
        species,
        prepared,
        dataset_scalars=scalars,
    )
    ionisation_rates = neutral_ionisation_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=scalars,
    )
    charge_exchange_rates = neutral_charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=scalars,
    )
    return collision_rates, ionisation_rates, charge_exchange_rates


def test_composed_active_sources_sum_dthe_reaction_collision_and_neutral_terms() -> None:
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        species,
        prepared,
        _ion_boundary,
        layout,
        _state,
        active_fields,
    ) = _build_context(_DTHE_INPUT)
    collision_rates, ionisation_rates, charge_exchange_rates = _rate_inputs(
        config,
        species,
        prepared,
        scalars,
    )

    expected: dict[str, object] = {}
    add_field_rhs_contribution(
        expected,
        fixed_layout_dthe_reaction_field_rhs_from_active_fields(
            config,
            active_fields=active_fields,
            species=species,
            dataset_scalars=scalars,
        ),
    )
    add_field_rhs_contribution(
        expected,
        fixed_layout_collision_friction_heat_exchange_field_rhs_from_active_fields(
            config,
            active_fields=active_fields,
            species=species,
            collision_rates={
                key: value[layout.active_slices]
                for key, value in collision_rates.items()
            },
        ),
    )
    add_field_rhs_contribution(
        expected,
        fixed_layout_neutral_parallel_diffusion_field_rhs_from_active_fields(
            config,
            active_fields=active_fields,
            layout=layout,
            species_templates=runtime_model.species_templates,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=scalars,
            collision_rates=collision_rates,
            ionisation_rates=ionisation_rates,
            charge_exchange_rates=charge_exchange_rates,
        ),
    )

    composed = fixed_layout_recycling_source_field_rhs_from_active_fields(
        config,
        active_fields=active_fields,
        layout=layout,
        species=species,
        species_templates=runtime_model.species_templates,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        collision_rates=collision_rates,
        ionisation_rates=ionisation_rates,
        charge_exchange_rates=charge_exchange_rates,
        include_dthe_reactions=True,
        include_pointwise_collisions=True,
        include_neutral_parallel_diffusion=True,
    )

    assert composed.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(
            np.asarray(composed[name]),
            np.asarray(expected[name]),
            rtol=1.0e-9,
            atol=1.0e-10,
        )


def test_composed_active_sources_promote_to_fixed_array_rhs_and_jvp() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        species,
        prepared,
        _ion_boundary,
        layout,
        state,
        _active_fields,
    ) = _build_context(_DTHE_INPUT)
    collision_rates, ionisation_rates, charge_exchange_rates = _rate_inputs(
        config,
        species,
        prepared,
        scalars,
    )
    pressure_index = layout.field_names.index("Pd")

    def field_rhs(active_fields: dict[str, object], _feedback: object) -> dict[str, object]:
        return fixed_layout_recycling_source_field_rhs_from_active_fields(
            config,
            active_fields=active_fields,
            layout=layout,
            species=species,
            species_templates=runtime_model.species_templates,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=scalars,
            collision_rates={
                key: jnp.asarray(value, dtype=jnp.float64)
                for key, value in collision_rates.items()
            },
            ionisation_rates={
                key: jnp.asarray(value, dtype=jnp.float64)
                for key, value in ionisation_rates.items()
            },
            charge_exchange_rates={
                key: jnp.asarray(value, dtype=jnp.float64)
                for key, value in charge_exchange_rates.items()
            },
            include_dthe_reactions=True,
            include_pointwise_collisions=True,
            include_neutral_parallel_diffusion=True,
        )

    fixed_rhs = build_fixed_array_rhs(field_rhs, layout=layout)

    def qoi(scale):
        fields = list(state.field_values)
        fields[pressure_index] = jnp.asarray(fields[pressure_index]) * scale
        rhs_state = fixed_rhs(
            RecyclingFixedState(
                field_values=tuple(fields),
                feedback_values=jnp.asarray(state.feedback_values, dtype=jnp.float64),
            )
        )
        rhs_fields = {
            name: value
            for name, value in zip(layout.field_names, rhs_state.field_values, strict=True)
        }
        return jnp.sum(rhs_fields["Nd"]) + 0.05 * jnp.sum(rhs_fields["Pd"])

    value, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    step = 1.0e-5
    finite_difference = (
        qoi(jnp.array(1.0 + step)) - qoi(jnp.array(1.0 - step))
    ) / (2.0 * step)

    assert np.isfinite(float(value))
    assert np.isfinite(float(tangent))
    np.testing.assert_allclose(
        float(tangent),
        float(finite_difference),
        rtol=1.0e-5,
        atol=1.0e-8,
    )


def test_composed_active_sources_sum_hydrogen_neutral_and_target_terms() -> None:
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        species,
        prepared,
        ion_boundary,
        layout,
        _state,
        active_fields,
    ) = _build_context(_HYDROGEN_INPUT)
    collision_rates, ionisation_rates, charge_exchange_rates = _rate_inputs(
        config,
        species,
        prepared,
        scalars,
    )
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    neutrals = tuple(
        sp for sp in species.values() if sp.charge == 0.0 and sp.name != "e"
    )
    expected: dict[str, object] = {}
    add_field_rhs_contribution(
        expected,
        fixed_layout_neutral_parallel_diffusion_field_rhs_from_active_fields(
            config,
            active_fields=active_fields,
            layout=layout,
            species_templates=runtime_model.species_templates,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=scalars,
            collision_rates=collision_rates,
            ionisation_rates=ionisation_rates,
            charge_exchange_rates=charge_exchange_rates,
        ),
    )
    add_field_rhs_contribution(
        expected,
        fixed_layout_target_recycling_field_rhs(
            ions=ions,
            prepared=prepared,
            neutrals=neutrals,
            ion_velocity=ion_boundary.velocity,
            layout=layout,
            mesh=mesh,
            metrics=metrics,
            gamma_i=0.0,
        ),
    )

    composed = fixed_layout_recycling_source_field_rhs_from_active_fields(
        config,
        active_fields=active_fields,
        layout=layout,
        species=species,
        species_templates=runtime_model.species_templates,
        prepared=prepared,
        ion_velocity=ion_boundary.velocity,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        collision_rates=collision_rates,
        ionisation_rates=ionisation_rates,
        charge_exchange_rates=charge_exchange_rates,
        include_neutral_parallel_diffusion=True,
        include_target_recycling=True,
    )

    assert composed.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(
            np.asarray(composed[name]),
            np.asarray(expected[name]),
            rtol=1.0e-9,
            atol=1.0e-10,
        )
