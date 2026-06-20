from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import (
    _build_fixed_full_field_recycling_rhs,
    _build_promoted_active_source_recycling_rhs,
    build_recycling_1d_bdf2_residual_context,
    build_recycling_1d_backward_euler_residual_context,
    _build_recycling_runtime_model,
    _build_recycling_state_fields,
    _current_feedback_errors,
    _prepare_open_field_states,
)
from jax_drb.native.recycling_active_sources import (
    add_field_rhs_contribution,
    assemble_fixed_layout_recycling_field_rhs_from_sources,
    fixed_layout_recycling_source_field_rhs_from_active_fields,
)
from jax_drb.native.recycling_collision_closure import (
    apply_collision_closure,
    fixed_layout_collision_friction_heat_exchange_field_rhs_from_active_fields,
    fixed_layout_collision_transport_field_rhs_from_prepared,
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
_BOUNDED_PROMOTED_DTHE_OVERRIDES = (
    "hermes:components=(d+, d, t+, t, he+, he, e, sheath_boundary, "
    "braginskii_collisions, braginskii_friction, braginskii_heat_exchange, "
    "recycling, reactions, electron_force_balance, neutral_parallel_diffusion)",
    "d+:type=(evolve_density, evolve_pressure, evolve_momentum, noflow_boundary)",
    "t+:type=(evolve_density, evolve_pressure, evolve_momentum, noflow_boundary)",
    "he+:type=(evolve_density, evolve_pressure, evolve_momentum, noflow_boundary)",
)
_BOUNDED_PROMOTED_DTHE_FEEDBACK_OVERRIDES = (
    "hermes:components=(d+, d, t+, t, he+, he, e, sheath_boundary, "
    "braginskii_collisions, braginskii_friction, braginskii_heat_exchange, "
    "recycling, reactions, electron_force_balance, neutral_parallel_diffusion)",
)
_BOUNDED_COLLISION_TRANSPORT_DTHE_OVERRIDES = (
    "hermes:components=(d+, d, t+, t, he+, he, e, braginskii_collisions, "
    "braginskii_thermal_force, braginskii_ion_viscosity, braginskii_conduction)",
)


def _build_context(input_path: Path, overrides: tuple[str, ...] = ()):
    config = load_bout_input(input_path)
    if overrides:
        config = apply_bout_overrides(config, overrides)
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


def test_collision_transport_field_rhs_matches_full_collision_closure() -> None:
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
        _active_fields,
    ) = _build_context(
        _DTHE_INPUT,
        overrides=_BOUNDED_COLLISION_TRANSPORT_DTHE_OVERRIDES,
    )
    collision_rates, _ionisation_rates, charge_exchange_rates = _rate_inputs(
        config,
        species,
        prepared,
        scalars,
    )

    full_terms = apply_collision_closure(
        config,
        dict(species),
        dict(prepared),
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        collision_rates=collision_rates,
        cx_rates=charge_exchange_rates,
    )
    expected: dict[str, object] = {}
    for name, sp in species.items():
        if sp.has_pressure and sp.pressure_name in layout.field_names:
            expected[sp.pressure_name] = (
                (2.0 / 3.0) * full_terms.energy_source[name]
            )[layout.active_slices]
        if sp.has_momentum and sp.momentum_name in layout.field_names:
            expected[sp.momentum_name] = full_terms.momentum_source[name][
                layout.active_slices
            ]

    actual = fixed_layout_collision_transport_field_rhs_from_prepared(
        config,
        species=species,
        prepared=prepared,
        layout=layout,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        collision_rates=collision_rates,
        charge_exchange_rates=charge_exchange_rates,
    )

    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(
            np.asarray(actual[name]),
            np.asarray(expected[name]),
            rtol=1.0e-9,
            atol=1.0e-10,
            err_msg=f"collision transport RHS mismatch for {name}",
        )


def test_collision_transport_field_rhs_is_jvp_transformable() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    (
        config,
        mesh,
        metrics,
        scalars,
        _runtime_model,
        species,
        prepared,
        _ion_boundary,
        layout,
        _state,
        _active_fields,
    ) = _build_context(
        _DTHE_INPUT,
        overrides=_BOUNDED_COLLISION_TRANSPORT_DTHE_OVERRIDES,
    )
    collision_rates, _ionisation_rates, charge_exchange_rates = _rate_inputs(
        config,
        species,
        prepared,
        scalars,
    )

    def qoi(scale):
        scaled_prepared = {
            name: state.__class__(
                density=jnp.asarray(state.density, dtype=jnp.float64),
                pressure=jnp.asarray(state.pressure, dtype=jnp.float64) * scale,
                temperature=jnp.asarray(state.temperature, dtype=jnp.float64) * scale,
                velocity=jnp.asarray(state.velocity, dtype=jnp.float64),
                momentum=jnp.asarray(state.momentum, dtype=jnp.float64),
                momentum_error=jnp.asarray(
                    state.momentum_error,
                    dtype=jnp.float64,
                ),
            )
            for name, state in prepared.items()
        }
        rhs = fixed_layout_collision_transport_field_rhs_from_prepared(
            config,
            species=species,
            prepared=scaled_prepared,
            layout=layout,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=scalars,
            collision_rates={
                key: jnp.asarray(value, dtype=jnp.float64)
                for key, value in collision_rates.items()
            },
            charge_exchange_rates={
                key: jnp.asarray(value, dtype=jnp.float64)
                for key, value in charge_exchange_rates.items()
            },
        )
        return sum(jnp.sum(jnp.asarray(value, dtype=jnp.float64)) for value in rhs.values())

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
        rtol=1.0e-4,
        atol=1.0e-7,
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


def test_active_source_assembly_inserts_neutral_sources_once() -> None:
    (
        _config,
        mesh,
        metrics,
        _scalars,
        runtime_model,
        species,
        prepared,
        ion_boundary,
        layout,
        _state,
        _active_fields,
    ) = _build_context(_HYDROGEN_INPUT)
    active_shape = layout.active_shape
    density_source = np.linspace(0.1, 0.3, np.prod(active_shape), dtype=np.float64).reshape(active_shape)
    pressure_source = np.linspace(0.2, 0.5, np.prod(active_shape), dtype=np.float64).reshape(active_shape)
    momentum_source = np.linspace(-0.4, 0.4, np.prod(active_shape), dtype=np.float64).reshape(active_shape)

    baseline = assemble_fixed_layout_recycling_field_rhs_from_sources(
        source_field_rhs={},
        layout=layout,
        species=species,
        prepared=prepared,
        ion_velocity=ion_boundary.velocity,
        mesh=mesh,
        metrics=metrics,
        explicit_pressure_sources=runtime_model.explicit_pressure_sources,
    )
    sourced = assemble_fixed_layout_recycling_field_rhs_from_sources(
        source_field_rhs={
            "Nd": density_source,
            "Pd": pressure_source,
            "NVd": momentum_source,
        },
        layout=layout,
        species=species,
        prepared=prepared,
        ion_velocity=ion_boundary.velocity,
        mesh=mesh,
        metrics=metrics,
        explicit_pressure_sources=runtime_model.explicit_pressure_sources,
    )

    np.testing.assert_allclose(
        np.asarray(sourced["Nd"]) - np.asarray(baseline["Nd"]),
        density_source,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(sourced["Pd"]) - np.asarray(baseline["Pd"]),
        pressure_source,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(sourced["NVd"]) - np.asarray(baseline["NVd"]),
        momentum_source,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_active_source_assembly_deferred_sources_match_full_scatter() -> None:
    (
        _config,
        mesh,
        metrics,
        _scalars,
        runtime_model,
        species,
        prepared,
        ion_boundary,
        layout,
        _state,
        _active_fields,
    ) = _build_context(_HYDROGEN_INPUT)
    active_shape = layout.active_shape
    source_count = int(np.prod(active_shape))
    source_field_rhs = {
        "Nd": np.linspace(0.1, 0.3, source_count, dtype=np.float64).reshape(active_shape),
        "Pd": np.linspace(0.2, 0.5, source_count, dtype=np.float64).reshape(active_shape),
        "NVd": np.linspace(-0.4, 0.4, source_count, dtype=np.float64).reshape(active_shape),
    }

    deferred = assemble_fixed_layout_recycling_field_rhs_from_sources(
        source_field_rhs=source_field_rhs,
        layout=layout,
        species=species,
        prepared=prepared,
        ion_velocity=ion_boundary.velocity,
        mesh=mesh,
        metrics=metrics,
        explicit_pressure_sources=runtime_model.explicit_pressure_sources,
        defer_active_source_scatter=True,
    )
    scattered = assemble_fixed_layout_recycling_field_rhs_from_sources(
        source_field_rhs=source_field_rhs,
        layout=layout,
        species=species,
        prepared=prepared,
        ion_velocity=ion_boundary.velocity,
        mesh=mesh,
        metrics=metrics,
        explicit_pressure_sources=runtime_model.explicit_pressure_sources,
        defer_active_source_scatter=False,
    )

    assert deferred.keys() == scattered.keys()
    for name in deferred:
        np.testing.assert_allclose(
            np.asarray(deferred[name]),
            np.asarray(scattered[name]),
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_active_source_assembly_active_transport_matches_full_transport_fallback() -> None:
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
    ) = _build_context(_DTHE_INPUT)
    collision_rates, ionisation_rates, charge_exchange_rates = _rate_inputs(
        config,
        species,
        prepared,
        scalars,
    )
    source_field_rhs = fixed_layout_recycling_source_field_rhs_from_active_fields(
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

    active_transport = assemble_fixed_layout_recycling_field_rhs_from_sources(
        source_field_rhs=source_field_rhs,
        layout=layout,
        species=species,
        prepared=prepared,
        ion_velocity=ion_boundary.velocity,
        mesh=mesh,
        metrics=metrics,
        explicit_pressure_sources=runtime_model.explicit_pressure_sources,
        use_active_transport_terms=True,
    )
    full_transport = assemble_fixed_layout_recycling_field_rhs_from_sources(
        source_field_rhs=source_field_rhs,
        layout=layout,
        species=species,
        prepared=prepared,
        ion_velocity=ion_boundary.velocity,
        mesh=mesh,
        metrics=metrics,
        explicit_pressure_sources=runtime_model.explicit_pressure_sources,
        use_active_transport_terms=False,
    )

    assert active_transport.keys() == full_transport.keys()
    for name in active_transport:
        np.testing.assert_allclose(
            np.asarray(active_transport[name]),
            np.asarray(full_transport[name]),
            rtol=1.0e-9,
            atol=1.0e-10,
            err_msg=f"active transport mismatch for {name}",
        )


def test_active_source_assembly_is_jvp_transformable() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    (
        _config,
        mesh,
        metrics,
        _scalars,
        runtime_model,
        species,
        prepared,
        ion_boundary,
        layout,
        _state,
        _active_fields,
    ) = _build_context(_HYDROGEN_INPUT)
    active_shape = layout.active_shape
    base_density_source = jnp.linspace(0.1, 0.3, int(np.prod(active_shape)), dtype=jnp.float64).reshape(active_shape)
    base_pressure_source = jnp.linspace(0.2, 0.5, int(np.prod(active_shape)), dtype=jnp.float64).reshape(active_shape)
    base_momentum_source = jnp.linspace(-0.4, 0.4, int(np.prod(active_shape)), dtype=jnp.float64).reshape(active_shape)

    def qoi(scale):
        assembled = assemble_fixed_layout_recycling_field_rhs_from_sources(
            source_field_rhs={
                "Nd": base_density_source * scale,
                "Pd": base_pressure_source * scale,
                "NVd": base_momentum_source * scale,
            },
            layout=layout,
            species=species,
            prepared=prepared,
            ion_velocity=ion_boundary.velocity,
            mesh=mesh,
            metrics=metrics,
            explicit_pressure_sources=runtime_model.explicit_pressure_sources,
        )
        return (
            jnp.sum(assembled["Nd"])
            + 0.1 * jnp.sum(assembled["Pd"])
            + 0.01 * jnp.sum(assembled["NVd"])
        )

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


def test_promoted_active_source_rhs_matches_bounded_full_rhs() -> None:
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        _species,
        _prepared,
        _ion_boundary,
        layout,
        state,
        _active_fields,
    ) = _build_context(_DTHE_INPUT, overrides=_BOUNDED_PROMOTED_DTHE_OVERRIDES)

    assert not runtime_model.controllers
    full_rhs = _build_fixed_full_field_recycling_rhs(
        config,
        runtime_model=runtime_model,
        layout=layout,
        base_feedback_integrals={},
        feedback_previous_errors=None,
        feedback_timestep=None,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    promoted_rhs = _build_promoted_active_source_recycling_rhs(
        config,
        runtime_model=runtime_model,
        layout=layout,
        base_feedback_integrals={},
        feedback_previous_errors=None,
        feedback_timestep=None,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )

    expected = full_rhs(state)
    actual = promoted_rhs(state)

    assert tuple(layout.feedback_names) == ()
    for name, expected_value, actual_value in zip(
        layout.field_names,
        expected.field_values,
        actual.field_values,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual_value),
            np.asarray(expected_value),
            rtol=1.0e-9,
            atol=1.0e-10,
            err_msg=f"promoted active source RHS mismatch for {name}",
        )


def test_promoted_active_source_rhs_matches_bounded_feedback_full_rhs() -> None:
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        _species,
        _prepared,
        _ion_boundary,
        layout,
        state,
        _active_fields,
    ) = _build_context(
        _DTHE_INPUT,
        overrides=_BOUNDED_PROMOTED_DTHE_FEEDBACK_OVERRIDES,
    )
    feedback_integrals = {name: 0.0 for name in runtime_model.feedback_names}

    assert runtime_model.controllers
    assert tuple(layout.feedback_names) == runtime_model.feedback_names
    full_rhs = _build_fixed_full_field_recycling_rhs(
        config,
        runtime_model=runtime_model,
        layout=layout,
        base_feedback_integrals=feedback_integrals,
        feedback_previous_errors=None,
        feedback_timestep=None,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    promoted_rhs = _build_promoted_active_source_recycling_rhs(
        config,
        runtime_model=runtime_model,
        layout=layout,
        base_feedback_integrals=feedback_integrals,
        feedback_previous_errors=None,
        feedback_timestep=None,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )

    expected = full_rhs(state)
    actual = promoted_rhs(state)

    for name, expected_value, actual_value in zip(
        layout.field_names,
        expected.field_values,
        actual.field_values,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual_value),
            np.asarray(expected_value),
            rtol=1.0e-9,
            atol=1.0e-10,
            err_msg=f"promoted active feedback RHS mismatch for {name}",
        )
    np.testing.assert_allclose(
        np.asarray(actual.feedback_values),
        np.asarray(expected.feedback_values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_promoted_active_source_rhs_matches_feedback_predictor_full_rhs() -> None:
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        _species,
        _prepared,
        _ion_boundary,
        layout,
        state,
        _active_fields,
    ) = _build_context(
        _DTHE_INPUT,
        overrides=_BOUNDED_PROMOTED_DTHE_FEEDBACK_OVERRIDES,
    )
    fields = _build_recycling_state_fields(runtime_model)
    feedback_integrals = {name: 0.25 for name in runtime_model.feedback_names}
    previous_errors = _current_feedback_errors(
        fields,
        controllers=runtime_model.controllers,
        mesh=mesh,
    )

    full_rhs = _build_fixed_full_field_recycling_rhs(
        config,
        runtime_model=runtime_model,
        layout=layout,
        base_feedback_integrals=feedback_integrals,
        feedback_previous_errors=previous_errors,
        feedback_timestep=2.5e-5,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    promoted_rhs = _build_promoted_active_source_recycling_rhs(
        config,
        runtime_model=runtime_model,
        layout=layout,
        base_feedback_integrals=feedback_integrals,
        feedback_previous_errors=previous_errors,
        feedback_timestep=2.5e-5,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )

    expected = full_rhs(state)
    actual = promoted_rhs(state)

    for name, expected_value, actual_value in zip(
        layout.field_names,
        expected.field_values,
        actual.field_values,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual_value),
            np.asarray(expected_value),
            rtol=1.0e-9,
            atol=1.0e-10,
            err_msg=f"promoted active feedback predictor mismatch for {name}",
        )
    np.testing.assert_allclose(
        np.asarray(actual.feedback_values),
        np.asarray(expected.feedback_values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_promoted_active_source_rhs_matches_full_dthe_collision_transport_fixture() -> None:
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        _species,
        _prepared,
        _ion_boundary,
        layout,
        state,
        _active_fields,
    ) = _build_context(_DTHE_INPUT)
    feedback_integrals = {name: 0.0 for name in runtime_model.feedback_names}

    full_rhs = _build_fixed_full_field_recycling_rhs(
        config,
        runtime_model=runtime_model,
        layout=layout,
        base_feedback_integrals=feedback_integrals,
        feedback_previous_errors=None,
        feedback_timestep=None,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    promoted_rhs = _build_promoted_active_source_recycling_rhs(
        config,
        runtime_model=runtime_model,
        layout=layout,
        base_feedback_integrals=feedback_integrals,
        feedback_previous_errors=None,
        feedback_timestep=None,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )

    expected = full_rhs(state)
    actual = promoted_rhs(state)

    for name, expected_value, actual_value in zip(
        layout.field_names,
        expected.field_values,
        actual.field_values,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual_value),
            np.asarray(expected_value),
            rtol=1.0e-9,
            atol=1.0e-10,
            err_msg=f"promoted active full D/T/He RHS mismatch for {name}",
        )
    np.testing.assert_allclose(
        np.asarray(actual.feedback_values),
        np.asarray(expected.feedback_values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_promoted_active_source_backward_euler_residual_is_jvp_transformable() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        _species,
        _prepared,
        _ion_boundary,
        _layout,
        _state,
        _active_fields,
    ) = _build_context(_DTHE_INPUT, overrides=_BOUNDED_PROMOTED_DTHE_OVERRIDES)
    fields = _build_recycling_state_fields(runtime_model)
    context = build_recycling_1d_backward_euler_residual_context(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals={},
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=1.0e-5,
        rhs_backend="promoted_active_sources",
    )
    packed_state = jnp.asarray(context.packed_previous_state, dtype=jnp.float64)

    def qoi(scale):
        residual = context.residual(packed_state * scale)
        return jnp.sum(residual)

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
        rtol=1.0e-4,
        atol=1.0e-7,
    )


def test_promoted_active_source_feedback_residual_is_jvp_transformable() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        _species,
        _prepared,
        _ion_boundary,
        _layout,
        _state,
        _active_fields,
    ) = _build_context(
        _DTHE_INPUT,
        overrides=_BOUNDED_PROMOTED_DTHE_FEEDBACK_OVERRIDES,
    )
    fields = _build_recycling_state_fields(runtime_model)
    context = build_recycling_1d_backward_euler_residual_context(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals={name: 0.0 for name in runtime_model.feedback_names},
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=1.0e-5,
        evolve_feedback_integrals=True,
        rhs_backend="promoted_active_sources",
    )
    packed_state = jnp.asarray(context.packed_previous_state, dtype=jnp.float64)

    def qoi(scale):
        residual = context.residual(packed_state * scale)
        return jnp.sum(residual)

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
        rtol=1.0e-4,
        atol=1.0e-7,
    )


def test_promoted_active_source_bdf2_residual_is_jvp_transformable() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        _species,
        _prepared,
        _ion_boundary,
        _layout,
        _state,
        _active_fields,
    ) = _build_context(_DTHE_INPUT, overrides=_BOUNDED_PROMOTED_DTHE_OVERRIDES)
    fields = _build_recycling_state_fields(runtime_model)
    context = build_recycling_1d_bdf2_residual_context(
        config,
        fields,
        fields,
        runtime_model=runtime_model,
        feedback_integrals={},
        previous_feedback_integrals={},
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=1.0e-5,
        rhs_backend="promoted_active_sources",
    )
    packed_state = jnp.asarray(context.packed_previous_state, dtype=jnp.float64)

    def qoi(scale):
        residual = context.residual(packed_state * scale)
        return jnp.sum(residual)

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
        rtol=1.0e-4,
        atol=1.0e-7,
    )
