from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_collisions import compute_collision_frequencies
from jax_drb.native.recycling_1d import (
    _build_recycling_runtime_model,
    _build_recycling_state_fields,
)
from jax_drb.native.recycling_fixed_residual import (
    build_fixed_array_rhs,
    fixed_state_from_fields,
)
from jax_drb.native.recycling_layout import build_recycling_packed_state_layout
from jax_drb.native.recycling_neutral_diffusion import (
    apply_neutral_parallel_diffusion,
    configured_component_names,
    fixed_layout_neutral_parallel_diffusion_field_rhs_from_active_fields,
)
from jax_drb.native.recycling_reactions import (
    neutral_charge_exchange_collision_rates,
    neutral_ionisation_collision_rates,
)
from jax_drb.native.recycling_setup import build_species_field_overrider, initialize_species
from jax_drb.native.recycling_state import prepare_species_state
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars


_INPUT_1D = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")
_INPUT_1D_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "reference-root"
    / "tests"
    / "integrated"
    / "1D-recycling"
    / "data"
    / "BOUT.inp"
)


class _ComponentConfig:
    def __init__(self, sections: dict[str, dict[str, object]]) -> None:
        self._sections = sections

    def has_section(self, section: str) -> bool:
        return section in self._sections

    def has_option(self, section: str, key: str) -> bool:
        return key in self._sections.get(section, {})

    def parsed(self, section: str, key: str) -> object:
        return self._sections[section][key]


def _build_prepared_case(*, overrides: tuple[str, ...] = ()) -> tuple[object, object, object, dict[str, object], dict[str, object]]:
    config = apply_bout_overrides(load_bout_input(_input_1d_path()), overrides)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    species = initialize_species(config, mesh=mesh, dataset_scalars=scalars)
    prepared = {name: prepare_species_state(sp, mesh=mesh) for name, sp in species.items()}
    return config, mesh, metrics, species, prepared


def _input_1d_path() -> Path:
    if _INPUT_1D.exists():
        return _INPUT_1D
    if _INPUT_1D_FIXTURE.exists():
        return _INPUT_1D_FIXTURE
    raise AssertionError("1D recycling fixture deck is missing")


def _build_runtime_case(*, overrides: tuple[str, ...] = ()):
    config = apply_bout_overrides(load_bout_input(_input_1d_path()), overrides)
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
    prepared = {name: prepare_species_state(sp, mesh=mesh) for name, sp in species.items()}
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
        layout,
        state,
        active_fields,
    )


def test_configured_component_names_prefers_model_then_hermes_and_handles_missing_sections() -> None:
    assert configured_component_names(_ComponentConfig({})) == ()
    assert configured_component_names(_ComponentConfig({"hermes": {"components": "neutral_parallel_diffusion"}})) == (
        "neutral_parallel_diffusion",
    )
    assert configured_component_names(
        _ComponentConfig(
            {
                "model": {"components": ("density", " neutral_parallel_diffusion ")},
                "hermes": {"components": "ignored"},
            }
        )
    ) == ("density", "neutral_parallel_diffusion")


def test_neutral_parallel_diffusion_returns_zero_when_component_disabled() -> None:
    config, mesh, metrics, species, prepared = _build_prepared_case()

    terms = apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(RunConfiguration.from_config(config)),
    )

    assert all(np.allclose(value, 0.0) for value in terms.density_source.values())
    assert all(np.allclose(value, 0.0) for value in terms.energy_source.values())
    assert all(np.allclose(value, 0.0) for value in terms.momentum_source.values())
    assert terms.diagnostics == {}


def test_neutral_parallel_diffusion_returns_zero_when_dneut_is_not_positive() -> None:
    config, mesh, metrics, species, prepared = _build_prepared_case(
        overrides=(
            "model:components=neutral_parallel_diffusion",
            "neutral_parallel_diffusion:dneut=0.0",
        )
    )

    terms = apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(RunConfiguration.from_config(config)),
    )

    assert all(np.allclose(value, 0.0) for value in terms.density_source.values())
    assert all(np.allclose(value, 0.0) for value in terms.energy_source.values())
    assert all(np.allclose(value, 0.0) for value in terms.momentum_source.values())
    assert terms.diagnostics == {}


def test_neutral_parallel_diffusion_raises_on_unsupported_mode() -> None:
    config, mesh, metrics, species, prepared = _build_prepared_case(
        overrides=(
            "model:components=neutral_parallel_diffusion",
            "neutral_parallel_diffusion:dneut=1.0",
            "neutral_parallel_diffusion:diffusion_collisions_mode=unsupported_mode",
        )
    )

    with pytest.raises(NotImplementedError, match="Unsupported neutral_parallel_diffusion"):
        apply_neutral_parallel_diffusion(
            config,
            species=species,
            prepared=prepared,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=resolved_dataset_scalars(RunConfiguration.from_config(config)),
        )


def test_neutral_parallel_diffusion_diagnose_emits_profile_fields() -> None:
    config, mesh, metrics, species, prepared = _build_prepared_case(
        overrides=(
            "model:components=neutral_parallel_diffusion",
            "neutral_parallel_diffusion:dneut=1.0",
            "neutral_parallel_diffusion:diagnose=true",
        )
    )

    terms = apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(RunConfiguration.from_config(config)),
    )

    assert "Dd_Dpar" in terms.diagnostics
    assert "Sd_Dpar" in terms.diagnostics
    assert "Ed_Dpar" in terms.diagnostics
    assert "Fd_Dpar" in terms.diagnostics
    assert np.isfinite(terms.diagnostics["Dd_Dpar"]).all()


def test_neutral_parallel_diffusion_multispecies_mode_produces_finite_terms() -> None:
    config, mesh, metrics, species, prepared = _build_prepared_case(
        overrides=(
            "model:components=neutral_parallel_diffusion",
            "neutral_parallel_diffusion:dneut=1.0",
            "neutral_parallel_diffusion:diffusion_collisions_mode=multispecies",
            "neutral_parallel_diffusion:diagnose=true",
        )
    )
    prepared_d = prepared["d"]
    density = np.asarray(prepared_d.density, dtype=np.float64, copy=True)
    pressure = np.asarray(prepared_d.pressure, dtype=np.float64, copy=True)
    temperature = np.asarray(prepared_d.temperature, dtype=np.float64, copy=True)
    momentum = np.asarray(prepared_d.momentum, dtype=np.float64, copy=True)
    velocity = np.asarray(prepared_d.velocity, dtype=np.float64, copy=True)
    density[:, mesh.ystart : mesh.yend + 1, :] *= np.linspace(1.0, 1.3, mesh.yend - mesh.ystart + 1)[None, :, None]
    pressure[:, mesh.ystart : mesh.yend + 1, :] *= np.linspace(1.0, 1.6, mesh.yend - mesh.ystart + 1)[None, :, None]
    temperature = pressure / np.maximum(density, 1.0e-8)
    prepared["d"] = replace(
        prepared_d,
        density=density,
        pressure=pressure,
        temperature=temperature,
        momentum=momentum,
        velocity=velocity,
    )

    terms = apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(RunConfiguration.from_config(config)),
    )

    assert np.isfinite(terms.density_source["d"]).all()
    assert np.isfinite(terms.energy_source["d"]).all()
    assert np.isfinite(terms.momentum_source["d"]).all()
    assert float(np.nanmax(np.abs(terms.density_source["d"]))) > 0.0


def test_neutral_parallel_diffusion_accepts_precomputed_rates() -> None:
    config, mesh, metrics, species, prepared = _build_prepared_case(
        overrides=(
            "model:components=neutral_parallel_diffusion",
            "neutral_parallel_diffusion:dneut=1.0",
            "neutral_parallel_diffusion:diffusion_collisions_mode=multispecies",
        )
    )
    scalars = resolved_dataset_scalars(RunConfiguration.from_config(config))
    collision_rates = compute_collision_frequencies(config, species, prepared, dataset_scalars=scalars)
    ionisation_rates = neutral_ionisation_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=scalars,
    )
    cx_rates = neutral_charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=scalars,
    )

    baseline = apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    reused = apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        collision_rates=collision_rates,
        ionisation_rates=ionisation_rates,
        charge_exchange_rates=cx_rates,
    )

    for name in baseline.density_source:
        np.testing.assert_allclose(reused.density_source[name], baseline.density_source[name])
    for name in baseline.energy_source:
        np.testing.assert_allclose(reused.energy_source[name], baseline.energy_source[name])
    for name in baseline.momentum_source:
        np.testing.assert_allclose(reused.momentum_source[name], baseline.momentum_source[name])


def test_neutral_parallel_diffusion_is_jax_jvp_transformable_with_precomputed_rates() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    config, mesh, metrics, species, prepared = _build_prepared_case(
        overrides=(
            "model:components=neutral_parallel_diffusion",
            "neutral_parallel_diffusion:dneut=1.0",
            "neutral_parallel_diffusion:diffusion_collisions_mode=multispecies",
            "neutral_parallel_diffusion:diagnose=true",
        )
    )
    scalars = resolved_dataset_scalars(RunConfiguration.from_config(config))
    collision_rates = {
        key: jnp.asarray(value, dtype=jnp.float64)
        for key, value in compute_collision_frequencies(config, species, prepared, dataset_scalars=scalars).items()
    }
    ionisation_rates = {
        key: jnp.asarray(value, dtype=jnp.float64)
        for key, value in neutral_ionisation_collision_rates(
            config,
            species=species,
            prepared=prepared,
            dataset_scalars=scalars,
        ).items()
    }
    cx_rates = {
        key: jnp.asarray(value, dtype=jnp.float64)
        for key, value in neutral_charge_exchange_collision_rates(
            config,
            species=species,
            prepared=prepared,
            dataset_scalars=scalars,
        ).items()
    }

    def qoi(scale):
        transformed = {}
        for name, state in prepared.items():
            density = jnp.asarray(state.density, dtype=jnp.float64)
            pressure = jnp.asarray(state.pressure, dtype=jnp.float64)
            if name == "d":
                pressure = pressure * scale
            transformed[name] = replace(
                state,
                density=density,
                pressure=pressure,
                temperature=pressure / jnp.maximum(density, 1.0e-8),
                momentum=jnp.asarray(state.momentum, dtype=jnp.float64),
                velocity=jnp.asarray(state.velocity, dtype=jnp.float64),
            )
        terms = apply_neutral_parallel_diffusion(
            config,
            species=species,
            prepared=transformed,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=scalars,
            collision_rates=collision_rates,
            ionisation_rates=ionisation_rates,
            charge_exchange_rates=cx_rates,
        )
        return jnp.sum(terms.density_source["d"]) + 0.1 * jnp.sum(terms.energy_source["d"])

    value, tangent = jax.jvp(qoi, (jnp.array(1.0),), (jnp.array(1.0),))
    step = 1.0e-5
    finite_difference = (qoi(jnp.array(1.0 + step)) - qoi(jnp.array(1.0 - step))) / (2.0 * step)

    assert np.isfinite(float(value))
    assert np.isfinite(float(tangent))
    np.testing.assert_allclose(float(tangent), float(finite_difference), rtol=1.0e-5, atol=1.0e-8)


def test_fixed_layout_neutral_parallel_diffusion_matches_full_field_active_slice() -> None:
    (
        config,
        mesh,
        metrics,
        scalars,
        runtime_model,
        species,
        prepared,
        layout,
        _state,
        active_fields,
    ) = _build_runtime_case()
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
    cx_rates = neutral_charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=scalars,
    )

    full_terms = apply_neutral_parallel_diffusion(
        config,
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        collision_rates=collision_rates,
        ionisation_rates=ionisation_rates,
        charge_exchange_rates=cx_rates,
    )
    active_rhs = fixed_layout_neutral_parallel_diffusion_field_rhs_from_active_fields(
        config,
        active_fields=active_fields,
        layout=layout,
        species_templates=runtime_model.species_templates,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        collision_rates=collision_rates,
        ionisation_rates=ionisation_rates,
        charge_exchange_rates=cx_rates,
    )

    active_slices = layout.active_slices
    layout_fields = set(layout.field_names)
    for name, sp in species.items():
        if sp.density_name in layout_fields:
            np.testing.assert_allclose(
                np.asarray(active_rhs[sp.density_name]),
                np.asarray(full_terms.density_source[name][active_slices]),
                rtol=1.0e-9,
                atol=1.0e-10,
            )
        if sp.has_pressure and sp.pressure_name in layout_fields:
            np.testing.assert_allclose(
                np.asarray(active_rhs[sp.pressure_name]),
                (2.0 / 3.0)
                * np.asarray(full_terms.energy_source[name][active_slices]),
                rtol=1.0e-9,
                atol=1.0e-10,
            )
        if sp.has_momentum and sp.momentum_name in layout_fields:
            np.testing.assert_allclose(
                np.asarray(active_rhs[sp.momentum_name]),
                np.asarray(full_terms.momentum_source[name][active_slices]),
                rtol=1.0e-9,
                atol=1.0e-10,
            )


def test_fixed_layout_neutral_parallel_diffusion_promotes_to_fixed_array_rhs() -> None:
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
        layout,
        state,
        _active_fields,
    ) = _build_runtime_case()
    collision_rates = {
        key: jnp.asarray(value, dtype=jnp.float64)
        for key, value in compute_collision_frequencies(
            config,
            species,
            prepared,
            dataset_scalars=scalars,
        ).items()
    }
    ionisation_rates = {
        key: jnp.asarray(value, dtype=jnp.float64)
        for key, value in neutral_ionisation_collision_rates(
            config,
            species=species,
            prepared=prepared,
            dataset_scalars=scalars,
        ).items()
    }
    cx_rates = {
        key: jnp.asarray(value, dtype=jnp.float64)
        for key, value in neutral_charge_exchange_collision_rates(
            config,
            species=species,
            prepared=prepared,
            dataset_scalars=scalars,
        ).items()
    }
    neutral_pressure_index = layout.field_names.index("Pd")

    def field_rhs(active_fields: dict[str, object], _feedback: object) -> dict[str, object]:
        return fixed_layout_neutral_parallel_diffusion_field_rhs_from_active_fields(
            config,
            active_fields=active_fields,
            layout=layout,
            species_templates=runtime_model.species_templates,
            mesh=mesh,
            metrics=metrics,
            dataset_scalars=scalars,
            collision_rates=collision_rates,
            ionisation_rates=ionisation_rates,
            charge_exchange_rates=cx_rates,
        )

    fixed_rhs = build_fixed_array_rhs(field_rhs, layout=layout)

    def qoi(scale):
        scaled_fields = list(state.field_values)
        scaled_fields[neutral_pressure_index] = (
            jnp.asarray(scaled_fields[neutral_pressure_index], dtype=jnp.float64)
            * scale
        )
        rhs_state = fixed_rhs(
            type(state)(
                field_values=tuple(scaled_fields),
                feedback_values=jnp.asarray(state.feedback_values, dtype=jnp.float64),
            )
        )
        rhs_fields = {
            name: value
            for name, value in zip(layout.field_names, rhs_state.field_values, strict=True)
        }
        return jnp.sum(rhs_fields["Nd"]) + 0.1 * jnp.sum(rhs_fields["Pd"])

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
