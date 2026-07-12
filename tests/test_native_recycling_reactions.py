from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax import grad, jit
import jax.numpy as jnp

from jax_drb.config.boutinp import load_bout_input, parse_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import _initialize_species, _prepare_open_field_states
from jax_drb.native.recycling_reactions import (
    ReactionTerms,
    accumulate_terms,
    amjuel_ionisation,
    amjuel_recombination,
    charge_exchange,
    charge_exchange_collision_rates,
    fixed_layout_dthe_reaction_field_rhs_from_active_fields,
    fixed_layout_dthe_reaction_sources,
    fixed_layout_dthe_reaction_terms_from_active_fields,
    fixed_layout_hydrogen_reaction_sources,
    is_charge_exchange_reaction,
    neutral_charge_exchange_collision_rates,
    neutral_ionisation_collision_rates,
    reaction_sources,
)
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.reference.paths import default_reference_root


_REFERENCE_ROOT = default_reference_root()
_REFERENCE_BASE = _REFERENCE_ROOT if _REFERENCE_ROOT is not None else Path("/nonexistent-reference-root")
_INPUT_1D = _REFERENCE_BASE / "tests/integrated/1D-recycling/data/BOUT.inp"
_INPUT_DTHE = _REFERENCE_BASE / "tests/integrated/1D-recycling-dthe/data/BOUT.inp"


def _small_field(value: float) -> np.ndarray:
    return np.full((1, 1, 1), value, dtype=np.float64)


def _small_reaction_species() -> dict[str, SimpleNamespace]:
    return {
        "d": SimpleNamespace(
            name="d",
            charge=0.0,
            atomic_mass=2.0,
            density=_small_field(0.7),
            pressure=_small_field(0.21),
            momentum=_small_field(0.14),
            density_floor=1.0e-8,
        ),
        "d+": SimpleNamespace(
            name="d+",
            charge=1.0,
            atomic_mass=2.0,
            density=_small_field(0.5),
            pressure=_small_field(0.2),
            momentum=_small_field(0.05),
            density_floor=1.0e-8,
        ),
        "e": SimpleNamespace(
            name="e",
            charge=-1.0,
            atomic_mass=1.0 / 1836.0,
            density=_small_field(0.5),
            pressure=_small_field(0.3),
            momentum=_small_field(0.0),
            density_floor=1.0e-8,
        ),
    }


def _small_prepared(species: dict[str, SimpleNamespace]) -> dict[str, SimpleNamespace]:
    return {
        name: SimpleNamespace(
            density=sp.density,
            pressure=sp.pressure,
            temperature=sp.pressure / np.maximum(sp.density, 1.0e-8),
            momentum=sp.momentum,
        )
        for name, sp in species.items()
    }


def _small_scalars() -> dict[str, float]:
    return {"Nnorm": 1.0e19, "Tnorm": 10.0, "Omega_ci": 1.0e6}


def _stack_species_fields(species: dict[str, SimpleNamespace], names: tuple[str, ...], field_name: str) -> np.ndarray:
    return np.stack([np.asarray(getattr(species[name], field_name), dtype=np.float64) for name in names], axis=0)


def _active_slices(mesh):
    return (
        slice(mesh.xstart, mesh.xend + 1),
        slice(mesh.ystart, mesh.yend + 1),
        slice(None),
    )


def _dthe_active_fields(species, active_slices) -> dict[str, object]:
    atom_names = ("d", "t", "he")
    ion_names = ("d+", "t+", "he+")
    active_fields = {
        species[name].density_name: jnp.asarray(
            species[name].density[active_slices], dtype=jnp.float64
        )
        for name in (*atom_names, *ion_names)
    }
    active_fields.update(
        {
            species[name].pressure_name: jnp.asarray(
                species[name].pressure[active_slices], dtype=jnp.float64
            )
            for name in (*atom_names, *ion_names, "e")
        }
    )
    active_fields.update(
        {
            species[name].momentum_name: jnp.asarray(
                species[name].momentum[active_slices], dtype=jnp.float64
            )
            for name in (*atom_names, *ion_names)
        }
    )
    return active_fields


def _single_species_reaction_config():
    return parse_bout_input(
        """
[reactions]
diagnose = true
type = (
        d + e -> d+ + 2e,
        d+ + e -> d,
        d + d+ -> d+ + d,
       )
"""
    )


def test_accumulate_terms_adds_sources_and_merges_diagnostics() -> None:
    density_source = {"d": _small_field(1.0)}
    energy_source = {"d": _small_field(2.0)}
    momentum_source = {"d": _small_field(3.0)}
    diagnostics: dict[str, np.ndarray] = {"old": _small_field(4.0)}
    result = ReactionTerms(
        density_source={"d": _small_field(0.5)},
        energy_source={"d": _small_field(1.5)},
        momentum_source={"d": _small_field(2.5)},
        diagnostics={"new": _small_field(5.0)},
    )

    accumulate_terms(result, density_source, energy_source, momentum_source, diagnostics)

    np.testing.assert_allclose(density_source["d"], _small_field(1.5))
    np.testing.assert_allclose(energy_source["d"], _small_field(3.5))
    np.testing.assert_allclose(momentum_source["d"], _small_field(5.5))
    assert set(diagnostics) == {"old", "new"}


def test_is_charge_exchange_reaction_identifies_expected_pairs() -> None:
    assert is_charge_exchange_reaction(("d", "d+"), ("d+", "d"))
    assert is_charge_exchange_reaction(("d", "t+"), ("d+", "t"))
    assert not is_charge_exchange_reaction(("d", "e"), ("d+", "2e"))
    assert not is_charge_exchange_reaction(("d",), ("d+",))


def test_reaction_sources_handles_missing_and_malformed_reaction_blocks() -> None:
    species = _small_reaction_species()
    electron_density = _small_field(0.5)
    no_reactions = reaction_sources(
        parse_bout_input(""),
        species=species,
        electron_density=electron_density,
        dataset_scalars=_small_scalars(),
    )
    malformed = reaction_sources(
        parse_bout_input(
            """
[reactions]
type = malformed reaction
"""
        ),
        species=species,
        electron_density=electron_density,
        dataset_scalars=_small_scalars(),
    )

    assert no_reactions.diagnostics == {}
    assert malformed.diagnostics == {}
    assert all(np.allclose(value, 0.0) for value in no_reactions.density_source.values())


def test_reaction_sources_without_diagnostics_suppresses_dictionary_diagnostics() -> None:
    species = _small_reaction_species()
    electron_density = _small_field(0.5)
    scalars = _small_scalars()

    with_diagnostics = reaction_sources(
        _single_species_reaction_config(),
        species=species,
        electron_density=electron_density,
        dataset_scalars=scalars,
    )
    without_diagnostics = reaction_sources(
        _single_species_reaction_config(),
        species=species,
        electron_density=electron_density,
        dataset_scalars=scalars,
        include_diagnostics=False,
    )

    assert with_diagnostics.diagnostics
    assert without_diagnostics.diagnostics == {}
    for name in species:
        np.testing.assert_allclose(without_diagnostics.density_source[name], with_diagnostics.density_source[name])
        np.testing.assert_allclose(without_diagnostics.energy_source[name], with_diagnostics.energy_source[name])
        np.testing.assert_allclose(without_diagnostics.momentum_source[name], with_diagnostics.momentum_source[name])


def test_direct_reaction_wrappers_emit_expected_diagnostics() -> None:
    species = _small_reaction_species()
    electron_density = _small_field(0.5)
    scalars = _small_scalars()

    ionisation = amjuel_ionisation("d", "d+", species=species, electron_density=electron_density, dataset_scalars=scalars)
    recombination = amjuel_recombination("d", "d+", species=species, electron_density=electron_density, dataset_scalars=scalars)
    cx = charge_exchange(
        "d",
        "d+",
        "d",
        "d+",
        config=parse_bout_input(""),
        species=species,
        dataset_scalars=scalars,
    )

    assert "Sd+_iz" in ionisation.diagnostics
    assert "Sd+_rec" in recombination.diagnostics
    assert "Edd+_cx" in cx.diagnostics
    assert "Fdd+_cx" in cx.diagnostics
    assert np.isfinite(ionisation.diagnostics["Sd+_iz"]).all()
    assert np.isfinite(recombination.diagnostics["Sd+_rec"]).all()


def test_reaction_sources_uses_openadas_for_neon_ionisation_and_recombination() -> None:
    density = np.full((2, 3, 1), 0.2, dtype=np.float64)
    ion_density = np.full((2, 3, 1), 0.4, dtype=np.float64)
    electron_pressure = np.full((2, 3, 1), 1.2, dtype=np.float64)
    species = {
        "ne": SimpleNamespace(
            name="ne",
            charge=0.0,
            atomic_mass=20.0,
            density=density,
            pressure=0.5 * density,
            momentum=np.zeros_like(density),
            density_floor=1.0e-8,
        ),
        "ne+": SimpleNamespace(
            name="ne+",
            charge=1.0,
            atomic_mass=20.0,
            density=ion_density,
            pressure=0.8 * ion_density,
            momentum=np.zeros_like(ion_density),
            density_floor=1.0e-8,
        ),
        "e": SimpleNamespace(
            name="e",
            charge=-1.0,
            atomic_mass=1.0 / 1836.0,
            density=ion_density,
            pressure=electron_pressure,
            momentum=np.zeros_like(ion_density),
            density_floor=1.0e-8,
        ),
    }
    scalars = _small_scalars()

    ionisation = reaction_sources(
        parse_bout_input(
            """
[reactions]
type = ne + e -> ne+ + 2e
"""
        ),
        species=species,
        electron_density=ion_density,
        dataset_scalars=scalars,
    )
    recombination = reaction_sources(
        parse_bout_input(
            """
[reactions]
type = ne+ + e -> ne
"""
        ),
        species=species,
        electron_density=ion_density,
        dataset_scalars=scalars,
    )

    assert "Sne+_iz" in ionisation.diagnostics
    assert "Sne+_rec" in recombination.diagnostics
    assert np.isfinite(ionisation.diagnostics["Sne+_iz"]).all()
    assert np.isfinite(recombination.diagnostics["Sne+_rec"]).all()


def test_reaction_sources_include_cross_isotope_charge_exchange_diagnostics() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_INPUT_DTHE)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    species = _initialize_species(config, mesh=mesh)
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    electron_density = np.zeros_like(species["d+"].density, dtype=np.float64)
    for ion in ions:
        electron_density = electron_density + ion.charge * ion.density

    terms = reaction_sources(
        config,
        species=species,
        electron_density=electron_density,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    assert "Edt+_cx" in terms.diagnostics
    assert "Etd+_cx" in terms.diagnostics
    assert "Sdt+_cx" in terms.diagnostics
    assert "Std+_cx" in terms.diagnostics


def test_reaction_sources_preserve_jax_backend_and_support_grad() -> None:
    config = _single_species_reaction_config()
    scalars = _small_scalars()

    def species_for_scale(scale):
        shape = (1, 1, 1)
        atom_density = jnp.full(shape, 0.7, dtype=jnp.float64) * scale
        ion_density = jnp.full(shape, 0.5, dtype=jnp.float64)
        return {
            "d": SimpleNamespace(
                name="d",
                charge=0.0,
                atomic_mass=2.0,
                density=atom_density,
                pressure=jnp.full(shape, 0.21, dtype=jnp.float64),
                momentum=jnp.full(shape, 0.14, dtype=jnp.float64),
                density_floor=1.0e-8,
            ),
            "d+": SimpleNamespace(
                name="d+",
                charge=1.0,
                atomic_mass=2.0,
                density=ion_density,
                pressure=jnp.full(shape, 0.2, dtype=jnp.float64),
                momentum=jnp.full(shape, 0.05, dtype=jnp.float64),
                density_floor=1.0e-8,
            ),
            "e": SimpleNamespace(
                name="e",
                charge=-1.0,
                atomic_mass=1.0 / 1836.0,
                density=ion_density,
                pressure=jnp.full(shape, 0.3, dtype=jnp.float64),
                momentum=jnp.zeros(shape, dtype=jnp.float64),
                density_floor=1.0e-8,
            ),
        }

    def objective(scale):
        species = species_for_scale(scale)
        terms = reaction_sources(
            config,
            species=species,
            electron_density=species["d+"].density,
            dataset_scalars=scalars,
        )
        return (
            jnp.sum(terms.density_source["d+"])
            + jnp.sum(terms.energy_source["e"])
            + jnp.sum(terms.momentum_source["d+"])
        )

    value = jit(objective)(jnp.array(1.0, dtype=jnp.float64))
    derivative = grad(objective)(jnp.array(1.0, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))


def test_fixed_layout_hydrogen_reaction_sources_match_dictionary_path() -> None:
    species = _small_reaction_species()
    config = _single_species_reaction_config()
    scalars = _small_scalars()
    electron_density = species["d+"].density

    dictionary_terms = reaction_sources(
        config,
        species=species,
        electron_density=electron_density,
        dataset_scalars=scalars,
    )
    fixed_terms = fixed_layout_hydrogen_reaction_sources(
        atom_density=species["d"].density,
        atom_pressure=species["d"].pressure,
        atom_momentum=species["d"].momentum,
        ion_density=species["d+"].density,
        ion_pressure=species["d+"].pressure,
        ion_momentum=species["d+"].momentum,
        electron_density=electron_density,
        electron_pressure=species["e"].pressure,
        dataset_scalars=scalars,
    )

    np.testing.assert_allclose(fixed_terms.atom_density_source, dictionary_terms.density_source["d"])
    np.testing.assert_allclose(fixed_terms.ion_density_source, dictionary_terms.density_source["d+"])
    np.testing.assert_allclose(fixed_terms.electron_density_source, dictionary_terms.density_source["e"])
    np.testing.assert_allclose(fixed_terms.atom_energy_source, dictionary_terms.energy_source["d"])
    np.testing.assert_allclose(fixed_terms.ion_energy_source, dictionary_terms.energy_source["d+"])
    np.testing.assert_allclose(fixed_terms.electron_energy_source, dictionary_terms.energy_source["e"])
    np.testing.assert_allclose(fixed_terms.atom_momentum_source, dictionary_terms.momentum_source["d"])
    np.testing.assert_allclose(fixed_terms.ion_momentum_source, dictionary_terms.momentum_source["d+"])
    np.testing.assert_allclose(fixed_terms.electron_momentum_source, dictionary_terms.momentum_source["e"])


def test_fixed_layout_hydrogen_reaction_sources_support_jit_and_grad() -> None:
    scalars = _small_scalars()

    def objective(scale):
        shape = (1, 1, 1)
        fixed_terms = fixed_layout_hydrogen_reaction_sources(
            atom_density=jnp.full(shape, 0.7, dtype=jnp.float64) * scale,
            atom_pressure=jnp.full(shape, 0.21, dtype=jnp.float64),
            atom_momentum=jnp.full(shape, 0.14, dtype=jnp.float64),
            ion_density=jnp.full(shape, 0.5, dtype=jnp.float64),
            ion_pressure=jnp.full(shape, 0.2, dtype=jnp.float64),
            ion_momentum=jnp.full(shape, 0.05, dtype=jnp.float64),
            electron_density=jnp.full(shape, 0.5, dtype=jnp.float64),
            electron_pressure=jnp.full(shape, 0.3, dtype=jnp.float64),
            dataset_scalars=scalars,
        )
        return (
            jnp.sum(fixed_terms.ion_density_source)
            + jnp.sum(fixed_terms.electron_energy_source)
            + jnp.sum(fixed_terms.ion_momentum_source)
        )

    value = jit(objective)(jnp.array(1.0, dtype=jnp.float64))
    derivative = grad(objective)(jnp.array(1.0, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))


def test_fixed_layout_dthe_reaction_sources_match_dictionary_path() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_INPUT_DTHE)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    species = _initialize_species(config, mesh=mesh)
    atom_names = ("d", "t", "he")
    ion_names = ("d+", "t+", "he+")
    electron_density = sum(species[name].density for name in ion_names)
    scalars = resolved_dataset_scalars(run_config)

    dictionary_terms = reaction_sources(
        config,
        species=species,
        electron_density=electron_density,
        dataset_scalars=scalars,
    )
    fixed_terms = fixed_layout_dthe_reaction_sources(
        neutral_density=_stack_species_fields(species, atom_names, "density"),
        neutral_pressure=_stack_species_fields(species, atom_names, "pressure"),
        neutral_momentum=_stack_species_fields(species, atom_names, "momentum"),
        ion_density=_stack_species_fields(species, ion_names, "density"),
        ion_pressure=_stack_species_fields(species, ion_names, "pressure"),
        ion_momentum=_stack_species_fields(species, ion_names, "momentum"),
        electron_density=electron_density,
        electron_pressure=species["e"].pressure,
        dataset_scalars=scalars,
    )

    for index, name in enumerate(atom_names):
        np.testing.assert_allclose(fixed_terms.neutral_density_source[index], dictionary_terms.density_source[name])
        np.testing.assert_allclose(fixed_terms.neutral_energy_source[index], dictionary_terms.energy_source[name])
        np.testing.assert_allclose(fixed_terms.neutral_momentum_source[index], dictionary_terms.momentum_source[name])
    for index, name in enumerate(ion_names):
        np.testing.assert_allclose(fixed_terms.ion_density_source[index], dictionary_terms.density_source[name])
        np.testing.assert_allclose(fixed_terms.ion_energy_source[index], dictionary_terms.energy_source[name])
        np.testing.assert_allclose(fixed_terms.ion_momentum_source[index], dictionary_terms.momentum_source[name])
    np.testing.assert_allclose(fixed_terms.electron_density_source, dictionary_terms.density_source["e"])
    np.testing.assert_allclose(fixed_terms.electron_energy_source, dictionary_terms.energy_source["e"])
    np.testing.assert_allclose(fixed_terms.electron_momentum_source, dictionary_terms.momentum_source["e"])


def test_fixed_layout_dthe_active_reaction_terms_match_dictionary_active_slice() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_INPUT_DTHE)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    species = _initialize_species(config, mesh=mesh)
    atom_names = ("d", "t", "he")
    ion_names = ("d+", "t+", "he+")
    electron_density = sum(species[name].density for name in ion_names)
    scalars = resolved_dataset_scalars(run_config)
    active_slices = _active_slices(mesh)
    active_fields = _dthe_active_fields(species, active_slices)

    dictionary_terms = reaction_sources(
        config,
        species=species,
        electron_density=electron_density,
        dataset_scalars=scalars,
    )
    active_terms = fixed_layout_dthe_reaction_terms_from_active_fields(
        config,
        active_fields=active_fields,
        species=species,
        dataset_scalars=scalars,
    )

    for name in (*atom_names, *ion_names, "e"):
        np.testing.assert_allclose(
            np.asarray(active_terms.density_source[name]),
            np.asarray(dictionary_terms.density_source[name][active_slices]),
        )
        np.testing.assert_allclose(
            np.asarray(active_terms.energy_source[name]),
            np.asarray(dictionary_terms.energy_source[name][active_slices]),
        )
        np.testing.assert_allclose(
            np.asarray(active_terms.momentum_source[name]),
            np.asarray(dictionary_terms.momentum_source[name][active_slices]),
        )


def test_fixed_layout_dthe_active_reaction_field_rhs_matches_source_mapping() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_INPUT_DTHE)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    species = _initialize_species(config, mesh=mesh)
    atom_names = ("d", "t", "he")
    ion_names = ("d+", "t+", "he+")
    electron_density = sum(species[name].density for name in ion_names)
    scalars = resolved_dataset_scalars(run_config)
    active_slices = _active_slices(mesh)
    active_fields = _dthe_active_fields(species, active_slices)

    dictionary_terms = reaction_sources(
        config,
        species=species,
        electron_density=electron_density,
        dataset_scalars=scalars,
    )
    field_rhs = fixed_layout_dthe_reaction_field_rhs_from_active_fields(
        config,
        active_fields=active_fields,
        species=species,
        dataset_scalars=scalars,
    )

    for name in (*atom_names, *ion_names):
        sp = species[name]
        np.testing.assert_allclose(
            np.asarray(field_rhs[sp.density_name]),
            np.asarray(dictionary_terms.density_source[name][active_slices]),
        )
        np.testing.assert_allclose(
            np.asarray(field_rhs[sp.pressure_name]),
            (2.0 / 3.0)
            * np.asarray(dictionary_terms.energy_source[name][active_slices]),
        )
        np.testing.assert_allclose(
            np.asarray(field_rhs[sp.momentum_name]),
            np.asarray(dictionary_terms.momentum_source[name][active_slices]),
        )
    np.testing.assert_allclose(
        np.asarray(field_rhs[species["e"].pressure_name]),
        (2.0 / 3.0) * np.asarray(dictionary_terms.energy_source["e"][active_slices]),
    )


def test_fixed_layout_dthe_active_reaction_terms_report_missing_fields() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_INPUT_DTHE)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    species = _initialize_species(config, mesh=mesh)

    with pytest.raises(KeyError, match="Missing active reaction fields"):
        fixed_layout_dthe_reaction_terms_from_active_fields(
            config,
            active_fields={},
            species=species,
            dataset_scalars=resolved_dataset_scalars(run_config),
        )


def test_reaction_sources_without_diagnostics_matches_fixed_dthe_dictionary_path() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_INPUT_DTHE)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    species = _initialize_species(config, mesh=mesh)
    electron_density = sum(species[name].density for name in ("d+", "t+", "he+"))
    scalars = resolved_dataset_scalars(run_config)

    dictionary_terms = reaction_sources(
        config,
        species=species,
        electron_density=electron_density,
        dataset_scalars=scalars,
    )
    lean_terms = reaction_sources(
        config,
        species=species,
        electron_density=electron_density,
        dataset_scalars=scalars,
        include_diagnostics=False,
    )

    assert lean_terms.diagnostics == {}
    for name in species:
        np.testing.assert_allclose(lean_terms.density_source[name], dictionary_terms.density_source[name])
        np.testing.assert_allclose(lean_terms.energy_source[name], dictionary_terms.energy_source[name])
        np.testing.assert_allclose(lean_terms.momentum_source[name], dictionary_terms.momentum_source[name])


def test_fixed_layout_dthe_reaction_sources_apply_atom_charge_exchange_multipliers() -> None:
    scalars = _small_scalars()
    shape = (3, 1, 1, 1)
    neutral_density = np.asarray([0.7, 0.6, 0.02], dtype=np.float64).reshape(shape)
    ion_density = np.asarray([0.5, 0.4, 0.01], dtype=np.float64).reshape(shape)
    common_kwargs = {
        "neutral_density": neutral_density,
        "neutral_pressure": np.asarray([0.21, 0.18, 0.006], dtype=np.float64).reshape(shape),
        "neutral_momentum": np.asarray([0.14, 0.09, 0.0], dtype=np.float64).reshape(shape),
        "ion_density": ion_density,
        "ion_pressure": np.asarray([0.2, 0.16, 0.004], dtype=np.float64).reshape(shape),
        "ion_momentum": np.asarray([0.05, 0.04, 0.0], dtype=np.float64).reshape(shape),
        "electron_density": np.sum(ion_density, axis=0),
        "electron_pressure": np.full((1, 1, 1), 0.3, dtype=np.float64),
        "dataset_scalars": scalars,
    }

    base_terms = fixed_layout_dthe_reaction_sources(**common_kwargs)
    boosted_terms = fixed_layout_dthe_reaction_sources(**common_kwargs, cx_multipliers=(3.0, 1.0, 1.0))

    np.testing.assert_allclose(boosted_terms.charge_exchange_rate[0, 0], 3.0 * base_terms.charge_exchange_rate[0, 0])
    np.testing.assert_allclose(boosted_terms.charge_exchange_rate[0, 1], 3.0 * base_terms.charge_exchange_rate[0, 1])
    np.testing.assert_allclose(boosted_terms.charge_exchange_rate[1, 0], base_terms.charge_exchange_rate[1, 0])


def test_fixed_layout_dthe_reaction_sources_accept_species_specific_density_floors() -> None:
    scalars = _small_scalars()
    shape = (3, 1, 1, 1)
    neutral_density = np.asarray([0.7, 0.6, 1.0e-12], dtype=np.float64).reshape(shape)
    ion_density = np.asarray([0.5, 0.4, 1.0e-12], dtype=np.float64).reshape(shape)
    common_kwargs = {
        "neutral_density": neutral_density,
        "neutral_pressure": np.asarray([0.21, 0.18, 1.0e-7], dtype=np.float64).reshape(shape),
        "neutral_momentum": np.asarray([0.14, 0.09, 0.0], dtype=np.float64).reshape(shape),
        "ion_density": ion_density,
        "ion_pressure": np.asarray([0.2, 0.16, 1.0e-7], dtype=np.float64).reshape(shape),
        "ion_momentum": np.asarray([0.05, 0.04, 0.0], dtype=np.float64).reshape(shape),
        "electron_density": np.sum(ion_density, axis=0),
        "electron_pressure": np.full((1, 1, 1), 0.3, dtype=np.float64),
        "dataset_scalars": scalars,
    }

    uniform_floor_terms = fixed_layout_dthe_reaction_sources(**common_kwargs)
    species_floor_terms = fixed_layout_dthe_reaction_sources(
        **common_kwargs,
        neutral_density_floors=(1.0e-8, 1.0e-8, 1.0e-3),
        ion_density_floors=(1.0e-8, 1.0e-8, 1.0e-3),
    )

    assert np.all(np.isfinite(species_floor_terms.neutral_energy_source))
    assert np.max(np.abs(species_floor_terms.neutral_energy_source[2] - uniform_floor_terms.neutral_energy_source[2])) > 0.0


def test_fixed_layout_dthe_reaction_sources_support_jit_and_grad() -> None:
    scalars = _small_scalars()

    def objective(scale):
        shape = (3, 1, 1, 1)
        neutral_density = jnp.asarray([0.7, 0.6, 0.02], dtype=jnp.float64).reshape(shape) * scale
        ion_density = jnp.asarray([0.5, 0.4, 0.01], dtype=jnp.float64).reshape(shape)
        fixed_terms = fixed_layout_dthe_reaction_sources(
            neutral_density=neutral_density,
            neutral_pressure=jnp.asarray([0.21, 0.18, 0.006], dtype=jnp.float64).reshape(shape),
            neutral_momentum=jnp.asarray([0.14, 0.09, 0.0], dtype=jnp.float64).reshape(shape),
            ion_density=ion_density,
            ion_pressure=jnp.asarray([0.2, 0.16, 0.004], dtype=jnp.float64).reshape(shape),
            ion_momentum=jnp.asarray([0.05, 0.04, 0.0], dtype=jnp.float64).reshape(shape),
            electron_density=jnp.sum(ion_density, axis=0),
            electron_pressure=jnp.full((1, 1, 1), 0.3, dtype=jnp.float64),
            dataset_scalars=scalars,
        )
        return (
            jnp.sum(fixed_terms.ion_density_source)
            + jnp.sum(fixed_terms.electron_energy_source)
            + jnp.sum(fixed_terms.neutral_momentum_source)
        )

    value = jit(objective)(jnp.array(1.0, dtype=jnp.float64))
    derivative = grad(objective)(jnp.array(1.0, dtype=jnp.float64))

    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))


def test_collision_rate_helpers_handle_missing_and_malformed_reactions() -> None:
    species = _small_reaction_species()
    prepared = _small_prepared(species)
    scalars = _small_scalars()
    malformed = parse_bout_input(
        """
[reactions]
type = malformed reaction
"""
    )
    missing_species = parse_bout_input(
        """
[reactions]
type = x + e -> x+ + 2e
"""
    )

    assert neutral_ionisation_collision_rates(parse_bout_input(""), species=species, prepared=prepared, dataset_scalars=scalars) == {}
    assert neutral_ionisation_collision_rates(malformed, species=species, prepared=prepared, dataset_scalars=scalars) == {}
    assert neutral_ionisation_collision_rates(missing_species, species=species, prepared=prepared, dataset_scalars=scalars) == {}
    assert neutral_charge_exchange_collision_rates(parse_bout_input(""), species=species, prepared=prepared, dataset_scalars=scalars) == {}
    assert neutral_charge_exchange_collision_rates(malformed, species=species, prepared=prepared, dataset_scalars=scalars) == {}
    assert charge_exchange_collision_rates(parse_bout_input(""), species=species, prepared=prepared, dataset_scalars=scalars) == {}
    assert charge_exchange_collision_rates(malformed, species=species, prepared=prepared, dataset_scalars=scalars) == {}


def test_charge_exchange_collision_rates_cover_atoms_and_ions() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    species = _initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    rates = charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    assert "d" in rates
    assert "d+" in rates


def test_charge_exchange_collision_rates_cover_cross_isotope_pairs() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_INPUT_DTHE)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    species = _initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    rates = charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    assert "d" in rates
    assert "t" in rates
    assert "d+" in rates
    assert "t+" in rates


def test_neutral_ionisation_collision_rates_match_reaction_diagnostic_per_density() -> None:
    if _REFERENCE_ROOT is None:
        pytest.skip("external hermes-3 reference checkout not available")
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    species = _initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    electron_density = np.zeros_like(species["d+"].density, dtype=np.float64)
    for ion in ions:
        electron_density = electron_density + ion.charge * ion.density

    ionisation_rates = neutral_ionisation_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    terms = reaction_sources(
        config,
        species=species,
        electron_density=electron_density,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    active = (mesh.xstart, mesh.yend, 0)
    expected = float(terms.diagnostics["Sd+_iz"][active] / species["d"].density[active])
    actual = float(ionisation_rates["d"][active])
    assert actual == pytest.approx(expected)


def test_neutral_ionisation_collision_rates_use_openadas_for_neon() -> None:
    config = parse_bout_input(
        """
[reactions]
type = ne + e -> ne+ + 2e
"""
    )
    density = np.full((2, 3, 1), 0.2, dtype=np.float64)
    ion_density = np.full((2, 3, 1), 0.4, dtype=np.float64)
    electron_pressure = np.full((2, 3, 1), 1.2, dtype=np.float64)
    species = {
        "ne": SimpleNamespace(name="ne", charge=0.0),
        "ne+": SimpleNamespace(name="ne+", charge=1.0),
        "e": SimpleNamespace(name="e", charge=-1.0, pressure=electron_pressure, density_floor=1.0e-8),
    }
    prepared = {
        "ne": SimpleNamespace(density=density),
        "ne+": SimpleNamespace(density=ion_density),
        "e": SimpleNamespace(density=ion_density),
    }

    rates = neutral_ionisation_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars={"Nnorm": 1.0e19, "Tnorm": 10.0, "Omega_ci": 1.0e6},
    )

    assert "ne" in rates
    assert np.isfinite(rates["ne"]).all()
    assert float(np.max(rates["ne"])) > 0.0
