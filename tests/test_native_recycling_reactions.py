from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
    is_charge_exchange_reaction,
    neutral_charge_exchange_collision_rates,
    neutral_ionisation_collision_rates,
    reaction_sources,
)
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.runtime.run_config import RunConfiguration


_INPUT_1D = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")
_INPUT_DTHE = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling-dthe/data/BOUT.inp")


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
