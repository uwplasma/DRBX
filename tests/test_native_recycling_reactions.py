from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.recycling_1d import _initialize_species, _prepare_open_field_states
from jax_drb.native.recycling_reactions import (
    charge_exchange_collision_rates,
    is_charge_exchange_reaction,
    neutral_ionisation_collision_rates,
    reaction_sources,
)
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.runtime.run_config import RunConfiguration


_INPUT_1D = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")
_INPUT_DTHE = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling-dthe/data/BOUT.inp")


def test_is_charge_exchange_reaction_identifies_expected_pairs() -> None:
    assert is_charge_exchange_reaction(("d", "d+"), ("d+", "d"))
    assert is_charge_exchange_reaction(("d", "t+"), ("d+", "t"))
    assert not is_charge_exchange_reaction(("d", "e"), ("d+", "2e"))
    assert not is_charge_exchange_reaction(("d",), ("d+",))


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
