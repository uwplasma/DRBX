from __future__ import annotations

from pathlib import Path

import numpy as np

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native import run_curated_case
from jax_drb.native.recycling_1d import (
    _compute_collision_frequencies,
    _electron_density,
    _initialize_species,
    _load_amjuel_rate,
    _prepare_open_field_states,
    _reaction_sources,
)
from jax_drb.parity.arrays import (
    build_array_payload_from_summary_payload,
    compare_array_payloads,
    load_portable_array_payload,
)
from jax_drb.parity.compare import compare_summary_payloads, load_summary_json
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars


_REFERENCE_ROOT = Path("/Users/rogerio/local/hermes-3")
_BASELINE_DIR = Path("/Users/rogerio/local/jax_drb/references/baselines/reference")
_ARRAY_BASELINE_DIR = Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays")
_DTHE_INPUT = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling-dthe/data/BOUT.inp")


def test_amjuel_rate_tables_are_packaged_for_recycling_branch() -> None:
    hydrogen_iz_coeffs, hydrogen_iz_energy_coeffs, hydrogen_iz_heating = _load_amjuel_rate("d", "iz")
    helium_rec_coeffs, helium_rec_energy_coeffs, helium_rec_heating = _load_amjuel_rate("he", "rec")

    assert hydrogen_iz_coeffs.shape == (9, 9)
    assert helium_rec_coeffs.shape == (9, 9)
    assert hydrogen_iz_energy_coeffs.shape == (9, 9)
    assert helium_rec_energy_coeffs.shape == (9, 9)
    assert np.isfinite(hydrogen_iz_heating)
    assert np.isfinite(helium_rec_heating)


def test_recycling_1d_rhs_matches_summary_baseline() -> None:
    expected = load_summary_json(_BASELINE_DIR / "recycling_1d_rhs.json")
    actual = run_curated_case("recycling_1d_rhs", reference_root=_REFERENCE_ROOT).payload

    comparison = compare_summary_payloads(expected, actual, scalar_rtol=1.0e-6, scalar_atol=1.0e-9)

    assert comparison.ok, comparison.issues


def test_recycling_1d_rhs_matches_array_baseline() -> None:
    expected = load_portable_array_payload(_ARRAY_BASELINE_DIR / "recycling_1d_rhs.npz")
    result = run_curated_case("recycling_1d_rhs", reference_root=_REFERENCE_ROOT)
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=1.0e-6, array_atol=1.0e-9)

    assert comparison.ok, comparison.issues


def test_recycling_dthe_rhs_matches_summary_baseline() -> None:
    expected = load_summary_json(_BASELINE_DIR / "recycling_dthe_rhs.json")
    actual = run_curated_case("recycling_dthe_rhs", reference_root=_REFERENCE_ROOT).payload

    comparison = compare_summary_payloads(expected, actual, scalar_rtol=5.0e-2, scalar_atol=1.0e-9)

    assert comparison.ok, comparison.issues


def test_recycling_dthe_rhs_matches_array_baseline() -> None:
    expected = load_portable_array_payload(_ARRAY_BASELINE_DIR / "recycling_dthe_rhs.npz")
    result = run_curated_case("recycling_dthe_rhs", reference_root=_REFERENCE_ROOT)
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=5.0e-2, array_atol=1.0e-9)

    assert comparison.ok, comparison.issues


def test_recycling_dthe_reaction_sources_include_cross_isotope_charge_exchange() -> None:
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    species = _initialize_species(config, mesh=mesh)
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    reaction_terms = _reaction_sources(
        config,
        species=species,
        electron_density=_electron_density(ions),
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    assert "Edt+_cx" in reaction_terms.diagnostics
    assert "Etd+_cx" in reaction_terms.diagnostics
    assert "Sdt+_cx" in reaction_terms.diagnostics
    assert "Std+_cx" in reaction_terms.diagnostics


def test_recycling_dthe_collision_rates_cover_asymmetric_ion_neutral_pairs() -> None:
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    species = _initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(species, mesh=mesh, metrics=metrics)
    collision_rates = _compute_collision_frequencies(
        config,
        species,
        prepared,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    for pair in (("d", "t+"), ("t+", "d"), ("t", "d+"), ("d+", "t")):
        assert pair in collision_rates

    active = (0, mesh.ystart, 0)
    assert np.isfinite(float(collision_rates[("t", "d+")][active]))
    assert np.isfinite(float(collision_rates[("d", "t+")][active]))
    assert float(collision_rates[("t", "d+")][active]) != float(collision_rates[("d", "t+")][active])
