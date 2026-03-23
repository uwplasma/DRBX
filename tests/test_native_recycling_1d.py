from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import load_bout_input
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native import run_curated_case
from jax_drb.native.recycling_1d import (
    _advance_feedback_integrals,
    _charge_exchange_collision_rates,
    _compute_collision_frequencies,
    _compute_recycling_1d_packed_rhs,
    _current_feedback_errors,
    _electron_density,
    _build_recycling_runtime_model,
    _build_recycling_state_fields,
    advance_recycling_1d_implicit_history,
    _initialize_species,
    _hydrogen_cx_sigmav,
    _load_amjuel_rate,
    _prepare_open_field_states,
    _reaction_sources,
    _recycling_evolving_variable_names,
    compute_recycling_1d_rhs,
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


def test_charge_exchange_collision_rates_include_both_atom_and_ion_species() -> None:
    input_path = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    species = _initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(species, mesh=mesh, metrics=metrics)
    cx_rates = _charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    assert "d" in cx_rates
    assert "d+" in cx_rates

    active = (mesh.xstart, mesh.ystart, 0)
    atom_temperature = prepared["d"].temperature
    ion_temperature = prepared["d+"].temperature
    teff = np.clip(
        (atom_temperature / species["d"].atomic_mass + ion_temperature / species["d+"].atomic_mass)
        * resolved_dataset_scalars(run_config)["Tnorm"],
        0.01,
        10000.0,
    )
    sigma_v = _hydrogen_cx_sigmav(teff, resolved_dataset_scalars(run_config))

    assert float(cx_rates["d"][active]) == pytest.approx(
        float(prepared["d+"].density[active] * sigma_v[active])
    )
    assert float(cx_rates["d+"][active]) == pytest.approx(
        float(prepared["d"].density[active] * sigma_v[active])
    )
    assert float(cx_rates["d"][active]) != float(cx_rates["d+"][active])


def test_single_species_feedback_diagnostics_are_present_and_zero_initially() -> None:
    input_path = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    result = compute_recycling_1d_rhs(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    assert "Sd+_feedback" in result.variables
    assert "density_feedback_src_mult_d+" in result.variables
    assert float(np.asarray(result.variables["density_feedback_src_mult_d+"]).reshape(-1)[0]) == 0.0


def test_multispecies_feedback_controller_detects_initial_helium_density_error() -> None:
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    result = compute_recycling_1d_rhs(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    multiplier = float(np.asarray(result.variables["density_feedback_src_mult_he+"]).reshape(-1)[0])
    proportional = float(np.asarray(result.variables["density_feedback_src_p_he+"]).reshape(-1)[0])
    source = np.asarray(result.variables["She+_feedback"][0])

    assert multiplier == pytest.approx(495.0)
    assert proportional == pytest.approx(495.0)
    assert np.allclose(source, 0.0, rtol=0.0, atol=0.0)


def test_feedback_integrals_advance_with_reference_trapezoid_rule() -> None:
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    fields = _build_recycling_state_fields(runtime_model)
    previous_errors = _current_feedback_errors(fields, controllers=runtime_model.controllers, mesh=mesh)

    updated = _advance_feedback_integrals(
        fields,
        controllers=runtime_model.controllers,
        feedback_integrals={name: 0.0 for name in runtime_model.feedback_names},
        feedback_previous_errors=previous_errors,
        mesh=mesh,
        timestep=1.0,
    )

    assert updated["he+"] == pytest.approx(previous_errors["he+"])
    assert updated["d+"] == pytest.approx(previous_errors["d+"])
    assert updated["t+"] == pytest.approx(previous_errors["t+"])


def test_runtime_model_packed_rhs_matches_uncached_path() -> None:
    input_path = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)

    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=scalars,
    )
    field_names = _recycling_evolving_variable_names(runtime_model.species_templates)
    fields = _build_recycling_state_fields(runtime_model)
    feedback_integrals = {name: 0.0 for name in runtime_model.feedback_names}

    uncached = _compute_recycling_1d_packed_rhs(
        config,
        fields,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=runtime_model.feedback_names,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )
    cached = _compute_recycling_1d_packed_rhs(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=feedback_integrals,
        field_names=field_names,
        feedback_names=runtime_model.feedback_names,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )

    assert np.allclose(cached, uncached, rtol=1.0e-12, atol=1.0e-12)


def test_recycling_continuation_history_produces_finite_small_step() -> None:
    input_path = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)

    history = advance_recycling_1d_implicit_history(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=25.0,
        steps=1,
        solver_mode="continuation",
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=10,
    )

    assert history.variable_history["Nd+"].shape[0] == 2
    assert np.isfinite(history.variable_history["Nd+"]).all()
    assert np.isfinite(history.variable_history["Pe"]).all()
