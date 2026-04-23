from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.config.boutinp import apply_bout_overrides, load_bout_input
import jax_drb.native.recycling_1d as recycling_1d_mod
import jax_drb.native.runner as native_runner
from jax_drb.native.mesh import StructuredMesh, build_structured_mesh
from jax_drb.native.metrics import StructuredMetrics, build_structured_metrics
from jax_drb.native.open_field import limit_free
from jax_drb.native import run_curated_case
from jax_drb.native.reference_dump import (
    LocalReferenceSnapshot,
    load_local_reference_snapshot,
    load_local_reference_snapshot_cache,
    save_local_reference_snapshot_cache,
    synthesize_local_reference_snapshot_from_active_history,
)
from jax_drb.native.recycling_1d import (
    OpenFieldSpecies,
    _FullSheathSettings,
    _PreparedSpeciesState,
    _SimpleSheathSettings,
    _apply_electron_sheath_boundary,
    _advance_feedback_integrals,
    _charge_exchange_collision_rates,
    _compute_recycling_1d_rhs_from_species,
    _compute_recycling_1d_packed_rhs,
    _current_feedback_errors,
    _electron_zero_current_velocity,
    _electron_density,
    _grad_par_electron_force_balance_open,
    _apply_neutral_target_density_guards,
    _build_recycling_runtime_model,
    _build_recycling_state_fields,
    advance_recycling_1d_implicit_history,
    _apply_ion_sheath_boundary,
    _initialize_species,
    _hydrogen_cx_sigmav,
    _load_amjuel_rate,
    _load_openadas_rate,
    _eval_openadas_rate,
    _neutral_ionisation_collision_rates,
    _prepare_open_field_states,
    _reaction_sources,
    _recycling_evolving_variable_names,
    _resolve_species_numeric_option,
    _soft_floor,
    _target_recycling_sources,
    advance_recycling_1d_backward_euler_step,
    advance_recycling_1d_bdf2_step,
    compute_recycling_1d_rhs,
)
from jax_drb.native.recycling_rhs_terms import ElectronPressureRhsTerms, IonRhsTerms
from jax_drb.parity.arrays import (
    build_array_payload_from_summary_payload,
    compare_array_payloads,
    load_portable_array_payload,
)
from jax_drb.parity.compare import compare_summary_payloads, load_summary_json
from jax_drb.parity.diff import build_scaled_array_diff_entries
from jax_drb.reference.cases import load_reference_cases
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.native.units import resolved_dataset_scalars


_REFERENCE_ROOT = Path("/Users/rogerio/local/hermes-3")
_BASELINE_DIR = Path("/Users/rogerio/local/jax_drb/references/baselines/reference")
_ARRAY_BASELINE_DIR = Path("/Users/rogerio/local/jax_drb/references/baselines/reference_arrays")
_DTHE_INPUT = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling-dthe/data/BOUT.inp")
_INPUT_1D = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")
_TOKAMAK_RECYCLING_INPUT = Path("/Users/rogerio/local/hermes-3/examples/tokamak-2D/recycling/BOUT.inp")


def _run_open_field_case_against_committed_baseline(case_name: str):
    case = next(case for case in load_reference_cases() if case.name == case_name)
    result = run_curated_case(case_name, reference_root=_REFERENCE_ROOT)
    expected = load_portable_array_payload(_ARRAY_BASELINE_DIR / f"{case_name}.npz")
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)
    entries = build_scaled_array_diff_entries(
        expected["variables"],
        actual["variables"],
        compare_variables=case.compare_variables,
    )
    return {entry.field: entry for entry in entries}


def test_amjuel_rate_tables_are_packaged_for_recycling_branch() -> None:
    hydrogen_iz_coeffs, hydrogen_iz_energy_coeffs, hydrogen_iz_heating = _load_amjuel_rate("d", "iz")
    helium_rec_coeffs, helium_rec_energy_coeffs, helium_rec_heating = _load_amjuel_rate("he", "rec")

    assert hydrogen_iz_coeffs.shape == (9, 9)
    assert helium_rec_coeffs.shape == (9, 9)
    assert hydrogen_iz_energy_coeffs.shape == (9, 9)
    assert helium_rec_energy_coeffs.shape == (9, 9)
    assert np.isfinite(hydrogen_iz_heating)
    assert np.isfinite(helium_rec_heating)


def test_openadas_neon_rate_tables_are_packaged_for_recycling_branch() -> None:
    ionisation_coeffs, radiation_coeffs, log_temperature, log_density, electron_heating = _load_openadas_rate("ne", "iz")

    assert ionisation_coeffs.shape == (30, 24)
    assert radiation_coeffs.shape == (30, 24)
    assert log_temperature.shape == (30,)
    assert log_density.shape == (24,)
    assert electron_heating < 0.0

    evaluated = _eval_openadas_rate(
        np.full((2, 2, 1), 5.0, dtype=np.float64),
        np.full((2, 2, 1), 2.0e18, dtype=np.float64),
        ionisation_coeffs,
        log_temperature=log_temperature,
        log_density=log_density,
    )
    assert np.all(np.isfinite(evaluated))
    assert np.all(evaluated > 0.0)


def test_apply_electron_sheath_boundary_matches_full_hermes_lower_boundary_formula() -> None:
    mesh = StructuredMesh(
        nx=1,
        ny=1,
        nz=1,
        mxg=0,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=0,
        jyseps1_2=0,
        jyseps2_2=0,
        ny_inner=1,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=np.array([0.0], dtype=np.float64),
        y=np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )
    ones = np.ones((1, 3, 1), dtype=np.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g22=ones,
        g33=ones,
        g_22=ones,
        g23=np.zeros_like(ones),
        Bxy=ones,
    )
    electron_density = np.array([[[0.0], [4.0], [2.0]]], dtype=np.float64)
    electron_pressure = np.array([[[0.0], [8.0], [4.0]]], dtype=np.float64)
    electron_velocity = np.zeros((1, 3, 1), dtype=np.float64)
    ion_density = np.array([[[0.0], [3.0], [1.5]]], dtype=np.float64)
    ion_temperature = np.array([[[0.0], [1.5], [0.75]]], dtype=np.float64)
    zero = np.zeros((1, 3, 1), dtype=np.float64)
    ion = OpenFieldSpecies(
        name="d+",
        density=ion_density,
        pressure=ion_density * ion_temperature,
        momentum=zero,
        charge=1.0,
        atomic_mass=2.0,
        density_floor=1.0e-8,
        has_pressure=True,
        has_momentum=True,
        noflow_lower_y=False,
        noflow_upper_y=False,
        target_recycle=False,
        recycle_as=None,
        target_recycle_multiplier=0.0,
        target_recycle_energy=0.0,
        target_fast_recycle_fraction=0.0,
        target_fast_recycle_energy_factor=1.0,
    )
    prepared_ions = {
        "d+": _PreparedSpeciesState(
            density=ion_density,
            pressure=ion_density * ion_temperature,
            temperature=ion_temperature,
            velocity=zero,
            momentum=zero,
            momentum_error=zero,
        )
    }
    full_settings = _FullSheathSettings(
        secondary_electron_coef=0.2,
        sin_alpha=np.array([[[1.0], [1.0], [0.4]]], dtype=np.float64),
        lower_y=True,
        upper_y=False,
        wall_potential=np.array([[[0.0], [0.3], [0.0]]], dtype=np.float64),
        floor_potential=True,
    )

    result = _apply_electron_sheath_boundary(
        electron_pressure=electron_pressure,
        electron_density=electron_density,
        electron_velocity=electron_velocity,
        electron_mass=1.0 / 1836.0,
        electron_density_floor=1.0e-8,
        ion_velocity={"d+": zero},
        ions=(ion,),
        prepared_ions=prepared_ions,
        mesh=mesh,
        metrics=metrics,
        simple_settings=None,
        full_settings=full_settings,
    )

    j = mesh.ystart
    jp = j + 1
    jm = j - 1
    ne_center = float(electron_density[0, j, 0])
    ne_guard = float(limit_free(electron_density[0, jp, 0], electron_density[0, j, 0], 0.0))
    te_center = float(electron_pressure[0, j, 0] / electron_density[0, j, 0])
    te_neighbor = float((electron_pressure[0, jp, 0] / electron_density[0, jp, 0]))
    te_guard = float(limit_free(te_neighbor, te_center, 0.0))
    ni_center = float(ion_density[0, j, 0])
    ni_neighbor = float(ion_density[0, jp, 0])
    grad_ne = float(electron_density[0, jp, 0] - electron_density[0, j, 0])
    grad_ni = float(ni_neighbor - ni_center)
    if abs(grad_ni) < 2.0e-3:
        grad_ne = 2.0e-3
        grad_ni = 2.0e-3
    s_i = float(np.clip(0.5 * (3.0 * ni_center / ne_center - ni_neighbor / electron_density[0, jp, 0]), 0.0, 1.0))
    c_i_sq = float(
        np.clip(((5.0 / 3.0) * ion_temperature[0, j, 0] + s_i * te_center * grad_ne / grad_ni) / ion.atomic_mass, 0.0, 100.0)
    )
    me = 1.0 / 1836.0
    ion_sum = float(s_i * ion.charge * full_settings.sin_alpha[0, jp, 0] * np.sqrt(c_i_sq))
    phi_center = te_center * np.log(
        np.sqrt(te_center / (me * (2.0 * np.pi))) * (1.0 - full_settings.secondary_electron_coef) / ion_sum
    ) + full_settings.wall_potential[0, j, 0]
    tesheath = 0.5 * (te_guard + te_center)
    nesheath = 0.5 * (ne_guard + ne_center)
    phisheath = max(phi_center, full_settings.wall_potential[0, j, 0])
    gamma_e = max(
        2.0 / (1.0 - full_settings.secondary_electron_coef)
        + (phisheath - full_settings.wall_potential[0, j, 0]) / max(tesheath, 1.0e-5),
        0.0,
    )
    vesheath = -np.sqrt(tesheath / (2.0 * np.pi * me)) * (1.0 - full_settings.secondary_electron_coef) * np.exp(
        -(phisheath - full_settings.wall_potential[0, j, 0]) / tesheath
    )
    expected_q = ((gamma_e - 1.0 - 1.0 / ((5.0 / 3.0) - 1.0)) * tesheath - 0.5 * me * vesheath * vesheath)
    expected_q *= nesheath * vesheath
    expected_q = min(expected_q, 0.0)

    assert result.velocity[0, jm, 0] == pytest.approx(2.0 * vesheath)
    assert result.energy_source[0, j, 0] == pytest.approx(expected_q)
def test_compute_recycling_1d_rhs_applies_neutral_pressure_source_overrides() -> None:
    config = load_bout_input(_TOKAMAK_RECYCLING_INPUT)
    snapshot = load_local_reference_snapshot_cache(
        Path("/Users/rogerio/local/jax_drb/references/baselines/reference_snapshots/tokamak_recycling_rhs_snapshot.npz"),
        field_names=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe"),
        optional_field_names=("SNd+", "SNVd+", "SPd+", "SNd", "SNVd", "SPd", "SPe"),
        scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
    )
    override = np.full_like(snapshot.fields["Pd"], 0.125, dtype=np.float64)

    base = compute_recycling_1d_rhs(
        config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=snapshot.scalar_values,
        field_overrides=snapshot.fields,
        pressure_source_overrides={"d+": np.asarray(snapshot.optional_fields["SPd+"], dtype=np.float64)},
    )
    overridden = compute_recycling_1d_rhs(
        config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=snapshot.scalar_values,
        field_overrides=snapshot.fields,
        pressure_source_overrides={
            "d+": np.asarray(snapshot.optional_fields["SPd+"], dtype=np.float64),
            "d": override,
        },
    )

    np.testing.assert_allclose(
        np.asarray(overridden.variables["ddt(Pd)"][0]) - np.asarray(base.variables["ddt(Pd)"][0]),
        override,
    )


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


def test_recycling_1d_short_window_stays_within_operational_baseline_band() -> None:
    entries = _run_open_field_case_against_committed_baseline("recycling_1d_short_window")

    assert entries["Nd+"].max_abs_diff < 4.8e-2
    assert entries["Pd+"].max_abs_diff < 6.5e-2
    assert entries["NVd+"].max_abs_diff < 1.1e-1
    assert entries["Nd"].max_abs_diff < 2.8e-2
    assert entries["Pd"].max_abs_diff < 8.5e-3
    assert entries["NVd"].max_abs_diff < 3.5e-3
    assert entries["Pe"].max_abs_diff < 2.1e-2


def test_tokamak_recycling_rhs_matches_summary_baseline() -> None:
    expected = load_summary_json(_BASELINE_DIR / "tokamak_recycling_rhs.json")
    actual = run_curated_case("tokamak_recycling_rhs", reference_root=_REFERENCE_ROOT).payload

    comparison = compare_summary_payloads(expected, actual, scalar_rtol=1.0e-6, scalar_atol=1.0e-9)

    assert comparison.ok, comparison.issues


def test_tokamak_recycling_rhs_matches_array_baseline() -> None:
    expected = load_portable_array_payload(_ARRAY_BASELINE_DIR / "tokamak_recycling_rhs.npz")
    result = run_curated_case("tokamak_recycling_rhs", reference_root=_REFERENCE_ROOT)
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=1.0e-6, array_atol=1.0e-9)

    assert comparison.ok, comparison.issues


def test_tokamak_recycling_dthe_rhs_matches_summary_baseline() -> None:
    expected = load_summary_json(_BASELINE_DIR / "tokamak_recycling_dthe_rhs.json")
    actual = run_curated_case("tokamak_recycling_dthe_rhs", reference_root=_REFERENCE_ROOT).payload

    comparison = compare_summary_payloads(expected, actual, scalar_rtol=1.0e-6, scalar_atol=1.0e-9)

    assert comparison.ok, comparison.issues


def test_tokamak_recycling_dthe_rhs_matches_array_baseline() -> None:
    expected = load_portable_array_payload(_ARRAY_BASELINE_DIR / "tokamak_recycling_dthe_rhs.npz")
    result = run_curated_case("tokamak_recycling_dthe_rhs", reference_root=_REFERENCE_ROOT)
    actual = build_array_payload_from_summary_payload(result.payload, result.variables)

    comparison = compare_array_payloads(expected, actual, array_rtol=1.0e-6, array_atol=1.0e-9)

    assert comparison.ok, comparison.issues


def test_tokamak_recycling_dthene_rhs_matches_summary_baseline() -> None:
    expected = load_summary_json(_BASELINE_DIR / "tokamak_recycling_dthene_rhs.json")
    actual = run_curated_case("tokamak_recycling_dthene_rhs", reference_root=_REFERENCE_ROOT).payload

    comparison = compare_summary_payloads(expected, actual, scalar_rtol=1.0e-6, scalar_atol=1.0e-9)

    assert comparison.ok, comparison.issues


def test_tokamak_recycling_dthene_rhs_matches_array_baseline() -> None:
    expected = load_portable_array_payload(_ARRAY_BASELINE_DIR / "tokamak_recycling_dthene_rhs.npz")
    result = run_curated_case("tokamak_recycling_dthene_rhs", reference_root=_REFERENCE_ROOT)
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


@pytest.mark.parametrize(
    ("input_path", "expected_solver_mode"),
    [
        (_INPUT_1D, "continuation"),
        (_DTHE_INPUT, "bdf"),
    ],
)
def test_recycling_one_step_selects_expected_transient_solver_mode(
    input_path: Path,
    expected_solver_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)

    calls: list[str] = []

    def fake_advance(*args, **kwargs):
        calls.append(kwargs["solver_mode"])
        field_history = {
            "Nd+": np.zeros((2, mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64),
        }
        return SimpleNamespace(variable_history=field_history)

    monkeypatch.setattr(native_runner, "advance_recycling_1d_implicit_history", fake_advance)

    time_points, variables = native_runner._execute_recycling_1d_case(
        config,
        run_config,
        mesh,
        metrics,
        parity_mode="one_step",
    )

    assert calls == [expected_solver_mode]
    assert time_points == (0.0, run_config.time.timestep)
    assert "Nd+" in variables


def test_recycling_one_step_runtime_override_selects_requested_solver_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = apply_bout_overrides(
        load_bout_input(_INPUT_1D),
        ("runtime:recycling_transient_solver_mode=adaptive_bdf",),
    )
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)

    calls: list[str] = []

    def fake_advance(*args, **kwargs):
        calls.append(kwargs["solver_mode"])
        field_history = {
            "Nd+": np.zeros((2, mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64),
        }
        return SimpleNamespace(variable_history=field_history)

    monkeypatch.setattr(native_runner, "advance_recycling_1d_implicit_history", fake_advance)

    native_runner._execute_recycling_1d_case(
        config,
        run_config,
        mesh,
        metrics,
        parity_mode="one_step",
    )

    assert calls == ["adaptive_bdf"]


def test_recycling_one_step_progress_callback_receives_interval_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)

    progress_events: list[dict[str, object]] = []

    def fake_advance(*args, **kwargs):
        progress_callback = kwargs["progress_callback"]
        assert progress_callback is not None
        progress_callback(
            {
                "interval_index": 1,
                "steps": 1,
                "solver_mode": kwargs["solver_mode"],
                "accepted_dt": float(run_config.time.timestep),
                "stored_states": 2,
            }
        )
        field_history = {
            "Nd+": np.zeros((2, mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64),
        }
        return SimpleNamespace(variable_history=field_history)

    monkeypatch.setattr(native_runner, "advance_recycling_1d_implicit_history", fake_advance)

    native_runner._execute_recycling_1d_case(
        config,
        run_config,
        mesh,
        metrics,
        parity_mode="one_step",
        progress_callback=progress_events.append,
    )

    assert progress_events == [
        {
            "interval_index": 1,
            "steps": 1,
            "solver_mode": "continuation",
            "accepted_dt": float(run_config.time.timestep),
            "stored_states": 2,
        }
    ]


def test_recycling_1d_one_step_uses_committed_snapshot_without_field_templates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = _INPUT_1D
    if not input_path.exists():
        pytest.skip("open-field recycling reference input is unavailable")

    mesh = StructuredMesh(
        nx=1,
        ny=3,
        nz=1,
        mxg=0,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=2,
        jyseps1_2=2,
        jyseps2_2=2,
        ny_inner=3,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=np.array([0.0], dtype=np.float64),
        y=np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )
    ones = np.ones((1, 5, 1), dtype=np.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g22=ones,
        g33=ones,
        g_22=ones,
        g23=np.zeros_like(ones),
        Bxy=ones,
    )
    initial_fields = {
        "Nd+": np.ones((1, 5, 1), dtype=np.float64),
        "Pd+": 2.0 * np.ones((1, 5, 1), dtype=np.float64),
        "NVd+": np.zeros((1, 5, 1), dtype=np.float64),
        "Nd": np.ones((1, 5, 1), dtype=np.float64),
        "Pd": 0.5 * np.ones((1, 5, 1), dtype=np.float64),
        "NVd": np.zeros((1, 5, 1), dtype=np.float64),
        "Pe": 3.0 * np.ones((1, 5, 1), dtype=np.float64),
    }
    snapshot = LocalReferenceSnapshot(
        mesh=mesh,
        metrics=metrics,
        fields=initial_fields,
        optional_fields={},
        scalar_values={"Nnorm": 1.0e17, "Tnorm": 1.0, "Bnorm": 1.0, "Cs0": 1.0, "Omega_ci": 1.0, "rho_s0": 1.0},
    )
    snapshot_cache = tmp_path / "recycling_1d_rhs_snapshot.npz"
    save_local_reference_snapshot_cache(snapshot, snapshot_cache)
    array_history_path = tmp_path / "recycling_1d_one_step.npz"
    active = (slice(mesh.xstart, mesh.xend + 1), slice(mesh.ystart, mesh.yend + 1), slice(None))
    np.savez_compressed(
        array_history_path,
        **{
            "__metadata__": np.array([], dtype=np.float64),
            "var__Nd+": np.stack([initial_fields["Nd+"][active], 2.0 * np.ones((1, 3, 1), dtype=np.float64)], axis=0),
            "var__Pd+": np.stack([initial_fields["Pd+"][active], 4.0 * np.ones((1, 3, 1), dtype=np.float64)], axis=0),
            "var__NVd+": np.stack([initial_fields["NVd+"][active], 5.0 * np.ones((1, 3, 1), dtype=np.float64)], axis=0),
            "var__Nd": np.stack([initial_fields["Nd"][active], 6.0 * np.ones((1, 3, 1), dtype=np.float64)], axis=0),
            "var__Pd": np.stack([initial_fields["Pd"][active], 7.0 * np.ones((1, 3, 1), dtype=np.float64)], axis=0),
            "var__NVd": np.stack([initial_fields["NVd"][active], 8.0 * np.ones((1, 3, 1), dtype=np.float64)], axis=0),
            "var__Pe": np.stack([initial_fields["Pe"][active], 9.0 * np.ones((1, 3, 1), dtype=np.float64)], axis=0),
        },
    )

    monkeypatch.setattr(
        native_runner,
        "_open_field_snapshot_cache_path",
        lambda case_name: snapshot_cache if case_name == "recycling_1d_rhs" else tmp_path / f"{case_name}.missing",
    )
    monkeypatch.setattr(
        native_runner,
        "run_reference_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reference run should not be used when open-field cache is present")),
    )

    captured: dict[str, object] = {}
    evolved_history = {name: np.stack([value, value + 1.0], axis=0) for name, value in initial_fields.items()}

    def fake_history(*args, **kwargs):
        captured["initial_fields"] = kwargs["initial_fields"]
        captured["solver_mode"] = kwargs["solver_mode"]
        return SimpleNamespace(variable_history=evolved_history, feedback_integral_history={})

    monkeypatch.setattr(native_runner, "advance_recycling_1d_implicit_history", fake_history)

    case = next(case for case in load_reference_cases() if case.name == "recycling_1d_one_step")
    result = native_runner._run_open_field_recycling_one_step_case(
        case,
        input_path=input_path,
        reference_root=_REFERENCE_ROOT,
    )

    assert captured["solver_mode"] == "continuation"
    np.testing.assert_allclose(captured["initial_fields"]["Nd+"], initial_fields["Nd+"])
    np.testing.assert_allclose(captured["initial_fields"]["Pe"], initial_fields["Pe"])
    assert result.time_points == (0.0, 5000.0)
    assert np.asarray(result.variables["Nd+"]).shape == (2, 1, 3, 1)


def test_continuation_output_interval_uses_small_startup_substeps_on_first_open_field_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=dataset_scalars,
    )
    fields = _build_recycling_state_fields(runtime_model)
    observed_steps: list[tuple[str, float]] = []

    def fake_be(*args, **kwargs):
        observed_steps.append(("be", float(kwargs["timestep"])))
        return (
            {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()},
            dict(kwargs["feedback_integrals"]),
            SimpleNamespace(residual_inf_norm=0.0, active_size=1, nonlinear_iterations=1, linear_iterations=1),
        )

    def fake_bdf2(*args, **kwargs):
        observed_steps.append(("bdf2", float(kwargs["timestep"])))
        return (
            {name: np.asarray(value, dtype=np.float64, copy=True) for name, value in fields.items()},
            dict(kwargs["feedback_integrals"]),
            SimpleNamespace(residual_inf_norm=0.0, active_size=1, nonlinear_iterations=1, linear_iterations=1),
        )

    monkeypatch.setattr(recycling_1d_mod, "advance_recycling_1d_backward_euler_step", fake_be)
    monkeypatch.setattr(recycling_1d_mod, "advance_recycling_1d_bdf2_step", fake_bdf2)

    recycling_1d_mod._advance_recycling_1d_output_interval(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals={},
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        output_timestep=25.0,
        suggested_dt=25.0,
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=10,
        startup_warmup=True,
    )

    assert observed_steps == [
        ("be", 6.25),
        ("bdf2", 6.25),
        ("bdf2", 6.25),
        ("bdf2", 6.25),
    ]


def test_run_curated_recycling_case_applies_manifest_overrides_on_default_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_config_case(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            payload={"case_name": kwargs["case_name"], "parity_mode": kwargs["parity_mode"]},
            variables={},
            time_points=(0.0,),
            run_config=RunConfiguration.from_config(config),
            mesh=SimpleNamespace(),
            metrics=SimpleNamespace(),
        )

    monkeypatch.setattr(native_runner, "run_config_case", fake_run_config_case)

    native_runner.run_curated_case("recycling_1d_short_window", reference_root=_REFERENCE_ROOT)

    applied_config = captured["config"]
    applied_run_config = RunConfiguration.from_config(applied_config)
    assert applied_run_config.time.nout == 5
    assert captured["kwargs"]["parity_mode"] == "short_window"


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


def test_charge_exchange_collision_rates_include_both_atom_and_ion_species() -> None:
    input_path = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")
    config = load_bout_input(input_path)
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


def test_charge_exchange_collision_rates_apply_species_rate_multiplier() -> None:
    config = apply_bout_overrides(load_bout_input(_DTHE_INPUT), ("d:K_cx_multiplier=3.0",))
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    species = _initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
    )

    cx_rates = _charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=scalars,
    )

    active = (mesh.xstart, mesh.ystart, 0)
    atom_temperature = prepared["d"].temperature
    other_atom_temperature = prepared["t"].temperature
    ion_temperature_same = prepared["d+"].temperature
    ion_temperature_cross = prepared["t+"].temperature
    sigma_same = _hydrogen_cx_sigmav(
        np.clip(
            (atom_temperature / species["d"].atomic_mass + ion_temperature_same / species["d+"].atomic_mass)
            * scalars["Tnorm"],
            0.01,
            10000.0,
        ),
        scalars,
    )
    sigma_cross = _hydrogen_cx_sigmav(
        np.clip(
            (atom_temperature / species["d"].atomic_mass + ion_temperature_cross / species["t+"].atomic_mass)
            * scalars["Tnorm"],
            0.01,
            10000.0,
        ),
        scalars,
    )
    sigma_cross_into_d = _hydrogen_cx_sigmav(
        np.clip(
            (other_atom_temperature / species["t"].atomic_mass + ion_temperature_same / species["d+"].atomic_mass)
            * scalars["Tnorm"],
            0.01,
            10000.0,
        ),
        scalars,
    )
    expected_d_atom = 3.0 * (
        prepared["d+"].density[active] * sigma_same[active]
        + prepared["t+"].density[active] * sigma_cross[active]
    )
    expected_d_ion = 3.0 * (
        prepared["d"].density[active] * sigma_same[active]
    ) + (
        prepared["t"].density[active] * sigma_cross_into_d[active]
    )
    expected_t_ion_from_d = 3.0 * prepared["d"].density[active] * sigma_cross[active]

    assert float(cx_rates["d"][active]) == pytest.approx(float(expected_d_atom))
    assert float(cx_rates["d+"][active]) == pytest.approx(float(expected_d_ion))
    assert float(cx_rates["t+"][active]) >= float(expected_t_ion_from_d)


def test_prepare_open_field_states_keeps_dump_backed_ion_guards_when_preserving_ion_target_state_only() -> None:
    config = load_bout_input(Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-recycling/data/BOUT.inp"))
    snapshot = load_local_reference_snapshot(
        Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-recycling/data/BOUT.dmp.0.nc"),
        field_names=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe"),
        scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
    )
    mesh = snapshot.mesh
    metrics = snapshot.metrics
    species = _initialize_species(config, mesh=mesh, field_overrides=snapshot.fields)

    prepared_default, ion_boundary_default, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=snapshot.scalar_values,
        apply_sheath_boundaries=True,
        preserve_dump_target_state=True,
        preserve_dump_ion_target_state_only=False,
    )
    prepared_ion_only, ion_boundary_ion_only, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=snapshot.scalar_values,
        apply_sheath_boundaries=True,
        preserve_dump_target_state=True,
        preserve_dump_ion_target_state_only=True,
    )

    lower_guard = (slice(None), mesh.ystart - 1, slice(None))
    np.testing.assert_allclose(
        ion_boundary_ion_only.momentum["d+"][lower_guard],
        prepared_ion_only["d+"].momentum[lower_guard],
    )
    np.testing.assert_allclose(
        ion_boundary_default.momentum["d+"][lower_guard],
        prepared_default["d+"].momentum[lower_guard],
    )
    assert np.any(np.abs(ion_boundary_default.energy_source["d+"]) > 0.0)
    np.testing.assert_allclose(ion_boundary_ion_only.energy_source["d+"], 0.0)

def test_ion_sheath_boundary_reconstructs_velocity_with_density_floor() -> None:
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    species = _initialize_species(config, mesh=mesh)
    ion = species["d+"]

    density = np.asarray(ion.density, dtype=np.float64, copy=True)
    momentum = np.asarray(ion.momentum, dtype=np.float64, copy=True)
    density[:, mesh.yend, :] = 0.5 * ion.density_floor
    momentum[:, mesh.yend, :] = 3.0 * ion.atomic_mass * ion.density_floor
    floored_ion = OpenFieldSpecies(
        **{
            **ion.__dict__,
            "density": density,
            "momentum": momentum,
        }
    )

    electron_density = np.ones_like(density, dtype=np.float64)
    electron_pressure = np.ones_like(density, dtype=np.float64)
    result = _apply_ion_sheath_boundary(
        (floored_ion,),
        electron_pressure=electron_pressure,
        electron_density=electron_density,
        electron_density_floor=species["e"].density_floor,
        mesh=mesh,
        metrics=metrics,
    )

    active = (mesh.xstart, mesh.yend, 0)
    expected_velocity = momentum[active] / (ion.atomic_mass * _soft_floor(density[active], ion.density_floor))
    assert result.velocity["d+"][active] == pytest.approx(expected_velocity)


def test_ion_simple_sheath_energy_source_matches_hermes_formula() -> None:
    mesh = StructuredMesh(
        nx=1,
        ny=1,
        nz=1,
        mxg=0,
        myg=1,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=0,
        jyseps1_2=0,
        jyseps2_2=0,
        ny_inner=1,
        has_lower_y_target=True,
        has_upper_y_target=False,
        x=jnp.arange(1, dtype=jnp.float64),
        y=jnp.arange(3, dtype=jnp.float64) - 1.0,
        z=jnp.arange(1, dtype=jnp.float64),
    )
    ones = jnp.ones((1, 3, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g33=ones,
        g22=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    ion_density = 2.0 * np.ones((1, 3, 1), dtype=np.float64)
    ion_temperature = np.ones((1, 3, 1), dtype=np.float64)
    ion_velocity = np.zeros((1, 3, 1), dtype=np.float64)
    electron_density = 2.0 * np.ones((1, 3, 1), dtype=np.float64)
    electron_pressure = 4.0 * np.ones((1, 3, 1), dtype=np.float64)

    ion = OpenFieldSpecies(
        name="d+",
        density=ion_density,
        pressure=ion_density * ion_temperature,
        momentum=2.0 * ion_density * ion_velocity,
        charge=1.0,
        atomic_mass=2.0,
        density_floor=1.0e-8,
        has_pressure=True,
        has_momentum=True,
        noflow_lower_y=False,
        noflow_upper_y=False,
        target_recycle=False,
        recycle_as=None,
        target_recycle_multiplier=0.0,
        target_recycle_energy=0.0,
        target_fast_recycle_fraction=0.0,
        target_fast_recycle_energy_factor=1.0,
    )
    settings = _SimpleSheathSettings(
        gamma_e=4.5,
        gamma_i=2.5,
        secondary_electron_coef=0.0,
        sheath_ion_polytropic=1.0,
        lower_y=True,
        upper_y=False,
        no_flow=False,
        density_boundary_mode=1.0,
        pressure_boundary_mode=1.0,
        temperature_boundary_mode=1.0,
        wall_potential=np.zeros((1, 3, 1), dtype=np.float64),
    )

    result = _apply_ion_sheath_boundary(
        (ion,),
        electron_pressure=electron_pressure,
        electron_density=electron_density,
        electron_density_floor=1.0e-8,
        mesh=mesh,
        metrics=metrics,
        simple_settings=settings,
    )

    nisheath = 2.0
    tesheath = 2.0
    tisheath = 1.0
    c_i_sq = (settings.sheath_ion_polytropic * tisheath + tesheath) / ion.atomic_mass
    visheath = -np.sqrt(c_i_sq)
    expected_q = settings.gamma_i * tisheath * nisheath * visheath
    expected_q -= (2.5 * tisheath + 0.5 * ion.atomic_mass * visheath * visheath) * nisheath * visheath

    assert result.energy_source["d+"][0, mesh.ystart, 0] == pytest.approx(expected_q)
    assert result.velocity["d+"][0, mesh.ystart - 1, 0] == pytest.approx(2.0 * visheath)


def test_compute_recycling_1d_rhs_uses_boundary_conditioned_electron_velocity_for_pe_rhs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)
    species = _initialize_species(config, mesh=mesh)
    captured: dict[str, np.ndarray] = {}

    _, _, electron_boundary = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
    )

    def _capture_electron_pressure_rhs_terms(
        *,
        explicit_pressure_source: np.ndarray,
        electron_pressure: np.ndarray,
        electron_velocity: np.ndarray,
        electron_fastest_wave: np.ndarray,
        electron_energy_source: np.ndarray,
        mesh,
        metrics,
    ) -> ElectronPressureRhsTerms:
        captured["electron_velocity"] = np.asarray(electron_velocity, dtype=np.float64)
        zeros = np.zeros_like(np.asarray(electron_pressure, dtype=np.float64))
        return ElectronPressureRhsTerms(
            explicit_pressure_source=zeros,
            parallel_divergence=zeros,
            parallel_advection=zeros,
            energy_source=zeros,
            total=zeros,
        )

    monkeypatch.setattr(
        "jax_drb.native.recycling_1d._assemble_electron_pressure_rhs_terms",
        _capture_electron_pressure_rhs_terms,
    )

    compute_recycling_1d_rhs(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
    )

    np.testing.assert_allclose(captured["electron_velocity"], electron_boundary.velocity)


def test_recycling_rhs_uses_boundary_conditioned_density_for_parallel_electric_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)
    species = _initialize_species(config, mesh=mesh)

    prepared, ion_boundary, electron_boundary = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        apply_sheath_boundaries=True,
    )

    sentinel_density = np.full_like(prepared["d+"].density, 7.0, dtype=np.float64)
    sentinel_electron_density = np.full_like(electron_boundary.density, 11.0, dtype=np.float64)
    d_state = prepared["d+"]
    modified_prepared = dict(prepared)
    modified_prepared["d+"] = d_state.__class__(
        density=sentinel_density,
        pressure=d_state.pressure,
        temperature=d_state.temperature,
        velocity=d_state.velocity,
        momentum=species["d+"].atomic_mass * sentinel_density * d_state.velocity,
        momentum_error=d_state.momentum_error,
    )
    modified_electron_boundary = electron_boundary.__class__(
        density=sentinel_electron_density,
        temperature=electron_boundary.temperature,
        pressure=electron_boundary.pressure,
        velocity=electron_boundary.velocity,
        momentum=electron_boundary.momentum,
        energy_source=electron_boundary.energy_source,
    )

    monkeypatch.setattr(
        recycling_1d_mod,
        "_prepare_open_field_states",
        lambda *args, **kwargs: (modified_prepared, ion_boundary, modified_electron_boundary),
    )
    monkeypatch.setattr(
        recycling_1d_mod,
        "_grad_par_electron_force_balance_open",
        lambda pressure, *, mesh, metrics: np.zeros_like(pressure, dtype=np.float64),
    )

    captured_densities: list[np.ndarray] = []

    def capture_parallel_electric_force(density, *, charge, epar):
        captured_densities.append(np.asarray(density, dtype=np.float64))
        return np.zeros_like(np.asarray(density, dtype=np.float64))

    monkeypatch.setattr(recycling_1d_mod, "apply_parallel_electric_force", capture_parallel_electric_force)

    _compute_recycling_1d_rhs_from_species(
        config,
        species=species,
        controllers={},
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=dataset_scalars,
        feedback_integrals=None,
    )

    assert captured_densities
    np.testing.assert_allclose(captured_densities[0], sentinel_density)
    assert not np.allclose(captured_densities[0], species["d+"].density)


def test_multispecies_neutral_charge_exchange_collision_rates_include_cross_isotope_channels() -> None:
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)
    species = _initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    cx_rates = _charge_exchange_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=dataset_scalars,
    )

    active = (mesh.xstart, mesh.ystart, 0)
    d_same = float(prepared["d+"].density[active] * _hydrogen_cx_sigmav(
        np.clip(
            (prepared["d"].temperature / species["d"].atomic_mass + prepared["d+"].temperature / species["d+"].atomic_mass)
            * dataset_scalars["Tnorm"],
            0.01,
            10000.0,
        ),
        dataset_scalars,
    )[active])
    d_cross = float(prepared["t+"].density[active] * _hydrogen_cx_sigmav(
        np.clip(
            (prepared["d"].temperature / species["d"].atomic_mass + prepared["t+"].temperature / species["t+"].atomic_mass)
            * dataset_scalars["Tnorm"],
            0.01,
            10000.0,
        ),
        dataset_scalars,
    )[active])
    t_same = float(prepared["t+"].density[active] * _hydrogen_cx_sigmav(
        np.clip(
            (prepared["t"].temperature / species["t"].atomic_mass + prepared["t+"].temperature / species["t+"].atomic_mass)
            * dataset_scalars["Tnorm"],
            0.01,
            10000.0,
        ),
        dataset_scalars,
    )[active])
    t_cross = float(prepared["d+"].density[active] * _hydrogen_cx_sigmav(
        np.clip(
            (prepared["t"].temperature / species["t"].atomic_mass + prepared["d+"].temperature / species["d+"].atomic_mass)
            * dataset_scalars["Tnorm"],
            0.01,
            10000.0,
        ),
        dataset_scalars,
    )[active])

    assert float(cx_rates["d"][active]) == pytest.approx(d_same + d_cross, rel=1.0e-12, abs=1.0e-12)
    assert float(cx_rates["t"][active]) == pytest.approx(t_same + t_cross, rel=1.0e-12, abs=1.0e-12)


def test_apply_neutral_target_density_guards_extrapolates_boundary_density() -> None:
    mesh = StructuredMesh(
        nx=1,
        ny=2,
        nz=1,
        mxg=0,
        myg=2,
        symmetric_global_x=False,
        symmetric_global_y=False,
        jyseps1_1=0,
        jyseps2_1=1,
        jyseps1_2=1,
        jyseps2_2=1,
        ny_inner=2,
        has_lower_y_target=True,
        has_upper_y_target=True,
        x=np.array([0.0], dtype=np.float64),
        y=np.arange(6, dtype=np.float64),
        z=np.array([0.0], dtype=np.float64),
    )
    field = np.zeros((1, 6, 1), dtype=np.float64)
    field[0, 2, 0] = 1.5
    field[0, 3, 0] = 0.25

    guarded = _apply_neutral_target_density_guards(
        field,
        mesh=mesh,
        lower_y=True,
        upper_y=True,
    )

    assert guarded[0, 1, 0] == pytest.approx(2.75)
    assert guarded[0, 4, 0] == pytest.approx(0.0)


def test_neutral_ionisation_collision_rates_match_reaction_diagnostic_per_density() -> None:
    input_path = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    dataset_scalars = resolved_dataset_scalars(run_config)
    species = _initialize_species(config, mesh=mesh)
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    ionisation_rates = _neutral_ionisation_collision_rates(
        config,
        species=species,
        prepared=prepared,
        dataset_scalars=dataset_scalars,
    )
    reaction_terms = _reaction_sources(
        config,
        species=species,
        electron_density=_electron_density(tuple(sp for sp in species.values() if sp.charge > 0.0)),
        dataset_scalars=dataset_scalars,
    )

    active = (mesh.xstart, mesh.yend, 0)
    expected = float(reaction_terms.diagnostics["Sd+_iz"][active] / species["d"].density[active])
    actual = float(ionisation_rates["d"][active])

    assert actual == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)


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


def test_recycling_source_overrides_replace_total_density_momentum_and_pressure_sources() -> None:
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    shape = (mesh.nx, mesh.local_ny, mesh.nz)
    density_overrides = {
        "d+": np.full(shape, 1.25, dtype=np.float64),
        "d": np.full(shape, -0.75, dtype=np.float64),
    }
    momentum_overrides = {
        "d+": np.full(shape, 2.5, dtype=np.float64),
        "d": np.full(shape, -1.5, dtype=np.float64),
    }
    pressure_overrides = {
        "d+": np.full(shape, 3.5, dtype=np.float64),
        "d": np.full(shape, -2.0, dtype=np.float64),
        "e": np.full(shape, 4.5, dtype=np.float64),
    }
    baseline = compute_recycling_1d_rhs(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )

    result = compute_recycling_1d_rhs(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
        density_source_overrides=density_overrides,
        pressure_source_overrides=pressure_overrides,
        momentum_source_overrides=momentum_overrides,
    )

    active = (mesh.xstart, mesh.ystart, 0)
    assert float(np.asarray(result.variables["SNVd+"])[0][active]) == pytest.approx(2.5)
    assert float(np.asarray(result.variables["SNVd"])[0][active]) == pytest.approx(-1.5)
    assert abs(
        float(np.asarray(result.variables["ddt(Nd+)"])[0][active])
        - float(np.asarray(baseline.variables["ddt(Nd+)"])[0][active])
    ) > 0.5
    assert abs(
        float(np.asarray(result.variables["ddt(Nd)"])[0][active])
        - float(np.asarray(baseline.variables["ddt(Nd)"])[0][active])
    ) > 0.5
    assert abs(
        float(np.asarray(result.variables["ddt(Pe)"])[0][active])
        - float(np.asarray(baseline.variables["ddt(Pe)"])[0][active])
    ) > 1.0


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


def test_backward_euler_implicit_controller_state_does_not_reapply_trapezoid_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=scalars,
    )
    fields = _build_recycling_state_fields(runtime_model)
    integrals = {name: 0.0 for name in runtime_model.feedback_names}
    field_names = runtime_model.field_names
    captured_feedback_timestep: list[float | None] = []

    def fake_rhs_from_species(*args, **kwargs):
        captured_feedback_timestep.append(kwargs.get("feedback_timestep"))
        zeros = {
            f"ddt({name})": np.zeros((1,) + np.asarray(fields[name]).shape, dtype=np.float64)
            for name in field_names
        }
        return SimpleNamespace(variables=zeros, feedback_integral_rhs={name: 0.0 for name in runtime_model.feedback_names})

    def fake_sparse_newton_system(residual, initial_state, **kwargs):
        residual(np.asarray(initial_state, dtype=np.float64))
        return np.asarray(initial_state, dtype=np.float64), SimpleNamespace(
            residual_inf_norm=0.0,
            active_shape=(initial_state.size,),
            nonlinear_iterations=0,
            linear_iterations=0,
        )

    monkeypatch.setattr(recycling_1d_mod, "_compute_recycling_1d_rhs_from_species", fake_rhs_from_species)
    monkeypatch.setattr(recycling_1d_mod, "solve_sparse_newton_system", fake_sparse_newton_system)

    advance_recycling_1d_backward_euler_step(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=25.0,
        solver_mode="sparse",
        evolve_feedback_integrals=True,
    )

    assert captured_feedback_timestep
    assert all(value is None for value in captured_feedback_timestep)


def test_backward_euler_explicit_controller_update_keeps_trapezoid_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_bout_input(_DTHE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=scalars,
    )
    fields = _build_recycling_state_fields(runtime_model)
    integrals = {name: 0.0 for name in runtime_model.feedback_names}
    field_names = runtime_model.field_names
    captured_feedback_timestep: list[float | None] = []

    def fake_rhs_from_species(*args, **kwargs):
        captured_feedback_timestep.append(kwargs.get("feedback_timestep"))
        zeros = {
            f"ddt({name})": np.zeros((1,) + np.asarray(fields[name]).shape, dtype=np.float64)
            for name in field_names
        }
        return SimpleNamespace(variables=zeros, feedback_integral_rhs={name: 0.0 for name in runtime_model.feedback_names})

    def fake_sparse_newton_system(residual, initial_state, **kwargs):
        residual(np.asarray(initial_state, dtype=np.float64))
        return np.asarray(initial_state, dtype=np.float64), SimpleNamespace(
            residual_inf_norm=0.0,
            active_shape=(initial_state.size,),
            nonlinear_iterations=0,
            linear_iterations=0,
        )

    monkeypatch.setattr(recycling_1d_mod, "_compute_recycling_1d_rhs_from_species", fake_rhs_from_species)
    monkeypatch.setattr(recycling_1d_mod, "solve_sparse_newton_system", fake_sparse_newton_system)

    advance_recycling_1d_backward_euler_step(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=25.0,
        solver_mode="sparse",
        evolve_feedback_integrals=False,
    )

    assert captured_feedback_timestep
    assert all(value == 25.0 for value in captured_feedback_timestep)


def test_backward_euler_sparse_solver_uses_explicit_rhs_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=scalars,
    )
    fields = _build_recycling_state_fields(runtime_model)
    field_names = runtime_model.field_names
    integrals = {name: 0.0 for name in runtime_model.feedback_names}
    packed_previous = recycling_1d_mod._pack_recycling_active_state(
        fields,
        feedback_integrals=integrals,
        field_names=field_names,
        feedback_names=(),
        mesh=mesh,
    )
    rhs = np.full_like(packed_previous, 0.25)
    captured_initial_states: list[np.ndarray] = []

    def fake_packed_rhs(*args, **kwargs):
        return rhs

    def fake_sparse_newton_system(residual, initial_state, **kwargs):
        captured_initial_states.append(np.asarray(initial_state, dtype=np.float64))
        return np.asarray(initial_state, dtype=np.float64), SimpleNamespace(
            residual_inf_norm=0.0,
            active_shape=(initial_state.size,),
            nonlinear_iterations=0,
            linear_iterations=0,
        )

    monkeypatch.setattr(recycling_1d_mod, "_compute_recycling_1d_packed_rhs", fake_packed_rhs)
    monkeypatch.setattr(recycling_1d_mod, "solve_sparse_newton_system", fake_sparse_newton_system)

    advance_recycling_1d_backward_euler_step(
        config,
        fields,
        runtime_model=runtime_model,
        feedback_integrals=integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=25.0,
        solver_mode="sparse",
        evolve_feedback_integrals=False,
    )

    assert captured_initial_states
    expected = packed_previous + 25.0 * rhs
    np.testing.assert_allclose(captured_initial_states[0], expected, rtol=1.0e-12, atol=1.0e-12)


def test_bdf2_sparse_solver_uses_explicit_rhs_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=scalars,
    )
    fields = _build_recycling_state_fields(runtime_model)
    previous_fields = {
        name: np.asarray(value, dtype=np.float64, copy=True) * 0.99
        for name, value in fields.items()
    }
    field_names = runtime_model.field_names
    integrals = {name: 0.0 for name in runtime_model.feedback_names}
    packed_previous = recycling_1d_mod._pack_recycling_active_state(
        fields,
        feedback_integrals=integrals,
        field_names=field_names,
        feedback_names=(),
        mesh=mesh,
    )
    rhs = np.full_like(packed_previous, 0.125)
    captured_initial_states: list[np.ndarray] = []

    def fake_packed_rhs(*args, **kwargs):
        return rhs

    def fake_sparse_newton_system(residual, initial_state, **kwargs):
        captured_initial_states.append(np.asarray(initial_state, dtype=np.float64))
        return np.asarray(initial_state, dtype=np.float64), SimpleNamespace(
            residual_inf_norm=0.0,
            active_shape=(initial_state.size,),
            nonlinear_iterations=0,
            linear_iterations=0,
        )

    monkeypatch.setattr(recycling_1d_mod, "_compute_recycling_1d_packed_rhs", fake_packed_rhs)
    monkeypatch.setattr(recycling_1d_mod, "solve_sparse_newton_system", fake_sparse_newton_system)

    advance_recycling_1d_bdf2_step(
        config,
        fields,
        previous_fields,
        runtime_model=runtime_model,
        feedback_integrals=integrals,
        previous_feedback_integrals=integrals,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=25.0,
        solver_mode="sparse",
        evolve_feedback_integrals=False,
    )

    assert captured_initial_states
    expected = packed_previous + 25.0 * rhs
    np.testing.assert_allclose(captured_initial_states[0], expected, rtol=1.0e-12, atol=1.0e-12)


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


@pytest.mark.slow
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


@pytest.mark.slow
def test_recycling_adaptive_be_history_produces_finite_small_step() -> None:
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
        solver_mode="adaptive_be",
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=10,
    )

    assert history.variable_history["Nd+"].shape[0] == 2
    assert np.isfinite(history.variable_history["Nd+"]).all()
    assert np.isfinite(history.variable_history["Pe"]).all()


@pytest.mark.slow
def test_recycling_adaptive_bdf_history_produces_finite_small_step() -> None:
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
        solver_mode="adaptive_bdf",
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=10,
    )

    assert history.variable_history["Nd+"].shape[0] == 2
    assert np.isfinite(history.variable_history["Nd+"]).all()
    assert np.isfinite(history.variable_history["Pe"]).all()


@pytest.mark.slow
def test_recycling_bdf_history_supplies_sparse_jacobian_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling-dthe/data/BOUT.inp")
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    scalars = resolved_dataset_scalars(run_config)

    captured: dict[str, object] = {}

    def fake_solve_ivp(fun, t_span, y0, **kwargs):
        captured["kwargs"] = kwargs
        rhs0 = np.asarray(fun(0.0, y0), dtype=np.float64)
        jacobian = kwargs.get("jac")
        assert callable(jacobian)
        jac0 = jacobian(0.0, y0)
        assert jac0.shape == (y0.size, y0.size)
        return SimpleNamespace(
            success=True,
            message="ok",
            y=np.stack([np.asarray(y0, dtype=np.float64), np.asarray(y0, dtype=np.float64)], axis=1),
        )

    import scipy.integrate

    monkeypatch.setattr(scipy.integrate, "solve_ivp", fake_solve_ivp)

    history = advance_recycling_1d_implicit_history(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=25.0,
        steps=1,
        solver_mode="bdf",
    )

    kwargs = captured["kwargs"]
    assert callable(kwargs["jac"])
    assert kwargs["jac_sparsity"] is not None
    assert history.variable_history["Nd+"].shape[0] == 2


@pytest.mark.slow
def test_recycling_bdf2_step_produces_finite_small_step() -> None:
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
    fields0 = _build_recycling_state_fields(runtime_model)
    integrals0 = {name: 0.0 for name in runtime_model.feedback_names}
    fields1, integrals1, _ = advance_recycling_1d_backward_euler_step(
        config,
        fields0,
        runtime_model=runtime_model,
        feedback_integrals=integrals0,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=25.0,
        solver_mode="sparse",
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=10,
    )
    fields2, integrals2, _ = advance_recycling_1d_bdf2_step(
        config,
        fields1,
        fields0,
        runtime_model=runtime_model,
        feedback_integrals=integrals1,
        previous_feedback_integrals=integrals0,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=25.0,
        solver_mode="sparse",
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=10,
    )

    assert np.isfinite(fields2["Nd+"]).all()
    assert np.isfinite(fields2["Pe"]).all()
    assert np.isfinite(np.asarray(list(integrals2.values()), dtype=np.float64)).all()


@pytest.mark.slow
def test_recycling_backward_euler_advances_feedback_integrals_from_accepted_state() -> None:
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
    fields0 = _build_recycling_state_fields(runtime_model)
    integrals0 = {name: 0.0 for name in runtime_model.feedback_names}
    previous_errors = _current_feedback_errors(fields0, controllers=runtime_model.controllers, mesh=mesh)
    fields1, integrals1, _ = advance_recycling_1d_backward_euler_step(
        config,
        fields0,
        runtime_model=runtime_model,
        feedback_integrals=integrals0,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=25.0,
        solver_mode="sparse",
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=10,
    )
    expected_integrals = _advance_feedback_integrals(
        fields1,
        controllers=runtime_model.controllers,
        feedback_integrals=integrals0,
        feedback_previous_errors=previous_errors,
        mesh=mesh,
        timestep=25.0,
    )

    assert integrals1 == pytest.approx(expected_integrals)


@pytest.mark.slow
def test_recycling_backward_euler_can_evolve_feedback_integrals_in_implicit_state() -> None:
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
    fields0 = _build_recycling_state_fields(runtime_model)
    integrals0 = {name: 0.0 for name in runtime_model.feedback_names}

    _, implicit_integrals, implicit_info = advance_recycling_1d_backward_euler_step(
        config,
        fields0,
        runtime_model=runtime_model,
        feedback_integrals=integrals0,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=25.0,
        solver_mode="sparse",
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=10,
        evolve_feedback_integrals=True,
    )
    _, explicit_integrals, explicit_info = advance_recycling_1d_backward_euler_step(
        config,
        fields0,
        runtime_model=runtime_model,
        feedback_integrals=integrals0,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=scalars,
        timestep=25.0,
        solver_mode="sparse",
        residual_tolerance=1.0e-8,
        max_nonlinear_iterations=10,
        evolve_feedback_integrals=False,
    )

    assert np.isfinite(np.asarray(list(implicit_integrals.values()), dtype=np.float64)).all()
    assert implicit_info.active_size == explicit_info.active_size + len(runtime_model.feedback_names)
