from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from jax_drb.config.boutinp import load_bout_input
import jax_drb.native.runner as native_runner
from jax_drb.native.mesh import StructuredMesh, build_structured_mesh
from jax_drb.native.metrics import StructuredMetrics, build_structured_metrics
from jax_drb.native import run_curated_case
from jax_drb.native.reference_dump import load_local_reference_snapshot
from jax_drb.native.recycling_1d import (
    ElectronPressureRhsTerms,
    OpenFieldSpecies,
    _SimpleSheathSettings,
    _assemble_electron_pressure_rhs_terms,
    _advance_feedback_integrals,
    _charge_exchange_collision_rates,
    _compute_collision_frequencies,
    _compute_recycling_1d_packed_rhs,
    _current_feedback_errors,
    _electron_zero_current_velocity,
    _electron_density,
    _grad_par_electron_force_balance_open,
    _build_recycling_runtime_model,
    _build_recycling_state_fields,
    advance_recycling_1d_implicit_history,
    _apply_ion_sheath_boundary,
    _initialize_species,
    _hydrogen_cx_sigmav,
    _load_amjuel_rate,
    _neutral_ionisation_collision_rates,
    _prepare_open_field_states,
    _reaction_sources,
    _recycling_evolving_variable_names,
    _sanitize_recycling_fields,
    _soft_floor,
    _target_recycling_sources,
    _ion_thermal_force_pair,
    advance_recycling_1d_backward_euler_step,
    advance_recycling_1d_bdf2_step,
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
_INPUT_1D = Path("/Users/rogerio/local/hermes-3/tests/integrated/1D-recycling/data/BOUT.inp")


def test_amjuel_rate_tables_are_packaged_for_recycling_branch() -> None:
    hydrogen_iz_coeffs, hydrogen_iz_energy_coeffs, hydrogen_iz_heating = _load_amjuel_rate("d", "iz")
    helium_rec_coeffs, helium_rec_energy_coeffs, helium_rec_heating = _load_amjuel_rate("he", "rec")

    assert hydrogen_iz_coeffs.shape == (9, 9)
    assert helium_rec_coeffs.shape == (9, 9)
    assert hydrogen_iz_energy_coeffs.shape == (9, 9)
    assert helium_rec_energy_coeffs.shape == (9, 9)
    assert np.isfinite(hydrogen_iz_heating)
    assert np.isfinite(helium_rec_heating)


def test_electron_pressure_rhs_terms_sum_to_total() -> None:
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
        has_upper_y_target=True,
        x=jnp.array([0.0], dtype=jnp.float64),
        y=jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float64),
        z=jnp.array([0.0], dtype=jnp.float64),
    )
    ones = jnp.ones((1, 3, 1), dtype=jnp.float64)
    metrics = StructuredMetrics(
        dx=ones,
        dy=ones,
        dz=ones,
        J=ones,
        g11=ones,
        g22=ones,
        g33=ones,
        g_22=ones,
        g23=jnp.zeros_like(ones),
        Bxy=ones,
    )
    explicit = np.full((1, 3, 1), 2.0, dtype=np.float64)
    pressure = np.array([[[3.0], [4.0], [5.0]]], dtype=np.float64)
    velocity = np.array([[[-1.0], [-0.5], [0.0]]], dtype=np.float64)
    fastest_wave = np.full((1, 3, 1), 1.5, dtype=np.float64)
    energy_source = np.full((1, 3, 1), 0.75, dtype=np.float64)

    terms = _assemble_electron_pressure_rhs_terms(
        explicit_pressure_source=explicit,
        electron_pressure=pressure,
        electron_velocity=velocity,
        electron_fastest_wave=fastest_wave,
        electron_energy_source=energy_source,
        mesh=mesh,
        metrics=metrics,
    )

    assert isinstance(terms, ElectronPressureRhsTerms)
    np.testing.assert_allclose(
        terms.total,
        terms.explicit_pressure_source + terms.parallel_divergence + terms.parallel_advection + terms.energy_source,
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
    prepared, _, _ = _prepare_open_field_states(
        species,
        config=config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
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


def test_target_recycling_sources_use_prepared_ion_state() -> None:
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
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    neutrals = tuple(sp for sp in species.values() if sp.charge == 0.0 and sp.name != "e")
    ion_velocity = {ion.name: prepared[ion.name].velocity for ion in ions}

    baseline = _target_recycling_sources(
        ions=ions,
        prepared=prepared,
        neutrals=neutrals,
        ion_velocity=ion_velocity,
        mesh=mesh,
        metrics=metrics,
        gamma_i=2.5,
    )

    distorted_ions = tuple(
        OpenFieldSpecies(
            **{
                **ion.__dict__,
                "density": ion.density * 3.0,
                "pressure": ion.pressure * 5.0,
            }
        )
        for ion in ions
    )
    distorted = _target_recycling_sources(
        ions=distorted_ions,
        prepared=prepared,
        neutrals=neutrals,
        ion_velocity=ion_velocity,
        mesh=mesh,
        metrics=metrics,
        gamma_i=2.5,
    )

    for neutral in ("d",):
        np.testing.assert_allclose(
            distorted.density_source[neutral],
            baseline.density_source[neutral],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            distorted.energy_source[neutral],
            baseline.energy_source[neutral],
            rtol=0.0,
            atol=0.0,
        )


def test_recycling_rhs_passes_configured_sheath_gamma_i_to_target_recycling(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_bout_input(Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-recycling/data/BOUT.inp"))
    run_config = RunConfiguration.from_config(config)
    dataset_scalars = resolved_dataset_scalars(run_config)
    snapshot = load_local_reference_snapshot(
        Path("/Users/rogerio/local/hermes-3/tests/integrated/2D-recycling/data/BOUT.dmp.0.nc"),
        field_names=("Nd+", "Pd+", "NVd+", "Nd", "Pd", "NVd", "Pe"),
        scalar_names=("Nnorm", "Tnorm", "Bnorm", "Cs0", "Omega_ci", "rho_s0"),
    )

    captured: list[float] = []
    original = compute_recycling_1d_rhs.__globals__["compute_target_recycling_sources"]

    def wrapper(*args, **kwargs):
        captured.append(float(kwargs["gamma_i"]))
        return original(*args, **kwargs)

    monkeypatch.setitem(compute_recycling_1d_rhs.__globals__, "compute_target_recycling_sources", wrapper)

    compute_recycling_1d_rhs(
        config,
        mesh=snapshot.mesh,
        metrics=snapshot.metrics,
        dataset_scalars=dataset_scalars,
        field_overrides=snapshot.fields,
        preserve_dump_target_state=True,
    )

    assert captured
    assert all(value == pytest.approx(2.5) for value in captured)


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
    assert np.any(np.abs(ion_boundary_ion_only.energy_source["d+"]) > 0.0)

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


def test_electron_zero_current_velocity_uses_prepared_ion_density() -> None:
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
    ions = tuple(sp for sp in species.values() if sp.charge > 0.0)
    ion_velocity = {ion.name: prepared[ion.name].velocity for ion in ions}
    electron_density = prepared["e"].density

    baseline = _electron_zero_current_velocity(
        ions,
        prepared=prepared,
        ion_velocity=ion_velocity,
        electron_density=electron_density,
    )

    distorted_ions = tuple(
        OpenFieldSpecies(
            **{
                **ion.__dict__,
                "density": ion.density * 4.0,
            }
        )
        for ion in ions
    )
    distorted = _electron_zero_current_velocity(
        distorted_ions,
        prepared=prepared,
        ion_velocity=ion_velocity,
        electron_density=electron_density,
    )

    np.testing.assert_allclose(distorted, baseline, rtol=0.0, atol=0.0)

def test_electron_force_balance_gradient_matches_bout_dy_over_sqrt_g22_stencil() -> None:
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)

    field = np.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    for j in range(mesh.local_ny):
        field[:, j, :] = float(j)

    gradient = _grad_par_electron_force_balance_open(
        field,
        mesh=mesh,
        metrics=metrics,
    )

    dy = np.asarray(metrics.dy, dtype=np.float64)
    g_22 = np.asarray(metrics.g_22, dtype=np.float64)
    expected = np.zeros_like(field, dtype=np.float64)
    for i in range(mesh.xstart, mesh.xend + 1):
        for j in range(mesh.ystart, mesh.yend + 1):
            for k in range(mesh.nz):
                expected[i, j, k] = 0.5 * (field[i, j + 1, k] - field[i, j - 1, k]) / (
                    dy[i, j, k] * np.sqrt(g_22[i, j, k])
                )

    active = (slice(mesh.xstart, mesh.xend + 1), slice(mesh.ystart, mesh.yend + 1), slice(None))
    np.testing.assert_allclose(gradient[active], expected[active], rtol=1.0e-12, atol=1.0e-12)


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


def test_ion_thermal_force_pair_is_enabled_for_dt_when_mass_override_is_set() -> None:
    config = load_bout_input(_DTHE_INPUT)
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

    pair = _ion_thermal_force_pair(
        "d+",
        "t+",
        species=species,
        prepared=prepared,
        mesh=mesh,
        metrics=metrics,
        override_mass_restrictions=True,
    )

    assert pair is not None
    light_name, heavy_name, heavy_force = pair
    active = (mesh.xstart, mesh.yend, 0)

    assert light_name == "d+"
    assert heavy_name == "t+"
    assert np.isfinite(float(heavy_force[active]))
    assert heavy_force.shape == species["t+"].density.shape


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


def test_neutral_pressure_default_floor_is_zero_without_override() -> None:
    config = load_bout_input(_INPUT_1D)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    runtime_model = _build_recycling_runtime_model(
        config,
        mesh=mesh,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    fields = _build_recycling_state_fields(runtime_model)
    fields["Nd"][:] = 1.7e-2
    fields["Pd"][:] = 1.0e-5
    fields["Nd+"][:] = 1.7e-2
    fields["Pd+"][:] = 1.0e-5

    sanitized = _sanitize_recycling_fields(config, fields)

    assert np.allclose(sanitized["Pd"], 1.0e-5)
    assert np.allclose(sanitized["Pd+"], 1.7e-3)


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
