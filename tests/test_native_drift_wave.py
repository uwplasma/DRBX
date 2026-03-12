from __future__ import annotations

from pathlib import Path

import numpy as np

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.drift_wave import (
    DriftWaveState,
    _assemble_density_field,
    _div_par_fvv_periodic,
    _div_par_periodic,
    _div_par_scalar_periodic,
    _electron_ion_collision_frequency,
    _grad_par_periodic,
    advance_drift_wave_history_adaptive,
    build_drift_wave_benchmark,
    compute_drift_wave_rhs,
    initialize_drift_wave_state,
)
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.arrays import load_portable_array_payload
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.validation import analyze_drift_wave_array_payload


_DRIFT_WAVE_INPUT = """
nout = 50
timestep = 10

[mesh]
nx = 5
ny = 32
nz = 27

Lx = 0.01
Ly = 10
Lz = 0.01

B = 0.2
inv_Ln = 10
ixseps1 = nx
ixseps2 = nx

dr = Lx / (nx - 4)
dx = dr * B
dy = Ly / ny
dz = Lz / nz

g11 = B^2
g22 = 1
g33 = 1
J = 1 / B

[mesh:paralleltransform]
type = identity

[solver]
mxstep = 10000

[model]
components = (i, e, vorticity, sound_speed, braginskii_collisions, braginskii_friction, braginskii_heat_exchange)

[vorticity]
diamagnetic = false
diamagnetic_polarisation = false
average_atomic_mass = 1
bndry_flux = false
poloidal_flows = false

[vorticity:laplacian]
inner_boundary_flags = 2
outer_boundary_flags = 2

[i]
type = evolve_density, fixed_velocity, fixed_temperature
charge = 1
AA = 1
velocity = 0
temperature = 100

[Ni]
function = 1 + 1e-3 * sin(z - y)
bndry_xin = neumann(mesh:inv_Ln * units:meters^2 * units:Tesla / mesh:B)
bndry_xout = neumann(mesh:inv_Ln * units:meters^2 * units:Tesla / mesh:B)

[e]
type = quasineutral, evolve_momentum, fixed_temperature
charge = -1
AA = 1/1836
temperature = 100
"""


def _build_case():
    config = parse_bout_input(_DRIFT_WAVE_INPUT)
    run_config = RunConfiguration.from_config(config)
    mesh = build_structured_mesh(config, run_config)
    metrics = build_structured_metrics(config, run_config, mesh)
    benchmark = build_drift_wave_benchmark(
        config,
        mesh=mesh,
        metrics=metrics,
        dataset_scalars=resolved_dataset_scalars(run_config),
    )
    state = initialize_drift_wave_state(config, mesh=mesh)
    return config, run_config, mesh, benchmark, state


def _trim_active_cells(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)[:, 2:3, 2:-2, :]


def test_adaptive_drift_wave_one_step_matches_locked_amplitudes() -> None:
    _, run_config, mesh, benchmark, state = _build_case()
    history = advance_drift_wave_history_adaptive(
        state,
        mesh=mesh,
        benchmark=benchmark,
        timestep=run_config.time.timestep,
        steps=1,
        rtol=1e-6,
        atol=1e-8,
        max_step=1.0,
        initial_step=0.25,
    )

    trimmed_momentum = _trim_active_cells(history.electron_momentum_history)
    trimmed_vorticity = _trim_active_cells(history.vorticity_history)
    trimmed_phi = _trim_active_cells(history.potential_history)

    assert np.isclose(float(trimmed_momentum[-1].min()), -6.303724e-06, rtol=5e-4, atol=1e-9)
    assert np.isclose(float(trimmed_momentum[-1].max()), 6.303724e-06, rtol=5e-4, atol=1e-9)
    assert np.isclose(float(trimmed_vorticity[-1].min()), -3.713728e-05, rtol=3e-3, atol=1e-8)
    assert np.isclose(float(trimmed_vorticity[-1].max()), 3.699342e-05, rtol=3e-3, atol=1e-8)
    assert np.isclose(float(trimmed_phi[-1].min()), -3.596485e-06, rtol=5e-4, atol=1e-9)
    assert np.isclose(float(trimmed_phi[-1].max()), 3.598159e-06, rtol=5e-4, atol=1e-9)


def test_adaptive_full_drift_wave_branch_stays_bounded_over_short_probe() -> None:
    _, run_config, mesh, benchmark, state = _build_case()
    history = advance_drift_wave_history_adaptive(
        state,
        mesh=mesh,
        benchmark=benchmark,
        timestep=run_config.time.timestep,
        steps=10,
        rtol=1e-6,
        atol=1e-8,
        max_step=1.0,
        initial_step=0.25,
        include_parallel_transport=True,
        include_phi_dissipation=True,
    )

    trimmed_momentum = _trim_active_cells(history.electron_momentum_history)
    trimmed_vorticity = _trim_active_cells(history.vorticity_history)
    trimmed_phi = _trim_active_cells(history.potential_history)

    assert np.isfinite(trimmed_momentum).all()
    assert np.isfinite(trimmed_vorticity).all()
    assert np.isfinite(trimmed_phi).all()
    assert float(np.max(np.abs(trimmed_momentum[-1]))) < 1.0e-3
    assert float(np.max(np.abs(trimmed_vorticity[-1]))) < 5.0e-2
    assert float(np.max(np.abs(trimmed_phi[-1]))) < 1.0e-3


def test_locked_one_step_parallel_terms_stay_small_against_drive_terms() -> None:
    _, _, _, benchmark, _ = _build_case()
    arrays = np.load(
        Path(__file__).resolve().parents[1]
        / "references"
        / "baselines"
        / "reference_arrays"
        / "drift_wave_one_step.npz"
    )
    state = DriftWaveState(
        ion_density=np.asarray(arrays["var__Ni"][-1, 0], dtype=np.float64),
        electron_momentum=np.asarray(arrays["var__NVe"][-1, 0], dtype=np.float64),
        vorticity=np.asarray(arrays["var__Vort"][-1, 0], dtype=np.float64),
    )
    phi = np.asarray(arrays["var__phi"][-1, 0], dtype=np.float64)

    electron_density = state.ion_density
    electron_density_limited = np.maximum(electron_density, benchmark.density_floor)
    electron_pressure = electron_density * benchmark.electron_temperature
    electron_velocity = state.electron_momentum / (benchmark.electron_atomic_mass * electron_density_limited)
    collision_frequency = _electron_ion_collision_frequency(electron_density, benchmark=benchmark)

    pressure_term = -_grad_par_periodic(electron_pressure, benchmark=benchmark)
    divjpar_term = _div_par_periodic(
        (benchmark.electron_charge / benchmark.electron_atomic_mass) * state.electron_momentum,
        benchmark=benchmark,
    )
    parflux_term = -benchmark.electron_atomic_mass * _div_par_fvv_periodic(
        electron_density_limited,
        electron_velocity,
        benchmark.fastest_wave,
        benchmark=benchmark,
    )
    phi_dissipation_term = -_div_par_scalar_periodic(-phi, benchmark.sound_speed, benchmark=benchmark)
    collision_term = -benchmark.momentum_coefficient * collision_frequency * state.electron_momentum

    assert np.isclose(float(np.max(np.abs(pressure_term))), 6.37698e-07, rtol=2e-3, atol=1e-10)
    assert np.isclose(float(np.max(np.abs(divjpar_term))), 7.37425e-06, rtol=2e-3, atol=1e-10)
    assert np.isclose(float(np.max(np.abs(parflux_term))), 1.88078e-08, rtol=5e-2, atol=5e-11)
    assert np.isclose(float(np.max(np.abs(phi_dissipation_term))), 4.92867e-10, rtol=5e-2, atol=5e-12)
    assert np.isclose(float(np.max(np.abs(collision_term))), 1.33849e-08, rtol=5e-2, atol=5e-11)


def test_density_boundary_reconstruction_uses_gradient_times_dx() -> None:
    _, _, mesh, benchmark, state = _build_case()
    field = _assemble_density_field(state.ion_density, benchmark=benchmark, mesh=mesh)
    active = np.asarray(state.ion_density, dtype=np.float64)

    assert np.isclose(float(benchmark.density_gradient_inner[0, 0] * benchmark.dx), 0.1, rtol=1e-12, atol=1e-12)
    assert np.allclose(field[mesh.xstart, mesh.ystart : mesh.yend + 1, :], active, rtol=1e-12, atol=1e-12)
    assert np.allclose(
        field[mesh.xstart - 1, mesh.ystart : mesh.yend + 1, :],
        active - 0.1,
        rtol=1e-8,
        atol=1e-8,
    )
    assert np.allclose(
        field[mesh.xend + 1, mesh.ystart : mesh.yend + 1, :],
        active + 0.1,
        rtol=1e-8,
        atol=1e-8,
    )


def test_evolved_state_rhs_tracks_reference_diagnostics() -> None:
    _, _, mesh, benchmark, _ = _build_case()
    diagnostics = load_portable_array_payload(
        Path(__file__).resolve().parents[1]
        / "references"
        / "baselines"
        / "reference_arrays"
        / "drift_wave_one_step_diagnostics.npz"
    )
    variables = diagnostics["variables"]
    state = DriftWaveState(
        ion_density=np.asarray(variables["Ni"][-1, 0], dtype=np.float64),
        electron_momentum=np.asarray(variables["NVe"][-1, 0], dtype=np.float64),
        vorticity=np.asarray(variables["Vort"][-1, 0], dtype=np.float64),
    )
    rhs = compute_drift_wave_rhs(state, mesh=mesh, benchmark=benchmark)

    density_ref = np.asarray(variables["ddt(Ni)"][-1, 0], dtype=np.float64)
    momentum_ref = np.asarray(variables["ddt(NVe)"][-1, 0], dtype=np.float64)
    vorticity_ref = np.asarray(variables["ddt(Vort)"][-1, 0], dtype=np.float64)

    density_diff = np.asarray(rhs.density, dtype=np.float64) - density_ref
    momentum_diff = np.asarray(rhs.momentum, dtype=np.float64) - momentum_ref
    vorticity_diff = np.asarray(rhs.vorticity, dtype=np.float64) - vorticity_ref

    assert float(np.max(np.abs(density_diff))) < 3.0e-7
    assert float(np.corrcoef(np.ravel(rhs.density), np.ravel(density_ref))[0, 1]) > 0.75
    assert float(np.max(np.abs(momentum_diff))) < 5.0e-9
    assert float(np.max(np.abs(vorticity_diff))) < 5.0e-10


def test_adaptive_reduced_drift_wave_short_window_matches_benchmark_scalars() -> None:
    config, run_config, mesh, benchmark, state = _build_case()
    dataset_scalars = resolved_dataset_scalars(run_config)
    history = advance_drift_wave_history_adaptive(
        state,
        mesh=mesh,
        benchmark=benchmark,
        timestep=run_config.time.timestep,
        steps=run_config.time.nout,
        rtol=1e-6,
        atol=1e-8,
        max_step=1.0,
        initial_step=0.25,
        include_parallel_transport=False,
        include_phi_dissipation=False,
    )
    payload = {
        "time_points": [run_config.time.timestep * index for index in range(run_config.time.nout + 1)],
        "variables": {
            "Ni": _trim_active_cells(history.ion_density_history),
            "Ne": _trim_active_cells(history.ion_density_history),
            "NVe": _trim_active_cells(history.electron_momentum_history),
            "Vort": _trim_active_cells(history.vorticity_history),
            "phi": _trim_active_cells(history.potential_history),
        },
    }
    result = analyze_drift_wave_array_payload(
        payload,
        config=config,
        dataset_scalars=dataset_scalars,
        fit_points=10,
    )

    assert np.isclose(result.measured_gamma_over_wstar, 0.27478899792606437, rtol=1e-2, atol=2e-3)
    assert np.isclose(result.measured_omega_over_wstar, 0.23224315136107215, rtol=2e-2, atol=3e-3)
