from __future__ import annotations

import numpy as np

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.drift_wave import (
    advance_drift_wave_history_adaptive,
    build_drift_wave_benchmark,
    initialize_drift_wave_state,
)
from jax_drb.native.mesh import build_structured_mesh
from jax_drb.native.metrics import build_structured_metrics
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.runtime.run_config import RunConfiguration


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
    return run_config, mesh, benchmark, state


def _trim_active_cells(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)[:, 2:3, 2:-2, :]


def test_adaptive_drift_wave_one_step_matches_locked_amplitudes() -> None:
    run_config, mesh, benchmark, state = _build_case()
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
    assert np.isclose(float(trimmed_vorticity[-1].min()), -3.713728e-05, rtol=5e-4, atol=1e-8)
    assert np.isclose(float(trimmed_vorticity[-1].max()), 3.699342e-05, rtol=5e-4, atol=1e-8)
    assert np.isclose(float(trimmed_phi[-1].min()), -3.596485e-06, rtol=5e-4, atol=1e-9)
    assert np.isclose(float(trimmed_phi[-1].max()), 3.598159e-06, rtol=5e-4, atol=1e-9)


def test_adaptive_full_drift_wave_branch_stays_bounded_over_short_probe() -> None:
    run_config, mesh, benchmark, state = _build_case()
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
