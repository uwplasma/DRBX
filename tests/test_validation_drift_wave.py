from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.arrays import load_portable_array_payload
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.validation import analyze_drift_wave_array_payload, compute_drift_wave_benchmark_scalars


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


def _build_validation_config():
    config = parse_bout_input(_DRIFT_WAVE_INPUT)
    run_config = RunConfiguration.from_config(config)
    return config, resolved_dataset_scalars(run_config)


def test_analyze_drift_wave_tracks_synthetic_growth_and_frequency() -> None:
    config, dataset_scalars = _build_validation_config()
    benchmark = compute_drift_wave_benchmark_scalars(config, dataset_scalars=dataset_scalars)

    nt = 12
    nz = 32
    n0 = 1.0
    gamma = 0.2 * benchmark.wstar
    omega = 0.3 * benchmark.wstar
    time_seconds = np.linspace(0.0, 2.0e-7, nt)
    angle = 2.0 * np.pi * np.arange(nz, dtype=np.float64) / float(nz)
    phase = omega * time_seconds[:, None]
    amplitude = 1.0e-3 * np.exp(gamma * time_seconds[:, None])
    density = n0 + amplitude * np.cos(angle[None, :] - phase)
    payload = {
        "time_points": (time_seconds * float(dataset_scalars["Omega_ci"])).tolist(),
        "variables": {"Ni": density[:, None, None, :]},
    }

    result = analyze_drift_wave_array_payload(
        payload,
        config=config,
        dataset_scalars=dataset_scalars,
        fit_points=6,
    )

    assert result.fit_points == 6
    assert result.equilibrium_density == pytest.approx(n0, rel=0.0, abs=1e-12)
    assert result.measured_gamma_over_wstar == pytest.approx(0.2, rel=5e-3, abs=5e-4)
    assert result.measured_omega_over_wstar == pytest.approx(0.3, rel=5e-3, abs=5e-4)


def test_analyze_drift_wave_matches_committed_short_window_baseline() -> None:
    config, dataset_scalars = _build_validation_config()
    payload = load_portable_array_payload(
        Path(__file__).resolve().parents[1]
        / "references"
        / "baselines"
        / "reference_arrays"
        / "drift_wave_short_window.npz"
    )

    result = analyze_drift_wave_array_payload(
        payload,
        config=config,
        dataset_scalars=dataset_scalars,
        fit_points=10,
    )

    assert result.fit_points == 10
    assert result.benchmark.sigmapar_over_wstar == pytest.approx(1.0542443560168462, rel=1e-10, abs=1e-12)
    assert result.benchmark.analytic_gamma_over_wstar == pytest.approx(0.2861001063781666, rel=1e-10, abs=1e-12)
    assert result.benchmark.analytic_omega_over_wstar == pytest.approx(0.22863593640793803, rel=1e-10, abs=1e-12)
    assert result.measured_gamma_over_wstar == pytest.approx(0.27478899792606437, rel=1e-9, abs=1e-12)
    assert result.measured_omega_over_wstar == pytest.approx(0.23224315136107215, rel=1e-9, abs=1e-12)
