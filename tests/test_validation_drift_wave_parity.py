from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.arrays import load_portable_array_payload
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.validation import (
    compare_drift_wave_array_payloads,
    save_drift_wave_parity_plot,
    write_drift_wave_parity_json,
)


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


def _reference_payload():
    return load_portable_array_payload(
        Path(__file__).resolve().parents[1]
        / "references"
        / "baselines"
        / "reference_arrays"
        / "drift_wave_short_window.npz"
    )


def test_compare_drift_wave_identical_payload_has_zero_errors() -> None:
    config, dataset_scalars = _build_validation_config()
    payload = _reference_payload()

    result = compare_drift_wave_array_payloads(
        payload,
        payload,
        config=config,
        dataset_scalars=dataset_scalars,
        fit_points=10,
    )

    assert result.expected.measured_gamma_over_wstar == pytest.approx(result.actual.measured_gamma_over_wstar, rel=0.0, abs=0.0)
    assert result.expected.measured_omega_over_wstar == pytest.approx(result.actual.measured_omega_over_wstar, rel=0.0, abs=0.0)
    assert set(result.variable_errors) >= {"Ni", "NVe", "Vort", "phi"}
    for variable in result.variable_errors.values():
        assert variable.max_abs_error == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert variable.rms_error == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert np.all(variable.max_abs_error_history == 0.0)
        assert np.all(variable.rms_error_history == 0.0)


def test_compare_drift_wave_tracks_density_offset_error() -> None:
    config, dataset_scalars = _build_validation_config()
    expected = _reference_payload()
    actual = {
        **expected,
        "variables": {name: np.asarray(value, dtype=np.float64).copy() for name, value in expected["variables"].items()},
    }
    actual["variables"]["Ni"] += 1.0e-4

    result = compare_drift_wave_array_payloads(
        expected,
        actual,
        config=config,
        dataset_scalars=dataset_scalars,
        fit_points=10,
    )

    assert result.expected.measured_gamma_over_wstar == pytest.approx(result.actual.measured_gamma_over_wstar, rel=1e-12, abs=1e-12)
    assert result.expected.measured_omega_over_wstar == pytest.approx(result.actual.measured_omega_over_wstar, rel=1e-12, abs=1e-12)
    assert result.variable_errors["Ni"].max_abs_error == pytest.approx(1.0e-4, rel=1e-12, abs=1e-12)
    assert result.variable_errors["Ni"].rms_error == pytest.approx(1.0e-4, rel=1e-12, abs=1e-12)
    assert np.allclose(result.variable_errors["Ni"].max_abs_error_history, 1.0e-4)
    assert np.allclose(result.variable_errors["Ni"].rms_error_history, 1.0e-4)
    assert result.variable_errors["NVe"].max_abs_error == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_drift_wave_parity_json_and_plot_outputs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    config, dataset_scalars = _build_validation_config()
    payload = _reference_payload()
    result = compare_drift_wave_array_payloads(
        payload,
        payload,
        config=config,
        dataset_scalars=dataset_scalars,
        fit_points=10,
    )

    json_path = write_drift_wave_parity_json(result, tmp_path / "drift_wave_parity.json")
    plot_path = save_drift_wave_parity_plot(result, tmp_path / "drift_wave_parity.png")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["expected"]["measured_gamma_over_wstar"] == pytest.approx(result.expected.measured_gamma_over_wstar, rel=0.0, abs=0.0)
    assert data["actual"]["measured_omega_over_wstar"] == pytest.approx(result.actual.measured_omega_over_wstar, rel=0.0, abs=0.0)
    assert data["variable_errors"]["Ni"]["max_abs_error"] == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
