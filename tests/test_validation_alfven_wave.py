from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from jax_drb.config.boutinp import load_bout_input, parse_bout_input
from jax_drb.native.units import resolved_dataset_scalars
from jax_drb.parity.arrays import load_portable_array_payload
from jax_drb.runtime.run_config import RunConfiguration
from jax_drb.validation import (
    analyze_alfven_wave_array_payload,
    compare_alfven_wave_array_payloads,
    compute_alfven_wave_benchmark_scalars,
)


_ALFVEN_WAVE_INPUT = """
nout = 10
timestep = 10

[mesh]
nx = 5
ny = 32
nz = 27
Lx = 0.1
Ly = 10
Lz = 1
B = 0.2
dx = Lx / (nx - 4)
dy = Ly / ny
dz = Lz / nz
g11 = 1
g22 = 1
g33 = 1
J = 1

[mesh:paralleltransform]
type = identity

[model]
components = (e, i, electromagnetic, vorticity)
Nnorm = 1e19
Tnorm = 100
Bnorm = 0.2

[e]
AA = 1/1836

[i]
AA = 2
density = 1e19
"""


def _build_validation_config():
    config = parse_bout_input(_ALFVEN_WAVE_INPUT)
    run_config = RunConfiguration.from_config(config)
    return config, resolved_dataset_scalars(run_config)


def _load_committed_reference_config():
    relative = Path("tests/integrated/alfven-wave/data/BOUT.inp")
    reference_root = os.environ.get("JAX_DRB_REFERENCE_ROOT")
    candidates = []
    if reference_root:
        candidates.append(Path(reference_root) / relative)
    candidates.append(
        Path(__file__).resolve().parent / "fixtures" / "reference-root" / relative
    )
    input_path = next((path for path in candidates if path.exists()), None)
    if input_path is None:
        pytest.skip("alfven-wave reference input is unavailable")
    config = load_bout_input(input_path)
    run_config = RunConfiguration.from_config(config)
    return config, resolved_dataset_scalars(run_config)


def test_analyze_alfven_wave_tracks_synthetic_frequency() -> None:
    config, dataset_scalars = _build_validation_config()
    benchmark = compute_alfven_wave_benchmark_scalars(config, dataset_scalars=dataset_scalars)

    nt = 16
    ny = 32
    nz = 27
    omega = 0.8 * benchmark.analytic_omega
    time_seconds = np.linspace(0.0, 3.0 * (2.0 * np.pi / omega), nt)
    phase = omega * time_seconds[:, None, None]
    field = 1.0e-3 * np.cos(phase) * np.ones((nt, ny, nz), dtype=np.float64)
    payload = {
        "time_points": (time_seconds * float(dataset_scalars["Omega_ci"])).tolist(),
        "variables": {"phi": field[:, None, :, :]},
    }

    result = analyze_alfven_wave_array_payload(
        payload,
        config=config,
        dataset_scalars=dataset_scalars,
        field_variable="phi",
        x_index=0,
    )

    assert result.field_variable == "phi"
    assert result.measured_omega == pytest.approx(omega, rel=2e-2, abs=1e-6)
    assert result.measured_phase_speed == pytest.approx(omega / benchmark.kpar, rel=2e-2, abs=1e-6)


def test_analyze_alfven_wave_matches_committed_short_window_baseline() -> None:
    reference_npz = (
        Path(__file__).resolve().parents[1]
        / "references"
        / "baselines"
        / "reference_arrays"
        / "alfven_wave_short_window.npz"
    )
    if not reference_npz.exists():
        pytest.skip("Committed Alfven-wave short-window baseline is unavailable")

    config, dataset_scalars = _load_committed_reference_config()
    payload = load_portable_array_payload(reference_npz)
    result = analyze_alfven_wave_array_payload(
        payload,
        config=config,
        dataset_scalars=dataset_scalars,
        field_variable="phi",
        x_index=2,
    )

    assert result.benchmark.analytic_phase_speed > 0.0
    assert result.measured_phase_speed > 0.0
    assert result.relative_phase_speed_error < 5e-2


def test_analyze_alfven_wave_matches_committed_medium_window_baseline() -> None:
    reference_npz = (
        Path(__file__).resolve().parents[1]
        / "references"
        / "baselines"
        / "reference_arrays"
        / "alfven_wave_medium_window.npz"
    )
    if not reference_npz.exists():
        pytest.skip("Committed Alfven-wave medium-window baseline is unavailable")

    config, dataset_scalars = _load_committed_reference_config()
    payload = load_portable_array_payload(reference_npz)
    result = analyze_alfven_wave_array_payload(
        payload,
        config=config,
        dataset_scalars=dataset_scalars,
        field_variable="phi",
        x_index=2,
    )

    assert result.benchmark.analytic_phase_speed > 0.0
    assert result.measured_phase_speed > 0.0
    assert result.relative_phase_speed_error < 2e-2


def test_compare_alfven_wave_array_payloads_reports_small_history_errors() -> None:
    config, dataset_scalars = _build_validation_config()
    time_seconds = np.linspace(0.0, 4.0e-7, 8)
    history = np.cos(np.linspace(0.0, 4.0 * np.pi, 8))[:, None, None, None]
    expected = {
        "time_points": (time_seconds * float(dataset_scalars["Omega_ci"])).tolist(),
        "variables": {"phi": history},
    }
    actual = {
        "time_points": expected["time_points"],
        "variables": {"phi": history * 1.001},
    }

    result = compare_alfven_wave_array_payloads(
        expected,
        actual,
        config=config,
        dataset_scalars=dataset_scalars,
        field_variable="phi",
        x_index=0,
    )

    assert result.mean_square_max_abs_error > 0.0
    assert result.mean_square_rms_error > 0.0
