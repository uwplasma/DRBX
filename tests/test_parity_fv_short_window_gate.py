from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaxdrb.benchmarking import (
    compute_fluctuation_rms,
    compute_frequency_psd,
    compute_ky_psd,
    finite_run_gate,
)
from jaxdrb.core.state import DRBSystemState
from jaxdrb.driver import build_system_from_config

_REF_PATH = Path(__file__).resolve().parent / "fixtures" / "parity_fv_short_window_reference.npz"


def _cfg() -> dict:
    return {
        "engine": "parity_fv",
        "geometry": {
            "kind": "slab",
            "nx": 10,
            "ny": 8,
            "nz": 9,
            "Lx": 1.0,
            "Ly": 1.2,
            "Lz": 2.1,
            "bxcv_const": 0.35,
            "open_field_line": False,
        },
        "physics": {"source_n0": 0.01},
        "terms": {"parallel_on": True, "curvature_on": True, "sheath_on": False},
        "numerics": {
            "poisson_scale": 1.7,
            "parity_poisson_solver": "spectral_xy",
            "parallel_pressure_flux_coeff": 5.0 / 3.0,
            "parallel_pressure_work_coeff": 2.0 / 3.0,
            "vorticity_parallel_coeff": 1.0,
            "curvature_coeff": 1.0,
        },
        "initial": {"n0": 1.0, "Te0": 1.0, "omega0": 0.0},
    }


def _deterministic_state(shape: tuple[int, int, int]) -> DRBSystemState:
    nz, nx, ny = shape
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz)[:, None, None]
    x = jnp.linspace(0.0, 2.0 * jnp.pi, nx)[None, :, None]
    y = jnp.linspace(0.0, 2.0 * jnp.pi, ny)[None, None, :]

    return DRBSystemState(
        n=1.0 + 0.08 * jnp.sin(z + 0.3 * x) + 0.04 * jnp.cos(2.0 * x - 0.5 * y),
        omega=0.02 * jnp.sin(2.0 * x + y) - 0.015 * jnp.cos(z),
        vpar_e=0.05 * jnp.sin(z) + 0.02 * jnp.cos(x),
        vpar_i=-0.04 * jnp.cos(z - 0.2 * y) + 0.015 * jnp.sin(x),
        Te=1.1 + 0.06 * jnp.cos(1.5 * z - 0.7 * y) + 0.03 * jnp.sin(x + y),
        Ti=None,
        psi=None,
        N=None,
    )


def _state_add(a: DRBSystemState, b: DRBSystemState, scale: float = 1.0) -> DRBSystemState:
    return DRBSystemState(
        n=a.n + scale * b.n,
        omega=a.omega + scale * b.omega,
        vpar_e=a.vpar_e + scale * b.vpar_e,
        vpar_i=a.vpar_i + scale * b.vpar_i,
        Te=a.Te + scale * b.Te,
        Ti=None,
        psi=None,
        N=None,
    )


def _rk4_step(system, state: DRBSystemState, t: float, dt: float) -> DRBSystemState:
    k1 = system.rhs(t, state)
    k2 = system.rhs(t + 0.5 * dt, _state_add(state, k1, 0.5 * dt))
    k3 = system.rhs(t + 0.5 * dt, _state_add(state, k2, 0.5 * dt))
    k4 = system.rhs(t + dt, _state_add(state, k3, dt))
    return DRBSystemState(
        n=state.n + (dt / 6.0) * (k1.n + 2.0 * k2.n + 2.0 * k3.n + k4.n),
        omega=state.omega + (dt / 6.0) * (k1.omega + 2.0 * k2.omega + 2.0 * k3.omega + k4.omega),
        vpar_e=state.vpar_e
        + (dt / 6.0) * (k1.vpar_e + 2.0 * k2.vpar_e + 2.0 * k3.vpar_e + k4.vpar_e),
        vpar_i=state.vpar_i
        + (dt / 6.0) * (k1.vpar_i + 2.0 * k2.vpar_i + 2.0 * k3.vpar_i + k4.vpar_i),
        Te=state.Te + (dt / 6.0) * (k1.Te + 2.0 * k2.Te + 2.0 * k3.Te + k4.Te),
        Ti=None,
        psi=None,
        N=None,
    )


def _collect_short_window() -> dict[str, np.ndarray]:
    built = build_system_from_config(_cfg())
    state = _deterministic_state(built.state.n.shape)
    dt = 0.01
    nsteps = 10
    times = np.linspace(0.0, dt * nsteps, nsteps + 1, dtype=np.float64)

    hist: dict[str, list[np.ndarray]] = {"n": [], "Te": [], "omega": [], "phi": []}
    for istep, tval in enumerate(times):
        _, _, phi, _ = built.system.rhs_terms(float(tval), state)
        hist["n"].append(np.asarray(state.n, dtype=np.float64))
        hist["Te"].append(np.asarray(state.Te, dtype=np.float64))
        hist["omega"].append(np.asarray(state.omega, dtype=np.float64))
        hist["phi"].append(np.asarray(phi, dtype=np.float64))
        if istep < nsteps:
            state = _rk4_step(built.system, state, float(tval), dt)

    data: dict[str, np.ndarray] = {"times": times}
    fluct_diags: dict[str, np.ndarray] = {}
    for field, series in hist.items():
        arr = np.asarray(series, dtype=np.float64)
        _, rms_fluct, _ = compute_fluctuation_rms(arr, equilibrium_mode="t0")
        data[f"rms_{field}_fluct"] = rms_fluct
        fluct_diags[f"rms_{field}_fluct"] = rms_fluct

    passed, reason, growth, peak = finite_run_gate(
        fluct_diags,
        max_growth_factor=12.0,
        max_rms_abs=0.05,
    )
    data["gate_passed"] = np.asarray([1.0 if passed else 0.0], dtype=np.float64)
    data["gate_growth"] = np.asarray([growth], dtype=np.float64)
    data["gate_peak"] = np.asarray([peak], dtype=np.float64)
    data["gate_reason_code"] = np.asarray([0.0 if reason == "ok" else -1.0], dtype=np.float64)

    n_hist = np.asarray(hist["n"], dtype=np.float64)
    n_fluct = n_hist - n_hist[0:1]
    probe = n_fluct[:, n_hist.shape[1] // 2, n_hist.shape[2] // 2, n_hist.shape[3] // 2]
    freq_hz, psd_n_f = compute_frequency_psd(probe, dt=dt, nperseg=8)
    data["freq_hz"] = freq_hz
    data["psd_n_f"] = psd_n_f

    n_plane = n_fluct[-1, n_hist.shape[1] // 2]
    ky_m1, psd_n_ky = compute_ky_psd(n_plane, dy=1.2 / 8.0, axis_y=-1)
    data["ky_m-1"] = ky_m1
    data["psd_n_ky"] = psd_n_ky

    phi_hist = np.asarray(hist["phi"], dtype=np.float64)
    data["n_fluct_last_midz"] = n_plane
    data["phi_fluct_last_midz"] = (phi_hist[-1] - phi_hist[0])[phi_hist.shape[1] // 2]
    return data


def test_parity_fv_short_window_regression_and_gate() -> None:
    with np.load(_REF_PATH) as ref:
        got = _collect_short_window()
        assert set(got) == set(ref.files)
        for key in sorted(got):
            np.testing.assert_allclose(
                got[key],
                np.asarray(ref[key], dtype=np.float64),
                rtol=1e-11,
                atol=1e-12,
                err_msg=f"Mismatch in parity_fv short-window channel: {key}",
            )
        assert float(got["gate_passed"][0]) == 1.0
