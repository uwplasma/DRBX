from __future__ import annotations

import numpy as np

from jaxdrb.benchmarking import finite_run_gate
from jaxdrb.driver import run_simulation


def _cfg() -> dict:
    return {
        "geometry": {
            "kind": "plane",
            "nx": 24,
            "ny": 24,
            "Lx": 30.0,
            "Ly": 20.0,
            "bc_x": "periodic",
            "bc_y": "periodic",
        },
        "physics": {
            "nonlinear_on": True,
            "curvature_on": True,
            "curvature_model": "uniform",
            "curvature_coeff": 0.05,
            "omega_n": 0.06,
            "omega_Te": 0.02,
            "source_on": True,
            "source_profile": "gaussian_xy",
            "source_x0": 15.0,
            "source_y0": 10.0,
            "source_width_x": 4.0,
            "source_width_y": 4.0,
            "source_n0": 0.02,
            "source_Te0": 0.01,
            "sol_on": False,
            "em_on": False,
            "hot_ion_on": False,
            "neutrals_on": False,
            "boussinesq": True,
        },
        "transport": {
            "Dn": 2.0e-3,
            "DOmega": 2.0e-3,
            "DTe": 2.0e-3,
            "Dn4": 2.0e-6,
            "DOmega4": 2.0e-6,
            "DTe4": 2.0e-6,
        },
        "numerics": {
            "poisson": "spectral",
            "bracket": "arakawa",
            "poisson_scale": 1.0,
        },
        "closures": {
            "sheath_on": False,
            "sheath_bc_on": False,
        },
        "initial": {
            "amplitude": 2.0e-3,
            "seed": 2,
            "noise_mode": "physical",
            "noise_fields": ["n", "omega", "Te"],
        },
        "time": {
            "method": "rk4_scan",
            "dt": 2.0e-3,
            "nsteps": 140,
            "save_every": 2,
            "diag_mode": "full",
            "save_fields": True,
            "snapshot_fields": ["n", "Te", "omega", "phi"],
            "return_numpy": True,
        },
    }


def test_nonlinear_stats_window_finite_and_nontrivial():
    out = run_simulation(_cfg(), as_numpy=True).diagnostics
    passed, reason, growth, peak = finite_run_gate(
        {
            "rms_n_fluct": np.asarray(out["rms_n_fluct"]),
            "rms_Te_fluct": np.asarray(out["rms_Te_fluct"]),
            "rms_omega_fluct": np.asarray(out["rms_omega_fluct"]),
            "rms_phi_fluct": np.asarray(out["rms_phi_fluct"]),
        },
        max_growth_factor=500.0,
        max_rms_abs=100.0,
    )
    assert passed, (reason, growth, peak)

    n = np.asarray(out["rms_n_fluct"])
    start = n.size // 2
    window = n[start:]
    assert np.all(np.isfinite(window))
    assert float(np.mean(window)) > 1e-6
    assert float(np.std(window)) > 1e-8
