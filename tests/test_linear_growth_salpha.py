from __future__ import annotations

import numpy as np

from jaxdrb.driver import run_simulation


def _make_cfg() -> dict:
    return {
        "geometry": {
            "kind": "axisymmetric_analytic",
            "model": "salpha",
            "nx": 8,
            "ny": 8,
            "nz": 8,
            "Lx": 1.0,
            "Ly": 1.0,
            "Lz": 2 * np.pi,
            "q": 2.0,
            "R0": 2.0,
            "r0": 0.2,
            "epsilon": 0.1,
            "shat": 1.0,
            "alpha": 0.0,
            "curvature_model": "logB",
            "epsilon_x_grad": 0.5,
            "theta_ballooning_on": True,
            "linear_shear_on": True,
            "theta_scale": 2.0,
            "open_field_line": False,
            "bc_x": "periodic",
            "bc_y": "periodic",
        },
        "physics": {
            "omega_n": 30.0,
            "nonlinear_on": False,
            "em_on": False,
            "hot_ion_on": False,
            "neutrals_on": False,
            "boussinesq": True,
            "curvature_on": True,
            "curvature_coeff": 1.0,
        },
        "closures": {
            "sheath": {
                "sheath_on": False,
                "sheath_bc_on": False,
            }
        },
        "numerics": {
            "poisson": "spectral",
            "bracket": "arakawa",
        },
        "transport": {
            "Dn": 0.0,
            "DOmega": 0.0,
            "DTe": 0.0,
        },
        "initial": {"amplitude": 1e-3, "seed": 0},
        "time": {
            "method": "rk4_scan",
            "dt": 1e-2,
            "nsteps": 200,
            "save_every": 5,
            "diag_mode": "basic",
            "poisson_warm_start": True,
            "return_numpy": True,
        },
    }


def _growth_rate(times: np.ndarray, series: np.ndarray, start: int, end: int) -> float:
    t = np.asarray(times[start:end], dtype=np.float64)
    y = np.asarray(series[start:end], dtype=np.float64)
    y = np.clip(y, 1e-12, None)
    coeff = np.polyfit(t, np.log(y), 1)
    return float(coeff[0])


def test_linear_growth_salpha_small() -> None:
    result = run_simulation(_make_cfg())
    times = np.asarray(result.diagnostics["t"], dtype=np.float64)
    rms_n = np.asarray(result.diagnostics["rms_n"], dtype=np.float64)

    # Use a mid-time window to avoid transient startup.
    gamma = _growth_rate(times, rms_n, start=20, end=30)

    assert gamma > 0.2
    assert gamma < 20.0
