from __future__ import annotations

import numpy as np

from jaxdrb.driver import run_simulation


def _make_cfg(*, warm_start: bool) -> dict:
    return {
        "physics": {
            "boussinesq": False,
            "nonlinear_on": False,
        },
        "closures": {
            "sheath": {
                "sheath_on": False,
                "sheath_bc_on": False,
            }
        },
        "numerics": {
            "poisson": "cg_fd",
            "poisson_cg_maxiter": 200,
            "poisson_cg_tol": 1e-10,
            "poisson_cg_atol": 0.0,
            "polarization_preconditioner": "jacobi",
        },
        "geometry": {
            "kind": "plane",
            "nx": 8,
            "ny": 8,
            "Lx": float(2 * np.pi),
            "Ly": float(2 * np.pi),
            "bc_x": "periodic",
            "bc_y": "periodic",
            "dealias": False,
        },
        "initial": {"amplitude": 1e-3},
        "time": {
            "method": "rk4_scan",
            "dt": 1e-2,
            "nsteps": 20,
            "save_every": 5,
            "diag_mode": "basic",
            "poisson_track_iters": True,
            "poisson_warm_start": warm_start,
        },
    }


def test_poisson_iter_stats_warm_start_reduces_mean() -> None:
    cold = run_simulation(_make_cfg(warm_start=False))
    warm = run_simulation(_make_cfg(warm_start=True))

    mean_cold = float(cold.diagnostics["poisson_iters_mean_all"])
    mean_warm = float(warm.diagnostics["poisson_iters_mean_all"])

    assert mean_cold > 0.0
    assert mean_warm <= mean_cold
