from __future__ import annotations

import numpy as np

from jaxdrb.benchmarking import compute_target_fluxes
from jaxdrb.driver import run_simulation


def _cfg() -> dict:
    return {
        "geometry": {
            "kind": "salpha",
            "nx": 12,
            "ny": 12,
            "nz": 8,
            "Lx": 6.0,
            "Ly": 6.0,
            "Lz": 2.0 * np.pi,
            "bc_x": "periodic",
            "bc_y": "periodic",
            "open_field_line": True,
            "shat": 0.7,
            "alpha": 0.0,
            "q": 1.4,
            "R0": 3.0,
            "epsilon": 0.18,
            "curvature0": 0.18,
            "curvature_model": "vector_xy",
            "theta_scale": 1.0,
        },
        "physics": {
            "nonlinear_on": True,
            "curvature_on": True,
            "curvature_coeff": 1.0,
            "omega_n": 0.08,
            "omega_Te": 0.04,
            "source_on": False,
            "sol_on": True,
            "sol_parallel_loss_on": True,
            "sol_parallel_loss_model": "bohm_exp",
            "sol_parallel_loss_coeff": 0.2,
            "em_on": False,
            "hot_ion_on": False,
            "neutrals_on": False,
            "boussinesq": True,
        },
        "closures": {
            "sheath_on": True,
            "sheath_bc_on": True,
            "sheath_bc_model": "bohm_current",
            "sheath_nu_factor": 1.0,
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
            "parallel_flux_conservative": True,
            "parallel_limiter": "mc",
            "poisson_scale": 1.0,
        },
        "initial": {
            "amplitude": 1.0e-3,
            "seed": 1,
            "noise_mode": "physical",
            "noise_fields": ["n", "omega", "Te"],
        },
        "time": {
            "method": "rk4_imex_strang",
            "dt": 8.0e-4,
            "nsteps": 80,
            "save_every": 4,
            "diag_mode": "full",
            "save_fields": True,
            "snapshot_fields": ["n", "Te", "vpar_i"],
            "return_numpy": True,
        },
    }


def test_sheath_target_flux_sanity():
    out = run_simulation(_cfg(), as_numpy=True).diagnostics
    n = np.asarray(out["snapshots_n"], dtype=np.float64)
    te = np.asarray(out["snapshots_Te"], dtype=np.float64)
    vi = np.asarray(out["snapshots_vpar_i"], dtype=np.float64)

    gamma_t, qe_t, qi_t = compute_target_fluxes(n, vi, te, axis_par=1)

    assert np.all(np.isfinite(gamma_t))
    assert np.all(np.isfinite(qe_t))
    assert np.all(np.isfinite(qi_t))
    assert float(np.mean(gamma_t)) > 0.0
    assert float(np.mean(qe_t)) > 0.0
