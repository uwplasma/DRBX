from __future__ import annotations

import numpy as np

from jaxdrb.benchmarking import finite_run_gate
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
            "em_on": False,
            "hot_ion_on": False,
            "neutrals_on": False,
            "boussinesq": True,
        },
        "closures": {
            "sol": {
                "sol_on": True,
                "sol_parallel_loss_on": True,
                "sol_parallel_loss_model": "bohm_exp",
                "sol_parallel_loss_coeff": 0.2,
                "sol_sheath_phi_on": True,
                "sol_sheath_phi_dissipation_on": True,
            },
            "sheath": {
                "sheath_on": True,
                "sheath_bc_on": True,
                "sheath_bc_model": "bohm_current",
                "sheath_nu_factor": 1.0,
            },
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
        "initial": {
            "amplitude": 2.0e-3,
            "seed": 2,
            "noise_mode": "physical",
            "noise_fields": ["n", "omega", "Te"],
        },
        "time": {
            "method": "rk4_imex_strang",
            "dt": 8.0e-4,
            "nsteps": 20,
            "save_every": 2,
            "diag_mode": "full",
            "save_fields": True,
            "snapshot_fields": ["n", "omega", "Te", "phi"],
            "return_numpy": True,
            "progress": False,
        },
    }


def test_open_field_physics_gate():
    out = run_simulation(_cfg(), as_numpy=True).diagnostics
    passed, reason, growth, peak = finite_run_gate(
        {
            "rms_n_fluct": np.asarray(out["rms_n_fluct"]),
            "rms_Te_fluct": np.asarray(out["rms_Te_fluct"]),
            "rms_omega_fluct": np.asarray(out["rms_omega_fluct"]),
            "rms_phi_fluct": np.asarray(out["rms_phi_fluct"]),
        },
        max_growth_factor=300.0,
        max_rms_abs=50.0,
    )
    assert passed, (reason, growth, peak)

    n = np.asarray(out["rms_n_fluct"])
    assert np.all(np.isfinite(n))
    assert float(np.mean(n)) > 1e-6
