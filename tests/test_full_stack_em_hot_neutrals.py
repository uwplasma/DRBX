from __future__ import annotations

import numpy as np

from jaxdrb.benchmarking import finite_run_gate
from jaxdrb.driver import run_simulation


def _cfg() -> dict:
    return {
        "geometry": {
            "kind": "salpha",
            "nx": 10,
            "ny": 10,
            "nz": 6,
            "Lx": 4.0,
            "Ly": 4.0,
            "Lz": 2.0 * np.pi,
            "bc_x": "periodic",
            "bc_y": "periodic",
            "open_field_line": False,
            "shat": 0.6,
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
            "omega_n": 0.06,
            "omega_Te": 0.03,
            "omega_Ti": 0.02,
            "em_on": True,
            "beta": 1.0e-3,
            "hot_ion_on": True,
            "neutrals_on": True,
            "boussinesq": True,
        },
        "transport": {
            "Dn": 2.0e-3,
            "DOmega": 2.0e-3,
            "DTe": 2.0e-3,
            "DTi": 2.0e-3,
            "Dpsi": 1.0e-3,
            "Dvpar": 1.0e-3,
        },
        "closures": {
            "neutrals": {
                "enabled": True,
                "Dn0": 1.0e-2,
                "nu_ion": 0.4,
                "nu_rec": 0.2,
                "nu_cx_omega": 0.1,
            }
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
            "seed": 4,
            "noise_mode": "physical",
            "noise_fields": ["n", "omega", "Te", "vpar_e", "vpar_i"],
        },
        "time": {
            "method": "rk4_imex_strang",
            "dt": 5.0e-4,
            "nsteps": 16,
            "save_every": 4,
            "diag_mode": "full",
            "save_fields": True,
            "snapshot_fields": ["n", "Te", "omega", "phi", "Ti", "psi", "N"],
            "return_numpy": True,
        },
    }


def test_full_stack_em_hot_neutrals_short_run() -> None:
    out = run_simulation(_cfg(), as_numpy=True).diagnostics
    passed, reason, growth, peak = finite_run_gate(
        {
            "rms_n_fluct": np.asarray(out["rms_n_fluct"]),
            "rms_Te_fluct": np.asarray(out["rms_Te_fluct"]),
            "rms_omega_fluct": np.asarray(out["rms_omega_fluct"]),
            "rms_phi_fluct": np.asarray(out["rms_phi_fluct"]),
        },
        max_growth_factor=1.0e6,
        max_rms_abs=1.0e6,
    )
    assert passed, (reason, growth, peak)

    for key in ("rms_n", "rms_Te", "rms_omega", "rms_phi"):
        arr = np.asarray(out[key])
        assert np.all(np.isfinite(arr))

    for field in ("n", "Te", "omega", "phi", "Ti", "psi", "N"):
        key = f"snapshots_{field}"
        assert key in out
        snaps = np.asarray(out[key])
        assert snaps.size > 0
        assert np.all(np.isfinite(snaps))
