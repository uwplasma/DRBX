import numpy as np

from jaxdrb.driver import run_simulation


def _base_cfg():
    return {
        "geometry": {
            "kind": "salpha",
            "nx": 8,
            "ny": 8,
            "nz": 4,
            "Lx": 1.0,
            "Ly": 1.0,
            "Lz": 6.283185307179586,
            "bc_x": "periodic",
            "bc_y": "periodic",
            "curvature_model": "vector_xy",
        },
        "physics": {"em_on": False, "hot_ion_on": False, "boussinesq": True},
        "numerics": {"perp_operator": "spectral", "poisson": "spectral"},
        "initial": {"noise_amplitude": 0.1, "noise_fields": ["omega"], "noise_seed": 0.0},
        "time": {
            "method": "rk4_scan",
            "dt": 1e-3,
            "nsteps": 4,
            "save_every": 1,
            "diag_mode": "full",
            "diag_phi_use_guess_only": True,
            "poisson_warm_start": False,
            "poisson_track_iters": True,
            "progress": False,
        },
    }


def test_carry_phi_eliminates_diag_poisson():
    cfg = _base_cfg()

    cfg["time"]["carry_phi"] = False
    out_no_carry = run_simulation(cfg)
    rms_phi_no = np.asarray(out_no_carry.diagnostics["rms_phi"])

    cfg["time"]["carry_phi"] = True
    out_carry = run_simulation(cfg)
    rms_phi_yes = np.asarray(out_carry.diagnostics["rms_phi"])

    assert np.all(rms_phi_no == 0.0)
    assert np.any(rms_phi_yes != 0.0)
    assert "poisson_iters_mean" in out_carry.diagnostics
