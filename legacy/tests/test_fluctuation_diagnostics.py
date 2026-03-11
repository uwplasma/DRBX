from __future__ import annotations

import numpy as np

from jaxdrb.driver import run_simulation


def _cfg() -> dict:
    return {
        "geometry": {
            "kind": "plane",
            "nx": 16,
            "ny": 16,
            "Lx": 8.0,
            "Ly": 8.0,
            "bc_x": "periodic",
            "bc_y": "periodic",
        },
        "physics": {
            "nonlinear_on": True,
            "em_on": False,
            "hot_ion_on": False,
            "neutrals_on": False,
            "boussinesq": True,
            "curvature_on": False,
            "omega_n": 0.0,
            "omega_Te": 0.0,
            "source_on": False,
            "sol_on": False,
        },
        "transport": {
            "Dn": 1.0e-3,
            "DOmega": 1.0e-3,
            "DTe": 1.0e-3,
            "Dn4": 0.0,
            "DOmega4": 0.0,
            "DTe4": 0.0,
        },
        "numerics": {"poisson": "spectral", "bracket": "arakawa"},
        "closures": {"sheath_on": False, "sheath_bc_on": False},
        "initial": {
            "amplitude": 1.0e-3,
            "seed": 0,
            "noise_mode": "physical",
            "noise_fields": ["n", "omega", "Te"],
        },
        "time": {
            "method": "rk4_scan",
            "dt": 1.0e-3,
            "nsteps": 20,
            "save_every": 2,
            "diag_mode": "full",
            "save_fields": True,
            "snapshot_fields": ["n", "Te", "omega", "phi"],
            "return_numpy": True,
        },
    }


def _manual_rms_fluct(snaps: np.ndarray) -> np.ndarray:
    base = snaps[0]
    delta = snaps - base[None, ...]
    axes = tuple(range(1, delta.ndim))
    return np.sqrt(np.mean(delta * delta, axis=axes))


def test_fluctuation_rms_matches_snapshots():
    out = run_simulation(_cfg(), as_numpy=True).diagnostics

    for field in ("n", "Te", "omega", "phi"):
        snap_key = f"snapshots_{field}"
        eq_key = f"equilibrium_{field}"
        rms_key = f"rms_{field}_fluct"
        assert snap_key in out
        assert eq_key in out
        assert rms_key in out

        snaps = np.asarray(out[snap_key])
        eq = np.asarray(out[eq_key])
        rms = np.asarray(out[rms_key])

        assert np.allclose(eq, snaps[0], rtol=1e-12, atol=1e-12)
        manual = _manual_rms_fluct(snaps)
        assert np.allclose(rms, manual, rtol=1e-10, atol=1e-12)
