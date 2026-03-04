from __future__ import annotations

import numpy as np

from jaxdrb.driver import build_system_from_config


def _base_cfg() -> dict:
    return {
        "geometry": {
            "kind": "plane",
            "nx": 5,
            "ny": 6,
            "Lx": 1.0,
            "Ly": 1.2,
            "bc_x": "periodic",
            "bc_y": "periodic",
        },
        "physics": {"hot_ion_on": False},
        "initial": {"n0": 1.0, "Te0": 1.0},
    }


def test_initial_state_npz_overrides_channels(tmp_path) -> None:
    cfg = _base_cfg()
    shape = (5, 6)
    n = np.full(shape, 1.7, dtype=np.float64)
    Te = np.full(shape, 0.9, dtype=np.float64)
    omega = np.full(shape, -0.02, dtype=np.float64)
    vpar_e = np.full(shape, 0.03, dtype=np.float64)
    vpar_i = np.full(shape, -0.01, dtype=np.float64)
    p = tmp_path / "state.npz"
    np.savez(p, n=n, Te=Te, omega=omega, vpar_e=vpar_e, vpar_i=vpar_i)
    cfg["initial"]["state_npz"] = str(p)

    built = build_system_from_config(cfg)
    np.testing.assert_allclose(np.asarray(built.state.n), n)
    np.testing.assert_allclose(np.asarray(built.state.Te), Te)
    np.testing.assert_allclose(np.asarray(built.state.omega), omega)
    np.testing.assert_allclose(np.asarray(built.state.vpar_e), vpar_e)
    np.testing.assert_allclose(np.asarray(built.state.vpar_i), vpar_i)


def test_initial_state_npz_phi_backfills_omega(tmp_path) -> None:
    cfg = _base_cfg()
    x = np.linspace(0.0, 2.0 * np.pi, 5)[:, None]
    y = np.linspace(0.0, 2.0 * np.pi, 6)[None, :]
    phi = 0.1 * np.sin(2.0 * x - y)
    n = np.ones((5, 6), dtype=np.float64)
    Te = np.ones((5, 6), dtype=np.float64)

    p = tmp_path / "state_phi.npz"
    np.savez(p, n=n, Te=Te, phi=phi)
    cfg["initial"]["state_npz"] = str(p)

    built = build_system_from_config(cfg)
    expected = built.system._omega_from_phi(
        np.asarray(phi, dtype=np.float64),
        np.asarray(n, dtype=np.float64),
        Te=np.asarray(Te, dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(built.state.omega), np.asarray(expected), rtol=1e-12, atol=1e-12
    )
