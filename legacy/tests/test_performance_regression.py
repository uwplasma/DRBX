from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdrb.driver import build_system_from_config
from jaxdrb.integrators import build_rk4_scan


def _make_cfg() -> dict:
    return {
        "geometry": {
            "kind": "plane",
            "nx": 16,
            "ny": 16,
            "Lx": float(2 * np.pi),
            "Ly": float(2 * np.pi),
            "bc_x": "periodic",
            "bc_y": "periodic",
            "dealias": False,
        },
        "physics": {
            "nonlinear_on": False,
            "em_on": False,
            "hot_ion_on": False,
            "neutrals_on": False,
            "boussinesq": True,
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
        "initial": {"amplitude": 1e-3, "seed": 0},
    }


def _block_until_ready(tree):
    def _block(x):
        if hasattr(x, "block_until_ready"):
            return x.block_until_ready()
        return x

    jax.tree_util.tree_map(_block, tree)


def test_perf_regression_16x16() -> None:
    if jax.config.jax_disable_jit:
        pytest.skip("JIT disabled; performance regression is not meaningful.")

    built = build_system_from_config(_make_cfg())
    system = built.system
    state = built.state

    # Short-window explicit gate (t_end = 20)
    dt = 0.1
    steps = 200
    save_every = 200

    def diag_fn(t, y, *, phi_guess=None):
        _ = (t, y, phi_guess)
        return jnp.asarray(0.0)

    runner, _, _ = build_rk4_scan(system.rhs, dt, steps, save_every, diag_fn)

    # Compile + warm-up.
    _block_until_ready(runner(state))

    start = time.perf_counter()
    out = runner(state)
    _block_until_ready(out)
    elapsed = time.perf_counter() - start
    time_per_step = elapsed / steps

    assert time_per_step < 0.1
