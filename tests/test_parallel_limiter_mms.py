from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.terms.parallel import _flux_divergence_open


def _flux_error(nz: int, limiter: str) -> float:
    Lz = 2.0 * np.pi
    z = np.linspace(0.0, Lz, nz)
    dz = Lz / (nz - 1)
    k = 2.0
    f = np.sin(k * z)
    v = np.ones_like(f)
    div = np.asarray(_flux_divergence_open(jnp.asarray(f), jnp.asarray(v), dz, limiter))
    exact = v * (k * np.cos(k * z))
    err = np.sqrt(np.mean((div[1:-1] - exact[1:-1]) ** 2))
    return float(err)


def test_parallel_limiter_open_flux_second_order():
    e32 = _flux_error(32, "mc")
    e64 = _flux_error(64, "mc")
    e128 = _flux_error(128, "mc")

    assert e64 < e32
    assert e128 < e64
    assert (e32 / e64) > 2.0
    assert (e64 / e128) > 2.0


def test_parallel_limiter_none_is_fromm_second_order():
    e32 = _flux_error(32, "none")
    e64 = _flux_error(64, "none")
    e128 = _flux_error(128, "none")

    assert e64 < e32
    assert e128 < e64
    assert (e32 / e64) > 1.8
    assert (e64 / e128) > 1.8
