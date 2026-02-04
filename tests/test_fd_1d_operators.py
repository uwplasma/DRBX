from __future__ import annotations

import math

import jax.numpy as jnp

from jaxdrb.operators.fd import d1_open, d1_open_fv, d1_open_sbp21, d1_periodic


def _smooth_periodic_function(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    f = jnp.sin(2.0 * x) + 0.3 * jnp.cos(3.0 * x)
    df = 2.0 * jnp.cos(2.0 * x) - 0.9 * jnp.sin(3.0 * x)
    return f, df


def test_d1_periodic_is_second_order() -> None:
    """Convergence check for periodic centered difference."""

    errs = []
    hs = []
    for n in [64, 128, 256]:
        L = 2 * math.pi
        dx = L / n
        x = jnp.linspace(0.0, L, n, endpoint=False)
        f, df = _smooth_periodic_function(x)
        df_num = d1_periodic(f, dx)
        err = jnp.sqrt(jnp.mean((df_num - df) ** 2))
        errs.append(float(err))
        hs.append(dx)

    # Estimate observed order from successive refinements.
    p1 = math.log(errs[0] / errs[1]) / math.log(hs[0] / hs[1])
    p2 = math.log(errs[1] / errs[2]) / math.log(hs[1] / hs[2])
    assert p1 > 1.8
    assert p2 > 1.8


def test_d1_open_is_second_order_interior() -> None:
    """Open-grid derivative should be second order in the interior (one-sided at boundaries)."""

    errs = []
    hs = []
    for n in [65, 129, 257]:
        L = 2 * math.pi
        dx = L / (n - 1)
        x = jnp.linspace(0.0, L, n, endpoint=True)
        f, df = _smooth_periodic_function(x)
        df_num = d1_open(f, dx)
        # Exclude boundary points where one-sided stencils dominate.
        err = jnp.sqrt(jnp.mean((df_num[2:-2] - df[2:-2]) ** 2))
        errs.append(float(err))
        hs.append(dx)

    p1 = math.log(errs[0] / errs[1]) / math.log(hs[0] / hs[1])
    p2 = math.log(errs[1] / errs[2]) / math.log(hs[1] / hs[2])
    assert p1 > 1.8
    assert p2 > 1.8


def test_d1_open_sbp21_is_first_order_at_boundaries() -> None:
    """SBP21 should converge (but only first order at boundaries)."""

    errs = []
    hs = []
    for n in [65, 129, 257]:
        L = 2 * math.pi
        dx = L / (n - 1)
        x = jnp.linspace(0.0, L, n, endpoint=True)
        f, df = _smooth_periodic_function(x)
        df_num = d1_open_sbp21(f, dx)
        err = jnp.sqrt(jnp.mean((df_num - df) ** 2))
        errs.append(float(err))
        hs.append(dx)

    p = math.log(errs[0] / errs[1]) / math.log(hs[0] / hs[1])
    assert p > 0.8


def test_d1_open_fv_telescopes() -> None:
    """Finite-volume style derivative should satisfy a telescoping sum property."""

    n = 129
    L = 1.0
    dx = L / (n - 1)
    x = jnp.linspace(0.0, L, n, endpoint=True)
    f = x**3 - 0.2 * x**2 + 0.1
    df = d1_open_fv(f, dx)
    approx = float(jnp.sum(df) * dx)
    exact = float(f[-1] - f[0])
    assert abs(approx - exact) < 5e-3
