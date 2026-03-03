from __future__ import annotations

import jax.numpy as jnp

from jaxdrb.parity_fv.flux_parallel import div_parallel_fv


def test_div_parallel_constant_state_is_zero() -> None:
    nz, nx, ny = 8, 3, 4
    f = 2.0 * jnp.ones((nz, nx, ny))
    v = 0.5 * jnp.ones((nz, nx, ny))
    div = div_parallel_fv(f, v, dz=0.1, limiter="mc")
    assert float(jnp.max(jnp.abs(div))) == 0.0


def test_div_parallel_boundary_flux_balance() -> None:
    nz, nx, ny = 10, 2, 3
    f = jnp.linspace(1.0, 2.0, nz)[:, None, None] * jnp.ones((1, nx, ny))
    v = jnp.ones_like(f)
    low = jnp.zeros((nx, ny))
    high = jnp.zeros((nx, ny))
    div = div_parallel_fv(
        f,
        v,
        dz=0.2,
        limiter="minmod",
        boundary_flux_low=low,
        boundary_flux_high=high,
    )
    # Integral of divergence equals boundary flux difference.
    total = jnp.sum(div, axis=0) * 0.2
    assert float(jnp.max(jnp.abs(total))) < 1e-10
