from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.operators.brackets import poisson_bracket_arakawa


def test_arakawa_bracket_conserves_quadratic_invariant() -> None:
    jax.config.update("jax_enable_x64", True)
    nx, ny = 32, 32
    dx = 2.0 * jnp.pi / nx
    dy = 2.0 * jnp.pi / ny
    key = jax.random.PRNGKey(0)
    phi = jax.random.normal(key, (nx, ny))
    f = jax.random.normal(jax.random.PRNGKey(1), (nx, ny))

    j = poisson_bracket_arakawa(phi, f, dx, dy)
    invariant = jnp.mean(f * j)

    assert abs(float(invariant)) < 1.0e-10


def test_arakawa_bracket_antisymmetry() -> None:
    jax.config.update("jax_enable_x64", True)
    nx, ny = 32, 32
    dx = 2.0 * jnp.pi / nx
    dy = 2.0 * jnp.pi / ny
    phi = jax.random.normal(jax.random.PRNGKey(2), (nx, ny))
    f = jax.random.normal(jax.random.PRNGKey(3), (nx, ny))
    g = jax.random.normal(jax.random.PRNGKey(4), (nx, ny))

    j_fg = poisson_bracket_arakawa(phi, f, dx, dy)
    j_gf = poisson_bracket_arakawa(phi, g, dx, dy)
    antisym = jnp.mean(f * j_gf + g * j_fg)

    assert abs(float(antisym)) < 1.0e-10
