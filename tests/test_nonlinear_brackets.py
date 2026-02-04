from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.operators.brackets import poisson_bracket_arakawa, poisson_bracket_centered
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.spectral import poisson_bracket_spectral


def _rand_fields(key, grid: Grid2D):
    k1, k2 = jax.random.split(key, 2)
    a = jax.random.normal(k1, (grid.nx, grid.ny))
    b = jax.random.normal(k2, (grid.nx, grid.ny))
    return a, b


def test_arakawa_bracket_integral_is_zero():
    grid = Grid2D.make(nx=32, ny=24, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    a, b = _rand_fields(jax.random.key(0), grid)
    j = poisson_bracket_arakawa(a, b, grid.dx, grid.dy)
    # For a periodic Jacobian, the domain integral should vanish.
    assert jnp.abs(jnp.mean(j)) < 1e-10


def test_spectral_bracket_integral_is_zero():
    grid = Grid2D.make(nx=32, ny=24, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    a, b = _rand_fields(jax.random.key(1), grid)
    j = poisson_bracket_spectral(a, b, kx=grid.kx, ky=grid.ky, dealias_mask=grid.dealias_mask)
    assert jnp.abs(jnp.mean(j)) < 1e-10


def test_centered_bracket_integral_is_small():
    grid = Grid2D.make(nx=32, ny=24, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    a, b = _rand_fields(jax.random.key(2), grid)
    j = poisson_bracket_centered(a, b, grid.dx, grid.dy)
    assert jnp.abs(jnp.mean(j)) < 1e-8
