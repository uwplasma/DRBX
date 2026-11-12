"""Fast jitted-solve checks run on the GPU CI lane (``pytest -m gpu``).

They use the default JAX backend, so they also pass on CPU; the workflow asserts
that the backend is a GPU before running them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.native.hasegawa_wakatani import (
    HasegawaWakataniParameters,
    hw_grid,
    hw_run,
    particle_flux,
    potential_from_vorticity,
)

pytestmark = pytest.mark.gpu

_N = 16
_LENGTH = 2.0 * np.pi * 4.0


def _seed(amplitude: float = 1.0e-2):
    rng = np.random.default_rng(1)
    field = np.fft.fft2(rng.standard_normal((_N, _N))) * (amplitude / _N)
    field[0, 0] = 0.0
    return jnp.array(field), jnp.array(field * 0.5)


def _energy(zeta, density, grid):
    phi = potential_from_vorticity(zeta, grid)
    return jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density) ** 2)


def test_jitted_run_stays_on_default_device() -> None:
    grid = hw_grid(_N, _LENGTH)
    params = HasegawaWakataniParameters(adiabaticity=1.0, gradient=1.0, hyperviscosity=1.0e-3)
    zeta, density = _seed()
    zeta_f, _ = jax.jit(lambda z, n: hw_run(z, n, grid, params, dt=1.0e-2, steps=20))(zeta, density)
    assert zeta_f.devices() == {jax.devices()[0]}
    assert bool(jnp.all(jnp.isfinite(zeta_f)))


def test_ideal_run_conserves_energy() -> None:
    grid = hw_grid(_N, _LENGTH)
    params = HasegawaWakataniParameters(adiabaticity=0.0, gradient=0.0, hyperviscosity=0.0)
    zeta, density = _seed()
    e0 = float(_energy(zeta, density, grid))
    zeta_f, density_f = hw_run(zeta, density, grid, params, dt=1.0e-3, steps=100)
    assert float(_energy(zeta_f, density_f, grid)) == pytest.approx(e0, rel=1e-4)


def test_particle_flux_is_finite_and_outward() -> None:
    grid = hw_grid(_N, _LENGTH)
    params = HasegawaWakataniParameters(adiabaticity=1.0, gradient=1.0, hyperviscosity=1.0e-3)
    zeta, density = _seed(1.0)
    zeta_f, density_f = hw_run(zeta, density, grid, params, dt=1.0e-2, steps=60)
    flux = float(particle_flux(zeta_f, density_f, grid))
    assert np.isfinite(flux)


def test_gradient_through_run_matches_finite_difference() -> None:
    grid = hw_grid(_N, _LENGTH)
    zeta, density = _seed(1.0e-3)

    def final_energy(alpha):
        params = HasegawaWakataniParameters(adiabaticity=alpha, gradient=1.0, hyperviscosity=5.0e-3)
        zeta_f, density_f = hw_run(zeta, density, grid, params, dt=1.0e-2, steps=40)
        return _energy(zeta_f, density_f, grid)

    grad = float(jax.jit(jax.grad(final_energy))(1.0))
    step = 1.0e-3
    fd = (float(final_energy(1.0 + step)) - float(final_energy(1.0 - step))) / (2.0 * step)
    assert np.isfinite(grad)
    assert grad == pytest.approx(fd, rel=1e-3, abs=1e-6)
