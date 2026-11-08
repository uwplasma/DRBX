"""Self-contained gate for the hydrogenic atomic reaction-rate coefficients.

Pins physically-correct behaviour of the packaged AMJUEL ionization /
recombination fits and the AMJUEL charge-exchange polynomial, plus specific
spot values (so a corrupted coefficient table is caught), vectorization, and
differentiability. No external database or hermes-3 binary is needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.native.neutrals.atomic_rates import (
    charge_exchange_rate_coefficient,
    energy_loss_coefficient,
    rate_coefficient,
)

jax.config.update("jax_enable_x64", True)

N_REF = 1.0e19


def _iz(te):
    return float(rate_coefficient("d", "iz", jnp.asarray(te), jnp.asarray(N_REF)))


def _rec(te):
    return float(rate_coefficient("d", "rec", jnp.asarray(te), jnp.asarray(N_REF)))


def _cx(t_eff):
    return float(charge_exchange_rate_coefficient(jnp.asarray(t_eff)))


def test_ionization_rises_with_temperature() -> None:
    # Negligible below the ionization threshold, rising steeply through 3-30 eV.
    assert _iz(1.0) < 1.0e-18
    assert _iz(3.0) < _iz(10.0) < _iz(30.0)
    assert 1.0e-14 < _iz(30.0) < 1.0e-13
    # Spot value pins the actual coefficient table (m^3/s at 10 eV, 1e19 m^-3).
    assert _iz(10.0) == pytest.approx(8.72e-15, rel=2.0e-2)


def test_recombination_dominates_cold_plasma() -> None:
    # Recombination rises as the plasma cools -- the detachment driver.
    assert _rec(1.0) > _rec(10.0) > _rec(100.0)
    assert _rec(1.0) > 0.0
    # At a cold target (~1 eV) recombination exceeds ionization.
    assert _rec(1.0) > _iz(1.0)


def test_charge_exchange_magnitude_and_trend() -> None:
    assert _cx(1.0) < _cx(10.0) < _cx(100.0)
    assert 1.0e-14 < _cx(10.0) < 1.0e-13


def test_recombination_returns_binding_energy_at_low_temperature() -> None:
    # The 13.6 eV recombination potential energy is returned to the electrons, so
    # the net electron energy channel becomes a heating term (negative loss) at
    # low temperature.
    assert float(energy_loss_coefficient("d", "rec", jnp.asarray(1.0), jnp.asarray(N_REF))) < 0.0


def test_rates_are_vectorized_and_jit_grad_transparent() -> None:
    te = jnp.array([1.0, 10.0, 100.0])
    ne = jnp.full_like(te, N_REF)
    rates = rate_coefficient("d", "iz", te, ne)
    assert rates.shape == te.shape
    assert np.all(np.asarray(rates) > 0.0)

    jitted = jax.jit(lambda t: rate_coefficient("d", "iz", t, jnp.asarray(N_REF)))
    assert float(jitted(jnp.asarray(10.0))) == pytest.approx(_iz(10.0), rel=1e-10)

    gradient = float(jax.grad(lambda t: rate_coefficient("d", "iz", t, jnp.asarray(N_REF)))(jnp.asarray(10.0)))
    assert np.isfinite(gradient) and gradient > 0.0


def test_amjuel_fit_clamps_below_validity_range() -> None:
    # Below the fitted floor (0.1 eV) the clamp makes the rate constant.
    assert _iz(0.05) == pytest.approx(_iz(0.1), rel=1e-10)


def test_hydrogen_rates_pin_amjuel_2020_values() -> None:
    # <sigma v> [m^3/s] at n_e = 1e19 m^-3 from the AMJUEL report of
    # January 13, 2020 (H.4 2.1.5 p.135, H.4 2.1.8 p.141, H.2 3.1.8 p.43),
    # evaluated from coefficients transcribed independently of the package.
    from drbx.native.neutrals.atomic_rates import charge_exchange_rate_coefficient, rate_coefficient

    temperatures = np.array([1.0, 5.0, 20.0, 100.0])
    ionization = np.array([2.03e-20, 1.67e-15, 2.15e-14, 4.04e-14])
    charge_exchange = np.array([9.21e-15, 1.69e-14, 2.78e-14, 4.42e-14])
    np.testing.assert_allclose(np.asarray(rate_coefficient("d", "iz", temperatures, 1.0e19)), ionization, rtol=3e-3)
    np.testing.assert_allclose(np.asarray(charge_exchange_rate_coefficient(temperatures)), charge_exchange, rtol=3e-3)
    np.testing.assert_allclose(float(rate_coefficient("d", "rec", 1.0, 1.0e19)), 8.57e-19, rtol=3e-3)
    np.testing.assert_allclose(float(rate_coefficient("d", "rec", 100.0, 1.0e19)), 6.17e-21, rtol=3e-3)


def test_every_rate_is_differentiable_and_finite_over_the_fit_range() -> None:
    # N0: one rates module serves the SD1D and FCI paths; all channels (including the
    # radiated energy per ionisation, AMJUEL H.10 2.1.5) have finite jax.grad in T and n_e.
    from drbx.native.neutrals import atomic_rates as ar

    temperatures = jnp.array([0.05, 0.5, 2.0, 10.0, 100.0, 1.0e3, 2.0e4])  # includes both clamps
    densities = jnp.array([1.0e12, 1.0e16, 1.0e19, 1.0e21, 1.0e23])
    t, n = jnp.meshgrid(temperatures, densities)
    channels = (
        lambda t, n: ar.rate_coefficient("d", "iz", t, n),
        lambda t, n: ar.rate_coefficient("d", "rec", t, n),
        lambda t, n: ar.energy_loss_coefficient("d", "iz", t, n),
        lambda t, n: ar.energy_loss_coefficient("d", "rec", t, n),
        lambda t, n: ar.charge_exchange_rate_coefficient(t),
        lambda t, n: ar.sd1d_ionisation_rate(t),
        lambda t, n: ar.sd1d_recombination_rate(n, t),
        lambda t, n: ar.sd1d_charge_exchange_rate(t),
    )
    for f in channels:
        value = jax.vmap(jax.vmap(f))(t, n)
        grads = jax.vmap(jax.vmap(jax.grad(f, argnums=(0, 1))))(t, n)
        assert np.all(np.isfinite(np.asarray(value)))
        assert all(np.all(np.isfinite(np.asarray(g))) for g in grads)


def test_ionisation_energy_cost_is_at_least_the_ionisation_potential() -> None:
    # Electron energy lost per ionisation <sigma v E>/<sigma v> (radiation + 13.6 eV) is >= 13.6 eV
    # and falls with temperature.
    te = jnp.array([2.0, 5.0, 20.0, 100.0])
    cost = np.asarray(energy_loss_coefficient("d", "iz", te, N_REF) / rate_coefficient("d", "iz", te, N_REF))
    assert np.all(cost >= 13.6) and 20.0 < cost[-1] < 30.0  # ~24 eV at 100 eV, ~119 eV at 2 eV (multistep radiation)
    assert cost[0] > cost[-1]
