"""Fluid neutrals on the FCI grid (``native/fci_neutral_fluid.py``, stage N1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.fci_braginskii.native import fci_neutral_fluid as FL
from drbx.fci_braginskii.native import fci_neutrals as F
from tests.fci_braginskii.test_fci_neutrals import PARAMS, _random_plasma, annulus, sheared_slab

jax.config.update("jax_enable_x64", True)
GEOMETRIES = [sheared_slab(), annulus()]
IDS = ["sheared", "annulus"]


def test_flux_limiter_is_smooth_bounded_and_saturates_at_the_limit():
    gamma_c = jnp.array([1e-6, 0.1, 1.0, 3.0, 1e3])
    f = FL.flux_limiter(gamma_c, 1.0)
    np.testing.assert_allclose(np.asarray(f), (1.0 + np.asarray(gamma_c) ** 2) ** -0.5, rtol=1e-14)
    assert float(f[0]) == pytest.approx(1.0, abs=1e-11) and float(f[-1] * gamma_c[-1]) == pytest.approx(1.0, rel=1e-6)
    assert np.all(np.diff(np.asarray(f * gamma_c)) > 0.0)  # limited flux still monotone in Gamma_c
    assert np.isfinite(float(jax.grad(lambda g: FL.flux_limiter(g, 1.0))(0.0)))


def test_transport_coefficients_follow_the_afn_closure():
    n, _, Te, Ti, Nn = _random_plasma((4, 3, 3), 3)
    Tn = 0.05 + 0.1 * Te
    nu_cx, nu_iz = F.collision_frequencies(n, Te, Ti, PARAMS, Tn)
    dp, kappa, eta = FL.transport_coefficients(n, Te, Ti, Nn, Tn, PARAMS)
    np.testing.assert_allclose(np.asarray(dp), 1.0 / np.asarray(nu_cx + nu_iz), rtol=1e-13)  # tau_n / m
    np.testing.assert_allclose(np.asarray(eta), np.asarray(Nn * Tn * dp), rtol=1e-13)  # p_n tau_n
    np.testing.assert_allclose(np.asarray(kappa), 2.5 * np.asarray(eta), rtol=1e-13)  # 5 p_n tau_n / (2 m)


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=IDS)
def test_transport_matches_the_diffusive_model_at_uniform_temperature(geometry):
    grid = F.build_neutral_grid(geometry)
    _, _, _, _, Nn = _random_plasma(geometry.shape, 1)
    dp = jnp.asarray(0.05 + np.random.default_rng(2).random(geometry.shape))
    tn = jnp.full(geometry.shape, 0.4)
    n_new, _, rel_n, _ = FL.neutral_transport(Nn, tn, dp, 0.0 * dp, 0.05, grid, PARAMS)
    reference, _ = F.implicit_diffusion(Nn, 0.4 * dp, 0.05, grid, PARAMS)  # D = T/nu
    np.testing.assert_allclose(np.asarray(n_new), np.asarray(reference), rtol=1e-7)
    assert float(rel_n) < 1e-9


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=IDS)
def test_transport_conserves_nuclei_and_energy_exactly_with_sources(geometry):
    grid = F.build_neutral_grid(geometry)
    _, _, Te, _, Nn = _random_plasma(geometry.shape, 4)
    dp = jnp.asarray(0.05 + np.random.default_rng(5).random(geometry.shape))
    kappa = 2.5 * dp * Nn * Te
    source = jnp.asarray(np.random.default_rng(6).random(geometry.shape))
    dt, v = 0.04, grid.volume
    n_new, t_new, rel_n, rel_e = FL.neutral_transport(Nn, Te, dp, kappa, dt, grid, PARAMS, source)
    assert float(jnp.sum(v * jnp.abs(n_new - Nn))) > 1e-3 and float(rel_e) < 1e-9
    nuclei = float(jnp.sum(v * Nn))
    assert abs(float(jnp.sum(v * n_new)) - nuclei - dt * float(jnp.sum(v * source))) < 1e-13 * nuclei
    w0, w1 = float(jnp.sum(v * 1.5 * Nn * Te)), float(jnp.sum(v * 1.5 * n_new * t_new))
    assert abs(w1 - w0 - dt * float(jnp.sum(v * 1.5 * Te * source))) < 1e-13 * w0  # atoms born at local T_n


def test_pure_conduction_relaxes_a_hot_spot_without_changing_the_energy():
    geometry = sheared_slab()
    grid = F.build_neutral_grid(geometry)
    shape = geometry.shape
    nn = jnp.full(shape, 0.3)
    tn = 0.2 + 0.5 * jnp.exp(-((jnp.arange(shape[0]) - 8.0) ** 2 / 6.0))[:, None, None] * jnp.ones(shape)
    kappa = jnp.full(shape, 0.2)
    n_new, t_new, _, rel_e = FL.neutral_transport(nn, tn, 0.0 * kappa, kappa, 0.1, grid, PARAMS)
    np.testing.assert_allclose(np.asarray(n_new), 0.3, rtol=1e-14)
    assert float(jnp.max(t_new)) < float(jnp.max(tn)) and float(jnp.min(t_new)) > float(jnp.min(tn)) - 1e-12
    v = grid.volume
    assert abs(float(jnp.sum(v * t_new)) - float(jnp.sum(v * tn))) < 1e-13 * float(jnp.sum(v * tn))
    assert float(rel_e) < 1e-9


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=IDS)
def test_full_step_conserves_nuclei_and_total_energy_between_plasma_and_neutrals(geometry):
    # Exchange terms: sum_V (S_plasma + S_neutral) = 0 for nuclei and energy (plasma kinetic + thermal,
    # neutral thermal, plus the electron energy radiated); wall_step_check (PR 72) is not on main yet.
    grid = F.build_neutral_grid(geometry)
    params = F.NeutralParameters(
        reference_density_m3=5.0e18, reference_temperature_ev=PARAMS.reference_temperature_ev, reaction_substeps=3
    )
    n, Vi, Te, Ti, Nn = _random_plasma(geometry.shape, 7)
    Tn = 0.05 + 0.3 * Te
    changed = 0.0
    for _ in range(4):
        before = F.inventories(n, Vi, Te, Ti, Nn, grid, params, Tn)
        n, Vi, Te, Ti, Nn, Tn_new, ledger, _ = FL.neutral_fluid_step(n, Vi, Te, Ti, Nn, Tn, 0.02, grid, params)
        changed += float(jnp.max(jnp.abs(Tn_new - Tn)))
        Tn = Tn_new
        after = F.inventories(n, Vi, Te, Ti, Nn, grid, params, Tn)
        assert abs(float(after[0] - before[0])) < 1e-14 * float(before[0])
        sink = float(ledger[3])
        assert sink > 0.0
        assert abs(float(after[1] + after[2] + sink - before[1] - before[2])) < 1e-12 * sink
        assert float(ledger[7]) < 1e-9 and float(ledger[8]) < 1e-9
    assert changed > 1e-3 and float(jnp.min(Nn)) > 0.0 and float(jnp.min(Tn)) > 0.0


def test_limiter_engages_on_steep_gradients_and_caps_the_particle_flux():
    geometry = sheared_slab()
    grid = F.build_neutral_grid(geometry)
    n, _, Te, Ti, _ = _random_plasma(geometry.shape, 9)
    ramp = jnp.linspace(1e-3, 1.0, geometry.shape[0])[:, None, None] * jnp.ones(geometry.shape)
    Tn = jnp.full(geometry.shape, 0.1)
    dp, _, fp, _ = FL.limited_coefficients(1e-3 * n, Te, Ti, ramp, Tn, grid, PARAMS)  # long mfp
    assert float(jnp.min(fp)) < 0.5
    gamma = dp * F.grad_norm(F._fine(ramp * Tn, grid), grid)
    assert np.all(np.asarray(gamma) <= np.asarray(ramp * jnp.sqrt(Tn / (2 * jnp.pi))) * (1 + 1e-12))


def test_fluid_step_is_differentiable():
    geometry = sheared_slab(shape=(6, 4, 3))
    grid = F.build_neutral_grid(geometry)
    n, Vi, Te, Ti, Nn = _random_plasma(geometry.shape, 11)
    Tn = 0.05 + 0.3 * Te

    def functional(scale):
        out = FL.neutral_fluid_step(n, Vi, Te, scale * Ti, Nn, Tn, 0.02, grid, PARAMS)
        return jnp.sum(grid.volume * out[4] * out[5])

    g = float(jax.grad(functional)(1.0))
    eps = 1e-5
    fd = (float(functional(1.0 + eps)) - float(functional(1.0 - eps))) / (2 * eps)
    assert np.isfinite(g) and g == pytest.approx(fd, rel=1e-4)
