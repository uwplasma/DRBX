"""Implicit Spitzer-Harm parallel conduction: coefficients, 1D support-pair checks, HSX wiring."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from drbx.fci_braginskii import run as driver
from drbx.fci_braginskii.native.fci_drb_EB_rhs import LocalFciDrbEBRhs
from drbx.fci_braginskii.native.fci_parallel_conduction import (
    backward_euler_conduction,
    conductivity,
    spitzer_prefactors,
)
from drbx.fci_braginskii.native.fci_support_pair import build_weighted_negative_adjoint

jax.config.update("jax_enable_x64", True)


def _pair(cells: int):
    """1D closed-wall support pair: face differences averaged to cells, no wall flux."""

    dx = 1.0 / cells
    faces = (jnp.eye(cells, k=1) - jnp.eye(cells))[:-1] / dx
    average = jnp.zeros((cells, cells - 1)).at[0, 0].set(1.0).at[-1, -1].set(1.0)
    for i in range(1, cells - 1):
        average = average.at[i, i - 1].set(0.5).at[i, i].set(0.5)
    gradient = average @ faces
    mass = jnp.full((cells,), dx)
    divergence = build_weighted_negative_adjoint(lambda f: gradient @ f, mass, mass)
    return (lambda f: gradient @ f), divergence, mass


def _dense_solve(apply_a, rhs):
    matrix = jax.vmap(apply_a)(jnp.eye(rhs.size)).T
    return jnp.linalg.solve(matrix, rhs), None


def _step(cells, kappa_of_t, temperature, density, dt):
    gradient, divergence, mass = _pair(cells)
    kappa = kappa_of_t(temperature)
    increment, _ = backward_euler_conduction(
        lambda y: divergence(kappa * gradient(y)), 1.5 * density, temperature, dt, _dense_solve
    )
    return temperature + increment, mass


def _semi_discrete(cells, end_time):
    """Exact solution of ``dT/dt = D G T / 1.5`` (kappa = n = 1) and of the continuum problem."""

    gradient, divergence, mass = _pair(cells)
    matrix = np.asarray(jax.vmap(lambda y: divergence(gradient(y)))(jnp.eye(cells)).T) / 1.5
    x = (np.arange(cells) + 0.5) / cells
    initial = 1.0 + 0.2 * np.cos(np.pi * x)
    exact = 1.0 + 0.2 * np.cos(np.pi * x) * np.exp(-np.pi**2 * end_time / 1.5)
    return expm(matrix * end_time) @ initial, exact, initial


def test_spitzer_prefactors_reference_values():
    kappa_e, kappa_i, te0 = spitzer_prefactors(5.0e-4, 1836.0, 1.0, 1.0e19, 15.0)
    assert te0 == pytest.approx(24.0, rel=5e-3)
    assert kappa_e == pytest.approx(75.0, rel=1e-2)  # 3.6e6 m^2/s over c_s0 L_ref
    assert kappa_i / kappa_e == pytest.approx(3.9 / 3.16 * np.sqrt(2.0 / 1836.0), rel=1e-12)
    # kappa is n-independent, ~ T^(5/2)/ln(Lambda): the prefactor scales that way.
    assert spitzer_prefactors(5e-4, 1836.0, 1.0, 1e19, 30.0)[0] == pytest.approx(kappa_e / 2, rel=1e-12)
    assert spitzer_prefactors(5e-4, 1836.0, 1.0, 2e19, 15.0)[0] == pytest.approx(kappa_e / 2, rel=1e-12)
    assert spitzer_prefactors(5e-4, 1836.0, 4.0, 1e19, 15.0)[1] == pytest.approx(kappa_i * 32.0, rel=1e-12)


def test_flux_limiter_caps_the_conductive_flux_at_the_free_streaming_flux():
    temperature = jnp.array([1.0, 2.0, 4.0, 8.0])
    density = jnp.ones(4)
    gradient = lambda t: jnp.full_like(t, 50.0)
    free = conductivity(temperature, density, 3.0, 1836.0, 0.0, gradient)
    np.testing.assert_allclose(free, 3.0 * temperature**2.5)
    limited = conductivity(temperature, density, 3.0, 1836.0, 0.2, gradient)
    free_streaming = 0.2 * density * temperature * jnp.sqrt(1836.0 * temperature)
    assert np.all(limited < free)
    assert np.all(limited * 50.0 < free_streaming)
    weak = conductivity(temperature, density, 3.0, 1836.0, 0.2, lambda t: jnp.full_like(t, 1e-9))
    np.testing.assert_allclose(weak, free, rtol=1e-6)


def test_manufactured_cosine_converges_at_second_order_in_space_and_first_in_time():
    errors = []
    for cells in (16, 32, 64):
        semi, exact, _ = _semi_discrete(cells, 0.05)
        errors.append(np.linalg.norm(semi - exact) / np.sqrt(cells))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 1.9), (errors, orders)
    # Backward Euler against the semi-discrete solution (spatial error removed).
    cells, end_time = 32, 0.05
    reference, _, initial = _semi_discrete(cells, end_time)
    time_errors = []
    for steps in (10, 20, 40):
        temperature = jnp.asarray(initial)
        for _ in range(steps):
            temperature, _ = _step(cells, lambda t: jnp.ones_like(t), temperature, jnp.ones(cells), end_time / steps)
        time_errors.append(np.linalg.norm(np.asarray(temperature) - reference))
    time_orders = np.log2(np.array(time_errors[:-1]) / np.array(time_errors[1:]))
    assert np.all(np.abs(time_orders - 1.0) < 0.15), (time_errors, time_orders)


def test_closed_walls_conserve_the_domain_energy_and_keep_the_hot_spot_positive():
    cells = 48
    x = (jnp.arange(cells) + 0.5) / cells
    temperature = 0.05 + 4.0 * jnp.exp(-(((x - 0.3) / 0.05) ** 2))
    density = 1.0 + 0.5 * jnp.sin(3.0 * x)
    kappa_of_t = lambda t: 75.0 * jnp.maximum(t, 1e-6) ** 2.5  # chi dt/dx^2 ~ 1e4
    energy0 = None
    for _ in range(20):
        gradient, divergence, mass = _pair(cells)
        kappa = kappa_of_t(temperature)
        increment, _ = backward_euler_conduction(
            lambda y: divergence(kappa * gradient(y)), 1.5 * density, temperature, 7.5e-4, _dense_solve
        )
        energy0 = jnp.sum(mass * 1.5 * density * temperature) if energy0 is None else energy0
        temperature = temperature + increment
        assert float(jnp.sum(mass * 1.5 * density * increment)) == pytest.approx(0.0, abs=1e-13)
        assert float(temperature.min()) > 0.05 - 1e-12
    assert float(jnp.sum(mass * 1.5 * density * temperature)) == pytest.approx(float(energy0), rel=1e-13)
    assert float(temperature.max()) < 4.05


def test_conduction_default_follows_the_wall_closure():
    assert driver.conduction_prefactors(driver.HsxBlobConfig())[0] == pytest.approx(75.0, rel=1e-2)
    assert driver.conduction_prefactors(driver.HsxBlobConfig(wall_closure="zero-flow")) == (0.0, 0.0)
    assert driver.conduction_prefactors(driver.HsxBlobConfig(parallel_conduction="off")) == (0.0, 0.0)
    assert driver.conduction_prefactors(
        driver.HsxBlobConfig(wall_closure="zero-flow", parallel_conduction="spitzer")
    )[0] > 0.0
    with pytest.raises(ValueError, match="parallel_conduction"):
        driver.HsxBlobConfig(parallel_conduction="braginskii")
    with pytest.raises(ValueError, match="heat_flux_limiter"):
        driver.HsxBlobConfig(heat_flux_limiter=-0.1)


@pytest.mark.slow
def test_hsx_conduction_is_energy_neutral_and_finite(tmp_path, monkeypatch):
    """Shipped-coil HSX at 16-cubed: each conduction stage has zero domain energy.

    The geometry is built from the deck ``[geometry]`` table and cached under
    ``output/hsx_geometry``, as in the Bohm regression test of test_fci_wall_closure.
    """

    records = []
    solve = LocalFciDrbEBRhs.solve_implicit_parallel_conduction

    def recording_solve(self, state_owned, *, solve_dt):
        stage, increment, infos = solve(self, state_owned, solve_dt=solve_dt)
        _, owner_mass = self._ledger_masses()
        energy = jnp.stack([
            jnp.sum(owner_mass * 1.5 * state_owned.density * increment.Te),
            jnp.sum(owner_mass * 1.5 * state_owned.density * increment.Ti),
            jnp.sum(owner_mass * 1.5 * state_owned.density * jnp.abs(increment.Te)),
            jnp.min(jnp.where(owner_mass > 0.0, stage.Te, 1.0)),
        ])
        jax.debug.callback(lambda values, flags: records.append((values, flags)), energy,
                           jnp.stack([jnp.asarray(i.failed) for i in infos]))
        return stage, increment, infos

    monkeypatch.setattr(LocalFciDrbEBRhs, "solve_implicit_parallel_conduction", recording_solve)
    deck = tmp_path / "hsx.toml"
    deck.write_text(
        '[model]\nbackend = "fci_braginskii"\n'
        "[geometry]\nresolution = [16, 16, 16]\nmetric_mesh_shape = [16, 16, 16]\n"
        "metric_poloidal_modes = 6\nmetric_toroidal_modes = 3\ntoroidal_modes = 3\n"
        f'cache_dir = "{Path(__file__).resolve().parents[2] / "output" / "hsx_geometry"}"\n'
        f'[fci_braginskii]\noutput = "{tmp_path / "hsx"}"\n'
        "final_time = 0.0075\nnum_steps = 4\nsave_every = 4\n"
    )
    from drbx.cli import main

    assert main(["run", str(deck)]) in (0, None)
    assert records
    for values, failed in records:
        assert abs(values[0]) <= 1e-9 * max(values[2], 1e-300) + 1e-14
        assert abs(values[1]) <= 1e-9 * max(abs(values[0]), 1.0) + 1e-14
        assert values[3] > 0.0 and not np.any(failed)
