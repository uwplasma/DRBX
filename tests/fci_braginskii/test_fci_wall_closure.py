"""Bohm sheath wall closure and wall-face flux contract of the HSX backend.

The flux-tube tests drive the production five-field parallel material
operator (explicit legs plus the selected wall legs with the sheath rows)
on a 1D open field line with walls at both ends and a uniform particle and
heat source, to steady state.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.fci_braginskii.native.fci_parallel_production_flux import (
    parallel_short_wall_material_data,
    parallel_target_row_material_residual,
)
from drbx.fci_braginskii.native.fci_wall_closure import (
    SheathClosure,
    bohm_wall_state,
    sheath_wall_leg_residual,
    sound_speed,
    wall_face_fluxes,
)

jax.config.update("jax_enable_x64", True)

TAU = 1.0


def test_stangeby_floating_wall_gamma():
    sheath = SheathClosure()
    expected = 2.0 - 0.5 * math.log(2.0 * math.pi * 2.0 / 1836.0)
    assert float(sheath.gamma_e(1.0, 1836.0)) == pytest.approx(expected, rel=1e-14)
    assert float(sheath.gamma_e(1.0, 1836.0)) == pytest.approx(4.49, abs=0.01)
    assert SheathClosure(electron_heat_transmission=5.0).gamma_e(1.0, 1836.0) == 5.0
    with pytest.raises(ValueError):
        SheathClosure(ion_heat_transmission=2.0)


def test_bohm_state_sign_inequality_momentum_and_current():
    rng = np.random.default_rng(0)
    center = np.stack(
        [rng.uniform(0.5, 2.0, 64), rng.uniform(0.5, 2.0, 64), rng.uniform(0.5, 2.0, 64),
         rng.uniform(-3.0, 3.0, 64), rng.uniform(-3.0, 3.0, 64)], axis=-1)
    trace = center.copy()
    for s in (1.0, -1.0):
        wall = np.asarray(bohm_wall_state(trace, center, s, TAU))
        cs = np.asarray(sound_speed(trace[:, 1], trace[:, 2], TAU))
        assert np.all(s * wall[:, 3] >= cs * (1 - 1e-14))  # Bohm inequality, outward
        np.testing.assert_allclose(wall[:, 3], np.where(s * center[:, 3] > cs, center[:, 3], s * cs))
        np.testing.assert_array_equal(wall[:, 3], wall[:, 4])  # local ambipolar
        n, Te, Ti, Vi, Ve = center.T
        np.testing.assert_allclose(
            wall[:, 0] * (wall[:, 3] ** 2 + wall[:, 1] + TAU * wall[:, 2]),
            n * Vi * Vi + n * (Te + TAU * Ti), rtol=1e-13)
        fluxes = wall_face_fluxes(wall, s, TAU, 4.5, 2.5)
        np.testing.assert_array_equal(np.asarray(fluxes.current), 0.0)
        assert np.all(np.asarray(fluxes.particle) > 0.0)  # always outward


def test_wall_leg_removes_exactly_the_contract_fluxes():
    """No double counting: inventory rates of a wall leg are the flux differences."""

    rng = np.random.default_rng(1)
    center = np.stack(
        [rng.uniform(0.5, 2.0, 32), rng.uniform(0.5, 2.0, 32), rng.uniform(0.5, 2.0, 32),
         rng.uniform(-0.8, 0.8, 32), rng.uniform(-0.8, 0.8, 32)], axis=-1)
    dx, ge, gi = 0.03, 4.49, 3.0
    for s in (1.0, -1.0):
        wall = bohm_wall_state(center, center, s, TAU)
        rate = np.asarray(sheath_wall_leg_residual(wall, center, dx, s, TAU, ge, gi))
        flux = wall_face_fluxes(wall, s, TAU, ge, gi, center[:, 3])
        n, Te, Ti, Vi, Ve = center.T
        out = -1.0 / dx  # outward flux through the leg, minus the owner-centre flux along b
        np.testing.assert_allclose(rate[:, 0], out * (flux.particle - s * n * Vi), rtol=1e-12)
        np.testing.assert_allclose(
            1.5 * (Te * rate[:, 0] + n * rate[:, 1]),
            out * (flux.electron_energy - s * 2.5 * n * Te * Vi), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            1.5 * TAU * (Ti * rate[:, 0] + n * rate[:, 2]),
            out * (flux.ion_energy - s * 2.5 * TAU * n * Ti * Vi), rtol=1e-12, atol=1e-12)
        # momentum flux is continuous across the half leg (owner subsonic)
        np.testing.assert_allclose(Vi * rate[:, 0] + n * rate[:, 3], 0.0, atol=1e-10)
        np.testing.assert_array_equal(rate[:, 4], 0.0)


def _flux_tube(cells: int, *, isothermal: bool, mu: float = 25.0, periods: int = 10):
    length = 2.0
    dx = length / cells
    source = 0.5 * np.ones(cells)
    heating = 2.0 * np.ones(cells)
    sheath = SheathClosure()
    ge, gi = float(sheath.gamma_e(TAU, mu)), sheath.ion_heat_transmission
    backward = jnp.zeros(cells, bool).at[0].set(True)
    forward = jnp.zeros(cells, bool).at[-1].set(True)
    rows = jnp.asarray((1.0, 1.0, 1.0, 1.0, 0.0))

    def rates(u):
        minus = jnp.concatenate([u[:1], u[:-1]])
        plus = jnp.concatenate([u[1:], u[-1:]])
        bst = jnp.where(backward[:, None], bohm_wall_state(minus, u, -1.0, TAU), minus)
        fst = jnp.where(forward[:, None], bohm_wall_state(plus, u, 1.0, TAU), plus)
        kw = dict(backward_wall=backward, forward_wall=forward,
                  backward_wall_state=bst, forward_wall_state=fst)
        r, _ = parallel_target_row_material_residual(u, minus, plus, dx, dx, TAU, mu, **kw)
        rw, _, info = parallel_short_wall_material_data(u, minus, plus, dx, dx, TAU, mu, **kw)
        r = r + rw
        r = r + jnp.where(backward[:, None], rows * (
            sheath_wall_leg_residual(bst, u, dx, -1.0, TAU, ge, gi) - info["backward_residual"]), 0.0)
        r = r + jnp.where(forward[:, None], rows * (
            sheath_wall_leg_residual(fst, u, dx, 1.0, TAU, ge, gi) - info["forward_residual"]), 0.0)
        n, Te, Ti, V = u[:, 0], u[:, 1], u[:, 2], u[:, 3]
        r = r + jnp.stack([source + 0 * n, (heating / 2 - 1.5 * Te * source) / (1.5 * n),
                           (heating / 2 - 1.5 * Ti * source) / (1.5 * n), -source * V / n,
                           -source * V / n], -1)
        if isothermal:
            r = r.at[:, 1:3].set(0.0)
        return r, bst, fst

    @partial(jax.jit, static_argnums=2)
    def advance(u, dt, steps):
        def body(u, _):
            u = u + dt * rates(u)[0]
            return u.at[:, 4].set(u[:, 3]), None  # quasineutral ambipolar flow
        return jax.lax.scan(body, u, None, length=steps)[0]

    u = jnp.stack([jnp.ones(cells)] * 3 + [jnp.zeros(cells)] * 2, -1)
    dt = 0.2 * dx / (math.sqrt(3.0 * mu) + 2.0)
    for _ in range(periods):
        u = advance(u, dt, int(2.0 / dt))
    r, bst, fst = rates(u)
    walls = [wall_face_fluxes(bst[0], -1.0, TAU, ge, gi, u[0, 3]),
             wall_face_fluxes(fst[-1], 1.0, TAU, ge, gi, u[-1, 3])]
    total = {k: float(sum(getattr(w, k) for w in walls)) for k in walls[0]._fields}
    return dict(
        u=np.asarray(u), dx=dx, residual=float(jnp.max(jnp.abs(r[:, :4]))),
        wall_state=np.asarray(fst[-1]), total=total, source=float(source.sum() * dx),
        heating=float(heating.sum() * dx),
        dissipation=float(0.5 * np.sum(source * np.asarray(u[:, 3]) ** 2) * dx),
    )


@pytest.mark.slow
def test_two_point_flux_tube_isothermal_converges_to_analytic_profile():
    """Exact steady isothermal solution with Bohm targets (L2-rung substitute).

    Uniform source S on [0, 2L]: Gamma = S (s - L), n (1 + M^2) = n_u,
    M = 1 at the targets, so n_u = 2 S L / c_s, n_t = n_u / 2 and
    M(s) = (1 - sqrt(1 - 4 a^2)) / (2 a), a = |s - L| / (2 L).
    """

    errors = []
    for cells in (10, 20, 40):
        run = _flux_tube(cells, isothermal=True)
        assert run["residual"] < 1e-6
        u, dx = run["u"], run["dx"]
        cs = math.sqrt(2.0)
        s = (np.arange(cells) + 0.5) * dx
        a = np.abs(s - 1.0) / 2.0
        mach = np.where(a > 0, (1 - np.sqrt(np.maximum(1 - 4 * a * a, 0))) / np.maximum(2 * a, 1e-300), 0)
        n_exact = (2 * 0.5 * 1.0 / cs) / (1 + mach * mach)
        errors.append(float(np.sqrt(np.mean((u[:, 0] - n_exact) ** 2))))
        # target: sonic, n_t = n_u/2 with n_u = 2 S L / c_s, particle ledger closed
        assert run["wall_state"][3] == pytest.approx(cs, rel=1e-12)
        n_u = 1.0 / cs
        assert run["wall_state"][0] == pytest.approx(n_u / 2, rel=0.03)
        assert run["total"]["particle"] == pytest.approx(run["source"], rel=2e-3)
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    # Observed order about 0.55 (errors 3.7e-2, 2.4e-2, 1.7e-2): the exact
    # profile has a square-root singularity at the sonic target and the wall
    # legs are first order, so only convergence is asserted.
    assert np.all(orders > 0.4), (errors, orders)


@pytest.mark.slow
def test_flux_tube_particle_and_energy_ledger_with_sheath_heat_flux():
    """Sources = wall losses at steady state, with q = gamma n T c_s at the target."""

    run = _flux_tube(20, isothermal=False)
    assert run["residual"] < 1e-4
    total = run["total"]
    assert total["particle"] == pytest.approx(run["source"], rel=3e-3)
    wall = run["wall_state"]
    cs = math.sqrt(wall[1] + TAU * wall[2])
    assert wall[3] == pytest.approx(cs, rel=1e-12)
    ge = float(SheathClosure().gamma_e(TAU, 25.0))
    # per-target sheath heat flux q_e = gamma_e n_t Te_t c_s (total-gamma convention)
    half = wall_face_fluxes(wall, 1.0, TAU, ge, 2.5)
    assert float(half.electron_energy) == pytest.approx(ge * wall[0] * wall[1] * cs, rel=1e-12)
    # total energy: heating = thermal sheath losses + flow kinetic energy + source dissipation
    lost = total["electron_energy"] + total["ion_energy"] + total["kinetic_energy"] + run["dissipation"]
    assert lost == pytest.approx(run["heating"], rel=1e-2)
