"""Bohm sheath wall closure and wall-face flux contract of the HSX backend.

The flux-tube tests drive the production five-field parallel material
operator (explicit legs plus the selected wall legs with the sheath rows)
on a 1D open field line with walls at both ends and a uniform particle and
heat source, to steady state.
"""

from __future__ import annotations

import math
import os
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.fci_braginskii.native.fci_drb_EB_rhs import LocalFciDrbEBRhs
from drbx.fci_braginskii.native.fci_support_pair import build_support_pair, build_weighted_negative_adjoint
from drbx.fci_braginskii.native.fci_wall_closure import (
    SheathClosure,
    bohm_wall_state,
    current_gamma_e,
    sheath_current,
    sheath_lambda,
    sheath_tube_lengths,
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


def test_sheath_entrance_is_not_hotter_than_its_owner():
    """A hotter wall-endpoint trace must not set the energy flux of a cooler owner."""

    center = np.array([[0.5, 0.08, 0.6, 0.0, 0.0]])
    hot = np.array([[0.5, 0.45, 0.8, 0.0, 0.0]])
    cold = np.array([[0.5, 0.05, 0.4, 0.0, 0.0]])
    for s in (1.0, -1.0):
        wall = np.asarray(bohm_wall_state(hot, center, s, TAU))
        np.testing.assert_array_equal(wall[:, 1:3], center[:, 1:3])
        wall = np.asarray(bohm_wall_state(cold, center, s, TAU))
        np.testing.assert_array_equal(wall[:, 1:3], cold[:, 1:3])


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


def test_sheath_current_leg_removes_the_electron_flux():
    """With a wall current the density and electron energy legs carry n Ve = n Vi - j."""

    rng = np.random.default_rng(2)
    center = np.stack(
        [rng.uniform(0.5, 2.0, 16), rng.uniform(0.5, 2.0, 16), rng.uniform(0.5, 2.0, 16),
         rng.uniform(-0.8, 0.8, 16), rng.uniform(-0.8, 0.8, 16)], axis=-1)
    dx, gi = 0.03, 2.5
    ge = np.asarray(current_gamma_e(rng.uniform(-1.0, 4.0, 16)))
    for s in (1.0, -1.0):
        wall = bohm_wall_state(center, center, s, TAU)
        j = rng.uniform(-2.0, 2.0, 16)
        rate = np.asarray(sheath_wall_leg_residual(
            wall, center, dx, s, TAU, ge, gi, conservative_bulk=True, wall_current=j))
        flux = wall_face_fluxes(wall, s, TAU, ge, gi, center[:, 3], wall_current=j)
        n, Te, Ti, Vi, _ = center.T
        out = -1.0 / dx
        # conservative bulk: the leg is the wall flux alone, here the electron flux
        np.testing.assert_allclose(rate[:, 0], out * flux.particle, rtol=1e-12)
        np.testing.assert_allclose(
            1.5 * (Te * rate[:, 0] + n * rate[:, 1]), out * flux.electron_energy,
            rtol=1e-12, atol=1e-12)
        # ions keep the ion flow: gamma_i n_w Ti_w Vi_w leaves the ion energy
        np.testing.assert_allclose(
            1.5 * TAU * (Ti * rate[:, 0] + n * rate[:, 2]),
            out * TAU * gi * wall[:, 2] * wall[:, 0] * wall[:, 3] * s,
            rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(flux.particle, s * (wall[:, 0] * wall[:, 3] - j))
    assert float(current_gamma_e(-2.0)) == 2.0 and float(current_gamma_e(2.84)) == pytest.approx(4.84)


def test_conservative_bulk_leg_is_the_wall_flux_alone():
    """With the flux-form bulk, each leg carries only the contract flux (plus the mirror share for Vi)."""

    rng = np.random.default_rng(2)
    center = np.stack(
        [rng.uniform(0.5, 2.0, 32), rng.uniform(0.5, 2.0, 32), rng.uniform(0.5, 2.0, 32),
         rng.uniform(-0.8, 0.8, 32), rng.uniform(-3.0, 3.0, 32)], axis=-1)
    dx, ge, gi = rng.uniform(0.01, 0.1, 32), 4.49, 3.0
    for s in (1.0, -1.0):
        wall = bohm_wall_state(center, center, s, TAU)
        rate = np.asarray(sheath_wall_leg_residual(wall, center, dx, s, TAU, ge, gi, conservative_bulk=True))
        flux = wall_face_fluxes(wall, s, TAU, ge, gi, center[:, 3])
        n, Te, Ti, Vi, _ = center.T
        np.testing.assert_allclose(rate[:, 0], -np.asarray(flux.particle) / dx, rtol=1e-13)
        np.testing.assert_allclose(1.5 * (Te * rate[:, 0] + n * rate[:, 1]),
                                   -np.asarray(flux.electron_energy) / dx, rtol=1e-12)
        np.testing.assert_allclose(1.5 * TAU * (Ti * rate[:, 0] + n * rate[:, 2]),
                                   -np.asarray(flux.ion_energy) / dx, rtol=1e-12)
        # momentum leg: signed wall momentum flux less the leg's mirror force s p_c/dx
        mirror = s * n * (Te + TAU * Ti) / dx
        np.testing.assert_allclose(Vi * rate[:, 0] + n * rate[:, 3],
                                   -s * np.asarray(flux.parallel_momentum) / dx + mirror, rtol=1e-11, atol=1e-11)


def test_tube_lengths_use_the_interior_leg():
    dxm, dxp = jnp.asarray([1.0, 0.2, 0.3, 0.5]), jnp.asarray([0.1, 2.0, 0.4, 0.5])
    backward = jnp.asarray([False, True, True, False])
    forward = jnp.asarray([True, False, True, False])
    tube_minus, tube_plus = (np.asarray(t) for t in sheath_tube_lengths(dxm, dxp, backward, forward))
    assert tube_plus[0] == 1.0 and tube_minus[1] == 2.0  # single wall leg: interior length
    assert tube_minus[2] == tube_plus[2] == pytest.approx(0.7)  # both legs on walls


def test_conservative_parallel_rates_are_sum_free_with_reported_sources():
    """Any weighted pair D = -M^-1 G^T M with G(1) = 0: no parallel loss in the bulk."""

    rng = np.random.default_rng(3)
    cells = 40
    mass = rng.uniform(0.5, 2.0, cells)
    G = rng.normal(size=(cells, cells))
    G -= G.sum(axis=1, keepdims=True) / cells  # constant-exact gradient
    D = -(G.T * mass[None, :]) / mass[:, None]
    gradient = lambda f: jnp.einsum("ij,...j->...i", G, f)
    divergence = lambda f: jnp.einsum("ij,...j->...i", D, f)
    primitive = {k: jnp.asarray(rng.uniform(lo, hi, cells)) for k, lo, hi in (
        ("density", 0.5, 2.0), ("Te", 0.5, 2.0), ("Ti", 0.5, 2.0), ("Vi", -1.0, 1.0), ("Ve", -3.0, 3.0))}
    host = SimpleNamespace(parameters=SimpleNamespace(tau=TAU, mi_over_me=25.0))
    rates, sources = LocalFciDrbEBRhs._conservative_parallel_rates(host, primitive, gradient, divergence)
    n, te, ti, vi = (np.asarray(primitive[k]) for k in ("density", "Te", "Ti", "Vi"))
    dn, dte, dti, dvi, dve = np.asarray(rates).T
    assert abs(np.sum(mass * dn)) < 1e-12
    np.testing.assert_allclose(np.sum(mass * 1.5 * (te * dn + n * dte)), np.sum(mass * sources[:, 0]), atol=1e-11)
    np.testing.assert_allclose(np.sum(mass * 1.5 * TAU * (ti * dn + n * dti)), np.sum(mass * sources[:, 1]),
                               atol=1e-11)
    # momentum: d(n Vi) is sum-free up to the reported mirror/current force
    np.testing.assert_allclose(np.sum(mass * (vi * dn + n * dvi)), np.sum(mass * sources[:, 2]), atol=1e-11)


@pytest.mark.slow
def test_hsx_sheath_step_ledger_closes(tmp_path):
    """Real HSX (set DRBX_TEST_GEOMETRY_BUNDLE): applied wall flux is the only parallel loss."""

    bundle = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
    if bundle is None:
        pytest.skip("set DRBX_TEST_GEOMETRY_BUNDLE to run the HSX sheath ledger")
    from drbx.cli import main

    deck = tmp_path / "hsx.toml"
    deck.write_text(
        '[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\n'
        f'geometry = "{bundle}"\noutput = "{tmp_path / "out" / "hsx"}"\n'
        "final_time = 0.0015\nnum_steps = 2\nsave_every = 2\n"
    )
    assert main(["run", str(deck)]) in (0, None)
    data = np.load(tmp_path / "out" / "hsx.npz")
    rows, columns = data["wall_ledger"], list(data["wall_ledger_columns"])
    col = {name: rows[:, k] for k, name in enumerate(columns)}
    wall, other = col["step_particle_wall"][1:], col["step_particle_other"][1:]
    assert np.all(wall > 0.0)
    np.testing.assert_allclose(np.diff(col["particle_inventory"]), other - wall, rtol=0, atol=1e-9 * wall.max())
    # what is not the wall is perpendicular transport only (the parallel bulk is sum-free)
    assert np.all(np.abs(other) < 1e-3 * wall)
    for kind in ("electron", "ion"):
        inventory = np.diff(col[f"{kind}_energy_inventory"])
        applied = (col[f"step_{kind}_energy_other"] - col[f"step_{kind}_energy_wall"]
                   + col[f"step_{kind}_energy_product"])[1:]
        np.testing.assert_allclose(inventory, applied, rtol=0, atol=1e-9 * abs(inventory).max())
    # parallel momentum P = sum V n Vi: flux-form Vi row, same applied-step identity
    momentum = np.diff(col["momentum_inventory"])
    applied = (col["step_momentum_other"] - col["step_momentum_wall"] + col["step_momentum_product"])[1:]
    scale = max(abs(momentum).max(), abs(col["step_momentum_other"][1:]).max())
    np.testing.assert_allclose(momentum, applied, rtol=0, atol=1e-9 * scale)
    # the applied flux is the contract flux of the step states to first order in dt
    dt = np.diff(col["time"])
    contract = 0.5 * (col["particle"][1:] + col["particle"][:-1]) * dt
    np.testing.assert_allclose(wall, contract, rtol=5e-3)


def flux_tube_bulk_rates(cells: int, dx: float, mu: float, epsilon: float = 0.1):
    """``rates(u, ge, gi)``: pair-based bulk plus production wall rows of a 1D open field line.

    The support pair is built as on HSX (wall rows close with the owner value), the bulk
    rows are ``LocalFciDrbEBRhs._conservative_parallel_rates`` with the forward-leg
    face-flux dissipation ``epsilon``, and each wall leg is the wall flux alone.
    """

    core = jnp.ones(cells, bool)
    mass = jnp.full((cells,), dx)

    def divergence0(q):
        minus = jnp.concatenate([q[:1], q[:-1]])
        plus = jnp.concatenate([q[1:], q[-1:]])
        return (plus - minus) / (2.0 * dx)

    gradient, divergence = build_support_pair(divergence0, mass, active=core, core=core)
    leg = jnp.ones(cells, bool).at[-1].set(False)
    forward_gradient = lambda f: jnp.where(leg, (jnp.roll(f, -1) - f) / dx, 0.0)
    adjoint = build_weighted_negative_adjoint(forward_gradient, mass, mass, dual_active=leg)
    laplacian = lambda speed, f: adjoint(epsilon * speed * dx * forward_gradient(f))
    host = SimpleNamespace(parameters=SimpleNamespace(tau=TAU, mi_over_me=mu))
    backward = jnp.zeros(cells, bool).at[0].set(True)
    forward = jnp.zeros(cells, bool).at[-1].set(True)

    def rates(u, ge, gi):
        minus = jnp.concatenate([u[:1], u[:-1]])
        plus = jnp.concatenate([u[1:], u[-1:]])
        bst = jnp.where(backward[:, None], bohm_wall_state(minus, u, -1.0, TAU), minus)
        fst = jnp.where(forward[:, None], bohm_wall_state(plus, u, 1.0, TAU), plus)
        primitive = {k: u[:, i] for i, k in enumerate(("density", "Te", "Ti", "Vi", "Ve"))}
        r, _ = LocalFciDrbEBRhs._conservative_parallel_rates(host, primitive, gradient, divergence, laplacian)
        r = r.at[:, 4].set(0.0)  # quasineutral ambipolar flow slaves Ve to Vi
        r = r + jnp.where(backward[:, None], sheath_wall_leg_residual(bst, u, dx, -1.0, TAU, ge, gi, conservative_bulk=True), 0.0)
        r = r + jnp.where(forward[:, None], sheath_wall_leg_residual(fst, u, dx, 1.0, TAU, ge, gi, conservative_bulk=True), 0.0)
        return r, bst, fst

    return rates


@pytest.mark.slow
def test_hsx_zero_flow_wall_is_a_closed_box_for_particles_and_thermal_energy(tmp_path):
    """Real HSX (set DRBX_TEST_GEOMETRY_BUNDLE): no wall flux, parallel bulk sum-free."""

    bundle = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
    if bundle is None:
        pytest.skip("set DRBX_TEST_GEOMETRY_BUNDLE to run the HSX zero-flow ledger")
    from drbx.cli import main

    deck = tmp_path / "hsx.toml"
    deck.write_text(
        '[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\n'
        f'geometry = "{bundle}"\noutput = "{tmp_path / "out" / "hsx"}"\n'
        'wall_closure = "zero-flow"\nfinal_time = 0.0015\nnum_steps = 2\nsave_every = 2\n'
    )
    assert main(["run", str(deck)]) in (0, None)
    data = np.load(tmp_path / "out" / "hsx.npz")
    col = {name: data["wall_ledger"][:, k] for k, name in enumerate(data["wall_ledger_columns"])}
    inventory = col["particle_inventory"]
    assert np.all(np.abs(col["step_particle_wall"][1:]) <= 1e-12 * inventory[0])
    np.testing.assert_allclose(np.diff(inventory), col["step_particle_other"][1:], rtol=0, atol=1e-12 * inventory[0])
    for kind in ("electron", "ion"):
        assert np.all(np.abs(col[f"step_{kind}_energy_wall"][1:]) <= 1e-12 * col[f"{kind}_energy_inventory"][0])


def _flux_tube(cells: int, *, isothermal: bool, mu: float = 25.0, periods: int = 10):
    length = 2.0
    dx = length / cells
    source = 0.5 * np.ones(cells)
    heating = 2.0 * np.ones(cells)
    sheath = SheathClosure()
    ge, gi = float(sheath.gamma_e(TAU, mu)), sheath.ion_heat_transmission
    bulk = flux_tube_bulk_rates(cells, dx, mu)

    def rates(u):
        r, bst, fst = bulk(u, ge, gi)
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
    # particle and thermal ledgers close to round-off; the kinetic-thermal exchange is central-difference
    # truncation (1.2% at 20 cells, 0.9% at 40, 0.7% at 80)
    assert lost == pytest.approx(run["heating"], rel=2e-2)
    assert total["particle"] == pytest.approx(run["source"], rel=1e-4)


def test_sheath_lambda_and_current_linearization():
    """Lambda = ln[(1-d)/sqrt(2 pi (me/mi)(1+Ti/Te))]; j vanishes at phi = Lambda Te."""

    assert float(sheath_lambda(0.0, 1836.0)) == pytest.approx(2.8387, abs=1e-4)
    assert float(sheath_lambda(1.0, 1836.0)) == pytest.approx(2.4922, abs=1e-4)
    assert float(sheath_lambda(1.0, 1836.0, 0.5)) == pytest.approx(2.4922 + math.log(0.5), abs=1e-4)
    lam, te = 2.5, 1.3
    for s in (1.0, -1.0):
        j, dj = sheath_current(0.7, 1.1, te, lam * te, lam, s)
        assert abs(float(j)) < 1e-15 and float(s * dj) > 0.0
        phi = jnp.linspace(-1.0, 6.0, 9)
        fd = jax.vmap(jax.grad(lambda p, s=s: sheath_current(0.7, 1.1, te, p, lam, s)[0]))(phi)
        np.testing.assert_allclose(fd, sheath_current(0.7, 1.1, te, phi, lam, s)[1], rtol=1e-12)
    with pytest.raises(ValueError):
        SheathClosure(potential="floating")


def _sheath_flux_tube(dt, steps, *, implicit, cells=16):
    """1D flux tube: Ohm's law current, cell capacitance and sheath walls.

    ``omega = -C phi`` per cell, ``d omega/dt + D(j) = 0`` with the interior
    current ``j = sigma (G(phi) - G(pe)/n)`` (electron parallel force
    balance) and the sheath-limited endpoint current at both walls.  The
    implicit step is the production structure: the sheath current is
    linearized about the stage potential, the affine part goes to the right
    side and ``J phi`` to the operator.
    """

    length, sigma, cap, lam = 2.0, 5.0, 0.2, 2.5
    dx = length / cells
    x = (np.arange(cells) + 0.5) * dx
    n = 1.0 - 0.4 * ((x - 1.0) / 1.0) ** 2  # two-point-like density
    te = 1.0 + 0.0 * x
    cs = np.sqrt(2.0 * te)
    # face current from cell potentials: j_{i+1/2} = sigma [(phi_{i+1}-phi_i) - Te ln(n_{i+1}/n_i)] / dx
    lnn = np.log(n)
    grad = np.zeros((cells - 1, cells))
    grad[np.arange(cells - 1), np.arange(cells - 1)] = -1.0 / dx
    grad[np.arange(cells - 1), np.arange(1, cells)] = 1.0 / dx
    interior = sigma * grad
    pressure = -sigma * te[0] * (grad @ lnn)

    def divergence(j_faces, j_backward, j_forward):
        faces = np.concatenate([[j_backward], j_faces, [j_forward]])
        return (faces[1:] - faces[:-1]) / dx

    phi = np.zeros(cells)
    for _ in range(steps):
        jb, djb = (float(v) for v in sheath_current(n[0], cs[0], te[0], phi[0], lam, -1.0))
        jf, djf = (float(v) for v in sheath_current(n[-1], cs[-1], te[-1], phi[-1], lam, 1.0))
        if implicit:
            # -C (phi' - phi)/dt + D(j(phi')) = 0, j linear in phi' (interior exact, walls linearized)
            A = np.zeros((cells, cells))
            for k in range(cells):
                e = np.zeros(cells); e[k] = 1.0
                A[:, k] = divergence(interior @ e, djb * e[0], djf * e[-1])
            rhs = -divergence(pressure, jb - djb * phi[0], jf - djf * phi[-1])
            phi = np.linalg.solve(cap / dt * np.eye(cells) - A, cap / dt * phi - rhs)
        else:
            phi = phi + dt / cap * divergence(interior @ phi + pressure, jb, jf)
        if not np.all(np.isfinite(phi)) or np.max(np.abs(phi)) > 1e6:
            return None
    jb = float(sheath_current(n[0], cs[0], te[0], phi[0], lam, -1.0)[0])
    jf = float(sheath_current(n[-1], cs[-1], te[-1], phi[-1], lam, 1.0)[0])
    return {"phi": phi, "lam": lam, "te": te, "n": n, "wall_current": (jb, jf),
            "interior_current": interior @ phi + pressure}


def test_flux_tube_floating_sheath_potential_drop():
    """phi_wall - phi_entrance = -Lambda Te and zero current at steady state."""

    run = _sheath_flux_tube(dt=10.0, steps=200, implicit=True)
    phi, lam, te, n = run["phi"], run["lam"], run["te"], run["n"]
    # wall at phi = 0: the entrance (owner) potential is Lambda Te
    assert 0.0 - phi[0] == pytest.approx(-lam * te[0], rel=1e-10)
    assert 0.0 - phi[-1] == pytest.approx(-lam * te[-1], rel=1e-10)
    assert max(abs(v) for v in run["wall_current"]) < 1e-10
    assert np.max(np.abs(run["interior_current"])) < 1e-10
    # Boltzmann electrons along the tube: phi - Te ln n is constant
    np.testing.assert_allclose(phi - te * np.log(n), lam * te[0] - te[0] * np.log(n[0]), atol=1e-10)
    # the explicit wall current (the PR 1b attempt) diverges at the same step
    assert _sheath_flux_tube(dt=10.0, steps=200, implicit=False) is None


@pytest.mark.slow
def test_hsx_bohm_default_runs_past_the_old_te_collapse(tmp_path):
    """Shipped-coil HSX at 16-cubed, default deck to t = 0.18.

    A wall-endpoint trace hotter than the owner used to drain a wall-bounded
    cell's energy at the neighbours' temperature: Te fell to 0.07 and the
    run failed near t = 0.17 (dt = 7.5e-4).  The geometry is built from the
    deck ``[geometry]`` table and cached under ``output/hsx_geometry``.
    """

    from pathlib import Path

    from drbx.cli import main

    cache = Path(__file__).resolve().parents[2] / "output" / "hsx_geometry"
    deck = tmp_path / "hsx.toml"
    deck.write_text(
        '[model]\nbackend = "fci_braginskii"\n'
        "[geometry]\nresolution = [16, 16, 16]\nmetric_mesh_shape = [16, 16, 16]\n"
        "metric_poloidal_modes = 6\nmetric_toroidal_modes = 3\ntoroidal_modes = 3\n"
        f'cache_dir = "{cache}"\n'
        f'[fci_braginskii]\noutput = "{tmp_path / "out" / "hsx"}"\n'
        "final_time = 0.18\nnum_steps = 240\nsave_every = 240\n"
    )
    assert main(["run", str(deck)]) in (0, None)
    data = np.load(tmp_path / "out" / "hsx.npz")
    for name in ("density", "Te", "Ti", "Vi", "Ve", "phi"):
        assert np.all(np.isfinite(data[name]))
    assert data["Te"][-1].min() > 0.1
    assert np.abs(data["Ve"][-1]).max() < 20.0
