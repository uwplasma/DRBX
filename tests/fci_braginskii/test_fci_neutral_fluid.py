"""Fluid neutrals on the FCI grid (``native/fci_neutral_fluid.py``, stage N1)."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.fci_braginskii.native import fci_neutral_fluid as FL
from drbx.fci_braginskii.native import fci_neutrals as F
from tests.fci_braginskii.test_fci_neutrals import PARAMS, _geometry, _random_plasma, annulus, sheared_slab

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


def _transport(grid, Nn, Tn, dp, kappa, dt, un=None, eta=None, source=None, params=PARAMS):
    zero = jnp.zeros_like(Nn)
    return FL.neutral_transport(Nn, Tn, zero if un is None else un, dp, kappa, zero if eta is None else eta, dt, grid,
                                params, source)[:4]


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=IDS)
def test_transport_reduces_to_pressure_diffusion_when_the_inertial_response_equals_it(geometry):
    # at rest, the parallel flux is -dt b.grad p and the diffusive one -D_p grad p: equal for D_p = dt (model 1)
    grid = F.build_neutral_grid(geometry)
    _, _, _, _, Nn = _random_plasma(geometry.shape, 1)
    dt, tn = 0.05, jnp.full(geometry.shape, 0.4)
    n_new, _, _, rel = _transport(grid, Nn, tn, jnp.full(geometry.shape, dt), 0.0 * Nn, dt)
    diffusion = jax.jacfwd(lambda v: F.neutral_diffusion_rate(v.reshape(geometry.shape), 0.4 * dt * jnp.ones(geometry.shape), grid).ravel())
    reference = jnp.linalg.solve(jnp.eye(Nn.size) - dt * diffusion(Nn.ravel()), Nn.ravel()).reshape(geometry.shape)
    np.testing.assert_allclose(np.asarray(n_new), np.asarray(reference), rtol=1e-7)
    assert float(jnp.max(rel)) < 1e-9


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=IDS)
def test_transport_conserves_nuclei_exactly_and_total_energy_to_truncation(geometry):
    grid = F.build_neutral_grid(geometry)
    i, j, k = (jnp.arange(m) / m for m in geometry.shape)
    smooth = lambda a, b, c: jnp.sin(2 * jnp.pi * (a * i[:, None, None] + b * j[None, :, None] + c * k[None, None, :]))
    Nn, Te = 0.3 + 0.1 * smooth(1, 0, 1), 0.5 + 0.2 * smooth(0, 1, 1)
    dp, un, source = 0.05 + 0.02 * smooth(1, 1, 0), 0.3 * smooth(1, 1, 1), 0.2 + 0.1 * smooth(0, 1, 2)
    v = grid.volume

    def energy(n, u, t):
        return float(jnp.sum(v * (1.5 * n * t + 0.5 * n * u**2)))

    errors = []
    for dt in (0.02, 0.01):
        n_new, u_new, t_new, rel = _transport(grid, Nn, Te, dp, 2.5 * dp * Nn * Te, dt, un, Nn * Te * dp, source)
        nuclei = float(jnp.sum(v * Nn))
        assert abs(float(jnp.sum(v * n_new)) - nuclei - dt * float(jnp.sum(v * source))) < 1e-13 * nuclei
        assert float(jnp.max(rel)) < 1e-8
        errors.append(abs(energy(n_new, u_new, t_new) - energy(Nn, un, Te) - dt * float(jnp.sum(v * 1.5 * Te * source))))
    assert max(errors) < 3e-3 * energy(Nn, un, Te), errors  # truncation error of the kinetic-work/viscous-heating terms


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=IDS)
def test_parallel_acceleration_runs_down_the_pressure_gradient_and_the_flow_is_advected(geometry):
    grid = F.build_neutral_grid(geometry)
    shape = geometry.shape
    ramp = (1.0 + 0.5 * jnp.sin(jnp.arange(shape[0]) * 0.5))[:, None, None] * jnp.ones(shape)
    tn = jnp.full(shape, 0.3)
    dt = 0.01
    _, u_new, _, rel = _transport(grid, ramp, tn, 0.0 * ramp, 0.0 * ramp, dt)
    gp = FL._par_grad(ramp * tn, grid)
    assert float(jnp.sum(grid.volume * u_new * gp)) < 0.0  # accelerates against b.grad p
    np.testing.assert_allclose(np.asarray(u_new * ramp)[1:-1], -dt * np.asarray(gp)[1:-1], rtol=0.5)  # M ~ -dt b.grad p
    # uniform pressure and a flow: density changes only through div(n u b) (nuclei conserved) and u stays finite
    n2, u2, _, rel = _transport(grid, ramp, 0.3 / ramp, 0.0 * ramp, 0.0 * ramp, dt, un=0.2 * jnp.cos(ramp))
    assert float(jnp.max(jnp.abs(n2 - ramp))) > 1e-4 and bool(jnp.all(jnp.isfinite(u2)))
    assert abs(float(jnp.sum(grid.volume * (n2 - ramp)))) < 1e-13 * float(jnp.sum(grid.volume * ramp))


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=IDS)
def test_wall_absorption_removes_exactly_the_reported_nuclei_and_energy(geometry):
    grid = F.build_neutral_grid(geometry)
    _, _, Te, _, _ = _random_plasma(geometry.shape, 3)
    Nn, zero, v, dt = 0.1 / Te, jnp.zeros(geometry.shape), grid.volume, 0.02  # uniform pressure: no drive but the wall
    closed = FL.neutral_transport(Nn, Te, zero, zero, zero, zero, dt, grid, PARAMS)
    assert float(closed[4][0]) == 0.0 and abs(float(jnp.sum(v * (closed[0] - Nn)))) < 1e-13 * float(jnp.sum(v * Nn))
    params = dataclasses.replace(PARAMS, wall_absorption=0.4, wall_energy_reflection=0.3)
    n_new, u_new, t_new, rel, wall = FL.neutral_transport(Nn, Te, zero, zero, zero, zero, dt, grid, params)
    assert float(wall[0]) > 0.0 and float(wall[1]) > float(wall[0])  # each absorbed atom takes 2 T > 1
    assert abs(float(jnp.sum(v * (Nn - n_new))) - float(wall[0])) < 1e-13 * float(jnp.sum(v * Nn))
    assert float(jnp.max(rel)) < 1e-9
    w0, w1 = float(jnp.sum(v * 1.5 * Nn * Te)), float(jnp.sum(v * 1.5 * n_new * t_new))
    assert abs(w0 - w1 - float(wall[1])) < 0.05 * float(wall[1])  # up to the (small) kinetic work of the wall-driven flow


def test_nuclei_check_fails_on_a_one_percent_leak():
    # the check compares the nuclei change with an expectation built from the unleaked source
    geometry = sheared_slab()
    grid = F.build_neutral_grid(geometry)
    n, Vi, Te, Ti, Nn = _random_plasma(geometry.shape, 2)
    source, v, Tn = 0.1 * jnp.ones(geometry.shape), grid.volume, 0.05 + 0.3 * Te
    closure = []
    for leak in (0.0, 0.01):
        out = FL.neutral_fluid_step(n, Vi, Te, Ti, Nn, 0 * Nn, Tn, 0.02, grid, PARAMS, volume_source=(1 - leak) * source)
        change = float(jnp.sum(v * (out[0] + out[4] - n - Nn)))
        closure.append(abs(change - 0.02 * float(jnp.sum(v * source))) / (0.02 * float(jnp.sum(v * source))))
    assert closure[0] < 1e-12 and closure[1] == pytest.approx(0.01, rel=1e-6)


def test_inactive_cells_with_no_neutrals_stay_finite():
    geometry = sheared_slab()
    grid = F.build_neutral_grid(geometry)
    mask = jnp.arange(geometry.shape[1])[None, :, None] != 3  # a column of cells outside the domain
    grid = dataclasses.replace(grid, owner=grid.owner & mask, volume=jnp.where(mask, grid.volume, 0.0))
    n, Vi, Te, Ti, Nn = _random_plasma(geometry.shape, 2)
    Nn, Tn = jnp.where(mask, Nn, 0.0), 0.05 + 0.3 * Te
    out = FL.neutral_fluid_step(n, Vi, Te, Ti, Nn, jnp.zeros(geometry.shape), Tn, 0.02, grid, PARAMS)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in out)
    assert float(FL.flux_limiter(0.0, 0.0)) == 1.0


def test_viscosity_diffuses_a_velocity_jump_and_conserves_momentum_of_that_stage():
    geometry = sheared_slab()
    grid = F.build_neutral_grid(geometry)
    shape = geometry.shape
    nn, tn = jnp.full(shape, 0.5), jnp.full(shape, 0.3)  # uniform p: no pressure drive
    jump = jnp.where(jnp.arange(shape[0]) < shape[0] // 2, 1.0, -1.0)[:, None, None] * jnp.ones(shape)
    eta = jnp.full(shape, 0.2)
    _, u_new, _, rel = _transport(grid, nn, tn, 0.0 * nn, 0.0 * nn, 0.05, un=jump, eta=eta)
    assert float(jnp.max(jnp.abs(u_new))) < 1.0 and float(jnp.max(jnp.abs(u_new))) > 0.3
    v = grid.volume
    assert abs(float(jnp.sum(v * nn * (u_new - jump)))) < 1e-12 * float(jnp.sum(v * nn))


def test_pure_conduction_relaxes_a_hot_spot_without_changing_the_energy():
    geometry = sheared_slab()
    grid = F.build_neutral_grid(geometry)
    shape = geometry.shape
    tn = 0.2 + 0.5 * jnp.exp(-((jnp.arange(shape[0]) - 8.0) ** 2 / 6.0))[:, None, None] * jnp.ones(shape)
    nn = 0.06 / tn  # uniform pressure: no flow
    kappa = jnp.full(shape, 0.2)
    n_new, u_new, t_new, rel = _transport(grid, nn, tn, 0.0 * kappa, kappa, 0.1)
    np.testing.assert_allclose(np.asarray(n_new), np.asarray(nn), rtol=1e-12)
    assert float(jnp.max(jnp.abs(u_new))) < 1e-12
    assert float(jnp.max(t_new)) < float(jnp.max(tn)) and float(jnp.min(t_new)) > float(jnp.min(tn)) - 1e-12
    v = grid.volume
    assert abs(float(jnp.sum(v * nn * t_new)) - float(jnp.sum(v * nn * tn))) < 1e-13 * float(jnp.sum(v * nn * tn))
    assert float(rel[2]) < 1e-9


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=IDS)
def test_reactions_exchange_nuclei_momentum_and_energy_exactly(geometry):
    # sum_V (plasma + neutral): nuclei, momentum and thermal+kinetic energy are conserved; the electron loss is the only sink
    grid = F.build_neutral_grid(geometry)
    params = F.NeutralParameters(
        reference_density_m3=5.0e18, reference_temperature_ev=PARAMS.reference_temperature_ev, reaction_substeps=3
    )
    n, Vi, Te, Ti, Nn = _random_plasma(geometry.shape, 7)
    Tn, un = 0.05 + 0.3 * Te, 0.4 * jnp.cos(Nn * 40.0)
    v = grid.volume
    total = lambda *x: [float(a) for a in (
        F.inventories(*x[:5], grid, params, x[6], x[5])[0], jnp.sum(v * (x[0] * x[1] + x[4] * x[5])),
        sum(F.inventories(*x[:5], grid, params, x[6], x[5])[1:]))]
    before = total(n, Vi, Te, Ti, Nn, un, Tn)
    out = FL.neutral_reactions(n, Vi, Te, Ti, Nn, un, Tn, 0.02, grid, params)
    after = total(*out[:7])
    inc = out[7]
    assert abs(after[0] - before[0]) < 1e-14 * before[0]
    assert abs(after[1] - before[1]) < 1e-13 * abs(before[1]) + 1e-15
    assert float(inc[3]) > 0.0 and abs(after[2] + float(inc[3]) - before[2]) < 1e-12 * float(inc[3])
    assert float(jnp.min(out[4])) > 0.0 and float(jnp.max(jnp.abs(out[5] - un))) > 1e-4  # the friction acts


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=IDS)
def test_full_step_runs_conserves_nuclei_and_stays_finite(geometry):
    grid = F.build_neutral_grid(geometry)
    params = F.NeutralParameters(
        reference_density_m3=5.0e18, reference_temperature_ev=PARAMS.reference_temperature_ev, reaction_substeps=3
    )
    n, Vi, Te, Ti, Nn = _random_plasma(geometry.shape, 7)
    Tn, un, v = 0.05 + 0.3 * Te, jnp.zeros(geometry.shape), grid.volume
    for _ in range(4):
        nuclei = float(jnp.sum(v * (n + Nn)))
        n, Vi, Te, Ti, Nn, un, Tn, ledger, _ = FL.neutral_fluid_step(n, Vi, Te, Ti, Nn, un, Tn, 0.02, grid, params)
        assert abs(float(jnp.sum(v * (n + Nn))) - nuclei) < 1e-13 * nuclei
        assert float(ledger[8]) < 1e-9 and float(ledger[17]) < 1e-9 and abs(float(ledger[6])) < 1e-13 * nuclei
    assert float(jnp.min(Nn)) > 0.0 and float(jnp.min(Tn)) > 0.0 and float(jnp.max(jnp.abs(un))) > 1e-3


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
        out = FL.neutral_fluid_step(n, Vi, Te, scale * Ti, Nn, 0.1 * Vi, Tn, 0.02, grid, PARAMS)
        return jnp.sum(grid.volume * out[4] * (out[6] + out[5] ** 2))

    g = float(jax.grad(functional)(1.0))
    eps = 1e-5
    fd = (float(functional(1.0 + eps)) - float(functional(1.0 - eps))) / (2 * eps)
    assert np.isfinite(g) and g == pytest.approx(fd, rel=1e-4)


def _penetration(tau_scale=1.0, steps=121):
    """Neutrals entering a slab of frozen uniform plasma: ``(lambda_fit / lambda_pen, mean |dn/dt| / (nu_iz n), sink / source)``.

    In the diffusive limit ``D = T_n / nu`` and the steady profile decays as ``exp(-x/lambda_pen)`` with
    ``lambda_pen = sqrt(T_n / (nu_iz nu)) = lambda_mfp / sqrt(nu_iz / nu)``; the run starts on it.
    """

    shape = (96, 4, 4)

    def cov(U, V, W):
        e = np.zeros(U.shape + (3, 3))
        e[..., 0, 0] = e[..., 1, 1] = e[..., 2, 2] = 1.0
        return e

    n, Te, Ti, Tn = (jnp.full(shape, v) for v in (1.0, 0.12, 0.15, 0.15))
    nu_cx, nu_iz = (float(x[0, 0, 0]) for x in F.collision_frequencies(n, Te, Ti, PARAMS, Tn))
    lam = float(np.sqrt(0.15 / (nu_iz * (nu_cx + nu_iz))))
    length = 8.0 * lam
    geometry = _geometry(cov, shape, (length, 1.0, 1.0))
    grid, x = F.build_neutral_grid(geometry), np.asarray(geometry.grid.x.centers)
    dt, wall_density = 0.03 / (nu_cx + nu_iz), 0.3  # nu dt = 0.03: the explicit friction's splitting error is 3%
    flux = wall_density * nu_iz * lam
    source = jnp.zeros(shape).at[-1].set(flux / (length / shape[0]))
    profile = wall_density * np.exp(-(length - x) / lam)[:, None, None] * np.ones(shape)
    original = FL.transport_coefficients
    FL.transport_coefficients = lambda *a: tuple(tau_scale * c for c in original(*a))
    try:
        step = jax.jit(lambda nn, un, tn: FL.neutral_fluid_step(
            n, jnp.zeros(shape), Te, Ti, nn, un, tn, dt, grid, PARAMS, volume_source=source))
        nn, un, tn = jnp.asarray(profile), jnp.zeros(shape), Tn
        for _ in range(steps):
            previous = nn
            out = step(nn, un, tn)
            nn, un, tn = out[4:7]
    finally:
        FL.transport_coefficients = original
    window = (x > length - 4 * lam) & (x < length - 0.5 * lam)
    line = np.asarray(nn[:, 0, 0])
    slope = np.polyfit(x[window], np.log(line[window]), 1)[0]
    drift = np.asarray((nn - previous)[:, 0, 0] / previous[:, 0, 0])[window] / dt / nu_iz
    sink = nu_iz * float(jnp.sum(nn[:, 0, 0])) * length / shape[0]
    return 1.0 / slope / lam, float(np.mean(np.abs(drift))), sink / flux


def test_slab_penetration_length_matches_the_mean_free_path_scaling():
    ratio, drift, balance = _penetration()
    assert ratio == pytest.approx(1.0, abs=0.03) and drift < 0.1 and balance == pytest.approx(1.0, abs=0.02)
    # the check has teeth: a 25% error in the diffusion coefficient fails the steadiness measure
    assert _penetration(tau_scale=1.25)[1] > 0.15
