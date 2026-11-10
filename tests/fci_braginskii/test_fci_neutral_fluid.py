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
    f = FL.flux_limiter(gamma_c, 1.0, 1.0, 2.0)
    np.testing.assert_allclose(np.asarray(f), (1.0 + np.asarray(gamma_c) ** 2) ** -0.5, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(FL.flux_limiter(gamma_c, 1.0)), (1.0 + (np.asarray(gamma_c) / 3.0) ** 4) ** -0.25, rtol=1e-14)
    assert float(f[0]) == pytest.approx(1.0, abs=1e-11) and float(f[-1] * gamma_c[-1]) == pytest.approx(1.0, rel=1e-6)
    assert np.all(np.diff(np.asarray(f * gamma_c)) > 0.0)  # limited flux still monotone in Gamma_c
    assert np.isfinite(float(jax.grad(lambda g: FL.flux_limiter(g, 1.0))(0.0)))


def test_transport_coefficients_follow_the_afn_closure():
    n, _, Te, Ti, Nn = _random_plasma((4, 3, 3), 3)
    Tn = 0.05 + 0.1 * Te
    nu_cx, nu_iz = F.collision_frequencies(n, Te, Ti, PARAMS, Tn)
    dp, kappa, eta = FL.transport_coefficients(n, Te, Ti, Nn, Tn, PARAMS)
    nu = np.asarray(nu_cx + nu_iz)
    np.testing.assert_allclose(np.asarray(dp), (1.0 + 0.8 * np.asarray(nu_iz) / nu) / nu, rtol=1e-13)  # tau_n / m (N6 factor)
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
    assert np.all(np.asarray(gamma) <= 3.0 * np.asarray(ramp * jnp.sqrt(Tn / (2 * jnp.pi))) * (1 + 1e-12))


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


@pytest.mark.parametrize("plasma", ["uniform neutrals", "cold plasma"])
def test_n6_closure_is_finite_and_differentiable_at_zero_gradient_and_zero_rates(plasma):
    """The wall albedo (Kn_w from |grad n_n|) and tau_n = (1 + 0.8 nu_iz/nu)/nu stay finite where the gradient or nu vanishes."""

    params = dataclasses.replace(PARAMS, wall_absorption=0.7)
    geometry = sheared_slab(shape=(6, 4, 3))
    grid = F.build_neutral_grid(geometry)
    n, Vi, Te, Ti, Nn = _random_plasma(geometry.shape, 11)
    Nn = jnp.full(geometry.shape, 0.3) if plasma == "uniform neutrals" else Nn
    Te = 1.0e-8 * Te if plasma == "cold plasma" else Te
    Tn = jnp.full(geometry.shape, 0.1)

    def functional(scale):
        out = FL.neutral_fluid_step(n, Vi, Te, Ti, scale * Nn, 0.0 * Vi, Tn, 0.02, grid, params)
        return jnp.sum(grid.volume * (out[4] + out[6]))

    assert np.isfinite(float(jax.grad(functional)(1.0))) and np.isfinite(float(functional(1.0)))


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


def _kinetic_slab(nu_cx, nu_iz, T, length, thick, cells=80, nv=60, vmax=7.0, profile=None):
    """Deterministic oracle: steady 1D atoms of temperature T, charge exchange re-emits a Maxwellian, ionisation sinks, unit
    flux emitted by the wall.  Exact cell-wise streaming along ``nv`` speeds, scattering source solved as a dense linear
    system.  ``thick``: reflecting wall (mirror = plane source on the unfolded line) and plasma to depth ``length``; else a
    layer of that thickness between the absorbing wall and a mirror.  Returns ``(depth, n)`` per cell, wall first."""

    N, h = 2 * cells, length / cells
    j = np.arange(N)
    f = np.ones(N) if profile is None else profile((np.abs(j + 0.5 - cells) if thick else np.minimum(j + 0.5, N - j - 0.5)) * h)
    nu_cx, nu = nu_cx * f, (nu_cx + nu_iz) * f
    xg, wg = np.polynomial.legendre.leggauss(nv)
    v = 0.5 * vmax * np.sqrt(T) * (xg + 1)
    W = 0.5 * vmax * np.sqrt(T) * wg * np.exp(-(v**2) / (2 * T)) / np.sqrt(2 * np.pi * T)  # weights of g = f / Maxwellian
    a = nu[:, None] * h / v
    e = np.exp(-a)
    inject = {1: cells if thick else 0, -1: cells - 1 if thick else N - 1}  # cell where the wall flux enters
    q = np.zeros((N, N + 1))
    q[j, j] = nu_cx  # unit scattering source in one cell per column
    n = np.zeros((N, N + 1))
    for sign in (1, -1):
        g = np.zeros((N + 1, nv))
        for c in range(N) if sign > 0 else range(N - 1, -1, -1):
            g[N] += np.sqrt(2 * np.pi / T) * (c == inject[sign])  # last column: the wall emission (unit flux)
            eq = q[c][:, None] / nu[c]
            n[c] += (W * (eq + (g - eq) * ((1 - e[c]) / a[c]))).sum(1)
            g = g * e[c] + eq * (1 - e[c])
    nn = np.linalg.solve(np.eye(N) - n[:, :N], n[:, N])[cells:]
    return (np.arange(cells) + 0.5) * h, nn if thick else nn[::-1]


def _rates(Te, params=PARAMS):
    one = jnp.ones((1, 1, 1))
    return tuple(float(x[0, 0, 0]) for x in F.collision_frequencies(one, Te * one, 0.15 * one, params, 0.15 * one))


def _fluid_slab(Te, length, thick, nx, steps, profile=None):
    """The fluid model on the same problem (unit wall flux as a volume source in the wall cell): ``(depth, n, nu_cx, nu_iz, T_n)``."""

    params = dataclasses.replace(PARAMS, wall_absorption=0.0 if thick else 1.0)
    shape = (nx, 4, 4)

    def cov(U, V, W):
        e = np.zeros(U.shape + (3, 3))
        e[..., 0, 0] = e[..., 1, 1] = e[..., 2, 2] = 1.0
        return e

    depth = (np.arange(nx)[::-1] + 0.5) * length / nx  # wall is the last x cell
    n = jnp.ones(shape) * jnp.asarray(np.ones(nx) if profile is None else profile(depth))[:, None, None]
    Ti = jnp.full(shape, 0.15)
    nu_cx, nu_iz = _rates(Te, params)
    grid, h = F.build_neutral_grid(_geometry(cov, shape, (length, 1.0, 1.0))), length / nx
    source = jnp.zeros(shape).at[-1].set(1.0 / h)
    dt = 0.1 / (nu_cx + nu_iz)
    lam_pen = np.sqrt(0.15 / (nu_iz * (nu_cx + nu_iz)))
    start = np.exp(-(np.arange(nx)[::-1] + 0.5) * h / lam_pen) / (nu_iz * lam_pen) if thick else np.full(nx, 0.1)
    # injected atoms carry the wall's flux-weighted energy 2 T_n, which the absorbing wall also takes away
    energy = (1.5 if thick else 2.0) * 0.15 * source
    step = jax.jit(lambda nn, un, tn: FL.neutral_fluid_step(
        n, jnp.zeros(shape), Te * jnp.ones(shape), Ti, nn, un, tn, dt, grid, params, volume_source=source, volume_energy=energy))
    carry = (jnp.asarray(start)[:, None, None] * jnp.ones(shape), jnp.zeros(shape), Ti)
    carry, _ = jax.lax.scan(lambda c, _: (step(*c)[4:7], None), carry, None, length=steps)
    return (np.arange(nx) + 0.5) * h, np.asarray(carry[0][::-1, 0, 0]), nu_cx, nu_iz, np.asarray(carry[2][::-1, 0, 0])


def _moments(depth, n, nu_iz):
    """(ionised fraction of the wall flux, mean ionisation depth)."""

    return nu_iz * n.sum() * (depth[1] - depth[0]), (depth * n).sum() / n.sum()


@pytest.mark.parametrize("Te, tolerance", [(0.12, 0.03), (0.5, 0.05), (2.0, 0.05)])
def test_slab_penetration_matches_the_kinetic_oracle_up_to_kn_grad_0p8(Te, tolerance):
    """Kn_grad = lambda_mfp / lambda_pen = sqrt(nu_iz / nu) = 0.09, 0.60, 0.80: mean ionisation depth within the tolerance.

    Without the N6 limiter and kinetic factor the depth is 0.99, 0.65 and 0.53 of the oracle's."""

    nu_cx, nu_iz = _rates(Te)
    length = 6.0 * np.sqrt(0.15 / (nu_iz * (nu_cx + nu_iz)))
    s_f, n_f = _fluid_slab(Te, length, True, 40, 200)[:2]
    s_k, n_k = _kinetic_slab(nu_cx, nu_iz, 0.15, length, True)
    assert _moments(s_f, n_f, nu_iz)[1] / _moments(s_k, n_k, nu_iz)[1] == pytest.approx(1.0, abs=tolerance)


@pytest.mark.parametrize("Kn, nx", [(0.3, 48), (1.0, 32), (3.0, 24), (10.0, 16)])
def test_thin_layer_ionisation_matches_the_kinetic_oracle(Kn, nx):
    """A layer of thickness lambda_mfp / Kn between the absorbing wall and a mirror: ionised fraction of the wall flux.

    Without the Knudsen-layer wall albedo the ratio is 0.66 and 0.85 at Kn = 0.3 and 1."""

    nu_cx, nu_iz = _rates(0.5)
    length = np.sqrt(0.15) / (nu_cx + nu_iz) / Kn
    s_f, n_f = _fluid_slab(0.5, length, False, nx, 150)[:2]
    s_k, n_k = _kinetic_slab(nu_cx, nu_iz, 0.15, length, False)
    assert _moments(s_f, n_f, nu_iz)[0] / _moments(s_k, n_k, nu_iz)[0] == pytest.approx(1.0, abs=0.05)


def test_ballistic_vacuum_gap_ionisation_matches_the_kinetic_oracle():
    """Absorbing wall, a gap of 3 mean free paths at 1e-3 of the plasma density, then the plasma: ionised fraction within 5%.

    (The neutral density in the gap is 20% low: the limiter flux 3 Gamma_lim exceeds the pure-beam flux 2 Gamma_lim.)"""

    nu_cx, nu_iz = _rates(0.5)
    gap = 3.0 * np.sqrt(0.15) / (nu_cx + nu_iz)
    profile = lambda depth: np.where(depth < gap, 1.0e-3, 1.0)
    s_f, n_f = _fluid_slab(0.5, 3.0 * gap, False, 48, 150, profile)[:2]
    s_k, n_k = _kinetic_slab(nu_cx, nu_iz, 0.15, 3.0 * gap, False, profile=profile)
    ionised = lambda s, n: nu_iz * (profile(s) * n).sum() * (s[1] - s[0])
    assert ionised(s_f, n_f) / ionised(s_k, n_k) == pytest.approx(1.0, abs=0.05)


def test_fine_cells_with_a_strong_wall_source_keep_the_temperature_positive_and_finite():
    """96 cells over 0.1 mean free paths and a wall source that triples the density per step used to lose the Krylov
    convergence (relative residual 1e-2 to 1) and then T_n < 0, NaN."""

    nu_cx, nu_iz = _rates(0.5)
    length = 0.1 * np.sqrt(0.15) / (nu_cx + nu_iz)
    _, n, _, _, tn = _fluid_slab(0.5, length, False, 96, 40)
    assert np.all(np.isfinite(n)) and n.min() > 0.0 and tn.min() > 0.0


def test_fine_cell_solves_converge_to_1e6_with_a_short_restart(monkeypatch):
    """The banded x-line preconditioner keeps every implicit solve below 1e-6 with restart 30 (point Jacobi needed 200)."""

    residuals, solve = [], FL._gmres

    def recording(*args, **kwargs):
        x, relres = solve(*args, **kwargs)
        jax.debug.callback(lambda r: residuals.append(float(r)), relres)
        return x, relres

    monkeypatch.setattr(FL, "_gmres", recording)
    nu_cx, nu_iz = _rates(0.5)
    _fluid_slab(0.5, 0.1 * np.sqrt(0.15) / (nu_cx + nu_iz), False, 96, 40)
    assert PARAMS.gmres_restart <= 60 and len(residuals) == 120 and max(residuals) < 1.0e-6


@pytest.mark.parametrize("nx", [1, 2, 3, 4, 5, 9])
def test_line_solver_inverts_an_x_pentadiagonal_operator_on_short_lines(nx):
    """The measured bands are exact for an x-banded operator, also when the line is shorter than the 5-colour pattern."""

    shape = (nx, 4, 2)
    rng = np.random.default_rng(nx)
    coef = {d: jnp.asarray(0.2 * rng.standard_normal(shape)) for d in range(-2, 3)}
    coef[0] = coef[0] + 2.0

    def operator(u):
        out = 0.0
        for d, c in coef.items():
            shifted = jnp.roll(u, -d, axis=0)
            valid = (jnp.arange(nx) + d >= 0) & (jnp.arange(nx) + d < nx)
            out = out + c * jnp.where(valid[:, None, None], shifted, 0.0)
        return out

    owner = jnp.ones(shape, bool)
    solve = jax.jit(FL._line_solver(operator, owner, jnp.full(shape, 2.0)))
    v = jnp.asarray(rng.standard_normal(shape))
    np.testing.assert_allclose(np.asarray(operator(solve(v))), np.asarray(v), atol=1e-10)
    assert np.all(np.isfinite(np.asarray(solve(jnp.zeros(shape)))))


@pytest.mark.parametrize("nx", [2, 3, 4])
def test_fluid_step_on_very_short_x_lines_stays_finite(nx):
    nu_cx, nu_iz = _rates(0.5)
    _, n, _, _, tn = _fluid_slab(0.5, np.sqrt(0.15) / (nu_cx + nu_iz), False, nx, 10)
    assert np.all(np.isfinite(n)) and n.min() > 0.0 and tn.min() > 0.0
