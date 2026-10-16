"""Diffusive neutrals for the HSX FCI backend (``native/fci_neutrals.py``).

Self-contained: the operator is built on analytic geometries (a sheared
slab with a constant non-orthogonal metric and a cylindrical annulus with a
varying metric) passed as duck-typed ``FciGeometry3D`` stand-ins.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.fci_braginskii.native import fci_neutrals as F
from drbx.fci_braginskii.run import HsxBlobConfig, neutral_parameters

jax.config.update("jax_enable_x64", True)


def _grid1d(n, length):
    faces = np.linspace(0.0, length, n + 1)
    return SimpleNamespace(centers=0.5 * (faces[1:] + faces[:-1]), widths=np.diff(faces), faces=faces)


def _metric(cov):
    """MetricGeometry stand-in from covariant components ``cov[..., i, j]``."""

    con = np.linalg.inv(cov)
    names = {}
    for i in range(3):
        for j in range(i, 3):
            names[f"g{i + 1}{j + 1}"] = con[..., i, j]
            names[f"g_{i + 1}{j + 1}"] = cov[..., i, j]
    return SimpleNamespace(J=np.sqrt(np.linalg.det(cov)), **names)


def _geometry(cov_fn, shape, lengths):
    """Cell and face metrics from ``cov_fn(u, v, w)`` on a logical box."""

    gx, gy, gz = (_grid1d(n, length) for n, length in zip(shape, lengths))

    def at(u, v, w):
        U, V, W = np.meshgrid(u, v, w, indexing="ij")
        return _metric(cov_fn(U, V, W))

    return SimpleNamespace(
        shape=shape,
        grid=SimpleNamespace(x=gx, y=gy, z=gz),
        cell_metric=at(gx.centers, gy.centers, gz.centers),
        face_metric=SimpleNamespace(
            x=at(gx.faces, gy.centers, gz.centers),
            y=at(gx.centers, gy.faces, gz.centers),
            z=at(gx.centers, gy.centers, gz.faces),
        ),
    )


def sheared_slab(shape=(16, 8, 6), shear=0.3):
    """X = u + shear v, Y = v, Z = w: constant non-orthogonal metric."""

    def cov(U, V, W):
        e = np.zeros(U.shape + (3, 3))
        e[..., 0, 0] = 1.0
        e[..., 0, 1] = e[..., 1, 0] = shear
        e[..., 1, 1] = 1.0 + shear**2
        e[..., 2, 2] = 1.0
        return e

    return _geometry(cov, shape, (1.0, 1.0, 1.0))


def annulus(shape=(12, 16, 6), r0=0.3, width=0.5):
    """Cylindrical annulus r = r0 + width u, theta, z: varying J and g^22."""

    def cov(U, V, W):
        r = r0 + width * U
        e = np.zeros(U.shape + (3, 3))
        e[..., 0, 0] = width**2
        e[..., 1, 1] = r**2
        e[..., 2, 2] = 1.0
        # a small u-theta cross term so the cross fluxes are exercised
        e[..., 0, 1] = e[..., 1, 0] = 0.05 * width * r
        return e

    return _geometry(cov, shape, (1.0, 2.0 * np.pi, 1.0))


PARAMS = F.NeutralParameters(reference_density_m3=5.0e18, reference_temperature_ev=F.reference_temperature_ev(5.0e-4))


def _random_plasma(shape, seed):
    rng = np.random.default_rng(seed)
    n = jnp.asarray(0.5 + rng.random(shape))
    Vi = jnp.asarray(0.5 * rng.normal(size=shape))
    Te = jnp.asarray(0.2 + rng.random(shape))
    Ti = jnp.asarray(0.2 + rng.random(shape))
    Nn = jnp.asarray(0.01 + 0.2 * rng.random(shape))
    return n, Vi, Te, Ti, Nn


def test_reference_scales_match_the_backend_normalization():
    # docs: rho* = 5e-4 is hydrogen at ~24 eV, B = 1 T; one time unit ~21 microseconds
    assert F.reference_temperature_ev(5.0e-4) == pytest.approx(23.95, rel=1e-3)
    assert F.time_unit_s(F.reference_temperature_ev(5.0e-4)) == pytest.approx(2.09e-5, rel=1e-2)


@pytest.mark.parametrize("geometry", [sheared_slab(), annulus()], ids=["sheared", "annulus"])
def test_constant_density_is_a_null_of_the_diffusion(geometry):
    grid = F.build_neutral_grid(geometry)
    rng = np.random.default_rng(0)
    D = jnp.asarray(0.01 + rng.random(geometry.shape))  # strongly varying D
    rate = F.neutral_diffusion_rate(jnp.full(geometry.shape, 0.37), D, grid)
    assert float(jnp.max(jnp.abs(rate))) == 0.0
    uniform, _relres = F.implicit_diffusion(jnp.full(geometry.shape, 0.37), D, 0.3, grid, PARAMS)
    np.testing.assert_allclose(np.asarray(uniform), 0.37, rtol=1e-14)


@pytest.mark.parametrize("geometry", [sheared_slab(), annulus()], ids=["sheared", "annulus"])
def test_diffusion_is_conservative_with_a_reflecting_wall(geometry):
    grid = F.build_neutral_grid(geometry)
    _, _, _, _, Nn = _random_plasma(geometry.shape, 1)
    D = jnp.asarray(0.05 + np.random.default_rng(2).random(geometry.shape))
    rate = F.neutral_diffusion_rate(Nn, D, grid)
    scale = float(jnp.sum(grid.volume * jnp.abs(rate)))
    assert abs(float(jnp.sum(grid.volume * rate))) < 1e-14 * scale
    new, relres = F.implicit_diffusion(Nn, D, 0.05, grid, PARAMS)
    total = float(jnp.sum(grid.volume * Nn))
    assert abs(float(jnp.sum(grid.volume * new)) - total) < 1e-14 * total
    assert float(relres) < 1e-9
    # a wall influx enters exactly
    influx = jnp.full(geometry.shape[1:], 0.2)
    new, _ = F.implicit_diffusion(Nn, D, 0.05, grid, PARAMS, wall_influx=influx)
    np.testing.assert_allclose(float(jnp.sum(grid.volume * new)) - total, 0.05 * float(jnp.sum(influx)), rtol=1e-12)


def test_diffusion_operator_is_symmetric_negative_on_orthogonal_metric():
    geometry = annulus(shape=(6, 8, 4))
    # remove the cross term: pure diagonal metric
    for m in (geometry.cell_metric, geometry.face_metric.x, geometry.face_metric.y, geometry.face_metric.z):
        m.g12 = np.zeros_like(m.g12)
    grid = F.build_neutral_grid(geometry)
    D = jnp.asarray(0.1 + np.random.default_rng(3).random(geometry.shape))
    size = int(np.prod(geometry.shape))
    apply = lambda v: (grid.volume * F.neutral_diffusion_rate(v.reshape(geometry.shape), D, grid)).ravel()
    A = np.asarray(jax.vmap(apply)(jnp.eye(size))).T
    np.testing.assert_allclose(A, A.T, atol=1e-13 * np.max(np.abs(A)))
    assert np.max(np.linalg.eigvalsh(0.5 * (A + A.T))) < 1e-12


def test_reaction_hook_conserves_particles_and_energy_exactly():
    shape = (5, 4, 3)
    owner = jnp.ones(shape, dtype=bool)
    for seed in range(4):
        n, Vi, Te, Ti, Nn = _random_plasma(shape, seed)
        plasma, out = F.reaction_source(n, Vi, Te, Ti, Nn, PARAMS, owner)
        # energy identity of the transfer model with stationary neutrals
        identity = plasma.ion_energy + Vi * plasma.momentum - 0.5 * Vi**2 * plasma.particle + out.neutral_energy
        scale = float(jnp.max(out.charge_exchange_rate + out.ionization_rate)) * (1.0 + float(jnp.max(Vi**2)))
        assert float(jnp.max(jnp.abs(identity))) < 1e-13 * scale
        np.testing.assert_array_equal(np.asarray(plasma.particle + out.neutral_density), 0.0)
        dt = 0.3
        n2, Vi2, Te2, Ti2 = F.apply_volumetric_source(n, Vi, Te, Ti, plasma, dt, tau=PARAMS.tau, owner=owner)
        np.testing.assert_allclose(np.asarray(n2 - n), dt * np.asarray(plasma.particle), atol=1e-15)
        np.testing.assert_allclose(np.asarray(n2 * Vi2 - n * Vi), dt * np.asarray(plasma.momentum), atol=1e-14)
        e_before = 1.5 * n * Te + 1.5 * n * Ti + 0.5 * n * Vi**2
        e_after = 1.5 * n2 * Te2 + 1.5 * n2 * Ti2 + 0.5 * n2 * Vi2**2
        expected = dt * (plasma.electron_energy + plasma.ion_energy + Vi * plasma.momentum - 0.5 * Vi**2 * plasma.particle)
        np.testing.assert_allclose(np.asarray(e_after - e_before), np.asarray(expected), atol=1e-14)


@pytest.mark.parametrize("geometry", [sheared_slab(), annulus()], ids=["sheared", "annulus"])
def test_closed_box_conserves_nuclei_and_energy(geometry):
    grid = F.build_neutral_grid(geometry)
    params = F.NeutralParameters(
        reference_density_m3=5.0e18, reference_temperature_ev=PARAMS.reference_temperature_ev, reaction_substeps=3
    )
    state = _random_plasma(geometry.shape, 7)
    sinks = 0.0
    for _ in range(5):
        before = F.inventories(*state, grid, params)
        n, Vi, Te, Ti, Nn, ledger, _ = F.neutral_step(*state, 0.02, grid, params)
        state = (n, Vi, Te, Ti, Nn)
        after = F.inventories(*state, grid, params)
        assert abs(float(after[0] - before[0])) < 1e-14 * float(before[0])
        sink = float(ledger[3] + ledger[4])
        assert sink > 0.0
        energy_error = float(after[1] + after[2] + sink - before[1] - before[2])
        assert abs(energy_error) < 1e-12 * sink
        assert abs(float(ledger[6])) < 1e-14 * float(before[0])
        sinks += sink
    assert float(jnp.min(state[4])) > 0.0


def test_pure_ionisation_follows_the_logistic_solution():
    """Handoff 6.3: fixed Te, quasineutral, ionisation only -> logistic n_i(t)."""

    shape = (1, 1, 1)
    owner = jnp.ones(shape, dtype=bool)
    Te = jnp.full(shape, 1.0)
    n0, total = 0.2, 1.0
    from drbx.native.neutrals.atomic_rates import rate_coefficient

    params = PARAMS
    t_unit = F.time_unit_s(params.reference_temperature_ev)
    # K in backend units; evaluate where the density-dependent fit is flat enough:
    # we hold K fixed by evaluating it once and using a constant-K reference.
    def k_of(n):
        return float(rate_coefficient("d", "iz", params.reference_temperature_ev, n * params.reference_density_m3)) * params.reference_density_m3 * t_unit

    n = jnp.full(shape, n0)
    Nn = jnp.full(shape, total - n0)
    dt, steps = 0.01, 400
    t = 0.0
    errs = []
    for _ in range(steps):
        _plasma, out = F.reaction_source(n, jnp.zeros(shape), Te, Te, Nn, params, owner)
        rate = out.ionization_rate  # only ionisation is kept
        # midpoint (Heun) step of the pure-ionisation ODE with the model's rate
        k1 = rate
        _, out2 = F.reaction_source(n + dt * k1, jnp.zeros(shape), Te, Te, Nn - dt * k1, params, owner)
        n, Nn = n + 0.5 * dt * (k1 + out2.ionization_rate), Nn - 0.5 * dt * (k1 + out2.ionization_rate)
        t += dt
        k_of(float(n[0, 0, 0]))
        errs.append(abs(float(n[0, 0, 0] + Nn[0, 0, 0]) - total))
    # The AMJUEL fit depends weakly on n_e; compare with the logistic using the
    # rate coefficient at the final density (the spread of K over the run bounds the error).
    k_lo, k_hi = sorted((k_of(n0), k_of(float(n[0, 0, 0]))))
    logistic = lambda K: total / (1.0 + (total / n0 - 1.0) * np.exp(-K * total * t))
    assert logistic(k_lo) - 1e-6 <= float(n[0, 0, 0]) <= logistic(k_hi) + 1e-6
    assert max(errs) < 1e-14


def _slab_steady_state(nx):
    """Steady N with diffusion D, loss nu N and wall influx G: analytic cosh profile."""

    geometry = sheared_slab(shape=(nx, 4, 3), shear=0.4)
    grid = F.build_neutral_grid(geometry)
    D, nu, G = 0.02, 0.5, 1.0
    Dfield = jnp.full(geometry.shape, D)
    area = 1.0 * 1.0  # dy dz over the whole periodic face
    influx = jnp.full(geometry.shape[1:], G * area / (geometry.shape[1] * geometry.shape[2]))
    source = F.neutral_diffusion_rate(jnp.zeros(geometry.shape), Dfield, grid, influx)

    def operator(v):
        return nu * v - F.neutral_diffusion_rate(v, Dfield, grid)

    solution, _ = jax.scipy.sparse.linalg.gmres(operator, source, tol=1e-13, atol=0.0, restart=80, maxiter=50)
    # The sheared map X = u + s v has the physical wall-normal distance u / sqrt(1 + s^2)... the
    # x faces are planes u = const, whose unit normal grad u has |grad u| = sqrt(g^11).
    g11 = float(geometry.cell_metric.g11[0, 0, 0])
    length = 1.0 / np.sqrt(g11)  # physical slab thickness normal to the walls
    k = np.sqrt(nu / D)
    s = np.asarray(geometry.grid.x.centers) * length
    # influx per unit wall area: G * area / (physical wall area = J sqrt(g11) * 1 * 1)
    J = float(geometry.cell_metric.J[0, 0, 0])
    flux_density = G * area / (J * np.sqrt(g11))
    exact = flux_density / (D * k) * np.cosh(k * s) / np.sinh(k * length)
    numeric = np.asarray(solution)[:, 0, 0]
    return np.max(np.abs(numeric - exact)) / np.max(exact)


def test_one_dimensional_penetration_matches_the_analytic_profile():
    """Neutral penetration ``D N'' = nu N`` with a wall influx: second-order convergence."""

    e1, e2 = _slab_steady_state(32), _slab_steady_state(64)
    assert e2 < 2e-3
    assert e1 / e2 > 3.5


def test_config_defaults_keep_neutrals_off_and_validate():
    assert HsxBlobConfig().neutrals is False
    assert neutral_parameters(HsxBlobConfig()) is None
    on = neutral_parameters(HsxBlobConfig(neutrals=True))
    assert on.reference_temperature_ev == pytest.approx(F.reference_temperature_ev(5.0e-4))
    with pytest.raises(ValueError, match="restart"):
        neutral_parameters(HsxBlobConfig(neutrals=True, restart_from="x.npz"))
    with pytest.raises(ValueError, match="neutral_mfp_max"):
        neutral_parameters(HsxBlobConfig(neutrals=True, neutral_mfp_max=0.0))


def test_rate_validity_counts_flag_out_of_range_inputs():
    owner = jnp.ones((4,), dtype=bool)
    Te0 = PARAMS.reference_temperature_ev
    Te = jnp.asarray([1.0, 0.001 / Te0, 2.0e4 / Te0, 1.0])
    n = jnp.asarray([1.0, 1.0, 1.0, 1.0e-6])
    counts = F.rate_validity_counts(n, Te, Te, PARAMS, owner)
    # the cold-ion cell stays in range for CX: T_cx = Ti + T_n (3 eV)
    assert [float(c) for c in counts] == [2.0, 1.0, 1.0]


def test_knudsen_map_scales_with_the_mean_free_path():
    geometry = annulus()
    grid = F.build_neutral_grid(geometry)
    n, _, Te, Ti, Nn = _random_plasma(geometry.shape, 4)
    mfp, _kn_cell, kn_grad = F.knudsen_map(n, Te, Ti, Nn, grid, PARAMS)
    mfp2, _kn_cell2, _ = F.knudsen_map(2.0 * n, Te, Ti, Nn, grid, PARAMS)
    assert float(jnp.min(mfp)) > 0.0 and float(jnp.min(kn_grad)) > 0.0
    # doubling the plasma density roughly halves the mean free path (rates depend weakly on n_e)
    ratio = np.asarray(mfp2 / mfp)
    assert 0.4 < ratio.min() and ratio.max() < 0.6


@pytest.mark.slow
def test_hsx_sheath_and_neutrals_ledgers_close(tmp_path):
    """Real HSX (set DRBX_TEST_GEOMETRY_BUNDLE): sheath + neutrals, both ledgers close."""

    import os

    bundle = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
    if bundle is None:
        pytest.skip("set DRBX_TEST_GEOMETRY_BUNDLE to run the HSX neutral ledger")
    from drbx.cli import main

    deck = tmp_path / "hsx.toml"
    deck.write_text(
        '[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\n'
        f'geometry = "{bundle}"\noutput = "{tmp_path / "out" / "hsx"}"\n'
        "final_time = 0.0015\nnum_steps = 2\nsave_every = 2\nneutrals = true\n"
    )
    assert main(["run", str(deck)]) in (0, None)
    data = np.load(tmp_path / "out" / "hsx.npz")
    neutral = {name: data["neutral_ledger"][:, k] for k, name in enumerate(data["neutral_ledger_columns"])}
    # the split conserves nuclei and closes energy with the reported sinks
    np.testing.assert_allclose(neutral["nuclei_after"], neutral["nuclei_before"], rtol=1e-14)
    sinks = neutral["electron_energy_loss"] + neutral["neutral_reservoir_energy"]
    assert np.all(neutral["ionisation"] > 0.0) and np.all(sinks > 0.0)
    energy = (neutral["plasma_energy_after"] + neutral["neutral_energy_after"] + sinks
              - neutral["plasma_energy_before"] - neutral["neutral_energy_before"])
    assert np.all(np.abs(energy) < 1e-9 * sinks)
    assert np.all(neutral["out_of_range_Te"] == 0.0)
    # wall ledger: inventory change = applied wall/other + the split's change
    rows, columns = data["wall_ledger"], list(data["wall_ledger_columns"])
    col = {name: rows[:, k] for k, name in enumerate(columns)}
    wall, other = col["step_particle_wall"][1:], col["step_particle_other"][1:]
    np.testing.assert_allclose(
        np.diff(col["particle_inventory"]), other - wall + neutral["split_wall_particle_inventory"],
        rtol=0, atol=1e-9 * wall.max(),
    )
    assert data["neutral_density"].shape[0] == 2
    assert np.all(np.isfinite(data["neutral_knudsen_gradient"]))
