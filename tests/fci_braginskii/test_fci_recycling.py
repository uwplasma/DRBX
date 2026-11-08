"""Wall recycling of the HSX FCI backend (exhaust PR 3).

The recycled neutral source is ``R`` times the applied ion wall loss of the
IMEX step, deposited in the wall owner cells.  The unit tests use the
analytic geometries of ``test_fci_neutrals.py``; the flux-tube test drives
the production wall-leg rows of ``test_fci_wall_closure.py`` with a
recycled ionisation source; the slow HSX test needs the geometry bundle.
"""

from __future__ import annotations

import math
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.fci_braginskii.native import fci_neutral_fluid as FL
from drbx.fci_braginskii.native import fci_neutrals as F
from drbx.fci_braginskii.native.fci_wall_closure import SheathClosure, wall_face_fluxes
from drbx.fci_braginskii.run import HsxBlobConfig, recycling_parameters

from .test_fci_neutrals import _random_plasma, annulus, sheared_slab
from .test_fci_wall_closure import flux_tube_bulk_rates

jax.config.update("jax_enable_x64", True)

TE0 = F.reference_temperature_ev(5.0e-4)
PARAMS = F.NeutralParameters(reference_density_m3=5.0e18, reference_temperature_ev=TE0, reaction_substeps=2)


def _wall_loss(geometry, seed):
    """A positive applied wall loss rate on the outer radial layer (wall cells)."""

    rng = np.random.default_rng(seed)
    loss = np.zeros(geometry.shape)
    loss[-2:] = 0.1 + rng.random(loss[-2:].shape)
    return jnp.asarray(loss)


@pytest.mark.parametrize("geometry", [sheared_slab(), annulus()], ids=["sheared", "annulus"])
@pytest.mark.parametrize("coefficient", [1.0, 0.9])
def test_recycling_closes_nuclei_with_the_wall_loss(geometry, coefficient):
    """Wall step then split step: d(n + N) = -(1 - R) x wall loss; R = 1 is a closed box."""

    grid = F.build_neutral_grid(geometry)
    n, Vi, Te, Ti, Nn = _random_plasma(geometry.shape, 3)
    un, Tn = jnp.zeros(geometry.shape), 0.05 + 0.3 * Te
    dt = 0.02
    total0 = float(F.inventories(n, Vi, Te, Ti, Nn, grid, PARAMS)[0])
    total = total0
    for step in range(20):
        loss = _wall_loss(geometry, step)
        wall = dt * float(jnp.sum(grid.volume * jnp.where(grid.owner, loss, 0.0)))
        n = jnp.where(grid.owner, n - dt * loss, n)  # the IMEX wall leg
        source = F.recycled_neutral_source(loss, coefficient, grid.owner)
        before = float(F.inventories(n, Vi, Te, Ti, Nn, grid, PARAMS)[0])
        n, Vi, Te, Ti, Nn, un, Tn, ledger, _ = FL.neutral_fluid_step(
            n, Vi, Te, Ti, Nn, un, Tn, dt, grid, PARAMS, volume_source=source, volume_energy=0.125 * source)
        after = float(F.inventories(n, Vi, Te, Ti, Nn, grid, PARAMS)[0])
        # the split adds exactly the recycled particles, and reports them
        assert float(ledger[7]) == pytest.approx(coefficient * wall, rel=1e-13)
        assert abs(after - before - coefficient * wall) < 1e-14 * total0
        assert float(ledger[13]) == pytest.approx(0.125 * coefficient * wall, rel=1e-13)  # their energy
        assert abs((after - total) + (1.0 - coefficient) * wall) < 1e-13 * total0
        total = after
    if coefficient == 1.0:
        assert abs(total - total0) < 1e-13 * total0
    assert float(jnp.min(jnp.where(grid.owner, Nn, 1.0))) > 0.0


def test_recycling_off_is_the_unchanged_split():
    geometry = sheared_slab()
    grid = F.build_neutral_grid(geometry)
    state = (*_random_plasma(geometry.shape, 5), jnp.zeros(geometry.shape))
    state = (*state[:5], state[5], 0.05 + 0.3 * state[2])
    plain = FL.neutral_fluid_step(*state, 0.02, grid, PARAMS)
    zero = FL.neutral_fluid_step(*state, 0.02, grid, PARAMS, volume_source=jnp.zeros(geometry.shape))
    for a, b in zip(plain, zero):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-15)


def test_recycling_config_defaults_and_energy():
    assert HsxBlobConfig().recycling is False
    assert recycling_parameters(HsxBlobConfig()) is None
    on = recycling_parameters(HsxBlobConfig(neutrals=True, recycling=True))
    assert on.coefficient == 0.99 and on.energy == "franck-condon"
    assert on.energy_per_atom(TE0) == pytest.approx(3.0 / TE0)
    wall = recycling_parameters(HsxBlobConfig(neutrals=True, recycling=True, recycling_energy="wall",
                                              recycling_wall_temperature_ev=0.05))
    assert wall.energy_per_atom(1.0) == pytest.approx(0.075)
    with pytest.raises(ValueError, match="requires neutrals"):
        recycling_parameters(HsxBlobConfig(recycling=True))
    with pytest.raises(ValueError, match="Bohm"):
        recycling_parameters(HsxBlobConfig(neutrals=True, recycling=True, wall_closure="zero-flow"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        recycling_parameters(HsxBlobConfig(neutrals=True, recycling=True, recycling_coefficient=1.5))


# ----------------------------------------------------------------- flux tube
TAU = 1.0


def _recycling_flux_tube(coefficient, *, cells=20, mu=25.0, periods=None, e_iz=0.3, e_rec=0.1, decay=0.2):
    """Production wall-leg rows plus a recycled ionisation source near each target.

    Core source 0.5 and heating 2.0 per unit length on [0, 2]; each target
    returns ``R`` times its own Bohm particle flux as ions within ``decay``
    of the target (instant ionisation), costing ``e_iz`` of electron energy
    per ionisation and bringing ``e_rec`` per recycled atom to the ions.
    """

    length = 2.0
    dx = length / cells
    s = (np.arange(cells) + 0.5) * dx
    core = 0.5 * np.ones(cells)
    heating = 2.0 * np.ones(cells)
    shape_b = np.exp(-s / decay)
    shape_f = np.exp(-(length - s) / decay)
    shape_b, shape_f = shape_b / (shape_b.sum() * dx), shape_f / (shape_f.sum() * dx)
    sheath = SheathClosure()
    ge, gi = float(sheath.gamma_e(TAU, mu)), sheath.ion_heat_transmission
    bulk = flux_tube_bulk_rates(cells, dx, mu)

    def rates(u):
        r, bst, fst = bulk(u, ge, gi)
        # the recycled flux is the wall closure's own particle flux, per target
        flux_b = wall_face_fluxes(bst[0], -1.0, TAU, ge, gi, u[0, 3]).particle
        flux_f = wall_face_fluxes(fst[-1], 1.0, TAU, ge, gi, u[-1, 3]).particle
        iz = coefficient * (flux_b * shape_b + flux_f * shape_f)
        source = core + iz
        n, Te, Ti, V = u[:, 0], u[:, 1], u[:, 2], u[:, 3]
        pe = heating / 2 - e_iz * iz
        pi = heating / 2 + e_rec * iz
        r = r + jnp.stack([source, (pe - 1.5 * Te * source) / (1.5 * n),
                           (pi - 1.5 * Ti * source) / (1.5 * n), -source * V / n, -source * V / n], -1)
        return r, bst, fst, flux_b + flux_f

    @partial(jax.jit, static_argnums=2)
    def advance(u, dt, steps):
        def body(u, _):
            u = u + dt * rates(u)[0]
            return u.at[:, 4].set(u[:, 3]), None
        return jax.lax.scan(body, u, None, length=steps)[0]

    u = jnp.stack([jnp.ones(cells)] * 3 + [jnp.zeros(cells)] * 2, -1)
    dt = 0.2 * dx / (math.sqrt(3.0 * mu) + 2.0)
    # relaxation slows like 1 / (1 - R): scale the integration time with it
    for _ in range(periods or int(math.ceil(30.0 / (1.0 - coefficient)))):
        u = advance(u, dt, int(2.0 / dt))
    r, _, fst, wall = rates(u)
    return dict(residual=float(jnp.max(jnp.abs(r[:, :4]))), target=np.asarray(fst[-1]),
                wall=float(wall), core=float(core.sum() * dx), u=np.asarray(u))


@pytest.mark.slow
def test_flux_tube_high_recycling_trend():
    """Fixed heating and core source: R up => target density up, target temperature down."""

    runs = {R: _recycling_flux_tube(R) for R in (0.0, 0.5, 0.8)}
    for R, run in runs.items():
        assert run["residual"] < 1e-5, (R, run["residual"])
    n_t = [runs[R]["target"][0] for R in (0.0, 0.5, 0.8)]
    te_t = [runs[R]["target"][1] for R in (0.0, 0.5, 0.8)]
    assert n_t[0] < n_t[1] < n_t[2]
    assert te_t[0] > te_t[1] > te_t[2]


@pytest.mark.slow
def test_flux_tube_particle_balance_converges():
    """(1 - R) wall flux = core source at steady state, to 0.5% at R = 0.8 (the worst case).

    The bulk rows are the characteristic primitive-variable operator, which
    telescopes to the wall-face flux only up to truncation error, and
    1 / (1 - R) amplifies it.  The gap at R = 0.8 is 0.8% with 20 cells and
    0.31% with 40 (order ~1.6); at R = 0 it is 0.16% and 0.05%.  The wall
    closure's face flux is exact, so this is resolution, not a closure defect.
    """

    R = 0.8
    run = _recycling_flux_tube(R, cells=40)
    assert run["residual"] < 1e-5, run["residual"]
    assert (1.0 - R) * run["wall"] == pytest.approx(run["core"], rel=5e-3)


@pytest.mark.slow
def test_hsx_recycling_ledger_closes(tmp_path):
    """Real HSX (set DRBX_TEST_GEOMETRY_BUNDLE): d(nuclei) = explicit sum - (1 - R) face wall loss per step."""

    bundle = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
    if bundle is None:
        pytest.skip("set DRBX_TEST_GEOMETRY_BUNDLE to run the HSX recycling ledger")
    from drbx.cli import main

    for coefficient in (1.0, 0.9):
        out = tmp_path / f"R{coefficient}"
        deck = tmp_path / f"hsx_{coefficient}.toml"
        deck.write_text(
            '[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\n'
            f'geometry = "{bundle}"\noutput = "{out / "hsx"}"\n'
            "final_time = 0.00225\nnum_steps = 3\nsave_every = 3\nneutrals = true\n"
            f"recycling = true\nrecycling_coefficient = {coefficient}\n"
        )
        assert main(["run", str(deck)]) in (0, None)
        data = np.load(out / "hsx.npz")
        rec = {name: data["recycling_ledger"][:, k] for k, name in enumerate(data["recycling_ledger_columns"])}
        wall = rec["wall_ion_loss"]
        assert np.all(wall > 0.0)
        # the deposit is R times the applied implicit loss; the wall loss is the face flux of the stage states
        np.testing.assert_allclose(rec["recycled_particles"], coefficient * wall, rtol=5e-3)
        np.testing.assert_allclose(rec["nuclei_change"], rec["nuclei_change_expected"], rtol=0, atol=5e-3 * wall.max())
        neutral = {name: data["neutral_ledger"][:, k] for k, name in enumerate(data["neutral_ledger_columns"])}
        np.testing.assert_allclose(neutral["wall_influx"], rec["recycled_particles"], rtol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("leak", [0.0, 0.01])
def test_hsx_fluid_neutral_ledger_closes_and_detects_a_one_percent_leak(tmp_path, monkeypatch, leak):
    """Real HSX: the recycling ledger (wall loss from the stage states, independent of the neutral update) closes to
    1e-3 of the wall loss with the fluid neutrals and absorbing wall, and fails when 1% of the deposit is dropped."""

    bundle = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
    if bundle is None:
        pytest.skip("set DRBX_TEST_GEOMETRY_BUNDLE to run the HSX recycling ledger")
    from drbx.cli import main
    from drbx.fci_braginskii.native import fci_neutrals

    original = fci_neutrals.recycled_neutral_source
    monkeypatch.setattr(fci_neutrals, "recycled_neutral_source", lambda *a: (1.0 - leak) * original(*a))
    deck = tmp_path / "hsx.toml"
    deck.write_text(
        '[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\n'
        f'geometry = "{bundle}"\noutput = "{tmp_path / "out" / "hsx"}"\n'
        "final_time = 0.00225\nnum_steps = 3\nsave_every = 3\nneutrals = true\nneutral_model = \"fluid\"\n"
        "neutral_wall_absorption = 0.1\nneutral_wall_energy_reflection = 0.5\n"
        "recycling = true\nrecycling_coefficient = 0.9\n"
    )
    assert main(["run", str(deck)]) in (0, None)
    data = np.load(tmp_path / "out" / "hsx.npz")
    rec = {name: data["recycling_ledger"][:, k] for k, name in enumerate(data["recycling_ledger_columns"])}
    closure = np.max(np.abs(rec["nuclei_change"] - rec["nuclei_change_expected"])) / rec["wall_ion_loss"].max()
    assert (closure < 1e-3) if leak == 0.0 else (closure > 5e-3), closure
    assert data["neutral_velocity"].shape == data["neutral_temperature"].shape == data["neutral_density"].shape
    neutral = {name: data["neutral_ledger"][:, k] for k, name in enumerate(data["neutral_ledger_columns"])}
    assert np.all(neutral["wall_absorbed"] > 0.0) and np.all(np.isfinite(data["neutral_knudsen_gradient"]))
