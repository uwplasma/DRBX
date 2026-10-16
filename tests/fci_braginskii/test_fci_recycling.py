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

from drbx.fci_braginskii.native import fci_neutrals as F
from drbx.fci_braginskii.native.fci_parallel_production_flux import (
    parallel_short_wall_material_data,
    parallel_target_row_material_residual,
)
from drbx.fci_braginskii.native.fci_wall_closure import (
    SheathClosure,
    bohm_wall_state,
    sheath_wall_leg_residual,
    wall_face_fluxes,
)
from drbx.fci_braginskii.run import HsxBlobConfig, recycling_parameters

from .test_fci_neutrals import _random_plasma, annulus, sheared_slab

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
    dt = 0.02
    total0 = float(F.inventories(n, Vi, Te, Ti, Nn, grid, PARAMS)[0])
    total = total0
    for step in range(20):
        loss = _wall_loss(geometry, step)
        wall = dt * float(jnp.sum(grid.volume * jnp.where(grid.owner, loss, 0.0)))
        n = jnp.where(grid.owner, n - dt * loss, n)  # the IMEX wall leg
        source = F.recycled_neutral_source(loss, coefficient, grid.owner)
        before = F.inventories(n, Vi, Te, Ti, Nn, grid, PARAMS)
        n, Vi, Te, Ti, Nn, ledger, _ = F.neutral_step(n, Vi, Te, Ti, Nn, dt, grid, PARAMS, volume_source=source)
        after = F.inventories(n, Vi, Te, Ti, Nn, grid, PARAMS)
        # the split adds exactly the recycled particles, and reports them
        assert float(ledger[7]) == pytest.approx(coefficient * wall, rel=1e-13)
        assert abs(float(after[0] - before[0]) - coefficient * wall) < 1e-14 * total0
        # energy: the fixed-T_n inventory gains 3/2 T_n per recycled atom
        sink = float(ledger[3] + ledger[4])
        energy = float(after[1] + after[2] + sink - before[1] - before[2]) - 1.5 * PARAMS.neutral_temperature * float(ledger[7])
        assert abs(energy) < 1e-12 * max(sink, 1.5 * PARAMS.neutral_temperature * float(ledger[7]))
        new_total = float(after[0])
        assert abs((new_total - total) + (1.0 - coefficient) * wall) < 1e-13 * total0
        total = new_total
    if coefficient == 1.0:
        assert abs(total - total0) < 1e-13 * total0
    assert float(jnp.min(jnp.where(grid.owner, Nn, 1.0))) > 0.0


def test_recycling_off_is_the_unchanged_split():
    geometry = sheared_slab()
    grid = F.build_neutral_grid(geometry)
    state = _random_plasma(geometry.shape, 5)
    plain = F.neutral_step(*state, 0.02, grid, PARAMS)
    zero = F.neutral_step(*state, 0.02, grid, PARAMS, volume_source=None)
    for a, b in zip(plain, zero):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


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


def _recycling_flux_tube(coefficient, *, cells=20, mu=25.0, periods=30, e_iz=0.3, e_rec=0.1, decay=0.2):
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
    backward = jnp.zeros(cells, bool).at[0].set(True)
    forward = jnp.zeros(cells, bool).at[-1].set(True)
    rows = jnp.asarray((1.0, 1.0, 1.0, 1.0, 0.0))

    def rates(u):
        minus = jnp.concatenate([u[:1], u[:-1]])
        plus = jnp.concatenate([u[1:], u[-1:]])
        bst = jnp.where(backward[:, None], bohm_wall_state(minus, u, -1.0, TAU), minus)
        fst = jnp.where(forward[:, None], bohm_wall_state(plus, u, 1.0, TAU), plus)
        kw = dict(backward_wall=backward, forward_wall=forward, backward_wall_state=bst, forward_wall_state=fst)
        r, _ = parallel_target_row_material_residual(u, minus, plus, dx, dx, TAU, mu, **kw)
        rw, _, info = parallel_short_wall_material_data(u, minus, plus, dx, dx, TAU, mu, **kw)
        r = r + rw
        r = r + jnp.where(backward[:, None], rows * (
            sheath_wall_leg_residual(bst, u, dx, -1.0, TAU, ge, gi) - info["backward_residual"]), 0.0)
        r = r + jnp.where(forward[:, None], rows * (
            sheath_wall_leg_residual(fst, u, dx, 1.0, TAU, ge, gi) - info["forward_residual"]), 0.0)
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
    for _ in range(periods):
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
        # particle balance at steady state: (1 - R) wall flux = core source
        assert (1.0 - R) * run["wall"] == pytest.approx(run["core"], rel=5e-3)
    n_t = [runs[R]["target"][0] for R in (0.0, 0.5, 0.8)]
    te_t = [runs[R]["target"][1] for R in (0.0, 0.5, 0.8)]
    assert n_t[0] < n_t[1] < n_t[2]
    assert te_t[0] > te_t[1] > te_t[2]


@pytest.mark.slow
def test_hsx_recycling_ledger_closes(tmp_path):
    """Real HSX (set DRBX_TEST_GEOMETRY_BUNDLE): d(nuclei) = other - (1 - R) wall per step."""

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
        np.testing.assert_allclose(rec["recycled_particles"], coefficient * wall, rtol=1e-12)
        np.testing.assert_allclose(rec["nuclei_change"], rec["nuclei_change_expected"], rtol=0, atol=1e-9 * wall.max())
        neutral = {name: data["neutral_ledger"][:, k] for k, name in enumerate(data["neutral_ledger_columns"])}
        np.testing.assert_allclose(neutral["wall_influx"], rec["recycled_particles"], rtol=1e-12)
