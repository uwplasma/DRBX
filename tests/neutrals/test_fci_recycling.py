"""Prescribed wall-loss recycling on synthetic grids."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.neutrals import fci_neutral_fluid as FL
from drbx.neutrals import fci_neutrals as F

from tests.neutrals.test_fci_neutrals import _random_plasma, annulus, sheared_slab

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
