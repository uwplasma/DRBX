from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_drb.native.fci_2_field_rhs import Fci2FieldState
from jax_drb.native.fci_4_field_rhs import Fci4FieldState
from jax_drb.native.fci_drb_EB_rhs import FciDrbEBState
from jax_drb.native.fci_drb_rhs import FciDrbState


def _round_trip_state(state):
    leaves, treedef = jax.tree_util.tree_flatten(state)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    for name in state.field_names():
        assert jnp.allclose(getattr(rebuilt, name), getattr(state, name))


def _check_state_helpers(state, *, field_names):
    assert state.field_names() == field_names
    zero = state.zeros_like()
    for name in field_names:
        assert jnp.allclose(getattr(zero, name), 0.0)
    doubled = state.axpy(state, scale=1.0)
    for name in field_names:
        assert jnp.allclose(getattr(doubled, name), 2.0 * getattr(state, name))
    _round_trip_state(state)


def test_2field_state_helpers() -> None:
    state = Fci2FieldState(
        density=jnp.ones((2, 2, 2)),
        v_parallel=2.0 * jnp.ones((2, 2, 2)),
        density_background=3.0 * jnp.ones((2, 2, 2)),
    )
    _check_state_helpers(state, field_names=("density", "v_parallel", "density_background"))


def test_4field_state_helpers() -> None:
    state = Fci4FieldState(
        density=jnp.ones((2, 2, 2)),
        omega=2.0 * jnp.ones((2, 2, 2)),
        v_ion_parallel=3.0 * jnp.ones((2, 2, 2)),
        v_electron_parallel=4.0 * jnp.ones((2, 2, 2)),
    )
    _check_state_helpers(state, field_names=("density", "omega", "v_ion_parallel", "v_electron_parallel"))


def test_eb_state_helpers() -> None:
    state = FciDrbEBState(
        density=jnp.ones((2, 2, 2)),
        phi=2.0 * jnp.ones((2, 2, 2)),
        Te=3.0 * jnp.ones((2, 2, 2)),
        Ti=4.0 * jnp.ones((2, 2, 2)),
        Vi=5.0 * jnp.ones((2, 2, 2)),
        Ve=6.0 * jnp.ones((2, 2, 2)),
        vorticity=7.0 * jnp.ones((2, 2, 2)),
    )
    _check_state_helpers(state, field_names=("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity"))


def test_drb_state_helpers() -> None:
    state = FciDrbState(
        ion_density=jnp.ones((2, 2, 2)),
        electron_density=2.0 * jnp.ones((2, 2, 2)),
        neutral_density=3.0 * jnp.ones((2, 2, 2)),
        ion_pressure=4.0 * jnp.ones((2, 2, 2)),
        electron_pressure=5.0 * jnp.ones((2, 2, 2)),
        neutral_pressure=6.0 * jnp.ones((2, 2, 2)),
        ion_momentum=7.0 * jnp.ones((2, 2, 2)),
        neutral_momentum=8.0 * jnp.ones((2, 2, 2)),
        vorticity=9.0 * jnp.ones((2, 2, 2)),
    )
    _check_state_helpers(
        state,
        field_names=(
            "ion_density",
            "electron_density",
            "neutral_density",
            "ion_pressure",
            "electron_pressure",
            "neutral_pressure",
            "ion_momentum",
            "neutral_momentum",
            "vorticity",
        ),
    )
