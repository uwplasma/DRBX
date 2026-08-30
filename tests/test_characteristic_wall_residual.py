from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.characteristic_wall_residual import (
    solve_incoming_characteristic_state,
)


def test_diagonal_incoming_subspace_solves_only_available_components():
    interior = jnp.asarray((1.0, 2.0, 3.0, 4.0))
    target = jnp.asarray((2.0, 9.0, 5.0, 8.0))
    projector = jnp.diag(jnp.asarray((1.0, 0.0, 1.0, 0.0)))
    state, info = solve_incoming_characteristic_state(
        interior, target, projector
    )
    np.testing.assert_allclose(state, (2.0, 2.0, 5.0, 4.0), atol=2e-12)
    assert int(info["incoming_rank"]) == 2
    assert bool(info["solve_valid"])
    assert float(info["retained_error"]) < 1e-12


def test_oblique_projector_does_not_amplify_full_primitive_residual():
    basis = np.asarray(((1.0, 0.0), (20.0, 1.0), (0.0, 1.0)))
    left = np.linalg.pinv(basis)
    projector = jnp.asarray(basis @ left)
    interior = jnp.asarray((1.0, 1.0, 1.0))
    target = jnp.asarray((1.1, 0.8, 1.2))
    state, info = solve_incoming_characteristic_state(
        interior, target, projector, thermodynamic_components=3
    )
    assert float(info["correction_amplification"]) <= 1.0 + 1e-10
    assert float(info["retained_error"]) < 1e-10
    assert bool(jnp.all(jnp.isfinite(state)))


def test_negative_unconstrained_result_is_not_limited():
    projector = jnp.eye(2)
    state, info = solve_incoming_characteristic_state(
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((-2.0, 3.0)),
        projector,
        thermodynamic_components=2,
    )
    assert float(state[0]) < 0.0
    assert not bool(info["thermodynamic_admissible"])
    assert not bool(info["positivity_limited"])


def test_invalid_input_propagates_nonfinite_state_without_fallback():
    state, info = solve_incoming_characteristic_state(
        jnp.ones(2), jnp.asarray((jnp.nan, 1.0)), jnp.eye(2)
    )
    assert not bool(info["solve_valid"])
    assert bool(info["fallback"])
    assert not bool(jnp.all(jnp.isfinite(state)))


def test_batched_solver_is_jittable():
    interior = jnp.ones((2, 3))
    target = jnp.asarray(((1.1, 0.9, 1.2), (0.8, 1.3, 1.1)))
    projector = jnp.broadcast_to(jnp.diag(jnp.asarray((1.0, 0.0, 1.0))), (2, 3, 3))
    state, info = jax.jit(solve_incoming_characteristic_state)(
        interior, target, projector, thermodynamic_components=3
    )
    assert state.shape == (2, 3)
    assert bool(jnp.all(info["solve_valid"]))
