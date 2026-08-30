from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.characteristic_wall_residual import (
    apply_maximally_dissipative_characteristic_wall,
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


def test_direct_map_diagonal_exact_absorber_and_diagnostics():
    interior = jnp.asarray((3.0, 2.0, 4.0))
    reference = jnp.asarray((1.0, 1.0, 1.0))
    eigenvalues = jnp.asarray((-2.0, 0.0, 3.0))
    state, info = apply_maximally_dissipative_characteristic_wall(
        interior, reference, eigenvalues, jnp.eye(3), jnp.eye(3),
        thermodynamic_components=3,
    )
    np.testing.assert_allclose(state, (1.0, 2.0, 4.0))
    assert int(info["incoming_rank"]) == 1
    assert int(info["outgoing_rank"]) == 1
    assert int(info["stationary_rank"]) == 1
    assert bool(info["solve_valid"])
    assert float(info["outgoing_retained_error"]) < 1e-12
    assert float(info["modal_reconstruction_error"]) < 1e-12


def test_direct_map_oblique_basis_retains_outgoing_mode():
    right = jnp.asarray(((1.0, 1.0), (0.0, 1.0)))
    left = jnp.linalg.inv(right)
    interior = right @ jnp.asarray((2.0, 5.0))
    state, info = apply_maximally_dissipative_characteristic_wall(
        interior, jnp.zeros(2), jnp.asarray((-1.0, 2.0)), right, left
    )
    np.testing.assert_allclose(state, right @ jnp.asarray((0.0, 5.0)))
    assert float(info["outgoing_retained_error"]) < 1e-12


def test_direct_map_power_and_incoming_energy_do_not_increase():
    state, info = apply_maximally_dissipative_characteristic_wall(
        jnp.asarray((2.0, 3.0)), jnp.zeros(2), jnp.asarray((-2.0, 1.0)),
        jnp.eye(2), jnp.eye(2),
    )
    np.testing.assert_allclose(state, (0.0, 3.0))
    assert float(info["incoming_energy_after"]) <= float(
        info["incoming_energy_before"]
    )
    np.testing.assert_allclose(info["boundary_power_before"], 1.0)
    np.testing.assert_allclose(info["boundary_power_after"], 9.0)
    np.testing.assert_allclose(info["domain_energy_rate_before"], -1.0)
    np.testing.assert_allclose(info["domain_energy_rate_after"], -9.0)


def test_direct_map_oriented_speeds_and_nonzero_source():
    # The supplied oriented speeds carry the lower/upper normal orientation;
    # the map itself does not infer or reverse that orientation.
    _, lower = apply_maximally_dissipative_characteristic_wall(
        jnp.asarray((2.0, 3.0)), jnp.zeros(2), jnp.asarray((-1.0, 1.0)),
        jnp.eye(2), jnp.eye(2), incoming_source=jnp.asarray((7.0, 8.0)),
    )
    upper_state, upper = apply_maximally_dissipative_characteristic_wall(
        jnp.asarray((2.0, 3.0)), jnp.zeros(2), jnp.asarray((1.0, -1.0)),
        jnp.eye(2), jnp.eye(2), incoming_source=jnp.asarray((7.0, 8.0)),
    )
    np.testing.assert_allclose(upper_state, (2.0, 8.0))
    assert int(lower["incoming_rank"]) == int(upper["incoming_rank"]) == 1


def test_direct_map_is_invariant_to_dual_eigenvector_scaling():
    scales = jnp.asarray((4.0, 0.25))
    right = jnp.diag(scales)
    left = jnp.diag(1.0 / scales)
    unscaled, _ = apply_maximally_dissipative_characteristic_wall(
        jnp.asarray((2.0, 3.0)), jnp.zeros(2), jnp.asarray((-1.0, 2.0)),
        jnp.eye(2), jnp.eye(2)
    )
    scaled, info = apply_maximally_dissipative_characteristic_wall(
        jnp.asarray((2.0, 3.0)), jnp.zeros(2), jnp.asarray((-1.0, 2.0)),
        right, left
    )
    np.testing.assert_allclose(scaled, unscaled)
    assert bool(info["solve_valid"])


def test_direct_map_invalid_input_and_inadmissible_state_are_distinct():
    state, info = apply_maximally_dissipative_characteristic_wall(
        jnp.asarray((jnp.nan, 1.0)), jnp.zeros(2), jnp.asarray((-1.0, 1.0)),
        jnp.eye(2), jnp.eye(2), thermodynamic_components=2,
    )
    assert not bool(info["solve_valid"])
    assert bool(info["fallback"])
    assert bool(jnp.any(jnp.isnan(state)))
    negative, negative_info = apply_maximally_dissipative_characteristic_wall(
        jnp.asarray((-2.0, 1.0)), jnp.zeros(2), jnp.asarray((-1.0, 1.0)),
        jnp.eye(2), jnp.eye(2), thermodynamic_components=2,
    )
    assert bool(jnp.all(jnp.isfinite(negative)))
    assert not bool(negative_info["thermodynamic_admissible"])
    assert bool(negative_info["solve_valid"])
    assert not bool(negative_info["fallback"])


def test_direct_map_symmetrizer_is_spd_and_zeroes_incoming_amplitude():
    right = jnp.asarray(((1.0, 1.0), (0.0, 1.0)))
    left = jnp.linalg.inv(right)
    lambdas = jnp.asarray((-2.0, 3.0))
    a = right @ jnp.diag(lambdas) @ left
    h = left.T @ left
    np.testing.assert_allclose(h @ a, (h @ a).T, atol=1e-12)
    assert bool(jnp.all(jnp.linalg.eigvalsh(h) > 0.0))

    interior = right @ jnp.asarray((4.0, 5.0))
    state, info = apply_maximally_dissipative_characteristic_wall(
        interior, jnp.zeros(2), lambdas, right, left
    )
    wall_amplitudes = left @ state
    np.testing.assert_allclose(wall_amplitudes[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(wall_amplitudes[1], 5.0, atol=1e-12)
    assert bool(info["solve_valid"])


def test_direct_map_complex_basis_uses_real_part_but_honors_invalid_spectrum():
    complex_right = jnp.eye(2, dtype=jnp.complex128) + 1j * jnp.ones((2, 2))
    state, info = apply_maximally_dissipative_characteristic_wall(
        jnp.asarray((2.0, 3.0)), jnp.zeros(2), jnp.asarray((-1.0, 1.0)),
        complex_right, complex_right, spectral_valid=False,
    )
    assert bool(info["fallback"])
    assert bool(jnp.all(jnp.isnan(state)))


def test_direct_map_batched_jit():
    interior = jnp.asarray(((2.0, 3.0), (4.0, 5.0)))
    state, info = jax.jit(apply_maximally_dissipative_characteristic_wall)(
        interior, jnp.zeros(2), jnp.asarray((-1.0, 1.0)),
        jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
        jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
    )
    np.testing.assert_allclose(state, ((0.0, 3.0), (0.0, 5.0)))
    assert bool(jnp.all(info["solve_valid"]))
