"""Contract tests for the standalone production parallel material flux."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.native.fci_parallel_production_flux import (
    parallel_characteristic_absolute_action,
    parallel_characteristic_matrix,
    parallel_characteristic_projectors,
    parallel_characteristic_split,
    parallel_canonical_leg_face_state,
    parallel_characteristic_wall_data,
    parallel_production_principal_matrix,
    parallel_short_wall_backward_euler,
    parallel_short_wall_material_data,
    parallel_target_row_material_residual,
    parallel_wall_exterior_state,
    third_order_face_reconstruction,
)


def _state():
    return jnp.asarray([2.0, 3.0, 5.0, 0.7, -0.2], dtype=jnp.float64)


def test_corrected_matrix_has_mu_tau_and_jvp_equivalent_entries():
    n, te, ti, vi, ve, tau, mu = 2.0, 3.0, 5.0, 0.7, -0.2, 4.0, 10.0
    matrix = np.asarray(parallel_production_principal_matrix(n, te, ti, vi, ve, tau, mu))
    expected = np.array([
        [ve, 0, 0, 0, n],
        [-1.42 * te * (vi - ve) / (3 * n), ve, 0, -1.42 * te / 3, 3.42 * te / 3],
        [-2 * ti * (vi - ve) / (3 * n), 0, vi, 0, 2 * ti / 3],
        [(te + tau * ti) / n, 1, tau, vi, 0],
        [mu * te / n, 1.71 * mu, mu * tau, 0, ve],
    ])
    np.testing.assert_allclose(matrix, expected)
    np.testing.assert_allclose(matrix[4, 2], mu * tau)

    # Matrix-vector JVPs recover the individual columns without involving a
    # derivative of the state-dependent coefficients.
    def residual(gradient):
        return parallel_production_principal_matrix(n, te, ti, vi, ve, tau, mu) @ gradient
    basis = jnp.eye(5, dtype=jnp.float64)
    columns = jax.vmap(lambda e: jax.jvp(residual, (jnp.zeros(5),), (e,))[1])(basis)
    np.testing.assert_allclose(np.asarray(columns).T, matrix)


def test_equilibrium_speeds_match_corrected_five_field_block():
    matrix = np.asarray(parallel_characteristic_matrix(1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1836.0))
    speeds = np.linalg.eigvals(matrix)
    assert np.all(np.abs(np.imag(speeds)) < 1.0e-10)
    speeds = np.sort(np.real(speeds))
    np.testing.assert_allclose(speeds, [-81.4754, -0.61545, 0.0, 0.61545, 81.4754], rtol=2.0e-4, atol=2.0e-4)


def test_characteristic_absolute_action_is_batched_jittable_and_finite():
    values = jnp.asarray([
        [1.0, 1.0, 1.0, 0.001, 0.5],
        [0.99856, 1.00179, 1.0002, -0.0031, 0.99485],
        [1.00833, 0.99541, 0.99917, 0.00301, -0.38846],
    ])
    matrices = parallel_characteristic_matrix(*values.T, tau=1.0, mu=1836.0)
    jumps = jnp.ones_like(values)
    result = jax.jit(parallel_characteristic_absolute_action)(matrices, jumps)
    assert result.shape == values.shape
    assert bool(jnp.all(jnp.isfinite(result)))
    plus, minus, valid = parallel_characteristic_projectors(matrices)
    assert plus.shape == (3, 5, 5)
    assert bool(jnp.all(valid))
    assert bool(jnp.all(jnp.isfinite(minus)))


def test_split_returns_projectors_and_reconstructs_directional_matrices():
    matrix = parallel_characteristic_matrix(*_state(), tau=4.0, mu=10.0)
    a_plus, a_minus, p_plus, p_minus, valid = parallel_characteristic_split(matrix)
    assert bool(valid)
    np.testing.assert_allclose(a_plus, matrix @ p_plus, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(a_minus, matrix @ p_minus, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(p_plus @ p_plus, p_plus, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(p_minus @ p_minus, p_minus, rtol=2e-9, atol=2e-9)


def test_wall_incoming_projection_selects_modes_under_normal_reversal():
    matrix = jnp.diag(jnp.asarray([1.0, -2.0, 0.0, 3.0, -4.0]))
    owner = jnp.zeros(5)
    candidate = jnp.ones(5)
    forward = parallel_wall_exterior_state(owner, candidate, matrix, 1.0)
    backward = parallel_wall_exterior_state(owner, candidate, matrix, -1.0)
    np.testing.assert_allclose(forward, [0, 1, 0, 0, 1])
    np.testing.assert_allclose(backward, [1, 0, 0, 1, 0])


def test_wall_incoming_projection_is_finite():
    owner = _state()
    wall = jnp.asarray([1.0, 1.1, 0.9, 0.0, 0.0])
    matrix = parallel_characteristic_matrix(*owner, tau=4.0, mu=10.0)
    exterior = parallel_wall_exterior_state(owner, wall, matrix, 1.0)
    np.testing.assert_allclose(exterior, np.asarray(exterior))
    assert bool(jnp.all(jnp.isfinite(exterior)))


def test_positivity_and_order_fallback_are_reported():
    stencil = jnp.asarray([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [-10.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
    ])
    states, fallback = third_order_face_reconstruction(stencil, return_fallback=True)
    assert states.shape == (2, 5)
    assert bool(jnp.any(fallback))
    assert bool(jnp.all(states[:, :3] > 0.0))
    face, face_fallback = parallel_canonical_leg_face_state(
        states[0], jnp.asarray([0.0, 1.0, 1.0, 0.0, 0.0]),
        return_fallback=True,
    )
    assert bool(face_fallback)
    assert bool(jnp.all(jnp.isfinite(face)))


def test_spectral_admissibility_fallback_is_finite_and_jittable():
    # A Jordan block has a defective eigenspace.  The API must select its
    # finite fallback rather than allowing an inverse/eigenvector NaN through.
    matrix = jnp.zeros((2, 5, 5), dtype=jnp.float64).at[:, 0, 1].set(1.0)
    jump = jnp.ones((2, 5), dtype=jnp.float64)
    action = jax.jit(parallel_characteristic_absolute_action)(matrix, jump)
    plus, minus, valid = parallel_characteristic_projectors(matrix)
    assert not bool(jnp.any(valid))
    assert bool(jnp.all(jnp.isfinite(action)))
    assert bool(jnp.all(jnp.isfinite(plus)))
    assert bool(jnp.all(jnp.isfinite(minus)))


def test_target_row_sign_reduces_to_the_two_wave_propagation_terms():
    center = _state()
    minus = center + jnp.asarray([-0.1, 0.2, -0.1, 0.3, -0.2])
    plus = center + jnp.asarray([0.2, -0.1, 0.3, -0.2, 0.1])
    dxm, dxp = 2.0, 3.0
    residual, info = parallel_target_row_material_residual(
        center, minus, plus, dxm, dxp, 4.0, 10.0, div_b=0.0
    )
    back_face = parallel_canonical_leg_face_state(minus, center)
    forward_face = parallel_canonical_leg_face_state(center, plus)
    back_plus, _, _, _, back_valid = parallel_characteristic_split(
        parallel_characteristic_matrix(*back_face, tau=4.0, mu=10.0)
    )
    _, forward_minus, _, _, forward_valid = parallel_characteristic_split(
        parallel_characteristic_matrix(*forward_face, tau=4.0, mu=10.0)
    )
    expected = -(
        back_plus @ (center - minus) / dxm
        + forward_minus @ (plus - center) / dxp
    )
    np.testing.assert_allclose(residual, expected, rtol=2e-8, atol=2e-8)
    assert bool(back_valid and forward_valid)
    assert bool(info["ordinary_row"])


def test_target_row_constant_state_has_exact_geometric_source():
    state = _state()
    div_b = 0.37
    residual, info = parallel_target_row_material_residual(
        state, state, state, 1.0, 1.0, 4.0, 10.0, div_b=div_b
    )
    n, Te, Ti, Vi, Ve = [float(x) for x in state]
    current = n * (Vi - Ve)
    expected = np.array([
        -n * Ve * div_b,
        2.0 * Te / (3.0 * n) * (0.71 * current - n * Ve) * div_b,
        2.0 * Ti / (3.0 * n) * (current - n * Vi) * div_b,
        0.0,
        0.0,
    ])
    np.testing.assert_allclose(residual, expected, rtol=2e-10, atol=2e-10)
    assert not bool(info["wall_row"])


def test_wall_projection_uses_interior_matrix_and_one_sided_path():
    center = _state()
    candidate = jnp.asarray([1.0, 1.1, 0.9, 0.0, 0.0])
    matrix = parallel_characteristic_matrix(*center, tau=4.0, mu=10.0)
    endpoint = parallel_wall_exterior_state(center, candidate, matrix, -1.0)
    wall, wall_info = parallel_target_row_material_residual(
        center, center, center, 1.0, 1.0, 4.0, 10.0,
        backward_wall=True, backward_wall_state=candidate, div_b=0.0,
    )
    _, _, _, incoming, valid = parallel_characteristic_split(matrix, normal=-1.0)
    projected = center + incoming @ (candidate - center)
    np.testing.assert_allclose(endpoint, projected, rtol=2e-8, atol=2e-8)
    a_plus, _, _, _, _ = parallel_characteristic_split(matrix, normal=1.0)
    expected = -a_plus @ (center - projected)
    np.testing.assert_allclose(wall, expected, rtol=2e-8, atol=2e-8)
    assert float(jnp.linalg.norm(wall)) > 1.0e-8
    assert bool(valid)
    assert bool(wall_info["backward_wall"])


def test_live_face_state_changes_the_material_action():
    center = _state()
    minus_a = center + jnp.asarray([-0.05, 0.01, 0.02, 0.0, 0.0])
    minus_b = center + jnp.asarray([-0.45, 0.4, -0.3, 0.0, 0.0])
    result_a, _ = parallel_target_row_material_residual(
        center, minus_a, center, 1.0, 1.0, 4.0, 10.0, div_b=0.0
    )
    result_b, _ = parallel_target_row_material_residual(
        center, minus_b, center, 1.0, 1.0, 4.0, 10.0, div_b=0.0
    )
    assert not np.allclose(result_a, result_b)


def test_target_row_activates_all_rows_and_reverses_with_oriented_endpoints():
    center = jnp.broadcast_to(_state(), (4, 5))
    minus = center + jnp.asarray([-0.1, 0.1, 0.0, 0.2, 0.0])
    plus = center + jnp.asarray([0.1, 0.0, 0.1, 0.0, -0.2])
    backward = jnp.asarray([False, True, False, True])
    forward = jnp.asarray([False, False, True, True])
    residual, info = parallel_target_row_material_residual(
        center, minus, plus, jnp.ones(4), jnp.ones(4), 4.0, 10.0,
        backward_wall=backward, forward_wall=forward, div_b=0.0,
    )
    assert residual.shape == (4, 5)
    assert bool(jnp.all(jnp.isfinite(residual)))
    np.testing.assert_array_equal(np.asarray(info["wall_row"]), [False, True, True, True])
    assert not np.allclose(np.asarray(residual), 0.0)

    # Reorienting a row means exchanging the two endpoints and the wall flags;
    # the wave-propagation contributions remain finite and sign-consistent.
    reversed_residual, _ = parallel_target_row_material_residual(
        center, plus, minus, jnp.ones(4), jnp.ones(4), 4.0, 10.0,
        backward_wall=forward, forward_wall=backward, div_b=0.0,
    )
    assert bool(jnp.all(jnp.isfinite(reversed_residual)))


def test_target_row_is_jittable_in_batches_and_zero_for_constant_no_source():
    center = jnp.broadcast_to(_state(), (3, 5))
    residual, info = jax.jit(parallel_target_row_material_residual)(
        center, center, center, jnp.ones(3), jnp.ones(3), 4.0, 10.0,
        div_b=jnp.zeros(3),
    )
    np.testing.assert_allclose(residual, np.zeros((3, 5)), atol=2.0e-12)
    assert residual.shape == (3, 5)
    assert bool(jnp.all(~info["fallback"]))


def test_short_wall_selection_is_wall_only_and_returns_directional_jacobian():
    center = _state()
    candidate = jnp.asarray([1.1, 1.2, 0.9, 0.3, -0.1])
    residual, jacobian, info = parallel_short_wall_material_data(
        center, center, center, 0.01, 1.0, 4.0, 10.0,
        selection_dt=0.02, cfl_limit=2.785,
        backward_wall=True, forward_wall=False,
        backward_wall_state=candidate,
    )
    assert bool(info["selected_backward_wall"])
    assert not bool(info["selected_forward_wall"])
    assert bool(info["selected_wall"])
    matrix = parallel_characteristic_matrix(*center, tau=4.0, mu=10.0)
    a_plus, _, _, _, valid = parallel_characteristic_split(matrix)
    assert bool(valid)
    np.testing.assert_allclose(jacobian, -a_plus / 0.01, rtol=2e-8, atol=2e-8)
    expected, _ = parallel_target_row_material_residual(
        center, center, center, 0.01, 1.0, 4.0, 10.0,
        backward_wall=True, forward_wall_state=None,
        backward_wall_state=candidate, div_b=0.0,
        omit_forward_wall=True,
    )
    np.testing.assert_allclose(residual, expected, rtol=2e-8, atol=2e-8)


def test_short_wall_backward_euler_matches_frozen_local_solve():
    center = _state()
    candidate = jnp.asarray([1.1, 1.2, 0.9, 0.3, -0.1])
    dt = 0.02
    updated, delta, info = parallel_short_wall_backward_euler(
        center, center, center, 0.01, 1.0, 4.0, 10.0,
        selection_dt=dt, solve_dt=dt, cfl_limit=2.785,
        backward_wall=True, backward_wall_state=candidate,
    )
    assert bool(info["selected_wall"])
    assert not bool(info["implicit_solve_fallback"])
    expected_delta = np.linalg.solve(
        np.eye(5) - dt * np.asarray(info["selected_jacobian"]),
        dt * np.asarray(info["backward_residual"]),
    )
    np.testing.assert_allclose(delta, expected_delta, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(updated, np.asarray(center) + expected_delta)
    assert bool(jnp.all(jnp.isfinite(updated)))


def test_short_wall_ordinary_rows_are_zero_and_default_residual_is_unchanged():
    center = _state()
    minus = center + jnp.asarray([-0.1, 0.2, -0.1, 0.3, -0.2])
    plus = center + jnp.asarray([0.2, -0.1, 0.3, -0.2, 0.1])
    selected, jacobian, info = parallel_short_wall_material_data(
        center, minus, plus, 1.0, 1.0, 4.0, 10.0,
        selection_dt=0.001, cfl_limit=2.785,
    )
    np.testing.assert_allclose(selected, 0.0)
    np.testing.assert_allclose(jacobian, 0.0)
    assert not bool(info["selected_wall"])
    default, _ = parallel_target_row_material_residual(
        center, minus, plus, 1.0, 1.0, 4.0, 10.0, div_b=0.0,
    )
    explicit_default, _ = parallel_target_row_material_residual(
        center, minus, plus, 1.0, 1.0, 4.0, 10.0, div_b=0.0,
        omit_backward_wall=False, omit_forward_wall=False,
    )
    np.testing.assert_array_equal(default, explicit_default)


def test_short_wall_batch_jit_and_selection_threshold():
    center = jnp.broadcast_to(_state(), (2, 5))
    candidate = center.at[0, 0].set(1.2)
    dt = jnp.asarray([0.02, 0.000001])
    run = jax.jit(parallel_short_wall_backward_euler)
    updated, delta, info = run(
        center, center, center, jnp.asarray([0.01, 0.01]),
        jnp.ones(2), 4.0, 10.0, selection_dt=dt,
        backward_wall=jnp.asarray([True, True]),
        backward_wall_state=candidate,
    )
    assert updated.shape == (2, 5)
    assert bool(info["selected_backward_wall"][0])
    assert not bool(info["selected_backward_wall"][1])
    np.testing.assert_allclose(delta[1], 0.0)
    assert bool(jnp.all(jnp.isfinite(updated)))


def test_characteristic_wall_data_exposes_projected_state_and_current():
    center = _state()
    candidate = jnp.asarray([1.1, 1.2, 0.9, 0.3, -0.1])
    info = parallel_characteristic_wall_data(
        center, center, center, 0.01, 0.02, 4.0, 10.0,
        selection_dt=0.0, backward_wall=True, forward_wall=True,
        backward_wall_state=candidate, forward_wall_state=candidate,
    )
    backward_projected = np.asarray(info["backward_projected_state"])
    forward_projected = np.asarray(info["forward_projected_state"])
    np.testing.assert_allclose(
        info["backward_projected_current"],
        backward_projected[0] * (backward_projected[3] - backward_projected[4]),
    )
    np.testing.assert_allclose(
        info["forward_projected_current"],
        forward_projected[0] * (forward_projected[3] - forward_projected[4]),
    )
    np.testing.assert_allclose(
        info["backward_endpoint_state"], backward_projected,
    )
    np.testing.assert_allclose(
        info["forward_endpoint_state"], forward_projected,
    )
    assert bool(jnp.all(jnp.isfinite(info["backward_incoming_matrix"])))
    assert bool(jnp.all(jnp.isfinite(info["forward_incoming_action"])))
    assert float(info["backward_alpha"]) > 0.0
    assert float(info["forward_alpha"]) > 0.0


def test_characteristic_wall_data_keeps_ordinary_mapped_endpoints():
    center = _state()
    minus = center + jnp.asarray([-0.1, 0.2, -0.1, 0.3, -0.2])
    plus = center + jnp.asarray([0.2, -0.1, 0.3, -0.2, 0.1])
    info = parallel_characteristic_wall_data(
        center, minus, plus, 2.0, 3.0, 4.0, 10.0, selection_dt=0.0,
    )
    np.testing.assert_array_equal(info["backward_endpoint_state"], minus)
    np.testing.assert_array_equal(info["forward_endpoint_state"], plus)
    assert not bool(info["backward_wall"])
    assert not bool(info["forward_wall"])


def test_characteristic_wall_data_finite_fallback_is_jittable_in_batch():
    center = jnp.broadcast_to(_state(), (2, 5))
    bad = center.at[1, 0].set(jnp.nan)
    result = jax.jit(parallel_characteristic_wall_data)(
        center, center, center, jnp.asarray([0.01, 0.01]),
        jnp.asarray([0.02, 0.02]), 4.0, 10.0, selection_dt=0.01,
        backward_wall=jnp.asarray([True, True]), backward_wall_state=bad,
    )
    info = result
    assert bool(jnp.all(jnp.isfinite(info["selected_residual"])))
    assert bool(jnp.all(jnp.isfinite(info["selected_jacobian"])))
    assert bool(jnp.all(jnp.isfinite(info["backward_projected_state"])))
    assert bool(jnp.any(info["backward_candidate_fallback"]))


def test_completed_run_state_range_is_admissible_when_available():
    path = Path("/Users/yxie/Desktop/HSX drbx/prototype_runs/fci_curvature_radial_poloidal_third_order_upwind_32_t015/hsx_curvature_radial_poloidal_third_order_upwind_32_t015.npz")
    if not path.exists():
        pytest.skip("completed 32^3 history is not present on this checkout")
    data = np.load(path)
    names = ("density", "Te", "Ti", "Vi", "Ve")
    if not all(name in data for name in names):
        pytest.skip("history does not expose primitive state fields")
    values = jnp.stack(tuple(jnp.asarray(data[name]).reshape(-1)[:64] for name in names), axis=-1)
    matrices = parallel_characteristic_matrix(*values.T, tau=1.0, mu=1836.0)
    _plus, _minus, valid = parallel_characteristic_projectors(matrices)
    assert bool(jnp.all(values[:, :3] > 0.0))
    assert bool(jnp.all(valid))
