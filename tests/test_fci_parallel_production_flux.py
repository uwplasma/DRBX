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


def test_wall_residual_solve_uses_interior_matrix_and_one_sided_path():
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
    data = parallel_characteristic_wall_data(
        center, center, center, 1.0, 1.0, 4.0, 10.0,
        backward_wall=True, backward_wall_state=candidate,
    )
    solved = data["backward_endpoint_state"]
    np.testing.assert_allclose(
        (np.eye(5) - np.asarray(incoming)) @ np.asarray(solved - center),
        np.zeros(5), atol=2.0e-8, rtol=0.0,
    )
    assert np.linalg.norm(np.asarray(solved - candidate)) <= np.linalg.norm(
        np.asarray(center - candidate)
    ) + 1.0e-12
    a_plus, _, _, _, _ = parallel_characteristic_split(matrix, normal=1.0)
    expected = -a_plus @ (center - solved)
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


def _energy_wall_selection_batch():
    equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0))
    center = jnp.broadcast_to(
        equilibrium + jnp.asarray((0.08, -0.02, 0.03, 0.0, 0.0)),
        (4, 5),
    )
    minus = center + jnp.asarray((
        (0.01, -0.01, 0.02, 0.03, -0.02),
        (-0.02, 0.01, -0.01, -0.02, 0.03),
        (0.03, 0.02, -0.02, 0.01, 0.02),
        (-0.01, -0.02, 0.01, 0.02, -0.01),
    ))
    plus = center + jnp.asarray((
        (-0.01, 0.02, -0.01, -0.02, 0.01),
        (0.02, -0.01, 0.03, 0.01, -0.02),
        (-0.03, 0.01, 0.02, -0.01, 0.03),
        (0.01, 0.01, -0.02, 0.02, 0.01),
    ))
    backward_wall = jnp.asarray((True, False, True, False))
    forward_wall = jnp.asarray((False, True, True, False))
    return (
        equilibrium, center, minus, plus,
        jnp.asarray((1.0e-3, 100.0, 0.5, 2.0)),
        jnp.asarray((100.0, 1.0e-3, 0.5, 2.0)),
        backward_wall, forward_wall,
    )


@pytest.mark.parametrize(
    "wall_law", ("primitive-least-residual", "energy-absorbing")
)
def test_all_physical_wall_selection_ignores_cfl_and_keeps_ordinary_legs_off(
    wall_law,
):
    (
        equilibrium, center, minus, plus, dx_minus, dx_plus,
        backward_wall, forward_wall,
    ) = _energy_wall_selection_batch()
    selected, jacobian, info = parallel_short_wall_material_data(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0,
        selection_dt=1.0e-6, cfl_limit=1.0e12,
        parallel_short_leg_selection="all-physical-walls",
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=minus, forward_wall_state=plus,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law=wall_law,
    )
    np.testing.assert_array_equal(info["selected_backward_wall"], backward_wall)
    np.testing.assert_array_equal(info["selected_forward_wall"], forward_wall)
    np.testing.assert_array_equal(
        info["selected_wall"], backward_wall | forward_wall
    )
    assert not bool(info["selected_wall"][3])
    assert bool(jnp.all(jnp.isfinite(selected)))
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    for name, mask in (
        ("backward_wall_thermodynamic_admissible", backward_wall),
        ("forward_wall_thermodynamic_admissible", forward_wall),
    ):
        assert bool(jnp.all(info[name][mask]))


@pytest.mark.parametrize(
    "wall_law", ("primitive-least-residual", "energy-absorbing")
)
def test_all_physical_wall_target_omits_exactly_selected_directional_actions(
    wall_law,
):
    (
        equilibrium, center, minus, plus, dx_minus, dx_plus,
        backward_wall, forward_wall,
    ) = _energy_wall_selection_batch()
    baseline, _ = parallel_target_row_material_residual(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0,
        selection_dt=0.0, cfl_limit=1.0e12,
        parallel_short_leg_selection="cfl",
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=minus, forward_wall_state=plus,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law=wall_law,
        div_b=0.0,
    )
    filtered, filtered_info = parallel_target_row_material_residual(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0,
        selection_dt=1.0e-6, cfl_limit=1.0e12,
        parallel_short_leg_selection="all-physical-walls",
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=minus, forward_wall_state=plus,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law=wall_law,
        div_b=0.0,
    )
    selected, _, selected_info = parallel_short_wall_material_data(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0,
        selection_dt=1.0e-6, cfl_limit=1.0e12,
        parallel_short_leg_selection="all-physical-walls",
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=minus, forward_wall_state=plus,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law=wall_law,
    )
    np.testing.assert_allclose(baseline - filtered, selected, rtol=2e-10, atol=2e-11)
    np.testing.assert_array_equal(filtered_info["omitted_backward_wall"], backward_wall)
    np.testing.assert_array_equal(filtered_info["omitted_forward_wall"], forward_wall)
    np.testing.assert_array_equal(
        selected_info["selected_backward_wall"], backward_wall
    )
    np.testing.assert_array_equal(selected_info["selected_forward_wall"], forward_wall)
    np.testing.assert_allclose(filtered[3], baseline[3], atol=2e-12)


@pytest.mark.parametrize(
    "wall_law", ("primitive-least-residual", "energy-absorbing")
)
def test_all_physical_wall_backward_euler_includes_both_walls_once_on_long_legs(
    wall_law,
):
    (
        equilibrium, center, minus, plus, dx_minus, dx_plus,
        backward_wall, forward_wall,
    ) = _energy_wall_selection_batch()
    solve_dt = 1.0e-3
    selected, _, selected_info = parallel_short_wall_material_data(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0,
        selection_dt=1.0e-6, cfl_limit=1.0e12,
        parallel_short_leg_selection="all-physical-walls",
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=minus, forward_wall_state=plus,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law=wall_law,
    )
    updated, delta, info = parallel_short_wall_backward_euler(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0,
        selection_dt=1.0e-6, solve_dt=solve_dt, cfl_limit=1.0e12,
        parallel_short_leg_selection="all-physical-walls",
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=minus, forward_wall_state=plus,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law=wall_law,
    )
    expected = np.linalg.solve(
        np.eye(5)[None, :, :] - solve_dt * np.asarray(info["selected_jacobian"]),
        solve_dt * np.asarray(selected)[..., None],
    )[..., 0]
    np.testing.assert_allclose(delta, expected, rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(updated, np.asarray(center) + expected)
    assert bool(jnp.all(jnp.isfinite(updated)))
    assert bool(jnp.all(info["implicit_finite"]))
    assert not bool(jnp.any(info["implicit_solve_fallback"]))
    np.testing.assert_array_equal(
        info["selected_backward_wall"], selected_info["selected_backward_wall"]
    )
    np.testing.assert_array_equal(
        info["selected_forward_wall"], selected_info["selected_forward_wall"]
    )
    assert bool(jnp.all(info["backward_wall_thermodynamic_admissible"][backward_wall]))
    assert bool(jnp.all(info["forward_wall_thermodynamic_admissible"][forward_wall]))


def test_default_short_leg_selection_is_exactly_explicit_cfl():
    (
        equilibrium, center, minus, plus, dx_minus, dx_plus,
        backward_wall, forward_wall,
    ) = _energy_wall_selection_batch()
    common = dict(
        selection_dt=0.02, cfl_limit=2.785,
        backward_wall=backward_wall, forward_wall=forward_wall,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law="energy-absorbing",
    )
    default_data = parallel_short_wall_material_data(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0, **common
    )
    explicit_data = parallel_short_wall_material_data(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0,
        parallel_short_leg_selection="cfl", **common
    )
    for lhs, rhs in zip(default_data[:2], explicit_data[:2], strict=True):
        np.testing.assert_array_equal(lhs, rhs)
    assert default_data[2].keys() == explicit_data[2].keys()
    for name in default_data[2]:
        np.testing.assert_array_equal(default_data[2][name], explicit_data[2][name])

    default_target = parallel_target_row_material_residual(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0, div_b=0.0, **common
    )
    explicit_target = parallel_target_row_material_residual(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0, div_b=0.0,
        parallel_short_leg_selection="cfl", **common
    )
    np.testing.assert_array_equal(default_target[0], explicit_target[0])
    assert default_target[1].keys() == explicit_target[1].keys()
    for name in default_target[1]:
        np.testing.assert_array_equal(default_target[1][name], explicit_target[1][name])

    default_be = parallel_short_wall_backward_euler(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0,
        solve_dt=0.02, **common
    )
    explicit_be = parallel_short_wall_backward_euler(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0,
        solve_dt=0.02, parallel_short_leg_selection="cfl", **common
    )
    for lhs, rhs in zip(default_be[:2], explicit_be[:2], strict=True):
        np.testing.assert_array_equal(lhs, rhs)
    assert default_be[2].keys() == explicit_be[2].keys()
    for name in default_be[2]:
        np.testing.assert_array_equal(default_be[2][name], explicit_be[2][name])


def test_short_leg_selection_rejects_unknown_selector_for_all_public_helpers():
    center = _state()
    with pytest.raises(ValueError, match="parallel_short_leg_selection"):
        parallel_target_row_material_residual(
            center, center, center, 1.0, 1.0, 4.0, 10.0,
            parallel_short_leg_selection="invalid",
        )
    with pytest.raises(ValueError, match="parallel_short_leg_selection"):
        parallel_short_wall_material_data(
            center, center, center, 1.0, 1.0, 4.0, 10.0,
            selection_dt=0.01, parallel_short_leg_selection="invalid",
        )
    with pytest.raises(ValueError, match="parallel_short_leg_selection"):
        parallel_short_wall_backward_euler(
            center, center, center, 1.0, 1.0, 4.0, 10.0,
            selection_dt=0.01, solve_dt=0.01,
            parallel_short_leg_selection="invalid",
        )


@pytest.mark.parametrize(
    ("backward_wall", "forward_wall", "dx_minus", "dx_plus"),
    (
        (True, False, 0.01, 1.0),
        (True, False, 1.0, 1.0),
        (False, True, 1.0, 0.01),
        (False, True, 1.0, 1.0),
    ),
)
def test_all_physical_wall_selected_action_is_dissipative_for_short_and_long_legs(
    backward_wall, forward_wall, dx_minus, dx_plus
):
    equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0))
    center = equilibrium + jnp.asarray((0.08, -0.02, 0.03, 0.0, 0.0))
    matrix = np.asarray(
        parallel_characteristic_matrix(*center, tau=4.0, mu=10.0)
    )
    values, right = np.linalg.eig(matrix)
    assert np.max(np.abs(np.imag(values))) < 1.0e-10
    left = np.linalg.inv(np.real(right))
    H = left.T @ left
    selected, _, info = parallel_short_wall_material_data(
        center, center, center, dx_minus, dx_plus, 4.0, 10.0,
        selection_dt=1.0e-6, cfl_limit=1.0e12,
        parallel_short_leg_selection="all-physical-walls",
        backward_wall=backward_wall, forward_wall=forward_wall,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law="energy-absorbing",
    )
    assert bool(info["selected_wall"])
    assert bool(jnp.all(jnp.isfinite(selected)))
    perturbation = np.asarray(center - equilibrium)
    selected = np.asarray(selected)
    rate = float(perturbation @ H @ selected)
    tolerance = 2.0e-10 * max(
        1.0, np.linalg.norm(perturbation @ H) * np.linalg.norm(selected)
    )
    assert rate <= tolerance


def test_characteristic_wall_data_exposes_projected_state_and_linearized_current():
    center = _state()
    candidate = jnp.asarray([1.1, 1.2, 0.9, 0.3, -0.1])
    info = parallel_characteristic_wall_data(
        center, center, center, 0.01, 0.02, 4.0, 10.0,
        selection_dt=0.0, backward_wall=True, forward_wall=True,
        backward_wall_state=candidate, forward_wall_state=candidate,
    )
    backward_projected = np.asarray(info["backward_projected_state"])
    forward_projected = np.asarray(info["forward_projected_state"])
    center_current = center[0] * (center[3] - center[4])
    for direction, projected in (
        ("backward", backward_projected),
        ("forward", forward_projected),
    ):
        delta = projected - np.asarray(center)
        expected = (
            center_current
            + (center[3] - center[4]) * delta[0]
            + center[0] * (delta[3] - delta[4])
        )
        np.testing.assert_allclose(info[f"{direction}_projected_current"], expected)
        np.testing.assert_allclose(
            info[f"{direction}_wall_characteristic_current"], expected,
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


def test_characteristic_wall_residual_solve_removes_fatal_projection_amplification():
    # Rounded values from the late 48^3 wall failure.  The former oblique
    # projection generated O(10^3) primitive/current artifacts.  The reduced
    # residual solve stays in the incoming subspace without amplifying the
    # primitive candidate mismatch.
    center = jnp.asarray(
        [0.3985, 0.63365, 2.65956, -0.26323, -69.8749],
        dtype=jnp.float64,
    )
    candidate = jnp.asarray(
        [1.31505, 1.05696, 0.94729, -0.08072, -17.5004],
        dtype=jnp.float64,
    )
    info = parallel_characteristic_wall_data(
        center, center, center, 0.0806799, 0.0378996, 1.0, 1836.0,
        backward_wall=True, backward_wall_state=candidate,
    )
    characteristic = float(info["backward_wall_characteristic_current"])
    nonlinear = float(info["backward_wall_projected_nonlinear_current"])
    remainder = float(info["backward_wall_current_quadratic_remainder"])
    assert abs(characteristic) < 100.0
    assert abs(nonlinear) < 100.0
    assert abs(remainder) < 100.0
    assert float(info["backward_wall_correction_amplification"]) <= 1.0 + 1e-10
    assert bool(jnp.all(info["backward_wall_projected_state"][:3] > 0.0))
    assert nonlinear == pytest.approx(characteristic + remainder)


def test_characteristic_wall_current_matches_nonlinear_current_to_first_order():
    center = _state()
    direction = jnp.asarray([0.2, -0.3, 0.1, 0.4, -0.25])

    def error(scale):
        candidate = center + scale * direction
        info = parallel_characteristic_wall_data(
            center, center, center, 1.0, 1.0, 4.0, 10.0,
            backward_wall=True, backward_wall_state=candidate,
        )
        return abs(float(info["backward_wall_current_quadratic_remainder"]))

    coarse = error(1.0e-3)
    fine = error(5.0e-4)
    same = parallel_characteristic_wall_data(
        center, center, center, 1.0, 1.0, 4.0, 10.0,
        backward_wall=True, backward_wall_state=center,
    )
    np.testing.assert_array_equal(
        same["backward_wall_characteristic_current"],
        center[0] * (center[3] - center[4]),
    )
    assert coarse > 0.0
    # Halving a perturbation must quarter the omitted second-order product.
    assert fine / coarse == pytest.approx(0.25, rel=5.0e-6, abs=1.0e-12)


def test_characteristic_wall_data_keeps_ordinary_mapped_endpoints():
    center = _state()
    minus = center + jnp.asarray([-0.1, 0.2, -0.1, 0.3, -0.2])
    plus = center + jnp.asarray([0.2, -0.1, 0.3, -0.2, 0.1])
    info = parallel_characteristic_wall_data(
        center, minus, plus, 2.0, 3.0, 4.0, 10.0, selection_dt=0.0,
    )
    np.testing.assert_array_equal(info["backward_endpoint_state"], minus)
    np.testing.assert_array_equal(info["forward_endpoint_state"], plus)
    np.testing.assert_array_equal(
        info["backward_endpoint_current"],
        minus[0] * (minus[3] - minus[4]),
    )
    np.testing.assert_array_equal(
        info["forward_endpoint_current"],
        plus[0] * (plus[3] - plus[4]),
    )
    assert not bool(info["backward_wall"])
    assert not bool(info["forward_wall"])


def test_characteristic_wall_data_invalid_candidate_is_reported_and_propagates():
    center = jnp.broadcast_to(_state(), (2, 5))
    bad = center.at[1, 0].set(jnp.nan)
    result = jax.jit(parallel_characteristic_wall_data)(
        center, center, center, jnp.asarray([0.01, 0.01]),
        jnp.asarray([0.02, 0.02]), 4.0, 10.0, selection_dt=0.01,
        backward_wall=jnp.asarray([True, True]), backward_wall_state=bad,
    )
    info = result
    assert bool(jnp.all(jnp.isfinite(info["selected_residual"][0])))
    assert bool(jnp.all(jnp.isfinite(info["selected_jacobian"][0])))
    assert bool(jnp.all(jnp.isfinite(info["backward_projected_state"][0])))
    assert bool(info["backward_candidate_fallback"][1])
    assert bool(info["backward_wall_solve_fallback"][1])
    assert not bool(jnp.all(jnp.isfinite(info["backward_projected_state"][1])))


def test_failed_two_wall_hotspot_is_admissible_and_parallel_flux_is_restoring():
    center = jnp.asarray(
        [0.3644197911, 0.6428237257, 2.5541440404, -0.0804481294, -71.3303178127]
    )
    backward = jnp.asarray(
        [1.25963548, 1.04422820, 0.97527152, -0.04519514, -17.86656166]
    )
    forward = jnp.asarray(
        [0.64737824, 0.78424595, 1.86109526, -0.04614816, -47.15524079]
    )
    residual, info = parallel_target_row_material_residual(
        center, backward, forward, 0.08067992, 0.03789958, 1.0, 1836.0,
        backward_wall=True, forward_wall=True,
        backward_wall_state=backward, forward_wall_state=forward,
    )
    wall = parallel_characteristic_wall_data(
        center, backward, forward, 0.08067992, 0.03789958, 1.0, 1836.0,
        backward_wall=True, forward_wall=True,
        backward_wall_state=backward, forward_wall_state=forward,
    )
    assert bool(jnp.all(wall["backward_endpoint_state"][:3] > 0.0))
    assert bool(jnp.all(wall["forward_endpoint_state"][:3] > 0.0))
    assert abs(float(wall["backward_endpoint_state"][3])) < 100.0
    assert abs(float(wall["forward_endpoint_state"][3])) < 100.0
    assert float(center[4] * residual[4]) < 0.0
    assert bool(info["admissible"])


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


def test_energy_absorbing_wall_ignores_scalar_candidates_and_preserves_ordinary_endpoints():
    center = _state()
    equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0))
    first = parallel_characteristic_wall_data(
        center, center + 0.2, center - 0.1, 0.2, 0.3, 4.0, 10.0,
        backward_wall=True, forward_wall=True,
        backward_wall_state=jnp.asarray((9.0, 8.0, 7.0, 6.0, 5.0)),
        forward_wall_state=jnp.asarray((-9.0, -8.0, -7.0, -6.0, -5.0)),
        equilibrium=equilibrium,
        parallel_characteristic_wall_law="energy-absorbing",
    )
    second = parallel_characteristic_wall_data(
        center, center + 0.2, center - 0.1, 0.2, 0.3, 4.0, 10.0,
        backward_wall=True, forward_wall=True,
        backward_wall_state=jnp.full((5,), jnp.nan),
        forward_wall_state=jnp.full((5,), jnp.nan),
        equilibrium=equilibrium,
        parallel_characteristic_wall_law="energy-absorbing",
    )
    for name in ("backward_endpoint_state", "forward_endpoint_state",
                 "backward_wall_characteristic_current",
                 "forward_wall_characteristic_current"):
        np.testing.assert_allclose(first[name], second[name])
    assert bool(jnp.all(~second["backward_candidate_fallback"]))
    assert bool(jnp.all(~second["forward_candidate_fallback"]))

    ordinary = parallel_characteristic_wall_data(
        center, center + 0.2, center - 0.1, 0.2, 0.3, 4.0, 10.0,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law="energy-absorbing",
    )
    np.testing.assert_array_equal(ordinary["backward_endpoint_state"], center + 0.2)
    np.testing.assert_array_equal(ordinary["forward_endpoint_state"], center - 0.1)


def test_energy_absorbing_wall_is_jittable_and_invalid_selector_is_explicit():
    center = jnp.broadcast_to(_state(), (2, 5))
    run = jax.jit(
        parallel_target_row_material_residual,
        static_argnames=("parallel_characteristic_wall_law",),
    )
    residual, info = run(
        center, center, center, jnp.asarray((0.2, 0.3)),
        jnp.asarray((0.3, 0.2)), 4.0, 10.0,
        backward_wall=jnp.asarray((True, False)),
        forward_wall=jnp.asarray((True, True)),
        parallel_characteristic_wall_law="energy-absorbing",
    )
    assert residual.shape == (2, 5)
    assert bool(jnp.all(jnp.isfinite(residual)))
    with pytest.raises(ValueError, match="parallel_characteristic_wall_law"):
        parallel_target_row_material_residual(
            _state(), _state(), _state(), 0.2, 0.3, 4.0, 10.0,
        parallel_characteristic_wall_law="bad-law",
        )


def test_default_wall_law_is_exactly_explicit_primitive_least_residual():
    center = _state()
    minus = center + jnp.asarray([-0.1, 0.2, -0.1, 0.3, -0.2])
    plus = center + jnp.asarray([0.2, -0.1, 0.3, -0.2, 0.1])
    kwargs = dict(
        backward_wall=True,
        forward_wall=True,
        backward_wall_state=center + 0.1,
        forward_wall_state=center - 0.15,
        equilibrium=jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0)),
        div_b=0.0,
    )
    default_residual, default_info = parallel_target_row_material_residual(
        center, minus, plus, 0.2, 0.3, 4.0, 10.0, **kwargs
    )
    explicit_residual, explicit_info = parallel_target_row_material_residual(
        center, minus, plus, 0.2, 0.3, 4.0, 10.0,
        parallel_characteristic_wall_law="primitive-least-residual", **kwargs
    )
    np.testing.assert_array_equal(default_residual, explicit_residual)
    assert default_info.keys() == explicit_info.keys()
    for name in default_info:
        np.testing.assert_array_equal(default_info[name], explicit_info[name])


def test_primitive_all_wall_uses_operator_trace_not_equilibrium_reference():
    center = _state()
    minus = center + jnp.asarray((-0.08, 0.04, -0.03, 0.12, -0.06))
    plus = center + jnp.asarray((0.05, -0.02, 0.07, -0.09, 0.11))
    common = dict(
        backward_wall=True,
        forward_wall=True,
        backward_wall_state=minus,
        forward_wall_state=plus,
        parallel_short_leg_selection="all-physical-walls",
        parallel_characteristic_wall_law="primitive-least-residual",
    )
    first = parallel_characteristic_wall_data(
        center, minus, plus, 0.2, 0.3, 4.0, 10.0,
        equilibrium=jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0)),
        **common,
    )
    second = parallel_characteristic_wall_data(
        center, minus, plus, 0.2, 0.3, 4.0, 10.0,
        equilibrium=jnp.asarray((2.0, 1.5, 0.7, 0.4, -0.3)),
        **common,
    )
    for name in (
        "backward_endpoint_state",
        "forward_endpoint_state",
        "backward_wall_characteristic_current",
        "forward_wall_characteristic_current",
        "selected_residual",
        "selected_jacobian",
    ):
        np.testing.assert_allclose(first[name], second[name], rtol=0.0, atol=0.0)
    assert not bool(first["backward_candidate_ignored"])
    assert not bool(first["forward_candidate_ignored"])
    assert bool(first["selected_backward_wall"])
    assert bool(first["selected_forward_wall"])


def test_energy_absorbing_omitted_reference_is_normalized_equilibrium():
    center = _state()
    equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0))
    kwargs = dict(
        backward_wall=True,
        forward_wall=True,
        backward_wall_state=jnp.full((5,), jnp.nan),
        forward_wall_state=jnp.full((5,), jnp.nan),
        parallel_characteristic_wall_law="energy-absorbing",
    )
    omitted = parallel_characteristic_wall_data(
        center, center, center, 1.0, 1.0, 4.0, 10.0, **kwargs
    )
    explicit = parallel_characteristic_wall_data(
        center, center, center, 1.0, 1.0, 4.0, 10.0,
        equilibrium=equilibrium, **kwargs
    )
    for name in (
        "backward_endpoint_state", "forward_endpoint_state",
        "backward_wall_characteristic_current", "forward_wall_characteristic_current",
    ):
        np.testing.assert_array_equal(omitted[name], explicit[name])


@pytest.mark.parametrize(
    ("backward_wall", "forward_wall"),
    ((True, False), (False, True), (True, True)),
)
def test_energy_absorbing_wall_residual_is_dissipative_in_live_modal_energy(
    backward_wall, forward_wall
):
    equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0))
    center = equilibrium + jnp.asarray((0.08, -0.02, 0.03, 0.0, 0.0))
    matrix = np.asarray(
        parallel_characteristic_matrix(*center, tau=4.0, mu=10.0)
    )
    eigenvalues, right = np.linalg.eig(matrix)
    assert np.max(np.abs(np.imag(eigenvalues))) < 1.0e-10
    right = np.real(right)
    left = np.linalg.inv(right)
    H = left.T @ left
    assert np.all(np.linalg.eigvalsh(H) > 0.0)
    np.testing.assert_allclose(H @ matrix, (H @ matrix).T, atol=2.0e-10, rtol=2.0e-10)

    residual, info = parallel_target_row_material_residual(
        center, center, center, 1.0, 1.0, 4.0, 10.0,
        backward_wall=backward_wall,
        forward_wall=forward_wall,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law="energy-absorbing",
        div_b=0.0,
    )
    perturbation = np.asarray(center - equilibrium)
    residual = np.asarray(residual)
    rate = float(perturbation @ H @ residual)
    tolerance = 2.0e-10 * max(1.0, np.linalg.norm(perturbation @ H) * np.linalg.norm(residual))
    assert rate <= tolerance
    assert bool(info["admissible"])

    # The lower and upper physical boundaries retain only outward modes, so
    # their oriented modal boundary powers cannot be negative.
    wall_info = parallel_characteristic_wall_data(
        center, center, center, 1.0, 1.0, 4.0, 10.0,
        backward_wall=backward_wall,
        forward_wall=forward_wall,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law="energy-absorbing",
    )
    if backward_wall:
        assert float(wall_info["backward_wall_boundary_power_after"]) >= -tolerance
    if forward_wall:
        assert float(wall_info["forward_wall_boundary_power_after"]) >= -tolerance


@pytest.mark.parametrize(("direction", "normal"), (("backward", -1.0), ("forward", 1.0)))
def test_energy_absorbing_endpoint_preserves_outgoing_and_stationary_amplitudes(
    direction, normal
):
    equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0))
    center = equilibrium + jnp.asarray((0.08, -0.02, 0.03, 0.0, 0.0))
    matrix = np.asarray(
        parallel_characteristic_matrix(*center, tau=4.0, mu=10.0)
    )
    values, right = np.linalg.eig(matrix)
    values = np.real(values)
    left = np.linalg.inv(np.real(right))
    wall_info = parallel_characteristic_wall_data(
        center, center, center, 1.0, 1.0, 4.0, 10.0,
        backward_wall=direction == "backward",
        forward_wall=direction == "forward",
        backward_wall_state=jnp.full((5,), jnp.nan),
        forward_wall_state=jnp.full((5,), jnp.nan),
        equilibrium=equilibrium,
        parallel_characteristic_wall_law="energy-absorbing",
    )
    before = left @ (np.asarray(center) - np.asarray(equilibrium))
    after = left @ np.asarray(wall_info[f"{direction}_endpoint_state"] - equilibrium)
    incoming = normal * values < -1.0e-10
    np.testing.assert_allclose(after[incoming], 0.0, atol=2.0e-9, rtol=0.0)
    np.testing.assert_allclose(after[~incoming], before[~incoming], atol=2.0e-9, rtol=2.0e-9)
    assert bool(wall_info[f"{direction}_candidate_ignored"])
    assert not bool(wall_info[f"{direction}_candidate_finite"])


def test_energy_absorbing_current_uses_the_same_projected_wall_state():
    equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0))
    center = equilibrium + jnp.asarray((0.08, -0.02, 0.03, 0.0, 0.0))
    wall_info = parallel_characteristic_wall_data(
        center, center, center, 1.0, 1.0, 4.0, 10.0,
        backward_wall=True,
        forward_wall=True,
        backward_wall_state=jnp.full((5,), jnp.nan),
        forward_wall_state=jnp.full((5,), jnp.nan),
        equilibrium=equilibrium,
        parallel_characteristic_wall_law="energy-absorbing",
    )
    center_np = np.asarray(center)
    center_current = center_np[0] * (center_np[3] - center_np[4])
    for direction in ("backward", "forward"):
        endpoint = np.asarray(wall_info[f"{direction}_endpoint_state"])
        delta = endpoint - center_np
        expected = (
            center_current
            + (center_np[3] - center_np[4]) * delta[0]
            + center_np[0] * (delta[3] - delta[4])
        )
        np.testing.assert_allclose(
            wall_info[f"{direction}_wall_characteristic_current"], expected,
            rtol=2.0e-12, atol=2.0e-12,
        )
        np.testing.assert_allclose(
            wall_info[f"{direction}_incoming_action"], endpoint - center_np,
            rtol=2.0e-12, atol=2.0e-12,
        )


def test_energy_absorbing_invalid_reference_propagates_nan_and_is_inadmissible():
    equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0))
    invalid = equilibrium.at[0].set(jnp.nan)
    center = equilibrium + jnp.asarray((0.08, -0.02, 0.03, 0.0, 0.0))
    residual, diagnostics = parallel_target_row_material_residual(
        center, center, center, 1.0, 1.0, 4.0, 10.0,
        backward_wall=True,
        equilibrium=invalid,
        parallel_characteristic_wall_law="energy-absorbing",
        div_b=0.0,
    )
    assert bool(jnp.any(~jnp.isfinite(residual)))
    assert not bool(diagnostics["admissible"])
    wall_info = parallel_characteristic_wall_data(
        center, center, center, 1.0, 1.0, 4.0, 10.0,
        backward_wall=True,
        equilibrium=invalid,
        parallel_characteristic_wall_law="energy-absorbing",
    )
    assert bool(jnp.any(~jnp.isfinite(wall_info["backward_endpoint_state"])))
    assert bool(wall_info["backward_wall_solve_fallback"])
    assert not bool(wall_info["backward_wall_thermodynamic_admissible"])


def test_energy_absorbing_short_wall_selector_threads_through_backward_euler():
    equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0))
    center = equilibrium + jnp.asarray((0.08, -0.02, 0.03, 0.0, 0.0))
    nan_candidate = jnp.full((5,), jnp.nan)
    selected, jacobian, info = parallel_short_wall_material_data(
        center, center, center, 0.01, 0.02, 4.0, 10.0,
        selection_dt=0.02, cfl_limit=2.785,
        backward_wall=True, forward_wall=True,
        backward_wall_state=nan_candidate, forward_wall_state=nan_candidate,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law="energy-absorbing",
    )
    assert bool(info["selected_backward_wall"])
    assert bool(info["selected_forward_wall"])
    assert bool(jnp.all(jnp.isfinite(selected)))
    assert bool(jnp.all(jnp.isfinite(jacobian)))

    updated, delta, euler_info = parallel_short_wall_backward_euler(
        center, center, center, 0.01, 0.02, 4.0, 10.0,
        selection_dt=0.02, solve_dt=0.02, cfl_limit=2.785,
        backward_wall=True, forward_wall=True,
        backward_wall_state=nan_candidate, forward_wall_state=nan_candidate,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law="energy-absorbing",
    )
    expected = np.linalg.solve(
        np.eye(5) - 0.02 * np.asarray(euler_info["selected_jacobian"]),
        0.02 * np.asarray(selected),
    )
    np.testing.assert_allclose(delta, expected, rtol=2.0e-9, atol=2.0e-10)
    np.testing.assert_allclose(updated, np.asarray(center) + expected)
    assert bool(euler_info["selected_wall"])


def test_energy_absorbing_ignores_nan_wall_traces_but_not_ordinary_endpoints():
    equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0, 0.0))
    center = equilibrium + jnp.asarray((0.08, -0.02, 0.03, 0.0, 0.0))
    nan_minus = center.at[0].set(jnp.nan)
    nan_plus = center.at[1].set(jnp.nan)
    nan_candidate = jnp.full((5,), jnp.nan)
    residual, info = parallel_target_row_material_residual(
        center, nan_minus, nan_plus, 1.0, 1.0, 4.0, 10.0,
        backward_wall=True, forward_wall=True,
        backward_wall_state=nan_candidate, forward_wall_state=nan_candidate,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law="energy-absorbing",
        div_b=0.0,
    )
    assert bool(jnp.all(jnp.isfinite(residual)))
    assert bool(info["backward_candidate_ignored"])
    assert bool(info["forward_candidate_ignored"])
    assert not bool(info["backward_candidate_finite"])
    assert not bool(info["forward_candidate_finite"])
    for name in (
        "backward_clipped", "forward_clipped", "positivity_fallback",
        "backward_candidate_fallback", "forward_candidate_fallback",
        "fallback",
    ):
        assert not bool(info[name])
    assert bool(info["admissible"])

    ordinary_residual, ordinary_info = parallel_target_row_material_residual(
        center, nan_minus, center, 1.0, 1.0, 4.0, 10.0,
        equilibrium=equilibrium,
        parallel_characteristic_wall_law="energy-absorbing",
        div_b=0.0,
    )
    assert bool(jnp.any(~jnp.isfinite(ordinary_residual)))
    assert bool(ordinary_info["backward_clipped"])
    assert bool(ordinary_info["fallback"])
    assert not bool(ordinary_info["admissible"])
