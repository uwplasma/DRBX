"""Contract tests for the standalone production parallel material flux."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.fci_braginskii.native.fci_parallel_production_flux import (
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


def test_physical_boundary_state_is_consumed_directly_across_rank_changes():
    wall = jnp.asarray([1.1, 0.9, 1.2, 0.0, 0.0], dtype=jnp.float64)
    centers = jnp.asarray(
        [
            [1.0, 1.0, 1.0, -2.0, -2.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 2.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    walls = jnp.broadcast_to(wall, centers.shape)
    info = jax.jit(parallel_characteristic_wall_data)(
        centers,
        centers,
        centers,
        jnp.ones(3),
        jnp.ones(3),
        1.0,
        1836.0,
        backward_wall=jnp.ones(3, dtype=bool),
        forward_wall=jnp.ones(3, dtype=bool),
        backward_wall_state=walls,
        forward_wall_state=walls,
    )
    np.testing.assert_allclose(info["backward_endpoint_state"], walls, atol=0.0)
    np.testing.assert_allclose(info["forward_endpoint_state"], walls, atol=0.0)
    np.testing.assert_allclose(
        info["backward_wall_characteristic_current"], 0.0, atol=0.0
    )
    np.testing.assert_allclose(
        info["forward_wall_characteristic_current"], 0.0, atol=0.0
    )
    assert bool(jnp.all(info["backward_wall_solve_valid"]))
    assert bool(jnp.all(info["forward_wall_solve_valid"]))
    assert not bool(jnp.any(info["backward_candidate_ignored"]))
    assert not bool(jnp.any(info["forward_candidate_ignored"]))
    # The incoming rank changes with the live state, but the physical state
    # remains a fixed five-component payload and never gates the boundary.
    assert len(set(np.asarray(info["backward_wall_incoming_count"]).tolist())) > 1
    assert len(set(np.asarray(info["forward_wall_incoming_count"]).tolist())) > 1


def test_physical_boundary_state_rejects_nonfinite_or_nonpositive_trace():
    center = jnp.asarray([1.0, 1.0, 1.0, 0.1, -0.1], dtype=jnp.float64)
    for wall in (
        jnp.asarray([jnp.nan, 1.0, 1.0, 0.0, 0.0]),
        jnp.asarray([-1.0, 1.0, 1.0, 0.0, 0.0]),
    ):
        info = parallel_characteristic_wall_data(
            center,
            center,
            center,
            1.0,
            1.0,
            1.0,
            1836.0,
            backward_wall=True,
            backward_wall_state=wall,
        )
        assert not bool(info["backward_wall_solve_valid"])
        assert bool(jnp.any(jnp.isnan(info["backward_endpoint_state"])))


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
        selection_dt=0.02,
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
    # The physical-boundary-state law accepts the candidate directly as the
    # wall state, so the selected backward action is exactly the live
    # characteristic fluctuation between the owner and that candidate.
    # ``parallel_target_row_material_residual`` cannot supply this value
    # directly any more: every physical-wall direction is always selected
    # for the local backward-Euler split, so its explicit residual always
    # omits it.
    expected = -a_plus @ (center - candidate) / 0.01
    np.testing.assert_allclose(residual, expected, rtol=2e-8, atol=2e-8)


def test_short_wall_backward_euler_matches_frozen_local_solve():
    center = _state()
    candidate = jnp.asarray([1.1, 1.2, 0.9, 0.3, -0.1])
    dt = 0.02
    updated, delta, info = parallel_short_wall_backward_euler(
        center, center, center, 0.01, 1.0, 4.0, 10.0,
        selection_dt=dt, solve_dt=dt,
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


def test_short_wall_backward_euler_uses_the_same_physical_boundary_state():
    center = _state()
    no_flow_wall = jnp.asarray([2.0, 3.0, 5.0, 0.0, 0.0])
    dt = 2.0e-4
    updated, delta, info = parallel_short_wall_backward_euler(
        center,
        center,
        center,
        0.01,
        1.0,
        4.0,
        10.0,
        selection_dt=dt,
        solve_dt=dt,
        backward_wall=True,
        backward_wall_state=no_flow_wall,
    )
    np.testing.assert_allclose(
        info["backward_endpoint_state"], no_flow_wall, atol=0.0, rtol=0.0
    )
    assert bool(info["selected_backward_wall"])
    assert bool(info["backward_wall_solve_valid"])
    assert bool(info["implicit_finite"])
    assert bool(jnp.all(jnp.isfinite(updated)))
    assert bool(jnp.all(jnp.isfinite(delta)))


def test_short_wall_ordinary_rows_are_zero_and_default_residual_is_unchanged():
    center = _state()
    minus = center + jnp.asarray([-0.1, 0.2, -0.1, 0.3, -0.2])
    plus = center + jnp.asarray([0.2, -0.1, 0.3, -0.2, 0.1])
    selected, jacobian, info = parallel_short_wall_material_data(
        center, minus, plus, 1.0, 1.0, 4.0, 10.0,
        selection_dt=0.001,
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


def test_short_wall_batch_jit_selects_every_physical_wall_row():
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
    # Every physical wall row is selected for the local backward-Euler split
    # regardless of the local characteristic CFL, including a vanishingly
    # small selection interval.
    assert bool(info["selected_backward_wall"][0])
    assert bool(info["selected_backward_wall"][1])
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


def test_all_physical_wall_selection_keeps_ordinary_legs_off():
    (
        equilibrium, center, minus, plus, dx_minus, dx_plus,
        backward_wall, forward_wall,
    ) = _energy_wall_selection_batch()
    selected, jacobian, info = parallel_short_wall_material_data(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0,
        selection_dt=1.0e-6,
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=minus, forward_wall_state=plus,
        equilibrium=equilibrium,
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


def test_all_physical_wall_backward_euler_includes_both_walls_once_on_long_legs():
    (
        equilibrium, center, minus, plus, dx_minus, dx_plus,
        backward_wall, forward_wall,
    ) = _energy_wall_selection_batch()
    solve_dt = 1.0e-3
    selected, _, selected_info = parallel_short_wall_material_data(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0,
        selection_dt=1.0e-6,
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=minus, forward_wall_state=plus,
        equilibrium=equilibrium,
    )
    updated, delta, info = parallel_short_wall_backward_euler(
        center, minus, plus, dx_minus, dx_plus, 4.0, 10.0,
        selection_dt=1.0e-6, solve_dt=solve_dt,
        backward_wall=backward_wall, forward_wall=forward_wall,
        backward_wall_state=minus, forward_wall_state=plus,
        equilibrium=equilibrium,
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


def test_characteristic_wall_current_has_no_quadratic_remainder():
    # The physical-boundary-state law exports the actual nonlinear wall
    # current directly (it is not a first-order modal projection), so the
    # quadratic remainder against the characteristic current is exactly
    # zero for every perturbation size.
    center = _state()
    direction = jnp.asarray([0.2, -0.3, 0.1, 0.4, -0.25])

    def remainder(scale):
        candidate = center + scale * direction
        info = parallel_characteristic_wall_data(
            center, center, center, 1.0, 1.0, 4.0, 10.0,
            backward_wall=True, backward_wall_state=candidate,
        )
        return abs(float(info["backward_wall_current_quadratic_remainder"]))

    assert remainder(1.0e-3) == 0.0
    assert remainder(5.0e-4) == 0.0
    same = parallel_characteristic_wall_data(
        center, center, center, 1.0, 1.0, 4.0, 10.0,
        backward_wall=True, backward_wall_state=center,
    )
    np.testing.assert_array_equal(
        same["backward_wall_characteristic_current"],
        center[0] * (center[3] - center[4]),
    )


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


def test_failed_two_wall_hotspot_is_admissible():
    # Rounded values from the late 48^3 wall failure that motivated the
    # local backward-Euler short-leg split: both endpoints are physical
    # walls on a very short mapped leg.
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
    # Both physical-wall directions are always selected for the local
    # backward-Euler split, so the explicit residual carries none of the
    # material action here; the stiff local action lives in the selected
    # wall residual instead, which the implicit split (not this explicit
    # residual) is responsible for damping.
    np.testing.assert_allclose(residual, 0.0, atol=0.0)
    assert bool(jnp.all(jnp.isfinite(wall["selected_residual"])))
    assert bool(jnp.all(jnp.isfinite(wall["selected_jacobian"])))
    assert bool(info["admissible"])


def test_short_wall_backward_euler_linearization_increment_is_a_newton_step():
    center = _state()
    candidate = jnp.asarray([2.0, 3.0, 5.0, 0.0, 0.0])
    dt = 2.0e-4
    kwargs = dict(selection_dt=dt, solve_dt=dt, backward_wall=True,
                  backward_wall_state=candidate)
    _, plain, info = parallel_short_wall_backward_euler(
        center, center, center, 0.01, 1.0, 4.0, 10.0, **kwargs)
    _, zero, _ = parallel_short_wall_backward_euler(
        center, center, center, 0.01, 1.0, 4.0, 10.0,
        linearization_increment=jnp.zeros(5), **kwargs)
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(zero))
    m_k = jnp.asarray([1e-3, -2e-3, 5e-4, 1e-3, -1e-3])
    _, newton, _ = parallel_short_wall_backward_euler(
        center, center, center, 0.01, 1.0, 4.0, 10.0,
        linearization_increment=m_k, **kwargs)
    jac = np.asarray(info["selected_jacobian"])
    expected = np.linalg.solve(
        np.eye(5) - dt * jac,
        dt * (np.asarray(info["backward_residual"]) - jac @ np.asarray(m_k)),
    )
    np.testing.assert_allclose(newton, expected, rtol=2e-9, atol=2e-9)
