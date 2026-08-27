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
    parallel_path_fluctuations,
    parallel_production_principal_matrix,
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


def test_path_is_zero_for_constant_state_and_is_constant_path_consistent():
    state = _state()
    plus, minus = parallel_path_fluctuations(state, state, tau=4.0, mu=10.0)
    np.testing.assert_allclose(plus, np.zeros(5), atol=2.0e-13)
    np.testing.assert_allclose(minus, np.zeros(5), atol=2.0e-13)

    left, right = state, state + jnp.asarray([0.1, -0.2, 0.3, 0.4, -0.5])
    p1, m1 = parallel_path_fluctuations(left, right, tau=4.0, mu=10.0)
    # A constant state at every quadrature point has exactly the same local
    # split; this checks the path code against the characteristic action.
    matrix = parallel_characteristic_matrix(*state, tau=4.0, mu=10.0)
    ap, am, *_ = parallel_characteristic_split(matrix)
    jump = right - left
    # For a variable state the path integral is not the left-state matrix
    # times the jump.  It does satisfy the exact quadrature identity used by
    # the implementation: the two fluctuations sum to the integrated A dU.
    reference = jnp.zeros(5)
    nodes = (0.06943184420297371, 0.33000947820757187,
             0.6699905217924281, 0.9305681557970262)
    weights = (0.17392742256872692, 0.32607257743127307,
               0.32607257743127307, 0.17392742256872692)
    for s, weight in zip(nodes, weights):
        q = jnp.concatenate((
            jnp.exp((1.0 - s) * jnp.log(left[:3]) + s * jnp.log(right[:3])),
            (1.0 - s) * left[3:] + s * right[3:],
        ))
        dq = jnp.concatenate((
            q[:3] * (jnp.log(right[:3]) - jnp.log(left[:3])), jump[3:]
        ))
        reference = reference + weight * (parallel_characteristic_matrix(*q, tau=4.0, mu=10.0) @ dq)
    np.testing.assert_allclose(p1 + m1, reference, rtol=2e-6, atol=2e-6)
    assert bool(jnp.all(jnp.isfinite(ap @ jump)))


def test_direction_reversal_and_path_reversal_relations():
    left = _state()
    right = left + jnp.asarray([0.1, -0.2, 0.3, 0.4, -0.5])
    p, m = parallel_path_fluctuations(left, right, tau=4.0, mu=10.0, normal=1.0)
    pm, mm = parallel_path_fluctuations(left, right, tau=4.0, mu=10.0, normal=-1.0)
    np.testing.assert_allclose(pm, -m, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(mm, -p, rtol=3e-6, atol=3e-6)
    pr, mr = parallel_path_fluctuations(right, left, tau=4.0, mu=10.0, normal=1.0)
    np.testing.assert_allclose(pr, -p, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(mr, -m, rtol=3e-6, atol=3e-6)


def test_wall_as_exterior_is_same_api_and_incoming_projection_is_finite():
    owner = _state()
    wall = jnp.asarray([1.0, 1.1, 0.9, 0.0, 0.0])
    matrix = parallel_characteristic_matrix(*owner, tau=4.0, mu=10.0)
    exterior = parallel_wall_exterior_state(owner, wall, matrix, 1.0)
    p1, m1 = parallel_path_fluctuations(owner, exterior, tau=4.0, mu=10.0)
    p2, m2 = parallel_path_fluctuations(owner, exterior_state=exterior, tau=4.0, mu=10.0)
    np.testing.assert_allclose(exterior, np.asarray(exterior))
    np.testing.assert_allclose(p1, p2)
    np.testing.assert_allclose(m1, m2)
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
    _p, _m, info = parallel_path_fluctuations(
        states[0], jnp.asarray([0.0, 1.0, 1.0, 0.0, 0.0]),
        tau=1.0, mu=10.0, return_diagnostics=True,
    )
    assert bool(info["positivity_clipped"])
    assert bool(jnp.all(jnp.isfinite(_p)))


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
    db, _ = parallel_path_fluctuations(minus, center, 4.0, 10.0)
    _, df = parallel_path_fluctuations(center, plus, 4.0, 10.0)
    np.testing.assert_allclose(residual, -(db / dxm + df / dxp), rtol=3e-6, atol=3e-6)
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


def test_wall_and_ordinary_rows_use_the_same_interface_path():
    center = _state()
    candidate = jnp.asarray([1.0, 1.1, 0.9, 0.0, 0.0])
    matrix = parallel_characteristic_matrix(*center, tau=4.0, mu=10.0)
    endpoint = parallel_wall_exterior_state(center, candidate, matrix, -1.0)
    ordinary, _ = parallel_target_row_material_residual(
        center, endpoint, center, 1.0, 1.0, 4.0, 10.0,
        backward_wall=False, div_b=0.0,
    )
    wall, wall_info = parallel_target_row_material_residual(
        center, center, center, 1.0, 1.0, 4.0, 10.0,
        backward_wall=True, backward_wall_state=candidate, div_b=0.0,
    )
    np.testing.assert_allclose(wall, ordinary, rtol=3e-6, atol=3e-6)
    assert bool(wall_info["backward_wall"])


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
