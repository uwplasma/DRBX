"""Focused contract tests for the fixed-cost coupled lower preconditioner."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from drbx.native.fci_boundary_imex_preconditioner import (
    build_fixed_cost_lower_preconditioner,
    build_fixed_cost_material_line_sweeps,
    build_fixed_cost_three_block_lower_preconditioner,
    build_fixed_cost_symmetric_mam_preconditioner,
    build_phi_current_schur_action,
    estimate_hutchinson_owner_diagonal,
    build_fixed_cost_multiplicative_schur_smoother,
    build_augmented_polarization_inverse,
)
from work.boundary_load_audit.compare_physics_leading_schur import analyze_schur


def test_fixed_cost_lower_preconditioner_sign_order_and_lane_projection():
    size = 2
    state = SimpleNamespace(density=jnp.ones(size))
    calls = {"m": 0, "a": 0, "j": 0}
    context = SimpleNamespace(unpack=lambda _: (state, 0.0))
    rng = np.random.default_rng(17)
    M = np.eye(6 * size) + .05 * rng.standard_normal((6 * size, 6 * size))
    A = np.eye(size + 1) + .05 * rng.standard_normal((size + 1, size + 1))
    C = .1 * rng.standard_normal((size + 1, 6 * size))
    J = np.block([[M, np.zeros((6 * size, size + 1))], [C, A]])

    def material_inverse(rhs):
        calls["m"] += 1
        out = np.zeros_like(rhs)
        out[:6 * size] = np.linalg.solve(M, np.asarray(rhs[:6 * size]))
        out[6 * size:] = 91.0  # must be projected by the production API
        return jnp.asarray(out)

    def polarization_inverse(rhs):
        calls["a"] += 1
        out = np.zeros_like(rhs)
        out[6 * size:] = np.linalg.solve(A, np.asarray(rhs[6 * size:]))
        out[:6 * size] = -37.0  # must be projected by the production API
        return jnp.asarray(out)

    def complete_action(dm):
        calls["j"] += 1
        return jnp.asarray(J @ np.asarray(dm))

    preconditioner = build_fixed_cost_lower_preconditioner(
        context, jnp.zeros(J.shape[0]), material_inverse=material_inverse,
        polarization_inverse=polarization_inverse,
        complete_jacobian_action=complete_action)
    rhs = np.arange(1.0, J.shape[0] + 1.0)
    result = np.asarray(preconditioner(rhs))
    expected_m = np.linalg.solve(M, rhs[:6 * size])
    expected_a = np.linalg.solve(A, rhs[6 * size:] - C @ expected_m)
    expected = np.r_[expected_m, expected_a]
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
    assert calls == {"m": 1, "a": 1, "j": 1}


def test_material_line_sweeps_use_complete_action_and_fixed_count():
    size = 2
    state = SimpleNamespace(density=jnp.ones(size))
    context = SimpleNamespace(unpack=lambda _: (state, 0.0),
                              active_owned=jnp.ones(size, dtype=bool))
    matrix = np.eye(6 * size) * 1.1
    matrix += 0.06 * (np.ones((6 * size, 6 * size)) - np.eye(6 * size))
    calls = {"action": 0, "inverse": 0}

    def inverse(value):
        calls["inverse"] += 1
        out = np.zeros_like(np.asarray(value))
        out[:6 * size] = np.asarray(value)[:6 * size] / 1.1
        return jnp.asarray(out)

    def action(value):
        calls["action"] += 1
        return jnp.asarray(np.r_[matrix @ np.asarray(value)[:6 * size],
                                 np.zeros(2 * size + 1)])

    candidate = build_fixed_cost_material_line_sweeps(
        context, jnp.zeros(7 * size + 1), material_inverse=inverse,
        material_action=action, sweeps=2, damping=0.5)
    rhs = np.r_[np.arange(1.0, 6 * size + 1.0), np.ones(2 * size + 1)]
    result = np.asarray(candidate(rhs))
    assert calls == {"action": 2, "inverse": 3}
    initial = np.linalg.norm(rhs[:6 * size])
    final = np.linalg.norm(matrix @ result[:6 * size] - rhs[:6 * size])
    assert final < initial
    assert np.all(result[6 * size:] == 0.0)


def test_three_block_lower_preconditioner_exact_and_two_actions():
    size = 1
    state = SimpleNamespace(density=jnp.ones(size))
    context = SimpleNamespace(unpack=lambda _: (state, 0.0),
                              active_owned=jnp.ones(size, dtype=bool))
    nq, nw, na = 5, 1, 2
    rng = np.random.default_rng(31)
    Mq = np.eye(nq) + .1 * rng.standard_normal((nq, nq))
    Mw = np.array([[1.7]])
    Ma = np.eye(na) + .1 * rng.standard_normal((na, na))
    J = np.block([[Mq, np.zeros((nq, nw)), np.zeros((nq, na))],
                  [.15 * rng.standard_normal((nw, nq)), Mw, np.zeros((nw, na))],
                  [.15 * rng.standard_normal((na, nq)), .15 * rng.standard_normal((na, nw)), Ma]])
    calls = {"action": 0}
    def action(value):
        calls["action"] += 1
        return jnp.asarray(J @ np.asarray(value))
    def inv_q(value):
        out = np.zeros(8); out[:nq] = np.linalg.solve(Mq, np.asarray(value)[:nq]); return jnp.asarray(out)
    def inv_w(value):
        out = np.zeros(8); out[nq] = np.asarray(value)[nq] / Mw[0, 0]; return jnp.asarray(out)
    def inv_a(value):
        out = np.zeros(8); out[nq+nw:] = np.linalg.solve(Ma, np.asarray(value)[nq+nw:]); return jnp.asarray(out)
    pre = build_fixed_cost_three_block_lower_preconditioner(
        context, jnp.zeros(8), primitive_inverse=inv_q,
        vorticity_inverse=inv_w, algebraic_inverse=inv_a,
        complete_jacobian_action=action)
    rhs = rng.standard_normal(8)
    result = np.asarray(pre(rhs))
    np.testing.assert_allclose(J @ result, rhs, rtol=1e-10, atol=1e-10)
    assert calls["action"] == 2


def test_three_block_zero_algebraic_probe_has_no_phi_or_lambda_correction():
    size = 1
    state = SimpleNamespace(density=jnp.ones(size))
    context = SimpleNamespace(unpack=lambda _: (state, 0.0),
                              active_owned=jnp.ones(size, dtype=bool))
    calls = {"action": 0}
    matrix = np.eye(8)
    matrix[6, 0] = 0.4
    matrix[7, 5] = -0.3
    def action(value):
        calls["action"] += 1
        return jnp.asarray(matrix @ np.asarray(value))
    identity = lambda value: jnp.asarray(value)
    zero_algebraic = lambda value: jnp.zeros_like(value)
    pre = build_fixed_cost_three_block_lower_preconditioner(
        context, jnp.zeros(8), primitive_inverse=identity,
        vorticity_inverse=identity, algebraic_inverse=zero_algebraic,
        complete_jacobian_action=action)
    result = np.asarray(pre(np.ones(8)))
    assert np.all(result[6:] == 0.0)
    assert calls["action"] == 2


def test_symmetric_mam_exact_block_ldi_and_four_actions():
    size = 1
    state = SimpleNamespace(density=jnp.ones(size))
    context = SimpleNamespace(unpack=lambda _: (state, 0.0),
                              active_owned=jnp.ones(size, dtype=bool))
    rng = np.random.default_rng(73)
    Mq = np.eye(5) + .08 * rng.standard_normal((5, 5))
    Mw = np.array([[1.4]])
    Ma = np.eye(2) + .08 * rng.standard_normal((2, 2))
    J = np.block([[Mq, np.zeros((5, 1)), np.zeros((5, 2))],
                  [.1 * rng.standard_normal((1, 5)), Mw, np.zeros((1, 2))],
                  [.1 * rng.standard_normal((2, 5)), .1 * rng.standard_normal((2, 1)), Ma]])
    calls = {"action": 0}
    def action(value):
        calls["action"] += 1
        return jnp.asarray(J @ np.asarray(value))
    def qinv(value):
        out = np.zeros(8); out[:5] = np.linalg.solve(Mq, np.asarray(value)[:5]); return jnp.asarray(out)
    def winv(value):
        out = np.zeros(8); out[5] = np.asarray(value)[5] / Mw[0, 0]; return jnp.asarray(out)
    def ainv(value):
        out = np.zeros(8); out[6:] = np.linalg.solve(Ma, np.asarray(value)[6:]); return jnp.asarray(out)
    pre = build_fixed_cost_symmetric_mam_preconditioner(
        context, jnp.zeros(8), primitive_inverse=qinv,
        vorticity_inverse=winv, algebraic_inverse=ainv,
        complete_jacobian_action=action)
    rhs = rng.standard_normal(8)
    result = np.asarray(pre(rhs))
    np.testing.assert_allclose(result, np.linalg.solve(J, rhs), rtol=1e-10, atol=1e-10)
    assert calls["action"] == 4


def test_phi_current_schur_has_derived_subtracted_two_way_sign():
    n = jnp.array([2.0, 4.0]); b = jnp.array([3.0, 5.0])
    action = build_phi_current_schur_action(
        lambda value: jnp.zeros_like(value), lambda value: value,
        lambda value: value, n, b, 0.1, 4.0,
        jnp.ones(2, dtype=bool))
    # dt^2 * mu * B^2 / n * D(n*G(phi)) for phi=(1,1).
    expected = -np.array([0.1**2 * 4.0 * 3.0**2 / 2.0 * 2.0,
                          0.1**2 * 4.0 * 5.0**2 / 4.0 * 4.0])
    np.testing.assert_allclose(action(jnp.ones(2)), expected)


def test_phi_current_schur_preserves_constant_null_and_owner_projection():
    active = jnp.array([True, False, True])
    def gradient(value):
        mean = jnp.sum(jnp.where(active, value, 0.0)) / jnp.sum(active)
        return value - mean
    action = build_phi_current_schur_action(
        lambda value: gradient(value), gradient, lambda value: -value,
        jnp.ones(3), jnp.ones(3), 0.2, 10.0, active,
        projector=lambda value: value - jnp.sum(jnp.where(active, value, 0.0)) / jnp.sum(active))
    result = np.asarray(action(jnp.ones(3)))
    assert np.allclose(result, 0.0)
    assert result[1] == 0.0


def test_dense_physics_schur_comparison_reports_right_preconditioned_residual():
    S = np.diag([2.0, 3.0])
    S0 = np.diag([1.0, 1.5])
    report = analyze_schur(S, S0, np.array([2.0, 3.0]))
    assert report["dofs"] == 2
    assert report["relative_S_minus_S0"] > 0.0
    assert report["true_relative_residual"] < 1.0e-10


def test_phi_current_schur_matches_explicit_three_by_three_elimination():
    # D=1 and G=-1 give the weighted-adjoint, positive-energy orientation.
    n, b, dt, mu = 2.0, 3.0, 0.1, 4.0
    K = b * b / n
    M = np.array([[1.0]])
    B = np.array([[-dt * mu * -1.0]])
    # C acts on Ve through j=n(Vi-Ve), hence the current response carries
    # the minus sign represented by this block.
    C = np.array([[-dt * K * n]])
    A = np.array([[2.0]])
    expected = float(A[0, 0] - (C @ np.linalg.solve(M, B))[0, 0])
    schur = build_phi_current_schur_action(
        lambda value: 2.0 * value, lambda value: -value,
        lambda value: value, n, b, dt, mu, jnp.ones(1))
    np.testing.assert_allclose(np.asarray(schur(jnp.ones(1))), expected)


def test_hutchinson_owner_diagonal_is_deterministic_and_masked():
    active = jnp.array([True, False, True, True])
    matrix = np.diag([2.0, 9.0, 4.0, 8.0])
    action = lambda value: matrix @ np.asarray(value).ravel()
    d1, safe1, info1 = estimate_hutchinson_owner_diagonal(action, active, probe_count=32, seed=4)
    d2, safe2, info2 = estimate_hutchinson_owner_diagonal(action, active, probe_count=32, seed=4)
    np.testing.assert_allclose(d1, d2); np.testing.assert_allclose(safe1, safe2)
    np.testing.assert_allclose(d1, [2.0, 0.0, 4.0, 8.0]); assert info1 == info2


def test_multiplicative_schur_smoother_fixed_action_count_and_mask():
    active = jnp.array([True, True, False]); calls = {"s0": 0, "pa": 0}
    def s0(value): calls["s0"] += 1; return 2.0 * jnp.asarray(value)
    def pa(value): calls["pa"] += 1; return 0.5 * jnp.asarray(value)
    pre = build_fixed_cost_multiplicative_schur_smoother(
        jnp.array([2.0, 4.0, 7.0]), s0, pa, active)
    out = np.asarray(pre(jnp.array([2.0, 8.0, 5.0])))
    assert calls == {"s0": 2, "pa": 1}; assert out[2] == 0.0
    assert np.all(np.isfinite(out))


def test_augmented_phi_inverse_uses_packed_offset_and_gauge():
    active = jnp.array([True, False, True])
    aug = build_augmented_polarization_inverse(
        lambda value: value, jnp.ones(3), jnp.array([1.0, 1.0, 1.0]),
        active_owned=active)
    packed = jnp.zeros(7 * 3 + 1).at[6 * 3:7 * 3].set(jnp.array([2.0, 9.0, 4.0])).at[7 * 3].set(3.0)
    out = np.asarray(aug(packed))
    assert out.shape == (22,)
    assert np.allclose(out[:18], 0.0)
    assert np.allclose(out[18:21], [2.0, 0.0, 4.0])
    assert np.isfinite(out[21])
