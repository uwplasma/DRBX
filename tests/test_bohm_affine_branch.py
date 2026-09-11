"""Focused algebraic checks for the work-only finite Bohm correction."""
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "work/boundary_load_audit"))

from bohm_affine_branch import (
    affine_branch_data, affine_maximum_increment, affine_reduced_constant,
    selected_bohm_maximum_with_shift,
)


def test_affine_offset_reproduces_maximum_on_both_candidate_branches():
    values = (np.array([[3., 1., 2.], [1., 4., 2.], [1., 1., 1.]]),)
    derivatives = (np.array([[-3., 5., 1.], [4., -2., 1.], [0., 0., 0.]]),)
    masks, offsets, rows = affine_branch_data(values, derivatives)
    increment = affine_maximum_increment(values[0], derivatives[0], masks[0], offsets[0])
    cb, outward, _ = values[0]
    dcb, doutward, _ = derivatives[0]
    np.testing.assert_allclose(np.maximum(cb, outward) + increment,
                               np.maximum(cb + dcb, outward + doutward))
    assert rows[0]["changed_from_base"] == 2


def test_base_mask_has_zero_affine_offset():
    values = (np.array([[3., 1., 2.], [1., 4., 2.], [1., 1., 1.]]),)
    zeros = (np.zeros_like(values[0]),)
    masks, offsets, _ = affine_branch_data(values, zeros)
    np.testing.assert_array_equal(masks[0], values[0][0] >= values[0][1])
    np.testing.assert_array_equal(offsets[0], 0.)


def test_inactive_faces_preserve_previous_mask_and_receive_no_offset():
    values = (np.array([[1., 1.], [2., 2.], [1., 0.]]),)
    derivatives = (np.array([[2., 2.], [-2., -2.], [0., 0.]]),)
    previous = (np.array([False, False]),)
    masks, offsets, rows = affine_branch_data(values, derivatives, previous)
    np.testing.assert_array_equal(masks[0], [True, False])
    np.testing.assert_array_equal(offsets[0], [-1., 0.])
    assert rows[0]["changed_from_previous"] == 1


def test_joint_jvp_is_linear_and_auxiliary_shift_does_not_change_zero_shift_primal():
    mask = jnp.array([True, False, False])
    cb = jnp.array([2., 2., 4.])
    outward = jnp.array([1., 3., 4.])

    def action(dcb, doutward, dshift):
        (_value, tangent) = jax.jvp(
            lambda a, b, s: selected_bohm_maximum_with_shift(a, b, mask, s),
            (cb, outward, jnp.zeros_like(cb)), (dcb, doutward, dshift))
        return tangent

    d = (jnp.array([1., 2., 3.]), jnp.array([4., 5., 6.]), jnp.array([7., 8., 9.]))
    e = (jnp.array([2., 1., 0.]), jnp.array([3., 2., 1.]), jnp.array([1., 3., 5.]))
    np.testing.assert_allclose(action(*(2*d[i]-3*e[i] for i in range(3))),
                               2*action(*d)-3*action(*e))
    value = selected_bohm_maximum_with_shift(cb, outward, mask, jnp.zeros_like(cb))
    np.testing.assert_array_equal(value, jnp.maximum(cb, outward))


def test_reduced_constant_includes_saved_algebraic_residual_and_offset():
    A = np.array([[2., 0.], [0., 4.]])
    B = np.array([[3., -1.]])
    r0u, r0z = np.array([5.]), np.array([.2, -.4])
    eu, ez = np.array([.7]), np.array([.6, .8])
    reduced, dz, rhs = affine_reduced_constant(
        r0u, r0z, eu, ez, lambda x: B @ x, lambda x: np.linalg.solve(A, x))
    np.testing.assert_allclose(rhs, r0z + ez)
    np.testing.assert_allclose(dz, -np.linalg.solve(A, r0z + ez))
    np.testing.assert_allclose(reduced, r0u + eu - B @ np.linalg.solve(A, r0z + ez))


def test_crossed_branch_coupled_model_needs_offset_and_has_correct_schur_sign():
    # At x=0 outward wins: max(1+x, 2-x)=2.  The endpoint x=1 enters
    # the cb branch, whose affine intercept relative to the base maximum is -1.
    value = np.array([[1.], [2.], [1.]])
    derivative = np.array([[1.], [-1.], [0.]])
    masks, offsets, _ = affine_branch_data((value,), (derivative,))
    assert masks[0][0]
    np.testing.assert_allclose(offsets[0], [-1.])

    # Synthetic coupled linearization:
    # Ru = r0u + 7*x + 3*z + 5*offset
    # Rz = r0z + 18*x + 4*z + 11*offset.
    # r0u is chosen so the correctly eliminated crossed model solves x=1.
    r0u, r0z = np.array([3.55]), np.array([.4])
    B, A = np.array([[3.]]), np.array([[4.]])
    eu, ez = 5.*offsets[0], 11.*offsets[0]
    constant, dz0, _ = affine_reduced_constant(
        r0u, r0z, eu, ez, lambda z: B @ z, lambda rhs: np.linalg.solve(A, rhs))
    reduced_action = np.array([[7.]]) - B @ np.linalg.solve(A, np.array([[18.]]))
    correction = np.linalg.solve(reduced_action, -constant)
    np.testing.assert_allclose(correction, [1.])
    dz = dz0 - np.linalg.solve(A, np.array([[18.]])) @ correction
    np.testing.assert_allclose(r0z + 18.*correction + 4.*dz + ez, 0., atol=1.e-14)
    np.testing.assert_allclose(r0u + 7.*correction + 3.*dz + eu, 0., atol=1.e-14)

    no_intercept, _, _ = affine_reduced_constant(
        r0u, r0z, np.zeros(1), np.zeros(1),
        lambda z: B @ z, lambda rhs: np.linalg.solve(A, rhs))
    assert not np.allclose(np.linalg.solve(reduced_action, -no_intercept), correction)
