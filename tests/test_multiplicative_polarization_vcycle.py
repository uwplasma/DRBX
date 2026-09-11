"""Focused tests for the opt-in fixed-cost polarization V-cycle."""
import jax.numpy as jnp
import numpy as np

from drbx.native.fci_boundary_imex_preconditioner import (
    build_multiplicative_polarization_vcycle, build_augmented_polarization_inverse)


def test_vcycle_order_sign_and_fixed_callback_counts():
    calls = []
    def op(x):
        calls.append("operator")
        return 2.0 * x
    def smooth(x):
        calls.append("smooth")
        return 0.25 * x
    def coarse(r):
        calls.append("coarse")
        return 0.5 * r
    cycle = build_multiplicative_polarization_vcycle(op, smooth, coarse,
                                                     active_owned=jnp.array([True, False]))
    result = np.asarray(cycle(jnp.array([4.0, 9.0])))
    # inactive lane is projected; pre -> residual -> coarse -> residual -> post
    np.testing.assert_allclose(result, [2.0, 0.0])
    assert calls == ["smooth", "operator", "coarse", "operator", "smooth"]


def test_vcycle_beats_additive_jacobi_coarse_on_two_mode_system():
    diagonal = jnp.array([1.0, 20.0])
    op = lambda x: diagonal * x
    jacobi = lambda x: jnp.array([x[0], 0.05 * x[1]])
    coarse = lambda r: jnp.array([0.0, 0.0475 * r[1]])
    additive = lambda r: jacobi(r) + coarse(r)
    cycle = build_multiplicative_polarization_vcycle(op, jacobi, coarse)
    rhs = jnp.array([1.0, 1.0])
    add_residual = np.linalg.norm(np.asarray(rhs - op(additive(rhs))))
    cycle_residual = np.linalg.norm(np.asarray(rhs - op(cycle(rhs))))
    assert cycle_residual < add_residual


def test_vcycle_applies_weighted_quotient_projector_to_all_lanes():
    weights = jnp.array([1.0, 3.0])
    def quotient(x):
        return x - jnp.sum(weights * x) / jnp.sum(weights)
    calls = {"smooth": 0, "coarse": 0}
    def smooth(x):
        calls["smooth"] += 1
        return x
    def coarse(x):
        calls["coarse"] += 1
        return x
    cycle = build_multiplicative_polarization_vcycle(lambda x: x, smooth, coarse,
                                                     projector=quotient)
    result = np.asarray(cycle(jnp.array([2.0, 2.0])))
    np.testing.assert_allclose(result, 0.0, atol=1e-12)
    assert calls == {"smooth": 2, "coarse": 1}


def test_packed_augmented_adapter_separates_operator_and_gauge_weights():
    mass = jnp.array([1.0, 3.0]); gauge = jnp.array([3.0, 1.0])
    adapter = build_augmented_polarization_inverse(lambda phi: phi, mass, gauge)
    packed = jnp.zeros(15).at[12:14].set(jnp.array([2.0, 4.0])).at[14].set(.5)
    result = np.asarray(adapter(packed))
    np.testing.assert_allclose(result[:12], 0.0)
    np.testing.assert_allclose(result[12:14], [0.0, 2.0])
    np.testing.assert_allclose(result[14], 3.5)
    # A multidimensional phi is still passed to the phi-only callback, never packed.
    seen = []
    adapter3 = build_augmented_polarization_inverse(lambda phi: seen.append(phi.shape) or phi,
                                                     jnp.ones((1, 1, 2)), jnp.ones((1, 1, 2)))
    adapter3(jnp.zeros(15))
    assert seen == [(1, 1, 2)]


def test_two_cycle_is_fixed_second_defect_correction():
    calls = []
    op = lambda x: calls.append("operator") or 2.0 * x
    smooth = lambda x: calls.append("smooth") or 0.25 * x
    coarse = lambda x: calls.append("coarse") or 0.5 * x
    cycle = build_multiplicative_polarization_vcycle(op, smooth, coarse, applications=2)
    cycle(jnp.array([1.0]))
    assert calls == ["smooth", "operator", "coarse", "operator", "smooth",
                     "operator", "smooth", "operator", "coarse", "operator", "smooth"]
