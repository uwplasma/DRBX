"""``small_batched_eig`` agrees with LAPACK on curvature-form matrices."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from drbx.fci_braginskii.native.fci_curvature_production_flux import curvature_principal_matrix  # noqa: E402
from drbx.fci_braginskii.native.fci_helpers import small_batched_eig, small_batched_inv  # noqa: E402


def _abs_action(values, vectors, x):
    projected = np.einsum("nij,nj->ni", np.linalg.inv(vectors), x)
    return np.einsum("nij,nj->ni", vectors, np.abs(values) * projected).real


def test_curvature_matrices_match_lapack():
    rng = np.random.default_rng(0)
    n = 20000
    u = lambda lo, hi: rng.uniform(lo, hi, n)
    matrix = np.asarray(curvature_principal_matrix(u(0.05, 3), u(0.01, 3), u(0.01, 3), u(0.5, 2), u(0.3, 2)))
    values, vectors = (np.asarray(v) for v in jax.jit(small_batched_eig)(matrix))
    np.testing.assert_allclose(
        np.einsum("nij,njk->nik", matrix, vectors), vectors * values[:, None, :], atol=1e-13
    )
    ref_values, ref_vectors = np.linalg.eig(matrix)
    x = rng.standard_normal((n, 4))
    got, want = _abs_action(values, vectors, x), _abs_action(ref_values, ref_vectors, x)
    assert np.max(np.abs(got - want) / np.abs(want).max(1, keepdims=True)) < 1e-12


def test_complex_spectra_are_flagged_like_lapack():
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((20000, 4, 4))
    matrix[..., 3] = 0.0
    values = np.asarray(jax.jit(small_batched_eig)(matrix)[0])
    ref = np.linalg.eig(matrix)[0]
    real = np.abs(ref.imag).max(1) < 1e-10 * (1 + np.abs(ref.real).max(1))
    assert np.array_equal(np.abs(values.imag).max(1) == 0.0, real)
    np.testing.assert_allclose(np.sort(values.real[real], 1), np.sort(ref.real[real], 1), atol=1e-12)


def test_degenerate_scaled_and_nonfinite_matrices():
    base = np.asarray(curvature_principal_matrix(1.0, 0.1, 0.1, 1.0, 0.5))
    scalar = np.diag([2.0, 2.0, 2.0, 0.0])
    scalar[3, :3] = (1.0, 2.0, 3.0)
    cases = [np.zeros((4, 4)), scalar, -base, 1e-160 * base, 1e100 * base, 1e-8 * base]
    for matrix in cases:
        values, vectors = (np.asarray(v) for v in jax.jit(small_batched_eig)(matrix))
        assert np.all(np.isfinite(vectors)) and np.all(values.imag == 0.0)
        np.testing.assert_allclose(
            matrix @ vectors, vectors * values[None, :], atol=1e-13 * np.abs(matrix).max() + 1e-300
        )
        assert abs(np.linalg.det(vectors)) > 1e-3
    for bad in (np.nan, np.inf):
        matrix = base.copy()
        matrix[0, 0] = bad
        values, vectors = (np.asarray(v) for v in jax.jit(small_batched_eig)(matrix))
        assert np.all(values.imag != 0.0) and np.array_equal(vectors, np.eye(4))


def test_inverse_matches_lapack_on_eigenvector_matrices():
    rng = np.random.default_rng(2)
    n = 20000
    u = lambda lo, hi: rng.uniform(lo, hi, n)
    matrix = np.asarray(curvature_principal_matrix(u(0.05, 3), u(0.01, 3), u(0.01, 3), u(0.5, 2), u(0.3, 2)))
    vectors = np.asarray(jax.jit(small_batched_eig)(matrix)[1])
    got, want = np.asarray(jax.jit(small_batched_inv)(vectors)), np.linalg.inv(vectors)
    assert np.max(np.abs(got - want)) < 1e-11 * np.abs(want).max()
    np.testing.assert_array_equal(np.asarray(small_batched_inv(np.eye(4))), np.eye(4))


def test_absolute_action_gradient_matches_finite_differences():
    from drbx.fci_braginskii.native.fci_curvature_production_flux import curvature_characteristic_absolute_action

    matrix = curvature_principal_matrix(1.0, 0.4, 0.3, 1.2, 0.8)
    x = np.asarray((0.3, -0.2, 0.5, 0.1))
    f = lambda v: jnp.sum(curvature_characteristic_absolute_action(matrix, v) ** 2)
    grad = np.asarray(jax.grad(f)(jnp.asarray(x)))
    step = 1.0e-6
    fd = [(f(x + step * e) - f(x - step * e)) / (2 * step) for e in np.eye(4)]
    np.testing.assert_allclose(grad, fd, rtol=1e-7)
