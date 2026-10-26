"""Chunked host ``geev`` must be bitwise the single NumPy call."""

from __future__ import annotations

import numpy as np

from drbx.fci_braginskii.native import fci_helpers


def _batch(n: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    matrices = rng.standard_normal((n, size, size))
    # Symmetric halves give real spectra; the rest generically have complex
    # pairs, so chunk boundaries fall between real and complex cases.
    half = n // 2
    matrices[:half] = 0.5 * (matrices[:half] + np.swapaxes(matrices[:half], -1, -2))
    return matrices


def test_host_eig_chunked_matches_single_call_bitwise(monkeypatch):
    monkeypatch.setattr(fci_helpers, "_host_eig_threads", lambda: 4)
    for size in (4, 5):
        matrix = _batch(3 * 4096 + 17, size)
        values, vectors = fci_helpers._host_eig(matrix)
        ref_values, ref_vectors = np.linalg.eig(matrix)
        assert values.dtype == np.complex128 and vectors.dtype == np.complex128
        assert values.shape == matrix.shape[:-1] and vectors.shape == matrix.shape
        np.testing.assert_array_equal(values, ref_values.astype(np.complex128))
        np.testing.assert_array_equal(vectors, ref_vectors.astype(np.complex128))
        assert np.any(ref_values.imag != 0) and np.any(np.all(ref_values.imag == 0, -1))


def test_host_eig_reshapes_leading_batch_dims():
    matrix = _batch(2 * 3 * 1500, 4).reshape((2, 3, 1500, 4, 4))
    values, vectors = fci_helpers._host_eig(matrix)
    ref_values, ref_vectors = np.linalg.eig(matrix)
    np.testing.assert_array_equal(values, ref_values.astype(np.complex128))
    np.testing.assert_array_equal(vectors, ref_vectors.astype(np.complex128))
