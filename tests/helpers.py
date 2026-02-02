from __future__ import annotations

import numpy as np
from jax.flatten_util import ravel_pytree


def state_to_vec(state) -> np.ndarray:
    """Flatten an eqx.Module/JAX PyTree state into a 1D vector."""

    flat, _ = ravel_pytree(state)
    return np.asarray(flat)


def dense_matrix(matvec, y0) -> np.ndarray:
    """Form the dense Jacobian matrix by applying `matvec` to basis vectors.

    This is intended for small test problems only.
    """

    flat0, unravel = ravel_pytree(y0)
    n = int(flat0.size)
    I = np.eye(n, dtype=np.complex128)
    A = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        w = matvec(unravel(I[:, j]))
        A[:, j] = state_to_vec(w)
    return A


def leading_eig_dense(matvec, y0) -> np.complex128:
    A = dense_matrix(matvec, y0)
    eigs = np.linalg.eigvals(A)
    return np.complex128(eigs[np.argmax(eigs.real)])
