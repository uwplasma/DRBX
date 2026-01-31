from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxdrb.models.cold_ion_drb import State


def vec_to_state(v: np.ndarray, nl: int) -> State:
    vv = jnp.asarray(v)
    return State(
        n=vv[:nl],
        omega=vv[nl : 2 * nl],
        vpar_e=vv[2 * nl : 3 * nl],
        vpar_i=vv[3 * nl : 4 * nl],
        Te=vv[4 * nl : 5 * nl],
    )


def state_to_vec(s: State) -> jnp.ndarray:
    return jnp.concatenate([s.n, s.omega, s.vpar_e, s.vpar_i, s.Te])


def dense_matrix(matvec, nl: int) -> np.ndarray:
    """Form the dense Jacobian matrix by applying `matvec` to basis vectors.

    This is intended for small test problems only.
    """

    n = 5 * nl
    I = np.eye(n, dtype=np.complex128)
    A = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        w = matvec(vec_to_state(I[:, j], nl))
        A[:, j] = np.asarray(state_to_vec(w))
    return A


def leading_eig_dense(matvec, nl: int) -> np.complex128:
    A = dense_matrix(matvec, nl)
    eigs = np.linalg.eigvals(A)
    return np.complex128(eigs[np.argmax(eigs.real)])

