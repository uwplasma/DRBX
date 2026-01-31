from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _flatten_pytree(x: Any) -> tuple[np.ndarray, Any]:
    leaves, treedef = jax.tree_util.tree_flatten(x)
    flat = np.concatenate([np.ravel(np.asarray(leaf)) for leaf in leaves]).astype(np.complex128)
    shapes = [np.asarray(leaf).shape for leaf in leaves]
    dtypes = [np.asarray(leaf).dtype for leaf in leaves]
    meta = (treedef, shapes, dtypes, [leaf.size for leaf in leaves])
    return flat, meta


def _unflatten_pytree(v: np.ndarray, meta: Any) -> Any:
    treedef, shapes, dtypes, sizes = meta
    leaves = []
    offset = 0
    for shape, dtype, size in zip(shapes, dtypes, sizes, strict=True):
        chunk = v[offset : offset + size].reshape(shape).astype(dtype)
        leaves.append(jnp.asarray(chunk))
        offset += size
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _inner(x: np.ndarray, y: np.ndarray) -> np.complex128:
    return np.vdot(x, y)


def _norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.real(_inner(x, x))))


@dataclass
class ArnoldiResult:
    eigenvalues: np.ndarray
    residual_norms: np.ndarray
    H: np.ndarray


def arnoldi_eigs(
    matvec,
    v0,
    *,
    m: int = 40,
    nev: int = 6,
    seed: int | None = None,
) -> ArnoldiResult:
    """Basic matrix-free Arnoldi for leading Ritz eigenvalues (largest real part).

    Notes:
      - This is not implicitly restarted; keep `m` modest.
      - `matvec` accepts/returns the same pytree type as `v0`.
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    v0_flat, meta = _flatten_pytree(v0)
    n = v0_flat.size
    if _norm(v0_flat) == 0.0:
        v0_flat = rng.normal(size=n) + 1j * rng.normal(size=n)

    q = np.zeros((m + 1, n), dtype=np.complex128)
    h = np.zeros((m + 1, m), dtype=np.complex128)

    q[0] = v0_flat / _norm(v0_flat)

    k = m
    for j in range(m):
        w = matvec(_unflatten_pytree(q[j], meta))
        w_flat, _ = _flatten_pytree(w)

        # Modified Gram-Schmidt
        for i in range(j + 1):
            hij = _inner(q[i], w_flat)
            h[i, j] = hij
            w_flat = w_flat - hij * q[i]

        # Re-orthogonalize (helps when Krylov dimension grows)
        for i in range(j + 1):
            hij = _inner(q[i], w_flat)
            h[i, j] = h[i, j] + hij
            w_flat = w_flat - hij * q[i]

        hj1 = _norm(w_flat)
        h[j + 1, j] = hj1
        if hj1 == 0.0:
            k = j + 1
            break
        q[j + 1] = w_flat / hj1

    Hk = h[:k, :k]
    evals, evecs = np.linalg.eig(Hk)

    if k == 0:
        raise RuntimeError("Arnoldi produced an empty Krylov space.")

    if k == 1:
        resids = np.zeros((1,), dtype=float)
    else:
        beta = h[k, k - 1] if k < h.shape[0] else 0.0
        resids = np.abs(beta * evecs[-1, :]).astype(float)

    idx = np.argsort(np.real(evals))[::-1]
    evals_out = evals[idx][:nev]
    resids_out = resids[idx][:nev]

    return ArnoldiResult(eigenvalues=evals_out, residual_norms=resids_out, H=Hk)


@dataclass
class ArnoldiLeadingResult:
    eigenvalue: np.complex128
    residual_norm: float
    n_restart: int


def arnoldi_leading_eig(
    matvec,
    v0,
    *,
    m: int = 80,
    n_restart: int = 6,
    seed: int | None = None,
    tol: float = 1e-6,
) -> ArnoldiLeadingResult:
    """Restarted Arnoldi targeting the eigenvalue with largest real part.

    This is a lightweight (explicit) restart strategy: after each Arnoldi cycle we restart from
    the Ritz vector associated with the current largest-real-part Ritz value.
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    v_flat, meta = _flatten_pytree(v0)
    n = v_flat.size
    if _norm(v_flat) == 0.0:
        v_flat = rng.normal(size=n) + 1j * rng.normal(size=n)

    best_eval = np.complex128(0.0 + 0.0j)
    best_resid = float("inf")

    for r in range(n_restart):
        q = np.zeros((m + 1, n), dtype=np.complex128)
        h = np.zeros((m + 1, m), dtype=np.complex128)

        q[0] = v_flat / _norm(v_flat)
        k = m
        for j in range(m):
            w = matvec(_unflatten_pytree(q[j], meta))
            w_flat, _ = _flatten_pytree(w)

            for i in range(j + 1):
                hij = _inner(q[i], w_flat)
                h[i, j] = hij
                w_flat = w_flat - hij * q[i]

            for i in range(j + 1):
                hij = _inner(q[i], w_flat)
                h[i, j] = h[i, j] + hij
                w_flat = w_flat - hij * q[i]

            hj1 = _norm(w_flat)
            h[j + 1, j] = hj1
            if hj1 == 0.0:
                k = j + 1
                break
            q[j + 1] = w_flat / hj1

        Hk = h[:k, :k]
        evals, evecs = np.linalg.eig(Hk)

        if k == 1:
            resids = np.zeros((1,), dtype=float)
        else:
            beta = h[k, k - 1] if k < h.shape[0] else 0.0
            resids = np.abs(beta * evecs[-1, :]).astype(float)

        idx = int(np.argmax(evals.real))
        best_eval = np.complex128(evals[idx])
        best_resid = float(resids[idx])

        # Restart from Ritz vector v = Q_k y
        y = evecs[:, idx]
        v_flat = (q[:k].T @ y).astype(np.complex128)
        nrm = _norm(v_flat)
        if nrm == 0.0:
            v_flat = rng.normal(size=n) + 1j * rng.normal(size=n)
            nrm = _norm(v_flat)
        v_flat = v_flat / nrm

        if best_resid / (abs(best_eval) + 1.0) < tol:
            return ArnoldiLeadingResult(best_eval, best_resid, n_restart=r + 1)

    return ArnoldiLeadingResult(best_eval, best_resid, n_restart=n_restart)


@dataclass
class ArnoldiLeadingVectorResult:
    """Leading-eigenvalue result including a Ritz vector approximation.

    The returned vector is a Ritz vector from the Arnoldi factorization (not an exact eigenvector
    of the full operator). It is typically sufficient for plotting eigenfunction structure.
    """

    eigenvalue: np.complex128
    residual_norm: float
    vector: Any


def arnoldi_leading_ritz_vector(
    matvec,
    v0,
    *,
    m: int = 80,
    seed: int | None = None,
) -> ArnoldiLeadingVectorResult:
    """Return the leading Ritz eigenvalue and its associated Ritz vector.

    This routine runs a single Arnoldi cycle of dimension `m` and constructs the Ritz vector
    v = Q_k y for the Ritz eigenpair of H_k with the largest real part.
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    v0_flat, meta = _flatten_pytree(v0)
    n = v0_flat.size
    if _norm(v0_flat) == 0.0:
        v0_flat = rng.normal(size=n) + 1j * rng.normal(size=n)

    q = np.zeros((m + 1, n), dtype=np.complex128)
    h = np.zeros((m + 1, m), dtype=np.complex128)
    q[0] = v0_flat / _norm(v0_flat)

    k = m
    for j in range(m):
        w = matvec(_unflatten_pytree(q[j], meta))
        w_flat, _ = _flatten_pytree(w)

        for i in range(j + 1):
            hij = _inner(q[i], w_flat)
            h[i, j] = hij
            w_flat = w_flat - hij * q[i]

        for i in range(j + 1):
            hij = _inner(q[i], w_flat)
            h[i, j] = h[i, j] + hij
            w_flat = w_flat - hij * q[i]

        hj1 = _norm(w_flat)
        h[j + 1, j] = hj1
        if hj1 == 0.0:
            k = j + 1
            break
        q[j + 1] = w_flat / hj1

    if k == 0:
        raise RuntimeError("Arnoldi produced an empty Krylov space.")

    Hk = h[:k, :k]
    evals, evecs = np.linalg.eig(Hk)
    idx = int(np.argmax(evals.real))
    lam = np.complex128(evals[idx])

    if k == 1:
        resid = 0.0
    else:
        beta = h[k, k - 1] if k < h.shape[0] else 0.0
        resid = float(np.abs(beta * evecs[-1, idx]))

    y = evecs[:, idx]
    v_flat = (q[:k].T @ y).astype(np.complex128)
    nrm = _norm(v_flat)
    if nrm == 0.0:
        raise RuntimeError("Arnoldi produced a zero Ritz vector.")
    v_flat = v_flat / nrm

    v = _unflatten_pytree(v_flat, meta)
    return ArnoldiLeadingVectorResult(eigenvalue=lam, residual_norm=resid, vector=v)
