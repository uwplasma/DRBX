"""Host-side spectral and energy audits of the linear nodal SBP operator (SciPy).

The JAX operator is assembled into a sparse matrix by probing (index ``k * P + p`` for eta plane ``k`` and node ``p``);
the rest are small ARPACK/SuperLU based diagnostics: the energy identity residual, the numerical abscissa in the
``H``-norm, and residual-gated eigenvalue bands from a Cayley transform and from the rightmost edge.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MAX_UNKNOWNS = 40_000


def probe_colors(n_eta: int, eta_reach: int = 3) -> int:
    """Smallest divisor ``c`` of ``n_eta`` with ``c >= 2 * eta_reach + 1`` (``n_eta`` itself if none)."""
    need = 2 * int(eta_reach) + 1
    for c in range(need, n_eta + 1):
        if n_eta % c == 0:
            return c
    return int(n_eta)


def assemble_by_probing(apply_lin: Callable, P: int, n_eta: int, eta_reach: int = 3, batch: int = 256,
                        allow_large: bool = False) -> sp.csr_matrix:
    """Sparse matrix of the linear map ``apply_lin: (n_eta, P) -> (n_eta, P)``, by colouring.

    Colours are ``(node p, eta residue mod c)`` with ``c = probe_colors(n_eta, eta_reach)``; the response of a probe
    at plane ``k`` is read on planes ``k - reach .. k + reach`` (all planes if ``c == n_eta``). Probes are
    applied through ``jax.vmap`` in batches of ``batch``. Refuses ``P * n_eta > 40_000`` unless ``allow_large``.
    """
    import jax
    import jax.numpy as jnp

    P, n_eta = int(P), int(n_eta)
    if P * n_eta > MAX_UNKNOWNS and not allow_large:
        raise ValueError(f"{P * n_eta} unknowns exceed {MAX_UNKNOWNS}; pass allow_large=True to assemble anyway")
    c = probe_colors(n_eta, eta_reach)
    if c == n_eta:
        offsets = np.arange(n_eta)
    else:
        offsets = np.unique(np.arange(-eta_reach, eta_reach + 1) % n_eta)
        offsets = np.where(offsets > n_eta // 2, offsets - n_eta, offsets)
    colors = [(p, r) for p in range(P) for r in range(c)]
    fn = jax.jit(jax.vmap(apply_lin))
    rows, cols, vals = [], [], []
    for s in range(0, len(colors), batch):
        chunk = colors[s:s + batch]
        probes = np.zeros((len(chunk), n_eta, P))
        for i, (p, r) in enumerate(chunk):
            probes[i, r::c, p] = 1.0
        resp = np.asarray(fn(jnp.asarray(probes)))
        for i, (p, r) in enumerate(chunk):
            for k in range(r, n_eta, c):
                planes = (k + offsets) % n_eta if c < n_eta else np.arange(n_eta)
                block = resp[i][planes]                                  # (len(planes), P)
                kk, qq = np.nonzero(block)
                rows.append(planes[kk] * P + qq)
                cols.append(np.full(kk.size, k * P + p))
                vals.append(block[kk, qq])
    n = n_eta * P
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def energy_identity(T_apply: Callable, Hp, wall_term_fn: Callable, rng: np.random.Generator,
                    scale_apply: Callable | None = None) -> float:
    """Relative residual ``|g^T Hp T g - wall_term(g)| / (|g|_Hp |A g|_Hp)`` for a random ``g (n_eta, P)``.

    ``scale_apply`` is the operator ``A`` of the scale (default: ``T_apply`` itself).
    """
    Hp = np.asarray(Hp)
    g = rng.standard_normal(Hp.shape)
    Tg = np.asarray(T_apply(g))
    Ag = Tg if scale_apply is None else np.asarray(scale_apply(g))
    scale = np.sqrt(np.sum(Hp * g * g)) * np.sqrt(np.sum(Hp * Ag * Ag))
    return float(abs(np.sum(Hp * g * Tg) - wall_term_fn(g)) / scale)


def numerical_abscissa(L, H, dense_below: int = 1500) -> float:
    """Largest eigenvalue of ``(S + S^T) / 2`` with ``S = H^(1/2) L H^(-1/2)``; ``H`` is the diagonal norm (any shape).

    ARPACK ``eigsh`` (``LA``, Krylov dimension 60) above ``dense_below`` unknowns; below that a dense symmetric
    eigensolve, which is exact and avoids ARPACK stalling on the wide spectrum of a small operator.
    """
    h = np.asarray(H, dtype=np.float64).reshape(-1)
    d = sp.diags(np.sqrt(h))
    di = sp.diags(1.0 / np.sqrt(h))
    S = (d @ sp.csr_matrix(L) @ di).tocsr()
    sym = 0.5 * (S + S.T)
    if sym.shape[0] <= max(dense_below, 3):
        return float(np.linalg.eigvalsh(sym.toarray()).max())
    return float(spla.eigsh(sym, k=1, which="LA", ncv=min(sym.shape[0] - 1, 60), maxiter=20000,
                            return_eigenvectors=False)[0])


def _gate(L, values, vectors, tol):
    keep, res = [], []
    for i, lam in enumerate(values):
        x = vectors[:, i]
        r = np.linalg.norm(L @ x - lam * x) / max(np.linalg.norm(L @ x), np.linalg.norm(lam * x), 1e-300)
        res.append(r)
        if r <= tol:
            keep.append(i)
    order = sorted(keep, key=lambda i: -values[i].real)
    return np.array([values[i] for i in order]), np.array([res[i] for i in order])


def cayley_band(L, sigma: float, k: int = 4, tol: float = 1e-8, maxiter: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues of ``L`` nearest the real shift ``sigma`` (in the Cayley sense), from one ``splu``.

    ``mu`` are the largest-magnitude eigenvalues of ``(L - sigma I)^-1 (L + sigma I)`` and
    ``lambda = sigma (mu + 1) / (mu - 1)``. Pairs with relative residual ``|L x - lambda x|`` above ``tol`` are
    dropped. Returns ``(eigenvalues sorted by decreasing real part, residuals)``.
    """
    L = sp.csc_matrix(L)
    n = L.shape[0]
    ident = sp.identity(n, format="csc")
    lu = spla.splu((L - sigma * ident).tocsc())
    Lp = (L + sigma * ident).tocsr()
    op = spla.LinearOperator((n, n), matvec=lambda x: lu.solve(Lp @ x), dtype=np.float64)
    kk = min(k, n - 2)
    try:
        mu, vec = spla.eigs(op, k=kk, which="LM", ncv=min(n - 1, max(4 * kk, 40)), maxiter=maxiter)
    except spla.ArpackNoConvergence as exc:
        mu, vec = exc.eigenvalues, exc.eigenvectors
        if len(mu) == 0:
            return np.array([]), np.array([])
    lam = sigma * (mu + 1.0) / (mu - 1.0)
    return _gate(L, lam, vec, tol)


def edge_band(L, k: int = 10, maxiter: int = 100, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Rightmost eigenvalues of ``L`` (ARPACK ``LR``), residual-gated like :func:`cayley_band`.

    Partially converged ARPACK results are used (and gated) when it does not fully converge.
    """
    L = sp.csc_matrix(L)
    try:
        kk = min(k, L.shape[0] - 2)
        lam, vec = spla.eigs(L, k=kk, which="LR", ncv=min(L.shape[0] - 1, max(4 * kk, 40)), maxiter=maxiter)
    except spla.ArpackNoConvergence as exc:
        lam, vec = exc.eigenvalues, exc.eigenvectors
        if len(lam) == 0:
            return np.array([]), np.array([])
    return _gate(L, lam, vec, tol)
