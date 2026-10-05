"""Matrix-free definiteness audit of the nodal Laplacian (the large-N counterpart of ``definiteness_audit``).

The package audit assembles the full sparse energy matrix on the host (about ``12 N`` nonzeros per row, ``~70`` bytes of host
transient per nonzero): ``~1 GB`` at N32 but ``~4`` and ``~13 GB`` at N48 and N64, over the campaign's 4 GB budget. Here the same
quantity is computed without assembling ``M``: LOBPCG for the smallest algebraic eigenvalues of ``S = H^-1/2 M H^-1/2`` (``M`` the
JAX energy form with zero wall data, Dirichlet or conormal Neumann), preconditioned by the in-plane block LDU of the Dirichlet
matrix (``tools.build_merged_preconditioner``) and, for Neumann, with the constants' null direction constrained out, exactly as
``definiteness_audit(method="lobpcg")`` does. The cost is bounded by ``maxiter`` iterations of ``k`` matvecs and ``k`` block solves.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import scipy.sparse.linalg as spla


def matrix_free_audit(plan, kind: str, prec, *, k: int = 4, maxiter: int = 600, tol_rel_rho: float = 1e-15,
                      seed: int = 0, return_vectors: bool = True, probe_symmetry: bool = True) -> dict:
    import jax
    import jax.numpy as jnp
    from drbx.native.fci_perpendicular_plane_preconditioner import apply_plane_preconditioner
    from drbx.native.fci_perpendicular_sbp_laplacian import laplacian_form

    t0 = time.perf_counter()
    plan = jax.tree_util.tree_map(jnp.asarray, plan)                   # device-resident: no host-to-device copy per call
    st = plan.structure
    E, P = st.n_eta, st.P
    h = (np.asarray(plan.Hp) * st.deta).ravel()
    sq = np.sqrt(h)
    n = E * P
    Mv_ = jax.jit(lambda lp, v: laplacian_form(lp, v.reshape(E, P, -1), None, kind, None, 1.0,
                                               neumann_mode="conormal").reshape(E * P, -1))       # (n, F) columns at once
    Pa_ = jax.jit(jax.vmap(apply_plane_preconditioner, in_axes=(None, 1), out_axes=1))
    sqj = jnp.asarray(sq)

    def S_mm(X):
        X = np.asarray(X)
        X = X.reshape(n, -1)
        return np.asarray(Mv_(plan, jnp.asarray(X) / sqj[:, None])) / sq[:, None]

    def S_mv(x):
        return S_mm(np.asarray(x).reshape(n, 1))[:, 0]

    def T_mm(R):
        R = np.asarray(R).reshape(n, -1)
        return sq[:, None] * np.asarray(Pa_(prec, jnp.asarray(sq[:, None] * R)))

    def T_mv(r):
        return T_mm(np.asarray(r).reshape(n, 1))[:, 0]

    S = spla.LinearOperator((n, n), matvec=S_mv, matmat=S_mm, dtype=np.float64)
    T = spla.LinearOperator((n, n), matvec=T_mv, matmat=T_mm, dtype=np.float64)
    rho = float(np.abs(spla.eigsh(S, k=1, which="LM", tol=1e-4, return_eigenvectors=False)[0]))
    v0 = sq / np.linalg.norm(sq)
    out = dict(n=int(n), rho=rho, kind=kind, method="lobpcg-matrix-free")
    if probe_symmetry:
        rng = np.random.default_rng(seed + 1)
        asym = []
        for _ in range(3):
            a, b = rng.standard_normal(n), rng.standard_normal(n)
            asym.append(abs(a @ S_mv(b) - b @ S_mv(a)) / (np.linalg.norm(S_mv(a)) * np.linalg.norm(b)))
        out["h_symmetry_probe"] = float(max(asym))
    X = np.random.default_rng(seed).standard_normal((n, k))
    Y = v0[:, None] if kind == "neumann" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        low, vecs, hist = spla.lobpcg(S, X, M=T, Y=Y, tol=tol_rel_rho * rho, maxiter=maxiter, largest=False,
                                      retResidualNormsHistory=True)
    order = np.argsort(low)
    low = np.asarray(low)[order]
    out.update(lowest=low.tolist(), residuals=(np.asarray(hist[-1])[order] / rho).tolist(), iterations=len(hist) - 1,
               n_negative=int((low < -1e-9 * rho).sum()), negative=bool(low[0] < -1e-9 * rho), min_eig_rel=float(low[0] / rho),
               null_defect=float(np.linalg.norm(S_mv(v0)) / rho) if kind == "neumann" else None,
               seconds=time.perf_counter() - t0, parameters=dict(k=k, maxiter=maxiter, tol_rel_rho=tol_rel_rho))
    if return_vectors:
        out["vectors"] = np.asarray(vecs)[:, order]
    return out
