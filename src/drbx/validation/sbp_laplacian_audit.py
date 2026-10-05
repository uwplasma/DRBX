"""Audits of the nodal SBP perpendicular Laplacian (SciPy).

The host sparse assembly :class:`LaplacianAssembly` lives in :mod:`drbx.geometry.sbp_laplacian_assembly` (the plane-block
preconditioner is factorised from it, and production code must not import from ``validation``); it is re-exported here.

Audits: :func:`h_symmetry` (``H^-1/2 M H^-1/2`` against its transpose), :func:`definiteness_audit` (lowest eigenvalues of that
operator by dense, LOBPCG or shift-invert solves, flagging a negative direction, with a cost bound),
:func:`audit_laplacian_plan` (both for the Dirichlet and Neumann matrices of a plan) and :func:`trace_constants` (the exact
constants behind the penalty rule, dense, small cases).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from drbx.geometry.sbp_laplacian import LaplacianPlan
from drbx.geometry.sbp_laplacian_assembly import LaplacianAssembly

__all__ = ["LaplacianAssembly", "h_symmetry", "definiteness_audit", "audit_laplacian_plan", "trace_constants",
           "DEFINITENESS_MAX_UNKNOWNS"]

#: size bound of the shift-invert path of :func:`definiteness_audit` (the LU fills in; larger operators use LOBPCG)
DEFINITENESS_MAX_UNKNOWNS = 6_000


# ---------------------------------------------------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------------------------------------------------
def _normalized(M, H):
    s = sp.diags(1.0 / np.sqrt(np.asarray(H, dtype=np.float64).ravel()))
    return (s @ sp.csr_matrix(M) @ s).tocsr()


def h_symmetry(M, H) -> float:
    """``max |S - S^T| / max |S|`` of ``S = H^-1/2 M H^-1/2`` (zero for an ``H``-symmetric operator)."""
    S = _normalized(M, H)
    num = abs(S - S.T).max()
    return float(num / abs(S).max())


def _plane_block_preconditioner(M, planes: int, shift: float):
    """``r -> blockdiag_k(M_kk + shift)^-1 r`` (one sparse LU per eta plane of the flat index ``k * P + p``)."""
    n = M.shape[0]
    P = n // planes
    M = sp.csr_matrix(M)
    lus = [spla.splu((M[k * P:(k + 1) * P, k * P:(k + 1) * P] + shift * sp.identity(P)).tocsc()) for k in range(planes)]

    def apply(r):
        r = np.asarray(r)
        if r.ndim == 2:
            return np.column_stack([apply(r[:, i]) for i in range(r.shape[1])])
        return np.concatenate([lu.solve(np.ascontiguousarray(r[k * P:(k + 1) * P])) for k, lu in enumerate(lus)])

    return spla.LinearOperator((n, n), matvec=apply, matmat=apply, dtype=np.float64)


def definiteness_audit(M, H, *, k: int = 4, kind: str = "dirichlet", method: str = "auto", planes: int | None = None,
                       shift: float | None = None, tol: float = 1e-9, max_unknowns: int = DEFINITENESS_MAX_UNKNOWNS,
                       allow_large: bool = False, dense_below: int = 1500, lobpcg_maxiter: int = 80,
                       lobpcg_tol: float = 1e-7, return_vectors: bool = False) -> dict:
    """Lowest eigenvalues of the ``H``-normalised symmetric part of ``M`` and a flag for any negative direction.

    ``S = H^-1/2 (M + M^T)/2 H^-1/2``. Methods: ``"dense"`` (full eigensolve; ``"auto"`` below ``dense_below`` unknowns);
    ``"lobpcg"`` (``"auto"`` above, needs ``planes``, the number of eta planes, since ``M`` is block-structured by plane:
    LOBPCG for the smallest algebraic eigenvalues preconditioned with the exact in-plane blocks; the cost is bounded by
    ``lobpcg_maxiter`` matvecs and block solves, about 3 s at 1.5e4 unknowns on the family-A testbed, and the result carries
    the residual norms); ``"shift-invert"`` (Lanczos on one sparse LU of ``S + sigma``, ``sigma = 1e-6 rho`` by default; accurate
    near zero but the fill makes the cost grow faster than linearly: about 2 s at 4e3 and 55 s at 1.5e4 unknowns; bounded by
    ``max_unknowns`` unless ``allow_large``).

    A Neumann operator has the constants as an exact null direction: ``negative`` counts eigenvalues below ``-tol rho`` and
    ``null_defect`` is ``|S v0| / |v0|`` (relative to ``rho``) of ``v0 = H^1/2 1``; with LOBPCG the null direction is
    constrained out and ``lowest`` are the lowest non-null eigenvalues.

    Shift-invert returns the eigenvalues *nearest zero* (accurate, but a strongly negative direction far from the cluster
    would be missed); LOBPCG and the dense solve return the smallest algebraic eigenvalues, so ``"auto"`` prefers LOBPCG
    whenever ``planes`` is given.

    ``return_vectors`` (opt-in, default off) adds ``vectors (n, k)``: the unit eigenvectors of ``S`` (columns ordered as
    ``lowest``; ``H^-1/2`` times a column is the corresponding vector of nodal values) to locate a negative direction.

    Returns ``dict(n, rho, lowest (ascending), negative, n_negative, min_eig_rel, null_defect, seconds, method[, residuals,
    vectors])``.
    """
    t0 = time.perf_counter()
    h = np.asarray(H, dtype=np.float64).ravel()
    S = _normalized(M, h)
    S = ((S + S.T) * 0.5).tocsr()
    n = S.shape[0]
    if method == "auto":
        method = "dense" if n <= dense_below else ("lobpcg" if planes is not None else "shift-invert")
    extra = {}
    v0 = np.sqrt(h) / np.linalg.norm(np.sqrt(h))
    rho = None
    if method == "dense":
        ev, evec = np.linalg.eigh(S.toarray())
        rho = float(np.abs(ev).max())
        low = ev[:k]
        vecs = evec[:, :k]
    elif method == "shift-invert":
        if n > max_unknowns and not allow_large:
            raise ValueError(f"{n} unknowns exceed the shift-invert bound {max_unknowns}; "
                             "use method='lobpcg' or allow_large=True")
        rho = float(np.abs(spla.eigsh(S, k=1, which="LM", tol=1e-4, return_eigenvectors=False)[0]))
        sigma = -(1e-6 * rho if shift is None else shift)
        low, vecs = spla.eigsh(S, k=k, sigma=sigma, which="LM", tol=1e-10, return_eigenvectors=True)
    elif method == "lobpcg":
        if planes is None:
            raise ValueError("method='lobpcg' needs planes (the number of eta planes)")
        rho = float(np.abs(spla.eigsh(S, k=1, which="LM", tol=1e-4, return_eigenvectors=False)[0]))
        sq = np.sqrt(h)
        prec_in = _plane_block_preconditioner(M if kind == "neumann" else sp.csr_matrix(M), planes,
                                              1e-9 * rho * float(h.min()) if shift is None else shift)
        T = spla.LinearOperator(S.shape, matvec=lambda r: sq * prec_in.matvec(sq * np.ravel(r)),
                                matmat=lambda r: sq[:, None] * prec_in.matmat(sq[:, None] * r), dtype=np.float64)
        X = np.random.default_rng(0).standard_normal((n, k))
        Y = v0[:, None] if kind == "neumann" else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)           # the residuals are returned instead
            low, vecs, hist = spla.lobpcg(S, X, M=T, Y=Y, tol=lobpcg_tol * rho, maxiter=lobpcg_maxiter, largest=False,
                                          retResidualNormsHistory=True)
        extra["residuals"] = (np.asarray(hist[-1]) / rho).tolist()
        extra["iterations"] = len(hist) - 1
    else:
        raise ValueError(f"unknown method {method!r}")
    low = np.asarray(low, dtype=np.float64)
    order = np.argsort(low)
    low = low[order]
    if return_vectors:
        extra["vectors"] = np.asarray(vecs)[:, order]
    null_defect = float(np.linalg.norm(S @ v0) / rho) if kind == "neumann" else None
    return {"n": int(n), "rho": rho, "lowest": low.tolist(), "n_negative": int((low < -tol * rho).sum()),
            "negative": bool(low[0] < -tol * rho), "min_eig_rel": float(low[0] / rho), "null_defect": null_defect,
            "seconds": time.perf_counter() - t0, "method": method, **extra}


def audit_laplacian_plan(lp: LaplacianPlan, coeff=None, c_kappa: float = 1.0, *, kinds=("dirichlet", "neumann"), k: int = 4,
                         raise_on_negative: bool = False, **kwargs) -> dict:
    """Plan-build audit: ``H``-symmetry and the lowest eigenvalues of the Dirichlet and Neumann energy matrices.

    Assembles ``M`` on the host (:class:`LaplacianAssembly`) and runs :func:`h_symmetry` and :func:`definiteness_audit`
    (``kwargs`` are passed to the latter: ``max_unknowns``, ``allow_large``, ...). Returns
    ``dict(<kind>=dict(h_symmetry, **definiteness), flags=[...], seconds)``; ``flags`` names every kind with a negative
    direction (``raise_on_negative`` raises ``ValueError`` instead). The Neumann matrix is semidefinite only up to the
    narrow form's indefinite remainders (see the P09 study): a flag there is information, not necessarily an error.
    """
    t0 = time.perf_counter()
    asm = LaplacianAssembly(lp, coeff, c_kappa)
    out, flags = {}, []
    for kind in kinds:
        M = asm.matrix(kind)
        res = {"h_symmetry": h_symmetry(M, asm.H)}
        res.update(definiteness_audit(M, asm.H, k=k, kind=kind, planes=lp.structure.n_eta, **kwargs))
        out[kind] = res
        if res["negative"]:
            flags.append(kind)
    if flags and raise_on_negative:
        raise ValueError(f"the Laplacian energy matrix has a negative direction for: {flags}")
    out["flags"] = flags
    out["seconds"] = time.perf_counter() - t0
    return out


def trace_constants(asm: LaplacianAssembly) -> dict:
    """Exact ``C_X = sup |F_X f|^2_Omega / E_block(f)`` for the core side, the ring side and the wall (dense; small cases).

    ``E_block`` is the volume form of the block (core nodes or ring nodes), regularised along the constants; these are the
    constants the penalty rule ``tau = C_c / 4 + C_r / 2``, ``tau_w = 2 C_w`` (:func:`~drbx.geometry.sbp_laplacian.penalty_rule`)
    approximates by local ratios.
    """
    st = asm.st
    E, P, Nc = asm.E, asm.P, st.Nc
    Fc, Fr, Fw = asm.flux_ops()
    n = E * P
    vol = sp.csr_matrix((n, n))
    Gu, cu = asm.f_uu
    Gt, ct = asm.f_tt
    Ge, ce = asm.f_ee
    for a, c, b in [(Gu, cu, Gu), (Gt, ct, Gt), (Ge, ce, Ge)] + [(asm.D[i], c, asm.D[j]) for i, j, c in asm.coll] \
            + [(asm.shell, asm.kappa_w, asm.shell)]:
        vol = vol + a.T @ sp.diags(c) @ b
    H = asm.H
    plane_idx = np.arange(E)[:, None] * P
    core_idx = (plane_idx + np.arange(Nc)[None, :]).ravel()
    ring_idx = (plane_idx + np.arange(Nc, P)[None, :]).ravel()
    out = {}
    for name, idx, Fop in (("C_c", core_idx, Fc), ("C_r", ring_idx, Fr), ("C_w", ring_idx, Fw)):
        Md = vol[idx][:, idx].toarray()
        Hb = H[idx]
        Md[np.diag_indices_from(Md)] += 1e-10 * np.abs(Md).max() / Hb.max() * Hb
        Fd = Fop[:, idx].toarray()
        K = Fd.T @ (asm.Om[:, None] * Fd)
        out[name] = float(sla.eigh(K, Md, eigvals_only=True, subset_by_index=[len(idx) - 1, len(idx) - 1])[0])
    return out
