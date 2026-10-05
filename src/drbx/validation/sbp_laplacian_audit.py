"""Host sparse assembly and audits of the nodal SBP perpendicular Laplacian (SciPy).

:class:`LaplacianAssembly` assembles the energy matrix ``M`` (``v^T M f = a(f, v)``, flat index ``k * P + p`` for plane
``k`` and node ``p``) of a :class:`~drbx.geometry.sbp_laplacian.LaplacianPlan` as a sum of sparse factors
``A^T diag(c) B`` and gives the boundary-data vector, independently of the JAX apply of
:mod:`drbx.native.fci_perpendicular_sbp_laplacian` (dense angular matrices instead of FFTs, explicit transposes instead of
``jax.vjp``), so the two check each other. ``inplane=True`` keeps only the couplings inside one eta plane: these are exactly the
diagonal plane blocks of the full matrix (every eta derivative has a zero diagonal block) and what the plane-block
preconditioner is factorised from; the full matrix is for small cases and audits.

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

from drbx.geometry.sbp_laplacian import LaplacianPlan, face_coefficients, fourier_half_interp, stag_eta, stag_fourier
from drbx.geometry.sbp_operators import TWO_PI, deta_matrix, ring_dtheta

__all__ = ["LaplacianAssembly", "h_symmetry", "definiteness_audit", "audit_laplacian_plan", "trace_constants",
           "DEFINITENESS_MAX_UNKNOWNS"]

#: size bound of the shift-invert path of :func:`definiteness_audit` (the LU fills in; larger operators use LOBPCG)
DEFINITENESS_MAX_UNKNOWNS = 6_000


def _pad(M, left: int, total: int):
    M = sp.csr_matrix(M)
    return sp.hstack([sp.csr_matrix((M.shape[0], left)), M, sp.csr_matrix((M.shape[0], total - left - M.shape[1]))], format="csr")


class LaplacianAssembly:
    """Sparse factors of the Laplacian energy form of ``lp`` for a polarization coefficient ``coeff (E, P)`` (or ``None``)."""

    def __init__(self, lp: LaplacianPlan, coeff=None, c_kappa: float = 1.0) -> None:
        t0 = time.perf_counter()
        st = lp.structure
        self.lp, self.st = lp, st
        E, P, Nc, m, N = st.n_eta, st.P, st.Nc, st.m, st.N
        deta = st.deta
        self.E, self.P = E, P
        A0 = np.asarray(lp.A)
        if coeff is None:
            A = A0
            auu, att, aee = (np.asarray(x) for x in (lp.auu_f, lp.att_h, lp.aee_h))
            c_core, c_wall, s_in, s_wall = 1.0, 1.0, 1.0, 1.0
        else:
            coeff = np.asarray(coeff, dtype=np.float64)
            A = coeff[..., None, None] * A0
            auu, att, aee = face_coefficients(A, np.asarray(lp.Iu), fourier_half_interp(N, np.pi / st.n),
                                              fourier_half_interp(E, 0.0), Nc, m, N)
            cr = coeff[:, Nc:].reshape(E, m, N)
            c_in = np.einsum("r,erj->ej", np.asarray(lp.t_in)[:4], cr[:, :4])
            c_wall = np.einsum("r,erj->ej", np.asarray(lp.t_out)[-4:], cr[:, -4:])
            c_core_tr = np.einsum("jp,ep->ej", np.asarray(lp.Rx), coeff[:, :Nc])
            s_in = max(np.abs(c_in).max(), np.abs(c_core_tr).max())
            s_wall = np.abs(c_wall).max()
            c_core = coeff[:, :Nc]
        self.A, self.auu = A, auu
        self.tau = float(lp.tau) * s_in
        self.tau_w = float(lp.tau_w) * s_wall
        self.c_wall = c_wall
        W = np.asarray(lp.wxy) * deta
        self.W = W
        self.H = (np.asarray(lp.Hp) * deta).ravel()
        self.Om = np.full(E * N, TWO_PI / N * deta)
        delta = np.pi / st.n
        IE, IP_ = sp.identity(E, format="csr"), sp.identity(P, format="csr")
        eye_m, eye_N = sp.identity(m, format="csr"), sp.identity(N, format="csr")

        def plane(M):                                                 # plane-local operator on all E planes
            return sp.kron(IE, sp.csr_matrix(M), format="csr")

        # collocated derivatives
        D1p = sp.block_diag([sp.csr_matrix(lp.core_D1), sp.kron(sp.csr_matrix(lp.Du), eye_N)], format="csr")
        D2p = sp.block_diag([sp.csr_matrix(lp.core_D2), sp.kron(eye_m, sp.csr_matrix(ring_dtheta(N, delta)))], format="csr")
        self.D = [plane(D1p), plane(D2p), sp.kron(sp.csr_matrix(deta_matrix(E, deta)), IP_, format="csr")]
        # staggered
        Gu = plane(_pad(sp.kron(sp.csr_matrix(lp.Dp), eye_N), Nc, P))
        Gt = plane(_pad(sp.kron(eye_m, sp.csr_matrix(stag_fourier(N, delta))), Nc, P))
        Ge_dense = stag_eta(E, deta)
        Ge = sp.kron(sp.csr_matrix(Ge_dense), IP_, format="csr")
        # traces
        t_in, t_out = np.asarray(lp.t_in), np.asarray(lp.t_out)
        self.Tc = plane(_pad(sp.csr_matrix(lp.Rx), 0, P))
        self.Tr = plane(_pad(sp.kron(sp.csr_matrix(t_in[None, :]), eye_N), Nc, P))
        self.Tw = plane(_pad(sp.kron(sp.csr_matrix(t_out[None, :]), eye_N), Nc, P))
        # shell
        shell = sp.vstack([sp.csr_matrix(lp.core_IP), sp.csr_matrix((P - Nc, Nc))], format="csr")
        shell = _pad(shell, 0, P)
        self.shell = plane(shell)
        kap = np.zeros((E, P))
        kap[:, :Nc] = W[None, :Nc] * c_kappa * np.asarray(lp.kappa) * c_core
        self.kappa_w = kap.ravel()
        # face weights
        wfw = (np.asarray(lp.wf) * (TWO_PI / N * deta))[None, :, None] * auu
        mask = np.ones((P, 3, 3)) - np.eye(3)[None]
        mask[:Nc] = 1.0
        mask[:Nc, 2, 2] = 0.0
        self.coll = [(i, j, (W[None, :] * A[..., i, j] * mask[None, :, i, j]).ravel()) for i in range(3) for j in range(3)
                     if np.any(mask[:, i, j])]
        self.f_uu = (Gu, wfw.ravel())
        self.f_tt = (Gt, (W[Nc:].reshape(m, N)[None] * att).ravel())
        self.f_ee = (Ge, (W[None, :] * aee).ravel())
        self.Ge_diag = ((Ge_dense ** 2).T @ (W[None, :] * aee))               # (E, P): in-plane diagonal of Ge^T c Ge
        # flux traces (+u): full (with the eta terms) and in-plane
        rows = np.arange(E * N)
        e_idx, j_idx = rows // N, rows % N
        sel_in = sp.csr_matrix((np.ones(E * N), (rows, (e_idx * (m + 1) + 0) * N + j_idx)), shape=(E * N, E * (m + 1) * N))
        sel_wall = sp.csr_matrix((np.ones(E * N), (rows, (e_idx * (m + 1) + m) * N + j_idx)), shape=(E * N, E * (m + 1) * N))
        self._sel = (sel_in, sel_wall)
        self._Gu = Gu
        self._auu = auu
        self._cs = np.tile(np.asarray(lp.cos_g), E)
        self._sn = np.tile(np.asarray(lp.sin_g), E)
        self.R_c = st.R_c
        # wall conversion
        self._phys = None
        if lp.wall_alpha is not None:
            self._phys = (np.asarray(lp.wall_alpha) * c_wall, np.asarray(lp.wall_beta_th) * c_wall,
                          np.asarray(lp.wall_beta_eta) * c_wall)
        self._Dth_w = sp.kron(IE, sp.csr_matrix(ring_dtheta(N, delta)), format="csr")
        self._De_w = sp.kron(sp.csr_matrix(deta_matrix(E, deta)), eye_N, format="csr")
        self.build_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------ flux operators
    def flux_ops(self, inplane: bool = False):
        """``(F_c, F_r, F_w)``: the +u flux traces ``(E N, E P)`` of the core side, the ring side and the wall."""
        E, N, P, Nc, m = self.E, self.st.N, self.P, self.st.Nc, self.st.m
        A = self.A
        comps = (0, 1) if inplane else (0, 1, 2)
        diag = lambda v: sp.diags(np.asarray(v).ravel())  # noqa: E731
        coll = lambda T, js: sum(T @ diag(A[..., 0, j]) @ self.D[j] for j in js)  # noqa: E731
        Fr = diag(self._auu[:, 0]) @ self._sel[0] @ self._Gu + coll(self.Tr, [j for j in (1, 2) if j in comps])
        Fw = diag(self._auu[:, m]) @ self._sel[1] @ self._Gu + coll(self.Tw, [j for j in (1, 2) if j in comps])
        maskc = np.zeros((E, P))
        maskc[:, :Nc] = 1.0
        Fx = sum(diag(A[..., 0, j] * maskc) @ self.D[j] for j in comps)
        Fy = sum(diag(A[..., 1, j] * maskc) @ self.D[j] for j in comps)
        Fc = self.R_c * (diag(self._cs) @ self.Tc @ Fx + diag(self._sn) @ self.Tc @ Fy)
        return Fc.tocsr(), Fr.tocsr(), Fw.tocsr()

    # ------------------------------------------------------------------ the matrix
    def factors(self, kind: str = "dirichlet", mode: str = "conormal", inplane: bool = False):
        """List of ``(A, c, B)`` with ``M = sum A^T diag(c) B`` (``inplane``: the diagonal plane blocks only)."""
        Fc, Fr, Fw = self.flux_ops(inplane)
        fac = []
        Gu, cu = self.f_uu
        Gt, ct = self.f_tt
        fac += [(Gu, cu, Gu), (Gt, ct, Gt)]
        if not inplane:
            Ge, ce = self.f_ee
            fac.append((Ge, ce, Ge))
        for i, j, c in self.coll:
            if inplane and 2 in (i, j):
                continue
            fac.append((self.D[i], c, self.D[j]))
        fac.append((self.shell, self.kappa_w, self.shell))
        # SIPG
        jmp = self.Tr - self.Tc
        fav = 0.5 * (Fc + Fr)
        fac += [(jmp, self.Om, fav), (fav, self.Om, jmp), (jmp, self.tau * self.Om, jmp)]
        if kind == "dirichlet":
            fac += [(self.Tw, -self.Om, Fw), (Fw, -self.Om, self.Tw), (self.Tw, self.tau_w * self.Om, self.Tw)]
        elif mode == "physical":
            if self._phys is None:
                raise ValueError("the plan has no wall metric (ginv_u) for physical-normal Neumann data")
            _a, bth, bet = self._phys
            fac.append((self.Tw, -self.Om, sp.diags(bth.ravel()) @ self._Dth_w @ self.Tw))
            if not inplane:
                fac.append((self.Tw, -self.Om, sp.diags(bet.ravel()) @ self._De_w @ self.Tw))
        return fac

    def matrix(self, kind: str = "dirichlet", mode: str = "conormal", inplane: bool = False) -> sp.csr_matrix:
        """Sparse ``M`` of the field kind (``"dirichlet"`` / ``"neumann"``; ``mode`` for the Neumann datum type)."""
        n = self.E * self.P
        M = sp.csr_matrix((n, n))
        for a, c, b in self.factors(kind, mode, inplane):
            M = M + a.T @ sp.diags(c) @ b
        if inplane:
            M = M + sp.diags(self.Ge_diag.ravel())
        return M.tocsr()

    def data_vector(self, g, kind: str = "dirichlet", mode: str = "conormal") -> np.ndarray:
        """``b`` with ``L f = -H^-1 (M f - b)`` for wall data ``g (E, N)``: Dirichlet value, conormal flux or physical normal."""
        g = np.asarray(g, dtype=np.float64).ravel()
        if kind == "dirichlet":
            _Fc, _Fr, Fw = self.flux_ops()
            return -(Fw.T @ (self.Om * g)) + self.tau_w * (self.Tw.T @ (self.Om * g))
        if mode == "physical":
            g = self._phys[0].ravel() * g
        return self.Tw.T @ (self.Om * g)

    def apply_L(self, f, g, kind: str = "dirichlet", mode: str = "conormal") -> np.ndarray:
        """``L f`` as ``(E, P)`` for ``f (E, P)`` and wall data ``g (E, N)``."""
        f = np.asarray(f, dtype=np.float64).ravel()
        M = self.matrix(kind, mode)
        return (-(M @ f - self.data_vector(g, kind, mode)) / self.H).reshape(self.E, self.P)


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
