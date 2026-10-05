"""Host sparse assembly of the nodal SBP perpendicular Laplacian (SciPy), shared by the audits and the plane preconditioner.

:class:`LaplacianAssembly` assembles the energy matrix ``M`` (``v^T M f = a(f, v)``, flat index ``k * P + p`` for plane
``k`` and node ``p``) of a :class:`~drbx.geometry.sbp_laplacian.LaplacianPlan` as a sum of sparse factors
``A^T diag(c) B`` and gives the boundary-data vector, independently of the JAX apply of
:mod:`drbx.native.fci_perpendicular_sbp_laplacian` (dense angular matrices instead of FFTs, explicit transposes instead of
``jax.vjp``), so the two check each other. ``inplane=True`` keeps only the couplings inside one eta plane: these are exactly the
diagonal plane blocks of the full matrix (every eta derivative has a zero diagonal block) and what the plane-block
preconditioner is factorised from; the full matrix is for small cases and audits.

Windowed in-plane assembly (bounded memory). The in-plane block of plane ``k`` only needs the plan data of plane ``k`` and, through
the ``eta eta`` diagonal ``Ge_diag`` (``(Ge^2)^T W a_ee,h``), the half-plane coefficients of the planes ``k - 2 .. k + 1``; the
polarization coefficient enters through the global face coefficients and penalty scalings. ``LaplacianAssembly(lp, coeff, c_kappa,
window=(k0, k1))`` assembles only the planes ``k0 .. k1 - 1`` (``inplane=True`` only) from the *same* global data
(:func:`global_assembly_data`, computed once and reusable through ``shared=``), so that every in-plane entry is bitwise the entry
of the full assembly. :func:`iter_inplane_blocks` yields the in-plane CSR blocks of consecutive groups of planes.
"""
from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterator

import numpy as np
import scipy.sparse as sp

from drbx.geometry.sbp_laplacian import LaplacianPlan, face_coefficients, fourier_half_interp, stag_eta, stag_fourier
from drbx.geometry.sbp_operators import TWO_PI, deta_matrix, ring_dtheta

__all__ = ["LaplacianAssembly", "global_assembly_data", "iter_inplane_blocks", "plane_group_size"]

#: per-plane arrays of a :class:`LaplacianPlan` (leading axis ``n_eta``) that a window slices
_PLANE_FIELDS = ("Hp", "A", "auu_f", "att_h", "aee_h", "kappa", "wall_alpha", "wall_beta_th", "wall_beta_eta")


def _pad(M, left: int, total: int):
    M = sp.csr_matrix(M)
    return sp.hstack([sp.csr_matrix((M.shape[0], left)), M, sp.csr_matrix((M.shape[0], total - left - M.shape[1]))], format="csr")


def global_assembly_data(lp: LaplacianPlan, coeff=None) -> dict:
    """Quantities of the assembly that couple the planes: ``A``, the face coefficients ``auu, att, aee``, ``c_core``, ``c_wall``, the
    penalty scalings ``s_in, s_wall`` (maxima over all planes), and ``Ge_diag (E, P)``, the in-plane diagonal of ``Ge^T c Ge``
    (through the periodic ``eta`` interpolation of ``a_ee`` and the staggered ``eta`` derivative).

    Computed from all ``E`` planes of ``lp``; a windowed :class:`LaplacianAssembly` takes the rows of its planes.
    """
    st = lp.structure
    E, Nc, m, N = st.n_eta, st.Nc, st.m, st.N
    deta = st.deta
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
    W = np.asarray(lp.wxy) * deta
    Ge_diag = (stag_eta(E, deta) ** 2).T @ (W[None, :] * aee)
    return dict(A=A, auu=auu, att=att, aee=aee, c_core=c_core, c_wall=c_wall, s_in=s_in, s_wall=s_wall, Ge_diag=Ge_diag)


def _slice_plan(lp: LaplacianPlan, k0: int, k1: int) -> LaplacianPlan:
    """``lp`` restricted to the planes ``k0 .. k1 - 1`` (``n_eta = k1 - k0``; no periodic wrap)."""
    kw = {}
    for name in _PLANE_FIELDS:
        v = getattr(lp, name)
        if v is not None:
            kw[name] = np.asarray(v)[k0:k1]
    return dataclasses.replace(lp, structure=dataclasses.replace(lp.structure, n_eta=k1 - k0), **kw)


def _slice_shared(sh: dict, k0: int, k1: int) -> dict:
    out = dict(sh)
    for name in ("A", "auu", "att", "aee", "Ge_diag"):
        out[name] = sh[name][k0:k1]
    for name in ("c_core", "c_wall"):
        if np.ndim(sh[name]):
            out[name] = sh[name][k0:k1]
    return out


class LaplacianAssembly:
    """Sparse factors of the Laplacian energy form of ``lp`` for a polarization coefficient ``coeff (E, P)`` (or ``None``).

    ``window=(k0, k1)`` assembles only the planes ``k0 .. k1 - 1`` (``E = k1 - k0``; ``inplane=True`` matrices only, bitwise the
    rows and columns of those planes of the full in-plane matrix); ``shared`` is the result of :func:`global_assembly_data` of
    the full plan, to compute it once for many windows.
    """

    def __init__(self, lp: LaplacianPlan, coeff=None, c_kappa: float = 1.0, *, window: tuple[int, int] | None = None,
                 shared: dict | None = None) -> None:
        t0 = time.perf_counter()
        sh = global_assembly_data(lp, coeff) if shared is None else shared
        self.window = None if window is None else (int(window[0]), int(window[1]))
        if window is not None:
            k0, k1 = self.window
            if not 0 <= k0 < k1 <= lp.structure.n_eta:
                raise ValueError(f"window {window} is not inside the {lp.structure.n_eta} planes")
            lp, sh = _slice_plan(lp, k0, k1), _slice_shared(sh, k0, k1)
        st = lp.structure
        self.lp, self.st = lp, st
        E, P, Nc, m, N = st.n_eta, st.P, st.Nc, st.m, st.N
        deta = st.deta
        self.E, self.P = E, P
        A, auu, att, aee, c_core, c_wall = (sh[k] for k in ("A", "auu", "att", "aee", "c_core", "c_wall"))
        self.A, self.auu = A, auu
        self.tau = float(lp.tau) * sh["s_in"]
        self.tau_w = float(lp.tau_w) * sh["s_wall"]
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
        self.Ge_diag = sh["Ge_diag"]                                          # (E, P): in-plane diagonal of Ge^T c Ge
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
        if self.window is not None and not inplane:
            raise ValueError("a windowed LaplacianAssembly gives the in-plane matrix only (inplane=True)")
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
        if self.window is not None:
            raise ValueError("a windowed LaplacianAssembly gives the in-plane matrix only")
        g = np.asarray(g, dtype=np.float64).ravel()
        if kind == "dirichlet":
            _Fc, _Fr, Fw = self.flux_ops()
            return -(Fw.T @ (self.Om * g)) + self.tau_w * (self.Tw.T @ (self.Om * g))
        if mode == "physical":
            g = self._phys[0].ravel() * g
        return self.Tw.T @ (self.Om * g)

    def apply_L(self, f, g, kind: str = "dirichlet", mode: str = "conormal") -> np.ndarray:
        """``L f`` as ``(E, P)`` for ``f (E, P)`` and wall data ``g (E, N)``."""
        if self.window is not None:
            raise ValueError("a windowed LaplacianAssembly gives the in-plane matrix only")
        f = np.asarray(f, dtype=np.float64).ravel()
        M = self.matrix(kind, mode)
        return (-(M @ f - self.data_vector(g, kind, mode)) / self.H).reshape(self.E, self.P)



#: default host-memory budget of one group of planes: stored entries of the in-plane matrix of the group (each costs about
#: 24 bytes through the COO transients of the factorisation)
GROUP_NNZ_TARGET = 12_000_000


def plane_group_size(st, target_nnz: int = GROUP_NNZ_TARGET) -> int:
    """Number of planes per group so that the in-plane matrix of a group has about ``target_nnz`` entries.

    A ring row couples about 13 rings (the radial reach of the staggered closure) with a dense ``N x N`` block each.
    """
    per_plane = 13 * st.m * st.N ** 2 + 2 * st.Nc * st.m * st.N
    return int(max(1, min(st.n_eta, target_nnz // max(per_plane, 1))))


def iter_inplane_blocks(lp: LaplacianPlan, coeff=None, c_kappa: float = 1.0, *, kind: str = "dirichlet",
                        group_planes: int | None = None, shared: dict | None = None) -> Iterator[tuple[int, int, sp.csr_matrix]]:
    """Yield ``(k0, k1, M_group)``: the in-plane CSR matrix ``((k1 - k0) P, (k1 - k0) P)`` of consecutive groups of planes.

    Each ``M_group`` is bitwise ``LaplacianAssembly(lp, coeff, c_kappa).matrix(kind, inplane=True)[k0 P:k1 P, k0 P:k1 P]``, but only
    the group is assembled, so the host memory is bounded by the group size (``group_planes``; default
    :func:`plane_group_size`) instead of growing with the number of planes. ``shared`` is :func:`global_assembly_data` if already
    computed.
    """
    E = lp.structure.n_eta
    g = plane_group_size(lp.structure) if group_planes is None else int(group_planes)
    if g < 1:
        raise ValueError(f"group_planes must be positive, got {g}")
    sh = global_assembly_data(lp, coeff) if shared is None else shared
    for k0 in range(0, E, g):
        k1 = min(k0 + g, E)
        yield k0, k1, LaplacianAssembly(lp, coeff, c_kappa, window=(k0, k1), shared=sh).matrix(kind, inplane=True)
