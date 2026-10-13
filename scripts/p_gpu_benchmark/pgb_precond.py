"""Plane-preconditioner variants of the P GPU benchmark (jax; reads the package, writes nothing into it).

Per plane the in-plane Dirichlet block is ``[[M_cc, M_cr], [M_rc, M_rr]]`` (``Nc`` dense core nodes, ``m`` rings of ``N`` nodes). Variants,
all applying ``blockdiag_k(M_k^-1)`` to a flat plane-major vector of ``E P`` entries:

* ``core_schur``       the package (:mod:`drbx.native.fci_perpendicular_plane_preconditioner`): banded ring ``L D L^T`` + stored
                       ``Z = D^-1 L^-1 M_rc`` (``(E, S B, Nc)``) + ``S_c^-1``.
* ``core_schur_zfree`` the same factors ``lo, dinv, sinv`` without ``Z``: with ``A = L D L^T`` the ring block,
                       ``z_c = S_c^-1 (r_c - M_cr A^-1 r_r)``, ``x_r = A^-1 r_r - A^-1 (M_rc z_c)`` -- two banded ring solves and the
                       coupling ``M_rc`` stored only over the rings that couple to the core (``(E, nr, Nc)``).
* ``block_jacobi``     :func:`solvax.block_jacobi` on the dense plane blocks (batched LU, partial pivoting).
* ``block_thomas``     :func:`solvax.block_thomas_factor` / ``_solve`` on the rings (block tridiagonal, ``rings_per_block = ring reach``)
                       and the dense core by the projected Schur complement (``schur_projected_precond`` algebra with
                       ``A^-1 M_rc`` precomputed), as ``work/solvax_audit_20261012/bench_plane_precond.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import solvax
from jax.scipy.linalg import lu_solve

_NP = {"f32": np.float32, "f64": np.float64}
_JNP = {"f32": jnp.float32, "f64": jnp.float64}
DTYPE_NAME = {"float32": "f32", "float64": "f64"}


# ---------------------------------------------------------------------------
# host in-plane blocks
# ---------------------------------------------------------------------------
class PlaneBlocks:
    """The in-plane CSR groups ``[(k0, k1, M)]`` (consecutive planes) of the Dirichlet operator with the plane structure."""

    def __init__(self, groups, *, n_planes: int, nc: int, m: int, N: int):
        self.groups = list(groups)
        self.E, self.nc, self.m, self.N = int(n_planes), int(nc), int(m), int(N)
        self.P = self.nc + self.m * self.N
        if self.groups[0][0] != 0 or self.groups[-1][1] != self.E:
            raise ValueError("the groups must cover planes 0..E-1")
        self._reach = None

    @property
    def nnz(self) -> int:
        return int(sum(g[2].nnz for g in self.groups))

    @property
    def csr_bytes(self) -> int:
        return int(sum(g[2].data.nbytes + g[2].indices.nbytes + g[2].indptr.nbytes for g in self.groups))

    def plane_dense(self, p: int) -> np.ndarray:
        for k0, k1, mat in self.groups:
            if k0 <= p < k1:
                q = p - k0
                return np.asarray(mat[q * self.P:(q + 1) * self.P, q * self.P:(q + 1) * self.P].toarray())
        raise IndexError(p)

    def iter_dense(self):
        for p in range(self.E):
            yield self.plane_dense(p)

    def dense_stack(self, p0: int, p1: int, dtype=np.float64) -> np.ndarray:
        out = np.empty((p1 - p0, self.P, self.P), dtype)
        for i, p in enumerate(range(p0, p1)):
            out[i] = self.plane_dense(p)
        return out

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """``blockdiag_k(M_k) x`` for the flat ``x (E P,)`` (host, float64)."""
        x = np.asarray(x, np.float64)
        out = np.empty_like(x)
        for k0, k1, mat in self.groups:
            out[k0 * self.P:k1 * self.P] = mat @ x[k0 * self.P:k1 * self.P]
        return out

    def ring_reach(self) -> int:
        """Largest ring distance ``|i - j|`` of an in-plane ring-ring coupling over all planes."""
        if self._reach is None:
            reach = 0
            for _k0, _k1, mat in self.groups:
                coo = mat.tocoo()
                lp_r, lp_c = coo.row % self.P, (coo.col % self.P)
                keep = (lp_r >= self.nc) & (lp_c >= self.nc)
                if keep.any():
                    ri, ci = (lp_r[keep] - self.nc) // self.N, (lp_c[keep] - self.nc) // self.N
                    reach = max(reach, int(np.abs(ri - ci).max()))
            self._reach = max(reach, 1)
        return self._reach

    def coupled_ring_rows(self) -> int:
        """``nr``: ring rows (from the first ring) that couple to the core, over all planes."""
        nr = 0
        for _k0, _k1, mat in self.groups:
            coo = mat.tocoo()
            lp_r, lp_c = coo.row % self.P, (coo.col % self.P)
            keep = (lp_r >= self.nc) & (lp_c < self.nc)
            if keep.any():
                nr = max(nr, int(lp_r[keep].max()) + 1 - self.nc)
        return nr

    def mrc(self, nr: int) -> np.ndarray:
        """``M_rc`` of every plane restricted to the first ``nr`` ring rows: ``(E, nr, Nc)`` float64."""
        out = np.zeros((self.E, nr, self.nc))
        for k0, k1, mat in self.groups:
            for q in range(k1 - k0):
                blk = mat[q * self.P + self.nc:q * self.P + self.nc + nr, q * self.P:q * self.P + self.nc]
                out[k0 + q] = blk.toarray()
        return out


def restrict_groups(groups_iter, n_planes: int):
    """Keep the consecutive groups covering the first ``n_planes`` planes (the groups must tile it exactly)."""
    out = []
    for k0, k1, mat in groups_iter:
        if k0 >= n_planes:
            break
        if k1 > n_planes:
            raise ValueError("group size must divide the number of restricted planes")
        out.append((k0, k1, mat))
    return out


def plane_group_choice(E_all: int, default_group: int, max_planes: int | None) -> tuple[int, int]:
    """``(planes used, group_planes)``: all planes with the default group, or the first ``max_planes`` with a divisor group."""
    if not max_planes or max_planes >= E_all:
        return E_all, default_group
    g = max(d for d in range(1, max_planes + 1) if max_planes % d == 0 and d <= max(default_group, 1))
    return int(max_planes), g


# ---------------------------------------------------------------------------
# core_schur_zfree
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ZMeta:
    S: int
    B: int
    w: int
    E: int
    Nc: int
    m: int
    N: int
    nr: int
    dtype: str


@jax.tree_util.register_pytree_node_class
@dataclass(eq=False)
class ZfreePreconditioner:
    """``lo (S, E, B, w, B)``, ``dinv (S, E, B, B)``, ``sinv (E, Nc, Nc)`` (shared with a :class:`CoreSchurPreconditioner`) and the
    core coupling ``mrc (E, nr, Nc)`` = the first ``nr`` ring rows of ``M_rc`` (the rings that couple to the core)."""

    lo: Any
    dinv: Any
    sinv: Any
    mrc: Any
    meta: _ZMeta
    info: dict = field(default_factory=dict, compare=False, repr=False)

    def tree_flatten(self):
        return (self.lo, self.dinv, self.sinv, self.mrc), self.meta

    @classmethod
    def tree_unflatten(cls, meta, children):
        return cls(*children, meta, {})

    @property
    def nbytes(self) -> int:
        return int(sum(x.nbytes for x in (self.lo, self.dinv, self.sinv, self.mrc)))


def _ring_solve(lo, dinv, b):
    """``A^-1 b`` for the ring block ``A = L D L^T`` of the package factors; ``b (S, E, B)`` -> ``(S, E, B)`` (the scans of
    :func:`drbx.native.fci_perpendicular_plane_preconditioner.apply_core_schur_preconditioner` with no core coupling)."""
    _s, n_e, big_b, w, _ = lo.shape
    zero = jnp.zeros((n_e, w, big_b), lo.dtype)

    def fwd(carry, xs):
        lo_s, di, bs = xs
        ys = bs - jnp.einsum("pajb,pjb->pa", lo_s, carry)
        return jnp.concatenate([ys[:, None], carry[:, :-1]], axis=1), jnp.einsum("pab,pb->pa", di, ys)

    _, d = jax.lax.scan(fwd, zero, (lo, dinv, b))

    def bwd(carry, xs):
        lo_s, us = xs
        xs_ = us - carry[:, 0]
        upd = jnp.einsum("pajb,pa->pjb", lo_s, xs_)
        return jnp.concatenate([carry[:, 1:], zero[:, :1]], axis=1) + upd, xs_

    _, x = jax.lax.scan(bwd, zero, (lo, d), reverse=True)
    return x


def apply_zfree_preconditioner(prec: ZfreePreconditioner, r):
    """``M^-1 r`` plane by plane without ``Z`` (two ring solves); ``r`` flat plane-major."""
    meta = prec.meta
    n_s, big_b, n_e, nc, nr_ = meta.S, meta.B, prec.lo.shape[1], meta.Nc, meta.nr
    n_ring = meta.m * meta.N
    fd = prec.lo.dtype
    rp = r.reshape(n_e, nc + n_ring).astype(fd)
    rc = rp[:, :nc]
    pad = n_s * big_b - n_ring
    b = jnp.pad(rp[:, nc:], ((0, 0), (0, pad))).reshape(n_e, n_s, big_b).transpose(1, 0, 2)
    ax = _ring_solve(prec.lo, prec.dinv, b)                                   # A^-1 r_r  (S, E, B)
    axf = ax.transpose(1, 0, 2).reshape(n_e, n_s * big_b)
    zc = jnp.einsum("pcd,pd->pc", prec.sinv, rc - jnp.einsum("pnc,pn->pc", prec.mrc, axf[:, :nr_]))
    t = jnp.einsum("pnc,pc->pn", prec.mrc, zc)                               # M_rc z_c on the coupled rows
    b2 = jnp.pad(t, ((0, 0), (0, n_s * big_b - nr_))).reshape(n_e, n_s, big_b).transpose(1, 0, 2)
    x = (ax - _ring_solve(prec.lo, prec.dinv, b2)).transpose(1, 0, 2).reshape(n_e, n_s * big_b)[:, :n_ring]
    return jnp.concatenate([zc, x], axis=1).reshape(-1).astype(r.dtype)


def build_zfree_preconditioner(core, blocks: PlaneBlocks, nr: int | None = None) -> ZfreePreconditioner:
    """Z-free variant of a built :class:`CoreSchurPreconditioner` (the factors are shared, ``Z`` is not used)."""
    nr = blocks.coupled_ring_rows() if nr is None else nr
    mrc = blocks.mrc(nr).astype(_NP[DTYPE_NAME[core.meta.dtype]])
    meta = _ZMeta(S=core.meta.S, B=core.meta.B, w=core.meta.w, E=blocks.E, Nc=core.meta.Nc, m=core.meta.m, N=core.meta.N, nr=nr,
                  dtype=core.meta.dtype)
    prec = ZfreePreconditioner(core.lo, core.dinv, core.sinv, jnp.asarray(mrc), meta)
    prec.info.update(nr=nr, coupled_rings=nr / core.meta.N, mrc_density=float((mrc != 0).mean()), storage_bytes=prec.nbytes)
    return prec


# ---------------------------------------------------------------------------
# solvax block_jacobi (dense plane blocks)
# ---------------------------------------------------------------------------
def chunk_planes_for(P: int, E: int, budget_bytes: float = 2.0e9) -> int:
    return int(max(1, min(E, budget_bytes // (4 * 8 * P * P))))


def est_block_jacobi_bytes(E: int, P: int, dt: str) -> int:
    return int(E * P * P * np.dtype(_NP[dt]).itemsize + E * P * 4)


def build_block_jacobi(blocks: PlaneBlocks, dt: str, chunk: int | None = None):
    """``(modules, sizes)``: :func:`solvax.block_jacobi` of consecutive chunks of planes (bounded transient memory)."""
    chunk = chunk or chunk_planes_for(blocks.P, blocks.E)
    mods, sizes = [], []
    for p0 in range(0, blocks.E, chunk):
        p1 = min(p0 + chunk, blocks.E)
        stack = jnp.asarray(blocks.dense_stack(p0, p1, _NP[dt]))
        bj = solvax.block_jacobi(stack)
        del stack
        jax.block_until_ready(jax.tree_util.tree_leaves(bj))
        mods.append(bj)
        sizes.append(p1 - p0)
    return mods, tuple(sizes)


def apply_block_jacobi(mods, r, *, sizes, P, dt):
    outs, off = [], 0
    for mod, g in zip(mods, sizes):
        outs.append(mod(r[off * P:(off + g) * P].astype(_JNP[dt])))
        off += g
    return jnp.concatenate(outs).astype(r.dtype)


# ---------------------------------------------------------------------------
# solvax block_thomas + bordered core
# ---------------------------------------------------------------------------
def thomas_layout(blocks: PlaneBlocks) -> dict:
    rpb = blocks.ring_reach()                       # |i - j| <= reach  =>  super-ring distance <= 1 iff rpb >= reach
    n_s = -(-blocks.m // rpb)
    return dict(rpb=rpb, n_s=n_s, Bk=rpb * blocks.N, pad=n_s * rpb * blocks.N)


def est_block_thomas_bytes(E: int, Nc: int, m: int, N: int, reach: int, dt: str) -> int:
    rpb = max(reach, 1)
    n_s = -(-m // rpb)
    bk = rpb * N
    isz = np.dtype(_NP[dt]).itemsize
    return int(E * (3 * n_s * bk * bk + 2 * n_s * bk * Nc + Nc * Nc) * isz)


def build_block_thomas(blocks: PlaneBlocks, dt: str, chunk: int | None = None):
    """State ``(fac, ainv_b, slu, c_rows)`` stacked over the planes (host-concatenated chunks, then on the device)."""
    lay = thomas_layout(blocks)
    n_s, Bk, pad = lay["n_s"], lay["Bk"], lay["pad"]
    nc, m, N, P = blocks.nc, blocks.m, blocks.N, blocks.P
    mN = m * N
    npd = _NP[dt]
    chunk = chunk or max(1, min(blocks.E, int(1.5e9 // (8 * P * P * 4))))

    def build(lo, di, up, b_cols, c_rows, d):
        fac = solvax.block_thomas_factor(lo, di, up)

        def a_inv(r):
            return solvax.block_thomas_solve(fac, r.reshape(n_s, Bk)).reshape(-1)

        ainv_b = jax.vmap(a_inv, in_axes=1, out_axes=1)(b_cols)
        schur = c_rows @ ainv_b - d
        return fac, ainv_b, jax.scipy.linalg.lu_factor(schur)

    vbuild = jax.jit(jax.vmap(build))
    pieces = []
    for p0 in range(0, blocks.E, chunk):
        p1 = min(p0 + chunk, blocks.E)
        dense = blocks.dense_stack(p0, p1, np.float64)
        g = p1 - p0
        mrr = np.zeros((g, pad, pad))
        mrr[:, :mN, :mN] = dense[:, nc:, nc:]
        idx = np.arange(mN, pad)
        mrr[:, idx, idx] = 1.0
        mb = mrr.reshape(g, n_s, Bk, n_s, Bk)
        diag = np.stack([mb[:, s, :, s, :] for s in range(n_s)], 1)
        lower = np.stack([mb[:, s, :, s - 1, :] if s > 0 else np.zeros((g, Bk, Bk)) for s in range(n_s)], 1)
        upper = np.stack([mb[:, s, :, s + 1, :] if s < n_s - 1 else np.zeros((g, Bk, Bk)) for s in range(n_s)], 1)
        mrc = np.zeros((g, pad, nc))
        mrc[:, :mN] = dense[:, nc:, :nc]
        mcr = np.zeros((g, nc, pad))
        mcr[:, :, :mN] = dense[:, :nc, nc:]
        mcc = dense[:, :nc, :nc]
        del mrr, mb, dense
        args = tuple(jnp.asarray(a.astype(npd)) for a in (lower, diag, upper, mrc, mcr, mcc))
        fac, ainv_b, slu = vbuild(*args)
        jax.block_until_ready((fac, ainv_b, slu))
        pieces.append(jax.tree_util.tree_map(np.asarray, (fac, ainv_b, slu, args[4])))
        del args, fac, ainv_b, slu
    state = jax.tree_util.tree_map(lambda *xs: jnp.asarray(np.concatenate(xs, axis=0)), *pieces)
    jax.block_until_ready(state)
    return state, lay


def apply_block_thomas(state, r, *, nc, mN, n_s, Bk, dt):
    fac, ainv_b, slu, c_rows = state
    n_e = ainv_b.shape[0]
    pad = n_s * Bk
    rp = r.reshape(n_e, nc + mN).astype(_JNP[dt])
    rc = rp[:, :nc]
    rr = jnp.pad(rp[:, nc:], ((0, 0), (0, pad - mN)))

    def one(fac_, ainv_b_, slu_, c_rows_, rc_, rr_):
        ax = solvax.block_thomas_solve(fac_, rr_.reshape(n_s, Bk)).reshape(-1)
        y = lu_solve(slu_, c_rows_ @ ax - rc_)
        return y, ax - ainv_b_ @ y

    y, x = jax.vmap(one)(fac, ainv_b, slu, c_rows, rc, rr)
    return jnp.concatenate([y, x[:, :mN]], axis=1).reshape(-1).astype(r.dtype)


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------
def dense_reference(blocks: PlaneBlocks, rhs: np.ndarray) -> np.ndarray:
    """Exact ``blockdiag(M_k)^-1 rhs`` by dense float64 LU on the host (plane by plane)."""
    out = np.empty_like(rhs)
    P = blocks.P
    for p in range(blocks.E):
        out[p * P:(p + 1) * P] = np.linalg.solve(blocks.plane_dense(p), rhs[p * P:(p + 1) * P])
    return out


def rel_err(x, ref, P: int | None = None) -> tuple[float, float]:
    """``(global relative 2-norm error, largest per-plane relative error)``."""
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    g = float(np.linalg.norm(x - ref) / np.linalg.norm(ref))
    if P is None:
        return g, g
    d = (x - ref).reshape(-1, P)
    n = ref.reshape(-1, P)
    return g, float((np.linalg.norm(d, axis=1) / np.linalg.norm(n, axis=1)).max())
