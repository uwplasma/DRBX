"""Exact per-plane ring-block preconditioner for the P07 Dirichlet potential solve (block-Jacobi over eta planes).

The Dirichlet operator ``A`` of ``fci_perpendicular_p07_sparse`` couples the owners of one eta plane strongly (in-plane
radial/angular diffusion) and the planes only weakly.  The preconditioner solves every in-plane diagonal block
``A_kk`` *exactly* and ignores the inter-plane coupling, ``M^-1 = blockdiag_k(A_kk^-1)``.

Setup (host, once per matrix)
    Within a plane the rings (radial index) are merged in radial order into *super-rings* of at most ``B`` owners
    (``B`` = the largest number of owners of one ring in one plane, or ``max_block``).  The owners of a ring are ordered
    by an angular key.  The in-plane block then is block banded with ``S`` super-rings, block size ``B`` and block
    bandwidth ``w`` (the largest super-ring distance of an in-plane coupling, read from the matrix).  All planes share
    the partition (rings with fewer owners in a plane are padded with identity rows and no coupling).  Every plane is
    factorised by banded block Gaussian elimination without pivoting (vectorised over the planes) in float64 in LDU form;
    a ``ValueError`` is raised if the 1-norm condition number of a Schur-complement diagonal block exceeds ``cond_max``.
    The stored blocks are cast to ``factor_dtype`` (``"float32"`` halves the memory traffic of the apply and, on the real
    N32/48/64 exports, leaves the FGMRES iteration count unchanged).

Apply (JAX only, jittable, no callbacks)
    gather ``r`` into the padded ``(S, planes, B)`` layout, a forward ``lax.scan`` over super-rings and a backward scan
    (batched matmuls over the planes), scatter back.  :func:`apply_plane_preconditioner` takes the
    :class:`PlanePreconditioner` pytree as an *argument*, so the factor arrays are traced inputs and not compile-time
    constants of the jitted program.  Vectors are float64 on the outside, ``factor_dtype`` inside.

The arithmetic is the production port of ``scripts/p08_step5_local/preconditioners.plane_ring_block``.  Design notes:
``work/p08_step5_solver_studies_20261001/README.md``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

__all__ = ["PlanePreconditioner", "apply_plane_preconditioner", "build_plane_preconditioner", "owner_layout"]

_FACTOR_DTYPES = {"float64": np.float64, "float32": np.float32}


# ---------------------------------------------------------------------------
# owner ring / plane / angular key
# ---------------------------------------------------------------------------
from .owner_plane_layout import owner_layout


# ---------------------------------------------------------------------------
# the pytree
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Meta:
    """Static (hashable) part of a :class:`PlanePreconditioner`."""

    S: int
    B: int
    w: int
    P: int
    n_owners: int
    dtype: str


@jax.tree_util.register_pytree_node_class
@dataclass(eq=False)
class PlanePreconditioner:
    """Factors of the plane blocks (JAX pytree).

    Array leaves: ``lo`` ``(S, P, w, B, B)`` (block ``(s, s - w + j)``), ``up`` ``(S, P, w, B, B)`` (block
    ``(s, s + 1 + e)``), ``dinv`` ``(S, P, B, B)`` (inverse Schur-complement diagonal blocks), ``idx`` ``(S, P, B)``
    int32 (gather map, ``n_owners`` marks padding) and ``pos`` ``(n_owners,)`` int32 (scatter map).  Static metadata
    ``S, B, w, P, n_owners, dtype``.  ``info`` is a host dict (``S, B, w, storage_bytes, max_block_cond1,
    setup_seconds, ...``) that is not part of the pytree: it is empty on copies produced by jax transformations.
    """

    lo: Any
    up: Any
    dinv: Any
    idx: Any
    pos: Any
    meta: _Meta
    info: dict = field(default_factory=dict, compare=False, repr=False)

    def tree_flatten(self):
        return (self.lo, self.up, self.dinv, self.idx, self.pos), self.meta

    @classmethod
    def tree_unflatten(cls, meta, children):
        return cls(*children, meta, {})

    @property
    def S(self) -> int:
        return self.meta.S

    @property
    def B(self) -> int:
        return self.meta.B

    @property
    def w(self) -> int:
        return self.meta.w

    @property
    def P(self) -> int:
        return self.meta.P

    @property
    def n_owners(self) -> int:
        return self.meta.n_owners

    @property
    def dtype(self) -> str:
        return self.meta.dtype

    @property
    def nbytes(self) -> int:
        return int(sum(x.nbytes for x in (self.lo, self.up, self.dinv, self.idx, self.pos)))


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
def apply_plane_preconditioner(prec: PlanePreconditioner, r: jnp.ndarray) -> jnp.ndarray:
    """``z = M^-1 r`` (banded block-LDU solve of every plane block); jittable with ``prec`` as an argument.

    ``r`` (``(n_owners,)``, float64) is gathered into the padded layout and cast to the factor dtype, both scans run
    in that dtype, the result is scattered and cast back to the dtype of ``r``.
    """
    meta = prec.meta
    n_ps, w, big_b = meta.P, meta.w, meta.B
    fd = prec.lo.dtype
    b = jnp.concatenate([r.astype(fd), jnp.zeros((1,), fd)])[prec.idx]
    zero = jnp.zeros((n_ps, w, big_b), fd)

    def fwd(carry, xs):                                               # carry[:, j] = y_{s - w + j}
        lo, bs = xs
        ys = bs - jnp.einsum("pjab,pjb->pa", lo, carry)
        return jnp.concatenate([carry[:, 1:], ys[:, None]], axis=1), ys

    _, y = jax.lax.scan(fwd, zero, (prec.lo, b))

    def bwd(carry, xs):                                               # carry[:, e] = x_{s + 1 + e}
        up, di, ys = xs
        xs_ = jnp.einsum("pab,pb->pa", di, ys - jnp.einsum("pjab,pjb->pa", up, carry))
        return jnp.concatenate([xs_[:, None], carry[:, :-1]], axis=1), xs_

    _, x = jax.lax.scan(bwd, zero, (prec.up, prec.dinv, y), reverse=True)
    return x.reshape(-1)[prec.pos].astype(r.dtype)


# ---------------------------------------------------------------------------
# setup (host)
# ---------------------------------------------------------------------------
def _layout(a: sp.csr_matrix, owner_ring, owner_plane, owner_theta, max_block):
    """Super-ring partition and padded slot layout (see the module docstring)."""
    n = a.shape[0]
    arrs = []
    for name, v in (("owner_ring", owner_ring), ("owner_plane", owner_plane), ("owner_theta", owner_theta)):
        v = np.asarray(v).reshape(-1)
        if v.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {v.shape}")
        arrs.append(v)
    ring, plane, theta = arrs
    rings, ring_rank = np.unique(ring, return_inverse=True)
    planes, plane_rank = np.unique(plane, return_inverse=True)
    ring_rank, plane_rank = ring_rank.reshape(-1), plane_rank.reshape(-1)
    n_r, n_p = rings.size, planes.size
    key = ring_rank.astype(np.int64) * n_p + plane_rank
    order = np.lexsort((theta, key))                                   # by (ring, plane), then angular key
    cnt_flat = np.bincount(key, minlength=n_r * n_p)
    starts = np.cumsum(cnt_flat) - cnt_flat
    theta_pos = np.empty(n, dtype=np.int64)
    theta_pos[order] = np.arange(n) - starts[key[order]]
    counts = cnt_flat.reshape(n_r, n_p)
    cnt_ring = counts.max(axis=1)
    bmax = int(cnt_ring.max())
    block = bmax if max_block is None else int(max_block)
    if block < bmax:
        raise ValueError(f"max_block={block} is smaller than the largest ring ({bmax} owners in one plane)")
    super_of_ring = np.zeros(n_r, dtype=np.int64)
    ring_offset = np.zeros(n_r, dtype=np.int64)
    s_idx, used = 0, 0
    for r in range(n_r):
        c = int(cnt_ring[r])
        if used > 0 and used + c > block:
            s_idx, used = s_idx + 1, 0
        super_of_ring[r], ring_offset[r] = s_idx, used
        used += c
    n_s = s_idx + 1
    super_of = super_of_ring[ring_rank]
    slot = ring_offset[ring_rank] + theta_pos
    coo = a.tocoo()
    inplane = plane_rank[coo.row] == plane_rank[coo.col]
    ds = np.abs(super_of[coo.col[inplane]] - super_of[coo.row[inplane]])
    w = int(ds.max()) if ds.size else 0
    return dict(n=n, n_rings=int(n_r), n_planes=int(n_p), plane_rank=plane_rank, B=block, S=n_s, w=w,
                super_of=super_of, slot=slot)


def _factor_banded(f: np.ndarray, cond_max: float) -> float:
    """In-place block LDU (no pivoting) of block-banded ``f[s, p, d]`` = block ``(s, s + d - w)`` of plane ``p``.

    On return ``f[s, :, w]`` holds ``D_s^{-1}``, ``f[r, :, d < w]`` the multipliers ``L_{r,s} = A~_{r,s} D_s^{-1}`` and
    ``f[s, :, d > w]`` the upper blocks of the eliminated matrix.  Returns the largest 1-norm condition number of a
    diagonal block; raises ``ValueError`` above ``cond_max``.
    """
    n_s, _, n_d, _, _ = f.shape
    w = (n_d - 1) // 2
    cmax = 0.0
    for s in range(n_s):
        d = f[s, :, w]
        try:
            dinv = np.linalg.inv(d)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"singular Schur-complement diagonal block at super-ring {s}") from exc
        cond = float(np.max(np.abs(d).sum(axis=-2).max(axis=-1) * np.abs(dinv).sum(axis=-2).max(axis=-1)))
        if not np.isfinite(cond) or cond > cond_max:
            raise ValueError(f"ill-conditioned Schur-complement diagonal block at super-ring {s}: cond1 = {cond:.3e} "
                             f"> cond_max = {cond_max:.1e}")
        cmax = max(cmax, cond)
        f[s, :, w] = dinv
        top = min(s + w, n_s - 1)
        for r in range(s + 1, top + 1):
            f[r, :, s - r + w] = f[r, :, s - r + w] @ dinv
        for r in range(s + 1, top + 1):
            lrs = f[r, :, s - r + w]
            for c in range(s + 1, top + 1):
                f[r, :, c - r + w] -= lrs @ f[s, :, c - s + w]
    return cmax


def build_plane_preconditioner(matrix: sp.spmatrix, owner_ring: np.ndarray, owner_plane: np.ndarray,
                               owner_theta: np.ndarray, *, factor_dtype: str = "float32",
                               max_block: int | None = None, cond_max: float = 1e12) -> PlanePreconditioner:
    """Factorise the in-plane diagonal blocks of ``matrix`` (the P07 Dirichlet linear part, scipy CSR).

    ``owner_ring`` / ``owner_plane`` / ``owner_theta`` come from :func:`owner_layout`.  ``factor_dtype``
    (``"float32"`` or ``"float64"``) is the storage and apply dtype of the factors; the factorisation is float64.
    Needs ``jax_enable_x64`` (float64 vectors).  ``prec.info`` records ``n_owners, n_rings, n_planes, S, B, w,
    padded_fraction, storage_bytes, max_block_cond1, factor_dtype, setup_seconds`` and ``setup_breakdown_s``.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the plane preconditioner needs float64: set JAX_ENABLE_X64=true before importing jax")
    if factor_dtype not in _FACTOR_DTYPES:
        raise ValueError(f"factor_dtype must be one of {sorted(_FACTOR_DTYPES)}, got {factor_dtype!r}")
    fdt = _FACTOR_DTYPES[factor_dtype]
    t_start = time.perf_counter()
    a = sp.csr_matrix(matrix, dtype=np.float64)
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"matrix must be square, got {a.shape}")
    lay = _layout(a, owner_ring, owner_plane, owner_theta, max_block)
    t_layout = time.perf_counter() - t_start
    n, n_ps, n_s, big_b = lay["n"], lay["n_planes"], lay["S"], lay["B"]
    w = max(lay["w"], 1)
    plane_rank, super_of, slot = lay["plane_rank"], lay["super_of"], lay["slot"]

    t0 = time.perf_counter()
    idx = np.full((n_s, n_ps, big_b), n, dtype=np.int32)
    idx[super_of, plane_rank, slot] = np.arange(n, dtype=np.int32)
    occ = idx != n
    if int(occ.sum()) != n:
        raise ValueError("two owners share a (super-ring, plane, slot): inconsistent ring/theta data")
    pos = ((super_of * n_ps + plane_rank) * big_b + slot).astype(np.int32)
    coo = a.tocoo()
    keep = plane_rank[coo.row] == plane_rank[coo.col]
    rows, cols, vals = coo.row[keep], coo.col[keep], coo.data[keep]
    f = np.zeros((n_s, n_ps, 2 * w + 1, big_b, big_b))
    f[super_of[rows], plane_rank[rows], super_of[cols] - super_of[rows] + w, slot[rows], slot[cols]] = vals
    ps_, pp_, pb_ = np.nonzero(~occ)
    f[ps_, pp_, w, pb_, pb_] = 1.0                                       # padding: identity, no coupling
    t_assemble = time.perf_counter() - t0
    t0 = time.perf_counter()
    cond = _factor_banded(f, cond_max)
    t_factor = time.perf_counter() - t0
    t0 = time.perf_counter()
    lo = jnp.asarray(np.ascontiguousarray(f[:, :, :w], dtype=fdt))
    up = jnp.asarray(np.ascontiguousarray(f[:, :, w + 1:], dtype=fdt))
    dinv = jnp.asarray(np.ascontiguousarray(f[:, :, w], dtype=fdt))
    del f
    idx_j, pos_j = jnp.asarray(idx), jnp.asarray(pos)
    t_jax = time.perf_counter() - t0
    meta = _Meta(S=int(n_s), B=int(big_b), w=int(w), P=int(n_ps), n_owners=int(n), dtype=factor_dtype)
    prec = PlanePreconditioner(lo, up, dinv, idx_j, pos_j, meta)
    prec.info.update(
        n_owners=int(n), n_rings=lay["n_rings"], n_planes=int(n_ps), S=int(n_s), B=int(big_b), w=int(w),
        padded_fraction=1.0 - n / float(n_s * n_ps * big_b), factor_dtype=factor_dtype,
        storage_bytes=prec.nbytes, max_block_cond1=float(cond), setup_seconds=time.perf_counter() - t_start,
        setup_breakdown_s={"layout": t_layout, "assemble": t_assemble, "factor": t_factor, "to_jax": t_jax},
        params={"max_block": max_block, "cond_max": cond_max, "factor_dtype": factor_dtype})
    return prec
