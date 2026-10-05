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

Opt-in addition (the functions above are unchanged): :func:`build_core_schur_preconditioner` /
:func:`apply_core_schur_preconditioner` for the nodal SBP Laplacian, where every plane is a dense core followed by
equal rings; see the section "core / ring split" at the end of this module.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

__all__ = ["PlanePreconditioner", "apply_plane_preconditioner", "build_plane_preconditioner", "owner_layout",
           "CoreSchurPreconditioner", "apply_core_schur_preconditioner", "build_core_schur_preconditioner"]

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


# ---------------------------------------------------------------------------
# core / ring split (nodal SBP Laplacian): exact symmetric plane blocks with the dense core eliminated by a Schur complement
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _SchurMeta:
    """Static (hashable) part of a :class:`CoreSchurPreconditioner`."""

    S: int
    B: int
    w: int
    E: int
    Nc: int
    m: int
    N: int
    rings_per_block: int
    dtype: str


@jax.tree_util.register_pytree_node_class
@dataclass(eq=False)
class CoreSchurPreconditioner:
    """Factors of the in-plane blocks ``[core, rings]`` of a symmetric positive definite matrix (JAX pytree).

    Per plane ``M_k = [[M_cc, M_cr], [M_rc, M_rr]]``.  The ring part ``M_rr`` (``m`` rings of ``N`` nodes, merged into ``S``
    super-rings of ``B = rings_per_block * N`` slots, block bandwidth ``w``) is factorised as ``L D L^T`` (``L`` unit block
    lower triangular, ``D`` block diagonal; only the lower blocks of ``M_rr`` are read, so the result is exactly symmetric).  With
    ``Y = L^-1 M_rc`` and ``Z = D^-1 Y`` the core Schur complement is ``S_c = M_cc - Y^T Z`` and
    ``M_k^-1 [r_c; r_r] = [z_c; L^-T (d - Z z_c)]`` with ``y = L^-1 r_r``, ``d = D^-1 y`` and ``z_c = S_c^-1 (r_c - Z^T y)``.

    Array leaves (``E`` planes): ``lo (S, E, B, w, B)`` with ``lo[s, k, :, e, :]`` = block ``(s, s - 1 - e)`` of ``L``;
    ``dinv (S, E, B, B)`` (``D_s^-1``); ``z (E, S B, Nc)``; ``sinv (E, Nc, Nc)`` (``S_c^-1``).  ``info`` is a host dict that
    is not part of the pytree.
    """

    lo: Any
    dinv: Any
    z: Any
    sinv: Any
    meta: _SchurMeta
    info: dict = field(default_factory=dict, compare=False, repr=False)

    def tree_flatten(self):
        return (self.lo, self.dinv, self.z, self.sinv), self.meta

    @classmethod
    def tree_unflatten(cls, meta, children):
        return cls(*children, meta, {})

    @property
    def n_owners(self) -> int:
        return self.meta.E * (self.meta.Nc + self.meta.m * self.meta.N)

    @property
    def dtype(self) -> str:
        return self.meta.dtype

    @property
    def nbytes(self) -> int:
        return int(sum(x.nbytes for x in (self.lo, self.dinv, self.z, self.sinv)))


def apply_core_schur_preconditioner(prec: CoreSchurPreconditioner, r: jnp.ndarray) -> jnp.ndarray:
    """``z = M^-1 r`` plane by plane; jittable with ``prec`` as an argument.  ``r`` is the flat ``(E P,)`` vector (plane-major)."""
    meta = prec.meta
    n_s, big_b, w, n_e, n_c = meta.S, meta.B, meta.w, meta.E, meta.Nc
    n_ring = meta.m * meta.N
    fd = prec.lo.dtype
    rp = r.reshape(n_e, n_c + n_ring).astype(fd)
    rc = rp[:, :n_c]
    rr = jnp.pad(rp[:, n_c:], ((0, 0), (0, n_s * big_b - n_ring)))
    b = rr.reshape(n_e, n_s, big_b).transpose(1, 0, 2)                  # (S, E, B)

    def fwd(carry, xs):                                                 # carry[:, e] = y_{s - 1 - e}
        lo, di, bs = xs
        ys = bs - jnp.einsum("pajb,pjb->pa", lo, carry)
        return jnp.concatenate([ys[:, None], carry[:, :-1]], axis=1), (ys, jnp.einsum("pab,pb->pa", di, ys))

    zero = jnp.zeros((n_e, w, big_b), fd)
    _, (y, d) = jax.lax.scan(fwd, zero, (prec.lo, prec.dinv, b))        # y = L^-1 r_r, d = D^-1 y
    y, d = (v.transpose(1, 0, 2).reshape(n_e, n_s * big_b) for v in (y, d))
    zc = jnp.einsum("pcd,pd->pc", prec.sinv, rc - jnp.einsum("pnc,pn->pc", prec.z, y))
    u = (d - jnp.einsum("pnc,pc->pn", prec.z, zc)).reshape(n_e, n_s, big_b).transpose(1, 0, 2)

    def bwd(carry, xs):                                                 # carry[:, q] = pending update of row s - q
        lo, us = xs
        xs_ = us - carry[:, 0]
        upd = jnp.einsum("pajb,pa->pjb", lo, xs_)
        return jnp.concatenate([carry[:, 1:], zero[:, :1]], axis=1) + upd, xs_

    _, x = jax.lax.scan(bwd, zero, (prec.lo, u), reverse=True)
    x = x.transpose(1, 0, 2).reshape(n_e, n_s * big_b)[:, :n_ring]
    return jnp.concatenate([zc, x], axis=1).reshape(-1).astype(r.dtype)


def _sym_inverse(a: np.ndarray, what: str, cond_max: float) -> tuple[np.ndarray, float]:
    """Inverse of the symmetric part of the stacked matrices ``a (..., n, n)`` and the largest 1-norm condition number."""
    a = 0.5 * (a + np.swapaxes(a, -1, -2))
    try:
        ainv = np.linalg.inv(a)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"singular {what}") from exc
    ainv = 0.5 * (ainv + np.swapaxes(ainv, -1, -2))
    cond = float(np.max(np.abs(a).sum(axis=-2).max(axis=-1) * np.abs(ainv).sum(axis=-2).max(axis=-1)))
    if not np.isfinite(cond) or cond > cond_max:
        raise ValueError(f"ill-conditioned {what}: cond1 = {cond:.3e} > cond_max = {cond_max:.1e}")
    return ainv, cond


def _choose_rings_per_block(mat: sp.csr_matrix, n_core: int, n_rings: int, ring_size: int, max_rings: int = 8) -> int:
    """Rings per super-ring that minimise the stored ring factor ``S B^2 w`` (the apply is memory bound).

    The coupled ring pairs are read from the first plane of ``mat``; ``w`` is the exact block bandwidth of each candidate.
    """
    big_p = n_core + n_rings * ring_size
    row = np.repeat(np.arange(big_p), np.diff(mat.indptr[:big_p + 1]))
    col = mat.indices[:mat.indptr[big_p]].astype(np.int64)
    keep = (row >= n_core) & (col >= n_core) & (col < big_p)
    pairs = np.unique(((row[keep] - n_core) // ring_size) * n_rings + (col[keep] - n_core) // ring_size)
    ra, rb = pairs // n_rings, pairs % n_rings
    best, best_cost = 1, None
    for rpb in range(1, min(max_rings, n_rings) + 1):
        w = max(int(np.abs(ra // rpb - rb // rpb).max()) if pairs.size else 0, 1)
        n_s = -(-n_rings // rpb)
        cost = n_s * (rpb * ring_size) ** 2 * w
        if best_cost is None or cost < best_cost:
            best, best_cost = rpb, cost
    return best


def build_core_schur_preconditioner(blocks, *, n_planes: int, n_core: int, n_rings: int, ring_size: int,
                                    factor_dtype: str = "float32", rings_per_block: int | str = "auto", cond_max: float = 1e12
                                    ) -> CoreSchurPreconditioner:
    """Factorise the in-plane blocks of a symmetric positive definite matrix, group of planes by group of planes.

    ``blocks`` yields ``(k0, k1, M)`` for consecutive groups of planes (e.g. :func:`drbx.geometry.sbp_laplacian_assembly.
    iter_inplane_blocks`), ``M`` the scipy CSR in-plane matrix ``((k1 - k0) P, (k1 - k0) P)`` of the group with ``P = n_core +
    n_rings * ring_size`` and the nodes of one plane ordered ``[core, ring 0, ring 1, ...]`` (rings of ``ring_size`` nodes), so
    the host memory is bounded by one group.  ``rings_per_block`` rings form one super-ring of the banded ring factorisation
    (the last one is padded with identity rows); ``"auto"`` takes the value that minimises the stored ring factor ``S B^2 w``
    for the coupling pattern of the first plane (the apply is memory bound; N48: 3, N64: 2 on the HSX metrics).  The factorisation is float64 (the lower triangle of the ring blocks and the
    full core block are read); ``factor_dtype`` is the storage and apply dtype.  ``info`` records the layout, ``storage_bytes``,
    ``max_block_cond1`` (ring diagonal blocks and core Schur complement), ``setup_seconds`` and its breakdown.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the plane preconditioner needs float64: set JAX_ENABLE_X64=true before importing jax")
    if factor_dtype not in _FACTOR_DTYPES:
        raise ValueError(f"factor_dtype must be one of {sorted(_FACTOR_DTYPES)}, got {factor_dtype!r}")
    if rings_per_block != "auto" and not (isinstance(rings_per_block, (int, np.integer)) and rings_per_block >= 1):
        raise ValueError(f"rings_per_block must be a positive integer or 'auto', got {rings_per_block!r}")
    fdt = _FACTOR_DTYPES[factor_dtype]
    t_start = time.perf_counter()
    nc, m, nn, e_all = int(n_core), int(n_rings), int(ring_size), int(n_planes)
    big_p = nc + m * nn
    blocks = iter(blocks)
    first = next(blocks, None)
    if first is None:
        raise ValueError("no plane groups")
    if rings_per_block == "auto":
        rings_per_block = _choose_rings_per_block(sp.csr_matrix(first[2]), nc, m, nn)
    rings_per_block = int(rings_per_block)
    blocks = itertools.chain([first], blocks)
    big_b = rings_per_block * nn
    n_s = -(-m // rings_per_block)
    pad_len = n_s * big_b
    t = dict(read=0.0, factor=0.0, schur=0.0, store=0.0)
    out: dict[str, np.ndarray] = {}
    cmax, nnz_all, k_next = 0.0, 0, 0
    infos = []
    for k0, k1, mat in blocks:
        if k0 != k_next:
            raise ValueError(f"the plane groups must be consecutive: expected a group starting at {k_next}, got {k0}")
        g = k1 - k0
        mat = sp.csr_matrix(mat)
        if mat.shape != (g * big_p, g * big_p):
            raise ValueError(f"group {k0}:{k1} has shape {mat.shape}, expected {(g * big_p,) * 2}")
        mat.sum_duplicates()
        nnz_all += mat.nnz
        t0 = time.perf_counter()
        # --- read the blocks of every plane (lower triangle of the rings; the full core block and ring-core coupling)
        mcc = np.zeros((g, nc, nc))
        y = np.zeros((g, pad_len, nc))                                   # M_rc, then Y = L^-1 M_rc, then Z = D^-1 Y in place
        parts = []
        w_g = 0
        for p in range(g):
            a, b = mat.indptr[p * big_p], mat.indptr[(p + 1) * big_p]
            cnt = np.diff(mat.indptr[p * big_p:(p + 1) * big_p + 1])
            row = np.repeat(np.arange(big_p), cnt)
            col = mat.indices[a:b].astype(np.int64) - p * big_p
            if col.size and (col.min() < 0 or col.max() >= big_p):
                raise ValueError("the matrix couples different planes: pass the in-plane blocks")
            val = mat.data[a:b]
            cc = (row < nc) & (col < nc)
            mcc[p, row[cc], col[cc]] = val[cc]
            rc = (row >= nc) & (col < nc)
            y[p, row[rc] - nc, col[rc]] = val[rc]
            rr = (row >= nc) & (col >= nc)
            ra, ca = row[rr] - nc, col[rr] - nc
            sr, sc = ra // big_b, ca // big_b
            low = sr >= sc
            parts.append((sr[low], sc[low], ra[low] % big_b, ca[low] % big_b, val[rr][low]))
            if low.any():
                w_g = max(w_g, int((sr[low] - sc[low]).max()))
        w = max(w_g, 1)
        lowf = np.zeros((n_s, g, w + 1, big_b, big_b))                   # [s, p, d] = block (s, s - d); d = 0 diagonal
        for p, (sr, sc, ia, ib, val) in enumerate(parts):
            lowf[sr, p, sr - sc, ia, ib] = val
        parts.clear()
        pad_slots = np.arange(m * nn - (n_s - 1) * big_b, big_b)          # slots beyond the last ring: identity rows
        lowf[n_s - 1, :, 0, pad_slots, pad_slots] = 1.0
        t["read"] += time.perf_counter() - t0
        # --- banded block LDL^T of the rings (vectorised over the planes of the group)
        t0 = time.perf_counter()
        dinv = np.empty((n_s, g, big_b, big_b))
        for s in range(n_s):
            dinv[s], cnd = _sym_inverse(lowf[s, :, 0], f"ring diagonal block at super-ring {s}", cond_max)
            cmax = max(cmax, cnd)
            top = min(s + w, n_s - 1)
            at = {r: lowf[r, :, r - s].copy() for r in range(s + 1, top + 1)}
            lm = {r: at[r] @ dinv[s] for r in at}
            for r in at:
                for c in range(s + 1, r):
                    lowf[r, :, r - c] -= lm[r] @ at[c].transpose(0, 2, 1)
                lowf[r, :, 0] -= lm[r] @ at[r].transpose(0, 2, 1)
                lowf[r, :, r - s] = lm[r]
        t["factor"] += time.perf_counter() - t0
        # --- Y = L^-1 M_rc and the core Schur complement S_c = M_cc - Y^T D^-1 Y
        t0 = time.perf_counter()
        yv = y.reshape(g, n_s, big_b, nc)
        for s in range(n_s):
            for e in range(min(w, s)):
                yv[:, s] -= lowf[s, :, e + 1] @ yv[:, s - 1 - e]
        sc_mat = mcc.copy()
        for s in range(n_s):
            ys = yv[:, s].copy()
            yv[:, s] = dinv[s] @ ys                                       # Z = D^-1 Y, in place
            sc_mat -= ys.transpose(0, 2, 1) @ yv[:, s]
        sinv, cnd = _sym_inverse(sc_mat, f"core Schur complement of planes {k0}..{k1 - 1}", cond_max)
        cmax = max(cmax, cnd)
        t["schur"] += time.perf_counter() - t0
        # --- store (group results cast to the factor dtype straight into the final arrays)
        t0 = time.perf_counter()
        if not out:
            out = dict(lo=np.zeros((n_s, e_all, big_b, w, big_b), fdt), dinv=np.empty((n_s, e_all, big_b, big_b), fdt),
                       z=np.empty((e_all, pad_len, nc), fdt), sinv=np.empty((e_all, nc, nc), fdt))
        elif w > out["lo"].shape[3]:                                      # a later group is wider: widen the stored band
            old = out["lo"]
            out["lo"] = np.zeros(old.shape[:3] + (w, big_b), fdt)
            out["lo"][:, :, :, :old.shape[3]] = old
        lo_g = lowf[:, :, 1:].transpose(0, 1, 3, 2, 4)                    # (S, g, B, w, B)
        out["lo"][:, k0:k1, :, :w] = lo_g
        out["dinv"][:, k0:k1] = dinv
        out["z"][k0:k1] = y
        out["sinv"][k0:k1] = sinv
        t["store"] += time.perf_counter() - t0
        infos.append(dict(planes=g, nnz=int(mat.nnz), w=w_g))
        del mat, mcc, y, yv, lowf, dinv, sc_mat, sinv
        k_next = k1
    if k_next != e_all:
        raise ValueError(f"the plane groups cover {k_next} planes, expected {e_all}")
    t0 = time.perf_counter()
    w_store = out["lo"].shape[3]
    prec = CoreSchurPreconditioner(jnp.asarray(out["lo"]), jnp.asarray(out["dinv"]), jnp.asarray(out["z"]),
                                   jnp.asarray(out["sinv"]),
                                   _SchurMeta(S=n_s, B=big_b, w=w_store, E=e_all, Nc=nc, m=m, N=nn,
                                              rings_per_block=rings_per_block, dtype=factor_dtype))
    del out
    t["to_jax"] = time.perf_counter() - t0
    prec.info.update(
        n_owners=e_all * big_p, n_planes=e_all, n_core=nc, n_rings=m, ring_size=nn, S=n_s, B=big_b, w=w_store,
        rings_per_block=rings_per_block, padded_fraction=1.0 - m * nn / float(pad_len), factor_dtype=factor_dtype,
        storage_bytes=prec.nbytes, max_block_cond1=float(cmax), nnz=nnz_all, groups=infos,
        setup_seconds=time.perf_counter() - t_start, setup_breakdown_s=t,
        params={"rings_per_block": rings_per_block, "cond_max": cond_max, "factor_dtype": factor_dtype})
    return prec
