"""Exact per-plane ring-block preconditioner for the Dirichlet perpendicular potential solve (block-Jacobi over eta planes).

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

The arithmetic is the production port of the nodal reference and plane-ring factor build step 5 ``plane_ring_block`` preconditioner.

Opt-in addition (the functions above are unchanged): :func:`build_core_schur_preconditioner` /
:func:`apply_core_schur_preconditioner` for the nodal SBP Laplacian, where every plane is a dense core followed by
equal rings; see the section "core / ring split" at the end of this module.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

__all__ = [
           "CoreSchurPreconditioner", "apply_core_schur_preconditioner", "build_core_schur_preconditioner"]

_FACTOR_DTYPES = {"float64": np.float64, "float32": np.float32}


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
    period_nfp: int = 1


@jax.tree_util.register_pytree_node_class
@dataclass(eq=False)
class CoreSchurPreconditioner:
    """Factors of the in-plane blocks ``[core, rings]`` of a symmetric positive definite matrix (JAX pytree).

    Per plane ``M_k = [[M_cc, M_cr], [M_rc, M_rr]]``.  The ring part ``M_rr`` (``m`` rings of ``N`` nodes, merged into ``S``
    super-rings of ``B = rings_per_block * N`` slots, block bandwidth ``w``) is factorised as ``L D L^T`` (``L`` unit block
    lower triangular, ``D`` block diagonal; only the lower blocks of ``M_rr`` are read, so the result is exactly symmetric).  With
    ``Y = L^-1 M_rc`` and ``Z = D^-1 Y`` the core Schur complement is ``S_c = M_cc - Y^T Z`` and
    ``M_k^-1 [r_c; r_r] = [z_c; L^-T (d - Z z_c)]`` with ``y = L^-1 r_r``, ``d = D^-1 y`` and ``z_c = S_c^-1 (r_c - Z^T y)``.

    Array leaves (``E`` stored planes; metadata ``E`` remains the logical plane count with period reuse): ``lo (S, E, B, w, B)`` with ``lo[s, k, :, e, :]`` = block ``(s, s - 1 - e)`` of ``L``;
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


def apply_core_schur_preconditioner(prec: CoreSchurPreconditioner, r: jnp.ndarray, *, plane_offset=0) -> jnp.ndarray:
    """Apply the plane factors, optionally reusing a checked field-period bank.

    ``meta.E`` is the global logical plane count; a periodic bank stores ``E / period_nfp`` planes. RHS periods are
    batched with ``vmap`` without expanding the stored factors. ``plane_offset`` is the global index of the first RHS
    plane (used inside eta ``shard_map``); partial periods are padded and discarded after the solve. Nonperiodic factors
    retain the original plane-local apply, including the existing ability to apply a sliced factor shard.
    """
    if prec.meta.period_nfp == 1:
        return _apply_core_schur_bank(prec, r)
    ep = prec.z.shape[0]
    plane_size = prec.meta.Nc + prec.meta.m * prec.meta.N
    if r.ndim != 1 or r.size % plane_size:
        raise ValueError("the RHS must contain whole planes in a flat vector")
    planes = r.size // plane_size
    groups = (planes + ep - 1) // ep
    index = (jnp.arange(ep) + plane_offset) % ep
    bank = CoreSchurPreconditioner(prec.lo[:, index], prec.dinv[:, index], prec.z[index], prec.sinv[index], prec.meta)
    rhs = jnp.pad(r, (0, groups * ep * plane_size - r.size)).reshape(groups, ep * plane_size)
    return jax.vmap(lambda v: _apply_core_schur_bank(bank, v))(rhs).reshape(-1)[:r.size]


def _apply_core_schur_bank(prec: CoreSchurPreconditioner, r: jnp.ndarray) -> jnp.ndarray:
    """Original plane-local scan kernel; plane count is read from the leaves for legacy factor shards."""
    meta = prec.meta
    n_s, big_b, w, n_e, n_c = meta.S, meta.B, meta.w, prec.z.shape[0], meta.Nc
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


def _build_core_schur_factors(blocks, *, n_planes: int, n_core: int, n_rings: int, ring_size: int,
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
    if min(nc, m, nn, e_all) < 1:
        raise ValueError("core size, radial rings, angular ring size and planes must be positive independently")
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


_PERIOD_REUSE_RTOL = 1e-12


def _check_period_matrices(blocks, n_planes: int, plane_size: int, period: int) -> float:
    """Check all full CSR plane matrices, retaining only the first period; return the largest relative max norm."""
    first, next_plane, largest = [], 0, 0.0
    for k0, k1, group in blocks:
        if k0 != next_plane or not k0 < k1 <= n_planes:
            raise ValueError("the plane groups must be consecutive and cover the requested planes")
        group = sp.csr_matrix(group, copy=True)
        if group.shape != ((k1 - k0) * plane_size,) * 2:
            raise ValueError("plane group has an incompatible shape")
        group.sum_duplicates()
        if not np.all(np.isfinite(group.data)):
            raise ValueError("plane matrices must be finite")
        for k in range(k0, k1):
            j = k - k0
            rows = group[j * plane_size:(j + 1) * plane_size]
            if rows.indices.size and (rows.indices.min() < j * plane_size or rows.indices.max() >= (j + 1) * plane_size):
                raise ValueError("the matrix couples different planes: pass the in-plane blocks")
            matrix = rows[:, j * plane_size:(j + 1) * plane_size]
            if k < period:
                first.append(matrix)
            else:
                reference = first[k % period]
                difference = matrix - reference
                norm = float(np.max(np.abs(reference.data), initial=0.0))
                error = float(np.max(np.abs(difference.data), initial=0.0)) / max(norm, 1e-12)
                largest = max(largest, error)
        next_plane = k1
    if next_plane != n_planes:
        raise ValueError(f"the plane groups cover {next_plane} planes, expected {n_planes}")
    return largest


def build_core_schur_preconditioner(blocks, *, n_planes: int, n_core: int, n_rings: int, ring_size: int,
                                    factor_dtype: str = "float32", rings_per_block: int | str = "auto", cond_max: float = 1e12,
                                    period_nfp: int = 1) -> CoreSchurPreconditioner:
    """Build core-Schur factors, optionally checking and reusing one field period.

    ``period_nfp=1`` (default) preserves the original build and storage. An integer > 1 requests reuse only if E is
    divisible by nfp and EVERY later plane matrix matches its first-period counterpart with relative max-entry norm
    <= 1e-12 (reference norm floored at 1e-12). Otherwise all original plane matrices are factorized, without reuse.
    ``info["period_reuse"]`` records the request, check, outcome and stored/logical plane counts; ``meta.period_nfp``
    records the effective factor-bank interpretation through JIT/pytree transformations.

    For reuse, ``blocks`` must be a deterministic, replayable zero-argument factory or re-iterable collection of
    ``(k0, k1, CSR)`` groups. One-shot iterators are rejected: silently caching the whole operator or synthesizing
    unverified fallback matrices would defeat bounded-memory setup or exact fallback. The check retains only the first
    period's CSR matrices. A second assembly pass factors only the first period on success, or all planes on fallback.
    Both passes must describe the same frozen operator; mutating inputs during a build is unsupported.
    """
    if isinstance(period_nfp, (bool, np.bool_)) or not isinstance(period_nfp, (int, np.integer)) or period_nfp < 1:
        raise ValueError("period_nfp must be a positive integer")
    period_nfp = int(period_nfp)
    start = time.perf_counter()
    reusable = period_nfp > 1 and n_planes % period_nfp == 0
    error = None
    reason = "disabled" if period_nfp == 1 else "plane_count_not_divisible"
    factory = blocks if callable(blocks) else lambda: iter(blocks)
    if reusable:
        if not callable(blocks) and iter(blocks) is blocks:
            raise ValueError("period reuse needs a replayable blocks factory or collection")
        error = _check_period_matrices(factory(), n_planes, n_core + n_rings * ring_size, n_planes // period_nfp)
        reusable = error <= _PERIOD_REUSE_RTOL
        reason = "verified" if reusable else "matrix_mismatch"
    check_seconds = time.perf_counter() - start
    stored = n_planes // period_nfp if reusable else n_planes

    def selected():
        for k0, k1, matrix in factory():
            if k0 >= stored:
                break
            stop = min(k1, stored)
            size = (stop - k0) * (n_core + n_rings * ring_size)
            yield k0, stop, matrix if stop == k1 else matrix[:size, :size]
            if stop == stored:
                break

    # When disabled or falling back, do not truncate: preserve the original builder's validation and arithmetic.
    factors = _build_core_schur_factors(selected() if reusable else factory(), n_planes=stored, n_core=n_core,
                                       n_rings=n_rings, ring_size=ring_size, factor_dtype=factor_dtype,
                                       rings_per_block=rings_per_block, cond_max=cond_max)
    factors.meta = replace(factors.meta, E=int(n_planes), period_nfp=period_nfp if reusable else 1)
    factors.info["n_owners"] = factors.n_owners
    factors.info["n_planes"] = int(n_planes)
    factors.info["period_reuse"] = dict(requested_nfp=period_nfp, active=reusable, reason=reason,
                                      logical_planes=int(n_planes), stored_planes=int(stored),
                                      matrix_rtol=_PERIOD_REUSE_RTOL, max_relative_difference=error,
                                      check_seconds=check_seconds)
    factors.info["params"]["period_nfp"] = period_nfp
    factors.info["setup_seconds"] = time.perf_counter() - start
    return factors
