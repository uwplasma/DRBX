"""Preconditioner candidates for the P07 Dirichlet potential solve (P08 step 5, local solver studies).

Every builder takes ``(op, *, owner_plane=None, **params)`` -- ``op`` is a ``kind == "dirichlet"``
:class:`~drbx.native.fci_perpendicular_p07_sparse.P07SparseOperator`, ``owner_plane`` the eta-plane index of every
owner -- and returns a :class:`Preconditioner` whose ``apply`` (``r -> M^{-1} r`` on float64 vectors) can be passed as the
``preconditioner`` argument of :func:`drbx.native.fci_perpendicular_p07_solve.solve_p07_dirichlet_jit` (a static jit
argument, so ``apply`` is hashable by identity and its data enter the compiled program as constants).

====================  =====================================================================  ==================
builder               what it is                                                             ``jax_native``
====================  =====================================================================  ==================
:func:`jacobi`        ``r / diag(A)``                                                        yes
:func:`chebyshev`     fixed-degree Chebyshev polynomial in ``D^-1 A`` (BCSR matvecs)         yes
:func:`plane_block`   exact sparse LU of every eta-plane diagonal block (host)               no (pure_callback)
:func:`smoothed_aggregation`  algebraic multigrid V-cycle (host setup, JAX apply)            yes
:func:`ilu`           SuperLU incomplete LU (host, sequential triangular solves)             no (pure_callback)
:func:`plane_ring_block`  exact per-plane solve: banded block LU over radial super-rings         yes
:func:`plane_bgs_forward`, :func:`plane_bgs_multicolor`  block Gauss-Seidel across eta planes     yes (``plane_solver="host"``: no)
====================  =====================================================================  ==================

The ring/plane family (:func:`plane_ring_block`, :func:`plane_bgs_forward`, :func:`plane_bgs_multicolor`) additionally
takes ``owner_ring`` (radial index) and ``owner_theta`` (any key whose order within a ring and plane is the angular order,
e.g. the smallest raw theta index of the owner); see the section "ring/plane block solvers" below.

All of them are *fixed linear operators* (no data-dependent branching, no inner iteration to a tolerance), so plain right
preconditioning is valid; the flexible GMRES of solvax tolerates them as well.

``setup_seconds`` is the host wall time of the build (including the conversion of the data to JAX arrays); ``nbytes`` is the
approximate size of the data held by the preconditioner (BCSR counted as float64 data + int32 indices; note that the
Chebyshev and multigrid candidates hold their own copy of the matrix levels).
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jax.experimental import sparse as jsparse

__all__ = ["Preconditioner", "jacobi", "chebyshev", "plane_block", "smoothed_aggregation", "ilu",
           "power_iteration_rho", "chebyshev_interval", "build_hierarchy", "RingLayout", "ring_layout",
           "plane_ring_block", "plane_bgs_forward", "plane_bgs_multicolor", "make_solver"]


@dataclass
class Preconditioner:
    """A built preconditioner; ``apply`` is traceable (``r -> z``, float64 vectors of length ``n``).

    Optionally ``data`` (a pytree of JAX arrays) and ``apply_with(data, r)`` give the same operator with its large arrays
    as an *argument*: ``apply(r) == apply_with(data, r)``.  :func:`make_solver` passes ``data`` as a jit argument, so the
    arrays are not embedded in the compiled program as constants (warm-up = compile of the program only).
    """

    name: str
    apply: Callable[[jnp.ndarray], jnp.ndarray]
    setup_seconds: float = 0.0
    nbytes: int = 0
    jax_native: bool = True
    info: dict = field(default_factory=dict)
    data: Any = None                                               # optional pytree of arrays (see ``apply_with``)
    apply_with: Callable[[Any, jnp.ndarray], jnp.ndarray] | None = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _timed(build: Callable[..., Preconditioner]) -> Callable[..., Preconditioner]:
    """Fill ``setup_seconds`` with the wall time of the whole builder call."""

    @functools.wraps(build)
    def wrapper(*args, **kwargs) -> Preconditioner:
        t0 = time.perf_counter()
        prec = build(*args, **kwargs)
        prec.setup_seconds = time.perf_counter() - t0
        return prec

    return wrapper


def _require_x64() -> None:
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the P07 preconditioners need float64: set JAX_ENABLE_X64=true before importing jax")


def _host_csr(op: Any) -> sp.csr_matrix:
    csr = sp.csr_matrix(op.matrix, dtype=np.float64)
    csr.sort_indices()
    if csr.nnz < 2**31 - 1:
        csr.indices = csr.indices.astype(np.int32)
        csr.indptr = csr.indptr.astype(np.int32)
    return csr


def _bcsr(csr: sp.spmatrix) -> jsparse.BCSR:
    csr = sp.csr_matrix(csr, dtype=np.float64)
    if csr.nnz < 2**31 - 1:
        csr.indices = csr.indices.astype(np.int32)
        csr.indptr = csr.indptr.astype(np.int32)
    return jsparse.BCSR.from_scipy_sparse(csr)


def _bcsr_nbytes(m: jsparse.BCSR) -> int:
    return int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes)


def _lu_nnz(lu: Any) -> int:
    return int(lu.L.nnz + lu.U.nnz)


def _host_apply(fn: Callable[[np.ndarray], np.ndarray], n: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Wrap a numpy function ``r -> z`` as a jittable apply through ``jax.pure_callback``."""
    spec = jax.ShapeDtypeStruct((n,), jnp.float64)

    def host(r):
        return np.ascontiguousarray(fn(np.asarray(r, dtype=np.float64)), dtype=np.float64)

    def apply(r: jnp.ndarray) -> jnp.ndarray:
        return jax.pure_callback(host, spec, r, vmap_method="sequential")

    return apply


def power_iteration_rho(matrix: sp.spmatrix, dinv: np.ndarray, iters: int = 30, seed: int = 0) -> float:
    """Spectral-radius estimate of ``D^-1 A`` by power iteration (host).

    Returns the largest ``||B x_k|| / ||x_k||`` seen (``B = D^-1 A``, unit-norm iterates); this is a lower bound of
    ``||B||_2`` and converges from below to ``rho(B)`` for a dominant real eigenvalue.
    """
    n = matrix.shape[0]
    x = np.random.default_rng(seed).standard_normal(n)
    x /= np.linalg.norm(x)
    rho = 0.0
    for _ in range(max(int(iters), 1)):
        y = dinv * (matrix @ x)
        norm = float(np.linalg.norm(y))
        if norm == 0.0:
            break
        rho = max(rho, norm)
        x = y / norm
    return rho


def chebyshev_interval(lam_max: float, lower_ratio: float) -> tuple[float, float]:
    """Chebyshev interval ``[lam_max / lower_ratio, 1.1 lam_max]``."""
    return lam_max / float(lower_ratio), 1.1 * lam_max


def _cheb_apply(matvec: Callable, dinv: jnp.ndarray, rhs: jnp.ndarray, degree: int, lo: float, hi: float):
    """``degree`` steps of the Chebyshev iteration for ``(D^-1 A) z = D^-1 rhs`` from ``z0 = 0`` (Saad, Alg. 12.1).

    The result is a fixed polynomial of degree ``degree - 1`` in ``D^-1 A`` applied to ``D^-1 rhs`` (``degree - 1``
    matvecs; the matvec that would update the residual after the last step is skipped).  ``degree = 1`` is the scaled
    Jacobi step ``D^-1 rhs / theta`` with ``theta = (lo + hi) / 2``.
    """
    theta, delta = 0.5 * (hi + lo), 0.5 * (hi - lo)
    sigma = theta / delta
    rho = 1.0 / sigma
    res = dinv * rhs
    d = res / theta
    z = d
    for _ in range(1, int(degree)):
        res = res - dinv * matvec(d)
        rho_new = 1.0 / (2.0 * sigma - rho)
        d = rho_new * rho * d + (2.0 * rho_new / delta) * res
        z = z + d
        rho = rho_new
    return z


# ---------------------------------------------------------------------------
# 1. Jacobi
# ---------------------------------------------------------------------------
@_timed
def jacobi(op: Any, *, owner_plane: np.ndarray | None = None) -> Preconditioner:
    """Diagonal inverse ``z = r / diag(A)``."""
    _require_x64()
    diag = jnp.asarray(sp.csr_matrix(op.matrix).diagonal(), dtype=jnp.float64)
    if not bool(jnp.all(diag > 0)):
        raise ValueError("operator diagonal must be strictly positive")
    return Preconditioner("jacobi", lambda r: r / diag, nbytes=int(diag.nbytes), jax_native=True,
                          info={"n": int(diag.shape[0])})


# ---------------------------------------------------------------------------
# 2. Chebyshev polynomial
# ---------------------------------------------------------------------------
@_timed
def chebyshev(op: Any, *, owner_plane: np.ndarray | None = None, degree: int = 4, lower_ratio: float = 30.0,
              power_iters: int = 30) -> Preconditioner:
    """Chebyshev polynomial preconditioner for ``D^-1 A`` (``D = diag A``).

    ``lambda_max`` of ``D^-1 A`` comes from ``power_iters`` power iterations (host); the polynomial is the ``degree``-step
    Chebyshev iteration for ``D^-1 A z = D^-1 r`` from ``z0 = 0`` on the interval ``[lambda_max / lower_ratio,
    1.1 lambda_max]``.  The degree is fixed, so the map ``r -> z`` is a fixed linear operator (a polynomial in ``D^-1 A``
    times ``D^-1``): plain right preconditioning is valid and flexible GMRES tolerates it.  ``degree - 1`` matvecs per
    application; ``degree = 1`` is a scaled Jacobi.
    """
    _require_x64()
    if degree < 1:
        raise ValueError("degree must be >= 1")
    a = _host_csr(op)
    diag = a.diagonal()
    if not np.all(diag > 0):
        raise ValueError("operator diagonal must be strictly positive")
    dinv_np = 1.0 / diag
    lam_max = power_iteration_rho(a, dinv_np, power_iters)
    lo, hi = chebyshev_interval(lam_max, lower_ratio)
    mat, dinv = _bcsr(a), jnp.asarray(dinv_np)
    deg = int(degree)

    def apply(r: jnp.ndarray) -> jnp.ndarray:
        return _cheb_apply(lambda v: mat @ v, dinv, r, deg, lo, hi)

    return Preconditioner(f"chebyshev{deg}", apply, nbytes=_bcsr_nbytes(mat) + int(dinv.nbytes), jax_native=True,
                          info={"degree": deg, "matvecs_per_apply": deg - 1, "lambda_max": lam_max, "lower_ratio": lower_ratio,
                                "interval": [lo, hi], "power_iters": power_iters})


# ---------------------------------------------------------------------------
# 3. eta-plane block solves
# ---------------------------------------------------------------------------
@_timed
def plane_block(op: Any, *, owner_plane: np.ndarray | None = None) -> Preconditioner:
    """Block-Jacobi over eta planes: exact sparse LU (SuperLU) of ``A[plane_k, plane_k]`` for every plane ``k``."""
    if owner_plane is None:
        raise ValueError("plane_block needs owner_plane (eta-plane index of every owner)")
    a = sp.csr_matrix(op.matrix, dtype=np.float64)
    n = a.shape[0]
    plane = np.asarray(owner_plane).reshape(-1)
    if plane.shape != (n,):
        raise ValueError(f"owner_plane must have shape ({n},), got {plane.shape}")
    order = np.argsort(plane, kind="stable")
    sorted_plane = plane[order]
    keys, starts = np.unique(sorted_plane, return_index=True)
    bounds = np.append(starts, n)
    blocks = []
    for j in range(len(keys)):
        idx = order[bounds[j]:bounds[j + 1]]
        blocks.append((int(bounds[j]), int(bounds[j + 1]), spla.splu(sp.csc_matrix(a[idx][:, idx]))))

    def solve(r: np.ndarray) -> np.ndarray:
        rp = r[order]
        zp = np.empty_like(rp)
        for lo, hi, lu in blocks:
            zp[lo:hi] = lu.solve(rp[lo:hi])
        z = np.empty_like(zp)
        z[order] = zp
        return z

    lu_nnz = sum(_lu_nnz(lu) for _, _, lu in blocks)
    sizes = [hi - lo for lo, hi, _ in blocks]
    return Preconditioner("plane_block", _host_apply(solve, n), nbytes=int(lu_nnz * 12 + n * 8), jax_native=False,
                          info={"n_planes": len(blocks), "block_sizes": sizes, "total_lu_nnz": int(lu_nnz),
                                "block_size_min": int(min(sizes)), "block_size_max": int(max(sizes))})


# ---------------------------------------------------------------------------
# 4. smoothed-aggregation algebraic multigrid
# ---------------------------------------------------------------------------
def _strong_graph(a: sp.csr_matrix, theta: float) -> sp.csr_matrix:
    """Strong-connection graph ``|s_ij| >= theta sqrt(|a_ii a_jj|)`` (``i != j``) of ``S = (|A| + |A|^T) / 2``.

    The returned CSR carries the strengths ``|s_ij|`` as data (symmetric pattern).
    """
    absa = abs(a)
    s = ((absa + absa.T) * 0.5).tocoo()
    d = np.abs(a.diagonal())
    keep = (s.row != s.col) & (s.data >= theta * np.sqrt(d[s.row] * d[s.col]))
    g = sp.csr_matrix((s.data[keep], (s.row[keep], s.col[keep])), shape=a.shape)
    g.sort_indices()
    return g


def _aggregate(strong: sp.csr_matrix) -> tuple[np.ndarray, int]:
    """Greedy aggregation on the strong graph; returns ``(aggregate_index_per_node, n_aggregates)``.

    Pass 1 seeds an aggregate at every unaggregated node whose strong neighbours are all unaggregated (the aggregate is the
    seed plus all its strong neighbours; a node without strong neighbours becomes a singleton).  Pass 2 (vectorised, uses
    the pass-1 snapshot) attaches each remaining node to the aggregate of its strongest aggregated neighbour.  Pass 3
    turns whatever is left (cannot occur after pass 2 for a symmetric graph, kept as a safeguard) into singletons.
    """
    n = strong.shape[0]
    indptr, indices, data = strong.indptr, strong.indices, strong.data
    agg = np.full(n, -1, dtype=np.int64)
    count = 0
    for i in range(n):
        if agg[i] >= 0:
            continue
        nb = indices[indptr[i]:indptr[i + 1]]
        if nb.size and (agg[nb] >= 0).any():
            continue
        agg[i] = count
        agg[nb] = count
        count += 1
    rest = np.flatnonzero(agg < 0)
    if rest.size:
        snap = agg.copy()
        counts = indptr[rest + 1] - indptr[rest]
        total = int(counts.sum())
        if total:
            rows = np.repeat(rest, counts)
            offs = np.repeat(np.cumsum(counts) - counts, counts)
            pos = np.repeat(indptr[rest], counts) + (np.arange(total) - offs)
            cols, w = indices[pos], np.abs(data[pos])
            ok = snap[cols] >= 0
            rows, cols, w = rows[ok], cols[ok], w[ok]
            if rows.size:
                order = np.lexsort((-w, rows))
                rows, cols = rows[order], cols[order]
                first = np.concatenate(([True], rows[1:] != rows[:-1]))
                agg[rows[first]] = snap[cols[first]]
    left = np.flatnonzero(agg < 0)
    if left.size:
        agg[left] = count + np.arange(left.size)
        count += left.size
    return agg, int(count)


def _filtered(a: sp.csr_matrix, strong: sp.csr_matrix) -> sp.csr_matrix:
    """``A`` restricted to its strong off-diagonal pattern; weak entries are lumped onto the diagonal (row sums kept)."""
    pattern = strong.copy()
    pattern.data = np.ones_like(pattern.data)
    off = a - sp.diags(a.diagonal())
    kept = off.multiply(pattern).tocsr()
    lumped = np.asarray(off.sum(axis=1)).ravel() - np.asarray(kept.sum(axis=1)).ravel()
    diag = a.diagonal() + lumped
    diag = np.where(diag > 0, diag, a.diagonal())
    return (kept + sp.diags(diag)).tocsr()


@dataclass
class _HostLevel:
    a: sp.csr_matrix
    p: sp.csr_matrix | None = None
    r: sp.csr_matrix | None = None
    dinv: np.ndarray | None = None
    rho: float = 0.0
    omega: float = 0.0


def build_hierarchy(a: sp.csr_matrix, *, strength: float = 0.08, max_coarse: int = 1500, max_levels: int = 10,
                    prolongator_smoothing: bool = True, filter_smoothing: bool = False, power_iters: int = 30,
                    timings: dict | None = None) -> list[_HostLevel]:
    """Host smoothed-aggregation hierarchy of ``a`` (levels fine to coarse; the last one has ``p = r = None``)."""
    tm = timings if timings is not None else {}

    def tick(key: str, t0: float) -> None:
        tm[key] = tm.get(key, 0.0) + time.perf_counter() - t0

    levels: list[_HostLevel] = []
    cur = sp.csr_matrix(a)
    while True:
        n = cur.shape[0]
        diag = cur.diagonal()
        if not np.all(diag > 0):
            raise ValueError(
                f"level {len(levels)} has a non-positive diagonal (min {diag.min():.3e}); "
                "try volume_scaling=True (M-symmetric working matrix)")
        t0 = time.perf_counter()
        dinv = 1.0 / diag
        rho = power_iteration_rho(cur, dinv, power_iters)
        tick("power_iteration", t0)
        lvl = _HostLevel(cur, dinv=dinv, rho=rho)
        levels.append(lvl)
        if n <= max_coarse or len(levels) >= max_levels:
            break
        t0 = time.perf_counter()
        strong = _strong_graph(cur, strength)
        tick("strength", t0)
        t0 = time.perf_counter()
        agg, nc = _aggregate(strong)
        tick("aggregation", t0)
        if nc >= 0.9 * n or nc < 1:
            break                                              # coarsening stalled: stop at this level
        t0 = time.perf_counter()
        sizes = np.bincount(agg, minlength=nc).astype(np.float64)
        p = sp.csr_matrix((1.0 / np.sqrt(sizes[agg]), (np.arange(n), agg)), shape=(n, nc))
        if prolongator_smoothing:
            if filter_smoothing:
                asm = _filtered(cur, strong)
                dsm = 1.0 / asm.diagonal()
                rho_s = power_iteration_rho(asm, dsm, power_iters)
            else:
                asm, dsm, rho_s = cur, dinv, rho
            omega_p = 4.0 / (3.0 * rho_s)
            p = (p - omega_p * sp.diags(dsm) @ (asm @ p)).tocsr()
            p.eliminate_zeros()
        p.sort_indices()
        tick("prolongator", t0)
        t0 = time.perf_counter()
        r = p.T.tocsr()
        ac = (r @ (cur @ p)).tocsr()
        ac.eliminate_zeros()
        ac.sort_indices()
        tick("galerkin", t0)
        lvl.p, lvl.r = p, r
        cur = ac
    return levels


@_timed
def smoothed_aggregation(op: Any, *, owner_plane: np.ndarray | None = None, strength: float = 0.08,
                         max_coarse: int = 1500, smoother: str = "jacobi", omega: float | None = None,
                         presmooth: int = 1, postsmooth: int = 1, cheb_degree: int = 2, cheb_lower_ratio: float = 10.0,
                         prolongator_smoothing: bool = True, max_levels: int = 10, power_iters: int = 30,
                         volume_scaling: bool = False, filter_smoothing: bool = False,
                         keep_host: bool = False) -> Preconditioner:
    """Smoothed-aggregation algebraic multigrid V-cycle (host setup with scipy, JAX apply).

    Setup (see :func:`_strong_graph`, :func:`_aggregate`): strength of connection on the symmetric part, greedy
    aggregation, piecewise-constant tentative prolongator (columns normalised), smoothed prolongator
    ``P = (I - omega_p D^-1 A) P0`` with ``omega_p = 4 / (3 rho(D^-1 A))`` (``prolongator_smoothing``), ``R = P^T``,
    Galerkin ``A_c = R A P``; recursion until ``n <= max_coarse`` or ``max_levels`` levels (or a stalled coarsening);
    the coarsest level is solved by dense LU.

    V-cycle: ``presmooth`` / ``postsmooth`` sweeps of ``smoother``: ``"jacobi"`` (damped, ``omega`` or per level
    ``4 / (3 rho)``) or ``"chebyshev"`` (``cheb_degree`` steps on ``[rho / cheb_lower_ratio, 1.1 rho]``, rho by power
    iteration).  The first presmoothing sweep starts from zero; the cycle is a fixed linear operator.

    Optional extensions, both off by default (spec-conform): ``volume_scaling`` builds the hierarchy for the
    M-symmetrised matrix ``M^{1/2} A M^{-1/2}`` (``M = diag(V)``; the operator is ``M^-1 K`` with ``K`` nearly
    symmetric, so Galerkin ``P^T A P`` on the plain matrix is not the natural coarse operator when ``V`` varies) and wraps
    the cycle as ``z = M^{-1/2} cycle(M^{1/2} r)``; ``filter_smoothing`` smooths the prolongator with ``A`` filtered to
    its strong pattern (weak entries lumped to the diagonal), which keeps ``P`` and the coarse operators sparser.
    ``keep_host`` stores the scipy hierarchy under ``info["_host"]`` (for tests; memory-heavy, off by default).
    """
    _require_x64()
    if smoother not in ("jacobi", "chebyshev"):
        raise ValueError(f"smoother must be 'jacobi' or 'chebyshev', got {smoother!r}")
    if presmooth < 0 or postsmooth < 0 or cheb_degree < 1:
        raise ValueError("presmooth/postsmooth must be >= 0 and cheb_degree >= 1")
    timings: dict = {}
    t0 = time.perf_counter()
    a = _host_csr(op)
    n = a.shape[0]
    scale = None
    if volume_scaling:
        vol = np.asarray(op.owner_volume, dtype=np.float64).reshape(-1)
        scale = np.sqrt(vol)
        a = (sp.diags(scale) @ a @ sp.diags(1.0 / scale)).tocsr()
        a.sort_indices()
    timings["prepare"] = time.perf_counter() - t0
    nnz0 = int(a.nnz)

    levels = build_hierarchy(a, strength=strength, max_coarse=max_coarse, max_levels=max_levels,
                             prolongator_smoothing=prolongator_smoothing, filter_smoothing=filter_smoothing,
                             power_iters=power_iters, timings=timings)
    t0 = time.perf_counter()
    jlev = []
    nbytes = 0
    for lvl in levels[:-1]:
        mats = (_bcsr(lvl.a), _bcsr(lvl.p), _bcsr(lvl.r))
        dinv = jnp.asarray(lvl.dinv)
        if smoother == "jacobi":
            w = float(omega) if omega is not None else 4.0 / (3.0 * lvl.rho)
            lvl.omega = w
            interval = None
        else:
            interval = chebyshev_interval(lvl.rho, cheb_lower_ratio)
            w = 0.0
        jlev.append((*mats, dinv, w, interval))
        nbytes += sum(_bcsr_nbytes(m) for m in mats) + int(dinv.nbytes)
    coarse = levels[-1]
    nc = coarse.a.shape[0]
    if nc > 10000:
        raise ValueError(f"coarsest level has {nc} unknowns (> 10000): dense LU not sensible; "
                         "raise max_coarse/max_levels or the strength threshold")
    lu_piv = sla.lu_factor(coarse.a.toarray())
    lu_dev = (jnp.asarray(lu_piv[0]), jnp.asarray(lu_piv[1], dtype=jnp.int32))
    nbytes += int(lu_dev[0].nbytes + lu_dev[1].nbytes)
    timings["to_jax_and_lu"] = time.perf_counter() - t0
    n_lev = len(levels)
    sm = smoother
    cdeg = int(cheb_degree)
    npre, npost = int(presmooth), int(postsmooth)
    svec = None if scale is None else (jnp.asarray(scale), jnp.asarray(1.0 / scale))

    def smooth_step(lv, b, x):
        """One sweep ``x + S (b - A x)``; ``x is None`` means ``x = 0``."""
        mat, _, _, dinv, w, interval = lv
        res = b if x is None else b - mat @ x
        if sm == "jacobi":
            corr = w * dinv * res
        else:
            corr = _cheb_apply(lambda v: mat @ v, dinv, res, cdeg, interval[0], interval[1])
        return corr if x is None else x + corr

    def vcycle(level: int, b: jnp.ndarray) -> jnp.ndarray:
        if level == n_lev - 1:
            return jax.scipy.linalg.lu_solve(lu_dev, b)
        lv = jlev[level]
        mat, pmat, rmat = lv[0], lv[1], lv[2]
        x = None
        for _ in range(npre):
            x = smooth_step(lv, b, x)
        if x is None:
            x = jnp.zeros_like(b)
            rc = rmat @ b
        else:
            rc = rmat @ (b - mat @ x)
        x = x + pmat @ vcycle(level + 1, rc)
        for _ in range(npost):
            x = smooth_step(lv, b, x)
        return x

    if svec is None:
        def apply(r: jnp.ndarray) -> jnp.ndarray:
            return vcycle(0, r)
    else:
        def apply(r: jnp.ndarray) -> jnp.ndarray:
            return svec[1] * vcycle(0, svec[0] * r)

    ns = [lv.a.shape[0] for lv in levels]
    nnzs = [int(lv.a.nnz) for lv in levels]
    info = {
        "levels": [{"n": m, "nnz": z, "nnz_per_row": z / m} for m, z in zip(ns, nnzs)],
        "n_levels": n_lev,
        "operator_complexity": sum(nnzs) / nnz0,
        "grid_complexity": sum(ns) / n,
        "coarsest_n": nc,
        "coarsening_stalled": bool(nc > max_coarse),
        "rho": [lv.rho for lv in levels],
        "jacobi_omega": [lv.omega for lv in levels[:-1]] if smoother == "jacobi" else None,
        "setup_breakdown_s": timings,
        "params": {"strength": strength, "max_coarse": max_coarse, "smoother": smoother, "omega": omega,
                   "presmooth": presmooth, "postsmooth": postsmooth, "cheb_degree": cheb_degree,
                   "cheb_lower_ratio": cheb_lower_ratio, "prolongator_smoothing": prolongator_smoothing,
                   "max_levels": max_levels, "volume_scaling": volume_scaling, "filter_smoothing": filter_smoothing},
    }
    if keep_host:
        info["_host"] = {"levels": levels, "scale": scale}
    return Preconditioner(f"sa_{smoother}", apply, nbytes=nbytes, jax_native=True, info=info)


# ---------------------------------------------------------------------------
# 5. incomplete LU
# ---------------------------------------------------------------------------
@_timed
def ilu(op: Any, *, owner_plane: np.ndarray | None = None, drop_tol: float = 1e-4, fill_factor: float = 10.0) -> Preconditioner:
    """SuperLU incomplete LU (``scipy.sparse.linalg.spilu``); host apply (sequential triangular solves). Reference only."""
    a = sp.csc_matrix(op.matrix, dtype=np.float64)
    n = a.shape[0]
    lu = spla.spilu(a, drop_tol=drop_tol, fill_factor=fill_factor)
    nnz = _lu_nnz(lu)
    return Preconditioner("ilu", _host_apply(lambda r: lu.solve(r), n), nbytes=int(nnz * 12 + n * 8), jax_native=False,
                          info={"lu_nnz": nnz, "lu_nnz_over_a_nnz": nnz / a.nnz, "drop_tol": drop_tol,
                                "fill_factor": fill_factor})


# ---------------------------------------------------------------------------
# 6. ring/plane block solvers: exact JAX-native plane solve and block Gauss-Seidel across planes
# ---------------------------------------------------------------------------
@dataclass
class RingLayout:
    """Padded (super-ring, slot) layout of the owners of every eta plane (all planes share the partition).

    ``ring_rank`` / ``plane_rank`` are the ranks of the ring / plane labels among the sorted unique labels, ``theta_pos``
    the position of the owner among the owners of its (ring, plane) (order of ``owner_theta``).  Rings are merged
    greedily, in ring order, into *super-rings* of at most ``B`` owners per plane (``B`` = largest number of owners of one
    ring in one plane, or ``max_block``): ``super_of_ring[ring]``, ``ring_offset[ring]`` (first slot of the ring inside
    its super-ring).  The owner sits in slot ``ring_offset + theta_pos`` of super-ring ``super_of``; the unused slots
    of a super-ring are padding.  ``w`` is the largest super-ring distance of any in-plane coupling (block bandwidth).
    """

    n: int
    rings: np.ndarray
    planes: np.ndarray
    ring_rank: np.ndarray
    plane_rank: np.ndarray
    theta_pos: np.ndarray
    counts: np.ndarray                 # (n_rings, n_planes) owners of every (ring, plane)
    B: int
    S: int
    w: int
    super_of_ring: np.ndarray
    ring_offset: np.ndarray
    super_of: np.ndarray
    slot: np.ndarray
    ring_reach: np.ndarray             # per ring: max |ring offset| of an in-plane coupling
    uniform_planes: bool               # every plane has the same owner count in every ring

    def info(self) -> dict:
        n_pl = int(self.planes.size)
        return {"n_owners": int(self.n), "n_rings": int(self.rings.size), "n_planes": n_pl, "S": int(self.S),
                "B": int(self.B), "w": int(self.w), "padded_fraction": 1.0 - self.n / float(self.S * n_pl * self.B),
                "uniform_planes": bool(self.uniform_planes),
                "ring_counts_max": [int(c) for c in self.counts.max(axis=1)],
                "super_rings": [[int(self.rings[r]) for r in np.flatnonzero(self.super_of_ring == s)]
                                for s in range(self.S)],
                "ring_reach": [int(x) for x in self.ring_reach]}


def ring_layout(matrix: sp.spmatrix, owner_ring: np.ndarray, owner_plane: np.ndarray, owner_theta: np.ndarray,
                max_block: int | None = None) -> RingLayout:
    """Super-ring partition, padded slot layout and block bandwidth of the in-plane blocks of ``matrix`` (see :class:`RingLayout`)."""
    a = sp.csr_matrix(matrix)
    n = a.shape[0]
    arrs = []
    for name, v in (("owner_ring", owner_ring), ("owner_plane", owner_plane), ("owner_theta", owner_theta)):
        if v is None:
            raise ValueError(f"{name} is required (ring, eta-plane and angular key of every owner)")
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
    reach = np.zeros(n_r, dtype=np.int64)
    np.maximum.at(reach, ring_rank[coo.row[inplane]], np.abs(ring_rank[coo.col[inplane]] - ring_rank[coo.row[inplane]]))
    return RingLayout(n=n, rings=rings, planes=planes, ring_rank=ring_rank, plane_rank=plane_rank, theta_pos=theta_pos,
                      counts=counts, B=block, S=n_s, w=w, super_of_ring=super_of_ring, ring_offset=ring_offset,
                      super_of=super_of, slot=slot, ring_reach=reach, uniform_planes=bool(np.all(counts == counts[:, :1])))


def _factor_banded(f: np.ndarray, cond_max: float) -> float:
    """In-place block LDU (no pivoting) of block-banded ``f[s, p, d]`` = block ``(s, s + d - w)`` of plane ``p``.

    On return ``f[s, :, w]`` holds ``D_s^{-1}`` (inverse of the Schur-complement diagonal block), ``f[r, :, d < w]`` the
    multipliers ``L_{r,s} = A~_{r,s} D_s^{-1}`` and ``f[s, :, d > w]`` the upper blocks ``U_{s,c}`` of the eliminated
    matrix.  Returns the largest 1-norm condition number of a diagonal block; raises ``ValueError`` above ``cond_max``.
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


@dataclass
class _PlaneSolver:
    """Exact solve of the diagonal plane blocks of a subset of planes; ``solve`` maps compact ``(n_sub,)`` vectors
    (ordered like ``owners``) to compact vectors.  JAX kind: ``gather`` / ``scatter`` are the padded-layout maps."""

    owners: np.ndarray
    solve: Callable[[jnp.ndarray], jnp.ndarray]
    nbytes: int
    info: dict
    gather: Callable[[jnp.ndarray], jnp.ndarray] | None = None        # (n_sub,) -> (S, n_planes_sub, B), padding 0
    scatter: Callable[[jnp.ndarray], jnp.ndarray] | None = None       # (S, n_planes_sub, B) -> (n_sub,)
    jax_native: bool = True
    data: Any = None                                                  # JAX kind: dict of the factor/index arrays
    apply_with: Callable[[Any, jnp.ndarray], jnp.ndarray] | None = None


def _inplane_entries(a: sp.csr_matrix, lay: RingLayout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = a.tocoo()
    keep = lay.plane_rank[coo.row] == lay.plane_rank[coo.col]
    return coo.row[keep], coo.col[keep], coo.data[keep]


_FACTOR_DTYPES = {"float64": np.float64, "float32": np.float32}


def _jax_plane_apply(data: dict, v: jnp.ndarray, n_ps: int, w: int, big_b: int) -> jnp.ndarray:
    """Banded block-LDU solve of the plane blocks with the arrays in ``data`` (``lo, up, dinv, idx, pos``).

    ``r`` is gathered into the padded layout and cast to the dtype of the factor arrays, both scans run in that dtype, the
    result is scattered and cast back to the dtype of ``v``.
    """
    lo_j, up_j, dinv_j = data["lo"], data["up"], data["dinv"]
    fd = lo_j.dtype
    b = jnp.concatenate([v.astype(fd), jnp.zeros((1,), fd)])[data["idx"]]
    zero = jnp.zeros((n_ps, w, big_b), fd)

    def fwd(carry, xs):                                               # carry[:, j] = y_{s - w + j}
        lo, bs = xs
        ys = bs - jnp.einsum("pjab,pjb->pa", lo, carry)
        return jnp.concatenate([carry[:, 1:], ys[:, None]], axis=1), ys

    _, y = jax.lax.scan(fwd, zero, (lo_j, b))

    def bwd(carry, xs):                                               # carry[:, e] = x_{s + 1 + e}
        up, di, ys = xs
        xs_ = jnp.einsum("pab,pb->pa", di, ys - jnp.einsum("pjab,pjb->pa", up, carry))
        return jnp.concatenate([xs_[:, None], carry[:, :-1]], axis=1), xs_

    _, x = jax.lax.scan(bwd, zero, (up_j, dinv_j, y), reverse=True)
    return x.reshape(-1)[data["pos"]].astype(v.dtype)


def _plane_solver(a: sp.csr_matrix, lay: RingLayout, planes_sel: np.ndarray, *, kind: str, cond_max: float,
                  entries: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                  factor_dtype: str = "float64") -> _PlaneSolver:
    """Plane-block solver for the planes with ranks ``planes_sel`` (``kind`` ``"jax"``: banded block LU; ``"host"``: SuperLU)."""
    n = lay.n
    tm: dict = {}
    t0 = time.perf_counter()
    planes_sel = np.asarray(planes_sel, dtype=np.int64)
    sub = np.flatnonzero(np.isin(lay.plane_rank, planes_sel))
    n_sub = int(sub.size)
    loc = np.full(n, -1, dtype=np.int64)
    loc[sub] = np.arange(n_sub)
    n_ps = int(planes_sel.size)
    pl = np.full(lay.planes.size, -1, dtype=np.int64)
    pl[planes_sel] = np.arange(n_ps)
    if kind == "host":
        lus = []
        for k in planes_sel:
            own = np.flatnonzero(lay.plane_rank == k)
            lus.append((loc[own], spla.splu(sp.csc_matrix(a[own][:, own]))))
        tm["factor"] = time.perf_counter() - t0

        def solve_np(v: np.ndarray) -> np.ndarray:
            z = np.empty_like(v)
            for lp, lu in lus:
                z[lp] = lu.solve(v[lp])
            return z

        lu_nnz = sum(_lu_nnz(lu) for _, lu in lus)
        return _PlaneSolver(sub, _host_apply(solve_np, n_sub), int(lu_nnz * 12 + n_sub * 8),
                            {"kind": "host", "n_planes": n_ps, "total_lu_nnz": int(lu_nnz), "setup_breakdown_s": tm},
                            jax_native=False)
    if kind != "jax":
        raise ValueError(f"plane solver kind must be 'jax' or 'host', got {kind!r}")
    if factor_dtype not in _FACTOR_DTYPES:
        raise ValueError(f"factor_dtype must be one of {sorted(_FACTOR_DTYPES)}, got {factor_dtype!r}")
    fdt = _FACTOR_DTYPES[factor_dtype]
    n_s, big_b, w = lay.S, lay.B, max(lay.w, 1)
    ploc, s_of, slot = pl[lay.plane_rank[sub]], lay.super_of[sub], lay.slot[sub]
    idx = np.full((n_s, n_ps, big_b), n_sub, dtype=np.int32)
    idx[s_of, ploc, slot] = np.arange(n_sub, dtype=np.int32)
    occ = idx != n_sub
    if int(occ.sum()) != n_sub:
        raise ValueError("two owners share a (super-ring, plane, slot): inconsistent ring/theta data")
    pos = ((s_of * n_ps + ploc) * big_b + slot).astype(np.int32)
    rows, cols, vals = entries if entries is not None else _inplane_entries(a, lay)
    sel = np.isin(lay.plane_rank[rows], planes_sel)
    rows, cols, vals = rows[sel], cols[sel], vals[sel]
    f = np.zeros((n_s, n_ps, 2 * w + 1, big_b, big_b))
    f[lay.super_of[rows], pl[lay.plane_rank[rows]], lay.super_of[cols] - lay.super_of[rows] + w,
      lay.slot[rows], lay.slot[cols]] = vals
    ps_, pp_, pb_ = np.nonzero(~occ)
    f[ps_, pp_, w, pb_, pb_] = 1.0                                       # padding: identity, no coupling
    tm["assemble"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    cond = _factor_banded(f, cond_max)
    tm["factor"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    # the factorisation is float64 on the host; the stored blocks are cast to factor_dtype
    data = {"lo": jnp.asarray(np.ascontiguousarray(f[:, :, :w], dtype=fdt)),        # (S, P, w, B, B): block (s, s - w + j)
            "up": jnp.asarray(np.ascontiguousarray(f[:, :, w + 1:], dtype=fdt)),     # (S, P, w, B, B): block (s, s + 1 + e)
            "dinv": jnp.asarray(np.ascontiguousarray(f[:, :, w], dtype=fdt)),        # (S, P, B, B)
            "idx": jnp.asarray(idx), "pos": jnp.asarray(pos)}
    del f
    tm["to_jax"] = time.perf_counter() - t1

    def apply_with(d: dict, v: jnp.ndarray) -> jnp.ndarray:
        return _jax_plane_apply(d, v, n_ps, w, big_b)

    def solve(v: jnp.ndarray) -> jnp.ndarray:
        return apply_with(data, v)

    def gather(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([v, jnp.zeros((1,), v.dtype)])[data["idx"]]

    def scatter(x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape(-1)[data["pos"]]

    nbytes = int(sum(x.nbytes for x in data.values()))
    info = {"kind": "jax", "S": n_s, "B": big_b, "w": w, "n_planes": n_ps, "n_owners": n_sub,
            "factor_dtype": factor_dtype, "padded_fraction": 1.0 - n_sub / float(n_s * n_ps * big_b),
            "storage_bytes": nbytes, "max_block_cond1": cond, "setup_breakdown_s": tm}
    return _PlaneSolver(sub, solve, nbytes, info, gather, scatter, True, data, apply_with)


@_timed
def plane_ring_block(op: Any, *, owner_ring: np.ndarray | None = None, owner_plane: np.ndarray | None = None,
                     owner_theta: np.ndarray | None = None, max_block: int | None = None, cond_max: float = 1e12,
                     factor_dtype: str = "float64", keep_host: bool = False) -> Preconditioner:
    """Exact solve of every eta-plane diagonal block ``A_kk`` (same result as :func:`plane_block`), JAX-native.

    Setup (host): within a plane the rings are merged in radial order into *super-rings* of at most ``B`` owners
    (``B`` = largest ring, or ``max_block``), so the in-plane block ``A_kk`` becomes block banded with ``S`` super-rings,
    block size ``B`` and block bandwidth ``w`` (largest super-ring distance of an in-plane coupling, from the data).  All
    planes share the partition (rings with fewer owners in a plane are padded: identity diagonal, no coupling).  Every
    plane block is factorised by banded block Gaussian elimination without pivoting (vectorised over the planes), in LDU
    form: strictly lower multipliers ``L_{r,s}`` (``S x w`` blocks), upper blocks ``U_{s,c}`` (``S x w``) and the explicit
    inverses of the Schur-complement diagonal blocks ``D_s^{-1}`` (``S`` blocks); a ``ValueError`` is raised if one of
    them has a 1-norm condition number above ``cond_max``.

    Apply (JAX only, no callbacks, jittable): gather ``r`` into the padded ``(S, planes, B)`` layout, a forward
    ``lax.scan`` over super-rings (``y_s = b_s - sum_j L_{s,j} y_{s-w+j}``, batched matmuls over the planes) and a backward
    scan (``x_s = D_s^{-1} (y_s - sum_e U_{s,e} x_{s+1+e})``), scatter back.  Storage:
    ``itemsize * S * planes * (2 w + 1) * B^2`` bytes plus two int32 index arrays.

    ``factor_dtype`` (``"float64"`` or ``"float32"``): the factorisation is always float64 on the host; the stored ``L``,
    ``U`` and ``D^-1`` blocks are cast to ``factor_dtype`` (float32 halves storage and memory traffic).  The apply casts the
    gathered ``r`` to ``factor_dtype``, runs both scans in it and returns float64; a float32 solve is still an exact-ish
    fixed linear operator (relative error ~1e-6), fine as an FGMRES preconditioner.

    The factor arrays are also exposed as ``Preconditioner.data`` with ``apply_with(data, r)`` (see :func:`make_solver`).
    ``keep_host`` stores the solver object under ``info["_solver"]`` (for tests).
    """
    _require_x64()
    a = _host_csr(op)
    t0 = time.perf_counter()
    lay = ring_layout(a, owner_ring, owner_plane, owner_theta, max_block)
    t_layout = time.perf_counter() - t0
    solver = _plane_solver(a, lay, np.arange(lay.planes.size), kind="jax", cond_max=cond_max,
                           factor_dtype=factor_dtype)
    info = {**lay.info(), **{k: v for k, v in solver.info.items() if k not in ("n_owners", "n_planes")}}
    info["setup_breakdown_s"] = {"layout": t_layout, **solver.info["setup_breakdown_s"]}
    info["params"] = {"max_block": max_block, "cond_max": cond_max, "factor_dtype": factor_dtype}
    if keep_host:
        info["_solver"] = solver
        info["_layout"] = lay
    return Preconditioner("plane_jax" if factor_dtype == "float64" else "plane_jax32", solver.solve,
                          nbytes=solver.nbytes, jax_native=True, info=info, data=solver.data,
                          apply_with=solver.apply_with)


def _bgs(name: str, op: Any, *, owner_ring, owner_plane, owner_theta, colors: int | None, symmetric: bool,
         plane_solver: str, max_block: int | None, cond_max: float, keep_host: bool) -> Preconditioner:
    _require_x64()
    if plane_solver not in ("jax", "host"):
        raise ValueError(f"plane_solver must be 'jax' or 'host', got {plane_solver!r}")
    tm: dict = {}
    t0 = time.perf_counter()
    a = _host_csr(op)
    n = a.shape[0]
    lay = ring_layout(a, owner_ring, owner_plane, owner_theta, max_block)
    tm["layout"] = time.perf_counter() - t0
    n_p = int(lay.planes.size)
    if colors is None:
        raw_colors = np.arange(n_p)                                       # one plane per color: sequential sweep
    else:
        if int(colors) < 1:
            raise ValueError("colors must be >= 1")
        raw_colors = np.asarray(lay.planes).astype(np.int64) % int(colors)
    _, color_of_plane = np.unique(raw_colors, return_inverse=True)
    color_of_plane = color_of_plane.reshape(-1)
    n_c = int(color_of_plane.max()) + 1
    t0 = time.perf_counter()
    coo = a.tocoo()
    pr, pc_ = lay.plane_rank[coo.row], lay.plane_rank[coo.col]
    cross = pr != pc_
    bad = cross & (color_of_plane[pr] == color_of_plane[pc_])
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(f"planes of one color are coupled (planes {lay.planes[pr[i]]} and {lay.planes[pc_[i]]} share color "
                         f"{color_of_plane[pr[i]]}): colors={colors} is incompatible with the inter-plane reach "
                         f"(offsets {sorted(set((lay.planes[pc_[cross]] - lay.planes[pr[cross]]).tolist()))[:12]})")
    tm["color_check"] = time.perf_counter() - t0
    entries = _inplane_entries(a, lay)
    order = list(range(n_c)) + (list(range(n_c - 2, -1, -1)) if symmetric else [])
    solvers, mats, rows_j = [], [], []
    nbytes = 0
    sub_tm: dict = {}
    for c in range(n_c):
        sol = _plane_solver(a, lay, np.flatnonzero(color_of_plane == c), kind=plane_solver, cond_max=cond_max,
                            entries=entries)
        solvers.append(sol)
        rows_j.append(jnp.asarray(sol.owners.astype(np.int32)))
        for k, v in sol.info["setup_breakdown_s"].items():
            sub_tm[k] = sub_tm.get(k, 0.0) + v
        t0 = time.perf_counter()
        need_mat = c > 0 or symmetric                                      # color 0 first sweep has z = 0
        mats.append(_bcsr(a[sol.owners]) if need_mat else None)
        tm["bcsr"] = tm.get("bcsr", 0.0) + time.perf_counter() - t0
        nbytes += sol.nbytes + int(rows_j[-1].nbytes) + (_bcsr_nbytes(mats[-1]) if need_mat else 0)
    tm.update({f"plane_solver_{k}": v for k, v in sub_tm.items()})
    nnz_c = [int(a[s.owners].nnz) for s in solvers]

    def apply(r: jnp.ndarray) -> jnp.ndarray:
        z = jnp.zeros_like(r)
        for step, c in enumerate(order):
            res = r[rows_j[c]] if step == 0 else r[rows_j[c]] - mats[c] @ z
            z = z.at[rows_j[c]].add(solvers[c].solve(res))
        return z

    info = {**lay.info(), "n_colors": n_c, "symmetric": bool(symmetric), "plane_solver": plane_solver,
            "colors": colors, "sweep_order": order, "plane_solves_per_apply": len(order),
            "matvec_equivalents_per_apply": sum(nnz_c[c] for c in order[1:]) / float(a.nnz),
            "planes_per_color": [int(np.sum(color_of_plane == c)) for c in range(n_c)],
            "interplane_offsets": sorted(set((lay.planes[pc_[cross]] - lay.planes[pr[cross]]).tolist()))[:16],
            "setup_breakdown_s": tm, "params": {"max_block": max_block, "cond_max": cond_max}}
    if plane_solver == "jax":
        info["storage_bytes_plane_solvers"] = int(sum(s.nbytes for s in solvers))
        info["max_block_cond1"] = max(s.info["max_block_cond1"] for s in solvers)
    if keep_host:
        info["_solvers"] = solvers
        info["_layout"] = lay
        info["_color_of_plane"] = color_of_plane
    return Preconditioner(name, apply, nbytes=nbytes, jax_native=plane_solver == "jax", info=info)


@_timed
def plane_bgs_forward(op: Any, *, owner_ring: np.ndarray | None = None, owner_plane: np.ndarray | None = None,
                      owner_theta: np.ndarray | None = None, plane_solver: str = "jax", max_block: int | None = None,
                      cond_max: float = 1e12, keep_host: bool = False) -> Preconditioner:
    """One forward block Gauss-Seidel sweep over the eta planes ``k = 0..n-1`` (one plane per color; sequential).

    ``z_k = A_kk^{-1} (r_k - sum_{l updated} A_kl z_l)`` from ``z = 0``, implemented as ``C = n_planes`` color steps of
    :func:`plane_bgs_multicolor`.  A fixed linear operator; kept as the iteration-count reference of the multicolor
    variants (sequential and compile-heavy: ``n_planes`` unrolled plane solves).  ``plane_solver`` is ``"jax"``
    (:func:`plane_ring_block` factors) or ``"host"`` (SuperLU through ``pure_callback``).
    """
    return _bgs("bgs_fwd", op, owner_ring=owner_ring, owner_plane=owner_plane, owner_theta=owner_theta, colors=None,
                symmetric=False, plane_solver=plane_solver, max_block=max_block, cond_max=cond_max, keep_host=keep_host)


@_timed
def plane_bgs_multicolor(op: Any, *, owner_ring: np.ndarray | None = None, owner_plane: np.ndarray | None = None,
                         owner_theta: np.ndarray | None = None, colors: int = 4, symmetric: bool = False,
                         plane_solver: str = "jax", max_block: int | None = None, cond_max: float = 1e12,
                         keep_host: bool = False) -> Preconditioner:
    """Multicolor block Gauss-Seidel across eta planes: color ``c = k mod colors`` (planes of one color must be uncoupled).

    Starting from ``z = 0``, for ``c = 0..colors-1``: ``z|_c += A_cc^{-1} (r - A z)|_c`` where ``A_cc`` is the block
    diagonal of the plane blocks of color ``c`` (solved by the exact plane solver, batched over that color's planes) and
    ``(A z)|_c`` is a precomputed row-subset BCSR matvec (one sweep = one matvec in total plus the plane solves).
    ``symmetric=True`` adds the backward sweep ``c = colors-2..0`` (the repeated last color would be a zero update, since
    its residual vanishes right after its solve, so it is skipped).  Raises ``ValueError`` if two planes of one color are
    coupled (e.g. ``colors = 4`` needs ``n_planes % 4 == 0`` and an inter-plane reach <= 2).  A fixed linear operator.
    """
    nm = "bgs_mc%d%s%s" % (int(colors), "_sym" if symmetric else "", "_host" if plane_solver == "host" else "")
    return _bgs(nm, op, owner_ring=owner_ring, owner_plane=owner_plane, owner_theta=owner_theta, colors=int(colors),
                symmetric=symmetric, plane_solver=plane_solver, max_block=max_block, cond_max=cond_max,
                keep_host=keep_host)


# ---------------------------------------------------------------------------
# solver helper: preconditioner data as a jit argument
# ---------------------------------------------------------------------------
def make_solver(system: Any, prec: Preconditioner, config: Any) -> Callable:
    """Jitted FGMRES solve ``run(rhs, boundary_term=None, x0=None) -> (x, info)`` for the P07 Dirichlet ``system``.

    Same solve as :func:`drbx.native.fci_perpendicular_p07_solve.solve_p07_dirichlet_jit`, but the system arrays and, if
    the preconditioner provides ``data`` / ``apply_with``, its data are *arguments* of the jitted function (not closure
    constants), so the compile does not embed (or constant-fold) hundreds of MB of factors: the first call measures the
    compile of the program only.  Preconditioners without ``data`` fall back to the closure ``prec.apply``.  ``x0 = None``
    is a zero initial guess.  ``run.jitted`` is the underlying jitted function ``(system, rhs, boundary_term, x0, data)``.
    """
    from drbx.native.fci_perpendicular_p07_solve import solve_p07_dirichlet

    use_data = prec.apply_with is not None and prec.data is not None
    data = prec.data if use_data else None

    @jax.jit
    def jitted(system_, rhs, boundary_term, x0, data_):
        precond = (lambda r: prec.apply_with(data_, r)) if use_data else prec.apply
        return solve_p07_dirichlet(system_, rhs, boundary_term=boundary_term, config=config, x0=x0,
                                   preconditioner=precond)

    def run(rhs, boundary_term=None, x0=None):
        rhs = jnp.asarray(rhs, dtype=jnp.float64)
        bt = jnp.zeros_like(rhs) if boundary_term is None else jnp.asarray(boundary_term, dtype=jnp.float64)
        x0_ = jnp.zeros_like(rhs) if x0 is None else jnp.asarray(x0, dtype=jnp.float64)
        return jitted(system, rhs, bt, x0_, data)

    run.jitted = jitted
    run.uses_data_argument = use_data
    return run
