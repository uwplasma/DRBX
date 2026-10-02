"""eta-sharded Dirichlet potential solve of the P07 perpendicular operator (P08 step 6, stage B).

Design: ``work/p08_step6_sharding_20261002/design.md`` (section "Sharded phi solve").  The single-device solve
(:mod:`drbx.native.fci_perpendicular_phi_solver`) is reproduced inside ``jax.shard_map`` over the one-axis mesh ``"z"`` of
:mod:`drbx.native.fci_perpendicular_sharding`; the single-device modules are not touched.

Layout
------
The owners are relabelled plane-major (``new = plane * m + rank``, :func:`plane_major_permutation`) and shard ``s`` of
``Sz`` owns the planes ``[s p, (s + 1) p)``, ``p = n / Sz``.  Per shard :class:`ShardedPhiSolver` holds, stacked along a
leading shard axis,

* the CSR rows of the owned owners with columns remapped to the local extended index ``e m + rank`` of the window
  ``[lower halo | owned | upper halo]`` (``(p + 2 h) m`` columns, ``h = 2``: the P07 matrix couples the planes ``-2 .. +2``;
  every column must lie in the window, otherwise the lowering raises and reports the offending plane offsets), stored as
  a BCSR triple padded to a common ``nnz`` (zero entries in the last row) with the entries of every row kept in their
  original order;
* the rows of the Dirichlet boundary blocks (``dirichlet_value``, ``dirichlet_tangential``) of the owned owners, so
  ``B g`` is formed per shard from the replicated global trace ``g``;
* the owner volumes and the diagonal;
* the per-plane block-Jacobi factors of the shard's own planes.  They are the plane slices of the factors of the global
  single-device :class:`PlanePreconditioner` (identical arrays, only the gather / scatter maps are re-indexed to the
  shard): the super-ring partition of that preconditioner depends on all planes (ring counts, bandwidth), so building
  each shard from its own block would not reproduce the global factors; an existing :class:`PhiSolver` donates its
  preconditioner, an operator builds it once on the host.

Solve
-----
:func:`sharded_solve_phi` runs solvax ``gmres`` (right preconditioned, restarted, flexible) inside ``shard_map``.  Every
reduction of ``gmres`` goes through ``inner_product`` (basis dots, norms, tolerance), here
``psum(sum(volume * a * b))``, the matvec is ``A_local @ exchange_plane_halo(v)`` for the owned rows only, and the
preconditioner is plane-local.  The reported residual ``||rhs - B g - A x||_M`` is recomputed after the solve exactly as
in ``solve_p07_dirichlet``.  All vectors are plane-major ``(n * m,)`` arrays (``to_plane_major`` / ``from_plane_major``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import lax
from jax.experimental import sparse as jsparse
from jax.sharding import Mesh, PartitionSpec as P
from solvax.krylov import gmres

from drbx.native.fci_perpendicular_p07_solve import P07SolveConfig
from drbx.native.fci_perpendicular_phi_solver import PHI_RTOL_DEFAULT, PhiSolver, phi_solver_from_operator
from drbx.native.fci_perpendicular_plane_preconditioner import (
    PlanePreconditioner, apply_plane_preconditioner)
from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData
from drbx.native.fci_perpendicular_sharding import (
    AXIS, exchange_plane_halo, place_sharded, plane_major_permutation)

__all__ = ["PHI_HALO", "ShardedPhiSolver", "shard_phi_solver", "sharded_boundary_term", "sharded_matvec",
           "sharded_preconditioner_apply", "sharded_solve_phi"]

#: halo planes of the matvec: the P07 matrix couples the eta planes ``-2 .. +2`` (measured on the N32/N48/N64 exports)
PHI_HALO = 2


@dataclass(frozen=True)
class _Meta:
    """Static (hashable) part of a :class:`ShardedPhiSolver`."""

    n: int              # eta planes
    m: int              # owners per plane
    n_shards: int
    halo: int
    p: int              # planes per shard
    qd: int             # Dirichlet value columns of B
    qt: int             # Dirichlet tangential columns of B (2 per point)
    config: P07SolveConfig

    @property
    def rows(self) -> int:
        return self.p * self.m

    @property
    def window(self) -> int:
        return (self.p + 2 * self.halo) * self.m


@jax.tree_util.register_pytree_node_class
class ShardedPhiSolver:
    """Stacked per-shard operator, boundary blocks and preconditioner of the potential solve (leading axis ``Sz``).

    Array leaves (all with leading axis ``Sz``): ``a_*`` the BCSR triple ``(data, indices, indptr)`` of the owned rows
    (``(p m, (p + 2 h) m)``), ``bv_*`` / ``bt_*`` the same for the rows of ``dirichlet_value`` ``(p m, Qd)`` and
    ``dirichlet_tangential`` ``(p m, 2 Qd)`` (empty arrays when the block has no columns), ``diagonal`` and ``volume``
    ``(p m,)`` and ``prec`` (a :class:`PlanePreconditioner` of ``p`` planes with stacked leaves).  ``perm`` / ``inverse``
    (host, not part of the pytree) are the plane-major relabelling; ``setup_seconds`` and ``info`` are host records.
    """

    def __init__(self, a, bv, bt, diagonal, volume, prec, meta: _Meta, perm=None, inverse=None,
                 setup_seconds=None, info=None):
        self.a, self.bv, self.bt = a, bv, bt
        self.diagonal, self.volume, self.prec, self.meta = diagonal, volume, prec, meta
        self.perm, self.inverse = perm, inverse
        self.setup_seconds = {} if setup_seconds is None else setup_seconds
        self.info = {} if info is None else info

    @property
    def n_shards(self) -> int:
        return self.meta.n_shards

    @property
    def n_owners(self) -> int:
        return self.meta.n * self.meta.m

    @property
    def config(self) -> P07SolveConfig:
        return self.meta.config

    def place(self, mesh: Mesh) -> "ShardedPhiSolver":
        """The solver with every leaf ``device_put`` with ``P("z")`` (a no-op for placed leaves); unlike
        :func:`place_sharded` it keeps the host records ``perm`` / ``inverse`` / ``setup_seconds`` / ``info``."""
        placed = place_sharded(self, mesh)
        placed.perm, placed.inverse, placed.setup_seconds, placed.info = (
            self.perm, self.inverse, self.setup_seconds, self.info)
        return placed

    def tree_flatten(self):
        return (self.a, self.bv, self.bt, self.diagonal, self.volume, self.prec), self.meta

    @classmethod
    def tree_unflatten(cls, meta, children):
        return cls(*children, meta)


# --------------------------------------------------------------------------
# Host lowering
# --------------------------------------------------------------------------

def _stack_csr(mats: list[sp.csr_matrix]):
    """Stack equally shaped CSR matrices into ``(data, indices, indptr)`` arrays padded to the common ``nnz``
    (padding entries are zero, column 0, appended to the last row)."""
    rows = mats[0].shape[0]
    nnz = max(1, max(int(m.nnz) for m in mats))
    data = np.zeros((len(mats), nnz))
    indices = np.zeros((len(mats), nnz), dtype=np.int32)
    indptr = np.zeros((len(mats), rows + 1), dtype=np.int32)
    for s, mat in enumerate(mats):
        k = int(mat.nnz)
        data[s, :k], indices[s, :k], indptr[s, :] = mat.data, mat.indices, mat.indptr
        indptr[s, rows] = nnz
    return jnp.asarray(data), jnp.asarray(indices), jnp.asarray(indptr)


def _local_matrix(csr: sp.csr_matrix, rows_old: np.ndarray, perm: np.ndarray, s: int, meta: _Meta) -> sp.csr_matrix:
    """Rows ``rows_old`` of ``csr`` (original entry order) with columns remapped to the extended window of shard ``s``."""
    n, m, p, h = meta.n, meta.m, meta.p, meta.halo
    sub = csr[rows_old]
    new = perm[sub.indices.astype(np.int64)]
    plane, rank = new // m, new % m
    ext = (plane - s * p + h) % n                                  # position in the window [s p - h, (s + 1) p + h)
    bad = ext >= p + 2 * h
    if bad.any():
        row_plane = s * p + np.repeat(np.arange(meta.rows), np.diff(sub.indptr)) // m
        offsets = np.unique((plane[bad] - row_plane[bad] + n // 2) % n - n // 2)
        raise ValueError(
            f"shard {s}: {int(bad.sum())} matrix entries couple eta planes outside the halo window "
            f"(halo = {h}); offending plane offsets {offsets.tolist()}")
    return sp.csr_matrix((sub.data, (ext * m + rank).astype(np.int32), sub.indptr), shape=(meta.rows, meta.window))


def _shard_preconditioner(prec: PlanePreconditioner, perm, inverse, meta: _Meta) -> PlanePreconditioner:
    """Plane slices of the global factors, re-indexed to the owners of each shard (stacked, leading axis ``Sz``)."""
    n, m, p, n_shards = meta.n, meta.m, meta.p, meta.n_shards
    if prec.P != n or prec.n_owners != n * m:
        raise ValueError(f"the preconditioner covers {prec.P} planes / {prec.n_owners} owners, expected {n} / {n * m}")
    big_b = prec.B
    lo, up, dinv = (np.asarray(x) for x in (prec.lo, prec.up, prec.dinv))
    idx, pos = np.asarray(prec.idx).astype(np.int64), np.asarray(prec.pos).astype(np.int64)
    out: dict[str, list] = {k: [] for k in ("lo", "up", "dinv", "idx", "pos")}
    for s in range(n_shards):
        planes = slice(s * p, (s + 1) * p)
        out["lo"].append(lo[:, planes])
        out["up"].append(up[:, planes])
        out["dinv"].append(dinv[:, planes])
        real = idx[:, planes] < n * m                              # padding slots hold ``n_owners``
        local = perm[np.minimum(idx[:, planes], n * m - 1)] - s * p * m
        if not np.all((local[real] >= 0) & (local[real] < p * m)):
            raise ValueError(f"shard {s}: the preconditioner gather map leaves the shard's planes")
        out["idx"].append(np.where(real, local, p * m).astype(np.int32))
        pg = pos[inverse[s * p * m:(s + 1) * p * m]]               # flat (super-ring, plane, slot) position per owner
        slot, rest = pg % big_b, pg // big_b
        plane, sup = rest % n, rest // n
        if not np.all(plane // p == s):
            raise ValueError(f"shard {s}: the preconditioner scatter map leaves the shard's planes")
        out["pos"].append(((sup * p + plane - s * p) * big_b + slot).astype(np.int32))
    stacked = {k: jnp.asarray(np.stack(v)) for k, v in out.items()}
    meta_prec = replace(prec.meta, P=p, n_owners=p * m)
    return PlanePreconditioner(stacked["lo"], stacked["up"], stacked["dinv"], stacked["idx"], stacked["pos"], meta_prec)


def shard_phi_solver(solver_or_op: PhiSolver | Any, raw_to_owner, n: int, n_shards: int, halo: int = PHI_HALO, *,
                     factor_dtype: str = "float32", rtol: float | None = None, restart: int | None = None,
                     max_restarts: int | None = None, mesh: Mesh | None = None) -> ShardedPhiSolver:
    """Lower a :class:`PhiSolver` (or a ``"dirichlet"`` :class:`P07SparseOperator`) to ``n_shards`` eta shards.

    ``raw_to_owner`` is the ``(n^3,)`` owner map and ``n`` the plane count (``n_shards`` must divide it and ``n / n_shards
    >= halo`` unless there is one shard).  For a ``PhiSolver`` its operator, preconditioner factors and solve options
    are used (``factor_dtype``, ``rtol``, ``restart`` and ``max_restarts`` are ignored); for an operator the global
    preconditioner is built first and the options default to those of :func:`phi_solver_from_operator`.  ``mesh``
    (optional) places the arrays with ``P("z")``.  Raises ``ValueError`` when a matrix entry reaches beyond ``halo``
    planes.  ``setup_seconds`` records the host phases.
    """
    n, n_shards, halo = int(n), int(n_shards), int(halo)
    if n_shards < 1 or n % n_shards:
        raise ValueError(f"the plane count {n} must be divisible by the number of shards {n_shards}")
    p = n // n_shards
    if halo < 0 or (n_shards > 1 and p < halo):
        raise ValueError(f"the halo ({halo}) must be non-negative and not wider than the {p} planes of a shard")
    seconds: dict = {}
    t0 = time.perf_counter()
    if isinstance(solver_or_op, PhiSolver):
        solver = solver_or_op
    else:
        solver = phi_solver_from_operator(
            solver_or_op, raw_to_owner, n, factor_dtype=factor_dtype,
            rtol=PHI_RTOL_DEFAULT if rtol is None else rtol, restart=50 if restart is None else restart,
            max_restarts=40 if max_restarts is None else max_restarts)
    seconds["global_solver"] = time.perf_counter() - t0
    op = solver.op
    t0 = time.perf_counter()
    perm, inverse, m = plane_major_permutation(raw_to_owner, n)
    if len(perm) != op.n_owners:
        raise ValueError(f"raw_to_owner has {len(perm)} owners but the operator has {op.n_owners}")
    meta = _Meta(n=n, m=m, n_shards=n_shards, halo=halo, p=p, qd=int(op.dirichlet_value.shape[1]),
                 qt=int(op.dirichlet_tangential.shape[1]), config=solver.config)
    seconds["layout"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    csr = sp.csr_matrix(op.matrix, dtype=np.float64)
    bv, bt = sp.csr_matrix(op.dirichlet_value, dtype=np.float64), sp.csr_matrix(op.dirichlet_tangential,
                                                                               dtype=np.float64)
    diagonal = np.asarray(csr.diagonal(), dtype=np.float64)
    if not np.all(diagonal > 0.0):
        raise ValueError(f"operator diagonal must be strictly positive (min {diagonal.min():.3e})")
    volume = np.asarray(op.owner_volume, dtype=np.float64).reshape(-1)
    if op.neumann_normal.nnz:
        raise ValueError("the sharded potential solve needs a Dirichlet operator with an empty Neumann block")
    a_list, bv_list, bt_list, diag_list, vol_list = [], [], [], [], []
    for s in range(n_shards):
        rows_old = inverse[s * p * m:(s + 1) * p * m]
        a_list.append(_local_matrix(csr, rows_old, perm, s, meta))
        bv_list.append(bv[rows_old])
        bt_list.append(bt[rows_old])
        diag_list.append(diagonal[rows_old])
        vol_list.append(volume[rows_old])
    a = _stack_csr(a_list)
    bv_s = _stack_csr(bv_list) if meta.qd else None
    bt_s = _stack_csr(bt_list) if meta.qt else None
    seconds["matrix"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    prec = _shard_preconditioner(solver.prec, perm, inverse, meta)
    seconds["preconditioner"] = time.perf_counter() - t0
    info = {"nnz_per_shard": [int(x.nnz) for x in a_list], "nnz_padded": int(a[0].shape[1]),
            "prec_bytes_per_shard": int(prec.nbytes // n_shards)}
    sharded = ShardedPhiSolver(a, bv_s, bt_s, jnp.asarray(np.stack(diag_list)), jnp.asarray(np.stack(vol_list)), prec,
                               meta, perm, inverse, seconds, info)
    return sharded if mesh is None else sharded.place(mesh)


# --------------------------------------------------------------------------
# Device side
# --------------------------------------------------------------------------

def _bcsr(triple, shape):
    return jsparse.BCSR(tuple(triple), shape=shape)


def _local_matvec(sh: ShardedPhiSolver, v):
    """``A_local @ [halo exchange of v]``: owned rows of the global product (inside ``shard_map``, local blocks)."""
    meta = sh.meta
    ext = exchange_plane_halo(v.reshape(meta.p, meta.m), meta.halo, AXIS, meta.n_shards).reshape(-1)
    return _bcsr(sh.a, (meta.rows, meta.window)) @ ext


def _local_boundary_term(sh: ShardedPhiSolver, g_val, g_tan):
    """``B g`` of the owned rows from the replicated traces ``g_val`` ``(Qd,)`` and ``g_tan`` ``(2 Qd,)``."""
    meta = sh.meta
    out = jnp.zeros((meta.rows,), sh.volume.dtype)
    if meta.qd:
        out = out + _bcsr(sh.bv, (meta.rows, meta.qd)) @ g_val
    if meta.qt:
        out = out + _bcsr(sh.bt, (meta.rows, meta.qt)) @ g_tan
    return out


def _unstack(tree):
    return jax.tree_util.tree_map(lambda a: a[0], tree)


def _traces(sharded: ShardedPhiSolver, bc: BoundaryData):
    """Global Dirichlet traces ``(g_val (Qd,), g_tan (2 Qd,))`` of the single field of ``bc`` (its Neumann data is not
    used: the lowering requires an empty Neumann block)."""
    meta = sharded.meta
    parts = []
    for name, data, cols, shape in (("dirichlet_value", bc.dirichlet_value, meta.qd, (meta.qd, 1)),
                                    ("dirichlet_tangential", bc.dirichlet_tangential, meta.qt, (meta.qt // 2, 2, 1))):
        if data is None:
            if cols:
                raise ValueError(f"BoundaryData.{name} is missing but the operator has {cols} columns")
            parts.append(jnp.zeros((cols,)))
            continue
        arr = jnp.asarray(data, dtype=sharded.volume.dtype)
        if arr.shape != shape:
            raise ValueError(f"BoundaryData.{name} has shape {tuple(arr.shape)}, expected {shape} "
                             "(the potential solve carries one field)")
        parts.append(arr.reshape(-1))
    return parts[0], parts[1]


def _vector(x, sharded: ShardedPhiSolver, name: str):
    arr = jnp.asarray(x, dtype=sharded.volume.dtype).reshape(-1)
    if arr.shape != (sharded.n_owners,):
        raise ValueError(f"{name} must have shape ({sharded.n_owners},), got {tuple(arr.shape)}")
    return arr


def _smap(fn, mesh: Mesh, in_specs, out_specs):
    return jax.shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)


def _check_mesh(sharded: ShardedPhiSolver, mesh: Mesh) -> None:
    if int(mesh.shape[AXIS]) != sharded.n_shards:
        raise ValueError(f"the mesh has {mesh.shape[AXIS]} shards along {AXIS!r}, the solver {sharded.n_shards}")


@partial(jax.jit, static_argnames=("mesh",))
def _matvec_jit(sharded, v, *, mesh):
    return _smap(lambda sh, x: _local_matvec(_unstack(sh), x), mesh, (P(AXIS), P(AXIS)), P(AXIS))(sharded, v)


def sharded_matvec(sharded: ShardedPhiSolver, v_pm, mesh: Mesh):
    """``A v`` for a plane-major vector (one halo exchange plus the local BCSR product), plane-major result."""
    _check_mesh(sharded, mesh)
    return _matvec_jit(sharded.place(mesh), _vector(v_pm, sharded, "v"), mesh=mesh)


@partial(jax.jit, static_argnames=("mesh",))
def _boundary_jit(sharded, g_val, g_tan, *, mesh):
    return _smap(lambda sh, gv, gt: _local_boundary_term(_unstack(sh), gv, gt), mesh, (P(AXIS), P(), P()),
                 P(AXIS))(sharded, g_val, g_tan)


def sharded_boundary_term(sharded: ShardedPhiSolver, bc: BoundaryData, mesh: Mesh):
    """``B g`` as a plane-major ``(n * m,)`` array, each shard forming its rows from the replicated ``bc``."""
    _check_mesh(sharded, mesh)
    g_val, g_tan = _traces(sharded, bc)
    return _boundary_jit(sharded.place(mesh), g_val, g_tan, mesh=mesh)


@partial(jax.jit, static_argnames=("mesh",))
def _precond_jit(sharded, r, *, mesh):
    return _smap(lambda sh, x: apply_plane_preconditioner(_unstack(sh).prec, x), mesh, (P(AXIS), P(AXIS)),
                 P(AXIS))(sharded, r)


def sharded_preconditioner_apply(sharded: ShardedPhiSolver, r_pm, mesh: Mesh):
    """The plane-local preconditioner ``M^-1 r`` of every shard (plane-major in and out)."""
    _check_mesh(sharded, mesh)
    return _precond_jit(sharded.place(mesh), _vector(r_pm, sharded, "r"), mesh=mesh)


@partial(jax.jit, static_argnames=("mesh", "has_bc"))
def _solve_jit(sharded, rhs, x0, g_val, g_tan, *, mesh, has_bc):
    def body(sh, rhs_s, x0_s, gv, gt):
        sh = _unstack(sh)
        config = sh.meta.config
        rhs_eff = rhs_s - _local_boundary_term(sh, gv, gt) if has_bc else rhs_s
        volume = sh.volume

        def inner(a, b):
            return lax.psum(jnp.sum(volume * a * b), AXIS)

        def matvec(v):
            return _local_matvec(sh, v)

        sol = gmres(matvec, rhs_eff, x0=x0_s, precond=lambda r: apply_plane_preconditioner(sh.prec, r),
                    inner_product=inner, restart=config.restart, rtol=config.rtol, atol=config.atol,
                    max_restarts=config.max_restarts)
        x = sol.x
        residual = rhs_eff - matvec(x)
        residual_norm = jnp.sqrt(inner(residual, residual))
        rhs_norm = jnp.sqrt(inner(rhs_eff, rhs_eff))
        tolerance = jnp.maximum(config.atol, config.rtol * rhs_norm)
        info = {"iterations": sol.iterations, "residual_norm": residual_norm,
                "relative_residual": residual_norm / jnp.where(rhs_norm > 0.0, rhs_norm, 1.0),
                "converged": residual_norm <= tolerance, "rhs_norm": rhs_norm}
        return x, info

    return _smap(body, mesh, (P(AXIS), P(AXIS), P(AXIS), P(), P()), (P(AXIS), P()))(sharded, rhs, x0, g_val, g_tan)


def sharded_solve_phi(sharded: ShardedPhiSolver, rhs_pm, bc: BoundaryData | None = None, *, x0_pm=None,
                      mesh: Mesh) -> tuple[jax.Array, dict]:
    """Solve ``A phi = rhs - B g`` eta-sharded; the sharded counterpart of :func:`solve_phi`.

    ``rhs_pm`` / ``x0_pm`` are plane-major ``(n * m,)`` vectors, ``bc`` the (global, replicated) Dirichlet data of one
    field (``None``: ``g = 0``).  Returns ``(phi, info)`` with the plane-major potential as a ``jax.Array`` sharded
    ``P("z")`` and the keys of :func:`solve_phi` (``iterations``, ``converged``, ``residual_norm`` =
    ``||rhs - B g - A phi||_M`` recomputed after the solve, ``relative_residual``, ``rhs_norm``, ``seconds`` including the
    device synchronisation).
    """
    _check_mesh(sharded, mesh)
    placed = sharded.place(mesh)
    rhs = _vector(rhs_pm, sharded, "rhs")
    x0 = jnp.zeros_like(rhs) if x0_pm is None else _vector(x0_pm, sharded, "x0")
    has_bc = bc is not None
    g_val, g_tan = _traces(sharded, bc) if has_bc else (jnp.zeros((sharded.meta.qd,)), jnp.zeros((sharded.meta.qt,)))
    t0 = time.perf_counter()
    x, raw = _solve_jit(placed, rhs, x0, g_val, g_tan, mesh=mesh, has_bc=has_bc)
    x = jax.block_until_ready(x)
    seconds = time.perf_counter() - t0
    info = {"iterations": int(raw["iterations"]), "converged": bool(raw["converged"]),
            "residual_norm": float(raw["residual_norm"]), "relative_residual": float(raw["relative_residual"]),
            "rhs_norm": float(raw["rhs_norm"]), "seconds": seconds}
    return x, info
