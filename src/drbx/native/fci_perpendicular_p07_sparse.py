"""Host scipy-sparse export of the P07 owner action (P08 step 5, phi audits).

Design: ``work/p08_step5_phi_audits_20261001/design_notes.md`` section D1.

``export_p07_sparse(plan, kind)`` assembles, directly from the plan arrays (vectorized COO -> CSR, no probing of
``p07_action``), the matrices of the all-fields-same-kind owner action of ``fci_perpendicular_p07_operator``::

    p07_action(plan, u, bc, kind)  ==  A @ u + B_val @ g_D + B_tan @ g_T.reshape(2 Qd, F) + B_nn @ g_N

``A`` is ``(n_owners, n_owners)``; ``B_val`` ``(n_owners, Qd)`` (Dirichlet trace values), ``B_tan`` ``(n_owners, 2 Qd)``
(column ``2 q + a``, ``a``: 0 theta, 1 eta, the trace tangential gradient) and ``B_nn`` ``(n_owners, Qn)``
(physical-normal data). The face flux is the integrated row (donors, Dirichlet lift ``-w * g_D`` at conditioned
batches, tangential term of every face of a batch); for ``kind = "neumann"`` the rows of the faces in
``p07.neumann_face`` are replaced by the Neumann restoration ``sum_q integrand[q] . gradient_N[q]``; family 0
(collapsed r=0) faces carry zero flux. The owner level is ``diag(1 / owner_volume) @ S @ F`` with ``-1`` at the lower
and ``+1`` at the upper owner of each face. The Neumann rows keep the Dirichlet lift/tangential columns of the faces
they do not replace, so ``B_val`` / ``B_tan`` of the Neumann kind are computed, not assumed empty.

``save_p07_sparse`` / ``load_p07_sparse`` cache the export in one ``.npz`` with an exact caller-supplied identity.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData
from drbx.stencils.operator_plan import IntegratedNeumannRows, PerpendicularPlan

__all__ = ["KINDS", "P07SparseOperator", "export_p07_sparse", "boundary_source", "apply_p07_sparse",
           "save_p07_sparse", "load_p07_sparse"]

KINDS = ("dirichlet", "neumann")
SCHEMA = "drbx.p07-sparse.v1"
_MATRICES = ("matrix", "dirichlet_value", "dirichlet_tangential", "neumann_normal")
_ARRAYS = ("owner_volume", "dirichlet_points", "neumann_points")


@dataclass(frozen=True)
class P07SparseOperator:
    kind: str
    matrix: sp.csr_matrix                   # (n, n) linear part
    dirichlet_value: sp.csr_matrix          # (n, Qd)
    dirichlet_tangential: sp.csr_matrix     # (n, 2 Qd), column 2 q + a (a: 0 theta, 1 eta)
    neumann_normal: sp.csr_matrix           # (n, Qn)
    owner_volume: np.ndarray                # (n,)
    dirichlet_points: np.ndarray            # (Qd, 3)
    neumann_points: np.ndarray              # (Qn, 3)

    @property
    def n_owners(self) -> int:
        return int(self.matrix.shape[0])


def _csr(rows, cols, vals, shape) -> sp.csr_matrix:
    m = sp.coo_matrix((np.asarray(vals, dtype=np.float64).ravel(),
                       (np.asarray(rows, dtype=np.int64).ravel(), np.asarray(cols, dtype=np.int64).ravel())),
                      shape=shape).tocsr()
    m.sum_duplicates()
    m.eliminate_zeros()
    m.indices = m.indices.astype(np.int64)
    m.indptr = m.indptr.astype(np.int64)
    return m


def _face_matrices(p07, n_owners: int):
    """Dirichlet face-flux matrices ``(F_A, F_val, F_tan)`` of shapes ``(Fp, n)``, ``(Fp, Qd)``, ``(Fp, 2 Qd)``."""
    rows = p07.rows
    fp, qd = int(rows.face_count), int(rows.boundary_query_count)
    # chunk-padding faces carry the out-of-range id ``fp`` and zero weights: they are dropped
    real = [np.asarray(b.face_ids, dtype=np.int64) < fp for b in rows.batches]
    seen = np.concatenate([np.asarray(b.face_ids, dtype=np.int64)[r] for b, r in zip(rows.batches, real)]
                          or [np.empty(0, np.int64)])
    if len(np.unique(seen)) != len(seen):
        raise ValueError("P07 batches overlap in faces: the sparse export assumes disjoint batches")
    ra, ca, va = [], [], []
    rv, cv, vv = [], [], []
    rt, ct, vt = [], [], []
    for b, r in zip(rows.batches, real):
        face = np.asarray(b.face_ids, dtype=np.int64)[r]
        donors = np.asarray(b.donor_ids, dtype=np.int64)[r]
        w = np.asarray(b.weights, dtype=np.float64)[r]
        d = donors.shape[1]
        ra.append(np.repeat(face, d)); ca.append(donors.ravel()); va.append(w.ravel())
        if not qd or b.boundary_donor_ids is None:         # no boundary arrays: a bucket without conditioned faces
            continue
        cond = np.asarray(b.conditioned, dtype=bool)[r]
        if cond.any():
            bd = np.asarray(b.boundary_donor_ids, dtype=np.int64)[r][cond]
            rv.append(np.repeat(face[cond], d)); cv.append(bd.ravel()); vv.append(-w[cond].ravel())
        tid = np.asarray(b.tangential_ids, dtype=np.int64)[r]            # (f, Q)
        tw = np.asarray(b.tangential_weights, dtype=np.float64)[r]       # (f, Q, 2)
        q = tid.shape[1]
        for a in range(2):
            rt.append(np.repeat(face, q)); ct.append((2 * tid + a).ravel()); vt.append(tw[..., a].ravel())
    cat = lambda xs: np.concatenate(xs) if xs else np.empty(0)
    return (_csr(cat(ra), cat(ca), cat(va), (fp, n_owners)),
            _csr(cat(rv), cat(cv), cat(vv), (fp, qd)),
            _csr(cat(rt), cat(ct), cat(vt), (fp, 2 * qd)))


def _neumann_matrices(p07, n_owners: int, qn: int):
    """Neumann restoration rows ``(N_A (Fn, n), N_nn (Fn, Qn))`` of the faces ``p07.neumann_face``."""
    nr = p07.neumann
    fn = len(p07.neumann_face)
    if isinstance(nr, IntegratedNeumannRows):          # per-face union layout: already contracted with the integrand
        face = np.repeat(np.arange(fn, dtype=np.int64), nr.donor_ids.shape[1])
        bface = np.repeat(np.arange(fn, dtype=np.int64), nr.boundary_ids.shape[1])
        return (_csr(face, np.asarray(nr.donor_ids, dtype=np.int64).ravel(),
                     np.asarray(nr.weights, dtype=np.float64).ravel(), (fn, n_owners)),
                _csr(bface, np.asarray(nr.boundary_ids, dtype=np.int64).ravel(),
                     np.asarray(nr.boundary_weights, dtype=np.float64).ravel(), (fn, qn)))
    integrand = np.asarray(p07.integrand, dtype=np.float64)                  # (Fn, 9, 3)
    if integrand.shape != (fn, 9, 3) or len(nr.donor_ids) != 9 * fn:
        raise ValueError("P07 Neumann rows and integrand disagree")
    w = integrand.reshape(fn * 9, 3)                                         # (9 Fn, 3) weight of row r, component a
    face_of_row = np.repeat(np.arange(fn, dtype=np.int64), 9)
    donors = np.asarray(nr.donor_ids, dtype=np.int64)                        # (R, W)
    gw = np.asarray(nr.gradient_weights, dtype=np.float64)                   # (R, 3, W)
    ev = np.einsum("ra,rad->rd", w, gw)                                      # (R, W)
    bids = np.asarray(nr.boundary_ids, dtype=np.int64)                       # (R, 28)
    bgw = np.asarray(nr.boundary_gradient_weights, dtype=np.float64)         # (R, 3, 28)
    eb = np.einsum("ra,raj->rj", w, bgw)                                     # (R, 28)
    na = _csr(np.repeat(face_of_row, donors.shape[1]), donors.ravel(), ev.ravel(), (fn, n_owners))
    nn = _csr(np.repeat(face_of_row, bids.shape[1]), bids.ravel(), eb.ravel(), (fn, qn))
    return na, nn


def export_p07_sparse(plan: PerpendicularPlan, kind: str) -> P07SparseOperator:
    """Sparse form of ``p07_action(plan, u, bc, kind)`` for all fields of the same ``kind``; see the module docstring."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    p07 = plan.p07
    if p07 is None:
        raise ValueError("the plan has no P07 rows")
    vol = np.asarray(p07.owner_volume, dtype=np.float64)
    n = len(vol)
    fp, qd = int(p07.rows.face_count), int(p07.rows.boundary_query_count)
    dpts = np.asarray(plan.dirichlet_points, dtype=np.float64).reshape(-1, 3)
    npts = np.asarray(plan.neumann_points, dtype=np.float64).reshape(-1, 3)
    if qd != len(dpts):
        raise ValueError("plan.dirichlet_points disagrees with the P07 boundary query count")
    qn = len(npts)
    fa, fv, ft = _face_matrices(p07, n)
    fnn = sp.csr_matrix((fp, qn))
    if kind == "neumann" and p07.neumann is not None:
        face = np.asarray(p07.neumann_face, dtype=np.int64)
        keep = np.ones(fp)
        keep[face] = 0.0
        d_keep = sp.diags(keep)
        na, nn = _neumann_matrices(p07, n, qn)
        scatter = sp.csr_matrix((np.ones(len(face)), (face, np.arange(len(face)))), shape=(fp, len(face)))
        fa = d_keep @ fa + scatter @ na
        fv, ft = d_keep @ fv, d_keep @ ft
        fnn = scatter @ nn
    live = sp.diags((np.asarray(p07.family) != 0).astype(np.float64))
    lower = np.asarray(p07.rows.lower_owner, dtype=np.int64)
    upper = np.asarray(p07.rows.upper_owner, dtype=np.int64)
    ids = np.arange(fp, dtype=np.int64)
    lm, um = lower >= 0, upper >= 0
    scat = sp.coo_matrix((np.concatenate([-np.ones(lm.sum()), np.ones(um.sum())]),
                          (np.concatenate([lower[lm], upper[um]]), np.concatenate([ids[lm], ids[um]]))),
                         shape=(n, fp)).tocsr()
    left = sp.diags(1.0 / vol) @ scat @ live

    def finish(f):
        m = sp.csr_matrix(left @ f)
        m.sum_duplicates()
        m.eliminate_zeros()
        m.indices = m.indices.astype(np.int64)
        m.indptr = m.indptr.astype(np.int64)
        return m

    return P07SparseOperator(kind, finish(fa), finish(fv), finish(ft), finish(fnn), vol.copy(), dpts, npts)


def boundary_source(op: P07SparseOperator, bc: BoundaryData) -> np.ndarray:
    """``(n, F)`` boundary contribution ``B g``; a ``None`` part of ``bc`` is allowed only if its block is empty."""
    parts = (("dirichlet_value", op.dirichlet_value, bc.dirichlet_value, lambda g: g),
             ("dirichlet_tangential", op.dirichlet_tangential, bc.dirichlet_tangential,
              lambda g: g.reshape(2 * g.shape[0], g.shape[2])),
             ("neumann_normal", op.neumann_normal, bc.neumann_normal, lambda g: g))
    out = None
    for name, block, data, flat in parts:
        if data is None:
            if block.nnz:
                raise ValueError(f"BoundaryData.{name} is missing but the {op.kind} operator has nonzero "
                                 f"{name} coefficients")
            continue
        g = flat(np.asarray(data, dtype=np.float64))
        if g.ndim != 2 or g.shape[0] != block.shape[1]:
            raise ValueError(f"BoundaryData.{name} has shape {np.shape(data)} for {block.shape[1]} columns")
        contrib = block @ g
        out = contrib if out is None else out + contrib
    if out is None:
        raise ValueError("BoundaryData carries no arrays: the number of fields is unknown")
    return out


def apply_p07_sparse(op: P07SparseOperator, fields, bc: BoundaryData | None = None) -> np.ndarray:
    """``A u (+ B g)`` as ``(n, F)``; ``bc is None`` gives the linear part only."""
    u = np.asarray(fields, dtype=np.float64)
    if u.ndim != 2 or u.shape[0] != op.n_owners:
        raise ValueError("fields must have shape (n_owners, F)")
    out = op.matrix @ u
    if bc is not None:
        out = out + boundary_source(op, bc)
    return out


def save_p07_sparse(path: str | Path, op: P07SparseOperator, identity: dict) -> None:
    """Atomically write one ``.npz`` (CSR parts + arrays + JSON identity)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"schema": SCHEMA, "kind": op.kind, "identity": identity}
    arrays = {"metadata_json": np.asarray(json.dumps(meta, sort_keys=True))}
    for name in _MATRICES:
        m = sp.csr_matrix(getattr(op, name))
        arrays[f"{name}_data"] = m.data
        arrays[f"{name}_indices"] = m.indices
        arrays[f"{name}_indptr"] = m.indptr
        arrays[f"{name}_shape"] = np.asarray(m.shape, dtype=np.int64)
    for name in _ARRAYS:
        arrays[name] = np.asarray(getattr(op, name))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def load_p07_sparse(path: str | Path, expected_identity: dict | None = None) -> P07SparseOperator:
    """Load a ``save_p07_sparse`` file; ``ValueError`` on a schema or (if given) identity mismatch."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(np.asarray(z["metadata_json"]).item()))
        if meta.get("schema") != SCHEMA:
            raise ValueError(f"P07 sparse schema mismatch: {meta.get('schema')!r} != {SCHEMA!r}")
        if expected_identity is not None and meta.get("identity") != json.loads(json.dumps(expected_identity)):
            raise ValueError("P07 sparse identity mismatch")
        mats = {}
        for name in _MATRICES:
            shape = tuple(int(s) for s in z[f"{name}_shape"])
            mats[name] = sp.csr_matrix((z[f"{name}_data"], z[f"{name}_indices"], z[f"{name}_indptr"]), shape=shape)
        arrays = {name: np.asarray(z[name]) for name in _ARRAYS}
    return P07SparseOperator(meta["kind"], **mats, **arrays)
