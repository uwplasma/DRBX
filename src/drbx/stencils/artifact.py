"""The per-grid row artifact: CSR storage for prepared perpendicular rows.

Scope (P08 step 1, task 3 "Artifact I/O" — see
``work/p08_step1_consolidation_design_20260928/design.md`` sections 2, 3, 6):
pack the host, geometry-only row objects already produced by the existing
package builders

- ``drbx.geometry.fci_perpendicular_reconstruction.PointRows`` (R1 cell, R2
  face common, R3 face side requests),
- ``drbx.geometry.fci_perpendicular_neumann_trace.NeumannPointRows`` (the
  return type of ``prepare_neumann_point_rows``, the wall-lattice Neumann
  rows), and
- ``drbx.geometry.fci_perpendicular_integrated_rows.IntegratedFaceRow`` (the
  return type of ``prepare_integrated_face_rows``, the R4 P07 integrated
  face rows)

into a compact per-target CSR (``row_ptr``/``donor``/weights), save it to a
chunked on-disk layout under ``row_artifact/N{n}/``, reject a load whose
identity or per-chunk sha256 does not match, and ``expand()`` the CSR back
into the *same* row objects, bitwise: donor order and coefficients are
copied, never re-summed.

Deliberate simplifications relative to the full design (left to later
tasks): this module does not itself write ``census.npz`` / ``topology.npz``
/ ``geometry.npz`` (task 1/2 outputs) or a single artifact-wide
``boundary_queries.npz``; each packed chunk carries its own deduplicated
query-point table instead of sharing one across the whole grid. R4 chunks
do not yet carry the precontracted P07N Neumann sub-payload mentioned in
design §3 ("R4 also stores the precontracted P07N Neumann rows") or the
vectorized CSR→JAX lowering (design §4, task 8); those depend on the
builder/runner (task 4) and are out of scope here. Nothing in this module
imports from ``scripts/``.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from drbx.geometry.fci_perpendicular_reconstruction import PointRows
from drbx.geometry.fci_perpendicular_neumann_trace import NeumannPointRows
from drbx.geometry.fci_perpendicular_integrated_rows import IntegratedFaceRow

SCHEMA = "drbx.p-row-artifact.v2"
#: The previous schema, kept only so tooling (e.g.
#: ``scripts/p_shared/backfill_neumann_tags.py``) can recognize an artifact
#: that predates the Neumann ``request``/``radial_degree`` tags (see
#: ``NeumannRowChunk``) and upgrade it in place; ``load_row_artifact`` never
#: accepts this schema string.
SCHEMA_V1_NEUMANN_UNTAGGED = "drbx.p-row-artifact.v1"

REQUEST_KINDS = ("R1", "R2", "R3", "R4", "neumann")
BC_VARIANTS = ("", "D", "N")
#: The request kinds a Neumann row's own ``request`` tag may take -- the
#: *originating* request that needed this wall-lattice companion row (see
#: ``drbx.stencils.builder.NeumannRowRequest.source``). Never ``"neumann"``
#: itself or an R-kind that has no Neumann companion.
NEUMANN_SOURCE_KINDS = ("R1", "R2", "R3", "R4")


# --------------------------------------------------------------------------
# JSON-safety, hashing, and the identity/manifest helpers
# --------------------------------------------------------------------------

def _json_safe(value):
    """Recursively convert numpy scalars/arrays to plain JSON-serializable types.

    Python's ``json`` module round-trips finite floats exactly (repr-based),
    so this never loses bits for the diagnostics/identity payloads it wraps.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path) -> str:
    """sha256 of a geometry/topology/metric-cache/sidecar/MAKEGRID file."""
    return hash_bytes(Path(path).read_bytes())


def hash_source(target) -> str:
    """sha256 of a builder/provider's source blob.

    ``target`` is a path, or a Python module/class/function, whose source
    file is hashed.
    """
    if isinstance(target, (str, Path)):
        return hash_file(target)
    source_file = inspect.getsourcefile(target) or inspect.getfile(target)
    if source_file is None:
        raise ValueError("cannot locate a source file to hash for this target")
    return hash_file(source_file)


def _numpy_blas_platform() -> dict:
    try:
        config = np.show_config(mode="dicts")
        blas = config.get("Build Dependencies", {}).get("blas", {})
        blas_info = {"name": blas.get("name"), "version": blas.get("version")}
    except Exception:
        blas_info = {"name": None, "version": None}
    return {"numpy_version": np.__version__, "blas": blas_info, "platform": platform.platform()}


def build_identity(*, component_hashes: dict, source_hashes: dict, policy: dict) -> dict:
    """Assemble the manifest identity: component/source hashes, policy, platform.

    ``component_hashes`` covers the geometry, topology, metric-cache,
    sidecar, and MAKEGRID files (design §3); ``source_hashes`` covers every
    builder/provider source blob (see ``hash_source``); ``policy`` is the
    numeric-option dict (SVD/residual tolerances, quadrature, thresholds,
    Neumann condition cap, degrees, seam policy, ...). numpy/BLAS/platform
    are recorded automatically.
    """
    identity = {"component_hashes": dict(component_hashes),
                "source_hashes": dict(source_hashes),
                "policy": dict(policy)}
    identity.update(_numpy_blas_platform())
    return _json_safe(identity)


def _common_dtype(blocks):
    """Result dtype of a list of arrays (float64 when empty); never narrows longdouble."""
    return np.result_type(*blocks) if blocks else np.dtype(np.float64)


class _QueryTable:
    """Deduplicates float64 (3,) query points by exact bit pattern.

    Every stored point is a verbatim copy of the array it was given, so a
    later lookup returns the identical bits (never a recomputed value) —
    required for the "donor/target trace-query ids" to expand bitwise.
    """

    def __init__(self):
        self._index: dict[bytes, int] = {}
        self._points: list[np.ndarray] = []

    def add(self, point) -> int:
        point = np.array(point, dtype=np.float64, copy=True)
        if point.shape != (3,):
            raise ValueError("a query point must have shape (3,)")
        key = point.tobytes()
        idx = self._index.get(key)
        if idx is None:
            idx = len(self._points)
            self._index[key] = idx
            self._points.append(point)
        return idx

    def add_many(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("query points must have shape (count, 3)")
        return np.array([self.add(p) for p in points], dtype=np.int32)

    def array(self) -> np.ndarray:
        if not self._points:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack(self._points, axis=0)


def _broadcast(value, n, name):
    if isinstance(value, (list, tuple, np.ndarray)) and not isinstance(value, str):
        value = list(value)
        if len(value) != n:
            raise ValueError(f"{name} must supply exactly one entry per row")
        return value
    return [value] * n


# --------------------------------------------------------------------------
# R1 / R2 / R3 point-row chunks (PointRows)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PointRowChunk:
    """Per-target CSR for a batch of ``PointRows`` sources.

    Every target (one evaluated point) keeps its own ``donor_ptr`` slice, so
    the donor list of the source row it came from is copied once per point
    it covers — matching the shared-donor-column convention the package
    builders already use (``PointRows.value``/``gradient`` are dense over
    one donor set per source row). ``source_ptr`` marks where each original
    ``PointRows`` object's points begin/end, so ``expand_point_rows`` can
    reconstruct the exact same grouping.
    """

    request: np.ndarray            # (T,) '<U8', R1/R2/R3
    entity_id: np.ndarray          # (T,) int64, face-or-raw id
    quad_node: np.ndarray          # (T,) int16, quadrature node q
    family: np.ndarray             # (T,) '<U32'
    conditioned: np.ndarray        # (T,) bool
    bc_variant: np.ndarray         # (T,) '<U1', '' / 'D' / 'N'
    radial_degree: np.ndarray      # (T,) int16
    target_point: np.ndarray       # (T, 3) float64, trace_target_points[q]
    donor_ptr: np.ndarray          # (T+1,) int64
    donor: np.ndarray              # (nnz,) int32
    value: np.ndarray              # (nnz,) float64
    has_gradient: np.ndarray       # (T,) bool
    gradient_ptr: np.ndarray       # (T+1,) int64
    gradient: np.ndarray           # (3, gradient_nnz) float64
    donor_query: np.ndarray        # (nnz,) int32, valid where conditioned
    source_ptr: np.ndarray         # (S+1,) int64
    source_diagnostics_json: str   # JSON list of length S
    query_table: np.ndarray        # (Q, 3) float64


def pack_point_rows(rows: Sequence[PointRows], *, request, entity_id, quad_node=None,
                    bc_variant="", radial_degree=0, store_gradient=True,
                    query_table: _QueryTable | None = None) -> PointRowChunk:
    """Pack a sequence of ``PointRows`` (one per host builder call) into a chunk.

    ``request``/``entity_id``/``bc_variant``/``radial_degree`` are broadcast
    if scalar, else must supply one entry per row in ``rows``.
    ``store_gradient`` selects, per row, whether its gradient triple is
    persisted (R1/R2 keep it; R3 side rows may drop it — design §3).
    ``quad_node`` defaults to the point's position within its source row.
    """
    rows = list(rows)
    n = len(rows)
    request = _broadcast(request, n, "request")
    entity_id = _broadcast(entity_id, n, "entity_id")
    bc_variant = _broadcast(bc_variant, n, "bc_variant")
    radial_degree = _broadcast(radial_degree, n, "radial_degree")
    store_gradient = _broadcast(store_gradient, n, "store_gradient")
    if quad_node is None:
        quad_node = [np.arange(len(row.trace_target_points)) for row in rows]
    else:
        quad_node = list(quad_node)
        if len(quad_node) != n:
            raise ValueError("quad_node must supply one index array per row")
    for value in request:
        if value not in REQUEST_KINDS:
            raise ValueError(f"unsupported request kind: {value!r}")
    for value in bc_variant:
        if value not in BC_VARIANTS:
            raise ValueError(f"unsupported BC variant: {value!r}")

    table = query_table if query_table is not None else _QueryTable()
    req_col, ent_col, q_col, fam_col = [], [], [], []
    cond_col, bcv_col, deg_col, has_grad_col = [], [], [], []
    target_points = []
    donor_ptr = [0]
    donor_blocks, value_blocks, donor_query_blocks = [], [], []
    gradient_ptr = [0]
    gradient_blocks = []
    source_ptr = [0]
    diagnostics = []

    for s, row in enumerate(rows):
        donor_ids = np.asarray(row.donor_ids)
        d = len(donor_ids)
        q = len(row.trace_target_points)
        if len(quad_node[s]) != q:
            raise ValueError("quad_node entry length must match the row's point count")
        if row.value.shape != (q, d) or row.gradient.shape != (q, 3, d):
            raise ValueError("point row value/gradient shape does not match its donor/target counts")
        conditioned = bool(row.boundary_conditioned)
        if conditioned:
            if row.trace_donor_points.shape != (d, 3):
                raise ValueError("a conditioned point row needs one trace donor point per donor")
            donor_query = table.add_many(row.trace_donor_points)
        else:
            donor_query = np.full(d, -1, dtype=np.int32)
        family = str(row.diagnostics.get("family", ""))
        keep_gradient = bool(store_gradient[s])
        for local in range(q):
            req_col.append(request[s]); ent_col.append(int(entity_id[s]))
            q_col.append(int(quad_node[s][local])); fam_col.append(family)
            cond_col.append(conditioned); bcv_col.append(bc_variant[s])
            deg_col.append(int(radial_degree[s]))
            target_points.append(np.asarray(row.trace_target_points[local], dtype=np.float64))
            donor_blocks.append(donor_ids)
            value_blocks.append(row.value[local])
            donor_query_blocks.append(donor_query)
            donor_ptr.append(donor_ptr[-1] + d)
            has_grad_col.append(keep_gradient)
            if keep_gradient:
                gradient_blocks.append(row.gradient[local])
                gradient_ptr.append(gradient_ptr[-1] + d)
            else:
                gradient_ptr.append(gradient_ptr[-1])
        source_ptr.append(source_ptr[-1] + q)
        diagnostics.append(_json_safe(row.diagnostics))

    donor = np.concatenate(donor_blocks).astype(np.int32) if donor_blocks else np.zeros(0, dtype=np.int32)
    value = np.concatenate(value_blocks) if value_blocks else np.zeros(0)
    donor_query = (np.concatenate(donor_query_blocks).astype(np.int32) if donor_query_blocks
                   else np.zeros(0, dtype=np.int32))
    gradient = (np.concatenate(gradient_blocks, axis=1) if gradient_blocks
                else np.zeros((3, 0)))

    return PointRowChunk(
        request=np.array(req_col, dtype="<U8"),
        entity_id=np.array(ent_col, dtype=np.int64),
        quad_node=np.array(q_col, dtype=np.int16),
        family=np.array(fam_col, dtype="<U32"),
        conditioned=np.array(cond_col, dtype=bool),
        bc_variant=np.array(bcv_col, dtype="<U1"),
        radial_degree=np.array(deg_col, dtype=np.int16),
        target_point=np.array(target_points, dtype=np.float64).reshape(-1, 3),
        donor_ptr=np.array(donor_ptr, dtype=np.int64),
        donor=donor,
        value=value,
        has_gradient=np.array(has_grad_col, dtype=bool),
        gradient_ptr=np.array(gradient_ptr, dtype=np.int64),
        gradient=gradient,
        donor_query=donor_query,
        source_ptr=np.array(source_ptr, dtype=np.int64),
        source_diagnostics_json=json.dumps(diagnostics, sort_keys=True),
        query_table=table.array(),
    )


def expand_point_rows(chunk: PointRowChunk) -> tuple[PointRows, ...]:
    """Reconstruct the exact ``PointRows`` tuple that was packed, bitwise."""
    diagnostics = json.loads(chunk.source_diagnostics_json)
    table = chunk.query_table
    result = []
    for s in range(len(chunk.source_ptr) - 1):
        t0, t1 = int(chunk.source_ptr[s]), int(chunk.source_ptr[s + 1])
        q = t1 - t0
        widths = np.diff(chunk.donor_ptr[t0:t1 + 1])
        d = int(widths[0]) if len(widths) else 0
        if np.any(widths != d):
            raise ValueError("corrupted point-row chunk: a source's targets disagree on donor width")
        donor_ids = (chunk.donor[chunk.donor_ptr[t0]:chunk.donor_ptr[t0] + d].astype(np.int64)
                     if d else np.zeros(0, dtype=np.int64))
        value = np.zeros((q, d))
        keep_gradient = bool(chunk.has_gradient[t0]) if q else False
        if np.any(chunk.has_gradient[t0:t1] != keep_gradient):
            raise ValueError("corrupted point-row chunk: a source's targets disagree on gradient storage")
        gradient = np.zeros((q, 3, d)) if keep_gradient else np.zeros((q, 3, 0))
        for local, t in enumerate(range(t0, t1)):
            p0, p1 = int(chunk.donor_ptr[t]), int(chunk.donor_ptr[t + 1])
            if not np.array_equal(chunk.donor[p0:p1].astype(np.int64), donor_ids):
                raise ValueError("corrupted point-row chunk: donor ids differ within one source group")
            value[local] = chunk.value[p0:p1]
            if keep_gradient:
                g0, g1 = int(chunk.gradient_ptr[t]), int(chunk.gradient_ptr[t + 1])
                gradient[local] = chunk.gradient[:, g0:g1]
        conditioned = bool(chunk.conditioned[t0]) if q else False
        if np.any(chunk.conditioned[t0:t1] != conditioned):
            raise ValueError("corrupted point-row chunk: a source's targets disagree on conditioning")
        if conditioned:
            dq = chunk.donor_query[chunk.donor_ptr[t0]:chunk.donor_ptr[t0] + d]
            trace_donor_points = table[dq] if d else np.empty((0, 3))
        else:
            trace_donor_points = np.empty((0, 3))
        trace_target_points = chunk.target_point[t0:t1].copy()
        result.append(PointRows(donor_ids, value, gradient, conditioned,
                                trace_donor_points, trace_target_points, diagnostics[s]))
    return tuple(result)


# --------------------------------------------------------------------------
# Neumann wall-lattice rows (NeumannPointRows)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NeumannRowChunk:
    """Per-target CSR for a batch of ``NeumannPointRows``.

    One target per row (no grouping is needed: ``prepare_neumann_point_rows``
    already returns one row per requested point). The 28 wall-lattice points
    are stored once per target as ids into the chunk's deduplicated query
    table (design §3: "Neumann rows with their 28 wall-lattice ids").

    ``request``/``radial_degree`` tag each row with the *originating* request
    that needed this Neumann companion (``NeumannRowRequest.source`` /
    ``.radial_degree`` in ``drbx.stencils.builder``) -- the same idiom
    ``PointRowChunk.request``/``.radial_degree`` uses. This matters because,
    in a faces unit, an R2 and an R3 row can share the same
    ``(entity_id, quad_node)`` key (R3's Neumann row is tagged at the face's
    own census row index, not the side-doubled R3 point-row id) while using
    different degrees -- without this tag a consumer cannot tell the two
    apart.
    """

    entity_id: np.ndarray           # (T,) int64
    quad_node: np.ndarray           # (T,) int16
    request: np.ndarray             # (T,) '<U8', the originating R1/R2/R3/R4
    radial_degree: np.ndarray       # (T,) int8
    donor_ptr: np.ndarray           # (T+1,) int64
    donor: np.ndarray               # (nnz,) int32
    value: np.ndarray               # (nnz,) float64
    gradient: np.ndarray            # (3, nnz) float64
    wall_query: np.ndarray          # (T, 28) int32
    boundary_value: np.ndarray      # (T, 28) builder dtype (longdouble; float64 on arm64)
    boundary_gradient: np.ndarray   # (T, 3, 28) builder dtype (longdouble; float64 on arm64)
    condition: np.ndarray           # (T,) float64
    constraint_residual: np.ndarray  # (T,) float64
    query_table: np.ndarray         # (Q, 3) float64


def pack_neumann_rows(rows: Sequence[NeumannPointRows], *, entity_id, request, radial_degree,
                      quad_node=None, query_table: _QueryTable | None = None) -> NeumannRowChunk:
    """Pack a sequence of ``NeumannPointRows`` into a chunk.

    ``request`` (the originating ``R1``/``R2``/``R3``/``R4`` request) and
    ``radial_degree`` are required -- callers must state them explicitly,
    there is no silent default -- and are broadcast if scalar, else must
    supply one entry per row in ``rows`` (matching ``entity_id``'s own
    broadcast convention). ``quad_node`` defaults to each row's index within
    ``rows``.
    """
    rows = list(rows)
    n = len(rows)
    entity_id = _broadcast(entity_id, n, "entity_id")
    quad_node = _broadcast(quad_node if quad_node is not None else list(range(n)), n, "quad_node")
    request = _broadcast(request, n, "request")
    radial_degree = _broadcast(radial_degree, n, "radial_degree")
    for value in request:
        if value not in NEUMANN_SOURCE_KINDS:
            raise ValueError(f"unsupported Neumann source request: {value!r}")
    table = query_table if query_table is not None else _QueryTable()
    donor_ptr = [0]
    donor_blocks, value_blocks, gradient_blocks = [], [], []
    wall_query, boundary_value, boundary_gradient, condition, residual = [], [], [], [], []
    for row in rows:
        d = len(row.donor_ids)
        if row.value.shape != (d,) or row.gradient.shape != (3, d):
            raise ValueError("Neumann row value/gradient shape does not match its donor count")
        if (row.boundary_points.shape != (28, 3) or row.boundary_value.shape != (28,)
                or row.boundary_gradient.shape != (3, 28)):
            raise ValueError("a Neumann row must carry all 28 wall-lattice points")
        donor_blocks.append(np.asarray(row.donor_ids))
        value_blocks.append(row.value)
        gradient_blocks.append(row.gradient)
        donor_ptr.append(donor_ptr[-1] + d)
        wall_query.append(table.add_many(row.boundary_points))
        # Keep the builder's own dtype: prepare_neumann_point_rows returns these as
        # longdouble, which is float64 on arm64 but 80-bit extended on x86 (Perlmutter).
        boundary_value.append(np.asarray(row.boundary_value))
        boundary_gradient.append(np.asarray(row.boundary_gradient))
        condition.append(float(row.condition))
        residual.append(float(row.constraint_residual))
    donor = np.concatenate(donor_blocks).astype(np.int32) if donor_blocks else np.zeros(0, dtype=np.int32)
    value = np.concatenate(value_blocks) if value_blocks else np.zeros(0)
    gradient = np.concatenate(gradient_blocks, axis=1) if gradient_blocks else np.zeros((3, 0))
    return NeumannRowChunk(
        entity_id=np.array(entity_id, dtype=np.int64),
        quad_node=np.array(quad_node, dtype=np.int16),
        request=np.array(request, dtype="<U8"),
        radial_degree=np.array(radial_degree, dtype=np.int8),
        donor_ptr=np.array(donor_ptr, dtype=np.int64),
        donor=donor, value=value, gradient=gradient,
        wall_query=np.array(wall_query, dtype=np.int32).reshape(n, 28),
        boundary_value=np.array(boundary_value, dtype=_common_dtype(boundary_value)).reshape(n, 28),
        boundary_gradient=np.array(boundary_gradient, dtype=_common_dtype(boundary_gradient)).reshape(n, 3, 28),
        condition=np.array(condition, dtype=np.float64),
        constraint_residual=np.array(residual, dtype=np.float64),
        query_table=table.array(),
    )


def expand_neumann_rows(chunk: NeumannRowChunk) -> tuple[NeumannPointRows, ...]:
    result = []
    for t in range(len(chunk.entity_id)):
        p0, p1 = int(chunk.donor_ptr[t]), int(chunk.donor_ptr[t + 1])
        donor_ids = chunk.donor[p0:p1].astype(np.int64)
        value = chunk.value[p0:p1].copy()
        gradient = chunk.gradient[:, p0:p1].copy()
        boundary_points = chunk.query_table[chunk.wall_query[t]]
        result.append(NeumannPointRows(donor_ids, value, gradient, boundary_points,
                                       chunk.boundary_value[t].copy(), chunk.boundary_gradient[t].copy(),
                                       float(chunk.condition[t]), float(chunk.constraint_residual[t])))
    return tuple(result)


# --------------------------------------------------------------------------
# R4 P07 integrated face rows (IntegratedFaceRow)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class IntegratedRowChunk:
    """Per-target CSR for a batch of ``IntegratedFaceRow`` (one per face)."""

    entity_id: np.ndarray           # (T,) int64, the P07 face id
    family: np.ndarray              # (T,) int16, the P07 topology census code
    conditioned: np.ndarray         # (T,) bool
    donor_ptr: np.ndarray           # (T+1,) int64
    donor: np.ndarray               # (nnz,) int32
    weight: np.ndarray              # (nnz,) float64
    donor_query: np.ndarray         # (nnz,) int32, valid where conditioned
    value_loading: np.ndarray       # (nnz,) float64, valid where conditioned
    target_points: np.ndarray       # (T, 9, 3) float64
    tangential_loading: np.ndarray  # (T, 9, 2) float64, valid where conditioned
    query_table: np.ndarray         # (Q, 3) float64


def pack_integrated_rows(rows: Sequence[IntegratedFaceRow], *, entity_id,
                         query_table: _QueryTable | None = None) -> IntegratedRowChunk:
    rows = list(rows)
    n = len(rows)
    entity_id = _broadcast(entity_id, n, "entity_id")
    table = query_table if query_table is not None else _QueryTable()
    family, conditioned = [], []
    donor_ptr = [0]
    donor_blocks, weight_blocks, donor_query_blocks, value_loading_blocks = [], [], [], []
    target_points, tangential_loading = [], []
    for row in rows:
        d = len(row.donor_ids)
        if row.weights.shape != (d,):
            raise ValueError("integrated row weight shape does not match its donor count")
        if row.trace_target_points.shape != (9, 3):
            raise ValueError("an integrated row needs 9 q3 target points")
        family.append(int(row.family))
        is_conditioned = bool(row.boundary_conditioned)
        conditioned.append(is_conditioned)
        donor_blocks.append(np.asarray(row.donor_ids))
        weight_blocks.append(row.weights)
        target_points.append(np.asarray(row.trace_target_points, dtype=np.float64))
        if is_conditioned:
            if row.trace_donor_points.shape != (d, 3):
                raise ValueError("a conditioned integrated row needs one trace donor point per donor")
            if row.tangential_loading.shape != (9, 2):
                raise ValueError("a conditioned integrated row needs (9, 2) tangential loading")
            if row.value_loading.shape != (d,):
                raise ValueError("a conditioned integrated row needs per-donor value loading")
            donor_query_blocks.append(table.add_many(row.trace_donor_points))
            value_loading_blocks.append(np.asarray(row.value_loading, dtype=np.float64).copy())
            tangential_loading.append(np.asarray(row.tangential_loading, dtype=np.float64).copy())
        else:
            donor_query_blocks.append(np.full(d, -1, dtype=np.int32))
            value_loading_blocks.append(np.zeros(d))
            tangential_loading.append(np.zeros((9, 2)))
        donor_ptr.append(donor_ptr[-1] + d)
    donor = np.concatenate(donor_blocks).astype(np.int32) if donor_blocks else np.zeros(0, dtype=np.int32)
    weight = np.concatenate(weight_blocks) if weight_blocks else np.zeros(0)
    donor_query = (np.concatenate(donor_query_blocks).astype(np.int32) if donor_query_blocks
                   else np.zeros(0, dtype=np.int32))
    value_loading = np.concatenate(value_loading_blocks) if value_loading_blocks else np.zeros(0)
    return IntegratedRowChunk(
        entity_id=np.array(entity_id, dtype=np.int64),
        family=np.array(family, dtype=np.int16),
        conditioned=np.array(conditioned, dtype=bool),
        donor_ptr=np.array(donor_ptr, dtype=np.int64),
        donor=donor, weight=weight, donor_query=donor_query, value_loading=value_loading,
        target_points=np.array(target_points, dtype=np.float64).reshape(n, 9, 3),
        tangential_loading=np.array(tangential_loading, dtype=np.float64).reshape(n, 9, 2),
        query_table=table.array(),
    )


def expand_integrated_rows(chunk: IntegratedRowChunk) -> tuple[IntegratedFaceRow, ...]:
    result = []
    for t in range(len(chunk.entity_id)):
        p0, p1 = int(chunk.donor_ptr[t]), int(chunk.donor_ptr[t + 1])
        donor_ids = chunk.donor[p0:p1].astype(np.int64)
        weights = chunk.weight[p0:p1].copy()
        conditioned = bool(chunk.conditioned[t])
        target_points = chunk.target_points[t].copy()
        if conditioned:
            trace_donor_points = chunk.query_table[chunk.donor_query[p0:p1]]
            value_loading = chunk.value_loading[p0:p1].copy()
            tangential_loading = chunk.tangential_loading[t].copy()
        else:
            trace_donor_points = np.empty((0, 3))
            value_loading = np.empty(0)
            tangential_loading = np.empty((0, 2))
        result.append(IntegratedFaceRow(donor_ids, weights, conditioned, trace_donor_points,
                                        target_points, value_loading, tangential_loading,
                                        int(chunk.family[t])))
    return tuple(result)


# --------------------------------------------------------------------------
# Chunk <-> npz-array codecs
# --------------------------------------------------------------------------

_POINT_CHUNK_ARRAY_FIELDS = (
    "request", "entity_id", "quad_node", "family", "conditioned", "bc_variant",
    "radial_degree", "target_point", "donor_ptr", "donor", "value",
    "has_gradient", "gradient_ptr", "gradient", "donor_query", "source_ptr", "query_table",
)
_NEUMANN_CHUNK_ARRAY_FIELDS = (
    "entity_id", "quad_node", "request", "radial_degree", "donor_ptr", "donor", "value", "gradient",
    "wall_query", "boundary_value", "boundary_gradient", "condition",
    "constraint_residual", "query_table",
)
_INTEGRATED_CHUNK_ARRAY_FIELDS = (
    "entity_id", "family", "conditioned", "donor_ptr", "donor", "weight",
    "donor_query", "value_loading", "target_points", "tangential_loading", "query_table",
)


def _point_chunk_to_arrays(chunk: PointRowChunk) -> dict:
    arrays = {name: np.asarray(getattr(chunk, name)) for name in _POINT_CHUNK_ARRAY_FIELDS}
    arrays["source_ptr"] = np.asarray(chunk.source_ptr)
    arrays["source_diagnostics_json"] = np.asarray(chunk.source_diagnostics_json)
    return arrays


def _arrays_to_point_chunk(arrays: dict) -> PointRowChunk:
    kwargs = {name: np.asarray(arrays[name]) for name in _POINT_CHUNK_ARRAY_FIELDS}
    kwargs["source_ptr"] = np.asarray(arrays["source_ptr"])
    kwargs["source_diagnostics_json"] = str(np.asarray(arrays["source_diagnostics_json"]).item())
    return PointRowChunk(**kwargs)


def _neumann_chunk_to_arrays(chunk: NeumannRowChunk) -> dict:
    return {name: np.asarray(getattr(chunk, name)) for name in _NEUMANN_CHUNK_ARRAY_FIELDS}


def _arrays_to_neumann_chunk(arrays: dict) -> NeumannRowChunk:
    kwargs = {name: np.asarray(arrays[name]) for name in _NEUMANN_CHUNK_ARRAY_FIELDS}
    return NeumannRowChunk(**kwargs)


def _integrated_chunk_to_arrays(chunk: IntegratedRowChunk) -> dict:
    return {name: np.asarray(getattr(chunk, name)) for name in _INTEGRATED_CHUNK_ARRAY_FIELDS}


def _arrays_to_integrated_chunk(arrays: dict) -> IntegratedRowChunk:
    kwargs = {name: np.asarray(arrays[name]) for name in _INTEGRATED_CHUNK_ARRAY_FIELDS}
    return IntegratedRowChunk(**kwargs)


_GROUP_CODECS = {
    "cells": (_point_chunk_to_arrays, _arrays_to_point_chunk),
    "faces": (_point_chunk_to_arrays, _arrays_to_point_chunk),
    "neumann": (_neumann_chunk_to_arrays, _arrays_to_neumann_chunk),
    "p07": (_integrated_chunk_to_arrays, _arrays_to_integrated_chunk),
}


# --------------------------------------------------------------------------
# The chunked on-disk artifact: row_artifact/N{n}/manifest.json + rows/*.npz
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RowArtifact:
    """The loaded per-grid row artifact: one chunk tuple per request group."""

    n: int
    identity: dict
    cells: tuple = ()
    faces: tuple = ()
    neumann: tuple = ()
    p07: tuple = ()

    @classmethod
    def load(cls, root, n: int, identity: dict) -> "RowArtifact":
        return load_row_artifact(root, n, identity)

    def save(self, root) -> Path:
        return save_row_artifact(root, self.n, identity=self.identity, cells=self.cells,
                                 faces=self.faces, neumann=self.neumann, p07=self.p07)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    __import__("os").replace(temporary, path)


def save_row_artifact(root, n: int, *, identity: dict, cells=(), faces=(),
                      neumann=(), p07=()) -> Path:
    """Write ``row_artifact/N{n}/manifest.json`` and its ``rows/*.npz`` chunks.

    Every chunk file gets a sha256 recorded in the manifest, alongside
    ``identity`` (schema ``drbx.p-row-artifact.v1``); see ``load_row_artifact``.
    """
    root = Path(root)
    grid_dir = root / f"N{int(n)}"
    groups = {"cells": tuple(cells), "faces": tuple(faces), "neumann": tuple(neumann), "p07": tuple(p07)}
    manifest_chunks: dict = {}
    for group, chunks in groups.items():
        to_arrays, _ = _GROUP_CODECS[group]
        entries = []
        for index, chunk in enumerate(chunks):
            arrays = to_arrays(chunk)
            relative = f"rows/{group}_{index}.npz"
            buffer = io.BytesIO()
            np.savez(buffer, **arrays)
            data = buffer.getvalue()
            _atomic_write_bytes(grid_dir / relative, data)
            entries.append({"file": relative, "sha256": hash_bytes(data)})
        manifest_chunks[group] = entries
    manifest = {"schema": SCHEMA, "identity": _json_safe(identity), "chunks": manifest_chunks}
    _atomic_write_bytes(grid_dir / "manifest.json",
                        json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8"))
    return grid_dir


def load_row_artifact(root, n: int, identity: dict) -> RowArtifact:
    """Load a row artifact, rejecting an identity or per-chunk sha256 mismatch."""
    grid_dir = Path(root) / f"N{int(n)}"
    manifest_path = grid_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no row artifact manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"row artifact schema mismatch: {manifest.get('schema')!r} != {SCHEMA!r}")
    if manifest.get("identity") != _json_safe(identity):
        raise ValueError("row artifact identity mismatch: geometry/topology/policy/platform inputs differ")
    groups: dict = {}
    for group, entries in manifest.get("chunks", {}).items():
        _, from_arrays = _GROUP_CODECS[group]
        chunks = []
        for entry in entries:
            path = grid_dir / entry["file"]
            data = path.read_bytes()
            actual = hash_bytes(data)
            if actual != entry["sha256"]:
                raise ValueError(f"row artifact chunk corrupted: {entry['file']} "
                                 f"(sha256 {actual} != manifest {entry['sha256']})")
            with np.load(io.BytesIO(data), allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            chunks.append(from_arrays(arrays))
        groups[group] = tuple(chunks)
    return RowArtifact(int(n), manifest["identity"], groups.get("cells", ()),
                       groups.get("faces", ()), groups.get("neumann", ()), groups.get("p07", ()))
