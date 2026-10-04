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

On disk (schema v3, P08 step 2; see
``work/p08_step2_layout_loader_design_20260929/design.md`` section 2) the point
chunks are re-keyed by source, and the P07 chunks keep their conditioned-only
arrays compact; decoding reproduces the in-memory chunks bitwise, and the v2
layout (schema v2) is still read.
A v3 point chunk may additionally store its unconditioned singleton / ringwise /
centered_radial sources as exact tensor factors (``src_encoding`` 1, members ``tr_*``;
design section 6, ``drbx.stencils.tensor_rows``), and a ``cell_stencil="symmetric"`` cell row
``1/2 (A + B)`` as the factors of its two parts (``src_encoding`` 2: two consecutive sources of the
``tr_*`` rows, A then B, sharing their tables); decoding expands them (and merges a pair) bitwise,
so the in-memory chunk is the same either way. A v3 file without ``src_encoding`` is all CSR.

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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence

import numpy as np

from drbx.geometry.fci_perpendicular_reconstruction import PairedFactors, PointRows
from drbx.geometry.fci_perpendicular_neumann_trace import NeumannPointRows
from drbx.geometry.fci_perpendicular_integrated_rows import IntegratedFaceRow
from .tensor_rows import TensorRows, expand_tensor_rows, merge_paired_expansion, verified_tensor_rows

SCHEMA = "drbx.p-row-artifact.v3"
#: The previous schema: per-target point chunks and dense P07 side arrays. It is
#: read by ``load_row_artifact`` / ``decode_chunk`` but never written.
SCHEMA_V2 = "drbx.p-row-artifact.v2"
#: Every schema string ``load_row_artifact`` accepts.
SUPPORTED_SCHEMAS = (SCHEMA, SCHEMA_V2)

REQUEST_KINDS = ("R1", "R2", "R3", "R4", "neumann")
BC_VARIANTS = ("", "D", "N")
#: The request kinds a Neumann row's own ``request`` tag may take -- the
#: *originating* request that needed this wall-lattice companion row (see
#: ``drbx.stencils.builder.NeumannRowRequest.source``). Never ``"neumann"``
#: itself or an R-kind that has no Neumann companion.
NEUMANN_SOURCE_KINDS = ("R1", "R2", "R3", "R4")
#: ``src_encoding`` of a v3 point source: its rows are stored as CSR (``src_donor`` ...), as the factors of one
#: tensor source, or as the factors of two consecutive tensor sources A, B whose merge ``1/2 (A + B)`` (the
#: ``cell_stencil="symmetric"`` cell row) is the row. A code is the number of ``TensorRows`` sources it occupies.
ENCODING_CSR, ENCODING_TENSOR, ENCODING_PAIRED = 0, 1, 2


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


# Historical P import remains available and keeps its exact-bit semantics.
from .query_tables import ExactQueryTable as _QueryTable


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


# -- v2: one dense array per ``PointRowChunk`` / ``IntegratedRowChunk`` field ----
# The decoders below are live (v2 files are read); the encoders are private and only
# used by tests to build v2 fixtures -- ``encode_chunk`` writes v3 only.

def _point_chunk_to_arrays_v2(chunk: PointRowChunk) -> dict:
    arrays = {name: np.asarray(getattr(chunk, name)) for name in _POINT_CHUNK_ARRAY_FIELDS}
    arrays["source_ptr"] = np.asarray(chunk.source_ptr)
    arrays["source_diagnostics_json"] = np.asarray(chunk.source_diagnostics_json)
    return arrays


def _arrays_to_point_chunk_v2(arrays: dict) -> PointRowChunk:
    kwargs = {name: np.asarray(arrays[name]) for name in _POINT_CHUNK_ARRAY_FIELDS}
    kwargs["source_ptr"] = np.asarray(arrays["source_ptr"])
    kwargs["source_diagnostics_json"] = str(np.asarray(arrays["source_diagnostics_json"]).item())
    return PointRowChunk(**kwargs)


def _integrated_chunk_to_arrays_v2(chunk: IntegratedRowChunk) -> dict:
    return {name: np.asarray(getattr(chunk, name)) for name in _INTEGRATED_CHUNK_ARRAY_FIELDS}


def _arrays_to_integrated_chunk_v2(arrays: dict) -> IntegratedRowChunk:
    kwargs = {name: np.asarray(arrays[name]) for name in _INTEGRATED_CHUNK_ARRAY_FIELDS}
    return IntegratedRowChunk(**kwargs)


# -- v3: source-major point chunks, conditioned-only P07 side arrays ------------
#
# Lossless: decoding reproduces the in-memory chunk field for field (dtype,
# shape, bytes). The encoders check every invariant the re-keying relies on and
# raise ``ValueError`` on a violation -- nothing lossy is ever stored silently.

#: dtypes the v3 point decoder produces (and hence the only dtypes the encoder
#: accepts for the fields it re-keys).
_POINT_V3_DTYPES = {
    "request": np.dtype("<U8"), "entity_id": np.dtype(np.int64), "family": np.dtype("<U32"),
    "conditioned": np.dtype(bool), "bc_variant": np.dtype("<U1"),
    "radial_degree": np.dtype(np.int16), "has_gradient": np.dtype(bool),
    "donor_ptr": np.dtype(np.int64), "donor": np.dtype(np.int32),
    "gradient_ptr": np.dtype(np.int64), "donor_query": np.dtype(np.int32),
    "source_ptr": np.dtype(np.int64),
}
_INTEGRATED_V3_DTYPES = {
    "conditioned": np.dtype(bool), "donor_ptr": np.dtype(np.int64), "donor": np.dtype(np.int32),
    "donor_query": np.dtype(np.int32), "value_loading": np.dtype(np.float64),
    "tangential_loading": np.dtype(np.float64),
}


def _require_dtypes(chunk, expected: dict, kind: str) -> None:
    for name, dtype in expected.items():
        actual = np.asarray(getattr(chunk, name)).dtype
        if actual != dtype:
            raise ValueError(f"v3 {kind} encoding needs {name} dtype {dtype}, got {actual}")


def _ptr_from_counts(counts: np.ndarray) -> np.ndarray:
    ptr = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, dtype=np.int64, out=ptr[1:])
    return ptr


def _flat_index(base: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``np.repeat(base, counts) + [0..counts[0]), [0..counts[1]), ...`` (int64)."""
    ptr = _ptr_from_counts(counts)
    return np.repeat(np.asarray(base, dtype=np.int64) - ptr[:-1], counts) + np.arange(ptr[-1], dtype=np.int64)


def _point_chunk_to_arrays_v3(chunk: PointRowChunk, factors=None, stats=None) -> dict:
    """v3 members of a point chunk. ``factors`` (one ``PointFactors``, ``PairedFactors`` or ``None`` per source, from
    ``StructuredReconstruction.rows_with_factors``) opts unconditioned sources into the tensor encoding: a
    source is stored as factors only if its bitwise expansion (for a ``PairedFactors``, the merge of its two parts)
    reproduces the chunk's rows, otherwise it stays CSR. ``stats`` (a dict) receives the counts."""
    _require_dtypes(chunk, _POINT_V3_DTYPES, "point chunk")
    source_ptr = np.asarray(chunk.source_ptr)
    n_sources = len(source_ptr) - 1
    if n_sources < 0 or source_ptr[0] != 0:
        raise ValueError("v3 point chunk encoding needs source_ptr to start at 0")
    n_targets = int(source_ptr[-1])
    counts = np.diff(source_ptr)
    if np.any(counts <= 0):
        raise ValueError("v3 point chunk encoding needs at least one target per source")
    first = source_ptr[:-1]
    source_of = np.repeat(np.arange(n_sources), counts)

    def per_target(name):
        array = np.asarray(getattr(chunk, name))
        if array.shape[:1] != (n_targets,):
            raise ValueError(f"v3 point chunk encoding: {name} must have one entry per target")
        return array

    def per_source(name):
        array = per_target(name)
        head = array[first]
        if not np.array_equal(array, head[source_of]):
            raise ValueError(f"v3 point chunk encoding: {name} differs between the targets of one source")
        return head

    def coded(name, table, dtype):
        head = per_source(name)
        codes = np.full(n_sources, -1, dtype=dtype)
        for index, entry in enumerate(table):
            codes[head == entry] = index
        if np.any(codes < 0):
            raise ValueError(f"v3 point chunk encoding: unsupported {name} value")
        return codes

    src_request = coded("request", REQUEST_KINDS, np.int8)
    src_bc_variant = coded("bc_variant", BC_VARIANTS, np.int8)
    table, src_family = np.unique(per_source("family"), return_inverse=True)
    if len(table) > np.iinfo(np.int16).max:
        raise ValueError("v3 point chunk encoding: too many distinct families")
    src_entity_id = per_source("entity_id")
    src_conditioned = per_source("conditioned")
    src_radial_degree = per_source("radial_degree")
    src_has_gradient = per_source("has_gradient")
    per_target("quad_node")
    per_target("target_point")

    donor_ptr = np.asarray(chunk.donor_ptr)
    donor = np.asarray(chunk.donor)
    donor_query = np.asarray(chunk.donor_query)
    widths = np.diff(donor_ptr)
    if (len(donor_ptr) != n_targets + 1 or donor_ptr[0] != 0 or np.any(widths < 0)
            or donor_ptr[-1] != len(donor) or donor_query.shape != donor.shape
            or np.asarray(chunk.value).shape != donor.shape):
        raise ValueError("v3 point chunk encoding: inconsistent donor_ptr/donor/value/donor_query")
    src_width = widths[first]
    if not np.array_equal(widths, src_width[source_of]):
        raise ValueError("v3 point chunk encoding: the targets of one source differ in donor count")
    reference = _flat_index(donor_ptr[first][source_of], widths)
    if not (np.array_equal(donor, donor[reference]) and np.array_equal(donor_query, donor_query[reference])):
        raise ValueError("v3 point chunk encoding: the targets of one source differ in donor list "
                         "or donor_query")
    src_donor_ptr = _ptr_from_counts(src_width)
    take = _flat_index(donor_ptr[first], src_width)
    src_donor = donor[take]
    src_query = donor_query[take]
    entry_conditioned = np.repeat(src_conditioned, src_width)
    if np.any(src_query[~entry_conditioned] != -1):
        raise ValueError("v3 point chunk encoding: an unconditioned source carries donor_query != -1")
    src_donor_query = src_query[entry_conditioned]

    gradient_ptr = _ptr_from_counts(np.where(np.asarray(chunk.has_gradient), widths, 0))
    gradient = np.asarray(chunk.gradient)
    if (not np.array_equal(gradient_ptr, np.asarray(chunk.gradient_ptr)) or gradient.ndim != 2
            or gradient.shape != (3, gradient_ptr[-1])):
        raise ValueError("v3 point chunk encoding: gradient_ptr/gradient do not match has_gradient "
                         "and the donor counts")

    value = np.asarray(chunk.value)
    src_query_ptr = _ptr_from_counts(src_width[src_conditioned])
    extra = {}
    if factors is not None:
        family_names = np.asarray(table)[src_family.reshape(-1)]
        tensor = _select_tensor_sources(
            chunk, factors, stats, family_names=family_names, src_conditioned=src_conditioned,
            src_has_gradient=src_has_gradient, src_width=src_width, src_donor_ptr=src_donor_ptr,
            src_donor=src_donor, first=first, counts=counts, gradient=gradient)
        if tensor is not None:
            rows, encoding = tensor
            is_tensor = encoding > 0
            entry_csr = np.repeat(~is_tensor, src_width)
            target_csr = np.repeat(~is_tensor, counts)
            has_gradient = np.asarray(chunk.has_gradient)
            entry_target_csr = np.repeat(target_csr, widths)
            gradient_csr = np.repeat(target_csr[has_gradient], widths[has_gradient])
            src_donor = src_donor[entry_csr]
            src_donor_ptr = _ptr_from_counts(np.where(is_tensor, 0, src_width))
            value, gradient = value[entry_target_csr], gradient[:, gradient_csr]
            extra = {"src_encoding": encoding, **rows.to_arrays()}

    return {
        "src_request": src_request,
        "src_entity_id": src_entity_id,
        "src_family": src_family.astype(np.int16).reshape(-1),
        "family_table": np.asarray(json.dumps([str(name) for name in table.tolist()])),
        "src_conditioned": src_conditioned,
        "src_bc_variant": src_bc_variant,
        "src_radial_degree": src_radial_degree,
        "src_has_gradient": src_has_gradient,
        "source_ptr": source_ptr,
        "src_donor_ptr": src_donor_ptr,
        "src_donor": src_donor,
        "src_donor_query_ptr": src_query_ptr,
        "src_donor_query": src_donor_query,
        "quad_node": np.asarray(chunk.quad_node),
        "target_point": np.asarray(chunk.target_point),
        "value": value,
        "gradient": gradient,
        "query_table": np.asarray(chunk.query_table),
        "source_diagnostics_json": np.asarray(chunk.source_diagnostics_json),
        **extra,
    }


def _candidate_family(factors):
    """The family a captured candidate is stored as: a ``PointFactors``' own; a ``PairedFactors``' common family of
    A and B, ``None`` if they differ (such a pair stays CSR)."""
    if isinstance(factors, PairedFactors):
        return factors.a.family if factors.a.family == factors.b.family else None
    return factors.family


def _select_tensor_sources(chunk, factors, stats, *, family_names, src_conditioned, src_has_gradient, src_width,
                           src_donor_ptr, src_donor, first, counts, gradient):
    """``(TensorRows, encoding (S,) int8)`` for the candidate sources whose bitwise expansion reproduces the
    chunk's rows, or ``None`` if there are none. ``encoding`` is ``ENCODING_TENSOR`` for a source stored as one
    tensor source, ``ENCODING_PAIRED`` for a source stored as two (a ``PairedFactors``: A then B, consecutive in the
    ``TensorRows``, whose merge is the row), ``ENCODING_CSR`` otherwise. Candidates: unconditioned sources with
    captured factors of their own family (both parts of a pair). Failing candidates stay CSR and are counted in
    ``stats``: ``tensor_sources`` (stored as factors, a pair included) + ``fallback_sources`` = ``candidate_sources``,
    and ``paired_sources`` (``paired_targets``, ``paired_by_family``) is the part of the tensor sources stored as pairs."""
    factors = list(factors)
    n_sources = len(first)
    if len(factors) != n_sources:
        raise ValueError("v3 point chunk encoding: factors must supply exactly one entry (or None) per source")
    with_factors = np.array([s for s, f in enumerate(factors) if f is not None and not src_conditioned[s]], dtype=np.int64)
    candidate = np.array([s for s in with_factors if _candidate_family(factors[s]) == family_names[s]], dtype=np.int64)
    donor_ptr = np.asarray(chunk.donor_ptr)
    gradient_ptr = np.asarray(chunk.gradient_ptr)
    value = np.asarray(chunk.value)

    def expected_for(indices):
        sel = candidate[indices]
        width, count = src_width[sel], counts[sel]
        span = count * width
        has_gradient = src_has_gradient[sel]
        return {
            "donor_ptr": _ptr_from_counts(width),
            "donor": src_donor[_flat_index(src_donor_ptr[sel], width)].astype(np.int64),
            "value": value[_flat_index(donor_ptr[first[sel]], span)],
            "gradient": gradient[:, _flat_index(gradient_ptr[first[sel]][has_gradient], span[has_gradient])],
        }

    rows, accepted = (verified_tensor_rows([factors[s] for s in candidate], src_has_gradient[candidate], expected_for)
                      if len(candidate) else (None, np.zeros(0, dtype=bool)))
    encoding = np.zeros(n_sources, dtype=np.int8)
    paired = np.array([isinstance(factors[s], PairedFactors) for s in candidate], dtype=bool)
    encoding[candidate[accepted]] = np.where(paired[accepted], ENCODING_PAIRED, ENCODING_TENSOR)
    if stats is not None:
        def bump(key, amount):
            stats[key] = stats.get(key, 0) + amount
        stored, pair = encoding > 0, encoding == ENCODING_PAIRED
        bump("sources", n_sources)
        bump("candidate_sources", len(with_factors))
        bump("tensor_sources", int(stored.sum()))
        bump("paired_sources", int(pair.sum()))
        bump("fallback_sources", int(len(with_factors) - stored.sum()))
        bump("tensor_targets", int(counts[stored].sum()))
        bump("paired_targets", int(counts[pair].sum()))
        rejected = np.setdiff1d(with_factors, candidate[accepted])
        for name in np.unique(family_names[with_factors]) if len(with_factors) else ():
            for label, count in (("tensor", int((stored & (family_names == name)).sum())),
                                 ("paired", int((pair & (family_names == name)).sum())),
                                 ("fallback", int((np.isin(np.arange(n_sources), rejected) & (family_names == name)).sum()))):
                by = stats.setdefault(f"{label}_by_family", {})
                by[str(name)] = by.get(str(name), 0) + count
    return (rows, encoding) if (encoding > 0).any() else None


def _tensor_rows(arrays, encoding, counts, src_has_gradient) -> TensorRows:
    """The chunk's ``TensorRows`` from its ``tr_*`` members: one source per ``ENCODING_TENSOR`` source and two (A
    then B) per ``ENCODING_PAIRED`` source, in chunk order (a source's code is the number of sources it occupies)."""
    rows = TensorRows.from_arrays(arrays, has_gradient=np.repeat(src_has_gradient, encoding),
                                  target_counts=np.repeat(counts, encoding))
    a = (np.cumsum(encoding) - encoding)[encoding == ENCODING_PAIRED]
    if len(a) and np.any(rows.family[a] != rows.family[a + 1]):
        raise ValueError("corrupted v3 point-row chunk: the two parts of a paired source differ in family")
    return rows


def _merge_tensor_sources(arrays, encoding, codes, counts, src_has_gradient, src_width, src_donor, value, gradient):
    """Expand the tensor sources whose code is in ``codes`` (merging a pair) and interleave them with the stored CSR
    sources: the full ``(src_donor_ptr, src_donor, value, gradient, src_width)`` of the chunk, in source order. A
    tensor source with another code stays as stored (no donors)."""
    expand = np.isin(encoding, codes)
    tensor = _tensor_rows(arrays, encoding, counts, src_has_gradient)
    multiplicity = encoding[expand]
    if not np.array_equal(expand, encoding > 0):                 # only some of the tensor sources: their rows
        tensor = tensor.take_sources(_flat_index((np.cumsum(encoding) - encoding)[expand], multiplicity))
    expansion = merge_paired_expansion(expand_tensor_rows(tensor), np.repeat(counts[expand], multiplicity),
                                       np.repeat(src_has_gradient[expand], multiplicity), multiplicity)
    width = src_width.copy()
    width[expand] = np.diff(expansion.donor_ptr)
    ptr = _ptr_from_counts(width)
    entry_csr = np.repeat(~expand, width)
    donor = np.empty(ptr[-1], dtype=np.int32)
    donor[entry_csr] = src_donor
    donor[~entry_csr] = expansion.donor
    widths = np.repeat(width, counts)
    target_csr = np.repeat(~expand, counts)
    value_csr = np.repeat(target_csr, widths)
    full_value = np.empty(int(widths.sum()))
    full_value[value_csr] = value
    full_value[~value_csr] = expansion.value
    has_gradient = np.repeat(src_has_gradient, counts)
    gradient_csr = np.repeat(target_csr[has_gradient], widths[has_gradient])
    full_gradient = np.empty((3, len(gradient_csr)))
    full_gradient[:, gradient_csr] = gradient
    full_gradient[:, ~gradient_csr] = expansion.gradient
    return ptr, donor, full_value, full_gradient, width


def _arrays_to_point_chunk_v3(arrays: dict, expand: bool = True,
                              codes: tuple = (ENCODING_TENSOR, ENCODING_PAIRED)) -> PointRowChunk:
    """The chunk of v3 members. With ``expand`` the tensor sources whose ``src_encoding`` is in ``codes`` are expanded
    (a pair merged) into CSR; any other tensor source is left as stored, with no donors, value or gradient."""
    source_ptr = np.asarray(arrays["source_ptr"], dtype=np.int64)
    counts = np.diff(source_ptr)
    n_sources = len(counts)
    n_targets = int(source_ptr[-1])
    if len(arrays["target_point"]) != n_targets or len(arrays["quad_node"]) != n_targets:
        raise ValueError("corrupted v3 point-row chunk: target counts disagree with source_ptr")
    table = json.loads(str(np.asarray(arrays["family_table"]).item()))

    def repeated(values):
        return np.repeat(values, counts)

    src_donor_ptr = np.asarray(arrays["src_donor_ptr"], dtype=np.int64)
    src_width = np.diff(src_donor_ptr)
    src_conditioned = np.asarray(arrays["src_conditioned"], dtype=bool)
    src_donor = np.asarray(arrays["src_donor"], dtype=np.int32)
    src_donor_query = np.asarray(arrays["src_donor_query"], dtype=np.int32)
    if (len(src_width) != n_sources or src_donor_ptr[-1] != len(src_donor)
            or src_width[src_conditioned].sum() != len(src_donor_query)
            or not np.array_equal(_ptr_from_counts(src_width[src_conditioned]),
                                  np.asarray(arrays["src_donor_query_ptr"], dtype=np.int64))):
        raise ValueError("corrupted v3 point-row chunk: inconsistent source donor pointers")
    src_has_gradient = np.asarray(arrays["src_has_gradient"], dtype=bool)
    value, gradient = np.asarray(arrays["value"]), np.asarray(arrays["gradient"])
    if "src_encoding" in arrays:
        encoding = np.asarray(arrays["src_encoding"], dtype=np.int64)
        is_tensor = encoding > 0
        if (len(encoding) != n_sources or np.any((encoding < 0) | (encoding > ENCODING_PAIRED))
                or np.any(src_width[is_tensor] != 0) or np.any(src_conditioned[is_tensor])):
            raise ValueError("corrupted v3 point-row chunk: inconsistent src_encoding")
        if expand and np.isin(encoding, codes).any():
            src_donor_ptr, src_donor, value, gradient, src_width = _merge_tensor_sources(
                arrays, encoding, codes, counts, src_has_gradient, src_width, src_donor, value, gradient)
    entry_conditioned = np.repeat(src_conditioned, src_width)
    src_query = np.full(len(src_donor), -1, dtype=np.int32)
    src_query[entry_conditioned] = src_donor_query

    widths = repeated(src_width)
    donor_ptr = _ptr_from_counts(widths)
    take = _flat_index(repeated(src_donor_ptr[:-1]), widths)
    has_gradient = repeated(src_has_gradient)

    return PointRowChunk(
        request=repeated(np.array(REQUEST_KINDS, dtype="<U8")[np.asarray(arrays["src_request"])]),
        entity_id=repeated(np.asarray(arrays["src_entity_id"], dtype=np.int64)),
        quad_node=np.asarray(arrays["quad_node"]),
        family=repeated(np.asarray(table, dtype="<U32")[np.asarray(arrays["src_family"])]),
        conditioned=repeated(src_conditioned),
        bc_variant=repeated(np.array(BC_VARIANTS, dtype="<U1")[np.asarray(arrays["src_bc_variant"])]),
        radial_degree=repeated(np.asarray(arrays["src_radial_degree"], dtype=np.int16)),
        target_point=np.asarray(arrays["target_point"]),
        donor_ptr=donor_ptr,
        donor=src_donor[take],
        value=value,
        has_gradient=has_gradient,
        gradient_ptr=_ptr_from_counts(np.where(has_gradient, widths, 0)),
        gradient=gradient,
        donor_query=src_query[take],
        source_ptr=source_ptr,
        source_diagnostics_json=str(np.asarray(arrays["source_diagnostics_json"]).item()),
        query_table=np.asarray(arrays["query_table"]),
    )


def _integrated_chunk_to_arrays_v3(chunk: IntegratedRowChunk) -> dict:
    _require_dtypes(chunk, _INTEGRATED_V3_DTYPES, "integrated chunk")
    conditioned = np.asarray(chunk.conditioned)
    donor_ptr = np.asarray(chunk.donor_ptr)
    donor_query = np.asarray(chunk.donor_query)
    value_loading = np.asarray(chunk.value_loading)
    tangential = np.asarray(chunk.tangential_loading)
    n = len(conditioned)
    widths = np.diff(donor_ptr)
    if (len(donor_ptr) != n + 1 or donor_ptr[0] != 0 or np.any(widths < 0)
            or donor_ptr[-1] != len(np.asarray(chunk.donor)) or donor_query.shape != (donor_ptr[-1],)
            or value_loading.shape != (donor_ptr[-1],) or tangential.shape != (n, 9, 2)):
        raise ValueError("v3 integrated chunk encoding: inconsistent donor_ptr/donor_query/loading shapes")
    entry_conditioned = np.repeat(conditioned, widths)
    if np.any(donor_query[~entry_conditioned] != -1):
        raise ValueError("v3 integrated chunk encoding: an unconditioned row carries donor_query != -1")
    if np.any(np.ascontiguousarray(value_loading[~entry_conditioned]).view(np.uint64) != 0):
        raise ValueError("v3 integrated chunk encoding: an unconditioned row carries value_loading != +0.0")
    if np.any(np.ascontiguousarray(tangential[~conditioned]).view(np.uint64) != 0):
        raise ValueError("v3 integrated chunk encoding: an unconditioned row carries tangential_loading != +0.0")
    arrays = {name: np.asarray(getattr(chunk, name)) for name in _INTEGRATED_CHUNK_ARRAY_FIELDS}
    for name in ("donor_query", "value_loading", "tangential_loading"):
        del arrays[name]
    arrays["cond_donor_ptr"] = _ptr_from_counts(widths[conditioned])
    arrays["cond_donor_query"] = donor_query[entry_conditioned]
    arrays["cond_value_loading"] = value_loading[entry_conditioned]
    arrays["cond_tangential_loading"] = tangential[conditioned]
    return arrays


def _arrays_to_integrated_chunk_v3(arrays: dict) -> IntegratedRowChunk:
    conditioned = np.asarray(arrays["conditioned"], dtype=bool)
    donor_ptr = np.asarray(arrays["donor_ptr"], dtype=np.int64)
    widths = np.diff(donor_ptr)
    cond_donor_query = np.asarray(arrays["cond_donor_query"], dtype=np.int32)
    cond_value_loading = np.asarray(arrays["cond_value_loading"], dtype=np.float64)
    cond_tangential = np.asarray(arrays["cond_tangential_loading"], dtype=np.float64)
    if (not np.array_equal(_ptr_from_counts(widths[conditioned]), np.asarray(arrays["cond_donor_ptr"], dtype=np.int64))
            or cond_donor_query.shape != cond_value_loading.shape
            or cond_donor_query.shape != (widths[conditioned].sum(),)
            or cond_tangential.shape != (int(conditioned.sum()), 9, 2)):
        raise ValueError("corrupted v3 integrated-row chunk: conditioned side arrays disagree with donor_ptr")
    entry_conditioned = np.repeat(conditioned, widths)
    donor_query = np.full(int(donor_ptr[-1]), -1, dtype=np.int32)
    donor_query[entry_conditioned] = cond_donor_query
    value_loading = np.zeros(int(donor_ptr[-1]), dtype=np.float64)
    value_loading[entry_conditioned] = cond_value_loading
    tangential = np.zeros((len(conditioned), 9, 2), dtype=np.float64)
    tangential[conditioned] = cond_tangential
    kwargs = {name: np.asarray(arrays[name]) for name in _INTEGRATED_CHUNK_ARRAY_FIELDS
              if name not in ("donor_query", "value_loading", "tangential_loading")}
    return IntegratedRowChunk(donor_query=donor_query, value_loading=value_loading,
                              tangential_loading=tangential, **kwargs)


# -- Neumann: unchanged in v3 ---------------------------------------------------

def _neumann_chunk_to_arrays(chunk: NeumannRowChunk) -> dict:
    return {name: np.asarray(getattr(chunk, name)) for name in _NEUMANN_CHUNK_ARRAY_FIELDS}


def _arrays_to_neumann_chunk(arrays: dict) -> NeumannRowChunk:
    kwargs = {name: np.asarray(arrays[name]) for name in _NEUMANN_CHUNK_ARRAY_FIELDS}
    return NeumannRowChunk(**kwargs)


# -- schema dispatch: encoders write the current schema; decoders read either ---

def _point_chunk_to_arrays(chunk: PointRowChunk, factors=None, stats=None) -> dict:
    return _point_chunk_to_arrays_v3(chunk, factors, stats)


def _arrays_to_point_chunk(arrays: dict) -> PointRowChunk:
    """Decode either layout (v3 is recognized by its source-major ``src_request``)."""
    return _arrays_to_point_chunk_v3(arrays) if "src_request" in arrays else _arrays_to_point_chunk_v2(arrays)


def _integrated_chunk_to_arrays(chunk: IntegratedRowChunk) -> dict:
    return _integrated_chunk_to_arrays_v3(chunk)


def _arrays_to_integrated_chunk(arrays: dict) -> IntegratedRowChunk:
    """Decode either layout (v3 is recognized by ``cond_donor_query``)."""
    return (_arrays_to_integrated_chunk_v3(arrays) if "cond_donor_query" in arrays
            else _arrays_to_integrated_chunk_v2(arrays))


_GROUP_CODECS = {
    "cells": (_point_chunk_to_arrays, _arrays_to_point_chunk),
    "faces": (_point_chunk_to_arrays, _arrays_to_point_chunk),
    "neumann": (_neumann_chunk_to_arrays, _arrays_to_neumann_chunk),
    "p07": (_integrated_chunk_to_arrays, _arrays_to_integrated_chunk),
}


def encode_chunk(group: str, chunk, *, factors=None, stats=None) -> bytes:
    """The uncompressed ``np.savez`` bytes of one chunk in the current (v3) layout.

    ``factors`` (point chunks only): one ``PointFactors``, ``PairedFactors`` or ``None`` per source, see
    ``_point_chunk_to_arrays_v3``; ``stats`` (a dict) accumulates the tensor/paired/fallback source counts."""
    to_arrays, _ = _GROUP_CODECS[group]
    if factors is not None and group not in ("cells", "faces"):
        raise ValueError("factors apply to point chunks (cells, faces) only")
    buffer = io.BytesIO()
    np.savez(buffer, **(to_arrays(chunk, factors=factors, stats=stats) if factors is not None
                        else to_arrays(chunk)))
    return buffer.getvalue()


def decode_chunk(group: str, data: bytes):
    """The in-memory chunk stored in ``data`` (either schema's layout)."""
    _, from_arrays = _GROUP_CODECS[group]
    with np.load(io.BytesIO(data), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    return from_arrays(arrays)


def decode_chunk_factored(group: str, data: bytes):
    """``(chunk, tensor_rows, is_tensor)`` of a point chunk *without* expanding its single tensor sources.

    ``chunk`` is the stored view: every tag, ``source_ptr``, ``quad_node`` and ``target_point`` as in
    ``decode_chunk``, but a tensor source (``src_encoding`` 1) has zero donors and stores no value or gradient (the
    CSR arrays hold the other sources only). ``tensor_rows`` (a ``drbx.stencils.tensor_rows.TensorRows``, ``None``
    if the chunk has no such source) holds those sources in chunk order; ``is_tensor`` is the per-source mask.
    Expanding them (``expand_tensor_rows``) and interleaving gives exactly ``decode_chunk``'s chunk.

    A paired source (``src_encoding`` 2, a ``cell_stencil="symmetric"`` cell row stored as its two parts) is not
    exposed as factors here: it is expanded and merged, so it is an ordinary CSR source of ``chunk`` (donors, value
    and gradient as in ``decode_chunk``) and ``is_tensor`` is False for it."""
    if group not in ("cells", "faces"):
        raise ValueError("only point chunks (cells, faces) carry tensor sources")
    with np.load(io.BytesIO(data), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if "src_request" not in arrays:
        return _arrays_to_point_chunk_v2(arrays), None, np.zeros(len(arrays["source_ptr"]) - 1, dtype=bool)
    encoding = (np.asarray(arrays["src_encoding"], dtype=np.int64) if "src_encoding" in arrays
                else np.zeros(len(arrays["source_ptr"]) - 1, dtype=np.int64))
    chunk = _arrays_to_point_chunk_v3(arrays, codes=(ENCODING_PAIRED,))
    is_tensor = encoding == ENCODING_TENSOR
    tensor = None
    if is_tensor.any():
        counts = np.diff(np.asarray(arrays["source_ptr"], dtype=np.int64))
        tensor = _tensor_rows(arrays, encoding, counts, np.asarray(arrays["src_has_gradient"], dtype=bool))
        if (encoding == ENCODING_PAIRED).any():                   # only the single sources' rows
            tensor = tensor.take_sources((np.cumsum(encoding) - encoding)[is_tensor])
    return chunk, tensor, is_tensor


def chunk_counts(group: str, chunk) -> tuple[int, int]:
    """``(sources, targets)``: a point chunk's ``PointRows`` and evaluated points;
    one of each per row for the Neumann and integrated chunks."""
    if group in ("cells", "faces"):
        return len(chunk.source_ptr) - 1, int(chunk.source_ptr[-1])
    return len(chunk.entity_id), len(chunk.entity_id)


def chunk_file_stats(group: str, path) -> dict:
    """``{"sources", "targets", "bytes"}`` of a chunk file, reading only its small
    index member (npz members load lazily), for either schema."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as source:
        if group in ("cells", "faces"):
            source_ptr = source["source_ptr"]
            sources, targets = len(source_ptr) - 1, int(source_ptr[-1])
        else:
            sources = targets = len(source["entity_id"])
    return {"sources": sources, "targets": targets, "bytes": path.stat().st_size}


def chunk_mismatches(actual, expected) -> list[str]:
    """Names of the chunk fields that differ in dtype, shape or bytes (empty if the
    two chunks are identical, bit for bit)."""
    if type(actual) is not type(expected):
        return [f"type {type(actual).__name__} != {type(expected).__name__}"]
    bad = []
    for field in fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(b, str):
            if a != b:
                bad.append(field.name)
            continue
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape or np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
            bad.append(field.name)
    return bad


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
                      neumann=(), p07=(), point_factors=None, stats=None) -> Path:
    """Write ``row_artifact/N{n}/manifest.json`` and its ``rows/*.npz`` chunks.

    Every chunk file gets a sha256 recorded in the manifest, alongside
    ``identity`` and ``schema`` (always the current v3 layout); see
    ``load_row_artifact``. Each entry also records the chunk's
    ``sources``, ``targets`` and file ``bytes``. ``point_factors`` (optional,
    ``{"cells": [...], "faces": [...]}``, one per-source factors sequence per chunk) stores the
    factored sources of those point chunks as tensors; ``stats`` accumulates the counts.
    """
    root = Path(root)
    grid_dir = root / f"N{int(n)}"
    groups = {"cells": tuple(cells), "faces": tuple(faces), "neumann": tuple(neumann), "p07": tuple(p07)}
    manifest_chunks: dict = {}
    for group, chunks in groups.items():
        entries = []
        for index, chunk in enumerate(chunks):
            relative = f"rows/{group}_{index}.npz"
            chunk_factors = (point_factors or {}).get(group)
            data = encode_chunk(group, chunk, stats=stats,
                                factors=None if chunk_factors is None else chunk_factors[index])
            _atomic_write_bytes(grid_dir / relative, data)
            sources, targets = chunk_counts(group, chunk)
            entry = {"file": relative, "sha256": hash_bytes(data),
                     "sources": sources, "targets": targets, "bytes": len(data)}
            entries.append(entry)
        manifest_chunks[group] = entries
    manifest = {"schema": SCHEMA, "identity": _json_safe(identity), "chunks": manifest_chunks}
    _atomic_write_bytes(grid_dir / "manifest.json",
                        json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8"))
    return grid_dir


def load_row_artifact(root, n: int, identity: dict) -> RowArtifact:
    """Load a row artifact (v3, or the older v2 layout), rejecting a schema,
    identity or per-chunk sha256 mismatch."""
    grid_dir = Path(root) / f"N{int(n)}"
    manifest_path = grid_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no row artifact manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError(f"row artifact schema mismatch: {manifest.get('schema')!r} not in {SUPPORTED_SCHEMAS!r}")
    if manifest.get("identity") != _json_safe(identity):
        raise ValueError("row artifact identity mismatch: geometry/topology/policy/platform inputs differ")
    groups: dict = {}
    for group, entries in manifest.get("chunks", {}).items():
        chunks = []
        for entry in entries:
            data = (grid_dir / entry["file"]).read_bytes()
            actual = hash_bytes(data)
            if actual != entry["sha256"]:
                raise ValueError(f"row artifact chunk corrupted: {entry['file']} "
                                 f"(sha256 {actual} != manifest {entry['sha256']})")
            chunks.append(decode_chunk(group, data))
        groups[group] = tuple(chunks)
    return RowArtifact(int(n), manifest["identity"], groups.get("cells", ()),
                       groups.get("faces", ()), groups.get("neumann", ()), groups.get("p07", ()))
