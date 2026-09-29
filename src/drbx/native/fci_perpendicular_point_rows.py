"""Array-only JAX application of host-prepared perpendicular point rows.

The owner field and prescribed boundary arrays are runtime inputs. No campaign,
metric, fitting, analytic field, or Python boundary callback enters this kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from drbx.geometry.fci_perpendicular_reconstruction import PointRowContext, PointRows


class PointRowBatch(NamedTuple):
    target_ids: np.ndarray
    donor_ids: np.ndarray
    value_weights: np.ndarray
    gradient_weights: np.ndarray
    donor_query_ids: np.ndarray
    target_query_ids: np.ndarray
    conditioned: np.ndarray


class PointRowPayload(NamedTuple):
    batches: tuple[PointRowBatch, ...]
    output_template: np.ndarray
    boundary_query_template: np.ndarray


class BoundaryArrays(NamedTuple):
    """Prescribed g and its two tangential derivatives at boundary queries."""

    values: np.ndarray
    tangential_gradients: np.ndarray


@dataclass(frozen=True)
class PointRowPlan:
    payload: PointRowPayload
    boundary_points: np.ndarray
    target_points: np.ndarray
    row_metadata: tuple[dict, ...]
    donor_eta_offset_sets: tuple[tuple[tuple[int, ...], ...], ...]
    bucket_summary: tuple[tuple[str, int, int, int], ...]
    boundary_kind: str = "dirichlet"
    boundary_kinds: tuple[str, ...] | None = None


def _bucket_size(donor_count: int) -> int:
    return 16 * ((int(donor_count) + 15) // 16) if donor_count else 0


def lower_point_rows(context: PointRowContext, rows: tuple[PointRows, ...] | list[PointRows],
                     *, boundary_kind: str = "dirichlet",
                     boundary_kinds: tuple[str, ...] | None = None) -> PointRowPlan:
    """Lower only requested rows into donor-count buckets and exact BC queries.

    ``boundary_kind`` is fixed during preparation. Unsupported conditioned
    kinds fail rather than silently borrowing the Dirichlet lift.
    """
    if boundary_kind not in ("dirichlet", "none"):
        raise ValueError(f"unsupported boundary kind: {boundary_kind}")
    if boundary_kinds is not None:
        boundary_kinds = tuple(boundary_kinds)
        if not boundary_kinds or any(kind not in ("dirichlet", "none") for kind in boundary_kinds):
            raise ValueError("per-field boundary kinds must be explicit Dirichlet or none")
        if any(kind != boundary_kind for kind in boundary_kinds):
            raise ValueError("mixed boundary kinds require separate prepared point-row plans")
    queries: list[np.ndarray] = []
    query_ids: dict[tuple[float, float, float], int] = {}

    def query_id(point):
        key = tuple(map(float, point))
        if key not in query_ids:
            query_ids[key] = len(queries)
            queries.append(np.asarray(point, dtype=np.float64).copy())
        return query_ids[key]

    flat = []
    points = []
    metadata = []
    eta_sets = []
    for source_id, row in enumerate(rows):
        if row.boundary_conditioned and boundary_kind != "dirichlet":
            raise ValueError("conditioned point row requires a fixed Dirichlet boundary kind")
        donor = np.asarray(row.donor_ids, dtype=np.int64)
        if np.any(donor < 0) or np.any(donor >= len(context.vol)):
            raise ValueError("point row has invalid compact donor IDs")
        if row.value.shape != (len(row.trace_target_points), len(donor)) or row.gradient.shape != (len(row.trace_target_points), 3, len(donor)):
            raise ValueError("point row coefficient shapes are inconsistent")
        for local_id, point in enumerate(row.trace_target_points):
            target_id = len(points)
            points.append(np.asarray(point, dtype=np.float64))
            donor_queries = ([query_id(q) for q in row.trace_donor_points]
                             if row.boundary_conditioned else [0]*len(donor))
            target_query = query_id(point) if row.boundary_conditioned else 0
            # Record every member plane. An aggregate owner is never split into
            # extra evolved unknowns, even if a future topology spans planes.
            target_plane = int(np.argmin(np.abs(context.centers[2]-point[2])))
            offsets = []
            for oid in donor:
                planes = np.unique(context.members(int(oid)) % context.n)
                offsets.append(tuple(sorted(int((int(k)-target_plane+context.n//2)%context.n-context.n//2)
                                            for k in planes)))
            eta_sets.append(tuple(offsets))
            flat.append((target_id, donor, np.asarray(row.value[local_id]),
                         np.asarray(row.gradient[local_id]), donor_queries,
                         target_query, bool(row.boundary_conditioned),
                         str(row.diagnostics.get("family", "unknown"))))
            metadata.append({"source_request": source_id, "source_point": local_id,
                             "family": row.diagnostics.get("family"),
                             "boundary_conditioned": bool(row.boundary_conditioned),
                             **row.diagnostics})
    buckets: dict[tuple[str, bool, int], list] = {}
    for item in flat:
        buckets.setdefault((item[-1], item[-2], _bucket_size(len(item[1]))), []).append(item)
    prepared = []
    summary = []
    for (family, conditioned, width), items in sorted(buckets.items(), key=lambda entry: entry[0]):
        count = len(items)
        ids = np.zeros((count, width), dtype=np.int64)
        values = np.zeros((count, width), dtype=np.float64)
        gradients = np.zeros((count, 3, width), dtype=np.float64)
        donor_queries = np.zeros((count, width), dtype=np.int64)
        targets = np.empty(count, dtype=np.int64)
        target_queries = np.zeros(count, dtype=np.int64)
        for q, (target, donor, value, gradient, dq, tq, _, _) in enumerate(items):
            d = len(donor)
            targets[q] = target
            ids[q, :d] = donor
            values[q, :d] = value
            gradients[q, :, :d] = gradient
            donor_queries[q, :d] = dq
            target_queries[q] = tq
        prepared.append(PointRowBatch(targets, ids, values, gradients,
                                      donor_queries, target_queries,
                                      np.full(count, conditioned, dtype=bool)))
        summary.append((family, count, width, sum(len(item[1]) for item in items)))
    boundary_points = np.asarray(queries, dtype=np.float64).reshape((-1, 3))
    target_points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    payload = PointRowPayload(tuple(prepared), np.zeros(len(points), dtype=np.float64),
                              np.zeros(len(boundary_points), dtype=np.float64))
    return PointRowPlan(payload, boundary_points, target_points, tuple(metadata),
                        tuple(eta_sets), tuple(summary), boundary_kind, boundary_kinds)


def save_point_row_plan(path: str | Path, plan: PointRowPlan, identity: dict) -> None:
    """Atomically cache only prepared arrays, exact query points, and identity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"schema": "drbx.perpendicular-point-rows.v1", "identity": identity,
                "row_metadata": plan.row_metadata,
                "donor_eta_offset_sets": plan.donor_eta_offset_sets,
                "bucket_summary": plan.bucket_summary,
                "boundary_kind": plan.boundary_kind,
                "boundary_kinds": plan.boundary_kinds,
                "batch_count": len(plan.payload.batches)}
    arrays = {"metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
              "boundary_points": plan.boundary_points,
              "target_points": plan.target_points}
    for q, batch in enumerate(plan.payload.batches):
        for name, array in zip(PointRowBatch._fields, batch, strict=True):
            arrays[f"batch_{q}_{name}"] = np.asarray(array)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_point_row_plan(path: str | Path, expected_identity: dict) -> PointRowPlan:
    """Reject stale caches before yielding an array-only JAX payload."""
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(np.asarray(source["metadata_json"]).item()))
        if metadata.get("schema") != "drbx.perpendicular-point-rows.v1":
            raise ValueError("unsupported point-row cache schema")
        if metadata.get("identity") != expected_identity:
            raise ValueError("point-row cache identity differs from requested inputs/policy")
        batches = tuple(PointRowBatch(*(np.asarray(source[f"batch_{q}_{name}"])
                                        for name in PointRowBatch._fields))
                        for q in range(int(metadata["batch_count"])))
        boundary_points = np.asarray(source["boundary_points"])
        target_points = np.asarray(source["target_points"])
    return PointRowPlan(PointRowPayload(batches, np.zeros(len(target_points)),
                                       np.zeros(len(boundary_points))),
                        boundary_points, target_points,
                        tuple(metadata["row_metadata"]),
                        tuple(tuple(tuple(int(v) for v in donor) for donor in row)
                              for row in metadata["donor_eta_offset_sets"]),
                        tuple(tuple(row) for row in metadata["bucket_summary"]),
                        metadata["boundary_kind"],
                        None if metadata["boundary_kinds"] is None else tuple(metadata["boundary_kinds"]))


def _validate_inputs(payload: PointRowPayload, owner_fields, boundary):
    if owner_fields.ndim != 2:
        raise ValueError("owner_fields must have shape (owners, fields)")
    if payload.boundary_query_template.shape[0]:
        if boundary is None:
            raise ValueError("conditioned point rows require explicit boundary arrays")
        expected = (payload.boundary_query_template.shape[0], owner_fields.shape[1])
        if boundary.values.shape != expected or boundary.tangential_gradients.shape != (expected[0], 2, expected[1]):
            raise ValueError("boundary values/tangential gradients have incompatible shapes")


def apply_point_rows(payload: PointRowPayload, owner_fields, boundary: BoundaryArrays | None = None,
                     *, values: bool = True, gradients: bool = True):
    """Apply rows to batched owner fields, with no exact normal BC derivative."""
    if not values and not gradients:
        raise ValueError("request values, gradients, or both")
    fields = jnp.asarray(owner_fields)
    _validate_inputs(payload, fields, boundary)
    if boundary is None:
        bc_values = jnp.zeros((1, fields.shape[1]), dtype=fields.dtype)
        bc_tangent = jnp.zeros((1, 2, fields.shape[1]), dtype=fields.dtype)
    else:
        bc_values = jnp.asarray(boundary.values)
        bc_tangent = jnp.asarray(boundary.tangential_gradients)
    count = payload.output_template.shape[0]
    out_value = jnp.zeros((count, fields.shape[1]), dtype=fields.dtype) if values else None
    out_gradient = jnp.zeros((count, 3, fields.shape[1]), dtype=fields.dtype) if gradients else None
    for batch in payload.batches:
        data = fields[batch.donor_ids]
        conditioned = jnp.asarray(batch.conditioned)
        data = data-jnp.where(conditioned[:, None, None], bc_values[batch.donor_query_ids], 0.0)
        if values:
            v = jnp.einsum("rd,rdf->rf", batch.value_weights, data)
            v = v+jnp.where(conditioned[:, None], bc_values[batch.target_query_ids], 0.0)
            out_value = out_value.at[batch.target_ids].set(v)
        if gradients:
            g = jnp.einsum("rad,rdf->raf", batch.gradient_weights, data)
            tangential = jnp.where(conditioned[:, None, None], bc_tangent[batch.target_query_ids], 0.0)
            g = g.at[:, 1:, :].add(tangential)
            out_gradient = out_gradient.at[batch.target_ids].set(g)
    return out_value, out_gradient


def point_values(payload: PointRowPayload, owner_fields, boundary: BoundaryArrays | None = None):
    return apply_point_rows(payload, owner_fields, boundary, gradients=False)[0]


def point_gradients(payload: PointRowPayload, owner_fields, boundary: BoundaryArrays | None = None):
    return apply_point_rows(payload, owner_fields, boundary, values=False)[1]
