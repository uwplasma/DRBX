"""Source-major, array-only JAX application of host-prepared point rows.

A *source* is one packed ``PointRows`` object: its quadrature nodes ("targets")
share one donor list. ``SourceRowBatch`` therefore stores one donor list per
source and dense ``(sources, nodes, donors)`` coefficient blocks, so the owner
field is gathered once per source instead of once per target. Value-only
buckets carry no gradient arrays and unconditioned buckets carry no boundary
query arrays (those fields are ``None``, an empty pytree node).

The conditioned-row semantics are exactly those of
``drbx.native.fci_perpendicular_point_rows.apply_point_rows`` (which stays the
equivalence reference): subtract the prescribed g at every donor query, add g at
the target, and add the two tangential g derivatives to gradient components 1
and 2. Nothing but arrays enters the kernel. The payload is registered as a
pytree with static counts, so it may be passed to ``jax.jit`` as an argument
(no large baked-in constants) or closed over. Tensor-encoded sources (task D2) ride in the
same payload as ``tensor_batches`` plus grid-global ``tensor_tables``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.fci_perpendicular_point_rows import BoundaryArrays
from drbx.native.fci_perpendicular_tensor_rows import (
    TensorRowBatch, TensorTables, apply_tensor_batch, prepare_fields)


class SourceRowBatch(NamedTuple):
    """Sources of one (family, conditioned, has_gradient, nodes, donor width) bucket."""

    value_slots: np.ndarray                 # (S, q) int32, output row of each node's value
    donor_ids: np.ndarray                   # (S, w) int32, zero padded
    value: np.ndarray                       # (S, q, w) float64, zero padded
    gradient_slots: np.ndarray | None       # (S, q) int32, gradient buckets only
    gradient: np.ndarray | None             # (S, q, 3, w) float64, gradient buckets only
    donor_query: np.ndarray | None          # (S, w) int32, conditioned buckets only
    target_query: np.ndarray | None         # (S, q) int32, conditioned buckets only


@dataclass(frozen=True)
class SourceRowPayload:
    """Bucketed source rows plus the (static) output and boundary-table sizes.

    ``batches`` are the CSR-encoded sources (dense per-source coefficient blocks).
    ``tensor_batches`` (with their grid-global ``tensor_tables``) are the tensor-encoded,
    unconditioned sources, applied without materializing weights
    (``drbx.native.fci_perpendicular_tensor_rows``). Both write disjoint rows of the same
    output arrays, so a mixed plan is one ``apply_source_rows`` call.
    """

    batches: tuple[SourceRowBatch, ...]
    n_targets: int              # rows of the value output
    n_gradient_targets: int     # rows of the gradient output
    n_boundary_queries: int     # rows the prescribed boundary arrays must have
    tensor_batches: tuple[TensorRowBatch, ...] = ()
    tensor_tables: TensorTables | None = None


jax.tree_util.register_dataclass(
    SourceRowPayload, data_fields=["batches", "tensor_batches", "tensor_tables"],
    meta_fields=["n_targets", "n_gradient_targets", "n_boundary_queries"])


def payload_nbytes(payload: SourceRowPayload) -> int:
    """Total bytes of the array leaves of a payload."""
    return sum(int(np.asarray(a).nbytes) for a in jax.tree_util.tree_leaves(payload))


def _validate_inputs(payload: SourceRowPayload, fields, boundary):
    if fields.ndim != 2:
        raise ValueError("owner_fields must have shape (owners, fields)")
    if payload.tensor_batches and payload.tensor_tables is None:
        raise ValueError("tensor batches require their tensor tables")
    if payload.n_boundary_queries:
        if boundary is None:
            raise ValueError("conditioned point rows require explicit boundary arrays")
        expected = (payload.n_boundary_queries, fields.shape[1])
        if (boundary.values.shape != expected
                or boundary.tangential_gradients.shape != (expected[0], 2, expected[1])):
            raise ValueError("boundary values/tangential gradients have incompatible shapes")


def apply_source_rows(payload: SourceRowPayload, owner_fields,
                      boundary: BoundaryArrays | None = None,
                      *, values: bool = True, gradients: bool = True):
    """Apply source rows to batched owner fields.

    The body is a single jitted computation (``static_argnames`` values/gradients),
    so an eager call and a call inside an outer ``jax.jit`` run the same compiled
    contraction and agree bitwise; op-by-op eager execution would not (XLA picks a
    different summation order for a standalone dot than for one fused with its
    gathers).

    Returns ``(values (n_targets, F), gradients (n_gradient_targets, 3, F))``;
    an entry is ``None`` when not requested. Gradient rows are indexed by the
    plan's ``gradient_slot`` (only targets that store a gradient have one).
    Donors are gathered once per source; no exact normal BC derivative is used.
    Tensor batches (``payload.tensor_batches``: unconditioned singleton / ringwise /
    centered_radial sources kept as 1-D factors) are contracted directly with the gathered
    donor blocks by ``drbx.native.fci_perpendicular_tensor_rows.apply_tensor_batch``
    (no dense weights, no boundary lift) into their own rows of the same outputs.
    """
    if not values and not gradients:
        raise ValueError("request values, gradients, or both")
    fields = jnp.asarray(owner_fields)
    _validate_inputs(payload, fields, boundary)
    return _apply_source_rows(payload, fields, boundary, values=values, gradients=gradients)


@partial(jax.jit, static_argnames=("values", "gradients"))
def _apply_source_rows(payload, fields, boundary, *, values, gradients):
    nf = fields.shape[1]
    if boundary is None:
        bc_values = bc_tangent = None
    else:
        bc_values = jnp.asarray(boundary.values)
        bc_tangent = jnp.asarray(boundary.tangential_gradients)
    out_value = jnp.zeros((payload.n_targets, nf), dtype=fields.dtype) if values else None
    out_gradient = (jnp.zeros((payload.n_gradient_targets, 3, nf), dtype=fields.dtype)
                    if gradients else None)
    for batch in payload.batches:
        has_gradient = batch.gradient is not None
        if not values and not (gradients and has_gradient):
            continue
        conditioned = batch.donor_query is not None
        data = fields[batch.donor_ids]                       # (S, w, F), once per source
        if conditioned:
            data = data - bc_values[batch.donor_query]
        if values:
            v = jnp.einsum("sqd,sdf->sqf", batch.value, data)
            if conditioned:
                v = v + bc_values[batch.target_query]
            out_value = out_value.at[batch.value_slots.reshape(-1)].set(
                v.reshape(-1, nf), unique_indices=True)
        if gradients and has_gradient:
            g = jnp.einsum("sqad,sdf->sqaf", batch.gradient, data)
            if conditioned:
                g = g.at[:, :, 1:, :].add(bc_tangent[batch.target_query])
            out_gradient = out_gradient.at[batch.gradient_slots.reshape(-1)].set(
                g.reshape(-1, 3, nf), unique_indices=True)
    context = (prepare_fields(payload.tensor_tables, payload.tensor_batches, fields,
                              values=values, gradients=gradients)
               if payload.tensor_batches else None)
    for batch in payload.tensor_batches:
        value, gradient = apply_tensor_batch(payload.tensor_tables, batch, fields,
                                             values=values, gradients=gradients, context=context)
        if value is not None:
            out_value = out_value.at[batch.value_slots].set(value, unique_indices=True)
        if gradient is not None:
            out_gradient = out_gradient.at[batch.gradient_slots].set(gradient, unique_indices=True)
    return out_value, out_gradient


def source_values(payload: SourceRowPayload, owner_fields, boundary: BoundaryArrays | None = None):
    return apply_source_rows(payload, owner_fields, boundary, gradients=False)[0]


def source_gradients(payload: SourceRowPayload, owner_fields, boundary: BoundaryArrays | None = None):
    return apply_source_rows(payload, owner_fields, boundary, values=False)[1]
