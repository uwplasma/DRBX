"""Hand-built payload checks of the source-major kernel (no loader involved)."""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.native.fci_perpendicular_point_rows import BoundaryArrays
from drbx.native.fci_perpendicular_source_rows import (
    SourceRowBatch, SourceRowPayload, apply_source_rows, payload_nbytes, source_gradients,
    source_values)


def _payload(rng):
    """Two buckets: an unconditioned value-only one and a conditioned gradient one."""
    owners, nf, queries = 20, 3, 7
    plain = SourceRowBatch(
        value_slots=np.array([[0, 1], [2, 3]], dtype=np.int32),
        donor_ids=rng.integers(0, owners, size=(2, 16)).astype(np.int32),
        value=rng.normal(size=(2, 2, 16)),
        gradient_slots=None, gradient=None, donor_query=None, target_query=None)
    lifted = SourceRowBatch(
        value_slots=np.array([[4, 5, 6]], dtype=np.int32),
        donor_ids=rng.integers(0, owners, size=(1, 32)).astype(np.int32),
        value=rng.normal(size=(1, 3, 32)),
        gradient_slots=np.array([[0, 1, 2]], dtype=np.int32),
        gradient=rng.normal(size=(1, 3, 3, 32)),
        donor_query=rng.integers(0, queries, size=(1, 32)).astype(np.int32),
        target_query=rng.integers(0, queries, size=(1, 3)).astype(np.int32))
    payload = SourceRowPayload((plain, lifted), n_targets=7, n_gradient_targets=3,
                               n_boundary_queries=queries)
    fields = rng.normal(size=(owners, nf))
    boundary = BoundaryArrays(rng.normal(size=(queries, nf)), rng.normal(size=(queries, 2, nf)))
    return payload, fields, boundary


def _reference(payload, fields, boundary):
    plain, lifted = payload.batches
    values = np.zeros((7, fields.shape[1]))
    gradients = np.zeros((3, 3, fields.shape[1]))
    for s in range(2):
        for q in range(2):
            values[plain.value_slots[s, q]] = plain.value[s, q] @ fields[plain.donor_ids[s]]
    data = fields[lifted.donor_ids[0]] - boundary.values[lifted.donor_query[0]]
    for q in range(3):
        values[lifted.value_slots[0, q]] = (lifted.value[0, q] @ data
                                            + boundary.values[lifted.target_query[0, q]])
        g = np.einsum("ad,df->af", lifted.gradient[0, q], data)
        g[1:] += boundary.tangential_gradients[lifted.target_query[0, q]]
        gradients[lifted.gradient_slots[0, q]] = g
    return values, gradients


def test_kernel_matches_numpy_dirichlet_lift_semantics():
    rng = np.random.default_rng(0)
    payload, fields, boundary = _payload(rng)
    values, gradients = apply_source_rows(payload, fields, boundary)
    ref_values, ref_gradients = _reference(payload, fields, boundary)
    np.testing.assert_allclose(values, ref_values, rtol=0, atol=1e-13)
    np.testing.assert_allclose(gradients, ref_gradients, rtol=0, atol=1e-13)
    np.testing.assert_allclose(source_values(payload, fields, boundary), ref_values, rtol=0, atol=1e-13)
    np.testing.assert_allclose(source_gradients(payload, fields, boundary), ref_gradients, rtol=0, atol=1e-13)
    only_gradients = apply_source_rows(payload, fields, boundary, values=False)
    assert only_gradients[0] is None
    np.testing.assert_allclose(only_gradients[1], ref_gradients, rtol=0, atol=1e-13)
    only_values = apply_source_rows(payload, fields, boundary, gradients=False)
    assert only_values[1] is None
    np.testing.assert_allclose(only_values[0], ref_values, rtol=0, atol=1e-13)
    assert payload_nbytes(payload) == sum(a.nbytes for b in payload.batches for a in b if a is not None)


def test_payload_is_a_pytree_with_static_counts_and_none_fields():
    rng = np.random.default_rng(1)
    payload, fields, boundary = _payload(rng)
    leaves, treedef = jax.tree_util.tree_flatten(payload)
    assert len(leaves) == 3 + 7    # None fields carry no leaves (plain bucket 3, lifted bucket 7)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert (rebuilt.n_targets, rebuilt.n_gradient_targets, rebuilt.n_boundary_queries) == (7, 3, 7)
    assert rebuilt.batches[0].gradient is None and rebuilt.batches[0].donor_query is None
    eager = apply_source_rows(payload, fields, boundary)
    jitted = jax.jit(apply_source_rows, static_argnames=("values", "gradients"))(payload, fields, boundary)
    for a, b in zip(eager, jitted):
        np.testing.assert_array_equal(a, b)


def test_kernel_jvp_is_the_zero_boundary_linear_action_and_input_validation():
    rng = np.random.default_rng(2)
    payload, fields, boundary = _payload(rng)
    tangent = rng.normal(size=fields.shape)
    zero = BoundaryArrays(np.zeros_like(boundary.values), np.zeros_like(boundary.tangential_gradients))
    _, jvp = jax.jvp(lambda f: apply_source_rows(payload, f, boundary), (jnp.asarray(fields),),
                     (jnp.asarray(tangent),))
    linear = apply_source_rows(payload, tangent, zero)
    for a, b in zip(jvp, linear):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-13)
    with pytest.raises(ValueError, match="explicit boundary arrays"):
        apply_source_rows(payload, fields)
    with pytest.raises(ValueError, match="incompatible shapes"):
        apply_source_rows(payload, fields, BoundaryArrays(boundary.values[:, :2], boundary.tangential_gradients))
    with pytest.raises(ValueError, match="values, gradients"):
        apply_source_rows(payload, fields, boundary, values=False, gradients=False)
    with pytest.raises(ValueError, match="owners, fields"):
        apply_source_rows(payload, fields[:, 0], boundary)
    empty = SourceRowPayload((), 0, 0, 0)
    values, gradients = apply_source_rows(empty, fields)
    assert values.shape == (0, 3) and gradients.shape == (0, 3, 3)
