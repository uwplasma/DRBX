"""Focused real-HSX and algebraic checks for Phase A point-row extraction."""
from __future__ import annotations

import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry.fci_perpendicular_reconstruction import PointRowContext, StructuredReconstruction
from drbx.native.fci_perpendicular_point_rows import (BoundaryArrays, apply_point_rows,
    load_point_row_plan, lower_point_rows, point_gradients, point_values)


DATA = Path(__file__).parent / "data/p_shared_point_rows"


def _real_fixture(n):
    with np.load(DATA/f"N{n}.plan.npz", allow_pickle=False) as source:
        identity = json.loads(str(source["metadata_json"].item()))["identity"]
    plan = load_point_row_plan(DATA/f"N{n}.plan.npz", identity)
    with np.load(DATA/f"N{n}.heldout.npz", allow_pickle=False) as source:
        donor = np.asarray(source["donor_ids"])
        fields = np.zeros((int(donor.max())+1, source["donor_values"].shape[1]))
        fields[donor] = source["donor_values"]
        boundary = BoundaryArrays(np.asarray(source["boundary_values"]),
                                  np.asarray(source["boundary_tangential"]))
        expected = (np.asarray(source["expected_values"]),
                    np.asarray(source["expected_gradients"]))
    return plan, fields, boundary, expected, identity


@pytest.mark.parametrize("n", (32, 48, 64))
def test_real_hsx_point_rows_eager_jit_and_boundary(n):
    plan, fields, boundary, expected, identity = _real_fixture(n)
    assert len(plan.row_metadata) == 34
    assert len(plan.boundary_points) > 0
    assert {item["family"] for item in plan.row_metadata} == {
        "coupled_quartic", "ringwise", "singleton", "boundary_transverse", "quartic_wall"}
    assert len(plan.bucket_summary) >= 5
    assert all(width <= 160 for _, _, width, _ in plan.bucket_summary)
    assert len(plan.donor_eta_offset_sets) == len(plan.target_points)
    value, gradient = apply_point_rows(plan.payload, fields, boundary)
    np.testing.assert_allclose(value, expected[0], atol=2e-11, rtol=0)
    np.testing.assert_allclose(gradient, expected[1], atol=2e-9, rtol=0)
    compiled = jax.jit(lambda f, b: apply_point_rows(plan.payload, f, b))
    value_jit, gradient_jit = compiled(fields, boundary)
    np.testing.assert_allclose(value_jit, expected[0], atol=2e-11, rtol=0)
    np.testing.assert_allclose(gradient_jit, expected[1], atol=2e-9, rtol=0)
    np.testing.assert_allclose(point_values(plan.payload, fields, boundary), expected[0], atol=2e-11, rtol=0)
    np.testing.assert_allclose(point_gradients(plan.payload, fields, boundary), expected[1], atol=2e-9, rtol=0)
    with pytest.raises(ValueError, match="boundary arrays"):
        apply_point_rows(plan.payload, fields)
    with pytest.raises(ValueError, match="incompatible shapes"):
        apply_point_rows(plan.payload, fields, BoundaryArrays(boundary.values[:-1], boundary.tangential_gradients))
    with pytest.raises(ValueError, match="identity differs"):
        load_point_row_plan(DATA/f"N{n}.plan.npz", {**identity, "resolution": -1})


def test_real_hsx_jvp_matches_linear_action_and_finite_difference():
    plan, fields, boundary, _, _ = _real_fixture(32)
    direction = np.zeros_like(fields)
    ids = np.unique(np.concatenate([batch.donor_ids.ravel() for batch in plan.payload.batches]))
    direction[ids] = 0.01*np.sin(ids[:, None]+np.arange(fields.shape[1])[None, :])
    tangent = BoundaryArrays(0.02*np.cos(np.arange(boundary.values.size).reshape(boundary.values.shape)),
                             0.03*np.sin(np.arange(boundary.tangential_gradients.size).reshape(boundary.tangential_gradients.shape)))

    def action(f, bv, bt):
        value, gradient = apply_point_rows(plan.payload, f, BoundaryArrays(bv, bt))
        return jnp.concatenate((value.ravel(), gradient.ravel()))

    primals = (jnp.asarray(fields), jnp.asarray(boundary.values), jnp.asarray(boundary.tangential_gradients))
    tangents = (jnp.asarray(direction), jnp.asarray(tangent.values), jnp.asarray(tangent.tangential_gradients))
    _, jvp = jax.jvp(action, primals, tangents)
    exact = action(*tangents)
    np.testing.assert_allclose(jvp, exact, atol=3e-12, rtol=0)
    step = 1e-4
    finite = (action(*(x+step*d for x, d in zip(primals, tangents, strict=True)))-action(*primals))/step
    np.testing.assert_allclose(jvp, finite, atol=2e-8, rtol=2e-8)


def _toy_context(n=8):
    faces = (np.linspace(0, 1, n+1), np.linspace(0, 2*np.pi, n+1), np.linspace(0, 2*np.pi, n+1))
    centers = tuple((x[:-1]+x[1:])/2 for x in faces)
    ijk = np.array(np.unravel_index(np.arange(n**3), (n, n, n))).T
    points = np.column_stack([centers[a][ijk[:, a]] for a in range(3)])
    centroids = np.column_stack((points[:, 0]*np.cos(points[:, 1]), points[:, 0]*np.sin(points[:, 1])))
    return PointRowContext.from_arrays(faces=faces, centers=centers,
        raw_to_owner=np.arange(n**3), raw_volume=np.ones(n**3),
        owner_volume=np.ones(n**3), owner_centroid_xy=centroids,
        eta_period=2*np.pi, dr=1/n, dtheta=2*np.pi/n, deta=2*np.pi/n)


def test_constant_linear_and_periodic_functionals():
    context = _toy_context()
    builder = StructuredReconstruction(context)
    point = context.pts[np.ravel_multi_index((4, 0, 7), (8, 8, 8))]
    shifted = point.copy(); shifted[1] += 2*np.pi; shifted[2] -= 2*np.pi
    row = builder.rows((4, 0, 7), np.stack((point, shifted)), location="cell")
    plan = lower_point_rows(context, [row], boundary_kind="none", boundary_kinds=("none", "none"))
    fields = np.column_stack((np.ones(8**3), context.pts[:, 0]))
    value, gradient = apply_point_rows(plan.payload, fields)
    np.testing.assert_allclose(value, np.array([[1, point[0]], [1, point[0]]]), atol=2e-13, rtol=0)
    np.testing.assert_allclose(gradient[:, :, 0], 0, atol=2e-12, rtol=0)
    np.testing.assert_allclose(gradient[:, :, 1], [[1, 0, 0], [1, 0, 0]], atol=2e-12, rtol=0)
    assert len(plan.boundary_points) == 0


def test_dirichlet_lift_uses_tangential_data_and_recovers_normal_slope():
    context = _toy_context()
    builder = StructuredReconstruction(context)
    point = context.pts[np.ravel_multi_index((6, 1, 0), (8, 8, 8))][None, :]
    row = builder.rows((6, 1, 0), point, location="cell")
    assert row.boundary_conditioned
    plan = lower_point_rows(context, [row], boundary_kind="dirichlet", boundary_kinds=("dirichlet",))
    q = context.pts
    fields = (1+0.3*np.sin(q[:, 1])+0.2*np.cos(q[:, 2])+0.41*(q[:, 0]-1))[:, None]
    boundary_points = plan.boundary_points
    prescribed = (1+0.3*np.sin(boundary_points[:, 1])+0.2*np.cos(boundary_points[:, 2]))[:, None]
    tangent = np.stack((0.3*np.cos(boundary_points[:, 1]), -0.2*np.sin(boundary_points[:, 2])), axis=1)[:, :, None]
    value, gradient = apply_point_rows(plan.payload, fields, BoundaryArrays(prescribed, tangent))
    target = point[0]
    np.testing.assert_allclose(value[0, 0], 1+0.3*np.sin(target[1])+0.2*np.cos(target[2])+0.41*(target[0]-1), atol=3e-13, rtol=0)
    np.testing.assert_allclose(gradient[0, :, 0], [0.41, 0.3*np.cos(target[1]), -0.2*np.sin(target[2])], atol=3e-12, rtol=0)
    with pytest.raises(ValueError, match="requires a fixed Dirichlet"):
        lower_point_rows(context, [row], boundary_kind="none")
    with pytest.raises(ValueError, match="mixed boundary kinds"):
        lower_point_rows(context, [row], boundary_kind="dirichlet", boundary_kinds=("dirichlet", "none"))


def test_complete_owner_observation_uses_stored_raw_volumes():
    base = _toy_context()
    n = base.n
    raw_owner = np.arange(n**3)
    raw_owner[1:] -= 1
    raw_volume = np.ones(n**3); raw_volume[1] = 3
    owner_volume = np.bincount(raw_owner, weights=raw_volume)
    centroid = np.zeros((len(owner_volume), 2))
    for axis in range(2):
        centroid[:, axis] = np.bincount(raw_owner, weights=raw_volume*base.xy[:, axis])/owner_volume
    context = PointRowContext.from_arrays(faces=base.faces, centers=base.centers,
        raw_to_owner=raw_owner, raw_volume=raw_volume, owner_volume=owner_volume,
        owner_centroid_xy=centroid, eta_period=base.g.eta_period,
        dr=base.g.dr, dtheta=base.g.dtheta, deta=base.g.deta)
    np.testing.assert_array_equal(context.members(0), [0, 1])
    np.testing.assert_allclose(context.observe_owners(np.arange(n**3, dtype=float), [0]), [0.75], rtol=0, atol=0)
