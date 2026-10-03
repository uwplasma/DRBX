"""P09 per-face union layout of the face (R2/R3) and P07 (R4) Neumann rows against the per-row reference layout.

``lower_perpendicular_plan(neumann_layout="face_union")`` (default) stores one donor union per conditioned face and the
P07 rows pre-contracted with the q3 integrand; ``"rows"`` keeps one padded donor list per row. The two plans apply the
same linear maps, so every state, flux, owner action and sparse export agrees to rounding (the contraction order and the
summation of donors repeated within a face differ). Fast, synthetic world.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native.fci_perpendicular_neumann_rows import (
    NeumannFacePayload, apply_neumann_integrated_face_rows, apply_neumann_point_rows)
from drbx.native.fci_perpendicular_p07_operator import p07_action, p07_face_flux
from drbx.native.fci_perpendicular_p07_sparse import export_p07_sparse
from drbx.native.fci_perpendicular_reconstruction_state import (
    boundary_data_from_callables, cell_state, face_state, face_state_pair)
from drbx.stencils import operator_plan as OP
from drbx.stencils.operator_plan import IntegratedNeumannRows, NeumannFaceRows, NeumannRows
from tests.perpendicular_synthetic import NF, Boundary, lower_world, make_world

OWNERS = [0, 1, 2, 40, 96, 99, 100, 114]
KINDS = ("neumann", "dirichlet", "neumann")
TOL = 1e-12


@pytest.fixture(scope="module")
def world():
    return make_world(owners=OWNERS)


@pytest.fixture(scope="module")
def plan(world):
    return lower_world(world)


@pytest.fixture(scope="module")
def rows_plan(world):
    return lower_world(world, neumann_layout="rows")


@pytest.fixture(scope="module")
def bc(plan):
    boundary = Boundary()
    return boundary_data_from_callables(plan, boundary.dirichlet, boundary.normal)


@pytest.fixture(scope="module")
def fields(world):
    return jnp.asarray(np.random.default_rng(5).normal(size=(world.n_owners, NF)))


def _close(a, b, tol=TOL):
    if a is None or b is None:
        assert a is None and b is None
        return
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.max(np.abs(a - b)) <= tol * max(np.max(np.abs(b)), 1e-300)


def test_layouts_and_shared_tables(plan, rows_plan):
    f, g = plan.faces, rows_plan.faces
    assert isinstance(f.common_neumann, NeumannFaceRows) and isinstance(g.common_neumann, NeumannRows)
    assert isinstance(plan.p07.neumann, IntegratedNeumannRows) and isinstance(rows_plan.p07.neumann, NeumannRows)
    assert isinstance(plan.cells.neumann, NeumannRows)                                  # cells stay per row
    np.testing.assert_array_equal(plan.neumann_points, rows_plan.neumann_points)
    np.testing.assert_array_equal(plan.dirichlet_points, rows_plan.dirichlet_points)
    np.testing.assert_array_equal(plan.p07.neumann_face, rows_plan.p07.neumann_face)
    with pytest.raises(ValueError, match="neumann_layout"):
        OP.lower_perpendicular_plan(grid=None, census=None, geometry=None, raw_volume=None, owner_volume=None,
                                    neumann_layout="bogus")


def test_union_rows_are_face_major_complete_and_padded(plan, rows_plan):
    f, qf = plan.faces, plan.faces.nodes
    for union, target, ref in ((f.common_neumann, f.common_neumann_target, rows_plan.faces.common_neumann_target),
                               (f.side_neumann, f.side_neumann_target, rows_plan.faces.side_neumann_target)):
        assert len(target) == len(ref) == union.donor_ids.shape[0] * qf
        np.testing.assert_array_equal(target, np.sort(ref))                              # same rows, face-major
        t = target.reshape(-1, qf)
        assert np.all(t // qf == t[:, :1] // qf) and np.all(t % qf == np.arange(qf))
        assert union.donor_ids.shape[1] % 16 == 0 and union.boundary_ids.shape[1] % 4 == 0
        assert union.donor_ids.dtype == np.int32 and union.boundary_ids.dtype == np.int32
        w = union.value_weights
        assert w.shape == union.donor_ids.shape[:1] + (qf,) + union.donor_ids.shape[1:]
        for face in range(len(t)):                              # the donors with a weight are distinct within a face
            used = union.donor_ids[face][(w[face] != 0).any(axis=0)]
            assert len(np.unique(used)) == len(used)
    p = plan.p07.neumann
    assert p.donor_ids.shape[0] == p.boundary_ids.shape[0] == len(plan.p07.neumann_face)
    assert p.weights.shape == p.donor_ids.shape and p.boundary_weights.shape == p.boundary_ids.shape
    assert p.boundary_ids.max() < len(plan.neumann_points)
    # the union is much narrower than the per-row layout summed over a face's nodes
    assert p.donor_ids.shape[1] < 9 * rows_plan.p07.neumann.donor_ids.shape[1]


def test_union_apply_matches_the_per_row_rows(plan, rows_plan, fields, bc):
    f, g = plan.faces, rows_plan.faces
    for name, tname in (("common_neumann", "common_neumann_target"), ("side_neumann", "side_neumann_target")):
        un, rw = getattr(f, name), getattr(g, name)
        assert isinstance(un.payload(0), NeumannFacePayload)
        uv, ug = apply_neumann_point_rows(un.payload(0), fields, bc.neumann_normal)
        rv, rg = apply_neumann_point_rows(rw.payload(0), fields, bc.neumann_normal)
        # the union holds the rows face-major: put the per-row rows into the same order
        order = np.argsort(getattr(g, tname), kind="stable")
        _close(uv, np.asarray(rv)[order])
        _close(ug, None if rg is None else np.asarray(rg)[order])
        assert (ug is None) == (rg is None)
        uv_only, ug_off = apply_neumann_point_rows(un.payload(0), fields, bc.neumann_normal, gradients=False)
        assert ug_off is None
        np.testing.assert_array_equal(np.asarray(uv_only), np.asarray(uv))


def test_face_state_equals_the_per_row_layout(plan, rows_plan, fields, bc):
    du, nu = face_state_pair(plan, fields, bc)
    dr, nr = face_state_pair(rows_plan, fields, bc)
    for a, b in zip(jax.tree_util.tree_leaves((du, nu)), jax.tree_util.tree_leaves((dr, nr))):
        _close(a, b)
    for kinds in (KINDS, "neumann", "dirichlet"):
        for kw in (dict(), dict(gradients=False), dict(value_columns=(0, 1, 2), gradient_columns=(0, 2)),
                   dict(value_columns=(2, 1), gradient_columns=(2,))):
            su = face_state(plan, fields, bc, kinds, **kw)
            sr = face_state(rows_plan, fields, bc, kinds, **kw)
            for a, b in zip(jax.tree_util.tree_leaves(su), jax.tree_util.tree_leaves(sr)):
                _close(a, b)
    cu, cr = cell_state(plan, fields, bc, KINDS), cell_state(rows_plan, fields, bc, KINDS)
    for a, b in zip(jax.tree_util.tree_leaves(cu), jax.tree_util.tree_leaves(cr)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))                      # cells: same rows, bitwise


@pytest.mark.parametrize("kinds", (KINDS, "neumann", "dirichlet"))
def test_p07_equals_the_per_row_layout(plan, rows_plan, fields, bc, kinds):
    for columns in (None, 3):
        _close(p07_face_flux(plan, fields, bc, kinds, columns=columns),
               p07_face_flux(rows_plan, fields, bc, kinds, columns=columns))
        _close(p07_action(plan, fields, bc, kinds, columns=columns), p07_action(rows_plan, fields, bc, kinds, columns=columns))
    # the contracted payload gives the restoration directly
    restored = apply_neumann_integrated_face_rows(plan.p07.neumann.payload(0), fields, bc.neumann_normal)
    reference = apply_neumann_integrated_face_rows(rows_plan.p07.neumann.payload(0), fields, bc.neumann_normal,
                                                   rows_plan.p07.integrand)
    _close(restored, reference)


@pytest.mark.parametrize("kind", ("dirichlet", "neumann"))
def test_p07_sparse_export_equals_the_per_row_layout(plan, rows_plan, kind):
    a, b = export_p07_sparse(plan, kind), export_p07_sparse(rows_plan, kind)
    for name in ("matrix", "dirichlet_value", "dirichlet_tangential", "neumann_normal"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.shape == y.shape
        scale = max(abs(y).max() if y.nnz else 0.0, 1e-300)
        assert (abs(x - y).max() if (x - y).nnz else 0.0) <= TOL * scale, name
    assert a.neumann_normal.nnz > 0 or kind == "dirichlet"


def test_face_union_helper_matches_a_dense_reference():
    rng = np.random.default_rng(3)
    fn, q, w, c = 5, 4, 12, 3
    ids = rng.integers(0, 9, size=(fn, q, w)).astype(np.int32)                         # repeated donors within a face and row
    value = rng.normal(size=(fn, q, w)) * (rng.random((fn, q, w)) < 0.7)
    grad = rng.normal(size=(fn, q, c, w)) * (rng.random((fn, q, c, w)) < 0.5)
    union, (uv, ug, none) = OP._face_union(ids, [value, grad, None], 16)
    assert none is None and union.shape == (fn, 16) and uv.shape == (fn, q, 16) and ug.shape == (fn, q, c, 16)
    x = rng.normal(size=9)
    for f in range(fn):
        for k in range(q):
            np.testing.assert_allclose(uv[f, k] @ x[union[f]], value[f, k] @ x[ids[f, k]], rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(ug[f, k] @ x[union[f]], grad[f, k] @ x[ids[f, k]], rtol=1e-13, atol=1e-13)
        used = (value[f] != 0) | (grad[f] != 0).any(axis=1)
        expected = np.unique(ids[f][used])                                              # ascending distinct active donors
        np.testing.assert_array_equal(union[f][:len(expected)], expected)
        assert not np.any(uv[f][:, len(expected):]) and not np.any(ug[f][..., len(expected):])
    # an all-inactive input still has one (padding) block of width
    empty, (we,) = OP._face_union(ids, [np.zeros_like(value)], 4)
    assert empty.shape == (fn, 4) and not np.any(we)


@pytest.mark.parametrize("layout", ("face_union", "rows"))
def test_value_only_common_neumann_rows(world, plan, fields, bc, layout):
    """``common_neumann_gradients=False`` drops the R2 gradient weights: values (and Dirichlet gradients) unchanged, a
    Neumann gradient request raises."""
    slim = lower_world(world, neumann_layout=layout, common_neumann_gradients=False)
    full = lower_world(world, neumann_layout=layout)
    assert slim.faces.common_neumann.gradient_weights is None and slim.faces.common_neumann.value_weights is not None
    assert slim.faces.common_neumann.boundary_gradient_weights is None
    assert full.faces.common_neumann.gradient_weights is not None
    assert OP.plan_nbytes(slim) < OP.plan_nbytes(full)
    kinds = ("neumann", "dirichlet", "neumann")
    kw = dict(value_columns=(0, 1, 2), gradient_columns=(1,))                        # the Dirichlet column's gradient
    for a, b in zip(jax.tree_util.tree_leaves(face_state(slim, fields, bc, kinds, **kw)),
                    jax.tree_util.tree_leaves(face_state(full, fields, bc, kinds, **kw))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    for a, b in zip(jax.tree_util.tree_leaves(face_state(slim, fields, bc, kinds, gradients=False)),
                    jax.tree_util.tree_leaves(face_state(full, fields, bc, kinds, gradients=False))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError, match="common_neumann_gradients=False"):
        face_state(slim, fields, bc, kinds)
    with pytest.raises(ValueError, match="common_neumann_gradients=False"):
        face_state_pair(slim, fields, bc)
    for a, b in zip(jax.tree_util.tree_leaves(p07_action(slim, fields, bc, kinds)),
                    jax.tree_util.tree_leaves(p07_action(full, fields, bc, kinds))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
