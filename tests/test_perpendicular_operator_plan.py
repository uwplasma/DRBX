"""Operator plan, reconstruction state and P07 operator on synthetic rows (P08 step 2b, E3; fast).

A bounded owner closure of the synthetic ``tests.perpendicular_synthetic`` world (real census, random shaped
rows incl. conditioned rows, Neumann rows, a missing upper side at the wall and the collapsed r=0 face) is
lowered twice (in-memory rows, and through an on-disk artifact of several chunks per group) and checked
against the host applications of ``scripts/p_shared`` (``apply.py`` and the replay's side helpers) on the same
row objects; eager == jit bitwise; the P07 action is linear in the fields for fixed boundary data.
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

from drbx.native.fci_perpendicular_p07_operator import p07_action, p07_face_flux
from drbx.native.fci_perpendicular_reconstruction_state import (
    boundary_data_from_callables, cell_state, cell_state_pair, face_state, face_state_pair,
    zero_boundary_data)
from drbx.stencils import artifact as art
from drbx.stencils.geometry_arrays import _ARRAY_FIELDS as ARRAY_FIELDS, GeometryArrays
from drbx.stencils.loader import _unique_rows
from drbx.stencils.operator_plan import lower_perpendicular_plan_from_artifact, pack_owner_rows
from drbx.geometry.fci_perpendicular_integrated_rows import contract_face_tensor
from tests.perpendicular_synthetic import NF, Boundary, host_cells, host_faces, lower_world, make_world

OWNERS = [0, 1, 2, 40, 96, 99, 100, 114]
KINDS = ("dirichlet", "neumann", "dirichlet")
TOL = 1e-13
RAW_FIELDS = tuple(f for f in ARRAY_FIELDS if f.endswith(("_raw_h", "raw_points", "raw_jacobian", "raw_J", "raw_B", "raw_K", "raw_weight", "raw_tensor", "raw_divergence")))


@pytest.fixture(scope="module")
def world():
    return make_world(owners=OWNERS)


@pytest.fixture(scope="module")
def plan(world):
    return lower_world(world)


@pytest.fixture(scope="module")
def boundary():
    return Boundary()


@pytest.fixture(scope="module")
def fields(world):
    return np.random.default_rng(5).normal(size=(world.n_owners, NF))


def _bc(plan, boundary):
    return boundary_data_from_callables(plan, boundary.dirichlet, boundary.normal)


def _close(a, b, tol=TOL):
    a, b = np.asarray(a), np.asarray(b)
    scale = max(float(np.max(np.abs(b))), 1e-300)
    assert np.max(np.abs(a - b)) <= tol * scale, (np.max(np.abs(a - b)), scale)


# --------------------------------------------------------------------------
# Plan structure
# --------------------------------------------------------------------------

def test_plan_boundary_tables_and_index_maps(world, plan):
    for points in (plan.dirichlet_points, plan.neumann_points):
        assert len(_unique_rows(points)[0]) == len(points)          # deduplicated by bit pattern
    qd, qn = len(plan.dirichlet_points), len(plan.neumann_points)
    assert plan.cells.rows.n_boundary_queries == plan.faces.rows.n_boundary_queries == qd
    assert plan.p07.rows.boundary_query_count == qd
    for batch in list(plan.cells.rows.batches) + list(plan.faces.rows.batches):
        if batch.donor_query is not None:
            assert batch.donor_query.max() < qd and batch.target_query.max() < qd
    for rows in (plan.cells.neumann, plan.faces.common_neumann, plan.faces.side_neumann, plan.p07.neumann):
        assert rows.boundary_ids.max() < qn
    # every conditioned row's points are in the table: dirichlet_points holds the cells' donor queries
    conditioned = [world.row_index[("R1", int(r))] for r in plan.cells.raw_ids
                   if world.row_index[("R1", int(r))].boundary_conditioned]
    table = {tuple(p) for p in plan.dirichlet_points.tolist()}
    for row in conditioned:
        assert all(tuple(p) in table for p in row.trace_donor_points.tolist() + row.trace_target_points.tolist())
    # cells
    c = plan.cells
    np.testing.assert_array_equal(c.raw_ids, world.raw_ids)
    np.testing.assert_array_equal(c.raw_owner, world.grid.raw_to_owner[world.raw_ids])
    w = world.geometry.p06_raw_weight * world.geometry.p06_raw_J / np.maximum(world.geometry.p06_raw_B, 1e-30)
    np.testing.assert_array_equal(c.evolution_weight, w)
    ev = np.zeros(world.n_owners)
    np.add.at(ev, c.raw_owner, w)
    np.testing.assert_array_equal(c.evolution_volume, ev)
    assert int(c.neumann_cell.size) == int(c.conditioned.sum())
    # faces
    f = plan.faces
    census = world.census
    assert np.array_equal(f.census_row, world.face_rows)
    assert not f.collapsed.any() and np.all(f.face_multiplier == 1.0)
    np.testing.assert_array_equal(f.wall, (census.axis[world.face_rows] == 0) & (census.i[world.face_rows] == world.n))
    np.testing.assert_array_equal(f.wall_faces, np.flatnonzero(f.wall))
    assert f.wall_faces.dtype == np.int32 and f.wall.any()
    np.testing.assert_array_equal(f.upper_present, census.raw_hi[world.face_rows] >= 0)
    np.testing.assert_array_equal(f.p07_valid, census.p07_id[world.face_rows] >= 0)
    assert f.has_missing_side and not f.upper_present.all() and f.lower_present.all()
    assert np.all(f.fallback_query[f.upper_present] == 0)
    # p07
    p = plan.p07
    assert set(p.p07_id.tolist()) == set(census.p07_id[world.p07_rows].tolist())
    assert (p.family == 0).any() and (p.face_index[p.family == 0] == -1).all()
    assert np.array_equal(p.family[p.neumann_face], census.family[p.census_row[p.neumann_face]])
    assert set(np.unique(p.family[p.conditioned]).tolist()) <= {1, 2, 4}
    pos = np.searchsorted(world.face_rows, p.census_row[p.neumann_face])
    expected = contract_face_tensor(world.geometry.p06_face_weight[pos], world.geometry.p07_face_tensor[pos],
                                    census.axis[p.census_row[p.neumann_face]])
    np.testing.assert_array_equal(p.integrand, expected)


def test_plan_is_a_pytree_with_static_counts(plan):
    leaves, tree = jax.tree_util.tree_flatten(plan)
    again = jax.tree_util.tree_unflatten(tree, leaves)
    assert again.n == plan.n and again.p07.rows.face_count == plan.p07.rows.face_count
    assert all(hasattr(leaf, "shape") for leaf in leaves)


# --------------------------------------------------------------------------
# State vs the host applications
# --------------------------------------------------------------------------

def test_cell_state_matches_host(world, plan, boundary, fields):
    bc = _bc(plan, boundary)
    (vd, gd), (vn, gn) = host_cells(world, fields, boundary)
    for kinds in (("dirichlet",) * NF, ("neumann",) * NF, KINDS):
        state = cell_state(plan, fields, bc, kinds)
        is_n = np.array([k == "neumann" for k in kinds])
        _close(state.value, np.where(is_n, vn, vd))
        _close(state.gradient, np.where(is_n, gn, gd))
    d, n = cell_state_pair(plan, fields, bc)
    _close(d.value, vd); _close(d.gradient, gd); _close(n.value, vn); _close(n.gradient, gn)
    assert np.max(np.abs(vn - vd)) > 1e-3          # the Neumann rows really differ from the lift
    only_grad = cell_state(plan, fields, bc, KINDS, values=False)
    assert only_grad.value is None
    _close(only_grad.gradient, np.where(np.array([k == "neumann" for k in KINDS]), gn, gd))


def test_face_state_matches_host(world, plan, boundary, fields):
    bc = _bc(plan, boundary)
    h = host_faces(world, fields, boundary)
    is_n = np.array([k == "neumann" for k in KINDS])
    for kinds in (("dirichlet",) * NF, ("neumann",) * NF, KINDS):
        m = np.array([k == "neumann" for k in kinds])
        s = face_state(plan, fields, bc, kinds)
        _close(s.value, np.where(m, h["cvn"], h["cvd"]))
        _close(s.gradient, np.where(m, h["cgn"], h["cgd"]))
        _close(s.lower, np.where(m, h["ln"], h["ld"]))
        _close(s.upper, np.where(m, h["un"], h["ud"]))
    d, n = face_state_pair(plan, fields, bc)
    _close(d.value, h["cvd"]); _close(d.gradient, h["cgd"]); _close(d.lower, h["ld"]); _close(d.upper, h["ud"])
    _close(n.value, h["cvn"]); _close(n.gradient, h["cgn"]); _close(n.lower, h["ln"]); _close(n.upper, h["un"])
    # the missing-side rules: Dirichlet takes the trace, Neumann copies the other side
    miss = ~plan.faces.upper_present
    assert miss.any()
    np.testing.assert_array_equal(np.asarray(n.upper)[miss], np.asarray(n.lower)[miss])
    assert np.max(np.abs(np.asarray(d.upper)[miss] - np.asarray(n.upper)[miss])) > 1e-3
    assert np.max(np.abs(h["cvn"] - h["cvd"])) > 1e-3 and np.max(np.abs(h["ln"] - h["ld"])) > 1e-3
    assert face_state(plan, fields, bc, KINDS, gradients=False).gradient is None
    del is_n


def test_dirichlet_fields_need_no_neumann_data(world, plan, boundary, fields):
    bc = boundary_data_from_callables(plan, boundary.dirichlet, None)
    assert bc.neumann_normal is None
    (vd, _), _ = host_cells(world, fields, boundary)
    _close(cell_state(plan, fields, bc, "dirichlet").value, vd)
    with pytest.raises(ValueError, match="neumann_normal"):
        cell_state(plan, fields, bc, "neumann")


# --------------------------------------------------------------------------
# P07
# --------------------------------------------------------------------------

def _host_p07(world, plan, ov, boundary):
    from p_shared import apply as A
    p = plan.p07
    Fp = len(p.p07_id)
    flux_d, flux_n = np.zeros((Fp, NF)), np.zeros((Fp, NF))
    for f, (pid, cr) in enumerate(zip(p.p07_id, p.census_row)):
        row = world.row_index[int(pid)]
        flux_d[f] = A.apply_integrated_row(row, ov, boundary.dirichlet)
        flux_n[f] = flux_d[f]
        if row.boundary_conditioned:
            k = int(np.flatnonzero(p.neumann_face == f)[0])
            nrows = [world.neumann_index[("R4", int(pid), q)] for q in range(9)]
            data = [boundary.normal(nr.boundary_points) for nr in nrows]
            flux_n[f] = A.p07_neumann_face_flux(nrows, ov, data, np.asarray(p.integrand)[k])
    owner = A.p07_scatter_flux  # (flux, lower, upper, volume)
    return flux_d, flux_n, owner


def test_p07_matches_host_for_every_kind_mix(world, plan, boundary, fields):
    bc = _bc(plan, boundary)
    flux_d, flux_n, scatter = _host_p07(world, plan, fields, boundary)
    lo, up, vol = np.asarray(plan.p07.lower_owner), np.asarray(plan.p07.upper_owner), world.owner_volume
    for kinds in (("dirichlet",) * NF, ("neumann",) * NF, KINDS):
        m = np.array([k == "neumann" for k in kinds])
        expected_flux = np.where(m, flux_n, flux_d)
        expected_flux[plan.p07.family == 0] = 0.0
        _close(p07_face_flux(plan, fields, bc, kinds), expected_flux)
        _close(p07_action(plan, fields, bc, kinds), scatter(expected_flux, lo, up, vol))
    assert np.max(np.abs(flux_n - flux_d)) > 1e-3
    np.testing.assert_array_equal(np.asarray(p07_face_flux(plan, fields, bc, "neumann"))[plan.p07.family == 0], 0.0)


def test_p07_only_plan_and_dirichlet_only_data(world, boundary, fields):
    plan = lower_world(world, include=("p07",))
    assert plan.cells is None and plan.faces is None
    bc = boundary_data_from_callables(plan, boundary.dirichlet, None)
    full = lower_world(world)
    _close(p07_action(plan, fields, bc, "dirichlet"),
           p07_action(full, fields, _bc(full, boundary), "dirichlet"))


# --------------------------------------------------------------------------
# Eager == jit, linearity
# --------------------------------------------------------------------------

def test_eager_equals_jit_bitwise(plan, boundary, fields):
    bc = _bc(plan, boundary)
    for kinds in (("dirichlet",) * NF, KINDS):
        for eager, function in (
                (cell_state(plan, fields, bc, kinds), lambda p, f, b: cell_state(p, f, b, kinds)),
                (face_state(plan, fields, bc, kinds), lambda p, f, b: face_state(p, f, b, kinds)),
                (p07_action(plan, fields, bc, kinds), lambda p, f, b: p07_action(p, f, b, kinds))):
            jitted = jax.jit(function)(plan, fields, bc)
            for a, b in zip(jax.tree_util.tree_leaves(eager), jax.tree_util.tree_leaves(jitted)):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_p07_jvp_is_the_linear_part(plan, boundary, fields):
    bc = _bc(plan, boundary)
    tangent = np.random.default_rng(9).normal(size=fields.shape)
    for kinds in (KINDS,):
        primal, jvp = jax.jvp(lambda f: p07_action(plan, f, bc, kinds), (jnp.asarray(fields),), (jnp.asarray(tangent),))
        _close(primal, p07_action(plan, fields, bc, kinds))
        _close(jvp, p07_action(plan, tangent, zero_boundary_data(plan, NF), kinds))
        # affine: action(f) = A f + boundary part, so action(f + t) - action(f) = A t
        _close(np.asarray(p07_action(plan, fields + tangent, bc, kinds)) - np.asarray(primal), jvp, tol=1e-12)


# --------------------------------------------------------------------------
# The artifact path
# --------------------------------------------------------------------------

def test_artifact_path_matches_in_memory_plan(world, plan, boundary, fields, tmp_path):
    def half(items):
        return items[:len(items) // 2], items[len(items) // 2:]
    p07_ids = world.census.p07_id[world.p07_rows]
    chunks = {"cells": [], "faces": [], "p07": [], "neumann": []}
    for raw, face, p07 in zip(half(world.raw_ids), half(world.face_rows), half(p07_ids)):
        packed = pack_owner_rows(world.row_index, world.neumann_index, raw_ids=raw, face_rows=face, p07_ids=p07)
        for key in chunks:
            chunks[key].extend(getattr(packed, key))
    root = tmp_path / "artifact"
    art.save_row_artifact(root, world.n, identity={"synthetic": True}, **chunks)
    from_disk = lower_perpendicular_plan_from_artifact(
        root, world.n, grid=world.grid, census=world.census, geometry=world.geometry, raw_volume=world.raw_volume,
        owner_volume=world.owner_volume, identity={"synthetic": True}, raw_ids=world.raw_ids,
        face_rows=world.face_rows, p07_rows=world.p07_rows)
    assert len(from_disk.dirichlet_points) == len(plan.dirichlet_points)
    bc_a, bc_b = _bc(plan, boundary), _bc(from_disk, boundary)
    for kinds in (("dirichlet",) * NF, KINDS):
        for x, y in zip(jax.tree_util.tree_leaves(cell_state(plan, fields, bc_a, kinds)),
                        jax.tree_util.tree_leaves(cell_state(from_disk, fields, bc_b, kinds))):
            _close(x, y)
        for x, y in zip(jax.tree_util.tree_leaves(face_state(plan, fields, bc_a, kinds)),
                        jax.tree_util.tree_leaves(face_state(from_disk, fields, bc_b, kinds))):
            _close(x, y)
        _close(p07_action(from_disk, fields, bc_b, kinds), p07_action(plan, fields, bc_a, kinds))
    with pytest.raises(ValueError, match="identity"):
        lower_perpendicular_plan_from_artifact(
            root, world.n, grid=world.grid, census=world.census, geometry=world.geometry,
            raw_volume=world.raw_volume, owner_volume=world.owner_volume, identity={"other": 1},
            raw_ids=world.raw_ids, face_rows=world.face_rows, p07_rows=world.p07_rows)


def test_lowering_rejects_inconsistent_inputs(world):
    from drbx.stencils.operator_plan import lower_perpendicular_plan
    packed = pack_owner_rows(world.row_index, world.neumann_index, raw_ids=world.raw_ids,
                             face_rows=world.face_rows, p07_ids=world.census.p07_id[world.p07_rows])
    common = dict(grid=world.grid, census=world.census, geometry=world.geometry, raw_volume=world.raw_volume,
                  owner_volume=world.owner_volume, cell_chunks=packed.cells, face_chunks=packed.faces,
                  p07_chunks=packed.p07, neumann_chunks=packed.neumann, raw_ids=world.raw_ids,
                  face_rows=world.face_rows, p07_rows=world.p07_rows)
    with pytest.raises(ValueError, match="aligned with raw_ids"):
        lower_perpendicular_plan(**{**common, "raw_ids": world.raw_ids[:-1]})
    with pytest.raises(ValueError, match="one R1 target per raw id"):
        lower_perpendicular_plan(**{**common, "raw_ids": world.raw_ids[1:], "geometry": _drop_first_raw(world)})
    with pytest.raises(ValueError, match="Neumann"):
        lower_perpendicular_plan(**{**common, "neumann_chunks": ()})
    with pytest.raises(ValueError, match="exactly the P07 faces"):
        lower_perpendicular_plan(**{**common, "p07_rows": world.p07_rows[1:]})


def _drop_first_raw(world):
    g = world.geometry
    arrays = {name: getattr(g, name)[1:] if name in RAW_FIELDS else getattr(g, name) for name in ARRAY_FIELDS}
    return GeometryArrays(schema=g.schema, identity=GeometryArrays._compute_identity(arrays), **arrays)
