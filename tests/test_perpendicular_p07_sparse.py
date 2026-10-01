"""Sparse export of the P07 owner action vs ``p07_action`` on the synthetic world (fast).

``A u + B_val g_D + B_tan g_T + B_nn g_N`` equals ``p07_action(plan, u, bc, kind)`` to roundoff for both kinds; the
linear part equals the action with zero boundary data; ``boundary_source`` equals the action of zero fields; the
``.npz`` round trip is bitwise and a stale identity is rejected. The plan must contain conditioned P07 faces and
Neumann rows so that every boundary path is exercised.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native.fci_perpendicular_p07_operator import p07_action
from drbx.native.fci_perpendicular_p07_sparse import (
    KINDS, apply_p07_sparse, boundary_source, export_p07_sparse, load_p07_sparse, save_p07_sparse)
from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData, zero_boundary_data
from tests.perpendicular_synthetic import Boundary, lower_world, make_world

OWNERS = [0, 1, 2, 40, 96, 99, 100, 114]
NF = 3
TOL = 1e-12


@pytest.fixture(scope="module")
def plan():
    return lower_world(make_world(owners=OWNERS))


@pytest.fixture(scope="module")
def u(plan):
    return np.random.default_rng(1).normal(size=(len(plan.p07.owner_volume), NF))


@pytest.fixture(scope="module")
def bc(plan):
    rng = np.random.default_rng(2)
    qd, qn = len(plan.dirichlet_points), len(plan.neumann_points)
    return BoundaryData(rng.normal(size=(qd, NF)), rng.normal(size=(qd, 2, NF)), rng.normal(size=(qn, NF)))


def _action(plan, u, bc, kind):
    return np.asarray(p07_action(plan, u, bc, kind))


def test_plan_exercises_boundary_paths(plan):
    assert plan.p07.rows.boundary_query_count > 0
    assert plan.p07.neumann is not None
    assert any(bool(np.any(b.conditioned)) for b in plan.p07.rows.batches)
    assert len(plan.neumann_points) > 0


@pytest.mark.parametrize("kind", KINDS)
def test_matches_action(plan, u, bc, kind):
    op = export_p07_sparse(plan, kind)
    assert op.kind == kind and op.n_owners == len(plan.p07.owner_volume)
    assert op.matrix.shape == (op.n_owners, op.n_owners)
    assert op.dirichlet_value.shape == (op.n_owners, len(plan.dirichlet_points))
    assert op.dirichlet_tangential.shape == (op.n_owners, 2 * len(plan.dirichlet_points))
    assert op.neumann_normal.shape == (op.n_owners, len(plan.neumann_points))
    ref = _action(plan, u, bc, kind)
    mine = apply_p07_sparse(op, u, bc)
    scale = float(np.max(np.abs(ref)))
    diff = float(np.max(np.abs(mine - ref)))
    print(f"{kind}: max|diff|={diff:.3e} scale={scale:.3e} nnz A={op.matrix.nnz} B_val={op.dirichlet_value.nnz} "
          f"B_tan={op.dirichlet_tangential.nnz} B_nn={op.neumann_normal.nnz}")
    assert scale > 0 and diff <= TOL * scale
    assert op.matrix.nnz > 0


@pytest.mark.parametrize("kind", KINDS)
def test_linear_part_and_boundary_source(plan, u, bc, kind):
    op = export_p07_sparse(plan, kind)
    zero = zero_boundary_data(plan, NF)
    lin = _action(plan, u, zero, kind)
    scale = float(np.max(np.abs(lin)))
    assert float(np.max(np.abs(apply_p07_sparse(op, u) - lin))) <= TOL * scale
    src = _action(plan, np.zeros_like(u), bc, kind)
    mine = boundary_source(op, bc)
    assert float(np.max(np.abs(mine - src))) <= TOL * max(float(np.max(np.abs(src))), 1.0)


def test_dirichlet_kind_has_no_normal_data_and_missing_parts(plan, u, bc):
    op = export_p07_sparse(plan, "dirichlet")
    assert op.neumann_normal.nnz == 0
    no_n = BoundaryData(bc.dirichlet_value, bc.dirichlet_tangential, None)
    assert float(np.max(np.abs(boundary_source(op, no_n) - boundary_source(op, bc)))) == 0.0
    with pytest.raises(ValueError):
        boundary_source(op, BoundaryData(None, bc.dirichlet_tangential, bc.neumann_normal))
    nop = export_p07_sparse(plan, "neumann")
    assert nop.neumann_normal.nnz > 0
    with pytest.raises(ValueError):
        boundary_source(nop, no_n)


def test_neumann_lift_blocks_consistent_with_action(plan, u, bc):
    """B_val / B_tan of the Neumann kind (faces not replaced keep their Dirichlet lift) match the action's response."""
    op = export_p07_sparse(plan, "neumann")
    qd = len(plan.dirichlet_points)
    print(f"neumann kind: B_val nnz={op.dirichlet_value.nnz} B_tan nnz={op.dirichlet_tangential.nnz}")
    for part in ("value", "tangential"):
        data = BoundaryData(bc.dirichlet_value if part == "value" else np.zeros((qd, NF)),
                            bc.dirichlet_tangential if part == "tangential" else np.zeros((qd, 2, NF)),
                            np.zeros_like(bc.neumann_normal))
        ref = _action(plan, np.zeros_like(u), data, "neumann")
        mine = boundary_source(op, data)
        assert float(np.max(np.abs(mine - ref))) <= TOL * max(float(np.max(np.abs(ref))), 1.0)
        block = op.dirichlet_value if part == "value" else op.dirichlet_tangential
        assert (block.nnz > 0) == bool(np.any(ref != 0))


def test_save_load_round_trip(plan, u, bc, tmp_path):
    identity = {"case": "synthetic", "n": 3, "flags": [1, 2]}
    for kind in KINDS:
        op = export_p07_sparse(plan, kind)
        path = tmp_path / f"{kind}.npz"
        save_p07_sparse(path, op, identity)
        back = load_p07_sparse(path, identity)
        assert back.kind == kind
        for name in ("matrix", "dirichlet_value", "dirichlet_tangential", "neumann_normal"):
            a, b = getattr(op, name), getattr(back, name)
            assert a.shape == b.shape
            assert np.array_equal(a.data, b.data) and np.array_equal(a.indices, b.indices)
            assert np.array_equal(a.indptr, b.indptr)
        for name in ("owner_volume", "dirichlet_points", "neumann_points"):
            assert np.array_equal(getattr(op, name), getattr(back, name))
        assert np.array_equal(apply_p07_sparse(op, u, bc), apply_p07_sparse(back, u, bc))
        assert load_p07_sparse(path).kind == kind
        with pytest.raises(ValueError):
            load_p07_sparse(path, {**identity, "n": 4})


def test_load_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, metadata_json=np.asarray('{"schema": "other", "kind": "dirichlet", "identity": {}}'))
    with pytest.raises(ValueError):
        load_p07_sparse(path)


def test_unknown_kind(plan):
    with pytest.raises(ValueError):
        export_p07_sparse(plan, "robin")
