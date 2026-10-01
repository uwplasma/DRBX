"""Sparse P07 export vs ``p07_action`` on the real N32 owner-closure plan, final options (slow).

Same bounded closure as ``test_perpendicular_p07_operator_real`` but with the final campaign options
(``curvature="autodiff"``, ``face_quadrature="q2"``, ``inner_support="fixed_radius"``). Random fields and random
boundary data; both kinds; max difference relative to the action's scale and the nnz of every block are reported.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native.fci_perpendicular_p07_operator import p07_action
from drbx.native.fci_perpendicular_p07_sparse import KINDS, apply_p07_sparse, boundary_source, export_p07_sparse
from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData, zero_boundary_data

GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
N = 32
NF = 3
TOL = 1e-11
REPORT: dict = {}

pytestmark = pytest.mark.slow

try:
    from p_shared.replay_support import DEFAULT_SIDECAR as SIDECAR
except Exception:  # pragma: no cover - scripts tree unavailable
    SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"

_have = (GEOMETRY / f"{N}x{N}x{N}" / "base_geometry.npz").is_file() and Path(SIDECAR).is_file()
needs_inputs = pytest.mark.skipif(not _have, reason="HSX N32 geometry/sidecar inputs are unavailable")


@pytest.fixture(scope="module")
def plan():
    from p_shared import owner_closure as oc
    from p_shared.replay_support import build_environment
    from drbx.stencils.loader import LoaderGrid
    from drbx.stencils.operator_plan import lower_perpendicular_plan_from_rows

    env = build_environment(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR, curvature="autodiff",
                            face_quadrature="q2", inner_support="fixed_radius")
    owners = sorted(set(oc.select_owners(env.t, env.census).values()))
    provider = oc.load_provider_for_env(SIDECAR, curvature="autodiff", face_quadrature="q2")
    built = oc.build_owner_rows(env, owners, provider=provider)
    t = env.t
    grid = LoaderGrid.from_arrays(n=N, raw_to_owner=t.ro, eta_centers=t.centers[2])
    return lower_perpendicular_plan_from_rows(
        built["row_index"], built["neumann_index"], grid=grid, census=env.census, geometry=built["geometry"],
        raw_volume=t.rv, owner_volume=t.vol, raw_ids=built["raw_ids"], face_rows=built["face_row_indices"],
        p07_rows=built["p07_row_indices"])


@needs_inputs
@pytest.mark.parametrize("kind", KINDS)
def test_sparse_matches_action_on_real_closure(plan, kind):
    p07 = plan.p07
    assert p07.rows.boundary_query_count > 0 and p07.neumann is not None
    n = len(p07.owner_volume)
    qd, qn = len(plan.dirichlet_points), len(plan.neumann_points)
    rng = np.random.default_rng(7)
    u = rng.normal(size=(n, NF))
    bc = BoundaryData(rng.normal(size=(qd, NF)), rng.normal(size=(qd, 2, NF)), rng.normal(size=(qn, NF)))
    op = export_p07_sparse(plan, kind)
    ref = np.asarray(p07_action(plan, u, bc, kind))
    mine = apply_p07_sparse(op, u, bc)
    scale = float(np.max(np.abs(ref)))
    diff = float(np.max(np.abs(mine - ref)))
    lin_ref = np.asarray(p07_action(plan, u, zero_boundary_data(plan, NF), kind))
    lin_diff = float(np.max(np.abs(apply_p07_sparse(op, u) - lin_ref)))
    src_ref = np.asarray(p07_action(plan, np.zeros_like(u), bc, kind))
    src_diff = float(np.max(np.abs(boundary_source(op, bc) - src_ref)))
    REPORT[kind] = dict(
        n=n, Qd=qd, Qn=qn, scale=scale, max_abs=diff, rel=diff / scale, lin_rel=lin_diff / scale,
        src_rel=src_diff / max(float(np.max(np.abs(src_ref))), 1.0),
        nnz=dict(A=op.matrix.nnz, B_val=op.dirichlet_value.nnz, B_tan=op.dirichlet_tangential.nnz,
                 B_nn=op.neumann_normal.nnz))
    print(f"\nP07 sparse real closure [{kind}]: {REPORT[kind]}")
    assert diff <= TOL * scale
    assert lin_diff <= TOL * scale
    assert REPORT[kind]["src_rel"] <= TOL
