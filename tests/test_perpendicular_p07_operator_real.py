"""P07 / P07N JAX operator on real N32 owner-closure rows vs the host replay (P08 step 2b, E3; slow).

The bounded 12-owner closure of ``p_shared.owner_closure`` (real rows and geometry for the incident
faces and raw cells) is lowered into a ``PerpendicularPlan``; boundary data is evaluated once at the
plan's point tables with the campaign field adapters; ``p07_action`` is compared per owner with the
host replay's ``assemble_owner_terms`` output (``p07_global_N``, ``p07n_global_N``, ``p07n_global_D``),
restricted to the closure's owners as ``compare_to_oracle`` does, and through ``compare_to_oracle``
against the frozen oracles. Also: the plan's stored conditioned-face integrand vs the live
``contract_face_tensor(weight, _perpendicular_flux_tensor)`` path.
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

GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
N = 32
CAMPAIGNS = ("p07", "p07n")
REPORT: dict = {}
#: max |JAX - host| over the closure's owners relative to the term's scale (max |host|). The host's own
#: arithmetic differs only by roundoff order (lift subtracted per donor vs after the contraction); measured
#: 1e-13..2e-12 (owner action = flux / owner volume amplifies the face roundoff).
TOL = 1e-11

pytestmark = pytest.mark.slow

_have = ((GEOMETRY / f"{N}x{N}x{N}" / "base_geometry.npz").is_file() and SIDECAR.is_file())
needs_inputs = pytest.mark.skipif(not _have, reason="HSX N32 geometry/sidecar inputs are unavailable")


@pytest.fixture(scope="module")
def closure():
    from p_shared import owner_closure as oc
    from p_shared import replay_units as ru
    from p_shared.replay_support import DEFAULT_PATHS, build_environment
    from drbx.stencils.loader import LoaderGrid
    from drbx.stencils.operator_plan import lower_perpendicular_plan_from_rows

    env = build_environment(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR, curvature="fd", face_quadrature="q3", inner_support="profile7")
    owners = sorted(set(oc.select_owners(env.t, env.census).values()))
    built = oc.build_owner_rows(env, owners, provider=oc.load_provider_for_env(SIDECAR, curvature="fd", face_quadrature="q3"))
    paths = dict(DEFAULT_PATHS)
    oracle = ru._load_oracle_owner_values(env, paths, CAMPAIGNS + ("p05n_frozen",))
    host = oc.assemble_owner_terms(env, built, CAMPAIGNS, oracle)
    t = env.t
    grid = LoaderGrid.from_arrays(n=N, raw_to_owner=t.ro, eta_centers=t.centers[2])
    plan = lower_perpendicular_plan_from_rows(
        built["row_index"], built["neumann_index"], grid=grid, census=env.census, geometry=built["geometry"],
        raw_volume=t.rv, owner_volume=t.vol, raw_ids=built["raw_ids"], face_rows=built["face_row_indices"],
        p07_rows=built["p07_row_indices"])
    return dict(env=env, oc=oc, ru=ru, paths=paths, oracle=oracle, host=host, built=built, plan=plan,
                owners=np.asarray(owners, dtype=np.int64))


def _dense(closure, pair):
    oc, env = closure["oc"], closure["env"]
    return oc.owner_values_from_pairs(pair, closure["owners"], len(env.t.vol)) / env.t.vol[closure["owners"], None]


def _compare(name, mine, host):
    diff = np.abs(mine - host)
    scale = float(np.max(np.abs(host)))
    REPORT[name] = dict(max_abs=float(diff.max()), scale=scale, max_rel_to_scale=float(diff.max() / scale))
    return REPORT[name]["max_rel_to_scale"]


@needs_inputs
def test_plan_of_the_real_closure_is_consistent(closure):
    plan, built, env = closure["plan"], closure["built"], closure["env"]
    p = plan.p07
    assert len(p.p07_id) == len(built["p07_row_indices"])
    assert set(np.unique(p.family[p.conditioned]).tolist()) <= {1, 2, 4} and p.conditioned.any()
    f = plan.faces
    assert f.has_missing_side and f.wall.any()
    assert plan.cells.conditioned.any() and f.common_conditioned.any() and (f.lower_conditioned | f.upper_conditioned).any()
    for points in (plan.dirichlet_points, plan.neumann_points):
        assert len(points) > 0
    REPORT["plan"] = dict(cells=len(plan.cells.raw_ids), faces=len(f.census_row), p07_faces=len(p.p07_id),
                          conditioned_p07=int(p.conditioned.sum()), dirichlet_points=len(plan.dirichlet_points),
                          neumann_points=len(plan.neumann_points))


@needs_inputs
def test_p07_conditioned_integrand_vs_live_path(closure):
    """The stored (geometry.npz-sourced) conditioned-face integrand vs the live host path
    ``contract_face_tensor(_quadrature weight, env.ref._perpendicular_flux_tensor, axis)``."""
    from drbx.geometry.fci_perpendicular_integrated_rows import contract_face_tensor
    env, ru, plan = closure["env"], closure["ru"], closure["plan"]
    p = plan.p07
    census_rows = p.census_row[p.neumann_face]
    keys = env.census.keys()[census_rows]
    points, weight = ru.pshared_provider._quadrature(env.t.faces, keys, 3, face=True)
    tensor = env.ref._perpendicular_flux_tensor(points.reshape(-1, 3)).reshape(len(keys), 9, 3, 3)
    live = contract_face_tensor(weight, tensor, keys[:, 0])
    diff = np.abs(np.asarray(p.integrand) - live)
    REPORT["integrand"] = dict(bitwise=bool(np.array_equal(np.asarray(p.integrand), live)), max_abs=float(diff.max()),
                               max_rel=float((diff / np.maximum(np.abs(live), 1e-300)).max()), faces=len(keys))
    assert diff.max() <= 1e-9 * np.abs(live).max()


@needs_inputs
def test_p07_and_p07n_actions_match_host_replay(closure):
    from p_shared.campaign_fields import P07Adapter, P07NAdapter
    from drbx.native.fci_perpendicular_p07_operator import p07_action
    from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables

    env, plan, host, oc = closure["env"], closure["plan"], closure["host"], closure["oc"]
    period = env.t.g.eta_period
    owners = closure["owners"]
    dense = lambda pair: _dense(closure, pair)
    mine = {}

    a07 = P07Adapter(env.ref, closure["oracle"]["p07"]["owner_values"])
    bc = boundary_data_from_callables(plan, a07.dirichlet, a07.normal)
    action = p07_action(plan, a07.owner_values, bc, a07.field_kinds)
    mine["p07_global_N"] = np.asarray(action)
    assert _compare("p07_global_N", mine["p07_global_N"][owners], dense(host["p07"]["p07_global_N"])) < TOL

    a07n = P07NAdapter(env.ref, period, closure["oracle"]["p07n"]["owner_values"])
    bc = boundary_data_from_callables(plan, a07n.dirichlet, a07n.normal)
    for key, kinds in (("p07n_global_N", a07n.reconstructions["N"].field_kinds),
                       ("p07n_global_D", a07n.reconstructions["D"].field_kinds)):
        mine[key] = np.asarray(p07_action(plan, a07n.owner_values, bc, kinds))
        assert _compare(key, mine[key][owners], dense(host["p07"][key])) < TOL

    # the frozen-oracle comparison of the JAX terms (host O_q3 kept), with the step-1 tolerances
    vol = env.t.vol
    uniq = np.arange(len(vol))
    out = {"cells": {}, "faces": {}, "p07": {key: (uniq, mine[key] * vol[:, None]) for key in mine}}
    out["p07"]["p07n_global_O_q3"] = host["p07"]["p07n_global_O_q3"]
    table = oc.compare_to_oracle(env, out, owners, closure["paths"], CAMPAIGNS)
    REPORT["oracle_rows"] = [(r["campaign"], r["term"], r["max_abs"], r["ratio_to_oracle_NR"], r["pass"]) for r in table]
    assert all(r["pass"] for r in table)
    print("\nE3 real N32 closure report:", REPORT)


@needs_inputs
def test_real_plan_eager_equals_jit_and_jvp_is_linear(closure):
    import jax.numpy as jnp
    from p_shared.campaign_fields import P07NAdapter
    from drbx.native.fci_perpendicular_p07_operator import p07_action
    from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables, zero_boundary_data

    env, plan = closure["env"], closure["plan"]
    a = P07NAdapter(env.ref, env.t.g.eta_period, closure["oracle"]["p07n"]["owner_values"])
    bc = boundary_data_from_callables(plan, a.dirichlet, a.normal)
    kinds = a.reconstructions["N"].field_kinds
    eager = p07_action(plan, a.owner_values, bc, kinds)
    jitted = jax.jit(lambda p, f, b: p07_action(p, f, b, kinds))(plan, a.owner_values, bc)
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))
    tangent = np.random.default_rng(3).normal(size=a.owner_values.shape)
    _, jvp = jax.jvp(lambda f: p07_action(plan, f, bc, kinds), (jnp.asarray(a.owner_values),), (jnp.asarray(tangent),))
    linear = p07_action(plan, tangent, zero_boundary_data(plan, tangent.shape[1]), kinds)
    scale = float(np.max(np.abs(np.asarray(linear))))
    err = float(np.max(np.abs(np.asarray(jvp) - np.asarray(linear))))
    REPORT["jvp_vs_zero_bc_apply_rel"] = err / scale
    assert err <= 1e-12 * scale


@needs_inputs
def test_real_closure_state_matches_host_applications(closure):
    """D and N reconstructions of every raw cell and face of the real closure (P05N-frozen physical fields:
    conditioned cells/common/side rows, wall faces with a missing upper side) vs ``p_shared.apply`` and the
    replay's own side helpers applied to the same row objects."""
    from types import SimpleNamespace
    from p_shared.campaign_fields import build_adapter
    from drbx.native.fci_perpendicular_reconstruction_state import (
        boundary_data_from_callables, cell_state_pair, face_state_pair)
    from tests.perpendicular_synthetic import host_cells, host_faces

    env, plan, built = closure["env"], closure["plan"], closure["built"]
    a = build_adapter("p05n_frozen", ref=env.ref, period=env.t.g.eta_period,
                      owner_values=closure["oracle"]["p05n_frozen"]["owner_values"])
    bc = boundary_data_from_callables(plan, a.dirichlet, a.normal)
    world = SimpleNamespace(raw_ids=built["raw_ids"], face_rows=built["face_row_indices"], census=env.census,
                            n=N, row_index=built["row_index"], neumann_index=built["neumann_index"])
    ov = a.owner_values
    (vd, gd), (vn, gn) = host_cells(world, ov, a)
    cd, cn = cell_state_pair(plan, ov, bc)
    h = host_faces(world, ov, a)
    fd, fn = face_state_pair(plan, ov, bc)

    def rel(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return float(np.max(np.abs(x - y)) / max(float(np.max(np.abs(y))), 1e-300))

    errors = {"cell_value_D": rel(cd.value, vd), "cell_gradient_D": rel(cd.gradient, gd),
              "cell_value_N": rel(cn.value, vn), "cell_gradient_N": rel(cn.gradient, gn),
              "face_value_D": rel(fd.value, h["cvd"]), "face_gradient_D": rel(fd.gradient, h["cgd"]),
              "face_value_N": rel(fn.value, h["cvn"]), "face_gradient_N": rel(fn.gradient, h["cgn"]),
              "lower_D": rel(fd.lower, h["ld"]), "upper_D": rel(fd.upper, h["ud"]),
              "lower_N": rel(fn.lower, h["ln"]), "upper_N": rel(fn.upper, h["un"])}
    REPORT["state_rel_errors"] = errors
    print("\nE3 real state errors (rel to scale):", errors)
    assert max(errors.values()) < 1e-12, errors
    # the Neumann reconstruction really differs from the Dirichlet lift on this closure
    assert np.max(np.abs(gn - gd)) > 1e-6 and np.max(np.abs(h["cvn"] - h["cvd"])) > 1e-6
    assert np.max(np.abs(h["ln"] - h["ld"])) > 1e-6 and (~plan.faces.upper_present).any()


@pytest.fixture(scope="module")
def tensor_artifact_plan(closure, tmp_path_factory):
    """The closure's rows rebuilt with ``capture_factors=True`` and written as a v3 artifact with tensor-factored
    unconditioned sources (the production encoding), lowered through ``lower_perpendicular_plan_from_artifact``."""
    from drbx.stencils import artifact as art
    from drbx.stencils import builder as stencil_builder
    from drbx.stencils.loader import LoaderGrid
    from drbx.stencils.operator_plan import lower_perpendicular_plan_from_artifact

    env, built = closure["env"], closure["built"]
    geometry = built["geometry"]
    kwargs = dict(normal_coefficients=env.normal_coefficients, patch_cache={})
    r1, n1 = stencil_builder.build_r1_cell_rows(env.S, env.ctx, built["raw_ids"], capture_factors=True, **kwargs)
    r2, n2 = stencil_builder.build_r2_face_rows(env.S, env.ctx, env.census, built["face_row_indices"],
                                                geometry.face_points, capture_factors=True, **kwargs)
    r3, n3 = stencil_builder.build_r3_side_rows(env.S, env.ctx, env.census, built["face_row_indices"],
                                                geometry.face_points, capture_factors=True, **kwargs)
    face_pos = np.searchsorted(built["face_row_indices"], built["p07_row_indices"])
    r4, n4 = stencil_builder.build_r4_p07_rows(
        env.ctx, env.census, built["p07_row_indices"], geometry.face_points[face_pos],
        geometry.p06_face_weight[face_pos], geometry.p07_face_tensor[face_pos], **kwargs)

    def pack(requests):
        return art.pack_point_rows(
            [r.row for r in requests], request=[r.request for r in requests], entity_id=[r.entity_id for r in requests],
            bc_variant=[r.bc_variant for r in requests], radial_degree=[r.radial_degree for r in requests],
            store_gradient=[not str(r.request).startswith("R3") for r in requests])

    neumann = n1 + n2 + n3 + n4
    chunks = dict(
        cells=(pack(r1),), faces=(pack(r2 + r3),), p07=(art.pack_integrated_rows(
            [r.row for r in r4], entity_id=[r.entity_id for r in r4]),),
        neumann=(art.pack_neumann_rows(
            [r.row for r in neumann], entity_id=[r.entity_id for r in neumann], quad_node=[r.quad_node for r in neumann],
            request=[r.source for r in neumann], radial_degree=[r.radial_degree for r in neumann]),))
    stats: dict = {}
    root = tmp_path_factory.mktemp("tensor_artifact")
    art.save_row_artifact(root, N, identity={"e3": "tensor"}, stats=stats, **chunks,
                          point_factors={"cells": [[r.factors for r in r1]], "faces": [[r.factors for r in r2 + r3]]})
    t = env.t
    plan = lower_perpendicular_plan_from_artifact(
        root, N, grid=LoaderGrid.from_arrays(n=N, raw_to_owner=t.ro, eta_centers=t.centers[2]), census=env.census,
        geometry=geometry, raw_volume=t.rv, owner_volume=t.vol, raw_ids=built["raw_ids"],
        face_rows=built["face_row_indices"], p07_rows=built["p07_row_indices"])
    REPORT["tensor_encoding"] = {k: v for k, v in stats.items()}
    return plan


@needs_inputs
def test_tensor_encoded_artifact_plan_matches_in_memory_plan(closure, tensor_artifact_plan):
    """Same state and P07 action from the tensor-factored artifact (tensor sources never expanded) as from the
    CSR in-memory plan of the same rows."""
    import jax.numpy as jnp
    from p_shared.campaign_fields import P07NAdapter, build_adapter
    from drbx.native.fci_perpendicular_p07_operator import p07_action
    from drbx.native.fci_perpendicular_reconstruction_state import (
        boundary_data_from_callables, cell_state_pair, face_state_pair)

    env, plan, tensor_plan = closure["env"], closure["plan"], tensor_artifact_plan
    assert any(len(b.value_slots) for b in tensor_plan.faces.rows.tensor_batches + tensor_plan.cells.rows.tensor_batches)
    a = build_adapter("p05n_frozen", ref=env.ref, period=env.t.g.eta_period,
                      owner_values=closure["oracle"]["p05n_frozen"]["owner_values"])
    ov = a.owner_values

    def rel(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return float(np.max(np.abs(x - y)) / max(float(np.max(np.abs(y))), 1e-300))

    states = []
    for p in (plan, tensor_plan):
        bc = boundary_data_from_callables(p, a.dirichlet, a.normal)
        states.append((cell_state_pair(p, ov, bc), face_state_pair(p, ov, bc)))
    (cells_a, faces_a), (cells_b, faces_b) = states
    errors = {}
    for name, x, y in (("cells", cells_a, cells_b), ("faces", faces_a, faces_b)):
        for kind, sx, sy in zip(("D", "N"), x, y):
            for field, u, v in zip(sx._fields, sx, sy):
                errors[f"{name}_{kind}_{field}"] = rel(u, v)
    a07n = P07NAdapter(env.ref, env.t.g.eta_period, closure["oracle"]["p07n"]["owner_values"])
    kinds = a07n.reconstructions["N"].field_kinds
    actions = [p07_action(p, a07n.owner_values, boundary_data_from_callables(p, a07n.dirichlet, a07n.normal), kinds)
               for p in (plan, tensor_plan)]
    errors["p07n_N"] = rel(actions[1], actions[0])
    REPORT["tensor_vs_csr_plan_rel"] = errors
    print("\nE3 tensor-artifact plan vs CSR plan (rel):", errors, REPORT.get("tensor_encoding"))
    assert max(errors.values()) < 1e-12, errors
