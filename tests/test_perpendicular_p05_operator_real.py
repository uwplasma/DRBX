"""P05 / P05N JAX bracket operator on real N32 owner-closure rows vs the host replay (P08 step 2b, E5; slow).

The bounded 12-owner closure of ``p_shared.owner_closure`` is lowered into a ``PerpendicularPlan``; boundary data
is evaluated once at the plan's point tables with the campaign field adapters; ``p05_terms`` / ``p05n_action`` are
compared per owner with the host replay's ``assemble_owner_terms`` output (``p05_centered``,
``p05_live_jump_owner_num`` and its per-face ``p05_live_jump_values``, ``p05n_{frozen,upwind}_{raw,face}_{N,D}``),
restricted to the closure's owners as ``compare_to_oracle`` does, and through ``compare_to_oracle`` against the
frozen oracles with the JAX terms substituted. Also: the plan's stored ``h`` / ``|J|`` / q3 weight vs the live
``env.ref._metric`` / quadrature values the host uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
N = 32
CAMPAIGNS = ("p05", "p05n_frozen", "p05n_upwind")
REPORT: dict = {}
#: max |JAX - host| over the closure's owners relative to the term's scale (max |host|); design gate G1.
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

    env = build_environment(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR, curvature="fd")
    owners = sorted(set(oc.select_owners(env.t, env.census).values()))
    built = oc.build_owner_rows(env, owners, provider=oc.load_provider_for_env(SIDECAR, curvature="fd"))
    paths = dict(DEFAULT_PATHS)
    oracle = ru._load_oracle_owner_values(env, paths, CAMPAIGNS)
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


def _p05_terms(closure):
    from p_shared.campaign_fields import P05Adapter
    from drbx.native.fci_perpendicular_p05_operator import p05_terms
    from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables
    env, plan = closure["env"], closure["plan"]
    a = P05Adapter(env.ref, closure["oracle"]["p05"]["owner_values"])
    bc = boundary_data_from_callables(plan, a.dirichlet, a.normal)
    return p05_terms(plan, a.owner_values, bc, a.field_kinds, a.pairs), (a, bc)


def _p05n_terms(closure, name):
    from p_shared.campaign_fields import build_adapter
    from drbx.native.fci_perpendicular_p05_operator import p05n_action
    from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables
    env, plan = closure["env"], closure["plan"]
    a = build_adapter(name, ref=env.ref, period=env.t.g.eta_period, owner_values=closure["oracle"][name]["owner_values"])
    bc = boundary_data_from_callables(plan, a.dirichlet, a.normal)
    role = a.reconstructions["role"]
    return p05n_action(plan, a.owner_values, bc, role.field_kinds, a.n_pair_index, a.d_pair_index,
                       columns=role.columns), (a, bc, role)


def _conditioning_floor(closure, name):
    """Absolute change of each P05N owner term under a random 1-ulp relative perturbation of the owner fields
    (max over two seeds): the roundoff floor of the *operator itself*. The face terms are jumps of smooth fields
    (``upper - lower`` ~ 1e-5 of the O(1) side values), divided by tiny owner volumes, so their scale is far
    below what one ulp of input noise produces; JAX-vs-host reconstruction roundoff (state agreement ~1e-15 on
    O(1) values) must lie below this floor."""
    from drbx.native.fci_perpendicular_p05_operator import p05n_action
    owners = closure["owners"]
    r, (a, bc, role) = _p05n_terms(closure, name)
    fields = np.asarray(a.owner_values)
    floor = {}
    for seed in (0, 1):
        pert = fields * (1 + 2.2e-16 * np.random.default_rng(seed).standard_normal(fields.shape))
        r2 = p05n_action(closure["plan"], pert, bc, role.field_kinds, a.n_pair_index, a.d_pair_index,
                         columns=role.columns)
        for term in ("raw_N", "raw_D", "face_N", "face_D"):
            d = float(np.max(np.abs(np.asarray(getattr(r2, term))[owners] - np.asarray(getattr(r, term))[owners])))
            floor[term] = max(floor.get(term, 0.0), d)
    return floor


@needs_inputs
def test_plan_geometry_is_the_live_host_geometry(closure):
    """The plan's ``h`` / ``|J|`` (geometry.npz) and q3 weight vs the live values ``_cells_unit_core`` /
    ``assemble_owner_terms`` compute (``env.ref._metric``, ``_quadrature``)."""
    env, ru, plan, built = closure["env"], closure["ru"], closure["plan"], closure["built"]
    t = env.t
    c, f = plan.cells, plan.faces
    metric = env.ref._metric(t.pts[built["raw_ids"]])
    h, jac = metric["bcov"] / metric["B"][:, None], np.abs(metric["J"])
    keys = env.census.keys()[built["face_row_indices"]]
    points, weight = ru.pshared_provider._quadrature(t.faces, keys, 3, face=True)
    mf = env.ref._metric(points.reshape(-1, 3))
    hf = (mf["bcov"] / mf["B"][:, None]).reshape(len(keys), 9, 3)
    rep = dict(cell_h_bitwise=bool(np.array_equal(c.h, h)), cell_h_max=float(np.max(np.abs(c.h - h))),
               cell_jac_bitwise=bool(np.array_equal(c.jac, jac)), cell_jac_max=float(np.max(np.abs(c.jac - jac))),
               face_h_bitwise=bool(np.array_equal(f.h, hf)), face_h_max=float(np.max(np.abs(f.h - hf))),
               face_weight_bitwise=bool(np.array_equal(f.weight, weight)),
               face_weight_max=float(np.max(np.abs(f.weight - weight))))
    REPORT["geometry_vs_live"] = rep
    print("\nE5 plan geometry vs live host:", rep)
    assert max(rep["cell_h_max"], rep["cell_jac_max"], rep["face_h_max"], rep["face_weight_max"]) <= 1e-13


@needs_inputs
def test_p05_and_p05n_terms_match_host_replay(closure):
    """Every P05 / P05N operator term (centered, live jump, N and D raw/face) vs ``assemble_owner_terms``."""
    from drbx.native.fci_perpendicular_p05_operator import P05NTerms  # noqa: F401
    host, plan, env, owners = closure["host"], closure["plan"], closure["env"], closure["owners"]
    vol = env.t.vol
    mask = plan.faces.p07_valid
    mine_cells, mine_faces = {}, {}

    t, _ = _p05_terms(closure)
    mine_cells["p05_centered"] = np.asarray(t.centered_numerator)
    mine_faces["p05_live_jump_owner_num"] = np.asarray(t.jump_numerator)
    assert _compare("p05_centered", np.asarray(t.centered_owner)[owners],
                    _dense(closure, host["cells"]["p05_centered"])) < TOL
    assert _compare("p05_live_jump_owner_num", np.asarray(t.jump_owner)[owners],
                    _dense(closure, host["faces"]["p05_live_jump_owner_num"])) < TOL
    # per-face jump values keyed by p07 id (the step-1 gate compares them with the saved upwind chunks)
    pids = plan.faces.p07_id[mask].astype(np.int64)
    order = np.argsort(pids)
    host_ids = np.asarray(host["faces"]["p05_live_jump_p07ids"])
    host_vals = np.asarray(host["faces"]["p05_live_jump_values"])
    horder = np.argsort(host_ids)
    np.testing.assert_array_equal(pids[order], host_ids[horder])
    mine_vals = np.asarray(t.face_jump)[mask]
    diff = np.abs(mine_vals[order] - host_vals[horder])
    REPORT["p05_live_jump_values"] = dict(max_abs=float(diff.max()), scale=float(np.abs(host_vals).max()),
                                          faces=int(len(pids)),
                                          max_rel_to_scale=float(diff.max() / np.abs(host_vals).max()))
    assert REPORT["p05_live_jump_values"]["max_rel_to_scale"] < TOL
    assert np.all(np.asarray(t.face_jump)[~mask] == 0.0)
    mine_faces["p05_live_jump_p07ids"], mine_faces["p05_live_jump_values"] = pids, mine_vals

    for name in ("p05n_frozen", "p05n_upwind"):
        r, _ = _p05n_terms(closure, name)
        floors = _conditioning_floor(closure, name)
        for suffix in ("N", "D"):
            key = f"{name}_raw_{suffix}"
            mine_cells[key] = np.asarray(getattr(r, f"raw_{suffix}_numerator"))
            assert _compare(key, np.asarray(getattr(r, f"raw_{suffix}"))[owners],
                            _dense(closure, host["cells"][key])) < TOL
            key = f"{name}_face_{suffix}"
            mine_faces[key] = np.asarray(getattr(r, f"face_{suffix}_numerator"))
            rel = _compare(key, np.asarray(getattr(r, f"face_{suffix}"))[owners], _dense(closure, host["faces"][key]))
            floor = floors[f"face_{suffix}"]
            REPORT[key]["ulp_perturbation_floor_abs"] = floor
            # the gate for a face term is its conditioning floor: 1e-11 of the term scale is below one ulp of
            # input noise here (see _conditioning_floor); the arithmetic itself is verified separately on the
            # host state (test_operator_arithmetic_on_host_state_matches_host_replay, <= 1e-13 of scale)
            assert rel < TOL or REPORT[key]["max_abs"] <= 10 * floor, (key, REPORT[key])
            assert rel < 1e-8
    closure["mine"] = (mine_cells, mine_faces)
    print("\nE5 real N32 closure report:", REPORT)


@needs_inputs
def test_operator_arithmetic_on_host_state_matches_host_replay(closure):
    """The operator arithmetic alone: host D/N reconstructions of the same rows (``p_shared.apply`` and the replay's
    side helpers) fed to ``p05_terms_from_state`` reproduce every host term to roundoff of the term scale, for P05
    (D state, ``p07_valid`` domain) and both P05N catalogues (role-selected state, all faces)."""
    from types import SimpleNamespace
    from p_shared.campaign_fields import P05Adapter, build_adapter
    from drbx.native.fci_perpendicular_p05_operator import p05_terms_from_state
    from tests.perpendicular_synthetic import host_cells, host_faces

    env, plan, built, host, owners = closure["env"], closure["plan"], closure["built"], closure["host"], closure["owners"]
    vol = env.t.vol
    world = SimpleNamespace(raw_ids=built["raw_ids"], face_rows=built["face_row_indices"], census=env.census, n=N,
                            row_index=built["row_index"], neumann_index=built["neumann_index"])
    errors = {}

    def check(key, mine_num, hostpair):
        errors[key] = _compare(key + ".host_state", np.asarray(mine_num)[owners] / vol[owners, None],
                               _dense(closure, hostpair))

    a = P05Adapter(env.ref, closure["oracle"]["p05"]["owner_values"])
    zero_normal = SimpleNamespace(dirichlet=a.dirichlet, normal=lambda q: np.zeros((len(q), a.n_fields)))  # D only
    (cd, _cn), fh = host_cells(world, a.owner_values, zero_normal), host_faces(world, a.owner_values, zero_normal)
    t = p05_terms_from_state(plan, cd[1], fh["cgd"], fh["ld"], fh["ud"], a.pairs)
    check("p05_centered", t.centered_numerator, host["cells"]["p05_centered"])
    check("p05_live_jump_owner_num", t.jump_numerator, host["faces"]["p05_live_jump_owner_num"])
    for name in ("p05n_frozen", "p05n_upwind"):
        a = build_adapter(name, ref=env.ref, period=env.t.g.eta_period, owner_values=closure["oracle"][name]["owner_values"])
        (cd, cn), fh = host_cells(world, a.owner_values, a), host_faces(world, a.owner_values, a)
        role = a.reconstructions["role"]
        cols, isn = np.asarray(role.columns), np.asarray([k == "neumann" for k in role.field_kinds])
        grad = np.where(isn[None, None, :], cn[1][:, :, cols], cd[1][:, :, cols])
        cg = np.where(isn[None, None, None, :], fh["cgn"][..., cols], fh["cgd"][..., cols])
        lo = np.where(isn[None, None, :], fh["ln"][..., cols], fh["ld"][..., cols])
        up = np.where(isn[None, None, :], fh["un"][..., cols], fh["ud"][..., cols])
        P = len(a.n_pair_index)
        t = p05_terms_from_state(plan, grad, cg, lo, up, tuple(a.n_pair_index) + tuple(a.d_pair_index),
                                 jump_mask=np.ones(len(plan.faces.census_row), dtype=bool))
        check(f"{name}_raw_N", t.centered_numerator[:, :P], host["cells"][f"{name}_raw_N"])
        check(f"{name}_raw_D", t.centered_numerator[:, P:], host["cells"][f"{name}_raw_D"])
        check(f"{name}_face_N", t.jump_numerator[:, :P], host["faces"][f"{name}_face_N"])
        check(f"{name}_face_D", t.jump_numerator[:, P:], host["faces"][f"{name}_face_D"])
    REPORT["host_state_rel"] = errors
    print("\nE5 operator arithmetic on host state (rel to term scale):", errors)
    assert max(errors.values()) < 1e-13, errors


@needs_inputs
def test_compare_to_oracle_with_jax_terms(closure):
    """``owner_closure.compare_to_oracle`` with the JAX terms substituted (host ``raw_R`` and the P05 antisymmetry
    kept), same tolerances as the step-1 preflight."""
    if "mine" not in closure:
        test_p05_and_p05n_terms_match_host_replay(closure)
    oc, env, host = closure["oc"], closure["env"], closure["host"]
    mine_cells, mine_faces = closure["mine"]
    uniq = np.arange(len(env.t.vol))
    out = {"cells": dict(host["cells"]), "faces": dict(host["faces"]), "p07": dict(host["p07"])}
    for key, value in mine_cells.items():
        out["cells"][key] = (uniq, value)
    for key, value in mine_faces.items():
        out["faces"][key] = value if key.startswith("p05_live_jump_p") or key.endswith("values") else (uniq, value)
    table = oc.compare_to_oracle(env, out, closure["owners"], closure["paths"], CAMPAIGNS)
    host_table = oc.compare_to_oracle(env, host, closure["owners"], closure["paths"], CAMPAIGNS)
    rows = [(r["campaign"], r["term"], r["max_abs"], r["ratio_to_oracle_NR"], r["pass"]) for r in table]
    REPORT["oracle_rows_jax"] = rows
    REPORT["oracle_rows_host"] = [(r["campaign"], r["term"], r["max_abs"], r["ratio_to_oracle_NR"], r["pass"])
                                  for r in host_table]
    for row in rows:
        print("E5 compare_to_oracle:", row)
    assert len(rows) == len(host_table) and rows
    assert all(r["pass"] for r in table)


@needs_inputs
def test_real_plan_eager_equals_jit(closure):
    from drbx.native.fci_perpendicular_p05_operator import p05_terms, p05n_action
    plan = closure["plan"]
    t, (a, bc) = _p05_terms(closure)
    jitted = jax.jit(lambda p, f, b: p05_terms(p, f, b, a.field_kinds, a.pairs))(plan, a.owner_values, bc)
    for x, y in zip(t, jitted):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    r, (an, bcn, role) = _p05n_terms(closure, "p05n_frozen")
    jn = jax.jit(lambda p, f, b: p05n_action(p, f, b, role.field_kinds, an.n_pair_index, an.d_pair_index,
                                             columns=role.columns))(plan, an.owner_values, bcn)
    for x, y in zip(r, jn):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
