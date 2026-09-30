"""P06N / P06-legacy JAX curvature operator on real N32 owner-closure rows vs the host replay (P08 step 2b, E4; slow).

Same setup as ``tests/test_perpendicular_p07_operator_real.py``: the bounded 12-owner closure of
``p_shared.owner_closure`` (real rows and geometry) is lowered into a ``PerpendicularPlan``; boundary data
is evaluated once at the plan's point tables with the campaign adapters; ``p06_action`` is compared per
owner with the host replay's ``assemble_owner_terms`` output for the P06 terms (``p06n_raw_material`` /
``remainder`` / ``total``, ``p06n_faces_correction``, ``p06legacy_raw_centered``,
``p06legacy_faces_correction``), restricted to the closure owners as ``compare_to_oracle`` does (the
q1 evolution volume is complete only for owners whose raw cells are all in the closure), and through
``compare_to_oracle`` against the frozen oracles with the JAX terms substituted (host ``R_*`` kept).
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
CAMPAIGNS = ("p06n", "p06_legacy")
REPORT: dict = {}
#: max |JAX - host| over the closure's owners relative to the term's scale (max |host|)
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
    """Sparse host pair -> ``(len(owners), ...)`` numerator at the closure owners (no division)."""
    return closure["oc"].owner_values_from_pairs(pair, closure["owners"], len(closure["env"].t.vol))


def _compare(name, mine, host, scale=None):
    """max |mine - host| relative to the term's scale (default: max |host| of this array). A term is scaled by its
    largest value over all variants/fields (``scale``): the constant-field control variants are cancellation
    roundoff (~2e-12) and have no scale of their own."""
    diff = np.abs(np.asarray(mine) - host)
    scale = float(np.max(np.abs(host))) if scale is None else float(scale)
    REPORT[name] = dict(max_abs=float(diff.max()), scale=scale, max_rel_to_scale=float(diff.max() / scale))
    return REPORT[name]["max_rel_to_scale"]


@pytest.fixture(scope="module")
def p06n(closure):
    """JAX P06N terms for every variant plus the host arithmetic's per-variant layout."""
    from p_shared.campaign_fields import P06NAdapter
    from drbx.native.fci_perpendicular_p06_operator import bc_columns, p06_action, p06n_layout
    from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables

    env, plan = closure["env"], closure["plan"]
    adapter = P06NAdapter(env.ref, env.t.g.eta_period, closure["oracle"]["p06n"]["owner_values"])
    recs = [adapter.reconstructions[name] for name in adapter.variant_names]
    columns, kinds, groups = p06n_layout(recs)
    bc = boundary_data_from_callables(plan, adapter.dirichlet, adapter.normal)
    fields = adapter.owner_values[:, columns]
    action = p06_action(plan, fields, bc_columns(bc, columns), kinds, groups)
    REPORT["p06n_layout"] = dict(variants=len(recs), unique_columns=len(columns), neumann_columns=kinds.count("neumann"))
    return dict(adapter=adapter, action=action, columns=columns, kinds=kinds, groups=groups, bc=bc, fields=fields)


@pytest.fixture(scope="module")
def legacy(closure):
    """JAX P06-legacy terms for the five legacy fields (all Dirichlet, seam multiplier)."""
    from p_shared.campaign_fields import P06LegacyAdapter, legacy_seam_multiplier
    from drbx.native.fci_perpendicular_p06_operator import p06_action
    from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables

    env, plan = closure["env"], closure["plan"]
    adapter = P06LegacyAdapter(env.ref, closure["oracle"]["p06_legacy"]["owner_values"])
    multiplier = legacy_seam_multiplier(env.census.keys()[plan.faces.census_row])
    out = {}
    for name, field in adapter:
        bc = boundary_data_from_callables(plan, field.dirichlet)
        out[name] = p06_action(plan, field.owner_values, bc, field.field_kinds, face_multiplier=multiplier)
    REPORT["legacy_seam_faces"] = int((multiplier == 2.0).sum())
    return dict(adapter=adapter, actions=out, multiplier=multiplier)


@needs_inputs
def test_plan_of_the_real_closure_has_wall_and_seam_faces(closure, legacy):
    plan, env = closure["plan"], closure["env"]
    f = plan.faces
    assert f.wall.any() and f.has_missing_side and plan.cells.conditioned.any() and f.common_conditioned.any()
    assert legacy["multiplier"].max() == 2.0                                   # the closure contains legacy seam faces
    # the plan's evolution volume is the host's, at every closure owner (raw cells complete for these owners)
    host_ev = _dense(closure, closure["host"]["cells"]["q1_evolution_volume"])
    ev = np.asarray(plan.cells.evolution_volume)[closure["owners"]]
    assert np.max(np.abs(ev - host_ev)) <= 1e-13 * np.max(host_ev)
    owners_complete = np.isin(env.t.ro, closure["owners"]).sum() == len(plan.cells.raw_ids)
    assert owners_complete                                                     # all raw cells of the closure owners present


@needs_inputs
def test_p06n_terms_match_host_replay(closure, p06n):
    host, action = closure["host"], p06n["action"]
    owners = closure["owners"]
    names = p06n["adapter"].variant_names
    ev = np.maximum(_dense(closure, host["cells"]["q1_evolution_volume"]), 1e-300)[:, None]
    worst = {}
    for label in ("material", "remainder", "total"):
        mean = np.asarray(getattr(action, label))
        host_means = [_dense(closure, host["cells"][f"p06n_raw_{label}"][vi]) / ev for vi in range(len(names))]
        scale = max(float(np.max(np.abs(h))) for h in host_means)
        for vi, name in enumerate(names):
            worst[label] = max(worst.get(label, 0.0), _compare(f"p06n_{label}[{name}]", mean[vi][owners], host_means[vi], scale))
    host_nums = [_dense(closure, host["faces"]["p06n_faces_correction"][vi]) for vi in range(len(names))]
    # The q3 correction is a small fluctuation term (a jump of near-equal states): its own scale is ~1e-3 of the q1
    # material's, and the state roundoff (<= ~1e-13 relative, lift-subtraction order) is amplified by that
    # cancellation. It is gated against the scale of the term it is added to (material); its own-scale error is
    # reported and loosely gated.
    scale_mat_num = max(float(np.max(np.abs(_dense(closure, host["cells"]["p06n_raw_material"][vi]))))
                        for vi in range(len(names)))
    scale_mat = max(float(np.max(np.abs(_dense(closure, host["cells"]["p06n_raw_material"][vi]) / ev)))
                    for vi in range(len(names)))
    own_num = max(float(np.max(np.abs(h))) for h in host_nums)
    own_div = max(float(np.max(np.abs(h / ev))) for h in host_nums)
    own = {}
    for vi, name in enumerate(names):
        num = np.asarray(action.correction_numerator)[vi][owners]
        div = np.asarray(action.correction)[vi][owners]
        worst["correction_numerator"] = max(worst.get("correction_numerator", 0.0), _compare(
            f"p06n_corr[{name}]", num, host_nums[vi], scale_mat_num))
        worst["correction"] = max(worst.get("correction", 0.0), _compare(
            f"p06n_corr_div[{name}]", div, host_nums[vi] / ev, scale_mat))
        own["correction_numerator"] = max(own.get("correction_numerator", 0.0), float(np.max(np.abs(num - host_nums[vi])) / own_num))
        own["correction"] = max(own.get("correction", 0.0), float(np.max(np.abs(div - host_nums[vi] / ev)) / own_div))
    REPORT["p06n_correction_own_scale_rel"] = own
    print("\nE4 real N32 P06N correction error relative to its own scale:", own)
    assert max(own.values()) < 1e-8, own
    REPORT["p06n_worst_rel_to_scale"] = worst
    print("\nE4 real N32 P06N worst rel-to-scale:", worst)
    assert max(worst.values()) < TOL, worst


@needs_inputs
def test_p06_legacy_terms_match_host_replay(closure, legacy):
    host = closure["host"]
    owners = closure["owners"]
    ev = np.maximum(_dense(closure, host["cells"]["q1_evolution_volume"]), 1e-300)[:, None]
    worst, own = {}, {}
    fields = list(legacy["actions"])
    host_mean = {(f, t): _dense(closure, host["cells"]["p06legacy_raw_centered"][f][t]) / ev
                 for f in fields for t in ("material", "remainder", "total")}
    host_corr = {f: _dense(closure, host["faces"]["p06legacy_faces_correction"][f]) / ev for f in fields}
    for field_name, action in legacy["actions"].items():
        for term in ("material", "remainder", "total"):
            scale = max(float(np.max(np.abs(host_mean[(f, term)]))) for f in fields)
            worst[term] = max(worst.get(term, 0.0), _compare(f"p06legacy_{field_name}_{term}",
                                                             np.asarray(getattr(action, term))[owners],
                                                             host_mean[(field_name, term)], scale))
        scale = max(float(np.max(np.abs(host_mean[(f, "material")]))) for f in fields)    # see the P06N test
        worst["correction"] = max(worst.get("correction", 0.0), _compare(
            f"p06legacy_{field_name}_corr", np.asarray(action.correction)[owners], host_corr[field_name], scale))
        own_scale = max(float(np.max(np.abs(host_corr[f]))) for f in fields)
        own["correction"] = max(own.get("correction", 0.0), float(
            np.max(np.abs(np.asarray(action.correction)[owners] - host_corr[field_name])) / own_scale))
    REPORT["p06_legacy_correction_own_scale_rel"] = own
    print("\nE4 real N32 P06-legacy correction error relative to its own scale:", own)
    assert max(own.values()) < 1e-8, own
    REPORT["p06_legacy_worst_rel_to_scale"] = worst
    print("\nE4 real N32 P06-legacy worst rel-to-scale:", worst)
    assert set(legacy["actions"]) == set(legacy["adapter"].field_names)
    assert max(worst.values()) < TOL, worst


@needs_inputs
def test_jax_terms_pass_compare_to_oracle(closure, p06n, legacy):
    """``compare_to_oracle`` with the JAX P06 terms substituted (host ``R_*`` terms kept), every row."""
    env, host, oc = closure["env"], closure["host"], closure["oc"]
    n_total = len(env.t.vol)
    uniq = np.arange(n_total)
    action = p06n["action"]
    V = action.material.shape[0]
    cells = {"q1_evolution_volume": (uniq, np.asarray(action.evolution_volume))}
    for label, arr in (("material", action.material_numerator), ("remainder", action.remainder_numerator)):
        cells[f"p06n_raw_{label}"] = [(uniq, np.asarray(arr)[v]) for v in range(V)]
    cells["p06n_raw_total"] = [(uniq, np.asarray(action.material_numerator)[v] + np.asarray(action.remainder_numerator)[v])
                               for v in range(V)]
    for label in ("R_material", "R_remainder", "R_total"):
        cells[f"p06n_raw_{label}"] = host["cells"][f"p06n_raw_{label}"]
    faces = {"p06n_faces_correction": [(uniq, np.asarray(action.correction_numerator)[v]) for v in range(V)]}
    cells["p06legacy_raw_centered"] = {}
    faces["p06legacy_faces_correction"] = {}
    for name, a in legacy["actions"].items():
        cells["p06legacy_raw_centered"][name] = {
            "material": (uniq, np.asarray(a.material_numerator)), "remainder": (uniq, np.asarray(a.remainder_numerator)),
            "total": (uniq, np.asarray(a.material_numerator) + np.asarray(a.remainder_numerator))}
        faces["p06legacy_faces_correction"][name] = (uniq, np.asarray(a.correction_numerator))
    out = {"cells": cells, "faces": faces, "p07": {}}
    rows = oc.compare_to_oracle(env, out, closure["owners"], closure["paths"], CAMPAIGNS)
    # the same table for the host terms: the JAX rows must pass wherever the host rows do
    host_rows = oc.compare_to_oracle(env, {"cells": host["cells"], "faces": host["faces"], "p07": {}},
                                     closure["owners"], closure["paths"], CAMPAIGNS)
    assert [r["term"] for r in rows] == [r["term"] for r in host_rows]
    REPORT["oracle_rows"] = [(r["campaign"], r["term"], r["max_abs"], r["ratio_to_oracle_NR"], r["pass"]) for r in rows]
    failed = [(r["campaign"], r["term"], r["max_abs"]) for r in rows if not r["pass"]]
    host_failed = [(r["campaign"], r["term"]) for r in host_rows if not r["pass"]]
    REPORT["oracle_summary"] = dict(rows=len(rows), jax_failed=failed, host_failed=host_failed)
    print("\nE4 compare_to_oracle rows:", len(rows), "JAX failed:", failed, "host failed:", host_failed)
    assert [(r["campaign"], r["term"]) for r in rows if not r["pass"]] == host_failed
    assert not failed
    worst_row = max(abs(r["max_abs"] - h["max_abs"]) for r, h in zip(rows, host_rows))
    REPORT["oracle_max_abs_row_difference_vs_host"] = worst_row
    print("E4 max |max_abs(JAX row) - max_abs(host row)| over the table:", worst_row)
    print("E4 rows per campaign:", {c: sum(r["campaign"] == c for r in rows) for c in {r["campaign"] for r in rows}},
          "all pass:", all(r["pass"] for r in rows))


@needs_inputs
def test_real_plan_eager_equals_jit(closure, p06n, legacy):
    from drbx.native.fci_perpendicular_p06_operator import bc_columns, p06_action

    plan, ov = closure["plan"], p06n["fields"]
    bc = bc_columns(p06n["bc"], p06n["columns"])
    kinds, groups = p06n["kinds"], p06n["groups"]
    jitted = jax.jit(lambda p, f, b: p06_action(p, f, b, kinds, groups))(plan, jnp.asarray(ov), bc)
    for a, b in zip(p06n["action"], jitted):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
