"""G3 glue on real HSX data, bounded (slow): the campaign's own replay-stage code path on a real schema-v3
tensor artifact of the N32 owner closure, against the host terms, E6's JAX terms and the frozen oracles; the
campaign's preflight function; and the full-grid comparison function against ``replay_units.reduce_grid``.

* ``test_replay_stage_on_a_real_tensor_artifact_matches_host_and_e6``: the closure's rows (12 owners) are rebuilt
  with ``capture_factors=True`` and written as a v3 artifact directory (tensor-factored unconditioned sources,
  ``geometry.npz``, ``census.npz``, ``build_identity.json``); ``p08_step2_global.replay.run_replay_stage`` -- the
  function the remote campaign runs, with the owner set restricted through ``selection`` -- streams the plan from
  it, evaluates the seven campaign keys sequentially in column blocks with checkpoints, and its per-owner operator
  terms must pass the design-section-8 policy against the host ``assemble_owner_terms`` (and E6's JAX terms).
  A second call resumes from the checkpoints without recomputing.
* ``test_campaign_preflight_grid_passes_at_n32``: ``campaign.preflight_grid`` (blocked path, all seven keys).
* ``test_comparison_matches_reduce_grid``: ``comparison.compare_operator_terms`` equals ``replay_units.reduce_grid``
  term by term (bitwise) on synthetic full-grid unit outputs built from the saved N32 oracle arrays; for the terms
  under the step 3.0 rules (``STEP3_0_TERMS``) the arithmetic (ratios, region norms) is bitwise equal and only the
  verdict rule differs, which the test checks explicitly.

Total ~3 min. Needs the HSX N32 geometry/sidecar and the frozen campaigns' N32 oracle arrays (skipped otherwise).
"""
from __future__ import annotations

import json
import sys
import time
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
CAMPAIGNS = ("p05", "p05n_frozen", "p05n_upwind", "p06n", "p06_legacy", "p07", "p07n")

pytestmark = pytest.mark.slow

_have = (GEOMETRY / f"{N}x{N}x{N}" / "base_geometry.npz").is_file() and SIDECAR.is_file()
needs_inputs = pytest.mark.skipif(not _have, reason="HSX N32 geometry/sidecar inputs are unavailable")


def _paths():
    from p_shared.replay_support import DEFAULT_PATHS
    return dict(DEFAULT_PATHS)


def _need_oracles():
    from p_shared import owner_closure as oc
    if not oc.oracle_available(_paths(), CAMPAIGNS, n=N):
        pytest.skip("the frozen campaigns' N32 oracle arrays are unavailable")


@pytest.fixture(scope="module")
def closure():
    from p_shared import jax_replay as jr
    from p_shared import owner_closure as oc
    from p_shared import replay_units as ru
    from p_shared.replay_support import build_environment

    _need_oracles()
    env = build_environment(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR, curvature="fd", face_quadrature="q3", inner_support="profile7")
    owners = np.asarray(oc.selection_fixture(env.t, env.census)["owners"], dtype=np.int64)
    built = oc.build_owner_rows(env, owners, provider=oc.load_provider_for_env(SIDECAR, curvature="fd", face_quadrature="q3"))
    paths = _paths()
    oracle = ru._load_oracle_owner_values(env, paths, CAMPAIGNS)
    host_out = oc.assemble_owner_terms(env, built, CAMPAIGNS, oracle)
    e6 = jr.JaxOwnerClosure(env, built, CAMPAIGNS, oracle)
    e6_out = e6.evaluate(host_only_from=host_out)
    t = env.t
    host_terms = jr.normalized_terms(host_out, vol=t.vol, owners=owners)
    e6_terms = jr.normalized_terms(e6_out, vol=t.vol, owners=owners)
    perturbed = [jr.normalized_terms(e6.evaluate(perturb_seed=seed), vol=t.vol, owners=owners)
                 for seed in jr.FLOOR_SEEDS]
    floors = jr.conditioning_floors(e6_terms, perturbed)
    return dict(env=env, owners=owners, built=built, oracle=oracle, paths=paths, host_terms=host_terms,
                e6_terms=e6_terms, floors=floors, jr=jr)


def write_tensor_artifact(root, env, built, identity) -> dict:
    """The closure's rows rebuilt with ``capture_factors=True`` and written as a schema-v3 artifact directory
    (``manifest.json`` + ``rows/``, ``geometry.npz``, ``census.npz``, ``build_identity.json``). Returns the
    ``artifact.py`` tensor/fallback counts."""
    from drbx.stencils import artifact as art
    from drbx.stencils import builder as stencil_builder

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
        cells=(pack(r1),), faces=(pack(r2 + r3),),
        p07=(art.pack_integrated_rows([r.row for r in r4], entity_id=[r.entity_id for r in r4]),),
        neumann=(art.pack_neumann_rows(
            [r.row for r in neumann], entity_id=[r.entity_id for r in neumann], quad_node=[r.quad_node for r in neumann],
            request=[r.source for r in neumann], radial_degree=[r.radial_degree for r in neumann]),))
    stats: dict = {}
    grid_dir = art.save_row_artifact(root, N, identity=identity, stats=stats, **chunks,
                                     point_factors={"cells": [[r.factors for r in r1]],
                                                    "faces": [[r.factors for r in r2 + r3]]})
    geometry.save(grid_dir / "geometry.npz")
    env.census.save(grid_dir / "census.npz")
    (grid_dir / "build_identity.json").write_text(json.dumps(identity))
    return stats


@needs_inputs
def test_replay_stage_on_a_real_tensor_artifact_matches_host_and_e6(closure, tmp_path):
    from p08_step2_global import campaign, replay

    jr, env, built, owners = closure["jr"], closure["env"], closure["built"], closure["owners"]
    identity = {"g3": "bounded-n32"}
    stats = write_tensor_artifact(tmp_path / "artifact", env, built, identity)
    assert stats["tensor_sources"] > 0                                    # really the production tensor encoding
    selection = dict(raw_ids=built["raw_ids"], face_rows=built["face_row_indices"],
                     p07_rows=built["p07_row_indices"])
    cfg = {**campaign.config(), "tensor_encoding_required": False}
    kwargs = dict(artifact_root=tmp_path / "artifact", output=tmp_path / "replay" / "N32", n=N, input_root=WORKSPACE,
                  sidecar_path=SIDECAR, paths=closure["paths"], campaigns=CAMPAIGNS, campaign_identity="bounded",
                  cfg=cfg, selection=selection, compare=False, return_out=True, artifact_identity=identity,
                  log=lambda m: None)
    started = time.time()
    result = replay.run_replay_stage(**kwargs)
    first = time.time() - started
    out = result["out"]
    assert set(result["campaigns"]) == set(CAMPAIGNS) and not any(r["resumed"] for r in result["campaigns"].values())
    assert result["plan"]["cells"] == len(built["raw_ids"]) and result["plan"]["plan_bytes"] > 0
    assert result["jax"]["backend"] == "cpu" and result["jax"]["x64"] is True
    assert result["artifact"]["groups"]["faces"]["bytes"] > 0

    terms = jr.normalized_terms(out, vol=env.t.vol, owners=owners)
    # no host-only MMS reference terms are produced ...
    assert not [name for name in terms if jr.classify_term(name) == "host_only"]
    assert "cells.p05n_frozen_raw_R" not in terms and "p07.p07n_global_O_q3" not in terms
    # ... every other term of the host closure is, and passes the uniform policy of design section 8
    operator_terms = {name: v for name, v in closure["host_terms"].items() if jr.classify_term(name) != "host_only"}
    assert set(operator_terms) == set(terms) and len(terms) == 23
    rows = jr.compare_terms(operator_terms, terms, closure["floors"])
    failed = [r for r in rows if not r["pass"]]
    assert failed == [], failed
    for r in rows:
        if r["kind"] == "cancellation":
            assert r["floor"] and r["floor"] > 0
    # ... and agrees with E6's JAX terms (CSR plan from the in-memory rows, unblocked) at the roundoff level
    e6_rows = jr.compare_terms({n: v for n, v in closure["e6_terms"].items() if n in terms}, terms, closure["floors"])
    assert [r for r in e6_rows if not r["pass"]] == []
    print("\nG3 bounded end-to-end (tensor artifact, blocked replay) vs host:",
          {r["term"]: (f"{r['max_rel_to_scale']:.2e}", None if r['over_floor'] is None else round(r['over_floor'], 2))
           for r in sorted(rows, key=lambda r: -r["max_rel_to_scale"])[:6]}, f"stage {first:.1f}s")

    # the frozen-oracle comparison at the closure owners with these terms (host MMS references taken from the host)
    from p_shared import owner_closure as oc
    host_only = {"cells": {k: v for k, v in closure_host_out(closure).get("cells", {}).items()
                           if k.split(".")[0] in jr._HOST_ONLY_ROOTS or k.startswith("p06n_raw_R_")},
                 "p07": {k: v for k, v in closure_host_out(closure).get("p07", {}).items()
                         if k in jr._HOST_ONLY_ROOTS}}
    merged = {"cells": {**out["cells"], **host_only["cells"]}, "faces": out["faces"],
              "p07": {**out["p07"], **host_only["p07"]}}
    oracle_rows = oc.compare_to_oracle(env, merged, owners, closure["paths"], CAMPAIGNS)
    assert all(r["pass"] for r in oracle_rows) and len(oracle_rows) > 100

    # resume: every campaign checkpoint is valid, nothing is recomputed (and the plan is not lowered)
    started = time.time()
    again = replay.run_replay_stage(**kwargs)
    assert all(r["resumed"] for r in again["campaigns"].values()) and again["plan"] is None
    terms_again = jr.normalized_terms(again["out"], vol=env.t.vol, owners=owners)
    for name, arrays in terms.items():
        for a, b in zip(arrays, terms_again[name]):
            np.testing.assert_array_equal(a, b)                           # the checkpoints round-trip bitwise
    print("resume:", f"{time.time() - started:.1f}s")


def closure_host_out(closure) -> dict:
    """The host ``assemble_owner_terms`` output (only its host-only MMS references are used, as in E6)."""
    from p_shared import owner_closure as oc
    if "_host_out" not in closure:
        closure["_host_out"] = oc.assemble_owner_terms(closure["env"], closure["built"], CAMPAIGNS, closure["oracle"])
    return closure["_host_out"]


@needs_inputs
def test_campaign_preflight_grid_passes_at_n32(tmp_path):
    from p08_step2_global import campaign

    _need_oracles()
    case = campaign.preflight_grid(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR, output=tmp_path, paths=_paths())
    assert case["all_pass"] is True, (case["policy_failures"], case["oracle_failures"], case["uniq_mismatches"])
    assert case["jax"]["backend"] == "cpu" and case["jax"]["x64"] is True
    assert case["blocking"] == {"column_block": campaign.config()["column_block"],
                                "variant_block": campaign.config()["p06n_variant_block"],
                                "boundary_batch": campaign.config()["boundary_batch"]}
    assert case["policy_rows"] == 29 and case["oracle_rows"] >= 143
    assert Path(case["closure_report"]).is_file()
    print("\nG3 preflight N32:", {k: case[k] for k in ("policy_rows", "oracle_rows", "peak_rss_gib", "seconds")})


# ---------------------------------------------------------------------------
# comparison.compare_operator_terms == replay_units.reduce_grid on synthetic full-grid unit outputs
# ---------------------------------------------------------------------------
def _synthetic_units(env, paths, rng):
    """Full-grid unit outputs (one cells / faces / p07 unit, sparse pairs over every owner) whose owner-space
    values are the saved oracle arrays with a 1e-9 relative perturbation, R terms included (for reduce_grid)."""
    from p_shared.replay_support import _load_p05_upwind
    import p06_structured_global.numerics as p06numerics

    t, n = env.t, env.n
    owners = len(t.vol)
    full = np.arange(owners, dtype=np.int64)
    vol = np.asarray(t.vol)
    ev = 1.0 + rng.uniform(size=owners)

    def pair(array, weight):
        array = np.asarray(array, dtype=np.float64)
        w = np.asarray(weight, dtype=np.float64).reshape((-1,) + (1,) * (array.ndim - 1))
        return full, array * w * (1.0 + 1e-9 * rng.normal(size=array.shape))

    cells = {"q1_evolution_volume": (full, ev)}
    faces: dict = {}
    p07: dict = {}
    with np.load(paths["p05"] / f"N{n}.owner_results.npz", allow_pickle=False) as z:
        cells["p05_centered"] = pair(z["centered"], vol)
    cells["p05_antisymmetry_max"] = 3e-15
    with np.load(paths["p05"] / "reuse_inputs" / f"N{n}.reuse.npz", allow_pickle=False) as z:
        faces["p05_live_jump_owner_num"] = pair(z["old_U_minus_A"], vol)
    upwind = _load_p05_upwind(paths["p05_upwind_chunks"], n)
    faces["p05_live_jump_p07ids"] = np.arange(len(upwind), dtype=np.int64)
    faces["p05_live_jump_values"] = upwind * (1.0 + 1e-9 * rng.normal(size=upwind.shape))
    for name, root in (("p05n_frozen", paths["p05n_frozen"]), ("p05n_upwind", paths["p05n_p06n_upwind"] / "p05n_upwind")):
        with np.load(root / f"N{n}.raw.npz", allow_pickle=False) as z:
            for suf in ("N", "D", "R"):
                cells[f"{name}_raw_{suf}"] = pair(z[suf], vol)
        with np.load(root / f"N{n}.faces.npz", allow_pickle=False) as z:
            for suf in ("N", "D"):
                faces[f"{name}_face_{suf}"] = pair(z[suf], vol)
    root = paths["p05n_p06n_upwind"] / "p06n"
    with np.load(root / f"N{n}.raw.npz", allow_pickle=False) as z:
        for label in ("material", "remainder", "total", "R_material", "R_remainder", "R_total"):
            cells[f"p06n_raw_{label}"] = [pair(z[label][v], ev) for v in range(z[label].shape[0])]
    with np.load(root / f"N{n}.faces.npz", allow_pickle=False) as z:
        faces["p06n_faces_correction"] = [pair(z["correction"][v], np.ones(owners)) for v in range(z["correction"].shape[0])]
    with np.load(paths["p06_legacy"] / f"N{n}.npz", allow_pickle=False) as z:
        saved = {k: z[k].copy() for k in z.files if k != "metadata_json"}
    cells["p06legacy_raw_centered"] = {}
    faces["p06legacy_faces_correction"] = {}
    for f in p06numerics.FIELD_NAMES:
        cells["p06legacy_raw_centered"][f] = {term: pair(saved[f"candidate:centered:{f}:{term}"], ev)
                                              for term in ("material", "remainder", "total")}
        faces["p06legacy_faces_correction"][f] = pair(saved[f"candidate:U:{f}:material"]
                                                      - saved[f"candidate:centered:{f}:material"], ev)
    with np.load(paths["p07"] / f"N{n}.global.npz", allow_pickle=False) as z:
        p07["p07_global_N"] = pair(z["action"], vol)
    with np.load(paths["p07n"] / f"N{n}.global.npz", allow_pickle=False) as z:
        p07["p07n_global_N"] = pair(z["N"], vol)
        p07["p07n_global_D"] = pair(z["D"], vol)
        p07["p07n_global_O_q3"] = pair(z["O_q3"], vol)
    return {"cells": cells, "faces": faces, "p07": p07}


#: terms whose comparison rule is the step 3.0 one in ``comparison`` (``reduce_grid`` keeps the literal step-1 rule)
STEP3_0_TERMS = ({("p05", "live_jump_vs_upwind"), ("p06n", "raw_material"), ("p06n", "raw_remainder"),
                  ("p06n", "raw_total")}
                 | {(c, t) for c in ("p05n_frozen", "p05n_upwind") for t in ("face_N", "face_D")})
_VERDICT_KEYS = {"pointwise", "pass", "kind", "tier_b_mode", "roundoff_floor", "worst_ratio", "worst_region",
                 "total_pointwise_violations", "roundoff_floor_variants", "worst_roundoff_floor_margin",
                 "expected_roundoff_control", "roundoff_classification_mismatch"}


def _numbers_only(obj):
    """Drop the verdict fields (rule-dependent) and keep the arithmetic (ratios, region diff/archived norms)."""
    if isinstance(obj, dict):
        return {k: _numbers_only(v) for k, v in obj.items() if k not in _VERDICT_KEYS}
    if isinstance(obj, list):
        return [_numbers_only(v) for v in obj]
    return obj


def _assert_step3_0_rule(name, term, ours, reference):
    if name == "p06n":
        for v_ours, v_ref in zip(ours["variants"], reference["variants"]):
            assert v_ref["tier_b_mode"] == "ratio"
            assert v_ours["tier_b_mode"] == ("roundoff_floor" if v_ours["expected_roundoff_control"] else "ratio")
            assert not v_ours.get("roundoff_classification_mismatch")
        assert sum(v["tier_b_mode"] == "roundoff_floor" for v in ours["variants"]) == 4      # the four controls
    else:
        assert ours["pointwise"]["mode"] == "floor" and ours["pointwise"]["floor_source"] == "model"
        assert reference["pointwise"]["mode"] == "scaled"


@needs_inputs
def test_comparison_matches_reduce_grid(tmp_path):
    from p08_step2_global import comparison
    from p_shared import replay_units as ru
    from p_shared import runner
    from p_shared.replay_support import build_environment

    _need_oracles()
    paths = _paths()
    env = build_environment(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR, curvature="fd", face_quadrature="q3", inner_support="profile7")
    units_out = _synthetic_units(env, paths, np.random.default_rng(7))

    output = tmp_path / "reduce"
    plan = {}
    for stage, section in (("cells", "cells"), ("faces", "faces"), ("p07", "p07")):
        unit = {"stage": stage, "n": N, "start": 0, "stop": 1}
        runner.write_unit(output, unit, {"synthetic": True},
                          chunks={"chunk": ru._pack_unit_arrays(units_out[section])}, started=time.time())
        plan[stage] = [unit]
    reference = ru.reduce_grid(output=output, artifact_root=tmp_path, n=N, input_root=WORKSPACE, sidecar_path=SIDECAR,
                               paths=paths, campaigns=CAMPAIGNS, plan=plan)

    host_only_keys = {"p05n_frozen_raw_R", "p05n_upwind_raw_R", "p06n_raw_R_material", "p06n_raw_R_remainder",
                      "p06n_raw_R_total"}
    ours_in = {"cells": {k: v for k, v in units_out["cells"].items() if k not in host_only_keys},
               "faces": units_out["faces"],
               "p07": {k: v for k, v in units_out["p07"].items() if k != "p07n_global_O_q3"}}
    ours = comparison.compare_operator_terms(env=env, out=ours_in, paths=paths, campaigns=CAMPAIGNS, n=N)

    def norm(obj):
        return json.dumps(obj, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))

    assert list(ours) == list(reference["campaigns"])
    omitted = {name: set(terms) for name, terms in comparison.OMITTED_TERMS.items()}
    compared = 0
    for name, result in ours.items():
        ref_terms = reference["campaigns"][name]["terms"]
        assert set(result["terms"]) == {t for t in ref_terms if t not in omitted.get(name, set())}, name
        for term, value in result["terms"].items():
            if (name, term) in STEP3_0_TERMS:
                # the step 3.0 rules change the verdict rules only: same arithmetic (ratios, regions, diff norms)
                assert norm(_numbers_only(value)) == norm(_numbers_only(ref_terms[term])), (name, term)
                _assert_step3_0_rule(name, term, value, ref_terms[term])
            else:
                assert norm(value) == norm(ref_terms[term]), (name, term)      # bitwise: same ratios, cap, violations
            compared += 1
        if name == "p05":
            assert result["antisymmetry_max"] == reference["campaigns"]["p05"]["antisymmetry_max"]
    assert compared == 3 + 2 * 4 + 4 + 24 + 1 + 2                            # p05, 2x p05n, p06n, legacy, p07, p07n
    print("\ncomparison == reduce_grid on", compared, "terms")
