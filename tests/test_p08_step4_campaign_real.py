"""Step-4 smoke campaign on real HSX data, bounded (slow): the campaign's preflight function and the replay-stage
code path with the **final operator options** (autodiff K, q2 P05/P06 faces, fixed_radius inner donor support) on
the N32 owner closure.

* ``test_campaign_preflight_grid_passes_at_n32``: ``campaign.preflight_grid`` (blocked path, all seven keys) with the
  pinned options: every host-vs-JAX policy row passes, no ``uniq_mismatches``; the ``compare_to_oracle`` rows are
  recorded as informational (they are expected to differ from the frozen fd / q3 / profile7 oracles).
* ``test_replay_stage_on_a_real_final_option_artifact``: the closure's rows (12 owners, rebuilt with
  ``capture_factors=True``) are written as a schema-v3 artifact (q2 geometry, ``coupled_quartic`` rows from the C3
  support); ``p08_step2_global.replay.run_replay_stage`` with ``operator_options`` = the pinned options streams the
  plan from it and evaluates the seven campaign keys; its per-owner terms must pass the design-section-8 policy
  against the host ``assemble_owner_terms``, and the smoke finiteness check (``smoke.finite_report``) must pass.

Total ~4 min. Needs the HSX N32 geometry/sidecar and the frozen campaigns' N32 oracle arrays (skipped otherwise).
"""
from __future__ import annotations

import json
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
CAMPAIGNS = ("p05", "p05n_frozen", "p05n_upwind", "p06n", "p06_legacy", "p07", "p07n")
FINAL = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius"}

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
    env = build_environment(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR, **FINAL)
    owners = np.asarray(oc.selection_fixture(env.t, env.census)["owners"], dtype=np.int64)
    built = oc.build_owner_rows(env, owners, provider=oc.load_provider_for_env(
        SIDECAR, curvature=FINAL["curvature"], face_quadrature=FINAL["face_quadrature"]))
    oracle = ru._load_oracle_owner_values(env, _paths(), CAMPAIGNS)
    host_out = oc.assemble_owner_terms(env, built, CAMPAIGNS, oracle)
    closure_ = jr.JaxOwnerClosure(env, built, CAMPAIGNS, oracle)
    closure_out = closure_.evaluate(host_only_from=host_out)
    t = env.t
    host_terms = jr.normalized_terms(host_out, vol=t.vol, owners=owners)
    closure_terms = jr.normalized_terms(closure_out, vol=t.vol, owners=owners)
    perturbed = [jr.normalized_terms(closure_.evaluate(perturb_seed=seed), vol=t.vol, owners=owners)
                 for seed in jr.FLOOR_SEEDS]
    floors = jr.conditioning_floors(closure_terms, perturbed)
    return dict(env=env, owners=owners, built=built, host_terms=host_terms, floors=floors, jr=jr)


def write_tensor_artifact(root, env, built, identity) -> dict:
    """The closure's rows rebuilt with ``capture_factors=True`` and written as a schema-v3 artifact directory with the
    final-option geometry (q2 faces, q3 P07 set). Returns the ``artifact.py`` tensor/fallback counts and the family
    counts of the point rows."""
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
        env.ctx, env.census, built["p07_row_indices"], geometry.p07_points[face_pos], geometry.p07_weight[face_pos],
        geometry.p07_face_tensor[face_pos], inner_support=FINAL["inner_support"], **kwargs)

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
def test_campaign_preflight_grid_passes_at_n32(tmp_path):
    from p08_step4_global import campaign

    _need_oracles()
    case = campaign.preflight_grid(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR, output=tmp_path, paths=_paths())
    assert case["all_pass"] is True, (case["policy_failures"], case["uniq_mismatches"], case["options_recorded"])
    assert case["operator_options"] == FINAL and case["options_recorded"] is True
    assert case["jax"]["backend"] == "cpu" and case["jax"]["x64"] is True
    cfg = campaign.config()
    assert case["blocking"] == {"column_block": cfg["column_block"], "variant_block": cfg["p06n_variant_block"],
                                "boundary_batch": cfg["boundary_batch"]}
    assert case["policy_rows"] == 29 and case["oracle_informational"]["rows"] >= 143
    assert case["oracle_informational"]["gate"] is False
    report = json.loads(Path(case["closure_report"]).read_text())
    assert (report["curvature"], report["face_quadrature"], report["inner_support"]) == \
           ("autodiff", "q2", "fixed_radius")
    print("\nstep-4 preflight N32:", {k: case[k] for k in ("policy_rows", "peak_rss_gib", "seconds")},
          "oracle rows differing from the frozen oracles:",
          len(case["oracle_informational"]["rows_not_matching_frozen_oracles"]),
          "max ratio_to_oracle_NR:", case["oracle_informational"]["max_ratio_to_oracle_NR"])


@needs_inputs
def test_replay_stage_on_a_real_final_option_artifact(closure, tmp_path):
    from p08_step2_global import replay
    from p08_step4_global import campaign, smoke

    jr, env, built, owners = closure["jr"], closure["env"], closure["built"], closure["owners"]
    identity = {"policy": __import__("p_shared.build_artifact", fromlist=["x"]).build_policy(**FINAL),
                "step4": "bounded-n32"}
    stats = write_tensor_artifact(tmp_path / "artifact", env, built, identity)
    assert stats["tensor_sources"] > 0
    smoke.check_artifact_options(identity, FINAL)
    selection = dict(raw_ids=built["raw_ids"], face_rows=built["face_row_indices"],
                     p07_rows=built["p07_row_indices"])
    cfg = {**campaign.config(), "tensor_encoding_required": False}
    result = replay.run_replay_stage(
        artifact_root=tmp_path / "artifact", output=tmp_path / "replay" / "N32", n=N, input_root=WORKSPACE,
        sidecar_path=SIDECAR, paths=_paths(), campaigns=CAMPAIGNS, campaign_identity="bounded", cfg=cfg,
        selection=selection, compare=False, return_out=True, artifact_identity=identity, operator_options=FINAL,
        log=lambda m: None)
    out = result["out"]
    assert result["plan"]["cells"] == len(built["raw_ids"]) and result["jax"]["x64"] is True

    # smoke gate: every term array is finite
    report = smoke.finite_report(out, CAMPAIGNS)
    assert report["all_finite"] is True and report["missing_campaigns"] == [], report["nonfinite_total"]
    assert report["elements_total"] > 0

    # the final-option JAX terms agree with the host closure at the roundoff / floor level (design section 8)
    terms = jr.normalized_terms(out, vol=env.t.vol, owners=owners)
    operator_terms = {name: v for name, v in closure["host_terms"].items() if jr.classify_term(name) != "host_only"}
    assert set(operator_terms) == set(terms)
    rows = jr.compare_terms(operator_terms, terms, closure["floors"])
    assert [r for r in rows if not r["pass"]] == []
    print("\nstep-4 bounded replay (final options): elements", report["elements_total"], "max abs per campaign",
          {c: f"{e['max_abs']:.2e}" for c, e in report["campaigns"].items()})
