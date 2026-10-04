"""Fast tests of the P08 step-4 campaign (``scripts/p08_step4_global``): pinned final-operator options in the
configuration and identity, CLI, preflight gating (the oracle rows never gate), the build / replay stage wiring
(options passed explicitly), the smoke finiteness check, the informational comparison with the frozen oracles
(exceptions recorded, never raised), the smoke stage's outputs / resume, ``validate`` and the step-2 default-options
guarantee of ``run_replay_stage``.

Fully synthetic / ``tmp_path`` based: no real geometry, row artifact or oracle data (the real bounded check is
``tests/test_p08_step4_campaign_real.py`` (removed 4 October 2026: it needed the oracle arrays retired on 2 October), slow).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
sys.path.insert(0, str(REPO / "scripts"))

from p08_step2_global import campaign as step2, comparison, replay  # noqa: E402
from p08_step4_global import campaign, smoke                         # noqa: E402
from p_shared import build_artifact as ba                            # noqa: E402
from p_shared import jax_replay as jr                                # noqa: E402
from p_shared import oracle_manifest as om                           # noqa: E402
from p_shared import replay_units as ru                              # noqa: E402

FINAL = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius"}
OLD = {"curvature": "fd", "face_quadrature": "q3", "inner_support": "profile7"}


def _fake_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "_input_manifest", lambda: {"files": []})
    monkeypatch.setattr(campaign, "committed_oracle_manifest",
                        lambda: {"schema": om.MANIFEST_SCHEMA, "workspace_root": "", "campaigns": {}})
    workspace = tmp_path / "workspace"
    sidecar = workspace / campaign.config()["canonical_sidecar_relative"]
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"schema": "drbx.hsx-continuous-mms-reference", "metric_cache": {"path": "x"},
                                   "makegrid": {"path": "x"}, "artifact": {"path": "x"}}))
    return workspace


def _argv(command, workspace, output, *extra):
    return [command, "--input-root", str(workspace), "--output", str(output), *extra]


# ---------------------------------------------------------------------------
# Configuration, identity, CLI
# ---------------------------------------------------------------------------
def test_config_pins_the_three_final_options_and_inherits_the_rest():
    cfg = campaign.config()
    assert cfg["operator_options"] == FINAL == campaign.OPERATOR_OPTIONS == campaign.operator_options()
    assert cfg["schema"] == "drbx.p08-step4-smoke-campaign-v1"
    assert cfg["resolutions"] == [32, 48, 64] == cfg["allowed_resolutions"]
    assert cfg["oracle_comparison_is_gate"] is False and cfg["acceptance_gate"] == "none"
    for key in ("campaigns", "column_block", "p06n_variant_block", "boundary_batch", "wall_cache",
                "tensor_encoding_required", "omitted_host_only_terms", "cell_chunk_size", "face_chunk_size",
                "p07_chunk_size", "oracle_default_paths", "tier_b_ratio_tolerance"):
        assert cfg[key] == step2.config()[key]                       # inherited, not restated
    assert "operator_options" not in step2.config()


def test_config_refuses_a_configuration_that_drifts_from_the_pinned_bundle(monkeypatch):
    monkeypatch.setattr(campaign, "OPERATOR_OPTIONS", {**FINAL, "face_quadrature": "q3"})
    with pytest.raises(ValueError, match="differ from the pinned final bundle"):
        campaign.config()


def test_source_hashes_cover_step2_this_package_and_the_option_modules():
    hashes = campaign.source_hashes()
    assert len(hashes) == len(campaign.SOURCE_FILES) == len(set(campaign.SOURCE_FILES))
    for rel, digest in hashes.items():
        assert (REPO / rel).is_file(), rel
        assert len(digest) == 64
    for needed in (*step2.SOURCE_FILES, "scripts/p08_step4_global/campaign.py", "scripts/p08_step4_global/smoke.py",
                   "scripts/p08_step4_global/configuration.json", "scripts/p_shared/curvature_reference.py",
                   "scripts/p_shared/face_quadrature.py", "scripts/p_shared/inner_support.py",
                   "src/drbx/geometry/curvature_autodiff.py", "src/drbx/geometry/fci_perpendicular_reconstruction.py",
                   "src/drbx/geometry/fci_perpendicular_integrated_rows.py", "src/drbx/stencils/geometry_arrays.py",
                   "src/drbx/stencils/builder.py"):
        assert needed in hashes


def test_identity_includes_the_options_and_differs_from_step2(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    one = campaign.verify(input_root=workspace, output=tmp_path / "one", oracle_root=None)
    assert campaign.verify(input_root=workspace, output=tmp_path / "one", oracle_root=None) == one
    saved = json.loads((tmp_path / "one" / "campaign_manifest.json").read_text())
    assert saved["campaign"] == "p08_step4_global" and saved["operator_options"] == FINAL
    assert saved["configuration"]["operator_options"] == FINAL and saved["jax_backend"] == "cpu"
    # a different option (any one of the three) is a different campaign identity ...
    real_config = campaign.config
    for key, other in (("curvature", "fd"), ("face_quadrature", "q3"), ("inner_support", "profile7")):
        monkeypatch.setattr(campaign, "config",
                            lambda key=key, other=other: {**real_config(),
                                                          "operator_options": {**FINAL, key: other}})
        two = campaign.verify(input_root=workspace, output=tmp_path / f"two_{key}", oracle_root=None)
        assert two != one, key
        with pytest.raises(ValueError, match="campaign identity changed"):    # ... and the folder is not reusable
            campaign.verify(input_root=workspace, output=tmp_path / "one", oracle_root=None)
    monkeypatch.setattr(campaign, "config", real_config)
    # step 2 over the same inputs has its own identity
    monkeypatch.setattr(step2, "_input_manifest", lambda: {"files": []})
    monkeypatch.setattr(step2, "committed_oracle_manifest",
                        lambda: {"schema": om.MANIFEST_SCHEMA, "workspace_root": "", "campaigns": {}})
    assert step2.verify(input_root=workspace, output=tmp_path / "s2", oracle_root=None) != one


def test_verify_refuses_other_oracle_root_and_non_cpu_backend(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    campaign.verify(input_root=workspace, output=tmp_path / "o", oracle_root=None)
    with pytest.raises(ValueError, match="oracle root changed"):
        campaign.verify(input_root=workspace, output=tmp_path / "o", oracle_root=tmp_path / "elsewhere")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with pytest.raises(ValueError, match="CPU backend required"):
        campaign.verify(input_root=workspace, output=tmp_path / "o", oracle_root=None)


def test_cli_parsing():
    a = campaign.parse(["run", "--input-root", "/x", "--output", "/y"])
    assert a.resolutions == [32, 48, 64] and a.workers == 4 and a.oracle_root is None
    a = campaign.parse(["preflight", "--input-root", "/x", "--output", "/y", "--resolutions", "32", "--workers", "96",
                        "--memory-budget-gib", "400", "--worker-memory-gib", "3.5", "--memory-reserve-gib", "8",
                        "--max-tasks-per-worker", "50", "--oracle-root", "/o"])
    assert a.resolutions == [32] and a.workers == 96 and a.memory_budget_gib == 400.0
    assert (a.worker_memory_gib, a.memory_reserve_gib, a.max_tasks_per_worker) == (3.5, 8.0, 50)
    assert a.oracle_root == Path("/o")
    a = campaign.parse(["run-stage", "--input-root", "/x", "--output", "/y", "--stage", "replay", "--n", "64"])
    assert (a.stage, a.n) == ("replay", 64)
    for bad in (["run", "--resolutions", "16"], ["frobnicate"], ["run-stage", "--stage", "cells"],
                ["run", "--max-units", "2"]):
        with pytest.raises(SystemExit):
            campaign.parse([*bad, "--input-root", "/x", "--output", "/y"])


def test_resolutions_and_run_stage_arguments_are_checked(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="--stage and --n are required"):
        campaign.main(_argv("run-stage", workspace, tmp_path / "out"))
    with pytest.raises(ValueError, match="unique and ascending"):
        campaign.main(_argv("verify-inputs", workspace, tmp_path / "out", "--resolutions", "48", "32"))


def test_verify_inputs_command_locks_and_records(tmp_path, monkeypatch, capsys):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    campaign.main(_argv("verify-inputs", workspace, output))
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "verify-inputs" and payload["status"] == "complete"
    assert (output / ".campaign.lock").exists() and list((output / "invocations").glob("*_verify-inputs.json"))
    assert json.loads((output / "last_exit.json").read_text())["status"] == "complete"


# ---------------------------------------------------------------------------
# Preflight gating
# ---------------------------------------------------------------------------
def _closure_payload(**over):
    payload = {"all_diff_pass": True, "oracle_jax_all_pass": False, "uniq_mismatches": [],
               "curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius",
               "diff_table": [{}] * 29, "diff_failures": [],
               "oracle_jax": [{"campaign": "p05", "term": "centered", "pass": False, "ratio_to_oracle_NR": 0.7},
                              {"campaign": "p07", "term": "global_N", "pass": True, "ratio_to_oracle_NR": 0.01}],
               "plan": {"cells": 1}, "blocking": {"column_block": 8}, "peak_rss_gib": 1.5, "wall_seconds": 2.0}
    return {**payload, **over}


def _preflight(monkeypatch, tmp_path, **over):
    seen = {}

    def fake_check(**kwargs):
        seen.update(kwargs)
        return _closure_payload(**over)

    monkeypatch.setattr(jr, "run_jax_owner_closure_check", fake_check)
    case = campaign.preflight_grid(n=32, input_root=tmp_path, sidecar_path=tmp_path / "s.json", output=tmp_path,
                                   paths={"p05": tmp_path})
    return case, seen


def test_preflight_grid_passes_the_final_options_explicitly_and_oracle_rows_do_not_gate(monkeypatch, tmp_path):
    case, seen = _preflight(monkeypatch, tmp_path)
    cfg = campaign.config()
    assert (seen["curvature"], seen["face_quadrature"], seen["inner_support"]) == ("autodiff", "q2", "fixed_radius")
    assert (seen["column_block"], seen["variant_block"], seen["boundary_batch"]) == \
           (cfg["column_block"], cfg["p06n_variant_block"], cfg["boundary_batch"])
    assert seen["campaigns"] == tuple(cfg["campaigns"]) and seen["n"] == 32
    assert case["all_pass"] is True and case["options_recorded"] is True and case["operator_options"] == FINAL
    info = case["oracle_informational"]                      # the oracle rows fail but are informational only
    assert info["gate"] is False and info["rows_not_matching_frozen_oracles"] == [("p05", "centered")]
    assert info["max_ratio_to_oracle_NR"] == 0.7 and case["policy_rows"] == 29
    assert case["jax"]["backend"] == "cpu" and case["jax"]["x64"] is True


@pytest.mark.parametrize("over", [{"all_diff_pass": False, "diff_failures": ["cells.p05_centered"]},
                                  {"uniq_mismatches": [("cells", "p05_centered")]},
                                  {"face_quadrature": "q3"},               # payload does not record q2
                                  {"curvature": "fd"}])
def test_preflight_grid_gates_on_policy_rows_structure_and_recorded_options(monkeypatch, tmp_path, over):
    case, _ = _preflight(monkeypatch, tmp_path, **over)
    assert case["all_pass"] is False


def test_preflight_grid_requires_cpu_and_x64(monkeypatch, tmp_path):
    monkeypatch.setattr(replay, "jax_info", lambda: {"backend": "gpu", "x64": True})
    with pytest.raises(ValueError, match="CPU backend required"):
        campaign.preflight_grid(n=32, input_root=tmp_path, sidecar_path=tmp_path / "s.json", output=tmp_path, paths={})
    monkeypatch.setattr(replay, "jax_info", lambda: {"backend": "cpu", "x64": False})
    with pytest.raises(ValueError, match="x64 required"):
        campaign.preflight_grid(n=32, input_root=tmp_path, sidecar_path=tmp_path / "s.json", output=tmp_path, paths={})


def _fake_case(n, ok=True):
    return {"n": n, "all_pass": ok, "policy_failures": [] if ok else ["cells.p05_centered"]}


def test_run_and_validate_refuse_without_a_matching_preflight(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="run requires a matching preflight"):
        campaign.main(_argv("run", workspace, output, "--resolutions", "32", "--workers", "1"))
    with pytest.raises(ValueError, match="a matching preflight is required before validate"):
        campaign.main(_argv("validate", workspace, output, "--resolutions", "32"))
    campaign.write(output / "preflight.json", {"identity": "someone-else", "cases": {"32": _fake_case(32)},
                                               "all_pass": True})
    with pytest.raises(ValueError, match="run requires a matching preflight"):
        campaign.main(_argv("run", workspace, output, "--resolutions", "32"))


def test_run_refuses_failed_or_missing_grid_preflight(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    identity = campaign.verify(input_root=workspace, output=output, oracle_root=None)
    campaign.write(output / "preflight.json", {"identity": identity, "all_pass": False,
                                               "cases": {"32": _fake_case(32), "48": _fake_case(48, ok=False)}})
    with pytest.raises(ValueError, match="preflight did not pass; refusing to run"):
        campaign.main(_argv("run", workspace, output, "--resolutions", "32", "48"))
    with pytest.raises(ValueError, match=r"no case for N\[64\]"):
        campaign.main(_argv("run", workspace, output, "--resolutions", "32", "64"))
    monkeypatch.setattr(campaign, "build_artifact_stage", lambda **kw: (_ for _ in ()).throw(RuntimeError("build")))
    with pytest.raises(RuntimeError, match="build"):           # N32 alone passes the gate (and fails at the build)
        campaign.main(_argv("run", workspace, output, "--resolutions", "32"))


def test_preflight_command_is_resumable_and_merges_grids(tmp_path, monkeypatch, capsys):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    calls = []

    def fake_preflight_grid(*, n, **kw):
        calls.append(n)
        return _fake_case(n, ok=(n != 48 or calls.count(48) > 1))

    monkeypatch.setattr(campaign, "preflight_grid", fake_preflight_grid)
    campaign.main(_argv("preflight", workspace, output, "--resolutions", "32", "48"))
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["all_pass"] is False and calls == [32, 48]
    campaign.main(_argv("preflight", workspace, output, "--resolutions", "32", "48"))
    assert calls == [32, 48, 48]
    assert json.loads((output / "preflight.json").read_text())["all_pass"] is True


# ---------------------------------------------------------------------------
# Stages: the build and the replay get the pinned options explicitly
# ---------------------------------------------------------------------------
def _policy_identity(options=FINAL):
    return {"policy": ba.build_policy(options["curvature"], options["face_quadrature"], options["inner_support"])}


def test_build_stage_passes_the_pinned_options_and_step1_chunk_sizes(tmp_path, monkeypatch):
    seen = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        return {"identity": _policy_identity(), "wall_seconds": 1.0}

    monkeypatch.setattr(ba, "run_full_build", fake_build)
    args = campaign.parse(["run", "--input-root", str(tmp_path), "--output", str(tmp_path / "o"), "--workers", "96",
                           "--memory-budget-gib", "400", "--worker-memory-gib", "3.5"])
    receipt = campaign.build_artifact_stage(n=48, args=args, sidecar_path=tmp_path / "s.json",
                                            artifact_root=tmp_path / "artifact")
    cfg = campaign.config()
    assert receipt["wall_seconds"] == 1.0
    assert (seen["curvature"], seen["face_quadrature"], seen["inner_support"]) == ("autodiff", "q2", "fixed_radius")
    assert seen["n"] == 48 and seen["workers"] == 96 and "max_units" not in seen
    for key in ("cell_chunk_size", "face_chunk_size", "p07_chunk_size", "geometry_raw_chunk_size",
                "geometry_face_chunk_size"):
        assert seen[key] == cfg[key] == campaign.step1.config()[key]
    # the build identity must record the options: a receipt of a default / other build is refused
    monkeypatch.setattr(ba, "run_full_build", lambda **kw: {"identity": _policy_identity(OLD)})
    with pytest.raises(ValueError, match="not the pinned"):
        campaign.build_artifact_stage(n=48, args=args, sidecar_path=tmp_path / "s.json", artifact_root=tmp_path / "a")


def test_build_policy_records_the_final_options():
    policy = ba.build_policy("autodiff", "q2", "fixed_radius")
    assert policy["curvature"] == "autodiff" and policy["quadrature"] == {"raw": "q1", "face": "q2", "p07_face": "q3"}
    assert policy["inner_support"] == "fixed_radius"
    smoke.check_artifact_options({"policy": policy}, FINAL)
    for bad_options in ({**FINAL, "curvature": "fd"}, {**FINAL, "face_quadrature": "q3"},
                        {**FINAL, "inner_support": "profile7"}):
        with pytest.raises(ValueError, match="not the pinned"):
            smoke.check_artifact_options(_policy_identity(bad_options), FINAL)


def test_csr_only_build_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("P_SHARED_CSR_ONLY", "1")
    args = campaign.parse(["run", "--input-root", str(tmp_path), "--output", str(tmp_path)])
    with pytest.raises(ValueError, match="P_SHARED_CSR_ONLY"):
        campaign.build_artifact_stage(n=32, args=args, sidecar_path=tmp_path, artifact_root=tmp_path)


def test_replay_stage_passes_the_pinned_options_to_the_smoke_stage(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(smoke, "run_smoke_stage", lambda **kw: seen.update(kw) or {"smoke_pass": True})
    args = campaign.parse(["run", "--input-root", str(tmp_path), "--output", str(tmp_path / "o")])
    campaign.replay_stage(n=32, args=args, identity="abc", sidecar_path=tmp_path / "s.json", paths={"p05": tmp_path},
                          artifact_root=tmp_path / "o" / "artifact")
    assert seen["operator_options"] == FINAL and seen["campaign_identity"] == "abc" and seen["n"] == 32
    assert seen["output"] == tmp_path / "o" / "replay" / "N32"
    assert seen["campaigns"] == tuple(campaign.config()["campaigns"])


def test_run_builds_then_replays_each_grid_then_validates(tmp_path, monkeypatch, capsys):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    identity = campaign.verify(input_root=workspace, output=output, oracle_root=None)
    campaign.write(output / "preflight.json", {"identity": identity, "all_pass": True,
                                               "cases": {str(n): _fake_case(n) for n in (32, 48, 64)}})
    log = []
    monkeypatch.setattr(campaign, "build_artifact_stage",
                        lambda *, n, args, sidecar_path, artifact_root: log.append(("build", n)) or {})

    def fake_replay(*, n, args, identity, sidecar_path, paths, artifact_root):
        log.append(("replay", n))
        _write_smoke(args.output, n, identity)
        return {"smoke_pass": True, "wall_seconds": 3.0}

    monkeypatch.setattr(campaign, "replay_stage", fake_replay)
    campaign.main(_argv("run", workspace, output))                    # default resolutions 32 48 64
    assert log == [("build", 32), ("replay", 32), ("build", 48), ("replay", 48), ("build", 64), ("replay", 64)]
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["operational_complete"] is True and summary["smoke_pass"] is True
    validation = json.loads((output / "validation.json").read_text())
    assert validation["resolutions"] == [32, 48, 64] and validation["operator_options"] == FINAL
    assert validation["acceptance_gate"] is None and validation["scope"]["oracle_comparison_is_gate"] is False
    assert (output / "summary" / "step4_summary.json").is_file()


def test_run_stage_dispatch(tmp_path, monkeypatch, capsys):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    calls = []
    monkeypatch.setattr(campaign, "build_artifact_stage",
                        lambda **kw: calls.append(("artifact", kw["n"])) or {"wall_seconds": 2.0})
    monkeypatch.setattr(campaign, "replay_stage",
                        lambda **kw: calls.append(("replay", kw["n"])) or {"wall_seconds": 3.0, "smoke_pass": True})
    campaign.main(_argv("run-stage", workspace, output, "--stage", "artifact", "--n", "48"))
    campaign.main(_argv("run-stage", workspace, output, "--stage", "replay", "--n", "64"))
    assert calls == [("artifact", 48), ("replay", 64)]
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["stage"] == "replay"


# ---------------------------------------------------------------------------
# The smoke check
# ---------------------------------------------------------------------------
CAMPAIGNS = tuple(campaign.config()["campaigns"])


def _good_out(seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.array([1, 4, 6], dtype=np.int64)
    return {"cells": {"q1_evolution_volume": (uniq, rng.uniform(1, 2, size=3)),
                      "p05_centered": (uniq, rng.normal(size=(3, 8))), "p05_antisymmetry_max": 1e-15,
                      "p05n_frozen_raw_N": (uniq, rng.normal(size=(3, 4))),
                      "p05n_upwind_raw_N": (uniq, rng.normal(size=(3, 4))),
                      "p06n_raw_material": [(uniq, rng.normal(size=(3, 4))), (uniq, rng.normal(size=(3, 4)))],
                      "p06legacy_raw_centered": {"a": {"total": (uniq, rng.normal(size=(3, 4)))}}},
            "faces": {"p05_live_jump_p07ids": np.array([3, 9], dtype=np.int64),
                      "p05_live_jump_values": rng.normal(size=(2, 8)),
                      "p05n_frozen_face_N": (uniq, rng.normal(size=(3, 4)))},
            "p07": {"p07_global_N": (uniq, rng.normal(size=(3, 4))), "p07n_global_N": (uniq, rng.normal(size=(3, 4)))}}


def test_campaign_of_uses_the_longest_prefix():
    assert [smoke.campaign_of(k) for k in ("p05_centered", "p05n_frozen_raw_N", "p05n_upwind_face_D", "p06n_raw_total",
                                           "p06legacy_raw_centered", "p07_global_N", "p07n_global_N",
                                           "p05_live_jump_values", "q1_evolution_volume", "p05_antisymmetry_max")] == \
           ["p05", "p05n_frozen", "p05n_upwind", "p06n", "p06_legacy", "p07", "p07n", "p05", None, "p05"]


def test_finite_report_passes_a_finite_out_and_reports_counts_and_maxima():
    out = _good_out()
    report = smoke.finite_report(out, CAMPAIGNS)
    assert report["all_finite"] is True and report["nonfinite_total"] == 0 and report["missing_campaigns"] == []
    assert set(report["campaigns"]) == set(CAMPAIGNS)
    term = report["campaigns"]["p05"]["terms"]["cells.p05_centered"]
    assert term["elements"] == 24 and term["nonfinite"] == 0
    assert term["max_abs"] == pytest.approx(float(np.max(np.abs(out["cells"]["p05_centered"][1]))))
    assert "faces.p05_live_jump_values" in report["campaigns"]["p05"]["terms"]
    assert "faces.p05_live_jump_p07ids" in report["campaigns"]["p05"]["terms"]     # ids: counted as 0 float arrays
    assert report["campaigns"]["p05"]["terms"]["faces.p05_live_jump_p07ids"]["elements"] == 0
    assert report["campaigns"]["p06n"]["terms"]["cells.p06n_raw_material"]["arrays"] == 2     # per-variant arrays
    assert report["campaigns"]["p06_legacy"]["terms"]["cells.p06legacy_raw_centered"]["elements"] == 12
    assert "cells.q1_evolution_volume" in report["shared"]


@pytest.mark.parametrize("where, bad, nan, inf", [
    (("cells", "p05_centered", 1), (2, 3), 1, 0),                       # NaN in a numerator
    (("faces", "p05_live_jump_values"), (0, 0), 0, 1),                  # +Inf in a per-face value
    (("p07", "p07_global_N", 1), (1, 1), 0, 1),
])
def test_finite_report_flags_nan_and_inf(where, bad, nan, inf):
    out = _good_out()
    section, key, *idx = where
    array = out[section][key] if not idx else out[section][key][idx[0]]
    array[bad] = np.nan if nan else -np.inf
    report = smoke.finite_report(out, CAMPAIGNS)
    campaign_name = smoke.campaign_of(key)
    assert report["all_finite"] is False and report["nonfinite_total"] == 1
    entry = report["campaigns"][campaign_name]
    assert entry["nonfinite"] == 1 and entry["all_finite"] is False
    term = entry["terms"][f"{section}.{key}"]
    assert (term["nan"], term["inf"]) == (nan, inf)
    assert np.isfinite(term["max_abs"])                                 # the maximum is over the finite entries


def test_finite_report_flags_a_nested_nan_and_a_campaign_without_output():
    out = _good_out()
    out["cells"]["p06n_raw_material"][1][1][0, 0] = np.nan
    out["cells"]["p06legacy_raw_centered"]["a"]["total"][1][2, 2] = np.inf
    assert smoke.finite_report(out, CAMPAIGNS)["nonfinite_total"] == 2
    out = _good_out()
    del out["p07"]["p07n_global_N"]
    report = smoke.finite_report(out, CAMPAIGNS)
    assert report["missing_campaigns"] == ["p07n"] and report["all_finite"] is False
    assert smoke.finite_report({"cells": {}, "faces": {}, "p07": {}}, CAMPAIGNS)["all_finite"] is False


def test_sanitize_makes_nan_reports_writable():
    payload = smoke.sanitize({"a": np.nan, "b": [np.inf, -np.inf], "c": np.float64(2.5), "d": np.arange(2),
                              "e": (1, np.int64(2)), "f": None})
    assert payload == {"a": "nan", "b": ["inf", "-inf"], "c": 2.5, "d": [0, 1], "e": [1, 2], "f": None}
    json.dumps(payload, allow_nan=False)


# ---------------------------------------------------------------------------
# Informational change versus the frozen oracles
# ---------------------------------------------------------------------------
def _fake_results():
    term = {"pass": False, "worst_ratio": 7.5, "worst_region": "wall",
            "pointwise": {"violations": 3, "max_abs_diff": 0.25}}
    variants = {"variants": [{"pointwise": {"violations": 1, "max_abs_diff": 0.5}, "worst_ratio": 2.0},
                             {"pointwise": {"violations": 0, "max_abs_diff": 0.1}, "worst_ratio": None}],
                "worst_ratio": 2.0, "total_pointwise_violations": 1, "pass": False}
    pointwise_only = {"pointwise": {"violations": 0, "max_abs_diff": 1e-3}, "pass": True}
    return {"p05": {"status": "ok", "terms": {"centered": term, "live_jump_vs_upwind": pointwise_only}},
            "p06n": {"status": "ok", "terms": {"raw_total": variants}}}


def test_change_summary_records_ratio_region_and_max_abs_diff(monkeypatch, tmp_path):
    monkeypatch.setattr(comparison, "compare_operator_terms", lambda **kw: _fake_results())
    record = smoke.change_vs_frozen_oracles(env=None, out={}, paths={}, campaigns=CAMPAIGNS, n=32,
                                            full_path=tmp_path / "full.json")
    assert record["status"] == "ok" and record["informational"] is True and record["gate"] is False
    assert "NOT A GATE" in record["note"]
    centered = record["campaigns"]["p05"]["centered"]
    assert centered == {"tier_b_ratio": 7.5, "worst_region": "wall", "max_abs_diff": 0.25, "pointwise_violations": 3,
                        "nominal_pass": False}
    assert record["campaigns"]["p06n"]["raw_total"]["max_abs_diff"] == 0.5
    assert record["campaigns"]["p05"]["live_jump_vs_upwind"]["tier_b_ratio"] is None
    assert record["max_tier_b_ratio"] == 7.5 and record["max_tier_b_ratio_term"] == "p05.centered"
    assert (tmp_path / "full.json").is_file() and record["full_results"] == "full.json"


def test_change_comparison_exception_is_recorded_not_raised(monkeypatch):
    def boom(**kw):
        raise ValueError("shape (3, 4) != (3, 5)")

    monkeypatch.setattr(comparison, "compare_operator_terms", boom)
    record = smoke.change_vs_frozen_oracles(env=None, out={}, paths={}, campaigns=CAMPAIGNS, n=32)
    assert record["status"] == "error" and "shape (3, 4) != (3, 5)" in record["error"]
    assert "ValueError" in record["traceback"] and record["gate"] is False and "campaigns" not in record


# ---------------------------------------------------------------------------
# The smoke stage (replay stubbed)
# ---------------------------------------------------------------------------
def _write_artifact(root, n, identity):
    from drbx.stencils import artifact as art
    grid = Path(root) / f"N{n}"
    (grid / "rows").mkdir(parents=True, exist_ok=True)
    chunks = {"cells": [{"file": "rows/c0.npz", "sha256": "0", "bytes": 100, "sources": 3, "targets": 3}],
              "faces": [{"file": "rows/f0.npz", "sha256": "0", "bytes": 300, "sources": 5, "targets": 45}],
              "neumann": [], "p07": []}
    (grid / "manifest.json").write_text(json.dumps({"schema": art.SCHEMA, "identity": identity, "chunks": chunks}))
    (grid / "build_identity.json").write_text(json.dumps(identity))
    (grid / "geometry.npz").write_bytes(b"x" * 64)
    (grid / "rows" / "c0.npz").write_bytes(b"x" * 100)
    (grid / "build_receipt.json").write_text(json.dumps({
        "identity": identity, "wall_seconds": 12.0, "cpu_seconds": 400.0, "peak_rss_gib": 2.5, "effective_workers": 96,
        "bytes_per_row_kind": {"cells": 100, "faces": 300, "neumann": 7, "p07": 11},
        "row_counts": {"cells": 3, "faces": 5, "neumann": 2, "p07": 1}, "chunk_files": 2,
        "diagnostics": {"point_row_families": {"singleton": 10, "coupled_quartic": 4}, "total_point_rows": 14,
                        "tensor_encoding": {"tensor_sources": 7, "fallback_sources": 0},
                        "p07_integrated_row_families": {"1": 3}}}))
    return grid


def _stub_replay(monkeypatch, out_factory, calls):
    def fake_run_replay_stage(**kw):
        calls.append(kw)
        records = {c: {"campaign": c, "seconds": 1.0, "peak_rss_gib": 2.0, "resumed": False} for c in kw["campaigns"]}
        return {"n": kw["n"], "out": out_factory(), "campaigns": records, "plan": {"cells": 3, "plan_bytes": 1234},
                "phases": {"environment": {"seconds": 1.0, "peak_rss_gib": 1.0, "rss_method": "ru_maxrss"},
                           "campaigns": {"seconds": 7.0, "peak_rss_gib": 3.0, "rss_method": "ru_maxrss"}},
                "jax": replay.jax_info(), "process_lifetime_peak_rss_gib": 3.5,
                "config": {"column_block": 8}, "wall_seconds": 8.0}

    monkeypatch.setattr(replay, "run_replay_stage", fake_run_replay_stage)
    monkeypatch.setattr(smoke, "build_environment", lambda **kw: SimpleNamespace(opts=kw))


def _stage_kwargs(tmp_path, **extra):
    cfg = campaign.config()
    return dict(artifact_root=tmp_path / "artifact", output=tmp_path / "replay" / "N8", n=8, input_root=tmp_path,
                sidecar_path=tmp_path / "s.json", paths={}, campaigns=CAMPAIGNS, campaign_identity="abc",
                cfg={**cfg, "tensor_encoding_required": True}, operator_options=FINAL, log=lambda m: None, **extra)


def test_smoke_stage_runs_the_replay_with_the_pinned_options_writes_outputs_and_resumes(tmp_path, monkeypatch):
    _write_artifact(tmp_path / "artifact", 8, _policy_identity())
    calls = []
    _stub_replay(monkeypatch, _good_out, calls)
    monkeypatch.setattr(comparison, "compare_operator_terms", lambda **kw: (
        calls.append(("compare", kw["env"].opts)) or _fake_results()))
    result = smoke.run_smoke_stage(**_stage_kwargs(tmp_path))
    output = tmp_path / "replay" / "N8"
    assert result["smoke_pass"] is True
    replay_call, compare_call = calls
    assert replay_call["operator_options"] == FINAL and replay_call["compare"] is False
    assert replay_call["return_out"] is True and replay_call["output"] == output
    assert compare_call[0] == "compare" and compare_call[1] == FINAL | {"n": 8, "input_root": tmp_path,
                                                                        "sidecar_path": tmp_path / "s.json"}
    body = json.loads((output / "smoke.json").read_text())
    assert body["schema"] == smoke.SCHEMA and body["smoke_pass"] is True and body["operator_options"] == FINAL
    assert body["finite"]["all_finite"] is True and body["finite"]["nonfinite_total"] == 0
    assert body["artifact"]["coupled_quartic"] == 4 and body["artifact"]["bytes_per_row_kind"]["faces"] == 300
    assert body["artifact"]["tensor_encoding"] == {"tensor_sources": 7, "fallback_sources": 0}
    assert body["artifact"]["build"]["wall_seconds"] == 12.0 and body["artifact"]["policy"]["curvature"] == "autodiff"
    assert body["replay"]["plan"]["plan_bytes"] == 1234 and {"smoke_finite_check", "change_vs_frozen_oracles"} <= \
        set(body["replay"]["phases"])
    assert body["change_vs_frozen_oracles"]["status"] == "ok" and body["change_vs_frozen_oracles"]["gate"] is False
    report = (output / "report.md").read_text()
    assert "Smoke: PASS" in report and "not a gate" in report and "coupled_quartic: 4" in report
    assert (output / "change_vs_frozen_oracles.json").is_file() and (output / "plan_info.json").is_file()
    # a finished stage is skipped; a tampered smoke.json or another identity at the same output raises
    again = smoke.run_smoke_stage(**_stage_kwargs(tmp_path))
    assert again["skipped"] is True and len(calls) == 2
    with pytest.raises(ValueError, match="stale smoke receipt"):
        smoke.run_smoke_stage(**{**_stage_kwargs(tmp_path), "campaign_identity": "other"})
    (output / "smoke.json").write_text("{}")
    with pytest.raises(ValueError, match="does not match"):
        smoke.run_smoke_stage(**_stage_kwargs(tmp_path))


def test_smoke_stage_fails_on_nan_still_writes_json_and_records_a_comparison_error(tmp_path, monkeypatch):
    _write_artifact(tmp_path / "artifact", 8, _policy_identity())

    def bad_out():
        out = _good_out()
        out["cells"]["p05_centered"][1][0, 0] = np.nan
        return out

    _stub_replay(monkeypatch, bad_out, [])
    monkeypatch.setattr(comparison, "compare_operator_terms",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("comparison blew up")))
    result = smoke.run_smoke_stage(**_stage_kwargs(tmp_path))
    assert result["smoke_pass"] is False                        # the comparison error did not stop the stage
    body = json.loads((tmp_path / "replay" / "N8" / "smoke.json").read_text())
    assert body["smoke_pass"] is False and body["finite"]["nonfinite_total"] == 1
    assert body["finite"]["campaigns"]["p05"]["terms"]["cells.p05_centered"]["nan"] == 1
    assert body["change_vs_frozen_oracles"]["status"] == "error"
    assert "comparison blew up" in body["change_vs_frozen_oracles"]["error"]
    report = (tmp_path / "replay" / "N8" / "report.md").read_text()
    assert "Smoke: FAIL" in report and "Comparison did not run" in report


def test_smoke_stage_refuses_an_artifact_built_with_other_options_or_without_tensor_sources(tmp_path, monkeypatch):
    _write_artifact(tmp_path / "artifact", 8, _policy_identity(OLD))
    _stub_replay(monkeypatch, _good_out, [])
    with pytest.raises(ValueError, match="not the pinned"):
        smoke.run_smoke_stage(**_stage_kwargs(tmp_path))
    grid = _write_artifact(tmp_path / "artifact", 8, _policy_identity())
    receipt = json.loads((grid / "build_receipt.json").read_text())
    receipt["diagnostics"]["tensor_encoding"]["tensor_sources"] = 0
    (grid / "build_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="no tensor-encoded sources"):
        smoke.run_smoke_stage(**_stage_kwargs(tmp_path))


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def _write_smoke(output, n, identity, *, finite=True, campaigns=CAMPAIGNS, options=FINAL):
    path = output / "replay" / f"N{n}"
    path.mkdir(parents=True, exist_ok=True)
    body = {"schema": smoke.SCHEMA, "n": n, "campaign_identity": identity, "operator_options": options,
            "smoke_pass": finite,
            "finite": {"all_finite": finite, "nonfinite_total": 0 if finite else 5,
                       "campaigns": {c: {"all_finite": finite} for c in campaigns}},
            "artifact": {"path": f"/pscratch/x/artifact/N{n}", "total_bytes": 1000 * n,
                         "bytes_per_row_kind": {"cells": 1, "faces": 2, "neumann": 3, "p07": 4}, "row_counts": {},
                         "point_row_families": {"coupled_quartic": 9}, "coupled_quartic": 9, "tensor_encoding": {},
                         "build": {"wall_seconds": 100.0, "peak_rss_gib": 3.0}},
            "replay": {"wall_seconds": 50.0, "peak_rss_gib": 20.0, "plan": {"plan_bytes": 77}},
            "change_vs_frozen_oracles": {"status": "ok", "max_tier_b_ratio": 3.0, "max_tier_b_ratio_term": "p05.centered"}}
    (path / "smoke.json").write_text(json.dumps(body))


def _passing_preflight(output, identity, grids=(32,), ok=True):
    campaign.write(output / "preflight.json", {"identity": identity, "all_pass": ok,
                                               "cases": {str(n): _fake_case(n, ok) for n in grids}})


def test_validate_checks_grids_identity_options_and_campaigns(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    identity = campaign.verify(input_root=workspace, output=output, oracle_root=None)
    _passing_preflight(output, identity, grids=(32, 48))
    args = ["--resolutions", "32", "48"]
    with pytest.raises(ValueError, match="no smoke.json for N32"):
        campaign.main(_argv("validate", workspace, output, *args))
    _write_smoke(output, 32, identity)
    with pytest.raises(ValueError, match="no smoke.json for N48"):               # a missing requested grid
        campaign.main(_argv("validate", workspace, output, *args))
    _write_smoke(output, 48, "another")
    with pytest.raises(ValueError, match="another campaign identity"):
        campaign.main(_argv("validate", workspace, output, *args))
    _write_smoke(output, 48, identity, options=OLD)
    with pytest.raises(ValueError, match="other operator options"):
        campaign.main(_argv("validate", workspace, output, *args))
    _write_smoke(output, 48, identity, campaigns=CAMPAIGNS[:-1])
    with pytest.raises(ValueError, match="lacks campaigns"):
        campaign.main(_argv("validate", workspace, output, *args))
    _write_smoke(output, 48, identity)
    campaign.main(_argv("validate", workspace, output, *args))
    validation = json.loads((output / "validation.json").read_text())
    assert validation["operational_complete"] is True and validation["smoke_pass"] is True
    assert validation["grids"]["48"]["coupled_quartic"] == 9 and validation["grids"]["32"]["preflight_pass"] is True
    assert validation["grids"]["32"]["change_vs_frozen_oracles"]["gate"] is False
    summary = json.loads((output / "summary" / "step4_summary.json").read_text())
    g = summary["grids"]["48"]
    assert g["artifact_path"] == "/pscratch/x/artifact/N48" and g["total_bytes"] == 48000
    assert g["bytes_per_row_kind"] == {"cells": 1, "faces": 2, "neumann": 3, "p07": 4} and g["coupled_quartic"] == 9
    assert (g["build_wall_seconds"], g["build_peak_rss_gib"], g["replay_wall_seconds"], g["replay_peak_rss_gib"]) == \
           (100.0, 3.0, 50.0, 20.0)


def test_validate_smoke_pass_needs_finite_terms_and_a_passing_preflight(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    identity = campaign.verify(input_root=workspace, output=output, oracle_root=None)
    _passing_preflight(output, identity)
    _write_smoke(output, 32, identity, finite=False)
    campaign.main(_argv("validate", workspace, output, "--resolutions", "32"))
    validation = json.loads((output / "validation.json").read_text())
    assert validation["smoke_pass"] is False and validation["grids"]["32"]["all_finite"] is False
    # a failed preflight stops validate itself
    _passing_preflight(output, identity, ok=False)
    with pytest.raises(ValueError, match="preflight did not pass; refusing to validate"):
        campaign.main(_argv("validate", workspace, output, "--resolutions", "32"))


# ---------------------------------------------------------------------------
# Step 2 keeps its historic options (the only change to it is the optional keyword)
# ---------------------------------------------------------------------------
def _stub_step2_stage(monkeypatch, tmp_path):
    from drbx.stencils import artifact as art
    grid = tmp_path / "artifact" / "N8"
    grid.mkdir(parents=True)
    (grid / "manifest.json").write_text(json.dumps({"schema": art.SCHEMA, "identity": {"a": 1}, "chunks": {}}))
    (grid / "build_identity.json").write_text(json.dumps({"a": 1}))
    seen = {}
    monkeypatch.setattr(replay, "build_environment", lambda **kw: seen.update(kw) or SimpleNamespace(n=8))
    monkeypatch.setattr(ru, "_load_oracle_owner_values", lambda env, paths, campaigns: {})
    monkeypatch.setattr(replay, "evaluate_campaigns",
                        lambda *a, **k: ({"cells": {}, "faces": {}, "p07": {}}, {}))
    return seen


def _step2_kwargs(tmp_path, **extra):
    return dict(artifact_root=tmp_path / "artifact", output=tmp_path / "replay", n=8, input_root=tmp_path,
                sidecar_path=tmp_path / "s.json", paths={}, campaigns=("p05",), campaign_identity="abc",
                cfg={**step2.config(), "tensor_encoding_required": False}, compare=False, log=lambda m: None, **extra)


def test_step2_replay_stage_default_options_are_the_historic_literals(tmp_path, monkeypatch):
    seen = _stub_step2_stage(monkeypatch, tmp_path)
    replay.run_replay_stage(**_step2_kwargs(tmp_path))
    assert {k: seen[k] for k in OLD} == OLD and replay.DEFAULT_OPERATOR_OPTIONS == OLD
    assert set(seen) == {"n", "input_root", "sidecar_path", *OLD}              # nothing else was added to the call


def test_step2_replay_stage_accepts_explicit_options_and_refuses_malformed_ones(tmp_path, monkeypatch):
    seen = _stub_step2_stage(monkeypatch, tmp_path)
    replay.run_replay_stage(**_step2_kwargs(tmp_path, operator_options=FINAL))
    assert {k: seen[k] for k in FINAL} == FINAL
    with pytest.raises(ValueError, match="exactly the keys"):
        replay.run_replay_stage(**_step2_kwargs(tmp_path, operator_options={"curvature": "autodiff"}))
