"""Fast tests of the P08 step-2b G3 campaign (``scripts/p08_step2_global``): CLI parsing, identity / resume /
refusal logic, preflight gating, checkpointed campaign evaluation, report writing, and the column-blocking
equivalence (blocked == unblocked) of the operators on the synthetic perpendicular world.

Fully synthetic / ``tmp_path`` based: no real geometry, no row artifact, no oracle data (the real end-to-end
check is ``tests/test_p08_step2_campaign_real.py``, slow).
"""
from __future__ import annotations

import dataclasses
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

from p08_step2_global import campaign, comparison, replay  # noqa: E402
from p_shared import jax_replay as jr  # noqa: E402
from p_shared import oracle_manifest as om  # noqa: E402
from p_shared import replay_units as ru  # noqa: E402
from p_shared import runner  # noqa: E402

D, N = "dirichlet", "neumann"


# ---------------------------------------------------------------------------
# Fake workspace (as tests/test_p08_step1_campaign.py): the immutable 17-file inputs and the oracle manifest
# are stubbed empty; the canonical sidecar is a real, minimal, schema-shaped file.
# ---------------------------------------------------------------------------
def _fake_workspace(tmp_path, monkeypatch, *, empty_oracle_manifest=True):
    monkeypatch.setattr(campaign, "_input_manifest", lambda: {"files": []})
    if empty_oracle_manifest:
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
# Configuration, sources, CLI
# ---------------------------------------------------------------------------
def test_config_contract_and_inheritance():
    cfg = campaign.config()
    from p_shared.replay_support import CAMPAIGN_FUNCS
    step1 = campaign.step1.config()
    assert cfg["campaigns"] == list(CAMPAIGN_FUNCS) and len(cfg["campaigns"]) == 7
    assert cfg["resolutions"] == [32, 48] and cfg["allowed_resolutions"] == [32, 48, 64]
    for key in ("cell_chunk_size", "face_chunk_size", "p07_chunk_size", "geometry_raw_chunk_size",
                "geometry_face_chunk_size", "tier_b_ratio_tolerance", "oracle_default_paths",
                "canonical_sidecar_relative", "pointwise_cap_scale_factor"):
        assert cfg[key] == step1[key]                                # inherited, not restated
    assert cfg["artifact_schema"] == "drbx.p-row-artifact.v3" and cfg["tensor_encoding_required"] is True
    assert cfg["boundary_batch"] % 4096 == 0 and cfg["column_block"] >= 5 and cfg["p06n_variant_block"] >= 1
    assert cfg["omitted_host_only_terms"] == comparison.OMITTED_TERMS
    assert cfg["operator_terms_only"] is True


def test_source_hashes_cover_existing_files_and_the_step1_and_jax_stacks():
    hashes = campaign.source_hashes()
    assert len(hashes) == len(campaign.SOURCE_FILES) == len(set(campaign.SOURCE_FILES))
    for rel, digest in hashes.items():
        assert (REPO / rel).is_file(), rel
        assert len(digest) == 64
    for needed in (*campaign.step1.SOURCE_FILES, "scripts/p08_step2_global/replay.py",
                   "scripts/p08_step2_global/comparison.py", "scripts/p08_step2_global/configuration.json",
                   "scripts/p_shared/jax_replay.py", "src/drbx/stencils/operator_plan.py",
                   "src/drbx/native/fci_perpendicular_p06_operator.py"):
        assert needed in hashes


def test_oracle_and_input_manifests_are_the_step1_files():
    step1 = REPO / "scripts/p08_step1_global"
    assert campaign.committed_oracle_manifest() == json.loads((step1 / "oracle_manifest.json").read_text())
    assert campaign._input_manifest() == json.loads((step1 / "input_manifest.json").read_text())
    assert not (REPO / "scripts/p08_step2_global/oracle_manifest.json").exists()      # nothing to repack


def test_cli_parsing():
    a = campaign.parse(["run", "--input-root", "/x", "--output", "/y"])
    assert a.resolutions == [32, 48] and a.workers == 4 and a.max_units is None and a.oracle_root is None
    a = campaign.parse(["run", "--input-root", "/x", "--output", "/y", "--resolutions", "32", "48", "64",
                        "--workers", "128", "--memory-budget-gib", "400", "--worker-memory-gib", "3.5",
                        "--memory-reserve-gib", "8", "--oracle-root", "/o", "--max-units", "2"])
    assert a.resolutions == [32, 48, 64] and a.workers == 128 and a.memory_budget_gib == 400.0
    assert a.worker_memory_gib == 3.5 and a.memory_reserve_gib == 8.0 and a.max_units == 2
    assert a.oracle_root == Path("/o")
    a = campaign.parse(["run-stage", "--input-root", "/x", "--output", "/y", "--stage", "replay", "--n", "48"])
    assert (a.stage, a.n) == ("replay", 48)
    for bad in (["run", "--resolutions", "16"], ["frobnicate"], ["run-stage", "--stage", "cells"],
                ["pack-oracles"]):
        with pytest.raises(SystemExit):
            campaign.parse([*bad, "--input-root", "/x", "--output", "/y"] if bad[0] != "frobnicate"
                           else [*bad, "--input-root", "/x", "--output", "/y"])


def test_run_stage_requires_stage_and_n(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="--stage and --n are required"):
        campaign.main(_argv("run-stage", workspace, tmp_path / "out"))


def test_resolutions_must_be_unique_and_ascending(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unique and ascending"):
        campaign.main(_argv("verify-inputs", workspace, tmp_path / "out", "--resolutions", "48", "32"))


# ---------------------------------------------------------------------------
# Identity / verify
# ---------------------------------------------------------------------------
def test_verify_is_stable_records_step2_manifest_and_refuses_changes(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "campaign_output"
    identity = campaign.verify(input_root=workspace, output=output, oracle_root=None)
    assert campaign.verify(input_root=workspace, output=output, oracle_root=None) == identity
    saved = json.loads((output / "campaign_manifest.json").read_text())
    assert saved["campaign"] == "p08_step2_global" and saved["identity"] == identity and saved["jax_x64"] is True
    assert saved["jax_backend"] == "cpu"
    for name in ("oracle_manifest.json", "localized_sidecar.json"):
        assert (output / name).is_file()
    # another oracle root at the same output, or a changed source, must raise
    with pytest.raises(ValueError, match="oracle root changed"):
        campaign.verify(input_root=workspace, output=output, oracle_root=tmp_path / "elsewhere")
    monkeypatch.setattr(campaign, "source_hashes", lambda: {"fake/path.py": "0" * 64})
    with pytest.raises(ValueError, match="campaign identity changed"):
        campaign.verify(input_root=workspace, output=output, oracle_root=None)


def test_identity_differs_from_step1_identity(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign.step1, "_input_manifest", lambda: {"files": []})
    monkeypatch.setattr(campaign.step1, "committed_oracle_manifest",
                        lambda: {"schema": om.MANIFEST_SCHEMA, "workspace_root": "", "campaigns": {}})
    one = campaign.step1.verify(input_root=workspace, output=tmp_path / "one", oracle_root=None)
    two = campaign.verify(input_root=workspace, output=tmp_path / "two", oracle_root=None)
    assert one != two
    with pytest.raises(ValueError, match="campaign identity changed"):          # a step-1 folder is not reusable
        campaign.verify(input_root=workspace, output=tmp_path / "one", oracle_root=None)


def test_verify_refuses_missing_immutable_input_and_missing_oracle_files(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, "_input_manifest",
                        lambda: {"files": [{"path": "does/not/exist.npz", "bytes": 1, "sha256": "0" * 64}]})
    with pytest.raises(ValueError, match="missing or changed immutable input"):
        campaign.verify(input_root=workspace, output=tmp_path / "o1", oracle_root=None)
    monkeypatch.setattr(campaign, "_input_manifest", lambda: {"files": []})
    monkeypatch.setattr(campaign, "committed_oracle_manifest", campaign.step1.committed_oracle_manifest)
    with pytest.raises(ValueError, match="oracle verification failed"):
        campaign.verify(input_root=workspace, output=tmp_path / "o2", oracle_root=tmp_path / "nowhere")


def test_verify_inputs_command_locks_and_records(tmp_path, monkeypatch, capsys):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    campaign.main(_argv("verify-inputs", workspace, output))
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "verify-inputs" and payload["status"] == "complete"
    assert (output / ".campaign.lock").exists()
    assert list((output / "invocations").glob("*_verify-inputs.json"))
    assert json.loads((output / "last_exit.json").read_text())["status"] == "complete"


# ---------------------------------------------------------------------------
# Preflight gating
# ---------------------------------------------------------------------------
def _fake_case(n, ok=True):
    return {"n": n, "all_pass": ok, "policy_failures": [] if ok else ["cells.p05_centered"]}


def test_run_and_validate_refuse_without_a_matching_preflight(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="run requires a matching preflight"):
        campaign.main(_argv("run", workspace, output, "--resolutions", "32", "--workers", "1"))
    with pytest.raises(ValueError, match="a matching preflight is required before validate"):
        campaign.main(_argv("validate", workspace, output, "--resolutions", "32"))
    # a preflight of another identity is not a matching one
    write = campaign.write
    write(output / "preflight.json", {"identity": "someone-else", "cases": {"32": _fake_case(32)}, "all_pass": True})
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
    with pytest.raises(ValueError, match="preflight did not pass; refusing to validate"):
        campaign.main(_argv("validate", workspace, output, "--resolutions", "48"))
    with pytest.raises(ValueError, match=r"no case for N\[64\]"):
        campaign.main(_argv("run", workspace, output, "--resolutions", "32", "64"))
    # ... but N32 alone passes the gate (fails later, at the build, which this test does not want)
    monkeypatch.setattr(campaign, "build_artifact_stage", lambda **kw: (_ for _ in ()).throw(RuntimeError("build")))
    with pytest.raises(RuntimeError, match="build"):
        campaign.main(_argv("run", workspace, output, "--resolutions", "32"))


def test_preflight_command_is_resumable_and_merges_grids(tmp_path, monkeypatch, capsys):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    calls = []

    def fake_preflight_grid(*, n, **kw):
        calls.append(n)
        return _fake_case(n, ok=(n != 48 or calls.count(48) > 1))      # N48 fails the first time only

    monkeypatch.setattr(campaign, "preflight_grid", fake_preflight_grid)
    campaign.main(_argv("preflight", workspace, output, "--resolutions", "32", "48"))
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["all_pass"] is False
    saved = json.loads((output / "preflight.json").read_text())
    assert set(saved["cases"]) == {"32", "48"} and saved["all_pass"] is False and calls == [32, 48]
    campaign.main(_argv("preflight", workspace, output, "--resolutions", "32", "48"))   # 32 kept, 48 redone
    assert calls == [32, 48, 48]
    assert json.loads((output / "preflight.json").read_text())["all_pass"] is True
    campaign.main(_argv("preflight", workspace, output, "--resolutions", "32"))         # nothing to redo
    assert calls == [32, 48, 48]
    assert set(json.loads((output / "preflight.json").read_text())["cases"]) == {"32", "48"}   # merged, not dropped


def test_preflight_grid_requires_cpu_and_x64(monkeypatch, tmp_path):
    monkeypatch.setattr(replay, "jax_info", lambda: {"backend": "gpu", "x64": True})
    with pytest.raises(ValueError, match="CPU backend required"):
        campaign.preflight_grid(n=32, input_root=tmp_path, sidecar_path=tmp_path / "s.json", output=tmp_path,
                                paths={})
    monkeypatch.setattr(replay, "jax_info", lambda: {"backend": "cpu", "x64": False})
    with pytest.raises(ValueError, match="x64 required"):
        campaign.preflight_grid(n=32, input_root=tmp_path, sidecar_path=tmp_path / "s.json", output=tmp_path,
                                paths={})


def test_preflight_grid_passes_the_blocked_path_and_summarizes(monkeypatch, tmp_path):
    seen = {}

    def fake_check(**kwargs):
        seen.update(kwargs)
        return {"all_diff_pass": True, "oracle_jax_all_pass": True, "uniq_mismatches": [],
                "diff_table": [{}] * 29, "diff_failures": [], "oracle_jax": [{"campaign": "p05", "term": "t", "pass": True}],
                "plan": {"cells": 1}, "blocking": {"column_block": 8}, "peak_rss_gib": 1.5, "wall_seconds": 2.0}

    monkeypatch.setattr(jr, "run_jax_owner_closure_check", fake_check)
    case = campaign.preflight_grid(n=32, input_root=tmp_path, sidecar_path=tmp_path / "s.json", output=tmp_path,
                                   paths={"p05": tmp_path})
    cfg = campaign.config()
    assert case["all_pass"] is True and case["jax"]["backend"] == "cpu" and case["jax"]["x64"] is True
    assert (seen["column_block"], seen["variant_block"], seen["boundary_batch"]) == \
           (cfg["column_block"], cfg["p06n_variant_block"], cfg["boundary_batch"])
    assert seen["campaigns"] == tuple(cfg["campaigns"]) and seen["n"] == 32
    # one failing gate fails the case
    monkeypatch.setattr(jr, "run_jax_owner_closure_check",
                        lambda **kw: {**fake_check(**kw), "oracle_jax_all_pass": False})
    assert campaign.preflight_grid(n=32, input_root=tmp_path, sidecar_path=tmp_path / "s.json", output=tmp_path,
                                   paths={})["all_pass"] is False


# ---------------------------------------------------------------------------
# run: stage orchestration, skipping, validate
# ---------------------------------------------------------------------------
def _passing_preflight(output, identity, grids=(32, 48)):
    campaign.write(output / "preflight.json", {"identity": identity, "all_pass": True,
                                               "cases": {str(n): _fake_case(n) for n in grids}})


def _fake_replay_json(output, n, identity, campaigns, ok=True):
    path = output / "replay" / f"N{n}"
    path.mkdir(parents=True, exist_ok=True)
    body = {"schema": replay.SCHEMA, "n": n, "campaign_identity": identity,
            "campaigns": {c: {"status": "ok", "terms": {"t": {"pass": ok}}} for c in campaigns}}
    (path / "replay.json").write_text(json.dumps(body))


def test_run_builds_then_replays_each_grid_then_validates(tmp_path, monkeypatch, capsys):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    identity = campaign.verify(input_root=workspace, output=output, oracle_root=None)
    _passing_preflight(output, identity)
    cfg = campaign.config()
    log = []

    def fake_build(*, n, args, sidecar_path, artifact_root):
        log.append(("build", n, artifact_root))
        return {"wall_seconds": 1.0}

    def fake_replay(*, n, args, identity, sidecar_path, paths, artifact_root):
        log.append(("replay", n))
        _fake_replay_json(args.output, n, identity, cfg["campaigns"])
        return {"all_terms_pass": True, "wall_seconds": 3.0}

    monkeypatch.setattr(campaign, "build_artifact_stage", fake_build)
    monkeypatch.setattr(campaign, "replay_stage", fake_replay)
    campaign.main(_argv("run", workspace, output))                         # default resolutions 32 48
    assert [x[:2] for x in log] == [("build", 32), ("replay", 32), ("build", 48), ("replay", 48)]
    assert log[0][2] == output.resolve() / "artifact"
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["operational_complete"] is True and summary["all_terms_pass"] is True
    validation = json.loads((output / "validation.json").read_text())
    assert validation["resolutions"] == [32, 48] and validation["scope"]["operator_terms_only"] is True
    assert all(validation["campaign_status"]["48"][c] == "ok" for c in cfg["campaigns"])
    assert validation["campaign_pass"]["32"]["p07"] is True


def test_bounded_run_stops_after_the_build(tmp_path, monkeypatch, capsys):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    identity = campaign.verify(input_root=workspace, output=output, oracle_root=None)
    _passing_preflight(output, identity)
    monkeypatch.setattr(campaign, "build_artifact_stage", lambda **kw: {"wall_seconds": 1.0})
    monkeypatch.setattr(campaign, "replay_stage", lambda **kw: pytest.fail("replay must not run on a partial build"))
    campaign.main(_argv("run", workspace, output, "--resolutions", "32", "--max-units", "1"))
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["grids"][0]["status"] == "bounded_build_only" and "operational_complete" not in summary
    assert not (output / "validation.json").exists()
    monkeypatch.setattr(campaign, "build_artifact_stage", lambda **kw: {"geometry_complete": False})
    campaign.main(_argv("run", workspace, output, "--resolutions", "32", "--max-units", "1"))
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["grids"][0]["status"] == "geometry_incomplete"


def test_run_stage_dispatch(tmp_path, monkeypatch, capsys):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    calls = []
    monkeypatch.setattr(campaign, "build_artifact_stage",
                        lambda **kw: calls.append(("artifact", kw["n"])) or {"wall_seconds": 2.0})
    monkeypatch.setattr(campaign, "replay_stage",
                        lambda **kw: calls.append(("replay", kw["n"])) or {"wall_seconds": 3.0, "all_terms_pass": True})
    campaign.main(_argv("run-stage", workspace, output, "--stage", "artifact", "--n", "48"))
    campaign.main(_argv("run-stage", workspace, output, "--stage", "replay", "--n", "32"))
    assert calls == [("artifact", 48), ("replay", 32)]      # recovery stages are not preflight-gated
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["stage"] == "replay"


def test_validate_checks_identity_grids_and_campaigns(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    identity = campaign.verify(input_root=workspace, output=output, oracle_root=None)
    _passing_preflight(output, identity, grids=(32,))
    cfg = campaign.config()
    args = ["--resolutions", "32"]
    with pytest.raises(ValueError, match="no replay.json for N32"):
        campaign.main(_argv("validate", workspace, output, *args))
    _fake_replay_json(output, 32, "another", cfg["campaigns"])
    with pytest.raises(ValueError, match="another campaign identity"):
        campaign.main(_argv("validate", workspace, output, *args))
    _fake_replay_json(output, 32, identity, cfg["campaigns"][:-1])
    with pytest.raises(ValueError, match="lacks campaigns"):
        campaign.main(_argv("validate", workspace, output, *args))
    _fake_replay_json(output, 32, identity, cfg["campaigns"], ok=False)
    campaign.main(_argv("validate", workspace, output, *args))
    validation = json.loads((output / "validation.json").read_text())
    assert validation["all_terms_pass"] is False and validation["operational_complete"] is True


def test_csr_only_build_is_refused(monkeypatch):
    monkeypatch.setenv("P_SHARED_CSR_ONLY", "1")
    with pytest.raises(ValueError, match="P_SHARED_CSR_ONLY"):
        campaign._refuse_csr_only()
    monkeypatch.delenv("P_SHARED_CSR_ONLY")
    campaign._refuse_csr_only()


def test_failure_records_last_exit(tmp_path, monkeypatch):
    workspace = _fake_workspace(tmp_path, monkeypatch)
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["campaign.py", *_argv("run", workspace, output)])
    with pytest.raises(ValueError, match="run requires a matching preflight"):
        campaign.main()
    # the __main__ handler is what writes the failure record; exercise it through runpy-free replication
    campaign.write(output / "last_exit.json", {"command": "run", "status": "failed", "error": "x", "time": 0.0})
    assert json.loads((output / "last_exit.json").read_text())["status"] == "failed"


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def _write_manifest(root, n, *, schema, identity, groups=None):
    grid = Path(root) / f"N{n}"
    grid.mkdir(parents=True, exist_ok=True)
    chunks = groups if groups is not None else {"cells": [{"file": "rows/cells_0.npz", "sha256": "0", "bytes": 100,
                                                            "sources": 3, "targets": 3}],
                                                 "faces": [{"file": "rows/faces_0.npz", "sha256": "0", "bytes": 300,
                                                            "sources": 5, "targets": 45}], "neumann": [], "p07": []}
    (grid / "manifest.json").write_text(json.dumps({"schema": schema, "identity": identity, "chunks": chunks}))
    (grid / "build_identity.json").write_text(json.dumps(identity))
    return grid


def test_load_artifact_schema_identity_and_tensor_checks(tmp_path):
    from drbx.stencils import artifact as art
    identity = {"a": 1}
    grid = _write_manifest(tmp_path, 32, schema=art.SCHEMA, identity=identity)
    loaded = replay.load_artifact(tmp_path, 32)
    assert loaded["summary"]["groups"]["faces"] == {"files": 1, "bytes": 300, "sources": 5, "targets": 45}
    assert loaded["summary"]["total_bytes"] == 400 and loaded["receipt"] is None
    with pytest.raises(ValueError, match="identity mismatch"):
        replay.load_artifact(tmp_path, 32, identity={"a": 2})
    # tensor requirement reads the build receipt
    (grid / "build_receipt.json").write_text(json.dumps({"diagnostics": {"tensor_encoding": {
        "tensor_sources": 0, "fallback_sources": 0}}, "wall_seconds": 5.0}))
    with pytest.raises(ValueError, match="no tensor-encoded sources"):
        replay.load_artifact(tmp_path, 32, require_tensor=True)
    (grid / "build_receipt.json").write_text(json.dumps({"diagnostics": {"tensor_encoding": {
        "tensor_sources": 7, "fallback_sources": 1}}, "wall_seconds": 5.0}))
    loaded = replay.load_artifact(tmp_path, 32, require_tensor=True)
    assert loaded["summary"]["tensor_encoding"]["tensor_sources"] == 7
    _write_manifest(tmp_path, 48, schema=art.SCHEMA_V2, identity=identity)
    with pytest.raises(ValueError, match="needs a drbx.p-row-artifact.v3 artifact"):
        replay.load_artifact(tmp_path, 48)


# ---------------------------------------------------------------------------
# Checkpointed campaign evaluation
# ---------------------------------------------------------------------------
def _fake_out(seed):
    rng = np.random.default_rng(seed)
    uniq = np.array([1, 4, 6], dtype=np.int64)
    return {"cells": {"p05_centered": (uniq, rng.normal(size=(3, 8))), "p05_antisymmetry_max": 1e-15,
                      "p06n_raw_material": [(uniq, rng.normal(size=(3, 4))), (uniq, rng.normal(size=(3, 4)))],
                      "p06legacy_raw_centered": {"a": {"total": (uniq, rng.normal(size=(3, 4)))}}},
            "faces": {"p05_live_jump_p07ids": np.array([3, 9], dtype=np.int64),
                      "p05_live_jump_values": rng.normal(size=(2, 8))},
            "p07": {"p07_global_N": (uniq, rng.normal(size=(3, 4)))}}


def _assert_same(a, b):
    assert type(a) is type(b) or (isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)))
    if isinstance(a, dict):
        assert list(a) == list(b)
        for k in a:
            _assert_same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_same(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


def test_evaluate_campaigns_checkpoints_resumes_and_refuses_stale_or_corrupt(tmp_path, monkeypatch):
    env = SimpleNamespace(n=8)
    outs = {"p05": _fake_out(0), "p06n": _fake_out(1), "p07": _fake_out(2)}
    computed = []

    def fake_evaluate(env_, plan, oracle, campaign_, cfg, log=None):
        computed.append(campaign_)
        assert plan == "PLAN"
        return outs[campaign_], {"campaign": campaign_, "seconds": 1.5, "setup_seconds": 0.5,
                                 "boundary_data_seconds": 0.2, "operator_seconds": {"p": 1.0}, "peak_rss_gib": 2.5}

    monkeypatch.setattr(replay, "evaluate_campaign", fake_evaluate)
    output = tmp_path / "replay" / "N8"
    ident = replay.checkpoint_identity(campaign_identity="abc", artifact_identity={"x": 1}, n=8,
                                       cfg={"column_block": 8, "p06n_variant_block": 2, "boundary_batch": 4096})
    made = []
    factory = lambda: made.append(1) or "PLAN"
    campaigns = ("p05", "p06n", "p07")
    merged, records = replay.evaluate_campaigns(env, factory, {}, campaigns, {}, output=output, identity=ident,
                                                n=8, log=lambda m: None)
    assert computed == list(campaigns) and made == [1] and not any(r["resumed"] for r in records.values())
    assert set(merged) == {"cells", "faces", "p07"} and "p05_centered" in merged["cells"]
    for c in campaigns:
        assert runner.valid_unit(output, replay.checkpoint_unit(8, c), ident)
    # resume: nothing recomputed, the plan is not even lowered, the merged terms are the same
    computed.clear(); made.clear()
    merged2, records2 = replay.evaluate_campaigns(env, factory, {}, campaigns, {}, output=output, identity=ident,
                                                  n=8, log=lambda m: None)
    assert computed == [] and made == [] and all(r["resumed"] and r["seconds"] == 1.5 for r in records2.values())
    _assert_same(merged["cells"]["p05_centered"], merged2["cells"]["p05_centered"])
    _assert_same(merged["cells"]["p06n_raw_material"], merged2["cells"]["p06n_raw_material"])
    _assert_same(merged["cells"]["p06legacy_raw_centered"], merged2["cells"]["p06legacy_raw_centered"])
    _assert_same(merged["faces"]["p05_live_jump_values"], merged2["faces"]["p05_live_jump_values"])
    assert merged2["cells"]["p05_antisymmetry_max"] == 1e-15
    # one missing checkpoint: only that campaign is recomputed
    runner.receipt_path(output, replay.checkpoint_unit(8, "p06n")).unlink()
    replay.evaluate_campaigns(env, factory, {}, campaigns, {}, output=output, identity=ident, n=8,
                              log=lambda m: None)
    assert computed == ["p06n"] and made == [1]
    # another identity or a corrupt file raises instead of recomputing
    other = {**ident, "campaign": "xyz"}
    with pytest.raises(ValueError, match="stale checkpoint"):
        replay.evaluate_campaigns(env, factory, {}, campaigns, {}, output=output, identity=other, n=8,
                                  log=lambda m: None)
    path = runner.unit_path(output, replay.checkpoint_unit(8, "p07"))
    with path.open("r+b") as f:
        f.write(b"\x00" * 8)
    with pytest.raises(ValueError, match="corrupt checkpoint"):
        replay.evaluate_campaigns(env, factory, {}, campaigns, {}, output=output, identity=ident, n=8,
                                  log=lambda m: None)


def test_checkpoint_identity_depends_on_campaign_artifact_and_blocking():
    base = dict(campaign_identity="abc", artifact_identity={"x": 1}, n=8,
                cfg={"column_block": 8, "p06n_variant_block": 2, "boundary_batch": 4096, "wall_cache": False})
    ident = replay.checkpoint_identity(**base)
    assert replay.checkpoint_identity(**{**base, "campaign_identity": "abd"}) != ident
    assert replay.checkpoint_identity(**{**base, "artifact_identity": {"x": 2}}) != ident
    assert replay.checkpoint_identity(**{**base, "n": 9}) != ident
    assert replay.checkpoint_identity(**{**base, "cfg": {**base["cfg"], "column_block": 16}}) != ident
    assert json.loads(json.dumps(ident)) == ident                            # JSON-safe (valid_unit compares reloads)


# ---------------------------------------------------------------------------
# The stage: receipts, report, skip / refusal
# ---------------------------------------------------------------------------
def _stub_stage(monkeypatch, tmp_path, *, results=None):
    from drbx.stencils import artifact as art
    _write_manifest(tmp_path / "artifact", 8, schema=art.SCHEMA, identity={"a": 1})
    monkeypatch.setattr(replay, "build_environment", lambda **kw: SimpleNamespace(n=8))
    monkeypatch.setattr(ru, "_load_oracle_owner_values", lambda env, paths, campaigns: {})
    calls = {"evaluate": 0, "compare": 0}

    def fake_evaluate_campaigns(env, plan_factory, oracle, campaigns, cfg, *, output, identity, n, log):
        calls["evaluate"] += 1
        return _fake_out(0), {c: {"campaign": c, "seconds": 1.0, "resumed": False} for c in campaigns}

    def fake_compare(*, env, out, paths, campaigns, n):
        calls["compare"] += 1
        return results or {c: {"campaign": c, "status": "ok",
                               "terms": {"global_N": {"pass": True, "worst_ratio": 0.5, "worst_region": "all",
                                                      "pointwise": {"violations": 0}}}} for c in campaigns}

    monkeypatch.setattr(replay, "evaluate_campaigns", fake_evaluate_campaigns)
    monkeypatch.setattr(replay.comparison, "compare_operator_terms", fake_compare)
    return calls


def _stage_kwargs(tmp_path, **extra):
    cfg = {**campaign.config(), "tensor_encoding_required": False}
    return dict(artifact_root=tmp_path / "artifact", output=tmp_path / "replay" / "N8", n=8, input_root=tmp_path,
                sidecar_path=tmp_path / "s.json", paths={}, campaigns=("p05", "p07"), campaign_identity="abc",
                cfg=cfg, log=lambda m: None, **extra)


def test_replay_stage_writes_step1_format_outputs_and_skips_when_complete(tmp_path, monkeypatch):
    calls = _stub_stage(monkeypatch, tmp_path)
    out = replay.run_replay_stage(**_stage_kwargs(tmp_path))
    output = tmp_path / "replay" / "N8"
    assert calls == {"evaluate": 1, "compare": 1} and out["all_terms_pass"] is True
    replay_json = json.loads((output / "replay.json").read_text())
    assert replay_json["schema"] == replay.SCHEMA and replay_json["n"] == 8
    assert set(replay_json["campaigns"]) == {"p05", "p07"} and replay_json["campaign_identity"] == "abc"
    assert replay_json["scope"]["operator_terms_only"] is True
    assert replay_json["scope"]["omitted_host_only_terms"]["p07n"] == ["global_O_q3"]
    assert replay_json["g3"]["jax"]["backend"] == "cpu" and replay_json["g3"]["artifact"]["total_bytes"] == 400
    assert replay_json["g3"]["boundary_data_route"] == "live"
    report = (output / "report.md").read_text()
    assert report.startswith("# P08 step 2b G3 JAX full-grid replay gate -- N8")
    assert "Omitted host-only MMS reference terms" in report and "raw_R" in report and "O_q3" in report
    assert "| p05 | ok | 1 |" in report and "PASS" in report
    receipts = json.loads((output / "receipts.json").read_text())
    assert set(receipts["phases"]) >= {"environment", "campaigns", "compare"}
    assert (output / "stage_receipt.json").is_file()
    # second run: skipped entirely
    again = replay.run_replay_stage(**_stage_kwargs(tmp_path))
    assert again["skipped"] is True and calls == {"evaluate": 1, "compare": 1}
    # a tampered replay.json is never skipped silently
    (output / "replay.json").write_text("{}")
    with pytest.raises(ValueError, match="does not match"):
        replay.run_replay_stage(**_stage_kwargs(tmp_path))
    # another campaign identity at the same output raises
    (output / "replay.json").write_text(json.dumps(replay_json, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="stale replay stage receipt"):
        replay.run_replay_stage(**{**_stage_kwargs(tmp_path), "campaign_identity": "other"})


def test_replay_stage_reports_a_failing_term(tmp_path, monkeypatch):
    bad = {"p05": {"campaign": "p05", "status": "ok",
                   "terms": {"centered": {"pass": False, "worst_ratio": 3.0, "worst_region": "wall",
                                          "pointwise": {"violations": 2}}}}}
    _stub_stage(monkeypatch, tmp_path, results=bad)
    out = replay.run_replay_stage(**{**_stage_kwargs(tmp_path), "campaigns": ("p05",)})
    assert out["all_terms_pass"] is False
    assert "FAIL" in (tmp_path / "replay" / "N8" / "report.md").read_text()
    assert json.loads((tmp_path / "replay" / "N8" / "replay.json").read_text())["all_terms_pass"] is False


def test_replay_stage_selection_mode_needs_no_comparison(tmp_path, monkeypatch):
    calls = _stub_stage(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="needs the full grid"):
        replay.run_replay_stage(**_stage_kwargs(tmp_path, selection={"raw_ids": []}))
    out = replay.run_replay_stage(**_stage_kwargs(tmp_path, selection={"raw_ids": []}, compare=False,
                                                  return_out=True))
    assert calls == {"evaluate": 1, "compare": 0} and "p05_centered" in out["out"]["cells"]
    assert not (tmp_path / "replay" / "N8" / "replay.json").exists()


def test_step3_0_control_list_is_a_tripwire_on_the_roundoff_floor_classification():
    variants = ("main", "control_constant_dirichlet", "control_constant_neumann", "main:D",
                "control_constant_dirichlet:D", "control_constant_neumann:D")
    controls = comparison._expected_roundoff_variants(variants, ("control_constant_dirichlet", "control_constant_neumann"))
    assert controls == {"control_constant_dirichlet", "control_constant_neumann", "control_constant_dirichlet:D",
                        "control_constant_neumann:D"}

    def result(mode):
        return {"tier_b_mode": mode, "pass": True}

    good = [result("ratio" if v not in controls else "roundoff_floor") for v in variants]
    comparison._check_roundoff_classification(good, variants, controls)
    assert all(r["pass"] and "roundoff_classification_mismatch" not in r for r in good)
    assert [r["expected_roundoff_control"] for r in good] == [v in controls for v in variants]
    # a real variant that slipped onto the floor gate (a weakened gate), and a control that did not, both fail
    weakened = [result("roundoff_floor") for _ in variants]
    comparison._check_roundoff_classification(weakened, variants, controls)
    assert [r["pass"] for r in weakened] == [v in controls for v in variants]
    missing = [result("ratio") for _ in variants]
    comparison._check_roundoff_classification(missing, variants, controls)
    assert [r["pass"] for r in missing] == [v not in controls for v in variants]


def test_step3_0_call_sites_pass_the_rules_only_for_the_named_terms():
    """AST check of ``compare_operator_terms``: the P05 per-face jump, P05N ``face_N``/``face_D`` (cancellation) and
    the P06N raw terms (roundoff floor) get the step 3.0 arguments; no other comparison does."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(comparison.compare_operator_terms)))
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in ("compare_owner_term", "compare_pointwise_only"):
            label = ast.unparse(node.args[0])                        # the term name expression
            calls[label] = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
    new_args = {"kind", "constituent_scale", "floor_abs", "tier_b", "roundoff_scale_l2"}
    with_rule = {label: sorted(new_args & set(kw)) for label, kw in calls.items() if new_args & set(kw)}
    assert with_rule == {
        "'p05.live_jump_vs_upwind'": ["constituent_scale", "kind"],
        "'p05n.face_N'": ["constituent_scale", "kind"],
        "'p05n.face_D'": ["constituent_scale", "kind"],
        "f'p06n.raw_{label}[{name_}]'": ["roundoff_scale_l2", "tier_b"],
    }
    assert calls["'p05.live_jump_vs_upwind'"]["kind"] == "'cancellation'"
    assert calls["'p05n.face_N'"]["kind"] == calls["'p05n.face_D'"]["kind"] == "'cancellation'"
    assert calls["f'p06n.raw_{label}[{name_}]'"]["tier_b"] == "'auto'"
    assert len(calls) == 13                                            # every comparison site was seen
    # step 1 keeps the literal rules
    reduce_source = inspect.getsource(ru.reduce_grid)
    for token in ("kind=", "tier_b=", "roundoff_scale_l2", "constituent_scale", "floor_abs"):
        assert token not in reduce_source, token


def test_omitted_terms_are_exactly_the_host_only_references():
    assert set(comparison.OMITTED_TERMS) == {"p05n_frozen", "p05n_upwind", "p06n", "p07n"}
    for terms in comparison.OMITTED_TERMS.values():
        assert all(t.startswith(("raw_R", "global_O")) for t in terms)
    # jax_replay's own host-only roots are the same references
    assert jr._HOST_ONLY_ROOTS == {"p05n_frozen_raw_R", "p05n_upwind_raw_R", "p07n_global_O_q3"}


# ---------------------------------------------------------------------------
# Blocking helpers and blocked == unblocked on the synthetic perpendicular world
# ---------------------------------------------------------------------------
def test_greedy_blocks_and_batched_callable():
    needs = [{0, 1}, {1, 2}, {2, 3}, {4, 5}, {6, 7}, {0, 7}]
    assert jr.greedy_blocks(needs, 4) == [[0, 1, 2], [3, 4], [5]]
    assert jr.greedy_blocks(needs, 100) == [[0, 1, 2, 3, 4, 5]]
    assert jr.greedy_blocks(needs, 100, limit=2) == [[0, 1], [2, 3], [4, 5]]
    assert jr.greedy_blocks([set(range(9))], 4) == [[0]]                     # oversized item: its own block
    assert jr.greedy_blocks([], 4) == []
    calls = []

    def fn(points):
        calls.append(len(points))
        return points[:, :2] * 2.0, points[:, :, None] * np.ones((1, 1, 3))

    points = np.random.default_rng(0).normal(size=(11, 3))
    whole = fn(points)
    calls.clear()
    seen = []
    parts = jr.batched_callable(fn, 4, lambda done, total: seen.append((done, total)))(points)
    assert calls == [4, 4, 3] and seen == [(4, 11), (8, 11), (11, 11)]
    for a, b in zip(whole, parts):
        np.testing.assert_array_equal(a, b)
    assert jr.batched_callable(fn, None) is fn and jr.batched_callable(None, 4) is None
    calls.clear()
    jr.batched_callable(fn, 64)(points)
    assert calls == [11]
    single = jr.batched_callable(lambda p: p[:, 0], 5)(points)
    np.testing.assert_array_equal(single, points[:, 0])


class _World:
    """The bounded synthetic closure of the P06 operator tests (admissible, positive rows)."""
    owners = [0, 1, 2, 40, 96, 99, 100, 114]
    nphys = 8

    def __init__(self):
        from tests.perpendicular_synthetic import Boundary, lower_world, make_world
        from drbx.native.fci_perpendicular_reconstruction_state import boundary_data_from_callables
        world = make_world(owners=self.owners)
        for key, row in list(world.row_index.items()):
            if hasattr(row, "value") and getattr(row, "value").ndim == 2 and hasattr(row, "gradient"):
                a = np.abs(row.value)
                world.row_index[key] = dataclasses.replace(row, value=a / a.sum(axis=-1, keepdims=True))
        for key, row in list(world.neumann_index.items()):
            a = np.abs(row.value)
            world.neumann_index[key] = dataclasses.replace(row, value=a / a.sum(),
                                                           boundary_value=0.02 * row.boundary_value)

        class Positive(Boundary):
            def dirichlet(self, points):
                v, g = super().dirichlet(points)
                return 1.0 + 0.3 * v, 0.3 * g

        self.world = world
        self.plan = lower_world(world)
        boundary = Positive(self.nphys)
        self.bc = boundary_data_from_callables(self.plan, boundary.dirichlet, boundary.normal)
        rng = np.random.default_rng(3)
        self.fields = 1.0 + 0.2 * rng.uniform(size=(world.n_owners, self.nphys))


@pytest.fixture(scope="module")
def blocked_world():
    return _World()


def _close(a, b, rtol=1e-12):
    a, b = np.asarray(a), np.asarray(b)
    scale = max(float(np.max(np.abs(b))), 1e-300)
    assert a.shape == b.shape
    assert float(np.max(np.abs(a - b))) <= rtol * scale, (float(np.max(np.abs(a - b))), scale)


def test_blocked_p05_equals_unblocked(blocked_world):
    from drbx.native.fci_perpendicular_p05_operator import p05_terms
    w = blocked_world
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (1, 6)]
    kinds = (D,) * 8
    whole = p05_terms(w.plan, w.fields, w.bc, kinds, pairs)
    assert len(jr.greedy_blocks(pairs, 4)) == 3                              # really blocked
    blocked = jr.blocked_p05_terms(w.plan, w.fields, w.bc, kinds, pairs, column_block=4)
    for name in ("centered_numerator", "jump_numerator", "face_jump"):
        assert isinstance(getattr(blocked, name), np.ndarray)
        _close(getattr(blocked, name), getattr(whole, name))
        assert np.max(np.abs(getattr(blocked, name))) > 0
    assert abs(blocked.antisymmetry - float(whole.antisymmetry)) <= 1e-13


def test_blocked_p05n_equals_unblocked(blocked_world):
    from drbx.native.fci_perpendicular_p05_operator import p05n_action
    w = blocked_world
    columns = [0, 1, 2, 3, 0, 1]                             # roles: physical fields 0, 1 appear as N and D
    kinds = (N, N, D, N, D, D)
    n_pairs = [(0, 1), (2, 3), (0, 3)]
    d_pairs = [(4, 5), (2, 3), (4, 3)]
    whole = p05n_action(w.plan, w.fields, w.bc, kinds, n_pairs, d_pairs, columns=columns)
    assert len(jr.greedy_blocks([set(n) | set(d) for n, d in zip(n_pairs, d_pairs)], 5)) > 1
    blocked = jr.blocked_p05n_action(w.plan, w.fields, w.bc, kinds, n_pairs, d_pairs, columns=columns,
                                     column_block=5)
    for name in ("raw_N_numerator", "raw_D_numerator", "face_N_numerator", "face_D_numerator"):
        _close(getattr(blocked, name), getattr(whole, name))
        assert np.max(np.abs(getattr(blocked, name))) > 0
    assert abs(blocked.antisymmetry - float(whole.antisymmetry)) <= 1e-13


def test_blocked_p06n_equals_unblocked(blocked_world):
    from drbx.native.fci_perpendicular_p06_operator import bc_columns, p06_action, p06n_layout
    w = blocked_world
    recs = (SimpleNamespace(columns=np.arange(5), field_kinds=(D,) * 5),
            SimpleNamespace(columns=np.array([0, 5, 2, 6, 7]), field_kinds=(D, N, N, D, N)))
    columns, kinds, groups = p06n_layout(recs)
    whole = p06_action(w.plan, w.fields[:, columns], bc_columns(w.bc, columns), kinds, groups)
    assert len(jr.greedy_blocks([set(zip(r.columns.tolist(), r.field_kinds)) for r in recs], 8, 1)) == 2
    blocked = jr.blocked_p06n_action(w.plan, w.fields, w.bc, recs, column_block=8, variant_block=1)
    for name in ("material_numerator", "remainder_numerator", "correction_numerator"):
        assert getattr(blocked, name).shape == (2, w.world.n_owners, 4)
        _close(getattr(blocked, name), getattr(whole, name), rtol=1e-11)


def test_blocked_p07_numerator_equals_unblocked(blocked_world):
    w = blocked_world
    kinds = (N, D, N, N, D, N, D, D)
    whole = jr.JaxOwnerClosure._p07_numerator(SimpleNamespace(plan=w.plan), w.fields, w.bc, kinds)
    blocked = jr.JaxOwnerClosure._p07_numerator(SimpleNamespace(plan=w.plan, column_block=5), w.fields, w.bc, kinds)
    assert blocked.shape == whole.shape == (w.world.n_owners, 8)
    _close(blocked, whole)
    assert np.max(np.abs(whole)) > 0


def test_closure_option_validation():
    with pytest.raises(ValueError, match="at least 5"):
        jr.JaxOwnerClosure(SimpleNamespace(n=8, t=None), None, ("p05",), {}, plan="plan", column_block=4)
    with pytest.raises(ValueError, match="lowered plan or the built owner rows"):
        jr.JaxOwnerClosure(SimpleNamespace(n=8, t=None), None, ("p05",), {})
