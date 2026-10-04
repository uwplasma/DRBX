"""Fast tests of the P08 step-5 compact_c3 re-freeze campaign (``scripts/p08_step5_compact_c3``): the pinned four-key
contract in the configuration (and its drift refusal), the 5.3 scientific contract copied unchanged, the inherited step-4
settings, the identity (configuration, inherited settings, sources, sidecar, oracle entries) and the resume refusals
(identity, oracle root, artifact identity, artifact options), the folder layout, the four-key options reaching
``run_full_build`` / ``build_environment`` / the closure check / every stage call, the two preflights (closure before the
build, owner-subset gate after the N32 build), the run flow, ``validate`` and ``analyze`` on synthetic outputs.

Fully synthetic / ``tmp_path`` based (the heavy functions are monkeypatched); the real bounded preflight is
``tests/test_p08_step5_compact_c3_campaign_real.py`` (removed 4 October 2026: it needed the oracle arrays retired on 2 October) (slow).
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

from p_shared import build_artifact as ba                                          # noqa: E402
from p_shared import runner                                                         # noqa: E402
from p08_step4_global import campaign as step4_campaign                             # noqa: E402
from p08_step5_combined import campaign as step5_campaign                          # noqa: E402
from p08_step5_combined import checkpoints, jaxstage, references, reduction         # noqa: E402
from p08_step5_compact_c3 import campaign                                           # noqa: E402
from tests.test_p08_step5_combined_campaign import _write_grid                      # noqa: E402

FINAL = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius",
         "bfield_toroidal": "compact_c3"}
THREE = {k: v for k, v in FINAL.items() if k != "bfield_toroidal"}


# ---------------------------------------------------------------------------
# Configuration and sources
# ---------------------------------------------------------------------------
def test_config_pins_the_four_options_and_the_scientific_contract():
    cfg = campaign.config()
    assert cfg["schema"] == "drbx.p08-step5-compact-c3-v1" == campaign.SCHEMA
    assert cfg["resolutions"] == [32, 48, 64]
    assert cfg["operator_options"] == FINAL == campaign.operator_options() == campaign.OPERATOR_OPTIONS
    assert cfg["params"] == {"rho_star": 0.05, "tau": 1.0,
                             "diffusion": {"density": 0.01, "Te": 0.01, "Ti": 0.01, "vorticity": 0.01}}
    assert tuple(cfg["variants"]) == ("main_phi_dirichlet", "heldout_phi_dirichlet", "dirichlet_rich",
                                      "control_constant_dirichlet")
    assert cfg["phi_solve_rtol"] == 1e-11 and cfg["phi_solve_rtol_production_default"] == 1e-8
    assert (cfg["phi_solve_boundary"], cfg["phi_solve_preconditioner"], cfg["phi_solve_factor_dtype"]) == \
           ("dirichlet", "block_jacobi_eta_plane", "float32")
    assert cfg["order_criterion"]["informational"] is True and cfg["consistency_tolerance"] == 1e-8
    assert "2 October 2026" in cfg["re_freeze"]["decision"] and "compact_c3" in cfg["re_freeze"]["decision"]
    assert "main_phi_neumann" in cfg["deferred"]["neumann_phi_variants"]


def test_config_is_the_step_5_3_contract_plus_the_fourth_option():
    ours, theirs = campaign.config(), step5_campaign.config()
    assert theirs["operator_options"] == THREE and ours["operator_options"] == {**THREE, "bfield_toroidal": "compact_c3"}
    own_keys = {"schema", "roadmap_step", "operator_options", "re_freeze", "inherited_from_step4", "note"}
    for key in set(theirs) - own_keys:
        assert ours[key] == theirs[key], key                       # catalogue, variants, params, phi solve, gates ...
    assert set(ours) - set(theirs) == {"re_freeze", "inherited_from_step4"}


@pytest.mark.parametrize("name,value", [("OPERATOR_OPTIONS", THREE),
                                        ("OPERATOR_OPTIONS", {**FINAL, "bfield_toroidal": "spline"}),
                                        ("OPERATOR_OPTIONS", {**FINAL, "face_quadrature": "q3"}),
                                        ("PARAMS", {"rho_star": 0.1, "tau": 1.0, "diffusion": {}}),
                                        ("VARIANTS", ("main_phi_dirichlet",)),
                                        ("PHI_SOLVE", {**campaign.PHI_SOLVE, "phi_solve_rtol": 1e-8}),
                                        ("INHERITED_KEYS", ("campaigns",))])
def test_config_refuses_drift_from_the_pinned_literals(monkeypatch, name, value):
    monkeypatch.setattr(campaign, name, value)
    with pytest.raises(ValueError, match="differ from the pinned|differs from the pinned"):
        campaign.config()


def test_inherited_settings_come_from_step_4_not_from_a_copy():
    inh = campaign.inherited_settings()
    merged = step4_campaign.config()
    assert set(inh) == set(campaign.INHERITED_KEYS)
    for key, value in inh.items():
        assert value == merged[key], key
    assert inh["cell_chunk_size"] == 4096 and inh["face_chunk_size"] == 2048 and inh["p07_chunk_size"] == 2048
    assert inh["campaigns"] == ["p05", "p05n_frozen", "p05n_upwind", "p06n", "p06_legacy", "p07", "p07n"]
    assert inh["preflight_floor_seeds"] == [0, 1] and inh["wall_cache"] is False
    assert "operator_options" not in inh                           # step 4's three-key options are never inherited


def test_source_hashes_cover_the_package_5_3_step_4_and_the_c3_evaluator_sources():
    hashes = campaign.source_hashes()
    assert len(hashes) == len(campaign.SOURCE_FILES) == len(set(campaign.SOURCE_FILES))
    for rel, digest in hashes.items():
        assert (REPO / rel).is_file(), rel
        assert len(digest) == 64
    for needed in (*step5_campaign.SOURCE_FILES, *step4_campaign.SOURCE_FILES,
                   "scripts/p08_step5_compact_c3/campaign.py", "scripts/p08_step5_compact_c3/configuration.json",
                   "scripts/p_shared/build_artifact.py", "scripts/p_shared/bfield.py", "scripts/p_shared/provider.py",
                   "src/drbx/geometry/Bfield_evaluator.py", "src/drbx/geometry/compact_toroidal.py",
                   "src/drbx/geometry/jax_bfield_evaluator.py", "scripts/p08_step4_global/campaign.py",
                   "scripts/p08_step4_global/configuration.json", "scripts/p08_step1_global/oracle_manifest.json"):
        assert needed in hashes


def test_folder_layout_helpers(tmp_path):
    assert campaign.sidecar_path(tmp_path) == tmp_path / "localized_sidecar.json"
    assert campaign.artifact_root(tmp_path) == tmp_path / "artifact"
    assert campaign.inputs_path(tmp_path) == tmp_path / "provenance" / "inputs.json"
    # the 5.3 modules put their outputs under <output> when output == step4 == <OUT>
    assert references.references_dir(tmp_path, 48) == tmp_path / "N48" / "references"
    assert references.work_dir(tmp_path) == tmp_path / "work"


# ---------------------------------------------------------------------------
# Synthetic world: workspace, sidecar, oracle files, committed manifest
# ---------------------------------------------------------------------------
def _fake_world(tmp_path, monkeypatch, *, input_files=()):
    workspace = tmp_path / "workspace"
    sidecar = workspace / step4_campaign.config()["canonical_sidecar_relative"]
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"schema": "x", "metric_cache": {"path": "x"}, "makegrid": {"path": "x"},
                                   "artifact": {"path": "x"}}))
    files = []
    for rel, data in input_files:
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files.append({"path": rel, "bytes": len(data), "sha256": runner.sha256_file(path)})
    monkeypatch.setattr(step4_campaign, "_input_manifest", lambda: {"files": files})
    entries = {}
    other = {}
    for n in (32, 48, 64):
        path = workspace / "work" / "p06n" / f"N{n}.owner_values.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, values=np.arange(4.0) + n)
        raw = workspace / "work" / "p06n" / f"N{n}.raw.npz"
        np.savez(raw, x=np.arange(3.0))
        entries[str(n)] = {"files": [
            {"path": f"work/p06n/N{n}.owner_values.npz", "bytes": path.stat().st_size, "sha256": runner.sha256_file(path)},
            {"path": f"work/p06n/N{n}.raw.npz", "bytes": raw.stat().st_size, "sha256": runner.sha256_file(raw)}],
            "missing": []}
        p05 = workspace / "work" / "p05" / f"N{n}.owner_results.npz"
        p05.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p05, v=np.zeros(2) + n)
        other[str(n)] = {"files": [{"path": f"work/p05/N{n}.owner_results.npz", "bytes": p05.stat().st_size,
                                    "sha256": runner.sha256_file(p05)}], "missing": []}
    manifest = {"schema": "x", "workspace_root": "", "campaigns": {"p06n": entries, "p05": other}}
    monkeypatch.setattr(step4_campaign, "committed_oracle_manifest", lambda: manifest)
    return workspace


def _verify(workspace, output, grids=(32, 48, 64), oracle_root=None, closure=False):
    return campaign.verify_inputs(input_root=workspace, oracle_root=oracle_root, output=output, grids=list(grids),
                                  closure_oracles=closure)


def _identity_json(options=FINAL, **extra):
    return {"policy": ba.build_policy(**options), "n": 32, **extra}


def _write_artifact(output, n, identity=None):
    identity = identity or _identity_json(n=n)
    gdir = Path(output) / "artifact" / f"N{n}"
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "build_identity.json").write_text(json.dumps(identity))
    (gdir / "manifest.json").write_text(json.dumps({"n": n}))
    (gdir / "build_receipt.json").write_text(json.dumps({
        "n": n, "identity": identity, "wall_seconds": 12.5, "cpu_seconds": 100.0, "peak_rss_gib": 2.5,
        "effective_workers": 4, "row_counts": {"cells": 1, "faces": 2, "neumann": 3, "p07": 4},
        "bytes_per_row_kind": {"cells": 10, "faces": 20, "neumann": 5, "p07": 15},
        "diagnostics": {"point_row_families": {"coupled_quartic": 7}}}))
    return identity


# ---------------------------------------------------------------------------
# verify-inputs / identity / resume refusals
# ---------------------------------------------------------------------------
def test_verify_inputs_records_provenance_and_the_identity_is_stable(tmp_path, monkeypatch):
    workspace = _fake_world(tmp_path, monkeypatch, input_files=[("a/in.bin", b"abc"), ("b/in2.bin", b"defg")])
    identity, record = _verify(workspace, tmp_path / "out")
    assert _verify(workspace, tmp_path / "out")[0] == identity
    saved = json.loads((tmp_path / "out" / "provenance" / "inputs.json").read_text())
    assert saved["identity"] == identity and saved["campaign"] == "p08_step5_compact_c3"
    assert saved["operator_options"] == FINAL and saved["configuration"]["operator_options"] == FINAL
    assert saved["inherited"] == campaign.inherited_settings() and saved["input_manifest_files"] == 2
    assert saved["grids"] == {} and set(saved["oracle_p06n_owner_values"]) == {"32", "48", "64"}
    assert saved["jax"]["backend"] == "cpu" and saved["jax"]["x64"] is True
    assert (tmp_path / "out" / "localized_sidecar.json").is_file()
    assert saved["localized_sidecar_sha256"] == runner.sha256_file(tmp_path / "out" / "localized_sidecar.json")
    assert record["identity"] == identity


def test_identity_depends_on_configuration_inherited_settings_and_sources(tmp_path, monkeypatch):
    workspace = _fake_world(tmp_path, monkeypatch)
    one = _verify(workspace, tmp_path / "one")[0]
    real_config, real_inherited = campaign.config, campaign.inherited_settings
    monkeypatch.setattr(campaign, "config", lambda: {**real_config(), "consistency_tolerance": 1e-3})
    two = _verify(workspace, tmp_path / "two")[0]
    with pytest.raises(ValueError, match="campaign identity changed"):
        _verify(workspace, tmp_path / "one")
    monkeypatch.setattr(campaign, "config", real_config)
    monkeypatch.setattr(campaign, "inherited_settings", lambda cfg=None: {**real_inherited(), "cell_chunk_size": 1024})
    three = _verify(workspace, tmp_path / "three")[0]
    monkeypatch.setattr(campaign, "inherited_settings", real_inherited)
    monkeypatch.setattr(campaign, "SOURCE_FILES", campaign.SOURCE_FILES[:-1])
    four = _verify(workspace, tmp_path / "four")[0]
    assert len({one, two, three, four}) == 4


def test_a_changed_sidecar_or_oracle_entry_changes_the_identity(tmp_path, monkeypatch):
    workspace = _fake_world(tmp_path, monkeypatch)
    one = _verify(workspace, tmp_path / "one")[0]
    sidecar = workspace / step4_campaign.config()["canonical_sidecar_relative"]
    sidecar.write_text(json.dumps({"schema": "y", "metric_cache": {"path": "x"}, "makegrid": {"path": "x"},
                                   "artifact": {"path": "x"}}))
    assert _verify(workspace, tmp_path / "two")[0] != one
    with pytest.raises(ValueError, match="localized reference sidecar changed"):
        _verify(workspace, tmp_path / "one")
    manifest = step4_campaign.committed_oracle_manifest()
    changed = json.loads(json.dumps(manifest))
    changed["campaigns"]["p06n"]["48"]["files"][0]["sha256"] = "0" * 64
    monkeypatch.setattr(step4_campaign, "committed_oracle_manifest", lambda: changed)
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify(workspace, tmp_path / "three")


def test_verify_inputs_checks_immutable_inputs_oracle_files_oracle_root_and_backend(tmp_path, monkeypatch):
    workspace = _fake_world(tmp_path, monkeypatch, input_files=[("a/in.bin", b"abc")])
    _verify(workspace, tmp_path / "out")
    (workspace / "a" / "in.bin").write_bytes(b"abd")                       # same size, other hash
    with pytest.raises(ValueError, match="missing or changed immutable input"):
        _verify(workspace, tmp_path / "o1")
    (workspace / "a" / "in.bin").write_bytes(b"abc")
    with pytest.raises(ValueError, match="oracle verification failed"):    # no oracle files under that root
        _verify(workspace, tmp_path / "o2", oracle_root=tmp_path / "elsewhere")
    import shutil
    shutil.copytree(workspace / "work", tmp_path / "elsewhere" / "work")
    with pytest.raises(ValueError, match="oracle root changed"):           # intact copy, but another root
        _verify(workspace, tmp_path / "out", oracle_root=tmp_path / "elsewhere")
    np.savez(workspace / "work" / "p06n" / "N48.owner_values.npz", values=np.zeros(4))      # corrupted oracle file
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify(workspace, tmp_path / "o3")
    _verify(workspace, tmp_path / "o4", grids=(32,))                       # N48 is not requested
    with pytest.raises(ValueError, match="not a directory"):
        _verify(tmp_path / "nowhere", tmp_path / "o5")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with pytest.raises(ValueError, match="CPU backend required"):
        _verify(workspace, tmp_path / "o6", grids=(32,))


def test_closure_oracles_of_all_campaigns_are_checked_for_the_preflight_only(tmp_path, monkeypatch):
    workspace = _fake_world(tmp_path, monkeypatch)
    (workspace / "work" / "p05" / "N32.owner_results.npz").unlink()          # a non-p06n N32 oracle file
    _verify(workspace, tmp_path / "ok", grids=(32,))                        # run does not read it
    with pytest.raises(ValueError, match="p05 N32: missing"):
        _verify(workspace, tmp_path / "pre", grids=(32,), closure=True)


def test_artifacts_are_recorded_and_a_changed_identity_or_wrong_options_refused(tmp_path, monkeypatch):
    workspace = _fake_world(tmp_path, monkeypatch)
    out = tmp_path / "out"
    identity, record = _verify(workspace, out)
    assert record["grids"] == {}
    _write_artifact(out, 32)
    identity2, record = _verify(workspace, out, grids=(32,))
    assert identity2 == identity and set(record["grids"]) == {"32"}
    rec = record["grids"]["32"]
    assert rec["artifact_identity_sha256"] == runner.digest(json.loads((out / "artifact/N32/build_identity.json").read_text()))
    assert rec["artifact_dir"] == str((out / "artifact" / "N32").resolve())
    # the provenance keeps grids across invocations that do not request them
    assert set(_verify(workspace, out, grids=(48,))[1]["grids"]) == {"32"}
    # a different artifact identity of the same grid: refused (new folder required)
    _write_artifact(out, 32, _identity_json(n=32, other="source-hash-changed"))
    with pytest.raises(ValueError, match="artifact identity of N32 changed; use a new output folder"):
        _verify(workspace, out)
    # an artifact built with the three-key (spline) options is not accepted at all
    _write_artifact(tmp_path / "old", 48, _identity_json(THREE, n=48))
    with pytest.raises(ValueError, match="not the pinned"):
        _verify(workspace, tmp_path / "old")


def test_record_artifact_updates_the_provenance_and_refuses_changes(tmp_path, monkeypatch):
    workspace = _fake_world(tmp_path, monkeypatch)
    out = tmp_path / "out"
    _, inputs = _verify(workspace, out)
    with pytest.raises(ValueError, match="not assembled"):
        campaign.record_artifact(out, inputs, 32)
    _write_artifact(out, 32)
    rec = campaign.record_artifact(out, inputs, 32)
    assert inputs["grids"]["32"] == rec
    assert json.loads(campaign.inputs_path(out).read_text())["grids"]["32"] == rec
    _write_artifact(out, 32, _identity_json(n=32, other="x"))
    with pytest.raises(ValueError, match="changed"):
        campaign.record_artifact(out, inputs, 32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_parsing():
    base = ["--input-root", "/x", "--output", "/y"]
    a = campaign.parse(["run", *base])
    assert a.resolutions == [32, 48, 64] and a.workers == 4 and a.oracle_root is None and a.stage is None
    a = campaign.parse(["preflight", *base, "--workers", "96", "--oracle-root", "/o", "--memory-budget-gib", "400",
                        "--worker-memory-gib", "3.5", "--memory-reserve-gib", "8", "--max-tasks-per-worker", "50",
                        "--resolutions", "32"])
    assert a.workers == 96 and a.oracle_root == Path("/o") and a.resolutions == [32]
    assert (a.memory_budget_gib, a.worker_memory_gib, a.memory_reserve_gib, a.max_tasks_per_worker) == (400.0, 3.5, 8.0, 50)
    for stage in ("artifact", "references", "jax", "reduce"):
        a = campaign.parse(["run-stage", *base, "--stage", stage, "--n", "48"])
        assert (a.stage, a.n) == (stage, 48)
    for command in ("verify-inputs", "validate", "analyze"):
        assert campaign.parse([command, *base]).command == command
    for bad in (["run", "--resolutions", "16"], ["frobnicate"], ["run-stage", "--stage", "cells"],
                ["run", "--max-units", "2"], ["run", "--step4-campaign", "/s"]):
        with pytest.raises(SystemExit):
            campaign.parse([*bad, *base])


def test_main_arguments_are_checked_and_verify_inputs_locks_and_records(tmp_path, monkeypatch, capsys):
    workspace = _fake_world(tmp_path, monkeypatch)
    argv = ["--input-root", str(workspace)]
    with pytest.raises(ValueError, match="--stage and --n are required"):
        campaign.main(["run-stage", *argv, "--output", str(tmp_path / "o1")])
    with pytest.raises(ValueError, match="unique and ascending"):
        campaign.main(["verify-inputs", *argv, "--output", str(tmp_path / "o1"), "--resolutions", "48", "32"])
    with pytest.raises(ValueError, match="not among the verified"):
        campaign.main(["run-stage", *argv, "--output", str(tmp_path / "o1"), "--stage", "jax", "--n", "64",
                       "--resolutions", "32"])
    out = tmp_path / "out"
    campaign.main(["verify-inputs", *argv, "--output", str(out)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "verify-inputs" and payload["status"] == "complete"
    assert (out / ".runner.lock").exists() and list((out / "invocations").glob("*_verify-inputs.json"))
    assert json.loads((out / "last_exit.json").read_text())["status"] == "complete"
    assert (out / "provenance" / "inputs.json").is_file() and (out / "localized_sidecar.json").is_file()


# ---------------------------------------------------------------------------
# Artifact stage: the four-key options reach run_full_build; the build identity must record them
# ---------------------------------------------------------------------------
def _stage_args(output, workspace="/ws", **over):
    base = dict(output=Path(output), input_root=Path(workspace), oracle_root=None, workers=3, memory_budget_gib=None,
                worker_memory_gib=None, memory_reserve_gib=1.0, max_tasks_per_worker=7, resolutions=[32, 48, 64])
    return SimpleNamespace(**{**base, **over})


def test_artifact_stage_passes_the_four_options_the_inherited_chunk_sizes_and_the_folder_layout(tmp_path, monkeypatch):
    out = tmp_path / "out"
    inputs = {"grids": {}}
    calls = []

    def fake_build(**kw):
        calls.append(kw)
        identity = _write_artifact(out, kw["n"], _identity_json(n=kw["n"]))
        return {"n": kw["n"], "identity": identity, "wall_seconds": 3.0}

    monkeypatch.setattr(ba, "run_full_build", fake_build)
    monkeypatch.setattr(campaign.build_artifact_mod, "run_full_build", fake_build)
    receipt = campaign.artifact_stage(n=48, args=_stage_args(out), identity="ID", inputs=inputs)
    (kw,) = calls
    assert {k: kw[k] for k in FINAL} == FINAL and kw["bfield_toroidal"] == "compact_c3"
    assert kw["n"] == 48 and kw["workers"] == 3 and kw["max_tasks_per_worker"] == 7
    assert kw["output"] == out / "artifact" and kw["sidecar_path"] == out / "localized_sidecar.json"
    inh = campaign.inherited_settings()
    for key in ("cell_chunk_size", "face_chunk_size", "p07_chunk_size", "geometry_raw_chunk_size",
                "geometry_face_chunk_size"):
        assert kw[key] == inh[key] == step4_campaign.config()[key]
    assert receipt["artifact_identity_sha256"] == inputs["grids"]["48"]["artifact_identity_sha256"]
    assert json.loads(campaign.inputs_path(out).read_text())["grids"]["48"] == inputs["grids"]["48"]
    # memory-capped pools follow step 1's rule
    campaign.artifact_stage(n=48, args=_stage_args(out, workers=96, memory_budget_gib=20.0, worker_memory_gib=4.0),
                            identity="ID", inputs=inputs)
    assert calls[-1]["workers"] == 4


def test_artifact_stage_refuses_a_build_that_does_not_record_the_c3_evaluator(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(campaign.build_artifact_mod, "run_full_build",
                        lambda **kw: {"identity": _identity_json(THREE), "wall_seconds": 1.0})
    with pytest.raises(ValueError, match="not the pinned"):
        campaign.artifact_stage(n=32, args=_stage_args(out), identity="ID", inputs={"grids": {}})
    monkeypatch.setenv(ba.CSR_ONLY_ENV, "1")
    with pytest.raises(ValueError, match="tensor-encoded"):
        campaign.artifact_stage(n=32, args=_stage_args(out), identity="ID", inputs={"grids": {}})


# ---------------------------------------------------------------------------
# Preflight 1: the closure check with the four options (no build)
# ---------------------------------------------------------------------------
def _closure_payload(**over):
    payload = {"all_diff_pass": True, "diff_failures": [], "uniq_mismatches": [], "diff_table": [{}] * 29,
               "oracle_jax": [{"campaign": "p05", "term": "t", "pass": False, "ratio_to_oracle_NR": 2.5},
                              {"campaign": "p06n", "term": "u", "pass": True, "ratio_to_oracle_NR": None}],
               "plan": {"cells": 12}, "blocking": {"column_block": 8}, "peak_rss_gib": 1.5, "wall_seconds": 4.0,
               "curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius",
               "bfield_toroidal": "compact_c3"}
    return {**payload, **over}


def _fake_env(method="compact_c3", provenance="compact_c3", env_option="compact_c3"):
    ref = SimpleNamespace(provenance={"bfield_toroidal": provenance} if provenance else {},
                          bfield_evaluator=SimpleNamespace(toroidal_method=method))
    return SimpleNamespace(ref=ref, bfield_toroidal=env_option)


def _closure_world(monkeypatch, tmp_path, payload=None, **env_kw):
    from p_shared import jax_replay as jr
    from p_shared import replay_support as rs
    seen = {"env": [], "closure": []}
    monkeypatch.setattr(rs, "build_environment", lambda **kw: seen["env"].append(kw) or _fake_env(**env_kw))

    def fake_closure(**kw):
        seen["closure"].append(kw)
        return payload if payload is not None else _closure_payload()

    monkeypatch.setattr(jr, "run_jax_owner_closure_check", fake_closure)
    return seen


def _preflight(tmp_path, **kw):
    return campaign.preflight_grid(n=32, input_root=Path("/ws"), sidecar=tmp_path / "side.json", output=tmp_path,
                                   paths={"p": Path("/o")})


def test_closure_preflight_passes_the_four_options_inherited_blocking_and_records_the_evaluator(monkeypatch, tmp_path):
    seen = _closure_world(monkeypatch, tmp_path)
    case = _preflight(tmp_path)
    assert case["all_pass"] is True and case["options_recorded"] is True and case["operator_options"] == FINAL
    (env_kw,), (cl_kw,) = seen["env"], seen["closure"]
    assert {k: env_kw[k] for k in FINAL} == FINAL and {k: cl_kw[k] for k in FINAL} == FINAL
    inh = campaign.inherited_settings()
    assert cl_kw["campaigns"] == tuple(inh["campaigns"]) and cl_kw["floor_seeds"] == (0, 1)
    assert (cl_kw["column_block"], cl_kw["variant_block"], cl_kw["boundary_batch"]) == (8, 2, 32768)
    assert cl_kw["output"] == tmp_path / "preflight" / "N32_owner_closure.json"
    assert case["bfield"] == {"env_ref_provenance_bfield_toroidal": "compact_c3",
                              "evaluator_toroidal_method": "compact_c3", "env_bfield_toroidal": "compact_c3",
                              "ok": True, "environment_seconds": case["bfield"]["environment_seconds"]}
    assert case["jax"]["backend"] == "cpu" and case["jax"]["x64"] is True
    assert case["policy_rows"] == 29 and case["oracle_informational"]["gate"] is False
    assert case["oracle_informational"]["rows_not_matching_frozen_oracles"] == [("p05", "t")]   # does not gate
    assert case["oracle_informational"]["max_ratio_to_oracle_NR"] == 2.5


@pytest.mark.parametrize("payload_over,env_kw", [
    ({"all_diff_pass": False, "diff_failures": ["x"]}, {}),
    ({"uniq_mismatches": [1]}, {}),
    ({"bfield_toroidal": None}, {}),                         # payload silently lacks the fourth key -> "spline"
    ({"inner_support": "profile7"}, {}),
    ({}, {"method": "spline"}),                               # the evaluator is not the compact C3 one
    ({}, {"provenance": None}),
    ({}, {"env_option": "spline"})])
def test_closure_preflight_gates_on_policy_rows_structure_recorded_options_and_the_evaluator(
        monkeypatch, tmp_path, payload_over, env_kw):
    payload = _closure_payload(**payload_over)
    if payload_over.get("bfield_toroidal", 1) is None:
        del payload["bfield_toroidal"]
    _closure_world(monkeypatch, tmp_path, payload=payload, **env_kw)
    assert _preflight(tmp_path)["all_pass"] is False


def test_closure_preflight_requires_cpu_and_x64(monkeypatch, tmp_path):
    _closure_world(monkeypatch, tmp_path)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with pytest.raises(ValueError, match="CPU backend required"):
        _preflight(tmp_path)


def test_preflight_command_is_resumable_and_run_is_gated(tmp_path, monkeypatch, capsys):
    workspace = _fake_world(tmp_path, monkeypatch)
    argv = ["--input-root", str(workspace)]
    out = tmp_path / "out"
    calls = []

    def fake_preflight_grid(*, n, input_root, sidecar, output, paths):
        calls.append((n, sidecar))
        return {"n": n, "all_pass": len(calls) > 1}

    monkeypatch.setattr(campaign, "preflight_grid", fake_preflight_grid)
    with pytest.raises(SystemExit):
        campaign.main(["preflight", *argv, "--output", str(out)])
    assert calls == [(32, out / "localized_sidecar.json")]
    assert json.loads((out / "last_exit.json").read_text())["status"] == "failed_gates"
    with pytest.raises(ValueError, match="preflight did not pass"):
        campaign.main(["run", *argv, "--output", str(out), "--resolutions", "32"])
    capsys.readouterr()
    campaign.main(["preflight", *argv, "--output", str(out)])                # failed one is recomputed
    campaign.main(["preflight", *argv, "--output", str(out)])                # passed one is not redone
    assert len(calls) == 2 and json.loads((out / "preflight.json").read_text())["all_pass"] is True
    with pytest.raises(ValueError, match="requires a matching preflight"):
        campaign.main(["run", *argv, "--output", str(tmp_path / "fresh"), "--resolutions", "32"])


# ---------------------------------------------------------------------------
# Preflight 2: the owner-subset gate on the freshly built N32 artifact
# ---------------------------------------------------------------------------
def test_artifact_gate_grid_wires_the_four_options_the_campaign_folder_and_the_5_3_verdict(tmp_path, monkeypatch):
    from p_shared import owner_closure as oc
    from p_shared import replay_support as rs

    out = tmp_path / "out"
    seen = {}
    env = SimpleNamespace(t=SimpleNamespace(), census=None)
    monkeypatch.setattr(rs, "build_environment", lambda **kw: seen.setdefault("env", kw) and env)
    monkeypatch.setattr(oc, "select_owners", lambda t, census: {"a": 3, "b": 1, "c": 3})
    monkeypatch.setattr(references, "_prr", lambda: SimpleNamespace(p06n_state=lambda e, name: ("state", name)))
    monkeypatch.setattr(references, "reference_chunk",
                        lambda e, owners, *, states, params: seen.update(owners=list(owners), states=states) or {"r": 1})
    prep = SimpleNamespace(owner_volume=np.ones(5), setup_seconds={}, plan_summary={"p": 1})
    monkeypatch.setattr(jaxstage, "prepare", lambda **kw: seen.setdefault("prepare", kw) and prep)
    vinfo = {"gates": {"finite": True}, "consistency": {"relative_error_M": 1e-12}}
    monkeypatch.setattr(jaxstage, "run_variant", lambda p, v, cfg, omega_rhs=None: seen.update(variant=v, omega=omega_rhs)
                        or ({"a": 1}, vinfo))
    monkeypatch.setattr(step5_campaign, "preflight_checks",
                        lambda **kw: seen.update(verdict_kw=kw) or {"checks": {"finite": True}, "all_pass": True,
                                                                     "n_owners_subset": 2, "n_minus_o_psi_subset": 1e-3,
                                                                     "prescribed_sanity_worst": 1e-3,
                                                                     "arm_difference_worst": 1e-9})
    inputs = {"grids": {"32": {"artifact_identity_sha256": "a" * 64}}}
    case = campaign.artifact_gate_grid(n=32, args=_stage_args(out, oracle_root=Path("/o")), identity="ID", inputs=inputs)
    assert case["all_pass"] is True and case["variant"] == "main_phi_dirichlet"
    assert {k: seen["env"][k] for k in FINAL} == FINAL and seen["env"]["sidecar_path"] == out / "localized_sidecar.json"
    assert {k: seen["prepare"][k] for k in ("options", "step4", "inputs")} == {"options": FINAL, "step4": out, "inputs": inputs}
    assert seen["prepare"]["env"] is env and seen["owners"] == [1, 3] and seen["omega"] is None
    assert seen["states"] == {"main": ("state", "main_phi_dirichlet")}
    assert seen["verdict_kw"]["fs"] == "main" and seen["verdict_kw"]["cfg"]["operator_options"] == FINAL
    detail = json.loads((out / "preflight" / "N32_artifact_subset.json").read_text())
    assert detail["operator_options"] == FINAL and detail["subset_owners"] == [1, 3]


def test_artifact_gate_is_resumable_and_bound_to_the_artifact(tmp_path, monkeypatch):
    out = tmp_path / "out"
    inputs = {"grids": {"32": {"artifact_identity_sha256": "A"}}}
    calls = []
    monkeypatch.setattr(campaign, "artifact_gate_grid",
                        lambda **kw: calls.append(kw) or {"n": 32, "all_pass": len(calls) > 1})
    with pytest.raises(ValueError, match="artifact of N48 is not built|artifact of N32 is not built"):
        campaign.artifact_gate(args=_stage_args(out), identity="ID", inputs={"grids": {}})
    assert campaign.artifact_gate(args=_stage_args(out), identity="ID", inputs=inputs)["all_pass"] is False
    assert campaign.artifact_gate(args=_stage_args(out), identity="ID", inputs=inputs)["all_pass"] is True   # failed: redone
    assert campaign.artifact_gate(args=_stage_args(out), identity="ID", inputs=inputs)["all_pass"] is True   # passed: kept
    assert len(calls) == 2
    saved = json.loads((out / "preflight_artifact.json").read_text())
    assert saved["identity"] == "ID" and saved["artifact_identity_sha256"] == "A" and saved["all_pass"] is True
    other = {"grids": {"32": {"artifact_identity_sha256": "B"}}}                # another artifact: gate is redone
    campaign.artifact_gate(args=_stage_args(out), identity="ID", inputs=other)
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# Stage calls and the run flow
# ---------------------------------------------------------------------------
def test_stage_calls_get_the_four_options_the_campaign_folder_as_step4_and_its_sidecar(tmp_path, monkeypatch):
    out = tmp_path / "out"
    seen = {}
    monkeypatch.setattr(references, "references_stage", lambda **kw: seen.setdefault("references", kw) and {"n_owners": 1})
    monkeypatch.setattr(jaxstage, "jax_stage", lambda **kw: seen.setdefault("jax", kw) and {"computed": [], "resumed": []})
    monkeypatch.setattr(reduction, "reduce_grid", lambda **kw: seen.setdefault("reduce", kw) and {"gates": {}})
    inputs = {"grids": {"64": {"artifact_identity_sha256": "A"}}}
    args = _stage_args(out, oracle_root=Path("/oracles"))
    campaign.references_stage(n=64, args=args, identity="ID")
    campaign.jax_stage(n=64, args=args, identity="ID", inputs=inputs)
    campaign.reduce_stage(n=64, args=args, identity="ID", inputs=inputs)
    r, j, d = seen["references"], seen["jax"], seen["reduce"]
    assert r["options"] == j["options"] == FINAL and r["cfg"] == j["cfg"] == d["cfg"] == campaign.config()
    assert r["sidecar_path"] == out / "localized_sidecar.json" and r["output"] == j["output"] == d["output"] == out
    assert j["step4"] == out and j["inputs"] is inputs and d["inputs"] is inputs
    assert j["paths"] == step4_campaign.oracle_paths(Path("/oracles"), Path("/ws")) and r["workers"] == 3
    assert r["max_tasks_per_worker"] == 7
    for fn in (campaign.jax_stage, campaign.reduce_stage):
        with pytest.raises(ValueError, match="N32 is not built"):
            fn(n=32, args=args, identity="ID", inputs=inputs)


def _stub_stages(monkeypatch, order, *, gate_pass=True, grid_pass=lambda n: True):
    monkeypatch.setattr(campaign, "artifact_stage", lambda *, n, args, identity, inputs:
                        order.append(("artifact", n)) or {"wall_seconds": 1.0, "reentry": None})
    monkeypatch.setattr(campaign, "artifact_gate", lambda *, args, identity, inputs:
                        order.append(("gate", 32)) or {"all_pass": gate_pass})
    monkeypatch.setattr(campaign, "references_stage", lambda *, n, args, identity:
                        order.append(("references", n)) or {"n_owners": 5, "skipped": False})
    monkeypatch.setattr(campaign, "jax_stage", lambda *, n, args, identity, inputs:
                        order.append(("jax", n)) or {"computed": ["a"], "resumed": []})
    monkeypatch.setattr(campaign, "reduce_stage", lambda *, n, args, identity, inputs:
                        order.append(("reduce", n)) or {"gates": {"grid_pass": grid_pass(n), "solver_gates_pass": True,
                                                                  "all_finite": True}})


def test_run_grid_order_and_the_gate_only_at_n32(tmp_path, monkeypatch):
    order = []
    _stub_stages(monkeypatch, order)
    args = _stage_args(tmp_path / "out")
    grid = campaign.run_grid(n=32, args=args, identity="ID", inputs={})
    assert order == [("artifact", 32), ("gate", 32), ("references", 32), ("jax", 32), ("reduce", 32)]
    assert grid["grid_pass"] is True and grid["status"] == "complete" and grid["artifact_gate_pass"] is True
    order.clear()
    grid = campaign.run_grid(n=64, args=args, identity="ID", inputs={})
    assert order == [("artifact", 64), ("references", 64), ("jax", 64), ("reduce", 64)]
    assert grid["variants_computed"] == ["a"] and "artifact_gate_pass" not in grid


def test_a_failing_artifact_gate_stops_before_the_expensive_stages(tmp_path, monkeypatch):
    order = []
    _stub_stages(monkeypatch, order, gate_pass=False)
    grid = campaign.run_grid(n=32, args=_stage_args(tmp_path / "out"), identity="ID", inputs={})
    assert order == [("artifact", 32), ("gate", 32)]
    assert grid["grid_pass"] is False and grid["status"] == "failed_artifact_gate"


def test_run_flow_runs_the_grids_in_order_stops_after_a_failing_grid_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    workspace = _fake_world(tmp_path, monkeypatch)
    argv = ["--input-root", str(workspace)]
    out = tmp_path / "out"
    identity, _ = _verify(workspace, out)
    runner.write_json(out / "preflight.json", {"identity": identity, "all_pass": True, "cases": {"32": {"all_pass": True}}})
    order = []
    _stub_stages(monkeypatch, order, grid_pass=lambda n: n != 48)
    done = []
    monkeypatch.setattr(campaign, "validate", lambda **kw: done.append(("validate", kw["args"].resolutions))
                        or {"solver_gates_pass": True, "all_finite": True})
    monkeypatch.setattr(campaign, "analyze", lambda **kw: done.append(("analyze", kw["args"].resolutions)))
    with pytest.raises(SystemExit):
        campaign.main(["run", *argv, "--output", str(out)])
    assert [o for o in order if o[0] == "artifact"] == [("artifact", 32), ("artifact", 48)]   # N64 is not built
    last = json.loads((out / "last_exit.json").read_text())
    assert last["status"] == "failed_gates" and last["summary"]["operational_complete"] is False
    assert [g["n"] for g in last["summary"]["grids"]] == [32, 48]
    assert done == [("validate", [32, 48]), ("analyze", [32, 48])]
    # all grids pass: 32 -> 48 -> 64, validate + analyze on all, exit zero
    order.clear()
    done.clear()
    _stub_stages(monkeypatch, order)
    campaign.main(["run", *argv, "--output", str(out)])
    assert [o for o in order if o[0] == "artifact"] == [("artifact", 32), ("artifact", 48), ("artifact", 64)]
    assert [o for o in order if o[0] == "gate"] == [("gate", 32)]
    assert done == [("validate", [32, 48, 64]), ("analyze", [32, 48, 64])]
    assert json.loads((out / "last_exit.json").read_text())["summary"]["operational_complete"] is True
    capsys.readouterr()


def test_run_stage_dispatch(tmp_path, monkeypatch, capsys):
    workspace = _fake_world(tmp_path, monkeypatch)
    argv = ["--input-root", str(workspace), "--output", str(tmp_path / "out")]
    order = []
    _stub_stages(monkeypatch, order)
    for stage, n in (("artifact", 48), ("references", 48), ("jax", 48), ("reduce", 48), ("artifact", 32)):
        campaign.main(["run-stage", *argv, "--stage", stage, "--n", str(n)])
        assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["stage"] == stage
    assert order == [("artifact", 48), ("references", 48), ("jax", 48), ("reduce", 48), ("artifact", 32), ("gate", 32)]
    # a failing N32 gate after run-stage artifact exits nonzero
    _stub_stages(monkeypatch, order, gate_pass=False)
    with pytest.raises(SystemExit):
        campaign.main(["run-stage", *argv, "--stage", "artifact", "--n", "32"])
    assert json.loads((tmp_path / "out" / "last_exit.json").read_text())["status"] == "failed_gates"


# ---------------------------------------------------------------------------
# validate / analyze on synthetic outputs
# ---------------------------------------------------------------------------
def _synthetic_run(output, *, exponent=2.0, gates_ok=True, options=FINAL, gate=True, identity="ID"):
    """Artifacts of the three grids (receipts), the closure preflight, the N32 gate and the three reduced grids."""
    cfg = campaign.config()
    inputs = {"grids": {}}
    for n in (32, 48, 64):
        art_identity = _write_artifact(output, n, _identity_json(options, n=n))
        inputs["grids"][str(n)] = {"artifact_identity_sha256": runner.digest(art_identity)}
    for n in (32, 48, 64):
        e = 1e-2 * (32.0 / n) ** exponent
        _write_grid(output, cfg, n, err_presc=e, err_solved=2 * e, err_psi=1e-3 * (32.0 / n) ** 2, gates_ok=gates_ok,
                    inputs=inputs, identity=identity)
        reduction.reduce_grid(n=n, cfg=cfg, output=output, identity=identity, inputs=inputs)
    runner.write_json(output / "preflight.json", {"identity": identity, "all_pass": True, "cases": {"32": {"all_pass": True}}})
    if gate:
        runner.write_json(output / "preflight_artifact.json", {
            "identity": identity, "n": 32, "artifact_identity_sha256": inputs["grids"]["32"]["artifact_identity_sha256"],
            "case": {"all_pass": True}, "all_pass": True})
    return inputs


def _vargs(output, grids=(32, 48, 64)):
    return SimpleNamespace(output=Path(output), resolutions=list(grids))


def test_validate_records_the_gates_artifacts_and_the_informational_order(tmp_path):
    inputs = _synthetic_run(tmp_path)
    payload = campaign.validate(args=_vargs(tmp_path), identity="ID", inputs=inputs)
    saved = json.loads((tmp_path / "validation.json").read_text())
    assert saved == json.loads(json.dumps(checkpoints.jsonable(payload)))
    assert saved["identity"] == "ID" and saved["acceptance_gate"] is None and saved["user_decides_acceptance"] is True
    assert saved["operator_options"] == FINAL and saved["resolutions"] == [32, 48, 64]
    assert saved["solver_gates_pass"] is True and saved["all_finite"] is True and saved["grids_pass"] is True
    assert saved["preflight_closure_pass"] is True and saved["artifact_gate_pass"] is True
    assert saved["headline_order_criterion"]["pass"] is True and saved["headline_order_criterion"]["informational"] is True
    assert saved["scope"] == {"step4_smoke_replay_against_frozen_oracles": False, "neumann_phi_variants": False}
    art = saved["artifacts"]["48"]
    assert art["policy"]["bfield_toroidal"] == "compact_c3" and art["policy"]["inner_support"] == "fixed_radius"
    assert art["coupled_quartic"] == 7 and art["row_bytes_total"] == 50 and art["build_peak_rss_gib"] == 2.5
    assert art["path"] == str((tmp_path / "artifact" / "N48").resolve())
    assert saved["grids"]["48"]["solver"]["main_phi_dirichlet"]["iterations"] == 55
    assert saved["grids"]["64"]["grid_pass"] is True and saved["grids"]["64"]["artifact_options_ok"] is True
    summary = json.loads((tmp_path / "summary" / "artifacts.json").read_text())
    assert summary["operator_options"] == FINAL and set(summary["artifacts"]) == {"32", "48", "64"}


def test_validate_requires_the_preflights_and_a_matching_artifact(tmp_path):
    inputs = _synthetic_run(tmp_path, gate=False)
    with pytest.raises(ValueError, match="artifact owner-subset gate has not passed"):
        campaign.validate(args=_vargs(tmp_path), identity="ID", inputs=inputs)
    campaign.validate(args=_vargs(tmp_path, (48, 64)), identity="ID", inputs=inputs)       # N32 not requested
    with pytest.raises(ValueError, match="requires a matching preflight"):
        campaign.validate(args=_vargs(tmp_path, (48,)), identity="OTHER", inputs=inputs)
    runner.write_json(tmp_path / "preflight.json", {"identity": "ID", "all_pass": False, "cases": {"32": {"all_pass": False}}})
    with pytest.raises(ValueError, match="preflight did not pass"):
        campaign.validate(args=_vargs(tmp_path, (48,)), identity="ID", inputs=inputs)
    # a gate recorded for another artifact does not count
    inputs = _synthetic_run(tmp_path / "b")
    other = {"grids": {**inputs["grids"], "32": {"artifact_identity_sha256": "different"}}}
    with pytest.raises(ValueError, match="artifact owner-subset gate has not passed"):
        campaign.validate(args=_vargs(tmp_path / "b"), identity="ID", inputs=other)
    with pytest.raises(ValueError, match="artifact of N48 is not built"):
        campaign.validate(args=_vargs(tmp_path / "b", (48,)), identity="ID", inputs={"grids": {}})


def test_validate_refuses_an_artifact_with_other_options_or_a_reduction_on_another_artifact(tmp_path):
    inputs = _synthetic_run(tmp_path / "a")
    _write_artifact(tmp_path / "a", 48, _identity_json(THREE, n=48))                        # spline-era artifact
    with pytest.raises(ValueError, match="not the pinned"):
        campaign.validate(args=_vargs(tmp_path / "a"), identity="ID", inputs=inputs)
    inputs = _synthetic_run(tmp_path / "b")
    changed = {"grids": {**inputs["grids"], "64": {"artifact_identity_sha256": "different"}}}
    with pytest.raises(ValueError, match="differs from provenance/inputs.json"):
        campaign.validate(args=_vargs(tmp_path / "b"), identity="ID", inputs=changed)
    # the artifact on disk matches the record but the reduction was made with another recorded artifact
    inputs = _synthetic_run(tmp_path / "c")
    inputs_other = {"grids": {**inputs["grids"]}}
    gdir = tmp_path / "c" / "artifact" / "N64"
    identity = {**json.loads((gdir / "build_identity.json").read_text()), "rebuilt": True}
    _write_artifact(tmp_path / "c", 64, identity)
    inputs_other["grids"]["64"] = {"artifact_identity_sha256": runner.digest(identity)}
    with pytest.raises(ValueError, match="computed on another artifact"):
        campaign.validate(args=_vargs(tmp_path / "c"), identity="ID", inputs=inputs_other)


def test_validate_records_failing_solver_gates_and_a_failing_order_without_raising(tmp_path):
    inputs = _synthetic_run(tmp_path / "gates", gates_ok=False)
    saved = campaign.validate(args=_vargs(tmp_path / "gates"), identity="ID", inputs=inputs)
    assert saved["solver_gates_pass"] is False and saved["grids_pass"] is False and saved["all_finite"] is True
    inputs = _synthetic_run(tmp_path / "order", exponent=1.0)
    saved = campaign.validate(args=_vargs(tmp_path / "order"), identity="ID", inputs=inputs)
    assert saved["grids_pass"] is True and saved["headline_order_criterion"]["pass"] is False   # informational only


def test_analyze_writes_the_5_3_report_into_the_campaign_summary_folder(tmp_path):
    _synthetic_run(tmp_path)
    payload = campaign.analyze(args=_vargs(tmp_path), identity="ID")
    assert (tmp_path / "summary" / "step5_combined_report.md").is_file()
    saved = json.loads((tmp_path / "summary" / "step5_combined_summary.json").read_text())
    assert saved["operator_options"] == FINAL and saved["identity"] == "ID" and saved["grids"] == [32, 48, 64]
    assert payload["headline_order_criterion"]["pass"] is True
    assert str(FINAL["bfield_toroidal"]) in (tmp_path / "summary" / "step5_combined_report.md").read_text()
    with pytest.raises(ValueError, match="another campaign identity"):
        campaign.analyze(args=_vargs(tmp_path), identity="OTHER")
