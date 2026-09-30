"""Tests for ``scripts/p08_inner_support_eval`` (CLI parsing, identity and refusal, resume, validation, dispatch invariants).

Fast and synthetic: no HSX inputs, no jax, no geometry.  The heavy steps (input hashing of the real manifest, the sidecar
localization, the ``run.py`` subprocess) are replaced by tmp_path fixtures or tiny stand-in commands.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]              # .../DRBX
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p08_inner_support_eval import campaign, dispatch, sample_build   # noqa: E402
from p_shared.inner_support import INNER_SUPPORT_CHOICES            # noqa: E402

CANDS = {"C1": "last_aggregate", "C2": "last_aggregate_nearest28", "C3": "fixed_radius"}


# ---------------------------------------------------------------------------
# package contents and configuration
# ---------------------------------------------------------------------------
def test_configuration_and_vendored_files():
    cfg = campaign.config()
    assert cfg["candidates"] == CANDS
    assert set(cfg["candidates"].values()) <= set(INNER_SUPPORT_CHOICES)
    assert cfg["baseline_inner_support"] == "profile7"
    assert cfg["grids"] == [32, 48, 64] and cfg["per_ring"] == 12
    assert cfg["preflight"]["grid"] == 32 and cfg["preflight"]["per_ring"] == 1
    assert len(cfg["bands"]) == 6 and all(lo < hi for lo, hi in cfg["bands"])
    assert cfg["subprocess_env"] == {"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "true", "OMP_NUM_THREADS": "1",
                                     "VECLIB_MAXIMUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    package = SCRIPTS / "p08_inner_support_eval"
    assert sample_build.Q_SAMPLE == package / "sample.json"          # vendored, package-relative
    assert sample_build.Q_SAMPLE.stat().st_size < 100_000
    sample = json.loads(sample_build.Q_SAMPLE.read_text())
    assert {z["N"] for z in sample} == {32, 48, 64}
    assert (package / "README.md").is_file() and (package / "__init__.py").is_file()


def test_source_files_exist_and_no_workspace_paths():
    hashes = campaign.source_hashes()
    assert set(hashes) == set(campaign.SOURCE_FILES) and all(len(h) == 64 for h in hashes.values())
    package = SCRIPTS / "p08_inner_support_eval"
    for name in ("run.py", "sample_build.py", "q_fields.py", "campaign.py"):
        text = (package / name).read_text()
        assert "DEFAULT_SIDECAR" not in text and "'work'" not in text and '"work"' not in text
        assert "parents[1] / 'DRBX'" not in text and "HERE.parents[1]" not in text


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def test_parse_candidates():
    assert campaign.parse_candidates("C2") == {"C2": "last_aggregate_nearest28"}
    assert campaign.parse_candidates("C1=last_aggregate, C3") == {"C1": "last_aggregate", "C3": "fixed_radius"}
    assert campaign.parse_candidates("X9=any_aggregate") == {"X9": "any_aggregate"}
    for bad in ("", "C9", "C1=nonsense", "C1=profile7", "C1,C1", "C1=fixed_radius,C3", "C0=fixed_radius", "1x=fixed_radius"):
        with pytest.raises(ValueError):
            campaign.parse_candidates(bad)


def test_parse_grids():
    assert campaign.parse_grids("32,48,64") == (32, 48, 64)
    assert campaign.parse_grids("64") == (64,)
    for bad in ("", "16", "32,32", "a,b", "32,50"):
        with pytest.raises(ValueError):
            campaign.parse_grids(bad)


def test_cli_defaults_and_required_arguments(capsys):
    ap = campaign.build_parser()
    a = ap.parse_args(["run", "--output", "o", "--input-root", "r"])
    assert a.fn is campaign.cmd_run and a.jobs == 1 and a.per_ring == 12 and a.grids == "32,48,64"
    assert campaign.parse_candidates(a.candidates) == CANDS           # the default is C1, C2, C3
    a = ap.parse_args(["run", "--output", "o", "--input-root", "r", "--jobs", "4", "--grids", "32", "--per-ring", "3",
                       "--candidates", "C2"])
    assert (a.jobs, a.grids, a.per_ring, a.candidates) == (4, "32", 3, "C2")
    a = ap.parse_args(["preflight", "--output", "o", "--input-root", "r", "--candidates", "C2"])
    assert a.fn is campaign.cmd_preflight and a.candidates == "C2"
    assert not hasattr(a, "per_ring")                                 # the preflight per-ring is fixed at 1
    assert ap.parse_args(["verify-inputs", "--output", "o", "--input-root", "r"]).fn is campaign.cmd_verify_inputs
    assert ap.parse_args(["validate", "--output", "o"]).fn is campaign.cmd_validate
    assert ap.parse_args(["analyze", "--output", "o"]).fn is campaign.cmd_analyze
    for argv in (["run", "--output", "o"], ["run", "--input-root", "r"], ["run", "--output", "o", "--input-root", "r", "--jobs", "0"],
                 ["bogus"], []):
        with pytest.raises(SystemExit):
            ap.parse_args(argv)
    capsys.readouterr()


def test_main_reports_bad_arguments_without_traceback(capsys):
    rc = campaign.main(["run", "--output", "o", "--input-root", "r", "--grids", "16"])
    assert rc == 2 and "grids must be" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def _input_root(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    files = {"a.bin": b"alpha" * 10, "sub/b.bin": b"beta" * 7}
    recs = []
    for rel, data in files.items():
        (root / rel).write_bytes(data)
        recs.append(dict(path=rel, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return root, {"files": recs}


def test_verify_input_files_detects_missing_resized_and_changed(tmp_path):
    root, manifest = _input_root(tmp_path)
    assert campaign.verify_input_files(root, manifest, say=lambda m: None) == 2
    (root / "a.bin").write_bytes(b"x" * 50)                           # same size, other content
    with pytest.raises(ValueError, match="a.bin"):
        campaign.verify_input_files(root, manifest, say=lambda m: None)
    assert campaign.verify_input_files(root, manifest, hash_files=False, say=lambda m: None) == 2   # size only
    (root / "a.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="a.bin"):
        campaign.verify_input_files(root, manifest, hash_files=False, say=lambda m: None)
    (root / "sub/b.bin").unlink()
    with pytest.raises(ValueError, match="b.bin"):
        campaign.verify_input_files(root, manifest, say=lambda m: None)


def test_ensure_inputs_hashes_once_per_stamp(tmp_path, monkeypatch):
    root, manifest = _input_root(tmp_path)
    monkeypatch.setattr(campaign, "load_input_manifest", lambda: (manifest, "m" * 64))
    calls = []
    real = campaign.sha256_file
    monkeypatch.setattr(campaign, "sha256_file", lambda p: calls.append(p) or real(p))
    out = tmp_path / "out"
    campaign.ensure_inputs(root, out, say=lambda m: None)
    assert len(calls) == 2 and (out / "inputs_verified.json").is_file()
    campaign.ensure_inputs(root, out, say=lambda m: None)
    assert len(calls) == 2                                            # stamp: size check only
    campaign.ensure_inputs(root, out, force=True, say=lambda m: None)
    assert len(calls) == 4
    (root / "a.bin").write_bytes(b"y" * 50)                           # a hash-only change is caught by a forced verify
    with pytest.raises(ValueError):
        campaign.ensure_inputs(root, out, force=True, say=lambda m: None)


# ---------------------------------------------------------------------------
# identity and refusal
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_inputs(tmp_path, monkeypatch):
    state = dict(manifest_sha="m" * 64, sidecar_body={"metric": 1})

    def localize(input_root, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "localized_sidecar.json"
        body = dict(state["sidecar_body"], root=str(input_root))
        if path.exists() and json.loads(path.read_text()) != body:
            raise ValueError("localized reference sidecar changed")
        path.write_text(json.dumps(body))
        return path

    monkeypatch.setattr(campaign, "load_input_manifest", lambda: ({"files": []}, state["manifest_sha"]))
    monkeypatch.setattr(campaign, "localize_sidecar", localize)
    return state


def test_prepare_campaign_writes_manifest_and_refuses_identity_changes(tmp_path, fake_inputs):
    out, root = tmp_path / "out", tmp_path / "root"
    kw = dict(input_root=root, folder=out, candidates={"C2": "last_aggregate_nearest28"}, per_ring=12, say=lambda m: None)
    identity, sidecar = campaign.prepare_campaign(**kw)
    man = json.loads((out / "campaign_manifest.json").read_text())
    assert man["identity"] == identity and sidecar == out / "localized_sidecar.json"
    assert man["candidates"] == {"C2": "last_aggregate_nearest28"} and man["per_ring"] == 12
    assert man["input_manifest_sha256"] == "m" * 64
    assert man["localized_sidecar_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert man["configuration"] == campaign.config()
    assert {"commit", "dirty", "dirty_sources", "source_hashes"} <= set(man)
    assert campaign.prepare_campaign(**kw)[0] == identity             # the same identity reuses the folder
    for change in (dict(per_ring=6), dict(candidates={"C3": "fixed_radius"}),
                   dict(candidates={"C2": "last_aggregate_nearest28", "C3": "fixed_radius"})):
        with pytest.raises(ValueError, match="identity changed"):
            campaign.prepare_campaign(**{**kw, **change})
    fake_inputs["manifest_sha"] = "n" * 64                            # other input manifest
    with pytest.raises(ValueError, match="identity changed"):
        campaign.prepare_campaign(**kw)
    fake_inputs["manifest_sha"] = "m" * 64
    with pytest.raises(ValueError, match="new output folder"):        # other input root -> other localized sidecar
        campaign.prepare_campaign(**{**kw, "input_root": tmp_path / "elsewhere"})
    fake_inputs["sidecar_body"] = {"metric": 2}
    with pytest.raises(ValueError, match="new output folder"):
        campaign.prepare_campaign(**kw)


def test_identity_depends_on_sources(tmp_path, fake_inputs, monkeypatch):
    kw = dict(input_root=tmp_path, candidates={"C2": "last_aggregate_nearest28"}, per_ring=12, say=lambda m: None)
    a, _ = campaign.prepare_campaign(folder=tmp_path / "a", **kw)
    real = campaign.source_hashes()
    monkeypatch.setattr(campaign, "source_hashes", lambda: {**real, campaign.SOURCE_FILES[0]: "0" * 64})
    b, _ = campaign.prepare_campaign(folder=tmp_path / "b", **kw)
    assert a != b
    with pytest.raises(ValueError, match="identity changed"):
        campaign.prepare_campaign(folder=tmp_path / "a", **kw)


# ---------------------------------------------------------------------------
# tasks, resume, validation (synthetic outputs)
# ---------------------------------------------------------------------------
IDENT = "f" * 64


def _write_task(folder, cand, n, *, identity=IDENT, per_ring=12, failures=None, violations=0, log_file=True,
                support=None, same_keys=True):
    support = support or CANDS[cand]
    p = campaign.task_paths(folder, cand, n)
    p["dir"].mkdir(parents=True, exist_ok=True)
    common = dict(n=n, identity=identity, per_ring=per_ring, inner_support=support, candidate=cand,
                  failures=failures or {})
    p["results"].write_text(json.dumps(dict(common, dispatch_check=dict(n_violations=violations, same_keys=same_keys,
                                                                        identical={"R1": 1}, changed={"R1": 2}))))
    p["receipt"].write_text(json.dumps(dict(common, n_owners=5, wall_seconds=1.5, peak_rss_gib=0.5)))
    if log_file:
        p["log"].write_text("log\n")


def _check(folder, cand, n, **kw):
    return campaign.check_task(folder, cand, CANDS[cand], n, identity=kw.pop("identity", IDENT),
                               per_ring=kw.pop("per_ring", 12), **kw)


def test_plan_orders_finest_grid_first():
    tasks = campaign.plan(CANDS, (32, 64, 48))
    assert [t[2] for t in tasks] == [64] * 3 + [48] * 3 + [32] * 3
    assert [t[0] for t in tasks[:3]] == ["C1", "C2", "C3"] and tasks[1] == ("C2", "last_aggregate_nearest28", 64)


def test_check_task_conditions(tmp_path):
    _write_task(tmp_path, "C1", 32)
    assert _check(tmp_path, "C1", 32) == []
    assert _check(tmp_path, "C1", 48) and "missing" in _check(tmp_path, "C1", 48)[0]
    assert _check(tmp_path, "C1", 32, identity="0" * 64)
    assert _check(tmp_path, "C1", 32, per_ring=6)
    assert campaign.check_task(tmp_path, "C1", "fixed_radius", 32, identity=IDENT, per_ring=12)   # other support
    _write_task(tmp_path, "C2", 32, failures={"7": "RuntimeError"})
    assert any("failures" in q for q in _check(tmp_path, "C2", 32))
    assert _check(tmp_path, "C2", 32, strict=False) and any("failures" in q for q in _check(tmp_path, "C2", 32, strict=False))
    _write_task(tmp_path, "C3", 32, violations=3)
    assert any("3 violations" in q for q in _check(tmp_path, "C3", 32))
    assert _check(tmp_path, "C3", 32, strict=False) == []             # violations are a validate matter, not a resume matter
    _write_task(tmp_path, "C3", 48, log_file=False)
    assert any("run_N48.log" in q for q in _check(tmp_path, "C3", 48))
    assert _check(tmp_path, "C3", 48, strict=False) == []


def test_resume_skips_only_complete_matching_tasks(tmp_path):
    _write_task(tmp_path, "C1", 64)                                   # complete: skipped
    _write_task(tmp_path, "C2", 64, identity="0" * 64)                # other identity: rerun
    _write_task(tmp_path, "C3", 64, failures={"1": "x"})              # failures: rerun
    _write_task(tmp_path, "C1", 48, per_ring=6)                       # other per-ring: rerun
    calls = []

    def fake_runner(folder, cand, support, n, **kw):
        calls.append((cand, n, support, kw["identity"], kw["per_ring"]))
        _write_task(folder, cand, n)
        return dict(cand=cand, n=n, status="ok", returncode=0, seconds=0.0)

    tasks = campaign.plan(CANDS, (64, 48))
    res = campaign.run_tasks(tmp_path, tasks, jobs=2, per_ring=12, input_root=tmp_path, sidecar=tmp_path / "s.json",
                             identity=IDENT, runner=fake_runner)
    skipped = {(r["cand"], r["n"]) for r in res if r["status"] == "skipped"}
    assert skipped == {("C1", 64)}
    assert {(c, n) for c, n, *_ in calls} == {("C2", 64), ("C3", 64), ("C1", 48), ("C2", 48), ("C3", 48)}
    assert all(i == IDENT and pr == 12 for *_, i, pr in calls)
    assert len(res) == 6
    calls.clear()                                                     # everything is complete now
    res = campaign.run_tasks(tmp_path, tasks, jobs=2, per_ring=12, input_root=tmp_path, sidecar=tmp_path / "s.json",
                             identity=IDENT, runner=fake_runner)
    assert not calls and {r["status"] for r in res} == {"skipped"}


def test_run_task_subprocess_env_log_and_stale_output(tmp_path, monkeypatch):
    _write_task(tmp_path, "C1", 32)                                   # a stale output must be removed before the rerun
    probe = tmp_path / "probe.json"
    code = ("import json, os, sys; keys = ['JAX_PLATFORMS', 'JAX_ENABLE_X64', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', "
            "'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']; "
            "json.dump(dict({k: os.environ.get(k) for k in keys}, cwd=os.getcwd()), open(sys.argv[1], 'w')); print('hello')")
    seen = {}

    def fake_command(n, support, cand, **kw):
        seen.update(n=n, support=support, cand=cand, **kw)
        return [sys.executable, "-c", code, str(probe)]

    monkeypatch.setattr(campaign, "run_command", fake_command)
    r = campaign.run_task(tmp_path, "C1", "last_aggregate", 32, per_ring=12, input_root=tmp_path, sidecar=tmp_path / "s.json",
                          identity=IDENT)
    assert r["status"] == "ok" and r["returncode"] == 0
    env = json.loads(probe.read_text())
    assert {k: env[k] for k in campaign.config()["subprocess_env"]} == campaign.config()["subprocess_env"]
    assert Path(env["cwd"]).resolve() == SCRIPTS.resolve()
    p = campaign.task_paths(tmp_path, "C1", 32)
    assert "hello" in p["log"].read_text() and not p["results"].exists() and not p["receipt"].exists()
    assert seen["out"] == p["dir"] and seen["identity"] == IDENT
    monkeypatch.setattr(campaign, "run_command", lambda *a, **k: [sys.executable, "-c", "import sys; sys.exit(3)"])
    r = campaign.run_task(tmp_path, "C1", "last_aggregate", 32, per_ring=12, input_root=tmp_path, sidecar=tmp_path / "s.json",
                          identity=IDENT)
    assert r["status"] == "failed" and r["returncode"] == 3


def test_run_command_shape():
    cmd = campaign.run_command(64, "fixed_radius", "C3", per_ring=12, input_root=Path("/in"), sidecar=Path("/o/s.json"),
                               identity="abc", out=Path("/o/C3"))
    assert cmd[1:4] == ["-m", "p08_inner_support_eval.run", "64"]
    flags = dict(zip(cmd[4::2], cmd[5::2]))
    assert flags == {"--per-ring": "12", "--c1": "fixed_radius", "--candidate": "C3", "--input-root": "/in",
                     "--sidecar": "/o/s.json", "--identity": "abc", "--out": "/o/C3"}


def _manifest_folder(tmp_path, grids=(32,), **task_kw):
    man = dict(identity=IDENT, candidates=CANDS, per_ring=12)
    (tmp_path / "campaign_manifest.json").write_text(json.dumps(man))
    for cand in CANDS:
        for n in grids:
            _write_task(tmp_path, cand, n, **task_kw)


def test_validate_passes_and_reports_problems(tmp_path):
    _manifest_folder(tmp_path)
    assert campaign.main(["validate", "--output", str(tmp_path), "--grids", "32"]) == 0
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["passed"] and len(report["tasks"]) == 3 and report["problems"] == []
    assert campaign.main(["validate", "--output", str(tmp_path)]) == 1          # N48 and N64 are missing
    report = json.loads((tmp_path / "validation.json").read_text())
    assert not report["passed"] and sum("missing" in q for q in report["problems"]) >= 12
    _write_task(tmp_path, "C2", 32, violations=2)
    _write_task(tmp_path, "C3", 32, failures={"9": "boom"})
    campaign.main(["validate", "--output", str(tmp_path), "--grids", "32"])
    problems = json.loads((tmp_path / "validation.json").read_text())["problems"]
    assert any("C2 N32" in q and "2 violations" in q for q in problems) and any("C3 N32" in q and "failures" in q for q in problems)
    assert campaign.main(["validate", "--output", str(tmp_path / "nowhere")]) == 2


def test_analyze_refuses_an_incomplete_campaign(tmp_path, capsys):
    _manifest_folder(tmp_path)
    assert campaign.main(["analyze", "--output", str(tmp_path)]) == 2            # N48 and N64 missing
    assert "incomplete" in capsys.readouterr().err
    assert not (tmp_path / "summary" / "comparison.json").exists()


# ---------------------------------------------------------------------------
# dispatch invariants on fake rows
# ---------------------------------------------------------------------------
def prow(tag, *, family="ringwise", policy=None, bc=False):
    diag = {"family": family}
    if policy:
        diag["donor_policy"] = policy
    return SimpleNamespace(donor_ids=np.array([tag, tag + 1]), value=np.array([float(tag)]), gradient=np.zeros((1, 3)),
                           boundary_conditioned=bc, trace_target_points=np.zeros((1, 3)), diagnostics=diag)


def irow(tag, family):
    return SimpleNamespace(donor_ids=np.array([tag]), weights=np.array([1.0 + 0.0 * tag]), family=family)


CQ = "coupled_quartic"


def check(c0, cand, support="last_aggregate"):
    return dispatch.dispatch_check(c0, cand, candidate_support=support)


def test_dispatch_point_rows_identical_uncoupled():
    c0 = {("R1", 0): prow(1), ("R2", 1): prow(2, family="centered_radial", bc=True), ("R3", 2): prow(3)}
    cand = {k: prow(v.donor_ids[0], family=v.diagnostics["family"], bc=v.boundary_conditioned) for k, v in c0.items()}
    out = check(c0, cand)
    assert out["n_violations"] == 0 and out["same_keys"] and out["identical"] == {"R1": 1, "R2": 1, "R3": 1}
    assert out["changed"] == {}


def test_dispatch_point_rows_changed_rules():
    # uncoupled -> coupled and changed: allowed (re-dispatch), for any candidate
    assert check({("R1", 0): prow(1)}, {("R1", 0): prow(9, family=CQ)})["n_violations"] == 0
    # uncoupled in both but changed: violation
    out = check({("R2", 0): prow(1)}, {("R2", 0): prow(9)})
    assert out["n_violations"] == 1 and out["violations"][0]["kind"] == "R2" and out["changed"] == {"R2": 1}
    # coupled in both, changed, no nearest28 policy: violation; with the policy: allowed
    c0 = {("R3", 0): prow(1, family=CQ)}
    assert check(c0, {("R3", 0): prow(9, family=CQ)})["n_violations"] == 1
    assert check(c0, {("R3", 0): prow(9, family=CQ, policy="nearest28")}, "last_aggregate_nearest28")["n_violations"] == 0
    assert check(c0, {("R3", 0): prow(9, family=CQ, policy="other")})["n_violations"] == 1
    # coupled in both and identical: fine
    assert check(c0, {("R3", 0): prow(1, family=CQ)})["n_violations"] == 0
    # coupled in C0 but changed to an uncoupled row: violation (a changed row must be coupled in the candidate)
    assert check(c0, {("R3", 0): prow(9)})["n_violations"] == 1
    # the point-row policy is read from the rows, not from the candidate name
    assert check(c0, {("R3", 0): prow(9, family=CQ, policy="nearest28")}, "last_aggregate")["n_violations"] == 0


def test_dispatch_integrated_rows():
    c0 = {5: irow(1, 5), 6: irow(2, 6), 7: irow(3, 7), 8: irow(4, 4)}
    same = {5: irow(1, 5), 6: irow(2, 6), 7: irow(3, 7), 8: irow(4, 4)}
    out = check(c0, same)
    assert out["n_violations"] == 0 and out["identical"] == {"R4": 4}
    # family 5/6 re-dispatched to family 7 (changed): allowed
    redispatched = {**same, 5: irow(11, 7), 6: irow(12, 7)}
    out = check(c0, redispatched)
    assert out["n_violations"] == 0 and out["changed"] == {"R4": 2}
    # a changed row that stays uncoupled, or turns uncoupled from coupled
    assert check(c0, {**same, 8: irow(14, 4)})["n_violations"] == 1
    assert check(c0, {**same, 5: irow(11, 5)})["n_violations"] == 1
    assert check(c0, {**same, 7: irow(13, 6)})["n_violations"] == 1
    # family-7 rows change only for a nearest-28 candidate
    changed7 = {**same, 7: irow(13, 7)}
    assert check(c0, changed7, "last_aggregate")["n_violations"] == 1
    assert check(c0, changed7, "fixed_radius")["n_violations"] == 1
    assert check(c0, changed7, "last_aggregate_nearest28")["n_violations"] == 0
    # weights alone make a row non-identical
    bumped = irow(3, 7)
    bumped.weights = np.array([2.0])
    assert check(c0, {**same, 7: bumped}, "last_aggregate")["n_violations"] == 1


def test_dispatch_violation_list_is_capped_but_count_is_full():
    n = 35
    c0 = {("R1", i): prow(i) for i in range(n)}
    cand = {("R1", i): prow(1000 + i) for i in range(n)}              # every row changed while uncoupled
    out = check(c0, cand)
    assert out["n_violations"] == n and len(out["violations"]) == dispatch.MAX_LISTED == 20
    mixed = {**{("R1", i): prow(1000 + i) for i in range(25)}, **{("R1", i): prow(i) for i in range(25, n)}}
    assert check(c0, mixed)["n_violations"] == 25


def test_dispatch_key_mismatch_counts():
    c0 = {("R1", 0): prow(1), ("R1", 1): prow(2)}
    out = check(c0, {("R1", 0): prow(1), ("R1", 7): prow(5)})
    assert out["same_keys"] is False and out["n_violations"] == 2     # one row absent, one extra
    assert any(v["kind"] == "missing" for v in out["violations"])
    assert check({}, {})["n_violations"] == 0


def test_nearest28_support_detection():
    assert dispatch.is_nearest28_support("last_aggregate_nearest28")
    assert not any(dispatch.is_nearest28_support(s) for s in INNER_SUPPORT_CHOICES if s != "last_aggregate_nearest28")
