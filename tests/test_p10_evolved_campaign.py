"""P10 evolved MMS, chunk C7 (campaign driver): the run matrix, the input packing, the scheduler, the run-directory classification, the
variant comparisons, the launcher stages and an end-to-end synthetic campaign (verify -> preflight -> run -> validate in
subprocesses on the n = 16 synthetic bundle, with resume, a deliberately failing unit and an identity mismatch)."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from p10_evolved_mms import campaign as C                         # noqa: E402
from p10_evolved_mms import inputs as INP                         # noqa: E402
from p10_evolved_mms import reduce as RD                          # noqa: E402
from p10_evolved_mms import run_all as RA                         # noqa: E402

SCRIPTS = _REPO_ROOT / "scripts"
CONFIG = C.load_config()


@pytest.fixture(scope="module")
def units():
    return C.build_matrix(CONFIG)


# ---------------------------------------------------------------------------------------------------------------------
# 1. the matrix
# ---------------------------------------------------------------------------------------------------------------------
def test_matrix_counts_ids_and_steps(units):
    assert len(units) == 69
    assert len({u.id for u in units}) == 69 and len({u.out_dir for u in units}) == 69
    cnt = C.matrix_counts(units)
    assert cnt["by_stage"] == {"main": 60, "variant": 9}
    assert cnt["by_experiment"] == {"E1": 24, "E2": 12, "E3": 24, "dt_half": 2, "rtol_tight": 2, "e3d": 3, "w1": 2}
    assert cnt["by_arm"] == {"raw": 30, "filtered": 39}            # the variants are all filtered
    assert cnt["by_n"] == {"32": 24, "48": 21, "64": 24}
    assert cnt["by_source"] == {"continuum": 39, "discrete": 30}
    main = [u for u in units if u.stage == "main"]
    for arm in ("raw", "filtered"):
        for n in (32, 48, 64):
            assert sum(u.arm == arm and u.n == n for u in main) == 10          # E1 4 + E2 2 + E3 4
    modes = {(u.experiment, u.mode, u.pattern) for u in main}
    assert modes == {("E1", "diffusion", "NNN-D"), ("E1", "diffusion", "DDDD"), ("E2", "hyperbolic", "NNN-D"),
                     ("E3", "coupled", "NNN-D"), ("E3", "coupled", "DDDD")}
    assert {u.source for u in main if u.mode == "hyperbolic"} == {"continuum", "discrete"}
    T = CONFIG["T"]
    for u in main:
        assert u.nsteps == {32: 50, 48: 62, 64: 71}[u.n] and u.dt == T / u.nsteps and u.T == T
        assert u.params_override == {} and u.opts_override == {} and u.variant is None
        assert u.out_dir == f"main/{u.arm}/n{u.n}/{u.mode}/{u.pattern}/{u.source}"
        assert u.id == f"main.{u.arm}.n{u.n}.{u.mode}.{u.pattern}.{u.source}"


def test_matrix_variants(units):
    var = [u for u in units if u.stage == "variant"]
    assert all(u.arm == "filtered" and u.mode == "coupled" and u.pattern == "NNN-D" and u.source == "continuum" for u in var)
    by = {}
    for u in var:
        by.setdefault(u.variant, []).append(u)
        assert u.out_dir == f"variants/{u.variant}/filtered/n{u.n}/coupled/NNN-D/continuum"
        assert u.id == f"var.{u.variant}.filtered.n{u.n}.coupled.NNN-D.continuum"
    assert {k: sorted(x.n for x in v) for k, v in by.items()} == {"dt_half": [32, 64], "rtol_tight": [32, 64], "e3d": [32, 48, 64],
                                                                  "w1": [32, 64]}
    base = {32: 50, 48: 62, 64: 71}
    for u in by["dt_half"]:
        assert u.nsteps == 2 * base[u.n] and u.dt == CONFIG["T"] / u.nsteps and u.opts_override == {} and u.params_override == {}
    for u in by["rtol_tight"]:
        assert u.opts_override == {"phi_rtol": 1e-13} and u.params_override == {} and u.nsteps == base[u.n]
    for u in by["e3d"]:
        assert u.opts_override == {"curvature_jump_dissipation": True, "curvature_c_kappa": 1.0} and u.nsteps == base[u.n]
    for u in by["w1"]:
        assert u.params_override == {"w1": 1.0} and u.opts_override == {} and u.nsteps == base[u.n]
    # records are frozen and JSON round-trip
    with pytest.raises(dataclasses.FrozenInstanceError):
        var[0].n = 1
    assert C.Unit.from_json(json.loads(json.dumps(var[0].to_json()))) == var[0]


def test_main_and_variant_layouts_are_discovered_by_the_reducer(units, tmp_path):
    for u in units:
        u.path(tmp_path).mkdir(parents=True)
    found = RD.discover(tmp_path / "main")
    want = {(m, p) for m, p in (("diffusion", "NNN-D"), ("diffusion", "DDDD"), ("hyperbolic", "NNN-D"), ("coupled", "NNN-D"),
                                ("coupled", "DDDD"))}
    assert set(found) == {"raw", "filtered"}
    for arm in found:
        assert set(found[arm]) == want and all(ns == {32, 48, 64} for ns in found[arm].values())
    for name, ns in (("dt_half", {32, 64}), ("rtol_tight", {32, 64}), ("e3d", {32, 48, 64}), ("w1", {32, 64})):
        f = RD.discover(tmp_path / "variants" / name)
        assert set(f) == {"filtered"} and f["filtered"] == {("coupled", "NNN-D"): ns}


def test_config_override_and_manifest(tmp_path, units):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rho_star": 1.0}))
    with pytest.raises(ValueError, match="may only set"):
        C.load_config(bad)
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"resolutions": [16], "nsteps": {"16": 5}, "T": 1e-4}))
    cfg = C.load_config(ok)
    assert cfg["resolutions"] == [16] and cfg["T"] == 1e-4 and cfg["rho_star"] == CONFIG["rho_star"]
    m = C.build_manifest(CONFIG, units, spec=None, commit="abc", inputs_manifest_sha256="d" * 64, synthetic=False,
                         config_file_sha256=C.file_sha(C.CONFIG_PATH))
    assert m["configuration_sha256"] == C.canonical_sha(CONFIG) and m["git_commit"] == "abc" and len(m["units"]) == 69
    assert m["counts"]["total"] == 69 and m["inputs_manifest_sha256"] == "d" * 64
    assert [u.id for u in C.units_from_manifest(m)] == [u.id for u in units]
    assert C.manifest_difference(m, m) == []
    m2 = dict(m, inputs_manifest_sha256="e" * 64)
    assert C.manifest_difference(m, m2) == ["inputs_manifest_sha256"]
    # a unit's parameters come from the configuration, with the unit override on top
    kw = C.params_kwargs(CONFIG, {"w1": 1.0})
    assert kw["w1"] == 1.0 and kw["rho_star"] == 4.5e-4 and kw["D"] == [1.0e-5, 1.2e-5, 1.4e-5, 0.8e-5] and kw["a_omega"] == 2200.0


def test_read_git_commit(tmp_path):
    from p10_evolved_mms import bundle as bundle_mod

    assert C.read_git_commit() == bundle_mod.git_commit()                      # the logic of bundle.git_commit
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    assert C.read_git_commit(tmp_path) == "unknown"
    (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    assert C.read_git_commit(tmp_path) == "a" * 40
    (git / "HEAD").write_text("b" * 40 + "\n")                                  # detached
    assert C.read_git_commit(tmp_path) == "b" * 40
    assert C.read_git_commit(tmp_path / "nowhere") == "unknown"


# ---------------------------------------------------------------------------------------------------------------------
# 2. inputs
# ---------------------------------------------------------------------------------------------------------------------
def _stand_ins(tmp_path):
    """Small stand-in files at the source paths of every ``(arm, N)``; returns ``(nodal_roots, laplacian_root)``."""
    rng = np.random.default_rng(0)
    nodal = {arm: tmp_path / "src" / f"nodal_{arm}" for arm in ("raw", "filtered")}
    lap = tmp_path / "src" / "laplacian"
    for arm in nodal:
        for n in (32, 48, 64):
            for path in (nodal[arm] / f"N{n}" / "nodal_metric.npz", lap / arm / f"N{n}" / "laplacian_metric.npz"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(rng.bytes(64 + n))
    return nodal, lap


def test_input_paths_follow_bundle_metric_paths(tmp_path):
    from p10_evolved_mms import bundle as bundle_mod

    for f in INP.source_files():                                              # default roots (nothing is opened)
        want = bundle_mod.metric_paths(f["arm"], f["n"])[0 if f["kind"] == "nodal" else 1]
        assert Path(f["source"]) == want
    assert len(INP.source_files()) == 12
    dest = tmp_path / "packed"
    for arm in ("raw", "filtered"):
        for n in (32, 48, 64):
            got = bundle_mod.metric_paths(arm, n, nodal_root=INP.nodal_root(dest, arm), laplacian_root=INP.laplacian_root(dest))
            assert got == (dest / INP.rel_paths(arm, n)[0], dest / INP.rel_paths(arm, n)[1])


def test_pack_and_verify_inputs(tmp_path, capsys):
    nodal, lap = _stand_ins(tmp_path)
    dest = tmp_path / "packed"
    m = INP.pack_inputs(dest, nodal_roots=nodal, laplacian_root_=lap)
    assert len(m["files"]) == 12 and [e["path"] for e in m["files"]] == sorted(e["path"] for e in m["files"])
    assert {e["path"] for e in m["files"]} >= {"nodal/raw/N32/nodal_metric.npz", "laplacian/filtered/N64/laplacian_metric.npz"}
    for e in m["files"]:
        p = dest / e["path"]
        assert p.stat().st_size == e["bytes"] and INP.sha256_file(p) == e["sha256"]
    rep = INP.verify_inputs(dest, require=[("raw", 32), ("filtered", 64)])
    assert rep["ok"] and rep["n_files"] == 12 and rep["errors"] == []
    sha = INP.manifest_sha256(dest)
    INP.pack_inputs(dest, nodal_roots=nodal, laplacian_root_=lap)               # deterministic and idempotent
    assert INP.manifest_sha256(dest) == sha
    assert INP.file_shas(dest)[("raw", 32)]["nodal"] == next(e["sha256"] for e in m["files"] if e["path"] == "nodal/raw/N32/nodal_metric.npz")
    assert INP.main(["verify", str(dest)]) == 0
    # a pair that is not packed
    assert not INP.verify_inputs(dest, require=[("raw", 16)])["ok"]
    # tamper: one flipped byte, a truncated file, a missing file
    victim = dest / "nodal/raw/N32/nodal_metric.npz"
    orig = victim.read_bytes()
    data = bytearray(orig)
    data[3] ^= 0xFF
    victim.write_bytes(bytes(data))
    rep = INP.verify_inputs(dest)
    assert not rep["ok"] and any("sha256" in e and "nodal/raw/N32" in e for e in rep["errors"])
    assert INP.main(["verify", str(dest)]) == 1
    victim.write_bytes(bytes(data[:-5]))
    assert any("size" in e for e in INP.verify_inputs(dest)["errors"])
    victim.unlink()
    assert any("missing" in e for e in INP.verify_inputs(dest)["errors"])
    # an unlisted file is an extra, not an error
    victim.write_bytes(orig)
    (dest / "extra.npz").write_bytes(b"x")
    rep = INP.verify_inputs(dest)
    assert rep["ok"] and rep["extras"] == ["extra.npz"]
    # an unsafe manifest path and a missing manifest
    man = json.loads((dest / INP.MANIFEST).read_text())
    man["files"][0]["path"] = "../escape.npz"
    (dest / INP.MANIFEST).write_text(json.dumps(man))
    assert any("unsafe" in e for e in INP.verify_inputs(dest)["errors"])
    (dest / INP.MANIFEST).unlink()
    assert not INP.verify_inputs(dest)["ok"]


def test_pack_with_a_missing_source_copies_nothing(tmp_path):
    nodal, lap = _stand_ins(tmp_path)
    (lap / "filtered" / "N48" / "laplacian_metric.npz").unlink()
    dest = tmp_path / "packed"
    with pytest.raises(FileNotFoundError, match="N48"):
        INP.pack_inputs(dest, nodal_roots=nodal, laplacian_root_=lap)
    assert not dest.exists() or not any(dest.rglob("*.npz"))
    # the CLI packs a subset
    rc = INP.main(["pack", str(dest), "--arms", "raw", "--resolutions", "32", "--laplacian-root", str(lap),
                   "--nodal-root", f"raw={nodal['raw']}"])
    assert rc == 0 and len(json.loads((dest / INP.MANIFEST).read_text())["files"]) == 2


# ---------------------------------------------------------------------------------------------------------------------
# 3. the scheduler (pure)
# ---------------------------------------------------------------------------------------------------------------------
def test_priority_order_is_largest_first(units):
    order = C.order_units(units)
    assert order[0].id == "var.dt_half.filtered.n64.coupled.NNN-D.continuum"            # N64, most steps, coupled
    assert [u.n for u in order] == sorted((u.n for u in units), reverse=True)
    for n in (64, 48, 32):
        costs = [u.cost for u in order if u.n == n]
        assert costs == sorted(costs, reverse=True)
    n64 = [u for u in order if u.n == 64]
    assert [u.mode for u in n64 if u.stage == "main"] == sorted((u.mode for u in n64 if u.stage == "main"),
                                                               key=lambda m: -C.MODE_COST[m])
    assert [u.id for u in order] == [u.id for u in C.order_units(list(reversed(units)))]       # deterministic


def test_memory_table():
    assert C.parse_memory_table(None) == {32: 1.5, 48: 3.0, 64: 4.5}
    assert C.parse_memory_table("64=6,16=0.5") == {32: 1.5, 48: 3.0, 64: 6.0, 16: 0.5}
    t = C.DEFAULT_MEMORY_GB
    assert [C.unit_memory_gb(n, t) for n in (32, 48, 64)] == [1.5, 3.0, 4.5]
    assert C.unit_memory_gb(16, t) == 1.5 and C.unit_memory_gb(40, t) == 3.0 and C.unit_memory_gb(128, t) == 4.5


def test_can_start_rules():
    assert not C.can_start(0.0, 2, 1.0, 2, None)                                       # worker limit
    assert C.can_start(3.0, 1, 1.0, 2, None)                                           # no budget
    assert C.can_start(4.5, 1, 1.5, 3, 6.0) and not C.can_start(4.5, 1, 3.0, 3, 6.0)    # 6.0 within / 7.5 over
    assert C.can_start(0.0, 0, 9.0, 2, 6.0)                                            # a lone unit always starts


def test_simulated_schedule_respects_workers_and_memory_budget(units):
    order = C.order_units(units)
    mem = lambda u: C.unit_memory_gb(u.n)
    ev = C.simulate_schedule(order, 3, 6.0, mem)
    assert [e["id"] for e in ev] == [u.id for u in order]                              # head-of-line: starts in priority order
    assert all(e["in_use_gb"] <= 6.0 + 1e-12 and e["n_running"] <= 3 for e in ev)
    assert all(a["start"] <= b["start"] for a, b in zip(ev, ev[1:]))
    n64 = [e for e in ev if "n64" in e["id"]]
    for a in n64:                                                                      # two 4.5 GiB units never overlap under 6 GiB
        for b in n64:
            if a is not b:
                assert a["end"] <= b["start"] or b["end"] <= a["start"]
    assert max(e["n_running"] for e in C.simulate_schedule(order, 3, None, mem)) == 3
    assert max(e["n_running"] for e in C.simulate_schedule(order, 8, 9.0, mem)) <= 8
    assert max(e["in_use_gb"] for e in C.simulate_schedule(order, 8, 9.0, mem)) <= 9.0 + 1e-12
    lone = C.simulate_schedule(order[:3], 3, 3.0, mem)                                 # 4.5 GiB each, budget 3: one at a time
    assert [e["n_running"] for e in lone] == [1, 1, 1]


def test_pool_runs_subprocesses_within_workers_and_budget(tmp_path):
    class Item:
        def __init__(self, i, code=0):
            self.id, self.n, self.cost, self.code = f"t{i}", 32, 1.0, code

    def run(items, workers, budget):
        live, peak, done = [0], [0], {}

        def launch(it):
            live[0] += 1
            peak[0] = max(peak[0], live[0])
            return RA.spawn([sys.executable, "-c", f"import time, sys; a = bytearray(40_000_000); time.sleep(0.5); sys.exit({it.code})"],
                            os.environ, tmp_path / "pool.log", cwd=SCRIPTS)

        def finish(it, rc, gib, t0, t1):
            live[0] -= 1
            done[it.id] = (rc, gib)

        RA.run_pool(items, workers=workers, budget_gb=budget, mem_of=lambda it: 1.0, launch=launch, finish=finish)
        return peak[0], done

    peak, done = run([Item(i, 7 if i == 2 else 0) for i in range(4)], 2, None)
    assert peak == 2 and {k: v[0] for k, v in done.items()} == {"t0": 0, "t1": 0, "t2": 7, "t3": 0}
    assert all(v[1] > 0.01 for v in done.values())                                      # per-child peak RSS from wait4
    assert run([Item(i) for i in range(3)], 3, 1.5)[0] == 1                             # budget 1.5 GiB, 1 GiB each
    assert run([Item(i) for i in range(3)], 3, 2.0)[0] == 2


# ---------------------------------------------------------------------------------------------------------------------
# 4. run-directory classification
# ---------------------------------------------------------------------------------------------------------------------
SHA = "a" * 64


def _fake_run_dir(root, unit, *, sha=SHA, status="complete", dt=None, snapshots=True, step=None):
    d = unit.path(root)
    (d / "snapshots").mkdir(parents=True, exist_ok=True)
    snap_steps = [round(f * unit.nsteps) for f in (0.2, 0.4, 0.6, 0.8, 1.0)]
    step = unit.nsteps if step is None else step
    rec = {"schema": "drbx.p10-run-v1", "identity": {"bundle_sha256": sha, "arm": unit.arm, "n": unit.n}, "mode": unit.mode,
           "source": unit.source, "pattern": unit.pattern, "params": C.params_kwargs(CONFIG, unit.params_override),
           "dt": unit.dt if dt is None else dt, "T": unit.T, "nsteps": unit.nsteps, "chunk": unit.chunk,
           "snapshot_steps": snap_steps, "status": status, "stop_reason": "x" if status == "stopped" else None,
           "stop_step": 7 if status == "stopped" else None, "chunks": [{"step_end": step}]}
    (d / "run.json").write_text(json.dumps(rec))
    for name in ["geometry.npz"] + [f"snapshots/snap_{k:02d}.npz" for k in range(6)]:
        if snapshots:
            (d / name).write_bytes(b"")
        else:
            (d / name).unlink(missing_ok=True)
    return d


def test_classify_run_states(units, tmp_path):
    u = units[0]
    cls = lambda: C.classify_run(tmp_path, u, CONFIG, SHA)
    assert cls()["status"] == "pending"
    _fake_run_dir(tmp_path, u)
    assert cls()["status"] == "complete"
    assert C.classify_run(tmp_path, u, CONFIG, "b" * 64)["status"] == "mismatch"                # a different bundle
    assert "bundle_sha256" in C.classify_run(tmp_path, u, CONFIG, "b" * 64)["reason"]
    _fake_run_dir(tmp_path, u, dt=u.dt * 1.01)
    assert cls()["status"] == "mismatch"
    _fake_run_dir(tmp_path, u, snapshots=False)
    assert cls()["status"] == "damaged"                                                          # complete without its outputs
    _fake_run_dir(tmp_path, u, step=3)
    assert cls()["status"] == "damaged"                                                          # complete, last step 3 of 50
    _fake_run_dir(tmp_path, u, status="running", step=10)
    c = cls()
    assert c["status"] == "resumable" and c["step"] == 10 and "no checkpoint" in c["reason"]
    (u.path(tmp_path) / "checkpoint.npz").write_bytes(b"")
    assert cls()["reason"] is None
    _fake_run_dir(tmp_path, u, status="stopped", step=7)
    assert cls()["status"] == "stopped"
    _fake_run_dir(tmp_path, u, status="weird")
    assert cls()["status"] == "damaged"
    (u.path(tmp_path) / "run.json").write_text("{not json")
    assert cls()["status"] == "damaged"
    # a variant unit carries its overrides in the expected parameters
    w1 = next(x for x in units if x.variant == "w1")
    _fake_run_dir(tmp_path, w1)
    assert C.classify_run(tmp_path, w1, CONFIG, SHA)["status"] == "complete"
    rec = json.loads((w1.path(tmp_path) / "run.json").read_text())
    rec["params"]["w1"] = 0.0
    (w1.path(tmp_path) / "run.json").write_text(json.dumps(rec))
    assert C.classify_run(tmp_path, w1, CONFIG, SHA)["status"] == "mismatch"


# ---------------------------------------------------------------------------------------------------------------------
# 5. variant comparisons (fabricated runs, known numbers)
# ---------------------------------------------------------------------------------------------------------------------
def _fake_snapshot_run(d: Path, q, qe, H, t=1.0, status="complete"):
    (d / "snapshots").mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"status": status}))
    masks = np.zeros((2, *H.shape), dtype=bool)
    masks[0], masks[1, 0] = True, True
    np.savez(d / "geometry.npz", H=H, region_names=np.array(["global", "wall"]), region_masks=masks)
    np.savez(d / "snapshots" / "snap_00.npz", t=0.0, q=qe, q_exact=qe)
    np.savez(d / "snapshots" / "snap_05.npz", t=t, q=q, q_exact=qe)


def _hnorm(a, H, plane0_only=False):
    a2 = H * a ** 2
    return float(np.sqrt(a2[0].sum() if plane0_only else a2.sum()))


def test_variant_comparisons_numbers(tmp_path):
    cfg = dict(CONFIG, resolutions=[32], arms=["filtered"], gated_arms=["filtered"], nsteps={"32": 50})
    spec = {"main": [{"experiment": "E3", "mode": "coupled", "patterns": ["NNN-D"]}],
            "variants": {"dt_half": {"ns": [32], "nsteps_factor": 2, "comparison": "dt_half"},
                         "rtol_tight": {"ns": [32], "opts_override": {"phi_rtol": 1e-13}, "comparison": "rtol_tight"},
                         "e3d": {"ns": [32], "opts_override": {"curvature_c_kappa": 1.0}, "comparison": "own"},
                         "w1": {"ns": [32], "params_override": {"w1": 1.0}, "comparison": "own"}}}
    us = C.build_matrix(cfg, spec)
    by = {u.id: u for u in us}
    rng = np.random.default_rng(1)
    E, P = 2, 5
    H = rng.uniform(0.5, 1.5, (E, P))
    qe = rng.standard_normal((E, P, 4))
    a, b, d1, d2, e3, w = (rng.standard_normal((E, P, 4)) * s for s in (1e-3, 1e-5, 1e-6, 1e-7, 2e-3, 3e-3))
    qc, qo = qe + a, qe + b
    mc = by["main.filtered.n32.coupled.NNN-D.continuum"]
    mo = by["main.filtered.n32.coupled.NNN-D.discrete"]
    _fake_snapshot_run(mc.path(tmp_path), qc, qe, H)
    _fake_snapshot_run(mo.path(tmp_path), qo, qe, H)
    _fake_snapshot_run(by["var.dt_half.filtered.n32.coupled.NNN-D.continuum"].path(tmp_path), qc + d1, qe, H)
    _fake_snapshot_run(by["var.rtol_tight.filtered.n32.coupled.NNN-D.continuum"].path(tmp_path), qc + d2, qe, H)
    _fake_snapshot_run(by["var.e3d.filtered.n32.coupled.NNN-D.continuum"].path(tmp_path), qe + e3, qe, H)
    qe_w1 = qe + 0.5                                                                  # the w1 exact solution differs
    _fake_snapshot_run(by["var.w1.filtered.n32.coupled.NNN-D.continuum"].path(tmp_path), qe_w1 + w, qe_w1, H)
    out = C.variant_comparisons(tmp_path, us, spec)
    assert out["rtol_budget_reference"] == 0.01 and set(out["variants"]) == {"dt_half", "rtol_tight", "e3d", "w1"}
    for f, name in enumerate(C.FIELD_NAMES):
        cell = out["variants"]["dt_half"]["entries"][0]["fields"][name]["global"]
        assert cell["dt_diff"] == pytest.approx(_hnorm(d1[..., f], H)) and cell["N-R"] == pytest.approx(_hnorm(a[..., f], H))
        assert cell["O-R"] == pytest.approx(_hnorm(b[..., f], H))
        assert cell["dt_diff/N-R"] == pytest.approx(cell["dt_diff"] / cell["N-R"]) and cell["dt_diff/O-R"] == pytest.approx(cell["dt_diff"] / cell["O-R"])
        wall = out["variants"]["dt_half"]["entries"][0]["fields"][name]["wall"]
        assert wall["dt_diff"] == pytest.approx(_hnorm(d1[..., f], H, True))
        rt = out["variants"]["rtol_tight"]["entries"][0]["fields"][name]["global"]
        assert rt["rtol_diff"] == pytest.approx(_hnorm(d2[..., f], H)) and rt["rtol_diff/N-R"] == pytest.approx(_hnorm(d2[..., f], H) / _hnorm(a[..., f], H))
        e = out["variants"]["e3d"]["entries"][0]["fields"][name]["global"]
        assert e["variant_N-R"] == pytest.approx(_hnorm(e3[..., f], H)) and e["main_N-R"] == pytest.approx(_hnorm(a[..., f], H))
        assert e["variant/main"] == pytest.approx(_hnorm(e3[..., f], H) / _hnorm(a[..., f], H))
        assert out["variants"]["w1"]["entries"][0]["fields"][name]["global"]["variant_N-R"] == pytest.approx(_hnorm(w[..., f], H))
    assert all(v["entries"][0]["status"] == "ok" for v in out["variants"].values())
    # an unusable main run is reported, an unusable variant run too; a zero denominator gives None, never an error
    _fake_snapshot_run(mc.path(tmp_path), qc, qe, H, status="running")
    out = C.variant_comparisons(tmp_path, us, spec)
    assert "main continuum run unusable" in out["variants"]["dt_half"]["entries"][0]["status"]
    assert out["variants"]["e3d"]["entries"][0]["main_N-R_status"].startswith("unavailable")
    _fake_snapshot_run(mc.path(tmp_path), qe, qe, H)                                   # N - R = 0 exactly
    out = C.variant_comparisons(tmp_path, us, spec)
    assert out["variants"]["rtol_tight"]["entries"][0]["fields"]["n"]["global"]["rtol_diff/N-R"] is None
    (by["var.w1.filtered.n32.coupled.NNN-D.continuum"].path(tmp_path) / "run.json").write_text(json.dumps({"status": "stopped"}))
    out = C.variant_comparisons(tmp_path, us, spec)
    assert "variant run unusable" in out["variants"]["w1"]["entries"][0]["status"]


# ---------------------------------------------------------------------------------------------------------------------
# 6. the launcher without compute
# ---------------------------------------------------------------------------------------------------------------------
def test_child_environment_and_tests_command():
    env = RA.child_env(3)
    assert (env["JAX_PLATFORMS"], env["JAX_ENABLE_X64"], env["CUDA_VISIBLE_DEVICES"]) == ("cpu", "true", "")
    assert env["NPROC"] == env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "3"
    assert str(SCRIPTS) in env["PYTHONPATH"].split(os.pathsep)
    cmd = RA.default_tests_command()
    assert cmd[1:4] == ["-m", "pytest", "-q"] and "not slow" in cmd and "tests/test_p10_evolved_campaign.py" in cmd
    assert all(c.startswith("tests/test_p10_evolved_") for c in cmd if c.startswith("tests/"))


def _cli(*args, **kw):
    return subprocess.run([sys.executable, "-m", "p10_evolved_mms.run_all", *map(str, args)], cwd=SCRIPTS, capture_output=True, text=True,
                          timeout=kw.get("timeout", 600))


def test_dry_run_prints_the_plan_and_computes_nothing(tmp_path, capsys):
    root = tmp_path / "root"
    rc = RA.main(["--root", str(root), "--stage", "all", "--workers", "3", "--threads", "2", "--memory-gb", "10", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and not root.exists()
    assert "units             69" in out and "dry run: nothing was computed" in out
    first = next(ln for ln in out.splitlines() if " var.dt_half.filtered.n64" in ln)
    assert first.split()[0] == "1" and first.split()[1] == "var.dt_half.filtered.n64.coupled.NNN-D.continuum"
    assert sum("unverified" in ln for ln in out.splitlines()) == 69
    sim = next(ln for ln in out.splitlines() if "max in-use estimate" in ln)
    assert float(sim.split("max in-use estimate")[1].split("GiB")[0]) <= 10.0 and "max concurrent 3" in sim
    # --only restricts the selection
    RA.main(["--root", str(root), "--stage", "run", "--only", r"n64\.coupled", "--dry-run"])
    out = capsys.readouterr().out
    assert "selected          12" in out and "n48" not in out.split("--- stage run")[1]


def test_all_stops_at_the_first_failing_stage_and_a_root_is_one_campaign(tmp_path):
    root = tmp_path / "root"
    fail = shlex.join([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])
    p = _cli("--root", root, "--stage", "all", "--synthetic", "--tests-command", fail)
    assert p.returncode == 1, p.stdout + p.stderr
    tests = json.loads((root / "receipts" / "tests.json").read_text())
    assert tests["status"] == "failed" and tests["returncode"] == 3 and tests["summary"] == "boom" and not tests["passed"]
    assert (root / "logs" / "tests.log").read_text().strip() == "boom"
    assert not (root / "receipts" / "verify.json").exists()                                       # stopped after tests
    camp = json.loads((root / "receipts" / "campaign.json").read_text())
    assert list(camp["stages"]) == ["tests"] and camp["stages"]["tests"]["status"] == "failed"
    assert camp["git_commit"] == C.read_git_commit() and camp["units"]["total"] == 69 and camp["units"]["counts"] == {"unverified": 69}
    man = json.loads((root / "manifest.json").read_text())
    assert man["configuration_sha256"] == C.canonical_sha(CONFIG) and len(man["units"]) == 69 and man["synthetic"] is True
    ok = shlex.join([sys.executable, "-c", "print('1 passed')"])
    p = _cli("--root", root, "--stage", "tests", "--synthetic", "--tests-command", ok)
    assert p.returncode == 0
    assert json.loads((root / "receipts" / "tests.json").read_text())["summary"] == "1 passed"
    # another matrix under the same root is refused
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"T": 1e-4}))
    p = _cli("--root", root, "--stage", "tests", "--synthetic", "--config-override", ov, "--tests-command", ok)
    assert p.returncode == 1 and "different campaign" in p.stderr
    # the run stage needs the verify stage first
    p = _cli("--root", root, "--stage", "run", "--synthetic")
    assert p.returncode == 1 and "verify stage first" in p.stdout


# ---------------------------------------------------------------------------------------------------------------------
# 7. end to end on the synthetic bundle
# ---------------------------------------------------------------------------------------------------------------------
E2E_OVERRIDE = {
    "resolutions": [16], "arms": ["raw"], "gated_arms": ["raw"], "nsteps": {"16": 5}, "T": 1e-4,
    "main": [{"experiment": "E1", "mode": "diffusion", "patterns": ["NNN-D"]}],
    "variants": {"dt_half": {"ns": [16], "nsteps_factor": 2, "mode": "diffusion", "comparison": "dt_half"},
                 "rtol_tight": {"ns": [16], "mode": "diffusion", "opts_override": {"phi_rtol": 1e-13}, "comparison": "rtol_tight"},
                 "bad": {"ns": [16], "mode": "diffusion", "opts_override": {"no_such_option": 1}, "comparison": "own"}},
}


def test_end_to_end_synthetic_campaign(tmp_path):
    t_start = time.time()
    ov = tmp_path / "override.json"
    ov.write_text(json.dumps(E2E_OVERRIDE))
    root = tmp_path / "root"
    base = ["--root", root, "--workers", 2, "--threads", 1, "--synthetic", "--config-override", ov]
    main_c, main_d = "main.raw.n16.diffusion.NNN-D.continuum", "main.raw.n16.diffusion.NNN-D.discrete"
    dt_half, rtol, bad = (f"var.{v}.raw.n16.diffusion.NNN-D.continuum" for v in ("dt_half", "rtol_tight", "bad"))
    rec = lambda rel: json.loads((root / rel).read_text())

    # verify ---------------------------------------------------------------------------------------------------------
    p = _cli(*base, "--stage", "verify")
    assert p.returncode == 0, p.stdout + p.stderr
    v = rec("receipts/verify.json")
    b = v["bundles"]["raw/n16"]
    assert v["status"] == "ok" and b["status"] == "ok" and b["synthetic"] and b["agreement"]["jac"] == "bitwise"
    assert b["worker"]["backend"] == "cpu" and (root / "bundles/raw/n16/identity.json").is_file() and (root / "bundles/raw/n16/bundle.npz").is_file()
    sha = b["bundle_sha256"]
    assert json.loads((root / "bundles/raw/n16/identity.json").read_text())["bundle_sha256"] == sha
    p = _cli(*base, "--stage", "verify")                                                          # idempotent: the bundle is reused
    assert p.returncode == 0 and rec("receipts/verify.json")["bundles"]["raw/n16"]["reused"] is True

    # preflight ------------------------------------------------------------------------------------------------------
    p = _cli(*base, "--stage", "preflight")
    assert p.returncode == 0, p.stdout + p.stderr
    pf = rec("receipts/preflight.json")
    e = pf["entries"]["raw/n16/diffusion/NNN-D"]
    assert pf["status"] == "ok" and e["status"] == "ok" and e["lambda_dt"] <= 1.3 and e["lambda_dt_half"] == e["lambda_dt"] / 2
    assert e["dt"] == 1e-4 / 5 and e["method"] == "arpack" and e["call_wall_s"] > 0 and e["peak_rss_gib_after_call"] > 0
    pj = rec("preflight/raw/n16/diffusion_NNN-D/preflight.json")
    assert pj["schema"] == "drbx.p10-preflight-v1" and pj["identity"]["bundle_sha256"] == sha and pj["rightmost_skipped"] is True

    # an interrupted unit: a partial run of the discrete main unit (checkpoint at step 2), resumed by the launcher ---------
    from p10_evolved_mms import bundle as bundle_mod, evolve as EV
    from p10_evolved_mms.fields import default_params

    man = rec("manifest.json")
    unit = next(u for u in C.units_from_manifest(man) if u.id == main_d)
    kw = C.params_kwargs(man["configuration"], unit.params_override)
    kw["D"] = np.asarray(kw["D"])

    class Interrupted(Exception):
        pass

    def stop_after_two(chunk):
        if chunk["step_end"] >= 2:
            raise Interrupted

    with pytest.raises(Interrupted):
        EV.run(bundle_mod.load_bundle(root / "bundles/raw/n16", expected_identity=sha), unit.mode, unit.source, default_params(**kw),
               unit.pattern, dt=unit.dt, T=unit.T, out_dir=unit.path(root), chunk=unit.chunk, on_chunk=stop_after_two)
    state = C.classify_run(root, unit, man["configuration"], sha)
    assert state["status"] == "resumable" and state["step"] == 2

    # run ------------------------------------------------------------------------------------------------------------
    p = _cli(*base, "--stage", "run")
    assert p.returncode == 1, p.stdout + p.stderr                                                 # the failing unit makes the stage incomplete
    assert "var.bad" in p.stdout and "1 resumed" in p.stdout
    run = rec("receipts/run.json")
    assert run["status"] == "incomplete" and run["failed"] == [bad] and run["counts"] == {"complete": 4, "failed": 1}
    for uid in (main_c, main_d, dt_half, rtol):
        r = rec(f"receipts/units/{uid}.json")
        a = r["attempts"][0]
        assert r["status"] == "complete" and a["exit_code"] == 0 and a["backend"] == "cpu" and a["peak_rss_gib"] > 0.05 and a["wall_s"] > 0
        assert a["started"] <= a["ended"] and (root / "logs" / f"{uid}.log").is_file() and r["backend"] == "cpu"
        d = root / r["unit"]["out_dir"]
        run_json = json.loads((d / "run.json").read_text())                                       # the run directory of the contract
        assert run_json["schema"] == "drbx.p10-run-v1" and run_json["status"] == "complete" and run_json["identity"]["bundle_sha256"] == sha
        assert run_json["nsteps"] == r["unit"]["nsteps"] and run_json["dt"] == r["unit"]["dt"] and run_json["T"] == 1e-4
        assert (d / "geometry.npz").is_file() and (d / "checkpoint.npz").is_file()
        assert sorted(f.name for f in (d / "snapshots").glob("snap_*.npz")) == [f"snap_{k:02d}.npz" for k in range(6)]
    assert rec(f"receipts/units/{main_d}.json")["attempts"][0]["resumed_from_step"] == 2
    assert rec(f"receipts/units/{main_c}.json")["attempts"][0]["resumed_from_step"] is None
    rb = rec(f"receipts/units/{bad}.json")
    assert rb["status"] == "failed" and rb["attempts"][0]["exit_code"] == 2 and "no_such_option" in rb["attempts"][0]["worker"]["error"]
    assert "Traceback" in (root / "logs" / f"{bad}.log").read_text()
    # the units sit in the layout the reducer discovers
    assert RD.discover(root / "main") == {"raw": {("diffusion", "NNN-D"): {16}}}
    camp = rec("receipts/campaign.json")
    assert camp["units"]["counts"] == {"complete": 4, "failed": 1} and camp["backend"] == ["cpu"] and camp["workers"] == 2
    assert camp["units"]["ids"]["failed"] == [bad] and set(camp["peak_rss_gib"]) == {main_c, main_d, dt_half, rtol, bad}
    assert camp["configuration_sha256"] == man["configuration_sha256"] and camp["stages"]["run"]["status"] == "incomplete"

    # resume: the second run skips the complete units and retries (and again fails) only the bad one -----------------------
    attempts_before = {uid: len(rec(f"receipts/units/{uid}.json")["attempts"]) for uid in (main_c, main_d, dt_half, rtol, bad)}
    p = _cli(*base, "--stage", "run")
    assert p.returncode == 1 and "4 already finished" in p.stdout
    run = rec("receipts/run.json")
    assert run["launched"] == 1 and run["skipped_finished"] == 4 and list(run["this_invocation"]) == [bad]
    attempts_after = {uid: len(rec(f"receipts/units/{uid}.json")["attempts"]) for uid in attempts_before}
    assert attempts_after == {**attempts_before, bad: attempts_before[bad] + 1}

    # validate: incomplete because of the failing unit, but the analysis is produced -----------------------------------------
    p = _cli(*base, "--stage", "validate")
    assert p.returncode == 1, p.stdout + p.stderr
    val = rec("receipts/validate.json")
    assert val["status"] == "failed" and set(val["not_complete"]) == {bad} and val["not_complete"][bad]["status"] == "failed"
    assert val["analysis"]["returncode"] == 0
    for rel in ("analysis/main/analysis.json", "analysis/main/orders.csv", "analysis/main/report.md", "analysis/variants/dt_half/analysis.json",
                "analysis/variants/rtol_tight/analysis.json", "analysis/variants.json"):
        assert (root / rel).is_file(), rel
    main_an = rec("analysis/main/analysis.json")
    assert [g["mode"] for g in main_an["groups"]] == ["diffusion"] and main_an["missing_runs"] == [] and main_an["excluded_runs"] == []
    comp = rec("analysis/variants.json")
    assert comp["variants"]["dt_half"]["entries"][0]["status"] == "ok" and comp["variants"]["rtol_tight"]["entries"][0]["status"] == "ok"
    cell = comp["variants"]["dt_half"]["entries"][0]["fields"]["n"]["global"]
    assert {"dt_diff", "N-R", "O-R", "dt_diff/N-R", "dt_diff/O-R"} <= set(cell)
    assert "rtol_diff/N-R" in comp["variants"]["rtol_tight"]["entries"][0]["fields"]["Omega"]["global"]
    assert "unusable" in comp["variants"]["bad"]["entries"][0]["status"]

    # all, without the failing unit: every stage is idempotent and ok -------------------------------------------------------
    ok_cmd = shlex.join([sys.executable, "-c", "pass"])
    p = _cli(*base, "--stage", "all", "--only", r"^(?!var\.bad)", "--tests-command", ok_cmd)
    assert p.returncode == 0, p.stdout + p.stderr
    camp = rec("receipts/campaign.json")
    assert {k: s["status"] for k, s in camp["stages"].items()} == {s: "ok" for s in RA.STAGES}
    run = rec("receipts/run.json")
    assert run["status"] == "ok" and run["launched"] == 0 and run["skipped_finished"] == 4
    assert rec("receipts/validate.json")["status"] == "ok" and rec("receipts/validate.json")["not_complete"] == {}
    assert camp["units"]["counts"] == {"complete": 4, "failed": 1}

    # an identity mismatch is reported and never rerun or relabelled ---------------------------------------------------------
    rj = root / unit.out_dir.replace("discrete", "continuum") / "run.json"
    stored = json.loads(rj.read_text())
    stored["identity"]["bundle_sha256"] = "0" * 64
    rj.write_text(json.dumps(stored))
    before = rj.read_text()
    n_attempts = len(rec(f"receipts/units/{main_c}.json")["attempts"])
    p = _cli(*base, "--stage", "run", "--only", re.escape(main_c))
    assert p.returncode == 1 and "BLOCKED" in p.stdout and "bundle_sha256" in p.stdout
    assert rj.read_text() == before and len(rec(f"receipts/units/{main_c}.json")["attempts"]) == n_attempts
    assert rec("receipts/run.json")["blocked"][main_c]["status"] == "mismatch"
    print(f"end-to-end synthetic campaign: {time.time() - t_start:.1f} s")
