"""Seconds-scale tests of the P GPU benchmark harness: N32, 2 planes, CPU."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pgb_cases as K
import pgb_env
import pgb_inputs as I
import pgb_scaling as S

BENCH = Path(__file__).resolve().parent.parent
PY = sys.executable
E2 = 2


@pytest.fixture(scope="module")
def n32(inputs_dir):
    import jax.numpy as jnp
    from drbx.geometry.sbp_laplacian import build_laplacian_plan
    from drbx.geometry.sbp_laplacian_assembly import iter_inplane_blocks
    from drbx.native.fci_perpendicular_plane_preconditioner import build_core_schur_preconditioner
    import pgb_precond as PC

    metric = I.Metric(I.metric_path(inputs_dir, 32), 32)
    plan = build_laplacian_plan(metric.layout, metric.metric)
    st = plan.structure
    groups = PC.restrict_groups(iter_inplane_blocks(plan, group_planes=E2), E2)
    blocks = PC.PlaneBlocks(groups, n_planes=E2, nc=st.Nc, m=st.m, N=st.N)
    cores = {dt: build_core_schur_preconditioner(blocks.groups, n_planes=E2, n_core=st.Nc, n_rings=st.m, ring_size=st.N,
                                                 factor_dtype=fd) for dt, fd in (("f32", "float32"), ("f64", "float64"))}
    rhs = np.random.default_rng(0).standard_normal(E2 * st.P)
    return dict(blocks=blocks, cores=cores, rhs=rhs, st=st, ref=PC.dense_reference(blocks, rhs), jnp=jnp, PC=PC)


def test_zfree_apply_equals_stored_z(n32):
    import jax
    from drbx.native.fci_perpendicular_plane_preconditioner import apply_core_schur_preconditioner
    PC, jnp = n32["PC"], n32["jnp"]
    for dt, tol in (("f64", 1e-12), ("f32", 1e-5)):
        core = n32["cores"][dt]
        zp = PC.build_zfree_preconditioner(core, n32["blocks"])
        r = jnp.asarray(n32["rhs"])
        y_z = np.asarray(jax.jit(PC.apply_zfree_preconditioner)(zp, r))
        y_s = np.asarray(jax.jit(apply_core_schur_preconditioner)(core, r))
        assert np.linalg.norm(y_z - y_s) / np.linalg.norm(y_s) < tol, dt
        assert PC.rel_err(y_z, n32["ref"])[0] < (1e-10 if dt == "f64" else 1e-5)
        assert zp.nbytes < core.nbytes                   # no Z: smaller storage
        assert zp.meta.nr % n32["st"].N == 0 and 0 < zp.meta.nr < n32["st"].m * n32["st"].N


def test_solvax_variants_match_dense(n32):
    import jax
    PC, jnp = n32["PC"], n32["jnp"]
    r = jnp.asarray(n32["rhs"])
    P = n32["st"].P
    mods, sizes = PC.build_block_jacobi(n32["blocks"], "f64")
    y = jax.jit(lambda m, v: PC.apply_block_jacobi(m, v, sizes=sizes, P=P, dt="f64"))(mods, r)
    assert PC.rel_err(np.asarray(y), n32["ref"])[0] < 1e-10
    state, lay = PC.build_block_thomas(n32["blocks"], "f64")
    y = jax.jit(lambda s, v: PC.apply_block_thomas(s, v, nc=n32["st"].Nc, mN=n32["st"].m * n32["st"].N, n_s=lay["n_s"], Bk=lay["Bk"],
                                                   dt="f64"))(state, r)
    assert PC.rel_err(np.asarray(y), n32["ref"])[0] < 1e-10


def test_case_json_schema_and_skip(inputs_dir, tmp_path):
    import pgb_worker as W
    w = W.Worker(inputs=inputs_dir, out=tmp_path, backend="cpu", n=32, reps=2, solve_reps=1, max_planes=E2,
                 patterns=["build__N32__core_schur__f64", "precond__N32__core_schur_zfree__f64", "pcg__N32__core_schur__f64"])
    w.run()
    for cid, status in (("build__N32__core_schur__f64", "ok"), ("precond__N32__core_schur_zfree__f64", "ok"),
                        ("pcg__N32__core_schur__f64", "skipped")):
        rec = K.read_record(K.case_path(tmp_path, "cpu", cid))
        assert K.check_record_schema(rec) == [], cid
        assert rec["status"] == status
    spec = {s["case_id"]: s for s in K.case_table([32])}
    rec = K.read_record(K.case_path(tmp_path, "cpu", "precond__N32__core_schur_zfree__f64"))
    assert K.check_case_results(K.load_campaign(tmp_path), spec[rec["case_id"]], rec) == []
    assert rec["results"]["rel_diff_vs_stored_z"] < 1e-12 and rec["results"]["reference"] == "dense_lu_f64_host"
    assert rec["identity"]["metric_sha256"] == K.INPUT_FILES[32]["sha256"]
    # resumable: a second run with the same filter finds nothing pending
    w2 = W.Worker(inputs=inputs_dir, out=tmp_path, backend="cpu", n=32, max_planes=E2,
                  patterns=["build__N32__core_schur__f64"])
    assert w2.pending() == []
    assert K.check_record_schema({"schema": "x"})                                    # malformed records are reported


def _synthetic_ok(spec, backend="cpu"):
    r = {}
    if spec["kind"] == "build":
        r = dict(storage_bytes=10, host_build_s=0.1)
    elif spec["kind"] == "precond":
        r = dict(rel_err_vs_ref=1e-12, rel_residual=1e-13, apply=dict(median_s=1e-3), reference="x", rel_diff_vs_stored_z=0.0)
    elif spec["kind"] == "pcg":
        r = dict(converged=True, relative_residual=1e-12, true_residual_hinv=1e-12, iterations=4)
    elif spec["kind"] == "sharding":
        r = dict(rel_diff_vs_single=0.0)
    else:
        r = dict(shape_model={})
    return dict(schema=K.CASE_SCHEMA, case_id=spec["case_id"], kind=spec["kind"], backend=backend, n=spec["n"], arm="filtered",
                status="ok", reason=None, params={}, results=r, identity={}, env={}, created_utc="t", elapsed_s=0.0)


def _write_campaign(tmp_path, ns=(32,)):
    camp = K.build_campaign(ns)
    (tmp_path / "campaign.json").write_text(json.dumps(camp))
    (tmp_path / "preflight_cpu.json").write_text(json.dumps({"ok": True}))
    for spec in camp["cases"]:
        p = K.case_path(tmp_path, "cpu", spec["case_id"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_synthetic_ok(spec)))
    return camp


def test_validate_catches_missing_failed_and_out_of_tolerance(tmp_path):
    camp = _write_campaign(tmp_path)
    assert K.validate_campaign(tmp_path)["ok"]
    # missing case
    victim = K.case_path(tmp_path, "cpu", "precond__N32__core_schur__f32")
    victim.rename(victim.with_suffix(".bak"))
    rep = K.validate_campaign(tmp_path)
    assert not rep["ok"] and any("precond__N32__core_schur__f32: case file missing" in p for p in rep["backends"]["cpu"]["problems"])
    victim.with_suffix(".bak").rename(victim)
    assert K.validate_campaign(tmp_path)["ok"]
    # tolerance violated (f32 core_schur above 1e-5)
    rec = json.loads(victim.read_text())
    rec["results"]["rel_err_vs_ref"] = 3e-5
    victim.write_text(json.dumps(rec))
    assert not K.validate_campaign(tmp_path)["ok"]
    rec["results"]["rel_err_vs_ref"] = 1e-7
    victim.write_text(json.dumps(rec))
    # PCG not converged
    pcg = K.case_path(tmp_path, "cpu", "pcg__N32__core_schur__f64")
    rec = json.loads(pcg.read_text())
    rec["results"]["converged"] = False
    pcg.write_text(json.dumps(rec))
    assert not K.validate_campaign(tmp_path)["ok"]
    rec["results"]["converged"] = True
    pcg.write_text(json.dumps(rec))
    # failed / disallowed skip / allowed skip
    bj = K.case_path(tmp_path, "cpu", "precond__N32__block_jacobi__f64")
    rec = json.loads(bj.read_text())
    rec.update(status="skipped", reason="skipped: memory (test)", results={})
    bj.write_text(json.dumps(rec))
    assert K.validate_campaign(tmp_path)["ok"]
    rec.update(status="skipped", reason="skipped: whatever")
    bj.write_text(json.dumps(rec))
    assert not K.validate_campaign(tmp_path)["ok"]
    rec.update(status="failed", reason="boom")
    bj.write_text(json.dumps(rec))
    assert not K.validate_campaign(tmp_path)["ok"]
    # core_schur may never be skipped
    rec = json.loads(victim.read_text())
    rec.update(status="skipped", reason="skipped: memory (test)", results={})
    victim.write_text(json.dumps(rec))
    assert not K.validate_campaign(tmp_path)["ok"]
    # CLI exit code
    r = subprocess.run([PY, str(BENCH / "run_benchmark.py"), "validate", "--out", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 1 and "VALIDATION FAILED" in r.stdout


def test_campaign_json_is_current():
    assert json.loads((BENCH / "campaign.json").read_text()) == json.loads(json.dumps(K.build_campaign()))
    c = K.build_campaign()
    assert {f["sha256"] for f in c["inputs"]["files"].values()} and c["remote_environment"]["solvax"] == "0.8.7"
    ids = [s["case_id"] for s in c["cases"]]
    assert len(ids) == len(set(ids)) == 41


def test_env_configuration_gpu_does_not_force_cpu():
    assert "JAX_PLATFORMS" not in pgb_env.build_env("gpu", environ={})
    assert pgb_env.build_env("cpu", 3, environ={})["JAX_PLATFORMS"] == "cpu"
    e = pgb_env.build_env("cpu", None, environ={pgb_env.THREADS_ENV: "5"})
    assert e["OMP_NUM_THREADS"] == "5" and "intra_op_parallelism_threads=5" in e["XLA_FLAGS"]
    with pytest.raises(SystemExit):
        pgb_env.build_env("gpu", environ={"JAX_PLATFORMS": "cpu"})


def test_power_fit_and_shape_model():
    f = S.power_fit([32, 48, 64], [3 * n ** 4.5 for n in (32, 48, 64)])
    assert abs(f["exponent"] - 4.5) < 1e-9 and abs(S.evaluate_fit(f, 128) / (3 * 128 ** 4.5) - 1) < 1e-9
    st = {"E": 32, "Nc": 56, "m": 28, "N": 32}
    comp = S.core_schur_components(st, 2, 2, "f32")
    assert sum(comp.values()) == 28844032                                  # the stored bytes of the measured N32 f32 build


def test_sharded_apply_equals_single_device(inputs_dir):
    """Two host CPU devices (a fresh process with ``--xla_force_host_platform_device_count=2``)."""
    code = f"""
import sys, json
sys.path.insert(0, {str(BENCH)!r})
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import pgb_inputs as I, pgb_precond as PC, pgb_shard as SH
from drbx.geometry.sbp_laplacian import build_laplacian_plan
from drbx.geometry.sbp_laplacian_assembly import iter_inplane_blocks
from drbx.native.fci_perpendicular_plane_preconditioner import build_core_schur_preconditioner
m = I.Metric(I.metric_path({str(inputs_dir)!r}, 32), 32)
plan = build_laplacian_plan(m.layout, m.metric); st = plan.structure
groups = PC.restrict_groups(iter_inplane_blocks(plan, group_planes=2), 2)
prec = build_core_schur_preconditioner(groups, n_planes=2, n_core=st.Nc, n_rings=st.m, ring_size=st.N, factor_dtype="float32")
rhs = np.random.default_rng(0).standard_normal(2 * st.P)
print("RESULT", json.dumps(SH.run_sharding(prec, rhs, reps=2)))
"""
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2", JAX_PLATFORMS="cpu")
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(r.stdout.split("RESULT", 1)[1])
    assert res["n_devices_used"] == 2 and res["planes_per_device"] == 1
    assert res["rel_diff_vs_single"] < 1e-5
