#!/usr/bin/env python
"""P (perpendicular) path GPU benchmark campaign of DRBX: plane preconditioners, full psi PCG solve, core_schur build / storage
scaling, optional eta sharding, per backend (``gpu``: JAX CUDA on one A100; ``cpu``: the same node's CPUs).

Stages (one campaign folder ``--out``; see ``pgb_cases`` for its layout)::

    python run_benchmark.py preflight --out OUT --inputs INPUTS --backend gpu|cpu [--n 32 48 64]
    python run_benchmark.py run       --out OUT --inputs INPUTS --backend gpu|cpu [--n 32 48 64] [--reps 20] [--solve-reps 5]
                                      [--max-planes K] [--cases GLOB ...] [--force] [--cpu-threads T]
    python run_benchmark.py validate  --out OUT [--backend gpu cpu]

``INPUTS`` is the packed P10 input directory (``laplacian/filtered/N<n>/laplacian_metric.npz``, ``inputs_manifest.json``). ``run`` is
resumable (one JSON per case; completed cases are skipped, failed ones retried) and starts every resolution in a fresh process.
CPU thread count: ``--cpu-threads`` or the environment variable ``P_GPU_BENCH_CPU_THREADS``. ``validate`` exits 1 on any
missing / failed case or tolerance violation.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pgb_cases as K                                                                      # noqa: E402
import pgb_env                                                                             # noqa: E402
import pgb_inputs as I                                                                     # noqa: E402


# ---------------------------------------------------------------------------
def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _git_rev() -> str:
    env = os.environ.get("P_GPU_BENCH_SOURCE_REV")
    if env:
        return env
    try:
        r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except Exception:                                                                      # noqa: BLE001
        return "unknown"


def cmd_preflight(a) -> int:
    ns = tuple(a.n)
    env_set = pgb_env.configure(a.backend, a.cpu_threads)
    out = Path(a.out)
    campaign = K.build_campaign(ns)
    problems, rec = [], {"stage": "preflight", "backend": a.backend, "ns": list(ns), "ok": False,
                         "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env_set": env_set,
                         "host": {"node": platform.node(), "python": sys.version.split()[0], "platform": platform.platform(),
                                  "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")
                                            and k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_JOB_PARTITION",
                                                      "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS")},
                                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                  "affinity_cpus": pgb_env.affinity_cpus(), "cpu_threads": pgb_env.resolve_threads(a.cpu_threads),
                                  "source_rev": _git_rev()}}
    ver = I.verify_inputs(a.inputs, ns, campaign)
    rec["inputs"] = ver
    problems += ver["problems"]
    if not ver["manifest"].get("matches_campaign", True):
        rec.setdefault("warnings", []).append("inputs_manifest.json differs from the campaign's manifest sha256 (files were checked "
                                              "individually)")
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        import pgb_timing as T
        e = T.environment_record()
        rec["jax_env"] = e
        try:
            import solvax
            rec["solvax"] = solvax.__version__
            if solvax.__version__ != K.REMOTE_ENVIRONMENT["solvax"]:
                rec.setdefault("warnings", []).append(f"solvax {solvax.__version__} != campaign {K.REMOTE_ENVIRONMENT['solvax']}")
        except Exception as exc:                                                           # noqa: BLE001
            problems.append(f"solvax import failed: {exc}")
        if jax.__version__ != K.REMOTE_ENVIRONMENT["jax_local_cpu_version"]:
            rec.setdefault("warnings", []).append(f"jax {jax.__version__} != campaign {K.REMOTE_ENVIRONMENT['jax_local_cpu_version']}")
        want = ("gpu", "cuda") if a.backend == "gpu" else ("cpu",)
        if e["default_backend"] not in want or any(d["platform"] not in want for d in e["devices"]):
            problems.append(f"--backend {a.backend} but jax.devices() = {[d['str'] for d in e['devices']]} "
                            f"(default backend {e['default_backend']!r})")
        x = jnp.arange(8.0) @ jnp.arange(8.0)
        x.block_until_ready()
        rec["smoke_op_device"] = str(next(iter(x.devices())))
        if a.backend == "gpu" and not e.get("memory", {}).get("bytes_limit"):
            problems.append("GPU device reports no memory stats")
        import drbx
        rec["drbx"] = {"version": getattr(drbx, "__version__", None), "file": drbx.__file__}
        # identity of the embedded metric arrays (also checks the layout nodes)
        ids = {}
        for n in ns:
            try:
                ids[str(n)] = I.Metric(I.metric_path(a.inputs, n), n).identity
            except Exception as exc:                                                       # noqa: BLE001
                problems.append(f"N{n}: metric load failed: {exc}")
        rec["metric_identity"] = ids
    except SystemExit:
        raise
    except Exception as exc:                                                               # noqa: BLE001
        import traceback
        problems.append(f"jax initialisation failed: {type(exc).__name__}: {exc}")
        rec["traceback"] = traceback.format_exc()[-3000:]
    rec["problems"] = problems
    rec["ok"] = not problems
    _write(out / f"preflight_{a.backend}.json", rec)
    print(json.dumps({k: rec[k] for k in ("backend", "ok", "problems")}, indent=1))
    if "jax_env" in rec:
        je = rec["jax_env"]
        print(f"jax {je['jax']}: {je['device_count']} device(s): {[d['kind'] or d['platform'] for d in je['devices']]}; "
              f"memory {je['memory']}")
    return 0 if rec["ok"] else 2


def _worker_cmd(a, n: int) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--out", a.out, "--inputs", a.inputs, "--backend", a.backend,
           "--n", str(n), "--reps", str(a.reps), "--solve-reps", str(a.solve_reps)]
    if a.cpu_threads is not None:
        cmd += ["--cpu-threads", str(a.cpu_threads)]
    if a.max_planes:
        cmd += ["--max-planes", str(a.max_planes)]
    if a.force:
        cmd += ["--force"]
    if a.mem_budget_gb is not None:
        cmd += ["--mem-budget-gb", str(a.mem_budget_gb)]
    if a.cases:
        cmd += ["--cases", *a.cases]
    return cmd


def cmd_run(a) -> int:
    ns = tuple(a.n)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pgb_env.configure(a.backend, a.cpu_threads)
    camp = K.build_campaign(ns)
    camp["run"] = dict(backend=a.backend, resolutions=list(ns), reps=a.reps, solve_reps=a.solve_reps, max_planes=a.max_planes,
                       cpu_threads=pgb_env.resolve_threads(a.cpu_threads), cases_filter=a.cases, force=a.force,
                       started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), argv=sys.argv)
    _write(out / "campaign.json", camp)
    if not a.skip_preflight:
        cmd = [sys.executable, str(Path(__file__).resolve()), "preflight", "--out", a.out, "--inputs", a.inputs, "--backend", a.backend,
               "--n", *map(str, ns)] + (["--cpu-threads", str(a.cpu_threads)] if a.cpu_threads is not None else [])
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print("preflight failed; aborting (see preflight_%s.json)" % a.backend)
            return rc
    (out / "logs").mkdir(exist_ok=True)
    failures = []
    for n in ns:
        t0 = time.time()
        logp = out / "logs" / f"worker_{a.backend}_N{n}.log"
        print(f"[run] N{n} ({a.backend}) -> {logp}", flush=True)
        with open(logp, "a") as fh:
            fh.write(f"\n===== {datetime.now(timezone.utc).isoformat(timespec='seconds')} worker N{n} =====\n")
            fh.flush()
            rc = subprocess.run(_worker_cmd(a, n), stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"[run] N{n} worker exit {rc} after {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            failures.append((n, rc))
    # scaling case (derived from the build / zfree cases of this backend)
    _scaling_case(a, out, ns)
    rep = K.validate_campaign(out, [a.backend])
    b = rep["backends"].get(a.backend, {})
    print(f"[run] {b.get('present', 0)}/{b.get('expected', 0)} case files; {len(b.get('problems', []))} problem(s) "
          f"(run `validate` for details); worker failures: {failures}")
    return 0 if not failures and not b.get("problems") else 1


def _scaling_case(a, out: Path, ns) -> None:
    import pgb_scaling as S
    path = K.case_path(out, a.backend, "scaling")
    if path.is_file() and not a.force and json.loads(path.read_text()).get("status") in ("ok", "skipped"):
        return
    t0 = time.perf_counter()
    spec = dict(case_id="scaling", kind="scaling", n=None, method=None, dtype=None)

    def rec(status, reason, params, results):
        r = dict(schema=K.CASE_SCHEMA, **spec, backend=a.backend, arm=K.ARM, status=status, reason=reason, params=params, results=results,
                 identity={"inputs_sha256": {str(n): K.INPUT_FILES[n]["sha256"] for n in ns}}, env={"backend": a.backend},
                 created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), elapsed_s=time.perf_counter() - t0)
        _write(path, r)

    if a.max_planes:
        rec("skipped", K.RESTRICTED_SKIP, {}, {})
        return
    builds, zfree = {dt: {} for dt in K.DTYPES}, {dt: {} for dt in K.DTYPES}
    for n in ns:
        for dt in K.DTYPES:
            for store, cid in ((builds, K.case_id("build", n, "core_schur", dt)), (zfree, K.case_id("precond", n, "core_schur_zfree", dt))):
                p = K.case_path(out, a.backend, cid)
                if p.is_file():
                    r = json.loads(p.read_text())
                    if r["status"] == "ok":
                        store[dt][n] = r["results"]
    if not any(builds.values()):
        rec("failed", "no build case results", {}, {})
        return
    try:
        res = S.compute_scaling(builds, zfree, I.layout_structure)
        rec("ok", None, {"fit_points": {dt: sorted(builds[dt]) for dt in K.DTYPES}}, res)
    except Exception as exc:                                                               # noqa: BLE001
        import traceback
        rec("failed", f"{type(exc).__name__}: {exc}", {"traceback": traceback.format_exc()[-3000:]}, {})


def cmd_worker(a) -> int:
    pgb_env.configure(a.backend, a.cpu_threads)
    import pgb_worker as W
    w = W.Worker(inputs=a.inputs, out=a.out, backend=a.backend, n=a.n[0], reps=a.reps, solve_reps=a.solve_reps,
                 max_planes=a.max_planes, force=a.force, patterns=a.cases, mem_budget_gb=a.mem_budget_gb)
    w.run()
    return 0


def cmd_validate(a) -> int:
    rep = K.validate_campaign(a.out, a.backend)
    out = Path(a.out)
    _write(out / "validation.json", rep)
    for b, r in rep.get("backends", {}).items():
        ok = sum(1 for s in r["cases"].values() if s == "ok")
        sk = sum(1 for s in r["cases"].values() if s == "skipped")
        print(f"[{b}] {r['present']}/{r['expected']} case files: {ok} ok, {sk} skipped; {len(r['problems'])} problem(s)")
        for p in r["problems"]:
            print("   -", p)
    for p in rep.get("problems", []):
        print("-", p)
    if rep["restricted"]:
        print("NOTE: this run was restricted to the first planes (--max-planes): not a full campaign")
    print("VALIDATION", "PASSED" if rep["ok"] else "FAILED")
    return 0 if rep["ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p, backend_required=True):
        p.add_argument("--out", required=True, help="campaign folder")
        p.add_argument("--backend", required=backend_required, choices=K.BACKENDS)
        p.add_argument("--cpu-threads", type=int, default=None, help=f"CPU backend threads (default: ${pgb_env.THREADS_ENV})")

    def inputs(p):
        p.add_argument("--inputs", required=True, help="packed P10 inputs directory (laplacian/filtered/N<n>/..., inputs_manifest.json)")
        p.add_argument("--n", type=int, nargs="+", default=list(K.DEFAULT_NS), choices=sorted(K.INPUT_FILES))

    def runopts(p):
        p.add_argument("--reps", type=int, default=20, help="timed repetitions of the precond / matvec applies")
        p.add_argument("--solve-reps", type=int, default=5, help="timed repetitions of the full PCG solve")
        p.add_argument("--max-planes", type=int, default=None, help="smoke / test only: first K planes (precond cases only)")
        p.add_argument("--cases", nargs="+", default=None, help="only case ids matching these globs")
        p.add_argument("--force", action="store_true", help="recompute cases that already have a record")
        p.add_argument("--mem-budget-gb", type=float, default=None, help="override the free-memory budget used to skip cases")

    p = sub.add_parser("preflight", help="identity / sha checks and backend detection")
    common(p); inputs(p)
    p = sub.add_parser("run", help="run all cases (resumable)")
    common(p); inputs(p); runopts(p)
    p.add_argument("--skip-preflight", action="store_true")
    p = sub.add_parser("worker", help=argparse.SUPPRESS)
    common(p); inputs(p); runopts(p)
    p = sub.add_parser("validate", help="check completeness and tolerances")
    p.add_argument("--out", required=True)
    p.add_argument("--backend", nargs="+", choices=K.BACKENDS, default=None)
    a = ap.parse_args(argv)
    return {"preflight": cmd_preflight, "run": cmd_run, "worker": cmd_worker, "validate": cmd_validate}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
