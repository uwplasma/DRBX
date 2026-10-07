"""Campaign launcher of the P10 evolved MMS (chunk C7): tests, input / bundle verification, preflight, the 69 evolution runs and
the reduction, node-local, with process-level parallelism, resumable and self-describing.

Invocation (from ``DRBX/scripts``, or with ``DRBX/scripts`` on ``PYTHONPATH``; the launcher itself never imports JAX)::

    cd DRBX/scripts
    python -m p10_evolved_mms.run_all --root ROOT --inputs INPUTS --stage {tests,verify,preflight,run,validate,all} \\
        --workers W --threads K [--memory-gb M] [--memory-table 32=1.5,48=3.0,64=4.5] [--only REGEX] [--dry-run]

``INPUTS`` is a directory written by ``python -m p10_evolved_mms.inputs pack``. ``ROOT`` is the campaign root (created; one
campaign per root: ``ROOT/manifest.json`` is written once and every later invocation must describe the same matrix,
configuration and inputs). ``--only REGEX`` selects units by ``re.search`` on their id (``main.filtered.n64.coupled.NNN-D.continuum``,
``var.dt_half.filtered.n32.coupled.NNN-D.continuum``); the bundle builds and preflight calls of a stage are restricted to what the
selected units need, ``validate`` judges the selected units. ``--dry-run`` prints the plan and the simulated schedule and computes
nothing. ``JAX_COMPILATION_CACHE_DIR`` set in the launcher's environment is inherited by every compute subprocess.

Every compute step is a subprocess with ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=true CUDA_VISIBLE_DEVICES= NPROC=K OMP_NUM_THREADS=K
OPENBLAS_NUM_THREADS=K`` (also ``MKL_NUM_THREADS``, ``VECLIB_MAXIMUM_THREADS``); the unit and bundle subprocesses record
``jax.default_backend()`` (and fail unless it is ``cpu``). ``W x K`` should not exceed the number of cores.

Stages (``all`` runs 1-5 in order and stops at the first stage whose status is not ``ok``; every stage is idempotent)
-------------------------------------------------------------------------------------------------------------------
1. ``tests``      ``pytest -q -m "not slow" tests/test_p10_evolved_*.py`` from the repository root; ``receipts/tests.json``,
                  ``logs/tests.log``.
2. ``verify``     re-hash the packed inputs against ``inputs_manifest.json`` (and its sha256 against the campaign manifest); build
                  every needed ``(arm, N)`` bundle once into ``bundles/{arm}/n{N}/`` (``bundle.build_bundle`` from the packed
                  files, in parallel under the memory budget; an existing bundle built from the same file hashes is reused);
                  check the identities (arm identity, file hashes = packed inputs, ``Hp`` / ``|J|`` agreement recorded);
                  ``receipts/verify.json`` holds the bundle identities every later unit is checked against.
3. ``preflight``  per ``(arm, N)`` one subprocess loads the bundle and calls ``preflight.preflight(..., rightmost=False)`` for each
                  ``(mode, pattern)`` of the main matrix (ARPACK ``lambda_max`` at ``t = 0, T/2, T``) into
                  ``preflight/{arm}/n{N}/{mode}_{pattern}/preflight.json``; then ``lambda_max * dt <= 1.3`` for the configured
                  ``dt`` (the ``dt/2`` variants are trivially half): a violation fails the stage with the offending groups and
                  the step count that would satisfy the gate. Time and peak RSS per call in ``receipts/preflight.json``.
4. ``run``        units in priority order (descending N, then descending cost: N64 coupled first), at most W concurrent
                  subprocesses and (with ``--memory-gb``) the sum of the per-unit memory estimates within the budget (a unit alone
                  always starts). Skipped: ``complete`` units and units stopped by ``evolve``'s stop rule (a result). Resumed:
                  interrupted units (``run.json`` status ``running``) through ``evolve.run(resume=True)``, i.e. from the last
                  checkpoint, bitwise like the uninterrupted run. Never rerun or relabelled: a stored run whose identity,
                  parameters or ``dt`` differ from the unit (reported as ``mismatch``) or whose outputs are damaged. A failed
                  unit does not stop the others; the stage is ``ok`` only if every selected unit is complete.
                  ``logs/{unit_id}.log``, ``receipts/units/{unit_id}.json`` (attempts with start / end, exit status, wall time, peak
                  RSS of the child from ``wait4``, backend, the worker's own record).
5. ``validate``   every selected unit complete with the verified bundle identity? ``reduce.reduce_campaign`` of ``main`` into
                  ``analysis/main/`` and of every variant tree into ``analysis/variants/{variant}/`` (also when units are missing:
                  the reducer reports them), and the variant comparisons ``analysis/variants.json`` (machine output, no verdicts).

``receipts/campaign.json`` is rewritten after every stage: commit, configuration sha256, input-manifest sha256, per-stage
status / time, unit counts (complete / failed / stopped / ...), backend, workers / threads / memory budget, peak RSS per unit.

Exit status: 0 if the requested stage(s) are ``ok``, 1 otherwise, 130 after SIGINT / SIGTERM (children are terminated; their
checkpoints make the next invocation resume them).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import NamedTuple

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p10_evolved_mms import campaign as C                                                      # noqa: E402
from p10_evolved_mms import inputs as INP                                                      # noqa: E402

STAGES = ("tests", "verify", "preflight", "run", "validate")
WORKERS = ("bundle", "preflight", "unit", "analyze")
RECEIPT_SCHEMA = "drbx.p10-campaign-receipt-v1"
SYNTHETIC_N_ETA = 8
EXIT_MISMATCH = 3
_STOP = threading.Event()
_IN_POOL = [False]


class StageError(RuntimeError):
    """A stage cannot proceed (clear message for the operator)."""


class Task(NamedTuple):
    """A schedulable item that is not a unit (bundle build, preflight process): ``id``, ``n`` and ``cost`` as :class:`campaign.Unit`."""

    id: str
    n: int
    cost: float
    arm: str


# ---------------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False, default=str))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def rss_gib(ru) -> float:
    """Peak RSS in GiB of a ``resource.struct_rusage`` (``ru_maxrss`` is bytes on macOS, KiB on Linux)."""
    return ru.ru_maxrss / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)


def self_peak_gib() -> float:
    return rss_gib(resource.getrusage(resource.RUSAGE_SELF))


def child_env(threads: int) -> dict:
    k = str(int(threads))
    env = dict(os.environ)
    env.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true", CUDA_VISIBLE_DEVICES="", NPROC=k, OMP_NUM_THREADS=k,
               OPENBLAS_NUM_THREADS=k, MKL_NUM_THREADS=k, VECLIB_MAXIMUM_THREADS=k, PYTHONDONTWRITEBYTECODE="1")
    paths = [str(SCRIPTS), str(REPO / "src"), str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _handle_signal(signum, _frame):
    _STOP.set()
    if not _IN_POOL[0]:
        raise KeyboardInterrupt(f"signal {signum}")


def spawn(cmd, env, log_path: Path, *, cwd, header: str | None = None, append: bool = True):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "ab" if append else "wb")
    if header:
        fh.write((header + "\n").encode())
        fh.flush()
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL)
    return proc, fh


def _reap(proc, fh, status, ru) -> tuple[int, float]:
    proc.returncode = os.waitstatus_to_exitcode(status)       # Popen must not reap again (wait4 already did)
    fh.close()
    return proc.returncode, rss_gib(ru)


def run_blocking(cmd, env, log_path: Path, *, cwd, header=None, append=False):
    """Run one subprocess to completion: ``(returncode, wall_s, peak_rss_gib_of_child)``."""
    t0 = time.time()
    proc, fh = spawn(cmd, env, log_path, cwd=cwd, header=header, append=append)
    try:
        _pid, status, ru = os.wait4(proc.pid, 0)
    except BaseException:
        proc.kill()
        os.wait4(proc.pid, 0)
        fh.close()
        raise
    rc, gib = _reap(proc, fh, status, ru)
    return rc, time.time() - t0, gib


def run_pool(items, *, workers: int, budget_gb, mem_of, launch, finish) -> tuple[bool, list]:
    """Run ``items`` (priority order) with at most ``workers`` concurrent subprocesses within the memory budget (rule
    :func:`campaign.can_start`). ``launch(item) -> (proc, fh)``; ``finish(item, returncode, peak_gib, t0, t1)``. Returns
    ``(interrupted, items_not_started)``; on SIGINT / SIGTERM the running children are terminated (SIGTERM, then SIGKILL) and
    reported through ``finish`` with their signal exit code."""
    pending, running = list(items), {}
    _IN_POOL[0] = True
    try:
        while pending or running:
            if _STOP.is_set():
                break
            while pending and C.can_start(sum(r["mem"] for r in running.values()), len(running), mem_of(pending[0]), workers,
                                          budget_gb):
                item = pending.pop(0)
                need = mem_of(item)
                if budget_gb is not None and not running and need > budget_gb:
                    log(f"WARNING {item.id}: its memory estimate {need:.1f} GiB exceeds the budget {budget_gb:.1f} GiB; running alone")
                proc, fh = launch(item)
                running[proc.pid] = {"item": item, "proc": proc, "fh": fh, "t0": time.time(), "mem": need}
            progressed = False
            for pid in list(running):
                try:
                    wpid, status, ru = os.wait4(pid, os.WNOHANG)
                except ChildProcessError:                       # reaped elsewhere: cannot happen, but never hang
                    wpid, status, ru = pid, 255 << 8, resource.getrusage(resource.RUSAGE_CHILDREN)
                if wpid == pid:
                    r = running.pop(pid)
                    rc, gib = _reap(r["proc"], r["fh"], status, ru)
                    finish(r["item"], rc, gib, r["t0"], time.time())
                    progressed = True
            if not progressed:
                time.sleep(0.2)
        interrupted = _STOP.is_set()
        if interrupted and running:
            for r in running.values():
                r["proc"].terminate()
            deadline = time.time() + 30.0
            while running and time.time() < deadline:
                for pid in list(running):
                    wpid, status, ru = os.wait4(pid, os.WNOHANG)
                    if wpid == pid:
                        r = running.pop(pid)
                        rc, gib = _reap(r["proc"], r["fh"], status, ru)
                        finish(r["item"], rc, gib, r["t0"], time.time())
                time.sleep(0.2)
            for pid in list(running):
                running[pid]["proc"].kill()
                _wpid, status, ru = os.wait4(pid, 0)
                r = running.pop(pid)
                rc, gib = _reap(r["proc"], r["fh"], status, ru)
                finish(r["item"], rc, gib, r["t0"], time.time())
        return interrupted, pending
    finally:
        _IN_POOL[0] = False


# ---------------------------------------------------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------------------------------------------------
class Ctx:
    """The launcher state: arguments, the effective configuration, the matrix, the selected units and the paths."""

    def __init__(self, args, *, write_manifest: bool = True):
        self.args = args
        self.root = Path(args.root)
        self.inputs = None if args.inputs is None else Path(args.inputs)
        self.workers = max(1, int(args.workers))
        self.threads = max(1, int(args.threads))
        self.memory_gb = args.memory_gb
        self.mem_table = C.parse_memory_table(args.memory_table)
        self.only = args.only
        self.tests_command = shlex.split(args.tests_command) if args.tests_command else None
        self.commit = C.read_git_commit()
        if getattr(args, "worker", None):                    # a subprocess: the campaign manifest is the single source of truth
            self.manifest = C.load_manifest(self.root)
            self.config, self.spec = self.manifest["configuration"], self.manifest["spec"]
            self.synthetic = bool(self.manifest["synthetic"])
            self.inputs_sha = self.manifest["inputs_manifest_sha256"]
            self.all_units = C.units_from_manifest(self.manifest)
            self.units = C.select_units(self.all_units, self.only)
            return
        self.synthetic = bool(args.synthetic)
        self.config = C.load_config(args.config_override)
        self.spec = C.load_spec_override(args.config_override)
        self.all_units = C.build_matrix(self.config, self.spec)
        self.units = C.select_units(self.all_units, self.only)
        if self.only and not self.units:
            raise StageError(f"--only {self.only!r} selects no unit")
        self.inputs_sha = None
        if not self.synthetic:
            if self.inputs is not None and (self.inputs / INP.MANIFEST).is_file():
                self.inputs_sha = INP.manifest_sha256(self.inputs)
            elif write_manifest:
                raise StageError("--inputs must be a directory written by `python -m p10_evolved_mms.inputs pack` "
                                 f"(no {INP.MANIFEST} in {self.inputs})")
        self.manifest = self._manifest(write_manifest)

    def _manifest(self, write: bool) -> dict:
        new = C.build_manifest(self.config, self.all_units, spec=self.spec, commit=self.commit, inputs_manifest_sha256=self.inputs_sha,
                               synthetic=self.synthetic, config_file_sha256=C.file_sha(C.CONFIG_PATH))
        if not write:                                          # dry run: describe, never compare or write
            return new
        path = self.root / "manifest.json"
        if path.is_file():
            old = C.load_manifest(self.root)
            diff = C.manifest_difference(old, new)
            if diff:
                raise StageError(f"{path} describes a different campaign (differs in {diff}); use another --root")
            return old
        write_json(path, new)
        return new

    # paths ---------------------------------------------------------------------------------------------------------
    def receipts(self, name: str) -> Path:
        return self.root / "receipts" / name

    def bundle_dir(self, arm: str, n: int) -> Path:
        return self.root / "bundles" / arm / f"n{n}"

    def preflight_dir(self, arm, n, mode, pattern) -> Path:
        return self.root / "preflight" / arm / f"n{n}" / f"{mode}_{pattern}"

    def log_path(self, name: str) -> Path:
        return self.root / "logs" / f"{name}.log"

    def mem_of(self, item) -> float:
        return C.unit_memory_gb(item.n, self.mem_table)

    def bundle_shas(self, required: bool = True) -> dict:
        """``{"arm/nN": bundle_sha256}`` from ``receipts/verify.json``."""
        rec = read_json(self.receipts("verify.json"), {})
        shas = {k: v["bundle_sha256"] for k, v in (rec.get("bundles") or {}).items() if v.get("status") == "ok"}
        if required:
            need = {f"{u.arm}/n{u.n}" for u in self.units}
            missing = sorted(need - set(shas))
            if missing:
                raise StageError(f"no verified bundle for {missing}: run the verify stage first (receipts/verify.json)")
        return shas

    def worker_cmd(self, task: str, **kw) -> list:
        cmd = [sys.executable, "-m", "p10_evolved_mms.run_all", "--worker", task, "--root", str(self.root)]
        if self.inputs is not None:
            cmd += ["--inputs", str(self.inputs)]
        if self.only:
            cmd += ["--only", self.only]
        for k, v in kw.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        return cmd


def unit_state(ctx: Ctx, units, shas: dict) -> dict:
    """``{unit_id: classify_run(...)}`` (``unverified`` for a unit whose bundle identity is not known yet)."""
    out = {}
    for u in units:
        sha = shas.get(f"{u.arm}/n{u.n}")
        out[u.id] = (C.classify_run(ctx.root, u, ctx.config, sha) if sha is not None else
                     {"status": "unverified", "reason": "no verified bundle identity", "run_status": None, "step": None})
    return out


def reported_state(ctx: Ctx, units, shas: dict) -> dict:
    """:func:`unit_state` with the outcome of the last launch folded in for reporting: a never-started unit whose last attempt
    failed is ``failed`` (or ``mismatch`` for a bundle identity mismatch), not ``pending``."""
    st = unit_state(ctx, units, shas)
    for u in units:
        if st[u.id]["status"] in ("pending", "resumable"):
            rec = read_json(ctx.receipts(f"units/{u.id}.json"))
            if rec and rec.get("status") in ("failed", "identity_mismatch"):
                last = rec["attempts"][-1]
                st[u.id] = {**st[u.id], "status": "failed" if rec["status"] == "failed" else "mismatch",
                            "reason": (last.get("worker") or {}).get("error") or f"exit code {last.get('exit_code')}"}
    return st


# ---------------------------------------------------------------------------------------------------------------------
# stage 1: tests
# ---------------------------------------------------------------------------------------------------------------------
def default_tests_command() -> list:
    files = sorted(str(p.relative_to(REPO)) for p in (REPO / "tests").glob("test_p10_evolved_*.py"))
    if not files:
        raise StageError(f"no tests/test_p10_evolved_*.py under {REPO}")
    return [sys.executable, "-m", "pytest", "-q", "-m", "not slow", "-p", "no:cacheprovider", *files]


def stage_tests(ctx: Ctx) -> dict:
    cmd = ctx.tests_command or default_tests_command()
    log(f"tests: {' '.join(cmd[:8])}{' ...' if len(cmd) > 8 else ''}")
    rc, wall, gib = run_blocking(cmd, child_env(ctx.threads), ctx.log_path("tests"), cwd=REPO)
    lines = [ln for ln in C.tail(ctx.log_path("tests"), 400).splitlines() if ln.strip()]
    return {"status": "ok" if rc == 0 else "failed", "command": cmd, "cwd": str(REPO), "returncode": rc, "passed": rc == 0,
            "summary": lines[-1] if lines else "", "log": "logs/tests.log", "tail": "\n".join(lines[-25:]), "wall_s": wall,
            "peak_rss_gib": gib}


# ---------------------------------------------------------------------------------------------------------------------
# stage 2: verify (inputs + bundles)
# ---------------------------------------------------------------------------------------------------------------------
def _read_identity(d: Path):
    return read_json(d / "identity.json")


def _bundle_problems(ctx: Ctx, arm: str, n: int, ident: dict, shas: dict | None) -> list:
    bad = []
    if ident.get("arm") != arm or ident.get("n") != n:
        bad.append(f"identity is of ({ident.get('arm')}, N{ident.get('n')}), not ({arm}, N{n})")
    if not ident.get("bundle_sha256"):
        bad.append("no bundle_sha256")
    if ctx.synthetic:
        if ident.get("synthetic") is not True:
            bad.append("a non-synthetic bundle in a synthetic campaign")
    else:
        if ident.get("synthetic"):
            bad.append("a synthetic bundle in a real campaign")
        want = shas or {}
        for key, kind in (("nodal_metric_file_sha256", "nodal"), ("laplacian_metric_file_sha256", "laplacian")):
            if ident.get(key) != want.get(kind):
                bad.append(f"{key} {ident.get(key)} != the packed input {want.get(kind)}")
    if (arm == "raw") != (ident.get("arm_identity") is None):
        bad.append(f"arm_identity {'set' if ident.get('arm_identity') is not None else 'None'} for arm {arm!r}")
    agree = ident.get("agreement") or {}
    for key in ("jac", "Hp"):
        if key not in agree:
            bad.append(f"no {key} agreement record")
    return bad


def _reusable_bundle(ctx: Ctx, arm: str, n: int, shas: dict | None) -> bool:
    d = ctx.bundle_dir(arm, n)
    ident = _read_identity(d)
    return bool(ident and (d / "bundle.npz").is_file() and not _bundle_problems(ctx, arm, n, ident, shas))


def stage_verify(ctx: Ctx) -> dict:
    pairs = sorted({(u.arm, u.n) for u in ctx.units})
    rec: dict = {"inputs": None, "bundles": {}, "problems": []}
    shas_all: dict = {}
    if ctx.synthetic:
        rec["inputs"] = {"synthetic": True}
    else:
        rep = INP.verify_inputs(ctx.inputs, require=pairs)
        rec["inputs"] = {"ok": rep["ok"], "n_files": rep["n_files"], "total_bytes": rep["total_bytes"],
                         "manifest_sha256": rep["manifest_sha256"], "extras": rep["extras"], "errors": rep["errors"]}
        if not rep["ok"]:
            raise StageError("input verification failed:\n  " + "\n  ".join(rep["errors"]))
        if rep["manifest_sha256"] != ctx.manifest["inputs_manifest_sha256"]:
            raise StageError("the inputs manifest changed since the campaign manifest was written")
        shas_all = INP.file_shas(ctx.inputs)
        log(f"verify: inputs ok ({rep['n_files']} files, {rep['total_bytes'] / 2**20:.1f} MiB)")
    tasks = [Task(f"bundle.{arm}.n{n}", n, float(n) ** 3, arm) for arm, n in pairs
             if not _reusable_bundle(ctx, arm, n, shas_all.get((arm, n)))]
    for arm, n in pairs:
        if not any(t.id == f"bundle.{arm}.n{n}" for t in tasks):
            log(f"verify: bundle {arm}/n{n} reused ({ctx.bundle_dir(arm, n)})")
    builds: dict = {}

    def launch(t: Task):
        log(f"verify: building {t.id} (estimate {ctx.mem_of(t):.1f} GiB)")
        return spawn(ctx.worker_cmd("bundle", arm=t.arm, n=t.n), child_env(ctx.threads), ctx.log_path(t.id), cwd=SCRIPTS,
                     header=f"=== {C.iso_now()} {t.id}", append=False)

    def finish(t: Task, rc, gib, t0, t1):
        builds[t.id] = {"returncode": rc, "wall_s": t1 - t0, "peak_rss_gib_child": gib}
        log(f"verify: {t.id} exited {rc} in {t1 - t0:.1f} s, peak RSS {gib:.2f} GiB")

    interrupted, _left = run_pool(C.order_units(tasks), workers=ctx.workers, budget_gb=ctx.memory_gb, mem_of=ctx.mem_of,
                                  launch=launch, finish=finish)
    if interrupted:
        return {"status": "interrupted", **rec}
    prev = (read_json(ctx.receipts("verify.json"), {}) or {}).get("bundles") or {}
    arm_ids: dict = {}
    failed = []
    for arm, n in pairs:
        key = f"{arm}/n{n}"
        tid = f"bundle.{arm}.n{n}"
        d = ctx.bundle_dir(arm, n)
        ident = _read_identity(d)
        b = builds.get(tid)
        if b is not None and b["returncode"] != 0:
            entry = {"status": "failed", "problems": [f"build exited {b['returncode']}"], "build": b,
                     "log_tail": C.tail(ctx.log_path(tid), 12)}
        elif ident is None or not (d / "bundle.npz").is_file():
            entry = {"status": "failed", "problems": ["no bundle written"], "build": b}
        else:
            probs = _bundle_problems(ctx, arm, n, ident, shas_all.get((arm, n)))
            worker = read_json(ctx.receipts(f"workers/{tid}.json"), {}) if b is not None else {}
            entry = {"status": "ok" if not probs else "failed", "problems": probs,
                     "bundle_sha256": ident.get("bundle_sha256"), "arm": arm, "n": n, "path": str(d.relative_to(ctx.root)),
                     "agreement": ident.get("agreement"), "arm_identity": ident.get("arm_identity"),
                     "nodal_metric_identity": ident.get("nodal_metric_identity"),
                     "laplacian_metric_identity": ident.get("laplacian_metric_identity"),
                     "nodal_metric_file_sha256": ident.get("nodal_metric_file_sha256"),
                     "laplacian_metric_file_sha256": ident.get("laplacian_metric_file_sha256"),
                     "sidecar_sha256": ident.get("sidecar_sha256"), "layout_sha256": ident.get("layout_sha256"),
                     "nodal_plan_sha256": ident.get("nodal_plan_sha256"),
                     "laplacian_plan_sha256": ident.get("laplacian_plan_sha256"), "bundle_git_commit": ident.get("git_commit"),
                     "synthetic": bool(ident.get("synthetic")), "reused": b is None, "build": b, "worker": worker}
            arm_ids.setdefault(arm, set()).add(json.dumps(ident.get("arm_identity"), sort_keys=True))
        rec["bundles"][key] = entry
        if entry["status"] != "ok":
            failed.append(key)
            log(f"verify: {key} FAILED: {'; '.join(entry['problems'])}")
    rec["notes"] = [f"arm {arm}: {len(v)} distinct arm_identity values across N" for arm, v in arm_ids.items() if len(v) > 1]
    merged = dict(prev)
    merged.update(rec["bundles"])
    rec["bundles"] = merged
    rec["failed"] = failed
    return {"status": "ok" if not failed else "failed", **rec}


# ---------------------------------------------------------------------------------------------------------------------
# stage 3: preflight
# ---------------------------------------------------------------------------------------------------------------------
def _preflight_valid(ctx: Ctx, d: Path, arm, n, mode, pattern, sha: str) -> dict | None:
    res = read_json(d / "preflight.json")
    if not res or res.get("schema") != "drbx.p10-preflight-v1":
        return None
    ident = res.get("identity") or {}
    want_params = C.params_kwargs(ctx.config)
    if (ident.get("bundle_sha256") != sha or res.get("mode") != mode or res.get("pattern") != pattern
            or res.get("T") != float(ctx.config["T"]) or res.get("params") != want_params):
        return None
    return res


def stage_preflight(ctx: Ctx) -> dict:
    shas = ctx.bundle_shas()
    main = [u for u in ctx.units if u.stage == "main"]
    groups = C.groups_of(main)
    if not groups:
        return {"status": "ok", "note": "no main unit selected", "entries": {}}
    todo: dict = {}
    for arm, n, mode, pattern in groups:
        if _preflight_valid(ctx, ctx.preflight_dir(arm, n, mode, pattern), arm, n, mode, pattern, shas[f"{arm}/n{n}"]) is None:
            todo.setdefault((arm, n), []).append((mode, pattern))
    tasks = [Task(f"preflight.{arm}.n{n}", n, float(n) ** 3 * len(g), arm) for (arm, n), g in sorted(todo.items())]
    procs: dict = {}

    def launch(t: Task):
        n = t.n
        log(f"preflight: {t.id}: {len(todo[(t.arm, n)])} call(s) (estimate {ctx.mem_of(t):.1f} GiB)")
        return spawn(ctx.worker_cmd("preflight", arm=t.arm, n=n), child_env(ctx.threads), ctx.log_path(t.id), cwd=SCRIPTS,
                     header=f"=== {C.iso_now()} {t.id}", append=False)

    def finish(t: Task, rc, gib, t0, t1):
        procs[t.id] = {"returncode": rc, "wall_s": t1 - t0, "peak_rss_gib_child": gib}
        log(f"preflight: {t.id} exited {rc} in {t1 - t0:.1f} s, peak RSS {gib:.2f} GiB")

    interrupted, _left = run_pool(C.order_units(tasks), workers=ctx.workers, budget_gb=ctx.memory_gb, mem_of=ctx.mem_of,
                                  launch=launch, finish=finish)
    if interrupted:
        return {"status": "interrupted"}
    prev = (read_json(ctx.receipts("preflight.json"), {}) or {}).get("entries") or {}
    entries, violations, failures, warnings = {}, [], [], []
    for arm, n, mode, pattern in groups:
        key = f"{arm}/n{n}/{mode}/{pattern}"
        proc = procs.get(f"preflight.{arm}.n{n}")
        res = _preflight_valid(ctx, ctx.preflight_dir(arm, n, mode, pattern), arm, n, mode, pattern, shas[f"{arm}/n{n}"])
        if res is None:
            failures.append(key)
            entries[key] = {"status": "failed", "process": proc, "log_tail": C.tail(ctx.log_path(f"preflight.{arm}.n{n}"), 12)}
            log(f"preflight: {key} FAILED (no valid preflight.json)")
            continue
        unit = next(u for u in main if u.group == (arm, n, mode, pattern))
        lam = max(res["lambda_max"])
        wrec = (read_json(ctx.receipts(f"workers/preflight.{arm}.n{n}.json"), {}) or {}).get("calls", {}).get(f"{mode}_{pattern}", {})
        entry = {"status": "ok", "arm": arm, "n": n, "mode": mode, "pattern": pattern, "dt": unit.dt, "nsteps": unit.nsteps,
                 "T": unit.T, "lambda_max": res["lambda_max"], "lambda_max_used": lam, "lambda_dt": lam * unit.dt,
                 "lambda_dt_half": lam * unit.dt / 2.0, "gate": C.PREFLIGHT_GATE, "method": res.get("method"),
                 "converged": res.get("converged"), "rule_dt": res.get("dt"), "rule_nsteps": res.get("nsteps"),
                 "preflight_seconds": res.get("seconds"), "call_wall_s": wrec.get("wall_s"),
                 "peak_rss_gib_after_call": wrec.get("peak_rss_gib"), "reused": (proc is None),
                 "path": str(ctx.preflight_dir(arm, n, mode, pattern).relative_to(ctx.root) / "preflight.json")}
        if lam * unit.dt > C.PREFLIGHT_GATE:
            entry["status"] = "violation"
            entry["min_nsteps"] = C.ceil_nsteps(unit.T, lam)
            violations.append(entry)
        if res.get("method") != "arpack" or not res.get("converged"):
            warnings.append(f"{key}: lambda_max by method {res.get('method')!r}, converged {res.get('converged')}")
        entries[key] = entry
    for e in violations:
        log(f"preflight: VIOLATION {e['arm']}/n{e['n']}/{e['mode']}/{e['pattern']}: lambda_max * dt = {e['lambda_dt']:.4g} > "
            f"{C.PREFLIGHT_GATE} (lambda_max {e['lambda_max_used']:.4g}, dt {e['dt']:.4g}, nsteps {e['nsteps']}; "
            f"at least {e['min_nsteps']} steps are needed): the campaign is stopped")
    for w in warnings:
        log(f"preflight: WARNING {w}")
    merged = dict(prev)
    merged.update(entries)
    status = "ok" if not violations and not failures else "failed"
    return {"status": status, "entries": merged, "violations": [f"{e['arm']}/n{e['n']}/{e['mode']}/{e['pattern']}" for e in violations],
            "failures": failures, "warnings": warnings}


# ---------------------------------------------------------------------------------------------------------------------
# stage 4: run
# ---------------------------------------------------------------------------------------------------------------------
def _unit_receipt_path(ctx: Ctx, unit) -> Path:
    return ctx.receipts(f"units/{unit.id}.json")


def _worker_path(ctx: Ctx, unit) -> Path:
    return ctx.receipts(f"workers/unit_{unit.id}.json")


def stage_run(ctx: Ctx) -> dict:
    shas = ctx.bundle_shas()
    state = unit_state(ctx, ctx.units, shas)
    todo = C.order_units([u for u in ctx.units if state[u.id]["status"] in ("pending", "resumable")])
    skipped = [u.id for u in ctx.units if state[u.id]["status"] in ("complete", "stopped")]
    blocked = {u.id: state[u.id] for u in ctx.units if state[u.id]["status"] in ("mismatch", "damaged")}
    log(f"run: {len(ctx.units)} units selected: {len(todo)} to run ({sum(state[u.id]['status'] == 'resumable' for u in todo)} "
        f"resumed), {len(skipped)} already finished, {len(blocked)} blocked (identity mismatch / damaged)")
    for uid, st in blocked.items():
        log(f"run: BLOCKED {uid}: {st['status']}: {st['reason']}")
    by_id = {u.id: u for u in todo}
    results: dict = {}

    def launch(u):
        rp = _unit_receipt_path(ctx, u)
        prev = read_json(rp, {}) or {}
        attempt = len(prev.get("attempts", [])) + 1
        _worker_path(ctx, u).unlink(missing_ok=True)
        log(f"run: start {u.id} (attempt {attempt}, {state[u.id]['status']}, estimate {ctx.mem_of(u):.1f} GiB, "
            f"step {state[u.id]['step']}/{u.nsteps})")
        return spawn(ctx.worker_cmd("unit", unit_id=u.id), child_env(ctx.threads), ctx.log_path(u.id), cwd=SCRIPTS,
                     header=f"=== {C.iso_now()} attempt {attempt} {u.id}")

    def finish(u, rc, gib, t0, t1):
        worker = read_json(_worker_path(ctx, u), {}) or {}
        run_state = C.classify_run(ctx.root, u, ctx.config, shas[f"{u.arm}/n{u.n}"])
        if rc == 0 and run_state["status"] == "complete":
            status = "complete"
        elif rc == 1 and run_state["status"] == "stopped":
            status = "stopped"
        elif rc == EXIT_MISMATCH or run_state["status"] in ("mismatch", "damaged"):
            status = "identity_mismatch" if rc == EXIT_MISMATCH or run_state["status"] == "mismatch" else "damaged"
        elif rc < 0 and _STOP.is_set():
            status = "interrupted"
        else:
            status = "failed"
        att = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t1)),
               "wall_s": t1 - t0, "exit_code": rc, "status": status, "run_status": run_state["status"], "step": run_state["step"],
               "peak_rss_gib": gib, "estimate_gib": ctx.mem_of(u), "backend": worker.get("backend"),
               "resumed_from_step": worker.get("resumed_from_step"), "worker": worker}
        if status in ("failed", "identity_mismatch", "damaged"):
            att["log_tail"] = C.tail(ctx.log_path(u.id), 15)
        rp = _unit_receipt_path(ctx, u)
        rec = read_json(rp, {}) or {"unit_id": u.id, "unit": u.to_json(), "attempts": []}
        rec["attempts"].append(att)
        rec.update(status=status, backend=worker.get("backend", rec.get("backend")),
                   peak_rss_gib=max(a["peak_rss_gib"] for a in rec["attempts"]), log=f"logs/{u.id}.log",
                   wall_s_total=sum(a["wall_s"] for a in rec["attempts"]))
        write_json(rp, rec)
        results[u.id] = status
        log(f"run: {u.id} -> {status} in {t1 - t0:.1f} s, peak RSS {gib:.2f} GiB")

    interrupted, left = run_pool(todo, workers=ctx.workers, budget_gb=ctx.memory_gb, mem_of=ctx.mem_of, launch=launch, finish=finish)
    final = reported_state(ctx, ctx.units, shas)
    counts: dict = {}
    for st in final.values():
        counts[st["status"]] = counts.get(st["status"], 0) + 1
    failed = sorted(uid for uid, s in results.items() if s == "failed")
    for u in left:
        results[u.id] = "not_started"
    ok = all(st["status"] == "complete" for st in final.values()) and not interrupted
    return {"status": "interrupted" if interrupted else ("ok" if ok else "incomplete"), "selected": len(ctx.units),
            "launched": len(todo) - len(left), "skipped_finished": len(skipped), "blocked": blocked, "counts": counts,
            "failed": failed, "stopped": sorted(u for u, s in final.items() if s["status"] == "stopped"),
            "this_invocation": results}


# ---------------------------------------------------------------------------------------------------------------------
# stage 5: validate
# ---------------------------------------------------------------------------------------------------------------------
def stage_validate(ctx: Ctx) -> dict:
    shas = ctx.bundle_shas()
    state = reported_state(ctx, ctx.units, shas)
    counts: dict = {}
    for st in state.values():
        counts[st["status"]] = counts.get(st["status"], 0) + 1
    not_complete = {uid: {"status": s["status"], "reason": s["reason"]} for uid, s in state.items() if s["status"] != "complete"}
    commits = sorted({(read_json(u.path(ctx.root) / "run.json", {}) or {}).get("git_commit") or "?" for u in ctx.units
                      if state[u.id]["status"] in ("complete", "stopped")})
    rc, wall, gib = run_blocking(ctx.worker_cmd("analyze"), child_env(ctx.threads), ctx.log_path("analyze"), cwd=SCRIPTS,
                                 header=f"=== {C.iso_now()} analyze", append=False)
    worker = read_json(ctx.receipts("workers/analyze.json"), {}) or {}
    analysis_ok = rc == 0
    ok = analysis_ok and not not_complete
    if not_complete:
        log(f"validate: {len(not_complete)} of {len(ctx.units)} selected units are not complete: " + ", ".join(sorted(not_complete)[:6])
            + (" ..." if len(not_complete) > 6 else ""))
    if not analysis_ok:
        log("validate: the analysis subprocess failed:\n" + C.tail(ctx.log_path("analyze"), 15))
    return {"status": "ok" if ok else "failed", "selected": len(ctx.units), "counts": counts, "not_complete": not_complete,
            "git_commits_of_runs": commits, "manifest_git_commit": ctx.manifest.get("git_commit"),
            "analysis": {"returncode": rc, "wall_s": wall, "peak_rss_gib": gib, **worker}}


STAGE_FUNCS = {"tests": stage_tests, "verify": stage_verify, "preflight": stage_preflight, "run": stage_run,
               "validate": stage_validate}


# ---------------------------------------------------------------------------------------------------------------------
# receipts and the stage driver
# ---------------------------------------------------------------------------------------------------------------------
def write_campaign_receipt(ctx: Ctx) -> None:
    stages = {}
    for s in STAGES:
        r = read_json(ctx.receipts(f"{s}.json"))
        if r:
            stages[s] = {k: r.get(k) for k in ("status", "started", "ended", "seconds", "commit", "selected_units")}
    shas = ctx.bundle_shas(required=False)
    units = ctx.all_units
    state = reported_state(ctx, units, shas)
    counts, ids = {}, {}
    for uid, st in state.items():
        counts[st["status"]] = counts.get(st["status"], 0) + 1
        ids.setdefault(st["status"], []).append(uid)
    peaks, backends, wall = {}, set(), {}
    for u in units:
        rec = read_json(_unit_receipt_path(ctx, u))
        if rec:
            peaks[u.id] = rec.get("peak_rss_gib")
            wall[u.id] = rec.get("wall_s_total")
            if rec.get("backend"):
                backends.add(rec["backend"])
    by_n: dict = {}
    for u in units:
        if peaks.get(u.id) is not None:
            by_n[str(u.n)] = max(by_n.get(str(u.n), 0.0), peaks[u.id])
    inv = (read_json(ctx.receipts("invocations.json"), []) or [])[-20:]
    write_json(ctx.receipts("campaign.json"), {
        "schema": RECEIPT_SCHEMA, "updated": C.iso_now(), "git_commit": ctx.commit, "manifest_git_commit": ctx.manifest.get("git_commit"),
        "configuration_sha256": ctx.manifest["configuration_sha256"], "inputs_manifest_sha256": ctx.manifest["inputs_manifest_sha256"],
        "synthetic": ctx.synthetic, "stages": stages, "units": {"total": len(units), "counts": counts, "ids": ids},
        "backend": sorted(backends), "workers": ctx.workers, "threads": ctx.threads, "memory_gb": ctx.memory_gb,
        "memory_table_gib": ctx.mem_table, "peak_rss_gib": peaks, "peak_rss_gib_max_by_n": by_n, "wall_s_by_unit": wall,
        "host": {"platform": sys.platform, "cpu_count": os.cpu_count(), "python": sys.version.split()[0]},
        "invocations": inv})


def run_stage(ctx: Ctx, name: str) -> str:
    t0, started = time.time(), C.iso_now()
    log(f"=== stage {name} ===")
    try:
        rec = STAGE_FUNCS[name](ctx)
    except StageError as exc:
        log(f"stage {name} FAILED: {exc}")
        rec = {"status": "failed", "error": str(exc)}
    except KeyboardInterrupt:
        rec = {"status": "interrupted"}
        _finalize_stage(ctx, name, rec, started, t0)
        raise
    except Exception as exc:                                                       # noqa: BLE001
        log(f"stage {name} FAILED with an unexpected error:\n{traceback.format_exc()}")
        rec = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    _finalize_stage(ctx, name, rec, started, t0)
    log(f"stage {name}: {rec['status']} ({time.time() - t0:.1f} s)")
    return rec["status"]


def _finalize_stage(ctx: Ctx, name: str, rec: dict, started: str, t0: float) -> None:
    rec = {"stage": name, "started": started, "ended": C.iso_now(), "seconds": time.time() - t0, "commit": ctx.commit,
           "selected_units": len(ctx.units), "only": ctx.only, "workers": ctx.workers, "threads": ctx.threads,
           "memory_gb": ctx.memory_gb, **rec}
    write_json(ctx.receipts(f"{name}.json"), rec)
    write_campaign_receipt(ctx)


def log_invocation(ctx: Ctx, stage: str) -> None:
    path = ctx.receipts("invocations.json")
    inv = read_json(path, []) or []
    inv.append({"time": C.iso_now(), "stage": stage, "argv": sys.argv, "commit": ctx.commit, "workers": ctx.workers,
                "threads": ctx.threads, "memory_gb": ctx.memory_gb, "only": ctx.only})
    write_json(path, inv)


# ---------------------------------------------------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------------------------------------------------
def dry_run(ctx: Ctx, stage: str) -> int:
    stages = list(STAGES) if stage == "all" else [stage]
    counts = C.matrix_counts(ctx.all_units)
    print(f"campaign root     {ctx.root}")
    print(f"git commit        {ctx.commit}")
    print(f"configuration sha {C.canonical_sha(ctx.config)}")
    print(f"inputs manifest   {ctx.inputs_sha}")
    print(f"units             {counts['total']} (stage {counts['by_stage']}; arm {counts['by_arm']}; N {counts['by_n']})")
    print(f"selected          {len(ctx.units)}" + (f" (--only {ctx.only!r})" if ctx.only else ""))
    print(f"workers x threads {ctx.workers} x {ctx.threads} (cpu count {os.cpu_count()}); memory budget "
          f"{'none' if ctx.memory_gb is None else f'{ctx.memory_gb} GiB'}; per-unit estimates GiB {ctx.mem_table}")
    if ctx.workers * ctx.threads > (os.cpu_count() or 1):
        print(f"WARNING: workers x threads = {ctx.workers * ctx.threads} exceeds the {os.cpu_count()} cores")
    pairs = sorted({(u.arm, u.n) for u in ctx.units})
    groups = C.groups_of([u for u in ctx.units if u.stage == "main"])
    shas = ctx.bundle_shas(required=False)
    for s in stages:
        print(f"\n--- stage {s}")
        if s == "tests":
            print("  " + " ".join(ctx.tests_command or ["python", "-m", "pytest", "-q", "-m", "not slow", "tests/test_p10_evolved_*.py"]))
        elif s == "verify":
            print(f"  verify the inputs manifest, build {len(pairs)} bundles: " + ", ".join(f"{a}/n{n}" for a, n in pairs))
        elif s == "preflight":
            print(f"  {len(groups)} (arm, N, mode, pattern) calls in {len(pairs)} processes (rightmost=False; gate "
                  f"lambda_max * dt <= {C.PREFLIGHT_GATE})")
        elif s == "run":
            order = C.order_units(ctx.units)
            state = reported_state(ctx, ctx.units, shas)
            events = C.simulate_schedule([u for u in order if state[u.id]["status"] in ("pending", "resumable", "unverified")],
                                         ctx.workers, ctx.memory_gb, ctx.mem_of)
            start = {e["id"]: e for e in events}
            print(f"  {'#':>3} {'unit':<58} {'N':>3} {'steps':>5} {'dt':>10} {'GiB':>4} {'cost':>8} {'state':>10} {'sim.start':>9} {'conc':>4}")
            for i, u in enumerate(order, 1):
                st = state[u.id]["status"]
                ev = start.get(u.id)
                print(f"  {i:>3} {u.id:<58} {u.n:>3} {u.nsteps:>5} {u.dt:>10.4g} {ctx.mem_of(u):>4.1f} {u.cost:>8.0f} {st:>10} "
                      f"{('%9.0f' % ev['start']) if ev else '        -'} {ev['n_running'] if ev else '-':>4}")
            if events:
                print(f"  simulated (cost units): makespan {max(e['end'] for e in events):.0f}, max concurrent "
                      f"{max(e['n_running'] for e in events)}, max in-use estimate {max(e['in_use_gb'] for e in events):.1f} GiB")
        elif s == "validate":
            print("  completeness + reduce main / each variant tree + variants.json")
    print("\n(dry run: nothing was computed or written)")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# workers (subprocess entry points; these import JAX)
# ---------------------------------------------------------------------------------------------------------------------
def _backend_check() -> dict:
    import jax

    backend = jax.default_backend()
    if backend != "cpu":
        raise RuntimeError(f"jax.default_backend() = {backend!r}, expected 'cpu' (JAX_PLATFORMS=cpu)")
    return {"backend": backend, "devices": [str(d) for d in jax.devices()], "x64": bool(jax.config.jax_enable_x64),
            "jax": jax.__version__}


def worker_bundle(args) -> int:
    t0 = time.time()
    ctx = Ctx(args, write_manifest=False)
    arm, n = args.arm, int(args.n)
    wrec = {"task": f"bundle.{arm}.n{n}", "started": C.iso_now(), **_backend_check()}
    out = ctx.bundle_dir(arm, n)
    from p10_evolved_mms import bundle as B

    if ctx.synthetic:
        from p10_evolved_mms import synthetic as syn

        b = syn.synthetic_bundle(n, SYNTHETIC_N_ETA, arm)
        B.save_bundle(b, out)
    else:
        b = B.build_bundle(arm, n, out, nodal_root=INP.nodal_root(ctx.inputs, arm), laplacian_root=INP.laplacian_root(ctx.inputs))
    wrec.update(bundle_sha256=b.identity["bundle_sha256"], wall_s=time.time() - t0, peak_rss_gib=self_peak_gib(), ended=C.iso_now())
    write_json(ctx.receipts(f"workers/bundle.{arm}.n{n}.json"), wrec)
    print(json.dumps({"bundle_sha256": wrec["bundle_sha256"], "wall_s": wrec["wall_s"]}), flush=True)
    return 0


def worker_preflight(args) -> int:
    ctx = Ctx(args, write_manifest=False)
    arm, n = args.arm, int(args.n)
    shas = ctx.bundle_shas()
    wpath = ctx.receipts(f"workers/preflight.{arm}.n{n}.json")
    wrec = {"task": f"preflight.{arm}.n{n}", "started": C.iso_now(), **_backend_check(), "calls": {}}
    import numpy as np
    from p10_evolved_mms import bundle as B, preflight as PF
    from p10_evolved_mms.fields import default_params

    todo = [g for g in C.groups_of([u for u in ctx.units if u.stage == "main"]) if g[0] == arm and g[1] == n]
    todo = [g for g in todo if _preflight_valid(ctx, ctx.preflight_dir(*g), *g, shas[f"{arm}/n{n}"]) is None]
    if not todo:
        return 0
    bundle = B.load_bundle(ctx.bundle_dir(arm, n), expected_identity=shas[f"{arm}/n{n}"])
    kw = C.params_kwargs(ctx.config)
    kw["D"] = np.asarray(kw["D"])
    p = default_params(**kw)
    code = 0
    for _arm, _n, mode, pattern in todo:
        t0 = time.time()
        try:
            res = PF.preflight(bundle, mode, pattern, p, float(ctx.config["T"]), out=ctx.preflight_dir(arm, n, mode, pattern),
                               rightmost=False, log=lambda s: print(s, flush=True))
            wrec["calls"][f"{mode}_{pattern}"] = {"wall_s": time.time() - t0, "peak_rss_gib": self_peak_gib(), "method": res["method"],
                                                  "converged": res["converged"], "lambda_max": res["lambda_max"]}
        except Exception as exc:                                                    # noqa: BLE001
            traceback.print_exc()
            wrec["calls"][f"{mode}_{pattern}"] = {"error": f"{type(exc).__name__}: {exc}", "wall_s": time.time() - t0}
            code = 2
        write_json(wpath, wrec)
    return code


def worker_unit(args) -> int:
    t0 = time.time()
    ctx = Ctx(args, write_manifest=False)
    unit = next(u for u in ctx.all_units if u.id == args.unit_id)
    wrec = {"unit_id": unit.id, "pid": os.getpid(), "started": C.iso_now(), "status": "error"}
    code = 2
    try:
        wrec.update(_backend_check())
        import numpy as np
        from p10_evolved_mms import bundle as B, evolve as EV
        from p10_evolved_mms.fields import default_params

        sha = ctx.bundle_shas(required=False).get(f"{unit.arm}/n{unit.n}")
        if sha is None:
            raise RuntimeError(f"no verified bundle identity for {unit.arm}/n{unit.n} (receipts/verify.json)")
        try:
            bundle = B.load_bundle(ctx.bundle_dir(unit.arm, unit.n), expected_identity=sha)
        except ValueError as exc:
            wrec.update(status="identity_mismatch", error=str(exc))
            print(f"IDENTITY MISMATCH (bundle): {exc}", flush=True)
            code = EXIT_MISMATCH
            return code
        kw = C.params_kwargs(ctx.config, unit.params_override)
        kw["D"] = np.asarray(kw["D"])
        p = default_params(**kw)
        try:
            record = EV.run(bundle, unit.mode, unit.source, p, unit.pattern, dt=unit.dt, T=unit.T, out_dir=unit.path(ctx.root),
                            chunk=unit.chunk, resume=True, opts_override=dict(unit.opts_override) or None,
                            on_chunk=lambda c: print(json.dumps(c), flush=True))
        except ValueError as exc:
            if "cannot resume" in str(exc):
                wrec.update(status="identity_mismatch", error=str(exc))
                print(f"IDENTITY MISMATCH (stored run): {exc}", flush=True)
                code = EXIT_MISMATCH
                return code
            raise
        resumes = record.get("resumes") or []
        wrec.update(status=record["status"], stop_reason=record.get("stop_reason"), stop_step=record.get("stop_step"),
                    resumed_from_step=(resumes[-1]["step"] if resumes else None), n_resumes=len(resumes),
                    max_cg_iterations=max((c["max_cg_iterations"] for c in record["chunks"]), default=None),
                    timings=record.get("timings"), run_git_commit=record.get("git_commit"))
        code = 0 if record["status"] == "complete" else 1
        return code
    except Exception as exc:                                                        # noqa: BLE001
        traceback.print_exc()
        wrec.update(status="error", error=f"{type(exc).__name__}: {exc}")
        code = 2
        return code
    finally:
        wrec.update(wall_s=time.time() - t0, peak_rss_gib=self_peak_gib(), ended=C.iso_now(), exit_code=code)
        write_json(_worker_path(ctx, unit), wrec)


def _analysis_summary(analysis: dict) -> dict:
    return {"groups": len(analysis["groups"]), "runs": len(analysis["runs"]), "missing_runs": len(analysis["missing_runs"]),
            "excluded_runs": len(analysis["excluded_runs"]), "warnings": analysis["warnings"]}


def worker_analyze(args) -> int:
    ctx = Ctx(args, write_manifest=False)
    from p10_evolved_mms import reduce as RD

    summary: dict = {"started": C.iso_now(), "main": None, "variants": {}}
    out = ctx.root / "analysis"
    main_units = [u for u in ctx.units if u.stage == "main"]
    if main_units:
        analysis = RD.reduce_campaign(ctx.root / "main", expected_n=sorted({u.n for u in main_units}))
        RD.write_outputs(analysis, out / "main")
        summary["main"] = _analysis_summary(analysis)
    for name in sorted({u.variant for u in ctx.units if u.variant}):
        vu = [u for u in ctx.units if u.variant == name]
        analysis = RD.reduce_campaign(ctx.root / "variants" / name, expected_n=sorted({u.n for u in vu}))
        RD.write_outputs(analysis, out / "variants" / name)
        summary["variants"][name] = _analysis_summary(analysis)
    comp = C.variant_comparisons(ctx.root, ctx.all_units, ctx.manifest["spec"])
    keep = {u.id for u in ctx.units}
    for name, v in comp["variants"].items():
        v["entries"] = [e for e in v["entries"] if e["unit"] in keep]
    comp["variants"] = {k: v for k, v in comp["variants"].items() if v["entries"]}
    write_json(out / "variants.json", comp)
    summary["variants_json"] = {k: [e["status"] for e in v["entries"]] for k, v in comp["variants"].items()}
    summary["ended"] = C.iso_now()
    write_json(ctx.receipts("workers/analyze.json"), summary)
    return 0


def worker_main(args) -> int:
    fn = {"bundle": worker_bundle, "preflight": worker_preflight, "unit": worker_unit, "analyze": worker_analyze}[args.worker]
    try:
        return fn(args)
    except Exception:                                                               # noqa: BLE001
        traceback.print_exc()
        return 2


# ---------------------------------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="campaign root (created)")
    ap.add_argument("--inputs", type=Path, help="packed inputs directory (python -m p10_evolved_mms.inputs pack)")
    ap.add_argument("--stage", choices=STAGES + ("all",))
    ap.add_argument("--workers", type=int, default=1, help="concurrent subprocesses")
    ap.add_argument("--threads", type=int, default=1, help="threads per subprocess (NPROC, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS)")
    ap.add_argument("--memory-gb", type=float, help="budget for the sum of the per-unit memory estimates (GiB)")
    ap.add_argument("--memory-table", help="per-N estimates in GiB, e.g. 32=1.5,48=3.0,64=4.5 (the default)")
    ap.add_argument("--only", help="regular expression selecting unit ids (re.search)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the simulated schedule; compute nothing")
    # hooks for the tests and internal worker entry points (not part of the operator interface)
    ap.add_argument("--synthetic", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config-override", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--tests-command", help=argparse.SUPPRESS)
    ap.add_argument("--worker", choices=WORKERS, help=argparse.SUPPRESS)
    ap.add_argument("--arm", help=argparse.SUPPRESS)
    ap.add_argument("--n", help=argparse.SUPPRESS)
    ap.add_argument("--unit-id", help=argparse.SUPPRESS)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.worker:
        return worker_main(args)
    if not args.stage:
        ap.error("--stage is required")
    try:
        ctx = Ctx(args, write_manifest=not args.dry_run)
    except (StageError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        return dry_run(ctx, args.stage)
    if ctx.workers * ctx.threads > (os.cpu_count() or 1):
        log(f"WARNING: workers x threads = {ctx.workers * ctx.threads} exceeds the {os.cpu_count()} cores")
    log_invocation(ctx, args.stage)
    status = "ok"
    _STOP.clear()
    old = {s: signal.signal(s, _handle_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for name in (STAGES if args.stage == "all" else (args.stage,)):
            status = run_stage(ctx, name)
            if status != "ok":
                break
    except KeyboardInterrupt:
        log("interrupted: children terminated; rerun the same command to resume")
        return 130
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    if status == "interrupted":
        log("interrupted: children terminated; rerun the same command to resume")
        return 130
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
