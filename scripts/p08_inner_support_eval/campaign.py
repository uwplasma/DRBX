#!/usr/bin/env python3
"""P08 inner-support evaluation campaign: C0 (profile7) vs candidate inner supports, node-local and resumable.

Run as ``python -m p08_inner_support_eval.campaign <command> ...`` from ``DRBX/scripts`` (see ``README.md``):

* ``verify-inputs``  hash-check the immutable HSX inputs, localize the continuum sidecar, write ``campaign_manifest.json``;
* ``preflight``      one grid-N32, ``--per-ring 1`` run of one candidate (into ``<output>/preflight``);
* ``run``            every grid x candidate ``run.py`` as an independent single-threaded subprocess, ``--jobs`` at a time;
* ``validate``       every expected file exists, no failures, zero dispatch violations;
* ``analyze``        ``analyze.py`` per candidate and ``compare_candidates`` over all, into ``<output>/summary/``.

The campaign identity covers the configuration, the candidates, per-ring, the input-manifest hash, the localized-sidecar
hash and the hashes of the sources this evaluation depends on (the git commit and dirty flag are recorded alongside);
an output folder with a different identity is refused.  Heavy imports (jax, the ``p08_step1_global`` helpers) are
deferred to the commands that need them, so parsing and the identity logic stay fast.

Do not put this package's own directory first on ``sys.path``; ``DRBX/scripts`` must be on the path.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = Path(__file__).resolve().parents[1]            # .../DRBX/scripts
REPO = SCRIPTS.parent                                    # .../DRBX
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SCHEMA = "drbx.p08-inner-support-eval-v1"
RUN_MODULE = "p08_inner_support_eval.run"
INPUT_MANIFEST_PATH = SCRIPTS / "p08_step1_global" / "input_manifest.json"
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

#: the harness and everything it reads (paths relative to ``REPO``); hashed into the identity
SOURCE_FILES = [
    *(f"scripts/p08_inner_support_eval/{name}" for name in (
        "campaign.py", "run.py", "q_fields.py", "sample_build.py", "dispatch.py", "analyze.py", "compare_candidates.py",
        "configuration.json", "sample.json")),
    "scripts/p08_step1_global/campaign.py",
    "scripts/p08_step1_global/input_manifest.json",
    "scripts/q_fci_layered_global/fields.py",
    "scripts/p_shared/inner_support.py",
    "scripts/p_shared/owner_closure.py",
    "scripts/p_shared/apply.py",
    "scripts/p_shared/provider.py",
    "scripts/p_shared/replay_support.py",
    "src/drbx/stencils/builder.py",
    "src/drbx/stencils/census.py",
    "src/drbx/geometry/fci_perpendicular_reconstruction.py",
    "src/drbx/geometry/fci_perpendicular_integrated_rows.py",
    "src/drbx/geometry/_fci_perpendicular_point_primitives.py",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def log(msg) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".campaign.lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another campaign process holds {folder}/.campaign.lock") from None
        yield


def config() -> dict:
    cfg = read_json(HERE / "configuration.json")
    if cfg["schema"] != SCHEMA:
        raise ValueError("unsupported P08 inner-support evaluation configuration")
    return cfg


# ---------------------------------------------------------------------------
# candidates and grids
# ---------------------------------------------------------------------------
def parse_candidates(text, cfg=None) -> dict:
    """``"C1=last_aggregate,C2=..."`` or bare configured names (``"C2"``) -> ordered ``{name: inner_support}``."""
    cfg = cfg or config()
    known = cfg["candidates"]
    out = {}
    for item in (t.strip() for t in str(text).split(",")):
        if not item:
            continue
        name, sep, support = item.partition("=")
        name, support = name.strip(), support.strip()
        if not sep:
            if name not in known:
                raise ValueError(f"unknown candidate {name!r}; give NAME=inner_support (configured: {sorted(known)})")
            support = known[name]
        if not NAME_RE.match(name) or name == "C0":
            raise ValueError(f"invalid candidate name {name!r}")
        from p_shared.inner_support import INNER_SUPPORT_CHOICES
        if support not in INNER_SUPPORT_CHOICES:
            raise ValueError(f"candidate {name}: inner_support must be one of {INNER_SUPPORT_CHOICES}, got {support!r}")
        if support == cfg["baseline_inner_support"]:
            raise ValueError(f"candidate {name}: {support!r} is the C0 baseline")
        if name in out:
            raise ValueError(f"duplicate candidate {name!r}")
        out[name] = support
    if not out:
        raise ValueError("no candidates given")
    if len(set(out.values())) != len(out):
        raise ValueError("two candidates share one inner_support")
    return out


def parse_grids(text, cfg=None) -> tuple:
    cfg = cfg or config()
    try:
        grids = tuple(int(t) for t in str(text).split(",") if t.strip())
    except ValueError:
        raise ValueError(f"grids must be comma-separated integers, got {text!r}") from None
    bad = [n for n in grids if n not in cfg["grids"]]
    if not grids or bad or len(set(grids)) != len(grids):
        raise ValueError(f"grids must be distinct members of {cfg['grids']}, got {text!r}")
    return grids


def positive_int(text):
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


# ---------------------------------------------------------------------------
# inputs, sources, identity
# ---------------------------------------------------------------------------
def _step1():
    """``p08_step1_global.campaign`` (imported lazily: it pulls in the whole p_shared stack)."""
    return importlib.import_module("p08_step1_global.campaign")


def load_input_manifest():
    """``(manifest dict, sha256 of input_manifest.json)`` -- the step-1 campaign's immutable input manifest."""
    return _step1()._input_manifest(), sha256_file(INPUT_MANIFEST_PATH)


def localize_sidecar(input_root, folder) -> Path:
    """``p08_step1_global.campaign.localize_sidecar``: ``<folder>/localized_sidecar.json`` pointing under ``input_root``."""
    return _step1().localize_sidecar(Path(input_root), Path(folder))


def verify_input_files(input_root, manifest, *, hash_files=True, say=log) -> int:
    """Existence, size and sha256 of every manifest file under ``input_root`` (the check of
    ``p08_step1_global.campaign.verify``); ``hash_files=False`` skips the hash.  Raises ``ValueError`` listing every
    missing or changed file; returns the number of files checked."""
    input_root = Path(input_root)
    bad = []
    for rec in manifest["files"]:
        path = input_root / rec["path"]
        if not path.is_file() or path.stat().st_size != rec["bytes"]:
            bad.append(str(path))
            continue
        if hash_files:
            say(f"  hashing {rec['path']} ({rec['bytes'] / 2 ** 20:.1f} MiB)")
            if sha256_file(path) != rec["sha256"]:
                bad.append(str(path))
    if bad:
        raise ValueError("missing or changed immutable input: " + ", ".join(bad))
    return len(manifest["files"])


def ensure_inputs(input_root, output, *, force=False, say=log) -> dict:
    """Hash-verify the immutable inputs once per ``output`` (stamp ``<output>/inputs_verified.json``, keyed by the
    manifest hash and the resolved input root); later calls only re-check existence and size.  ``force`` re-hashes."""
    input_root, output = Path(input_root), Path(output)
    manifest, manifest_sha = load_input_manifest()
    stamp_path = output / "inputs_verified.json"
    stamp = read_json(stamp_path) if stamp_path.exists() else None
    current = stamp is not None and stamp.get("input_manifest_sha256") == manifest_sha \
        and stamp.get("input_root") == str(input_root.resolve())
    if current and not force:
        verify_input_files(input_root, manifest, hash_files=False, say=say)
        return stamp
    say(f"verifying {len(manifest['files'])} immutable inputs under {input_root}")
    t0 = time.time()
    n = verify_input_files(input_root, manifest, hash_files=True, say=say)
    stamp = dict(input_manifest_sha256=manifest_sha, input_root=str(input_root.resolve()), files=n,
                 hash_seconds=time.time() - t0, verified_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    write_json(stamp_path, stamp)
    say(f"inputs verified ({n} files, {stamp['hash_seconds']:.0f} s)")
    return stamp


def source_hashes() -> dict:
    return {p: sha256_file(REPO / p) for p in SOURCE_FILES}


def git_state() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL)

    try:
        commit = git("rev-parse", "HEAD").strip()
        dirty_all = [ln for ln in git("status", "--porcelain").splitlines() if ln.strip()]
        dirty_src = [ln for ln in git("status", "--porcelain", "--", *SOURCE_FILES).splitlines() if ln.strip()]
    except Exception:
        return dict(commit=None, dirty=None, dirty_sources=None)
    return dict(commit=commit, dirty=bool(dirty_all), dirty_sources=dirty_src[:50])


def prepare_campaign(*, input_root, folder, candidates, per_ring, kind="campaign", say=log) -> tuple:
    """Localize the sidecar, compute the identity and write ``campaign_manifest.json`` into ``folder`` -- or refuse
    (``ValueError``) when ``folder`` already holds a manifest of a different identity.  Returns ``(identity, sidecar)``."""
    input_root, folder = Path(input_root), Path(folder)
    cfg = config()
    _manifest, manifest_sha = load_input_manifest()
    try:
        sidecar = localize_sidecar(input_root, folder)
    except ValueError as exc:
        raise ValueError(f"{exc}; use a new output folder") from None
    sidecar_sha = sha256_file(sidecar)
    sources = source_hashes()
    identity = digest(dict(configuration=cfg, candidates=dict(candidates), per_ring=int(per_ring),
                           input_manifest_sha256=manifest_sha, localized_sidecar_sha256=sidecar_sha, sources=sources))
    path = folder / "campaign_manifest.json"
    if path.exists():
        saved = read_json(path)
        if saved["identity"] != identity:
            raise ValueError(f"campaign identity changed relative to {path}; use a new output folder")
        return identity, sidecar
    write_json(path, dict(schema=SCHEMA, kind=kind, identity=identity, configuration=cfg,
                          candidates=dict(candidates), per_ring=int(per_ring),
                          input_manifest_sha256=manifest_sha, localized_sidecar_sha256=sidecar_sha,
                          input_root=str(input_root.resolve()), source_hashes=sources, **git_state(),
                          python=sys.version, platform=platform.platform(),
                          created=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    say(f"wrote {path} (identity {identity[:12]})")
    return identity, sidecar


# ---------------------------------------------------------------------------
# tasks: one (candidate, grid) = one run.py subprocess
# ---------------------------------------------------------------------------
def task_paths(folder, cand, n) -> dict:
    d = Path(folder) / cand
    return dict(dir=d, results=d / f"results_N{n}.json", receipt=d / f"receipt_N{n}.json", log=d / f"run_N{n}.log")


def plan(candidates, grids) -> list:
    """Every grid x candidate, finest grid first (the long jobs start first)."""
    return [(cand, support, n) for n in sorted(grids, reverse=True) for cand, support in candidates.items()]


def check_task(folder, cand, support, n, *, identity, per_ring, strict=True) -> list:
    """Problems with one task's outputs (empty list = complete and consistent).  ``strict`` adds the failure and
    dispatch-violation checks and the log file."""
    p = task_paths(folder, cand, n)
    tag = f"{cand} N{n}"
    problems = []
    for what in ("results", "receipt") + (("log",) if strict else ()):
        if not p[what].is_file():
            problems.append(f"{tag}: missing {p[what].name}")
    if problems:
        return problems
    try:
        results, receipt = read_json(p["results"]), read_json(p["receipt"])
    except ValueError as exc:
        return [f"{tag}: unreadable output ({exc})"]
    for what, obj in (("receipt", receipt), ("results", results)):
        if obj.get("identity") != identity:
            problems.append(f"{tag}: {what} identity differs from the campaign manifest")
        if (obj.get("n"), obj.get("inner_support"), obj.get("per_ring")) != (n, support, per_ring):
            problems.append(f"{tag}: {what} was written for n={obj.get('n')}, inner_support={obj.get('inner_support')}, "
                            f"per_ring={obj.get('per_ring')}")
    if results.get("failures") or receipt.get("failures"):
        problems.append(f"{tag}: owner-build failures {sorted(results.get('failures') or receipt.get('failures'))[:5]}")
    if strict:
        dc = results.get("dispatch_check") or {}
        if dc.get("n_violations") != 0:
            problems.append(f"{tag}: dispatch check has {dc.get('n_violations')} violations")
        if dc.get("same_keys") is not True:
            problems.append(f"{tag}: candidate row keys differ from C0's")
    return problems


_CHILDREN: set = set()
_CHILD_LOCK = threading.Lock()


def subprocess_env(cfg) -> dict:
    env = dict(os.environ)
    env.update(cfg["subprocess_env"])
    return env


def run_command(n, support, cand, *, per_ring, input_root, sidecar, identity, out) -> list:
    return [sys.executable, "-m", RUN_MODULE, str(n), "--per-ring", str(per_ring), "--c1", support, "--candidate", cand,
            "--input-root", str(input_root), "--sidecar", str(sidecar), "--identity", identity, "--out", str(out)]


def run_task(folder, cand, support, n, *, per_ring, input_root, sidecar, identity, timeout=None) -> dict:
    p = task_paths(folder, cand, n)
    p["dir"].mkdir(parents=True, exist_ok=True)
    for what in ("results", "receipt"):                      # a stale output must never survive a failed rerun
        p[what].unlink(missing_ok=True)
    cmd = run_command(n, support, cand, per_ring=per_ring, input_root=input_root, sidecar=sidecar, identity=identity,
                      out=p["dir"])
    t0 = time.time()
    with p["log"].open("w") as lf:
        lf.write("# " + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=SCRIPTS, env=subprocess_env(config()), stdout=lf, stderr=subprocess.STDOUT)
        with _CHILD_LOCK:
            _CHILDREN.add(proc)
        try:
            rc = proc.wait(timeout=timeout)
            status = "ok" if rc == 0 else "failed"
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc, status = None, "timeout"
        finally:
            with _CHILD_LOCK:
                _CHILDREN.discard(proc)
    return dict(cand=cand, n=n, status=status, returncode=rc, seconds=time.time() - t0)


def _kill_children():
    with _CHILD_LOCK:
        procs = list(_CHILDREN)
    for proc in procs:
        with contextlib.suppress(Exception):
            proc.terminate()


def run_tasks(folder, tasks, *, jobs, per_ring, input_root, sidecar, identity, timeout=None, runner=None) -> list:
    """Run ``tasks`` (``(cand, support, n)``) ``jobs`` at a time; skip the complete ones.  ``runner`` (tests) replaces
    :func:`run_task`.  Returns one result dict per task (``status``: ``skipped``/``ok``/``failed``/``timeout``)."""
    runner = runner or run_task
    results, todo = [], []
    for cand, support, n in tasks:
        problems = check_task(folder, cand, support, n, identity=identity, per_ring=per_ring, strict=False)
        if problems:
            todo.append((cand, support, n))
            if any("identity" in q or "written for" in q for q in problems):
                log(f"{cand} N{n}: existing output does not match this campaign; rerunning")
        else:
            results.append(dict(cand=cand, n=n, status="skipped", returncode=0, seconds=0.0))
            log(f"{cand} N{n}: complete (resume), skipped")

    def one(item):
        cand, support, n = item
        log(f"{cand} N{n}: start ({support})")
        r = runner(folder, cand, support, n, per_ring=per_ring, input_root=input_root, sidecar=sidecar,
                   identity=identity, timeout=timeout)
        log(f"{cand} N{n}: {r['status']} in {r['seconds']:.0f} s")
        return r

    try:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            results.extend(pool.map(one, todo))
    except BaseException:
        _kill_children()
        raise
    return results


def memory_note(jobs, cfg) -> None:
    try:
        phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        return
    need = jobs * cfg["peak_rss_gib_n64"]
    if need > 0.9 * phys:
        log(f"WARNING: {jobs} jobs x ~{cfg['peak_rss_gib_n64']:.0f} GiB (N64 peak) = {need:.0f} GiB vs {phys:.0f} GiB physical")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_verify_inputs(args) -> int:
    cfg = config()
    candidates = parse_candidates(args.candidates, cfg)
    with lock(args.output):
        ensure_inputs(args.input_root, args.output, force=True)
        identity, sidecar = prepare_campaign(input_root=args.input_root, folder=args.output, candidates=candidates,
                                             per_ring=args.per_ring)
    log(f"inputs verified; identity {identity[:16]}; sidecar {sidecar}")
    return 0


def cmd_run(args) -> int:
    cfg = config()
    candidates = parse_candidates(args.candidates, cfg)
    grids = parse_grids(args.grids, cfg)
    with lock(args.output):
        ensure_inputs(args.input_root, args.output)
        identity, sidecar = prepare_campaign(input_root=args.input_root, folder=args.output, candidates=candidates,
                                             per_ring=args.per_ring)
        memory_note(args.jobs, cfg)
        log(f"{len(candidates)} candidates x {len(grids)} grids, {args.jobs} at a time")
        res = run_tasks(args.output, plan(candidates, grids), jobs=args.jobs, per_ring=args.per_ring,
                        input_root=args.input_root, sidecar=sidecar, identity=identity)
    bad = [r for r in res if r["status"] not in ("ok", "skipped")]
    log(f"done: {sum(r['status'] == 'ok' for r in res)} run, {sum(r['status'] == 'skipped' for r in res)} skipped, "
        f"{len(bad)} failed" + ("".join(f"\n  {r['cand']} N{r['n']}: {r['status']} (see {task_paths(args.output, r['cand'], r['n'])['log']})"
                                        for r in bad)))
    return 1 if bad else 0


def cmd_preflight(args) -> int:
    cfg = config()
    pre = cfg["preflight"]
    candidates = parse_candidates(args.candidates or next(iter(cfg["candidates"])), cfg)
    if len(candidates) != 1:
        raise ValueError("preflight takes exactly one candidate")
    (cand, support), = candidates.items()
    folder = Path(args.output) / "preflight"
    with lock(folder):
        ensure_inputs(args.input_root, args.output)
        identity, sidecar = prepare_campaign(input_root=args.input_root, folder=folder, candidates=candidates,
                                             per_ring=pre["per_ring"], kind="preflight")
        t0 = time.time()
        res = run_tasks(folder, plan(candidates, (pre["grid"],)), jobs=1, per_ring=pre["per_ring"],
                        input_root=args.input_root, sidecar=sidecar, identity=identity,
                        timeout=pre["timeout_seconds"])
        wall = time.time() - t0
    (r,) = res
    if r["status"] in ("ok", "skipped"):
        problems = check_task(folder, cand, support, pre["grid"], identity=identity, per_ring=pre["per_ring"])
    else:
        problems = [f"{cand} N{pre['grid']}: {r['status']} (see {task_paths(folder, cand, pre['grid'])['log']})"]
    p = task_paths(folder, cand, pre["grid"])
    if p["receipt"].is_file() and p["results"].is_file():
        rc, rs = read_json(p["receipt"]), read_json(p["results"])
        dc = rs["dispatch_check"]
        log(f"preflight {cand} ({support}) N{pre['grid']} per-ring {pre['per_ring']}: status {r['status']}, "
            f"wall {wall:.0f} s (run.py {rc['wall_seconds']:.0f} s), peak RSS {rc['peak_rss_gib']:.2f} GiB, "
            f"{rc['n_owners']} owners, dispatch violations {dc['n_violations']} "
            f"(identical {dc['identical']}, changed {dc['changed']}), failures {len(rs['failures'])}")
    for q in problems:
        log(f"PREFLIGHT PROBLEM: {q}")
    log("preflight " + ("FAILED" if problems else "passed"))
    return 1 if problems else 0


def load_manifest(output) -> dict:
    path = Path(output) / "campaign_manifest.json"
    if not path.is_file():
        raise ValueError(f"no campaign_manifest.json in {output}; run verify-inputs first")
    return read_json(path)


def cmd_validate(args) -> int:
    cfg = config()
    man = load_manifest(args.output)
    grids = parse_grids(args.grids, cfg)
    problems, rows = [], []
    for cand, support, n in plan(man["candidates"], grids):
        problems += check_task(args.output, cand, support, n, identity=man["identity"], per_ring=man["per_ring"])
        p = task_paths(args.output, cand, n)
        if p["results"].is_file() and p["receipt"].is_file():
            rc, rs = read_json(p["receipt"]), read_json(p["results"])
            rows.append(dict(candidate=cand, n=n, n_owners=rc.get("n_owners"), wall_seconds=rc.get("wall_seconds"),
                             peak_rss_gib=rc.get("peak_rss_gib"),
                             dispatch_violations=rs.get("dispatch_check", {}).get("n_violations"),
                             failures=len(rs.get("failures") or {})))
    report = dict(passed=not problems, identity=man["identity"], grids=list(grids), candidates=man["candidates"],
                  problems=problems, tasks=rows)
    write_json(Path(args.output) / "validation.json", report)
    for q in problems:
        log(f"VALIDATION PROBLEM: {q}")
    log("validation " + ("passed" if not problems else f"FAILED ({len(problems)} problems)")
        + f"; {len(rows)} tasks, wrote {Path(args.output) / 'validation.json'}")
    return 0 if not problems else 1


def cmd_analyze(args) -> int:
    from p08_inner_support_eval import analyze as analyze_mod
    from p08_inner_support_eval import compare_candidates as compare_mod
    cfg = config()
    man = load_manifest(args.output)
    grids = parse_grids(args.grids, cfg)
    problems = []
    for cand, support, n in plan(man["candidates"], grids):
        problems += check_task(args.output, cand, support, n, identity=man["identity"], per_ring=man["per_ring"],
                               strict=False)
    if problems:
        raise ValueError("cannot analyze an incomplete campaign:\n  " + "\n  ".join(problems))
    out = Path(args.output) / "summary"
    out.mkdir(exist_ok=True)
    summaries = {}
    for cand in man["candidates"]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            analyze_mod.main(Path(args.output) / cand, summary_path=out / f"{cand}_summary.json")
        (out / f"{cand}_analyze.txt").write_text(buf.getvalue())
        summaries[cand] = out / f"{cand}_summary.json"
        log(f"{cand}: analyzed ({', '.join(str(n) for n in grids)})")
    text, data = compare_mod.compare(summaries, bands=cfg["bands"])
    write_json(out / "comparison.json", data)
    (out / "comparison.txt").write_text(text)
    print(text, end="")
    log(f"wrote {out}/ (per-candidate summary JSON and text, comparison.json, comparison.txt)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    cfg = config()
    default_cands = ",".join(f"{k}={v}" for k, v in cfg["candidates"].items())
    ap = argparse.ArgumentParser(prog="python -m p08_inner_support_eval.campaign", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, *, inputs=False):
        p.add_argument("--output", required=True, help="the campaign folder")
        if inputs:
            p.add_argument("--input-root", required=True,
                           help="immutable HSX input root (holds the files of input_manifest.json)")

    def cfgopts(p):
        p.add_argument("--candidates", default=default_cands,
                       help="NAME=inner_support,... (default %(default)s); a bare configured name is accepted")
        p.add_argument("--per-ring", type=positive_int, default=cfg["per_ring"])

    p = sub.add_parser("verify-inputs", help="hash-check the inputs, localize the sidecar, write the manifest")
    common(p, inputs=True)
    cfgopts(p)
    p.set_defaults(fn=cmd_verify_inputs)

    p = sub.add_parser("preflight", help="N32, --per-ring 1, one candidate, into <output>/preflight")
    common(p, inputs=True)
    p.add_argument("--candidates", default=None, help="exactly one candidate (name or NAME=inner_support); default: the first configured")
    p.set_defaults(fn=cmd_preflight)

    p = sub.add_parser("run", help="every grid x candidate as an independent subprocess, --jobs at a time")
    common(p, inputs=True)
    cfgopts(p)
    p.add_argument("--grids", default=",".join(str(n) for n in cfg["grids"]))
    p.add_argument("--jobs", type=positive_int, default=1, help="concurrent run.py processes (each ~6 GiB at N64)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("validate", help="expected files exist, no failures, zero dispatch violations")
    common(p)
    p.add_argument("--grids", default=",".join(str(n) for n in cfg["grids"]))
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("analyze", help="analyze.py per candidate and the candidate comparison, into <output>/summary/")
    common(p)
    p.add_argument("--grids", default=",".join(str(n) for n in cfg["grids"]))
    p.set_defaults(fn=cmd_analyze)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
