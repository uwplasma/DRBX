#!/usr/bin/env python3
"""P08 step 2b, G3: the full-grid JAX replay campaign.

Builds each grid's schema-v3 (tensor-factored) row artifact with the step-1 machinery (unit-parallel,
``p_shared.build_artifact.run_full_build`` through ``p08_step1_global.campaign._build_artifact``), then
replays the six frozen perpendicular campaigns (seven campaign keys) with the JAX operators of
``drbx.native`` in **one process on the full grid** and compares the operator terms with the frozen
oracles -- see ``design.md`` (``work/p08_step2b_operator_assembly_design_20260929``, section 3 "G3") and
this package's ``README.md``.

Commands: ``verify-inputs``, ``preflight`` (the JAX owner-closure gate of design section 8, per grid),
``run`` (artifact build -> JAX replay stage -> comparison, per grid), ``validate``, and
``run-stage --stage {artifact,replay} --n N`` for recovery.

Run as ``python -m p08_step2_global.campaign`` from ``DRBX/scripts`` (never put this package's directory
first on ``sys.path``; ``DRBX/scripts`` must be on the path for every ``p_shared`` import).
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent            # .../DRBX/scripts
REPO = SCRIPTS.parent            # .../DRBX
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p_shared import build_artifact as build_artifact_mod                       # noqa: E402
from p_shared import oracle_manifest as om                                        # noqa: E402
from p_shared import runner                                                       # noqa: E402
from p_shared.replay_support import CAMPAIGN_FUNCS                                # noqa: E402
from p08_step1_global import campaign as step1                                    # noqa: E402

GRIDS = (32, 48, 64)
STAGES = ("artifact", "replay")
SCHEMA = "drbx.p08-step2b-g3-campaign-v1"

# Reused unchanged from the step-1 campaign (same 17-file input manifest, same oracle manifest and paths).
sha = step1.sha
digest = step1.digest
write = step1.write
lock = step1.lock
localize_sidecar = step1.localize_sidecar
oracle_paths = step1.oracle_paths
_effective_workers = step1._effective_workers
_sidecar_path = step1._sidecar_path


def config() -> dict:
    """This campaign's configuration merged over step 1's (chunk sizes, tier-B / pointwise-cap policy, oracle
    paths and the canonical sidecar are inherited; ``resolutions`` and ``schema`` are overridden)."""
    own = json.loads((HERE / "configuration.json").read_text())
    if own["schema"] != SCHEMA:
        raise ValueError("unsupported P08 step-2b G3 campaign contract")
    if own["campaigns"] != list(CAMPAIGN_FUNCS):
        raise ValueError("campaign catalogue changed relative to p_shared.replay_support.CAMPAIGN_FUNCS")
    if not set(own["resolutions"]) <= set(own["allowed_resolutions"]) <= set(GRIDS):
        raise ValueError("bad resolutions in configuration.json")
    return {**step1.config(), **own}


def _input_manifest() -> dict:
    """The step-1 (= P05N/P06N/P07N) immutable 17-file input manifest (tests monkeypatch this)."""
    return step1._input_manifest()


def committed_oracle_manifest() -> dict:
    """The committed step-1 oracle manifest -- the same files, already extracted on the remote."""
    return step1.committed_oracle_manifest()


# ---------------------------------------------------------------------------
# Identity: this package, the step-1 machinery it reuses, the JAX operator stack and every source of the
# artifact build (which the step-1 list already pins), the committed oracle manifest, the configuration.
# ---------------------------------------------------------------------------
_EXTRA_SOURCES = [
    "scripts/p08_step2_global/campaign.py",
    "scripts/p08_step2_global/replay.py",
    "scripts/p08_step2_global/comparison.py",
    "scripts/p08_step2_global/configuration.json",
    "scripts/p_shared/jax_replay.py",
    "scripts/p_shared/campaign_fields.py",
    "scripts/p_shared/owner_closure.py",
    "scripts/p_shared/selection.py",
    "src/drbx/stencils/loader.py",
    "src/drbx/stencils/operator_plan.py",
    "src/drbx/stencils/tensor_rows.py",
    "src/drbx/native/fci_perpendicular_source_rows.py",
    "src/drbx/native/fci_perpendicular_tensor_rows.py",
    "src/drbx/native/fci_perpendicular_point_rows.py",
    "src/drbx/native/fci_perpendicular_neumann_rows.py",
    "src/drbx/native/fci_perpendicular_reconstruction_state.py",
    "src/drbx/native/fci_perpendicular_midpoint_bracket.py",
    "src/drbx/native/fci_perpendicular_p05_operator.py",
    "src/drbx/native/fci_perpendicular_p06_operator.py",
    "src/drbx/native/fci_perpendicular_p07_operator.py",
]
SOURCE_FILES = list(dict.fromkeys([*step1.SOURCE_FILES, *_EXTRA_SOURCES]))


def source_hashes() -> dict:
    return {p: sha(REPO / p) for p in SOURCE_FILES}


def verify(*, input_root: Path, output: Path, oracle_root: Path | None) -> dict:
    """Verify the immutable inputs, this campaign's sources and the oracle files (the committed step-1
    manifest, hash-checked under ``ORACLE_ROOT``); localize the sidecar; record/check the identity. Refuses
    (raises) if the identity, the localized sidecar or ``ORACLE_ROOT`` changed relative to a previous run at the
    same ``output``. Mirrors ``p08_step1_global.campaign.verify`` with this package's identity."""
    input_root = Path(input_root); output = Path(output)
    cfg = config()
    im = _input_manifest()
    for rec in im["files"]:
        path = input_root / rec["path"]
        if not path.is_file() or path.stat().st_size != rec["bytes"] or sha(path) != rec["sha256"]:
            raise ValueError(f"missing or changed immutable input: {path}")
    sources = source_hashes()
    manifest = committed_oracle_manifest()
    resolved_oracle_root = Path(oracle_root) if oracle_root is not None else input_root
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    identity = digest({"configuration": cfg, "input_manifest": im, "sources": sources, "commit": commit,
                       "oracle_manifest": manifest})
    manifest_path = output / "campaign_manifest.json"
    local = localize_sidecar(input_root, output)
    oracle_errors = om.verify_manifest(manifest, resolved_oracle_root)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["identity"] != identity:
            raise ValueError("campaign identity changed; use a new output folder")
        if saved["localized_sidecar_sha256"] != sha(local):
            raise ValueError("localized sidecar hash changed")
        if saved.get("oracle_root") != str(resolved_oracle_root.resolve()):
            raise ValueError("oracle root changed; use a new output folder")
    if oracle_errors:
        raise ValueError("oracle verification failed:\n" + "\n".join(oracle_errors[:50]) +
                         (f"\n(+{len(oracle_errors) - 50} more)" if len(oracle_errors) > 50 else ""))
    if not manifest_path.exists():
        import jax
        if jax.default_backend() != "cpu":
            raise ValueError("JAX CPU backend required")
        if not jax.config.jax_enable_x64:
            raise ValueError("JAX x64 required (JAX_ENABLE_X64=true)")
        write(manifest_path, {"identity": identity, "campaign": "p08_step2_global", "configuration": cfg,
                              "input_manifest": im, "source_hashes": sources, "commit": commit,
                              "input_root": str(input_root.resolve()),
                              "oracle_root": str(resolved_oracle_root.resolve()),
                              "oracle_manifest": manifest, "oracle_verified": True,
                              "localized_sidecar_sha256": sha(local),
                              "python": sys.version, "platform": platform.platform(),
                              "jax_backend": jax.default_backend(), "jax_x64": bool(jax.config.jax_enable_x64)})
    write(output / "oracle_manifest.json", manifest)
    for name in ("logs", "invocations", "executions"):
        (output / name).mkdir(exist_ok=True)
    return identity


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _refuse_csr_only() -> None:
    if not build_artifact_mod.tensor_encoding_enabled():
        raise ValueError(f"{build_artifact_mod.CSR_ONLY_ENV} is set: the G3 replay needs the tensor-encoded (v3) "
                         "artifact; unset it")


def build_artifact_stage(*, n: int, args, sidecar_path: Path, artifact_root: Path) -> dict:
    """The artifact stage: step 1's ``_build_artifact`` (unit-parallel, resumable, tensor encoding on)."""
    _refuse_csr_only()
    effective = _effective_workers(args.workers, args.memory_budget_gib, args.worker_memory_gib,
                                   args.memory_reserve_gib)
    return step1._build_artifact(n=n, input_root=args.input_root, sidecar_path=sidecar_path, output=artifact_root,
                                 workers=effective, memory_budget_gib=args.memory_budget_gib,
                                 worker_memory_gib=args.worker_memory_gib,
                                 memory_reserve_gib=args.memory_reserve_gib,
                                 max_tasks_per_worker=args.max_tasks_per_worker,
                                 max_units=getattr(args, "max_units", None))


def replay_stage(*, n: int, args, identity: str, sidecar_path: Path, paths: dict, artifact_root: Path) -> dict:
    from p08_step2_global import replay
    cfg = config()
    return replay.run_replay_stage(
        artifact_root=artifact_root, output=args.output / "replay" / f"N{n}", n=n, input_root=args.input_root,
        sidecar_path=sidecar_path, paths=paths, campaigns=tuple(cfg["campaigns"]), campaign_identity=identity,
        cfg=cfg)


def run_grid(*, n: int, args, identity: str, sidecar_path: Path, paths: dict) -> dict:
    """One grid of ``run``: artifact build (skipped when assembled) then the JAX replay stage (skipped when
    complete). ``--max-units`` (local testing only) stops after the bounded build."""
    artifact_root = args.output / "artifact"
    build_receipt = build_artifact_stage(n=n, args=args, sidecar_path=sidecar_path, artifact_root=artifact_root)
    if not build_receipt.get("geometry_complete", True):
        return {"n": n, "status": "geometry_incomplete", "note": "bounded (--max-units) geometry stage; resume"}
    if args.max_units is not None:
        return {"n": n, "status": "bounded_build_only",
                "note": "--max-units builds a partial artifact; the JAX replay needs the full one"}
    result = replay_stage(n=n, args=args, identity=identity, sidecar_path=sidecar_path, paths=paths,
                          artifact_root=artifact_root)
    return {"n": n, "status": "complete", "all_terms_pass": result.get("all_terms_pass"),
            "wall_seconds": result.get("wall_seconds"), "skipped": bool(result.get("skipped", False))}


# ---------------------------------------------------------------------------
# Preflight (per grid: the JAX owner closure of design section 8)
# ---------------------------------------------------------------------------
def preflight_grid(*, n: int, input_root: Path, sidecar_path: Path, output: Path, paths: dict) -> dict:
    """The bounded gate of grid ``n``: 12 owners, host + JAX assembly through the blocked full-grid path, the
    uniform policy of design section 8 and ``compare_to_oracle`` with the JAX terms. Also records (and
    requires) the CPU backend and x64."""
    from p08_step2_global import replay
    from p_shared import jax_replay as jr

    cfg = config()
    info = replay.jax_info()
    if info["backend"] != "cpu":
        raise ValueError(f"JAX CPU backend required, got {info['backend']!r}")
    if not info["x64"]:
        raise ValueError("JAX x64 required (JAX_ENABLE_X64=true)")
    started = time.time()
    report = output / "preflight" / f"N{n}_owner_closure.json"
    payload = jr.run_jax_owner_closure_check(
        n=n, input_root=input_root, sidecar_path=sidecar_path, paths=paths, campaigns=tuple(cfg["campaigns"]),
        floor_seeds=tuple(cfg["preflight_floor_seeds"]), wall_cache=bool(cfg["wall_cache"]), output=report,
        column_block=cfg["column_block"], variant_block=cfg["p06n_variant_block"],
        boundary_batch=cfg["boundary_batch"], curvature="fd")
    all_pass = bool(payload["all_diff_pass"] and payload["oracle_jax_all_pass"] and not payload["uniq_mismatches"])
    return {"n": int(n), "all_pass": all_pass, "jax": info,
            "policy_rows": len(payload["diff_table"]), "policy_failures": payload["diff_failures"],
            "oracle_rows": len(payload["oracle_jax"]),
            "oracle_failures": [(r["campaign"], r["term"]) for r in payload["oracle_jax"] if not r["pass"]],
            "uniq_mismatches": payload["uniq_mismatches"], "plan": payload["plan"], "blocking": payload["blocking"],
            "peak_rss_gib": payload["peak_rss_gib"], "closure_seconds": payload["wall_seconds"],
            "closure_report": str(report), "seconds": time.time() - started}


def _load_preflight(output: Path, identity: str) -> dict | None:
    path = output / "preflight.json"
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    return saved if saved.get("identity") == identity else None


def preflight(*, args, identity) -> dict:
    output = args.output
    previous = _load_preflight(output, identity)
    cases = dict(previous["cases"]) if previous else {}
    paths = oracle_paths(args.oracle_root, args.input_root)
    for n in args.resolutions:
        done = cases.get(str(n))
        if done is not None and done.get("all_pass"):
            continue                                    # resumable: a grid that passed is not redone
        cases[str(n)] = preflight_grid(n=n, input_root=args.input_root, sidecar_path=_sidecar_path(args),
                                       output=output, paths=paths)
        write(output / "preflight.json", {"identity": identity, "cases": cases,
                                          "all_pass": all(c.get("all_pass") for c in cases.values())})
    payload = {"identity": identity, "cases": cases, "all_pass": all(c.get("all_pass") for c in cases.values())}
    write(output / "preflight.json", payload)
    return payload


def _require_preflight(output: Path, identity: str, resolutions, *, what: str) -> dict:
    payload = _load_preflight(output, identity)
    if payload is None:
        raise ValueError(f"{what} requires a matching preflight first" if what == "run"
                         else f"a matching preflight is required before {what}")
    missing = [n for n in resolutions if str(n) not in payload["cases"]]
    if missing:
        raise ValueError(f"the preflight has no case for N{missing}; run preflight for those resolutions")
    if not all(payload["cases"][str(n)].get("all_pass") for n in resolutions):
        raise ValueError(f"preflight did not pass; refusing to {what}")
    return payload


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(*, args, identity) -> dict:
    output = args.output
    _require_preflight(output, identity, args.resolutions, what="validate")
    cfg = config()
    replay_summaries = {}
    for n in args.resolutions:
        path = output / "replay" / f"N{n}" / "replay.json"
        if not path.is_file():
            raise ValueError(f"no replay.json for N{n}; run the replay stage first")
        summary = json.loads(path.read_text())
        if summary.get("campaign_identity") != identity:
            raise ValueError(f"replay.json of N{n} belongs to another campaign identity")
        missing = [c for c in cfg["campaigns"] if c not in summary["campaigns"]]
        if missing:
            raise ValueError(f"replay.json of N{n} lacks campaigns {missing}")
        replay_summaries[str(n)] = summary
    campaign_pass = {n: {name: bool(all(t.get("pass", False) for t in c["terms"].values()))
                         for name, c in r["campaigns"].items()} for n, r in replay_summaries.items()}
    payload = {"identity": identity, "operational_complete": True, "resolutions": list(args.resolutions),
               "scope": {"operator_terms_only": True, "omitted_host_only_terms": cfg["omitted_host_only_terms"]},
               "campaign_status": {n: {name: c.get("status") for name, c in r["campaigns"].items()}
                                   for n, r in replay_summaries.items()},
               "campaign_pass": campaign_pass,
               "all_terms_pass": bool(all(all(v.values()) for v in campaign_pass.values()))}
    write(output / "validation.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("verify-inputs", "preflight", "run", "validate", "run-stage"))
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolutions", type=int, nargs="+", choices=GRIDS, default=[32, 48])
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--n", type=int, choices=GRIDS)
    p.add_argument("--workers", type=int, default=4, help="artifact build process-pool size (the JAX replay is "
                                                          "one process)")
    p.add_argument("--memory-budget-gib", type=float, default=None)
    p.add_argument("--worker-memory-gib", type=float, default=None)
    p.add_argument("--memory-reserve-gib", type=float, default=1.0)
    p.add_argument("--max-tasks-per-worker", type=int, default=None)
    p.add_argument("--max-units", type=int, default=None,
                   help="Local testing only: cap the artifact build's units per stage (see the step-1 "
                        "campaign). A bounded `run` stops after the build; the JAX replay never runs on a "
                        "partial artifact. Never used for a real campaign run.")
    p.add_argument("--oracle-root", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    args.output = args.output.resolve()
    args.input_root = args.input_root.resolve()
    if args.resolutions != sorted(set(args.resolutions)):
        raise ValueError("resolutions must be unique and ascending")
    with lock(args.output):
        for name in ("logs", "invocations"):
            (args.output / name).mkdir(exist_ok=True, parents=True)
        identity = verify(input_root=args.input_root, output=args.output, oracle_root=args.oracle_root)
        write(args.output / "invocations" / f"{time.time_ns()}_{args.command}.json",
              {"argv": list(sys.argv if argv is None else argv), "identity": identity, "time": time.time()})
        summary: dict = {"command": args.command}
        if args.command == "verify-inputs":
            summary.update(status="complete", identity=identity)
        else:
            paths = oracle_paths(args.oracle_root, args.input_root)
            sidecar_path = _sidecar_path(args)
            if args.command == "preflight":
                payload = preflight(args=args, identity=identity)
                summary.update(all_pass=payload["all_pass"])
            elif args.command == "run-stage":
                if args.stage is None or args.n is None:
                    raise ValueError("--stage and --n are required for run-stage")
                artifact_root = args.output / "artifact"
                if args.stage == "artifact":
                    receipt = build_artifact_stage(n=args.n, args=args, sidecar_path=sidecar_path,
                                                   artifact_root=artifact_root)
                    summary.update(stage="artifact", n=args.n, wall_seconds=receipt.get("wall_seconds"))
                else:
                    result = replay_stage(n=args.n, args=args, identity=identity, sidecar_path=sidecar_path,
                                          paths=paths, artifact_root=artifact_root)
                    summary.update(stage="replay", n=args.n, wall_seconds=result.get("wall_seconds"),
                                   all_terms_pass=result.get("all_terms_pass"))
            elif args.command == "run":
                _require_preflight(args.output, identity, args.resolutions, what="run")
                grids = [run_grid(n=n, args=args, identity=identity, sidecar_path=sidecar_path, paths=paths)
                         for n in args.resolutions]
                if all(g["status"] == "complete" for g in grids):
                    payload = validate(args=args, identity=identity)
                    summary.update(operational_complete=payload["operational_complete"],
                                   all_terms_pass=payload["all_terms_pass"])
                summary.update(status="complete", grids=grids)
            else:                                                       # validate
                payload = validate(args=args, identity=identity)
                summary.update(status="complete", operational_complete=payload["operational_complete"],
                               all_terms_pass=payload["all_terms_pass"])
        write(args.output / "last_exit.json", {"command": args.command, "status": "complete", "time": time.time(),
                                               "summary": summary})
        print(json.dumps(summary, default=str), flush=True)
        return summary


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            a = parse()
            write(a.output / "last_exit.json", {"command": a.command, "status": "failed", "error": repr(exc),
                                                "time": time.time()})
        finally:
            raise
