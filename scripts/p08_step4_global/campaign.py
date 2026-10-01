#!/usr/bin/env python3
"""P08 step 4: full-grid artifact build and smoke check of the final bundle operator (N32 / N48 / N64).

The final operator options -- ``curvature="autodiff"``, ``face_quadrature="q2"``, ``inner_support="fixed_radius"`` --
are pinned in ``configuration.json`` (and checked against the literals :data:`OPERATOR_OPTIONS` below) and passed
explicitly to the artifact build, the environment and the closure preflight; no default is relied on. Per grid:

1. **artifact** -- ``p_shared.build_artifact.run_full_build`` (unit-parallel, resumable, schema v3 with tensor
   encoding) with step 1's chunk sizes and the pinned options; the build identity must record them;
2. **replay** (the smoke stage, :mod:`p08_step4_global.smoke`) -- the step-2 JAX replay of the seven campaign keys
   on the full grid with the pinned options; the gate is that every term array is finite. The comparison with the
   frozen step-1 oracles is recorded as *informational* change data only: those oracles describe the old
   fd / q3 / profile7 operator, so there is no MMS acceptance gate.

``preflight`` is the gate before ``run``: the bounded 12-owner JAX closure check with the final options (host-vs-JAX
policy rows all pass, no ``uniq_mismatches``); its ``compare_to_oracle`` rows are recorded as informational.

Commands: ``verify-inputs``, ``preflight``, ``run``, ``validate``, ``run-stage --stage {artifact,replay} --n N``.
Run as ``python -m p08_step4_global.campaign`` from ``DRBX/scripts`` (never put this package's directory first on
``sys.path``; ``DRBX/scripts`` must be on the path for every ``p_shared`` import). The artifacts stay on the remote
for step 5; their absolute paths are in ``summary/step4_summary.json``.
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
from p_shared.replay_support import CAMPAIGN_FUNCS                                # noqa: E402
from p08_step1_global import campaign as step1                                    # noqa: E402
from p08_step2_global import campaign as step2                                    # noqa: E402

GRIDS = (32, 48, 64)
STAGES = ("artifact", "replay")
SCHEMA = "drbx.p08-step4-smoke-campaign-v1"
#: the final operator bundle, pinned: ``configuration.json`` must say exactly this (and is hashed into the identity)
OPERATOR_OPTIONS = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius"}

# Reused unchanged from the step-1 / step-2 campaigns (same 17-file input manifest, same oracle manifest and paths).
sha = step1.sha
digest = step1.digest
write = step1.write
lock = step1.lock
localize_sidecar = step1.localize_sidecar
oracle_paths = step1.oracle_paths
_effective_workers = step1._effective_workers
_sidecar_path = step1._sidecar_path
_refuse_csr_only = step2._refuse_csr_only
_load_preflight = step2._load_preflight
_require_preflight = step2._require_preflight


def config() -> dict:
    """This campaign's configuration merged over step 2's (which is merged over step 1's): blocking / boundary
    settings, chunk sizes, oracle paths and the canonical sidecar are inherited; ``schema``, ``resolutions`` and
    ``operator_options`` are this package's."""
    own = json.loads((HERE / "configuration.json").read_text())
    if own["schema"] != SCHEMA:
        raise ValueError("unsupported P08 step-4 smoke campaign contract")
    if own["operator_options"] != OPERATOR_OPTIONS:
        raise ValueError(f"configuration.json operator_options {own['operator_options']} differ from the pinned "
                         f"final bundle {OPERATOR_OPTIONS}")
    merged = {**step2.config(), **own}
    if merged["campaigns"] != list(CAMPAIGN_FUNCS):
        raise ValueError("campaign catalogue changed relative to p_shared.replay_support.CAMPAIGN_FUNCS")
    if not set(merged["resolutions"]) <= set(merged["allowed_resolutions"]) <= set(GRIDS):
        raise ValueError("bad resolutions in configuration.json")
    return merged


def operator_options(cfg: dict | None = None) -> dict:
    """The pinned options, read from the configuration (so that a changed configuration changes every call)."""
    return dict((cfg or config())["operator_options"])


def _input_manifest() -> dict:
    """The step-1 (= P05N/P06N/P07N) immutable 17-file input manifest (tests monkeypatch this)."""
    return step1._input_manifest()


def committed_oracle_manifest() -> dict:
    """The committed step-1 oracle manifest -- the same files, already extracted on the remote."""
    return step1.committed_oracle_manifest()


# ---------------------------------------------------------------------------
# Identity: this package, the step-2 JAX stack and the step-1 machinery (which pin every source of the artifact
# build), the option modules the final operator uses, the committed oracle manifest, the merged configuration
# (including ``operator_options``).
# ---------------------------------------------------------------------------
_EXTRA_SOURCES = [
    "scripts/p08_step4_global/campaign.py",
    "scripts/p08_step4_global/smoke.py",
    "scripts/p08_step4_global/configuration.json",
    # the option modules of the final bundle (curvature / face quadrature / inner donor support)
    "scripts/p_shared/curvature_reference.py",
    "scripts/p_shared/face_quadrature.py",
    "scripts/p_shared/inner_support.py",
    "src/drbx/geometry/curvature_autodiff.py",
    "src/drbx/geometry/fci_perpendicular_reconstruction.py",
    "src/drbx/geometry/fci_perpendicular_integrated_rows.py",
    "src/drbx/stencils/geometry_arrays.py",
    "src/drbx/stencils/builder.py",
]
SOURCE_FILES = list(dict.fromkeys([*step2.SOURCE_FILES, *_EXTRA_SOURCES]))


def source_hashes() -> dict:
    return {p: sha(REPO / p) for p in SOURCE_FILES}


def verify(*, input_root: Path, output: Path, oracle_root: Path | None) -> dict:
    """Verify the immutable inputs, this campaign's sources and the oracle files (the committed step-1 manifest,
    hash-checked under ``ORACLE_ROOT``); localize the sidecar; record/check the identity. Refuses (raises) if the
    identity, the localized sidecar or ``ORACLE_ROOT`` changed relative to a previous run at the same ``output``,
    or if JAX is not on the CPU backend with x64. Mirrors ``p08_step2_global.campaign.verify``."""
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
    import jax
    if jax.default_backend() != "cpu":
        raise ValueError("JAX CPU backend required")
    if not jax.config.jax_enable_x64:
        raise ValueError("JAX x64 required (JAX_ENABLE_X64=true)")
    if not manifest_path.exists():
        write(manifest_path, {"identity": identity, "campaign": "p08_step4_global", "configuration": cfg,
                              "operator_options": operator_options(cfg),
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
def build_artifact_stage(*, n: int, args, sidecar_path: Path, artifact_root: Path) -> dict:
    """The artifact stage: ``run_full_build`` with the pinned options and step 1's chunk sizes (unit-parallel,
    resumable, tensor encoding on). The build identity must record the pinned options."""
    from p08_step4_global import smoke
    _refuse_csr_only()
    cfg = config()
    options = operator_options(cfg)
    effective = _effective_workers(args.workers, args.memory_budget_gib, args.worker_memory_gib,
                                   args.memory_reserve_gib)
    receipt = build_artifact_mod.run_full_build(
        n=n, input_root=args.input_root, sidecar_path=sidecar_path, output=artifact_root, workers=effective,
        memory_budget_gib=args.memory_budget_gib, worker_memory_gib=args.worker_memory_gib,
        memory_reserve_gib=args.memory_reserve_gib,
        cell_chunk_size=cfg["cell_chunk_size"], face_chunk_size=cfg["face_chunk_size"],
        p07_chunk_size=cfg["p07_chunk_size"], geometry_raw_chunk_size=cfg["geometry_raw_chunk_size"],
        geometry_face_chunk_size=cfg["geometry_face_chunk_size"], max_tasks_per_worker=args.max_tasks_per_worker,
        **options)
    smoke.check_artifact_options(receipt["identity"], options)
    return receipt


def replay_stage(*, n: int, args, identity: str, sidecar_path: Path, paths: dict, artifact_root: Path) -> dict:
    from p08_step4_global import smoke
    cfg = config()
    return smoke.run_smoke_stage(
        artifact_root=artifact_root, output=args.output / "replay" / f"N{n}", n=n, input_root=args.input_root,
        sidecar_path=sidecar_path, paths=paths, campaigns=tuple(cfg["campaigns"]), campaign_identity=identity,
        cfg=cfg, operator_options=operator_options(cfg))


def run_grid(*, n: int, args, identity: str, sidecar_path: Path, paths: dict) -> dict:
    """One grid of ``run``: artifact build (skipped when assembled) then the smoke stage (skipped when complete)."""
    artifact_root = args.output / "artifact"
    build_artifact_stage(n=n, args=args, sidecar_path=sidecar_path, artifact_root=artifact_root)
    result = replay_stage(n=n, args=args, identity=identity, sidecar_path=sidecar_path, paths=paths,
                          artifact_root=artifact_root)
    return {"n": n, "status": "complete", "smoke_pass": result.get("smoke_pass"),
            "wall_seconds": result.get("wall_seconds"), "skipped": bool(result.get("skipped", False))}


# ---------------------------------------------------------------------------
# Preflight (per grid: the bounded JAX owner closure with the final options)
# ---------------------------------------------------------------------------
def preflight_grid(*, n: int, input_root: Path, sidecar_path: Path, output: Path, paths: dict) -> dict:
    """The bounded gate of grid ``n``: 12 owners, host + JAX assembly through the blocked full-grid path with the
    pinned final options. Pass = every host-vs-JAX policy row passes, no ``uniq_mismatches``, and the payload
    records the pinned options. The ``compare_to_oracle`` rows (``oracle_jax``) are informational: the frozen
    oracles describe the old operator. Also requires the CPU backend and x64."""
    from p08_step2_global import replay
    from p_shared import jax_replay as jr

    cfg = config()
    options = operator_options(cfg)
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
        boundary_batch=cfg["boundary_batch"], curvature=options["curvature"],
        face_quadrature=options["face_quadrature"], inner_support=options["inner_support"])
    recorded = {"curvature": payload.get("curvature", "fd"), "face_quadrature": payload.get("face_quadrature", "q3"),
                "inner_support": payload.get("inner_support", "profile7")}
    options_recorded = recorded == options
    all_pass = bool(payload["all_diff_pass"] and not payload["uniq_mismatches"] and options_recorded)
    oracle_rows = payload["oracle_jax"]
    ratios = [r["ratio_to_oracle_NR"] for r in oracle_rows if r.get("ratio_to_oracle_NR") is not None]
    return {"n": int(n), "all_pass": all_pass, "operator_options": options, "options_recorded": options_recorded,
            "jax": info,
            "policy_rows": len(payload["diff_table"]), "policy_failures": payload["diff_failures"],
            "uniq_mismatches": payload["uniq_mismatches"],
            "oracle_informational": {"gate": False, "rows": len(oracle_rows),
                                     "rows_not_matching_frozen_oracles": [(r["campaign"], r["term"])
                                                                          for r in oracle_rows if not r["pass"]],
                                     "max_ratio_to_oracle_NR": max(ratios) if ratios else None,
                                     "note": "frozen oracles describe the old fd/q3/profile7 operator; "
                                             "differences are expected and do not gate"},
            "plan": payload["plan"], "blocking": payload["blocking"],
            "peak_rss_gib": payload["peak_rss_gib"], "closure_seconds": payload["wall_seconds"],
            "closure_report": str(report), "seconds": time.time() - started}


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


# ---------------------------------------------------------------------------
# Validation and summary
# ---------------------------------------------------------------------------
def validate(*, args, identity) -> dict:
    """Requires the preflight pass and ``replay/N{n}/smoke.json`` of this identity for every requested grid; writes
    ``validation.json`` and ``summary/step4_summary.json``. ``smoke_pass`` = preflight pass and every term finite."""
    output = args.output
    preflight_payload = _require_preflight(output, identity, args.resolutions, what="validate")
    cfg = config()
    options = operator_options(cfg)
    grids, summary_grids = {}, {}
    for n in args.resolutions:
        path = output / "replay" / f"N{n}" / "smoke.json"
        if not path.is_file():
            raise ValueError(f"no smoke.json for N{n}; run the replay stage first")
        smoke = json.loads(path.read_text())
        if smoke.get("campaign_identity") != identity:
            raise ValueError(f"smoke.json of N{n} belongs to another campaign identity")
        if smoke.get("operator_options") != options:
            raise ValueError(f"smoke.json of N{n} was produced with other operator options")
        missing = [c for c in cfg["campaigns"] if c not in smoke["finite"]["campaigns"]]
        if missing:
            raise ValueError(f"smoke.json of N{n} lacks campaigns {missing}")
        art, rep, change = smoke["artifact"], smoke["replay"], smoke["change_vs_frozen_oracles"]
        preflight_pass = bool(preflight_payload["cases"][str(n)].get("all_pass"))
        grids[str(n)] = {
            "smoke_pass": bool(smoke["smoke_pass"] and preflight_pass), "all_finite": bool(smoke["finite"]["all_finite"]),
            "nonfinite_total": smoke["finite"]["nonfinite_total"], "preflight_pass": preflight_pass,
            "artifact_total_bytes": art["total_bytes"], "bytes_per_row_kind": art["bytes_per_row_kind"],
            "row_counts": art["row_counts"], "point_row_families": art["point_row_families"],
            "coupled_quartic": art["coupled_quartic"], "tensor_encoding": art["tensor_encoding"],
            "change_vs_frozen_oracles": {"informational": True, "gate": False, "status": change["status"],
                                         "max_tier_b_ratio": change.get("max_tier_b_ratio"),
                                         "max_tier_b_ratio_term": change.get("max_tier_b_ratio_term"),
                                         "error": change.get("error")}}
        summary_grids[str(n)] = {
            "artifact_path": art["path"], "total_bytes": art["total_bytes"], "bytes_per_row_kind": art["bytes_per_row_kind"],
            "coupled_quartic": art["coupled_quartic"], "build_wall_seconds": art["build"].get("wall_seconds"),
            "build_peak_rss_gib": art["build"].get("peak_rss_gib"), "replay_wall_seconds": rep["wall_seconds"],
            "replay_peak_rss_gib": rep["peak_rss_gib"], "plan_bytes": (rep.get("plan") or {}).get("plan_bytes"),
            "smoke_pass": grids[str(n)]["smoke_pass"]}
    payload = {"identity": identity, "operational_complete": True, "resolutions": list(args.resolutions),
               "operator_options": options, "acceptance_gate": None,
               "scope": {"smoke_only": True, "oracle_comparison_is_gate": False},
               "smoke_pass": bool(all(g["smoke_pass"] for g in grids.values())), "grids": grids}
    write(output / "validation.json", payload)
    write(output / "summary" / "step4_summary.json", {"identity": identity, "operator_options": options,
                                                        "smoke_pass": payload["smoke_pass"], "grids": summary_grids})
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("verify-inputs", "preflight", "run", "validate", "run-stage"))
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolutions", type=int, nargs="+", choices=GRIDS, default=[32, 48, 64])
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--n", type=int, choices=GRIDS)
    p.add_argument("--workers", type=int, default=4, help="artifact build process-pool size (the JAX replay is "
                                                          "one process)")
    p.add_argument("--memory-budget-gib", type=float, default=None)
    p.add_argument("--worker-memory-gib", type=float, default=None)
    p.add_argument("--memory-reserve-gib", type=float, default=1.0)
    p.add_argument("--max-tasks-per-worker", type=int, default=None)
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
                                   smoke_pass=result.get("smoke_pass"))
            elif args.command == "run":
                _require_preflight(args.output, identity, args.resolutions, what="run")
                grids = [run_grid(n=n, args=args, identity=identity, sidecar_path=sidecar_path, paths=paths)
                         for n in args.resolutions]
                payload = validate(args=args, identity=identity)
                summary.update(status="complete", grids=grids, operational_complete=payload["operational_complete"],
                               smoke_pass=payload["smoke_pass"])
            else:                                                       # validate
                payload = validate(args=args, identity=identity)
                summary.update(status="complete", operational_complete=payload["operational_complete"],
                               smoke_pass=payload["smoke_pass"])
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
