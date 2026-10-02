#!/usr/bin/env python3
"""P08 step 5 re-freeze on the ``compact_c3`` magnetic evaluator: full-grid artifact + static Dirichlet-phi combined MMS.

ONE campaign, per grid 32 -> 48 -> 64, all inside a single output folder ``<OUT>``:

1. ``artifact`` -- the full-grid P row artifact (``p_shared.build_artifact.run_full_build``, step 1's chunk sizes) with
   the four operator options ``curvature=autodiff``, ``face_quadrature=q2``, ``inner_support=fixed_radius`` and
   ``bfield_toroidal=compact_c3`` (Q's compact C3 toroidal interpolation of the B evaluator; "C3" alone elsewhere means
   the inner donor support). The build identity must record all four (``smoke.check_artifact_options``). At N32 only, the
   freshly built artifact is gated by the 5.3-style bounded owner-subset check (:func:`artifact_gate`) before any
   expensive stage runs;
2. ``references`` -> ``jax`` -> ``reduce`` -- the step-5.3 static Dirichlet-phi combined MMS stages
   (:mod:`p08_step5_combined`) on that artifact. The campaign output folder doubles as the "step-4 folder" of those
   stage functions: ``<OUT>/localized_sidecar.json``, ``<OUT>/artifact/N{n}/``, and the 5.3 outputs (``N{n}/references``,
   ``N{n}/results.npz``, ``N{n}/summary.json``, ``work/_chunks``, ``summary/``, ``provenance/``) exactly where the 5.3
   modules put them.

This is the one-time re-freeze of the references on ``compact_c3`` (user decision, 2 October 2026). The step-4 smoke
replay against the frozen step-1 oracles is NOT part of it (``p08_step2_global.replay`` rejects four-key options);
finiteness of the full-grid RHS is covered by the jax-stage gates. The packages ``p08_step4_global`` and
``p08_step5_combined`` pin exactly three options and are not touched: this module passes its four-key options directly
to the lower-level functions and reads the inherited step-4 build / preflight settings through ``step4.config()``.

Commands: ``verify-inputs``; ``preflight`` (N32, bounded, no build: the 12-owner host-vs-JAX closure check with the four
options; must pass before ``run``); ``run`` (per grid artifact -> [N32 owner-subset gate] -> references -> jax -> reduce,
then ``validate`` and ``analyze``); ``run-stage --stage {artifact,references,jax,reduce} --n N``; ``validate``;
``analyze``. Run as ``python -m p08_step5_compact_c3.campaign`` from ``DRBX/scripts``. See the README.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import gc
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

import numpy as np                                                                  # noqa: E402

from p_shared import build_artifact as build_artifact_mod                          # noqa: E402
from p_shared import oracle_manifest as om                                          # noqa: E402
from p_shared import runner                                                         # noqa: E402
from p_shared.replay_support import CAMPAIGN_FUNCS                                  # noqa: E402
from p08_step2_global import campaign as step2                                      # noqa: E402
from p08_step4_global import campaign as step4                                      # noqa: E402
from p08_step4_global import smoke                                                  # noqa: E402
from p08_step5_combined import analysis, checkpoints, references, reduction         # noqa: E402
from p08_step5_combined import campaign as step5                                    # noqa: E402
from p08_step5_export import campaign as export                                     # noqa: E402

GRIDS = (32, 48, 64)
STAGES = ("artifact", "references", "jax", "reduce")
SCHEMA = "drbx.p08-step5-compact-c3-v1"
#: the final operator bundle on the compact_c3 magnetic evaluator, pinned: ``configuration.json`` must say exactly this
#: (and is hashed into the identity)
OPERATOR_OPTIONS = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius",
                    "bfield_toroidal": "compact_c3"}
#: the pinned G3.3 parameters (rho_star, tau, D_f of n, Te, Ti, omega)
PARAMS = {"rho_star": 0.05, "tau": 1.0, "diffusion": {"density": 0.01, "Te": 0.01, "Ti": 0.01, "vorticity": 0.01}}
#: the pinned phi solve: Dirichlet, block-Jacobi eta-plane preconditioner, float32 factors, MEASUREMENT tolerance 1e-11
PHI_SOLVE = {"phi_solve_boundary": "dirichlet", "phi_solve_preconditioner": "block_jacobi_eta_plane",
             "phi_solve_factor_dtype": "float32", "phi_solve_rtol": 1e-11, "phi_solve_restart": 50,
             "phi_solve_max_restarts": 40}
#: the pinned catalogue (frozen P06N variants with Dirichlet phi)
VARIANTS = ("main_phi_dirichlet", "heldout_phi_dirichlet", "dirichlet_rich", "control_constant_dirichlet")
#: the settings read from ``step4.config()`` (build chunk sizes, closure-preflight blocking, catalogue of campaigns)
INHERITED_KEYS = ("campaigns", "cell_chunk_size", "face_chunk_size", "p07_chunk_size", "geometry_raw_chunk_size",
                  "geometry_face_chunk_size", "column_block", "p06n_variant_block", "wall_cache",
                  "preflight_floor_seeds", "boundary_batch")

sha = runner.sha256_file
digest = runner.digest
write = runner.write_json
lock = runner.lock
oracle_paths = step4.oracle_paths
_effective_workers = step4._effective_workers
_refuse_csr_only = step2._refuse_csr_only


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def config() -> dict:
    """``configuration.json``, refused if it drifts from the pinned literals above."""
    cfg = json.loads((HERE / "configuration.json").read_text())
    if cfg["schema"] != SCHEMA:
        raise ValueError("unsupported P08 step-5 compact_c3 campaign contract")
    if cfg["operator_options"] != OPERATOR_OPTIONS:
        raise ValueError(f"configuration.json operator_options {cfg['operator_options']} differ from the pinned "
                         f"final bundle {OPERATOR_OPTIONS}")
    if cfg["params"] != PARAMS:
        raise ValueError(f"configuration.json params {cfg['params']} differ from the pinned {PARAMS}")
    for key, value in PHI_SOLVE.items():
        if cfg.get(key) != value:
            raise ValueError(f"configuration.json {key}={cfg.get(key)!r} differs from the pinned {value!r}")
    if tuple(cfg["variants"]) != VARIANTS:
        raise ValueError(f"configuration.json variants {tuple(cfg['variants'])} differ from the pinned {VARIANTS}")
    if tuple(cfg["inherited_from_step4"]["keys"]) != INHERITED_KEYS:
        raise ValueError("configuration.json inherited_from_step4 keys differ from the pinned list")
    if not set(cfg["resolutions"]) <= set(GRIDS):
        raise ValueError("bad resolutions in configuration.json")
    if cfg["phi_solve_rtol"] > cfg["phi_solve_rtol_production_default"]:
        raise ValueError("the measurement tolerance must not be looser than the production default")
    return cfg


def operator_options(cfg: dict | None = None) -> dict:
    """The pinned four-key options, read from the configuration (so that a changed configuration changes every call)."""
    return dict((cfg or config())["operator_options"])


def inherited_settings(cfg: dict | None = None) -> dict:
    """The step-4 (merged over step 2 / step 1) build chunk sizes and bounded-preflight settings this campaign reuses."""
    cfg = cfg or config()
    merged = step4.config()
    out = {key: merged[key] for key in cfg["inherited_from_step4"]["keys"]}
    if out["campaigns"] != list(CAMPAIGN_FUNCS):
        raise ValueError("campaign catalogue changed relative to p_shared.replay_support.CAMPAIGN_FUNCS")
    return out


_OWN_SOURCES = [
    "scripts/p08_step5_compact_c3/campaign.py",
    "scripts/p08_step5_compact_c3/configuration.json",
]
_C3_SOURCES = [
    "scripts/p_shared/build_artifact.py",
    "scripts/p_shared/bfield.py",
    "scripts/p_shared/provider.py",
    "src/drbx/geometry/Bfield_evaluator.py",
    "src/drbx/geometry/compact_toroidal.py",
    "src/drbx/geometry/jax_bfield_evaluator.py",
]
#: the step-4 / step-2 / step-1 modules and files this campaign reads settings or functions from
_INHERITED_SOURCES = [
    "scripts/p08_step4_global/campaign.py",
    "scripts/p08_step4_global/configuration.json",
    "scripts/p08_step2_global/campaign.py",
    "scripts/p08_step2_global/configuration.json",
    "scripts/p08_step1_global/campaign.py",
    "scripts/p08_step1_global/configuration.json",
    "scripts/p08_step1_global/input_manifest.json",
    "scripts/p08_step1_global/oracle_manifest.json",
]
#: this package, the 5.3 sources, the artifact-build sources of step 4 (which pin the option modules), the compact_c3
#: evaluator sources and the inherited settings files
SOURCE_FILES = list(dict.fromkeys([*_OWN_SOURCES, *step5.SOURCE_FILES, *step4.SOURCE_FILES, *_C3_SOURCES,
                                   *_INHERITED_SOURCES]))


def source_hashes() -> dict:
    return {p: sha(REPO / p) for p in SOURCE_FILES}


def _commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Folder layout (the campaign output folder is also the "step-4 folder" of the 5.3 stage functions)
# ---------------------------------------------------------------------------
def sidecar_path(output) -> Path:
    return Path(output) / "localized_sidecar.json"


def artifact_root(output) -> Path:
    return Path(output) / "artifact"


def inputs_path(output) -> Path:
    return Path(output) / "provenance" / "inputs.json"


# ---------------------------------------------------------------------------
# Inputs and identity
# ---------------------------------------------------------------------------
def p06n_owner_value_entries() -> dict:
    """The committed oracle manifest's P06N ``owner_values`` entry of every grid (the frozen owner averages the 5.3 arms
    are fed with)."""
    p06n = (step4.committed_oracle_manifest().get("campaigns") or {}).get("p06n")
    if not p06n:
        raise ValueError("the committed oracle manifest has no p06n entry")
    entries = {}
    for n in GRIDS:
        files = [e for e in (p06n.get(str(n)) or {}).get("files", []) if e["path"].endswith(f"N{n}.owner_values.npz")]
        if len(files) != 1:
            raise ValueError(f"the committed oracle manifest has no unique p06n N{n} owner_values entry")
        entries[str(n)] = files[0]
    return entries


def verify_oracle_owner_values(entries: dict, oracle_root: Path, grids) -> list[str]:
    """Hash-check the P06N ``owner_values`` files of ``grids`` under ``oracle_root`` (only these are read by ``run``)."""
    sliced = {"campaigns": {"p06n": {str(n): {"files": [entries[str(n)]], "missing": []} for n in grids}}}
    return om.verify_manifest(sliced, oracle_root)


def read_artifacts(output, options: dict) -> dict:
    """The assembled artifacts under ``<output>/artifact`` (``build_receipt.json`` is written last by the build): per
    grid the directory, the sha256 of the digest of ``build_identity.json`` (what ``jaxstage`` checks) and of
    ``manifest.json``. An assembled artifact whose build identity does not carry the pinned options is an error."""
    found = {}
    for n in GRIDS:
        gdir = artifact_root(output) / f"N{n}"
        if not all((gdir / name).is_file() for name in ("build_receipt.json", "build_identity.json", "manifest.json")):
            continue
        identity = json.loads((gdir / "build_identity.json").read_text())
        smoke.check_artifact_options(identity, options)
        found[str(n)] = {"artifact_dir": str(gdir.resolve()), "artifact_identity_sha256": digest(identity),
                         "manifest_sha256": sha(gdir / "manifest.json")}
    return found


def verify_inputs(*, input_root: Path, oracle_root: Path | None, output: Path, grids,
                  closure_oracles: bool = False) -> tuple[str, dict]:
    """Verify the immutable inputs (all hashes), the P06N owner-value oracle files of ``grids`` (all seven campaigns'
    N32 oracle files when ``closure_oracles``, for the closure preflight) and the backend; localize the sidecar; record
    / check ``provenance/inputs.json``. Returns ``(identity, record)``. Refuses (raises) if the identity
    (configuration, inherited settings, sources, input manifest, localized sidecar, P06N oracle entries), the oracle
    root, the localized sidecar or the identity of an already-built artifact changed relative to a previous run at the
    same ``output`` (new folder required)."""
    cfg = config()
    inherited = inherited_settings(cfg)
    options = operator_options(cfg)
    input_root, output = Path(input_root), Path(output)
    if not input_root.is_dir():
        raise ValueError(f"input root is not a directory: {input_root}")
    manifest = step4._input_manifest()
    for rec in manifest["files"]:
        path = input_root / rec["path"]
        if not path.is_file() or path.stat().st_size != rec["bytes"] or sha(path) != rec["sha256"]:
            raise ValueError(f"missing or changed immutable input: {path}")
    info = export.jax_check()
    local = step4.localize_sidecar(input_root, output)
    resolved_oracle_root = Path(oracle_root) if oracle_root is not None else input_root
    entries = p06n_owner_value_entries()
    errors = verify_oracle_owner_values(entries, resolved_oracle_root, [int(g) for g in grids])
    if closure_oracles:
        errors += om.verify_manifest(step4.committed_oracle_manifest(), resolved_oracle_root, grids=[32])
    if errors:
        raise ValueError("oracle verification failed:\n" + "\n".join(errors[:50]) +
                         (f"\n(+{len(errors) - 50} more)" if len(errors) > 50 else ""))
    sources = source_hashes()
    identity = digest({"configuration": cfg, "inherited": inherited, "input_manifest": manifest, "sources": sources,
                       "localized_sidecar_sha256": sha(local), "oracle_p06n_owner_values": entries})
    artifacts = read_artifacts(output, options)
    path = inputs_path(output)
    previous = json.loads(path.read_text()) if path.is_file() else None
    if previous is not None:
        if previous["identity"] != identity:
            raise ValueError("campaign identity changed; use a new output folder")
        if previous.get("oracle_root") != str(resolved_oracle_root.resolve()):
            raise ValueError("oracle root changed; use a new output folder")
        for n, rec in artifacts.items():
            old = previous["grids"].get(n)
            if old is not None and old["artifact_identity_sha256"] != rec["artifact_identity_sha256"]:
                raise ValueError(f"artifact identity of N{n} changed; use a new output folder")
    record = {"identity": identity, "campaign": "p08_step5_compact_c3", "configuration": cfg, "inherited": inherited,
              "operator_options": options, "source_hashes": sources, "commit": _commit(),
              "input_manifest_files": len(manifest["files"]), "localized_sidecar_sha256": sha(local),
              "oracle_p06n_owner_values": entries,
              "grids": {**(previous["grids"] if previous else {}), **artifacts},
              "input_root": str(input_root.resolve()), "oracle_root": str(resolved_oracle_root.resolve()),
              "jax": info, "python": sys.version, "platform": platform.platform()}
    write(path, record)
    return identity, record


def record_artifact(output, inputs: dict, n: int) -> dict:
    """After a build: read the assembled artifact of grid ``n``, refuse a changed identity relative to the one already
    recorded, update ``inputs`` (in place) and ``provenance/inputs.json``. Returns the grid's record."""
    found = read_artifacts(output, operator_options())
    rec = found.get(str(n))
    if rec is None:
        raise ValueError(f"the artifact of N{n} is not assembled under {artifact_root(output) / f'N{n}'}")
    old = inputs["grids"].get(str(n))
    if old is not None and old["artifact_identity_sha256"] != rec["artifact_identity_sha256"]:
        raise ValueError(f"artifact identity of N{n} changed; use a new output folder")
    inputs["grids"][str(n)] = rec
    write(inputs_path(output), inputs)
    return rec


def _require_artifact(inputs: dict, n: int) -> None:
    if str(n) not in inputs["grids"]:
        raise ValueError(f"the artifact of N{n} is not built; run the artifact stage first")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def artifact_stage(*, n: int, args, identity: str, inputs: dict) -> dict:
    """``run_full_build`` with the four pinned options and step 1's chunk sizes (unit-parallel, resumable, tensor
    encoding on; skipped when assembled). The build identity must record all four options."""
    _refuse_csr_only()
    cfg = config()
    options = operator_options(cfg)
    inh = inherited_settings(cfg)
    effective = _effective_workers(args.workers, args.memory_budget_gib, args.worker_memory_gib,
                                   args.memory_reserve_gib)
    receipt = build_artifact_mod.run_full_build(
        n=n, input_root=args.input_root, sidecar_path=sidecar_path(args.output), output=artifact_root(args.output),
        workers=effective, memory_budget_gib=args.memory_budget_gib, worker_memory_gib=args.worker_memory_gib,
        memory_reserve_gib=args.memory_reserve_gib,
        cell_chunk_size=inh["cell_chunk_size"], face_chunk_size=inh["face_chunk_size"],
        p07_chunk_size=inh["p07_chunk_size"], geometry_raw_chunk_size=inh["geometry_raw_chunk_size"],
        geometry_face_chunk_size=inh["geometry_face_chunk_size"], max_tasks_per_worker=args.max_tasks_per_worker,
        **options)
    smoke.check_artifact_options(receipt["identity"], options)
    rec = record_artifact(args.output, inputs, n)
    return {**receipt, "artifact_identity_sha256": rec["artifact_identity_sha256"]}


def references_stage(*, n: int, args, identity: str) -> dict:
    cfg = config()
    effective = _effective_workers(args.workers, args.memory_budget_gib, args.worker_memory_gib,
                                   args.memory_reserve_gib)
    return references.references_stage(
        n=n, cfg=cfg, options=operator_options(cfg), input_root=args.input_root,
        sidecar_path=sidecar_path(args.output), output=args.output, identity=identity, workers=effective,
        max_tasks_per_worker=args.max_tasks_per_worker)


def jax_stage(*, n: int, args, identity: str, inputs: dict) -> dict:
    from p08_step5_combined import jaxstage
    _require_artifact(inputs, n)
    cfg = config()
    return jaxstage.jax_stage(n=n, cfg=cfg, options=operator_options(cfg), step4=args.output,
                              input_root=args.input_root, output=args.output, identity=identity, inputs=inputs,
                              paths=oracle_paths(args.oracle_root, args.input_root))


def reduce_stage(*, n: int, args, identity: str, inputs: dict) -> dict:
    _require_artifact(inputs, n)
    return reduction.reduce_grid(n=n, cfg=config(), output=args.output, identity=identity, inputs=inputs)


def run_grid(*, n: int, args, identity: str, inputs: dict) -> dict:
    """One grid of ``run``: artifact (skipped when assembled), at N32 the owner-subset gate (stops the grid when it
    fails), then references, JAX stage, reduction (each skipped / resumed when its outputs are valid)."""
    build = artifact_stage(n=n, args=args, identity=identity, inputs=inputs)
    out = {"n": int(n), "artifact_reentry": build.get("reentry"), "artifact_wall_seconds": build.get("wall_seconds")}
    if n == 32:
        gate = artifact_gate(args=args, identity=identity, inputs=inputs)
        out["artifact_gate_pass"] = gate["all_pass"]
        if not gate["all_pass"]:
            return {**out, "status": "failed_artifact_gate", "grid_pass": False}
    refs = references_stage(n=n, args=args, identity=identity)
    jax_summary = jax_stage(n=n, args=args, identity=identity, inputs=inputs)
    summary = reduce_stage(n=n, args=args, identity=identity, inputs=inputs)
    return {**out, "status": "complete", "grid_pass": summary["gates"]["grid_pass"],
            "solver_gates_pass": summary["gates"]["solver_gates_pass"], "all_finite": summary["gates"]["all_finite"],
            "references_skipped": bool(refs.get("skipped")), "variants_computed": jax_summary["computed"],
            "variants_resumed": jax_summary["resumed"]}


# ---------------------------------------------------------------------------
# Preflight 1 (before any build): the bounded 12-owner host-vs-JAX closure check, N32
# ---------------------------------------------------------------------------
def preflight_grid(*, n: int, input_root: Path, sidecar: Path, output: Path, paths: dict) -> dict:
    """The bounded gate of grid ``n``: 12 owners, host + JAX assembly through the blocked full-grid path with the four
    pinned options (inherited blocking). Pass = every host-vs-JAX policy row passes, no ``uniq_mismatches``, the payload
    records the pinned options, and the environment's reference carries ``bfield_toroidal == "compact_c3"`` (provenance
    and the evaluator's ``toroidal_method``). The ``compare_to_oracle`` rows are informational (the frozen oracles
    describe the old fd / q3 / profile7 spline operator). Also requires the CPU backend and x64."""
    from p_shared import jax_replay as jr
    from p_shared.replay_support import build_environment

    cfg = config()
    options = operator_options(cfg)
    inh = inherited_settings(cfg)
    info = export.jax_check()
    started = time.time()
    env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar), **options)
    provenance = dict(getattr(env.ref, "provenance", None) or {})
    evaluator = getattr(getattr(env.ref, "bfield_evaluator", None), "toroidal_method", None)
    env_bfield = getattr(env, "bfield_toroidal", None)
    env_seconds = time.time() - started
    del env
    gc.collect()
    evaluator_ok = bool(provenance.get("bfield_toroidal") == options["bfield_toroidal"]
                        and evaluator == options["bfield_toroidal"] and env_bfield == options["bfield_toroidal"])
    report = Path(output) / "preflight" / f"N{n}_owner_closure.json"
    payload = jr.run_jax_owner_closure_check(
        n=n, input_root=input_root, sidecar_path=sidecar, paths=paths, campaigns=tuple(inh["campaigns"]),
        floor_seeds=tuple(inh["preflight_floor_seeds"]), wall_cache=bool(inh["wall_cache"]), output=report,
        column_block=inh["column_block"], variant_block=inh["p06n_variant_block"],
        boundary_batch=inh["boundary_batch"], **options)
    recorded = {"curvature": payload.get("curvature", "fd"), "face_quadrature": payload.get("face_quadrature", "q3"),
                "inner_support": payload.get("inner_support", "profile7"),
                "bfield_toroidal": payload.get("bfield_toroidal", "spline")}
    options_recorded = recorded == options
    all_pass = bool(payload["all_diff_pass"] and not payload["uniq_mismatches"] and options_recorded and evaluator_ok)
    oracle_rows = payload["oracle_jax"]
    ratios = [r["ratio_to_oracle_NR"] for r in oracle_rows if r.get("ratio_to_oracle_NR") is not None]
    return {"n": int(n), "all_pass": all_pass, "operator_options": options, "options_recorded": options_recorded,
            "bfield": {"env_ref_provenance_bfield_toroidal": provenance.get("bfield_toroidal"),
                       "evaluator_toroidal_method": evaluator, "env_bfield_toroidal": env_bfield,
                       "ok": evaluator_ok, "environment_seconds": env_seconds},
            "jax": info, "policy_rows": len(payload["diff_table"]), "policy_failures": payload["diff_failures"],
            "uniq_mismatches": payload["uniq_mismatches"],
            "oracle_informational": {"gate": False, "rows": len(oracle_rows),
                                     "rows_not_matching_frozen_oracles": [(r["campaign"], r["term"])
                                                                          for r in oracle_rows if not r["pass"]],
                                     "max_ratio_to_oracle_NR": max(ratios) if ratios else None,
                                     "note": "frozen oracles describe the old fd/q3/profile7 spline operator; "
                                             "differences are expected and do not gate"},
            "plan": payload["plan"], "blocking": payload["blocking"],
            "peak_rss_gib": payload["peak_rss_gib"], "closure_seconds": payload["wall_seconds"],
            "closure_report": str(report), "seconds": time.time() - started}


def _load_json_for(path: Path, identity: str) -> dict | None:
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    return saved if saved.get("identity") == identity else None


def _load_preflight(output: Path, identity: str) -> dict | None:
    return _load_json_for(Path(output) / "preflight.json", identity)


def preflight(*, args, identity: str) -> dict:
    output = Path(args.output)
    previous = _load_preflight(output, identity)
    cases = dict(previous["cases"]) if previous else {}
    done = cases.get("32")
    if done is None or not done.get("all_pass"):                 # resumable: a passed preflight is not redone
        cases["32"] = checkpoints.jsonable(preflight_grid(
            n=32, input_root=args.input_root, sidecar=sidecar_path(output), output=output,
            paths=oracle_paths(args.oracle_root, args.input_root)))
    payload = {"identity": identity, "cases": cases, "all_pass": bool(all(c.get("all_pass") for c in cases.values()))}
    write(output / "preflight.json", payload)
    return payload


def _require_preflight(output: Path, identity: str, *, what: str) -> dict:
    payload = _load_preflight(output, identity)
    if payload is None:
        raise ValueError(f"{what} requires a matching preflight first")
    if not (payload["cases"].get("32") or {}).get("all_pass"):
        raise ValueError(f"preflight did not pass; refusing to {what}")
    return payload


# ---------------------------------------------------------------------------
# Preflight 2 (after the N32 build): the 5.3-style owner-subset gate on the freshly built artifact
# ---------------------------------------------------------------------------
def artifact_gate_grid(*, n: int, args, identity: str, inputs: dict) -> dict:
    """The bounded gate of the built artifact of grid ``n`` (5.3's ``preflight_grid`` with this campaign's four-key
    options, configuration and folder): the reference stage on the 12-owner subset (the pool-free
    ``references.reference_chunk``) and the JAX stage on the FULL plan of the built artifact for one variant (both arms,
    solver gates). In the solved arm the right-hand side is ``A psi_bar + B g`` (the full-grid ``O(psi)`` does not exist
    yet); the subset ``O(psi)`` is compared with it (sign / consistency check). Verdict: ``step5.preflight_checks``."""
    from p_shared import owner_closure as oc
    from p_shared.replay_support import build_environment
    from p08_step5_combined import jaxstage

    cfg = config()
    options = operator_options(cfg)
    info = export.jax_check()
    started = time.time()
    variant = cfg["preflight"]["variant"]
    fs = cfg["variants"][variant]["field_set"]
    references.check_catalogue(cfg)
    env = build_environment(n=n, input_root=args.input_root, sidecar_path=sidecar_path(args.output), **options)
    owners = np.asarray(sorted(set(oc.select_owners(env.t, env.census).values())), dtype=np.int64)
    states = {fs: references._prr().p06n_state(env, cfg["field_sets"][fs]["representative"])}
    t0 = time.time()
    ref = references.reference_chunk(env, owners, states=states, params=cfg["params"])
    ref_seconds = time.time() - t0
    prep = jaxstage.prepare(n=n, cfg=cfg, options=options, step4=args.output, input_root=args.input_root,
                            paths=oracle_paths(args.oracle_root, args.input_root), inputs=inputs, env=env)
    arrays, vinfo = jaxstage.run_variant(prep, variant, cfg, omega_rhs=None)
    verdict = step5.preflight_checks(ref=ref, arrays=arrays, info=vinfo, owners=owners, volume=prep.owner_volume,
                                     fs=fs, cfg=cfg)
    detail = {"n": int(n), "variant": variant, "jax": info, "operator_options": options, "subset_owners": owners,
              "reference_seconds": ref_seconds, "setup_seconds": prep.setup_seconds, "plan": prep.plan_summary,
              "variant_info": vinfo, **verdict, "seconds": time.time() - started}
    write(Path(args.output) / "preflight" / f"N{n}_artifact_subset.json", checkpoints.jsonable(detail))
    return {"n": int(n), "all_pass": verdict["all_pass"], "checks": verdict["checks"], "variant": variant,
            "n_owners_subset": verdict["n_owners_subset"], "n_minus_o_psi_subset": verdict["n_minus_o_psi_subset"],
            "prescribed_sanity_worst": verdict["prescribed_sanity_worst"],
            "arm_difference_worst": verdict["arm_difference_worst"], "solver_gates": vinfo["gates"],
            "consistency_error": vinfo["consistency"]["relative_error_M"], "seconds": time.time() - started}


def _load_artifact_gate(output: Path, identity: str, inputs: dict) -> dict | None:
    saved = _load_json_for(Path(output) / "preflight_artifact.json", identity)
    if saved is None or saved.get("artifact_identity_sha256") != checkpoints.artifact_sha(inputs, 32):
        return None
    return saved


def artifact_gate(*, args, identity: str, inputs: dict) -> dict:
    """The N32 owner-subset gate (resumable: a passed gate of the same identity and artifact is not redone)."""
    output = Path(args.output)
    _require_artifact(inputs, 32)
    previous = _load_artifact_gate(output, identity, inputs)
    if previous is not None and previous["case"].get("all_pass"):
        return previous["case"]
    case = checkpoints.jsonable(artifact_gate_grid(n=32, args=args, identity=identity, inputs=inputs))
    write(output / "preflight_artifact.json", {"identity": identity, "n": 32,
                                               "artifact_identity_sha256": checkpoints.artifact_sha(inputs, 32),
                                               "case": case, "all_pass": bool(case["all_pass"])})
    return case


# ---------------------------------------------------------------------------
# Validation and analysis
# ---------------------------------------------------------------------------
def artifact_record(output, n: int, options: dict) -> dict:
    """Receipt data of the built artifact of grid ``n`` (plain JSON reads): the options recorded in the build identity
    (must equal the pinned four), row counts, bytes, point-row family counts (``coupled_quartic`` = the C3 inner
    support), build wall seconds and peak worker RSS."""
    gdir = artifact_root(output) / f"N{n}"
    identity = json.loads((gdir / "build_identity.json").read_text())
    smoke.check_artifact_options(identity, options)
    receipt = json.loads((gdir / "build_receipt.json").read_text())
    if receipt.get("identity") != identity:
        raise ValueError(f"build_receipt.json of N{n} belongs to another build identity")
    policy = identity.get("policy", {})
    families = dict((receipt.get("diagnostics") or {}).get("point_row_families", {}))
    per_kind = dict(receipt.get("bytes_per_row_kind") or {})
    return {"path": str(gdir.resolve()), "artifact_identity_sha256": digest(identity),
            "policy": {k: policy.get(k) for k in ("curvature", "quadrature", "inner_support", "bfield_toroidal")},
            "row_counts": receipt.get("row_counts"), "bytes_per_row_kind": per_kind,
            "row_bytes_total": int(sum(per_kind.values())), "coupled_quartic": families.get("coupled_quartic"),
            "point_row_families": families, "build_wall_seconds": receipt.get("wall_seconds"),
            "build_cpu_seconds": receipt.get("cpu_seconds"), "build_peak_rss_gib": receipt.get("peak_rss_gib"),
            "effective_workers": receipt.get("effective_workers"), "reentry": receipt.get("reentry")}


def validate(*, args, identity: str, inputs: dict) -> dict:
    """Requires the closure preflight pass, the N32 artifact-gate pass (when N32 is requested), the artifact of every
    requested grid (receipts with the pinned four options) and the reduction of every requested grid (this identity and
    artifact); writes ``validation.json`` and ``summary/artifacts.json``. ``solver_gates_pass`` = every variant's
    discrete-consistency gate and convergence on every grid; ``all_finite`` = every term finite; ``grids_pass`` adds the
    preflights and the artifact options. The order criterion is informational (``acceptance_gate`` is ``None``)."""
    output = Path(args.output)
    _require_preflight(output, identity, what="validate")
    cfg = config()
    options = operator_options(cfg)
    gate = None
    if 32 in args.resolutions:
        gate = _load_artifact_gate(output, identity, inputs)
        if gate is None or not gate["case"].get("all_pass"):
            raise ValueError("the N32 artifact owner-subset gate has not passed for this artifact; refusing to validate")
    grids, results_by_n, artifacts = {}, {}, {}
    for n in args.resolutions:
        _require_artifact(inputs, n)
        artifacts[str(n)] = artifact_record(output, n, options)
        if artifacts[str(n)]["artifact_identity_sha256"] != checkpoints.artifact_sha(inputs, n):
            raise ValueError(f"artifact of N{n} differs from provenance/inputs.json")
        summary, results = reduction.load_grid(output, n, identity)
        if summary.get("jax_identity") != checkpoints.jax_identity(identity, n, checkpoints.artifact_sha(inputs, n)):
            raise ValueError(f"the reduction of N{n} was computed on another artifact")
        results_by_n[int(n)] = results
        grids[str(n)] = {**summary["gates"], "n_owners": summary["n_owners"], "artifact_options_ok": True,
                         "solver": {v: {"consistency_error": rec["info"]["consistency"]["relative_error_M"],
                                        "iterations": rec["info"]["solved_solve"]["iterations"],
                                        "seconds": rec["info"]["solved_solve"]["seconds"],
                                        "converged": rec["info"]["gates"]["converged"]}
                                    for v, rec in summary["variants"].items()}}
    headline = analysis.headline_criterion(results_by_n, cfg)
    payload = {"identity": identity, "operational_complete": True, "resolutions": list(args.resolutions),
               "operator_options": options, "acceptance_gate": None, "user_decides_acceptance": True,
               "preflight_closure_pass": True, "artifact_gate_pass": None if gate is None else True,
               "solver_gates_pass": bool(all(g["solver_gates_pass"] for g in grids.values())),
               "all_finite": bool(all(g["all_finite"] for g in grids.values())),
               "grids_pass": bool(all(g["grid_pass"] for g in grids.values())),
               "headline_order_criterion": {k: headline[k] for k in ("informational", "user_decides", "pass",
                                                                      "min_order", "term", "grids", "rows_failed",
                                                                      "rows_undefined")},
               "scope": {"step4_smoke_replay_against_frozen_oracles": False, "neumann_phi_variants": False},
               "artifacts": artifacts, "grids": grids}
    write(output / "validation.json", checkpoints.jsonable(payload))
    write(output / "summary" / "artifacts.json", checkpoints.jsonable(
        {"identity": identity, "operator_options": options, "artifacts": artifacts}))
    return payload


def analyze(*, args, identity: str) -> dict:
    return analysis.analyze(output=args.output, identity=identity, grids=args.resolutions, cfg=config())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("verify-inputs", "preflight", "run", "run-stage", "validate", "analyze"))
    p.add_argument("--input-root", type=Path, required=True,
                   help="the immutable HSX input root holding the files of p08_step1_global/input_manifest.json")
    p.add_argument("--output", type=Path, required=True, help="the campaign folder (new; identity-checked)")
    p.add_argument("--oracle-root", type=Path, default=None,
                   help="where the P06N oracle files of the committed oracle manifest live (default: --input-root)")
    p.add_argument("--resolutions", type=int, nargs="+", choices=GRIDS, default=list(GRIDS))
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--n", type=int, choices=GRIDS)
    p.add_argument("--workers", type=int, default=4, help="process-pool size of the artifact build and the reference "
                                                          "stage (the JAX stage is one process)")
    p.add_argument("--memory-budget-gib", type=float, default=None)
    p.add_argument("--worker-memory-gib", type=float, default=None)
    p.add_argument("--memory-reserve-gib", type=float, default=1.0)
    p.add_argument("--max-tasks-per-worker", type=int, default=None)
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
        verify_grids = sorted(set(args.resolutions) | {32}) if args.command == "preflight" else args.resolutions
        identity, inputs = verify_inputs(input_root=args.input_root, oracle_root=args.oracle_root, output=args.output,
                                         grids=verify_grids, closure_oracles=args.command == "preflight")
        write(args.output / "invocations" / f"{time.time_ns()}_{args.command}.json",
              {"argv": list(sys.argv if argv is None else argv), "identity": identity, "time": time.time()})
        summary: dict = {"command": args.command, "identity": identity, "status": "complete"}
        ok = True
        if args.command == "preflight":
            payload = preflight(args=args, identity=identity)
            summary.update(all_pass=payload["all_pass"], cases=payload["cases"])
            ok = payload["all_pass"]
        elif args.command == "run-stage":
            if args.stage is None or args.n is None:
                raise ValueError("--stage and --n are required for run-stage")
            if args.n not in args.resolutions:
                raise ValueError(f"--n {args.n} is not among the verified --resolutions {args.resolutions}")
            if args.stage == "artifact":
                receipt = artifact_stage(n=args.n, args=args, identity=identity, inputs=inputs)
                summary.update(stage="artifact", n=args.n, wall_seconds=receipt.get("wall_seconds"),
                               reentry=receipt.get("reentry"))
                if args.n == 32:
                    gate = artifact_gate(args=args, identity=identity, inputs=inputs)
                    summary.update(artifact_gate_pass=gate["all_pass"])
                    ok = bool(gate["all_pass"])
            elif args.stage == "references":
                result = references_stage(n=args.n, args=args, identity=identity)
                summary.update(stage="references", n=args.n, n_owners=result["n_owners"], skipped=result["skipped"])
            elif args.stage == "jax":
                result = jax_stage(n=args.n, args=args, identity=identity, inputs=inputs)
                summary.update(stage="jax", n=args.n, computed=result["computed"], resumed=result["resumed"])
            else:
                result = reduce_stage(n=args.n, args=args, identity=identity, inputs=inputs)
                summary.update(stage="reduce", n=args.n, gates=result["gates"])
                ok = result["gates"]["grid_pass"]
        elif args.command == "run":
            _require_preflight(args.output, identity, what="run")
            done, requested = [], list(args.resolutions)
            for n in requested:
                grid = run_grid(n=n, args=args, identity=identity, inputs=inputs)
                done.append(grid)
                if not grid["grid_pass"]:
                    ok = False
                    break                                       # the data is checkpointed; do not burn the finer grids
            summary["grids"] = done
            completed = [g["n"] for g in done if g["status"] == "complete"]
            if completed:
                args.resolutions = completed
                payload = validate(args=args, identity=identity, inputs=inputs)
                analyze(args=args, identity=identity)
                summary.update(solver_gates_pass=payload["solver_gates_pass"], all_finite=payload["all_finite"])
            summary.update(operational_complete=len(completed) == len(requested))
        elif args.command == "validate":
            payload = validate(args=args, identity=identity, inputs=inputs)
            summary.update(solver_gates_pass=payload["solver_gates_pass"], all_finite=payload["all_finite"],
                           headline_order_pass=payload["headline_order_criterion"]["pass"])
            ok = payload["grids_pass"]
        elif args.command == "analyze":
            payload = analyze(args=args, identity=identity)
            summary.update(grids=payload["grids"], headline_order_pass=payload["headline_order_criterion"]["pass"])
        if not ok:
            summary["status"] = "failed_gates"
        write(args.output / "last_exit.json", {"command": args.command, "status": summary["status"],
                                               "time": time.time(), "summary": checkpoints.jsonable(summary)})
        print(json.dumps(checkpoints.jsonable(summary), default=str), flush=True)
        if not ok:
            raise SystemExit(1)
        return summary


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        try:
            a = parse()
            write(a.output / "last_exit.json", {"command": a.command, "status": "failed", "error": repr(exc),
                                                "time": time.time()})
        finally:
            raise
