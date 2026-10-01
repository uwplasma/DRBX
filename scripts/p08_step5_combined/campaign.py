#!/usr/bin/env python3
"""P08 step 5.3: static Dirichlet-phi combined full-grid MMS (N32 / N48 / N64), remote campaign.

Per grid, in three resumable stages (``run`` does all of them for each grid, 32 -> 48 -> 64):

1. ``references`` -- the re-frozen owner references of the final operator (:mod:`p08_step5_combined.references`):
   per field set the bracket / curvature / diffusion references of n, Te, Ti, omega over ALL owners (owner chunks, spawn
   process pool, ``--workers``), plus the positive owner operator ``O(psi)`` and the autodiff midpoint reference of
   ``psi = phi + tau Ti``;
2. ``jax`` -- one process: environment, step-4 artifact, full plan, Dirichlet phi solver, then per variant both arms
   (prescribed ``phi_bar``; solved: P07 Dirichlet solve of ``psi`` with the exact wall trace) through
   ``perpendicular_rhs`` and the solver gates (:mod:`p08_step5_combined.jaxstage`);
3. ``reduce`` -- ``N{n}/results.npz`` + ``N{n}/summary.json`` (:mod:`p08_step5_combined.reduction`).

``analyze`` writes the across-grid orders (``summary/step5_combined_report.md`` / ``.json``); ``validate`` records the
pass/fail gates (solver consistency and convergence, finiteness; the headline order criterion is informational: the
user decides acceptance). The pinned scientific contract is ``configuration.json`` (checked against the literals
below); see the README.

Commands: ``verify-inputs``, ``preflight`` (N32, bounded; must pass before ``run``), ``run``, ``run-stage --stage
{references,jax,reduce} --n N``, ``validate``, ``analyze``. Run as ``python -m p08_step5_combined.campaign`` from
``DRBX/scripts``. The inputs are the step-4 campaign folder (``--step4-campaign``: ``artifact/N{n}``,
``localized_sidecar.json``, ``validation.json``, ``oracle_manifest.json``, ``campaign_manifest.json``); no artifact
is rebuilt here.
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

import numpy as np                                                                  # noqa: E402

from p_shared import oracle_manifest as om                                          # noqa: E402
from p_shared import runner                                                         # noqa: E402
from p08_step1_global import campaign as step1                                      # noqa: E402
from p08_step5_combined import analysis, checkpoints, references, reduction         # noqa: E402
from p08_step5_export import campaign as export                                     # noqa: E402

GRIDS = (32, 48, 64)
STAGES = ("references", "jax", "reduce")
SCHEMA = "drbx.p08-step5-combined-dirichlet-v1"
#: the final operator bundle, pinned: ``configuration.json`` must say exactly this (and is hashed into the identity)
OPERATOR_OPTIONS = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius"}
#: the pinned G3.3 parameters (rho_star, tau, D_f of n, Te, Ti, omega)
PARAMS = {"rho_star": 0.05, "tau": 1.0, "diffusion": {"density": 0.01, "Te": 0.01, "Ti": 0.01, "vorticity": 0.01}}
#: the pinned phi solve: Dirichlet, block-Jacobi eta-plane preconditioner, float32 factors, MEASUREMENT tolerance 1e-11
PHI_SOLVE = {"phi_solve_boundary": "dirichlet", "phi_solve_preconditioner": "block_jacobi_eta_plane",
             "phi_solve_factor_dtype": "float32", "phi_solve_rtol": 1e-11, "phi_solve_restart": 50,
             "phi_solve_max_restarts": 40}
#: the pinned catalogue (frozen P06N variants with Dirichlet phi)
VARIANTS = ("main_phi_dirichlet", "heldout_phi_dirichlet", "dirichlet_rich", "control_constant_dirichlet")

sha = runner.sha256_file
digest = runner.digest
write = runner.write_json
lock = step1.lock
oracle_paths = step1.oracle_paths
_effective_workers = step1._effective_workers


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def config() -> dict:
    """``configuration.json``, refused if it drifts from the pinned literals above."""
    cfg = json.loads((HERE / "configuration.json").read_text())
    if cfg["schema"] != SCHEMA:
        raise ValueError("unsupported P08 step-5.3 combined campaign contract")
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
    if not set(cfg["resolutions"]) <= set(GRIDS):
        raise ValueError("bad resolutions in configuration.json")
    if cfg["phi_solve_rtol"] > cfg["phi_solve_rtol_production_default"]:
        raise ValueError("the measurement tolerance must not be looser than the production default")
    return cfg


def operator_options(cfg: dict | None = None) -> dict:
    return dict((cfg or config())["operator_options"])


SOURCE_FILES = [
    "scripts/p08_step5_combined/campaign.py",
    "scripts/p08_step5_combined/references.py",
    "scripts/p08_step5_combined/jaxstage.py",
    "scripts/p08_step5_combined/checkpoints.py",
    "scripts/p08_step5_combined/reduction.py",
    "scripts/p08_step5_combined/analysis.py",
    "scripts/p08_step5_combined/configuration.json",
    "scripts/p08_step5_export/campaign.py",
    "scripts/p08_step4_global/smoke.py",
    "scripts/p08_step2_global/replay.py",
    "scripts/p_shared/runner.py",
    "scripts/p_shared/jax_replay.py",
    "scripts/p_shared/owner_closure.py",
    "scripts/p_shared/replay_units.py",
    "scripts/p_shared/replay_support.py",
    "scripts/p_shared/campaign_fields.py",
    "scripts/p_shared/curvature_reference.py",
    "scripts/p_shared/perpendicular_reference_rhs.py",
    "scripts/p06n_field_derived_global/p06n_catalogue.json",
    "scripts/p06n_field_derived_global/core.py",
    "src/drbx/native/fci_perpendicular_rhs.py",
    "src/drbx/native/fci_perpendicular_p05_operator.py",
    "src/drbx/native/fci_perpendicular_p06_operator.py",
    "src/drbx/native/fci_perpendicular_p07_operator.py",
    "src/drbx/native/fci_perpendicular_reconstruction_state.py",
    "src/drbx/native/fci_perpendicular_p07_sparse.py",
    "src/drbx/native/fci_perpendicular_p07_solve.py",
    "src/drbx/native/fci_perpendicular_phi_solver.py",
    "src/drbx/native/fci_perpendicular_plane_preconditioner.py",
    "src/drbx/stencils/operator_plan.py",
]


def source_hashes() -> dict:
    return {p: sha(REPO / p) for p in SOURCE_FILES}


def _commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Inputs and identity
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    return export._read_json(path)


def read_step4(step4: Path, grids) -> dict:
    """The step-4 inputs for ``grids``: ``p08_step5_export.campaign.read_step4`` (validation / manifest identities,
    smoke and preflight passes, pinned options in ``validation.json``, artifact files) plus the step-4 oracle manifest's
    P06N ``owner_values`` entries (the frozen owner averages the arms are fed with)."""
    step4 = Path(step4)
    record = export.read_step4(step4, grids)
    manifest = _read_json(step4 / "oracle_manifest.json")
    p06n = (manifest.get("campaigns") or {}).get("p06n")
    if not p06n:
        raise ValueError("the step-4 oracle manifest has no p06n entry")
    entries = {}
    for n in GRIDS:
        files = [e for e in (p06n.get(str(n)) or {}).get("files", []) if e["path"].endswith(f"N{n}.owner_values.npz")]
        if len(files) != 1 and n in [int(g) for g in grids]:
            raise ValueError(f"the step-4 oracle manifest has no unique p06n N{n} owner_values entry")
        if files:
            entries[str(n)] = files[0]
    record["oracle_p06n_owner_values"] = entries
    record["oracle_manifest_sha256"] = sha(step4 / "oracle_manifest.json")
    return record


def verify_oracle_owner_values(inputs: dict, oracle_root: Path, grids) -> list[str]:
    """Hash-check the P06N ``owner_values`` files of ``grids`` under ``oracle_root`` (only these are read here)."""
    sliced = {"campaigns": {"p06n": {str(n): {"files": [inputs["oracle_p06n_owner_values"][str(n)]], "missing": []}
                                       for n in grids}}}
    return om.verify_manifest(sliced, oracle_root)


def verify_inputs(*, step4: Path, input_root: Path, oracle_root: Path | None, output: Path, grids) -> tuple[str, dict]:
    """Validate the step-4 inputs, the immutable input files, the P06N oracle files and the backend; record / check
    ``provenance/inputs.json``. Returns ``(identity, record)``. Refuses if the identity (configuration, sources,
    step-4 campaign, localized sidecar, oracle manifest) changed relative to a previous run at the same ``output``."""
    cfg = config()
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise ValueError(f"input root is not a directory: {input_root}")
    for rec in step1._input_manifest()["files"]:
        path = input_root / rec["path"]
        if not path.is_file() or path.stat().st_size != rec["bytes"]:
            raise ValueError(f"missing or changed immutable input: {path}")
    info = export.jax_check()
    inputs = read_step4(step4, grids)
    resolved_oracle_root = Path(oracle_root) if oracle_root is not None else input_root
    errors = verify_oracle_owner_values(inputs, resolved_oracle_root, [int(g) for g in grids])
    if errors:
        raise ValueError("oracle verification failed:\n" + "\n".join(errors[:50]))
    sources = source_hashes()
    identity = digest({"configuration": cfg, "sources": sources, "step4_identity": inputs["step4_identity"],
                       "localized_sidecar_sha256": inputs["localized_sidecar_sha256"],
                       "oracle_p06n_owner_values": inputs["oracle_p06n_owner_values"]})
    path = Path(output) / "provenance" / "inputs.json"
    previous = json.loads(path.read_text()) if path.is_file() else None
    if previous is not None:
        if previous["identity"] != identity:
            raise ValueError("campaign identity changed; use a new output folder")
        if previous.get("oracle_root") != str(resolved_oracle_root.resolve()):
            raise ValueError("oracle root changed; use a new output folder")
        for n, rec in inputs["grids"].items():
            old = previous["grids"].get(n)
            if old is not None and old["artifact_identity_sha256"] != rec["artifact_identity_sha256"]:
                raise ValueError(f"artifact identity of N{n} changed; use a new output folder")
    merged = {**(previous["grids"] if previous else {}), **inputs["grids"]}
    record = {"identity": identity, "campaign": "p08_step5_combined", "configuration": cfg,
              "operator_options": operator_options(cfg), "source_hashes": sources, "commit": _commit(),
              "step4_identity": inputs["step4_identity"], "step4_campaign": inputs["step4_campaign"],
              "localized_sidecar_sha256": inputs["localized_sidecar_sha256"],
              "oracle_manifest_sha256": inputs["oracle_manifest_sha256"],
              "oracle_p06n_owner_values": inputs["oracle_p06n_owner_values"], "grids": merged,
              "input_root": str(input_root.resolve()), "oracle_root": str(resolved_oracle_root.resolve()),
              "jax": info, "python": sys.version, "platform": platform.platform()}
    write(path, record)
    return identity, record


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _sidecar(step4) -> Path:
    return Path(step4) / "localized_sidecar.json"


def references_stage(*, n: int, args, identity: str) -> dict:
    cfg = config()
    effective = _effective_workers(args.workers, args.memory_budget_gib, args.worker_memory_gib,
                                   args.memory_reserve_gib)
    return references.references_stage(
        n=n, cfg=cfg, options=operator_options(cfg), input_root=args.input_root, sidecar_path=_sidecar(args.step4_campaign),
        output=args.output, identity=identity, workers=effective, max_tasks_per_worker=args.max_tasks_per_worker)


def jax_stage(*, n: int, args, identity: str, inputs: dict) -> dict:
    from p08_step5_combined import jaxstage
    cfg = config()
    return jaxstage.jax_stage(n=n, cfg=cfg, options=operator_options(cfg), step4=args.step4_campaign,
                              input_root=args.input_root, output=args.output, identity=identity, inputs=inputs,
                              paths=oracle_paths(args.oracle_root, args.input_root))


def reduce_stage(*, n: int, args, identity: str, inputs: dict) -> dict:
    return reduction.reduce_grid(n=n, cfg=config(), output=args.output, identity=identity, inputs=inputs)


def run_grid(*, n: int, args, identity: str, inputs: dict) -> dict:
    """One grid of ``run``: references, JAX stage, reduction (each skipped / resumed when its outputs are valid)."""
    refs = references_stage(n=n, args=args, identity=identity)
    jax_summary = jax_stage(n=n, args=args, identity=identity, inputs=inputs)
    summary = reduce_stage(n=n, args=args, identity=identity, inputs=inputs)
    return {"n": int(n), "status": "complete", "grid_pass": summary["gates"]["grid_pass"],
            "solver_gates_pass": summary["gates"]["solver_gates_pass"], "all_finite": summary["gates"]["all_finite"],
            "references_skipped": bool(refs.get("skipped")), "variants_computed": jax_summary["computed"],
            "variants_resumed": jax_summary["resumed"]}


# ---------------------------------------------------------------------------
# Preflight (N32, bounded)
# ---------------------------------------------------------------------------
def _subset_relative(diff, ref, volume, fallback) -> float | None:
    """L2 of ``diff`` over the owner subset relative to ``fallback`` (the largest reference-term L2)."""
    from p_shared.replay_support import owner_weighted_l2
    l2 = owner_weighted_l2(diff, volume)
    return None if (l2 is None or not fallback) else float(l2 / fallback)


def preflight_checks(*, ref: dict, arrays: dict, info: dict, owners, volume, fs: str, cfg: dict) -> dict:
    """The preflight verdict from the subset references ``ref`` (owners ``owners``) and the full-grid ``arrays`` /
    ``info`` of one variant: finite, solver gates, the subset N - O of psi (sign check), the prescribed-arm sanity at the
    subset owners (against ``sanity_relative_l2_max``) and the arm difference. Pure (unit tested)."""
    from p_shared.replay_support import owner_weighted_l2
    spec = cfg["preflight"]
    owners = np.asarray(owners, dtype=np.int64)
    vol = np.asarray(volume, dtype=np.float64)[owners]
    finite = bool(all(np.all(np.isfinite(v)) for v in ref.values()) and info["gates"]["finite"])
    o_ref = np.asarray(ref[references.psi_key(fs, "O")], dtype=np.float64)
    n_psi = np.asarray(arrays["psi_N"], dtype=np.float64)[owners]
    o_l2 = owner_weighted_l2(o_ref, vol)
    n_minus_o = float(owner_weighted_l2(n_psi - o_ref, vol) / o_l2) if o_l2 else None
    fields = {}
    for field in references.FIELDS:
        scale = max(owner_weighted_l2(np.asarray(ref[references.ref_key(fs, field, t)]), vol)
                    for t in reduction.PRODUCTION_TERMS)
        presc = np.asarray(arrays[checkpoints.arm_key("presc", field, "total")])[owners]
        solved = np.asarray(arrays[checkpoints.arm_key("solved", field, "total")])[owners]
        r_total = np.asarray(ref[references.ref_key(fs, field, "total")])
        fields[field] = {"presc_vs_ref": _subset_relative(presc - r_total, r_total, vol, scale),
                         "solved_vs_ref": _subset_relative(solved - r_total, r_total, vol, scale),
                         "solved_vs_presc": _subset_relative(solved - presc, r_total, vol, scale), "scale_l2": scale}

    def worst(key):
        vals = [f[key] for f in fields.values()]
        return None if any(v is None for v in vals) else max(vals)

    sanity, arm_diff = worst("presc_vs_ref"), worst("solved_vs_presc")
    checks = {"finite": finite, "solver_gates": bool(info["gates_pass"]),
              "n_minus_o_psi": bool(n_minus_o is not None and n_minus_o <= spec["n_minus_o_relative_max"]),
              "prescribed_sanity": bool(sanity is not None and sanity <= spec["sanity_relative_l2_max"]),
              "arm_difference": bool(arm_diff is not None and arm_diff <= spec["arm_difference_relative_max"])}
    return {"checks": checks, "all_pass": bool(all(checks.values())), "n_owners_subset": int(len(owners)),
            "n_minus_o_psi_subset": n_minus_o, "prescribed_sanity_worst": sanity, "arm_difference_worst": arm_diff,
            "fields": fields, "thresholds": dict(spec)}


def preflight_grid(*, n: int, args, identity: str, inputs: dict) -> dict:
    """The bounded gate: reference stage on a small owner subset (the pool-free :func:`references.reference_chunk`) and
    the JAX stage on the FULL N32 plan for one variant (both arms, solver gates). In the solved arm the right-hand side
    is ``A psi_bar + B g`` (the reference stage's ``O(psi)`` exists on the subset only); the subset ``O(psi)`` is
    compared with ``A psi_bar + B g`` (sign / consistency check of the reference)."""
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
    env = build_environment(n=n, input_root=args.input_root, sidecar_path=_sidecar(args.step4_campaign), **options)
    owners = np.asarray(sorted(set(oc.select_owners(env.t, env.census).values())), dtype=np.int64)
    states = {fs: references._prr().p06n_state(env, cfg["field_sets"][fs]["representative"])}
    t0 = time.time()
    ref = references.reference_chunk(env, owners, states=states, params=cfg["params"])
    ref_seconds = time.time() - t0
    prep = jaxstage.prepare(n=n, cfg=cfg, options=options, step4=args.step4_campaign, input_root=args.input_root,
                            paths=oracle_paths(args.oracle_root, args.input_root), inputs=inputs, env=env)
    arrays, vinfo = jaxstage.run_variant(prep, variant, cfg, omega_rhs=None)
    verdict = preflight_checks(ref=ref, arrays=arrays, info=vinfo, owners=owners, volume=prep.owner_volume, fs=fs,
                               cfg=cfg)
    detail = {"n": int(n), "variant": variant, "jax": info, "operator_options": options, "subset_owners": owners,
              "reference_seconds": ref_seconds, "setup_seconds": prep.setup_seconds, "plan": prep.plan_summary,
              "variant_info": vinfo, **verdict, "seconds": time.time() - started}
    write(Path(args.output) / "preflight" / f"N{n}_preflight.json", checkpoints.jsonable(detail))
    return {"n": int(n), "all_pass": verdict["all_pass"], "checks": verdict["checks"], "variant": variant,
            "n_owners_subset": verdict["n_owners_subset"], "n_minus_o_psi_subset": verdict["n_minus_o_psi_subset"],
            "prescribed_sanity_worst": verdict["prescribed_sanity_worst"],
            "arm_difference_worst": verdict["arm_difference_worst"], "solver_gates": vinfo["gates"],
            "consistency_error": vinfo["consistency"]["relative_error_M"], "seconds": time.time() - started}


def _load_preflight(output: Path, identity: str) -> dict | None:
    path = Path(output) / "preflight.json"
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    return saved if saved.get("identity") == identity else None


def preflight(*, args, identity: str, inputs: dict) -> dict:
    output = Path(args.output)
    previous = _load_preflight(output, identity)
    cases = dict(previous["cases"]) if previous else {}
    done = cases.get("32")
    if done is None or not done.get("all_pass"):                 # resumable: a passed preflight is not redone
        cases["32"] = checkpoints.jsonable(preflight_grid(n=32, args=args, identity=identity, inputs=inputs))
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
# Validation and analysis
# ---------------------------------------------------------------------------
def validate(*, args, identity: str, inputs: dict) -> dict:
    """Requires the preflight pass and the reduction of every requested grid (this identity); writes
    ``validation.json``. ``solver_gates_pass`` = every variant's discrete-consistency gate and convergence on every
    grid; ``all_finite`` = every term finite. The order criterion is informational (``acceptance_gate`` is ``None``)."""
    output = Path(args.output)
    _require_preflight(output, identity, what="validate")
    cfg = config()
    grids, results_by_n = {}, {}
    for n in args.resolutions:
        summary, results = reduction.load_grid(output, n, identity)
        results_by_n[int(n)] = results
        grids[str(n)] = {**summary["gates"], "n_owners": summary["n_owners"],
                         "solver": {v: {"consistency_error": rec["info"]["consistency"]["relative_error_M"],
                                        "iterations": rec["info"]["solved_solve"]["iterations"],
                                        "seconds": rec["info"]["solved_solve"]["seconds"],
                                        "converged": rec["info"]["gates"]["converged"]}
                                    for v, rec in summary["variants"].items()}}
    headline = analysis.headline_criterion(results_by_n, cfg)
    payload = {"identity": identity, "operational_complete": True, "resolutions": list(args.resolutions),
               "operator_options": operator_options(cfg), "acceptance_gate": None, "user_decides_acceptance": True,
               "solver_gates_pass": bool(all(g["solver_gates_pass"] for g in grids.values())),
               "all_finite": bool(all(g["all_finite"] for g in grids.values())),
               "grids_pass": bool(all(g["grid_pass"] for g in grids.values())),
               "headline_order_criterion": {k: headline[k] for k in ("informational", "user_decides", "pass",
                                                                      "min_order", "term", "grids", "rows_failed",
                                                                      "rows_undefined")},
               "grids": grids}
    write(output / "validation.json", checkpoints.jsonable(payload))
    return payload


def analyze(*, args, identity: str) -> dict:
    return analysis.analyze(output=args.output, identity=identity, grids=args.resolutions, cfg=config())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("verify-inputs", "preflight", "run", "run-stage", "validate", "analyze"))
    p.add_argument("--step4-campaign", type=Path, required=True,
                   help="the step-4 output's campaign/ folder (artifact/, localized_sidecar.json, validation.json, "
                        "oracle_manifest.json, campaign_manifest.json)")
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--oracle-root", type=Path, default=None,
                   help="where the oracle files of the step-4 oracle manifest live (default: --input-root)")
    p.add_argument("--resolutions", type=int, nargs="+", choices=GRIDS, default=list(GRIDS))
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--n", type=int, choices=GRIDS)
    p.add_argument("--workers", type=int, default=4, help="reference-stage process-pool size (the JAX stage is one "
                                                          "process)")
    p.add_argument("--memory-budget-gib", type=float, default=None)
    p.add_argument("--worker-memory-gib", type=float, default=None)
    p.add_argument("--memory-reserve-gib", type=float, default=1.0)
    p.add_argument("--max-tasks-per-worker", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    args.output = args.output.resolve()
    args.input_root = args.input_root.resolve()
    args.step4_campaign = args.step4_campaign.resolve()
    if args.resolutions != sorted(set(args.resolutions)):
        raise ValueError("resolutions must be unique and ascending")
    with lock(args.output):
        for name in ("logs", "invocations"):
            (args.output / name).mkdir(exist_ok=True, parents=True)
        verify_grids = sorted(set(args.resolutions) | {32}) if args.command == "preflight" else args.resolutions
        identity, inputs = verify_inputs(step4=args.step4_campaign, input_root=args.input_root,
                                         oracle_root=args.oracle_root, output=args.output, grids=verify_grids)
        write(args.output / "invocations" / f"{time.time_ns()}_{args.command}.json",
              {"argv": list(sys.argv if argv is None else argv), "identity": identity, "time": time.time()})
        summary: dict = {"command": args.command, "identity": identity, "status": "complete"}
        ok = True
        if args.command == "preflight":
            payload = preflight(args=args, identity=identity, inputs=inputs)
            summary.update(all_pass=payload["all_pass"], cases=payload["cases"])
            ok = payload["all_pass"]
        elif args.command == "run-stage":
            if args.stage is None or args.n is None:
                raise ValueError("--stage and --n are required for run-stage")
            if args.n not in args.resolutions:
                raise ValueError(f"--n {args.n} is not among the verified --resolutions {args.resolutions}")
            if args.stage == "references":
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
            args.resolutions = [g["n"] for g in done]
            payload = validate(args=args, identity=identity, inputs=inputs)
            analyze(args=args, identity=identity)
            summary.update(operational_complete=len(done) == len(requested),
                           solver_gates_pass=payload["solver_gates_pass"], all_finite=payload["all_finite"])
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
