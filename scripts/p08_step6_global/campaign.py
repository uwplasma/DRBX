#!/usr/bin/env python3
"""P08 step 6: transverse-wave check on the ``compact_c3`` re-freeze artifacts (full grids N32 / N48 / N64), remote campaign.

The P reconstruction switches from a coupled Cartesian quartic to a ringwise fit at ``u = 0.21`` (inner support
``fixed_radius``); per-ring sampling showed a weak band ``u`` 0.12 - 0.21. The catalogue fields of the re-freeze campaign
(``p08_step5_compact_c3``) have little degree >= 4 content there, so this campaign measures the combined perpendicular RHS
(prescribed-phi and solved-phi arms) and the Dirichlet psi solve for NEW manufactured fields with transverse (degree >= 4)
structure around ``u = 0.21``, globally and per region (the P06N region masks of 5.3 plus ``u`` bands), exactly like step
5.3 does for the catalogue fields. No artifact is rebuilt: the re-freeze campaign folder (``--refreeze-campaign``) is read
only (artifacts, provenance, validation, the catalogue summary).

Per grid, resumable, one writer (``runner.lock``): ``owner_values`` (pooled owner averages of the new fields, the frozen
P06N routine) -> ``references`` (pooled, 5.3's reference chunk on the transverse states) -> ``jax`` (one process: both
variants, both arms, psi solve, solver gates; 5.3's ``run_variant``) -> ``reduce`` (5.3's reduction unchanged) ->
``sharding`` (subprocess with 8 forced host devices: eta-sharded RHS and psi solve against the single device, see
:mod:`p08_step6_global.sharding`; ``pass`` / ``fail``, ``pending`` when ``--sharding-max-shards`` capped it). ``validate`` records the 5.3 gates; ``analyze`` writes the 5.3-style report, the u-band tables and the
comparison with the re-freeze catalogue.

Commands: ``verify-inputs``; ``preflight`` (N32, bounded, needs no artifact rows and no build; must pass before ``run``);
``run``; ``run-stage --stage {owner_values,references,jax,reduce,sharding} --n N``; ``validate``; ``analyze``. Run as
``python -m p08_step6_global.campaign`` from ``DRBX/scripts``. See the README.
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
import math
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
from p08_step4_global import campaign as step4                                      # noqa: E402
from p08_step4_global import smoke                                                  # noqa: E402
from p08_step5_combined import checkpoints, reduction                               # noqa: E402
from p08_step5_combined import campaign as step5                                    # noqa: E402
from p08_step5_combined import references as ref5                                   # noqa: E402
from p08_step5_compact_c3 import campaign as c3                                     # noqa: E402
from p08_step5_export import campaign as export                                     # noqa: E402
from p08_step6_global import analysis, fields as tfields, sharding                  # noqa: E402
from p08_step6_global import references as refs6                                    # noqa: E402

GRIDS = (32, 48, 64)
STAGES = ("owner_values", "references", "jax", "reduce", "sharding")
SCHEMA = "drbx.p08-step6-global-v1"
REFREEZE_CAMPAIGN = "p08_step5_compact_c3"
REFREEZE_SCHEMA = "drbx.p08-step5-compact-c3-v1"
#: the operator bundle of the re-freeze artifacts, pinned: ``configuration.json`` must say exactly this, and so must the
#: re-freeze campaign (its ``validation.json``, ``provenance/inputs.json`` and every artifact build identity)
OPERATOR_OPTIONS = {"curvature": "autodiff", "face_quadrature": "q2", "inner_support": "fixed_radius",
                    "bfield_toroidal": "compact_c3"}
PARAMS = {"rho_star": 0.05, "tau": 1.0, "diffusion": {"density": 0.01, "Te": 0.01, "Ti": 0.01, "vorticity": 0.01}}
PHI_SOLVE = {"phi_solve_boundary": "dirichlet", "phi_solve_preconditioner": "block_jacobi_eta_plane",
             "phi_solve_factor_dtype": "float32", "phi_solve_rtol": 1e-11, "phi_solve_restart": 50,
             "phi_solve_max_restarts": 40}
#: the sharding stage settings, pinned (``configuration.json`` must say exactly this)
SHARDING = {"schema": "drbx.p08-step6-sharding-config.v1", "shard_counts": [2, 4, 8], "min_block_planes": 3,
            "devices": 8, "max_shards": None, "plan_halo": 3, "phi_halo": 2, "transverse_variant": "transverse_dirichlet",
            "catalogue_variant": "main_phi_dirichlet",
            "catalogue_kinds": ["neumann", "neumann", "neumann", "dirichlet", "dirichlet"], "rhs_arm": "prescribed_phi",
            "rhs_gate": 1e-12, "phi_iteration_tolerance": 1, "phi_solution_rtol_factor": 10.0}
VARIANTS = ("transverse_dirichlet", "transverse_phi_wave_dirichlet")
FIELD_SETS_OF_VARIANTS = {"transverse_dirichlet": "transverse", "transverse_phi_wave_dirichlet": "transverse_phi_wave"}
U_BAND_EDGES = [0.0, 0.06, 0.12, 0.21, 0.27, 0.40, 1.0]
#: the field definitions and field sets (single source: :mod:`p08_step6_global.fields`; ``configuration.json`` must equal them)
FIELDS = tfields.PINNED_FIELDS
FIELD_SETS = tfields.PINNED_FIELD_SETS
#: the ``localized_sidecar.json`` keys rewritten per machine (everything else must agree with the re-freeze sidecar)
SIDECAR_LOCAL_KEYS = (("metric_cache", "path"), ("makegrid", "path"), ("artifact", "path"))

sha = runner.sha256_file
digest = runner.digest
write = runner.write_json
lock = runner.lock
oracle_paths = step4.oracle_paths
_effective_workers = c3._effective_workers


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def config() -> dict:
    """``configuration.json``, refused if it drifts from the pinned literals above."""
    cfg = json.loads((HERE / "configuration.json").read_text())
    if cfg["schema"] != SCHEMA:
        raise ValueError("unsupported P08 step-6 global campaign contract")
    if cfg["operator_options"] != OPERATOR_OPTIONS:
        raise ValueError(f"configuration.json operator_options {cfg['operator_options']} differ from the pinned "
                         f"re-freeze bundle {OPERATOR_OPTIONS}")
    if cfg["params"] != PARAMS:
        raise ValueError(f"configuration.json params {cfg['params']} differ from the pinned {PARAMS}")
    for key, value in PHI_SOLVE.items():
        if cfg.get(key) != value:
            raise ValueError(f"configuration.json {key}={cfg.get(key)!r} differs from the pinned {value!r}")
    if tuple(cfg["variants"]) != VARIANTS:
        raise ValueError(f"configuration.json variants {tuple(cfg['variants'])} differ from the pinned {VARIANTS}")
    for variant, spec in cfg["variants"].items():
        if spec["field_set"] != FIELD_SETS_OF_VARIANTS[variant] or spec["kinds"] != ["dirichlet"] * 5 or spec["constant"]:
            raise ValueError(f"configuration.json variant {variant!r} differs from the pinned definition")
    if cfg["fields"] != FIELDS:
        raise ValueError("configuration.json fields differ from the pinned definitions (p08_step6_global.fields)")
    if {fs: rec["fields"] for fs, rec in cfg["field_sets"].items()} != FIELD_SETS:
        raise ValueError("configuration.json field_sets differ from the pinned definitions (p08_step6_global.fields)")
    if cfg["u_band_edges"] != U_BAND_EDGES:
        raise ValueError(f"configuration.json u_band_edges {cfg['u_band_edges']} differ from the pinned {U_BAND_EDGES}")
    if cfg["refreeze"]["campaign"] != REFREEZE_CAMPAIGN or cfg["refreeze"]["schema"] != REFREEZE_SCHEMA:
        raise ValueError("configuration.json refreeze block differs from the pinned re-freeze campaign")
    if tuple(cfg["stages"]) != STAGES:
        raise ValueError(f"configuration.json stages {tuple(cfg['stages'])} differ from the pinned {STAGES}")
    if cfg["resolutions"] != list(GRIDS):
        raise ValueError("bad resolutions in configuration.json")
    if {k: v for k, v in cfg["sharding"].items() if k != "note"} != SHARDING:
        raise ValueError(f"configuration.json sharding block {cfg['sharding']} differs from the pinned {SHARDING}")
    if cfg["phi_solve_rtol"] > cfg["phi_solve_rtol_production_default"]:
        raise ValueError("the measurement tolerance must not be looser than the production default")
    return cfg


def operator_options(cfg: dict | None = None) -> dict:
    return dict((cfg or config())["operator_options"])


_OWN_SOURCES = [
    "scripts/p08_step6_global/campaign.py",
    "scripts/p08_step6_global/configuration.json",
    "scripts/p08_step6_global/fields.py",
    "scripts/p08_step6_global/references.py",
    "scripts/p08_step6_global/jaxstage.py",
    "scripts/p08_step6_global/analysis.py",
    "scripts/p08_step6_global/sharding.py",
]
_EXTRA_SOURCES = [
    "scripts/perpendicular_structured/reconstruction.py",
    "scripts/p07_combined_global/kernels.py",
    "scripts/p06n_field_derived_global/fields.py",
    "scripts/p05n_field_derived_global/operator.py",
]
#: this package, the re-freeze package's sources (which include the 5.3, step-4/2/1 and compact_c3 evaluator sources the
#: environment and the stages run on) and the grid-context sources the owner averages read
SOURCE_FILES = list(dict.fromkeys([*_OWN_SOURCES, *c3.SOURCE_FILES, *_EXTRA_SOURCES]))


def source_hashes() -> dict:
    return {p: sha(REPO / p) for p in SOURCE_FILES}


def _commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Folder layout
# ---------------------------------------------------------------------------
def sidecar_path(output) -> Path:
    return Path(output) / "localized_sidecar.json"


def inputs_path(output) -> Path:
    return Path(output) / "provenance" / "inputs.json"


def sharding_path(output, n: int) -> Path:
    return Path(output) / f"N{int(n)}" / "sharding.json"


# ---------------------------------------------------------------------------
# The re-freeze campaign (input) and identity
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    if not Path(path).is_file():
        raise ValueError(f"missing input file: {path}")
    return json.loads(Path(path).read_text())


def _strip_local_paths(sidecar: dict) -> dict:
    out = json.loads(json.dumps(sidecar))
    for outer, inner in SIDECAR_LOCAL_KEYS:
        if isinstance(out.get(outer), dict):
            out[outer].pop(inner, None)
    out.pop("metric_query_batch_size", None)
    return out


def check_artifact_files(gdir: Path, manifest: dict) -> None:
    """The on-disk completeness of an artifact folder (a real run): ``geometry.npz``, ``census.npz`` and every row chunk
    file of the manifest with its recorded size."""
    for name in ("geometry.npz", "census.npz"):
        if not (gdir / name).is_file():
            raise ValueError(f"missing artifact file: {gdir / name}")
    missing = []
    for group, chunks in (manifest.get("chunks") or {}).items():
        for chunk in chunks:
            path = gdir / chunk["file"]
            if not path.is_file() or path.stat().st_size != int(chunk["bytes"]):
                missing.append(str(path))
    if missing:
        raise ValueError(f"{len(missing)} artifact row file(s) missing or of the wrong size, e.g. {missing[0]}")


def read_refreeze(refreeze, *, grids, metadata_only: bool = False) -> dict:
    """Validate the re-freeze campaign folder and return its record. Refuses (``ValueError``) a folder that is not a
    complete ``p08_step5_compact_c3`` campaign, carries other operator options (in ``provenance/inputs.json``,
    ``validation.json`` or an artifact build identity), or whose artifact identities (build identities on disk vs
    ``provenance/inputs.json`` vs ``validation.json``) disagree. For the ``grids`` requested, the artifact folder must be
    complete on disk (``geometry.npz``, ``census.npz``, every row chunk of the manifest) unless ``metadata_only`` (local
    testing against a stripped copy; refused for any command that runs a stage)."""
    refreeze = Path(refreeze)
    if not refreeze.is_dir():
        raise ValueError(f"re-freeze campaign is not a directory: {refreeze}")
    from p08_step2_global import replay

    inputs = _read_json(refreeze / "provenance" / "inputs.json")
    validation = _read_json(refreeze / "validation.json")
    if inputs.get("campaign") != REFREEZE_CAMPAIGN or (inputs.get("configuration") or {}).get("schema") != REFREEZE_SCHEMA:
        raise ValueError(f"{refreeze} is not a {REFREEZE_CAMPAIGN} campaign ({REFREEZE_SCHEMA})")
    for label, options in (("provenance/inputs.json", inputs.get("operator_options")),
                           ("provenance/inputs.json configuration", (inputs.get("configuration") or {}).get("operator_options")),
                           ("validation.json", validation.get("operator_options"))):
        if options != OPERATOR_OPTIONS:
            raise ValueError(f"re-freeze {label} operator options {options} differ from the pinned {OPERATOR_OPTIONS}")
    identity = inputs.get("identity")
    if not identity or validation.get("identity") != identity:
        raise ValueError("re-freeze validation.json and provenance/inputs.json carry different campaign identities")
    if validation.get("operational_complete") is not True:
        raise ValueError("the re-freeze campaign is not operationally complete (validation.json)")
    missing = [n for n in GRIDS if str(n) not in (inputs.get("grids") or {}) or
               str(n) not in (validation.get("artifacts") or {})]
    if missing:
        raise ValueError(f"the re-freeze campaign has no artifact record for the grids {missing}")
    grids_record = {}
    for n in GRIDS:
        rec, vrec = inputs["grids"][str(n)], validation["artifacts"][str(n)]
        if vrec.get("artifact_identity_sha256") != rec.get("artifact_identity_sha256"):
            raise ValueError(f"re-freeze N{n}: validation.json and provenance/inputs.json disagree on the artifact identity")
        policy = vrec.get("policy") or {}
        seen = {"curvature": policy.get("curvature"), "face_quadrature": (policy.get("quadrature") or {}).get("face"),
                "inner_support": policy.get("inner_support"), "bfield_toroidal": policy.get("bfield_toroidal")}
        if seen != OPERATOR_OPTIONS:
            raise ValueError(f"re-freeze N{n}: validation.json records the artifact policy {seen}, not {OPERATOR_OPTIONS}")
        gdir = refreeze / "artifact" / f"N{n}"
        on_disk = n in [int(g) for g in grids]
        if on_disk:
            loaded = replay.load_artifact(refreeze / "artifact", n)          # manifest schema + identity
            smoke.check_artifact_options(loaded["identity"], OPERATOR_OPTIONS)
            if digest(loaded["identity"]) != rec["artifact_identity_sha256"]:
                raise ValueError(f"re-freeze N{n}: the build identity on disk differs from provenance/inputs.json")
            if sha(gdir / "manifest.json") != rec.get("manifest_sha256"):
                raise ValueError(f"re-freeze N{n}: manifest.json differs from provenance/inputs.json")
            if not metadata_only:
                check_artifact_files(gdir, loaded["manifest"])
        grids_record[str(n)] = {"artifact_dir": str(gdir.resolve()), "artifact_identity_sha256": rec["artifact_identity_sha256"],
                                "manifest_sha256": rec.get("manifest_sha256"), "checked_on_disk": on_disk,
                                "files_checked": bool(on_disk and not metadata_only)}
    summary_path = refreeze / "summary" / "step5_combined_summary.json"
    summary = _read_json(summary_path)
    if summary.get("identity") != identity:
        raise ValueError("the re-freeze catalogue summary belongs to another campaign identity")
    sidecar = refreeze / "localized_sidecar.json"
    if not sidecar.is_file():
        raise ValueError(f"missing input file: {sidecar}")
    return {"path": str(refreeze.resolve()), "identity": identity, "inputs_sha256": sha(refreeze / "provenance" / "inputs.json"),
            "validation_sha256": sha(refreeze / "validation.json"), "summary_sha256": sha(summary_path),
            "localized_sidecar": str(sidecar.resolve()), "grids_pass": validation.get("grids_pass"),
            "solver_gates_pass": validation.get("solver_gates_pass"), "grids": grids_record}


def verify_inputs(*, refreeze, input_root: Path, oracle_root: Path | None, output: Path, grids,
                  oracle_grids=None, metadata_only: bool = False) -> tuple[str, dict]:
    """Verify the re-freeze campaign, the immutable inputs (all hashes), the P06N owner-value oracle files of
    ``oracle_grids`` (default ``grids``; the preflight reads only N32) and the backend; localize the sidecar (it must
    agree with the re-freeze one up to the machine-specific paths); record / check ``provenance/inputs.json``. Returns
    ``(identity, record)``. Refuses (raises) if the identity (configuration, sources, input manifest, localized sidecar,
    P06N oracle entries, the re-freeze identity and artifact identities), the oracle root or an artifact identity
    changed relative to a previous run at the same ``output`` (new folder required)."""
    cfg = config()
    options = operator_options(cfg)
    input_root, output = Path(input_root), Path(output)
    if not input_root.is_dir():
        raise ValueError(f"input root is not a directory: {input_root}")
    grids = [int(g) for g in grids]
    oracle_grids = grids if oracle_grids is None else [int(g) for g in oracle_grids]
    rf = read_refreeze(refreeze, grids=grids, metadata_only=metadata_only)
    manifest = step4._input_manifest()
    for rec in manifest["files"]:
        path = input_root / rec["path"]
        if not path.is_file() or path.stat().st_size != rec["bytes"] or sha(path) != rec["sha256"]:
            raise ValueError(f"missing or changed immutable input: {path}")
    info = export.jax_check()
    local = step4.localize_sidecar(input_root, output)
    if _strip_local_paths(json.loads(local.read_text())) != _strip_local_paths(json.loads(Path(rf["localized_sidecar"]).read_text())):
        raise ValueError("the localized sidecar differs from the re-freeze campaign's beyond the machine-specific paths")
    resolved_oracle_root = Path(oracle_root) if oracle_root is not None else input_root
    entries = c3.p06n_owner_value_entries()
    errors = c3.verify_oracle_owner_values(entries, resolved_oracle_root, oracle_grids)
    if errors:
        raise ValueError("oracle verification failed:\n" + "\n".join(errors[:50]) +
                         (f"\n(+{len(errors) - 50} more)" if len(errors) > 50 else ""))
    sources = source_hashes()
    identity = digest({"configuration": cfg, "input_manifest": manifest, "sources": sources,
                       "localized_sidecar_sha256": sha(local), "oracle_p06n_owner_values": entries,
                       "refreeze_identity": rf["identity"],
                       "refreeze_artifacts": {n: rec["artifact_identity_sha256"] for n, rec in rf["grids"].items()}})
    path = inputs_path(output)
    previous = json.loads(path.read_text()) if path.is_file() else None
    if previous is not None:
        if previous["identity"] != identity:
            raise ValueError("campaign identity changed; use a new output folder")
        if previous.get("oracle_root") != str(resolved_oracle_root.resolve()):
            raise ValueError("oracle root changed; use a new output folder")
        for n, rec in rf["grids"].items():
            old = (previous.get("grids") or {}).get(n)
            if old is not None and old["artifact_identity_sha256"] != rec["artifact_identity_sha256"]:
                raise ValueError(f"artifact identity of N{n} changed; use a new output folder")
    record = {"identity": identity, "campaign": "p08_step6_global", "configuration": cfg, "operator_options": options,
              "source_hashes": sources, "commit": _commit(), "input_manifest_files": len(manifest["files"]),
              "localized_sidecar_sha256": sha(local), "oracle_p06n_owner_values": entries,
              "refreeze": {k: v for k, v in rf.items() if k != "grids"}, "grids": rf["grids"],
              "metadata_only": bool(metadata_only), "input_root": str(input_root.resolve()),
              "oracle_root": str(resolved_oracle_root.resolve()), "jax": info, "python": sys.version,
              "platform": platform.platform()}
    write(path, record)
    return identity, record


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _workers(args) -> int:
    return _effective_workers(args.workers, args.memory_budget_gib, args.worker_memory_gib, args.memory_reserve_gib)


def owner_values_stage(*, n: int, args, identity: str) -> dict:
    return refs6.owner_values_stage(n=n, cfg=config(), input_root=args.input_root, output=args.output, identity=identity,
                                    workers=_workers(args), max_tasks_per_worker=args.max_tasks_per_worker)


def references_stage(*, n: int, args, identity: str) -> dict:
    cfg = config()
    return refs6.references_stage(
        n=n, cfg=cfg, options=operator_options(cfg), input_root=args.input_root, sidecar_path=sidecar_path(args.output),
        output=args.output, identity=identity, workers=_workers(args), max_tasks_per_worker=args.max_tasks_per_worker)


def _require_artifact(inputs: dict, n: int) -> None:
    if str(n) not in inputs["grids"]:
        raise ValueError(f"the artifact of N{n} is not recorded in provenance/inputs.json")


def jax_stage(*, n: int, args, identity: str, inputs: dict) -> dict:
    from p08_step6_global import jaxstage
    _require_artifact(inputs, n)
    if inputs.get("metadata_only"):
        raise ValueError("the inputs were verified with --metadata-only; the jax stage needs the artifact rows")
    cfg = config()
    return jaxstage.jax_stage(n=n, cfg=cfg, options=operator_options(cfg),
                              artifact_root=Path(inputs["refreeze"]["path"]) / "artifact",
                              sidecar_path=sidecar_path(args.output), input_root=args.input_root, output=args.output,
                              identity=identity, inputs=inputs)


def reduce_stage(*, n: int, args, identity: str, inputs: dict) -> dict:
    _require_artifact(inputs, n)
    return reduction.reduce_grid(n=n, cfg=config(), output=args.output, identity=identity, inputs=inputs)


def sharding_stage(*, n: int, args, identity: str, inputs: dict) -> dict:
    """The sharding stage: :func:`p08_step6_global.sharding.run`; the record is stored in ``N{n}/sharding.json``."""
    result = sharding.run(n=n, args=args, identity=identity, inputs=inputs, cfg=config())
    status = result.get("status")
    if status not in sharding.KNOWN_STATUSES:
        raise ValueError(f"the sharding stage returned the unknown status {status!r}; known: {sharding.KNOWN_STATUSES}")
    write(sharding_path(args.output, n), checkpoints.jsonable({"identity": identity, "n": int(n), **result}))
    return result


def load_sharding(output, n: int, identity: str) -> dict:
    """The stored sharding record of grid ``n`` (``{"status": "not_implemented"}`` when the hook has not stored one)."""
    path = sharding_path(output, n)
    if not path.is_file():
        return {"status": "not_implemented", "recorded": False}
    saved = json.loads(path.read_text())
    if saved.get("identity") != identity:
        raise ValueError(f"the sharding record of N{n} belongs to another campaign identity")
    return {**saved, "recorded": True}


def run_grid(*, n: int, args, identity: str, inputs: dict) -> dict:
    """One grid of ``run``: owner values, references, JAX stage, reduction, then the sharding hook (each skipped / resumed
    when its outputs are valid; the hook is stored in ``N{n}/sharding.json``)."""
    ov = owner_values_stage(n=n, args=args, identity=identity)
    refs = references_stage(n=n, args=args, identity=identity)
    jax_summary = jax_stage(n=n, args=args, identity=identity, inputs=inputs)
    summary = reduce_stage(n=n, args=args, identity=identity, inputs=inputs)
    shard = sharding_stage(n=n, args=args, identity=identity, inputs=inputs)
    return {"n": int(n), "status": "complete", "grid_pass": summary["gates"]["grid_pass"],
            "solver_gates_pass": summary["gates"]["solver_gates_pass"], "all_finite": summary["gates"]["all_finite"],
            "owner_values_skipped": bool(ov.get("skipped")), "references_skipped": bool(refs.get("skipped")),
            "variants_computed": jax_summary["computed"], "variants_resumed": jax_summary["resumed"],
            "sharding_status": shard.get("status")}


# ---------------------------------------------------------------------------
# Preflight (N32, bounded; no artifact rows, no build)
# ---------------------------------------------------------------------------
def _finite_nonzero(arrays: dict) -> tuple:
    finite = bool(all(np.all(np.isfinite(v)) for v in arrays.values()))
    zero = sorted(k for k, v in arrays.items() if not np.any(np.asarray(v) != 0.0))
    return finite, zero


def preflight_grid(*, n: int, input_root: Path, sidecar: Path, output: Path, paths: dict) -> dict:
    """The bounded gate of grid ``n`` (N32): the environment with the four pinned options (``compact_c3`` evaluator
    verified); the owner averages of the new fields (chunked routine vs the one-shot ``live_observations`` formula, bit
    for bit; vs an independent exactly-rounded per-owner sum on the 12 closure owners; and the same routine applied to
    the frozen P06N catalogue reproduces the committed P06N ``owner_values`` file); the references of both field sets on
    the 12 closure owners (finite, non-zero; the independent production-formula check of 5.3's references); the Hessians
    and gradients against central differences (random points, the support points of the closure owners); the axis
    smoothness (finite derivatives and a theta-independent value at ``u -> 0``); and the u-band partition. Needs no
    artifact."""
    from p_shared import owner_closure as oc
    from p_shared import perpendicular_reference_rhs as prr
    from p_shared.replay_support import build_environment
    import p06n_field_derived_global.core as p06n_core

    cfg = config()
    options = operator_options(cfg)
    spec = cfg["preflight"]
    info = export.jax_check()
    started = time.time()
    env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar), **options)
    provenance = dict(getattr(env.ref, "provenance", None) or {})
    evaluator = getattr(getattr(env.ref, "bfield_evaluator", None), "toroidal_method", None)
    env_bfield = getattr(env, "bfield_toroidal", None)
    environment_seconds = time.time() - started
    t = env.t
    period = float(t.g.eta_period)
    checks: dict = {}
    detail: dict = {}
    checks["bfield_compact_c3"] = bool(provenance.get("bfield_toroidal") == options["bfield_toroidal"]
                                       and evaluator == options["bfield_toroidal"] and env_bfield == options["bfield_toroidal"])
    detail["bfield"] = {"env_ref_provenance_bfield_toroidal": provenance.get("bfield_toroidal"),
                        "evaluator_toroidal_method": evaluator, "env_bfield_toroidal": env_bfield,
                        "environment_seconds": environment_seconds}
    checks["eta_period_2pi"] = bool(abs(period - 2.0 * math.pi) < 1e-12)
    detail["eta_period"] = period
    columns = refs6.columns_from_config(cfg, period)
    owners = np.asarray(sorted(set(oc.select_owners(t, env.census).values())), dtype=np.int64)
    detail["subset_owners"] = owners

    # 1. owner averages -------------------------------------------------------------------------------------------
    t0 = time.time()
    n_owners = int(len(t.vol))
    chunk = int(cfg["owner_chunk_size"])
    chunked = np.concatenate([tfields.owner_average_chunk(t, columns.values, s, min(s + chunk, n_owners))
                              for s in range(0, n_owners, chunk)])
    full = tfields.owner_average(t, columns.values)
    exact = tfields.owner_average_exact_sum(t, columns.values, owners)
    multi = np.flatnonzero(np.bincount(t.ro, minlength=n_owners) > 1)
    multi = multi[:: max(1, len(multi) // 8)][:8]
    exact_multi = tfields.owner_average_exact_sum(t, columns.values, multi)
    d_chunk = float(np.max(np.abs(chunked - full)))
    d_exact = float(max(np.max(np.abs(exact - full[owners])), np.max(np.abs(exact_multi - full[multi]))))
    tol_avg = float(spec["owner_average_tolerance"])
    checks["owner_average_chunked_bitwise"] = bool(d_chunk == 0.0)
    checks["owner_average_vs_exact_sum"] = bool(d_exact <= tol_avg)
    checks["owner_average_finite"] = bool(np.all(np.isfinite(chunked)))
    # the same routine on the frozen P06N catalogue vs the committed owner_values file
    frozen_fn = lambda pts: np.column_stack([p06n_core.p06n_fields.evaluate(env.ref, pts, name, period)[0]
                                             for name in p06n_core.NAMES])                    # noqa: E731
    mine_frozen = np.concatenate([tfields.owner_average_chunk(t, frozen_fn, s, min(s + chunk, n_owners))
                                  for s in range(0, n_owners, chunk)])
    oracle_file = Path(paths["p05n_p06n_upwind"]) / "p06n" / f"N{n}.owner_values.npz"
    with np.load(oracle_file, allow_pickle=False) as z:
        oracle = z["values"].copy()
    d_oracle = float(np.max(np.abs(mine_frozen - oracle))) if oracle.shape == mine_frozen.shape else float("inf")
    checks["owner_average_reproduces_frozen_p06n"] = bool(d_oracle <= float(spec["oracle_owner_values_tolerance"]))
    detail["owner_average"] = {"chunked_vs_oneshot_max_abs": d_chunk, "vs_exact_sum_max_abs": d_exact,
                               "exact_sum_owners": len(owners) + len(multi), "multi_cell_owners_checked": len(multi),
                               "p06n_reproduction_max_abs": d_oracle, "p06n_oracle_file": str(oracle_file),
                               "column_range": {name: [float(chunked[:, i].min()), float(chunked[:, i].max())]
                                                for i, name in enumerate(columns.names)},
                               "seconds": time.time() - t0}

    # 2. references on the 12 closure owners ------------------------------------------------------------------------
    t0 = time.time()
    states = columns.states()
    ref = ref5.reference_chunk(env, owners, states=states, params=cfg["params"])
    finite, zero = _finite_nonzero({k: v for k, v in ref.items() if k != "owners"})
    checks["references_finite"] = finite
    totals = {fs: {field: float(np.linalg.norm(ref[ref5.ref_key(fs, field, "total")])) for field in ref5.FIELDS}
              for fs in states}
    psi = {fs: {"O": float(np.linalg.norm(ref[ref5.psi_key(fs, "O")])),
                "R_mid": float(np.linalg.norm(ref[ref5.psi_key(fs, "R_mid")])),
                "O_minus_R_mid_relative": float(np.linalg.norm(ref[ref5.psi_key(fs, "O")] - ref[ref5.psi_key(fs, "R_mid")])
                                                / max(np.linalg.norm(ref[ref5.psi_key(fs, "R_mid")]), 1e-300))}
           for fs in states}
    checks["references_nonzero"] = bool(all(v > 0.0 for fs in totals.values() for v in fs.values()) and
                                        all(r["O"] > 0.0 and r["R_mid"] > 0.0 for r in psi.values()))
    params = prr.ReferenceParams(rho_star=float(cfg["params"]["rho_star"]), tau=float(cfg["params"]["tau"]),
                                 diffusion={f: float(cfg["params"]["diffusion"][f]) for f in ref5.FIELDS})
    formula = {fs: prr.production_formula_check(env, owners, state, params)["max_rel"] for fs, state in states.items()}
    checks["production_formula_check"] = bool(all(v <= float(spec["production_formula_max_rel"]) for v in formula.values()))
    detail["references"] = {"owners": len(owners), "exactly_zero_arrays": zero, "total_l2": totals, "psi": psi,
                            "production_formula_max_rel": formula, "seconds": time.time() - t0}

    # 3. Hessian / gradient finite-difference checks -------------------------------------------------------------
    t0 = time.time()
    rng = np.random.default_rng(int(spec["seed"]))
    random_points = np.column_stack((rng.uniform(0.005, 0.995, int(spec["random_points"])),
                                     rng.uniform(0.0, 2.0 * np.pi, int(spec["random_points"])),
                                     rng.uniform(0.0, 2.0 * np.pi, int(spec["random_points"]))))
    support = prr.owner_support(env, owners)
    support_points = support.points[:: max(1, len(support.points) // 256)]
    band_points = np.column_stack((rng.uniform(0.10, 0.30, 64), rng.uniform(0.0, 2.0 * np.pi, 64),
                                   rng.uniform(0.0, 2.0 * np.pi, 64)))
    fd = {}
    for fs, state in states.items():
        fd[fs] = {label: tfields.fd_defects(state, pts, float(spec["fd_step"]))
                  for label, pts in (("random", random_points), ("closure_support", support_points), ("switch_band", band_points))}
    worst_h = max(d["hessian_fd_rel"] for fs in fd.values() for d in fs.values())
    worst_g = max(d["gradient_fd_rel"] for fs in fd.values() for d in fs.values())
    worst_asym = max(d["hessian_asymmetry"] for fs in fd.values() for d in fs.values())
    checks["hessian_vs_fd_gradients"] = bool(worst_h <= float(spec["hessian_fd_relative_max"]))
    checks["gradient_vs_fd_values"] = bool(worst_g <= float(spec["gradient_fd_relative_max"]))
    checks["hessian_symmetric"] = bool(worst_asym <= 1e-12)
    detail["derivatives"] = {"fd_step": float(spec["fd_step"]), "worst_hessian_fd_rel": worst_h,
                             "worst_gradient_fd_rel": worst_g, "worst_hessian_asymmetry": worst_asym, "cases": fd,
                             "seconds": time.time() - t0}

    # 4. axis smoothness ------------------------------------------------------------------------------------------
    theta = np.linspace(0.0, 2.0 * np.pi, 9)[:-1]
    eta = np.array([0.0, 1.3, 4.1])
    axis = {}
    axis_ok = True
    for fs, state in states.items():
        worst_spread_excess, worst_ratio, finite_all = 0.0, 0.0, True
        for u in spec["axis_u"]:
            pts = np.array([[u, th, e] for th in theta for e in eta])
            v, g, h = state.values_gradients_hessians(pts)
            finite_all &= bool(np.all(np.isfinite(v)) and np.all(np.isfinite(g)) and np.all(np.isfinite(h)))
            # the fields are smooth functions of (x, y): the value at radius u varies with theta by at most O(u)
            # (exactly 0 at u = 0) and d_theta f = u * (tangential derivative) is O(u)
            vv = v.reshape(v.shape[0], len(theta), len(eta))
            spread = float(np.max(np.abs(vv - vv[:, :1, :])))
            worst_spread_excess = max(worst_spread_excess, spread - 20.0 * u)
            worst_ratio = max(worst_ratio, float(np.max(np.abs(g[:, :, 1])) - 10.0 * u))
        axis[fs] = {"finite": finite_all, "value_theta_spread_excess_over_20u": worst_spread_excess,
                    "theta_derivative_excess_over_10u": worst_ratio}
        axis_ok &= bool(finite_all and worst_spread_excess <= float(spec["axis_value_tolerance"]) and worst_ratio <= 1e-9)
    checks["axis_smooth"] = bool(axis_ok)
    detail["axis"] = {"u": spec["axis_u"], "field_sets": axis}

    # 5. u-band partition on this grid ------------------------------------------------------------------------------
    u_owner = refs6.owner_u(t)
    bands = refs6.band_masks(u_owner, cfg["u_band_edges"])
    counts = {name: int(mask.sum()) for name, mask in bands.items()}
    checks["u_bands_partition_owners"] = bool(sum(counts.values()) == n_owners and
                                              np.all(sum(m.astype(int) for m in bands.values()) == 1) and
                                              all(c > 0 for c in counts.values()))
    detail["u_bands"] = {"edges": cfg["u_band_edges"], "owner_counts": counts}

    all_pass = bool(all(checks.values()))
    out = {"n": int(n), "all_pass": all_pass, "checks": checks, "operator_options": options, "jax": info,
           "peak_rss_gib": runner.peak_rss_gib(), "rss_method": "ru_maxrss", "seconds": time.time() - started, **detail}
    write(Path(output) / "preflight" / f"N{n}_transverse.json", checkpoints.jsonable(out))
    return out


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
# Validation and analysis
# ---------------------------------------------------------------------------
def validate(*, args, identity: str, inputs: dict) -> dict:
    """Requires the preflight pass, the reduction of every requested grid (this identity and the re-freeze artifact it was
    computed on) and records the 5.3 gates per variant: ``solver_gates_pass`` = every variant's discrete-consistency gate
    and convergence on every grid; ``all_finite`` = every term finite; ``grids_pass`` adds nothing but a failing sharding
    record. The sharding hook is reported per grid; ``not_implemented`` / ``pending`` is recorded as pending and does not
    fail anything. The order criterion is informational (``acceptance_gate`` is ``None``)."""
    output = Path(args.output)
    _require_preflight(output, identity, what="validate")
    cfg = config()
    grids, results_by_n, shard = {}, {}, {}
    for n in args.resolutions:
        _require_artifact(inputs, n)
        summary, results = reduction.load_grid(output, n, identity)
        if summary.get("jax_identity") != checkpoints.jax_identity(identity, n, checkpoints.artifact_sha(inputs, n)):
            raise ValueError(f"the reduction of N{n} was computed on another artifact")
        results_by_n[int(n)] = results
        shard[str(n)] = load_sharding(output, n, identity)
        grids[str(n)] = {**summary["gates"], "n_owners": summary["n_owners"],
                         "artifact_identity_sha256": checkpoints.artifact_sha(inputs, n),
                         "solver": {v: {"consistency_error": rec["info"]["consistency"]["relative_error_M"],
                                        "iterations": rec["info"]["solved_solve"]["iterations"],
                                        "seconds": rec["info"]["solved_solve"]["seconds"],
                                        "converged": rec["info"]["gates"]["converged"]}
                                    for v, rec in summary["variants"].items()}}
    statuses = {n: rec["status"] for n, rec in shard.items()}
    if any(s in sharding.FAIL_STATUSES for s in statuses.values()):
        sharding_status = "fail"
    elif all(s in sharding.PASS_STATUSES for s in statuses.values()):
        sharding_status = "pass"
    else:
        sharding_status = "pending"
    headline = analysis.ana5.headline_criterion(results_by_n, cfg)
    payload = {"identity": identity, "operational_complete": True, "resolutions": list(args.resolutions),
               "operator_options": operator_options(cfg), "acceptance_gate": None, "user_decides_acceptance": True,
               "preflight_pass": True,
               "solver_gates_pass": bool(all(g["solver_gates_pass"] for g in grids.values())),
               "all_finite": bool(all(g["all_finite"] for g in grids.values())),
               "grids_pass": bool(all(g["grid_pass"] for g in grids.values()) and sharding_status != "fail"),
               "sharding": {"status": sharding_status, "grids": shard,
                            "pending": sharding_status == "pending"},
               "stages_pending": ["sharding"] if sharding_status == "pending" else [],
               "headline_order_criterion": {k: headline[k] for k in ("informational", "user_decides", "pass", "min_order",
                                                                      "term", "grids", "rows_failed", "rows_undefined")},
               "refreeze": {k: inputs["refreeze"][k] for k in ("path", "identity", "grids_pass", "solver_gates_pass")},
               "fields": cfg["fields"], "field_sets": cfg["field_sets"], "grids": grids}
    write(output / "validation.json", checkpoints.jsonable(payload))
    return payload


def analyze(*, args, identity: str, inputs: dict) -> dict:
    refreeze = Path(inputs["refreeze"]["path"])
    catalogue = analysis.read_catalogue_summary(refreeze, inputs["refreeze"]["identity"])
    return analysis.analyze(output=args.output, identity=identity, grids=args.resolutions, cfg=config(),
                            catalogue=catalogue)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("verify-inputs", "preflight", "run", "run-stage", "validate", "analyze"))
    p.add_argument("--refreeze-campaign", type=Path, required=True,
                   help="the completed p08_step5_compact_c3 campaign folder (artifact/N{n}, provenance/inputs.json, "
                        "validation.json, localized_sidecar.json, summary/); read only")
    p.add_argument("--input-root", type=Path, required=True,
                   help="the immutable HSX input root holding the files of p08_step1_global/input_manifest.json")
    p.add_argument("--output", type=Path, required=True, help="the campaign folder (new; identity-checked)")
    p.add_argument("--oracle-root", type=Path, default=None,
                   help="where the P06N oracle files of the committed oracle manifest live (default: --input-root)")
    p.add_argument("--resolutions", type=int, nargs="+", choices=GRIDS, default=list(GRIDS))
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--n", type=int, choices=GRIDS)
    p.add_argument("--workers", type=int, default=4, help="process-pool size of the owner_values and references stages "
                                                          "(the JAX stage is one process)")
    p.add_argument("--memory-budget-gib", type=float, default=None)
    p.add_argument("--worker-memory-gib", type=float, default=None)
    p.add_argument("--memory-reserve-gib", type=float, default=1.0)
    p.add_argument("--max-tasks-per-worker", type=int, default=None)
    p.add_argument("--sharding-max-shards", type=int, default=None,
                   help="cap the shard counts of the sharding stage (e.g. 4 to skip Sz = 8 where memory is short); a capped "
                        "run that passes is recorded as pending and is redone without the cap")
    p.add_argument("--metadata-only", action="store_true",
                   help="LOCAL TESTING ONLY (verify-inputs, preflight): do not require the artifact rows and geometry of "
                        "the re-freeze campaign on disk (a stripped copy); refused for every other command")
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    args.output = args.output.resolve()
    args.input_root = args.input_root.resolve()
    if args.resolutions != sorted(set(args.resolutions)):
        raise ValueError("resolutions must be unique and ascending")
    if args.metadata_only and args.command not in ("verify-inputs", "preflight"):
        raise ValueError("--metadata-only is only allowed for verify-inputs and preflight (the stages need the artifact rows)")
    with lock(args.output):
        for name in ("logs", "invocations"):
            (args.output / name).mkdir(exist_ok=True, parents=True)
        verify_grids = sorted(set(args.resolutions) | {32}) if args.command == "preflight" else args.resolutions
        oracle_grids = [32] if args.command == "preflight" else (args.resolutions if args.command == "verify-inputs" else [])
        identity, inputs = verify_inputs(refreeze=args.refreeze_campaign, input_root=args.input_root,
                                         oracle_root=args.oracle_root, output=args.output, grids=verify_grids,
                                         oracle_grids=oracle_grids, metadata_only=args.metadata_only)
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
            if args.stage == "owner_values":
                result = owner_values_stage(n=args.n, args=args, identity=identity)
                summary.update(stage="owner_values", n=args.n, n_owners=result["n_owners"], skipped=result["skipped"])
            elif args.stage == "references":
                result = references_stage(n=args.n, args=args, identity=identity)
                summary.update(stage="references", n=args.n, n_owners=result["n_owners"], skipped=result["skipped"])
            elif args.stage == "jax":
                result = jax_stage(n=args.n, args=args, identity=identity, inputs=inputs)
                summary.update(stage="jax", n=args.n, computed=result["computed"], resumed=result["resumed"])
            elif args.stage == "reduce":
                result = reduce_stage(n=args.n, args=args, identity=identity, inputs=inputs)
                summary.update(stage="reduce", n=args.n, gates=result["gates"])
                ok = result["gates"]["grid_pass"]
            else:
                result = sharding_stage(n=args.n, args=args, identity=identity, inputs=inputs)
                summary.update(stage="sharding", n=args.n, sharding_status=result.get("status"))
                ok = result.get("status") not in sharding.FAIL_STATUSES
        elif args.command == "run":
            _require_preflight(args.output, identity, what="run")
            done, requested = [], list(args.resolutions)
            for n in requested:
                grid = run_grid(n=n, args=args, identity=identity, inputs=inputs)
                done.append(grid)
                if not grid["grid_pass"] or grid["sharding_status"] in sharding.FAIL_STATUSES:
                    ok = False
                    break                                       # the data is checkpointed; do not burn the finer grids
            summary["grids"] = done
            completed = [g["n"] for g in done if g["status"] == "complete"]
            if completed:
                args.resolutions = completed
                payload = validate(args=args, identity=identity, inputs=inputs)
                analyze(args=args, identity=identity, inputs=inputs)
                summary.update(solver_gates_pass=payload["solver_gates_pass"], all_finite=payload["all_finite"],
                               sharding_status=payload["sharding"]["status"])
            summary.update(operational_complete=len(completed) == len(requested))
        elif args.command == "validate":
            payload = validate(args=args, identity=identity, inputs=inputs)
            summary.update(solver_gates_pass=payload["solver_gates_pass"], all_finite=payload["all_finite"],
                           headline_order_pass=payload["headline_order_criterion"]["pass"],
                           sharding_status=payload["sharding"]["status"])
            ok = payload["grids_pass"]
        elif args.command == "analyze":
            payload = analyze(args=args, identity=identity, inputs=inputs)
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
