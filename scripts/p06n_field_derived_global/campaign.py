#!/usr/bin/env python3
"""Resumable node-local CPU campaign for the P06N global static qualification
of the accepted P06 curvature operator under physical-normal Neumann rows.

Mirrors scripts/p05n_field_derived_global/campaign.py's CLI, resumability
(chunk NPZ + JSON receipts bound to a campaign identity digest of
configuration + source hashes + input manifest + commit) and stage structure,
with three stages: 'observations' (owner-averaged live fields), 'raw' (the q1
material/remainder volume term and the exact reference R, at raw midpoints),
and 'faces' (the q3 characteristic face correction, in the accepted P06 face
index space -- see core.py's module docstring).

Do not put this package's own directory first on sys.path (it contains
operator.py, which would shadow the stdlib operator module). Put
DRBX/scripts on sys.path and run this as
`python -m p06n_field_derived_global.campaign` from DRBX/scripts.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import fcntl
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from multiprocessing import get_context
from contextlib import contextmanager

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]              # .../DRBX

from p06n_field_derived_global import core
from p06n_field_derived_global import preflight as p06n_preflight

STAGES = ("observations", "raw", "faces")
ZERO_JUMP_KINDS = ("physical_wall", "radial_n_minus_1", "transverse_last_two_layers")
STATE = {}


# ---------------------------------------------------------------------------
# Small persistence helpers (mirrors p05n_field_derived_global/campaign.py).
# ---------------------------------------------------------------------------
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def encode(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=encode, allow_nan=False) + "\n")
    tmp.replace(path)


def save(path, **data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **data)
    tmp.replace(path)


def rss():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)


@contextmanager
def lock(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output / ".campaign.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def config():
    cfg = json.loads((HERE / "configuration.json").read_text())
    if cfg["schema"] != "drbx.p06n-field-derived-static-global-v1" or cfg["resolutions"] != [32, 48, 64]:
        raise ValueError("unsupported frozen P06N field-derived contract")
    if cfg["physical_fields"] != list(core.NAMES):
        raise ValueError("physical field catalogue changed")
    if cfg["catalogue_reference"] != "p06n_catalogue.json" or sha(HERE / cfg["catalogue_reference"]) != cfg["catalogue_sha256"]:
        raise ValueError("frozen catalogue file changed")
    if cfg["cases"] != list(core.CASE_NAMES) or cfg["gated_cases"] != list(core.GATED_CASES) or \
       cfg["heldout_cases"] != list(core.HELDOUT_CASES) or cfg["control_cases"] != list(core.CONTROL_CASES):
        raise ValueError("case catalogue scope changed")
    return cfg


def _fixture_files():
    rel = ["scripts/p06n_field_derived_global/preflight_fixtures/fixtures_manifest.json",
           "scripts/p06n_field_derived_global/preflight_fixtures/extract.py"]
    for n in (32, 48, 64):
        rel.append(f"scripts/p06n_field_derived_global/preflight_fixtures/N{n}.selection.json")
        rel.append(f"scripts/p06n_field_derived_global/preflight_fixtures/N{n}.replay.npz")
    return rel


def source_hashes():
    rel = ["scripts/p06n_field_derived_global/campaign.py", "scripts/p06n_field_derived_global/core.py",
           "scripts/p06n_field_derived_global/fields.py", "scripts/p06n_field_derived_global/rows.py",
           "scripts/p06n_field_derived_global/operator.py", "scripts/p06n_field_derived_global/preflight.py",
           "scripts/p06n_field_derived_global/configuration.json",
           "scripts/p06n_field_derived_global/input_manifest.json",
           "scripts/p06n_field_derived_global/p06n_catalogue.json",
           "scripts/p06n_field_derived_global/README.md",
           "scripts/p05n_field_derived_global/fields.py", "scripts/p05n_field_derived_global/core.py",
           "scripts/p05n_field_derived_global/operator.py",
           "scripts/p06_structured_global/numerics.py",
           "scripts/p07_combined_global/kernels.py", "scripts/p07_combined_global/topology.py",
           "scripts/p07_diffusion_global/numerics.py",
           "scripts/perpendicular_structured/reconstruction.py",
           "scripts/perpendicular_structured/reference_geometry.py",
           "src/drbx/geometry/fci_perpendicular_neumann_trace.py",
           "src/drbx/geometry/fci_perpendicular_reconstruction.py",
           "src/drbx/native/fci_perpendicular_face_corrections.py",
           "src/drbx/native/fci_curvature_production_flux.py",
           "src/drbx/native/fci_operators.py"] + _fixture_files()
    return {p: sha(REPO / p) for p in rel}


def localize_sidecar(input_root, output):
    """Same localized reference sidecar p05n/p07n use (identical underlying
    continuum reference geometry); see p05n_field_derived_global/campaign.py's
    localize_sidecar for the equivalence note."""
    input_root = Path(input_root); output = Path(output)
    side = json.loads((input_root / "DRBX/work/perpendicular_second_order_hsx_p01_p03/continuous_reference_sidecar.json").read_text())
    side["metric_cache"]["path"] = str((input_root / "hsx_metric_d58d392545fd3917efeb83b6.npz").resolve())
    side["makegrid"]["path"] = str((input_root / "mgrid_res2p5cm_180pln.nc").resolve())
    side["artifact"]["path"] = str((input_root / "prototype_runs/geometry/hsx_fci_64x64x64").resolve())
    side["metric_query_batch_size"] = 4096
    local = output / "reference_sidecar.json"
    if local.exists():
        if json.loads(local.read_text()) != side:
            raise ValueError("localized reference sidecar changed")
    else:
        write(local, side)
    return local


def verify(input_root, output):
    input_root = Path(input_root); output = Path(output)
    cfg = config()
    im = json.loads((HERE / "input_manifest.json").read_text())
    for rec in im["files"]:
        path = input_root / rec["path"]
        if not path.is_file() or path.stat().st_size != rec["bytes"] or sha(path) != rec["sha256"]:
            raise ValueError(f"missing or changed immutable input: {path}")
    sources = source_hashes()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    identity = digest({"configuration": cfg, "input_manifest": im, "sources": sources, "commit": commit,
                        "catalogue": core.CASES})
    manifest_path = output / "campaign_manifest.json"
    local = localize_sidecar(input_root, output)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["identity"] != identity:
            raise ValueError("campaign identity changed; use a new output folder")
        if saved["localized_sidecar_sha256"] != sha(local):
            raise ValueError("localized sidecar hash changed")
    else:
        import jax
        if jax.default_backend() != "cpu":
            raise ValueError("JAX CPU backend required")
        write(manifest_path, {"identity": identity, "configuration": cfg, "inputs": im, "source_hashes": sources,
                               "commit": commit, "input_root": str(input_root.resolve()),
                               "localized_sidecar_sha256": sha(local),
                               "python": sys.version, "platform": platform.platform(),
                               "jax_backend": jax.default_backend()})
    for name in ("chunks", "scratch", "cache", "logs", "invocations", "executions"):
        (output / name).mkdir(exist_ok=True)
    return identity


# ---------------------------------------------------------------------------
# Chunk id ranges (no separate topology stage: both raw-cell and accepted
# face-id ranges are pure, cheap functions of n; see core.all_face_ids).
# ---------------------------------------------------------------------------
def ids_for(n, stage):
    if stage == "faces":
        return core.all_face_ids(n)
    return np.arange(n ** 3, dtype=np.int64)


def effective(args):
    if args.workers is None or args.workers < 1:
        raise ValueError("--workers must be positive for computational stages")
    if args.worker_memory_gib is None or args.worker_memory_gib <= 0 or args.memory_budget_gib is None or args.memory_budget_gib <= 0:
        raise ValueError("--worker-memory-gib and --memory-budget-gib required")
    count = min(args.workers, int((args.memory_budget_gib - args.memory_reserve_gib) // args.worker_memory_gib))
    if count < 1:
        raise ValueError("memory budget cannot fit one worker plus reserve")
    return count


def plan(output, identity, resolutions):
    cfg = config()
    chunks = {"observations": cfg["observation_chunk"], "raw": cfg["raw_chunk"], "faces": cfg["face_chunk"]}
    data = {"identity": identity, "resolutions": resolutions, "stages": {}}
    for stage in STAGES:
        data["stages"][stage] = {}
        for n in resolutions:
            ids = ids_for(n, stage); size = chunks[stage]
            data["stages"][stage][str(n)] = [{"stage": stage, "n": n, "start": j, "stop": min(j + size, len(ids))}
                                              for j in range(0, len(ids), size)]
    path = Path(output) / "plan.json"
    if path.exists() and json.loads(path.read_text()) != data:
        raise ValueError("plan/configuration changed")
    write(path, data)
    return data


# ---------------------------------------------------------------------------
# Chunk execution.
# ---------------------------------------------------------------------------
def unit_path(output, unit):
    return Path(output) / "chunks" / f"N{unit['n']}" / unit["stage"] / f"{unit['start']:07d}-{unit['stop'] - 1:07d}.npz"


def unit_ids(unit):
    return ids_for(unit["n"], unit["stage"])[unit["start"]:unit["stop"]]


def valid_unit(output, unit, identity):
    path = unit_path(output, unit); receipt = path.with_suffix(".json")
    if not path.exists() and not receipt.exists():
        return False
    if not path.exists() or not receipt.exists():
        return False
    item = json.loads(receipt.read_text())
    if item["identity"] != identity or item["unit"] != unit:
        raise ValueError(f"stale checkpoint {path}")
    if sha(path) != item["sha256"]:
        raise ValueError(f"corrupt checkpoint {path}")
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z["ids"], unit_ids(unit)):
            raise ValueError(f"wrong chunk IDs {path}")
        for name in z.files:
            if z[name].dtype.kind in "fc" and not np.isfinite(z[name]).all():
                raise ValueError(f"nonfinite {path}:{name}")
        expected = {"observations": ("numerator", "owner_ids"),
                    "raw": ("material", "remainder", "total", "R_material", "R_remainder", "R_total",
                            "evolution_weight", "owner_ids"),
                    "faces": ("keys", "lo", "hi", "correction_lo", "correction_hi")}[unit["stage"]]
        if any(key not in z for key in expected):
            raise ValueError(f"incomplete chunk {path}")
    return True


def initialize(input_root, output, n, stage, identity, worker_memory_gib):
    global STATE
    output = Path(output)
    os.environ["XDG_CACHE_HOME"] = str(output / "cache")
    os.environ["DRBX_CACHE_DIR"] = str(output / "cache/jax")
    os.environ["TMPDIR"] = str(output / "scratch")
    import jax
    if jax.default_backend() != "cpu":
        raise ValueError("JAX CPU backend required in worker")
    t, ref = core.load(input_root, output / "reference_sidecar.json", n)
    S = core.StructuredReconstruction(t)
    ctx = core.context(t)
    period = t.g.eta_period
    lattice = core.WallLattice(t, ref, period)
    normal_coefficients = lattice.normal  # one bound object: patch_cache keys on its id
    owner_values = None
    if stage in ("raw", "faces"):
        path = output / f"N{n}.owner_values.npz"
        record = json.loads((output / f"N{n}.observations.reduction.json").read_text())
        if sha(path) != record["owner_values_sha256"]:
            raise ValueError("owner values hash changed")
        with np.load(path, allow_pickle=False) as z:
            owner_values = z["values"].copy()
    STATE = {"t": t, "ref": ref, "S": S, "ctx": ctx, "normal_coefficients": normal_coefficients,
             "normal_data_fn": lattice.normal_data, "owner_values": owner_values, "stage": stage,
             "output": output, "identity": identity, "n": n, "worker_memory_gib": worker_memory_gib,
             "patch_cache": {}}


def compute(unit):
    s = STATE; ids = unit_ids(unit); stage = s["stage"]; started = time.monotonic()
    period = s["t"].g.eta_period
    cfg = config()
    if stage == "observations":
        data = core.observation_chunk(s["t"], s["ref"], ids, period)
    elif stage == "raw":
        rc = core.raw_chunk(s["t"], s["S"], s["ref"], s["ctx"], ids, s["owner_values"], s["normal_coefficients"],
                             s["patch_cache"], period, normal_data_override=s["normal_data_fn"])
        data = {k: v for k, v in rc.items() if k != "variants"}
        if len(s["patch_cache"]) > cfg["patch_cache_limit"]:
            s["patch_cache"].clear()
    else:
        keys = core.face_keys_for_ids(s["n"], ids)
        fc = core.face_chunk(s["t"], s["S"], s["ref"], s["ctx"], keys, s["owner_values"], s["normal_coefficients"],
                              s["patch_cache"], period, order=cfg["face_quadrature"],
                              normal_data_override=s["normal_data_fn"], dedupe=True)
        data = {"keys": fc["keys"], "lo": fc["lo"], "hi": fc["hi"], "correction_lo": fc["correction_lo"],
                "correction_hi": fc["correction_hi"], "condition_max": fc["condition_max"],
                "residual_max": fc["residual_max"], "wall_exterior_defect_max": fc["wall_exterior_defect_max"],
                "wall_faces_seen": fc["wall_faces_seen"]}
        for kind in ZERO_JUMP_KINDS:
            data[f"zero_{kind}"] = fc["zero_face_max"].get(kind, 0.0)
        if len(s["patch_cache"]) > cfg["patch_cache_limit"]:
            s["patch_cache"].clear()
    if rss() > s["worker_memory_gib"]:
        raise MemoryError(f"{stage} worker exceeded declared memory cap: {rss():.3f} GiB")
    data.pop("ids", None)
    path = unit_path(s["output"], unit)
    save(path, ids=ids, **data)
    record = {"identity": s["identity"], "unit": unit, "sha256": sha(path), "seconds": time.monotonic() - started,
              "peak_rss_gib": rss(), "pid": os.getpid(), "jax_backend": "cpu"}
    write(path.with_suffix(".json"), record)
    if not valid_unit(s["output"], unit, s["identity"]):
        raise ValueError("new chunk did not validate")
    return record


def run_stage(args, stage, stage_units, identity):
    count = effective(args); output = args.output; records = []; resumed = 0; run = 0
    for n in args.resolutions:
        units = stage_units[str(n)]; todo = []
        for u in units:
            if valid_unit(output, u, identity):
                resumed += 1
            else:
                todo.append(u)
        if args.max_units is not None:
            remain = max(0, args.max_units - run); todo = todo[:remain]
        if not todo:
            continue
        with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn"), initializer=initialize,
                                  initargs=(str(args.input_root), str(output), n, stage, identity, args.worker_memory_gib),
                                  max_tasks_per_child=args.max_tasks_per_worker) as pool:
            pending = {}; iterator = iter(todo)

            def fill():
                for _ in range(max(0, 2 * count - len(pending))):
                    u = next(iterator, None)
                    if u is None:
                        break
                    pending[pool.submit(compute, u)] = u
            fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    u = pending.pop(future)
                    try:
                        records.append(future.result()); run += 1
                    except Exception as exc:
                        write(output / "failure.json", {"stage": stage, "unit": u, "error": repr(exc), "time": time.time()})
                        raise
                write(output / "progress.json", {"stage": stage, "n": n, "completed_this_invocation": run, "time": time.time()})
                fill()
    write(output / "executions" / f"{time.time_ns()}_{stage}.json",
          {"stage": stage, "requested_workers": args.workers, "effective_workers": count,
           "memory_budget_gib": args.memory_budget_gib, "worker_memory_gib": args.worker_memory_gib,
           "memory_reserve_gib": args.memory_reserve_gib, "executed_units": run, "resumed_units": resumed,
           "peak_worker_rss_gib": max((r["peak_rss_gib"] for r in records), default=0),
           "worker_seconds": sum(r["seconds"] for r in records)})
    return run


def complete_chunks(output, units, stage, n, identity):
    cursor = 0; expected = ids_for(n, stage) if units else np.empty(0, dtype=np.int64)
    for u in units:
        if u["start"] != cursor or u["stop"] <= cursor or not valid_unit(output, u, identity):
            raise ValueError(f"incomplete/overlapping stage at {cursor}: {u}")
        cursor = u["stop"]
    if cursor != len(expected):
        raise ValueError(f"missing tail stage IDs {cursor}/{len(expected)}")
    return len(expected)


# ---------------------------------------------------------------------------
# Reduction.
# ---------------------------------------------------------------------------
def reduce_observations(input_root, output, n, units, identity):
    t = core.load_context(n, input_root)
    values = np.zeros((len(t.vol), len(core.NAMES))); volume = np.zeros(len(t.vol))
    complete_chunks(output, units, "observations", n, identity)
    for u in units:
        with np.load(unit_path(output, u)) as z:
            ids = z["ids"]; oid = z["owner_ids"]
            if not np.array_equal(oid, t.ro[ids]):
                raise ValueError("observation owner mismatch")
            np.add.at(values, oid, z["numerator"]); np.add.at(volume, oid, t.rv[ids])
    if not np.allclose(volume, t.vol, rtol=1e-12, atol=1e-12):
        raise ValueError("incomplete raw observation members")
    values /= t.vol[:, None]
    path = output / f"N{n}.owner_values.npz"; save(path, values=values)
    return {"owner_values_sha256": sha(path), "owners": len(t.vol), "raw_members": n ** 3}


def reduce_raw(input_root, output, n, units, identity):
    t = core.load_context(n, input_root)
    V = len(core.VARIANT_NAMES)
    owners = len(t.vol)
    material = np.zeros((V, owners, 4)); remainder = np.zeros((V, owners, 4)); total = np.zeros((V, owners, 4))
    r_material = np.zeros((V, owners, 4)); r_remainder = np.zeros((V, owners, 4)); r_total = np.zeros((V, owners, 4))
    evolution_volume = np.zeros(owners)
    condition_max = 0.0; residual_max = 0.0; closure_max = 0.0
    exact_input_defect_max = np.zeros((V, 4))
    complete_chunks(output, units, "raw", n, identity)
    for u in units:
        with np.load(unit_path(output, u), allow_pickle=False) as z:
            ids = z["ids"]; oid = z["owner_ids"]; w = z["evolution_weight"]
            if not np.array_equal(oid, t.ro[ids]):
                raise ValueError("raw owner mismatch")
            for vi in range(V):
                np.add.at(material[vi], oid, w[:, None] * z["material"][vi])
                np.add.at(remainder[vi], oid, w[:, None] * z["remainder"][vi])
                np.add.at(total[vi], oid, w[:, None] * z["total"][vi])
                np.add.at(r_material[vi], oid, w[:, None] * z["R_material"][vi])
                np.add.at(r_remainder[vi], oid, w[:, None] * z["R_remainder"][vi])
                np.add.at(r_total[vi], oid, w[:, None] * z["R_total"][vi])
            np.add.at(evolution_volume, oid, w)
            condition_max = max(condition_max, float(z["condition_max"]))
            residual_max = max(residual_max, float(z["residual_max"]))
            closure_max = max(closure_max, float(z["closure_max"]))
            exact_input_defect_max = np.maximum(exact_input_defect_max, np.max(z["exact_input_defect"], axis=1))
    nonzero = evolution_volume > 0
    for arr in (material, remainder, total, r_material, r_remainder, r_total):
        arr[:, nonzero] /= evolution_volume[nonzero][None, :, None]
        arr[:, ~nonzero] = np.nan
    path = output / f"N{n}.raw.npz"
    save(path, material=material, remainder=remainder, total=total, R_material=r_material,
         R_remainder=r_remainder, R_total=r_total, evolution_volume=evolution_volume,
         condition_max=condition_max, residual_max=residual_max, closure_max=closure_max,
         exact_input_defect_max=exact_input_defect_max)
    return {"raw_sha256": sha(path), "condition_max": condition_max, "residual_max": residual_max,
            "closure_max": closure_max, "exact_input_defect_max": exact_input_defect_max.tolist(),
            "owners_with_zero_evolution_volume": int(np.sum(~nonzero))}


def reduce_faces(input_root, output, n, units, identity):
    t = core.load_context(n, input_root)
    V = len(core.VARIANT_NAMES)
    owners = len(t.vol)
    correction = np.zeros((V, owners, 4))
    zero_max = {kind: 0.0 for kind in ZERO_JUMP_KINDS}
    condition_max = 0.0; residual_max = 0.0; wall_exterior_defect_max = 0.0; wall_faces_seen = 0
    complete_chunks(output, units, "faces", n, identity)
    for u in units:
        with np.load(unit_path(output, u), allow_pickle=False) as z:
            lo = z["lo"]; hi = z["hi"]
            valid_lo = lo >= 0; valid_hi = hi >= 0
            for vi in range(V):
                np.add.at(correction[vi], lo[valid_lo], z["correction_lo"][vi][valid_lo])
                np.add.at(correction[vi], hi[valid_hi], z["correction_hi"][vi][valid_hi])
            condition_max = max(condition_max, float(z.get("condition_max", 0.0)))
            residual_max = max(residual_max, float(z.get("residual_max", 0.0)))
            wall_exterior_defect_max = max(wall_exterior_defect_max, float(z.get("wall_exterior_defect_max", 0.0)))
            wall_faces_seen += int(z.get("wall_faces_seen", 0))
            for kind in ZERO_JUMP_KINDS:
                zero_max[kind] = max(zero_max[kind], float(z.get(f"zero_{kind}", 0.0)))
    path = output / f"N{n}.faces.npz"
    save(path, correction=correction)
    return {"faces_sha256": sha(path), "zero_jump_max": zero_max, "condition_max": condition_max,
            "residual_max": residual_max, "wall_exterior_defect_max": wall_exterior_defect_max,
            "wall_faces_seen": wall_faces_seen}


def stats(error, weight, regions):
    """The P05N-convention report: {l2, max_abs, regions: {owners, l2, max_abs}},
    owner-physical-volume weighted (mirrors p05n_field_derived_global.campaign.stats)."""
    finite = np.isfinite(error) & np.isfinite(weight)
    error = np.where(finite, error, 0.0); weight = np.where(finite, weight, 0.0)
    sq = weight * error ** 2; total = float(np.sum(sq)); V = float(np.sum(weight))
    out = {"l2": float(np.sqrt(total / V)) if V > 0 else None, "max_abs": float(np.max(np.abs(error[finite]))) if np.any(finite) else None,
           "regions": {}}
    for label, mask in regions.items():
        m = mask & finite
        v = float(np.sum(weight[m]))
        out["regions"][label] = {"owners": int(np.sum(mask)),
                                  "l2": float(np.sqrt(np.sum(sq[m]) / v)) if v > 0 else None,
                                  "max_abs": float(np.max(np.abs(error[m]))) if np.any(m) else None}
    return out


def accepted_stats(error, weight, regions):
    """The accepted-P06 report format (``p06_structured_global.numerics``'s
    own ``_compact_statistics``: absolute_l2, relative_l2,
    maximum_absolute_error, signed_volume_weighted_mean_error, and a
    per-region squared_error_fraction/volume_fraction breakdown), called
    unmodified via a minimal ``owner_volume``/``masks`` shim. Same
    owner-physical-volume weight as ``stats``' P05N-convention report
    (``data.owner_volume`` there is the same portable owner volume ``t.vol``
    is here -- see this package's README.md, "Reduction outputs"); the two
    differ only in reported statistics/region taxonomy, not in the
    underlying weight, so the region breakdown uses this package's own
    region labels (``core.regional_masks``), not the accepted campaign's own
    (axis_core/first_ring/aggregate_interface/theta_seam/eta_seam) partition.

    Note: ``_compact_statistics`` is called here as ``(error, 0, shim)`` (its
    own ``actual - exact`` gives back ``error`` unchanged), so its
    ``relative_l2`` field -- defined relative to ``exact`` -- is not
    meaningful for an already-differenced N-R/N-D error and should be
    ignored; every other field is exactly its own accepted formula.
    """
    from types import SimpleNamespace
    shim = SimpleNamespace(owner_volume=weight, masks=regions)
    return core.p06numerics._compact_statistics(error + 0.0, np.zeros_like(error), shim)


def reduce_global(args, identity):
    output = args.output; cfg = config(); cases = {}
    for n in args.resolutions:
        t = core.load_context(n, args.input_root)
        with np.load(output / f"N{n}.raw.npz", allow_pickle=False) as z:
            total = z["total"].copy(); r_total = z["R_total"].copy()
            material = z["material"].copy(); remainder = z["remainder"].copy()
            evolution_volume = z["evolution_volume"].copy()
            condition_max = float(z["condition_max"]); residual_max = float(z["residual_max"])
            closure_max = float(z["closure_max"])
        with np.load(output / f"N{n}.faces.npz", allow_pickle=False) as z:
            correction = z["correction"].copy()
        centered_total = total
        u_material = material + correction / np.maximum(evolution_volume, 1e-300)[None, :, None]
        u_total = u_material + remainder
        owner_volume_weight = np.asarray(t.vol)
        regions = core.regional_masks(t)

        def _both(error):
            return {"p05n_norm": stats(error, owner_volume_weight, regions),
                    "accepted_norm": accepted_stats(error, owner_volume_weight, regions)}

        results = {}
        for ci, name in enumerate(core.CASE_NAMES):
            vi_n = core.VARIANT_NAMES.index(name)
            vi_d = core.VARIANT_NAMES.index(f"{name}:D")
            eq = {}
            for eqi, eqname in enumerate(core.EQUATIONS):
                eq[eqname] = {
                    "centered": {"N_minus_R": _both(centered_total[vi_n, :, eqi] - r_total[vi_n, :, eqi]),
                                 "N_minus_D": _both(centered_total[vi_n, :, eqi] - centered_total[vi_d, :, eqi])},
                    "U": {"N_minus_R": _both(u_total[vi_n, :, eqi] - r_total[vi_n, :, eqi]),
                          "N_minus_D": _both(u_total[vi_n, :, eqi] - u_total[vi_d, :, eqi])},
                }
            results[name] = eq
        constant_action_max = 0.0
        for cname in core.CONTROL_CASES:
            vi = core.VARIANT_NAMES.index(cname)
            constant_action_max = max(constant_action_max, float(np.nanmax(np.abs(centered_total[vi]))),
                                       float(np.nanmax(np.abs(u_total[vi]))))
        cases[str(n)] = {"owners": len(t.vol), "fields": results, "condition_max": condition_max,
                          "residual_max": residual_max, "closure_max": closure_max,
                          "constant_action_max": constant_action_max,
                          "arrays_sha256": {"raw": sha(output / f"N{n}.raw.npz"), "faces": sha(output / f"N{n}.faces.npz")}}
    orders = {}
    for cand in ("centered", "U"):
        orders[cand] = {}
        for name in core.GATED_CASES:
            for eqname in core.GATED_EQUATIONS:
                e = np.array([cases[str(n)]["fields"][name][eqname][cand]["N_minus_R"]["p05n_norm"]["l2"] for n in args.resolutions])
                entry = {"errors": e.tolist()}
                if len(e) == 3 and np.all(np.asarray(e, dtype=object) != None) and np.all(e > 0):
                    p = np.log(e[:-1] / e[1:]) / np.log(np.array([48 / 32, 64 / 48]))
                    entry["orders"] = p.tolist()
                    entry["order_pass"] = bool(np.all(p >= cfg["global_l2_order_minimum"]))
                orders[cand][f"{name}:{eqname}"] = entry
    global_order_pass = (len(args.resolutions) == 3 and
                          all(orders[cand][key].get("order_pass", False)
                              for cand in orders for key in orders[cand]))
    constant_pass = all(cases[str(n)]["constant_action_max"] <= cfg["constant_action_absolute_maximum"]
                         for n in args.resolutions)
    summary = {"identity": identity, "computation_completed": len(args.resolutions) == 3, "cases": cases,
               "orders": orders, "global_order_pass": global_order_pass, "constant_pass": constant_pass,
               "gated_cases": list(core.GATED_CASES), "heldout_cases": list(core.HELDOUT_CASES),
               "control_cases": list(core.CONTROL_CASES), "production_promoted": False,
               "note": "gated_cases are every nonconstant case, including the held-out cases (frozen catalogue gate); "
                       "N_minus_D is reported per case/equation/candidate, not gated"}
    write(output / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Preflight / equivalence.
# ---------------------------------------------------------------------------
def preflight_one(input_root, output, n, oracle=False):
    return p06n_preflight.run(input_root, n, output, oracle=oracle)


def preflight_stage(args, identity, oracle=False):
    output = args.output
    path = output / ("equivalence.json" if oracle else "preflight.json")
    cases = {}
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] == identity:
            cases = saved["cases"]
    todo = [n for n in args.resolutions if str(n) not in cases]
    count = max(1, min(len(todo), effective(args))) if todo else 0
    if count > 1:
        with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn")) as pool:
            futures = {n: pool.submit(preflight_one, args.input_root, output, n, oracle) for n in todo}
            for n in todo:
                cases[str(n)] = futures[n].result()
                write(path, {"identity": identity, "cases": cases})
    else:
        for n in todo:
            cases[str(n)] = preflight_one(args.input_root, output, n, oracle)
            write(path, {"identity": identity, "cases": cases})
    write(path, {"identity": identity, "cases": cases})
    return cases


def validate(args, planned, identity):
    if planned["identity"] != identity:
        raise ValueError("plan identity changed")
    output = args.output
    for stage in STAGES:
        for n in args.resolutions:
            complete_chunks(output, planned["stages"][stage][str(n)], stage, n, identity)
    summary = json.loads((output / "summary.json").read_text())
    if summary["identity"] != identity or not summary["computation_completed"]:
        raise ValueError("incomplete results")
    write(output / "validation.json", {"identity": identity, "operational_complete": True,
                                        "scientific_global_order_pass": summary["global_order_pass"],
                                        "constant_pass": summary["constant_pass"]})


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("verify-inputs", "plan", "preflight", "verify-equivalence",
                                       "run-stage", "reduce-stage", "run", "validate"))
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolutions", type=int, nargs="+", choices=(32, 48, 64), default=[32, 48, 64])
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--workers", type=int)
    p.add_argument("--memory-budget-gib", type=float); p.add_argument("--worker-memory-gib", type=float)
    p.add_argument("--memory-reserve-gib", type=float, default=1.0); p.add_argument("--max-tasks-per-worker", type=int, default=32)
    p.add_argument("--max-units", type=int)
    return p.parse_args()


def main():
    args = parse(); args.output = args.output.resolve(); args.input_root = args.input_root.resolve()
    if args.resolutions != sorted(set(args.resolutions)):
        raise ValueError("resolutions must be unique and ascending")
    if args.command in ("run-stage", "run"):
        effective(args)
    with lock(args.output):
        for name in ("scratch", "cache", "logs"):
            (args.output / name).mkdir(exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(args.output / "cache")
        os.environ["DRBX_CACHE_DIR"] = str(args.output / "cache/jax")
        os.environ["TMPDIR"] = str(args.output / "scratch")
        identity = verify(args.input_root, args.output)
        write(args.output / "invocations" / f"{time.time_ns()}_{args.command}.json",
              {"argv": sys.argv, "identity": identity, "time": time.time()})
        if args.command == "verify-inputs":
            return
        if args.command in ("plan", "run"):
            planned = plan(args.output, identity, args.resolutions)
        elif args.command in ("preflight", "verify-equivalence"):
            planned = None
        else:
            planned = json.loads((args.output / "plan.json").read_text())
            if planned["identity"] != identity or planned["resolutions"] != args.resolutions:
                raise ValueError("plan identity/resolutions mismatch")
        if args.command == "plan":
            return
        if args.command == "preflight":
            preflight_stage(args, identity)
        elif args.command == "verify-equivalence":
            preflight_stage(args, identity, oracle=True)
        elif args.command == "run-stage":
            if args.stage is None:
                raise ValueError("--stage required")
            run_stage(args, args.stage, planned["stages"][args.stage], identity)
        elif args.command == "reduce-stage":
            if args.stage == "observations":
                for n in args.resolutions:
                    result = reduce_observations(args.input_root, args.output, n, planned["stages"]["observations"][str(n)], identity)
                    write(args.output / f"N{n}.observations.reduction.json", {"identity": identity, **result})
            elif args.stage == "raw":
                for n in args.resolutions:
                    result = reduce_raw(args.input_root, args.output, n, planned["stages"]["raw"][str(n)], identity)
                    write(args.output / f"N{n}.raw.reduction.json", {"identity": identity, **result})
            elif args.stage == "faces":
                for n in args.resolutions:
                    result = reduce_faces(args.input_root, args.output, n, planned["stages"]["faces"][str(n)], identity)
                    write(args.output / f"N{n}.faces.reduction.json", {"identity": identity, **result})
                reduce_global(args, identity)
            else:
                raise ValueError("--stage required")
        elif args.command == "run":
            if not (args.output / "preflight.json").exists() or json.loads((args.output / "preflight.json").read_text())["identity"] != identity:
                raise ValueError("complete preflight required before main run")
            run_stage(args, "observations", planned["stages"]["observations"], identity)
            for n in args.resolutions:
                result = reduce_observations(args.input_root, args.output, n, planned["stages"]["observations"][str(n)], identity)
                write(args.output / f"N{n}.observations.reduction.json", {"identity": identity, **result})
            for stage in ("raw", "faces"):
                run_stage(args, stage, planned["stages"][stage], identity)
            for n in args.resolutions:
                result = reduce_raw(args.input_root, args.output, n, planned["stages"]["raw"][str(n)], identity)
                write(args.output / f"N{n}.raw.reduction.json", {"identity": identity, **result})
                result = reduce_faces(args.input_root, args.output, n, planned["stages"]["faces"][str(n)], identity)
                write(args.output / f"N{n}.faces.reduction.json", {"identity": identity, **result})
            reduce_global(args, identity)
            if args.resolutions == [32, 48, 64]:
                validate(args, planned, identity)
        elif args.command == "validate":
            validate(args, planned, identity)
    print(json.dumps({"command": args.command, "status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        a = parse()
        write(a.output / "last_exit.json", {"command": a.command, "status": "failed", "error": repr(exc), "time": time.time()})
        raise
