#!/usr/bin/env python3
"""Resumable node-local CPU campaign for the P05N field-derived global qualification.

Mirrors scripts/p07n_field_derived_global/campaign.py's CLI and resumability
(chunk NPZ + JSON receipts bound to a campaign identity digest of
configuration + source hashes + input manifest), with a simpler three-stage
pipeline ('observations', 'raw', 'faces') since P05N's reference R is cheap
and computed alongside the centered action rather than as its own MMS stage.

Do not put this package's own directory first on sys.path (it contains
operator.py, which would shadow the stdlib operator module). Put
DRBX/scripts on sys.path and run this as `python -m p05n_field_derived_global.campaign`
or invoke it directly with DRBX/scripts on sys.path (see README.md).
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

from p05n_field_derived_global import core
from p05n_field_derived_global import preflight as p05n_preflight

STAGES = ("observations", "raw", "faces")
STATE = {}


# ---------------------------------------------------------------------------
# Small persistence helpers (mirrors p07n_field_derived_global/campaign.py).
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
    if cfg["schema"] != "drbx.p05n-field-derived-static-global-v1" or cfg["resolutions"] != [32, 48, 64]:
        raise ValueError("unsupported frozen P05N field-derived contract")
    if cfg["physical_fields"] != list(core.NAMES):
        raise ValueError("physical field catalogue changed")
    return cfg


def _fixture_files():
    """Every vendored preflight fixture (tracked in the package), by resolution."""
    rel = ["scripts/p05n_field_derived_global/preflight_fixtures/fixtures_manifest.json",
           "scripts/p05n_field_derived_global/preflight_fixtures/extract.py"]
    for n in (32, 48, 64):
        rel.append(f"scripts/p05n_field_derived_global/preflight_fixtures/N{n}.selection.json")
        rel.append(f"scripts/p05n_field_derived_global/preflight_fixtures/N{n}.accepted_p05_replay.npz")
    return rel


def source_hashes():
    rel = ["scripts/p05n_field_derived_global/campaign.py", "scripts/p05n_field_derived_global/core.py",
           "scripts/p05n_field_derived_global/fields.py", "scripts/p05n_field_derived_global/rows.py",
           "scripts/p05n_field_derived_global/operator.py", "scripts/p05n_field_derived_global/preflight.py",
           "scripts/p05n_field_derived_global/configuration.json",
           "scripts/p05n_field_derived_global/input_manifest.json",
           "scripts/p05n_field_derived_global/p05n_catalogue.json",
           "scripts/p05n_field_derived_global/README.md",
           "scripts/p07n_field_derived_global/fields.py",
           "scripts/p07n_field_derived_global/campaign.py",
           "scripts/p07_combined_global/kernels.py", "scripts/p07_combined_global/topology.py",
           "scripts/p07_diffusion_global/numerics.py",
           "scripts/perpendicular_structured/reconstruction.py",
           "scripts/p05_direct_midpoint_global/direct_operator.py",
           "src/drbx/geometry/fci_perpendicular_neumann_trace.py",
           "src/drbx/geometry/fci_perpendicular_reconstruction.py",
           "src/drbx/native/fci_perpendicular_face_corrections.py"] + _fixture_files()
    return {p: sha(REPO / p) for p in rel}


def localize_sidecar(input_root, output):
    """Read the raw continuum reference sidecar and patch its absolute file
    paths to this input_root, exactly as scripts/p07n_field_derived_global's
    campaign.py does (verified locally to give a bit-identical `ref._metric`
    to the previously used already-localized sidecar). Idempotent: on resume,
    the existing localized copy must be byte-identical or verify() fails.
    """
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
                        "catalogue": {"roles": core.ROLES, "pairings": core.PAIRINGS,
                                      "dirichlet_counterpart": core.DIRICHLET_COUNTERPART}})
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
    for name in ("chunks", "scratch", "cache", "logs", "invocations"):
        (output / name).mkdir(exist_ok=True)
    return identity


# ---------------------------------------------------------------------------
# Topology / planning.
# ---------------------------------------------------------------------------
def load_topology(output, n):
    with np.load(Path(output) / f"N{n}.topology.npz", allow_pickle=False) as z:
        return z["face_ids"].copy(), z["endpoints"].copy()


def topo_one(input_root, output, n):
    started = time.process_time()
    core.pt.census(n, str(input_root), str(output))
    face_ids, endpoints = load_topology(output, n)
    t = core.load_context(n, input_root)
    return {"n": n, "faces": len(face_ids), "owners": len(t.vol), "cpu_seconds": time.process_time() - started,
            "rss_gib": rss()}


def effective(args):
    if args.workers is None or args.workers < 1:
        raise ValueError("--workers must be positive for computational stages")
    if args.worker_memory_gib is None or args.worker_memory_gib <= 0 or args.memory_budget_gib is None or args.memory_budget_gib <= 0:
        raise ValueError("--worker-memory-gib and --memory-budget-gib required")
    count = min(args.workers, int((args.memory_budget_gib - args.memory_reserve_gib) // args.worker_memory_gib))
    if count < 1:
        raise ValueError("memory budget cannot fit one worker plus reserve")
    return count


def topology_stage(args, identity):
    output = args.output
    path = output / "topology_summary.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError("topology identity changed")
        for n in args.resolutions:
            if sha(output / f"N{n}.topology.npz") != saved["hashes"][str(n)]:
                raise ValueError("topology hash changed")
        return
    items = [topo_one(args.input_root, output, n) for n in args.resolutions]
    write(path, {"identity": identity, "results": items,
                 "hashes": {str(n): sha(output / f"N{n}.topology.npz") for n in args.resolutions}})


def ids_for(output, n, stage):
    if stage == "faces":
        return load_topology(output, n)[0]
    return np.arange(n ** 3, dtype=np.int64)


def plan(output, identity, resolutions):
    cfg = config()
    chunks = {"observations": cfg["observation_chunk"], "raw": cfg["raw_chunk"], "faces": cfg["face_chunk"]}
    data = {"identity": identity, "resolutions": resolutions, "stages": {}}
    for stage in STAGES:
        data["stages"][stage] = {}
        for n in resolutions:
            ids = ids_for(output, n, stage); size = chunks[stage]
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


def unit_ids(output, unit):
    return ids_for(output, unit["n"], unit["stage"])[unit["start"]:unit["stop"]]


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
        if not np.array_equal(z["ids"], unit_ids(output, unit)):
            raise ValueError(f"wrong chunk IDs {path}")
        for name in z.files:
            if z[name].dtype.kind in "fc" and not np.isfinite(z[name]).all():
                raise ValueError(f"nonfinite {path}:{name}")
        expected = {"observations": ("numerator", "owner_ids"),
                    "raw": ("N", "D", "R", "owner_ids", "raw_volume"),
                    "faces": ("N", "D", "endpoints")}[unit["stage"]]
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
    lattice = core.WallLattice(t, ref, t.g.eta_period)
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
             "normal_data_fn": lattice.normal_data,
             "owner_values": owner_values, "stage": stage, "output": output, "identity": identity, "n": n,
             "worker_memory_gib": worker_memory_gib, "patch_cache": {}}


def compute(unit):
    s = STATE; ids = unit_ids(s["output"], unit); stage = s["stage"]; started = time.monotonic()
    period = s["t"].g.eta_period
    cfg = config()
    if stage == "observations":
        data = core.observation_chunk(s["t"], s["ref"], ids, period)
    elif stage == "raw":
        data = core.raw_chunk(s["t"], s["S"], s["ref"], s["ctx"], ids, s["owner_values"],
                               s["normal_coefficients"], s["patch_cache"], period,
                               normal_data_override=s["normal_data_fn"])
        if len(s["patch_cache"]) > cfg["patch_cache_limit"]:
            s["patch_cache"].clear()
    else:
        all_ids, all_endpoints = load_topology(s["output"], s["n"])
        pos = np.searchsorted(all_ids, ids)
        if not np.array_equal(all_ids[pos], ids):
            raise ValueError("face topology changed")
        data = core.face_chunk(s["t"], s["S"], s["ref"], s["ctx"], ids, all_endpoints[pos], s["owner_values"],
                                s["normal_coefficients"], s["patch_cache"], period, order=cfg["face_quadrature"],
                                normal_data_override=s["normal_data_fn"])
        if len(s["patch_cache"]) > cfg["patch_cache_limit"]:
            s["patch_cache"].clear()
    if rss() > s["worker_memory_gib"]:
        raise MemoryError(f"{stage} worker exceeded declared memory cap: {rss():.3f} GiB")
    path = unit_path(s["output"], unit)
    save(path, **data)
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


def complete_chunks(output, units, identity):
    cursor = 0; expected = ids_for(output, units[0]["n"], units[0]["stage"]) if units else np.empty(0, dtype=np.int64)
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
    complete_chunks(output, units, identity)
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


def regional_masks(t):
    n = t.n
    count = np.bincount(t.ro, minlength=len(t.vol))
    rawcount = count[t.ro].reshape((n, n, n))
    trans = np.zeros((n, n, n), bool)
    diff = rawcount[1:] != rawcount[:-1]
    trans[1:] |= diff; trans[:-1] |= diff
    transition_owner = np.bincount(t.ro, weights=trans.reshape(-1), minlength=len(t.vol)) > 0
    radius = np.zeros((n, n, n), np.int64); radius[:] = np.arange(n)[:, None, None]
    def owner_has(mask3d):
        return np.bincount(t.ro, weights=mask3d.reshape(-1), minlength=len(t.vol)) > 0
    axis_core = owner_has(radius == 0)
    first_ring = owner_has(radius == 1)
    wall = owner_has(radius == n - 1)
    last_two_layers = owner_has(radius >= n - 2)
    transition = transition_owner & ~axis_core & ~wall
    aggregate = (count > 1) & ~axis_core & ~wall & ~transition
    ordinary = ~(axis_core | wall | transition | aggregate)
    return {"axis_core": axis_core, "first_ring": first_ring, "physical_wall": wall,
            "transverse_last_two_layers": last_two_layers, "transition": transition,
            "aggregate": aggregate, "ordinary": ordinary, "interior": ~(axis_core | wall | last_two_layers)}


def project_raw(numerator, raw_owner, raw_volume, owner_count):
    out = np.zeros((owner_count,) + numerator.shape[1:])
    np.add.at(out, raw_owner, raw_volume.reshape(-1, *([1] * (numerator.ndim - 1))) * numerator)
    return out


def reduce_raw(input_root, output, n, units, identity):
    t = core.load_context(n, input_root)
    P = len(core.PAIR_NAMES)
    N = np.zeros((len(t.vol), P)); D = np.zeros((len(t.vol), P)); R = np.zeros((len(t.vol), P))
    volume = np.zeros(len(t.vol))
    condition_max = 0.0; residual_max = 0.0; antisymmetry_max = 0.0; exact_input_defect_max = np.zeros(P)
    complete_chunks(output, units, identity)
    for u in units:
        with np.load(unit_path(output, u), allow_pickle=False) as z:
            ids = z["ids"]; oid = z["owner_ids"]; vol = z["raw_volume"]
            if not np.array_equal(oid, t.ro[ids]):
                raise ValueError("raw owner mismatch")
            np.add.at(N, oid, vol[:, None] * z["N"]); np.add.at(D, oid, vol[:, None] * z["D"])
            np.add.at(R, oid, vol[:, None] * z["R"]); np.add.at(volume, oid, vol)
            condition_max = max(condition_max, float(z["condition_max"]))
            residual_max = max(residual_max, float(z["residual_max"]))
            antisymmetry_max = max(antisymmetry_max, float(z["antisymmetry"]))
            exact_input_defect_max = np.maximum(exact_input_defect_max, z["exact_input_defect"])
    if not np.allclose(volume, t.vol, rtol=1e-12, atol=1e-12):
        raise ValueError("incomplete raw members in reduced raw stage")
    N /= t.vol[:, None]; D /= t.vol[:, None]; R /= t.vol[:, None]
    path = output / f"N{n}.raw.npz"
    save(path, N=N, D=D, R=R, condition_max=condition_max, residual_max=residual_max,
         antisymmetry_max=antisymmetry_max, exact_input_defect_max=exact_input_defect_max)
    return {"raw_sha256": sha(path), "condition_max": condition_max, "residual_max": residual_max,
            "antisymmetry_max": antisymmetry_max, "exact_input_defect_max": exact_input_defect_max.tolist()}


def reduce_faces(input_root, output, n, units, identity):
    t = core.load_context(n, input_root)
    P = len(core.PAIR_NAMES)
    N = np.zeros((len(t.vol), P)); D = np.zeros((len(t.vol), P))
    zero_max = {"physical_wall": 0.0, "radial_n_minus_1": 0.0, "transverse_last_two_layers": 0.0}
    condition_max = 0.0; residual_max = 0.0
    complete_chunks(output, units, identity)
    all_ids, all_endpoints = load_topology(output, n)
    for u in units:
        with np.load(unit_path(output, u), allow_pickle=False) as z:
            ids = z["ids"]; ep = z["endpoints"]
            pos = np.searchsorted(all_ids, ids)
            if not np.array_equal(all_ids[pos], ids) or not np.array_equal(ep, all_endpoints[pos]):
                raise ValueError("face topology changed")
            lo = ep[:, 0]; hi = ep[:, 1]
            valid_lo = lo >= 0; valid_hi = hi >= 0
            np.add.at(N, lo[valid_lo], z["N"][valid_lo]); np.add.at(D, lo[valid_lo], z["D"][valid_lo])
            np.add.at(N, hi[valid_hi], -z["N"][valid_hi]); np.add.at(D, hi[valid_hi], -z["D"][valid_hi])
            keys = core.pt.decode(n, ids)
            for row in range(len(ids)):
                kind = core._face_kind(n, keys[row])
                if kind is not None:
                    zero_max[kind] = max(zero_max[kind], float(np.max(np.abs(z["N"][row]))),
                                          float(np.max(np.abs(z["D"][row]))))
            condition_max = max(condition_max, float(z.get("condition_max", 0.0)))
            residual_max = max(residual_max, float(z.get("residual_max", 0.0)))
    N /= t.vol[:, None]; D /= t.vol[:, None]
    path = output / f"N{n}.faces.npz"
    save(path, N=N, D=D)
    return {"faces_sha256": sha(path), "zero_jump_max": zero_max, "condition_max": condition_max,
            "residual_max": residual_max}


def stats(error, volume, regions):
    sq = volume * error ** 2; total = float(np.sum(sq)); V = float(np.sum(volume))
    out = {"l2": float(np.sqrt(total / V)), "max_abs": float(np.max(np.abs(error))), "regions": {}}
    for label, mask in regions.items():
        v = float(np.sum(volume[mask]))
        out["regions"][label] = {"owners": int(np.sum(mask)),
                                  "l2": float(np.sqrt(np.sum(sq[mask]) / v)) if v > 0 else None,
                                  "max_abs": float(np.max(np.abs(error[mask]))) if np.any(mask) else None}
    return out


def reduce_global(args, identity):
    output = args.output; cfg = config(); cases = {}
    for n in args.resolutions:
        t = core.load_context(n, args.input_root)
        with np.load(output / f"N{n}.raw.npz", allow_pickle=False) as z:
            N_centered = z["N"].copy(); D_centered = z["D"].copy(); R = z["R"].copy()
            condition_max = float(z["condition_max"]); residual_max = float(z["residual_max"])
            antisymmetry_max = float(z["antisymmetry_max"]); exact_input_defect_max = z["exact_input_defect_max"].copy()
        with np.load(output / f"N{n}.faces.npz", allow_pickle=False) as z:
            N_jump = z["N"].copy(); D_jump = z["D"].copy()
        N1 = N_centered; N2 = N_centered + N_jump
        D1 = D_centered; D2 = D_centered + D_jump
        regions = regional_masks(t)
        results = {}
        for col, name in enumerate(core.PAIR_NAMES):
            results[name] = {
                "candidate_direct_centered": {"N_minus_R": stats(N1[:, col] - R[:, col], t.vol, regions),
                                               "N_minus_D": stats(N1[:, col] - D1[:, col], t.vol, regions)},
                "candidate_direct_centered_plus_live_U_minus_A": {
                    "N_minus_R": stats(N2[:, col] - R[:, col], t.vol, regions),
                    "N_minus_D": stats(N2[:, col] - D2[:, col], t.vol, regions)},
            }
        constant_action_max = max(float(np.max(np.abs(N1[:, core.PAIR_NAMES.index(p)])))
                                   for p in core.CONSTANT_PAIRS)
        constant_action_max = max(constant_action_max,
                                   max(float(np.max(np.abs(N2[:, core.PAIR_NAMES.index(p)])))
                                       for p in core.CONSTANT_PAIRS))
        cases[str(n)] = {"owners": len(t.vol), "fields": results, "condition_max": condition_max,
                          "residual_max": residual_max, "antisymmetry_max": antisymmetry_max,
                          "exact_input_defect_max": exact_input_defect_max.tolist(),
                          "constant_action_max": constant_action_max,
                          "arrays_sha256": {"raw": sha(output / f"N{n}.raw.npz"),
                                            "faces": sha(output / f"N{n}.faces.npz")}}
    orders = {}
    for cand in ("candidate_direct_centered", "candidate_direct_centered_plus_live_U_minus_A"):
        orders[cand] = {}
        for name in core.GATED_PAIRS:
            e = np.array([cases[str(n)]["fields"][name][cand]["N_minus_R"]["l2"] for n in args.resolutions])
            entry = {"errors": e.tolist()}
            if len(e) == 3 and np.all(e > 0):
                p = np.log(e[:-1] / e[1:]) / np.log(np.array([48 / 32, 64 / 48]))
                entry["orders"] = p.tolist()
                entry["order_pass"] = bool(np.all(p >= cfg["global_l2_order_minimum"]))
            orders[cand][name] = entry
    global_order_pass = (len(args.resolutions) == 3 and
                          all(orders[cand][name].get("order_pass", False)
                              for cand in orders for name in core.GATED_PAIRS))
    constant_pass = all(cases[str(n)]["constant_action_max"] <= cfg["constant_action_absolute_maximum"]
                         for n in args.resolutions)
    antisymmetry_pass = all(cases[str(n)]["antisymmetry_max"] <= cfg["antisymmetry_absolute_maximum"]
                             for n in args.resolutions)
    summary = {"identity": identity, "computation_completed": len(args.resolutions) == 3, "cases": cases,
               "orders": orders, "global_order_pass": global_order_pass, "constant_pass": constant_pass,
               "antisymmetry_pass": antisymmetry_pass, "gated_pairs": list(core.GATED_PAIRS),
               "control_pairs": list(core.CONTROL_PAIRS), "production_promoted": False,
               "accuracy_record_note": "N_minus_D is reported per candidate/pairing, not gated; "
                                       "the scientific gate is global_order_pass on N_minus_R for gated_pairs, both candidates"}
    write(output / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Preflight (small-scale, non-chunked): gates (i)/(ii)/(iii).
# ---------------------------------------------------------------------------
def preflight_one(input_root, output, n, oracle=False):
    return p05n_preflight.run(input_root, n, output, oracle=oracle)


def preflight_stage(args, identity, oracle=False):
    output = args.output
    path = output / ("equivalence.json" if oracle else "preflight.json")
    cases = {}
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] == identity:
            cases = saved["cases"]
    todo = [n for n in args.resolutions if str(n) not in cases]
    # The per-grid preflights are independent: run them concurrently, bounded
    # by the same worker and memory controls as the main stages.
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
    topo = json.loads((output / "topology_summary.json").read_text())
    if topo["identity"] != identity:
        raise ValueError("topology summary identity changed")
    for stage in STAGES:
        for n in args.resolutions:
            complete_chunks(output, planned["stages"][stage][str(n)], identity)
    summary = json.loads((output / "summary.json").read_text())
    if summary["identity"] != identity or not summary["computation_completed"]:
        raise ValueError("incomplete results")
    write(output / "validation.json", {"identity": identity, "operational_complete": True,
                                        "scientific_global_order_pass": summary["global_order_pass"],
                                        "constant_pass": summary["constant_pass"]})


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("verify-inputs", "topology", "plan", "preflight", "verify-equivalence",
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
    if args.command in ("topology", "run-stage", "run"):
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
        if args.command in ("topology", "plan", "run"):
            topology_stage(args, identity)
        if args.command == "topology":
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
