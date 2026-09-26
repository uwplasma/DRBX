#!/usr/bin/env python3
"""Restartable N32/N48/N64 P05 reconstructed raw-midpoint bracket campaign."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import fcntl
import hashlib
import json
import math
import os.path
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
try:  # Package import in tests; direct script execution on Perlmutter.
    from .direct_operator import candidate_variants, direct_pair_actions, orders_from_rms, project_raw_to_owners, regional_owner_masks, weighted_stats
except ImportError:
    sys.path.insert(0, str(REPO / "scripts"))
    from p05_direct_midpoint_global.direct_operator import candidate_variants, direct_pair_actions, orders_from_rms, project_raw_to_owners, regional_owner_masks, weighted_stats

P05_NUMERICS = "scripts/p05_structured_global/numerics.py"
P05_INPUTS = "scripts/p05_structured_global/input_manifest.json"
REUSE_BUNDLE = HERE / "reuse_bundle/reuse_inputs_v1.zip"
REUSE_MANIFEST = HERE / "reuse_bundle/reuse_manifest.json"
CONFIG_PATH = HERE / "configuration.json"
SOURCE_FILES = ("campaign.py", "direct_operator.py", "bundle_reuse.py", "configuration.json", "reuse_bundle/reuse_inputs_v1.zip", "reuse_bundle/reuse_manifest.json")
STATE = {}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def json_safe(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_safe, allow_nan=False) + "\n")
    temp.replace(path)


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def config():
    cfg = json.loads(CONFIG_PATH.read_text())
    if cfg.get("schema") != "drbx.p05-direct-midpoint-global-v1":
        raise ValueError("unsupported direct-midpoint campaign configuration")
    return cfg


def git_info():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    return {"commit": commit, "branch": branch}


def source_identity():
    producer = json.loads(REUSE_MANIFEST.read_text())["producer"]["producer_sources"]
    for relative, record in producer.items():
        path = REPO / relative
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"reused P05 source dependency changed: {relative}")
    return {name: {"bytes": (HERE / name).stat().st_size, "sha256": sha256(HERE / name)} for name in SOURCE_FILES}


def resolve_input(input_root, entry):
    relative = Path(entry["path"])
    direct = Path(input_root) / relative
    if direct.is_file():
        return direct.resolve(), "manifest_path"
    if relative.parts[0] == "geometry_artifacts":
        n = int(relative.parts[-2][:2])
        canonical = Path(input_root) / "prototype_runs/geometry" / f"hsx_fci_{n}x{n}x{n}" / relative.name
        if canonical.is_file():
            return canonical.resolve(), "prototype_runs_geometry_mapping"
    raise FileNotFoundError(f"required P05 input missing at manifest or canonical path: {entry['path']}")


def input_manifest():
    return json.loads((REPO / P05_INPUTS).read_text())


def assert_adoptable_output(output):
    allowed = {".campaign.lock", "chunks", "executions", "logs", "cache", "scratch",
               "reuse_inputs", "runtime_inputs", "reference_sidecar.json"}
    if any(output.iterdir()) and any(path.name not in allowed for path in output.iterdir()):
        raise ValueError("refusing to adopt a nonempty output directory without its identity manifest")


def verify_inputs(input_root, output):
    input_root = Path(input_root).resolve(); output = Path(output).resolve()
    cfg = config(); reuse_manifest = json.loads(REUSE_MANIFEST.read_text())
    if not REUSE_BUNDLE.is_file() or sha256(REUSE_BUNDLE) != reuse_manifest["archive"]["sha256"]:
        raise ValueError("committed reuse bundle failed its manifest")
    sources = source_identity()
    original_inputs = {"files": reuse_manifest["runtime_geometry_inputs"]}
    if reuse_manifest["cases"] != cfg["cases"] or reuse_manifest["pairs"] != cfg["pairs"] or reuse_manifest["fields"] != cfg["fields"]:
        raise ValueError("reuse bundle field/case mapping differs from the frozen campaign configuration")
    records = []
    resolved = {}
    for entry in original_inputs["files"]:
        path, mapping = resolve_input(input_root, entry)
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"input size mismatch: {entry['path']}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"input hash mismatch: {entry['path']}")
        records.append({"manifest_path": entry["path"], "resolved_name": str(path.relative_to(input_root)) if path.is_relative_to(input_root) else path.name,
                        "bytes": entry["bytes"], "sha256": actual, "path_mapping": mapping})
        resolved[entry["path"]] = path

    # Independently check the geometry identities carried by the old producer manifest.
    geometry_entries = {row["path"]: row for row in original_inputs["files"] if row["path"].startswith("geometry_artifacts/")}
    for record in records:
        if record["manifest_path"] in geometry_entries and record["sha256"] != geometry_entries[record["manifest_path"]]["sha256"]:
            raise ValueError("geometry identity differs from the old campaign producer")

    output.mkdir(parents=True, exist_ok=True)
    for name in ("chunks", "executions", "logs", "cache", "scratch", "reuse_inputs", "runtime_inputs"):
        (output / name).mkdir(exist_ok=True)
    # Materialize the exact reused numerical arrays inside the downloadable campaign folder.
    archive_copy = output / "reuse_inputs/reuse_inputs_v1.zip"
    if not archive_copy.exists(): shutil.copyfile(REUSE_BUNDLE, archive_copy)
    elif sha256(archive_copy) != reuse_manifest["archive"]["sha256"]: raise ValueError("output reuse bundle changed")
    with zipfile.ZipFile(archive_copy) as zf:
        zf.extractall(output / "reuse_inputs")
    embedded_manifest = json.loads((output / "reuse_inputs/reuse_manifest.json").read_text())
    if embedded_manifest["producer"]["campaign_identity"] != reuse_manifest["producer"]["campaign_identity"]:
        raise ValueError("embedded reuse identity mismatch")
    local_reuse = {}
    for name, row in reuse_manifest["resolution_artifacts"].items():
        path = output / "reuse_inputs" / name
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"materialized reuse artifact hash mismatch: {name}")
        local_reuse[name] = {"bytes": row["bytes"], "sha256": row["sha256"]}

    # Make the P05 loader's expected geometry layout from verified immutable inputs.
    geometry_root = output / "runtime_inputs/geometry_artifacts/rlp_convergence_32_48_64_20260917"
    geometry_root.mkdir(parents=True, exist_ok=True)
    for n in cfg["resolutions"]:
        manifest_rel = f"geometry_artifacts/rlp_convergence_32_48_64_20260917/{n}x{n}x{n}/base_geometry.npz"
        source_dir = resolved[manifest_rel].parent
        target_dir = geometry_root / f"{n}x{n}x{n}"
        if target_dir.is_symlink() and target_dir.resolve() != source_dir.resolve():
            target_dir.unlink()
        if not target_dir.exists(): target_dir.symlink_to(source_dir, target_is_directory=True)
        if target_dir.resolve() != source_dir.resolve(): raise ValueError("runtime geometry mapping is inconsistent")

    sidecar_entry = next(x for x in original_inputs["files"] if x["path"].endswith("continuous_reference_sidecar.json"))
    side = json.loads(resolved[sidecar_entry["path"]].read_text())
    metric = next(x for x in records if x["manifest_path"].endswith("hsx_metric_d58d392545fd3917efeb83b6.npz"))
    makegrid = next(x for x in records if x["manifest_path"].endswith("mgrid_res2p5cm_180pln.nc"))
    side["metric_cache"]["path"] = str(input_root / metric["manifest_path"])
    if not Path(side["metric_cache"]["path"]).is_file(): side["metric_cache"]["path"] = str(resolved[next(x["manifest_path"] for x in records if x is metric)])
    side["makegrid"]["path"] = str(input_root / makegrid["manifest_path"])
    if not Path(side["makegrid"]["path"]).is_file(): side["makegrid"]["path"] = str(resolved[next(x["manifest_path"] for x in records if x is makegrid)])
    side["metric_query_batch_size"] = 4096
    sidecar_path = output / "reference_sidecar.json"
    save_json(sidecar_path, side)

    git = git_info()
    content = {"schema": cfg["schema"], "git": git, "config": cfg, "sources": sources,
               "reuse_bundle_sha256": reuse_manifest["archive"]["sha256"],
               "reuse_producer_campaign_identity": reuse_manifest["producer"]["campaign_identity"],
               "inputs": records, "reuse_arrays": local_reuse,
               "localized_reference_sidecar_sha256": sha256(sidecar_path),
               "geometry_loader_mapping": "runtime_inputs symlinks point to verified canonical or manifest geometry files"}
    ident = digest(content)
    manifest_path = output / "campaign_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["identity"] != ident:
            raise ValueError("output folder belongs to a different campaign/input identity; use a new folder")
    else:
        assert_adoptable_output(output)
        save_json(manifest_path, {"identity": ident, "content": content, "preflight_complete": False, "run_complete": False})
    save_json(output / "input_verification.json", {"identity": ident, "verified": True, "files": records,
                                                      "reuse_bundle": reuse_manifest["archive"],
                                                      "cpu_backend_required": "JAX_PLATFORMS=cpu; CUDA_VISIBLE_DEVICES=''"})
    return ident


def current_identity(output):
    manifest = json.loads((Path(output) / "campaign_manifest.json").read_text())
    if manifest["identity"] != digest(manifest["content"]): raise ValueError("campaign manifest content digest mismatch")
    if manifest["content"]["git"] != git_info(): raise ValueError("repository revision/branch differs from verified campaign")
    if manifest["content"]["sources"] != source_identity(): raise ValueError("campaign code or reused source identity changed")
    return manifest["identity"]


def load_framework():
    try:
        from p05_structured_global import numerics as k
        from p07_diffusion_global import numerics as reference_numerics
    except ImportError:
        from scripts.p05_structured_global import numerics as k
        from scripts.p07_diffusion_global import numerics as reference_numerics
    return k, reference_numerics


def initialize_worker(input_root, output, n):
    output = Path(output); input_root = Path(input_root)
    for name in ("DRBX_CACHE_DIR", "JAX_COMPILATION_CACHE_DIR", "TMPDIR"):
        path = output / ("scratch" if name == "TMPDIR" else "cache")
        path.mkdir(exist_ok=True)
        os.environ[name] = str(path)
    k, reference_numerics = load_framework()
    import jax
    backend = jax.default_backend()
    if backend != "cpu": raise RuntimeError(f"campaign worker initialized JAX backend {backend}; CPU is required")
    t = k.load_context(n, output / "runtime_inputs")
    S = k.StructuredReconstruction(t)
    ref = reference_numerics.reference(output / "reference_sidecar.json", verify_hashes=False)
    with np.load(output / "reuse_inputs" / f"N{n}.reuse.npz", allow_pickle=False) as data:
        reuse = {name: data[name] for name in data.files}
    if not np.array_equal(t.vol, reuse["volume"]): raise ValueError(f"N{n} stored owner-volume mismatch")
    STATE.clear(); STATE.update(k=k, t=t, S=S, ref=ref, reuse=reuse, output=output, n=n, cpu_backend=backend)


def reference_points(raw_ids):
    """Match the producer's 16-cell batch layout and memoize those exact point references."""
    state = STATE; t=state["t"]; k=state["k"]; ref=state["ref"]
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    values = np.empty((len(raw_ids), len(k.PAIRS)))
    slotmap = {int(raw): i for i, raw in enumerate(raw_ids)}
    cache = STATE.setdefault("ref_block_cache", {})
    for raw in raw_ids:
        first = int(raw // 16 * 16)
        if first not in cache:
            stop = min(first + 16, t.n**3)
            block_ids = np.arange(first, stop, dtype=np.int64)
            block = k.reference_cells(t, ref, block_ids, order=1)
            cache[first] = block["numerator"] / block["volume"][:, None]
            if len(cache) > 128: cache.pop(next(iter(cache)))
        values[slotmap[int(raw)]] = cache[first][int(raw)-first]
    return values


def boundary_trace(k, ref, points):
    return k.boundary_trace(ref, points)


def compute_owner(k, t, S, ref, reuse, owner, labels):
    raw_ids = np.flatnonzero(t.ro == int(owner))
    if not len(raw_ids): raise ValueError(f"owner {owner} has no raw members")
    points = t.pts[raw_ids]
    keys = np.array(np.unravel_index(raw_ids, (t.n,)*3)).T
    gradients = np.empty((len(raw_ids), 3, len(k.FIELDS)))
    boundary_count=0; max_residual=0.0
    for j, (point, key) in enumerate(zip(points, keys)):
        row = S.rows(tuple(map(int,key)), point[None,:], location="cell")
        value, gradient = row.apply(reuse["observations"], lambda q: boundary_trace(k, ref, q))
        gradients[j] = gradient[0]
        boundary_count += int(row.boundary_conditioned)
        max_residual = max(max_residual, float(row.diagnostics.get("max_residual", 0.0)))
    metric = ref._metric(points)
    h = metric["bcov"] / metric["B"][:,None]
    jac = np.abs(metric["J"])
    direct, antisymmetry = direct_pair_actions(h, jac, gradients, k.PAIRS)
    volume = t.rv[raw_ids]
    projected = np.einsum("r,rp->p", volume, direct) / t.vol[owner]
    constant_action = direct[:, 5]
    reference = reference_points(raw_ids)
    exact_ref = np.einsum("r,rp->p", volume, reference) / t.vol[owner]
    reference_delta = exact_ref - reuse["reference"][owner]
    # The only wider replay envelope applies to the known batch-sensitive omega reference slot.
    omega_delta = float(abs(reference_delta[0]))
    smooth_delta = float(np.max(np.abs(reference_delta[1:])))
    cfg = config()
    if smooth_delta > cfg["smooth_reference_replay_abs_tolerance"]:
        raise ValueError(f"smooth analytic reference replay mismatch owner={owner}: {smooth_delta}")
    if omega_delta > cfg["actual_omega_batched_replay_abs_tolerance"]:
        raise ValueError(f"actual-omega batched reference replay exceeds its scoped tolerance owner={owner}: {omega_delta}")
    # Singleton-vs-batched and same-mode repeat check documents the historical nested-derivative noise.
    sample_raw = int(raw_ids[0])
    once = k.reference_cells(t, ref, np.array([sample_raw]), order=1)["numerator"]
    again = k.reference_cells(t, ref, np.array([sample_raw]), order=1)["numerator"]
    singleton_repeat = float(np.max(np.abs(once-again)))
    block_value = reference_points([sample_raw])[0]
    singleton_value = once[0] / t.rv[sample_raw]
    singleton_batch = float(abs(singleton_value[0]-block_value[0]))
    if singleton_repeat > 1e-12 or singleton_batch > cfg["actual_omega_batched_replay_abs_tolerance"]:
        raise ValueError(f"actual-omega singleton/batched check failed owner={owner}")
    return {"owner": int(owner), "labels": labels, "raw_ids": raw_ids,
            "raw_reference_replay_max_abs": float(np.max(np.abs(reference_delta))),
            "omega_reference_replay_abs": omega_delta, "smooth_reference_replay_max_abs": smooth_delta,
            "singleton_repeat_max_abs": singleton_repeat, "singleton_vs_batched_omega_abs": singleton_batch,
            "projected": projected, "raw_actions": direct, "boundary_conditioned_rows": boundary_count,
            "support_residual_max": max_residual, "constant_action_max_abs": float(np.max(np.abs(constant_action))),
            "constant_gradient_max_abs": float(np.max(np.abs(gradients[:,:,6]))),
            "argument_antisymmetry_max_abs": antisymmetry,
            "max_gradient_abs": float(np.max(np.abs(gradients))),
            "finite": bool(np.isfinite(projected).all() and np.isfinite(gradients).all())}


def select_preflight_owners(t, cfg):
    n=t.n; raw_owner=t.ro; selections={}
    labels={}
    fractions=cfg["preflight_angular_fractions_eighths"]
    for numerator_theta, numerator_eta in fractions:
        j=int(round(numerator_theta*n/8))%n; k=int(round(numerator_eta*n/8))%n
        for distance in cfg["preflight_wall_radial_indices_from_end"]:
            i=n-distance
            raw=int(np.ravel_multi_index((i,j,k),(n,n,n)))
            owner=int(raw_owner[raw])
            region="wall" if distance==1 else "adjacent"
            labels.setdefault(owner,set()).add(f"{region}:r{i}:theta{j}:eta{k}")
    records=[{"owner":owner,"labels":sorted(tags),"member_count":int(np.count_nonzero(raw_owner==owner))}
             for owner,tags in sorted(labels.items())]
    if len(records)>cfg["preflight_owner_count_limit_per_resolution"]:
        raise ValueError("deterministic preflight sample exceeded its frozen owner cap")
    return records


def preflight_units(output,n):
    return json.loads((Path(output)/f"N{n}.preflight_owners.json").read_text())["owners"]


def unit_record(stage,n,index,entry=None):
    if stage=="preflight": return {"stage":stage,"n":n,"index":index,"owner":int(entry["owner"])}
    chunk=config()["raw_chunk_size"]; start=index*chunk
    return {"stage":stage,"n":n,"index":index,"start":start,"stop":min(start+chunk,n**3)}


def chunk_path(output,unit):
    if unit["stage"]=="preflight": return Path(output)/"chunks"/f"N{unit['n']}.preflight_owner_{unit['owner']:07d}.npz"
    return Path(output)/"chunks"/f"N{unit['n']}.raw_{unit['start']:07d}_{unit['stop']:07d}.npz"


def valid_chunk(output,unit,identity):
    path=chunk_path(output,unit);receipt=path.with_suffix(".json")
    if not path.is_file() or not receipt.is_file(): return False
    try:
        r=json.loads(receipt.read_text())
        return r["identity"]==identity and r["unit"]==unit and r["sha256"]==sha256(path)
    except (OSError,KeyError,ValueError): return False


def worker_task(payload):
    stage=payload["stage"]; unit=payload["unit"]
    state=STATE;k=state["k"];t=state["t"];S=state["S"];ref=state["ref"];reuse=state["reuse"]
    start=time.perf_counter()
    if stage=="preflight":
        entry=payload["entry"]
        result=compute_owner(k,t,S,ref,reuse,entry["owner"],entry["labels"])
        if result["support_residual_max"]>1e-9: raise ValueError(f"preflight support residual too large owner={entry['owner']}")
        if result["constant_action_max_abs"]>1e-8 or result["constant_gradient_max_abs"]>1e-8:
            raise ValueError(f"preflight constant identity failed owner={entry['owner']}")
        if not result["finite"]: raise FloatingPointError("nonfinite preflight owner result")
        # A physical-wall row and the adjacent reconstruction region must use prescribed trace data.
        radial=np.array(np.unravel_index(result["raw_ids"],(t.n,)*3)).T[:,0]
        if np.any(radial>=t.n-6) and result["boundary_conditioned_rows"]==0:
            raise ValueError(f"expected BC-conditioned rows were not used owner={entry['owner']}")
        arrays={"owner":np.array(result["owner"]),"raw_ids":result["raw_ids"],"labels":np.asarray(result["labels"]),
                "cpu_backend":np.asarray(state["cpu_backend"]),
                "projected":result["projected"],"raw_actions":result["raw_actions"],
                "raw_reference_replay_max_abs":np.array(result["raw_reference_replay_max_abs"]),
                "omega_reference_replay_abs":np.array(result["omega_reference_replay_abs"]),
                "smooth_reference_replay_max_abs":np.array(result["smooth_reference_replay_max_abs"]),
                "singleton_repeat_max_abs":np.array(result["singleton_repeat_max_abs"]),
                "singleton_vs_batched_omega_abs":np.array(result["singleton_vs_batched_omega_abs"]),
                "boundary_conditioned_rows":np.array(result["boundary_conditioned_rows"]),
                "support_residual_max":np.array(result["support_residual_max"]),
                "constant_action_max_abs":np.array(result["constant_action_max_abs"]),
                "constant_gradient_max_abs":np.array(result["constant_gradient_max_abs"]),
                "argument_antisymmetry_max_abs":np.array(result["argument_antisymmetry_max_abs"])}
    else:
        ids=np.arange(unit["start"],unit["stop"],dtype=np.int64)
        raw_result=compute_raw_direct(k,t,S,ref,reuse,ids)
        arrays={"ids":ids,"action":raw_result["action"],"cpu_backend":np.asarray(state["cpu_backend"]),
                "support_residual_max":np.array(raw_result["support_residual_max"]),
                "constant_gradient_max_abs":np.array(raw_result["constant_gradient_max_abs"]),
                "constant_action_max_abs":np.array(raw_result["constant_action_max_abs"]),
                "argument_antisymmetry_max_abs":np.array(raw_result["argument_antisymmetry_max_abs"])}
    if not all(np.isfinite(np.asarray(value)).all() for value in arrays.values() if np.asarray(value).dtype.kind in "fiu"):
        raise FloatingPointError(f"nonfinite result in {unit}")
    return {"arrays":arrays,"seconds":time.perf_counter()-start,"peak_rss_gib":peak_rss_gib()}


def compute_raw_direct(k,t,S,ref,reuse,ids):
    points=t.pts[ids];keys=np.array(np.unravel_index(ids,(t.n,)*3)).T
    gradients=np.empty((len(ids),3,len(k.FIELDS)));maxres=0.0
    for j,(point,key) in enumerate(zip(points,keys)):
        row=S.rows(tuple(map(int,key)),point[None,:],location="cell")
        _,g=row.apply(reuse["observations"],lambda q:boundary_trace(k,ref,q))
        gradients[j]=g[0];maxres=max(maxres,float(row.diagnostics.get("max_residual",0.0)))
    metric=ref._metric(points);h=metric["bcov"]/metric["B"][:,None];jac=np.abs(metric["J"])
    action,anti=direct_pair_actions(h,jac,gradients,k.PAIRS)
    return {"action":action,"support_residual_max":maxres,
            "constant_gradient_max_abs":float(np.max(np.abs(gradients[:,:,6]))),
            "constant_action_max_abs":float(np.max(np.abs(action[:,5]))),
            "argument_antisymmetry_max_abs":anti}


def peak_rss_gib():
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(2**30 if sys.platform=="darwin" else 2**20)


def workers(args):
    if args.workers is None or args.workers < 1: raise ValueError("allocation must choose --workers")
    if args.memory_budget_gib is None or args.worker_memory_gib is None or args.memory_reserve_gib is None:
        raise ValueError("memory-aware pool requires total budget, per-worker estimate, and reserve")
    available=args.memory_budget_gib-args.memory_reserve_gib
    if available<=0 or args.worker_memory_gib<=0: raise ValueError("memory budget/reserve leaves no worker capacity")
    effective=min(args.workers,int(available//args.worker_memory_gib))
    if effective<1: raise ValueError("memory budget cannot safely admit one worker")
    if args.max_tasks_per_worker is None or args.max_tasks_per_worker<1:
        raise ValueError("--max-tasks-per-worker must be positive")
    return effective


def atomic_chunk(output,unit,identity,result):
    path=chunk_path(output,unit)
    save_npz(path,**result["arrays"])
    receipt={"identity":identity,"unit":unit,"sha256":sha256(path),"seconds":result["seconds"],
             "peak_rss_gib":result["peak_rss_gib"],"writer":"parent process"}
    save_json(path.with_suffix(".json"),receipt)
    return receipt


def execute_stage(input_root,output,identity,stage,args):
    effective=workers(args); output=Path(output)
    for n in config()["resolutions"]:
        if stage=="preflight":
            entries=preflight_units(output,n)
            units=[unit_record(stage,n,i,entry) for i,entry in enumerate(entries)]
        else:
            units=[unit_record(stage,n,i) for i in range(math.ceil(n**3/config()["raw_chunk_size"]))]
            entries=None
        todo=[u for u in units if not valid_chunk(output,u,identity)]
        start=time.perf_counter(); receipts=[]
        if todo:
            initargs=(str(input_root),str(output),n)
            with ProcessPoolExecutor(max_workers=effective,mp_context=get_context("spawn"),
                                     initializer=initialize_worker,initargs=initargs,
                                     max_tasks_per_child=args.max_tasks_per_worker) as pool:
                futures={}
                for unit in todo:
                    payload={"stage":stage,"unit":unit}
                    if stage=="preflight": payload["entry"]=entries[unit["index"]]
                    futures[pool.submit(worker_task,payload)]=unit
                for future in as_completed(futures):
                    unit=futures[future]
                    try: result=future.result()
                    except BaseException as exc:
                        save_json(output/"failure.json",{"stage":stage,"unit":unit,"error":repr(exc)})
                        raise
                    receipts.append(atomic_chunk(output,unit,identity,result))
                    save_json(output/"progress.json",{"stage":stage,"n":n,"completed":len(units)-len(todo)+len(receipts),"total":len(units),"unix_time":time.time()})
        execution={"stage":stage,"resolution":n,"requested_workers":args.workers,"effective_workers":effective,
                   "memory_budget_gib":args.memory_budget_gib,"worker_memory_gib":args.worker_memory_gib,
                   "memory_reserve_gib":args.memory_reserve_gib,"max_tasks_per_worker":args.max_tasks_per_worker,
                   "seconds":time.perf_counter()-start,"resumed_units":len(units)-len(todo),"executed_units":len(todo),
                   "peak_worker_rss_gib":max((r["peak_rss_gib"] for r in receipts),default=None)}
        save_json(output/"executions"/f"{stage}_N{n}_{time.time_ns()}.json",execution)
    marker=output/f"{stage}.complete.json"
    save_json(marker,{"identity":identity,"stage":stage,"completed":True,"resolutions":config()["resolutions"]})


def run_preflight(input_root,output,identity,args):
    k,_=load_framework(); output=Path(output)
    summaries={}
    for n in config()["resolutions"]:
        t=k.load_context(n,output/"runtime_inputs")
        records=select_preflight_owners(t,config())
        if not (8 <= len(records) <= config()["preflight_owner_count_limit_per_resolution"]):
            raise ValueError(f"unexpected bounded preflight owner count N{n}: {len(records)}")
        save_json(output/f"N{n}.preflight_owners.json",{"identity":identity,"n":n,"owners":records,
                                                          "full_owner_membership":"workers use all raw cells where raw_owner==owner"})
        summaries[str(n)]={"owner_count":len(records),"raw_member_count":sum(x["member_count"] for x in records),
                           "wall_angle_sample_count":8,"radial_layers_from_wall":config()["preflight_wall_radial_indices_from_end"]}
    execute_stage(input_root,output,identity,"preflight",args)
    validation={"identity":identity,"preflight_complete":True,"sample":summaries,
                "worker_count_memory_capped":True,"actual_omega_specific_tolerance":config()["actual_omega_batched_replay_abs_tolerance"],
                "smooth_reference_tolerance":config()["smooth_reference_replay_abs_tolerance"]}
    omega=[];smooth=[];repeat=[];batch=[];constant=[];bc=[];backends=set()
    for n in config()["resolutions"]:
        for unit in (unit_record("preflight",n,i,e) for i,e in enumerate(preflight_units(output,n))):
            if not valid_chunk(output,unit,identity): raise ValueError(f"preflight chunk missing or stale: {unit}")
            with np.load(chunk_path(output,unit),allow_pickle=False) as z:
                omega.append(float(z["omega_reference_replay_abs"]))
                smooth.append(float(z["smooth_reference_replay_max_abs"]))
                repeat.append(float(z["singleton_repeat_max_abs"]))
                batch.append(float(z["singleton_vs_batched_omega_abs"]))
                constant.append(float(z["constant_action_max_abs"]))
                bc.append(int(z["boundary_conditioned_rows"]))
                backends.add(str(z["cpu_backend"]))
    if backends!={"cpu"}: raise RuntimeError(f"preflight workers did not all use JAX CPU backend: {sorted(backends)}")
    validation["cpu_backend"]=sorted(backends)
    validation["jax_platform_environment"]=os.environ.get("JAX_PLATFORMS")
    validation["cuda_visible_devices"]=os.environ.get("CUDA_VISIBLE_DEVICES")
    validation["max_actual_omega_owner_reference_replay_abs"]=max(omega)
    validation["max_smooth_owner_reference_replay_abs"]=max(smooth)
    validation["max_singleton_repeat_abs"]=max(repeat)
    validation["max_singleton_vs_batched_omega_abs"]=max(batch)
    validation["max_constant_action_abs"]=max(constant)
    validation["boundary_conditioned_raw_rows"]=sum(bc)
    save_json(output/"preflight_validation.json",validation)
    manifest=json.loads((output/"campaign_manifest.json").read_text());manifest["preflight_complete"]=True;save_json(output/"campaign_manifest.json",manifest)


def raw_units(n):
    size=config()["raw_chunk_size"]
    return [unit_record("run",n,i) for i in range(math.ceil(n**3/size))]


def run_campaign(input_root,output,identity,args):
    manifest=json.loads((Path(output)/"campaign_manifest.json").read_text())
    if not manifest.get("preflight_complete"): raise ValueError("the bounded all-resolution preflight must complete before global execution")
    for n in config()["resolutions"]:
        for unit in (unit_record("preflight",n,i,e) for i,e in enumerate(preflight_units(output,n))):
            if not valid_chunk(output,unit,identity): raise ValueError(f"missing preflight checkpoint: {unit}")
    execute_stage(input_root,output,identity,"run",args)
    manifest=json.loads((Path(output)/"campaign_manifest.json").read_text());manifest["run_complete"]=True;save_json(Path(output)/"campaign_manifest.json",manifest)


def reduce_validate(input_root,output,identity):
    try:
        from p05_structured_global import numerics as k
    except ImportError:
        from scripts.p05_structured_global import numerics as k
    output=Path(output);summary={};result_identity={}
    for n in config()["resolutions"]:
        t=k.load_context(n,output/"runtime_inputs")
        with np.load(output/"reuse_inputs"/f"N{n}.reuse.npz",allow_pickle=False) as source:
            ref=source["reference"];volume=source["volume"];oldC=source["old_C"];jump=source["old_U_minus_A"]
            oldO=source["old_exact_input_O"];oldOslots=source["old_exact_input_O_pair_slots"]
            oldOvalid=source["exact_input_valid_slots"][:,2]
        nraw=n**3;raw_action=np.empty((nraw,len(k.PAIRS)));coverage=np.zeros(nraw,dtype=np.uint8)
        maxconst=0.;maxgrad=0.;maxres=0.;maxanti=0.
        for unit in raw_units(n):
            if not valid_chunk(output,unit,identity): raise ValueError(f"missing/incompatible run checkpoint: {unit}")
            with np.load(chunk_path(output,unit),allow_pickle=False) as data:
                ids=data["ids"]
                if not np.array_equal(ids,np.arange(unit["start"],unit["stop"])): raise ValueError("raw checkpoint coverage mismatch")
                if np.any(coverage[ids]): raise ValueError("duplicate raw action coverage")
                raw_action[ids]=data["action"];coverage[ids]=1
                maxconst=max(maxconst,float(data["constant_action_max_abs"]))
                maxgrad=max(maxgrad,float(data["constant_gradient_max_abs"]))
                maxres=max(maxres,float(data["support_residual_max"]))
                maxanti=max(maxanti,float(data["argument_antisymmetry_max_abs"]))
        if not np.all(coverage==1): raise ValueError("raw action coverage has gaps")
        ijk=np.array(np.unravel_index(np.arange(nraw),(n,n,n))).T
        candidate_centered=project_raw_to_owners(raw_action,t.rv,t.ro,t.vol)
        candidate=candidate_variants(candidate_centered,jump)
        error=candidate-ref[:,:,None]
        oldCerror=oldC-ref
        oldOerror=oldO-ref[:,oldOslots]
        old_o_col_by_pair={int(pair):column for column,pair in enumerate(oldOslots)}
        masks,region_info=regional_owner_masks(t.ro,ijk,len(t.vol),n)
        categories={"direct_centered":{},"direct_centered_plus_saved_U_minus_A":{}}
        for variant in range(2):
            name=config()["candidates"][variant]
            signed=error[:,:,variant]
            categories[name]["global"]={"rms":weighted_stats(signed,t.vol)["rms"],"maximum":weighted_stats(signed,t.vol)["maximum"],
                                         "signed_error":None}
            categories[name]["regions"]={region:weighted_stats(signed,t.vol,mask) for region,mask in masks.items()}
        for pair in range(len(k.PAIRS)):
            for variant,name in enumerate(config()["candidates"]):
                categories[name]["global"].setdefault("per_case",{})[k.CASES[pair]]={
                    "rms":float(np.sqrt(np.dot(t.vol,error[:,pair,variant]**2)/t.vol.sum())),
                    "maximum":float(np.max(np.abs(error[:,pair,variant])))}
                for region,mask in masks.items():
                    stat=weighted_stats(error[:,pair:pair+1,variant],t.vol,mask)
                    categories[name]["regions"][region].setdefault("per_case",{})[k.CASES[pair]]={"rms":float(stat["rms"][0]),"maximum":float(stat["maximum"][0])} if stat["rms"] is not None else {"rms":None,"maximum":None}
        old_baseline={"C":{},"exact_input_O":{}}
        for pair,case in enumerate(k.CASES):
            old_baseline["C"][case]={"rms":float(np.sqrt(np.dot(t.vol,oldCerror[:,pair]**2)/t.vol.sum())),"maximum":float(np.max(np.abs(oldCerror[:,pair])))}
            if oldOvalid[pair]:
                old_error=oldOerror[:,old_o_col_by_pair[pair]]
                old_baseline["exact_input_O"][case]={"available":True,"rms":float(np.sqrt(np.dot(t.vol,old_error**2)/t.vol.sum())),"maximum":float(np.max(np.abs(old_error)))}
            else:
                old_baseline["exact_input_O"][case]={"available":False,"reason":"producer exact-input B/C oracle invalid for actual-vorticity pair"}
        save_npz(output/f"N{n}.owner_results.npz",candidate=candidate,centered=candidate_centered,jump=jump,
                 reference=ref,error=error,old_C=oldC,old_C_error=oldCerror,
                 old_exact_input_O=oldO,old_exact_input_O_pair_slots=oldOslots,
                 old_exact_input_O_valid=oldOvalid,raw_owner_volume=t.vol,coverage=coverage,
                 old_exact_input_O_error=oldOerror)
        summary[str(n)]={"owner_count":len(t.vol),"raw_cell_count":nraw,"global":categories,
                         "old_baseline":old_baseline,"regions":region_info,"constant_candidate_max_abs":maxconst,
                         "constant_gradient_max_abs":maxgrad,"support_residual_max":maxres,
                         "pointwise_argument_antisymmetry_max_abs":maxanti,
                         "finite":bool(np.isfinite(candidate).all() and np.isfinite(error).all()),
                         "preflight_reference_noise_is_operational_only":True}
        result_identity[str(n)]={"file":f"N{n}.owner_results.npz","bytes":(output/f"N{n}.owner_results.npz").stat().st_size,"sha256":sha256(output/f"N{n}.owner_results.npz")}
    orders={};regional_orders={}
    primary=[i for i,name in enumerate(config()["cases"]) if name!="constant_control"]
    for variant,name in enumerate(config()["candidates"]):
        orders[name]={};gate=True
        for pair in primary:
            rms=[summary[str(n)]["global"][name]["per_case"][config()["cases"][pair]]["rms"] for n in config()["resolutions"]]
            values=orders_from_rms(np.asarray(rms)[:,None],config()["resolutions"])[:,0]
            order_values=[float(x) if np.isfinite(x) else None for x in values]
            passed=all(x is not None and x>=config()["global_minimum_order"] for x in order_values)
            orders[name][config()["cases"][pair]]={"rms_N32_N48_N64":rms,"orders_N32_N48_N64":order_values,"gate_pass":passed}
            gate &= passed
        orders[name]["global_accuracy_gate_pass"]=bool(gate)
        regional_orders[name]={}
        for region in summary["32"]["global"][name]["regions"]:
            regional_orders[name][region]={}
            for pair in range(len(k.PAIRS)):
                case=k.CASES[pair]
                rms=[summary[str(n)]["global"][name]["regions"][region]["per_case"][case]["rms"] for n in config()["resolutions"]]
                if any(value is None for value in rms):
                    regional_order=[None,None]
                    rebound_32_48=rebound_48_64=None
                else:
                    regional_order=orders_from_rms(np.asarray(rms)[:,None],config()["resolutions"])[:,0]
                    regional_order=[float(x) if np.isfinite(x) else None for x in regional_order]
                    rebound_32_48=bool(rms[1]>rms[0]);rebound_48_64=bool(rms[2]>rms[1])
                regional_orders[name][region][case]={
                    "rms_N32_N48_N64":rms,
                    "orders_N32_N48_N64":regional_order,
                    "rebound_N32_to_N48":rebound_32_48,
                    "rebound_N48_to_N64":rebound_48_64,
                    "qualification_gate":False}
    cfg=config()
    implementation_pass=all(
        summary[str(n)]["finite"]
        and summary[str(n)]["constant_candidate_max_abs"]<=cfg["constant_action_tolerance"]
        and summary[str(n)]["constant_gradient_max_abs"]<=cfg["constant_gradient_tolerance"]
        and summary[str(n)]["support_residual_max"]<=cfg["support_residual_tolerance"]
        and summary[str(n)]["pointwise_argument_antisymmetry_max_abs"]<=cfg["argument_antisymmetry_tolerance"]
        for n in cfg["resolutions"])
    save_json(output/"summary.json",{"identity":identity,"cases":k.CASES,"pairs":k.PAIRS,"fields":k.FIELDS,
                                    "candidate_names":config()["candidates"],"results":summary,"orders":orders,
                                    "regional_orders_and_rebounds":regional_orders,
                                    "accuracy_criterion_frozen_in_config_before_results":True,
                                    "regional_rebound_is_diagnostic_not_a_global_gate":True,
                                    "implementation_pass":implementation_pass,
                                    "preflight_operational_status":json.loads((output/"preflight_validation.json").read_text()),
                                    "scientific_accuracy_gate_pass":{name:orders[name]["global_accuracy_gate_pass"] for name in config()["candidates"]},
                                    "production_qualified":False,"conservation_or_nonlinear_stability_claimed":False})
    save_json(output/"result_manifest.json",{"identity":identity,"owner_results":result_identity})
    completion={"identity":identity,"raw_chunk_coverage_complete":True,"results_written":True,
                "operational_validation":"passed","scientific_accuracy_gate_pass":{name:orders[name]["global_accuracy_gate_pass"] for name in config()["candidates"]},
                "interpretation":"not performed; local task will analyze returned outputs"}
    save_json(output/"completion.json",completion)


def lock(output):
    path=Path(output)/".campaign.lock";path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open("a")
    try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as exc: stream.close();raise RuntimeError("another campaign command owns this output folder") from exc
    return stream


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("verify-inputs","preflight","run","validate"))
    parser.add_argument("--input-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int)
    parser.add_argument("--memory-budget-gib",type=float)
    parser.add_argument("--worker-memory-gib",type=float)
    parser.add_argument("--memory-reserve-gib",type=float)
    parser.add_argument("--max-tasks-per-worker",type=int,default=32)
    return parser.parse_args()


def main(args=None):
    args=parse_args() if args is None else args
    args.input_root=Path(args.input_root).resolve();args.output=Path(args.output).resolve()
    output_lock=lock(args.output)
    try:
        if args.command=="verify-inputs":
            identity=verify_inputs(args.input_root,args.output)
            print(json.dumps({"verified":True,"identity":identity,"output":str(args.output)},flush=True))
            return
        identity=current_identity(args.output)
        manifest=json.loads((args.output/"campaign_manifest.json").read_text())
        if args.command=="preflight":
            run_preflight(args.input_root,args.output,identity,args)
        elif args.command=="run":
            run_campaign(args.input_root,args.output,identity,args)
        else:
            if not manifest.get("preflight_complete") or not manifest.get("run_complete"):
                raise ValueError("validation requires completed preflight and run stages")
            reduce_validate(args.input_root,args.output,identity)
    finally:
        output_lock.close()


if __name__=="__main__": main()
