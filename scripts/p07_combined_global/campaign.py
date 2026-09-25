#!/usr/bin/env python3
"""Gated, resumable, node-local CPU campaign for combined P07 q3 diffusion."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from multiprocessing import get_context
from contextlib import contextmanager
from functools import lru_cache

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
sys.dont_write_bytecode = True

import numpy as np
import candidate
import kernels as k
import topology

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGES = ("support", "observations", "assembly", "reference", "control")
STATE = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, default=encode, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def encode(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def source_identity():
    paths = [HERE / name for name in ("campaign.py", "candidate.py", "kernels.py", "topology.py", "configuration.json", "input_manifest.json")]
    paths += [REPO / name for name in ("scripts/p07_diffusion_global/numerics.py", "hsx_mms_continuum_reference.py", "src/drbx/geometry/fci_perpendicular_bracket.py", "src/drbx/geometry/fci_boundary_functional_reconstruction.py")]
    return {str(p.relative_to(REPO)): sha(p) for p in paths}


def settings():
    cfg=json.loads((HERE / "configuration.json").read_text())
    if (cfg.get("schema") != "drbx.p07-combined-global-v2" or cfg.get("reference_volume_quadrature") != 1
        or cfg.get("candidate_face_quadrature") != 3 or cfg.get("control_volume_quadratures") != [1]
        or cfg.get("reference_convention") != "physical_raw_volume_midpoint_projection"
        or cfg.get("integrated_reference_controls") is not False):
        raise ValueError("unsupported midpoint diffusion contract")
    return cfg


def required_inputs():
    return json.loads((HERE / "input_manifest.json").read_text())


def verify_inputs(input_root, output):
    input_root = Path(input_root).resolve(); output = Path(output).resolve()
    cfg, manifest = settings(), required_inputs()
    for item in manifest["files"]:
        path = input_root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"missing or changed immutable input: {path}")
    sources = source_identity()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    identity = digest({"config": cfg, "inputs": manifest, "sources": sources, "commit": commit})
    destination = output / "campaign_manifest.json"
    if destination.exists() and json.loads(destination.read_text())["identity"] != identity:
        raise ValueError("campaign identity changed; use a new output folder")
    sidecar = json.loads((input_root / "DRBX/work/perpendicular_second_order_hsx_p01_p03/continuous_reference_sidecar.json").read_text())
    sidecar["metric_cache"]["path"] = str((input_root / "hsx_metric_d58d392545fd3917efeb83b6.npz").resolve())
    sidecar["makegrid"]["path"] = str((input_root / "mgrid_res2p5cm_180pln.nc").resolve())
    sidecar["artifact"]["path"] = str((input_root / "prototype_runs/geometry/hsx_fci_64x64x64").resolve())
    sidecar["metric_query_batch_size"] = 4096
    localized = output / "reference_sidecar.json"
    if localized.exists():
        if json.loads(localized.read_text()) != sidecar: raise ValueError("localized reference sidecar changed")
    else: atomic_json(localized, sidecar)
    if not destination.exists():
        atomic_json(destination, {"identity": identity, "configuration": cfg, "input_manifest": manifest,
                                  "source_hashes": sources, "commit": commit, "input_root": str(input_root),
                                  "localized_sidecar_sha256": sha(localized),
                                  "python": sys.version, "platform": platform.platform()})
    for name in ("scratch", "cache", "logs", "chunks"):
        (output / name).mkdir(exist_ok=True)
    return identity


def current_identity(output):
    path = Path(output) / "campaign_manifest.json"
    if not path.exists(): raise ValueError("verify-inputs must run first")
    data = json.loads(path.read_text())
    if data["source_hashes"] != source_identity() or data["configuration"] != settings() or data["input_manifest"] != required_inputs():
        raise ValueError("campaign source or configuration changed")
    if sha(Path(output) / "reference_sidecar.json") != data["localized_sidecar_sha256"]:
        raise ValueError("localized reference sidecar changed")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    if commit != data["commit"]: raise ValueError("campaign source commit changed")
    return data["identity"]


@contextmanager
def locked(output):
    Path(output).mkdir(parents=True, exist_ok=True)
    with (Path(output) / ".campaign.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def topology_stage(input_root, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    identity = current_identity(output)
    previous = output / "topology_summary.json"
    if previous.exists():
        saved = json.loads(previous.read_text())
        if saved.get("identity") != identity: raise ValueError("stale topology identity")
        if all(sha(output / f"N{n}.topology.npz") == saved["topology_sha256"][str(n)]
               for n in settings()["resolutions"]):
            return
        raise ValueError("corrupt or incomplete topology checkpoint")
    k.configure(input_root)
    summary = {}
    for n in settings()["resolutions"]:
        summary[str(n)] = topology.census(n, input_root, output)
        t = k.load(n)
        owners, labels = k.num.select_owners(t.g, t.centers)
        raw_ids = np.flatnonzero(np.isin(t.ro, owners))
        atomic_npz(output / f"N{n}.control_selection.npz", owner_ids=owners, raw_ids=raw_ids)
        atomic_json(output / f"N{n}.control_selection.json", {"owner_ids": owners, "labels": labels,
                                                               "raw_count": len(raw_ids)})
    atomic_json(previous, {"identity": identity, "resolutions": summary,
                           "topology_sha256": {str(n): sha(output / f"N{n}.topology.npz") for n in settings()["resolutions"]},
                           "collapsed_rows_included_in_face_totals": True})


@lru_cache(maxsize=8)
def load_topology(n, output):
    with np.load(Path(output) / f"N{n}.topology.npz", allow_pickle=False) as z:
        return z["face_ids"].copy(), z["family"].copy(), z["endpoints"].copy()


@lru_cache(maxsize=32)
def selected_ids(stage, n, output):
    output = Path(output)
    if stage in ("support", "assembly"):
        ids, family, _ = load_topology(n, output)
        return ids[np.isin(family, (6, 7))] if stage == "support" else ids
    if stage == "control":
        with np.load(output / f"N{n}.control_selection.npz") as z: return z["raw_ids"]
    return np.arange(n**3, dtype=np.int64)


def create_plan(output):
    output = Path(output); cfg = settings(); ident = current_identity(output)
    sizes = {"support": cfg["support_chunk"], "assembly": cfg["assembly_chunk"],
             "observations": cfg["observation_chunk"], "reference": cfg["reference_chunk"],
             "control": cfg["control_chunk"]}
    plan = {"identity": ident, "stages": {}}
    for stage in STAGES:
        per_n = {}
        for n in cfg["resolutions"]:
            ids = selected_ids(stage, n, output)
            per_n[str(n)] = [{"n": n, "stage": stage, "start": start,
                              "stop": min(start + sizes[stage], len(ids))}
                             for start in range(0, len(ids), sizes[stage])]
        plan["stages"][stage] = per_n
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("existing plan differs from frozen configuration")
    atomic_json(path, plan)
    return plan


def unit_path(output, unit):
    return Path(output) / "chunks" / f"N{unit['n']}" / unit["stage"] / f"{unit['start']:07d}-{unit['stop']-1:07d}.npz"


def unit_ids(unit, output):
    return selected_ids(unit["stage"], unit["n"], output)[unit["start"]:unit["stop"]]


def valid_unit(output, unit, ident):
    path = unit_path(output, unit); receipt = path.with_suffix(".json")
    if not path.exists() and not receipt.exists(): return False
    if not receipt.exists(): return False
    data = json.loads(receipt.read_text())
    if data["identity"] != ident or data["unit"] != unit: raise ValueError(f"stale receipt: {receipt}")
    if not path.exists() or sha(path) != data["sha256"]: raise ValueError(f"corrupt chunk: {path}")
    ids = unit_ids(unit, output)
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z["ids"], ids): raise ValueError(f"wrong chunk IDs: {path}")
        for name in z.files:
            if z[name].dtype.kind in "fc" and not np.isfinite(z[name]).all():
                raise ValueError(f"nonfinite chunk {path}:{name}")
        if unit["stage"] == "support" and (len(z["supported"]) != len(ids) or len(z["max_residual"]) != len(ids)):
            raise ValueError(f"invalid support shape: {path}")
        if unit["stage"] == "support":
            if len(z["row_offsets"]) != len(ids) + 1 or z["row_offsets"][0] != 0 or not np.all(np.diff(z["row_offsets"]) >= 0):
                raise ValueError(f"invalid support row offsets: {path}")
            if z["row_offsets"][-1] != len(z["donor_ids"]) or len(z["donor_ids"]) != len(z["coefficients"]):
                raise ValueError(f"invalid support row coefficients: {path}")
        if unit["stage"] == "assembly":
            if len(z["family"]) != len(ids) or len(z["endpoints"]) != len(ids) or len(z["flux"]) != len(ids) or len(z["oracle_flux"]) != len(ids):
                raise ValueError(f"invalid assembly shape: {path}")
            if len(z["row_offsets"]) != len(ids) + 1 or z["row_offsets"][0] != 0 or not np.all(np.diff(z["row_offsets"]) >= 0):
                raise ValueError(f"invalid row offsets: {path}")
            if z["row_offsets"][-1] != len(z["donor_ids"]) or len(z["donor_ids"]) != len(z["coefficients"]) or len(z["coefficients"]) != len(z["BC_value_coefficients"]):
                raise ValueError(f"invalid assembly coefficient lengths: {path}")
            bc = z["BC_face_positions"]
            if len(z["BC_tangential"]) != len(bc) or z["BC_tangential"].shape[1:] != (9, 2) or z["BC_trace_target_points"].shape != (len(bc), 9, 3):
                raise ValueError(f"invalid BC channels: {path}")
            if z["BC_trace_donor_points"].shape != (len(z["coefficients"]), 3):
                raise ValueError(f"invalid BC donor traces: {path}")
            face_ids, family, endpoints = load_topology(unit["n"], output)
            positions = np.searchsorted(face_ids, ids)
            if not np.array_equal(z["family"], family[positions]) or not np.array_equal(z["endpoints"], endpoints[positions]):
                raise ValueError(f"assembly topology changed: {path}")
            if not np.array_equal(bc, np.flatnonzero(np.isin(z["family"], (1, 2, 4)))):
                raise ValueError(f"missing BC face channel: {path}")
            for j, label in enumerate(z["family"]):
                a, b = int(z["row_offsets"][j]), int(z["row_offsets"][j+1])
                if label == 0 and (a != b or np.any(z["flux"][j] != 0) or np.any(z["oracle_flux"][j] != 0)):
                    raise ValueError(f"nonzero collapsed face: {path}")
                if label in (1, 2, 4) and not np.allclose(z["BC_value_coefficients"][a:b], -z["coefficients"][a:b], rtol=0, atol=1e-12):
                    raise ValueError(f"invalid BC value channel: {path}")
        if unit["stage"] in ("observations", "reference", "control"):
            required = {"owner_ids", "numerator"} if unit["stage"] == "observations" else {"owner_ids", "numerator_q1", "volume_q1"}
            if not required.issubset(z.files) or len(z["owner_ids"]) != len(ids): raise ValueError(f"invalid cell chunk: {path}")
    return True


def prerequisite(output, stage):
    needed = {"support": (), "observations": ("support",), "assembly": ("support", "observations"),
              "reference": ("assembly",), "control": ("reference",)}[stage]
    hashes = {}
    for name in needed:
        path = Path(output) / f"{name}_reduction.json"
        if not path.exists(): raise ValueError(f"{stage} requires complete {name} reduction")
        data = json.loads(path.read_text())
        if not data.get("all_complete") or (name == "support" and data.get("unsupported_count", 1)):
            raise ValueError(f"{stage} blocked by {name} reduction")
        hashes[name] = sha(path)
    return hashes


def stage_identity(output, stage):
    return digest({"campaign": current_identity(output), "stage": stage,
                   "prerequisites": prerequisite(output, stage)})


def initialize_worker(input_root, output, stage, n, identity):
    global STATE
    output = Path(output); input_root = Path(input_root)
    os.environ["DRBX_CACHE_DIR"] = str(output / "cache" / "jax")
    os.environ["XDG_CACHE_HOME"] = str(output / "cache")
    os.environ["TMPDIR"] = str(output / "scratch")
    k.configure(input_root)
    t = k.load(n)
    ref = k.num.reference(output / "reference_sidecar.json", verify_hashes=False)
    values = None
    if stage == "assembly":
        path = output / "owner_values.npz"
        receipt = json.loads((output / "observations_reduction.json").read_text())
        if sha(path) != receipt["owner_values_sha256"]: raise ValueError("changed owner observations")
        with np.load(path, allow_pickle=False) as z:
            values = z[f"N{n}"].copy()
        if values.shape != (len(t.vol), 4) or not np.isfinite(values).all():
            raise ValueError("invalid owner observations")
    STATE = {"t": t, "ref": ref, "values": values, "output": output, "stage": stage,
             "n": n, "identity": identity,
             "reference_context": {"geometry": t.g, "faces": t.faces, "reference": ref}}
    if stage == "assembly":
        plan = json.loads((output / "plan.json").read_text())
        STATE["support_units"] = plan["stages"]["support"][str(n)]
        STATE["support_identity"] = stage_identity(output, "support")
        STATE["support_cache"] = {}


def prepared_support_rows(face_ids, family, state):
    needed = np.asarray(face_ids)[np.isin(family, (6, 7))]
    if not len(needed): return {}
    all_ids = selected_ids("support", state["n"], state["output"])
    positions = np.searchsorted(all_ids, needed)
    if not np.array_equal(all_ids[positions], needed): raise ValueError("assembly face missing support certificate")
    size = settings()["support_chunk"]
    result = {}
    for chunk_number in np.unique(positions // size):
        unit = state["support_units"][int(chunk_number)]
        if chunk_number not in state["support_cache"]:
            if not valid_unit(state["output"], unit, state["support_identity"]):
                raise ValueError("missing verified support row")
            with np.load(unit_path(state["output"], unit), allow_pickle=False) as z:
                state["support_cache"][int(chunk_number)] = (z["ids"].copy(), z["row_offsets"].copy(),
                                                              z["donor_ids"].copy(), z["coefficients"].copy())
            if len(state["support_cache"]) > 8:
                state["support_cache"].pop(next(iter(state["support_cache"])))
        source_ids, offsets, donors, coefficients = state["support_cache"][int(chunk_number)]
        for fid in needed[positions // size == chunk_number]:
            local = int(np.searchsorted(source_ids, fid))
            if local >= len(source_ids) or source_ids[local] != fid: raise ValueError("support row face mismatch")
            a, b = int(offsets[local]), int(offsets[local + 1])
            result[int(fid)] = (donors[a:b], coefficients[a:b])
    return result


def boundary_data(ref, rows):
    donor_parts = [bc["trace_donor_points"] for ids, _, bc in rows if bc is not None]
    target_parts = [bc["trace_target_points"] for ids, _, bc in rows if bc is not None]
    donor_points = np.concatenate(donor_parts) if donor_parts else np.empty((0, 3))
    target_points = np.concatenate(target_parts) if target_parts else np.empty((0, 3))
    donor_value = np.empty((len(donor_points), 4))
    target_gradient = np.empty((len(target_points), 4, 3))
    for start in range(0, len(donor_points), 4096):
        stop = min(start + 4096, len(donor_points))
        donor_value[start:stop] = k.num.fields(ref, donor_points[start:stop])[0]
    for start in range(0, len(target_points), 4096):
        stop = min(start + 4096, len(target_points))
        target_gradient[start:stop] = k.num.fields(ref, target_points[start:stop])[1]
    flux = np.zeros((len(rows), 4)); donor_cursor = target_cursor = 0
    for j, (ids, _, bc) in enumerate(rows):
        if bc is None: continue
        count = len(ids); qcount = len(bc["trace_target_points"])
        flux[j] = bc["value_loading"] @ donor_value[donor_cursor:donor_cursor + count]
        flux[j] += np.einsum("qa,qfa->f", bc["tangential_loading"],
                             target_gradient[target_cursor:target_cursor + qcount, :, 1:])
        donor_cursor += count; target_cursor += qcount
    assert donor_cursor == len(donor_points) and target_cursor == len(target_points)
    return flux


def reference_cells(t,ref,ids,order=1):
    """Pointwise analytic diffusion projected using physical raw volumes."""
    result=k.num.cell_chunk({"geometry":t.g,"faces":t.faces,"reference":ref},ids,order)
    if order==1:
        point=result["numerator"]/result["continuous_volume"][:,None]
        result["numerator"]=t.rv[ids,None]*point
        result["continuous_volume"]=t.rv[ids].copy()
    return result


def reference_orders(stage):
    return (1,) if stage=="reference" else (1,"1_halfstep")


def compute_unit(unit):
    s = STATE; t = s["t"]; ref = s["ref"]; stage = unit["stage"]
    ids = unit_ids(unit, s["output"]); begun = time.monotonic()
    if stage in ("support", "assembly"):
        all_ids, all_family, all_endpoints = load_topology(t.n, s["output"])
        positions = np.searchsorted(all_ids, ids)
        family = all_family[positions]
        endpoints = all_endpoints[positions]
    if stage == "support":
        outcome, cost = candidate.support_execute(t, ref, ids, family)
        offsets = np.r_[0, np.cumsum([len(x["row_ids"]) for x in outcome])]
        data = {"ids": ids, "family": family,
                "supported": np.asarray([x["supported"] for x in outcome]),
                "full_rank": np.asarray([x["full_rank"] for x in outcome]),
                "max_residual": np.asarray([x["max_residual"] for x in outcome]),
                "min_actual_rank": np.asarray([x["min_actual_rank"] for x in outcome]),
                "min_uniform_rank": np.asarray([x["min_uniform_rank"] for x in outcome]),
                "cubic_expansion": np.asarray([x["cubic_expansion"] for x in outcome]),
                "quartic_expansion": np.asarray([x["quartic_expansion"] for x in outcome]),
                "max_plane_donors": np.asarray([x["max_plane_donors"] for x in outcome]),
                "row_offsets": offsets,
                "donor_ids": np.concatenate([x["row_ids"] for x in outcome]).astype(np.int32),
                "coefficients": np.concatenate([x["row_coefficients"] for x in outcome])}
    elif stage == "observations":
        ijk = np.array(np.unravel_index(ids, (t.n,) * 3)).T
        points = np.column_stack([t.centers[a][ijk[:, a]] for a in range(3)])
        values = k.num.fields(ref, points)[0]
        data = {"ids": ids, "owner_ids": t.ro[ids], "numerator": t.rv[ids, None] * values}
        cost = {"faces": len(ids)}
    elif stage in ("reference", "control"):
        orders = reference_orders(stage)
        data = {"ids": ids, "owner_ids": t.ro[ids]}
        for order in orders:
            original_step=ref.finite_difference_step
            try:
                ref.finite_difference_step=settings()["reference_control_step"] if order=="1_halfstep" else settings()["reference_step"]
                result=reference_cells(t,ref,ids,1 if order=="1_halfstep" else order)
            finally:ref.finite_difference_step=original_step
            data[f"numerator_q{order}"] = result["numerator"]
            data[f"volume_q{order}"] = result["continuous_volume"]
        cost = {"cells": len(ids)}
    else:
        prepared = prepared_support_rows(ids, family, s)
        rows, cost, points, integ = candidate.assembly_rows(t, ref, ids, family, prepared)
        offsets = np.r_[0, np.cumsum([len(x[0]) for x in rows])]
        donors = np.concatenate([x[0] for x in rows]).astype(np.int32)
        coef = np.concatenate([x[1] for x in rows])
        bcidx = np.flatnonzero(np.isin(family, (1, 2, 4)))
        bcvalue = np.concatenate([x[2]["value_loading"] if x[2] is not None else np.zeros(len(x[1])) for x in rows])
        bct = np.stack([rows[j][2]["tangential_loading"] for j in bcidx]) if len(bcidx) else np.empty((0, 9, 2))
        bctp = np.stack([rows[j][2]["trace_target_points"] for j in bcidx]) if len(bcidx) else np.empty((0, 9, 3))
        bcdp = np.concatenate([x[2]["trace_donor_points"] if x[2] is not None else np.zeros((len(x[1]), 3)) for x in rows])
        flux = np.array([coef[offsets[j]:offsets[j + 1]] @ s["values"][rows[j][0]] for j in range(len(rows))])
        flux += boundary_data(ref, rows)
        oracle = np.zeros((len(ids), 4)); active = np.flatnonzero(family != 0)
        for start in range(0, len(active), 256):
            selected = active[start:start + 256]
            grads = k.num.fields(ref, points[selected].reshape(-1, 3))[1].reshape(len(selected), 9, 4, 3)
            oracle[selected] = np.einsum("fqa,fqja->fj", integ[selected], grads)
        data = {"ids": ids, "family": family, "endpoints": endpoints,
                "row_offsets": offsets, "donor_ids": donors, "coefficients": coef,
                "BC_value_coefficients": bcvalue, "BC_trace_donor_points": bcdp,
                "BC_face_positions": bcidx, "BC_tangential": bct,
                "BC_trace_target_points": bctp, "flux": flux, "oracle_flux": oracle}
    path = unit_path(s["output"], unit)
    atomic_npz(path, **data)
    receipt = {"identity": s["identity"], "unit": unit, "sha256": sha(path),
               "seconds": time.monotonic() - begun, "peak_rss_gib": k.rss(), "pid": os.getpid(), "cost": cost}
    atomic_json(path.with_suffix(".json"), receipt)
    if not valid_unit(s["output"], unit, s["identity"]): raise ValueError("just-written unit failed validation")
    return {"seconds": receipt["seconds"], "peak_rss_gib": receipt["peak_rss_gib"]}


def effective_workers(args):
    if args.workers is None or args.workers < 1: raise ValueError("--workers must be chosen by remote allocation setup")
    count = args.workers
    if args.memory_budget_gib is not None:
        if args.worker_memory_gib is None or args.worker_memory_gib <= 0: raise ValueError("memory budget requires --worker-memory-gib")
        count = min(count, int((args.memory_budget_gib - args.memory_reserve_gib) // args.worker_memory_gib))
        if count < 1: raise ValueError("memory budget cannot fit one worker and reserve")
    return count


def run_stage(args, stage, plan):
    output = args.output; ident = stage_identity(output, stage); effective = effective_workers(args)
    for n in settings()["resolutions"]:
        units = plan["stages"][stage][str(n)]
        todo = [unit for unit in units if not valid_unit(output, unit, ident)]
        records = []; started = time.monotonic()
        if todo:
            with ProcessPoolExecutor(max_workers=effective, mp_context=get_context("spawn"),
                                     initializer=initialize_worker,
                                     initargs=(str(args.input_root), str(output), stage, n, ident),
                                     max_tasks_per_child=args.max_tasks_per_worker) as pool:
                iterator = iter(todo); pending = {}
                def submit():
                    for _ in range(max(0, 2 * effective - len(pending))):
                        unit = next(iterator, None)
                        if unit is None: break
                        pending[pool.submit(compute_unit, unit)] = unit
                submit()
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        unit = pending.pop(future)
                        try: records.append(future.result())
                        except Exception as error:
                            atomic_json(output / "failure.json", {"stage": stage, "n": n, "unit": unit,
                                                                   "error": repr(error), "unix_time": time.time()})
                            raise
                    atomic_json(output / "progress.json", {"stage": stage, "n": n,
                                                               "complete": len(units) - len(todo) + len(records),
                                                               "total": len(units), "unix_time": time.time()})
                    submit()
        atomic_json(output / f"N{n}.{stage}.execution.json", {"identity": ident, "requested_workers": args.workers,
                     "effective_workers": effective, "memory_budget_gib": args.memory_budget_gib,
                     "worker_memory_gib": args.worker_memory_gib, "memory_reserve_gib": args.memory_reserve_gib,
                     "resumed_units": len(units) - len(todo), "executed_units": len(todo),
                     "seconds": time.monotonic() - started,
                     "peak_worker_rss_gib": max([x["peak_rss_gib"] for x in records], default=0)})
        record = json.loads((output / f"N{n}.{stage}.execution.json").read_text())
        atomic_json(output / "executions" / f"{time.time_ns()}_N{n}_{stage}.json", record)


def reduce_stage(input_root, output, stage, plan):
    output = Path(output); ident = stage_identity(output, stage); k.configure(input_root)
    summary = {"identity": ident, "stage": stage, "resolutions": {}, "all_complete": True}
    owner_values = {}; reference = {}; controls = {}; unsupported_total = 0
    for n in settings()["resolutions"]:
        units = plan["stages"][stage][str(n)]
        expected = selected_ids(stage, n, output); cursor = 0; t = k.load(n)
        if stage == "observations":
            accum = np.zeros((len(t.vol), 4)); covered_volume = np.zeros(len(t.vol))
        if stage in ("reference", "control"):
            orders = reference_orders(stage)
            numerator = {q: np.zeros((len(t.vol), 4)) for q in orders}
            physical_volume = {q: np.zeros(len(t.vol)) for q in orders}
        bad = []; max_residual = 0.; rank_deficient = 0; nnz = 0; boundary_faces = 0
        with np.load(output / f"N{n}.topology.npz", allow_pickle=False) as top:
            face_ids, face_family, face_endpoints = top["face_ids"], top["family"], top["endpoints"]
        for unit in units:
            if unit["start"] != cursor or unit["stop"] <= cursor:
                raise ValueError(f"missing or duplicate {stage} plan coverage at N{n}:{cursor}")
            if not valid_unit(output, unit, ident): raise ValueError(f"missing {stage} chunk: {unit}")
            path = unit_path(output, unit)
            with np.load(path, allow_pickle=False) as z:
                ids = z["ids"]
                if not np.array_equal(ids, expected[cursor:unit["stop"]]):
                    raise ValueError(f"noncontiguous or duplicate {stage} IDs: {path}")
                if stage == "support":
                    positions = np.searchsorted(face_ids, ids)
                    if not np.array_equal(z["family"], face_family[positions]): raise ValueError("support class mismatch")
                    bad.extend(map(int, ids[~z["supported"]]))
                    max_residual = max(max_residual, float(np.max(z["max_residual"])))
                    rank_deficient += int(np.sum(~z["full_rank"]))
                    if np.any(z["max_residual"] > settings()["target_residual_tolerance"]):
                        bad.extend(map(int, ids[z["max_residual"] > settings()["target_residual_tolerance"]]))
                    if np.any(z["cubic_expansion"] > settings()["original_cubic_max_expansions"]):
                        raise ValueError("cubic expansion exceeds frozen policy")
                    if np.any(z["quartic_expansion"] > settings()["quartic_additional_max_expansions"]):
                        raise ValueError("quartic expansion exceeds frozen policy")
                    if np.any((z["donor_ids"] < 0) | (z["donor_ids"] >= len(t.vol))):
                        raise ValueError("invalid support-row donor")
                    for j in np.flatnonzero(z["supported"]):
                        a, b = int(z["row_offsets"][j]), int(z["row_offsets"][j+1])
                        if a == b or abs(float(np.sum(z["coefficients"][a:b]))) > 1e-8:
                            raise ValueError("invalid supported face row")
                elif stage == "assembly":
                    positions = np.searchsorted(face_ids, ids)
                    fam = face_family[positions]
                    if not np.array_equal(z["family"], fam) or not np.array_equal(z["endpoints"], face_endpoints[positions]):
                        raise ValueError("assembly topology mismatch")
                    offsets, coeff, bcval = z["row_offsets"], z["coefficients"], z["BC_value_coefficients"]
                    bcidx = np.flatnonzero(np.isin(fam, (1, 2, 4)))
                    if not np.array_equal(z["BC_face_positions"], bcidx): raise ValueError("missing or extra BC face")
                    if np.any((z["donor_ids"] < 0) | (z["donor_ids"] >= len(t.vol))): raise ValueError("invalid donor ID")
                    for j, label in enumerate(fam):
                        a, b = int(offsets[j]), int(offsets[j + 1])
                        if label == 0:
                            if b != a or not np.all(z["flux"][j] == 0) or not np.all(z["oracle_flux"][j] == 0):
                                raise ValueError("collapsed-axis row is not exactly zero")
                        elif label in (1, 2, 4):
                            if not np.allclose(bcval[a:b], -coeff[a:b], rtol=0, atol=1e-12):
                                raise ValueError("BC value channel is not minus state row")
                        elif not np.all(bcval[a:b] == 0):
                            raise ValueError("unexpected BC value channel")
                        if label != 0 and abs(float(np.sum(coeff[a:b] + bcval[a:b]))) > 1e-8:
                            raise ValueError("constant action is not zero")
                    nnz += len(coeff); boundary_faces += len(bcidx)
                elif stage == "observations":
                    if not np.array_equal(z["owner_ids"], t.ro[ids]): raise ValueError("observation owner mismatch")
                    np.add.at(accum, z["owner_ids"], z["numerator"])
                    np.add.at(covered_volume, z["owner_ids"], t.rv[ids])
                else:
                    if not np.array_equal(z["owner_ids"], t.ro[ids]): raise ValueError("reference owner mismatch")
                    for q in orders:
                        if z[f"numerator_q{q}"].shape != (len(ids), 4) or z[f"volume_q{q}"].shape != (len(ids),):
                            raise ValueError("reference shape mismatch")
                        np.add.at(numerator[q], z["owner_ids"], z[f"numerator_q{q}"])
                        np.add.at(physical_volume[q], z["owner_ids"], z[f"volume_q{q}"])
            cursor = unit["stop"]
        if cursor != len(expected): raise ValueError(f"incomplete {stage} coverage N{n}")
        info = {"chunks": len(units), "ids": cursor, "complete": True}
        if stage == "support":
            info.update(unsupported_face_ids=sorted(set(bad)), rank_deficient_target_checked=rank_deficient,
                        max_residual=max_residual)
            unsupported_total += len(set(bad))
        elif stage == "assembly": info.update(state_nnz=nnz, BC_faces=boundary_faces)
        elif stage == "observations":
            if not np.allclose(covered_volume, t.vol, atol=1e-12, rtol=1e-12):
                raise ValueError("owner physical-volume coverage mismatch")
            owner_values[f"N{n}"] = accum / t.vol[:, None]
            if not np.isfinite(owner_values[f"N{n}"]).all(): raise ValueError("nonfinite owner values")
            info["owners"] = len(t.vol)
        else:
            if np.any(physical_volume[1] <= 0) and stage == "reference":
                raise ValueError("nonpositive continuous volume")
            if stage == "reference":
                reference[f"N{n}.numerator_q1"] = numerator[1]
                reference[f"N{n}.volume_q1"] = physical_volume[1]
            else:
                with np.load(output / f"N{n}.control_selection.npz", allow_pickle=False) as z:
                    owners = z["owner_ids"]
                for q in orders:
                    if np.any(physical_volume[q][owners] <= 0): raise ValueError("incomplete control owner")
                    controls[f"N{n}.average_q{q}"] = numerator[q][owners] / physical_volume[q][owners, None]
                    controls[f"N{n}.volume_q{q}"] = physical_volume[q][owners]
                controls[f"N{n}.owner_ids"] = owners
                info["sample_owners"] = len(owners)
        summary["resolutions"][str(n)] = info
    if stage == "support": summary["unsupported_count"] = unsupported_total
    if stage == "observations":
        path = output / "owner_values.npz"; atomic_npz(path, **owner_values); summary["owner_values_sha256"] = sha(path)
    if stage == "reference":
        path = output / "reference_midpoint.npz"; atomic_npz(path, **reference); summary["reference_sha256"] = sha(path)
    if stage == "control":
        path = output / "reference_controls.npz"; atomic_npz(path, **controls); summary["controls_sha256"] = sha(path)
    atomic_json(output / f"{stage}_reduction.json", summary)
    if unsupported_total: raise ValueError(f"{unsupported_total} unsupported support faces; dependent stages stopped")
    return summary


def region_masks(t, face_endpoints, face_family):
    n = t.n; radius = t.g.owner_flat_ids // (n * n)
    masks = k.num.masks(t.g)
    masks["axis_core"] = radius == 0
    masks["first_ring"] = radius == 1
    coupled = np.zeros(len(t.vol), bool); ringwise = np.zeros(len(t.vol), bool)
    for label, destination in ((7, coupled), (6, ringwise)):
        pairs = face_endpoints[face_family == label]
        destination[pairs[pairs >= 0]] = True
    masks["coupled_region"] = coupled
    masks["ringwise_region"] = ringwise
    masks["coupled_ringwise_join"] = coupled & ringwise
    masks["aggregate_interface"] = masks["transition"]
    masks["physical_wall"] = masks["boundary"]
    theta_raw = np.zeros((n, n, n), bool); theta_raw[:, (0, n-1), :] = True
    eta_raw = np.zeros((n, n, n), bool); eta_raw[:, :, (0, n-1)] = True
    masks["theta_seam"] = np.bincount(t.ro, weights=theta_raw.ravel(), minlength=len(t.vol)) > 0
    masks["eta_seam"] = np.bincount(t.ro, weights=eta_raw.ravel(), minlength=len(t.vol)) > 0
    return masks


def field_stats(error, volume, masks):
    e2 = volume * error * error; total = float(np.sum(e2)); V = float(np.sum(volume))
    out = {"l2": float(np.sqrt(total / V)), "max_abs": float(np.max(np.abs(error))),
           "squared_error_integral": total, "regions": {}}
    for label, mask in masks.items():
        local = float(np.sum(e2[mask])); vol = float(np.sum(volume[mask]))
        out["regions"][label] = {"owners": int(np.sum(mask)), "physical_volume": vol,
                                  "l2": float(np.sqrt(local / vol)) if vol > 0 else None,
                                  "max_abs": float(np.max(np.abs(error[mask]))) if mask.any() else None,
                                  "global_squared_error_fraction": local / max(total, 1e-300)}
    return out


def summarize(input_root, output, plan):
    output = Path(output); cfg = settings(); base_identity = current_identity(output)
    for stage in STAGES:
        path = output / f"{stage}_reduction.json"
        if not path.exists(): raise ValueError(f"missing {stage} reduction")
        record = json.loads(path.read_text())
        if not record["all_complete"] or record["identity"] != stage_identity(output, stage):
            raise ValueError(f"invalid {stage} reduction")
        if stage == "support" and record["unsupported_count"]: raise ValueError("unsupported supports")
    with np.load(output / "reference_midpoint.npz", allow_pickle=False) as reference:
        reference_data = {name: reference[name].copy() for name in reference.files}
    with np.load(output / "reference_controls.npz", allow_pickle=False) as controls:
        control_data = {name: controls[name].copy() for name in controls.files}
    k.configure(input_root); cases = {}; arrays_hashes = {}; assembly_ident = stage_identity(output, "assembly")
    for n in cfg["resolutions"]:
        t = k.load(n); total = np.zeros((len(t.vol), 4)); exact = np.zeros_like(total)
        with np.load(output / f"N{n}.topology.npz", allow_pickle=False) as z:
            all_endpoints, all_family = z["endpoints"].copy(), z["family"].copy()
        boundary_net = np.zeros(4)
        for unit in plan["stages"]["assembly"][str(n)]:
            if not valid_unit(output, unit, assembly_ident):
                raise ValueError("missing assembly data during summary")
            with np.load(unit_path(output, unit), allow_pickle=False) as z:
                pairs, flux, oracle = z["endpoints"], z["flux"], z["oracle_flux"]
                for column, sign in ((0, -1), (1, 1)):
                    owner = pairs[:, column]; active = owner >= 0
                    np.add.at(total, owner[active], sign * flux[active])
                    np.add.at(exact, owner[active], sign * oracle[active])
                boundary_net += np.sum(np.where(pairs[:, 0, None] < 0, flux, 0)
                                       - np.where(pairs[:, 1, None] < 0, flux, 0), axis=0)
        numerator = reference_data[f"N{n}.numerator_q1"]
        continuous_volume = reference_data[f"N{n}.volume_q1"]
        if np.any(continuous_volume <= 0): raise ValueError("uncovered reference owners")
        if not np.allclose(continuous_volume,t.vol,rtol=1e-12,atol=1e-12):raise ValueError("midpoint owner volume mismatch")
        action = total / t.vol[:, None]
        oracle_action = exact / t.vol[:, None]
        target = numerator / continuous_volume[:, None]
        if not (np.isfinite(action).all() and np.isfinite(oracle_action).all() and np.isfinite(target).all()):
            raise ValueError("nonfinite global action or reference")
        masks = region_masks(t, all_endpoints, all_family)
        selected = control_data[f"N{n}.owner_ids"]
        if not np.allclose(target[selected],control_data[f"N{n}.average_q1"],atol=1e-11,rtol=1e-10):
            raise ValueError("control midpoint reference differs from global midpoint")
        stats = {}; control_stats = {}
        for field, name in enumerate(cfg["fields"]):
            stats[name] = field_stats(action[:, field] - target[:, field], t.vol, masks)
            stats[name]["reconstruction_q3_l2"] = field_stats(action[:, field] - oracle_action[:, field], t.vol, masks)["l2"]
            sample_volume = t.vol[selected]
            sample_spatial = np.sqrt(np.sum(sample_volume * (action[selected, field] - target[selected, field])**2) / np.sum(sample_volume))
            midpoint=control_data[f"N{n}.average_q1"][:,field]
            half_delta=control_data[f"N{n}.average_q1_halfstep"][:,field]-midpoint
            norm=lambda delta:float(np.sqrt(np.sum(sample_volume*delta**2)/np.sum(sample_volume)))
            control_stats[name]={"sample_spatial_l2":float(sample_spatial),
                "midpoint_halfstep_l2":norm(half_delta),
                "midpoint_reference_check_pass":bool(norm(half_delta)<=cfg["bounded_reference_fraction_maximum"]*sample_spatial)}
        case_path = output / f"N{n}.global.npz"
        atomic_npz(case_path, owner_ids=np.arange(len(t.vol)), owner_flat_ids=t.g.owner_flat_ids,
                   volume=t.vol, midpoint_reference_volume=continuous_volume, action=action,
                   oracle_q3=oracle_action, reference_midpoint=target,
                   **{f"region_{name}": mask for name, mask in masks.items()})
        arrays_hashes[str(n)] = sha(case_path)
        cases[str(n)] = {"owners": len(t.vol), "stats": stats, "reference_controls": control_stats,
                         "boundary_balance_residual": (np.sum(total, axis=0) - boundary_net).tolist(),
                         "global_arrays_sha256": arrays_hashes[str(n)]}
        if np.max(np.abs(np.sum(total, axis=0) - boundary_net)) > 1e-7:
            raise ValueError("shared-face conservation failed")
    orders = {}
    for field in cfg["fields"]:
        errors = np.array([cases[str(n)]["stats"][field]["l2"] for n in cfg["resolutions"]])
        if np.isfinite(errors).all() and np.all(errors > 0):
            interval = np.log(errors[:-1] / errors[1:]) / np.log(np.array([48 / 32, 64 / 48]))
            reported = interval.tolist()
            passed = bool(np.isfinite(interval).all() and np.all(interval >= cfg["global_l2_order_minimum"]))
        else:
            reported = [None, None]; passed = False
        orders[field] = {"errors": errors.tolist(), "orders": reported, "order_pass": passed}
    result = {"identity": base_identity, "computation_completed": True, "resolutions": cases,
              "fields": orders, "global_order_pass": all(x["order_pass"] for x in orders.values()),
              "reference_qualified_by_bounded_checks": all(
                  cases[str(n)]["reference_controls"][f][key]
                  for n in cfg["resolutions"] for f in cfg["fields"]
                  for key in ("midpoint_reference_check_pass",)),
              "reference_convention":"physical_raw_volume_midpoint_projection",
              "integrated_reference_controls":False,
              "production_promoted": False, "elliptic_evolved_energy_qualification": "separate milestones"}
    atomic_json(output / "summary.json", result)
    return result


def validate(output, plan):
    output = Path(output); base = current_identity(output)
    if plan["identity"] != base: raise ValueError("plan identity mismatch")
    topology_record = json.loads((output / "topology_summary.json").read_text())
    if topology_record["identity"] != base or not topology_record["collapsed_rows_included_in_face_totals"]:
        raise ValueError("topology identity mismatch")
    for stage in STAGES:
        record = json.loads((output / f"{stage}_reduction.json").read_text())
        if record["identity"] != stage_identity(output, stage) or not record["all_complete"]:
            raise ValueError(f"invalid {stage} reduction")
        if stage == "support" and record.get("unsupported_count", 1):
            raise ValueError("unsupported support faces")
        for n in settings()["resolutions"]:
            units = plan["stages"][stage][str(n)]
            if record["resolutions"][str(n)]["ids"] != len(selected_ids(stage, n, output)):
                raise ValueError(f"incomplete {stage} coverage N{n}")
            for unit in units:
                if not valid_unit(output, unit, record["identity"]):
                    raise ValueError(f"missing {stage} unit N{n}")
    summary = json.loads((output / "summary.json").read_text())
    if summary["identity"] != base or not summary["computation_completed"]:
        raise ValueError("invalid summary")
    for n in settings()["resolutions"]:
        if sha(output / f"N{n}.global.npz") != summary["resolutions"][str(n)]["global_arrays_sha256"]:
            raise ValueError(f"corrupt global arrays N{n}")
    atomic_json(output / "validation.json", {"identity": base, "operational_complete": True,
                                                  "scientific_global_order_pass": summary["global_order_pass"],
                                                  "reference_bounded_checks_pass": summary["reference_qualified_by_bounded_checks"]})


def smoke(input_root, output):
    """One complete N32 wall/join owner action plus one face of every family."""
    output = Path(output); k.configure(input_root); t = k.load(32)
    ref = k.num.reference(output / "reference_sidecar.json", verify_hashes=False)
    values = k.num.owner_values(t.centers, t.ro, t.rv, t.vol, ref)
    with np.load(output / "N32.topology.npz", allow_pickle=False) as z:
        face_ids, family, endpoints = z["face_ids"], z["family"], z["endpoints"]
    chosen_owners = np.array((int(t.ro[np.ravel_multi_index((31, 0, 0), (32,)*3)]), 544))
    positions = np.flatnonzero(np.isin(endpoints, chosen_owners).any(axis=1))
    positions = np.unique(np.r_[positions, [np.flatnonzero(family == f)[0] for f in range(8)]])
    ids, fam = face_ids[positions], family[positions]
    support_positions = np.flatnonzero(np.isin(fam, (6, 7)))
    certificates, _ = candidate.support_execute(t, ref, ids[support_positions], fam[support_positions])
    if not all(item["supported"] for item in certificates): raise ValueError("smoke support failed")
    rows, cost, points, integ = candidate.assembly_rows(t, ref, ids, fam)
    bc_flux = boundary_data(ref, rows)
    flux = np.array([w @ values[don] for don, w, _ in rows]) + bc_flux
    oracle = np.zeros_like(flux)
    for j in np.flatnonzero(fam != 0):
        gradient = k.num.fields(ref, points[j])[1]
        oracle[j] = np.einsum("qa,qfa->f", integ[j], gradient)
    action = np.zeros((2, 4)); exact_face = np.zeros_like(action)
    for j, position in enumerate(positions):
        for endpoint, sign in ((0, -1), (1, 1)):
            hit = np.flatnonzero(chosen_owners == endpoints[position, endpoint])
            if len(hit): action[hit[0]] += sign * flux[j]; exact_face[hit[0]] += sign * oracle[j]
    selected_raw = np.flatnonzero(np.isin(t.ro, chosen_owners))
    reference = reference_cells(t,ref,selected_raw,1)
    ref_num = np.zeros((2, 4)); ref_vol = np.zeros(2)
    for row, oid in enumerate(t.ro[selected_raw]):
        j = int(np.flatnonzero(chosen_owners == oid)[0]); ref_num[j] += reference["numerator"][row]; ref_vol[j] += reference["continuous_volume"][row]
    result = {"n": 32, "owners": chosen_owners.tolist(), "incident_and_family_faces": len(ids),
              "family_counts": {str(f): int(np.sum(fam == f)) for f in range(8)},
              "support_faces": len(certificates), "max_support_residual": max((x["max_residual"] for x in certificates), default=0),
              "action": (action / t.vol[chosen_owners, None]).tolist(),
              "oracle_q3": (exact_face / t.vol[chosen_owners, None]).tolist(),
              "reference_midpoint": (ref_num / ref_vol[:, None]).tolist(),
              "BC_faces": int(np.sum(np.isin(fam, (1, 2, 4)))), "assembly_cost": cost}
    if not np.isfinite(np.asarray(result["action"])).all() or np.any(ref_vol <= 0): raise ValueError("smoke pipeline incomplete")
    atomic_json(output / "smoke.json", result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify-inputs", "topology", "plan", "smoke", "run-stage", "reduce-stage", "summarize", "validate", "run"))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--memory-budget-gib", type=float)
    parser.add_argument("--worker-memory-gib", type=float)
    parser.add_argument("--memory-reserve-gib", type=float, default=0.0)
    parser.add_argument("--max-tasks-per-worker", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args(); args.input_root = args.input_root.resolve(); args.output = args.output.resolve()
    with locked(args.output):
        for name in ("scratch", "cache", "logs"):
            (args.output / name).mkdir(exist_ok=True)
        os.environ["TMPDIR"] = str(args.output / "scratch")
        os.environ["XDG_CACHE_HOME"] = str(args.output / "cache")
        os.environ["DRBX_CACHE_DIR"] = str(args.output / "cache" / "jax")
        if args.command in ("verify-inputs", "run", "topology", "plan", "smoke", "run-stage", "reduce-stage", "summarize", "validate"):
            verify_inputs(args.input_root, args.output)
        if args.command == "verify-inputs": return
        if args.command in ("topology", "run", "plan", "smoke"):
            topology_stage(args.input_root, args.output)
        if args.command == "topology": return
        if args.command == "smoke":
            smoke(args.input_root, args.output)
            return
        if args.command in ("plan", "run"):
            plan = create_plan(args.output)
        else:
            plan = json.loads((args.output / "plan.json").read_text())
            if plan["identity"] != current_identity(args.output): raise ValueError("stale plan")
        if args.command == "plan": return
        if args.command == "run":
            for stage in STAGES:
                run_stage(args, stage, plan)
                reduce_stage(args.input_root, args.output, stage, plan)
            summarize(args.input_root, args.output, plan)
            validate(args.output, plan)
        elif args.command == "run-stage":
            if args.stage is None: raise ValueError("--stage is required")
            run_stage(args, args.stage, plan)
        elif args.command == "reduce-stage":
            if args.stage is None: raise ValueError("--stage is required")
            reduce_stage(args.input_root, args.output, args.stage, plan)
        elif args.command == "summarize": summarize(args.input_root, args.output, plan)
        elif args.command == "validate": validate(args.output, plan)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        args = parse_args()
        atomic_json(args.output / "last_exit.json", {"status": "failed", "command": args.command,
                                                   "stage": args.stage, "error": repr(error), "unix_time": time.time()})
        raise
    else:
        args = parse_args()
        atomic_json(args.output / "last_exit.json", {"status": "complete", "command": args.command,
                                                   "stage": args.stage, "unix_time": time.time()})
