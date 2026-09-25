"""Checkpointed, parallel raw-midpoint observations for P06 preparation.

Workers evaluate immutable analytic fields. The controller alone writes chunks.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

import parallel_runner as runner

_STATE = {}
SCHEMA = "drbx.p06-structured-observation-chunk-v1"


def _worker_init(deployment_root: str, output: str, runtime_path: str, resolution: int):
    numeric = runner._bootstrap(Path(deployment_root), Path(deployment_root), Path(output))
    runtime = json.loads(Path(runtime_path).read_text())
    context = numeric.cubic._load_context(Path(runtime["paths"]["geometry"]),
                                           Path(runtime["paths"]["baseline"]), resolution)
    reference = numeric.integrated._reference(Path(runtime["paths"]["reference_sidecar"]), verify_hashes=False)
    _STATE.clear()
    _STATE.update(numeric=numeric, context=context, reference=reference,
                  time_value=float(runtime["time"]), raw_points=numeric._raw_points(context))


def _worker(unit: dict) -> tuple[dict, dict[str, np.ndarray], float]:
    numeric = _STATE["numeric"]
    indices = np.arange(int(unit["first"]), int(unit["last"]), dtype=np.int64)
    points = _STATE["raw_points"][indices]
    values = np.empty((len(numeric.FIELD_NAMES), 5, len(indices)), dtype=np.float64)
    for state, name in enumerate(numeric.FIELD_NAMES):
        values[state] = numeric._evaluate_fields(name, _STATE["reference"], points,
                                                 _STATE["time_value"])[0]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"nonfinite raw observation in {unit['id']}")
    return unit, {"indices": indices, "values": values}, numeric._max_rss_gib()


def _chunk_path(output: Path, unit: dict) -> Path:
    return output / f"N{unit['resolution']}.observation.chunks" / f"{unit['id']}.npz"


def _validate_chunk(path: Path, unit: dict, identity: str, numeric):
    arrays, meta = numeric._load_npz(path, SCHEMA)
    expected = np.arange(unit["first"], unit["last"], dtype=np.int64)
    if (meta.get("status") != "complete" or meta.get("identity") != identity
        or not np.array_equal(arrays["indices"], expected)
        or arrays["values"].shape != (len(numeric.FIELD_NAMES), 5, len(expected))
        or meta.get("array_sha256") != numeric._array_hash(*(arrays[name] for name in sorted(arrays)))
        or not np.all(np.isfinite(arrays["values"]))):
        raise ValueError(f"stale or corrupt observation checkpoint: {path}")
    return meta


def run(args, resolution: int) -> dict:
    output = Path(args.output).resolve()
    config_path = Path(__file__).with_name("configuration.json")
    input_manifest = Path(__file__).with_name("input_manifest.json")
    runtime, runtime_path, numeric = runner._materialize(config_path, runner.HERE.parents[1],
                                                           Path(args.input_root).resolve(), output)
    n = int(resolution)
    chunk = int(json.loads(config_path.read_text())["observation_chunk"])
    manifest = json.loads((output / "campaign_manifest.json").read_text())
    identity = runner._digest({"stage":"observations", "campaign":manifest["sha256"],
                               "runtime_sha256":runner._sha256(runtime_path),
                               "sources":numeric._source_identity(runtime),
                               "resolution":n,"chunk":chunk,
                               "input_manifest_sha256":runner._sha256(input_manifest)})
    units = [{"id":f"obs_{first:07d}_{min(first+chunk,n**3):07d}",
              "first":first,"last":min(first+chunk,n**3),"resolution":n}
             for first in range(0,n**3,chunk)]
    plan = {"schema":"drbx.p06-structured-observation-plan-v1", "resolution":n,
            "identity":identity,"units":units}
    plan_path = output / f"N{n}.observation.plan.json"
    receipt_path = output / f"N{n}.observation-receipt.json"
    requested = int(args.workers)
    workers = runner._effective_worker_count(requested,
        memory_budget_gib=args.memory_budget_gib,
        worker_memory_gib=args.worker_memory_gib,
        memory_reserve_gib=args.memory_reserve_gib)
    started = time.perf_counter()
    with runner._exclusive_output(output):
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("observation plan identity changed")
        else:
            runner._atomic_json(plan_path, plan)
        reused=[]; pending=[]; hashes={}
        for unit in units:
            path=_chunk_path(output,unit)
            if path.exists():
                saved_meta=_validate_chunk(path,unit,identity,numeric)
                hashes[unit["id"]]=saved_meta["array_sha256"]
                reused.append(unit["id"])
            else:
                pending.append(unit)
        computed=[]; peak=0.0
        if pending:
            context=mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers,mp_context=context,
                    max_tasks_per_child=int(args.max_tasks_per_worker),
                    initializer=_worker_init,
                    initargs=(str(runner.HERE.parents[1]),str(output),str(runtime_path),n)) as pool:
                iterator=iter(pending); active={}
                for _ in range(min(2*workers,len(pending))):
                    unit=next(iterator,None)
                    if unit is not None:active[pool.submit(_worker,unit)]=unit
                while active:
                    done,_=wait(active,return_when=FIRST_COMPLETED)
                    for future in done:
                        unit=active.pop(future)
                        completed, arrays, rss=future.result()
                        if completed != unit:
                            raise ValueError("observation worker returned another unit")
                        peak=max(peak,float(rss))
                        path=_chunk_path(output,unit)
                        meta={"schema":SCHEMA,"status":"complete","identity":identity,
                              "unit_id":unit["id"],"maximum_worker_rss_gib":float(rss),
                              "array_sha256":numeric._array_hash(*(arrays[name] for name in sorted(arrays)))}
                        numeric._write_npz(path,arrays,meta)
                        _validate_chunk(path,unit,identity,numeric)
                        hashes[unit["id"]]=meta["array_sha256"]
                        computed.append(unit["id"])
                        next_unit=next(iterator,None)
                        if next_unit is not None:active[pool.submit(_worker,next_unit)]=next_unit
        receipt={"schema":"drbx.p06-structured-observation-receipt-v1","status":"complete",
                 "resolution":n,"identity":identity,"plan_sha256":runner._sha256(plan_path),
                 "requested_workers":requested,"effective_workers":workers,
                 "memory_budget_gib":args.memory_budget_gib,"worker_memory_gib":args.worker_memory_gib,
                 "memory_reserve_gib":args.memory_reserve_gib,
                 "max_tasks_per_worker":args.max_tasks_per_worker,
                 "computed_units":sorted(computed),"reused_units":sorted(reused),
                 "chunk_array_hashes":dict(sorted(hashes.items())),
                 "data_sha256":runner._digest(dict(sorted(hashes.items()))),
                 "maximum_observed_worker_rss_gib":peak,"seconds":time.perf_counter()-started}
        if not pending and receipt_path.exists():
            previous=json.loads(receipt_path.read_text())
            if (previous.get("status")=="complete" and previous.get("identity")==identity
                and previous.get("plan_sha256")==receipt["plan_sha256"]
                and previous.get("data_sha256")==receipt["data_sha256"]):
                return previous
        runner._atomic_json(receipt_path,receipt)
    return receipt
