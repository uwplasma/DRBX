#!/usr/bin/env python3
"""Local-only replay of the frozen global kernels against bounded P06 evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
os.environ.setdefault("HSX_SOURCE_ROOT", str(_REPO))
os.environ.setdefault("HSX_DEPLOYMENT_ROOT", str(_REPO.parent))

import parallel_runner
import numerics


def _chunks(values: np.ndarray, size: int):
    for first in range(0, len(values), size):
        yield values[first:first+size]


def replay(args: argparse.Namespace) -> dict:
    runtime, _runtime_path, _numeric = parallel_runner._materialize(
        args.config, args.deployment_root, args.input_root, args.output
    )
    results = {}
    maximum = 0.0
    for n in args.resolutions:
        prepare, _ = numerics._load_npz(args.output/f"N{n}.prepare.npz", numerics.PREPARE_SCHEMA)
        context = numerics.cubic._load_context(Path(runtime["paths"]["geometry"]), Path(runtime["paths"]["baseline"]), n)
        reference = numerics.integrated._reference(Path(runtime["paths"]["reference_sidecar"]), verify_hashes=False)
        with np.load(args.policy_root/f"N{n}.npz", allow_pickle=False) as bounded, np.load(args.original_root/f"N{n}.npz", allow_pickle=False) as original:
            rows = np.asarray(bounded["selected_rows"], dtype=np.int64)
            owners = np.asarray(original["owner_indices"], dtype=np.int64)[rows]
            expected = {
                field: {
                    "centered": np.asarray(bounded[f"{field}:cell_point:total"]),
                    "U": np.asarray(bounded[f"{field}:cell_point:face_point:U"]),
                }
                for field in numerics.FIELD_NAMES
            }
        lookup = {int(owner): row for row, owner in enumerate(owners)}
        raw_owner = prepare["raw_owner"].astype(np.int64)
        raw = np.flatnonzero(np.isin(raw_owner, owners))
        keys = numerics._raw_keys(n, raw)
        faces = set()
        for axis in range(3):
            upper = keys.copy(); upper[:, axis] += 1
            faces.update(map(int, numerics._face_index(n, axis, *keys.T)))
            faces.update(map(int, numerics._face_index(n, axis, *upper.T)))
        evolution = np.zeros(len(owners))
        centered = {field: np.zeros((len(owners),4)) for field in numerics.FIELD_NAMES}
        correction = {field: np.zeros((len(owners),4)) for field in numerics.FIELD_NAMES}
        for indices in _chunks(raw, 32):
            arrays, _ = numerics._compute_cells(context, reference, prepare, indices, time_value=float(runtime["time"]), curl_step=float(runtime["curl_step"]))
            local = np.asarray([lookup[int(raw_owner[index])] for index in indices])
            np.add.at(evolution, local, arrays["evolution_volume"])
            for field in numerics.FIELD_NAMES:
                np.add.at(centered[field], local, arrays[f"candidate:{field}:total"])
        face_indices = np.asarray(sorted(faces), dtype=np.int64)
        for indices in _chunks(face_indices, 64):
            arrays, _ = numerics._compute_faces(context, reference, prepare, indices, time_value=float(runtime["time"]))
            for row in range(len(indices)):
                for side, valid_name, raw_name in ((0,"lower_valid","lower_raw"),(1,"upper_valid","upper_raw")):
                    if not bool(arrays[valid_name][row]):
                        continue
                    owner = int(raw_owner[int(arrays[raw_name][row])])
                    if owner not in lookup:
                        continue
                    for state, field in enumerate(numerics.FIELD_NAMES):
                        correction[field][lookup[owner]] += arrays["correction"][state,row,side]
        case = {}
        for field in numerics.FIELD_NAMES:
            actual_centered = centered[field]/evolution[:,None]
            actual_u = (centered[field]+correction[field])/evolution[:,None]
            differences = {
                "centered_max_abs": float(np.max(np.abs(actual_centered-expected[field]["centered"]))),
                "U_max_abs": float(np.max(np.abs(actual_u-expected[field]["U"]))),
            }
            maximum = max(maximum, *differences.values())
            case[field] = differences
        results[str(n)] = case
    payload = {
        "schema": "drbx.p06-curvature-global-bounded-replay-v1",
        "status": "pass" if maximum < args.tolerance else "fail",
        "tolerance": args.tolerance,
        "maximum_abs_difference": maximum,
        "cases": results,
        "note": "local validation only; bounded numerical answers are not campaign inputs",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    return payload


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    here=Path(__file__).resolve().parent; repo=here.parents[1]
    parser.add_argument("--config",type=Path,default=here/"configuration.json")
    parser.add_argument("--deployment-root",type=Path,default=repo)
    parser.add_argument("--input-root",type=Path,default=repo.parent)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--policy-root",type=Path,required=True)
    parser.add_argument("--original-root",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--resolutions",type=int,nargs="+",default=[32])
    parser.add_argument("--tolerance",type=float,default=2.0e-10)
    args=parser.parse_args()
    result=replay(args)
    print(json.dumps(result,sort_keys=True))
    return 0 if result["status"]=="pass" else 1


if __name__=="__main__":
    raise SystemExit(main())
