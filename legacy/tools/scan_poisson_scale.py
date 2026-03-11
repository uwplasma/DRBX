from __future__ import annotations

import argparse
import copy
import csv
import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from jaxdrb.driver import run_simulation


def _parse_scales(text: str) -> list[float]:
    vals = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("No valid poisson scales were provided.")
    return vals


def _score_against_target(diag: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> float:
    fields = ("rms_n_fluct", "rms_Te_fluct", "rms_omega_fluct", "rms_phi_fluct")
    score = 0.0
    nt = np.asarray(target["times"], dtype=np.float64)
    mt = np.asarray(diag["times"], dtype=np.float64)
    for key in fields:
        if key not in diag or key not in target:
            continue
        y_target = np.asarray(target[key], dtype=np.float64)
        y_model = np.asarray(diag[key], dtype=np.float64)
        y_interp = np.interp(nt, mt, y_model)
        den = np.maximum(np.abs(y_target), 1e-12)
        rel = (y_interp - y_target) / den
        score += float(np.mean(rel * rel))
    return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan poisson_scale with finite/spike gating and optional target scoring."
    )
    parser.add_argument("--config", required=True, help="Base TOML config path.")
    parser.add_argument(
        "--scales",
        required=True,
        help="Comma-separated poisson_scale values, e.g. '1e-6,3e-6,1e-5'.",
    )
    parser.add_argument(
        "--target-rms",
        default="",
        help="Optional target RMS npz (times + rms_*_fluct) for scoring.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.0,
        help="Optional override for time.dt during scan.",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=0,
        help="Optional override for time.nsteps during scan.",
    )
    parser.add_argument(
        "--max-growth-factor",
        type=float,
        default=1.0e4,
        help="Reject run if any RMS channel exceeds this max/value_at_start ratio.",
    )
    parser.add_argument(
        "--max-rms-abs",
        type=float,
        default=1.0e8,
        help="Reject run if any RMS channel exceeds this absolute threshold.",
    )
    parser.add_argument(
        "--out-csv",
        default="docs/figures/poisson_scale_scan.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("rb") as f:
        base_cfg: dict[str, Any] = tomllib.load(f)

    scales = _parse_scales(args.scales)
    target = None
    if args.target_rms:
        target = dict(np.load(args.target_rms))

    rows: list[dict[str, Any]] = []
    for scale in scales:
        cfg = copy.deepcopy(base_cfg)
        numerics = dict(cfg.get("numerics", {}))
        numerics["poisson_scale"] = float(scale)
        cfg["numerics"] = numerics
        if args.dt > 0.0 or args.nsteps > 0:
            time_cfg = dict(cfg.get("time", {}))
            if args.dt > 0.0:
                time_cfg["dt"] = float(args.dt)
            if args.nsteps > 0:
                time_cfg["nsteps"] = int(args.nsteps)
            cfg["time"] = time_cfg

        try:
            result = run_simulation(cfg, as_numpy=True)
            diag = dict(result.diagnostics)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "poisson_scale": scale,
                    "passed": 0,
                    "reason": f"exception:{type(exc).__name__}",
                    "score": math.inf,
                }
            )
            continue

        keys = ("rms_n_fluct", "rms_Te_fluct", "rms_omega_fluct", "rms_phi_fluct")
        finite = True
        fail_reason = "ok"
        growth = 0.0
        peak = 0.0
        for key in keys:
            if key not in diag:
                finite = False
                fail_reason = f"missing:{key}"
                break
            arr = np.asarray(diag[key], dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                finite = False
                fail_reason = f"nonfinite:{key}"
                break
            ref_idx = 1 if arr.size > 1 else 0
            a0 = max(abs(float(arr[ref_idx])), 1e-8)
            amax = float(np.max(np.abs(arr)))
            growth = max(growth, amax / a0)
            peak = max(peak, amax)

        passed = int(finite and growth <= args.max_growth_factor and peak <= args.max_rms_abs)
        reason = "ok" if passed else fail_reason
        if not passed and reason == "ok":
            reason = "gate_failed"
        score = _score_against_target(diag, target) if passed and target is not None else math.inf
        if passed and target is None:
            score = growth

        rows.append(
            {
                "poisson_scale": scale,
                "passed": passed,
                "reason": reason,
                "score": score,
                "max_growth_factor": growth,
                "max_rms_abs": peak,
                "rms_n_fluct_end": float(np.asarray(diag.get("rms_n_fluct", [np.nan]))[-1]),
                "rms_Te_fluct_end": float(np.asarray(diag.get("rms_Te_fluct", [np.nan]))[-1]),
                "rms_omega_fluct_end": float(np.asarray(diag.get("rms_omega_fluct", [np.nan]))[-1]),
                "rms_phi_fluct_end": float(np.asarray(diag.get("rms_phi_fluct", [np.nan]))[-1]),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "poisson_scale",
        "passed",
        "reason",
        "score",
        "max_growth_factor",
        "max_rms_abs",
        "rms_n_fluct_end",
        "rms_Te_fluct_end",
        "rms_omega_fluct_end",
        "rms_phi_fluct_end",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    passed_rows = [r for r in rows if int(r.get("passed", 0)) == 1]
    if passed_rows:
        best = min(passed_rows, key=lambda r: float(r["score"]))
        print(
            f"Best passing poisson_scale={best['poisson_scale']} score={best['score']} "
            f"growth={best['max_growth_factor']} peak={best['max_rms_abs']}"
        )
    else:
        print("No passing poisson_scale candidates.")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
