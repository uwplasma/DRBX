from __future__ import annotations

import argparse
import copy
import csv
import math
import tomllib
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from jaxdrb.benchmarking import finite_run_gate
from jaxdrb.driver import run_simulation

FIELDS = ("n", "Te", "omega", "phi")


@dataclass
class Candidate:
    omega_mult: float
    source_mult: float
    dn_mult: float
    domega_mult: float
    poisson_scale: float

    def key(self) -> str:
        return (
            f"om={self.omega_mult:g}|src={self.source_mult:g}|"
            f"dn={self.dn_mult:g}|dw={self.domega_mult:g}|ps={self.poisson_scale:g}"
        )


def _parse_list(text: str, cast=float) -> list[Any]:
    values: list[Any] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(cast(token))
    if not values:
        raise ValueError(f"Expected at least one value in '{text}'")
    return values


def _parse_grid(text: str) -> tuple[int, int, int]:
    vals = _parse_list(text, int)
    if len(vals) != 3:
        raise ValueError("grid override must be 'nx,ny,nz'")
    return int(vals[0]), int(vals[1]), int(vals[2])


def _load_hermes_rms(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path)
    out = {"times": np.asarray(raw["times"], dtype=np.float64)}
    for f in FIELDS:
        key = f"rms_{f}_fluct"
        if key not in raw:
            raise KeyError(f"Missing '{key}' in Hermes RMS file '{path}'")
        out[key] = np.asarray(raw[key], dtype=np.float64)
    return out


def _channel_rel_error(
    t_model: np.ndarray,
    y_model: np.ndarray,
    t_target: np.ndarray,
    y_target: np.ndarray,
) -> tuple[float, float]:
    y_interp = np.interp(t_target, t_model, y_model)
    scale_floor = max(1e-8, 0.05 * float(np.max(np.abs(y_target))))
    denom = np.maximum(np.abs(y_target), scale_floor)
    rel = np.abs(y_interp - y_target) / denom
    return float(np.mean(rel)), float(np.max(rel))


def _make_cfg(
    base_cfg: dict[str, Any],
    c: Candidate,
    nsteps: int,
    save_every: int,
    grid_override: tuple[int, int, int] | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    physics_physical = dict(cfg.get("physics_physical", {}))
    physics_physical["omega_n"] = float(physics_physical["omega_n"]) * c.omega_mult
    physics_physical["source_n0"] = float(physics_physical["source_n0"]) * c.source_mult
    cfg["physics_physical"] = physics_physical

    transport = dict(cfg.get("transport", {}))
    transport["Dn"] = float(transport["Dn"]) * c.dn_mult
    transport["DOmega"] = float(transport["DOmega"]) * c.domega_mult
    cfg["transport"] = transport

    numerics = dict(cfg.get("numerics", {}))
    numerics["poisson_scale"] = float(c.poisson_scale)
    cfg["numerics"] = numerics

    time_cfg = dict(cfg.get("time", {}))
    time_cfg["nsteps"] = int(nsteps)
    time_cfg["save_every"] = int(max(1, save_every))
    time_cfg["save_fields"] = True
    time_cfg.setdefault("snapshot_fields", ["n", "Te", "omega", "phi"])
    time_cfg["return_numpy"] = True
    time_cfg["diag_mode"] = "full"
    cfg["time"] = time_cfg

    if grid_override is not None:
        nx, ny, nz = grid_override
        geom = dict(cfg.get("geometry", {}))
        geom["nx"] = int(nx)
        geom["ny"] = int(ny)
        geom["nz"] = int(nz)
        cfg["geometry"] = geom

    return cfg


def _evaluate_candidate(
    base_cfg: dict[str, Any],
    c: Candidate,
    hermes: dict[str, np.ndarray],
    t_end: float,
    max_growth_factor: float,
    max_rms_abs: float,
    grid_override: tuple[int, int, int] | None,
) -> dict[str, float | int | str]:
    time_cfg = base_cfg.get("time", {})
    dt = float(time_cfg.get("dt", 0.0))
    if dt <= 0.0:
        raise ValueError("Config must contain [time].dt > 0")
    nsteps = int(max(1, math.ceil(t_end / dt)))
    save_every = max(1, nsteps // 20)
    cfg = _make_cfg(base_cfg, c, nsteps, save_every, grid_override)
    result = run_simulation(cfg, as_numpy=True)
    diag = dict(result.diagnostics)

    passed, reason, growth, peak = finite_run_gate(
        diag,
        max_growth_factor=max_growth_factor,
        max_rms_abs=max_rms_abs,
    )
    if not passed:
        return {
            "candidate": c.key(),
            "passed": 0,
            "reason": str(reason),
            "score": math.inf,
            "mean_rel": math.inf,
            "max_rel": math.inf,
            "growth": float(growth),
            "peak": float(peak),
        }

    t_model = np.asarray(diag["times"], dtype=np.float64)
    target_mask = hermes["times"] <= (t_model[-1] + 1e-12)
    t_target = np.asarray(hermes["times"][target_mask], dtype=np.float64)
    if t_target.size < 2:
        raise RuntimeError("Hermes target window has <2 points; increase t_end or target file.")

    means = []
    maxes = []
    for field in FIELDS:
        key = f"rms_{field}_fluct"
        y_model = np.asarray(diag[key], dtype=np.float64)
        y_target = np.asarray(hermes[key][target_mask], dtype=np.float64)
        mean_rel, max_rel = _channel_rel_error(t_model, y_model, t_target, y_target)
        means.append(mean_rel)
        maxes.append(max_rel)

    mean_rel = float(np.mean(means))
    max_rel = float(np.max(maxes))
    return {
        "candidate": c.key(),
        "passed": 1,
        "reason": "ok",
        "score": mean_rel + 0.1 * max_rel,
        "mean_rel": mean_rel,
        "max_rel": max_rel,
        "growth": float(growth),
        "peak": float(peak),
        "nsteps": nsteps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibration loop for Hermes/jax parity. Runs short windows first, "
            "keeps finite candidates, and picks best RMS-fluctuation match."
        )
    )
    parser.add_argument("--config", required=True, help="Base jax_drb TOML config.")
    parser.add_argument(
        "--hermes-rms",
        required=True,
        help="Hermes RMS npz file (times + rms_{n,Te,omega,phi}_fluct).",
    )
    parser.add_argument(
        "--omega-mults",
        default="0.9,1.0,1.1",
        help="Comma-separated multipliers for physics_physical.omega_n.",
    )
    parser.add_argument(
        "--source-mults",
        default="0.9,1.0,1.1",
        help="Comma-separated multipliers for physics_physical.source_n0.",
    )
    parser.add_argument(
        "--dn-mults",
        default="0.8,1.0,1.2",
        help="Comma-separated multipliers for transport.Dn.",
    )
    parser.add_argument(
        "--domega-mults",
        default="0.8,1.0,1.2",
        help="Comma-separated multipliers for transport.DOmega.",
    )
    parser.add_argument(
        "--poisson-scales",
        default="1e-4,2e-4,3e-4",
        help="Comma-separated absolute values for numerics.poisson_scale.",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=0.1,
        help="End time for calibration window.",
    )
    parser.add_argument(
        "--rtol-target",
        type=float,
        default=1e-1,
        help="Target max relative error for pass/fail summary (not hard reject).",
    )
    parser.add_argument(
        "--max-growth-factor",
        type=float,
        default=1e4,
        help="Finite-run gate growth threshold.",
    )
    parser.add_argument(
        "--max-rms-abs",
        type=float,
        default=1e4,
        help="Finite-run gate absolute RMS threshold.",
    )
    parser.add_argument(
        "--grid-override",
        default="",
        help="Optional reduced grid override 'nx,ny,nz' for fast calibration.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=40,
        help="Cap number of evaluated candidates (deterministic order).",
    )
    parser.add_argument(
        "--out-csv",
        default="docs/figures/parity_calibration_scan.csv",
        help="CSV output summary.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with cfg_path.open("rb") as f:
        base_cfg = tomllib.load(f)
    hermes = _load_hermes_rms(Path(args.hermes_rms).resolve())

    omega_mults = _parse_list(args.omega_mults, float)
    source_mults = _parse_list(args.source_mults, float)
    dn_mults = _parse_list(args.dn_mults, float)
    domega_mults = _parse_list(args.domega_mults, float)
    poisson_scales = _parse_list(args.poisson_scales, float)
    grid_override = _parse_grid(args.grid_override) if args.grid_override else None

    candidates = [
        Candidate(om, src, dn, dw, ps)
        for om, src, dn, dw, ps in product(
            omega_mults, source_mults, dn_mults, domega_mults, poisson_scales
        )
    ]
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    rows: list[dict[str, float | int | str]] = []
    for i, c in enumerate(candidates, start=1):
        print(f"[{i}/{len(candidates)}] {c.key()}")
        try:
            row = _evaluate_candidate(
                base_cfg=base_cfg,
                c=c,
                hermes=hermes,
                t_end=float(args.t_end),
                max_growth_factor=float(args.max_growth_factor),
                max_rms_abs=float(args.max_rms_abs),
                grid_override=grid_override,
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "candidate": c.key(),
                "passed": 0,
                "reason": f"exception:{type(exc).__name__}",
                "score": math.inf,
                "mean_rel": math.inf,
                "max_rel": math.inf,
                "growth": math.inf,
                "peak": math.inf,
            }
        row["rtol_target"] = float(args.rtol_target)
        row["meets_rtol"] = 1 if row.get("max_rel", math.inf) <= float(args.rtol_target) else 0
        rows.append(row)

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "candidate",
        "passed",
        "reason",
        "score",
        "mean_rel",
        "max_rel",
        "meets_rtol",
        "rtol_target",
        "growth",
        "peak",
        "nsteps",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    passing = [r for r in rows if int(r.get("passed", 0)) == 1]
    if passing:
        best = min(passing, key=lambda r: float(r["score"]))
        print(
            "Best passing candidate: "
            f"{best['candidate']} score={best['score']:.3e} "
            f"mean_rel={best['mean_rel']:.3e} max_rel={best['max_rel']:.3e} "
            f"meets_rtol={bool(best['meets_rtol'])}"
        )
    else:
        print("No passing candidates.")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
