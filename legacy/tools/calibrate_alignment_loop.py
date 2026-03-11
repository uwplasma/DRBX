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
    phi_dissipation_on: bool
    phi_sheath_dissipation_on: bool
    core_vorticity_damping_on: bool

    def key(self) -> str:
        return (
            f"om={self.omega_mult:g}|src={self.source_mult:g}|"
            f"dn={self.dn_mult:g}|dw={self.domega_mult:g}|ps={self.poisson_scale:g}|"
            f"phi_diss={int(self.phi_dissipation_on)}|"
            f"phi_sheath={int(self.phi_sheath_dissipation_on)}|"
            f"core_omega_damp={int(self.core_vorticity_damping_on)}"
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


def _parse_bool_list(text: str) -> list[bool]:
    vals: list[bool] = []
    for token in text.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key in ("1", "true", "t", "yes", "y", "on"):
            vals.append(True)
        elif key in ("0", "false", "f", "no", "n", "off"):
            vals.append(False)
        else:
            raise ValueError(f"Invalid boolean token '{token}'")
    if not vals:
        raise ValueError(f"Expected at least one boolean in '{text}'")
    return vals


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
    coeff_path_override: str | None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    physics_physical = dict(cfg.get("physics_physical", {}))
    physics_physical["omega_n"] = float(physics_physical["omega_n"]) * c.omega_mult
    physics_physical["source_n0"] = float(physics_physical["source_n0"]) * c.source_mult
    cfg["physics_physical"] = physics_physical

    transport = dict(cfg.get("transport", {}))
    transport["Dn"] = float(transport["Dn"]) * c.dn_mult
    transport["DOmega"] = float(transport["DOmega"]) * c.domega_mult
    transport["phi_dissipation_on"] = bool(c.phi_dissipation_on)
    transport["core_vorticity_damping_on"] = bool(c.core_vorticity_damping_on)
    cfg["transport"] = transport

    closures = dict(cfg.get("closures", {}))
    closures["sol_sheath_phi_dissipation_on"] = bool(c.phi_sheath_dissipation_on)
    cfg["closures"] = closures

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

    geom = dict(cfg.get("geometry", {}))
    if grid_override is not None:
        nx, ny, nz = grid_override
        geom["nx"] = int(nx)
        geom["ny"] = int(ny)
        geom["nz"] = int(nz)
    if coeff_path_override:
        geom["coeff_path"] = str(coeff_path_override)
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
    coeff_path_override: str | None,
) -> dict[str, float | int | str]:
    time_cfg = base_cfg.get("time", {})
    dt = float(time_cfg.get("dt", 0.0))
    if dt <= 0.0:
        raise ValueError("Config must contain [time].dt > 0")
    nsteps = int(max(1, math.ceil(t_end / dt)))
    save_every = max(1, nsteps // 20)
    cfg = _make_cfg(
        base_cfg,
        c,
        nsteps,
        save_every,
        grid_override,
        coeff_path_override,
    )
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
            "Calibration loop for Hermes/jax alignment. Runs short windows first, "
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
        "--phi-dissipation-on",
        default="1",
        help="Comma-separated booleans for transport.phi_dissipation_on.",
    )
    parser.add_argument(
        "--phi-sheath-dissipation-on",
        default="1",
        help="Comma-separated booleans for closures.sol_sheath_phi_dissipation_on.",
    )
    parser.add_argument(
        "--core-vorticity-damping-on",
        default="1",
        help="Comma-separated booleans for transport.core_vorticity_damping_on.",
    )
    parser.add_argument(
        "--stages",
        default="0.1,0.5,1.0",
        help=(
            "Comma-separated end times for staged promotion (e.g. 0.1,0.5,1.0). "
            "Only finite-gated candidates are promoted."
        ),
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
        "--grid-short", default="", help="Reduced-grid override for stage-1: nx,ny,nz."
    )
    parser.add_argument(
        "--grid-long",
        default="",
        help="Grid override for stage>=2: nx,ny,nz (default: base config grid).",
    )
    parser.add_argument(
        "--coeff-path-short",
        default="",
        help="Optional stage-1 geometry coeff_path override (reduced-grid coefficients).",
    )
    parser.add_argument(
        "--coeff-path-long",
        default="",
        help="Optional stage>=2 geometry coeff_path override.",
    )
    parser.add_argument(
        "--promote-top-k",
        type=int,
        default=12,
        help="Promote at most top-k finite candidates at each stage.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=40,
        help="Cap number of evaluated candidates (deterministic order).",
    )
    parser.add_argument(
        "--out-csv",
        default="docs/figures/alignment_calibration_scan.csv",
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
    phi_dissipation_on = _parse_bool_list(args.phi_dissipation_on)
    phi_sheath_dissipation_on = _parse_bool_list(args.phi_sheath_dissipation_on)
    core_vorticity_damping_on = _parse_bool_list(args.core_vorticity_damping_on)
    stages = _parse_list(args.stages, float)
    if any(t <= 0.0 for t in stages):
        raise ValueError("All stage end times must be > 0.")
    short_grid = _parse_grid(args.grid_short) if args.grid_short else None
    long_grid = _parse_grid(args.grid_long) if args.grid_long else None
    short_coeff = str(Path(args.coeff_path_short).resolve()) if args.coeff_path_short else None
    long_coeff = str(Path(args.coeff_path_long).resolve()) if args.coeff_path_long else None

    candidates = [
        Candidate(om, src, dn, dw, ps, phi_diss, phi_sheath, core_damp)
        for om, src, dn, dw, ps, phi_diss, phi_sheath, core_damp in product(
            omega_mults,
            source_mults,
            dn_mults,
            domega_mults,
            poisson_scales,
            phi_dissipation_on,
            phi_sheath_dissipation_on,
            core_vorticity_damping_on,
        )
    ]
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    rows: list[dict[str, float | int | str]] = []
    active: list[Candidate] = candidates
    for stage_idx, t_end in enumerate(stages, start=1):
        if not active:
            print(f"[stage {stage_idx}] no active candidates, stopping.")
            break
        grid_override = short_grid if stage_idx == 1 else long_grid
        coeff_path_override = short_coeff if stage_idx == 1 else long_coeff
        stage_rows: list[tuple[Candidate, dict[str, float | int | str]]] = []
        print(
            f"[stage {stage_idx}] t_end={t_end:g} "
            f"candidates={len(active)} grid={grid_override if grid_override else 'base'} "
            f"coeff={coeff_path_override if coeff_path_override else 'base'}"
        )
        for i, c in enumerate(active, start=1):
            print(f"  [{i}/{len(active)}] {c.key()}")
            try:
                row = _evaluate_candidate(
                    base_cfg=base_cfg,
                    c=c,
                    hermes=hermes,
                    t_end=float(t_end),
                    max_growth_factor=float(args.max_growth_factor),
                    max_rms_abs=float(args.max_rms_abs),
                    grid_override=grid_override,
                    coeff_path_override=coeff_path_override,
                )
            except Exception as exc:  # noqa: BLE001
                row = {
                    "candidate": c.key(),
                    "passed": 0,
                    "reason": f"exception:{type(exc).__name__}:{exc}",
                    "score": math.inf,
                    "mean_rel": math.inf,
                    "max_rel": math.inf,
                    "growth": math.inf,
                    "peak": math.inf,
                }
            row["stage"] = stage_idx
            row["t_end"] = float(t_end)
            row["grid"] = (
                "base"
                if grid_override is None
                else f"{grid_override[0]}x{grid_override[1]}x{grid_override[2]}"
            )
            row["rtol_target"] = float(args.rtol_target)
            row["meets_rtol"] = 1 if row.get("max_rel", math.inf) <= float(args.rtol_target) else 0
            stage_rows.append((c, row))
            rows.append(row)

        passing = [(cand, row) for cand, row in stage_rows if int(row.get("passed", 0)) == 1]
        passing.sort(key=lambda cr: float(cr[1]["score"]))
        active = [cand for cand, _ in passing[: max(1, int(args.promote_top_k))]]
        print(f"[stage {stage_idx}] passing={len(passing)} " f"promoted={len(active)}")

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "stage",
        "t_end",
        "grid",
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
