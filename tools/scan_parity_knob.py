from __future__ import annotations

import argparse
import copy
import csv
import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from jaxdrb.benchmarking import finite_run_gate, load_bundle_npz
from jaxdrb.driver import run_simulation

FIELDS = ("n", "Te", "omega", "phi")


def _parse_list(text: str, cast=float) -> list[Any]:
    values: list[Any] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(cast(token))
    if not values:
        raise ValueError(f"Expected at least one value in '{text}'")
    return values


def _set_by_dotted_key(cfg: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    if len(parts) == 1:
        cfg[key] = value
        return
    cur: dict[str, Any] = cfg
    for part in parts[:-1]:
        nxt = cur.get(part, None)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


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


def _prepare_cfg(
    base_cfg: dict[str, Any],
    cfg_path: Path,
    t_end: float,
    dt: float,
    method: str,
    solver: str,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    geom = dict(cfg.get("geometry", {}))
    coeff_path = geom.get("coeff_path", None)
    if isinstance(coeff_path, str) and not Path(coeff_path).is_absolute():
        candidate = (cfg_path.parent / coeff_path).resolve()
        if not candidate.exists():
            if cfg_path.parent.name == "benchmarks":
                repo_root = cfg_path.parents[2]
            else:
                repo_root = cfg_path.parents[0]
            candidate = (repo_root / coeff_path).resolve()
        geom["coeff_path"] = str(candidate)
    cfg["geometry"] = geom

    time_cfg = dict(cfg.get("time", {}))
    nsteps = int(max(1, math.ceil(t_end / dt)))
    time_cfg["dt"] = float(dt)
    time_cfg["nsteps"] = int(nsteps)
    time_cfg["save_every"] = 1
    time_cfg["save_fields"] = True
    time_cfg["snapshot_fields"] = ["n", "Te", "omega", "phi"]
    time_cfg["return_numpy"] = True
    time_cfg["diag_mode"] = "full"
    time_cfg["progress"] = False
    time_cfg["method"] = str(method)
    if str(method).lower() == "diffrax":
        time_cfg["solver"] = str(solver)
        time_cfg["adaptive"] = False
        time_cfg["rtol"] = float(rtol)
        time_cfg["atol"] = float(atol)
        if str(solver).lower() in (
            "implicit_euler",
            "kvaerno3",
            "kvaerno4",
            "kvaerno5",
            "kencarp3",
            "kencarp4",
            "kencarp5",
        ):
            time_cfg["imex_linear_solver"] = "gmres"
            time_cfg["imex_linear_max_steps"] = 25
            time_cfg["imex_linear_restart"] = 20
            time_cfg["imex_root_solver"] = "verychord"
            time_cfg["imex_root_max_steps"] = 6
    cfg["time"] = time_cfg
    return cfg


def _score_candidate(
    diagnostics: dict[str, np.ndarray],
    hermes: dict[str, np.ndarray],
) -> tuple[float, float]:
    t_model = np.asarray(diagnostics["times"], dtype=np.float64)
    target_mask = hermes["times"] <= (t_model[-1] + 1e-12)
    t_target = np.asarray(hermes["times"][target_mask], dtype=np.float64)
    means = []
    maxes = []
    for field in FIELDS:
        key = f"rms_{field}_fluct"
        y_model = np.asarray(diagnostics[key], dtype=np.float64)
        y_target = np.asarray(hermes[key][target_mask], dtype=np.float64)
        mean_rel, max_rel = _channel_rel_error(t_model, y_model, t_target, y_target)
        means.append(mean_rel)
        maxes.append(max_rel)
    return float(np.mean(means)), float(np.max(maxes))


def _load_hermes_bundle(path: Path) -> dict[str, np.ndarray]:
    bundle = load_bundle_npz(path)
    out = {"times": np.asarray(bundle.times_norm, dtype=np.float64)}
    for field in FIELDS:
        out[f"rms_{field}_fluct"] = np.asarray(bundle.diagnostics[f"rms_{field}_fluct"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a single parity knob against Hermes.")
    parser.add_argument("--config", required=True, help="Base jax_drb TOML config.")
    parser.add_argument("--hermes-bundle", required=True, help="Hermes benchmark bundle (.npz).")
    parser.add_argument("--key", required=True, help="Dotted config key to scan.")
    parser.add_argument(
        "--values",
        required=True,
        help="Comma-separated values for the scan.",
    )
    parser.add_argument("--t-end", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--method", default="rk4_imex_strang")
    parser.add_argument("--solver", default="kvaerno5")
    parser.add_argument("--rtol", type=float, default=1e-1)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--max-growth-factor", type=float, default=500.0)
    parser.add_argument("--max-rms-abs", type=float, default=200.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-config", default="")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    base_cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    hermes = _load_hermes_bundle(Path(args.hermes_bundle).resolve())
    values = _parse_list(args.values, float)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "scan_results.csv"

    rows: list[dict[str, Any]] = []
    best = None
    best_cfg = None

    for value in values:
        cfg = _prepare_cfg(
            base_cfg,
            cfg_path,
            args.t_end,
            args.dt,
            args.method,
            args.solver,
            args.rtol,
            args.atol,
        )
        _set_by_dotted_key(cfg, args.key, value)
        result = run_simulation(cfg, as_numpy=True)
        diag = dict(result.diagnostics)
        passed, reason, growth, peak = finite_run_gate(
            diag,
            max_growth_factor=float(args.max_growth_factor),
            max_rms_abs=float(args.max_rms_abs),
        )
        mean_rel = math.inf
        max_rel = math.inf
        if passed:
            mean_rel, max_rel = _score_candidate(diag, hermes)
        row = {
            "value": float(value),
            "passed": int(passed),
            "reason": str(reason),
            "mean_rel": float(mean_rel),
            "max_rel": float(max_rel),
            "growth": float(growth),
            "peak": float(peak),
        }
        rows.append(row)
        if passed:
            if best is None or mean_rel < best["mean_rel"]:
                best = row
                best_cfg = cfg

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["value", "passed", "reason", "mean_rel", "max_rel", "growth", "peak"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if args.out_config and best_cfg is not None:
        out_cfg = Path(args.out_config).resolve()
        try:
            import tomli_w
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("tomli_w is required to write TOML configs.") from exc
        out_cfg.write_text(tomli_w.dumps(best_cfg), encoding="utf-8")

    print(f"Wrote scan results to {out_csv}")
    if best is None:
        print("No passing candidates found.")
    else:
        print(f"Best candidate: value={best['value']} mean_rel={best['mean_rel']:.3e} max_rel={best['max_rel']:.3e}")


if __name__ == "__main__":
    main()
