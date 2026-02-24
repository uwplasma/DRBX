from __future__ import annotations

import argparse
import copy
import math
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from jaxdrb.benchmarking import finite_run_gate
from jaxdrb.driver import run_simulation


def _parse_stages(text: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid stage token '{token}'. Use name:t_end")
        name, value = token.split(":", 1)
        name = name.strip()
        t_end = float(value.strip())
        if t_end <= 0.0:
            raise ValueError(f"Stage '{name}' has non-positive t_end={t_end}")
        out.append((name, t_end))
    if not out:
        raise ValueError("No valid stages provided.")
    return out


def _save_run_npz(path: Path, result) -> None:
    payload = dict(result.diagnostics)
    if "times" not in payload:
        payload["times"] = np.asarray(result.times)
    if "t" not in payload:
        payload["t"] = np.asarray(result.times)
    np.savez(path, **payload)


def _run_build_bundle(repo_root: Path, cfg_path: Path, input_npz: Path, output_npz: Path) -> None:
    cmd = [
        sys.executable,
        "tools/build_benchmark_bundle.py",
        "--code",
        "jax",
        "--input",
        str(input_npz),
        "--output",
        str(output_npz),
        "--config",
        str(cfg_path),
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run staged jax_drb benchmark windows with finite-run gating."
    )
    parser.add_argument("--config", required=True, help="Base jax_drb TOML config.")
    parser.add_argument(
        "--stages",
        default="short:0.5,onset:1.0,saturated:3.0",
        help="Comma-separated stage list: name:t_end (normalized time).",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for stage files.")
    parser.add_argument(
        "--max-growth-factor",
        type=float,
        default=200.0,
        help="Finite-run gate: maximum growth factor across fluctuation RMS channels.",
    )
    parser.add_argument(
        "--max-rms-abs",
        type=float,
        default=20.0,
        help="Finite-run gate: maximum absolute fluctuation RMS.",
    )
    parser.add_argument(
        "--hermes-bundle",
        default="",
        help="Optional Hermes benchmark bundle for side-by-side stage panels.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with cfg_path.open("rb") as f:
        base_cfg: dict[str, Any] = tomllib.load(f)

    time_cfg = base_cfg.get("time", {})
    if not isinstance(time_cfg, dict) or "dt" not in time_cfg:
        raise ValueError("Config must contain [time].dt")
    dt = float(time_cfg["dt"])
    if dt <= 0.0:
        raise ValueError("time.dt must be > 0")

    stage_defs = _parse_stages(args.stages)
    for stage_name, t_end in stage_defs:
        cfg = copy.deepcopy(base_cfg)
        tcfg = dict(cfg.get("time", {}))
        nsteps = int(max(1, math.ceil(t_end / dt)))
        tcfg["nsteps"] = nsteps
        tcfg.setdefault("save_every", max(1, nsteps // 200))
        tcfg["save_fields"] = True
        tcfg.setdefault("snapshot_fields", ["n", "Te", "omega", "phi"])
        tcfg["return_numpy"] = True
        cfg["time"] = tcfg

        print(f"[stage={stage_name}] running nsteps={nsteps} dt={dt} t_end={t_end}")
        result = run_simulation(cfg, as_numpy=True)
        diagnostics = dict(result.diagnostics)

        passed, reason, growth, peak = finite_run_gate(
            diagnostics,
            max_growth_factor=float(args.max_growth_factor),
            max_rms_abs=float(args.max_rms_abs),
        )
        print(
            f"[stage={stage_name}] gate passed={passed} reason={reason} "
            f"growth={growth:.3e} peak={peak:.3e}"
        )

        run_npz = out_dir / f"jax_{stage_name}.npz"
        _save_run_npz(run_npz, result)

        bundle_npz = out_dir / f"bundle_jax_{stage_name}.npz"
        _run_build_bundle(repo_root, cfg_path, run_npz, bundle_npz)

        if args.hermes_bundle:
            panel_png = out_dir / f"panel_{stage_name}.png"
            subprocess.run(
                [
                    sys.executable,
                    "tools/plot_benchmark_panel.py",
                    "--hermes",
                    str(Path(args.hermes_bundle).resolve()),
                    "--jax",
                    str(bundle_npz),
                    "--out",
                    str(panel_png),
                ],
                cwd=repo_root,
                check=True,
            )

        if not passed:
            print(f"[stage={stage_name}] stopping due to finite-run gate failure")
            break


if __name__ == "__main__":
    main()
