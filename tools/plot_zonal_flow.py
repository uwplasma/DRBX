from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_frames(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key in data:
        return np.asarray(data[key])
    snap = key.replace("snapshots_", "snapshot_")
    if snap in data:
        return np.asarray(data[snap])[None, ...]
    raise ValueError(f"Missing '{key}' or '{snap}' in output.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot zonal flow profile from snapshots.")
    parser.add_argument("input", help="Path to jax_drb output .npz")
    parser.add_argument("--config", required=True, help="Path to input TOML")
    parser.add_argument("--field", default="snapshots_phi", help="Snapshot field key")
    parser.add_argument(
        "--n-average",
        type=int,
        default=20,
        help="Number of last frames to average (0=use last frame only).",
    )
    parser.add_argument(
        "--out",
        default="docs/figures/nonlinear_zonal_flow.png",
        help="Output image path",
    )
    args = parser.parse_args()

    data = np.load(args.input)
    frames = _load_frames(data, args.field)
    frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)

    if frames.ndim == 4:
        # Use midplane z for 3D fields
        frames = frames[:, frames.shape[1] // 2]

    n_avg = int(args.n_average)
    if n_avg > 0 and frames.shape[0] > n_avg:
        frames = frames[-n_avg:]
    zonal_phi = np.mean(frames, axis=2)
    zonal_phi = np.mean(zonal_phi, axis=0)

    # Grid from config
    try:
        from jaxdrb.io import load_config

        cfg = load_config(args.config)
        geom = cfg.data.get("geometry", {})
        Lx = float(geom.get("Lx", 1.0))
    except Exception:
        Lx = 1.0
    nx = zonal_phi.shape[0]
    x = np.linspace(0.0, Lx, nx, endpoint=True)
    dx = x[1] - x[0] if nx > 1 else 1.0
    v_ey = -np.gradient(zonal_phi, dx, axis=0)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(x, v_ey, color="#1f77b4", lw=2.0, label=r"$v_{E,y}$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$v_{E,y}$")
    ax.set_title("Zonal Flow (time-averaged)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
