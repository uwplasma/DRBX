from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.diagnostics import zonal_mean


def _latest_snapshot(data: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    if key in data:
        arr = np.asarray(data[key])
        return arr
    if f"snapshot_{key}" in data:
        return np.asarray(data[f"snapshot_{key}"])
    if f"snapshots_{key}" in data:
        arr = np.asarray(data[f"snapshots_{key}"])
        return arr[-1]
    return None


def _slice_midplane(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[arr.shape[0] // 2]
    raise ValueError(f"Unsupported snapshot shape: {arr.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot zonal mean profiles from output .npz")
    parser.add_argument("input", help="Path to jax_drb output .npz")
    parser.add_argument(
        "--out",
        default="docs/figures/nonlinear_zonal_profile.png",
        help="Output image path",
    )
    args = parser.parse_args()

    data = np.load(args.input)
    n = _latest_snapshot(data, "n")
    phi = _latest_snapshot(data, "phi")
    omega = _latest_snapshot(data, "omega")
    Te = _latest_snapshot(data, "Te")

    if n is None or phi is None or omega is None or Te is None:
        raise ValueError("Missing snapshots for n/phi/omega/Te.")

    n = np.nan_to_num(_slice_midplane(n), nan=0.0, posinf=0.0, neginf=0.0)
    phi = np.nan_to_num(_slice_midplane(phi), nan=0.0, posinf=0.0, neginf=0.0)
    omega = np.nan_to_num(_slice_midplane(omega), nan=0.0, posinf=0.0, neginf=0.0)
    Te = np.nan_to_num(_slice_midplane(Te), nan=0.0, posinf=0.0, neginf=0.0)

    zn = zonal_mean(n, axis=1)
    zphi = zonal_mean(phi, axis=1)
    zomega = zonal_mean(omega, axis=1)
    zTe = zonal_mean(Te, axis=1)
    x = np.arange(zn.size)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.plot(x, zn, label="zonal n", lw=2)
    ax.plot(x, zTe, label="zonal Te", lw=2)
    ax.plot(x, zomega, label="zonal omega", lw=2)
    ax.plot(x, zphi, label="zonal phi", lw=2)
    ax.set_title("Zonal Mean Profiles")
    ax.set_xlabel("x index")
    ax.set_ylabel("zonal mean")
    ax.legend(loc="best", frameon=False, ncol=2)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
