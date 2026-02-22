from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.diagnostics import isotropic_spectrum


def _latest_snapshot(data: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    if key in data:
        return np.asarray(data[key])
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
    parser = argparse.ArgumentParser(description="Plot isotropic spectra from output .npz")
    parser.add_argument("input", help="Path to jax_drb output .npz")
    parser.add_argument(
        "--out",
        default="docs/figures/nonlinear_spectrum.png",
        help="Output image path",
    )
    parser.add_argument("--dx", type=float, default=1.0, help="Grid spacing in x")
    parser.add_argument("--dy", type=float, default=1.0, help="Grid spacing in y")
    args = parser.parse_args()

    data = np.load(args.input)
    n = _latest_snapshot(data, "n")
    omega = _latest_snapshot(data, "omega")
    Te = _latest_snapshot(data, "Te")
    phi = _latest_snapshot(data, "phi")

    if n is None or omega is None or Te is None or phi is None:
        raise ValueError("Missing snapshots for n/omega/Te/phi.")

    n = np.nan_to_num(_slice_midplane(n), nan=0.0, posinf=0.0, neginf=0.0)
    omega = np.nan_to_num(_slice_midplane(omega), nan=0.0, posinf=0.0, neginf=0.0)
    Te = np.nan_to_num(_slice_midplane(Te), nan=0.0, posinf=0.0, neginf=0.0)
    phi = np.nan_to_num(_slice_midplane(phi), nan=0.0, posinf=0.0, neginf=0.0)

    spectra = {
        "n": isotropic_spectrum(n, dx=args.dx, dy=args.dy),
        "Te": isotropic_spectrum(Te, dx=args.dx, dy=args.dy),
        "omega": isotropic_spectrum(omega, dx=args.dx, dy=args.dy),
        "phi": isotropic_spectrum(phi, dx=args.dx, dy=args.dy),
    }

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
    for label, spec in spectra.items():
        ax.loglog(spec.k + 1e-12, spec.power + 1e-16, label=label, lw=2)
    ax.set_title("Isotropic Power Spectra")
    ax.set_xlabel("k")
    ax.set_ylabel("P(k)")
    ax.legend(loc="best", frameon=False, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
