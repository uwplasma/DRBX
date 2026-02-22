from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.diagnostics import pdf_1d


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
    parser = argparse.ArgumentParser(description="Plot PDFs from output .npz")
    parser.add_argument("input", help="Path to jax_drb output .npz")
    parser.add_argument(
        "--out",
        default="docs/figures/nonlinear_pdfs.png",
        help="Output image path",
    )
    parser.add_argument("--bins", type=int, default=80, help="Histogram bins")
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

    fields = {"n": n, "Te": Te, "omega": omega, "phi": phi}

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    for ax, (label, field) in zip(axes.flat, fields.items(), strict=False):
        vals = field - np.mean(field)
        centers, hist = pdf_1d(vals, bins=args.bins)
        ax.plot(centers, hist, lw=2)
        ax.set_title(f"PDF {label}'")
        ax.set_xlabel(label)
        ax.set_ylabel("P")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
