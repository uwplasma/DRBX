from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import maybe_lowpass


def _slice_midplane(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[arr.shape[0] // 2]
    raise ValueError(f"Unsupported snapshot shape: {arr.shape}")


def _limits(field: np.ndarray, *, symmetric: bool) -> tuple[float, float]:
    flat = field.ravel()
    vmin, vmax = np.percentile(flat, [2.0, 98.0])
    if symmetric:
        vmax = float(max(abs(vmin), abs(vmax)))
        vmin = -vmax
    return float(vmin), float(vmax)


def _plot_field(ax, field: np.ndarray, title: str, *, cmap: str, symmetric: bool) -> None:
    vmin, vmax = _limits(field, symmetric=symmetric)
    im = ax.imshow(field, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.tick_params(length=3, width=0.8)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.035)


def _fluctuation(field: np.ndarray, mode: str, key: str) -> np.ndarray:
    if mode == "auto":
        if key in ("snapshot_phi", "snapshot_omega"):
            return field - np.mean(field)
        return field - np.mean(field, axis=1, keepdims=True)
    if mode == "none":
        return field
    if mode == "mean":
        return field - np.mean(field)
    if mode == "zonal":
        return field - np.mean(field, axis=1, keepdims=True)
    raise ValueError(f"Unknown fluctuation mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot nonlinear snapshot panel from a jax_drb .npz file."
    )
    parser.add_argument("input", help="Path to jax_drb output .npz")
    parser.add_argument(
        "--out",
        default="docs/figures/nonlinear_panel.png",
        help="Output image path",
    )
    parser.add_argument(
        "--fluct",
        default="auto",
        choices=("auto", "none", "mean", "zonal"),
        help="Subtract mean or zonal mean before plotting.",
    )
    parser.add_argument(
        "--lowpass",
        type=float,
        default=None,
        help="Optional low-pass fraction (0-1) for smoother visuals.",
    )
    args = parser.parse_args()

    data = np.load(args.input)
    fields = [
        ("snapshot_n", "n"),
        ("snapshot_phi", "phi"),
        ("snapshot_omega", "omega"),
        ("snapshot_Te", "Te"),
    ]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    for idx, (ax, (key, title)) in enumerate(zip(axes.flat, fields, strict=False)):
        if key not in data:
            ax.axis("off")
            continue
        field = _slice_midplane(data[key])
        field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
        field = _fluctuation(field, args.fluct, key)
        field = maybe_lowpass(field, args.lowpass)
        cmap = "viridis" if key in ("snapshot_n", "snapshot_Te") else "coolwarm"
        symmetric = key in ("snapshot_phi", "snapshot_omega")
        label = f"{title}'" if args.fluct != "none" else title
        _plot_field(ax, field, label, cmap=cmap, symmetric=symmetric)
        if idx // 2 == 1:
            ax.set_xlabel("x")
        if idx % 2 == 0:
            ax.set_ylabel("y")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
