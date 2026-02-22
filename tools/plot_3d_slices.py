from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import maybe_lowpass


def _latest_snapshot(data: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    if key in data:
        return np.asarray(data[key])
    if f"snapshot_{key}" in data:
        return np.asarray(data[f"snapshot_{key}"])
    if f"snapshots_{key}" in data:
        arr = np.asarray(data[f"snapshots_{key}"])
        return arr[-1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 3D slices from a 3D snapshot.")
    parser.add_argument("input", help="Path to jax_drb output .npz")
    parser.add_argument("--field", default="n", help="Field name")
    parser.add_argument("--out", default="docs/figures/field_aligned_3d.png", help="Output image")
    parser.add_argument(
        "--lowpass",
        type=float,
        default=None,
        help="Optional low-pass fraction (0-1) for smoother visuals.",
    )
    parser.add_argument(
        "--fluct",
        default="none",
        choices=("none", "mean", "zonal"),
        help="Subtract mean or zonal mean before plotting.",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Use symmetric color limits about zero.",
    )
    args = parser.parse_args()

    data = np.load(args.input)
    field = _latest_snapshot(data, args.field)
    if field is None:
        raise ValueError(f"Missing snapshots for '{args.field}'.")
    if field.ndim != 3:
        raise ValueError(f"Expected 3D field, got {field.shape}.")
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    if args.fluct == "mean":
        field = field - np.mean(field)
    elif args.fluct == "zonal":
        field = field - np.mean(field, axis=2, keepdims=True)

    nz, nx, ny = field.shape
    z0, x0, y0 = nz // 2, nx // 2, ny // 2
    slice_xy = maybe_lowpass(field[z0], args.lowpass)
    slice_xz = maybe_lowpass(field[:, :, y0], args.lowpass)
    slice_yz = maybe_lowpass(field[:, x0, :], args.lowpass)

    vmin, vmax = np.percentile(field, [2.0, 98.0])
    if args.symmetric:
        vmax = float(max(abs(vmin), abs(vmax)))
        vmin = -vmax

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=True)
    cmap = "coolwarm" if args.symmetric else "viridis"
    ims = [
        axes[0].imshow(slice_xy, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax),
        axes[1].imshow(slice_xz, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax),
        axes[2].imshow(slice_yz, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax),
    ]
    axes[0].set_title("xy @ z mid")
    axes[1].set_title("xz @ y mid")
    axes[2].set_title("yz @ x mid")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(ims[0], ax=axes, fraction=0.025, pad=0.02)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
