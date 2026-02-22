from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _slice_midplane(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[arr.shape[0] // 2]
    raise ValueError(f"Unsupported snapshot shape: {arr.shape}")


def _plot_field(ax, field: np.ndarray, title: str) -> None:
    im = ax.imshow(field, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


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
    args = parser.parse_args()

    data = np.load(args.input)
    fields = [
        ("snapshot_n", "n"),
        ("snapshot_phi", "phi"),
        ("snapshot_omega", "omega"),
        ("snapshot_Te", "Te"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, fields, strict=False):
        if key not in data:
            ax.axis("off")
            continue
        field = _slice_midplane(data[key])
        _plot_field(ax, field, title)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)


if __name__ == "__main__":
    main()
