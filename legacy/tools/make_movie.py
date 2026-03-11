from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import maybe_lowpass


def _extract_frames(arr: np.ndarray, z_index: int | None) -> np.ndarray:
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        z0 = z_index if z_index is not None else arr.shape[1] // 2
        return arr[:, z0]
    raise ValueError(f"Unsupported snapshot shape: {arr.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a GIF movie from snapshots.")
    parser.add_argument("input", help="Path to jax_drb output .npz")
    parser.add_argument("--field", default="snapshots_n", help="Snapshot field key")
    parser.add_argument("--out", default="docs/figures/blob_movie.gif", help="Output GIF")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--z-index", type=int, default=None, help="Z index for 3D fields")
    parser.add_argument(
        "--fluct",
        default="none",
        choices=("none", "mean", "zonal"),
        help="Subtract mean or zonal mean before plotting.",
    )
    parser.add_argument(
        "--lowpass",
        type=float,
        default=None,
        help="Optional low-pass fraction (0-1) for smoother visuals.",
    )
    parser.add_argument(
        "--skip-fraction",
        type=float,
        default=0.0,
        help="Skip this fraction of early frames (0-1) to avoid linear transients.",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Use symmetric color limits about zero.",
    )
    parser.add_argument(
        "--range-tail",
        action="store_true",
        help="Compute color limits from the last portion of frames.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.2,
        help="Fraction of final frames to use for color limits.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Explicit vmin for the color range.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Explicit vmax for the color range.",
    )
    args = parser.parse_args()

    data = np.load(args.input)
    if args.field not in data:
        raise ValueError(f"Missing '{args.field}' in output.")
    frames = _extract_frames(np.asarray(data[args.field]), args.z_index)
    frames = frames[:: max(args.stride, 1)]
    if args.skip_fraction > 0.0:
        start = int(max(0, min(frames.shape[0] - 1, args.skip_fraction * frames.shape[0])))
        frames = frames[start:]
    frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
    if args.fluct != "none":
        if args.fluct == "mean":
            frames = frames - frames.mean(axis=(1, 2), keepdims=True)
        elif args.fluct == "zonal":
            frames = frames - frames.mean(axis=2, keepdims=True)
    if args.lowpass:
        frames = np.stack([maybe_lowpass(frame, args.lowpass) for frame in frames], axis=0)

    range_frames = frames
    if args.range_tail and frames.shape[0] > 1:
        frac = float(args.tail_fraction)
        frac = min(max(frac, 0.05), 1.0)
        start = int(max(0, frames.shape[0] * (1.0 - frac)))
        range_frames = frames[start:]
    vmin, vmax = np.percentile(range_frames, [2.0, 98.0])
    if args.vmin is not None:
        vmin = float(args.vmin)
    if args.vmax is not None:
        vmax = float(args.vmax)
    if args.symmetric:
        vmax = float(max(abs(vmin), abs(vmax)))
        vmin = -vmax
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    cmap = "coolwarm" if args.symmetric else "viridis"
    im = ax.imshow(frames[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(args.field.replace("snapshots_", ""))
    ax.set_xticks([])
    ax.set_yticks([])

    try:
        from matplotlib import animation

        def update(i):
            im.set_data(frames[i])
            return (im,)

        anim = animation.FuncAnimation(fig, update, frames=frames.shape[0], blit=True)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        anim.save(out, writer="pillow", fps=12)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
