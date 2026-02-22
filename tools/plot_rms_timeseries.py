from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RMS time series from a jax_drb .npz file.")
    parser.add_argument("input", help="Path to jax_drb output .npz")
    parser.add_argument(
        "--out",
        default="docs/figures/nonlinear_rms_timeseries.png",
        help="Output image path",
    )
    args = parser.parse_args()

    data = np.load(args.input)
    times = data.get("times")
    if times is None:
        raise ValueError("Missing 'times' array in output.")

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    series = [
        ("rms_n", "RMS n"),
        ("rms_Te", "RMS Te"),
        ("rms_omega", "RMS omega"),
        ("rms_phi", "RMS phi"),
    ]
    for (key, label), color in zip(series, colors, strict=False):
        if key in data:
            ax.plot(times, data[key], label=label, color=color, lw=2)
    ax.set_title("RMS Time Series")
    ax.set_xlabel("t")
    ax.set_ylabel("RMS")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
