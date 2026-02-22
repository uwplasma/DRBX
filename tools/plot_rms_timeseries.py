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

    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label in [
        ("rms_n", "RMS n"),
        ("rms_Te", "RMS Te"),
        ("rms_omega", "RMS omega"),
        ("rms_phi", "RMS phi"),
    ]:
        if key in data:
            ax.plot(times, data[key], label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("RMS")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)


if __name__ == "__main__":
    main()
