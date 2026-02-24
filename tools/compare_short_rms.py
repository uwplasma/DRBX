from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare short-window RMS between Hermes and jax_drb."
    )
    parser.add_argument("--hermes", required=True, help="Hermes RMS npz file (times + rms_*).")
    parser.add_argument("--jax", required=True, help="jax_drb RMS npz file (times + rms_*).")
    parser.add_argument(
        "--out-plot", default="docs/figures/short_rms_compare.png", help="Output plot path."
    )
    parser.add_argument(
        "--out-csv",
        default="docs/figures/short_rms_compare.csv",
        help="Output CSV summary path.",
    )
    parser.add_argument(
        "--metric",
        choices=("fluct", "total"),
        default="fluct",
        help="Compare fluctuation RMS channels (default) or total RMS channels.",
    )
    args = parser.parse_args()

    h = np.load(args.hermes)
    j = np.load(args.jax)
    th = np.asarray(h["times"])
    tj = np.asarray(j["times"])

    suffix = "_fluct" if args.metric == "fluct" else ""
    fields = [
        (f"rms_n{suffix}", "RMS n", "#1f77b4"),
        (f"rms_Te{suffix}", "RMS Te", "#ff7f0e"),
        (f"rms_omega{suffix}", "RMS omega", "#2ca02c"),
        (f"rms_phi{suffix}", "RMS phi", "#d62728"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    rows = []
    for key, label, color in fields:
        if key not in h or key not in j:
            continue
        yh = np.asarray(h[key])
        yj = np.asarray(j[key])
        yj_h = np.interp(th, tj, yj)
        rel_end = (yj_h[-1] - yh[-1]) / (abs(yh[-1]) + 1e-12)
        slope_h = (yh[-1] - yh[0]) / max(th[-1] - th[0], 1e-30)
        slope_j = (yj_h[-1] - yj_h[0]) / max(th[-1] - th[0], 1e-30)
        rows.append((key, yh[0], yh[-1], yj_h[0], yj_h[-1], rel_end, slope_h, slope_j))

        ax.plot(th, yh, "--", lw=2, color=color, label=f"Hermes {label}")
        ax.plot(tj, yj, "-", lw=1.8, color=color, alpha=0.8, label=f"jax_drb {label}")

    title_metric = "fluctuation RMS" if args.metric == "fluct" else "RMS"
    ax.set_title(f"{title_metric} comparison")
    ax.set_xlabel("t")
    ax.set_ylabel("RMS")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)

    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=220)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("field,hermes_start,hermes_end,jax_start,jax_end,rel_end,slope_hermes,slope_jax\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"Saved plot: {out_plot}")
    print(f"Saved table: {out_csv}")


if __name__ == "__main__":
    main()
