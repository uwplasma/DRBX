#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gbs_io import plot_cross_coherence, plot_cross_phase, select_steps


def main() -> None:
    p = argparse.ArgumentParser(description="Plot cross coherence and phase diagnostics.")
    p.add_argument("h5", nargs="?", default="results_turb_00.h5")
    p.add_argument("--field1", default="phi")
    p.add_argument("--field2", default="Ne")
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--output-dir", default="gbs_plots")
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--range", dest="value_range", default="-4,4")
    args = p.parse_args()

    h5 = Path(args.h5)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    steps = select_steps(h5, start=args.start, end=args.end, stride=args.stride)
    rng = [float(x.strip()) for x in args.value_range.split(",")]
    if len(rng) != 2:
        raise SystemExit("range must be a,b")
    value_range = (rng[0], rng[1])

    plot_cross_coherence(
        h5,
        args.field1,
        args.field2,
        steps=steps,
        bins=args.bins,
        value_range=value_range,
        output_prefix=outdir / f"cross_{args.field1}_{args.field2}",
    )

    plot_cross_phase(
        h5,
        args.field1,
        args.field2,
        steps=steps,
        output_prefix=outdir / f"cross_phase_{args.field1}_{args.field2}",
    )


if __name__ == "__main__":
    main()
