#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gbs_io import plot_pdf, select_steps


def _parse_pair(value: str) -> tuple[float, float] | None:
    if value.strip().lower() in ("auto", ""):
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected a,b or 'auto'")
    return float(parts[0]), float(parts[1])


def _parse_slice(value: str) -> tuple[int | None, int | None]:
    if not value:
        return (None, None)
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected start,end")
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if parts[1] else None
    return start, end


def main() -> None:
    p = argparse.ArgumentParser(description="Plot PDF diagnostics for GBS outputs.")
    p.add_argument("h5", nargs="?", default="results_turb_00.h5")
    p.add_argument("--fields", default="Ne,phi")
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--range", dest="value_range", default="auto")
    p.add_argument(
        "--x-slice", default="", help="x index slice start,end (inclusive start, exclusive end)"
    )
    p.add_argument(
        "--z-slice", default="", help="z index slice start,end (inclusive start, exclusive end)"
    )
    p.add_argument("--output-dir", default="gbs_plots")
    args = p.parse_args()

    h5 = Path(args.h5)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    steps = select_steps(h5, start=args.start, end=args.end, stride=args.stride)
    value_range = _parse_pair(args.value_range)
    x_slice = _parse_slice(args.x_slice)
    z_slice = _parse_slice(args.z_slice)

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    for field in fields:
        out = outdir / f"pdf_{field}.png"
        plot_pdf(
            h5,
            field,
            steps=steps,
            nbins=args.bins,
            value_range=value_range,
            x_slice=x_slice,
            z_slice=z_slice,
            output=out,
        )


if __name__ == "__main__":
    main()
