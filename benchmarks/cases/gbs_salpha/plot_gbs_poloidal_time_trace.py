#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gbs_io import make_poloidal_time_trace_movie, select_steps


def main() -> None:
    p = argparse.ArgumentParser(description="Create a poloidal time trace movie from GBS output.")
    p.add_argument("h5", nargs="?", default="results_turb_00.h5")
    p.add_argument("--field", default="Ne")
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--output", default="poloidal_time_trace.gif")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--x-index", type=int)
    p.add_argument("--y-index", type=int)
    p.add_argument("--z-index", type=int)
    args = p.parse_args()

    steps = select_steps(Path(args.h5), start=args.start, end=args.end, stride=args.stride)
    make_poloidal_time_trace_movie(
        Path(args.h5),
        args.field,
        steps=steps,
        output=Path(args.output),
        fps=args.fps,
        x_index=args.x_index,
        y_index=args.y_index,
        z_index=args.z_index,
    )


if __name__ == "__main__":
    main()
