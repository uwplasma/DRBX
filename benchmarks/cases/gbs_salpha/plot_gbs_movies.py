#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from jaxdrb.plot import make_movie_rect, make_movie_poloidal


def main() -> None:
    p = argparse.ArgumentParser(description="Make GBS movies from an HDF5 output file.")
    p.add_argument("h5", nargs="?", default="results_turb_00.h5")
    p.add_argument("--field", default="Ne")
    p.add_argument("--cut", default="pol", choices=["pol", "tor", "rad"])
    p.add_argument("--axes", default="zxy", help="Axis order in HDF5 (default: zxy)")
    p.add_argument("--output", default=None)
    p.add_argument("--poloidal", action="store_true")
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()

    h5_path = Path(args.h5)
    if args.poloidal:
        output = Path(args.output) if args.output else Path(f"movie_{args.field}_poloidal.gif")
        make_movie_poloidal(h5_path, args.field, axes=args.axes, output=output, fps=args.fps)
    else:
        output = Path(args.output) if args.output else Path(f"movie_{args.field}_{args.cut}.gif")
        make_movie_rect(h5_path, args.field, cut=args.cut, axes=args.axes, output=output, fps=args.fps)


if __name__ == "__main__":
    main()
