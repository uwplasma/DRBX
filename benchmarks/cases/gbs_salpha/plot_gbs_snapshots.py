#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gbs_io import plot_snapshot, plot_poloidal


def main() -> None:
    p = argparse.ArgumentParser(description="Plot GBS snapshots from an HDF5 output file.")
    p.add_argument("h5", nargs="?", default="results_turb_00.h5")
    p.add_argument("--fields", default="Ne,Te,vpare,vpari,omega,phi")
    p.add_argument("--cut", default="pol", choices=["pol", "tor", "rad"])
    p.add_argument("--axes", default="zxy", help="Axis order in HDF5 (default: zxy)")
    p.add_argument("--output-dir", default="gbs_plots")
    p.add_argument("--poloidal", action="store_true", default=True, help="Also make poloidal plots")
    args = p.parse_args()

    h5_path = Path(args.h5)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    for field in fields:
        out = outdir / f"{field}_{args.cut}.png"
        plot_snapshot(h5_path, field, cut=args.cut, axes=args.axes, output=out)
        if args.poloidal:
            out_pol = outdir / f"{field}_poloidal.png"
            plot_poloidal(h5_path, field, axes=args.axes, output=out_pol)


if __name__ == "__main__":
    main()
