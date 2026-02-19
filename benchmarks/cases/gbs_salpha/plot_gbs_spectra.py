#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gbs_io import plot_power_spectrum


def main() -> None:
    p = argparse.ArgumentParser(description="Plot GBS 1D power spectra.")
    p.add_argument("h5", nargs="?", default="results_turb_00.h5")
    p.add_argument("--field", default="Ne")
    p.add_argument("--axis", default="y", choices=["x", "y", "z"])
    p.add_argument("--axes", default="zxy", help="Axis order in HDF5 (default: zxy)")
    p.add_argument("--backend", default="numpy", choices=["numpy", "jax"])
    p.add_argument("--output", default=None)
    args = p.parse_args()

    h5_path = Path(args.h5)
    output = Path(args.output) if args.output else Path(f"spectrum_{args.field}_k{args.axis}.png")
    plot_power_spectrum(h5_path, args.field, axis=args.axis, axes=args.axes, backend=args.backend, output=output)


if __name__ == "__main__":
    main()
