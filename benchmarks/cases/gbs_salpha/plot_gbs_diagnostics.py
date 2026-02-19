#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from jaxdrb.plot import plot_0d_time_traces


def main() -> None:
    p = argparse.ArgumentParser(description="Plot GBS 0D diagnostics time traces.")
    p.add_argument("h5", nargs="?", default="results_turb_00.h5")
    p.add_argument("--fields", default="globtheta,globtemperature,globomega")
    p.add_argument("--output", default="diagnostics_0d.png")
    args = p.parse_args()

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    plot_0d_time_traces(Path(args.h5), fields=fields, output=args.output)


if __name__ == "__main__":
    main()
