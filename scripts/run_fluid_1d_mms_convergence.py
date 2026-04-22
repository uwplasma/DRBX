#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jax_drb.validation.fluid_1d_mms_convergence import (
    build_fluid_1d_mms_convergence_report,
)

build_convergence_report = build_fluid_1d_mms_convergence_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a fast fluid-1D manufactured-solution convergence campaign and emit a JSON report."
    )
    parser.add_argument("--resolution", dest="resolutions", action="append", type=int, default=[], help="Interior ny resolution to include. Repeat for multiple resolutions.")
    parser.add_argument("--timestep", type=float, default=0.05, help="Timestep for each stored step.")
    parser.add_argument("--steps", type=int, default=2, help="Number of stored steps to advance before comparing to the exact solution.")
    parser.add_argument("--substeps", type=int, default=20, help="RK4 substeps per stored step.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    resolutions = tuple(args.resolutions) if args.resolutions else (32, 64, 128)
    report = build_fluid_1d_mms_convergence_report(
        resolutions=resolutions,
        timestep=float(args.timestep),
        steps=int(args.steps),
        substeps=int(args.substeps),
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
