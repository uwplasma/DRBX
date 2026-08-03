#!/usr/bin/env python3
"""Focused developer diagnostics for the FCI embedded-control-volume path.

This wrapper deliberately keeps expensive attribution runs out of automated
convergence tests.  It uses the same CLI driver as the tests, writes reports
outside ``tests/``, and makes one- versus four-shard comparisons explicit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SHIFTED_TORUS_DRIVER = ROOT / "tests" / "test_fci_cutwall_shifted_torus_4field.py"


def _run_driver(
    *,
    resolution: int,
    shard_counts: tuple[int, int, int],
    debug: bool,
    report: Path | None,
) -> int:
    command = [
        sys.executable,
        str(SHIFTED_TORUS_DRIVER),
        "--resolutions",
        str(int(resolution)),
        "--shard-counts",
        *(str(int(value)) for value in shard_counts),
        "--enable-agglomeration",
        "--operator-convergence-only",
        "--skip-operator-phi-solve",
    ]
    if debug:
        command.append("--debug-operator-failures")
    if report is None:
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"wrote diagnostic report: {report}")
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("geometry", "worst-face", "operator", "compare-shards"),
    )
    parser.add_argument("--resolution", type=int, default=40)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "output" / "fci_diagnostics")
    args = parser.parse_args()
    if args.resolution <= 0:
        parser.error("--resolution must be positive")

    if args.command == "geometry":
        raise SystemExit(
            _run_driver(
                resolution=args.resolution,
                shard_counts=(1, 1, 1),
                debug=False,
                report=args.report_dir / f"geometry_n{args.resolution}.txt",
            )
        )
    if args.command in {"worst-face", "operator"}:
        raise SystemExit(
            _run_driver(
                resolution=args.resolution,
                shard_counts=(1, 1, 1),
                debug=True,
                report=args.report_dir / f"{args.command}_n{args.resolution}.txt",
            )
        )
    status = _run_driver(
        resolution=args.resolution,
        shard_counts=(1, 1, 1),
        debug=True,
        report=args.report_dir / f"compare_single_n{args.resolution}.txt",
    )
    status |= _run_driver(
        resolution=args.resolution,
        shard_counts=(1, 1, 4),
        debug=True,
        report=args.report_dir / f"compare_1x1x4_n{args.resolution}.txt",
    )
    raise SystemExit(status)


if __name__ == "__main__":
    main()
