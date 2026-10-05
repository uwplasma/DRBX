#!/usr/bin/env python3
"""Resumable stage driver of the P09 nodal-bracket static gates (M3c).

Stages per resolution (each its own process; the heavy ones one N at a time):

1. ``extract``    -- nodal HSX metric of the family-A layout (``extract_metric.py``);
2. ``references`` -- exact nodal fields and brackets (``references.py``; ``p05`` needs the provider, ~3 GB);
3. ``static``     -- production ``sbp_bracket`` and no-dissipation control on the nodes (``static.py``);
4. ``reduce``     -- global errors, orders, gates, regions, reference budget over all finished resolutions (``reduce.py``);
5. ``spectral``   -- energy identity, abscissa, |lambda|max / dt, ``c_kappa`` sweep per resolution (``spectral.py``; the Cayley
   band and the RK4 power iteration are N32 only: ``spectral.py 32 --parts cayley,rk4``), then ``spectral.py --summarize``;
6. ``refreeze``   -- the P08 nodal bracket reference, prescribed-phi arm (``refreeze.py``).

    python campaign.py run --out ROOT [--grids 32,48,64]
    python campaign.py run-stage --stage static --n 32 --out ROOT
    python campaign.py status --out ROOT

A stage is skipped when its receipt exists (``N{n}/<stage>_receipt.json``, ``summary.json`` for ``reduce``); ``--force`` reruns.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "configuration.json").read_text())
GRIDS = tuple(CONFIG["resolutions"])
STAGES = ("extract", "references", "static", "spectral", "refreeze")
SCRIPT = {"extract": "extract_metric.py", "references": "references.py", "static": "static.py", "reduce": "reduce.py",
          "spectral": "spectral.py", "refreeze": "refreeze.py"}
RECEIPT = {"extract": "extract_receipt.json", "references": "references_receipt.json", "static": "static_receipt.json",
           "spectral": "spectral/sweep.json", "refreeze": "nodal_bracket_reference_manifest.json"}


def done(out: Path, stage: str, n: int | None) -> bool:
    if stage == "reduce":
        return (out / "summary.json").exists()
    return (out / f"N{n}" / RECEIPT[stage]).exists()


def run_stage(stage: str, n: int | None, out: Path, extra=()) -> int:
    cmd = [sys.executable, str(HERE / SCRIPT[stage])] + ([str(n)] if n is not None else []) + ["--out", str(out), *extra]
    print(f"[{time.strftime('%H:%M:%S')}] stage {stage} N{n}: {' '.join(cmd[1:])}", flush=True)
    return subprocess.run(cmd, check=False).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("run", "run-stage", "status"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", choices=STAGES + ("reduce",))
    ap.add_argument("--n", type=int)
    ap.add_argument("--grids", default=",".join(map(str, GRIDS)))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    grids = [int(x) for x in args.grids.split(",")]
    if args.command == "status":
        for n in grids:
            print(n, {s: done(args.out, s, n) for s in STAGES})
        print("reduce", done(args.out, "reduce", None))
        return 0
    if args.command == "run-stage":
        return run_stage(args.stage, args.n, args.out)
    for stage in STAGES:                                   # stage-major: all extractions first, one process each
        for n in grids:
            if done(args.out, stage, n) and not args.force:
                print(f"skip {stage} N{n} (receipt exists)")
                continue
            if run_stage(stage, n, args.out) != 0:
                print(f"stage {stage} N{n} failed", file=sys.stderr)
                return 1
    if not done(args.out, "reduce", None) or args.force:
        return run_stage("reduce", None, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
