#!/usr/bin/env python3
"""Resumable stage driver of the P09 nodal curvature campaign (M6).

Stages per arm (``raw`` reported, ``filtered`` gated) and resolution (each its own process, one at a time):

1. ``references`` -- exact nodal fields and curvature RHS of every catalogue case (``curv_references.py``; MMS via the lean reference);
2. ``static``     -- the three variants on the exact fields, plane-chunked (``curv_static.py``);
3. ``audit``      -- scalar energy identity, discrete divergence per region, W2 interchange (``curv_audit.py``);
4. ``refreeze``   -- the P08 nodal curvature reference, prescribed arm (``curv_refreeze.py``);
5. ``reduce``     -- N - R tables, orders, regions, gates over every finished resolution (``curv_reduce.py``);
6. ``spectrum``   -- linearised spectrum and RK4 dt (N32; ``curv_audit.py --parts spectrum``, one variant per process);
7. ``eigensolve`` -- the analytic-eigensolve investigation (``curv_eigensolve.py``).

    python campaign.py run [--arms raw,filtered] [--grids 32,48,64]
    python campaign.py status

A stage is skipped when its receipt exists (``--force`` reruns).
"""
from __future__ import annotations

import curv_common as C

import argparse
import subprocess
import sys
import time

HERE = C.HERE
SCRIPT = {"references": "curv_references.py", "static": "curv_static.py", "audit": "curv_audit.py", "refreeze": "curv_refreeze.py"}
RECEIPT = {"references": "references_receipt.json", "static": "static_receipt.json", "audit": "audit/w2.json",
           "refreeze": "nodal_curvature_reference_manifest.json"}


def done(arm, stage, n):
    return (C.arm_dir(arm, n) / RECEIPT[stage]).exists()


def run_stage(stage, arm, n, extra=()):
    cmd = [sys.executable, str(HERE / SCRIPT[stage]), str(n), "--arm", arm, *extra]
    print(f"[{time.strftime('%H:%M:%S')}] stage {stage} {arm} N{n}", flush=True)
    return subprocess.run(cmd, check=False).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("run", "status"))
    ap.add_argument("--arms", default=",".join(C.ARM_ROOT))
    ap.add_argument("--grids", default=",".join(map(str, C.GRIDS)))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    arms, grids = a.arms.split(","), [int(x) for x in a.grids.split(",")]
    if a.command == "status":
        for arm in arms:
            for n in grids:
                print(arm, n, {s: done(arm, s, n) for s in RECEIPT})
        return 0
    for stage in ("references", "static", "audit", "refreeze"):
        for arm in arms:
            for n in grids:
                if done(arm, stage, n) and not a.force:
                    print(f"skip {stage} {arm} N{n}")
                    continue
                if run_stage(stage, arm, n, ("--parts", "energy,divergence,w2") if stage == "audit" else ()) != 0:
                    print(f"stage {stage} {arm} N{n} failed", file=sys.stderr)
                    return 1
    subprocess.run([sys.executable, str(HERE / "curv_reduce.py"), "--arms", ",".join(arms)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
