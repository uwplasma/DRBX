#!/usr/bin/env python3
"""Temporal self-convergence study of the HSX FCI Braginskii IMEX step.

Runs the HSX blob driver at ``--num-steps N, 2N, 4N, 8N`` to the same final
time (float64 history, tight GMRES), then prints per-field differences
``|u_N - u_2N|/|u|`` (control-volume weighted L2) and observed orders.

Variants are selected with the diagnostic switches of
:mod:`drbx.fci_braginskii.run` (all defaults reproduce production):

``--scheme``            ``ssp222`` (production), ``ars222`` (globally stiffly
                        accurate), ``ark2`` (ESDIRK stage order 2, stiffly
                        accurate; staged execution only)
``--stage-sweeps``      block Gauss-Seidel sweeps between the material wall
                        block and the current/phi pair inside one stage
``--material-newton``   frozen-Jacobian Newton iterations of the material
                        wall backward-Euler block
``--smooth-sign``       tanh width replacing the sharp characteristic split

Example::

    python scripts/hsx_imex_time_order.py --geometry <bundle> \\
        --final-time 0.003 --base-steps 4 --tag ssp222 --out-dir order/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

FIELDS = ("density", "Te", "Ti", "phi", "Vi", "Ve", "vorticity")


def _child(argv: list[str]) -> None:
    import functools

    from drbx.fci_braginskii import run as drv
    from drbx.fci_braginskii.native import fci_parallel_production_flux as flux

    scheme, sweeps, newton, smooth = argv[:4]
    drv.IMEX_SCHEME = scheme
    drv.IMPLICIT_STAGE_SWEEPS = int(sweeps)
    drv.MATERIAL_NEWTON_ITERATIONS = int(newton)
    flux._DIAG_SMOOTH_SIGN_WIDTH = float(smooth)
    drv.run_full_eb = functools.partial(drv.run_full_eb, history_dtype="float64")
    drv.main(argv[4:])


def order_table(paths: list[Path]) -> str:
    data = [np.load(path) for path in paths]
    weight = data[0]["owner_aggregate_volume"] * data[0]["owner_active"]

    def norm(x):
        return float(np.sqrt(np.sum(weight * x * x)))

    lines = ["field | " + " | ".join(
        f"d{i}" for i in range(len(data) - 1)
    ) + " | " + " | ".join(f"p{i}" for i in range(len(data) - 2))]
    for field in FIELDS:
        values = [d[field][-1].astype(np.float64) for d in data]
        reference = norm(values[-1])
        diffs = [norm(b - a) for a, b in zip(values, values[1:])]
        orders = [np.log2(x / y) for x, y in zip(diffs, diffs[1:])]
        lines.append(
            f"{field} | " + " | ".join(f"{x / reference:.3e}" for x in diffs)
            + " | " + " | ".join(f"{p:.2f}" for p in orders)
        )
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--final-time", type=float, default=0.003)
    parser.add_argument("--base-steps", type=int, default=4)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--tag", default="ssp222")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--scheme", default="ssp222")
    parser.add_argument("--stage-sweeps", type=int, default=1)
    parser.add_argument("--material-newton", type=int, default=0)
    parser.add_argument("--smooth-sign", type=float, default=0.0)
    args, extra = parser.parse_known_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for level in range(args.levels):
        steps = args.base_steps * 2 ** level
        out = args.out_dir / f"{args.tag}_n{steps}.npz"
        paths.append(out)
        if out.exists():
            continue
        cmd = [
            sys.executable, __file__, "--child", args.scheme,
            str(args.stage_sweeps), str(args.material_newton),
            str(args.smooth_sign), "--geometry", str(args.geometry),
            "--final-time", str(args.final_time), "--num-steps", str(steps),
            "--save-every", str(steps), "--output", str(out),
            "--gmres-target-tolerance", "1e-10",
            "--gmres-acceptance-tolerance", "1e-8",
            "--gmres-max-iterations", "2000", "--gmres-restart", "200",
            "--no-phase-timing", *extra,
        ]
        with open(out.with_suffix(".log"), "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    print(order_table(paths))


if __name__ == "__main__":
    main()
