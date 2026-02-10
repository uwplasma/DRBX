"""DRB2D conservative energy drift gate.

Runs the DRB2D conservative testbed (periodic slab, no drives/dissipation)
and enforces a relative energy drift threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.nonlinear.conservative import energy_drift, energy_time_series  # noqa: E402
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState  # noqa: E402
from jaxdrb.nonlinear.grid import Grid2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--Lx", type=float, default=20.0)
    parser.add_argument("--Ly", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=1.0e-2)
    parser.add_argument("--nsteps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rel-span", type=float, default=5e-4)
    parser.add_argument("--max-rel-end", type=float, default=5e-4)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=args.Lx, Ly=args.Ly, dealias=False)
    params = DRB2DParams(
        omega_n=0.0,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=False,
        operator_dissipative_on=False,
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(int(args.seed))
    shape = (grid.nx, grid.ny)
    n = 1e-3 * jax.random.normal(key, shape)
    omega = 1e-3 * jax.random.normal(jax.random.key(1), shape)
    vpar_e = 1e-3 * jax.random.normal(jax.random.key(2), shape)
    vpar_i = 1e-3 * jax.random.normal(jax.random.key(3), shape)
    Te = 1e-3 * jax.random.normal(jax.random.key(4), shape)
    y0 = DRB2DState(n=n, omega=omega, vpar_e=vpar_e, vpar_i=vpar_i, Te=Te)

    E = energy_time_series(
        y0=y0,
        rhs=lambda t, y: model.rhs(t, y),
        energy=model.energy,
        t0=0.0,
        dt=float(args.dt),
        nsteps=int(args.nsteps),
    )
    drift = energy_drift(E)
    metrics = {
        "rel_span": float(drift["rel_span"]),
        "rel_end": float(drift["rel_end"]),
        "E0": float(drift["E0"]),
        "Emin": float(drift["Emin"]),
        "Emax": float(drift["Emax"]),
    }

    print(
        f"[drb2d-gate] rel_span={metrics['rel_span']:.3e} rel_end={metrics['rel_end']:.3e}",
        flush=True,
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2))

    failures: list[str] = []
    if metrics["rel_span"] > float(args.max_rel_span):
        failures.append(
            f"rel_span={metrics['rel_span']:.3e} > {float(args.max_rel_span):.3e}"
        )
    if abs(metrics["rel_end"]) > float(args.max_rel_end):
        failures.append(
            f"rel_end={metrics['rel_end']:.3e} > {float(args.max_rel_end):.3e}"
        )
    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
