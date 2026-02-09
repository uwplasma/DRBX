"""Core-kernel performance regression gate.

This script benchmarks two hot paths used throughout nonlinear and linear workflows:

1) HW2D nonlinear RK4 stepping throughput (steps/s),
2) matrix-free linear `J·v` throughput via repeated matvecs (matvec/s).

It is designed for CI regression gating with conservative thresholds.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import State as ColdState
from jaxdrb.models.cold_ion_drb import equilibrium
from jaxdrb.models.params import DRBParams
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, hw2d_random_ic
from jaxdrb.nonlinear.stepper import rk4_scan


def _bench_hw2d_steps_per_second(*, nx: int, ny: int, nsteps: int, repeats: int) -> float:
    grid = Grid2D.make(nx=nx, ny=ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    params = HW2DParams(kappa=1.0, alpha=0.5, Dn=2e-4, DOmega=2e-4, bracket="spectral")
    model = HW2DModel(params=params, grid=grid)
    y0 = hw2d_random_ic(jax.random.key(0), grid, amp=1e-3, include_neutrals=False)

    @jax.jit
    def run_once(y):
        _, y_end = rk4_scan(y, t0=0.0, dt=0.05, nsteps=nsteps, rhs=model.rhs)
        return y_end

    # Compile.
    y = run_once(y0)
    jax.block_until_ready(y.n)

    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        y = run_once(y)
        jax.block_until_ready(y.n)
        t1 = time.perf_counter()
        rates.append(nsteps / max(t1 - t0, 1e-12))
    return float(jnp.median(jnp.asarray(rates)))


def _bench_linear_matvecs_per_second(*, nl: int, niter: int, repeats: int) -> float:
    geom = SlabGeometry.make(nl=nl, shat=0.4, curvature0=0.2, length=2 * jnp.pi)
    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )
    y_eq = equilibrium(nl)
    v0 = ColdState.random(jax.random.key(1), nl, amplitude=1e-3)
    matvec = linear_matvec(y_eq, params, geom, kx=0.0, ky=0.3)

    @jax.jit
    def apply_many(v):
        def body(i, x):
            return matvec(x)

        return jax.lax.fori_loop(0, niter, body, v)

    # Compile.
    v = apply_many(v0)
    jax.block_until_ready(v.n)

    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        v = apply_many(v)
        jax.block_until_ready(v.n)
        t1 = time.perf_counter()
        rates.append(niter / max(t1 - t0, 1e-12))
    return float(jnp.median(jnp.asarray(rates)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Core-kernel performance regression gate.")
    p.add_argument("--min-hw2d-steps-s", type=float, default=8.0)
    p.add_argument("--min-matvec-s", type=float, default=120.0)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", False)
    warnings.filterwarnings(
        "ignore", message="Explicitly requested dtype .* is not available, and will be truncated.*"
    )

    hw2d_rate = _bench_hw2d_steps_per_second(nx=96, ny=96, nsteps=120, repeats=3)
    matvec_rate = _bench_linear_matvecs_per_second(nl=48, niter=300, repeats=3)

    metrics = {
        "hw2d_steps_per_second": hw2d_rate,
        "linear_matvecs_per_second": matvec_rate,
        "thresholds": {
            "min_hw2d_steps_per_second": float(args.min_hw2d_steps_s),
            "min_linear_matvecs_per_second": float(args.min_matvec_s),
        },
    }

    print(
        "[perf-gate] "
        f"HW2D={hw2d_rate:.1f} steps/s (min {args.min_hw2d_steps_s:.1f}), "
        f"Linear matvec={matvec_rate:.1f} /s (min {args.min_matvec_s:.1f})"
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2))

    failures = []
    if hw2d_rate < args.min_hw2d_steps_s:
        failures.append(
            f"HW2D throughput regression: {hw2d_rate:.1f} < {float(args.min_hw2d_steps_s):.1f}"
        )
    if matvec_rate < args.min_matvec_s:
        failures.append(
            f"Linear matvec throughput regression: {matvec_rate:.1f} < {float(args.min_matvec_s):.1f}"
        )
    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
