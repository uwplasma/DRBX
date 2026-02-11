"""Micro-benchmark: HW2D RHS + Diffrax fixed-step throughput.

Run:
  python benchmarks/bench_hw2d_step.py

This is not a pytest benchmark: it is a quick sanity/performance check that the
nonlinear kernel is JIT-compiling and stepping efficiently.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, hw2d_random_ic
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def main() -> None:
    jax.config.update("jax_enable_x64", False)

    grid = Grid2D.make(nx=128, ny=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    params = HW2DParams(kappa=1.0, alpha=0.5, Dn=2e-4, DOmega=2e-4, bracket="spectral")
    model = HW2DModel(params=params, grid=grid)

    y0 = hw2d_random_ic(jax.random.key(0), grid, amp=1e-3, include_neutrals=False)

    dt = 0.05
    nsteps = 200

    def rhs(t, y):
        return model.rhs(t, y)

    solvers = ["euler", "dopri5", "tsit5"]
    for solver in solvers:
        # Warm-up + compile.
        _, y_end = diffeqsolve_fixed_steps(
            rhs,
            y0=y0,
            t0=0.0,
            dt=dt,
            nsteps=2,
            solver=solver,
        )
        jax.block_until_ready(y_end.n)

        t0 = time.time()
        _, y_end = diffeqsolve_fixed_steps(
            rhs,
            y0=y0,
            t0=0.0,
            dt=dt,
            nsteps=nsteps,
            solver=solver,
        )
        jax.block_until_ready(y_end.n)
        t1 = time.time()

        steps_per_s = nsteps / (t1 - t0)
        print(
            f"HW2D Diffrax {solver}: {steps_per_s:.1f} steps/s for {grid.nx}x{grid.ny}"
        )


if __name__ == "__main__":
    main()
