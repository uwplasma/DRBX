"""Compare Diffrax solvers on a short DRB2D nonlinear run.

This script produces a small solver-comparison figure used in the docs/README.
It measures accuracy vs a higher-accuracy reference and records runtime per solver.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def _make_state(key, shape, amp: float) -> DRB2DState:
    return DRB2DState(
        n=amp * jax.random.normal(key, shape),
        omega=amp * jax.random.normal(jax.random.key(1), shape),
        vpar_e=amp * jax.random.normal(jax.random.key(2), shape),
        vpar_i=amp * jax.random.normal(jax.random.key(3), shape),
        Te=amp * jax.random.normal(jax.random.key(4), shape),
    )


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()
    jax.config.update("jax_enable_x64", False)

    out_dir = Path("out_drb2d_solver_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=48, ny=48, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    params = DRB2DParams(
        omega_n=0.8,
        omega_Te=0.3,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=True,
        curvature_coeff=0.6,
        Dn=1e-3,
        DOmega=1e-3,
        DTe=1e-3,
        Dn4=1e-5,
        DOmega4=1e-5,
        DTe4=1e-5,
        mu_zonal_omega=0.1,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=True,
    )
    model = DRB2DModel(params=params, grid=grid)
    y0 = _make_state(jax.random.key(0), (grid.nx, grid.ny), amp=3e-3)

    dt = 0.03
    nsteps = 200
    tmax = dt * nsteps
    t = np.linspace(dt, tmax, nsteps)

    def run_solver(*, solver: str, dt_local: float, nsteps_local: int) -> tuple[np.ndarray, float]:
        def rhs(ti, yi):
            return model.rhs(ti, yi)

        # Warm-up compile.
        _, y_end = diffeqsolve_fixed_steps(rhs, y0=y0, t0=0.0, dt=dt_local, nsteps=2, solver=solver)
        jax.block_until_ready(y_end.n)

        t0 = time.perf_counter()
        ys, y_end = diffeqsolve_fixed_steps(
            rhs,
            y0=y0,
            t0=0.0,
            dt=dt_local,
            nsteps=nsteps_local,
            solver=solver,
        )
        jax.block_until_ready(y_end.n)
        t1 = time.perf_counter()

        Es = np.asarray(jax.device_get(jax.vmap(model.energy)(ys)))
        return Es, t1 - t0

    # Reference run: smaller time step.
    ref_dt = dt / 3.0
    ref_steps = int(nsteps * 3)
    E_ref, _ = run_solver(solver="dopri8", dt_local=ref_dt, nsteps_local=ref_steps)
    E_ref = E_ref[::3]

    solvers = ["euler", "dopri5", "tsit5", "dopri8"]
    errors = {}
    runtimes = {}
    energies = {}
    for solver in solvers:
        Es, wall = run_solver(solver=solver, dt_local=dt, nsteps_local=nsteps)
        energies[solver] = Es
        runtimes[solver] = wall
        errors[solver] = np.abs(Es - E_ref) / (np.abs(E_ref) + 1e-12)

    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.2))
    ax = axs[0]
    for solver in solvers:
        ax.plot(t, errors[solver], label=solver)
    ax.set_xlabel("t")
    ax.set_ylabel("relative |E - E_ref|")
    ax.set_title("DRB2D solver accuracy vs reference")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axs[1]
    names = list(solvers)
    vals = [runtimes[s] for s in names]
    ax.bar(names, vals, color="tab:blue", alpha=0.85)
    ax.set_ylabel("wall time (s)")
    ax.set_title("DRB2D fixed-step runtime")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "drb2d_solver_comparison.png", dpi=220)
    np.savez(
        out_dir / "drb2d_solver_comparison.npz",
        t=t,
        solvers=np.array(solvers),
        errors=np.vstack([errors[s] for s in solvers]),
        runtimes=np.array([runtimes[s] for s in solvers]),
        E_ref=E_ref,
    )
    print(f"[drb2d-solver] wrote {out_dir}")


if __name__ == "__main__":
    main()
