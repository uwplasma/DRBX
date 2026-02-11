"""DRB2D energy-budget diagnostics with curvature and drives.

This example runs a short nonlinear DRB2D simulation and compares a finite-difference
estimate of dE/dt to a term-by-term energy budget computed directly from the RHS.
It includes curvature + background drives so the budget shows non-trivial injection.

Outputs (in --out):
  - panel_budget.png: energy time series and budget closure
  - timeseries.npz: saved time series and budget terms
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.drb2d_em import DRB2DEMModel, DRB2DEMParams, DRB2DEMState
from jaxdrb.nonlinear.drb2d_hot_ion import (
    DRB2DHotIonModel,
    DRB2DHotIonParams,
    DRB2DHotIonState,
)
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.stepper import rk4_step


def run_time_series(
    *,
    model: DRB2DModel,
    y0,
    dt: float,
    tmax: float,
    stride: int,
) -> dict[str, jnp.ndarray]:
    nsteps = int(jnp.ceil(tmax / dt))
    nrec = max(1, nsteps // stride)

    @jax.jit
    def advance_chunk(t: jnp.ndarray, y):
        def body(i, carry):
            t_, y_ = carry
            y_next = rk4_step(y_, t_, dt, model.rhs)
            return (t_ + dt, y_next)

        t_end, y_end = jax.lax.fori_loop(0, stride, body, (t, y))
        return t_end, y_end

    ts = []
    Es = []
    budgets = {
        k: []
        for k in [
            "E_dot_adv",
            "E_dot_parallel",
            "E_dot_curvature",
            "E_dot_drive",
            "E_dot_diss",
            "E_dot_total",
        ]
    }

    t = jnp.asarray(0.0)
    y = y0
    for i in range(nrec):
        t, y = advance_chunk(t, y)
        E = model.energy(y)
        budget = model.energy_budget(y)

        if not (jnp.isfinite(E) & jnp.isfinite(budget["E_dot_total"])):
            raise FloatingPointError(f"Non-finite diagnostics at i={i}, t={float(t):.3f}.")

        ts.append(t)
        Es.append(E)
        for k in budgets:
            budgets[k].append(budget[k])

        if (i + 1) % max(1, nrec // 10) == 0 or i == 0:
            print(
                f"[drb2d-budget] rec {i + 1}/{nrec} t={float(t):.3f} "
                f"E={float(E):.3e} E_dot_total={float(budget['E_dot_total']):+.3e}"
            )

    out = {"t": jnp.stack(ts), "E": jnp.stack(Es)}
    for k, v in budgets.items():
        out[k] = jnp.stack(v)
    return out


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--tmax", type=float, default=20.0)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", type=str, default="cold", choices=["cold", "hot-ion", "em"])
    p.add_argument("--out", type=str, default="out_drb2d_budget")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)

    key = jax.random.key(args.seed)
    shape = (grid.nx, grid.ny)

    if args.model == "cold":
        params = DRB2DParams(
            omega_n=0.8,
            omega_Te=0.3,
            kpar=0.0,
            eta=0.2,
            me_hat=0.2,
            curvature_on=True,
            curvature_coeff=0.6,
            Dn=1e-3,
            DOmega=1e-3,
            DTe=1e-3,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        model = DRB2DModel(params=params, grid=grid)
        y0 = DRB2DState(
            n=1e-3 * jax.random.normal(key, shape),
            omega=1e-3 * jax.random.normal(jax.random.key(args.seed + 1), shape),
            vpar_e=1e-3 * jax.random.normal(jax.random.key(args.seed + 2), shape),
            vpar_i=1e-3 * jax.random.normal(jax.random.key(args.seed + 3), shape),
            Te=1e-3 * jax.random.normal(jax.random.key(args.seed + 4), shape),
        )
    elif args.model == "hot-ion":
        params = DRB2DHotIonParams(
            omega_n=0.8,
            omega_Te=0.3,
            omega_Ti=0.2,
            kpar=0.0,
            eta=0.2,
            me_hat=0.2,
            tau_i=1.0,
            alpha_Te_ohm=1.71,
            alpha_Ti=1.0,
            curvature_on=True,
            curvature_coeff=0.6,
            Dn=1e-3,
            DOmega=1e-3,
            DTe=1e-3,
            DTi=1e-3,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        model = DRB2DHotIonModel(params=params, grid=grid)
        y0 = DRB2DHotIonState(
            n=1e-3 * jax.random.normal(key, shape),
            omega=1e-3 * jax.random.normal(jax.random.key(args.seed + 1), shape),
            vpar_e=1e-3 * jax.random.normal(jax.random.key(args.seed + 2), shape),
            vpar_i=1e-3 * jax.random.normal(jax.random.key(args.seed + 3), shape),
            Te=1e-3 * jax.random.normal(jax.random.key(args.seed + 4), shape),
            Ti=1e-3 * jax.random.normal(jax.random.key(args.seed + 5), shape),
        )
    else:
        params = DRB2DEMParams(
            omega_n=0.8,
            omega_Te=0.3,
            kpar=0.0,
            eta=0.2,
            me_hat=0.2,
            beta=0.2,
            Dpsi=1e-3,
            curvature_on=True,
            curvature_coeff=0.6,
            Dn=1e-3,
            DOmega=1e-3,
            DTe=1e-3,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        model = DRB2DEMModel(params=params, grid=grid)
        y0 = DRB2DEMState(
            n=1e-3 * jax.random.normal(key, shape),
            omega=1e-3 * jax.random.normal(jax.random.key(args.seed + 1), shape),
            psi=1e-3 * jax.random.normal(jax.random.key(args.seed + 2), shape),
            vpar_i=1e-3 * jax.random.normal(jax.random.key(args.seed + 3), shape),
            Te=1e-3 * jax.random.normal(jax.random.key(args.seed + 4), shape),
        )

    print(
        f"[drb2d-budget] grid=({grid.nx},{grid.ny}) dt={args.dt} tmax={args.tmax} "
        f"stride={args.stride} curvature={params.curvature_coeff} omega_n={params.omega_n} "
        f"kpar={params.kpar}"
    )

    series = run_time_series(
        model=model, y0=y0, dt=float(args.dt), tmax=float(args.tmax), stride=int(args.stride)
    )
    suffix = "" if args.model == "cold" else f"_{args.model.replace('-', '_')}"
    jnp.savez(out_dir / f"timeseries{suffix}.npz", **series)

    t = series["t"]
    E = series["E"]
    Edot = series["E_dot_total"]
    dE_dt_fd = jnp.gradient(E, t)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax = axs[0, 0]
    ax.plot(t, E, lw=2, label="E")
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_title("DRB2D energy time series")
    ax.legend()

    ax = axs[0, 1]
    ax.plot(t, dE_dt_fd, lw=2, label=r"$dE/dt$ (FD)")
    ax.plot(t, Edot, lw=2, label=r"$\dot E$ (budget)")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("t")
    ax.set_title("Energy budget closure")
    ax.legend()

    ax = axs[1, 0]
    ax.plot(t, series["E_dot_adv"], lw=2, label="adv")
    ax.plot(t, series["E_dot_parallel"], lw=2, label="parallel")
    ax.plot(t, series["E_dot_curvature"], lw=2, label="curvature")
    ax.plot(t, series["E_dot_drive"], lw=2, label="drive")
    ax.plot(t, series["E_dot_diss"], lw=2, label="diss")
    ax.set_xlabel("t")
    ax.set_title("Budget term decomposition")
    ax.legend()

    ax = axs[1, 1]
    ax.plot(t, series["E_dot_total"], lw=2, label="budget")
    ax.plot(t, dE_dt_fd - series["E_dot_total"], lw=2, label="closure residual")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("t")
    ax.set_title("Budget residual")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"panel_budget{suffix}.png", dpi=220)
    plt.close(fig)

    print(f"[drb2d-budget] wrote {out_dir}")


if __name__ == "__main__":
    main()
