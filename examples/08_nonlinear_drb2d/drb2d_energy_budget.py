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
import numpy as np

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.drb2d_em import DRB2DEMModel, DRB2DEMParams, DRB2DEMState
from jaxdrb.nonlinear.drb2d_hot_ion import (
    DRB2DHotIonModel,
    DRB2DHotIonParams,
    DRB2DHotIonState,
)
from jaxdrb.nonlinear.grid import Grid2D


def run_time_series(
    *,
    model,
    y0,
    dt: float,
    tmax: float,
    save_stride: int,
    solver: str,
    adaptive: bool,
    rtol: float,
    atol: float,
    max_steps: int,
    progress: bool,
) -> dict[str, jnp.ndarray]:
    frame_dt = float(dt) * int(save_stride)
    save_ts = jnp.arange(frame_dt, float(tmax) + 1e-12, frame_dt)

    sol = model.diffeqsolve(
        y0=y0,
        t0=0.0,
        t1=float(tmax),
        dt0=float(dt),
        save_ts=save_ts,
        solver=solver,
        adaptive=bool(adaptive),
        rtol=float(rtol),
        atol=float(atol),
        max_steps=int(max_steps),
        progress=bool(progress),
    )

    @jax.jit
    def diagnostics(ys):
        E = jax.vmap(model.energy)(ys)
        budget = jax.vmap(model.energy_budget)(ys)
        return E, budget

    E, budget = diagnostics(sol.ys)
    if not (jnp.all(jnp.isfinite(E)) & jnp.all(jnp.isfinite(budget["E_dot_total"]))):
        raise FloatingPointError("Non-finite diagnostics encountered in DRB2D energy-budget run.")

    out = {"t": sol.ts, "E": E}
    for k, v in budget.items():
        out[k] = v
    return out


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--tmax", type=float, default=20.0)
    p.add_argument("--save-stride", type=int, default=10)
    p.add_argument("--solver", type=str, default="dopri8")
    p.add_argument("--fixed-step", action="store_true")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--max-steps", type=int, default=250_000)
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
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
        f"save_stride={args.save_stride} solver={args.solver} adaptive={not args.fixed_step} "
        f"curvature={params.curvature_coeff} omega_n={params.omega_n} kpar={params.kpar}"
    )

    series = run_time_series(
        model=model,
        y0=y0,
        dt=float(args.dt),
        tmax=float(args.tmax),
        save_stride=int(args.save_stride),
        solver=str(args.solver),
        adaptive=not bool(args.fixed_step),
        rtol=float(args.rtol),
        atol=float(args.atol),
        max_steps=int(args.max_steps),
        progress=bool(args.progress),
    )
    suffix = "" if args.model == "cold" else f"_{args.model.replace('-', '_')}"
    jnp.savez(out_dir / f"timeseries{suffix}.npz", **series)

    t = np.asarray(jax.device_get(series["t"]))
    E = np.asarray(jax.device_get(series["E"]))
    Edot = np.asarray(jax.device_get(series["E_dot_total"]))
    dE_dt_fd = np.gradient(E, t)

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
    ax.plot(t, np.asarray(jax.device_get(series["E_dot_adv"])), lw=2, label="adv")
    ax.plot(t, np.asarray(jax.device_get(series["E_dot_parallel"])), lw=2, label="parallel")
    ax.plot(t, np.asarray(jax.device_get(series["E_dot_curvature"])), lw=2, label="curvature")
    ax.plot(t, np.asarray(jax.device_get(series["E_dot_drive"])), lw=2, label="drive")
    ax.plot(t, np.asarray(jax.device_get(series["E_dot_diss"])), lw=2, label="diss")
    ax.set_xlabel("t")
    ax.set_title("Budget term decomposition")
    ax.legend()

    ax = axs[1, 1]
    ax.plot(t, Edot, lw=2, label="budget")
    ax.plot(t, dE_dt_fd - Edot, lw=2, label="closure residual")
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
