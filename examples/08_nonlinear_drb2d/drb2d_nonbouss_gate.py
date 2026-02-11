"""Non-Boussinesq polarization gate for DRB2D.

This script compares Boussinesq vs non-Boussinesq energy in the small-perturbation
limit and validates the non-Boussinesq energy-rate consistency.

Outputs (in --out):
  - nonbouss_energy.png: energy time series (non-Boussinesq run)
  - metrics.json: summary of energy-rate consistency
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.analysis.plotting import save_json, set_mpl_style
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--tmax", type=float, default=10.0)
    p.add_argument("--solver", type=str, default="dopri8")
    p.add_argument("--fixed-step", action="store_true")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--progress", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="out_drb2d_nonbouss")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)

    params_nb = DRB2DParams(
        omega_n=0.0,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        boussinesq=False,
        non_boussinesq_perturbed_density_on=True,
        n0=1.0,
        n0_min=1e-6,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    params_b = DRB2DParams(
        omega_n=0.0,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        boussinesq=True,
        non_boussinesq_perturbed_density_on=True,
        n0=1.0,
        n0_min=1e-6,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    model_nb = DRB2DModel(params=params_nb, grid=grid)
    model_b = DRB2DModel(params=params_b, grid=grid)

    key = jax.random.key(args.seed)
    shape = (grid.nx, grid.ny)
    amp = 1e-6
    y = DRB2DState(
        n=amp * jax.random.normal(key, shape),
        omega=amp * jax.random.normal(jax.random.key(args.seed + 1), shape),
        vpar_e=amp * jax.random.normal(jax.random.key(args.seed + 2), shape),
        vpar_i=amp * jax.random.normal(jax.random.key(args.seed + 3), shape),
        Te=amp * jax.random.normal(jax.random.key(args.seed + 4), shape),
    )

    rhs = model_nb.rhs(0.0, y)
    edot = float(model_nb.energy_rate(y, rhs))
    eps = 1e-6
    y_plus = DRB2DState(
        n=y.n + eps * rhs.n,
        omega=y.omega + eps * rhs.omega,
        vpar_e=y.vpar_e + eps * rhs.vpar_e,
        vpar_i=y.vpar_i + eps * rhs.vpar_i,
        Te=y.Te + eps * rhs.Te,
    )
    y_minus = DRB2DState(
        n=y.n - eps * rhs.n,
        omega=y.omega - eps * rhs.omega,
        vpar_e=y.vpar_e - eps * rhs.vpar_e,
        vpar_i=y.vpar_i - eps * rhs.vpar_i,
        Te=y.Te - eps * rhs.Te,
    )
    edot_fd = float((model_nb.energy(y_plus) - model_nb.energy(y_minus)) / (2 * eps))

    Eb = float(model_b.energy(y))
    Enb = float(model_nb.energy(y))
    rel_E = abs(Eb - Enb) / max(abs(Eb), 1e-12)

    dt = float(args.dt)
    save_ts = jnp.arange(dt, float(args.tmax) + 1e-12, dt)
    sol = model_nb.diffeqsolve(
        y0=y,
        t0=0.0,
        t1=float(args.tmax),
        dt0=dt,
        save_ts=save_ts,
        solver=str(args.solver),
        adaptive=not bool(args.fixed_step),
        rtol=float(args.rtol),
        atol=float(args.atol),
        max_steps=int(args.max_steps),
        progress=bool(args.progress),
    )
    ts = np.asarray(jax.device_get(sol.ts))
    Es = np.asarray(jax.device_get(jax.vmap(model_nb.energy)(sol.ys)))

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.0))
    ax.plot(ts, Es, lw=2)
    ax.set_xlabel("t")
    ax.set_ylabel("E (non-Bouss)")
    ax.set_title("DRB2D non-Boussinesq energy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "nonbouss_energy.png", dpi=220)
    plt.close(fig)

    save_json(
        out_dir / "metrics.json",
        {
            "edot_full": edot,
            "edot_fd": edot_fd,
            "rel_edot": abs(edot - edot_fd) / max(abs(edot_fd), 1e-12),
            "rel_E_bouss_vs_nonbouss": rel_E,
        },
    )
    print(f"[drb2d-nb] rel_edot={abs(edot - edot_fd) / max(abs(edot_fd), 1e-12):.3e}")
    print(f"[drb2d-nb] rel_E_bouss_vs_nonbouss={rel_E:.3e}")
    print(f"[drb2d-nb] wrote {out_dir}")


if __name__ == "__main__":
    main()
