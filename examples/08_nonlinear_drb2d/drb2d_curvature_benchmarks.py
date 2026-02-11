"""DRB2D curvature-drive benchmarks (interchange vs resistive-like coupling).

We estimate linear growth rates from short nonlinear runs (small amplitude)
and compare curvature trends for:
  - interchange-like (kpar=0, eta=0)
  - resistive-like (kpar>0, eta>0)

Outputs (in --out):
  - drb2d_curvature_benchmarks.png
  - curvature_benchmarks.npz
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
from jaxdrb.nonlinear.grid import Grid2D


def _growth_rate(E: np.ndarray, t: np.ndarray) -> float:
    mask = t <= (0.5 * t[-1])
    slope, _ = np.polyfit(t[mask], np.log(np.maximum(E[mask], 1e-30)), 1)
    return 0.5 * float(slope)


def _run_growth(model: DRB2DModel, y0: DRB2DState, dt: float, nsteps: int) -> float:
    y = y0
    E = []
    t = []
    for k in range(nsteps):
        rhs = model.rhs(0.0, y)
        y = DRB2DState(
            n=y.n + dt * rhs.n,
            omega=y.omega + dt * rhs.omega,
            vpar_e=y.vpar_e + dt * rhs.vpar_e,
            vpar_i=y.vpar_i + dt * rhs.vpar_i,
            Te=y.Te + dt * rhs.Te,
        )
        E.append(float(model.energy(y)))
        t.append((k + 1) * dt)
    return _growth_rate(np.asarray(E), np.asarray(t))


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--nsteps", type=int, default=120)
    p.add_argument("--out", type=str, default="out_drb2d_curvature_benchmarks")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)

    key = jax.random.key(0)
    shape = (grid.nx, grid.ny)
    amp = 1e-6
    y0 = DRB2DState(
        n=amp * jax.random.normal(key, shape),
        omega=amp * jax.random.normal(jax.random.key(1), shape),
        vpar_e=jnp.zeros_like(jax.random.normal(jax.random.key(2), shape)),
        vpar_i=jnp.zeros_like(jax.random.normal(jax.random.key(3), shape)),
        Te=amp * jax.random.normal(jax.random.key(4), shape),
    )

    curv_vals = np.linspace(0.0, 0.64, 5)
    gamma_interchange = []
    gamma_resistive = []

    for curv in curv_vals:
        params_interchange = DRB2DParams(
            omega_n=0.0,
            omega_Te=0.0,
            kpar=0.0,
            eta=0.0,
            me_hat=0.2,
            curvature_on=True,
            curvature_coeff=float(curv),
            Dn=0.0,
            DOmega=0.0,
            DTe=0.0,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        params_resistive = DRB2DParams(
            omega_n=0.0,
            omega_Te=0.0,
            kpar=0.3,
            eta=0.2,
            me_hat=0.2,
            curvature_on=True,
            curvature_coeff=float(curv),
            Dn=0.0,
            DOmega=0.0,
            DTe=0.0,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        gamma_interchange.append(
            _run_growth(DRB2DModel(params=params_interchange, grid=grid), y0, args.dt, args.nsteps)
        )
        gamma_resistive.append(
            _run_growth(DRB2DModel(params=params_resistive, grid=grid), y0, args.dt, args.nsteps)
        )
        print(
            f"[drb2d-curv] curv={curv:.2f} "
            f"gamma_int={gamma_interchange[-1]:.3e} "
            f"gamma_res={gamma_resistive[-1]:.3e}"
        )

    gamma_interchange = np.asarray(gamma_interchange)
    gamma_resistive = np.asarray(gamma_resistive)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
    ax.plot(curv_vals, gamma_interchange, "o-", lw=2, label="interchange (kpar=0, eta=0)")
    ax.plot(curv_vals, gamma_resistive, "s-", lw=2, label="resistive-like (kpar=0.3, eta=0.2)")
    ax.set_xlabel("curvature coefficient")
    ax.set_ylabel("linear growth proxy γ")
    ax.set_title("DRB2D curvature-drive benchmark")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "drb2d_curvature_benchmarks.png", dpi=220)
    plt.close(fig)

    np.savez(
        out_dir / "curvature_benchmarks.npz",
        curvature=curv_vals,
        gamma_interchange=gamma_interchange,
        gamma_resistive=gamma_resistive,
    )
    print(f"[drb2d-curv] wrote {out_dir}")


if __name__ == "__main__":
    main()
