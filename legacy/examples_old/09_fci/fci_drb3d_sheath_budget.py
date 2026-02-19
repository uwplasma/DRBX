"""FCI DRB3D sheath-budget demo: energy/mass decay under sheath damping."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.fci.drb3d import FCIDRB3DModel, FCIDRB3DParams, FCIDRB3DState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="out_fci_drb3d_sheath")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--nsteps", type=int, default=120)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = FCISlabGrid.make(
        nx=48,
        ny=48,
        nz=24,
        Lx=2 * jnp.pi,
        Ly=2 * jnp.pi,
        Lz=6.0,
        Bx=0.2,
        By=0.1,
        Bz=1.0,
        open_field_line=True,
    )
    params = FCIDRB3DParams(
        kappa=0.0,
        alpha=0.0,
        kpar=0.0,
        Dn=0.0,
        DOmega=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        sheath_nu=1.2,
    )
    model = FCIDRB3DModel(params=params, grid=grid)

    n0 = 2e-3 * jnp.ones((grid.nz, grid.nx, grid.ny))
    omega0 = jnp.zeros((grid.nz, grid.nx, grid.ny))
    y0 = FCIDRB3DState(n=n0, omega=omega0)

    ys, _ = diffeqsolve_fixed_steps(
        rhs=lambda t, y: model.rhs(t, y),
        y0=y0,
        t0=0.0,
        dt=float(args.dt),
        nsteps=int(args.nsteps),
        solver="dopri5",
        save_every=1,
        max_steps=int(args.nsteps * 2 + 50),
    )

    def energy_of(y):
        return model.energy(y)

    def mass_of(y):
        return jnp.mean(y.n)

    E = jnp.array(
        [energy_of(FCIDRB3DState(n=ys.n[i], omega=ys.omega[i])) for i in range(ys.n.shape[0])]
    )
    M = jnp.array(
        [mass_of(FCIDRB3DState(n=ys.n[i], omega=ys.omega[i])) for i in range(ys.n.shape[0])]
    )
    t = jnp.arange(1, ys.n.shape[0] + 1) * float(args.dt)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.8))
    ax.semilogy(t, E / E[0], label="E/E0")
    ax.semilogy(t, M / M[0], label="M/M0")
    ax.set_xlabel("t")
    ax.set_ylabel("normalized diagnostics")
    ax.set_title("FCI DRB3D sheath damping (energy + mass)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fci_drb3d_sheath_budget.png", dpi=220)
    plt.close(fig)

    print(f"[fci-drb3d] wrote {out_dir / 'fci_drb3d_sheath_budget.png'}")


if __name__ == "__main__":
    main()
