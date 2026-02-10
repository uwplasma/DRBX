"""
DRB2D conservative nonlinear gate (periodic slab).

Runs a short nonlinear evolution of the 2D conservative DRB testbed (no drives/dissipation,
no parallel coupling) and verifies energy stability.
"""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, set_mpl_style
from jaxdrb.nonlinear.conservative import energy_drift, energy_time_series
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def main() -> None:
    out_dir = Path("out/examples/08_nonlinear_drb2d/drb2d_conservative_gate")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_mpl_style()

    grid = Grid2D.make(nx=48, ny=48, Lx=20.0, Ly=20.0, dealias=False)
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

    key = jax.random.key(0)
    shape = (grid.nx, grid.ny)
    n = 1e-3 * jax.random.normal(key, shape)
    omega = 1e-3 * jax.random.normal(jax.random.key(1), shape)
    vpar_e = 1e-3 * jax.random.normal(jax.random.key(2), shape)
    vpar_i = 1e-3 * jax.random.normal(jax.random.key(3), shape)
    Te = 1e-3 * jax.random.normal(jax.random.key(4), shape)
    y0 = DRB2DState(n=n, omega=omega, vpar_e=vpar_e, vpar_i=vpar_i, Te=Te)

    dt = 1e-2
    nsteps = 250
    E = energy_time_series(
        y0=y0,
        rhs=lambda t, y: model.rhs(t, y),
        energy=model.energy,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
    )
    drift = energy_drift(E)

    import matplotlib.pyplot as plt

    t = np.arange(1, nsteps + 1) * dt
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
    ax.plot(t, np.asarray(E), lw=2.0)
    ax.set_xlabel("t")
    ax.set_ylabel("E")
    ax.set_title("DRB2D conservative energy time series")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "drb2d_conservative_energy.png", dpi=220)
    plt.close(fig)

    assets_dir = ROOT / "docs" / "assets" / "images"
    if assets_dir.exists():
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
        ax.plot(t, np.asarray(E), lw=2.0)
        ax.set_xlabel("t")
        ax.set_ylabel("E")
        ax.set_title("DRB2D conservative energy time series")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(assets_dir / "drb2d_conservative_energy.png", dpi=220)
        plt.close(fig)

    save_json(
        out_dir / "metrics.json",
        {
            "rel_span": float(drift["rel_span"]),
            "rel_end": float(drift["rel_end"]),
            "E0": float(drift["E0"]),
            "Emin": float(drift["Emin"]),
            "Emax": float(drift["Emax"]),
            "dt": dt,
            "nsteps": nsteps,
            "grid": {"nx": grid.nx, "ny": grid.ny, "Lx": grid.Lx, "Ly": grid.Ly},
            "params": params.__dict__,
        },
    )
    print(f"rel_span={float(drift['rel_span']):.3e} rel_end={float(drift['rel_end']):.3e}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
