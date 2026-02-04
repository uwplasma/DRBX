"""Method of Manufactured Solutions (MMS) for the nonlinear HW2D model.

This example shows how to use MMS to *stress test* spatial discretizations:

  1) Choose analytic fields (n, phi, omega) that are smooth and periodic.
  2) Derive forcing terms (S_n, S_omega) so the *continuum PDE* is satisfied exactly.
  3) Evaluate the *discrete* residual on a sequence of grids to estimate convergence order.

Why this is useful
------------------
MMS validates that:
  - the nonlinear Poisson bracket implementation is correct,
  - the Poisson (polarization) inversion is consistent,
  - boundary-condition handling and diffusion terms behave as expected,
  - code refactors do not silently change the underlying operators.

Run
---
  python examples/scripts/08_nonlinear_hw2d/mms_hw2d_convergence.py

Outputs
-------
  - `residual_convergence.png`: L2 residual vs grid spacing for different brackets.
  - `residual_table.json`: raw numbers (for regression tracking).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams
from jaxdrb.nonlinear.mms_hw2d import MMSHW2D


def mms_error(*, nx: int, bracket: str) -> float:
    Lx = 2 * jnp.pi
    Ly = 2 * jnp.pi
    grid = Grid2D.make(nx=nx, ny=nx, Lx=Lx, Ly=Ly, dealias=False)
    params = HW2DParams(
        kappa=0.7,
        alpha=0.4,
        Dn=1e-3,
        DOmega=2e-3,
        bracket=bracket,  # type: ignore[arg-type]
        poisson="spectral",
        dealias_on=False,
    )
    model = HW2DModel(params=params, grid=grid)

    mms = MMSHW2D(kx=3, ky=2, sigma=0.2, Aphi=0.25, An=0.2, phase=0.3)
    t = 0.4

    x1 = jnp.linspace(0.0, Lx, nx, endpoint=False)
    y1 = jnp.linspace(0.0, Ly, nx, endpoint=False)
    X, Y = jnp.meshgrid(x1, y1, indexing="ij")

    y = mms.state(X, Y, t, Lx=Lx, Ly=Ly)
    dy = model.rhs(t, y)
    f = mms.forcing(
        X,
        Y,
        t,
        Lx=Lx,
        Ly=Ly,
        kappa=params.kappa,
        alpha=params.alpha,
        Dn=params.Dn,
        DOmega=params.DOmega,
    )

    dt_n = mms.sigma * y.n
    dt_w = mms.sigma * y.omega
    rn = dt_n - dy.n - f.n
    rw = dt_w - dy.omega - f.omega
    return float(jnp.sqrt(jnp.mean(rn**2 + rw**2)))


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=str,
        default="out/examples/08_nonlinear_hw2d/mms_hw2d_convergence",
        help="Output directory.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    brackets = ["spectral", "arakawa", "centered"]
    nxs = [16, 32, 64, 128]

    table: dict[str, list[dict[str, float]]] = {}
    for b in brackets:
        rows = []
        for nx in nxs:
            err = mms_error(nx=nx, bracket=b)
            rows.append({"nx": float(nx), "dx": float(2 * jnp.pi / nx), "residual_l2": float(err)})
            print(f"[mms] bracket={b:8s} nx={nx:4d} residual={err:10.3e}")
        table[b] = rows

    (out_dir / "residual_table.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for b in brackets:
        dx = [r["dx"] for r in table[b]]
        err = [r["residual_l2"] for r in table[b]]
        ax.loglog(dx, err, "o-", label=b)

    # Reference slope ~ dx^2.
    ref_dx = jnp.array([table["arakawa"][1]["dx"], table["arakawa"][-1]["dx"]])
    ref_err0 = table["arakawa"][1]["residual_l2"]
    ref = ref_err0 * (ref_dx / ref_dx[0]) ** 2
    ax.loglog(ref_dx, ref, "k--", label=r"$\propto \Delta x^2$")

    ax.set_xlabel(r"grid spacing $\Delta x$")
    ax.set_ylabel(r"MMS residual (L2)")
    ax.set_title("HW2D MMS residual vs grid spacing")
    ax.legend()
    fig.savefig(out_dir / "residual_convergence.png", dpi=220)
    plt.close(fig)

    print(f"[mms] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
