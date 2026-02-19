"""Verify the FD+CG Poisson solver for Dirichlet and Neumann BCs.

Many nonlinear drift-reduced fluid models include a polarization (Poisson-like) closure.
Once the domain is non-periodic (targets, walls, X-points), FFT solvers are not applicable,
so a robust, differentiable elliptic solver becomes a key ingredient.

This script checks:

1) Dirichlet: u=0 at boundaries, u = sin(pi x) sin(pi y)
2) Neumann:   du/dn=0 at boundaries, u = cos(pi x) cos(pi y) (compared up to a constant)
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.bc import BC2D
from jaxdrb.nonlinear.fd import inv_laplacian_cg, laplacian


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=65)
    p.add_argument("--ny", type=int, default=65)
    p.add_argument("--out", type=str, default="out_poisson_cg_verify")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nx = int(args.nx)
    ny = int(args.ny)
    Lx = 1.0
    Ly = 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x = jnp.linspace(0.0, Lx, nx)
    y = jnp.linspace(0.0, Ly, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Dirichlet case.
    bcD = BC2D.dirichlet(x=0.0, y=0.0)
    uD = jnp.sin(math.pi * X / Lx) * jnp.sin(math.pi * Y / Ly)
    rhsD = laplacian(uD, dx, dy, bcD)
    uD_rec = inv_laplacian_cg(rhsD, dx=dx, dy=dy, bc=bcD, maxiter=1200, tol=1e-12)
    errD = uD_rec - uD

    # Neumann case.
    bcN = BC2D.neumann(x=0.0, y=0.0)
    uN = jnp.cos(math.pi * X / Lx) * jnp.cos(math.pi * Y / Ly)
    rhsN = laplacian(uN, dx, dy, bcN)
    uN_rec = inv_laplacian_cg(rhsN, dx=dx, dy=dy, bc=bcN, maxiter=2000, tol=1e-12)
    uN0 = uN - jnp.mean(uN)
    uN1 = uN_rec - jnp.mean(uN_rec)
    errN = uN1 - uN0

    print(f"[poisson-cg] Dirichlet max|err|={float(jnp.max(jnp.abs(errD))):.3e}")
    print(f"[poisson-cg] Neumann   max|err|={float(jnp.max(jnp.abs(errN))):.3e} (up to constant)")

    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (title, arr) in zip(
        axs[0],
        [
            ("u (Dirichlet)", uD),
            ("u_rec (Dirichlet)", uD_rec),
            ("err (Dirichlet)", errD),
        ],
    ):
        im = ax.imshow(arr.T, origin="lower", cmap="RdBu_r", aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax, (title, arr) in zip(
        axs[1],
        [
            ("u (Neumann, zero-mean)", uN0),
            ("u_rec (Neumann, zero-mean)", uN1),
            ("err (Neumann)", errN),
        ],
    ):
        im = ax.imshow(arr.T, origin="lower", cmap="RdBu_r", aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("FD+CG Poisson solver verification", y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir / "panel.png", dpi=220)
    plt.close(fig)

    print(f"[poisson-cg] wrote {out_dir / 'panel.png'}")


if __name__ == "__main__":
    main()
