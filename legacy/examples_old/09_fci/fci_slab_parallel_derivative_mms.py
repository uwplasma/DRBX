"""FCI slab MMS-style check: parallel derivative convergence.

This script demonstrates the key FCI idea on a simple slab:

- fields are represented on perpendicular (x,y) planes,
- the parallel derivative is built by mapping a point to the next/previous plane along B,
  interpolating at the mapped footpoints, then differencing along the field line.

We use an analytic constant-B map so the only errors come from interpolation + finite differencing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.fci.map import SlabFCIConfig, make_slab_fci_map
from jaxdrb.fci.parallel import parallel_derivative_centered


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="out_fci_slab_mms")
    p.add_argument("--nx0", type=int, default=48, help="Base x resolution (refinement doubles it).")
    p.add_argument("--ny0", type=int, default=48, help="Base y resolution (refinement doubles it).")
    p.add_argument("--dz0", type=float, default=0.4)
    p.add_argument("--nref", type=int, default=5, help="Number of dz refinements.")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    Lx = 2 * jnp.pi
    Ly = 2 * jnp.pi

    kx = 2.0
    ky = 3.0
    kz = -1.0
    Bx = 0.4
    By = 0.2
    Bz = 1.0
    Bnorm = float(jnp.sqrt(Bx**2 + By**2 + Bz**2))

    def f(x, y, z):
        return jnp.sin(kx * x + ky * y + kz * z)

    def dpar_exact(x, y):
        phase = kx * x + ky * y
        return ((Bx * kx + By * ky + Bz * kz) / Bnorm) * jnp.cos(phase)

    dzs = []
    errs = []

    dz = float(args.dz0)
    for r in range(int(args.nref)):
        nx = int(args.nx0) * (2**r)
        ny = int(args.ny0) * (2**r)
        dx = float(Lx / nx)
        dy = float(Ly / ny)

        xs = jnp.arange(nx) * dx
        ys = jnp.arange(ny) * dy
        X, Y = jnp.meshgrid(xs, ys, indexing="ij")

        cfg = SlabFCIConfig(
            x0=0.0,
            y0=0.0,
            dx=dx,
            dy=dy,
            nx=nx,
            ny=ny,
            dz=dz,
            Bx=Bx,
            By=By,
            Bz=Bz,
        )
        fwd, bwd = make_slab_fci_map(cfg)

        fk = f(X, Y, 0.0)
        fkp = f(X, Y, dz)
        fkm = f(X, Y, -dz)
        dnum = parallel_derivative_centered(fk, f_kp1=fkp, f_km1=fkm, map_fwd=fwd, map_bwd=bwd)
        dex = dpar_exact(X, Y)

        rel = jnp.sqrt(jnp.mean((dnum - dex) ** 2)) / jnp.maximum(jnp.sqrt(jnp.mean(dex**2)), 1e-12)
        dzs.append(dz)
        errs.append(float(rel))
        print(f"[fci-mms] level={r} nx={nx} dz={dz:.4f} rel_L2={float(rel):.3e}")
        dz = dz / 2.0

    dzs = jnp.array(dzs)
    errs = jnp.array(errs)
    jnp.savez(out_dir / "mms.npz", dz=dzs, rel_err=errs)

    # Estimate observed order p from successive refinements.
    p_obs = jnp.log(errs[:-1] / errs[1:]) / jnp.log(2.0)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
    ax.loglog(dzs, errs, "o-", lw=2, label="FCI centered derivative")
    ax.loglog(dzs, errs[0] * (dzs / dzs[0]) ** 2, "--", lw=1.5, label="O(dz^2) guide")
    ax.set_xlabel(r"$\Delta z$")
    ax.set_ylabel("relative L2 error")
    ax.set_title("FCI slab MMS-style convergence (constant B)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.6))
    ax.plot(dzs[1:], p_obs, "o-", lw=2)
    ax.axhline(2.0, color="k", lw=1.0, alpha=0.5)
    ax.set_xlabel(r"$\Delta z$")
    ax.set_ylabel("observed order")
    ax.set_title("Observed convergence order")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "order.png", dpi=220)
    plt.close(fig)

    print(f"[fci-mms] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
