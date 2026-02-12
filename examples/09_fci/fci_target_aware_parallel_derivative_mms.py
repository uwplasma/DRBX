"""FCI slab MMS-style check: target-aware parallel derivative near plates.

This script validates the *target-aware* parallel derivative operator used for open
field lines that intersect material plates.

We use a simple slab with straight field lines (no in-plane shift) and Dirichlet
plates at z=±Lz/2. The grid uses **cell-centered planes** so the target lies at a
half-step beyond the first/last planes, matching the common FCI picture where
field lines hit the plate between perpendicular planes.

The target-aware operator switches to a non-uniform second-order stencil near the
plates using distance-to-target metadata carried by the map.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.bc import BC1D
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.fci.parallel import parallel_derivative_target_aware_3d


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="out_fci_target_bc_mms")
    p.add_argument("--nx", type=int, default=48)
    p.add_argument("--ny", type=int, default=48)
    p.add_argument("--nz0", type=int, default=16)
    p.add_argument("--nref", type=int, default=5, help="Number of nz refinements (doubles nz).")
    p.add_argument("--Lz", type=float, default=6.0)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    Lx = 2 * jnp.pi
    Ly = 2 * jnp.pi
    Lz = float(args.Lz)

    kx = 2.0
    ky = 3.0

    dzs = []
    errs = []

    for r in range(int(args.nref)):
        nz = int(args.nz0) * (2**r)
        grid = FCISlabGrid.make(
            nx=int(args.nx),
            ny=int(args.ny),
            nz=nz,
            Lx=float(Lx),
            Ly=float(Ly),
            Lz=float(Lz),
            Bx=0.0,
            By=0.0,
            Bz=1.0,
            open_field_line=True,
            cell_centered=True,
        )
        bc = BC1D.dirichlet(left=0.0, right=0.0, nu=0.0)

        xs = grid.x0 + grid.dx * jnp.arange(grid.nx)
        ys = grid.y0 + grid.dy * jnp.arange(grid.ny)
        X, Y = jnp.meshgrid(xs, ys, indexing="ij")

        z = grid.l
        z_left = -0.5 * float(Lz)
        phase_xy = kx * X + ky * Y
        sin_xy = jnp.sin(phase_xy)
        sin_z = jnp.sin(jnp.pi * (z - z_left) / float(Lz))
        cos_z = jnp.cos(jnp.pi * (z - z_left) / float(Lz))
        f = sin_xy[None, :, :] * sin_z[:, None, None]

        dnum = parallel_derivative_target_aware_3d(
            f,
            map_fwd=grid.map_fwd,
            map_bwd=grid.map_bwd,
            open_field_line=True,
            bc=bc,
        )
        dex = sin_xy[None, :, :] * (jnp.pi / float(Lz)) * cos_z[:, None, None]

        rel = jnp.sqrt(jnp.mean((dnum - dex) ** 2)) / jnp.maximum(jnp.sqrt(jnp.mean(dex**2)), 1e-12)
        dzs.append(float(grid.dz))
        errs.append(float(rel))
        print(f"[fci-target-mms] level={r} nz={nz} dz={grid.dz:.4f} rel_L2={float(rel):.3e}")

    dzs = jnp.array(dzs)
    errs = jnp.array(errs)
    jnp.savez(out_dir / "mms.npz", dz=dzs, rel_err=errs)

    p_obs = jnp.log(errs[:-1] / errs[1:]) / jnp.log(2.0)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
    ax.loglog(dzs, errs, "o-", lw=2, label="FCI target-aware derivative")
    ax.loglog(dzs, errs[0] * (dzs / dzs[0]) ** 2, "--", lw=1.5, label="O(dz^2) guide")
    ax.set_xlabel(r"$\Delta z$")
    ax.set_ylabel("relative L2 error")
    ax.set_title("FCI target-aware MMS convergence (Dirichlet plates)")
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

    print(f"[fci-target-mms] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
