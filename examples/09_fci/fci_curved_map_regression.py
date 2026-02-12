"""FCI curved-map convergence: spatially varying Bx(y) field-line shift."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.fci.map import SlabFCIConfig, make_slab_fci_map_variable_B


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="out_fci_curved_map")
    p.add_argument("--nx0", type=int, default=48)
    p.add_argument("--ny0", type=int, default=48)
    p.add_argument("--dz0", type=float, default=0.3)
    p.add_argument("--nref", type=int, default=4)
    p.add_argument("--shear", type=float, default=0.15)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    Lx = 2 * jnp.pi
    Ly = 2 * jnp.pi

    dz = float(args.dz0)
    errs = []
    dzs = []

    for r in range(int(args.nref)):
        nx = int(args.nx0) * (2**r)
        ny = int(args.ny0) * (2**r)
        dx = float(Lx / nx)
        dy = float(Ly / ny)

        cfg = SlabFCIConfig(
            x0=0.0,
            y0=0.0,
            dx=dx,
            dy=dy,
            nx=nx,
            ny=ny,
            dz=dz,
            Bx=0.0,
            By=0.0,
            Bz=1.0,
        )

        xs = cfg.x0 + cfg.dx * jnp.arange(cfg.nx)
        ys = cfg.y0 + cfg.dy * jnp.arange(cfg.ny)
        X, Y = jnp.meshgrid(xs, ys, indexing="ij")

        Bx = float(args.shear) * (Y - 0.5 * Ly)
        By = jnp.zeros_like(Bx)
        fwd, _ = make_slab_fci_map_variable_B(cfg, Bx=Bx, By=By, Bz=1.0)

        f = jnp.sin(2.0 * X) + 0.3 * jnp.cos(3.0 * Y) + 0.2 * jnp.sin(X + 2.0 * Y)
        shift_x = Bx * dz
        shift_y = By * dz
        Xs = jnp.mod(X + shift_x, Lx)
        Ys = jnp.mod(Y + shift_y, Ly)
        f_ref = jnp.sin(2.0 * Xs) + 0.3 * jnp.cos(3.0 * Ys) + 0.2 * jnp.sin(Xs + 2.0 * Ys)
        f_bilin = fwd.apply(f)

        err = jnp.sqrt(jnp.mean((f_bilin - f_ref) ** 2))
        errs.append(float(err))
        dzs.append(float(dz))
        print(f"[fci-curved] level={r} nx={nx} dz={dz:.4f} rel_L2={float(err):.3e}")
        dz = dz / 2.0

    dzs = jnp.array(dzs)
    errs = jnp.array(errs)
    jnp.savez(out_dir / "curved_map.npz", dz=dzs, rel_err=errs)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
    ax.loglog(dzs, errs, "o-", lw=2, label="curved-map error")
    ax.loglog(dzs, errs[0] * (dzs / dzs[0]) ** 2, "--", lw=1.5, label="O(dz^2) guide")
    ax.set_xlabel(r"$\Delta z$")
    ax.set_ylabel("relative L2 error")
    ax.set_title("FCI curved-map convergence (Bx varies with y)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "curved_map_convergence.png", dpi=220)
    plt.close(fig)

    print(f"[fci-curved] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
