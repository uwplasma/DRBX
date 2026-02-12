"""FCI hello-world: analytic map, parallel derivative, and line integral."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.fci.integrate import line_integral_mapped
from jaxdrb.fci.map import SlabFCIConfig, make_slab_fci_map
from jaxdrb.fci.parallel import parallel_derivative_centered_3d


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="out_fci_hello_world")
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=24)
    p.add_argument("--Lx", type=float, default=2 * jnp.pi)
    p.add_argument("--Ly", type=float, default=2 * jnp.pi)
    p.add_argument("--dz", type=float, default=0.25)
    p.add_argument("--Bx", type=float, default=0.4)
    p.add_argument("--By", type=float, default=0.2)
    p.add_argument("--Bz", type=float, default=1.0)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nx = int(args.nx)
    ny = int(args.ny)
    nz = int(args.nz)
    dx = float(args.Lx) / float(nx)
    dy = float(args.Ly) / float(ny)
    dz = float(args.dz)

    cfg = SlabFCIConfig(
        x0=0.0,
        y0=0.0,
        dx=dx,
        dy=dy,
        nx=nx,
        ny=ny,
        dz=dz,
        Bx=float(args.Bx),
        By=float(args.By),
        Bz=float(args.Bz),
    )
    map_fwd, map_bwd = make_slab_fci_map(cfg)

    xs = cfg.x0 + cfg.dx * jnp.arange(cfg.nx)
    ys = cfg.y0 + cfg.dy * jnp.arange(cfg.ny)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")

    # Analytic test field.
    kx = 2.0
    ky = 3.0
    kz = -1.0

    z = dz * jnp.arange(nz)
    phase0 = kx * X + ky * Y
    f_planes = jnp.sin(phase0[None, :, :] + kz * z[:, None, None])

    # Parallel derivative (centered) on the stack.
    dpar = parallel_derivative_centered_3d(
        f_planes, map_fwd=map_fwd, map_bwd=map_bwd, open_field_line=False
    )
    mid = nz // 2
    B = jnp.array([cfg.Bx, cfg.By, cfg.Bz])
    b = B / jnp.linalg.norm(B)
    phase_mid = phase0 + kz * z[mid]
    dpar_exact = (b[0] * kx + b[1] * ky + b[2] * kz) * jnp.cos(phase_mid)

    # Line integral along the field line in reference coordinates.
    dl0 = float(map_fwd.dl[0, 0])
    L = dl0 * float(nz - 1)
    alpha = b[0] * kx + b[1] * ky + b[2] * kz
    alpha_safe = jnp.where(jnp.abs(alpha) < 1e-8, 1.0, alpha)
    integral_exact = (jnp.cos(phase0) - jnp.cos(phase0 + alpha * L)) / alpha_safe
    integral_exact = jnp.where(jnp.abs(alpha) < 1e-8, L * jnp.sin(phase0), integral_exact)

    integral_num = line_integral_mapped(f_planes, map_fwd=map_fwd, dl=map_fwd.dl, periodic=False)

    dpar_err = dpar[mid] - dpar_exact
    integ_err = integral_num - integral_exact

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    im0 = axes[0].imshow(dpar_err.T, origin="lower", cmap="coolwarm")
    axes[0].set_title(r"$\partial_\parallel f$ error (mid plane)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(integ_err.T, origin="lower", cmap="coolwarm")
    axes[1].set_title("Line-integral error (mapped)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    fig.suptitle("FCI hello-world: analytic slab map + integration")
    fig.tight_layout()
    fig.savefig(out_dir / "fci_hello_world.png", dpi=220)
    plt.close(fig)

    rel_dpar = jnp.sqrt(jnp.mean(dpar_err**2)) / jnp.maximum(
        jnp.sqrt(jnp.mean(dpar_exact**2)), 1e-12
    )
    rel_int = jnp.sqrt(jnp.mean(integ_err**2)) / jnp.maximum(
        jnp.sqrt(jnp.mean(integral_exact**2)), 1e-12
    )
    print(f"[fci-hello] rel_dpar={float(rel_dpar):.3e} rel_integral={float(rel_int):.3e}")
    print(f"[fci-hello] wrote {out_dir / 'fci_hello_world.png'}")


if __name__ == "__main__":
    main()
