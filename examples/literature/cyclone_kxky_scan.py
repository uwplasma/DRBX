from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.scan import scan_kx_ky
from jaxdrb.geometry.tokamak import SAlphaGeometry
from jaxdrb.models.params import DRBParams


def main() -> None:
    """2D (kx, ky) scan on an s-alpha geometry (Cyclone-like parameters)."""

    out_dir = Path("out_cyclone_kxky_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))

    geom = SAlphaGeometry.cyclone_base_case(nl=64)

    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=5e-3,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )

    ky = np.linspace(0.1, 1.0, 16)
    kx = np.linspace(-1.0, 1.0, 25)

    res = scan_kx_ky(params, geom, kx=kx, ky=ky, arnoldi_m=20, nev=3, seed=0)

    np.savez(
        out_dir / "results_2d.npz", kx=res.kx, ky=res.ky, gamma=res.gamma_eigs, omega=res.omega_eigs
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    im = ax.pcolormesh(res.ky, res.kx, res.gamma_eigs, shading="auto", cmap="magma")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$k_x$")
    ax.set_title(r"Cyclone-like $s$-$\alpha$: leading $\gamma(k_x,k_y)$")
    fig.colorbar(im, ax=ax, label=r"$\gamma$")
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_kxky.png", dpi=160)
    plt.close(fig)

    gmax = np.max(res.gamma_eigs, axis=0)
    kx_star = res.kx[np.argmax(res.gamma_eigs, axis=0)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res.ky, gmax, "o-")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\max_{k_x}\,\gamma$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_ky_max_over_kx.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res.ky, kx_star, "o-")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$k_x^*(k_y)$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "kx_star_vs_ky.png", dpi=160)
    plt.close(fig)

    print(f"Wrote {out_dir / 'gamma_kxky.png'}")
    print(f"Wrote {out_dir / 'gamma_ky_max_over_kx.png'}")
    print(f"Wrote {out_dir / 'kx_star_vs_ky.png'}")


if __name__ == "__main__":
    main()
