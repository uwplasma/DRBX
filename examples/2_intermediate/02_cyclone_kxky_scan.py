from __future__ import annotations

"""
02_cyclone_kxky_scan.py

Purpose
-------
Scan the leading eigenvalue on a 2D (kx, ky) grid for a standard analytic s-alpha geometry
(Cyclone Base Case-like parameters).

This script showcases:
- 2D spectral scans (kx, ky) common in flux-tube literature,
- identifying the most unstable point on the grid,
- plotting a heatmap plus 1D summaries (max over kx, kx*(ky)),
- plotting a leading eigenfunction and Ritz spectrum at the most unstable point.

Run
---
  python examples/2_intermediate/02_cyclone_kxky_scan.py

Outputs
-------
Written to `out/2_intermediate/cyclone_kxky_scan/`.
"""

import os
from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import (
    save_eigenfunction_panel,
    save_eigenvalue_spectrum,
    save_geometry_overview,
    save_json,
    save_kxky_heatmap,
    set_mpl_style,
)
from jaxdrb.analysis.scan import scan_kx_ky
from jaxdrb.geometry.tokamak import SAlphaGeometry
from jaxdrb.linear.arnoldi import arnoldi_eigs, arnoldi_leading_ritz_vector
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import State, equilibrium
from jaxdrb.models.params import DRBParams


def main() -> None:
    out_dir = Path("out/2_intermediate/cyclone_kxky_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    nl = 64
    geom = SAlphaGeometry.cyclone_base_case(nl=nl)

    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        kperp2_min=1e-6,
    )

    # Keep the grid modest so it runs quickly on laptops.
    ky = np.linspace(0.1, 1.0, 18)
    kx = np.linspace(-1.0, 1.0, 29)

    print("Running (kx, ky) scan (Cyclone-like s-alpha)…", flush=True)
    res = scan_kx_ky(
        params,
        geom,
        kx=kx,
        ky=ky,
        arnoldi_m=25,
        arnoldi_tol=2e-3,
        nev=4,
        seed=0,
        verbose=True,
        print_every_kx=1,
    )

    save_kxky_heatmap(
        out_dir,
        kx=res.kx,
        ky=res.ky,
        z=res.gamma_eigs,
        zlabel=r"$\gamma$",
        title=r"Cyclone-like $s$-$\alpha$: leading $\gamma(k_x,k_y)$",
        filename="gamma_kxky.png",
    )
    save_kxky_heatmap(
        out_dir,
        kx=res.kx,
        ky=res.ky,
        z=res.omega_eigs,
        zlabel=r"$\omega$",
        title=r"Cyclone-like $s$-$\alpha$: leading $\omega(k_x,k_y)$",
        filename="omega_kxky.png",
        cmap="viridis",
    )

    # Identify kx*(ky) and max_kx gamma.
    gamma_max = np.max(res.gamma_eigs, axis=0)
    idx_star = np.argmax(res.gamma_eigs, axis=0)
    kx_star = res.kx[idx_star]

    # Identify global max on the grid.
    ix, iy = np.unravel_index(np.argmax(res.gamma_eigs), res.gamma_eigs.shape)
    kx_best = float(res.kx[ix])
    ky_best = float(res.ky[iy])
    lam_best = complex(res.gamma_eigs[ix, iy] + 1j * res.omega_eigs[ix, iy])
    print(
        f"Most unstable grid point: kx={kx_best:+.3f} ky={ky_best:.3f}  "
        f"lambda={lam_best.real:+.4e}{lam_best.imag:+.4e}i",
        flush=True,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
    ax.plot(res.ky, gamma_max, "o-")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\max_{k_x}\,\gamma$")
    ax.set_title(r"Most unstable growth rate vs $k_y$")
    fig.savefig(out_dir / "gamma_ky_max_over_kx.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
    ax.plot(res.ky, kx_star, "o-")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$k_x^*(k_y)$")
    ax.set_title(r"$k_x^*$ maximizing $\gamma$ at each $k_y$")
    fig.savefig(out_dir / "kx_star_vs_ky.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
    ax.plot(res.ky, np.maximum(gamma_max, 0.0) / res.ky, "o-")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\max(\gamma,0)/k_y$")
    ax.set_title(r"Transport proxy at $k_x^*(k_y)$")
    fig.savefig(out_dir / "gamma_over_ky_max_over_kx.png", dpi=200)
    plt.close(fig)

    # Eigenfunction + spectrum at the global best point.
    y_eq = equilibrium(nl)
    v0 = State.random(jax.random.PRNGKey(0), nl, amplitude=1e-3)
    matvec = linear_matvec(y_eq, params, geom, kx=kx_best, ky=ky_best)

    save_geometry_overview(out_dir, geom=geom, kx=kx_best, ky=ky_best, filename="geometry.png")

    print("Computing leading Ritz vector (for eigenfunction plot)…", flush=True)
    ritz = arnoldi_leading_ritz_vector(matvec, v0, m=90, seed=0)
    save_eigenfunction_panel(
        out_dir,
        geom=geom,
        state=ritz.vector,
        eigenvalue=ritz.eigenvalue,
        kx=kx_best,
        ky=ky_best,
        kperp2_min=params.kperp2_min,
        filename="eigenfunctions.png",
    )
    arn = arnoldi_eigs(matvec, v0, m=90, nev=90, seed=0)
    save_eigenvalue_spectrum(out_dir, eigenvalues=arn.eigenvalues, highlight=ritz.eigenvalue)

    np.savez(
        out_dir / "results.npz", kx=res.kx, ky=res.ky, gamma=res.gamma_eigs, omega=res.omega_eigs
    )
    save_json(
        out_dir / "params.json",
        {
            "geom": {
                "type": "salpha_cyclone",
                "nl": nl,
                "q": float(geom.q),
                "shat": float(geom.shat),
                "alpha": float(geom.alpha),
                "epsilon": float(geom.epsilon),
                "curvature0": float(geom.curvature0),
            },
            "grid": {
                "kx": [float(kx.min()), float(kx.max()), int(kx.size)],
                "ky": [float(ky.min()), float(ky.max()), int(ky.size)],
            },
            "params": params.__dict__,
        },
    )
    save_json(
        out_dir / "summary.json",
        {
            "best_point": {
                "kx": kx_best,
                "ky": ky_best,
                "lambda": {"real": lam_best.real, "imag": lam_best.imag},
            },
        },
    )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
