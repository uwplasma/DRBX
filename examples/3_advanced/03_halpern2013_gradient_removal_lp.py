from __future__ import annotations

"""
03_halpern2013_gradient_removal_lp.py

Purpose
-------
Demonstrate the "gradient removal" SOL width estimate used in:

  F. Halpern et al., "Ideal ballooning modes in the tokamak scrape-off layer",
  Phys. Plasmas 20, 052306 (2013).

The key transport proxy is:

  Gamma_perp ∝ (gamma/ky)_max

and a self-consistent SOL scale length Lp is obtained by solving (see Halpern 2013, Eq. (20)):

  Lp = q * (gamma/ky)_max(Lp)

In `jaxdrb` we demonstrate the *algorithmic workflow* using a simplified electrostatic model.

Run
---
  python examples/3_advanced/03_halpern2013_gradient_removal_lp.py

Outputs
-------
Written to `out/3_advanced/halpern2013_gradient_removal/`.
"""

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.lp import solve_lp_fixed_point
from jaxdrb.analysis.plotting import (
    save_json,
    save_lp_history,
    save_scan_panels,
    set_mpl_style,
)
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.tokamak import CircularTokamakGeometry
from jaxdrb.models.params import DRBParams


def main() -> None:
    out_dir = Path("out/3_advanced/halpern2013_gradient_removal")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    nl = 64
    q = 3.0
    geom = CircularTokamakGeometry.make(nl=nl, shat=0.8, q=q, R0=1.0, epsilon=0.18, curvature0=0.18)

    ky = np.linspace(0.05, 0.8, 18)
    kx = 0.0

    base = DRBParams(
        omega_n=1.0,
        omega_Te=0.0,
        eta=1.0,
        me_hat=5e-3,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        kperp2_min=1e-6,
    )

    # Two effective gradient strengths inspired by Halpern fig. 8 (mapped onto omega_n here).
    omega_n_cases = [7.15, 12.5]
    scans = []
    for omega_n in omega_n_cases:
        print(f"Computing gamma/ky curve for omega_n={omega_n:g}…", flush=True)
        p = DRBParams(**{**base.__dict__, "omega_n": float(omega_n)})
        scans.append(
            scan_ky(
                p,
                geom,
                ky=ky,
                kx=kx,
                arnoldi_m=30,
                arnoldi_tol=2e-3,
                nev=6,
                do_initial_value=False,
                verbose=True,
                print_every=3,
                seed=0,
            )
        )

    save_scan_panels(
        out_dir,
        ky=ky,
        gamma=scans[-1].gamma_eigs,
        omega=scans[-1].omega_eigs,
        title=r"Halpern 2013 workflow: example $\gamma(k_y)$",
        filename="scan_panel_example.png",
    )

    # --- Fixed-point solve for Lp ---
    print("\nSolving fixed point for Lp (gradient removal)…", flush=True)
    lp = solve_lp_fixed_point(
        base,
        geom,
        q=q,
        ky=ky,
        Lp0=20.0,
        omega_n_scale=1.0,
        relax=0.6,
        max_iter=12,
        arnoldi_m=25,
        arnoldi_tol=2e-3,
        nev=4,
        seed=1,
        verbose=True,
    )
    save_lp_history(out_dir, history=lp.history, q=q, filename="lp_fixed_point_history.png")

    # --- Simple scaling study: vary "drive" and recompute Lp ---
    curvature_grid = np.linspace(0.10, 0.30, 5)
    Lp_out = np.zeros_like(curvature_grid)
    ky_star_out = np.zeros_like(curvature_grid)
    ratio_out = np.zeros_like(curvature_grid)

    print("\nComputing Lp vs curvature drive (toy scaling study)…", flush=True)
    for i, c0 in enumerate(curvature_grid):
        print(f"[{i+1}/{curvature_grid.size}] curvature0={c0:.3f}", flush=True)
        geom_i = CircularTokamakGeometry.make(
            nl=nl, shat=0.8, q=q, R0=1.0, epsilon=0.18, curvature0=float(c0)
        )
        res = solve_lp_fixed_point(
            base,
            geom_i,
            q=q,
            ky=ky,
            Lp0=lp.Lp,
            omega_n_scale=1.0,
            relax=0.7,
            max_iter=10,
            arnoldi_m=22,
            arnoldi_tol=2e-3,
            nev=4,
            seed=1,
            verbose=False,
        )
        Lp_out[i] = res.Lp
        ky_star_out[i] = res.ky_star
        ratio_out[i] = res.gamma_over_ky_star

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)

    ax = axs[0]
    ax.plot(curvature_grid, Lp_out, "o-")
    ax.set_xlabel(r"curvature drive (`curvature0`)")
    ax.set_ylabel(r"$L_p$")
    ax.set_title(r"Fixed-point $L_p$")

    ax = axs[1]
    ax.plot(curvature_grid, ky_star_out, "o-")
    ax.set_xlabel(r"curvature drive (`curvature0`)")
    ax.set_ylabel(r"$k_{y,*}$")
    ax.set_title(r"$k_{y,*}$ maximizing $\gamma/k_y$")

    fig.savefig(out_dir / "lp_scaling_curvature0.png", dpi=220)
    plt.close(fig)

    # Also plot the two gamma/ky curves on one figure.
    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    for omega_n, s in zip(omega_n_cases, scans, strict=True):
        ax.plot(ky, np.maximum(s.gamma_eigs, 0.0) / ky, "o-", label=rf"$\omega_n={omega_n:g}$")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\max(\gamma,0)/k_y$")
    ax.set_title(r"Transport proxy curves (Halpern 2013 workflow)")
    ax.legend()
    fig.savefig(out_dir / "gamma_over_ky_curves.png", dpi=220)
    plt.close(fig)

    np.savez(
        out_dir / "results.npz",
        ky=ky,
        omega_n=np.asarray(omega_n_cases),
        gamma=np.stack([s.gamma_eigs for s in scans], axis=0),
        omega=np.stack([s.omega_eigs for s in scans], axis=0),
        curvature0=curvature_grid,
        Lp=Lp_out,
        ky_star=ky_star_out,
        gamma_over_ky_star=ratio_out,
        lp_fixed_point_history=lp.history,
    )
    save_json(
        out_dir / "params.json",
        {
            "q": q,
            "base": base.__dict__,
            "omega_n_cases": omega_n_cases,
            "curvature_grid": curvature_grid.tolist(),
        },
    )
    save_json(
        out_dir / "summary.json",
        {
            "lp_fixed_point": {
                "Lp": lp.Lp,
                "ky_star": lp.ky_star,
                "gamma_over_ky_star": lp.gamma_over_ky_star,
            }
        },
    )

    print(f"\nWrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
