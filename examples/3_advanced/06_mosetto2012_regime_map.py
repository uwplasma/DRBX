from __future__ import annotations

"""
06_mosetto2012_regime_map.py

Purpose
-------
Create a Mosetto-like "regime map" showing which linear branch dominates across a 2D scan in:

  - collisionality-like parameter: `eta`
  - background drive strength: `omega_n` (proxy for R/Ln or R/Lp in local models)

and classify the dominant instability into four labels:

  - InDW: inertial drift-wave-like
  - RDW: resistive drift-wave-like
  - InBM: inertial ballooning-like
  - RBM: resistive ballooning-like

Important caveat
----------------
This is a **workflow + visualization** replica rather than a quantitatively exact reproduction of
any specific Mosetto figure: our model is simplified, electrostatic by default, and we separate
branches by toggling parameters and curvature.

Classification strategy used here (transparent + hackable):
----------------------------------------------------------
For each (eta, omega_n) point, we compute the peak growth rate over ky for each branch candidate:

  - DW branches: curvature_off, (me_hat small / large) for resistive vs inertial
  - BM branches: curvature_on,  (me_hat small / large) for resistive vs inertial

We then assign the regime label corresponding to the maximum of the four peak growth rates.

Run
---
  python examples/3_advanced/06_mosetto2012_regime_map.py

Outputs
-------
Written to `out/3_advanced/mosetto2012_regime_map/`:

  - `regime_map.png`: colored map of dominant regime
  - `gamma_max_maps.png`: four panels with gamma_max(eta, omega_n) per branch
  - `results.npz`: numeric results (gamma maxima etc.)
"""

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, set_mpl_style
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.models.params import DRBParams


def _gamma_max_over_ky(scan) -> float:
    return float(np.max(scan.gamma_eigs))


def main() -> None:
    out_dir = Path("out/3_advanced/mosetto2012_regime_map")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    fast = os.environ.get("JAXDRB_FAST", "1") != "0"
    if fast:
        print("JAXDRB_FAST=1: using a coarse grid for a quick, pedagogic run.", flush=True)
    else:
        print("JAXDRB_FAST=0: using a finer grid (may take a long time).", flush=True)

    # Geometry: slab, with curvature amplitude that can drive BM-like modes when enabled.
    nl = 24 if fast else 64
    geom = SlabGeometry.make(nl=nl, length=float(2 * np.pi), shat=0.4, curvature0=0.2)
    kx = 0.0

    # ky scan used at each parameter point.
    ky = np.linspace(0.08, 1.0, 8 if fast else 28)

    # Parameter grids (log for eta, linear for omega_n).
    eta_grid = np.logspace(-2.0, 1.0, 5 if fast else 18)
    omega_n_grid = np.linspace(0.0, 2.0, 6 if fast else 21)

    # Branch definition knobs (qualitative):
    # - resistive-like: small electron inertia (me_hat small) and finite eta
    # - inertial-like: larger me_hat and weak eta
    me_resistive = 1e-3
    me_inertial = 0.08

    base = DRBParams(
        omega_n=0.8,  # overwritten
        omega_Te=0.0,
        eta=1.0,  # overwritten
        me_hat=0.05,  # overwritten per-branch
        curvature_on=False,  # overwritten per-branch
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )

    # Store gamma_max per branch: [eta, omega_n]
    gamma_in_dw = np.zeros((eta_grid.size, omega_n_grid.size))
    gamma_r_dw = np.zeros_like(gamma_in_dw)
    gamma_in_bm = np.zeros_like(gamma_in_dw)
    gamma_r_bm = np.zeros_like(gamma_in_dw)

    npoints = eta_grid.size * omega_n_grid.size
    print(f"Scanning (eta, omega_n) grid… ({npoints} points, 4 branches per point)", flush=True)
    for i, eta in enumerate(eta_grid):
        for j, omega_n in enumerate(omega_n_grid):
            # Resistive DW-like: curvature off
            p = DRBParams(**{**base.__dict__, "eta": float(eta), "omega_n": float(omega_n)})

            s_rdw = scan_ky(
                DRBParams(**{**p.__dict__, "me_hat": float(me_resistive), "curvature_on": False}),
                geom,
                ky=ky,
                kx=kx,
                arnoldi_m=16 if fast else 32,
                arnoldi_tol=6e-3 if fast else 2e-3,
                nev=2 if fast else 6,
                do_initial_value=False,
                verbose=False,
                seed=0,
            )
            gamma_r_dw[i, j] = _gamma_max_over_ky(s_rdw)

            s_idw = scan_ky(
                DRBParams(**{**p.__dict__, "me_hat": float(me_inertial), "curvature_on": False}),
                geom,
                ky=ky,
                kx=kx,
                arnoldi_m=16 if fast else 32,
                arnoldi_tol=6e-3 if fast else 2e-3,
                nev=2 if fast else 6,
                do_initial_value=False,
                verbose=False,
                seed=0,
            )
            gamma_in_dw[i, j] = _gamma_max_over_ky(s_idw)

            # Ballooning-like branches: curvature on
            s_rbm = scan_ky(
                DRBParams(**{**p.__dict__, "me_hat": float(me_resistive), "curvature_on": True}),
                geom,
                ky=ky,
                kx=kx,
                arnoldi_m=16 if fast else 32,
                arnoldi_tol=6e-3 if fast else 2e-3,
                nev=2 if fast else 6,
                do_initial_value=False,
                verbose=False,
                seed=0,
            )
            gamma_r_bm[i, j] = _gamma_max_over_ky(s_rbm)

            s_ibm = scan_ky(
                DRBParams(**{**p.__dict__, "me_hat": float(me_inertial), "curvature_on": True}),
                geom,
                ky=ky,
                kx=kx,
                arnoldi_m=16 if fast else 32,
                arnoldi_tol=6e-3 if fast else 2e-3,
                nev=2 if fast else 6,
                do_initial_value=False,
                verbose=False,
                seed=0,
            )
            gamma_in_bm[i, j] = _gamma_max_over_ky(s_ibm)

        print(f"[{i+1:02d}/{eta_grid.size}] eta={eta:8.2e} done", flush=True)

    # Classification: pick max of the four gamma_max values.
    stack = np.stack([gamma_in_dw, gamma_r_dw, gamma_in_bm, gamma_r_bm], axis=0)
    labels = ["InDW", "RDW", "InBM", "RBM"]
    idx = np.argmax(stack, axis=0)  # [eta, omega_n]

    # Make a nice categorical colormap.
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    im = ax.pcolormesh(
        omega_n_grid,
        np.log10(eta_grid),
        idx,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xlabel(r"drive $\omega_n$  (proxy for $R/L_n$)")
    ax.set_ylabel(r"$\log_{10}(\eta)$  (collisionality-like)")
    ax.set_title("Dominant regime label (workflow replica)")

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(labels)
    fig.savefig(out_dir / "regime_map.png", dpi=240)
    plt.close(fig)

    # Also show gamma_max maps for each branch (helps debug the classification).
    fig, axs = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)
    panels = [
        ("InDW: max gamma over ky", gamma_in_dw),
        ("RDW: max gamma over ky", gamma_r_dw),
        ("InBM: max gamma over ky", gamma_in_bm),
        ("RBM: max gamma over ky", gamma_r_bm),
    ]
    for ax, (title, arr) in zip(axs.reshape(-1), panels, strict=True):
        im = ax.pcolormesh(omega_n_grid, np.log10(eta_grid), arr, shading="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel(r"$\omega_n$")
        ax.set_ylabel(r"$\log_{10}(\eta)$")
        fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\gamma_{\max}$")
    fig.savefig(out_dir / "gamma_max_maps.png", dpi=240)
    plt.close(fig)

    np.savez(
        out_dir / "results.npz",
        eta=eta_grid,
        omega_n=omega_n_grid,
        gamma_in_dw=gamma_in_dw,
        gamma_r_dw=gamma_r_dw,
        gamma_in_bm=gamma_in_bm,
        gamma_r_bm=gamma_r_bm,
        regime_index=idx,
    )
    save_json(
        out_dir / "params.json",
        {
            "geometry": {"type": "slab", "nl": nl, "shat": 0.4, "curvature0": 0.2},
            "kx": kx,
            "ky": {"min": float(ky.min()), "max": float(ky.max()), "n": int(ky.size)},
            "eta": {
                "min": float(eta_grid.min()),
                "max": float(eta_grid.max()),
                "n": int(eta_grid.size),
            },
            "omega_n": {
                "min": float(omega_n_grid.min()),
                "max": float(omega_n_grid.max()),
                "n": int(omega_n_grid.size),
            },
            "branch_definition": {
                "me_hat_resistive": me_resistive,
                "me_hat_inertial": me_inertial,
                "labels": labels,
            },
            "notes": [
                "This is a qualitative workflow replica; branch labeling is by parameter toggles.",
                "Interpret omega_n as a proxy for R/Ln (or R/Lp) only in a reduced/local sense.",
            ],
        },
    )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
