from __future__ import annotations

"""
06_mosetto2012_regime_map.py

Purpose
-------
Create a Mosetto-like instability *regime map* following the conceptual picture in:

  A. Mosetto, F. D. Halpern, S. Jolliet, and P. Ricci,
  "Low-frequency linear-mode regimes in the tokamak scrape-off layer",
  Phys. Plasmas 19, 112103 (2012).

In the electrostatic limit, Mosetto et al. categorize the dominant linear instability into
four regimes:

  - InDW: inertial drift wave
  - RDW:  resistive drift wave
  - InBM: inertial ballooning mode
  - RBM:  resistive ballooning mode

This script produces a **workflow-aligned** version of that map using the drift-reduced model
already implemented in `jaxdrb`:

  - Horizontal axis: a collisionality-like knob (here `eta`, the normalized parallel resistivity).
  - Vertical axis: a background drive knob (here `omega_n`, a proxy for R/L_n in local models).

Classification (ablation-based)
-------------------------------
Rather than toggling branches by hand, we classify the dominant mode *at each parameter point*
via two simple ablations:

1) Drift-wave-like vs ballooning-like:
   Compare the peak growth rate with curvature **on** vs **off**.
2) Inertial vs resistive:
   Compare the peak growth rate with electron inertia **on** vs **off** (`me_hat=0`).

This yields a 2×2 decision table → {InDW, RDW, InBM, RBM}.

Important caveat
----------------
Mosetto Fig. 6(a) is a schematic overview of regimes in SOL parameter space. Reproducing the
*quantitative* boundaries requires matching the full model and boundary conditions used in the
paper. This example aims to:

  - match the **qualitative** region structure,
  - be transparent about the classification rule,
  - be fast enough to run routinely.

Run
---
  python examples/3_advanced/06_mosetto2012_regime_map.py

Environment knobs
-----------------
Set `JAXDRB_FAST=0` for a finer grid (slower).

Outputs
-------
Written to `out/3_advanced/mosetto2012_regime_map/`:

  - `mosetto2012_fig6a_like.png`: dominant-regime map + helper panels
  - `results.npz`: numeric arrays
"""

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, set_mpl_style
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.tokamak import CircularTokamakGeometry
from jaxdrb.models.params import DRBParams


def _gamma_max(scan) -> float:
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

    # Geometry: large-aspect-ratio circular tokamak with shear and curvature.
    nl = 24 if fast else 96
    geom = CircularTokamakGeometry.make(
        nl=nl, shat=0.8, q=3.0, R0=1.0, epsilon=0.18, curvature0=0.18
    )

    kx = 0.0
    ky = np.linspace(0.08, 0.9, 6 if fast else 30)

    # Parameter scan ranges chosen to highlight the qualitative 4-regime structure.
    eta_grid = np.logspace(-2.3, 0.4, 6 if fast else 26)  # "collisionality-like"
    omega_n_grid = np.linspace(0.0, 2.2, 7 if fast else 31)  # proxy for R/L_n

    # Baseline parameters (electrostatic, cold ions).
    base = DRBParams(
        omega_n=1.0,  # overwritten
        omega_Te=0.0,
        eta=0.1,  # overwritten
        me_hat=0.05,  # inertia on in "full" runs
        curvature_on=True,
        beta=0.0,
        tau_i=0.0,
        boussinesq=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        kperp2_min=1e-6,
    )

    # Classification thresholds (heuristic, but robust):
    # - DW if curvature-off retains at least this fraction of the growth rate.
    # - Resistive if inertia-off retains at least this fraction of the growth rate.
    frac_threshold = 0.5
    tiny = 1e-12

    gamma_full = np.zeros((eta_grid.size, omega_n_grid.size))
    gamma_no_curv = np.zeros_like(gamma_full)
    gamma_no_inertia = np.zeros_like(gamma_full)
    label_idx = np.full_like(gamma_full, fill_value=-1, dtype=int)

    labels = ["InDW", "RDW", "InBM", "RBM"]

    npoints = eta_grid.size * omega_n_grid.size
    print(f"Scanning (eta, omega_n) grid… ({npoints} points; 3 scans per point)", flush=True)

    for i, eta in enumerate(eta_grid):
        for j, omega_n in enumerate(omega_n_grid):
            p = DRBParams(**{**base.__dict__, "eta": float(eta), "omega_n": float(omega_n)})

            arn_m = 14 if fast else 40
            arn_tol = 7e-3 if fast else 2e-3
            arn_nev = 2 if fast else 6

            s_full = scan_ky(
                p,
                geom,
                ky=ky,
                kx=kx,
                arnoldi_m=arn_m,
                arnoldi_tol=arn_tol,
                arnoldi_max_m=5 * nl,
                nev=arn_nev,
                do_initial_value=False,
                verbose=False,
                seed=0,
            )
            g_full = _gamma_max(s_full)

            s_noc = scan_ky(
                DRBParams(**{**p.__dict__, "curvature_on": False}),
                geom,
                ky=ky,
                kx=kx,
                arnoldi_m=arn_m,
                arnoldi_tol=arn_tol,
                arnoldi_max_m=5 * nl,
                nev=arn_nev,
                do_initial_value=False,
                verbose=False,
                seed=0,
            )
            g_noc = _gamma_max(s_noc)

            s_noi = scan_ky(
                DRBParams(**{**p.__dict__, "me_hat": 0.0}),
                geom,
                ky=ky,
                kx=kx,
                arnoldi_m=arn_m,
                arnoldi_tol=arn_tol,
                arnoldi_max_m=5 * nl,
                nev=arn_nev,
                do_initial_value=False,
                verbose=False,
                seed=0,
            )
            g_noi = _gamma_max(s_noi)

            gamma_full[i, j] = g_full
            gamma_no_curv[i, j] = g_noc
            gamma_no_inertia[i, j] = g_noi

            if g_full <= tiny:
                label_idx[i, j] = -1
                continue

            dw_like = g_noc >= frac_threshold * g_full
            resistive_like = g_noi >= frac_threshold * g_full

            if dw_like and (not resistive_like):
                label_idx[i, j] = 0  # InDW
            elif dw_like and resistive_like:
                label_idx[i, j] = 1  # RDW
            elif (not dw_like) and (not resistive_like):
                label_idx[i, j] = 2  # InBM
            else:
                label_idx[i, j] = 3  # RBM

        print(f"[{i+1:02d}/{eta_grid.size}] eta={eta:9.3e} done", flush=True)

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    fig = plt.figure(figsize=(12.0, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    # --- Dominant label map ---
    ax = fig.add_subplot(gs[0, :])
    cmap = ListedColormap(["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap.N)

    im = ax.pcolormesh(
        np.log10(eta_grid),
        omega_n_grid,
        label_idx.T,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xlabel(r"$\log_{10}(\eta)$  (collisionality-like)")
    ax.set_ylabel(r"drive $\omega_n$  (proxy for $R/L_n$)")
    ax.set_title("Dominant instability label (Mosetto-like regime map; ablation-classified)")

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], pad=0.01)
    cbar.ax.set_yticklabels(labels)

    # --- Helper panels: ablation ratios ---
    r_curv = np.where(gamma_full > tiny, gamma_no_curv / gamma_full, np.nan)
    r_inertia = np.where(gamma_full > tiny, gamma_no_inertia / gamma_full, np.nan)

    ax1 = fig.add_subplot(gs[1, 0])
    im1 = ax1.pcolormesh(np.log10(eta_grid), omega_n_grid, r_curv.T, shading="auto", cmap="viridis")
    ax1.axhline(0.0, color="k", lw=0.5, alpha=0.3)
    ax1.set_xlabel(r"$\log_{10}(\eta)$")
    ax1.set_ylabel(r"$\omega_n$")
    ax1.set_title(r"Curvature-off ratio: $\gamma_{\mathrm{no\,curv}}/\gamma_{\mathrm{full}}$")
    fig.colorbar(im1, ax=ax1, shrink=0.88)

    ax2 = fig.add_subplot(gs[1, 1])
    im2 = ax2.pcolormesh(
        np.log10(eta_grid), omega_n_grid, r_inertia.T, shading="auto", cmap="viridis"
    )
    ax2.axhline(0.0, color="k", lw=0.5, alpha=0.3)
    ax2.set_xlabel(r"$\log_{10}(\eta)$")
    ax2.set_ylabel(r"$\omega_n$")
    ax2.set_title(r"Inertia-off ratio: $\gamma_{m_e=0}/\gamma_{\mathrm{full}}$")
    fig.colorbar(im2, ax=ax2, shrink=0.88)

    fig.savefig(out_dir / "mosetto2012_fig6a_like.png", dpi=240)
    plt.close(fig)

    np.savez(
        out_dir / "results.npz",
        eta=eta_grid,
        omega_n=omega_n_grid,
        ky=ky,
        gamma_full=gamma_full,
        gamma_no_curv=gamma_no_curv,
        gamma_no_inertia=gamma_no_inertia,
        label_index=label_idx,
    )
    save_json(
        out_dir / "params.json",
        {
            "paper": "Mosetto et al. (2012) Phys. Plasmas 19, 112103",
            "geometry": {
                "type": "circular_tokamak",
                "nl": nl,
                "shat": 0.8,
                "q": 3.0,
                "epsilon": 0.18,
            },
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
            "classification": {
                "method": "ablation ratios vs full model",
                "frac_threshold": frac_threshold,
                "labels": labels,
                "rules": {
                    "DW": "gamma_no_curv >= frac_threshold * gamma_full",
                    "resistive": "gamma_me_hat0 >= frac_threshold * gamma_full",
                },
            },
            "base_params": base.__dict__,
        },
    )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
