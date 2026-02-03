from __future__ import annotations

"""
04_open_slab_sheath_ky_scan.py

Purpose
-------
Demonstrate the effect of a simple SOL/sheath closure on linear stability scans.

This script uses an *open* field-line geometry (non-periodic ∇_||) and optionally enables a
volumetric sheath-loss rate:

  nu_sh ~ (2 / L_parallel) * sheath_nu_factor

which damps (n, Te, omega, v_||e, v_||i) and is intended as a lightweight proxy for end-plate
losses at Bohm sheaths in reduced SOL models.

Run
---
  python examples/1_simple/04_open_slab_sheath_ky_scan.py

Outputs
-------
Written to `out/1_simple/open_slab_sheath_ky_scan/`.
"""

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, save_scan_panels, set_mpl_style  # noqa: E402
from jaxdrb.analysis.scan import scan_ky  # noqa: E402
from jaxdrb.geometry.slab import OpenSlabGeometry  # noqa: E402
from jaxdrb.models.params import DRBParams  # noqa: E402


def main() -> None:
    out_dir = Path("out/1_simple/open_slab_sheath_ky_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    fast = os.environ.get("JAXDRB_FAST", "1") != "0"
    if fast:
        print("JAXDRB_FAST=1: using a coarse grid for a quick run.", flush=True)
    else:
        print("JAXDRB_FAST=0: using a finer grid (slower).", flush=True)

    nl = 49 if fast else 97
    length = 6.0
    geom = OpenSlabGeometry.make(nl=nl, length=length, shat=0.0, curvature0=0.0)

    kx = 0.0
    ky = np.linspace(0.05, 1.2, 18 if fast else 36)

    base = dict(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=False,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        kperp2_min=1e-6,
    )

    params_no = DRBParams(**base)
    params_loss = DRBParams(**base, sheath_loss_on=True, sheath_loss_nu_factor=1.0)
    params_bc = DRBParams(**base, sheath_bc_on=True, sheath_bc_nu_factor=1.0)

    print("Running ky scan (open slab, no sheath)…", flush=True)
    scan0 = scan_ky(
        params_no,
        geom,
        ky=ky,
        kx=kx,
        arnoldi_m=25 if fast else 45,
        arnoldi_tol=1e-3,
        nev=6,
        do_initial_value=True,
        tmax=12.0 if fast else 25.0,
        dt0=0.02,
        nsave=110 if fast else 180,
        verbose=True,
        print_every=1,
        seed=0,
    )

    print("Running ky scan (open slab, with volumetric sheath-loss proxy)…", flush=True)
    scan1 = scan_ky(
        params_loss,
        geom,
        ky=ky,
        kx=kx,
        arnoldi_m=25 if fast else 45,
        arnoldi_tol=1e-3,
        nev=6,
        do_initial_value=True,
        tmax=12.0 if fast else 25.0,
        dt0=0.02,
        nsave=110 if fast else 180,
        verbose=True,
        print_every=1,
        seed=1,
    )

    print("Running ky scan (open slab, with Loizu-style MPSE sheath BCs)…", flush=True)
    scan2 = scan_ky(
        params_bc,
        geom,
        ky=ky,
        kx=kx,
        arnoldi_m=25 if fast else 45,
        arnoldi_tol=1e-3,
        nev=6,
        do_initial_value=True,
        tmax=12.0 if fast else 25.0,
        dt0=0.02,
        nsave=110 if fast else 180,
        verbose=True,
        print_every=1,
        seed=2,
    )

    # Panels + overlays.
    save_scan_panels(
        out_dir,
        ky=scan0.ky,
        gamma=scan0.gamma_eigs,
        omega=scan0.omega_eigs,
        gamma_iv=scan0.gamma_iv,
        title="Open slab: ky scan (no sheath)",
        filename="scan_panel_no_sheath.png",
    )
    save_scan_panels(
        out_dir,
        ky=scan1.ky,
        gamma=scan1.gamma_eigs,
        omega=scan1.omega_eigs,
        gamma_iv=scan1.gamma_iv,
        title="Open slab: ky scan (volumetric sheath-loss proxy)",
        filename="scan_panel_sheath_loss.png",
    )
    save_scan_panels(
        out_dir,
        ky=scan2.ky,
        gamma=scan2.gamma_eigs,
        omega=scan2.omega_eigs,
        gamma_iv=scan2.gamma_iv,
        title="Open slab: ky scan (MPSE Bohm sheath BCs, Loizu-style)",
        filename="scan_panel_sheath_bc.png",
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(scan0.ky, scan0.gamma_eigs, "o-", label="no sheath")
    ax.plot(scan1.ky, scan1.gamma_eigs, "s--", label="sheath loss (volumetric)")
    ax.plot(scan2.ky, scan2.gamma_eigs, "^-.", label="sheath BC (MPSE)")
    ax.axhline(0.0, color="k", alpha=0.25, lw=1)
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title(r"Open slab: effect of sheath closures on $\gamma(k_y)$")
    ax.legend()
    fig.savefig(out_dir / "gamma_overlay.png", dpi=220)
    plt.close(fig)

    save_json(
        out_dir / "params.json",
        {
            "geom": {
                "type": "open_slab",
                "nl": nl,
                "length": float(length),
                "shat": 0.0,
                "curvature0": 0.0,
            },
            "kx": kx,
            "ky": {"min": float(ky.min()), "max": float(ky.max()), "n": int(ky.size)},
            "params_no": params_no.__dict__,
            "params_sheath_loss": params_loss.__dict__,
            "params_sheath_bc": params_bc.__dict__,
        },
    )
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
