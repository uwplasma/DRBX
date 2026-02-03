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

    nl = 65
    length = 6.0
    geom = OpenSlabGeometry.make(nl=nl, length=length, shat=0.0, curvature0=0.0)

    kx = 0.0
    ky = np.linspace(0.05, 1.2, 28)

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

    params_no_sheath = DRBParams(**base, sheath_on=False)
    params_sheath = DRBParams(**base, sheath_on=True, sheath_nu_factor=1.0)

    print("Running ky scan (open slab, no sheath)…", flush=True)
    scan0 = scan_ky(
        params_no_sheath,
        geom,
        ky=ky,
        kx=kx,
        arnoldi_m=35,
        arnoldi_tol=1e-3,
        nev=6,
        do_initial_value=True,
        tmax=20.0,
        dt0=0.02,
        nsave=140,
        verbose=True,
        print_every=1,
        seed=0,
    )

    print("Running ky scan (open slab, with sheath losses)…", flush=True)
    scan1 = scan_ky(
        params_sheath,
        geom,
        ky=ky,
        kx=kx,
        arnoldi_m=35,
        arnoldi_tol=1e-3,
        nev=6,
        do_initial_value=True,
        tmax=20.0,
        dt0=0.02,
        nsave=140,
        verbose=True,
        print_every=1,
        seed=1,
    )

    # Two separate panels + a quick overlay figure.
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
        title="Open slab: ky scan (with sheath losses)",
        filename="scan_panel_sheath.png",
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(scan0.ky, scan0.gamma_eigs, "o-", label="no sheath")
    ax.plot(scan1.ky, scan1.gamma_eigs, "s--", label="sheath losses")
    ax.axhline(0.0, color="k", alpha=0.25, lw=1)
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title(r"Open slab: effect of sheath-loss closure on $\gamma(k_y)$")
    ax.legend()
    fig.savefig(out_dir / "gamma_overlay.png", dpi=220)
    plt.close(fig)

    save_json(
        out_dir / "params.json",
        {
            "geom": {"type": "open_slab", "nl": nl, "length": float(length), "shat": 0.0, "curvature0": 0.0},
            "kx": kx,
            "ky": {"min": float(ky.min()), "max": float(ky.max()), "n": int(ky.size)},
            "params_no_sheath": params_no_sheath.__dict__,
            "params_sheath": params_sheath.__dict__,
        },
    )
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()

