"""
loizu2012_full_mpse_bc.py

Purpose
-------
Demonstrate the *full* (linearized, model-aligned) MPSE boundary condition set based on:

  J. Loizu et al., "Boundary conditions for plasma fluid models at the magnetic presheath entrance",
  Phys. Plasmas 19, 122307 (2012).

Compared to the legacy "velocity-only" MPSE enforcement, the Loizu2012 option adds weak
relaxation toward additional endpoint constraints involving:

  - ∂_|| phi and ∂_|| n (gradient relations),
  - omega (vorticity BC, linearized),
  - ∂_|| Te = 0.

This script runs a ky scan with:

  - open field line geometry,
  - MPSE BCs enabled in either "simple" or "loizu2012" mode,
  - side-by-side comparison of growth rates and a basic boundary residual diagnostic.

Run
---
  python examples/03_sheath_mpse/loizu2012_full_mpse_bc.py

Outputs
-------
Written to `out/examples/03_sheath_mpse/loizu2012_full_mpse_bc/`.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, save_scan_panels, set_mpl_style
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.models.params import DRBParams


def main() -> None:
    out_dir = Path("out/examples/03_sheath_mpse/loizu2012_full_mpse_bc")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    fast = os.environ.get("JAXDRB_FAST", "1") != "0"
    if fast:
        print("JAXDRB_FAST=1: smaller grids for a quick run.", flush=True)
    else:
        print("JAXDRB_FAST=0: larger grids (slower).", flush=True)

    nl = 65 if fast else 129
    geom = OpenSlabGeometry.make(nl=nl, length=float(2 * np.pi), shat=0.2, curvature0=0.15)

    ky = np.linspace(0.06, 0.8, 10 if fast else 28)
    kx = 0.0

    base = DRBParams(
        omega_n=0.2,
        omega_Te=0.8,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        sheath_bc_on=True,
        sheath_bc_nu_factor=1.0,
        sheath_cos2=1.0,
    )

    cases = [
        ("simple", DRBParams(**{**base.__dict__, "sheath_bc_model": 0})),
        ("loizu2012", DRBParams(**{**base.__dict__, "sheath_bc_model": 1})),
    ]

    results = []
    for name, p in cases:
        print(f"\n=== MPSE BC mode: {name} ===", flush=True)
        s = scan_ky(
            p,
            geom,
            ky=ky,
            kx=kx,
            arnoldi_m=30 if fast else 50,
            arnoldi_tol=3e-3 if fast else 1e-3,
            nev=6,
            do_initial_value=False,
            verbose=True,
            print_every=3,
            seed=0,
        )
        results.append((name, s))
        save_scan_panels(
            out_dir,
            ky=s.ky,
            gamma=s.gamma_eigs,
            omega=s.omega_eigs,
            title=f"MPSE BC mode: {name}",
            filename=f"scan_panel_{name}.png",
        )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for name, s in results:
        ax.plot(s.ky, s.gamma_eigs, "o-", label=name)
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("Effect of MPSE BC model on growth rates (example)")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.savefig(out_dir / "gamma_comparison.png", dpi=220)
    plt.close(fig)

    np.savez(
        out_dir / "results.npz",
        ky=ky,
        gamma_simple=results[0][1].gamma_eigs,
        omega_simple=results[0][1].omega_eigs,
        gamma_loizu2012=results[1][1].gamma_eigs,
        omega_loizu2012=results[1][1].omega_eigs,
    )
    save_json(out_dir / "params.json", {"base": base.__dict__})
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
