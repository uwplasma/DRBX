"""
parallel_closures_effects.py

Purpose
-------
Demonstrate how simple *parallel closures* and *volumetric sinks* affect linear growth rates.

This is not a full Braginskii transport model; rather, it provides robust knobs that are useful
in SOL-like settings and are directly motivated by common drift-reduced workflows:

  - parallel electron heat conduction:  χ_|| ∂_||^2 Te
  - parallel flow diffusion/viscosity:  ν_|| ∂_||^2 v_||
  - optional volumetric sinks:         -ν_sink * f

We scan `chi_par_Te` and show how the leading growth rate decreases as parallel conduction
increases (all else fixed).

Run
---
  python examples/scripts/04_closures_transport/parallel_closures_effects.py

Outputs
-------
Written to `out/examples/04_closures_transport/parallel_closures_effects/`.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, save_scan_panels, set_mpl_style
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.models.params import DRBParams


def main() -> None:
    out_dir = Path("out/examples/04_closures_transport/parallel_closures_effects")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    fast = os.environ.get("JAXDRB_FAST", "1") != "0"
    if fast:
        print("JAXDRB_FAST=1: smaller grids for a quick run.", flush=True)
    else:
        print("JAXDRB_FAST=0: larger grids (slower).", flush=True)

    # Open field line geometry (useful for SOL-like workflows).
    nl = 33 if fast else 65
    geom = OpenSlabGeometry.make(nl=nl, length=float(2 * np.pi), shat=0.4, curvature0=0.2)

    ky = np.linspace(0.10, 0.8, 8 if fast else 18)
    kx = 0.0

    base = DRBParams(
        # Use a Te-driven case so parallel heat conduction has a clear stabilizing effect.
        omega_n=0.2,
        omega_Te=1.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        # Keep sheath BCs off in this demo so the effect of closures is easier to interpret.
        sheath_bc_on=False,
        # No extra closures by default:
        chi_par_Te=0.0,
        nu_par_e=0.0,
        nu_par_i=0.0,
        nu_sink_n=0.0,
        nu_sink_Te=0.0,
        nu_sink_vpar=0.0,
    )

    chi_grid = np.array([0.0, 0.05, 0.1, 0.2] if fast else [0.0, 0.05, 0.1, 0.2, 0.4], dtype=float)
    gamma_max = np.zeros_like(chi_grid)
    omega_at_max = np.zeros_like(chi_grid)
    ky_at_max = np.zeros_like(chi_grid)

    scans = []
    for i, chi in enumerate(chi_grid):
        print(f"[{i + 1}/{chi_grid.size}] chi_par_Te={chi:g}", flush=True)
        p = DRBParams(**{**base.__dict__, "chi_par_Te": float(chi)})
        s = scan_ky(
            p,
            geom,
            ky=ky,
            kx=kx,
            arnoldi_m=24 if fast else 40,
            arnoldi_tol=5e-3 if fast else 2e-3,
            arnoldi_max_m=None if fast else 5 * nl,
            nev=3 if fast else 6,
            do_initial_value=False,
            verbose=True,
            print_every=2,
            seed=0,
        )
        scans.append(s)
        idx = int(np.argmax(s.gamma_eigs))
        gamma_max[i] = float(s.gamma_eigs[idx])
        omega_at_max[i] = float(s.omega_eigs[idx])
        ky_at_max[i] = float(s.ky[idx])

        if i == 0:
            save_scan_panels(
                out_dir,
                ky=s.ky,
                gamma=s.gamma_eigs,
                omega=s.omega_eigs,
                title=r"Reference scan ($\chi_{\parallel}=0$)",
                filename="scan_panel_reference.png",
            )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.plot(chi_grid, gamma_max, "o-")
    ax.set_xlabel(r"$\chi_{\parallel, T_e}$")
    ax.set_ylabel(r"$\gamma_{\max}$")
    ax.set_title("Parallel Te conduction reduces growth (example)")
    ax.grid(True, alpha=0.35)
    fig.savefig(out_dir / "gamma_max_vs_chi_par_Te.png", dpi=220)
    plt.close(fig)

    np.savez(
        out_dir / "results.npz",
        chi_par_Te=chi_grid,
        gamma_max=gamma_max,
        omega_at_max=omega_at_max,
        ky_at_max=ky_at_max,
        ky=ky,
        gamma=np.stack([s.gamma_eigs for s in scans], axis=0),
        omega=np.stack([s.omega_eigs for s in scans], axis=0),
    )
    save_json(
        out_dir / "params.json",
        {
            "geometry": {"type": "open_slab", "nl": nl, "shat": 0.4, "curvature0": 0.2},
            "base_params": base.__dict__,
            "kx": kx,
            "ky": {"min": float(ky.min()), "max": float(ky.max()), "n": int(ky.size)},
            "chi_par_Te_grid": chi_grid.tolist(),
        },
    )
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
