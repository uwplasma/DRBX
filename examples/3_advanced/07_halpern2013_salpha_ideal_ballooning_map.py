from __future__ import annotations

"""
07_halpern2013_salpha_ideal_ballooning_map.py

Purpose
-------
Create a Halpern-like "s-alpha diagram" showing the growth rate of a curvature-driven branch as a
function of:

  - magnetic shear: shat
  - s-alpha parameter: alpha

and highlight how the most unstable growth rate changes across (shat, alpha).

Interpretation and caveats
--------------------------
Halpern et al. (2013) discuss ideal ballooning physics in the tokamak SOL, including electromagnetic
effects. In `jaxdrb`, we provide a **workflow-replica** diagram:

  - geometry: s-alpha analytic model (`SAlphaGeometry`)
  - physics: electromagnetic model variant (`--model em`) with finite `beta`

This is not a full ideal MHD ballooning benchmark; the goal is a clear, reproducible scan that can
be extended.

Run
---
  python examples/3_advanced/07_halpern2013_salpha_ideal_ballooning_map.py

Environment knobs
-----------------
Set `JAXDRB_FAST=0` for a finer grid (slower).

Outputs
-------
Written to `out/3_advanced/halpern2013_salpha_ideal_ballooning_map/`:

  - `gamma_max_shat_alpha.png`: heatmap of gamma(shat, alpha) at a fixed ky
  - `results.npz`: arrays
"""

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, set_mpl_style
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.tokamak import SAlphaGeometry
from jaxdrb.models.params import DRBParams
from jaxdrb.models.registry import get_model


def main() -> None:
    out_dir = Path("out/3_advanced/halpern2013_salpha_ideal_ballooning_map")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    fast = os.environ.get("JAXDRB_FAST", "1") != "0"
    if fast:
        print("JAXDRB_FAST=1: using a coarse grid for a quick, pedagogic run.", flush=True)
    else:
        print("JAXDRB_FAST=0: using a finer grid (may take a long time).", flush=True)

    model = get_model("em")

    nl = 48 if fast else 96
    q = 1.4
    R0 = 1.0
    epsilon = 0.18
    curvature0 = epsilon

    # In an ideal ballooning (MHD-like) setting, the growth rate is scale-invariant with respect
    # to perpendicular wavelength, so it is common to report gamma as a function of (shat, alpha)
    # at a representative k_y. We follow that *workflow* here by fixing k_y.
    kx = 0.0
    ky0 = 0.3
    ky = np.array([ky0], dtype=float)

    # Parameter grids
    shat_grid = np.linspace(0.0, 1.5, 6 if fast else 26)
    alpha_grid = np.linspace(0.0, 2.0, 7 if fast else 31)

    # Model parameters: choose a finite beta and low resistivity to emphasize inductive response.
    # NOTE: This is still a reduced model, not an ideal-MHD limit.
    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=0.1,
        me_hat=0.05,
        beta=0.15,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        Dpsi=0.0,
    )

    gamma = np.zeros((shat_grid.size, alpha_grid.size))

    print(f"Scanning (shat, alpha)… ({shat_grid.size}×{alpha_grid.size}) at ky={ky0:g}", flush=True)
    for i, shat in enumerate(shat_grid):
        for j, alpha in enumerate(alpha_grid):
            geom = SAlphaGeometry.make(
                nl=nl,
                length=float(2 * np.pi),
                shat=float(shat),
                alpha=float(alpha),
                q=q,
                R0=R0,
                epsilon=epsilon,
                curvature0=curvature0,
            )
            s = scan_ky(
                params,
                geom,
                ky=ky,
                kx=kx,
                nl=nl,
                model=model,
                arnoldi_m=24 if fast else 40,
                arnoldi_tol=3e-3 if fast else 2e-3,
                arnoldi_max_m=5 * nl,
                nev=3 if fast else 6,
                do_initial_value=False,
                verbose=False,
                seed=0,
            )
            gamma[i, j] = float(s.gamma_eigs[0])
        print(f"[{i+1:02d}/{shat_grid.size}] shat={shat:6.3f} done", flush=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    im = ax.pcolormesh(alpha_grid, shat_grid, gamma, shading="auto", cmap="magma")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\hat{s}$")
    ax.set_title(
        rf"Electromagnetic model: $\gamma(\hat{{s}},\alpha)$ at $k_y={ky0:g}$ (workflow replica)"
    )
    fig.colorbar(im, ax=ax, label=r"$\gamma$")
    fig.savefig(out_dir / "gamma_max_shat_alpha.png", dpi=240)
    plt.close(fig)

    np.savez(
        out_dir / "results.npz",
        shat=shat_grid,
        alpha=alpha_grid,
        gamma=gamma,
        ky=ky,
        params=np.array([params.omega_n, params.eta, params.me_hat, params.beta], dtype=float),
    )
    save_json(
        out_dir / "params.json",
        {
            "model": model.name,
            "geometry": {
                "type": "salpha",
                "q": q,
                "R0": R0,
                "epsilon": epsilon,
                "curvature0": curvature0,
            },
            "nl": nl,
            "kx": kx,
            "ky": {"min": float(ky.min()), "max": float(ky.max()), "n": int(ky.size)},
            "shat": {
                "min": float(shat_grid.min()),
                "max": float(shat_grid.max()),
                "n": int(shat_grid.size),
            },
            "alpha": {
                "min": float(alpha_grid.min()),
                "max": float(alpha_grid.max()),
                "n": int(alpha_grid.size),
            },
            "params": params.__dict__,
            "notes": [
                "This is a workflow replica (not a full ideal-MHD ballooning benchmark).",
                "Try varying beta, eta, and the ky range to explore different trends.",
            ],
        },
    )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
