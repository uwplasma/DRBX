from __future__ import annotations

"""
04_stellarator_nearaxis_essos.py

Purpose
-------
Run `jaxdrb` on a **near-axis stellarator configuration** via ESSOS.

This replaces the older pyQSC-based workflow with the near-axis solver in ESSOS, while keeping
the `jaxdrb` philosophy unchanged:

  - build a 1D field-line geometry along a periodic parallel coordinate,
  - reduce perpendicular operators to algebraic k_perp^2(l),
  - compute linear growth rates by matrix-free Arnoldi (J·v).

Scientific context
------------------
Near-axis expansions are widely used for rapid design iteration and for reduced turbulence and
stability studies. Jorge & Landreman (2021) discuss how to obtain geometric quantities along a
field line from a near-axis solution and compare them to full equilibria.

This example is intentionally lightweight and pedagogic: it uses a **local orthonormal**
perpendicular metric around the traced field line (so k_perp^2 ≈ kx^2 + ky^2) while retaining
the **field-line curvature** drive. This is enough to demonstrate how changing near-axis
parameters affects instability, and it is a stepping stone toward a full Clebsch-metric
implementation.

Run
---
  python examples/3_advanced/04_stellarator_nearaxis_essos.py

Environment knobs
-----------------
Set `JAXDRB_FAST=0` for a slower, higher-resolution run.

Outputs
-------
Written to `out/3_advanced/stellarator_nearaxis_essos/`.
"""

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import (
    save_geometry_overview,
    save_json,
    save_scan_panels,
    set_mpl_style,
)
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.essos import near_axis_fieldline_to_tabulated
from jaxdrb.geometry.tabulated import TabulatedGeometry
from jaxdrb.models.params import DRBParams
from jaxdrb.models.registry import get_model


def main() -> None:
    out_dir = Path("out/3_advanced/stellarator_nearaxis_essos")
    out_dir.mkdir(parents=True, exist_ok=True)

    fast = os.environ.get("JAXDRB_FAST", "1") != "0"

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    # A simple nfp=3 near-axis configuration (matches the ESSOS near-axis example style).
    rc = np.array([1.0, 0.045])
    zs = np.array([0.0, -0.045])
    etabar = -0.9
    nfp = 3
    r = 0.1
    alpha0 = 0.0

    geom_file = out_dir / "geom_nearaxis.npz"
    print("Building near-axis tabulated geometry via ESSOS…", flush=True)
    res = near_axis_fieldline_to_tabulated(
        rc=rc,
        zs=zs,
        etabar=etabar,
        nfp=nfp,
        r=r,
        alpha=alpha0,
        nphi=81 if fast else 161,
        out_path=geom_file,
    )
    print(f"Wrote geometry: {res.path}", flush=True)

    geom = TabulatedGeometry.from_npz(geom_file)
    save_geometry_overview(out_dir, geom=geom, kx=0.0, ky=0.3)

    # A standard cold-ion electrostatic case with curvature drive.
    # This is not a dedicated "benchmark point" from the NAQS paper; it is a baseline that
    # produces a clear unstable branch.
    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=2.0,
        me_hat=1e-3,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )

    model = get_model("cold-ion-es")
    ky = np.linspace(0.08, 1.0, 14) if fast else np.linspace(0.06, 1.2, 26)
    kx = 0.0

    print("Running ky scan…", flush=True)
    scan = scan_ky(
        params,
        geom,
        ky=ky,
        kx=kx,
        nl=int(geom.l.size),
        model=model,
        arnoldi_m=24,
        arnoldi_tol=2e-3,
        nev=6,
        do_initial_value=False,
        verbose=True,
        print_every=2,
        seed=0,
    )

    save_scan_panels(
        out_dir,
        ky=scan.ky,
        gamma=scan.gamma_eigs,
        omega=scan.omega_eigs,
        title="Near-axis stellarator (ESSOS): ky scan (cold-ion ES)",
        filename="scan_panel.png",
    )

    np.savez(
        out_dir / "results.npz", ky=scan.ky, gamma_eigs=scan.gamma_eigs, omega_eigs=scan.omega_eigs
    )
    save_json(
        out_dir / "params.json",
        {
            "model": model.name,
            "geom_file": str(geom_file),
            "meta": res.meta,
            "kx": kx,
            "ky": {"min": float(ky.min()), "max": float(ky.max()), "n": int(ky.size)},
            "params": params.__dict__,
            "notes": [
                "Perp metric uses a local orthonormal frame around the field line (see src/jaxdrb/geometry/essos.py)."
            ],
        },
    )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
