from __future__ import annotations

"""
09_essos_nearaxis_r01.py

Purpose
-------
Replace the old pyQSC-based near-axis example with ESSOS by:

  1) building a near-axis configuration (`essos.fields.near_axis`),
  2) converting a field line at r=0.1 into a `TabulatedGeometry` file,
  3) running the standard `jaxdrb` ky-scan workflow (branches + Lp proxy).

This example is inspired by the near-axis geometry discussion in:

  R. Jorge & M. Landreman, Plasma Phys. Control. Fusion 63 (2021) 014001.

Run
---
  python examples/3_advanced/09_essos_nearaxis_r01.py

Outputs
-------
Written to `out/3_advanced/essos_nearaxis_r01/`.
"""

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.lp import solve_lp_fixed_point
from jaxdrb.analysis.plotting import (
    save_geometry_overview,
    save_json,
    save_lp_history,
    save_scan_panels,
    set_mpl_style,
)
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.essos import near_axis_fieldline_to_tabulated
from jaxdrb.geometry.tabulated import TabulatedGeometry
from jaxdrb.models.params import DRBParams
from jaxdrb.models.registry import get_model


def main() -> None:
    out_dir = Path("out/3_advanced/essos_nearaxis_r01")
    out_dir.mkdir(parents=True, exist_ok=True)

    fast = os.environ.get("JAXDRB_FAST", "1") != "0"

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    # Near-axis configuration matching ESSOS' nfp=3 example (see essos/examples/optimize_coils_for_nearaxis.py)
    rc = np.array([1.0, 0.045])
    zs = np.array([0.0, -0.045])
    etabar = -0.9
    nfp = 3
    r = 0.1
    alpha0 = 0.0

    geom_file = out_dir / "geom_nearaxis_r01.npz"
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

    ky = np.linspace(0.08, 1.0, 14) if fast else np.linspace(0.06, 1.2, 26)
    kx = 0.0
    q_eff = 3.0  # only used in the Halpern fixed-point rule; not a true safety factor here

    model = get_model("cold-ion-es")

    # Two parameter sets to expose resistive-like vs inertial-like behavior.
    cases = [
        (
            "RDW-like",
            DRBParams(
                omega_n=0.8, eta=2.0, me_hat=1e-3, curvature_on=True, Dn=0.01, DOmega=0.01, DTe=0.01
            ),
        ),
        (
            "InDW-like",
            DRBParams(
                omega_n=0.8, eta=0.1, me_hat=0.08, curvature_on=True, Dn=0.01, DOmega=0.01, DTe=0.01
            ),
        ),
    ]

    scans = []
    for name, params in cases:
        print(f"Running ky scan for case: {name}", flush=True)
        s = scan_ky(
            params,
            geom,
            ky=ky,
            kx=kx,
            nl=int(geom.l.size),
            model=model,
            arnoldi_m=30,
            arnoldi_tol=2e-3,
            nev=6,
            do_initial_value=False,
            verbose=True,
            print_every=2,
            seed=0,
        )
        scans.append((name, params, s))
        save_scan_panels(
            out_dir,
            ky=s.ky,
            gamma=s.gamma_eigs,
            omega=s.omega_eigs,
            title=f"ESSOS near-axis (r=0.1): {name}",
            filename=f"scan_panel_{name.replace(' ', '_').replace('-', '_')}.png",
        )

    # Halpern fixed-point Lp workflow (toy mapping): omega_n ~ 1/Lp
    print("\nSolving fixed point for Lp (toy workflow)…", flush=True)
    base = cases[0][1]
    lp = solve_lp_fixed_point(
        base,
        geom,
        q=q_eff,
        ky=ky,
        Lp0=20.0,
        omega_n_scale=1.0,
        max_iter=6 if fast else 15,
        relax=0.7,
        arnoldi_m=16 if fast else 24,
        arnoldi_tol=3e-3,
        nev=4,
        verbose=True,
    )
    save_lp_history(out_dir, history=lp.history, q=q_eff, filename="lp_fixed_point_history.png")

    np.savez(
        out_dir / "results.npz",
        ky=ky,
        gamma=np.stack([s.gamma_eigs for _, _, s in scans], axis=0),
        omega=np.stack([s.omega_eigs for _, _, s in scans], axis=0),
        labels=np.array([name for name, _, _ in scans]),
        Lp=lp.Lp,
        ky_star=lp.ky_star,
        gamma_over_ky_star=lp.gamma_over_ky_star,
    )
    save_json(
        out_dir / "params.json",
        {
            "geom_file": str(geom_file),
            "near_axis": {
                "rc": rc.tolist(),
                "zs": zs.tolist(),
                "etabar": etabar,
                "nfp": nfp,
                "r": r,
            },
            "kx": kx,
            "ky": {"min": float(ky.min()), "max": float(ky.max()), "n": int(ky.size)},
            "cases": [{"name": name, "params": p.__dict__} for name, p, _ in scans],
            "Lp_fixed_point": {"q_eff": q_eff, "Lp": lp.Lp, "ky_star": lp.ky_star},
            "notes": [
                "Perp metric uses a local orthonormal frame around the field line (see src/jaxdrb/geometry/essos.py).",
                "q_eff is a heuristic scale for the Halpern fixed-point rule in a stellarator setting.",
            ],
        },
    )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
