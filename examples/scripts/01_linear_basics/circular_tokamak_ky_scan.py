"""
circular_tokamak_ky_scan.py

Purpose
-------
Run a basic ky scan in a large-aspect-ratio circular tokamak geometry.

This example is still "simple" (single ky scan, single kx), but it turns ON curvature
so you can see how magnetic geometry changes the growth rates and eigenfunctions.

Run
---
  python examples/scripts/01_linear_basics/circular_tokamak_ky_scan.py

Outputs
-------
Written to `out/examples/01_linear_basics/circular_tokamak_ky_scan/`.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import (
    save_eigenfunction_panel,
    save_eigenvalue_spectrum,
    save_geometry_overview,
    save_json,
    save_scan_panels,
    set_mpl_style,
)
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.tokamak import CircularTokamakGeometry
from jaxdrb.linear.arnoldi import arnoldi_eigs, arnoldi_leading_ritz_vector
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import State, equilibrium
from jaxdrb.models.params import DRBParams


def main() -> None:
    out_dir = Path("out/examples/01_linear_basics/circular_tokamak_ky_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    # --- Geometry (analytic circular tokamak) ---
    nl = 64
    geom = CircularTokamakGeometry.make(
        nl=nl,
        length=float(2 * np.pi),
        shat=0.8,
        q=3.0,
        R0=1.0,
        epsilon=0.18,
        curvature0=0.18,
    )

    # --- Physics parameters ---
    params = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        kperp2_min=1e-6,
    )

    kx = 0.0
    ky = np.linspace(0.05, 1.0, 26)

    print("Running ky scan (circular tokamak)…", flush=True)
    scan = scan_ky(
        params,
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
        seed=0,
    )

    ratio = np.maximum(scan.gamma_eigs, 0.0) / scan.ky
    i_star = int(np.argmax(ratio))
    ky_star = float(scan.ky[i_star])
    lam_star = complex(scan.gamma_eigs[i_star] + 1j * scan.omega_eigs[i_star])

    scan_panel = save_scan_panels(
        out_dir,
        ky=scan.ky,
        gamma=scan.gamma_eigs,
        omega=scan.omega_eigs,
        gamma_iv=scan.gamma_iv,
        title="Circular tokamak: ky scan",
        filename="scan_panel.png",
    )
    geom_fig = save_geometry_overview(out_dir, geom=geom, kx=kx, ky=ky_star)

    y_eq = equilibrium(nl)
    v0 = State.random(jax.random.PRNGKey(0), nl, amplitude=1e-3)
    matvec = linear_matvec(y_eq, params, geom, kx=kx, ky=ky_star)

    print("Computing leading Ritz vector (for eigenfunction plot)…", flush=True)
    ritz = arnoldi_leading_ritz_vector(matvec, v0, m=80, seed=0)
    eigfun_fig = save_eigenfunction_panel(
        out_dir,
        geom=geom,
        state=ritz.vector,
        eigenvalue=ritz.eigenvalue,
        kx=kx,
        ky=ky_star,
        kperp2_min=params.kperp2_min,
        filename="eigenfunctions.png",
    )

    arn = arnoldi_eigs(matvec, v0, m=80, nev=80, seed=0)
    spectrum_fig = save_eigenvalue_spectrum(
        out_dir, eigenvalues=arn.eigenvalues, highlight=ritz.eigenvalue
    )

    np.savez(
        out_dir / "results.npz",
        ky=scan.ky,
        gamma_eigs=scan.gamma_eigs,
        omega_eigs=scan.omega_eigs,
        gamma_iv=scan.gamma_iv,
        omega_iv=scan.omega_iv,
        eigs=scan.eigs,
        ky_star=ky_star,
        lam_star=np.asarray(lam_star),
    )

    save_json(
        out_dir / "params.json",
        {
            "geom": {
                "type": "circular_tokamak",
                "nl": nl,
                "shat": float(geom.shat),
                "q": float(geom.q),
                "R0": float(geom.R0),
                "epsilon": float(geom.epsilon),
                "curvature0": float(geom.curvature0),
            },
            "kx": kx,
            "ky": {"min": float(ky.min()), "max": float(ky.max()), "n": int(ky.size)},
            "params": params.__dict__,
        },
    )
    save_json(
        out_dir / "summary.json",
        {
            "ky_star": ky_star,
            "lambda_star": {"real": lam_star.real, "imag": lam_star.imag},
            "ritz_residual_norm": float(ritz.residual_norm),
            "figures": {
                "scan_panel": str(scan_panel),
                "geometry_overview": str(geom_fig),
                "eigenfunctions": str(eigfun_fig),
                "spectrum": str(spectrum_fig),
            },
        },
    )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
