"""
salpha_cyclone_ky_scan.py

Purpose
-------
Run a ky scan in an analytic s-alpha tokamak geometry using Cyclone-like parameters.

The "Cyclone Base Case" is most often discussed in gyrokinetic contexts, but the same
geometry parameters are useful as a convenient analytic testbed for local field-line models.

Run
---
  python examples/01_linear_basics/salpha_cyclone_ky_scan.py

Outputs
-------
Written to `out/examples/01_linear_basics/salpha_cyclone_ky_scan/`.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
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
from jaxdrb.geometry.tokamak import SAlphaGeometry
from jaxdrb.linear.arnoldi import arnoldi_eigs, arnoldi_leading_ritz_vector
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import State, equilibrium
from jaxdrb.models.params import DRBParams


def main() -> None:
    out_dir = Path("out/examples/01_linear_basics/salpha_cyclone_ky_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    nl = 64
    geom = SAlphaGeometry.cyclone_base_case(nl=nl)

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

    print("Running ky scan (s-alpha / Cyclone-like)…", flush=True)
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
        title="s-alpha (Cyclone-like): ky scan",
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
                "type": "salpha_cyclone",
                "nl": nl,
                "shat": float(geom.shat),
                "alpha": float(geom.alpha),
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
