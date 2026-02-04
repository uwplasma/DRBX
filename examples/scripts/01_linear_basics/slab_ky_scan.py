"""
slab_ky_scan.py

Purpose
-------
Minimal "hello world" run for jax_drb:

- Choose a simple shear-slab geometry.
- Scan over ky for a single kx.
- Compute growth rates by:
  - matrix-free Arnoldi eigenvalues (Ritz values),
  - initial-value (Diffrax) time evolution of dv/dt = J v.
- Save multiple diagnostic figures:
  - a multi-panel scan summary,
  - geometry coefficients along the field line,
  - a leading eigenfunction panel (mode structure),
  - a Ritz spectrum scatter plot.

Run
---
From the repository root:

  python examples/scripts/01_linear_basics/slab_ky_scan.py

Outputs
-------
Written to `out/examples/01_linear_basics/slab_ky_scan/`.
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
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.arnoldi import arnoldi_eigs, arnoldi_leading_ritz_vector
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import State, equilibrium
from jaxdrb.models.params import DRBParams


def main() -> None:
    out_dir = Path("out/examples/01_linear_basics/slab_ky_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    # --- Geometry (simple slab) ---
    nl = 64
    geom = SlabGeometry.make(nl=nl, length=float(2 * np.pi), shat=0.0, curvature0=0.0)

    # --- Physics parameters ---
    # Curvature is OFF here so we isolate drift-wave-like dynamics.
    params = DRBParams(
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

    kx = 0.0
    ky = np.linspace(0.05, 1.2, 28)

    print("Running ky scan (slab)…", flush=True)
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
        print_every=1,
        seed=0,
    )

    # Choose ky* that maximizes max(gamma,0)/ky (transport proxy used in SOL literature).
    ratio = np.maximum(scan.gamma_eigs, 0.0) / scan.ky
    i_star = int(np.argmax(ratio))
    ky_star = float(scan.ky[i_star])
    lam_star = complex(scan.gamma_eigs[i_star] + 1j * scan.omega_eigs[i_star])

    print(
        f"ky*={ky_star:.4f}  gamma*={lam_star.real:+.4e}  omega*={lam_star.imag:+.4e}", flush=True
    )

    # Save a compact scan panel and geometry overview at ky*.
    scan_panel = save_scan_panels(
        out_dir,
        ky=scan.ky,
        gamma=scan.gamma_eigs,
        omega=scan.omega_eigs,
        gamma_iv=scan.gamma_iv,
        title="Slab: ky scan (curvature off)",
        filename="scan_panel.png",
    )
    geom_fig = save_geometry_overview(out_dir, geom=geom, kx=kx, ky=ky_star)

    # Compute a leading Ritz vector at ky* so we can plot eigenfunctions.
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

    # Also save a richer Ritz spectrum at ky*.
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
            "geom": {"type": "slab", "nl": nl, "shat": 0.0, "curvature0": 0.0},
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
