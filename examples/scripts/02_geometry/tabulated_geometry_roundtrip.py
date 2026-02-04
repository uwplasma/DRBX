"""
tabulated_geometry_roundtrip.py

Purpose
-------
Demonstrate and validate the "tabulated geometry" interface:

1) Start from an analytic geometry (here: shear-slab).
2) Export its field-line coefficients to a `.npz` file.
3) Reload the file via `TabulatedGeometry`.
4) Confirm that linear growth rates match between analytic and tabulated geometries.

This is a key engineering requirement: swapping geometries should only change the geometry
provider, not the model core.

Run
---
  python examples/scripts/02_geometry/tabulated_geometry_roundtrip.py

Outputs
-------
Written to `out/examples/02_geometry/tabulated_geometry_roundtrip/`.
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
from jaxdrb.geometry.tabulated import TabulatedGeometry
from jaxdrb.linear.arnoldi import arnoldi_eigs, arnoldi_leading_ritz_vector
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import State, equilibrium
from jaxdrb.models.params import DRBParams


def _write_geometry_npz(geom, path: Path) -> None:
    coeffs = geom.coefficients()
    payload = {k: np.asarray(v) for k, v in coeffs.items()}
    np.savez(path, **payload)


def main() -> None:
    out_dir = Path("out/examples/02_geometry/tabulated_geometry_roundtrip")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    # --- Geometry ---
    nl = 96
    geom = SlabGeometry.make(nl=nl, length=float(2 * np.pi), shat=0.8, curvature0=0.2)

    geom_file = out_dir / "slab_geometry.npz"
    _write_geometry_npz(geom, geom_file)
    tab = TabulatedGeometry.from_npz(geom_file)

    # --- Physics ---
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
    ky = np.linspace(0.05, 1.0, 24)

    print("Running ky scan on analytic slab geometry…", flush=True)
    scan_analytic = scan_ky(
        params,
        geom,
        ky=ky,
        kx=kx,
        arnoldi_m=35,
        arnoldi_tol=1e-3,
        nev=6,
        do_initial_value=False,
        verbose=True,
        print_every=2,
        seed=0,
    )

    print("Running ky scan on TabulatedGeometry (same coefficients)…", flush=True)
    scan_tab = scan_ky(
        params,
        tab,
        ky=ky,
        kx=kx,
        arnoldi_m=35,
        arnoldi_tol=1e-3,
        nev=6,
        do_initial_value=False,
        verbose=True,
        print_every=2,
        seed=0,
    )

    # --- Validate the round-trip ---
    rel = np.max(np.abs(scan_tab.gamma_eigs - scan_analytic.gamma_eigs)) / (
        np.max(np.abs(scan_analytic.gamma_eigs)) + 1e-12
    )
    print(f"Round-trip max relative error in gamma: {rel:.3e}", flush=True)

    # Choose ky* from the tabulated scan.
    ratio = np.maximum(scan_tab.gamma_eigs, 0.0) / scan_tab.ky
    i_star = int(np.argmax(ratio))
    ky_star = float(scan_tab.ky[i_star])

    # Plots + diagnostics
    save_scan_panels(
        out_dir,
        ky=scan_tab.ky,
        gamma=scan_tab.gamma_eigs,
        omega=scan_tab.omega_eigs,
        title="TabulatedGeometry: ky scan (slab round-trip)",
        filename="scan_panel_tabulated.png",
    )
    save_scan_panels(
        out_dir,
        ky=scan_analytic.ky,
        gamma=scan_analytic.gamma_eigs,
        omega=scan_analytic.omega_eigs,
        title="Analytic slab: ky scan",
        filename="scan_panel_analytic.png",
    )
    save_geometry_overview(out_dir, geom=geom, kx=kx, ky=ky_star, filename="geom_analytic.png")
    save_geometry_overview(out_dir, geom=tab, kx=kx, ky=ky_star, filename="geom_tabulated.png")

    y_eq = equilibrium(nl)
    v0 = State.random(jax.random.PRNGKey(0), nl, amplitude=1e-3)
    matvec = linear_matvec(y_eq, params, tab, kx=kx, ky=ky_star)

    print("Computing leading Ritz vector at ky* (tabulated)…", flush=True)
    ritz = arnoldi_leading_ritz_vector(matvec, v0, m=80, seed=0)
    save_eigenfunction_panel(
        out_dir,
        geom=tab,
        state=ritz.vector,
        eigenvalue=ritz.eigenvalue,
        kx=kx,
        ky=ky_star,
        kperp2_min=params.kperp2_min,
        filename="eigenfunctions_tabulated.png",
    )
    arn = arnoldi_eigs(matvec, v0, m=80, nev=80, seed=0)
    save_eigenvalue_spectrum(out_dir, eigenvalues=arn.eigenvalues, highlight=ritz.eigenvalue)

    np.savez(
        out_dir / "results.npz",
        ky=scan_tab.ky,
        gamma_tab=scan_tab.gamma_eigs,
        gamma_analytic=scan_analytic.gamma_eigs,
        omega_tab=scan_tab.omega_eigs,
        omega_analytic=scan_analytic.omega_eigs,
        ky_star=ky_star,
        rel_error_gamma=rel,
    )
    save_json(
        out_dir / "summary.json",
        {
            "roundtrip": {"rel_error_gamma_max": float(rel), "ky_star": ky_star},
            "geom_file": str(geom_file),
        },
    )
    save_json(
        out_dir / "params.json",
        {
            "geom": {
                "type": "slab",
                "nl": nl,
                "shat": float(geom.shat),
                "curvature0": float(geom.curvature0),
            },
            "kx": kx,
            "ky": {"min": float(ky.min()), "max": float(ky.max()), "n": int(ky.size)},
            "params": params.__dict__,
        },
    )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
