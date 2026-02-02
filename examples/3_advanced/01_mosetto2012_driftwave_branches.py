from __future__ import annotations

"""
01_mosetto2012_driftwave_branches.py

Purpose
-------
Reproduce the *workflow* of drift-wave branch separation discussed in:

  A. Mosetto et al., "Low-frequency linear-mode regimes in the tokamak scrape-off layer",
  Phys. Plasmas 19, 112103 (2012).

In the SOL/edge literature, it is common to separate "branches" by ordering:
- resistive drift-wave-like (small electron inertia, finite resistivity),
- inertial drift-wave-like (finite inertia, weak resistivity).

The default `jaxdrb` model is electrostatic and simplified, so expect **qualitative** agreement rather than
quantitative reproduction of any specific figure. The point is to provide a clear, hackable
reference implementation of the analysis steps used in the literature.

Run
---
  python examples/3_advanced/01_mosetto2012_driftwave_branches.py

Outputs
-------
Written to `out/3_advanced/mosetto2012_driftwave_branches/`.
"""

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
from jaxdrb.analysis.scan import Scan1DResult, scan_ky
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.arnoldi import arnoldi_eigs, arnoldi_leading_ritz_vector
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import State, equilibrium
from jaxdrb.models.params import DRBParams


def _ky_star(scan: Scan1DResult) -> tuple[float, int]:
    ratio = np.maximum(scan.gamma_eigs, 0.0) / scan.ky
    i = int(np.argmax(ratio))
    return float(scan.ky[i]), i


def _run_branch(
    out_dir: Path,
    *,
    name: str,
    params: DRBParams,
    geom,
    kx: float,
    ky: np.ndarray,
    seed: int,
) -> tuple[Scan1DResult, float]:
    print(f"\n=== Branch: {name} ===", flush=True)
    scan = scan_ky(
        params,
        geom,
        ky=ky,
        kx=kx,
        arnoldi_m=35,
        arnoldi_tol=2e-3,
        nev=8,
        do_initial_value=True,
        tmax=25.0,
        dt0=0.02,
        nsave=160,
        verbose=True,
        print_every=1,
        seed=seed,
    )

    ky_star, _i = _ky_star(scan)
    print(f"[{name}] ky*={ky_star:.4f}", flush=True)

    save_scan_panels(
        out_dir,
        ky=scan.ky,
        gamma=scan.gamma_eigs,
        omega=scan.omega_eigs,
        gamma_iv=scan.gamma_iv,
        title=f"Mosetto 2012 drift-wave workflow: {name}",
        filename=f"scan_panel_{name}.png",
    )
    save_geometry_overview(out_dir, geom=geom, kx=kx, ky=ky_star, filename=f"geometry_{name}.png")

    # Eigenfunction and spectrum at ky* (use a larger Arnoldi space for cleaner vectors).
    nl = int(getattr(geom, "l").size)
    y_eq = equilibrium(nl)
    v0 = State.random(jax.random.PRNGKey(0), nl, amplitude=1e-3)
    matvec = linear_matvec(y_eq, params, geom, kx=kx, ky=ky_star)

    print(f"[{name}] computing leading Ritz vector at ky*…", flush=True)
    ritz = arnoldi_leading_ritz_vector(matvec, v0, m=100, seed=seed)
    save_eigenfunction_panel(
        out_dir,
        geom=geom,
        state=ritz.vector,
        eigenvalue=ritz.eigenvalue,
        kx=kx,
        ky=ky_star,
        kperp2_min=params.kperp2_min,
        filename=f"eigenfunctions_{name}.png",
    )

    arn = arnoldi_eigs(matvec, v0, m=100, nev=100, seed=seed)
    save_eigenvalue_spectrum(
        out_dir,
        eigenvalues=arn.eigenvalues,
        highlight=ritz.eigenvalue,
        filename=f"spectrum_{name}.png",
    )

    return scan, ky_star


def main() -> None:
    out_dir = Path("out/3_advanced/mosetto2012_driftwave_branches")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    # --- Geometry: slab, curvature OFF to isolate drift-wave-like dynamics ---
    nl = 64
    geom = SlabGeometry.make(nl=nl, length=float(2 * np.pi), shat=0.0, curvature0=0.0)

    ky = np.linspace(0.05, 1.2, 32)
    kx = 0.0

    # Resistive drift-wave-like: small electron inertia, finite resistivity
    params_rdw = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=2e-3,
        curvature_on=False,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        kperp2_min=1e-6,
    )

    # Inertial drift-wave-like: finite inertia, weak resistivity
    params_idw = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=0.02,
        me_hat=0.5,
        curvature_on=False,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        kperp2_min=1e-6,
    )

    scan_rdw, ky_star_rdw = _run_branch(
        out_dir, name="rdw_like", params=params_rdw, geom=geom, kx=kx, ky=ky, seed=0
    )
    scan_idw, ky_star_idw = _run_branch(
        out_dir, name="idw_like", params=params_idw, geom=geom, kx=kx, ky=ky, seed=1
    )

    # --- Overlay plot: gamma(ky) and gamma/ky(ky) for both branches ---
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(7.2, 7.6), sharex=True, constrained_layout=True)
    ax = axs[0]
    ax.plot(ky, scan_rdw.gamma_eigs, "o-", label="RDW-like (eig)")
    ax.plot(ky, scan_idw.gamma_eigs, "s--", label="IDW-like (eig)")
    ax.axvline(ky_star_rdw, color="k", alpha=0.25, linestyle="--")
    ax.axvline(ky_star_idw, color="k", alpha=0.25, linestyle=":")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("Mosetto 2012 workflow: drift-wave branches (curvature off)")
    ax.legend()

    ax = axs[1]
    ax.plot(ky, np.maximum(scan_rdw.gamma_eigs, 0.0) / ky, "o-", label="RDW-like")
    ax.plot(ky, np.maximum(scan_idw.gamma_eigs, 0.0) / ky, "s--", label="IDW-like")
    ax.axvline(ky_star_rdw, color="k", alpha=0.25, linestyle="--")
    ax.axvline(ky_star_idw, color="k", alpha=0.25, linestyle=":")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\max(\gamma,0)/k_y$")
    ax.legend()
    fig.savefig(out_dir / "branches_overlay.png", dpi=220)
    plt.close(fig)

    np.savez(
        out_dir / "results.npz",
        ky=ky,
        gamma_rdw=scan_rdw.gamma_eigs,
        omega_rdw=scan_rdw.omega_eigs,
        gamma_idw=scan_idw.gamma_eigs,
        omega_idw=scan_idw.omega_eigs,
        gamma_rdw_iv=scan_rdw.gamma_iv,
        gamma_idw_iv=scan_idw.gamma_iv,
        ky_star_rdw=ky_star_rdw,
        ky_star_idw=ky_star_idw,
    )
    save_json(
        out_dir / "params.json",
        {
            "geom": {"type": "slab", "nl": nl, "shat": 0.0, "curvature0": 0.0},
            "params_rdw": params_rdw.__dict__,
            "params_idw": params_idw.__dict__,
        },
    )
    save_json(
        out_dir / "summary.json",
        {"ky_star": {"rdw_like": ky_star_rdw, "idw_like": ky_star_idw}},
    )

    print(f"\nWrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
