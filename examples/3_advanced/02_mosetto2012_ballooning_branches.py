from __future__ import annotations

"""
02_mosetto2012_ballooning_branches.py

Purpose
-------
Demonstrate curvature-driven ("ballooning-like") branches and shear trends, inspired by:

  A. Mosetto et al., Phys. Plasmas 19, 112103 (2012).

We run ky scans while:
- turning curvature ON,
- varying magnetic shear (ŝ) in a simple slab geometry,
- toggling resistive-like vs inertial-like parameter orderings.

Run
---
  python examples/3_advanced/02_mosetto2012_ballooning_branches.py

Outputs
-------
Written to `out/3_advanced/mosetto2012_ballooning_branches/`.
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
    set_mpl_style,
)
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.arnoldi import arnoldi_eigs, arnoldi_leading_ritz_vector
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import State, equilibrium
from jaxdrb.models.params import DRBParams


def main() -> None:
    out_dir = Path("out/3_advanced/mosetto2012_ballooning_branches")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    nl = 64
    ky = np.linspace(0.05, 1.0, 36)
    kx = 0.0

    # Two parameter sets to highlight different closures.
    params_rbm = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=2e-3,
        curvature_on=True,
        Dn=0.02,
        DOmega=0.02,
        DTe=0.02,
        kperp2_min=1e-6,
    )
    params_inbm = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=0.02,
        me_hat=0.5,
        curvature_on=True,
        Dn=0.02,
        DOmega=0.02,
        DTe=0.02,
        kperp2_min=1e-6,
    )

    shats = [0.0, 0.5, 1.0]
    curvature0 = 0.35

    scans_rbm = []
    scans_inbm = []
    for shat in shats:
        geom = SlabGeometry.make(
            nl=nl, length=float(2 * np.pi), shat=float(shat), curvature0=curvature0
        )
        print(f"Scanning ŝ={shat:g} (RBM-like)…", flush=True)
        scans_rbm.append(
            scan_ky(
                params_rbm,
                geom,
                ky=ky,
                kx=kx,
                arnoldi_m=35,
                arnoldi_tol=2e-3,
                nev=6,
                do_initial_value=False,
                verbose=True,
                print_every=4,
                seed=0,
            )
        )
        print(f"Scanning ŝ={shat:g} (InBM-like)…", flush=True)
        scans_inbm.append(
            scan_ky(
                params_inbm,
                geom,
                ky=ky,
                kx=kx,
                arnoldi_m=35,
                arnoldi_tol=2e-3,
                nev=6,
                do_initial_value=False,
                verbose=True,
                print_every=4,
                seed=1,
            )
        )

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(7.4, 8.2), sharex=True, constrained_layout=True)

    ax = axs[0]
    for shat, s in zip(shats, scans_rbm, strict=True):
        ax.plot(ky, s.gamma_eigs, marker="o", label=rf"ŝ={shat:g}")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("Ballooning-like branch (RBM-like ordering)")
    ax.legend(title="shear", fontsize=9)

    ax = axs[1]
    for shat, s in zip(shats, scans_inbm, strict=True):
        ax.plot(ky, s.gamma_eigs, marker="o", label=rf"ŝ={shat:g}")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("Ballooning-like branch (InBM-like ordering)")
    ax.legend(title="shear", fontsize=9)

    fig.savefig(out_dir / "gamma_ky_shear_scan.png", dpi=220)
    plt.close(fig)

    # Pick the most-unstable case among all shears for an eigenfunction panel (RBM-like).
    gmax = -np.inf
    best = None
    for shat, s in zip(shats, scans_rbm, strict=True):
        i = int(np.argmax(s.gamma_eigs))
        if s.gamma_eigs[i] > gmax:
            gmax = float(s.gamma_eigs[i])
            best = (float(shat), float(ky[i]))
    assert best is not None
    shat_best, ky_best = best

    geom_best = SlabGeometry.make(
        nl=nl, length=float(2 * np.pi), shat=shat_best, curvature0=curvature0
    )
    save_geometry_overview(out_dir, geom=geom_best, kx=kx, ky=ky_best, filename="geometry_best.png")

    y_eq = equilibrium(nl)
    v0 = State.random(jax.random.PRNGKey(0), nl, amplitude=1e-3)
    matvec = linear_matvec(y_eq, params_rbm, geom_best, kx=kx, ky=ky_best)

    print(f"Computing eigenfunction at best point (ŝ={shat_best:g}, ky={ky_best:.3f})…", flush=True)
    ritz = arnoldi_leading_ritz_vector(matvec, v0, m=100, seed=0)
    save_eigenfunction_panel(
        out_dir,
        geom=geom_best,
        state=ritz.vector,
        eigenvalue=ritz.eigenvalue,
        kx=kx,
        ky=ky_best,
        kperp2_min=params_rbm.kperp2_min,
        filename="eigenfunctions_best_rbm_like.png",
    )
    arn = arnoldi_eigs(matvec, v0, m=100, nev=100, seed=0)
    save_eigenvalue_spectrum(
        out_dir,
        eigenvalues=arn.eigenvalues,
        highlight=ritz.eigenvalue,
        filename="spectrum_best_rbm_like.png",
    )

    np.savez(
        out_dir / "results.npz",
        ky=ky,
        shat=np.asarray(shats),
        gamma_rbm=np.stack([s.gamma_eigs for s in scans_rbm], axis=0),
        gamma_inbm=np.stack([s.gamma_eigs for s in scans_inbm], axis=0),
        shat_best=shat_best,
        ky_best=ky_best,
    )
    save_json(
        out_dir / "params.json",
        {
            "curvature0": curvature0,
            "params_rbm_like": params_rbm.__dict__,
            "params_inbm_like": params_inbm.__dict__,
            "shats": shats,
        },
    )
    save_json(
        out_dir / "summary.json",
        {"best_rbm_like": {"shat": shat_best, "ky": ky_best, "gamma": gmax}},
    )

    print(f"\nWrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
