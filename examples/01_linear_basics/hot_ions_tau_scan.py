"""
hot_ions_tau_scan.py

Purpose
-------
Demonstrate the hot-ion electrostatic model variant by comparing spectra for different ion
temperature ratios tau_i = Ti0/Te0.

This is a qualitative example meant to showcase:

- adding an ion temperature field (Ti),
- including ion-pressure effects in parallel dynamics and curvature forcing,
- comparing instability trends as tau_i is varied.

Run
---
  python examples/01_linear_basics/hot_ions_tau_scan.py

Outputs
-------
Written to `out/examples/01_linear_basics/hot_ions_tau_scan/`.
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
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.linear.arnoldi import arnoldi_eigs, arnoldi_leading_ritz_vector
from jaxdrb.linear.matvec import linear_matvec_from_rhs
from jaxdrb.models.params import DRBParams
from jaxdrb.models.registry import get_model


def main() -> None:
    out_dir = Path("out/examples/01_linear_basics/hot_ions_tau_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    model = get_model("hot-ion-es")
    print(f"Using model: {model.name}", flush=True)

    # Geometry: slab with curvature on for ballooning-like drive.
    nl = 64
    geom = SlabGeometry.make(nl=nl, length=float(2 * np.pi), shat=0.5, curvature0=0.2)
    kx = 0.0
    ky = np.linspace(0.05, 1.2, 30)

    taus = [0.0, 1.0]
    scans = []
    for tau_i in taus:
        params = DRBParams(
            omega_n=0.8,
            omega_Te=0.0,
            omega_Ti=0.8,
            tau_i=tau_i,
            eta=1.0,
            me_hat=0.05,
            curvature_on=True,
            Dn=0.01,
            DOmega=0.01,
            DTe=0.01,
            DTi=0.01,
        )
        print(f"Running ky scan for tau_i={tau_i:g}…", flush=True)
        s = scan_ky(
            params,
            geom,
            ky=ky,
            kx=kx,
            nl=nl,
            model=model,
            arnoldi_m=40,
            arnoldi_tol=1e-3,
            nev=6,
            do_initial_value=True,
            tmax=18.0,
            dt0=0.02,
            nsave=140,
            verbose=True,
            print_every=1,
            seed=0,
        )
        scans.append((tau_i, params, s))
        save_scan_panels(
            out_dir,
            ky=s.ky,
            gamma=s.gamma_eigs,
            omega=s.omega_eigs,
            gamma_iv=s.gamma_iv,
            title=rf"Hot-ion ES model: $\tau_i={tau_i:g}$ (slab, curvature on)",
            filename=f"scan_panel_tau_{tau_i:g}.png".replace(".", "p"),
        )

    # Geometry overview at ky* from the tau_i=1 scan
    tau_ref, params_ref, scan_ref = scans[-1]
    ratio = np.maximum(scan_ref.gamma_eigs, 0.0) / scan_ref.ky
    i_star = int(np.argmax(ratio))
    ky_star = float(scan_ref.ky[i_star])
    save_geometry_overview(out_dir, geom=geom, kx=kx, ky=ky_star)

    # Eigenfunction + spectrum at ky* for tau_ref
    print(f"Computing eigenfunctions at ky*={ky_star:.4f} for tau_i={tau_ref:g}…", flush=True)
    y_eq = model.equilibrium(nl)
    rhs_kwargs = {}
    if model.default_eq is not None:
        rhs_kwargs["eq"] = model.default_eq(nl)
    v0 = model.random_state(jax.random.PRNGKey(0), nl, amplitude=1e-3)
    matvec = linear_matvec_from_rhs(
        model.rhs, y_eq, params_ref, geom, kx=kx, ky=ky_star, rhs_kwargs=rhs_kwargs
    )
    ritz = arnoldi_leading_ritz_vector(matvec, v0, m=80, seed=0)
    arn = arnoldi_eigs(matvec, v0, m=80, nev=80, seed=0)
    save_eigenfunction_panel(
        out_dir,
        geom=geom,
        state=ritz.vector,
        eigenvalue=ritz.eigenvalue,
        kx=kx,
        ky=ky_star,
        filename="eigenfunctions_tau_ref.png",
    )
    save_eigenvalue_spectrum(out_dir, eigenvalues=arn.eigenvalues, highlight=ritz.eigenvalue)

    # Overlay plot
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    for tau_i, _params, s in scans:
        axs[0].plot(s.ky, s.gamma_eigs, "o-", label=rf"$\tau_i={tau_i:g}$")
        axs[1].plot(s.ky, np.maximum(s.gamma_eigs, 0.0) / s.ky, "o-", label=rf"$\tau_i={tau_i:g}$")
    axs[0].set_xlabel(r"$k_y$")
    axs[0].set_ylabel(r"$\gamma$")
    axs[0].set_title("Growth rate")
    axs[1].set_xlabel(r"$k_y$")
    axs[1].set_ylabel(r"$\max(\gamma,0)/k_y$")
    axs[1].set_title("Transport proxy")
    for ax in axs:
        ax.legend(frameon=False)
    fig.savefig(out_dir / "tau_overlay.png", dpi=220)
    plt.close(fig)

    np.savez(
        out_dir / "results.npz",
        ky=scan_ref.ky,
        taus=np.asarray(taus),
        gamma_eigs=np.stack([s.gamma_eigs for _, _, s in scans], axis=0),
        omega_eigs=np.stack([s.omega_eigs for _, _, s in scans], axis=0),
        gamma_iv=np.stack([s.gamma_iv for _, _, s in scans], axis=0),
        ky_star=ky_star,
        tau_ref=tau_ref,
    )
    save_json(
        out_dir / "params.json",
        {
            "model": model.name,
            "geom": {"type": "slab", "nl": nl, "shat": 0.5, "curvature0": 0.2},
            "kx": kx,
            "ky": {"min": float(ky.min()), "max": float(ky.max()), "n": int(ky.size)},
            "taus": taus,
        },
    )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
