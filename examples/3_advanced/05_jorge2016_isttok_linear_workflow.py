from __future__ import annotations

"""
05_jorge2016_isttok_linear_workflow.py

Purpose
-------
Demonstrate an ISTTOK-inspired linear workflow based on:

  R. Jorge et al., Phys. Plasmas 23, 102511 (2016). DOI: 10.1063/1.4964783

The original paper is primarily nonlinear and includes SOL-specific physics (sources and sheath
boundary conditions). `jaxdrb` is electrostatic and uses periodic field-line boundary conditions,
so this script is a *qualitative* workflow replica rather than a quantitative match.

What this script does:
  1) Build a circular tokamak geometry with parameters close to the ISTTOK values quoted in the
     paper (q≈8, epsilon≈a/R≈0.18, no shear).
  2) Run ky scans for two parameter orderings that expose resistive-like vs inertial-like branches.
  3) Plot:
       - gamma(ky), omega(ky),
       - transport proxy max(gamma,0)/ky,
       - an Lp proxy curve: Lp(ky) = q * max(gamma,0)/ky, highlighting ky*.
  4) Plot a representative eigenfunction panel at ky*.

Run
---
  python examples/3_advanced/05_jorge2016_isttok_linear_workflow.py

Outputs
-------
Written to `out/3_advanced/jorge2016_isttok_linear_workflow/`.
"""

import os
import sys
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import (  # noqa: E402
    save_eigenfunction_panel,
    save_eigenvalue_spectrum,
    save_geometry_overview,
    save_json,
    save_scan_panels,
    set_mpl_style,
)
from jaxdrb.analysis.scan import scan_ky  # noqa: E402
from jaxdrb.geometry.tokamak import CircularTokamakGeometry  # noqa: E402
from jaxdrb.linear.arnoldi import arnoldi_eigs, arnoldi_leading_ritz_vector  # noqa: E402
from jaxdrb.linear.matvec import linear_matvec  # noqa: E402
from jaxdrb.models.cold_ion_drb import State, equilibrium  # noqa: E402
from jaxdrb.models.params import DRBParams  # noqa: E402


def main() -> None:
    out_dir = Path("out/3_advanced/jorge2016_isttok_linear_workflow")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    # --- ISTTOK-like parameters (from Jorge et al. 2016, large-aspect-ratio discussion) ---
    # Paper quotes (in rho_s0 units): R ~ 504 rho_s0, a ~ 93 rho_s0, q ~ 8, no shear.
    q = 8.0
    epsilon = 93.0 / 504.0  # ~ 0.1845

    nl = 96
    geom = CircularTokamakGeometry.make(
        nl=nl,
        length=float(2 * np.pi),
        shat=0.0,
        q=q,
        R0=1.0,
        epsilon=float(epsilon),
        curvature0=float(epsilon),
    )

    # ky scan range: keep modest for a laptop-friendly run.
    ky = np.linspace(0.05, 1.0, 28)
    kx = 0.0

    # Two branch-like parameter sets (qualitative).
    # - "resistive-like": small electron inertia, larger eta
    # - "inertial-like": larger inertia, smaller eta
    base = dict(
        omega_n=0.9,
        omega_Te=0.0,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        kperp2_min=1e-6,
    )
    params_res = DRBParams(**{**base, "eta": 1.0, "me_hat": 2e-3})
    params_in = DRBParams(**{**base, "eta": 0.05, "me_hat": 0.5})

    print("Scanning ky (ISTTOK-like geometry): resistive-like case…", flush=True)
    scan_res = scan_ky(
        params_res,
        geom,
        ky=ky,
        kx=kx,
        arnoldi_m=35,
        arnoldi_tol=2e-3,
        nev=8,
        do_initial_value=False,
        verbose=True,
        print_every=2,
        seed=0,
    )

    print("\nScanning ky (ISTTOK-like geometry): inertial-like case…", flush=True)
    scan_in = scan_ky(
        params_in,
        geom,
        ky=ky,
        kx=kx,
        arnoldi_m=35,
        arnoldi_tol=2e-3,
        nev=8,
        do_initial_value=False,
        verbose=True,
        print_every=2,
        seed=1,
    )

    def ky_star_and_lp(scan):
        ratio = np.maximum(scan.gamma_eigs, 0.0) / scan.ky
        i = int(np.argmax(ratio))
        ky_star = float(scan.ky[i])
        lp_curve = q * ratio
        return ky_star, lp_curve

    ky_star_res, Lp_curve_res = ky_star_and_lp(scan_res)
    ky_star_in, Lp_curve_in = ky_star_and_lp(scan_in)

    save_scan_panels(
        out_dir,
        ky=scan_res.ky,
        gamma=scan_res.gamma_eigs,
        omega=scan_res.omega_eigs,
        title="ISTTOK-like (Jorge 2016): resistive-like branch",
        filename="scan_panel_resistive.png",
    )
    save_scan_panels(
        out_dir,
        ky=scan_in.ky,
        gamma=scan_in.gamma_eigs,
        omega=scan_in.omega_eigs,
        title="ISTTOK-like (Jorge 2016): inertial-like branch",
        filename="scan_panel_inertial.png",
    )
    save_geometry_overview(
        out_dir, geom=geom, kx=kx, ky=ky_star_res, filename="geometry_overview.png"
    )

    # Lp proxy curves (Lp(ky) = q * max(gamma,0)/ky).
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(ky, Lp_curve_res, "o-", label="resistive-like")
    ax.plot(ky, Lp_curve_in, "s--", label="inertial-like")
    ax.axvline(ky_star_res, color="k", alpha=0.25, linestyle="--")
    ax.axvline(ky_star_in, color="k", alpha=0.25, linestyle=":")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$L_p(k_y)=q\,\max(\gamma,0)/k_y$")
    ax.set_title(r"ISTTOK-like $L_p$ proxy curve and maximizing $k_{y,*}$")
    ax.legend()
    fig.savefig(out_dir / "Lp_proxy_vs_ky.png", dpi=220)
    plt.close(fig)

    # Eigenfunction at ky* for the resistive-like case (as an example mode structure plot).
    y_eq = equilibrium(nl)
    v0 = State.random(jax.random.PRNGKey(0), nl, amplitude=1e-3)
    matvec = linear_matvec(y_eq, params_res, geom, kx=kx, ky=ky_star_res)

    print("\nComputing leading Ritz vector at ky* (resistive-like)…", flush=True)
    ritz = arnoldi_leading_ritz_vector(matvec, v0, m=120, seed=0)
    save_eigenfunction_panel(
        out_dir,
        geom=geom,
        state=ritz.vector,
        eigenvalue=ritz.eigenvalue,
        kx=kx,
        ky=ky_star_res,
        kperp2_min=params_res.kperp2_min,
        filename="eigenfunctions_resistive_ky_star.png",
    )
    arn = arnoldi_eigs(matvec, v0, m=120, nev=120, seed=0)
    save_eigenvalue_spectrum(
        out_dir,
        eigenvalues=arn.eigenvalues,
        highlight=ritz.eigenvalue,
        filename="spectrum_resistive_ky_star.png",
    )

    np.savez(
        out_dir / "results.npz",
        ky=ky,
        gamma_res=scan_res.gamma_eigs,
        omega_res=scan_res.omega_eigs,
        gamma_in=scan_in.gamma_eigs,
        omega_in=scan_in.omega_eigs,
        ky_star_res=ky_star_res,
        ky_star_in=ky_star_in,
        Lp_proxy_res=Lp_curve_res,
        Lp_proxy_in=Lp_curve_in,
        q=q,
        epsilon=epsilon,
    )
    save_json(
        out_dir / "params.json",
        {
            "reference": {
                "paper": "Jorge et al. (2016) Phys. Plasmas 23, 102511",
                "doi": "10.1063/1.4964783",
            },
            "geom": {
                "type": "circular_tokamak",
                "nl": nl,
                "q": q,
                "shat": 0.0,
                "epsilon": float(epsilon),
                "curvature0": float(epsilon),
            },
            "kx": kx,
            "ky": {"min": float(ky.min()), "max": float(ky.max()), "n": int(ky.size)},
            "params_resistive_like": params_res.__dict__,
            "params_inertial_like": params_in.__dict__,
            "note": "Electrostatic + periodic-in-l model; results are qualitative workflow demonstrations.",
        },
    )
    save_json(
        out_dir / "summary.json",
        {
            "ky_star": {"resistive_like": ky_star_res, "inertial_like": ky_star_in},
            "Lp_proxy_star": {
                "resistive_like": float(np.max(Lp_curve_res)),
                "inertial_like": float(np.max(Lp_curve_in)),
            },
        },
    )

    print(f"\nWrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
