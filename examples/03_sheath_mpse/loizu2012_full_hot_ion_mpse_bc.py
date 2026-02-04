"""
loizu2012_full_hot_ion_mpse_bc.py

Purpose
-------
Demonstrate the *full* (linearized, model-aligned) MPSE boundary condition set (Loizu 2012)
for the **hot-ion** drift-reduced Braginskii variant:

  J. Loizu et al., "Boundary conditions for plasma fluid models at the magnetic presheath entrance",
  Phys. Plasmas 19, 122307 (2012).

In addition to the electron-temperature entrance constraint (Loizu Eq. (23)),
`jaxdrb` enforces a matching hot-ion endpoint constraint:

  ∂_|| T_i = 0  (Neumann at the MPSE nodes)

This script runs a ky scan and produces a representative eigenfunction plot to highlight that
the leading mode satisfies the intended endpoint behavior.

Run
---
  python examples/03_sheath_mpse/loizu2012_full_hot_ion_mpse_bc.py

Outputs
-------
Written to `out/examples/03_sheath_mpse/loizu2012_full_hot_ion_mpse_bc/`.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, save_scan_panels, set_mpl_style
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.linear.arnoldi import arnoldi_leading_ritz_vector
from jaxdrb.linear.matvec import linear_matvec_from_rhs
from jaxdrb.models.cold_ion_drb import Equilibrium, phi_from_omega
from jaxdrb.models.params import DRBParams
from jaxdrb.models.registry import get_model
import jax


def main() -> None:
    out_dir = Path("out/examples/03_sheath_mpse/loizu2012_full_hot_ion_mpse_bc")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    fast = os.environ.get("JAXDRB_FAST", "1") != "0"
    if fast:
        print("JAXDRB_FAST=1: smaller grids for a quick run.", flush=True)
    else:
        print("JAXDRB_FAST=0: larger grids (slower).", flush=True)

    nl = 65 if fast else 129
    geom = OpenSlabGeometry.make(nl=nl, length=float(2 * np.pi), shat=0.2, curvature0=0.15)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)

    ky = np.linspace(0.06, 0.8, 10 if fast else 28)
    kx = 0.0

    model = get_model("hot-ion-es")

    base = DRBParams(
        omega_n=0.2,
        omega_Te=0.8,
        omega_Ti=0.2,
        tau_i=1.0,
        eta=1.0,
        me_hat=0.05,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        DTi=0.01,
        sheath_bc_on=True,
        sheath_bc_nu_factor=1.0,
        sheath_cos2=1.0,
    )

    cases = [
        ("simple", DRBParams(**{**base.__dict__, "sheath_bc_model": 0})),
        ("loizu2012_full", DRBParams(**{**base.__dict__, "sheath_bc_model": 1})),
    ]

    results = []
    for name, p in cases:
        print(f"\n=== Hot-ion MPSE BC mode: {name} ===", flush=True)
        s = scan_ky(
            p,
            geom,
            ky=ky,
            kx=kx,
            model=model,
            eq=eq,
            arnoldi_m=30 if fast else 50,
            arnoldi_tol=3e-3 if fast else 1e-3,
            nev=6,
            do_initial_value=False,
            verbose=True,
            print_every=3,
            seed=0,
        )
        results.append((name, s))
        save_scan_panels(
            out_dir,
            ky=s.ky,
            gamma=s.gamma_eigs,
            omega=s.omega_eigs,
            title=f"Hot-ion MPSE BC mode: {name}",
            filename=f"scan_panel_{name}.png",
        )

    # Representative eigenfunction plot for the full Loizu2012 case at the most unstable ky.
    name_full, scan_full = results[1]
    i_star = int(np.argmax(scan_full.gamma_eigs))
    ky_star = float(scan_full.ky[i_star])
    print(f"\nRepresentative eigenfunction: ky* = {ky_star:.4f}", flush=True)

    # Re-run Arnoldi to capture the leading Ritz vector (eigenfunction).
    rhs = model.rhs
    y_eq = model.equilibrium(nl)
    matvec = linear_matvec_from_rhs(
        rhs, y_eq, cases[1][1], geom, kx=kx, ky=ky_star, rhs_kwargs={"eq": eq}
    )
    v0 = model.random_state(jax.random.PRNGKey(0), nl, amplitude=1e-3)
    ritz = arnoldi_leading_ritz_vector(matvec, v0, m=40 if fast else 70, nev=6, seed=0)
    y = ritz.vector

    k2 = np.asarray(geom.kperp2(kx, ky_star))
    phi = phi_from_omega(
        y.omega,
        k2,
        kperp2_min=cases[1][1].kperp2_min,
        boussinesq=cases[1][1].boussinesq,
        n0=eq.n0,
        n0_min=cases[1][1].n0_min,
    )

    import matplotlib.pyplot as plt

    l = np.asarray(geom.l)
    fig, ax = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)

    def plot_line(a, arr, title):
        a.plot(l, np.real(np.asarray(arr)), lw=2.0)
        a.set_title(title)
        a.set_xlabel(r"$l$")
        a.grid(True, alpha=0.3)

    plot_line(ax[0, 0], phi, r"$\Re(\phi)$ (from $\omega$)")
    plot_line(ax[0, 1], y.Te, r"$\Re(T_e)$")
    plot_line(ax[1, 0], y.Ti, r"$\Re(T_i)$")
    plot_line(ax[1, 1], y.vpar_i, r"$\Re(v_{\parallel i})$")

    # Endpoint residuals for the Neumann-like temperature constraints.
    Ti_res = float(
        np.abs(np.asarray(y.Ti)[0] - np.asarray(y.Ti)[1])
        + np.abs(np.asarray(y.Ti)[-1] - np.asarray(y.Ti)[-2])
    )
    Te_res = float(
        np.abs(np.asarray(y.Te)[0] - np.asarray(y.Te)[1])
        + np.abs(np.asarray(y.Te)[-1] - np.asarray(y.Te)[-2])
    )
    fig.suptitle(
        f"Hot-ion Loizu2012 MPSE full-set eigenfunction (ky={ky_star:.3f})\n"
        f"endpoint residuals: |ΔTe|={Te_res:.2e}, |ΔTi|={Ti_res:.2e}",
        fontsize=12,
    )
    fig.savefig(out_dir / "eigenfunctions_hot_ion_loizu2012_full.png", dpi=220)
    plt.close(fig)

    # Save summary outputs.
    np.savez(
        out_dir / "results.npz",
        ky=ky,
        gamma_simple=results[0][1].gamma_eigs,
        omega_simple=results[0][1].omega_eigs,
        gamma_loizu2012=results[1][1].gamma_eigs,
        omega_loizu2012=results[1][1].omega_eigs,
        ky_star=ky_star,
        eigenvalue=np.asarray(ritz.eigenvalue),
    )
    save_json(out_dir / "params.json", {"base": base.__dict__})
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
