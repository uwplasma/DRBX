from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.lp import solve_lp_fixed_point
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.tokamak import CircularTokamakGeometry
from jaxdrb.models.params import DRBParams


def main() -> None:
    """Gradient-removal SOL width estimate inspired by Halpern et al. (2013).

    Reference:
      - F. Halpern et al., Phys. Plasmas 20, 052306 (2013)

    The paper uses a "gradient removal" saturation rule in which the dominant transport
    scales like (gamma/ky)_max, and obtains a self-consistent SOL scale length Lp by solving:

        (gamma/ky)_max(Lp) = Lp / q

    Here we demonstrate the same *algorithmic* workflow using the v1 electrostatic model.
    """

    out_dir = Path("out_halpern2013_gradient_removal")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))

    geom = CircularTokamakGeometry.make(
        nl=64, shat=0.8, q=3.0, R0=1.0, epsilon=0.18, curvature0=0.18
    )

    # Use a modest ky grid and focus on gamma/ky.
    ky = np.linspace(0.05, 0.8, 16)

    # Baseline params (electrostatic).
    base = DRBParams(
        omega_n=1.0,
        omega_Te=0.0,
        eta=1.0,
        me_hat=5e-3,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )

    # Two normalized pressure-gradient strengths R/Lp used in Halpern fig 8 (we mimic using omega_n).
    grads = [7.15, 12.5]
    scans = []
    for omega_n in grads:
        params = DRBParams(**{**base.__dict__, "omega_n": float(omega_n)})
        scans.append(
            scan_ky(
                params, geom, ky=ky, kx=0.0, arnoldi_m=25, nev=3, do_initial_value=False, seed=0
            )
        )

    # Self-consistent Lp estimate vs an effective drive knob.
    # We vary curvature0 as an "edge-drive" surrogate.
    curvature_grid = np.linspace(0.08, 0.32, 3)
    Lp_out = np.zeros_like(curvature_grid)
    ky_star_out = np.zeros_like(curvature_grid)

    for i, c0 in enumerate(curvature_grid):
        geom_i = CircularTokamakGeometry.make(
            nl=64, shat=0.8, q=3.0, R0=1.0, epsilon=0.18, curvature0=float(c0)
        )
        res = solve_lp_fixed_point(
            base,
            geom_i,
            q=3.0,
            ky=ky,
            Lp0=20.0,
            omega_n_scale=1.0,
            relax=0.6,
            max_iter=8,
            arnoldi_m=20,
            nev=3,
            seed=1,
        )
        Lp_out[i] = res.Lp
        ky_star_out[i] = res.ky_star

    np.savez(
        out_dir / "results.npz",
        ky=ky,
        omega_n=np.asarray(grads),
        gamma_over_ky=np.stack([s.gamma_eigs / ky for s in scans], axis=0),
        curvature0=curvature_grid,
        Lp=Lp_out,
        ky_star=ky_star_out,
    )

    (out_dir / "params.json").write_text(
        json.dumps({"base": base.__dict__, "grads_omega_n": grads}, indent=2, sort_keys=True) + "\n"
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for omega_n, s in zip(grads, scans, strict=True):
        ax.plot(ky, s.gamma_eigs / ky, label=rf"$\omega_n={omega_n:g}$")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma/k_y$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_over_ky_two_gradients.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curvature_grid, Lp_out, "o-")
    ax.set_xlabel(r"curvature drive ($\mathrm{curvature0}$)")
    ax.set_ylabel(r"$L_p$ (fixed-point estimate)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "Lp_vs_curvature0.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curvature_grid, ky_star_out, "o-")
    ax.set_xlabel(r"curvature drive ($\mathrm{curvature0}$)")
    ax.set_ylabel(r"$k_{y,*}$ maximizing $\gamma/k_y$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ky_star_vs_curvature0.png", dpi=160)
    plt.close(fig)

    print(f"Wrote {out_dir / 'gamma_over_ky_two_gradients.png'}")
    print(f"Wrote {out_dir / 'Lp_vs_curvature0.png'}")


if __name__ == "__main__":
    main()
