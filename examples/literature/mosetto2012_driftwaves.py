from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.models.params import DRBParams


def main() -> None:
    """Drift-wave-like branches inspired by Mosetto et al. (2012).

    Reference:
      - A. Mosetto et al., Phys. Plasmas 19, 112103 (2012)

    This script demonstrates how to:
      - turn off curvature to isolate drift-wave-like dynamics,
      - toggle electron inertia vs resistivity to expose "resistive-like" and "inertial-like" branches,
      - compute gamma(k_y) and the ky maximizing gamma/ky (a common SOL transport proxy).
    """

    out_dir = Path("out_mosetto2012_driftwaves")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))

    nl = 64
    geom = SlabGeometry.make(nl=nl, shat=0.0, curvature0=0.0)

    ky = np.linspace(0.05, 1.2, 32)

    # "Resistive drift wave"-like: small electron inertia, finite resistivity
    params_rdw = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=2e-3,
        curvature_on=False,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )

    # "Inertial drift wave"-like: finite inertia, weak resistivity
    params_idw = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=0.02,
        me_hat=0.5,
        curvature_on=False,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
    )

    scan_rdw = scan_ky(
        params_rdw, geom, ky=ky, kx=0.0, arnoldi_m=30, nev=4, do_initial_value=True, tmax=20.0
    )
    scan_idw = scan_ky(
        params_idw, geom, ky=ky, kx=0.0, arnoldi_m=30, nev=4, do_initial_value=True, tmax=20.0
    )

    # Save results
    np.savez(
        out_dir / "results.npz",
        ky=ky,
        gamma_rdw=scan_rdw.gamma_eigs,
        omega_rdw=scan_rdw.omega_eigs,
        gamma_idw=scan_idw.gamma_eigs,
        omega_idw=scan_idw.omega_eigs,
        gamma_rdw_iv=scan_rdw.gamma_iv,
        gamma_idw_iv=scan_idw.gamma_iv,
    )
    (out_dir / "params.json").write_text(
        json.dumps(
            {"rdw": params_rdw.__dict__, "idw": params_idw.__dict__}, indent=2, sort_keys=True
        )
        + "\n"
    )

    # Diagnostics
    def ky_star(scan):
        ratio = scan.gamma_eigs / scan.ky
        i = int(np.argmax(ratio))
        return float(scan.ky[i]), float(ratio[i]), float(scan.gamma_eigs[i])

    ky_r, r_r, g_r = ky_star(scan_rdw)
    ky_i, r_i, g_i = ky_star(scan_idw)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ky, scan_rdw.gamma_eigs, "o-", label="RDW-like (eig)")
    ax.plot(ky, scan_idw.gamma_eigs, "s--", label="IDW-like (eig)")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_ky.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ky, scan_rdw.gamma_iv, "o-", label="RDW-like (IV)")
    ax.plot(ky, scan_idw.gamma_iv, "s--", label="IDW-like (IV)")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_ky_initial_value.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ky, scan_rdw.gamma_eigs / ky, "o-", label="RDW-like")
    ax.plot(ky, scan_idw.gamma_eigs / ky, "s--", label="IDW-like")
    ax.axvline(ky_r, color="k", alpha=0.25, linestyle="--")
    ax.axvline(ky_i, color="k", alpha=0.25, linestyle="--")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma/k_y$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_over_ky.png", dpi=160)
    plt.close(fig)

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "rdw_like": {"ky_star": ky_r, "gamma_over_ky_star": r_r, "gamma_star": g_r},
                "idw_like": {"ky_star": ky_i, "gamma_over_ky_star": r_i, "gamma_star": g_i},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"Wrote {out_dir / 'gamma_ky.png'}")
    print(f"Wrote {out_dir / 'gamma_over_ky.png'}")
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
