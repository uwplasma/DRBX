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
    """Ballooning-like branches inspired by Mosetto et al. (2012).

    Reference:
      - A. Mosetto et al., Phys. Plasmas 19, 112103 (2012)

    This script demonstrates curvature-driven modes and the effect of magnetic shear.
    """

    out_dir = Path("out_mosetto2012_ballooning")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))

    nl = 64
    ky = np.linspace(0.05, 1.0, 40)

    params_rbm = DRBParams(
        omega_n=0.8,
        omega_Te=0.0,
        eta=1.0,
        me_hat=2e-3,
        curvature_on=True,
        Dn=0.02,
        DOmega=0.02,
        DTe=0.02,
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
    )

    shats = [0.0, 0.5, 1.0]
    curvature0 = 0.35

    curves = {}
    for shat in shats:
        geom = SlabGeometry.make(nl=nl, shat=shat, curvature0=curvature0)
        curves[f"rbm_shat_{shat:g}"] = scan_ky(
            params_rbm, geom, ky=ky, kx=0.0, do_initial_value=False, seed=0
        )
        curves[f"inbm_shat_{shat:g}"] = scan_ky(
            params_inbm, geom, ky=ky, kx=0.0, do_initial_value=False, seed=1
        )

    # Save results
    np.savez(
        out_dir / "results.npz",
        ky=ky,
        shat=np.asarray(shats),
        gamma_rbm=np.stack([curves[f"rbm_shat_{s:g}"].gamma_eigs for s in shats], axis=0),
        gamma_inbm=np.stack([curves[f"inbm_shat_{s:g}"].gamma_eigs for s in shats], axis=0),
    )
    (out_dir / "params.json").write_text(
        json.dumps(
            {
                "curvature0": curvature0,
                "rbm_like": params_rbm.__dict__,
                "inbm_like": params_inbm.__dict__,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for shat in shats:
        ax.plot(ky, curves[f"rbm_shat_{shat:g}"].gamma_eigs, label=f"RBM-like ŝ={shat:g}")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_ky_rbm_like.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for shat in shats:
        ax.plot(ky, curves[f"inbm_shat_{shat:g}"].gamma_eigs, label=f"InBM-like ŝ={shat:g}")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_ky_inbm_like.png", dpi=160)
    plt.close(fig)

    print(f"Wrote {out_dir / 'gamma_ky_rbm_like.png'}")
    print(f"Wrote {out_dir / 'gamma_ky_inbm_like.png'}")


if __name__ == "__main__":
    main()
