"""Shear-Alfvén dispersion verification (Zhu et al. 2018 / GDB).

This reproduces the *verification pattern* used in the GDB code paper:

  B. Zhu et al., Computer Physics Communications 232 (2018) 46–58
  DOI: 10.1016/j.cpc.2018.06.002

We use the simplified linear SAW model from their Section 4.2 and compare
the analytical phase speed v_SAW(Te) against eigenvalues of the 3x3 linear
matrix (single ky,k|| mode).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from jaxdrb.analysis.plotting import set_mpl_style
from jaxdrb.verification.gdb2018 import saw_linear_matrix, saw_phase_speed_sq


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    set_mpl_style()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="out_saw_gdb2018")
    p.add_argument("--ky", type=float, default=2.0)
    p.add_argument("--kpar", type=float, default=0.3)
    p.add_argument("--Te-min", type=float, default=0.5)
    p.add_argument("--Te-max", type=float, default=3.0)
    p.add_argument("--nTe", type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parameters taken from the caption context of Fig. 6 in Zhu et al. (2018).
    n0 = 1.0
    alpha_m = 2e-4
    alpha_d = 2e-3
    de2 = 6.4e-7
    eps_R = 0.6

    Tes = np.linspace(args.Te_min, args.Te_max, int(args.nTe))
    v_analytic = []
    v_matrix = []

    for Te in Tes:
        v2 = saw_phase_speed_sq(
            Te=float(Te),
            ky=float(args.ky),
            n0=n0,
            alpha_m=alpha_m,
            alpha_d=alpha_d,
            eps_R=eps_R,
            de2=de2,
        )
        v_analytic.append(np.sqrt(v2))

        M = np.asarray(
            saw_linear_matrix(
                kpar=float(args.kpar),
                Te=float(Te),
                ky=float(args.ky),
                n0=n0,
                alpha_m=alpha_m,
                alpha_d=alpha_d,
                eps_R=eps_R,
                de2=de2,
            )
        )
        evals = np.linalg.eigvals(M)
        w = np.max(np.abs(np.imag(evals)))
        v_matrix.append(w / abs(float(args.kpar)))

    v_analytic = np.asarray(v_analytic)
    v_matrix = np.asarray(v_matrix)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    ax.plot(Tes, v_analytic, lw=2.5, label="analytic (Eq. 68)")
    ax.plot(Tes, v_matrix, "o", ms=6, label="matrix eigenvalues")
    ax.set_xlabel(r"$T_e$")
    ax.set_ylabel(r"$v_{\mathrm{SAW}} = |\omega|/|k_\parallel|$")
    ax.set_title("SAW dispersion verification (Zhu et al. 2018 / GDB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "saw_speed_vs_Te.png", dpi=220)
    plt.close(fig)

    # Save raw data.
    np.savez(out_dir / "saw_speed_vs_Te.npz", Te=Tes, v_analytic=v_analytic, v_matrix=v_matrix)
    print(f"[saw-gdb2018] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
