"""
drb_operator_split_diagnostics.py

Purpose
-------
Show conservative/source/dissipative operator splitting diagnostics for the cold-ion DRB branch.

The script computes RHS component norms across a ky scan, verifies exact reconstruction
`RHS_total = RHS_conservative + RHS_source + RHS_dissipative`, and demonstrates toggle behavior.

Run
---
  python examples/10_verification/drb_operator_split_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.plotting import save_json, set_mpl_style
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.models.cold_ion_drb import Equilibrium, State, rhs_nonlinear, rhs_nonlinear_decomposed
from jaxdrb.models.params import DRBParams


def _state_l2_norm(s: State) -> float:
    return float(
        np.sqrt(
            np.mean(np.abs(np.asarray(s.n)) ** 2)
            + np.mean(np.abs(np.asarray(s.omega)) ** 2)
            + np.mean(np.abs(np.asarray(s.vpar_e)) ** 2)
            + np.mean(np.abs(np.asarray(s.vpar_i)) ** 2)
            + np.mean(np.abs(np.asarray(s.Te)) ** 2)
        )
    )


def main() -> None:
    out_dir = Path("out/examples/10_verification/drb_operator_split_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_mpl_style()

    nl = 64
    geom = SlabGeometry.make(nl=nl, shat=0.7, curvature0=0.22)
    eq = Equilibrium.constant(nl, n0=1.0, Te0=1.0)
    y = State.random(jax.random.key(1234), nl, amplitude=1e-2)
    base = DRBParams(
        omega_n=1.0,
        omega_Te=0.4,
        eta=0.8,
        me_hat=0.2,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.02,
        chi_par_Te=0.05,
        nu_par_e=0.02,
        nu_par_i=0.02,
        nu_sink_n=0.01,
        nu_sink_Te=0.01,
        nu_sink_vpar=0.01,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )
    ky = np.linspace(0.08, 0.9, 28)

    norm_cons = np.zeros_like(ky)
    norm_src = np.zeros_like(ky)
    norm_diss = np.zeros_like(ky)
    norm_total = np.zeros_like(ky)
    reconstruction_rel = np.zeros_like(ky)
    source_fraction = np.zeros_like(ky)
    diss_fraction = np.zeros_like(ky)
    cons_fraction = np.zeros_like(ky)

    print(f"Computing split diagnostics on {ky.size} ky points...", flush=True)
    for i, ky_i in enumerate(ky):
        split = rhs_nonlinear_decomposed(0.0, y, base, geom, kx=0.0, ky=float(ky_i), eq=eq)
        total = rhs_nonlinear(0.0, y, base, geom, kx=0.0, ky=float(ky_i), eq=eq)
        rsum = split.total()
        n_cons = _state_l2_norm(split.conservative)
        n_src = _state_l2_norm(split.source)
        n_diss = _state_l2_norm(split.dissipative)
        n_tot = _state_l2_norm(total)
        n_err = _state_l2_norm(
            State(
                n=np.asarray(rsum.n - total.n),
                omega=np.asarray(rsum.omega - total.omega),
                vpar_e=np.asarray(rsum.vpar_e - total.vpar_e),
                vpar_i=np.asarray(rsum.vpar_i - total.vpar_i),
                Te=np.asarray(rsum.Te - total.Te),
            )
        )
        norm_cons[i] = n_cons
        norm_src[i] = n_src
        norm_diss[i] = n_diss
        norm_total[i] = n_tot
        reconstruction_rel[i] = n_err / max(n_tot, 1e-30)
        denom = max(n_cons + n_src + n_diss, 1e-30)
        cons_fraction[i] = n_cons / denom
        source_fraction[i] = n_src / denom
        diss_fraction[i] = n_diss / denom
    print("Done split sweep.", flush=True)

    params_cons_only = DRBParams(
        **{
            **base.__dict__,
            "operator_split_on": True,
            "operator_conservative_on": True,
            "operator_source_on": False,
            "operator_dissipative_on": False,
        }
    )
    params_diss_only = DRBParams(
        **{
            **base.__dict__,
            "operator_split_on": True,
            "operator_conservative_on": False,
            "operator_source_on": False,
            "operator_dissipative_on": True,
        }
    )
    params_src_only = DRBParams(
        **{
            **base.__dict__,
            "operator_split_on": True,
            "operator_conservative_on": False,
            "operator_source_on": True,
            "operator_dissipative_on": False,
        }
    )
    norm_cons_toggle = np.zeros_like(ky)
    norm_diss_toggle = np.zeros_like(ky)
    norm_src_toggle = np.zeros_like(ky)
    for i, ky_i in enumerate(ky):
        norm_cons_toggle[i] = _state_l2_norm(
            rhs_nonlinear(0.0, y, params_cons_only, geom, kx=0.0, ky=float(ky_i), eq=eq)
        )
        norm_diss_toggle[i] = _state_l2_norm(
            rhs_nonlinear(0.0, y, params_diss_only, geom, kx=0.0, ky=float(ky_i), eq=eq)
        )
        norm_src_toggle[i] = _state_l2_norm(
            rhs_nonlinear(0.0, y, params_src_only, geom, kx=0.0, ky=float(ky_i), eq=eq)
        )

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    ax = axs.ravel()

    ax[0].plot(ky, norm_total, lw=2.0, color="k", label="total")
    ax[0].plot(ky, norm_cons, lw=1.8, label="conservative")
    ax[0].plot(ky, norm_src, lw=1.8, label="source")
    ax[0].plot(ky, norm_diss, lw=1.8, label="dissipative")
    ax[0].set_xlabel(r"$k_y$")
    ax[0].set_ylabel(r"$\|RHS\|_{L^2}$")
    ax[0].set_title("RHS component norms")
    ax[0].grid(alpha=0.25)
    ax[0].legend(fontsize=8, ncol=2)

    ax[1].plot(ky, cons_fraction, lw=2.0, label="conservative share")
    ax[1].plot(ky, source_fraction, lw=2.0, label="source share")
    ax[1].plot(ky, diss_fraction, lw=2.0, label="dissipative share")
    ax[1].set_xlabel(r"$k_y$")
    ax[1].set_ylabel("fraction of split norm sum")
    ax[1].set_title("Relative contribution by split component")
    ax[1].grid(alpha=0.25)
    ax[1].legend(fontsize=8)

    ax[2].semilogy(ky, reconstruction_rel + 1e-30, lw=2.0, color="#1f77b4")
    ax[2].set_xlabel(r"$k_y$")
    ax[2].set_ylabel(r"$\|RHS_{sum} - RHS\|/\|RHS\|$")
    ax[2].set_title("Split reconstruction residual")
    ax[2].grid(alpha=0.25)

    ax[3].plot(ky, norm_cons_toggle, lw=2.0, label="toggle: conservative only")
    ax[3].plot(ky, norm_src_toggle, lw=2.0, label="toggle: source only")
    ax[3].plot(ky, norm_diss_toggle, lw=2.0, label="toggle: dissipative only")
    ax[3].set_xlabel(r"$k_y$")
    ax[3].set_ylabel(r"$\|RHS\|_{L^2}$")
    ax[3].set_title("RHS norms with split toggles")
    ax[3].grid(alpha=0.25)
    ax[3].legend(fontsize=8)

    fig.suptitle("Cold-ion DRB operator splitting diagnostics", fontsize=13)
    fig.savefig(out_dir / "drb_operator_split_diagnostics.png", dpi=220)
    plt.close(fig)

    np.savez(
        out_dir / "results.npz",
        ky=ky,
        norm_total=norm_total,
        norm_conservative=norm_cons,
        norm_source=norm_src,
        norm_dissipative=norm_diss,
        reconstruction_rel=reconstruction_rel,
        source_fraction=source_fraction,
        diss_fraction=diss_fraction,
        conservative_fraction=cons_fraction,
        norm_toggle_conservative_only=norm_cons_toggle,
        norm_toggle_source_only=norm_src_toggle,
        norm_toggle_dissipative_only=norm_diss_toggle,
    )
    save_json(
        out_dir / "params.json",
        {
            "base_params": base.__dict__,
            "nl": nl,
            "ky_min": float(ky[0]),
            "ky_max": float(ky[-1]),
            "nky": int(ky.size),
            "max_reconstruction_rel": float(np.max(reconstruction_rel)),
        },
    )
    print(
        f"max reconstruction relative error = {float(np.max(reconstruction_rel)):.3e}", flush=True
    )
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
