"""
drb_operator_split_diagnostics.py

Purpose
-------
Show conservative/source/dissipative operator splitting diagnostics for DRB variants.

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
from jaxdrb.models.cold_ion_drb import (
    Equilibrium as ColdEquilibrium,
    State as ColdState,
    rhs_nonlinear as cold_rhs,
    rhs_nonlinear_decomposed as cold_split,
)
from jaxdrb.models.em_drb import Equilibrium as EMEquilibrium
from jaxdrb.models.em_drb import State as EMState
from jaxdrb.models.em_drb import rhs_nonlinear as em_rhs
from jaxdrb.models.em_drb import rhs_nonlinear_decomposed as em_split
from jaxdrb.models.hot_ion_drb import Equilibrium as HotEquilibrium
from jaxdrb.models.hot_ion_drb import State as HotState
from jaxdrb.models.hot_ion_drb import rhs_nonlinear as hot_rhs
from jaxdrb.models.hot_ion_drb import rhs_nonlinear_decomposed as hot_split
from jaxdrb.models.params import DRBParams


def _l2_norm(*arrays: np.ndarray) -> float:
    return float(np.sqrt(sum(np.mean(np.abs(np.asarray(a)) ** 2) for a in arrays)))


def _norm_cold(s: ColdState) -> float:
    return _l2_norm(s.n, s.omega, s.vpar_e, s.vpar_i, s.Te)


def _norm_hot(s: HotState) -> float:
    return _l2_norm(s.n, s.omega, s.vpar_e, s.vpar_i, s.Te, s.Ti)


def _norm_em(s: EMState) -> float:
    return _l2_norm(s.n, s.omega, s.psi, s.vpar_i, s.Te)


def _plot_split(
    label: str,
    ky: np.ndarray,
    norm_total: np.ndarray,
    norm_cons: np.ndarray,
    norm_src: np.ndarray,
    norm_diss: np.ndarray,
    reconstruction_rel: np.ndarray,
    cons_fraction: np.ndarray,
    source_fraction: np.ndarray,
    diss_fraction: np.ndarray,
    norm_cons_toggle: np.ndarray,
    norm_src_toggle: np.ndarray,
    norm_diss_toggle: np.ndarray,
    out_path: Path,
) -> None:
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

    fig.suptitle(f"{label} DRB operator splitting diagnostics", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_outputs(
    out_dir: Path,
    label: str,
    ky: np.ndarray,
    norm_total: np.ndarray,
    norm_cons: np.ndarray,
    norm_src: np.ndarray,
    norm_diss: np.ndarray,
    reconstruction_rel: np.ndarray,
    source_fraction: np.ndarray,
    diss_fraction: np.ndarray,
    cons_fraction: np.ndarray,
    norm_cons_toggle: np.ndarray,
    norm_src_toggle: np.ndarray,
    norm_diss_toggle: np.ndarray,
    base: DRBParams,
    nl: int,
) -> None:
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
            "label": label,
            "base_params": base.__dict__,
            "nl": nl,
            "ky_min": float(ky[0]),
            "ky_max": float(ky[-1]),
            "nky": int(ky.size),
            "max_reconstruction_rel": float(np.max(reconstruction_rel)),
        },
    )


def _run_cold_ion(geom: SlabGeometry, ky: np.ndarray, assets_dir: Path, nl: int) -> None:
    out_dir = Path("out/examples/10_verification/drb_operator_split_diagnostics/cold_ion")
    out_dir.mkdir(parents=True, exist_ok=True)

    eq = ColdEquilibrium.constant(nl, n0=1.0, Te0=1.0)
    y = ColdState.random(jax.random.key(1234), nl, amplitude=1e-2)
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

    norm_cons = np.zeros_like(ky)
    norm_src = np.zeros_like(ky)
    norm_diss = np.zeros_like(ky)
    norm_total = np.zeros_like(ky)
    reconstruction_rel = np.zeros_like(ky)
    source_fraction = np.zeros_like(ky)
    diss_fraction = np.zeros_like(ky)
    cons_fraction = np.zeros_like(ky)

    for i, ky_i in enumerate(ky):
        split = cold_split(0.0, y, base, geom, kx=0.0, ky=float(ky_i), eq=eq)
        total = cold_rhs(0.0, y, base, geom, kx=0.0, ky=float(ky_i), eq=eq)
        rsum = split.total()
        n_cons = _norm_cold(split.conservative)
        n_src = _norm_cold(split.source)
        n_diss = _norm_cold(split.dissipative)
        n_tot = _norm_cold(total)
        n_err = _norm_cold(
            ColdState(
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
        norm_cons_toggle[i] = _norm_cold(
            cold_rhs(0.0, y, params_cons_only, geom, kx=0.0, ky=float(ky_i), eq=eq)
        )
        norm_diss_toggle[i] = _norm_cold(
            cold_rhs(0.0, y, params_diss_only, geom, kx=0.0, ky=float(ky_i), eq=eq)
        )
        norm_src_toggle[i] = _norm_cold(
            cold_rhs(0.0, y, params_src_only, geom, kx=0.0, ky=float(ky_i), eq=eq)
        )

    _plot_split(
        "Cold-ion",
        ky,
        norm_total,
        norm_cons,
        norm_src,
        norm_diss,
        reconstruction_rel,
        cons_fraction,
        source_fraction,
        diss_fraction,
        norm_cons_toggle,
        norm_src_toggle,
        norm_diss_toggle,
        out_dir / "drb_operator_split_diagnostics.png",
    )

    if assets_dir.exists():
        _plot_split(
            "Cold-ion",
            ky,
            norm_total,
            norm_cons,
            norm_src,
            norm_diss,
            reconstruction_rel,
            cons_fraction,
            source_fraction,
            diss_fraction,
            norm_cons_toggle,
            norm_src_toggle,
            norm_diss_toggle,
            assets_dir / "drb_operator_split_diagnostics.png",
        )

    _save_outputs(
        out_dir,
        "cold_ion",
        ky,
        norm_total,
        norm_cons,
        norm_src,
        norm_diss,
        reconstruction_rel,
        source_fraction,
        diss_fraction,
        cons_fraction,
        norm_cons_toggle,
        norm_src_toggle,
        norm_diss_toggle,
        base,
        nl,
    )


def _run_hot_ion(geom: SlabGeometry, ky: np.ndarray, assets_dir: Path, nl: int) -> None:
    out_dir = Path("out/examples/10_verification/drb_operator_split_diagnostics/hot_ion")
    out_dir.mkdir(parents=True, exist_ok=True)

    eq = HotEquilibrium.constant(nl, n0=1.0, Te0=1.0)
    y = HotState.random(jax.random.key(2234), nl, amplitude=1e-2)
    base = DRBParams(
        omega_n=1.0,
        omega_Te=0.4,
        omega_Ti=0.3,
        eta=0.8,
        me_hat=0.2,
        tau_i=0.7,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.02,
        DTi=0.02,
        chi_par_Te=0.05,
        chi_par_Ti=0.04,
        nu_par_e=0.02,
        nu_par_i=0.02,
        nu_sink_n=0.01,
        nu_sink_Te=0.01,
        nu_sink_vpar=0.01,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )

    norm_cons = np.zeros_like(ky)
    norm_src = np.zeros_like(ky)
    norm_diss = np.zeros_like(ky)
    norm_total = np.zeros_like(ky)
    reconstruction_rel = np.zeros_like(ky)
    source_fraction = np.zeros_like(ky)
    diss_fraction = np.zeros_like(ky)
    cons_fraction = np.zeros_like(ky)

    for i, ky_i in enumerate(ky):
        split = hot_split(0.0, y, base, geom, kx=0.0, ky=float(ky_i), eq=eq)
        total = hot_rhs(0.0, y, base, geom, kx=0.0, ky=float(ky_i), eq=eq)
        rsum = split.total()
        n_cons = _norm_hot(split.conservative)
        n_src = _norm_hot(split.source)
        n_diss = _norm_hot(split.dissipative)
        n_tot = _norm_hot(total)
        n_err = _norm_hot(
            HotState(
                n=np.asarray(rsum.n - total.n),
                omega=np.asarray(rsum.omega - total.omega),
                vpar_e=np.asarray(rsum.vpar_e - total.vpar_e),
                vpar_i=np.asarray(rsum.vpar_i - total.vpar_i),
                Te=np.asarray(rsum.Te - total.Te),
                Ti=np.asarray(rsum.Ti - total.Ti),
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
        norm_cons_toggle[i] = _norm_hot(
            hot_rhs(0.0, y, params_cons_only, geom, kx=0.0, ky=float(ky_i), eq=eq)
        )
        norm_diss_toggle[i] = _norm_hot(
            hot_rhs(0.0, y, params_diss_only, geom, kx=0.0, ky=float(ky_i), eq=eq)
        )
        norm_src_toggle[i] = _norm_hot(
            hot_rhs(0.0, y, params_src_only, geom, kx=0.0, ky=float(ky_i), eq=eq)
        )

    out_path = out_dir / "drb_operator_split_diagnostics_hot_ion.png"
    _plot_split(
        "Hot-ion",
        ky,
        norm_total,
        norm_cons,
        norm_src,
        norm_diss,
        reconstruction_rel,
        cons_fraction,
        source_fraction,
        diss_fraction,
        norm_cons_toggle,
        norm_src_toggle,
        norm_diss_toggle,
        out_path,
    )

    if assets_dir.exists():
        _plot_split(
            "Hot-ion",
            ky,
            norm_total,
            norm_cons,
            norm_src,
            norm_diss,
            reconstruction_rel,
            cons_fraction,
            source_fraction,
            diss_fraction,
            norm_cons_toggle,
            norm_src_toggle,
            norm_diss_toggle,
            assets_dir / "drb_operator_split_diagnostics_hot_ion.png",
        )

    _save_outputs(
        out_dir,
        "hot_ion",
        ky,
        norm_total,
        norm_cons,
        norm_src,
        norm_diss,
        reconstruction_rel,
        source_fraction,
        diss_fraction,
        cons_fraction,
        norm_cons_toggle,
        norm_src_toggle,
        norm_diss_toggle,
        base,
        nl,
    )


def _run_em(geom: SlabGeometry, ky: np.ndarray, assets_dir: Path, nl: int) -> None:
    out_dir = Path("out/examples/10_verification/drb_operator_split_diagnostics/em")
    out_dir.mkdir(parents=True, exist_ok=True)

    eq = EMEquilibrium.constant(nl, n0=1.0, Te0=1.0)
    y = EMState.random(jax.random.key(3234), nl, amplitude=1e-2)
    base = DRBParams(
        omega_n=1.0,
        omega_Te=0.4,
        eta=0.8,
        me_hat=0.2,
        beta=0.4,
        Dpsi=0.01,
        curvature_on=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.02,
        chi_par_Te=0.05,
        nu_par_i=0.02,
        nu_sink_n=0.01,
        nu_sink_Te=0.01,
        nu_sink_vpar=0.01,
        sheath_bc_on=False,
        sheath_loss_on=False,
        sheath_end_damp_on=False,
    )

    norm_cons = np.zeros_like(ky)
    norm_src = np.zeros_like(ky)
    norm_diss = np.zeros_like(ky)
    norm_total = np.zeros_like(ky)
    reconstruction_rel = np.zeros_like(ky)
    source_fraction = np.zeros_like(ky)
    diss_fraction = np.zeros_like(ky)
    cons_fraction = np.zeros_like(ky)

    for i, ky_i in enumerate(ky):
        split = em_split(0.0, y, base, geom, kx=0.0, ky=float(ky_i), eq=eq)
        total = em_rhs(0.0, y, base, geom, kx=0.0, ky=float(ky_i), eq=eq)
        rsum = split.total()
        n_cons = _norm_em(split.conservative)
        n_src = _norm_em(split.source)
        n_diss = _norm_em(split.dissipative)
        n_tot = _norm_em(total)
        n_err = _norm_em(
            EMState(
                n=np.asarray(rsum.n - total.n),
                omega=np.asarray(rsum.omega - total.omega),
                psi=np.asarray(rsum.psi - total.psi),
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
        norm_cons_toggle[i] = _norm_em(
            em_rhs(0.0, y, params_cons_only, geom, kx=0.0, ky=float(ky_i), eq=eq)
        )
        norm_diss_toggle[i] = _norm_em(
            em_rhs(0.0, y, params_diss_only, geom, kx=0.0, ky=float(ky_i), eq=eq)
        )
        norm_src_toggle[i] = _norm_em(
            em_rhs(0.0, y, params_src_only, geom, kx=0.0, ky=float(ky_i), eq=eq)
        )

    out_path = out_dir / "drb_operator_split_diagnostics_em.png"
    _plot_split(
        "EM",
        ky,
        norm_total,
        norm_cons,
        norm_src,
        norm_diss,
        reconstruction_rel,
        cons_fraction,
        source_fraction,
        diss_fraction,
        norm_cons_toggle,
        norm_src_toggle,
        norm_diss_toggle,
        out_path,
    )

    if assets_dir.exists():
        _plot_split(
            "EM",
            ky,
            norm_total,
            norm_cons,
            norm_src,
            norm_diss,
            reconstruction_rel,
            cons_fraction,
            source_fraction,
            diss_fraction,
            norm_cons_toggle,
            norm_src_toggle,
            norm_diss_toggle,
            assets_dir / "drb_operator_split_diagnostics_em.png",
        )

    _save_outputs(
        out_dir,
        "em",
        ky,
        norm_total,
        norm_cons,
        norm_src,
        norm_diss,
        reconstruction_rel,
        source_fraction,
        diss_fraction,
        cons_fraction,
        norm_cons_toggle,
        norm_src_toggle,
        norm_diss_toggle,
        base,
        nl,
    )


def main() -> None:
    set_mpl_style()

    nl = 64
    geom = SlabGeometry.make(nl=nl, shat=0.7, curvature0=0.22)
    ky = np.linspace(0.08, 0.9, 28)
    assets_dir = ROOT / "docs" / "assets" / "images"

    print(f"Computing split diagnostics on {ky.size} ky points...", flush=True)
    _run_cold_ion(geom, ky, assets_dir, nl)
    _run_hot_ion(geom, ky, assets_dir, nl)
    _run_em(geom, ky, assets_dir, nl)
    print("Done split sweep.", flush=True)


if __name__ == "__main__":
    main()
