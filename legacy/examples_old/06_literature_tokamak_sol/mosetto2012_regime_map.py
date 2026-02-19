"""
mosetto2012_regime_map.py

Purpose
-------
Build a Mosetto-style instability regime map with a **quantitative 4-regime calibration**
(InDW, RDW, InBM, RBM), and optionally compare it against the expensive solver-ablation
classification used in earlier workflow replicas.

Calibration model
-----------------
Following the transition discussion in Mosetto et al. (2012, Sec. V), we use:

- RDW/RBM threshold (s^≈0):
    R/Ln = 2(1+g) / [0.085 (1+1.71 g)]^2
- InDW/InBM threshold (s^≈0):
    R/Ln = 2(1+g) / [0.17 (1+1.71 g)]^2
- RDW/InDW threshold proxy d_crit(s^):
    fit constrained by d_crit(0)=3.55 and d_crit(5)=1.12.

`jaxdrb` knobs are mapped to these paper proxies through explicit calibration factors
(`MosettoCalibration`), documented in `src/jaxdrb/analysis/mosetto_regime.py`.

Run
---
  python examples/06_literature_tokamak_sol/mosetto2012_regime_map.py

Optional slower comparison against solver-ablation labels:
  python examples/06_literature_tokamak_sol/mosetto2012_regime_map.py --classifier both

Outputs
-------
Written to `out/examples/06_literature_tokamak_sol/mosetto2012_regime_map/`:

  - `mosetto2012_fig6a_like.png`
  - `results.npz`
  - `params.json`
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.analysis.mosetto_regime import (
    MosettoCalibration,
    classify_grid,
    threshold_d_resistive_inertial,
    threshold_rln_indw_inbm,
    threshold_rln_rdw_rbm,
)
from jaxdrb.analysis.plotting import save_json, set_mpl_style
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.tokamak import CircularTokamakGeometry
from jaxdrb.models.params import DRBParams


def _gamma_max(scan) -> float:
    return float(np.max(scan.gamma_eigs))


def _ablation_label_grid(
    *,
    eta_grid: np.ndarray,
    omega_n_grid: np.ndarray,
    geom,
    ky: np.ndarray,
    base: DRBParams,
    fast: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute the solver-ablation label map (slow, optional)."""
    frac_threshold = 0.5
    tiny = 1e-12

    gamma_full = np.zeros((eta_grid.size, omega_n_grid.size))
    gamma_no_curv = np.zeros_like(gamma_full)
    gamma_no_inertia = np.zeros_like(gamma_full)
    label_idx = np.full_like(gamma_full, fill_value=-1, dtype=int)

    print(
        f"[ablation] scanning {eta_grid.size * omega_n_grid.size} points "
        "(3 Arnoldi ky-scans per point)",
        flush=True,
    )

    for i, eta in enumerate(eta_grid):
        for j, omega_n in enumerate(omega_n_grid):
            p = DRBParams(**{**base.__dict__, "eta": float(eta), "omega_n": float(omega_n)})

            arn_m = 14 if fast else 40
            arn_tol = 7e-3 if fast else 2e-3
            arn_nev = 2 if fast else 6

            kwargs = dict(
                ky=ky,
                kx=0.0,
                arnoldi_m=arn_m,
                arnoldi_tol=arn_tol,
                arnoldi_max_m=5 * int(getattr(geom, "l").size),
                nev=arn_nev,
                do_initial_value=False,
                verbose=False,
                seed=0,
            )

            g_full = _gamma_max(scan_ky(p, geom, **kwargs))
            g_noc = _gamma_max(
                scan_ky(DRBParams(**{**p.__dict__, "curvature_on": False}), geom, **kwargs)
            )
            g_noi = _gamma_max(scan_ky(DRBParams(**{**p.__dict__, "me_hat": 0.0}), geom, **kwargs))

            gamma_full[i, j] = g_full
            gamma_no_curv[i, j] = g_noc
            gamma_no_inertia[i, j] = g_noi

            if g_full <= tiny:
                label_idx[i, j] = -1
                continue

            dw_like = g_noc >= frac_threshold * g_full
            resistive_like = g_noi >= frac_threshold * g_full

            if dw_like and (not resistive_like):
                label_idx[i, j] = 0  # InDW
            elif dw_like and resistive_like:
                label_idx[i, j] = 1  # RDW
            elif (not dw_like) and (not resistive_like):
                label_idx[i, j] = 2  # InBM
            else:
                label_idx[i, j] = 3  # RBM

        print(f"[ablation] [{i + 1:02d}/{eta_grid.size}] eta={eta:9.3e}", flush=True)

    extras = {
        "gamma_full": gamma_full,
        "gamma_no_curv": gamma_no_curv,
        "gamma_no_inertia": gamma_no_inertia,
        "frac_threshold": np.asarray(frac_threshold),
    }
    return label_idx, extras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier",
        choices=["calibrated", "ablation", "both"],
        default="calibrated",
        help=(
            "Regime classifier to plot. 'calibrated' is the default quantitative 4-regime map; "
            "'both' overlays calibrated + ablation comparison (slow)."
        ),
    )
    parser.add_argument(
        "--rln-per-omega-n",
        type=float,
        default=40.0,
        help="Calibration factor mapping omega_n -> R/Ln proxy.",
    )
    parser.add_argument(
        "--d-scale",
        type=float,
        default=1.0,
        help="Calibration factor in d proxy: d = d_scale * eta/me_hat.",
    )
    parser.add_argument(
        "--gamma-ratio",
        type=float,
        default=1.0,
        help="g=Ln/LT ratio entering Mosetto threshold formulas.",
    )
    args = parser.parse_args()

    out_dir = Path("out/examples/06_literature_tokamak_sol/mosetto2012_regime_map")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    fast = os.environ.get("JAXDRB_FAST", "1") != "0"
    print(f"JAXDRB_FAST={'1' if fast else '0'} | classifier={args.classifier}", flush=True)

    nl = 24 if fast else 96
    geom = CircularTokamakGeometry.make(
        nl=nl, shat=0.8, q=3.0, R0=1.0, epsilon=0.18, curvature0=0.18
    )

    # Match previous pedagogic ranges.
    eta_grid = np.logspace(-2.3, 0.4, 6 if fast else 26)
    omega_n_grid = np.linspace(0.0, 2.2, 7 if fast else 31)
    ky = np.linspace(0.08, 0.9, 6 if fast else 30)

    base = DRBParams(
        omega_n=1.0,
        omega_Te=0.0,
        eta=0.1,
        me_hat=0.05,
        curvature_on=True,
        beta=0.0,
        tau_i=0.0,
        boussinesq=True,
        Dn=0.01,
        DOmega=0.01,
        DTe=0.01,
        kperp2_min=1e-6,
    )

    cal = MosettoCalibration(
        gamma_ratio=float(args.gamma_ratio),
        rln_per_omega_n=float(args.rln_per_omega_n),
        d_scale=float(args.d_scale),
    )

    labels_cal, extras_cal = classify_grid(
        eta=eta_grid,
        omega_n=omega_n_grid,
        me_hat=base.me_hat,
        shat=float(getattr(geom, "shat", 0.0)),
        calibration=cal,
    )

    labels_ab = None
    extras_ab: dict[str, np.ndarray] | None = None
    if args.classifier in {"ablation", "both"}:
        labels_ab, extras_ab = _ablation_label_grid(
            eta_grid=eta_grid,
            omega_n_grid=omega_n_grid,
            geom=geom,
            ky=ky,
            base=base,
            fast=fast,
        )

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap.N)
    labels_name = ["InDW", "RDW", "InBM", "RBM"]

    if args.classifier == "calibrated":
        fig, axs = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    else:
        fig, axs = plt.subplots(2, 2, figsize=(13.0, 9.6), constrained_layout=True)

    # --- Calibrated map
    ax0 = axs[0] if args.classifier == "calibrated" else axs[0, 0]
    im0 = ax0.pcolormesh(
        np.log10(eta_grid), omega_n_grid, labels_cal.T, shading="auto", cmap=cmap, norm=norm
    )
    cbar = fig.colorbar(im0, ax=ax0, ticks=[0, 1, 2, 3], pad=0.01)
    cbar.ax.set_yticklabels(labels_name)

    d_thr = threshold_d_resistive_inertial(float(getattr(geom, "shat", 0.0)), cal)
    eta_thr = d_thr * float(base.me_hat) / max(cal.d_scale, 1e-12)
    wn_thr_i = threshold_rln_indw_inbm(cal.gamma_ratio) / max(cal.rln_per_omega_n, 1e-12)
    wn_thr_r = threshold_rln_rdw_rbm(cal.gamma_ratio) / max(cal.rln_per_omega_n, 1e-12)

    ax0.axvline(
        np.log10(max(eta_thr, 1e-12)), color="k", ls="--", lw=1.5, alpha=0.8, label=r"$d_{crit}$"
    )
    ax0.axhline(wn_thr_i, color="k", ls=":", lw=1.5, alpha=0.8, label=r"$R/L_n$ InBM/InDW")
    ax0.axhline(wn_thr_r, color="k", ls="-.", lw=1.5, alpha=0.8, label=r"$R/L_n$ RBM/RDW")
    ax0.set_xlabel(r"$\log_{10}(\eta)$")
    ax0.set_ylabel(r"$\omega_n$ (proxy for $R/L_n$)")
    ax0.set_title("Calibrated 4-regime map (Mosetto thresholds)")
    ax0.legend(loc="lower right", fontsize=8)

    if args.classifier == "calibrated":
        ax1 = axs[1]
        rln = extras_cal["rln_proxy"]
        dproxy = extras_cal["d_proxy"]
        h1 = ax1.contourf(np.log10(eta_grid), omega_n_grid, rln.T, levels=16, cmap="viridis")
        fig.colorbar(h1, ax=ax1, pad=0.01, label=r"$R/L_n$ proxy")
        ax1.contour(
            np.log10(eta_grid),
            omega_n_grid,
            dproxy.T,
            levels=[d_thr],
            colors="w",
            linewidths=2.0,
            linestyles="--",
        )
        ax1.set_xlabel(r"$\log_{10}(\eta)$")
        ax1.set_ylabel(r"$\omega_n$")
        ax1.set_title(r"Proxy fields and $d=d_{crit}$ contour")
    else:
        # --- Ablation map
        assert labels_ab is not None and extras_ab is not None
        ax1 = axs[0, 1]
        im1 = ax1.pcolormesh(
            np.log10(eta_grid), omega_n_grid, labels_ab.T, shading="auto", cmap=cmap, norm=norm
        )
        cbar1 = fig.colorbar(im1, ax=ax1, ticks=[0, 1, 2, 3], pad=0.01)
        cbar1.ax.set_yticklabels(labels_name)
        ax1.set_xlabel(r"$\log_{10}(\eta)$")
        ax1.set_ylabel(r"$\omega_n$")
        ax1.set_title("Solver ablation map")

        # --- Mismatch panel
        ax2 = axs[1, 0]
        mismatch = labels_cal != labels_ab
        mismatch_mean = float(np.mean(mismatch))
        mm = ax2.pcolormesh(
            np.log10(eta_grid),
            omega_n_grid,
            mismatch.T.astype(float),
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        fig.colorbar(mm, ax=ax2, pad=0.01, label="mismatch (0/1)")
        ax2.set_xlabel(r"$\log_{10}(\eta)$")
        ax2.set_ylabel(r"$\omega_n$")
        ax2.set_title(f"Calibrated vs ablation mismatch (mean={mismatch_mean:.2f})")

        # --- Ablation helper ratio panel
        ax3 = axs[1, 1]
        gfull = extras_ab["gamma_full"]
        ratio = np.where(gfull > 1e-12, extras_ab["gamma_no_curv"] / gfull, np.nan)
        h3 = ax3.pcolormesh(
            np.log10(eta_grid), omega_n_grid, ratio.T, shading="auto", cmap="viridis"
        )
        fig.colorbar(h3, ax=ax3, pad=0.01, label=r"$\gamma_{no-curv}/\gamma_{full}$")
        ax3.set_xlabel(r"$\log_{10}(\eta)$")
        ax3.set_ylabel(r"$\omega_n$")
        ax3.set_title("Ablation helper: curvature-off ratio")

    fig.savefig(out_dir / "mosetto2012_fig6a_like.png", dpi=240)
    plt.close(fig)

    out = {
        "eta": eta_grid,
        "omega_n": omega_n_grid,
        "ky": ky,
        "label_index_calibrated": labels_cal,
        "d_proxy": extras_cal["d_proxy"],
        "rln_proxy": extras_cal["rln_proxy"],
        "rln_threshold": extras_cal["rln_threshold"],
        "d_threshold": extras_cal["d_threshold"],
    }
    if labels_ab is not None and extras_ab is not None:
        out["label_index_ablation"] = labels_ab
        out["gamma_full"] = extras_ab["gamma_full"]
        out["gamma_no_curv"] = extras_ab["gamma_no_curv"]
        out["gamma_no_inertia"] = extras_ab["gamma_no_inertia"]
        out["mismatch"] = (labels_cal != labels_ab).astype(float)

    np.savez(out_dir / "results.npz", **out)

    save_json(
        out_dir / "params.json",
        {
            "paper": "Mosetto et al. (2012) Phys. Plasmas 19, 112103",
            "classifier": args.classifier,
            "geometry": {
                "type": "circular_tokamak",
                "nl": nl,
                "shat": float(getattr(geom, "shat", 0.0)),
                "q": 3.0,
                "epsilon": 0.18,
                "curvature0": 0.18,
            },
            "base_params": base.__dict__,
            "calibration": {
                "gamma_ratio": cal.gamma_ratio,
                "rln_per_omega_n": cal.rln_per_omega_n,
                "d_scale": cal.d_scale,
                "dcrit0": cal.dcrit0,
                "dcrit5": cal.dcrit5,
                "derived": {
                    "d_crit": d_thr,
                    "eta_transition": eta_thr,
                    "omega_n_transition_inertial": wn_thr_i,
                    "omega_n_transition_resistive": wn_thr_r,
                },
            },
        },
    )
    if labels_ab is not None:
        print(
            f"Calibrated-vs-ablation mismatch fraction = {float(np.mean(labels_cal != labels_ab)):.3f}",
            flush=True,
        )

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
