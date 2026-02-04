#!/usr/bin/env python3
"""Sheath heat transmission + secondary electron emission (SEE) effects.

This example uses an *open* slab field line with Bohm/MPSE boundary conditions and
compares linear growth rates for a drift-wave-like setup with:

  1) MPSE Bohm BCs only
  2) MPSE Bohm BCs + sheath heat transmission (electron energy losses)
  3) MPSE Bohm BCs + heat transmission + SEE (via a reduced floating-potential shift Λ_eff)

The energy loss model is a lightweight closure intended as a bridge toward more
quantitative SOL modeling (sources/sinks, recycling, and full sheath models).

Run (from repo root):
  python examples/scripts/03_sheath_mpse/sheath_heat_see_effects.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from jaxdrb.analysis.plotting import save_geometry_overview, save_json, set_mpl_style
from jaxdrb.analysis.scan import scan_ky
from jaxdrb.geometry.slab import OpenSlabGeometry
from jaxdrb.models.params import DRBParams
from jaxdrb.models.registry import DEFAULT_MODEL


def _ky_star_from_gamma(ky: np.ndarray, gamma: np.ndarray) -> float:
    ratio = np.maximum(gamma, 0.0) / ky
    return float(ky[int(np.argmax(ratio))])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=str,
        default="out/examples/03_sheath_mpse/sheath_heat_see_effects",
        help="Output directory.",
    )
    p.add_argument("--nl", type=int, default=65)
    p.add_argument("--length", type=float, default=6.0)
    p.add_argument("--kx", type=float, default=0.0)
    p.add_argument("--ky-min", type=float, default=0.05)
    p.add_argument("--ky-max", type=float, default=1.2)
    p.add_argument("--nky", type=int, default=36)
    p.add_argument("--omega-n", type=float, default=0.8)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--me-hat", type=float, default=0.05)
    p.add_argument("--sheath-gamma-e", type=float, default=0.0)
    p.add_argument("--see-yields", type=float, nargs="*", default=[0.0, 0.2, 0.4])
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))
    set_mpl_style()

    geom = OpenSlabGeometry.make(nl=args.nl, length=args.length, shat=0.0, curvature0=0.0)
    ky = np.linspace(args.ky_min, args.ky_max, args.nky)

    base = dict(
        omega_n=float(args.omega_n),
        omega_Te=0.0,
        eta=float(args.eta),
        me_hat=float(args.me_hat),
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        sheath_bc_on=True,
        sheath_bc_model=0,  # velocity-only MPSE (avoid mixing with Loizu2012 Te-gradient constraint here)
        sheath_bc_nu_factor=1.0,
        sheath_lambda=3.28,
        sheath_delta=0.0,
    )

    configs: list[tuple[str, DRBParams]] = []
    configs.append(("mpse_only", DRBParams(**base, sheath_heat_on=False)))
    configs.append(
        (
            "mpse_heat",
            DRBParams(
                **base,
                sheath_heat_on=True,
                sheath_gamma_auto=(float(args.sheath_gamma_e) <= 0.0),
                sheath_gamma_e=float(args.sheath_gamma_e),
            ),
        )
    )
    for see in args.see_yields:
        if float(see) <= 0.0:
            continue
        configs.append(
            (
                f"mpse_heat_see{see:.2f}",
                DRBParams(
                    **base,
                    sheath_heat_on=True,
                    sheath_gamma_auto=(float(args.sheath_gamma_e) <= 0.0),
                    sheath_gamma_e=float(args.sheath_gamma_e),
                    sheath_see_on=True,
                    sheath_see_yield=float(see),
                ),
            )
        )

    save_json(
        out_dir / "run_config.json",
        {
            "geom": "slab-open",
            "nl": int(args.nl),
            "length": float(args.length),
            "kx": float(args.kx),
            "ky": {"min": float(args.ky_min), "max": float(args.ky_max), "n": int(args.nky)},
            "base_params": base,
            "see_yields": [float(x) for x in args.see_yields],
        },
    )

    # Always save a geometry overview for reproducibility / sanity checks.
    ky_star_geom = float(ky[int(np.argmax(1.0 / ky))])  # just to pick a representative point
    save_geometry_overview(out_dir, geom=geom, kx=float(args.kx), ky=ky_star_geom)

    results: dict[str, dict[str, np.ndarray]] = {}
    for name, params in configs:
        print(f"\n=== {name} ===", flush=True)
        print(
            f"sheath_heat_on={params.sheath_heat_on}  "
            f"gamma_auto={params.sheath_gamma_auto}  "
            f"SEE={params.sheath_see_on} (delta={params.sheath_see_yield})",
            flush=True,
        )
        res = scan_ky(
            params,
            geom,
            ky=ky,
            kx=float(args.kx),
            model=DEFAULT_MODEL,
            arnoldi_m=40,
            arnoldi_tol=5e-3,
            nev=6,
            seed=0,
            do_initial_value=False,
            verbose=True,
            print_every=3,
        )
        results[name] = {"gamma": res.gamma_eigs, "omega": res.omega_eigs}

    np.savez(
        out_dir / "results_sheath_heat_see.npz",
        ky=ky,
        **{f"gamma_{k}": v["gamma"] for k, v in results.items()},
        **{f"omega_{k}": v["omega"] for k, v in results.items()},
    )

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    ax = axs[0]
    for name, _params in configs:
        ax.plot(ky, results[name]["gamma"], marker="o", label=name)
    ax.axhline(0.0, color="k", alpha=0.3)
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("Growth rate (eigenvalue)")
    ax.legend(ncols=2)

    ax = axs[1]
    for name, _params in configs:
        ratio = np.maximum(results[name]["gamma"], 0.0) / ky
        ax.plot(ky, ratio, marker="o", label=name)
        ax.axvline(_ky_star_from_gamma(ky, results[name]["gamma"]), color="k", alpha=0.08)
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$(\gamma/k_y)_+$")
    ax.set_title(r"Transport proxy: $\max(\gamma,0)/k_y$")

    fig.savefig(out_dir / "sheath_heat_see_scan.png", dpi=220)
    plt.close(fig)

    # Report ky* for each config.
    summary = {}
    for name, _params in configs:
        gamma = results[name]["gamma"]
        ratio = np.maximum(gamma, 0.0) / ky
        i_star = int(np.argmax(ratio))
        ky_star = float(ky[i_star])
        summary[name] = {
            "ky_star": ky_star,
            "gamma_at_ky_star": float(gamma[i_star]),
            "ratio_star": float(ratio[i_star]),
        }
        print(f"[{name}] ky*={ky_star:.4f}  (gamma/ky)*={ratio[i_star]:.4e}", flush=True)
    save_json(out_dir / "summary.json", summary)

    print(f"\nWrote {out_dir / 'results_sheath_heat_see.npz'}", flush=True)
    print(f"Wrote {out_dir / 'sheath_heat_see_scan.png'}", flush=True)
    print(f"Wrote {out_dir / 'geometry_overview.png'}", flush=True)


if __name__ == "__main__":
    main()
