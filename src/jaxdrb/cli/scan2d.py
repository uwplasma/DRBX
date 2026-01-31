from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from jaxdrb.analysis.scan import scan_kx_ky
from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.geometry.tabulated import TabulatedGeometry
from jaxdrb.geometry.tokamak import CircularTokamakGeometry, SAlphaGeometry
from jaxdrb.models.params import DRBParams


def main() -> None:
    parser = argparse.ArgumentParser(prog="jaxdrb-scan2d")
    parser.add_argument("--geom", choices=["slab", "tabulated", "tokamak", "salpha"], required=True)
    parser.add_argument("--geom-file", type=str, default=None)
    parser.add_argument("--nl", type=int, default=64)
    parser.add_argument("--length", type=float, default=float(2 * np.pi))
    parser.add_argument("--shat", type=float, default=0.8)
    parser.add_argument("--q", type=float, default=1.4)
    parser.add_argument("--R0", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.18)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--curvature0", type=float, default=0.0)

    parser.add_argument("--omega-n", type=float, default=0.8)
    parser.add_argument("--omega-Te", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--me-hat", type=float, default=0.05)
    parser.add_argument("--no-curvature", action="store_true")
    parser.add_argument("--Dn", type=float, default=0.01)
    parser.add_argument("--DOmega", type=float, default=0.01)
    parser.add_argument("--DTe", type=float, default=0.01)
    parser.add_argument("--kperp2-min", type=float, default=1e-6)

    parser.add_argument("--ky-min", type=float, required=True)
    parser.add_argument("--ky-max", type=float, required=True)
    parser.add_argument("--nky", type=int, default=32)
    parser.add_argument("--kx-min", type=float, required=True)
    parser.add_argument("--kx-max", type=float, required=True)
    parser.add_argument("--nkx", type=int, default=33)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--arnoldi-m", type=int, default=40)
    parser.add_argument("--arnoldi-max-m", type=int, default=None)
    parser.add_argument("--arnoldi-tol", type=float, default=1e-3)
    parser.add_argument("--nev", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))

    if args.geom == "slab":
        geom = SlabGeometry.make(
            nl=args.nl, length=args.length, shat=args.shat, curvature0=args.curvature0
        )
    elif args.geom == "tokamak":
        geom = CircularTokamakGeometry.make(
            nl=args.nl,
            length=args.length,
            shat=args.shat,
            q=args.q,
            R0=args.R0,
            epsilon=args.epsilon,
            curvature0=args.curvature0 if args.curvature0 != 0.0 else None,
        )
    elif args.geom == "salpha":
        geom = SAlphaGeometry.make(
            nl=args.nl,
            length=args.length,
            shat=args.shat,
            alpha=args.alpha,
            q=args.q,
            R0=args.R0,
            epsilon=args.epsilon,
            curvature0=args.curvature0 if args.curvature0 != 0.0 else None,
        )
    else:
        if args.geom_file is None:
            raise SystemExit("--geom-file is required for --geom tabulated")
        geom = TabulatedGeometry.from_npz(args.geom_file)
        if geom.l.size != args.nl:
            raise SystemExit(f"--nl={args.nl} but geometry file has nl={geom.l.size}")

    run_cfg = {
        "geom": args.geom,
        "geom_file": args.geom_file,
        "nl": args.nl,
        "length": args.length,
        "shat": args.shat,
        "q": args.q,
        "R0": args.R0,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "curvature0": args.curvature0,
        "curvature0_effective": float(getattr(geom, "curvature0", args.curvature0)),
        "omega_n": args.omega_n,
        "omega_Te": args.omega_Te,
        "eta": args.eta,
        "me_hat": args.me_hat,
        "curvature_on": not args.no_curvature,
        "Dn": args.Dn,
        "DOmega": args.DOmega,
        "DTe": args.DTe,
        "kperp2_min": args.kperp2_min,
        "ky_min": args.ky_min,
        "ky_max": args.ky_max,
        "nky": args.nky,
        "kx_min": args.kx_min,
        "kx_max": args.kx_max,
        "nkx": args.nkx,
        "arnoldi_m": args.arnoldi_m,
        "arnoldi_max_m": args.arnoldi_max_m,
        "arnoldi_tol": args.arnoldi_tol,
        "nev": args.nev,
        "seed": args.seed,
    }
    (out_dir / "params.json").write_text(json.dumps(run_cfg, indent=2, sort_keys=True) + "\n")

    params = DRBParams(
        omega_n=args.omega_n,
        omega_Te=args.omega_Te,
        eta=args.eta,
        me_hat=args.me_hat,
        curvature_on=not args.no_curvature,
        Dn=args.Dn,
        DOmega=args.DOmega,
        DTe=args.DTe,
        kperp2_min=args.kperp2_min,
    )

    ky_grid = np.linspace(args.ky_min, args.ky_max, args.nky)
    kx_grid = np.linspace(args.kx_min, args.kx_max, args.nkx)

    res = scan_kx_ky(
        params,
        geom,
        kx=kx_grid,
        ky=ky_grid,
        arnoldi_m=args.arnoldi_m,
        arnoldi_tol=args.arnoldi_tol,
        arnoldi_max_m=args.arnoldi_max_m,
        nev=args.nev,
        seed=args.seed,
    )

    np.savez(
        out_dir / "results_2d.npz",
        kx=res.kx,
        ky=res.ky,
        gamma_eigs=res.gamma_eigs,
        omega_eigs=res.omega_eigs,
    )

    import matplotlib.pyplot as plt

    # Heatmap of growth rates
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    im = ax.pcolormesh(res.ky, res.kx, res.gamma_eigs, shading="auto", cmap="magma")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$k_x$")
    ax.set_title(r"Leading growth rate $\gamma(k_x,k_y)$")
    fig.colorbar(im, ax=ax, label=r"$\gamma$")
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_kxky.png", dpi=160)
    plt.close(fig)

    # For each ky, show the maximized gamma over kx
    gmax = np.max(res.gamma_eigs, axis=0)
    kx_star = res.kx[np.argmax(res.gamma_eigs, axis=0)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res.ky, gmax, "o-", label=r"$\max_{k_x}\,\gamma$")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_ky_max_over_kx.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res.ky, kx_star, "o-")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$k_x^*(k_y)$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "kx_star_vs_ky.png", dpi=160)
    plt.close(fig)

    print(f"Wrote {out_dir / 'results_2d.npz'}")
    print(f"Wrote {out_dir / 'gamma_kxky.png'}")
    print(f"Wrote {out_dir / 'gamma_ky_max_over_kx.png'}")
    print(f"Wrote {out_dir / 'kx_star_vs_ky.png'}")


if __name__ == "__main__":
    main()
