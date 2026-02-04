from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from jaxdrb.analysis.scan import scan_kx_ky
from jaxdrb.analysis.plotting import save_kxky_heatmap, set_mpl_style
from jaxdrb.geometry.slab import OpenSlabGeometry, SlabGeometry
from jaxdrb.geometry.tabulated import TabulatedGeometry
from jaxdrb.geometry.tokamak import (
    CircularTokamakGeometry,
    OpenCircularTokamakGeometry,
    OpenSAlphaGeometry,
    SAlphaGeometry,
)
from jaxdrb.models.params import DRBParams
from jaxdrb.models.bcs import LineBCs
from jaxdrb.models.registry import DEFAULT_MODEL, MODELS, get_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="jaxdrb-scan2d")
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL.name)
    parser.add_argument(
        "--geom",
        choices=[
            "slab",
            "slab-open",
            "tabulated",
            "tokamak",
            "tokamak-open",
            "salpha",
            "salpha-open",
        ],
        required=True,
    )
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
    parser.add_argument("--omega-Ti", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--me-hat", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--tau-i", type=float, default=0.0)
    parser.add_argument("--no-curvature", action="store_true")
    parser.add_argument("--no-boussinesq", action="store_true")
    parser.add_argument("--Dn", type=float, default=0.01)
    parser.add_argument("--DOmega", type=float, default=0.01)
    parser.add_argument("--DTe", type=float, default=0.01)
    parser.add_argument("--DTi", type=float, default=0.01)
    parser.add_argument("--Dpsi", type=float, default=0.0)
    parser.add_argument("--kperp2-min", type=float, default=1e-6)

    parser.add_argument(
        "--line-bc",
        choices=["none", "dirichlet", "neumann"],
        default="none",
        help="Apply a user BC along l to all fields (benchmarking/nonlinear-prep; may conflict with MPSE).",
    )
    parser.add_argument(
        "--line-bc-value", type=float, default=0.0, help="Dirichlet value at l-ends"
    )
    parser.add_argument(
        "--line-bc-grad", type=float, default=0.0, help="Neumann gradient at l-ends"
    )
    parser.add_argument(
        "--line-bc-nu",
        type=float,
        default=0.0,
        help="BC relaxation rate (0 disables BC enforcement).",
    )

    parser.add_argument("--sheath", action="store_true", help="Alias for --sheath-bc (MPSE BCs)")
    parser.add_argument(
        "--no-sheath-bc",
        action="store_true",
        help="Disable MPSE Bohm sheath boundary conditions even for *-open geometries.",
    )
    parser.add_argument(
        "--sheath-bc", action="store_true", help="Enable Loizu-style MPSE Bohm sheath BCs"
    )
    parser.add_argument(
        "--sheath-bc-nu-factor", type=float, default=1.0, help="BC enforcement rate factor (~2/L||)"
    )
    parser.add_argument(
        "--sheath-lambda", type=float, default=3.28, help="Lambda = 0.5 ln(mi/(2π me))"
    )
    parser.add_argument(
        "--sheath-delta",
        type=float,
        default=0.0,
        help="Ion transmission correction (cold ions -> 0)",
    )

    parser.add_argument(
        "--sheath-loss",
        action="store_true",
        help="Enable volumetric end-loss proxy (nu_sh ~ 2/L||)",
    )
    parser.add_argument(
        "--sheath-loss-nu-factor", type=float, default=1.0, help="Multiplier for nu_sh"
    )

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
    set_mpl_style()

    model = get_model(args.model)

    if args.geom == "slab":
        geom = SlabGeometry.make(
            nl=args.nl, length=args.length, shat=args.shat, curvature0=args.curvature0
        )
    elif args.geom == "slab-open":
        geom = OpenSlabGeometry.make(
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
    elif args.geom == "tokamak-open":
        geom = OpenCircularTokamakGeometry.make(
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
    elif args.geom == "salpha-open":
        geom = OpenSAlphaGeometry.make(
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
        "model": args.model,
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
        "omega_Ti": args.omega_Ti,
        "eta": args.eta,
        "me_hat": args.me_hat,
        "beta": args.beta,
        "tau_i": args.tau_i,
        "curvature_on": not args.no_curvature,
        "boussinesq": not args.no_boussinesq,
        "Dn": args.Dn,
        "DOmega": args.DOmega,
        "DTe": args.DTe,
        "DTi": args.DTi,
        "Dpsi": args.Dpsi,
        "kperp2_min": args.kperp2_min,
        "line_bc": args.line_bc,
        "line_bc_value": float(args.line_bc_value),
        "line_bc_grad": float(args.line_bc_grad),
        "line_bc_nu": float(args.line_bc_nu),
        "sheath_bc_on": bool(args.sheath or args.sheath_bc)
        or (args.geom.endswith("-open") and not args.no_sheath_bc),
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

    if args.line_bc == "dirichlet":
        line_bcs = LineBCs.all_dirichlet(value=float(args.line_bc_value), nu=float(args.line_bc_nu))
    elif args.line_bc == "neumann":
        line_bcs = LineBCs.all_neumann(grad=float(args.line_bc_grad), nu=float(args.line_bc_nu))
    else:
        line_bcs = LineBCs.disabled()

    params = DRBParams(
        omega_n=args.omega_n,
        omega_Te=args.omega_Te,
        omega_Ti=args.omega_Ti,
        eta=args.eta,
        me_hat=args.me_hat,
        beta=args.beta,
        tau_i=args.tau_i,
        curvature_on=not args.no_curvature,
        boussinesq=not args.no_boussinesq,
        Dn=args.Dn,
        DOmega=args.DOmega,
        DTe=args.DTe,
        DTi=args.DTi,
        Dpsi=args.Dpsi,
        kperp2_min=args.kperp2_min,
        sheath_bc_on=bool(args.sheath or args.sheath_bc)
        or (args.geom.endswith("-open") and not args.no_sheath_bc),
        sheath_bc_nu_factor=float(args.sheath_bc_nu_factor),
        sheath_lambda=float(args.sheath_lambda),
        sheath_delta=float(args.sheath_delta),
        sheath_loss_on=bool(args.sheath_loss),
        sheath_loss_nu_factor=float(args.sheath_loss_nu_factor),
        line_bcs=line_bcs,
    )

    ky_grid = np.linspace(args.ky_min, args.ky_max, args.nky)
    kx_grid = np.linspace(args.kx_min, args.kx_max, args.nkx)

    res = scan_kx_ky(
        params,
        geom,
        kx=kx_grid,
        ky=ky_grid,
        model=model,
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

    save_kxky_heatmap(
        out_dir,
        kx=res.kx,
        ky=res.ky,
        z=res.gamma_eigs,
        zlabel=r"$\gamma$",
        title=rf"Leading growth rate $\gamma(k_x,k_y)$ ({args.model}, {args.geom})",
        filename="gamma_kxky.png",
        cmap="magma",
    )

    # For each ky, show the maximized gamma over kx and report the argmax location.
    gmax = np.max(res.gamma_eigs, axis=0)
    kx_star = res.kx[np.argmax(res.gamma_eigs, axis=0)]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(res.ky, gmax, "o-", label=r"$\max_{k_x}\,\gamma$")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.legend()
    fig.savefig(out_dir / "gamma_ky_max_over_kx.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(res.ky, kx_star, "o-")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$k_x^*(k_y)$")
    fig.savefig(out_dir / "kx_star_vs_ky.png", dpi=220)
    plt.close(fig)

    print(f"Wrote {out_dir / 'results_2d.npz'}", flush=True)
    print(f"Wrote {out_dir / 'gamma_kxky.png'}", flush=True)
    print(f"Wrote {out_dir / 'gamma_ky_max_over_kx.png'}", flush=True)
    print(f"Wrote {out_dir / 'kx_star_vs_ky.png'}", flush=True)


if __name__ == "__main__":
    main()
