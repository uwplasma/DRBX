from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np

from jaxdrb.geometry.slab import SlabGeometry
from jaxdrb.geometry.tabulated import TabulatedGeometry
from jaxdrb.geometry.tokamak import CircularTokamakGeometry, SAlphaGeometry
from jaxdrb.linear.arnoldi import arnoldi_eigs
from jaxdrb.linear.growthrate import estimate_growth_rate
from jaxdrb.linear.matvec import linear_matvec
from jaxdrb.models.cold_ion_drb import State, equilibrium
from jaxdrb.models.params import DRBParams


def main() -> None:
    parser = argparse.ArgumentParser(prog="jaxdrb-scan")
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
    parser.add_argument("--kx", type=float, default=0.0)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--arnoldi-m", type=int, default=40)
    parser.add_argument("--arnoldi-max-m", type=int, default=None)
    parser.add_argument("--arnoldi-tol", type=float, default=1e-3)
    parser.add_argument("--nev", type=int, default=6)
    parser.add_argument("--tmax", type=float, default=30.0)
    parser.add_argument("--dt0", type=float, default=0.01)
    parser.add_argument("--nsave", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))

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
        "kx": args.kx,
        "arnoldi_m": args.arnoldi_m,
        "arnoldi_max_m": args.arnoldi_max_m,
        "arnoldi_tol": args.arnoldi_tol,
        "nev": args.nev,
        "tmax": args.tmax,
        "dt0": args.dt0,
        "nsave": args.nsave,
        "seed": args.seed,
    }

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

    run_cfg["curvature0_effective"] = float(getattr(geom, "curvature0", args.curvature0))
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

    gamma_eigs = np.zeros((args.nky,), dtype=float)
    omega_eigs = np.zeros((args.nky,), dtype=float)
    gamma_iv = np.zeros((args.nky,), dtype=float)
    eigs = np.zeros((args.nky, args.nev), dtype=np.complex128)

    y_eq = equilibrium(args.nl)

    key = jax.random.PRNGKey(args.seed)
    for i, ky in enumerate(ky_grid):
        key, subkey = jax.random.split(key)
        v0 = State.random(subkey, args.nl, amplitude=1e-3)

        matvec = linear_matvec(y_eq, params, geom, kx=args.kx, ky=float(ky))
        max_m = args.arnoldi_max_m
        if max_m is None:
            max_m = 5 * args.nl
        max_m = min(max_m, 5 * args.nl)

        m = min(args.arnoldi_m, max_m)
        arn = arnoldi_eigs(matvec, v0, m=m, nev=args.nev, seed=args.seed)
        lead_idx = int(np.argmax(np.real(arn.eigenvalues)))
        lead = arn.eigenvalues[lead_idx]
        rel_resid = float(arn.residual_norms[lead_idx] / (abs(lead) + 1.0))
        while rel_resid > args.arnoldi_tol and m < max_m:
            m = min(int(np.ceil(m * 2.0)), max_m)
            arn = arnoldi_eigs(matvec, v0, m=m, nev=args.nev, seed=args.seed)
            lead_idx = int(np.argmax(np.real(arn.eigenvalues)))
            lead = arn.eigenvalues[lead_idx]
            rel_resid = float(arn.residual_norms[lead_idx] / (abs(lead) + 1.0))

        eigs[i, : len(arn.eigenvalues)] = arn.eigenvalues
        gamma_eigs[i] = float(np.real(lead))
        omega_eigs[i] = float(np.imag(lead))

        gr = estimate_growth_rate(
            matvec,
            v0,
            tmax=args.tmax,
            dt0=args.dt0,
            nsave=args.nsave,
            fit_window=0.5,
        )
        gamma_iv[i] = gr.gamma

        print(
            f"ky={ky:8.4f}  gamma_eig={gamma_eigs[i]:10.4e}  gamma_iv={gamma_iv[i]:10.4e}  "
            f"m={m:4d}  rel_res={rel_resid:9.2e}"
        )

    np.savez(
        out_dir / "results.npz",
        ky=ky_grid,
        gamma_eigs=gamma_eigs,
        omega_eigs=omega_eigs,
        gamma_iv=gamma_iv,
        eigs=eigs,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ky_grid, gamma_eigs, "o-", label="Re(eig)")
    ax.plot(ky_grid, gamma_iv, "s--", label="init-value")
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"$\gamma$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "gamma_ky.png", dpi=150)
    plt.close(fig)

    print(f"Wrote {out_dir / 'results.npz'}")
    print(f"Wrote {out_dir / 'gamma_ky.png'}")
