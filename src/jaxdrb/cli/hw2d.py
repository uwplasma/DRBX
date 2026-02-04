from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, hw2d_random_ic
from jaxdrb.nonlinear.neutrals import NeutralParams
from jaxdrb.nonlinear.stepper import rk4_scan


def main() -> None:
    parser = argparse.ArgumentParser(prog="jaxdrb-hw2d")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--Lx", type=float, default=float(2 * jnp.pi))
    parser.add_argument("--Ly", type=float, default=float(2 * jnp.pi))
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--tmax", type=float, default=40.0)
    parser.add_argument("--save-stride", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", type=float, default=1e-3)

    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--Dn", type=float, default=2e-4)
    parser.add_argument("--DOmega", type=float, default=2e-4)
    parser.add_argument(
        "--bracket", choices=["spectral", "arakawa", "centered"], default="spectral"
    )
    parser.add_argument("--poisson", choices=["spectral", "cg_fd"], default="spectral")
    parser.add_argument("--no-dealias", action="store_true")
    parser.add_argument(
        "--bc-x",
        choices=["periodic", "dirichlet", "neumann"],
        default="periodic",
        help="x boundary condition",
    )
    parser.add_argument(
        "--bc-y",
        choices=["periodic", "dirichlet", "neumann"],
        default="periodic",
        help="y boundary condition",
    )
    parser.add_argument(
        "--bc-value-x", type=float, default=0.0, help="Dirichlet value at x boundaries"
    )
    parser.add_argument(
        "--bc-value-y", type=float, default=0.0, help="Dirichlet value at y boundaries"
    )
    parser.add_argument("--bc-grad-x", type=float, default=0.0, help="Neumann grad at x boundaries")
    parser.add_argument("--bc-grad-y", type=float, default=0.0, help="Neumann grad at y boundaries")
    parser.add_argument(
        "--bc-enforce-nu",
        type=float,
        default=0.0,
        help="Boundary relaxation rate for evolving fields (0 disables)",
    )

    parser.add_argument("--neutrals", action="store_true")
    parser.add_argument("--Dn0", type=float, default=1e-3)
    parser.add_argument("--nu-ion", type=float, default=0.2)
    parser.add_argument("--nu-rec", type=float, default=0.02)
    parser.add_argument(
        "--n-background", type=float, default=1.0, help="Background density used in ionization"
    )
    parser.add_argument("--neutral-source", type=float, default=0.0)
    parser.add_argument("--neutral-sink", type=float, default=0.0)

    parser.add_argument("--out", type=str, default="out_hw2d_cli")
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    jax.config.update("jax_enable_x64", True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = Grid2D.make(
        nx=args.nx,
        ny=args.ny,
        Lx=args.Lx,
        Ly=args.Ly,
        dealias=not args.no_dealias,
        bc_x=args.bc_x,
        bc_y=args.bc_y,
        bc_value_x=float(args.bc_value_x),
        bc_value_y=float(args.bc_value_y),
        bc_grad_x=float(args.bc_grad_x),
        bc_grad_y=float(args.bc_grad_y),
    )
    neutrals = NeutralParams(
        enabled=bool(args.neutrals),
        Dn0=float(args.Dn0),
        nu_ion=float(args.nu_ion),
        nu_rec=float(args.nu_rec),
        n_background=float(args.n_background),
        S0=float(args.neutral_source),
        nu_sink=float(args.neutral_sink),
    )
    params = HW2DParams(
        kappa=float(args.kappa),
        alpha=float(args.alpha),
        Dn=float(args.Dn),
        DOmega=float(args.DOmega),
        bracket=args.bracket,
        poisson=args.poisson,
        dealias_on=not args.no_dealias,
        bc_enforce_nu=float(args.bc_enforce_nu),
        neutrals=neutrals,
    )
    model = HW2DModel(params=params, grid=grid)

    y0 = hw2d_random_ic(
        jax.random.key(args.seed),
        grid,
        amp=float(args.amp),
        include_neutrals=bool(args.neutrals),
    )

    dt = float(args.dt)
    nsteps = int(jnp.ceil(args.tmax / dt))
    save_stride = int(args.save_stride)
    nchunks = max(1, nsteps // save_stride)

    (out_dir / "params.json").write_text(
        json.dumps(
            {
                "grid": {"nx": grid.nx, "ny": grid.ny, "Lx": grid.Lx, "Ly": grid.Ly},
                "model": {
                    "kappa": params.kappa,
                    "alpha": params.alpha,
                    "Dn": params.Dn,
                    "DOmega": params.DOmega,
                    "bracket": params.bracket,
                    "poisson": params.poisson,
                    "dealias_on": params.dealias_on,
                    "bc_x": args.bc_x,
                    "bc_y": args.bc_y,
                    "bc_value_x": float(args.bc_value_x),
                    "bc_value_y": float(args.bc_value_y),
                    "bc_grad_x": float(args.bc_grad_x),
                    "bc_grad_y": float(args.bc_grad_y),
                    "bc_enforce_nu": float(args.bc_enforce_nu),
                },
                "neutrals": {
                    "enabled": neutrals.enabled,
                    "Dn0": neutrals.Dn0,
                    "nu_ion": neutrals.nu_ion,
                    "nu_rec": neutrals.nu_rec,
                    "S0": neutrals.S0,
                    "nu_sink": neutrals.nu_sink,
                },
                "time": {"dt": dt, "tmax": float(args.tmax), "save_stride": save_stride},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        f"[jaxdrb-hw2d] grid=({grid.nx},{grid.ny}) dt={dt} nsteps={nsteps} "
        f"chunks={nchunks} bracket={params.bracket} neutrals={neutrals.enabled}"
    )

    def rhs(t, y):
        return model.rhs(t, y)

    t = 0.0
    y = y0
    ts = []
    Es = []
    Zs = []
    nbar = []
    Nbar = []

    for k in range(nchunks):
        _, y = rk4_scan(y, t0=t, dt=dt, nsteps=save_stride, rhs=rhs)
        t = t + dt * save_stride
        diag = model.diagnostics(y)
        ts.append(t)
        Es.append(float(diag["E"]))
        Zs.append(float(diag["Z"]))
        nbar.append(float(jnp.mean(y.n)))
        if y.N is not None:
            Nbar.append(float(jnp.mean(y.N)))
        print(f"[jaxdrb-hw2d] chunk {k+1}/{nchunks} t={t:.3f} E={Es[-1]:.3e} Z={Zs[-1]:.3e}")

    jnp.savez(out_dir / "timeseries.npz", t=jnp.array(ts), E=jnp.array(Es), Z=jnp.array(Zs))

    if Nbar:
        jnp.savez(
            out_dir / "means.npz", t=jnp.array(ts), nbar=jnp.array(nbar), Nbar=jnp.array(Nbar)
        )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(ts, Es, label="E")
    ax.plot(ts, Zs, label="Z")
    ax.set_xlabel("t")
    ax.set_yscale("log")
    ax.set_title("HW2D diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics.png", dpi=200)
    plt.close(fig)

    if Nbar:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(ts, nbar, label="<n>")
        ax.plot(ts, Nbar, label="<N>")
        ax.set_xlabel("t")
        ax.set_title("Mean densities")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "means.png", dpi=200)
        plt.close(fig)

    phi = model.phi_from_omega(y.omega)
    for name, arr in {"n": y.n, "phi": phi, "omega": y.omega}.items():
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        im = ax.imshow(arr.T, origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    if y.N is not None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        im = ax.imshow(y.N.T, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title("N (neutrals)")
        fig.colorbar(im, ax=ax, shrink=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / "N.png", dpi=200)
        plt.close(fig)

    print(f"[jaxdrb-hw2d] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
