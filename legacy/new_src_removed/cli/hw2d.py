from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, hw2d_random_ic
from jaxdrb.nonlinear.neutrals import NeutralParams
from jaxdrb.analysis.plotting import robust_symmetric_vlim


def main() -> None:
    parser = argparse.ArgumentParser(prog="jaxdrb-hw2d")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--Lx", type=float, default=float(2 * jnp.pi))
    parser.add_argument("--Ly", type=float, default=float(2 * jnp.pi))
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--tmax", type=float, default=40.0)
    parser.add_argument("--save-stride", type=int, default=20)
    parser.add_argument(
        "--solver",
        type=str,
        default="tsit5",
        choices=[
            "tsit5",
            "dopri5",
            "dopri8",
            "euler",
            "implicit_euler",
            "kvaerno3",
            "kvaerno4",
            "kvaerno5",
            "kencarp3",
            "kencarp4",
            "kencarp5",
        ],
        help="Diffrax solver to use. Implicit solvers can help for stiff closure/dissipation.",
    )
    parser.add_argument(
        "--fixed-step",
        action="store_true",
        help="Use constant step size dt (disables adaptive PID control).",
    )
    parser.add_argument("--rtol", type=float, default=1e-5, help="Diffrax relative tolerance.")
    parser.add_argument("--atol", type=float, default=1e-8, help="Diffrax absolute tolerance.")
    parser.add_argument(
        "--max-steps", type=int, default=300_000, help="Maximum number of Diffrax steps."
    )
    parser.add_argument(
        "--progress", action="store_true", help="Show Diffrax progress meter (can be verbose)."
    )
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
    parser.add_argument(
        "--nu-cx-omega",
        type=float,
        default=0.0,
        help="Charge-exchange-like vorticity drag coefficient (domega <- domega - nu_cx_omega*N*omega).",
    )

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
        nu_cx_omega=float(args.nu_cx_omega),
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
    save_stride = int(args.save_stride)
    frame_dt = dt * save_stride
    save_ts = jnp.arange(0.0, float(args.tmax) + 1e-12, frame_dt)
    nframes = int(save_ts.size)

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
                    "nu_cx_omega": neutrals.nu_cx_omega,
                },
                "time": {
                    "dt0": dt,
                    "tmax": float(args.tmax),
                    "save_stride": save_stride,
                    "solver": args.solver,
                    "adaptive": (not args.fixed_step),
                    "rtol": float(args.rtol),
                    "atol": float(args.atol),
                    "max_steps": int(args.max_steps),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        f"[jaxdrb-hw2d] grid=({grid.nx},{grid.ny}) dt0={dt} tmax={args.tmax} "
        f"save_stride={save_stride} frames={nframes} solver={args.solver} "
        f"adaptive={not args.fixed_step} bracket={params.bracket} neutrals={neutrals.enabled}"
    )

    sol = model.diffeqsolve(
        y0=y0,
        t0=0.0,
        t1=float(args.tmax),
        dt0=dt,
        save_ts=save_ts,
        solver=args.solver,
        adaptive=not args.fixed_step,
        rtol=float(args.rtol),
        atol=float(args.atol),
        max_steps=int(args.max_steps),
        progress=bool(args.progress),
    )

    ts = [float(t) for t in np.asarray(save_ts)]
    Es = []
    Zs = []
    nbar = []
    Nbar = []
    y_frames = sol.ys
    for k in range(nframes):
        yk = type(y0)(
            n=y_frames.n[k],
            omega=y_frames.omega[k],
            N=None if y_frames.N is None else y_frames.N[k],
        )
        diag = model.diagnostics(yk)
        Es.append(float(diag["E"]))
        Zs.append(float(diag["Z"]))
        nbar.append(float(jnp.mean(yk.n)))
        if yk.N is not None:
            Nbar.append(float(jnp.mean(yk.N)))
        print(f"[jaxdrb-hw2d] frame {k + 1}/{nframes} t={ts[k]:.3f} E={Es[-1]:.3e} Z={Zs[-1]:.3e}")

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

    y_end = type(y0)(
        n=y_frames.n[-1],
        omega=y_frames.omega[-1],
        N=None if y_frames.N is None else y_frames.N[-1],
    )
    phi = model.phi_from_omega(y_end.omega)
    for name, arr in {"n": y_end.n, "phi": phi, "omega": y_end.omega}.items():
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        arr_np = np.asarray(arr)
        vmax = robust_symmetric_vlim(arr_np, q=0.995)
        im = ax.imshow(
            arr_np.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax
        )
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    if y_end.N is not None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        im = ax.imshow(y_end.N.T, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title("N (neutrals)")
        fig.colorbar(im, ax=ax, shrink=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / "N.png", dpi=200)
        plt.close(fig)

    print(f"[jaxdrb-hw2d] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
