"""Nonlinear 2D Hasegawa–Wakatani drift-wave testbed (periodic).

This script is a nonlinear milestone for jaxdrb:
  - fully JAX-jitted pseudo-spectral operators (FFT Poisson solve, dealiasing)
  - conservative Arakawa bracket option (finite-difference, periodic)
  - fast fixed-step time stepping using `jax.lax.scan`

It produces a small results folder with:
  - time traces of energy-like diagnostics
  - snapshots of (n, phi, omega)
  - a quick Fourier spectrum plot
"""

from __future__ import annotations

import argparse
import json
import os
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, hw2d_random_ic
from jaxdrb.nonlinear.stepper import rk4_scan


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--nsteps", type=int, default=800)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--save-stride", type=int, default=20)
    parser.add_argument("--out", type=str, default="out_hw2d")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grid and parameters (qualitative HW turbulence regime).
    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=True)
    params = HW2DParams(
        kappa=1.0,  # background gradient drive
        alpha=0.5,  # finite resistive coupling -> drift waves
        Dn=2e-4,
        DOmega=2e-4,
        bracket="spectral",
        dealias_on=True,
    )
    model = HW2DModel(params=params, grid=grid)

    key = jax.random.key(0)
    y0 = hw2d_random_ic(key, grid, amp=1e-3, include_neutrals=False)

    # Time stepping.
    dt = float(args.dt)
    nsteps = int(args.nsteps)
    save_stride = int(args.save_stride)

    print(f"[hw2d] grid=({grid.nx},{grid.ny}) dt={dt} nsteps={nsteps} save_stride={save_stride}")

    def rhs(t, y):
        return model.rhs(t, y)

    # Run in chunks so we can save diagnostic snapshots without storing everything.
    t = 0.0
    y = y0
    ts = []
    Es = []
    Zs = []
    snapshots = []

    nchunks = nsteps // save_stride
    for k in range(nchunks):
        _, y = rk4_scan(y, t0=t, dt=dt, nsteps=save_stride, rhs=rhs)
        t = t + dt * save_stride
        diag = model.diagnostics(y)
        ts.append(t)
        Es.append(float(diag["E"]))
        Zs.append(float(diag["Z"]))
        snapshots.append(
            {
                "n": jnp.asarray(y.n),
                "omega": jnp.asarray(y.omega),
            }
        )
        print(f"[hw2d] chunk {k+1}/{nchunks} t={t:.3f} E={Es[-1]:.3e} Z={Zs[-1]:.3e}")

    # Save parameters + time series.
    (out_dir / "params.json").write_text(
        json.dumps(
            {
                "grid": {"nx": grid.nx, "ny": grid.ny, "Lx": grid.Lx, "Ly": grid.Ly},
                "params": eqx_to_dict(params),
                "dt": dt,
                "nsteps": nsteps,
                "save_stride": save_stride,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    jnp.savez(
        out_dir / "timeseries.npz",
        t=jnp.array(ts),
        E=jnp.array(Es),
        Z=jnp.array(Zs),
    )

    # Plot time traces.
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

    # Final snapshot plots.
    phi = model.phi_from_omega(y.omega)
    fields = {"n": y.n, "phi": phi, "omega": y.omega}
    for name, arr in fields.items():
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        im = ax.imshow(arr.T, origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    # Simple spectrum of n.
    n_hat = jnp.fft.fft2(y.n)
    spec = jnp.abs(n_hat) ** 2
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(jnp.log10(spec.T + 1e-30), origin="lower", aspect="auto", cmap="magma")
    ax.set_title("log10 |n_k|^2")
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "spectrum_n.png", dpi=200)
    plt.close(fig)

    print(f"[hw2d] wrote results to {out_dir}")


def eqx_to_dict(obj):
    # Small helper for example scripts; avoids importing private Equinox internals.
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [eqx_to_dict(x) for x in obj]
    if dataclasses.is_dataclass(obj):
        return {k: eqx_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if hasattr(obj, "__dict__"):
        return {k: eqx_to_dict(v) for k, v in vars(obj).items()}
    return str(obj)


if __name__ == "__main__":
    main()
