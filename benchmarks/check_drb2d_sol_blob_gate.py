#!/usr/bin/env python3
"""SOL-style DRB2D blob-transport gate (closed→open radial setup)."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.analysis.plotting import save_json
from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def _blob_center_x(n: np.ndarray, x: np.ndarray, mask_open: np.ndarray) -> float:
    n_fluct = n - np.mean(n)
    n_pos = np.maximum(n_fluct, 0.0) * mask_open
    denom = np.sum(n_pos) + 1e-30
    return float(np.sum(n_pos * x) / denom)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=48)
    p.add_argument("--ny", type=int, default=48)
    p.add_argument("--dt", type=float, default=0.015)
    p.add_argument("--tmax", type=float, default=80.0)
    p.add_argument("--save-every", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=Path, default=None)

    p.add_argument("--xs-frac", type=float, default=0.6)
    p.add_argument("--sol-width", type=float, default=0.08)
    p.add_argument("--min-blob-velocity", type=float, default=2e-3)
    p.add_argument("--min-mean-flux", type=float, default=5e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=2 * jnp.pi, Ly=2 * jnp.pi, dealias=False)
    xs = float(args.xs_frac) * float(grid.Lx)
    params = DRB2DParams(
        omega_n=0.8,
        omega_Te=0.25,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=True,
        curvature_coeff=0.7,
        Dn=3e-3,
        DOmega=3e-3,
        DTe=3e-3,
        Dn4=6e-5,
        DOmega4=6e-5,
        DTe4=6e-5,
        mu_zonal_omega=0.08,
        mu_lin_n=0.1,
        mu_lin_omega=0.25,
        mu_lin_Te=0.1,
        bracket="arakawa",
        poisson="spectral",
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=True,
        operator_dissipative_on=True,
        sol_on=True,
        sol_xs=xs,
        sol_width=float(args.sol_width) * float(grid.Lx),
        sol_n_core=1.0,
        sol_n_sol=0.2,
        sol_Te_core=1.0,
        sol_Te_sol=0.25,
        sol_relax_core=0.08,
        sol_relax_open=0.25,
        sol_sink_open_n=0.08,
        sol_sink_open_Te=0.05,
        sol_sink_open_omega=0.02,
    )
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(int(args.seed))
    amp = 5e-3
    shape = (grid.nx, grid.ny)
    y0 = DRB2DState(
        n=amp * jax.random.normal(key, shape),
        omega=amp * jax.random.normal(jax.random.key(int(args.seed) + 1), shape),
        vpar_e=amp * jax.random.normal(jax.random.key(int(args.seed) + 2), shape),
        vpar_i=amp * jax.random.normal(jax.random.key(int(args.seed) + 3), shape),
        Te=amp * jax.random.normal(jax.random.key(int(args.seed) + 4), shape),
    )

    save_ts = jnp.arange(float(args.save_every), float(args.tmax) + 1e-12, float(args.save_every))
    sol = model.diffeqsolve(
        y0=y0,
        t0=0.0,
        t1=float(args.tmax),
        dt0=float(args.dt),
        save_ts=save_ts,
        solver="tsit5",
        adaptive=True,
        rtol=1e-5,
        atol=1e-8,
        max_steps=400_000,
        progress=False,
    )

    omega_ts = jnp.real(sol.ys.omega)
    finite_ok = bool(jnp.all(jnp.isfinite(omega_ts)))

    x = np.asarray(grid.x)[:, None]
    mask_open = (x > xs).astype(float)
    flux = []
    x_cm = []
    for i in range(omega_ts.shape[0]):
        n_i = np.asarray(jax.device_get(sol.ys.n[i]))
        phi_i = np.asarray(jax.device_get(model.phi_from_omega(sol.ys.omega[i], n=sol.ys.n[i])))
        vEx = -np.gradient(phi_i, float(grid.dy), axis=1)
        flux.append(float(np.mean(n_i * vEx * mask_open)))
        x_cm.append(_blob_center_x(n_i, x, mask_open))
    flux = np.asarray(flux)
    x_cm = np.asarray(x_cm)

    tail = slice(int(2 * len(save_ts) / 3), None)
    coeffs = np.polyfit(np.asarray(save_ts)[tail], x_cm[tail], deg=1)
    v_blob = float(coeffs[0])
    mean_flux_tail = float(np.mean(flux[tail]))

    metrics = {
        "finite_ok": finite_ok,
        "blob_velocity_tail": v_blob,
        "mean_flux_tail": mean_flux_tail,
        "xs": xs,
    }
    print(
        "[drb2d-sol-gate] "
        f"finite={metrics['finite_ok']} "
        f"v_blob={metrics['blob_velocity_tail']:.3e} "
        f"flux={metrics['mean_flux_tail']:.3e}"
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.json_out, metrics)

    failures: list[str] = []
    if not metrics["finite_ok"]:
        failures.append("state contains non-finite values")
    if metrics["blob_velocity_tail"] < float(args.min_blob_velocity):
        failures.append(
            f"blob_velocity_tail {metrics['blob_velocity_tail']:.3e} < {float(args.min_blob_velocity):.3e}"
        )
    if metrics["mean_flux_tail"] < float(args.min_mean_flux):
        failures.append(
            f"mean_flux_tail {metrics['mean_flux_tail']:.3e} < {float(args.min_mean_flux):.3e}"
        )

    if failures:
        raise SystemExit(" | ".join(failures))


if __name__ == "__main__":
    main()
