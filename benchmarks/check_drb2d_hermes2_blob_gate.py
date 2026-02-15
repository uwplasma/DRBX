"""Hermes-2 blob2d proxy gate (DRB2D)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState  # noqa: E402
from jaxdrb.nonlinear.grid import Grid2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.003)
    p.add_argument("--tmax", type=float, default=6.0)
    p.add_argument("--save-every", type=int, default=12)
    p.add_argument("--curvature", type=float, default=-(1.0 / (1.5**2)))
    p.add_argument("--Dn", type=float, default=1e-3)
    p.add_argument("--DOmega", type=float, default=1.2e-3)
    p.add_argument("--DTe", type=float, default=1e-3)
    p.add_argument("--mu-lin-n", type=float, default=0.0)
    p.add_argument("--mu-lin-omega", type=float, default=0.02)
    p.add_argument("--mu-lin-Te", type=float, default=0.0)
    p.add_argument("--bc-x", type=str, default="neumann")
    p.add_argument("--bc-y", type=str, default="periodic")
    p.add_argument("--poisson", type=str, default="cg_fd")
    p.add_argument("--poisson-preconditioner", type=str, default="spectral")
    p.add_argument("--poisson-cg-maxiter", type=int, default=120)
    p.add_argument("--poisson-cg-tol", type=float, default=5e-6)
    p.add_argument("--poisson-gauge-epsilon", type=float, default=1e-6)
    p.add_argument("--min-dx-cm", type=float, default=5e-3)
    p.add_argument("--min-mean-flux", type=float, default=1e-10)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def hermes_blob_profile(x: np.ndarray, y: np.ndarray, *, Lx: float, Ly: float) -> np.ndarray:
    sigma = 0.21 / 4.0
    x0 = 0.33
    y0 = 0.5
    xn = x / Lx
    yn = y / Ly
    blob = np.exp(-(((xn - x0) / sigma) ** 2)) * np.exp(-(((yn - y0) / sigma) ** 2))
    return 1.0 + 0.27 * blob


def blob_center(x: np.ndarray, n: np.ndarray, *, n0: float) -> float:
    n_fluct = n - n0
    pos = np.maximum(n_fluct, 0.0)
    denom = np.sum(pos) + 1e-12
    return float(np.sum(x * pos) / denom)


def radial_flux(n: np.ndarray, phi: np.ndarray, *, dy: float, n0: float) -> float:
    dphi_dy = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dy)
    v_ex = -dphi_dy
    n_fluct = n - n0
    return float(np.mean(n_fluct * v_ex))


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", True)

    grid = Grid2D.make(
        nx=int(args.nx),
        ny=int(args.ny),
        Lx=float(args.Lx),
        Ly=float(args.Ly),
        dealias=False,
        bc_x=str(args.bc_x),
        bc_y=str(args.bc_y),
    )
    poisson = str(args.poisson).lower()
    if poisson == "auto":
        poisson = "spectral" if (grid.bc.kind_x == 0 and grid.bc.kind_y == 0) else "cg_fd"

    params = DRB2DParams(
        log_n=False,
        log_Te=False,
        kpar=0.0,
        eta=0.0,
        me_hat=1.0,
        curvature_on=True,
        curvature_coeff=float(args.curvature),
        omega_n=0.0,
        omega_Te=0.0,
        sol_on=False,
        Dn=float(args.Dn),
        DOmega=float(args.DOmega),
        DTe=float(args.DTe),
        mu_lin_n=float(args.mu_lin_n),
        mu_lin_omega=float(args.mu_lin_omega),
        mu_lin_Te=float(args.mu_lin_Te),
        bracket="arakawa",
        bracket_zero_mean=bool(grid.bc.kind_x != 0 or grid.bc.kind_y != 0),
        poisson=poisson,
        poisson_preconditioner=str(args.poisson_preconditioner),
        poisson_cg_maxiter=int(args.poisson_cg_maxiter),
        poisson_cg_tol=float(args.poisson_cg_tol),
        poisson_gauge_epsilon=float(args.poisson_gauge_epsilon),
        dealias_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=True,
        operator_dissipative_on=True,
    )
    model = DRB2DModel(params=params, grid=grid)

    x = np.asarray(grid.x)[:, None]
    y = np.asarray(grid.y)[None, :]
    n0 = hermes_blob_profile(x, y, Lx=float(args.Lx), Ly=float(args.Ly))
    Te0 = 1.0 + 1.2 * (n0 - 1.0)
    omega0 = np.zeros_like(n0)
    v0 = np.zeros_like(n0)
    y0 = DRB2DState(
        n=jnp.asarray(n0),
        omega=jnp.asarray(omega0),
        vpar_e=jnp.asarray(v0),
        vpar_i=jnp.asarray(v0),
        Te=jnp.asarray(Te0),
    )

    dt = float(args.dt)
    nsteps = int(np.ceil(float(args.tmax) / dt))
    ys, _ = model.diffeqsolve_fixed_steps(
        y0=y0,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
        solver="dopri5",
        save_every=int(args.save_every),
        progress=False,
    )

    n_series = np.asarray(ys.n)
    omega_series = np.asarray(ys.omega)
    n_series = np.concatenate([n0[None, ...], n_series], axis=0)
    omega_series = np.concatenate([omega0[None, ...], omega_series], axis=0)

    finite_ok = bool(np.all(np.isfinite(n_series)) and np.all(np.isfinite(omega_series)))

    x_cm = []
    flux = []
    for n_i, w_i in zip(n_series, omega_series, strict=False):
        phi_i = np.asarray(model.phi_from_omega(jnp.asarray(w_i)))
        x_cm.append(blob_center(x, n_i, n0=1.0))
        flux.append(radial_flux(n_i, phi_i, dy=float(grid.dy), n0=1.0))
    x_cm = np.asarray(x_cm)
    flux = np.asarray(flux)

    tail = max(1, int(0.6 * len(x_cm)))
    dx_cm = float(np.abs(np.mean(x_cm[tail:]) - x_cm[0]))
    mean_flux = float(np.mean(flux[tail:]))

    metrics = {
        "finite_ok": finite_ok,
        "dx_cm": dx_cm,
        "mean_flux": mean_flux,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    ok = finite_ok and (dx_cm > float(args.min_dx_cm)) and (mean_flux > float(args.min_mean_flux))
    if not ok:
        raise SystemExit(f"Hermes2 blob gate failed: {metrics}")


if __name__ == "__main__":
    main()
