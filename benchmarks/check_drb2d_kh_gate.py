"""Kelvin–Helmholtz 2D vorticity benchmark gate (DRB2D)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState  # noqa: E402
from jaxdrb.nonlinear.grid import Grid2D  # noqa: E402
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps  # noqa: E402
from jaxdrb.nonlinear.spectral import ddx as ddx_spec  # noqa: E402
from jaxdrb.nonlinear.spectral import ddy as ddy_spec  # noqa: E402
from jaxdrb.nonlinear.spectral import inv_laplacian  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=48)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--Lx", type=float, default=2.0 * jnp.pi)
    p.add_argument("--Ly", type=float, default=2.0 * jnp.pi)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--tmax", type=float, default=6.0)
    p.add_argument("--u0", type=float, default=3.0)
    p.add_argument("--shear-width", type=float, default=0.12)
    p.add_argument("--pert-amp", type=float, default=0.1)
    p.add_argument("--pert-mode", type=int, default=4)
    p.add_argument("--nu", type=float, default=3e-3)
    p.add_argument("--nu4", type=float, default=1e-6)
    p.add_argument("--max-energy-ratio", type=float, default=0.99)
    p.add_argument("--max-enstrophy-ratio", type=float, default=0.97)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def omega_shear(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    Lx: float,
    Ly: float,
    u0: float,
    shear_width: float,
    pert_amp: float,
    pert_mode: int,
) -> jnp.ndarray:
    y0 = 0.25 * Ly
    y1 = 0.75 * Ly
    a = float(shear_width)
    sech0 = 1.0 / jnp.cosh((y - y0) / a)
    sech1 = 1.0 / jnp.cosh((y - y1) / a)
    omega0 = (u0 / a) * (sech1**2 - sech0**2)
    if pert_amp != 0.0 and pert_mode > 0:
        envelope = jnp.exp(-(((y - y0) / a) ** 2)) + jnp.exp(-(((y - y1) / a) ** 2))
        omega0 = omega0 + pert_amp * u0 * envelope * jnp.sin(2.0 * jnp.pi * pert_mode * x / Lx)
    return omega0 - jnp.mean(omega0)


def energy_enstrophy(omega: jnp.ndarray, *, grid: Grid2D) -> tuple[jnp.ndarray, jnp.ndarray]:
    phi = inv_laplacian(omega, grid.k2, k2_min=1e-6)
    dphi_dx = ddx_spec(phi, grid.kx)
    dphi_dy = ddy_spec(phi, grid.ky)
    energy = 0.5 * jnp.mean(dphi_dx**2 + dphi_dy**2)
    enstrophy = 0.5 * jnp.mean(omega**2)
    return energy, enstrophy


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", True)

    grid = Grid2D.make(nx=args.nx, ny=args.ny, Lx=float(args.Lx), Ly=float(args.Ly), dealias=False)
    params = DRB2DParams(
        log_n=False,
        log_Te=False,
        kpar=0.0,
        eta=0.0,
        me_hat=1.0,
        curvature_on=False,
        curvature_coeff=0.0,
        omega_n=0.0,
        omega_Te=0.0,
        sol_on=False,
        Dn=0.0,
        DTe=0.0,
        DOmega=float(args.nu),
        Dn4=0.0,
        DTe4=0.0,
        DOmega4=float(args.nu4),
        mu_lin_n=0.0,
        mu_lin_Te=0.0,
        mu_lin_omega=0.0,
        mu_zonal_omega=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        k2_min=1e-6,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=True,
        operator_dissipative_on=True,
    )
    model = DRB2DModel(params=params, grid=grid)

    x = grid.x[:, None]
    y = grid.y[None, :]
    omega0 = omega_shear(
        x,
        y,
        Lx=float(args.Lx),
        Ly=float(args.Ly),
        u0=float(args.u0),
        shear_width=float(args.shear_width),
        pert_amp=float(args.pert_amp),
        pert_mode=int(args.pert_mode),
    )
    n0 = jnp.ones_like(omega0)
    Te0 = jnp.ones_like(omega0)
    v0 = jnp.zeros_like(omega0)
    y0 = DRB2DState(n=n0, omega=omega0, vpar_e=v0, vpar_i=v0, Te=Te0)

    dt = float(args.dt)
    nsteps = int(jnp.ceil(float(args.tmax) / dt))
    ys, _ = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
        solver="dopri5",
        save_every=max(1, int(nsteps // 3)),
        progress=False,
    )
    omega_series = ys.omega
    finite_ok = bool(jnp.all(jnp.isfinite(omega_series)))

    e0, z0 = energy_enstrophy(omega_series[0], grid=grid)
    e1, z1 = energy_enstrophy(omega_series[-1], grid=grid)
    e_ratio = float(e1 / (e0 + 1e-12))
    z_ratio = float(z1 / (z0 + 1e-12))

    metrics = {
        "finite_ok": finite_ok,
        "energy_ratio": e_ratio,
        "enstrophy_ratio": z_ratio,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    ok = (
        finite_ok
        and (e_ratio < float(args.max_energy_ratio))
        and (z_ratio < float(args.max_enstrophy_ratio))
    )
    if not ok:
        raise SystemExit(f"KH gate failed: {metrics}")


if __name__ == "__main__":
    main()
