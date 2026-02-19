from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps
from jaxdrb.nonlinear.spectral import ddx as ddx_spec
from jaxdrb.nonlinear.spectral import ddy as ddy_spec
from jaxdrb.nonlinear.spectral import inv_laplacian


def _omega_shear(
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


def _energy_enstrophy(omega: jnp.ndarray, *, grid: Grid2D) -> tuple[jnp.ndarray, jnp.ndarray]:
    phi = inv_laplacian(omega, grid.k2, k2_min=1e-6)
    dphi_dx = ddx_spec(phi, grid.kx)
    dphi_dy = ddy_spec(phi, grid.ky)
    energy = 0.5 * jnp.mean(dphi_dx**2 + dphi_dy**2)
    enstrophy = 0.5 * jnp.mean(omega**2)
    return energy, enstrophy


def test_kh2d_energy_enstrophy_decay() -> None:
    jax.config.update("jax_enable_x64", True)

    nx, ny = 48, 96
    Lx, Ly = 2.0 * np.pi, 2.0 * np.pi
    grid = Grid2D.make(nx=nx, ny=ny, Lx=Lx, Ly=Ly, dealias=False)

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
        DOmega=3e-3,
        Dn4=0.0,
        DTe4=0.0,
        DOmega4=1e-6,
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
    omega0 = _omega_shear(
        x,
        y,
        Lx=Lx,
        Ly=Ly,
        u0=3.0,
        shear_width=0.12,
        pert_amp=0.1,
        pert_mode=4,
    )
    n0 = jnp.ones_like(omega0)
    Te0 = jnp.ones_like(omega0)
    v0 = jnp.zeros_like(omega0)
    y0 = DRB2DState(n=n0, omega=omega0, vpar_e=v0, vpar_i=v0, Te=Te0)

    dt = 0.02
    nsteps = int(np.ceil(6.0 / dt))
    ys, _ = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
        solver="dopri5",
        save_every=30,
        progress=False,
    )
    omega_series = ys.omega
    e0, z0 = _energy_enstrophy(omega_series[0], grid=grid)
    e1, z1 = _energy_enstrophy(omega_series[-1], grid=grid)

    assert jnp.isfinite(e0) and jnp.isfinite(e1)
    assert jnp.isfinite(z0) and jnp.isfinite(z1)
    assert e1 < e0 * 0.99
    assert z1 < z0 * 0.97
