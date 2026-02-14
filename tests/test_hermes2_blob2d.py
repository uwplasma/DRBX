from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def _hermes_blob_profile(x: np.ndarray, y: np.ndarray, *, Lx: float, Ly: float) -> np.ndarray:
    sigma = 0.21 / 4.0
    x0 = 0.33
    y0 = 0.5
    xn = x / Lx
    yn = y / Ly
    blob = np.exp(-(((xn - x0) / sigma) ** 2)) * np.exp(-(((yn - y0) / sigma) ** 2))
    return 1.0 + 0.27 * blob


def _blob_center(x: np.ndarray, n: np.ndarray, *, n0: float) -> float:
    n_fluct = n - n0
    pos = np.maximum(n_fluct, 0.0)
    denom = np.sum(pos) + 1e-12
    return float(np.sum(x * pos) / denom)


def test_hermes2_blob2d_propagates_outward() -> None:
    jax.config.update("jax_enable_x64", True)

    nx, ny = 48, 96
    Lx, Ly = 1.0, 1.0
    grid = Grid2D.make(nx=nx, ny=ny, Lx=Lx, Ly=Ly, dealias=False, bc_x="periodic", bc_y="periodic")

    params = DRB2DParams(
        log_n=False,
        log_Te=False,
        kpar=0.0,
        eta=0.0,
        me_hat=1.0,
        curvature_on=True,
        curvature_coeff=-(1.0 / (1.5**2)),
        omega_n=0.0,
        omega_Te=0.0,
        sol_on=False,
        Dn=3e-3,
        DOmega=3e-3,
        DTe=3e-3,
        mu_lin_n=0.05,
        mu_lin_omega=0.05,
        mu_lin_Te=0.05,
        bracket="arakawa",
        bracket_zero_mean=False,
        poisson="spectral",
        dealias_on=False,
        operator_split_on=True,
        operator_conservative_on=True,
        operator_source_on=True,
        operator_dissipative_on=True,
    )
    model = DRB2DModel(params=params, grid=grid)

    x = np.asarray(grid.x)[:, None]
    y = np.asarray(grid.y)[None, :]
    n0 = _hermes_blob_profile(x, y, Lx=Lx, Ly=Ly)
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

    dt = 0.002
    nsteps = int(np.ceil(6.0 / dt))
    ys, _ = model.diffeqsolve_fixed_steps(
        y0=y0,
        t0=0.0,
        dt=dt,
        nsteps=nsteps,
        solver="dopri5",
        save_every=20,
        progress=False,
    )
    n_series = np.asarray(ys.n)
    n_series = np.concatenate([n0[None, ...], n_series], axis=0)
    x_cm = np.array([_blob_center(x, n_i, n0=1.0) for n_i in n_series])

    assert np.all(np.isfinite(x_cm))
    assert float(np.abs(x_cm[-1] - x_cm[0])) > 0.005
