from __future__ import annotations

import numpy as np

import jax

from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def _growth_rate(E: np.ndarray, t: np.ndarray) -> float:
    mask = t <= (0.5 * t[-1])
    slope, _ = np.polyfit(t[mask], np.log(np.maximum(E[mask], 1e-30)), 1)
    return 0.5 * float(slope)


def _run_growth(params: DRB2DParams) -> float:
    grid = Grid2D.make(nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    model = DRB2DModel(params=params, grid=grid)

    key = jax.random.key(0)
    shape = (grid.nx, grid.ny)
    n = 1e-6 * jax.random.normal(key, shape)
    omega = 1e-6 * jax.random.normal(jax.random.key(1), shape)
    vpar_e = jax.numpy.zeros_like(n)
    vpar_i = jax.numpy.zeros_like(n)
    Te = 1e-6 * jax.random.normal(jax.random.key(2), shape)
    y = DRB2DState(n=n, omega=omega, vpar_e=vpar_e, vpar_i=vpar_i, Te=Te)

    dt = 0.05
    nsteps = 120
    E = []
    t = []
    for k in range(nsteps):
        rhs = model.rhs(0.0, y)
        y = DRB2DState(
            n=y.n + dt * rhs.n,
            omega=y.omega + dt * rhs.omega,
            vpar_e=y.vpar_e + dt * rhs.vpar_e,
            vpar_i=y.vpar_i + dt * rhs.vpar_i,
            Te=y.Te + dt * rhs.Te,
        )
        E.append(float(model.energy(y)))
        t.append((k + 1) * dt)
    return _growth_rate(np.asarray(E), np.asarray(t))


def test_curvature_benchmark_interchange_and_resistive_trends() -> None:
    base = dict(
        omega_n=0.0,
        omega_Te=0.0,
        me_hat=0.2,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )

    gamma_interchange_off = _run_growth(
        DRB2DParams(
            **base,
            kpar=0.0,
            eta=0.0,
            curvature_on=True,
            curvature_coeff=0.0,
        )
    )
    gamma_interchange_on = _run_growth(
        DRB2DParams(
            **base,
            kpar=0.0,
            eta=0.0,
            curvature_on=True,
            curvature_coeff=0.5,
        )
    )
    assert gamma_interchange_on > gamma_interchange_off

    gamma_resistive_off = _run_growth(
        DRB2DParams(
            **base,
            kpar=0.3,
            eta=0.2,
            curvature_on=True,
            curvature_coeff=0.0,
        )
    )
    gamma_resistive_on = _run_growth(
        DRB2DParams(
            **base,
            kpar=0.3,
            eta=0.2,
            curvature_on=True,
            curvature_coeff=0.5,
        )
    )
    assert gamma_resistive_on > gamma_resistive_off
