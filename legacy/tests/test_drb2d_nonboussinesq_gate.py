from __future__ import annotations

import numpy as np

import jax

from jaxdrb.nonlinear.drb2d import DRB2DModel, DRB2DParams, DRB2DState
from jaxdrb.nonlinear.grid import Grid2D


def test_drb2d_nonboussinesq_energy_rate_gate() -> None:
    grid = Grid2D.make(nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)

    params_nb = DRB2DParams(
        omega_n=0.0,
        omega_Te=0.0,
        kpar=0.0,
        eta=0.0,
        me_hat=0.2,
        curvature_on=False,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        boussinesq=False,
        non_boussinesq_perturbed_density_on=True,
        n0=1.0,
        n0_min=1e-6,
        bracket="arakawa",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    model_nb = DRB2DModel(params=params_nb, grid=grid)

    key = jax.random.key(0)
    shape = (grid.nx, grid.ny)
    amp = 1e-6
    y = DRB2DState(
        n=amp * jax.random.normal(key, shape),
        omega=amp * jax.random.normal(jax.random.key(1), shape),
        vpar_e=amp * jax.random.normal(jax.random.key(2), shape),
        vpar_i=amp * jax.random.normal(jax.random.key(3), shape),
        Te=amp * jax.random.normal(jax.random.key(4), shape),
    )

    rhs = model_nb.rhs(0.0, y)
    edot = float(model_nb.energy_rate(y, rhs))
    eps = 1e-6
    y_plus = DRB2DState(
        n=y.n + eps * rhs.n,
        omega=y.omega + eps * rhs.omega,
        vpar_e=y.vpar_e + eps * rhs.vpar_e,
        vpar_i=y.vpar_i + eps * rhs.vpar_i,
        Te=y.Te + eps * rhs.Te,
    )
    y_minus = DRB2DState(
        n=y.n - eps * rhs.n,
        omega=y.omega - eps * rhs.omega,
        vpar_e=y.vpar_e - eps * rhs.vpar_e,
        vpar_i=y.vpar_i - eps * rhs.vpar_i,
        Te=y.Te - eps * rhs.Te,
    )
    edot_fd = float((model_nb.energy(y_plus) - model_nb.energy(y_minus)) / (2 * eps))
    rel = abs(edot - edot_fd) / max(abs(edot_fd), 1e-12)
    assert rel < 5e-4

    params_b = DRB2DParams(
        omega_n=params_nb.omega_n,
        omega_Te=params_nb.omega_Te,
        kpar=params_nb.kpar,
        eta=params_nb.eta,
        me_hat=params_nb.me_hat,
        curvature_on=params_nb.curvature_on,
        curvature_coeff=params_nb.curvature_coeff,
        Dn=params_nb.Dn,
        DOmega=params_nb.DOmega,
        DTe=params_nb.DTe,
        boussinesq=True,
        non_boussinesq_perturbed_density_on=params_nb.non_boussinesq_perturbed_density_on,
        n0=params_nb.n0,
        n0_min=params_nb.n0_min,
        bracket=params_nb.bracket,
        poisson=params_nb.poisson,
        dealias_on=params_nb.dealias_on,
        operator_split_on=params_nb.operator_split_on,
    )
    model_b = DRB2DModel(params=params_b, grid=grid)
    E_b = float(model_b.energy(y))
    E_nb = float(model_nb.energy(y))
    rel_E = abs(E_b - E_nb) / max(abs(E_b), 1e-12)
    assert rel_E < 0.5
