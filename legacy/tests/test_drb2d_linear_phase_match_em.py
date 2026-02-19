from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from jaxdrb.linear.arnoldi import arnoldi_eigs
from jaxdrb.linear.growthrate import estimate_growth_rate
from jaxdrb.nonlinear.drb2d_em import DRB2DEMModel, DRB2DEMParams, DRB2DEMState
from jaxdrb.nonlinear.grid import Grid2D

    def kperp2(self, kx: float, ky: float) -> jnp.ndarray:  # type: ignore[override]
        return jnp.asarray([self.kperp2_value])

    def dpar(self, f: jnp.ndarray) -> jnp.ndarray:
        return 1j * float(self.kpar) * f

    def curvature(self, kx: float, ky: float, f: jnp.ndarray) -> jnp.ndarray:
        if self.curvature_coeff == 0.0:
            return jnp.zeros_like(f)
        return -1j * float(self.curvature_coeff) * float(ky) * f


def _drb2d_growth_rate(*, kx: float, ky: float, kpar: float, curvature_coeff: float) -> float:
    grid = Grid2D.make(nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    params = DRB2DEMParams(
        omega_n=0.8,
        omega_Te=0.3,
        kpar=kpar,
        eta=0.5,
        me_hat=0.2,
        beta=0.2,
        Dpsi=0.0,
        curvature_on=True,
        curvature_coeff=curvature_coeff,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="spectral",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    model = DRB2DEMModel(params=params, grid=grid)

    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mode = np.exp(1j * (kx * X + ky * Y))
    amp = 1e-6
    v0 = DRB2DEMState(
        n=jnp.asarray(amp * mode),
        omega=jnp.asarray(amp * mode),
        vpar_e=jnp.zeros_like(jnp.asarray(mode)),
        psi=jnp.asarray(amp * mode),
        vpar_i=jnp.zeros_like(jnp.asarray(mode)),
        Te=jnp.asarray(amp * mode),
    )

    zero = jnp.zeros((grid.nx, grid.ny), dtype=jnp.complex128)
    y_zero = DRB2DEMState(n=zero, omega=zero, vpar_e=zero, psi=zero, vpar_i=zero, Te=zero)
    _, jvp_fn = jax.linearize(lambda y: model.rhs(0.0, y), y_zero)
    res = estimate_growth_rate(jvp_fn, v0, tmax=20.0, dt0=0.02, nsave=120, fit_window=0.5)
    return float(res.gamma)


def test_drb2d_linear_phase_matches_linear_solver_em() -> None:
    kx = 1.0
    ky = 1.0
    kpar = 0.2
    curvature_coeff = 0.4

    grid = Grid2D.make(nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    params = DRB2DEMParams(
        omega_n=0.8,
        omega_Te=0.3,
        kpar=kpar,
        eta=0.5,
        me_hat=0.2,
        beta=0.2,
        Dpsi=0.0,
        curvature_on=True,
        curvature_coeff=curvature_coeff,
        Dn=0.0,
        DOmega=0.0,
        DTe=0.0,
        bracket="spectral",
        poisson="spectral",
        dealias_on=False,
        operator_split_on=False,
    )
    model = DRB2DEMModel(params=params, grid=grid)
    zero = jnp.zeros((grid.nx, grid.ny), dtype=jnp.complex128)
    y_zero = DRB2DEMState(n=zero, omega=zero, vpar_e=zero, psi=zero, vpar_i=zero, Te=zero)
    _, lin = jax.linearize(lambda y: model.rhs(0.0, y), y_zero)

    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mode = np.exp(1j * (kx * X + ky * Y))
    amp = 1e-6
    v0 = DRB2DEMState(
        n=jnp.asarray(amp * mode),
        omega=jnp.asarray(amp * mode),
        vpar_e=jnp.zeros_like(jnp.asarray(mode)),
        psi=jnp.asarray(amp * mode),
        vpar_i=jnp.zeros_like(jnp.asarray(mode)),
        Te=jnp.asarray(amp * mode),
    )
    arn = arnoldi_eigs(lin, v0, m=30, nev=6, seed=0)
    gamma_lin = float(np.max(arn.eigenvalues.real))

    gamma_drb2d = _drb2d_growth_rate(kx=kx, ky=ky, kpar=kpar, curvature_coeff=curvature_coeff)

    rel_err = abs(gamma_drb2d - gamma_lin) / max(abs(gamma_lin), 1e-12)
    assert rel_err < 0.15
