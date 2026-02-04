from __future__ import annotations

import math

import jax.numpy as jnp

from jaxdrb.nonlinear.grid import Grid2D
from jaxdrb.nonlinear.hw2d import HW2DModel, HW2DParams, HW2DState
from jaxdrb.nonlinear.stepper import rk4_step


def test_hw2d_single_mode_growth_matches_analytic_eigs() -> None:
    """A single Fourier mode evolves according to the analytic 2x2 linear operator.

    With a single mode present, the nonlinear Poisson bracket vanishes, so the full RHS reduces to
    a linear system for that mode. This is a standard verification pattern in drift-wave codes.
    """

    nx = 32
    ny = 32
    grid = Grid2D.make(nx=nx, ny=ny, Lx=2 * math.pi, Ly=2 * math.pi, dealias=False)
    params = HW2DParams(
        kappa=1.0,
        alpha=0.6,
        Dn=2e-3,
        DOmega=3e-3,
        nu4_n=0.0,
        nu4_omega=0.0,
        bracket="spectral",
        poisson="spectral",
        dealias_on=False,
    )
    model = HW2DModel(params=params, grid=grid)

    # Pick a Fourier mode and build real fields.
    i = 3
    j = 4
    k2 = float(grid.k2[i, j])
    ky = float(grid.ky[i, j])

    # Analytic 2x2 operator in (n_hat, w_hat).
    # dn = -kappa i ky phi + alpha(phi-n) - Dn k2 n
    # dw = -kappa i ky n + alpha(phi-n) - DOmega k2 w
    # phi = -w/k2
    a11 = -params.alpha - params.Dn * k2
    a12 = (-1j * params.kappa * ky) * (-1.0 / k2) + params.alpha * (-1.0 / k2)
    a21 = (-1j * params.kappa * ky) * 1.0 + (-params.alpha)
    a22 = params.alpha * (-1.0 / k2) - params.DOmega * k2
    A = jnp.array([[a11, a12], [a21, a22]], dtype=jnp.complex128)
    evals, evecs = jnp.linalg.eig(A)
    idx = int(jnp.argmax(jnp.real(evals)))
    lam = evals[idx]

    # Initialize exactly in the leading eigenvector to avoid transient mixing.
    amp = 1e-6
    v = evecs[:, idx]
    v = v / jnp.maximum(jnp.linalg.norm(v), 1e-30)
    n_hat0 = amp * v[0]
    w_hat0 = amp * v[1]

    hat_n = jnp.zeros((nx, ny), dtype=jnp.complex128).at[i, j].set(n_hat0)
    hat_w = jnp.zeros((nx, ny), dtype=jnp.complex128).at[i, j].set(w_hat0)
    hat_n = hat_n.at[(-i) % nx, (-j) % ny].set(jnp.conj(hat_n[i, j]))
    hat_w = hat_w.at[(-i) % nx, (-j) % ny].set(jnp.conj(hat_w[i, j]))

    n0 = jnp.fft.ifft2(hat_n).real
    w0 = jnp.fft.ifft2(hat_w).real
    y = HW2DState(n=n0, omega=w0)

    # Integrate for short time and measure amplification of the selected Fourier coefficient.
    dt = 0.02
    nsteps = 200  # t=4
    t = 0.0
    for _ in range(nsteps):
        y = rk4_step(y, t, dt, model.rhs)
        t += dt
    n_hat1 = jnp.fft.fft2(y.n)[i, j]

    # Compare growth rates from amplitude ratios.
    g_num = jnp.log(jnp.abs(n_hat1) / jnp.abs(n_hat0)) / t
    g_ex = jnp.real(lam)
    assert float(jnp.abs(g_num - g_ex)) < 2e-3
