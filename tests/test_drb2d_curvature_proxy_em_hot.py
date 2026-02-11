from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxdrb.linear.growthrate import estimate_growth_rate
from jaxdrb.nonlinear.drb2d_em import DRB2DEMModel, DRB2DEMParams, DRB2DEMState
from jaxdrb.nonlinear.drb2d_hot_ion import (
    DRB2DHotIonModel,
    DRB2DHotIonParams,
    DRB2DHotIonState,
)
from jaxdrb.nonlinear.grid import Grid2D


def _linear_gamma_hot(model: DRB2DHotIonModel, v0: DRB2DHotIonState) -> float:
    zero = jnp.zeros_like(v0.n)
    y_zero = DRB2DHotIonState(n=zero, omega=zero, vpar_e=zero, vpar_i=zero, Te=zero, Ti=zero)
    _, jvp_fn = jax.linearize(lambda y: model.rhs(0.0, y), y_zero)
    res = estimate_growth_rate(jvp_fn, v0, tmax=15.0, dt0=0.02, nsave=120, fit_window=0.5)
    return float(res.gamma)


def _linear_gamma_em(model: DRB2DEMModel, v0: DRB2DEMState) -> float:
    zero = jnp.zeros_like(v0.n)
    y_zero = DRB2DEMState(n=zero, omega=zero, psi=zero, vpar_i=zero, Te=zero)
    _, jvp_fn = jax.linearize(lambda y: model.rhs(0.0, y), y_zero)
    res = estimate_growth_rate(jvp_fn, v0, tmax=15.0, dt0=0.02, nsave=120, fit_window=0.5)
    return float(res.gamma)


def test_em_hot_drive_threshold_tracks_proxy() -> None:
    grid = Grid2D.make(nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dealias=False)
    x = np.asarray(grid.x)
    y = np.asarray(grid.y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx = 0.0
    ky = 1.0
    mode = np.exp(1j * (kx * X + ky * Y))
    amp = 1e-6
    v0_hot = DRB2DHotIonState(
        n=jnp.asarray(amp * mode),
        omega=jnp.asarray(amp * mode),
        vpar_e=jnp.zeros_like(jnp.asarray(mode)),
        vpar_i=jnp.zeros_like(jnp.asarray(mode)),
        Te=jnp.asarray(amp * mode),
        Ti=jnp.asarray(amp * mode),
    )
    v0_em = DRB2DEMState(
        n=jnp.asarray(amp * mode),
        omega=jnp.asarray(amp * mode),
        psi=jnp.asarray(amp * mode),
        vpar_i=jnp.zeros_like(jnp.asarray(mode)),
        Te=jnp.asarray(amp * mode),
    )

    g_ref = 0.3
    omega_crit = g_ref * ky * (1.0 + (kx**2 + ky**2) / 4.0)
    omega_low = 0.5 * omega_crit
    omega_high = 1.5 * omega_crit

    def gamma_hot(omega_n: float) -> float:
        params_hot = DRB2DHotIonParams(
            omega_n=float(omega_n),
            omega_Te=0.0,
            omega_Ti=0.0,
            kpar=0.0,
            eta=0.0,
            me_hat=0.2,
            tau_i=1.0,
            curvature_on=True,
            curvature_coeff=float(g_ref),
            Dn=0.0,
            DOmega=0.0,
            DTe=0.0,
            DTi=0.0,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        return _linear_gamma_hot(DRB2DHotIonModel(params=params_hot, grid=grid), v0_hot)

    def gamma_em(omega_n: float) -> float:
        params_em = DRB2DEMParams(
            omega_n=float(omega_n),
            omega_Te=0.0,
            kpar=0.0,
            eta=0.0,
            me_hat=0.2,
            beta=0.2,
            Dpsi=0.0,
            curvature_on=True,
            curvature_coeff=float(g_ref),
            Dn=0.0,
            DOmega=0.0,
            DTe=0.0,
            bracket="arakawa",
            poisson="spectral",
            dealias_on=False,
        )
        return _linear_gamma_em(DRB2DEMModel(params=params_em, grid=grid), v0_em)

    def gamma_proxy(omega_n: float) -> float:
        kperp2 = kx**2 + ky**2
        b = -g_ref * ky
        c = -(g_ref * ky / kperp2) * (g_ref * ky - omega_n)
        disc = b * b - 4.0 * c
        root = np.sqrt(disc + 0.0j)
        w1 = 0.5 * (g_ref * ky + root)
        w2 = 0.5 * (g_ref * ky - root)
        return float(max(w1.imag, w2.imag))

    gamma_hot_low = gamma_hot(omega_low)
    gamma_hot_high = gamma_hot(omega_high)
    gamma_em_low = gamma_em(omega_low)
    gamma_em_high = gamma_em(omega_high)

    assert gamma_hot_high > gamma_hot_low
    assert gamma_em_high > gamma_em_low

    omega_scan = np.linspace(0.0, 0.9, 5)
    gamma_hot_scan = np.asarray([gamma_hot(float(w)) for w in omega_scan])
    gamma_em_scan = np.asarray([gamma_em(float(w)) for w in omega_scan])
    gamma_proxy_scan = np.asarray([gamma_proxy(float(w)) for w in omega_scan])

    corr_hot = np.corrcoef(gamma_hot_scan, gamma_proxy_scan)[0, 1]
    corr_em = np.corrcoef(gamma_em_scan, gamma_proxy_scan)[0, 1]
    assert corr_hot > 0.4
    assert corr_em > 0.4
