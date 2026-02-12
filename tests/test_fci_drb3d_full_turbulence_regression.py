from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdrb.fci.drb3d_full import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.integrate import diffeqsolve_fixed_steps


def test_fci_drb3d_full_turbulence_statistics_regression() -> None:
    """Long-time regression gate for 3D FCI turbulence statistics.

    This guards against silent drift to either:
    - near-laminar decay (too little fluctuation level), or
    - purely zonal collapse (zonal fraction too large).
    """

    grid = FCISlabGrid.make(
        nx=18,
        ny=18,
        nz=8,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=4.0,
        Bx=0.0,
        By=0.25,
        Bz=1.0,
        open_field_line=False,
    )
    shape = (grid.nz, grid.nx, grid.ny)
    key = jax.random.key(1)
    k = jax.random.split(key, 5)
    amp = 5e-4
    y0 = FCIDRB3DFullState(
        n=amp * jax.random.normal(k[0], shape),
        omega=amp * jax.random.normal(k[1], shape),
        vpar_e=amp * jax.random.normal(k[2], shape),
        vpar_i=amp * jax.random.normal(k[3], shape),
        Te=amp * jax.random.normal(k[4], shape),
    )
    params = FCIDRB3DFullParams(
        omega_n=3.0,
        omega_Te=2.0,
        kappa=1.0,
        alpha=0.45,
        eta_par=0.03,
        Dn=2e-4,
        DOmega=2e-4,
        Dvpar=2e-4,
        DTe=2e-4,
        chi_par=4e-4,
        sheath_on=False,
        perp_operator="spectral",
        bracket="arakawa",
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)

    ys, _ = diffeqsolve_fixed_steps(
        model.rhs,
        y0=y0,
        t0=0.0,
        dt=0.01,
        nsteps=800,
        save_every=20,
        solver="dopri5",
    )
    n = ys.n
    omega = ys.omega
    assert bool(jnp.isfinite(n).all())
    assert bool(jnp.isfinite(omega).all())

    n_fluct = n - jnp.mean(n, axis=(-1, -2), keepdims=True)
    omega_fluct = omega - jnp.mean(omega, axis=(-1, -2), keepdims=True)
    n_rms = jnp.sqrt(jnp.mean(n_fluct**2, axis=(-1, -2, -3)))
    omega_rms = jnp.sqrt(jnp.mean(omega_fluct**2, axis=(-1, -2, -3)))

    omega_zonal = jnp.mean(omega_fluct, axis=-1, keepdims=True)
    zonal_fraction = jnp.sqrt(jnp.mean(omega_zonal**2, axis=(-1, -2, -3))) / jnp.maximum(
        omega_rms, 1e-30
    )

    tail = slice(n_rms.shape[0] // 2, None)
    n_rms_mean = float(jnp.mean(n_rms[tail]))
    omega_rms_mean = float(jnp.mean(omega_rms[tail]))
    zonal_fraction_mean = float(jnp.mean(zonal_fraction[tail]))

    assert 3e-4 < n_rms_mean < 2.5e-3
    assert 1.0e-3 < omega_rms_mean < 4.5e-3
    assert 0.1 < zonal_fraction_mean < 0.5
