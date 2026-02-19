from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.fci.drb3d import FCIDRB3DModel, FCIDRB3DParams, FCIDRB3DState
from jaxdrb.fci.grid import FCISlabGrid
from jaxdrb.nonlinear.fd import div_n_grad


def test_fci_drb3d_nonboussinesq_polarization_spd_identity() -> None:
    grid = FCISlabGrid.make(
        nx=28,
        ny=24,
        nz=4,
        Lx=2 * math.pi,
        Ly=2 * math.pi,
        Lz=2.0,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
        open_field_line=False,
    )
    params = FCIDRB3DParams(
        kappa=0.0,
        alpha=0.0,
        kpar=0.0,
        Dn=0.0,
        DOmega=0.0,
        bracket="arakawa",
        poisson="fd_cg",
        boussinesq=False,
        non_boussinesq_perturbed_density_on=True,
        n0=1.0,
        n0_min=1e-6,
        poisson_preconditioner="spectral",
        poisson_maxiter=500,
        poisson_tol=1e-11,
        dealias_on=False,
        sheath_nu=0.0,
    )
    model = FCIDRB3DModel(params=params, grid=grid)
    bc = BC2D.periodic()

    key = jax.random.key(0)
    amp = 1e-2
    n = amp * jax.random.normal(key, (grid.nz, grid.nx, grid.ny))
    omega = amp * jax.random.normal(jax.random.key(1), (grid.nz, grid.nx, grid.ny))
    omega0 = omega - jnp.mean(omega, axis=(1, 2), keepdims=True)
    y = FCIDRB3DState(n=n, omega=omega)

    phi = model._phi_from_omega(y.omega, n=y.n)

    n_eff = jnp.maximum(params.n0 + y.n, params.n0_min)

    def residual_plane(phi_p, n_p, om_p):
        return -div_n_grad(phi_p, n_p, grid.dx, grid.dy, bc) - om_p

    res = jax.vmap(residual_plane)(phi, n_eff, omega0)
    rel_res = jnp.sqrt(jnp.mean(res**2)) / jnp.maximum(jnp.sqrt(jnp.mean(omega0**2)), 1e-14)
    assert float(rel_res) < 2e-6

    def spd_mismatch_plane(phi_p, n_p, om_p):
        lhs = jnp.mean(phi_p * om_p)
        du_xp = (jnp.roll(phi_p, -1, axis=0) - phi_p) / grid.dx
        du_yp = (jnp.roll(phi_p, -1, axis=1) - phi_p) / grid.dy
        n_xp = 0.5 * (n_p + jnp.roll(n_p, -1, axis=0))
        n_yp = 0.5 * (n_p + jnp.roll(n_p, -1, axis=1))
        rhs = jnp.mean(n_xp * du_xp**2 + n_yp * du_yp**2)
        return jnp.abs(lhs - rhs) / jnp.maximum(jnp.abs(rhs), 1e-14)

    mismatch = jax.vmap(spd_mismatch_plane)(phi, n_eff, omega0)
    assert float(jnp.max(mismatch)) < 2e-6
