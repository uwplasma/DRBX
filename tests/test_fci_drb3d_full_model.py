from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxdrb.fci.drb3d_full import FCIDRB3DFullModel, FCIDRB3DFullParams, FCIDRB3DFullState
from jaxdrb.fci.grid import FCISlabGrid


def _rand_state(key, shape) -> FCIDRB3DFullState:
    k = jax.random.split(key, 5)
    return FCIDRB3DFullState(
        n=jax.random.normal(k[0], shape),
        omega=jax.random.normal(k[1], shape),
        vpar_e=jax.random.normal(k[2], shape),
        vpar_i=jax.random.normal(k[3], shape),
        Te=jax.random.normal(k[4], shape),
    )


def test_fci_drb3d_full_conservative_gate_periodic() -> None:
    grid = FCISlabGrid.make(
        nx=18,
        ny=16,
        nz=8,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=4.0,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
        open_field_line=False,
    )
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.0,
        Dn=0.0,
        DOmega=0.0,
        Dvpar=0.0,
        DTe=0.0,
        chi_par=0.0,
        sheath_on=False,
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)

    key = jax.random.key(7)
    base = _rand_state(key, (1, grid.nx, grid.ny))
    y0 = FCIDRB3DFullState(
        n=jnp.broadcast_to(base.n, (grid.nz, grid.nx, grid.ny)),
        omega=jnp.broadcast_to(base.omega, (grid.nz, grid.nx, grid.ny)),
        vpar_e=jnp.zeros((grid.nz, grid.nx, grid.ny)),
        vpar_i=jnp.zeros((grid.nz, grid.nx, grid.ny)),
        Te=jnp.zeros((grid.nz, grid.nx, grid.ny)),
    )

    dy = model.rhs(0.0, y0)
    erate = float(model.energy_rate(y0, dy))
    prate = float(model.particle_rate(dy))
    crate = float(jnp.mean(dy.vpar_i - dy.vpar_e))

    assert abs(prate) < 1e-11
    assert abs(crate) < 1e-11
    assert abs(erate) < 5e-9


def test_fci_drb3d_full_sheath_budget_consistency() -> None:
    grid = FCISlabGrid.make(
        nx=14,
        ny=12,
        nz=16,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=6.0,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
        open_field_line=True,
        cell_centered=True,
    )
    params = FCIDRB3DFullParams(
        omega_n=0.0,
        omega_Te=0.0,
        kappa=0.0,
        alpha=0.0,
        eta_par=0.0,
        Dn=0.0,
        DOmega=0.0,
        Dvpar=0.0,
        DTe=0.0,
        chi_par=0.0,
        sheath_on=True,
        sheath_nu_mom=1.4,
        sheath_nu_particle=0.8,
        sheath_nu_energy=0.6,
        sheath_gamma_e=3.0,
    )
    model = FCIDRB3DFullModel(params=params, grid=grid)

    key = jax.random.key(13)
    base = _rand_state(key, (1, grid.nx, grid.ny))
    y0 = FCIDRB3DFullState(
        n=jnp.broadcast_to(base.n, (grid.nz, grid.nx, grid.ny)),
        omega=jnp.zeros((grid.nz, grid.nx, grid.ny)),
        vpar_e=jnp.broadcast_to(base.vpar_e, (grid.nz, grid.nx, grid.ny)),
        vpar_i=jnp.broadcast_to(base.vpar_i, (grid.nz, grid.nx, grid.ny)),
        Te=jnp.abs(jnp.broadcast_to(base.Te, (grid.nz, grid.nx, grid.ny))) + 0.2,
    )

    dy = model.rhs(0.0, y0)
    p_rate = float(model.particle_rate(dy))
    e_rate = float(model.energy_rate(y0, dy))
    p_sh, e_sh = model.sheath_budget_rates(y0)

    assert p_rate < 0.0
    assert e_rate < 0.0
    assert abs(p_rate - float(p_sh)) < 1e-10
    assert abs(e_rate - float(e_sh)) < 5e-9


def test_fci_drb3d_full_operator_split_reconstruction() -> None:
    grid = FCISlabGrid.make(
        nx=10,
        ny=10,
        nz=8,
        Lx=2.0 * jnp.pi,
        Ly=2.0 * jnp.pi,
        Lz=4.0,
        Bx=0.0,
        By=0.0,
        Bz=1.0,
        open_field_line=True,
        cell_centered=True,
    )
    key = jax.random.key(101)
    y0 = _rand_state(key, (grid.nz, grid.nx, grid.ny))
    params_full = FCIDRB3DFullParams(
        omega_n=0.5,
        omega_Te=0.4,
        kappa=0.6,
        alpha=0.2,
        eta_par=0.1,
        Dn=1e-3,
        DOmega=1e-3,
        Dvpar=1e-3,
        DTe=1e-3,
        chi_par=2e-2,
        sheath_on=True,
        sheath_nu_mom=0.7,
        sheath_nu_particle=0.5,
        sheath_nu_energy=0.4,
        operator_split_on=False,
    )
    model_full = FCIDRB3DFullModel(params=params_full, grid=grid)
    dy_full = model_full.rhs(0.0, y0)

    params_split = eqx.tree_at(lambda p: p.operator_split_on, params_full, True)
    model_split = FCIDRB3DFullModel(params=params_split, grid=grid)
    dy_split = model_split.rhs(0.0, y0)

    err = jnp.sqrt(
        jnp.mean(
            (dy_full.n - dy_split.n) ** 2
            + (dy_full.omega - dy_split.omega) ** 2
            + (dy_full.vpar_e - dy_split.vpar_e) ** 2
            + (dy_full.vpar_i - dy_split.vpar_i) ** 2
            + (dy_full.Te - dy_split.Te) ** 2
        )
    )
    assert float(err) < 1e-11
