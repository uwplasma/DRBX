from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.closures.neutrals import NeutralParams, rhs_neutral
from jaxdrb.core.geometry_2d import Geometry2DAdapter
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.terms.context import build_context
from jaxdrb.core.terms.neutrals import neutrals_terms
from jaxdrb.geometry.plane import Grid2D


def test_rhs_neutral_balances_ionization_recombination() -> None:
    N = jnp.ones((4, 4))
    n = 0.5 * jnp.ones_like(N)
    omega = jnp.ones_like(N)
    adv_N = jnp.zeros_like(N)
    lap_N = jnp.zeros_like(N)
    params = NeutralParams(
        enabled=True,
        Dn0=0.0,
        n_background=1.0,
        n_floor=1e-6,
        N_floor=1e-6,
        nu_ion=2.0,
        nu_rec=0.5,
        S0=0.0,
        nu_sink=0.0,
        nu_cx_omega=0.25,
    )

    dN, dn, domega = rhs_neutral(N=N, n=n, omega=omega, dn0=params, adv_N=adv_N, lap_N=lap_N)

    n_abs = params.n_background + n
    N_abs = N
    ion = params.nu_ion * n_abs * N_abs
    rec = params.nu_rec * n_abs
    expected_dN = -ion + rec
    expected_dn = ion - rec
    expected_domega = -params.nu_cx_omega * N_abs * omega

    assert jnp.allclose(dN, expected_dN, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(dn, expected_dn, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(domega, expected_domega, rtol=1e-12, atol=1e-12)


def test_neutrals_terms_disabled_returns_zero() -> None:
    params = DRBSystemParams()
    grid = Grid2D.make(
        nx=8,
        ny=8,
        Lx=2 * np.pi,
        Ly=2 * np.pi,
        dealias=False,
        bc_x="periodic",
        bc_y="periodic",
    )
    geom = Geometry2DAdapter(grid=grid, params=params)
    state = DRBSystemState(
        n=jnp.ones((8, 8)),
        omega=jnp.zeros((8, 8)),
        vpar_e=jnp.zeros((8, 8)),
        vpar_i=jnp.zeros((8, 8)),
        Te=jnp.ones((8, 8)),
        Ti=None,
        psi=None,
        N=jnp.ones((8, 8)),
    )
    ctx = build_context(params, geom, state)
    term = neutrals_terms(ctx, state)

    assert jnp.allclose(term.n, 0.0)
    assert jnp.allclose(term.omega, 0.0)
    assert jnp.allclose(term.N, 0.0)


def test_neutrals_terms_enabled_produces_exchange() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {"neutrals_on": True},
            "closure": {
                "neutrals": {
                    "enabled": True,
                    "nu_ion": 1.0,
                    "nu_rec": 0.2,
                    "nu_cx_omega": 0.1,
                }
            },
        },
    )
    grid = Grid2D.make(
        nx=8,
        ny=8,
        Lx=2 * np.pi,
        Ly=2 * np.pi,
        dealias=False,
        bc_x="periodic",
        bc_y="periodic",
    )
    geom = Geometry2DAdapter(grid=grid, params=params)
    state = DRBSystemState(
        n=jnp.ones((8, 8)),
        omega=jnp.ones((8, 8)),
        vpar_e=jnp.zeros((8, 8)),
        vpar_i=jnp.zeros((8, 8)),
        Te=jnp.ones((8, 8)),
        Ti=None,
        psi=None,
        N=jnp.ones((8, 8)),
    )
    ctx = build_context(params, geom, state)
    term = neutrals_terms(ctx, state)

    assert jnp.any(term.n != 0.0)
    assert jnp.any(term.N != 0.0)
    assert jnp.any(term.omega != 0.0)
    assert jnp.allclose(term.Te, 0.0)
    assert term.Ti is None
