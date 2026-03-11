from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.geometry_2d import Geometry2DAdapter
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.terms.braginskii import (
    braginskii_friction_terms,
    braginskii_heat_exchange_terms,
    classical_diffusion_terms,
)
from jaxdrb.core.terms.context import build_context
from jaxdrb.geometry.plane import Grid2D
from jaxdrb.operators.fd2d import div_n_grad


def _make_geom(params: DRBSystemParams) -> Geometry2DAdapter:
    grid = Grid2D.make(
        nx=16,
        ny=16,
        Lx=2 * np.pi,
        Ly=2 * np.pi,
        dealias=False,
        bc_x="periodic",
        bc_y="periodic",
    )
    return Geometry2DAdapter(grid=grid, params=params)


def test_braginskii_heat_exchange_conserves_pair() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {"hot_ion_on": True},
            "transport": {
                "braginskii_heat_exchange_on": True,
                "braginskii_nu_ei": 2.0,
                "braginskii_nu_floor": 1e-8,
            },
        },
    )
    geom = _make_geom(params)
    n = jnp.ones((geom.grid.nx, geom.grid.ny))
    Te = jnp.ones_like(n)
    Ti = 2.0 * jnp.ones_like(n)
    state = DRBSystemState(
        n=n,
        omega=jnp.zeros_like(n),
        vpar_e=jnp.zeros_like(n),
        vpar_i=jnp.zeros_like(n),
        Te=Te,
        Ti=Ti,
        psi=None,
        N=None,
    )
    ctx = build_context(params, geom, state)
    term = braginskii_heat_exchange_terms(ctx, state)

    A_e = float(params.me_hat)
    A_i = 1.0
    nu = float(params.braginskii_nu_ei)
    Ti_eff = float(params.tau_i) + Ti
    Q_ei = 3.0 * (A_e / (A_e + A_i)) * nu * n * (Ti_eff - Te)
    dTe_expected = (2.0 / 3.0) * Q_ei / n
    dTi_expected = -(2.0 / 3.0) * Q_ei / n

    assert jnp.allclose(term.Te, dTe_expected, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(term.Ti, dTi_expected, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(term.Te + term.Ti, 0.0, rtol=1e-12, atol=1e-12)


def test_braginskii_friction_conserves_momentum() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {"hot_ion_on": True},
            "transport": {
                "braginskii_friction_on": True,
                "braginskii_nu_ei": 1.0,
                "braginskii_friction_coeff": 0.5,
            },
        },
    )
    geom = _make_geom(params)
    n = jnp.ones((geom.grid.nx, geom.grid.ny))
    vpar_e = jnp.zeros_like(n)
    vpar_i = jnp.ones_like(n)
    state = DRBSystemState(
        n=n,
        omega=jnp.zeros_like(n),
        vpar_e=vpar_e,
        vpar_i=vpar_i,
        Te=jnp.ones_like(n),
        Ti=jnp.ones_like(n),
        psi=None,
        N=None,
    )
    ctx = build_context(params, geom, state)
    term = braginskii_friction_terms(ctx, state)

    coeff = float(params.braginskii_friction_coeff)
    nu = float(params.braginskii_nu_ei)
    dv = vpar_i - vpar_e
    dvpar_e_expected = coeff * float(params.me_hat) * nu * dv
    dvpar_i_expected = -dvpar_e_expected

    assert jnp.allclose(term.vpar_e, dvpar_e_expected, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(term.vpar_i, dvpar_i_expected, rtol=1e-12, atol=1e-12)


def test_classical_diffusion_custom_coeffs() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {"hot_ion_on": True},
            "transport": {
                "classical_diffusion_on": True,
                "classical_diffusion_custom_D": 0.2,
                "classical_diffusion_custom_kappa_e": 0.3,
                "classical_diffusion_custom_kappa_i": 0.4,
            },
        },
    )
    geom = _make_geom(params)
    x = jnp.asarray(geom.grid.x)[:, None]
    y = jnp.asarray(geom.grid.y)[None, :]
    n = 1.0 + 0.1 * jnp.cos(x)
    Te = 1.0 + 0.2 * jnp.sin(y)
    Ti = 1.0 + 0.3 * jnp.sin(x)
    state = DRBSystemState(
        n=n,
        omega=jnp.zeros_like(n),
        vpar_e=jnp.zeros_like(n),
        vpar_i=jnp.zeros_like(n),
        Te=Te,
        Ti=Ti,
        psi=None,
        N=None,
    )
    ctx = build_context(params, geom, state)
    term = classical_diffusion_terms(ctx, state)

    D = float(params.classical_diffusion_custom_D)
    expected_dn = div_n_grad(
        n,
        jnp.full_like(n, D),
        dx=geom.grid.dx,
        dy=geom.grid.dy,
        bc=ctx.bcs.n,
    )
    n_eff = jnp.maximum(ctx.n_phys, float(params.n0_min))
    expected_dTe = (
        div_n_grad(
            Te,
            jnp.full_like(Te, float(params.classical_diffusion_custom_kappa_e)),
            dx=geom.grid.dx,
            dy=geom.grid.dy,
            bc=ctx.bcs.Te,
        )
        / n_eff
    )
    expected_dTi = (
        div_n_grad(
            Ti,
            jnp.full_like(Ti, float(params.classical_diffusion_custom_kappa_i)),
            dx=geom.grid.dx,
            dy=geom.grid.dy,
            bc=ctx.bcs.Ti,
        )
        / n_eff
    )

    assert jnp.allclose(term.n, expected_dn, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(term.Te, expected_dTe, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(term.Ti, expected_dTi, rtol=1e-12, atol=1e-12)
