from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.geometry_2d import Geometry2DAdapter
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.terms.context import build_context
from jaxdrb.core.terms.diamagnetic import diamagnetic_current_terms, diamagnetic_terms
from jaxdrb.core.terms.ops import ddx
from jaxdrb.geometry.plane import Grid2D


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
    geom = Geometry2DAdapter(grid=grid, params=params)
    x = jnp.asarray(grid.x)
    curv_x = 1.0 + 0.2 * jnp.sin(x)
    curv_x = curv_x[:, None] * jnp.ones((1, grid.ny))
    curv_y = jnp.zeros_like(curv_x)
    object.__setattr__(geom, "curv_x", curv_x)
    object.__setattr__(geom, "curv_y", curv_y)
    return geom


def _make_state(grid: Grid2D) -> DRBSystemState:
    x = jnp.asarray(grid.x)[:, None]
    y = jnp.asarray(grid.y)[None, :]
    n = 1.0 + 0.2 * jnp.cos(x) + 0.1 * jnp.cos(y)
    Te = 1.0 + 0.3 * jnp.sin(x)
    z = jnp.zeros_like(n)
    return DRBSystemState(
        n=n,
        omega=z,
        vpar_e=z,
        vpar_i=z,
        Te=Te,
        Ti=None,
        psi=None,
        N=None,
    )


def _diamag_flux_expected(
    *,
    params: DRBSystemParams,
    geom: Geometry2DAdapter,
    f: jnp.ndarray,
    T: jnp.ndarray,
    q: float,
    curv_x: jnp.ndarray,
    diamag_form: float,
) -> jnp.ndarray:
    vdx = (T / q) * curv_x
    div_form = ddx(params, geom, f * vdx, geom.grid.bc)
    grad_form = curv_x * ddx(params, geom, f * T / q, geom.grid.bc)
    return diamag_form * div_form + (1.0 - diamag_form) * grad_form


def test_diamagnetic_form_mixing() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {
                "diamagnetic_on": True,
                "diamag_density_model": "electron",
                "diamag_form": 1.0,
            },
            "numerics": {"poisson": "spectral"},
        },
    )
    geom = _make_geom(params)
    y = _make_state(geom.grid)
    ctx = build_context(params, geom, y)

    term = diamagnetic_terms(ctx, y)
    curv_x = jnp.asarray(geom.curv_x)
    q_e = -1.0
    flux_div = _diamag_flux_expected(
        params=params,
        geom=geom,
        f=ctx.n_phys,
        T=ctx.Te_phys,
        q=q_e,
        curv_x=curv_x,
        diamag_form=1.0,
    )
    dn_div = -flux_div
    assert jnp.allclose(term.n, dn_div, rtol=1e-10, atol=1e-10)

    params_grad = update_params_from_dict(
        params,
        {"physics": {"diamag_form": 0.0}},
    )
    geom_grad = _make_geom(params_grad)
    y_grad = _make_state(geom_grad.grid)
    ctx_grad = build_context(params_grad, geom_grad, y_grad)
    term_grad = diamagnetic_terms(ctx_grad, y_grad)
    flux_grad = _diamag_flux_expected(
        params=params_grad,
        geom=geom_grad,
        f=ctx_grad.n_phys,
        T=ctx_grad.Te_phys,
        q=q_e,
        curv_x=jnp.asarray(geom_grad.curv_x),
        diamag_form=0.0,
    )
    dn_grad = -flux_grad
    assert jnp.allclose(term_grad.n, dn_grad, rtol=1e-10, atol=1e-10)
    assert not jnp.allclose(dn_div, dn_grad, rtol=1e-6, atol=1e-6)


def test_diamagnetic_pressure_to_temperature_conversion() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {
                "diamagnetic_on": True,
                "diamag_density_model": "electron",
                "diamag_form": 0.3,
            },
            "numerics": {"poisson": "spectral"},
        },
    )
    geom = _make_geom(params)
    y = _make_state(geom.grid)
    ctx = build_context(params, geom, y)

    term = diamagnetic_terms(ctx, y)
    curv_x = jnp.asarray(geom.curv_x)
    q_e = -1.0
    diamag_form = float(params.diamag_form)

    flux_n = _diamag_flux_expected(
        params=params,
        geom=geom,
        f=ctx.n_phys,
        T=ctx.Te_phys,
        q=q_e,
        curv_x=curv_x,
        diamag_form=diamag_form,
    )
    dn = -flux_n

    pe = ctx.n_phys * ctx.Te_phys
    flux_pe = _diamag_flux_expected(
        params=params,
        geom=geom,
        f=pe,
        T=ctx.Te_phys,
        q=q_e,
        curv_x=curv_x,
        diamag_form=diamag_form,
    )
    dpe = -2.5 * flux_pe
    n_eff = jnp.maximum(ctx.n_phys, float(params.n0_min))
    dTe_expected = (dpe - ctx.Te_phys * dn) / n_eff
    assert jnp.allclose(term.Te, dTe_expected, rtol=1e-10, atol=1e-10)


def test_diamagnetic_current_mass_weighting_in_hermes_mode() -> None:
    params_scaled = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {
                "diamagnetic_current_on": True,
                "hot_ion_on": False,
                "average_atomic_mass": 2.5,
                "diamagnetic_current_scale": 1.0,
            },
            "numerics": {
                "poisson": "spectral",
                "poisson_b_weighted": True,
                "poisson_b_weighted_mode": "scaled",
            },
        },
    )
    geom_scaled = _make_geom(params_scaled)
    y_scaled = _make_state(geom_scaled.grid)
    ctx_scaled = build_context(params_scaled, geom_scaled, y_scaled)
    term_scaled = diamagnetic_current_terms(ctx_scaled, y_scaled)
    assert float(jnp.sqrt(jnp.mean(term_scaled.omega**2))) > 0.0

    params_hermes = update_params_from_dict(
        params_scaled,
        {
            "numerics": {"poisson_b_weighted_mode": "hermes"},
        },
    )
    geom_hermes = _make_geom(params_hermes)
    y_hermes = _make_state(geom_hermes.grid)
    ctx_hermes = build_context(params_hermes, geom_hermes, y_hermes)
    term_hermes = diamagnetic_current_terms(ctx_hermes, y_hermes)
    np.testing.assert_allclose(
        term_hermes.omega,
        float(params_hermes.average_atomic_mass) * term_scaled.omega,
        rtol=1e-12,
        atol=1e-12,
    )

    params_hermes_nomass = update_params_from_dict(
        params_hermes,
        {"physics": {"diamagnetic_current_mass_weighted": False}},
    )
    geom_nomass = _make_geom(params_hermes_nomass)
    y_nomass = _make_state(geom_nomass.grid)
    ctx_nomass = build_context(params_hermes_nomass, geom_nomass, y_nomass)
    term_nomass = diamagnetic_current_terms(ctx_nomass, y_nomass)
    np.testing.assert_allclose(term_nomass.omega, term_scaled.omega, rtol=1e-12, atol=1e-12)
