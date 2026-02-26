from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.bc import BC2D
from jaxdrb.core.geometry_2d import Geometry2DAdapter
from jaxdrb.core.params import DRBSystemParams, update_params_from_dict
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.terms.braginskii import classical_diffusion_terms
from jaxdrb.core.terms.context import build_context
from jaxdrb.core.terms.fields import _diamagnetic_polarisation_term
from jaxdrb.geometry.plane import Grid2D


def _l2_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _make_geom(nx: int, ny: int, params: DRBSystemParams) -> Geometry2DAdapter:
    grid = Grid2D.make(
        nx=nx,
        ny=ny,
        Lx=2 * np.pi,
        Ly=2 * np.pi,
        dealias=False,
        bc_x="periodic",
        bc_y="periodic",
    )
    return Geometry2DAdapter(grid=grid, params=params)


def _mms_field(grid: Grid2D, kx: float, ky: float) -> jnp.ndarray:
    x = jnp.asarray(grid.x)[:, None]
    y = jnp.asarray(grid.y)[None, :]
    return jnp.sin(kx * x) * jnp.cos(ky * y)


def test_diamagnetic_polarisation_mms_second_order() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {"diamagnetic_polarisation_on": True, "tau_i": 1.0},
        },
    )
    kx = 2.0
    ky = 3.0
    errors = []
    for n in (16, 32, 64):
        geom = _make_geom(n, n, params)
        p_i = _mms_field(geom.grid, kx, ky)
        n_phys = jnp.ones_like(p_i)
        Ti = p_i - n_phys
        term = _diamagnetic_polarisation_term(params, geom, n_phys, Ti, BC2D.periodic())
        exact = -(kx**2 + ky**2) * p_i
        errors.append(_l2_error(np.asarray(term), np.asarray(exact)))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert (errors[0] / errors[1]) > 3.0
    assert (errors[1] / errors[2]) > 3.0


def test_classical_diffusion_mms_second_order() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {"hot_ion_on": True},
            "transport": {
                "classical_diffusion_on": True,
                "classical_diffusion_custom_D": 1.0,
                "classical_diffusion_custom_kappa_e": 1.0,
                "classical_diffusion_custom_kappa_i": 1.0,
            },
        },
    )
    kx = 2.0
    ky = 1.0
    errors = []
    for n in (16, 32, 64):
        geom = _make_geom(n, n, params)
        nfield = _mms_field(geom.grid, kx, ky)
        Te = _mms_field(geom.grid, kx, ky)
        Ti = _mms_field(geom.grid, kx, ky)
        state = DRBSystemState(
            n=nfield,
            omega=jnp.zeros_like(nfield),
            vpar_e=jnp.zeros_like(nfield),
            vpar_i=jnp.zeros_like(nfield),
            Te=Te,
            Ti=Ti,
            psi=None,
            N=None,
        )
        ctx = build_context(params, geom, state)
        term = classical_diffusion_terms(ctx, state)
        exact = -(kx**2 + ky**2) * nfield
        errors.append(_l2_error(np.asarray(term.n), np.asarray(exact)))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert (errors[0] / errors[1]) > 3.0
    assert (errors[1] / errors[2]) > 3.0


def test_coupled_diffusion_and_diamag_mms() -> None:
    params = update_params_from_dict(
        DRBSystemParams(),
        {
            "physics": {"diamagnetic_polarisation_on": True, "tau_i": 1.0, "hot_ion_on": True},
            "transport": {
                "classical_diffusion_on": True,
                "classical_diffusion_custom_D": 1.0,
                "classical_diffusion_custom_kappa_e": 1.0,
                "classical_diffusion_custom_kappa_i": 1.0,
            },
        },
    )
    kx = 1.0
    ky = 2.0
    errors = []
    for n in (16, 32, 64):
        geom = _make_geom(n, n, params)
        p_i = _mms_field(geom.grid, kx, ky)
        nfield = _mms_field(geom.grid, kx, ky)
        Ti = p_i - nfield
        Te = _mms_field(geom.grid, kx, ky)
        state = DRBSystemState(
            n=nfield,
            omega=jnp.zeros_like(nfield),
            vpar_e=jnp.zeros_like(nfield),
            vpar_i=jnp.zeros_like(nfield),
            Te=Te,
            Ti=Ti,
            psi=None,
            N=None,
        )
        ctx = build_context(params, geom, state)
        term_diff = classical_diffusion_terms(ctx, state)
        term_diamag = _diamagnetic_polarisation_term(params, geom, nfield, Ti, BC2D.periodic())
        exact = -(kx**2 + ky**2) * nfield
        combined = term_diff.n + term_diamag
        exact_combined = exact + (-(kx**2 + ky**2) * p_i)
        errors.append(_l2_error(np.asarray(combined), np.asarray(exact_combined)))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert (errors[0] / errors[1]) > 3.0
    assert (errors[1] / errors[2]) > 3.0
