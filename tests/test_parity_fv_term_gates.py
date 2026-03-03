from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState
from jaxdrb.driver import build_system_from_config
from jaxdrb.parity_fv import pressure_parallel_tendencies


def _cfg(*, bxcv_const: float = 0.0, curvature_on: bool = True) -> dict:
    return {
        "engine": "parity_fv",
        "geometry": {
            "kind": "slab",
            "nx": 12,
            "ny": 8,
            "nz": 10,
            "Lx": 1.0,
            "Ly": 1.0,
            "Lz": 2.0,
            "bxcv_const": bxcv_const,
        },
        "terms": {"parallel_on": True, "curvature_on": curvature_on},
        "initial": {"n0": 1.0, "Te0": 1.0, "omega0": 0.0},
    }


def test_parity_fv_term_map_contains_density_pressure_vorticity_channels() -> None:
    built = build_system_from_config(_cfg())
    split, term_map, _, _ = built.system.rhs_terms(0.0, built.state)
    assert {"parallel", "curvature", "volume_source"}.issubset(set(term_map))
    total = split.total()
    assert total.n.shape == built.state.n.shape
    assert total.Te.shape == built.state.Te.shape
    assert total.omega.shape == built.state.omega.shape


def test_parallel_density_pressure_vorticity_zero_on_uniform_state() -> None:
    built = build_system_from_config(_cfg())
    y = DRBSystemState(
        n=jnp.ones_like(built.state.n),
        omega=jnp.zeros_like(built.state.omega),
        vpar_e=jnp.zeros_like(built.state.vpar_e),
        vpar_i=jnp.zeros_like(built.state.vpar_i),
        Te=jnp.ones_like(built.state.Te),
        Ti=None,
        psi=None,
        N=None,
    )
    _, term_map, _, _ = built.system.rhs_terms(0.0, y)
    par = term_map["parallel"]
    assert np.allclose(np.asarray(par.n), 0.0, atol=1e-14)
    assert np.allclose(np.asarray(par.Te), 0.0, atol=1e-14)
    assert np.allclose(np.asarray(par.omega), 0.0, atol=1e-14)


def test_pressure_parallel_flux_coeff_scales_pressure_tendency() -> None:
    nz, nx, ny = 12, 6, 4
    z = jnp.linspace(0.0, 1.0, nz)[:, None, None]
    n = 1.0 + 0.1 * z + jnp.zeros((nz, nx, ny))
    Te = 1.0 + 0.2 * z + jnp.zeros((nz, nx, ny))
    vpar = 0.3 + 0.0 * n
    dn = jnp.zeros_like(n)
    dpe1, _ = pressure_parallel_tendencies(
        n,
        Te,
        vpar,
        dn_parallel=dn,
        dz=0.1,
        limiter="mc",
        n_floor=1e-12,
        Te_floor=1e-12,
        flux_coeff=1.0,
        work_coeff=0.0,
    )
    dpe2, _ = pressure_parallel_tendencies(
        n,
        Te,
        vpar,
        dn_parallel=dn,
        dz=0.1,
        limiter="mc",
        n_floor=1e-12,
        Te_floor=1e-12,
        flux_coeff=2.0,
        work_coeff=0.0,
    )
    mask = np.abs(np.asarray(dpe1)) > 1e-14
    ratio = np.asarray(dpe2)[mask] / np.asarray(dpe1)[mask]
    assert ratio.size > 0
    assert np.allclose(ratio, 2.0, atol=5e-3, rtol=5e-3)


def test_vorticity_curvature_term_requires_bxcv() -> None:
    base = build_system_from_config(_cfg(bxcv_const=0.0, curvature_on=True))
    with_curv = build_system_from_config(_cfg(bxcv_const=0.7, curvature_on=True))
    y = DRBSystemState(
        n=1.0
        + 0.2 * jnp.sin(jnp.linspace(0.0, 2.0 * jnp.pi, base.state.n.shape[1]))[None, :, None]
        + jnp.zeros_like(base.state.n),
        omega=jnp.zeros_like(base.state.omega),
        vpar_e=jnp.zeros_like(base.state.vpar_e),
        vpar_i=jnp.zeros_like(base.state.vpar_i),
        Te=1.0
        + 0.1 * jnp.cos(jnp.linspace(0.0, 2.0 * jnp.pi, base.state.Te.shape[1]))[None, :, None]
        + jnp.zeros_like(base.state.Te),
        Ti=None,
        psi=None,
        N=None,
    )

    _, t0, _, _ = base.system.rhs_terms(0.0, y)
    _, t1, _, _ = with_curv.system.rhs_terms(0.0, y)
    omega0 = np.asarray(t0["curvature"].omega)
    omega1 = np.asarray(t1["curvature"].omega)
    assert np.allclose(omega0, 0.0, atol=1e-14)
    assert float(np.max(np.abs(omega1))) > 0.0
