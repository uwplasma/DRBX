from __future__ import annotations

import numpy as np

from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.terms.context import build_context
from jaxdrb.core.terms.parallel import parallel_conservative_terms, parallel_vars
from jaxdrb.driver import build_system_from_config


def _cfg(parallel_use_sheath_targets: bool, *, sheath_flux_mode: str = "boundary_flux") -> dict:
    return {
        "geometry": {
            "kind": "salpha",
            "nx": 12,
            "ny": 12,
            "nz": 10,
            "Lx": 6.0,
            "Ly": 6.0,
            "Lz": 2.0 * np.pi,
            "bc_x": "periodic",
            "bc_y": "periodic",
            "open_field_line": True,
            "shat": 0.7,
            "alpha": 0.0,
            "q": 1.4,
            "R0": 3.0,
            "epsilon": 0.18,
            "curvature0": 0.18,
        },
        "physics": {
            "nonlinear_on": True,
            "curvature_on": False,
            "em_on": False,
            "hot_ion_on": True,
            "neutrals_on": False,
            "boussinesq": True,
        },
        "closures": {
            "sheath_on": True,
            "sheath_bc_on": True,
            "sheath_bc_model": "bohm_current",
        },
        "numerics": {
            "poisson": "spectral",
            "parallel_flux_conservative": True,
            "parallel_momentum_model": "conservative",
            "parallel_limiter": "mc",
            "parallel_flux_scheme": "lax",
            "parallel_use_sheath_targets": parallel_use_sheath_targets,
            "parallel_sheath_flux_mode": sheath_flux_mode,
        },
    }


def _boundary_rms(arr: np.ndarray) -> float:
    b = np.concatenate([arr[0].ravel(), arr[-1].ravel()])
    return float(np.sqrt(np.mean(b * b)))


def _build_parallel_n_rhs(use_targets: bool) -> np.ndarray:
    built = build_system_from_config(_cfg(use_targets))
    y0 = built.state
    shape = np.asarray(y0.vpar_e).shape
    vz = np.linspace(-0.15, 0.15, shape[0], dtype=np.float64)[:, None, None]
    nz_profile = np.linspace(-0.05, 0.05, shape[0], dtype=np.float64)[:, None, None]
    vpar_e = np.broadcast_to(vz, shape).copy()
    vpar_i = -0.5 * vpar_e
    n_profile = np.broadcast_to(nz_profile, shape).copy()
    n_field = 1.0 + n_profile
    y = DRBSystemState(
        n=n_field,
        omega=y0.omega,
        vpar_e=vpar_e,
        vpar_i=vpar_i,
        Te=y0.Te + 0.6 + 0.2 * n_profile,
        Ti=None if y0.Ti is None else y0.Ti + 0.4 + 0.1 * n_profile,
        psi=None,
        N=None,
    )
    ctx = build_context(built.system.params, built.system.geom, y)
    par = parallel_vars(ctx, y)
    term = parallel_conservative_terms(ctx, y, par)
    return np.asarray(term.n, dtype=np.float64)


def _build_parallel_jpar_div_rhs(*, sheath_flux_mode: str) -> np.ndarray:
    built = build_system_from_config(_cfg(True, sheath_flux_mode=sheath_flux_mode))
    y0 = built.state
    shape = np.asarray(y0.vpar_e).shape
    vz = np.linspace(-0.15, 0.15, shape[0], dtype=np.float64)[:, None, None]
    nz_profile = np.linspace(-0.05, 0.05, shape[0], dtype=np.float64)[:, None, None]
    vpar_e = np.broadcast_to(vz, shape).copy()
    vpar_i = -0.5 * vpar_e
    n_profile = np.broadcast_to(nz_profile, shape).copy()
    n_field = 1.0 + n_profile
    y = DRBSystemState(
        n=n_field,
        omega=y0.omega,
        vpar_e=vpar_e,
        vpar_i=vpar_i,
        Te=y0.Te + 0.6 + 0.2 * n_profile,
        Ti=None if y0.Ti is None else y0.Ti + 0.4 + 0.1 * n_profile,
        psi=None,
        N=None,
    )
    ctx = build_context(built.system.params, built.system.geom, y)
    par = parallel_vars(ctx, y)
    return np.asarray(par.dpar_j, dtype=np.float64)


def test_parallel_sheath_targets_boost_open_field_boundary_flux() -> None:
    dn_no = _build_parallel_n_rhs(False)
    dn_yes = _build_parallel_n_rhs(True)

    rms_no = _boundary_rms(dn_no)
    rms_yes = _boundary_rms(dn_yes)

    assert np.isfinite(rms_no)
    assert np.isfinite(rms_yes)
    assert rms_yes > 5.0 * max(rms_no, 1e-14)
    assert np.max(np.abs(dn_yes[0])) > 0.0
    assert np.max(np.abs(dn_yes[-1])) > 0.0


def test_parallel_jpar_boundary_flux_mode_concentrates_sheath_response() -> None:
    dj_boundary = _build_parallel_jpar_div_rhs(sheath_flux_mode="boundary_flux")
    dj_replace = _build_parallel_jpar_div_rhs(sheath_flux_mode="replace_boundary")

    bnd_boundary = _boundary_rms(dj_boundary)
    bnd_replace = _boundary_rms(dj_replace)
    inner_boundary = float(np.sqrt(np.mean(dj_boundary[1:-1] ** 2)))
    inner_replace = float(np.sqrt(np.mean(dj_replace[1:-1] ** 2)))

    assert np.isfinite(bnd_boundary)
    assert np.isfinite(bnd_replace)
    assert bnd_boundary > 5.0 * max(bnd_replace, 1e-14)
    # Boundary-flux mode should primarily alter sheath-adjacent planes.
    assert inner_boundary < 0.25 * bnd_boundary
    assert inner_replace > 0.2 * bnd_replace
