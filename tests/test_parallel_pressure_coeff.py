from __future__ import annotations

import numpy as np

from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.terms.context import build_context
from jaxdrb.core.terms.parallel import parallel_conservative_terms, parallel_vars
from jaxdrb.driver import build_system_from_config


def _cfg(
    pressure_flux_coeff: float,
    pressure_work_coeff: float = 0.0,
    pressure_model: str = "custom",
) -> dict:
    return {
        "geometry": {
            "kind": "salpha",
            "nx": 16,
            "ny": 16,
            "nz": 16,
            "Lx": 4.0,
            "Ly": 4.0,
            "Lz": 2.0 * np.pi,
            "bc_x": "periodic",
            "bc_y": "periodic",
            "open_field_line": False,
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
            "hot_ion_on": False,
            "neutrals_on": False,
            "boussinesq": True,
            "parallel_pressure_model": pressure_model,
            "parallel_pressure_flux_coeff": pressure_flux_coeff,
            "parallel_pressure_work_coeff": pressure_work_coeff,
        },
        "numerics": {
            "poisson": "spectral",
            "parallel_flux_conservative": True,
            "parallel_momentum_model": "conservative",
            "parallel_limiter": "mc",
            "parallel_flux_scheme": "rusanov",
            "parallel_use_sheath_targets": False,
        },
    }


def _parallel_terms(
    pressure_flux_coeff: float,
    pressure_work_coeff: float = 0.0,
    pressure_model: str = "custom",
) -> tuple[np.ndarray, np.ndarray]:
    built = build_system_from_config(
        _cfg(
            pressure_flux_coeff,
            pressure_work_coeff=pressure_work_coeff,
            pressure_model=pressure_model,
        )
    )
    y0 = built.state
    shape = np.asarray(y0.n).shape

    zz = np.linspace(0.0, 2.0 * np.pi, shape[0], endpoint=False, dtype=np.float64)[:, None, None]
    xx = np.linspace(0.0, 1.0, shape[1], endpoint=False, dtype=np.float64)[None, :, None]
    yy = np.linspace(0.0, 1.0, shape[2], endpoint=False, dtype=np.float64)[None, None, :]

    n = 1.0 + 0.08 * np.sin(zz) + 0.03 * np.cos(2.0 * np.pi * xx)
    Te = 1.0 + 0.06 * np.cos(zz) + 0.02 * np.sin(2.0 * np.pi * yy)
    vpar_e = 0.2 * np.sin(zz + 0.2) + 0.03 * np.cos(2.0 * np.pi * xx)

    y = DRBSystemState(
        n=n,
        omega=y0.omega,
        vpar_e=vpar_e,
        vpar_i=np.zeros_like(vpar_e),
        Te=Te,
        Ti=None,
        psi=None,
        N=None,
    )
    ctx = build_context(built.system.params, built.system.geom, y)
    par = parallel_vars(ctx, y)
    term = parallel_conservative_terms(ctx, y, par)
    dpe = ctx.n_phys * np.asarray(term.Te) + ctx.Te_phys * np.asarray(term.n)
    return np.asarray(dpe, dtype=np.float64), np.asarray(term.Te, dtype=np.float64)


def test_parallel_pressure_flux_coeff_scales_parallel_pe_term() -> None:
    dpe_1, _ = _parallel_terms(1.0)
    dpe_15, _ = _parallel_terms(1.5)

    ratio = dpe_15 / np.maximum(np.abs(dpe_1), 1e-12)
    core = np.abs(dpe_1) > 1e-8
    ratio_core = np.abs(ratio[core])

    assert ratio_core.size > 0
    assert np.isfinite(ratio_core).all()
    # Away from tiny values, pressure-parallel term should scale linearly.
    assert np.allclose(ratio_core, 1.5, rtol=2e-2, atol=2e-2)


def test_parallel_pressure_model_hermes_vgradp_matches_explicit_coefficients() -> None:
    _, dte_model = _parallel_terms(1.0, pressure_model="hermes_vgradp")
    _, dte_coeff = _parallel_terms(5.0 / 3.0, pressure_work_coeff=2.0 / 3.0)
    assert np.allclose(dte_model, dte_coeff, rtol=1e-10, atol=1e-10)


def test_parallel_pressure_model_hermes_pdivv_matches_explicit_coefficients() -> None:
    _, dte_model = _parallel_terms(1.0, pressure_model="hermes_pdivv")
    _, dte_coeff = _parallel_terms(1.0, pressure_work_coeff=0.0)
    assert np.allclose(dte_model, dte_coeff, rtol=1e-10, atol=1e-10)
