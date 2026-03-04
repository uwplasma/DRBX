from __future__ import annotations

import numpy as np

from jaxdrb.benchmarking import compute_target_fluxes
from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.terms.context import build_context
from jaxdrb.core.terms.sol import sol_parallel_loss, sol_sheath_phi_term
from jaxdrb.driver import build_system_from_config, run_simulation


def _cfg() -> dict:
    return {
        "geometry": {
            "kind": "salpha",
            "nx": 12,
            "ny": 12,
            "nz": 8,
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
            "curvature_model": "vector_xy",
            "theta_scale": 1.0,
        },
        "physics": {
            "nonlinear_on": True,
            "curvature_on": True,
            "curvature_coeff": 1.0,
            "omega_n": 0.08,
            "omega_Te": 0.04,
            "source_on": False,
            "em_on": False,
            "hot_ion_on": False,
            "neutrals_on": False,
            "boussinesq": True,
        },
        "closures": {
            "sol": {
                "sol_on": True,
                "sol_parallel_loss_on": True,
                "sol_parallel_loss_model": "bohm_exp",
                "sol_parallel_loss_coeff": 0.2,
                "sol_sheath_phi_on": True,
                "sol_sheath_phi_dissipation_on": True,
            },
            "sheath": {
                "sheath_on": True,
                "sheath_bc_on": True,
                "sheath_bc_model": "bohm_current",
                "sheath_nu_factor": 1.0,
            },
        },
        "transport": {
            "Dn": 2.0e-3,
            "DOmega": 2.0e-3,
            "DTe": 2.0e-3,
        },
        "numerics": {
            "poisson": "spectral",
            "bracket": "arakawa",
            "parallel_flux_conservative": True,
            "parallel_limiter": "mc",
            "poisson_scale": 1.0,
        },
        "initial": {
            "amplitude": 1.0e-3,
            "seed": 1,
            "noise_mode": "physical",
            "noise_fields": ["n", "omega", "Te"],
        },
        "time": {
            "method": "rk4_imex_strang",
            "dt": 8.0e-4,
            "nsteps": 20,
            "save_every": 5,
            "diag_mode": "full",
            "save_fields": True,
            "snapshot_fields": ["n", "omega", "Te", "vpar_i"],
            "return_numpy": True,
        },
    }


def test_sheath_and_sol_parallel_loss_gate() -> None:
    cfg = _cfg()
    out = run_simulation(cfg, as_numpy=True).diagnostics
    n = np.asarray(out["snapshots_n"][-1], dtype=np.float64)
    te = np.asarray(out["snapshots_Te"][-1], dtype=np.float64)
    omega = np.asarray(out["snapshots_omega"][-1], dtype=np.float64)
    vi = np.asarray(out["snapshots_vpar_i"][-1], dtype=np.float64)

    gamma_t, qe_t, qi_t = compute_target_fluxes(n[None, ...], vi[None, ...], te[None, ...])
    assert np.all(np.isfinite(gamma_t))
    assert np.all(np.isfinite(qe_t))
    assert np.all(np.isfinite(qi_t))

    built = build_system_from_config(cfg)
    y = DRBSystemState(
        n=built.state.n.at[:].set(n),
        omega=built.state.omega.at[:].set(omega),
        vpar_e=built.state.vpar_e,
        vpar_i=built.state.vpar_i.at[:].set(vi),
        Te=built.state.Te.at[:].set(te),
        Ti=None,
        psi=None,
        N=None,
    )
    ctx = build_context(built.system.params, built.system.geom, y)
    loss = sol_parallel_loss(
        built.system.params,
        y,
        ctx.phi,
        n_phys=ctx.n_phys,
        Te_phys=ctx.Te_phys,
        mask_open=ctx.mask_open,
    )
    sheath = sol_sheath_phi_term(
        built.system.params,
        y,
        ctx.phi,
        n_phys=ctx.n_phys,
        Te_phys=ctx.Te_phys,
        mask_open=ctx.mask_open,
    )

    assert np.all(np.isfinite(np.asarray(loss.n)))
    assert np.all(np.isfinite(np.asarray(sheath.omega)))
    assert float(np.mean(np.abs(loss.n))) > 0.0
    assert float(np.mean(np.abs(sheath.omega))) > 0.0
