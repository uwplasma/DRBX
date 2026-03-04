from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaxdrb.core.state import DRBSystemState
from jaxdrb.driver import build_system_from_config

_REF_PATH = Path(__file__).resolve().parent / "fixtures" / "drb_fv_one_step_reference.npz"


def _cfg() -> dict:
    return {
        "engine": "drb_fv",
        "geometry": {
            "kind": "slab",
            "nx": 6,
            "ny": 5,
            "nz": 7,
            "Lx": 1.0,
            "Ly": 1.2,
            "Lz": 2.1,
            "bxcv_const": 0.35,
            "open_field_line": False,
        },
        "physics": {"source_n0": 0.015},
        "terms": {"parallel_on": True, "curvature_on": True, "sheath_on": False},
        "numerics": {
            "poisson_scale": 1.7,
            "fv_poisson_solver": "spectral_xy",
            "parallel_pressure_flux_coeff": 5.0 / 3.0,
            "parallel_pressure_work_coeff": 2.0 / 3.0,
            "vorticity_parallel_coeff": 1.0,
            "curvature_coeff": 1.0,
        },
        "initial": {"n0": 1.0, "Te0": 1.0, "omega0": 0.0},
    }


def _deterministic_state(shape: tuple[int, int, int]) -> DRBSystemState:
    nz, nx, ny = shape
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz)[:, None, None]
    x = jnp.linspace(0.0, 2.0 * jnp.pi, nx)[None, :, None]
    y = jnp.linspace(0.0, 2.0 * jnp.pi, ny)[None, None, :]

    n = 1.0 + 0.08 * jnp.sin(z + 0.3 * x) + 0.04 * jnp.cos(2.0 * x - 0.5 * y)
    Te = 1.1 + 0.06 * jnp.cos(1.5 * z - 0.7 * y) + 0.03 * jnp.sin(x + y)
    omega = 0.02 * jnp.sin(2.0 * x + y) - 0.015 * jnp.cos(z)
    vpar_e = 0.05 * jnp.sin(z) + 0.02 * jnp.cos(x)
    vpar_i = -0.04 * jnp.cos(z - 0.2 * y) + 0.015 * jnp.sin(x)

    return DRBSystemState(
        n=n,
        omega=omega,
        vpar_e=vpar_e,
        vpar_i=vpar_i,
        Te=Te,
        Ti=None,
        psi=None,
        N=None,
    )


def _collect_terms_one_step() -> dict[str, np.ndarray]:
    built = build_system_from_config(_cfg())
    state = _deterministic_state(built.state.n.shape)
    data: dict[str, np.ndarray] = {}
    dt = 0.01
    fields = ("n", "Te", "omega", "vpar_e", "vpar_i")

    for step, tval in ((0, 0.0), (1, dt)):
        split, term_map, phi, _ = built.system.rhs_terms(tval, state)
        total = split.total()
        data[f"step{step}_phi"] = np.asarray(phi, dtype=np.float64)
        for field in fields:
            data[f"step{step}_total_{field}"] = np.asarray(
                getattr(total, field),
                dtype=np.float64,
            )
        for name, term in term_map.items():
            for field in fields:
                data[f"step{step}_{name}_{field}"] = np.asarray(
                    getattr(term, field),
                    dtype=np.float64,
                )
        state = DRBSystemState(
            n=state.n + dt * total.n,
            omega=state.omega + dt * total.omega,
            vpar_e=state.vpar_e + dt * total.vpar_e,
            vpar_i=state.vpar_i + dt * total.vpar_i,
            Te=state.Te + dt * total.Te,
            Ti=None,
            psi=None,
            N=None,
        )
    return data


def test_drb_fv_one_step_term_audit_regression() -> None:
    with np.load(_REF_PATH) as ref:
        got = _collect_terms_one_step()
        assert set(got) == set(ref.files)
        for key in sorted(got):
            np.testing.assert_allclose(
                got[key],
                np.asarray(ref[key], dtype=np.float64),
                rtol=1e-11,
                atol=1e-12,
                err_msg=f"Mismatch in one-step alignment audit channel: {key}",
            )
