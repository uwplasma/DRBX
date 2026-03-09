from __future__ import annotations

import tomllib
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaxdrb.core.state import DRBSystemState
from jaxdrb.core.terms.context import build_context
from jaxdrb.core.terms.parallel import parallel_vars
from jaxdrb.driver import build_system_from_config
from jaxdrb.legacy_hermes.rhs import (
    build_reduced_mirror_term_cache,
    density_rhs_terms,
    pressure_rhs_terms,
)

_CFG = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "open_field_line"
    / "input_tokamak_bxcv_alignment_strict_early.toml"
)


def _load_cfg() -> dict:
    return tomllib.loads(_CFG.read_text(encoding="utf-8"))


def _synthetic_parallel_state(state: DRBSystemState) -> DRBSystemState:
    nz, nx, ny = (int(v) for v in state.n.shape)
    z = jnp.linspace(-1.0, 1.0, nz, dtype=jnp.float64)[:, None, None]
    x = jnp.linspace(-1.0, 1.0, nx, dtype=jnp.float64)[None, :, None]
    yb = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False, dtype=jnp.float64)[None, None, :]
    vpar_e = 0.03 * jnp.sin(1.3 * z) * (1.0 + 0.2 * x) * jnp.cos(yb)
    vpar_i = -0.02 * jnp.cos(0.7 * z) * (1.0 - 0.1 * x) * jnp.sin(yb)
    return DRBSystemState(
        n=state.n,
        omega=state.omega,
        vpar_e=vpar_e,
        vpar_i=vpar_i,
        Te=state.Te,
        Ti=state.Ti,
        psi=state.psi,
        N=state.N,
    )


def test_density_and_pressure_rhs_terms_match_pressure_space_identities() -> None:
    built = build_system_from_config(_load_cfg())
    y = _synthetic_parallel_state(built.state)
    ctx = build_context(built.system.params, built.system.geom, y)
    par = parallel_vars(ctx, y)

    density = density_rhs_terms(ctx, y, par=par)
    electron = pressure_rhs_terms(
        ctx,
        y,
        density_terms=density,
        par=par,
        species="electron",
    )

    direct_n_adv = (
        -ctx.geom.exb_flux_divergence(
            ctx.phi,
            ctx.n_prepared,
            bc_phi=ctx.bcs.phi,
            bc_adv=ctx.bcs.n,
            positive=True,
        )
        * ctx.nonlinear_scale
    )
    direct_pe_adv = (
        -ctx.geom.exb_flux_divergence(
            ctx.phi,
            ctx.pe_prepared,
            bc_phi=ctx.bcs.phi,
            bc_adv=ctx.bcs.Te,
            positive=True,
        )
        * ctx.nonlinear_scale
    )
    n_eff = np.asarray(jnp.maximum(ctx.n_prepared, float(ctx.params.n0_min)))
    direct_te_adv = (
        np.asarray(electron.pressure_advection)
        - np.asarray(ctx.Te_prepared) * np.asarray(density.advection)
    ) / n_eff

    np.testing.assert_allclose(np.asarray(density.advection), np.asarray(direct_n_adv))
    np.testing.assert_allclose(np.asarray(electron.pressure_advection), np.asarray(direct_pe_adv))
    np.testing.assert_allclose(np.asarray(electron.temperature_advection), direct_te_adv)
    np.testing.assert_allclose(
        np.asarray(electron.pressure_parallel_total),
        np.asarray(electron.pressure_parallel_flux) + np.asarray(electron.pressure_parallel_work),
    )

    if y.Ti is not None:
        ion = pressure_rhs_terms(
            ctx,
            y,
            density_terms=density,
            par=par,
            species="ion",
        )
        direct_pi_adv = (
            -ctx.geom.exb_flux_divergence(
                ctx.phi,
                ctx.pi_prepared,
                bc_phi=ctx.bcs.phi,
                bc_adv=ctx.bcs.Ti,
                positive=True,
            )
            * ctx.nonlinear_scale
        )
        direct_ti_adv = (
            np.asarray(ion.pressure_advection)
            - np.asarray(ctx.Ti_prepared) * np.asarray(density.advection)
        ) / n_eff
        np.testing.assert_allclose(np.asarray(ion.pressure_advection), np.asarray(direct_pi_adv))
        np.testing.assert_allclose(np.asarray(ion.temperature_advection), direct_ti_adv)


def test_reduced_mirror_term_cache_reconstructs_pressure_space_terms() -> None:
    built = build_system_from_config(_load_cfg())
    y = _synthetic_parallel_state(built.state)
    ctx = build_context(built.system.params, built.system.geom, y)
    par = parallel_vars(ctx, y)
    cache = build_reduced_mirror_term_cache(ctx, y, par=par)

    n_eff = np.asarray(jnp.maximum(ctx.n_prepared, float(ctx.params.n0_min)))
    pe_adv = n_eff * np.asarray(cache.advection.Te) + np.asarray(ctx.Te_prepared) * np.asarray(
        cache.advection.n
    )
    pe_par = n_eff * np.asarray(cache.parallel.Te) + np.asarray(ctx.Te_prepared) * np.asarray(
        cache.parallel.n
    )

    np.testing.assert_allclose(pe_adv, np.asarray(cache.electron_pressure.pressure_advection))
    np.testing.assert_allclose(
        pe_par,
        np.asarray(cache.electron_pressure.pressure_parallel_total),
        rtol=1e-10,
        atol=1e-18,
    )

    if cache.ion_pressure is not None and y.Ti is not None:
        pi_adv = n_eff * np.asarray(cache.advection.Ti) + np.asarray(ctx.Ti_prepared) * np.asarray(
            cache.advection.n
        )
        pi_par = n_eff * np.asarray(cache.parallel.Ti) + np.asarray(ctx.Ti_prepared) * np.asarray(
            cache.parallel.n
        )
        np.testing.assert_allclose(pi_adv, np.asarray(cache.ion_pressure.pressure_advection))
        np.testing.assert_allclose(
            pi_par,
            np.asarray(cache.ion_pressure.pressure_parallel_total),
            rtol=1e-10,
            atol=1e-18,
        )
