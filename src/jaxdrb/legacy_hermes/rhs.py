"""Hermes density/pressure RHS ordering mirror.

This module assembles the reduced-model density and pressure channels in the
same staging as Hermes Stage 1:

1. density `finally()` ExB and parallel pieces
2. pressure `finally()` ExB and parallel/work pieces
3. conversion back to the evolved temperature variables used by `jax_drb`

The active strict runtime still uses the unified scheduler, so this mirror
layer returns advection and parallel term groups that can be dropped into the
existing term map without changing solver infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from jaxdrb.core.state import DRBSystemState


@dataclass(frozen=True)
class MirrorDensityRhsTerms:
    """Density contributions from Hermes `finally()`."""

    advection: jnp.ndarray
    parallel: jnp.ndarray


@dataclass(frozen=True)
class MirrorPressureRhsTerms:
    """Pressure-space and temperature-space contributions from Hermes `finally()`."""

    pressure_advection: jnp.ndarray
    pressure_parallel_flux: jnp.ndarray
    pressure_parallel_work: jnp.ndarray
    pressure_parallel_total: jnp.ndarray
    temperature_advection: jnp.ndarray
    temperature_parallel: jnp.ndarray


@dataclass(frozen=True)
class ReducedMirrorTermCache:
    """Cached reduced-model term groups routed through the Hermes mirror layer."""

    advection: DRBSystemState
    parallel: DRBSystemState
    density: MirrorDensityRhsTerms
    electron_pressure: MirrorPressureRhsTerms
    ion_pressure: MirrorPressureRhsTerms | None


def _boundary_fluxes(
    ctx,
    par,
    f: jnp.ndarray,
    *,
    ghost_low: str,
    ghost_high: str,
    vel_low: str,
    vel_high: str,
) -> tuple[
    jnp.ndarray | None,
    jnp.ndarray | None,
    jnp.ndarray | None,
    jnp.ndarray | None,
    jnp.ndarray | None,
    jnp.ndarray | None,
]:
    sheath_flux_mode = str(
        getattr(ctx.params, "parallel_sheath_flux_mode", "boundary_flux")
    ).lower()
    use_boundary_flux = par.sheath_data is not None and sheath_flux_mode == "boundary_flux"
    if not use_boundary_flux:
        return None, None, None, None, None, None
    assert par.sheath_data is not None
    boundary_flux_scale = float(getattr(ctx.params, "parallel_boundary_flux_scale", 1.0))
    left = (
        0.5
        * (f[0] + getattr(par.sheath_data, ghost_low))
        * getattr(par.sheath_data, vel_low)
        * boundary_flux_scale
    )
    right = (
        0.5
        * (f[-1] + getattr(par.sheath_data, ghost_high))
        * getattr(par.sheath_data, vel_high)
        * boundary_flux_scale
    )
    ghost_f_low = getattr(par.sheath_data, ghost_low)
    ghost_f_high = getattr(par.sheath_data, ghost_high)
    ghost_v_low = getattr(par.sheath_data, vel_low.replace("_sheath_", "_ghost_"))
    ghost_v_high = getattr(par.sheath_data, vel_high.replace("_sheath_", "_ghost_"))
    return left, right, ghost_f_low, ghost_f_high, ghost_v_low, ghost_v_high


def _pressure_temperature_rhs(
    ctx,
    *,
    pressure_term: jnp.ndarray,
    density_term: jnp.ndarray,
    temperature: jnp.ndarray,
) -> jnp.ndarray:
    n_eff = jnp.maximum(ctx.n_prepared, float(ctx.params.n0_min))
    return (pressure_term - temperature * density_term) / n_eff


def density_rhs_terms(ctx, y: DRBSystemState, par=None) -> MirrorDensityRhsTerms:
    """Mirror the density channel from Hermes `EvolveDensity::finally()`."""

    from jaxdrb.core.terms.parallel import _dpar_flux_conservative, _fastest_wave, parallel_vars

    if par is None:
        par = parallel_vars(ctx, y)

    if bool(ctx.params.nonlinear_on):
        adv = (
            -ctx.geom.exb_flux_divergence(
                ctx.phi,
                ctx.n_prepared,
                bc_phi=ctx.bcs.phi,
                bc_adv=ctx.bcs.n,
                positive=True,
            )
            * ctx.nonlinear_scale
        )
    else:
        adv = jnp.zeros_like(ctx.n_prepared)

    n_blow, n_bhigh, n_glow, n_ghigh, ve_glow, ve_ghigh = _boundary_fluxes(
        ctx,
        par,
        ctx.n_prepared,
        ghost_low="n_ghost_low",
        ghost_high="n_ghost_high",
        vel_low="ve_sheath_low",
        vel_high="ve_sheath_high",
    )
    parallel = -_dpar_flux_conservative(
        ctx,
        ctx.n_prepared,
        par.vpar_e_flux,
        wave=_fastest_wave(ctx),
        boundary_flux_low=n_blow,
        boundary_flux_high=n_bhigh,
        ghost_low_f=n_glow,
        ghost_high_f=n_ghigh,
        ghost_low_v=ve_glow,
        ghost_high_v=ve_ghigh,
    )
    return MirrorDensityRhsTerms(advection=adv, parallel=parallel)


def pressure_rhs_terms(
    ctx,
    y: DRBSystemState,
    *,
    density_terms: MirrorDensityRhsTerms,
    par=None,
    species: str = "electron",
) -> MirrorPressureRhsTerms:
    """Mirror the pressure channel from Hermes `EvolvePressure::finally()`."""

    from jaxdrb.core.terms.parallel import (
        _dpar_flux_conservative,
        _fastest_wave,
        _pressure_transport_coeffs,
        parallel_vars,
    )

    if par is None:
        par = parallel_vars(ctx, y)

    species_key = str(species).lower()
    if species_key in {"electron", "e"}:
        pressure = ctx.pe_prepared
        temperature = ctx.Te_prepared
        bc_adv = ctx.bcs.Te
        velocity = par.vpar_e_flux
        ghost_low = "pe_ghost_low"
        ghost_high = "pe_ghost_high"
        vel_low = "ve_sheath_low"
        vel_high = "ve_sheath_high"
    elif species_key in {"ion", "i"}:
        pressure = ctx.pi_prepared
        temperature = ctx.Ti_prepared
        bc_adv = ctx.bcs.Ti
        velocity = par.vpar_i_flux
        ghost_low = "pi_ghost_low"
        ghost_high = "pi_ghost_high"
        vel_low = "vi_sheath_low"
        vel_high = "vi_sheath_high"
    else:
        raise ValueError(f"Unsupported species={species!r}; expected 'electron' or 'ion'.")

    if bool(ctx.params.nonlinear_on):
        pressure_advection = (
            -ctx.geom.exb_flux_divergence(
                ctx.phi,
                pressure,
                bc_phi=ctx.bcs.phi,
                bc_adv=bc_adv,
                positive=True,
            )
            * ctx.nonlinear_scale
        )
    else:
        pressure_advection = jnp.zeros_like(pressure)

    pressure_transport_coeff, pressure_work_coeff = _pressure_transport_coeffs(ctx)
    p_blow, p_bhigh, p_glow, p_ghigh, v_glow, v_ghigh = _boundary_fluxes(
        ctx,
        par,
        pressure,
        ghost_low=ghost_low,
        ghost_high=ghost_high,
        vel_low=vel_low,
        vel_high=vel_high,
    )
    pressure_parallel_flux = -pressure_transport_coeff * _dpar_flux_conservative(
        ctx,
        pressure,
        velocity,
        wave=_fastest_wave(ctx),
        boundary_flux_low=p_blow,
        boundary_flux_high=p_bhigh,
        ghost_low_f=p_glow,
        ghost_high_f=p_ghigh,
        ghost_low_v=v_glow,
        ghost_high_v=v_ghigh,
    )
    pressure_parallel_work = (
        pressure_work_coeff * (velocity * ctx.geom.dpar(pressure, bc_kind="dirichlet"))
        if pressure_work_coeff != 0.0
        else jnp.zeros_like(pressure_parallel_flux)
    )
    pressure_parallel_total = pressure_parallel_flux + pressure_parallel_work
    temperature_advection = _pressure_temperature_rhs(
        ctx,
        pressure_term=pressure_advection,
        density_term=density_terms.advection,
        temperature=temperature,
    )
    temperature_parallel = _pressure_temperature_rhs(
        ctx,
        pressure_term=pressure_parallel_total,
        density_term=density_terms.parallel,
        temperature=temperature,
    )
    return MirrorPressureRhsTerms(
        pressure_advection=pressure_advection,
        pressure_parallel_flux=pressure_parallel_flux,
        pressure_parallel_work=pressure_parallel_work,
        pressure_parallel_total=pressure_parallel_total,
        temperature_advection=temperature_advection,
        temperature_parallel=temperature_parallel,
    )


def build_reduced_mirror_term_cache(ctx, y: DRBSystemState, par=None) -> ReducedMirrorTermCache:
    """Assemble the reduced-model advection and parallel term groups via Hermes order."""

    from jaxdrb.core.terms.advection import exb_advection_terms
    from jaxdrb.core.terms.parallel import parallel_conservative_terms, parallel_vars

    if par is None:
        par = parallel_vars(ctx, y)

    adv_base = exb_advection_terms(ctx, y)
    par_base = parallel_conservative_terms(ctx, y, par)
    density_terms = density_rhs_terms(ctx, y, par=par)
    electron_pressure = pressure_rhs_terms(
        ctx,
        y,
        density_terms=density_terms,
        par=par,
        species="electron",
    )
    ion_pressure = (
        pressure_rhs_terms(
            ctx,
            y,
            density_terms=density_terms,
            par=par,
            species="ion",
        )
        if ctx.hot_on and y.Ti is not None
        else None
    )

    advection = DRBSystemState(
        n=density_terms.advection,
        omega=adv_base.omega,
        vpar_e=adv_base.vpar_e,
        vpar_i=adv_base.vpar_i,
        Te=electron_pressure.temperature_advection,
        Ti=(
            ion_pressure.temperature_advection
            if ion_pressure is not None and y.Ti is not None
            else None
        ),
        psi=adv_base.psi if y.psi is not None else None,
        N=adv_base.N if y.N is not None else None,
    )
    parallel = DRBSystemState(
        n=density_terms.parallel,
        omega=par_base.omega,
        vpar_e=par_base.vpar_e,
        vpar_i=par_base.vpar_i,
        Te=electron_pressure.temperature_parallel,
        Ti=(
            ion_pressure.temperature_parallel
            if ion_pressure is not None and y.Ti is not None
            else None
        ),
        psi=par_base.psi if y.psi is not None else None,
        N=par_base.N if y.N is not None else None,
    )
    return ReducedMirrorTermCache(
        advection=advection,
        parallel=parallel,
        density=density_terms,
        electron_pressure=electron_pressure,
        ion_pressure=ion_pressure,
    )
