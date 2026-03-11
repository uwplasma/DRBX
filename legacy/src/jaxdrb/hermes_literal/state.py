from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from .field import pad_field3d
from .types import Field3DLayout

if TYPE_CHECKING:
    from jaxdrb.core.geometry import GeometryAdapter
    from jaxdrb.core.params import DRBSystemParams

    from .bcs import FieldBCs


@dataclass(frozen=True)
class LiteralSpeciesState:
    density: jnp.ndarray | None = None
    pressure: jnp.ndarray | None = None
    temperature: jnp.ndarray | None = None
    velocity: jnp.ndarray | None = None
    density_guarded: jnp.ndarray | None = None
    pressure_guarded: jnp.ndarray | None = None
    temperature_guarded: jnp.ndarray | None = None
    velocity_guarded: jnp.ndarray | None = None
    layout: Field3DLayout | None = None
    AA: float | None = None
    charge: float | None = None


@dataclass(frozen=True)
class LiteralFieldsState:
    phi: jnp.ndarray | None = None
    sound_speed: jnp.ndarray | None = None
    fastest_wave: jnp.ndarray | None = None
    phi_guarded: jnp.ndarray | None = None
    sound_speed_guarded: jnp.ndarray | None = None
    fastest_wave_guarded: jnp.ndarray | None = None
    layout: Field3DLayout | None = None


@dataclass(frozen=True)
class LiteralStage1State:
    electrons: LiteralSpeciesState
    ions: LiteralSpeciesState
    fields: LiteralFieldsState


def _x_periodic(bc) -> bool:
    return bool(getattr(bc, "kind_x", None) == 0)


def _guard_field(
    field: jnp.ndarray,
    *,
    x_periodic: bool,
    parallel_periodic: bool,
    guard_width: int = 2,
) -> tuple[jnp.ndarray, Field3DLayout]:
    return pad_field3d(
        jnp.asarray(field, dtype=jnp.float64),
        x_periodic=x_periodic,
        parallel_periodic=parallel_periodic,
        guard_width=guard_width,
    )


def build_literal_stage1_state(
    *,
    params: DRBSystemParams,
    geom: GeometryAdapter,
    bcs: FieldBCs,
    density: jnp.ndarray,
    electron_temperature: jnp.ndarray,
    ion_temperature: jnp.ndarray,
    electron_pressure: jnp.ndarray,
    ion_pressure: jnp.ndarray,
    phi: jnp.ndarray,
    vpar_e: jnp.ndarray,
    vpar_i: jnp.ndarray,
    guard_width: int = 2,
) -> LiteralStage1State:
    """Build the prepared literal Stage 1 state carried by the runtime context."""

    from .sound_speed import compute_fastest_wave

    grid = getattr(geom, "grid", None)
    parallel_periodic = not bool(getattr(grid, "open_field_line", False))

    density_guarded, layout = _guard_field(
        density,
        x_periodic=_x_periodic(bcs.n),
        parallel_periodic=parallel_periodic,
        guard_width=guard_width,
    )
    pe_guarded, _ = _guard_field(
        electron_pressure,
        x_periodic=_x_periodic(bcs.Te),
        parallel_periodic=parallel_periodic,
        guard_width=guard_width,
    )
    Te_guarded, _ = _guard_field(
        electron_temperature,
        x_periodic=_x_periodic(bcs.Te),
        parallel_periodic=parallel_periodic,
        guard_width=guard_width,
    )
    ve_guarded, _ = _guard_field(
        vpar_e,
        x_periodic=_x_periodic(bcs.vpar_e),
        parallel_periodic=parallel_periodic,
        guard_width=guard_width,
    )
    pi_guarded, _ = _guard_field(
        ion_pressure,
        x_periodic=_x_periodic(bcs.Ti),
        parallel_periodic=parallel_periodic,
        guard_width=guard_width,
    )
    Ti_guarded, _ = _guard_field(
        ion_temperature,
        x_periodic=_x_periodic(bcs.Ti),
        parallel_periodic=parallel_periodic,
        guard_width=guard_width,
    )
    vi_guarded, _ = _guard_field(
        vpar_i,
        x_periodic=_x_periodic(bcs.vpar_i),
        parallel_periodic=parallel_periodic,
        guard_width=guard_width,
    )
    phi_guarded, _ = _guard_field(
        phi,
        x_periodic=_x_periodic(bcs.phi),
        parallel_periodic=parallel_periodic,
        guard_width=guard_width,
    )

    electrons = LiteralSpeciesState(
        density=density,
        pressure=electron_pressure,
        temperature=electron_temperature,
        velocity=vpar_e,
        density_guarded=density_guarded,
        pressure_guarded=pe_guarded,
        temperature_guarded=Te_guarded,
        velocity_guarded=ve_guarded,
        layout=layout,
        AA=max(float(params.me_hat), 1.0e-12),
        charge=-1.0,
    )
    ions = LiteralSpeciesState(
        density=density,
        pressure=ion_pressure,
        temperature=ion_temperature,
        velocity=vpar_i,
        density_guarded=density_guarded,
        pressure_guarded=pi_guarded,
        temperature_guarded=Ti_guarded,
        velocity_guarded=vi_guarded,
        layout=layout,
        AA=max(float(getattr(params, "average_atomic_mass", 1.0)), 1.0e-12),
        charge=1.0,
    )

    wave = compute_fastest_wave(
        (electrons, ions), temperature_floor=float(params.temperature_floor)
    )
    sound_speed_guarded, _ = _guard_field(
        wave.sound_speed,
        x_periodic=_x_periodic(bcs.n),
        parallel_periodic=parallel_periodic,
        guard_width=guard_width,
    )
    fastest_wave_guarded, _ = _guard_field(
        wave.fastest_wave,
        x_periodic=_x_periodic(bcs.n),
        parallel_periodic=parallel_periodic,
        guard_width=guard_width,
    )
    fields = LiteralFieldsState(
        phi=phi,
        sound_speed=wave.sound_speed,
        fastest_wave=wave.fastest_wave,
        phi_guarded=phi_guarded,
        sound_speed_guarded=sound_speed_guarded,
        fastest_wave_guarded=fastest_wave_guarded,
        layout=layout,
    )
    return LiteralStage1State(electrons=electrons, ions=ions, fields=fields)
