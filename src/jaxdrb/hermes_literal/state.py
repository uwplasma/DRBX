from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class LiteralSpeciesState:
    density: jnp.ndarray | None = None
    pressure: jnp.ndarray | None = None
    temperature: jnp.ndarray | None = None
    velocity: jnp.ndarray | None = None
    AA: float | None = None
    charge: float | None = None


@dataclass(frozen=True)
class LiteralFieldsState:
    phi: jnp.ndarray | None = None
    sound_speed: jnp.ndarray | None = None
    fastest_wave: jnp.ndarray | None = None


@dataclass(frozen=True)
class LiteralStage1State:
    electrons: LiteralSpeciesState
    ions: LiteralSpeciesState
    fields: LiteralFieldsState
