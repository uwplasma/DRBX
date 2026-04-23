from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeutralMixedState:
    density: np.ndarray
    pressure: np.ndarray
    momentum: np.ndarray


@dataclass(frozen=True)
class NeutralMixedRhsResult:
    density: np.ndarray
    pressure: np.ndarray
    momentum: np.ndarray
    diffusion: np.ndarray
    density_parallel_flow: np.ndarray
    pressure_parallel_flow: np.ndarray


@dataclass(frozen=True)
class NeutralMixedHistoryResult:
    density_history: np.ndarray
    pressure_history: np.ndarray
    momentum_history: np.ndarray


@dataclass(frozen=True)
class NeutralMixedImplicitStepInfo:
    residual_inf_norm: float
    active_shape: tuple[int, int, int]
    nonlinear_iterations: int
    linear_iterations: int


@dataclass(frozen=True)
class PreparedNeutralMixedState:
    density: np.ndarray
    pressure: np.ndarray
    momentum: np.ndarray
    density_limited: np.ndarray
    pressure_limited: np.ndarray
    temperature: np.ndarray
    temperature_limited: np.ndarray
    velocity: np.ndarray
    diffusion: np.ndarray
    diffusion_density: np.ndarray
    diffusion_pressure: np.ndarray
    diffusion_momentum: np.ndarray
    conductivity: np.ndarray
    viscosity: np.ndarray
    log_pressure: np.ndarray
    sound_speed: np.ndarray
