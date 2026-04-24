from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Recycling1DRhsResult:
    variables: dict[str, np.ndarray]
    feedback_integral_rhs: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Recycling1DHistoryResult:
    variable_history: dict[str, np.ndarray]
    feedback_integral_history: dict[str, np.ndarray]


@dataclass(frozen=True)
class Recycling1DImplicitStepInfo:
    residual_inf_norm: float
    active_size: int
    nonlinear_iterations: int
    linear_iterations: int
    diagnostics: dict[str, float | int | bool] = field(default_factory=dict)


RecyclingProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class SimpleSheathSettings:
    gamma_e: float
    gamma_i: float
    secondary_electron_coef: float
    sheath_ion_polytropic: float
    lower_y: bool
    upper_y: bool
    no_flow: bool
    density_boundary_mode: float
    pressure_boundary_mode: float
    temperature_boundary_mode: float
    wall_potential: np.ndarray


@dataclass(frozen=True)
class FullSheathSettings:
    secondary_electron_coef: float
    sin_alpha: np.ndarray
    lower_y: bool
    upper_y: bool
    wall_potential: np.ndarray
    floor_potential: bool


@dataclass(frozen=True)
class DensityFeedbackTerms:
    density_source: dict[str, np.ndarray]
    energy_source: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]
    feedback_integral_rhs: dict[str, float]


@dataclass(frozen=True)
class IonBoundaryResult:
    density: dict[str, np.ndarray]
    pressure: dict[str, np.ndarray]
    temperature: dict[str, np.ndarray]
    velocity: dict[str, np.ndarray]
    momentum: dict[str, np.ndarray]
    energy_source: dict[str, np.ndarray]


@dataclass(frozen=True)
class ElectronBoundaryResult:
    density: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    velocity: np.ndarray
    momentum: np.ndarray
    energy_source: np.ndarray
