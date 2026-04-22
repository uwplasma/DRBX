from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def current_feedback_errors(
    fields: Mapping[str, np.ndarray],
    *,
    controllers: Mapping[str, Any],
    mesh: Any,
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for name, controller in controllers.items():
        density_name = f"N{name}"
        if density_name not in fields:
            continue
        upstream_density = float(np.asarray(fields[density_name], dtype=np.float64)[mesh.xstart, mesh.ystart, 0])
        errors[name] = controller.density_upstream - upstream_density
    return errors


def advance_feedback_integrals(
    fields: Mapping[str, np.ndarray],
    *,
    controllers: Mapping[str, Any],
    feedback_integrals: Mapping[str, float],
    feedback_previous_errors: Mapping[str, float],
    mesh: Any,
    timestep: float,
) -> dict[str, float]:
    updated = {name: float(value) for name, value in feedback_integrals.items()}
    current_errors = current_feedback_errors(fields, controllers=controllers, mesh=mesh)
    for name, controller in controllers.items():
        current_error = float(current_errors.get(name, 0.0))
        previous_error = float(feedback_previous_errors.get(name, current_error))
        integral = float(feedback_integrals.get(name, 0.0)) + float(timestep) * 0.5 * (current_error + previous_error)
        if controller.density_integral_positive and integral < 0.0:
            integral = 0.0
        updated[name] = integral
    return updated


def advance_feedback_integrals_from_predictor(
    *,
    controllers: Mapping[str, Any],
    feedback_integrals: Mapping[str, float],
    feedback_previous_errors: Mapping[str, float],
    predictor_feedback_errors: Mapping[str, float],
    timestep: float,
) -> dict[str, float]:
    updated = {name: float(value) for name, value in feedback_integrals.items()}
    for name, controller in controllers.items():
        previous_error = float(feedback_previous_errors.get(name, 0.0))
        predictor_error = float(predictor_feedback_errors.get(name, previous_error))
        integral = float(feedback_integrals.get(name, 0.0)) + float(timestep) * 0.5 * (predictor_error + previous_error)
        if controller.density_integral_positive and integral < 0.0:
            integral = 0.0
        updated[name] = integral
    return updated


def sanitize_feedback_integrals(
    feedback_integrals: Mapping[str, float],
    *,
    controllers: Mapping[str, Any],
) -> dict[str, float]:
    sanitized = {name: float(value) for name, value in feedback_integrals.items()}
    for name, controller in controllers.items():
        integral = float(sanitized.get(name, 0.0))
        if controller.density_integral_positive and integral < 0.0:
            integral = 0.0
        sanitized[name] = integral
    return sanitized


def feedback_integral_vector(
    feedback_integrals: Mapping[str, float],
    *,
    feedback_names: tuple[str, ...],
) -> np.ndarray:
    return np.asarray([float(feedback_integrals.get(name, 0.0)) for name in feedback_names], dtype=np.float64)


def feedback_error_vector(
    feedback_errors: Mapping[str, float],
    *,
    feedback_names: tuple[str, ...],
) -> np.ndarray:
    return np.asarray([float(feedback_errors.get(name, 0.0)) for name in feedback_names], dtype=np.float64)
