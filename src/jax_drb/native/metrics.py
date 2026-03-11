from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp

from ..config.boutinp import BoutConfig
from ..runtime.run_config import RunConfiguration
from .expression import ArrayExpressionEvaluator
from .mesh import StructuredMesh, broadcast_to_field_shape
from .units import resolved_dataset_scalars


@dataclass(frozen=True)
class StructuredMetrics:
    dx: jnp.ndarray
    dy: jnp.ndarray
    dz: jnp.ndarray
    J: jnp.ndarray
    g11: jnp.ndarray
    g23: jnp.ndarray
    Bxy: jnp.ndarray


def build_structured_metrics(
    config: BoutConfig,
    run_config: RunConfiguration,
    mesh: StructuredMesh,
) -> StructuredMetrics:
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    scalars = resolved_dataset_scalars(run_config)
    rho_s0 = scalars["rho_s0"]
    Bnorm = scalars["Bnorm"]

    raw_dx = _metric_value(config, evaluator, "dx", default=1.0)
    raw_dy = _metric_value(config, evaluator, "dy", default=1.0)
    raw_dz = _metric_value(config, evaluator, "dz", default=1.0)
    raw_J = _metric_value(config, evaluator, "J", default=1.0)
    raw_g11 = _metric_value(config, evaluator, "g11", default=1.0)
    raw_g23 = _metric_value(config, evaluator, "g23", default=0.0)
    raw_Bxy = _metric_value(config, evaluator, "Bxy", default=1.0)

    return StructuredMetrics(
        dx=broadcast_to_field_shape(raw_dx, mesh) / (rho_s0 * rho_s0 * Bnorm),
        dy=broadcast_to_field_shape(raw_dy, mesh),
        dz=broadcast_to_field_shape(raw_dz, mesh),
        J=broadcast_to_field_shape(raw_J, mesh) / rho_s0,
        g11=broadcast_to_field_shape(raw_g11, mesh) / (rho_s0 * rho_s0),
        g23=broadcast_to_field_shape(raw_g23, mesh),
        Bxy=broadcast_to_field_shape(raw_Bxy, mesh) / Bnorm,
    )


def _metric_value(
    config: BoutConfig,
    evaluator: ArrayExpressionEvaluator,
    key: str,
    *,
    default: float,
) -> Any:
    if not config.has_section("mesh") or not config.has_option("mesh", key):
        return default
    return evaluator.resolve_option("mesh", key)
