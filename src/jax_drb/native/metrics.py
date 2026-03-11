from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp

from ..config.boutinp import BoutConfig
from ..config.model import has_model_section, locate_model_section
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
    g33: jnp.ndarray
    g22: jnp.ndarray
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
    normalize_metric = _normalize_metric_enabled(config)

    raw_dx = _metric_value(config, evaluator, "dx", default=1.0)
    raw_dy = _metric_value(config, evaluator, "dy", default=1.0)
    raw_dz = _metric_value(config, evaluator, "dz", default=(2.0 * math.pi) / float(mesh.nz))
    raw_J = _metric_value(config, evaluator, "J", default=1.0)
    raw_g11 = _metric_value(config, evaluator, "g11", default=1.0)
    raw_g33 = _metric_value(config, evaluator, "g33", default=1.0)
    raw_g22 = _metric_value(config, evaluator, "g22", default=1.0)
    raw_g23 = _metric_value(config, evaluator, "g23", default=0.0)
    raw_Bxy = _metric_value(config, evaluator, "Bxy", default=1.0)

    return StructuredMetrics(
        dx=_normalize_dx(broadcast_to_field_shape(raw_dx, mesh), normalize_metric=normalize_metric, rho_s0=rho_s0, Bnorm=Bnorm),
        dy=broadcast_to_field_shape(raw_dy, mesh),
        dz=broadcast_to_field_shape(raw_dz, mesh),
        J=_normalize_J(broadcast_to_field_shape(raw_J, mesh), normalize_metric=normalize_metric, rho_s0=rho_s0),
        g11=_normalize_g11(broadcast_to_field_shape(raw_g11, mesh), normalize_metric=normalize_metric, rho_s0=rho_s0),
        g33=_normalize_g33(broadcast_to_field_shape(raw_g33, mesh), normalize_metric=normalize_metric, rho_s0=rho_s0),
        g22=broadcast_to_field_shape(raw_g22, mesh),
        g23=broadcast_to_field_shape(raw_g23, mesh),
        Bxy=_normalize_Bxy(broadcast_to_field_shape(raw_Bxy, mesh), normalize_metric=normalize_metric, Bnorm=Bnorm),
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


def _normalize_metric_enabled(config: BoutConfig) -> bool:
    if not has_model_section(config):
        return True
    model_section = locate_model_section(config)
    if not config.has_option(model_section, "normalise_metric"):
        return True
    return bool(config.parsed(model_section, "normalise_metric"))


def _normalize_dx(value: jnp.ndarray, *, normalize_metric: bool, rho_s0: float, Bnorm: float) -> jnp.ndarray:
    if not normalize_metric:
        return value
    return value / (rho_s0 * rho_s0 * Bnorm)


def _normalize_J(value: jnp.ndarray, *, normalize_metric: bool, rho_s0: float) -> jnp.ndarray:
    if not normalize_metric:
        return value
    return value / rho_s0


def _normalize_g11(value: jnp.ndarray, *, normalize_metric: bool, rho_s0: float) -> jnp.ndarray:
    if not normalize_metric:
        return value
    return value / (rho_s0 * rho_s0)


def _normalize_g33(value: jnp.ndarray, *, normalize_metric: bool, rho_s0: float) -> jnp.ndarray:
    if not normalize_metric:
        return value
    return value * (rho_s0 * rho_s0)


def _normalize_Bxy(value: jnp.ndarray, *, normalize_metric: bool, Bnorm: float) -> jnp.ndarray:
    if not normalize_metric:
        return value
    return value / Bnorm
