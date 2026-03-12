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

    if _recalculate_metric_enabled(run_config):
        recalculated = _recalculate_orthogonal_metrics(
            config,
            evaluator,
            mesh,
            rho_s0=rho_s0,
            Bnorm=Bnorm,
            raw_dx=raw_dx,
            raw_dy=raw_dy,
            raw_dz=raw_dz,
            raw_Bxy=raw_Bxy,
        )
        raw_dx = recalculated.dx
        raw_dy = recalculated.dy
        raw_dz = recalculated.dz
        raw_J = recalculated.J
        raw_g11 = recalculated.g11
        raw_g33 = recalculated.g33
        raw_g22 = recalculated.g22
        raw_g23 = recalculated.g23
        raw_Bxy = recalculated.Bxy

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


def _recalculate_metric_enabled(run_config: RunConfiguration) -> bool:
    return bool(run_config.normalization and run_config.normalization.metric_policy.recalculate_metric)


def _recalculate_orthogonal_metrics(
    config: BoutConfig,
    evaluator: ArrayExpressionEvaluator,
    mesh: StructuredMesh,
    *,
    rho_s0: float,
    Bnorm: float,
    raw_dx: Any,
    raw_dy: Any,
    raw_dz: Any,
    raw_Bxy: Any,
) -> StructuredMetrics:
    Rxy = _metric_value(config, evaluator, "Rxy", default=1.0)
    Bpxy = _metric_value(config, evaluator, "Bpxy", default=1.0)
    Btxy = _metric_value(config, evaluator, "Btxy", default=0.0)
    hthe = _metric_value(config, evaluator, "hthe", default=1.0)
    sinty = _metric_value(config, evaluator, "sinty", default=0.0)

    normalized_Rxy = broadcast_to_field_shape(Rxy, mesh) / rho_s0
    normalized_hthe = broadcast_to_field_shape(hthe, mesh) / rho_s0
    normalized_sinty = broadcast_to_field_shape(sinty, mesh) * (rho_s0 * rho_s0 * Bnorm)
    normalized_Bpxy = broadcast_to_field_shape(Bpxy, mesh) / Bnorm
    normalized_Btxy = broadcast_to_field_shape(Btxy, mesh) / Bnorm
    normalized_Bxy = broadcast_to_field_shape(raw_Bxy, mesh) / Bnorm

    if run_parallel_transform_type(config) == "shifted":
        normalized_sinty = jnp.zeros_like(normalized_sinty, dtype=jnp.float64)

    sign_Bp = jnp.where(jnp.min(normalized_Bpxy) < 0.0, -1.0, 1.0)
    g11_normalized = jnp.square(normalized_Rxy * normalized_Bpxy)
    g22 = 1.0 / jnp.square(normalized_hthe)
    g33_normalized = jnp.square(normalized_sinty) * g11_normalized + jnp.square(normalized_Bxy) / g11_normalized
    g23 = -sign_Bp * normalized_Btxy / (normalized_hthe * normalized_Bpxy * normalized_Rxy)
    J_normalized = normalized_hthe / normalized_Bpxy

    return StructuredMetrics(
        dx=broadcast_to_field_shape(raw_dx, mesh),
        dy=broadcast_to_field_shape(raw_dy, mesh),
        dz=broadcast_to_field_shape(raw_dz, mesh),
        J=J_normalized * rho_s0,
        g11=g11_normalized * (rho_s0 * rho_s0),
        g33=g33_normalized / (rho_s0 * rho_s0),
        g22=g22,
        g23=g23,
        Bxy=broadcast_to_field_shape(raw_Bxy, mesh),
    )


def run_parallel_transform_type(config: BoutConfig) -> str:
    if not config.has_section("mesh:paralleltransform") or not config.has_option("mesh:paralleltransform", "type"):
        return "identity"
    return str(config.parsed("mesh:paralleltransform", "type"))


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
