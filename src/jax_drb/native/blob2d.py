from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..config.boutinp import BoutConfig, NumericResolver
from .expression import ArrayExpressionEvaluator
from .metrics import StructuredMetrics
from .mesh import StructuredMesh, broadcast_to_field_shape


@dataclass(frozen=True)
class Blob2DBenchmark:
    electron_temperature: float
    curvature_z: np.ndarray
    dz: np.ndarray


@dataclass(frozen=True)
class Blob2DRhsResult:
    electron_density: np.ndarray
    electron_pressure: np.ndarray
    potential: np.ndarray
    density_rhs: np.ndarray
    vorticity_rhs: np.ndarray


def build_blob2d_benchmark(
    config: BoutConfig,
    *,
    mesh: StructuredMesh,
    metrics: StructuredMetrics,
    dataset_scalars: Mapping[str, float],
) -> Blob2DBenchmark:
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    resolver = NumericResolver(config)
    electron_temperature = resolver.resolve("e", "temperature") / float(dataset_scalars["Tnorm"])
    curvature_raw = (
        broadcast_to_field_shape(evaluator.resolve_option("mesh", "bxcvz"), mesh)
        if config.has_option("mesh", "bxcvz")
        else np.zeros((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    )
    curvature_z = 2.0 * float(dataset_scalars["rho_s0"]) ** 2 * np.asarray(curvature_raw, dtype=np.float64)
    return Blob2DBenchmark(
        electron_temperature=electron_temperature,
        curvature_z=curvature_z,
        dz=np.asarray(metrics.dz, dtype=np.float64),
    )


def initialize_blob2d_density(config: BoutConfig, *, mesh: StructuredMesh) -> np.ndarray:
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())
    return np.asarray(broadcast_to_field_shape(evaluator.resolve_option("Ne", "function"), mesh), dtype=np.float64)


def compute_blob2d_rhs(
    electron_density: np.ndarray,
    *,
    benchmark: Blob2DBenchmark,
) -> Blob2DRhsResult:
    density = np.asarray(electron_density, dtype=np.float64)
    pressure = density * benchmark.electron_temperature
    potential = np.zeros_like(density, dtype=np.float64)
    density_rhs = np.zeros_like(density, dtype=np.float64)
    vorticity_rhs = benchmark.curvature_z * (
        np.roll(pressure, shift=-1, axis=-1) - np.roll(pressure, shift=1, axis=-1)
    ) / (2.0 * benchmark.dz)
    return Blob2DRhsResult(
        electron_density=density,
        electron_pressure=pressure,
        potential=potential,
        density_rhs=density_rhs,
        vorticity_rhs=np.asarray(vorticity_rhs, dtype=np.float64),
    )
