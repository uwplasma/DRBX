"""Logical-frame nodal metric adapter."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class NodalMetricProvider(Protocol):
    """Metric evaluator at logical points ``(Q, 3) = (u, theta, eta)``: ``(h (Q, 3), jac (Q,), B (Q,), K (Q, 3))``."""

    def nodal_metric(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


class _GeometryProviderMetric:
    def __init__(self, provider) -> None:
        self.provider = provider

    def nodal_metric(self, points):
        points = np.asarray(points, dtype=np.float64)
        h, jac = self.provider.bracket_metric_samples(points)
        _J, B, K = self.provider.magnetic_curvature_samples(points)
        return np.asarray(h), np.abs(np.asarray(jac)), np.asarray(B), np.asarray(K)


def metric_from_geometry_provider(provider) -> NodalMetricProvider:
    """Adapt a :class:`~drbx.stencils.geometry_arrays.GeometryProvider`: ``bracket_metric_samples`` (``h``, ``abs`` jacobian) and ``magnetic_curvature_samples`` (``B``, ``K``)."""
    return _GeometryProviderMetric(provider)
