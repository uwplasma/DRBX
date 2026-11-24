"""Geometry-only subset of the P continuum reference and provider."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ...evaluators.magnetic_field_evaluator import bfield_evaluator_from_makegrid
from ...evaluators.metric_evaluator import MetricEvaluator

from .curvature_reference import AutodiffCurvatureReference
from .eta_filter import apply_eta_filter
from .options import PerpendicularBuildOptions


class ContinuumMmsReference:
    def __init__(self, metric_evaluator, bfield_evaluator, B0):
        if not np.isfinite(B0) or B0 <= 0.0:
            raise ValueError("B0 must be finite and positive")
        self.metric_evaluator = metric_evaluator
        self.bfield_evaluator = bfield_evaluator
        self.B0 = float(B0)
        self.metric_query_batch_size = 4096

    def _metric_batch(self, q: np.ndarray) -> dict[str, np.ndarray]:
        metric = self.metric_evaluator.evaluate(q, reject_nonpositive_J=False)
        if hasattr(self.metric_evaluator, "project_magnetic_field"):
            magnetic = self.metric_evaluator.project_magnetic_field(
                metric, self.bfield_evaluator
            )
        else:
            magnetic = self.metric_evaluator.evaluate_magnetic_field(
                q, self.bfield_evaluator, reject_nonpositive_J=False
            )
        J = np.asarray(metric.signed_J, dtype=np.float64)
        gcov = np.asarray(metric.covariant_metric, dtype=np.float64)
        gcontra = np.asarray(metric.contravariant_metric, dtype=np.float64)
        bcontra = np.asarray(magnetic.B_contravariant, dtype=np.float64) / self.B0
        bmag = np.asarray(magnetic.magnitude, dtype=np.float64) / self.B0
        bmag = np.maximum(bmag, 1.0e-30)
        bunit = bcontra / bmag[..., None]
        bcov = np.einsum("...ij,...j->...i", gcov, bunit)
        return {"J": J, "gcov": gcov, "gcontra": gcontra, "b": bunit, "bcov": bcov, "B": bmag}

    def _metric(self, q: np.ndarray) -> dict[str, np.ndarray]:
        points = np.asarray(q, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("metric queries must have shape (n, 3)")
        size = self.metric_query_batch_size
        if len(points) <= size:
            return self._metric_batch(points)
        chunks = [
            self._metric_batch(points[first : first + size])
            for first in range(0, len(points), size)
        ]
        return {
            name: np.concatenate([chunk[name] for chunk in chunks], axis=0)
            for name in chunks[0]
        }


@contextmanager
def metric_reuse(reference):
    """Cache only identical queries within one kernel call, then release memory.

    Do not round coordinates or merge differing finite-difference stencils.
    References are private to each single-threaded campaign worker.
    """
    original = reference._metric
    had_instance = '_metric' in vars(reference)
    saved = vars(reference).get('_metric')
    cache = {}
    def metric(points):
        q = np.asarray(points, dtype=np.float64)
        key = (q.shape, q.tobytes())
        if key not in cache:
            if len(cache) >= 32:
                cache.clear()
            cache[key] = original(q)
        return cache[key]
    reference._metric = metric
    try:
        yield
    finally:
        if had_instance:
            reference._metric = saved
        else:
            del reference._metric


def curvature_geometry(reference, points):
    """Only J/B/K are consumed by the curvature campaign; preserve its K stencil."""
    with metric_reuse(reference):
        metric = reference._metric(points)
        curvature = reference._curvature(points)
    return SimpleNamespace(J=metric['J'], B=metric['B'], K=curvature)


class ScriptsGeometryProvider:
    def __init__(self, reference):
        self._reference = reference

    @property
    def reference(self):
        return self._reference

    def bracket_metric_samples(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        metric = self._reference._metric(points)
        h = metric["bcov"] / metric["B"][:, None]
        jac = np.abs(metric["J"])
        return h, jac

    # -- SBP curvature derivative and matrix wall SAT -------------------------------------------------------------
    def magnetic_curvature_samples(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        prepared = curvature_geometry(self._reference, points)
        return np.asarray(prepared.J), np.asarray(prepared.B), np.asarray(prepared.K)


def build_reference(options: PerpendicularBuildOptions, provenance: dict):
    """Restore the continuous evaluator and select the Cartesian field."""
    with np.load(Path(options.metric_cache), allow_pickle=False) as cached:
        metric = MetricEvaluator.from_cache_payload(cached, prefix="metric_evaluator_")
        B0 = float(np.asarray(cached["reference_magnetic_field"]).item())
    provenance["reference_magnetic_field"] = B0
    map_periods, _ = options.period_counts
    if metric.nfp is not None and metric.nfp != map_periods:
        raise ValueError(f"metric field periods {metric.nfp} differ from the requested map periods {map_periods}")
    field = bfield_evaluator_from_makegrid(
        options.makegrid, currents=np.asarray(options.makegrid_currents, dtype=np.float64),
        method="cubic", toroidal_method=options.bfield_toroidal, extrapolate=False)
    if options.eta_filter is not None:
        filter_nfp = options.eta_filter.nfp
        if map_periods % filter_nfp or field.nfp % filter_nfp:
            raise ValueError("the filter span must contain whole metric and MAKEGRID periods")
    reference = ContinuumMmsReference(metric, field, B0)
    apply_eta_filter(reference, provenance, options)
    return ScriptsGeometryProvider(AutodiffCurvatureReference(reference))
