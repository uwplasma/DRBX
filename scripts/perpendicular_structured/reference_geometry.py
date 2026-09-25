"""Batch-local exact-coordinate metric reuse for static reference evaluation."""
from contextlib import contextmanager
from functools import wraps
from types import SimpleNamespace
import numpy as np


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


def reuse_metrics(function):
    @wraps(function)
    def wrapped(context, reference, *args, **kwargs):
        with metric_reuse(reference):
            return function(context, reference, *args, **kwargs)
    return wrapped


def curvature_geometry(reference, points):
    """Only J/B/K are consumed by the curvature campaign; preserve its K stencil."""
    with metric_reuse(reference):
        metric = reference._metric(points)
        curvature = reference._curvature(points)
    return SimpleNamespace(J=metric['J'], B=metric['B'], K=curvature)
