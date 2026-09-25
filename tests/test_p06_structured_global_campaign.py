"""Contracts for the portable structured P06 curvature campaign."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CAMPAIGN = SCRIPTS / "p06_structured_global"
for entry in (ROOT, ROOT / "src", SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f"p06_structured_test_{name}", CAMPAIGN / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def numeric():
    return _module("numerics")


def test_dirichlet_catalogue_has_prescribed_zero_and_varying_traces_with_normal_gradient(numeric):
    period = 2 * np.pi
    theta = np.linspace(0, 2*np.pi, 11, endpoint=False)
    wall = np.column_stack((np.ones(len(theta)), theta, 0.17 + 0*theta))
    zero, dzero = numeric._dirichlet_fields(wall, period, variable=False)
    varying, dvarying = numeric._dirichlet_fields(wall, period, variable=True)
    assert np.max(np.abs(zero[4])) < 1e-14
    assert np.min(np.abs(dzero[4, :, 0])) > 0.01
    assert np.ptp(varying[4]) > 0.02
    assert np.max(np.abs(dvarying[4, :, 0])) > 0.01
    assert np.min(zero[:3]) > 0 and np.min(varying[:3]) > 0


def test_trace_callback_preserves_value_and_all_gradients_across_repeated_points(numeric):
    reference = SimpleNamespace(eta_period=2*np.pi)
    points = np.array(((1.,0.3,0.1),(1.,0.3,0.1),(1.,1.2,0.2)))
    callback = numeric._trace("variable_dirichlet", reference, 0.37)
    first = callback(points)
    second = callback(points[::-1])
    assert first[0].shape == (3,5) and first[1].shape == (3,3,5)
    np.testing.assert_array_equal(first[0][::-1], second[0])
    np.testing.assert_array_equal(first[1][::-1], second[1])
    assert abs(first[1][0,0,4]) > 1e-3


def test_curvature_geometry_avoids_unneeded_full_rhs_preparation(numeric):
    class Reference:
        def __init__(self):
            self.metric_calls = 0
            self.curvature_calls = 0

        def _metric(self, points):
            self.metric_calls += 1
            return {"J": 2 + points[:, 0], "B": 3 + points[:, 1]}

        def _curvature(self, points):
            self.curvature_calls += 1
            return np.column_stack((points[:, 0], points[:, 1], points[:, 2]))

        def prepare(self, points):
            raise AssertionError("full perpendicular RHS geometry is not used by P06")

    points = np.array(((0.2, 0.3, 0.4), (0.6, 0.7, 0.8)))
    reference = Reference()
    geometry = numeric._curvature_geometry(reference, points)
    np.testing.assert_array_equal(geometry.J, 2 + points[:, 0])
    np.testing.assert_array_equal(geometry.B, 3 + points[:, 1])
    np.testing.assert_array_equal(geometry.K, points)
    assert (reference.metric_calls, reference.curvature_calls) == (1, 1)


def test_face_incidence_preserves_radial_boundaries_and_periodic_seams(numeric):
    n = 8
    keys = np.array(((0,0,2,3),(0,n,2,3),(1,2,0,3),(1,2,n,3),(2,2,3,0),(2,2,3,n)))
    lower, lv, upper, uv = numeric._face_incidence(n, keys)
    assert not lv[0] and uv[0] and upper[0] == np.ravel_multi_index((0,2,3),(n,n,n))
    assert lv[1] and not uv[1] and lower[1] == np.ravel_multi_index((n-1,2,3),(n,n,n))
    assert lv[2] and uv[2] and lower[2] == np.ravel_multi_index((2,n-1,3),(n,n,n))
    assert lv[3] and uv[3] and upper[3] == np.ravel_multi_index((2,0,3),(n,n,n))
    assert lv[4] and uv[4] and lower[4] == np.ravel_multi_index((2,3,n-1),(n,n,n))
    assert lv[5] and uv[5] and upper[5] == np.ravel_multi_index((2,3,0),(n,n,n))


def test_global_plan_requires_complete_independent_reference_coverage():
    runner = _module("parallel_runner")
    n = 2
    numeric = SimpleNamespace(_face_count=lambda _: 3*(n+1)*n*n)
    base = {"schema":runner.PLAN_SCHEMA,"resolution":n,"coverage":"global"}
    units = [
        {"id":"faces","kind":"face","first":0,"last":numeric._face_count(n)},
        {"id":"cells","kind":"cell","first":0,"last":n**3},
    ]
    with pytest.raises(ValueError, match="incomplete global reference coverage"):
        runner._validate_plan({**base,"units":units},numeric)
    runner._validate_plan({**base,"units":[*units,{"id":"reference","kind":"reference","first":0,"last":n**3}]},numeric)


def test_allocation_memory_cap_is_applied_before_worker_start():
    runner = _module("parallel_runner")
    assert runner._effective_worker_count(8,memory_budget_gib=18,worker_memory_gib=4,
                                          memory_reserve_gib=2) == 4
    with pytest.raises(ValueError, match="cannot accommodate"):
        runner._effective_worker_count(2,memory_budget_gib=3,worker_memory_gib=4,
                                       memory_reserve_gib=1)
