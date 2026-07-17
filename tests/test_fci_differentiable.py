"""Fast gate for the differentiable non-axisymmetric FCI flagship (Phase 6).

Covers three things quickly (small grid, two RK4 steps):

1. ``build_shifted_torus_geometry`` produces a valid ``FciGeometry3D`` of the
   requested shape with a finite, non-degenerate, non-orthogonal metric.
2. The drift-reduced two-field FCI RHS is finite on the ported geometry.
3. The differentiability gate: ``jax.grad`` of the evolved density variance
   matches a central finite difference to tight tolerance, both for the full RK4
   rollout and for a single RHS evaluation.

The reusable rollout machinery lives in
``drbx.native.fci_differentiable_case``; the pedagogical script
``examples/stellarator/fci_differentiable.py`` drives the same API.
"""

from __future__ import annotations

import numpy as np
import pytest

from drbx.geometry import FciGeometry3D, build_shifted_torus_geometry
from drbx.native.fci_differentiable_case import (
    build_context,
    differentiability_report,
    seeded_initial_state,
    single_rhs,
    single_rhs_grad_and_fd,
)

SMALL_SHAPE = (10, 10, 6)
SMALL_STEPS = 2
SIGMA = 0.6
AMP0 = 0.1
DT = 1.0e-3
FD_STEP = 1.0e-5
GRAD_FD_TOLERANCE = 1.0e-3


@pytest.fixture(scope="module")
def small_context():
    """The shifted-torus context at tiny size, shared by every gate below."""

    return build_context(SMALL_SHAPE, sigma=SIGMA)


@pytest.fixture(scope="module")
def rollout_report(small_context):
    """Run the differentiable rollout once; the gradient gates assert on it."""

    return differentiability_report(
        small_context, amp0=AMP0, n_steps=SMALL_STEPS, dt=DT, fd_step=FD_STEP
    )


def test_build_shifted_torus_geometry_shape_and_metric() -> None:
    shape = (14, 12, 8)
    geometry = build_shifted_torus_geometry(shape, sigma=SIGMA)

    assert isinstance(geometry, FciGeometry3D)
    assert tuple(int(s) for s in geometry.shape) == shape
    assert tuple(int(s) for s in geometry.grid.shape) == shape

    metric = geometry.cell_metric
    for field in (metric.J, metric.g11, metric.g22, metric.g33, metric.g12, metric.g_12):
        array = np.asarray(field, dtype=np.float64)
        assert array.shape == shape
        assert np.all(np.isfinite(array))

    # Non-degenerate metric: Jacobian bounded away from zero.
    assert float(np.min(np.abs(np.asarray(metric.J)))) > 0.0
    # Non-orthogonal (off-diagonal) cross terms present on the shifted torus.
    assert float(np.max(np.abs(np.asarray(metric.g12)))) > 1.0e-6
    assert float(np.max(np.abs(np.asarray(metric.g_12)))) > 1.0e-6
    # Magnetic field magnitude finite and strictly positive.
    bmag = np.asarray(geometry.cell_bfield.Bmag, dtype=np.float64)
    assert np.all(np.isfinite(bmag))
    assert float(np.min(bmag)) > 0.0


def test_fci_2field_rhs_finite_on_shifted_torus(small_context) -> None:
    rhs = single_rhs(small_context, seeded_initial_state(small_context, AMP0))
    assert np.all(np.isfinite(np.asarray(rhs.density)))
    assert np.all(np.isfinite(np.asarray(rhs.v_parallel)))


def test_rollout_grad_matches_finite_difference(rollout_report) -> None:
    assert rollout_report["finite"] is True
    assert rollout_report["n_steps"] == SMALL_STEPS
    # The differentiability gate: autodiff grad agrees with central FD.
    assert rollout_report["rel_error"] < GRAD_FD_TOLERANCE
    assert abs(rollout_report["grad"]) > 0.0


def test_single_rhs_grad_matches_finite_difference(small_context) -> None:
    single = single_rhs_grad_and_fd(small_context, amp0=AMP0, fd_step=FD_STEP)
    assert single["rel_error"] < GRAD_FD_TOLERANCE
    assert abs(single["grad"]) > 0.0
