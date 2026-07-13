"""Fast gate for the differentiable non-axisymmetric FCI flagship (Phase 6).

Covers three things quickly (small grid, one/two RK4 steps):

1. ``build_shifted_torus_geometry`` produces a valid ``FciGeometry3D`` of the
   requested shape with a finite, non-degenerate, non-orthogonal metric.
2. The drift-reduced two-field FCI RHS is finite on the ported geometry.
3. The differentiability gate: ``jax.grad`` of the evolved density variance
   matches a central finite difference to tight tolerance, both for the full RK4
   rollout and for a single RHS evaluation. Also smoke-tests the demo ``main()``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from jax_drb.geometry import FciGeometry3D, build_shifted_torus_geometry


def _load_demo():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "stellarator" / "fci_differentiable_demo.py"
    spec = importlib.util.spec_from_file_location("fci_differentiable_demo", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo()

SMALL_SHAPE = (10, 10, 6)
SMALL_STEPS = 2
GRAD_FD_TOLERANCE = 1.0e-3


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    """Run the demo once at tiny size; every gate below asserts on its summary."""

    output_dir = tmp_path_factory.mktemp("fci_differentiable")
    summary = demo.main(
        output_dir=output_dir,
        shape=SMALL_SHAPE,
        sigma=0.6,
        n_steps=SMALL_STEPS,
        dt=1.0e-3,
        make_figure=True,
    )
    return output_dir, summary


def test_build_shifted_torus_geometry_shape_and_metric() -> None:
    shape = (14, 12, 8)
    geometry = build_shifted_torus_geometry(shape, sigma=0.6)

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


def test_fci_2field_rhs_finite_on_shifted_torus() -> None:
    ctx = demo.build_context(SMALL_SHAPE, sigma=0.6)
    rhs = demo.single_rhs(ctx, demo.seeded_initial_state(ctx, 0.1))
    assert np.all(np.isfinite(np.asarray(rhs.density)))
    assert np.all(np.isfinite(np.asarray(rhs.v_parallel)))


def test_rollout_grad_matches_finite_difference(demo_run) -> None:
    _output_dir, summary = demo_run
    assert summary["rollout_finite"] is True
    assert summary["differentiation_path"] == "multi_step_rollout"
    # The differentiability gate: autodiff grad agrees with central FD.
    assert summary["rollout_rel_error"] < GRAD_FD_TOLERANCE
    assert abs(summary["rollout_grad"]) > 0.0


def test_single_rhs_grad_matches_finite_difference(demo_run) -> None:
    _output_dir, summary = demo_run
    assert summary["single_rhs_finite"] is True
    assert summary["single_rhs_rel_error"] < GRAD_FD_TOLERANCE
    assert abs(summary["single_rhs_grad"]) > 0.0


def test_demo_main_smoke_writes_outputs(demo_run) -> None:
    output_dir, summary = demo_run
    assert (output_dir / "fci_differentiable_summary.json").exists()
    assert (output_dir / "fci_differentiable.png").exists()
    assert summary["geometry_shape"] == [int(s) for s in SMALL_SHAPE]
    assert summary["n_steps"] == SMALL_STEPS
