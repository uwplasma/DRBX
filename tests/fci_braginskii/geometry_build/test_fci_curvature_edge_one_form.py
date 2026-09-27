"""Focused checks for the opt-in shared-edge curvature payload.

Trimmed from the SRC/2D_fci version of this file: the vendored
``geometry_build`` subpackage only carries the FCI geometry/metric builder
closure, so tests exercising the outside driver stack were removed rather
than carried with broken imports. Removed here (and why):

- ``test_default_curvature_path_is_unchanged_and_shared_curl_closes``: used
  ``_build_polar_geometry`` from the SRC test helper module
  ``test_fci_geometry_axis_regular_curvature``, which is not one of the
  vendored/carried test files.
- ``test_exact_edge_selector_is_opt_in``: used ``simulate_hsx_blob``'s
  ``_build_parser``, which is out of scope for this vendored subpackage.
- ``test_shared_edge_curvature_faces_preserve_eta_shard_interfaces``: used
  both ``_build_polar_geometry`` (see above) and ``simulate_hsx_blob``'s
  ``_pack_curvature_face_coefficients``/``_unpack_curvature_face_coefficients``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from drbx.fci_braginskii.geometry_build.fci_geometry import (
    CurvatureEdgeOneForm3D,
)
from drbx.fci_braginskii.geometry_build.hsx_fci_builder import (
    _build_hsx_curvature_edge_one_form,
)


def _edge_payload(shape: tuple[int, int, int]) -> CurvatureEdgeOneForm3D:
    nx, ny, nz = shape
    return CurvatureEdgeOneForm3D(
        Az_xy=jnp.arange((nx + 1) * (ny + 1) * nz, dtype=jnp.float64).reshape(
            nx + 1, ny + 1, nz
        ),
        Ay_xz=jnp.ones((nx + 1, ny, nz + 1), dtype=jnp.float64),
        Ax_yz=jnp.zeros((nx, ny + 1, nz + 1), dtype=jnp.float64),
    )


def test_shared_edge_container_validates_shapes_and_finiteness():
    payload = _edge_payload((4, 8, 3))
    assert payload.Az_xy.shape == (5, 9, 3)
    assert payload.Ay_xz.shape == (5, 8, 4)
    assert payload.Ax_yz.shape == (4, 9, 4)
    with pytest.raises(ValueError, match="Ay_xz"):
        CurvatureEdgeOneForm3D(
            Az_xy=jnp.zeros((5, 9, 3)),
            Ay_xz=jnp.zeros((5, 7, 4)),
            Ax_yz=jnp.zeros((4, 9, 4)),
        )
    with pytest.raises(ValueError, match="finite"):
        CurvatureEdgeOneForm3D(
            Az_xy=payload.Az_xy.at[0, 0, 0].set(jnp.nan),
            Ay_xz=payload.Ay_xz,
            Ax_yz=payload.Ax_yz,
        )


def test_hsx_edge_sampling_closes_theta_and_eta_seams_without_axis_queries():
    class FakeMetricEvaluator:
        eta = np.asarray([0.0, 2.0 * np.pi])
        period = 2.0 * np.pi

        def evaluate(self, points):
            points = np.asarray(points)
            assert np.all(points[..., 0] > 0.0)
            eye = np.broadcast_to(np.eye(3), points.shape[:-1] + (3, 3))
            eye = np.array(eye, copy=True)
            eye[..., 0, 0] += np.where(
                np.isclose(points[..., 2], self.period), 3.0, 0.0
            )
            return type("Metric", (), {"g_cov": eye})()

        def evaluate_magnetic_field(self, points, bfield):
            points = np.asarray(points)
            b = np.zeros(points.shape[:-1] + (3,))
            b[..., 0] = 1.0
            b[..., 2] = 1.0
            return type("Magnetic", (), {"B_contravariant": b, "magnitude": np.ones(points.shape[:-1])})()

    payload = _build_hsx_curvature_edge_one_form(
        metric_evaluator=FakeMetricEvaluator(),
        bfield=object(),
        u_faces=np.linspace(0.0, 1.0, 5),
        v_faces=np.linspace(0.0, 2.0 * np.pi, 9),
        eta0=0.0,
        eta_period=2.0 * np.pi,
        nfp=2,
        neta=8,
        reference_magnetic_field=1.0,
    )
    assert payload.Az_xy.shape == (5, 9, 8)
    assert payload.Ay_xz.shape == (5, 8, 9)
    assert payload.Ax_yz.shape == (4, 9, 9)
    assert np.all(np.isfinite(np.asarray(payload.Az_xy)))
    np.testing.assert_array_equal(np.asarray(payload.Az_xy)[:, -1], np.asarray(payload.Az_xy)[:, 0])
    np.testing.assert_array_equal(np.asarray(payload.Ax_yz)[:, -1], np.asarray(payload.Ax_yz)[:, 0])
    np.testing.assert_array_equal(np.asarray(payload.Ay_xz)[:, :, 0], np.asarray(payload.Ay_xz)[:, :, -1])
    np.testing.assert_array_equal(np.asarray(payload.Ax_yz)[:, :, 0], np.asarray(payload.Ax_yz)[:, :, -1])
