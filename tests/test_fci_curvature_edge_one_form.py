"""Focused checks for the opt-in shared-edge curvature payload."""

from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry import (
    CurvatureEdgeOneForm3D,
    HaloLayout3D,
    build_local_curvature_face_coefficients,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fci_geometry_axis_regular_curvature import _build_polar_geometry
from drbx.geometry.hsx_fci_builder import (
    _build_hsx_curvature_edge_one_form,
)
from generate_hsx_fci_geometry import _parser as _producer_parser
from simulate_hsx_blob import (  # noqa: E402
    _build_parser as _consumer_parser,
    _pack_curvature_face_coefficients,
    _unpack_curvature_face_coefficients,
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


def test_default_curvature_path_is_unchanged_and_shared_curl_closes():
    geometry, domain = _build_polar_geometry(shape=(6, 12, 4), halo_width=1)
    default = build_local_curvature_face_coefficients(geometry, domain)
    explicit_default = build_local_curvature_face_coefficients(
        geometry, domain, shared_edge_one_form=None
    )
    for lhs, rhs in zip(default.axes, explicit_default.axes):
        np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))

    payload = _edge_payload(geometry.owned_shape)
    coefficients = build_local_curvature_face_coefficients(
        geometry, domain, shared_edge_one_form=payload
    )
    qx, qy, qz = coefficients.axes
    dx = geometry.spacing.dx_owned
    dy = geometry.spacing.dy_owned
    dz = geometry.spacing.dz_owned
    divergence = (
        (qx[1:] - qx[:-1]) / dx
        + (qy[:, 1:] - qy[:, :-1]) / dy
        + (qz[:, :, 1:] - qz[:, :, :-1]) / dz
    )
    assert float(jnp.max(jnp.abs(divergence))) < 2.0e-12


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


def test_exact_edge_selector_is_opt_in():
    base = (
        "--resolution", "4", "8", "8",
        "--metric-mesh-shape", "4", "8", "4",
        "--output", "artifact",
    )
    parser = _producer_parser()
    assert parser.parse_args(base).include_curvature_edge_one_form is False
    assert parser.parse_args(base + ("--include-curvature-edge-one-form",)).include_curvature_edge_one_form is True
    assert not hasattr(_consumer_parser().parse_args(()), "curvature_edge_one_form")


def test_shared_edge_curvature_faces_preserve_eta_shard_interfaces():
    """Packed host faces lower to eta shards without changing shared faces."""

    geometry, domain = _build_polar_geometry(shape=(6, 12, 6), halo_width=1)
    coefficients = build_local_curvature_face_coefficients(
        geometry,
        domain,
        shared_edge_one_form=_edge_payload(geometry.owned_shape),
    )
    packed = _pack_curvature_face_coefficients(coefficients)

    local_coefficients = []
    for start, stop in ((0, 3), (3, 6)):
        local_layout = HaloLayout3D((6, 12, stop - start), halo_width=1)
        local = _unpack_curvature_face_coefficients(
            packed[:, :, start:stop],
            local_layout,
        )
        local_coefficients.append(local)
        np.testing.assert_array_equal(
            np.asarray(local.x),
            np.asarray(coefficients.x)[:, :, start:stop],
        )
        np.testing.assert_array_equal(
            np.asarray(local.y),
            np.asarray(coefficients.y)[:, :, start:stop],
        )
        np.testing.assert_array_equal(
            np.asarray(local.z),
            np.asarray(coefficients.z)[:, :, start : stop + 1],
        )

    np.testing.assert_array_equal(
        np.asarray(local_coefficients[0].z)[:, :, -1],
        np.asarray(local_coefficients[1].z)[:, :, 0],
    )
