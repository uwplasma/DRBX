"""Focused tests for the simulation-driver FCI map cache contract."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import jax.numpy as jnp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drbx.geometry import CellCenteredGrid3D, FciMaps3D, Grid1D  # noqa: E402
import simulate_hsx_blob as hsx  # noqa: E402


def _grid(shape=(3, 4, 5)):
    nx, ny, nz = shape
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 2.0 * np.pi, ny + 1)
    z_faces = np.linspace(0.0, 2.0 * np.pi, nz + 1)
    return CellCenteredGrid3D(
        x=Grid1D(
            centers=jnp.asarray(0.5 * (x_faces[:-1] + x_faces[1:])),
            faces=jnp.asarray(x_faces),
        ),
        y=Grid1D(
            centers=jnp.asarray(0.5 * (y_faces[:-1] + y_faces[1:])),
            faces=jnp.asarray(y_faces),
        ),
        z=Grid1D(
            centers=jnp.asarray(0.5 * (z_faces[:-1] + z_faces[1:])),
            faces=jnp.asarray(z_faces),
        ),
    )


def _identity_maps(grid, *, shifted=False):
    nx, ny, nz = grid.shape
    x = np.asarray(grid.x.centers)
    y = np.asarray(grid.y.centers)
    z = np.asarray(grid.z.centers)
    ii = np.arange(nx, dtype=float)[:, None, None]
    jj = np.arange(ny, dtype=float)[None, :, None]
    kk = np.arange(nz)
    endpoint_forward_z = z[np.mod(kk + 1, nz)][None, None, :]
    endpoint_backward_z = z[np.mod(kk - 1, nz)][None, None, :]
    endpoint_x = x[:, None, None]
    endpoint_y = y[None, :, None]
    if shifted:
        forward_x = np.clip(ii + 0.25, 0.0, nx - 1.0)
        backward_x = np.clip(ii - 0.25, 0.0, nx - 1.0)
        dy = float(y[1] - y[0])
        forward_y = np.mod(jj + 0.5, float(ny))
        backward_y = np.mod(jj - 0.5, float(ny))
        endpoint_forward_y = np.mod(endpoint_y + 0.5 * dy, 2.0 * np.pi)
        endpoint_backward_y = np.mod(endpoint_y - 0.5 * dy, 2.0 * np.pi)
    else:
        forward_x = backward_x = np.broadcast_to(ii, grid.shape).copy()
        forward_y = backward_y = np.broadcast_to(jj, grid.shape).copy()
        endpoint_forward_y = endpoint_backward_y = np.broadcast_to(
            endpoint_y, grid.shape
        ).copy()
    arrays = {
        "forward_x": np.broadcast_to(forward_x, grid.shape).copy(),
        "forward_y": np.broadcast_to(forward_y, grid.shape).copy(),
        "backward_x": np.broadcast_to(backward_x, grid.shape).copy(),
        "backward_y": np.broadcast_to(backward_y, grid.shape).copy(),
        "forward_endpoint_x": np.broadcast_to(endpoint_x, grid.shape).copy(),
        "forward_endpoint_y": np.broadcast_to(endpoint_forward_y, grid.shape).copy(),
        "forward_endpoint_z": np.broadcast_to(endpoint_forward_z, grid.shape).copy(),
        "backward_endpoint_x": np.broadcast_to(endpoint_x, grid.shape).copy(),
        "backward_endpoint_y": np.broadcast_to(endpoint_backward_y, grid.shape).copy(),
        "backward_endpoint_z": np.broadcast_to(endpoint_backward_z, grid.shape).copy(),
        "forward_endpoint_b_contra_x": np.zeros(grid.shape),
        "forward_endpoint_b_contra_y": np.zeros(grid.shape),
        "forward_endpoint_b_contra_z": np.ones(grid.shape),
        "forward_endpoint_bmag": np.ones(grid.shape),
        "backward_endpoint_b_contra_x": np.zeros(grid.shape),
        "backward_endpoint_b_contra_y": np.zeros(grid.shape),
        "backward_endpoint_b_contra_z": np.ones(grid.shape),
        "backward_endpoint_bmag": np.ones(grid.shape),
        "forward_length": np.full(grid.shape, 1.25),
        "backward_length": np.full(grid.shape, 1.10),
        "forward_boundary": np.zeros(grid.shape, dtype=bool),
        "backward_boundary": np.zeros(grid.shape, dtype=bool),
    }
    return arrays


def test_identity_and_shift_maps_pass_full_torus_quality_validation():
    grid = _grid()
    for shifted in (False, True):
        maps = FciMaps3D(**_identity_maps(grid, shifted=shifted))
        report = hsx.validate_hsx_fci_maps(maps, grid)
        assert report["valid"] is True
        assert report["checks"]["periodic_eta_seam"] is True
        assert report["counts"]["forward_boundary"] == 0


def test_axis_regular_ghost_range_and_periodic_eta_seam_are_allowed():
    grid = _grid()
    arrays = _identity_maps(grid)
    nx, ny, nz = grid.shape
    arrays["forward_x"][0, :, :] = -0.5
    arrays["backward_x"][-1, :, :] = nx - 0.5
    arrays["forward_y"][:, 0, :] = -0.5e-10
    arrays["backward_y"][:, -1, :] = ny + 0.5e-10

    z_centers = np.asarray(grid.z.centers)
    z_period = float(grid.z.faces[-1] - grid.z.faces[0])
    arrays["forward_endpoint_z"][..., -1] = z_centers[0] + z_period
    arrays["backward_endpoint_z"][..., 0] = z_centers[-1] - z_period

    report = hsx.validate_hsx_fci_maps(FciMaps3D(**arrays), grid)
    assert report["valid"] is True
    assert report["checks"]["cell_centered_fractional_coordinates"] is True
    assert report["checks"]["periodic_eta_seam"] is True


def test_fci_map_cache_roundtrip_preserves_payload_and_schema():
    grid = _grid()
    maps = FciMaps3D(**_identity_maps(grid, shifted=True))
    original = {"metric_sentinel": np.asarray([7.0])}
    payload = hsx.add_fci_maps_to_metric_cache_payload(original, maps)
    assert np.array_equal(payload["metric_sentinel"], original["metric_sentinel"])
    assert set(hsx.FCI_MAP_FIELDS) == {
        "forward_x", "forward_y", "backward_x", "backward_y",
        "forward_endpoint_x", "forward_endpoint_y", "forward_endpoint_z",
        "backward_endpoint_x", "backward_endpoint_y", "backward_endpoint_z",
        "forward_endpoint_b_contra_x", "forward_endpoint_b_contra_y",
        "forward_endpoint_b_contra_z", "forward_endpoint_bmag",
        "backward_endpoint_b_contra_x", "backward_endpoint_b_contra_y",
        "backward_endpoint_b_contra_z", "backward_endpoint_bmag",
        "forward_length", "backward_length", "forward_boundary",
        "backward_boundary",
    }
    restored = hsx.fci_maps_from_metric_cache_payload(
        payload, expected_shape=grid.shape
    )
    for name in hsx.FCI_MAP_FIELDS:
        assert np.array_equal(np.asarray(getattr(restored, name)), np.asarray(getattr(maps, name)))


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda arrays: arrays["forward_length"].__setitem__((0, 0, 0), 0.0), "positive"),
        (lambda arrays: arrays["forward_x"].__setitem__((0, 0, 0), np.nan), "NaN"),
        (lambda arrays: arrays["forward_x"].__setitem__((0, 0, 0), 2.75), "fractional"),
        (lambda arrays: arrays["forward_y"].__setitem__((0, 0, 0), 4.25), "fractional"),
        (lambda arrays: arrays["forward_endpoint_z"].__setitem__((0, 0, -1), 0.0), "adjacent"),
        (lambda arrays: arrays["forward_boundary"].__setitem__((0, 0, 0), True), "lower-axis"),
    ],
)
def test_malformed_maps_are_rejected(mutation, message):
    grid = _grid()
    arrays = _identity_maps(grid)
    mutation(arrays)
    report = hsx.fci_map_quality_report(arrays, grid)
    assert report["valid"] is False
    assert any(message in error for error in report["errors"])
    with pytest.raises(ValueError, match="invalid HSX toroidal FCI maps"):
        hsx.validate_hsx_fci_maps(arrays, grid)


def test_malformed_cached_maps_fail_closed_before_construction():
    grid = _grid()
    maps = FciMaps3D(**_identity_maps(grid))
    payload = hsx.fci_maps_to_metric_cache_payload(maps)
    del payload["fci_maps_forward_x"]
    with pytest.raises(KeyError, match="missing"):
        hsx.fci_maps_from_metric_cache_payload(payload, expected_shape=grid.shape)

    payload = hsx.fci_maps_to_metric_cache_payload(maps)
    payload["fci_maps_shape"] = np.asarray((3, 4, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="does not match"):
        hsx.fci_maps_from_metric_cache_payload(payload, expected_shape=grid.shape)


def test_hsx_fci_map_cache_miss_traces_continuous_callback_and_reuses_cache(
    tmp_path, monkeypatch
):
    grid = _grid()
    expected = _identity_maps(grid)
    builder_calls = []
    evaluator_calls = []

    class FakeMetricEvaluator:
        def evaluate_magnetic_field(self, points, bfield):
            evaluator_calls.append((np.asarray(points), bfield))
            points = np.asarray(points)
            return SimpleNamespace(
                B_contravariant=np.tile(np.asarray((0.0, 0.0, 1.0)), (points.shape[0], 1)),
                magnitude=np.ones(points.shape[0]),
            )

    def fake_builder(grid_arg, callback, **kwargs):
        builder_calls.append(kwargs)
        callback(np.asarray([[0.25, 0.5, 0.75]]))
        return {name: jnp.asarray(value) for name, value in expected.items()} | {
            "dz": jnp.ones(grid_arg.shape)
        }

    monkeypatch.setattr(hsx, "build_fci_maps_from_callbacks", fake_builder)
    cache_path = tmp_path / "metric.npz"
    base_payload = {"metric_sentinel": np.asarray([7.0])}
    maps, payload, _ = hsx._build_or_load_hsx_fci_maps(
        grid=grid,
        topology="toroidal",
        construct_fci_maps=True,
        fci_trace_substeps=4,
        cache_payload=base_payload,
        cache_path=cache_path,
        metric_evaluator=FakeMetricEvaluator(),
        bfield=object(),
        makegrid_path=tmp_path / "mgrid.nc",
    )
    assert maps is not None
    assert len(builder_calls) == 1
    assert len(evaluator_calls) == 1
    assert cache_path.is_file()
    assert np.array_equal(payload["metric_sentinel"], np.asarray([7.0]))

    with np.load(cache_path, allow_pickle=False) as cached:
        cached_payload = {name: np.array(cached[name], copy=True) for name in cached.files}
    maps_again, _, _ = hsx._build_or_load_hsx_fci_maps(
        grid=grid,
        topology="toroidal",
        construct_fci_maps=True,
        fci_trace_substeps=4,
        cache_payload=cached_payload,
        cache_path=cache_path,
        metric_evaluator=None,
        bfield=None,
        makegrid_path=tmp_path / "missing-mgrid.nc",
    )
    assert maps_again is not None
    assert len(builder_calls) == 1


def test_hsx_fci_map_construction_fails_closed_for_square_topology(tmp_path):
    with pytest.raises(ValueError, match="only for topology='toroidal'"):
        hsx._build_or_load_hsx_fci_maps(
            grid=_grid(),
            topology="square",
            construct_fci_maps=True,
            fci_trace_substeps=4,
            cache_payload=None,
            cache_path=None,
            metric_evaluator=None,
            bfield=None,
            makegrid_path=tmp_path / "mgrid.nc",
        )


def test_hsx_fci_map_cache_regenerates_invalid_maps(tmp_path, monkeypatch):
    grid = _grid()
    expected = _identity_maps(grid)
    calls = []

    class FakeMetricEvaluator:
        def evaluate_magnetic_field(self, points, bfield):
            points = np.asarray(points)
            return SimpleNamespace(
                B_contravariant=np.tile(np.asarray((0.0, 0.0, 1.0)), (points.shape[0], 1)),
                magnitude=np.ones(points.shape[0]),
            )

    def fake_builder(grid_arg, callback, **kwargs):
        calls.append(True)
        callback(np.asarray([[0.25, 0.5, 0.75]]))
        return {name: jnp.asarray(value) for name, value in expected.items()}

    monkeypatch.setattr(hsx, "build_fci_maps_from_callbacks", fake_builder)
    cache_path = tmp_path / "metric.npz"
    payload = hsx.add_fci_maps_to_metric_cache_payload(
        {"metric_sentinel": np.asarray([1.0])}, FciMaps3D(**expected)
    )
    payload["fci_maps_source_fingerprint"] = np.asarray(
        hsx._hsx_fci_map_source_fingerprint()
    )
    payload["fci_maps_trace_substeps"] = np.asarray(4, dtype=np.int64)
    payload["fci_maps_forward_length"][0, 0, 0] = 0.0
    np.savez(cache_path, **payload)

    maps, _, _ = hsx._build_or_load_hsx_fci_maps(
        grid=grid,
        topology="toroidal",
        construct_fci_maps=True,
        fci_trace_substeps=4,
        cache_payload=payload,
        cache_path=cache_path,
        metric_evaluator=FakeMetricEvaluator(),
        bfield=object(),
        makegrid_path=tmp_path / "mgrid.nc",
    )
    assert maps is not None
    assert len(calls) == 1

    missing_payload = dict(payload)
    missing_payload.pop("fci_maps_forward_y")
    maps_missing, _, _ = hsx._build_or_load_hsx_fci_maps(
        grid=grid,
        topology="toroidal",
        construct_fci_maps=True,
        fci_trace_substeps=4,
        cache_payload=missing_payload,
        cache_path=cache_path,
        metric_evaluator=FakeMetricEvaluator(),
        bfield=object(),
        makegrid_path=tmp_path / "mgrid.nc",
    )
    assert maps_missing is not None
    assert len(calls) == 2
