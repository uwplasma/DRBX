"""Focused analytic tests for axis-regular FCI map generation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from drbx.geometry.fci_geometry import (  # noqa: E402
    CellCenteredGrid3D,
    Grid1D,
    build_fci_maps_from_callbacks,
    trace_fci_eta_plane_from_callbacks,
)


def _circular_shifted_torus_grid(shape: tuple[int, int, int] = (8, 8, 4)) -> CellCenteredGrid3D:
    nx, ny, nz = shape
    return CellCenteredGrid3D(
        x=Grid1D(
            centers=(jnp.arange(nx, dtype=jnp.float64) + 0.5) / nx,
            faces=jnp.linspace(0.0, 1.0, nx + 1),
        ),
        y=Grid1D(
            centers=2.0 * jnp.pi * (jnp.arange(ny, dtype=jnp.float64) + 0.5) / ny,
            faces=jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1),
        ),
        z=Grid1D(
            centers=2.0 * jnp.pi * (jnp.arange(nz, dtype=jnp.float64) + 0.5) / nz,
            faces=jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1),
        ),
    )


def _periodic(value: float, period: float) -> float:
    return float(np.mod(value, period))


def _constant_callback(*, radial: float, poloidal: float):
    def evaluate(points: np.ndarray):
        b = np.zeros((points.shape[0], 3), dtype=np.float64)
        b[:, 0] = radial
        b[:, 1] = poloidal
        b[:, 2] = 1.0
        return b, np.ones(points.shape[0], dtype=np.float64)

    return evaluate


def _axis_regular_radial_callback(points: np.ndarray):
    b = np.zeros((points.shape[0], 3), dtype=np.float64)
    b[:, 0] = -0.2 * np.cos(points[:, 1])
    b[:, 2] = 1.0
    return b, np.ones(points.shape[0], dtype=np.float64)


def test_axis_crossing_reflects_radius_and_advances_theta_by_pi() -> None:
    grid = _circular_shifted_torus_grid()
    trace = trace_fci_eta_plane_from_callbacks(
        grid,
        _axis_regular_radial_callback,
        eta_index=0,
        direction=1,
        substeps=1,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )

    i, j = 0, 0
    dz = float(grid.z.centers[1] - grid.z.centers[0])
    x0 = float(grid.x.centers[i])
    theta0 = float(grid.y.centers[j])
    expected_x = 0.2 * np.cos(theta0) * dz - x0
    expected_theta = _periodic(theta0 + np.pi, 2.0 * np.pi)

    np.testing.assert_allclose(float(trace["endpoint_x"][i, j]), expected_x, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(float(trace["endpoint_y"][i, j]), expected_theta, rtol=0.0, atol=2.0e-7)
    assert not bool(trace["boundary"][i, j])
    np.testing.assert_allclose(float(trace["length"][i, j]), dz, rtol=0.0, atol=2.0e-7)


def test_theta_and_eta_periodic_seams_are_not_physical_boundaries() -> None:
    grid = _circular_shifted_torus_grid()
    trace = trace_fci_eta_plane_from_callbacks(
        grid,
        _constant_callback(radial=0.0, poloidal=0.2),
        eta_index=grid.z.n - 1,
        direction=1,
        substeps=8,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )

    i, j = grid.x.n // 2, grid.y.n - 1
    dz = float(grid.z.centers[0] + (grid.z.faces[-1] - grid.z.faces[0]) - grid.z.centers[-1])
    expected_theta = _periodic(float(grid.y.centers[j]) + 0.2 * dz, 2.0 * np.pi)
    expected_eta = float(grid.z.centers[0])

    np.testing.assert_allclose(float(trace["endpoint_y"][i, j]), expected_theta, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(float(trace["endpoint_z"][i, j]), expected_eta, rtol=0.0, atol=2.0e-7)
    assert not bool(trace["boundary"][i, j])


def test_outer_radial_hit_remains_a_physical_boundary_with_endpoint_data() -> None:
    grid = _circular_shifted_torus_grid()
    maps = build_fci_maps_from_callbacks(
        grid,
        _constant_callback(radial=0.5, poloidal=0.0),
        substeps=8,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )

    i, j, k = grid.x.n - 1, 0, 0
    dz = float(grid.z.centers[1] - grid.z.centers[0])
    expected_length = (float(grid.x.faces[-1]) - float(grid.x.centers[i])) / 0.5

    assert bool(maps["forward_boundary"][i, j, k])
    np.testing.assert_allclose(float(maps["forward_endpoint_x"][i, j, k]), 1.0, rtol=0.0, atol=2.0e-7)
    np.testing.assert_allclose(float(maps["forward_length"][i, j, k]), expected_length, rtol=0.0, atol=2.0e-6)
    np.testing.assert_allclose(
        float(maps["forward_endpoint_z"][i, j, k]),
        float(grid.z.centers[k] + expected_length),
        rtol=0.0,
        atol=2.0e-7,
    )


def test_outer_wall_hit_interpolates_along_unwrapped_periodic_chord() -> None:
    """A theta-seam crossing must retain the nearby wall-hit coordinate."""

    grid = _circular_shifted_torus_grid()
    radial = 0.05
    poloidal = 0.5
    trace = trace_fci_eta_plane_from_callbacks(
        grid,
        _constant_callback(radial=radial, poloidal=poloidal),
        eta_index=0,
        direction=1,
        substeps=1,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )

    i = grid.x.n - 1
    j = grid.y.n - 1
    eta_to_wall = (
        float(grid.x.faces[-1]) - float(grid.x.centers[i])
    ) / radial
    theta_unwrapped = float(grid.y.centers[j]) + poloidal * eta_to_wall
    expected_theta = _periodic(theta_unwrapped, 2.0 * np.pi)
    expected_eta = float(grid.z.centers[0]) + eta_to_wall

    assert theta_unwrapped > 2.0 * np.pi
    assert bool(trace["boundary"][i, j])
    np.testing.assert_allclose(
        float(trace["endpoint_x"][i, j]), 1.0, rtol=0.0, atol=2.0e-7
    )
    np.testing.assert_allclose(
        float(trace["endpoint_y"][i, j]), expected_theta, rtol=0.0, atol=2.0e-7
    )
    np.testing.assert_allclose(
        float(trace["endpoint_z"][i, j]), expected_eta, rtol=0.0, atol=2.0e-7
    )


def test_memory_bounded_map_batches_preserve_every_seed_index() -> None:
    """A group larger than one callback batch must scatter without gaps."""

    grid = _circular_shifted_torus_grid(shape=(16, 16, 9))
    maps = build_fci_maps_from_callbacks(
        grid,
        _constant_callback(radial=0.0, poloidal=0.0),
        substeps=1,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
    )
    expected_x = np.broadcast_to(
        np.arange(grid.x.n, dtype=np.float64)[:, None, None], grid.shape
    )
    expected_y = np.broadcast_to(
        np.arange(grid.y.n, dtype=np.float64)[None, :, None], grid.shape
    )
    np.testing.assert_allclose(np.asarray(maps["forward_x"]), expected_x)
    np.testing.assert_allclose(np.asarray(maps["backward_x"]), expected_x)
    np.testing.assert_allclose(np.asarray(maps["forward_y"]), expected_y)
    np.testing.assert_allclose(np.asarray(maps["backward_y"]), expected_y)
    assert not bool(np.any(np.asarray(maps["forward_boundary"])))
    assert not bool(np.any(np.asarray(maps["backward_boundary"])))


def test_direction_checkpoint_reuses_completed_callback_traces(tmp_path) -> None:
    grid = _circular_shifted_torus_grid()
    checkpoint = tmp_path / "directions.npz"
    calls = 0

    def callback(points):
        nonlocal calls
        calls += 1
        return _constant_callback(radial=0.0, poloidal=0.1)(points)

    first = build_fci_maps_from_callbacks(
        grid,
        callback,
        substeps=1,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
        direction_checkpoint_path=checkpoint,
    )
    assert checkpoint.is_file()
    assert calls > 0

    calls = 0
    second = build_fci_maps_from_callbacks(
        grid,
        callback,
        substeps=1,
        periodic_axes=(False, True, True),
        axis_regular_axes=(True, False, False),
        direction_checkpoint_path=checkpoint,
    )
    assert calls == 0
    assert set(first) == set(second)
    for name in first:
        np.testing.assert_array_equal(np.asarray(first[name]), np.asarray(second[name]))
