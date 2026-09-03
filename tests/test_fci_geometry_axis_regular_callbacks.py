"""Focused tests for callback-backed, axis-regular FCI map generation."""

import numpy as np
import jax.numpy as jnp

from drbx.geometry.fci_geometry import (
    CellCenteredGrid3D,
    Grid1D,
    build_fci_maps_from_b_contravariant,
    build_fci_maps_from_callbacks,
    trace_fci_eta_plane_from_callbacks,
)


def _grid() -> CellCenteredGrid3D:
    return CellCenteredGrid3D(
        x=Grid1D(
            centers=jnp.asarray([0.125, 0.375, 0.625, 0.875]),
            faces=jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
        ),
        y=Grid1D(
            centers=jnp.asarray([np.pi / 4.0, 3.0 * np.pi / 4.0, 5.0 * np.pi / 4.0, 7.0 * np.pi / 4.0]),
            faces=jnp.linspace(0.0, 2.0 * np.pi, 5),
        ),
        z=Grid1D(
            centers=jnp.asarray([0.125, 0.375, 0.625, 0.875]),
            faces=jnp.linspace(0.0, 1.0, 5),
        ),
    )


def _constant_field(bx: float, by: float, *, bmag: float = 1.0):
    calls = []

    def evaluate(points):
        points = np.asarray(points, dtype=float)
        calls.append(points.copy())
        return (
            np.broadcast_to(np.asarray([bx, by, 1.0]), (points.shape[0], 3)).copy(),
            np.full(points.shape[0], bmag),
        )

    evaluate.calls = calls
    return evaluate


def test_callback_axis_crossing_reflects_radius_and_adds_pi():
    grid = _grid()
    calls = []

    def evaluator(points):
        points = np.asarray(points, dtype=float)
        calls.append(points.copy())
        # Smooth axis parity: B^u(theta + pi) = -B^u(theta).  Normalize so
        # the first theta center has B^u=-0.75.
        radial = -0.75 * np.cos(points[:, 1]) / np.cos(np.pi / 4.0)
        return (
            np.column_stack((radial, np.zeros_like(radial), np.ones_like(radial))),
            np.ones(points.shape[0]),
        )

    maps = build_fci_maps_from_callbacks(
        grid,
        evaluator,
        substeps=1,
        axis_regular_axes=(True, False, False),
    )

    # At r=0.125, one eta-plane step gives r=-0.0625 and therefore reflects
    # to r=0.0625 at theta+pi. The lower radial face is not a wall.
    assert np.isclose(float(maps["forward_endpoint_x"][0, 0, 0]), 0.0625)
    assert np.isclose(float(maps["forward_endpoint_y"][0, 0, 0]), 5.0 * np.pi / 4.0)
    assert not bool(maps["forward_boundary"][0, 0, 0])
    assert np.isclose(float(maps["forward_length"][0, 0, 0]), 0.25)
    assert any(
        not np.allclose(points[:, 0], grid.x.centers[0])
        for points in calls
        if points.size
    )


def test_callback_outer_wall_retains_hit_and_connection_length():
    grid = _grid()
    evaluator = _constant_field(4.0, 0.0)

    maps = build_fci_maps_from_callbacks(grid, evaluator, substeps=1)

    # The outermost radial center is 0.875. With dz=0.25 and dr/deta=4,
    # the wall is reached after eta distance 0.125/4.
    assert bool(maps["forward_boundary"][3, 0, 0])
    assert np.isclose(float(maps["forward_endpoint_x"][3, 0, 0]), 1.0)
    assert np.isclose(float(maps["forward_x"][3, 0, 0]), grid.x.n - 0.5)
    assert np.isclose(float(maps["forward_length"][3, 0, 0]), 0.125 / 4.0)
    assert np.isclose(float(maps["forward_endpoint_b_contra_x"][3, 0, 0]), 4.0)
    assert np.isclose(float(maps["forward_endpoint_b_contra_y"][3, 0, 0]), 0.0)
    assert np.isclose(float(maps["forward_endpoint_b_contra_z"][3, 0, 0]), 1.0)
    assert np.isclose(float(maps["forward_endpoint_bmag"][3, 0, 0]), 1.0)

    lower_maps = build_fci_maps_from_callbacks(
        grid,
        _constant_field(-4.0, 0.0),
        substeps=1,
    )
    assert bool(lower_maps["forward_boundary"][0, 0, 0])
    assert np.isclose(float(lower_maps["forward_endpoint_x"][0, 0, 0]), 0.0)
    assert np.isclose(float(lower_maps["forward_x"][0, 0, 0]), -0.5)


def test_callback_theta_periodic_seam_and_one_plane_api():
    grid = _grid()
    evaluator = _constant_field(0.0, 4.0)

    maps = build_fci_maps_from_callbacks(grid, evaluator, substeps=1)
    endpoint = float(maps["forward_endpoint_y"][0, 3, 0])
    expected = (7.0 * np.pi / 4.0 + 4.0 * 0.25) % (2.0 * np.pi)
    assert np.isclose(endpoint, expected)
    assert not bool(maps["forward_boundary"][0, 3, 0])

    plane = trace_fci_eta_plane_from_callbacks(
        grid,
        evaluator,
        eta_index=0,
        direction=-1,
        substeps=1,
    )
    assert plane["x_index"].shape == (grid.x.n, grid.y.n)
    assert np.all(np.isfinite(np.asarray(plane["y_index"])))
    assert np.all(~np.asarray(plane["boundary"]))


def test_callback_traces_batch_all_uniform_eta_seeds():
    grid = _grid()
    substeps = 3
    evaluator = _constant_field(0.0, 0.0)

    build_fci_maps_from_callbacks(grid, evaluator, substeps=substeps)

    # Each substep makes seven field evaluations per direction: speed at the
    # old point, four RK4 stages, speed at the trial endpoint, and speed at
    # the retained wall endpoint.  One final evaluation per direction stores
    # B at the traced endpoint itself. Uniform periodic eta has one forward
    # and one backward seed group, independent of nz.
    assert len(evaluator.calls) == 2 * (7 * substeps + 1)
    assert all(points.shape == (grid.x.n * grid.y.n * grid.z.n, 3) for points in evaluator.calls)


def test_callback_stores_field_evaluated_at_traced_endpoints():
    grid = _grid()

    def evaluator(points):
        points = np.asarray(points, dtype=float)
        x, y, z = points.T
        return (
            np.column_stack((0.15 + 0.2 * x, -0.1 + 0.03 * y, 1.0 + 0.1 * z)),
            2.0 + 0.4 * x - 0.02 * y + 0.05 * z,
        )

    maps = build_fci_maps_from_callbacks(grid, evaluator, substeps=2)

    for direction in ("forward", "backward"):
        points = np.stack(
            [
                np.asarray(maps[f"{direction}_endpoint_x"]),
                np.asarray(maps[f"{direction}_endpoint_y"]),
                np.asarray(maps[f"{direction}_endpoint_z"]),
            ],
            axis=-1,
        )
        expected_b, expected_bmag = evaluator(points.reshape(-1, 3))
        expected_b = expected_b.reshape(grid.shape + (3,))
        expected_bmag = expected_bmag.reshape(grid.shape)
        for axis, name in enumerate(("x", "y", "z")):
            np.testing.assert_allclose(
                np.asarray(maps[f"{direction}_endpoint_b_contra_{name}"]),
                expected_b[..., axis],
                rtol=0.0,
                atol=2.0e-7,
            )
        np.testing.assert_allclose(
            np.asarray(maps[f"{direction}_endpoint_bmag"]),
            expected_bmag,
            rtol=0.0,
            atol=2.0e-7,
        )


def test_cell_array_builder_preserves_axis_regular_option():
    grid = _grid()
    bcontra = jnp.zeros(grid.shape + (3,)).at[..., 0].set(-0.75).at[..., 2].set(1.0)
    bmag = jnp.ones(grid.shape)

    maps = build_fci_maps_from_b_contravariant(
        grid,
        bcontra,
        bmag,
        substeps=1,
        axis_regular_axes=(True, False, False),
    )

    assert np.isclose(float(maps["forward_endpoint_x"][0, 0, 0]), 0.0625, atol=1.0e-12)
    assert np.isclose(float(maps["forward_endpoint_y"][0, 0, 0]), 5.0 * np.pi / 4.0, atol=1.0e-12)
    assert not bool(maps["forward_boundary"][0, 0, 0])
