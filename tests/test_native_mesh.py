from __future__ import annotations

import numpy as np
import pytest

from drbx.config.boutinp import parse_bout_input
from drbx.native.mesh import (
    apply_field_boundaries,
    apply_neumann_x_guards,
    apply_x_boundary,
    apply_zero_dirichlet_x_guards,
    broadcast_to_field_shape,
    build_structured_mesh,
    communicate_y_guards,
    project_nonnegative_x_boundaries,
)
from drbx.runtime.run_config import RunConfiguration


def test_structured_mesh_matches_expected_coordinate_convention() -> None:
    config = parse_bout_input(
        """
        nout = 5
        timestep = 20

        [mesh]
        nx = 10
        ny = 10
        nz = 10

        [model]
        components = e

        [e]
        type = evolve_density
        """
    )
    mesh = build_structured_mesh(config, RunConfiguration.from_config(config))

    np.testing.assert_allclose(
        np.asarray(mesh.x),
        np.array([-0.25, -1.0 / 12.0, 1.0 / 12.0, 0.25, 5.0 / 12.0, 7.0 / 12.0, 0.75, 11.0 / 12.0, 13.0 / 12.0, 1.25]),
    )
    np.testing.assert_allclose(
        np.asarray(mesh.y),
        np.array([-0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15]),
    )
    assert mesh.has_lower_y_target is False
    assert mesh.has_upper_y_target is False


def test_structured_mesh_marks_explicit_y_boundary_inputs_as_open_field() -> None:
    config = parse_bout_input(
        """
        nout = 1
        timestep = 1

        [mesh]
        nx = 1
        ny = 6
        nz = 1
        ixseps1 = -1
        ixseps2 = -1

        [model]
        components = h

        [h]
        type = evolve_density
        """
    )

    mesh = build_structured_mesh(config, RunConfiguration.from_config(config))

    assert mesh.has_lower_y_target is False
    assert mesh.has_upper_y_target is True


def test_structured_mesh_rejects_missing_explicit_dimensions() -> None:
    config = parse_bout_input(
        """
        nout = 1
        timestep = 1

        [mesh]
        ny = 4
        nz = 1

        [model]
        components = h

        [h]
        type = evolve_density
        """
    )

    with pytest.raises(ValueError, match="requires explicit mesh nx, ny, and nz"):
        build_structured_mesh(config, RunConfiguration.from_config(config))


def test_structured_mesh_honors_asymmetric_coordinate_options() -> None:
    config = parse_bout_input(
        """
        nout = 1
        timestep = 1

        [mesh]
        nx = 6
        ny = 4
        nz = 2
        symmetricGlobalX = false
        symmetricGlobalY = false
        jyseps1_1 = -1
        jyseps2_1 = 1
        jyseps1_2 = 3
        jyseps2_2 = 5
        ny_inner = 2

        [model]
        components = h

        [h]
        type = evolve_density
        """
    )

    mesh = build_structured_mesh(config, RunConfiguration.from_config(config))

    np.testing.assert_allclose(np.asarray(mesh.x), np.arange(6, dtype=np.float64) / 2.0)
    np.testing.assert_allclose(np.asarray(mesh.y), np.array([-0.5, -0.25, 0.0, 0.25, 0.0, 0.25, 0.5, 0.75]))
    np.testing.assert_allclose(np.asarray(mesh.z), np.array([0.0, 0.5]))


def test_broadcast_to_field_shape_returns_exact_shape_without_copying_semantics() -> None:
    config = parse_bout_input(
        """
        nout = 1
        timestep = 1

        [mesh]
        nx = 4
        ny = 2
        nz = 1

        [model]
        components = h

        [h]
        type = evolve_density
        """
    )
    mesh = build_structured_mesh(config, RunConfiguration.from_config(config))
    exact = np.ones((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)
    scalar = np.array(2.0, dtype=np.float64)

    np.testing.assert_allclose(np.asarray(broadcast_to_field_shape(exact, mesh)), exact)
    np.testing.assert_allclose(np.asarray(broadcast_to_field_shape(scalar, mesh)), 2.0 * exact)


def test_expression_context_returns_broadcastable_normalized_coordinates() -> None:
    config = parse_bout_input(
        """
        nout = 1
        timestep = 1

        [mesh]
        nx = 4
        ny = 2
        nz = 2

        [model]
        components = h

        [h]
        type = evolve_density
        """
    )
    mesh = build_structured_mesh(config, RunConfiguration.from_config(config))

    context = mesh.expression_context(time=3.5)

    assert context["x"].shape == (4, 1, 1)
    assert context["y"].shape == (1, 6, 1)
    assert context["z"].shape == (1, 1, 2)
    assert float(context["t"]) == 3.5


def test_boundary_pipeline_reproduces_expected_guard_pattern() -> None:
    config = parse_bout_input(
        """
        nout = 5
        timestep = 20

        [mesh]
        nx = 10
        ny = 10
        nz = 10

        [model]
        components = e

        [e]
        type = evolve_density
        """
    )
    mesh = build_structured_mesh(config, RunConfiguration.from_config(config))
    field = np.ones((mesh.nx, mesh.local_ny, mesh.nz))

    field = np.asarray(apply_zero_dirichlet_x_guards(field, mesh))
    field = np.asarray(communicate_y_guards(field, mesh))
    field = np.asarray(project_nonnegative_x_boundaries(field, mesh))

    assert np.all(field[1, mesh.ystart : mesh.yend + 1, :] == 0.0)
    assert np.all(field[0, :, :] == 0.0)
    assert np.all(field[1, 0:mesh.ystart, :] == -1.0)
    assert np.all(field[1, mesh.yend + 1 :, :] == -1.0)


def test_neumann_x_guards_copy_interior_strips() -> None:
    config = parse_bout_input(
        """
        nout = 1
        timestep = 1

        [mesh]
        nx = 10
        ny = 10
        nz = 1

        [model]
        components = h

        [h]
        type = evolve_density
        """
    )
    mesh = build_structured_mesh(config, RunConfiguration.from_config(config))
    field = np.zeros((mesh.nx, mesh.local_ny, mesh.nz))
    field[2:8, mesh.ystart : mesh.yend + 1, 0] = np.arange(6)[:, None]

    bounded = np.asarray(apply_neumann_x_guards(field, mesh))

    assert np.all(bounded[1, mesh.ystart : mesh.yend + 1, 0] == bounded[2, mesh.ystart : mesh.yend + 1, 0])
    assert np.all(bounded[0, mesh.ystart : mesh.yend + 1, 0] == bounded[3, mesh.ystart : mesh.yend + 1, 0])
    assert np.all(bounded[8, mesh.ystart : mesh.yend + 1, 0] == bounded[7, mesh.ystart : mesh.yend + 1, 0])
    assert np.all(bounded[9, mesh.ystart : mesh.yend + 1, 0] == bounded[6, mesh.ystart : mesh.yend + 1, 0])


def test_apply_field_boundaries_combines_x_and_y_guard_logic() -> None:
    config = parse_bout_input(
        """
        nout = 1
        timestep = 1

        [mesh]
        nx = 10
        ny = 10
        nz = 1

        [model]
        components = h

        [h]
        type = evolve_density
        """
    )
    mesh = build_structured_mesh(config, RunConfiguration.from_config(config))
    field = np.zeros((mesh.nx, mesh.local_ny, mesh.nz))
    field[2:8, mesh.ystart : mesh.yend + 1, 0] = np.arange(6)[:, None]

    bounded = np.asarray(apply_field_boundaries(field, mesh, x_boundary="neumann"))

    assert np.all(bounded[:, 1, :] == bounded[:, 2, :])
    assert np.all(bounded[:, 0, :] == bounded[:, 3, :])
    assert np.all(bounded[:, 12, :] == bounded[:, 11, :])
    assert np.all(bounded[:, 13, :] == bounded[:, 10, :])


def test_x_boundary_helpers_are_noops_without_x_guards_and_reject_unknown_boundaries() -> None:
    config = parse_bout_input(
        """
        nout = 1
        timestep = 1
        MXG = 0

        [mesh]
        nx = 4
        ny = 4
        nz = 1

        [model]
        components = h

        [h]
        type = evolve_density
        """
    )
    mesh = build_structured_mesh(config, RunConfiguration.from_config(config))
    field = np.arange(mesh.nx * mesh.local_ny * mesh.nz, dtype=np.float64).reshape(mesh.nx, mesh.local_ny, mesh.nz)

    np.testing.assert_allclose(np.asarray(apply_zero_dirichlet_x_guards(field, mesh)), field)
    np.testing.assert_allclose(np.asarray(apply_neumann_x_guards(field, mesh)), field)
    np.testing.assert_allclose(np.asarray(project_nonnegative_x_boundaries(-field, mesh)), -field)

    with pytest.raises(NotImplementedError, match="Unsupported X boundary kind"):
        apply_x_boundary(field, mesh, "periodic")


def test_apply_field_boundaries_projects_negative_dirichlet_x_guards() -> None:
    config = parse_bout_input(
        """
        nout = 1
        timestep = 1

        [mesh]
        nx = 6
        ny = 2
        nz = 1

        [model]
        components = h

        [h]
        type = evolve_density
        """
    )
    mesh = build_structured_mesh(config, RunConfiguration.from_config(config))
    field = -np.ones((mesh.nx, mesh.local_ny, mesh.nz), dtype=np.float64)

    bounded = np.asarray(apply_field_boundaries(field, mesh, x_boundary="zero"))

    assert np.all(bounded[:mesh.xstart, mesh.ystart : mesh.yend + 1, :] >= 0.0)
    assert np.all(bounded[mesh.xend + 1 :, mesh.ystart : mesh.yend + 1, :] >= 0.0)
