from __future__ import annotations

import numpy as np

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.mesh import (
    apply_field_boundaries,
    apply_neumann_x_guards,
    apply_zero_dirichlet_x_guards,
    build_structured_mesh,
    communicate_y_guards,
    project_nonnegative_x_boundaries,
)
from jax_drb.runtime.run_config import RunConfiguration


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
