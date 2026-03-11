from __future__ import annotations

import math

import numpy as np

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.expression import ArrayExpressionEvaluator
from jax_drb.native.mesh import broadcast_to_field_shape, build_structured_mesh
from jax_drb.runtime.run_config import RunConfiguration


def test_array_expression_evaluator_resolves_mesh_references_on_structured_grid() -> None:
    config = parse_bout_input(
        """
        nout = 5
        timestep = 20

        [mesh]
        nx = 10
        ny = 10
        nz = 10
        yn = y / (2π)
        zn = z / (2π)

        [model]
        components = e

        [e]
        type = evolve_density

        [Ne]
        function = exp(-(x - 0.5)^2 - (mesh:yn - 0.5)^2 - (mesh:zn - 0.5)^2)
        """
    )
    mesh = build_structured_mesh(config, RunConfiguration.from_config(config))
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())

    field = np.asarray(evaluator.evaluate(config.raw("Ne", "function"), current_section="Ne"))

    expected = math.exp(-((1.0 / 12.0 - 0.5) ** 2 + (0.05 - 0.5) ** 2 + (0.5 - 0.5) ** 2))
    assert math.isclose(field[2, 2, 5], expected, rel_tol=1e-12, abs_tol=1e-12)


def test_array_expression_evaluator_supports_strict_heaviside_step() -> None:
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

        [Nh]
        function = H(x - 0.25) * H(0.75 - x)
        """
    )
    mesh = build_structured_mesh(config, RunConfiguration.from_config(config))
    evaluator = ArrayExpressionEvaluator(config, local_values=mesh.expression_context())

    field = np.asarray(
        broadcast_to_field_shape(
            evaluator.evaluate(config.raw("Nh", "function"), current_section="Nh"),
            mesh,
        )
    )

    assert field[3, 2, 0] == 0.0
    assert field[4, 2, 0] == 1.0
    assert field[6, 2, 0] == 0.0
