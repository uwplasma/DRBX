from __future__ import annotations

import math

import numpy as np

from jax_drb.config.boutinp import parse_bout_input
from jax_drb.native.expression import ArrayExpressionEvaluator
from jax_drb.native.mesh import build_structured_mesh
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
