from __future__ import annotations

import math

import numpy as np
import pytest

from drbx.config.boutinp import ROOT_SECTION, parse_bout_input
from drbx.native.expression import ArrayExpressionEvaluator
from drbx.native.mesh import broadcast_to_field_shape, build_structured_mesh
from drbx.runtime.run_config import RunConfiguration


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


def _evaluator(text: str, **external) -> ArrayExpressionEvaluator:
    return ArrayExpressionEvaluator(parse_bout_input(text), local_values={}, external_values=external or None)


def test_expression_evaluator_reproduces_closed_form_values_and_resolves_references() -> None:
    evaluator = _evaluator("a = 3\nb = a^2 + 1\n[s]\nc = b / 2 + s:d\nd = -min(a, 5, 2)\n")

    assert float(evaluator.resolve_option("s", "c")) == 10.0 / 2 + -2.0
    assert float(evaluator.evaluate("max(a, 7, 4) + sqrt(a + 1) + abs(-1)")) == 7.0 + 2.0 + 1.0
    assert float(evaluator.evaluate("sin(pi/2)^2 + cos(0)")) == 2.0
    assert float(evaluator.evaluate("a % 2")) == 1.0
    with pytest.raises(KeyError, match="Missing option"):
        evaluator.resolve_option("s", "nope")
    with pytest.raises(KeyError, match="Unknown array expression symbol"):
        evaluator.evaluate("nope + 1")


def test_expression_evaluator_rejects_cycles_lists_and_unsafe_constructs() -> None:
    evaluator = _evaluator("a = b + 1\nb = a + 1\nl = 1, 2\n")

    with pytest.raises(ValueError, match="Cyclic"):
        evaluator.resolve_option(ROOT_SECTION, "a")
    with pytest.raises(TypeError, match="list"):
        evaluator.resolve_option(ROOT_SECTION, "l")
    for unsafe in ('__import__("os")', "open(1)", "a.real", "'text'", "a < 1"):
        with pytest.raises((TypeError, SyntaxError, ValueError)):
            evaluator.evaluate(unsafe)
