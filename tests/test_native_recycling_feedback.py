from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from jax_drb.native.recycling_feedback import (
    advance_feedback_integrals,
    advance_feedback_integrals_from_predictor,
    current_feedback_errors,
    feedback_error_vector,
    feedback_integral_vector,
    sanitize_feedback_integrals,
)


def test_current_feedback_errors_uses_upstream_cell() -> None:
    mesh = SimpleNamespace(xstart=1, ystart=2)
    controllers = {
        "d+": SimpleNamespace(density_upstream=10.0),
        "he+": SimpleNamespace(density_upstream=4.0),
    }
    fields = {
        "Nd+": np.arange(2 * 4 * 1, dtype=np.float64).reshape(2, 4, 1),
        "Nhe+": np.full((2, 4, 1), 1.5, dtype=np.float64),
    }

    errors = current_feedback_errors(fields, controllers=controllers, mesh=mesh)

    assert errors["d+"] == 10.0 - float(fields["Nd+"][1, 2, 0])
    assert errors["he+"] == 4.0 - 1.5


def test_current_feedback_errors_skips_controller_without_density_field() -> None:
    mesh = SimpleNamespace(xstart=0, ystart=0)
    controllers = {
        "d+": SimpleNamespace(density_upstream=10.0),
        "missing+": SimpleNamespace(density_upstream=3.0),
    }
    fields = {"Nd+": np.asarray([[[8.0]]], dtype=np.float64)}

    errors = current_feedback_errors(fields, controllers=controllers, mesh=mesh)

    assert errors == {"d+": 2.0}


def test_advance_feedback_integrals_applies_trapezoid_and_positive_constraint() -> None:
    mesh = SimpleNamespace(xstart=0, ystart=0)
    controllers = {
        "d+": SimpleNamespace(density_upstream=1.0, density_integral_positive=False),
        "he+": SimpleNamespace(density_upstream=0.5, density_integral_positive=True),
    }
    fields = {
        "Nd+": np.asarray([[[0.6]]], dtype=np.float64),
        "Nhe+": np.asarray([[[0.7]]], dtype=np.float64),
    }

    updated = advance_feedback_integrals(
        fields,
        controllers=controllers,
        feedback_integrals={"d+": 0.2, "he+": -0.1},
        feedback_previous_errors={"d+": 0.8, "he+": -0.4},
        mesh=mesh,
        timestep=2.0,
    )

    assert updated["d+"] == 0.2 + 2.0 * 0.5 * ((1.0 - 0.6) + 0.8)
    assert updated["he+"] == 0.0


def test_advance_feedback_integrals_defaults_previous_error_and_missing_integral() -> None:
    mesh = SimpleNamespace(xstart=0, ystart=0)
    controllers = {
        "d+": SimpleNamespace(density_upstream=2.0, density_integral_positive=False),
        "missing+": SimpleNamespace(density_upstream=1.0, density_integral_positive=True),
    }
    fields = {"Nd+": np.asarray([[[1.25]]], dtype=np.float64)}

    updated = advance_feedback_integrals(
        fields,
        controllers=controllers,
        feedback_integrals={},
        feedback_previous_errors={},
        mesh=mesh,
        timestep=2.0,
    )

    assert updated["d+"] == 2.0 * (2.0 - 1.25)
    assert updated["missing+"] == 0.0


def test_predictor_feedback_integrals_and_vector_helpers() -> None:
    controllers = {
        "d+": SimpleNamespace(density_integral_positive=False),
        "he+": SimpleNamespace(density_integral_positive=True),
    }
    updated = advance_feedback_integrals_from_predictor(
        controllers=controllers,
        feedback_integrals={"d+": 0.0, "he+": -0.2},
        feedback_previous_errors={"d+": 1.0, "he+": -0.1},
        predictor_feedback_errors={"d+": 0.5, "he+": -0.4},
        timestep=4.0,
    )
    sanitized = sanitize_feedback_integrals(updated, controllers=controllers)

    assert updated["d+"] == 4.0 * 0.5 * (1.0 + 0.5)
    assert updated["he+"] == 0.0
    assert sanitized["he+"] == 0.0
    np.testing.assert_allclose(
        feedback_integral_vector(sanitized, feedback_names=("he+", "d+")),
        np.asarray([0.0, updated["d+"]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        feedback_error_vector({"he+": -0.2, "d+": 0.3}, feedback_names=("he+", "d+")),
        np.asarray([-0.2, 0.3], dtype=np.float64),
    )


def test_predictor_and_sanitize_feedback_integrals_use_defaults_and_clamps() -> None:
    controllers = {
        "d+": SimpleNamespace(density_integral_positive=False),
        "he+": SimpleNamespace(density_integral_positive=True),
    }

    updated = advance_feedback_integrals_from_predictor(
        controllers=controllers,
        feedback_integrals={},
        feedback_previous_errors={"d+": 0.25, "he+": -0.5},
        predictor_feedback_errors={},
        timestep=2.0,
    )
    sanitized = sanitize_feedback_integrals({"d+": 1.5, "he+": -0.25}, controllers=controllers)

    assert updated["d+"] == 0.5
    assert updated["he+"] == 0.0
    assert sanitized == {"d+": 1.5, "he+": 0.0}
    np.testing.assert_allclose(
        feedback_integral_vector(sanitized, feedback_names=("he+", "missing")),
        np.asarray([0.0, 0.0], dtype=np.float64),
    )
