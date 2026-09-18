"""Compiled continuous HSX magnetic field used by producer-side tracing.

The simulation runtime never constructs this object.  Geometry production
converts the qualified NumPy/SciPy metric and MAKEGRID evaluators once, then
uses this immutable PyTree inside the JAX RK4 tracer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .jax_bfield_evaluator import JaxComponentSplineBFieldEvaluator
from .jax_metric_evaluator import JaxMetricEvaluator


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxHsxMagneticField:
    """Point-local logical magnetic-field transform for HSX tracing."""

    metric: JaxMetricEvaluator
    bfield: JaxComponentSplineBFieldEvaluator

    @classmethod
    def from_evaluators(cls, metric_evaluator: Any, bfield_evaluator: Any):
        return cls(
            metric=JaxMetricEvaluator.from_metric_evaluator(metric_evaluator),
            bfield=JaxComponentSplineBFieldEvaluator.from_evaluator(
                bfield_evaluator
            ),
        )

    def tree_flatten(self):
        return (self.metric, self.bfield), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)

    def __call__(self, logical_points: Any):
        """Return ``(B_contravariant, |B|)`` for ``(..., 3)`` points."""

        points = jnp.asarray(logical_points, dtype=jnp.float64)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("logical_points must have shape (..., 3)")
        position, jacobian = self.metric.position_and_jacobian(points)
        field_cartesian = self.bfield.evaluate_cartesian(position)
        b_contravariant = jnp.linalg.solve(
            jacobian, field_cartesian[..., None]
        )[..., 0]
        magnitude = jnp.linalg.norm(field_cartesian, axis=-1)
        return b_contravariant, magnitude


def jax_hsx_magnetic_field_from_evaluators(
    metric_evaluator: Any, bfield_evaluator: Any
) -> JaxHsxMagneticField:
    """Construct the compiled tracing callback from producer evaluators."""

    return JaxHsxMagneticField.from_evaluators(metric_evaluator, bfield_evaluator)


__all__ = [
    "JaxHsxMagneticField",
    "jax_hsx_magnetic_field_from_evaluators",
]
