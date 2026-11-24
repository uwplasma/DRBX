"""Autodiff evaluation for nodal curvature and the perpendicular tensor."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...evaluators.curvature_autodiff import DEFAULT_MODE

from .eta_filter import EtaFilterArm

_OWN = frozenset({"_wrapped", "_autodiff_k", "_autodiff_perp", "_autodiff_mode", "_metric"})


def _jax_field(ref: Any, jax_bfield_class: Any):
    """``(jax_bfield, extra kwargs)`` of the autodiff evaluators: the JAX Cartesian evaluator of ``ref.bfield_evaluator``
    (no extra kwargs), or, for an eta-filtered reference, ``(None, {"logical_field": <JAX table twin>})``."""
    arm = getattr(ref, "eta_filter_arm", None)
    if isinstance(arm, EtaFilterArm):
        return None, {"logical_field": arm.jax_field()}
    return jax_bfield_class.from_evaluator(ref.bfield_evaluator), {}


class AutodiffCurvatureReference:
    """Delegating wrapper over a scripts continuum reference; ``_curvature`` is autodiff."""

    def __init__(self, reference: Any, *, mode: str = DEFAULT_MODE, block: int = 256) -> None:
        if isinstance(reference, AutodiffCurvatureReference):
            reference = reference.wrapped
        object.__setattr__(self, "_wrapped", reference)
        object.__setattr__(self, "_autodiff_k", None)
        object.__setattr__(self, "_autodiff_perp", None)
        object.__setattr__(self, "_autodiff_mode", (str(mode), int(block)))

    # -- delegation -----------------------------------------------------
    @property
    def wrapped(self) -> Any:
        """The wrapped (finite-difference) scripts reference."""
        return self._wrapped

    def __getattr__(self, name: str) -> Any:            # only reached when normal lookup fails
        if name in ("_wrapped", "_autodiff_k", "_autodiff_perp", "_autodiff_mode"):
            raise AttributeError(name)
        return getattr(self._wrapped, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped, name, value)

    def __delattr__(self, name: str) -> None:
        if name in _OWN:
            object.__delattr__(self, name)
        else:
            delattr(self._wrapped, name)

    # -- autodiff curvature ----------------------------------------------
    def autodiff(self):
        """The :class:`drbx.geometry.curvature_autodiff.AutodiffCurvature` of this reference (cached)."""
        if self._autodiff_k is None:
            from ...evaluators.curvature_autodiff import autodiff_curvature
            from ...evaluators.jax_magnetic_field_evaluator import JaxComponentSplineBFieldEvaluator
            from ...evaluators.jax_metric_evaluator import JaxMetricEvaluator

            ref = self._wrapped
            jax_metric = JaxMetricEvaluator.from_metric_evaluator(ref.metric_evaluator)
            jax_bfield, extra = _jax_field(ref, JaxComponentSplineBFieldEvaluator)
            mode, block = self._autodiff_mode
            object.__setattr__(self, "_autodiff_k", autodiff_curvature(jax_metric, jax_bfield, float(ref.B0),
                                                                       mode=mode, block=block, **extra))
        return self._autodiff_k

    def _curvature(self, q: np.ndarray) -> np.ndarray:
        return self.autodiff()(np.asarray(q, dtype=np.float64))

    def autodiff_perpendicular(self):
        """The :class:`drbx.geometry.curvature_autodiff.AutodiffPerpendicularGeometry` of this reference (cached)."""
        if self._autodiff_perp is None:
            from ...evaluators.curvature_autodiff import autodiff_perpendicular_geometry
            from ...evaluators.jax_magnetic_field_evaluator import JaxComponentSplineBFieldEvaluator
            from ...evaluators.jax_metric_evaluator import JaxMetricEvaluator

            ref = self._wrapped
            jax_metric = JaxMetricEvaluator.from_metric_evaluator(ref.metric_evaluator)
            jax_bfield, extra = _jax_field(ref, JaxComponentSplineBFieldEvaluator)
            mode, block = self._autodiff_mode
            object.__setattr__(self, "_autodiff_perp", autodiff_perpendicular_geometry(
                jax_metric, jax_bfield, float(ref.B0), mode=mode, block=block, **extra))
        return self._autodiff_perp


    def __repr__(self) -> str:
        return f"AutodiffCurvatureReference({self._wrapped!r})"
