"""``AutodiffCurvatureReference``: the frozen continuum reference with autodiff curvature.

The wrapped object is the scripts reference (``hsx_mms_continuum_reference``'s
``ContinuumReference``, as built by ``p07_diffusion_global.numerics.reference``).
The wrapper delegates every attribute to it -- reads, and writes of anything but
its own private state -- except ``_curvature``, which evaluates the autodiff
``K = (B/2|J|) curl(b_cov/B)`` of :mod:`drbx.geometry.curvature_autodiff` through
JAX evaluators built from ``reference.metric_evaluator`` /
``reference.bfield_evaluator`` / ``reference.B0``.  Frozen code that calls
``ref._curvature(points)`` (``perpendicular_structured.reference_geometry.
curvature_geometry``, ``p06_structured_global.numerics._face_geometry``'s interior
branch, P06N ``_continuum_terms`` inputs, ``perpendicular_reference_rhs``) therefore
sees autodiff ``K`` without any frozen file changing.

Notes on delegation:

* Writes (``ref.finite_difference_step = h``, as the campaigns' control sweeps do)
  reach the wrapped reference, so its finite-difference derivatives (``_derivative``,
  ``_perpendicular_geometry``, ...) keep working exactly as before.  ``_metric`` is
  the one exception: ``metric_reuse`` installs a per-call cache on the object it is
  handed, and here that is the wrapper's own instance attribute (it shadows the
  delegated method for the wrapper only; the wrapped object's own internal calls stay
  uncached, which changes speed but not values).
* Methods of the wrapped object that call ``self._curvature`` internally (for example
  ``prepare``) run with ``self`` = the wrapped object and therefore still use its
  finite-difference ``K``.  The P-path never calls them; use ``ref._curvature`` or
  ``curvature_geometry`` for autodiff.
* The wall rule of ``p06_structured_global.numerics._face_geometry`` is not used with
  autodiff K: :class:`p_shared.provider.ScriptsGeometryProvider` computes ``K`` at every
  node, wall nodes included, from this wrapper.

The JAX evaluators are constructed lazily on the first ``_curvature`` call.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from drbx.geometry.curvature_autodiff import DEFAULT_MODE
# adopted 30 September 2026 after QK1-QK4 (work/p08_bundle_autodiff_curvature_20260930); "fd" reproduces the
# frozen step 1-3 oracles and campaigns and must be passed explicitly for that
DEFAULT_CURVATURE = "autodiff"

_OWN = frozenset({"_wrapped", "_autodiff_k", "_autodiff_perp", "_autodiff_mode", "_metric"})


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
            from drbx.geometry.curvature_autodiff import autodiff_curvature
            from drbx.geometry.jax_bfield_evaluator import JaxComponentSplineBFieldEvaluator
            from drbx.geometry.jax_metric_evaluator import JaxMetricEvaluator

            ref = self._wrapped
            jax_metric = JaxMetricEvaluator.from_metric_evaluator(ref.metric_evaluator)
            jax_bfield = JaxComponentSplineBFieldEvaluator.from_evaluator(ref.bfield_evaluator)
            mode, block = self._autodiff_mode
            object.__setattr__(self, "_autodiff_k", autodiff_curvature(jax_metric, jax_bfield, float(ref.B0),
                                                                       mode=mode, block=block))
        return self._autodiff_k

    def _curvature(self, q: np.ndarray) -> np.ndarray:
        return self.autodiff()(np.asarray(q, dtype=np.float64))

    def autodiff_perpendicular(self):
        """The :class:`drbx.geometry.curvature_autodiff.AutodiffPerpendicularGeometry` of this reference (cached)."""
        if self._autodiff_perp is None:
            from drbx.geometry.curvature_autodiff import autodiff_perpendicular_geometry
            from drbx.geometry.jax_bfield_evaluator import JaxComponentSplineBFieldEvaluator
            from drbx.geometry.jax_metric_evaluator import JaxMetricEvaluator

            ref = self._wrapped
            jax_metric = JaxMetricEvaluator.from_metric_evaluator(ref.metric_evaluator)
            jax_bfield = JaxComponentSplineBFieldEvaluator.from_evaluator(ref.bfield_evaluator)
            mode, block = self._autodiff_mode
            object.__setattr__(self, "_autodiff_perp", autodiff_perpendicular_geometry(
                jax_metric, jax_bfield, float(ref.B0), mode=mode, block=block))
        return self._autodiff_perp

    def perpendicular_geometry_autodiff(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(tensor (Q, 3, 3), divergence (Q, 3))`` of ``J (g^ij - b^i b^j)`` by autodiff.

        Deliberately separate from ``_perpendicular_geometry`` (which stays the delegated finite-difference one, so
        artifact builds keep their FD ``p07_raw_divergence``).
        """
        return self.autodiff_perpendicular()(np.asarray(points, dtype=np.float64))

    def __repr__(self) -> str:
        return f"AutodiffCurvatureReference({self._wrapped!r})"


def wrap_reference(reference: Any, curvature: str = DEFAULT_CURVATURE) -> Any:
    """``reference`` itself for ``curvature="fd"``, the autodiff wrapper for ``"autodiff"``."""
    check_curvature(curvature)
    if curvature == "fd":
        return reference
    if isinstance(reference, AutodiffCurvatureReference):      # already wrapped (possibly with a chosen mode/block)
        return reference
    return AutodiffCurvatureReference(reference)


CURVATURE_CHOICES = ("fd", "autodiff")


PERPENDICULAR_GEOMETRY_METHODS = ("fd", "autodiff")


def perpendicular_geometry(reference: Any, points: np.ndarray, method: str = "fd"):
    """``(tensor, divergence)`` of the perpendicular flux tensor ``J (g^ij - b^i b^j)``.

    ``method="fd"`` is the reference's own ``_perpendicular_geometry`` (plain or wrapped reference); ``"autodiff"``
    uses :meth:`AutodiffCurvatureReference.perpendicular_geometry_autodiff` (a plain reference is wrapped first).
    """
    if method not in PERPENDICULAR_GEOMETRY_METHODS:
        raise ValueError(f"method must be one of {PERPENDICULAR_GEOMETRY_METHODS}, got {method!r}")
    points = np.asarray(points, dtype=np.float64)
    if method == "fd":
        return reference._perpendicular_geometry(points)
    if not isinstance(reference, AutodiffCurvatureReference):
        reference = AutodiffCurvatureReference(reference)
    return reference.perpendicular_geometry_autodiff(points)


def face_geometry(reference: Any, points: np.ndarray):
    """P06 face-node ``(J, B, K)`` for ``reference``: the frozen ``p06_structured_global.numerics._face_geometry``
    (one-sided finite-difference wall rule) for a plain reference; for an :class:`AutodiffCurvatureReference`,
    ``J`` and ``B`` from the same ``reference._metric(points)`` call and autodiff ``K`` at *every* node, wall nodes
    included.  Every P-path call site that used ``_face_geometry(env.ref, ...)`` goes through this."""
    points = np.asarray(points, dtype=np.float64)
    if isinstance(reference, AutodiffCurvatureReference):
        metric = reference._metric(points)
        return np.asarray(metric["J"]), np.asarray(metric["B"]), np.asarray(reference._curvature(points))
    import p06_structured_global.numerics as p06numerics
    return p06numerics._face_geometry(reference, points)


def check_curvature(curvature: str) -> str:
    if curvature not in CURVATURE_CHOICES:
        raise ValueError(f"curvature must be one of {CURVATURE_CHOICES}, got {curvature!r}")
    return curvature
