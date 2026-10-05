"""Autodiff magnetic curvature ``K = (B / 2|J|) curl(b_cov / B)`` in logical coordinates.

``K`` is the logical-coordinate (contravariant) curvature vector used by the
perpendicular curvature operator of the DRB models.  The finite-difference
reference (``hsx_mms_continuum_reference._curvature``) forms ``A = b_cov / B``
from the metric and the magnetic field, differentiates each component with a
4th-order central stencil and contracts::

    curl(A)_u     = d_theta A_eta - d_eta   A_theta
    curl(A)_theta = d_eta   A_u   - d_u     A_eta
    curl(A)_eta   = d_u     A_theta - d_theta A_u
    K             = 0.5 * B * curl(A) / max(|J|, 1e-30)

This module evaluates the same expression with :func:`jax.jacfwd` through the
JAX evaluators (:class:`~drbx.geometry.jax_metric_evaluator.JaxMetricEvaluator`
and :class:`~drbx.geometry.jax_bfield_evaluator.JaxComponentSplineBFieldEvaluator`)
that mirror the NumPy interpolants to about 4e-15.  There is no step size, no
one-sided rule at the radial ends, and every query point is independent (the
per-point function has no reductions over the point axis).  XLA compiles a
different program (different vectorization/FMA choices, roundoff ~1e-12 of K
after the curl cancellation) for each vmapped batch shape, so batch-size
independence has to be built in.  Points are processed in blocks of exactly
``block`` (the last block padded by repeating its final point and trimmed
afterwards), so each function is compiled once whatever the query size, and:

* ``mode="block"`` (the default) vmaps each block.  For a fixed ``block`` the result
  of a point does not depend on which other points share its call, on their order,
  or on the chunking of the caller: QK4 of the P08 bundle (K2) builds the geometry
  arrays of a bounded HSX unit (4608 raw cells, 4608 face rows = 41k q3 nodes
  including 1024 wall face rows) with geometry chunks of 4096 / 2048 / 1024 and finds
  every array, ``p06_raw_K`` and ``p06_face_K`` included, bitwise identical.  It is
  about 6-15x faster than sequential.
* ``mode="sequential"`` runs the per-point function under :func:`jax.lax.map` (one
  point per loop iteration): bitwise independent of everything by construction, but
  slower, and a different XLA program from the block one (values differ by ~1e-12
  relative).

Domain
------
* ``u = 1`` (the wall) is evaluated exactly: the metric map is a polynomial in
  ``u`` (Zernike radial basis, no clamping), and the B-field spline is
  periodic/reflected-padded in ``(R, Z)`` space, so both are smooth across
  ``u = 1`` and their derivatives exist there.
* ``u -> 0``: the logical frame is singular on the axis (``J ~ u``, and the
  contravariant components of ``K`` grow like ``1/u``: about 6e6 at ``u = 1e-6``
  against ~9 at ``u = 1``).  Autodiff is finite and accurate for every ``u > 0``
  (agreeing with the finite-difference reference to ~1e-8 relative at
  ``u = 1e-6``, ~1e-11 from ``u = 1e-3`` up; ``u <= ~1e-8`` cannot be compared, the
  finite-difference step is shrunk below the domain there), but the metric map is
  not evaluated on the axis itself: ``u = 0`` returns NaN.  Beyond the wall the
  polynomial map and the padded spline simply continue (``u = 1.001`` is
  finite); callers should not rely on that.

``curvature_divergence_identity`` evaluates ``d_i (|J| K^i / B)`` by autodiff.
Since ``|J| K / B = curl(A) / 2`` (up to the sign of ``J``), the divergence is
identically zero for any smooth ``A``; the returned value is the roundoff of the
second-derivative pipeline (about 1e-11 of the sum of the three term magnitudes on
real HSX points, the curl's cancellation amplified once more) and is a
self-consistency check that the finite-difference stencil cannot provide.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "AutodiffCurvature",
    "autodiff_curvature",
    "curvature_divergence_identity",
    "covariant_over_B_parts",
    "AutodiffPerpendicularGeometry",
    "autodiff_perpendicular_geometry",
    "perpendicular_flux_tensor_one",
]

_FLOOR = 1.0e-30
#: batching default, decided by QK4 (geometry chunk-size independence, bitwise) -- see the module docstring
DEFAULT_MODE = "block"


def _field_at(jax_bfield: Any, logical_field: Any, q, m, B0: float):
    """``(Bc, Bcontra)``: Cartesian field and the contravariant field divided by ``B0`` at the logical point(s) ``q``.

    ``logical_field=None`` (the default): the Cartesian field of ``jax_bfield`` at the mapped position, projected with
    the logical Jacobian matrix.  Otherwise ``logical_field.contravariant(q, m)`` (physical units) is the field -- for
    instance :class:`drbx.geometry.eta_filtered_field.JaxEtaFilterTable`, a field defined in logical coordinates --
    and the Cartesian field is the Jacobian matrix applied to it.
    """

    if logical_field is None:
        Bc = jax_bfield.evaluate_cartesian(m.position)
        return Bc, jnp.linalg.solve(m.jacobian_matrix, Bc[..., None])[..., 0] / B0
    Bcontra = logical_field.contravariant(q, m)
    return jnp.einsum("...ij,...j->...i", m.jacobian_matrix, Bcontra), Bcontra / B0


def covariant_over_B_parts(jax_metric: Any, jax_bfield: Any, B0: float, logical_field: Any = None) -> Callable:
    """Return ``parts(q) -> (A, B, J)`` for one logical point ``q`` of shape ``(3,)``.

    ``A = b_cov / B`` (covariant components of ``b_hat / |B|``), ``B = |B|/B0`` and
    ``J`` the signed metric Jacobian.  The expressions reproduce
    ``hsx_mms_continuum_reference._metric_batch`` (contravariant field from the
    Cartesian B and the logical Jacobian matrix, ``b_cov = g_cov b^i``).

    ``logical_field`` (default ``None``: the field is ``jax_bfield.evaluate_cartesian`` of the mapped position, as
    always) replaces the Cartesian evaluator by a field given in logical coordinates (``contravariant(q, metric)``,
    for example the eta-filtered table of :mod:`drbx.geometry.eta_filtered_field`); ``jax_bfield`` may then be ``None``.
    """

    B0 = float(B0)

    def parts(q):
        qb = q[None]
        m = jax_metric.evaluate(qb, reject_nonpositive_J=False)
        Bc, Bcontra = _field_at(jax_bfield, logical_field, qb, m, B0)
        bmag = jnp.maximum(jnp.linalg.norm(Bc, axis=-1) / B0, _FLOOR)
        bunit = Bcontra / bmag[..., None]
        bcov = jnp.einsum("...ij,...j->...i", m.g_cov, bunit)
        return (bcov / bmag[..., None])[0], bmag[0], m.J[0]

    return parts


def _curl_from_jacobian(d):
    # d[component, axis] = d_axis A_component
    return jnp.stack((d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]))


class AutodiffCurvature:
    """Jitted, vmapped autodiff curvature for one geometry.

    Calling the object with ``points`` of shape ``(Q, 3)`` (``(u, theta, eta)``)
    returns ``K`` as a float64 NumPy array ``(Q, 3)``.  See the module docstring
    for ``mode`` ("block", the default: vmapped blocks of ``block`` points; "sequential":
    ``lax.map`` over the points).  Every method (``K``, ``A``, the
    divergence identity) uses the same mode.
    """

    def __init__(self, jax_metric: Any, jax_bfield: Any, B0: float, *, mode: str = DEFAULT_MODE,
                 block: int = 256, logical_field: Any = None) -> None:
        if mode not in ("sequential", "block"):
            raise ValueError("mode must be 'sequential' or 'block'")
        if int(block) < 1:
            raise ValueError("block must be a positive integer")
        self.B0 = float(B0)
        self.mode = mode
        self.block = int(block)
        parts = covariant_over_B_parts(jax_metric, jax_bfield, self.B0, logical_field)
        self._parts = parts
        A_only = lambda q: parts(q)[0]  # noqa: E731

        def K_one(q):
            d = jax.jacfwd(A_only)(q)
            _, bmag, J = parts(q)
            return 0.5 * bmag * _curl_from_jacobian(d) / jnp.maximum(jnp.abs(J), _FLOOR)

        def flux_one(q):
            # |J| K^i / B  (= curl(A)/2 whenever |J| > the floor)
            _, bmag, J = parts(q)
            return jnp.abs(J) * K_one(q) / bmag

        def divergence_terms_one(q):
            return jnp.diagonal(jax.jacfwd(flux_one)(q))

        self._K_one = K_one
        lift = (lambda f: jax.jit(lambda q: jax.lax.map(f, q))) if mode == "sequential" else (lambda f: jax.jit(jax.vmap(f)))
        self._K = lift(K_one)
        self._A = lift(A_only)
        self._flux = lift(flux_one)
        self._div_terms = lift(divergence_terms_one)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _as_points(points: Any) -> np.ndarray:
        q = np.asarray(points, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("points must have shape (Q, 3)")
        return q

    def _apply(self, fn: Callable, points: Any, trailing: tuple[int, ...]) -> np.ndarray:
        q = self._as_points(points)
        if len(q) == 0:
            return np.zeros((0,) + trailing, dtype=np.float64)
        out = []
        for start in range(0, len(q), self.block):
            part = q[start:start + self.block]
            padded = np.concatenate([part, np.repeat(part[-1:], self.block - len(part), axis=0)])
            out.append(np.asarray(fn(jnp.asarray(padded)))[:len(part)])
        return np.concatenate(out, axis=0)

    # ------------------------------------------------------------------ API
    def __call__(self, points: Any) -> np.ndarray:
        """``K`` at ``points (Q, 3)`` -> ``(Q, 3)`` float64."""
        return self._apply(self._K, points, (3,))

    def covariant_over_B(self, points: Any) -> np.ndarray:
        """``A = b_cov / B`` at ``points`` -> ``(Q, 3)`` (the differentiated field)."""
        return self._apply(self._A, points, (3,))

    def divergence_identity_terms(self, points: Any) -> np.ndarray:
        """The three diagonal terms ``d_i (|J| K^i / B)`` -> ``(Q, 3)``; their sum is the identity."""
        return self._apply(self._div_terms, points, (3,))

    def divergence_identity(self, points: Any) -> np.ndarray:
        """``sum_i d_i (|J| K^i / B)`` -> ``(Q,)``; identically zero, so this is roundoff."""
        return self.divergence_identity_terms(points).sum(axis=-1)

    def flux(self, points: Any) -> np.ndarray:
        """``|J| K / B`` -> ``(Q, 3)`` (the field whose divergence vanishes)."""
        return self._apply(self._flux, points, (3,))


def autodiff_curvature(jax_metric: Any, jax_bfield: Any, B0: float, *, mode: str = DEFAULT_MODE,
                       block: int = 256, logical_field: Any = None) -> AutodiffCurvature:
    """Return the callable ``K(points (Q, 3)) -> (Q, 3)`` (float64), jitted and vmapped.

    The returned :class:`AutodiffCurvature` also carries ``divergence_identity``.  ``logical_field`` (default ``None``)
    is as in :func:`covariant_over_B_parts`.
    """
    return AutodiffCurvature(jax_metric, jax_bfield, B0, mode=mode, block=block, logical_field=logical_field)


def curvature_divergence_identity(jax_metric: Any, jax_bfield: Any, B0: float, points: Any,
                                  logical_field: Any = None) -> np.ndarray:
    """``d_i (|J| K^i / B)`` at ``points (Q, 3)`` by autodiff (zero up to roundoff)."""
    return AutodiffCurvature(jax_metric, jax_bfield, B0, logical_field=logical_field).divergence_identity(points)


def perpendicular_flux_tensor_one(jax_metric: Any, jax_bfield: Any, B0: float, logical_field: Any = None) -> Callable:
    """Return ``tensor(q) -> J (g^{ij} - b^i b^j)`` (``(3, 3)``) for one logical point ``q`` of shape ``(3,)``.

    ``b`` is the contravariant unit vector (contravariant field from the Cartesian B and the logical Jacobian matrix,
    as in :func:`covariant_over_B_parts`), ``g^{ij}`` the contravariant metric and ``J`` the *signed* Jacobian
    (``metric.signed_J``), exactly as ``hsx_mms_continuum_reference._perpendicular_flux_tensor``.  ``logical_field`` is
    as in :func:`covariant_over_B_parts`.
    """

    B0 = float(B0)

    def tensor(q):
        qb = q[None]
        m = jax_metric.evaluate(qb, reject_nonpositive_J=False)
        Bc, Bcontra = _field_at(jax_bfield, logical_field, qb, m, B0)
        bmag = jnp.maximum(jnp.linalg.norm(Bc, axis=-1) / B0, _FLOOR)
        bunit = Bcontra / bmag[..., None]
        projector = m.g_contra - jnp.einsum("...i,...j->...ij", bunit, bunit)
        return (m.J[..., None, None] * projector)[0]

    return tensor


class AutodiffPerpendicularGeometry:
    """Jitted, vmapped autodiff of the perpendicular flux tensor ``T^{ij} = J (g^{ij} - b^i b^j)`` and its divergence.

    ``__call__(points (Q, 3))`` returns ``(tensor (Q, 3, 3), divergence (Q, 3))`` with
    ``divergence[j] = sum_i d_i T^{ij}`` (``jax.jacfwd`` of the per-point tensor; no step size), the quantities of
    ``hsx_mms_continuum_reference._perpendicular_geometry``.  Batching (``mode``, ``block``, padding of the last
    block by repeating its final point) is the one of :class:`AutodiffCurvature`: for a fixed ``block`` the result of
    a point does not depend on the other points of the call or on the chunking of the caller.
    """

    def __init__(self, jax_metric: Any, jax_bfield: Any, B0: float, *, mode: str = DEFAULT_MODE,
                 block: int = 256, logical_field: Any = None) -> None:
        if mode not in ("sequential", "block"):
            raise ValueError("mode must be 'sequential' or 'block'")
        if int(block) < 1:
            raise ValueError("block must be a positive integer")
        self.B0 = float(B0)
        self.mode = mode
        self.block = int(block)
        tensor_one = perpendicular_flux_tensor_one(jax_metric, jax_bfield, self.B0, logical_field)
        self._tensor_one = tensor_one

        def geometry_one(q):
            jac = jax.jacfwd(tensor_one)(q)               # jac[k, j, i] = d_i T^{kj}
            tensor = tensor_one(q)
            divergence = jnp.einsum("iji->j", jac)        # sum_i d_i T^{ij}
            return jnp.concatenate([tensor.reshape(9), divergence])

        lift = (lambda f: jax.jit(lambda q: jax.lax.map(f, q))) if mode == "sequential" else (lambda f: jax.jit(jax.vmap(f)))
        self._geometry = lift(geometry_one)

    def __call__(self, points: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(tensor (Q, 3, 3), divergence (Q, 3))`` float64 at ``points (Q, 3)``."""
        q = AutodiffCurvature._as_points(points)
        if len(q) == 0:
            return np.zeros((0, 3, 3)), np.zeros((0, 3))
        out = []
        for start in range(0, len(q), self.block):
            part = q[start:start + self.block]
            padded = np.concatenate([part, np.repeat(part[-1:], self.block - len(part), axis=0)])
            out.append(np.asarray(self._geometry(jnp.asarray(padded)))[:len(part)])
        flat = np.concatenate(out, axis=0)
        return flat[:, :9].reshape(-1, 3, 3), flat[:, 9:]


def autodiff_perpendicular_geometry(jax_metric: Any, jax_bfield: Any, B0: float, *, mode: str = DEFAULT_MODE,
                                    block: int = 256, logical_field: Any = None) -> AutodiffPerpendicularGeometry:
    """Return the callable ``points (Q, 3) -> (tensor (Q, 3, 3), divergence (Q, 3))`` (float64), jitted and vmapped."""
    return AutodiffPerpendicularGeometry(jax_metric, jax_bfield, B0, mode=mode, block=block,
                                         logical_field=logical_field)
