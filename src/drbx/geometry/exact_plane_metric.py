r"""Toroidal metric whose constant-eta planes are exact level sets of a spline scalar potential.

A base toroidal :class:`~drbx.geometry.MetricEvaluator.MetricEvaluator` supplies the smooth channels
``R(u, theta, eta)`` and ``Z(u, theta, eta)``; they only decide where points sit inside each plane.  The toroidal angle
is not the base map's fitted ``eta + delta_phi`` but the exact solution of

.. math::

    \hat\eta(R, \phi, Z) = \eta

for the :class:`~drbx.geometry.spline_scalar_potential.SplineScalarPotentialEvaluator` ``eta_hat`` (solved at every
evaluation point by a fixed number of Newton steps started from the base map's ``phi = eta + delta_phi``).  ``eta_hat``
increases monotonically with ``phi`` in the vessel, so each level set is a graph over ``(R, Z)`` and, by the implicit
function theorem, as smooth as ``R``, ``Z`` and ``eta_hat``.  The Jacobian follows by implicit differentiation,

.. math::

    \phi_q = (\delta_{q\eta} - \hat\eta_R R_q - \hat\eta_Z Z_q) / \hat\eta_\phi, \qquad q \in (u, \theta, \eta),

so ``grad(eta_hat) . X_u = grad(eta_hat) . X_theta = 0`` exactly: the plane normal is ``grad(eta_hat)``.

* :class:`JaxExactPlaneMetricEvaluator` is the immutable PyTree runtime (base
  :class:`~drbx.geometry.jax_metric_evaluator.JaxMetricEvaluator` plus the
  :class:`~drbx.geometry.spline_scalar_potential.JaxSplinePotential`): pointwise, fixed shapes, the ``phi`` solve has
  implicit derivatives (``jax.lax.custom_root``), so ``jit``/``vmap`` and nested ``jacfwd`` (curvature, divergence
  identities) apply.  :meth:`JaxMetricEvaluator.from_metric_evaluator` dispatches to it for an exact-plane metric.
* :class:`ExactPlaneMetricEvaluator` is the NumPy-facing ``MetricEvaluator`` subclass (same public evaluation
  semantics); it evaluates the JAX runtime on fixed padded blocks and returns NumPy arrays.  It carries no fitted
  channels of its own (``_channels`` is empty), so code that reaches into the base representation fails instead of
  silently evaluating the base map.

The cache payload is the base map's payload plus ``plane_construction = "exact_level_set"`` and the spline potential
under ``exact_plane_eta_`` (a metric cache restores the full map; no separate eta is needed).
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .MetricEvaluator import (
    EXACT_PLANE_CONSTRUCTION,
    PLANE_CONSTRUCTION_KEY,
    MetricEvaluator,
    RegularizedMetricEvaluation,
    payload_plane_construction,
)
from .jax_metric_evaluator import JaxMetricEvaluation, JaxMetricEvaluator, JaxRegularizedMetricEvaluation
from .spline_scalar_potential import (
    DEFAULT_PHI_NEWTON_ITERATIONS,
    JaxSplinePotential,
    SplineScalarPotentialEvaluator,
)
from .._host_guards import host_asarray, host_bool

EXACT_PLANE_PAYLOAD_VERSION = 1
#: payload prefix (after the metric prefix) of the spline scalar potential
EXACT_PLANE_ETA_PREFIX = "exact_plane_eta_"
#: block sizes of the NumPy-facing evaluation
_BLOCKS = (512, 4096)
#: largest accepted |eta_hat(X) - eta| of a NumPy-facing evaluation [rad]
_LEVEL_SET_TOLERANCE = 1.0e-9


def _as_points(logical_points: Any):
    q = jnp.asarray(logical_points, dtype=jnp.float64)
    if q.ndim == 0 or q.shape[-1] != 3:
        raise ValueError("logical_points must have shape (..., 3)")
    return q


@jax.tree_util.register_pytree_node_class
class JaxExactPlaneMetricEvaluator:
    """PyTree runtime of :class:`ExactPlaneMetricEvaluator`: base Fourier--Zernike ``R, Z`` + exact spline planes.

    Children: ``base`` (:class:`JaxMetricEvaluator`) and ``spline`` (:class:`JaxSplinePotential`); static metadata:
    ``newton_iterations``.  Every method is pointwise over the leading point axis.  The map is evaluated wherever the
    base channels are (also beyond ``u = 1``, e.g. ``u = 1.02`` for tables extended past the wall); outside the spline
    knot box the spline continues its edge-cell polynomial.
    """

    __slots__ = ("base", "spline", "newton_iterations")

    def __init__(self, base: JaxMetricEvaluator, spline: JaxSplinePotential, *,
                 newton_iterations: int = DEFAULT_PHI_NEWTON_ITERATIONS) -> None:
        if not isinstance(base, JaxMetricEvaluator):
            raise TypeError("base must be a JaxMetricEvaluator")
        if not isinstance(spline, JaxSplinePotential):
            raise TypeError("spline must be a JaxSplinePotential")
        if int(newton_iterations) < 1:
            raise ValueError("newton_iterations must be positive")
        self.base, self.spline, self.newton_iterations = base, spline, int(newton_iterations)

    def tree_flatten(self):
        return (self.base, self.spline), (self.newton_iterations,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        instance = object.__new__(cls)
        object.__setattr__(instance, "base", children[0])
        object.__setattr__(instance, "spline", children[1])
        object.__setattr__(instance, "newton_iterations", aux_data[0])
        return instance

    # -- metadata (as JaxMetricEvaluator) ------------------------------------------------------------------------------
    @property
    def topology(self) -> str:
        return "toroidal"

    @property
    def plane_construction(self) -> str:
        return EXACT_PLANE_CONSTRUCTION

    @property
    def period(self):
        return self.base.period

    @property
    def nfp(self) -> int | None:
        return self.base.nfp

    @property
    def radial_degree(self) -> int:
        return self.base.radial_degree

    @property
    def poloidal_modes(self) -> tuple[int, ...]:
        return self.base.poloidal_modes

    @property
    def toroidal_modes(self) -> tuple[int, ...]:
        return self.base.toroidal_modes

    # -- construction ------------------------------------------------------------------------------------------------
    @classmethod
    def from_metric_evaluator(cls, evaluator: Any) -> "JaxExactPlaneMetricEvaluator":
        if not isinstance(evaluator, ExactPlaneMetricEvaluator):
            raise TypeError("JaxExactPlaneMetricEvaluator requires an ExactPlaneMetricEvaluator")
        return cls(JaxMetricEvaluator.from_metric_evaluator(evaluator.base_metric), evaluator.eta_potential.jax,
                   newton_iterations=evaluator.newton_iterations)

    @classmethod
    def from_cache_payload(cls, payload: Mapping[str, Any], *, prefix: str = "") -> "JaxExactPlaneMetricEvaluator":
        if payload_plane_construction(payload, prefix) != EXACT_PLANE_CONSTRUCTION:
            raise ValueError("payload does not describe an exact-plane metric")
        base = JaxMetricEvaluator._from_plain_cache_payload(payload, prefix=prefix)
        spline = SplineScalarPotentialEvaluator.from_payload(payload, prefix=prefix + EXACT_PLANE_ETA_PREFIX)
        return cls(base, spline.jax, newton_iterations=_payload_newton_iterations(payload, prefix))

    # -- core ----------------------------------------------------------------------------------------------------------
    def _solve(self, eta, R, Z, delta):
        phi = self.spline.solve_phi(R, Z, eta, eta + delta, iterations=self.newton_iterations)
        _, eta_R, eta_phi, eta_Z = self.spline.eta_and_partials(R, phi, Z)
        return phi, eta_R, eta_phi, eta_Z

    def _channels(self, q):
        values, du, dtheta, deta = self.base._values_and_derivatives(q)
        return (tuple(values[..., i] for i in range(3)), tuple(du[..., i] for i in range(3)),
                tuple(dtheta[..., i] for i in range(3)), tuple(deta[..., i] for i in range(3)))

    def _core(self, q):
        """``(position, jacobian, phi, R, Z)`` at ``q (..., 3)``."""

        (R, Z, delta), (Ru, Zu, _), (Rt, Zt, _), (Re, Ze, _) = self._channels(q)
        phi, eta_R, eta_phi, eta_Z = self._solve(q[..., 2], R, Z, delta)
        phi_u = -(eta_R * Ru + eta_Z * Zu) / eta_phi
        phi_t = -(eta_R * Rt + eta_Z * Zt) / eta_phi
        phi_e = (1.0 - eta_R * Re - eta_Z * Ze) / eta_phi
        cp, sp = jnp.cos(phi), jnp.sin(phi)

        def cart(rd, pd, zd):
            return jnp.stack((cp * rd - R * sp * pd, sp * rd + R * cp * pd, zd), axis=-1)

        position = jnp.stack((R * cp, R * sp, Z), axis=-1)
        jacobian = jnp.stack((cart(Ru, phi_u, Zu), cart(Rt, phi_t, Zt), cart(Re, phi_e, Ze)), axis=-1)
        return position, jacobian, phi, R, Z

    def _position_core(self, q):
        """``(position, phi, R, Z)`` from the base channel values only."""

        values = self.base._channel_values(*(q.reshape(-1, 3)[:, i] for i in range(3))).reshape(q.shape)
        R, Z, delta = values[..., 0], values[..., 1], values[..., 2]
        phi = self.spline.solve_phi(R, Z, q[..., 2], q[..., 2] + delta, iterations=self.newton_iterations)
        return jnp.stack((R * jnp.cos(phi), R * jnp.sin(phi), Z), axis=-1), phi, R, Z

    def position_and_jacobian(self, logical_points: Any):
        """Position and logical Jacobian (columns ``u, theta, eta``) with one base and one spline evaluation."""

        position, jacobian, *_ = self._core(_as_points(logical_points))
        return position, jacobian

    def position(self, logical_points: Any):
        return self._position_core(_as_points(logical_points))[0]

    def jacobian_matrix(self, logical_points: Any):
        return self.position_and_jacobian(logical_points)[1]

    def level_set_residual(self, logical_points: Any):
        """``eta_hat(X(q)) - eta`` (round-off for converged Newton solves)."""

        q = _as_points(logical_points)
        _, phi, R, Z = self._position_core(q)
        return self.spline.eta(R, phi, Z) - q[..., 2]

    def channel_values_and_derivatives(self, logical_points: Any):
        """``(R, Z, delta_phi)`` of the exact map (``delta_phi = phi - eta``) and their ``u, theta, eta`` derivatives."""

        q = _as_points(logical_points)
        (R, Z, delta), (Ru, Zu, _), (Rt, Zt, _), (Re, Ze, _) = self._channels(q)
        phi, eta_R, eta_phi, eta_Z = self._solve(q[..., 2], R, Z, delta)
        phi_u = -(eta_R * Ru + eta_Z * Zu) / eta_phi
        phi_t = -(eta_R * Rt + eta_Z * Zt) / eta_phi
        phi_e = (1.0 - eta_R * Re - eta_Z * Ze) / eta_phi
        return (jnp.stack((R, Z, phi - q[..., 2]), -1), jnp.stack((Ru, Zu, phi_u), -1),
                jnp.stack((Rt, Zt, phi_t), -1), jnp.stack((Re, Ze, phi_e - 1.0), -1))

    def evaluate(self, logical_points: Any, *, reject_nonpositive_J: bool = True):
        """Position, Jacobian, determinant and metric tensors (as :meth:`JaxMetricEvaluator.evaluate`)."""

        q = _as_points(logical_points)
        position, jacobian = self.position_and_jacobian(q)
        signed_J = jnp.linalg.det(jacobian)
        valid = jnp.isfinite(signed_J) & (signed_J > 0.0)
        if reject_nonpositive_J:
            try:
                if host_bool(np.any(~host_asarray(valid))):
                    raise ValueError("query contains nonpositive or nonfinite mesh Jacobian")
            except (TypeError, jax.errors.ConcretizationTypeError, jax.errors.TracerBoolConversionError):
                pass
        covariant = jnp.einsum("...ki,...kj->...ij", jacobian, jacobian)
        contravariant = jnp.linalg.inv(covariant)
        identity = jnp.eye(3, dtype=jnp.float64)
        residual = jnp.max(jnp.abs(jnp.einsum("...ik,...kj->...ij", covariant, contravariant) - identity),
                           axis=(-2, -1))
        return JaxMetricEvaluation(position, jacobian, signed_J, covariant, contravariant, residual, valid)

    def regularized_frame(self, logical_points: Any):
        """``(position, [X_u, X_theta/u, period/(2 pi) X_eta])``, finite on the axis ``u = 0``."""

        q = _as_points(logical_points)
        shape = q.shape[:-1]
        flat = q.reshape((-1, 3))
        u, theta, eta = flat[:, 0], flat[:, 1], flat[:, 2]
        base = self.base
        values = base._channel_values(u, theta, eta).reshape(shape + (3,))
        du = base._channel_values(u, theta, eta, derivative=True).reshape(shape + (3,))
        dt = base._channel_values(u, theta, eta, theta_over_u=True).reshape(shape + (3,))
        de = base._channel_values(u, theta, eta, eta_derivative=True).reshape(shape + (3,))
        R, Z, delta = values[..., 0], values[..., 1], values[..., 2]
        phi, eta_R, eta_phi, eta_Z = self._solve(q[..., 2], R, Z, delta)
        Ru, Zu, Rt, Zt, Re, Ze = du[..., 0], du[..., 1], dt[..., 0], dt[..., 1], de[..., 0], de[..., 1]
        phi_u = -(eta_R * Ru + eta_Z * Zu) / eta_phi
        phi_t = -(eta_R * Rt + eta_Z * Zt) / eta_phi
        phi_e = (1.0 - eta_R * Re - eta_Z * Ze) / eta_phi
        cp, sp = jnp.cos(phi), jnp.sin(phi)

        def cart(rd, pd, zd):
            return jnp.stack((cp * rd - R * sp * pd, sp * rd + R * cp * pd, zd), axis=-1)

        frame = jnp.stack((cart(Ru, phi_u, Zu), cart(Rt, phi_t, Zt),
                           (self.period / (2.0 * jnp.pi)) * cart(Re, phi_e, Ze)), axis=-1)
        return jnp.stack((R * cp, R * sp, Z), axis=-1), frame

    def evaluate_regularized(self, logical_points: Any):
        """Axis-regularized frame and metrics (as :meth:`JaxMetricEvaluator.evaluate_regularized`)."""

        position, frame = self.regularized_frame(logical_points)
        J = jnp.linalg.det(frame)
        cov = jnp.einsum("...ki,...kj->...ij", frame, frame)
        contra = jnp.linalg.inv(cov)
        identity = jnp.eye(3, dtype=jnp.float64)
        residual = jnp.max(jnp.abs(jnp.einsum("...ik,...kj->...ij", cov, contra) - identity), axis=(-2, -1))
        condition = jnp.linalg.cond(frame)
        valid = jnp.isfinite(J) & (J > 0.0) & jnp.isfinite(condition)
        return JaxRegularizedMetricEvaluation(position, frame, J, cov, contra, condition, residual, valid)


def _payload_newton_iterations(payload: Mapping[str, Any], prefix: str) -> int:
    key = f"{prefix}exact_plane_newton_iterations"
    try:
        return int(np.asarray(payload[key]).item())
    except KeyError as error:
        raise ValueError(f"exact-plane payload is missing {key!r}") from error


# ---------------------------------------------------------------------------------------------------------------------
# Jitted kernels of the NumPy-facing evaluator
# ---------------------------------------------------------------------------------------------------------------------
@jax.jit
def _position_kernel(runtime, q):
    position, phi, R, Z = runtime._position_core(q)
    return position, runtime.spline.eta(R, phi, Z) - q[:, 2]


@jax.jit
def _position_and_jacobian_kernel(runtime, q):
    position, jacobian, phi, R, Z = runtime._core(q)
    return position, jacobian, runtime.spline.eta(R, phi, Z) - q[:, 2]


@jax.jit
def _regularized_kernel(runtime, q):
    position, frame = runtime.regularized_frame(q)
    return position, frame


def _block_size(n: int) -> int:
    for size in _BLOCKS:
        if n <= size:
            return size
    return _BLOCKS[-1]


# ---------------------------------------------------------------------------------------------------------------------
# NumPy-facing evaluator
# ---------------------------------------------------------------------------------------------------------------------
class ExactPlaneMetricEvaluator(MetricEvaluator):
    """A toroidal ``MetricEvaluator`` with exact level-set planes of a spline ``eta`` (see the module docstring).

    Parameters
    ----------
    base_metric:
        Toroidal Fourier--Zernike ``MetricEvaluator`` supplying ``R`` and ``Z`` (and the Newton start
        ``phi = eta + delta_phi``); normally built on the same ``eta_potential``'s wall curves, so ``u = 1`` lies on
        the vessel.
    eta_potential:
        :class:`SplineScalarPotentialEvaluator` with the base map's field period.
    newton_iterations:
        Fixed Newton steps of the ``phi`` solve (from a start ~1e-4 rad off, 3 reach round-off).

    ``evaluate``/``position``/``jacobian_matrix``/``sample*``/``evaluate_regularized``/``evaluate_magnetic_field``
    keep the base class semantics (``u`` in ``[0, 1]``, the ``u = 0`` refusal of ``evaluate``, nonpositive-Jacobian
    rejection).  An evaluation additionally raises when a point leaves the spline knot box or its level-set residual
    ``|eta_hat(X) - eta|`` exceeds ``1e-9`` (a non-converged solve).
    """

    def __init__(self, base_metric: MetricEvaluator, eta_potential: SplineScalarPotentialEvaluator, *,
                 newton_iterations: int = DEFAULT_PHI_NEWTON_ITERATIONS) -> None:
        if not isinstance(base_metric, MetricEvaluator) or isinstance(base_metric, ExactPlaneMetricEvaluator):
            raise TypeError("base_metric must be a fitted (not exact-plane) MetricEvaluator")
        if base_metric.topology != "toroidal":
            raise ValueError("exact level-set planes require a toroidal base metric")
        if not isinstance(eta_potential, SplineScalarPotentialEvaluator):
            raise TypeError("eta_potential must be a SplineScalarPotentialEvaluator")
        if not np.isclose(base_metric.period, eta_potential.period, rtol=2e-12, atol=2e-12):
            raise ValueError("base metric and eta potential field periods disagree")
        if base_metric.nfp is not None and int(base_metric.nfp) != int(eta_potential.nfp):
            raise ValueError("base metric and eta potential nfp disagree")
        if int(newton_iterations) < 1:
            raise ValueError("newton_iterations must be positive")
        for name in ("_u", "_v", "_eta", "_topology", "_period", "_nfp", "_metric_spline_degree", "_radial_degree",
                     "_poloidal_modes", "_toroidal_modes", "_mmpde_result", "_mmpde_fit_scale"):
            setattr(self, name, getattr(base_metric, name))
        # No fitted channels of its own: the base map's channels are not this map's.
        self._channels = ()
        self._base_metric = base_metric
        self._eta_potential = eta_potential
        self._newton_iterations = int(newton_iterations)
        self._runtime = JaxExactPlaneMetricEvaluator(JaxMetricEvaluator.from_metric_evaluator(base_metric),
                                                     eta_potential.jax, newton_iterations=self._newton_iterations)

    # -- metadata ------------------------------------------------------------------------------------------------------
    @property
    def base_metric(self) -> MetricEvaluator:
        """The fitted base map (its planes are not this map's planes)."""
        return self._base_metric

    @property
    def eta_potential(self) -> SplineScalarPotentialEvaluator:
        return self._eta_potential

    @property
    def newton_iterations(self) -> int:
        return self._newton_iterations

    @property
    def plane_construction(self) -> str:
        return EXACT_PLANE_CONSTRUCTION

    @property
    def jax_runtime(self) -> JaxExactPlaneMetricEvaluator:
        return self._runtime

    def basis_cache_diagnostics(self) -> dict[str, Any]:
        return self._base_metric.basis_cache_diagnostics()

    # -- blocked evaluation --------------------------------------------------------------------------------------------
    def _logical(self, logical_points: Any) -> tuple[np.ndarray, tuple[int, ...]]:
        q = np.asarray(logical_points, dtype=np.float64)
        if q.ndim == 0 or q.shape[-1] != 3:
            raise ValueError("logical_points must have shape (..., 3)")
        if not np.all(np.isfinite(q)):
            raise ValueError("logical points must be finite")
        flat = q.reshape(-1, 3)
        if np.any((flat[:, 0] < self._u[0]) | (flat[:, 0] > self._u[-1])):
            raise ValueError("u and v queries must lie in [0, 1]")
        return flat, q.shape[:-1]

    def _run(self, kernel, flat: np.ndarray) -> list[np.ndarray]:
        n = flat.shape[0]
        block = _block_size(max(n, 1))
        pieces: list[list[np.ndarray]] | None = None
        for start in range(0, n, block):
            part = flat[start:start + block]
            count = part.shape[0]
            if count < block:
                part = np.concatenate([part, np.repeat(part[-1:], block - count, axis=0)])
            result = [np.asarray(value)[:count] for value in kernel(self._runtime, jnp.asarray(part))]
            if pieces is None:
                pieces = [[] for _ in result]
            for store, value in zip(pieces, result):
                store.append(value)
        return [np.concatenate(store, axis=0) for store in pieces]

    def _check(self, position: np.ndarray, residual: np.ndarray | None) -> None:
        R = np.hypot(position[:, 0], position[:, 1])
        box = self._eta_potential.knot_box
        tol_R, tol_Z = 1e-9 * box["h_R"], 1e-9 * box["h_Z"]
        if (R.min() < box["R_lo"] - tol_R or R.max() > box["R_hi"] + tol_R or position[:, 2].min() < box["Z_lo"] - tol_Z
                or position[:, 2].max() > box["Z_hi"] + tol_Z):
            raise ValueError("an exact-plane evaluation point lies outside the spline eta knot box")
        if residual is not None and not np.all(np.abs(residual) <= _LEVEL_SET_TOLERANCE):
            raise ValueError("the exact-plane phi solve did not converge (|eta_hat(X) - eta| > 1e-9)")

    def _position_only(self, logical_points: Any) -> np.ndarray:
        flat, shape = self._logical(logical_points)
        if flat.shape[0] == 0:
            return np.zeros(shape + (3,))
        position, residual = self._run(_position_kernel, flat)
        self._check(position, residual)
        return position.reshape(shape + (3,))

    def _position_and_jacobian(self, logical_points: Any) -> tuple[np.ndarray, np.ndarray]:
        flat, shape = self._logical(logical_points)
        if flat.shape[0] == 0:
            return np.zeros(shape + (3,)), np.zeros(shape + (3, 3))
        position, jacobian, residual = self._run(_position_and_jacobian_kernel, flat)
        self._check(position, residual)
        return position.reshape(shape + (3,)), jacobian.reshape(shape + (3, 3))

    def level_set_residual(self, logical_points: Any) -> np.ndarray:
        """``eta_hat(X(q)) - eta`` at logical points (round-off)."""

        flat, shape = self._logical(logical_points)
        _, residual = self._run(_position_kernel, flat)
        return residual.reshape(shape)

    def evaluate_regularized(self, logical_points: Any) -> RegularizedMetricEvaluation:
        """``[X_u, X_theta/u, (period/2 pi) X_eta]`` of the exact map (finite on the axis)."""

        flat, shape = self._logical(logical_points)
        position, H = self._run(_regularized_kernel, flat)
        self._check(position, None)
        position, H = position.reshape(shape + (3,)), H.reshape(shape + (3, 3))
        J = np.linalg.det(H)
        cov = np.einsum("...ki,...kj->...ij", H, H)
        contra = np.linalg.inv(cov)
        residual = np.max(np.abs(np.einsum("...ik,...kj->...ij", cov, contra) - np.eye(3)), axis=(-2, -1))
        condition = np.linalg.cond(H)
        valid = np.isfinite(J) & (J > 0) & np.isfinite(condition)
        return RegularizedMetricEvaluation(position, H, J, cov, contra, condition, residual, valid)

    def validate(self) -> None:
        """Raise unless the regularized frame is positive on the toroidal builder's validation grid.

        Same grid as ``build_metric_evaluator(..., validate=True)``: the axis, ``u = min(u_1/4, 1)``, the radial
        midpoints and the wall, at ``theta`` and ``eta`` nodes offset by 0.37 and 0.31 cells.
        """

        u, theta, eta = self._u, self._v, self._eta
        radial = np.unique(np.concatenate(([0.0, min(0.25 * u[1], 1.0)], 0.5 * (u[:-1] + u[1:]), [1.0])))
        points = np.stack(np.meshgrid(radial, theta + 0.37 * (2.0 * np.pi / theta.size),
                                      eta + 0.31 * (self._period / eta.size), indexing="ij"), axis=-1)
        regularized = self.evaluate_regularized(points)
        if np.any(~regularized.valid):
            raise ValueError("the exact-plane map folds on the validation grid; minimum regularized Jacobian is "
                             f"{float(np.nanmin(regularized.J_reg)):.6e}")
        residual = self.level_set_residual(points[1:])
        if np.max(np.abs(residual)) > _LEVEL_SET_TOLERANCE:
            raise ValueError("the exact-plane phi solve did not converge on the validation grid")

    # -- payload ------------------------------------------------------------------------------------------------------
    def to_cache_payload(self, *, prefix: str = "") -> dict[str, np.ndarray]:
        """Base map payload + the exact-plane flag + the spline potential (``prefix + exact_plane_eta_``)."""

        payload = self._base_metric.to_cache_payload(prefix=prefix)
        payload[f"{prefix}{PLANE_CONSTRUCTION_KEY}"] = np.asarray(EXACT_PLANE_CONSTRUCTION)
        payload[f"{prefix}exact_plane_payload_version"] = np.asarray(EXACT_PLANE_PAYLOAD_VERSION, dtype=np.int64)
        payload[f"{prefix}exact_plane_newton_iterations"] = np.asarray(self._newton_iterations, dtype=np.int64)
        payload.update(self._eta_potential.to_payload(prefix=prefix + EXACT_PLANE_ETA_PREFIX))
        return payload

    @classmethod
    def from_cache_payload(cls, payload: Any, *, prefix: str = "") -> "ExactPlaneMetricEvaluator":
        construction = payload_plane_construction(payload, prefix)
        if construction != EXACT_PLANE_CONSTRUCTION:
            raise ValueError(f"payload plane construction is {construction!r}, not {EXACT_PLANE_CONSTRUCTION!r}")
        version_key = f"{prefix}exact_plane_payload_version"
        if version_key not in payload or int(np.asarray(payload[version_key]).item()) != EXACT_PLANE_PAYLOAD_VERSION:
            raise ValueError("unsupported exact-plane payload version")
        base = MetricEvaluator._from_plain_cache_payload(payload, prefix=prefix)
        eta = SplineScalarPotentialEvaluator.from_payload(payload, prefix=prefix + EXACT_PLANE_ETA_PREFIX)
        return cls(base, eta, newton_iterations=_payload_newton_iterations(payload, prefix))


__all__ = [
    "EXACT_PLANE_ETA_PREFIX",
    "ExactPlaneMetricEvaluator",
    "JaxExactPlaneMetricEvaluator",
]
