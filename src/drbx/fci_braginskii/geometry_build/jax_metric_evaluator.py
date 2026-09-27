"""JAX runtime evaluator for toroidal Fourier--Zernike metric maps.

``MetricEvaluator`` is intentionally a NumPy/SciPy producer.  This module is
the small, immutable runtime representation used when the fitted map has to be
evaluated from JAX (including inside :func:`jax.jit`).  The conversion methods
consume the numeric payload returned by ``MetricEvaluator.to_cache_payload``;
no SciPy objects are retained by the runtime evaluator and SciPy is not used
while evaluating a query.

The point dimension is flattened only locally for the Fourier contraction.
There are no reductions over that dimension, so a jitted call can be used with
leading-point sharding in the same way as any other pointwise JAX kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np


_pytree_base = jax.tree_util.register_pytree_node_class


@_pytree_base
@dataclass(frozen=True)
class JaxMetricEvaluation:
    """JAX metric quantities returned by :meth:`JaxMetricEvaluator.evaluate`."""

    position: Any
    jacobian_matrix: Any
    signed_J: Any
    covariant_metric: Any
    contravariant_metric: Any
    inverse_residual: Any
    valid: Any

    def tree_flatten(self):
        return (self.position, self.jacobian_matrix, self.signed_J,
                self.covariant_metric, self.contravariant_metric,
                self.inverse_residual, self.valid), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def J(self):
        return self.signed_J

    @property
    def g_cov(self):
        return self.covariant_metric

    @property
    def g_contra(self):
        return self.contravariant_metric


@_pytree_base
@dataclass(frozen=True)
class JaxRegularizedMetricEvaluation:
    """Axis-regularized JAX metric quantities."""

    position: Any
    regularized_jacobian_matrix: Any
    regularized_J: Any
    covariant_metric: Any
    contravariant_metric: Any
    condition: Any
    inverse_residual: Any
    valid: Any

    def tree_flatten(self):
        return (self.position, self.regularized_jacobian_matrix,
                self.regularized_J, self.covariant_metric,
                self.contravariant_metric, self.condition,
                self.inverse_residual, self.valid), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def J(self):
        return self.regularized_J

    @property
    def J_reg(self):
        return self.regularized_J

    @property
    def jacobian_matrix(self):
        return self.regularized_jacobian_matrix

    @property
    def condition_number(self):
        return self.condition


@_pytree_base
@dataclass(frozen=True)
class JaxMetricEvaluator:
    """Immutable PyTree state for a toroidal Fourier--Zernike map.

    Construct instances with :meth:`from_metric_evaluator` or
    :meth:`from_cache_payload`.  ``coefficients`` has shape
    ``(3, n_active_theta, n_active_eta, radial_degree + 1)`` and contains the
    already-fitted complex Fourier/Zernike coefficients for ``R``, ``Z``, and
    ``delta_phi``.  The mode tuples are static metadata; all numerical arrays
    are PyTree leaves and may be placed on a device or replicated across
    devices.
    """

    coefficients: Any
    theta_modes: Any
    eta_modes: Any
    period: Any
    theta0: Any
    eta0: Any
    radial_degree: int
    theta_mode_values: tuple[int, ...]
    eta_mode_values: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(
            self, "coefficients", jnp.asarray(self.coefficients, dtype=jnp.complex128)
        )
        object.__setattr__(
            self, "theta_modes", jnp.asarray(self.theta_modes, dtype=jnp.int64)
        )
        object.__setattr__(self, "eta_modes", jnp.asarray(self.eta_modes, dtype=jnp.int64))
        object.__setattr__(self, "period", jnp.asarray(self.period, dtype=jnp.float64))
        object.__setattr__(self, "theta0", jnp.asarray(self.theta0, dtype=jnp.float64))
        object.__setattr__(self, "eta0", jnp.asarray(self.eta0, dtype=jnp.float64))
        object.__setattr__(self, "radial_degree", int(self.radial_degree))
        object.__setattr__(
            self, "theta_mode_values", tuple(int(x) for x in self.theta_mode_values)
        )
        object.__setattr__(self, "eta_mode_values", tuple(int(x) for x in self.eta_mode_values))
        if self.coefficients.ndim != 4 or self.coefficients.shape[0] != 3:
            raise ValueError("coefficients must have shape (3, ntheta, neta, degree + 1)")
        if self.coefficients.shape[1] != len(self.theta_mode_values):
            raise ValueError("theta mode metadata does not match coefficients")
        if self.coefficients.shape[2] != len(self.eta_mode_values):
            raise ValueError("eta mode metadata does not match coefficients")
        if self.coefficients.shape[3] != self.radial_degree + 1:
            raise ValueError("coefficient radial degree does not match metadata")

    def tree_flatten(self):
        children = (
            self.coefficients,
            self.theta_modes,
            self.eta_modes,
            self.period,
            self.theta0,
            self.eta0,
        )
        aux = (self.radial_degree, self.theta_mode_values, self.eta_mode_values)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        radial_degree, theta_mode_values, eta_mode_values = aux_data
        # JAX lowering may use abstract ``ArgInfo`` placeholders.  Bypass the
        # numeric constructor so PyTree reconstruction remains structural.
        instance = object.__new__(cls)
        for name, value in zip(
            ("coefficients", "theta_modes", "eta_modes", "period", "theta0", "eta0"),
            children,
        ):
            object.__setattr__(instance, name, value)
        object.__setattr__(instance, "radial_degree", radial_degree)
        object.__setattr__(instance, "theta_mode_values", theta_mode_values)
        object.__setattr__(instance, "eta_mode_values", eta_mode_values)
        return instance

    @property
    def topology(self) -> str:
        return "toroidal"

    @property
    def radial_mode_degree(self) -> int:
        """Compatibility spelling for the retained Zernike degree."""
        return self.radial_degree

    @property
    def poloidal_modes(self) -> tuple[int, ...]:
        return self.theta_mode_values

    @property
    def toroidal_modes(self) -> tuple[int, ...]:
        return self.eta_mode_values

    @property
    def nfp(self) -> int | None:
        # The period is all that enters evaluation.  This property is retained
        # only as a convenience; payload conversion does not need nfp.
        period = float(np.asarray(self.period))
        ratio = 2.0 * np.pi / period
        rounded = int(round(ratio))
        return rounded if rounded >= 1 and np.isclose(ratio, rounded) else None

    @classmethod
    def from_metric_evaluator(cls, evaluator: Any) -> "JaxMetricEvaluator":
        """Create runtime state from an existing toroidal ``MetricEvaluator``."""
        if getattr(evaluator, "topology", None) != "toroidal":
            raise ValueError("JaxMetricEvaluator requires a toroidal MetricEvaluator")
        return cls.from_cache_payload(evaluator.to_cache_payload())

    @classmethod
    def from_cache_payload(
        cls, payload: Mapping[str, Any], *, prefix: str = ""
    ) -> "JaxMetricEvaluator":
        """Create runtime state from ``MetricEvaluator.to_cache_payload``."""

        def get(name: str):
            key = f"{prefix}{name}"
            try:
                return np.asarray(payload[key])
            except KeyError as error:
                raise ValueError(f"cached MetricEvaluator is missing {key!r}") from error

        if "topology_code" not in payload and f"{prefix}topology_code" not in payload:
            raise ValueError("JaxMetricEvaluator requires a toroidal cache payload")
        if int(get("representation_version").item()) != 1:
            raise ValueError("unsupported MetricEvaluator representation version")
        if int(get("topology_code").item()) != 1:
            raise ValueError("JaxMetricEvaluator requires a toroidal cache payload")
        degree = int(get("radial_degree").item())
        if degree < 2:
            raise ValueError("cached radial_degree must be at least two")
        u = get("u")
        theta = get("v")
        eta = get("eta")
        period = float(get("period").item())
        if (
            u.ndim != 1
            or theta.ndim != 1
            or eta.ndim != 1
            or u.size < 2
            or not np.isclose(u[0], 0.0)
            or not np.isclose(u[-1], 1.0)
            or not np.isfinite(period)
            or period <= 0.0
        ):
            raise ValueError("invalid toroidal coordinate axes or period")
        coefficients = np.asarray(get("zernike_coefficients"), dtype=np.complex128)
        if coefficients.ndim != 4 or coefficients.shape[0] != 3:
            raise ValueError("cached Zernike coefficients must have shape (3, ntheta, neta, degree + 1)")
        if coefficients.shape[1:] != (theta.size, eta.size, degree + 1):
            raise ValueError("cached Zernike coefficient shape is inconsistent with axes")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("cached Zernike coefficients are nonfinite")

        requested_theta = {int(x) for x in get("poloidal_modes").ravel()}
        requested_eta = {int(x) for x in get("toroidal_modes").ravel()}
        theta_fft = np.fft.fftfreq(theta.size, d=1.0 / theta.size).round().astype(int)
        eta_fft = np.fft.fftfreq(eta.size, d=1.0 / eta.size).round().astype(int)
        theta_indices = np.asarray(
            [i for i, mode in enumerate(theta_fft) if int(mode) in requested_theta],
            dtype=np.int64,
        )
        eta_indices = np.asarray(
            [i for i, mode in enumerate(eta_fft) if int(mode) in requested_eta],
            dtype=np.int64,
        )
        if theta_indices.size == 0 or eta_indices.size == 0:
            raise ValueError("cached toroidal mode sets must not be empty")
        # Indexing in this order reproduces _FourierZernikeChannel's active
        # mode loops exactly, including FFT negative-frequency ordering.
        active = np.stack(
            [coefficients[:, i, eta_indices, :] for i in theta_indices], axis=1
        )
        return cls(
            active,
            theta_fft[theta_indices],
            eta_fft[eta_indices],
            period,
            float(theta[0]),
            float(eta[0]),
            degree,
            tuple(int(x) for x in theta_fft[theta_indices]),
            tuple(int(x) for x in eta_fft[eta_indices]),
        )

    @staticmethod
    def _jacobi_sequence(x, count: int, alpha: int, beta: int):
        """Return ``P_0 .. P_count^(alpha,beta)(x)`` by fixed recurrence."""
        p0 = jnp.ones_like(x, dtype=jnp.float64)
        values = [p0]
        if count == 0:
            return jnp.stack(values, axis=-1)
        p1 = 0.5 * ((alpha - beta) + (alpha + beta + 2) * x)
        values.append(p1)
        pm1, p = p0, p1
        for n in range(1, count):
            nn = float(n)
            a1 = 2.0 * (nn + 1.0) * (nn + alpha + beta + 1.0) * (2.0 * nn + alpha + beta)
            a2 = (2.0 * nn + alpha + beta + 1.0) * (
                (2.0 * nn + alpha + beta + 2.0) * (2.0 * nn + alpha + beta) * x
                + alpha * alpha - beta * beta
            )
            a3 = 2.0 * (nn + alpha) * (nn + beta) * (2.0 * nn + alpha + beta + 2.0)
            pn = (a2 * p - a3 * pm1) / a1
            values.append(pn)
            pm1, p = p, pn
        return jnp.stack(values, axis=-1)

    def _basis_for_mode(self, u, mode: int, *, derivative: bool = False, theta_over_u: bool = False):
        degree = self.radial_degree
        a = abs(int(mode))
        x = 2.0 * u * u - 1.0
        orders = tuple(range(a, degree + 1, 2))
        sequence = self._jacobi_sequence(x, (degree - a) // 2, 0, a)
        if theta_over_u:
            if a == 0:
                return jnp.zeros((u.size, degree + 1), dtype=jnp.float64)
            # The only nonzero axis limit is m=+/-1.  This expression is
            # finite at u=0 and agrees with the analytic limit used by the
            # NumPy evaluator.
            if a == 1:
                values = self._jacobi_sequence(x, (degree - 1) // 2, 0, 1)
            else:
                values = (u ** (a - 1))[..., None] * sequence
            full = jnp.zeros((u.size, degree + 1), dtype=jnp.float64)
            return full.at[:, jnp.asarray(orders)].set(values)

        radial = (u ** a)[..., None] * sequence
        full = jnp.zeros((u.size, degree + 1), dtype=jnp.float64)
        full = full.at[:, jnp.asarray(orders)].set(radial)
        if not derivative:
            return full

        derivative_values = []
        derivative_sequence = (
            self._jacobi_sequence(x, (degree - a) // 2 - 1, 1, a + 1)
            if (degree - a) // 2 > 0
            else None
        )
        for column, k in enumerate(range((degree - a) // 2 + 1)):
            p = sequence[:, k]
            if a:
                first = float(a) * (u ** (a - 1)) * p
            else:
                first = jnp.zeros_like(u)
            if k:
                second = (u ** a) * (2.0 * u * (float(k) + a + 1.0)) * derivative_sequence[:, k - 1]
            else:
                second = jnp.zeros_like(u)
            derivative_values.append(first + second)
        derivative = jnp.stack(derivative_values, axis=-1)
        derivative_full = jnp.zeros((u.size, degree + 1), dtype=jnp.float64)
        return derivative_full.at[:, jnp.asarray(orders)].set(derivative)

    def _channel_values(
        self,
        u,
        theta,
        eta,
        *,
        derivative: bool = False,
        theta_over_u: bool = False,
        eta_derivative: bool = False,
    ):
        basis = jnp.stack(
            [self._basis_for_mode(u, mode, derivative=derivative, theta_over_u=theta_over_u)
             for mode in self.theta_mode_values], axis=0
        )
        theta_modes = jnp.asarray(self.theta_mode_values, dtype=jnp.float64)
        eta_modes = jnp.asarray(self.eta_mode_values, dtype=jnp.float64)
        phase_theta = jnp.exp(1j * (theta - self.theta0)[:, None] * theta_modes[None, :])
        phase_eta = jnp.exp(
            1j * (2.0 * jnp.pi / self.period) * (eta - self.eta0)[:, None] * eta_modes[None, :]
        )
        if theta_over_u:
            phase_theta = phase_theta * (1j * theta_modes)[None, :]
        if eta_derivative:
            phase_eta = phase_eta * (
                1j * (2.0 * jnp.pi / self.period) * eta_modes
            )[None, :]
        # Contract radial degree and theta mode before introducing the eta
        # phase.  Materializing (point, channel, theta, eta) would otherwise
        # dominate tracing memory (about 100 MiB for 2048 HSX points).
        theta_contracted = jnp.einsum(
            "qmd,cmnd,qm->qcn",
            jnp.moveaxis(basis, 0, 1),
            self.coefficients,
            phase_theta,
        )
        return jnp.einsum("qcn,qn->qc", theta_contracted, phase_eta).real

    def _values_and_derivatives(self, q):
        shape = q.shape[:-1]
        flat = q.reshape((-1, 3))
        u, theta, eta = flat[:, 0], flat[:, 1], flat[:, 2]
        basis = jnp.stack(
            [self._basis_for_mode(u, mode) for mode in self.theta_mode_values], axis=0
        )
        derivative_basis = jnp.stack(
            [
                self._basis_for_mode(u, mode, derivative=True)
                for mode in self.theta_mode_values
            ],
            axis=0,
        )
        basis = jnp.moveaxis(basis, 0, 1)
        derivative_basis = jnp.moveaxis(derivative_basis, 0, 1)
        tm = jnp.asarray(self.theta_mode_values, dtype=jnp.float64)
        en = jnp.asarray(self.eta_mode_values, dtype=jnp.float64)
        pt = jnp.exp(1j * (theta - self.theta0)[:, None] * tm[None, :])
        pe = jnp.exp(1j * (2.0 * jnp.pi / self.period) * (eta - self.eta0)[:, None] * en[None, :])
        theta_value = jnp.einsum(
            "qmd,cmnd,qm->qcn", basis, self.coefficients, pt
        )
        theta_du = jnp.einsum(
            "qmd,cmnd,qm->qcn", derivative_basis, self.coefficients, pt
        )
        theta_derivative = jnp.einsum(
            "qmd,cmnd,qm->qcn",
            basis,
            self.coefficients,
            pt * (1j * tm)[None, :],
        )
        values = jnp.einsum("qcn,qn->qc", theta_value, pe).real
        du = jnp.einsum("qcn,qn->qc", theta_du, pe).real
        dtheta = jnp.einsum("qcn,qn->qc", theta_derivative, pe).real
        deta = jnp.einsum(
            "qcn,qn->qc",
            theta_value,
            pe * (1j * (2.0 * jnp.pi / self.period) * en)[None, :],
        ).real
        return tuple(x.reshape(shape + (3,)) for x in (values, du, dtheta, deta))

    def channel_values_and_derivatives(self, logical_points: Any):
        """Return ``(R, Z, delta_phi)`` and its ``u, theta, eta`` derivatives.

        Each returned array has shape ``(..., 3)``.  This is useful to callers
        that need the fitted cylindrical channels without forming Cartesian
        metrics.
        """
        q = jnp.asarray(logical_points, dtype=jnp.float64)
        if q.ndim == 0 or q.shape[-1] != 3:
            raise ValueError("logical_points must have shape (..., 3)")
        return self._values_and_derivatives(q)

    def position_and_jacobian(self, logical_points: Any):
        """Return position and logical Jacobian with one spectral evaluation.

        Tracing needs both quantities at every RK stage.  Sharing the fitted
        channel evaluation here avoids doing the Fourier--Zernike contraction
        twice inside the compiled magnetic-field callback.
        """
        q = jnp.asarray(logical_points, dtype=jnp.float64)
        if q.ndim == 0 or q.shape[-1] != 3:
            raise ValueError("logical_points must have shape (..., 3)")
        values, du, dtheta, deta = self._values_and_derivatives(q)
        R, Z, delta = (values[..., i] for i in range(3))
        Ru, Zu, d_u = (du[..., i] for i in range(3))
        Rt, Zt, d_t = (dtheta[..., i] for i in range(3))
        Re, Ze, d_e = (deta[..., i] for i in range(3))
        phi = q[..., 2] + delta
        cp, sp = jnp.cos(phi), jnp.sin(phi)

        def cart(rd, pd, zd):
            return jnp.stack(
                (cp * rd - R * sp * pd, sp * rd + R * cp * pd, zd),
                axis=-1,
            )

        position = jnp.stack((R * cp, R * sp, Z), axis=-1)
        jacobian = jnp.stack(
            (
                cart(Ru, d_u, Zu),
                cart(Rt, d_t, Zt),
                cart(Re, 1.0 + d_e, Ze),
            ),
            axis=-1,
        )
        return position, jacobian

    def position(self, logical_points: Any):
        """Evaluate Cartesian position at arbitrary ``(..., 3)`` points."""
        q = jnp.asarray(logical_points, dtype=jnp.float64)
        if q.ndim == 0 or q.shape[-1] != 3:
            raise ValueError("logical_points must have shape (..., 3)")
        values, _, _, _ = self._values_and_derivatives(q)
        R, _, delta = (values[..., i] for i in range(3))
        phi = q[..., 2] + delta
        return jnp.stack(
            (R * jnp.cos(phi), R * jnp.sin(phi), values[..., 1]), axis=-1
        )

    def jacobian_matrix(self, logical_points: Any):
        """Evaluate the logical-to-Cartesian Jacobian, with columns ``u,theta,eta``."""
        q = jnp.asarray(logical_points, dtype=jnp.float64)
        if q.ndim == 0 or q.shape[-1] != 3:
            raise ValueError("logical_points must have shape (..., 3)")
        return self.position_and_jacobian(q)[1]

    def evaluate(self, logical_points: Any, *, reject_nonpositive_J: bool = True):
        """Evaluate position, Jacobian, determinant, and derived metric tensors."""
        q = jnp.asarray(logical_points, dtype=jnp.float64)
        if q.ndim == 0 or q.shape[-1] != 3:
            raise ValueError("logical_points must have shape (..., 3)")
        position, jacobian = self.position_and_jacobian(q)
        signed_J = jnp.linalg.det(jacobian)
        valid = jnp.isfinite(signed_J) & (signed_J > 0.0)
        if reject_nonpositive_J:
            try:
                if bool(np.any(~np.asarray(valid))):
                    raise ValueError("query contains nonpositive or nonfinite mesh Jacobian")
            except (TypeError, jax.errors.ConcretizationTypeError, jax.errors.TracerBoolConversionError):
                pass
        covariant = jnp.einsum("...ki,...kj->...ij", jacobian, jacobian)
        contravariant = jnp.linalg.inv(covariant)
        identity = jnp.eye(3, dtype=jnp.float64)
        residual = jnp.max(jnp.abs(jnp.einsum("...ik,...kj->...ij", covariant, contravariant) - identity), axis=(-2, -1))
        return JaxMetricEvaluation(position, jacobian, signed_J, covariant, contravariant, residual, valid)

    def evaluate_regularized(self, logical_points: Any):
        """Evaluate the finite frame ``[X_u, X_theta/u, period/(2*pi) X_eta]``."""
        q = jnp.asarray(logical_points, dtype=jnp.float64)
        if q.ndim == 0 or q.shape[-1] != 3:
            raise ValueError("logical_points must have shape (..., 3)")
        shape = q.shape[:-1]
        flat = q.reshape((-1, 3))
        u, theta, eta = flat[:, 0], flat[:, 1], flat[:, 2]
        values = self._channel_values(u, theta, eta)
        du = self._channel_values(u, theta, eta, derivative=True)
        dtheta_over_u = self._channel_values(u, theta, eta, theta_over_u=True)
        d_eta = self._channel_values(
            u, theta, eta, eta_derivative=True
        )
        values, du, dtheta_over_u, d_eta = (x.reshape(shape + (3,)) for x in (values, du, dtheta_over_u, d_eta))
        R, Z, delta = (values[..., i] for i in range(3))
        Ru, Zu, d_u = (du[..., i] for i in range(3))
        Rt, Zt, d_t = (dtheta_over_u[..., i] for i in range(3))
        Re, Ze, d_e = (d_eta[..., i] for i in range(3))
        phi = q[..., 2] + delta
        cp, sp = jnp.cos(phi), jnp.sin(phi)

        def cart(rd, pd, zd):
            return jnp.stack((cp * rd - R * sp * pd, sp * rd + R * cp * pd, zd), axis=-1)

        eta_scale = self.period / (2.0 * jnp.pi)
        frame = jnp.stack((cart(Ru, d_u, Zu), cart(Rt, d_t, Zt), eta_scale * cart(Re, 1.0 + d_e, Ze)), axis=-1)
        J = jnp.linalg.det(frame)
        cov = jnp.einsum("...ki,...kj->...ij", frame, frame)
        contra = jnp.linalg.inv(cov)
        identity = jnp.eye(3, dtype=jnp.float64)
        residual = jnp.max(jnp.abs(jnp.einsum("...ik,...kj->...ij", cov, contra) - identity), axis=(-2, -1))
        condition = jnp.linalg.cond(frame)
        valid = jnp.isfinite(J) & (J > 0.0) & jnp.isfinite(condition)
        position = jnp.stack((R * cp, R * sp, Z), axis=-1)
        return JaxRegularizedMetricEvaluation(
            position, frame, J, cov, contra, condition, residual, valid
        )


# Explicit aliases make the runtime role discoverable without forcing callers
# to depend on one spelling of the class name.
JaxToroidalMetricEvaluator = JaxMetricEvaluator
ToroidalJaxMetricEvaluator = JaxMetricEvaluator


def jax_metric_evaluator_from_metric_evaluator(evaluator: Any) -> JaxMetricEvaluator:
    """Functional constructor matching :meth:`JaxMetricEvaluator.from_metric_evaluator`."""
    return JaxMetricEvaluator.from_metric_evaluator(evaluator)


def jax_metric_evaluator_from_cache_payload(
    payload: Mapping[str, Any], *, prefix: str = ""
) -> JaxMetricEvaluator:
    """Functional constructor for a serialized toroidal metric payload."""
    return JaxMetricEvaluator.from_cache_payload(payload, prefix=prefix)


# Short functional spelling for callers that prefer constructor-style APIs.
from_metric_evaluator = jax_metric_evaluator_from_metric_evaluator
from_cache_payload = jax_metric_evaluator_from_cache_payload


__all__ = [
    "JaxMetricEvaluation",
    "JaxRegularizedMetricEvaluation",
    "JaxMetricEvaluator",
    "JaxToroidalMetricEvaluator",
    "ToroidalJaxMetricEvaluator",
    "jax_metric_evaluator_from_metric_evaluator",
    "jax_metric_evaluator_from_cache_payload",
    "from_metric_evaluator",
    "from_cache_payload",
]
