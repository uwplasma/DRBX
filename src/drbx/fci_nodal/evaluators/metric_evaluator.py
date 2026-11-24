"""Smooth representations of structured ``D^2 x S^1`` meshes.

The square topology uses tensor-product splines on ``[0, 1]^2`` and Fourier
modes in eta.  The toroidal topology uses Fourier--Zernike coordinates
``(u, theta, eta)`` with an analytically regularized collapsed axis at
``u=0``.  :func:`build_metric_evaluator` is the single construction entry
point for both representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import null_space
from scipy.special import eval_jacobi

from .._metric_contract import FILTERED_MAP_CONSTRUCTION, FILTERED_MAP_CUTOFF_KEY

_METRIC_EVALUATOR_CACHE_FORMAT_VERSION = 1
#: Required payload tag for the exact metric.
PLANE_CONSTRUCTION_KEY = "plane_construction"
EXACT_PLANE_CONSTRUCTION = "exact_level_set"


def require_exact_plane_payload(payload: Any, prefix: str = "") -> None:
    """Check the explicit exact-plane tag without reading coefficient arrays."""
    key = f"{prefix}{PLANE_CONSTRUCTION_KEY}"
    if key not in payload or str(np.asarray(payload[key]).item()) != EXACT_PLANE_CONSTRUCTION:
        raise ValueError("metric payload must describe exact level-set planes")


def filtered_map_cutoff(payload: Any, prefix: str = "") -> int | None:
    """Read the tagged construction and check it against its Fourier modes."""
    tag = np.asarray(payload.get(f"{prefix}{PLANE_CONSTRUCTION_KEY}", "")).item()
    if tag != FILTERED_MAP_CONSTRUCTION:
        return None
    cutoff = np.asarray(payload.get(f"{prefix}{FILTERED_MAP_CUTOFF_KEY}", -1)).item()
    modes = np.asarray(payload.get(f"{prefix}toroidal_modes", ()))
    if (type(cutoff) is not int or cutoff < 0 or modes.size != 2 * cutoff + 1
            or not np.array_equal(modes, np.arange(-cutoff, cutoff + 1))):
        raise ValueError("eta-filtered metric cutoff does not match its toroidal modes")
    return cutoff


def require_metric_construction(payload, options, prefix="metric_evaluator_"):
    """Bind a metric archive to the construction requested by its case."""
    wanted = options.filtered_map_identity()
    if filtered_map_cutoff(payload, prefix) != wanted.get("metric_toroidal_modes"):
        raise ValueError("metric construction or cutoff differs from options")
    if wanted:
        if np.asarray(payload.get(f"{prefix}nfp", -1)).item() != wanted["map_filter_nfp"]:
            raise ValueError("eta-filtered metric periods differ from options")
    else:
        from .exact_plane_metric import require_exact_metric_payload
        require_exact_metric_payload(payload, prefix)


@dataclass(frozen=True)
class MetricEvaluation:
    """Metric quantities evaluated at logical points."""

    position: np.ndarray
    jacobian_matrix: np.ndarray
    signed_J: np.ndarray
    covariant_metric: np.ndarray
    contravariant_metric: np.ndarray
    inverse_residual: np.ndarray
    valid: np.ndarray

    @property
    def J(self) -> np.ndarray:
        return self.signed_J

    @property
    def g_cov(self) -> np.ndarray:
        return self.covariant_metric

    @property
    def g_contra(self) -> np.ndarray:
        return self.contravariant_metric


@dataclass(frozen=True)
class RegularizedMetricEvaluation:
    """The finite limiting metric frame at a toroidal coordinate axis."""

    position: np.ndarray
    regularized_jacobian_matrix: np.ndarray
    regularized_J: np.ndarray
    covariant_metric: np.ndarray
    contravariant_metric: np.ndarray
    condition: np.ndarray
    inverse_residual: np.ndarray
    valid: np.ndarray

    @property
    def J(self) -> np.ndarray:
        return self.regularized_J

    @property
    def jacobian_matrix(self) -> np.ndarray:
        return self.regularized_jacobian_matrix

    @property
    def J_reg(self) -> np.ndarray:
        return self.regularized_J


    @property
    def g_cov(self) -> np.ndarray:
        return self.covariant_metric

    @property
    def g_contra(self) -> np.ndarray:
        return self.contravariant_metric


@dataclass(frozen=True)
class MagneticFieldEvaluation:
    """Magnetic field and its covariant/contravariant components."""

    B_cartesian: np.ndarray
    B_contravariant: np.ndarray
    magnitude: np.ndarray

    @property
    def B(self) -> np.ndarray:
        return self.B_cartesian


class _FourierZernikeChannel:
    """Fourier in ``(theta, eta)`` and parity-regular Zernike in ``u``."""

    def __init__(self, u, theta, eta, samples, theta0, eta0, period,
                 radial_degree=None, poloidal_modes=None, toroidal_modes=None):
        self._u = np.asarray(u, dtype=float)
        self._theta = np.asarray(theta, dtype=float)
        self._eta = np.asarray(eta, dtype=float)
        self._theta0 = float(theta0)
        self._eta0 = float(eta0)
        self._period = float(period)
        self._radial_degree = int(radial_degree if radial_degree is not None else max(3, self._u.size - 1))
        if self._radial_degree < 2:
            raise ValueError("radial_degree must be at least two")
        if self._radial_degree // 2 + 1 > self._u.size:
            raise ValueError(
                "radial_degree has more m=0 Zernike coefficients than radial samples"
            )
        self._poloidal_modes = self._mode_selection(
            poloidal_modes, self._theta.size, default_limit=self._radial_degree
        )
        self._toroidal_modes = self._mode_selection(toroidal_modes, self._eta.size)
        if any(abs(m) > self._radial_degree for m in self._poloidal_modes):
            raise ValueError("poloidal mode exceeds radial_degree and has no admissible Zernike mode")
        if not frozenset((-1, 0, 1)).issubset(self._poloidal_modes):
            raise ValueError(
                "toroidal topology requires poloidal modes m=0 and m=+/-1"
            )
        if 0 not in self._toroidal_modes:
            raise ValueError("toroidal topology requires toroidal mode n=0")
        self._modes_theta = np.fft.fftfreq(self._theta.size, d=1.0 / self._theta.size)
        self._modes_eta = np.fft.fftfreq(self._eta.size, d=1.0 / self._eta.size)
        self._active_theta_indices = np.asarray(
            [
                index
                for index, mode in enumerate(self._modes_theta)
                if int(round(mode)) in self._poloidal_modes
            ],
            dtype=int,
        )
        self._active_eta_indices = np.asarray(
            [
                index
                for index, mode in enumerate(self._modes_eta)
                if int(round(mode)) in self._toroidal_modes
            ],
            dtype=int,
        )
        self._active_modes_theta = self._modes_theta[self._active_theta_indices]
        self._active_modes_eta = self._modes_eta[self._active_eta_indices]
        coeff = np.fft.fftn(np.asarray(samples, dtype=float), axes=(1, 2))
        coeff /= self._theta.size * self._eta.size
        self._coefficients = np.asarray(coeff, dtype=complex)
        # Kept for cache-payload compatibility and diagnostics.  Query-sized
        # basis arrays are deliberately *not* retained here: real HSX callers
        # issue many distinct quadrature queries, and retaining full query
        # bytes plus every radial matrix makes process memory grow without a
        # bound.  Evaluation-time reuse lives in the query-scoped ``prepared``
        # mapping shared by all three coordinate channels instead.
        self._basis = {}
        self._fit_coefficients = {}
        for im in self._active_theta_indices:
            m0 = self._modes_theta[im]
            m = int(round(m0))
            orders = list(range(abs(m), self._radial_degree + 1, 2))
            B = self._basis_matrix(m, self._u)
            coefficient_matrix = np.zeros(
                (self._active_eta_indices.size, len(orders)), dtype=complex
            )
            for local_ine, ine in enumerate(self._active_eta_indices):
                target = self._coefficients[:, im, ine]
                constraint_rows = [B[-1]]
                constraint_values = [target[-1]]
                if m == 0:
                    constraint_rows.append(B[0])
                    constraint_values.append(target[0])
                C = np.asarray(constraint_rows, dtype=complex)
                d = np.asarray(constraint_values, dtype=complex)
                particular = np.linalg.lstsq(C, d, rcond=None)[0]
                Z = null_space(np.asarray(C.real, dtype=float))
                if C.shape[1] and Z.shape[1]:
                    # Constraints are real, so use the same null space for
                    # real and imaginary Fourier coefficient parts.
                    real = particular.real + Z @ np.linalg.lstsq(B @ Z, target.real - B @ particular.real, rcond=None)[0]
                    imag = particular.imag + Z @ np.linalg.lstsq(B @ Z, target.imag - B @ particular.imag, rcond=None)[0]
                    coeff_mode = real + 1j * imag
                else:
                    coeff_mode = particular
                coefficient_matrix[local_ine] = coeff_mode
            self._fit_coefficients[im] = coefficient_matrix

    @staticmethod
    def _mode_selection(value, count, *, default_limit=None):
        modes = np.fft.fftfreq(count, d=1.0 / count).round().astype(int)
        if value is None:
            return frozenset(
                int(x) for x in modes
                if default_limit is None or abs(int(x)) <= int(default_limit)
            )
        if np.isscalar(value):
            if isinstance(value, (bool, np.bool_)) or int(value) != value:
                raise ValueError("mode cutoff must be a nonnegative integer")
            limit = int(value)
            if limit < 0 or limit > int(np.max(np.abs(modes))):
                raise ValueError("mode cutoff must be nonnegative")
            return frozenset(int(x) for x in modes if abs(int(x)) <= limit)
        raw = tuple(value)
        if any(isinstance(x, (bool, np.bool_)) or int(x) != x for x in raw):
            raise ValueError("requested Fourier modes must be integers")
        selected = frozenset(int(x) for x in raw)
        if not selected.issubset(set(int(x) for x in modes)):
            raise ValueError("requested Fourier mode is not present on the sampled axis")
        nyquist = -count // 2 if count % 2 == 0 else None
        if any(
            mode != 0
            and mode != nyquist
            and -mode not in selected
            for mode in selected
        ):
            raise ValueError("explicit Fourier mode sets must be conjugate symmetric")
        return selected

    def _basis_matrix(self, m, u):
        query = np.asarray(u, dtype=float)
        orders = list(range(abs(m), self._radial_degree + 1, 2))
        uu = query
        out = np.empty((uu.size, len(orders)), dtype=float)
        for j, ell in enumerate(orders):
            out[:, j] = uu ** abs(m) * eval_jacobi((ell - abs(m)) // 2, 0, abs(m), 2 * uu * uu - 1)
        return out

    def _basis_and_derivative(self, m, u):
        uu = np.asarray(u, dtype=float)
        orders = list(range(abs(m), self._radial_degree + 1, 2))
        value = np.empty((uu.size, len(orders)), dtype=float)
        deriv = np.empty_like(value)
        a = abs(m)
        x = 2 * uu * uu - 1
        for j, ell in enumerate(orders):
            k = (ell - a) // 2
            p = eval_jacobi(k, 0, a, x)
            value[:, j] = uu ** a * p
            deriv[:, j] = 0.0
            if a:
                deriv[:, j] += a * uu ** (a - 1) * p
            if k:
                deriv[:, j] += uu ** a * (2 * uu * (k + a + 1) * eval_jacobi(k - 1, 1, a + 1, x))
        return value, deriv

    def prepare(self, u, theta, eta):
        uf = np.asarray(u, dtype=float).reshape(-1)
        tf = np.asarray(theta, dtype=float).reshape(-1)
        ef = np.asarray(eta, dtype=float).reshape(-1)
        th = 2 * np.pi * (tf - self._theta0) / (2 * np.pi)
        ep = 2 * np.pi * (ef - self._eta0) / self._period
        return {"u": uf, "theta": tf, "eta": ef,
                "phase_theta": np.exp(1j * np.outer(th, self._active_modes_theta)),
                "phase_eta": np.exp(1j * np.outer(ep, self._active_modes_eta)),
                "basis": {}}

    def _radial_values(self, prepared, m, du=0, theta_over_u=False):
        u = prepared["u"]
        matching = np.flatnonzero(np.rint(self._modes_theta).astype(int) == m)
        if matching.size != 1 or int(matching[0]) not in self._fit_coefficients:
            return np.zeros((u.size, self._active_eta_indices.size), complex)
        im = int(matching[0])
        cache = prepared["basis"]
        cache_key = (self._radial_degree, int(m), int(du), bool(theta_over_u))
        if cache_key not in cache:
            B = self._basis_matrix(m, u)
            if theta_over_u:
                if m == 0:
                    B = np.zeros_like(B)
                else:
                    B = B.copy()
                    mask = u != 0
                    if np.any(mask):
                        B[mask] /= u[mask, None]
                    if np.any(~mask):
                        if abs(m) == 1:
                            orders = list(range(1, self._radial_degree + 1, 2))
                            B[~mask] = np.asarray([
                                eval_jacobi((ell - 1) // 2, 0, 1, -1.0)
                                for ell in orders
                            ])[None, :]
                        else:
                            B[~mask] = 0.0
            elif du:
                _, B = self._basis_and_derivative(m, u)
            cache[cache_key] = B
        B = cache[cache_key]
        return B @ self._fit_coefficients[im].T

    def evaluate_prepared(self, prepared, du=0, dv=0, deta=0, theta_over_u=False):
        values = np.zeros(prepared["u"].size, complex)
        for local_im, im in enumerate(self._active_theta_indices):
            m0 = self._modes_theta[im]
            m = int(round(m0))
            radial = self._radial_values(prepared, m, du=du, theta_over_u=theta_over_u)
            factor = (1j * m) if (theta_over_u or dv) else 1.0
            if deta:
                radial = radial * (
                    1j * 2 * np.pi * self._active_modes_eta / self._period
                ) ** deta
            values += (
                np.einsum(
                    "nq,nq->n", radial, prepared["phase_eta"], optimize=True
                )
                * prepared["phase_theta"][:, local_im]
                * factor
            )
        return values.real

    def evaluate(self, u, theta, eta, du=0, dv=0, deta=0):
        shape = np.asarray(theta).shape
        return self.evaluate_prepared(self.prepare(u, theta, eta), du, dv, deta).reshape(shape)

    def coefficient_array(self) -> np.ndarray:
        """Return radial coefficients padded by Zernike degree."""
        result = np.zeros(
            (
                self._theta.size,
                self._eta.size,
                self._radial_degree + 1,
            ),
            dtype=np.complex128,
        )
        for im, coefficients in self._fit_coefficients.items():
            m = int(round(self._modes_theta[im]))
            orders = np.arange(abs(m), self._radial_degree + 1, 2)
            for local_ine, ine in enumerate(self._active_eta_indices):
                result[im, ine, orders] = coefficients[local_ine]
        return result


class MetricEvaluator:
    """Evaluate a smooth ``D^2 x S^1`` mesh embedding and its metrics.

    The toroidal chart uses ``(u, theta, eta)`` with an analytically regularized axis at ``u=0``.

    Parameters
    ----------
    u, v:
        Strictly increasing non-periodic axes, normally spanning ``[0, 1]``.
    eta:
        Strictly increasing, uniformly spaced, endpoint-exclusive toroidal
        axis.  It spans exactly one ``period``.
    positions:
        Cartesian node positions with shape ``(nu, nv, neta, 3)``.
    period, nfp:
        Supply one of these.  ``period`` is in radians; ``nfp`` implies
        ``period = 2*pi/nfp``.
    radial_degree, poloidal_modes, toroidal_modes:
        Retained Fourier–Zernike mode ranges.
    """

    def __init__(
        self,
        u: Any,
        v: Any,
        eta: Any,
        positions: Any,
        *,
        period: float | None = None,
        nfp: int | None = None,
        mmpde_fit_scale: float = 1.0,
        metric_spline_degree: int = 3,
        topology: str = "square",
        radial_degree: int | None = None,
        poloidal_modes: Any = None,
        toroidal_modes: Any = None,
    ) -> None:
        topology = str(topology).lower()
        if topology != "toroidal":
            raise ValueError("Only toroidal topology is supported")
        self._topology = topology
        self._u = _axis(u, "u")
        self._v = _axis(v, "v")
        self._eta = _axis(eta, "eta")
        metric_spline_degree = _validate_metric_spline_degree(metric_spline_degree)
        if self._u.size < 2 or self._v.size < 2 or self._eta.size < 4:
            raise ValueError("u and v need at least 2 samples; eta needs at least 4")
        if not np.isclose(self._u[0], 0.0) or not np.isclose(self._u[-1], 1.0):
            raise ValueError("u must span [0, 1]")
        dtheta = np.diff(self._v)
        if self._v[0] < -1e-12 or not np.allclose(dtheta, dtheta[0], rtol=2e-10, atol=2e-12):
            raise ValueError("toroidal theta must be uniformly spaced")
        if not np.isclose(dtheta[0] * self._v.size, 2 * np.pi, rtol=2e-9, atol=2e-11):
            raise ValueError("toroidal theta must span one endpoint-exclusive 2pi period")
        if period is None and nfp is None:
            raise ValueError("supply period or nfp")
        if period is not None and (not np.isfinite(period) or period <= 0):
            raise ValueError("period must be positive and finite")
        if nfp is not None and (int(nfp) != nfp or int(nfp) < 1):
            raise ValueError("nfp must be a positive integer")
        implied_period = 2.0 * np.pi / int(nfp) if nfp is not None else None
        if period is None:
            period = implied_period
        elif implied_period is not None and not np.isclose(period, implied_period, rtol=2e-12, atol=2e-12):
            raise ValueError("period and nfp are inconsistent")
        self._period = float(period)
        self._nfp = None if nfp is None else int(nfp)
        self._metric_spline_degree = metric_spline_degree
        self._mmpde_result = None
        self._mmpde_fit_scale = float(mmpde_fit_scale)
        deta = np.diff(self._eta)
        if not np.allclose(deta, deta[0], rtol=2e-10, atol=2e-12):
            raise ValueError("eta must be uniformly spaced")
        if not np.isclose(deta[0] * self._eta.size, self._period, rtol=2e-9, atol=2e-11):
            raise ValueError("eta must span exactly one endpoint-exclusive period")
        values = np.asarray(positions, dtype=np.float64)
        expected = (self._u.size, self._v.size, self._eta.size, 3)
        if values.shape != expected:
            raise ValueError(f"positions must have shape {expected}")
        if not np.all(np.isfinite(values)):
            raise ValueError("positions must be finite")
        radius = np.hypot(values[..., 0], values[..., 1])
        if np.any(radius <= 0):
            raise ValueError("all positions must have R > 0")

        if radial_degree is not None and (
            isinstance(radial_degree, (bool, np.bool_))
            or int(radial_degree) != radial_degree
        ):
            raise ValueError("radial_degree must be an integer")
        self._radial_degree = None if radial_degree is None else int(radial_degree)
        self._poloidal_modes = poloidal_modes
        self._toroidal_modes = toroidal_modes
        if topology == "toroidal":
            scale = max(1.0, float(np.max(np.abs(values))))
            if np.max(np.ptp(values[0, :, :, :], axis=0)) > 5e-9 * scale:
                raise ValueError("toroidal axis positions must be independent of theta")

        phi = np.unwrap(np.arctan2(values[..., 1], values[..., 0]), axis=2)
        delta_phi = phi - self._eta[None, None, :]
        self._channels = tuple(
            _FourierZernikeChannel(
                self._u, self._v, self._eta, samples, self._v[0], self._eta[0],
                self._period, radial_degree, poloidal_modes, toroidal_modes,
            ) for samples in (radius, values[..., 2], delta_phi)
        )
        self._radial_degree = self._channels[0]._radial_degree
        self._poloidal_modes = tuple(
            sorted(self._channels[0]._poloidal_modes)
        )
        self._toroidal_modes = tuple(
            sorted(self._channels[0]._toroidal_modes)
        )
        axis_points = np.stack(
            np.meshgrid(
                [0.0],
                self._v,
                self._eta + 0.37 * self._period / self._eta.size,
                indexing="ij",
            ),
            axis=-1,
        )
        try:
            axis_metric = self.evaluate_regularized(axis_points)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "toroidal m=+/-1 modes produce a degenerate axis frame"
            ) from error
        if np.any(~axis_metric.valid):
            raise ValueError(
                "toroidal m=+/-1 modes must produce a positive, "
                "nondegenerate axis frame"
            )

    @property
    def period(self) -> float:
        return self._period

    @property
    def topology(self) -> str:
        return self._topology

    @property
    def theta(self) -> np.ndarray:
        if self._topology != "toroidal":
            raise AttributeError("theta is only defined for toroidal topology")
        return self._v.copy()

    @property
    def radial_degree(self) -> int | None:
        return self._radial_degree

    @property
    def poloidal_modes(self):
        return self._poloidal_modes

    @property
    def toroidal_modes(self):
        return self._toroidal_modes

    @property
    def nfp(self) -> int | None:
        return self._nfp

    @property
    def metric_spline_degree(self) -> int:
        return self._metric_spline_degree

    @property
    def u(self) -> np.ndarray:
        return self._u.copy()

    @property
    def v(self) -> np.ndarray:
        return self._v.copy()

    @property
    def eta(self) -> np.ndarray:
        return self._eta.copy()


    _filtered_map_cutoff = None

    def to_cache_payload(self, *, prefix: str = "") -> dict[str, np.ndarray]:
        """Serialize the fitted Fourier/spline representation as numeric arrays.

        Unlike caching sampled positions, this preserves the already-fitted
        channels and lets :meth:`from_cache_payload` restore the evaluator
        without invoking FITPACK again.  The payload contains no pickled
        Python objects and is safe to store in an ``allow_pickle=False`` NPZ.
        """

        key = lambda name: f"{prefix}{name}"
        channels = self._channels
        payload = {
            key("representation_version"): np.asarray(
                _METRIC_EVALUATOR_CACHE_FORMAT_VERSION, dtype=np.int64
            ),
            key("topology_code"): np.asarray(1, dtype=np.int64),
            key("u"): self._u.copy(),
            key("v"): self._v.copy(),
            key("eta"): self._eta.copy(),
            key("period"): np.asarray(self._period, dtype=np.float64),
            key("nfp"): np.asarray(
                -1 if self._nfp is None else self._nfp, dtype=np.int64
            ),
            key("metric_spline_degree"): np.asarray(
                self._metric_spline_degree, dtype=np.int64
            ),
            key("mmpde_fit_scale"): np.asarray(
                self._mmpde_fit_scale, dtype=np.float64
            ),
            key("radial_degree"): np.asarray(
                channels[0]._radial_degree, dtype=np.int64
            ),
            key("poloidal_modes"): np.asarray(
                sorted(channels[0]._poloidal_modes), dtype=np.int64
            ),
            key("toroidal_modes"): np.asarray(
                sorted(channels[0]._toroidal_modes), dtype=np.int64
            ),
            key("zernike_coefficients"): np.stack(
                [channel.coefficient_array() for channel in channels], axis=0
            ),
        }
        if self._filtered_map_cutoff is not None:
            payload[key(PLANE_CONSTRUCTION_KEY)] = np.asarray(FILTERED_MAP_CONSTRUCTION)
            payload[key(FILTERED_MAP_CUTOFF_KEY)] = np.asarray(self._filtered_map_cutoff)
        return payload

    @classmethod
    def from_cache_payload(
        cls,
        payload: Any,
        *,
        prefix: str = "",
    ) -> "MetricEvaluator":
        """Restore a tagged exact-plane or eta-filtered metric."""
        cutoff = filtered_map_cutoff(payload, prefix)
        if cutoff is not None:
            evaluator = cls._from_plain_cache_payload(payload, prefix=prefix)
            evaluator._filtered_map_cutoff = cutoff
            return evaluator
        from .exact_plane_metric import ExactPlaneMetricEvaluator

        return ExactPlaneMetricEvaluator.from_cache_payload(payload, prefix=prefix)

    @classmethod
    def _from_plain_cache_payload(
        cls,
        payload: Any,
        *,
        prefix: str = "",
    ) -> "MetricEvaluator":
        """Restore the private Fourier–Zernike base."""

        key = lambda name: f"{prefix}{name}"

        def array(name: str) -> np.ndarray:
            try:
                return np.asarray(payload[key(name)])
            except KeyError as error:
                raise ValueError(
                    f"cached MetricEvaluator is missing {key(name)!r}"
                ) from error

        version = int(array("representation_version").item())
        if version != _METRIC_EVALUATOR_CACHE_FORMAT_VERSION:
            raise ValueError(
                "cached MetricEvaluator representation version mismatch: "
                f"{version} != {_METRIC_EVALUATOR_CACHE_FORMAT_VERSION}"
            )
        u = _axis(array("u"), "cached u")
        v = _axis(array("v"), "cached v")
        eta = _axis(array("eta"), "cached eta")
        if int(array("topology_code").item()) != 1:
            raise ValueError("cached MetricEvaluator requires toroidal topology")
        if not np.isclose(u[0], 0.0) or not np.isclose(u[-1], 1.0):
            raise ValueError("cached u must span [0, 1]")
        dtheta = np.diff(v)
        if not np.allclose(dtheta, dtheta[0], rtol=2e-10, atol=2e-12):
            raise ValueError("cached theta must be uniformly spaced")
        if not np.isclose(
            dtheta[0] * v.size, 2.0 * np.pi, rtol=2e-9, atol=2e-11
        ):
            raise ValueError("cached theta must span an endpoint-exclusive period")
        period = float(array("period").item())
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("cached period must be positive and finite")
        deta = np.diff(eta)
        if eta.size < 4 or not np.allclose(
            deta,
            deta[0],
            rtol=2.0e-10,
            atol=2.0e-12,
        ):
            raise ValueError("cached eta must be uniformly spaced")
        if not np.isclose(
            deta[0] * eta.size,
            period,
            rtol=2.0e-9,
            atol=2.0e-11,
        ):
            raise ValueError("cached eta does not span its period")

        stored_nfp = int(array("nfp").item())
        nfp = None if stored_nfp == -1 else stored_nfp
        if nfp is not None and (
            nfp < 1
            or not np.isclose(
                period,
                2.0 * np.pi / nfp,
                rtol=2.0e-12,
                atol=2.0e-12,
            )
        ):
            raise ValueError("cached nfp is invalid or inconsistent with period")
        metric_spline_degree = _validate_metric_spline_degree(
            int(array("metric_spline_degree").item())
        )
        radial_degree = int(array("radial_degree").item())
        if radial_degree < 2 or radial_degree // 2 + 1 > u.size:
            raise ValueError("cached radial_degree is incompatible with radial samples")
        poloidal_modes = frozenset(
            int(value) for value in np.asarray(array("poloidal_modes")).ravel()
        )
        toroidal_modes = frozenset(
            int(value) for value in np.asarray(array("toroidal_modes")).ravel()
        )
        mode_theta = np.fft.fftfreq(v.size, d=1.0 / v.size)
        mode_eta = np.fft.fftfreq(eta.size, d=1.0 / eta.size)
        available_theta = frozenset(int(round(value)) for value in mode_theta)
        available_eta = frozenset(int(round(value)) for value in mode_eta)
        if (
            not poloidal_modes.issubset(available_theta)
            or not toroidal_modes.issubset(available_eta)
            or any(abs(mode) > radial_degree for mode in poloidal_modes)
            or not frozenset((-1, 0, 1)).issubset(poloidal_modes)
            or 0 not in toroidal_modes
        ):
            raise ValueError("cached Fourier-Zernike mode set is invalid")
        coefficients = np.asarray(
            array("zernike_coefficients"), dtype=np.complex128
        )
        expected = (3, v.size, eta.size, radial_degree + 1)
        if coefficients.shape != expected or not np.all(np.isfinite(coefficients)):
            raise ValueError(
                "cached Zernike coefficients have an invalid shape or values"
            )
        evaluator = cls.__new__(cls)
        evaluator._u = u
        evaluator._v = v
        evaluator._eta = eta
        evaluator._topology = "toroidal"
        evaluator._period = period
        evaluator._nfp = nfp
        evaluator._metric_spline_degree = metric_spline_degree
        evaluator._radial_degree = radial_degree
        evaluator._poloidal_modes = tuple(sorted(poloidal_modes))
        evaluator._toroidal_modes = tuple(sorted(toroidal_modes))
        evaluator._mmpde_result = None
        evaluator._mmpde_fit_scale = float(array("mmpde_fit_scale").item())
        channels = []
        for channel_index in range(3):
            channel = _FourierZernikeChannel.__new__(
                _FourierZernikeChannel
            )
            channel._u = u
            channel._theta = v
            channel._eta = eta
            channel._theta0 = float(v[0])
            channel._eta0 = float(eta[0])
            channel._period = period
            channel._radial_degree = radial_degree
            channel._poloidal_modes = poloidal_modes
            channel._toroidal_modes = toroidal_modes
            channel._modes_theta = mode_theta
            channel._modes_eta = mode_eta
            channel._active_theta_indices = np.asarray(
                [
                    index
                    for index, mode in enumerate(mode_theta)
                    if int(round(mode)) in poloidal_modes
                ],
                dtype=int,
            )
            channel._active_eta_indices = np.asarray(
                [
                    index
                    for index, mode in enumerate(mode_eta)
                    if int(round(mode)) in toroidal_modes
                ],
                dtype=int,
            )
            channel._active_modes_theta = mode_theta[
                channel._active_theta_indices
            ]
            channel._active_modes_eta = mode_eta[
                channel._active_eta_indices
            ]
            channel._basis = {}
            channel._coefficients = None
            channel._fit_coefficients = {}
            for im in channel._active_theta_indices:
                mode = int(round(mode_theta[im]))
                orders = np.arange(abs(mode), radial_degree + 1, 2)
                channel._fit_coefficients[int(im)] = coefficients[
                    channel_index, int(im), channel._active_eta_indices
                ][:, orders]
            channels.append(channel)
        evaluator._channels = tuple(channels)
        return evaluator

    def _channels_at(
        self,
        points: Any,
        derivative_orders: tuple[tuple[int, int, int], ...] = (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ),
    ) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
        q = np.asarray(points, dtype=np.float64)
        if q.ndim == 0 or q.shape[-1] != 3:
            raise ValueError("logical_points must have shape (..., 3)")
        if not np.all(np.isfinite(q)):
            raise ValueError("logical points must be finite")
        shape = q.shape[:-1]
        uf, vf, ef = (q[..., i].reshape(-1) for i in range(3))
        if np.any((uf < self._u[0]) | (uf > self._u[-1])):
            raise ValueError("u queries must lie in [0, 1]")
        prepared = self._channels[0].prepare(uf, vf, ef)
        return tuple(
            tuple(
                channel.evaluate_prepared(
                    prepared, du=du, dv=dv, deta=deta
                ).reshape(shape)
                for channel in self._channels
            )
            for du, dv, deta in derivative_orders
        ), shape

    def _position_only(self, logical_points: Any) -> np.ndarray:
        q = np.asarray(logical_points, dtype=np.float64)
        channels, shape = self._channels_at(q, ((0, 0, 0),))
        R, Z, delta = channels[0]
        phi = q[..., 2] + delta
        return np.stack((R * np.cos(phi), R * np.sin(phi), Z), axis=-1).reshape(
            shape + (3,)
        )

    def _position_and_jacobian(self, logical_points: Any) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(logical_points, dtype=np.float64)
        channels, shape = self._channels_at(q)
        (R, Z, delta), (Ru, Zu, du), (Rv, Zv, dv), (Re, Ze, de) = channels
        phi = q[..., 2] + delta
        cp, sp = np.cos(phi), np.sin(phi)
        pos = np.stack((R * cp, R * sp, Z), axis=-1)
        phi_derivatives = (du, dv, 1.0 + de)
        radial_derivatives = (Ru, Rv, Re)
        vertical_derivatives = (Zu, Zv, Ze)
        columns = []
        for rd, pd, zd in zip(radial_derivatives, phi_derivatives, vertical_derivatives):
            columns.append(np.stack((cp * rd - R * sp * pd, sp * rd + R * cp * pd, zd), axis=-1))
        A = np.stack(columns, axis=-1)
        return pos.reshape(shape + (3,)), A.reshape(shape + (3, 3))

    def position(self, logical_points: Any) -> np.ndarray:
        return self._position_only(logical_points)

    def jacobian_matrix(self, logical_points: Any) -> np.ndarray:
        return self._position_and_jacobian(logical_points)[1]

    def evaluate(self, logical_points: Any, *, reject_nonpositive_J: bool = True) -> MetricEvaluation:
        if self._topology == "toroidal":
            q = np.asarray(logical_points, dtype=float)
            if np.any(q[..., 0] == 0.0):
                raise ValueError("ordinary toroidal metric is singular at u=0; use evaluate_regularized")
        position, A = self._position_and_jacobian(logical_points)
        J = np.linalg.det(A)
        valid = np.isfinite(J) & (J > 0)
        if reject_nonpositive_J and np.any(~valid):
            raise ValueError("query contains nonpositive or nonfinite mesh Jacobian")
        g_cov = np.einsum("...ki,...kj->...ij", A, A)
        g_contra = np.linalg.inv(g_cov)
        residual = np.max(np.abs(np.einsum("...ik,...kj->...ij", g_cov, g_contra) - np.eye(3)), axis=(-2, -1))
        return MetricEvaluation(position, A, J, g_cov, g_contra, residual, valid)

    def basis_cache_diagnostics(self) -> dict[str, Any]:
        """Describe retained Fourier--Zernike radial-basis storage.

        Radial matrices used for an evaluation are query-scoped and shared by
        the three coordinate channels.  They disappear when that evaluation
        returns.  This method intentionally reports only persistent storage so
        long-running callers can verify that distinct queries do not build an
        unbounded live cache.
        """

        caches = [getattr(channel, "_basis", {}) for channel in self._channels]
        return {
            "policy": "query-scoped-shared-across-channels",
            "persistent_entries": int(sum(len(cache) for cache in caches)),
            "persistent_value_bytes": int(
                sum(
                    value.nbytes
                    for cache in caches
                    for value in cache.values()
                    if isinstance(value, np.ndarray)
                )
            ),
            "persistent_key_bytes": 0,
        }

    def evaluate_regularized(self, logical_points: Any) -> RegularizedMetricEvaluation:
        """Evaluate ``[X_u, X_theta/u, (period/2*pi) X_eta]``."""
        if self._topology != "toroidal":
            raise ValueError("evaluate_regularized is only available for toroidal topology")
        q = np.asarray(logical_points, dtype=float)
        if q.ndim == 0 or q.shape[-1] != 3 or not np.all(np.isfinite(q)):
            raise ValueError("logical_points must have shape (..., 3) and be finite")
        channels, shape = self._channels_at(q, ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)))
        (R, Z, delta), (Ru, Zu, du), _, (Re, Ze, de) = channels
        prepared = self._channels[0].prepare(q[..., 0].reshape(-1), q[..., 1].reshape(-1), q[..., 2].reshape(-1))
        theta_over_u = tuple(
            channel.evaluate_prepared(prepared, theta_over_u=True).reshape(shape)
            for channel in self._channels
        )
        Rt, Zt, dt = theta_over_u
        phi = q[..., 2] + delta
        cp, sp = np.cos(phi), np.sin(phi)
        def cart(rd, pd, zd):
            return np.stack((cp * rd - R * sp * pd, sp * rd + R * cp * pd, zd), axis=-1)
        eta_scale = self._period / (2.0 * np.pi)
        H = np.stack(
            (
                cart(Ru, du, Zu),
                cart(Rt, dt, Zt),
                eta_scale * cart(Re, 1.0 + de, Ze),
            ),
            axis=-1,
        )
        J = np.linalg.det(H)
        cov = np.einsum("...ki,...kj->...ij", H, H)
        contra = np.linalg.inv(cov)
        residual = np.max(np.abs(np.einsum("...ik,...kj->...ij", cov, contra) - np.eye(3)), axis=(-2, -1))
        condition = np.linalg.cond(H)
        valid = np.isfinite(J) & (J > 0) & np.isfinite(condition)
        return RegularizedMetricEvaluation(
            self._position_only(q), H, J, cov, contra, condition, residual, valid
        )

    def sample(self, u: Any, v: Any, eta: Any, *, reject_nonpositive_J: bool = True) -> MetricEvaluation:
        """Evaluate a structured tensor-product logical grid."""
        ug, vg, eg = np.meshgrid(np.asarray(u), np.asarray(v), np.asarray(eta), indexing="ij")
        return self.evaluate(np.stack((ug, vg, eg), axis=-1), reject_nonpositive_J=reject_nonpositive_J)


    def project_magnetic_field(
        self,
        metrics: MetricEvaluation,
        bfield_evaluator: Any,
    ) -> MagneticFieldEvaluation:
        """Project a Cartesian magnetic field through an existing metric sample."""

        position = np.asarray(metrics.position, dtype=np.float64)
        jacobian = np.asarray(metrics.jacobian_matrix, dtype=np.float64)
        if position.ndim == 0 or position.shape[-1] != 3:
            raise ValueError("metric position must have shape (..., 3)")
        if jacobian.shape != position.shape[:-1] + (3, 3):
            raise ValueError("metric Jacobian must have shape (..., 3, 3)")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(jacobian)):
            raise ValueError("metric sample must be finite")
        B = np.asarray(bfield_evaluator.evaluate_cartesian(position), dtype=np.float64)
        if B.shape != position.shape or B.shape[-1] != 3 or not np.all(np.isfinite(B)):
            raise ValueError("B-field evaluator must return finite vectors with shape (..., 3)")
        Bcontra = np.linalg.solve(jacobian, B[..., None])[..., 0]
        return MagneticFieldEvaluation(B, Bcontra, np.linalg.norm(B, axis=-1))

    def evaluate_magnetic_field(self, logical_points: Any, bfield_evaluator: Any, *, reject_nonpositive_J: bool = True) -> MagneticFieldEvaluation:
        """Evaluate the metric and project the Cartesian magnetic field."""

        metrics = self.evaluate(logical_points, reject_nonpositive_J=reject_nonpositive_J)
        return self.project_magnetic_field(metrics, bfield_evaluator)


def _validate_mesh_shape(mesh_shape: Any) -> tuple[int, int, int]:
    try:
        shape = tuple(int(value) for value in mesh_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("mesh_shape must contain three positive integers") from exc
    if len(shape) != 3 or any(value < 2 for value in shape):
        raise ValueError("mesh_shape must contain three integers of at least two")
    return shape


_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class ToroidalQualityReport:
    """Local, geometry-only diagnostics for a Fourier--Zernike mesh."""

    wall_error_rms: float
    wall_error_max: float
    eta_residual_rms: float
    eta_residual_max: float
    min_J_reg: float
    J_reg_p01: float
    J_reg_median: float
    J_reg_p99: float
    ordinary_J_min: float
    condition_max: float
    condition_p95: float
    condition_p99: float
    minimum_singular_value: float
    max_neighbor_log_J_reg_jump: float
    nonpositive_J_reg_count: int
    seam_residual_rms: float
    seam_residual_max: float
    fit_residual_rms: float
    fit_residual_max: float
    axis_sample_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def rotate_z(points: Any, angle: float) -> np.ndarray:
    """Rotate Cartesian points physically about the laboratory z axis."""
    value = _as_points(points)
    cosine, sine = np.cos(float(angle)), np.sin(float(angle))
    result = value.copy()
    result[..., 0] = cosine * value[..., 0] - sine * value[..., 1]
    result[..., 1] = sine * value[..., 0] + cosine * value[..., 1]
    return result


def _as_points(points: Any, name: str = "points") -> np.ndarray:
    value = np.asarray(points, dtype=float)
    if value.ndim < 2 or value.shape[-1] != 3 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must have shape (..., 3) and be finite")
    return value


def _periodic_curve(curve: Any) -> tuple[np.ndarray, bool]:
    value = _as_points(curve, "curve")
    if value.ndim != 2:
        raise ValueError("curve must have shape (n, 3)")
    duplicate = np.linalg.norm(value[0] - value[-1]) <= 1e-10 * max(1.0, np.ptp(value, axis=0).max())
    if duplicate:
        value = value[:-1]
    if value.shape[0] < 3:
        raise ValueError("a closed curve needs at least three distinct points")
    return value, duplicate


def resample_periodic_curve(curve: Any, count: int) -> np.ndarray:
    """Resample a closed Cartesian curve uniformly in periodic arc length.

    ``curve`` may include a repeated final point.  The returned array has
    exactly ``count`` endpoint-exclusive points and is suitable for FFTs.
    Linear interpolation is used on the periodic arc-length parameter, which
    avoids imposing a spline overshoot on a wall supplied by a file.
    """
    if int(count) != count or count < 3:
        raise ValueError("count must be an integer at least three")
    base, _ = _periodic_curve(curve)
    edges = np.linalg.norm(np.roll(base, -1, axis=0) - base, axis=1)
    if np.any(edges <= 0) or not np.isfinite(edges.sum()):
        raise ValueError("curve contains coincident or invalid consecutive points")
    s = np.concatenate(([0.0], np.cumsum(edges)))
    samples = np.vstack((base, base[0]))
    target = np.arange(int(count), dtype=float) * (s[-1] / int(count))
    return np.column_stack([np.interp(target, s, samples[:, k]) for k in range(3)])


def _curve_rz_orientation(curve: np.ndarray) -> float:
    radius = np.hypot(curve[:, 0], curve[:, 1])
    z = curve[:, 2]
    return float(0.5 * np.sum(radius * np.roll(z, -1) - np.roll(radius, -1) * z))


def _positive_jacobian_orientation(curve: np.ndarray) -> np.ndarray:
    """Orient an R/Z section for positive ``det[X_u,X_theta,X_eta]``."""
    if _curve_rz_orientation(curve) > 0.0:
        # Keep the phase anchor at index zero while reversing traversal.
        return np.concatenate((curve[:1], curve[:0:-1]), axis=0)
    return curve


def align_wall_curves(
    curves: Any, *, allow_reversal: bool = True, field_period: float | None = None
) -> np.ndarray:
    """Orient and cyclically align a sequence of closed wall curves.

    The cyclic shifts are selected jointly around the eta seam using a small
    dynamic program.  This makes the first/last field-period planes part of
    the same optimization and therefore handles a rotated field-period seam.
    Curves must have identical point counts and be ordered periodically in
    eta.  A returned curve is endpoint-exclusive.
    """
    values = _as_points(curves, "curves")
    if values.ndim != 3:
        raise ValueError("curves must have shape (neta, ntheta, 3)")
    neta, ntheta = values.shape[:2]
    if neta < 2 or ntheta < 3:
        raise ValueError("at least two curves and three poloidal points are required")
    oriented = values.copy()
    for k in range(neta):
        signed_area = _curve_rz_orientation(oriented[k])
        if signed_area == 0.0:
            raise ValueError("wall curve has zero signed R/Z area")
        if allow_reversal:
            oriented[k] = _positive_jacobian_orientation(oriented[k])
        elif signed_area > 0.0:
            raise ValueError("wall orientation gives a negative toroidal Jacobian")

    # Compare sections in a co-rotating frame.  Direct Cartesian costs bias
    # the phase toward pairing large-R points with small-R points as the
    # toroidal plane rotates, even for a perfectly axisymmetric torus.
    comparison = oriented
    if field_period is not None:
        comparison = np.stack(
            [
                rotate_z(oriented[k], -k * float(field_period) / neta)
                for k in range(neta)
            ],
            axis=0,
        )

    # Pair costs depend only on relative shift.  Build each matrix from one
    # O(ntheta^2) vector instead of recomputing O(ntheta^3) rolled curves.
    pair = np.empty((neta, ntheta, ntheta), dtype=float)
    shift_indices = (
        np.arange(ntheta)[None, :] - np.arange(ntheta)[:, None]
    ) % ntheta
    for k in range(neta):
        nxt = (k + 1) % neta
        relative_cost = np.asarray(
            [
                float(np.mean((comparison[k] - np.roll(comparison[nxt], shift, axis=0)) ** 2))
                for shift in range(ntheta)
            ]
        )
        pair[k] = relative_cost[shift_indices]

    # Fix the first section's phase anchor.  A simultaneous cyclic shift of
    # every section has the same objective and should not move the user's
    # chosen theta=0 reference.
    cost = np.full(ntheta, np.inf)
    cost[0] = 0.0
    predecessors = np.empty((neta - 1, ntheta), dtype=int)
    for k in range(neta - 1):
        candidates = cost[:, None] + pair[k]
        predecessors[k] = np.argmin(candidates, axis=0)
        cost = candidates[predecessors[k], np.arange(ntheta)]
    terminal = cost + pair[-1][:, 0]
    final = int(np.argmin(terminal))
    best_total = float(terminal[final])
    best_shifts = np.empty(neta, dtype=int)
    best_shifts[-1] = final
    for k in range(neta - 2, -1, -1):
        best_shifts[k] = predecessors[k][best_shifts[k + 1]]
    if not np.isfinite(best_total):
        raise ValueError("could not align wall curves")
    return np.stack([np.roll(oriented[k], best_shifts[k], axis=0) for k in range(neta)], axis=0)


def _call_eta(evaluator: Any, xyz: np.ndarray) -> np.ndarray:
    combined = getattr(evaluator, "evaluate_and_gradient_cartesian", None)
    if combined is not None:
        try:
            result, _ = combined(xyz, wrapped=False)
        except TypeError:
            result, _ = combined(xyz)
        result = np.asarray(result, dtype=float)
        if result.shape == xyz.shape[:-1] or result.shape == xyz.shape[:-1] + (1,):
            return result.reshape(xyz.shape[:-1])
    for name in ("evaluate_cartesian", "evaluate", "__call__"):
        function = getattr(evaluator, name, None)
        if function is not None:
            result = function(xyz)
            if hasattr(result, "value"):
                result = result.value
            result = np.asarray(result, dtype=float)
            if result.shape == xyz.shape[:-1] or result.shape == xyz.shape[:-1] + (1,):
                return result.reshape(xyz.shape[:-1])
    raise TypeError("eta_evaluator must be callable or expose evaluate_cartesian/evaluate")


def _eta_value_gradient(
    evaluator: Any, xyz: np.ndarray, *, finite_difference: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Use ScalarPotentialEvaluator's combined or separate Cartesian API."""
    combined = getattr(evaluator, "evaluate_and_gradient_cartesian", None)
    if combined is not None:
        value, gradient = combined(xyz, wrapped=False)
        return np.asarray(value, float).reshape(xyz.shape[:-1]), np.asarray(gradient, float).reshape(xyz.shape)
    value = _call_eta(evaluator, xyz)
    gradient_function = getattr(evaluator, "gradient_cartesian", None)
    if gradient_function is not None:
        gradient = gradient_function(xyz)
        return value, np.asarray(gradient, float).reshape(xyz.shape)
    gradient = np.empty_like(xyz)
    for component in range(3):
        offset = np.zeros_like(xyz)
        offset[..., component] = finite_difference
        gradient[..., component] = (
            _call_eta(evaluator, xyz + offset) - _call_eta(evaluator, xyz - offset)
        ) / (2.0 * finite_difference)
    return value, gradient


def cartesian_to_channels(
    values: Any, eta: Any, *, eta_axis: int = -1
) -> np.ndarray:
    """Convert Cartesian points to ``(R, Z, delta_phi)`` channels."""
    xyz = _as_points(values)
    eta_array = np.asarray(eta, dtype=float)
    radius = np.hypot(xyz[..., 0], xyz[..., 1])
    if np.any(radius <= 0):
        raise ValueError("Cartesian points must have positive cylindrical radius")
    phi, eta_array, radius, height = np.broadcast_arrays(
        np.unwrap(np.arctan2(xyz[..., 1], xyz[..., 0]), axis=eta_axis),
        eta_array,
        radius,
        xyz[..., 2],
    )
    return np.stack((radius, height, phi - eta_array), axis=-1)


def channels_to_cartesian(channels: Any, eta: Any) -> np.ndarray:
    """Convert ``(R, Z, delta_phi)`` channels back to Cartesian points."""
    values = np.asarray(channels, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("channels must have shape (..., 3) and be finite")
    radius, height, delta = np.moveaxis(values, -1, 0)
    phase = np.asarray(eta, dtype=float) + delta
    radius, height, phase = np.broadcast_arrays(radius, height, phase)
    if np.any(radius <= 0):
        raise ValueError("R must be positive")
    return np.stack((radius * np.cos(phase), radius * np.sin(phase), height), axis=-1)


def axis_regular_initializer(
    wall: Any,
    axis: Any,
    u: Any,
    *,
    poloidal_modes: int | None = None,
    toroidal_modes: int | None = None,
) -> np.ndarray:
    """Build a disk-times-circle mesh using radial Fourier--Zernike scaling."""
    wall_values = _as_points(wall, "wall")
    axis_values = _as_points(axis, "axis")
    radial = np.asarray(u, dtype=float)
    if (
        wall_values.ndim != 3
        or axis_values.ndim != 2
        or axis_values.shape[0] != wall_values.shape[1]
    ):
        raise ValueError("wall must be (ntheta, neta, 3) and axis must be (neta, 3)")
    if radial.ndim != 1 or radial.size < 2 or np.any(np.diff(radial) <= 0) or radial[0] < 0 or radial[-1] > 1:
        raise ValueError("u must be strictly increasing and lie in [0, 1]")
    ntheta, neta = wall_values.shape[:2]
    if not np.allclose(radial[0], 0.0) or not np.allclose(radial[-1], 1.0):
        raise ValueError("u must include both 0 and 1")
    if poloidal_modes is None:
        poloidal_modes = ntheta // 2
    if int(poloidal_modes) != poloidal_modes or poloidal_modes < 1:
        raise ValueError("poloidal_modes must be a positive integer")
    relative = wall_values - axis_values[None, :, :]
    # NumPy's inverse transform already applies the 1/ntheta normalization.
    coeff = np.fft.fft(relative, axis=0)
    keep = min(int(poloidal_modes), ntheta // 2)
    mask = np.zeros(ntheta, dtype=bool)
    mask[:keep + 1] = True
    if keep:
        mask[-keep:] = True
    coeff[~mask] = 0.0
    modes = np.fft.fftfreq(ntheta) * ntheta
    positions = np.empty((radial.size, ntheta, neta, 3), dtype=float)
    for i, radius in enumerate(radial):
        scale = np.where(modes == 0, radius * radius, radius ** np.abs(modes))
        positions[i] = axis_values[None, :, :] + np.fft.ifft(coeff * scale[:, None, None], axis=0).real
    positions[0] = axis_values[None, :, :]
    positions[-1] = wall_values
    if toroidal_modes is not None:
        if int(toroidal_modes) != toroidal_modes or toroidal_modes < 0:
            raise ValueError("toroidal_modes must be a nonnegative integer")
        count = neta
        nmode = np.fft.fftfreq(count) * count
        keep_t = np.abs(nmode) <= int(toroidal_modes)
        transformed = np.fft.fft(positions, axis=2)
        transformed[:, :, ~keep_t, :] = 0.0
        filtered = np.fft.ifft(transformed, axis=2).real
        filtered[0] = positions[0]
        filtered[-1] = positions[-1]
        positions = filtered
    return positions


def project_interior_eta(
    positions: Any,
    eta_evaluator: Any,
    eta: Any,
    *,
    period: float,
    eta_is_normalized: bool = False,
    iterations: int = 3,
    finite_difference: float = 1e-6,
) -> np.ndarray:
    """Project only interior nodes to wrapped eta surfaces along eta gradients."""
    result = _as_points(positions, "positions").copy()
    eta_axis = np.asarray(eta, dtype=float)
    if result.ndim != 4 or eta_axis.ndim != 1 or result.shape[2] != eta_axis.size:
        raise ValueError("positions must be (nu, ntheta, neta, 3), matching eta")
    if iterations < 0 or finite_difference <= 0:
        raise ValueError("iterations must be nonnegative and finite_difference positive")
    for _ in range(int(iterations)):
        interior = result[1:-1].reshape(-1, 3)
        values, gradient = _eta_value_gradient(
            eta_evaluator, interior, finite_difference=finite_difference
        )
        if eta_is_normalized:
            values = values * period
            gradient = gradient * period
        target_values = np.broadcast_to(
            eta_axis, result[1:-1].shape[:-1]
        ).reshape(-1)
        residual = values - target_values
        residual = (residual + 0.5 * period) % period - 0.5 * period
        denominator = np.einsum("ij,ij->i", gradient, gradient)
        valid = denominator > 1e-20
        interior[valid] -= residual[valid, None] * gradient[valid] / denominator[valid, None]
        result[1:-1] = interior.reshape(result[1:-1].shape)
    return result


def evaluate_toroidal_quality(
    positions: Any,
    u: Any,
    eta: Any,
    *,
    period: float,
    wall_reference: Any | None = None,
    axis_reference: Any | None = None,
    eta_evaluator: Any | None = None,
    eta_is_normalized: bool = False,
    axis_oversample: int = 8,
) -> ToroidalQualityReport:
    """Return axis-focused mesh diagnostics without invoking solver code.

    The radial samples are linearly oversampled between the first two or
    three supplied radial nodes, while theta and eta use periodic centered
    differences.  ``J_reg`` is the determinant of
    ``[X_u, X_theta/u, (period/2*pi) X_eta]``; it is the finite limit
    diagnostic and is not passed to a PDE operator as an ordinary metric
    tensor.
    """
    values = _as_points(positions, "positions")
    radial = np.asarray(u, float)
    eta_axis = np.asarray(eta, float)
    if values.ndim != 4 or radial.ndim != 1 or values.shape[0] != radial.size:
        raise ValueError("positions must be (nu, ntheta, neta, 3), matching u")
    if eta_axis.ndim != 1 or values.shape[2] != eta_axis.size or eta_axis.size < 4:
        raise ValueError("eta must match positions and contain at least four nodes")
    if axis_oversample < 2:
        raise ValueError("axis_oversample must be at least two")
    if np.any(np.diff(radial) <= 0) or radial[0] < 0 or radial[-1] > 1:
        raise ValueError("u must be strictly increasing in [0, 1]")
    near_end = radial[min(2, radial.size - 1)]
    near = np.linspace(max(radial[0], 1e-12), near_end, int(axis_oversample))
    u_eval = np.unique(np.concatenate((near, radial[1:])))
    flat = values.reshape(values.shape[0], -1)
    sampled = np.stack(
        [np.interp(u_eval, radial, flat[:, j]) for j in range(flat.shape[1])], axis=1
    ).reshape((u_eval.size,) + values.shape[1:])
    du = np.gradient(sampled, u_eval, axis=0, edge_order=2 if u_eval.size > 2 else 1)
    dtheta = _TWO_PI / values.shape[1]
    deta = period / values.shape[2]
    dth = (np.roll(sampled, -1, axis=1) - np.roll(sampled, 1, axis=1)) / (2.0 * dtheta)
    eta_forward = np.roll(sampled, -1, axis=2)
    eta_backward = np.roll(sampled, 1, axis=2)
    eta_forward[:, :, -1] = rotate_z(sampled[:, :, 0], period)
    eta_backward[:, :, 0] = rotate_z(sampled[:, :, -1], -period)
    de = (eta_forward - eta_backward) / (2.0 * deta)
    positive = u_eval > 0
    regular = np.stack(
        (
            du[positive],
            dth[positive] / u_eval[positive, None, None, None],
            (period / _TWO_PI) * de[positive],
        ),
        axis=-1,
    )
    j_reg = np.linalg.det(regular)
    condition = np.linalg.cond(np.moveaxis(regular, -1, -2))
    singular_values = np.linalg.svd(regular, compute_uv=False)
    finite_j = j_reg[np.isfinite(j_reg)]
    finite_condition = condition[np.isfinite(condition)]
    if finite_j.size == 0 or finite_condition.size == 0:
        raise ValueError("mesh has no finite positive-u quality samples")
    nonpositive = int(np.count_nonzero(~np.isfinite(j_reg) | (j_reg <= 0.0)))
    if nonpositive:
        max_log_jump = float("inf")
    else:
        log_j = np.log(j_reg)
        max_log_jump = float(
            max(
                np.max(np.abs(np.diff(log_j, axis=0))) if log_j.shape[0] > 1 else 0.0,
                np.max(np.abs(np.roll(log_j, -1, axis=1) - log_j)),
                np.max(np.abs(np.roll(log_j, -1, axis=2) - log_j)),
            )
        )

    wall_error = np.zeros(1)
    if wall_reference is not None:
        wall = np.asarray(wall_reference, float)
        if wall.shape == values.shape[1:]:
            wall_error = np.linalg.norm(values[-1] - wall, axis=-1).ravel()
        elif wall.shape == (values.shape[2], values.shape[1], 3):
            wall_error = np.linalg.norm(values[-1] - wall.transpose(1, 0, 2), axis=-1).ravel()
        else:
            raise ValueError("wall_reference has incompatible shape")
    axis_error = np.zeros(1)
    if axis_reference is not None:
        axis = _as_points(axis_reference, "axis_reference")
        if axis.shape != (values.shape[2], 3):
            raise ValueError("axis_reference must have shape (neta, 3)")
        axis_error = np.linalg.norm(values[0] - axis[None, :, :], axis=-1).ravel()
    fit_error = np.concatenate((wall_error, axis_error))

    eta_error = np.zeros(1)
    if eta_evaluator is not None:
        eta_value = _call_eta(eta_evaluator, values.reshape(-1, 3))
        if eta_is_normalized:
            eta_value *= period
        target = np.broadcast_to(eta_axis, values.shape[:-1]).reshape(-1)
        eta_error = ((eta_value - target + 0.5 * period) % period) - 0.5 * period
    seam_delta = period / values.shape[2]
    # Both terms represent the same physical endpoint at eta_0 + period.
    seam = rotate_z(values[:, :, 0], period) - rotate_z(values[:, :, -1], seam_delta)
    seam_error = np.linalg.norm(seam, axis=-1).ravel()
    return ToroidalQualityReport(
        wall_error_rms=float(np.sqrt(np.mean(wall_error ** 2))),
        wall_error_max=float(np.max(wall_error)),
        eta_residual_rms=float(np.sqrt(np.mean(eta_error ** 2))),
        eta_residual_max=float(np.max(np.abs(eta_error))),
        min_J_reg=float(np.min(finite_j)),
        J_reg_p01=float(np.percentile(finite_j, 1)),
        J_reg_median=float(np.median(finite_j)),
        J_reg_p99=float(np.percentile(finite_j, 99)),
        ordinary_J_min=float(
            np.min(
                j_reg
                * u_eval[positive, None, None]
                * (_TWO_PI / period)
            )
        ),
        condition_max=float(np.max(finite_condition)),
        condition_p95=float(np.percentile(finite_condition, 95)),
        condition_p99=float(np.percentile(finite_condition, 99)),
        minimum_singular_value=float(np.min(singular_values[..., -1])),
        max_neighbor_log_J_reg_jump=max_log_jump,
        nonpositive_J_reg_count=nonpositive,
        seam_residual_rms=float(np.sqrt(np.mean(seam_error ** 2))),
        seam_residual_max=float(np.max(seam_error)),
        fit_residual_rms=float(np.sqrt(np.mean(fit_error ** 2))),
        fit_residual_max=float(np.max(fit_error)),
        axis_sample_count=int(np.count_nonzero(positive)),
    )


def _build_toroidal_topology(
    wall_evaluator: Any,
    eta_evaluator: Any | None = None,
    *,
    u: Any,
    theta: Any,
    eta: Any,
    period: float | None = None,
    nfp: int | None = None,
    wall_curves: Any | None = None,
    radial_degree: int = 3,
    poloidal_modes: int | None = None,
    toroidal_modes: int | None = None,
    projection_iterations: int = 0,
    wall_sample_count: int | None = None,
    validate: bool = True,
) -> MetricEvaluator:
    """Build a toroidal metric using the projected centroid of aligned wall curves."""
    u = np.asarray(u, dtype=float)
    theta = np.asarray(theta, dtype=float)
    eta = np.asarray(eta, dtype=float)
    if period is None:
        if nfp is None:
            raise ValueError("supply period or nfp")
        period = _TWO_PI / int(nfp)
    period = float(period)
    if period <= 0 or eta.ndim != 1 or eta.size < 4 or not np.allclose(np.diff(eta), period / eta.size):
        raise ValueError("eta must be uniform, endpoint-exclusive, and span period")
    if theta.ndim != 1 or theta.size < 3 or not np.allclose(np.diff(theta), _TWO_PI / theta.size):
        raise ValueError("theta must be uniform endpoint-exclusive poloidal nodes")
    if wall_curves is None:
        if eta_evaluator is None:
            raise ValueError(
                "eta_evaluator is required when wall_curves are not supplied"
            )
        sampler = getattr(wall_evaluator, "constant_eta_boundary_curve", None)
        if sampler is None:
            raise TypeError(
                "wall_evaluator must expose constant_eta_boundary_curve"
            )
        if wall_sample_count is None:
            wall_sample_count = max(theta.size + 1, 4 * theta.size + 1)
        if int(wall_sample_count) != wall_sample_count or wall_sample_count < theta.size + 1:
            raise ValueError("wall_sample_count must be at least ntheta + 1")
        wall_curves = np.stack(
            [
                sampler(
                    eta_evaluator,
                    float(target),
                    npoints=int(wall_sample_count),
                )
                for target in eta
            ],
            axis=0,
        )
    wall_curves = _as_points(wall_curves, "wall_curves")
    if wall_curves.ndim != 3:
        raise ValueError("wall_curves must have shape (neta, nwall, 3) or transposed")
    if wall_curves.shape[0] == eta.size:
        sampled_wall = wall_curves
    elif wall_curves.shape[1] == eta.size:
        sampled_wall = wall_curves.transpose(1, 0, 2)
    else:
        raise ValueError("wall_curves must have shape (neta, nwall, 3) or transposed")
    sampled_wall = np.stack([resample_periodic_curve(curve, theta.size) for curve in sampled_wall])
    sampled_wall = align_wall_curves(sampled_wall, field_period=period)
    # Convert only after alignment: the cylindrical phase channel is then
    # continuous over eta and its poloidal mean is a well-defined centerline
    # guess even when WallEvaluator.centerline() is offset or unavailable.
    wall_channels = cartesian_to_channels(
        sampled_wall.transpose(1, 0, 2), eta[None, :], eta_axis=1
    )
    axis_channels = np.mean(wall_channels, axis=0)
    if not np.all(np.isfinite(axis_channels)) or np.any(axis_channels[:, 0] <= 0.0):
        raise ValueError("wall-centroid axis guess has invalid cylindrical channels")
    axis_points = channels_to_cartesian(axis_channels, eta)
    if eta_evaluator is not None:
        # Newton projection is along the eta gradient and uses a wrapped
        # residual so the field-period seam remains endpoint-exclusive.
        for _ in range(12):
            values, gradients = _eta_value_gradient(eta_evaluator, axis_points)
            if values.shape != eta.shape or gradients.shape != axis_points.shape:
                raise ValueError("eta evaluator returned invalid axis value/gradient shapes")
            if not np.all(np.isfinite(values)) or not np.all(np.isfinite(gradients)):
                raise ValueError("eta evaluator returned nonfinite axis values or gradients")
            residual = (values - eta + 0.5 * period) % period - 0.5 * period
            if np.max(np.abs(residual)) <= max(2.0e-11, 2.0e-10 * period):
                break
            denominator = np.einsum("ij,ij->i", gradients, gradients)
            if not np.all(np.isfinite(denominator)) or np.any(denominator <= 1.0e-24):
                raise ValueError("eta evaluator gradient is invalid or vanishes during axis projection")
            axis_points = axis_points - residual[:, None] * gradients / denominator[:, None]
            if not np.all(np.isfinite(axis_points)):
                raise ValueError("axis projection produced nonfinite points")
        else:
            raise ValueError("wall-centroid axis projection did not converge")
    else:
        # Preserve the existing wall-curves-only use case.  Without an
        # eta evaluator there is no eta value/gradient machinery with
        # which to perform the projection.
        residual = None
    axis_points = _as_points(axis_points, "axis_points")
    if axis_points.shape != (eta.size, 3):
        raise ValueError("axis_points must have shape (neta, 3)")
    contains = getattr(wall_evaluator, "contains_cartesian", None)
    if contains is not None and not np.all(np.asarray(contains(axis_points), dtype=bool)):
        raise ValueError("the coordinate axis must remain inside the wall")
    if eta_evaluator is not None:
        axis_eta = _call_eta(eta_evaluator, axis_points)
        axis_residual = (axis_eta - eta + 0.5 * period) % period - 0.5 * period
        if np.max(np.abs(axis_residual)) > 2.0e-6 * max(1.0, period):
            raise ValueError("wall-centroid projected axis does not satisfy the requested eta levels")
    if poloidal_modes is None:
        poloidal_modes = min(int(radial_degree), (theta.size - 1) // 2)
    # Apply modal radial scaling to the same periodic cylindrical channels
    # used by MetricEvaluator.  This avoids branch artefacts in delta-phi and
    # makes the u^{|m|} regularity explicit in the fitted variables.
    axis_channels = cartesian_to_channels(axis_points, eta, eta_axis=0)
    channel_positions = axis_regular_initializer(
        wall_channels, axis_channels, u,
        poloidal_modes=poloidal_modes, toroidal_modes=toroidal_modes,
    )
    positions = channels_to_cartesian(
        channel_positions, eta[None, None, :]
    )
    if eta_evaluator is not None and projection_iterations:
        positions = project_interior_eta(
            positions, eta_evaluator, eta, period=period,
            iterations=projection_iterations,
        )
    evaluator = MetricEvaluator(
        u,
        theta,
        eta,
        positions,
        period=period,
        nfp=nfp,
        topology="toroidal",
        radial_degree=radial_degree,
        poloidal_modes=poloidal_modes,
        toroidal_modes=toroidal_modes,
    )
    if validate:
        validate_fitted_map(evaluator)
    return evaluator


def validate_fitted_map(evaluator):
    u, theta, eta, period = evaluator.u, evaluator.v, evaluator.eta, evaluator.period
    radial = np.unique(np.concatenate(([0.0, min(0.25 * u[1], 1.0)], 0.5 * (u[:-1] + u[1:]), [1.0])))
    theta = theta + 0.37 * (_TWO_PI / theta.size)
    eta = eta + 0.31 * (period / eta.size)
    validation_points = np.stack(np.meshgrid(radial, theta, eta, indexing="ij"), axis=-1)
    regularized = evaluator.evaluate_regularized(validation_points)
    if np.any(~regularized.valid):
        minimum = float(np.nanmin(regularized.J_reg))
        raise ValueError(
            "toroidal Fourier-Zernike fit folds on the validation grid; "
            f"minimum regularized Jacobian is {minimum:.6e}"
        )


def build_metric_evaluator(
    eta_evaluator: Any,
    *,
    topology: str = "square",
    wall_evaluator: Any | None = None,
    mesh_shape: tuple[int, int, int] | None = None,
    logical_axes: tuple[Any, Any, Any] | None = None,
    wall_curves: Any | None = None,
    radial_degree: int = 3,
    poloidal_modes: int | None = None,
    toroidal_modes: int | None = None,
    projection_iterations: int = 0,
    wall_sample_count: int | None = None,
    validate: bool = True,
) -> MetricEvaluator:
    """Build an axis-regular Fourier–Zernike metric from wall curves."""
    selected_topology = str(topology).lower()
    if selected_topology != "toroidal":
        raise ValueError("Only toroidal topology is supported")

    try:
        period = float(eta_evaluator.period)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "eta_evaluator must provide a positive finite period"
        ) from error
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("eta_evaluator.period must be positive and finite")
    eta_nfp = getattr(eta_evaluator, "nfp", None)
    if eta_nfp is not None:
        if (
            isinstance(eta_nfp, (bool, np.bool_))
            or int(eta_nfp) != eta_nfp
            or int(eta_nfp) < 1
        ):
            raise ValueError("eta_evaluator.nfp must be a positive integer")
        eta_nfp = int(eta_nfp)
        if not np.isclose(
            period, 2.0 * np.pi / eta_nfp, rtol=2.0e-12, atol=2.0e-12
        ):
            raise ValueError(
                "eta_evaluator.period and eta_evaluator.nfp are inconsistent"
            )

    if mesh_shape is None:
        raise ValueError("toroidal wall construction requires mesh_shape")
    shape = _validate_mesh_shape(mesh_shape)

    if logical_axes is None:
        axes = (
            np.linspace(0.0, 1.0, shape[0]),
            _TWO_PI * np.arange(shape[1], dtype=np.float64) / shape[1],
            period * np.arange(shape[2], dtype=np.float64) / shape[2],
        )
    else:
        if len(logical_axes) != 3:
            raise ValueError("logical_axes must contain (u, theta, eta)")
        axes = tuple(np.asarray(axis, dtype=np.float64) for axis in logical_axes)
        if any(
            axis.ndim != 1
            or axis.size != count
            or not np.all(np.isfinite(axis))
            for axis, count in zip(axes, shape)
        ):
            raise ValueError(
                "logical axes must be finite one-dimensional arrays matching the mesh"
            )

    return _build_toroidal_topology(
        wall_evaluator,
        eta_evaluator,
        u=axes[0],
        theta=axes[1],
        eta=axes[2],
        period=period,
        nfp=eta_nfp,
        wall_curves=wall_curves,
        radial_degree=radial_degree,
        poloidal_modes=poloidal_modes,
        toroidal_modes=toroidal_modes,
        projection_iterations=projection_iterations,
        wall_sample_count=wall_sample_count,
        validate=validate,
    )


def _validate_metric_spline_degree(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("metric_spline_degree must be an integer from 1 through 3")
    try:
        degree = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("metric_spline_degree must be an integer from 1 through 3") from exc
    if degree != value or degree < 1 or degree > 3:
        raise ValueError("metric_spline_degree must be an integer from 1 through 3")
    return degree


def _axis(values: Any, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be a finite one-dimensional axis")
    if not np.all(np.diff(axis) > 0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis.copy()


__all__ = [
    "MagneticFieldEvaluation",
    "MetricEvaluation",
    "RegularizedMetricEvaluation",
    "MetricEvaluator",
    "ToroidalQualityReport",
    "align_wall_curves",
    "axis_regular_initializer",
    "build_metric_evaluator",
    "evaluate_toroidal_quality",
    "project_interior_eta",
    "resample_periodic_curve",
]
