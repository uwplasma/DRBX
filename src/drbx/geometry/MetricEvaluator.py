"""Periodic spline representation of a structured three-dimensional mesh.

The logical topology represented here is ``D^2 x S^1``.  The first two
coordinates are non-periodic coordinates on ``[0, 1]^2`` and the last one is
an endpoint-exclusive, unwrapped toroidal coordinate.  The embedding is
represented by periodic Fourier coefficients in the toroidal coordinate and
by tensor-product splines in the two disk coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import BSpline, RectBivariateSpline
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

from .solve_MMPDE import MMPDEOptions, MMPDEResult, solve_mmpde


_METRIC_EVALUATOR_CACHE_FORMAT_VERSION = 1


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
class MagneticFieldEvaluation:
    """Magnetic field and its covariant/contravariant components."""

    B_cartesian: np.ndarray
    B_contravariant: np.ndarray
    B_covariant: np.ndarray
    magnitude: np.ndarray

    @property
    def B(self) -> np.ndarray:
        return self.B_cartesian

    @property
    def abs_B(self) -> np.ndarray:
        return self.magnitude


@dataclass(frozen=True)
class MetricQualityRegion:
    """Quality statistics for one logical sampling region."""

    label: str
    sample_count: int
    valid_fraction: float
    scaled_J_min: float
    mapping_condition_max: float


@dataclass(frozen=True)
class MetricQualityLocation:
    """Location of an extremal quality diagnostic."""

    value: float
    region: str
    logical: tuple[float, float, float]
    cartesian: tuple[float, float, float]


@dataclass(frozen=True)
class MetricQualityReport:
    """Sampling-based quality diagnostics for a fitted metric map.

    The report is descriptive: it does not classify a mesh as good or bad by
    imposing application-dependent thresholds.
    """

    sample_count: int
    valid_fraction: float
    raw_J_min: float
    raw_J_p01: float
    raw_J_median: float
    raw_J_max: float
    raw_J_min_over_median: float
    scaled_J_min: float
    scaled_J_p01: float
    mapping_condition_median: float
    mapping_condition_p95: float
    mapping_condition_max: float
    max_neighbor_log_J_jump: float
    inverse_residual_max: float
    regions: tuple[MetricQualityRegion, ...] = ()
    worst_scaled_jacobian: MetricQualityLocation | None = None
    worst_mapping_condition: MetricQualityLocation | None = None
    max_neighbor_log_J_jump_axis: str = "none"
    max_neighbor_log_J_jump_endpoint_a: tuple[float, float, float] = (float("nan"),) * 3
    max_neighbor_log_J_jump_endpoint_b: tuple[float, float, float] = (float("nan"),) * 3

    def summary(self, label: str | None = None) -> str:
        prefix = f"{label}: " if label else ""
        return (
            f"{prefix}samples={self.sample_count}, valid={self.valid_fraction:.3f}, "
            f"J=[{self.raw_J_min:.6e}, {self.raw_J_max:.6e}], "
            f"J_p01={self.raw_J_p01:.6e}, J_med={self.raw_J_median:.6e}, "
            f"J_min/med={self.raw_J_min_over_median:.6e}, "
            f"scaled_J=[{self.scaled_J_min:.6e}, p01={self.scaled_J_p01:.6e}], "
            f"cond(F)=[med={self.mapping_condition_median:.6e}, "
            f"p95={self.mapping_condition_p95:.6e}, max={self.mapping_condition_max:.6e}], "
            f"max_dlogJ={self.max_neighbor_log_J_jump:.6e}, "
            f"inverse_residual_max={self.inverse_residual_max:.6e}"
        )

    def detailed_summary(self, label: str | None = None) -> str:
        """Return readable region and extremum diagnostics.

        This intentionally formats NaN and infinity rather than applying
        thresholds, so it remains useful for inverted or singular trial maps.
        """
        prefix = f"{label}:\n" if label else ""
        lines = [prefix.rstrip("\n")] if prefix else []
        for region in self.regions:
            lines.append(
                f"  {region.label}: samples={region.sample_count}, "
                f"valid={region.valid_fraction:.3f}, "
                f"scaled_J_min={region.scaled_J_min:.6e}, "
                f"cond(F)_max={region.mapping_condition_max:.6e}"
            )

        def location_line(name: str, location: MetricQualityLocation | None) -> str:
            if location is None:
                return f"  {name}: value=nan, region=none, q=(nan, nan, nan), x=(nan, nan, nan)"
            q = ", ".join(f"{value:.6e}" for value in location.logical)
            x = ", ".join(f"{value:.6e}" for value in location.cartesian)
            return f"  {name}: value={location.value:.6e}, region={location.region}, q=({q}), x=({x})"

        lines.append(location_line("worst_scaled_J", self.worst_scaled_jacobian))
        lines.append(location_line("worst_cond(F)", self.worst_mapping_condition))
        a = ", ".join(f"{value:.6e}" for value in self.max_neighbor_log_J_jump_endpoint_a)
        b = ", ".join(f"{value:.6e}" for value in self.max_neighbor_log_J_jump_endpoint_b)
        lines.append(
            f"  max_dlogJ: value={self.max_neighbor_log_J_jump:.6e}, "
            f"axis={self.max_neighbor_log_J_jump_axis}, q_a=({a}), q_b=({b})"
        )
        return "\n".join(lines)


class _FourierSplineChannel:
    """Fourier series in eta whose coefficients are splines in (u, v)."""

    def __init__(
        self,
        u: np.ndarray,
        v: np.ndarray,
        samples: np.ndarray,
        eta0: float,
        period: float,
        metric_spline_degree: int = 3,
    ) -> None:
        degree = _validate_metric_spline_degree(metric_spline_degree)
        if u.size < 2 or v.size < 2:
            raise ValueError("u and v axes need at least two samples")
        self._u = np.asarray(u, dtype=np.float64)
        self._v = np.asarray(v, dtype=np.float64)
        self._degree = degree
        self._eta0 = float(eta0)
        self._period = float(period)
        self._modes = np.fft.fftfreq(samples.shape[2], d=1.0 / samples.shape[2])
        coeffs = np.fft.fft(samples, axis=2) / samples.shape[2]
        self._coefficients = np.ascontiguousarray(np.moveaxis(coeffs, 2, 0))
        # Preserve the historical cubic behavior for small explicit meshes by
        # reducing only the effective FITPACK degree to the available samples.
        kx = min(degree, u.size - 1)
        ky = min(degree, v.size - 1)
        self._spline_coefficients = None
        self._knots_u = self._knots_v = None
        self._spline_degree_u = self._spline_degree_v = None
        if degree > 1:
            splines = tuple(
                (
                    RectBivariateSpline(u, v, coeff.real, kx=kx, ky=ky, s=0),
                    RectBivariateSpline(u, v, coeff.imag, kx=kx, ky=ky, s=0),
                )
                for coeff in self._coefficients
            )
            real0 = splines[0][0]
            self._knots_u, self._knots_v = real0.get_knots()
            self._spline_degree_u, self._spline_degree_v = real0.degrees
            count_u = len(self._knots_u) - self._spline_degree_u - 1
            count_v = len(self._knots_v) - self._spline_degree_v - 1
            self._spline_coefficients = np.stack(
                [
                    real.get_coeffs().reshape(count_u, count_v)
                    + 1j * imag.get_coeffs().reshape(count_u, count_v)
                    for real, imag in splines
                ],
                axis=0,
            )

    def prepare(self, u: np.ndarray, v: np.ndarray, eta: np.ndarray) -> dict[str, Any]:
        """Prepare query data shared by every channel and derivative order."""
        uf = np.asarray(u, dtype=np.float64).reshape(-1)
        vf = np.asarray(v, dtype=np.float64).reshape(-1)
        ef = np.asarray(eta, dtype=np.float64).reshape(-1)
        theta = 2.0 * np.pi * (ef - self._eta0) / self._period
        prepared: dict[str, Any] = {
            "u": uf,
            "v": vf,
            "phase": np.exp(1j * np.multiply.outer(theta, self._modes)),
            "basis": {},
        }
        if self._degree == 1:
            iu = np.clip(
                np.searchsorted(self._u, uf, side="right") - 1,
                0,
                self._u.size - 2,
            )
            iv = np.clip(
                np.searchsorted(self._v, vf, side="right") - 1,
                0,
                self._v.size - 2,
            )
            du_width = self._u[iu + 1] - self._u[iu]
            dv_width = self._v[iv + 1] - self._v[iv]
            prepared.update(
                {
                    "iu": iu,
                    "iv": iv,
                    "tu": (uf - self._u[iu]) / du_width,
                    "tv": (vf - self._v[iv]) / dv_width,
                    "du_width": du_width,
                    "dv_width": dv_width,
                }
            )
        return prepared

    def _linear_evaluate_prepared(
        self, prepared: dict[str, Any], du: int, dv: int
    ) -> np.ndarray:
        """Evaluate every Fourier coefficient on a shared linear query."""
        iu, iv = prepared["iu"], prepared["iv"]
        tu, tv = prepared["tu"], prepared["tv"]
        coefficient = self._coefficients
        c00 = coefficient[:, iu, iv].T
        c10 = coefficient[:, iu + 1, iv].T
        c01 = coefficient[:, iu, iv + 1].T
        c11 = coefficient[:, iu + 1, iv + 1].T
        if du and dv:
            return (c11 - c10 - c01 + c00) / (
                prepared["du_width"] * prepared["dv_width"]
            )[:, None]
        if du:
            return (
                (1.0 - tv)[:, None] * (c10 - c00)
                + tv[:, None] * (c11 - c01)
            ) / prepared["du_width"][:, None]
        if dv:
            return (
                (1.0 - tu)[:, None] * (c01 - c00)
                + tu[:, None] * (c11 - c10)
            ) / prepared["dv_width"][:, None]
        omt, omv = 1.0 - tu, 1.0 - tv
        return (
            (omt * omv)[:, None] * c00
            + (tu * omv)[:, None] * c10
            + (omt * tv)[:, None] * c01
            + (tu * tv)[:, None] * c11
        )

    def evaluate_prepared(
        self,
        prepared: dict[str, Any],
        du: int = 0,
        dv: int = 0,
        deta: int = 0,
    ) -> np.ndarray:
        """Evaluate this channel using shared phase and interpolation data."""
        if self._degree == 1:
            values = self._linear_evaluate_prepared(prepared, du, dv)
        else:
            key = (du, dv)
            if key not in prepared["basis"]:
                count_u = len(self._knots_u) - self._spline_degree_u - 1
                count_v = len(self._knots_v) - self._spline_degree_v - 1
                basis_u = BSpline(
                    self._knots_u,
                    np.eye(count_u),
                    self._spline_degree_u,
                    extrapolate=True,
                ).derivative(du)(prepared["u"])
                basis_v = BSpline(
                    self._knots_v,
                    np.eye(count_v),
                    self._spline_degree_v,
                    extrapolate=True,
                ).derivative(dv)(prepared["v"])
                prepared["basis"][key] = (basis_u, basis_v)
            basis_u, basis_v = prepared["basis"][key]
            # A single four-operand contraction creates a very large
            # mode-by-query intermediate for scattered points.  Sharing the
            # basis while contracting each small Fourier coefficient matrix
            # separately is substantially faster and uses bounded memory.
            values = np.stack(
                [
                    np.einsum(
                        "ni,ij,nj->n",
                        basis_u,
                        coefficient,
                        basis_v,
                        optimize=True,
                    )
                    for coefficient in self._spline_coefficients
                ],
                axis=1,
            )
        if deta:
            values *= (1j * 2.0 * np.pi * self._modes / self._period) ** deta
        return np.einsum(
            "nm,nm->n", values, prepared["phase"], optimize=True
        ).real

    def evaluate(
        self,
        u: np.ndarray,
        v: np.ndarray,
        eta: np.ndarray,
        du: int = 0,
        dv: int = 0,
        deta: int = 0,
    ) -> np.ndarray:
        """Compatibility wrapper for an independently evaluated channel."""
        shape = np.asarray(eta).shape
        prepared = self.prepare(u, v, eta)
        return self.evaluate_prepared(prepared, du, dv, deta).reshape(shape)


class MetricEvaluator:
    """Evaluate a smooth ``D^2 x S^1`` mesh embedding and its metrics.

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
    metric_spline_degree:
        Tensor-product spline degree in ``u`` and ``v``.  Valid values are
        1, 2, and 3; the default is 3.  The exact corners of the logical
        square can be nonsmooth after wall fitting and should not be used as
        metric query points.  Use cell centers or open boundary-face centers.
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
        mmpde_result: MMPDEResult | None = None,
        mmpde_fit_scale: float = 1.0,
        metric_spline_degree: int = 3,
    ) -> None:
        self._u = _axis(u, "u")
        self._v = _axis(v, "v")
        self._eta = _axis(eta, "eta")
        metric_spline_degree = _validate_metric_spline_degree(metric_spline_degree)
        if self._u.size < 2 or self._v.size < 2 or self._eta.size < 4:
            raise ValueError("u and v need at least 2 samples; eta needs at least 4")
        if not np.isclose(self._u[0], 0.0) or not np.isclose(self._u[-1], 1.0):
            raise ValueError("u must span [0, 1]")
        if not np.isclose(self._v[0], 0.0) or not np.isclose(self._v[-1], 1.0):
            raise ValueError("v must span [0, 1]")
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
        self._mmpde_result = mmpde_result
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

        phi = np.unwrap(np.arctan2(values[..., 1], values[..., 0]), axis=2)
        delta_phi = phi - self._eta[None, None, :]
        fit_degree = min(metric_spline_degree, self._u.size - 1, self._v.size - 1)
        self._channels = tuple(
            _FourierSplineChannel(
                self._u,
                self._v,
                samples,
                self._eta[0],
                self._period,
                fit_degree,
            )
            for samples in (radius, values[..., 2], delta_phi)
        )

    @property
    def period(self) -> float:
        return self._period

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

    @property
    def mmpde_result(self) -> MMPDEResult | None:
        """Diagnostics from the originating solve, if built by the pipeline."""
        return self._mmpde_result

    @property
    def mmpde_fit_scale(self) -> float:
        """Accepted fraction of the raw MMPDE displacement after fit checks."""
        return self._mmpde_fit_scale

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
                _METRIC_EVALUATOR_CACHE_FORMAT_VERSION,
                dtype=np.int64,
            ),
            key("u"): self._u.copy(),
            key("v"): self._v.copy(),
            key("eta"): self._eta.copy(),
            key("period"): np.asarray(self._period, dtype=np.float64),
            key("nfp"): np.asarray(
                -1 if self._nfp is None else self._nfp,
                dtype=np.int64,
            ),
            key("metric_spline_degree"): np.asarray(
                self._metric_spline_degree,
                dtype=np.int64,
            ),
            key("mmpde_fit_scale"): np.asarray(
                self._mmpde_fit_scale,
                dtype=np.float64,
            ),
            key("channel_degree"): np.asarray(
                channels[0]._degree,
                dtype=np.int64,
            ),
            key("modes"): channels[0]._modes.copy(),
            key("coefficients"): np.stack(
                [channel._coefficients for channel in channels],
                axis=0,
            ),
        }
        if channels[0]._spline_coefficients is not None:
            payload.update(
                {
                    key("spline_coefficients"): np.stack(
                        [
                            channel._spline_coefficients
                            for channel in channels
                        ],
                        axis=0,
                    ),
                    key("knots_u"): channels[0]._knots_u.copy(),
                    key("knots_v"): channels[0]._knots_v.copy(),
                    key("spline_degree_u"): np.asarray(
                        channels[0]._spline_degree_u,
                        dtype=np.int64,
                    ),
                    key("spline_degree_v"): np.asarray(
                        channels[0]._spline_degree_v,
                        dtype=np.int64,
                    ),
                }
            )
        return payload

    @classmethod
    def from_cache_payload(
        cls,
        payload: Any,
        *,
        prefix: str = "",
    ) -> "MetricEvaluator":
        """Restore an evaluator from :meth:`to_cache_payload` arrays."""

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
        if not np.isclose(u[0], 0.0) or not np.isclose(u[-1], 1.0):
            raise ValueError("cached u must span [0, 1]")
        if not np.isclose(v[0], 0.0) or not np.isclose(v[-1], 1.0):
            raise ValueError("cached v must span [0, 1]")
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
        channel_degree = _validate_metric_spline_degree(
            int(array("channel_degree").item())
        )
        modes = np.asarray(array("modes"), dtype=np.float64)
        coefficients = np.asarray(array("coefficients"), dtype=np.complex128)
        expected_coefficients = (3, eta.size, u.size, v.size)
        if coefficients.shape != expected_coefficients:
            raise ValueError(
                "cached channel coefficients have shape "
                f"{coefficients.shape}, expected {expected_coefficients}"
            )
        if modes.shape != (eta.size,) or not np.all(np.isfinite(modes)):
            raise ValueError("cached Fourier modes are invalid")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("cached channel coefficients are nonfinite")

        spline_coefficients = None
        knots_u = knots_v = None
        spline_degree_u = spline_degree_v = None
        if channel_degree > 1:
            spline_coefficients = np.asarray(
                array("spline_coefficients"),
                dtype=np.complex128,
            )
            knots_u = np.asarray(array("knots_u"), dtype=np.float64)
            knots_v = np.asarray(array("knots_v"), dtype=np.float64)
            spline_degree_u = int(array("spline_degree_u").item())
            spline_degree_v = int(array("spline_degree_v").item())
            count_u = len(knots_u) - spline_degree_u - 1
            count_v = len(knots_v) - spline_degree_v - 1
            expected_spline_coefficients = (
                3,
                eta.size,
                count_u,
                count_v,
            )
            if spline_coefficients.shape != expected_spline_coefficients:
                raise ValueError(
                    "cached spline coefficients have shape "
                    f"{spline_coefficients.shape}, expected "
                    f"{expected_spline_coefficients}"
                )
            if not all(
                np.all(np.isfinite(values))
                for values in (spline_coefficients, knots_u, knots_v)
            ):
                raise ValueError("cached spline representation is nonfinite")

        evaluator = cls.__new__(cls)
        evaluator._u = u
        evaluator._v = v
        evaluator._eta = eta
        evaluator._period = period
        evaluator._nfp = nfp
        evaluator._metric_spline_degree = metric_spline_degree
        evaluator._mmpde_result = None
        evaluator._mmpde_fit_scale = float(array("mmpde_fit_scale").item())
        channels = []
        for channel_index in range(3):
            channel = _FourierSplineChannel.__new__(_FourierSplineChannel)
            channel._u = u
            channel._v = v
            channel._degree = channel_degree
            channel._eta0 = float(eta[0])
            channel._period = period
            channel._modes = modes
            channel._coefficients = np.ascontiguousarray(
                coefficients[channel_index]
            )
            channel._spline_coefficients = (
                None
                if spline_coefficients is None
                else np.ascontiguousarray(
                    spline_coefficients[channel_index]
                )
            )
            channel._knots_u = knots_u
            channel._knots_v = knots_v
            channel._spline_degree_u = spline_degree_u
            channel._spline_degree_v = spline_degree_v
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
        if np.any((uf < self._u[0]) | (uf > self._u[-1])) or np.any((vf < self._v[0]) | (vf > self._v[-1])):
            raise ValueError("u and v queries must lie in [0, 1]")
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
        position, A = self._position_and_jacobian(logical_points)
        J = np.linalg.det(A)
        valid = np.isfinite(J) & (J > 0)
        if reject_nonpositive_J and np.any(~valid):
            raise ValueError("query contains nonpositive or nonfinite mesh Jacobian")
        g_cov = np.einsum("...ki,...kj->...ij", A, A)
        g_contra = np.linalg.inv(g_cov)
        residual = np.max(np.abs(np.einsum("...ik,...kj->...ij", g_cov, g_contra) - np.eye(3)), axis=(-2, -1))
        return MetricEvaluation(position, A, J, g_cov, g_contra, residual, valid)

    def sample(self, u: Any, v: Any, eta: Any, *, reject_nonpositive_J: bool = True) -> MetricEvaluation:
        """Evaluate a structured tensor-product logical grid."""
        ug, vg, eg = np.meshgrid(np.asarray(u), np.asarray(v), np.asarray(eta), indexing="ij")
        return self.evaluate(np.stack((ug, vg, eg), axis=-1), reject_nonpositive_J=reject_nonpositive_J)

    def cell_center_logical_points(self) -> np.ndarray:
        """Return logical cell centers, including the periodic eta seam cells."""
        u = 0.5 * (self._u[:-1] + self._u[1:])
        v = 0.5 * (self._v[:-1] + self._v[1:])
        eta = self._eta + 0.5 * self._period / self._eta.size
        return np.stack(np.meshgrid(u, v, eta, indexing="ij"), axis=-1)

    def sample_cell_centers(
        self, *, reject_nonpositive_J: bool = True
    ) -> MetricEvaluation:
        """Evaluate the finite-volume cell-center sampling grid."""
        return self.evaluate(
            self.cell_center_logical_points(),
            reject_nonpositive_J=reject_nonpositive_J,
        )

    def open_boundary_face_center_logical_points(self) -> np.ndarray:
        """Return all D2 boundary-face centers, excluding logical corners."""
        u = 0.5 * (self._u[:-1] + self._u[1:])
        v = 0.5 * (self._v[:-1] + self._v[1:])
        eta = self._eta + 0.5 * self._period / self._eta.size
        faces = [
            np.stack(np.meshgrid([value], v, eta, indexing="ij"), axis=-1)
            for value in (0.0, 1.0)
        ]
        faces.extend(
            np.stack(np.meshgrid(u, [value], eta, indexing="ij"), axis=-1)
            for value in (0.0, 1.0)
        )
        return np.concatenate([face.reshape(-1, 3) for face in faces], axis=0)

    def sample_open_boundary_faces(
        self, *, reject_nonpositive_J: bool = True
    ) -> MetricEvaluation:
        """Evaluate wall-face centers without querying square-chart corners."""
        return self.evaluate(
            self.open_boundary_face_center_logical_points(),
            reject_nonpositive_J=reject_nonpositive_J,
        )

    def quality_report(self) -> MetricQualityReport:
        """Return descriptive quality diagnostics at cells and open faces.

        Cell-scaled quantities use the local logical widths so that cell
        aspect and spacing are reflected in the shape diagnostics.  The
        calculation uses a pseudoinverse for singular samples to keep
        diagnostics available for invalid trial meshes instead of raising on
        a nonpositive Jacobian.
        """
        cell_points = self.cell_center_logical_points()
        face_points = self.open_boundary_face_center_logical_points()
        cell_points_flat = cell_points.reshape(-1, 3)
        all_points = np.concatenate((cell_points_flat, face_points), axis=0)
        all_positions, all_A = self._position_and_jacobian(all_points)
        raw_J = np.linalg.det(all_A)

        def local_width(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
            indices = np.searchsorted(axis, values, side="right") - 1
            indices = np.clip(indices, 0, axis.size - 2)
            return axis[indices + 1] - axis[indices]

        widths = np.stack(
            (
                local_width(self._u, all_points[:, 0]),
                local_width(self._v, all_points[:, 1]),
                np.full(all_points.shape[0], self._period / self._eta.size),
            ),
            axis=1,
        )
        scaled_A = all_A * widths[:, None, :]
        singular_values = np.linalg.svd(scaled_A, compute_uv=False)
        column_norms = np.linalg.norm(scaled_A, axis=1)
        denominator = np.prod(column_norms, axis=1)
        scaled_J = np.full(raw_J.shape, np.nan, dtype=float)
        np.divide(np.linalg.det(scaled_A), denominator, out=scaled_J, where=denominator > 0)
        condition = np.full(raw_J.shape, np.inf, dtype=float)
        positive_sigma = singular_values[:, -1] > 0
        np.divide(
            singular_values[:, 0],
            singular_values[:, -1],
            out=condition,
            where=positive_sigma,
        )
        valid = np.isfinite(raw_J) & (raw_J > 0)

        def finite_stat(values: np.ndarray, reducer: str) -> float:
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return float("nan")
            if reducer == "min":
                return float(np.min(finite))
            if reducer == "max":
                return float(np.max(finite))
            if reducer == "median":
                return float(np.median(finite))
            return float(np.percentile(finite, 1.0))

        raw_median = finite_stat(raw_J, "median")
        min_over_median = (
            float(np.min(raw_J[np.isfinite(raw_J)]) / raw_median)
            if np.isfinite(raw_median) and raw_median != 0 and np.any(np.isfinite(raw_J))
            else float("nan")
        )

        cell_count = cell_points_flat.shape[0]
        cell_J = raw_J[:cell_count].reshape(cell_points.shape[:-1])
        if np.all(np.isfinite(cell_J)) and np.all(cell_J > 0):
            log_J = np.log(cell_J)
            neighbor_candidates: list[tuple[float, str, np.ndarray, np.ndarray]] = []

            def add_neighbors(axis: int, label: str) -> None:
                if log_J.shape[axis] <= 1:
                    return
                differences = np.abs(np.diff(log_J, axis=axis))
                index = np.unravel_index(int(np.argmax(differences)), differences.shape)
                q_a_index = list(index)
                q_b_index = list(index)
                q_b_index[axis] += 1
                q_a = cell_points[tuple(q_a_index)]
                q_b = cell_points[tuple(q_b_index)]
                neighbor_candidates.append((float(differences[index]), label, q_a, q_b))

            add_neighbors(0, "u")
            add_neighbors(1, "v")
            if log_J.shape[2] > 1:
                differences = np.abs(log_J - np.roll(log_J, -1, axis=2))
                index = np.unravel_index(int(np.argmax(differences)), differences.shape)
                q_a = cell_points[index]
                q_b_index = list(index)
                q_b_index[2] = (q_b_index[2] + 1) % log_J.shape[2]
                q_b = cell_points[tuple(q_b_index)]
                seam = index[2] == log_J.shape[2] - 1
                if seam:
                    q_b = q_b.copy()
                    q_b[2] += self._period
                neighbor_candidates.append((
                    float(differences[index]),
                    "eta (periodic seam)" if seam else "eta",
                    q_a,
                    q_b,
                ))
            if neighbor_candidates:
                max_jump, jump_axis, jump_a, jump_b = max(neighbor_candidates, key=lambda item: item[0])
            else:
                max_jump, jump_axis = 0.0, "none"
                jump_a = jump_b = np.full(3, np.nan)
        else:
            max_jump, jump_axis = float("inf"), "none (nonpositive or nonfinite J)"
            jump_a = jump_b = np.full(3, np.nan)

        g_cov = np.einsum("...ki,...kj->...ij", all_A, all_A)
        inverse_residual_max = 0.0
        for metric in g_cov.reshape(-1, 3, 3):
            if np.all(np.isfinite(metric)):
                inverse = np.linalg.pinv(metric)
                residual = np.max(np.abs(metric @ inverse - np.eye(3)))
                inverse_residual_max = max(inverse_residual_max, float(residual))
            else:
                inverse_residual_max = float("inf")
                break

        u_face_count = (self._v.size - 1) * self._eta.size
        v_face_count = (self._u.size - 1) * self._eta.size
        u_min_start = cell_count
        u_max_start = u_min_start + u_face_count
        v_min_start = u_max_start + u_face_count
        v_max_start = v_min_start + v_face_count
        region_slices = (
            ("cell_center", slice(0, cell_count)),
            ("u_min_face", slice(u_min_start, u_max_start)),
            ("u_max_face", slice(u_max_start, v_min_start)),
            ("v_min_face", slice(v_min_start, v_max_start)),
            ("v_max_face", slice(v_max_start, raw_J.size)),
        )

        def location(index: int, value: float, region: str) -> MetricQualityLocation:
            return MetricQualityLocation(
                value=float(value),
                region=region,
                logical=tuple(float(item) for item in all_points[index]),
                cartesian=tuple(float(item) for item in all_positions[index]),
            )

        def extremum(which: str) -> MetricQualityLocation | None:
            values = scaled_J if which == "scaled_J" else condition
            candidates = np.where(~np.isnan(values))[0]
            if not candidates.size:
                return None
            index = int(candidates[np.argmin(values[candidates]) if which == "scaled_J" else np.argmax(values[candidates])])
            region = next(label for label, bounds in region_slices if bounds.start <= index < bounds.stop)
            return location(index, values[index], region)

        regions = []
        for label, bounds in region_slices:
            region_J = raw_J[bounds]
            region_scaled = scaled_J[bounds]
            region_condition = condition[bounds]
            region_valid = np.isfinite(region_J) & (region_J > 0)
            finite_scaled = region_scaled[np.isfinite(region_scaled)]
            condition_candidates = region_condition[~np.isnan(region_condition)]
            regions.append(MetricQualityRegion(
                label=label,
                sample_count=int(region_J.size),
                valid_fraction=float(np.mean(region_valid)) if region_J.size else float("nan"),
                scaled_J_min=float(np.min(finite_scaled)) if finite_scaled.size else float("nan"),
                mapping_condition_max=(
                    float(np.max(condition_candidates))
                    if condition_candidates.size else float("nan")
                ),
            ))

        return MetricQualityReport(
            sample_count=int(raw_J.size),
            valid_fraction=float(np.mean(valid)) if raw_J.size else float("nan"),
            raw_J_min=finite_stat(raw_J, "min"),
            raw_J_p01=finite_stat(raw_J, "p01"),
            raw_J_median=raw_median,
            raw_J_max=finite_stat(raw_J, "max"),
            raw_J_min_over_median=min_over_median,
            scaled_J_min=finite_stat(scaled_J, "min"),
            scaled_J_p01=finite_stat(scaled_J, "p01"),
            mapping_condition_median=finite_stat(condition, "median"),
            mapping_condition_p95=(
                float(np.percentile(condition[np.isfinite(condition)], 95.0))
                if np.any(np.isfinite(condition)) else float("inf")
            ),
            mapping_condition_max=(
                float(np.max(condition)) if condition.size else float("nan")
            ),
            max_neighbor_log_J_jump=max_jump,
            inverse_residual_max=float(inverse_residual_max),
            regions=tuple(regions),
            worst_scaled_jacobian=extremum("scaled_J"),
            worst_mapping_condition=extremum("condition"),
            max_neighbor_log_J_jump_axis=jump_axis,
            max_neighbor_log_J_jump_endpoint_a=tuple(float(item) for item in jump_a),
            max_neighbor_log_J_jump_endpoint_b=tuple(float(item) for item in jump_b),
        )

    def evaluate_magnetic_field(self, logical_points: Any, bfield_evaluator: Any, *, reject_nonpositive_J: bool = True) -> MagneticFieldEvaluation:
        metrics = self.evaluate(logical_points, reject_nonpositive_J=reject_nonpositive_J)
        B = np.asarray(bfield_evaluator.evaluate_cartesian(metrics.position), dtype=np.float64)
        if B.shape != metrics.position.shape or B.shape[-1] != 3 or not np.all(np.isfinite(B)):
            raise ValueError("B-field evaluator must return finite vectors with shape (..., 3)")
        Bcontra = np.linalg.solve(metrics.jacobian_matrix, B[..., None])[..., 0]
        Bcov = np.einsum("...ji,...j->...i", metrics.jacobian_matrix, B)
        return MagneticFieldEvaluation(B, Bcontra, Bcov, np.linalg.norm(B, axis=-1))

    def transform_magnetic_field(self, logical_points: Any, bfield_evaluator: Any, *, reject_nonpositive_J: bool = True) -> MagneticFieldEvaluation:
        """Alias for :meth:`evaluate_magnetic_field`."""
        return self.evaluate_magnetic_field(
            logical_points,
            bfield_evaluator,
            reject_nonpositive_J=reject_nonpositive_J,
        )


def _validate_mesh_shape(mesh_shape: Any) -> tuple[int, int, int]:
    try:
        shape = tuple(int(value) for value in mesh_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("mesh_shape must contain three positive integers") from exc
    if len(shape) != 3 or any(value < 2 for value in shape):
        raise ValueError("mesh_shape must contain three integers of at least two")
    return shape


def _wall_perimeter_indices(nu: int, nv: int) -> list[tuple[int, int]]:
    return (
        [(i, 0) for i in range(nu)]
        + [(nu - 1, j) for j in range(1, nv)]
        + [(i, nv - 1) for i in range(nu - 2, -1, -1)]
        + [(0, j) for j in range(nv - 2, 0, -1)]
    )


def _orient_wall_contour(curve: Any, perimeter: int) -> np.ndarray:
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != (perimeter + 1, 3) or not np.all(np.isfinite(values)):
        raise ValueError("wall boundary curve must have shape (perimeter + 1, 3) and be finite")
    if not np.allclose(values[0], values[-1], rtol=2e-7, atol=2e-9):
        raise ValueError("wall boundary curve must be closed")
    contour = values[:-1].copy()
    if np.any(np.hypot(contour[:, 0], contour[:, 1]) <= 0):
        raise ValueError("wall boundary curve must have positive cylindrical radius")
    radius = np.hypot(contour[:, 0], contour[:, 1])
    signed_area = 0.5 * np.sum(
        radius * np.roll(contour[:, 2], -1)
        - np.roll(radius, -1) * contour[:, 2]
    )
    if abs(signed_area) <= 1e-14:
        raise ValueError("wall boundary curve has degenerate R-Z area")
    # With increasing toroidal eta, a negative R-Z contour orientation gives
    # positive det(dX/d(u,v,eta)) for the Cartesian column convention used by
    # MetricEvaluator and solve_mmpde.
    if signed_area > 0:
        # Preserve WallEvaluator's poloidal phase anchor at contour[0].
        # Reversing the entire array would move that anchor to the end and
        # independently rolling each eta plane would twist mesh connectivity.
        contour = np.concatenate((contour[:1], contour[:0:-1]), axis=0)
    return np.vstack((contour, contour[0]))


def _harmonic_extend_boundaries(
    boundaries: np.ndarray, nu: int, nv: int
) -> np.ndarray:
    """Extend all eta-plane boundaries with one shared Laplacian factorization."""
    curves = np.asarray(boundaries, dtype=np.float64)
    if curves.ndim != 3 or curves.shape[1:] != (
        2 * (nu + nv) - 3,
        3,
    ):
        raise ValueError("boundaries have an incompatible shape")
    neta = curves.shape[0]
    positions = np.empty((nu, nv, neta, 3), dtype=np.float64)
    perimeter_indices = _wall_perimeter_indices(nu, nv)
    boundary_i, boundary_j = np.asarray(perimeter_indices, dtype=int).T
    positions[boundary_i, boundary_j] = np.moveaxis(curves[:, :-1], 0, 1)
    if nu <= 2 or nv <= 2:
        return positions

    interior = [(i, j) for i in range(1, nu - 1) for j in range(1, nv - 1)]
    lookup = {index: number for number, index in enumerate(interior)}
    matrix = lil_matrix((len(interior), len(interior)), dtype=np.float64)
    rhs = np.zeros((len(interior), neta, 3), dtype=np.float64)
    for row, (i, j) in enumerate(interior):
        matrix[row, row] = 4.0
        for neighbor in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if neighbor in lookup:
                matrix[row, lookup[neighbor]] = -1.0
            else:
                rhs[row] += positions[neighbor]
    factorization = splu(matrix.tocsc())
    solved = factorization.solve(rhs.reshape(len(interior), -1)).reshape(
        len(interior), neta, 3
    )
    interior_i, interior_j = np.asarray(interior, dtype=int).T
    positions[interior_i, interior_j] = solved
    return positions


def _harmonic_extend_boundary(boundary: np.ndarray, nu: int, nv: int) -> np.ndarray:
    """Compatibility wrapper for extending one wall contour."""
    return _harmonic_extend_boundaries(
        np.asarray(boundary, dtype=np.float64)[None, ...], nu, nv
    )[:, :, 0]


def build_wall_fitted_initial_mesh(
    eta_evaluator: Any,
    wall_evaluator: Any,
    mesh_shape: tuple[int, int, int],
    logical_axes: tuple[Any, Any, Any] | None = None,
) -> np.ndarray:
    """Build a wall-fitted ``D^2 x S^1`` Cartesian initial mesh.

    The wall contour at each endpoint-exclusive eta plane is assigned to the
    perimeter of ``[0, 1]^2`` in bottom-to-right-to-top-to-left order.  The
    interior Cartesian coordinates are the 5-point discrete harmonic
    extension of that boundary.  The returned array has shape
    ``(nu, nv, neta, 3)``; logical axes are generated internally when omitted.
    """
    nu, nv, neta = _validate_mesh_shape(mesh_shape)
    try:
        period = float(wall_evaluator.period)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("wall_evaluator must provide a positive finite period") from exc
    if not np.isfinite(period) or period <= 0:
        raise ValueError("wall_evaluator.period must be positive and finite")
    wall_nfp = getattr(wall_evaluator, "nfp", None)
    if wall_nfp is None or int(wall_nfp) != wall_nfp or int(wall_nfp) < 1:
        raise ValueError("wall_evaluator must provide a positive integer nfp")
    wall_nfp = int(wall_nfp)
    if not np.isclose(period, 2.0 * np.pi / wall_nfp, rtol=2e-7, atol=2e-10):
        raise ValueError("wall_evaluator period and nfp are inconsistent")
    if getattr(eta_evaluator, "period", None) is not None and not np.isclose(
        float(eta_evaluator.period), period, rtol=2e-7, atol=2e-10
    ):
        raise ValueError("eta evaluator period is inconsistent with the wall")
    if getattr(eta_evaluator, "nfp", None) is not None and int(eta_evaluator.nfp) != wall_nfp:
        raise ValueError("eta evaluator nfp is inconsistent with the wall")

    if logical_axes is None:
        axes = (
            np.linspace(0.0, 1.0, nu),
            np.linspace(0.0, 1.0, nv),
            np.arange(neta, dtype=np.float64) * period / neta,
        )
    else:
        if len(logical_axes) != 3:
            raise ValueError("logical_axes must contain (u, v, eta)")
        axes = tuple(np.asarray(axis, dtype=np.float64) for axis in logical_axes)
        if any(axis.ndim != 1 or axis.size != count or not np.all(np.isfinite(axis)) for axis, count in zip(axes, mesh_shape)):
            raise ValueError("logical axes must match mesh_shape and be finite one-dimensional arrays")
        if not np.isclose(axes[0][0], 0.0) or not np.isclose(axes[0][-1], 1.0):
            raise ValueError("u must span [0, 1]")
        if not np.isclose(axes[1][0], 0.0) or not np.isclose(axes[1][-1], 1.0):
            raise ValueError("v must span [0, 1]")
        if any(np.any(np.diff(axis) <= 0) for axis in axes):
            raise ValueError("logical axes must be strictly increasing")
        deta = np.diff(axes[2])
        if not np.allclose(deta, deta[0], rtol=2e-10, atol=2e-12) or not np.isclose(
            deta[0] * neta, period, rtol=2e-9, atol=2e-11
        ):
            raise ValueError("eta must be uniformly spaced over one endpoint-exclusive wall period")
    perimeter = 2 * (nu + nv) - 4
    curves = []
    for eta in axes[2]:
        curve = wall_evaluator.constant_eta_boundary_curve(
            eta_evaluator, float(eta), npoints=perimeter + 1
        )
        curves.append(_orient_wall_contour(curve, perimeter))
    rotation = np.array(
        [[np.cos(period), -np.sin(period), 0.0],
         [np.sin(period), np.cos(period), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    seam_curve = _orient_wall_contour(
        wall_evaluator.constant_eta_boundary_curve(
            eta_evaluator, float(axes[2][0] + period), npoints=perimeter + 1
        ),
        perimeter,
    )
    expected_seam = curves[0] @ rotation.T
    if not np.allclose(seam_curve, expected_seam, rtol=2e-5, atol=2e-7):
        raise ValueError("wall boundary curves fail rotational field-period seam closure")
    positions = _harmonic_extend_boundaries(np.asarray(curves), nu, nv)
    if not np.all(np.isfinite(positions)) or np.any(np.hypot(positions[..., 0], positions[..., 1]) <= 0):
        raise ValueError("wall-fitted initial mesh contains invalid Cartesian positions")
    return positions


def build_metric_evaluator(
    eta_evaluator: Any,
    initial_positions: Any | None = None,
    *,
    wall_evaluator: Any | None = None,
    mesh_shape: tuple[int, int, int] | None = None,
    logical_axes: tuple[Any, Any, Any] | None = None,
    monitor: Any | None = None,
    fixed_mask: Any | None = None,
    options: MMPDEOptions | None = None,
    projector: Any | None = None,
    metric_spline_degree: int | None = None,
) -> MetricEvaluator:
    """Solve an eta-constrained mesh and return its smooth metric evaluator.

    Supply either an explicit ``initial_positions`` array or a
    ``wall_evaluator`` together with ``mesh_shape``. In wall mode, constant-eta
    vessel contours define the physical D^2 boundary and a discrete harmonic
    extension supplies the initial interior. The first two logical axes are
    non-periodic and default to ``[0, 1]``; eta defaults to a uniform
    endpoint-exclusive axis over ``eta_evaluator.period``.
    ``metric_spline_degree`` defaults to 3 for explicit-position mode and to
    the safer degree 1 in wall mode.  Exact logical-square corners are not
    valid metric query points because wall-fitted corners can be nonsmooth.

    The internal projector applies an optional user projector followed by
    Newton projection onto the requested eta level.  The Newton correction is
    along ``gradient_cartesian(positions)`` and uses a periodic eta residual.
    No R-Z box is sampled by the metric fit: it is fit only to the solved
    structured mesh nodes.
    """
    if initial_positions is not None and (wall_evaluator is not None or mesh_shape is not None):
        raise ValueError("provide either initial_positions or wall_evaluator with mesh_shape, not both")
    if (wall_evaluator is None) != (mesh_shape is None):
        raise ValueError("wall_evaluator and mesh_shape must be provided together")
    wall_mode = wall_evaluator is not None
    selected_spline_degree = (
        1 if wall_mode and metric_spline_degree is None else
        3 if metric_spline_degree is None else metric_spline_degree
    )
    selected_spline_degree = _validate_metric_spline_degree(selected_spline_degree)
    if wall_mode:
        positions = build_wall_fitted_initial_mesh(
            eta_evaluator, wall_evaluator, mesh_shape, logical_axes=logical_axes
        )
    elif initial_positions is None:
        raise ValueError("provide initial_positions or wall_evaluator with mesh_shape")
    else:
        positions = np.asarray(initial_positions, dtype=np.float64)
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError("initial_positions must have shape (nu, nv, neta, 3)")
    if min(positions.shape[:3]) < 2 or not np.all(np.isfinite(positions)):
        raise ValueError("initial_positions must be finite with at least two nodes per axis")

    try:
        period = float(eta_evaluator.period)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("eta_evaluator must provide a positive finite period") from exc
    if not np.isfinite(period) or period <= 0:
        raise ValueError("eta_evaluator.period must be positive and finite")

    # Preserve field-period metadata when the eta evaluator provides it, but
    # do not require it: period-only evaluators remain supported.
    eta_nfp = getattr(eta_evaluator, "nfp", None)
    if eta_nfp is not None:
        if isinstance(eta_nfp, (bool, np.bool_)):
            raise ValueError("eta_evaluator.nfp must be a positive integer")
        try:
            eta_nfp_int = int(eta_nfp)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("eta_evaluator.nfp must be a positive integer") from exc
        if eta_nfp_int != eta_nfp or eta_nfp_int < 1:
            raise ValueError("eta_evaluator.nfp must be a positive integer")
        implied_period = 2.0 * np.pi / eta_nfp_int
        if not np.isclose(period, implied_period, rtol=2e-12, atol=2e-12):
            raise ValueError("eta_evaluator.period and eta_evaluator.nfp are inconsistent")
    else:
        eta_nfp_int = None

    shape = positions.shape[:3]
    if logical_axes is None:
        axes = (
            np.linspace(0.0, 1.0, shape[0]),
            np.linspace(0.0, 1.0, shape[1]),
            np.arange(shape[2], dtype=np.float64) * period / shape[2],
        )
    else:
        if len(logical_axes) != 3:
            raise ValueError("logical_axes must contain (u, v, eta)")
        axes = tuple(np.asarray(axis, dtype=np.float64) for axis in logical_axes)
        if any(axis.ndim != 1 or axis.size != n or not np.all(np.isfinite(axis)) for axis, n in zip(axes, shape)):
            raise ValueError("logical axes must be finite one-dimensional arrays matching initial_positions")
        if any(np.any(np.diff(axis) <= 0) for axis in axes):
            raise ValueError("logical axes must be strictly increasing")
        if not np.isclose(axes[0][0], 0.0) or not np.isclose(axes[0][-1], 1.0):
            raise ValueError("u must span [0, 1]")
        if not np.isclose(axes[1][0], 0.0) or not np.isclose(axes[1][-1], 1.0):
            raise ValueError("v must span [0, 1]")
        deta = np.diff(axes[2])
        if not np.allclose(deta, deta[0], rtol=2e-10, atol=2e-12) or not np.isclose(
            deta[0] * shape[2], period, rtol=2e-9, atol=2e-11
        ):
            raise ValueError("eta must be uniformly spaced over one endpoint-exclusive period")

    u, v, eta = axes
    target_eta = eta[None, None, :]
    if fixed_mask is None:
        fixed = np.zeros(shape, dtype=bool)
        fixed[0, :, :] = fixed[-1, :, :] = True
        fixed[:, 0, :] = fixed[:, -1, :] = True
    else:
        fixed = np.asarray(fixed_mask, dtype=bool)
        if fixed.shape != shape:
            raise ValueError("fixed_mask must have shape (nu, nv, neta)")
    if wall_mode:
        fixed[0, :, :] = fixed[-1, :, :] = True
        fixed[:, 0, :] = fixed[:, -1, :] = True

    def periodic_image(points: np.ndarray, turns: int) -> np.ndarray:
        angle = float(turns) * period
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return np.asarray(points, dtype=np.float64) @ rotation.T

    def eta_projector(candidate: np.ndarray) -> np.ndarray:
        projected = np.asarray(candidate, dtype=np.float64).copy()
        if projector is not None:
            projected = np.asarray(projector(projected.copy()), dtype=np.float64)
            if projected.shape != positions.shape or not np.all(np.isfinite(projected)):
                raise ValueError("projector must return finite positions with the input shape")
        flat_all = projected.reshape(-1, 3)
        free_flat = ~fixed.reshape(-1)
        if not np.any(free_flat):
            return projected
        flat = flat_all[free_flat].copy()
        targets = np.broadcast_to(target_eta, shape).reshape(-1)[free_flat]
        for _ in range(12):
            combined = getattr(
                eta_evaluator, "evaluate_and_gradient_cartesian", None
            )
            if combined is None:
                values = np.asarray(
                    eta_evaluator.evaluate_cartesian(flat, wrapped=True),
                    dtype=np.float64,
                ).reshape(-1)
                gradients = None
            else:
                values, gradients = combined(flat, wrapped=True)
                values = np.asarray(values, dtype=np.float64).reshape(-1)
                gradients = np.asarray(gradients, dtype=np.float64).reshape(-1, 3)
            residual = (values - targets + 0.5 * period) % period - 0.5 * period
            if np.max(np.abs(residual)) <= max(1.0e-11, 1.0e-10 * period):
                break
            if gradients is None:
                gradients = np.asarray(
                    eta_evaluator.gradient_cartesian(flat), dtype=np.float64
                ).reshape(-1, 3)
            denominator = np.einsum("ij,ij->i", gradients, gradients)
            if gradients.shape != flat.shape or not np.all(np.isfinite(gradients)) or np.any(denominator <= 1.0e-24):
                raise ValueError("eta_evaluator gradient is invalid or vanishes during Newton projection")
            flat -= (residual / denominator)[:, None] * gradients
            if not np.all(np.isfinite(flat)):
                raise ValueError("eta projection produced nonfinite positions")
        else:
            raise ValueError("eta projection did not converge")
        flat_all[free_flat] = flat
        return flat_all.reshape(shape + (3,))

    projected_initial = eta_projector(positions)
    projected_initial[fixed] = positions[fixed]
    initial_values = np.asarray(
        eta_evaluator.evaluate_cartesian(projected_initial.reshape(-1, 3), wrapped=True), dtype=np.float64
    ).reshape(shape)
    initial_residual = (initial_values - target_eta + 0.5 * period) % period - 0.5 * period
    if np.max(np.abs(initial_residual)) > max(5.0e-9, 5.0e-8 * period):
        raise ValueError("fixed boundary nodes are incompatible with the requested eta levels")

    result = solve_mmpde(
        projected_initial,
        logical_axes=(u, v, eta),
        monitor=monitor,
        fixed_mask=fixed,
        projector=eta_projector,
        periodic_image=periodic_image,
        options=options,
    )
    final_values = np.asarray(
        eta_evaluator.evaluate_cartesian(result.positions.reshape(-1, 3), wrapped=True), dtype=np.float64
    ).reshape(shape)
    final_residual = (final_values - target_eta + 0.5 * period) % period - 0.5 * period
    if np.max(np.abs(final_residual)) > max(5.0e-9, 5.0e-8 * period):
        raise ValueError("MMPDE result does not satisfy the requested eta levels")
    raw_mmpde_positions = result.positions.copy()
    accepted_positions = None
    accepted_scale = None
    accepted_evaluator = None
    for scale in [2.0**-level for level in range(21)] + [0.0]:
        if scale == 1.0:
            candidate = raw_mmpde_positions
        elif scale == 0.0:
            candidate = projected_initial
        else:
            candidate = eta_projector(
                projected_initial
                + scale * (raw_mmpde_positions - projected_initial)
            )
            candidate[fixed] = positions[fixed]
        trial = MetricEvaluator(
            u,
            v,
            eta,
            candidate,
            period=period,
            nfp=eta_nfp_int,
            metric_spline_degree=selected_spline_degree,
        )
        validation_points = np.concatenate(
            (
                trial.cell_center_logical_points().reshape(-1, 3),
                trial.open_boundary_face_center_logical_points(),
            ),
            axis=0,
        )
        validation_metrics = trial.evaluate(
            validation_points,
            reject_nonpositive_J=False
        )
        minimum_sampled_jacobian = float(
            np.min(validation_metrics.signed_J)
        )
        if np.isfinite(minimum_sampled_jacobian) and minimum_sampled_jacobian > 0:
            accepted_positions = candidate.copy()
            accepted_scale = scale
            accepted_evaluator = trial
            break
    if (
        accepted_positions is None
        or accepted_scale is None
        or accepted_evaluator is None
    ):
        raise ValueError(
            "fitted metric has a nonpositive Jacobian at a cell center or "
            "open boundary-face center"
        )
    result.positions = accepted_positions
    accepted_evaluator._mmpde_result = result
    accepted_evaluator._mmpde_fit_scale = accepted_scale
    return accepted_evaluator


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
    "MetricQualityLocation",
    "MetricQualityRegion",
    "MetricQualityReport",
    "MetricEvaluator",
    "build_metric_evaluator",
    "build_wall_fitted_initial_mesh",
]
