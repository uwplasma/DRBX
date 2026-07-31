"""Periodic spline representation of a structured three-dimensional mesh.

The logical topology represented here is ``D^2 x S^1``.  The first two
coordinates are non-periodic coordinates on ``[0, 1]^2`` and the last one is
an endpoint-exclusive, unwrapped toroidal coordinate.  The embedding is
represented by periodic Fourier coefficients in the toroidal coordinate and
by tensor-product splines in the two disk coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

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
    scaled_J_p01: float = float("nan")
    mapping_condition_p95: float = float("nan")
    stretch_p95: float = float("nan")
    stretch_max: float = float("nan")
    volume_p01_over_median: float = float("nan")
    volume_p99_over_p01: float = float("nan")
    nonpositive_J_count: int = 0


@dataclass(frozen=True)
class MetricQualityLocation:
    """Location of an extremal quality diagnostic."""

    value: float
    region: str
    logical: tuple[float, float, float]
    cartesian: tuple[float, float, float]
    cell_index: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class MetricQualityJumpLocation:
    """Location of a directional neighbour-coefficient jump."""

    value: float
    direction: str
    logical_a: tuple[float, float, float]
    logical_b: tuple[float, float, float]
    cartesian_a: tuple[float, float, float]
    cartesian_b: tuple[float, float, float]
    cell_index_a: tuple[int, int, int]
    cell_index_b: tuple[int, int, int]
    region_a: str = "unknown"
    region_b: str = "unknown"


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
    points_per_cell: int = 1
    cell_count: int = 0
    gauss_order: int = 1
    quadrature_sample_count: int = 0
    face_sample_count: int = 0
    nonpositive_J_count: int = 0
    nonpositive_J_fraction: float = float("nan")
    face_nonpositive_J_count: int = 0
    face_valid_fraction: float = float("nan")
    raw_volume_min: float = float("nan")
    volume_min_over_median: float = float("nan")
    volume_p01_over_median: float = float("nan")
    volume_p05_over_median: float = float("nan")
    volume_p95_over_median: float = float("nan")
    volume_p99_over_median: float = float("nan")
    volume_p99_over_p01: float = float("nan")
    volume_coefficient_of_variation: float = float("nan")
    scaled_J_p05: float = float("nan")
    scaled_J_median: float = float("nan")
    mapping_condition_p99: float = float("nan")
    stretch_median: float = float("nan")
    stretch_p95: float = float("nan")
    stretch_p99: float = float("nan")
    stretch_max: float = float("nan")
    inverse_residual_p99: float = float("nan")
    angle_cosines: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    directional_log_volume_jumps: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    directional_K_jumps: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    eta_constraint_residuals: Mapping[str, float] = field(default_factory=dict)
    periodic_seam_residuals: Mapping[str, float] = field(default_factory=dict)
    representation_metadata: Mapping[str, Any] = field(default_factory=dict)
    mmpde_metadata: Mapping[str, Any] = field(default_factory=dict)
    worst_jacobian: MetricQualityLocation | None = None
    worst_volume: MetricQualityLocation | None = None
    worst_stretch: MetricQualityLocation | None = None
    worst_angle_cosine: MetricQualityLocation | None = None
    worst_eta_constraint: MetricQualityLocation | None = None
    worst_volume_jump: MetricQualityJumpLocation | None = None
    worst_K_jump: MetricQualityJumpLocation | None = None

    @property
    def volume_min(self) -> float:
        """Raw minimum signed physical cell-volume measure."""

        return self.raw_volume_min

    @property
    def positive_J_fraction(self) -> float:
        """Fraction of interior quadrature samples with finite positive J."""

        return self.valid_fraction

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Backward-friendly short alias for :attr:`representation_metadata`."""
        return self.representation_metadata

    @property
    def eta_constraint(self) -> Mapping[str, float]:
        """Short alias for eta-surface residual statistics."""
        return self.eta_constraint_residuals

    @property
    def periodic_seam(self) -> Mapping[str, float]:
        """Short alias for periodic seam residual statistics."""
        return self.periodic_seam_residuals

    @property
    def directional_log_J_jumps(self) -> Mapping[str, tuple[float, float]]:
        """Legacy-named alias for cell-volume log-jump diagnostics.

        The old report called this quantity ``log_J`` because it sampled raw
        Jacobians at centres.  The expanded report intentionally compares the
        physical cell-volume measure ``J Δu Δv Δη`` instead.
        """
        return self.directional_log_volume_jumps

    def summary(self, label: str | None = None) -> str:
        prefix = f"{label}: " if label else ""
        return (
            f"{prefix}samples={self.sample_count} (quad={self.quadrature_sample_count}, "
            f"face={self.face_sample_count}, per_cell={self.points_per_cell}), "
            f"valid={self.valid_fraction:.3f}, "
            f"J=[{self.raw_J_min:.6e}, {self.raw_J_max:.6e}], "
            f"J_p01={self.raw_J_p01:.6e}, J_med={self.raw_J_median:.6e}, "
            f"J_min/med={self.raw_J_min_over_median:.6e}, "
            f"scaled_J=[{self.scaled_J_min:.6e}, p01={self.scaled_J_p01:.6e}, "
            f"p05={self.scaled_J_p05:.6e}], "
            f"cond(H)=[med={self.mapping_condition_median:.6e}, "
            f"p95={self.mapping_condition_p95:.6e}, p99={self.mapping_condition_p99:.6e}, "
            f"max={self.mapping_condition_max:.6e}], "
            f"max_dlogV={self.max_neighbor_log_J_jump:.6e}, "
            f"inverse_residual_max={self.inverse_residual_max:.6e}"
        )

    def detailed_summary(self, label: str | None = None) -> str:
        """Return readable region and extremum diagnostics.

        This intentionally formats NaN and infinity rather than applying
        thresholds, so it remains useful for inverted or singular trial maps.
        """
        prefix = f"{label}:\n" if label else ""
        lines = [prefix.rstrip("\n")] if prefix else []
        metadata = self.representation_metadata
        lines.extend((
            "  Representation:",
            f"    mesh shape={metadata.get('mesh_shape', 'unknown')}, "
            f"cell counts={metadata.get('cell_counts', 'unknown')}, "
            f"evaluator cells={metadata.get('evaluator_cell_counts', 'unknown')}, "
            f"spline=(u:{metadata.get('spline_degree_u', 'unknown')}, "
            f"v:{metadata.get('spline_degree_v', 'unknown')}), "
            f"eta={metadata.get('eta_representation', 'unknown')}, "
            f"MMPDE fit scale={metadata.get('mmpde_fit_scale', float('nan'))}",
            "  Sampling:",
            f"    cells={self.cell_count}, Gauss order={self.gauss_order}, "
            f"points/cell={self.points_per_cell}, quadrature={self.quadrature_sample_count}, "
            f"wall faces={self.face_sample_count}, total={self.sample_count}, "
            f"nonpositive interior J={self.nonpositive_J_count} "
            f"({self.nonpositive_J_fraction:.6e}), "
            f"nonpositive wall-face J={self.face_nonpositive_J_count}",
            "  Validity:",
            f"    J_min={self.raw_J_min:.6e}, valid fraction={self.valid_fraction:.6e}, "
            f"wall-face valid fraction={self.face_valid_fraction:.6e}, "
            f"inverse-metric residual p99/max={self.inverse_residual_p99:.6e}/"
            f"{self.inverse_residual_max:.6e}",
            "  Volume:",
            f"    raw min={self.raw_volume_min:.6e}, "
            f"min/med={self.volume_min_over_median:.6e}, "
            f"p01/med={self.volume_p01_over_median:.6e}, "
            f"p05/med={self.volume_p05_over_median:.6e}, p95/med={self.volume_p95_over_median:.6e}, "
            f"p99/med={self.volume_p99_over_median:.6e}, p99/p01={self.volume_p99_over_p01:.6e}, "
            f"cv={self.volume_coefficient_of_variation:.6e}",
            "  Shape:",
            f"    scaled J=min {self.scaled_J_min:.6e}, p01 {self.scaled_J_p01:.6e}, "
            f"p05 {self.scaled_J_p05:.6e}, median {self.scaled_J_median:.6e}",
            f"    cond(H)=median {self.mapping_condition_median:.6e}, p95 {self.mapping_condition_p95:.6e}, "
            f"p99 {self.mapping_condition_p99:.6e}, max {self.mapping_condition_max:.6e}",
            f"    stretch=median {self.stretch_median:.6e}, p95 {self.stretch_p95:.6e}, "
            f"p99 {self.stretch_p99:.6e}, max {self.stretch_max:.6e}",
        ))
        for name, (p95, maximum) in self.angle_cosines.items():
            lines.append(f"    |cos({name})|=p95 {p95:.6e}, max {maximum:.6e}")
        lines.append("  Smoothness:")
        for direction in sorted(set(self.directional_log_volume_jumps) | set(self.directional_K_jumps)):
            dv = self.directional_log_volume_jumps.get(direction, (float("nan"), float("nan")))
            dk = self.directional_K_jumps.get(direction, (float("nan"), float("nan")))
            lines.append(f"    {direction}: dlog(V) p95/max={dv[0]:.6e}/{dv[1]:.6e}, "
                         f"dK p95/max={dk[0]:.6e}/{dk[1]:.6e}")
        lines.append("  Regions:")
        for region in self.regions:
            lines.append(
                f"  {region.label}: samples={region.sample_count}, "
                f"nonpositive_J={region.nonpositive_J_count}, "
                f"valid={region.valid_fraction:.3f}, "
                f"scaled_J_min={region.scaled_J_min:.6e}, "
                f"scaled_J_p01={region.scaled_J_p01:.6e}, "
                f"cond(H)_p95/max={region.mapping_condition_p95:.6e}/{region.mapping_condition_max:.6e}, "
                f"stretch_p95/max={region.stretch_p95:.6e}/{region.stretch_max:.6e}, "
                f"V_p01/med={region.volume_p01_over_median:.6e}, "
                f"V_p99/p01={region.volume_p99_over_p01:.6e}"
            )

        def location_line(name: str, location: MetricQualityLocation | None) -> str:
            if location is None:
                return f"  {name}: value=nan, region=none, q=(nan, nan, nan), x=(nan, nan, nan)"
            q = ", ".join(f"{value:.6e}" for value in location.logical)
            x = ", ".join(f"{value:.6e}" for value in location.cartesian)
            return (
                f"  {name}: value={location.value:.6e}, "
                f"cell={location.cell_index}, region={location.region}, "
                f"q=({q}), x=({x})"
            )

        lines.append("  Constraints:")
        if self.eta_constraint_residuals:
            lines.append("    eta residual: " + ", ".join(
                f"{key}={value:.6e}" for key, value in self.eta_constraint_residuals.items()))
        if self.periodic_seam_residuals:
            lines.append("    periodic seam: " + ", ".join(
                f"{key}={value:.6e}" for key, value in self.periodic_seam_residuals.items()))
        lines.append("  Worst locations:")
        lines.append(location_line("worst_J", self.worst_jacobian))
        lines.append(location_line("worst_volume", self.worst_volume))
        lines.append(location_line("worst_scaled_J", self.worst_scaled_jacobian))
        lines.append(location_line("worst_cond(H)", self.worst_mapping_condition))
        lines.append(location_line("worst_stretch", self.worst_stretch))
        lines.append(location_line("worst_angle", self.worst_angle_cosine))
        lines.append(location_line("worst_eta_residual", self.worst_eta_constraint))
        a = ", ".join(f"{value:.6e}" for value in self.max_neighbor_log_J_jump_endpoint_a)
        b = ", ".join(f"{value:.6e}" for value in self.max_neighbor_log_J_jump_endpoint_b)
        lines.append(
            f"  max_dlogV: value={self.max_neighbor_log_J_jump:.6e}, "
            f"axis={self.max_neighbor_log_J_jump_axis}, q_a=({a}), q_b=({b})"
        )
        for name, jump in (("worst_volume_jump", self.worst_volume_jump), ("worst_K_jump", self.worst_K_jump)):
            if jump is not None:
                qa = ", ".join(f"{value:.6e}" for value in jump.logical_a)
                qb = ", ".join(f"{value:.6e}" for value in jump.logical_b)
                xa = ", ".join(f"{value:.6e}" for value in jump.cartesian_a)
                xb = ", ".join(f"{value:.6e}" for value in jump.cartesian_b)
                lines.append(
                    f"  {name}: value={jump.value:.6e}, direction={jump.direction}, "
                    f"cells={jump.cell_index_a}->{jump.cell_index_b}, "
                    f"regions={jump.region_a}->{jump.region_b}, "
                    f"q_a=({qa}), q_b=({qb}), x_a=({xa}), x_b=({xb})"
                )
        if self.mmpde_metadata:
            lines.append("  Optimization:")
            lines.append("    " + ", ".join(
                f"{key}={value}" for key, value in self.mmpde_metadata.items()))
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

    def quality_report(
        self,
        *,
        eta_evaluator: Any | None = None,
        gauss_order: int = 2,
        logical_cell_counts: tuple[int, int, int] | None = None,
    ) -> MetricQualityReport:
        """Return cell-resolved quality diagnostics for the fitted metric map.

        ``gauss_order`` is deliberately restricted to 2 or 3.  Two points in
        every logical direction is the normal benchmark mode; three is a more
        expensive validation mode for detecting spline-cell defects missed by
        centre samples.  Open wall-face centres are retained as supplemental
        samples, but volume/shape region statistics use quadrature points.

        ``logical_cell_counts`` optionally selects the uniform logical grid
        whose cell quality is being assessed.  This is useful when the
        evaluator's stored MMPDE node grid differs from the final PDE grid:
        the evaluator remains the source of the spline/Fourier map, while the
        report samples the requested computational cells.  If omitted, the
        evaluator's own node axes are used for backwards compatibility.
        """
        if gauss_order not in (2, 3):
            raise ValueError("gauss_order must be 2 or 3")
        if logical_cell_counts is not None:
            try:
                requested_counts = tuple(int(value) for value in logical_cell_counts)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "logical_cell_counts must contain three positive integers"
                ) from error
            if len(requested_counts) != 3 or any(value < 1 for value in requested_counts):
                raise ValueError(
                    "logical_cell_counts must contain three positive integers"
                )
        else:
            requested_counts = None
        if eta_evaluator is not None:
            eta_period = getattr(eta_evaluator, "period", None)
            if eta_period is None or not np.isfinite(eta_period) or eta_period <= 0.0:
                raise ValueError("eta_evaluator must provide a positive finite period")
            if not np.isclose(float(eta_period), self._period, rtol=2e-7, atol=2e-10):
                raise ValueError("eta_evaluator period is inconsistent with the metric period")
        nodes = (
            np.array((-1.0, 1.0), dtype=float) / np.sqrt(3.0)
            if gauss_order == 2 else np.array((-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)))
        )
        if requested_counts is None:
            u_edges = np.asarray(self._u, dtype=float)
            v_edges = np.asarray(self._v, dtype=float)
            eta_edges = np.concatenate(
                (np.asarray(self._eta, dtype=float),
                 [float(self._eta[0]) + self._period])
            )
        else:
            nu, nv, neta = requested_counts
            u_edges = np.linspace(self._u[0], self._u[-1], nu + 1)
            v_edges = np.linspace(self._v[0], self._v[-1], nv + 1)
            eta_edges = (
                float(self._eta[0])
                + np.arange(neta + 1, dtype=float) * self._period / neta
            )
        du, dv, deta = np.diff(u_edges), np.diff(v_edges), np.diff(eta_edges)
        uc = 0.5 * (u_edges[:-1] + u_edges[1:])
        vc = 0.5 * (v_edges[:-1] + v_edges[1:])
        ec = 0.5 * (eta_edges[:-1] + eta_edges[1:])
        offsets = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
        points_by_cell = np.empty((uc.size, vc.size, ec.size, offsets.shape[0], 3), dtype=float)
        points_by_cell[..., 0] = uc[:, None, None, None] + 0.5 * du[:, None, None, None] * offsets[None, None, None, :, 0]
        points_by_cell[..., 1] = vc[None, :, None, None] + 0.5 * dv[None, :, None, None] * offsets[None, None, None, :, 1]
        points_by_cell[..., 2] = (
            ec[None, None, :, None]
            + 0.5 * deta[None, None, :, None] * offsets[None, None, None, :, 2]
        )
        quadrature_points = points_by_cell.reshape(-1, 3)
        face_points = np.concatenate(
            (
                np.stack(
                    np.meshgrid([u_edges[0]], vc, ec, indexing="ij"),
                    axis=-1,
                ).reshape(-1, 3),
                np.stack(
                    np.meshgrid([u_edges[-1]], vc, ec, indexing="ij"),
                    axis=-1,
                ).reshape(-1, 3),
                np.stack(
                    np.meshgrid(uc, [v_edges[0]], ec, indexing="ij"),
                    axis=-1,
                ).reshape(-1, 3),
                np.stack(
                    np.meshgrid(uc, [v_edges[-1]], ec, indexing="ij"),
                    axis=-1,
                ).reshape(-1, 3),
            ),
            axis=0,
        )
        all_points = np.concatenate((quadrature_points, face_points), axis=0)

        def local_width(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
            indices = np.clip(np.searchsorted(axis, values, side="right") - 1, 0, axis.size - 2)
            return axis[indices + 1] - axis[indices]

        # A 64^3 benchmark has more than two million quadrature samples.
        # Retain scalar diagnostics, but process Jacobian matrices in chunks
        # so the report does not transiently hold several multi-hundred-MiB
        # arrays of positions, F, H, metrics, and singular vectors at once.
        sample_count = all_points.shape[0]
        raw_J = np.empty(sample_count, dtype=float)
        determinant_H = np.empty(sample_count, dtype=float)
        scaled_J = np.empty(sample_count, dtype=float)
        condition = np.empty(sample_count, dtype=float)
        stretch = np.empty(sample_count, dtype=float)
        angle_cosines = {name: np.empty(sample_count, dtype=float) for name in ("uv", "ueta", "veta")}
        inverse_residual = np.empty(sample_count, dtype=float)
        eta_absolute = np.empty(quadrature_points.shape[0], dtype=float) if eta_evaluator is not None else None
        chunk_size = 131_072
        for start in range(0, sample_count, chunk_size):
            stop = min(start + chunk_size, sample_count)
            points = all_points[start:stop]
            positions, A = self._position_and_jacobian(points)
            widths = np.stack((local_width(u_edges, points[:, 0]), local_width(v_edges, points[:, 1]), np.full(points.shape[0], deta[0])), axis=1)
            H = A * widths[:, None, :]
            singular_values = np.linalg.svd(H, compute_uv=False)
            lengths = np.linalg.norm(H, axis=1)
            raw_J[start:stop] = np.linalg.det(A)
            determinant_H[start:stop] = np.linalg.det(H)
            denominator = np.prod(lengths, axis=1)
            scaled_J[start:stop] = np.nan
            np.divide(determinant_H[start:stop], denominator, out=scaled_J[start:stop], where=denominator > 0.0)
            condition[start:stop] = np.inf
            np.divide(singular_values[:, 0], singular_values[:, -1], out=condition[start:stop], where=singular_values[:, -1] > 0.0)
            stretch[start:stop] = np.inf
            np.divide(np.max(lengths, axis=1), np.min(lengths, axis=1), out=stretch[start:stop], where=np.min(lengths, axis=1) > 0.0)
            angle_cosines["uv"][start:stop] = np.abs(np.sum(H[:, :, 0] * H[:, :, 1], axis=1)) / np.maximum(lengths[:, 0] * lengths[:, 1], np.finfo(float).tiny)
            angle_cosines["ueta"][start:stop] = np.abs(np.sum(H[:, :, 0] * H[:, :, 2], axis=1)) / np.maximum(lengths[:, 0] * lengths[:, 2], np.finfo(float).tiny)
            angle_cosines["veta"][start:stop] = np.abs(np.sum(H[:, :, 1] * H[:, :, 2], axis=1)) / np.maximum(lengths[:, 1] * lengths[:, 2], np.finfo(float).tiny)
            metric = np.einsum("...ki,...kj->...ij", A, A)
            inverse = np.linalg.pinv(metric)
            inverse_residual[start:stop] = np.max(np.abs(metric @ inverse - np.eye(3)), axis=(-2, -1))
            if eta_absolute is not None and start < quadrature_points.shape[0]:
                eta_stop = min(stop, quadrature_points.shape[0])
                try:
                    eta_values = np.asarray(eta_evaluator.evaluate_cartesian(positions[:eta_stop - start], wrapped=False), dtype=float)
                except TypeError:
                    eta_values = np.asarray(eta_evaluator.evaluate_cartesian(positions[:eta_stop - start]), dtype=float)
                expected_shape = (eta_stop - start,)
                if eta_values.shape != expected_shape:
                    raise ValueError(
                        "eta_evaluator must return one value per Cartesian point"
                    )
                if not np.all(np.isfinite(eta_values)):
                    raise ValueError("eta_evaluator returned nonfinite values")
                eta_residual = (eta_values - quadrature_points[start:eta_stop, 2] + 0.5 * self._period) % self._period - 0.5 * self._period
                eta_absolute[start:eta_stop] = np.abs(eta_residual)
        quadrature_count = quadrature_points.shape[0]
        quadrature = slice(0, quadrature_count)
        face_slice = slice(quadrature_count, sample_count)
        quadrature_valid = (
            np.isfinite(raw_J[quadrature]) & (raw_J[quadrature] > 0.0)
        )
        face_valid = (
            np.isfinite(raw_J[face_slice]) & (raw_J[face_slice] > 0.0)
        )
        cell_shape = (uc.size, vc.size, ec.size)
        q_per_cell = offsets.shape[0]
        cell_indices = np.stack(np.meshgrid(np.arange(uc.size), np.arange(vc.size), np.arange(ec.size), indexing="ij"), axis=-1).reshape(-1, 3)
        sample_cell_indices = np.repeat(cell_indices, q_per_cell, axis=0)

        def finite(values: np.ndarray) -> np.ndarray:
            return np.asarray(values)[np.isfinite(values)]

        def stat(values: np.ndarray, kind: str, percentile: float | None = None) -> float:
            raw_values = np.asarray(values).reshape(-1)
            non_nan = raw_values[~np.isnan(raw_values)]
            if non_nan.size == 0:
                return float("nan")
            # Preserve infinities for extrema so a singular mapping remains
            # visible instead of being silently omitted from the report.
            if kind == "min": return float(np.min(non_nan))
            if kind == "max": return float(np.max(non_nan))
            values = finite(non_nan)
            if values.size == 0:
                return float("nan")
            if kind == "median": return float(np.median(values))
            if kind == "mean": return float(np.mean(values))
            if kind == "std": return float(np.std(values))
            return float(np.percentile(values, float(percentile)))

        # Face centres are supplemental validity samples.  They must not bias
        # global cell volume, shape, stretch, or condition distributions.
        raw_median = stat(raw_J[quadrature], "median")
        volume_median = stat(determinant_H[quadrature], "median")
        volume_p01 = stat(determinant_H[quadrature], "percentile", 1.0)
        def ratio(numerator: float, denominator: float) -> float:
            return float(numerator / denominator) if np.isfinite(numerator) and np.isfinite(denominator) and denominator != 0.0 else float("nan")

        def region_name(cell: tuple[int, int, int]) -> str:
            i, j, _ = cell
            if i in (0, cell_shape[0] - 1) and j in (0, cell_shape[1] - 1): return "corner_adjacent"
            if i == 0: return "u_min_adjacent"
            if i == cell_shape[0] - 1: return "u_max_adjacent"
            if j == 0: return "v_min_adjacent"
            if j == cell_shape[1] - 1: return "v_max_adjacent"
            return "core"

        def location(index: int, value: float) -> MetricQualityLocation | None:
            if index < 0 or index >= quadrature_count: return None
            cell = tuple(int(x) for x in sample_cell_indices[index])
            position = self._position_only(all_points[index])
            return MetricQualityLocation(float(value), region_name(cell), tuple(float(x) for x in all_points[index]), tuple(float(x) for x in position), cell)

        def extreme(values: np.ndarray, want_min: bool) -> MetricQualityLocation | None:
            candidates = np.flatnonzero(np.isfinite(values[:quadrature_count]))
            if not candidates.size: return None
            local = values[candidates]
            index = int(candidates[np.argmin(local) if want_min else np.argmax(local)])
            return location(index, values[index])

        regions: list[MetricQualityRegion] = []
        all_cells = sample_cell_indices
        region_masks = {
            "all": np.ones(quadrature_count, dtype=bool),
            "core": (all_cells[:, 0] > 0) & (all_cells[:, 0] < cell_shape[0] - 1) & (all_cells[:, 1] > 0) & (all_cells[:, 1] < cell_shape[1] - 1),
            "wall_adjacent": (all_cells[:, 0] == 0) | (all_cells[:, 0] == cell_shape[0] - 1) | (all_cells[:, 1] == 0) | (all_cells[:, 1] == cell_shape[1] - 1),
            "u_min_adjacent": all_cells[:, 0] == 0,
            "u_max_adjacent": all_cells[:, 0] == cell_shape[0] - 1,
            "v_min_adjacent": all_cells[:, 1] == 0,
            "v_max_adjacent": all_cells[:, 1] == cell_shape[1] - 1,
            "corner_adjacent": ((all_cells[:, 0] == 0) | (all_cells[:, 0] == cell_shape[0] - 1)) & ((all_cells[:, 1] == 0) | (all_cells[:, 1] == cell_shape[1] - 1)),
        }
        for label, mask in region_masks.items():
            values = determinant_H[:quadrature_count][mask]
            median = stat(values, "median")
            regions.append(MetricQualityRegion(
                label, int(np.count_nonzero(mask)),
                float(np.mean(quadrature_valid[mask])) if np.any(mask) else float("nan"),
                stat(scaled_J[:quadrature_count][mask], "min"), stat(condition[:quadrature_count][mask], "max"),
                stat(scaled_J[:quadrature_count][mask], "percentile", 1.0), stat(condition[:quadrature_count][mask], "percentile", 95.0),
                stat(stretch[:quadrature_count][mask], "percentile", 95.0), stat(stretch[:quadrature_count][mask], "max"),
                ratio(stat(values, "percentile", 1.0), median), ratio(stat(values, "percentile", 99.0), stat(values, "percentile", 1.0)),
                int(np.count_nonzero(~quadrature_valid[mask])),
            ))

        # Cell representatives provide compact, direction-labelled smoothness
        # diagnostics without mixing arbitrary Gauss-point neighbours.
        centers = np.stack(np.meshgrid(uc, vc, ec, indexing="ij"), axis=-1)
        center_positions, center_A = self._position_and_jacobian(centers.reshape(-1, 3))
        center_A = center_A.reshape(cell_shape + (3, 3))
        center_J = np.linalg.det(center_A)
        center_volume = center_J * du[:, None, None] * dv[None, :, None] * deta[None, None, :]
        center_gcontra = np.linalg.pinv(np.einsum("...ki,...kj->...ij", center_A, center_A))
        K = center_J[..., None, None] * center_gcontra
        directional_log_volume_jumps: dict[str, tuple[float, float]] = {}
        directional_K_jumps: dict[str, tuple[float, float]] = {}
        jump_records: list[tuple[float, str, tuple[int, int, int], tuple[int, int, int], np.ndarray, np.ndarray, str]] = []

        def directional(axis: int, label: str, seam: bool = False) -> None:
            if seam:
                a_vol, b_vol = center_volume.take(-1, axis=axis), center_volume.take(0, axis=axis)
                a_K, b_K = K.take(-1, axis=axis), K.take(0, axis=axis)
            else:
                a_vol = np.take(center_volume, np.arange(center_volume.shape[axis] - 1), axis=axis)
                b_vol = np.take(center_volume, np.arange(1, center_volume.shape[axis]), axis=axis)
                a_K = np.take(K, np.arange(K.shape[axis] - 1), axis=axis)
                b_K = np.take(K, np.arange(1, K.shape[axis]), axis=axis)
            with np.errstate(divide="ignore", invalid="ignore"):
                dvol = np.abs(np.log(b_vol) - np.log(a_vol))
            knorm_a, knorm_b = np.linalg.norm(a_K, axis=(-2, -1)), np.linalg.norm(b_K, axis=(-2, -1))
            dK = np.linalg.norm(b_K - a_K, axis=(-2, -1)) / (0.5 * (knorm_a + knorm_b) + np.finfo(float).eps)
            directional_log_volume_jumps[label] = (stat(dvol, "percentile", 95.0), stat(dvol, "max"))
            directional_K_jumps[label] = (stat(dK, "percentile", 95.0), stat(dK, "max"))
            if np.any(np.isfinite(dvol)):
                local_index = tuple(int(value) for value in np.unravel_index(np.nanargmax(dvol), dvol.shape))
                if seam:
                    ia = (local_index[0], local_index[1], cell_shape[2] - 1)
                    ib = (local_index[0], local_index[1], 0)
                else:
                    ia = local_index
                    ib_list = list(local_index)
                    ib_list[axis] += 1
                    ib = tuple(ib_list)
                qa, qb = centers[ia], centers[ib]
                if seam: qb = qb.copy(); qb[2] += self._period
                jump_records.append((float(dvol[local_index]), label, ia, ib, qa, qb, "volume"))
            if np.any(np.isfinite(dK)):
                local_index = tuple(int(value) for value in np.unravel_index(np.nanargmax(dK), dK.shape))
                if seam:
                    ia = (local_index[0], local_index[1], cell_shape[2] - 1)
                    ib = (local_index[0], local_index[1], 0)
                else:
                    ia = local_index
                    ib_list = list(local_index)
                    ib_list[axis] += 1
                    ib = tuple(ib_list)
                qa, qb = centers[ia], centers[ib]
                if seam: qb = qb.copy(); qb[2] += self._period
                jump_records.append((float(dK[local_index]), label, ia, ib, qa, qb, "K"))

        directional(0, "u"); directional(1, "v")
        if cell_shape[2] > 1:
            directional(2, "eta")
            directional(2, "eta_seam", seam=True)

        def jump_location(kind: str) -> MetricQualityJumpLocation | None:
            candidates = [record for record in jump_records if record[-1] == kind]
            if not candidates: return None
            value, direction, ia, ib, qa, qb, _ = max(candidates, key=lambda record: record[0])
            xa = self._position_only(qa); xb = self._position_only(qb)
            return MetricQualityJumpLocation(
                value,
                direction,
                tuple(float(x) for x in qa),
                tuple(float(x) for x in qb),
                tuple(float(x) for x in xa),
                tuple(float(x) for x in xb),
                ia,
                ib,
                region_name(ia),
                region_name(ib),
            )

        inverse_residual_max = stat(np.asarray(inverse_residual), "max")

        eta_stats: dict[str, float] = {}
        eta_location: MetricQualityLocation | None = None
        if eta_absolute is not None:
            absolute = eta_absolute
            eta_stats = {"median": stat(absolute, "median"), "p95": stat(absolute, "percentile", 95.0), "p99": stat(absolute, "percentile", 99.0), "max": stat(absolute, "max")}
            eta_location = extreme(absolute, False)

        u_seam = (uc[:, None] + 0.5 * du[:, None] * nodes[None, :]).reshape(-1)
        v_seam = (vc[:, None] + 0.5 * dv[:, None] * nodes[None, :]).reshape(-1)
        uv_seam = np.stack(np.meshgrid(u_seam, v_seam, indexing="ij"), axis=-1).reshape(-1, 2)
        q0 = np.column_stack((uv_seam, np.full(uv_seam.shape[0], eta_edges[0])))
        q1 = q0.copy(); q1[:, 2] += self._period
        x0, A0 = self._position_and_jacobian(q0); x1, A1 = self._position_and_jacobian(q1)
        rotation = np.array(((np.cos(self._period), -np.sin(self._period), 0.0), (np.sin(self._period), np.cos(self._period), 0.0), (0.0, 0.0, 1.0)))
        x0r, A0r = x0 @ rotation.T, np.einsum("ij,...jk->...ik", rotation, A0)
        G0, G1 = np.einsum("...ki,...kj->...ij", A0, A0), np.einsum("...ki,...kj->...ij", A1, A1)
        seam_residuals = {
            "position": float(np.max(np.linalg.norm(x1 - x0r, axis=1) / np.maximum(np.linalg.norm(x0r, axis=1), np.finfo(float).eps))),
            "jacobian_matrix": float(np.max(np.linalg.norm(A1 - A0r, axis=(1, 2)) / np.maximum(np.linalg.norm(A0r, axis=(1, 2)), np.finfo(float).eps))),
            "metric_tensor": float(np.max(np.linalg.norm(G1 - G0, axis=(1, 2)) / np.maximum(np.linalg.norm(G0, axis=(1, 2)), np.finfo(float).eps))),
            "J": float(np.max(np.abs(np.linalg.det(A1) - np.linalg.det(A0)) / np.maximum(np.abs(np.linalg.det(A0)), np.finfo(float).eps))),
        }
        result = self._mmpde_result
        mmpde_metadata: dict[str, Any] = {}
        if result is not None:
            energy = np.asarray(result.energy_history, dtype=float)
            minimum_jacobian = np.asarray(
                result.minimum_jacobian_history, dtype=float
            )
            initial_energy = float(energy[0]) if energy.size else float("nan")
            final_energy = float(energy[-1]) if energy.size else float("nan")
            mmpde_metadata = {
                "converged": bool(result.converged),
                "iterations": int(result.iterations),
                "initial_energy": initial_energy,
                "final_energy": final_energy,
                "final_over_initial_energy": ratio(final_energy, initial_energy),
                "final_max_nodal_update": float(result.max_free_node_update),
                "initial_minimum_discrete_J": (
                    float(minimum_jacobian[0])
                    if minimum_jacobian.size else float("nan")
                ),
                "final_minimum_discrete_J": (
                    float(minimum_jacobian[-1])
                    if minimum_jacobian.size else float("nan")
                ),
            }
            for name, history in result.component_energy_history.items():
                history = np.asarray(history, dtype=float)
                mmpde_metadata[f"{name}_final_over_initial"] = (
                    ratio(float(history[-1]), float(history[0]))
                    if history.size else float("nan")
                )

        worst_volume_jump = jump_location("volume")
        worst_K_jump = jump_location("K")
        legacy_jump_axis = (
            "eta (periodic seam)"
            if worst_volume_jump is not None and worst_volume_jump.direction == "eta_seam"
            else worst_volume_jump.direction if worst_volume_jump is not None else "none"
        )
        legacy_jump_a = (
            worst_volume_jump.logical_a if worst_volume_jump is not None else (float("nan"),) * 3
        )
        legacy_jump_b = (
            worst_volume_jump.logical_b if worst_volume_jump is not None else (float("nan"),) * 3
        )
        worst_angle_name, worst_angle_values = max(
            angle_cosines.items(),
            key=lambda item: stat(item[1][quadrature], "max"),
        )
        worst_angle = extreme(worst_angle_values, False)
        if worst_angle is not None:
            worst_angle = MetricQualityLocation(
                worst_angle.value,
                f"{worst_angle.region}:{worst_angle_name}",
                worst_angle.logical,
                worst_angle.cartesian,
                worst_angle.cell_index,
            )
        q_raw_J = raw_J[quadrature]
        q_volume = determinant_H[quadrature]
        q_scaled_J = scaled_J[quadrature]
        q_condition = condition[quadrature]
        q_stretch = stretch[quadrature]
        q_inverse_residual = inverse_residual[quadrature]
        cell_count = int(np.prod(cell_shape))
        return MetricQualityReport(
            sample_count=int(all_points.shape[0]),
            valid_fraction=float(np.mean(quadrature_valid)),
            raw_J_min=stat(q_raw_J, "min"),
            raw_J_p01=stat(q_raw_J, "percentile", 1.0),
            raw_J_median=raw_median,
            raw_J_max=stat(q_raw_J, "max"),
            raw_J_min_over_median=ratio(stat(q_raw_J, "min"), raw_median),
            scaled_J_min=stat(q_scaled_J, "min"),
            scaled_J_p01=stat(q_scaled_J, "percentile", 1.0),
            mapping_condition_median=stat(q_condition, "median"),
            mapping_condition_p95=stat(q_condition, "percentile", 95.0),
            mapping_condition_max=stat(q_condition, "max"),
            max_neighbor_log_J_jump=max(
                (values[1] for values in directional_log_volume_jumps.values()),
                default=float("nan"),
            ),
            inverse_residual_max=stat(q_inverse_residual, "max"),
            inverse_residual_p99=stat(
                q_inverse_residual, "percentile", 99.0
            ),
            regions=tuple(regions),
            worst_scaled_jacobian=extreme(q_scaled_J, True),
            worst_mapping_condition=extreme(q_condition, False),
            max_neighbor_log_J_jump_axis=legacy_jump_axis,
            max_neighbor_log_J_jump_endpoint_a=legacy_jump_a,
            max_neighbor_log_J_jump_endpoint_b=legacy_jump_b,
            points_per_cell=q_per_cell,
            cell_count=cell_count,
            gauss_order=int(gauss_order),
            quadrature_sample_count=quadrature_count,
            face_sample_count=int(face_points.shape[0]),
            nonpositive_J_count=int(np.count_nonzero(~quadrature_valid)),
            nonpositive_J_fraction=float(np.mean(~quadrature_valid)),
            face_nonpositive_J_count=int(np.count_nonzero(~face_valid)),
            face_valid_fraction=(
                float(np.mean(face_valid)) if face_valid.size else float("nan")
            ),
            raw_volume_min=stat(q_volume, "min"),
            volume_min_over_median=ratio(stat(q_volume, "min"), volume_median),
            volume_p01_over_median=ratio(volume_p01, volume_median),
            volume_p05_over_median=ratio(
                stat(q_volume, "percentile", 5.0), volume_median
            ),
            volume_p95_over_median=ratio(
                stat(q_volume, "percentile", 95.0), volume_median
            ),
            volume_p99_over_median=ratio(
                stat(q_volume, "percentile", 99.0), volume_median
            ),
            volume_p99_over_p01=ratio(
                stat(q_volume, "percentile", 99.0), volume_p01
            ),
            volume_coefficient_of_variation=ratio(
                stat(q_volume, "std"), stat(q_volume, "mean")
            ),
            scaled_J_p05=stat(q_scaled_J, "percentile", 5.0),
            scaled_J_median=stat(q_scaled_J, "median"),
            mapping_condition_p99=stat(q_condition, "percentile", 99.0),
            stretch_median=stat(q_stretch, "median"),
            stretch_p95=stat(q_stretch, "percentile", 95.0),
            stretch_p99=stat(q_stretch, "percentile", 99.0),
            stretch_max=stat(q_stretch, "max"),
            angle_cosines={
                name: (
                    stat(values[quadrature], "percentile", 95.0),
                    stat(values[quadrature], "max"),
                )
                for name, values in angle_cosines.items()
            },
            directional_log_volume_jumps=directional_log_volume_jumps,
            directional_K_jumps=directional_K_jumps,
            eta_constraint_residuals=eta_stats,
            periodic_seam_residuals=seam_residuals,
            representation_metadata={
                "mesh_shape": (
                    int(self._u.size), int(self._v.size), int(self._eta.size)
                ),
                "cell_counts": cell_shape,
                "evaluator_cell_counts": (
                    int(self._u.size - 1),
                    int(self._v.size - 1),
                    int(self._eta.size),
                ),
                "quality_cell_counts": cell_shape,
                "cell_count": cell_count,
                "spline_degree_u": int(
                    self._channels[0]._spline_degree_u or 1
                ),
                "spline_degree_v": int(
                    self._channels[0]._spline_degree_v or 1
                ),
                "eta_representation": (
                    f"Fourier({self._channels[0]._modes.size} coefficients)"
                ),
                "mmpde_fit_scale": float(self._mmpde_fit_scale),
                "gauss_order": int(gauss_order),
                "total_quality_samples": int(all_points.shape[0]),
            },
            mmpde_metadata=mmpde_metadata,
            worst_jacobian=extreme(q_raw_J, True),
            worst_volume=extreme(q_volume, True),
            worst_stretch=extreme(q_stretch, False),
            worst_angle_cosine=worst_angle,
            worst_eta_constraint=eta_location,
            worst_volume_jump=worst_volume_jump,
            worst_K_jump=worst_K_jump,
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
    "MetricQualityJumpLocation",
    "MetricQualityRegion",
    "MetricQualityReport",
    "MetricEvaluator",
    "build_metric_evaluator",
    "build_wall_fitted_initial_mesh",
]
