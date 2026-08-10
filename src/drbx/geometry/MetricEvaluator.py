"""Smooth representations of structured ``D^2 x S^1`` meshes.

The square topology uses tensor-product splines on ``[0, 1]^2`` and Fourier
modes in eta.  The toroidal topology uses Fourier--Zernike coordinates
``(u, theta, eta)`` with an analytically regularized collapsed axis at
``u=0``.  :func:`build_metric_evaluator` is the single construction entry
point for both representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import BSpline, PchipInterpolator, RectBivariateSpline
from scipy.linalg import null_space
from scipy.special import eval_jacobi
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
    def condition_number(self) -> np.ndarray:
        return self.condition

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
        self._basis = {}
        self._fit_coefficients = {}
        for im in self._active_theta_indices:
            m0 = self._modes_theta[im]
            m = int(round(m0))
            orders = list(range(abs(m), self._radial_degree + 1, 2))
            coefficient_matrix = np.zeros(
                (self._active_eta_indices.size, len(orders)), dtype=complex
            )
            for local_ine, ine in enumerate(self._active_eta_indices):
                B = self._basis_matrix(m, self._u)
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
        key = (int(m), query.shape, query.tobytes())
        if key in self._basis:
            return self._basis[key]
        orders = list(range(abs(m), self._radial_degree + 1, 2))
        uu = query
        out = np.empty((uu.size, len(orders)), dtype=float)
        for j, ell in enumerate(orders):
            out[:, j] = uu ** abs(m) * eval_jacobi((ell - abs(m)) // 2, 0, abs(m), 2 * uu * uu - 1)
        self._basis[key] = out
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
        B = self._basis_matrix(m, u)
        if theta_over_u:
            if m == 0:
                return np.zeros((u.size, self._active_eta_indices.size), complex)
            B = B.copy()
            mask = u != 0
            if np.any(mask):
                B[mask] = self._basis_matrix(m, u[mask]) / u[mask, None]
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

    ``topology="square"`` retains the historical tensor-product chart on the
    disk.  ``topology="toroidal"`` uses ``(u, theta, eta)`` with a collapsed,
    analytically regularized axis at ``u=0``.

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
    topology, radial_degree, poloidal_modes, toroidal_modes:
        Select the legacy square representation or the axis-regular
        Fourier--Zernike representation and its retained mode ranges.
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
        topology: str = "square",
        radial_degree: int | None = None,
        poloidal_modes: Any = None,
        toroidal_modes: Any = None,
    ) -> None:
        topology = str(topology).lower()
        if topology not in ("square", "toroidal"):
            raise ValueError("topology must be 'square' or 'toroidal'")
        self._topology = topology
        self._u = _axis(u, "u")
        self._v = _axis(v, "v")
        self._eta = _axis(eta, "eta")
        metric_spline_degree = _validate_metric_spline_degree(metric_spline_degree)
        if self._u.size < 2 or self._v.size < 2 or self._eta.size < 4:
            raise ValueError("u and v need at least 2 samples; eta needs at least 4")
        if not np.isclose(self._u[0], 0.0) or not np.isclose(self._u[-1], 1.0):
            raise ValueError("u must span [0, 1]")
        if topology == "square":
            if not np.isclose(self._v[0], 0.0) or not np.isclose(self._v[-1], 1.0):
                raise ValueError("v must span [0, 1]")
        else:
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
        fit_degree = min(metric_spline_degree, self._u.size - 1, self._v.size - 1)
        if topology == "square":
            self._channels = tuple(
                _FourierSplineChannel(self._u, self._v, samples, self._eta[0], self._period, fit_degree)
                for samples in (radius, values[..., 2], delta_phi)
            )
        else:
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
        if self._topology == "toroidal":
            return {
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
        topology_code = (
            int(array("topology_code").item())
            if key("topology_code") in payload
            else 0
        )
        if topology_code not in (0, 1):
            raise ValueError("cached MetricEvaluator topology code is invalid")
        if not np.isclose(u[0], 0.0) or not np.isclose(u[-1], 1.0):
            raise ValueError("cached u must span [0, 1]")
        if topology_code == 0:
            if not np.isclose(v[0], 0.0) or not np.isclose(v[-1], 1.0):
                raise ValueError("cached v must span [0, 1]")
        else:
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
        if topology_code == 1:
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
        evaluator._topology = "square"
        evaluator._eta = eta
        evaluator._period = period
        evaluator._nfp = nfp
        evaluator._metric_spline_degree = metric_spline_degree
        evaluator._radial_degree = None
        evaluator._poloidal_modes = None
        evaluator._toroidal_modes = None
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
        if np.any((uf < self._u[0]) | (uf > self._u[-1])) or (
            self._topology == "square" and np.any((vf < self._v[0]) | (vf > self._v[-1]))
        ):
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

    def cell_center_logical_points(self) -> np.ndarray:
        """Return logical cell centers, including the periodic eta seam cells."""
        u = 0.5 * (self._u[:-1] + self._u[1:])
        v = (
            self._v + np.pi / self._v.size
            if self._topology == "toroidal"
            else 0.5 * (self._v[:-1] + self._v[1:])
        )
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
        eta = self._eta + 0.5 * self._period / self._eta.size
        if self._topology == "toroidal":
            theta = self._v + np.pi / self._v.size
            return np.stack(
                np.meshgrid([1.0], theta, eta, indexing="ij"), axis=-1
            ).reshape(-1, 3)
        v = 0.5 * (self._v[:-1] + self._v[1:])
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
        if self._topology == "toroidal":
            raise NotImplementedError(
                "use evaluate_toroidal_quality for an axis-regular toroidal map"
            )
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


def _build_square_topology(
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


def _cyclic_cost(a: np.ndarray, b: np.ndarray, shift: int) -> float:
    return float(np.mean((a - np.roll(b, shift, axis=0)) ** 2))


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
                _cyclic_cost(comparison[k], comparison[nxt], shift)
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


def reparameterize_centerline(
    wall_evaluator: Any,
    eta_evaluator: Any,
    target_eta: Any,
    *,
    period: float,
    phi_samples: int = 2049,
    eta_is_normalized: bool = False,
) -> np.ndarray:
    """Return fixed physical centerline points at endpoint-exclusive eta values."""
    target = np.asarray(target_eta, dtype=float)
    if target.ndim != 1 or target.size < 2 or not np.all(np.isfinite(target)):
        raise ValueError("target_eta must be a finite one-dimensional array")
    if phi_samples < 17:
        raise ValueError("phi_samples is too small")
    phi0 = float(getattr(wall_evaluator, "phi0", 0.0))
    phi_period = float(getattr(wall_evaluator, "period", period))
    phi = phi0 + np.linspace(0.0, phi_period, int(phi_samples), endpoint=True)
    rz = wall_evaluator.centerline(phi)
    rz = np.stack(rz[:2], axis=-1)
    axis = np.stack((rz[:, 0] * np.cos(phi), rz[:, 0] * np.sin(phi), rz[:, 1]), axis=-1)
    eta_values = _call_eta(eta_evaluator, axis)
    if eta_is_normalized:
        eta_values = eta_values * period
    eta_unwrapped = np.unwrap(_TWO_PI * eta_values / period) * period / _TWO_PI
    delta = eta_unwrapped[-1] - eta_unwrapped[0]
    if abs(abs(delta) - period) > max(2e-3 * period, 2e-8):
        raise ValueError(f"centerline eta covers {delta:g}, expected one period {period:g}")
    if delta < 0:
        eta_unwrapped = eta_unwrapped[::-1]
        axis = axis[::-1]
        phi = phi[::-1]
    if np.any(np.diff(eta_unwrapped) <= 0):
        raise ValueError("eta is not strictly monotone along the centerline")
    target_unwrapped = target + eta_unwrapped[0] - target[0]
    target_phi = PchipInterpolator(eta_unwrapped, phi)(target_unwrapped)

    def centerline_cartesian(query_phi: np.ndarray) -> np.ndarray:
        query_rz = wall_evaluator.centerline(query_phi)
        return np.stack(
            (
                query_rz[0] * np.cos(query_phi),
                query_rz[0] * np.sin(query_phi),
                query_rz[1],
            ),
            axis=-1,
        )

    # Refine the inverse parameterization while remaining exactly on the
    # existing physical centerline.
    for _ in range(5):
        candidate = centerline_cartesian(target_phi)
        value = _call_eta(eta_evaluator, candidate)
        if eta_is_normalized:
            value *= period
        residual = (value - target + 0.5 * period) % period - 0.5 * period
        if np.max(np.abs(residual)) <= 2.0e-11 * max(1.0, period):
            break
        step = 1.0e-6
        plus = _call_eta(
            eta_evaluator, centerline_cartesian(target_phi + step)
        )
        minus = _call_eta(
            eta_evaluator, centerline_cartesian(target_phi - step)
        )
        if eta_is_normalized:
            plus *= period
            minus *= period
        derivative = (
            (plus - minus + 0.5 * period) % period - 0.5 * period
        ) / (2.0 * step)
        if np.any(np.abs(derivative) < 1.0e-10):
            raise ValueError("eta is locally stationary along the centerline")
        target_phi -= residual / derivative
    return centerline_cartesian(target_phi)


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
    axis_points: Any | None = None,
    radial_degree: int = 3,
    poloidal_modes: int | None = None,
    toroidal_modes: int | None = None,
    projection_iterations: int = 0,
    eta_is_normalized: bool = False,
    resample_wall: bool = True,
    wall_sample_count: int | None = None,
    validate: bool = True,
) -> MetricEvaluator:
    """Build a geometry-ready toroidal ``MetricEvaluator`` without MMPDE.

    Wall curves are sampled at the supplied eta values unless explicitly
    supplied.  When no axis is supplied, the default axis is the poloidal
    centroid of those already aligned wall curves in cylindrical
    ``(R, Z, delta_phi)`` channels, followed by a wrapped Newton projection
    onto the requested eta surfaces.  ``reparameterize_centerline`` remains
    available as a standalone helper, but is not used for this default
    because the wall curves are the authoritative, phase-aligned topology.
    The returned object uses the toroidal topology of ``MetricEvaluator``
    directly.
    """
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
    if resample_wall:
        sampled_wall = np.stack([resample_periodic_curve(curve, theta.size) for curve in sampled_wall])
    elif sampled_wall.shape[1] != theta.size:
        raise ValueError("endpoint-exclusive wall_curves must contain ntheta points")
    sampled_wall = align_wall_curves(sampled_wall, field_period=period)
    explicit_axis = axis_points is not None
    # Convert only after alignment: the cylindrical phase channel is then
    # continuous over eta and its poloidal mean is a well-defined centerline
    # guess even when WallEvaluator.centerline() is offset or unavailable.
    wall_channels = cartesian_to_channels(
        sampled_wall.transpose(1, 0, 2), eta[None, :], eta_axis=1
    )
    if axis_points is None:
        axis_channels = np.mean(wall_channels, axis=0)
        if not np.all(np.isfinite(axis_channels)) or np.any(axis_channels[:, 0] <= 0.0):
            raise ValueError("wall-centroid axis guess has invalid cylindrical channels")
        axis_points = channels_to_cartesian(axis_channels, eta)
        if eta_evaluator is not None:
            # Newton projection is along the eta gradient and uses a wrapped
            # residual so the field-period seam remains endpoint-exclusive.
            for _ in range(12):
                values, gradients = _eta_value_gradient(eta_evaluator, axis_points)
                if eta_is_normalized:
                    values = values * period
                    gradients = gradients * period
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
    else:
        axis_points = _as_points(axis_points, "axis_points")
    axis_points = _as_points(axis_points, "axis_points")
    if axis_points.shape != (eta.size, 3):
        raise ValueError("axis_points must have shape (neta, 3)")
    contains = getattr(wall_evaluator, "contains_cartesian", None)
    if contains is not None and not np.all(np.asarray(contains(axis_points), dtype=bool)):
        raise ValueError("the coordinate axis must remain inside the wall")
    if eta_evaluator is not None:
        axis_eta = _call_eta(eta_evaluator, axis_points)
        if eta_is_normalized:
            axis_eta *= period
        axis_residual = (axis_eta - eta + 0.5 * period) % period - 0.5 * period
        if np.max(np.abs(axis_residual)) > 2.0e-6 * max(1.0, period):
            description = "explicit" if explicit_axis else "wall-centroid projected"
            raise ValueError(f"{description} axis does not satisfy the requested eta levels")
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
            eta_is_normalized=eta_is_normalized, iterations=projection_iterations,
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
        radial_validation = np.unique(
            np.concatenate(
                (
                    [0.0, min(0.25 * u[1], 1.0)],
                    0.5 * (u[:-1] + u[1:]),
                    [1.0],
                )
            )
        )
        theta_validation = theta + 0.37 * (_TWO_PI / theta.size)
        eta_validation = eta + 0.31 * (period / eta.size)
        validation_points = np.stack(
            np.meshgrid(
                radial_validation,
                theta_validation,
                eta_validation,
                indexing="ij",
            ),
            axis=-1,
        )
        regularized = evaluator.evaluate_regularized(validation_points)
        if np.any(~regularized.valid):
            minimum = float(np.nanmin(regularized.J_reg))
            raise ValueError(
                "toroidal Fourier-Zernike fit folds on the validation grid; "
                f"minimum regularized Jacobian is {minimum:.6e}"
            )
    return evaluator


def build_metric_evaluator(
    eta_evaluator: Any,
    initial_positions: Any | None = None,
    *,
    topology: str = "square",
    wall_evaluator: Any | None = None,
    mesh_shape: tuple[int, int, int] | None = None,
    logical_axes: tuple[Any, Any, Any] | None = None,
    monitor: Any | None = None,
    fixed_mask: Any | None = None,
    options: MMPDEOptions | None = None,
    projector: Any | None = None,
    metric_spline_degree: int | None = None,
    wall_curves: Any | None = None,
    axis_points: Any | None = None,
    radial_degree: int = 3,
    poloidal_modes: int | None = None,
    toroidal_modes: int | None = None,
    projection_iterations: int = 0,
    eta_is_normalized: bool = False,
    resample_wall: bool = True,
    wall_sample_count: int | None = None,
    validate: bool = True,
) -> MetricEvaluator:
    """Build either square- or toroidal-topology metric geometry.

    ``topology="square"`` preserves the historical wall-fitted MMPDE path.
    ``topology="toroidal"`` constructs an axis-regular Fourier--Zernike map;
    its second logical axis is endpoint-exclusive ``theta`` over ``2*pi``.
    Both paths return the same :class:`MetricEvaluator` type.
    """
    selected_topology = str(topology).lower()
    if selected_topology == "square":
        if wall_curves is not None or axis_points is not None:
            raise ValueError(
                "wall_curves and axis_points are only valid for toroidal topology"
            )
        return _build_square_topology(
            eta_evaluator,
            initial_positions,
            wall_evaluator=wall_evaluator,
            mesh_shape=mesh_shape,
            logical_axes=logical_axes,
            monitor=monitor,
            fixed_mask=fixed_mask,
            options=options,
            projector=projector,
            metric_spline_degree=metric_spline_degree,
        )
    if selected_topology != "toroidal":
        raise ValueError("topology must be 'square' or 'toroidal'")
    if any(value is not None for value in (monitor, fixed_mask, options, projector)):
        raise ValueError(
            "monitor, fixed_mask, options, and projector belong to the square MMPDE path"
        )
    if metric_spline_degree is not None:
        raise ValueError(
            "metric_spline_degree is only used by square topology; "
            "use radial_degree for toroidal topology"
        )

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

    if initial_positions is not None:
        if any(
            value is not None
            for value in (wall_evaluator, mesh_shape, wall_curves, axis_points)
        ):
            raise ValueError(
                "explicit toroidal positions cannot be combined with wall construction inputs"
            )
        positions = np.asarray(initial_positions, dtype=np.float64)
        if positions.ndim != 4 or positions.shape[-1] != 3:
            raise ValueError(
                "initial_positions must have shape (nu, ntheta, neta, 3)"
            )
        shape = positions.shape[:3]
    else:
        if mesh_shape is None:
            raise ValueError(
                "toroidal wall construction requires mesh_shape or explicit positions"
            )
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

    if initial_positions is not None:
        return MetricEvaluator(
            *axes,
            positions,
            period=period,
            nfp=eta_nfp,
            topology="toroidal",
            radial_degree=radial_degree,
            poloidal_modes=poloidal_modes,
            toroidal_modes=toroidal_modes,
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
        axis_points=axis_points,
        radial_degree=radial_degree,
        poloidal_modes=poloidal_modes,
        toroidal_modes=toroidal_modes,
        projection_iterations=projection_iterations,
        eta_is_normalized=eta_is_normalized,
        resample_wall=resample_wall,
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
    "MetricQualityLocation",
    "MetricQualityJumpLocation",
    "MetricQualityRegion",
    "MetricQualityReport",
    "MetricEvaluator",
    "ToroidalQualityReport",
    "align_wall_curves",
    "axis_regular_initializer",
    "build_metric_evaluator",
    "build_wall_fitted_initial_mesh",
    "evaluate_toroidal_quality",
    "project_interior_eta",
    "resample_periodic_curve",
]
