"""Independent continuum manufactured solution for the six-field EB system.

This module is deliberately separate from the production FCI/RLP operators. It
only samples a smooth metric representation and evaluates the smooth bulk
continuum-limit equations. In particular, no production RHS, finite-volume
operator, or RLP projection is used here. The caller can project the pointwise
source with the same owner-space quadrature used by the simulation.

The metric derivatives used by the curvature and polarization operators are
fourth-order centred differences of the smooth :class:`MetricEvaluator`.  The
derivatives of the manufactured fields themselves are analytic.  An optional
generalized-potential mode derives vorticity from the independent continuum
polarization operator; its structured-grid derivatives are supplied by the
MMS quadrature projector.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


def _lagrange_weights(nodes: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    """Evaluate cubic Lagrange weights for each coordinate.

    Geometry artifacts contain samples at cell centres rather than a fitted
    evaluator.  This small tensor-product interpolant gives the independent
    MMS reference a smooth, deterministic view of those serialized samples;
    it never consults MAKEGRID or a metric cache.
    """

    nodes = np.asarray(nodes, dtype=np.float64)
    coordinate = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    if nodes.ndim == 1:
        nodes = np.broadcast_to(nodes[None, :], (coordinate.size, nodes.size))
    if nodes.ndim != 2 or nodes.shape[0] != coordinate.size:
        raise ValueError("interpolation stencil must have shape (points, nodes)")
    weights = np.ones(nodes.shape, dtype=np.float64)
    for j in range(nodes.shape[1]):
        node = nodes[:, j]
        for k in range(nodes.shape[1]):
            other = nodes[:, k]
            if j != k:
                weights[:, j] *= (coordinate - other) / (node - other)
    return weights


class SerializedBFieldEvaluator:
    """Periodic tensor-product interpolant over artifact cell B samples."""

    def __init__(self, axes, B_contravariant, magnitude, *, periods=None):
        self.axes = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
        self.B_contravariant = np.asarray(B_contravariant, dtype=np.float64)
        self.magnitude = np.asarray(magnitude, dtype=np.float64)
        self.periods = tuple(periods or (None, None, None))
        if self.B_contravariant.shape != self.magnitude.shape + (3,):
            raise ValueError("B_contravariant and magnitude shapes are inconsistent")

    def _interpolate(self, points, values):
        q = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        # Tiny artifact fixtures used by contract tests may have fewer than
        # four samples along an axis.  They are not production MMS grids;
        # nearest-node lookup keeps those round-trip tests well-defined while
        # all real grids use the cubic branch below.
        if any(axis.size < 4 for axis in self.axes):
            indices = []
            for axis, (nodes, period) in enumerate(zip(self.axes, self.periods)):
                coordinate = q[:, axis]
                if period is not None:
                    coordinate = nodes[0] + np.mod(coordinate - nodes[0], period)
                indices.append(np.argmin(np.abs(coordinate[:, None] - nodes[None, :]), axis=1))
            return values[indices[0], indices[1], indices[2]]
        index_sets = []
        weight_sets = []
        for axis, (nodes, period) in enumerate(zip(self.axes, self.periods)):
            coordinate = q[:, axis].copy()
            if period is not None:
                spacing = float(period) / float(nodes.size)
                coordinate = nodes[0] + np.mod(coordinate - nodes[0], float(period))
                position = (coordinate - nodes[0]) / spacing
                base = np.floor(position).astype(np.int64)
                indices = (base[:, None] + np.arange(-1, 3)[None, :]) % nodes.size
                stencil_nodes = nodes[indices]
                stencil_nodes = stencil_nodes + float(period) * np.round(
                    (coordinate[:, None] - stencil_nodes) / float(period)
                )
            else:
                base = np.searchsorted(nodes, coordinate, side="right") - 1
                base = np.clip(base, 1, nodes.size - 3)
                indices = base[:, None] + np.arange(-1, 3)[None, :]
                stencil_nodes = nodes[indices]
            index_sets.append(indices)
            weight_sets.append(_lagrange_weights(stencil_nodes, coordinate))
        i0, i1, i2 = index_sets
        gathered = values[
            i0[:, :, None, None],
            i1[:, None, :, None],
            i2[:, None, None, :],
        ]
        result = np.einsum(
            "na,nb,nc,nabc...->n...",
            weight_sets[0], weight_sets[1], weight_sets[2], gathered,
        )
        return result

    def evaluate(self, points, *, reject_nonpositive_J=False):
        q = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        return SimpleNamespace(
            B_contravariant=self._interpolate(q, self.B_contravariant),
            magnitude=self._interpolate(q, self.magnitude),
        )


class SerializedMetricEvaluator:
    """Metric evaluator reconstructed solely from a geometry artifact."""

    def __init__(self, geometry):
        grid = geometry.grid
        self.axes = tuple(
            np.asarray(axis.centers, dtype=np.float64)
            for axis in (grid.x, grid.y, grid.z)
        )
        self.period = float(np.asarray(grid.z.faces)[-1] - np.asarray(grid.z.faces)[0])
        self.periods = (None, float(np.asarray(grid.y.faces)[-1] - np.asarray(grid.y.faces)[0]), self.period)
        metric = geometry.cell_metric
        self.J = np.asarray(metric.J, dtype=np.float64)
        self.g_cov = np.asarray(metric.g_cov, dtype=np.float64)
        self.g_contra = np.asarray(metric.g_contra, dtype=np.float64)
        self.bfield = SerializedBFieldEvaluator(
            self.axes,
            np.asarray(geometry.cell_bfield.B_contra, dtype=np.float64),
            np.asarray(geometry.cell_bfield.Bmag, dtype=np.float64),
            periods=self.periods,
        )

    def _interpolate(self, points, values):
        return self.bfield._interpolate(points, values)

    def evaluate(self, points, *, reject_nonpositive_J=False):
        q = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        return SimpleNamespace(
            signed_J=self._interpolate(q, self.J),
            covariant_metric=self._interpolate(q, self.g_cov),
            contravariant_metric=self._interpolate(q, self.g_contra),
        )

    def evaluate_magnetic_field(self, points, bfield=None, *, reject_nonpositive_J=False):
        return self.bfield.evaluate(points, reject_nonpositive_J=reject_nonpositive_J)


def build_continuum_reference_from_artifact(
    artifact: Any,
    *,
    B0: float | None = None,
    tau: float = 1.0,
    mi_over_me: float = 1836.0,
    rho_star: float = 1.0,
    Ve_nu: float = 1.0e-3,
    perp_diffusion: float = 1.0e-5,
    enable_generalized_potential: bool = True,
    metric_query_batch_size: int = 4096,
) -> "ContinuumMmsReference":
    """Construct the fixed MMS reference from one serialized artifact.

    The caller supplies the finest artifact once and reuses the returned
    reference for all coarser resolutions.  The loader has already supplied
    all arrays; this function performs no geometry generation or validation.
    """

    geometry = getattr(artifact, "global_geometry", getattr(artifact, "geometry", None))
    if geometry is None:
        raise TypeError("artifact must provide global_geometry or geometry")
    if B0 is None:
        # FciGeometry3D stores B/B0 and |B|/B0, not raw tesla.  The serialized
        # arrays are therefore already in the continuum reference's normalized
        # units and must not be divided by the producer's physical B0 again.
        B0 = 1.0
    evaluator = SerializedMetricEvaluator(geometry)
    return ContinuumMmsReference(
        evaluator,
        evaluator.bfield,
        float(B0),
        tau=tau,
        mi_over_me=mi_over_me,
        rho_star=rho_star,
        Ve_nu=Ve_nu,
        perp_diffusion=perp_diffusion,
        enable_generalized_potential=enable_generalized_potential,
        metric_query_batch_size=metric_query_batch_size,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_continuum_reference_from_sidecar(
    sidecar: str | Path,
    *,
    verify_hashes: bool = False,
    tau: float = 1.0,
    mi_over_me: float = 1836.0,
    rho_star: float = 1.0,
    Ve_nu: float = 1.0e-3,
    perp_diffusion: float = 1.0e-5,
    enable_generalized_potential: bool = True,
    metric_query_batch_size: int | None = None,
) -> "ContinuumMmsReference":
    """Restore a qualified frozen continuous HSX MMS reference.

    The sidecar records the exact producer ``MetricEvaluator`` checkpoint,
    matching MAKEGRID file/currents, physical normalization, and full-domain
    analytic MMS eta period.  The continuous evaluator's own ``period`` is one
    HSX field period and must not silently redefine the manufactured fields.
    """

    from drbx.geometry.Bfield_evaluator import bfield_evaluator_from_makegrid
    from drbx.geometry.MetricEvaluator import MetricEvaluator

    sidecar_path = Path(sidecar).resolve()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "drbx.hsx-continuous-mms-reference":
        raise ValueError("unsupported continuous MMS reference sidecar schema")
    if int(payload.get("version", -1)) != 1:
        raise ValueError("unsupported continuous MMS reference sidecar version")
    metric_cache = Path(payload["metric_cache"]["path"]).resolve()
    makegrid = Path(payload["makegrid"]["path"]).resolve()
    for name, path in (("metric_cache", metric_cache), ("makegrid", makegrid)):
        if not path.is_file():
            raise FileNotFoundError(f"continuous MMS {name} is missing: {path}")
        expected_size = int(payload[name]["size"])
        if path.stat().st_size != expected_size:
            raise ValueError(f"continuous MMS {name} size does not match sidecar")
        if verify_hashes and _sha256(path) != payload[name]["sha256"]:
            raise ValueError(f"continuous MMS {name} hash does not match sidecar")
    with np.load(metric_cache, allow_pickle=False) as cached:
        evaluator = MetricEvaluator.from_cache_payload(
            cached, prefix="metric_evaluator_"
        )
        cached_b0 = float(np.asarray(cached["reference_magnetic_field"]).item())
    B0 = float(payload["reference_magnetic_field"])
    if not np.isclose(cached_b0, B0, rtol=0.0, atol=1.0e-14):
        raise ValueError("sidecar B0 does not match the metric checkpoint")
    currents = np.asarray(payload["makegrid_currents"], dtype=np.float64)
    bfield = bfield_evaluator_from_makegrid(
        makegrid, currents=currents, method="cubic"
    )
    reference = ContinuumMmsReference(
        evaluator,
        bfield,
        B0,
        tau=tau,
        mi_over_me=mi_over_me,
        rho_star=rho_star,
        Ve_nu=Ve_nu,
        perp_diffusion=perp_diffusion,
        enable_generalized_potential=enable_generalized_potential,
        eta_period=float(payload["analytic_mms_eta_period"]),
        metric_query_batch_size=int(
            payload.get("metric_query_batch_size", 4096)
            if metric_query_batch_size is None
            else metric_query_batch_size
        ),
    )
    reference.provenance = {
        "sidecar": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
        "metric_cache": str(metric_cache),
        "metric_cache_sha256": payload["metric_cache"]["sha256"],
        "makegrid": str(makegrid),
        "makegrid_sha256": payload["makegrid"]["sha256"],
        "reference_magnetic_field": B0,
        "continuous_evaluator_period": float(evaluator.period),
        "analytic_mms_eta_period": float(payload["analytic_mms_eta_period"]),
        "qualification": payload.get("qualification"),
    }
    return reference


FIELDS = ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity")
EVOLVED_FIELDS = ("density", "Te", "Ti", "Vi", "Ve", "vorticity")


@dataclass(frozen=True)
class PointData:
    """Values, exact time derivatives, gradients, and Hessians at points."""

    values: Mapping[str, np.ndarray]
    time_derivatives: Mapping[str, np.ndarray]
    gradients: Mapping[str, np.ndarray]
    hessians: Mapping[str, np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.values[name]


@dataclass(frozen=True)
class PreparedGeometry:
    """Geometry coefficients cached for repeated MMS stage evaluations."""

    q: np.ndarray
    J: np.ndarray
    B: np.ndarray
    b: np.ndarray
    bcov: np.ndarray
    div_b: np.ndarray
    K: np.ndarray
    perpendicular_flux_tensor: np.ndarray
    perpendicular_flux_divergence: np.ndarray
    # Populated by the MMS projector for the generalized-potential mode.
    mms_omega: np.ndarray | None = None
    mms_omega_gradient: np.ndarray | None = None
    mms_omega_hessian: np.ndarray | None = None
    shape: tuple[int, ...] = ()


def _as_points(points: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    q = np.asarray(points, dtype=np.float64)
    if q.shape == (3,):
        return q.reshape(1, 3), ()
    if q.ndim < 2 or q.shape[-1] != 3:
        raise ValueError("logical points must have shape (..., 3)")
    return q.reshape((-1, 3)), q.shape[:-1]


def _radial(u: np.ndarray, power: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``u**power*(1-u**2)**4`` and its analytic derivative."""

    one = 1.0 - u * u
    base = np.power(u, power) * np.power(one, 4)
    if power == 0:
        derivative = -8.0 * u * np.power(one, 3)
    else:
        derivative = np.power(u, power - 1) * np.power(one, 3) * (
            power * one - 8.0 * u * u
        )
    return base, derivative


def _radial_second(u: np.ndarray, power: int) -> np.ndarray:
    """Return the analytic second derivative of ``u**power*(1-u**2)**4``."""

    one = 1.0 - u * u
    if power == 0:
        return -8.0 * np.power(one, 3) + 48.0 * u * u * np.power(one, 2)
    result = np.zeros_like(u)
    if power >= 2:
        result = result + power * (power - 1) * np.power(u, power - 2) * np.power(one, 4)
    return result - (16.0 * power + 8.0) * np.power(u, power) * np.power(one, 3) + 48.0 * np.power(u, power + 2) * np.power(one, 2)


def _term(
    q: np.ndarray,
    t: float,
    amplitude: float,
    m: int,
    n: int,
    frequency: float,
    phase: float,
    sine: bool,
    eta_period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One regular Fourier term and its ``u,theta,eta,t`` derivatives."""

    u, theta, eta = q.T
    radial, radial_u = _radial(u, abs(int(m)))
    eta_wave = 2.0 * np.pi * float(n) / float(eta_period)
    angle = float(m) * theta + eta_wave * eta + float(frequency) * float(t) + float(phase)
    trig = np.sin(angle) if sine else np.cos(angle)
    trig_d = np.cos(angle) if sine else -np.sin(angle)
    value = amplitude * radial * trig
    return (
        value,
        amplitude * radial_u * trig,
        amplitude * radial * trig_d * float(m),
        amplitude * radial * trig_d * eta_wave,
        amplitude * radial * trig_d * float(frequency),
    )


def _term_with_hessian(
    q: np.ndarray,
    t: float,
    amplitude: float,
    m: int,
    n: int,
    frequency: float,
    phase: float,
    sine: bool,
    eta_period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One regular Fourier term, including its analytic logical Hessian."""

    value, du, dtheta, deta, dt = _term(
        q, t, amplitude, m, n, frequency, phase, sine, eta_period
    )
    u, theta, eta = q.T
    radial, radial_u = _radial(u, abs(int(m)))
    radial_uu = _radial_second(u, abs(int(m)))
    eta_wave = 2.0 * np.pi * float(n) / float(eta_period)
    angle = float(m) * theta + eta_wave * eta + float(frequency) * float(t) + float(phase)
    trig = np.sin(angle) if sine else np.cos(angle)
    trig_d = np.cos(angle) if sine else -np.sin(angle)
    hessian = np.zeros((q.shape[0], 3, 3), dtype=np.float64)
    hessian[:, 0, 0] = amplitude * radial_uu * trig
    hessian[:, 0, 1] = hessian[:, 1, 0] = amplitude * radial_u * trig_d * float(m)
    hessian[:, 0, 2] = hessian[:, 2, 0] = amplitude * radial_u * trig_d * eta_wave
    hessian[:, 1, 1] = -amplitude * radial * trig * float(m) ** 2
    hessian[:, 1, 2] = hessian[:, 2, 1] = -amplitude * radial * trig * float(m) * eta_wave
    hessian[:, 2, 2] = -amplitude * radial * trig * eta_wave ** 2
    return value, du, dtheta, deta, dt, hessian


def _sum_terms(q: np.ndarray, t: float, terms: tuple[tuple[Any, ...], ...], eta_period: float):
    out = [np.zeros(q.shape[0], dtype=np.float64) for _ in range(5)]
    for term_args in terms:
        for i, value in enumerate(_term(q, t, *term_args, eta_period=eta_period)):
            out[i] += value
    return tuple(out)


def _sum_terms_with_hessian(
    q: np.ndarray,
    t: float,
    terms: tuple[tuple[Any, ...], ...],
    eta_period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out = [np.zeros(q.shape[0], dtype=np.float64) for _ in range(5)]
    hessian = np.zeros((q.shape[0], 3, 3), dtype=np.float64)
    for term_args in terms:
        values = _term_with_hessian(q, t, *term_args, eta_period=eta_period)
        for index, value in enumerate(values[:5]):
            out[index] += value
        hessian += values[5]
    return (*out, hessian)


class ContinuumMmsReference:
    """Pointwise bulk EB continuum reference on an arbitrary smooth metric.

    ``metric_evaluator`` is the smooth evaluator produced by the HSX metric
    builder.  ``bfield_evaluator`` is its matching Cartesian magnetic-field
    evaluator.  ``B0`` must be the same fixed reference field used when the
    production geometry was assembled; the reference normalizes both the
    contravariant field and its magnitude by this value.
    """

    def __init__(
        self,
        metric_evaluator: Any,
        bfield_evaluator: Any,
        B0: float,
        *,
        tau: float = 1.0,
        mi_over_me: float = 1836.0,
        rho_star: float = 1.0,
        Ve_nu: float = 1.0e-3,
        perp_diffusion: float = 0.0,
        finite_difference_step: float = 2.0e-4,
        enable_generalized_potential: bool = False,
        eta_period: float | None = None,
        metric_query_batch_size: int = 4096,
    ) -> None:
        if not np.isfinite(B0) or B0 <= 0.0:
            raise ValueError("B0 must be finite and positive")
        self.metric_evaluator = metric_evaluator
        self.bfield_evaluator = bfield_evaluator
        self.B0 = float(B0)
        self.tau = float(tau)
        self.mi_over_me = float(mi_over_me)
        self.rho_star = float(rho_star)
        self.Ve_nu = float(Ve_nu)
        self.perp_diffusion = float(perp_diffusion)
        if not np.isfinite(self.perp_diffusion) or self.perp_diffusion < 0.0:
            raise ValueError("perp_diffusion must be finite and nonnegative")
        self.finite_difference_step = float(finite_difference_step)
        if self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive")
        self.enable_generalized_potential = bool(enable_generalized_potential)
        if isinstance(metric_query_batch_size, (bool, np.bool_)):
            raise ValueError("metric_query_batch_size must be a positive integer")
        self.metric_query_batch_size = int(metric_query_batch_size)
        if self.metric_query_batch_size < 1:
            raise ValueError("metric_query_batch_size must be a positive integer")
        self.eta_period = float(
            getattr(metric_evaluator, "period", 2.0 * np.pi)
            if eta_period is None else eta_period
        )
        if not np.isfinite(self.eta_period) or self.eta_period <= 0.0:
            raise ValueError("eta_period must be positive and finite")
        # A static, axis-regular generalized potential.  The radial envelope
        # in ``_term`` supplies u**|m| at the axis and vanishes smoothly at
        # the outer radial edge.  Two angular modes keep the resulting
        # polarization/vorticity field nontrivial without introducing time
        # dependence into omega.
        self._psi_terms = (
            (0.012, 1, 2, 0.0, 0.15, False),
            (0.006, 2, -1, 0.0, -0.40, True),
        )

    def _metric_batch(self, q: np.ndarray) -> dict[str, np.ndarray]:
        metric = self.metric_evaluator.evaluate(q, reject_nonpositive_J=False)
        if hasattr(self.metric_evaluator, "project_magnetic_field"):
            magnetic = self.metric_evaluator.project_magnetic_field(
                metric, self.bfield_evaluator
            )
        else:
            magnetic = self.metric_evaluator.evaluate_magnetic_field(
                q, self.bfield_evaluator, reject_nonpositive_J=False
            )
        J = np.asarray(metric.signed_J, dtype=np.float64)
        gcov = np.asarray(metric.covariant_metric, dtype=np.float64)
        gcontra = np.asarray(metric.contravariant_metric, dtype=np.float64)
        bcontra = np.asarray(magnetic.B_contravariant, dtype=np.float64) / self.B0
        bmag = np.asarray(magnetic.magnitude, dtype=np.float64) / self.B0
        bmag = np.maximum(bmag, 1.0e-30)
        bunit = bcontra / bmag[..., None]
        bcov = np.einsum("...ij,...j->...i", gcov, bunit)
        return {"J": J, "gcov": gcov, "gcontra": gcontra, "b": bunit, "bcov": bcov, "B": bmag}

    def _metric(self, q: np.ndarray) -> dict[str, np.ndarray]:
        points = np.asarray(q, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("metric queries must have shape (n, 3)")
        size = self.metric_query_batch_size
        if len(points) <= size:
            return self._metric_batch(points)
        chunks = [
            self._metric_batch(points[first : first + size])
            for first in range(0, len(points), size)
        ]
        return {
            name: np.concatenate([chunk[name] for chunk in chunks], axis=0)
            for name in chunks[0]
        }

    def prepare(self, points: Any) -> PreparedGeometry:
        """Evaluate and cache all geometry coefficients used by the RHS."""

        q, shape = _as_points(points)
        metric = self._metric(q)
        perpendicular_flux_tensor, perpendicular_flux_divergence = (
            self._perpendicular_geometry(q)
        )
        prepared = PreparedGeometry(
            q=q.copy(),
            J=metric["J"].copy(),
            B=metric["B"].copy(),
            b=metric["b"].copy(),
            bcov=metric["bcov"].copy(),
            div_b=self._div_b(q).copy(),
            K=self._curvature(q).copy(),
            perpendicular_flux_tensor=perpendicular_flux_tensor.copy(),
            perpendicular_flux_divergence=perpendicular_flux_divergence.copy(),
            shape=shape,
        )
        if self.enable_generalized_potential:
            _, psi_du, psi_dtheta, psi_deta, _, psi_hessian = self._psi_raw(q)
            psi_gradient = np.stack((psi_du, psi_dtheta, psi_deta), axis=-1)
            omega = self._perpendicular_operator(
                q, psi_gradient, psi_hessian, prepared=prepared
            )
            prepared = PreparedGeometry(
                q=prepared.q,
                J=prepared.J,
                B=prepared.B,
                b=prepared.b,
                bcov=prepared.bcov,
                div_b=prepared.div_b,
                K=prepared.K,
                perpendicular_flux_tensor=prepared.perpendicular_flux_tensor,
                perpendicular_flux_divergence=prepared.perpendicular_flux_divergence,
                mms_omega=np.asarray(omega, dtype=np.float64).copy(),
                mms_omega_gradient=prepared.mms_omega_gradient,
                mms_omega_hessian=prepared.mms_omega_hessian,
                shape=prepared.shape,
            )
        return prepared

    def _shift(self, q: np.ndarray, axis: int, amount: float) -> np.ndarray:
        result = q.copy()
        result[:, axis] += amount
        if axis == 1:
            result[:, axis] = np.mod(result[:, axis], 2.0 * np.pi)
        elif axis == 2:
            result[:, axis] = np.mod(result[:, axis], self.eta_period)
        return result

    def _step(self, q: np.ndarray) -> np.ndarray:
        # Radial cell-centre and quadrature points are interior.  Shrinking the
        # step near either end keeps the independent stencil in the domain.
        h = np.full(q.shape[0], self.finite_difference_step, dtype=np.float64)
        h = np.minimum(h, 0.2 * np.maximum(q[:, 0], 1.0e-10))
        h = np.minimum(h, 0.2 * np.maximum(1.0 - q[:, 0], 1.0e-10))
        return np.maximum(h, 1.0e-8)

    def _derivative(self, function, q: np.ndarray, axis: int) -> np.ndarray:
        h = self._step(q)
        values = (
            function(self._shift(q, axis, -2.0 * h)),
            function(self._shift(q, axis, -h)),
            function(self._shift(q, axis, h)),
            function(self._shift(q, axis, 2.0 * h)),
        )
        denominator = 12.0 * h
        if np.ndim(values[0]) > 1:
            denominator = denominator.reshape(
                (q.shape[0],) + (1,) * (np.ndim(values[0]) - 1)
            )
        return (values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / denominator

    def _perpendicular_flux_tensor(self, q: np.ndarray) -> np.ndarray:
        """Return ``J*(g^ij-b^i b^j)`` for the perpendicular operator."""

        metric = self._metric(q)
        projector = metric["gcontra"] - np.einsum(
            "...i,...j->...ij", metric["b"], metric["b"]
        )
        return metric["J"][..., None, None] * projector

    def _perpendicular_geometry(
        self, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cache the metric-only tensor and its logical divergence once."""

        tensor = self._perpendicular_flux_tensor(q)
        derivatives = np.stack(
            tuple(self._derivative(self._perpendicular_flux_tensor, q, axis)
                  for axis in range(3)),
            axis=1,
        )
        # derivatives[..., i, k, j] = d_i [J P^(k j)]; contract the flux
        # index k with the derivative index i.
        divergence = np.einsum("niij->nj", derivatives)
        return tensor, divergence

    def _perpendicular_operator(
        self,
        q: np.ndarray,
        gradient: np.ndarray,
        hessian: np.ndarray,
        *,
        prepared: PreparedGeometry | None = None,
    ) -> np.ndarray:
        """Evaluate ``J^-1 d_i[J P^ij d_j f]`` from analytic field data."""

        if prepared is None:
            metric = self._metric(q)
            tensor, divergence = self._perpendicular_geometry(q)
            J = metric["J"]
        else:
            tensor = prepared.perpendicular_flux_tensor
            divergence = prepared.perpendicular_flux_divergence
            J = prepared.J
        numerator = np.einsum("...j,...j->...", divergence, gradient)
        numerator += np.einsum("...ij,...ij->...", tensor, hessian)
        return numerator / np.maximum(np.abs(J), 1.0e-30)

    def _curvature(self, q: np.ndarray) -> np.ndarray:
        def covariant_over_B(x: np.ndarray) -> np.ndarray:
            metric = self._metric(x)
            return metric["bcov"] / metric["B"][..., None]

        # curl(A)_u = d_theta A_eta - d_eta A_theta, etc.; K = B/(2J) curl(A).
        dtheta = self._derivative(covariant_over_B, q, 1)
        deta = self._derivative(covariant_over_B, q, 2)
        du = self._derivative(covariant_over_B, q, 0)
        curl = np.stack((dtheta[..., 2] - deta[..., 1], deta[..., 0] - du[..., 2], du[..., 1] - dtheta[..., 0]), axis=-1)
        metric = self._metric(q)
        return 0.5 * metric["B"][..., None] * curl / np.maximum(np.abs(metric["J"])[..., None], 1.0e-30)

    def _div_b(self, q: np.ndarray) -> np.ndarray:
        """Return ``div(b)=B*b.grad(1/B)`` from the smooth metric field."""

        def inverse_b(x: np.ndarray) -> np.ndarray:
            return 1.0 / np.maximum(self._metric(x)["B"], 1.0e-30)

        gradient_inverse_b = np.stack(
            tuple(self._derivative(inverse_b, q, axis) for axis in range(3)),
            axis=-1,
        )
        metric = self._metric(q)
        return metric["B"] * np.einsum(
            "...i,...i->...", metric["b"], gradient_inverse_b
        )

    def _polarization(self, q: np.ndarray, field_gradient) -> np.ndarray:
        def flux(x: np.ndarray) -> np.ndarray:
            metric = self._metric(x)
            grad = field_gradient(x)
            projector = metric["gcontra"] - np.einsum("...i,...j->...ij", metric["b"], metric["b"])
            return metric["J"][..., None] * np.einsum("...ij,...j->...i", projector, grad)

        metric = self._metric(q)
        divergence = np.zeros(q.shape[0], dtype=np.float64)
        for axis in range(3):
            divergence += self._derivative(lambda x, axis=axis: flux(x)[..., axis], q, axis)
        return divergence / np.maximum(np.abs(metric["J"]), 1.0e-30)

    def _psi_raw(
        self, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return static generalized-potential values and analytic derivatives."""

        return _sum_terms_with_hessian(
            q, 0.0, self._psi_terms, self.eta_period
        )

    def _fields_raw(self, q: np.ndarray, t: float) -> dict[str, tuple[np.ndarray, ...]]:
        # Each term tuple is (amplitude,m,n,frequency,phase,sine).  The eta
        # wave number is multiplied by 2*pi/period, so every term is periodic
        # on the evaluator's one-field-period interval.  The production HSX
        # geometry repeats this interval over the full torus.
        terms = {
            "density": ((0.070, 1, -1, 0.65, 0.10, False), (0.030, 2, 2, 1.10, 0.40, True)),
            "Te": ((0.040, 1, 2, 0.80, -0.20, False), (0.018, 3, -1, 1.25, 0.50, True)),
            "Ti": ((0.035, 1, -1, 0.90, 0.35, True), (0.020, 2, 2, 0.55, -0.15, False)),
            "Vi": ((0.018, 1, 1, 1.05, -0.10, False), (0.009, 2, -2, 0.75, 0.60, True)),
            "Ve": ((0.014, 1, -2, 1.20, 0.30, True), (0.008, 3, 1, 0.60, -0.45, False)),
        }
        raw = {
            name: _sum_terms_with_hessian(q, t, spec, self.eta_period)
            for name, spec in terms.items()
        }
        # The default regression mode chooses phi=-tau*(Ti-1), making omega=0
        # exactly.  Production MMS enables a static generalized potential psi:
        # phi=-tau*(Ti-1)+psi, with omega=L_perp psi assembled independently
        # from the cached metric tensor/divergence in ``prepare``.
        phi = tuple(-self.tau * value for value in raw["Ti"])
        if self.enable_generalized_potential:
            psi = self._psi_raw(q)
            phi = tuple(a + b for a, b in zip(phi, psi))
        raw["phi"] = phi
        return raw

    def evaluate(
        self,
        points: Any,
        time: float,
        *,
        prepared: PreparedGeometry | None = None,
    ) -> PointData:
        q, shape = _as_points(points)
        if prepared is not None and q.shape != prepared.q.shape:
            raise ValueError("prepared geometry does not match points")
        raw = self._fields_raw(q, float(time))
        values: dict[str, np.ndarray] = {}
        derivatives: dict[str, np.ndarray] = {}
        gradients: dict[str, np.ndarray] = {}
        hessians: dict[str, np.ndarray] = {}
        constants = {"density": 1.0, "Te": 1.0, "Ti": 1.0, "phi": 0.0, "Vi": 0.0, "Ve": 0.0}
        for name, (value, du, dtheta, deta, dt, hessian) in raw.items():
            values[name] = value + constants[name]
            derivatives[name] = dt
            gradients[name] = np.stack((du, dtheta, deta), axis=-1)
            hessians[name] = hessian

        if not self.enable_generalized_potential:
            values["vorticity"] = np.zeros(q.shape[0], dtype=np.float64)
            derivatives["vorticity"] = np.zeros(q.shape[0], dtype=np.float64)
            gradients["vorticity"] = np.zeros((q.shape[0], 3), dtype=np.float64)
            hessians["vorticity"] = np.zeros((q.shape[0], 3, 3), dtype=np.float64)
        else:
            if prepared is None:
                prepared = self.prepare(q)
            omega = prepared.mms_omega
            if omega is None:
                raise RuntimeError(
                    "generalized-potential reference requires prepared mms_omega"
                )
            values["vorticity"] = np.asarray(omega, dtype=np.float64)
            derivatives["vorticity"] = np.zeros(q.shape[0], dtype=np.float64)
            gradient = prepared.mms_omega_gradient
            hessian = prepared.mms_omega_hessian
            # The projector supplies structured fourth-order derivatives for
            # production chunks.  A small local fallback keeps direct point
            # calls useful in tests without importing production operators.
            if gradient is None or hessian is None:
                gradient, hessian = self._local_omega_derivatives(q)
            gradients["vorticity"] = np.asarray(gradient, dtype=np.float64)
            hessians["vorticity"] = np.asarray(hessian, dtype=np.float64)
        if shape:
            values = {name: val.reshape(shape) for name, val in values.items()}
            derivatives = {name: val.reshape(shape) for name, val in derivatives.items()}
            gradients = {name: val.reshape(shape + (3,)) for name, val in gradients.items()}
            hessians = {name: val.reshape(shape + (3, 3)) for name, val in hessians.items()}
        return PointData(values, derivatives, gradients, hessians)

    def _local_omega_derivatives(
        self, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fourth-order local fallback for non-projector diagnostic calls."""

        def omega_scalar(x: np.ndarray) -> np.ndarray:
            prepared = self.prepare(x)
            if prepared.mms_omega is None:
                raise RuntimeError("generalized-potential omega was not prepared")
            return np.asarray(prepared.mms_omega, dtype=np.float64)

        gradient = np.stack(
            tuple(self._derivative(omega_scalar, q, axis) for axis in range(3)),
            axis=-1,
        )
        # Differentiate each component of the gradient.  ``hessian[i,j]`` is
        # d/dq_i (grad_j), matching the field Hessian convention above.
        hessian = np.empty((q.shape[0], 3, 3), dtype=np.float64)
        for outer_axis in range(3):
            for inner_axis in range(3):
                hessian[:, outer_axis, inner_axis] = self._derivative(
                    lambda x, axis=inner_axis: self._derivative(
                        omega_scalar, x, axis
                    ),
                    q,
                    outer_axis,
                )
        return gradient, hessian

    def continuum_rhs(
        self, points: Any, time: float, *, prepared: PreparedGeometry | None = None
    ) -> dict[str, np.ndarray]:
        q, shape = _as_points(points) if prepared is None else (prepared.q, prepared.shape)
        if prepared is not None and q.shape != prepared.q.shape:
            raise ValueError("prepared geometry does not match points")
        data = self.evaluate(q, time, prepared=prepared)
        rhs = self._continuum_rhs_from_data(q, data, prepared=prepared)
        if shape:
            rhs = {name: value.reshape(shape) for name, value in rhs.items()}
        return rhs

    def continuum_terms(
        self, points: Any, time: float, *, prepared: PreparedGeometry | None = None
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return the independent continuum RHS split by production ledger lane."""

        q, shape = _as_points(points) if prepared is None else (
            prepared.q,
            prepared.shape,
        )
        if prepared is not None and q.shape != prepared.q.shape:
            raise ValueError("prepared geometry does not match points")
        data = self.evaluate(q, time, prepared=prepared)
        terms = self._continuum_terms_from_data(q, data, prepared=prepared)
        if shape:
            terms = {
                field: {
                    name: value.reshape(shape) for name, value in field_terms.items()
                }
                for field, field_terms in terms.items()
            }
        return terms

    def _continuum_rhs_from_data(
        self,
        q: np.ndarray,
        data: PointData,
        *,
        prepared: PreparedGeometry | None = None,
    ) -> dict[str, np.ndarray]:
        terms = self._continuum_terms_from_data(q, data, prepared=prepared)
        return {
            field: np.sum(np.stack(tuple(field_terms.values())), axis=0)
            for field, field_terms in terms.items()
        }

    def _continuum_terms_from_data(
        self,
        q: np.ndarray,
        data: PointData,
        *,
        prepared: PreparedGeometry | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        v, g = data.values, data.gradients
        h = data.hessians
        metric = self._metric(q) if prepared is None else None
        bmag = metric["B"] if prepared is None else prepared.B
        b = metric["b"] if prepared is None else prepared.b
        bcov = metric["bcov"] if prepared is None else prepared.bcov
        J = metric["J"] if prepared is None else prepared.J
        div_b = self._div_b(q) if prepared is None else prepared.div_b
        poisson = lambda a, c: np.sum(bcov * np.cross(a, c), axis=-1) / np.maximum(np.abs(J), 1.0e-30)
        parallel = lambda a: np.einsum("...i,...i->...", b, a)
        perpendicular = lambda name: self._perpendicular_operator(
            q, g[name], h[name], prepared=prepared
        )
        n = np.maximum(v["density"], 1.0e-30)
        Pe = n * v["Te"]
        Pi = n * v["Ti"]
        current = n * (v["Vi"] - v["Ve"])
        Peg = v["Te"][..., None] * g["density"] + n[..., None] * g["Te"]
        Pig = v["Ti"][..., None] * g["density"] + n[..., None] * g["Ti"]
        pg = Peg + self.tau * Pig
        cg = (v["Vi"] - v["Ve"])[..., None] * g["density"] + n[..., None] * (g["Vi"] - g["Ve"])
        dfg = v["Ve"][..., None] * g["density"] + n[..., None] * g["Ve"]
        K = self._curvature(q) if prepared is None else prepared.K
        curv = lambda a: np.einsum("...i,...i->...", K, a)
        zero = np.zeros_like(n)
        terms = {
            "density": {
                "poisson_bracket": -poisson(g["phi"], g["density"]) / (self.rho_star * bmag),
                "parallel_density_flux_divergence": -parallel(dfg) - n * v["Ve"] * div_b,
                "curvature": 2.0 * (curv(Peg) - n * curv(g["phi"])) / bmag,
                "perpendicular_diffusion": self.perp_diffusion * perpendicular("density"),
            },
            "Te": {
                "poisson_bracket": -poisson(g["phi"], g["Te"]) / (self.rho_star * bmag),
                "parallel_advection": -v["Ve"] * parallel(g["Te"]) + 2.0 * v["Te"] / (3.0 * n) * (0.71 * parallel(cg) - n * parallel(g["Ve"]) + (0.71 * current - n * v["Ve"]) * div_b),
                "curvature": 4.0 * v["Te"] / (3.0 * bmag) * (curv(Peg) / n + 2.5 * curv(g["Te"]) - curv(g["phi"])),
                "parallel_compression": zero,
                "perpendicular_diffusion": self.perp_diffusion * perpendicular("Te"),
            },
            "Ti": {
                "poisson_bracket": -poisson(g["phi"], g["Ti"]) / (self.rho_star * bmag),
                "parallel_advection": -v["Vi"] * parallel(g["Ti"]) + 2.0 * v["Ti"] / (3.0 * n) * (parallel(cg) - n * parallel(g["Vi"]) + (current - n * v["Vi"]) * div_b),
                "curvature": 4.0 * v["Ti"] / (3.0 * bmag) * (curv(Peg) / n - 2.5 * self.tau * curv(g["Ti"]) - curv(g["phi"])),
                "parallel_compression": zero,
                "perpendicular_diffusion": self.perp_diffusion * perpendicular("Ti"),
            },
            "Vi": {
                "poisson_bracket": -poisson(g["phi"], g["Vi"]) / (self.rho_star * bmag),
                "parallel_self_advection": -v["Vi"] * parallel(g["Vi"]) - parallel(pg) / n,
                "parallel_pressure": zero,
                "perpendicular_diffusion": self.perp_diffusion * perpendicular("Vi"),
            },
            "Ve": {
                "poisson_bracket": -poisson(g["phi"], g["Ve"]) / (self.rho_star * bmag),
                "parallel_self_advection": -v["Ve"] * parallel(g["Ve"]) - self.mi_over_me * (parallel(Peg) / n + 0.71 * parallel(g["Te"]) + self.tau * parallel(g["Ti"])),
                "collision": self.mi_over_me * self.Ve_nu * current,
                "electrostatic": self.mi_over_me * (parallel(g["phi"]) + self.tau * parallel(g["Ti"])),
                "electron_pressure": zero,
                "thermal_force": zero,
                "perpendicular_diffusion": self.perp_diffusion * perpendicular("Ve"),
            },
            "vorticity": {
                "poisson_bracket": -poisson(g["phi"], g["vorticity"]) / (self.rho_star * bmag),
                "parallel_advection": -v["Vi"] * parallel(g["vorticity"]),
                "parallel_current": bmag * bmag * (parallel(cg) + current * div_b) / n,
                "curvature": 2.0 * bmag * curv(pg) / n,
                "perpendicular_diffusion": self.perp_diffusion * perpendicular("vorticity"),
            },
        }
        return terms

    def source(
        self, points: Any, time: float, *, prepared: PreparedGeometry | None = None
    ) -> dict[str, np.ndarray]:
        q, shape = _as_points(points) if prepared is None else (prepared.q, prepared.shape)
        if prepared is not None and q.shape != prepared.q.shape:
            raise ValueError("prepared geometry does not match points")
        data = self.evaluate(q, time, prepared=prepared)
        rhs = self._continuum_rhs_from_data(q, data, prepared=prepared)
        result = {name: data.time_derivatives[name] - rhs[name] for name in EVOLVED_FIELDS}
        if shape:
            result = {name: value.reshape(shape) for name, value in result.items()}
        return result


def analytic_identity_metric_self_test() -> dict[str, float]:
    """Small smoke test for the reference without HSX files or operators."""

    class IdentityMetric:
        period = 2.0 * np.pi

        def evaluate(self, points, *, reject_nonpositive_J=False):
            q, _ = _as_points(points)
            n = q.shape[0]
            eye = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
            return SimpleNamespace(signed_J=np.ones(n), covariant_metric=eye, contravariant_metric=eye)

        def evaluate_magnetic_field(self, points, bfield, *, reject_nonpositive_J=False):
            q, _ = _as_points(points)
            return SimpleNamespace(
                B_contravariant=np.broadcast_to(np.array([0.0, 0.0, 1.0]), (q.shape[0], 3)).copy(),
                magnitude=np.ones(q.shape[0]),
            )

    class IdentityB:
        def evaluate_cartesian(self, positions):
            return np.broadcast_to(np.array([0.0, 0.0, 1.0]), np.asarray(positions).shape)

    reference = ContinuumMmsReference(IdentityMetric(), IdentityB(), 1.0, finite_difference_step=1.0e-4)
    points = np.array([[0.25, 0.3, 0.7], [0.55, 2.1, 4.2]], dtype=np.float64)
    derivative = reference._derivative(lambda x: x[:, 0] ** 3, points, 0)
    assert np.allclose(derivative, 3.0 * points[:, 0] ** 2, rtol=1.0e-8, atol=1.0e-10)
    polarization = reference._polarization(
        points, lambda x: np.stack((x[:, 0], np.zeros(x.shape[0]), np.zeros(x.shape[0])), axis=-1)
    )
    assert np.allclose(polarization, 1.0, rtol=1.0e-8, atol=1.0e-8)
    data = reference.evaluate(points, 0.11)
    source = reference.source(points, 0.11)
    prepared = reference.prepare(points)
    rhs = reference.continuum_rhs(points, 0.11)
    cached_rhs = reference.continuum_rhs(points, 0.11, prepared=prepared)
    assert all(np.allclose(rhs[name], cached_rhs[name], rtol=1.0e-11, atol=1.0e-11) for name in EVOLVED_FIELDS)
    terms = reference.continuum_terms(points, 0.11, prepared=prepared)
    term_closure = max(
        float(np.max(np.abs(
            np.sum(np.stack(tuple(terms[field].values())), axis=0)
            - rhs[field]
        )))
        for field in EVOLVED_FIELDS
    )
    assert term_closure <= 1.0e-13
    finite = float(max(np.max(np.abs(value)) for value in source.values()))
    assert all(np.all(np.isfinite(value)) for value in source.values())
    assert np.max(np.abs(data.values["vorticity"])) < 100.0
    return {
        "max_source": finite,
        "max_vorticity": float(np.max(np.abs(data.values["vorticity"]))),
        "max_term_closure": term_closure,
    }


__all__ = [
    "ContinuumMmsReference",
    "PointData",
    "PreparedGeometry",
    "analytic_identity_metric_self_test",
    "FIELDS",
    "EVOLVED_FIELDS",
]
