"""The eta-filtered logical magnetic field: a low-resolution MMS verification device (production uses the raw field).

The MAKEGRID field of HSX carries coil-ripple at high per-period toroidal harmonics (per-period ``m = 12``, ``k = 48``
per turn) that a low-resolution logical grid cannot resolve; through the curvature ``K ~ curl(b_cov / B)`` the ripple is
amplified roughly ``k``-fold.  The *eta filter* low-passes the contravariant flux density ``F^i = J B^i`` along ``eta``
at fixed logical ``(u, theta)``, keeping the per-field-period harmonics ``m <= M`` (``M = 3`` with ``nfp = 4``: ``k <=
12`` per turn, the largest nfp harmonic below the Nyquist of 32 eta cells per turn), and defines::

    B~^i = F~^i / J,         B~_cart = (dX/dq) B~^i,         h = b~_cov / |B~|,   |J|, K  all from B~.

The filter is a linear operator in ``eta`` only, so it commutes with ``d_u, d_theta, d_eta``: ``d_i (J B~^i) = 0`` holds
whenever it holds for the raw field, and the filter is local to the plasma (it never mixes ``(u, theta)`` columns).
The logical map ``(u, theta, eta)`` is shared with the raw arm.

Two forms are provided (both float64):

* :class:`EtaFilteredColumnField` -- the *column-exact* NumPy form (the oracle).  For every unique ``(u, theta)``
  it samples ``F`` at ``samples_per_period`` values of ``eta`` over one field period, takes the real FFT, keeps ``m <=
  M`` and evaluates the trigonometric polynomial at the requested ``eta``.  Works for any points, one column per
  distinct ``(u, theta)`` (columns are cached), but does not support autodiff.
* :class:`EtaFilterTable` -- the *tabulated* form: the coefficients ``f^i_m(u, theta)`` (``m <= M``, cos/sin) on a
  uniform ``(u, theta)`` grid (default 257 x 256, ``u in [0, 1.02]``, about 11 MB), interpolated by a tensor-product cubic
  B-spline (not-a-knot in ``u``, periodic in ``theta``), plus the exact trigonometric evaluation in ``eta``.
  :class:`JaxEtaFilterTable` is its JAX twin (a PyTree; differentiable), which
  :func:`drbx.geometry.curvature_autodiff.covariant_over_B_parts` accepts as ``logical_field``.
  :meth:`EtaFilterTable.check_against_columns` is the one-off equivalence check against the column form.

All forms share the interface ``contravariant(q, metric) -> B~^i`` (physical units), ``cartesian(q, metric)`` and
``flux_density(q)`` where ``q`` has shape ``(..., 3)`` ``(u, theta, eta)`` and ``metric`` is a metric evaluation of
``q`` (anything with ``.J`` the signed Jacobian and ``.jacobian_matrix``, NumPy or JAX; the NumPy forms evaluate
it themselves when it is omitted).

Quantities, once more: ``F`` and ``B~`` are in the units of the supplied B evaluator (Tesla for MAKEGRID), the
Jacobian is the signed determinant of ``dX/d(u, theta, eta)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .MetricEvaluator import MagneticFieldEvaluation

__all__ = [
    "ETA_FILTER_QUANTITY",
    "DEFAULT_TABLE_GRID",
    "EtaFilterSpec",
    "EtaFilteredColumnField",
    "EtaFilterTable",
    "JaxEtaFilterTable",
    "JaxMetricView",
    "flux_density",
]

#: the filtered quantity: the contravariant flux density ``J * B^i``
ETA_FILTER_QUANTITY = "J*B^i"
#: ``(nu, ntheta, u_max)`` of the tabulated form
DEFAULT_TABLE_GRID = (257, 256, 1.02)
#: ``u`` at which the table's axis row is evaluated (the logical frame is singular at ``u = 0``)
_U_AXIS = 1.0e-6


# ---------------------------------------------------------------------------------------------------------------------
# The option
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EtaFilterSpec:
    """The filter option: keep the per-field-period harmonics ``m <= max_harmonic_per_period`` of ``J B^i`` in eta.

    ``nfp`` is the number of field periods (the filter period is ``2 pi / nfp``), ``samples_per_period`` the number
    of equispaced eta samples of one period of a column.  ``max_harmonic_per_period = None`` keeps every sampled
    harmonic (``m <= samples_per_period // 2``: the trigonometric interpolant of the samples, a check device).
    """

    max_harmonic_per_period: int | None = 3
    nfp: int = 4
    samples_per_period: int = 64
    quantity: str = ETA_FILTER_QUANTITY

    def __post_init__(self) -> None:
        if self.quantity != ETA_FILTER_QUANTITY:
            raise ValueError(f"quantity must be {ETA_FILTER_QUANTITY!r}, got {self.quantity!r}")
        for name in ("nfp", "samples_per_period"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            object.__setattr__(self, name, int(value))
        if self.samples_per_period < 4:
            raise ValueError("samples_per_period must be at least 4")
        if self.max_harmonic_per_period is None:
            object.__setattr__(self, "max_harmonic_per_period", self.samples_per_period // 2)
        else:
            m = self.max_harmonic_per_period
            if isinstance(m, (bool, np.bool_)) or int(m) != m or int(m) < 0:
                raise ValueError(f"max_harmonic_per_period must be a nonnegative integer or None, got {m!r}")
            object.__setattr__(self, "max_harmonic_per_period", int(m))
        if self.max_harmonic_per_period > self.samples_per_period // 2:
            raise ValueError("max_harmonic_per_period exceeds the Nyquist harmonic samples_per_period // 2")

    @classmethod
    def from_mapping(cls, option: Mapping[str, Any]) -> "EtaFilterSpec":
        """The spec of ``{"quantity", "max_harmonic_per_period", "nfp", "samples_per_period"}`` (all keys required)."""
        keys = {"quantity", "max_harmonic_per_period", "nfp", "samples_per_period"}
        if set(option) != keys:
            raise ValueError(f"eta_filter must have exactly the keys {sorted(keys)}, got {sorted(option)}")
        return cls(max_harmonic_per_period=option["max_harmonic_per_period"], nfp=option["nfp"],
                   samples_per_period=option["samples_per_period"], quantity=option["quantity"])

    def as_dict(self) -> dict:
        return {"quantity": self.quantity, "max_harmonic_per_period": self.max_harmonic_per_period, "nfp": self.nfp,
                "samples_per_period": self.samples_per_period}

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.nfp

    @property
    def n_coefficients(self) -> int:
        """Real coefficients per component: the mean, ``M`` cosines and ``M`` sines."""
        return 2 * self.max_harmonic_per_period + 1


# ---------------------------------------------------------------------------------------------------------------------
# Harmonic analysis / synthesis of a column
# ---------------------------------------------------------------------------------------------------------------------
def _harmonic_coefficients(samples: np.ndarray, spec: EtaFilterSpec) -> np.ndarray:
    """Real coefficients ``(..., 2M + 1, 3)`` ``[c0, c_1..c_M, s_1..s_M]`` of ``samples (..., NS, 3)``.

    ``F~(eta) = c0 + sum_m c_m cos(m nfp eta) + s_m sin(m nfp eta)``, the samples at ``eta_s = s * period / NS``.
    """
    ns, M = spec.samples_per_period, spec.max_harmonic_per_period
    X = np.fft.rfft(samples, axis=-2) / ns
    c = 2.0 * X[..., : M + 1, :].real
    s = -2.0 * X[..., : M + 1, :].imag
    c[..., 0, :] = X[..., 0, :].real
    if ns % 2 == 0 and M == ns // 2:       # Nyquist: cos only, unit weight
        c[..., M, :] = X[..., M, :].real
        s[..., M, :] = 0.0
    return np.concatenate([c, s[..., 1:, :]], axis=-2)


def _synthesize(coef: np.ndarray, eta: np.ndarray, spec: EtaFilterSpec) -> np.ndarray:
    """``F~`` at ``eta (n,)`` from ``coef (n, 2M + 1, 3)``."""
    M = spec.max_harmonic_per_period
    arg = np.outer(np.asarray(eta, dtype=np.float64) * spec.nfp, np.arange(1, M + 1))
    out = coef[:, 0, :].copy()
    if M:
        out += np.einsum("nm,nmc->nc", np.cos(arg), coef[:, 1:M + 1, :])
        out += np.einsum("nm,nmc->nc", np.sin(arg), coef[:, M + 1:, :])
    return out


def flux_density(metric_evaluator: Any, bfield_evaluator: Any, q: Any) -> np.ndarray:
    """The raw flux density ``F^i = J B^i`` at logical points ``q (n, 3)`` -> ``(n, 3)``."""
    q = np.asarray(q, dtype=np.float64)
    metric = metric_evaluator.evaluate(q, reject_nonpositive_J=False)
    B = np.asarray(bfield_evaluator.evaluate_cartesian(np.asarray(metric.position, dtype=np.float64)), dtype=np.float64)
    jac = np.asarray(metric.jacobian_matrix, dtype=np.float64)
    return np.asarray(metric.signed_J, dtype=np.float64)[:, None] * np.linalg.solve(jac, B[..., None])[..., 0]


class JaxMetricView:
    """NumPy-facing view of a :class:`~drbx.geometry.jax_metric_evaluator.JaxMetricEvaluator` (jitted, fixed blocks).

    It evaluates the same map as ``MetricEvaluator`` (to about 4e-15) but does not clamp ``u`` to ``[0, 1]``, which the
    table needs for its extension beyond the wall (``u`` up to ``u_max``).  ``evaluate`` returns the three attributes
    the column form reads: ``position``, ``jacobian_matrix`` and ``signed_J``.
    """

    def __init__(self, jax_metric: Any, block: int = 8192) -> None:
        self.block = int(block)

        def one_block(q):
            position, jacobian = jax_metric.position_and_jacobian(q)
            return position, jacobian, jnp.linalg.det(jacobian)

        self._evaluate = jax.jit(one_block)

    def evaluate(self, q: Any, reject_nonpositive_J: bool = False) -> SimpleNamespace:
        q = np.asarray(q, dtype=np.float64)
        position, jacobian, det = [], [], []
        for start in range(0, len(q), self.block):
            part = q[start:start + self.block]
            padded = np.concatenate([part, np.repeat(part[-1:], self.block - len(part), axis=0)])
            p, j, d = self._evaluate(jnp.asarray(padded))
            position.append(np.asarray(p)[:len(part)]); jacobian.append(np.asarray(j)[:len(part)]); det.append(np.asarray(d)[:len(part)])
        return SimpleNamespace(position=np.concatenate(position), jacobian_matrix=np.concatenate(jacobian),
                               signed_J=np.concatenate(det))


# ---------------------------------------------------------------------------------------------------------------------
# (a) Column-exact NumPy form
# ---------------------------------------------------------------------------------------------------------------------
class EtaFilteredColumnField:
    """The column-exact eta-filtered field (the oracle).

    Parameters
    ----------
    metric_evaluator, bfield_evaluator:
        The logical map (``evaluate(q, reject_nonpositive_J=False)`` with ``position``, ``jacobian_matrix``,
        ``signed_J``; its ``period`` attribute, if present, must equal ``2 pi / nfp``) and the Cartesian field
        (``evaluate_cartesian(position)``): the raw arm's evaluators.
    spec:
        :class:`EtaFilterSpec`.
    column_chunk:
        Columns sampled per evaluator call (``column_chunk * samples_per_period`` points).
    max_cached_columns:
        The coefficients of the columns seen are cached (a column is one distinct ``(u, theta)``); when the cache
        exceeds this it drops its oldest half.
    """

    def __init__(self, metric_evaluator: Any, bfield_evaluator: Any, spec: EtaFilterSpec, *, column_chunk: int = 256,
                 max_cached_columns: int = 200_000) -> None:
        period = getattr(metric_evaluator, "period", None)
        if period is not None and not np.isclose(float(period), spec.period, rtol=1e-12, atol=0.0):
            raise ValueError(f"the metric evaluator's period {float(period)!r} is not 2 pi / nfp = {spec.period!r}")
        self.metric_evaluator = metric_evaluator
        self.bfield_evaluator = bfield_evaluator
        self.spec = spec
        self.column_chunk = int(column_chunk)
        self.max_cached_columns = int(max_cached_columns)
        self._cache: dict[bytes, np.ndarray] = {}

    # -- columns -------------------------------------------------------------------------------------------------
    def column_samples(self, u: Any, theta: Any, metric_evaluator: Any = None) -> np.ndarray:
        """``F`` of the columns ``(u, theta) (n,)`` at ``samples_per_period`` equispaced eta of one period -> ``(n, NS, 3)``.

        ``metric_evaluator`` (default: the field's own) can be a :class:`JaxMetricView`, to sample beyond ``u = 1``."""
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        ns = self.spec.samples_per_period
        eta = np.arange(ns, dtype=np.float64) * (self.spec.period / ns)
        q = np.stack([np.repeat(u, ns), np.repeat(theta, ns), np.tile(eta, len(u))], axis=1)
        return flux_density(self.metric_evaluator if metric_evaluator is None else metric_evaluator,
                            self.bfield_evaluator, q).reshape(len(u), ns, 3)

    def column_coefficients(self, u: Any, theta: Any) -> np.ndarray:
        """The filtered harmonic coefficients ``(n, 2M + 1, 3)`` of the columns ``(u, theta)`` (cached)."""
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        keys = np.stack([u, theta], axis=1)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        coef = np.empty((len(unique), self.spec.n_coefficients, 3), dtype=np.float64)
        missing = []
        for index, row in enumerate(unique):
            hit = self._cache.get(row.tobytes())
            if hit is None:
                missing.append(index)
            else:
                coef[index] = hit
        for start in range(0, len(missing), self.column_chunk):
            idx = np.asarray(missing[start:start + self.column_chunk])
            fresh = _harmonic_coefficients(self.column_samples(unique[idx, 0], unique[idx, 1]), self.spec)
            coef[idx] = fresh
            if len(self._cache) + len(idx) > self.max_cached_columns:
                for key in list(self._cache)[: len(self._cache) // 2]:
                    del self._cache[key]
            for index, value in zip(idx, fresh):
                self._cache[unique[index].tobytes()] = value
        return coef[inverse]

    # -- the field -----------------------------------------------------------------------------------------------
    def flux_density(self, q: Any) -> np.ndarray:
        """``F~^i = (J B^i)~`` at ``q (..., 3)``."""
        q = np.asarray(q, dtype=np.float64)
        flat = q.reshape(-1, 3)
        coef = self.column_coefficients(flat[:, 0], flat[:, 1])
        return _synthesize(coef, flat[:, 2], self.spec).reshape(q.shape)

    def contravariant(self, q: Any, metric: Any = None) -> np.ndarray:
        """``B~^i = F~^i / J`` at ``q (..., 3)``; ``metric`` (an evaluation of ``q``) is computed when omitted."""
        q = np.asarray(q, dtype=np.float64)
        if metric is None:
            metric = self.metric_evaluator.evaluate(q, reject_nonpositive_J=False)
        return self.flux_density(q) / np.asarray(metric.signed_J, dtype=np.float64)[..., None]

    def cartesian(self, q: Any, metric: Any = None) -> np.ndarray:
        """``B~`` in Cartesian components, ``(dX/dq) B~^i``."""
        if metric is None:
            metric = self.metric_evaluator.evaluate(np.asarray(q, dtype=np.float64), reject_nonpositive_J=False)
        contra = self.contravariant(q, metric)
        return np.einsum("...ij,...j->...i", np.asarray(metric.jacobian_matrix, dtype=np.float64), contra)

    def project_magnetic_field(self, q: Any, metric: Any) -> MagneticFieldEvaluation:
        """Drop-in for ``MetricEvaluator.project_magnetic_field(metric, bfield)`` with the filtered field at ``q``."""
        jacobian = np.asarray(metric.jacobian_matrix, dtype=np.float64)
        contra = self.contravariant(q, metric)
        B = np.einsum("...ij,...j->...i", jacobian, contra)
        if not np.all(np.isfinite(B)):
            raise ValueError("the filtered field is not finite")
        cov = np.einsum("...ji,...j->...i", jacobian, B)
        return MagneticFieldEvaluation(B, contra, cov, np.linalg.norm(B, axis=-1))


# ---------------------------------------------------------------------------------------------------------------------
# Tensor-product cubic B-spline: not-a-knot in u (uniform), periodic in theta (uniform)
# ---------------------------------------------------------------------------------------------------------------------
def _bspline_prefilter(values: np.ndarray) -> np.ndarray:
    """B-spline coefficients ``(nu + 2, ntheta, ncomp)`` interpolating ``values (nu, ntheta, ncomp)``.

    ``u`` (axis 0): uniform nodes, not-a-knot ends (the fourth difference of the first/last five coefficients vanishes: no
    third-derivative jump at the second/second-to-last node, so a global cubic is reproduced exactly), coefficient
    ``c'_j`` multiplies the basis centred at node ``j - 1``.  ``theta`` (axis 1): periodic; the circulant system is solved
    by FFT.
    """
    nu, nt, nc = values.shape
    if nu < 5 or nt < 4:
        raise ValueError("the table needs at least 5 u nodes and 4 theta nodes")
    n = nu + 2
    A = np.zeros((n, n))
    for k in range(nu):
        A[k, k:k + 3] = (1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0)
    A[nu, 0:5] = (1.0, -4.0, 6.0, -4.0, 1.0)                 # no third-derivative jump at the second node
    A[nu + 1, nu - 3:nu + 2] = (1.0, -4.0, 6.0, -4.0, 1.0)   # ... and at the second-to-last node
    rhs = np.zeros((n, nt * nc))
    rhs[:nu] = values.reshape(nu, nt * nc)
    c = np.linalg.solve(A, rhs).reshape(n, nt, nc)
    spectrum = np.fft.rfft(c, axis=1)
    symbol = (4.0 + 2.0 * np.cos(2.0 * np.pi * np.arange(spectrum.shape[1]) / nt)) / 6.0
    return np.fft.irfft(spectrum / symbol[None, :, None], n=nt, axis=1)


def _basis(t, xp):
    """The four uniform cubic B-spline weights at fractional position ``t`` in ``[0, 1]`` (last axis)."""
    t2 = t * t
    t3 = t2 * t
    return xp.stack([(1.0 - t) ** 3 / 6.0, (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
                     (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, t3 / 6.0], axis=-1)


# ---------------------------------------------------------------------------------------------------------------------
# (b) Tabulated forms: NumPy and JAX
# ---------------------------------------------------------------------------------------------------------------------
class EtaFilterTable:
    """The tabulated eta-filtered field (NumPy).

    ``values`` has shape ``(nu, ntheta, 2M + 1, 3)``: the filtered harmonic coefficients of ``F`` on the grid ``u_j = j
    u_max / (nu - 1)``, ``theta_k = 2 pi k / ntheta`` (the ``u = 0`` row is evaluated at ``u = 1e-6``: the logical
    frame is singular on the axis).  Queries interpolate the coefficients by the tensor-product cubic B-spline
    (queries beyond ``u_max`` extrapolate the end cubic) and evaluate the trigonometric polynomial at ``eta``.
    """

    def __init__(self, values: np.ndarray, spec: EtaFilterSpec, u_max: float, *, metric_evaluator: Any = None,
                 meta: Mapping[str, Any] | None = None) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 4 or values.shape[2:] != (spec.n_coefficients, 3):
            raise ValueError(f"values must have shape (nu, ntheta, {spec.n_coefficients}, 3), got {values.shape}")
        self.values = values
        self.spec = spec
        self.u_max = float(u_max)
        self.nu, self.ntheta = values.shape[:2]
        self.metric_evaluator = metric_evaluator
        self.meta = dict(meta or {})
        self._coef = _bspline_prefilter(values.reshape(self.nu, self.ntheta, -1))

    # -- construction --------------------------------------------------------------------------------------------
    @classmethod
    def from_column_field(cls, column_field: EtaFilteredColumnField, *, nu: int = DEFAULT_TABLE_GRID[0],
                          ntheta: int = DEFAULT_TABLE_GRID[1], u_max: float = DEFAULT_TABLE_GRID[2],
                          chunk: int = 512, progress: Any = None, sample_metric: Any = None) -> "EtaFilterTable":
        """Tabulate ``column_field`` (the grid columns themselves are not kept in its cache).

        ``MetricEvaluator`` rejects ``u > 1`` (the table extends to ``u_max``): pass ``sample_metric``, a
        :class:`JaxMetricView` of the same map, to sample the grid with it (it is the table's ``metric_evaluator`` otherwise
        unchanged: ``column_field.metric_evaluator``)."""
        u = np.maximum(np.linspace(0.0, float(u_max), int(nu)), _U_AXIS)
        theta = np.arange(int(ntheta)) * (2.0 * np.pi / int(ntheta))
        U, T = np.meshgrid(u, theta, indexing="ij")
        flat_u, flat_t = U.ravel(), T.ravel()
        spec = column_field.spec
        values = np.empty((len(flat_u), spec.n_coefficients, 3), dtype=np.float64)
        for start in range(0, len(flat_u), int(chunk)):
            sl = slice(start, start + int(chunk))
            values[sl] = _harmonic_coefficients(column_field.column_samples(flat_u[sl], flat_t[sl], sample_metric), spec)
            if progress is not None:
                progress(min(start + int(chunk), len(flat_u)), len(flat_u))
        return cls(values.reshape(int(nu), int(ntheta), spec.n_coefficients, 3), spec, u_max,
                   metric_evaluator=column_field.metric_evaluator)

    def save(self, path: str | Path, meta: Mapping[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = {**self.meta, **dict(meta or {})}
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            np.savez(handle, values=self.values, u_max=np.array(self.u_max), spec=np.array(json.dumps(self.spec.as_dict())),
                     meta=np.array(json.dumps(merged, sort_keys=True)))
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, *, metric_evaluator: Any = None) -> "EtaFilterTable":
        with np.load(Path(path), allow_pickle=False) as data:
            spec = EtaFilterSpec.from_mapping(json.loads(str(data["spec"])))
            return cls(data["values"], spec, float(data["u_max"]), metric_evaluator=metric_evaluator,
                       meta=json.loads(str(data["meta"])))

    # -- evaluation ----------------------------------------------------------------------------------------------
    def _locate(self, u, theta):
        xu = np.asarray(u, dtype=np.float64) * ((self.nu - 1) / self.u_max)
        iu = np.clip(np.floor(xu).astype(np.int64), 0, self.nu - 2)
        tu = xu - iu
        xt = np.mod(np.asarray(theta, dtype=np.float64), 2.0 * np.pi) * (self.ntheta / (2.0 * np.pi))
        jt = np.floor(xt).astype(np.int64)
        tt = xt - jt
        return iu, tu, jt % self.ntheta, tt

    def coefficients(self, u: Any, theta: Any) -> np.ndarray:
        """The interpolated harmonic coefficients ``(n, 2M + 1, 3)`` at ``(u, theta) (n,)``."""
        u = np.atleast_1d(np.asarray(u, dtype=np.float64))
        theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
        out = np.empty((len(u), self._coef.shape[2]), dtype=np.float64)
        a = np.arange(4)
        for start in range(0, len(u), 32768):                              # bounded gather memory
            sl = slice(start, start + 32768)
            iu, tu, jt, tt = self._locate(u[sl], theta[sl])
            rows = iu[:, None] + a[None, :]                                # c' index i + a
            cols = (jt[:, None] + a[None, :] - 1) % self.ntheta             # theta index j - 1 + b
            block = self._coef[rows[:, :, None], cols[:, None, :]]          # (n, 4, 4, ncomp)
            out[sl] = np.einsum("na,nb,nabc->nc", _basis(tu, np), _basis(tt, np), block)
        return out.reshape(len(out), self.spec.n_coefficients, 3)

    def flux_density(self, q: Any) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        flat = q.reshape(-1, 3)
        return _synthesize(self.coefficients(flat[:, 0], flat[:, 1]), flat[:, 2], self.spec).reshape(q.shape)

    def _metric(self, q, metric):
        if metric is not None:
            return metric
        if self.metric_evaluator is None:
            raise ValueError("this table has no metric evaluator: pass the metric evaluation of the points")
        return self.metric_evaluator.evaluate(np.asarray(q, dtype=np.float64), reject_nonpositive_J=False)

    def contravariant(self, q: Any, metric: Any = None) -> np.ndarray:
        metric = self._metric(q, metric)
        return self.flux_density(q) / np.asarray(metric.signed_J, dtype=np.float64)[..., None]

    def cartesian(self, q: Any, metric: Any = None) -> np.ndarray:
        metric = self._metric(q, metric)
        return np.einsum("...ij,...j->...i", np.asarray(metric.jacobian_matrix, dtype=np.float64),
                         self.contravariant(q, metric))

    def to_jax(self) -> "JaxEtaFilterTable":
        return JaxEtaFilterTable(self._coef, self.spec.max_harmonic_per_period, self.spec.nfp, self.nu, self.ntheta,
                                 self.u_max)

    # -- the one-off equivalence check -------------------------------------------------------------------------------
    def check_against_columns(self, column_field: EtaFilteredColumnField, points: Any) -> dict:
        """Compare the table with the column-exact form at logical ``points (n, 3)``.

        Returns the largest ``|F~_table - F~_column|`` relative to the largest ``|F~_column|`` (all points, all
        components) and the root-mean-square and maximum per-point relative error ``|dF| / |F|`` of the vector.
        """
        points = np.asarray(points, dtype=np.float64)
        ref = column_field.flux_density(points)
        got = self.flux_density(points)
        diff = got - ref
        per_point = np.linalg.norm(diff, axis=-1) / np.maximum(np.linalg.norm(ref, axis=-1), 1e-300)
        return {"points": int(len(points)), "max_abs_over_max_ref": float(np.abs(diff).max() / np.abs(ref).max()),
                "rms_point_relative": float(np.sqrt(np.mean(per_point ** 2))), "max_point_relative": float(per_point.max())}


@jax.tree_util.register_pytree_node_class
class JaxEtaFilterTable:
    """The JAX twin of :class:`EtaFilterTable` (a PyTree: the B-spline coefficients are the only leaf).

    ``contravariant(q, metric)`` / ``cartesian`` / ``flux_density`` are differentiable in ``q`` and jittable; ``metric``
    is the JAX metric evaluation of ``q`` (``.J`` signed Jacobian, ``.jacobian_matrix``).  This is the ``logical_field``
    of :func:`drbx.geometry.curvature_autodiff.covariant_over_B_parts`.
    """

    def __init__(self, coefficients: Any, max_harmonic: int, nfp: int, nu: int, ntheta: int, u_max: float) -> None:
        self.coefficients = coefficients if isinstance(coefficients, jax.Array) else jnp.asarray(coefficients, dtype=jnp.float64)
        self.max_harmonic = int(max_harmonic)
        self.nfp = int(nfp)
        self.nu = int(nu)
        self.ntheta = int(ntheta)
        self.u_max = float(u_max)

    def tree_flatten(self):
        return (self.coefficients,), (self.max_harmonic, self.nfp, self.nu, self.ntheta, self.u_max)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], *aux)

    def flux_density(self, q: Any) -> jax.Array:
        q = jnp.asarray(q, dtype=jnp.float64)
        M = self.max_harmonic
        xu = q[..., 0] * ((self.nu - 1) / self.u_max)
        iu = jnp.clip(jnp.floor(xu).astype(jnp.int64), 0, self.nu - 2)
        tu = xu - iu
        xt = jnp.mod(q[..., 1], 2.0 * jnp.pi) * (self.ntheta / (2.0 * jnp.pi))
        jt = jnp.floor(xt).astype(jnp.int64)
        tt = xt - jt
        a = jnp.arange(4)
        rows = iu[..., None] + a                                            # (..., 4)
        cols = (jt[..., None] + a - 1) % self.ntheta                         # (..., 4)
        block = self.coefficients[rows[..., :, None], cols[..., None, :]]    # (..., 4, 4, ncomp)
        coef = jnp.einsum("...a,...b,...abc->...c", _basis(tu, jnp), _basis(tt, jnp), block)
        coef = coef.reshape(coef.shape[:-1] + (2 * M + 1, 3))
        arg = (q[..., 2] * self.nfp)[..., None] * jnp.arange(1, M + 1)
        return (coef[..., 0, :] + jnp.einsum("...m,...mc->...c", jnp.cos(arg), coef[..., 1:M + 1, :])
                + jnp.einsum("...m,...mc->...c", jnp.sin(arg), coef[..., M + 1:, :]))

    def contravariant(self, q: Any, metric: Any) -> jax.Array:
        """``B~^i = F~^i / J`` at ``q (..., 3)`` for the metric evaluation ``metric`` of ``q``."""
        return self.flux_density(q) / metric.J[..., None]

    def cartesian(self, q: Any, metric: Any) -> jax.Array:
        return jnp.einsum("...ij,...j->...i", metric.jacobian_matrix, self.contravariant(q, metric))
