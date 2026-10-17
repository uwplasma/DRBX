r"""Local tensor-product B-spline magnetic scalar potential (JAX, float64).

Inside the vessel the vacuum field is a gradient, ``B = grad(chi)``, with

.. math::

    \chi = G\,(\phi-\phi_0) + \widetilde\Phi(R,\phi,Z), \qquad
    \eta = (\phi-\phi_0) + \widetilde\Phi/G,

``Phi~`` periodic over one field period and no poloidal secular term
(``I = 0``).  ``Phi~`` is a tensor product of uniform B-splines of odd degree
``p`` (quintic by default)::

    Phi~ = sum_{i,k,j} c[i, k, j] N_i(R) P_k(phi) N_j(Z)

``R`` and ``Z`` are non-periodic with ``n_cells + p`` functions per axis (cell
``j`` uses functions ``j..j+p``); ``phi`` is periodic with ``nphi`` cells over
one period.  Because the basis is local, every query gathers one
``(p+1)^3`` coefficient block.

Two parts live here:

* :class:`JaxSplinePotential` -- the immutable PyTree evaluator (values, first
  derivatives, the fixed-iteration Newton solve of ``eta_hat(R, phi, Z) = eta``
  for ``phi`` with implicit differentiation through :func:`jax.lax.custom_root`),
  usable inside ``jit``/``vmap``/nested ``jacfwd``;
  :class:`SplineScalarPotentialEvaluator` is its NumPy-facing
  :class:`~drbx.geometry.ScalarPotential_evaluator.ScalarPotentialEvaluator`.
* :func:`spline_scalar_potential_evaluator_from_bfield` -- the least-squares fit
  of ``grad(chi)`` to a magnetic field on Gauss points inside the vessel plus a
  Laplacian penalty on interior cells, solved matrix-free by JAX LSMR on a
  sum-factorised operator, warm-started from a coarse fit by exact dyadic
  B-spline prolongation.

The fit's knot box is fixed by construction (:func:`spline_knot_box`): ``Z`` is
centred on 0 with a half-width snapped up to ``Z_HALF_SNAP``, ``R_lo`` does not
depend on the cell width, and the ``phi`` knots start at ``phi0 = 0``.  With
this box the stellarator symmetry ``(R, phi, Z) -> (R, -phi, -Z)`` maps knots,
Gauss points and the vessel mask onto themselves, so the fit is
stellarator-symmetric to round-off, and coarse and fine boxes nest, which the
exact prolongation needs.  The fit refuses any other box.
"""

from __future__ import annotations

import json
import math
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .ScalarPotential_evaluator import ScalarPotentialEvaluator

SPLINE_POTENTIAL_REPRESENTATION = "uniform_bspline_tensor"
SPLINE_POTENTIAL_PAYLOAD_VERSION = 1
#: m; the knot-box Z half-width is a multiple of this, so boxes for h = 2 cm, 1 cm, 5 mm, ... share Z_lo
Z_HALF_SNAP = 0.01
#: fixed Newton iterations of the phi solve (quadratic convergence from a start within a fraction of a cell)
DEFAULT_PHI_NEWTON_ITERATIONS = 8
#: block sizes of the NumPy-facing evaluation (queries are padded to the next bucket, then chunked)
_EVAL_BUCKETS = (256, 2048, 16384)
_BOX_TOLERANCE_CELLS = 1.0e-9
_B_CHUNK = 100_000


# ---------------------------------------------------------------------------------------------------------------------
# Cardinal B-splines
# ---------------------------------------------------------------------------------------------------------------------
def _cardinal_table(degree: int, t):
    """``(n, degree+1)``: ``N_d(t + k)``, ``k = 0..d``, for the cardinal B-spline of degree ``d`` (de Boor recursion)."""

    n = t.shape[0]
    values = jnp.ones((n, 1), dtype=t.dtype)
    for q in range(1, degree + 1):
        k = jnp.arange(q + 1, dtype=t.dtype)
        zero = jnp.zeros((n, 1), dtype=t.dtype)
        lower = jnp.concatenate([values, zero], axis=1)
        upper = jnp.concatenate([zero, values], axis=1)
        values = ((t[:, None] + k) * lower + (q + 1 - t[:, None] - k) * upper) / q
    return values


def _shift_columns(values, lag: int, width: int):
    n, k = values.shape
    left = jnp.zeros((n, lag), dtype=values.dtype)
    right = jnp.zeros((n, width - k - lag), dtype=values.dtype)
    return jnp.concatenate([left, values, right], axis=1)


def cardinal_bspline_basis(degree: int, t, *, derivatives: int = 1):
    """Values and ``t``-derivatives of the ``degree + 1`` nonzero uniform B-splines on one cell.

    ``t`` (shape ``(n,)``) is the local coordinate in ``[0, 1]``.  Column ``m`` is function ``j + m`` in cell ``j``
    (the same in every cell on uniform knots).  Returns ``derivatives + 1`` arrays of shape ``(n, degree + 1)``
    (``derivatives`` in 0, 1, 2).  The tables are polynomials in ``t``, so autodiff through them is exact.
    """

    degree = int(degree)
    if degree < 2:
        raise ValueError("degree must be at least 2")
    if derivatives not in (0, 1, 2):
        raise ValueError("derivatives must be 0, 1 or 2")
    t = jnp.asarray(t)
    width = degree + 1
    out = [_cardinal_table(degree, t)[:, ::-1]]
    if derivatives >= 1:
        lower = _cardinal_table(degree - 1, t)
        out.append((_shift_columns(lower, 0, width) - _shift_columns(lower, 1, width))[:, ::-1])
    if derivatives >= 2:
        lower2 = _cardinal_table(degree - 2, t)
        out.append(
            (_shift_columns(lower2, 0, width) - 2.0 * _shift_columns(lower2, 1, width)
             + _shift_columns(lower2, 2, width))[:, ::-1]
        )
    return tuple(out)


# ---------------------------------------------------------------------------------------------------------------------
# JAX evaluator (PyTree)
# ---------------------------------------------------------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
class JaxSplinePotential:
    """Immutable PyTree of a uniform tensor-product B-spline ``Phi~`` and its ``eta``.

    Leaves: ``coefficients`` ``(nR + p, nphi, nZ + p)``, ``G`` and the box scalars ``R_lo, h_R, Z_lo, h_Z, phi0,
    period``.  Static metadata: ``degree, nR, nZ, nphi``.  All methods are pointwise over the leading point axis (no
    reductions over points), so ``vmap``/``jit``/nested ``jacfwd`` apply; derivatives with respect to the
    coefficients and ``G`` are exact as well.  Queries outside the knot box are evaluated with the clamped edge cell
    (polynomial continuation); the NumPy-facing :class:`SplineScalarPotentialEvaluator` rejects them instead.
    """

    __slots__ = ("coefficients", "G", "R_lo", "h_R", "Z_lo", "h_Z", "phi0", "period", "degree", "nR", "nZ", "nphi")

    def __init__(self, coefficients, G, R_lo, h_R, Z_lo, h_Z, phi0, period, *, degree: int, nR: int, nZ: int,
                 nphi: int) -> None:
        self.coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
        for name, value in (("G", G), ("R_lo", R_lo), ("h_R", h_R), ("Z_lo", Z_lo), ("h_Z", h_Z), ("phi0", phi0),
                            ("period", period)):
            setattr(self, name, jnp.asarray(value, dtype=jnp.float64))
        self.degree, self.nR, self.nZ, self.nphi = int(degree), int(nR), int(nZ), int(nphi)
        if self.coefficients.shape != (self.nR + self.degree, self.nphi, self.nZ + self.degree):
            raise ValueError("coefficients must have shape (nR + p, nphi, nZ + p)")

    def tree_flatten(self):
        children = (self.coefficients, self.G, self.R_lo, self.h_R, self.Z_lo, self.h_Z, self.phi0, self.period)
        return children, (self.degree, self.nR, self.nZ, self.nphi)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Bypass the numeric constructor: lowering may pass abstract placeholders.
        instance = object.__new__(cls)
        for name, value in zip(("coefficients", "G", "R_lo", "h_R", "Z_lo", "h_Z", "phi0", "period"), children):
            object.__setattr__(instance, name, value)
        for name, value in zip(("degree", "nR", "nZ", "nphi"), aux_data):
            object.__setattr__(instance, name, value)
        return instance

    # -- 1-D bases -------------------------------------------------------------------------------------------------
    def _axis(self, x, lower, width, ncell: int, derivatives: int):
        u = (x - lower) / width
        cell = jnp.clip(jnp.floor(u), 0, ncell - 1).astype(jnp.int32)
        tables = cardinal_bspline_basis(self.degree, u - cell, derivatives=derivatives)
        return (cell,) + tuple(table / width ** k for k, table in enumerate(tables))

    def _phi_axis(self, phi, derivatives: int):
        h_phi = self.period / self.nphi
        u = jnp.mod(phi - self.phi0, self.period) / h_phi
        cell = jnp.clip(jnp.floor(u), 0, self.nphi - 1).astype(jnp.int32)
        tables = cardinal_bspline_basis(self.degree, u - cell, derivatives=derivatives)
        return (cell,) + tuple(table / h_phi ** k for k, table in enumerate(tables))

    def _blocks(self, iR, iP, iZ):
        w = self.degree + 1
        padded = jnp.concatenate([self.coefficients, self.coefficients[:, : self.degree]], axis=1)
        return jax.vmap(lambda a, b, c: lax.dynamic_slice(padded, (a, b, c), (w, w, w)))(iR, iP, iZ)

    def _flat(self, R, phi, Z):
        R, phi, Z = jnp.broadcast_arrays(jnp.asarray(R, jnp.float64), jnp.asarray(phi, jnp.float64),
                                         jnp.asarray(Z, jnp.float64))
        return R.shape, R.reshape(-1), phi.reshape(-1), Z.reshape(-1)

    def terms(self, R, phi, Z):
        """``(Phi~, dPhi~/dR, dPhi~/dphi, dPhi~/dZ)`` at cylindrical points (arrays broadcast together)."""

        shape, R, phi, Z = self._flat(R, phi, Z)
        iR, vR, dR = self._axis(R, self.R_lo, self.h_R, self.nR, 1)
        iZ, vZ, dZ = self._axis(Z, self.Z_lo, self.h_Z, self.nZ, 1)
        iP, vP, dP = self._phi_axis(phi, 1)
        block = self._blocks(iR, iP, iZ)
        a0 = jnp.sum(block * vZ[:, None, None, :], axis=-1)
        a1 = jnp.sum(block * dZ[:, None, None, :], axis=-1)
        b00 = jnp.sum(a0 * vP[:, None, :], axis=-1)
        b10 = jnp.sum(a0 * dP[:, None, :], axis=-1)
        b01 = jnp.sum(a1 * vP[:, None, :], axis=-1)
        return tuple(x.reshape(shape) for x in (jnp.sum(b00 * vR, -1), jnp.sum(b00 * dR, -1), jnp.sum(b10 * vR, -1),
                                                 jnp.sum(b01 * vR, -1)))

    def value(self, R, phi, Z):
        """``Phi~`` only."""

        shape, R, phi, Z = self._flat(R, phi, Z)
        iR, vR = self._axis(R, self.R_lo, self.h_R, self.nR, 0)
        iZ, vZ = self._axis(Z, self.Z_lo, self.h_Z, self.nZ, 0)
        iP, vP = self._phi_axis(phi, 0)
        block = self._blocks(iR, iP, iZ)
        a0 = jnp.sum(block * vZ[:, None, None, :], axis=-1)
        return jnp.sum(jnp.sum(a0 * vP[:, None, :], axis=-1) * vR, -1).reshape(shape)

    def _value_and_dphi(self, R, phi, Z):
        shape, R, phi, Z = self._flat(R, phi, Z)
        iR, vR = self._axis(R, self.R_lo, self.h_R, self.nR, 0)
        iZ, vZ = self._axis(Z, self.Z_lo, self.h_Z, self.nZ, 0)
        iP, vP, dP = self._phi_axis(phi, 1)
        block = self._blocks(iR, iP, iZ)
        a0 = jnp.sum(jnp.sum(block * vZ[:, None, None, :], axis=-1) * vR[:, :, None], axis=1)
        return jnp.sum(a0 * vP, -1).reshape(shape), jnp.sum(a0 * dP, -1).reshape(shape)

    def eta(self, R, phi, Z):
        """Unwrapped ``eta_hat = phi - phi0 + Phi~ / G`` (``phi`` is the query angle itself)."""

        phi = jnp.asarray(phi, jnp.float64)
        return phi - self.phi0 + self.value(R, phi, Z) / self.G

    def eta_and_partials(self, R, phi, Z):
        """``(eta_hat, d eta/dR, d eta/dphi, d eta/dZ)`` (coordinate partials, ``d/dphi`` not divided by ``R``)."""

        phi = jnp.asarray(phi, jnp.float64)
        value, dR, dphi, dZ = self.terms(R, phi, Z)
        return phi - self.phi0 + value / self.G, dR / self.G, 1.0 + dphi / self.G, dZ / self.G

    def magnetic_field_cylindrical(self, R, phi, Z):
        """``grad(chi)`` physical cylindrical components ``(B_R, B_phi, B_Z)``, each shaped like the inputs."""

        _, dR, dphi, dZ = self.terms(R, phi, Z)
        return dR, (self.G + dphi) / jnp.asarray(R, jnp.float64), dZ

    def solve_phi(self, R, Z, eta, phi_start, *, iterations: int = DEFAULT_PHI_NEWTON_ITERATIONS):
        """``phi`` with ``eta_hat(R, phi, Z) = eta`` by ``iterations`` Newton steps from ``phi_start``.

        The iteration count is fixed (no data-dependent control flow) and the residual is not wrapped: the solution
        is the root near ``phi_start``.  Derivatives with respect to ``R``, ``Z``, ``eta`` and the spline leaves come
        from the implicit-function theorem through :func:`jax.lax.custom_root`
        (``dphi = (deta - eta_R dR - eta_Z dZ - d_c eta . dc) / eta_phi``), so they are exact for converged roots
        and compose with nested forward-mode differentiation; ``phi_start`` receives no derivative.
        """

        R, Z, eta, phi_start = jnp.broadcast_arrays(*(jnp.asarray(a, jnp.float64) for a in (R, Z, eta, phi_start)))
        iterations = int(iterations)

        def residual(phi):
            return self.eta(R, phi, Z) - eta

        def solve(_residual, start):
            def body(_, phi):
                value, dphi = self._value_and_dphi(R, phi, Z)
                return phi - (phi - self.phi0 + value / self.G - eta) / (1.0 + dphi / self.G)

            return lax.fori_loop(0, iterations, body, start)

        def tangent_solve(linear, rhs):
            # The residual is pointwise, so its linearization is diagonal.
            return rhs / linear(jnp.ones_like(rhs))

        return lax.custom_root(residual, lax.stop_gradient(phi_start), solve, tangent_solve)


# ---------------------------------------------------------------------------------------------------------------------
# Jitted kernels of the NumPy-facing evaluator
# ---------------------------------------------------------------------------------------------------------------------
@jax.jit
def _terms_kernel(spline, R, phi, Z):
    return jnp.stack(spline.terms(R, phi, Z))


@jax.jit
def _value_kernel(spline, R, phi, Z):
    return spline.value(R, phi, Z)


_SOLVE_KERNELS: dict[int, Callable] = {}


def _solve_kernel(iterations: int) -> Callable:
    """Jitted ``(spline, R, Z, eta, start) -> (phi, [residual, eta_R, eta_phi, eta_Z])`` for one iteration count."""

    kernel = _SOLVE_KERNELS.get(iterations)
    if kernel is None:
        def kernel(spline, R, Z, eta, start):
            phi = spline.solve_phi(R, Z, eta, start, iterations=iterations)
            value, dR, dphi, dZ = spline.eta_and_partials(R, phi, Z)
            return phi, jnp.stack((value - eta, dR, dphi, dZ))

        kernel = _SOLVE_KERNELS[iterations] = jax.jit(kernel)
    return kernel


def _bucket(n: int) -> int:
    for size in _EVAL_BUCKETS:
        if n <= size:
            return size
    return _EVAL_BUCKETS[-1]


def _blocked(kernel: Callable, arrays: tuple[np.ndarray, ...], n_out: tuple[tuple[int, ...], ...]) -> tuple:
    """Run ``kernel(*arrays_block)`` over fixed padded blocks; ``n_out`` gives each output's leading shape."""

    n = arrays[0].size
    outputs = [np.empty(lead + (n,)) for lead in n_out]
    if n == 0:
        return tuple(outputs)
    block = _bucket(n)
    for start in range(0, n, block):
        count = min(block, n - start)
        part = [a[start:start + count] for a in arrays]
        if count < block:
            part = [np.concatenate([a, np.repeat(a[-1:], block - count)]) for a in part]
        result = kernel(*(jnp.asarray(a) for a in part))
        if not isinstance(result, tuple):
            result = (result,)
        for out, value in zip(outputs, result):
            out[..., start:start + count] = np.asarray(value)[..., :count]
    return tuple(outputs)


def _points(values: Any, name: str) -> tuple[np.ndarray, tuple[int, ...]]:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1:] != (3,):
        raise ValueError(f"{name} must have shape (..., 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points.reshape((-1, 3)), points.shape[:-1]


# ---------------------------------------------------------------------------------------------------------------------
# NumPy-facing evaluator
# ---------------------------------------------------------------------------------------------------------------------
class SplineScalarPotentialEvaluator(ScalarPotentialEvaluator):
    """Uniform tensor-product B-spline scalar potential with ``eta = phi - phi0 + Phi~ / G`` (``I = 0``).

    Parameters
    ----------
    coefficients:
        ``(nR + p, nphi, nZ + p)`` B-spline coefficients of ``Phi~`` (``phi`` periodic over one field period).
    G:
        Secular toroidal coefficient (``B_phi = (G + dPhi~/dphi) / R``).
    R_lo, h_R, Z_lo, h_Z:
        Lower knot-box edges and cell widths in ``R`` and ``Z``; ``nR`` and ``nZ`` follow from the coefficient shape.
    nfp:
        Field periods; the ``phi`` knots span ``period = 2 pi / nfp`` from ``phi0``.
    degree:
        Odd B-spline degree ``p`` (``>= 3``).

    Evaluation runs the jitted :class:`JaxSplinePotential` kernels on fixed padded blocks and returns NumPy arrays.
    Queries outside the knot box raise ``ValueError`` (``phi`` is periodic).  ``evaluate_*(..., wrapped=False)``
    returns the unwrapped, quasi-periodic ``eta`` (``eta(phi + period) = eta(phi) + period``); ``wrapped=True``
    returns ``eta mod period``.  Cartesian queries use the principal branch ``phi = atan2(y, x)``.
    """

    def __init__(self, coefficients: Any, *, G: float, R_lo: float, h_R: float, Z_lo: float, h_Z: float, nfp: int,
                 degree: int = 5, phi0: float = 0.0, diagnostics: Mapping[str, Any] | None = None) -> None:
        coefficients = np.asarray(coefficients, dtype=np.float64)
        degree = int(degree)
        if degree < 3 or degree % 2 != 1:
            raise ValueError("degree must be an odd integer >= 3")
        if coefficients.ndim != 3:
            raise ValueError("coefficients must have shape (nR + p, nphi, nZ + p)")
        nR, nphi, nZ = coefficients.shape[0] - degree, coefficients.shape[1], coefficients.shape[2] - degree
        if nR < 1 or nZ < 1 or nphi < degree + 1:
            raise ValueError("coefficients need at least one R/Z cell and nphi >= degree + 1")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("coefficients must be finite")
        if isinstance(nfp, (bool, np.bool_)) or int(nfp) != nfp or int(nfp) < 1:
            raise ValueError("nfp must be a positive integer")
        for name, value in (("G", G), ("R_lo", R_lo), ("h_R", h_R), ("Z_lo", Z_lo), ("h_Z", h_Z), ("phi0", phi0)):
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(h_R) <= 0.0 or float(h_Z) <= 0.0:
            raise ValueError("h_R and h_Z must be positive")
        if float(R_lo) <= 0.0:
            raise ValueError("the knot box must have R_lo > 0")
        if abs(float(G)) <= np.finfo(np.float64).eps:
            raise ValueError("eta is undefined for G = 0")
        self._coefficients = coefficients.copy()
        self._G = float(G)
        self._R_lo, self._h_R, self._Z_lo, self._h_Z = float(R_lo), float(h_R), float(Z_lo), float(h_Z)
        self._phi0 = float(phi0)
        self._nfp = int(nfp)
        self._period = 2.0 * np.pi / self._nfp
        self._degree, self._nR, self._nZ, self._nphi = degree, nR, nZ, nphi
        self._R_hi = self._R_lo + nR * self._h_R
        self._Z_hi = self._Z_lo + nZ * self._h_Z
        self._diagnostics = MappingProxyType(dict(diagnostics) if diagnostics is not None else {})
        self._jax = JaxSplinePotential(self._coefficients, self._G, self._R_lo, self._h_R, self._Z_lo, self._h_Z,
                                       self._phi0, self._period, degree=degree, nR=nR, nZ=nZ, nphi=nphi)

    # -- metadata --------------------------------------------------------------------------------------------------
    @property
    def R(self) -> np.ndarray:
        """Knot positions in ``R`` (cell edges of the knot box)."""
        return self._R_lo + self._h_R * np.arange(self._nR + 1, dtype=np.float64)

    @property
    def phi(self) -> np.ndarray:
        """Endpoint-exclusive ``phi`` knots over one field period."""
        return self._phi0 + (self._period / self._nphi) * np.arange(self._nphi, dtype=np.float64)

    @property
    def Z(self) -> np.ndarray:
        """Knot positions in ``Z``."""
        return self._Z_lo + self._h_Z * np.arange(self._nZ + 1, dtype=np.float64)

    @property
    def nfp(self) -> int:
        return self._nfp

    @property
    def period(self) -> float:
        return self._period

    @property
    def phi0(self) -> float:
        return self._phi0

    @property
    def G(self) -> float:
        return self._G

    @property
    def I(self) -> float:  # noqa: E743 - interface name
        return 0.0

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def cell_counts(self) -> tuple[int, int, int]:
        """``(nR, nphi, nZ)`` knot cells."""
        return self._nR, self._nphi, self._nZ

    @property
    def knot_box(self) -> dict[str, float]:
        return {"R_lo": self._R_lo, "R_hi": self._R_hi, "h_R": self._h_R, "Z_lo": self._Z_lo, "Z_hi": self._Z_hi,
                "h_Z": self._h_Z, "phi0": self._phi0, "h_phi": self._period / self._nphi}

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients.copy()

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics

    @property
    def jax(self) -> JaxSplinePotential:
        """The immutable JAX PyTree of this potential."""
        return self._jax

    # -- checks ----------------------------------------------------------------------------------------------------
    def _check_box(self, R: np.ndarray, Z: np.ndarray) -> None:
        if R.size == 0:
            return
        if np.any(R <= 0.0):
            raise ValueError("cylindrical query points must have R > 0")
        tol_R, tol_Z = _BOX_TOLERANCE_CELLS * self._h_R, _BOX_TOLERANCE_CELLS * self._h_Z
        if R.min() < self._R_lo - tol_R or R.max() > self._R_hi + tol_R:
            raise ValueError(f"R query lies outside the spline knot box [{self._R_lo:.6g}, {self._R_hi:.6g}]")
        if Z.min() < self._Z_lo - tol_Z or Z.max() > self._Z_hi + tol_Z:
            raise ValueError(f"Z query lies outside the spline knot box [{self._Z_lo:.6g}, {self._Z_hi:.6g}]")

    def terms_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        """``(4, ...)``: ``Phi~`` and its ``R``, ``phi``, ``Z`` coordinate derivatives at cylindrical points."""

        points, shape = _points(points_rphiz, "points_rphiz")
        self._check_box(points[:, 0], points[:, 2])
        (out,) = _blocked(lambda R, p, Z: _terms_kernel(self._jax, R, p, Z),
                          (points[:, 0].copy(), points[:, 1].copy(), points[:, 2].copy()), ((4,),))
        return out.reshape((4,) + shape)

    def _phi_tilde(self, points: np.ndarray) -> np.ndarray:
        self._check_box(points[:, 0], points[:, 2])
        (out,) = _blocked(lambda R, p, Z: _value_kernel(self._jax, R, p, Z),
                          (points[:, 0].copy(), points[:, 1].copy(), points[:, 2].copy()), ((),))
        return out

    def _terms(self, points: np.ndarray) -> np.ndarray:
        self._check_box(points[:, 0], points[:, 2])
        (out,) = _blocked(lambda R, p, Z: _terms_kernel(self._jax, R, p, Z),
                          (points[:, 0].copy(), points[:, 1].copy(), points[:, 2].copy()), ((4,),))
        return out

    # -- eta -------------------------------------------------------------------------------------------------------
    def evaluate_cylindrical(self, points_rphiz: Any, *, wrapped: bool = False) -> np.ndarray:
        points, shape = _points(points_rphiz, "points_rphiz")
        eta = points[:, 1] - self._phi0 + self._phi_tilde(points) / self._G
        if wrapped:
            eta = np.mod(eta, self._period)
        return eta.reshape(shape)

    def gradient_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        points, shape = _points(points_rphiz, "points_rphiz")
        _, dR, dphi, dZ = self._terms(points)
        gradient = np.stack((dR / self._G, (1.0 + dphi / self._G) / points[:, 0], dZ / self._G), axis=-1)
        return gradient.reshape(shape + (3,))

    def evaluate_cartesian(self, points_xyz: Any, *, wrapped: bool = False) -> np.ndarray:
        points, shape = _points(points_xyz, "points_xyz")
        return self.evaluate_cylindrical(_cartesian_to_cylindrical(points), wrapped=wrapped).reshape(shape)

    def gradient_cartesian(self, points_xyz: Any) -> np.ndarray:
        return self.evaluate_and_gradient_cartesian(points_xyz)[1]

    def evaluate_and_gradient_cartesian(self, points_xyz: Any, *, wrapped: bool = False
                                        ) -> tuple[np.ndarray, np.ndarray]:
        """``eta`` and Cartesian ``grad(eta)`` with one shared spline evaluation."""

        points, shape = _points(points_xyz, "points_xyz")
        cylindrical = _cartesian_to_cylindrical(points)
        value, dR, dphi, dZ = self._terms(cylindrical)
        eta = cylindrical[:, 1] - self._phi0 + value / self._G
        if wrapped:
            eta = np.mod(eta, self._period)
        gR, gP, gZ = dR / self._G, (1.0 + dphi / self._G) / cylindrical[:, 0], dZ / self._G
        cosine, sine = np.cos(cylindrical[:, 1]), np.sin(cylindrical[:, 1])
        gradient = np.stack((gR * cosine - gP * sine, gR * sine + gP * cosine, gZ), axis=-1)
        return eta.reshape(shape), gradient.reshape(shape + (3,))

    # -- magnetic potential ------------------------------------------------------------------------------------------
    def evaluate_magnetic_potential_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        """``chi = G (phi - phi0) + Phi~`` (the query ``phi`` branch)."""

        points, shape = _points(points_rphiz, "points_rphiz")
        return (self._G * (points[:, 1] - self._phi0) + self._phi_tilde(points)).reshape(shape)

    def magnetic_field_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        points, shape = _points(points_rphiz, "points_rphiz")
        _, dR, dphi, dZ = self._terms(points)
        return np.stack((dR, (self._G + dphi) / points[:, 0], dZ), axis=-1).reshape(shape + (3,))

    def evaluate_magnetic_potential_cartesian(self, points_xyz: Any) -> np.ndarray:
        points, shape = _points(points_xyz, "points_xyz")
        return self.evaluate_magnetic_potential_cylindrical(_cartesian_to_cylindrical(points)).reshape(shape)

    def magnetic_field_cartesian(self, points_xyz: Any) -> np.ndarray:
        points, shape = _points(points_xyz, "points_xyz")
        cylindrical = _cartesian_to_cylindrical(points)
        field = self.magnetic_field_cylindrical(cylindrical)
        cosine, sine = np.cos(cylindrical[:, 1]), np.sin(cylindrical[:, 1])
        out = np.stack((field[:, 0] * cosine - field[:, 1] * sine, field[:, 0] * sine + field[:, 1] * cosine,
                        field[:, 2]), axis=-1)
        return out.reshape(shape + (3,))

    # -- phi solve -------------------------------------------------------------------------------------------------
    def solve_phi(self, R: Any, Z: Any, eta: Any, phi_start: Any | None = None, *,
                  iterations: int = DEFAULT_PHI_NEWTON_ITERATIONS, return_residual: bool = False):
        """``phi`` with ``eta_hat(R, phi, Z) = eta`` (unwrapped) at points ``(R, Z)``, by fixed-iteration Newton.

        ``phi_start`` defaults to ``eta + phi0`` (exact when ``Phi~ = 0``); the root near the start is returned.
        With ``return_residual=True`` also returns the unwrapped residual ``eta_hat - eta`` at the solution.
        Raises when ``d eta/d phi <= 0`` at a solution (the level set is not a graph over ``(R, Z)`` there).
        """

        R, Z, eta = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (R, Z, eta)))
        shape = R.shape
        start = (eta + self._phi0) if phi_start is None else np.broadcast_to(
            np.asarray(phi_start, dtype=np.float64), shape)
        flat = [np.ascontiguousarray(a).reshape(-1) for a in (R, Z, eta, start)]
        if not all(np.all(np.isfinite(a)) for a in flat):
            raise ValueError("solve_phi inputs must be finite")
        self._check_box(flat[0], flat[1])
        kernel = _solve_kernel(int(iterations))
        phi, extra = _blocked(lambda a, b, c, d: kernel(self._jax, a, b, c, d), tuple(flat), ((), (4,)))
        if np.any(~(extra[2] > 0.0)):
            raise ValueError("eta_hat is not increasing in phi at a solution: the level set is not a graph over (R, Z)")
        phi = phi.reshape(shape)
        return (phi, extra[0].reshape(shape)) if return_residual else phi

    # -- payload ---------------------------------------------------------------------------------------------------
    def to_payload(self, *, prefix: str = "") -> dict[str, np.ndarray]:
        """Numeric arrays (no pickled objects) that :meth:`from_payload` restores exactly."""

        key = lambda name: f"{prefix}{name}"  # noqa: E731
        diagnostics = json.dumps(_jsonable(dict(self._diagnostics)), sort_keys=True)
        return {
            key("representation"): np.asarray(SPLINE_POTENTIAL_REPRESENTATION),
            key("payload_version"): np.asarray(SPLINE_POTENTIAL_PAYLOAD_VERSION, dtype=np.int64),
            key("coefficients"): self._coefficients.copy(),
            key("G"): np.asarray(self._G, dtype=np.float64),
            key("degree"): np.asarray(self._degree, dtype=np.int64),
            key("nfp"): np.asarray(self._nfp, dtype=np.int64),
            key("phi0"): np.asarray(self._phi0, dtype=np.float64),
            key("R_lo"): np.asarray(self._R_lo, dtype=np.float64),
            key("h_R"): np.asarray(self._h_R, dtype=np.float64),
            key("Z_lo"): np.asarray(self._Z_lo, dtype=np.float64),
            key("h_Z"): np.asarray(self._h_Z, dtype=np.float64),
            key("diagnostics_json"): np.asarray(diagnostics),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, prefix: str = "") -> "SplineScalarPotentialEvaluator":
        def get(name: str) -> np.ndarray:
            try:
                return np.asarray(payload[f"{prefix}{name}"])
            except KeyError as error:
                raise ValueError(f"spline potential payload is missing {prefix}{name!r}") from error

        if str(get("representation").item()) != SPLINE_POTENTIAL_REPRESENTATION:
            raise ValueError("payload is not a uniform B-spline scalar potential")
        version = int(get("payload_version").item())
        if version != SPLINE_POTENTIAL_PAYLOAD_VERSION:
            raise ValueError(f"unsupported spline potential payload version {version}")
        return cls(get("coefficients"), G=float(get("G").item()), R_lo=float(get("R_lo").item()),
                   h_R=float(get("h_R").item()), Z_lo=float(get("Z_lo").item()), h_Z=float(get("h_Z").item()),
                   nfp=int(get("nfp").item()), degree=int(get("degree").item()), phi0=float(get("phi0").item()),
                   diagnostics=json.loads(str(get("diagnostics_json").item())))

    def save(self, path: Any) -> None:
        np.savez(path, **self.to_payload())

    @classmethod
    def load(cls, path: Any) -> "SplineScalarPotentialEvaluator":
        with np.load(path, allow_pickle=False) as payload:
            return cls.from_payload(payload)


def _cartesian_to_cylindrical(points: np.ndarray) -> np.ndarray:
    radius = np.hypot(points[:, 0], points[:, 1])
    if np.any(radius <= 0.0):
        raise ValueError("Cartesian query points must have R > 0")
    return np.column_stack((radius, np.arctan2(points[:, 1], points[:, 0]), points[:, 2]))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int, bool, np.bool_)):
        return value.item() if hasattr(value, "item") else value
    return value


# ---------------------------------------------------------------------------------------------------------------------
# The least-squares fit
# ---------------------------------------------------------------------------------------------------------------------
def wall_pseudo_radius(wall: Any, points_rphiz: Any, *, ntheta: int = 512) -> np.ndarray:
    """``rho = |p - c| / |wall(theta_p) - c|`` along the ray from the wall-section centroid ``c`` in each phi plane.

    ``rho < 1`` inside a star-shaped section; the fit's data mask is ``rho <= rho_data``.  Points are grouped by
    wrapped ``phi`` (one wall polygon per distinct plane).
    """

    p = np.asarray(points_rphiz, dtype=np.float64).reshape(-1, 3)
    out = np.empty(p.shape[0])
    wrapped = np.mod(p[:, 1] - wall.phi0, wall.period) + wall.phi0
    keys = np.round(wrapped, 12)
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
    for key in np.unique(keys):
        index = np.flatnonzero(keys == key)
        section = wall.cylindrical(float(key), theta)
        Rw, Zw = section[:, 0], section[:, 2]
        cR, cZ = Rw.mean(), Zw.mean()
        angle = np.arctan2(Zw - cZ, Rw - cR)
        radius = np.hypot(Zw - cZ, Rw - cR)
        order = np.argsort(angle)
        angle_sorted = np.r_[angle[order][-1] - 2.0 * np.pi, angle[order], angle[order][0] + 2.0 * np.pi]
        radius_sorted = np.r_[radius[order][-1], radius[order], radius[order][0]]
        query_angle = np.arctan2(p[index, 2] - cZ, p[index, 0] - cR)
        out[index] = np.hypot(p[index, 2] - cZ, p[index, 0] - cR) / np.interp(query_angle, angle_sorted, radius_sorted)
    return out.reshape(np.asarray(points_rphiz).shape[:-1])


def spline_knot_box(bfield: Any, wall: Any, h: float, pad: float, *, z_snap: float = Z_HALF_SNAP
                    ) -> tuple[float, float, int, int]:
    """``(R_lo, Z_lo, nR, nZ)`` of the fit's knot box for cell width ``h`` (the only box the fit accepts).

    ``R`` spans the vessel's ``R`` range padded by ``pad`` (clipped one table node inside the field table),
    ``R_lo`` independent of ``h``.  ``Z`` is centred on ``Z = 0`` with a half-width ``max|Z_wall| + pad`` snapped up
    to a multiple of ``z_snap``, so the knots are mirror symmetric and boxes for ``h``, ``h/2``, ... share ``Z_lo``
    whenever ``z_snap`` is a multiple of half the coarsest cell width (the fit uses ``max(Z_HALF_SNAP, h_coarse/2)``:
    1 cm for 2 cm / 1 cm levels).
    """

    h, pad, z_snap = float(h), float(pad), float(z_snap)
    if not (h > 0.0 and pad >= 0.0 and z_snap > 0.0):
        raise ValueError("h and z_snap must be positive and pad nonnegative")
    raw = np.asarray(wall.raw["RZ"], dtype=np.float64)
    table_R, table_Z = np.asarray(bfield.R, dtype=np.float64), np.asarray(bfield.Z, dtype=np.float64)
    R_lo = max(float(raw[..., 0].min()) - pad, float(table_R[1]))
    R_hi = min(float(raw[..., 0].max()) + pad, float(table_R[-2]))
    z_max = max(-float(raw[..., 1].min()), float(raw[..., 1].max())) + pad
    z_half = math.ceil(z_max / z_snap - 1e-9) * z_snap
    nR = int(math.ceil((R_hi - R_lo) / h - 1e-9))
    nZ = int(math.ceil(2.0 * z_half / h - 1e-9))
    Z_lo = -0.5 * nZ * h
    if Z_lo < float(table_Z[1]) or -Z_lo > float(table_Z[-2]):
        raise ValueError(f"the centred knot box Z = +-{-Z_lo:.4f} m leaves the magnetic-field table")
    if R_lo + nR * h > float(table_R[-1]):
        raise ValueError("the knot box leaves the magnetic-field table in R")
    return R_lo, Z_lo, nR, nZ


def _or_dilate(a: np.ndarray, p: int, axis: int, periodic: bool) -> np.ndarray:
    """Functions active from cells: ``out[i] = OR_{s=0..p} a[i-s]`` (non-periodic: length ``n + p``)."""

    n = a.shape[axis]
    if periodic:
        out = np.zeros_like(a)
        for s in range(p + 1):
            out |= np.roll(a, s, axis)
        return out
    shape = list(a.shape)
    shape[axis] = n + p
    out = np.zeros(shape, bool)
    for s in range(p + 1):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(s, s + n)
        out[tuple(sl)] |= a
    return out


def _and_touch(a: np.ndarray, p: int, axis: int, periodic: bool, n: int) -> np.ndarray:
    """Cells whose functions along this axis are all active: ``out[c] = AND_{s=0..p} a[c+s]``."""

    if periodic:
        out = np.ones_like(a)
        for s in range(p + 1):
            out &= np.roll(a, -s, axis)
        return out
    out = np.ones(tuple(n if ax == axis else a.shape[ax] for ax in range(a.ndim)), bool)
    for s in range(p + 1):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(s, s + n)
        out &= a[tuple(sl)]
    return out


def _gauss_01(order: int) -> tuple[np.ndarray, np.ndarray]:
    t, w = np.polynomial.legendre.leggauss(int(order))
    return 0.5 * (t + 1.0), 0.5 * w


def _banded_axis(local: np.ndarray, ncell: int):
    """Dense ``(ncell * g, ncell + p)``: row ``j*g + a`` holds ``local[a, m]`` in column ``j + m``."""

    g, width = local.shape
    rows = (np.arange(ncell)[:, None, None] * g + np.arange(g)[None, :, None]).repeat(width, axis=2)
    cols = (np.arange(ncell)[:, None, None] + np.arange(width)[None, None, :]).repeat(g, axis=1)
    values = jnp.broadcast_to(jnp.asarray(local)[None], (ncell, g, width))
    matrix = jnp.zeros((ncell * g, ncell + width - 1), dtype=jnp.float64)
    return matrix.at[rows.reshape(-1), cols.reshape(-1)].set(values.reshape(-1))


class _SplineFitSystem:
    """The weighted least-squares system of the spline fit (JAX operator, NumPy bookkeeping of the masks).

    Rows (all times ``sqrt(w)``, ``w = R * Gauss weight * cell widths``, divided by the field scale ``Bs``)::

        data (Gauss points with wall pseudo-radius <= rho_data):  Phi_R ; (Phi_phi + G)/R ; Phi_Z
        penalty (Gauss points of interior cells, all (p+1)^3 functions active):  sqrt(lam) h Lap(Phi~)

    Unknowns ``x = [c over active functions, G]``; a function is active when its support touches a data cell.  Rows
    are ordered globally ``[data R | data phi | data Z | penalty]``, each in C order of the
    ``(nR g, nphi g, nZ g)`` Gauss grid.  ``R`` and ``Z`` contractions are dense GEMMs with the square-root weights,
    ``1/R`` and ``1/Bs`` folded into the 1-D matrices; the periodic ``phi`` contraction is a banded sum of ``p + 1``
    shifted planes.
    """

    def __init__(self, wall: Any, *, R_lo: float, Z_lo: float, nR: int, nZ: int, h: float, p: int, nphi: int,
                 period: float, phi0: float, gauss: int, rho_data: float, lam: float) -> None:
        if abs(Z_lo + 0.5 * nZ * h) > 1e-12 or phi0 != 0.0:
            raise ValueError(
                "the spline fit requires the mirror-symmetric knot box of spline_knot_box (Z centred on 0, phi0 = 0); "
                f"got Z [{Z_lo}, {Z_lo + nZ * h}], phi0 {phi0}")
        if int(p) < 3 or int(p) % 2 != 1:
            raise ValueError("degree must be an odd integer >= 3")
        if int(nphi) < int(p) + 1:
            raise ValueError("nphi must be at least degree + 1")
        if int(gauss) < 1 or float(lam) < 0.0 or float(rho_data) <= 0.0:
            raise ValueError("gauss must be positive, lam nonnegative and rho_data positive")
        self.p, self.g, self.h, self.lam = int(p), int(gauss), float(h), float(lam)
        self.R_lo, self.Z_lo, self.nR, self.nZ, self.nphi = float(R_lo), float(Z_lo), int(nR), int(nZ), int(nphi)
        self.period, self.phi0, self.rho_data = float(period), float(phi0), float(rho_data)
        p, g = self.p, self.g
        self.hP = self.period / self.nphi
        t, wq = _gauss_01(g)
        self.Rg = (self.R_lo + self.h * (np.arange(self.nR)[:, None] + t)).ravel()
        self.Zg = (self.Z_lo + self.h * (np.arange(self.nZ)[:, None] + t)).ravel()
        self.Pg = (self.phi0 + self.hP * (np.arange(self.nphi)[:, None] + t)).ravel()
        self.nRg, self.nZg, self.nPg = self.nR * g, self.nZ * g, self.nphi * g
        self.nRf, self.nZf = self.nR + p, self.nZ + p
        self._sR = np.sqrt(self.Rg * np.tile(wq * self.h, self.nR))
        self._sP = np.sqrt(np.tile(wq * self.hP, self.nphi))
        self._sZ = np.sqrt(np.tile(wq * self.h, self.nZ))
        self._invR = 1.0 / self.Rg
        # 1-D basis tables at the Gauss points (JAX), (g, p + 1) each
        tables = [np.asarray(x) for x in cardinal_bspline_basis(p, jnp.asarray(t), derivatives=2)]
        self._locH = (tables[0], tables[1] / self.h, tables[2] / self.h ** 2)
        self._locP = (tables[0], tables[1] / self.hP, tables[2] / self.hP ** 2)
        self._ER = [_banded_axis(m, self.nR) for m in self._locH]
        self._EZ = [_banded_axis(m, self.nZ) for m in self._locH]
        # data mask on the Gauss grid, active functions, interior penalty cells
        dm = np.zeros((self.nRg, self.nPg, self.nZg), bool)
        RR, ZZ = np.meshgrid(self.Rg, self.Zg, indexing="ij")
        plane = np.empty((RR.size, 3))
        plane[:, 0], plane[:, 2] = RR.ravel(), ZZ.ravel()
        for k, angle in enumerate(self.Pg):
            plane[:, 1] = angle
            dm[:, k, :] = (wall_pseudo_radius(wall, plane) <= self.rho_data).reshape(self.nRg, self.nZg)
        del RR, ZZ, plane
        dcell = dm.reshape(self.nR, g, self.nphi, g, self.nZ, g).any(axis=(1, 3, 5))
        if not np.any(dcell):
            raise ValueError("the vessel mask selects no fit data point")
        active = _or_dilate(_or_dilate(_or_dilate(dcell, p, 0, False), p, 1, True), p, 2, False)
        pcell = _and_touch(_and_touch(_and_touch(active, p, 0, False, self.nR), p, 1, True, self.nphi), p, 2, False,
                           self.nZ)
        pm = np.repeat(np.repeat(np.repeat(pcell, g, 0), g, 1), g, 2)
        self.active = active
        self.act_idx = np.flatnonzero(active.ravel())
        self.n_active = int(self.act_idx.size)
        self.n = self.n_active + 1
        self.d_glob = np.flatnonzero(dm.ravel())
        self.p_glob = np.flatnonzero(pm.ravel())
        self.Nd, self.Np = int(self.d_glob.size), int(self.p_glob.size)
        self.m = 3 * self.Nd + self.Np
        self.n_data_cells, self.n_pen_cells = int(dcell.sum()), int(pcell.sum())
        self._dm, self._pm = dm, pm
        self.Bs = None
        self.b = None
        self.diag: dict[str, Any] = {}

    # -- scaling -----------------------------------------------------------------------------------------------------
    def set_scale(self, Bs: float) -> None:
        """Fold ``1/Bs``, the square-root weights and ``1/R`` into the 1-D matrices."""

        self.Bs = float(Bs)
        a = 1.0 / self.Bs
        pf = math.sqrt(self.lam) * self.h
        ER0, ER1, ER2 = self._ER
        EZ0, EZ1, EZ2 = self._EZ
        sR, sZ, invR = (jnp.asarray(v)[:, None] for v in (self._sR, self._sZ, self._invR))
        self.T1R = a * sR * ER1
        self.T2R = a * sR * invR * ER0
        self.T3R = a * sR * ER0
        self.PT1R = a * pf * sR * ER2
        self.PT2R = a * pf * sR * invR * ER1
        self.PT3R = a * pf * sR * invR ** 2 * ER0
        self.Z0, self.Z1, self.Z2 = sZ * EZ0, sZ * EZ1, pf * sZ * EZ2
        self.RS = jnp.concatenate([self.T1R, self.T2R, self.T3R, self.PT3R, self.PT1R + self.PT2R], axis=0)
        self.RST = self.RS.T
        t, wq = _gauss_01(self.g)
        sPa = np.sqrt(wq * self.hP)
        self.LP = [np.asarray(sPa[:, None] * m) for m in self._locP]
        weights = ((self._sR * self._invR * a)[:, None, None] * self._sP[None, :, None]
                   * self._sZ[None, None, :]).ravel()
        self.gcol = jnp.asarray(weights[self.d_glob])
        self._d_idx = jnp.asarray(self.d_glob.astype(np.int32))
        self._p_idx = jnp.asarray(self.p_glob.astype(np.int32))
        self._act = jnp.asarray(self.act_idx.astype(np.int32))
        self.matvec = jax.jit(self._matvec)
        self.rmatvec = jax.jit(self._rmatvec)

    # -- phi stage (banded, periodic) ---------------------------------------------------------------------------------
    def _phi_forward(self, k: int, Y):
        p, g, nphi = self.p, self.g, self.nphi
        L = self.LP[k]
        Yp = jnp.concatenate([Y, Y[:, :p]], axis=1)
        planes = []
        for a in range(g):
            acc = L[a, 0] * Yp[:, 0:nphi]
            for m in range(1, p + 1):
                acc = acc + L[a, m] * Yp[:, m:m + nphi]
            planes.append(acc)
        return jnp.stack(planes, axis=2).reshape(self.nRg, self.nPg, -1)

    def _phi_adjoint(self, k: int, Gf):
        p, g, nphi = self.p, self.g, self.nphi
        L = self.LP[k]
        G4 = Gf.reshape(self.nRg, nphi, g, -1)
        result = None
        for m in range(p + 1):
            T = L[0, m] * G4[:, :, 0]
            for a in range(1, g):
                T = T + L[a, m] * G4[:, :, a]
            T = jnp.roll(T, m, axis=1)
            result = T if result is None else result + T
        return result

    # -- operator ------------------------------------------------------------------------------------------------
    def _dense(self, x):
        flat = jnp.zeros(self.nRf * self.nphi * self.nZf, x.dtype).at[self._act].set(
            x[:-1], indices_are_sorted=True, unique_indices=True)
        return flat.reshape(self.nRf, self.nphi, self.nZf)

    def _matvec(self, x):
        nRg, nphi, nZg, nZf = self.nRg, self.nphi, self.nZg, self.nZf
        G = x[-1]
        C = self._dense(x)
        T = (self.RS @ C.reshape(self.nRf, nphi * nZf)).reshape(5, nRg * nphi, nZf)
        Ra, Rb, Rd, Rc, Rp = T[0], T[1], T[2], T[3], T[4]
        shape = (nRg, nphi, nZg)
        A = (Ra @ self.Z0.T).reshape(shape)
        B2 = (Rb @ self.Z0.T).reshape(shape)
        Cl = (Rc @ self.Z0.T).reshape(shape)
        Xd = (Rd @ self.Z1.T).reshape(shape)
        Bq = (Rp @ self.Z0.T + Rd @ self.Z2.T).reshape(shape)
        take = lambda arr, idx: jnp.take(arr.reshape(-1), idx, indices_are_sorted=True)  # noqa: E731
        y1 = take(self._phi_forward(0, A), self._d_idx)
        y2 = take(self._phi_forward(1, B2), self._d_idx) + G * self.gcol
        y3 = take(self._phi_forward(0, Xd), self._d_idx)
        y4 = take(self._phi_forward(0, Bq) + self._phi_forward(2, Cl), self._p_idx)
        return jnp.concatenate([y1, y2, y3, y4])

    def _rmatvec(self, y):
        nRg, nphi, nZg = self.nRg, self.nphi, self.nZg
        Nd = self.Nd
        full = nRg * self.nPg * nZg
        shape = (nRg, self.nPg, nZg)

        def scatter(idx, values):
            return jnp.zeros(full, y.dtype).at[idx].set(values, indices_are_sorted=True,
                                                         unique_indices=True).reshape(shape)

        y2 = y[Nd:2 * Nd]
        g1, g2, g3 = scatter(self._d_idx, y[:Nd]), scatter(self._d_idx, y2), scatter(self._d_idx, y[2 * Nd:3 * Nd])
        gp = scatter(self._p_idx, y[3 * Nd:])
        dG = jnp.dot(self.gcol, y2)
        gA, gB2, gD = self._phi_adjoint(0, g1), self._phi_adjoint(1, g2), self._phi_adjoint(0, g3)
        gBq, gCl = self._phi_adjoint(0, gp), self._phi_adjoint(2, gp)
        r2 = lambda arr: arr.reshape(nRg * nphi, nZg)  # noqa: E731
        Ra = r2(gA) @ self.Z0
        Rb = r2(gB2) @ self.Z0
        Rc = r2(gCl) @ self.Z0
        Rp = r2(gBq) @ self.Z0
        Rd = r2(gD) @ self.Z1 + r2(gBq) @ self.Z2
        stacked = jnp.stack([Ra, Rb, Rd, Rc, Rp]).reshape(5 * nRg, nphi * self.nZf)
        Ct = self.RST @ stacked
        return jnp.concatenate([Ct.reshape(-1)[self._act], dG[None]])

    # -- exact diag(A^T A) ----------------------------------------------------------------------------------------
    def _phi_dense(self):
        """Folded periodic phi matrices ``(nphi g, nphi)`` with square-root weights (columns of a row never collide
        because ``nphi >= p + 1``, so elementwise products commute with the fold)."""

        out = []
        for k in range(3):
            M = _banded_axis(self.LP[k], self.nphi)          # (nPg, nphi + p)
            out.append(M[:, :self.nphi].at[:, :self.p].add(M[:, self.nphi:]))
        return out

    def column_norms_squared(self):
        """``diag(A^T A)`` exactly: the squared rows of each separable term, plus the penalty cross terms."""

        P0, P1, P2 = self._phi_dense()
        T1R, T2R, T3R, PT1R, PT2R, PT3R = self.T1R, self.T2R, self.T3R, self.PT1R, self.PT2R, self.PT3R
        Z0, Z1, Z2 = self.Z0, self.Z1, self.Z2
        data = jnp.asarray(self._dm, dtype=jnp.float64)
        pen = jnp.asarray(self._pm, dtype=jnp.float64)

        @jax.jit
        def chain(W, MR, MP, MZ):
            X = jnp.einsum("abc,ck->abk", W, MZ)
            X = jnp.einsum("abk,bj->ajk", X, MP)
            return jnp.einsum("ajk,ai->ijk", X, MR)

        passes = [
            (data, T1R * T1R, P0 * P0, Z0 * Z0, 1.0), (data, T2R * T2R, P1 * P1, Z0 * Z0, 1.0),
            (data, T3R * T3R, P0 * P0, Z1 * Z1, 1.0),
            (pen, PT1R * PT1R, P0 * P0, Z0 * Z0, 1.0), (pen, PT2R * PT2R, P0 * P0, Z0 * Z0, 1.0),
            (pen, PT3R * PT3R, P2 * P2, Z0 * Z0, 1.0), (pen, T3R * T3R, P0 * P0, Z2 * Z2, 1.0),
            (pen, PT1R * PT2R, P0 * P0, Z0 * Z0, 2.0), (pen, PT1R * PT3R, P0 * P2, Z0 * Z0, 2.0),
            (pen, PT1R * T3R, P0 * P0, Z0 * Z2, 2.0), (pen, PT2R * PT3R, P0 * P2, Z0 * Z0, 2.0),
            (pen, PT2R * T3R, P0 * P0, Z0 * Z2, 2.0), (pen, PT3R * T3R, P2 * P0, Z0 * Z2, 2.0),
        ]
        total = None
        for W, MR, MP, MZ, factor in passes:
            term = factor * chain(W, MR, MP, MZ)
            total = term if total is None else total + term
        diag = total.reshape(-1)[self._act]
        return jnp.concatenate([diag, jnp.sum(self.gcol * self.gcol)[None]])

    # -- right-hand side -------------------------------------------------------------------------------------------
    def data_points(self) -> np.ndarray:
        ir, ip, iz = np.unravel_index(self.d_glob, (self.nRg, self.nPg, self.nZg))
        return np.column_stack((self.Rg[ir], self.Pg[ip], self.Zg[iz]))

    def assemble_rhs(self, bfield: Any, wall: Any) -> None:
        """``B`` at the data points -> ``Bs = median |B|``, the scaled operator and right-hand side ``b``."""

        points = self.data_points()
        B = np.empty((points.shape[0], 3))
        for start in range(0, points.shape[0], _B_CHUNK):
            B[start:start + _B_CHUNK] = np.asarray(
                bfield.evaluate_cylindrical(points[start:start + _B_CHUNK]), dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(B)):
            raise ValueError("bfield returned non-finite values at the fit data points")
        Bs = float(np.median(np.linalg.norm(B, axis=1)))
        self.set_scale(Bs)
        ir, ip, iz = np.unravel_index(self.d_glob, (self.nRg, self.nPg, self.nZg))
        sw = self._sR[ir] * self._sP[ip] * self._sZ[iz] / Bs
        b = np.zeros(self.m)
        for c in range(3):
            b[c * self.Nd:(c + 1) * self.Nd] = sw * B[:, c]
        self.b = jnp.asarray(b)
        del B, points, b
        # G check: circulation along the vessel reference axis (R_a(phi), phi, Z_a(phi)) over the table planes of
        # one period, (1/N) sum [R B_phi + B_R dR_a/dphi + B_Z dZ_a/dphi] = G when B = grad(G phi + Phi~).
        planes = np.asarray(getattr(bfield, "phi", self.phi0 + self.period * np.arange(90) / 90), dtype=np.float64)
        d = 1.0e-4
        aR, aZ = (np.asarray(v, dtype=np.float64) for v in wall.reference_axis(planes)[:2])
        aRp, aZp = (np.asarray(v, dtype=np.float64) for v in wall.reference_axis(planes + d)[:2])
        aRm, aZm = (np.asarray(v, dtype=np.float64) for v in wall.reference_axis(planes - d)[:2])
        Ba = np.asarray(bfield.evaluate_cylindrical(np.column_stack((aR, planes, aZ))), dtype=np.float64)
        G_loop = float(np.mean(aR * Ba[:, 1] + Ba[:, 0] * (aRp - aRm) / (2 * d) + Ba[:, 2] * (aZp - aZm) / (2 * d)))
        self.diag.update(Bs=Bs, G_loop=G_loop, G_loop_kind="circulation along the vessel reference axis")

    # -- misfit monitor ---------------------------------------------------------------------------------------------
    def misfit_kernel(self, sample: np.ndarray | None = None) -> Callable:
        """``x -> tan angle(grad chi, B)`` at the data points ``sample`` (indices into the data rows; all if None).

        The data rows of ``A x`` and of ``b`` are ``grad chi`` and ``B`` times the same positive weight, so the angle
        needs no further evaluation."""

        Nd = self.Nd
        idx = jnp.arange(Nd) if sample is None else jnp.asarray(np.asarray(sample, dtype=np.int32))
        b = self.b
        Bv = jnp.stack([b[idx], b[Nd + idx], b[2 * Nd + idx]], axis=-1)

        @jax.jit
        def kernel(x):
            y = self._matvec(x)
            gv = jnp.stack([y[idx], y[Nd + idx], y[2 * Nd + idx]], axis=-1)
            gn = gv / jnp.linalg.norm(gv, axis=-1, keepdims=True)
            bn = Bv / jnp.linalg.norm(Bv, axis=-1, keepdims=True)
            return jnp.linalg.norm(jnp.cross(gn, bn), axis=-1) / jnp.maximum(jnp.abs(jnp.sum(gn * bn, -1)), 1e-300)

        return kernel


# ---------------------------------------------------------------------------------------------------------------------
# LSMR (scipy's recurrences, damp = 0; scalar recurrences on the host, vector work in two jitted calls per iteration)
# ---------------------------------------------------------------------------------------------------------------------
def _sym_ortho(a: float, b: float) -> tuple[float, float, float]:
    if b == 0:
        return math.copysign(1.0, a), 0.0, abs(a)
    if a == 0:
        return 0.0, math.copysign(1.0, b), abs(b)
    if abs(b) > abs(a):
        tau = a / b
        s = math.copysign(1.0, b) / math.sqrt(1 + tau * tau)
        return s * tau, s, b / s
    tau = b / a
    c = math.copysign(1.0, a) / math.sqrt(1 + tau * tau)
    return c, c * tau, a / c


def _lsmr_kernels(system: _SplineFitSystem, dn):
    dnj = jnp.asarray(dn)

    @jax.jit
    def mv(y):
        return system._matvec(y / dnj)

    @jax.jit
    def rmv(z):
        return system._rmatvec(z) / dnj

    def _bidiag(u, v, alpha):
        u = system._matvec(v / dnj) - alpha * u
        beta = jnp.linalg.norm(u)
        u = jnp.where(beta > 0, u * (1.0 / beta), u)
        vv = system._rmatvec(u) / dnj - beta * v
        alpha_new = jnp.linalg.norm(vv)
        vv = jnp.where((beta > 0) & (alpha_new > 0), vv * (1.0 / alpha_new), jnp.where(beta > 0, vv, v))
        alpha_new = jnp.where(beta > 0, alpha_new, alpha)
        return u, vv, alpha_new, beta

    def _update(x, hvec, hbar, v, c1, c2, c3):
        hbar = hvec - c1 * hbar
        x = x + c2 * hbar
        hvec = v - c3 * hvec
        return x, hvec, hbar, jnp.linalg.norm(x)

    return mv, rmv, jax.jit(_bidiag, donate_argnums=(0, 1)), jax.jit(_update, donate_argnums=(0, 1, 2))


def _lsmr(system: _SplineFitSystem, dn, b, *, y0=None, atol: float = 1e-10, btol: float = 1e-10,
          conlim: float = 1e12, maxiter: int = 20000, callback: Callable | None = None):
    """LSMR on the column-scaled operator (``y = D x``), warm-started at ``y0`` (solves for ``y - y0``).

    Returns ``(y, istop, itn, normr, normar, normA, condA, normx)``; ``callback(itn, y_fn, normr, normar, normA,
    condA)`` returning True stops with ``istop = 8`` (``y_fn()`` returns the current iterate as a JAX array)."""

    mv, rmv, bidiag, update = _lsmr_kernels(system, dn)
    bj = jnp.asarray(b)
    if y0 is not None:
        y0 = jnp.asarray(y0)
        bj = bj - mv(y0)
    normb = float(jnp.linalg.norm(bj))
    n = system.n
    x = jnp.zeros(n)
    if normb == 0.0:
        return (x if y0 is None else y0), 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0
    beta = normb
    u = bj * (1.0 / beta)
    v = rmv(u)
    alpha = float(jnp.linalg.norm(v))
    v = v * (1.0 / alpha)
    itn, zetabar, alphabar, rho, rhobar, cbar, sbar = 0, alpha * beta, alpha, 1.0, 1.0, 1.0, 0.0
    hvec, hbar = v + 0.0, jnp.zeros(n)
    betadd, betad, rhodold, tautildeold, thetatilde, zeta, d = beta, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
    normA2 = alpha * alpha
    maxrbar, minrbar = 0.0, 1e100
    normA, condA, normx = math.sqrt(normA2), 1.0, 0.0
    istop, ctol = 0, 1.0 / conlim
    normr, normar = beta, alpha * beta
    current = (lambda: x) if y0 is None else (lambda: x + y0)
    while itn < maxiter:
        itn += 1
        u, v, alpha_j, beta_j = bidiag(u, v, jnp.asarray(alpha))
        alpha, beta = float(alpha_j), float(beta_j)
        chat, shat, alphahat = _sym_ortho(alphabar, 0.0)
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha
        rhobarold, zetaold = rhobar, zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar
        c1 = thetabar * rho / (rhoold * rhobarold)
        c2 = zeta / (rho * rhobar)
        c3 = thetanew / rho
        x, hvec, hbar, normx_j = update(x, hvec, hbar, v, c1, c2, c3)
        betaacute, betacheck = chat * betadd, -shat * betadd
        betahat, betadd = c * betaacute, -s * betaacute
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = math.sqrt(d + (betad - taud) ** 2 + betadd * betadd)
        normA2 += beta * beta
        normA = math.sqrt(normA2)
        normA2 += alpha * alpha
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
        normar = abs(zetabar)
        normx = float(normx_j)
        test1 = normr / normb
        test2 = normar / (normA * normr) if normA * normr != 0 else math.inf
        test3 = 1.0 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb
        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1
        if istop == 0 and callback is not None and callback(itn, current, normr, normar, normA, condA):
            istop = 8
        if istop > 0:
            break
    return current(), istop, itn, normr, normar, normA, condA, normx


class _MisfitMonitor:
    """Intrinsic stopping rule: every ``every`` iterations the tan-angle misfit ``angle(grad chi, B)`` RMS and max
    over a fixed subsample of the fit's own data points; stop when both change by less than ``rtol`` (relative)
    between consecutive checks, once ``itn >= min_iterations``."""

    def __init__(self, system: _SplineFitSystem, dn, *, every: int, rtol: float, min_iterations: int,
                 sample_size: int) -> None:
        self.every, self.rtol, self.min_iterations = int(every), float(rtol), int(min_iterations)
        count = min(int(sample_size), system.Nd)
        self.sample = np.unique(np.round(np.linspace(0, system.Nd - 1, count)).astype(np.int64))
        self._dn = jnp.asarray(dn)
        self._kernel = system.misfit_kernel(self.sample)
        self.history: list[dict[str, float]] = []

    def stats(self, y) -> tuple[float, float]:
        angle = np.asarray(self._kernel(jnp.asarray(y) / self._dn))
        return float(np.sqrt(np.mean(angle ** 2))), float(np.max(angle))

    def __call__(self, itn, y_fn, normr, normar, normA, condA) -> bool:
        if self.every <= 0 or itn % self.every:
            return False
        rms, mx = self.stats(y_fn())
        record = {"itn": int(itn), "rms": rms, "max": mx, "normr": float(normr), "normar": float(normar)}
        stop = False
        if self.history:
            previous = self.history[-1]
            change = max(abs(rms - previous["rms"]) / previous["rms"], abs(mx - previous["max"]) / previous["max"])
            record["relative_change"] = float(change)
            stop = change < self.rtol and itn >= self.min_iterations
        self.history.append(record)
        return stop


# ---------------------------------------------------------------------------------------------------------------------
# Exact dyadic prolongation (warm start)
# ---------------------------------------------------------------------------------------------------------------------
def _two_scale(p: int) -> np.ndarray:
    return np.array([math.comb(p + 1, q) for q in range(p + 2)], dtype=np.float64) / 2.0 ** p


def _prolongation_matrix(p: int, n_coarse: int, n_fine: int, *, periodic: bool):
    """``(n_fine, n_coarse)``: coarse function ``k`` = ``sum_q 2^-p C(p+1, q)`` fine function ``2k - p + q``
    (same lower edge, half the cell width; fine functions outside the fine box are dropped)."""

    a = _two_scale(p)
    rows, cols, vals = [], [], []
    for k in range(n_coarse):
        for q in range(p + 2):
            i = 2 * k - p + q
            if periodic:
                i %= n_fine
            elif not 0 <= i < n_fine:
                continue
            rows.append(i)
            cols.append(k)
            vals.append(a[q])
    return jnp.zeros((n_fine, n_coarse), dtype=jnp.float64).at[np.asarray(rows), np.asarray(cols)].add(
        jnp.asarray(vals))


def prolong_coefficients(coefficients, p: int, fine_shape: tuple[int, int, int]):
    """Exact refinement of ``(nRf_c, nphi_c, nZf_c)`` coefficients to a box with half the cell width in ``R``, ``Z``
    and ``phi`` (same lower edges; ``fine_shape = (nRf_f, 2 nphi_c, nZf_f)``)."""

    c = jnp.asarray(coefficients, dtype=jnp.float64)
    nRf, nphi, nZf = (int(v) for v in fine_shape)
    if nphi != 2 * c.shape[1]:
        raise ValueError("the fine phi cell count must be twice the coarse one")
    PR = _prolongation_matrix(p, c.shape[0], nRf, periodic=False)
    PP = _prolongation_matrix(p, c.shape[1], nphi, periodic=True)
    PZ = _prolongation_matrix(p, c.shape[2], nZf, periodic=False)
    return jnp.einsum("ia,jb,kc,abc->ijk", PR, PP, PZ, c)


# ---------------------------------------------------------------------------------------------------------------------
# Fit driver
# ---------------------------------------------------------------------------------------------------------------------
def _solve_system(system: _SplineFitSystem, *, y0_from: Callable | None, atol: float, btol: float, conlim: float,
                  maxiter: int, monitor: _MisfitMonitor | None, dn) -> tuple[Any, dict[str, Any]]:
    start = time.perf_counter()
    y0 = None if y0_from is None else y0_from(dn)
    y, istop, itn, normr, normar, normA, condA, normx = _lsmr(
        system, dn, system.b, y0=y0, atol=atol, btol=btol, conlim=conlim, maxiter=maxiter, callback=monitor)
    x = jnp.asarray(y) / jnp.asarray(dn)
    reason = {1: "atol/btol", 2: "atol (normal equations)", 3: "conlim", 4: "machine precision", 5: "machine precision",
              6: "machine precision", 7: "maxiter", 8: "misfit monitor"}.get(int(istop), "none")
    info = {"istop": int(istop), "stop_reason": reason, "iterations": int(itn), "normr": float(normr),
            "normar": float(normar), "normA": float(normA), "condA": float(condA), "atol": float(atol),
            "btol": float(btol), "conlim": float(conlim), "maxiter": int(maxiter),
            "seconds": float(time.perf_counter() - start)}
    return x, info


def _residual_stats(system: _SplineFitSystem, x) -> dict[str, float]:
    r = system.matvec(x) - system.b
    nd3 = 3 * system.Nd
    b = system.b
    return {"weighted_relative_l2_error": float(jnp.sqrt(jnp.sum(r[:nd3] ** 2) / jnp.sum(b[:nd3] ** 2))),
            "penalty_residual_norm": float(jnp.sqrt(jnp.sum(r[nd3:] ** 2)))}


def spline_scalar_potential_evaluator_from_bfield(
    bfield: Any,
    wall: Any,
    *,
    p: int = 5,
    h: float = 0.01,
    nphi: int = 90,
    coarse: Mapping[str, Any] | None = None,
    gauss: int = 3,
    rho_data: float = 1.02,
    lam: float = 1.0e-2,
    pad: float = 0.02,
    atol: float = 1.0e-10,
    btol: float = 1.0e-10,
    conlim: float = 1.0e12,
    maxiter: int = 5000,
    monitor_every: int = 100,
    monitor_rtol: float = 2.0e-3,
    monitor_min_iterations: int = 200,
    monitor_sample: int = 65536,
    log: Callable[[str], None] | None = None,
) -> SplineScalarPotentialEvaluator:
    """Fit ``chi = G phi + Phi~`` (uniform B-splines of degree ``p``) to ``bfield`` inside the vessel ``wall``.

    Least squares of ``grad(chi)`` against ``B`` (relative to ``Bs = median |B|``) on the ``gauss^3`` Gauss points of
    the knot cells with wall pseudo-radius ``<= rho_data`` (cylindrical volume weights), plus ``lam h^2 |Lap Phi~|^2``
    on the interior cells (every ``(p+1)^3`` function active).  ``G`` is an unknown; the result has ``I = 0``.  The knot
    box is :func:`spline_knot_box` (no other box is accepted).

    ``bfield`` is a magnetic-field evaluator (``R``, ``Z``, ``phi`` table axes, ``period``, ``nfp``,
    ``evaluate_cylindrical``) or a zero-argument callable returning one; a callable is loaded only for the
    right-hand-side assembly and released before the solves, so a large MAKEGRID evaluator is not resident during
    LSMR (an evaluator passed directly stays referenced by the caller).

    ``coarse`` (default ``dict(h=2 h, nphi=nphi // 2, maxiter=600)``; ``None``-valued keys take these defaults) is
    the warm start: the same fit with twice the cell width in ``R``, ``Z`` and ``phi``, ``maxiter`` LSMR iterations
    and no monitor, prolongated exactly to the fine basis.  Pass ``coarse=False`` for a cold start.

    The fine LSMR stops on the LSMR tolerances ``atol``/``btol``/``conlim``, on ``maxiter``, or on the intrinsic
    misfit monitor: every ``monitor_every`` iterations the tan-angle misfit ``angle(grad chi, B)`` RMS and max over
    ``monitor_sample`` fixed data points; stop when both change by less than ``monitor_rtol`` between checks, after
    at least ``monitor_min_iterations`` iterations (``monitor_every=0`` disables it).
    """

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("the spline scalar-potential fit requires JAX float64 (jax_enable_x64)")
    say = log if log is not None else (lambda message: None)
    total_start = time.perf_counter()
    loader = None
    if callable(bfield) and not hasattr(bfield, "evaluate_cylindrical"):
        loader = bfield
        bfield = loader()
    for name in ("R", "Z", "period", "nfp", "evaluate_cylindrical"):
        if not hasattr(bfield, name):
            raise TypeError(f"bfield must provide {name!r}")
    nfp = int(bfield.nfp)
    if int(wall.nfp) != nfp:
        raise ValueError(f"magnetic-field and wall field-period counts disagree: {nfp} != {wall.nfp}")
    period = float(bfield.period)
    if not np.isclose(period, 2.0 * np.pi / nfp, rtol=2e-12, atol=2e-12):
        raise ValueError("bfield.period is inconsistent with bfield.nfp")
    p, nphi, gauss = int(p), int(nphi), int(gauss)
    if coarse is False:
        coarse_options = None
    else:
        coarse_options = {"h": 2.0 * float(h), "nphi": nphi // 2, "maxiter": 600}
        coarse_options.update({k: v for k, v in dict(coarse or {}).items() if v is not None})
        unknown = set(coarse_options) - {"h", "nphi", "maxiter"}
        if unknown:
            raise ValueError(f"unknown coarse options {sorted(unknown)}")

    top_h = float(h) if coarse_options is None else float(coarse_options["h"])
    z_snap = max(Z_HALF_SNAP, 0.5 * top_h)

    def make(cell_width: float, phi_cells: int) -> _SplineFitSystem:
        R_lo, Z_lo, nR, nZ = spline_knot_box(bfield, wall, cell_width, pad, z_snap=z_snap)
        return _SplineFitSystem(wall, R_lo=R_lo, Z_lo=Z_lo, nR=nR, nZ=nZ, h=cell_width, p=p, nphi=phi_cells,
                                period=period, phi0=0.0, gauss=gauss, rho_data=rho_data, lam=lam)

    start = time.perf_counter()
    fine = make(float(h), nphi)
    systems = [fine]
    coarse_system = None
    if coarse_options is not None:
        coarse_system = make(float(coarse_options["h"]), int(coarse_options["nphi"]))
        if not (abs(coarse_system.h - 2.0 * fine.h) <= 1e-12 * fine.h and coarse_system.nphi * 2 == fine.nphi
                and abs(coarse_system.R_lo - fine.R_lo) <= 1e-12 and abs(coarse_system.Z_lo - fine.Z_lo) <= 1e-12):
            raise ValueError("the coarse warm-start box must nest in the fine box (2x cell width, same lower edges)")
        systems.append(coarse_system)
    for system in systems:
        system.assemble_rhs(bfield, wall)
    del systems
    G_loop = fine.diag["G_loop"]
    say(f"[spline-eta] systems built in {time.perf_counter() - start:.1f} s: fine {fine.nR}x{fine.nphi}x{fine.nZ} "
        f"cells, {fine.n} unknowns, {fine.Nd} data / {fine.Np} penalty points; G_loop {G_loop:.10g}")
    # The field evaluator is needed only for the right-hand sides.
    del bfield
    if loader is not None:
        import gc

        gc.collect()

    coarse_info = None
    y0_from = None
    if coarse_system is not None:
        dn_c = jnp.sqrt(coarse_system.column_norms_squared())
        if not bool(jnp.all(dn_c > 0)):
            raise RuntimeError("the coarse fit has a zero column")
        x_c, coarse_info = _solve_system(coarse_system, y0_from=None, atol=atol, btol=btol, conlim=conlim,
                                         maxiter=int(coarse_options["maxiter"]), monitor=None, dn=dn_c)
        coarse_info.update(h=coarse_system.h, nphi=coarse_system.nphi, unknowns=coarse_system.n)
        coef_c = coarse_system._dense(x_c)
        G_c = x_c[-1]
        say(f"[spline-eta] coarse fit: {coarse_info['iterations']} iterations ({coarse_info['stop_reason']}) in "
            f"{coarse_info['seconds']:.1f} s; G {float(G_c):.10g}")
        coef_f = prolong_coefficients(coef_c, p, (fine.nRf, fine.nphi, fine.nZf))
        x0 = jnp.concatenate([coef_f.reshape(-1)[jnp.asarray(fine.act_idx)], jnp.asarray(G_c)[None]])
        del coarse_system, coef_c, coef_f, dn_c, x_c
        y0_from = lambda dn: x0 * dn  # noqa: E731
    dn = jnp.sqrt(fine.column_norms_squared())
    if not bool(jnp.all(dn > 0)):
        raise RuntimeError("the fit has a zero column")
    monitor = None
    if int(monitor_every) > 0:
        monitor = _MisfitMonitor(fine, dn, every=monitor_every, rtol=monitor_rtol,
                                 min_iterations=monitor_min_iterations, sample_size=monitor_sample)
    x, info = _solve_system(fine, y0_from=y0_from, atol=atol, btol=btol, conlim=conlim, maxiter=int(maxiter),
                            monitor=monitor, dn=dn)
    G = float(x[-1])
    angle_all = np.asarray(fine.misfit_kernel(None)(x))
    stats = _residual_stats(fine, x)
    coef = np.asarray(fine._dense(x))
    say(f"[spline-eta] fine fit: {info['iterations']} iterations ({info['stop_reason']}) in {info['seconds']:.1f} s; "
        f"G {G:.10g} (G_loop {G_loop:.10g}); misfit rms {np.sqrt(np.mean(angle_all ** 2)):.3e} "
        f"max {angle_all.max():.3e}")
    diagnostics = {
        "method": "uniform tensor-product B-spline least squares of grad(chi) vs B, interior Laplacian penalty",
        "degree": p, "h": float(h), "nphi": nphi, "gauss": gauss, "rho_data": float(rho_data), "lam": float(lam),
        "pad": float(pad), "z_snap": float(z_snap),
        "knot_box": {"R_lo": fine.R_lo, "R_hi": fine.R_lo + fine.nR * fine.h, "Z_lo": fine.Z_lo,
                     "Z_hi": fine.Z_lo + fine.nZ * fine.h, "nR": fine.nR, "nZ": fine.nZ, "phi0": 0.0},
        "unknown_count": int(fine.n), "data_points": int(fine.Nd), "penalty_points": int(fine.Np),
        "data_cells": int(fine.n_data_cells), "penalty_cells": int(fine.n_pen_cells),
        "Bs": float(fine.Bs), "G": G, "I": 0.0, "G_loop": float(G_loop),
        "G_relative_difference_vs_loop": (G - G_loop) / G_loop,
        "lsmr": info, "coarse": coarse_info,
        "monitor": None if monitor is None else {"every": int(monitor_every), "rtol": float(monitor_rtol),
                                                 "min_iterations": int(monitor_min_iterations),
                                                 "sample_points": int(monitor.sample.size),
                                                 "history": monitor.history},
        "misfit_tan_angle_all_data": {"rms": float(np.sqrt(np.mean(angle_all ** 2))), "max": float(angle_all.max())},
        **stats,
        "seconds": float(time.perf_counter() - total_start),
    }
    return SplineScalarPotentialEvaluator(coef, G=G, R_lo=fine.R_lo, h_R=fine.h, Z_lo=fine.Z_lo, h_Z=fine.h, nfp=nfp,
                                          degree=p, phi0=0.0, diagnostics=diagnostics)


__all__ = [
    "DEFAULT_PHI_NEWTON_ITERATIONS",
    "JaxSplinePotential",
    "SplineScalarPotentialEvaluator",
    "Z_HALF_SNAP",
    "cardinal_bspline_basis",
    "prolong_coefficients",
    "spline_knot_box",
    "spline_scalar_potential_evaluator_from_bfield",
    "wall_pseudo_radius",
]
