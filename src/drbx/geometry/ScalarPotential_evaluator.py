r"""Magnetic scalar-potential evaluators fitted to a magnetic field.

The fitted potential is the weighted least-squares projection

.. math::

    \Phi^\star = \arg\min_\Phi
        \int_\Omega |\nabla\Phi-\mathbf B|^2\,R\,dR\,d\phi\,dZ.

The concrete implementation fits the magnetic potential

.. math::

    \Phi = I\theta_{\rm ref} + G(\phi-\phi_0) + \widetilde\Phi,

and exposes the normalized toroidal mesh coordinate

.. math::

    \eta = (\phi-\phi_0) + \widetilde\Phi/G.

Here ``Phi_tilde`` is periodic over one field period. It is expanded in
Chebyshev polynomials in ``R`` and ``Z`` and a Fourier series in ``phi``.
If a branch-aware reference axis is provided, ``I`` and ``G`` are fitted
jointly. Consequently eta, Phi, and their gradients are evaluated from one
analytic representation; no nodal-potential interpolation or finite
difference is performed after the fit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from numpy.polynomial.chebyshev import chebder, chebval, chebvander

from .Bfield_evaluator import BFieldEvaluator


ReferenceAxis = Callable[
    [np.ndarray],
    tuple[Any, Any, Any, Any],
]


class ScalarPotentialEvaluator(ABC):
    """General interface for a fitted scalar potential and mesh coordinate.

    Point arrays have shape ``(..., 3)``. Cylindrical points use
    ``(R, phi, Z)`` and cylindrical gradients use the physical-component
    ordering ``(d/dR, (1/R)d/dphi, d/dZ)``. The primary ``evaluate`` and
    ``gradient`` methods refer to normalized eta, not magnetic Phi.
    """

    @property
    @abstractmethod
    def R(self) -> np.ndarray:
        """Return the radial fit grid."""

    @property
    @abstractmethod
    def phi(self) -> np.ndarray:
        """Return the one-field-period toroidal fit grid."""

    @property
    @abstractmethod
    def Z(self) -> np.ndarray:
        """Return the vertical fit grid."""

    @property
    @abstractmethod
    def nfp(self) -> int:
        """Return the number of field periods."""

    @property
    @abstractmethod
    def period(self) -> float:
        """Return one toroidal field period in radians."""

    @property
    @abstractmethod
    def G(self) -> float:
        """Return the fitted secular toroidal-potential coefficient."""

    @property
    @abstractmethod
    def I(self) -> float:
        """Return the fitted secular poloidal-potential coefficient."""

    @property
    @abstractmethod
    def diagnostics(self) -> Mapping[str, Any]:
        """Return immutable fit diagnostics."""

    @abstractmethod
    def evaluate_cylindrical(
        self, points_rphiz: Any, *, wrapped: bool = False
    ) -> np.ndarray:
        """Evaluate eta at cylindrical points."""

    @abstractmethod
    def gradient_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        """Evaluate physical cylindrical components of grad(eta)."""

    @abstractmethod
    def evaluate_cartesian(
        self, points_xyz: Any, *, wrapped: bool = False
    ) -> np.ndarray:
        """Evaluate eta at Cartesian points using the principal phi branch."""

    @abstractmethod
    def gradient_cartesian(self, points_xyz: Any) -> np.ndarray:
        """Evaluate Cartesian components of grad(eta)."""

    @abstractmethod
    def evaluate_magnetic_potential_cylindrical(
        self, points_rphiz: Any
    ) -> np.ndarray:
        """Evaluate the branch-aware magnetic scalar potential Phi."""

    @abstractmethod
    def magnetic_field_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        """Evaluate the fitted field grad(Phi) in cylindrical components."""

    @abstractmethod
    def evaluate_magnetic_potential_cartesian(
        self, points_xyz: Any
    ) -> np.ndarray:
        """Evaluate Phi using principal cylindrical angle branches."""

    @abstractmethod
    def magnetic_field_cartesian(self, points_xyz: Any) -> np.ndarray:
        """Evaluate the fitted field grad(Phi) in Cartesian components."""

    def __call__(self, points_xyz: Any) -> np.ndarray:
        """Evaluate unwrapped normalized eta at Cartesian points."""

        return self.evaluate_cartesian(points_xyz)


class ChebyshevFourierScalarPotentialEvaluator(ScalarPotentialEvaluator):
    """Chebyshev-Fourier representation of a fitted scalar potential."""

    def __init__(
        self,
        R: Any,
        phi: Any,
        Z: Any,
        coefficients: Any,
        *,
        G: float,
        I: float = 0.0,
        nfp: int,
        reference_axis: ReferenceAxis | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        self._R = _axis(R, "R")
        self._phi = _axis(phi, "phi")
        self._Z = _axis(Z, "Z")
        if self._R.size < 2 or self._phi.size < 2 or self._Z.size < 2:
            raise ValueError("R, phi, and Z grids must each contain at least 2 points")
        if self._R[0] <= 0.0:
            raise ValueError("R grid must be strictly positive")
        if int(nfp) != nfp or int(nfp) < 1:
            raise ValueError("nfp must be a positive integer")
        if not np.isfinite(G) or not np.isfinite(I):
            raise ValueError("I and G must be finite")
        if I != 0.0 and reference_axis is None:
            raise ValueError("nonzero I requires a reference_axis")

        coefficient_array = np.asarray(coefficients, dtype=np.float64)
        if coefficient_array.ndim != 3:
            raise ValueError("coefficients must have shape (nR, nZ, nFourier)")
        if coefficient_array.shape[2] < 1 or coefficient_array.shape[2] % 2 != 1:
            raise ValueError("the Fourier coefficient dimension must be positive and odd")
        if not np.all(np.isfinite(coefficient_array)):
            raise ValueError("coefficients must be finite")

        self._coefficients = coefficient_array.copy()
        self._G = float(G)
        self._I = float(I)
        self._reference_axis = reference_axis
        self._nfp = int(nfp)
        self._period = 2.0 * np.pi / self._nfp
        self._phi0 = float(self._phi[0])
        expected_span = self._phi[-1] - self._phi[0] + (
            self._phi[1] - self._phi[0]
        )
        if not np.isclose(
            expected_span, self._period, rtol=2e-7, atol=2e-12
        ):
            raise ValueError(
                "phi grid must span one field period without a duplicate endpoint"
            )
        self._diagnostics = MappingProxyType(
            dict(diagnostics) if diagnostics is not None else {}
        )

    @property
    def R(self) -> np.ndarray:
        return self._R.copy()

    @property
    def phi(self) -> np.ndarray:
        return self._phi.copy()

    @property
    def Z(self) -> np.ndarray:
        return self._Z.copy()

    @property
    def nfp(self) -> int:
        return self._nfp

    @property
    def period(self) -> float:
        return self._period

    @property
    def G(self) -> float:
        return self._G

    @property
    def I(self) -> float:
        return self._I

    @property
    def I_over_G(self) -> float:
        return self._I / self._G if self._G != 0.0 else np.nan

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        return self._diagnostics

    @property
    def radial_degree(self) -> int:
        return self._coefficients.shape[0] - 1

    @property
    def vertical_degree(self) -> int:
        return self._coefficients.shape[1] - 1

    @property
    def toroidal_modes(self) -> int:
        return (self._coefficients.shape[2] - 1) // 2

    @property
    def circulation_per_period(self) -> float:
        """Return the change in eta after one field period."""

        return self._G * self._period

    def evaluate_cylindrical(
        self, points_rphiz: Any, *, wrapped: bool = False
    ) -> np.ndarray:
        """Evaluate normalized eta, optionally modulo one field period."""

        if abs(self._G) <= np.finfo(np.float64).eps:
            raise ValueError("normalized eta is undefined because fitted G is zero")
        points, leading_shape = _points(points_rphiz, "points_rphiz")
        self._check_bounds(points)
        value, _, _, _ = self._periodic_terms(points)
        eta = points[:, 1] - self._phi0 + value / self._G
        if wrapped:
            eta = np.mod(eta, self._period)
        return eta.reshape(leading_shape)

    def evaluate_phase_cylindrical(
        self, points_rphiz: Any, *, wrapped: bool = False
    ) -> np.ndarray:
        """Alias for normalized eta; wrapping must be requested explicitly."""

        return self.evaluate_cylindrical(points_rphiz, wrapped=wrapped)

    def gradient_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        if abs(self._G) <= np.finfo(np.float64).eps:
            raise ValueError("grad(eta) is undefined because fitted G is zero")
        points, leading_shape = _points(points_rphiz, "points_rphiz")
        self._check_bounds(points)
        _, derivative_R, derivative_phi, derivative_Z = self._periodic_terms(
            points
        )
        result = np.column_stack(
            (
                derivative_R / self._G,
                (1.0 + derivative_phi / self._G) / points[:, 0],
                derivative_Z / self._G,
            )
        )
        return result.reshape(leading_shape + (3,))

    def evaluate_cartesian(
        self, points_xyz: Any, *, wrapped: bool = False
    ) -> np.ndarray:
        """Evaluate eta using ``atan2(Y, X)`` as its toroidal branch."""

        points, leading_shape = _points(points_xyz, "points_xyz")
        cylindrical, _ = _cartesian_to_cylindrical(points)
        return self.evaluate_cylindrical(
            cylindrical, wrapped=wrapped
        ).reshape(leading_shape)

    def evaluate_phase_cartesian(
        self, points_xyz: Any, *, wrapped: bool = False
    ) -> np.ndarray:
        """Alias for normalized eta; wrapping must be requested explicitly."""

        points, leading_shape = _points(points_xyz, "points_xyz")
        cylindrical, _ = _cartesian_to_cylindrical(points)
        return self.evaluate_phase_cylindrical(
            cylindrical, wrapped=wrapped
        ).reshape(leading_shape)

    def gradient_cartesian(self, points_xyz: Any) -> np.ndarray:
        points, leading_shape = _points(points_xyz, "points_xyz")
        cylindrical, phi = _cartesian_to_cylindrical(points)
        gradient = self.gradient_cylindrical(cylindrical).reshape((-1, 3))
        cosine = np.cos(phi)
        sine = np.sin(phi)
        result = np.empty_like(gradient)
        result[:, 0] = gradient[:, 0] * cosine - gradient[:, 1] * sine
        result[:, 1] = gradient[:, 0] * sine + gradient[:, 1] * cosine
        result[:, 2] = gradient[:, 2]
        return result.reshape(leading_shape + (3,))

    def evaluate_magnetic_potential_cylindrical(
        self, points_rphiz: Any
    ) -> np.ndarray:
        """Evaluate ``Phi = I*theta_ref + G*(phi-phi0) + Phi_tilde``."""

        points, leading_shape = _points(points_rphiz, "points_rphiz")
        self._check_bounds(points)
        periodic_value, _, _, _ = self._periodic_terms(points)
        theta, _ = self._theta_reference_terms(points)
        potential = (
            self._I * theta
            + self._G * (points[:, 1] - self._phi0)
            + periodic_value
        )
        return potential.reshape(leading_shape)

    def magnetic_field_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        """Evaluate the fitted magnetic field ``grad(Phi)``."""

        points, leading_shape = _points(points_rphiz, "points_rphiz")
        self._check_bounds(points)
        _, derivative_R, derivative_phi, derivative_Z = self._periodic_terms(
            points
        )
        _, theta_gradient = self._theta_reference_terms(points)
        result = np.column_stack(
            (
                derivative_R,
                (self._G + derivative_phi) / points[:, 0],
                derivative_Z,
            )
        )
        result += self._I * theta_gradient
        return result.reshape(leading_shape + (3,))

    def evaluate_magnetic_potential_cartesian(
        self, points_xyz: Any
    ) -> np.ndarray:
        points, leading_shape = _points(points_xyz, "points_xyz")
        cylindrical, _ = _cartesian_to_cylindrical(points)
        return self.evaluate_magnetic_potential_cylindrical(
            cylindrical
        ).reshape(leading_shape)

    def magnetic_field_cartesian(self, points_xyz: Any) -> np.ndarray:
        points, leading_shape = _points(points_xyz, "points_xyz")
        cylindrical, phi = _cartesian_to_cylindrical(points)
        field = self.magnetic_field_cylindrical(cylindrical).reshape((-1, 3))
        cosine = np.cos(phi)
        sine = np.sin(phi)
        result = np.empty_like(field)
        result[:, 0] = field[:, 0] * cosine - field[:, 1] * sine
        result[:, 1] = field[:, 0] * sine + field[:, 1] * cosine
        result[:, 2] = field[:, 2]
        return result.reshape(leading_shape + (3,))

    def reconstructed_field_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        """Backward-compatible alias for the fitted magnetic field."""

        return self.magnetic_field_cylindrical(points_rphiz)

    def _theta_reference_terms(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._reference_axis is None:
            return np.zeros(points.shape[0]), np.zeros((points.shape[0], 3))
        return _theta_reference_terms(points, self._reference_axis)

    def _check_bounds(self, points: np.ndarray) -> None:
        if np.any(points[:, 0] <= 0.0):
            raise ValueError("cylindrical query points must have R > 0")
        if np.any(points[:, 0] < self._R[0]) or np.any(
            points[:, 0] > self._R[-1]
        ):
            raise ValueError("R query lies outside the scalar-potential fit domain")
        if np.any(points[:, 2] < self._Z[0]) or np.any(
            points[:, 2] > self._Z[-1]
        ):
            raise ValueError("Z query lies outside the scalar-potential fit domain")

    def _periodic_terms(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        scaled_R = _scale_to_chebyshev(points[:, 0], self._R[0], self._R[-1])
        scaled_Z = _scale_to_chebyshev(points[:, 2], self._Z[0], self._Z[-1])
        basis_R = chebvander(scaled_R, self.radial_degree)
        basis_Z = chebvander(scaled_Z, self.vertical_degree)
        derivative_R = _chebyshev_derivative_vander(
            scaled_R, self.radial_degree
        ) * (2.0 / (self._R[-1] - self._R[0]))
        derivative_Z = _chebyshev_derivative_vander(
            scaled_Z, self.vertical_degree
        ) * (2.0 / (self._Z[-1] - self._Z[0]))
        basis_phi, derivative_phi = _fourier_vander(
            points[:, 1], self._phi0, self._nfp, self.toroidal_modes
        )
        value = np.einsum(
            "ni,nj,nk,ijk->n",
            basis_R,
            basis_Z,
            basis_phi,
            self._coefficients,
            optimize=True,
        )
        dR = np.einsum(
            "ni,nj,nk,ijk->n",
            derivative_R,
            basis_Z,
            basis_phi,
            self._coefficients,
            optimize=True,
        )
        dphi = np.einsum(
            "ni,nj,nk,ijk->n",
            basis_R,
            basis_Z,
            derivative_phi,
            self._coefficients,
            optimize=True,
        )
        dZ = np.einsum(
            "ni,nj,nk,ijk->n",
            basis_R,
            derivative_Z,
            basis_phi,
            self._coefficients,
            optimize=True,
        )
        return value, dR, dphi, dZ


def scalar_potential_evaluator_from_bfield(
    bfield: BFieldEvaluator,
    *,
    radial_degree: int = 5,
    vertical_degree: int = 5,
    toroidal_modes: int = 3,
    sample_shape: tuple[int, int, int] | None = None,
    R_bounds: tuple[float, float] | None = None,
    Z_bounds: tuple[float, float] | None = None,
    mask: Callable[[np.ndarray], Any] | None = None,
    reference_axis: ReferenceAxis | None = None,
    rcond: float | None = None,
) -> ScalarPotentialEvaluator:
    """Fit Phi and return its normalized toroidal mesh coordinate eta.

    ``sample_shape`` is ordered ``(nR, nphi, nZ)``. The objective uses
    node-centered cylindrical volume weights. The constant periodic basis
    function is removed to fix the otherwise arbitrary additive gauge.

    If ``reference_axis`` is supplied, it must map a phi array to
    ``(R_axis, Z_axis, dR_axis/dphi, dZ_axis/dphi)``. The fit then includes
    ``I*theta_ref`` and solves for I and G jointly. The active mask must
    exclude the reference axis itself and any desired poloidal branch cut.
    Without a reference axis, I is fixed to zero.
    """

    if not isinstance(bfield, BFieldEvaluator):
        raise TypeError("bfield must implement BFieldEvaluator")
    radial_degree = _nonnegative_integer(radial_degree, "radial_degree")
    vertical_degree = _nonnegative_integer(vertical_degree, "vertical_degree")
    toroidal_modes = _nonnegative_integer(toroidal_modes, "toroidal_modes")
    if radial_degree < 1 or vertical_degree < 1:
        raise ValueError("radial_degree and vertical_degree must be at least 1")

    source_R = np.asarray(bfield.R, dtype=np.float64)
    source_phi = np.asarray(bfield.phi, dtype=np.float64)
    source_Z = np.asarray(bfield.Z, dtype=np.float64)
    R_min, R_max = _fit_bounds(R_bounds, source_R, "R")
    Z_min, Z_max = _fit_bounds(Z_bounds, source_Z, "Z")
    period = float(bfield.period)
    phi0 = float(source_phi[0])

    if sample_shape is None:
        sample_shape = (
            max(2 * (radial_degree + 1), 8),
            max(2 * (2 * toroidal_modes + 1), 8),
            max(2 * (vertical_degree + 1), 8),
        )
    if len(sample_shape) != 3:
        raise ValueError("sample_shape must be a three-integer (nR, nphi, nZ) tuple")
    nR, nphi, nZ = (
        _positive_integer(value, f"sample_shape[{index}]")
        for index, value in enumerate(sample_shape)
    )
    if min(nR, nphi, nZ) < 2:
        raise ValueError("sample_shape entries must each be at least 2")
    minimum_phi = 2 * toroidal_modes + 1
    if nphi < minimum_phi:
        raise ValueError(
            f"sample_shape nphi must be at least {minimum_phi} for "
            f"{toroidal_modes} Fourier modes"
        )

    fit_R = np.linspace(R_min, R_max, nR)
    fit_phi = phi0 + np.arange(nphi, dtype=np.float64) * period / nphi
    fit_Z = np.linspace(Z_min, Z_max, nZ)
    RR, PP, ZZ = np.meshgrid(fit_R, fit_phi, fit_Z, indexing="ij")
    all_points = np.stack((RR, PP, ZZ), axis=-1).reshape((-1, 3))
    active = np.ones(all_points.shape[0], dtype=bool)
    if mask is not None:
        mask_values = np.asarray(mask(all_points), dtype=bool)
        if mask_values.shape not in {(all_points.shape[0],), RR.shape}:
            raise ValueError(
                "mask must return shape (npoints,) or sample_shape"
            )
        active = mask_values.reshape(-1)
    if not np.any(active):
        raise ValueError("mask excludes every scalar-potential fit point")
    points = all_points[active]
    reference = np.asarray(
        bfield.evaluate_cylindrical(points), dtype=np.float64
    ).reshape((-1, 3))
    if not np.all(np.isfinite(reference)):
        raise ValueError("bfield returned non-finite values on the fit grid")

    scaled_R = _scale_to_chebyshev(points[:, 0], R_min, R_max)
    scaled_Z = _scale_to_chebyshev(points[:, 2], Z_min, Z_max)
    basis_R = chebvander(scaled_R, radial_degree)
    basis_Z = chebvander(scaled_Z, vertical_degree)
    derivative_R = _chebyshev_derivative_vander(
        scaled_R, radial_degree
    ) * (2.0 / (R_max - R_min))
    derivative_Z = _chebyshev_derivative_vander(
        scaled_Z, vertical_degree
    ) * (2.0 / (Z_max - Z_min))
    basis_phi, derivative_phi = _fourier_vander(
        points[:, 1], phi0, bfield.nfp, toroidal_modes
    )

    shape = (radial_degree + 1, vertical_degree + 1, 2 * toroidal_modes + 1)
    full_count = int(np.prod(shape))
    coefficient_mask = np.ones(full_count, dtype=bool)
    coefficient_mask[0] = False

    def tensor(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.einsum(
            "ni,nj,nk->nijk", left, middle, right, optimize=True
        ).reshape((points.shape[0], full_count))[:, coefficient_mask]

    feature_R = tensor(derivative_R, basis_Z, basis_phi)
    feature_phi = tensor(basis_R, basis_Z, derivative_phi)
    feature_Z = tensor(basis_R, derivative_Z, basis_phi)
    sample_count = points.shape[0]
    periodic_unknown_count = full_count - 1
    I_index = periodic_unknown_count if reference_axis is not None else None
    G_index = periodic_unknown_count + int(reference_axis is not None)
    unknown_count = G_index + 1
    matrix = np.zeros((3 * sample_count, unknown_count), dtype=np.float64)
    matrix[:sample_count, :periodic_unknown_count] = feature_R
    matrix[
        sample_count : 2 * sample_count, :periodic_unknown_count
    ] = (
        feature_phi / points[:, 0, None]
    )
    matrix[
        2 * sample_count :, :periodic_unknown_count
    ] = feature_Z
    if I_index is not None:
        _, theta_gradient = _theta_reference_terms(points, reference_axis)
        matrix[:sample_count, I_index] = theta_gradient[:, 0]
        matrix[
            sample_count : 2 * sample_count, I_index
        ] = theta_gradient[:, 1]
        matrix[2 * sample_count :, I_index] = theta_gradient[:, 2]
    matrix[sample_count : 2 * sample_count, G_index] = 1.0 / points[:, 0]
    right_hand_side = np.concatenate(
        (reference[:, 0], reference[:, 1], reference[:, 2])
    )

    dR_weights = _node_widths(fit_R)
    dZ_weights = _node_widths(fit_Z)
    dphi = period / nphi
    volume_grid = (
        RR * dR_weights[:, None, None] * dphi * dZ_weights[None, None, :]
    )
    volume = volume_grid.reshape(-1)[active]
    row_weight = np.sqrt(np.concatenate((volume, volume, volume)))
    weighted_matrix = matrix * row_weight[:, None]
    weighted_rhs = right_hand_side * row_weight
    column_scale = np.linalg.norm(weighted_matrix, axis=0)
    if np.any(column_scale <= np.finfo(np.float64).tiny):
        raise ValueError("the fit system contains an unconstrained basis coefficient")
    scaled_matrix = weighted_matrix / column_scale[None, :]
    scaled_solution, _, rank, singular_values = np.linalg.lstsq(
        scaled_matrix, weighted_rhs, rcond=rcond
    )
    if rank < scaled_matrix.shape[1]:
        raise ValueError(
            "the scalar-potential fit is rank deficient; increase sample_shape, "
            "reduce basis degrees/modes, or enlarge the active mask"
        )
    solution = scaled_solution / column_scale

    coefficients_flat = np.zeros(full_count, dtype=np.float64)
    coefficients_flat[coefficient_mask] = solution[:periodic_unknown_count]
    coefficients = coefficients_flat.reshape(shape)
    I = 0.0 if I_index is None else float(solution[I_index])
    G = float(solution[G_index])
    evaluator = ChebyshevFourierScalarPotentialEvaluator(
        fit_R,
        fit_phi,
        fit_Z,
        coefficients,
        G=G,
        I=I,
        nfp=bfield.nfp,
        reference_axis=reference_axis,
    )

    fitted = evaluator.magnetic_field_cylindrical(points).reshape((-1, 3))
    residual = fitted - reference
    residual_squared = np.sum(residual**2, axis=1)
    reference_squared = np.sum(reference**2, axis=1)
    reference_magnitude = np.sqrt(reference_squared)
    residual_magnitude = np.sqrt(residual_squared)
    weighted_error = float(np.sum(volume * residual_squared))
    weighted_reference = float(np.sum(volume * reference_squared))
    if abs(G) > np.finfo(np.float64).eps:
        eta_gradient = evaluator.gradient_cylindrical(points).reshape((-1, 3))
        phase_derivative = points[:, 0] * eta_gradient[:, 1]
        minimum_phase_derivative = float(np.min(phase_derivative))
        maximum_phase_derivative = float(np.max(phase_derivative))
        folded_fraction = float(np.mean(phase_derivative <= 0.0))
    else:
        minimum_phase_derivative = np.nan
        maximum_phase_derivative = np.nan
        folded_fraction = np.nan
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else np.inf
    )
    diagnostics: dict[str, Any] = {
        "sample_count": int(sample_count),
        "unknown_count": int(unknown_count),
        "rank": int(rank),
        "condition_number": condition_number,
        "G": G,
        "I": I,
        "I_over_G": I / G if G != 0.0 else np.nan,
        "weighted_residual_norm": float(np.sqrt(weighted_error)),
        "weighted_relative_l2_error": float(
            np.sqrt(weighted_error / weighted_reference)
        )
        if weighted_reference > 0.0
        else np.nan,
        "rms_absolute_error": float(np.sqrt(np.mean(residual_squared))),
        "relative_l2_error": float(
            np.sqrt(np.sum(residual_squared) / np.sum(reference_squared))
        )
        if np.sum(reference_squared) > 0.0
        else np.nan,
        "max_absolute_error": float(np.max(residual_magnitude)),
        "max_relative_error": float(
            np.max(
                residual_magnitude
                / np.maximum(reference_magnitude, np.finfo(np.float64).tiny)
            )
        ),
        "component_rms_errors": np.sqrt(np.mean(residual**2, axis=0)),
        "min_deta_dphi": minimum_phase_derivative,
        "max_deta_dphi": maximum_phase_derivative,
        "min_normalized_phase_derivative": minimum_phase_derivative,
        "max_normalized_phase_derivative": maximum_phase_derivative,
        "folded_fraction": folded_fraction,
        "sample_shape": (nR, nphi, nZ),
        "radial_degree": radial_degree,
        "vertical_degree": vertical_degree,
        "toroidal_modes": toroidal_modes,
    }
    return ChebyshevFourierScalarPotentialEvaluator(
        fit_R,
        fit_phi,
        fit_Z,
        coefficients,
        G=G,
        I=I,
        nfp=bfield.nfp,
        reference_axis=reference_axis,
        diagnostics=diagnostics,
    )


def _axis(values: Any, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size == 0 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be finite and non-empty")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _points(values: Any, name: str) -> tuple[np.ndarray, tuple[int, ...]]:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1:] != (3,):
        raise ValueError(f"{name} must have shape (..., 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points.reshape((-1, 3)), points.shape[:-1]


def _cartesian_to_cylindrical(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    radius = np.hypot(points[:, 0], points[:, 1])
    if np.any(radius <= 0.0):
        raise ValueError("Cartesian query points must have R > 0")
    phi = np.arctan2(points[:, 1], points[:, 0])
    return np.column_stack((radius, phi, points[:, 2])), phi


def _theta_reference_terms(
    points: np.ndarray,
    reference_axis: ReferenceAxis,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        values = reference_axis(points[:, 1])
    except Exception as error:
        raise ValueError("reference_axis failed at the requested phi values") from error
    if not isinstance(values, tuple) or len(values) != 4:
        raise ValueError(
            "reference_axis must return "
            "(R_axis, Z_axis, dR_axis_dphi, dZ_axis_dphi)"
        )
    axis_values = []
    for value in values:
        try:
            axis_values.append(
                np.broadcast_to(
                    np.asarray(value, dtype=np.float64), (points.shape[0],)
                )
            )
        except ValueError as error:
            raise ValueError(
                "reference_axis outputs must broadcast to the phi input shape"
            ) from error
    axis_R, axis_Z, derivative_axis_R, derivative_axis_Z = axis_values
    if not all(np.all(np.isfinite(value)) for value in axis_values):
        raise ValueError("reference_axis returned non-finite values")

    relative_R = points[:, 0] - axis_R
    relative_Z = points[:, 2] - axis_Z
    radius_squared = relative_R**2 + relative_Z**2
    if np.any(radius_squared <= 1e-24):
        raise ValueError(
            "theta_ref is singular on the reference axis; exclude an axis core "
            "from the fit mask and query domain"
        )
    theta = np.arctan2(relative_Z, relative_R)
    derivative_theta_phi = (
        relative_Z * derivative_axis_R
        - relative_R * derivative_axis_Z
    ) / radius_squared
    gradient = np.column_stack(
        (
            -relative_Z / radius_squared,
            derivative_theta_phi / points[:, 0],
            relative_R / radius_squared,
        )
    )
    return theta, gradient


def _scale_to_chebyshev(
    values: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    return 2.0 * (values - lower) / (upper - lower) - 1.0


def _chebyshev_derivative_vander(
    values: np.ndarray, degree: int
) -> np.ndarray:
    result = np.zeros((values.size, degree + 1), dtype=np.float64)
    for index in range(1, degree + 1):
        coefficient = np.zeros(index + 1, dtype=np.float64)
        coefficient[index] = 1.0
        result[:, index] = chebval(values, chebder(coefficient))
    return result


def _fourier_vander(
    phi: np.ndarray, phi0: float, nfp: int, modes: int
) -> tuple[np.ndarray, np.ndarray]:
    basis = np.empty((phi.size, 2 * modes + 1), dtype=np.float64)
    derivative = np.zeros_like(basis)
    basis[:, 0] = 1.0
    angle = nfp * (phi - phi0)
    for mode in range(1, modes + 1):
        cosine_index = 2 * mode - 1
        sine_index = 2 * mode
        cosine = np.cos(mode * angle)
        sine = np.sin(mode * angle)
        frequency = mode * nfp
        basis[:, cosine_index] = cosine
        basis[:, sine_index] = sine
        derivative[:, cosine_index] = -frequency * sine
        derivative[:, sine_index] = frequency * cosine
    return basis, derivative


def _node_widths(axis: np.ndarray) -> np.ndarray:
    widths = np.empty_like(axis)
    widths[1:-1] = 0.5 * (axis[2:] - axis[:-2])
    widths[0] = 0.5 * (axis[1] - axis[0])
    widths[-1] = 0.5 * (axis[-1] - axis[-2])
    return widths


def _fit_bounds(
    requested: tuple[float, float] | None,
    source: np.ndarray,
    label: str,
) -> tuple[float, float]:
    if requested is None:
        return float(source[0]), float(source[-1])
    if len(requested) != 2:
        raise ValueError(f"{label}_bounds must contain (lower, upper)")
    lower, upper = (float(requested[0]), float(requested[1]))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"{label}_bounds must be finite and increasing")
    tolerance = 1e-12 * max(1.0, abs(source[0]), abs(source[-1]))
    if lower < source[0] - tolerance or upper > source[-1] + tolerance:
        raise ValueError(f"{label}_bounds must lie inside the B-field domain")
    return lower, upper


def _nonnegative_integer(value: Any, name: str) -> int:
    integer = int(value)
    if integer != value or integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return integer


def _positive_integer(value: Any, name: str) -> int:
    integer = int(value)
    if integer != value or integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


__all__ = [
    "ReferenceAxis",
    "ScalarPotentialEvaluator",
    "ChebyshevFourierScalarPotentialEvaluator",
    "scalar_potential_evaluator_from_bfield",
]
