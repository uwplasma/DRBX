"""Shared interface for magnetic scalar-potential coordinates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np


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

    def evaluate_and_gradient_cartesian(
        self, points_xyz: Any, *, wrapped: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate eta and its Cartesian gradient.

        Concrete evaluators may override this to share basis construction. The
        default preserves compatibility for third-party implementations.
        """

        return (
            self.evaluate_cartesian(points_xyz, wrapped=wrapped),
            self.gradient_cartesian(points_xyz),
        )

    @abstractmethod
    def evaluate_magnetic_potential_cylindrical(
        self, points_rphiz: Any
    ) -> np.ndarray:
        """Evaluate the branch-aware magnetic scalar potential Phi."""

    @abstractmethod
    def magnetic_field_cylindrical(self, points_rphiz: Any) -> np.ndarray:
        """Evaluate the fitted field grad(Phi) in cylindrical components."""


    @abstractmethod
    def magnetic_field_cartesian(self, points_xyz: Any) -> np.ndarray:
        """Evaluate the fitted field grad(Phi) in Cartesian components."""

    def __call__(self, points_xyz: Any) -> np.ndarray:
        """Evaluate unwrapped normalized eta at Cartesian points."""

        return self.evaluate_cartesian(points_xyz)
