from __future__ import annotations

from dataclasses import dataclass

import jax
from .fci_model import FciModelState


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci4FieldState(FciModelState):
    density: jax.Array
    omega: jax.Array
    v_ion_parallel: jax.Array
    v_electron_parallel: jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci4FieldRhsParameters:
    """Physical and inversion parameters for the four-field model."""

    rho_star: float = 1.0
    Te: float = 1.0
    mi_over_me: float = 1836.0
    phi_inversion_tol: float = 1.0e-6
    phi_inversion_maxiter: int = 50
    phi_inversion_restart: int = 50
    phi_inversion_regularization: float = 0.0

    def tree_flatten(self):
        return (
            (
                self.rho_star,
                self.Te,
                self.mi_over_me,
                self.phi_inversion_tol,
                self.phi_inversion_maxiter,
                self.phi_inversion_restart,
                self.phi_inversion_regularization,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (
            rho_star,
            Te,
            mi_over_me,
            phi_inversion_tol,
            phi_inversion_maxiter,
            phi_inversion_restart,
            phi_inversion_regularization,
        ) = children
        return cls(
            rho_star=rho_star,
            Te=Te,
            mi_over_me=mi_over_me,
            phi_inversion_tol=phi_inversion_tol,
            phi_inversion_maxiter=phi_inversion_maxiter,
            phi_inversion_restart=phi_inversion_restart,
            phi_inversion_regularization=phi_inversion_regularization,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci4FieldFreeDecayParameters:
    """Four-field free-decay parameters, including perpendicular diffusion."""

    rho_star: float = 1.0
    Te: float = 1.0
    mi_over_me: float = 1836.0
    phi_inversion_tol: float = 1.0e-6
    phi_inversion_maxiter: int = 50
    phi_inversion_restart: int = 50
    phi_inversion_regularization: float = 0.0
    density_perp_diffusion: float = 1.0e-2
    omega_perp_diffusion: float = 1.0e-2
    v_ion_parallel_perp_diffusion: float = 1.0e-2
    v_electron_parallel_perp_diffusion: float = 1.0e-2

    def rhs_parameters(self) -> Fci4FieldRhsParameters:
        return Fci4FieldRhsParameters(
            rho_star=self.rho_star,
            Te=self.Te,
            mi_over_me=self.mi_over_me,
            phi_inversion_tol=self.phi_inversion_tol,
            phi_inversion_maxiter=self.phi_inversion_maxiter,
            phi_inversion_restart=self.phi_inversion_restart,
            phi_inversion_regularization=self.phi_inversion_regularization,
        )

    def tree_flatten(self):
        return (
            (
                self.rho_star,
                self.Te,
                self.mi_over_me,
                self.phi_inversion_tol,
                self.phi_inversion_maxiter,
                self.phi_inversion_restart,
                self.phi_inversion_regularization,
                self.density_perp_diffusion,
                self.omega_perp_diffusion,
                self.v_ion_parallel_perp_diffusion,
                self.v_electron_parallel_perp_diffusion,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (
            rho_star,
            Te,
            mi_over_me,
            phi_inversion_tol,
            phi_inversion_maxiter,
            phi_inversion_restart,
            phi_inversion_regularization,
            density_perp_diffusion,
            omega_perp_diffusion,
            v_ion_parallel_perp_diffusion,
            v_electron_parallel_perp_diffusion,
        ) = children
        return cls(
            rho_star=rho_star,
            Te=Te,
            mi_over_me=mi_over_me,
            phi_inversion_tol=phi_inversion_tol,
            phi_inversion_maxiter=phi_inversion_maxiter,
            phi_inversion_restart=phi_inversion_restart,
            phi_inversion_regularization=phi_inversion_regularization,
            density_perp_diffusion=density_perp_diffusion,
            omega_perp_diffusion=omega_perp_diffusion,
            v_ion_parallel_perp_diffusion=v_ion_parallel_perp_diffusion,
            v_electron_parallel_perp_diffusion=v_electron_parallel_perp_diffusion,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci4FieldBlobParameters:
    """Four-field blob/interchange parameters."""

    rho_star: float = 1.0
    Te: float = 1.0
    mi_over_me: float = 1836.0
    phi_inversion_tol: float = 1.0e-6
    phi_inversion_maxiter: int = 50
    phi_inversion_restart: int = 50
    phi_inversion_regularization: float = 0.0
    density_perp_diffusion: float = 1.0e-2
    omega_perp_diffusion: float = 1.0e-2
    v_ion_parallel_perp_diffusion: float = 1.0e-2
    v_electron_parallel_perp_diffusion: float = 1.0e-2

    def rhs_parameters(self) -> Fci4FieldRhsParameters:
        return Fci4FieldRhsParameters(
            rho_star=self.rho_star,
            Te=self.Te,
            mi_over_me=self.mi_over_me,
            phi_inversion_tol=self.phi_inversion_tol,
            phi_inversion_maxiter=self.phi_inversion_maxiter,
            phi_inversion_restart=self.phi_inversion_restart,
            phi_inversion_regularization=self.phi_inversion_regularization,
        )

    def tree_flatten(self):
        return (
            (
                self.rho_star,
                self.Te,
                self.mi_over_me,
                self.phi_inversion_tol,
                self.phi_inversion_maxiter,
                self.phi_inversion_restart,
                self.phi_inversion_regularization,
                self.density_perp_diffusion,
                self.omega_perp_diffusion,
                self.v_ion_parallel_perp_diffusion,
                self.v_electron_parallel_perp_diffusion,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (
            rho_star,
            Te,
            mi_over_me,
            phi_inversion_tol,
            phi_inversion_maxiter,
            phi_inversion_restart,
            phi_inversion_regularization,
            density_perp_diffusion,
            omega_perp_diffusion,
            v_ion_parallel_perp_diffusion,
            v_electron_parallel_perp_diffusion,
        ) = children
        return cls(
            rho_star=rho_star,
            Te=Te,
            mi_over_me=mi_over_me,
            phi_inversion_tol=phi_inversion_tol,
            phi_inversion_maxiter=phi_inversion_maxiter,
            phi_inversion_restart=phi_inversion_restart,
            phi_inversion_regularization=phi_inversion_regularization,
            density_perp_diffusion=density_perp_diffusion,
            omega_perp_diffusion=omega_perp_diffusion,
            v_ion_parallel_perp_diffusion=v_ion_parallel_perp_diffusion,
            v_electron_parallel_perp_diffusion=v_electron_parallel_perp_diffusion,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci4FieldRhsResult:
    rhs: Fci4FieldState

    def tree_flatten(self):
        return ((self.rhs,), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (rhs,) = children
        return cls(rhs=rhs)


__all__ = [
    "Fci4FieldBlobParameters",
    "Fci4FieldFreeDecayParameters",
    "Fci4FieldState",
    "Fci4FieldRhsParameters",
    "Fci4FieldRhsResult",
]
