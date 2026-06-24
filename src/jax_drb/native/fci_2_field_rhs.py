from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..geometry import FciGeometry3D
from .fci_operators import curvature_op, debug_only_grad_parallel_op, poisson_bracket_op


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci2FieldState:
    density: jax.Array
    v_parallel: jax.Array
    density_background: jax.Array

    def tree_flatten(self):
        return ((self.density, self.v_parallel, self.density_background), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci2FieldRhsParameters:
    """Placeholder parameter bundle for the reduced two-field FCI model."""

    rho_star: float = 1.0

    def tree_flatten(self):
        return ((self.rho_star,), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (rho_star,) = children
        return cls(rho_star=rho_star)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci2FieldRhsResult:
    rhs: Fci2FieldState

    def tree_flatten(self):
        return ((self.rhs,), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (rhs,) = children
        return cls(rhs=rhs)


def compute_2field_rhs(
    state: Fci2FieldState,
    *,
    geometry: FciGeometry3D,
    parameters: Fci2FieldRhsParameters = Fci2FieldRhsParameters(),
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    density_source: jax.Array | None = None,
    v_parallel_source: jax.Array | None = None,
) -> Fci2FieldRhsResult:
    """Assemble the reduced two-field FCI RHS."""

    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    magnetic_field = jnp.maximum(jnp.asarray(geometry.Bmag, dtype=jnp.float64), 1.0e-30)
    density = jnp.asarray(state.density, dtype=jnp.float64)
    v_parallel = jnp.asarray(state.v_parallel, dtype=jnp.float64)
    density_background = jnp.asarray(state.density_background, dtype=jnp.float64)
    phi = jnp.log(jnp.maximum(density, 1.0e-30) / jnp.maximum(density_background, 1.0e-30))

    poisson_density = poisson_bracket_op(phi, density, geometry, periodic_axes=periodic_axes)
    curvature_density = curvature_op(density, geometry, periodic_axes=periodic_axes)
    curvature_phi = curvature_op(phi, geometry, periodic_axes=periodic_axes)
    parallel_velocity_gradient = debug_only_grad_parallel_op(v_parallel, geometry, periodic_axes=periodic_axes)
    poisson_v_parallel = poisson_bracket_op(phi, v_parallel, geometry, periodic_axes=periodic_axes)

    density_rhs = (
        -(poisson_density / (rho_star * magnetic_field))
        + (2.0 / magnetic_field) * curvature_density
        - (2.0 * density / magnetic_field) * curvature_phi
        - density * parallel_velocity_gradient
    )
    v_parallel_rhs = -(poisson_v_parallel / (rho_star * magnetic_field))

    if density_source is not None:
        density_rhs = density_rhs + jnp.asarray(density_source, dtype=jnp.float64)
    if v_parallel_source is not None:
        v_parallel_rhs = v_parallel_rhs + jnp.asarray(v_parallel_source, dtype=jnp.float64)

    rhs = Fci2FieldState(
        density=density_rhs,
        v_parallel=v_parallel_rhs,
        density_background=jnp.zeros_like(density_background),
    )
    return Fci2FieldRhsResult(rhs=rhs)
