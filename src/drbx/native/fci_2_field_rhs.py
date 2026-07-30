from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from ..geometry import (
    LocalFciGeometry3D,
)
from .fci_model import FciModelState
from .fci_boundaries import (
    LocalStencil3D,
)
from .fci_operators import (
    local_curvature_op,
    local_grad_parallel_op_direct,
    local_poisson_bracket_op,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci2FieldState(FciModelState):
    density: jax.Array
    v_parallel: jax.Array
    density_background: jax.Array


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


def compute_local_2field_rhs(
    state: Fci2FieldState,
    *,
    geometry: LocalFciGeometry3D,
    stencil_builder: Callable[
        [jax.Array, LocalFciGeometry3D],
        LocalStencil3D,
    ],
    parameters: Fci2FieldRhsParameters = Fci2FieldRhsParameters(),
    curvature_coefficients: jax.Array,
    density_source: jax.Array | None = None,
    v_parallel_source: jax.Array | None = None,
) -> Fci2FieldRhsResult:
    """Assemble the local owned-cell two-field RHS.

    This is the only two-field RHS entry point intended for ``shard_map``.
    It accepts a :class:`LocalFciGeometry3D`, invokes only ``local_*``
    operators, and deliberately performs no host timing or
    ``block_until_ready`` calls.  Halo preparation belongs to the injected
    ``stencil_builder``.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "compute_local_2field_rhs requires LocalFciGeometry3D, got "
            f"{type(geometry).__name__}"
        )

    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    magnetic_field = jnp.maximum(
        jnp.asarray(geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
        1.0e-30,
    )
    density = jnp.asarray(state.density, dtype=jnp.float64)
    v_parallel = jnp.asarray(state.v_parallel, dtype=jnp.float64)
    density_background = jnp.asarray(state.density_background, dtype=jnp.float64)
    phi = jnp.log(jnp.maximum(density, 1.0e-30) / jnp.maximum(density_background, 1.0e-30))

    density_stencil = stencil_builder(density, geometry)
    phi_stencil = stencil_builder(phi, geometry)
    v_parallel_stencil = stencil_builder(v_parallel, geometry)

    poisson_density = local_poisson_bracket_op(phi_stencil, density_stencil, geometry)
    curvature_density = local_curvature_op(
        density_stencil, geometry, curvature_coefficients=curvature_coefficients
    )
    curvature_phi = local_curvature_op(
        phi_stencil, geometry, curvature_coefficients=curvature_coefficients
    )
    parallel_velocity_gradient = local_grad_parallel_op_direct(v_parallel_stencil, geometry)
    poisson_v_parallel = local_poisson_bracket_op(phi_stencil, v_parallel_stencil, geometry)

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

    return Fci2FieldRhsResult(
        rhs=Fci2FieldState(
            density=jnp.asarray(density_rhs, dtype=jnp.float64),
            v_parallel=jnp.asarray(v_parallel_rhs, dtype=jnp.float64),
            density_background=jnp.zeros_like(density_background),
        )
    )
