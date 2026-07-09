from __future__ import annotations

import time as time_module
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..geometry import (
    FciGeometry3D,
    build_local_stencil_from_field,
    LocalStencilBuilder,
)
from .fci_model import FciModelState
from .fci_boundaries import BoundaryFaceBC3D, CutWallBC3D, CutWallGeometry3D
from .fci_operators import curvature_op, grad_parallel_op_direct, poisson_bracket_op


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


def compute_2field_rhs(
    state: Fci2FieldState,
    *,
    geometry: FciGeometry3D,
    stencil_builder: LocalStencilBuilder = build_local_stencil_from_field,
    parameters: Fci2FieldRhsParameters = Fci2FieldRhsParameters(),
    curvature_coefficients: jax.Array,
    density_face_bc: BoundaryFaceBC3D,
    phi_face_bc: BoundaryFaceBC3D,
    v_parallel_face_bc: BoundaryFaceBC3D,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    density_cut_wall_geometry: CutWallGeometry3D | None = None,
    density_cut_wall_bc: CutWallBC3D | None = None,
    phi_cut_wall_geometry: CutWallGeometry3D | None = None,
    phi_cut_wall_bc: CutWallBC3D | None = None,
    v_parallel_cut_wall_geometry: CutWallGeometry3D | None = None,
    v_parallel_cut_wall_bc: CutWallBC3D | None = None,
    density_source: jax.Array | None = None,
    v_parallel_source: jax.Array | None = None,
) -> tuple[Fci2FieldRhsResult, jnp.ndarray]:
    """Assemble the reduced two-field FCI RHS with Neumann stencil reconstruction."""

    rho_star = jnp.asarray(parameters.rho_star, dtype=jnp.float64)
    magnetic_field = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64), 1.0e-30)

    density = jnp.asarray(state.density, dtype=jnp.float64)
    v_parallel = jnp.asarray(state.v_parallel, dtype=jnp.float64)
    density_background = jnp.asarray(state.density_background, dtype=jnp.float64)
    phi = jnp.log(jnp.maximum(density, 1.0e-30) / jnp.maximum(density_background, 1.0e-30))

    stencil_start = time_module.perf_counter()
    density_stencil = stencil_builder(
        density,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=density_face_bc,
        cut_wall_geometry=density_cut_wall_geometry,
        cut_wall_bc=density_cut_wall_bc,
    )
    phi_stencil = stencil_builder(
        phi,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=phi_face_bc,
        cut_wall_geometry=phi_cut_wall_geometry,
        cut_wall_bc=phi_cut_wall_bc,
    )
    v_parallel_stencil = stencil_builder(
        v_parallel,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=v_parallel_face_bc,
        cut_wall_geometry=v_parallel_cut_wall_geometry,
        cut_wall_bc=v_parallel_cut_wall_bc,
    )
    jax.block_until_ready(density_stencil.x.center)
    jax.block_until_ready(phi_stencil.x.center)
    jax.block_until_ready(v_parallel_stencil.x.center)
    stencil_time = time_module.perf_counter() - stencil_start

    operator_start = time_module.perf_counter()
    poisson_density = poisson_bracket_op(phi_stencil, density_stencil, geometry)
    curvature_density = curvature_op(density_stencil, geometry, curvature_coefficients=curvature_coefficients)
    curvature_phi = curvature_op(phi_stencil, geometry, curvature_coefficients=curvature_coefficients)
    parallel_velocity_gradient = grad_parallel_op_direct(v_parallel_stencil, geometry)
    poisson_v_parallel = poisson_bracket_op(phi_stencil, v_parallel_stencil, geometry)

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
    rhs_density = jnp.asarray(density_rhs, dtype=jnp.float64)
    rhs_v_parallel = jnp.asarray(v_parallel_rhs, dtype=jnp.float64)
    jax.block_until_ready(rhs_density)
    jax.block_until_ready(rhs_v_parallel)
    operator_time = time_module.perf_counter() - operator_start

    rhs = Fci2FieldState(
        density=rhs_density,
        v_parallel=rhs_v_parallel,
        density_background=jnp.zeros_like(density_background),
    )
    return Fci2FieldRhsResult(rhs=rhs), jnp.asarray([stencil_time, operator_time], dtype=jnp.float64)
