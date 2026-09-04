from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from itertools import permutations
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from solvax.precond import (
    additive_tridiagonal_line_preconditioner as solvax_line_preconditioner,
    jacobi as solvax_jacobi,
)

_pytree_base = jax.tree_util.register_pytree_node_class

from ..geometry import (
    HaloLayout3D,
    LocalBFieldGeometry,
    LocalAggregateCellGeometry3D,
    LocalControlVolumeCellGeometry3D,
    LocalCellVolumeGeometry3D,
    LocalDomain3D,
    LocalFciDirectionMap,
    LocalFciGeometry3D,
    FCI_DEP_CUT_WALL,
    FCI_DEP_INVALID,
    LocalCurvatureFaceCoefficients3D,
    LocalFciMaps3D,
    LocalFciStencilBuilder,
    LocalRegularFaceGeometry3D,
    LocalStencilBuilder,
    LocalConservativeStencilBuilder,
    build_local_conservative_stencil_from_field,
    build_local_direct_stencil_one_sided_physical_from_halo,
    build_local_fci_stencil_from_field,
    build_local_stencil_from_field,
)
from ..geometry.fci_geometry import (
    StencilBuilderContext,
    _first_derivative_3d,
)
from .fci_halo import (
    HaloExchange3D,
    accumulate_halo_contributions_to_owned,
    LocalHaloClosure3D,
    PhysicalGhostCellFiller3D,
    TopologyHaloFiller3D,
)
from .fci_gmres import (
    SolvaxGmresConfig,
    SolvaxGmresInfo,
    _spmd_remove_weighted_mean,
    solvax_gmres_solve,
)
from .fci_model import (
    inject_owned_field_to_halo,
    inject_owned_vector_field_to_halo,
)
from .characteristic_wall_residual import (
    solve_incoming_characteristic_state,
)
from .fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BC_NONE,
    BC_NORMALFLUX,
    BC_NOFLUX,
    LocalBoundaryFaceBC3D,
    LocalBoundaryFaceTrace3D,
    LocalControlVolumeFluxStencil3D,
    LocalCellGradient3D,
    LocalControlVolumeBoundaryBC3D,
    LocalControlVolumeFieldClosure3D,
    LocalControlVolumeFaceRows3D,
    LocalMomentFittedFaceRows3D,
    LocalControlVolumePolynomial3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalMomentReconstruction3D,
    LocalRegularBoundaryMomentClosure3D,
    LocalCutWallBC3D,
    LocalCutWallGeometry3D,
    LocalRegularFaceContributionRows3D,
    FaceFluxStencil3D,
    CoordinateFaceValues3D,
    ConservativeStencil3D,
    LocalStencil1D,
    LocalStencil3D,
    CV_FACE_CUT_WALL,
    CV_FACE_PHYSICAL_BOUNDARY,
    CV_RECONSTRUCTION_EQUATION_CELL,
    CV_RECONSTRUCTION_EQUATION_DIRICHLET,
    CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
)
from .fci_curvature_production_flux import (
    curvature_face_linearized_fluctuations,
    curvature_strict_principal_matrix,
    reconstruct_third_order_face_states,
)


# =============================================================================
# Parallel-gradient operators
# =============================================================================

def _take_stencil_finite_difference(stencil: LocalStencil1D) -> jnp.ndarray:
    """Apply a reconstructed 1D derivative stencil.

    The stencil arrays may represent either global/reference cells or local
    owned cells. The output has the same shape as the stencil.
    """

    if stencil.center.ndim != 3:
        raise ValueError(
            f"stencil center must be 3D, got shape {stencil.center.shape}"
        )

    minus = jnp.asarray(stencil.minus, dtype=jnp.float64)
    center = jnp.asarray(stencil.center, dtype=jnp.float64)
    plus = jnp.asarray(stencil.plus, dtype=jnp.float64)

    c_minus = jnp.asarray(stencil.derivative_minus_weight, dtype=jnp.float64)
    c_center = jnp.asarray(stencil.derivative_center_weight, dtype=jnp.float64)
    c_plus = jnp.asarray(stencil.derivative_plus_weight, dtype=jnp.float64)

    return c_minus * minus + c_center * center + c_plus * plus


def _active_cell_mask_owned(geometry: LocalFciGeometry3D) -> jnp.ndarray:
    """Return the owned active-cell mask for local/SPMD operator outputs."""

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "_active_cell_mask_owned requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    return jnp.asarray(geometry.active_cell_mask_owned, dtype=bool)


def _mask_inactive_owned(
    values: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    inactive_value: float | jnp.ndarray = 0.0,
    *,
    active_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Mask inactive owned cells in scalar or component-valued owned arrays."""

    array = jnp.asarray(values)
    mask = (
        _active_cell_mask_owned(geometry)
        if active_mask is None
        else jnp.asarray(active_mask, dtype=bool)
    )
    if mask.shape != geometry.owned_shape:
        raise ValueError(
            "active_mask must have shape "
            f"{geometry.owned_shape}, got {mask.shape}"
        )
    if array.shape[:3] != geometry.owned_shape:
        raise ValueError(
            "values must begin with geometry.owned_shape "
            f"{geometry.owned_shape}, got {array.shape}"
        )
    for _ in range(array.ndim - 3):
        mask = mask[..., None]
    return jnp.where(mask, array, jnp.asarray(inactive_value, dtype=array.dtype))


def _solver_active_mask(
    geometry: LocalFciGeometry3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None,
) -> jnp.ndarray:
    """Return the algebraic unknown mask for the perpendicular inverse.

    A control-volume solve has one degree of freedom per active aggregate
    owner.  Merged storage cells are representation aliases and must not be
    exposed to GMRES as additional unknowns.
    """

    if control_volume_geometry is None:
        return _active_cell_mask_owned(geometry)
    mask = jnp.asarray(
        control_volume_geometry.cells.is_active_owner,
        dtype=bool,
    )
    if mask.shape != geometry.owned_shape:
        raise ValueError(
            "control-volume owner mask must have shape "
            f"{geometry.owned_shape}, got {mask.shape}"
        )
    return mask


def _solver_volume_weights(
    geometry: LocalFciGeometry3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None,
) -> jnp.ndarray | None:
    """Return explicit mean/norm weights for an optional CV solve."""

    if control_volume_geometry is None:
        return None
    weights = jnp.asarray(
        control_volume_geometry.cells.aggregate_volume,
        dtype=jnp.float64,
    )
    if weights.shape != geometry.owned_shape:
        raise ValueError(
            "control-volume aggregate_volume must have shape "
            f"{geometry.owned_shape}, got {weights.shape}"
        )
    return weights


def _mask_state_inactive_owned(
    state,
    geometry: LocalFciGeometry3D,
    inactive_value: float | jnp.ndarray = 0.0,
):
    """Mask inactive owned cells in each owned-array leaf of a local RHS state."""

    def _mask_leaf(leaf):
        array = jnp.asarray(leaf)
        if array.ndim >= 3 and array.shape[:3] == geometry.owned_shape:
            return _mask_inactive_owned(array, geometry, inactive_value)
        return leaf

    return jax.tree_util.tree_map(_mask_leaf, state)


def local_grad_parallel_op_fci(
    stencil: LocalStencil1D,
    geometry: LocalFciGeometry3D,
) -> jnp.ndarray:
    """Local/domain-decomposed centered FCI parallel gradient.

    Computes ``grad_parallel(f)`` from a field-line stencil on owned cells.
    The stencil builder is responsible for using the prepared halo field,
    topology information, and cut-wall/boundary information to construct the
    stencil.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_grad_parallel_op_fci requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )

    if stencil.shape != geometry.owned_shape:
        raise ValueError(
            f"stencil must have shape {geometry.owned_shape}, "
            f"got {stencil.shape}"
        )

    return _mask_inactive_owned(_take_stencil_finite_difference(stencil), geometry)


def _normalize_fci_endpoint_values(
    values: jnp.ndarray | None,
    *,
    dtype: jnp.dtype,
    name: str,
) -> jnp.ndarray:
    """Normalize a direction's explicit cut-wall endpoint payload.

    The FCI dependency tables use a compact value-slot index.  An empty
    payload is retained as a valid homogeneous fallback for compatibility
    with the prototype builder; callers that have a physical wall closure
    should pass the corresponding direction's payload explicitly.
    """

    if values is None:
        return jnp.zeros((0,), dtype=dtype)
    values = jnp.asarray(values, dtype=dtype)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {values.shape}")
    return values


def _sample_fci_endpoint_rows(
    field_halo: jnp.ndarray,
    direction: LocalFciDirectionMap,
    *,
    remote_values: jnp.ndarray | None,
    cut_wall_values: jnp.ndarray,
    n_owned: int,
) -> jnp.ndarray:
    """Evaluate one mapped endpoint, including local/remote wall rows.

    This mirrors ``LocalFciStencilBuilder`` locally so the operator family can
    accept separate forward and backward wall payloads without changing the
    geometry module's public builder contract.
    """

    table = direction.local
    nx, ny, nz = field_halo.shape
    source_i = jnp.clip(table.source_i, 0, nx - 1)
    source_j = jnp.clip(table.source_j, 0, ny - 1)
    source_k = jnp.clip(table.source_k, 0, nz - 1)
    field_samples = field_halo[source_i, source_j, source_k]
    if int(cut_wall_values.size) == 0:
        wall_samples = jnp.zeros(table.value_slot.shape, dtype=field_halo.dtype)
    else:
        wall_samples = cut_wall_values[
            jnp.clip(table.value_slot, 0, int(cut_wall_values.size) - 1)
        ]
    samples = jnp.where(
        table.dependency_kind == FCI_DEP_CUT_WALL,
        wall_samples,
        field_samples,
    )
    active = table.active & (table.dependency_kind != FCI_DEP_INVALID)
    target = jnp.clip(table.target_flat, 0, n_owned - 1)
    endpoint = jnp.zeros((n_owned,), dtype=field_halo.dtype)
    endpoint = endpoint.at[target].add(jnp.where(active, table.weight * samples, 0.0))

    remote = direction.remote
    if remote is not None and remote.max_entries:
        expected = (remote.max_receive_values,)
        if remote.max_receive_values == 0:
            remote_samples = jnp.zeros(remote.receive_slot.shape, dtype=field_halo.dtype)
        else:
            if remote_values is None:
                raise ValueError("remote endpoint values are required for remote FCI rows")
            remote_values = jnp.asarray(remote_values, dtype=field_halo.dtype)
            if remote_values.shape != expected:
                raise ValueError(
                    "remote endpoint values must have shape "
                    f"{expected}, got {remote_values.shape}"
                )
            receive_slot = jnp.clip(remote.receive_slot, 0, remote.max_receive_values - 1)
            remote_samples = remote_values[receive_slot]
        remote_target = jnp.clip(remote.target_flat, 0, n_owned - 1)
        endpoint = endpoint.at[remote_target].add(
            jnp.where(remote.active, remote.weight * remote_samples, 0.0)
        )

    center = field_halo[direction.layout.owned_slices_cell].reshape((n_owned,))
    return jnp.where(direction.target_valid.reshape((n_owned,)), endpoint, center)


def _build_mapped_stencil_with_directional_walls(
    field_halo_full: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    *,
    forward_remote_values: jnp.ndarray | None,
    backward_remote_values: jnp.ndarray | None,
    forward_cut_wall_values: jnp.ndarray | None,
    backward_cut_wall_values: jnp.ndarray | None,
) -> LocalStencil1D:
    """Build a mapped stencil with independent forward/backward wall values."""

    field_halo_full = jnp.asarray(field_halo_full, dtype=jnp.float64)
    n_owned = int(np.prod(geometry.owned_shape))
    forward_values = _normalize_fci_endpoint_values(
        forward_cut_wall_values,
        dtype=field_halo_full.dtype,
        name="forward_cut_wall_values",
    )
    backward_values = _normalize_fci_endpoint_values(
        backward_cut_wall_values,
        dtype=field_halo_full.dtype,
        name="backward_cut_wall_values",
    )
    center = field_halo_full[geometry.layout.owned_slices_cell]
    forward = _sample_fci_endpoint_rows(
        field_halo_full,
        geometry.maps.forward,
        remote_values=forward_remote_values,
        cut_wall_values=forward_values,
        n_owned=n_owned,
    ).reshape(geometry.owned_shape)
    backward = _sample_fci_endpoint_rows(
        field_halo_full,
        geometry.maps.backward,
        remote_values=backward_remote_values,
        cut_wall_values=backward_values,
        n_owned=n_owned,
    ).reshape(geometry.owned_shape)
    if geometry.maps.forward.connection_length is None:
        raise ValueError("geometry.maps.forward.connection_length is required")
    if geometry.maps.backward.connection_length is None:
        raise ValueError("geometry.maps.backward.connection_length is required")
    dx_plus = jnp.maximum(
        jnp.asarray(geometry.maps.forward.connection_length, dtype=field_halo_full.dtype),
        1.0e-30,
    )
    dx_min = jnp.maximum(
        jnp.asarray(geometry.maps.backward.connection_length, dtype=field_halo_full.dtype),
        1.0e-30,
    )
    valid = geometry.maps.forward.target_valid & geometry.maps.backward.target_valid
    return LocalStencil1D(
        center=center,
        minus=jnp.where(valid, backward, center),
        plus=jnp.where(valid, forward, center),
        dx_min=jnp.where(valid, dx_min, 1.0),
        dx_plus=jnp.where(valid, dx_plus, 1.0),
    )


def _build_mapped_stencil(
    field_halo_full: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    context: StencilBuilderContext,
    *,
    fci_stencil_builder: LocalFciStencilBuilder,
    forward_remote_values: jnp.ndarray | None,
    backward_remote_values: jnp.ndarray | None,
    cut_wall_values: jnp.ndarray | None,
    forward_cut_wall_values: jnp.ndarray | None,
    backward_cut_wall_values: jnp.ndarray | None,
) -> LocalStencil1D:
    if forward_cut_wall_values is None and backward_cut_wall_values is None:
        return fci_stencil_builder(
            field_halo_full,
            geometry,
            context,
            forward_remote_values=forward_remote_values,
            backward_remote_values=backward_remote_values,
            cut_wall_values=cut_wall_values,
        )
    if fci_stencil_builder is not build_local_fci_stencil_from_field:
        raise ValueError(
            "direction-aware wall endpoint values require the built-in "
            "LocalFciStencilBuilder"
        )
    shared = _normalize_fci_endpoint_values(
        cut_wall_values,
        dtype=jnp.float64,
        name="cut_wall_values",
    )
    forward = shared if forward_cut_wall_values is None else forward_cut_wall_values
    backward = shared if backward_cut_wall_values is None else backward_cut_wall_values
    return _build_mapped_stencil_with_directional_walls(
        field_halo_full,
        geometry,
        forward_remote_values=forward_remote_values,
        backward_remote_values=backward_remote_values,
        forward_cut_wall_values=forward,
        backward_cut_wall_values=backward,
    )


# =============================================================================
def local_parallel_q_flux_div_fci_op(
    q_halo_full: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    *,
    context: StencilBuilderContext,
    fci_stencil_builder: LocalFciStencilBuilder = build_local_fci_stencil_from_field,
    forward_remote_q_values: jnp.ndarray | None = None,
    backward_remote_q_values: jnp.ndarray | None = None,
    cut_wall_q_values: jnp.ndarray | None = None,
    forward_cut_wall_q_values: jnp.ndarray | None = None,
    backward_cut_wall_q_values: jnp.ndarray | None = None,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Low-level mapped ``div(F b)`` for a prepared ``q=F/B`` halo.

    ``q_halo_full`` must already include the physical ghost/leg values filled
    by the EB boundary preparation.  This function never divides by the
    halo-shaped ``Bmag`` field.  Endpoint wall payloads are values of this
    same prepared ``q`` quantity and may be supplied independently for the
    forward and backward traces.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_parallel_q_flux_div_fci_op requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    if not isinstance(context, StencilBuilderContext):
        raise TypeError(
            "context must be a StencilBuilderContext, "
            f"got {type(context).__name__}"
        )
    q_halo_full = jnp.asarray(q_halo_full, dtype=jnp.float64)
    if q_halo_full.shape != geometry.halo_shape:
        raise ValueError(
            "q_halo_full must match geometry.halo_shape; "
            f"got {q_halo_full.shape}, expected {geometry.halo_shape}"
        )
    q_stencil = _build_mapped_stencil(
        q_halo_full,
        geometry,
        context,
        fci_stencil_builder=fci_stencil_builder,
        forward_remote_values=forward_remote_q_values,
        backward_remote_values=backward_remote_q_values,
        cut_wall_values=cut_wall_q_values,
        forward_cut_wall_values=forward_cut_wall_q_values,
        backward_cut_wall_values=backward_cut_wall_q_values,
    )
    grad_q = local_grad_parallel_op_fci(q_stencil, geometry)
    Bmag_owned = jnp.maximum(
        jnp.asarray(geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
        float(b_floor),
    )
    return _mask_inactive_owned(Bmag_owned * grad_q, geometry)


def local_parallel_div_b_fci_from_q_op(
    inverse_b_halo_full: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    *,
    context: StencilBuilderContext,
    **kwargs,
) -> jnp.ndarray:
    """Low-level mapped ``div(b)`` from a prepared ``q=1/B`` halo."""

    return local_parallel_q_flux_div_fci_op(
        inverse_b_halo_full,
        geometry,
        context=context,
        **kwargs,
    )


def local_conservative_parallel_flux_div_op(
    field_halo_full: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    *,
    context: StencilBuilderContext,
    fci_stencil_builder: LocalFciStencilBuilder = build_local_fci_stencil_from_field,
    forward_remote_q_values: jnp.ndarray | None = None,
    backward_remote_q_values: jnp.ndarray | None = None,
    cut_wall_q_values: jnp.ndarray | None = None,
    forward_cut_wall_q_values: jnp.ndarray | None = None,
    backward_cut_wall_q_values: jnp.ndarray | None = None,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Local/domain-decomposed FCI conservative parallel flux divergence.

    Computes ``div(F b)`` through the continuum identity

        ``div(F b) = B * grad_parallel(F / B)``

    using the local FCI interpolation stencil for ``F / B``. Any remote or
    cut-wall endpoint values passed to this function must already be values of
    ``F / B`` at those endpoints. This is compatible with the continuum flux
    identity, but the current interpolation rows do not guarantee exact
    globally conservative cancellation between neighboring targets.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_conservative_parallel_flux_div_op requires "
            f"LocalFciGeometry3D, got {type(geometry).__name__}"
        )
    if not isinstance(context, StencilBuilderContext):
        raise TypeError(
            "context must be a StencilBuilderContext, "
            f"got {type(context).__name__}"
        )
    if not isinstance(fci_stencil_builder, LocalFciStencilBuilder):
        raise TypeError(
            "fci_stencil_builder must be a LocalFciStencilBuilder, "
            f"got {type(fci_stencil_builder).__name__}"
        )
    if context.layout != geometry.layout:
        raise ValueError("geometry and context must share the same HaloLayout3D")

    field_halo_full = jnp.asarray(field_halo_full, dtype=jnp.float64)
    if field_halo_full.shape != geometry.halo_shape:
        raise ValueError(
            "field_halo_full must match geometry.halo_shape; "
            f"got {field_halo_full.shape}, expected {geometry.halo_shape}"
        )

    Bmag_halo = jnp.maximum(
        jnp.asarray(geometry.cell_bfield.Bmag_halo, dtype=jnp.float64),
        float(b_floor),
    )
    q_halo = field_halo_full / Bmag_halo
    return local_parallel_q_flux_div_fci_op(
        q_halo,
        geometry,
        context=context,
        fci_stencil_builder=fci_stencil_builder,
        forward_remote_q_values=forward_remote_q_values,
        backward_remote_q_values=backward_remote_q_values,
        cut_wall_q_values=cut_wall_q_values,
        forward_cut_wall_q_values=forward_cut_wall_q_values,
        backward_cut_wall_q_values=backward_cut_wall_q_values,
        b_floor=b_floor,
    )


def local_parallel_flux_div_fci_op(*args, **kwargs) -> jnp.ndarray:
    """Named FCI-family alias for the mapped ``div(F b)`` prototype.

    The implementation uses ``B * D(F/B)`` with the unequal-leg mapped
    endpoint stencil.  It is compatible with the continuum flux identity but
    is not an exact globally conservative transpose/finite-volume operator:
    independently interpolated FCI endpoint rows need not be shared by two
    neighboring cells.
    """

    return local_conservative_parallel_flux_div_op(*args, **kwargs)


def local_parallel_div_b_fci_op(
    geometry: LocalFciGeometry3D,
    *,
    context: StencilBuilderContext,
    inverse_b_halo_full: jnp.ndarray | None = None,
    fci_stencil_builder: LocalFciStencilBuilder = build_local_fci_stencil_from_field,
    forward_remote_q_values: jnp.ndarray | None = None,
    backward_remote_q_values: jnp.ndarray | None = None,
    cut_wall_q_values: jnp.ndarray | None = None,
    forward_cut_wall_q_values: jnp.ndarray | None = None,
    backward_cut_wall_q_values: jnp.ndarray | None = None,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return mapped ``div(b)`` for reuse by compatible mapped gradients.

    This is geometry-only for a fixed ``geometry`` and map/stencil closure, so
    callers should evaluate it once and cache the returned array for all
    compatible gradients in a stage.  The supplied wall values are values of
    ``1/B`` at the corresponding mapped endpoints; direction-specific payloads
    are preferred when the two traces have different closures.  For production
    EB use, pass ``inverse_b_halo_full`` or call
    :func:`local_parallel_div_b_fci_from_q_op`; omitting it retains the legacy
    convenience behavior, which derives ``1/B`` from the geometry halo.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_parallel_div_b_fci_op requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    if inverse_b_halo_full is not None:
        return local_parallel_div_b_fci_from_q_op(
            inverse_b_halo_full,
            geometry,
            context=context,
            fci_stencil_builder=fci_stencil_builder,
            forward_remote_q_values=forward_remote_q_values,
            backward_remote_q_values=backward_remote_q_values,
            cut_wall_q_values=cut_wall_q_values,
            forward_cut_wall_q_values=forward_cut_wall_q_values,
            backward_cut_wall_q_values=backward_cut_wall_q_values,
            b_floor=b_floor,
        )
    unit_field_halo = jnp.ones(geometry.halo_shape, dtype=jnp.float64)
    return local_conservative_parallel_flux_div_op(
        unit_field_halo,
        geometry,
        context=context,
        fci_stencil_builder=fci_stencil_builder,
        forward_remote_q_values=forward_remote_q_values,
        backward_remote_q_values=backward_remote_q_values,
        cut_wall_q_values=cut_wall_q_values,
        forward_cut_wall_q_values=forward_cut_wall_q_values,
        backward_cut_wall_q_values=backward_cut_wall_q_values,
        b_floor=b_floor,
    )


def local_grad_parallel_op_fci_compatible(
    field_halo_full: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    *,
    context: StencilBuilderContext,
    div_b: jnp.ndarray | None = None,
    fci_stencil_builder: LocalFciStencilBuilder = build_local_fci_stencil_from_field,
    forward_remote_q_values: jnp.ndarray | None = None,
    backward_remote_q_values: jnp.ndarray | None = None,
    cut_wall_q_values: jnp.ndarray | None = None,
    forward_cut_wall_q_values: jnp.ndarray | None = None,
    backward_cut_wall_q_values: jnp.ndarray | None = None,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Compute the compatible mapped parallel gradient.

    The definition is

        ``grad_parallel(f) = div(f b) - f * div(b)``.

    Passing ``div_b`` reuses the cacheable geometry-only result from
    :func:`local_parallel_div_b_fci_op`.  To annihilate constants at mapped
    physical endpoints, ``div_b`` must be formed with the same endpoint
    closure as the ``div(f b)`` call.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_grad_parallel_op_fci_compatible requires "
            f"LocalFciGeometry3D, got {type(geometry).__name__}"
        )
    if div_b is None:
        div_b = local_parallel_div_b_fci_op(
            geometry,
            context=context,
            fci_stencil_builder=fci_stencil_builder,
            forward_remote_q_values=forward_remote_q_values,
            backward_remote_q_values=backward_remote_q_values,
            cut_wall_q_values=cut_wall_q_values,
            forward_cut_wall_q_values=forward_cut_wall_q_values,
            backward_cut_wall_q_values=backward_cut_wall_q_values,
            b_floor=b_floor,
        )
    div_b = jnp.asarray(div_b, dtype=jnp.float64)
    if div_b.shape != geometry.owned_shape:
        raise ValueError(
            f"div_b must have shape {geometry.owned_shape}, got {div_b.shape}"
        )
    div_fb = local_conservative_parallel_flux_div_op(
        field_halo_full,
        geometry,
        context=context,
        fci_stencil_builder=fci_stencil_builder,
        forward_remote_q_values=forward_remote_q_values,
        backward_remote_q_values=backward_remote_q_values,
        cut_wall_q_values=cut_wall_q_values,
        forward_cut_wall_q_values=forward_cut_wall_q_values,
        backward_cut_wall_q_values=backward_cut_wall_q_values,
        b_floor=b_floor,
    )
    field_owned = jnp.asarray(field_halo_full, dtype=jnp.float64)[
        geometry.layout.owned_slices_cell
    ]
    return _mask_inactive_owned(div_fb - field_owned * div_b, geometry)


def local_grad_parallel_op_fci_compatible_from_q(
    q_halo_full: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    *,
    context: StencilBuilderContext,
    field_owned: jnp.ndarray | None = None,
    field_halo_full: jnp.ndarray | None = None,
    inverse_b_halo_full: jnp.ndarray | None = None,
    div_b: jnp.ndarray | None = None,
    fci_stencil_builder: LocalFciStencilBuilder = build_local_fci_stencil_from_field,
    forward_remote_q_values: jnp.ndarray | None = None,
    backward_remote_q_values: jnp.ndarray | None = None,
    cut_wall_q_values: jnp.ndarray | None = None,
    forward_cut_wall_q_values: jnp.ndarray | None = None,
    backward_cut_wall_q_values: jnp.ndarray | None = None,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Compatible mapped gradient from a prepared ``q=f/B`` halo.

    ``field_owned`` is the prepared physical field on owned cells.  A prepared
    ``field_halo_full`` may be supplied instead; only its owned slice is read.
    If both are omitted, the field is reconstructed only on owned cells as
    ``B_owned*q_owned``; no physical ``B`` ghost value is read.  ``div_b`` should be the cached
    result of :func:`local_parallel_div_b_fci_from_q_op` using the matching
    prepared ``q=1/B`` endpoint closure.
    """

    q_halo_full = jnp.asarray(q_halo_full, dtype=jnp.float64)
    if q_halo_full.shape != geometry.halo_shape:
        raise ValueError(
            "q_halo_full must match geometry.halo_shape; "
            f"got {q_halo_full.shape}, expected {geometry.halo_shape}"
        )
    if div_b is None and inverse_b_halo_full is None:
        raise ValueError(
            "provide div_b or a prepared inverse_b_halo_full; the low-level "
            "q API does not read physical Bmag ghost cells"
        )
    if div_b is None:
        div_b = local_parallel_div_b_fci_from_q_op(
            inverse_b_halo_full,
            geometry,
            context=context,
            fci_stencil_builder=fci_stencil_builder,
            forward_remote_q_values=forward_remote_q_values,
            backward_remote_q_values=backward_remote_q_values,
            cut_wall_q_values=cut_wall_q_values,
            forward_cut_wall_q_values=forward_cut_wall_q_values,
            backward_cut_wall_q_values=backward_cut_wall_q_values,
            b_floor=b_floor,
        )
    div_b = jnp.asarray(div_b, dtype=jnp.float64)
    if div_b.shape != geometry.owned_shape:
        raise ValueError(
            f"div_b must have shape {geometry.owned_shape}, got {div_b.shape}"
        )
    if field_owned is None and field_halo_full is not None:
        field_halo_full = jnp.asarray(field_halo_full, dtype=jnp.float64)
        if field_halo_full.shape != geometry.halo_shape:
            raise ValueError(
                "field_halo_full must match geometry.halo_shape; "
                f"got {field_halo_full.shape}, expected {geometry.halo_shape}"
            )
        field_owned = field_halo_full[geometry.layout.owned_slices_cell]
    if field_owned is None:
        Bmag_owned = jnp.maximum(
            jnp.asarray(geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
            float(b_floor),
        )
        field_owned = q_halo_full[geometry.layout.owned_slices_cell] * Bmag_owned
    else:
        field_owned = jnp.asarray(field_owned, dtype=jnp.float64)
        if field_owned.shape != geometry.owned_shape:
            raise ValueError(
                f"field_owned must have shape {geometry.owned_shape}, "
                f"got {field_owned.shape}"
            )
    div_fb = local_parallel_q_flux_div_fci_op(
        q_halo_full,
        geometry,
        context=context,
        fci_stencil_builder=fci_stencil_builder,
        forward_remote_q_values=forward_remote_q_values,
        backward_remote_q_values=backward_remote_q_values,
        cut_wall_q_values=cut_wall_q_values,
        forward_cut_wall_q_values=forward_cut_wall_q_values,
        backward_cut_wall_q_values=backward_cut_wall_q_values,
        b_floor=b_floor,
    )
    return _mask_inactive_owned(div_fb - field_owned * div_b, geometry)


def local_grad_parallel_op_fci_compatible_from_q_components(
    q_halo_full: jnp.ndarray,
    inverse_b_halo_full: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    *,
    context: StencilBuilderContext,
    field_owned: jnp.ndarray,
    forward_remote_q_values: jnp.ndarray | None = None,
    backward_remote_q_values: jnp.ndarray | None = None,
    forward_remote_inverse_b_values: jnp.ndarray | None = None,
    backward_remote_inverse_b_values: jnp.ndarray | None = None,
    fci_stencil_builder: LocalFciStencilBuilder = build_local_fci_stencil_from_field,
    b_floor: float = 1.0e-30,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split the compatible mapped gradient into its three stencil lanes.

    The returned component order is ``(backward, center, forward)``.  Their
    sum reconstructs :func:`local_grad_parallel_op_fci_compatible_from_q`
    when the latter uses the same prepared ``q=f/B`` and ``q=1/B`` payloads.
    The second return value contains the physical field reconstructed at the
    backward and forward mapped endpoints.  This helper is diagnostic-only;
    it deliberately reuses the production mapped-stencil builder so wall and
    remote endpoint semantics cannot drift from the operator being audited.
    """

    q_halo_full = jnp.asarray(q_halo_full, dtype=jnp.float64)
    inverse_b_halo_full = jnp.asarray(inverse_b_halo_full, dtype=jnp.float64)
    field_owned = jnp.asarray(field_owned, dtype=jnp.float64)
    if q_halo_full.shape != geometry.halo_shape:
        raise ValueError(
            "q_halo_full must match geometry.halo_shape; "
            f"got {q_halo_full.shape}, expected {geometry.halo_shape}"
        )
    if inverse_b_halo_full.shape != geometry.halo_shape:
        raise ValueError(
            "inverse_b_halo_full must match geometry.halo_shape; "
            f"got {inverse_b_halo_full.shape}, expected {geometry.halo_shape}"
        )
    if field_owned.shape != geometry.owned_shape:
        raise ValueError(
            f"field_owned must have shape {geometry.owned_shape}, "
            f"got {field_owned.shape}"
        )

    q_stencil = _build_mapped_stencil(
        q_halo_full,
        geometry,
        context,
        fci_stencil_builder=fci_stencil_builder,
        forward_remote_values=forward_remote_q_values,
        backward_remote_values=backward_remote_q_values,
        cut_wall_values=None,
        forward_cut_wall_values=None,
        backward_cut_wall_values=None,
    )
    inverse_b_stencil = _build_mapped_stencil(
        inverse_b_halo_full,
        geometry,
        context,
        fci_stencil_builder=fci_stencil_builder,
        forward_remote_values=forward_remote_inverse_b_values,
        backward_remote_values=backward_remote_inverse_b_values,
        cut_wall_values=None,
        forward_cut_wall_values=None,
        backward_cut_wall_values=None,
    )
    Bmag_owned = jnp.maximum(
        jnp.asarray(geometry.cell_bfield.Bmag_owned, dtype=jnp.float64),
        float(b_floor),
    )
    q_values = (q_stencil.minus, q_stencil.center, q_stencil.plus)
    inverse_b_values = (
        inverse_b_stencil.minus,
        inverse_b_stencil.center,
        inverse_b_stencil.plus,
    )
    weights = (
        q_stencil.derivative_minus_weight,
        q_stencil.derivative_center_weight,
        q_stencil.derivative_plus_weight,
    )
    components = jnp.stack(
        tuple(
            Bmag_owned
            * jnp.asarray(weight, dtype=jnp.float64)
            * (
                jnp.asarray(q_value, dtype=jnp.float64)
                - field_owned * jnp.asarray(inverse_b_value, dtype=jnp.float64)
            )
            for weight, q_value, inverse_b_value in zip(
                weights, q_values, inverse_b_values, strict=True
            )
        ),
        axis=0,
    )
    components = jax.vmap(lambda value: _mask_inactive_owned(value, geometry))(
        components
    )
    endpoint_values = jnp.stack(
        (
            q_stencil.minus
            / jnp.maximum(inverse_b_stencil.minus, float(b_floor)),
            q_stencil.plus
            / jnp.maximum(inverse_b_stencil.plus, float(b_floor)),
        ),
        axis=0,
    )
    endpoint_values = jax.vmap(
        lambda value: _mask_inactive_owned(value, geometry)
    )(endpoint_values)
    return components, endpoint_values


def _take_stencil_second_derivative(stencil: LocalStencil1D) -> jnp.ndarray:
    """Apply the quadratic exact second derivative for unequal mapped legs."""

    dm = jnp.maximum(jnp.asarray(stencil.dx_min, dtype=jnp.float64), 1.0e-30)
    dp = jnp.maximum(jnp.asarray(stencil.dx_plus, dtype=jnp.float64), 1.0e-30)
    denominator = jnp.maximum(dm + dp, 1.0e-30)
    return (
        2.0 * stencil.minus / (dm * denominator)
        - 2.0 * stencil.center / (dm * dp)
        + 2.0 * stencil.plus / (dp * denominator)
    )


def local_parallel_laplacian_fci_op(
    field_halo_full: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    *,
    context: StencilBuilderContext,
    fci_stencil_builder: LocalFciStencilBuilder = build_local_fci_stencil_from_field,
    forward_remote_values: jnp.ndarray | None = None,
    backward_remote_values: jnp.ndarray | None = None,
    cut_wall_values: jnp.ndarray | None = None,
    forward_cut_wall_values: jnp.ndarray | None = None,
    backward_cut_wall_values: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return the mapped second derivative along the field line.

    The formula is exact for quadratics in the traced arc-length coordinate
    and supports unequal forward/backward connection lengths.  It is a local
    support-operator approximation using the same mapped endpoint stencil as
    the first derivative.  It is *not* an exact globally conservative
    transpose operator: neighboring target cells generally do not share one
    algebraic face flux after interpolation.

    ``forward_cut_wall_values`` and ``backward_cut_wall_values`` are explicit
    field values at cut-wall endpoints.  They are direction-aware; the older
    ``cut_wall_values`` argument remains a shared compatibility fallback.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_parallel_laplacian_fci_op requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    stencil = _build_mapped_stencil(
        jnp.asarray(field_halo_full, dtype=jnp.float64),
        geometry,
        context,
        fci_stencil_builder=fci_stencil_builder,
        forward_remote_values=forward_remote_values,
        backward_remote_values=backward_remote_values,
        cut_wall_values=cut_wall_values,
        forward_cut_wall_values=forward_cut_wall_values,
        backward_cut_wall_values=backward_cut_wall_values,
    )
    return _mask_inactive_owned(_take_stencil_second_derivative(stencil), geometry)


def local_parallel_diffusion_fci_op(
    field_halo_full: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    *,
    context: StencilBuilderContext,
    diffusivity_halo_full: jnp.ndarray | None = None,
    inverse_b_halo_full: jnp.ndarray | None = None,
    b_floor: float = 1.0e-30,
    fci_stencil_builder: LocalFciStencilBuilder = build_local_fci_stencil_from_field,
    forward_remote_values: jnp.ndarray | None = None,
    backward_remote_values: jnp.ndarray | None = None,
    cut_wall_values: jnp.ndarray | None = None,
    forward_cut_wall_values: jnp.ndarray | None = None,
    backward_cut_wall_values: jnp.ndarray | None = None,
    forward_remote_diffusivity_values: jnp.ndarray | None = None,
    backward_remote_diffusivity_values: jnp.ndarray | None = None,
    cut_wall_diffusivity_values: jnp.ndarray | None = None,
    forward_cut_wall_diffusivity_values: jnp.ndarray | None = None,
    backward_cut_wall_diffusivity_values: jnp.ndarray | None = None,
    forward_cut_wall_bmag_values: jnp.ndarray | None = None,
    backward_cut_wall_bmag_values: jnp.ndarray | None = None,
    forward_remote_inverse_b_values: jnp.ndarray | None = None,
    backward_remote_inverse_b_values: jnp.ndarray | None = None,
    forward_cut_wall_inverse_b_values: jnp.ndarray | None = None,
    backward_cut_wall_inverse_b_values: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return a mapped conservative/support-form parallel diffusion.

    With ``F = kappa * grad_parallel(f)``, the local support form is

        ``B * (q_plus - q_minus) / ((d_plus + d_minus)/2)``,
        ``q_side = (kappa_face / B_face) * delta_f / d_side``.

    This is second-order for smooth fields and coefficients on smooth unequal
    legs.  It uses midpoint coefficient and ``B`` averages, and the same
    mapped endpoints as the gradient/divergence family.  It is conservative
    in this local flux-difference sense, but exact global conservation is not
    guaranteed because FCI interpolation does not enforce shared face rows.
    For production EB use, pass a prepared ``inverse_b_halo_full`` (and its
    remote/wall endpoint payloads) so no physical ``B`` ghost is read.  If it
    is omitted, the legacy convenience path derives ``B`` from the geometry
    halo.  Physical wall values for the field, diffusivity, and inverse ``B``
    must be supplied through their direction-aware endpoint arguments when a
    trace terminates at a wall.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_parallel_diffusion_fci_op requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    field_halo_full = jnp.asarray(field_halo_full, dtype=jnp.float64)
    if field_halo_full.shape != geometry.halo_shape:
        raise ValueError(
            "field_halo_full must match geometry.halo_shape; "
            f"got {field_halo_full.shape}, expected {geometry.halo_shape}"
        )
    if diffusivity_halo_full is None:
        diffusivity_halo_full = jnp.ones_like(field_halo_full)
    else:
        diffusivity_halo_full = jnp.asarray(diffusivity_halo_full, dtype=jnp.float64)
        if diffusivity_halo_full.shape != geometry.halo_shape:
            raise ValueError(
                "diffusivity_halo_full must match geometry.halo_shape; "
                f"got {diffusivity_halo_full.shape}, expected {geometry.halo_shape}"
            )

    field_stencil = _build_mapped_stencil(
        field_halo_full,
        geometry,
        context,
        fci_stencil_builder=fci_stencil_builder,
        forward_remote_values=forward_remote_values,
        backward_remote_values=backward_remote_values,
        cut_wall_values=cut_wall_values,
        forward_cut_wall_values=forward_cut_wall_values,
        backward_cut_wall_values=backward_cut_wall_values,
    )
    diffusivity_stencil = _build_mapped_stencil(
        diffusivity_halo_full,
        geometry,
        context,
        fci_stencil_builder=fci_stencil_builder,
        forward_remote_values=forward_remote_diffusivity_values,
        backward_remote_values=backward_remote_diffusivity_values,
        cut_wall_values=cut_wall_diffusivity_values,
        forward_cut_wall_values=forward_cut_wall_diffusivity_values,
        backward_cut_wall_values=backward_cut_wall_diffusivity_values,
    )
    if inverse_b_halo_full is None:
        bmag_halo = jnp.maximum(
            jnp.asarray(geometry.cell_bfield.Bmag_halo, dtype=jnp.float64),
            float(b_floor),
        )
        bmag_stencil = _build_mapped_stencil(
            bmag_halo,
            geometry,
            context,
            fci_stencil_builder=fci_stencil_builder,
            forward_remote_values=None,
            backward_remote_values=None,
            cut_wall_values=None,
            forward_cut_wall_values=forward_cut_wall_bmag_values,
            backward_cut_wall_values=backward_cut_wall_bmag_values,
        )
        b_center = jnp.maximum(bmag_stencil.center, float(b_floor))
        b_minus = jnp.maximum(
            0.5 * (b_center + bmag_stencil.minus), float(b_floor)
        )
        b_plus = jnp.maximum(
            0.5 * (b_center + bmag_stencil.plus), float(b_floor)
        )
    else:
        inverse_b_halo_full = jnp.asarray(inverse_b_halo_full, dtype=jnp.float64)
        if inverse_b_halo_full.shape != geometry.halo_shape:
            raise ValueError(
                "inverse_b_halo_full must match geometry.halo_shape; "
                f"got {inverse_b_halo_full.shape}, expected {geometry.halo_shape}"
            )
        inverse_b_stencil = _build_mapped_stencil(
            inverse_b_halo_full,
            geometry,
            context,
            fci_stencil_builder=fci_stencil_builder,
            forward_remote_values=forward_remote_inverse_b_values,
            backward_remote_values=backward_remote_inverse_b_values,
            cut_wall_values=None,
            forward_cut_wall_values=forward_cut_wall_inverse_b_values,
            backward_cut_wall_values=backward_cut_wall_inverse_b_values,
        )
        b_center = 1.0 / jnp.maximum(inverse_b_stencil.center, float(b_floor))
        b_minus = 1.0 / jnp.maximum(
            0.5 * (inverse_b_stencil.center + inverse_b_stencil.minus),
            float(b_floor),
        )
        b_plus = 1.0 / jnp.maximum(
            0.5 * (inverse_b_stencil.center + inverse_b_stencil.plus),
            float(b_floor),
        )
    dm = jnp.maximum(field_stencil.dx_min, 1.0e-30)
    dp = jnp.maximum(field_stencil.dx_plus, 1.0e-30)
    width = jnp.maximum(dm + dp, 1.0e-30)
    k_center = diffusivity_stencil.center
    k_minus = 0.5 * (k_center + diffusivity_stencil.minus)
    k_plus = 0.5 * (k_center + diffusivity_stencil.plus)
    q_minus = k_minus * (field_stencil.center - field_stencil.minus) / (dm * b_minus)
    q_plus = k_plus * (field_stencil.plus - field_stencil.center) / (dp * b_plus)
    result = b_center * 2.0 * (q_plus - q_minus) / width
    return _mask_inactive_owned(result, geometry)


def local_grad_parallel_op_direct(
    stencil: LocalStencil3D,
    geometry: LocalFciGeometry3D,
    *,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Local/domain-decomposed direct finite-difference parallel gradient.

    Computes ``grad_parallel(f) = b^i partial_i f`` on owned cells. The
    stencil must have been built from a fully prepared halo field, while this
    operator contracts the owned-cell derivatives with the owned portion of
    the halo-shaped local magnetic field.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_grad_parallel_op_direct requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )

    if stencil.shape != geometry.owned_shape:
        raise ValueError(
            f"stencil must have shape {geometry.owned_shape}, "
            f"got {stencil.shape}"
        )

    dfdx = _take_stencil_finite_difference(stencil.x)
    dfdy = _take_stencil_finite_difference(stencil.y)
    dfdz = _take_stencil_finite_difference(stencil.z)

    df = jnp.stack((dfdx, dfdy, dfdz), axis=-1)

    # Cell-centered local geometry is halo-shaped. Use the owned magnetic
    # field properties so the contraction matches the owned derivative shape.
    B_contra = jnp.asarray(
        geometry.cell_bfield.B_contra_owned,
        dtype=jnp.float64,
    )
    Bmag = jnp.asarray(
        geometry.cell_bfield.Bmag_owned,
        dtype=jnp.float64,
    )

    Bmag = jnp.maximum(Bmag, float(b_floor))
    b_contra = B_contra / Bmag[..., None]

    result = jnp.einsum("...i,...i->...", b_contra, df)
    return _mask_inactive_owned(result, geometry)


def local_grad_parallel_op_from_gradient(
    gradient: LocalCellGradient3D,
    geometry: LocalFciGeometry3D,
    *,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Local parallel gradient from a pre-reconstructed owned-cell gradient."""

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_grad_parallel_op_from_gradient requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    if not isinstance(gradient, LocalCellGradient3D):
        raise TypeError(
            "local_grad_parallel_op_from_gradient requires LocalCellGradient3D, "
            f"got {type(gradient).__name__}"
        )
    if gradient.shape != geometry.owned_shape:
        raise ValueError(
            f"gradient must have shape {geometry.owned_shape}, got {gradient.shape}"
        )

    use_centroid_geometry = (
        control_volume_geometry is not None
        and control_volume_geometry.has_centroid_operator_geometry
    )
    if control_volume_geometry is not None:
        if control_volume_geometry.layout != geometry.layout:
            raise ValueError(
                "control-volume geometry must share geometry.layout"
            )
    B_contra = jnp.asarray(
        (
            control_volume_geometry.centroid_B_contra
            if use_centroid_geometry
            else geometry.cell_bfield.B_contra_owned
        ),
        dtype=jnp.float64,
    )
    Bmag = jnp.maximum(
        jnp.asarray(
            (
                control_volume_geometry.centroid_Bmag
                if use_centroid_geometry
                else geometry.cell_bfield.Bmag_owned
            ),
            dtype=jnp.float64,
        ),
        float(b_floor),
    )
    b_contra = B_contra / Bmag[..., None]
    result = jnp.einsum("...i,...i->...", b_contra, gradient.gradient)
    result = jnp.where(jnp.asarray(gradient.valid, dtype=bool), result, 0.0)
    return _mask_inactive_owned(result, geometry)


def local_parallel_laplacian_direct_op(
    field_halo_full: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    context: StencilBuilderContext,
    first_stencil_builder: LocalStencilBuilder = build_local_stencil_from_field,
    intermediate_stencil_builder: LocalStencilBuilder = LocalStencilBuilder(
        build_local_direct_stencil_one_sided_physical_from_halo
    ),
    halo_exchange: HaloExchange3D,
    topology_filler: TopologyHaloFiller3D,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Compute a local chained parallel Laplacian with one-sided closure.

    The first derivative is built from the fully prepared input halo field.
    Its owned result is then injected into a fresh halo field, exchanged across
    shard interfaces, and topology-filled. The second derivative uses centered
    stencils away from true physical coordinate boundaries and nonuniform
    three-point one-sided stencils on those physical boundary planes. No
    physical ghost values are read for the intermediate derivative field.

    ``intermediate_stencil_builder`` defaults to the built-in one-sided
    physical-boundary builder and can be overridden with another correctly
    constructed ``LocalStencilBuilder``.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_parallel_laplacian_direct_op requires "
            f"LocalFciGeometry3D, got {type(geometry).__name__}"
        )
    if not isinstance(domain, LocalDomain3D):
        raise TypeError(
            "local_parallel_laplacian_direct_op requires "
            f"LocalDomain3D, got {type(domain).__name__}"
        )
    if not isinstance(halo_exchange, HaloExchange3D):
        raise TypeError(
            "halo_exchange must be a HaloExchange3D, "
            f"got {type(halo_exchange).__name__}"
        )
    if not isinstance(topology_filler, TopologyHaloFiller3D):
        raise TypeError(
            "topology_filler must be a TopologyHaloFiller3D, "
            f"got {type(topology_filler).__name__}"
        )
    if not isinstance(context, StencilBuilderContext):
        raise TypeError(
            "context must be a StencilBuilderContext, "
            f"got {type(context).__name__}"
        )
    if not isinstance(intermediate_stencil_builder, LocalStencilBuilder):
        raise TypeError(
            "intermediate_stencil_builder must be a LocalStencilBuilder, "
            f"got {type(intermediate_stencil_builder).__name__}"
        )
    if domain.layout != geometry.layout:
        raise ValueError("geometry and domain must share the same HaloLayout3D")
    if context.layout != geometry.layout:
        raise ValueError("geometry and context must share the same HaloLayout3D")
    if context.domain is None:
        raise ValueError("context.domain is required for the local stencil builders")

    field_halo_full = jnp.asarray(field_halo_full, dtype=jnp.float64)
    if field_halo_full.shape != geometry.halo_shape:
        raise ValueError(
            "field_halo_full must match geometry.halo_shape; "
            f"got {field_halo_full.shape}, expected {geometry.halo_shape}"
        )

    # First derivative of the prepared input field.
    first_stencil = first_stencil_builder(
        field_halo_full,
        geometry,
        context,
    )
    q_owned = local_grad_parallel_op_direct(
        first_stencil,
        geometry,
        b_floor=b_floor,
    )

    # The intermediate derivative is owned-shaped. Reconstruct its halo before
    # taking the second derivative. These stages intentionally do not perform
    # physical ghost filling; the one-sided stencil owns those side planes.
    q_halo = inject_owned_field_to_halo(q_owned, domain.layout)
    q_halo = halo_exchange(q_halo, domain)
    q_halo = topology_filler(q_halo, domain)

    second_stencil = intermediate_stencil_builder(
        q_halo,
        geometry,
        context,
    )
    return local_grad_parallel_op_direct(
        second_stencil,
        geometry,
        b_floor=b_floor,
    )


def local_grad_perp_op_direct(
    stencil: LocalStencil3D,
    geometry: LocalFciGeometry3D,
    *,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Local/domain-decomposed direct finite-difference perpendicular gradient.

    Computes the contravariant components of the perpendicular gradient:

        grad_perp(f)^i = P^{ij} partial_j f

    where:

        P^{ij} = g^{ij} - b^i b^j

    and ``b^i = B^i / |B|``. The stencil and all geometry used in the
    contraction are owned-shaped; halo exchange and boundary preparation are
    expected to have happened before this operator is called.

    Returns:
        An owned-cell array with shape ``geometry.owned_shape + (3,)``.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_grad_perp_op_direct requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )

    if stencil.shape != geometry.owned_shape:
        raise ValueError(
            f"stencil must have shape {geometry.owned_shape}, "
            f"got {stencil.shape}"
        )

    # Coordinate partial derivatives on owned cells:
    #
    #     df_j = partial_j f
    #
    dfdx = _take_stencil_finite_difference(stencil.x)
    dfdy = _take_stencil_finite_difference(stencil.y)
    dfdz = _take_stencil_finite_difference(stencil.z)
    df = jnp.stack((dfdx, dfdy, dfdz), axis=-1)

    # These properties explicitly select owned cells from the halo-padded
    # local geometry and return the shapes required by the owned stencil.
    g_contra = jnp.asarray(geometry.cell_metric.g_contra_owned, dtype=jnp.float64)
    B_contra = jnp.asarray(
        geometry.cell_bfield.B_contra_owned,
        dtype=jnp.float64,
    )
    Bmag = jnp.asarray(geometry.cell_bfield.Bmag_owned, dtype=jnp.float64)

    Bmag = jnp.maximum(Bmag, float(b_floor))
    b_contra = B_contra / Bmag[..., None]

    projector = g_contra - jnp.einsum(
        "...i,...j->...ij",
        b_contra,
        b_contra,
    )

    result = jnp.einsum("...ij,...j->...i", projector, df)
    return _mask_inactive_owned(result, geometry)


def local_perp_laplacian_local_op(
    field_halo_full: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    context: StencilBuilderContext,
    field_stencil_builder: LocalStencilBuilder = build_local_stencil_from_field,
    intermediate_stencil_builder: LocalStencilBuilder,
    halo_exchange: HaloExchange3D,
    topology_filler: TopologyHaloFiller3D,
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Compute the domain-decomposed pointwise perpendicular Laplacian.

    This evaluates

        ``(1 / J) partial_i (J P^{ij} partial_j f)``

    using owned-cell coordinate stencils. The ``topology_filler`` is applied
    to the intermediate contravariant vector density
    ``F^i = J P^{ij} partial_j f``, which is injected as one vector-valued halo
    field so all three components pass through one halo exchange and one
    topology filler call. At a lower polar axis this density uses the component
    transform ``diag(+1, -1, -1)``, not the ordinary contravariant-vector
    transform ``diag(-1, +1, +1)``. The intermediate stencil builder is
    responsible for the physical-boundary closure of each scalar component
    (for example, the one-sided builder used by the chained parallel
    Laplacian).

    This is a pointwise/local reconstruction operator, not the conservative
    face-flux finite-volume perpendicular Laplacian.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_perp_laplacian_local_op requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    if not isinstance(domain, LocalDomain3D):
        raise TypeError(
            "local_perp_laplacian_local_op requires LocalDomain3D, "
            f"got {type(domain).__name__}"
        )
    if not isinstance(context, StencilBuilderContext):
        raise TypeError(
            "context must be a StencilBuilderContext, "
            f"got {type(context).__name__}"
        )
    if not isinstance(field_stencil_builder, LocalStencilBuilder):
        raise TypeError(
            "field_stencil_builder must be a LocalStencilBuilder, "
            f"got {type(field_stencil_builder).__name__}"
        )
    if not isinstance(intermediate_stencil_builder, LocalStencilBuilder):
        raise TypeError(
            "intermediate_stencil_builder must be a LocalStencilBuilder, "
            f"got {type(intermediate_stencil_builder).__name__}"
        )
    if not isinstance(halo_exchange, HaloExchange3D):
        raise TypeError(
            "halo_exchange must be a HaloExchange3D, "
            f"got {type(halo_exchange).__name__}"
        )
    if not isinstance(topology_filler, TopologyHaloFiller3D):
        raise TypeError(
            "topology_filler must be a TopologyHaloFiller3D, "
            f"got {type(topology_filler).__name__}"
        )
    if geometry.layout != domain.layout:
        raise ValueError("geometry and domain must share the same HaloLayout3D")
    if context.layout != geometry.layout:
        raise ValueError("geometry and context must share the same HaloLayout3D")
    if context.domain is None:
        raise ValueError("context.domain is required for local stencil builders")

    field_halo_full = jnp.asarray(field_halo_full, dtype=jnp.float64)
    if field_halo_full.shape != geometry.halo_shape:
        raise ValueError(
            "field_halo_full must match geometry.halo_shape; "
            f"got {field_halo_full.shape}, expected {geometry.halo_shape}"
        )

    field_stencil = field_stencil_builder(
        field_halo_full,
        geometry,
        context,
    )
    if field_stencil.shape != geometry.owned_shape:
        raise ValueError(
            "field_stencil must have owned-cell shape; "
            f"got {field_stencil.shape}, expected {geometry.owned_shape}"
        )

    grad_f = local_grad_perp_op_direct(
        field_stencil,
        geometry,
        b_floor=b_floor,
    )
    J_owned = jnp.asarray(geometry.cell_metric.J_owned, dtype=jnp.float64)
    flux_owned = J_owned[..., None] * grad_f

    # Keep the three components together through the communication stages.
    # The scalar stencil builder is called only after the single vector halo
    # exchange/topology pass, once for each component.
    flux_halo = inject_owned_vector_field_to_halo(
        flux_owned,
        domain.layout,
    )
    flux_halo = halo_exchange(flux_halo, domain)
    flux_halo = topology_filler(flux_halo, domain)

    flux_stencils = tuple(
        intermediate_stencil_builder(
            flux_halo[..., component],
            geometry,
            context,
        )
        for component in range(3)
    )
    for component, flux_stencil in enumerate(flux_stencils):
        if flux_stencil.shape != geometry.owned_shape:
            raise ValueError(
                f"flux_{component}_stencil must have owned-cell shape; "
                f"got {flux_stencil.shape}, expected {geometry.owned_shape}"
            )

    div_flux = (
        _take_stencil_finite_difference(flux_stencils[0].x)
        + _take_stencil_finite_difference(flux_stencils[1].y)
        + _take_stencil_finite_difference(flux_stencils[2].z)
    )
    result = div_flux / jnp.maximum(J_owned, float(jacobian_floor))
    return _mask_inactive_owned(result, geometry)


def local_poisson_bracket_op(
    f_stencil: LocalStencil3D,
    g_stencil: LocalStencil3D,
    geometry: LocalFciGeometry3D,
    *,
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Compute the owned-cell logical Poisson bracket.

    The input stencils are assumed to be complete local stencils: their
    builders own halo exchange, topology filling, physical-boundary closure,
    and any cut-wall treatment. This operator only evaluates the owned-cell
    algebra using local geometry.

    The bracket is

        ``{f, g} = (1 / J) b_i epsilon^{ijk} partial_j f partial_k g``.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_poisson_bracket_op requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    if not isinstance(f_stencil, LocalStencil3D):
        raise TypeError(
            "f_stencil must be a LocalStencil3D, "
            f"got {type(f_stencil).__name__}"
        )
    if not isinstance(g_stencil, LocalStencil3D):
        raise TypeError(
            "g_stencil must be a LocalStencil3D, "
            f"got {type(g_stencil).__name__}"
        )
    if f_stencil.shape != geometry.owned_shape:
        raise ValueError(
            f"f_stencil must have shape {geometry.owned_shape}, "
            f"got {f_stencil.shape}"
        )
    if g_stencil.shape != geometry.owned_shape:
        raise ValueError(
            f"g_stencil must have shape {geometry.owned_shape}, "
            f"got {g_stencil.shape}"
        )

    df = jnp.stack(
        (
            _take_stencil_finite_difference(f_stencil.x),
            _take_stencil_finite_difference(f_stencil.y),
            _take_stencil_finite_difference(f_stencil.z),
        ),
        axis=-1,
    )
    dg = jnp.stack(
        (
            _take_stencil_finite_difference(g_stencil.x),
            _take_stencil_finite_difference(g_stencil.y),
            _take_stencil_finite_difference(g_stencil.z),
        ),
        axis=-1,
    )

    g_cov = jnp.asarray(geometry.cell_metric.g_cov_owned, dtype=jnp.float64)
    B_contra = jnp.asarray(
        geometry.cell_bfield.B_contra_owned,
        dtype=jnp.float64,
    )
    Bmag = jnp.asarray(geometry.cell_bfield.Bmag_owned, dtype=jnp.float64)
    Bmag = jnp.maximum(Bmag, float(b_floor))

    b_contra = B_contra / Bmag[..., None]
    b_covariant = jnp.einsum(
        "...ij,...j->...i",
        g_cov,
        b_contra,
    )
    cross = jnp.cross(df, dg, axis=-1)
    J_owned = jnp.asarray(geometry.cell_metric.J_owned, dtype=jnp.float64)

    result = jnp.sum(b_covariant * cross, axis=-1) / jnp.maximum(
        J_owned,
        float(jacobian_floor),
    )
    return _mask_inactive_owned(result, geometry)


def local_poisson_bracket_op_from_gradients(
    f_gradient: LocalCellGradient3D,
    g_gradient: LocalCellGradient3D,
    geometry: LocalFciGeometry3D,
    *,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Compute the owned-cell logical Poisson bracket from reconstructed gradients."""

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_poisson_bracket_op_from_gradients requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    if not isinstance(f_gradient, LocalCellGradient3D):
        raise TypeError(
            "f_gradient must be a LocalCellGradient3D, "
            f"got {type(f_gradient).__name__}"
        )
    if not isinstance(g_gradient, LocalCellGradient3D):
        raise TypeError(
            "g_gradient must be a LocalCellGradient3D, "
            f"got {type(g_gradient).__name__}"
        )
    if f_gradient.shape != geometry.owned_shape:
        raise ValueError(
            f"f_gradient must have shape {geometry.owned_shape}, "
            f"got {f_gradient.shape}"
        )
    if g_gradient.shape != geometry.owned_shape:
        raise ValueError(
            f"g_gradient must have shape {geometry.owned_shape}, "
            f"got {g_gradient.shape}"
        )

    use_centroid_geometry = (
        control_volume_geometry is not None
        and control_volume_geometry.has_centroid_operator_geometry
    )
    if control_volume_geometry is not None:
        if control_volume_geometry.layout != geometry.layout:
            raise ValueError(
                "control-volume geometry must share geometry.layout"
            )
    g_cov = jnp.asarray(
        (
            control_volume_geometry.centroid_g_cov
            if use_centroid_geometry
            else geometry.cell_metric.g_cov_owned
        ),
        dtype=jnp.float64,
    )
    B_contra = jnp.asarray(
        (
            control_volume_geometry.centroid_B_contra
            if use_centroid_geometry
            else geometry.cell_bfield.B_contra_owned
        ),
        dtype=jnp.float64,
    )
    Bmag = jnp.maximum(
        jnp.asarray(
            (
                control_volume_geometry.centroid_Bmag
                if use_centroid_geometry
                else geometry.cell_bfield.Bmag_owned
            ),
            dtype=jnp.float64,
        ),
        float(b_floor),
    )
    b_contra = B_contra / Bmag[..., None]
    b_covariant = jnp.einsum("...ij,...j->...i", g_cov, b_contra)
    cross = jnp.cross(f_gradient.gradient, g_gradient.gradient, axis=-1)
    J_owned = jnp.asarray(
        (
            control_volume_geometry.centroid_J
            if use_centroid_geometry
            else geometry.cell_metric.J_owned
        ),
        dtype=jnp.float64,
    )
    result = jnp.sum(b_covariant * cross, axis=-1) / jnp.maximum(
        J_owned,
        float(jacobian_floor),
    )
    valid = jnp.asarray(f_gradient.valid, dtype=bool) & jnp.asarray(
        g_gradient.valid,
        dtype=bool,
    )
    result = jnp.where(valid, result, 0.0)
    return _mask_inactive_owned(result, geometry)


def _compatible_flux_face_one_form(
    geometry: LocalFciGeometry3D,
    axis: int,
    *,
    b_floor: float,
) -> jnp.ndarray:
    """Return ``A_beta = (b/B)_beta`` on one owned coordinate-face family."""

    metric = geometry.face_metric.axes[axis]
    bfield = geometry.face_bfield.axes[axis]
    bmag = jnp.maximum(jnp.asarray(bfield.Bmag_owned, dtype=jnp.float64), b_floor)
    b_contra = jnp.asarray(bfield.B_contra_owned, dtype=jnp.float64) / bmag[..., None]
    b_covariant = jnp.einsum(
        "...ij,...j->...i",
        jnp.asarray(metric.g_cov_owned, dtype=jnp.float64),
        b_contra,
    )
    return b_covariant / bmag[..., None]


def _compatible_flux_divergence(
    flux: FaceFluxStencil3D,
    geometry: LocalFciGeometry3D,
    *,
    jacobian_floor: float,
) -> jnp.ndarray:
    """Evaluate the logical incidence divergence of a face flux density."""

    divergence = (
        (flux.x[1:] - flux.x[:-1])
        / jnp.maximum(jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64), jacobian_floor)
        + (flux.y[:, 1:] - flux.y[:, :-1])
        / jnp.maximum(jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64), jacobian_floor)
        + (flux.z[:, :, 1:] - flux.z[:, :, :-1])
        / jnp.maximum(jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64), jacobian_floor)
    )
    return divergence


def _compatible_flux_generator(
    stencil: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    *,
    domain: LocalDomain3D | None,
    axis_regular_axes: tuple[bool, bool, bool],
    b_floor: float,
) -> FaceFluxStencil3D:
    """Build ``U_s^alpha = epsilon^(alpha beta gamma) A_beta s_gamma``."""

    gradients = stencil.face_grad
    A_x_face, A_y_face, A_z_face = tuple(
        _compatible_flux_face_one_form(geometry, axis, b_floor=b_floor)
        for axis in range(3)
    )
    grad_x, grad_y, grad_z = gradients.x, gradients.y, gradients.z
    flux = FaceFluxStencil3D(
        x=A_x_face[..., 1] * grad_x[..., 2] - A_x_face[..., 2] * grad_x[..., 1],
        y=A_y_face[..., 2] * grad_y[..., 0] - A_y_face[..., 0] * grad_y[..., 2],
        z=A_z_face[..., 0] * grad_z[..., 1] - A_z_face[..., 1] * grad_z[..., 0],
    )

    # The collapsed lower-x face is a topological face, not a physical face.
    # Only the shard that owns the global lower side may alter it.
    if axis_regular_axes[0]:
        if domain is None:
            raise ValueError(
                "domain is required when axis_regular_axes[0] is enabled"
            )
        axis_owner = domain.runtime_has_axis_regular_lower(0)
        lower_x = flux.x[0]
        flux = FaceFluxStencil3D(
            x=flux.x.at[0].set(
                jnp.where(axis_owner, jnp.zeros_like(lower_x), lower_x)
            ),
            y=flux.y,
            z=flux.z,
        )
    return flux


def _axis_face_samples_from_halo(
    field_halo: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    *,
    axis: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the four cell samples surrounding every owned coordinate face.

    Face ``f`` lies between the second and third returned samples.  A two-cell
    halo therefore supplies the canonical third-order pair on shard and
    periodic faces without any special casing.  With a one-cell halo the
    outer samples are duplicated from their adjacent cells; callers then use
    the corresponding first-order fallback.
    """

    values = jnp.asarray(field_halo, dtype=jnp.float64)
    if values.shape != geometry.halo_shape:
        raise ValueError(
            "field_halo must match geometry.halo_shape; "
            f"got {values.shape}, expected {geometry.halo_shape}"
        )
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    layout = geometry.layout
    halo = int(layout.halo_width)
    if halo < 1:
        raise ValueError("characteristic face reconstruction requires halo_width >= 1")
    owned_shape = tuple(int(value) for value in geometry.owned_shape)

    def take(offset: int) -> jnp.ndarray:
        slices = [
            slice(halo, halo + owned_shape[component])
            for component in range(3)
        ]
        start = halo + int(offset)
        stop = start + owned_shape[axis] + 1
        slices[axis] = slice(start, stop)
        return values[tuple(slices)]

    left_owner = take(-1)
    right_owner = take(0)
    if halo >= 2:
        left_outer = take(-2)
        right_outer = take(1)
    else:
        left_outer = left_owner
        right_outer = right_owner
    return left_outer, left_owner, right_owner, right_outer


def _boundary_trace_planes(
    trace: LocalBoundaryFaceTrace3D | None,
    *,
    axis: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] | None:
    """Return lower/upper physical trace masks and values for one face axis."""

    if trace is None:
        return None
    names = ("x", "y", "z")
    values = jnp.asarray(getattr(trace, f"value_{names[axis]}"), dtype=jnp.float64)
    masks = jnp.asarray(getattr(trace, f"mask_{names[axis]}"), dtype=bool)
    if axis == 0:
        return masks[0], values[0], masks[-1], values[-1]
    if axis == 1:
        return masks[:, 0, :], values[:, 0, :], masks[:, -1, :], values[:, -1, :]
    return masks[:, :, 0], values[:, :, 0], masks[:, :, -1], values[:, :, -1]


def _third_order_scalar_face_states_from_halo(
    field_halo: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    *,
    boundary_trace: LocalBoundaryFaceTrace3D | None,
    axis_regular_axes: tuple[bool, bool, bool],
    positivity_floor: float | None,
) -> tuple[CoordinateFaceValues3D, CoordinateFaceValues3D, CoordinateFaceValues3D]:
    """Build scalar third-order left/right states and fallback masks.

    The reconstruction matches the production curvature/parallel stencil,

    ``qL=(-q[i-1]+5q[i]+2q[i+1])/6`` and
    ``qR=(2q[i]+5q[i+1]-q[i+2])/6``.

    Each side independently falls back to its adjacent first-order owner when
    the reconstruction is non-finite or violates a requested positivity
    floor.  Physical coordinate boundaries always use the established
    operator trace and adjacent owner state, exactly as the characteristic
    curvature/parallel paths fall back at a wall.  The returned mask is one
    where either side used a fallback.
    """

    boundary_trace = _validate_local_boundary_face_trace(
        boundary_trace, geometry.layout
    )
    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    if len(axis_regular_axes) != 3:
        raise ValueError("axis_regular_axes must have length 3")
    if positivity_floor is not None:
        positivity_floor = float(positivity_floor)
        if not np.isfinite(positivity_floor) or positivity_floor <= 0.0:
            raise ValueError("positivity_floor must be finite and positive")

    left_faces = []
    right_faces = []
    fallback_faces = []
    for axis in range(3):
        qm, q0, q1, qp = _axis_face_samples_from_halo(
            field_halo, geometry, axis=axis
        )
        if int(geometry.layout.halo_width) >= 2:
            left = (-qm + 5.0 * q0 + 2.0 * q1) / 6.0
            right = (2.0 * q0 + 5.0 * q1 - qp) / 6.0
            left_ok = jnp.isfinite(left)
            right_ok = jnp.isfinite(right)
            if positivity_floor is not None:
                left_ok = left_ok & (left > positivity_floor)
                right_ok = right_ok & (right > positivity_floor)
        else:
            left = q0
            right = q1
            left_ok = jnp.zeros_like(left, dtype=bool)
            right_ok = jnp.zeros_like(right, dtype=bool)

        # Match the established characteristic reconstruction contract: a
        # failed high-order side falls back to its adjacent owner, but a
        # non-finite owner is not silently repaired.  It must remain visible
        # to the stage-validity checks.
        q0_safe = q0
        q1_safe = q1
        if positivity_floor is not None:
            q0_safe = jnp.maximum(q0_safe, positivity_floor)
            q1_safe = jnp.maximum(q1_safe, positivity_floor)
        left = jnp.where(left_ok, left, q0_safe)
        right = jnp.where(right_ok, right, q1_safe)
        fallback = ~(left_ok & right_ok)

        trace_planes = _boundary_trace_planes(boundary_trace, axis=axis)
        if trace_planes is not None:
            lower_mask, lower_value, upper_mask, upper_value = trace_planes
            if positivity_floor is not None:
                lower_value = jnp.maximum(lower_value, positivity_floor)
                upper_value = jnp.maximum(upper_value, positivity_floor)
            if not (axis == 0 and axis_regular_axes[0]):
                lower_index = _axis_index_nd(axis, 0, left.ndim)
                left = left.at[lower_index].set(
                    jnp.where(lower_mask, lower_value, left[lower_index])
                )
                right = right.at[lower_index].set(
                    jnp.where(lower_mask, q1_safe[lower_index], right[lower_index])
                )
                fallback = fallback.at[lower_index].set(
                    jnp.where(lower_mask, True, fallback[lower_index])
                )
            upper_index = _axis_index_nd(axis, -1, left.ndim)
            left = left.at[upper_index].set(
                jnp.where(upper_mask, q0_safe[upper_index], left[upper_index])
            )
            right = right.at[upper_index].set(
                jnp.where(upper_mask, upper_value, right[upper_index])
            )
            fallback = fallback.at[upper_index].set(
                jnp.where(upper_mask, True, fallback[upper_index])
            )

        left_faces.append(left)
        right_faces.append(right)
        fallback_faces.append(fallback.astype(jnp.float64))

    return (
        CoordinateFaceValues3D(*left_faces),
        CoordinateFaceValues3D(*right_faces),
        CoordinateFaceValues3D(*fallback_faces),
    )


def _compatible_characteristic_regular_flux(
    generator_flux: FaceFluxStencil3D,
    left_argument: CoordinateFaceValues3D,
    right_argument: CoordinateFaceValues3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the complete ``U * q_upwind,3`` flux on shared regular faces."""

    fluxes = []
    for velocity, left, right in zip(
        (generator_flux.x, generator_flux.y, generator_flux.z),
        (left_argument.x, left_argument.y, left_argument.z),
        (right_argument.x, right_argument.y, right_argument.z),
        strict=True,
    ):
        upwind = jnp.where(velocity >= 0.0, left, right)
        fluxes.append(velocity * upwind)
    return tuple(fluxes)


def local_poisson_bracket_compatible_flux_op(
    f_stencil: ConservativeStencil3D,
    g_stencil: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    *,
    domain: LocalDomain3D | None = None,
    f_boundary_trace: LocalBoundaryFaceTrace3D | None = None,
    g_boundary_trace: LocalBoundaryFaceTrace3D | None = None,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
    f_field_closure: LocalControlVolumeFieldClosure3D | None = None,
    g_field_closure: LocalControlVolumeFieldClosure3D | None = None,
    f_control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
    g_control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    characteristic_scheme: str = "centered",
    g_field_halo: jnp.ndarray | None = None,
    g_positivity_floor: float | None = None,
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Compatible-face Poisson bracket, already divided by ``B``.

    Both inputs are conservative scalar stencils.  For each generator ``s``
    this constructs the shared face flux density

        ``U_s^alpha = epsilon^(alpha beta gamma) A_beta partial_gamma s``,
        ``A_beta = (b/B)_beta``.

    The action is evaluated as

        ``A_s(q) = [D(U_s q_face) - q_center D(U_s)] / J``.

    If supplied, ``f_boundary_trace`` and ``g_boundary_trace`` provide the
    operator-specific physical face values for ``f`` and ``g``.  The trace for
    the advected argument is applied inside each action, while topological
    axis faces remain untouched.  The centered selector returns
    ``B_c(f,g) = 0.5 * (A_f^c(g) - A_g^c(f))``.  The
    ``scalar-centered`` selector returns ``A_f^c(g)`` for exact operator
    diagnostics.  The ``scalar-third-order-upwind`` selector returns the pure
    advected-scalar action ``A_f^up(g)``.  The
    ``compatible-third-order-upwind`` selector
    preserves the centered compatible core and adds only the physical
    generator's characteristic correction,

        ``B_c(f,g) + A_f^up(g) - A_f^c(g)``.

    The characteristic
    speed is the physical normal E x B face flux ``U_f``; no dissipation
    coefficient is tunable.  Regular bulk and
    shard faces use the same third-order reconstruction as the production
    curvature/parallel systems.  Physical coordinate faces and compact RLP
    faces use first-order owner/wall traces.  Thermodynamic callers may supply
    ``g_positivity_floor`` for the same side-wise admissibility fallback used
    by those systems.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_poisson_bracket_compatible_flux_op requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    for name, stencil in (("f_stencil", f_stencil), ("g_stencil", g_stencil)):
        if not isinstance(stencil, ConservativeStencil3D):
            raise TypeError(
                f"{name} must be a ConservativeStencil3D, "
                f"got {type(stencil).__name__}"
            )
        if stencil.shape != geometry.owned_shape:
            raise ValueError(
                f"{name} must have shape {geometry.owned_shape}, got {stencil.shape}"
            )
    if characteristic_scheme not in (
        "centered",
        "scalar-centered",
        "scalar-third-order-upwind",
        "compatible-third-order-upwind",
    ):
        raise ValueError(
            "characteristic_scheme must be 'centered', 'scalar-centered', "
            "'scalar-third-order-upwind', or "
            "'compatible-third-order-upwind', got "
            f"{characteristic_scheme!r}"
        )
    characteristic_upwind = characteristic_scheme in (
        "scalar-third-order-upwind",
        "compatible-third-order-upwind",
    )
    if characteristic_upwind:
        if g_field_halo is None:
            raise ValueError(
                "g_field_halo is required for third-order characteristic "
                "Poisson-bracket upwinding"
            )
        g_field_halo = jnp.asarray(g_field_halo, dtype=jnp.float64)
        if g_field_halo.shape != geometry.halo_shape:
            raise ValueError(
                "g_field_halo must match geometry.halo_shape; "
                f"got {g_field_halo.shape}, expected {geometry.halo_shape}"
            )
        left_g, right_g, _g_reconstruction_fallback = (
            _third_order_scalar_face_states_from_halo(
                g_field_halo,
                geometry,
                boundary_trace=g_boundary_trace,
                axis_regular_axes=axis_regular_axes,
                positivity_floor=g_positivity_floor,
            )
        )

    supplied_geometries = tuple(
        geometry_value for geometry_value in (
            control_volume_geometry,
            f_control_volume_geometry,
            g_control_volume_geometry,
        ) if geometry_value is not None
    )
    if supplied_geometries:
        if control_volume_geometry is None:
            control_volume_geometry = supplied_geometries[0]
        if any(value is not control_volume_geometry for value in supplied_geometries):
            raise ValueError("f/g control-volume geometries must be the same topology")
    if control_volume_geometry is not None:
        if not isinstance(domain, LocalDomain3D):
            raise ValueError("domain is required with control_volume_geometry")
        if not isinstance(control_volume_geometry, LocalEmbeddedControlVolumeGeometry3D):
            raise TypeError("control_volume_geometry must be LocalEmbeddedControlVolumeGeometry3D")
        if control_volume_geometry.cells.layout != geometry.layout:
            raise ValueError("control-volume geometry must share geometry.layout")
        f_closure = _require_local_control_volume_field_closure(
            f_field_closure, control_volume_geometry
        )
        g_closure = _require_local_control_volume_field_closure(
            g_field_closure, control_volume_geometry
        )
        f_boundary_trace = _validate_local_boundary_face_trace(
            f_boundary_trace, geometry.layout
        )
        g_boundary_trace = _validate_local_boundary_face_trace(
            g_boundary_trace, geometry.layout
        )
        regular_faces = control_volume_geometry.regular_faces

        def _compact_action_flux(generator_closure, argument_closure):
            faces = control_volume_geometry.irregular_faces
            A = jnp.einsum(
                "...ab,...b->...a", jnp.asarray(faces.g_cov, dtype=jnp.float64),
                jnp.asarray(faces.B_contra, dtype=jnp.float64),
            ) / jnp.maximum(jnp.asarray(faces.Bmag, dtype=jnp.float64), b_floor)[..., None]**2
            grad = jnp.asarray(generator_closure.face_gradient, dtype=jnp.float64)
            U = jnp.stack((
                A[..., 1] * grad[..., 2] - A[..., 2] * grad[..., 1],
                A[..., 2] * grad[..., 0] - A[..., 0] * grad[..., 2],
                A[..., 0] * grad[..., 1] - A[..., 1] * grad[..., 0],
            ), axis=-1)
            normal = jnp.sum(
                jnp.asarray(faces.area_covector_weight, dtype=jnp.float64) * U,
                axis=-1,
            )
            valid = (
                jnp.asarray(generator_closure.face_gradient_valid, dtype=bool)
                & jnp.asarray(argument_closure.face_value_valid, dtype=bool)
                & jnp.asarray(faces.quadrature_active, dtype=bool)
            )
            weighted = jnp.sum(
                jnp.where(~jnp.asarray(faces.quadrature_active, dtype=bool), 0.0,
                          jnp.where(valid, normal * argument_closure.face_value, jnp.nan)),
                axis=(1, 2),
            )
            generator_only = jnp.sum(
                jnp.where(~jnp.asarray(faces.quadrature_active, dtype=bool), 0.0,
                          jnp.where(valid, normal, jnp.nan)), axis=(1, 2)
            )
            return weighted, generator_only

        def _compact_characteristic_flux(
            generator_closure,
            argument,
            argument_closure,
            argument_halo,
            positivity_floor,
        ):
            faces = control_volume_geometry.irregular_faces
            A = jnp.einsum(
                "...ab,...b->...a",
                jnp.asarray(faces.g_cov, dtype=jnp.float64),
                jnp.asarray(faces.B_contra, dtype=jnp.float64),
            ) / jnp.maximum(
                jnp.asarray(faces.Bmag, dtype=jnp.float64), b_floor
            )[..., None] ** 2
            grad = jnp.asarray(generator_closure.face_gradient, dtype=jnp.float64)
            U = jnp.stack(
                (
                    A[..., 1] * grad[..., 2] - A[..., 2] * grad[..., 1],
                    A[..., 2] * grad[..., 0] - A[..., 0] * grad[..., 2],
                    A[..., 0] * grad[..., 1] - A[..., 1] * grad[..., 0],
                ),
                axis=-1,
            )
            normal = jnp.sum(
                jnp.asarray(faces.area_covector_weight, dtype=jnp.float64) * U,
                axis=-1,
            )
            center = jnp.asarray(argument.x.center, dtype=jnp.float64)
            minus = center[
                faces.minus_owner_i,
                faces.minus_owner_j,
                faces.minus_owner_k,
            ]
            plus_local = center[
                faces.plus_owner_i,
                faces.plus_owner_j,
                faces.plus_owner_k,
            ]
            plus_remote = argument_halo[
                faces.remote_halo_i,
                faces.remote_halo_j,
                faces.remote_halo_k,
            ]
            centered = jnp.asarray(argument_closure.face_value, dtype=jnp.float64)
            plus = jnp.where(
                faces.has_plus_owner[:, None, None],
                plus_local[:, None, None],
                jnp.where(
                    faces.has_remote_owner[:, None, None],
                    plus_remote[:, None, None],
                    centered,
                ),
            )
            minus = jnp.broadcast_to(minus[:, None, None], centered.shape)
            if positivity_floor is not None:
                floor = float(positivity_floor)
                minus = jnp.maximum(minus, floor)
                plus = jnp.maximum(plus, floor)
            upwind = jnp.where(normal >= 0.0, minus, plus)
            valid = (
                jnp.asarray(generator_closure.face_gradient_valid, dtype=bool)
                & jnp.asarray(argument_closure.face_value_valid, dtype=bool)
                & jnp.asarray(faces.quadrature_active, dtype=bool)
                & jnp.isfinite(upwind)
            )
            weighted = jnp.sum(
                jnp.where(
                    ~jnp.asarray(faces.quadrature_active, dtype=bool),
                    0.0,
                    jnp.where(valid, normal * upwind, jnp.nan),
                ),
                axis=(1, 2),
            )
            generator_only = jnp.sum(
                jnp.where(
                    ~jnp.asarray(faces.quadrature_active, dtype=bool),
                    0.0,
                    jnp.where(valid, normal, jnp.nan),
                ),
                axis=(1, 2),
            )
            return weighted, generator_only

        def _action(generator, argument, generator_closure, argument_closure, argument_trace):
            dense_generator = _compatible_flux_generator(
                generator, geometry, domain=domain,
                axis_regular_axes=axis_regular_axes, b_floor=b_floor,
            )
            dense_argument = argument.face_values
            if argument_trace is not None:
                dense_argument = CoordinateFaceValues3D(
                    x=_apply_local_face_trace(dense_argument.x, axis=0, trace_value=argument_trace.value_x, trace_mask=argument_trace.mask_x, axis_regular_axes=axis_regular_axes),
                    y=_apply_local_face_trace(dense_argument.y, axis=1, trace_value=argument_trace.value_y, trace_mask=argument_trace.mask_y, axis_regular_axes=axis_regular_axes),
                    z=_apply_local_face_trace(dense_argument.z, axis=2, trace_value=argument_trace.value_z, trace_mask=argument_trace.mask_z, axis_regular_axes=axis_regular_axes),
                )
            weighted_dense = (
                dense_generator.x * dense_argument.x,
                dense_generator.y * dense_argument.y,
                dense_generator.z * dense_argument.z,
            )
            compact_weighted, compact_generator = _compact_action_flux(
                generator_closure, argument_closure
            )
            weighted_div = _local_control_volume_integrated_divergence(
                weighted_dense, compact_weighted, geometry, domain,
                control_volume_geometry, volume_floor=jacobian_floor,
            )
            generator_div = _local_control_volume_integrated_divergence(
                (dense_generator.x, dense_generator.y, dense_generator.z),
                compact_generator, geometry, domain,
                control_volume_geometry, volume_floor=jacobian_floor,
            )
            return weighted_div - argument.x.center * generator_div

        def _characteristic_action(
            generator,
            argument,
            generator_closure,
            argument_closure,
            left_argument,
            right_argument,
            argument_halo,
            positivity_floor,
        ):
            dense_generator = _compatible_flux_generator(
                generator,
                geometry,
                domain=domain,
                axis_regular_axes=axis_regular_axes,
                b_floor=b_floor,
            )
            regular_weighted = _compatible_characteristic_regular_flux(
                dense_generator,
                left_argument,
                right_argument,
            )
            compact_weighted, compact_generator = _compact_characteristic_flux(
                generator_closure,
                argument,
                argument_closure,
                argument_halo,
                positivity_floor,
            )
            weighted_div = _local_control_volume_integrated_divergence(
                regular_weighted,
                compact_weighted,
                geometry,
                domain,
                control_volume_geometry,
                volume_floor=jacobian_floor,
            )
            generator_div = _local_control_volume_integrated_divergence(
                (dense_generator.x, dense_generator.y, dense_generator.z),
                compact_generator,
                geometry,
                domain,
                control_volume_geometry,
                volume_floor=jacobian_floor,
            )
            return weighted_div - argument.x.center * generator_div

        centered_f_action = _action(
            f_stencil,
            g_stencil,
            f_closure,
            g_closure,
            g_boundary_trace,
        )
        centered_result = 0.5 * (
            centered_f_action
            - _action(
                g_stencil,
                f_stencil,
                g_closure,
                f_closure,
                f_boundary_trace,
            )
        )
        if characteristic_upwind:
            assert g_field_halo is not None
            characteristic_f_action = _characteristic_action(
                f_stencil,
                g_stencil,
                f_closure,
                g_closure,
                left_g,
                right_g,
                g_field_halo,
                g_positivity_floor,
            )
            result = (
                characteristic_f_action
                if characteristic_scheme == "scalar-third-order-upwind"
                else centered_result
                + characteristic_f_action
                - centered_f_action
            )
        elif characteristic_scheme == "scalar-centered":
            result = centered_f_action
        else:
            result = centered_result
        return _mask_inactive_owned(result, geometry)

    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    if len(axis_regular_axes) != 3:
        raise ValueError("axis_regular_axes must have length 3")
    if axis_regular_axes[1] or axis_regular_axes[2]:
        raise NotImplementedError(
            "local_poisson_bracket_compatible_flux_op only supports a lower x axis"
        )
    if axis_regular_axes[0] and domain is None:
        raise ValueError(
            "domain is required when axis_regular_axes[0] is enabled"
        )
    f_boundary_trace = _validate_local_boundary_face_trace(
        f_boundary_trace, geometry.layout
    )
    g_boundary_trace = _validate_local_boundary_face_trace(
        g_boundary_trace, geometry.layout
    )
    if domain is not None:
        if not isinstance(domain, LocalDomain3D):
            raise TypeError(
                "domain must be a LocalDomain3D or None, "
                f"got {type(domain).__name__}"
            )
        if domain.layout != geometry.layout:
            raise ValueError("domain and geometry must share the same HaloLayout3D")
    if b_floor <= 0.0 or jacobian_floor <= 0.0:
        raise ValueError("b_floor and jacobian_floor must be positive")

    def _action(
        generator: ConservativeStencil3D,
        argument: ConservativeStencil3D,
        argument_boundary_trace: LocalBoundaryFaceTrace3D | None,
    ) -> jnp.ndarray:
        flux = _compatible_flux_generator(
            generator,
            geometry,
            domain=domain,
            axis_regular_axes=axis_regular_axes,
            b_floor=b_floor,
        )
        argument_face_values = argument.face_values
        if argument_boundary_trace is not None:
            argument_face_values = CoordinateFaceValues3D(
                x=_apply_local_face_trace(
                    argument_face_values.x,
                    axis=0,
                    trace_value=argument_boundary_trace.value_x,
                    trace_mask=argument_boundary_trace.mask_x,
                    axis_regular_axes=axis_regular_axes,
                ),
                y=_apply_local_face_trace(
                    argument_face_values.y,
                    axis=1,
                    trace_value=argument_boundary_trace.value_y,
                    trace_mask=argument_boundary_trace.mask_y,
                    axis_regular_axes=axis_regular_axes,
                ),
                z=_apply_local_face_trace(
                    argument_face_values.z,
                    axis=2,
                    trace_value=argument_boundary_trace.value_z,
                    trace_mask=argument_boundary_trace.mask_z,
                    axis_regular_axes=axis_regular_axes,
                ),
            )
        generator_divergence = _compatible_flux_divergence(
            flux, geometry, jacobian_floor=jacobian_floor
        )
        weighted_flux = FaceFluxStencil3D(
            x=flux.x * argument_face_values.x,
            y=flux.y * argument_face_values.y,
            z=flux.z * argument_face_values.z,
        )
        weighted_divergence = _compatible_flux_divergence(
            weighted_flux, geometry, jacobian_floor=jacobian_floor
        )
        J = jnp.maximum(
            jnp.asarray(geometry.cell_metric.J_owned, dtype=jnp.float64),
            jacobian_floor,
        )
        return (weighted_divergence - argument.x.center * generator_divergence) / J

    def _characteristic_action(
        generator: ConservativeStencil3D,
        argument: ConservativeStencil3D,
        left_argument: CoordinateFaceValues3D,
        right_argument: CoordinateFaceValues3D,
    ) -> jnp.ndarray:
        generator_flux = _compatible_flux_generator(
            generator,
            geometry,
            domain=domain,
            axis_regular_axes=axis_regular_axes,
            b_floor=b_floor,
        )
        weighted_flux = _compatible_characteristic_regular_flux(
            generator_flux,
            left_argument,
            right_argument,
        )
        weighted_divergence = _compatible_flux_divergence(
            FaceFluxStencil3D(*weighted_flux),
            geometry,
            jacobian_floor=jacobian_floor,
        )
        generator_divergence = _compatible_flux_divergence(
            generator_flux,
            geometry,
            jacobian_floor=jacobian_floor,
        )
        J = jnp.maximum(
            jnp.asarray(geometry.cell_metric.J_owned, dtype=jnp.float64),
            jacobian_floor,
        )
        return (
            weighted_divergence
            - argument.x.center * generator_divergence
        ) / J

    centered_f_action = _action(f_stencil, g_stencil, g_boundary_trace)
    centered_result = 0.5 * (
        centered_f_action
        - _action(g_stencil, f_stencil, f_boundary_trace)
    )
    if characteristic_upwind:
        characteristic_f_action = _characteristic_action(
            f_stencil, g_stencil, left_g, right_g
        )
        result = (
            characteristic_f_action
            if characteristic_scheme == "scalar-third-order-upwind"
            else centered_result + characteristic_f_action - centered_f_action
        )
    elif characteristic_scheme == "scalar-centered":
        result = centered_f_action
    else:
        result = centered_result
    return _mask_inactive_owned(result, geometry)


def local_curvature_op(
    stencil: LocalStencil3D,
    geometry: LocalFciGeometry3D,
    *,
    curvature_coefficients: jnp.ndarray,
) -> jnp.ndarray:
    """Apply curvature coefficients to an owned local scalar-field stencil.

    ``curvature_coefficients`` is an owned-cell vector field with shape
    ``geometry.owned_shape + (3,)``. Halo-shaped coefficient fields must be
    sliced to the owned region before calling this operator.
    """

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_curvature_op requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    if not isinstance(stencil, LocalStencil3D):
        raise TypeError(
            "stencil must be a LocalStencil3D, "
            f"got {type(stencil).__name__}"
        )
    if stencil.shape != geometry.owned_shape:
        raise ValueError(
            f"stencil must have shape {geometry.owned_shape}, "
            f"got {stencil.shape}"
        )

    curvature_coefficients = jnp.asarray(
        curvature_coefficients,
        dtype=jnp.float64,
    )
    expected_coefficients_shape = geometry.owned_shape + (3,)
    if curvature_coefficients.shape != expected_coefficients_shape:
        raise ValueError(
            "curvature_coefficients must have owned-cell shape "
            f"{expected_coefficients_shape}, got {curvature_coefficients.shape}"
        )

    grad_f = jnp.stack(
        (
            _take_stencil_finite_difference(stencil.x),
            _take_stencil_finite_difference(stencil.y),
            _take_stencil_finite_difference(stencil.z),
        ),
        axis=-1,
    )
    result = jnp.einsum(
        "...i,...i->...",
        curvature_coefficients,
        grad_f,
    )
    return _mask_inactive_owned(result, geometry)


def local_curvature_op_from_gradient(
    gradient: LocalCellGradient3D,
    geometry: LocalFciGeometry3D,
    *,
    curvature_coefficients: jnp.ndarray,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
) -> jnp.ndarray:
    """Apply curvature coefficients to an owned reconstructed scalar gradient."""

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_curvature_op_from_gradient requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    if not isinstance(gradient, LocalCellGradient3D):
        raise TypeError(
            "gradient must be a LocalCellGradient3D, "
            f"got {type(gradient).__name__}"
        )
    if gradient.shape != geometry.owned_shape:
        raise ValueError(
            f"gradient must have shape {geometry.owned_shape}, got {gradient.shape}"
        )

    use_centroid_geometry = (
        control_volume_geometry is not None
        and control_volume_geometry.has_centroid_operator_geometry
    )
    if control_volume_geometry is not None:
        if control_volume_geometry.layout != geometry.layout:
            raise ValueError(
                "control-volume geometry must share geometry.layout"
            )
    curvature_coefficients = jnp.asarray(
        (
            control_volume_geometry.centroid_curvature
            if use_centroid_geometry
            else curvature_coefficients
        ),
        dtype=jnp.float64,
    )
    expected_coefficients_shape = geometry.owned_shape + (3,)
    if curvature_coefficients.shape != expected_coefficients_shape:
        raise ValueError(
            "curvature_coefficients must have owned-cell shape "
            f"{expected_coefficients_shape}, got {curvature_coefficients.shape}"
        )

    result = jnp.einsum(
        "...i,...i->...",
        curvature_coefficients,
        gradient.gradient,
    )
    result = jnp.where(jnp.asarray(gradient.valid, dtype=bool), result, 0.0)
    return _mask_inactive_owned(result, geometry)


def _build_local_laplacian_face_projectors(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    b_floor: float = 1.0e-30,
    parallel: bool,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build owned-face projectors for local projected Laplacians."""

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "_build_local_laplacian_face_projectors requires "
            f"LocalFciGeometry3D, got {type(geometry).__name__}"
        )
    if not isinstance(domain, LocalDomain3D):
        raise TypeError(
            "_build_local_laplacian_face_projectors requires "
            f"LocalDomain3D, got {type(domain).__name__}"
        )
    if geometry.layout != domain.layout:
        raise ValueError("geometry and domain must share the same HaloLayout3D")

    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    if len(axis_regular_axes) != 3:
        raise ValueError("axis_regular_axes must have length 3")
    if axis_regular_axes[1] or axis_regular_axes[2]:
        raise NotImplementedError(
            "axis_regular_axes currently only supports the lower x axis; "
            f"got axis_regular_axes={axis_regular_axes}"
        )

    b_floor_value = float(b_floor)
    if b_floor_value < 0.0:
        raise ValueError(f"b_floor must be nonnegative, got {b_floor}")

    face_locations = ("x_face", "y_face", "z_face")
    expected_face_shapes = tuple(
        domain.layout.location_owned_shape(location)
        for location in face_locations
    )

    def _face_projector(metric, bfield, *, family_axis: int) -> jnp.ndarray:
        g_contra = jnp.asarray(metric.g_contra_owned, dtype=jnp.float64)
        B_contra = jnp.asarray(bfield.B_contra_owned, dtype=jnp.float64)
        Bmag = jnp.asarray(bfield.Bmag_owned, dtype=jnp.float64)

        B_contra = jnp.where(jnp.isfinite(B_contra), B_contra, 0.0)
        Bmag = jnp.where(jnp.isfinite(Bmag), Bmag, b_floor_value)
        b = B_contra / jnp.maximum(Bmag[..., None], b_floor_value)

        projector = jnp.einsum("...i,...j->...ij", b, b)
        if not bool(parallel):
            projector = g_contra - projector
        projector = jnp.where(jnp.isfinite(projector), projector, 0.0)

        # The x-face owned index 0 is the global lower-x face only on the
        # shard that touches that side. Other shards also have a local index 0,
        # but that face is an internal shard interface and must not be zeroed.
        if family_axis == 0 and axis_regular_axes[0]:
            do_axis_lower = domain.runtime_has_axis_regular_lower(0)
            lower = jnp.where(
                do_axis_lower,
                jnp.zeros_like(projector[0]),
                projector[0],
            )
            projector = projector.at[0].set(lower)

        expected_shape = expected_face_shapes[family_axis] + (3, 3)
        if projector.shape != expected_shape:
            raise ValueError(
                f"local face projector for {face_locations[family_axis]} must "
                f"have shape {expected_shape}, got {projector.shape}"
            )
        return projector

    return (
        _face_projector(
            geometry.face_metric.x,
            geometry.face_bfield.x,
            family_axis=0,
        ),
        _face_projector(
            geometry.face_metric.y,
            geometry.face_bfield.y,
            family_axis=1,
        ),
        _face_projector(
            geometry.face_metric.z,
            geometry.face_bfield.z,
            family_axis=2,
        ),
    )


def build_local_perp_laplacian_face_projectors(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    b_floor: float = 1.0e-30,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build owned-face projectors for the local perpendicular Laplacian."""

    return _build_local_laplacian_face_projectors(
        geometry,
        domain,
        b_floor=b_floor,
        parallel=False,
        axis_regular_axes=axis_regular_axes,
    )


def build_local_parallel_laplacian_face_projectors(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    b_floor: float = 1.0e-30,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build owned-face projectors for the local parallel Laplacian."""

    return _build_local_laplacian_face_projectors(
        geometry,
        domain,
        b_floor=b_floor,
        parallel=True,
        axis_regular_axes=axis_regular_axes,
    )


def _patch_local_axis_face_gradients(
    face_grad: jnp.ndarray,
    *,
    values_owned: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    axis: int,
    axis_kind: jnp.ndarray,
    axis_value: jnp.ndarray,
    axis_mask: jnp.ndarray,
    axis_regular_axes: tuple[bool, bool, bool],
    neumann_normal_scheme: str = "logical",
    regular_boundary_closure: (
        LocalRegularBoundaryMomentClosure3D | None
    ) = None,
) -> jnp.ndarray:
    """Apply local physical face-gradient closures on the owned face grid."""

    if neumann_normal_scheme not in ("logical", "physical"):
        raise ValueError(
            "neumann_normal_scheme must be 'logical' or 'physical', got "
            f"{neumann_normal_scheme!r}"
        )

    if axis_regular_axes[1] or axis_regular_axes[2]:
        raise NotImplementedError(
            "axis_regular_axes currently only supports the lower x axis; "
            f"got axis_regular_axes={axis_regular_axes}"
        )

    if axis == 0 and axis_regular_axes[0]:
        # The global lower-x face is handled by the axis-regular topology path,
        # not by physical face closures.
        lower_patch_allowed = False
    else:
        lower_patch_allowed = True

    face_grad = jnp.asarray(face_grad, dtype=jnp.float64)
    values_owned = jnp.asarray(values_owned, dtype=jnp.float64)
    kind = jnp.asarray(axis_kind, dtype=jnp.int32)
    value = jnp.asarray(axis_value, dtype=jnp.float64)
    mask = jnp.asarray(axis_mask, dtype=bool)
    boundary_weights = None
    boundary_weights_valid = None
    if regular_boundary_closure is not None:
        if regular_boundary_closure.layout != geometry.layout:
            raise ValueError(
                "regular boundary normal derivative and geometry must share "
                "the same HaloLayout3D"
            )
        boundary_weights, _owner_weights, boundary_weights_valid = (
            regular_boundary_closure.axis_payload(axis)
        )

    if axis == 0:
        lower_value = value[0]
        upper_value = value[-1]
        lower_kind = kind[0]
        upper_kind = kind[-1]
        lower_mask = mask[0]
        upper_mask = mask[-1]
        lower_distance = jnp.asarray(geometry.grid.x.lower_center_to_face, dtype=jnp.float64)
        upper_distance = jnp.asarray(geometry.grid.x.upper_center_to_face, dtype=jnp.float64)
        lower_center = values_owned[0]
        upper_center = values_owned[-1]
        lower_next_center = values_owned[1] if geometry.owned_shape[0] > 1 else lower_center
        upper_prev_center = values_owned[-2] if geometry.owned_shape[0] > 1 else upper_center
    elif axis == 1:
        lower_value = value[:, 0, :]
        upper_value = value[:, -1, :]
        lower_kind = kind[:, 0, :]
        upper_kind = kind[:, -1, :]
        lower_mask = mask[:, 0, :]
        upper_mask = mask[:, -1, :]
        lower_distance = jnp.asarray(geometry.grid.y.lower_center_to_face, dtype=jnp.float64)
        upper_distance = jnp.asarray(geometry.grid.y.upper_center_to_face, dtype=jnp.float64)
        lower_center = values_owned[:, 0, :]
        upper_center = values_owned[:, -1, :]
        lower_next_center = values_owned[:, 1, :] if geometry.owned_shape[1] > 1 else lower_center
        upper_prev_center = values_owned[:, -2, :] if geometry.owned_shape[1] > 1 else upper_center
    else:
        lower_value = value[:, :, 0]
        upper_value = value[:, :, -1]
        lower_kind = kind[:, :, 0]
        upper_kind = kind[:, :, -1]
        lower_mask = mask[:, :, 0]
        upper_mask = mask[:, :, -1]
        lower_distance = jnp.asarray(geometry.grid.z.lower_center_to_face, dtype=jnp.float64)
        upper_distance = jnp.asarray(geometry.grid.z.upper_center_to_face, dtype=jnp.float64)
        lower_center = values_owned[:, :, 0]
        upper_center = values_owned[:, :, -1]
        lower_next_center = values_owned[:, :, 1] if geometry.owned_shape[2] > 1 else lower_center
        upper_prev_center = values_owned[:, :, -2] if geometry.owned_shape[2] > 1 else upper_center

    def _boundary_spacing(component: int, side: str) -> jnp.ndarray:
        spacing = (
            geometry.spacing.dx_owned,
            geometry.spacing.dy_owned,
            geometry.spacing.dz_owned,
        )[component]
        index = 0 if side == "lower" else -1
        if axis == 0:
            return spacing[index, :, :]
        if axis == 1:
            return spacing[:, index, :]
        return spacing[:, :, index]

    def _patch_tangential_components(
        plane: jnp.ndarray,
        *,
        face_value: jnp.ndarray,
        patch_mask: jnp.ndarray,
        side: str,
    ) -> jnp.ndarray:
        for component in range(3):
            if component == axis:
                continue
            plane_axis = component if component < axis else component - 1
            tangent = _first_derivative_3d(
                face_value,
                _boundary_spacing(component, side),
                axis=plane_axis,
                periodic=domain.periodic_axes[component],
            )
            plane = plane.at[..., component].set(
                jnp.where(patch_mask, tangent, plane[..., component])
            )
        return plane

    def _physical_neumann_coordinate_derivative(
        plane: jnp.ndarray,
        *,
        prescribed_outward: jnp.ndarray,
        side: str,
    ) -> jnp.ndarray:
        face_metric = (
            geometry.face_metric.x,
            geometry.face_metric.y,
            geometry.face_metric.z,
        )[axis].g_contra_owned
        metric_plane = face_metric[
            _axis_index_nd(axis, 0 if side == "lower" else -1, face_metric.ndim)
        ]
        gaa = jnp.maximum(metric_plane[..., axis, axis], 1.0e-30)
        cross = jnp.zeros_like(prescribed_outward, dtype=jnp.float64)
        for component in range(3):
            if component != axis:
                cross = cross + (
                    metric_plane[..., axis, component] * plane[..., component]
                )
        outward_sign = -1.0 if side == "lower" else 1.0
        return (
            outward_sign * prescribed_outward * jnp.sqrt(gaa) - cross
        ) / gaa

    lower_plane = face_grad[_axis_index_nd(axis, 0, face_grad.ndim)]
    lower_normal = lower_plane[..., axis]
    lower_coord = (
        -8.0 * lower_value
        + 9.0 * lower_center
        - lower_next_center
    ) / jnp.maximum(6.0 * lower_distance, 1.0e-30)
    upper_coord = (
        8.0 * upper_value
        - 9.0 * upper_center
        + upper_prev_center
    ) / jnp.maximum(6.0 * upper_distance, 1.0e-30)
    if boundary_weights is not None and boundary_weights_valid is not None:
        if geometry.owned_shape[axis] < 3:
            raise ValueError(
                "finite-volume regular boundary derivative requires at least "
                "three owned cells in the normal direction"
            )
        inward_values = jnp.moveaxis(values_owned, axis, 0)
        lower_samples = inward_values[:3]
        upper_samples = jnp.flip(inward_values[-3:], axis=0)
        lower_weights = boundary_weights[
            _axis_index_nd(axis, 0, boundary_weights.ndim)
        ]
        upper_weights = boundary_weights[
            _axis_index_nd(axis, -1, boundary_weights.ndim)
        ]
        lower_valid = boundary_weights_valid[
            _axis_index_nd(axis, 0, boundary_weights_valid.ndim)
        ]
        upper_valid = boundary_weights_valid[
            _axis_index_nd(axis, -1, boundary_weights_valid.ndim)
        ]
        lower_fv_coord = (
            lower_weights[..., 0] * lower_value
            + jnp.einsum("...m,m...->...", lower_weights[..., 1:], lower_samples)
        )
        upper_fv_coord = (
            upper_weights[..., 0] * upper_value
            + jnp.einsum("...m,m...->...", upper_weights[..., 1:], upper_samples)
        )
        lower_coord = jnp.where(lower_valid, lower_fv_coord, lower_coord)
        upper_coord = jnp.where(upper_valid, upper_fv_coord, upper_coord)
    if lower_patch_allowed:
        lower_tangent_mask = lower_mask & (
            (lower_kind == BC_DIRICHLET)
            | (
                (lower_kind == BC_NEUMANN)
                & (neumann_normal_scheme == "logical")
            )
        )
        lower_face_value = jnp.where(
            lower_kind == BC_DIRICHLET,
            lower_value,
            lower_center + lower_value * lower_distance,
        )
        lower_plane = _patch_tangential_components(
            lower_plane,
            face_value=lower_face_value,
            patch_mask=lower_tangent_mask,
            side="lower",
        )
        lower_plane = lower_plane.at[..., axis].set(
            jnp.where(lower_mask & (lower_kind == BC_DIRICHLET), lower_coord, lower_normal)
        )
        lower_neumann_coord = (
            _physical_neumann_coordinate_derivative(
                lower_plane,
                prescribed_outward=lower_value,
                side="lower",
            )
            if neumann_normal_scheme == "physical"
            else -lower_value
        )
        lower_plane = lower_plane.at[..., axis].set(
            jnp.where(
                lower_mask & (lower_kind == BC_NEUMANN),
                lower_neumann_coord,
                lower_plane[..., axis],
            )
        )
        face_grad = face_grad.at[_axis_index_nd(axis, 0, face_grad.ndim)].set(lower_plane)

    upper_plane = face_grad[_axis_index_nd(axis, -1, face_grad.ndim)]
    upper_normal = upper_plane[..., axis]
    upper_tangent_mask = upper_mask & (
        (upper_kind == BC_DIRICHLET)
        | (
            (upper_kind == BC_NEUMANN)
            & (neumann_normal_scheme == "logical")
        )
    )
    upper_face_value = jnp.where(
        upper_kind == BC_DIRICHLET,
        upper_value,
        upper_center + upper_value * upper_distance,
    )
    upper_plane = _patch_tangential_components(
        upper_plane,
        face_value=upper_face_value,
        patch_mask=upper_tangent_mask,
        side="upper",
    )
    upper_plane = upper_plane.at[..., axis].set(
        jnp.where(upper_mask & (upper_kind == BC_DIRICHLET), upper_coord, upper_normal)
    )
    upper_neumann_coord = (
        _physical_neumann_coordinate_derivative(
            upper_plane,
            prescribed_outward=upper_value,
            side="upper",
        )
        if neumann_normal_scheme == "physical"
        else upper_value
    )
    upper_plane = upper_plane.at[..., axis].set(
        jnp.where(
            upper_mask & (upper_kind == BC_NEUMANN),
            upper_neumann_coord,
            upper_plane[..., axis],
        )
    )
    face_grad = face_grad.at[_axis_index_nd(axis, -1, face_grad.ndim)].set(upper_plane)

    return face_grad


def _apply_local_face_flux_bc(
    flux: jnp.ndarray,
    *,
    axis: int,
    axis_kind: jnp.ndarray,
    axis_value: jnp.ndarray,
    axis_mask: jnp.ndarray,
    axis_regular_axes: tuple[bool, bool, bool],
) -> jnp.ndarray:
    """Apply local physical face flux boundary conditions on owned faces."""

    if axis_regular_axes[1] or axis_regular_axes[2]:
        raise NotImplementedError(
            "axis_regular_axes currently only supports the lower x axis; "
            f"got axis_regular_axes={axis_regular_axes}"
        )

    result = jnp.asarray(flux, dtype=jnp.float64)
    kind = jnp.asarray(axis_kind, dtype=jnp.int32)
    value = jnp.asarray(axis_value, dtype=jnp.float64)
    mask = jnp.asarray(axis_mask, dtype=bool)

    if axis == 0:
        lower_kind = kind[0]
        upper_kind = kind[-1]
        lower_value = value[0]
        upper_value = value[-1]
        lower_mask = mask[0]
        upper_mask = mask[-1]
        skip_lower = bool(axis_regular_axes[0])
    elif axis == 1:
        lower_kind = kind[:, 0, :]
        upper_kind = kind[:, -1, :]
        lower_value = value[:, 0, :]
        upper_value = value[:, -1, :]
        lower_mask = mask[:, 0, :]
        upper_mask = mask[:, -1, :]
        skip_lower = False
    else:
        lower_kind = kind[:, :, 0]
        upper_kind = kind[:, :, -1]
        lower_value = value[:, :, 0]
        upper_value = value[:, :, -1]
        lower_mask = mask[:, :, 0]
        upper_mask = mask[:, :, -1]
        skip_lower = False

    if not skip_lower:
        lower_plane = result[_axis_index_nd(axis, 0, result.ndim)]
        lower_plane = jnp.where(lower_mask & (lower_kind == BC_NORMALFLUX), lower_value, lower_plane)
        lower_plane = jnp.where(lower_mask & (lower_kind == BC_NOFLUX), 0.0, lower_plane)
        result = result.at[_axis_index_nd(axis, 0, result.ndim)].set(lower_plane)

    upper_plane = result[_axis_index_nd(axis, -1, result.ndim)]
    upper_plane = jnp.where(upper_mask & (upper_kind == BC_NORMALFLUX), upper_value, upper_plane)
    upper_plane = jnp.where(upper_mask & (upper_kind == BC_NOFLUX), 0.0, upper_plane)
    result = result.at[_axis_index_nd(axis, -1, result.ndim)].set(upper_plane)
    return result


def _apply_local_face_value_dirichlet_bc(
    face_value: jnp.ndarray,
    *,
    axis: int,
    axis_kind: jnp.ndarray,
    axis_value: jnp.ndarray,
    axis_mask: jnp.ndarray,
    axis_regular_axes: tuple[bool, bool, bool],
) -> jnp.ndarray:
    """Patch physical boundary scalar face values for Dirichlet data."""

    if axis_regular_axes[1] or axis_regular_axes[2]:
        raise NotImplementedError(
            "axis_regular_axes currently only supports the lower x axis; "
            f"got axis_regular_axes={axis_regular_axes}"
        )

    result = jnp.asarray(face_value, dtype=jnp.float64)
    kind = jnp.asarray(axis_kind, dtype=jnp.int32)
    value = jnp.asarray(axis_value, dtype=jnp.float64)
    mask = jnp.asarray(axis_mask, dtype=bool)

    if axis == 0:
        lower_kind = kind[0]
        upper_kind = kind[-1]
        lower_value = value[0]
        upper_value = value[-1]
        lower_mask = mask[0]
        upper_mask = mask[-1]
        skip_lower = bool(axis_regular_axes[0])
    elif axis == 1:
        lower_kind = kind[:, 0, :]
        upper_kind = kind[:, -1, :]
        lower_value = value[:, 0, :]
        upper_value = value[:, -1, :]
        lower_mask = mask[:, 0, :]
        upper_mask = mask[:, -1, :]
        skip_lower = False
    else:
        lower_kind = kind[:, :, 0]
        upper_kind = kind[:, :, -1]
        lower_value = value[:, :, 0]
        upper_value = value[:, :, -1]
        lower_mask = mask[:, :, 0]
        upper_mask = mask[:, :, -1]
        skip_lower = False

    if not skip_lower:
        lower_plane = result[_axis_index_nd(axis, 0, result.ndim)]
        lower_plane = jnp.where(
            lower_mask & (lower_kind == BC_DIRICHLET),
            lower_value,
            lower_plane,
        )
        result = result.at[_axis_index_nd(axis, 0, result.ndim)].set(lower_plane)

    upper_plane = result[_axis_index_nd(axis, -1, result.ndim)]
    upper_plane = jnp.where(
        upper_mask & (upper_kind == BC_DIRICHLET),
        upper_value,
        upper_plane,
    )
    result = result.at[_axis_index_nd(axis, -1, result.ndim)].set(upper_plane)
    return result


def _validate_local_boundary_face_trace(
    boundary_trace: LocalBoundaryFaceTrace3D | None,
    layout: HaloLayout3D,
) -> LocalBoundaryFaceTrace3D | None:
    """Validate the explicit scalar boundary-trace payload."""
    if boundary_trace is None:
        return None
    if not isinstance(boundary_trace, LocalBoundaryFaceTrace3D):
        raise TypeError(
            "boundary_trace must be a LocalBoundaryFaceTrace3D or None, "
            f"got {type(boundary_trace).__name__}"
        )
    if boundary_trace.layout != layout:
        raise ValueError("boundary_trace and geometry must share the same HaloLayout3D")
    for axis, name in enumerate(("x", "y", "z")):
        expected = layout.face_control_shape(axis=axis)
        for prefix in ("value", "mask"):
            value = jnp.asarray(getattr(boundary_trace, f"{prefix}_{name}"))
            if value.shape != expected:
                raise ValueError(
                    f"boundary_trace.{prefix}_{name} must have shape {expected}, "
                    f"got {value.shape}"
                )
    return boundary_trace


def _apply_local_face_trace(
    face_value: jnp.ndarray,
    *,
    axis: int,
    trace_value: jnp.ndarray,
    trace_mask: jnp.ndarray,
    axis_regular_axes: tuple[bool, bool, bool],
) -> jnp.ndarray:
    """Patch an explicit physical scalar trace, excluding topology faces."""
    result = jnp.asarray(face_value, dtype=jnp.float64)
    value = jnp.asarray(trace_value, dtype=jnp.float64)
    mask = jnp.asarray(trace_mask, dtype=bool)
    if axis == 0:
        lower_mask, upper_mask = mask[0], mask[-1]
        lower_value, upper_value = value[0], value[-1]
        skip_lower = bool(axis_regular_axes[0])
    elif axis == 1:
        lower_mask, upper_mask = mask[:, 0, :], mask[:, -1, :]
        lower_value, upper_value = value[:, 0, :], value[:, -1, :]
        skip_lower = False
    else:
        lower_mask, upper_mask = mask[:, :, 0], mask[:, :, -1]
        lower_value, upper_value = value[:, :, 0], value[:, :, -1]
        skip_lower = False
    if not skip_lower:
        result = result.at[_axis_index_nd(axis, 0, result.ndim)].set(
            jnp.where(lower_mask, lower_value, result[_axis_index_nd(axis, 0, result.ndim)])
        )
    upper_index = _axis_index_nd(axis, -1, result.ndim)
    return result.at[upper_index].set(
        jnp.where(upper_mask, upper_value, result[upper_index])
    )


def _curvature_bc_characteristic_wall_states(
    interior: jnp.ndarray,
    boundary_trace: jnp.ndarray,
    bmag: jnp.ndarray,
    tau: float | jnp.ndarray,
    normal: jnp.ndarray,
    *,
    interior_on_right: bool,
    positivity_floor: float = 1.0e-12,
    eigenvalue_tolerance: float = 1.0e-10,
    max_condition: float = 1.0e8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve copied-Neumann wall residuals for incoming curvature modes.

    ``boundary_trace`` is the numerical primitive trace already constructed
    from the model's physical BC and metric-aware ghost closure.  In the HSX
    wall model its thermodynamic entries are the existing homogeneous-Neumann
    copy/extrapolation.  They are *constraints*, not a complete exterior
    Riemann state: the outgoing and stationary characteristic content must
    still come from ``interior``.

    The strict curvature symbol has two electron-family modes, one ion-family
    mode, and one stationary vorticity mode.  At a one-sided face either the
    two electron modes or the one ion mode enter the domain.  All three
    thermodynamic primitive residuals are reduced onto that complete incoming
    subspace in one least-residual solve.  The vorticity Dirichlet condition
    belongs to the stationary/elliptic closure and is deliberately excluded
    from this propagating curvature block.

    ``interior_on_right`` selects modes travelling to increasing coordinate
    index at the lower wall and modes travelling to decreasing coordinate
    index at the upper wall.  The characteristic matrix is frozen at the raw
    operator trace, which remains the canonical face state used by the
    fluctuation solver.

    No limiter or physical-state fallback is applied.  Non-finite spectra or
    residual solves propagate non-finite wall states, and negative
    thermodynamic results remain visible to the RK stage validity checks.
    """

    interior = jnp.asarray(interior, dtype=jnp.float64)
    boundary_trace = jnp.asarray(boundary_trace, dtype=jnp.float64)
    interior, boundary_trace = jnp.broadcast_arrays(interior, boundary_trace)
    if interior.shape[-1] != 4:
        raise ValueError(
            "curvature wall states require a trailing four-field dimension"
        )
    if not isinstance(interior_on_right, bool):
        raise TypeError("interior_on_right must be bool")
    floor = jnp.asarray(positivity_floor, dtype=jnp.float64)
    trace_finite = jnp.all(jnp.isfinite(boundary_trace), axis=-1)
    trace_thermo_ok = jnp.all(boundary_trace[..., :3] > floor, axis=-1)
    # Do not sanitize a failed physical trace.  A non-finite or inadmissible
    # trace must remain visible in the eigensystem/solve and ultimately fail
    # the RK stage rather than selecting an interior or floored wall state.
    matrix = curvature_strict_principal_matrix(boundary_trace, bmag, tau)
    normal_matrix = jnp.asarray(normal, dtype=jnp.float64)[..., None, None] * matrix
    frozen = jax.lax.stop_gradient(normal_matrix)
    eigenvalues, eigenvectors = jnp.linalg.eig(frozen)
    inverse = jnp.linalg.inv(eigenvectors)
    real_values = jnp.real(eigenvalues)
    imaginary = jnp.abs(jnp.imag(eigenvalues))
    tolerance = jnp.asarray(eigenvalue_tolerance, dtype=jnp.float64)
    incoming = (
        real_values > tolerance
        if interior_on_right
        else real_values < -tolerance
    )
    projector = jnp.einsum(
        "...ik,...k,...kj->...ij",
        eigenvectors,
        incoming.astype(jnp.float64),
        inverse,
    )
    projector = jax.lax.stop_gradient(jnp.real(projector))

    vector_norm = jnp.linalg.norm(eigenvectors, axis=(-2, -1))
    inverse_norm = jnp.linalg.norm(inverse, axis=(-2, -1))
    condition = vector_norm * inverse_norm
    spectral_valid = (
        jnp.all(jnp.isfinite(normal_matrix), axis=(-2, -1))
        & jnp.all(
            imaginary <= tolerance * (1.0 + jnp.abs(real_values)), axis=-1
        )
        & jnp.isfinite(condition)
        & (condition <= jnp.asarray(max_condition, dtype=jnp.float64))
    )

    incoming_count = jnp.sum(incoming, axis=-1)
    solved, residual_info = solve_incoming_characteristic_state(
        interior,
        boundary_trace,
        projector,
        incoming_basis=eigenvectors,
        incoming_active=incoming,
        # Vorticity is the stationary/elliptic field in this strict
        # curvature block.  All three thermodynamic boundary equations are
        # presented to the incoming solve; no electron/ion primitive rows are
        # selected by hand.
        residual_weights=jnp.asarray((1.0, 1.0, 1.0, 0.0)),
        thermodynamic_components=3,
        positivity_floor=positivity_floor,
        spectral_valid=spectral_valid,
    )
    solve_valid = residual_info["solve_valid"]
    # Tangential faces have no incoming curvature modes and need no wall lift.
    tangent = spectral_valid & (incoming_count == 0)
    solved = jnp.where(tangent[..., None], interior, solved)
    solve_valid = solve_valid | tangent

    exterior = solved
    fallback = (
        (~trace_finite)
        | (~trace_thermo_ok)
        | (~solve_valid)
        | (~residual_info["thermodynamic_admissible"])
    )
    return exterior, boundary_trace, fallback


def local_curvature_production_path_op(
    stencils: tuple[ConservativeStencil3D, ConservativeStencil3D,
                    ConservativeStencil3D, ConservativeStencil3D],
    geometry: LocalFciGeometry3D,
    coefficients: LocalCurvatureFaceCoefficients3D,
    *,
    tau: float | jnp.ndarray,
    domain: LocalDomain3D | None = None,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
    equilibrium: jnp.ndarray | None = None,
    boundary_traces: tuple[
        LocalBoundaryFaceTrace3D,
        LocalBoundaryFaceTrace3D,
        LocalBoundaryFaceTrace3D,
        LocalBoundaryFaceTrace3D,
    ] | None = None,
    wall_flux_closure: Literal[
        "equilibrium-exterior-canonical-face-state",
        "bc-characteristic-operator-trace-canonical-face-state",
    ] = "equilibrium-exterior-canonical-face-state",
    positivity_floor: float = 1.0e-12,
    return_diagnostics: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Apply the owner-face production curvature wave-propagation operator.

    The four coupled variables are reconstructed on every active coordinate
    face and advanced with one canonical-face characteristic split.  Face
    contributions are scattered
    with equal/opposite signs to the two actual aggregate
    owners, then divided by the summed owner ``raw_volume/B`` measure.  This is
    a wave-propagation update; it does not add a centered scalar operator or a
    post-hoc penalty.  The first output is the full residual, while optional
    diagnostics expose its ``(u, theta, eta)`` directional residuals.
    """
    if len(stencils) != 4:
        raise ValueError("production curvature requires four coupled stencils")
    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError("geometry must be LocalFciGeometry3D")
    if not isinstance(coefficients, LocalCurvatureFaceCoefficients3D):
        raise TypeError("coefficients must be LocalCurvatureFaceCoefficients3D")
    centers = tuple(jnp.asarray(stencil.x.center, dtype=jnp.float64) for stencil in stencils)
    if any(value.shape != geometry.owned_shape for value in centers):
        raise ValueError("coupled curvature stencils must match geometry.owned_shape")
    state = jnp.stack(centers, axis=-1)
    if equilibrium is None:
        equilibrium = jnp.asarray((1.0, 1.0, 1.0, 0.0), dtype=jnp.float64)
    else:
        equilibrium = jnp.asarray(equilibrium, dtype=jnp.float64)
    if equilibrium.shape != (4,):
        raise ValueError("equilibrium must have shape (4,)")
    valid_wall_flux_closures = (
        "equilibrium-exterior-canonical-face-state",
        "bc-characteristic-operator-trace-canonical-face-state",
    )
    if wall_flux_closure not in valid_wall_flux_closures:
        raise ValueError(
            "wall_flux_closure must be one of "
            f"{valid_wall_flux_closures!r}, got {wall_flux_closure!r}"
        )
    if wall_flux_closure == "bc-characteristic-operator-trace-canonical-face-state":
        if boundary_traces is None or len(boundary_traces) != 4:
            raise ValueError(
                "the BC-characteristic curvature wall closure requires four "
                "primitive boundary traces"
            )
        for trace in boundary_traces:
            if not isinstance(trace, LocalBoundaryFaceTrace3D):
                raise TypeError(
                    "curvature boundary traces must be LocalBoundaryFaceTrace3D"
                )
            if trace.layout != geometry.layout:
                raise ValueError(
                    "curvature boundary traces must share geometry.layout"
                )
    if not (np.isfinite(float(positivity_floor)) and float(positivity_floor) > 0.0):
        raise ValueError("positivity_floor must be finite and positive")
    cells = control_volume_geometry.cells if control_volume_geometry is not None else None
    if cells is None:
        ni, nj, nk = geometry.owned_shape
        oi, oj, ok = jnp.meshgrid(
            jnp.arange(ni, dtype=jnp.int32),
            jnp.arange(nj, dtype=jnp.int32),
            jnp.arange(nk, dtype=jnp.int32), indexing="ij",
        )
        owner_active = _active_cell_mask_owned(geometry)
        bcell = jnp.maximum(jnp.asarray(geometry.cell_bfield.Bmag_owned, dtype=jnp.float64), 1.0e-30)
        cell_volume = (
            jnp.asarray(geometry.cell_metric.J_owned, dtype=jnp.float64)
            * jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64)
            * jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64)
            * jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64)
            / bcell
        )
        owner_is_remote = jnp.zeros_like(owner_active)
    else:
        oi = jnp.asarray(cells.owner_i, dtype=jnp.int32)
        oj = jnp.asarray(cells.owner_j, dtype=jnp.int32)
        ok = jnp.asarray(cells.owner_k, dtype=jnp.int32)
        owner_active = jnp.asarray(cells.is_active_owner, dtype=bool)
        owner_is_remote = jnp.asarray(cells.owner_is_remote, dtype=bool)
        cell_volume = jnp.asarray(cells.raw_volume, dtype=jnp.float64) / jnp.maximum(
            jnp.asarray(geometry.cell_bfield.Bmag_owned, dtype=jnp.float64), 1.0e-30
        )
    owner_volume = jnp.zeros(geometry.owned_shape, dtype=jnp.float64).at[oi, oj, ok].add(cell_volume)

    if domain is None:
        periodic = (False, False, False)
    else:
        periodic = tuple(bool(value) for value in domain.periodic_axes)
    face_residuals = []
    total_owner_integrated = jnp.zeros(
        geometry.owned_shape + (4,), dtype=jnp.float64
    )
    axis_names = ("x", "y", "z")
    face_bfields = (geometry.face_bfield.x, geometry.face_bfield.y, geometry.face_bfield.z)
    for axis, (name, coefficient, bfield) in enumerate(zip(axis_names, coefficients.axes, face_bfields)):
        moved = tuple(jnp.moveaxis(getattr(stencil, name).center, axis, 0) for stencil in stencils)
        # The LocalStencil's minus/plus entries are already the mapped halo
        # values, so they provide the support cells at an owned-domain edge.
        minus_values = tuple(jnp.moveaxis(getattr(stencil, name).minus, axis, 0) for stencil in stencils)
        plus_values = tuple(jnp.moveaxis(getattr(stencil, name).plus, axis, 0) for stencil in stencils)
        c = jnp.stack(moved, axis=-1)
        m = jnp.stack(minus_values, axis=-1)
        p = jnp.stack(plus_values, axis=-1)
        wall_trace_values = None
        wall_trace_mask = None
        if boundary_traces is not None:
            wall_trace_values = jnp.stack(
                tuple(
                    jnp.moveaxis(
                        jnp.asarray(
                            getattr(trace, f"value_{name}"), dtype=jnp.float64
                        ),
                        axis,
                        0,
                    )
                    for trace in boundary_traces
                ),
                axis=-1,
            )
            wall_trace_mask = jnp.all(
                jnp.stack(
                    tuple(
                        jnp.moveaxis(
                            jnp.asarray(getattr(trace, f"mask_{name}"), dtype=bool),
                            axis,
                            0,
                        )
                        for trace in boundary_traces
                    ),
                    axis=-1,
                ),
                axis=-1,
            )
        n = c.shape[0]
        if n < 2:
            interior_left = jnp.zeros((0,) + c.shape[1:], dtype=jnp.float64)
            interior_right = interior_left
        else:
            interior_left, interior_right, _ = reconstruct_third_order_face_states(
                m[:-1], c[:-1], c[1:], p[1:], positivity_floor=positivity_floor
            )
        is_periodic = periodic[axis]
        if is_periodic:
            lower_left, lower_right = m[:1], c[:1]
            upper_left, upper_right = c[-1:], p[-1:]
        elif axis == 0:
            lower_left = jnp.broadcast_to(equilibrium, c[:1].shape)
            lower_right = c[:1]
            upper_left = c[-1:]
            upper_right = jnp.broadcast_to(equilibrium, c[-1:].shape)
        else:
            # Non-periodic transverse boundaries are represented by the same
            # equilibrium exterior convention as the radial wall closure.
            lower_left = jnp.broadcast_to(equilibrium, c[:1].shape)
            lower_right = c[:1]
            upper_left = c[-1:]
            upper_right = jnp.broadcast_to(equilibrium, c[-1:].shape)
        # Work in an axis-first temporary layout.  Besides the fluctuation
        # update below, the physical face coefficient is needed here to decide
        # which curvature modes enter at each one-sided wall.
        qface = jnp.moveaxis(
            jnp.asarray(coefficient, dtype=jnp.float64), axis, 0
        )
        bface = jnp.moveaxis(
            jnp.asarray(bfield.Bmag_owned, dtype=jnp.float64), axis, 0
        )
        qnormal = qface / jnp.maximum(jnp.abs(bface), 1.0e-30)
        lower_wall_face_state = None
        upper_wall_face_state = None
        if (
            not is_periodic
            and wall_flux_closure
            == "bc-characteristic-operator-trace-canonical-face-state"
        ):
            assert wall_trace_values is not None
            assert wall_trace_mask is not None
            lower_exterior, lower_bc_face, _ = (
                _curvature_bc_characteristic_wall_states(
                    c[:1],
                    wall_trace_values[:1],
                    bface[:1],
                    tau,
                    qnormal[:1],
                    interior_on_right=True,
                    positivity_floor=positivity_floor,
                )
            )
            upper_exterior, upper_bc_face, _ = (
                _curvature_bc_characteristic_wall_states(
                    c[-1:],
                    wall_trace_values[-1:],
                    bface[-1:],
                    tau,
                    qnormal[-1:],
                    interior_on_right=False,
                    positivity_floor=positivity_floor,
                )
            )
            lower_active = wall_trace_mask[:1]
            upper_active = wall_trace_mask[-1:]
            lower_left = jnp.where(
                lower_active[..., None], lower_exterior, lower_left
            )
            upper_right = jnp.where(
                upper_active[..., None], upper_exterior, upper_right
            )
            lower_wall_face_state = jnp.where(
                lower_active[..., None], lower_bc_face, lower_right
            )
            upper_wall_face_state = jnp.where(
                upper_active[..., None], upper_bc_face, upper_left
            )
        left_face = jnp.concatenate((lower_left, interior_left, upper_left), axis=0)
        right_face = jnp.concatenate((lower_right, interior_right, upper_right), axis=0)
        # The axis-first layout keeps the face index at zero for all three
        # directions, while owner coordinates below are explicitly mapped
        # back to native (i,j,k) order.
        # ``face_values`` is the canonical conservative face reconstruction,
        # in native (i,j,k) face-grid order.  It supplies the material state
        # at ordinary faces independently of the one-sided traces used for
        # the jump.  A physical wall must not linearize against its exterior
        # equilibrium; use the adjacent interior reconstructed trace there.
        canonical_face_state = jnp.stack(
            tuple(
                jnp.moveaxis(
                    jnp.asarray(getattr(stencil.face_values, name), dtype=jnp.float64),
                    axis,
                    0,
                )
                for stencil in stencils
            ),
            axis=-1,
        )
        face_state = canonical_face_state
        if not is_periodic:
            face_state = face_state.at[0].set(
                right_face[0]
                if lower_wall_face_state is None
                else lower_wall_face_state[0]
            )
            face_state = face_state.at[-1].set(
                left_face[-1]
                if upper_wall_face_state is None
                else upper_wall_face_state[0]
            )
        # Coefficients store Q=J*K and owner measures are raw_volume/B;
        # the material matrix is the physical K-symbol, so use Q/B as its
        # face normal to avoid inserting an extra B in the update.
        dplus, dminus = curvature_face_linearized_fluctuations(
            left_face, right_face, face_state, bface, tau, normal=qnormal,
            positivity_floor=positivity_floor,
        )
        # Face areas are physical regular-face measures.  Use the control
        # volume's fractions/open masks when available so merged owners see
        # the same measures as the scalar conservative operator.
        if control_volume_geometry is not None:
            regular = control_volume_geometry.regular_faces
            area = jnp.asarray(getattr(regular, f"{name}_area"), dtype=jnp.float64)
            area = area * jnp.asarray(getattr(regular, f"{name}_area_fraction"), dtype=jnp.float64)
            area = jnp.where(jnp.asarray(getattr(regular, f"{name}_open_mask"), dtype=bool), area, 0.0)
            # ``regular_faces.*_area`` is a dimensionless open-face factor
            # (one for an uncut coordinate face).  The physical transverse
            # coordinate measure is supplied separately by the conservative
            # control-volume divergence, via dy*dz, dx*dz, or dx*dy.  Keep
            # the production wave-propagation path on the same measure so
            # an angular owner topology cannot amplify every face by the
            # inverse logical cell area.
            logical_measure = (
                jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64)
                * jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64)
                if axis == 0 else
                jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64)
                * jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64)
                if axis == 1 else
                jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64)
                * jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64)
            )
            area = area * _lift_cell_field_to_faces(
                logical_measure,
                axis=axis,
                periodic=False,
            )
            area = jnp.moveaxis(area, axis, 0)
        else:
            cell_area = (
                jnp.asarray(geometry.spacing.dy_owned) * jnp.asarray(geometry.spacing.dz_owned)
                if axis == 0 else
                jnp.asarray(geometry.spacing.dx_owned) * jnp.asarray(geometry.spacing.dz_owned)
                if axis == 1 else
                jnp.asarray(geometry.spacing.dx_owned) * jnp.asarray(geometry.spacing.dy_owned)
            )
            cell_area = jnp.moveaxis(cell_area, axis, 0)
            area = jnp.concatenate((cell_area[:1], 0.5 * (cell_area[:-1] + cell_area[1:]), cell_area[-1:]), axis=0)
        face_shape = qface.shape
        grid = jnp.indices(face_shape, dtype=jnp.int32)
        face_index = grid[0]
        # ``grid`` is axis-first; convert the two transverse coordinates back
        # to native array order before gathering owner maps.
        axis_coords = [grid[1], grid[2]]
        native_face_coords = [None, None, None]
        transverse_index = 0
        for native_axis in range(3):
            if native_axis == axis:
                native_face_coords[native_axis] = face_index
            else:
                native_face_coords[native_axis] = axis_coords[transverse_index]
                transverse_index += 1
        left_axis = face_index - 1
        right_axis = face_index
        left_valid = jnp.ones(face_shape, dtype=bool)
        right_valid = jnp.ones(face_shape, dtype=bool)
        if is_periodic:
            left_axis = jnp.mod(left_axis, n)
            right_axis = jnp.mod(right_axis, n)
            # Face n duplicates the lower periodic face and is not scattered.
            duplicate = face_index == n
            left_valid = left_valid & ~duplicate
            right_valid = right_valid & ~duplicate
        else:
            left_valid = left_valid & (face_index > 0)
            right_valid = right_valid & (face_index < n)
            left_axis = jnp.clip(left_axis, 0, n - 1)
            right_axis = jnp.clip(right_axis, 0, n - 1)
        raw_left = list(native_face_coords)
        raw_right = list(native_face_coords)
        raw_left[axis] = left_axis
        raw_right[axis] = right_axis
        left_owner = (oi[tuple(raw_left)], oj[tuple(raw_left)], ok[tuple(raw_left)])
        right_owner = (oi[tuple(raw_right)], oj[tuple(raw_right)], ok[tuple(raw_right)])
        same_owner = (
            left_valid & right_valid
            & (left_owner[0] == right_owner[0])
            & (left_owner[1] == right_owner[1])
            & (left_owner[2] == right_owner[2])
        )
        left_remote = jnp.asarray(owner_is_remote)[tuple(raw_left)]
        right_remote = jnp.asarray(owner_is_remote)[tuple(raw_right)]
        valid_left = left_valid & ~same_owner & ~left_remote
        valid_right = right_valid & ~same_owner & ~right_remote
        integrated_plus = jnp.where(valid_right[..., None], -dplus * area[..., None], 0.0)
        integrated_minus = jnp.where(valid_left[..., None], -dminus * area[..., None], 0.0)
        owner_integrated = jnp.zeros(geometry.owned_shape + (4,), dtype=jnp.float64)
        # D+ is right-going and updates the right owner; D- is left-going and
        # updates the left owner.
        owner_integrated = owner_integrated.at[left_owner].add(integrated_minus)
        owner_integrated = owner_integrated.at[right_owner].add(integrated_plus)
        # The interface fluctuations carry only the jump correction.  The
        # smooth physical transport is the total within-cell fluctuation
        # between the reconstructed left/right boundary states.
        cell_left = right_face[:-1]
        cell_right = left_face[1:]
        cell_q = 0.5 * (qnormal[:-1] + qnormal[1:])
        cell_b = jnp.moveaxis(jnp.asarray(geometry.cell_bfield.Bmag_owned, dtype=jnp.float64), axis, 0)
        cell_state = 0.5 * (cell_left + cell_right)
        cell_plus, cell_minus = curvature_face_linearized_fluctuations(
            cell_left, cell_right, cell_state, cell_b, tau, normal=cell_q,
            positivity_floor=positivity_floor,
        )
        cell_area = 0.5 * (area[:-1] + area[1:])
        cell_raw_coords = [grid[d][:-1] for d in range(3)]
        # Cell coordinates in this axis-first layout map back to native order.
        cell_native_coords = [None, None, None]
        transverse_index = 0
        for native_axis in range(3):
            if native_axis == axis:
                cell_native_coords[native_axis] = grid[0][:-1]
            else:
                cell_native_coords[native_axis] = axis_coords[transverse_index][:-1]
                transverse_index += 1
        cell_owner = (
            oi[tuple(cell_native_coords)], oj[tuple(cell_native_coords)],
            ok[tuple(cell_native_coords)],
        )
        cell_remote = jnp.asarray(owner_is_remote)[tuple(cell_native_coords)]
        total_integrated = jnp.where(
            (~cell_remote)[..., None],
            -(cell_plus + cell_minus) * cell_area[..., None],
            0.0,
        )
        cell_path_integrated = jnp.zeros(
            geometry.owned_shape + (4,), dtype=jnp.float64
        ).at[cell_owner].add(total_integrated)
        owner_integrated = owner_integrated + cell_path_integrated
        if return_diagnostics:
            owner_update = jnp.where(
                owner_active[..., None],
                owner_integrated / jnp.maximum(owner_volume[..., None], 1.0e-30),
                0.0,
            )
            expanded = owner_update[oi, oj, ok]
            expanded = jnp.where(cell_volume[..., None] > 0.0, expanded, 0.0)
            face_residuals.append(expanded)
        else:
            total_owner_integrated = total_owner_integrated + owner_integrated
    if not return_diagnostics:
        owner_update = jnp.where(
            owner_active[..., None],
            total_owner_integrated
            / jnp.maximum(owner_volume[..., None], 1.0e-30),
            0.0,
        )
        result = owner_update[oi, oj, ok]
        result = jnp.where(cell_volume[..., None] > 0.0, result, 0.0)
        return result
    directional = jnp.stack(face_residuals, axis=0)
    result = jnp.sum(directional, axis=0)
    return result, {"directional_residual": directional}


def local_curvature_conservative_op(
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    coefficients: LocalCurvatureFaceCoefficients3D,
    *,
    domain: LocalDomain3D | None = None,
    face_bc: LocalBoundaryFaceBC3D | None = None,
    boundary_trace: LocalBoundaryFaceTrace3D | None = None,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Apply the regular-grid conservative curvature operator.

    ``coefficients.x/y/z`` are the geometry-only shared-face flux densities
    ``Q^alpha = J K^alpha``.  The returned quantity is

        ``C(f) = B/J * partial_alpha(Q^alpha f_face)``.

    This deliberately uses only shared regular-coordinate faces. In
    particular, it does not construct a ``LocalControlVolumeFluxStencil3D``
    and does not apply normal-flux or no-flux boundary conditions: those BC
    kinds describe the physical scalar flux and are not curvature scalar
    values. Only Dirichlet scalar values are patched on boundary faces.

    Projected-fine aggregation is applied outside this regular-face operator.
    """
    if not isinstance(local, ConservativeStencil3D):
        raise TypeError(
            "local_curvature_conservative_op requires ConservativeStencil3D, "
            f"got {type(local).__name__}"
        )
    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_curvature_conservative_op requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    if not isinstance(coefficients, LocalCurvatureFaceCoefficients3D):
        raise TypeError(
            "coefficients must be LocalCurvatureFaceCoefficients3D, "
            f"got {type(coefficients).__name__}"
        )
    if local.shape != geometry.owned_shape:
        raise ValueError(
            f"local stencil must have shape {geometry.owned_shape}, got {local.shape}"
        )
    if coefficients.layout != geometry.layout:
        raise ValueError("geometry and coefficients must share the same HaloLayout3D")
    expected = tuple(geometry.layout.face_control_shape(axis=a) for a in range(3))
    for axis, name, value in zip((0, 1, 2), ("x", "y", "z"), (coefficients.x, coefficients.y, coefficients.z)):
        if jnp.asarray(value).shape != expected[axis]:
            raise ValueError(
                f"coefficients.{name} must have shape {expected[axis]}, got {jnp.asarray(value).shape}"
            )

    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    if len(axis_regular_axes) != 3:
        raise ValueError("axis_regular_axes must have length 3")
    if axis_regular_axes[1] or axis_regular_axes[2]:
        raise NotImplementedError(
            "axis_regular_axes currently only supports the lower x axis; "
            f"got axis_regular_axes={axis_regular_axes}"
        )
    if axis_regular_axes[0] and domain is None:
        raise ValueError(
            "domain is required when axis_regular_axes[0] is enabled so the "
            "runtime lower-x axis owner can be identified"
        )
    if domain is not None and domain.layout != geometry.layout:
        raise ValueError("domain and geometry must share the same HaloLayout3D")
    face_bc = face_bc or LocalBoundaryFaceBC3D.empty(geometry.layout)
    if not isinstance(face_bc, LocalBoundaryFaceBC3D):
        raise TypeError("face_bc must be LocalBoundaryFaceBC3D or None")
    if face_bc.layout != geometry.layout:
        raise ValueError("face_bc and geometry must share the same HaloLayout3D")
    boundary_trace = _validate_local_boundary_face_trace(boundary_trace, geometry.layout)

    x_face = _apply_local_face_value_dirichlet_bc(
        local.face_values.x, axis=0,
        axis_kind=face_bc.kind_x, axis_value=face_bc.value_x,
        axis_mask=face_bc.mask_x, axis_regular_axes=axis_regular_axes,
    )
    y_face = _apply_local_face_value_dirichlet_bc(
        local.face_values.y, axis=1,
        axis_kind=face_bc.kind_y, axis_value=face_bc.value_y,
        axis_mask=face_bc.mask_y, axis_regular_axes=axis_regular_axes,
    )
    z_face = _apply_local_face_value_dirichlet_bc(
        local.face_values.z, axis=2,
        axis_kind=face_bc.kind_z, axis_value=face_bc.value_z,
        axis_mask=face_bc.mask_z, axis_regular_axes=axis_regular_axes,
    )
    if boundary_trace is not None:
        x_face = _apply_local_face_trace(
            x_face, axis=0, trace_value=boundary_trace.value_x,
            trace_mask=boundary_trace.mask_x, axis_regular_axes=axis_regular_axes,
        )
        y_face = _apply_local_face_trace(
            y_face, axis=1, trace_value=boundary_trace.value_y,
            trace_mask=boundary_trace.mask_y, axis_regular_axes=axis_regular_axes,
        )
        z_face = _apply_local_face_trace(
            z_face, axis=2, trace_value=boundary_trace.value_z,
            trace_mask=boundary_trace.mask_z, axis_regular_axes=axis_regular_axes,
        )
    # Axis regularity is part of the coefficient complex: the geometry
    # builder supplies Q^rho=0 on a collapsed lower face while preserving
    # div_h(Q)=0.  Do not patch a completed curvature flux here; doing so
    # breaks the discrete div(curl) identity for constants.
    fluxes = (
        jnp.asarray(coefficients.x, dtype=jnp.float64) * x_face,
        jnp.asarray(coefficients.y, dtype=jnp.float64) * y_face,
        jnp.asarray(coefficients.z, dtype=jnp.float64) * z_face,
    )
    spacings = (geometry.spacing.dx_owned, geometry.spacing.dy_owned, geometry.spacing.dz_owned)
    divergence = sum(
        (
            flux[_axis_slice_nd(axis, 1, None, flux.ndim)]
            - flux[_axis_slice_nd(axis, None, -1, flux.ndim)]
        ) / jnp.maximum(jnp.asarray(spacing, dtype=jnp.float64), float(jacobian_floor))
        for axis, (flux, spacing) in enumerate(zip(fluxes, spacings))
    )
    J = jnp.maximum(jnp.asarray(geometry.cell_metric.J_owned, dtype=jnp.float64), float(jacobian_floor))
    B = jnp.asarray(geometry.cell_bfield.Bmag_owned, dtype=jnp.float64)
    return _mask_inactive_owned(B * divergence / J, geometry)


def local_curvature_conservative_components_op(
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    coefficients: LocalCurvatureFaceCoefficients3D,
    **kwargs,
) -> jnp.ndarray:
    """Return radial, poloidal, and toroidal curvature divergences.

    The leading output axis is ``(u, theta, eta)``.  Each component is
    evaluated by the unchanged regular-face operator with the other two face
    coefficient families set to zero.  The sum therefore closes to
    :func:`local_curvature_conservative_op` to roundoff without introducing
    an alternative diagnostic discretization.

    This routine is intentionally diagnostic: it performs three operator
    applications and should not replace the single production evaluation.
    """

    if not isinstance(coefficients, LocalCurvatureFaceCoefficients3D):
        raise TypeError(
            "coefficients must be LocalCurvatureFaceCoefficients3D, "
            f"got {type(coefficients).__name__}"
        )
    zero_axes = tuple(jnp.zeros_like(value) for value in coefficients.axes)
    components = []
    for active_axis in range(3):
        axes = list(zero_axes)
        axes[active_axis] = coefficients.axes[active_axis]
        directional_coefficients = LocalCurvatureFaceCoefficients3D(
            layout=coefficients.layout,
            x=axes[0],
            y=axes[1],
            z=axes[2],
        )
        components.append(
            local_curvature_conservative_op(
                local,
                geometry,
                directional_coefficients,
                **kwargs,
            )
        )
    return jnp.stack(tuple(components), axis=0)


def _local_axis_upwind_face_values_from_stencil(
    stencil: LocalStencil1D,
    coefficient: jnp.ndarray,
    *,
    axis: int,
    axis_kind: jnp.ndarray,
    axis_value: jnp.ndarray,
    axis_mask: jnp.ndarray,
    axis_regular_axes: tuple[bool, bool, bool],
    equilibrium_inflow: bool,
) -> jnp.ndarray:
    """Build upwind face values using only the axis stencil neighbors."""

    center = jnp.asarray(stencil.center, dtype=jnp.float64)
    minus = jnp.asarray(stencil.minus, dtype=jnp.float64)
    plus = jnp.asarray(stencil.plus, dtype=jnp.float64)
    q = jnp.asarray(coefficient, dtype=jnp.float64)
    # A cell-centered axis with n owned cells has n+1 faces.  The left
    # state on those faces is [minus[0], center[0], ..., center[n-1]],
    # while the right state is [center[0], ..., center[n-1], plus[-1]].
    # Do not drop the end cells here: the physical boundary faces are part
    # of the conservative divergence and are patched below when needed.
    left = jnp.concatenate(
        (jnp.expand_dims(minus[_axis_index_nd(axis, 0, minus.ndim)], axis=axis), center),
        axis=axis,
    )
    right = jnp.concatenate(
        (center, jnp.expand_dims(plus[_axis_index_nd(axis, -1, plus.ndim)], axis=axis)),
        axis=axis,
    )
    face = jnp.where(q >= 0.0, left, right)

    kind = jnp.asarray(axis_kind, dtype=jnp.int32)
    value = jnp.asarray(axis_value, dtype=jnp.float64)
    mask = jnp.asarray(axis_mask, dtype=bool)
    if axis == 0:
        lower_kind, upper_kind = kind[0], kind[-1]
        lower_value, upper_value = value[0], value[-1]
        lower_mask, upper_mask = mask[0], mask[-1]
        skip_lower = bool(axis_regular_axes[0])
    elif axis == 1:
        lower_kind, upper_kind = kind[:, 0, :], kind[:, -1, :]
        lower_value, upper_value = value[:, 0, :], value[:, -1, :]
        lower_mask, upper_mask = mask[:, 0, :], mask[:, -1, :]
        skip_lower = False
    else:
        lower_kind, upper_kind = kind[:, :, 0], kind[:, :, -1]
        lower_value, upper_value = value[:, :, 0], value[:, :, -1]
        lower_mask, upper_mask = mask[:, :, 0], mask[:, :, -1]
        skip_lower = False

    lower_face = face[_axis_index_nd(axis, 0, face.ndim)]
    upper_face = face[_axis_index_nd(axis, -1, face.ndim)]
    lower_inflow = lower_mask & (q[_axis_index_nd(axis, 0, q.ndim)] > 0.0)
    upper_inflow = upper_mask & (q[_axis_index_nd(axis, -1, q.ndim)] < 0.0)
    lower_neumann = lower_inflow & (lower_kind == BC_NEUMANN)
    upper_neumann = upper_inflow & (upper_kind == BC_NEUMANN)
    # The Neumann option uses the boundary trace already represented by the
    # ghost-filled axis stencil: the central midpoint of the interior and
    # exterior stencil values.  This is deliberately taken from only the
    # current axis stencil, so corners never require diagonal ghost reads.
    lower_neumann_trace = 0.5 * (
        minus[_axis_index_nd(axis, 0, minus.ndim)]
        + center[_axis_index_nd(axis, 0, center.ndim)]
    )
    upper_neumann_trace = 0.5 * (
        center[_axis_index_nd(axis, -1, center.ndim)]
        + plus[_axis_index_nd(axis, -1, plus.ndim)]
    )
    lower_trace = jnp.zeros_like(lower_face) if equilibrium_inflow else lower_neumann_trace
    upper_trace = jnp.zeros_like(upper_face) if equilibrium_inflow else upper_neumann_trace
    lower_face = jnp.where(lower_neumann, lower_trace, lower_face)
    upper_face = jnp.where(upper_neumann, upper_trace, upper_face)
    lower_face = jnp.where(
        lower_inflow & (lower_kind == BC_DIRICHLET), lower_value, lower_face
    )
    upper_face = jnp.where(
        upper_inflow & (upper_kind == BC_DIRICHLET), upper_value, upper_face
    )
    if not skip_lower:
        face = face.at[_axis_index_nd(axis, 0, face.ndim)].set(lower_face)
    return face.at[_axis_index_nd(axis, -1, face.ndim)].set(upper_face)


def local_curvature_upwind_conservative_op(
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    coefficients: LocalCurvatureFaceCoefficients3D,
    *,
    domain: LocalDomain3D | None = None,
    face_bc: LocalBoundaryFaceBC3D | None = None,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    inflow_closure: Literal["neumann", "equilibrium"] = "neumann",
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Apply curvature fluxes with upwind Neumann or equilibrium inflow."""

    if not isinstance(local, ConservativeStencil3D):
        raise TypeError("local must be ConservativeStencil3D")
    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError("geometry must be LocalFciGeometry3D")
    if not isinstance(coefficients, LocalCurvatureFaceCoefficients3D):
        raise TypeError("coefficients must be LocalCurvatureFaceCoefficients3D")
    if inflow_closure not in ("neumann", "equilibrium"):
        raise ValueError("inflow_closure must be 'neumann' or 'equilibrium'")
    if local.shape != geometry.owned_shape:
        raise ValueError(f"local stencil must have shape {geometry.owned_shape}")
    if coefficients.layout != geometry.layout:
        raise ValueError("geometry and coefficients must share the same HaloLayout3D")
    face_bc = face_bc or LocalBoundaryFaceBC3D.empty(geometry.layout)
    if face_bc.layout != geometry.layout:
        raise ValueError("face_bc and geometry must share the same HaloLayout3D")
    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    if len(axis_regular_axes) != 3 or axis_regular_axes[1] or axis_regular_axes[2]:
        raise NotImplementedError("axis_regular_axes only supports lower x")
    if axis_regular_axes[0]:
        raise NotImplementedError(
            "axis-regular upwind curvature requires reconstructed left/right "
            "face states; only centered materialized face values are supported"
        )
    if domain is not None and domain.layout != geometry.layout:
        raise ValueError("domain and geometry must share the same HaloLayout3D")
    faces = []
    for axis, name, stencil, coefficient in zip(
        (0, 1, 2), ("x", "y", "z"),
        (local.x, local.y, local.z),
        (coefficients.x, coefficients.y, coefficients.z),
    ):
        expected = geometry.layout.face_control_shape(axis=axis)
        if jnp.asarray(coefficient).shape != expected:
            raise ValueError(f"coefficients.{name} must have shape {expected}")
        faces.append(_local_axis_upwind_face_values_from_stencil(
            stencil, coefficient, axis=axis,
            axis_kind=getattr(face_bc, f"kind_{name}"),
            axis_value=getattr(face_bc, f"value_{name}"),
            axis_mask=getattr(face_bc, f"mask_{name}"),
            axis_regular_axes=axis_regular_axes,
            equilibrium_inflow=inflow_closure == "equilibrium",
        ))
    # As in the centered operator, axis regularity belongs to the compatible
    # coefficient complex.  A post-hoc lower-face flux edit would destroy
    # constant-state cancellation.
    fluxes = tuple(
        jnp.asarray(coefficient, dtype=jnp.float64) * face
        for coefficient, face in zip(
            (coefficients.x, coefficients.y, coefficients.z), faces
        )
    )
    spacings = (geometry.spacing.dx_owned, geometry.spacing.dy_owned, geometry.spacing.dz_owned)
    divergence = sum(
        (flux[_axis_slice_nd(axis, 1, None, flux.ndim)]
         - flux[_axis_slice_nd(axis, None, -1, flux.ndim)])
        / jnp.maximum(jnp.asarray(spacing, dtype=jnp.float64), float(jacobian_floor))
        for axis, (flux, spacing) in enumerate(zip(fluxes, spacings))
    )
    J = jnp.maximum(jnp.asarray(geometry.cell_metric.J_owned, dtype=jnp.float64), float(jacobian_floor))
    B = jnp.asarray(geometry.cell_bfield.Bmag_owned, dtype=jnp.float64)
    return _mask_inactive_owned(B * divergence / J, geometry)


def _build_local_parallel_flux_cut_wall_payload(
    *,
    local: ConservativeStencil3D,
    cut_wall_geometry: LocalCutWallGeometry3D,
    cut_wall_bc: LocalCutWallBC3D,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Build the padded embedded-wall flux payload for ``div(f b)``."""

    if not isinstance(cut_wall_geometry, LocalCutWallGeometry3D):
        raise TypeError(
            "_build_local_parallel_flux_cut_wall_payload requires "
            f"LocalCutWallGeometry3D, got {type(cut_wall_geometry).__name__}"
        )
    if not isinstance(cut_wall_bc, LocalCutWallBC3D):
        raise TypeError(
            "_build_local_parallel_flux_cut_wall_payload requires "
            f"LocalCutWallBC3D, got {type(cut_wall_bc).__name__}"
        )
    if cut_wall_geometry.max_wall_faces != cut_wall_bc.max_wall_faces:
        raise ValueError(
            "cut_wall_geometry and cut_wall_bc must use the same padded wall-face length"
        )

    max_wall_faces = int(cut_wall_geometry.max_wall_faces)
    if max_wall_faces == 0:
        return jnp.zeros((0,), dtype=jnp.float64)

    active = jnp.asarray(cut_wall_geometry.active, dtype=bool) & jnp.asarray(
        cut_wall_bc.active,
        dtype=bool,
    )
    cut_wall_kind = jnp.asarray(cut_wall_bc.kind, dtype=jnp.int32)
    cut_wall_value = jnp.asarray(cut_wall_bc.value, dtype=jnp.float64)

    supported = (
        (cut_wall_kind == BC_NONE)
        | (cut_wall_kind == BC_DIRICHLET)
        | (cut_wall_kind == BC_NEUMANN)
        | (cut_wall_kind == BC_NORMALFLUX)
        | (cut_wall_kind == BC_NOFLUX)
    )
    active = active & supported
    cut_wall_kind = jnp.where(supported, cut_wall_kind, BC_NONE)
    cut_wall_value = jnp.where(supported, cut_wall_value, 0.0)

    owner_i = jnp.asarray(cut_wall_geometry.owner_i, dtype=jnp.int32)
    owner_j = jnp.asarray(cut_wall_geometry.owner_j, dtype=jnp.int32)
    owner_k = jnp.asarray(cut_wall_geometry.owner_k, dtype=jnp.int32)
    f_owner = jnp.asarray(local.x.center, dtype=jnp.float64)[
        owner_i,
        owner_j,
        owner_k,
    ]

    distance = jnp.asarray(cut_wall_geometry.distance, dtype=jnp.float64)
    f_wall = f_owner
    f_wall = jnp.where(cut_wall_kind == BC_DIRICHLET, cut_wall_value, f_wall)
    f_wall = jnp.where(
        cut_wall_kind == BC_NEUMANN,
        f_owner + cut_wall_value * jnp.abs(distance),
        f_wall,
    )

    bmag = jnp.maximum(
        jnp.asarray(cut_wall_geometry.Bmag, dtype=jnp.float64),
        float(b_floor),
    )
    b_wall = jnp.asarray(cut_wall_geometry.B_contra, dtype=jnp.float64) / bmag[..., None]
    wall_flux_area = (
        jnp.asarray(cut_wall_geometry.J, dtype=jnp.float64)
        * f_wall
        * jnp.einsum(
            "...i,...i->...",
            jnp.asarray(cut_wall_geometry.area_covector, dtype=jnp.float64),
            b_wall,
        )
    )
    wall_flux_area = jnp.where(
        cut_wall_kind == BC_NORMALFLUX,
        cut_wall_value,
        wall_flux_area,
    )
    wall_flux_area = jnp.where(cut_wall_kind == BC_NOFLUX, 0.0, wall_flux_area)
    wall_flux_area = jnp.where(active, wall_flux_area, 0.0)

    sign = jnp.asarray(cut_wall_geometry.sign, dtype=jnp.float64)
    if sign.shape != wall_flux_area.shape:
        raise ValueError(
            f"cut_wall_geometry.sign must have shape {wall_flux_area.shape}, got {sign.shape}"
        )
    return sign * wall_flux_area


def _regular_face_row_legacy_flux(
    regular_flux: FaceFluxStencil3D,
    rows: LocalRegularFaceContributionRows3D,
) -> jnp.ndarray:
    face_axis = jnp.asarray(rows.face_axis, dtype=jnp.int32)
    face_i = jnp.asarray(rows.face_i, dtype=jnp.int32)
    face_j = jnp.asarray(rows.face_j, dtype=jnp.int32)
    face_k = jnp.asarray(rows.face_k, dtype=jnp.int32)
    x_value = regular_flux.x[
        jnp.clip(face_i, 0, regular_flux.x.shape[0] - 1),
        jnp.clip(face_j, 0, regular_flux.x.shape[1] - 1),
        jnp.clip(face_k, 0, regular_flux.x.shape[2] - 1),
    ]
    y_value = regular_flux.y[
        jnp.clip(face_i, 0, regular_flux.y.shape[0] - 1),
        jnp.clip(face_j, 0, regular_flux.y.shape[1] - 1),
        jnp.clip(face_k, 0, regular_flux.y.shape[2] - 1),
    ]
    z_value = regular_flux.z[
        jnp.clip(face_i, 0, regular_flux.z.shape[0] - 1),
        jnp.clip(face_j, 0, regular_flux.z.shape[1] - 1),
        jnp.clip(face_k, 0, regular_flux.z.shape[2] - 1),
    ]
    return jnp.where(face_axis == 0, x_value, jnp.where(face_axis == 1, y_value, z_value))


def _regular_face_row_positions(
    geometry: LocalFciGeometry3D,
    regular_face_geometry: LocalRegularFaceGeometry3D,
    rows: LocalRegularFaceContributionRows3D,
) -> jnp.ndarray:
    face_axis = jnp.asarray(rows.face_axis, dtype=jnp.int32)
    face_i = jnp.asarray(rows.face_i, dtype=jnp.int32)
    face_j = jnp.asarray(rows.face_j, dtype=jnp.int32)
    face_k = jnp.asarray(rows.face_k, dtype=jnp.int32)
    x_positions = _owned_face_logical_positions(
        geometry,
        regular_face_geometry,
        face_axis=0,
    )
    y_positions = _owned_face_logical_positions(
        geometry,
        regular_face_geometry,
        face_axis=1,
    )
    z_positions = _owned_face_logical_positions(
        geometry,
        regular_face_geometry,
        face_axis=2,
    )
    x_pos = x_positions[
        jnp.clip(face_i, 0, x_positions.shape[0] - 1),
        jnp.clip(face_j, 0, x_positions.shape[1] - 1),
        jnp.clip(face_k, 0, x_positions.shape[2] - 1),
    ]
    y_pos = y_positions[
        jnp.clip(face_i, 0, y_positions.shape[0] - 1),
        jnp.clip(face_j, 0, y_positions.shape[1] - 1),
        jnp.clip(face_k, 0, y_positions.shape[2] - 1),
    ]
    z_pos = z_positions[
        jnp.clip(face_i, 0, z_positions.shape[0] - 1),
        jnp.clip(face_j, 0, z_positions.shape[1] - 1),
        jnp.clip(face_k, 0, z_positions.shape[2] - 1),
    ]
    return jnp.where(
        face_axis[:, None] == 0,
        x_pos,
        jnp.where(face_axis[:, None] == 1, y_pos, z_pos),
    )


def _regular_face_row_face_metric_values(
    geometry: LocalFciGeometry3D,
    rows: LocalRegularFaceContributionRows3D,
    *,
    b_floor: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    face_axis = jnp.asarray(rows.face_axis, dtype=jnp.int32)
    face_i = jnp.asarray(rows.face_i, dtype=jnp.int32)
    face_j = jnp.asarray(rows.face_j, dtype=jnp.int32)
    face_k = jnp.asarray(rows.face_k, dtype=jnp.int32)

    def _axis_values(axis: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        metric = geometry.face_metric.axes[axis]
        bfield = geometry.face_bfield.axes[axis]
        J = jnp.asarray(metric.J_owned, dtype=jnp.float64)
        B_contra = jnp.asarray(bfield.B_contra_owned, dtype=jnp.float64)
        Bmag = jnp.maximum(jnp.asarray(bfield.Bmag_owned, dtype=jnp.float64), float(b_floor))
        return (
            J[
                jnp.clip(face_i, 0, J.shape[0] - 1),
                jnp.clip(face_j, 0, J.shape[1] - 1),
                jnp.clip(face_k, 0, J.shape[2] - 1),
            ],
            (B_contra[..., axis] / Bmag)[
                jnp.clip(face_i, 0, Bmag.shape[0] - 1),
                jnp.clip(face_j, 0, Bmag.shape[1] - 1),
                jnp.clip(face_k, 0, Bmag.shape[2] - 1),
            ],
        )

    x_J, x_b = _axis_values(0)
    y_J, y_b = _axis_values(1)
    z_J, z_b = _axis_values(2)
    J = jnp.where(face_axis == 0, x_J, jnp.where(face_axis == 1, y_J, z_J))
    b_axis = jnp.where(face_axis == 0, x_b, jnp.where(face_axis == 1, y_b, z_b))
    return J, b_axis


def _regular_face_row_projector_values(
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    rows: LocalRegularFaceContributionRows3D,
) -> jnp.ndarray:
    face_axis = jnp.asarray(rows.face_axis, dtype=jnp.int32)
    face_i = jnp.asarray(rows.face_i, dtype=jnp.int32)
    face_j = jnp.asarray(rows.face_j, dtype=jnp.int32)
    face_k = jnp.asarray(rows.face_k, dtype=jnp.int32)
    x_projector, y_projector, z_projector = face_projectors
    x_value = x_projector[
        jnp.clip(face_i, 0, x_projector.shape[0] - 1),
        jnp.clip(face_j, 0, x_projector.shape[1] - 1),
        jnp.clip(face_k, 0, x_projector.shape[2] - 1),
        0,
        :,
    ]
    y_value = y_projector[
        jnp.clip(face_i, 0, y_projector.shape[0] - 1),
        jnp.clip(face_j, 0, y_projector.shape[1] - 1),
        jnp.clip(face_k, 0, y_projector.shape[2] - 1),
        1,
        :,
    ]
    z_value = z_projector[
        jnp.clip(face_i, 0, z_projector.shape[0] - 1),
        jnp.clip(face_j, 0, z_projector.shape[1] - 1),
        jnp.clip(face_k, 0, z_projector.shape[2] - 1),
        2,
        :,
    ]
    return jnp.where(
        face_axis[:, None] == 0,
        x_value,
        jnp.where(face_axis[:, None] == 1, y_value, z_value),
    )


def _regular_face_row_owner_payload(
    values_owned: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    rows: LocalRegularFaceContributionRows3D,
    aggregate_geometry: LocalAggregateCellGeometry3D | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    shape = geometry.owned_shape
    minus_i = jnp.clip(jnp.asarray(rows.minus_owner_i, dtype=jnp.int32), 0, shape[0] - 1)
    minus_j = jnp.clip(jnp.asarray(rows.minus_owner_j, dtype=jnp.int32), 0, shape[1] - 1)
    minus_k = jnp.clip(jnp.asarray(rows.minus_owner_k, dtype=jnp.int32), 0, shape[2] - 1)
    plus_i = jnp.clip(jnp.asarray(rows.plus_owner_i, dtype=jnp.int32), 0, shape[0] - 1)
    plus_j = jnp.clip(jnp.asarray(rows.plus_owner_j, dtype=jnp.int32), 0, shape[1] - 1)
    plus_k = jnp.clip(jnp.asarray(rows.plus_owner_k, dtype=jnp.int32), 0, shape[2] - 1)

    values = jnp.asarray(values_owned, dtype=jnp.float64)
    active = _active_cell_mask_owned(geometry)
    if aggregate_geometry is None:
        positions = _owned_cell_logical_positions(geometry)
    else:
        positions = jnp.asarray(aggregate_geometry.centroid, dtype=jnp.float64)

    minus_value = values[minus_i, minus_j, minus_k]
    plus_value = values[plus_i, plus_j, plus_k]
    minus_position = positions[minus_i, minus_j, minus_k]
    plus_position = positions[plus_i, plus_j, plus_k]
    owner_valid = (
        active[minus_i, minus_j, minus_k]
        & active[plus_i, plus_j, plus_k]
        & jnp.isfinite(minus_value)
        & jnp.isfinite(plus_value)
    )
    return minus_value, plus_value, minus_position, plus_position, owner_valid


def _build_regular_face_contribution_parallel_flux(
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    regular_face_geometry: LocalRegularFaceGeometry3D,
    rows: LocalRegularFaceContributionRows3D,
    *,
    aggregate_geometry: LocalAggregateCellGeometry3D | None = None,
    cell_gradient: LocalCellGradient3D | None = None,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray | None:
    if int(rows.max_rows) == 0:
        return None
    values_owned = jnp.asarray(local.x.center, dtype=jnp.float64)
    minus_value, plus_value, minus_position, plus_position, owner_valid = (
        _regular_face_row_owner_payload(values_owned, geometry, rows, aggregate_geometry)
    )
    face_position = _regular_face_row_positions(geometry, regular_face_geometry, rows)
    if cell_gradient is not None:
        gradient = jnp.asarray(cell_gradient.gradient, dtype=jnp.float64)
        shape = geometry.owned_shape
        minus_i = jnp.clip(jnp.asarray(rows.minus_owner_i, dtype=jnp.int32), 0, shape[0] - 1)
        minus_j = jnp.clip(jnp.asarray(rows.minus_owner_j, dtype=jnp.int32), 0, shape[1] - 1)
        minus_k = jnp.clip(jnp.asarray(rows.minus_owner_k, dtype=jnp.int32), 0, shape[2] - 1)
        plus_i = jnp.clip(jnp.asarray(rows.plus_owner_i, dtype=jnp.int32), 0, shape[0] - 1)
        plus_j = jnp.clip(jnp.asarray(rows.plus_owner_j, dtype=jnp.int32), 0, shape[1] - 1)
        plus_k = jnp.clip(jnp.asarray(rows.plus_owner_k, dtype=jnp.int32), 0, shape[2] - 1)
        minus_gradient = gradient[minus_i, minus_j, minus_k]
        plus_gradient = gradient[plus_i, plus_j, plus_k]
        minus_value = minus_value + jnp.einsum(
            "...i,...i->...",
            minus_gradient,
            face_position - minus_position,
        )
        plus_value = plus_value + jnp.einsum(
            "...i,...i->...",
            plus_gradient,
            face_position - plus_position,
        )
        owner_valid = owner_valid & jnp.all(jnp.isfinite(minus_gradient), axis=-1) & jnp.all(
            jnp.isfinite(plus_gradient),
            axis=-1,
        )
    face_value = 0.5 * (minus_value + plus_value)
    J, b_axis = _regular_face_row_face_metric_values(geometry, rows, b_floor=b_floor)
    row_flux = J * b_axis * face_value
    valid = (
        jnp.asarray(rows.active, dtype=bool)
        & jnp.asarray(rows.use_reconstructed_flux, dtype=bool)
        & owner_valid
        & jnp.isfinite(row_flux)
    )
    return jnp.where(valid, row_flux, 0.0)


def _build_regular_face_contribution_projected_flux(
    values_owned: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    rows: LocalRegularFaceContributionRows3D,
    *,
    aggregate_geometry: LocalAggregateCellGeometry3D | None = None,
    cell_gradient: LocalCellGradient3D | None = None,
) -> jnp.ndarray | None:
    if int(rows.max_rows) == 0 or cell_gradient is None:
        return None
    _minus_value, _plus_value, _minus_position, _plus_position, owner_valid = (
        _regular_face_row_owner_payload(values_owned, geometry, rows, aggregate_geometry)
    )
    shape = geometry.owned_shape
    minus_i = jnp.clip(jnp.asarray(rows.minus_owner_i, dtype=jnp.int32), 0, shape[0] - 1)
    minus_j = jnp.clip(jnp.asarray(rows.minus_owner_j, dtype=jnp.int32), 0, shape[1] - 1)
    minus_k = jnp.clip(jnp.asarray(rows.minus_owner_k, dtype=jnp.int32), 0, shape[2] - 1)
    plus_i = jnp.clip(jnp.asarray(rows.plus_owner_i, dtype=jnp.int32), 0, shape[0] - 1)
    plus_j = jnp.clip(jnp.asarray(rows.plus_owner_j, dtype=jnp.int32), 0, shape[1] - 1)
    plus_k = jnp.clip(jnp.asarray(rows.plus_owner_k, dtype=jnp.int32), 0, shape[2] - 1)
    gradient = jnp.asarray(cell_gradient.gradient, dtype=jnp.float64)
    row_gradient = 0.5 * (
        gradient[minus_i, minus_j, minus_k]
        + gradient[plus_i, plus_j, plus_k]
    )
    row_projector = _regular_face_row_projector_values(face_projectors, rows)
    J, _b_axis = _regular_face_row_face_metric_values(geometry, rows, b_floor=1.0)
    row_flux = J * jnp.einsum("...i,...i->...", row_projector, row_gradient)
    valid = (
        jnp.asarray(rows.active, dtype=bool)
        & jnp.asarray(rows.use_reconstructed_flux, dtype=bool)
        & owner_valid
        & jnp.all(jnp.isfinite(row_gradient), axis=-1)
        & jnp.isfinite(row_flux)
    )
    return jnp.where(valid, row_flux, 0.0)


def _corrected_dirichlet_wall_normal_gradient(
    *,
    cut_wall_value: jnp.ndarray,
    f_cell: jnp.ndarray,
    grad_tangent: jnp.ndarray,
    wall_center: jnp.ndarray,
    owner_center: jnp.ndarray,
    normal_contra: jnp.ndarray,
    normal_cov: jnp.ndarray,
    fallback_distance: jnp.ndarray,
) -> jnp.ndarray:
    """Dirichlet normal gradient with tangential owner-to-wall jump removed."""

    delta = jnp.asarray(wall_center, dtype=jnp.float64) - jnp.asarray(
        owner_center,
        dtype=jnp.float64,
    )
    normal_delta = jnp.einsum("...i,...i->...", normal_cov, delta)
    delta_tangent = delta - normal_delta[..., None] * normal_contra
    tangent_jump = jnp.einsum("...i,...i->...", grad_tangent, delta_tangent)

    safe_normal_delta = jnp.where(
        jnp.abs(normal_delta) > 1.0e-30,
        normal_delta,
        jnp.sign(normal_delta) * 1.0e-30,
    )
    safe_normal_delta = jnp.where(normal_delta == 0.0, 1.0e-30, safe_normal_delta)
    corrected = (cut_wall_value - f_cell - tangent_jump) / safe_normal_delta

    safe_distance = jnp.maximum(jnp.abs(fallback_distance), 1.0e-30)
    fallback = (cut_wall_value - f_cell) / safe_distance
    return jnp.where(jnp.abs(normal_delta) > 1.0e-30, corrected, fallback)


def local_parallel_flux_div_op(
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    face_bc: LocalBoundaryFaceBC3D | None = None,
    boundary_trace: LocalBoundaryFaceTrace3D | None = None,
    regular_face_geometry: LocalRegularFaceGeometry3D | None = None,
    regular_face_contribution_rows: LocalRegularFaceContributionRows3D | None = None,
    cell_volume: LocalCellVolumeGeometry3D | None = None,
    cut_wall_geometry: LocalCutWallGeometry3D | None = None,
    cut_wall_bc: LocalCutWallBC3D | None = None,
    cell_gradient: LocalCellGradient3D | None = None,
    aggregate_geometry: LocalAggregateCellGeometry3D | None = None,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
    field_closure: LocalControlVolumeFieldClosure3D | None = None,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the local conservative parallel flux divergence ``∇·(f b)``."""

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_parallel_flux_div_op requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    if not isinstance(domain, LocalDomain3D):
        raise TypeError(
            "local_parallel_flux_div_op requires LocalDomain3D, "
            f"got {type(domain).__name__}"
        )
    if not isinstance(local, ConservativeStencil3D):
        raise TypeError(
            "local_parallel_flux_div_op requires ConservativeStencil3D, "
            f"got {type(local).__name__}"
        )
    if geometry.layout != domain.layout:
        raise ValueError("geometry and domain must share the same HaloLayout3D")
    if local.shape != geometry.owned_shape:
        raise ValueError(
            f"local stencil must have shape {geometry.owned_shape}, got {local.shape}"
        )

    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    if len(axis_regular_axes) != 3:
        raise ValueError("axis_regular_axes must have length 3")
    if axis_regular_axes[1] or axis_regular_axes[2]:
        raise NotImplementedError(
            "axis_regular_axes currently only supports the lower x axis; "
            f"got axis_regular_axes={axis_regular_axes}"
        )

    regular_face_geometry = regular_face_geometry or geometry.regular_face_geometry
    cell_volume = cell_volume or geometry.cell_volume_geometry
    face_bc = face_bc or LocalBoundaryFaceBC3D.empty(geometry.layout)
    boundary_trace = _validate_local_boundary_face_trace(boundary_trace, geometry.layout)
    if cut_wall_geometry is None and cut_wall_bc is None:
        cut_wall_geometry = LocalCutWallGeometry3D.empty(0)
        cut_wall_bc = LocalCutWallBC3D.empty(0)
    elif cut_wall_geometry is None:
        cut_wall_geometry = LocalCutWallGeometry3D.empty(cut_wall_bc.n_wall_faces)
    elif cut_wall_bc is None:
        cut_wall_bc = LocalCutWallBC3D.empty(cut_wall_geometry.max_wall_faces)

    x_face_value = _apply_local_face_value_dirichlet_bc(
        local.face_values.x,
        axis=0,
        axis_kind=face_bc.kind_x,
        axis_value=face_bc.value_x,
        axis_mask=face_bc.mask_x,
        axis_regular_axes=axis_regular_axes,
    )
    y_face_value = _apply_local_face_value_dirichlet_bc(
        local.face_values.y,
        axis=1,
        axis_kind=face_bc.kind_y,
        axis_value=face_bc.value_y,
        axis_mask=face_bc.mask_y,
        axis_regular_axes=axis_regular_axes,
    )
    z_face_value = _apply_local_face_value_dirichlet_bc(
        local.face_values.z,
        axis=2,
        axis_kind=face_bc.kind_z,
        axis_value=face_bc.value_z,
        axis_mask=face_bc.mask_z,
        axis_regular_axes=axis_regular_axes,
    )
    if boundary_trace is not None:
        x_face_value = _apply_local_face_trace(
            x_face_value, axis=0, trace_value=boundary_trace.value_x,
            trace_mask=boundary_trace.mask_x, axis_regular_axes=axis_regular_axes,
        )
        y_face_value = _apply_local_face_trace(
            y_face_value, axis=1, trace_value=boundary_trace.value_y,
            trace_mask=boundary_trace.mask_y, axis_regular_axes=axis_regular_axes,
        )
        z_face_value = _apply_local_face_trace(
            z_face_value, axis=2, trace_value=boundary_trace.value_z,
            trace_mask=boundary_trace.mask_z, axis_regular_axes=axis_regular_axes,
        )

    def _unit_b_axis(bfield: LocalBFieldGeometry, axis: int) -> jnp.ndarray:
        B_contra = jnp.asarray(bfield.B_contra_owned, dtype=jnp.float64)
        Bmag = jnp.maximum(
            jnp.asarray(bfield.Bmag_owned, dtype=jnp.float64),
            float(b_floor),
        )
        return B_contra[..., axis] / Bmag

    x_flux = (
        jnp.asarray(geometry.face_metric.x.J_owned, dtype=jnp.float64)
        * _unit_b_axis(geometry.face_bfield.x, 0)
        * x_face_value
    )
    if axis_regular_axes[0]:
        do_axis_lower = domain.runtime_has_axis_regular_lower(0)
        lower = jnp.where(do_axis_lower, jnp.zeros_like(x_flux[0]), x_flux[0])
        x_flux = x_flux.at[0].set(lower)
    y_flux = (
        jnp.asarray(geometry.face_metric.y.J_owned, dtype=jnp.float64)
        * _unit_b_axis(geometry.face_bfield.y, 1)
        * y_face_value
    )
    z_flux = (
        jnp.asarray(geometry.face_metric.z.J_owned, dtype=jnp.float64)
        * _unit_b_axis(geometry.face_bfield.z, 2)
        * z_face_value
    )

    x_flux = _apply_local_face_flux_bc(
        x_flux,
        axis=0,
        axis_kind=face_bc.kind_x,
        axis_value=face_bc.value_x,
        axis_mask=face_bc.mask_x,
        axis_regular_axes=axis_regular_axes,
    )
    y_flux = _apply_local_face_flux_bc(
        y_flux,
        axis=1,
        axis_kind=face_bc.kind_y,
        axis_value=face_bc.value_y,
        axis_mask=face_bc.mask_y,
        axis_regular_axes=axis_regular_axes,
    )
    z_flux = _apply_local_face_flux_bc(
        z_flux,
        axis=2,
        axis_kind=face_bc.kind_z,
        axis_value=face_bc.value_z,
        axis_mask=face_bc.mask_z,
        axis_regular_axes=axis_regular_axes,
    )

    if control_volume_geometry is not None:
        if not isinstance(
            control_volume_geometry,
            LocalEmbeddedControlVolumeGeometry3D,
        ):
            raise TypeError(
                "control_volume_geometry must be "
                "LocalEmbeddedControlVolumeGeometry3D or None"
            )
        field_closure = _require_local_control_volume_field_closure(
            field_closure,
            control_volume_geometry,
        )
        return _local_control_volume_integrated_divergence(
            (x_flux, y_flux, z_flux),
            field_closure.parallel_flux,
            geometry,
            domain,
            control_volume_geometry,
            volume_floor=jacobian_floor,
        )

    cut_wall_flux = _build_local_parallel_flux_cut_wall_payload(
        local=local,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        b_floor=b_floor,
    )
    regular_face_contribution_flux = None
    if regular_face_contribution_rows is not None:
        regular_face_contribution_flux = _build_regular_face_contribution_parallel_flux(
            local,
            geometry,
            regular_face_geometry,
            regular_face_contribution_rows,
            aggregate_geometry=aggregate_geometry,
            cell_gradient=cell_gradient,
            b_floor=b_floor,
        )

    cv_flux = LocalControlVolumeFluxStencil3D(
        regular_flux=FaceFluxStencil3D(x=x_flux, y=y_flux, z=z_flux),
        regular_face_geometry=regular_face_geometry,
        cell_volume=cell_volume,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_flux=cut_wall_flux,
        regular_face_contribution_rows=regular_face_contribution_rows,
        regular_face_contribution_flux=regular_face_contribution_flux,
    )
    return local_divergence_conservative_op(
        cv_flux,
        geometry,
        jacobian_floor=jacobian_floor,
    )


def local_parallel_div_b_op(
    unit_stencil: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    regular_face_geometry: LocalRegularFaceGeometry3D | None = None,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
    field_closure: LocalControlVolumeFieldClosure3D | None = None,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return ``div(b)`` using the regular conservative face-flux path.

    ``unit_stencil`` must be built from a halo field containing one on every
    cell.  No field boundary condition is applied: this is geometry-only
    ``div(b)`` on regular cells and is intentionally not a control-volume or
    cut-wall operator.  The result can be cached and reused for every scalar
    compatible gradient in one RHS stage.
    """

    if not isinstance(unit_stencil, ConservativeStencil3D):
        raise TypeError(
            "local_parallel_div_b_op requires ConservativeStencil3D, "
            f"got {type(unit_stencil).__name__}"
        )
    return local_parallel_flux_div_op(
        unit_stencil,
        geometry,
        domain,
        regular_face_geometry=regular_face_geometry,
        control_volume_geometry=control_volume_geometry,
        field_closure=field_closure,
        axis_regular_axes=axis_regular_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
    )


def local_grad_parallel_op_conservative(
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    div_b: jnp.ndarray,
    boundary_trace: LocalBoundaryFaceTrace3D | None = None,
    regular_face_geometry: LocalRegularFaceGeometry3D | None = None,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
    field_closure: LocalControlVolumeFieldClosure3D | None = None,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Compute compatible regular-cell ``grad_parallel(f)``.

    The discretization is

        ``grad_parallel(f) = div(f b) - f div(b)``.

    Both divergences use :func:`local_parallel_flux_div_op`, so the scalar
    gradient is compatible with the conservative parallel flux operator on
    nonuniform geometry.  ``div_b`` should be computed once from
    :func:`local_parallel_div_b_op` using a unit-valued stencil.  In
    control-volume mode both divergences use the supplied direct closure.
    """

    if not isinstance(local, ConservativeStencil3D):
        raise TypeError(
            "local_grad_parallel_op_conservative requires ConservativeStencil3D, "
            f"got {type(local).__name__}"
        )
    if local.shape != geometry.owned_shape:
        raise ValueError(
            f"local stencil must have shape {geometry.owned_shape}, got {local.shape}"
        )
    div_b = jnp.asarray(div_b, dtype=jnp.float64)
    if div_b.shape != geometry.owned_shape:
        raise ValueError(
            f"div_b must have shape {geometry.owned_shape}, got {div_b.shape}"
        )
    field = jnp.asarray(local.x.center, dtype=jnp.float64)
    div_fb = local_parallel_flux_div_op(
        local,
        geometry,
        domain,
        regular_face_geometry=regular_face_geometry,
        control_volume_geometry=control_volume_geometry,
        field_closure=field_closure,
        boundary_trace=boundary_trace,
        axis_regular_axes=axis_regular_axes,
        b_floor=b_floor,
        jacobian_floor=jacobian_floor,
    )
    return _mask_inactive_owned(div_fb - field * div_b, geometry)


def _build_local_cut_wall_flux_payload(
    *,
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    cut_wall_geometry: LocalCutWallGeometry3D,
    cut_wall_bc: LocalCutWallBC3D,
    cell_gradient: LocalCellGradient3D | None = None,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Build the padded embedded-wall flux payload for the conservative control volume."""

    if not isinstance(cut_wall_geometry, LocalCutWallGeometry3D):
        raise TypeError(
            "_build_local_cut_wall_flux_payload requires LocalCutWallGeometry3D, "
            f"got {type(cut_wall_geometry).__name__}"
        )
    if not isinstance(cut_wall_bc, LocalCutWallBC3D):
        raise TypeError(
            "_build_local_cut_wall_flux_payload requires LocalCutWallBC3D, "
            f"got {type(cut_wall_bc).__name__}"
        )
    if cut_wall_geometry.max_wall_faces != cut_wall_bc.max_wall_faces:
        raise ValueError(
            "cut_wall_geometry and cut_wall_bc must use the same padded wall-face length"
        )

    max_wall_faces = int(cut_wall_geometry.max_wall_faces)
    if max_wall_faces == 0:
        return jnp.zeros((0,), dtype=jnp.float64)

    active = jnp.asarray(cut_wall_geometry.active, dtype=bool) & jnp.asarray(cut_wall_bc.active, dtype=bool)
    cut_wall_kind = jnp.asarray(cut_wall_bc.kind, dtype=jnp.int32)
    cut_wall_value = jnp.asarray(cut_wall_bc.value, dtype=jnp.float64)

    supported = (
        (cut_wall_kind == BC_NONE)
        | (cut_wall_kind == BC_DIRICHLET)
        | (cut_wall_kind == BC_NEUMANN)
        | (cut_wall_kind == BC_NORMALFLUX)
        | (cut_wall_kind == BC_NOFLUX)
    )
    active = active & supported
    cut_wall_kind = jnp.where(supported, cut_wall_kind, BC_NONE)
    cut_wall_value = jnp.where(supported, cut_wall_value, 0.0)

    owner_i = jnp.asarray(cut_wall_geometry.owner_i, dtype=jnp.int32)
    owner_j = jnp.asarray(cut_wall_geometry.owner_j, dtype=jnp.int32)
    owner_k = jnp.asarray(cut_wall_geometry.owner_k, dtype=jnp.int32)

    field = jnp.asarray(local.x.center, dtype=jnp.float64)
    dfdx_cell = _take_stencil_finite_difference(local.x)
    dfdy_cell = _take_stencil_finite_difference(local.y)
    dfdz_cell = _take_stencil_finite_difference(local.z)

    raw_grad_cell = jnp.stack(
        (
            dfdx_cell[owner_i, owner_j, owner_k],
            dfdy_cell[owner_i, owner_j, owner_k],
            dfdz_cell[owner_i, owner_j, owner_k],
        ),
        axis=-1,
    )
    if cell_gradient is not None:
        if not isinstance(cell_gradient, LocalCellGradient3D):
            raise TypeError(
                "cell_gradient must be a LocalCellGradient3D or None, "
                f"got {type(cell_gradient).__name__}"
            )
        if cell_gradient.shape != geometry.owned_shape:
            raise ValueError(
                f"cell_gradient must have shape {geometry.owned_shape}, "
                f"got {cell_gradient.shape}"
            )
        repaired_grad_cell = jnp.asarray(cell_gradient.gradient, dtype=jnp.float64)[
            owner_i,
            owner_j,
            owner_k,
        ]
        repaired_valid = jnp.asarray(cell_gradient.valid, dtype=bool)[
            owner_i,
            owner_j,
            owner_k,
        ]
        grad_cell = jnp.where(repaired_valid[..., None], repaired_grad_cell, raw_grad_cell)
    else:
        grad_cell = raw_grad_cell
    f_cell = field[owner_i, owner_j, owner_k]
    owner_center = jnp.stack(
        (
            jnp.asarray(geometry.grid.x.centers_owned, dtype=jnp.float64)[owner_i],
            jnp.asarray(geometry.grid.y.centers_owned, dtype=jnp.float64)[owner_j],
            jnp.asarray(geometry.grid.z.centers_owned, dtype=jnp.float64)[owner_k],
        ),
        axis=-1,
    )

    normal_contra = jnp.asarray(cut_wall_geometry.normal_contra, dtype=jnp.float64)
    normal_cov = jnp.einsum(
        "...ij,...j->...i",
        jnp.asarray(cut_wall_geometry.g_cov, dtype=jnp.float64),
        normal_contra,
    )
    g_cell = jnp.einsum("...i,...i->...", normal_contra, grad_cell)
    grad_tangent = grad_cell - g_cell[..., None] * normal_cov

    distance = jnp.asarray(cut_wall_geometry.distance, dtype=jnp.float64)
    g_dirichlet = _corrected_dirichlet_wall_normal_gradient(
        cut_wall_value=cut_wall_value,
        f_cell=f_cell,
        grad_tangent=grad_tangent,
        wall_center=jnp.asarray(cut_wall_geometry.center, dtype=jnp.float64),
        owner_center=owner_center,
        normal_contra=normal_contra,
        normal_cov=normal_cov,
        fallback_distance=distance,
    )
    g_neumann = cut_wall_value
    g_wall = g_cell
    g_wall = jnp.where(cut_wall_kind == BC_DIRICHLET, g_dirichlet, g_wall)
    g_wall = jnp.where(cut_wall_kind == BC_NEUMANN, g_neumann, g_wall)
    grad_wall = grad_tangent + g_wall[..., None] * normal_cov

    bmag = jnp.maximum(jnp.asarray(cut_wall_geometry.Bmag, dtype=jnp.float64), float(b_floor))
    b_wall = jnp.asarray(cut_wall_geometry.B_contra, dtype=jnp.float64) / bmag[..., None]
    projector = jnp.asarray(cut_wall_geometry.g_contra, dtype=jnp.float64) - jnp.einsum(
        "...i,...j->...ij",
        b_wall,
        b_wall,
    )
    wall_flux_area = jnp.asarray(cut_wall_geometry.J, dtype=jnp.float64) * jnp.einsum(
        "...i,...ij,...j->...",
        jnp.asarray(cut_wall_geometry.area_covector, dtype=jnp.float64),
        projector,
        grad_wall,
    )
    wall_flux_area = jnp.where(cut_wall_kind == BC_NORMALFLUX, cut_wall_value, wall_flux_area)
    wall_flux_area = jnp.where(cut_wall_kind == BC_NOFLUX, 0.0, wall_flux_area)
    wall_flux_area = jnp.where(active, wall_flux_area, 0.0)

    sign = jnp.asarray(cut_wall_geometry.sign, dtype=jnp.float64)
    if sign.shape != wall_flux_area.shape:
        raise ValueError(
            f"cut_wall_geometry.sign must have shape {wall_flux_area.shape}, got {sign.shape}"
        )
    return sign * wall_flux_area


def _shift_bool_mask(mask: jnp.ndarray, *, axis: int, offset: int, periodic: bool) -> jnp.ndarray:
    """Shift a cell mask by one index without wrapping on nonperiodic axes."""

    if offset == 0:
        return mask
    if periodic:
        return jnp.roll(mask, offset, axis=axis)

    shifted = jnp.zeros_like(mask, dtype=bool)
    if offset > 0:
        src = _axis_slice_nd(axis, None, -offset, mask.ndim)
        dst = _axis_slice_nd(axis, offset, None, mask.ndim)
    else:
        src = _axis_slice_nd(axis, -offset, None, mask.ndim)
        dst = _axis_slice_nd(axis, None, offset, mask.ndim)
    return shifted.at[dst].set(mask[src])


def _cut_wall_owner_cell_mask(
    geometry: LocalFciGeometry3D,
    *,
    cut_wall_geometry: LocalCutWallGeometry3D,
    cut_wall_bc: LocalCutWallBC3D,
) -> jnp.ndarray:
    """Owned-cell mask for cells carrying active cut-wall coordinate legs."""

    if int(cut_wall_geometry.max_wall_faces) == 0:
        return jnp.zeros(geometry.owned_shape, dtype=bool)

    active = (
        jnp.asarray(cut_wall_geometry.active, dtype=bool)
        & jnp.asarray(cut_wall_bc.active, dtype=bool)
    )
    owner_i = jnp.clip(
        jnp.asarray(cut_wall_geometry.owner_i, dtype=jnp.int32),
        0,
        geometry.owned_shape[0] - 1,
    )
    owner_j = jnp.clip(
        jnp.asarray(cut_wall_geometry.owner_j, dtype=jnp.int32),
        0,
        geometry.owned_shape[1] - 1,
    )
    owner_k = jnp.clip(
        jnp.asarray(cut_wall_geometry.owner_k, dtype=jnp.int32),
        0,
        geometry.owned_shape[2] - 1,
    )
    count = jnp.zeros(geometry.owned_shape, dtype=jnp.int32).at[
        owner_i,
        owner_j,
        owner_k,
    ].add(active.astype(jnp.int32))
    return count > 0


def _dilate_cut_wall_owner_mask_for_face_axis(
    owner_mask: jnp.ndarray,
    *,
    face_axis: int,
    periodic_axes: tuple[bool, bool, bool],
) -> jnp.ndarray:
    """Dilate owner cells in directions tangential to a face-gradient location."""

    result = jnp.asarray(owner_mask, dtype=bool)
    for axis in range(3):
        if axis == face_axis:
            continue
        result = (
            result
            | _shift_bool_mask(result, axis=axis, offset=1, periodic=periodic_axes[axis])
            | _shift_bool_mask(result, axis=axis, offset=-1, periodic=periodic_axes[axis])
        )
    return result


def _average_cell_gradients_to_faces(
    cell_grad: jnp.ndarray,
    *,
    face_axis: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Average owned cell gradients onto the face grid for ``face_axis``."""

    face_shape = list(cell_grad.shape[:3])
    face_shape[face_axis] += 1
    face_grad_sum = jnp.zeros(tuple(face_shape) + (3,), dtype=jnp.float64)
    face_grad_count = jnp.zeros(tuple(face_shape), dtype=jnp.float64)
    ones = jnp.ones(cell_grad.shape[:3], dtype=jnp.float64)
    n = int(cell_grad.shape[face_axis])

    lower_faces = _axis_slice_nd(face_axis, 0, n, 3)
    upper_faces = _axis_slice_nd(face_axis, 1, n + 1, 3)
    face_grad_sum = face_grad_sum.at[lower_faces + (slice(None),)].add(cell_grad)
    face_grad_sum = face_grad_sum.at[upper_faces + (slice(None),)].add(cell_grad)
    face_grad_count = face_grad_count.at[lower_faces].add(ones)
    face_grad_count = face_grad_count.at[upper_faces].add(ones)
    averaged = face_grad_sum / jnp.maximum(face_grad_count[..., None], 1.0)
    return averaged, face_grad_count


def _cell_mask_to_adjacent_face_mask(
    cell_mask: jnp.ndarray,
    *,
    face_axis: int,
) -> jnp.ndarray:
    """Mark faces adjacent to any masked cell."""

    face_shape = list(cell_mask.shape)
    face_shape[face_axis] += 1
    face_count = jnp.zeros(tuple(face_shape), dtype=jnp.int32)
    n = int(cell_mask.shape[face_axis])
    lower_faces = _axis_slice_nd(face_axis, 0, n, 3)
    upper_faces = _axis_slice_nd(face_axis, 1, n + 1, 3)
    cell_count = jnp.asarray(cell_mask, dtype=jnp.int32)
    face_count = face_count.at[lower_faces].add(cell_count)
    face_count = face_count.at[upper_faces].add(cell_count)
    return face_count > 0


def _shift_cell_array(
    values: jnp.ndarray,
    *,
    axis: int,
    offset: int,
    periodic: bool,
    fill_value: float | bool = 0.0,
) -> jnp.ndarray:
    """Shift an owned-cell array without wrapping on nonperiodic axes."""

    if offset == 0:
        return values
    if periodic:
        return jnp.roll(values, offset, axis=axis)

    shifted = jnp.full_like(values, fill_value)
    if offset > 0:
        src = _axis_slice_nd(axis, None, -offset, values.ndim)
        dst = _axis_slice_nd(axis, offset, None, values.ndim)
    else:
        src = _axis_slice_nd(axis, -offset, None, values.ndim)
        dst = _axis_slice_nd(axis, None, offset, values.ndim)
    return shifted.at[dst].set(values[src])


def _shift_cell_sample(
    value: jnp.ndarray,
    position: jnp.ndarray,
    valid: jnp.ndarray,
    *,
    shifts: tuple[int, int, int],
    periodic_axes: tuple[bool, bool, bool],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Shift cell samples and their validity mask by a static 3D offset."""

    shifted_value = value
    shifted_position = position
    shifted_valid = valid
    for axis, offset in enumerate(shifts):
        if int(offset) == 0:
            continue
        shifted_value = _shift_cell_array(
            shifted_value,
            axis=axis,
            offset=int(offset),
            periodic=periodic_axes[axis],
            fill_value=0.0,
        )
        shifted_position = _shift_cell_array(
            shifted_position,
            axis=axis,
            offset=int(offset),
            periodic=periodic_axes[axis],
            fill_value=0.0,
        )
        shifted_valid = _shift_cell_array(
            shifted_valid,
            axis=axis,
            offset=int(offset),
            periodic=periodic_axes[axis],
            fill_value=False,
        )
    return shifted_value, shifted_position, shifted_valid


def _owned_cell_logical_positions(geometry: LocalFciGeometry3D) -> jnp.ndarray:
    """Owned cell-center logical coordinates with shape ``owned_shape + (3,)``."""

    x = jnp.asarray(geometry.grid.x.centers_owned, dtype=jnp.float64)
    y = jnp.asarray(geometry.grid.y.centers_owned, dtype=jnp.float64)
    z = jnp.asarray(geometry.grid.z.centers_owned, dtype=jnp.float64)
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing="ij")
    return jnp.stack((xx, yy, zz), axis=-1)


def _owned_face_logical_positions(
    geometry: LocalFciGeometry3D,
    regular_face_geometry: LocalRegularFaceGeometry3D,
    *,
    face_axis: int,
) -> jnp.ndarray:
    """Open regular-face centroid logical coordinates for one face axis."""

    x_cells = jnp.asarray(geometry.grid.x.centers_owned, dtype=jnp.float64)
    y_cells = jnp.asarray(geometry.grid.y.centers_owned, dtype=jnp.float64)
    z_cells = jnp.asarray(geometry.grid.z.centers_owned, dtype=jnp.float64)
    x_faces = jnp.asarray(geometry.grid.x.faces_owned, dtype=jnp.float64)
    y_faces = jnp.asarray(geometry.grid.y.faces_owned, dtype=jnp.float64)
    z_faces = jnp.asarray(geometry.grid.z.faces_owned, dtype=jnp.float64)

    if face_axis == 0:
        xx, yy, zz = jnp.meshgrid(x_faces, y_cells, z_cells, indexing="ij")
        offset = regular_face_geometry.x_centroid_offset
    elif face_axis == 1:
        xx, yy, zz = jnp.meshgrid(x_cells, y_faces, z_cells, indexing="ij")
        offset = regular_face_geometry.y_centroid_offset
    else:
        xx, yy, zz = jnp.meshgrid(x_cells, y_cells, z_faces, indexing="ij")
        offset = regular_face_geometry.z_centroid_offset
    return jnp.stack((xx, yy, zz), axis=-1) + jnp.asarray(offset, dtype=jnp.float64)


def _cut_wall_face_gradient_sample_shifts(face_axis: int) -> tuple[tuple[int, int, int], ...]:
    """Static tangential sample offsets used for face-local reconstruction."""

    tangential_axes = tuple(axis for axis in range(3) if axis != face_axis)
    shifts: list[tuple[int, int, int]] = [(0, 0, 0)]
    for axis in tangential_axes:
        for offset in (-1, 1):
            current = [0, 0, 0]
            current[axis] = offset
            shifts.append(tuple(current))
    for offset_a in (-1, 1):
        for offset_b in (-1, 1):
            current = [0, 0, 0]
            current[tangential_axes[0]] = offset_a
            current[tangential_axes[1]] = offset_b
            shifts.append(tuple(current))
    return tuple(shifts)


def _accumulate_face_linear_sample(
    ata: jnp.ndarray,
    atb: jnp.ndarray,
    sample_count: jnp.ndarray,
    *,
    face_axis: int,
    face_positions: jnp.ndarray,
    sample_value: jnp.ndarray,
    sample_position: jnp.ndarray,
    sample_valid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Add one owned-cell sample field to both adjacent face-local fits."""

    n = int(sample_value.shape[face_axis])
    ones = jnp.ones_like(sample_value, dtype=jnp.float64)
    valid_weight = jnp.asarray(sample_valid, dtype=jnp.float64)
    sample_value = jnp.nan_to_num(
        jnp.asarray(sample_value, dtype=jnp.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    sample_position = jnp.asarray(sample_position, dtype=jnp.float64)
    for start, stop in ((0, n), (1, n + 1)):
        face_slice = _axis_slice_nd(face_axis, start, stop, 3)
        delta = sample_position - face_positions[face_slice + (slice(None),)]
        row = jnp.concatenate((ones[..., None], delta), axis=-1)
        weighted_row = valid_weight[..., None] * row
        ata_update = weighted_row[..., :, None] * row[..., None, :]
        atb_update = weighted_row * sample_value[..., None]
        ata = ata.at[face_slice + (slice(None), slice(None))].add(ata_update)
        atb = atb.at[face_slice + (slice(None),)].add(atb_update)
        sample_count = sample_count.at[face_slice].add(valid_weight)
    return ata, atb, sample_count


def _least_squares_cut_wall_face_gradient(
    field_owned: jnp.ndarray,
    *,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    regular_face_geometry: LocalRegularFaceGeometry3D,
    face_axis: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reconstruct face gradients from active-cell samples around a cut face.

    The fitted model is ``f = a + grad . (x - x_face_centroid)``.  The
    centroid is the open regular-face centroid, not the coordinate face center,
    so partial regular faces in edge/corner cut cells no longer project fluxes
    through the wrong point.  This intentionally targets only the cut-wall
    correction path; ordinary faces continue to use the original fast lifted
    face-gradient stencil.
    """

    face_shape = list(geometry.owned_shape)
    face_shape[face_axis] += 1
    face_shape_tuple = tuple(face_shape)
    ata = jnp.zeros(face_shape_tuple + (4, 4), dtype=jnp.float64)
    atb = jnp.zeros(face_shape_tuple + (4,), dtype=jnp.float64)
    sample_count = jnp.zeros(face_shape_tuple, dtype=jnp.float64)

    field = jnp.asarray(field_owned, dtype=jnp.float64)
    active = _active_cell_mask_owned(geometry) & jnp.isfinite(field)
    cell_positions = _owned_cell_logical_positions(geometry)
    face_positions = _owned_face_logical_positions(
        geometry,
        regular_face_geometry,
        face_axis=face_axis,
    )
    for shifts in _cut_wall_face_gradient_sample_shifts(face_axis):
        shifted_value, shifted_position, shifted_active = _shift_cell_sample(
            field,
            cell_positions,
            active,
            shifts=shifts,
            periodic_axes=domain.periodic_axes,
        )
        ata, atb, sample_count = _accumulate_face_linear_sample(
            ata,
            atb,
            sample_count,
            face_axis=face_axis,
            face_positions=face_positions,
            sample_value=shifted_value,
            sample_position=shifted_position,
            sample_valid=shifted_active,
        )

    eps = jnp.asarray(1.0e-14, dtype=jnp.float64)
    eye = jnp.eye(4, dtype=jnp.float64)
    coeff = jnp.linalg.solve(ata + eps * eye, atb[..., None])[..., 0]
    gradient = jnp.nan_to_num(coeff[..., 1:4], nan=0.0, posinf=0.0, neginf=0.0)
    valid = (sample_count >= 4.0) & jnp.all(jnp.isfinite(gradient), axis=-1)
    return gradient, valid


def _precompute_local_degree_two_reconstruction(
    cells: LocalControlVolumeCellGeometry3D,
    irregular_faces: LocalControlVolumeFaceRows3D,
    *,
    spacing_owned: jnp.ndarray | None = None,
    remote_sample_halo_indices: np.ndarray | None = None,
    remote_sample_centroids: np.ndarray | None = None,
    remote_sample_second_moments: np.ndarray | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
    coordinate_periods: tuple[float, float, float] | None = None,
    target_mask: jnp.ndarray | None = None,
    max_samples: int = 32,
    max_equations: int = 40,
    condition_limit: float = 1.0e4,
    svd_rcond: float = 1.0e-12,
) -> LocalMomentReconstruction3D:
    """Precompute finite-volume quadratic reconstruction transforms on the host.

    Neighborhood selection and rank-revealing SVD are intentionally outside
    JIT.  The returned transform maps field-dependent equation right-hand sides
    to gradient and Hessian coefficients with one batched matrix-vector product.
    """

    if not isinstance(cells, LocalControlVolumeCellGeometry3D):
        raise TypeError("cells must be a LocalControlVolumeCellGeometry3D")
    if not isinstance(irregular_faces, LocalControlVolumeFaceRows3D):
        raise TypeError("irregular_faces must be a LocalControlVolumeFaceRows3D")
    if cells.layout != irregular_faces.layout:
        raise ValueError("cells and irregular_faces must share one HaloLayout3D")
    max_samples = int(max_samples)
    max_equations = int(max_equations)
    if max_samples < 9:
        raise ValueError("max_samples must be at least 9 for quadratic reconstruction")
    if max_equations < max_samples:
        raise ValueError("max_equations must be at least max_samples")

    try:
        active_owner = np.asarray(cells.is_active_owner, dtype=bool)
        aggregate_target = np.asarray(cells.is_aggregate_target, dtype=bool)
        centroid = np.asarray(cells.centroid, dtype=np.float64)
        second_moment = np.asarray(cells.second_moment, dtype=np.float64)
        face_active = np.asarray(irregular_faces.active, dtype=bool)
        face_kind = np.asarray(irregular_faces.kind, dtype=np.int32)
        minus_owner = np.stack(
            (
                np.asarray(irregular_faces.minus_owner_i, dtype=np.int64),
                np.asarray(irregular_faces.minus_owner_j, dtype=np.int64),
                np.asarray(irregular_faces.minus_owner_k, dtype=np.int64),
            ),
            axis=-1,
        )
        plus_owner = np.stack(
            (
                np.asarray(irregular_faces.plus_owner_i, dtype=np.int64),
                np.asarray(irregular_faces.plus_owner_j, dtype=np.int64),
                np.asarray(irregular_faces.plus_owner_k, dtype=np.int64),
            ),
            axis=-1,
        )
        has_plus = np.asarray(irregular_faces.has_plus_owner, dtype=bool)
        has_remote = np.asarray(irregular_faces.has_remote_owner, dtype=bool)
        remote_centroid = np.asarray(
            irregular_faces.remote_centroid,
            dtype=np.float64,
        )
        remote_second_moment = np.asarray(
            irregular_faces.remote_second_moment,
            dtype=np.float64,
        )
        quadrature_points = np.asarray(
            irregular_faces.quadrature_points,
            dtype=np.float64,
        )
        area_weight = np.asarray(
            irregular_faces.area_covector_weight,
            dtype=np.float64,
        )
        quadrature_active = np.asarray(
            irregular_faces.quadrature_active,
            dtype=bool,
        )
        face_J = np.asarray(irregular_faces.J, dtype=np.float64)
    except (TypeError, jax.errors.TracerArrayConversionError) as exc:
        raise ValueError(
            "quadratic reconstruction metadata must be precomputed from concrete host arrays"
        ) from exc

    shape = cells.shape
    if spacing_owned is None:
        spacing = np.ones(shape + (3,), dtype=np.float64)
        for axis in range(3):
            coordinates = np.unique(centroid[..., axis][active_owner])
            differences = np.diff(np.sort(coordinates))
            differences = differences[differences > 1.0e-14]
            spacing[..., axis] = (
                float(np.median(differences)) if differences.size else 1.0
            )
    else:
        spacing = np.asarray(spacing_owned, dtype=np.float64)
        if spacing.shape != shape + (3,):
            raise ValueError(
                f"spacing_owned must have shape {shape + (3,)}, got {spacing.shape}"
            )
    spacing = np.maximum(np.abs(spacing), 1.0e-14)
    periodic_axes = tuple(bool(value) for value in periodic_axes)
    if len(periodic_axes) != 3:
        raise ValueError("periodic_axes must have length 3")
    if coordinate_periods is None:
        periods = np.ones((3,), dtype=np.float64)
    else:
        periods = np.asarray(coordinate_periods, dtype=np.float64)
        if periods.shape != (3,):
            raise ValueError("coordinate_periods must have length 3")
    for axis in range(3):
        if periodic_axes[axis] and (
            not np.isfinite(periods[axis]) or periods[axis] <= 0.0
        ):
            raise ValueError("periodic coordinate periods must be positive")

    def unwrap_displacement(displacement: np.ndarray) -> np.ndarray:
        result = np.asarray(displacement, dtype=np.float64).copy()
        for axis in range(3):
            if periodic_axes[axis]:
                result[..., axis] -= (
                    np.round(result[..., axis] / periods[axis])
                    * periods[axis]
                )
        return result
    if remote_sample_halo_indices is None:
        remote_halo_indices = np.zeros((0, 3), dtype=np.int32)
        remote_centroids = np.zeros((0, 3), dtype=np.float64)
        remote_second_moments = np.zeros((0, 3, 3), dtype=np.float64)
    else:
        remote_halo_indices = np.asarray(
            remote_sample_halo_indices,
            dtype=np.int32,
        )
        remote_centroids = np.asarray(
            remote_sample_centroids,
            dtype=np.float64,
        )
        remote_second_moments = np.asarray(
            remote_sample_second_moments,
            dtype=np.float64,
        )
        if remote_halo_indices.ndim != 2 or remote_halo_indices.shape[1] != 3:
            raise ValueError("remote_sample_halo_indices must have shape (n, 3)")
        if remote_centroids.shape != remote_halo_indices.shape:
            raise ValueError("remote_sample_centroids must have shape (n, 3)")
        if remote_second_moments.shape != (
            remote_halo_indices.shape[0],
            3,
            3,
        ):
            raise ValueError(
                "remote_sample_second_moments must have shape (n, 3, 3)"
            )
        halo_shape = np.asarray(cells.layout.cell_halo_shape, dtype=np.int32)
        if np.any(remote_halo_indices < 0) or np.any(
            remote_halo_indices >= halo_shape[None, :]
        ):
            raise ValueError("remote reconstruction sample halo index is out of bounds")
        if not (
            np.all(np.isfinite(remote_centroids))
            and np.all(np.isfinite(remote_second_moments))
        ):
            raise ValueError("remote reconstruction sample moments must be finite")
    remote_relative_indices = (
        remote_halo_indices - int(cells.layout.halo_width)
    )
    neighborhood_offsets = tuple(
        (di, dj, dk)
        for di in range(-2, 3)
        for dj in range(-2, 3)
        for dk in range(-2, 3)
        if (di, dj, dk) != (0, 0, 0)
    )
    remote_samples_by_relative_index: dict[
        tuple[int, int, int],
        list[int],
    ] = {}
    for remote_sample, relative_index in enumerate(remote_relative_indices):
        remote_samples_by_relative_index.setdefault(
            tuple(int(value) for value in relative_index),
            [],
        ).append(int(remote_sample))

    if target_mask is not None:
        requested = np.asarray(target_mask, dtype=bool)
        if requested.shape != shape:
            raise ValueError(f"target_mask must have shape {shape}, got {requested.shape}")
        # A fixture that owns compact transition faces explicitly must not
        # recursively promote every transition neighbour into another
        # reconstruction owner.  The supplied mask is the authoritative,
        # geometry-derived target set.
        touched = requested.copy()
    else:
        touched = aggregate_target.copy()
        for row in np.flatnonzero(face_active):
            minus = tuple(int(value) for value in minus_owner[row])
            touched[minus] = True
            if has_plus[row]:
                plus = tuple(int(value) for value in plus_owner[row])
                touched[plus] = True
    targets = np.argwhere(touched & active_owner)
    n_rows = int(targets.shape[0])
    if n_rows == 0:
        return LocalMomentReconstruction3D.empty(
            cells.layout,
            max_rows=0,
            max_equations=max_equations,
        )

    target_i = np.zeros((n_rows,), dtype=np.int32)
    target_j = np.zeros((n_rows,), dtype=np.int32)
    target_k = np.zeros((n_rows,), dtype=np.int32)
    equation_kind = np.zeros((n_rows, max_equations), dtype=np.int32)
    sample_i = np.zeros((n_rows, max_equations), dtype=np.int32)
    sample_j = np.zeros((n_rows, max_equations), dtype=np.int32)
    sample_k = np.zeros((n_rows, max_equations), dtype=np.int32)
    boundary_face_row = np.zeros((n_rows, max_equations), dtype=np.int32)
    boundary_patch = np.zeros((n_rows, max_equations), dtype=np.int32)
    boundary_quadrature = np.zeros((n_rows, max_equations), dtype=np.int32)
    equation_active = np.zeros((n_rows, max_equations), dtype=bool)
    rhs_transform = np.zeros((n_rows, 9, max_equations), dtype=np.float64)
    polynomial_order = np.zeros((n_rows,), dtype=np.int32)
    rank = np.zeros((n_rows,), dtype=np.int32)
    condition_number = np.full((n_rows,), np.inf, dtype=np.float64)
    target_row_for_cell = -np.ones(shape, dtype=np.int32)

    for row_index, target_array in enumerate(targets):
        target = tuple(int(value) for value in target_array)
        target_i[row_index], target_j[row_index], target_k[row_index] = target
        target_row_for_cell[target] = row_index
        target_position = centroid[target]
        target_m2 = second_moment[target]
        target_spacing = spacing[target]

        local_candidate_shell: dict[tuple[int, int, int], int] = {}
        remote_candidate_shell: dict[int, int] = {}
        for offset in neighborhood_offsets:
            shell = max(abs(value) for value in offset)
            raw_candidate = tuple(
                target[axis] + offset[axis]
                for axis in range(3)
            )
            local_candidate = list(raw_candidate)
            local_in_bounds = True
            for axis in range(3):
                if periodic_axes[axis]:
                    local_candidate[axis] %= shape[axis]
                elif not (0 <= local_candidate[axis] < shape[axis]):
                    local_in_bounds = False
                    break
            if local_in_bounds:
                candidate = tuple(local_candidate)
                if candidate != target and active_owner[candidate]:
                    local_candidate_shell[candidate] = min(
                        shell,
                        local_candidate_shell.get(candidate, shell),
                    )
            for remote_sample in remote_samples_by_relative_index.get(
                raw_candidate,
                (),
            ):
                remote_candidate_shell[remote_sample] = min(
                    shell,
                    remote_candidate_shell.get(remote_sample, shell),
                )

        candidates = np.asarray(
            tuple(local_candidate_shell),
            dtype=np.int32,
        ).reshape((-1, 3))
        candidate_shells = np.asarray(
            tuple(local_candidate_shell.values()),
            dtype=np.int32,
        )
        if candidates.size:
            candidate_positions = centroid[
                candidates[:, 0],
                candidates[:, 1],
                candidates[:, 2],
            ]
            scaled_distance = np.linalg.norm(
                unwrap_displacement(
                    candidate_positions - target_position[None, :]
                )
                / target_spacing[None, :],
                axis=1,
            )
            candidate_order = np.lexsort(
                (
                    candidates[:, 2],
                    candidates[:, 1],
                    candidates[:, 0],
                    scaled_distance,
                    candidate_shells,
                )
            )
            candidates = candidates[candidate_order]
            candidate_shells = candidate_shells[candidate_order]

        boundary_rows = np.flatnonzero(
            face_active
            & (
                (face_kind == CV_FACE_CUT_WALL)
                | (face_kind == CV_FACE_PHYSICAL_BOUNDARY)
            )
            & np.all(minus_owner == target_array[None, :], axis=1)
        )
        design_rows: list[np.ndarray] = []
        weights: list[float] = []
        metadata: list[tuple[int, tuple[int, int, int] | int]] = []
        sample_distances: list[float] = []
        sample_shells: list[int] = []

        def _quadratic_row(
            displacement: np.ndarray,
            moment_delta: np.ndarray,
        ) -> np.ndarray:
            scaled_displacement = displacement / target_spacing
            scaled_moment = moment_delta / (
                target_spacing[:, None] * target_spacing[None, :]
            )
            return np.asarray(
                (
                    scaled_displacement[0],
                    scaled_displacement[1],
                    scaled_displacement[2],
                    0.5 * scaled_moment[0, 0],
                    0.5 * scaled_moment[1, 1],
                    0.5 * scaled_moment[2, 2],
                    scaled_moment[0, 1],
                    scaled_moment[0, 2],
                    scaled_moment[1, 2],
                ),
                dtype=np.float64,
            )

        for candidate_array, candidate_shell in zip(
            candidates,
            candidate_shells,
        ):
            candidate = tuple(int(value) for value in candidate_array)
            displacement = unwrap_displacement(
                centroid[candidate] - target_position
            )
            moment_delta = (
                second_moment[candidate]
                + np.outer(displacement, displacement)
                - target_m2
            )
            scaled_distance_squared = float(
                np.dot(displacement / target_spacing, displacement / target_spacing)
            )
            design_rows.append(_quadratic_row(displacement, moment_delta))
            weights.append(1.0 / max(scaled_distance_squared, 1.0e-12))
            metadata.append((CV_RECONSTRUCTION_EQUATION_CELL, candidate))
            sample_distances.append(np.sqrt(scaled_distance_squared))
            sample_shells.append(int(candidate_shell))

        remote_candidates = np.asarray(
            tuple(remote_candidate_shell),
            dtype=np.int64,
        )
        if remote_candidates.size:
            remote_distance = np.linalg.norm(
                unwrap_displacement(
                    remote_centroids[remote_candidates]
                    - target_position[None, :]
                )
                / target_spacing[None, :],
                axis=1,
            )
            remote_order = np.lexsort(
                (
                    remote_candidates,
                    remote_distance,
                    np.asarray(
                        tuple(remote_candidate_shell.values()),
                        dtype=np.int32,
                    ),
                )
            )
            remote_candidates = remote_candidates[remote_order]
        seen_remote_geometry: set[
            tuple[float, ...]
        ] = set()
        for remote_sample in remote_candidates:
            remote_geometry_key = tuple(
                np.round(
                    np.concatenate(
                        (
                            remote_centroids[remote_sample],
                            remote_second_moments[remote_sample].ravel(),
                        )
                    ),
                    decimals=13,
                )
            )
            if remote_geometry_key in seen_remote_geometry:
                continue
            seen_remote_geometry.add(remote_geometry_key)
            displacement = unwrap_displacement(
                remote_centroids[remote_sample] - target_position
            )
            moment_delta = (
                remote_second_moments[remote_sample]
                + np.outer(displacement, displacement)
                - target_m2
            )
            scaled_distance_squared = float(
                np.dot(
                    displacement / target_spacing,
                    displacement / target_spacing,
                )
            )
            design_rows.append(_quadratic_row(displacement, moment_delta))
            weights.append(1.0 / max(scaled_distance_squared, 1.0e-12))
            metadata.append(
                (
                    CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
                    tuple(int(value) for value in remote_halo_indices[remote_sample]),
                )
            )
            sample_distances.append(np.sqrt(scaled_distance_squared))
            sample_shells.append(
                int(remote_candidate_shell[int(remote_sample)])
            )

        sample_order = np.lexsort(
            (
                np.asarray(sample_distances, dtype=np.float64),
                np.asarray(sample_shells, dtype=np.int32),
            )
        )
        radius_one_samples = [
            int(index)
            for index in sample_order
            if sample_shells[int(index)] <= 1
        ][:max_samples]
        radius_two_samples = [
            int(index)
            for index in sample_order
            if sample_shells[int(index)] > 1
        ]
        sample_records = [
            (
                design_rows[index],
                weights[index],
                metadata[index],
            )
            for index in range(len(design_rows))
        ]
        design_rows = [
            sample_records[index][0]
            for index in radius_one_samples
        ]
        weights = [
            sample_records[index][1]
            for index in radius_one_samples
        ]
        metadata = [
            sample_records[index][2]
            for index in radius_one_samples
        ]
        deferred_radius_two = [
            sample_records[index]
            for index in radius_two_samples
        ]
        selected_sample_count = len(radius_one_samples)

        for face_row in boundary_rows:
            # Dirichlet data are collocated with flux quadrature.  A single
            # wall-centroid equation leaves tangential wall variation and the
            # boundary derivative underconstrained; retain every active
            # quadrature point as an independent polynomial equation.
            face_measure = np.abs(face_J[face_row]) * np.linalg.norm(
                area_weight[face_row],
                axis=-1,
            )
            face_measure = np.where(
                quadrature_active[face_row],
                face_measure,
                0.0,
            )
            total_face_measure = float(np.sum(face_measure))
            if not np.isfinite(total_face_measure) or total_face_measure <= 0.0:
                continue
            for patch in range(int(irregular_faces.max_patches)):
                for quadrature in range(4):
                    if not quadrature_active[face_row, patch, quadrature]:
                        continue
                    wall_point = quadrature_points[face_row, patch, quadrature]
                    displacement = unwrap_displacement(
                        wall_point - target_position
                    )
                    moment_delta = (
                        np.outer(displacement, displacement) - target_m2
                    )
                    scaled_distance_squared = float(
                        np.dot(
                            displacement / target_spacing,
                            displacement / target_spacing,
                        )
                    )
                    design_rows.append(_quadratic_row(displacement, moment_delta))
                    area_fraction = float(
                        face_measure[patch, quadrature] / total_face_measure
                    )
                    weights.append(
                        area_fraction
                        / max(scaled_distance_squared, 1.0e-12)
                    )
                    metadata.append(
                        (
                            CV_RECONSTRUCTION_EQUATION_DIRICHLET,
                            (int(face_row), int(patch), int(quadrature)),
                        )
                    )

        def _quadratic_quality(
            candidate_rows: list[np.ndarray],
            candidate_weights: list[float],
        ) -> tuple[int, float]:
            if len(candidate_rows) < 9:
                return 0, np.inf
            candidate_design = np.asarray(
                candidate_rows,
                dtype=np.float64,
            )
            candidate_sqrt_weight = np.sqrt(
                np.asarray(candidate_weights, dtype=np.float64)
            )
            singular_values = np.linalg.svd(
                candidate_sqrt_weight[:, None] * candidate_design,
                compute_uv=False,
            )
            if not singular_values.size or singular_values[0] <= 0.0:
                return 0, np.inf
            candidate_tolerance = float(svd_rcond) * singular_values[0]
            candidate_rank = int(
                np.sum(singular_values > candidate_tolerance)
            )
            candidate_condition = (
                float(singular_values[0] / singular_values[-1])
                if (
                    singular_values.size >= 9
                    and singular_values[-1] > candidate_tolerance
                )
                else np.inf
            )
            return candidate_rank, candidate_condition

        selected_rank, selected_condition = _quadratic_quality(
            design_rows,
            weights,
        )
        for deferred_row, deferred_weight, deferred_metadata in (
            deferred_radius_two
        ):
            if (
                selected_rank >= 9
                and selected_condition <= float(condition_limit)
            ):
                break
            if selected_sample_count >= max_samples:
                break
            design_rows.append(deferred_row)
            weights.append(deferred_weight)
            metadata.append(deferred_metadata)
            selected_sample_count += 1
            selected_rank, selected_condition = _quadratic_quality(
                design_rows,
                weights,
            )

        if len(design_rows) > max_equations:
            design_rows = design_rows[:max_equations]
            weights = weights[:max_equations]
            metadata = metadata[:max_equations]
        equation_count = len(design_rows)
        if equation_count < 3:
            continue
        coefficient_scale = np.asarray(
            (
                1.0 / target_spacing[0],
                1.0 / target_spacing[1],
                1.0 / target_spacing[2],
                1.0 / target_spacing[0] ** 2,
                1.0 / target_spacing[1] ** 2,
                1.0 / target_spacing[2] ** 2,
                1.0 / (target_spacing[0] * target_spacing[1]),
                1.0 / (target_spacing[0] * target_spacing[2]),
                1.0 / (target_spacing[1] * target_spacing[2]),
            ),
            dtype=np.float64,
        )

        design = np.asarray(design_rows, dtype=np.float64)
        sqrt_weight = np.sqrt(np.asarray(weights, dtype=np.float64))
        weighted_design = sqrt_weight[:, None] * design
        singular = np.linalg.svd(weighted_design, compute_uv=False)
        tolerance = (
            float(svd_rcond) * singular[0]
            if singular.size and singular[0] > 0.0
            else np.inf
        )
        full_rank = int(np.sum(singular > tolerance))
        full_condition = (
            float(singular[0] / singular[-1])
            if singular.size >= 9 and singular[-1] > tolerance
            else np.inf
        )

        if full_rank >= 9 and full_condition <= float(condition_limit):
            selected_columns = 9
            order = 2
            weighted_selected = weighted_design
            transform = np.linalg.pinv(
                weighted_selected,
                rcond=float(svd_rcond),
            ) * sqrt_weight[None, :]
            row_transform = coefficient_scale[:, None] * transform
            selected_rank = full_rank
            selected_condition = full_condition
        else:
            linear_design = weighted_design[:, :3]
            linear_singular = np.linalg.svd(linear_design, compute_uv=False)
            linear_tolerance = (
                float(svd_rcond) * linear_singular[0]
                if linear_singular.size and linear_singular[0] > 0.0
                else np.inf
            )
            selected_rank = int(np.sum(linear_singular > linear_tolerance))
            if selected_rank < 3:
                continue
            selected_condition = float(
                linear_singular[0] / linear_singular[-1]
            )
            selected_columns = 3
            order = 1
            linear_transform = np.linalg.pinv(
                linear_design,
                rcond=float(svd_rcond),
            ) * sqrt_weight[None, :]
            row_transform = np.zeros((9, equation_count), dtype=np.float64)
            row_transform[:3, :] = (
                coefficient_scale[:3, None] * linear_transform
            )

        polynomial_order[row_index] = order
        rank[row_index] = selected_rank
        condition_number[row_index] = selected_condition
        rhs_transform[row_index, :, :equation_count] = row_transform
        equation_active[row_index, :equation_count] = True
        for equation_index, (kind, payload) in enumerate(metadata):
            equation_kind[row_index, equation_index] = int(kind)
            if kind in (
                CV_RECONSTRUCTION_EQUATION_CELL,
                CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
            ):
                sample = payload
                sample_i[row_index, equation_index] = int(sample[0])
                sample_j[row_index, equation_index] = int(sample[1])
                sample_k[row_index, equation_index] = int(sample[2])
            else:
                boundary_face_row[row_index, equation_index] = int(payload[0])
                boundary_patch[row_index, equation_index] = int(payload[1])
                boundary_quadrature[row_index, equation_index] = int(payload[2])

    active = polynomial_order > 0
    return LocalMomentReconstruction3D(
        layout=cells.layout,
        target_i=jnp.asarray(target_i),
        target_j=jnp.asarray(target_j),
        target_k=jnp.asarray(target_k),
        equation_kind=jnp.asarray(equation_kind),
        sample_i=jnp.asarray(sample_i),
        sample_j=jnp.asarray(sample_j),
        sample_k=jnp.asarray(sample_k),
        boundary_face_row=jnp.asarray(boundary_face_row),
        boundary_patch=jnp.asarray(boundary_patch),
        boundary_quadrature=jnp.asarray(boundary_quadrature),
        equation_active=jnp.asarray(equation_active),
        rhs_transform=jnp.asarray(rhs_transform),
        active=jnp.asarray(active),
        target_row_for_cell=jnp.asarray(target_row_for_cell),
        polynomial_order=jnp.asarray(polynomial_order),
        rank=jnp.asarray(rank),
        condition_number=jnp.asarray(condition_number),
        max_rows=n_rows,
        max_equations=max_equations,
    )


def _precompute_local_cubic_reconstruction(
    cells: LocalControlVolumeCellGeometry3D,
    irregular_faces: LocalControlVolumeFaceRows3D,
    *,
    spacing_owned: jnp.ndarray,
    remote_sample_halo_indices: np.ndarray | None = None,
    remote_sample_centroids: np.ndarray | None = None,
    remote_sample_second_moments: np.ndarray | None = None,
    remote_sample_third_moments: np.ndarray | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
    coordinate_periodic_axes: tuple[bool, bool, bool] | None = None,
    coordinate_periods: tuple[float, float, float] | None = None,
    target_mask: jnp.ndarray | None = None,
    distance_weight_target_mask: jnp.ndarray | None = None,
    max_samples: int = 48,
    max_equations: int = 64,
    condition_limit: float = 1.0e6,
    svd_rcond: float = 1.0e-12,
    boundary_weight_scale: float = 1.0,
    distance_row_weight_exponent: float = 0.0,
) -> LocalMomentReconstruction3D:
    """Precompute 19-coefficient cubic finite-volume reconstruction rows.

    This deliberately lives beside the quadratic builder during migration.
    Remote samples are compact halo references, so aggregate ownership stays
    local while the fitted stencil remains decomposition-compatible.
    """
    boundary_weight_scale = float(boundary_weight_scale)
    if (
        not np.isfinite(boundary_weight_scale)
        or boundary_weight_scale <= 0.0
    ):
        raise ValueError(
            "boundary_weight_scale must be finite and positive"
        )
    distance_row_weight_exponent = float(distance_row_weight_exponent)
    if (
        not np.isfinite(distance_row_weight_exponent)
        or distance_row_weight_exponent < 0.0
    ):
        raise ValueError(
            "distance_row_weight_exponent must be finite and nonnegative"
        )

    def distance_observation_weight(
        distance_squared: np.ndarray | float,
        *,
        localized: np.ndarray | bool,
    ) -> np.ndarray:
        distance_squared = np.asarray(distance_squared, dtype=np.float64)
        # Exact legacy behavior: ``sqrt(weight)`` gives a 1/d row multiplier
        # in the weighted least-squares system.
        legacy = 1.0 / np.maximum(distance_squared, 1.0e-12)
        if distance_row_weight_exponent == 0.0:
            return legacy
        # ``sqrt(weight)`` is the actual WLS row multiplier.  The unit floor
        # avoids singularly over-weighting wall points inside one local cell
        # width while the exponent controls decay of farther observations.
        distance = np.maximum(np.sqrt(distance_squared), 1.0)
        localized_weight = distance ** (
            -2.0 * distance_row_weight_exponent
        )
        localized_array = np.asarray(localized, dtype=bool)
        while localized_array.ndim < distance_squared.ndim:
            localized_array = localized_array[..., None]
        return np.where(localized_array, localized_weight, legacy)
    shape = cells.shape
    active = np.asarray(cells.is_active_owner, dtype=bool)
    centroid = np.asarray(cells.centroid, dtype=np.float64)
    m2 = np.asarray(cells.second_moment, dtype=np.float64)
    m3 = np.asarray(cells.third_moment, dtype=np.float64)
    spacing = np.maximum(np.asarray(spacing_owned, dtype=np.float64), 1.0e-14)
    if spacing.shape != shape + (3,):
        raise ValueError("spacing_owned must match control-volume owned shape")
    requested = (
        np.asarray(target_mask, dtype=bool)
        if target_mask is not None
        else active.copy()
    )
    if requested.shape != shape:
        raise ValueError("target_mask must match control-volume owned shape")
    distance_weight_target = (
        requested & active
        if distance_weight_target_mask is None
        else np.asarray(distance_weight_target_mask, dtype=bool)
    )
    if distance_weight_target.shape != shape:
        raise ValueError(
            "distance_weight_target_mask must match control-volume owned shape"
        )
    distance_weight_target = distance_weight_target & requested & active
    targets = np.argwhere(requested & active)
    n_rows = len(targets)
    if n_rows == 0:
        return LocalMomentReconstruction3D.empty(
            cells.layout,
            max_rows=0,
            max_equations=max_equations,
            coefficient_count=19,
        )
    periods = np.asarray(
        coordinate_periods if coordinate_periods is not None else (1.0, 1.0, 1.0),
        dtype=np.float64,
    )
    # Local index periodicity and global coordinate periodicity are distinct
    # on a decomposed periodic axis. Local candidates must not wrap within a
    # shard, while remote samples across the global seam must still be
    # unwrapped to the nearest periodic image.
    periodic_axes = tuple(bool(value) for value in periodic_axes)
    coordinate_periodic_axes = tuple(
        bool(value) for value in (
            periodic_axes
            if coordinate_periodic_axes is None
            else coordinate_periodic_axes
        )
    )
    if remote_sample_halo_indices is None:
        remote_indices = np.zeros((0, 3), dtype=np.int32)
        remote_centroids = np.zeros((0, 3), dtype=np.float64)
        remote_m2 = np.zeros((0, 3, 3), dtype=np.float64)
        remote_m3 = np.zeros((0, 3, 3, 3), dtype=np.float64)
    else:
        remote_indices = np.asarray(remote_sample_halo_indices, dtype=np.int32)
        remote_centroids = np.asarray(remote_sample_centroids, dtype=np.float64)
        remote_m2 = np.asarray(remote_sample_second_moments, dtype=np.float64)
        remote_m3 = np.asarray(remote_sample_third_moments, dtype=np.float64)
        if (
            remote_indices.ndim != 2 or remote_indices.shape[1] != 3
            or remote_centroids.shape != remote_indices.shape
            or remote_m2.shape != (remote_indices.shape[0], 3, 3)
            or remote_m3.shape != (remote_indices.shape[0], 3, 3, 3)
        ):
            raise ValueError("remote cubic reconstruction sample metadata has inconsistent shapes")

    def unwrap(delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=np.float64).copy()
        for axis, is_periodic in enumerate(coordinate_periodic_axes):
            if is_periodic:
                delta[..., axis] -= np.round(delta[..., axis] / periods[axis]) * periods[axis]
        return delta

    def cubic_row(delta: np.ndarray, dm2: np.ndarray, dm3: np.ndarray, h: np.ndarray) -> np.ndarray:
        d = delta / h
        q = dm2 / (h[:, None] * h[None, :])
        c = dm3 / (h[:, None, None] * h[None, :, None] * h[None, None, :])
        return np.asarray((
            d[0], d[1], d[2],
            0.5*q[0, 0], 0.5*q[1, 1], 0.5*q[2, 2], q[0, 1], q[0, 2], q[1, 2],
            c[0, 0, 0]/6.0, c[1, 1, 1]/6.0, c[2, 2, 2]/6.0,
            c[0, 0, 1]/2.0, c[0, 0, 2]/2.0, c[0, 1, 1]/2.0,
            c[0, 2, 2]/2.0, c[1, 1, 2]/2.0, c[1, 2, 2]/2.0, c[0, 1, 2],
        ), dtype=np.float64)

    def translated_m3(delta: np.ndarray, sample_m2: np.ndarray, sample_m3: np.ndarray) -> np.ndarray:
        return (
            sample_m3
            + delta[:, None, None] * sample_m2[None, :, :]
            + delta[None, :, None] * sample_m2[:, None, :]
            + delta[None, None, :] * sample_m2[:, :, None]
            + delta[:, None, None] * delta[None, :, None] * delta[None, None, :]
        )

    def cubic_rows(
        delta: np.ndarray,
        sample_m2: np.ndarray,
        sample_m3: np.ndarray,
        target_m2: np.ndarray,
        target_m3: np.ndarray,
        h: np.ndarray,
    ) -> np.ndarray:
        """Vectorized counterpart of ``cubic_row`` for candidate batches."""

        dm2 = (
            sample_m2
            + delta[..., :, None] * delta[..., None, :]
            - target_m2
        )
        dm3 = (
            sample_m3
            + delta[..., :, None, None] * sample_m2[..., None, :, :]
            + delta[..., None, :, None] * sample_m2[..., :, None, :]
            + delta[..., None, None, :] * sample_m2[..., :, :, None]
            + delta[..., :, None, None]
            * delta[..., None, :, None]
            * delta[..., None, None, :]
            - target_m3
        )
        d = delta / h
        q = dm2 / (h[:, None] * h[None, :])
        c = dm3 / (
            h[:, None, None]
            * h[None, :, None]
            * h[None, None, :]
        )
        return np.stack(
            (
                d[..., 0], d[..., 1], d[..., 2],
                0.5 * q[..., 0, 0], 0.5 * q[..., 1, 1],
                0.5 * q[..., 2, 2], q[..., 0, 1], q[..., 0, 2],
                q[..., 1, 2], c[..., 0, 0, 0] / 6.0,
                c[..., 1, 1, 1] / 6.0, c[..., 2, 2, 2] / 6.0,
                c[..., 0, 0, 1] / 2.0, c[..., 0, 0, 2] / 2.0,
                c[..., 0, 1, 1] / 2.0, c[..., 0, 2, 2] / 2.0,
                c[..., 1, 1, 2] / 2.0, c[..., 1, 2, 2] / 2.0,
                c[..., 0, 1, 2],
            ),
            axis=-1,
        )

    face_active = np.asarray(irregular_faces.active, dtype=bool)
    face_kind = np.asarray(irregular_faces.kind, dtype=np.int32)
    minus = np.stack((np.asarray(irregular_faces.minus_owner_i), np.asarray(irregular_faces.minus_owner_j), np.asarray(irregular_faces.minus_owner_k)), axis=-1)
    qpoints = np.asarray(irregular_faces.quadrature_points, dtype=np.float64)
    qactive = np.asarray(irregular_faces.quadrature_active, dtype=bool)
    measure = np.abs(np.asarray(irregular_faces.J, dtype=np.float64)) * np.linalg.norm(np.asarray(irregular_faces.area_covector_weight, dtype=np.float64), axis=-1)
    target_i = np.zeros((n_rows,), dtype=np.int32); target_j = target_i.copy(); target_k = target_i.copy()
    equation_kind = np.zeros((n_rows, max_equations), dtype=np.int32)
    sample_i = np.zeros_like(equation_kind); sample_j = np.zeros_like(equation_kind); sample_k = np.zeros_like(equation_kind)
    boundary_face_row = np.zeros_like(equation_kind); boundary_patch = np.zeros_like(equation_kind); boundary_quadrature = np.zeros_like(equation_kind)
    equation_active = np.zeros((n_rows, max_equations), dtype=bool)
    transform_out = np.zeros((n_rows, 19, max_equations), dtype=np.float64)
    order = np.zeros((n_rows,), dtype=np.int32); rank = np.zeros((n_rows,), dtype=np.int32); condition = np.full((n_rows,), np.inf)
    row_for_cell = -np.ones(shape, dtype=np.int32)
    target_i[:] = targets[:, 0]
    target_j[:] = targets[:, 1]
    target_k[:] = targets[:, 2]
    row_for_cell[tuple(targets.T)] = np.arange(n_rows, dtype=np.int32)

    # The common guard-ring case has no wall equation and a complete local
    # radius-two stencil.  Assemble those fixed-shape systems in chunks and
    # hand a stack of matrices to NumPy's batched SVD/pseudoinverse kernels.
    # Rows near a wall, shard edge, or rank deficiency deliberately retain
    # the detailed path below, where variable boundary/remote equations are
    # part of the reconstruction contract.
    boundary_target = np.zeros(shape, dtype=bool)
    boundary_rows = face_active & (
        (face_kind == CV_FACE_CUT_WALL)
        | (face_kind == CV_FACE_PHYSICAL_BOUNDARY)
    )
    if np.any(boundary_rows):
        boundary_target[tuple(minus[boundary_rows].T)] = True
    radius_two_offsets = np.asarray(
        [
            (di, dj, dk)
            for di in range(-2, 3)
            for dj in range(-2, 3)
            for dk in range(-2, 3)
            if (di, dj, dk) != (0, 0, 0)
        ],
        dtype=np.int32,
    )
    radius_two_shell = np.max(np.abs(radius_two_offsets), axis=1)
    radius_two_tie = np.arange(radius_two_offsets.shape[0], dtype=np.float64)
    processed = np.zeros((n_rows,), dtype=bool)
    if max_samples >= 19 and all(size >= 5 for size in shape):
        target_interior = np.all(
            (targets >= 2)
            & (targets <= np.asarray(shape, dtype=np.int32)[None, :] - 3),
            axis=1,
        )
        batch_candidate_rows = np.flatnonzero(
            target_interior & ~boundary_target[tuple(targets.T)]
        )
        batch_size = 256
        selected_count = min(int(max_samples), int(radius_two_offsets.shape[0]))
        for begin in range(0, len(batch_candidate_rows), batch_size):
            rows = batch_candidate_rows[begin : begin + batch_size]
            target_batch = targets[rows]
            candidates = target_batch[:, None, :] + radius_two_offsets[None, :, :]
            candidate_active = active[
                candidates[..., 0],
                candidates[..., 1],
                candidates[..., 2],
            ]
            if not np.any(np.sum(candidate_active, axis=1) >= selected_count):
                continue

            x0_batch = centroid[
                target_batch[:, 0], target_batch[:, 1], target_batch[:, 2]
            ]
            m20_batch = m2[
                target_batch[:, 0], target_batch[:, 1], target_batch[:, 2]
            ]
            m30_batch = m3[
                target_batch[:, 0], target_batch[:, 1], target_batch[:, 2]
            ]
            h_batch = spacing[
                target_batch[:, 0], target_batch[:, 1], target_batch[:, 2]
            ]
            candidate_centroid = centroid[
                candidates[..., 0], candidates[..., 1], candidates[..., 2]
            ]
            candidate_m2 = m2[
                candidates[..., 0], candidates[..., 1], candidates[..., 2]
            ]
            candidate_m3 = m3[
                candidates[..., 0], candidates[..., 1], candidates[..., 2]
            ]
            delta = unwrap(candidate_centroid - x0_batch[:, None, :])
            scaled_delta = delta / h_batch[:, None, :]
            d2 = np.einsum("bsi,bsi->bs", scaled_delta, scaled_delta)
            priority = (
                1.0e3 * radius_two_shell[None, :]
                + d2
                + 1.0e-9 * radius_two_tie[None, :]
            )
            priority = np.where(candidate_active, priority, np.inf)
            selected_indices = np.argsort(priority, axis=1)[:, :selected_count]
            selected_finite = np.take_along_axis(
                np.isfinite(priority), selected_indices, axis=1
            )
            batch_eligible = np.all(selected_finite, axis=1)
            if not np.any(batch_eligible):
                continue

            selected_delta = np.take_along_axis(
                delta,
                np.broadcast_to(
                    selected_indices[..., None],
                    selected_indices.shape + (3,),
                ),
                axis=1,
            )
            selected_m2 = np.take_along_axis(
                candidate_m2,
                np.broadcast_to(
                    selected_indices[..., None, None],
                    selected_indices.shape + (3, 3),
                ),
                axis=1,
            )
            selected_m3 = np.take_along_axis(
                candidate_m3,
                np.broadcast_to(
                    selected_indices[..., None, None, None],
                    selected_indices.shape + (3, 3, 3),
                ),
                axis=1,
            )
            selected_d2 = np.take_along_axis(d2, selected_indices, axis=1)
            dm2 = (
                selected_m2
                + selected_delta[..., :, None] * selected_delta[..., None, :]
                - m20_batch[:, None, :, :]
            )
            dm3 = (
                selected_m3
                + selected_delta[..., :, None, None]
                * selected_m2[..., None, :, :]
                + selected_delta[..., None, :, None]
                * selected_m2[..., :, None, :]
                + selected_delta[..., None, None, :]
                * selected_m2[..., :, :, None]
                + selected_delta[..., :, None, None]
                * selected_delta[..., None, :, None]
                * selected_delta[..., None, None, :]
                - m30_batch[:, None, :, :, :]
            )
            d = selected_delta / h_batch[:, None, :]
            q = dm2 / (
                h_batch[:, None, :, None] * h_batch[:, None, None, :]
            )
            c = dm3 / (
                h_batch[:, None, :, None, None]
                * h_batch[:, None, None, :, None]
                * h_batch[:, None, None, None, :]
            )
            design = np.stack(
                (
                    d[..., 0], d[..., 1], d[..., 2],
                    0.5 * q[..., 0, 0], 0.5 * q[..., 1, 1],
                    0.5 * q[..., 2, 2], q[..., 0, 1], q[..., 0, 2],
                    q[..., 1, 2], c[..., 0, 0, 0] / 6.0,
                    c[..., 1, 1, 1] / 6.0, c[..., 2, 2, 2] / 6.0,
                    c[..., 0, 0, 1] / 2.0, c[..., 0, 0, 2] / 2.0,
                    c[..., 0, 1, 1] / 2.0, c[..., 0, 2, 2] / 2.0,
                    c[..., 1, 1, 2] / 2.0, c[..., 1, 2, 2] / 2.0,
                    c[..., 0, 1, 2],
                ),
                axis=-1,
            )
            weights = distance_observation_weight(
                selected_d2,
                localized=distance_weight_target[tuple(target_batch.T)],
            )
            weighted = np.sqrt(weights)[..., None] * design
            try:
                singular = np.linalg.svd(weighted, compute_uv=False)
            except np.linalg.LinAlgError:
                continue
            tolerance = svd_rcond * singular[:, :1]
            batch_rank = np.sum(singular > tolerance, axis=1).astype(np.int32)
            batch_condition = np.where(
                batch_rank >= 19,
                singular[:, 0] / np.maximum(singular[:, -1], 1.0e-300),
                np.inf,
            )
            batch_valid = (
                batch_eligible
                & (batch_rank >= 19)
                & (batch_condition <= condition_limit)
            )
            if not np.any(batch_valid):
                continue
            try:
                pseudoinverse = np.linalg.pinv(
                    weighted[batch_valid],
                    rcond=svd_rcond,
                )
            except np.linalg.LinAlgError:
                continue
            valid_rows = rows[batch_valid]
            valid_h = h_batch[batch_valid]
            scale = np.stack(
                (
                    1 / valid_h[:, 0], 1 / valid_h[:, 1], 1 / valid_h[:, 2],
                    1 / valid_h[:, 0] ** 2, 1 / valid_h[:, 1] ** 2,
                    1 / valid_h[:, 2] ** 2,
                    1 / (valid_h[:, 0] * valid_h[:, 1]),
                    1 / (valid_h[:, 0] * valid_h[:, 2]),
                    1 / (valid_h[:, 1] * valid_h[:, 2]),
                    1 / valid_h[:, 0] ** 3, 1 / valid_h[:, 1] ** 3,
                    1 / valid_h[:, 2] ** 3,
                    1 / (valid_h[:, 0] ** 2 * valid_h[:, 1]),
                    1 / (valid_h[:, 0] ** 2 * valid_h[:, 2]),
                    1 / (valid_h[:, 0] * valid_h[:, 1] ** 2),
                    1 / (valid_h[:, 0] * valid_h[:, 2] ** 2),
                    1 / (valid_h[:, 1] ** 2 * valid_h[:, 2]),
                    1 / (valid_h[:, 1] * valid_h[:, 2] ** 2),
                    1 / (valid_h[:, 0] * valid_h[:, 1] * valid_h[:, 2]),
                ),
                axis=1,
            )
            transform_out[valid_rows, :, :selected_count] = (
                scale[:, :, None]
                * pseudoinverse
                * np.sqrt(weights[batch_valid])[:, None, :]
            )
            selected_candidates = np.take_along_axis(
                candidates,
                np.broadcast_to(
                    selected_indices[..., None],
                    selected_indices.shape + (3,),
                ),
                axis=1,
            )[batch_valid]
            equation_kind[valid_rows, :selected_count] = (
                CV_RECONSTRUCTION_EQUATION_CELL
            )
            sample_i[valid_rows, :selected_count] = selected_candidates[..., 0]
            sample_j[valid_rows, :selected_count] = selected_candidates[..., 1]
            sample_k[valid_rows, :selected_count] = selected_candidates[..., 2]
            equation_active[valid_rows, :selected_count] = True
            order[valid_rows] = 3
            rank[valid_rows] = batch_rank[batch_valid]
            condition[valid_rows] = batch_condition[batch_valid]
            processed[valid_rows] = True
    radius_three_offsets = np.asarray(
        [
            (di, dj, dk)
            for di in range(-3, 4)
            for dj in range(-3, 4)
            for dk in range(-3, 4)
            if (di, dj, dk) != (0, 0, 0)
        ],
        dtype=np.int32,
    )
    radius_three_shell = np.max(np.abs(radius_three_offsets), axis=1)
    for r, target_array in enumerate(targets):
        if processed[r]:
            continue
        target = tuple(int(x) for x in target_array)
        x0, m20, m30, h = centroid[target], m2[target], m3[target], spacing[target]
        records: list[tuple[int, float, np.ndarray, tuple[int, tuple[int, int, int]]]] = []
        raw_candidate = target_array[None, :] + radius_three_offsets
        candidate = raw_candidate.copy()
        candidate_in_bounds = np.ones((len(candidate),), dtype=bool)
        for axis in range(3):
            if periodic_axes[axis]:
                candidate[:, axis] %= shape[axis]
            else:
                candidate_in_bounds &= (
                    (candidate[:, axis] >= 0)
                    & (candidate[:, axis] < shape[axis])
                )
        safe_candidate = np.clip(
            candidate,
            0,
            np.asarray(shape, dtype=np.int32)[None, :] - 1,
        )
        candidate_in_bounds &= active[
            safe_candidate[:, 0],
            safe_candidate[:, 1],
            safe_candidate[:, 2],
        ]
        candidate = candidate[candidate_in_bounds]
        candidate_shell = radius_three_shell[candidate_in_bounds]
        if len(candidate):
            candidate_centroid = centroid[
                candidate[:, 0],
                candidate[:, 1],
                candidate[:, 2],
            ]
            candidate_m2 = m2[
                candidate[:, 0],
                candidate[:, 1],
                candidate[:, 2],
            ]
            candidate_m3 = m3[
                candidate[:, 0],
                candidate[:, 1],
                candidate[:, 2],
            ]
            candidate_delta = unwrap(candidate_centroid - x0[None, :])
            candidate_scaled_delta = candidate_delta / h[None, :]
            candidate_distance2 = np.sum(
                candidate_scaled_delta * candidate_scaled_delta,
                axis=1,
            )
            candidate_design = cubic_rows(
                candidate_delta,
                candidate_m2,
                candidate_m3,
                m20,
                m30,
                h,
            )
            records.extend(
                (
                    int(shell),
                    float(distance2),
                    design,
                    (
                        CV_RECONSTRUCTION_EQUATION_CELL,
                        tuple(int(value) for value in candidate_index),
                    ),
                )
                for shell, distance2, design, candidate_index in zip(
                    candidate_shell,
                    candidate_distance2,
                    candidate_design,
                    candidate,
                )
            )
        for remote_index, (remote_position, remote_second, remote_third) in enumerate(
            zip(remote_centroids, remote_m2, remote_m3)
        ):
            delta = unwrap(remote_position - x0)
            scaled_delta = delta / h
            d2 = float(np.dot(scaled_delta, scaled_delta))
            if d2 > 27.0 + 1.0e-12:
                continue
            dm2 = remote_second + np.outer(delta, delta) - m20
            dm3 = translated_m3(delta, remote_second, remote_third) - m30
            # Rank remote candidates by their actual geometric shell. Marking
            # every remote sample as shell three sorts adjacent cross-shard
            # owners behind the local sample cap and makes the fit spuriously
            # one-sided at decomposition boundaries.
            remote_shell = max(
                1,
                min(
                    3,
                    int(np.ceil(np.max(np.abs(scaled_delta)) - 1.0e-12)),
                ),
            )
            records.append((
                remote_shell,
                d2,
                cubic_row(delta, dm2, dm3, h),
                (CV_RECONSTRUCTION_EQUATION_REMOTE_CELL, tuple(int(v) for v in remote_indices[remote_index])),
            ))
        records.sort(key=lambda item: (item[0], item[1], item[3][1]))
        selected = records[:max_samples]
        boundary_rows = np.flatnonzero(face_active & ((face_kind == CV_FACE_CUT_WALL) | (face_kind == CV_FACE_PHYSICAL_BOUNDARY)) & np.all(minus == target_array, axis=1))
        for fr in boundary_rows:
            total = float(np.sum(np.where(qactive[fr], measure[fr], 0.0)))
            if total <= 0.0: continue
            for patch in range(irregular_faces.max_patches):
                for quad in range(4):
                    if not qactive[fr, patch, quad]: continue
                    delta = unwrap(qpoints[fr, patch, quad] - x0); dm2 = np.outer(delta, delta) - m20
                    dm3 = delta[:, None, None] * delta[None, :, None] * delta[None, None, :] - m30
                    d2 = float(np.dot(delta/h, delta/h))
                    selected.append((0, d2, cubic_row(delta, dm2, dm3, h), (CV_RECONSTRUCTION_EQUATION_DIRICHLET, (int(fr), patch, quad))))
        selected = selected[:max_equations]
        if len(selected) < 19: continue
        design = np.stack([item[2] for item in selected])
        weights = distance_observation_weight(
            np.asarray([item[1] for item in selected], dtype=np.float64),
            localized=bool(distance_weight_target[target]),
        )
        # Each wall face shares one normalized distance weight.
        for idx, item in enumerate(selected):
            if item[3][0] == CV_RECONSTRUCTION_EQUATION_DIRICHLET:
                fr, patch, quad = item[3][1]; weights[idx] *= measure[fr, patch, quad] / max(float(np.sum(np.where(qactive[fr], measure[fr], 0.0))), 1e-30)
                weights[idx] *= boundary_weight_scale
        weighted = np.sqrt(weights)[:, None] * design; singular = np.linalg.svd(weighted, compute_uv=False)
        tolerance = svd_rcond * singular[0] if singular.size else np.inf; full_rank = int(np.sum(singular > tolerance)); cond = float(singular[0]/singular[-1]) if full_rank >= 19 else np.inf
        rank[r] = full_rank
        condition[r] = cond
        if full_rank < 19 or cond > condition_limit: continue
        scale = np.asarray((
            1/h[0], 1/h[1], 1/h[2], 1/h[0]**2, 1/h[1]**2, 1/h[2]**2, 1/(h[0]*h[1]), 1/(h[0]*h[2]), 1/(h[1]*h[2]),
            1/h[0]**3, 1/h[1]**3, 1/h[2]**3, 1/(h[0]**2*h[1]), 1/(h[0]**2*h[2]), 1/(h[0]*h[1]**2), 1/(h[0]*h[2]**2), 1/(h[1]**2*h[2]), 1/(h[1]*h[2]**2), 1/(h[0]*h[1]*h[2]),
        ))
        transform_out[r, :, :len(selected)] = scale[:, None] * np.linalg.pinv(weighted, rcond=svd_rcond) * np.sqrt(weights)[None, :]
        order[r] = 3; equation_active[r, :len(selected)] = True
        for e, (_, _, _, (kind, payload)) in enumerate(selected):
            equation_kind[r, e] = kind
            if kind in (CV_RECONSTRUCTION_EQUATION_CELL, CV_RECONSTRUCTION_EQUATION_REMOTE_CELL):
                sample_i[r, e], sample_j[r, e], sample_k[r, e] = payload
            else:
                boundary_face_row[r, e], boundary_patch[r, e], boundary_quadrature[r, e] = payload
    return LocalMomentReconstruction3D(layout=cells.layout, target_i=jnp.asarray(target_i), target_j=jnp.asarray(target_j), target_k=jnp.asarray(target_k), equation_kind=jnp.asarray(equation_kind), sample_i=jnp.asarray(sample_i), sample_j=jnp.asarray(sample_j), sample_k=jnp.asarray(sample_k), boundary_face_row=jnp.asarray(boundary_face_row), boundary_patch=jnp.asarray(boundary_patch), boundary_quadrature=jnp.asarray(boundary_quadrature), equation_active=jnp.asarray(equation_active), rhs_transform=jnp.asarray(transform_out), active=jnp.asarray(order > 0), target_row_for_cell=jnp.asarray(row_for_cell), polynomial_order=jnp.asarray(order), rank=jnp.asarray(rank), condition_number=jnp.asarray(condition), max_rows=n_rows, max_equations=max_equations)


def build_local_control_volume_field_closure(
    field_halo: jnp.ndarray,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    boundary_bc: LocalControlVolumeBoundaryBC3D,
    domain: LocalDomain3D | None = None,
) -> LocalControlVolumeFieldClosure3D:
    """Evaluate static direct cubic face functionals for one scalar field.

    The compiler has already selected the observations and formed all three flux
    functionals.  This routine only performs bounded gathers and three weighted
    weighted sums, so it is safe to call under ``jax.jit`` and has no reconstruction or
    search-time work at runtime.
    """

    if not isinstance(control_volume_geometry, LocalEmbeddedControlVolumeGeometry3D):
        raise TypeError(
            "control_volume_geometry must be LocalEmbeddedControlVolumeGeometry3D"
        )
    if not isinstance(boundary_bc, LocalControlVolumeBoundaryBC3D):
        raise TypeError("boundary_bc must be LocalControlVolumeBoundaryBC3D")
    if domain is not None and not isinstance(domain, LocalDomain3D):
        raise TypeError("domain must be a LocalDomain3D or None")

    faces = control_volume_geometry.irregular_faces
    rows = control_volume_geometry.face_functionals
    if rows is None:
        raise ValueError("control-volume geometry requires direct face functionals")
    if not isinstance(rows, LocalMomentFittedFaceRows3D):
        raise TypeError("face_functionals must be LocalMomentFittedFaceRows3D")
    if rows.layout != control_volume_geometry.cells.layout:
        raise ValueError("face functionals must share the control-volume layout")
    if rows.max_rows != faces.max_rows:
        raise ValueError("face functionals must align with irregular face rows")
    if boundary_bc.max_rows != faces.max_rows:
        raise ValueError("boundary BC rows must align with irregular face rows")
    if boundary_bc.max_patches != faces.max_patches:
        raise ValueError("boundary BC patches must align with irregular face rows")

    field = jnp.asarray(field_halo, dtype=jnp.float64)
    if field.ndim != 3 or field.shape != rows.layout.cell_halo_shape:
        raise ValueError(
            "field_halo must have shape "
            f"{rows.layout.cell_halo_shape}, got {field.shape}"
        )
    if int(rows.max_rows) == 0:
        return LocalControlVolumeFieldClosure3D.empty(max_rows=0)

    # Clip every dynamic gather.  The accompanying masks make a malformed,
    # referenced coordinate invalid rather than allowing it to select a value.
    nx, ny, nz = rows.layout.owned_shape
    hx, hy, hz = rows.layout.cell_halo_shape
    owned_in_bounds = (
        (rows.owned_i >= 0) & (rows.owned_i < nx)
        & (rows.owned_j >= 0) & (rows.owned_j < ny)
        & (rows.owned_k >= 0) & (rows.owned_k < nz)
    )
    halo_in_bounds = (
        (rows.halo_i >= 0) & (rows.halo_i < hx)
        & (rows.halo_j >= 0) & (rows.halo_j < hy)
        & (rows.halo_k >= 0) & (rows.halo_k < hz)
    )
    boundary_in_bounds = (
        (rows.boundary_face_row >= 0) & (rows.boundary_face_row < boundary_bc.max_rows)
        & (rows.boundary_patch >= 0) & (rows.boundary_patch < boundary_bc.max_patches)
        & (rows.boundary_quadrature >= 0) & (rows.boundary_quadrature < 4)
    )
    if domain is None:
        boundary_kind_all = jnp.expand_dims(boundary_bc.kind, axis=0)
        boundary_active_all = jnp.expand_dims(boundary_bc.active, axis=0)
        boundary_value_all = jnp.expand_dims(boundary_bc.quadrature_value, axis=0)
        n_boundary_shards = 1
    else:
        gathered_kind = boundary_bc.kind
        gathered_active = boundary_bc.active
        gathered_value = boundary_bc.quadrature_value
        for axis, (axis_name, shard_count) in enumerate(
            zip(domain.mesh_axis_names, domain.shard_spec.shard_counts)
        ):
            if int(shard_count) > 1:
                if axis_name is None:
                    raise ValueError(
                        "decomposed control-volume boundary observations require "
                        f"a mesh axis name for axis {axis}"
                    )
                gathered_kind = jax.lax.all_gather(
                    gathered_kind, axis_name, axis=0, tiled=False
                )
                gathered_active = jax.lax.all_gather(
                    gathered_active, axis_name, axis=0, tiled=False
                )
                gathered_value = jax.lax.all_gather(
                    gathered_value, axis_name, axis=0, tiled=False
                )
            else:
                gathered_kind = jnp.expand_dims(gathered_kind, axis=0)
                gathered_active = jnp.expand_dims(gathered_active, axis=0)
                gathered_value = jnp.expand_dims(gathered_value, axis=0)
        boundary_kind_all = jnp.reshape(
            gathered_kind, (-1, boundary_bc.max_rows)
        )
        boundary_active_all = jnp.reshape(
            gathered_active, (-1, boundary_bc.max_rows)
        )
        boundary_value_all = jnp.reshape(
            gathered_value,
            (-1, boundary_bc.max_rows, boundary_bc.max_patches, 4),
        )
        n_boundary_shards = int(np.prod(domain.shard_spec.shard_counts))
    boundary_source = rows.boundary_source_shard
    boundary_source_in_bounds = (
        (boundary_source >= 0) & (boundary_source < n_boundary_shards)
    )
    owned = field[rows.layout.owned_slices_cell]
    owned_sample = owned[
        jnp.clip(rows.owned_i, 0, nx - 1),
        jnp.clip(rows.owned_j, 0, ny - 1),
        jnp.clip(rows.owned_k, 0, nz - 1),
    ]
    halo_sample = field[
        jnp.clip(rows.halo_i, 0, hx - 1),
        jnp.clip(rows.halo_j, 0, hy - 1),
        jnp.clip(rows.halo_k, 0, hz - 1),
    ]
    boundary_row = jnp.clip(rows.boundary_face_row, 0, boundary_bc.max_rows - 1)
    boundary_patch = jnp.clip(rows.boundary_patch, 0, boundary_bc.max_patches - 1)
    boundary_quad = jnp.clip(rows.boundary_quadrature, 0, 3)
    boundary_source_clipped = jnp.clip(boundary_source, 0, n_boundary_shards - 1)
    boundary_sample = boundary_value_all[
        boundary_source_clipped, boundary_row, boundary_patch, boundary_quad
    ]
    boundary_kind = boundary_kind_all[boundary_source_clipped, boundary_row]
    boundary_active = boundary_active_all[boundary_source_clipped, boundary_row]

    is_owned = rows.observation_kind == CV_RECONSTRUCTION_EQUATION_CELL
    is_halo = rows.observation_kind == CV_RECONSTRUCTION_EQUATION_REMOTE_CELL
    is_dirichlet = rows.observation_kind == CV_RECONSTRUCTION_EQUATION_DIRICHLET
    observation_value = jnp.where(
        is_owned,
        owned_sample,
        jnp.where(is_halo, halo_sample, boundary_sample),
    )
    # Padded observations may point anywhere.  Mask their values *before* the
    # dot product so a poisoned dummy slot cannot turn 0 * NaN into NaN.
    observation_value = jnp.where(rows.observation_active, observation_value, 0.0)
    observation_valid = (
        (~rows.observation_active)
        | (
            (is_owned & owned_in_bounds & jnp.isfinite(owned_sample))
            | (is_halo & halo_in_bounds & jnp.isfinite(halo_sample))
            | (
                is_dirichlet
                & boundary_in_bounds
                & boundary_source_in_bounds
                & boundary_active
                & (boundary_kind == BC_DIRICHLET)
                & jnp.isfinite(boundary_sample)
            )
        )
    )
    weights_finite = jnp.all(
        (~rows.observation_active)
        | (
            jnp.isfinite(rows.projected_flux_weights)
            & jnp.isfinite(rows.parallel_flux_weights)
            & jnp.isfinite(rows.parallel_gradient_flux_weights)
        ),
        axis=1,
    )
    projected_flux = jnp.einsum(
        "re,re->r", rows.projected_flux_weights, observation_value
    )
    parallel_flux = jnp.einsum(
        "re,re->r", rows.parallel_flux_weights, observation_value
    )
    parallel_gradient_flux = jnp.einsum(
        "re,re->r", rows.parallel_gradient_flux_weights, observation_value
    )
    # The scalar traces are direct per-quadrature functionals.  Mask the
    # gathered observations before contraction so padded slots cannot poison
    # an otherwise valid row through NaN propagation.
    face_value = jnp.einsum(
        "rpqe,re->rpq", rows.value_weights, observation_value
    )
    face_gradient = jnp.einsum(
        "rpqce,re->rpqc", rows.logical_gradient_weights, observation_value
    )
    trace_weights_finite = (
        jnp.all(
            (~rows.active[:, None, None, None])
            | jnp.isfinite(rows.value_weights),
            axis=(1, 2, 3),
        )
        & jnp.all(
            (~rows.active[:, None, None, None, None])
            | jnp.isfinite(rows.logical_gradient_weights),
            axis=(1, 2, 3, 4),
        )
    )
    # The functional tensors are fixed at four patches for JIT shape
    # stability, while a particular face may use fewer patches/quadrature
    # points.  Never expose padded locations as usable traces.
    if int(faces.max_patches) > int(rows.max_patches):
        raise ValueError("face quadrature patches exceed face-functional patches")
    quadrature_active = jnp.zeros(
        (rows.max_rows, rows.max_patches, 4), dtype=bool
    ).at[:, : int(faces.max_patches), :].set(
        jnp.asarray(faces.quadrature_active, dtype=bool)
    )
    has_neighbor = faces.has_plus_owner | faces.has_remote_owner
    target_kind = boundary_bc.kind
    target_active = boundary_bc.active
    supported_target_bc = (
        (target_kind == BC_DIRICHLET)
        | (target_kind == BC_NORMALFLUX)
        | (target_kind == BC_NOFLUX)
    )
    target_boundary_valid = (
        has_neighbor
        | (target_active & supported_target_bc)
    )
    normal_flux = jnp.sum(
        jnp.where(
            faces.quadrature_active,
            faces.J
            * jnp.linalg.norm(faces.area_covector_weight, axis=-1)
            * boundary_bc.quadrature_value,
            0.0,
        ),
        axis=(1, 2),
    )
    use_normal_flux = (~has_neighbor) & (target_kind == BC_NORMALFLUX) & target_active
    use_no_flux = (~has_neighbor) & (target_kind == BC_NOFLUX) & target_active
    # A prescribed flux supplies no scalar value or gradient trace.  Keep the
    # flux closure usable, but mark these traces invalid so trace consumers
    # fail loudly instead of silently inventing a wall state.
    projected_flux = jnp.where(use_normal_flux, normal_flux, projected_flux)
    parallel_flux = jnp.where(use_normal_flux, normal_flux, parallel_flux)
    parallel_gradient_flux = jnp.where(
        use_normal_flux, normal_flux, parallel_gradient_flux
    )
    projected_flux = jnp.where(use_no_flux, 0.0, projected_flux)
    parallel_flux = jnp.where(use_no_flux, 0.0, parallel_flux)
    parallel_gradient_flux = jnp.where(
        use_no_flux, 0.0, parallel_gradient_flux
    )
    fitted_valid = (
        jnp.all(observation_valid, axis=1)
        & weights_finite
    )
    trace_valid = (
        jnp.broadcast_to(fitted_valid[:, None, None], face_value.shape)
        & jnp.broadcast_to(trace_weights_finite[:, None, None], face_value.shape)
        & quadrature_active
        & ~(use_normal_flux | use_no_flux)[:, None, None]
    )
    normal_inputs_valid = jnp.all(
        (~faces.quadrature_active)
        | (
            jnp.isfinite(faces.J)
            & jnp.all(jnp.isfinite(faces.area_covector_weight), axis=-1)
            & jnp.isfinite(boundary_bc.quadrature_value)
        ),
        axis=(1, 2),
    )
    # A prescribed normal flux replaces the fitted functional completely.
    # In particular, a row compiled with Dirichlet observations remains usable
    # for a field that prescribes NOFLUX/NORMALFLUX on that target face.
    functional_valid = jnp.where(
        use_normal_flux,
        normal_inputs_valid,
        jnp.where(use_no_flux, True, fitted_valid),
    )
    valid = (
        rows.active
        & faces.active
        & functional_valid
        & target_boundary_valid
        & jnp.isfinite(projected_flux)
        & jnp.isfinite(parallel_flux)
        & jnp.isfinite(parallel_gradient_flux)
    )
    return LocalControlVolumeFieldClosure3D(
        projected_flux=projected_flux,
        parallel_flux=parallel_flux,
        parallel_gradient_flux=parallel_gradient_flux,
        face_value=face_value,
        face_gradient=face_gradient,
        valid=valid,
        face_value_valid=trace_valid,
        face_gradient_valid=trace_valid,
        active=rows.active & faces.active,
        max_rows=rows.max_rows,
    )


def linear_combination_local_control_volume_closures(
    left: LocalControlVolumeFieldClosure3D,
    right: LocalControlVolumeFieldClosure3D,
    a: float | jnp.ndarray = 1.0,
    b: float | jnp.ndarray = 1.0,
) -> LocalControlVolumeFieldClosure3D:
    """Return the exact binary linear combination ``a*left + b*right``.

    This combines both integrated functionals and the stored quadrature
    traces.  A row is valid only when both operands and every corresponding
    active trace are valid; invalid active data is intentionally retained as
    NaN by ``LocalControlVolumeFieldClosure3D``.
    """
    if not isinstance(left, LocalControlVolumeFieldClosure3D) or not isinstance(right, LocalControlVolumeFieldClosure3D):
        raise TypeError("left and right must be LocalControlVolumeFieldClosure3D")
    if left.max_rows != right.max_rows:
        raise ValueError("closure row counts must match")
    aa = jnp.asarray(a, dtype=jnp.float64)
    bb = jnp.asarray(b, dtype=jnp.float64)
    active = jnp.asarray(left.active, dtype=bool) & jnp.asarray(right.active, dtype=bool)
    face_value_valid = jnp.asarray(left.face_value_valid, dtype=bool) & jnp.asarray(right.face_value_valid, dtype=bool)
    face_gradient_valid = jnp.asarray(left.face_gradient_valid, dtype=bool) & jnp.asarray(right.face_gradient_valid, dtype=bool)
    valid = active & jnp.asarray(left.valid, dtype=bool) & jnp.asarray(right.valid, dtype=bool)
    return LocalControlVolumeFieldClosure3D(
        projected_flux=aa * left.projected_flux + bb * right.projected_flux,
        parallel_flux=aa * left.parallel_flux + bb * right.parallel_flux,
        parallel_gradient_flux=aa * left.parallel_gradient_flux + bb * right.parallel_gradient_flux,
        face_value=aa * left.face_value + bb * right.face_value,
        face_gradient=aa * left.face_gradient + bb * right.face_gradient,
        valid=valid,
        face_value_valid=face_value_valid,
        face_gradient_valid=face_gradient_valid,
        active=active,
        max_rows=left.max_rows,
    )


def product_local_control_volume_closures(
    left: LocalControlVolumeFieldClosure3D,
    right: LocalControlVolumeFieldClosure3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
) -> LocalControlVolumeFieldClosure3D:
    """Multiply two primitive closures at face quadrature points.

    The three integrated fluxes are recomputed from the compact-face metric
    payload, rather than multiplying already-integrated primitive fluxes.  In
    particular, this preserves the covariance of a product on an angularly
    agglomerated face.
    """
    if not isinstance(left, LocalControlVolumeFieldClosure3D) or not isinstance(right, LocalControlVolumeFieldClosure3D):
        raise TypeError("left and right must be LocalControlVolumeFieldClosure3D")
    if not isinstance(control_volume_geometry, LocalEmbeddedControlVolumeGeometry3D):
        raise TypeError("control_volume_geometry must be LocalEmbeddedControlVolumeGeometry3D")
    faces = control_volume_geometry.irregular_faces
    if left.max_rows != faces.max_rows or right.max_rows != faces.max_rows:
        raise ValueError("closure rows must align with irregular face rows")
    qactive = jnp.asarray(faces.quadrature_active, dtype=bool)
    value_valid = jnp.asarray(left.face_value_valid, dtype=bool) & jnp.asarray(right.face_value_valid, dtype=bool)
    gradient_valid = jnp.asarray(left.face_gradient_valid, dtype=bool) & jnp.asarray(right.face_gradient_valid, dtype=bool)
    product_value = left.face_value * right.face_value
    product_gradient = left.face_value[..., None] * right.face_gradient + right.face_value[..., None] * left.face_gradient
    product_value_valid = value_valid & qactive
    product_gradient_valid = gradient_valid & value_valid & qactive
    area = jnp.asarray(faces.area_covector_weight, dtype=jnp.float64)
    J = jnp.asarray(faces.J, dtype=jnp.float64)
    Bcontra = jnp.asarray(faces.B_contra, dtype=jnp.float64)
    Bmag = jnp.asarray(faces.Bmag, dtype=jnp.float64)
    projector = jnp.asarray(faces.projector, dtype=jnp.float64)
    bunit = Bcontra / Bmag[..., None]
    parallel_weight = J * jnp.einsum("...a,...a->...", area, bunit)
    projected_weight = J[..., None] * jnp.einsum("...a,...ab->...b", area, projector)
    parallel_gradient_weight = parallel_weight[..., None] * bunit
    product_value_for_flux = jnp.where(product_value_valid, product_value, jnp.nan)
    product_gradient_for_flux = jnp.where(product_gradient_valid[..., None], product_gradient, jnp.nan)
    parallel_flux = jnp.sum(jnp.where(qactive, parallel_weight * product_value_for_flux, 0.0), axis=(1, 2))
    projected_integrand = jnp.einsum("...a,...a->...", projected_weight, product_gradient_for_flux)
    parallel_gradient_integrand = jnp.einsum("...a,...a->...", parallel_gradient_weight, product_gradient_for_flux)
    projected_flux = jnp.sum(jnp.where(qactive, projected_integrand, 0.0), axis=(1, 2))
    parallel_gradient_flux = jnp.sum(jnp.where(qactive, parallel_gradient_integrand, 0.0), axis=(1, 2))
    active = jnp.asarray(left.active, dtype=bool) & jnp.asarray(right.active, dtype=bool) & jnp.asarray(faces.active, dtype=bool)
    valid = active & jnp.asarray(left.valid, dtype=bool) & jnp.asarray(right.valid, dtype=bool)
    valid = valid & jnp.all(~qactive | (product_value_valid & product_gradient_valid), axis=(1, 2))
    return LocalControlVolumeFieldClosure3D(
        projected_flux=projected_flux,
        parallel_flux=parallel_flux,
        parallel_gradient_flux=parallel_gradient_flux,
        face_value=product_value,
        face_gradient=product_gradient,
        valid=valid,
        face_value_valid=product_value_valid,
        face_gradient_valid=product_gradient_valid,
        active=active,
        max_rows=faces.max_rows,
    )


def build_local_control_volume_polynomial_from_field(
    field_halo: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    context: StencilBuilderContext,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    boundary_bc: LocalControlVolumeBoundaryBC3D,
    regular_face_bc: LocalBoundaryFaceBC3D | None = None,
    *,
    halo_exchange: HaloExchange3D | None = None,
    topology_filler: TopologyHaloFiller3D | None = None,
) -> LocalControlVolumePolynomial3D:
    """Evaluate precomputed moment-aware reconstruction for one scalar field."""

    if not isinstance(control_volume_geometry, LocalEmbeddedControlVolumeGeometry3D):
        raise TypeError(
            "control_volume_geometry must be LocalEmbeddedControlVolumeGeometry3D"
        )
    if not isinstance(boundary_bc, LocalControlVolumeBoundaryBC3D):
        raise TypeError("boundary_bc must be LocalControlVolumeBoundaryBC3D")
    if control_volume_geometry.layout != geometry.layout:
        raise ValueError("control-volume geometry must share geometry.layout")
    rows = control_volume_geometry.reconstruction
    cells = control_volume_geometry.cells
    if boundary_bc.max_rows != control_volume_geometry.irregular_faces.max_rows:
        raise ValueError("boundary BC rows must align with irregular face rows")

    local = build_local_stencil_from_field(field_halo, geometry, context)
    baseline_gradient = jnp.stack(
        (
            _take_stencil_finite_difference(local.x),
            _take_stencil_finite_difference(local.y),
            _take_stencil_finite_difference(local.z),
        ),
        axis=-1,
    )
    baseline_gradient = jnp.nan_to_num(
        baseline_gradient,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    owned_values = jnp.asarray(
        field_halo[geometry.layout.owned_slices_cell],
        dtype=jnp.float64,
    )
    gradient = jnp.where(
        cells.is_active_owner[..., None],
        baseline_gradient,
        0.0,
    )
    regular_boundary_closure = control_volume_geometry.regular_boundary_closure
    effective_regular_face_bc = (
        regular_face_bc
        if regular_face_bc is not None
        else LocalBoundaryFaceBC3D.empty(geometry.layout)
    )
    hessian = jnp.zeros(geometry.owned_shape + (3, 3), dtype=jnp.float64)
    valid = cells.is_active_owner & jnp.all(jnp.isfinite(gradient), axis=-1)
    order_owned = jnp.zeros(geometry.owned_shape, dtype=jnp.int32)
    condition_owned = jnp.full(
        geometry.owned_shape,
        jnp.inf,
        dtype=jnp.float64,
    )
    if int(rows.max_rows) == 0:
        if regular_boundary_closure is not None:
            gradient = _patch_local_regular_boundary_owner_gradients(
                gradient,
                values_owned=owned_values,
                geometry=geometry,
                domain=domain,
                face_bc=effective_regular_face_bc,
                closure=regular_boundary_closure,
            )
        polynomial = LocalControlVolumePolynomial3D(
            gradient=gradient,
            hessian=hessian,
            valid=valid,
            polynomial_order=order_owned,
            condition_number=condition_owned,
            owner_values=owned_values,
        )
        return _attach_remote_control_volume_face_samples(
            polynomial,
            owned_values,
            cells,
            control_volume_geometry.irregular_faces,
            domain,
            halo_exchange=halo_exchange,
            topology_filler=topology_filler,
        )

    target_value = owned_values[rows.target_i, rows.target_j, rows.target_k]
    sample_value = owned_values[
        jnp.clip(rows.sample_i, 0, geometry.owned_shape[0] - 1),
        jnp.clip(rows.sample_j, 0, geometry.owned_shape[1] - 1),
        jnp.clip(rows.sample_k, 0, geometry.owned_shape[2] - 1),
    ]
    boundary_row = jnp.clip(
        rows.boundary_face_row,
        0,
        max(0, boundary_bc.max_rows - 1),
    )
    boundary_patch = jnp.clip(
        rows.boundary_patch,
        0,
        max(0, control_volume_geometry.irregular_faces.max_patches - 1),
    )
    boundary_quadrature = jnp.clip(rows.boundary_quadrature, 0, 3)
    if int(boundary_bc.max_rows) == 0:
        boundary_value = jnp.zeros_like(sample_value)
    else:
        boundary_value = boundary_bc.quadrature_value[
            boundary_row,
            boundary_patch,
            boundary_quadrature,
        ]
    remote_sample_value = field_halo[
        rows.sample_i,
        rows.sample_j,
        rows.sample_k,
    ]
    remote_sample_valid = jnp.isfinite(remote_sample_value)
    rhs = jnp.where(
        rows.equation_kind == CV_RECONSTRUCTION_EQUATION_CELL,
        sample_value - target_value[:, None],
        jnp.where(
            rows.equation_kind == CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
            remote_sample_value - target_value[:, None],
            boundary_value - target_value[:, None],
        ),
    )
    dirichlet_equation = (
        rows.equation_kind == CV_RECONSTRUCTION_EQUATION_DIRICHLET
    )
    if int(boundary_bc.max_rows) == 0:
        boundary_valid = jnp.zeros_like(dirichlet_equation)
    else:
        boundary_valid = (
            boundary_bc.active[boundary_row]
            & (boundary_bc.kind[boundary_row] == BC_DIRICHLET)
        )
    equation_valid = (
        rows.equation_active
        & (
            (rows.equation_kind == CV_RECONSTRUCTION_EQUATION_CELL)
            | (
                (rows.equation_kind == CV_RECONSTRUCTION_EQUATION_REMOTE_CELL)
                & remote_sample_valid
            )
            | (dirichlet_equation & boundary_valid)
        )
        & jnp.isfinite(rhs)
    )
    # Metadata is built with all Dirichlet rows active.  If a field supplies a
    # different BC kind, invalidate the complete row rather than applying a
    # transform whose normal equations no longer match its equation set.
    row_valid = (
        rows.active
        & jnp.all((~rows.equation_active) | equation_valid, axis=-1)
        & jnp.isfinite(target_value)
    )
    rhs = jnp.where(equation_valid, rhs, 0.0)
    coefficients = jnp.einsum("rie,re->ri", rows.rhs_transform, rhs)
    coefficients = jnp.nan_to_num(
        coefficients,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    row_gradient = coefficients[:, :3]
    row_hessian = jnp.zeros((int(rows.max_rows), 3, 3), dtype=jnp.float64)
    row_hessian = row_hessian.at[:, 0, 0].set(coefficients[:, 3])
    row_hessian = row_hessian.at[:, 1, 1].set(coefficients[:, 4])
    row_hessian = row_hessian.at[:, 2, 2].set(coefficients[:, 5])
    row_hessian = row_hessian.at[:, 0, 1].set(coefficients[:, 6])
    row_hessian = row_hessian.at[:, 1, 0].set(coefficients[:, 6])
    row_hessian = row_hessian.at[:, 0, 2].set(coefficients[:, 7])
    row_hessian = row_hessian.at[:, 2, 0].set(coefficients[:, 7])
    row_hessian = row_hessian.at[:, 1, 2].set(coefficients[:, 8])
    row_hessian = row_hessian.at[:, 2, 1].set(coefficients[:, 8])
    row_hessian = jnp.where(
        (rows.polynomial_order >= 2)[:, None, None],
        row_hessian,
        0.0,
    )
    row_third = jnp.zeros((int(rows.max_rows), 3, 3, 3), dtype=jnp.float64)
    if rows.rhs_transform.shape[1] == 19:
        cubic = coefficients[:, 9:]
        cubic_indices = (
            (0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 0, 1), (0, 0, 2),
            (0, 1, 1), (0, 2, 2), (1, 1, 2), (1, 2, 2), (0, 1, 2),
        )
        for index, axes in enumerate(cubic_indices):
            for permutation in set(permutations(axes)):
                row_third = row_third.at[(slice(None),) + permutation].set(cubic[:, index])
        row_third = jnp.where(
            (rows.polynomial_order == 3)[:, None, None, None], row_third, 0.0
        )
    # Padded per-shard row tables contain inactive rows whose sanitized target
    # index is (0, 0, 0).  A scatter-set over the complete padded table lets
    # those dummy rows overwrite a real reconstruction at that cell.  The
    # dense row map is authoritative and has exactly one row per target, so
    # gather through it instead.
    row_for_cell = jnp.asarray(rows.target_row_for_cell, dtype=jnp.int32)
    has_row = row_for_cell >= 0
    safe_row = jnp.clip(row_for_cell, 0, max(0, int(rows.max_rows) - 1))
    gathered_valid = row_valid[safe_row]
    gradient = jnp.where(
        has_row[..., None],
        jnp.where(gathered_valid[..., None], row_gradient[safe_row], 0.0),
        gradient,
    )
    hessian = jnp.where(
        has_row[..., None, None],
        jnp.where(
            gathered_valid[..., None, None],
            row_hessian[safe_row],
            0.0,
        ),
        hessian,
    )
    third_derivative = jnp.zeros(geometry.owned_shape + (3, 3, 3), dtype=jnp.float64)
    third_derivative = jnp.where(
        has_row[..., None, None, None],
        jnp.where(gathered_valid[..., None, None, None], row_third[safe_row], 0.0),
        third_derivative,
    )
    valid = jnp.where(has_row, gathered_valid, valid)
    order_owned = jnp.where(
        has_row,
        jnp.where(gathered_valid, rows.polynomial_order[safe_row], 0),
        order_owned,
    )
    condition_owned = jnp.where(
        has_row,
        jnp.where(
            gathered_valid,
            rows.condition_number[safe_row],
            jnp.inf,
        ),
        condition_owned,
    )
    # Apply the regular physical-boundary closure after row reconstruction.
    # Reconstruction rows are authoritative around embedded geometry, but a
    # regular first owner plane must retain the moment-consistent derivative
    # from its Dirichlet face and three inward finite-volume averages.
    if regular_boundary_closure is not None:
        gradient = _patch_local_regular_boundary_owner_gradients(
            gradient,
            values_owned=owned_values,
            geometry=geometry,
            domain=domain,
            face_bc=effective_regular_face_bc,
            closure=regular_boundary_closure,
        )
    gradient = jnp.where(cells.is_active_owner[..., None], gradient, 0.0)
    hessian = jnp.where(cells.is_active_owner[..., None, None], hessian, 0.0)
    third_derivative = jnp.where(
        cells.is_active_owner[..., None, None, None], third_derivative, 0.0
    )
    valid = valid & cells.is_active_owner
    polynomial = LocalControlVolumePolynomial3D(
        gradient=gradient,
        hessian=hessian,
        third_derivative=third_derivative,
        valid=valid,
        polynomial_order=order_owned,
        condition_number=condition_owned,
        owner_values=owned_values,
    )
    return _attach_remote_control_volume_face_samples(
        polynomial,
        owned_values,
        cells,
        control_volume_geometry.irregular_faces,
        domain,
        halo_exchange=halo_exchange,
        topology_filler=topology_filler,
    )


def _patch_local_regular_boundary_owner_gradients(
    gradient: jnp.ndarray,
    *,
    values_owned: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    face_bc: LocalBoundaryFaceBC3D,
    closure: LocalRegularBoundaryMomentClosure3D,
) -> jnp.ndarray:
    """Apply moment-aware Dirichlet derivatives at first owner centroids."""

    if closure.layout != geometry.layout or face_bc.layout != geometry.layout:
        raise ValueError(
            "regular boundary closure, face BC, and geometry must share layout"
        )
    result = jnp.asarray(gradient, dtype=jnp.float64)
    values_owned = jnp.asarray(values_owned, dtype=jnp.float64)
    kinds = (face_bc.kind_x, face_bc.kind_y, face_bc.kind_z)
    values = (face_bc.value_x, face_bc.value_y, face_bc.value_z)
    masks = (face_bc.mask_x, face_bc.mask_y, face_bc.mask_z)
    for axis in range(3):
        if geometry.owned_shape[axis] < 3:
            continue
        _face_weights, owner_weights, closure_valid = closure.axis_payload(
            axis
        )
        inward = jnp.moveaxis(values_owned, axis, 0)
        lower_samples = inward[:3]
        upper_samples = jnp.flip(inward[-3:], axis=0)
        for side, samples, cell_index, face_index in (
            ("lower", lower_samples, 0, 0),
            ("upper", upper_samples, -1, -1),
        ):
            if (
                axis == 0
                and side == "lower"
                and domain.axis_regular_axes[axis]
            ):
                continue
            try:
                physical = (
                    domain.runtime_has_physical_lower(axis)
                    if side == "lower"
                    else domain.runtime_has_physical_upper(axis)
                )
            except NameError:
                # Focused host-side operator checks do not bind mesh axis
                # names. Production shard_map calls retain runtime ownership.
                physical = (
                    domain.has_physical_lower(axis)
                    if side == "lower"
                    else domain.has_physical_upper(axis)
                )
            axis_weights = owner_weights[
                _axis_index_nd(axis, face_index, owner_weights.ndim)
            ]
            axis_valid = closure_valid[
                _axis_index_nd(axis, face_index, closure_valid.ndim)
            ]
            axis_kind = kinds[axis][
                _axis_index_nd(axis, face_index, kinds[axis].ndim)
            ]
            axis_value = values[axis][
                _axis_index_nd(axis, face_index, values[axis].ndim)
            ]
            axis_mask = masks[axis][
                _axis_index_nd(axis, face_index, masks[axis].ndim)
            ]
            derivative = (
                axis_weights[..., 0] * axis_value
                + jnp.einsum(
                    "...m,m...->...",
                    axis_weights[..., 1:],
                    samples,
                )
            )
            patch = (
                physical
                & axis_valid
                & axis_mask
                & (axis_kind == BC_DIRICHLET)
                & jnp.isfinite(derivative)
            )
            plane_index = _axis_index_nd(
                axis,
                cell_index,
                result.ndim,
            )
            plane = result[plane_index]
            plane = plane.at[..., axis].set(
                jnp.where(patch, derivative, plane[..., axis])
            )
            result = result.at[plane_index].set(plane)
    return result


def _attach_remote_control_volume_face_samples(
    polynomial: LocalControlVolumePolynomial3D,
    values_owned: jnp.ndarray,
    cells: LocalControlVolumeCellGeometry3D,
    faces: LocalControlVolumeFaceRows3D,
    domain: LocalDomain3D,
    *,
    halo_exchange: HaloExchange3D | None,
    topology_filler: TopologyHaloFiller3D | None,
) -> LocalControlVolumePolynomial3D:
    """Exchange mapped-owner polynomials and sample mirrored remote face rows."""

    quadrature_shape = (
        int(faces.max_rows),
        int(faces.max_patches),
        4,
    )
    remote_value = jnp.zeros(quadrature_shape, dtype=jnp.float64)
    remote_gradient = jnp.zeros(quadrature_shape + (3,), dtype=jnp.float64)
    remote_valid = jnp.zeros(quadrature_shape, dtype=bool)
    if int(faces.max_rows) == 0 or halo_exchange is None:
        return LocalControlVolumePolynomial3D(
            gradient=polynomial.gradient,
            hessian=polynomial.hessian,
            third_derivative=polynomial.third_derivative,
            valid=polynomial.valid,
            polynomial_order=polynomial.polynomial_order,
            condition_number=polynomial.condition_number,
            owner_values=polynomial.owner_values,
            remote_face_value=remote_value,
            remote_face_gradient=remote_gradient,
            remote_face_valid=remote_valid,
        )

    owner_index = (cells.owner_i, cells.owner_j, cells.owner_k)
    owner_value = jnp.asarray(values_owned, dtype=jnp.float64)[owner_index]
    owner_gradient = polynomial.gradient[owner_index]
    owner_hessian = polynomial.hessian[owner_index]
    owner_third = polynomial.third_derivative[owner_index]
    owner_valid = polynomial.valid[owner_index]
    packed_owned = jnp.concatenate(
        (
            owner_value[..., None],
            owner_gradient,
            owner_hessian.reshape(cells.shape + (9,)),
            owner_third.reshape(cells.shape + (27,)),
            owner_valid[..., None].astype(jnp.float64),
        ),
        axis=-1,
    )
    packed_halo = inject_owned_vector_field_to_halo(
        packed_owned,
        domain.layout,
    )
    packed_halo = halo_exchange(packed_halo, domain)
    if topology_filler is not None:
        packed_halo = topology_filler(packed_halo, domain)

    remote_payload = packed_halo[
        faces.remote_halo_i,
        faces.remote_halo_j,
        faces.remote_halo_k,
    ]
    remote_owner_value = remote_payload[:, 0]
    remote_owner_gradient = remote_payload[:, 1:4]
    remote_hessian = remote_payload[:, 4:13].reshape((-1, 3, 3))
    remote_third = remote_payload[:, 13:40].reshape((-1, 3, 3, 3))
    remote_owner_valid = remote_payload[:, 40] > 0.5
    remote_centroid = faces.remote_centroid
    remote_second_moment = faces.remote_second_moment
    remote_third_moment = faces.remote_third_moment
    displacement = faces.quadrature_points - remote_centroid[:, None, None, :]
    remote_gradient = (
        remote_owner_gradient[:, None, None, :]
        + jnp.einsum(
            "rij,rpqj->rpqi",
            remote_hessian,
            displacement,
        )
    )
    remote_gradient = remote_gradient + 0.5 * jnp.einsum(
        "rijk,rpqj,rpqk->rpqi",
        remote_third,
        displacement,
        displacement,
    )
    quadratic_moment = (
        displacement[..., :, None] * displacement[..., None, :]
        - remote_second_moment[:, None, None, :, :]
    )
    remote_value = (
        remote_owner_value[:, None, None]
        + jnp.einsum(
            "ri,rpqi->rpq",
            remote_owner_gradient,
            displacement,
        )
        + 0.5
        * jnp.einsum(
            "rij,rpqij->rpq",
            remote_hessian,
            quadratic_moment,
        )
    )
    remote_value = remote_value + (1.0 / 6.0) * jnp.einsum(
        "rijk,rpqijk->rpq",
        remote_third,
        displacement[..., :, None, None]
        * displacement[..., None, :, None]
        * displacement[..., None, None, :],
    )
    remote_value = remote_value - (1.0 / 6.0) * jnp.einsum(
        "rijk,rijk->r", remote_third, remote_third_moment
    )[:, None, None]
    remote_row = faces.has_remote_owner[:, None, None]
    remote_valid = (
        remote_row
        & faces.quadrature_active
        & remote_owner_valid[:, None, None]
        & jnp.isfinite(remote_value)
        & jnp.all(jnp.isfinite(remote_gradient), axis=-1)
    )
    remote_value = jnp.where(remote_valid, remote_value, 0.0)
    remote_gradient = jnp.where(
        remote_valid[..., None],
        remote_gradient,
        0.0,
    )
    return LocalControlVolumePolynomial3D(
        gradient=polynomial.gradient,
        hessian=polynomial.hessian,
        third_derivative=polynomial.third_derivative,
        valid=polynomial.valid,
        polynomial_order=polynomial.polynomial_order,
        condition_number=polynomial.condition_number,
        owner_values=polynomial.owner_values,
        remote_face_value=remote_value,
        remote_face_gradient=remote_gradient,
        remote_face_valid=remote_valid,
    )


def expand_local_control_volume_owner_field(
    values_owned: jnp.ndarray,
    cells: LocalControlVolumeCellGeometry3D,
    *,
    owner_values_halo: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Fill each positive-volume storage cell from its active owner.

    This storage expansion is used only while preparing halos and dense face
    stencils.  Conservative outputs are still accumulated into unique owners,
    and merged source output remains zero.
    """

    values = jnp.asarray(values_owned, dtype=jnp.float64)
    if values.shape != cells.shape:
        raise ValueError(
            f"values_owned must have shape {cells.shape}, got {values.shape}"
        )
    expanded = values[cells.owner_i, cells.owner_j, cells.owner_k]
    if owner_values_halo is None:
        try:
            if bool(jnp.any(cells.owner_is_remote)):
                raise ValueError("owner_values_halo is required when control-volume owners are remote")
        except jax.errors.TracerBoolConversionError:
            pass
    else:
        halo = jnp.asarray(owner_values_halo, dtype=values.dtype)
        if halo.shape[:3] != cells.layout.cell_halo_shape:
            raise ValueError("owner_values_halo must match cells.layout.cell_halo_shape")
        remote = halo[cells.remote_owner_halo_i, cells.remote_owner_halo_j, cells.remote_owner_halo_k]
        expanded = jnp.where(cells.owner_is_remote, remote, expanded)
    return jnp.where(cells.raw_volume > 0.0, expanded, 0.0)


def aggregate_local_control_volume_average(
    values_raw: jnp.ndarray,
    cells: LocalControlVolumeCellGeometry3D,
    domain: LocalDomain3D,
) -> jnp.ndarray:
    """Aggregate raw-cell averages onto unique local control-volume owners."""

    if not isinstance(cells, LocalControlVolumeCellGeometry3D):
        raise TypeError(
            "cells must be LocalControlVolumeCellGeometry3D, "
            f"got {type(cells).__name__}"
        )
    if not isinstance(domain, LocalDomain3D):
        raise TypeError(
            "domain must be LocalDomain3D, "
            f"got {type(domain).__name__}"
        )
    if cells.layout != domain.layout:
        raise ValueError("cells.layout must match domain.layout")

    values = jnp.asarray(values_raw)
    if values.ndim != 3 or tuple(values.shape) != tuple(cells.shape):
        raise ValueError(
            "values_raw must be a scalar field with cells.shape; "
            f"got {values.shape}, expected {cells.shape}"
        )
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError(
            "values_raw must have a floating-point dtype, "
            f"got {values.dtype}"
        )
    values = jnp.asarray(values, dtype=jnp.float64)

    raw_volume = jnp.asarray(cells.raw_volume, dtype=jnp.float64)
    integrated = raw_volume * values
    local_source = (raw_volume > 0.0) & ~cells.owner_is_remote
    owner_sum = jnp.zeros(cells.shape, dtype=jnp.float64).at[
        cells.owner_i,
        cells.owner_j,
        cells.owner_k,
    ].add(jnp.where(local_source, integrated, 0.0))

    remote_halo = jnp.zeros(cells.layout.cell_halo_shape, dtype=jnp.float64).at[
        cells.remote_owner_halo_i,
        cells.remote_owner_halo_j,
        cells.remote_owner_halo_k,
    ].add(jnp.where(cells.owner_is_remote & (raw_volume > 0.0), integrated, 0.0))
    owner_sum = owner_sum + accumulate_halo_contributions_to_owned(
        remote_halo,
        domain,
    )

    active = jnp.asarray(cells.is_active_owner, dtype=bool)
    safe_volume = jnp.where(
        active,
        jnp.asarray(cells.aggregate_volume, dtype=jnp.float64),
        1.0,
    )
    result = owner_sum / safe_volume
    return jnp.where(active, result, 0.0).astype(jnp.float64)


def local_control_volume_product_average(
    left_owned: jnp.ndarray,
    right_owned: jnp.ndarray,
    left_polynomial: LocalControlVolumePolynomial3D,
    right_polynomial: LocalControlVolumePolynomial3D,
    cells: LocalControlVolumeCellGeometry3D,
) -> jnp.ndarray:
    """Return a second-order control-volume average of a scalar product.

    The stored operands are finite-volume averages, so multiplying them drops
    the leading covariance term.  The quadratic reconstruction is centered in
    the aggregate fluid centroid and ``cells.second_moment`` is its normalized
    central moment, giving

    ``<left * right> = <left><right> + grad(left) M2 grad(right) + O(h^3)``.

    The correction is only used where both reconstructions are valid.  This is
    sufficient for second-order conservative fluxes without requiring third or
    fourth aggregate moments at runtime.
    """

    left = jnp.asarray(left_owned, dtype=jnp.float64)
    right = jnp.asarray(right_owned, dtype=jnp.float64)
    if left.shape != cells.shape:
        raise ValueError(
            f"left_owned must have shape {cells.shape}, got {left.shape}"
        )
    if right.shape != cells.shape:
        raise ValueError(
            f"right_owned must have shape {cells.shape}, got {right.shape}"
        )
    covariance = jnp.einsum(
        "...i,...ij,...j->...",
        left_polynomial.gradient,
        jnp.asarray(cells.second_moment, dtype=jnp.float64),
        right_polynomial.gradient,
    )
    valid = (
        jnp.asarray(cells.is_active_owner, dtype=bool)
        & jnp.asarray(left_polynomial.valid, dtype=bool)
        & jnp.asarray(right_polynomial.valid, dtype=bool)
        & jnp.isfinite(covariance)
    )
    product = left * right
    corrected = jnp.where(valid, product + covariance, product)
    return jnp.where(cells.is_active_owner, corrected, 0.0)


def evaluate_local_control_volume_polynomial(
    values_owned: jnp.ndarray,
    polynomial: LocalControlVolumePolynomial3D,
    cells: LocalControlVolumeCellGeometry3D,
    owner_i: jnp.ndarray,
    owner_j: jnp.ndarray,
    owner_k: jnp.ndarray,
    points: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate one finite-volume polynomial and its gradient at logical points."""

    values = jnp.asarray(values_owned, dtype=jnp.float64)
    if values.shape != cells.shape:
        raise ValueError(f"values_owned must have shape {cells.shape}, got {values.shape}")
    owner_i = jnp.asarray(owner_i, dtype=jnp.int32)
    owner_j = jnp.asarray(owner_j, dtype=jnp.int32)
    owner_k = jnp.asarray(owner_k, dtype=jnp.int32)
    points = jnp.asarray(points, dtype=jnp.float64)
    owner_value = values[owner_i, owner_j, owner_k]
    owner_centroid = cells.centroid[owner_i, owner_j, owner_k]
    owner_m2 = cells.second_moment[owner_i, owner_j, owner_k]
    owner_m3 = cells.third_moment[owner_i, owner_j, owner_k]
    owner_gradient = polynomial.gradient[owner_i, owner_j, owner_k]
    owner_hessian = polynomial.hessian[owner_i, owner_j, owner_k]
    owner_third = polynomial.third_derivative[owner_i, owner_j, owner_k]
    owner_valid = polynomial.valid[owner_i, owner_j, owner_k]
    displacement = points - owner_centroid
    point_gradient = owner_gradient + jnp.einsum(
        "...ij,...j->...i",
        owner_hessian,
        displacement,
    )
    point_gradient = point_gradient + 0.5 * jnp.einsum(
        "...ijk,...j,...k->...i",
        owner_third,
        displacement,
        displacement,
    )
    quadratic_moment = (
        displacement[..., :, None] * displacement[..., None, :]
        - owner_m2
    )
    point_value = (
        owner_value
        + jnp.einsum("...i,...i->...", owner_gradient, displacement)
        + 0.5
        * jnp.einsum("...ij,...ij->...", owner_hessian, quadratic_moment)
        + (1.0 / 6.0)
        * jnp.einsum(
            "...ijk,...ijk->...",
            owner_third,
            displacement[..., :, None, None]
            * displacement[..., None, :, None]
            * displacement[..., None, None, :]
            - owner_m3,
        )
    )
    return point_value, point_gradient, owner_valid


def replace_local_control_volume_projected_flux_with_owner_polynomials(
    field_closure: LocalControlVolumeFieldClosure3D,
    polynomial: LocalControlVolumePolynomial3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    domain: LocalDomain3D,
    *,
    radial_axis: int = 0,
) -> LocalControlVolumeFieldClosure3D:
    """Replace selected compact projected fluxes with owner-polynomial fluxes.

    Interior faces use one conservative flux obtained by averaging the two
    adjacent owner reconstructions.  A remote plus owner is evaluated from the
    polynomial payload already exchanged by
    :func:`build_local_control_volume_polynomial_from_field`; no second face
    row or residual scatter is introduced.

    The replacement is the default compact-face closure.  It excludes the
    first two owner layers next to each global radial boundary, where the
    regular moment closure and direct face functional remain authoritative.  If
    a selected owner polynomial is invalid, the closure row is invalidated
    rather than silently falling back to a different face formula.
    """

    if not isinstance(field_closure, LocalControlVolumeFieldClosure3D):
        raise TypeError(
            "field_closure must be a LocalControlVolumeFieldClosure3D"
        )
    if not isinstance(polynomial, LocalControlVolumePolynomial3D):
        raise TypeError("polynomial must be a LocalControlVolumePolynomial3D")
    if not isinstance(
        control_volume_geometry,
        LocalEmbeddedControlVolumeGeometry3D,
    ):
        raise TypeError(
            "control_volume_geometry must be a "
            "LocalEmbeddedControlVolumeGeometry3D"
        )
    if not isinstance(domain, LocalDomain3D):
        raise TypeError("domain must be a LocalDomain3D")
    radial_axis = int(radial_axis)
    if radial_axis not in (0, 1, 2):
        raise ValueError("radial_axis must be 0, 1, or 2")
    faces = control_volume_geometry.irregular_faces
    cells = control_volume_geometry.cells
    if field_closure.max_rows != faces.max_rows:
        raise ValueError(
            "field_closure rows must align with irregular face rows"
        )
    if polynomial.shape != cells.shape:
        raise ValueError("polynomial shape must match control-volume cells")
    if polynomial.owner_values is None:
        raise ValueError("polynomial.owner_values are required")

    quadrature_points = jnp.asarray(
        faces.quadrature_points,
        dtype=jnp.float64,
    )
    quadrature_shape = quadrature_points.shape[:-1]
    if polynomial.remote_face_gradient.shape != quadrature_shape + (3,):
        raise ValueError(
            "polynomial.remote_face_gradient must align with face quadrature"
        )
    if polynomial.remote_face_valid.shape != quadrature_shape:
        raise ValueError(
            "polynomial.remote_face_valid must align with face quadrature"
        )

    def _broadcast_owner(
        owner_i: jnp.ndarray,
        owner_j: jnp.ndarray,
        owner_k: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return tuple(
            jnp.broadcast_to(component[:, None, None], quadrature_shape)
            for component in (owner_i, owner_j, owner_k)
        )

    minus_owner = _broadcast_owner(
        faces.minus_owner_i,
        faces.minus_owner_j,
        faces.minus_owner_k,
    )
    _minus_value, minus_gradient, minus_valid = (
        evaluate_local_control_volume_polynomial(
            polynomial.owner_values,
            polynomial,
            cells,
            *minus_owner,
            quadrature_points,
        )
    )
    plus_owner = _broadcast_owner(
        faces.plus_owner_i,
        faces.plus_owner_j,
        faces.plus_owner_k,
    )
    _plus_value, local_plus_gradient, local_plus_valid = (
        evaluate_local_control_volume_polynomial(
            polynomial.owner_values,
            polynomial,
            cells,
            *plus_owner,
            quadrature_points,
        )
    )
    plus_gradient = jnp.where(
        faces.has_remote_owner[:, None, None, None],
        polynomial.remote_face_gradient,
        local_plus_gradient,
    )
    plus_valid = jnp.where(
        faces.has_remote_owner[:, None, None],
        polynomial.remote_face_valid,
        local_plus_valid,
    )

    quadrature_active = jnp.asarray(faces.quadrature_active, dtype=bool)
    face_weight = (
        jnp.asarray(faces.J, dtype=jnp.float64)[..., None]
        * jnp.asarray(faces.area_covector_weight, dtype=jnp.float64)
    )
    projector = jnp.asarray(faces.projector, dtype=jnp.float64)

    def _integrated_flux(gradient: jnp.ndarray) -> jnp.ndarray:
        point_flux = jnp.einsum(
            "...i,...ij,...j->...",
            face_weight,
            projector,
            gradient,
        )
        return jnp.sum(
            jnp.where(quadrature_active, point_flux, 0.0),
            axis=(-2, -1),
        )

    minus_flux = _integrated_flux(minus_gradient)
    plus_flux = _integrated_flux(plus_gradient)
    minus_row_valid = jnp.all(
        (~quadrature_active)
        | (
            minus_valid
            & jnp.all(jnp.isfinite(minus_gradient), axis=-1)
        ),
        axis=(-2, -1),
    ) & jnp.isfinite(minus_flux)
    plus_row_valid = jnp.all(
        (~quadrature_active)
        | (
            plus_valid
            & jnp.all(jnp.isfinite(plus_gradient), axis=-1)
        ),
        axis=(-2, -1),
    ) & jnp.isfinite(plus_flux)

    owner_axis = (
        (faces.minus_owner_i, faces.plus_owner_i, faces.remote_halo_i),
        (faces.minus_owner_j, faces.plus_owner_j, faces.remote_halo_j),
        (faces.minus_owner_k, faces.plus_owner_k, faces.remote_halo_k),
    )[radial_axis]
    minus_local_axis, plus_local_axis, remote_halo_axis = owner_axis
    local_size = int(domain.layout.owned_shape[radial_axis])
    global_size = int(domain.shard_spec.global_shape[radial_axis])
    if domain.shard_spec.shard_counts[radial_axis] == 1:
        # A named mesh axis can still be a singleton axis outside shard_map;
        # use the static shard metadata in that case.  runtime_shard_id is
        # only meaningful for a genuinely decomposed axis.
        global_start = jnp.asarray(
            domain.shard_spec.owned_start[radial_axis],
            dtype=jnp.int32,
        )
    else:
        runtime_shard = jnp.asarray(
            domain.runtime_shard_id(radial_axis),
            dtype=jnp.int32,
        )
        global_start = runtime_shard * local_size
    minus_global_axis = global_start + minus_local_axis
    local_plus_global_axis = global_start + plus_local_axis
    remote_plus_global_axis = (
        global_start
        + remote_halo_axis
        - int(domain.layout.halo_width)
    )
    plus_global_axis = jnp.where(
        faces.has_remote_owner,
        remote_plus_global_axis,
        local_plus_global_axis,
    )
    if domain.periodic_axes[radial_axis]:
        minus_radial_interior = jnp.ones_like(faces.active, dtype=bool)
        plus_radial_interior = jnp.ones_like(faces.active, dtype=bool)
    else:
        minus_radial_interior = (
            (minus_global_axis > 1)
            & (minus_global_axis < global_size - 2)
        )
        plus_radial_interior = (
            (plus_global_axis > 1)
            & (plus_global_axis < global_size - 2)
        )

    has_two_owners = faces.has_plus_owner | faces.has_remote_owner
    use_two_owner = (
        faces.active
        & has_two_owners
        & minus_radial_interior
        & plus_radial_interior
    )
    use_cut_wall_owner = (
        faces.active
        & (faces.kind == CV_FACE_CUT_WALL)
        & minus_radial_interior
    )
    selected_valid = jnp.where(
        use_two_owner,
        minus_row_valid & plus_row_valid,
        jnp.where(use_cut_wall_owner, minus_row_valid, True),
    )
    projected_flux = jnp.where(
        use_two_owner,
        0.5 * (minus_flux + plus_flux),
        jnp.where(
            use_cut_wall_owner,
            minus_flux,
            field_closure.projected_flux,
        ),
    )
    return dataclass_replace(
        field_closure,
        projected_flux=projected_flux,
        valid=field_closure.valid & selected_valid,
    )


def _require_local_control_volume_field_closure(
    field_closure: LocalControlVolumeFieldClosure3D | None,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
) -> LocalControlVolumeFieldClosure3D:
    """Validate the direct field closure against the compiled face rows."""

    if field_closure is None:
        raise ValueError("field_closure is required with control_volume_geometry")
    if not isinstance(field_closure, LocalControlVolumeFieldClosure3D):
        raise TypeError(
            "field_closure must be LocalControlVolumeFieldClosure3D with "
            "control_volume_geometry"
        )
    faces = control_volume_geometry.irregular_faces
    rows = control_volume_geometry.face_functionals
    if not isinstance(rows, LocalMomentFittedFaceRows3D):
        raise ValueError("control-volume geometry requires direct face functionals")
    if rows.max_rows != faces.max_rows:
        raise ValueError("face functionals must align with irregular face rows")
    if field_closure.max_rows != faces.max_rows:
        raise ValueError("field_closure.max_rows must align with irregular face rows")
    if field_closure.max_patches != faces.max_patches:
        raise ValueError(
            "field_closure.max_patches must align with irregular face-row geometry"
        )
    try:
        active_aligned = bool(
            jnp.all(field_closure.active == (rows.active & faces.active))
        )
        valid_aligned = bool(jnp.all(field_closure.valid == field_closure.active))
        trace_aligned = bool(
            jnp.all(
                (~jnp.asarray(faces.quadrature_active, dtype=bool))
                | jnp.asarray(field_closure.face_value_valid, dtype=bool)
            )
            & jnp.all(
                (~jnp.asarray(faces.quadrature_active, dtype=bool))
                | jnp.asarray(field_closure.face_gradient_valid, dtype=bool)
            )
        )
    except jax.errors.TracerBoolConversionError:
        active_aligned = True
        valid_aligned = True
        trace_aligned = True
    if not active_aligned:
        raise ValueError("field_closure.active must align with compiled face rows")
    if not valid_aligned:
        raise ValueError("every active direct face-functional row must be valid")
    if not trace_aligned:
        raise ValueError("direct face-value and face-gradient traces must be valid on active quadrature")
    return field_closure


def _local_control_volume_integrated_divergence(
    regular_flux: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    irregular_flux: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    *,
    volume_floor: float,
) -> jnp.ndarray:
    """Divergence of dense and compact integrated face fluxes."""

    regular_faces = control_volume_geometry.regular_faces
    spacing = (
        jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64),
        jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64),
        jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64),
    )
    logical_cell_measure = (
        spacing[1] * spacing[2],
        spacing[0] * spacing[2],
        spacing[0] * spacing[1],
    )
    face_area = (
        regular_faces.x_area,
        regular_faces.y_area,
        regular_faces.z_area,
    )
    face_fraction = (
        regular_faces.x_area_fraction,
        regular_faces.y_area_fraction,
        regular_faces.z_area_fraction,
    )
    face_open = (
        regular_faces.x_open_mask,
        regular_faces.y_open_mask,
        regular_faces.z_open_mask,
    )
    integrated_sum = jnp.zeros(geometry.owned_shape, dtype=jnp.float64)
    for axis in range(3):
        logical_area = _lift_cell_field_to_faces(
            logical_cell_measure[axis],
            axis=axis,
            periodic=False,
        )
        open_measure = (
            logical_area
            * jnp.asarray(face_area[axis], dtype=jnp.float64)
            * jnp.asarray(face_fraction[axis], dtype=jnp.float64)
        )
        integrated_face = jnp.where(
            jnp.asarray(face_open[axis], dtype=bool)
            & (open_measure > 0.0),
            jnp.asarray(regular_flux[axis], dtype=jnp.float64)
            * open_measure,
            0.0,
        )
        integrated_sum = integrated_sum + (
            integrated_face[_axis_slice_nd(axis, 1, None, 3)]
            - integrated_face[_axis_slice_nd(axis, None, -1, 3)]
        )

    cells = control_volume_geometry.cells
    local_source = (cells.raw_volume > 0.0) & ~cells.owner_is_remote
    owner_sum = jnp.zeros(geometry.owned_shape, dtype=jnp.float64).at[
        cells.owner_i,
        cells.owner_j,
        cells.owner_k,
    ].add(
        jnp.where(local_source, integrated_sum, 0.0)
    )
    remote_halo = jnp.zeros(cells.layout.cell_halo_shape, dtype=jnp.float64).at[
        cells.remote_owner_halo_i, cells.remote_owner_halo_j, cells.remote_owner_halo_k
    ].add(jnp.where((cells.raw_volume > 0.0) & cells.owner_is_remote, integrated_sum, 0.0))

    faces = control_volume_geometry.irregular_faces
    if int(faces.max_rows) > 0:
        row_flux = jnp.where(faces.active, irregular_flux, 0.0)
        owner_sum = owner_sum.at[
            faces.minus_owner_i,
            faces.minus_owner_j,
            faces.minus_owner_k,
        ].add(row_flux)
        owner_sum = owner_sum.at[
            faces.plus_owner_i,
            faces.plus_owner_j,
            faces.plus_owner_k,
        ].add(jnp.where(faces.has_plus_owner, -row_flux, 0.0))
        remote_halo = remote_halo.at[
            faces.remote_residual_halo_i, faces.remote_residual_halo_j, faces.remote_residual_halo_k
        ].add(jnp.where(faces.has_remote_residual, -row_flux, 0.0))

    owner_sum = owner_sum + accumulate_halo_contributions_to_owned(remote_halo, domain)

    result = owner_sum / jnp.maximum(
        cells.aggregate_volume,
        float(volume_floor),
    )
    return jnp.where(cells.is_active_owner, result, 0.0)


def _patch_cut_wall_local_face_gradients(
    x_face_grad: jnp.ndarray,
    y_face_grad: jnp.ndarray,
    z_face_grad: jnp.ndarray,
    *,
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    regular_face_geometry: LocalRegularFaceGeometry3D,
    cut_wall_geometry: LocalCutWallGeometry3D,
    cut_wall_bc: LocalCutWallBC3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Replace cut-wall-neighborhood face gradients with cut-cell-aware fits.

    The default face-gradient reconstruction differentiates a lifted face field.
    Near embedded-wall edges that lifted field can sample diagonal inactive cells.
    The cut-wall patch uses active-cell samples around the open face centroid
    where possible and falls back to averaged patched-cell gradients otherwise.
    """

    if int(cut_wall_geometry.max_wall_faces) == 0:
        return x_face_grad, y_face_grad, z_face_grad

    owner_mask = _cut_wall_owner_cell_mask(
        geometry,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
    )
    dfdx_cell = _take_stencil_finite_difference(local.x)
    dfdy_cell = _take_stencil_finite_difference(local.y)
    dfdz_cell = _take_stencil_finite_difference(local.z)
    cell_grad = jnp.nan_to_num(
        jnp.stack((dfdx_cell, dfdy_cell, dfdz_cell), axis=-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    face_grads = (x_face_grad, y_face_grad, z_face_grad)
    open_masks = (
        regular_face_geometry.x_open_mask,
        regular_face_geometry.y_open_mask,
        regular_face_geometry.z_open_mask,
    )
    patched: list[jnp.ndarray] = []
    for face_axis, face_grad in enumerate(face_grads):
        dilated_owner_mask = _dilate_cut_wall_owner_mask_for_face_axis(
            owner_mask,
            face_axis=face_axis,
            periodic_axes=domain.periodic_axes,
        )
        face_mask = _cell_mask_to_adjacent_face_mask(
            dilated_owner_mask,
            face_axis=face_axis,
        )
        averaged, _count = _average_cell_gradients_to_faces(
            cell_grad,
            face_axis=face_axis,
        )
        fitted, fit_valid = _least_squares_cut_wall_face_gradient(
            local.x.center,
            geometry=geometry,
            domain=domain,
            regular_face_geometry=regular_face_geometry,
            face_axis=face_axis,
        )
        open_mask = jnp.asarray(open_masks[face_axis], dtype=bool)
        replace_mask = face_mask & open_mask
        closed_cut_wall_mask = face_mask & (~open_mask)
        raw_face_grad = jnp.asarray(face_grad, dtype=jnp.float64)
        averaged = jnp.nan_to_num(averaged, nan=0.0, posinf=0.0, neginf=0.0)
        fitted = jnp.nan_to_num(fitted, nan=0.0, posinf=0.0, neginf=0.0)
        replacement = jnp.where(fit_valid[..., None], fitted, averaged)
        current = jnp.where(replace_mask[..., None], replacement, raw_face_grad)
        current = jnp.where(closed_cut_wall_mask[..., None], 0.0, current)
        patched.append(current)
    return patched[0], patched[1], patched[2]


def build_local_projected_laplacian_flux_stencil(
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    face_bc: LocalBoundaryFaceBC3D | None = None,
    regular_face_geometry: LocalRegularFaceGeometry3D | None = None,
    regular_face_contribution_rows: LocalRegularFaceContributionRows3D | None = None,
    cell_volume: LocalCellVolumeGeometry3D | None = None,
    cut_wall_geometry: LocalCutWallGeometry3D | None = None,
    cut_wall_bc: LocalCutWallBC3D | None = None,
    cell_gradient: LocalCellGradient3D | None = None,
    aggregate_geometry: LocalAggregateCellGeometry3D | None = None,
    regular_boundary_closure: (
        LocalRegularBoundaryMomentClosure3D | None
    ) = None,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    neumann_normal_scheme: str = "logical",
    b_floor: float = 1.0e-30,
) -> LocalControlVolumeFluxStencil3D:
    """Build the local face-flux stencil for a projected Laplacian."""

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "build_local_projected_laplacian_flux_stencil requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    if not isinstance(domain, LocalDomain3D):
        raise TypeError(
            "build_local_projected_laplacian_flux_stencil requires LocalDomain3D, "
            f"got {type(domain).__name__}"
        )
    if geometry.layout != domain.layout:
        raise ValueError("geometry and domain must share the same HaloLayout3D")
    if local.shape != geometry.owned_shape:
        raise ValueError(
            f"local stencil must have shape {geometry.owned_shape}, got {local.shape}"
        )
    if not hasattr(local, "face_grad"):
        raise TypeError("local must provide face_grad")

    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    if len(axis_regular_axes) != 3:
        raise ValueError("axis_regular_axes must have length 3")
    if axis_regular_axes[1] or axis_regular_axes[2]:
        raise NotImplementedError(
            "axis_regular_axes currently only supports the lower x axis; "
            f"got axis_regular_axes={axis_regular_axes}"
        )

    regular_face_geometry = regular_face_geometry or geometry.regular_face_geometry
    cell_volume = cell_volume or geometry.cell_volume_geometry
    face_bc = face_bc or LocalBoundaryFaceBC3D.empty(geometry.layout)
    if cut_wall_geometry is None and cut_wall_bc is None:
        cut_wall_geometry = LocalCutWallGeometry3D.empty(0)
        cut_wall_bc = LocalCutWallBC3D.empty(0)
    elif cut_wall_geometry is None:
        cut_wall_geometry = LocalCutWallGeometry3D.empty(cut_wall_bc.n_wall_faces)
    elif cut_wall_bc is None:
        cut_wall_bc = LocalCutWallBC3D.empty(cut_wall_geometry.max_wall_faces)

    if face_projectors is None:
        face_projectors = build_local_perp_laplacian_face_projectors(
            geometry,
            domain,
            b_floor=b_floor,
            axis_regular_axes=axis_regular_axes,
        )
    x_face_projector, y_face_projector, z_face_projector = face_projectors

    x_face_grad = jnp.asarray(local.face_grad.x, dtype=jnp.float64)
    y_face_grad = jnp.asarray(local.face_grad.y, dtype=jnp.float64)
    z_face_grad = jnp.asarray(local.face_grad.z, dtype=jnp.float64)

    x_face_grad, y_face_grad, z_face_grad = _patch_cut_wall_local_face_gradients(
        x_face_grad,
        y_face_grad,
        z_face_grad,
        local=local,
        geometry=geometry,
        domain=domain,
        regular_face_geometry=regular_face_geometry,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
    )

    values_owned = jnp.asarray(local.x.center, dtype=jnp.float64)
    x_face_grad = _patch_local_axis_face_gradients(
        x_face_grad,
        values_owned=values_owned,
        geometry=geometry,
        domain=domain,
        axis=0,
        axis_kind=face_bc.kind_x,
        axis_value=face_bc.value_x,
        axis_mask=face_bc.mask_x,
        axis_regular_axes=axis_regular_axes,
        neumann_normal_scheme=neumann_normal_scheme,
        regular_boundary_closure=regular_boundary_closure,
    )
    y_face_grad = _patch_local_axis_face_gradients(
        y_face_grad,
        values_owned=values_owned,
        geometry=geometry,
        domain=domain,
        axis=1,
        axis_kind=face_bc.kind_y,
        axis_value=face_bc.value_y,
        axis_mask=face_bc.mask_y,
        axis_regular_axes=axis_regular_axes,
        neumann_normal_scheme=neumann_normal_scheme,
        regular_boundary_closure=regular_boundary_closure,
    )
    z_face_grad = _patch_local_axis_face_gradients(
        z_face_grad,
        values_owned=values_owned,
        geometry=geometry,
        domain=domain,
        axis=2,
        axis_kind=face_bc.kind_z,
        axis_value=face_bc.value_z,
        axis_mask=face_bc.mask_z,
        axis_regular_axes=axis_regular_axes,
        neumann_normal_scheme=neumann_normal_scheme,
        regular_boundary_closure=regular_boundary_closure,
    )

    x_face_metric = geometry.face_metric.x
    y_face_metric = geometry.face_metric.y
    z_face_metric = geometry.face_metric.z

    x_flux = jnp.asarray(x_face_metric.J_owned, dtype=jnp.float64) * jnp.einsum(
        "...j,...j->...", x_face_projector[..., 0, :], x_face_grad
    )
    if axis_regular_axes[0]:
        do_axis_lower = domain.runtime_has_axis_regular_lower(0)
        lower = jnp.where(do_axis_lower, jnp.zeros_like(x_flux[0]), x_flux[0])
        x_flux = x_flux.at[0].set(lower)
    y_flux = jnp.asarray(y_face_metric.J_owned, dtype=jnp.float64) * jnp.einsum(
        "...j,...j->...", y_face_projector[..., 1, :], y_face_grad
    )
    z_flux = jnp.asarray(z_face_metric.J_owned, dtype=jnp.float64) * jnp.einsum(
        "...j,...j->...", z_face_projector[..., 2, :], z_face_grad
    )

    x_flux = _apply_local_face_flux_bc(
        x_flux,
        axis=0,
        axis_kind=face_bc.kind_x,
        axis_value=face_bc.value_x,
        axis_mask=face_bc.mask_x,
        axis_regular_axes=axis_regular_axes,
    )
    y_flux = _apply_local_face_flux_bc(
        y_flux,
        axis=1,
        axis_kind=face_bc.kind_y,
        axis_value=face_bc.value_y,
        axis_mask=face_bc.mask_y,
        axis_regular_axes=axis_regular_axes,
    )
    z_flux = _apply_local_face_flux_bc(
        z_flux,
        axis=2,
        axis_kind=face_bc.kind_z,
        axis_value=face_bc.value_z,
        axis_mask=face_bc.mask_z,
        axis_regular_axes=axis_regular_axes,
    )

    cut_wall_flux = _build_local_cut_wall_flux_payload(
        local=local,
        geometry=geometry,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        cell_gradient=cell_gradient,
        b_floor=b_floor,
    )
    regular_face_contribution_flux = None
    if regular_face_contribution_rows is not None:
        regular_face_contribution_flux = _build_regular_face_contribution_projected_flux(
            values_owned,
            geometry,
            face_projectors,
            regular_face_contribution_rows,
            aggregate_geometry=aggregate_geometry,
            cell_gradient=cell_gradient,
        )

    return LocalControlVolumeFluxStencil3D(
        regular_flux=FaceFluxStencil3D(x=x_flux, y=y_flux, z=z_flux),
        regular_face_geometry=regular_face_geometry,
        cell_volume=cell_volume,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_flux=cut_wall_flux,
        regular_face_contribution_rows=regular_face_contribution_rows,
        regular_face_contribution_flux=regular_face_contribution_flux,
    )


def local_divergence_conservative_op(
    cv_flux: LocalControlVolumeFluxStencil3D,
    geometry: LocalFciGeometry3D,
    *,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the local conservative divergence from a completed face-flux stencil."""

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "local_divergence_conservative_op requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    if cv_flux.shape != geometry.owned_shape:
        raise ValueError(
            f"cv_flux must have shape {geometry.owned_shape}, got {cv_flux.shape}"
        )

    def _divergence_from_face_flux(
        flux: jnp.ndarray,
        spacing: jnp.ndarray,
        *,
        axis: int,
        area: jnp.ndarray,
    ) -> jnp.ndarray:
        face_flux = jnp.asarray(flux, dtype=jnp.float64) * jnp.asarray(area, dtype=jnp.float64)
        h = jnp.asarray(spacing, dtype=jnp.float64)
        if h.shape != face_flux[_axis_slice_nd(axis, 1, None, face_flux.ndim)].shape:
            raise ValueError(
                f"local spacing for axis {axis} must match the owned cell shape; got {h.shape}"
            )
        return (
            face_flux[_axis_slice_nd(axis, 1, None, face_flux.ndim)]
            - face_flux[_axis_slice_nd(axis, None, -1, face_flux.ndim)]
        ) / jnp.maximum(h, 1.0e-30)

    div_flux = (
        _divergence_from_face_flux(
            cv_flux.regular_flux.x,
            geometry.spacing.dx_owned,
            axis=0,
            area=
                cv_flux.regular_face_geometry.x_area
                * cv_flux.regular_face_geometry.x_area_fraction
                * cv_flux.regular_face_geometry.x_open_mask,
        )
        + _divergence_from_face_flux(
            cv_flux.regular_flux.y,
            geometry.spacing.dy_owned,
            axis=1,
            area=
                cv_flux.regular_face_geometry.y_area
                * cv_flux.regular_face_geometry.y_area_fraction
                * cv_flux.regular_face_geometry.y_open_mask,
        )
        + _divergence_from_face_flux(
            cv_flux.regular_flux.z,
            geometry.spacing.dz_owned,
            axis=2,
            area=
                cv_flux.regular_face_geometry.z_area
                * cv_flux.regular_face_geometry.z_area_fraction
                * cv_flux.regular_face_geometry.z_open_mask,
        )
    )

    regular_face_rows = cv_flux.regular_face_contribution_rows
    if regular_face_rows is not None and int(regular_face_rows.max_rows) > 0:
        active = jnp.asarray(regular_face_rows.active, dtype=bool)

        row_face_value = _regular_face_row_legacy_flux(cv_flux.regular_flux, regular_face_rows)
        if cv_flux.regular_face_contribution_flux is not None:
            reconstructed = jnp.asarray(
                cv_flux.regular_face_contribution_flux,
                dtype=jnp.float64,
            )
            use_reconstructed = jnp.asarray(
                regular_face_rows.use_reconstructed_flux,
                dtype=bool,
            ) & jnp.isfinite(reconstructed)
            row_face_value = jnp.where(use_reconstructed, reconstructed, row_face_value)
        regular_face_contrib = jnp.zeros(geometry.owned_shape, dtype=jnp.float64)
        regular_face_contrib = regular_face_contrib.at[
            jnp.asarray(regular_face_rows.owner_i, dtype=jnp.int32),
            jnp.asarray(regular_face_rows.owner_j, dtype=jnp.int32),
            jnp.asarray(regular_face_rows.owner_k, dtype=jnp.int32),
        ].add(
            jnp.where(
                active,
                row_face_value
                * jnp.asarray(regular_face_rows.area, dtype=jnp.float64)
                * jnp.asarray(regular_face_rows.sign, dtype=jnp.float64),
                0.0,
            )
        )
        div_flux = div_flux + regular_face_contrib

    if cv_flux.cut_wall_geometry is not None and cv_flux.cut_wall_flux is not None and cv_flux.cut_wall_flux.size:
        cut_wall_contrib = jnp.zeros(geometry.owned_shape, dtype=jnp.float64)
        cut_wall_contrib = cut_wall_contrib.at[
            jnp.asarray(cv_flux.cut_wall_geometry.owner_i, dtype=jnp.int32),
            jnp.asarray(cv_flux.cut_wall_geometry.owner_j, dtype=jnp.int32),
            jnp.asarray(cv_flux.cut_wall_geometry.owner_k, dtype=jnp.int32),
        ].add(jnp.asarray(cv_flux.cut_wall_flux, dtype=jnp.float64))
        div_flux = div_flux + cut_wall_contrib

    effective_volume = jnp.asarray(cv_flux.cell_volume.volume, dtype=jnp.float64) * jnp.asarray(
        cv_flux.cell_volume.volume_fraction, dtype=jnp.float64
    )
    result = div_flux / jnp.maximum(effective_volume, float(jacobian_floor))
    return _mask_inactive_owned(result, geometry)


def build_local_perp_laplacian_stencil(
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    face_bc: LocalBoundaryFaceBC3D | None = None,
    regular_face_geometry: LocalRegularFaceGeometry3D | None = None,
    regular_face_contribution_rows: LocalRegularFaceContributionRows3D | None = None,
    cell_volume: LocalCellVolumeGeometry3D | None = None,
    cut_wall_geometry: LocalCutWallGeometry3D | None = None,
    cut_wall_bc: LocalCutWallBC3D | None = None,
    cell_gradient: LocalCellGradient3D | None = None,
    aggregate_geometry: LocalAggregateCellGeometry3D | None = None,
    regular_boundary_closure: (
        LocalRegularBoundaryMomentClosure3D | None
    ) = None,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    neumann_normal_scheme: str = "logical",
    b_floor: float = 1.0e-30,
) -> LocalControlVolumeFluxStencil3D:
    """Build the local conservative flux stencil for ``-∇·(P⊥∇f)``."""

    if face_projectors is None:
        face_projectors = build_local_perp_laplacian_face_projectors(
            geometry,
            domain,
            b_floor=b_floor,
            axis_regular_axes=axis_regular_axes,
        )
    return build_local_projected_laplacian_flux_stencil(
        local,
        geometry,
        domain,
        face_projectors=face_projectors,
        face_bc=face_bc,
        regular_face_geometry=regular_face_geometry,
        regular_face_contribution_rows=regular_face_contribution_rows,
        cell_volume=cell_volume,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        cell_gradient=cell_gradient,
        aggregate_geometry=aggregate_geometry,
        regular_boundary_closure=regular_boundary_closure,
        axis_regular_axes=axis_regular_axes,
        neumann_normal_scheme=neumann_normal_scheme,
        b_floor=b_floor,
    )


def local_perp_laplacian_conservative_op(
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    face_bc: LocalBoundaryFaceBC3D | None = None,
    regular_face_geometry: LocalRegularFaceGeometry3D | None = None,
    regular_face_contribution_rows: LocalRegularFaceContributionRows3D | None = None,
    cell_volume: LocalCellVolumeGeometry3D | None = None,
    cut_wall_geometry: LocalCutWallGeometry3D | None = None,
    cut_wall_bc: LocalCutWallBC3D | None = None,
    cell_gradient: LocalCellGradient3D | None = None,
    aggregate_geometry: LocalAggregateCellGeometry3D | None = None,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
    field_closure: LocalControlVolumeFieldClosure3D | None = None,
    control_volume_polynomial: LocalControlVolumePolynomial3D | None = None,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    neumann_normal_scheme: str = "logical",
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the domain-decomposed conservative perpendicular Laplacian."""

    effective_face_projectors = face_projectors
    if effective_face_projectors is None:
        effective_face_projectors = build_local_perp_laplacian_face_projectors(
            geometry,
            domain,
            b_floor=b_floor,
            axis_regular_axes=axis_regular_axes,
        )
    effective_regular_faces = (
        control_volume_geometry.regular_faces
        if control_volume_geometry is not None
        else regular_face_geometry
    )
    cv_flux = build_local_perp_laplacian_stencil(
        local,
        geometry,
        domain,
        face_projectors=effective_face_projectors,
        face_bc=face_bc,
        regular_face_geometry=effective_regular_faces,
        regular_face_contribution_rows=(
            None
            if control_volume_geometry is not None
            else regular_face_contribution_rows
        ),
        cell_volume=cell_volume,
        cut_wall_geometry=(
            None if control_volume_geometry is not None else cut_wall_geometry
        ),
        cut_wall_bc=None if control_volume_geometry is not None else cut_wall_bc,
        cell_gradient=(
            None if control_volume_geometry is not None else cell_gradient
        ),
        aggregate_geometry=(
            None if control_volume_geometry is not None else aggregate_geometry
        ),
        regular_boundary_closure=(
            control_volume_geometry.regular_boundary_closure
            if control_volume_geometry is not None
            else None
        ),
        axis_regular_axes=axis_regular_axes,
        neumann_normal_scheme=neumann_normal_scheme,
        b_floor=b_floor,
    )
    if control_volume_geometry is not None:
        if not isinstance(
            control_volume_geometry,
            LocalEmbeddedControlVolumeGeometry3D,
        ):
            raise TypeError(
                "control_volume_geometry must be "
                "LocalEmbeddedControlVolumeGeometry3D or None"
            )
        regular_flux = (
            cv_flux.regular_flux.x,
            cv_flux.regular_flux.y,
            cv_flux.regular_flux.z,
        )
        field_closure = _require_local_control_volume_field_closure(
            field_closure,
            control_volume_geometry,
        )
        return _local_control_volume_integrated_divergence(
            regular_flux,
            field_closure.projected_flux,
            geometry,
            domain,
            control_volume_geometry,
            volume_floor=jacobian_floor,
        )
    return local_divergence_conservative_op(cv_flux, geometry, jacobian_floor=jacobian_floor)


def local_parallel_laplacian_conservative_op(
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    face_bc: LocalBoundaryFaceBC3D | None = None,
    regular_face_geometry: LocalRegularFaceGeometry3D | None = None,
    regular_face_contribution_rows: LocalRegularFaceContributionRows3D | None = None,
    cell_volume: LocalCellVolumeGeometry3D | None = None,
    cut_wall_geometry: LocalCutWallGeometry3D | None = None,
    cut_wall_bc: LocalCutWallBC3D | None = None,
    cell_gradient: LocalCellGradient3D | None = None,
    aggregate_geometry: LocalAggregateCellGeometry3D | None = None,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
    field_closure: LocalControlVolumeFieldClosure3D | None = None,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    neumann_normal_scheme: str = "logical",
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the domain-decomposed conservative parallel Laplacian."""

    if face_projectors is None:
        face_projectors = build_local_parallel_laplacian_face_projectors(
            geometry,
            domain,
            b_floor=b_floor,
            axis_regular_axes=axis_regular_axes,
        )
    effective_regular_faces = (
        control_volume_geometry.regular_faces
        if control_volume_geometry is not None
        else regular_face_geometry
    )
    cv_flux = build_local_projected_laplacian_flux_stencil(
        local,
        geometry,
        domain,
        face_projectors=face_projectors,
        face_bc=face_bc,
        regular_face_geometry=effective_regular_faces,
        regular_face_contribution_rows=(
            None
            if control_volume_geometry is not None
            else regular_face_contribution_rows
        ),
        cell_volume=cell_volume,
        cut_wall_geometry=(
            None if control_volume_geometry is not None else cut_wall_geometry
        ),
        cut_wall_bc=None if control_volume_geometry is not None else cut_wall_bc,
        cell_gradient=(
            None if control_volume_geometry is not None else cell_gradient
        ),
        aggregate_geometry=(
            None if control_volume_geometry is not None else aggregate_geometry
        ),
        regular_boundary_closure=(
            control_volume_geometry.regular_boundary_closure
            if control_volume_geometry is not None
            else None
        ),
        axis_regular_axes=axis_regular_axes,
        neumann_normal_scheme=neumann_normal_scheme,
        b_floor=b_floor,
    )
    if control_volume_geometry is not None:
        field_closure = _require_local_control_volume_field_closure(
            field_closure,
            control_volume_geometry,
        )
        return _local_control_volume_integrated_divergence(
            (
                cv_flux.regular_flux.x,
                cv_flux.regular_flux.y,
                cv_flux.regular_flux.z,
            ),
            field_closure.parallel_gradient_flux,
            geometry,
            domain,
            control_volume_geometry,
            volume_floor=jacobian_floor,
        )
    return local_divergence_conservative_op(cv_flux, geometry, jacobian_floor=jacobian_floor)


def _axis_index_nd(axis: int, index: int, ndim: int) -> tuple[object, ...]:
    slices: list[object] = [slice(None)] * ndim
    slices[axis] = index
    return tuple(slices)


def _axis_slice_nd(axis: int, start: int | None, stop: int | None, ndim: int) -> tuple[object, ...]:
    slices: list[object] = [slice(None)] * ndim
    slices[axis] = slice(start, stop)
    return tuple(slices)


def _lift_cell_field_to_faces(field: jnp.ndarray, *, axis: int, periodic: bool) -> jnp.ndarray:
    """Map a cell-centered field onto the corresponding face grid along one axis."""

    values_3d = jnp.asarray(field, dtype=jnp.float64)
    axis_n = values_3d.shape[axis]
    face_shape = list(values_3d.shape)
    face_shape[axis] += 1

    if axis_n == 1:
        return jnp.broadcast_to(values_3d, tuple(face_shape))

    first = jnp.take(values_3d, 0, axis=axis)
    second = jnp.take(values_3d, 1, axis=axis)
    last = jnp.take(values_3d, -1, axis=axis)
    penultimate = jnp.take(values_3d, -2, axis=axis)

    if periodic:
        lower_ghost = last
        upper_ghost = first
    else:
        # Second-order ghost-cell extrapolation:
        #   q_{-1}  = 2 q_0 - q_1
        #   q_{n}   = 2 q_{n-1} - q_{n-2}
        # This lets the same face-average reconstruction be used at the
        # boundary without dropping to first order.
        lower_ghost = 2.0 * first - second
        upper_ghost = 2.0 * last - penultimate

    ext = jnp.concatenate(
        (
            jnp.expand_dims(lower_ghost, axis=axis),
            values_3d,
            jnp.expand_dims(upper_ghost, axis=axis),
        ),
        axis=axis,
    )
    return 0.5 * (
        jnp.take(ext, jnp.arange(axis_n + 1), axis=axis)
        + jnp.take(ext, jnp.arange(1, axis_n + 2), axis=axis)
    )


def _homogeneous_local_face_bc(
    face_bc: LocalBoundaryFaceBC3D,
) -> LocalBoundaryFaceBC3D:
    """Keep local face BC kinds and masks while removing affine data."""

    return dataclass_replace(
        face_bc,
        value_x=jnp.zeros_like(face_bc.value_x, dtype=jnp.float64),
        value_y=jnp.zeros_like(face_bc.value_y, dtype=jnp.float64),
        value_z=jnp.zeros_like(face_bc.value_z, dtype=jnp.float64),
    )


def _dirichlet_lift_correction_local_face_bc(
    face_bc: LocalBoundaryFaceBC3D,
) -> LocalBoundaryFaceBC3D:
    """Return local correction BCs for ``phi = phi_lift + u``."""

    return dataclass_replace(
        face_bc,
        value_x=jnp.where(face_bc.kind_x == BC_DIRICHLET, 0.0, face_bc.value_x),
        value_y=jnp.where(face_bc.kind_y == BC_DIRICHLET, 0.0, face_bc.value_y),
        value_z=jnp.where(face_bc.kind_z == BC_DIRICHLET, 0.0, face_bc.value_z),
    )


def _homogeneous_local_control_volume_boundary_bc(
    boundary_bc: LocalControlVolumeBoundaryBC3D,
) -> LocalControlVolumeBoundaryBC3D:
    """Keep compact boundary kinds while removing affine field data."""

    return LocalControlVolumeBoundaryBC3D(
        kind=boundary_bc.kind,
        centroid_value=jnp.zeros_like(
            boundary_bc.centroid_value,
            dtype=jnp.float64,
        ),
        quadrature_value=jnp.zeros_like(
            boundary_bc.quadrature_value,
            dtype=jnp.float64,
        ),
        active=boundary_bc.active,
        max_rows=boundary_bc.max_rows,
        max_patches=boundary_bc.max_patches,
    )


def _dirichlet_lift_correction_local_control_volume_boundary_bc(
    boundary_bc: LocalControlVolumeBoundaryBC3D,
) -> LocalControlVolumeBoundaryBC3D:
    """Return compact correction BCs for ``phi = phi_lift + u``."""

    is_dirichlet = boundary_bc.kind == BC_DIRICHLET
    return LocalControlVolumeBoundaryBC3D(
        kind=boundary_bc.kind,
        centroid_value=jnp.where(
            is_dirichlet,
            0.0,
            boundary_bc.centroid_value,
        ),
        quadrature_value=jnp.where(
            is_dirichlet[:, None, None],
            0.0,
            boundary_bc.quadrature_value,
        ),
        active=boundary_bc.active,
        max_rows=boundary_bc.max_rows,
        max_patches=boundary_bc.max_patches,
    )


def _principal_perp_laplacian_bands(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    face_bc: LocalBoundaryFaceBC3D,
    *,
    regularization_epsilon: float = 0.0,
    regular_face_geometry: LocalRegularFaceGeometry3D | None = None,
    effective_volume: jnp.ndarray | None = None,
) -> tuple[
    jnp.ndarray,
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Approximate ``-div(P_perp grad)`` by its axis-normal bands.

    Mixed metric couplings are deliberately omitted.  The resulting diagonal
    and nearest-neighbor bands are inexpensive geometry-aware inputs for
    SOLVAX point and line preconditioners; the full matrix-free operator
    remains unchanged.
    """

    use_explicit_geometry = (
        regular_face_geometry is not None or effective_volume is not None
    )
    locations = ("x_face", "y_face", "z_face")
    grids = (geometry.grid.x, geometry.grid.y, geometry.grid.z)
    spacings = (
        geometry.spacing.dx_owned,
        geometry.spacing.dy_owned,
        geometry.spacing.dz_owned,
    )
    face_metrics = (
        geometry.face_metric.x,
        geometry.face_metric.y,
        geometry.face_metric.z,
    )
    regular_faces = (
        geometry.regular_face_geometry
        if regular_face_geometry is None
        else regular_face_geometry
    )
    face_areas = (
        regular_faces.x_area,
        regular_faces.y_area,
        regular_faces.z_area,
    )
    face_area_fractions = (
        regular_faces.x_area_fraction,
        regular_faces.y_area_fraction,
        regular_faces.z_area_fraction,
    )
    face_open_masks = (
        regular_faces.x_open_mask,
        regular_faces.y_open_mask,
        regular_faces.z_open_mask,
    )
    bc_kinds = (face_bc.kind_x, face_bc.kind_y, face_bc.kind_z)
    bc_masks = (face_bc.mask_x, face_bc.mask_y, face_bc.mask_z)
    if effective_volume is None:
        effective_volume = (
            jnp.asarray(geometry.cell_volume_geometry.volume, dtype=jnp.float64)
            * jnp.asarray(
                geometry.cell_volume_geometry.volume_fraction,
                dtype=jnp.float64,
            )
        )
    else:
        effective_volume = jnp.asarray(effective_volume, dtype=jnp.float64)
        if effective_volume.shape != geometry.owned_shape:
            raise ValueError("effective_volume must match geometry.owned_shape")

    diagonal = jnp.zeros(geometry.owned_shape, dtype=jnp.float64)
    lower_bands: list[jnp.ndarray] = []
    upper_bands: list[jnp.ndarray] = []
    for axis in range(3):
        face_slices = geometry.layout.location_owned_slices(locations[axis])
        axis_slice = face_slices[axis]
        if axis_slice.start is None or axis_slice.stop is None:
            raise ValueError("owned face slices must have finite bounds")
        centers_halo = jnp.asarray(grids[axis].centers_halo, dtype=jnp.float64)
        center_distance_1d = (
            centers_halo[axis_slice.start:axis_slice.stop]
            - centers_halo[axis_slice.start - 1:axis_slice.stop - 1]
        )
        distance_shape = [1, 1, 1]
        distance_shape[axis] = int(center_distance_1d.shape[0])
        center_distance = jnp.reshape(center_distance_1d, distance_shape)

        projector = jnp.asarray(face_projectors[axis], dtype=jnp.float64)
        face_coefficient = (
            jnp.asarray(face_metrics[axis].J_owned, dtype=jnp.float64)
            * jnp.asarray(projector[..., axis, axis], dtype=jnp.float64)
            * jnp.asarray(face_areas[axis], dtype=jnp.float64)
            * jnp.asarray(face_area_fractions[axis], dtype=jnp.float64)
            * jnp.asarray(face_open_masks[axis], dtype=jnp.float64)
            / jnp.maximum(center_distance, 1.0e-30)
        )

        kind = jnp.asarray(bc_kinds[axis], dtype=jnp.int32)
        mask = jnp.asarray(bc_masks[axis], dtype=bool)
        prescribed_flux = mask & (
            (kind == BC_NEUMANN)
            | (kind == BC_NORMALFLUX)
            | (kind == BC_NOFLUX)
        )
        face_coefficient = jnp.where(
            prescribed_flux,
            0.0,
            face_coefficient,
        )
        # The production Dirichlet closure uses a second-order one-sided
        # normal derivative.  Its owner-cell diagonal coefficient is three
        # times the centered ghost-to-owner estimate used above.
        face_coefficient = jnp.where(
            mask & (kind == BC_DIRICHLET),
            3.0 * face_coefficient,
            face_coefficient,
        )

        lower_face = face_coefficient[
            _axis_slice_nd(axis, None, -1, face_coefficient.ndim)
        ]
        upper_face = face_coefficient[
            _axis_slice_nd(axis, 1, None, face_coefficient.ndim)
        ]
        cell_scale = (
            jnp.asarray(spacings[axis], dtype=jnp.float64)
            * jnp.maximum(effective_volume, 1.0e-30)
        )
        lower = -lower_face / jnp.maximum(cell_scale, 1.0e-30)
        upper = -upper_face / jnp.maximum(cell_scale, 1.0e-30)
        diagonal = diagonal - lower - upper
        lower_bands.append(lower)
        upper_bands.append(upper)

    diagonal = diagonal + jnp.asarray(
        regularization_epsilon,
        dtype=jnp.float64,
    )
    active = jnp.asarray(geometry.active_cell_mask_owned, dtype=bool)
    if not use_explicit_geometry:
        # Preserve the historical regular-storage behavior exactly.
        local_scale = jnp.max(jnp.where(active, jnp.abs(diagonal), 0.0))
        floor = jnp.maximum(local_scale * 1.0e-12, 1.0e-30)
        diagonal = jnp.where(active, jnp.maximum(jnp.abs(diagonal), floor), 1.0)
    else:
        # The pole path must see the assembled stiffness sign and must not
        # repair an invalid principal block by taking an absolute value or
        # inserting a diagonal floor.
        diagonal = jnp.where(active, diagonal, 0.0)
    lower_bands = [jnp.where(active, value, 0.0) for value in lower_bands]
    upper_bands = [jnp.where(active, value, 0.0) for value in upper_bands]
    return diagonal, tuple(lower_bands), tuple(upper_bands)


def _validate_concrete_angular_agglomeration_tree_assembly(
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    diagonal: jnp.ndarray,
    child_edge: jnp.ndarray,
    parent_i: jnp.ndarray,
    parent_j: jnp.ndarray,
    parent_k: jnp.ndarray,
    active: jnp.ndarray,
) -> None:
    """Validate the concrete nested owner tree before entering JAX tracing."""
    profile = control_volume_geometry.angular_group_sizes
    if profile is None:
        raise ValueError("angular-agglomeration tree requires angular_group_sizes")

    # The topology/profile is static, but the lowered owner payload can be a
    # dynamic shard_map argument.  Keep the cheap structural checks available
    # while tracing and defer only host-side NumPy/value/SPD checks.
    diagonal_shape = tuple(int(value) for value in diagonal.shape)
    if len(diagonal_shape) != 3:
        raise ValueError(
            "angular tree diagonal must be three-dimensional, "
            f"got shape={diagonal.shape}"
        )
    nx, ny, _nz = diagonal_shape
    q = tuple(int(value) for value in profile)
    if len(q) != nx or q[0] != ny:
        raise ValueError("angular group profile must have q[0] == ny and one entry per ring")
    for ring, value in enumerate(q):
        if value <= 0 or ny % value:
            raise ValueError("angular group profile must contain positive divisors of ny")
        if ring and (value > q[ring - 1] or q[ring - 1] % value):
            raise ValueError("angular group profile must be concretely nested and non-increasing")

    payload = (diagonal, child_edge, parent_i, parent_j, parent_k, active)
    cv_payload = (
        control_volume_geometry.cells.owner_i,
        control_volume_geometry.cells.owner_j,
        control_volume_geometry.cells.owner_k,
        control_volume_geometry.cells.aggregate_volume,
    )
    if any(isinstance(value, jax.core.Tracer) for value in (*payload, *cv_payload)):
        return
    try:
        diag = np.asarray(diagonal, dtype=np.float64)
        edge = np.asarray(child_edge, dtype=np.float64)
        pi = np.asarray(parent_i, dtype=np.int32)
        pj = np.asarray(parent_j, dtype=np.int32)
        pk = np.asarray(parent_k, dtype=np.int32)
        owner_active = np.asarray(active, dtype=bool)
        aggregate_volume = np.asarray(
            control_volume_geometry.cells.aggregate_volume, dtype=np.float64
        )
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        return
    nx, ny, nz = diag.shape
    expected_active = np.zeros((nx, ny, nz), dtype=bool)
    for i, value in enumerate(q):
        expected_active[i, ::value, :] = True
    if not np.array_equal(owner_active, expected_active):
        raise ValueError("angular-agglomeration active owners do not match the nested profile")
    owner_i = np.asarray(control_volume_geometry.cells.owner_i, dtype=np.int32)
    owner_j = np.asarray(control_volume_geometry.cells.owner_j, dtype=np.int32)
    owner_k = np.asarray(control_volume_geometry.cells.owner_k, dtype=np.int32)
    expected_i = np.broadcast_to(np.arange(nx)[:, None, None], (nx, ny, nz))
    expected_j = np.empty((nx, ny, nz), dtype=np.int32)
    for i, value in enumerate(q):
        expected_j[i] = np.broadcast_to(
            ((np.arange(ny) // value) * value)[:, None], (ny, nz)
        )
    expected_k = np.broadcast_to(np.arange(nz)[None, None, :], (nx, ny, nz))
    if not (
        np.array_equal(owner_i, expected_i)
        and np.array_equal(owner_j, expected_j)
        and np.array_equal(owner_k, expected_k)
    ):
        raise ValueError("angular-agglomeration owner map does not match the nested profile")
    if not np.all(np.isfinite(diag[owner_active])) or np.any(diag[owner_active] <= 0.0):
        raise ValueError("angular tree requires finite positive owner diagonals")
    if not np.all(np.isfinite(aggregate_volume[owner_active])) or np.any(aggregate_volume[owner_active] <= 0.0):
        raise ValueError("angular tree requires finite positive aggregate volumes")
    roots = owner_active.copy()
    roots[1:] = False
    if np.any(edge[roots] != 0.0):
        raise ValueError("angular tree roots cannot carry parent edges")
    for i in range(1, nx):
        for j in range(0, ny, q[i]):
            expected_j = (j // q[i - 1]) * q[i - 1]
            for k in range(nz):
                if (pi[i, j, k], pj[i, j, k], pk[i, j, k]) != (i - 1, expected_j, k):
                    raise ValueError("angular tree parent indices do not match the nested profile")
                if not np.isfinite(edge[i, j, k]) or edge[i, j, k] <= 0.0:
                    raise ValueError("each nonroot angular owner requires one finite positive total radial edge")
    if np.any(edge[~owner_active] != 0.0):
        raise ValueError("angular tree aliases cannot carry radial edges")

    # The stored graph is symmetric by construction: each child owns exactly
    # one scalar edge used in both matrix entries.  Positive Schur pivots are
    # the concrete SPD check for that symmetric tree matrix.
    pivots = diag.copy()
    for i in range(nx - 1, 0, -1):
        for j in range(0, ny, q[i]):
            p_j = (j // q[i - 1]) * q[i - 1]
            for k in range(nz):
                pivot = pivots[i, j, k]
                if not np.isfinite(pivot) or pivot <= 0.0:
                    raise ValueError("angular tree principal matrix is not positive definite")
                t = edge[i, j, k]
                pivots[i - 1, p_j, k] -= t * t / pivot
    if not np.all(np.isfinite(pivots[0, 0, :])) or np.any(pivots[0, 0, :] <= 0.0):
        raise ValueError("angular tree principal matrix is not positive definite")


def _assemble_angular_agglomeration_tree_principal_coefficients(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    face_bc: LocalBoundaryFaceBC3D,
    config: SolvaxGmresConfig,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Assemble the nested radial owner graph for angular agglomeration.

    ``child_edge[i,j,k]`` is the summed physical conductance between active
    owner ``(i,j,k)`` and its unique parent.  The coefficients are assembled
    directly as the owner-space principal operator ``P.T @ A_f @ P`` from
    the ordinary fine-grid faces.  Theta and eta owner couplings remain in
    the diagonal but are deliberately omitted from this line-u graph.
    """
    shard_counts = tuple(int(v) for v in domain.shard_spec.shard_counts)
    if shard_counts[0] != 1 or shard_counts[1] != 1:
        raise ValueError(
            "angular-agglomeration line-u supports eta-only sharding; "
            f"got shard_counts={shard_counts}"
        )
    if not control_volume_geometry.has_angular_agglomeration:
        raise ValueError("angular-agglomeration tree requires an angular group profile")
    cells = control_volume_geometry.cells
    nx, ny, nz = geometry.owned_shape
    profile = tuple(int(value) for value in control_volume_geometry.angular_group_sizes)

    # Static topology arrays.  Their values depend only on the nested profile,
    # so they are closed over by the runtime solve rather than rebuilt there.
    parent_i_np = np.full((nx, ny, nz), -1, dtype=np.int32)
    parent_j_np = np.full((nx, ny, nz), -1, dtype=np.int32)
    parent_k_np = np.full((nx, ny, nz), -1, dtype=np.int32)
    for i in range(1, nx):
        q_parent = profile[i - 1]
        for j in range(0, ny, profile[i]):
            parent_i_np[i, j, :] = i - 1
            parent_j_np[i, j, :] = (j // q_parent) * q_parent
            parent_k_np[i, j, :] = np.arange(nz, dtype=np.int32)
    parent_i = jnp.asarray(parent_i_np)
    parent_j = jnp.asarray(parent_j_np)
    parent_k = jnp.asarray(parent_k_np)

    spacing = (
        jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64),
        jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64),
        jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64),
    )
    logical_measure = (
        spacing[1] * spacing[2],
        spacing[0] * spacing[2],
        spacing[0] * spacing[1],
    )
    # RLP uses the unmodified fine-grid operator.  In particular, do not use
    # ``control_volume_geometry.regular_faces`` here: that legacy compact-face
    # payload closes faces crossed by an aggregate and therefore cannot form
    # P.T @ A_f @ P.
    regular = geometry.regular_face_geometry
    face_areas = (regular.x_area, regular.y_area, regular.z_area)
    face_fractions = (regular.x_area_fraction, regular.y_area_fraction, regular.z_area_fraction)
    face_open = (regular.x_open_mask, regular.y_open_mask, regular.z_open_mask)
    face_metrics = (geometry.face_metric.x, geometry.face_metric.y, geometry.face_metric.z)
    grids = (geometry.grid.x, geometry.grid.y, geometry.grid.z)
    locations = ("x_face", "y_face", "z_face")
    kinds = (face_bc.kind_x, face_bc.kind_y, face_bc.kind_z)
    masks = (face_bc.mask_x, face_bc.mask_y, face_bc.mask_z)
    conductances: list[jnp.ndarray] = []
    for axis in range(3):
        face_slices = geometry.layout.location_owned_slices(locations[axis])
        axis_slice = face_slices[axis]
        if axis_slice.start is None or axis_slice.stop is None:
            raise ValueError("owned face slices must have finite bounds")
        centers = jnp.asarray(grids[axis].centers_halo, dtype=jnp.float64)
        distance_1d = centers[axis_slice.start:axis_slice.stop] - centers[
            axis_slice.start - 1:axis_slice.stop - 1
        ]
        distance_shape = [1, 1, 1]
        distance_shape[axis] = int(distance_1d.shape[0])
        logical_area = _lift_cell_field_to_faces(
            logical_measure[axis], axis=axis, periodic=False
        )
        T = (
            jnp.asarray(face_metrics[axis].J_owned, dtype=jnp.float64)
            * jnp.asarray(face_projectors[axis][..., axis, axis], dtype=jnp.float64)
            * jnp.asarray(face_areas[axis], dtype=jnp.float64)
            * jnp.asarray(face_fractions[axis], dtype=jnp.float64)
            * jnp.asarray(face_open[axis], dtype=jnp.float64)
            * logical_area
            / jnp.maximum(jnp.reshape(distance_1d, distance_shape), 1.0e-30)
        )
        kind = jnp.asarray(kinds[axis], dtype=jnp.int32)
        mask = jnp.asarray(masks[axis], dtype=bool)
        prescribed_flux = mask & (
            (kind == BC_NEUMANN) | (kind == BC_NORMALFLUX) | (kind == BC_NOFLUX)
        )
        T = jnp.where(prescribed_flux, 0.0, T)
        T = jnp.where(mask & (kind == BC_DIRICHLET), 3.0 * T, T)
        conductances.append(T)
    Tx, Ty, Tz = conductances

    owner_i = jnp.asarray(cells.owner_i, dtype=jnp.int32)
    owner_j = jnp.asarray(cells.owner_j, dtype=jnp.int32)
    owner_k = jnp.asarray(cells.owner_k, dtype=jnp.int32)
    active = jnp.asarray(cells.is_active_owner, dtype=bool)
    aggregate_volume = jnp.asarray(cells.aggregate_volume, dtype=jnp.float64)
    diagonal = jnp.where(
        active,
        jnp.asarray(config.regularization_epsilon, dtype=jnp.float64)
        * aggregate_volume,
        0.0,
    )
    child_edge = jnp.zeros((nx, ny, nz), dtype=jnp.float64)

    def add_owner_edge(
        current_diagonal: jnp.ndarray,
        conductance: jnp.ndarray,
        left: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        right: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        distinct = (left[0] != right[0]) | (left[1] != right[1]) | (left[2] != right[2])
        edge_T = jnp.where(distinct, conductance, 0.0)
        current_diagonal = current_diagonal.at[left].add(edge_T)
        current_diagonal = current_diagonal.at[right].add(edge_T)
        return current_diagonal, edge_T

    fine_j = jnp.broadcast_to(jnp.arange(ny)[:, None], (ny, nz))
    fine_k = jnp.broadcast_to(jnp.arange(nz)[None, :], (ny, nz))

    # Radial fine faces map every child owner to its unique owner on the next
    # inner ring.  Multiple fine subfaces naturally sum into one tree edge.
    if nx > 1:
        for i in range(1, nx):
            left = (
                owner_i[i - 1, fine_j, fine_k],
                owner_j[i - 1, fine_j, fine_k],
                owner_k[i - 1, fine_j, fine_k],
            )
            right = (
                owner_i[i, fine_j, fine_k],
                owner_j[i, fine_j, fine_k],
                owner_k[i, fine_j, fine_k],
            )
            distinct = (left[0] != right[0]) | (left[1] != right[1]) | (left[2] != right[2])
            expected = (
                (right[0] == i)
                & (left[0] == parent_i[right])
                & (left[1] == parent_j[right])
                & (left[2] == parent_k[right])
            )
            fine_T = jnp.asarray(Tx[i], dtype=jnp.float64)
            try:
                if bool(jnp.any(~jnp.isfinite(fine_T) | (fine_T < 0.0))):
                    raise ValueError("fine radial conductances must be finite and nonnegative")
                if bool(jnp.any((fine_T > 0.0) & distinct & ~expected)):
                    raise ValueError("fine radial face does not connect a child to its declared parent")
            except jax.errors.TracerBoolConversionError:
                pass
            diagonal, edge_T = add_owner_edge(diagonal, fine_T, left, right)
            child_edge = child_edge.at[right].add(edge_T)

    # The lower radial face is the coordinate axis, not a physical boundary.
    # The upper radial wall contributes the one-sided boundary conductance.
    outer = (owner_i[-1], owner_j[-1], owner_k[-1])
    diagonal = diagonal.at[outer].add(jnp.asarray(Tx[-1], dtype=jnp.float64))

    # One representative of each periodic theta face.  Face j=0 connects the
    # final fine cell to cell zero; face j=ny duplicates it and is omitted.
    fine_i_y = jnp.broadcast_to(jnp.arange(nx)[:, None], (nx, nz))
    fine_k_y = jnp.broadcast_to(jnp.arange(nz)[None, :], (nx, nz))
    for j in range(ny):
        left_j = (j - 1) % ny
        left = (
            owner_i[fine_i_y, left_j, fine_k_y],
            owner_j[fine_i_y, left_j, fine_k_y],
            owner_k[fine_i_y, left_j, fine_k_y],
        )
        right = (
            owner_i[fine_i_y, j, fine_k_y],
            owner_j[fine_i_y, j, fine_k_y],
            owner_k[fine_i_y, j, fine_k_y],
        )
        diagonal, _ = add_owner_edge(
            diagonal, jnp.asarray(Ty[:, j, :], dtype=jnp.float64), left, right
        )

    # Eta is the only decomposed direction supported by this path.  A local
    # eta slab is not periodic: when it has a neighboring slab, both of its
    # local endpoint faces contribute to the diagonal of the adjacent local
    # owner, while their off-diagonal entries are handled by the distributed
    # operator.  On one device the final face is the duplicate periodic
    # representative of face zero, so retain the old one-face-per-edge loop.
    fine_i_z = jnp.broadcast_to(jnp.arange(nx)[:, None], (nx, ny))
    fine_j_z = jnp.broadcast_to(jnp.arange(ny)[None, :], (nx, ny))
    if shard_counts[2] == 1:
        for k in range(nz):
            left_k = (k - 1) % nz
            left = (
                owner_i[fine_i_z, fine_j_z, left_k],
                owner_j[fine_i_z, fine_j_z, left_k],
                owner_k[fine_i_z, fine_j_z, left_k],
            )
            right = (
                owner_i[fine_i_z, fine_j_z, k],
                owner_j[fine_i_z, fine_j_z, k],
                owner_k[fine_i_z, fine_j_z, k],
            )
            diagonal, _ = add_owner_edge(
                diagonal, jnp.asarray(Tz[:, :, k], dtype=jnp.float64), left, right
            )
    else:
        # Every owned eta face is included.  At k=0 only the right local
        # cell exists; at k=nz only the left local cell exists.  Thus a
        # cross-shard face contributes its conductance to this shard's
        # endpoint diagonal without manufacturing a local periodic edge.
        for k in range(nz + 1):
            conductance = jnp.asarray(Tz[:, :, k], dtype=jnp.float64)
            if k > 0:
                left = (
                    owner_i[fine_i_z, fine_j_z, k - 1],
                    owner_j[fine_i_z, fine_j_z, k - 1],
                    owner_k[fine_i_z, fine_j_z, k - 1],
                )
                diagonal = diagonal.at[left].add(conductance)
            if k < nz:
                right = (
                    owner_i[fine_i_z, fine_j_z, k],
                    owner_j[fine_i_z, fine_j_z, k],
                    owner_k[fine_i_z, fine_j_z, k],
                )
                diagonal = diagonal.at[right].add(conductance)

    diagonal = jnp.where(active, diagonal, 0.0)
    child_edge = jnp.where(active, child_edge, 0.0)
    result = (
        aggregate_volume,
        diagonal,
        child_edge,
        parent_i,
        parent_j,
        parent_k,
        active,
    )
    _validate_concrete_angular_agglomeration_tree_assembly(
        control_volume_geometry,
        diagonal,
        child_edge,
        parent_i,
        parent_j,
        parent_k,
        active,
    )
    return result


def _build_angular_agglomeration_line_u_preconditioner(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    face_bc: LocalBoundaryFaceBC3D,
    config: SolvaxGmresConfig,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    principal_coefficients: tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
        jnp.ndarray, jnp.ndarray, jnp.ndarray,
    ] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build the exact per-eta nested radial-tree line-u solve."""
    shard_counts = tuple(int(v) for v in domain.shard_spec.shard_counts)
    if shard_counts[0] != 1 or shard_counts[1] != 1:
        raise ValueError(
            "angular-agglomeration line-u supports eta-only sharding; "
            f"got shard_counts={shard_counts}"
        )
    if principal_coefficients is None:
        principal_coefficients = _assemble_angular_agglomeration_tree_principal_coefficients(
            geometry, domain, face_projectors, face_bc, config,
            control_volume_geometry,
        )
    aggregate_volume, diagonal, child_edge, parent_i, parent_j, parent_k, active = principal_coefficients
    _validate_concrete_angular_agglomeration_tree_assembly(
        control_volume_geometry, diagonal, child_edge,
        parent_i, parent_j, parent_k, active,
    )
    nx, ny, nz = geometry.owned_shape
    eta_index = jnp.broadcast_to(jnp.arange(nz, dtype=jnp.int32)[None, :], (ny, nz))

    def solve(residual: jnp.ndarray) -> jnp.ndarray:
        residual = jnp.asarray(residual, dtype=jnp.float64)
        if residual.shape != geometry.owned_shape:
            raise ValueError("angular tree residual must match geometry.owned_shape")
        rhs = jnp.where(active, aggregate_volume * residual, 0.0)
        pivots = jnp.asarray(diagonal, dtype=jnp.float64)

        def eliminate(level, state):
            current_pivots, current_rhs = state
            i = nx - 1 - level
            mask = active[i]
            edge = jnp.where(mask, child_edge[i], 0.0)
            pivot = current_pivots[i]
            p_j = jnp.maximum(parent_j[i], 0)
            schur = jnp.where(mask, -(edge * edge) / pivot, 0.0)
            rhs_update = jnp.where(mask, edge * current_rhs[i] / pivot, 0.0)
            current_pivots = current_pivots.at[i - 1, p_j, eta_index].add(schur)
            current_rhs = current_rhs.at[i - 1, p_j, eta_index].add(rhs_update)
            return current_pivots, current_rhs

        pivots, rhs = jax.lax.fori_loop(0, nx - 1, eliminate, (pivots, rhs))
        solution = jnp.zeros_like(residual).at[0, 0, :].set(rhs[0, 0, :] / pivots[0, 0, :])

        def substitute(level, current_solution):
            i = level + 1
            mask = active[i]
            p_j = jnp.maximum(parent_j[i], 0)
            parent_value = current_solution[i - 1, p_j, eta_index]
            value = (rhs[i] + child_edge[i] * parent_value) / pivots[i]
            return current_solution.at[i].set(jnp.where(mask, value, 0.0))

        solution = jax.lax.fori_loop(0, nx - 1, substitute, solution)
        return jnp.where(active, solution, 0.0)

    return solve


def _build_projected_owner_line_u_preconditioner(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    face_bc: LocalBoundaryFaceBC3D,
    config: SolvaxGmresConfig,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a fine-line/prolong/restrict preconditioner for plane-local owners.

    Corner/edge aggregates do not form the nested radial tree used by the
    angular RLP preconditioner.  Use the ordinary fine-grid line-u inverse as
    an approximation, prolonging each owner residual to its members and
    volume-averaging the resulting correction back to owner space.  The
    production operator remains the exact projected control-volume operator.
    """
    shard_counts = tuple(int(v) for v in domain.shard_spec.shard_counts)
    if shard_counts[0] != 1 or shard_counts[1] != 1:
        raise ValueError(
            "projected-owner line-u supports eta-only sharding; "
            f"got shard_counts={shard_counts}"
        )
    cells = control_volume_geometry.cells
    owner_i = jnp.asarray(cells.owner_i, dtype=jnp.int32)
    owner_j = jnp.asarray(cells.owner_j, dtype=jnp.int32)
    owner_k = jnp.asarray(cells.owner_k, dtype=jnp.int32)
    active = jnp.asarray(cells.is_active_owner, dtype=bool)
    raw_volume = jnp.asarray(cells.raw_volume, dtype=jnp.float64)
    aggregate_volume = jnp.asarray(cells.aggregate_volume, dtype=jnp.float64)
    diagonal, lower, upper = _principal_perp_laplacian_bands(
        geometry,
        domain,
        face_projectors,
        face_bc,
        regularization_epsilon=config.regularization_epsilon,
    )
    fine_line_solve = solvax_line_preconditioner(
        diagonal,
        ((0, lower[0], upper[0]),),
    )

    def solve(residual: jnp.ndarray) -> jnp.ndarray:
        residual = jnp.asarray(residual, dtype=jnp.float64)
        if residual.shape != geometry.owned_shape:
            raise ValueError(
                "projected-owner residual must match geometry.owned_shape"
            )
        prolonged = residual[owner_i, owner_j, owner_k]
        fine_correction = fine_line_solve(prolonged)
        restricted = jnp.zeros_like(residual).at[
            owner_i, owner_j, owner_k
        ].add(raw_volume * fine_correction)
        owner_correction = restricted / jnp.maximum(aggregate_volume, 1.0e-30)
        return jnp.where(active, owner_correction, 0.0)

    return solve


def build_solvax_perp_laplacian_preconditioner(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    face_bc: LocalBoundaryFaceBC3D,
    config: SolvaxGmresConfig,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
    """Build the configured local right preconditioner for SOLVAX FGMRES.

    This returns a fixed-cost local approximation to the inverse of the
    positive operator ``-L_perp``.  It intentionally does *not* invoke an
    inner Krylov solve, so it is safe to reuse as a block inside a larger
    matrix-free Newton--FGMRES preconditioner.
    """

    kind = config.preconditioner
    if kind == "none":
        return None
    if control_volume_geometry is not None:
        if kind != "line-u":
            raise ValueError(
                "RLP control-volume preconditioning supports only 'none' or 'line-u'"
            )
        if bool(getattr(control_volume_geometry, "has_angular_agglomeration", False)):
            return _build_angular_agglomeration_line_u_preconditioner(
                geometry,
                domain,
                face_projectors,
                face_bc,
                config,
                control_volume_geometry,
            )
        if bool(getattr(control_volume_geometry, "has_projected_owner_agglomeration", False)):
            return _build_projected_owner_line_u_preconditioner(
                geometry,
                domain,
                face_projectors,
                face_bc,
                config,
                control_volume_geometry,
            )
        raise ValueError(
            "control-volume line-u preconditioning requires a projected-owner "
            "agglomeration topology"
        )
    diagonal, lower, upper = _principal_perp_laplacian_bands(
        geometry,
        domain,
        face_projectors,
        face_bc,
        regularization_epsilon=config.regularization_epsilon,
    )
    if kind == "jacobi":
        return solvax_jacobi(diagonal)
    selected_axes = {
        "line-u": (0,),
        "line-v": (1,),
        "line-uv": (0, 1),
    }[kind]
    directions = tuple(
        (axis, lower[axis], upper[axis])
        for axis in selected_axes
    )
    return solvax_line_preconditioner(diagonal, directions)


@_pytree_base
@dataclass(frozen=True)
class LocalPerpLaplacianInverseSolver:
    """SOLVAX FGMRES adapter for local conservative perpendicular-Laplacian inversion."""

    geometry: LocalFciGeometry3D
    domain: LocalDomain3D
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None
    control_volume_boundary_bc: LocalControlVolumeBoundaryBC3D | None = None
    stencil_builder: LocalConservativeStencilBuilder = (
        build_local_conservative_stencil_from_field
    )
    halo_exchange: HaloExchange3D | None = None
    topology_filler: TopologyHaloFiller3D | None = None
    physical_ghost_filler: PhysicalGhostCellFiller3D | None = None
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None
    face_bc: LocalBoundaryFaceBC3D | None = None
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False)
    neumann_normal_scheme: str = "logical"
    b_floor: float = 1.0e-30
    jacobian_floor: float = 1.0e-30
    config: SolvaxGmresConfig = SolvaxGmresConfig()
    # Optional configured context for cut-wall stencil policies.
    stencil_builder_context: StencilBuilderContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, LocalFciGeometry3D):
            raise TypeError("geometry must be a LocalFciGeometry3D instance")
        if not isinstance(self.domain, LocalDomain3D):
            raise TypeError("domain must be a LocalDomain3D instance")
        if self.geometry.layout != self.domain.layout:
            raise ValueError("geometry and domain must share the same HaloLayout3D")
        if not isinstance(self.stencil_builder, LocalConservativeStencilBuilder):
            raise TypeError("stencil_builder must be a LocalConservativeStencilBuilder")
        if self.stencil_builder_context is not None:
            if not isinstance(self.stencil_builder_context, StencilBuilderContext):
                raise TypeError(
                    "stencil_builder_context must be a StencilBuilderContext or None"
                )
            if self.stencil_builder_context.layout != self.domain.layout:
                raise ValueError(
                    "stencil_builder_context must share the solver domain layout"
                )
            context_domain = self.stencil_builder_context.domain
            if context_domain is None:
                raise ValueError(
                    "stencil_builder_context must include the solver domain"
                )
            if (
                context_domain.layout != self.domain.layout
                or context_domain.shard_spec != self.domain.shard_spec
                or context_domain.mesh_axis_names != self.domain.mesh_axis_names
            ):
                raise ValueError(
                    "stencil_builder_context must share the solver domain metadata"
                )
        if self.halo_exchange is not None and not isinstance(self.halo_exchange, HaloExchange3D):
            raise TypeError("halo_exchange must be a HaloExchange3D or None")
        if self.topology_filler is not None and not isinstance(
            self.topology_filler,
            TopologyHaloFiller3D,
        ):
            raise TypeError("topology_filler must be a TopologyHaloFiller3D or None")
        if self.physical_ghost_filler is not None and not isinstance(
            self.physical_ghost_filler,
            PhysicalGhostCellFiller3D,
        ):
            raise TypeError(
                "physical_ghost_filler must be a PhysicalGhostCellFiller3D or None"
            )
        has_control_volume_geometry = self.control_volume_geometry is not None
        has_control_volume_bc = self.control_volume_boundary_bc is not None
        if has_control_volume_geometry != has_control_volume_bc:
            raise ValueError(
                "control_volume_geometry and control_volume_boundary_bc must "
                "either both be supplied or both be None"
            )
        if self.control_volume_geometry is not None:
            if not isinstance(
                self.control_volume_geometry,
                LocalEmbeddedControlVolumeGeometry3D,
            ):
                raise TypeError(
                    "control_volume_geometry must be a "
                    "LocalEmbeddedControlVolumeGeometry3D or None"
                )
            if self.control_volume_geometry.layout != self.geometry.layout:
                raise ValueError(
                    "control_volume_geometry must share geometry.layout"
                )
            if not isinstance(
                self.control_volume_boundary_bc,
                LocalControlVolumeBoundaryBC3D,
            ):
                raise TypeError(
                    "control_volume_boundary_bc must be a "
                    "LocalControlVolumeBoundaryBC3D or None"
                )
            if (
                self.control_volume_boundary_bc.max_rows
                != self.control_volume_geometry.irregular_faces.max_rows
            ):
                raise ValueError(
                    "control_volume_boundary_bc must align with irregular "
                    "face rows"
                )
        if self.face_bc is not None and not isinstance(self.face_bc, LocalBoundaryFaceBC3D):
            raise TypeError("face_bc must be a LocalBoundaryFaceBC3D or None")
        axis_regular_axes = tuple(bool(value) for value in self.axis_regular_axes)
        if len(axis_regular_axes) != 3:
            raise ValueError("axis_regular_axes must have length 3")
        object.__setattr__(self, "axis_regular_axes", axis_regular_axes)
        if self.neumann_normal_scheme not in ("logical", "physical"):
            raise ValueError(
                "neumann_normal_scheme must be 'logical' or 'physical'"
            )
        if not isinstance(self.config, SolvaxGmresConfig):
            raise TypeError("config must be a SolvaxGmresConfig instance")
        object.__setattr__(self, "b_floor", float(self.b_floor))
        object.__setattr__(self, "jacobian_floor", float(self.jacobian_floor))

    def _default_face_bc(self) -> LocalBoundaryFaceBC3D:
        return self.face_bc or LocalBoundaryFaceBC3D.empty(self.domain.layout)

    def _default_control_volume_boundary_bc(
        self,
    ) -> LocalControlVolumeBoundaryBC3D | None:
        return self.control_volume_boundary_bc

    def _apply_A(
        self,
        field_owned: jnp.ndarray,
        *,
        face_bc: LocalBoundaryFaceBC3D,
        control_volume_boundary_bc: LocalControlVolumeBoundaryBC3D | None,
        project_mean_zero: bool,
    ) -> jnp.ndarray:
        active_mask = _solver_active_mask(
            self.geometry,
            self.control_volume_geometry,
        )
        volume_weights = _solver_volume_weights(
            self.geometry,
            self.control_volume_geometry,
        )
        values = _mask_inactive_owned(
            field_owned,
            self.geometry,
            active_mask=active_mask,
        )
        if project_mean_zero:
            values = _spmd_remove_weighted_mean(
                values,
                self.geometry,
                self.domain,
                active_mask,
                volume_weights,
            )

        storage_values = values
        if self.control_volume_geometry is not None:
            owner_halo = inject_owned_field_to_halo(values, self.domain.layout)
            if self.halo_exchange is not None:
                owner_halo = self.halo_exchange(owner_halo, self.domain)
            storage_values = expand_local_control_volume_owner_field(
                values,
                self.control_volume_geometry.cells,
                owner_values_halo=owner_halo,
            )
        field_halo = inject_owned_field_to_halo(
            storage_values,
            self.domain.layout,
        )
        if self.physical_ghost_filler is not None:
            field_halo = LocalHaloClosure3D(
                physical_ghost_filler=self.physical_ghost_filler,
                halo_exchange=self.halo_exchange,
                topology_filler=self.topology_filler,
            )(
                field_halo,
                self.domain,
                face_bc,
            )
        else:
            if self.halo_exchange is not None:
                field_halo = self.halo_exchange(field_halo, self.domain)
            if self.topology_filler is not None:
                field_halo = self.topology_filler(field_halo, self.domain)

        context = self.stencil_builder_context
        if context is None and self.control_volume_geometry is not None:
            raise ValueError(
                "control-volume solves require an explicit StencilBuilderContext"
            )
        if context is None:
            context = StencilBuilderContext(
                layout=self.domain.layout,
                domain=self.domain,
            )
        local = self.stencil_builder(field_halo, self.geometry, context)
        face_projectors = self.face_projectors
        if face_projectors is None:
            face_projectors = build_local_perp_laplacian_face_projectors(
                self.geometry,
                self.domain,
                b_floor=self.b_floor,
                axis_regular_axes=self.axis_regular_axes,
            )
        if self.control_volume_geometry is not None:
            # Apply the ordinary direct-polar conservative operator on the
            # materialized fine storage, then restrict its volume average back
            # to the unique owner space.  In particular, do not pass compact
            # control-volume geometry or its projected field closure here:
            # those replace the fine-grid face fluxes and define the separate
            # independently fitted coarse-face operator.
            assert self.control_volume_geometry is not None
            fine_result = -local_perp_laplacian_conservative_op(
                local,
                self.geometry,
                self.domain,
                face_projectors=face_projectors,
                face_bc=face_bc,
                axis_regular_axes=self.axis_regular_axes,
                neumann_normal_scheme=self.neumann_normal_scheme,
                b_floor=self.b_floor,
                jacobian_floor=self.jacobian_floor,
            )
            result = aggregate_local_control_volume_average(
                fine_result,
                self.control_volume_geometry.cells,
                self.domain,
            )
        else:
            result = -local_perp_laplacian_conservative_op(
                local,
                self.geometry,
                self.domain,
                face_projectors=face_projectors,
                face_bc=face_bc,
                axis_regular_axes=self.axis_regular_axes,
                neumann_normal_scheme=self.neumann_normal_scheme,
                b_floor=self.b_floor,
                jacobian_floor=self.jacobian_floor,
            )
        if self.config.regularization_epsilon != 0.0:
            result = result + self.config.regularization_epsilon * values
        if project_mean_zero:
            result = _spmd_remove_weighted_mean(
                result,
                self.geometry,
                self.domain,
                active_mask,
                volume_weights,
            )
        return _mask_inactive_owned(
            result,
            self.geometry,
            active_mask=active_mask,
        )

    def solve_full_grid(
        self,
        rhs_owned: jnp.ndarray,
        *,
        guess_owned: jnp.ndarray | None = None,
        phi_guess_owned: jnp.ndarray | None = None,
        phi_lift_owned: jnp.ndarray | None = None,
        lift_owned: jnp.ndarray | None = None,
        return_diagnostics: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, SolvaxGmresInfo]:
        """Solve on the ordinary owned grid.

        Angular-RLP systems have a different unknown space and must use
        :meth:`solve_rlp_owner` explicitly.
        """
        if self.control_volume_geometry is not None or self.control_volume_boundary_bc is not None:
            raise ValueError(
                "solve_full_grid cannot be used with control-volume geometry/BC; "
                "use solve_rlp_owner"
            )
        return self._solve_common(
            rhs_owned,
            guess_owned=guess_owned,
            phi_guess_owned=phi_guess_owned,
            phi_lift_owned=phi_lift_owned,
            lift_owned=lift_owned,
            return_diagnostics=return_diagnostics,
            solve_mode="full-grid",
        )

    def solve_rlp_owner(
        self,
        rhs_owned: jnp.ndarray,
        *,
        guess_owned: jnp.ndarray | None = None,
        phi_guess_owned: jnp.ndarray | None = None,
        phi_lift_owned: jnp.ndarray | None = None,
        lift_owned: jnp.ndarray | None = None,
        return_diagnostics: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, SolvaxGmresInfo]:
        """Solve a projected-owner-space system explicitly.

        Angular RLP additionally requires lower-radial axis regularity;
        corner-edge owner agglomeration uses the ordinary square topology.
        """
        if self.control_volume_geometry is None or self.control_volume_boundary_bc is None:
            raise ValueError(
                "solve_rlp_owner requires control_volume_geometry and "
                "control_volume_boundary_bc"
            )
        if (
            self.control_volume_geometry.has_angular_agglomeration
            and self.axis_regular_axes[0] is not True
        ):
            raise ValueError(
                "angular solve_rlp_owner requires axis_regular_axes[0]=True"
            )
        context = self.stencil_builder_context
        if context is None:
            raise ValueError(
                "solve_rlp_owner requires an explicit StencilBuilderContext"
            )
        return self._solve_common(
            rhs_owned,
            guess_owned=guess_owned,
            phi_guess_owned=phi_guess_owned,
            phi_lift_owned=phi_lift_owned,
            lift_owned=lift_owned,
            return_diagnostics=return_diagnostics,
            solve_mode="rlp-owner",
        )

    def __call__(
        self,
        rhs_owned: jnp.ndarray,
        *,
        guess_owned: jnp.ndarray | None = None,
        phi_guess_owned: jnp.ndarray | None = None,
        phi_lift_owned: jnp.ndarray | None = None,
        lift_owned: jnp.ndarray | None = None,
        return_diagnostics: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, SolvaxGmresInfo]:
        """Backward-compatible ordinary-grid entry point."""
        return self.solve_full_grid(
            rhs_owned,
            guess_owned=guess_owned,
            phi_guess_owned=phi_guess_owned,
            phi_lift_owned=phi_lift_owned,
            lift_owned=lift_owned,
            return_diagnostics=return_diagnostics,
        )

    def _solve_common(
        self,
        rhs_owned: jnp.ndarray,
        *,
        guess_owned: jnp.ndarray | None = None,
        phi_guess_owned: jnp.ndarray | None = None,
        phi_lift_owned: jnp.ndarray | None = None,
        lift_owned: jnp.ndarray | None = None,
        return_diagnostics: bool = False,
        solve_mode: Literal["full-grid", "rlp-owner"],
    ) -> jnp.ndarray | tuple[jnp.ndarray, SolvaxGmresInfo]:
        if solve_mode not in ("full-grid", "rlp-owner"):
            raise ValueError(f"unknown phi solve mode: {solve_mode!r}")
        if solve_mode == "rlp-owner":
            if self.control_volume_geometry is None or self.control_volume_boundary_bc is None:
                raise ValueError("rlp-owner mode requires control-volume geometry and BC")
            if (
                self.control_volume_geometry.has_angular_agglomeration
                and self.axis_regular_axes[0] is not True
            ):
                raise ValueError(
                    "angular rlp-owner mode requires axis_regular_axes[0]=True"
                )
            context = self.stencil_builder_context
            if context is None:
                raise ValueError("rlp-owner mode requires an explicit stencil context")
        elif self.control_volume_geometry is not None or self.control_volume_boundary_bc is not None:
            raise ValueError("full-grid mode cannot carry control-volume geometry/BC")
        rhs = jnp.asarray(rhs_owned, dtype=jnp.float64)
        if rhs.shape != self.geometry.owned_shape:
            raise ValueError(
                f"rhs_owned must have shape {self.geometry.owned_shape}, got {rhs.shape}"
            )
        if guess_owned is not None and phi_guess_owned is not None:
            raise ValueError("use only one of guess_owned or phi_guess_owned")
        if guess_owned is None:
            guess_owned = phi_guess_owned
        if guess_owned is None:
            guess = jnp.zeros_like(rhs)
        else:
            guess = jnp.asarray(guess_owned, dtype=jnp.float64)
            if guess.shape != self.geometry.owned_shape:
                raise ValueError(
                    "guess_owned must have shape "
                    f"{self.geometry.owned_shape}, got {guess.shape}"
                )
        if phi_lift_owned is not None and lift_owned is not None:
            raise ValueError("use only one of phi_lift_owned or lift_owned")
        if phi_lift_owned is None:
            phi_lift_owned = lift_owned
        if phi_lift_owned is not None:
            lift = jnp.asarray(phi_lift_owned, dtype=jnp.float64)
            if lift.shape != self.geometry.owned_shape:
                raise ValueError(
                    "phi_lift_owned must have shape "
                    f"{self.geometry.owned_shape}, got {lift.shape}"
                )
        else:
            lift = None

        face_bc = self._default_face_bc()
        control_volume_boundary_bc = self._default_control_volume_boundary_bc()
        project_mean_zero = bool(self.config.project_mean_zero)
        active_mask = _solver_active_mask(
            self.geometry,
            self.control_volume_geometry,
        )
        volume_weights = _solver_volume_weights(
            self.geometry,
            self.control_volume_geometry,
        )
        rhs = _mask_inactive_owned(
            rhs,
            self.geometry,
            active_mask=active_mask,
        )
        guess = _mask_inactive_owned(
            guess,
            self.geometry,
            active_mask=active_mask,
        )
        if lift is not None:
            lift = jnp.asarray(lift, dtype=jnp.float64)
            if self.control_volume_geometry is not None:
                lift = _mask_inactive_owned(
                    lift,
                    self.geometry,
                    active_mask=active_mask,
                )

        if lift is None:
            homogeneous_face_bc = _homogeneous_local_face_bc(face_bc)
            homogeneous_control_volume_boundary_bc = (
                None
                if control_volume_boundary_bc is None
                else _homogeneous_local_control_volume_boundary_bc(
                    control_volume_boundary_bc,
                )
            )
            boundary_source = self._apply_A(
                jnp.zeros_like(rhs),
                face_bc=face_bc,
                control_volume_boundary_bc=control_volume_boundary_bc,
                project_mean_zero=project_mean_zero,
            )
            linear_rhs = _mask_inactive_owned(
                rhs - boundary_source,
                self.geometry,
                active_mask=active_mask,
            )
            initial_guess = _mask_inactive_owned(
                guess,
                self.geometry,
                active_mask=active_mask,
            )
        else:
            homogeneous_face_bc = _dirichlet_lift_correction_local_face_bc(face_bc)
            homogeneous_control_volume_boundary_bc = (
                None
                if control_volume_boundary_bc is None
                else _dirichlet_lift_correction_local_control_volume_boundary_bc(
                    control_volume_boundary_bc,
                )
            )
            lift_source = self._apply_A(
                lift,
                face_bc=face_bc,
                control_volume_boundary_bc=control_volume_boundary_bc,
                project_mean_zero=project_mean_zero,
            )
            linear_rhs = _mask_inactive_owned(
                rhs - lift_source,
                self.geometry,
                active_mask=active_mask,
            )
            initial_guess = _mask_inactive_owned(
                guess - lift,
                self.geometry,
                active_mask=active_mask,
            )

        if project_mean_zero:
            linear_rhs = _spmd_remove_weighted_mean(
                linear_rhs,
                self.geometry,
                self.domain,
                active_mask,
                volume_weights,
            )
            initial_guess = _spmd_remove_weighted_mean(
                initial_guess,
                self.geometry,
                self.domain,
                active_mask,
                volume_weights,
            )

        def apply_A(field_owned: jnp.ndarray) -> jnp.ndarray:
            return self._apply_A(
                field_owned,
                face_bc=homogeneous_face_bc,
                control_volume_boundary_bc=homogeneous_control_volume_boundary_bc,
                project_mean_zero=project_mean_zero,
            )

        preconditioner_projectors = self.face_projectors
        if preconditioner_projectors is None:
            preconditioner_projectors = build_local_perp_laplacian_face_projectors(
                self.geometry,
                self.domain,
                b_floor=self.b_floor,
                axis_regular_axes=self.axis_regular_axes,
            )
        raw_preconditioner = build_solvax_perp_laplacian_preconditioner(
            self.geometry,
            self.domain,
            preconditioner_projectors,
            homogeneous_face_bc,
            self.config,
            control_volume_geometry=self.control_volume_geometry,
        )
        preconditioner = None
        if raw_preconditioner is not None:
            def preconditioner(residual: jnp.ndarray) -> jnp.ndarray:
                residual = _mask_inactive_owned(
                    residual,
                    self.geometry,
                    active_mask=active_mask,
                )
                # CV line-u is already an owner-space pole-star solve.  The
                # non-CV branch remains the historical regular-storage path.
                correction = raw_preconditioner(residual)
                return _mask_inactive_owned(
                    correction,
                    self.geometry,
                    active_mask=active_mask,
                )
        solution, info = solvax_gmres_solve(
            apply_A,
            linear_rhs,
            initial_guess,
            self.geometry,
            self.domain,
            self.config,
            active_cell_mask=active_mask,
            preconditioner=preconditioner,
            volume_weights=volume_weights,
        )
        if lift is not None:
            inactive_solution = (
                0.0 if self.control_volume_geometry is not None else lift
            )
            solution = jnp.where(active_mask, lift + solution, inactive_solution)
        else:
            solution = _mask_inactive_owned(
                solution,
                self.geometry,
                active_mask=active_mask,
            )
        if return_diagnostics:
            return solution, info
        return solution

    def tree_flatten(self):
        children = (
            self.geometry,
            self.domain,
            self.stencil_builder,
            self.stencil_builder_context,
            self.halo_exchange,
            self.topology_filler,
            self.physical_ghost_filler,
            self.face_projectors,
            self.control_volume_geometry,
            self.control_volume_boundary_bc,
            self.face_bc,
            self.config,
        )
        aux_data = (
            self.axis_regular_axes,
            self.neumann_normal_scheme,
            self.b_floor,
            self.jacobian_floor,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            geometry,
            domain,
            stencil_builder,
            stencil_builder_context,
            halo_exchange,
            topology_filler,
            physical_ghost_filler,
            face_projectors,
            control_volume_geometry,
            control_volume_boundary_bc,
            face_bc,
            config,
        ) = children
        (
            axis_regular_axes,
            neumann_normal_scheme,
            b_floor,
            jacobian_floor,
        ) = aux_data
        return cls(
            geometry=geometry,
            domain=domain,
            stencil_builder=stencil_builder,
            stencil_builder_context=stencil_builder_context,
            halo_exchange=halo_exchange,
            topology_filler=topology_filler,
            physical_ghost_filler=physical_ghost_filler,
            face_projectors=face_projectors,
            control_volume_geometry=control_volume_geometry,
            control_volume_boundary_bc=control_volume_boundary_bc,
            face_bc=face_bc,
            axis_regular_axes=axis_regular_axes,
            neumann_normal_scheme=neumann_normal_scheme,
            b_floor=b_floor,
            jacobian_floor=jacobian_floor,
            config=config,
        )
