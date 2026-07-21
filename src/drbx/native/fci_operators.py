from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from functools import partial
from itertools import permutations
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

try:  # Optional external solver backend.
    import lineax as lx
except ImportError:  # pragma: no cover - depends on local optional install
    lx = None

_pytree_base = jax.tree_util.register_pytree_node_class

from ..geometry import (
    CellVolumeGeometry3D,
    CellCenteredGrid3D,
    ConservativeStencilBuilder,
    HaloLayout3D,
    FaceBFieldGeometry,
    FaceMetricGeometry,
    FciGeometry3D,
    FciMaps3D,
    Grid1D,
    LocalBFieldGeometry,
    LocalCellCenteredGrid3D,
    LocalAggregateCellGeometry3D,
    LocalControlVolumeCellGeometry3D,
    LocalCellVolumeGeometry3D,
    LocalDomain3D,
    LocalFciDirectionMap,
    LocalFciGeometry3D,
    LocalFciLocalDependencyTable,
    LocalFciMaps3D,
    LocalFciRemoteDependencyTable,
    LocalFciStencilBuilder,
    LocalFaceBFieldGeometry,
    LocalFaceMetricGeometry,
    LocalGrid1D,
    LocalMetricGeometry,
    LocalRegularFaceGeometry3D,
    LocalSpacing3D,
    LocalStencilBuilder,
    LocalConservativeStencilBuilder,
    NeighborMap3D,
    ShardSpec3D,
    Spacing3D,
    RegularFaceGeometry3D,
    build_conservative_stencil_from_field,
    build_local_conservative_stencil_from_field,
    build_local_direct_stencil_one_sided_physical_from_halo,
    build_local_fci_stencil_from_field,
    build_local_stencil_from_field,
)
from ..geometry.fci_geometry import (
    StencilBuilderContext,
    _global_axis_stencil_from_field,
)
from .fci import _first_derivative_3d
from .fci_halo import (
    HaloExchange3D,
    LocalHaloClosure3D,
    PhysicalGhostCellFiller3D,
    TopologyHaloFiller3D,
)
from .fci_gmres import (
    SpmdGmresConfig,
    SpmdGmresInfo,
    _spmd_remove_weighted_mean,
    spmd_gmres_solve,
)
from .fci_model import (
    inject_owned_field_to_halo,
    inject_owned_vector_field_to_halo,
)
from .fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BC_NONE,
    BC_NORMALFLUX,
    BC_NOFLUX,
    LocalBoundaryConditionBuilder,
    LocalBoundaryData3D,
    LocalBoundaryFaceBC3D,
    LocalControlVolumeFluxStencil3D,
    LocalCoordinateFaceValueReconstructor3D,
    LocalCoordinateNormalDerivativeConstructor3D,
    LocalCoordinateSideValues1D,
    LocalCoordinateSideValues3D,
    LocalCellGradient3D,
    LocalControlVolumeBoundaryBC3D,
    LocalControlVolumeFaceRows3D,
    LocalControlVolumePolynomial3D,
    LocalEmbeddedControlVolumeGeometry3D,
    LocalQuadraticReconstruction3D,
    LocalRegularBoundaryMomentClosure3D,
    LocalCutWallBC3D,
    LocalCutWallGeometry3D,
    LocalRegularFaceContributionRows3D,
    LocalCutWallNormalDerivativeConstructor3D,
    LocalCutWallValueReconstructor3D,
    CutWallBC3D,
    CutWallGeometry3D,
    BoundaryFaceBC3D,
    FaceFluxStencil3D,
    ConservativeStencil3D,
    LocalStencil1D,
    LocalStencil3D,
    CV_FACE_CUT_WALL,
    CV_FACE_PHYSICAL_BOUNDARY,
    CV_RECONSTRUCTION_EQUATION_CELL,
    CV_RECONSTRUCTION_EQUATION_DIRICHLET,
    CV_RECONSTRUCTION_EQUATION_REMOTE_CELL,
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
) -> jnp.ndarray:
    """Mask inactive owned cells in scalar or component-valued owned arrays."""

    array = jnp.asarray(values)
    mask = _active_cell_mask_owned(geometry)
    if array.shape[:3] != geometry.owned_shape:
        raise ValueError(
            "values must begin with geometry.owned_shape "
            f"{geometry.owned_shape}, got {array.shape}"
        )
    for _ in range(array.ndim - 3):
        mask = mask[..., None]
    return jnp.where(mask, array, jnp.asarray(inactive_value, dtype=array.dtype))


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


def grad_parallel_op_fci(
    stencil: LocalStencil1D,
    geometry: FciGeometry3D,
) -> jnp.ndarray:
    """Global/reference centered FCI parallel gradient.

    Computes ``grad_parallel(f)`` from a field-line stencil. This reference
    path assumes ``stencil.shape == geometry.shape``.
    """

    if stencil.shape != geometry.shape:
        raise ValueError(
            f"stencil must have shape {geometry.shape}, got {stencil.shape}"
        )

    return _take_stencil_finite_difference(stencil)


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


def local_conservative_parallel_flux_div_op(
    field_halo_full: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    *,
    context: StencilBuilderContext,
    fci_stencil_builder: LocalFciStencilBuilder = build_local_fci_stencil_from_field,
    forward_remote_q_values: jnp.ndarray | None = None,
    backward_remote_q_values: jnp.ndarray | None = None,
    cut_wall_q_values: jnp.ndarray | None = None,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Local/domain-decomposed FCI conservative parallel flux divergence.

    Computes ``div(F b)`` through the continuum identity

        ``div(F b) = B * grad_parallel(F / B)``

    using the local FCI interpolation stencil for ``F / B``. Any remote or
    cut-wall endpoint values passed to this function must already be values of
    ``F / B`` at those endpoints.
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
    q_stencil = fci_stencil_builder(
        q_halo,
        geometry,
        context,
        forward_remote_values=forward_remote_q_values,
        backward_remote_values=backward_remote_q_values,
        cut_wall_values=cut_wall_q_values,
    )
    grad_parallel_q = local_grad_parallel_op_fci(q_stencil, geometry)
    result = Bmag_halo[geometry.layout.owned_slices_cell] * grad_parallel_q
    return _mask_inactive_owned(result, geometry)


def grad_parallel_op_direct(
    stencil: LocalStencil3D,
    geometry: FciGeometry3D,
    *,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Global/reference direct finite-difference parallel gradient.

    Computes ``grad_parallel(f) = b^i partial_i f`` using coordinate-direction
    derivative stencils. This reference path assumes
    ``stencil.shape == geometry.shape`` and that the cell-centered magnetic
    field arrays are shaped like the stencil.
    """

    if stencil.shape != geometry.shape:
        raise ValueError(
            f"stencil must have shape {geometry.shape}, got {stencil.shape}"
        )

    dfdx = _take_stencil_finite_difference(
        stencil.x,
    )

    dfdy = _take_stencil_finite_difference(
        stencil.y,
    )

    dfdz = _take_stencil_finite_difference(
        stencil.z,
    )

    df = jnp.stack((dfdx, dfdy, dfdz), axis=-1)

    # Prefer the explicit B_contra/Bmag representation. Fall back to the
    # normalized field property for older global/reference geometry objects.
    if hasattr(geometry.cell_bfield, "B_contra") and hasattr(
        geometry.cell_bfield,
        "Bmag",
    ):
        B_contra = jnp.asarray(geometry.cell_bfield.B_contra, dtype=jnp.float64)
        Bmag = jnp.asarray(geometry.cell_bfield.Bmag, dtype=jnp.float64)
        Bmag = jnp.maximum(Bmag, float(b_floor))
        b_contra = B_contra / Bmag[..., None]
    else:
        b_contra = jnp.asarray(
            geometry.cell_bfield.b_contra,
            dtype=jnp.float64,
        )

    return jnp.einsum("...i,...i->...", b_contra, df)


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


def _build_global_conservative_stencil_compat(
    stencil_builder: ConservativeStencilBuilder,
    field: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool],
    face_bc: BoundaryFaceBC3D,
) -> ConservativeStencil3D:
    """Call either the legacy global builder or the current geometry builder."""

    try:
        return stencil_builder(
            field,
            geometry,
            periodic_axes=periodic_axes,
            face_bc=face_bc,
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        del face_bc
        return _global_axis_stencil_from_field(
            field,
            geometry,
            periodic_axes=periodic_axes,
        )


def parallel_laplacian_direct_op(
    field: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    stencil_builder: LocalStencilBuilder | ConservativeStencilBuilder = build_local_stencil_from_field,
    face_bc: BoundaryFaceBC3D | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
) -> jnp.ndarray:
    """Return the chained direct parallel Laplacian ``grad_parallel(grad_parallel(f))``.

    The operator is built by reconstructing a local stencil for ``field``,
    applying ``grad_parallel_op_direct`` once, reconstructing a second stencil
    for the intermediate field, and applying ``grad_parallel_op_direct`` again.
    """

    if not isinstance(stencil_builder, (LocalStencilBuilder, ConservativeStencilBuilder)):
        raise TypeError(
            "stencil_builder must be a LocalStencilBuilder or ConservativeStencilBuilder instance"
        )

    if face_bc is None:
        face_bc = BoundaryFaceBC3D.empty(RegularFaceGeometry3D.unit(geometry))

    first_stencil = _build_global_conservative_stencil_compat(
        stencil_builder,
        field,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=face_bc,
    )
    first_grad = grad_parallel_op_direct(first_stencil, geometry)

    second_stencil = _build_global_conservative_stencil_compat(
        stencil_builder,
        first_grad,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=face_bc,
    )
    return grad_parallel_op_direct(second_stencil, geometry)


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


def grad_perp_op(
    stencil: LocalStencil3D,
    geometry: FciGeometry3D,
    *,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the projected perpendicular gradient from a 3D local stencil."""

    if stencil.shape != geometry.shape:
        raise ValueError(
            f"stencil must have shape {geometry.shape}, got {stencil.shape}"
        )

    dfdx = _take_stencil_finite_difference(stencil.x)
    dfdy = _take_stencil_finite_difference(stencil.y)
    dfdz = _take_stencil_finite_difference(stencil.z)
    df = jnp.stack((dfdx, dfdy, dfdz), axis=-1)

    cell_metric = geometry.cell_metric
    cell_bfield = geometry.cell_bfield
    g = jnp.stack(
        [
            jnp.stack([cell_metric.g11, cell_metric.g12, cell_metric.g13], axis=-1),
            jnp.stack([cell_metric.g12, cell_metric.g22, cell_metric.g23], axis=-1),
            jnp.stack([cell_metric.g13, cell_metric.g23, cell_metric.g33], axis=-1),
        ],
        axis=-2,
    )
    b = jnp.asarray(cell_bfield.B_contra, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(cell_bfield.Bmag, dtype=jnp.float64), float(b_floor))
    b_unit = b / bmag[..., None]
    projector = g - jnp.einsum("...i,...j->...ij", b_unit, b_unit)
    return jnp.einsum("...ij,...j->...i", projector, df)


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

    using owned-cell coordinate stencils. The intermediate contravariant flux
    ``F^i = J P^{ij} partial_j f`` is injected as one vector-valued halo field,
    so all three components pass through one halo exchange and one topology
    filler call. The intermediate stencil builder is responsible for the
    physical-boundary closure of each scalar component (for example, the
    one-sided builder used by the chained parallel Laplacian).

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


def perp_laplacian_local_op(
    stencil: LocalStencil3D,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return a pointwise local approximation of the perpendicular Laplacian.

    This operator stays entirely on the cell-centered reconstruction layer:
    it uses ``LocalStencil3D`` to recover the field gradient, projects that
    gradient with the cell-centered perpendicular projector, forms a
    cell-centered flux, and then takes a standard logical divergence.

    Unlike ``perp_laplacian_conservative_op``, this path does not consume
    face-flux payloads or any cut-wall geometry.
    """

    if stencil.shape != geometry.shape:
        raise ValueError(f"stencil must have shape {geometry.shape}, got {stencil.shape}")

    periodic_axes = tuple(bool(value) for value in periodic_axes)

    dfdx = _take_stencil_finite_difference(stencil.x)
    dfdy = _take_stencil_finite_difference(stencil.y)
    dfdz = _take_stencil_finite_difference(stencil.z)
    grad_f = jnp.stack((dfdx, dfdy, dfdz), axis=-1)

    cell_metric = geometry.cell_metric
    cell_bfield = geometry.cell_bfield
    projector = cell_metric.g_contra
    b = jnp.asarray(cell_bfield.B_contra, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(cell_bfield.Bmag, dtype=jnp.float64), float(b_floor))
    b_unit = b / bmag[..., None]
    projector = projector - jnp.einsum("...i,...j->...ij", b_unit, b_unit)

    j = jnp.asarray(cell_metric.J, dtype=jnp.float64)
    flux = j[..., None] * jnp.einsum("...ij,...j->...i", projector, grad_f)
    div_flux = (
        _first_derivative_3d(flux[..., 0], geometry.spacing.dx, axis=0, periodic=periodic_axes[0])
        + _first_derivative_3d(flux[..., 1], geometry.spacing.dy, axis=1, periodic=periodic_axes[1])
        + _first_derivative_3d(flux[..., 2], geometry.spacing.dz, axis=2, periodic=periodic_axes[2])
    )
    return div_flux / jnp.maximum(j, float(jacobian_floor))


def poisson_bracket_op(
    f_stencil: LocalStencil3D,
    g_stencil: LocalStencil3D,
    geometry: FciGeometry3D,
    *,
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the logical Poisson bracket from local stencils."""

    metric = geometry.cell_metric
    cell_bfield = geometry.cell_bfield
    if f_stencil.shape != geometry.shape:
        raise ValueError(f"f_stencil must have shape {geometry.shape}, got {f_stencil.shape}")
    if g_stencil.shape != geometry.shape:
        raise ValueError(f"g_stencil must have shape {geometry.shape}, got {g_stencil.shape}")

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

    g_cov = metric.g_cov
    b = jnp.asarray(cell_bfield.B_contra, dtype=jnp.float64)
    bmag = jnp.maximum(jnp.asarray(cell_bfield.Bmag, dtype=jnp.float64), float(b_floor))
    b_unit = b / bmag[..., None]
    b_covariant = jnp.einsum("...ij,...j->...i", g_cov, b_unit)
    cross = jnp.cross(df, dg)
    return jnp.sum(b_covariant * cross, axis=-1) / jnp.maximum(
        jnp.asarray(metric.J, dtype=jnp.float64),
        float(jacobian_floor),
    )


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


def curvature_op(
    stencil: LocalStencil3D,
    geometry: FciGeometry3D,
    *,
    curvature_coefficients: jnp.ndarray,
) -> jnp.ndarray:
    """Return the curvature operator applied to a local scalar-field stencil."""

    if stencil.shape != geometry.shape:
        raise ValueError(f"stencil must have shape {geometry.shape}, got {stencil.shape}")
    if curvature_coefficients.shape != geometry.shape + (3,):
        raise ValueError(
            f"curvature_coefficients must have shape {geometry.shape + (3,)}, got {curvature_coefficients.shape}"
        )
    dfdx = _take_stencil_finite_difference(stencil.x)
    dfdy = _take_stencil_finite_difference(stencil.y)
    dfdz = _take_stencil_finite_difference(stencil.z)
    grad_f = jnp.stack((dfdx, dfdy, dfdz), axis=-1)
    return jnp.einsum("...i,...i->...", jnp.asarray(curvature_coefficients, dtype=jnp.float64), grad_f)


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


def _build_laplacian_face_projectors(
    geometry: FciGeometry3D,
    *,
    b_floor: float = 1.0e-30,
    parallel: bool,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build the geometry-only face projectors for projected Laplacians."""

    b_floor_value = float(b_floor)
    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    if axis_regular_axes[1] or axis_regular_axes[2]:
        raise NotImplementedError(
            "axis_regular_axes currently only supports the lower x axis for face projectors; "
            f"got axis_regular_axes={axis_regular_axes}"
        )

    def _axis_regularize_lower_x_face(projector: jnp.ndarray) -> jnp.ndarray:
        if not axis_regular_axes[0]:
            return projector
        if projector.shape[1] % 2 != 0:
            raise ValueError("axis-regular lower-x face projectors require an even poloidal grid")
        return projector.at[0].set(jnp.zeros_like(projector[0]))

    def _face_projector(metric, bfield, *, family_axis: int) -> jnp.ndarray:
        b_contra = jnp.asarray(bfield.B_contra, dtype=jnp.float64)
        b_contra = jnp.where(jnp.isfinite(b_contra), b_contra, 0.0)
        bmag = jnp.asarray(bfield.Bmag, dtype=jnp.float64)
        bmag = jnp.where(jnp.isfinite(bmag), bmag, b_floor_value)
        b = b_contra / jnp.maximum(bmag[..., None], b_floor_value)
        projector = jnp.einsum("...i,...j->...ij", b, b)
        if not parallel:
            projector = jnp.asarray(metric.g_contra, dtype=jnp.float64) - projector
        if family_axis == 0:
            projector = _axis_regularize_lower_x_face(projector)
        return jnp.where(jnp.isfinite(projector), projector, 0.0)

    return (
        _face_projector(geometry.face_metric.x, geometry.face_bfield.x, family_axis=0),
        _face_projector(geometry.face_metric.y, geometry.face_bfield.y, family_axis=1),
        _face_projector(geometry.face_metric.z, geometry.face_bfield.z, family_axis=2),
    )


def build_perp_laplacian_face_projectors(
    geometry: FciGeometry3D,
    *,
    b_floor: float = 1.0e-30,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build the geometry-only face projectors for the perpendicular Laplacian.

    The returned tensors are the face-family analogs of ``P_face``:
    ``g_contra - b_hat ⊗ b_hat`` evaluated on x-, y-, and z-face grids.
    """

    return _build_laplacian_face_projectors(
        geometry,
        b_floor=b_floor,
        parallel=False,
        axis_regular_axes=axis_regular_axes,
    )


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


def _build_projected_laplacian_stencil(
    local: ConservativeStencil3D,
    geometry: FciGeometry3D,
    *,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    face_bc: BoundaryFaceBC3D | None = None,
    regular_face_geometry: RegularFaceGeometry3D | None = None,
    cut_wall_geometry: CutWallGeometry3D | None = None,
    cut_wall_bc: CutWallBC3D | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    b_floor: float = 1.0e-30,
) -> LocalControlVolumeFluxStencil3D:
    """Build a boundary-complete control-volume flux stencil for a projected Laplacian.

    ``ConservativeStencil3D`` is the conservative-operator input type.
    It carries the cell-based reconstruction data needed to build regular
    coordinate-face fluxes, but it is intentionally separate from
    ``LocalStencil3D`` so local and conservative semantics cannot be mixed
    accidentally.

    Regular coordinate-face boundary conditions are applied while constructing
    the face fluxes. Cut-wall payloads are assembled here so the divergence
    kernel only performs flux balance.
    """

    if local.shape != geometry.shape:
        raise ValueError(f"local stencil must have shape {geometry.shape}, got {local.shape}")

    periodic_axes = tuple(bool(value) for value in periodic_axes)
    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    regular_face_geometry = regular_face_geometry or RegularFaceGeometry3D.unit(geometry)
    cell_volume = CellVolumeGeometry3D.unit(geometry)
    if face_bc is None:
        face_bc = BoundaryFaceBC3D.empty(regular_face_geometry)
    cut_wall_geometry = cut_wall_geometry or CutWallGeometry3D.empty()
    cut_wall_bc = cut_wall_bc or CutWallBC3D.empty()
    if face_projectors is None:
        face_projectors = build_perp_laplacian_face_projectors(geometry, b_floor=b_floor)
    x_face_projector, y_face_projector, z_face_projector = face_projectors

    values = jnp.asarray(local.x.center, dtype=jnp.float64)
    x_face_grad = jnp.asarray(local.face_grad.x, dtype=jnp.float64)
    y_face_grad = jnp.asarray(local.face_grad.y, dtype=jnp.float64)
    z_face_grad = jnp.asarray(local.face_grad.z, dtype=jnp.float64)

    def _require_face_shape(value: jnp.ndarray, expected_shape: tuple[int, ...], name: str) -> None:
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")

    _require_face_shape(x_face_grad, regular_face_geometry.x_area.shape + (3,), "x_face_grad")
    _require_face_shape(y_face_grad, regular_face_geometry.y_area.shape + (3,), "y_face_grad")
    _require_face_shape(z_face_grad, regular_face_geometry.z_area.shape + (3,), "z_face_grad")

    def _patch_axis_face_gradients(
        face_grad: jnp.ndarray,
        *,
        axis: int,
        axis_kind: jnp.ndarray,
        axis_value: jnp.ndarray,
        axis_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        if periodic_axes[axis]:
            return face_grad

        face_grad = jnp.asarray(face_grad, dtype=jnp.float64)
        kind = jnp.asarray(axis_kind, dtype=jnp.int32)
        value = jnp.asarray(axis_value, dtype=jnp.float64)
        mask = jnp.asarray(axis_mask, dtype=bool)

        if axis == 0:
            lower_value = value[0]
            upper_value = value[-1]
            lower_kind = kind[0]
            upper_kind = kind[-1]
            lower_mask = mask[0]
            upper_mask = mask[-1]
            lower_distance = jnp.asarray(geometry.grid.x.lower_center_to_face, dtype=jnp.float64)
            upper_distance = jnp.asarray(geometry.grid.x.upper_center_to_face, dtype=jnp.float64)
            lower_center = values[0]
            upper_center = values[-1]
        elif axis == 1:
            lower_value = value[:, 0, :]
            upper_value = value[:, -1, :]
            lower_kind = kind[:, 0, :]
            upper_kind = kind[:, -1, :]
            lower_mask = mask[:, 0, :]
            upper_mask = mask[:, -1, :]
            lower_distance = jnp.asarray(geometry.grid.y.lower_center_to_face, dtype=jnp.float64)
            upper_distance = jnp.asarray(geometry.grid.y.upper_center_to_face, dtype=jnp.float64)
            lower_center = values[:, 0, :]
            upper_center = values[:, -1, :]
        else:
            lower_value = value[:, :, 0]
            upper_value = value[:, :, -1]
            lower_kind = kind[:, :, 0]
            upper_kind = kind[:, :, -1]
            lower_mask = mask[:, :, 0]
            upper_mask = mask[:, :, -1]
            lower_distance = jnp.asarray(geometry.grid.z.lower_center_to_face, dtype=jnp.float64)
            upper_distance = jnp.asarray(geometry.grid.z.upper_center_to_face, dtype=jnp.float64)
            lower_center = values[:, :, 0]
            upper_center = values[:, :, -1]

        lower_kind_mask = lower_mask & (lower_kind == BC_DIRICHLET)
        upper_kind_mask = upper_mask & (upper_kind == BC_DIRICHLET)
        lower_plane = face_grad[_axis_index_nd(axis, 0, face_grad.ndim)]
        lower_normal = lower_plane[..., axis]
        lower_coord = (lower_center - lower_value) / jnp.maximum(lower_distance, 1.0e-30)
        if not (axis == 0 and axis_regular_axes[0]):
            lower_plane = lower_plane.at[..., axis].set(jnp.where(lower_kind_mask, lower_coord, lower_normal))
            face_grad = face_grad.at[_axis_index_nd(axis, 0, face_grad.ndim)].set(lower_plane)

        upper_plane = face_grad[_axis_index_nd(axis, -1, face_grad.ndim)]
        upper_normal = upper_plane[..., axis]
        upper_coord = (upper_value - upper_center) / jnp.maximum(upper_distance, 1.0e-30)
        upper_plane = upper_plane.at[..., axis].set(jnp.where(upper_kind_mask, upper_coord, upper_normal))
        face_grad = face_grad.at[_axis_index_nd(axis, -1, face_grad.ndim)].set(upper_plane)

        lower_kind_mask = lower_mask & (lower_kind == BC_NEUMANN)
        upper_kind_mask = upper_mask & (upper_kind == BC_NEUMANN)
        if not (axis == 0 and axis_regular_axes[0]):
            lower_plane = face_grad[_axis_index_nd(axis, 0, face_grad.ndim)]
            lower_normal = lower_plane[..., axis]
            lower_plane = lower_plane.at[..., axis].set(jnp.where(lower_kind_mask, -lower_value, lower_normal))
            face_grad = face_grad.at[_axis_index_nd(axis, 0, face_grad.ndim)].set(lower_plane)

        upper_plane = face_grad[_axis_index_nd(axis, -1, face_grad.ndim)]
        upper_normal = upper_plane[..., axis]
        upper_plane = upper_plane.at[..., axis].set(jnp.where(upper_kind_mask, upper_value, upper_normal))
        face_grad = face_grad.at[_axis_index_nd(axis, -1, face_grad.ndim)].set(upper_plane)

        return face_grad

    x_face_grad = _patch_axis_face_gradients(
        x_face_grad,
        axis=0,
        axis_kind=face_bc.kind_x,
        axis_value=face_bc.value_x,
        axis_mask=face_bc.mask_x,
    )
    y_face_grad = _patch_axis_face_gradients(
        y_face_grad,
        axis=1,
        axis_kind=face_bc.kind_y,
        axis_value=face_bc.value_y,
        axis_mask=face_bc.mask_y,
    )
    z_face_grad = _patch_axis_face_gradients(
        z_face_grad,
        axis=2,
        axis_kind=face_bc.kind_z,
        axis_value=face_bc.value_z,
        axis_mask=face_bc.mask_z,
    )

    x_face_metric = geometry.face_metric.x
    y_face_metric = geometry.face_metric.y
    z_face_metric = geometry.face_metric.z

    x_flux = jnp.asarray(x_face_metric.J, dtype=jnp.float64) * jnp.einsum(
        "...j,...j->...", x_face_projector[..., 0, :], x_face_grad
    )
    if axis_regular_axes[0]:
        x_flux = x_flux.at[0].set(jnp.zeros_like(x_flux[0]))
    y_flux = jnp.asarray(y_face_metric.J, dtype=jnp.float64) * jnp.einsum(
        "...j,...j->...", y_face_projector[..., 1, :], y_face_grad
    )
    z_flux = jnp.asarray(z_face_metric.J, dtype=jnp.float64) * jnp.einsum(
        "...j,...j->...", z_face_projector[..., 2, :], z_face_grad
    )

    def _apply_face_flux_bc(flux: jnp.ndarray, *, axis: int, axis_kind: jnp.ndarray, axis_value: jnp.ndarray, axis_mask: jnp.ndarray) -> jnp.ndarray:
        if periodic_axes[axis]:
            return flux
        result = jnp.asarray(flux, dtype=jnp.float64)
        if axis == 0:
            lower_kind = axis_kind[0]
            upper_kind = axis_kind[-1]
            lower_value = axis_value[0]
            upper_value = axis_value[-1]
            lower_mask = axis_mask[0]
            upper_mask = axis_mask[-1]
        elif axis == 1:
            lower_kind = axis_kind[:, 0, :]
            upper_kind = axis_kind[:, -1, :]
            lower_value = axis_value[:, 0, :]
            upper_value = axis_value[:, -1, :]
            lower_mask = axis_mask[:, 0, :]
            upper_mask = axis_mask[:, -1, :]
        else:
            lower_kind = axis_kind[:, :, 0]
            upper_kind = axis_kind[:, :, -1]
            lower_value = axis_value[:, :, 0]
            upper_value = axis_value[:, :, -1]
            lower_mask = axis_mask[:, :, 0]
            upper_mask = axis_mask[:, :, -1]

        if not (axis == 0 and axis_regular_axes[0]):
            lower_plane = result[_axis_index_nd(axis, 0, result.ndim)]
            lower_plane = jnp.where(lower_mask & (lower_kind == BC_NORMALFLUX), lower_value, lower_plane)
            lower_plane = jnp.where(lower_mask & (lower_kind == BC_NOFLUX), 0.0, lower_plane)
            result = result.at[_axis_index_nd(axis, 0, result.ndim)].set(lower_plane)

        upper_plane = result[_axis_index_nd(axis, -1, result.ndim)]
        upper_plane = jnp.where(upper_mask & (upper_kind == BC_NORMALFLUX), upper_value, upper_plane)
        upper_plane = jnp.where(upper_mask & (upper_kind == BC_NOFLUX), 0.0, upper_plane)
        result = result.at[_axis_index_nd(axis, -1, result.ndim)].set(upper_plane)
        return result

    x_flux = _apply_face_flux_bc(x_flux, axis=0, axis_kind=face_bc.kind_x, axis_value=face_bc.value_x, axis_mask=face_bc.mask_x)
    y_flux = _apply_face_flux_bc(y_flux, axis=1, axis_kind=face_bc.kind_y, axis_value=face_bc.value_y, axis_mask=face_bc.mask_y)
    z_flux = _apply_face_flux_bc(z_flux, axis=2, axis_kind=face_bc.kind_z, axis_value=face_bc.value_z, axis_mask=face_bc.mask_z)

    cut_wall_flux = _build_cut_wall_flux_payload(
        local=local,
        geometry=geometry,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        b_floor=b_floor,
    )

    return LocalControlVolumeFluxStencil3D(
        regular_flux=FaceFluxStencil3D(x=x_flux, y=y_flux, z=z_flux),
        regular_face_geometry=regular_face_geometry,
        cell_volume=cell_volume,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_flux=cut_wall_flux,
    )


def build_perp_laplacian_stencil(
    local: ConservativeStencil3D,
    geometry: FciGeometry3D,
    *,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    face_bc: BoundaryFaceBC3D | None = None,
    regular_face_geometry: RegularFaceGeometry3D | None = None,
    cut_wall_geometry: CutWallGeometry3D | None = None,
    cut_wall_bc: CutWallBC3D | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    b_floor: float = 1.0e-30,
) -> LocalControlVolumeFluxStencil3D:
    """Build a boundary-complete control-volume flux stencil for ``-∇·(P⊥∇f)``."""

    if face_projectors is None:
        face_projectors = build_perp_laplacian_face_projectors(
            geometry,
            b_floor=b_floor,
            axis_regular_axes=axis_regular_axes,
        )
    return _build_projected_laplacian_stencil(
        local,
        geometry,
        face_projectors=face_projectors,
        face_bc=face_bc,
        regular_face_geometry=regular_face_geometry,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        periodic_axes=periodic_axes,
        axis_regular_axes=axis_regular_axes,
        b_floor=b_floor,
    )


def _build_cut_wall_flux_payload(
    *,
    local: ConservativeStencil3D,
    geometry: FciGeometry3D,
    cut_wall_geometry: CutWallGeometry3D,
    cut_wall_bc: CutWallBC3D,
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Build the embedded-wall flux payload for the conservative control volume.

    The wall contribution is assembled here into an already-integrated flux-area
    term so the divergence kernel can remain a pure finite-volume balance.
    """

    if not cut_wall_geometry.n_wall_faces:
        return jnp.zeros((0,), dtype=jnp.float64)

    cut_wall_kind = jnp.asarray(cut_wall_bc.kind, dtype=jnp.int32)
    cut_wall_value = jnp.asarray(cut_wall_bc.value, dtype=jnp.float64)
    if cut_wall_kind.ndim != 1 or cut_wall_value.ndim != 1:
        raise ValueError("cut_wall_bc.kind and cut_wall_bc.value must be 1D")
    if cut_wall_kind.shape != cut_wall_value.shape:
        raise ValueError(
            f"cut_wall_bc.kind and cut_wall_bc.value must have the same shape, got {cut_wall_kind.shape} and {cut_wall_value.shape}"
        )
    if cut_wall_kind.size != cut_wall_geometry.n_wall_faces:
        raise ValueError(
            f"cut_wall_bc must have {cut_wall_geometry.n_wall_faces} entries, got {cut_wall_kind.size}"
        )
    supported = (
        (cut_wall_kind == BC_NONE)
        | (cut_wall_kind == BC_DIRICHLET)
        | (cut_wall_kind == BC_NEUMANN)
        | (cut_wall_kind == BC_NORMALFLUX)
        | (cut_wall_kind == BC_NOFLUX)
    )
    if not bool(jnp.all(supported)):
        raise NotImplementedError(
            "cut-wall BC kinds other than NONE, NOFLUX, and NORMALFLUX are not implemented"
        )

    owner_i = jnp.asarray(cut_wall_geometry.owner_i, dtype=jnp.int32)
    owner_j = jnp.asarray(cut_wall_geometry.owner_j, dtype=jnp.int32)
    owner_k = jnp.asarray(cut_wall_geometry.owner_k, dtype=jnp.int32)

    field = jnp.asarray(local.x.center, dtype=jnp.float64)
    dfdx_cell = _take_stencil_finite_difference(local.x)
    dfdy_cell = _take_stencil_finite_difference(local.y)
    dfdz_cell = _take_stencil_finite_difference(local.z)

    grad_cell = jnp.stack(
        (
            dfdx_cell[owner_i, owner_j, owner_k],
            dfdy_cell[owner_i, owner_j, owner_k],
            dfdz_cell[owner_i, owner_j, owner_k],
        ),
        axis=-1,
    )
    f_cell = field[owner_i, owner_j, owner_k]
    owner_center = jnp.stack(
        (
            jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)[owner_i],
            jnp.asarray(geometry.grid.y.centers, dtype=jnp.float64)[owner_j],
            jnp.asarray(geometry.grid.z.centers, dtype=jnp.float64)[owner_k],
        ),
        axis=-1,
    )

    normal_contra = jnp.asarray(cut_wall_geometry.normal_contra, dtype=jnp.float64)
    normal_cov = jnp.einsum("...ij,...j->...i", jnp.asarray(cut_wall_geometry.g_cov, dtype=jnp.float64), normal_contra)
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
    # For true cut walls, BC_NORMALFLUX is treated as an already integrated
    # wall flux-area contribution.
    wall_flux_area = jnp.where(cut_wall_kind == BC_NORMALFLUX, cut_wall_value, wall_flux_area)
    wall_flux_area = jnp.where(cut_wall_kind == BC_NOFLUX, 0.0, wall_flux_area)

    sign = jnp.asarray(cut_wall_geometry.sign, dtype=jnp.float64)
    if sign.shape != wall_flux_area.shape:
        raise ValueError(
            f"cut_wall_geometry.sign must have shape {wall_flux_area.shape}, got {sign.shape}"
        )
    return sign * wall_flux_area


def divergence_conservative_op(
    cv_flux: LocalControlVolumeFluxStencil3D,
    geometry: FciGeometry3D,
    *,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the conservative divergence from a completed control-volume flux stencil."""

    if cv_flux.shape != geometry.shape:
        raise ValueError(f"cv_flux must have shape {geometry.shape}, got {cv_flux.shape}")

    def _divergence_from_face_flux(flux: jnp.ndarray, spacing: jnp.ndarray | float, *, axis: int, area: jnp.ndarray) -> jnp.ndarray:
        face_flux = jnp.asarray(flux, dtype=jnp.float64) * jnp.asarray(area, dtype=jnp.float64)
        h = jnp.asarray(spacing, dtype=jnp.float64)
        if h.ndim == 0:
            h = jnp.ones_like(face_flux[_axis_slice_nd(axis, 1, None, face_flux.ndim)]) * h
        h = jnp.maximum(h, 1.0e-30)
        return (
            face_flux[_axis_slice_nd(axis, 1, None, face_flux.ndim)]
            - face_flux[_axis_slice_nd(axis, None, -1, face_flux.ndim)]
        ) / h

    div_flux = (
        _divergence_from_face_flux(
            cv_flux.regular_flux.x,
            geometry.spacing.dx,
            axis=0,
            area=
                cv_flux.regular_face_geometry.x_area
                * cv_flux.regular_face_geometry.x_area_fraction
                * cv_flux.regular_face_geometry.x_open_mask,
        )
        + _divergence_from_face_flux(
            cv_flux.regular_flux.y,
            geometry.spacing.dy,
            axis=1,
            area=
                cv_flux.regular_face_geometry.y_area
                * cv_flux.regular_face_geometry.y_area_fraction
                * cv_flux.regular_face_geometry.y_open_mask,
        )
        + _divergence_from_face_flux(
            cv_flux.regular_flux.z,
            geometry.spacing.dz,
            axis=2,
            area=
                cv_flux.regular_face_geometry.z_area
                * cv_flux.regular_face_geometry.z_area_fraction
                * cv_flux.regular_face_geometry.z_open_mask,
        )
    )

    if cv_flux.cut_wall_geometry is not None and cv_flux.cut_wall_flux is not None and cv_flux.cut_wall_flux.size:
        cut_wall_contrib = jnp.zeros(geometry.shape, dtype=jnp.float64)
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
    return result


def perp_laplacian_conservative_op(
    local: ConservativeStencil3D,
    geometry: FciGeometry3D,
    *,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    face_bc: BoundaryFaceBC3D | None = None,
    regular_face_geometry: RegularFaceGeometry3D | None = None,
    cut_wall_geometry: CutWallGeometry3D | None = None,
    cut_wall_bc: CutWallBC3D | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Build the conservative flux stencil and immediately apply divergence."""

    cv_flux = build_perp_laplacian_stencil(
        local,
        geometry,
        face_projectors=face_projectors,
        face_bc=face_bc,
        regular_face_geometry=regular_face_geometry,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        periodic_axes=periodic_axes,
        axis_regular_axes=axis_regular_axes,
        b_floor=b_floor,
    )
    return divergence_conservative_op(cv_flux, geometry, jacobian_floor=jacobian_floor)


def parallel_laplacian_conservative_op(
    local: ConservativeStencil3D,
    geometry: FciGeometry3D,
    *,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    face_bc: BoundaryFaceBC3D | None = None,
    regular_face_geometry: RegularFaceGeometry3D | None = None,
    cut_wall_geometry: CutWallGeometry3D | None = None,
    cut_wall_bc: CutWallBC3D | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the conservative parallel Laplacian ``∇·(b b·∇f)``."""

    if face_projectors is None:
        face_projectors = _build_laplacian_face_projectors(geometry, b_floor=b_floor, parallel=True)
    cv_flux = _build_projected_laplacian_stencil(
        local,
        geometry,
        face_projectors=face_projectors,
        face_bc=face_bc,
        regular_face_geometry=regular_face_geometry,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        periodic_axes=periodic_axes,
        b_floor=b_floor,
    )
    return divergence_conservative_op(cv_flux, geometry, jacobian_floor=jacobian_floor)


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
    regular_boundary_closure: (
        LocalRegularBoundaryMomentClosure3D | None
    ) = None,
) -> jnp.ndarray:
    """Apply local physical face-gradient closures on the owned face grid."""

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
            (lower_kind == BC_DIRICHLET) | (lower_kind == BC_NEUMANN)
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
        lower_plane = lower_plane.at[..., axis].set(
            jnp.where(lower_mask & (lower_kind == BC_NEUMANN), -lower_value, lower_plane[..., axis])
        )
        face_grad = face_grad.at[_axis_index_nd(axis, 0, face_grad.ndim)].set(lower_plane)

    upper_plane = face_grad[_axis_index_nd(axis, -1, face_grad.ndim)]
    upper_normal = upper_plane[..., axis]
    upper_tangent_mask = upper_mask & (
        (upper_kind == BC_DIRICHLET) | (upper_kind == BC_NEUMANN)
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
    upper_plane = upper_plane.at[..., axis].set(
        jnp.where(upper_mask & (upper_kind == BC_NEUMANN), upper_value, upper_plane[..., axis])
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


def _local_axis_face_values_from_stencil(
    stencil: LocalStencil1D,
    *,
    axis: int,
) -> jnp.ndarray:
    """Reconstruct scalar values onto owned control-volume faces."""

    center = jnp.asarray(stencil.center, dtype=jnp.float64)
    minus = jnp.asarray(stencil.minus, dtype=jnp.float64)
    plus = jnp.asarray(stencil.plus, dtype=jnp.float64)
    if center.ndim != 3:
        raise ValueError(f"stencil center must be 3D, got shape {center.shape}")

    lower = 0.5 * (
        center[_axis_index_nd(axis, 0, center.ndim)]
        + minus[_axis_index_nd(axis, 0, minus.ndim)]
    )
    upper_faces = 0.5 * (center + plus)
    return jnp.concatenate(
        (
            jnp.expand_dims(lower, axis=axis),
            upper_faces,
        ),
        axis=axis,
    )


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
    regular_face_geometry: LocalRegularFaceGeometry3D | None = None,
    regular_face_contribution_rows: LocalRegularFaceContributionRows3D | None = None,
    cell_volume: LocalCellVolumeGeometry3D | None = None,
    cut_wall_geometry: LocalCutWallGeometry3D | None = None,
    cut_wall_bc: LocalCutWallBC3D | None = None,
    cell_gradient: LocalCellGradient3D | None = None,
    aggregate_geometry: LocalAggregateCellGeometry3D | None = None,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D | None = None,
    boundary_bc: LocalControlVolumeBoundaryBC3D | None = None,
    field_reconstruction: LocalControlVolumePolynomial3D | None = None,
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
    if cut_wall_geometry is None and cut_wall_bc is None:
        cut_wall_geometry = LocalCutWallGeometry3D.empty(0)
        cut_wall_bc = LocalCutWallBC3D.empty(0)
    elif cut_wall_geometry is None:
        cut_wall_geometry = LocalCutWallGeometry3D.empty(cut_wall_bc.n_wall_faces)
    elif cut_wall_bc is None:
        cut_wall_bc = LocalCutWallBC3D.empty(cut_wall_geometry.max_wall_faces)

    x_face_value = _apply_local_face_value_dirichlet_bc(
        _local_axis_face_values_from_stencil(local.x, axis=0),
        axis=0,
        axis_kind=face_bc.kind_x,
        axis_value=face_bc.value_x,
        axis_mask=face_bc.mask_x,
        axis_regular_axes=axis_regular_axes,
    )
    y_face_value = _apply_local_face_value_dirichlet_bc(
        _local_axis_face_values_from_stencil(local.y, axis=1),
        axis=1,
        axis_kind=face_bc.kind_y,
        axis_value=face_bc.value_y,
        axis_mask=face_bc.mask_y,
        axis_regular_axes=axis_regular_axes,
    )
    z_face_value = _apply_local_face_value_dirichlet_bc(
        _local_axis_face_values_from_stencil(local.z, axis=2),
        axis=2,
        axis_kind=face_bc.kind_z,
        axis_value=face_bc.value_z,
        axis_mask=face_bc.mask_z,
        axis_regular_axes=axis_regular_axes,
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
        if boundary_bc is None or field_reconstruction is None:
            raise ValueError(
                "boundary_bc and field_reconstruction are required with "
                "control_volume_geometry"
            )
        irregular_flux = _local_control_volume_irregular_parallel_flux(
            jnp.asarray(local.x.center, dtype=jnp.float64),
            field_reconstruction,
            control_volume_geometry,
            boundary_bc,
            b_floor=b_floor,
        )
        return _local_control_volume_integrated_divergence(
            (x_flux, y_flux, z_flux),
            irregular_flux,
            geometry,
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


def _cell_least_squares_sample_shifts() -> tuple[tuple[int, int, int], ...]:
    """Static owned-cell sample offsets for local cell-centered LS gradients."""

    shifts: list[tuple[int, int, int]] = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                shifts.append((di, dj, dk))
    return tuple(shifts)


def _accumulate_cell_linear_sample(
    ata: jnp.ndarray,
    atb: jnp.ndarray,
    sample_count: jnp.ndarray,
    *,
    cell_positions: jnp.ndarray,
    sample_value: jnp.ndarray,
    sample_position: jnp.ndarray,
    sample_valid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Add one shifted cell sample to every owned-cell linear fit."""

    ones = jnp.ones_like(sample_value, dtype=jnp.float64)
    valid_weight = jnp.asarray(sample_valid, dtype=jnp.float64)
    sample_value = jnp.nan_to_num(
        jnp.asarray(sample_value, dtype=jnp.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    delta = jnp.asarray(sample_position, dtype=jnp.float64) - jnp.asarray(cell_positions, dtype=jnp.float64)
    row = jnp.concatenate((ones[..., None], delta), axis=-1)
    weighted_row = valid_weight[..., None] * row
    ata = ata + weighted_row[..., :, None] * row[..., None, :]
    atb = atb + weighted_row * sample_value[..., None]
    sample_count = sample_count + valid_weight
    return ata, atb, sample_count


def _accumulate_wall_linear_samples(
    ata: jnp.ndarray,
    atb: jnp.ndarray,
    sample_count: jnp.ndarray,
    *,
    geometry: LocalFciGeometry3D,
    cut_wall_geometry: LocalCutWallGeometry3D,
    cut_wall_bc: LocalCutWallBC3D | None,
    cut_wall_values: jnp.ndarray | None,
    target_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scatter Dirichlet wall samples into owner-cell linear fits."""

    if cut_wall_values is None or int(cut_wall_geometry.max_wall_faces) == 0:
        return ata, atb, sample_count

    values = jnp.asarray(cut_wall_values, dtype=jnp.float64)
    if values.shape != (int(cut_wall_geometry.max_wall_faces),):
        raise ValueError(
            "cut_wall_values must have one value per cut-wall row; "
            f"got {values.shape}, expected {(int(cut_wall_geometry.max_wall_faces),)}"
        )

    active = jnp.asarray(cut_wall_geometry.active, dtype=bool) & jnp.isfinite(values)
    if cut_wall_bc is not None:
        active = (
            active
            & jnp.asarray(cut_wall_bc.active, dtype=bool)
            & (jnp.asarray(cut_wall_bc.kind, dtype=jnp.int32) == BC_DIRICHLET)
        )

    owner_i = jnp.asarray(cut_wall_geometry.owner_i, dtype=jnp.int32)
    owner_j = jnp.asarray(cut_wall_geometry.owner_j, dtype=jnp.int32)
    owner_k = jnp.asarray(cut_wall_geometry.owner_k, dtype=jnp.int32)
    nx, ny, nz = geometry.owned_shape
    in_bounds = (
        (owner_i >= 0)
        & (owner_i < nx)
        & (owner_j >= 0)
        & (owner_j < ny)
        & (owner_k >= 0)
        & (owner_k < nz)
    )
    safe_i = jnp.clip(owner_i, 0, nx - 1)
    safe_j = jnp.clip(owner_j, 0, ny - 1)
    safe_k = jnp.clip(owner_k, 0, nz - 1)
    active = (
        active
        & in_bounds
        & _active_cell_mask_owned(geometry)[safe_i, safe_j, safe_k]
        & jnp.asarray(target_mask, dtype=bool)[safe_i, safe_j, safe_k]
    )

    cell_positions = _owned_cell_logical_positions(geometry)
    delta = jnp.asarray(cut_wall_geometry.center, dtype=jnp.float64) - cell_positions[safe_i, safe_j, safe_k]
    row = jnp.concatenate((jnp.ones((values.shape[0], 1), dtype=jnp.float64), delta), axis=-1)
    weight = active.astype(jnp.float64) * jnp.asarray(
        _GRADIENT_LS_BOUNDARY_SAMPLE_WEIGHT,
        dtype=jnp.float64,
    )
    weighted_row = weight[:, None] * row
    ata_update = weighted_row[..., :, None] * row[..., None, :]
    atb_update = weighted_row * jnp.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)[:, None]
    ata = ata.at[safe_i, safe_j, safe_k, :, :].add(ata_update)
    atb = atb.at[safe_i, safe_j, safe_k, :].add(atb_update)
    sample_count = sample_count.at[safe_i, safe_j, safe_k].add(weight)
    return ata, atb, sample_count


def _closed_regular_face_adjacent_cell_mask(
    geometry: LocalFciGeometry3D,
    regular_face_geometry: LocalRegularFaceGeometry3D,
) -> jnp.ndarray:
    """Mark cells adjacent to closed regular faces."""

    result = jnp.zeros(geometry.owned_shape, dtype=bool)
    face_data = (
        (
            0,
            regular_face_geometry.x_open_mask,
            regular_face_geometry.x_area_fraction,
        ),
        (
            1,
            regular_face_geometry.y_open_mask,
            regular_face_geometry.y_area_fraction,
        ),
        (
            2,
            regular_face_geometry.z_open_mask,
            regular_face_geometry.z_area_fraction,
        ),
    )
    for axis, open_mask, area_fraction in face_data:
        closed = (~jnp.asarray(open_mask, dtype=bool)) | (
            jnp.asarray(area_fraction, dtype=jnp.float64) <= 1.0e-12
        )
        n = int(geometry.owned_shape[axis])
        lower_faces = _axis_slice_nd(axis, 0, n, 3)
        upper_faces = _axis_slice_nd(axis, 1, n + 1, 3)
        result = result | closed[lower_faces] | closed[upper_faces]
    return result


def _partial_regular_face_adjacent_cell_mask(
    geometry: LocalFciGeometry3D,
    regular_face_geometry: LocalRegularFaceGeometry3D,
    *,
    face_fraction_tol: float = 1.0e-12,
) -> jnp.ndarray:
    """Mark cells adjacent to partially open regular faces."""

    result = jnp.zeros(geometry.owned_shape, dtype=bool)
    face_data = (
        (
            0,
            regular_face_geometry.x_open_mask,
            regular_face_geometry.x_area_fraction,
        ),
        (
            1,
            regular_face_geometry.y_open_mask,
            regular_face_geometry.y_area_fraction,
        ),
        (
            2,
            regular_face_geometry.z_open_mask,
            regular_face_geometry.z_area_fraction,
        ),
    )
    tol = float(face_fraction_tol)
    for axis, open_mask, area_fraction in face_data:
        fraction = jnp.asarray(area_fraction, dtype=jnp.float64)
        partial_face = (
            jnp.asarray(open_mask, dtype=bool)
            & (fraction > tol)
            & (fraction < 1.0 - tol)
        )
        n = int(geometry.owned_shape[axis])
        lower_faces = _axis_slice_nd(axis, 0, n, 3)
        upper_faces = _axis_slice_nd(axis, 1, n + 1, 3)
        result = result | partial_face[lower_faces] | partial_face[upper_faces]
    return result


def _regular_face_adjacent_component_mask(
    geometry: LocalFciGeometry3D,
    regular_face_geometry: LocalRegularFaceGeometry3D,
    *,
    mode: str,
    face_fraction_tol: float = 1.0e-12,
) -> jnp.ndarray:
    """Mark gradient components whose coordinate faces are closed or partial."""

    result = jnp.zeros(geometry.owned_shape + (3,), dtype=bool)
    face_data = (
        (
            0,
            regular_face_geometry.x_open_mask,
            regular_face_geometry.x_area_fraction,
        ),
        (
            1,
            regular_face_geometry.y_open_mask,
            regular_face_geometry.y_area_fraction,
        ),
        (
            2,
            regular_face_geometry.z_open_mask,
            regular_face_geometry.z_area_fraction,
        ),
    )
    tol = float(face_fraction_tol)
    for axis, open_mask, area_fraction in face_data:
        fraction = jnp.asarray(area_fraction, dtype=jnp.float64)
        if mode == "closed":
            face_mask = (~jnp.asarray(open_mask, dtype=bool)) | (fraction <= tol)
        elif mode == "partial":
            face_mask = (
                jnp.asarray(open_mask, dtype=bool)
                & (fraction > tol)
                & (fraction < 1.0 - tol)
            )
        else:
            raise ValueError(f"unknown regular-face component mode {mode!r}")
        n = int(geometry.owned_shape[axis])
        lower_faces = _axis_slice_nd(axis, 0, n, 3)
        upper_faces = _axis_slice_nd(axis, 1, n + 1, 3)
        component_mask = face_mask[lower_faces] | face_mask[upper_faces]
        result = result.at[..., axis].set(result[..., axis] | component_mask)
    return result


def _cut_wall_owner_count_and_min_distance_ratio(
    geometry: LocalFciGeometry3D,
    cut_wall_geometry: LocalCutWallGeometry3D,
    cut_wall_bc: LocalCutWallBC3D | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return active wall-row counts and smallest coordinate-distance ratio."""

    count = jnp.zeros(geometry.owned_shape, dtype=jnp.int32)
    min_ratio = jnp.full(geometry.owned_shape, jnp.inf, dtype=jnp.float64)
    if int(cut_wall_geometry.max_wall_faces) == 0:
        return count, min_ratio

    active = jnp.asarray(cut_wall_geometry.active, dtype=bool)
    if cut_wall_bc is not None:
        active = active & jnp.asarray(cut_wall_bc.active, dtype=bool)
    owner_i = jnp.asarray(cut_wall_geometry.owner_i, dtype=jnp.int32)
    owner_j = jnp.asarray(cut_wall_geometry.owner_j, dtype=jnp.int32)
    owner_k = jnp.asarray(cut_wall_geometry.owner_k, dtype=jnp.int32)
    nx, ny, nz = geometry.owned_shape
    in_bounds = (
        (owner_i >= 0)
        & (owner_i < nx)
        & (owner_j >= 0)
        & (owner_j < ny)
        & (owner_k >= 0)
        & (owner_k < nz)
    )
    safe_i = jnp.clip(owner_i, 0, nx - 1)
    safe_j = jnp.clip(owner_j, 0, ny - 1)
    safe_k = jnp.clip(owner_k, 0, nz - 1)
    active = active & in_bounds & _active_cell_mask_owned(geometry)[safe_i, safe_j, safe_k]
    count = count.at[safe_i, safe_j, safe_k].add(active.astype(jnp.int32))

    stencil_axis = jnp.asarray(cut_wall_geometry.stencil_axis, dtype=jnp.int32)
    wall_axis = jnp.asarray(
        getattr(cut_wall_geometry, "wall_axis", stencil_axis),
        dtype=jnp.int32,
    )
    axis = jnp.where(stencil_axis >= 0, stencil_axis, wall_axis)
    dx = jnp.asarray(geometry.spacing.dx_owned, dtype=jnp.float64)[safe_i, safe_j, safe_k]
    dy = jnp.asarray(geometry.spacing.dy_owned, dtype=jnp.float64)[safe_i, safe_j, safe_k]
    dz = jnp.asarray(geometry.spacing.dz_owned, dtype=jnp.float64)[safe_i, safe_j, safe_k]
    spacing = jnp.where(axis == 0, dx, jnp.where(axis == 1, dy, dz))
    ratio = jnp.asarray(cut_wall_geometry.stencil_distance, dtype=jnp.float64) / jnp.maximum(spacing, 1.0e-30)
    ratio_active = active & (axis >= 0) & jnp.isfinite(ratio) & (ratio > 0.0)
    row_ratio = jnp.where(ratio_active, ratio, jnp.inf)
    min_ratio = min_ratio.at[safe_i, safe_j, safe_k].min(row_ratio)
    return count, min_ratio


def _coordinate_stencil_neighbor_mask(mask: jnp.ndarray) -> jnp.ndarray:
    """Mark cells whose coordinate-gradient stencil can sample a mask.

    The local coordinate-gradient path samples only the axial ``+/-1``
    neighbors of a cell.  Keep the LS guard band to those cells; diagonal
    one-cell dilation is too broad and can replace a clean centered stencil
    with an asymmetric repair LS cloud.
    """

    source = jnp.asarray(mask, dtype=bool)
    result = jnp.zeros_like(source, dtype=bool)
    for axis in range(3):
        n = int(source.shape[axis])
        for offset in (-1, 1):
            src_slices: list[object] = [slice(None), slice(None), slice(None)]
            dst_slices: list[object] = [slice(None), slice(None), slice(None)]
            if offset > 0:
                src_slices[axis] = slice(0, n - offset)
                dst_slices[axis] = slice(offset, n)
            else:
                src_slices[axis] = slice(-offset, n)
                dst_slices[axis] = slice(0, n + offset)
            src = tuple(src_slices)
            dst = tuple(dst_slices)
            result = result.at[dst].set(result[dst] | source[src])
    return result


def _coordinate_stencil_neighbor_component_mask(mask: jnp.ndarray) -> jnp.ndarray:
    """Mark each component whose centered coordinate stencil samples a mask."""

    source = jnp.asarray(mask, dtype=bool)
    result = jnp.zeros(source.shape + (3,), dtype=bool)
    for axis in range(3):
        component_mask = jnp.zeros_like(source, dtype=bool)
        n = int(source.shape[axis])
        for offset in (-1, 1):
            src_slices: list[object] = [slice(None), slice(None), slice(None)]
            dst_slices: list[object] = [slice(None), slice(None), slice(None)]
            if offset > 0:
                src_slices[axis] = slice(0, n - offset)
                dst_slices[axis] = slice(offset, n)
            else:
                src_slices[axis] = slice(-offset, n)
                dst_slices[axis] = slice(0, n + offset)
            src = tuple(src_slices)
            dst = tuple(dst_slices)
            component_mask = component_mask.at[dst].set(component_mask[dst] | source[src])
        result = result.at[..., axis].set(component_mask)
    return result


def _coordinate_stencil_inactive_neighbor_component_mask(
    geometry: LocalFciGeometry3D,
) -> jnp.ndarray:
    """Mark components whose centered coordinate stencil samples inactive cells."""

    active = _active_cell_mask_owned(geometry)
    inactive = ~active
    result = jnp.zeros(active.shape + (3,), dtype=bool)
    for axis in range(3):
        component_mask = jnp.zeros_like(active, dtype=bool)
        n = int(active.shape[axis])
        for offset in (-1, 1):
            src_slices: list[object] = [slice(None), slice(None), slice(None)]
            dst_slices: list[object] = [slice(None), slice(None), slice(None)]
            if offset > 0:
                src_slices[axis] = slice(0, n - offset)
                dst_slices[axis] = slice(offset, n)
            else:
                src_slices[axis] = slice(-offset, n)
                dst_slices[axis] = slice(0, n + offset)
            src = tuple(src_slices)
            dst = tuple(dst_slices)
            component_mask = component_mask.at[dst].set(
                component_mask[dst] | inactive[src]
            )
        result = result.at[..., axis].set(component_mask)
    return result


def _cut_wall_owner_component_mask(
    geometry: LocalFciGeometry3D,
    cut_wall_geometry: LocalCutWallGeometry3D | None,
    cut_wall_bc: LocalCutWallBC3D | None,
) -> jnp.ndarray:
    """Mark wall-normal coordinate components for active cut-wall owner cells."""

    result = jnp.zeros(geometry.owned_shape + (3,), dtype=jnp.int32)
    if cut_wall_geometry is None or int(cut_wall_geometry.max_wall_faces) == 0:
        return result.astype(bool)

    active = jnp.asarray(cut_wall_geometry.active, dtype=bool)
    if cut_wall_bc is not None:
        active = active & jnp.asarray(cut_wall_bc.active, dtype=bool)

    owner_i = jnp.asarray(cut_wall_geometry.owner_i, dtype=jnp.int32)
    owner_j = jnp.asarray(cut_wall_geometry.owner_j, dtype=jnp.int32)
    owner_k = jnp.asarray(cut_wall_geometry.owner_k, dtype=jnp.int32)
    nx, ny, nz = geometry.owned_shape
    in_bounds = (
        (owner_i >= 0)
        & (owner_i < nx)
        & (owner_j >= 0)
        & (owner_j < ny)
        & (owner_k >= 0)
        & (owner_k < nz)
    )
    safe_i = jnp.clip(owner_i, 0, nx - 1)
    safe_j = jnp.clip(owner_j, 0, ny - 1)
    safe_k = jnp.clip(owner_k, 0, nz - 1)
    active = active & in_bounds & _active_cell_mask_owned(geometry)[safe_i, safe_j, safe_k]

    stencil_axis = jnp.asarray(cut_wall_geometry.stencil_axis, dtype=jnp.int32)
    wall_axis = jnp.asarray(
        getattr(cut_wall_geometry, "wall_axis", stencil_axis),
        dtype=jnp.int32,
    )
    axis = jnp.where(stencil_axis >= 0, stencil_axis, wall_axis)
    component = jnp.clip(axis, 0, 2)
    weight = (active & (axis >= 0) & (axis < 3)).astype(jnp.int32)
    result = result.at[safe_i, safe_j, safe_k, component].add(weight)
    return result > 0


def _cut_wall_owner_stencil_component_mask(
    geometry: LocalFciGeometry3D,
    cut_wall_geometry: LocalCutWallGeometry3D | None,
    cut_wall_bc: LocalCutWallBC3D | None,
) -> jnp.ndarray:
    """Mark owner components with explicit cut-wall replacement stencil axes."""

    result = jnp.zeros(geometry.owned_shape + (3,), dtype=jnp.int32)
    if cut_wall_geometry is None or int(cut_wall_geometry.max_wall_faces) == 0:
        return result.astype(bool)

    active = jnp.asarray(cut_wall_geometry.active, dtype=bool)
    if cut_wall_bc is not None:
        active = active & jnp.asarray(cut_wall_bc.active, dtype=bool)

    owner_i = jnp.asarray(cut_wall_geometry.owner_i, dtype=jnp.int32)
    owner_j = jnp.asarray(cut_wall_geometry.owner_j, dtype=jnp.int32)
    owner_k = jnp.asarray(cut_wall_geometry.owner_k, dtype=jnp.int32)
    nx, ny, nz = geometry.owned_shape
    in_bounds = (
        (owner_i >= 0)
        & (owner_i < nx)
        & (owner_j >= 0)
        & (owner_j < ny)
        & (owner_k >= 0)
        & (owner_k < nz)
    )
    safe_i = jnp.clip(owner_i, 0, nx - 1)
    safe_j = jnp.clip(owner_j, 0, ny - 1)
    safe_k = jnp.clip(owner_k, 0, nz - 1)
    active = active & in_bounds & _active_cell_mask_owned(geometry)[safe_i, safe_j, safe_k]

    stencil_axis = jnp.asarray(cut_wall_geometry.stencil_axis, dtype=jnp.int32)
    component = jnp.clip(stencil_axis, 0, 2)
    weight = (active & (stencil_axis >= 0) & (stencil_axis < 3)).astype(jnp.int32)
    result = result.at[safe_i, safe_j, safe_k, component].add(weight)
    return result > 0


def _local_gradient_ls_component_repair_mask(
    geometry: LocalFciGeometry3D,
    regular_face_geometry: LocalRegularFaceGeometry3D,
    *,
    cut_wall_geometry: LocalCutWallGeometry3D | None,
    cut_wall_bc: LocalCutWallBC3D | None,
    small_distance_ratio: float,
    agglomerated_volume_fraction_tol: float = 1.0e-12,
) -> jnp.ndarray:
    """Return component-wise LS replacement mask for owned-cell gradients."""

    active = _active_cell_mask_owned(geometry)
    agglomerated = (
        jnp.asarray(geometry.cell_volume_geometry.volume_fraction, dtype=jnp.float64)
        > 1.0 + float(agglomerated_volume_fraction_tol)
    )
    closed_component = _regular_face_adjacent_component_mask(
        geometry,
        regular_face_geometry,
        mode="closed",
    )
    partial_component = _regular_face_adjacent_component_mask(
        geometry,
        regular_face_geometry,
        mode="partial",
    )
    closed_face = jnp.any(closed_component, axis=-1)
    partial_face = jnp.any(partial_component, axis=-1)

    exact_one_wall = jnp.zeros(geometry.owned_shape, dtype=bool)
    multi_wall = jnp.zeros(geometry.owned_shape, dtype=bool)
    small_distance = jnp.zeros(geometry.owned_shape, dtype=bool)
    one_wall = jnp.zeros(geometry.owned_shape, dtype=bool)
    wall_component = jnp.zeros(geometry.owned_shape + (3,), dtype=bool)
    wall_stencil_component = jnp.zeros(geometry.owned_shape + (3,), dtype=bool)
    if cut_wall_geometry is not None:
        owner_count, min_ratio = _cut_wall_owner_count_and_min_distance_ratio(
            geometry,
            cut_wall_geometry,
            cut_wall_bc,
        )
        exact_one_wall = owner_count == 1
        one_wall = owner_count > 0
        multi_wall = owner_count >= 2
        small_distance = min_ratio < float(small_distance_ratio)
        wall_component = _cut_wall_owner_component_mask(
            geometry,
            cut_wall_geometry,
            cut_wall_bc,
        )
        wall_stencil_component = _cut_wall_owner_stencil_component_mask(
            geometry,
            cut_wall_geometry,
            cut_wall_bc,
        )

    irregular_source_mask = (
        agglomerated
        | closed_face
        | partial_face
        | multi_wall
        | small_distance
        | one_wall
    )
    direct_or_owner = (
        agglomerated | multi_wall | small_distance | closed_face | partial_face | one_wall
    )
    guard_component = _coordinate_stencil_neighbor_component_mask(irregular_source_mask) & (
        ~direct_or_owner[..., None]
    )
    unsafe_coordinate_component = (
        _coordinate_stencil_inactive_neighbor_component_mask(geometry)
        | wall_stencil_component
    )
    # Owning a cut-wall plane is not enough to repair clean tangential
    # derivatives.  For one-wall owners, use the boundary-constrained LS only
    # on the wall axis and coordinate legs that can actually sample bad data.
    wall_repair_component = wall_component & unsafe_coordinate_component
    one_wall_agglomerated_component = (
        (agglomerated & exact_one_wall)[..., None]
        & (wall_component | unsafe_coordinate_component)
    )
    all_component_repair = multi_wall | small_distance | (agglomerated & ~exact_one_wall)
    repair = (
        all_component_repair[..., None]
        | closed_component
        | partial_component
        | one_wall_agglomerated_component
        | (wall_repair_component & one_wall[..., None])
        | guard_component
    )
    return repair & active[..., None]


def precompute_local_physical_boundary_gradient_reconstruction(
    cells: LocalControlVolumeCellGeometry3D,
    irregular_faces: LocalControlVolumeFaceRows3D,
    *,
    spacing_owned: jnp.ndarray,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
    coordinate_periods: tuple[float, float, float] | None = None,
    max_equations: int = 48,
    svd_rcond: float = 1.0e-12,
    condition_limit: float = 1.0e8,
) -> LocalPhysicalBoundaryGradientReconstruction3D:
    """Precompute cubic physical-boundary gradient functionals on the host.

    The polynomial is anchored to the target control-volume average and uses
    all normalized monomials through degree three.  Cell equations include
    finite-volume second moments.  Third central moments are omitted; for the
    regular physical-boundary cells targeted here their contribution is
    fourth order because the centroid is J weighted.
    """

    if not isinstance(cells, LocalControlVolumeCellGeometry3D):
        raise TypeError("cells must be LocalControlVolumeCellGeometry3D")
    if not isinstance(irregular_faces, LocalControlVolumeFaceRows3D):
        raise TypeError(
            "irregular_faces must be LocalControlVolumeFaceRows3D"
        )
    if cells.layout != irregular_faces.layout:
        raise ValueError("cells and irregular_faces must share one layout")
    max_equations = int(max_equations)
    if max_equations < 19:
        raise ValueError("max_equations must be at least 19")

    active_owner = np.asarray(cells.is_active_owner, dtype=bool)
    centroid = np.asarray(cells.centroid, dtype=np.float64)
    second_moment = np.asarray(cells.second_moment, dtype=np.float64)
    spacing = np.asarray(spacing_owned, dtype=np.float64)
    if spacing.shape != cells.shape + (3,):
        raise ValueError(
            f"spacing_owned must have shape {cells.shape + (3,)}, "
            f"got {spacing.shape}"
        )
    spacing = np.maximum(np.abs(spacing), 1.0e-14)
    face_active = np.asarray(irregular_faces.active, dtype=bool)
    face_kind = np.asarray(irregular_faces.kind, dtype=np.int32)
    minus_owner = np.stack(
        (
            np.asarray(irregular_faces.minus_owner_i, dtype=np.int32),
            np.asarray(irregular_faces.minus_owner_j, dtype=np.int32),
            np.asarray(irregular_faces.minus_owner_k, dtype=np.int32),
        ),
        axis=-1,
    )
    boundary_axis = np.asarray(
        irregular_faces.boundary_normal_axis,
        dtype=np.int32,
    )
    points = np.asarray(
        irregular_faces.quadrature_points,
        dtype=np.float64,
    )
    quadrature_active = np.asarray(
        irregular_faces.quadrature_active,
        dtype=bool,
    )
    periodic_axes = tuple(bool(value) for value in periodic_axes)
    periods = np.asarray(
        (
            (1.0, 1.0, 1.0)
            if coordinate_periods is None
            else coordinate_periods
        ),
        dtype=np.float64,
    )

    def unwrap(displacement: np.ndarray) -> np.ndarray:
        result = np.asarray(displacement, dtype=np.float64).copy()
        for axis in range(3):
            if periodic_axes[axis]:
                result[..., axis] -= (
                    np.round(result[..., axis] / periods[axis])
                    * periods[axis]
                )
        return result

    def point_features(s: np.ndarray) -> np.ndarray:
        x, y, z = (float(value) for value in s)
        return np.asarray(
            (
                x,
                y,
                z,
                x * x,
                y * y,
                z * z,
                x * y,
                x * z,
                y * z,
                x**3,
                y**3,
                z**3,
                x * x * y,
                x * x * z,
                y * y * x,
                y * y * z,
                z * z * x,
                z * z * y,
                x * y * z,
            ),
            dtype=np.float64,
        )

    def average_features(
        displacement: np.ndarray,
        moment: np.ndarray,
        target_moment: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        d = displacement / scale
        M = moment / (scale[:, None] * scale[None, :])
        Mi = target_moment / (scale[:, None] * scale[None, :])
        x, y, z = d
        result = point_features(d)
        result[3] += M[0, 0] - Mi[0, 0]
        result[4] += M[1, 1] - Mi[1, 1]
        result[5] += M[2, 2] - Mi[2, 2]
        result[6] += M[0, 1] - Mi[0, 1]
        result[7] += M[0, 2] - Mi[0, 2]
        result[8] += M[1, 2] - Mi[1, 2]
        result[9] += 3.0 * x * M[0, 0]
        result[10] += 3.0 * y * M[1, 1]
        result[11] += 3.0 * z * M[2, 2]
        result[12] += y * M[0, 0] + 2.0 * x * M[0, 1]
        result[13] += z * M[0, 0] + 2.0 * x * M[0, 2]
        result[14] += x * M[1, 1] + 2.0 * y * M[0, 1]
        result[15] += z * M[1, 1] + 2.0 * y * M[1, 2]
        result[16] += x * M[2, 2] + 2.0 * z * M[0, 2]
        result[17] += y * M[2, 2] + 2.0 * z * M[1, 2]
        result[18] += x * M[1, 2] + y * M[0, 2] + z * M[0, 1]
        return result

    def point_gradient_matrix(s: np.ndarray, scale: np.ndarray) -> np.ndarray:
        x, y, z = (float(value) for value in s)
        derivative = np.zeros((3, 19), dtype=np.float64)
        derivative[0] = (
            1, 0, 0, 2*x, 0, 0, y, z, 0,
            3*x*x, 0, 0, 2*x*y, 2*x*z, y*y, 0, z*z, 0, y*z,
        )
        derivative[1] = (
            0, 1, 0, 0, 2*y, 0, x, 0, z,
            0, 3*y*y, 0, x*x, 0, 2*x*y, 2*y*z, 0, z*z, x*z,
        )
        derivative[2] = (
            0, 0, 1, 0, 0, 2*z, 0, x, y,
            0, 0, 3*z*z, 0, x*x, 0, y*y, 2*x*z, 2*y*z, x*y,
        )
        return derivative / scale[:, None]

    max_rows = int(irregular_faces.max_rows)
    equation_shape = (max_rows, max_equations)
    equation_kind = np.zeros(equation_shape, dtype=np.int32)
    sample_i = np.zeros(equation_shape, dtype=np.int32)
    sample_j = np.zeros(equation_shape, dtype=np.int32)
    sample_k = np.zeros(equation_shape, dtype=np.int32)
    boundary_face_row = np.zeros(equation_shape, dtype=np.int32)
    boundary_patch = np.zeros(equation_shape, dtype=np.int32)
    boundary_quadrature = np.zeros(equation_shape, dtype=np.int32)
    equation_active = np.zeros(equation_shape, dtype=bool)
    transform = np.zeros(
        (max_rows, 4, 3, max_equations),
        dtype=np.float64,
    )
    valid = np.zeros((max_rows,), dtype=bool)
    condition_number = np.full((max_rows,), np.inf, dtype=np.float64)
    physical_rows = np.flatnonzero(
        face_active & (face_kind == CV_FACE_PHYSICAL_BOUNDARY)
    )
    shape = np.asarray(cells.shape, dtype=np.int32)
    physical_row_lookup: dict[
        tuple[int, float, tuple[int, int, int]],
        int,
    ] = {}
    for physical_row in physical_rows:
        physical_axis = int(boundary_axis[physical_row])
        physical_coordinate = round(
            float(points[physical_row, 0, 0, physical_axis]),
            12,
        )
        physical_owner = tuple(
            int(value) for value in minus_owner[physical_row]
        )
        physical_row_lookup[
            (physical_axis, physical_coordinate, physical_owner)
        ] = int(physical_row)

    for row in physical_rows:
        owner = tuple(int(value) for value in minus_owner[row])
        axis = int(boundary_axis[row])
        target_position = centroid[owner]
        target_moment = second_moment[owner]
        target_scale = spacing[owner]
        face_coordinate = float(points[row, 0, 0, axis])

        cell_records: list[
            tuple[float, tuple[int, int, int], np.ndarray]
        ] = []
        for di in range(-2, 3):
            for dj in range(-2, 3):
                for dk in range(-2, 3):
                    candidate_list = [
                        owner[0] + di,
                        owner[1] + dj,
                        owner[2] + dk,
                    ]
                    in_bounds = True
                    for candidate_axis in range(3):
                        if periodic_axes[candidate_axis]:
                            candidate_list[candidate_axis] %= int(
                                shape[candidate_axis]
                            )
                        elif not (
                            0
                            <= candidate_list[candidate_axis]
                            < int(shape[candidate_axis])
                        ):
                            in_bounds = False
                            break
                    if not in_bounds:
                        continue
                    candidate = tuple(candidate_list)
                    if candidate == owner or not active_owner[candidate]:
                        continue
                    displacement = unwrap(
                        centroid[candidate] - target_position
                    )
                    distance = float(
                        np.linalg.norm(displacement / target_scale)
                    )
                    cell_records.append(
                        (
                            distance,
                            candidate,
                            average_features(
                                displacement,
                                second_moment[candidate],
                                target_moment,
                                target_scale,
                            ),
                        )
                    )
        cell_records.sort(
            key=lambda item: (
                abs(item[1][axis] - owner[axis]),
                item[0],
                item[1],
            )
        )
        selected_cells: list[
            tuple[float, tuple[int, int, int], np.ndarray]
        ] = []
        for normal_depth in range(3):
            layer = [
                record
                for record in cell_records
                if abs(record[1][axis] - owner[axis]) == normal_depth
            ]
            selected_cells.extend(layer[:10])

        boundary_records: list[
            tuple[float, int, int, int, np.ndarray]
        ] = []
        tangential_axes = tuple(
            candidate_axis
            for candidate_axis in range(3)
            if candidate_axis != axis
        )
        nearby_boundary_rows: set[int] = set()
        for first_offset in range(-2, 3):
            for second_offset in range(-2, 3):
                candidate_owner = list(owner)
                candidate_owner[tangential_axes[0]] += first_offset
                candidate_owner[tangential_axes[1]] += second_offset
                in_bounds = True
                for tangential_axis in tangential_axes:
                    if periodic_axes[tangential_axis]:
                        candidate_owner[tangential_axis] %= int(
                            shape[tangential_axis]
                        )
                    elif not (
                        0
                        <= candidate_owner[tangential_axis]
                        < int(shape[tangential_axis])
                    ):
                        in_bounds = False
                        break
                if not in_bounds:
                    continue
                other_row = physical_row_lookup.get(
                    (
                        axis,
                        round(face_coordinate, 12),
                        tuple(candidate_owner),
                    )
                )
                if other_row is not None:
                    nearby_boundary_rows.add(other_row)
        for other_row in sorted(nearby_boundary_rows):
            for patch in range(int(irregular_faces.max_patches)):
                for quadrature in range(4):
                    if not quadrature_active[
                        other_row,
                        patch,
                        quadrature,
                    ]:
                        continue
                    displacement = unwrap(
                        points[other_row, patch, quadrature]
                        - target_position
                    )
                    scaled = displacement / target_scale
                    boundary_records.append(
                        (
                            float(np.linalg.norm(scaled)),
                            int(other_row),
                            int(patch),
                            int(quadrature),
                            point_features(scaled)
                            + average_features(
                                np.zeros((3,), dtype=np.float64),
                                np.zeros((3, 3), dtype=np.float64),
                                target_moment,
                                target_scale,
                            ),
                        )
                    )
        boundary_records.sort(key=lambda item: item[:4])
        selected_boundary = boundary_records[:18]
        records = [
            (
                CV_RECONSTRUCTION_EQUATION_CELL,
                candidate,
                feature,
                distance,
            )
            for distance, candidate, feature in selected_cells
        ] + [
            (
                CV_RECONSTRUCTION_EQUATION_DIRICHLET,
                (face_row, patch, quadrature),
                feature,
                distance,
            )
            for distance, face_row, patch, quadrature, feature
            in selected_boundary
        ]
        records = records[:max_equations]
        if len(records) < 19:
            continue
        design = np.asarray([record[2] for record in records])
        weights = np.asarray(
            [1.0 / max(record[3] ** 2, 1.0e-8) for record in records],
            dtype=np.float64,
        )
        sqrt_weight = np.sqrt(weights)
        weighted_design = sqrt_weight[:, None] * design
        singular = np.linalg.svd(weighted_design, compute_uv=False)
        if not singular.size or singular[0] <= 0.0:
            continue
        tolerance = float(svd_rcond) * singular[0]
        rank = int(np.sum(singular > tolerance))
        condition = (
            float(singular[0] / singular[-1])
            if rank == 19 and singular[-1] > tolerance
            else np.inf
        )
        if rank < 19 or condition > float(condition_limit):
            continue
        inverse = (
            np.linalg.pinv(weighted_design, rcond=float(svd_rcond))
            * sqrt_weight[None, :]
        )
        for equation, (kind, payload, _feature, _distance) in enumerate(
            records
        ):
            equation_kind[row, equation] = int(kind)
            equation_active[row, equation] = True
            if kind == CV_RECONSTRUCTION_EQUATION_CELL:
                sample_i[row, equation] = int(payload[0])
                sample_j[row, equation] = int(payload[1])
                sample_k[row, equation] = int(payload[2])
            else:
                boundary_face_row[row, equation] = int(payload[0])
                boundary_patch[row, equation] = int(payload[1])
                boundary_quadrature[row, equation] = int(payload[2])
        for quadrature in range(4):
            point = points[row, 0, quadrature]
            scaled_point = unwrap(point - target_position) / target_scale
            transform[row, quadrature, :, : len(records)] = (
                point_gradient_matrix(scaled_point, target_scale)
                @ inverse
            )
        valid[row] = True
        condition_number[row] = condition

    return LocalPhysicalBoundaryGradientReconstruction3D(
        layout=cells.layout,
        equation_kind=jnp.asarray(equation_kind),
        sample_i=jnp.asarray(sample_i),
        sample_j=jnp.asarray(sample_j),
        sample_k=jnp.asarray(sample_k),
        boundary_face_row=jnp.asarray(boundary_face_row),
        boundary_patch=jnp.asarray(boundary_patch),
        boundary_quadrature=jnp.asarray(boundary_quadrature),
        equation_active=jnp.asarray(equation_active),
        gradient_transform=jnp.asarray(transform),
        active=jnp.asarray(valid),
        condition_number=jnp.asarray(condition_number),
        max_rows=max_rows,
        max_equations=max_equations,
    )


def precompute_local_quadratic_reconstruction(
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
) -> LocalQuadraticReconstruction3D:
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
        return LocalQuadraticReconstruction3D.empty(
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
    return LocalQuadraticReconstruction3D(
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


def precompute_local_cubic_reconstruction(
    cells: LocalControlVolumeCellGeometry3D,
    irregular_faces: LocalControlVolumeFaceRows3D,
    *,
    spacing_owned: jnp.ndarray,
    remote_sample_halo_indices: np.ndarray | None = None,
    remote_sample_centroids: np.ndarray | None = None,
    remote_sample_second_moments: np.ndarray | None = None,
    remote_sample_third_moments: np.ndarray | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, False, False),
    coordinate_periods: tuple[float, float, float] | None = None,
    target_mask: jnp.ndarray | None = None,
    max_samples: int = 48,
    max_equations: int = 64,
    condition_limit: float = 1.0e6,
    svd_rcond: float = 1.0e-12,
) -> LocalQuadraticReconstruction3D:
    """Precompute 19-coefficient cubic finite-volume reconstruction rows.

    This deliberately lives beside the quadratic builder during migration.
    Remote samples are compact halo references, so aggregate ownership stays
    local while the fitted stencil remains decomposition-compatible.
    """
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
    targets = np.argwhere(requested & active)
    n_rows = len(targets)
    if n_rows == 0:
        return LocalQuadraticReconstruction3D.empty(
            cells.layout, max_rows=0, max_equations=max_equations
        )
    periods = np.asarray(
        coordinate_periods if coordinate_periods is not None else (1.0, 1.0, 1.0),
        dtype=np.float64,
    )
    periodic_axes = tuple(bool(value) for value in periodic_axes)
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
        for axis, is_periodic in enumerate(periodic_axes):
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
    scale_power = np.asarray((
        1/spacing[..., 0].flat[0], 1/spacing[..., 1].flat[0], 1/spacing[..., 2].flat[0],
    ))  # Per-row scale is computed below.
    for r, target_array in enumerate(targets):
        target = tuple(int(x) for x in target_array); target_i[r], target_j[r], target_k[r] = target; row_for_cell[target] = r
        x0, m20, m30, h = centroid[target], m2[target], m3[target], spacing[target]
        records: list[tuple[int, float, np.ndarray, tuple[int, tuple[int, int, int]]]] = []
        for di in range(-3, 4):
            for dj in range(-3, 4):
                for dk in range(-3, 4):
                    if di == dj == dk == 0: continue
                    raw = (target[0]+di, target[1]+dj, target[2]+dk); candidate = list(raw); okay = True
                    for axis in range(3):
                        if periodic_axes[axis]: candidate[axis] %= shape[axis]
                        elif not (0 <= candidate[axis] < shape[axis]): okay = False; break
                    if not okay or not active[tuple(candidate)]: continue
                    candidate_t = tuple(candidate); delta = unwrap(centroid[candidate_t] - x0)
                    dm2 = m2[candidate_t] + np.outer(delta, delta) - m20
                    dm3 = translated_m3(delta, m2[candidate_t], m3[candidate_t]) - m30
                    d2 = float(np.dot(delta/h, delta/h)); shell = max(abs(di), abs(dj), abs(dk))
                    records.append((shell, d2, cubic_row(delta, dm2, dm3, h), (CV_RECONSTRUCTION_EQUATION_CELL, candidate_t)))
        for remote_index, (remote_position, remote_second, remote_third) in enumerate(
            zip(remote_centroids, remote_m2, remote_m3)
        ):
            delta = unwrap(remote_position - x0)
            d2 = float(np.dot(delta / h, delta / h))
            if d2 > 27.0 + 1.0e-12:
                continue
            dm2 = remote_second + np.outer(delta, delta) - m20
            dm3 = translated_m3(delta, remote_second, remote_third) - m30
            records.append((
                3,
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
        design = np.stack([item[2] for item in selected]); weights = np.asarray([1.0/max(item[1], 1e-12) for item in selected])
        # Each wall face shares one normalized distance weight.
        for idx, item in enumerate(selected):
            if item[3][0] == CV_RECONSTRUCTION_EQUATION_DIRICHLET:
                fr, patch, quad = item[3][1]; weights[idx] *= measure[fr, patch, quad] / max(float(np.sum(np.where(qactive[fr], measure[fr], 0.0))), 1e-30)
        weighted = np.sqrt(weights)[:, None] * design; singular = np.linalg.svd(weighted, compute_uv=False)
        tolerance = svd_rcond * singular[0] if singular.size else np.inf; full_rank = int(np.sum(singular > tolerance)); cond = float(singular[0]/singular[-1]) if full_rank >= 19 else np.inf
        if full_rank < 19 or cond > condition_limit: continue
        scale = np.asarray((
            1/h[0], 1/h[1], 1/h[2], 1/h[0]**2, 1/h[1]**2, 1/h[2]**2, 1/(h[0]*h[1]), 1/(h[0]*h[2]), 1/(h[1]*h[2]),
            1/h[0]**3, 1/h[1]**3, 1/h[2]**3, 1/(h[0]**2*h[1]), 1/(h[0]**2*h[2]), 1/(h[0]*h[1]**2), 1/(h[0]*h[2]**2), 1/(h[1]**2*h[2]), 1/(h[1]*h[2]**2), 1/(h[0]*h[1]*h[2]),
        ))
        transform_out[r, :, :len(selected)] = scale[:, None] * np.linalg.pinv(weighted, rcond=svd_rcond) * np.sqrt(weights)[None, :]
        order[r] = 3; rank[r] = full_rank; condition[r] = cond; equation_active[r, :len(selected)] = True
        for e, (_, _, _, (kind, payload)) in enumerate(selected):
            equation_kind[r, e] = kind
            if kind in (CV_RECONSTRUCTION_EQUATION_CELL, CV_RECONSTRUCTION_EQUATION_REMOTE_CELL):
                sample_i[r, e], sample_j[r, e], sample_k[r, e] = payload
            else:
                boundary_face_row[r, e], boundary_patch[r, e], boundary_quadrature[r, e] = payload
    return LocalQuadraticReconstruction3D(layout=cells.layout, target_i=jnp.asarray(target_i), target_j=jnp.asarray(target_j), target_k=jnp.asarray(target_k), equation_kind=jnp.asarray(equation_kind), sample_i=jnp.asarray(sample_i), sample_j=jnp.asarray(sample_j), sample_k=jnp.asarray(sample_k), boundary_face_row=jnp.asarray(boundary_face_row), boundary_patch=jnp.asarray(boundary_patch), boundary_quadrature=jnp.asarray(boundary_quadrature), equation_active=jnp.asarray(equation_active), rhs_transform=jnp.asarray(transform_out), active=jnp.asarray(order > 0), target_row_for_cell=jnp.asarray(row_for_cell), polynomial_order=jnp.asarray(order), rank=jnp.asarray(rank), condition_number=jnp.asarray(condition), max_rows=n_rows, max_equations=max_equations)


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
    return jnp.where(cells.raw_volume > 0.0, expanded, 0.0)


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


def _evaluate_local_regular_transition_functional(
    values_owned: jnp.ndarray,
    polynomial: LocalControlVolumePolynomial3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate dense-compatible compact full-face functionals.

    Transition rows replace only faces whose complete dense support touches a
    nonregular control volume.  Direct samples retain the stored regular cell
    average; virtual samples are the quadratic owner polynomial integrated
    over the logical regular cell represented by its precomputed moments.
    """

    transition = control_volume_geometry.regular_transition_faces
    cells = control_volume_geometry.cells
    n_faces = int(control_volume_geometry.irregular_faces.max_rows)
    # Minimal/legacy test fixtures may not provide transition metadata.  They
    # retain the irregular-row path rather than broadcasting an empty payload
    # against nonempty face rows.
    if int(transition.max_rows) != n_faces:
        return (
            jnp.zeros((n_faces,), dtype=jnp.float64),
            jnp.zeros((n_faces, 3), dtype=jnp.float64),
            jnp.zeros((n_faces,), dtype=bool),
        )
    storage_value = values_owned[
        transition.sample_storage_i,
        transition.sample_storage_j,
        transition.sample_storage_k,
    ]
    owner_value = values_owned[
        transition.sample_owner_i,
        transition.sample_owner_j,
        transition.sample_owner_k,
    ]
    owner_gradient = polynomial.gradient[
        transition.sample_owner_i,
        transition.sample_owner_j,
        transition.sample_owner_k,
    ]
    owner_hessian = polynomial.hessian[
        transition.sample_owner_i,
        transition.sample_owner_j,
        transition.sample_owner_k,
    ]
    owner_third = polynomial.third_derivative[
        transition.sample_owner_i,
        transition.sample_owner_j,
        transition.sample_owner_k,
    ]
    owner_valid = polynomial.valid[
        transition.sample_owner_i,
        transition.sample_owner_j,
        transition.sample_owner_k,
    ]
    virtual_value = (
        owner_value
        + jnp.einsum(
            "rsi,rsi->rs",
            owner_gradient,
            transition.sample_displacement,
        )
        + 0.5
        * jnp.einsum(
            "rsij,rsij->rs",
            owner_hessian,
            transition.sample_moment_delta,
        )
        + (1.0 / 6.0)
        * jnp.einsum(
            "rsijk,rsijk->rs",
            owner_third,
            transition.sample_third_moment_delta,
        )
    )
    sample_value = jnp.where(
        transition.sample_direct,
        storage_value,
        virtual_value,
    )
    sample_valid = transition.sample_active & (
        transition.sample_direct | owner_valid
    ) & (~transition.sample_remote)
    row_valid = (
        transition.active
        & transition.valid
        & jnp.all((~transition.sample_active) | sample_valid, axis=-1)
        & jnp.all(jnp.isfinite(sample_value) | (~transition.sample_active), axis=-1)
    )
    scalar = jnp.einsum(
        "rs,rs->r",
        transition.scalar_coefficients,
        sample_value,
    )
    gradient = jnp.einsum(
        "rcs,rs->rc",
        transition.gradient_coefficients,
        sample_value,
    )
    return scalar, gradient, row_valid


def _local_control_volume_irregular_parallel_flux(
    values_owned: jnp.ndarray,
    polynomial: LocalControlVolumePolynomial3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    boundary_bc: LocalControlVolumeBoundaryBC3D,
    *,
    b_floor: float,
) -> jnp.ndarray:
    """Integrate one parallel scalar flux for each unique irregular face."""

    faces = control_volume_geometry.irregular_faces
    cells = control_volume_geometry.cells
    canonical_values = (
        values_owned
        if polynomial.owner_values is None
        else polynomial.owner_values
    )
    if int(faces.max_rows) == 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    points = faces.quadrature_points
    minus_index = (
        faces.minus_owner_i[:, None, None],
        faces.minus_owner_j[:, None, None],
        faces.minus_owner_k[:, None, None],
    )
    plus_index = (
        faces.plus_owner_i[:, None, None],
        faces.plus_owner_j[:, None, None],
        faces.plus_owner_k[:, None, None],
    )
    minus_value, _minus_gradient, minus_valid = (
        evaluate_local_control_volume_polynomial(
            canonical_values,
            polynomial,
            cells,
            *minus_index,
            points,
        )
    )
    plus_value, _plus_gradient, plus_valid = (
        evaluate_local_control_volume_polynomial(
            canonical_values,
            polynomial,
            cells,
            *plus_index,
            points,
        )
    )
    quadrature_shape = faces.quadrature_points.shape[:-1]
    has_remote_samples = (
        polynomial.remote_face_value.shape == quadrature_shape
        and polynomial.remote_face_valid.shape == quadrature_shape
    )
    if has_remote_samples:
        neighbor_value = jnp.where(
            faces.has_remote_owner[:, None, None],
            polynomial.remote_face_value,
            plus_value,
        )
        neighbor_valid = jnp.where(
            faces.has_remote_owner[:, None, None],
            polynomial.remote_face_valid,
            plus_valid,
        )
    else:
        neighbor_value = plus_value
        neighbor_valid = plus_valid
    has_neighbor = faces.has_plus_owner | faces.has_remote_owner
    interior_value = 0.5 * (minus_value + neighbor_value)
    boundary_value = jnp.where(
        boundary_bc.kind[:, None, None] == BC_DIRICHLET,
        boundary_bc.quadrature_value,
        minus_value,
    )
    face_value = jnp.where(
        has_neighbor[:, None, None],
        interior_value,
        boundary_value,
    )
    transition_value, _transition_gradient, transition_valid = (
        _evaluate_local_regular_transition_functional(
            canonical_values,
            polynomial,
            control_volume_geometry,
        )
    )
    transition_rows = control_volume_geometry.regular_transition_faces
    transition_active = (
        transition_rows.active
        if int(transition_rows.max_rows) == int(faces.max_rows)
        else jnp.zeros((int(faces.max_rows),), dtype=bool)
    )
    face_value = jnp.where(
        transition_active[:, None, None],
        transition_value[:, None, None],
        face_value,
    )
    unit_b = faces.B_contra / jnp.maximum(
        faces.Bmag[..., None],
        float(b_floor),
    )
    directed_measure = jnp.einsum(
        "rpqi,rpqi->rpq",
        faces.area_covector_weight,
        unit_b,
    )
    quadrature_flux = faces.J * directed_measure * face_value
    normal_flux = (
        faces.J
        * jnp.linalg.norm(faces.area_covector_weight, axis=-1)
        * boundary_bc.quadrature_value
    )
    quadrature_flux = jnp.where(
        (
            (~has_neighbor)
            & boundary_bc.active
            & (boundary_bc.kind == BC_NORMALFLUX)
        )[:, None, None],
        normal_flux,
        quadrature_flux,
    )
    quadrature_flux = jnp.where(
        (
            (~has_neighbor)
            & boundary_bc.active
            & (boundary_bc.kind == BC_NOFLUX)
        )[:, None, None],
        0.0,
        quadrature_flux,
    )
    valid = (
        faces.quadrature_active
        & minus_valid
        & ((~has_neighbor[:, None, None]) | neighbor_valid)
        & jnp.isfinite(quadrature_flux)
    )
    valid = valid & (
        (~transition_active[:, None, None])
        | transition_valid[:, None, None]
    )
    return jnp.where(
        faces.active,
        jnp.sum(jnp.where(valid, quadrature_flux, 0.0), axis=(1, 2)),
        0.0,
    )


def _replace_local_cut_wall_dirichlet_normal_derivative(
    values_owned: jnp.ndarray,
    polynomial: LocalControlVolumePolynomial3D,
    cells: LocalControlVolumeCellGeometry3D,
    faces: LocalControlVolumeFaceRows3D,
    boundary_bc: LocalControlVolumeBoundaryBC3D,
    face_gradient: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the coordinate-normal Dirichlet closure on embedded box walls.

    The polynomial provides the tangential derivatives at a cut-wall
    quadrature point.  For an axis-aligned embedded wall, the missing normal
    coordinate derivative is better determined by the wall value plus the
    first two distinct inward control-volume samples.  This quadratic
    functional is second-order at the face and avoids amplifying the ordinary
    reconstruction error through a cubic one-sided extrapolation.
    """

    cut_wall_dirichlet = (
        faces.active
        & (faces.kind == CV_FACE_CUT_WALL)
        & (~faces.has_plus_owner)
        & (~faces.has_remote_owner)
        & faces.boundary_normal_stencil_valid
        & boundary_bc.active
        & (boundary_bc.kind == BC_DIRICHLET)
    )
    if int(faces.max_rows) == 0:
        return face_gradient, jnp.zeros((0,), dtype=bool)

    canonical_values = (
        values_owned
        if polynomial.owner_values is None
        else polynomial.owner_values
    )

    points = faces.quadrature_points
    axis_vector = jax.nn.one_hot(
        faces.boundary_normal_axis,
        3,
        dtype=points.dtype,
    )
    boundary_coordinate = jnp.einsum("rpqi,ri->rpq", points, axis_vector)
    inward_values = []
    inward_valid = []
    for sample_index in range(3):
        sample_coordinate = faces.boundary_sample_coordinate[:, sample_index]
        sample_points = (
            points
            + axis_vector[:, None, None, :]
            * (
                sample_coordinate[:, None, None, None]
                - boundary_coordinate[..., None]
            )
        )
        sample_value, _sample_gradient, sample_valid = (
            evaluate_local_control_volume_polynomial(
                canonical_values,
                polynomial,
                cells,
                faces.boundary_sample_owner_i[:, sample_index, None, None],
                faces.boundary_sample_owner_j[:, sample_index, None, None],
                faces.boundary_sample_owner_k[:, sample_index, None, None],
                sample_points,
            )
        )
        inward_values.append(sample_value)
        inward_valid.append(sample_valid)
    inward_values_array = jnp.stack(inward_values, axis=-1)
    inward_valid_array = jnp.all(jnp.stack(inward_valid, axis=-1), axis=-1)
    derivative_weights = faces.boundary_dcoordinate_weights
    normal_derivative = (
        derivative_weights[:, None, None, 0] * boundary_bc.quadrature_value
        + jnp.sum(
            derivative_weights[:, None, None, 1:] * inward_values_array,
            axis=-1,
        )
    )
    valid = (
        cut_wall_dirichlet[:, None, None]
        & faces.quadrature_active
        & inward_valid_array
        & jnp.isfinite(normal_derivative)
    )
    baseline_normal_derivative = jnp.einsum(
        "rpqi,ri->rpq",
        face_gradient,
        axis_vector,
    )
    corrected_gradient = face_gradient + axis_vector[:, None, None, :] * (
        normal_derivative - baseline_normal_derivative
    )[..., None]
    return jnp.where(valid[..., None], corrected_gradient, face_gradient), valid


def _local_control_volume_irregular_projected_flux(
    values_owned: jnp.ndarray,
    polynomial: LocalControlVolumePolynomial3D,
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D,
    boundary_bc: LocalControlVolumeBoundaryBC3D,
) -> jnp.ndarray:
    """Integrate one projected-gradient flux for each irregular face."""

    faces = control_volume_geometry.irregular_faces
    cells = control_volume_geometry.cells
    canonical_values = (
        values_owned
        if polynomial.owner_values is None
        else polynomial.owner_values
    )
    if int(faces.max_rows) == 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    points = faces.quadrature_points
    minus_index = (
        faces.minus_owner_i[:, None, None],
        faces.minus_owner_j[:, None, None],
        faces.minus_owner_k[:, None, None],
    )
    plus_index = (
        faces.plus_owner_i[:, None, None],
        faces.plus_owner_j[:, None, None],
        faces.plus_owner_k[:, None, None],
    )
    _minus_value, minus_gradient, minus_valid = (
        evaluate_local_control_volume_polynomial(
            canonical_values,
            polynomial,
            cells,
            *minus_index,
            points,
        )
    )
    _plus_value, plus_gradient, plus_valid = (
        evaluate_local_control_volume_polynomial(
            canonical_values,
            polynomial,
            cells,
            *plus_index,
            points,
        )
    )
    quadrature_shape = faces.quadrature_points.shape[:-1]
    has_remote_samples = (
        polynomial.remote_face_gradient.shape == quadrature_shape + (3,)
        and polynomial.remote_face_valid.shape == quadrature_shape
    )
    if has_remote_samples:
        neighbor_gradient = jnp.where(
            faces.has_remote_owner[:, None, None, None],
            polynomial.remote_face_gradient,
            plus_gradient,
        )
        neighbor_valid = jnp.where(
            faces.has_remote_owner[:, None, None],
            polynomial.remote_face_valid,
            plus_valid,
        )
    else:
        neighbor_gradient = plus_gradient
        neighbor_valid = plus_valid
    has_neighbor = faces.has_plus_owner | faces.has_remote_owner
    face_gradient = jnp.where(
        has_neighbor[:, None, None, None],
        0.5 * (minus_gradient + neighbor_gradient),
        minus_gradient,
    )
    _transition_value, transition_gradient, transition_valid = (
        _evaluate_local_regular_transition_functional(
            canonical_values,
            polynomial,
            control_volume_geometry,
        )
    )
    transition_rows = control_volume_geometry.regular_transition_faces
    transition_active = (
        transition_rows.active
        if int(transition_rows.max_rows) == int(faces.max_rows)
        else jnp.zeros((int(faces.max_rows),), dtype=bool)
    )
    face_gradient = jnp.where(
        transition_active[:, None, None, None],
        transition_gradient[:, None, None, :],
        face_gradient,
    )
    # Dirichlet wall values constrain the polynomial trace, but they do not
    # sufficiently constrain its one-sided coordinate-normal derivative.  At
    # an embedded wall, use the precomputed wall-plus-inward-CV functional for
    # that component only.  The cubic polynomial continues to supply all
    # tangential components, including their variation over each quadrature
    # patch.
    face_gradient, _cut_wall_normal_closure_valid = (
        _replace_local_cut_wall_dirichlet_normal_derivative(
            canonical_values,
            polynomial,
            cells,
            faces,
            boundary_bc,
            face_gradient,
        )
    )
    quadrature_flux = faces.J * jnp.einsum(
        "rpqi,rpqij,rpqj->rpq",
        faces.area_covector_weight,
        faces.projector,
        face_gradient,
    )
    normal_flux = (
        faces.J
        * jnp.linalg.norm(faces.area_covector_weight, axis=-1)
        * boundary_bc.quadrature_value
    )
    quadrature_flux = jnp.where(
        (
            (~has_neighbor)
            & boundary_bc.active
            & (boundary_bc.kind == BC_NORMALFLUX)
        )[:, None, None],
        normal_flux,
        quadrature_flux,
    )
    quadrature_flux = jnp.where(
        (
            (~has_neighbor)
            & boundary_bc.active
            & (boundary_bc.kind == BC_NOFLUX)
        )[:, None, None],
        0.0,
        quadrature_flux,
    )
    valid = (
        faces.quadrature_active
        & minus_valid
        & ((~has_neighbor[:, None, None]) | neighbor_valid)
        & jnp.isfinite(quadrature_flux)
    )
    valid = valid & (
        (~transition_active[:, None, None])
        | transition_valid[:, None, None]
    )
    return jnp.where(
        faces.active,
        jnp.sum(jnp.where(valid, quadrature_flux, 0.0), axis=(1, 2)),
        0.0,
    )


def _local_control_volume_integrated_divergence(
    regular_flux: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    irregular_flux: jnp.ndarray,
    geometry: LocalFciGeometry3D,
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
    owner_sum = jnp.zeros(geometry.owned_shape, dtype=jnp.float64).at[
        cells.owner_i,
        cells.owner_j,
        cells.owner_k,
    ].add(
        jnp.where(cells.raw_volume > 0.0, integrated_sum, 0.0)
    )

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
        regular_boundary_closure=regular_boundary_closure,
    )

    x_face_metric = geometry.face_metric.x
    y_face_metric = geometry.face_metric.y
    z_face_metric = geometry.face_metric.z

    x_flux = jnp.asarray(x_face_metric.J_owned, dtype=jnp.float64) * jnp.einsum(
        "...j,...j->...", x_face_projector[..., 0, :], x_face_grad
    )
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
    boundary_bc: LocalControlVolumeBoundaryBC3D | None = None,
    field_reconstruction: LocalControlVolumePolynomial3D | None = None,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
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
        if boundary_bc is None or field_reconstruction is None:
            raise ValueError(
                "boundary_bc and field_reconstruction are required with "
                "control_volume_geometry"
            )
        regular_flux = (
            cv_flux.regular_flux.x,
            cv_flux.regular_flux.y,
            cv_flux.regular_flux.z,
        )
        irregular_flux = _local_control_volume_irregular_projected_flux(
            jnp.asarray(local.x.center, dtype=jnp.float64),
            field_reconstruction,
            control_volume_geometry,
            boundary_bc,
        )
        return _local_control_volume_integrated_divergence(
            regular_flux,
            irregular_flux,
            geometry,
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
    boundary_bc: LocalControlVolumeBoundaryBC3D | None = None,
    field_reconstruction: LocalControlVolumePolynomial3D | None = None,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
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
        b_floor=b_floor,
    )
    if control_volume_geometry is not None:
        if boundary_bc is None or field_reconstruction is None:
            raise ValueError(
                "boundary_bc and field_reconstruction are required with "
                "control_volume_geometry"
            )
        irregular_flux = _local_control_volume_irregular_projected_flux(
            jnp.asarray(local.x.center, dtype=jnp.float64),
            field_reconstruction,
            control_volume_geometry,
            boundary_bc,
        )
        return _local_control_volume_integrated_divergence(
            (
                cv_flux.regular_flux.x,
                cv_flux.regular_flux.y,
                cv_flux.regular_flux.z,
            ),
            irregular_flux,
            geometry,
            control_volume_geometry,
            volume_floor=jacobian_floor,
        )
    return local_divergence_conservative_op(cv_flux, geometry, jacobian_floor=jacobian_floor)


def _has_dirichlet_regular_faces(face_bc: BoundaryFaceBC3D | None) -> bool:
    if face_bc is None:
        return False
    return bool(
        jnp.any((face_bc.kind_x == BC_DIRICHLET) & face_bc.mask_x)
        or jnp.any((face_bc.kind_y == BC_DIRICHLET) & face_bc.mask_y)
        or jnp.any((face_bc.kind_z == BC_DIRICHLET) & face_bc.mask_z)
    )


def _has_dirichlet_cut_walls(cut_wall_bc: CutWallBC3D | None) -> bool:
    if cut_wall_bc is None or not cut_wall_bc.kind.size:
        return False
    return bool(jnp.any(jnp.asarray(cut_wall_bc.kind, dtype=jnp.int32) == BC_DIRICHLET))


def _homogeneous_boundary_face_bc(face_bc: BoundaryFaceBC3D) -> BoundaryFaceBC3D:
    """Keep regular-face BC kinds/masks but remove affine boundary values."""

    return face_bc.replace(
        value_x=jnp.zeros_like(face_bc.value_x, dtype=jnp.float64),
        value_y=jnp.zeros_like(face_bc.value_y, dtype=jnp.float64),
        value_z=jnp.zeros_like(face_bc.value_z, dtype=jnp.float64),
    )


def _homogeneous_cut_wall_bc(cut_wall_bc: CutWallBC3D | None) -> CutWallBC3D:
    """Keep cut-wall BC kinds but remove affine wall values."""

    cut_wall_bc = cut_wall_bc or CutWallBC3D.empty()
    return CutWallBC3D(
        kind=jnp.asarray(cut_wall_bc.kind, dtype=jnp.int32),
        value=jnp.zeros_like(jnp.asarray(cut_wall_bc.value, dtype=jnp.float64)),
    )


def _homogeneous_boundary_payload(
    face_bc: BoundaryFaceBC3D,
    cut_wall_bc: CutWallBC3D | None,
) -> tuple[BoundaryFaceBC3D, CutWallBC3D]:
    """Return the homogeneous payload used by the linear inverse operator."""

    return _homogeneous_boundary_face_bc(face_bc), _homogeneous_cut_wall_bc(cut_wall_bc)


def _dirichlet_lift_correction_face_bc(face_bc: BoundaryFaceBC3D) -> BoundaryFaceBC3D:
    """Return correction BCs for ``phi = phi_lift + u`` on regular faces."""

    return face_bc.replace(
        value_x=jnp.where(face_bc.kind_x == BC_DIRICHLET, 0.0, face_bc.value_x),
        value_y=jnp.where(face_bc.kind_y == BC_DIRICHLET, 0.0, face_bc.value_y),
        value_z=jnp.where(face_bc.kind_z == BC_DIRICHLET, 0.0, face_bc.value_z),
    )


def _dirichlet_lift_correction_cut_wall_bc(cut_wall_bc: CutWallBC3D | None) -> CutWallBC3D:
    """Return correction BCs for ``phi = phi_lift + u`` on cut walls."""

    cut_wall_bc = cut_wall_bc or CutWallBC3D.empty()
    return CutWallBC3D(
        kind=jnp.asarray(cut_wall_bc.kind, dtype=jnp.int32),
        value=jnp.where(
            jnp.asarray(cut_wall_bc.kind, dtype=jnp.int32) == BC_DIRICHLET,
            0.0,
            jnp.asarray(cut_wall_bc.value, dtype=jnp.float64),
        ),
    )


def _perp_laplacian_from_field_and_bc(
    field: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    bc: FciBoundaryCondition,
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    raise NotImplementedError("legacy axis-level boundary conditions have been removed")


def _axis_index_nd(axis: int, index: int, ndim: int) -> tuple[object, ...]:
    slices: list[object] = [slice(None)] * ndim
    slices[axis] = index
    return tuple(slices)


def _axis_slice_nd(axis: int, start: int | None, stop: int | None, ndim: int) -> tuple[object, ...]:
    slices: list[object] = [slice(None)] * ndim
    slices[axis] = slice(start, stop)
    return tuple(slices)


def _axis_name(axis: int) -> str:
    return ("x", "y", "z")[int(axis)]


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


def _bc_periodic_axes(bc) -> tuple[bool, bool, bool]:
    raise NotImplementedError("legacy axis-level boundary conditions have been removed")


def _bc_axis_spec(bc, axis: int, *, periodic_axes: tuple[bool, bool, bool]) -> object | None:
    raise NotImplementedError("legacy axis-level boundary conditions have been removed")


def _homogeneous_axis_bc(axis_bc: object | None) -> object | None:
    raise NotImplementedError("legacy axis-level boundary conditions have been removed")


def _homogeneous_bc(bc: FciBoundaryCondition) -> FciBoundaryCondition:
    raise NotImplementedError("legacy axis-level boundary conditions have been removed")


def _bc_kind(bc) -> str:
    raise NotImplementedError("legacy axis-level boundary conditions have been removed")


def _broadcast_boundary_value(value, target_shape: tuple[int, int]) -> jnp.ndarray:
    array = jnp.asarray(value, dtype=jnp.float64)
    return jnp.broadcast_to(array, target_shape)


def _broadcast_axis_boundary_value(value, *, axis: int, field_shape: tuple[int, int, int]) -> jnp.ndarray:
    target_shape = tuple(field_shape[index] for index in range(3) if index != axis)
    return _broadcast_boundary_value(value, target_shape)


def _set_axis_plane(field: jnp.ndarray, *, axis: int, index: int, value: jnp.ndarray) -> jnp.ndarray:
    return field.at[_axis_index_nd(axis, index, field.ndim)].set(value)


def _apply_dirichlet_constraints(
    field: jnp.ndarray,
    *,
    axis_bcs: tuple[object | None, object | None, object | None],
    periodic_axes: tuple[bool, bool, bool],
) -> jnp.ndarray:
    constrained = jnp.asarray(field, dtype=jnp.float64)
    for axis, axis_bc in enumerate(axis_bcs):
        if periodic_axes[axis] or axis_bc is None:
            continue
        if _bc_kind(axis_bc) != "dirichlet":
            continue
        constrained = _set_axis_plane(
            constrained,
            axis=axis,
            index=0,
            value=_broadcast_axis_boundary_value(axis_bc.lower_value, axis=axis, field_shape=constrained.shape),
        )
        constrained = _set_axis_plane(
            constrained,
            axis=axis,
            index=-1,
            value=_broadcast_axis_boundary_value(axis_bc.upper_value, axis=axis, field_shape=constrained.shape),
        )
    return constrained


def _zero_dirichlet_boundary_residual(
    field: jnp.ndarray,
    *,
    axis_bcs: tuple[object | None, object | None, object | None],
    periodic_axes: tuple[bool, bool, bool],
) -> jnp.ndarray:
    residual = jnp.asarray(field, dtype=jnp.float64)
    for axis, axis_bc in enumerate(axis_bcs):
        if periodic_axes[axis] or axis_bc is None or _bc_kind(axis_bc) != "dirichlet":
            continue
        residual = _set_axis_plane(residual, axis=axis, index=0, value=jnp.zeros_like(residual[_axis_index_nd(axis, 0, residual.ndim)]))
        residual = _set_axis_plane(residual, axis=axis, index=-1, value=jnp.zeros_like(residual[_axis_index_nd(axis, -1, residual.ndim)]))
    return residual


def _dirichlet_boundary_flux(
    values: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    axis: int,
    side: str,
    periodic_axes: tuple[bool, bool, bool],
    b_floor: float = 1.0e-30,
) -> jnp.ndarray:
    field = jnp.asarray(values, dtype=jnp.float64)
    if axis == 0:
        metric = geometry.face_metric.x
        bfield = geometry.face_bfield.x
    elif axis == 1:
        metric = geometry.face_metric.y
        bfield = geometry.face_bfield.y
    else:
        metric = geometry.face_metric.z
        bfield = geometry.face_bfield.z

    b_unit = jnp.asarray(bfield.b_contra, dtype=jnp.float64)
    projector = jnp.asarray(metric.g_contra, dtype=jnp.float64) - jnp.einsum("...i,...j->...ij", b_unit, b_unit)

    dfdx = _first_derivative_3d(field, geometry.spacing.dx, axis=0, periodic=periodic_axes[0])
    dfdy = _first_derivative_3d(field, geometry.spacing.dy, axis=1, periodic=periodic_axes[1])
    dfdz = _first_derivative_3d(field, geometry.spacing.dz, axis=2, periodic=periodic_axes[2])
    grad_components = [dfdx, dfdy, dfdz]

    if side == "lower":
        if field.shape[axis] < 3:
            raise ValueError("dirichlet boundary flux requires at least 3 points along the selected axis")
        normal_boundary = (
            -3.0 * field[_axis_index_nd(axis, 0, field.ndim)]
            + 4.0 * field[_axis_index_nd(axis, 1, field.ndim)]
            - field[_axis_index_nd(axis, 2, field.ndim)]
        ) / (2.0 * jnp.asarray(geometry.spacing.dx if axis == 0 else geometry.spacing.dy if axis == 1 else geometry.spacing.dz, dtype=jnp.float64)[_axis_index_nd(axis, 0, field.ndim)])
        boundary_components = [component[_axis_index_nd(axis, 0, field.ndim)] for component in grad_components]
        j_boundary = jnp.asarray(metric.J, dtype=jnp.float64)[_axis_index_nd(axis, 0, field.ndim)]
        projector_row = projector[_axis_index_nd(axis, 0, field.ndim)][..., axis, :]
    elif side == "upper":
        if field.shape[axis] < 3:
            raise ValueError("dirichlet boundary flux requires at least 3 points along the selected axis")
        normal_boundary = (
            3.0 * field[_axis_index_nd(axis, -1, field.ndim)]
            - 4.0 * field[_axis_index_nd(axis, -2, field.ndim)]
            + field[_axis_index_nd(axis, -3, field.ndim)]
        ) / (2.0 * jnp.asarray(geometry.spacing.dx if axis == 0 else geometry.spacing.dy if axis == 1 else geometry.spacing.dz, dtype=jnp.float64)[_axis_index_nd(axis, -1, field.ndim)])
        boundary_components = [component[_axis_index_nd(axis, -1, field.ndim)] for component in grad_components]
        j_boundary = jnp.asarray(metric.J, dtype=jnp.float64)[_axis_index_nd(axis, -1, field.ndim)]
        projector_row = projector[_axis_index_nd(axis, -1, field.ndim)][..., axis, :]
    else:
        raise ValueError("side must be 'lower' or 'upper'")

    boundary_components[axis] = normal_boundary
    grad_boundary = jnp.stack(boundary_components, axis=-1)
    return j_boundary * jnp.einsum("...j,...j->...", projector_row, grad_boundary)


def _cell_volume_weights(geometry: FciGeometry3D) -> jnp.ndarray:
    return (
        jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dx, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dy, dtype=jnp.float64)
        * jnp.asarray(geometry.spacing.dz, dtype=jnp.float64)
    )


def _weighted_mean(field: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    weights = _cell_volume_weights(geometry)
    return jnp.sum(weights * field) / jnp.maximum(jnp.sum(weights), 1.0e-30)


def _weighted_l2(field: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    weights = _cell_volume_weights(geometry)
    return jnp.sqrt(jnp.sum(weights * field * field) / jnp.maximum(jnp.sum(weights), 1.0e-30))


def _remove_weighted_mean(field: jnp.ndarray, geometry: FciGeometry3D) -> jnp.ndarray:
    values = jnp.asarray(field, dtype=jnp.float64)
    return values - _weighted_mean(values, geometry)


def _set_weighted_mean(field: jnp.ndarray, geometry: FciGeometry3D, target_mean: object) -> jnp.ndarray:
    values = jnp.asarray(field, dtype=jnp.float64)
    target = jnp.asarray(target_mean, dtype=jnp.float64)
    return values + (target - _weighted_mean(values, geometry))


@_pytree_base
@dataclass(frozen=True)
class PerpLaplacianMgLevel:
    """One conservative perpendicular-Laplacian multigrid level."""

    geometry: FciGeometry3D
    stencil_builder: ConservativeStencilBuilder
    face_bc: BoundaryFaceBC3D
    regular_face_geometry: RegularFaceGeometry3D
    cut_wall_geometry: CutWallGeometry3D
    cut_wall_bc: CutWallBC3D
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    diag_inv: jnp.ndarray
    periodic_axes: tuple[bool, bool, bool]
    has_dirichlet: bool
    has_nullspace: bool

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.geometry.shape

    def tree_flatten(self):
        children = (
            self.geometry,
            self.stencil_builder,
            self.face_bc,
            self.regular_face_geometry,
            self.cut_wall_geometry,
            self.cut_wall_bc,
            self.face_projectors,
            self.diag_inv,
        )
        aux = (self.periodic_axes, self.has_dirichlet, self.has_nullspace)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        periodic_axes, has_dirichlet, has_nullspace = aux_data
        (
            geometry,
            stencil_builder,
            face_bc,
            regular_face_geometry,
            cut_wall_geometry,
            cut_wall_bc,
            face_projectors,
            diag_inv,
        ) = children
        return cls(
            geometry=geometry,
            stencil_builder=stencil_builder,
            face_bc=face_bc,
            regular_face_geometry=regular_face_geometry,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=cut_wall_bc,
            face_projectors=face_projectors,
            diag_inv=diag_inv,
            periodic_axes=periodic_axes,
            has_dirichlet=has_dirichlet,
            has_nullspace=has_nullspace,
        )


@_pytree_base
@dataclass(frozen=True)
class PerpLaplacianMgHierarchy:
    """Reusable V-cycle preconditioner payload for the conservative inverse."""

    levels: tuple[PerpLaplacianMgLevel, ...]
    pre_smooth: int = 2
    post_smooth: int = 2
    coarse_smooth: int = 16
    omega_jacobi: float = 0.65
    smoother: Literal["jacobi", "chebyshev"] = "chebyshev"
    chebyshev_order: int = 2
    spectral_radius_estimate: float | None = None
    direct_coarse_size: int = 512

    def tree_flatten(self):
        return self.levels, (
            self.pre_smooth,
            self.post_smooth,
            self.coarse_smooth,
            self.omega_jacobi,
            self.smoother,
            self.chebyshev_order,
            self.spectral_radius_estimate,
            self.direct_coarse_size,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            pre_smooth,
            post_smooth,
            coarse_smooth,
            omega_jacobi,
            smoother,
            chebyshev_order,
            spectral_radius_estimate,
            direct_coarse_size,
        ) = aux_data
        return cls(
            levels=tuple(children),
            pre_smooth=pre_smooth,
            post_smooth=post_smooth,
            coarse_smooth=coarse_smooth,
            omega_jacobi=omega_jacobi,
            smoother=smoother,
            chebyshev_order=chebyshev_order,
            spectral_radius_estimate=spectral_radius_estimate,
            direct_coarse_size=direct_coarse_size,
        )


def _zero_dirichlet_face_adjacent_cells(field: jnp.ndarray, face_bc: BoundaryFaceBC3D) -> jnp.ndarray:
    values = jnp.asarray(field, dtype=jnp.float64)
    lower_x_mask = (face_bc.kind_x[0] == BC_DIRICHLET) & face_bc.mask_x[0]
    upper_x_mask = (face_bc.kind_x[-1] == BC_DIRICHLET) & face_bc.mask_x[-1]
    lower_y_mask = (face_bc.kind_y[:, 0, :] == BC_DIRICHLET) & face_bc.mask_y[:, 0, :]
    upper_y_mask = (face_bc.kind_y[:, -1, :] == BC_DIRICHLET) & face_bc.mask_y[:, -1, :]
    lower_z_mask = (face_bc.kind_z[:, :, 0] == BC_DIRICHLET) & face_bc.mask_z[:, :, 0]
    upper_z_mask = (face_bc.kind_z[:, :, -1] == BC_DIRICHLET) & face_bc.mask_z[:, :, -1]

    values = values.at[0, :, :].set(jnp.where(lower_x_mask, 0.0, values[0, :, :]))
    values = values.at[-1, :, :].set(jnp.where(upper_x_mask, 0.0, values[-1, :, :]))
    values = values.at[:, 0, :].set(jnp.where(lower_y_mask, 0.0, values[:, 0, :]))
    values = values.at[:, -1, :].set(jnp.where(upper_y_mask, 0.0, values[:, -1, :]))
    values = values.at[:, :, 0].set(jnp.where(lower_z_mask, 0.0, values[:, :, 0]))
    values = values.at[:, :, -1].set(jnp.where(upper_z_mask, 0.0, values[:, :, -1]))
    return values


def _project_homogeneous_correction(field: jnp.ndarray, *, level: PerpLaplacianMgLevel) -> jnp.ndarray:
    values = jnp.asarray(field, dtype=jnp.float64)
    if level.has_dirichlet:
        values = _zero_dirichlet_face_adjacent_cells(values, level.face_bc)
    if level.has_nullspace:
        values = _remove_weighted_mean(values, level.geometry)
    return values


def _restrict_axis_cell_centered(values: jnp.ndarray, *, axis: int) -> jnp.ndarray:
    arr = jnp.moveaxis(jnp.asarray(values, dtype=jnp.float64), axis, 0)
    even = arr[0::2]
    odd = arr[1::2]
    if odd.shape[0] < even.shape[0]:
        odd = jnp.concatenate((odd, jnp.zeros_like(even[-1:])), axis=0)
        count = jnp.concatenate(
            (
                jnp.full((even.shape[0] - 1,) + (1,) * (even.ndim - 1), 2.0, dtype=jnp.float64),
                jnp.ones((1,) + (1,) * (even.ndim - 1), dtype=jnp.float64),
            ),
            axis=0,
        )
    else:
        count = jnp.full((even.shape[0],) + (1,) * (even.ndim - 1), 2.0, dtype=jnp.float64)
    return jnp.moveaxis((even + odd) / count, 0, axis)


def _restrict_axis_cell_sum(values: jnp.ndarray, *, axis: int) -> jnp.ndarray:
    arr = jnp.moveaxis(jnp.asarray(values, dtype=jnp.float64), axis, 0)
    even = arr[0::2]
    odd = arr[1::2]
    if odd.shape[0] < even.shape[0]:
        odd = jnp.concatenate((odd, jnp.zeros_like(even[-1:])), axis=0)
    return jnp.moveaxis(even + odd, 0, axis)


def _prolong_axis_cell_centered(values: jnp.ndarray, *, axis: int, target_size: int, periodic: bool) -> jnp.ndarray:
    arr = jnp.moveaxis(jnp.asarray(values, dtype=jnp.float64), axis, 0)
    nc = arr.shape[0]
    out = jnp.zeros((target_size,) + arr.shape[1:], dtype=arr.dtype)
    even_indices = jnp.arange(0, target_size, 2)
    out = out.at[even_indices].set(arr[: even_indices.shape[0]])
    odd_indices = jnp.arange(1, target_size, 2)
    if odd_indices.shape[0]:
        if periodic:
            right = jnp.roll(arr, -1, axis=0)
        else:
            right = jnp.concatenate((arr[1:], arr[-1:]), axis=0) if nc > 1 else arr
        odd_values = 0.5 * (arr[: odd_indices.shape[0]] + right[: odd_indices.shape[0]])
        out = out.at[odd_indices].set(odd_values)
    return jnp.moveaxis(out, 0, axis)


def _restrict_field_simple(field: jnp.ndarray, *, periodic_axes: tuple[bool, bool, bool]) -> jnp.ndarray:
    del periodic_axes
    values = jnp.asarray(field, dtype=jnp.float64)
    for axis in range(3):
        values = _restrict_axis_cell_centered(values, axis=axis)
    return values


def _restrict_field_sum(field: jnp.ndarray, *, periodic_axes: tuple[bool, bool, bool]) -> jnp.ndarray:
    del periodic_axes
    values = jnp.asarray(field, dtype=jnp.float64)
    for axis in range(3):
        values = _restrict_axis_cell_sum(values, axis=axis)
    return values


def _prolong_field(
    field: jnp.ndarray,
    coarse_level: PerpLaplacianMgLevel,
    fine_level: PerpLaplacianMgLevel,
) -> jnp.ndarray:
    values = jnp.asarray(field, dtype=jnp.float64)
    for axis in range(3):
        values = _prolong_axis_cell_centered(
            values,
            axis=axis,
            target_size=fine_level.shape[axis],
            periodic=fine_level.periodic_axes[axis],
        )
    if values.shape != fine_level.shape:
        raise ValueError(f"prolongation produced shape {values.shape}, expected {fine_level.shape}")
    return values


def _restrict_residual_jweighted(
    residual: jnp.ndarray,
    fine_level: PerpLaplacianMgLevel,
    coarse_level: PerpLaplacianMgLevel,
) -> jnp.ndarray:
    fine_weights = _cell_volume_weights(fine_level.geometry)
    weighted_residual = _restrict_field_sum(
        fine_weights * jnp.asarray(residual, dtype=jnp.float64),
        periodic_axes=fine_level.periodic_axes,
    )
    restricted_weights = _restrict_field_sum(fine_weights, periodic_axes=fine_level.periodic_axes)
    coarse_rhs = weighted_residual / jnp.maximum(restricted_weights, 1.0e-30)
    if coarse_rhs.shape != coarse_level.shape:
        raise ValueError(f"restricted residual shape {coarse_rhs.shape} does not match coarse level {coarse_level.shape}")
    if coarse_level.has_nullspace:
        coarse_rhs = _remove_weighted_mean(coarse_rhs, coarse_level.geometry)
    return coarse_rhs


def _cell_indices_for_coarsening(size: int) -> jnp.ndarray:
    return jnp.arange(0, int(size), 2, dtype=jnp.int32)


def _face_indices_for_coarsening(size: int) -> jnp.ndarray:
    indices = list(range(0, int(size), 2))
    if indices[-1] != int(size) - 1:
        indices.append(int(size) - 1)
    return jnp.asarray(indices, dtype=jnp.int32)


def _take_axes(values: jnp.ndarray, indices: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    result = jnp.asarray(values)
    for axis, index in enumerate(indices):
        result = jnp.take(result, index, axis=axis)
    return result


def _coarsen_cell_average(values: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(values, dtype=jnp.float64)
    for axis in range(3):
        result = _restrict_axis_cell_centered(result, axis=axis)
    return result


def _coarsen_cell_sum(values: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(values, dtype=jnp.float64)
    for axis in range(3):
        result = _restrict_axis_cell_sum(result, axis=axis)
    return result


def _coarsen_cell_weighted_average(values: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    weight_values = jnp.asarray(weights, dtype=jnp.float64)
    while weight_values.ndim < jnp.asarray(values).ndim:
        weight_values = weight_values[..., None]
    weighted_sum = _coarsen_cell_sum(jnp.asarray(values, dtype=jnp.float64) * weight_values)
    weight_sum = _coarsen_cell_sum(weight_values)
    return weighted_sum / jnp.maximum(weight_sum, 1.0e-30)


def _coarsen_face_field(values: jnp.ndarray, *, axis: int, reduce: str = "mean") -> jnp.ndarray:
    result = jnp.asarray(values, dtype=jnp.float64)
    normal_indices = _face_indices_for_coarsening(result.shape[axis])
    result = jnp.take(result, normal_indices, axis=axis)
    for tangent_axis in range(3):
        if tangent_axis == axis:
            continue
        if reduce == "sum":
            result = _restrict_axis_cell_sum(result, axis=tangent_axis)
        else:
            result = _restrict_axis_cell_centered(result, axis=tangent_axis)
    return result


def _coarsen_face_bool(values: jnp.ndarray, *, axis: int) -> jnp.ndarray:
    result = jnp.asarray(values).astype(jnp.float64)
    result = _coarsen_face_field(result, axis=axis, reduce="mean")
    return result > 0.5


def _coarsen_metric(metric, indices: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None, weights: jnp.ndarray | None = None):
    cls = metric.__class__
    if indices is not None:
        return cls(
            J=_take_axes(metric.J, indices),
            g11=_take_axes(metric.g11, indices),
            g22=_take_axes(metric.g22, indices),
            g33=_take_axes(metric.g33, indices),
            g12=_take_axes(metric.g12, indices),
            g13=_take_axes(metric.g13, indices),
            g23=_take_axes(metric.g23, indices),
            g_11=_take_axes(metric.g_11, indices),
            g_22=_take_axes(metric.g_22, indices),
            g_33=_take_axes(metric.g_33, indices),
            g_12=_take_axes(metric.g_12, indices),
            g_13=_take_axes(metric.g_13, indices),
            g_23=_take_axes(metric.g_23, indices),
        )
    metric_weights = jnp.asarray(metric.J if weights is None else weights, dtype=jnp.float64)
    return cls(
        J=_coarsen_cell_average(metric.J),
        g11=_coarsen_cell_weighted_average(metric.g11, metric_weights),
        g22=_coarsen_cell_weighted_average(metric.g22, metric_weights),
        g33=_coarsen_cell_weighted_average(metric.g33, metric_weights),
        g12=_coarsen_cell_weighted_average(metric.g12, metric_weights),
        g13=_coarsen_cell_weighted_average(metric.g13, metric_weights),
        g23=_coarsen_cell_weighted_average(metric.g23, metric_weights),
        g_11=_coarsen_cell_weighted_average(metric.g_11, metric_weights),
        g_22=_coarsen_cell_weighted_average(metric.g_22, metric_weights),
        g_33=_coarsen_cell_weighted_average(metric.g_33, metric_weights),
        g_12=_coarsen_cell_weighted_average(metric.g_12, metric_weights),
        g_13=_coarsen_cell_weighted_average(metric.g_13, metric_weights),
        g_23=_coarsen_cell_weighted_average(metric.g_23, metric_weights),
    )


def _coarsen_bfield(bfield, indices: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None, weights: jnp.ndarray | None = None):
    cls = bfield.__class__
    if indices is not None:
        return cls(
            B_contra=_take_axes(bfield.B_contra, indices),
            Bmag=_take_axes(bfield.Bmag, indices),
        )
    b_weights = jnp.ones_like(bfield.Bmag, dtype=jnp.float64) if weights is None else jnp.asarray(weights, dtype=jnp.float64)
    return cls(
        B_contra=_coarsen_cell_weighted_average(bfield.B_contra, b_weights),
        Bmag=_coarsen_cell_weighted_average(bfield.Bmag, b_weights),
    )


def _coarsen_face_metric(metric, *, axis: int):
    cls = metric.__class__
    weights = jnp.asarray(metric.J, dtype=jnp.float64)

    def weighted(values: jnp.ndarray) -> jnp.ndarray:
        weight_values = weights
        while weight_values.ndim < jnp.asarray(values).ndim:
            weight_values = weight_values[..., None]
        numerator = _coarsen_face_field(jnp.asarray(values, dtype=jnp.float64) * weight_values, axis=axis, reduce="sum")
        denominator = _coarsen_face_field(weight_values, axis=axis, reduce="sum")
        return numerator / jnp.maximum(denominator, 1.0e-30)

    return cls(
        J=_coarsen_face_field(metric.J, axis=axis, reduce="mean"),
        g11=weighted(metric.g11),
        g22=weighted(metric.g22),
        g33=weighted(metric.g33),
        g12=weighted(metric.g12),
        g13=weighted(metric.g13),
        g23=weighted(metric.g23),
        g_11=weighted(metric.g_11),
        g_22=weighted(metric.g_22),
        g_33=weighted(metric.g_33),
        g_12=weighted(metric.g_12),
        g_13=weighted(metric.g_13),
        g_23=weighted(metric.g_23),
    )


def _coarsen_face_bfield(bfield, *, axis: int, weights: jnp.ndarray):
    cls = bfield.__class__
    face_weights = jnp.asarray(weights, dtype=jnp.float64)

    def weighted(values: jnp.ndarray) -> jnp.ndarray:
        weight_values = face_weights
        while weight_values.ndim < jnp.asarray(values).ndim:
            weight_values = weight_values[..., None]
        numerator = _coarsen_face_field(jnp.asarray(values, dtype=jnp.float64) * weight_values, axis=axis, reduce="sum")
        denominator = _coarsen_face_field(weight_values, axis=axis, reduce="sum")
        return numerator / jnp.maximum(denominator, 1.0e-30)

    return cls(
        B_contra=weighted(bfield.B_contra),
        Bmag=weighted(bfield.Bmag),
    )


def _coarsen_geometry(geometry: FciGeometry3D) -> FciGeometry3D:
    cell_indices = tuple(_cell_indices_for_coarsening(size) for size in geometry.shape)
    grid = CellCenteredGrid3D(
        x=Grid1D.from_centers(jnp.take(geometry.grid.x.centers, cell_indices[0])),
        y=Grid1D.from_centers(jnp.take(geometry.grid.y.centers, cell_indices[1])),
        z=Grid1D.from_centers(jnp.take(geometry.grid.z.centers, cell_indices[2])),
    )
    shape = grid.shape
    maps = FciMaps3D(
        forward_x=_take_axes(geometry.maps.forward_x, cell_indices),
        forward_y=_take_axes(geometry.maps.forward_y, cell_indices),
        backward_x=_take_axes(geometry.maps.backward_x, cell_indices),
        backward_y=_take_axes(geometry.maps.backward_y, cell_indices),
        forward_endpoint_x=_take_axes(geometry.maps.forward_endpoint_x, cell_indices),
        forward_endpoint_y=_take_axes(geometry.maps.forward_endpoint_y, cell_indices),
        forward_endpoint_z=_take_axes(geometry.maps.forward_endpoint_z, cell_indices),
        backward_endpoint_x=_take_axes(geometry.maps.backward_endpoint_x, cell_indices),
        backward_endpoint_y=_take_axes(geometry.maps.backward_endpoint_y, cell_indices),
        backward_endpoint_z=_take_axes(geometry.maps.backward_endpoint_z, cell_indices),
        forward_length=2.0 * _take_axes(geometry.maps.forward_length, cell_indices),
        backward_length=2.0 * _take_axes(geometry.maps.backward_length, cell_indices),
        forward_boundary=_take_axes(geometry.maps.forward_boundary, cell_indices).astype(bool),
        backward_boundary=_take_axes(geometry.maps.backward_boundary, cell_indices).astype(bool),
    )
    spacing = Spacing3D(
        dx=jnp.broadcast_to(grid.x.widths[:, None, None], shape),
        dy=jnp.broadcast_to(grid.y.widths[None, :, None], shape),
        dz=jnp.broadcast_to(grid.z.widths[None, None, :], shape),
    )
    return FciGeometry3D(
        grid=grid,
        maps=maps,
        spacing=spacing,
        cell_metric=_coarsen_metric(geometry.cell_metric),
        face_metric=FaceMetricGeometry(
            x=_coarsen_face_metric(geometry.face_metric.x, axis=0),
            y=_coarsen_face_metric(geometry.face_metric.y, axis=1),
            z=_coarsen_face_metric(geometry.face_metric.z, axis=2),
        ),
        cell_bfield=_coarsen_bfield(geometry.cell_bfield, weights=geometry.cell_metric.J),
        face_bfield=FaceBFieldGeometry(
            x=_coarsen_face_bfield(geometry.face_bfield.x, axis=0, weights=geometry.face_metric.x.J),
            y=_coarsen_face_bfield(geometry.face_bfield.y, axis=1, weights=geometry.face_metric.y.J),
            z=_coarsen_face_bfield(geometry.face_bfield.z, axis=2, weights=geometry.face_metric.z.J),
        ),
    )


def _coarsen_regular_face_geometry(regular_face_geometry: RegularFaceGeometry3D) -> RegularFaceGeometry3D:
    return RegularFaceGeometry3D(
        x_area=_coarsen_face_field(regular_face_geometry.x_area, axis=0, reduce="mean"),
        y_area=_coarsen_face_field(regular_face_geometry.y_area, axis=1, reduce="mean"),
        z_area=_coarsen_face_field(regular_face_geometry.z_area, axis=2, reduce="mean"),
        x_area_fraction=_coarsen_face_field(regular_face_geometry.x_area_fraction, axis=0, reduce="mean"),
        y_area_fraction=_coarsen_face_field(regular_face_geometry.y_area_fraction, axis=1, reduce="mean"),
        z_area_fraction=_coarsen_face_field(regular_face_geometry.z_area_fraction, axis=2, reduce="mean"),
        x_open_mask=_coarsen_face_bool(regular_face_geometry.x_open_mask, axis=0),
        y_open_mask=_coarsen_face_bool(regular_face_geometry.y_open_mask, axis=1),
        z_open_mask=_coarsen_face_bool(regular_face_geometry.z_open_mask, axis=2),
    )


def _coarsen_face_bc(face_bc: BoundaryFaceBC3D) -> BoundaryFaceBC3D:
    def kind(values: jnp.ndarray, *, axis: int) -> jnp.ndarray:
        return jnp.rint(_coarsen_face_field(values, axis=axis, reduce="mean")).astype(jnp.int32)

    def mask(values: jnp.ndarray, *, axis: int) -> jnp.ndarray:
        return _coarsen_face_bool(values, axis=axis)

    return BoundaryFaceBC3D(
        kind_x=kind(face_bc.kind_x, axis=0),
        kind_y=kind(face_bc.kind_y, axis=1),
        kind_z=kind(face_bc.kind_z, axis=2),
        value_x=_coarsen_face_field(face_bc.value_x, axis=0, reduce="mean"),
        value_y=_coarsen_face_field(face_bc.value_y, axis=1, reduce="mean"),
        value_z=_coarsen_face_field(face_bc.value_z, axis=2, reduce="mean"),
        mask_x=mask(face_bc.mask_x, axis=0),
        mask_y=mask(face_bc.mask_y, axis=1),
        mask_z=mask(face_bc.mask_z, axis=2),
    )


def _build_approx_diag_inv(
    geometry: FciGeometry3D,
    face_bc: BoundaryFaceBC3D,
    regular_face_geometry: RegularFaceGeometry3D,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    *,
    floor: float = 1.0e-12,
) -> jnp.ndarray:
    jac = jnp.maximum(jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64), floor)
    dx2 = jnp.maximum(jnp.asarray(geometry.spacing.dx, dtype=jnp.float64) ** 2, floor)
    dy2 = jnp.maximum(jnp.asarray(geometry.spacing.dy, dtype=jnp.float64) ** 2, floor)
    dz2 = jnp.maximum(jnp.asarray(geometry.spacing.dz, dtype=jnp.float64) ** 2, floor)

    x_face_coeff = (
        jnp.asarray(geometry.face_metric.x.J, dtype=jnp.float64)
        * jnp.asarray(regular_face_geometry.x_area, dtype=jnp.float64)
        * jnp.asarray(regular_face_geometry.x_area_fraction, dtype=jnp.float64)
        * jnp.asarray(regular_face_geometry.x_open_mask, dtype=jnp.float64)
        * jnp.asarray(face_projectors[0][..., 0, 0], dtype=jnp.float64)
    )
    y_face_coeff = (
        jnp.asarray(geometry.face_metric.y.J, dtype=jnp.float64)
        * jnp.asarray(regular_face_geometry.y_area, dtype=jnp.float64)
        * jnp.asarray(regular_face_geometry.y_area_fraction, dtype=jnp.float64)
        * jnp.asarray(regular_face_geometry.y_open_mask, dtype=jnp.float64)
        * jnp.asarray(face_projectors[1][..., 1, 1], dtype=jnp.float64)
    )
    z_face_coeff = (
        jnp.asarray(geometry.face_metric.z.J, dtype=jnp.float64)
        * jnp.asarray(regular_face_geometry.z_area, dtype=jnp.float64)
        * jnp.asarray(regular_face_geometry.z_area_fraction, dtype=jnp.float64)
        * jnp.asarray(regular_face_geometry.z_open_mask, dtype=jnp.float64)
        * jnp.asarray(face_projectors[2][..., 2, 2], dtype=jnp.float64)
    )

    diag = jnp.abs(
        (
            x_face_coeff[:-1, :, :] + x_face_coeff[1:, :, :]
        )
        / dx2
        + (
            y_face_coeff[:, :-1, :] + y_face_coeff[:, 1:, :]
        )
        / dy2
        + (
            z_face_coeff[:, :, :-1] + z_face_coeff[:, :, 1:]
        )
        / dz2
    )
    diag = diag / jac
    lower_x_mask = (face_bc.kind_x[0] == BC_DIRICHLET) & face_bc.mask_x[0]
    upper_x_mask = (face_bc.kind_x[-1] == BC_DIRICHLET) & face_bc.mask_x[-1]
    lower_y_mask = (face_bc.kind_y[:, 0, :] == BC_DIRICHLET) & face_bc.mask_y[:, 0, :]
    upper_y_mask = (face_bc.kind_y[:, -1, :] == BC_DIRICHLET) & face_bc.mask_y[:, -1, :]
    lower_z_mask = (face_bc.kind_z[:, :, 0] == BC_DIRICHLET) & face_bc.mask_z[:, :, 0]
    upper_z_mask = (face_bc.kind_z[:, :, -1] == BC_DIRICHLET) & face_bc.mask_z[:, :, -1]

    diag = diag.at[0, :, :].set(jnp.where(lower_x_mask, 1.0, diag[0, :, :]))
    diag = diag.at[-1, :, :].set(jnp.where(upper_x_mask, 1.0, diag[-1, :, :]))
    diag = diag.at[:, 0, :].set(jnp.where(lower_y_mask, 1.0, diag[:, 0, :]))
    diag = diag.at[:, -1, :].set(jnp.where(upper_y_mask, 1.0, diag[:, -1, :]))
    diag = diag.at[:, :, 0].set(jnp.where(lower_z_mask, 1.0, diag[:, :, 0]))
    diag = diag.at[:, :, -1].set(jnp.where(upper_z_mask, 1.0, diag[:, :, -1]))
    return 1.0 / jnp.maximum(diag, floor)


def _can_coarsen_axis(size: int, *, periodic: bool) -> bool:
    del periodic
    return int(size) >= 4


def _can_coarsen_shape(shape: tuple[int, int, int], *, periodic_axes: tuple[bool, bool, bool]) -> bool:
    return all(_can_coarsen_axis(shape[axis], periodic=periodic_axes[axis]) for axis in range(3))


def _mg_apply_negative_perp_laplacian(field: jnp.ndarray, level: PerpLaplacianMgLevel) -> jnp.ndarray:
    values = jnp.asarray(field, dtype=jnp.float64)
    if values.shape != level.shape:
        raise ValueError(f"field must have shape {level.shape}, got {values.shape}")
    if level.has_nullspace:
        values = _remove_weighted_mean(values, level.geometry)
    local = _build_global_conservative_stencil_compat(
        level.stencil_builder,
        values,
        level.geometry,
        periodic_axes=level.periodic_axes,
        face_bc=level.face_bc,
    )
    result = -perp_laplacian_conservative_op(
        local,
        level.geometry,
        face_projectors=level.face_projectors,
        face_bc=level.face_bc,
        regular_face_geometry=level.regular_face_geometry,
        cut_wall_geometry=level.cut_wall_geometry,
        cut_wall_bc=level.cut_wall_bc,
        periodic_axes=level.periodic_axes,
    )
    if level.has_nullspace:
        result = _remove_weighted_mean(result, level.geometry)
    return result


def _jacobi_smooth_once(
    x: jnp.ndarray,
    rhs: jnp.ndarray,
    *,
    level: PerpLaplacianMgLevel,
    omega_jacobi: float,
) -> jnp.ndarray:
    residual = rhs - _mg_apply_negative_perp_laplacian(x, level)
    x_new = x + float(omega_jacobi) * level.diag_inv * residual
    return _project_homogeneous_correction(x_new, level=level)


def _jacobi_smooth(
    x: jnp.ndarray,
    rhs: jnp.ndarray,
    *,
    level: PerpLaplacianMgLevel,
    nsweeps: int,
    omega_jacobi: float,
) -> jnp.ndarray:
    def body(_, state):
        return _jacobi_smooth_once(state, rhs, level=level, omega_jacobi=omega_jacobi)

    return jax.lax.fori_loop(0, int(nsweeps), body, jnp.asarray(x, dtype=jnp.float64))


def _chebyshev_smooth(
    x: jnp.ndarray,
    rhs: jnp.ndarray,
    *,
    level: PerpLaplacianMgLevel,
    nsweeps: int,
    chebyshev_order: int,
    spectral_radius_estimate: float | None,
) -> jnp.ndarray:
    order = max(1, int(chebyshev_order))
    rho = float(spectral_radius_estimate) if spectral_radius_estimate is not None else 1.8
    lambda_max = max(rho, 1.0e-12)
    lambda_min = 0.1 * lambda_max
    center = 0.5 * (lambda_max + lambda_min)
    radius = 0.5 * (lambda_max - lambda_min)

    def step(iteration, state):
        theta = jnp.pi * (2.0 * (iteration % order) + 1.0) / (2.0 * order)
        damping = 1.0 / jnp.maximum(center - radius * jnp.cos(theta), 1.0e-12)
        residual = rhs - _mg_apply_negative_perp_laplacian(state, level)
        state = state + damping * level.diag_inv * residual
        return _project_homogeneous_correction(state, level=level)

    return jax.lax.fori_loop(0, max(0, int(nsweeps)) * order, step, jnp.asarray(x, dtype=jnp.float64))


def _smooth(
    x: jnp.ndarray,
    rhs: jnp.ndarray,
    *,
    level: PerpLaplacianMgLevel,
    nsweeps: int,
    hierarchy: PerpLaplacianMgHierarchy,
) -> jnp.ndarray:
    if hierarchy.smoother == "jacobi":
        return _jacobi_smooth(
            x,
            rhs,
            level=level,
            nsweeps=nsweeps,
            omega_jacobi=hierarchy.omega_jacobi,
        )
    if hierarchy.smoother == "chebyshev":
        return _chebyshev_smooth(
            x,
            rhs,
            level=level,
            nsweeps=nsweeps,
            chebyshev_order=hierarchy.chebyshev_order,
            spectral_radius_estimate=hierarchy.spectral_radius_estimate,
        )
    raise ValueError(f"unknown multigrid smoother {hierarchy.smoother!r}")


def _direct_coarse_solve(rhs: jnp.ndarray, level: PerpLaplacianMgLevel) -> jnp.ndarray:
    rhs_values = _project_homogeneous_correction(rhs, level=level)
    flat_rhs = jnp.ravel(rhs_values)
    n_values = flat_rhs.shape[0]
    identity = jnp.eye(n_values, dtype=jnp.float64)

    def apply_basis(column: jnp.ndarray) -> jnp.ndarray:
        basis = jnp.reshape(column, level.shape)
        basis = _project_homogeneous_correction(basis, level=level)
        return jnp.ravel(_project_homogeneous_correction(_mg_apply_negative_perp_laplacian(basis, level), level=level))

    matrix = jax.vmap(apply_basis, in_axes=1, out_axes=1)(identity)
    active_mask = jnp.ravel(jnp.abs(_project_homogeneous_correction(jnp.ones(level.shape, dtype=jnp.float64), level=level)) > 0.0)
    active_matrix_mask = active_mask[:, None] & active_mask[None, :]
    matrix = jnp.where(active_matrix_mask, matrix, identity)
    flat_rhs = jnp.where(active_mask, flat_rhs, 0.0)

    if level.has_nullspace:
        weights = jnp.ravel(_cell_volume_weights(level.geometry))
        weights = weights / jnp.maximum(jnp.sum(weights), 1.0e-30)
        constant = jnp.ones_like(weights)
        matrix = matrix + jnp.outer(constant, weights)
        flat_rhs = flat_rhs - jnp.sum(weights * flat_rhs)

    solution = jnp.linalg.solve(matrix, flat_rhs)
    solution = jnp.reshape(solution, level.shape)
    return _project_homogeneous_correction(solution, level=level)


def _mg_vcycle(
    level_index: int,
    x: jnp.ndarray,
    rhs: jnp.ndarray,
    hierarchy: PerpLaplacianMgHierarchy,
) -> jnp.ndarray:
    level = hierarchy.levels[level_index]
    x = _project_homogeneous_correction(x, level=level)
    rhs = _project_homogeneous_correction(rhs, level=level)

    if level_index == len(hierarchy.levels) - 1:
        if int(level.shape[0] * level.shape[1] * level.shape[2]) <= int(hierarchy.direct_coarse_size):
            return _direct_coarse_solve(rhs, level)
        return _smooth(x, rhs, level=level, nsweeps=hierarchy.coarse_smooth, hierarchy=hierarchy)

    x = _smooth(x, rhs, level=level, nsweeps=hierarchy.pre_smooth, hierarchy=hierarchy)
    residual = rhs - _mg_apply_negative_perp_laplacian(x, level)
    residual = _project_homogeneous_correction(residual, level=level)

    coarse_level = hierarchy.levels[level_index + 1]
    coarse_rhs = _restrict_residual_jweighted(residual, level, coarse_level)
    coarse_rhs = _project_homogeneous_correction(coarse_rhs, level=coarse_level)
    coarse_error = jnp.zeros_like(coarse_rhs)
    coarse_error = _mg_vcycle(level_index + 1, coarse_error, coarse_rhs, hierarchy)
    fine_correction = _prolong_field(coarse_error, coarse_level, level)
    fine_correction = _project_homogeneous_correction(fine_correction, level=level)
    x = x + fine_correction
    x = _smooth(x, rhs, level=level, nsweeps=hierarchy.post_smooth, hierarchy=hierarchy)
    return _project_homogeneous_correction(x, level=level)


def _build_mg_level(
    geometry: FciGeometry3D,
    *,
    stencil_builder: ConservativeStencilBuilder,
    face_bc: BoundaryFaceBC3D,
    regular_face_geometry: RegularFaceGeometry3D,
    cut_wall_geometry: CutWallGeometry3D,
    cut_wall_bc: CutWallBC3D,
    periodic_axes: tuple[bool, bool, bool],
    axis_regular_axes: tuple[bool, bool, bool],
    b_floor: float,
) -> PerpLaplacianMgLevel:
    has_dirichlet = _has_dirichlet_regular_faces(face_bc) or _has_dirichlet_cut_walls(cut_wall_bc)
    has_nullspace = not has_dirichlet
    face_projectors = build_perp_laplacian_face_projectors(
        geometry,
        b_floor=b_floor,
        axis_regular_axes=axis_regular_axes,
    )
    return PerpLaplacianMgLevel(
        geometry=geometry,
        stencil_builder=stencil_builder,
        face_bc=face_bc,
        regular_face_geometry=regular_face_geometry,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        face_projectors=face_projectors,
        diag_inv=_build_approx_diag_inv(geometry, face_bc, regular_face_geometry, face_projectors),
        periodic_axes=periodic_axes,
        has_dirichlet=has_dirichlet,
        has_nullspace=has_nullspace,
    )


def build_perp_laplacian_mg_hierarchy(
    geometry: FciGeometry3D,
    stencil_builder: ConservativeStencilBuilder = build_conservative_stencil_from_field,
    *,
    face_bc: BoundaryFaceBC3D | None = None,
    regular_face_geometry: RegularFaceGeometry3D | None = None,
    cut_wall_geometry: CutWallGeometry3D | None = None,
    cut_wall_bc: CutWallBC3D | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    max_levels: int | None = None,
    require_even_axis_sizes: tuple[bool, bool, bool] = (False, False, False),
    pre_smooth: int = 2,
    post_smooth: int = 2,
    coarse_smooth: int = 16,
    omega_jacobi: float = 0.65,
    smoother: Literal["jacobi", "chebyshev"] = "chebyshev",
    chebyshev_order: int = 2,
    spectral_radius_estimate: float | None = None,
    direct_coarse_size: int = 512,
    b_floor: float = 1.0e-30,
) -> PerpLaplacianMgHierarchy:
    """Build a reusable regular-face V-cycle hierarchy for ``-L_perp``."""

    if not isinstance(stencil_builder, ConservativeStencilBuilder):
        raise TypeError("stencil_builder must be a ConservativeStencilBuilder instance")
    periodic_axes = tuple(bool(value) for value in periodic_axes)
    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    regular_face_geometry = regular_face_geometry or RegularFaceGeometry3D.unit(geometry)
    face_bc = face_bc or BoundaryFaceBC3D.empty(regular_face_geometry)
    cut_wall_geometry = cut_wall_geometry or CutWallGeometry3D.empty()
    cut_wall_bc = cut_wall_bc or CutWallBC3D.empty()
    if cut_wall_geometry.n_wall_faces or cut_wall_bc.n_wall_faces:
        raise NotImplementedError("multigrid coarsening for non-empty cut-wall payloads is not implemented")
    require_even_axis_sizes = tuple(bool(value) for value in require_even_axis_sizes)
    if len(require_even_axis_sizes) != 3:
        raise ValueError(f"require_even_axis_sizes must have length 3, got {require_even_axis_sizes}")

    levels: list[PerpLaplacianMgLevel] = []
    current_geometry = geometry
    current_face_bc = face_bc
    current_regular_face_geometry = regular_face_geometry
    current_cut_wall_geometry = cut_wall_geometry
    current_cut_wall_bc = cut_wall_bc
    while True:
        levels.append(
            _build_mg_level(
                current_geometry,
                stencil_builder=stencil_builder,
                face_bc=current_face_bc,
                regular_face_geometry=current_regular_face_geometry,
                cut_wall_geometry=current_cut_wall_geometry,
                cut_wall_bc=current_cut_wall_bc,
                periodic_axes=periodic_axes,
                axis_regular_axes=axis_regular_axes,
                b_floor=b_floor,
            )
        )
        if max_levels is not None and len(levels) >= int(max_levels):
            break
        if not _can_coarsen_shape(current_geometry.shape, periodic_axes=periodic_axes):
            break
        next_shape = tuple(int(size // 2) for size in current_geometry.shape)
        if any(require_even_axis_sizes[axis] and (next_shape[axis] % 2) for axis in range(3)):
            break
        current_geometry = _coarsen_geometry(current_geometry)
        current_face_bc = _coarsen_face_bc(current_face_bc)
        current_regular_face_geometry = _coarsen_regular_face_geometry(current_regular_face_geometry)
        current_cut_wall_geometry = CutWallGeometry3D.empty()
        current_cut_wall_bc = CutWallBC3D.empty()

    return PerpLaplacianMgHierarchy(
        levels=tuple(levels),
        pre_smooth=int(pre_smooth),
        post_smooth=int(post_smooth),
        coarse_smooth=int(coarse_smooth),
        omega_jacobi=float(omega_jacobi),
        smoother=smoother,
        chebyshev_order=int(chebyshev_order),
        spectral_radius_estimate=spectral_radius_estimate,
        direct_coarse_size=int(direct_coarse_size),
    )


def build_perp_laplacian_solver_mg_hierarchy(
    geometry: FciGeometry3D,
    stencil_builder: ConservativeStencilBuilder = build_conservative_stencil_from_field,
    *,
    face_bc: BoundaryFaceBC3D | None = None,
    cut_wall_bc: CutWallBC3D | None = None,
    lifted: bool = False,
    regular_face_geometry: RegularFaceGeometry3D | None = None,
    cut_wall_geometry: CutWallGeometry3D | None = None,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    max_levels: int | None = None,
    require_even_axis_sizes: tuple[bool, bool, bool] = (False, True, False),
    pre_smooth: int = 2,
    post_smooth: int = 2,
    coarse_smooth: int = 16,
    omega_jacobi: float = 0.65,
    smoother: Literal["jacobi", "chebyshev"] = "chebyshev",
    chebyshev_order: int = 2,
    spectral_radius_estimate: float | None = None,
    direct_coarse_size: int = 512,
    b_floor: float = 1.0e-30,
) -> PerpLaplacianMgHierarchy:
    """Build an MG hierarchy for the exact linear operator used by the solver."""

    regular_face_geometry = regular_face_geometry or RegularFaceGeometry3D.unit(geometry)
    face_bc = face_bc or BoundaryFaceBC3D.empty(regular_face_geometry)
    cut_wall_bc = cut_wall_bc or CutWallBC3D.empty()
    if lifted:
        linear_face_bc = _dirichlet_lift_correction_face_bc(face_bc)
        linear_cut_wall_bc = _dirichlet_lift_correction_cut_wall_bc(cut_wall_bc)
    else:
        linear_face_bc, linear_cut_wall_bc = _homogeneous_boundary_payload(face_bc, cut_wall_bc)
    return build_perp_laplacian_mg_hierarchy(
        geometry,
        stencil_builder,
        face_bc=linear_face_bc,
        regular_face_geometry=regular_face_geometry,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=linear_cut_wall_bc,
        periodic_axes=periodic_axes,
        axis_regular_axes=axis_regular_axes,
        max_levels=max_levels,
        require_even_axis_sizes=require_even_axis_sizes,
        pre_smooth=pre_smooth,
        post_smooth=post_smooth,
        coarse_smooth=coarse_smooth,
        omega_jacobi=omega_jacobi,
        smoother=smoother,
        chebyshev_order=chebyshev_order,
        spectral_radius_estimate=spectral_radius_estimate,
        direct_coarse_size=direct_coarse_size,
        b_floor=b_floor,
    )


def _face_bc_equal(left: BoundaryFaceBC3D, right: BoundaryFaceBC3D) -> bool:
    return (
        bool(jnp.array_equal(left.kind_x, right.kind_x))
        and bool(jnp.array_equal(left.kind_y, right.kind_y))
        and bool(jnp.array_equal(left.kind_z, right.kind_z))
        and bool(jnp.array_equal(left.mask_x, right.mask_x))
        and bool(jnp.array_equal(left.mask_y, right.mask_y))
        and bool(jnp.array_equal(left.mask_z, right.mask_z))
        and bool(jnp.allclose(left.value_x, right.value_x, rtol=0.0, atol=0.0))
        and bool(jnp.allclose(left.value_y, right.value_y, rtol=0.0, atol=0.0))
        and bool(jnp.allclose(left.value_z, right.value_z, rtol=0.0, atol=0.0))
    )


def _face_bc_values_are_zero(face_bc: BoundaryFaceBC3D) -> bool:
    return (
        bool(jnp.allclose(face_bc.value_x, 0.0, rtol=0.0, atol=0.0))
        and bool(jnp.allclose(face_bc.value_y, 0.0, rtol=0.0, atol=0.0))
        and bool(jnp.allclose(face_bc.value_z, 0.0, rtol=0.0, atol=0.0))
    )


def _cut_wall_bc_values_are_zero(cut_wall_bc: CutWallBC3D) -> bool:
    return bool(jnp.allclose(jnp.asarray(cut_wall_bc.value, dtype=jnp.float64), 0.0, rtol=0.0, atol=0.0))


def _validate_mg_hierarchy_for_linear_operator(
    hierarchy: PerpLaplacianMgHierarchy,
    *,
    geometry: FciGeometry3D,
    periodic_axes: tuple[bool, bool, bool],
    face_bc: BoundaryFaceBC3D,
    cut_wall_bc: CutWallBC3D,
) -> None:
    level0 = hierarchy.levels[0]
    if level0.shape != geometry.shape:
        raise ValueError(f"mg_hierarchy level-0 shape must be {geometry.shape}, got {level0.shape}")
    if tuple(level0.periodic_axes) != tuple(periodic_axes):
        raise ValueError("mg_hierarchy periodic_axes must match the inverse solver")
    if level0.cut_wall_geometry.n_wall_faces or level0.cut_wall_bc.n_wall_faces or cut_wall_bc.n_wall_faces:
        raise NotImplementedError("multigrid preconditioning is only supported for empty cut-wall payloads")
    if not _face_bc_values_are_zero(face_bc) or not _cut_wall_bc_values_are_zero(cut_wall_bc):
        raise ValueError("multigrid preconditioning requires zero-valued linear boundary payloads")
    if not _face_bc_equal(level0.face_bc, face_bc):
        raise ValueError("mg_hierarchy face_bc must match the inverse solver linear boundary payload")


def mg_apply_preconditioner(rhs: jnp.ndarray, hierarchy: PerpLaplacianMgHierarchy) -> jnp.ndarray:
    """Apply one multigrid V-cycle as a Lineax-compatible preconditioner."""

    if not isinstance(hierarchy, PerpLaplacianMgHierarchy):
        raise TypeError("hierarchy must be a PerpLaplacianMgHierarchy")
    level0 = hierarchy.levels[0]
    rhs_values = jnp.asarray(rhs, dtype=jnp.float64)
    if rhs_values.shape != level0.shape:
        raise ValueError(f"rhs must have shape {level0.shape}, got {rhs_values.shape}")
    rhs_values = _project_homogeneous_correction(rhs_values, level=level0)
    return _mg_vcycle(0, jnp.zeros_like(rhs_values), rhs_values, hierarchy)


class PerpLaplacianInverseSolver:
    """Reusable Lineax solve adapter for repeated perpendicular-Laplacian inversions.

    The object builds one stable jitted solve closure per geometry/operator
    payload. Stage-dependent RHS, initial guess, and boundary values remain
    dynamic inputs to the cached solve. Nonzero regular-face and cut-wall
    values are lifted out of the Lineax operator so the matvec remains linear.
    """

    def __init__(
        self,
        geometry: FciGeometry3D,
        stencil_builder: ConservativeStencilBuilder,
        *,
        tol: float = 1.0e-6,
        maxiter: int = 50,
        restart: int = 50,
        face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
        regular_face_geometry: RegularFaceGeometry3D | None = None,
        cut_wall_geometry: CutWallGeometry3D | None = None,
        cut_wall_bc: CutWallBC3D | None = None,
        periodic_axes: tuple[bool, bool, bool] = (False, True, True),
        axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
        b_floor: float = 1.0e-30,
        jacobian_floor: float = 1.0e-30,
        project_mean_zero: bool | None = None,
        target_mean_phi: object | None = None,
        pin_point: tuple[int, int, int] | None = None,
        pin_value: float = 0.0,
        regularization_epsilon: float = 0.0,
        mg_hierarchy: PerpLaplacianMgHierarchy | None = None,
        gmres_debug: bool = False,
        check_residual: bool = True,
        stagnation_iters: int = 20,
    ) -> None:
        if lx is None:
            raise ImportError("lineax is required to invert the perpendicular Laplacian")
        if not isinstance(stencil_builder, ConservativeStencilBuilder):
            raise TypeError("stencil_builder must be a ConservativeStencilBuilder instance")
        self.geometry = geometry
        self.stencil_builder = stencil_builder
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        self.restart = int(restart)
        self.face_projectors = face_projectors or build_perp_laplacian_face_projectors(
            geometry,
            b_floor=b_floor,
            axis_regular_axes=axis_regular_axes,
        )
        self.regular_face_geometry = regular_face_geometry or RegularFaceGeometry3D.unit(geometry)
        self.cut_wall_geometry = cut_wall_geometry or CutWallGeometry3D.empty()
        self.cut_wall_bc = cut_wall_bc or CutWallBC3D.empty()
        self.periodic_axes = tuple(bool(value) for value in periodic_axes)
        self.axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
        self.b_floor = float(b_floor)
        self.jacobian_floor = float(jacobian_floor)
        self.project_mean_zero = bool(project_mean_zero) if project_mean_zero is not None else False
        self.target_mean_phi = target_mean_phi
        self.pin_point = tuple(int(index) for index in pin_point) if pin_point is not None else None
        self.pin_value = float(pin_value)
        self.regularization_epsilon = float(regularization_epsilon)
        if self.regularization_epsilon < 0.0:
            raise ValueError("regularization_epsilon must be non-negative")
        if mg_hierarchy is not None and self.regularization_epsilon != 0.0:
            raise ValueError("mg_hierarchy does not include regularization_epsilon; use regularization_epsilon=0.0")
        if self.pin_point is not None and len(self.pin_point) != 3:
            raise ValueError("pin_point must be a 3-tuple of integer indices")
        if self.pin_point is not None:
            if mg_hierarchy is not None:
                raise ValueError("mg_hierarchy does not include pinned rows; use pin_point=None")
            for axis_index, axis_size in zip(self.pin_point, self.geometry.shape):
                if not 0 <= axis_index < axis_size:
                    raise ValueError(
                        "pin_point indices must lie inside the geometry shape; "
                        f"got {self.pin_point} for shape {self.geometry.shape}"
                    )
        self.mg_hierarchy = mg_hierarchy
        self.gmres_debug = bool(gmres_debug)
        self.check_residual = bool(check_residual)
        self.stagnation_iters = int(stagnation_iters)
        # `throw` controls Python-side solver error handling and must stay static
        # under JIT so lineax can branch on it safely.
        self._solve_jit = jax.jit(self._solve_impl, static_argnums=(4,))
        self._solve_lifted_jit = jax.jit(self._solve_lifted_impl, static_argnums=(7,))

    def _apply_A(
        self,
        phi: jnp.ndarray,
        face_bc: BoundaryFaceBC3D,
        cut_wall_bc: CutWallBC3D,
        project_mean_zero: bool,
    ) -> jnp.ndarray:
        values = jnp.asarray(phi, dtype=jnp.float64)
        if project_mean_zero:
            values = _remove_weighted_mean(values, self.geometry)
        local = _build_global_conservative_stencil_compat(
            self.stencil_builder,
            values,
            self.geometry,
            periodic_axes=self.periodic_axes,
            face_bc=face_bc,
        )
        result = -perp_laplacian_conservative_op(
            local,
            self.geometry,
            face_projectors=self.face_projectors,
            face_bc=face_bc,
            regular_face_geometry=self.regular_face_geometry,
            cut_wall_geometry=self.cut_wall_geometry,
            cut_wall_bc=cut_wall_bc,
            periodic_axes=self.periodic_axes,
            axis_regular_axes=self.axis_regular_axes,
            b_floor=self.b_floor,
            jacobian_floor=self.jacobian_floor,
        )
        if self.regularization_epsilon != 0.0:
            result = result + self.regularization_epsilon * values
        if project_mean_zero:
            result = _remove_weighted_mean(result, self.geometry)
        return result

    def _solve_impl(
        self,
        omega: jnp.ndarray,
        phi_guess: jnp.ndarray,
        face_bc: BoundaryFaceBC3D,
        cut_wall_bc: CutWallBC3D,
        throw: bool,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        rhs = jnp.asarray(omega, dtype=jnp.float64)
        guess = jnp.asarray(phi_guess, dtype=jnp.float64)
        rhs_is_finite = jnp.all(jnp.isfinite(rhs))
        guess_is_finite = jnp.all(jnp.isfinite(guess))
        compatibility_boundary_source = self._apply_A(jnp.zeros_like(rhs), face_bc, cut_wall_bc, False)
        compatibility_rhs = rhs - compatibility_boundary_source
        compatibility_rhs_mean = _weighted_mean(compatibility_rhs, self.geometry)
        compatibility_rhs_l2 = _weighted_l2(compatibility_rhs, self.geometry)
        compatibility_rhs_ratio = jnp.abs(compatibility_rhs_mean) / jnp.maximum(compatibility_rhs_l2, 1.0e-30)
        project_mean_zero = bool(self.project_mean_zero)
        if project_mean_zero:
            rhs = _remove_weighted_mean(rhs, self.geometry)
            guess = _remove_weighted_mean(guess, self.geometry)
        if self.pin_point is not None:
            guess = guess.at[self.pin_point].set(self.pin_value)

        homogeneous_face_bc, homogeneous_cut_wall_bc = _homogeneous_boundary_payload(face_bc, cut_wall_bc)
        boundary_source = self._apply_A(jnp.zeros_like(rhs), face_bc, cut_wall_bc, project_mean_zero)
        linear_rhs = rhs - boundary_source
        if project_mean_zero:
            linear_rhs = _remove_weighted_mean(linear_rhs, self.geometry)
        projected_rhs_mean = _weighted_mean(linear_rhs, self.geometry)
        projected_rhs_l2 = _weighted_l2(linear_rhs, self.geometry)
        projected_rhs_ratio = jnp.abs(projected_rhs_mean) / jnp.maximum(projected_rhs_l2, 1.0e-30)
        if self.pin_point is not None:
            linear_rhs = linear_rhs.at[self.pin_point].set(self.pin_value)

        def apply_A(phi: jnp.ndarray) -> jnp.ndarray:
            values = self._apply_A(phi, homogeneous_face_bc, homogeneous_cut_wall_bc, project_mean_zero)
            if self.pin_point is not None:
                values = values.at[self.pin_point].set(phi[self.pin_point])
            return values

        structure = jax.ShapeDtypeStruct(self.geometry.shape, rhs.dtype)
        operator = lx.FunctionLinearOperator(apply_A, structure)
        solver = lx.GMRES(
            max_steps=self.maxiter,
            restart=self.restart,
            stagnation_iters=self.stagnation_iters,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        solve_options: dict[str, object] = {"y0": guess}
        if self.mg_hierarchy is not None:
            solve_options["preconditioner"] = lx.FunctionLinearOperator(
                lambda residual: mg_apply_preconditioner(residual, self.mg_hierarchy),
                structure,
            )
        solve = lx.linear_solve(operator, linear_rhs, solver, options=solve_options, throw=throw)
        phi = solve.value
        if project_mean_zero:
            phi = _remove_weighted_mean(phi, self.geometry)
        if self.target_mean_phi is not None:
            phi = _set_weighted_mean(phi, self.geometry, self.target_mean_phi)
        if self.pin_point is not None:
            phi = phi.at[self.pin_point].set(self.pin_value)
        phi_is_finite = jnp.all(jnp.isfinite(phi))
        final_residual = self._apply_A(phi, face_bc, cut_wall_bc, project_mean_zero) - rhs
        if self.pin_point is not None:
            final_residual = final_residual.at[self.pin_point].set(phi[self.pin_point] - self.pin_value)
        final_residual_l2 = jnp.linalg.norm(final_residual)
        final_residual_linf = jnp.max(jnp.abs(final_residual))
        rhs_norm = jnp.linalg.norm(rhs)
        final_residual_rel_l2 = final_residual_l2 / (rhs_norm + 1.0e-30)
        stats = getattr(solve, "stats", {})
        num_steps = stats.get("num_steps", jnp.asarray(-1, dtype=jnp.int32)) if isinstance(stats, dict) else jnp.asarray(-1, dtype=jnp.int32)
        return (
            phi,
            final_residual_l2,
            final_residual_linf,
            rhs_norm,
            final_residual_rel_l2,
            num_steps,
            rhs_is_finite,
            guess_is_finite,
            phi_is_finite,
            compatibility_rhs_mean,
            compatibility_rhs_l2,
            compatibility_rhs_ratio,
            projected_rhs_mean,
            projected_rhs_l2,
            projected_rhs_ratio,
        )

    def _solve_lifted_impl(
        self,
        omega: jnp.ndarray,
        phi_guess: jnp.ndarray,
        face_bc: BoundaryFaceBC3D,
        cut_wall_bc: CutWallBC3D,
        phi_lift: jnp.ndarray,
        correction_face_bc: BoundaryFaceBC3D,
        correction_cut_wall_bc: CutWallBC3D,
        throw: bool,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        rhs = jnp.asarray(omega, dtype=jnp.float64)
        guess = jnp.asarray(phi_guess, dtype=jnp.float64)
        lift = jnp.asarray(phi_lift, dtype=jnp.float64)
        rhs_is_finite = jnp.all(jnp.isfinite(rhs))
        guess_is_finite = jnp.all(jnp.isfinite(guess))
        lift_is_finite = jnp.all(jnp.isfinite(lift))
        project_mean_zero = bool(self.project_mean_zero)
        if project_mean_zero:
            rhs = _remove_weighted_mean(rhs, self.geometry)
        correction_guess = guess - lift
        if project_mean_zero:
            correction_guess = _remove_weighted_mean(correction_guess, self.geometry)

        lift_source = self._apply_A(lift, face_bc, cut_wall_bc, project_mean_zero)
        rhs_u = rhs - lift_source
        if project_mean_zero:
            rhs_u = _remove_weighted_mean(rhs_u, self.geometry)
        rhs_u_mean = _weighted_mean(rhs_u, self.geometry)
        rhs_u_l2_weighted = _weighted_l2(rhs_u, self.geometry)
        rhs_u_ratio = jnp.abs(rhs_u_mean) / jnp.maximum(rhs_u_l2_weighted, 1.0e-30)
        correction_pin_value = jnp.asarray(self.pin_value, dtype=jnp.float64)
        if self.pin_point is not None:
            correction_pin_value = correction_pin_value - lift[self.pin_point]
            correction_guess = correction_guess.at[self.pin_point].set(correction_pin_value)
            rhs_u = rhs_u.at[self.pin_point].set(correction_pin_value)

        def apply_A(u: jnp.ndarray) -> jnp.ndarray:
            values = self._apply_A(u, correction_face_bc, correction_cut_wall_bc, project_mean_zero)
            if self.pin_point is not None:
                values = values.at[self.pin_point].set(u[self.pin_point])
            return values

        structure = jax.ShapeDtypeStruct(self.geometry.shape, rhs.dtype)
        operator = lx.FunctionLinearOperator(apply_A, structure)
        solver = lx.GMRES(
            max_steps=self.maxiter,
            restart=self.restart,
            stagnation_iters=self.stagnation_iters,
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        solve_options: dict[str, object] = {"y0": correction_guess}
        if self.mg_hierarchy is not None:
            solve_options["preconditioner"] = lx.FunctionLinearOperator(
                lambda residual: mg_apply_preconditioner(residual, self.mg_hierarchy),
                structure,
            )
        solve = lx.linear_solve(operator, rhs_u, solver, options=solve_options, throw=throw)
        correction = solve.value
        if project_mean_zero:
            correction = _remove_weighted_mean(correction, self.geometry)
        phi = lift + correction
        if self.target_mean_phi is not None:
            phi = _set_weighted_mean(phi, self.geometry, self.target_mean_phi)
            correction = phi - lift
        if self.pin_point is not None:
            phi = phi.at[self.pin_point].set(self.pin_value)
            correction = correction.at[self.pin_point].set(correction_pin_value)
        phi_is_finite = jnp.all(jnp.isfinite(phi))

        correction_residual = self._apply_A(correction, correction_face_bc, correction_cut_wall_bc, project_mean_zero) - rhs_u
        physical_residual = self._apply_A(phi, face_bc, cut_wall_bc, project_mean_zero) - rhs
        if self.pin_point is not None:
            correction_residual = correction_residual.at[self.pin_point].set(correction[self.pin_point] - correction_pin_value)
            physical_residual = physical_residual.at[self.pin_point].set(phi[self.pin_point] - self.pin_value)
        correction_residual_l2 = jnp.linalg.norm(correction_residual)
        correction_residual_linf = jnp.max(jnp.abs(correction_residual))
        rhs_u_norm = jnp.linalg.norm(rhs_u)
        correction_residual_rel_l2 = correction_residual_l2 / (rhs_u_norm + 1.0e-30)
        physical_residual_l2 = jnp.linalg.norm(physical_residual)
        physical_residual_linf = jnp.max(jnp.abs(physical_residual))
        physical_rhs_norm = jnp.linalg.norm(rhs)
        lift_source_norm = jnp.linalg.norm(lift_source)
        stats = getattr(solve, "stats", {})
        num_steps = stats.get("num_steps", jnp.asarray(-1, dtype=jnp.int32)) if isinstance(stats, dict) else jnp.asarray(-1, dtype=jnp.int32)
        return (
            phi,
            correction_residual_l2,
            correction_residual_linf,
            rhs_u_norm,
            correction_residual_rel_l2,
            physical_residual_l2,
            physical_residual_linf,
            physical_rhs_norm,
            lift_source_norm,
            num_steps,
            rhs_is_finite,
            guess_is_finite,
            lift_is_finite,
            phi_is_finite,
            rhs_u_mean,
            rhs_u_l2_weighted,
            rhs_u_ratio,
        )

    def __call__(
        self,
        omega: jnp.ndarray,
        *,
        phi_guess: jnp.ndarray | None = None,
        face_bc: BoundaryFaceBC3D | None = None,
        cut_wall_bc: CutWallBC3D | None = None,
        phi_lift: jnp.ndarray | None = None,
        correction_face_bc: BoundaryFaceBC3D | None = None,
        correction_cut_wall_bc: CutWallBC3D | None = None,
        throw: bool = False,
        return_diagnostics: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, dict[str, object]]:
        rhs = jnp.asarray(omega, dtype=jnp.float64)
        if rhs.shape != self.geometry.shape:
            raise ValueError(f"omega must have shape {self.geometry.shape}, got {rhs.shape}")
        lift = None
        if phi_lift is not None:
            lift = jnp.asarray(phi_lift, dtype=jnp.float64)
            if lift.shape != self.geometry.shape:
                raise ValueError(f"phi_lift must have shape {self.geometry.shape}, got {lift.shape}")
            if phi_guess is None:
                phi_guess = lift
            else:
                phi_guess = jnp.asarray(phi_guess, dtype=jnp.float64)
        else:
            if phi_guess is None:
                phi_guess = jnp.zeros_like(rhs)
            else:
                phi_guess = jnp.asarray(phi_guess, dtype=jnp.float64)
        if phi_guess.shape != self.geometry.shape:
            raise ValueError(f"phi_guess must have shape {self.geometry.shape}, got {phi_guess.shape}")
        if face_bc is None:
            face_bc = BoundaryFaceBC3D.empty(self.regular_face_geometry)
        if cut_wall_bc is None:
            cut_wall_bc = self.cut_wall_bc
        elif not isinstance(cut_wall_bc, CutWallBC3D):
            raise TypeError("cut_wall_bc must be a CutWallBC3D instance")
        if phi_lift is not None:
            if correction_face_bc is None:
                correction_face_bc = _dirichlet_lift_correction_face_bc(face_bc)
            elif not isinstance(correction_face_bc, BoundaryFaceBC3D):
                raise TypeError("correction_face_bc must be a BoundaryFaceBC3D instance")
            if correction_cut_wall_bc is None:
                correction_cut_wall_bc = _dirichlet_lift_correction_cut_wall_bc(cut_wall_bc)
            elif not isinstance(correction_cut_wall_bc, CutWallBC3D):
                raise TypeError("correction_cut_wall_bc must be a CutWallBC3D instance")
            if self.mg_hierarchy is not None:
                _validate_mg_hierarchy_for_linear_operator(
                    self.mg_hierarchy,
                    geometry=self.geometry,
                    periodic_axes=self.periodic_axes,
                    face_bc=correction_face_bc,
                    cut_wall_bc=correction_cut_wall_bc,
                )
            (
                phi,
                residual_l2,
                residual_linf,
                rhs_u_norm,
                residual_rel_l2,
                physical_residual_l2,
                physical_residual_linf,
                physical_rhs_norm,
                lift_source_norm,
                num_steps,
                rhs_is_finite,
                guess_is_finite,
                lift_is_finite,
                phi_is_finite,
                rhs_u_mean,
                rhs_u_l2_weighted,
                rhs_u_ratio,
            ) = self._solve_lifted_jit(
                rhs,
                phi_guess,
                face_bc,
                cut_wall_bc,
                lift,
                correction_face_bc,
                correction_cut_wall_bc,
                throw,
            )
            if self.gmres_debug:
                print("PerpLaplacianInverseSolver lifted GMRES num_steps:", int(num_steps))
                print("PerpLaplacianInverseSolver lifted correction residual l2:", float(residual_l2))
                print("PerpLaplacianInverseSolver lifted correction residual linf:", float(residual_linf))
                print("PerpLaplacianInverseSolver lifted rhs_u l2:", float(rhs_u_norm))
                print("PerpLaplacianInverseSolver lifted correction residual relative l2:", float(residual_rel_l2))
                print("PerpLaplacianInverseSolver lifted physical rhs l2:", float(physical_rhs_norm))
                print("PerpLaplacianInverseSolver lifted lift source l2:", float(lift_source_norm))
                print("PerpLaplacianInverseSolver lifted physical residual l2:", float(physical_residual_l2))
                print("PerpLaplacianInverseSolver lifted physical residual linf:", float(physical_residual_linf))
                print("PerpLaplacianInverseSolver lifted rhs_u mean J:", float(rhs_u_mean))
                print("PerpLaplacianInverseSolver lifted rhs_u l2 J:", float(rhs_u_l2_weighted))
                print("PerpLaplacianInverseSolver lifted rhs_u ratio:", float(rhs_u_ratio))
                print("PerpLaplacianInverseSolver lifted input finite flags:", {
                    "rhs": bool(rhs_is_finite),
                    "phi_guess": bool(guess_is_finite),
                    "phi_lift": bool(lift_is_finite),
                    "phi": bool(phi_is_finite),
                })
            if self.check_residual:
                if (
                    not jnp.isfinite(residual_l2)
                    or not jnp.isfinite(residual_linf)
                    or not jnp.isfinite(rhs_u_norm)
                    or not jnp.isfinite(physical_residual_l2)
                ):
                    raise RuntimeError(
                        "PerpLaplacianInverseSolver lifted solve produced a non-finite residual: "
                        f"correction_l2={float(residual_l2):g}, correction_linf={float(residual_linf):g}, "
                        f"rhs_u_l2={float(rhs_u_norm):g}, physical_l2={float(physical_residual_l2):g}, "
                        f"num_steps={int(num_steps)}, rhs_finite={bool(rhs_is_finite)}, "
                        f"phi_guess_finite={bool(guess_is_finite)}, phi_lift_finite={bool(lift_is_finite)}, "
                        f"phi_finite={bool(phi_is_finite)}"
                    )
                correction_atol = float(self.tol)
                correction_rtol = float(self.tol)
                correction_limit = max(correction_atol, correction_rtol * max(float(rhs_u_norm), 1.0))
                if float(residual_l2) > correction_limit:
                    raise RuntimeError(
                        "PerpLaplacianInverseSolver lifted GMRES residual too large: "
                        f"correction_l2={float(residual_l2):g}, correction_rel_l2={float(residual_rel_l2):g}, "
                        f"correction_linf={float(residual_linf):g}, rhs_u_l2={float(rhs_u_norm):g}, "
                        f"physical_l2={float(physical_residual_l2):g}, physical_linf={float(physical_residual_linf):g}, "
                        f"lift_source_l2={float(lift_source_norm):g}, rhsUratio={float(rhs_u_ratio):g}, "
                        f"limit={correction_limit:g}"
                    )
            if return_diagnostics:
                return phi, {
                    "final_residual_l2": float(residual_l2),
                    "final_residual_linf": float(residual_linf),
                    "rhs_l2": float(rhs_u_norm),
                    "final_residual_rel_l2": float(residual_rel_l2),
                    "num_steps": int(num_steps),
                    "physical_rhs_l2": float(physical_rhs_norm),
                    "lift_source_l2": float(lift_source_norm),
                    "rhs_u_l2": float(rhs_u_norm),
                    "correction_residual_l2": float(residual_l2),
                    "correction_residual_rel_l2": float(residual_rel_l2),
                    "physical_residual_l2": float(physical_residual_l2),
                    "physical_residual_linf": float(physical_residual_linf),
                    "rhs_u_mean_J": float(rhs_u_mean),
                    "rhs_u_l2_J": float(rhs_u_l2_weighted),
                    "rhs_u_compatibility_ratio": float(rhs_u_ratio),
                    "rhs_finite": bool(rhs_is_finite),
                    "phi_guess_finite": bool(guess_is_finite),
                    "phi_lift_finite": bool(lift_is_finite),
                    "phi_finite": bool(phi_is_finite),
                    "lifted": True,
                }
            return phi
        if self.mg_hierarchy is not None:
            homogeneous_face_bc, homogeneous_cut_wall_bc = _homogeneous_boundary_payload(face_bc, cut_wall_bc)
            _validate_mg_hierarchy_for_linear_operator(
                self.mg_hierarchy,
                geometry=self.geometry,
                periodic_axes=self.periodic_axes,
                face_bc=homogeneous_face_bc,
                cut_wall_bc=homogeneous_cut_wall_bc,
            )
        (
            phi,
            residual_l2,
            residual_linf,
            rhs_norm,
            residual_rel_l2,
            num_steps,
            rhs_is_finite,
            guess_is_finite,
            phi_is_finite,
            compatibility_rhs_mean,
            compatibility_rhs_l2,
            compatibility_rhs_ratio,
            projected_rhs_mean,
            projected_rhs_l2,
            projected_rhs_ratio,
        ) = self._solve_jit(
            rhs,
            phi_guess,
            face_bc,
            cut_wall_bc,
            throw,
        )
        if self.gmres_debug:
            print("PerpLaplacianInverseSolver GMRES num_steps:", int(num_steps))
            print("PerpLaplacianInverseSolver GMRES final residual l2:", float(residual_l2))
            print("PerpLaplacianInverseSolver GMRES final residual linf:", float(residual_linf))
            print("PerpLaplacianInverseSolver GMRES rhs l2:", float(rhs_norm))
            print("PerpLaplacianInverseSolver GMRES final residual relative l2:", float(residual_rel_l2))
            print("PerpLaplacianInverseSolver pre-projection rhs mean J:", float(compatibility_rhs_mean))
            print("PerpLaplacianInverseSolver pre-projection rhs l2 J:", float(compatibility_rhs_l2))
            print("PerpLaplacianInverseSolver pre-projection rhs ratio:", float(compatibility_rhs_ratio))
            print("PerpLaplacianInverseSolver post-projection rhs mean J:", float(projected_rhs_mean))
            print("PerpLaplacianInverseSolver post-projection rhs l2 J:", float(projected_rhs_l2))
            print("PerpLaplacianInverseSolver post-projection rhs ratio:", float(projected_rhs_ratio))
            print("PerpLaplacianInverseSolver input finite flags:", {
                "rhs": bool(rhs_is_finite),
                "phi_guess": bool(guess_is_finite),
                "phi": bool(phi_is_finite),
            })
        if self.check_residual:
            if (
                not jnp.isfinite(residual_l2)
                or not jnp.isfinite(residual_linf)
                or not jnp.isfinite(rhs_norm)
            ):
                raise RuntimeError(
                    "PerpLaplacianInverseSolver produced a non-finite GMRES residual: "
                    f"l2={float(residual_l2):g}, linf={float(residual_linf):g}, rhs_l2={float(rhs_norm):g}, "
                    f"num_steps={int(num_steps)}, rhs_finite={bool(rhs_is_finite)}, "
                    f"phi_guess_finite={bool(guess_is_finite)}, phi_finite={bool(phi_is_finite)}"
                )
            if float(residual_rel_l2) > max(10.0 * self.tol, 1.0e-12):
                raise RuntimeError(
                    "PerpLaplacianInverseSolver GMRES residual too large: "
                    f"l2={float(residual_l2):g}, rel_l2={float(residual_rel_l2):g}, "
                    f"linf={float(residual_linf):g}, rhsCpre={float(compatibility_rhs_ratio):g}, "
                    f"rhsCpost={float(projected_rhs_ratio):g}"
                )
        if return_diagnostics:
            return phi, {
                "final_residual_l2": float(residual_l2),
                "final_residual_linf": float(residual_linf),
                "rhs_l2": float(rhs_norm),
                "final_residual_rel_l2": float(residual_rel_l2),
                "num_steps": int(num_steps),
                "rhs_mean_J": float(compatibility_rhs_mean),
                "rhs_l2_J": float(compatibility_rhs_l2),
                "rhs_compatibility_ratio": float(compatibility_rhs_ratio),
                "projected_rhs_mean_J": float(projected_rhs_mean),
                "projected_rhs_l2_J": float(projected_rhs_l2),
                "projected_rhs_compatibility_ratio": float(projected_rhs_ratio),
                "rhs_finite": bool(rhs_is_finite),
                "phi_guess_finite": bool(guess_is_finite),
                "phi_finite": bool(phi_is_finite),
            }
        return phi


def _homogeneous_local_face_bc(face_bc: LocalBoundaryFaceBC3D) -> LocalBoundaryFaceBC3D:
    return dataclass_replace(
        face_bc,
        value_x=jnp.zeros_like(face_bc.value_x, dtype=jnp.float64),
        value_y=jnp.zeros_like(face_bc.value_y, dtype=jnp.float64),
        value_z=jnp.zeros_like(face_bc.value_z, dtype=jnp.float64),
    )


def _homogeneous_local_cut_wall_bc(cut_wall_bc: LocalCutWallBC3D) -> LocalCutWallBC3D:
    return LocalCutWallBC3D(
        kind=cut_wall_bc.kind,
        value=jnp.zeros_like(cut_wall_bc.value, dtype=jnp.float64),
        active=cut_wall_bc.active,
        max_wall_faces=cut_wall_bc.max_wall_faces,
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


def _dirichlet_lift_correction_local_cut_wall_bc(
    cut_wall_bc: LocalCutWallBC3D,
) -> LocalCutWallBC3D:
    """Return local cut-wall correction BCs for ``phi = phi_lift + u``."""

    return LocalCutWallBC3D(
        kind=cut_wall_bc.kind,
        value=jnp.where(cut_wall_bc.kind == BC_DIRICHLET, 0.0, cut_wall_bc.value),
        active=cut_wall_bc.active,
        max_wall_faces=cut_wall_bc.max_wall_faces,
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


@_pytree_base
@dataclass(frozen=True)
class LocalPerpLaplacianInverseSolver:
    """SPMD GMRES adapter for local conservative perpendicular-Laplacian inversion."""

    geometry: LocalFciGeometry3D
    domain: LocalDomain3D
    control_volume_geometry: LocalEmbeddedControlVolumeGeometry3D
    control_volume_boundary_bc: LocalControlVolumeBoundaryBC3D
    stencil_builder: LocalConservativeStencilBuilder = (
        build_local_conservative_stencil_from_field
    )
    halo_exchange: HaloExchange3D | None = None
    topology_filler: TopologyHaloFiller3D | None = None
    physical_ghost_filler: PhysicalGhostCellFiller3D | None = None
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None
    face_bc: LocalBoundaryFaceBC3D | None = None
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False)
    b_floor: float = 1.0e-30
    jacobian_floor: float = 1.0e-30
    config: SpmdGmresConfig = SpmdGmresConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, LocalFciGeometry3D):
            raise TypeError("geometry must be a LocalFciGeometry3D instance")
        if not isinstance(self.domain, LocalDomain3D):
            raise TypeError("domain must be a LocalDomain3D instance")
        if self.geometry.layout != self.domain.layout:
            raise ValueError("geometry and domain must share the same HaloLayout3D")
        if not isinstance(self.stencil_builder, LocalConservativeStencilBuilder):
            raise TypeError("stencil_builder must be a LocalConservativeStencilBuilder")
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
        if not isinstance(
            self.control_volume_geometry,
            LocalEmbeddedControlVolumeGeometry3D,
        ):
            raise TypeError(
                "control_volume_geometry must be a "
                "LocalEmbeddedControlVolumeGeometry3D"
            )
        if self.control_volume_geometry.layout != self.geometry.layout:
            raise ValueError("control_volume_geometry must share geometry.layout")
        if not isinstance(
            self.control_volume_boundary_bc,
            LocalControlVolumeBoundaryBC3D,
        ):
            raise TypeError(
                "control_volume_boundary_bc must be a "
                "LocalControlVolumeBoundaryBC3D"
            )
        if (
            self.control_volume_boundary_bc.max_rows
            != self.control_volume_geometry.irregular_faces.max_rows
        ):
            raise ValueError(
                "control_volume_boundary_bc must align with irregular face rows"
            )
        if self.face_bc is not None and not isinstance(self.face_bc, LocalBoundaryFaceBC3D):
            raise TypeError("face_bc must be a LocalBoundaryFaceBC3D or None")
        axis_regular_axes = tuple(bool(value) for value in self.axis_regular_axes)
        if len(axis_regular_axes) != 3:
            raise ValueError("axis_regular_axes must have length 3")
        object.__setattr__(self, "axis_regular_axes", axis_regular_axes)
        if not isinstance(self.config, SpmdGmresConfig):
            raise TypeError("config must be a SpmdGmresConfig instance")
        object.__setattr__(self, "b_floor", float(self.b_floor))
        object.__setattr__(self, "jacobian_floor", float(self.jacobian_floor))

    def _default_face_bc(self) -> LocalBoundaryFaceBC3D:
        return self.face_bc or LocalBoundaryFaceBC3D.empty(self.domain.layout)

    def _default_control_volume_boundary_bc(
        self,
    ) -> LocalControlVolumeBoundaryBC3D:
        return self.control_volume_boundary_bc

    def _apply_A(
        self,
        field_owned: jnp.ndarray,
        *,
        face_bc: LocalBoundaryFaceBC3D,
        control_volume_boundary_bc: LocalControlVolumeBoundaryBC3D,
        project_mean_zero: bool,
    ) -> jnp.ndarray:
        active_mask = self.geometry.active_cell_mask_owned
        values = _mask_inactive_owned(field_owned, self.geometry)
        if project_mean_zero:
            values = _spmd_remove_weighted_mean(
                values,
                self.geometry,
                self.domain,
                active_mask,
            )

        storage_values = expand_local_control_volume_owner_field(
            values,
            self.control_volume_geometry.cells,
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

        context = StencilBuilderContext(
            layout=self.domain.layout,
            domain=self.domain,
        )
        local = self.stencil_builder(field_halo, self.geometry, context)
        field_reconstruction = build_local_control_volume_polynomial_from_field(
            field_halo,
            self.geometry,
            self.domain,
            context,
            self.control_volume_geometry,
            control_volume_boundary_bc,
            face_bc,
            halo_exchange=self.halo_exchange,
            topology_filler=self.topology_filler,
        )
        face_projectors = self.face_projectors
        if face_projectors is None:
            face_projectors = build_local_perp_laplacian_face_projectors(
                self.geometry,
                self.domain,
                b_floor=self.b_floor,
                axis_regular_axes=self.axis_regular_axes,
            )
        result = -local_perp_laplacian_conservative_op(
            local,
            self.geometry,
            self.domain,
            face_projectors=face_projectors,
            face_bc=face_bc,
            control_volume_geometry=self.control_volume_geometry,
            boundary_bc=control_volume_boundary_bc,
            field_reconstruction=field_reconstruction,
            axis_regular_axes=self.axis_regular_axes,
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
            )
        return _mask_inactive_owned(result, self.geometry)

    def __call__(
        self,
        rhs_owned: jnp.ndarray,
        *,
        guess_owned: jnp.ndarray | None = None,
        phi_guess_owned: jnp.ndarray | None = None,
        phi_lift_owned: jnp.ndarray | None = None,
        lift_owned: jnp.ndarray | None = None,
        return_diagnostics: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, SpmdGmresInfo]:
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
        active_mask = self.geometry.active_cell_mask_owned
        rhs = _mask_inactive_owned(rhs, self.geometry)
        guess = _mask_inactive_owned(guess, self.geometry)
        if lift is not None:
            lift = jnp.asarray(lift, dtype=jnp.float64)

        if lift is None:
            homogeneous_face_bc = _homogeneous_local_face_bc(face_bc)
            homogeneous_control_volume_boundary_bc = (
                _homogeneous_local_control_volume_boundary_bc(
                    control_volume_boundary_bc
                )
            )
            boundary_source = self._apply_A(
                jnp.zeros_like(rhs),
                face_bc=face_bc,
                control_volume_boundary_bc=control_volume_boundary_bc,
                project_mean_zero=project_mean_zero,
            )
            linear_rhs = _mask_inactive_owned(rhs - boundary_source, self.geometry)
            initial_guess = _mask_inactive_owned(guess, self.geometry)
        else:
            homogeneous_face_bc = _dirichlet_lift_correction_local_face_bc(face_bc)
            homogeneous_control_volume_boundary_bc = (
                _dirichlet_lift_correction_local_control_volume_boundary_bc(
                    control_volume_boundary_bc
                )
            )
            lift_source = self._apply_A(
                lift,
                face_bc=face_bc,
                control_volume_boundary_bc=control_volume_boundary_bc,
                project_mean_zero=project_mean_zero,
            )
            linear_rhs = _mask_inactive_owned(rhs - lift_source, self.geometry)
            initial_guess = _mask_inactive_owned(guess - lift, self.geometry)

        if project_mean_zero:
            linear_rhs = _spmd_remove_weighted_mean(
                linear_rhs,
                self.geometry,
                self.domain,
                active_mask,
            )
            initial_guess = _spmd_remove_weighted_mean(
                initial_guess,
                self.geometry,
                self.domain,
                active_mask,
            )

        def apply_A(field_owned: jnp.ndarray) -> jnp.ndarray:
            return self._apply_A(
                field_owned,
                face_bc=homogeneous_face_bc,
                control_volume_boundary_bc=homogeneous_control_volume_boundary_bc,
                project_mean_zero=project_mean_zero,
            )

        solution, info = spmd_gmres_solve(
            apply_A,
            linear_rhs,
            initial_guess,
            self.geometry,
            self.domain,
            self.config,
            active_cell_mask=active_mask,
        )
        if lift is not None:
            solution = jnp.where(active_mask, lift + solution, lift)
        else:
            solution = _mask_inactive_owned(solution, self.geometry)
        if return_diagnostics:
            return solution, info
        return solution

    def tree_flatten(self):
        children = (
            self.geometry,
            self.domain,
            self.stencil_builder,
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
            halo_exchange,
            topology_filler,
            physical_ghost_filler,
            face_projectors,
            control_volume_geometry,
            control_volume_boundary_bc,
            face_bc,
            config,
        ) = children
        axis_regular_axes, b_floor, jacobian_floor = aux_data
        return cls(
            geometry=geometry,
            domain=domain,
            stencil_builder=stencil_builder,
            halo_exchange=halo_exchange,
            topology_filler=topology_filler,
            physical_ghost_filler=physical_ghost_filler,
            face_projectors=face_projectors,
            control_volume_geometry=control_volume_geometry,
            control_volume_boundary_bc=control_volume_boundary_bc,
            face_bc=face_bc,
            axis_regular_axes=axis_regular_axes,
            b_floor=b_floor,
            jacobian_floor=jacobian_floor,
            config=config,
        )



def _face_average_3d(values: jnp.ndarray, *, axis: int, periodic: bool) -> jnp.ndarray:
    """Return values averaged onto the high face of each cell along one axis."""

    face = 0.5 * (values + jnp.roll(values, -1, axis=axis))
    if periodic:
        return face
    return face.at[_axis_index(axis, -1)].set(values[_axis_index(axis, -1)])


def _face_interpolate_3d_order4(values: jnp.ndarray, *, axis: int, periodic: bool) -> jnp.ndarray:
    """Fourth-order interpolation from nodes to high faces."""

    values = jnp.asarray(values, dtype=jnp.float64)
    if periodic:
        return (-jnp.roll(values, 1, axis=axis) + 9.0 * values + 9.0 * jnp.roll(values, -1, axis=axis) - jnp.roll(values, -2, axis=axis)) / 16.0

    if values.shape[axis] < 4:
        raise ValueError("Fourth-order face interpolation requires at least 4 points along the selected axis")
    ndim = values.ndim
    face = jnp.zeros_like(values)
    face = face.at[_axis_index_nd(axis, 0, ndim)].set(
        (
            5.0 * values[_axis_index_nd(axis, 0, ndim)]
            + 15.0 * values[_axis_index_nd(axis, 1, ndim)]
            - 5.0 * values[_axis_index_nd(axis, 2, ndim)]
            + values[_axis_index_nd(axis, 3, ndim)]
        )
        / 16.0
    )
    face = face.at[_axis_slice_nd(axis, 1, -2, ndim)].set(
        (
            -values[_axis_slice_nd(axis, None, -3, ndim)]
            + 9.0 * values[_axis_slice_nd(axis, 1, -2, ndim)]
            + 9.0 * values[_axis_slice_nd(axis, 2, -1, ndim)]
            - values[_axis_slice_nd(axis, 3, None, ndim)]
        )
        / 16.0
    )
    face = face.at[_axis_index_nd(axis, -2, ndim)].set(
        (
            values[_axis_index_nd(axis, -4, ndim)]
            - 5.0 * values[_axis_index_nd(axis, -3, ndim)]
            + 15.0 * values[_axis_index_nd(axis, -2, ndim)]
            + 5.0 * values[_axis_index_nd(axis, -1, ndim)]
        )
        / 16.0
    )
    return face.at[_axis_index_nd(axis, -1, ndim)].set(values[_axis_index_nd(axis, -1, ndim)])


def _face_derivative_3d_order4(
    values: jnp.ndarray,
    spacing: jnp.ndarray | float,
    *,
    axis: int,
    periodic: bool,
) -> jnp.ndarray:
    """Fourth-order first derivative from nodes to high faces."""

    values = jnp.asarray(values, dtype=jnp.float64)
    h = jnp.asarray(spacing, dtype=jnp.float64)
    if h.ndim == 0:
        h = jnp.ones_like(values) * h
    h = jnp.maximum(h, 1.0e-30)
    if periodic:
        return (jnp.roll(values, 1, axis=axis) - 27.0 * values + 27.0 * jnp.roll(values, -1, axis=axis) - jnp.roll(values, -2, axis=axis)) / (24.0 * h)

    if values.shape[axis] < 4:
        raise ValueError("Fourth-order face derivative requires at least 4 points along the selected axis")
    ndim = values.ndim
    derivative = jnp.zeros_like(values)
    derivative = derivative.at[_axis_index_nd(axis, 0, ndim)].set(
        (
            -23.0 * values[_axis_index_nd(axis, 0, ndim)]
            + 21.0 * values[_axis_index_nd(axis, 1, ndim)]
            + 3.0 * values[_axis_index_nd(axis, 2, ndim)]
            - values[_axis_index_nd(axis, 3, ndim)]
        )
        / (24.0 * h[_axis_index_nd(axis, 0, ndim)])
    )
    derivative = derivative.at[_axis_slice_nd(axis, 1, -2, ndim)].set(
        (
            values[_axis_slice_nd(axis, None, -3, ndim)]
            - 27.0 * values[_axis_slice_nd(axis, 1, -2, ndim)]
            + 27.0 * values[_axis_slice_nd(axis, 2, -1, ndim)]
            - values[_axis_slice_nd(axis, 3, None, ndim)]
        )
        / (24.0 * h[_axis_slice_nd(axis, 1, -2, ndim)])
    )
    return derivative.at[_axis_index_nd(axis, -2, ndim)].set(
        (
            values[_axis_index_nd(axis, -4, ndim)]
            - 3.0 * values[_axis_index_nd(axis, -3, ndim)]
            - 21.0 * values[_axis_index_nd(axis, -2, ndim)]
            + 23.0 * values[_axis_index_nd(axis, -1, ndim)]
        )
        / (24.0 * h[_axis_index_nd(axis, -2, ndim)])
    )


def _face_forward_difference_3d(
    values: jnp.ndarray,
    spacing: jnp.ndarray | float,
    *,
    axis: int,
    periodic: bool,
) -> jnp.ndarray:
    """Return a forward difference interpreted on the high face of each cell."""

    h = jnp.asarray(spacing, dtype=jnp.float64)
    if h.ndim == 0:
        h = jnp.ones_like(values) * h
    face_h = _face_average_3d(h, axis=axis, periodic=periodic)
    difference = (jnp.roll(values, -1, axis=axis) - values) / jnp.maximum(face_h, 1.0e-30)
    if periodic:
        return difference
    last = _axis_index(axis, -1)
    penultimate = _axis_index(axis, -2)
    antepenultimate = _axis_index(axis, -3)
    backward = (
        3.0 * values[last] - 4.0 * values[penultimate] + values[antepenultimate]
    ) / jnp.maximum(2.0 * face_h[last], 1.0e-30)
    return difference.at[last].set(backward)


def _axis_index(axis: int, index: int) -> tuple[object, object, object]:
    slices: list[object] = [slice(None), slice(None), slice(None)]
    slices[axis] = index
    return tuple(slices)
