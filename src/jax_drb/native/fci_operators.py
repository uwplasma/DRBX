from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp

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
from .fci_halo import HaloExchange3D, PhysicalGhostCellFiller3D, TopologyHaloFiller3D
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
    LocalCutWallBC3D,
    LocalCutWallGeometry3D,
    LocalCutWallNormalDerivativeConstructor3D,
    LocalCutWallValueReconstructor3D,
    CutWallBC3D,
    CutWallGeometry3D,
    BoundaryFaceBC3D,
    FaceFluxStencil3D,
    ConservativeStencil3D,
    LocalStencil1D,
    LocalStencil3D,
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

    return _take_stencil_finite_difference(stencil)


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
    return Bmag_halo[geometry.layout.owned_slices_cell] * grad_parallel_q


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

    return jnp.einsum("...i,...i->...", b_contra, df)


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

    return jnp.einsum("...ij,...j->...i", projector, df)


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
    return div_flux / jnp.maximum(J_owned, float(jacobian_floor))


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

    return jnp.sum(b_covariant * cross, axis=-1) / jnp.maximum(
        J_owned,
        float(jacobian_floor),
    )


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
    return jnp.einsum(
        "...i,...i->...",
        curvature_coefficients,
        grad_f,
    )


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

    normal_contra = jnp.asarray(cut_wall_geometry.normal_contra, dtype=jnp.float64)
    normal_cov = jnp.einsum("...ij,...j->...i", jnp.asarray(cut_wall_geometry.g_cov, dtype=jnp.float64), normal_contra)
    g_cell = jnp.einsum("...i,...i->...", normal_contra, grad_cell)
    grad_tangent = grad_cell - g_cell[..., None] * normal_cov

    distance = jnp.asarray(cut_wall_geometry.distance, dtype=jnp.float64)
    safe_distance = jnp.maximum(jnp.abs(distance), 1.0e-30)
    g_dirichlet = (cut_wall_value - f_cell) / safe_distance
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
    return div_flux / jnp.maximum(effective_volume, float(jacobian_floor))


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
    upper_coord = (
        8.0 * upper_value
        - 9.0 * upper_center
        + upper_prev_center
    ) / jnp.maximum(6.0 * upper_distance, 1.0e-30)
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


def local_parallel_flux_div_op(
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    face_bc: LocalBoundaryFaceBC3D | None = None,
    regular_face_geometry: LocalRegularFaceGeometry3D | None = None,
    cell_volume: LocalCellVolumeGeometry3D | None = None,
    cut_wall_geometry: LocalCutWallGeometry3D | None = None,
    cut_wall_bc: LocalCutWallBC3D | None = None,
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

    cut_wall_flux = _build_local_parallel_flux_cut_wall_payload(
        local=local,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        b_floor=b_floor,
    )

    cv_flux = LocalControlVolumeFluxStencil3D(
        regular_flux=FaceFluxStencil3D(x=x_flux, y=y_flux, z=z_flux),
        regular_face_geometry=regular_face_geometry,
        cell_volume=cell_volume,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_flux=cut_wall_flux,
    )
    return local_divergence_conservative_op(
        cv_flux,
        geometry,
        jacobian_floor=jacobian_floor,
    )


def _build_local_cut_wall_flux_payload(
    *,
    local: ConservativeStencil3D,
    cut_wall_geometry: LocalCutWallGeometry3D,
    cut_wall_bc: LocalCutWallBC3D,
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

    grad_cell = jnp.stack(
        (
            dfdx_cell[owner_i, owner_j, owner_k],
            dfdy_cell[owner_i, owner_j, owner_k],
            dfdz_cell[owner_i, owner_j, owner_k],
        ),
        axis=-1,
    )
    f_cell = field[owner_i, owner_j, owner_k]

    normal_contra = jnp.asarray(cut_wall_geometry.normal_contra, dtype=jnp.float64)
    normal_cov = jnp.einsum(
        "...ij,...j->...i",
        jnp.asarray(cut_wall_geometry.g_cov, dtype=jnp.float64),
        normal_contra,
    )
    g_cell = jnp.einsum("...i,...i->...", normal_contra, grad_cell)
    grad_tangent = grad_cell - g_cell[..., None] * normal_cov

    distance = jnp.asarray(cut_wall_geometry.distance, dtype=jnp.float64)
    safe_distance = jnp.maximum(jnp.abs(distance), 1.0e-30)
    g_dirichlet = (cut_wall_value - f_cell) / safe_distance
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


def build_local_projected_laplacian_flux_stencil(
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    face_bc: LocalBoundaryFaceBC3D | None = None,
    regular_face_geometry: LocalRegularFaceGeometry3D | None = None,
    cell_volume: LocalCellVolumeGeometry3D | None = None,
    cut_wall_geometry: LocalCutWallGeometry3D | None = None,
    cut_wall_bc: LocalCutWallBC3D | None = None,
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
    return div_flux / jnp.maximum(effective_volume, float(jacobian_floor))


def build_local_perp_laplacian_stencil(
    local: ConservativeStencil3D,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    face_bc: LocalBoundaryFaceBC3D | None = None,
    regular_face_geometry: LocalRegularFaceGeometry3D | None = None,
    cell_volume: LocalCellVolumeGeometry3D | None = None,
    cut_wall_geometry: LocalCutWallGeometry3D | None = None,
    cut_wall_bc: LocalCutWallBC3D | None = None,
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
        cell_volume=cell_volume,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
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
    cell_volume: LocalCellVolumeGeometry3D | None = None,
    cut_wall_geometry: LocalCutWallGeometry3D | None = None,
    cut_wall_bc: LocalCutWallBC3D | None = None,
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the domain-decomposed conservative perpendicular Laplacian."""

    cv_flux = build_local_perp_laplacian_stencil(
        local,
        geometry,
        domain,
        face_projectors=face_projectors,
        face_bc=face_bc,
        regular_face_geometry=regular_face_geometry,
        cell_volume=cell_volume,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        axis_regular_axes=axis_regular_axes,
        b_floor=b_floor,
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
    cell_volume: LocalCellVolumeGeometry3D | None = None,
    cut_wall_geometry: LocalCutWallGeometry3D | None = None,
    cut_wall_bc: LocalCutWallBC3D | None = None,
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
    cv_flux = build_local_projected_laplacian_flux_stencil(
        local,
        geometry,
        domain,
        face_projectors=face_projectors,
        face_bc=face_bc,
        regular_face_geometry=regular_face_geometry,
        cell_volume=cell_volume,
        cut_wall_geometry=cut_wall_geometry,
        cut_wall_bc=cut_wall_bc,
        axis_regular_axes=axis_regular_axes,
        b_floor=b_floor,
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


@_pytree_base
@dataclass(frozen=True)
class LocalPerpLaplacianInverseSolver:
    """SPMD GMRES adapter for local conservative perpendicular-Laplacian inversion."""

    geometry: LocalFciGeometry3D
    domain: LocalDomain3D
    stencil_builder: LocalConservativeStencilBuilder = (
        build_local_conservative_stencil_from_field
    )
    halo_exchange: HaloExchange3D | None = None
    topology_filler: TopologyHaloFiller3D | None = None
    physical_ghost_filler: PhysicalGhostCellFiller3D | None = None
    face_projectors: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None
    regular_face_geometry: LocalRegularFaceGeometry3D | None = None
    cut_wall_geometry: LocalCutWallGeometry3D | None = None
    cut_wall_bc: LocalCutWallBC3D | None = None
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
        if self.regular_face_geometry is not None and not isinstance(
            self.regular_face_geometry,
            LocalRegularFaceGeometry3D,
        ):
            raise TypeError(
                "regular_face_geometry must be a LocalRegularFaceGeometry3D or None"
            )
        if self.cut_wall_geometry is not None and not isinstance(
            self.cut_wall_geometry,
            LocalCutWallGeometry3D,
        ):
            raise TypeError("cut_wall_geometry must be a LocalCutWallGeometry3D or None")
        if self.cut_wall_bc is not None and not isinstance(self.cut_wall_bc, LocalCutWallBC3D):
            raise TypeError("cut_wall_bc must be a LocalCutWallBC3D or None")
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

    def _default_cut_wall_geometry(self) -> LocalCutWallGeometry3D:
        if self.cut_wall_geometry is not None:
            return self.cut_wall_geometry
        if self.cut_wall_bc is not None:
            return LocalCutWallGeometry3D.empty(self.cut_wall_bc.max_wall_faces)
        return LocalCutWallGeometry3D.empty(0)

    def _default_cut_wall_bc(self) -> LocalCutWallBC3D:
        if self.cut_wall_bc is not None:
            return self.cut_wall_bc
        return LocalCutWallBC3D.empty(self._default_cut_wall_geometry().max_wall_faces)

    def _apply_A(
        self,
        field_owned: jnp.ndarray,
        *,
        face_bc: LocalBoundaryFaceBC3D,
        cut_wall_bc: LocalCutWallBC3D,
        project_mean_zero: bool,
    ) -> jnp.ndarray:
        values = jnp.asarray(field_owned, dtype=jnp.float64)
        if project_mean_zero:
            values = _spmd_remove_weighted_mean(values, self.geometry, self.domain)

        field_halo = inject_owned_field_to_halo(values, self.domain.layout)
        if self.halo_exchange is not None:
            field_halo = self.halo_exchange(field_halo, self.domain)
        if self.topology_filler is not None:
            field_halo = self.topology_filler(field_halo, self.domain)
        if self.physical_ghost_filler is not None:
            field_halo = self.physical_ghost_filler(
                field_halo,
                self.domain,
                face_bc,
            )

        cut_wall_geometry = self._default_cut_wall_geometry()
        context = StencilBuilderContext(
            layout=self.domain.layout,
            domain=self.domain,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=cut_wall_bc,
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
        result = -local_perp_laplacian_conservative_op(
            local,
            self.geometry,
            self.domain,
            face_projectors=face_projectors,
            face_bc=face_bc,
            regular_face_geometry=self.regular_face_geometry,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=cut_wall_bc,
            axis_regular_axes=self.axis_regular_axes,
            b_floor=self.b_floor,
            jacobian_floor=self.jacobian_floor,
        )
        if self.config.regularization_epsilon != 0.0:
            result = result + self.config.regularization_epsilon * values
        if project_mean_zero:
            result = _spmd_remove_weighted_mean(result, self.geometry, self.domain)
        return result

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
        cut_wall_bc = self._default_cut_wall_bc()
        project_mean_zero = bool(self.config.project_mean_zero)

        if lift is None:
            homogeneous_face_bc = _homogeneous_local_face_bc(face_bc)
            homogeneous_cut_wall_bc = _homogeneous_local_cut_wall_bc(cut_wall_bc)
            boundary_source = self._apply_A(
                jnp.zeros_like(rhs),
                face_bc=face_bc,
                cut_wall_bc=cut_wall_bc,
                project_mean_zero=project_mean_zero,
            )
            linear_rhs = rhs - boundary_source
            initial_guess = guess
        else:
            homogeneous_face_bc = _dirichlet_lift_correction_local_face_bc(face_bc)
            homogeneous_cut_wall_bc = _dirichlet_lift_correction_local_cut_wall_bc(
                cut_wall_bc
            )
            lift_source = self._apply_A(
                lift,
                face_bc=face_bc,
                cut_wall_bc=cut_wall_bc,
                project_mean_zero=project_mean_zero,
            )
            linear_rhs = rhs - lift_source
            initial_guess = guess - lift

        if project_mean_zero:
            linear_rhs = _spmd_remove_weighted_mean(
                linear_rhs,
                self.geometry,
                self.domain,
            )
            initial_guess = _spmd_remove_weighted_mean(
                initial_guess,
                self.geometry,
                self.domain,
            )

        def apply_A(field_owned: jnp.ndarray) -> jnp.ndarray:
            return self._apply_A(
                field_owned,
                face_bc=homogeneous_face_bc,
                cut_wall_bc=homogeneous_cut_wall_bc,
                project_mean_zero=project_mean_zero,
            )

        solution, info = spmd_gmres_solve(
            apply_A,
            linear_rhs,
            initial_guess,
            self.geometry,
            self.domain,
            self.config,
        )
        if lift is not None:
            solution = lift + solution
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
            self.regular_face_geometry,
            self.cut_wall_geometry,
            self.cut_wall_bc,
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
            regular_face_geometry,
            cut_wall_geometry,
            cut_wall_bc,
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
            regular_face_geometry=regular_face_geometry,
            cut_wall_geometry=cut_wall_geometry,
            cut_wall_bc=cut_wall_bc,
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
