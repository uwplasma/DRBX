from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from solvax import gmres as solvax_gmres
from solvax.implicit import linear_solve as solvax_linear_solve
from ..fci._shared import (  # noqa: F401  (definitions shared by both FCI families)
    _take_stencil_finite_difference,
    _axis_slice_nd,
)


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
    build_conservative_stencil_from_field,
    RegularFaceGeometry3D,
    build_local_conservative_stencil_from_field,
    build_local_stencil_from_field,
)
from .fci import _first_derivative_3d
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
from ..geometry.fci_geometry import _axis_index_nd


# =============================================================================
# Parallel-gradient operators
# =============================================================================


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

    first_stencil = stencil_builder(
        field,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=face_bc,
    )
    first_grad = grad_parallel_op_direct(first_stencil, geometry)

    second_stencil = stencil_builder(
        first_grad,
        geometry,
        periodic_axes=periodic_axes,
        face_bc=face_bc,
    )
    return grad_parallel_op_direct(second_stencil, geometry)


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

    lower_plane = face_grad[_axis_index_nd(axis, 0, face_grad.ndim)]
    lower_normal = lower_plane[..., axis]
    lower_coord = (lower_center - lower_value) / jnp.maximum(lower_distance, 1.0e-30)
    if lower_patch_allowed:
        lower_plane = lower_plane.at[..., axis].set(
            jnp.where(lower_mask & (lower_kind == BC_DIRICHLET), lower_coord, lower_normal)
        )
        lower_plane = lower_plane.at[..., axis].set(
            jnp.where(lower_mask & (lower_kind == BC_NEUMANN), -lower_value, lower_plane[..., axis])
        )
        face_grad = face_grad.at[_axis_index_nd(axis, 0, face_grad.ndim)].set(lower_plane)

    upper_plane = face_grad[_axis_index_nd(axis, -1, face_grad.ndim)]
    upper_normal = upper_plane[..., axis]
    upper_coord = (upper_value - upper_center) / jnp.maximum(upper_distance, 1.0e-30)
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


def _homogeneous_bc(bc: FciBoundaryCondition) -> FciBoundaryCondition:
    raise NotImplementedError("legacy axis-level boundary conditions have been removed")


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
        gmres_debug: bool = False,
        check_residual: bool = True,
        stagnation_iters: int = 20,
    ) -> None:
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
        if self.pin_point is not None and len(self.pin_point) != 3:
            raise ValueError("pin_point must be a 3-tuple of integer indices")
        if self.pin_point is not None:
            for axis_index, axis_size in zip(self.pin_point, self.geometry.shape):
                if not 0 <= axis_index < axis_size:
                    raise ValueError(
                        "pin_point indices must lie inside the geometry shape; "
                        f"got {self.pin_point} for shape {self.geometry.shape}"
                    )
        self.gmres_debug = bool(gmres_debug)
        self.check_residual = bool(check_residual)
        self.stagnation_iters = int(stagnation_iters)
        # `throw` is kept in the solve signatures for API stability; the solvax
        # backend never raises inside the solve (non-convergence is caught by the
        # host-side check_residual gate on the diagnostic path).
        self._solve_jit = jax.jit(self._solve_impl, static_argnums=(4,))
        self._solve_lifted_jit = jax.jit(self._solve_lifted_impl, static_argnums=(7,))
        self._solve_fast_jit = jax.jit(self._solve_fast_impl, static_argnums=(4,))

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
        local = self.stencil_builder(
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


    def _gmres_flat(self, apply_A, rhs: jnp.ndarray, guess: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Restarted GMRES (solvax) on the flattened field.

        Returns ``(solution_field, num_steps)``. solvax stops on the true
        residual ``||b - A x|| <= max(atol, rtol * ||b||)`` with ``atol = rtol
        = self.tol``.
        The Krylov cycle length is capped at ``maxiter`` so the total iteration
        budget matches the previous backend.
        """

        shape = rhs.shape

        def matvec(flat: jnp.ndarray) -> jnp.ndarray:
            return jnp.ravel(apply_A(jnp.reshape(flat, shape)))

        precond = None

        # ``maxiter`` counts restart cycles (the semantics of the previous
        # backend, whose ``max_steps`` was outer cycles of length ``restart``),
        # so the total inner-iteration budget is ``maxiter * restart``. The
        # Krylov cycle is additionally capped at 64: solvax's fixed-shape
        # Arnoldi pays O(cycle) orthogonalization work per iteration whether or
        # not the slots are used, so very long requested cycles cost real time;
        # the cap preserves the total budget by adding restarts.
        budget = max(1, int(self.maxiter)) * max(1, int(self.restart))
        cycle = max(1, min(int(self.restart), 64))
        max_restarts = max(1, -(-budget // cycle))
        def krylov(apply, target, x0):
            solution = solvax_gmres(
                apply,
                target,
                x0=x0,
                precond=precond,
                restart=cycle,
                # Pure relative stopping on the true residual: the diagnostic
                # residual gate is relative, and an absolute atol=tol floor would
                # stop early whenever ||rhs|| < 1 (small-amplitude stages).
                rtol=self.tol,
                atol=0.0,
                max_restarts=max_restarts,
            )
            return solution.x, jnp.asarray(solution.iterations, dtype=jnp.int32)

        # Implicit-function VJP: reverse mode costs one transposed Krylov solve
        # (same preconditioner) instead of differentiating the while_loop.
        x, steps = solvax_linear_solve(
            matvec,
            jnp.ravel(rhs),
            lambda apply, b: krylov(apply, b, jax.lax.stop_gradient(jnp.ravel(guess))),
            transpose_solver=lambda apply_t, b: krylov(apply_t, b, jnp.zeros_like(b)),
            has_aux=True,
        )
        return jnp.reshape(x, shape), steps

    def _solve_fast_impl(
        self,
        omega: jnp.ndarray,
        phi_guess: jnp.ndarray,
        face_bc: BoundaryFaceBC3D,
        cut_wall_bc: CutWallBC3D,
        throw: bool,
    ) -> jnp.ndarray:
        """Solve for phi only: no diagnostic matvecs, no residual norms.

        This is the hot-loop path (one boundary-source application plus the
        GMRES solve); `_solve_impl` adds the compatibility and final-residual
        diagnostics and is kept for validation and debugging.
        """

        rhs = jnp.asarray(omega, dtype=jnp.float64)
        guess = jnp.asarray(phi_guess, dtype=jnp.float64)
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
        if self.pin_point is not None:
            linear_rhs = linear_rhs.at[self.pin_point].set(self.pin_value)

        def apply_A(phi: jnp.ndarray) -> jnp.ndarray:
            values = self._apply_A(phi, homogeneous_face_bc, homogeneous_cut_wall_bc, project_mean_zero)
            if self.pin_point is not None:
                values = values.at[self.pin_point].set(phi[self.pin_point])
            return values

        phi, num_steps = self._gmres_flat(apply_A, linear_rhs, guess)
        if project_mean_zero:
            phi = _remove_weighted_mean(phi, self.geometry)
        if self.target_mean_phi is not None:
            phi = _set_weighted_mean(phi, self.geometry, self.target_mean_phi)
        if self.pin_point is not None:
            phi = phi.at[self.pin_point].set(self.pin_value)
        return phi

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

        phi, num_steps = self._gmres_flat(apply_A, linear_rhs, guess)
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

        correction, num_steps = self._gmres_flat(apply_A, rhs_u, correction_guess)
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
        if not return_diagnostics and not self.check_residual and not self.gmres_debug:
            # Hot-loop path: phi only, no diagnostic matvecs, no host syncs.
            # Safe to call from inside jit-compiled code.
            return self._solve_fast_jit(rhs, phi_guess, face_bc, cut_wall_bc, throw)
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

