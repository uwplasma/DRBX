from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

import jax
import jax.numpy as jnp

from ..geometry.fci_geometry import (
    _DataclassPyTreeMixin,
    FciGeometry3D,
    LocalDomain3D,
    LocalFciGeometry3D,
    HaloLayout3D,
    RegularFaceGeometry3D,
    LocalRegularFaceGeometry3D,
    CellVolumeGeometry3D,
    LocalCellVolumeGeometry3D,
    LocalControlVolumeCellGeometry3D,
)
from .fci_model import (
    FciFieldBundle,
    FciModelState,
    assert_matching_field_names,
)
from .fci_helpers import (
    _as_bool_face_array,
    _as_coordinate_derivative_weight_array,
    _as_coordinate_face_tuple,
    _as_face_flux_array,
    _as_float64_array,
    _as_int_face_array,
    _as_int_stencil_array,
    _as_wall_face_array,
    _as_weight_stencil_array,
    _axis_regular_lower_x_face,
    _normalize_axis_flags,
    _local_cell_halo_array,
    _as_local_wall_array,
    _as_local_wall_int_array,
    _as_local_wall_bool_array,
    _as_local_wall_stencil_index_array,
    _as_local_wall_stencil_weight_array,
)


_pytree_base = jax.tree_util.register_pytree_node_class


BC_NONE = 0
BC_DIRICHLET = 1
BC_NEUMANN = 2
BC_NORMALFLUX = 3
BC_NOFLUX = 4

CV_FACE_NONE = 0
CV_FACE_INTERIOR = 1
CV_FACE_CUT_WALL = 2
CV_FACE_PARTIAL = 3
CV_FACE_PHYSICAL_BOUNDARY = 4

CV_RECONSTRUCTION_EQUATION_NONE = 0
CV_RECONSTRUCTION_EQUATION_CELL = 1
CV_RECONSTRUCTION_EQUATION_DIRICHLET = 2
CV_RECONSTRUCTION_EQUATION_REMOTE_CELL = 3


BoundaryPayloadT = TypeVar("BoundaryPayloadT")


@_pytree_base
@dataclass(frozen=True)
class LocalBoundaryData3D(_DataclassPyTreeMixin):
    """Model-shaped local boundary payload bundle.

    ``face_bc`` and ``cut_wall_bc`` are field bundles: each bundle field names
    the model field whose boundary payload it contains.  This allows a
    boundary builder to construct coupled BCs from the complete pre-BC state
    while preserving an unambiguous field-to-BC association.
    """

    face_bc: FciFieldBundle | None = None
    cut_wall_bc: FciFieldBundle | None = None

    def __post_init__(self) -> None:
        if self.face_bc is not None and not isinstance(self.face_bc, FciFieldBundle):
            raise TypeError("LocalBoundaryData3D.face_bc must be an FciFieldBundle or None")
        if self.cut_wall_bc is not None and not isinstance(self.cut_wall_bc, FciFieldBundle):
            raise TypeError(
                "LocalBoundaryData3D.cut_wall_bc must be an FciFieldBundle or None"
            )

    def tree_flatten(self):
        return ((self.face_bc, self.cut_wall_bc), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        face_bc, cut_wall_bc = children
        return cls(face_bc=face_bc, cut_wall_bc=cut_wall_bc)


@_pytree_base
@dataclass(frozen=True)
class LocalBoundaryPreparation3D(_DataclassPyTreeMixin):
    """Local boundary data plus optional remote dependency requests."""

    local_data: LocalBoundaryData3D | None = None
    remote_dependencies: FciFieldBundle | None = None

    def __post_init__(self) -> None:
        if self.local_data is not None and not isinstance(
            self.local_data,
            LocalBoundaryData3D,
        ):
            raise TypeError(
                "LocalBoundaryPreparation3D.local_data must be a "
                "LocalBoundaryData3D or None"
            )
        if self.remote_dependencies is not None and not isinstance(
            self.remote_dependencies,
            FciFieldBundle,
        ):
            raise TypeError(
                "LocalBoundaryPreparation3D.remote_dependencies must be an "
                "FciFieldBundle or None"
            )


@_pytree_base
@dataclass(frozen=True)
class LocalBoundaryRemoteDependencyTable(_DataclassPyTreeMixin):
    """Remote scalar requests used while finalizing local boundary payloads."""

    request_active: jnp.ndarray
    request_dependency_kind: jnp.ndarray
    request_source_global_i: jnp.ndarray
    request_source_global_j: jnp.ndarray
    request_source_global_k: jnp.ndarray
    request_source_shard_index: jnp.ndarray
    request_source_shard_linear: jnp.ndarray
    request_source_owner_local_i: jnp.ndarray
    request_source_owner_local_j: jnp.ndarray
    request_source_owner_local_k: jnp.ndarray
    request_value_slot: jnp.ndarray

    def __post_init__(self) -> None:
        request_active = jnp.asarray(self.request_active, dtype=bool)
        request_shape = tuple(int(v) for v in request_active.shape)
        if request_active.ndim != 1:
            raise ValueError(
                "LocalBoundaryRemoteDependencyTable.request_active must be 1D, "
                f"got {request_active.shape}"
            )
        object.__setattr__(self, "request_active", request_active)
        for name in (
            "request_dependency_kind",
            "request_source_global_i",
            "request_source_global_j",
            "request_source_global_k",
            "request_source_shard_linear",
            "request_source_owner_local_i",
            "request_source_owner_local_j",
            "request_source_owner_local_k",
            "request_value_slot",
        ):
            value = jnp.asarray(getattr(self, name), dtype=jnp.int32)
            if value.shape != request_shape:
                raise ValueError(
                    f"LocalBoundaryRemoteDependencyTable.{name} must have "
                    f"shape {request_shape}, got {value.shape}"
                )
            object.__setattr__(self, name, value)

        request_source_shard_index = jnp.asarray(
            self.request_source_shard_index,
            dtype=jnp.int32,
        )
        if (
            request_source_shard_index.ndim != 2
            or request_source_shard_index.shape[1] != 3
        ):
            raise ValueError(
                "LocalBoundaryRemoteDependencyTable.request_source_shard_index "
                "must have shape (max_receive_values, 3), got "
                f"{request_source_shard_index.shape}"
            )
        if int(request_source_shard_index.shape[0]) != request_shape[0]:
            raise ValueError(
                "LocalBoundaryRemoteDependencyTable.request_source_shard_index "
                "must match request_active length; got "
                f"{request_source_shard_index.shape[0]}, expected {request_shape[0]}"
            )
        object.__setattr__(
            self,
            "request_source_shard_index",
            request_source_shard_index,
        )

    @property
    def max_receive_values(self) -> int:
        return int(self.request_active.size)

    @property
    def has_requests(self) -> bool:
        return self.max_receive_values > 0

    @classmethod
    def empty(cls) -> "LocalBoundaryRemoteDependencyTable":
        return cls(
            request_active=jnp.zeros((0,), dtype=bool),
            request_dependency_kind=jnp.zeros((0,), dtype=jnp.int32),
            request_source_global_i=jnp.zeros((0,), dtype=jnp.int32),
            request_source_global_j=jnp.zeros((0,), dtype=jnp.int32),
            request_source_global_k=jnp.zeros((0,), dtype=jnp.int32),
            request_source_shard_index=jnp.zeros((0, 3), dtype=jnp.int32),
            request_source_shard_linear=jnp.zeros((0,), dtype=jnp.int32),
            request_source_owner_local_i=jnp.zeros((0,), dtype=jnp.int32),
            request_source_owner_local_j=jnp.zeros((0,), dtype=jnp.int32),
            request_source_owner_local_k=jnp.zeros((0,), dtype=jnp.int32),
            request_value_slot=jnp.zeros((0,), dtype=jnp.int32),
        )


@_pytree_base
@dataclass(frozen=True)
class LocalBoundaryConditionBuilder(_DataclassPyTreeMixin):
    """Prepare and finalize local boundary payloads around remote exchange."""

    prepare_fn: Callable[
        [
            FciModelState,
            LocalFciGeometry3D,
            LocalDomain3D,
            LocalCutWallGeometry3D | None,
        ],
        LocalBoundaryPreparation3D,
    ]
    finalize_fn: Callable[
        [
            LocalBoundaryPreparation3D,
            FciFieldBundle | None,
            FciModelState,
            LocalFciGeometry3D,
            LocalDomain3D,
            LocalCutWallGeometry3D | None,
        ],
        LocalBoundaryData3D,
    ]

    def prepare(
        self,
        state_halo_pre_bc: FciModelState,
        geometry: LocalFciGeometry3D,
        domain: LocalDomain3D,
        cut_wall_geometry: LocalCutWallGeometry3D | None,
    ) -> LocalBoundaryPreparation3D:
        if not isinstance(state_halo_pre_bc, FciModelState):
            raise TypeError(
                "LocalBoundaryConditionBuilder.prepare requires an "
                "FciModelState with physical ghost cells not yet filled"
            )
        state_halo_pre_bc.assert_field_shape(domain.layout.cell_halo_shape)
        result = self.prepare_fn(
            state_halo_pre_bc,
            geometry,
            domain,
            cut_wall_geometry,
        )
        if not isinstance(result, LocalBoundaryPreparation3D):
            raise TypeError(
                "LocalBoundaryConditionBuilder.prepare_fn must return "
                "LocalBoundaryPreparation3D"
            )
        if result.local_data is not None:
            self._validate_boundary_data_field_names(
                state_halo_pre_bc,
                result.local_data,
            )
        if result.remote_dependencies is not None:
            assert_matching_field_names(state_halo_pre_bc, result.remote_dependencies)
        return result

    def finalize(
        self,
        preparation: LocalBoundaryPreparation3D,
        remote_values: FciFieldBundle | None,
        state_halo_pre_bc: FciModelState,
        geometry: LocalFciGeometry3D,
        domain: LocalDomain3D,
        cut_wall_geometry: LocalCutWallGeometry3D | None,
    ) -> LocalBoundaryData3D:
        if not isinstance(preparation, LocalBoundaryPreparation3D):
            raise TypeError("preparation must be a LocalBoundaryPreparation3D")
        if not isinstance(state_halo_pre_bc, FciModelState):
            raise TypeError("state_halo_pre_bc must be an FciModelState")
        state_halo_pre_bc.assert_field_shape(domain.layout.cell_halo_shape)
        if remote_values is not None:
            assert_matching_field_names(state_halo_pre_bc, remote_values)
        result = self.finalize_fn(
            preparation,
            remote_values,
            state_halo_pre_bc,
            geometry,
            domain,
            cut_wall_geometry,
        )
        if not isinstance(result, LocalBoundaryData3D):
            raise TypeError(
                "LocalBoundaryConditionBuilder.finalize_fn must return "
                "LocalBoundaryData3D"
            )
        self._validate_boundary_data_field_names(state_halo_pre_bc, result)
        return result

    @staticmethod
    def _validate_boundary_data_field_names(
        state_halo_pre_bc: FciModelState,
        data: LocalBoundaryData3D,
    ) -> None:
        if data.face_bc is not None:
            assert_matching_field_names(state_halo_pre_bc, data.face_bc)
        if data.cut_wall_bc is not None:
            assert_matching_field_names(state_halo_pre_bc, data.cut_wall_bc)

    def tree_flatten(self):
        return (), (self.prepare_fn, self.finalize_fn)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        prepare_fn, finalize_fn = aux_data
        return cls(prepare_fn=prepare_fn, finalize_fn=finalize_fn)



@_pytree_base
@dataclass(frozen=True)
class CoordinateFaceValueReconstructor3D:
    """Dense coordinate-face value reconstructor for cell-centered fields."""

    def extrapolate(
        self,
        field: jnp.ndarray,
        geometry: FciGeometry3D,
        periodic_axes: tuple[bool, bool, bool] = (False, True, True),
        axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        values = _as_float64_array(field, "CoordinateFaceValueReconstructor3D.field")
        if values.shape != geometry.shape:
            raise ValueError(f"field must have shape {geometry.shape}, got {values.shape}")
        periodic_axes = _normalize_axis_flags(periodic_axes, "periodic_axes")
        axis_regular_axes = _normalize_axis_flags(axis_regular_axes, "axis_regular_axes")
        if any(periodic and axis_regular for periodic, axis_regular in zip(periodic_axes, axis_regular_axes)):
            raise ValueError(
                "periodic_axes and axis_regular_axes cannot both be True on the same axis; "
                f"got periodic_axes={periodic_axes}, axis_regular_axes={axis_regular_axes}"
            )
        if axis_regular_axes[1] or axis_regular_axes[2]:
            raise ValueError(
                "axis_regular_axes currently only supports the lower x axis; "
                f"got axis_regular_axes={axis_regular_axes}"
            )

        x_faces = jnp.empty((values.shape[0] + 1, values.shape[1], values.shape[2]), dtype=jnp.float64)
        y_faces = jnp.empty((values.shape[0], values.shape[1] + 1, values.shape[2]), dtype=jnp.float64)
        z_faces = jnp.empty((values.shape[0], values.shape[1], values.shape[2] + 1), dtype=jnp.float64)

        x_faces = x_faces.at[1:-1].set(0.5 * (values[:-1] + values[1:]))
        if periodic_axes[0]:
            x_periodic = 0.5 * (values[0] + values[-1])
            x_faces = x_faces.at[0].set(x_periodic)
            x_faces = x_faces.at[-1].set(x_periodic)
        elif axis_regular_axes[0]:
            x_faces = x_faces.at[0].set(_axis_regular_lower_x_face(values))
            x_faces = x_faces.at[-1].set(values[-1])
        else:
            x_faces = x_faces.at[0].set(values[0])
            x_faces = x_faces.at[-1].set(values[-1])

        y_faces = y_faces.at[:, 1:-1, :].set(0.5 * (values[:, :-1, :] + values[:, 1:, :]))
        if periodic_axes[1]:
            y_periodic = 0.5 * (values[:, 0, :] + values[:, -1, :])
            y_faces = y_faces.at[:, 0, :].set(y_periodic)
            y_faces = y_faces.at[:, -1, :].set(y_periodic)
        else:
            y_faces = y_faces.at[:, 0, :].set(values[:, 0, :])
            y_faces = y_faces.at[:, -1, :].set(values[:, -1, :])

        z_faces = z_faces.at[:, :, 1:-1].set(0.5 * (values[:, :, :-1] + values[:, :, 1:]))
        if periodic_axes[2]:
            z_periodic = 0.5 * (values[:, :, 0] + values[:, :, -1])
            z_faces = z_faces.at[:, :, 0].set(z_periodic)
            z_faces = z_faces.at[:, :, -1].set(z_periodic)
        else:
            z_faces = z_faces.at[:, :, 0].set(values[:, :, 0])
            z_faces = z_faces.at[:, :, -1].set(values[:, :, -1])
        return x_faces, y_faces, z_faces

    def extrapolate_neumann(
        self,
        field: jnp.ndarray,
        geometry: FciGeometry3D,
        normal_derivative: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        periodic_axes: tuple[bool, bool, bool] = (False, True, True),
        axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        values = _as_float64_array(field, "CoordinateFaceValueReconstructor3D.field")
        if values.shape != geometry.shape:
            raise ValueError(f"field must have shape {geometry.shape}, got {values.shape}")
        periodic_axes = _normalize_axis_flags(periodic_axes, "periodic_axes")
        axis_regular_axes = _normalize_axis_flags(axis_regular_axes, "axis_regular_axes")
        if any(periodic and axis_regular for periodic, axis_regular in zip(periodic_axes, axis_regular_axes)):
            raise ValueError(
                "periodic_axes and axis_regular_axes cannot both be True on the same axis; "
                f"got periodic_axes={periodic_axes}, axis_regular_axes={axis_regular_axes}"
            )
        if axis_regular_axes[1] or axis_regular_axes[2]:
            raise ValueError(
                "axis_regular_axes currently only supports the lower x axis; "
                f"got axis_regular_axes={axis_regular_axes}"
            )
        # normal_derivative is interpreted as the outward-pointing normal derivative
        # on the nonperiodic boundary faces. The lower x face is topological when
        # axis_regular_axes[0] is True, so it is left untouched.
        gx, gy, gz = _as_coordinate_face_tuple(
            normal_derivative,
            geometry,
            "CoordinateFaceValueReconstructor3D.normal_derivative",
        )
        x_faces, y_faces, z_faces = self.extrapolate(
            values,
            geometry,
            periodic_axes=periodic_axes,
            axis_regular_axes=axis_regular_axes,
        )
        dx = jnp.asarray(geometry.spacing.dx, dtype=jnp.float64)
        dy = jnp.asarray(geometry.spacing.dy, dtype=jnp.float64)
        dz = jnp.asarray(geometry.spacing.dz, dtype=jnp.float64)

        if not periodic_axes[0] and not axis_regular_axes[0]:
            x_faces = x_faces.at[0].set(values[0] + 0.5 * dx[0] * gx[0])
            x_faces = x_faces.at[-1].set(values[-1] + 0.5 * dx[-1] * gx[-1])
        elif not periodic_axes[0]:
            x_faces = x_faces.at[-1].set(values[-1] + 0.5 * dx[-1] * gx[-1])
        if not periodic_axes[1]:
            y_faces = y_faces.at[:, 0, :].set(values[:, 0, :] + 0.5 * dy[:, 0, :] * gy[:, 0, :])
            y_faces = y_faces.at[:, -1, :].set(values[:, -1, :] + 0.5 * dy[:, -1, :] * gy[:, -1, :])
        if not periodic_axes[2]:
            z_faces = z_faces.at[:, :, 0].set(values[:, :, 0] + 0.5 * dz[:, :, 0] * gz[:, :, 0])
            z_faces = z_faces.at[:, :, -1].set(values[:, :, -1] + 0.5 * dz[:, :, -1] * gz[:, :, -1])
        return x_faces, y_faces, z_faces

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls()


def _local_coordinate_face_values_from_halo(
    field_halo: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    layout: HaloLayout3D,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    values = _as_float64_array(field_halo, "LocalCoordinateFaceValueReconstructor3D.field_halo")
    if values.shape != layout.cell_halo_shape:
        raise ValueError(
            "field_halo must have shape layout.cell_halo_shape; "
            f"got {values.shape}, expected {layout.cell_halo_shape}"
        )
    if geometry.layout != layout:
        raise ValueError("geometry and layout must share the same HaloLayout3D")
    if layout.halo_width < 1:
        raise ValueError("local face reconstruction requires halo_width >= 1")

    h = layout.halo_width
    nx, ny, nz = layout.owned_shape

    x_faces = 0.5 * (
        values[h - 1 : h + nx, h : h + ny, h : h + nz]
        + values[h : h + nx + 1, h : h + ny, h : h + nz]
    )
    y_faces = 0.5 * (
        values[h : h + nx, h - 1 : h + ny, h : h + nz]
        + values[h : h + nx, h : h + ny + 1, h : h + nz]
    )
    z_faces = 0.5 * (
        values[h : h + nx, h : h + ny, h - 1 : h + nz]
        + values[h : h + nx, h : h + ny, h : h + nz + 1]
    )
    return x_faces, y_faces, z_faces


@_pytree_base
@dataclass(frozen=True)
class LocalCoordinateFaceValueReconstructor3D(_DataclassPyTreeMixin):
    """Reconstruct coordinate control-face values from a complete local halo field."""

    reconstruct_fn: Callable[
        [
            jnp.ndarray,
            "LocalFciGeometry3D",
            "HaloLayout3D",
        ],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ] = _local_coordinate_face_values_from_halo

    def extrapolate(
        self,
        field_halo: jnp.ndarray,
        geometry: "LocalFciGeometry3D",
        layout: "HaloLayout3D",
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.reconstruct_fn(field_halo, geometry, layout)

    def tree_flatten(self):
        return (), self.reconstruct_fn

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data)


@_pytree_base
@dataclass(frozen=True)
class LocalCoordinateSideValues1D(_DataclassPyTreeMixin):
    """Lower/upper side-plane payloads for one coordinate axis."""

    lower: jnp.ndarray
    upper: jnp.ndarray
    mask_lower: jnp.ndarray
    mask_upper: jnp.ndarray

    def __post_init__(self) -> None:
        lower = jnp.asarray(self.lower, dtype=jnp.float64)
        upper = jnp.asarray(self.upper, dtype=jnp.float64)
        mask_lower = jnp.asarray(self.mask_lower, dtype=bool)
        mask_upper = jnp.asarray(self.mask_upper, dtype=bool)
        if lower.shape != upper.shape:
            raise ValueError(
                "LocalCoordinateSideValues1D.lower and upper must have the same shape; "
                f"got lower={lower.shape}, upper={upper.shape}"
            )
        if mask_lower.shape != lower.shape or mask_upper.shape != lower.shape:
            raise ValueError(
                "LocalCoordinateSideValues1D masks must match the side-plane shape; "
                f"got lower={lower.shape}, mask_lower={mask_lower.shape}, mask_upper={mask_upper.shape}"
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "mask_lower", mask_lower)
        object.__setattr__(self, "mask_upper", mask_upper)

    def replace(self, **updates: object) -> "LocalCoordinateSideValues1D":
        allowed = {"lower", "upper", "mask_lower", "mask_upper"}
        unknown = set(updates) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown LocalCoordinateSideValues1D field(s): {names}")
        return LocalCoordinateSideValues1D(
            lower=updates.get("lower", self.lower),
            upper=updates.get("upper", self.upper),
            mask_lower=updates.get("mask_lower", self.mask_lower),
            mask_upper=updates.get("mask_upper", self.mask_upper),
        )

    def tree_flatten(self):
        return ((self.lower, self.upper, self.mask_lower, self.mask_upper), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@_pytree_base
@dataclass(frozen=True)
class LocalCoordinateSideValues3D(_DataclassPyTreeMixin):
    """Lower/upper coordinate-side payloads for the three axes."""

    x: LocalCoordinateSideValues1D
    y: LocalCoordinateSideValues1D
    z: LocalCoordinateSideValues1D

    def __post_init__(self) -> None:
        if not isinstance(self.x, LocalCoordinateSideValues1D):
            raise TypeError("LocalCoordinateSideValues3D.x must be a LocalCoordinateSideValues1D")
        if not isinstance(self.y, LocalCoordinateSideValues1D):
            raise TypeError("LocalCoordinateSideValues3D.y must be a LocalCoordinateSideValues1D")
        if not isinstance(self.z, LocalCoordinateSideValues1D):
            raise TypeError("LocalCoordinateSideValues3D.z must be a LocalCoordinateSideValues1D")

    def replace(self, **updates: object) -> "LocalCoordinateSideValues3D":
        allowed = {"x", "y", "z"}
        unknown = set(updates) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown LocalCoordinateSideValues3D field(s): {names}")
        return LocalCoordinateSideValues3D(
            x=updates.get("x", self.x),
            y=updates.get("y", self.y),
            z=updates.get("z", self.z),
        )

    def tree_flatten(self):
        return ((self.x, self.y, self.z), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


def _local_side_samples(
    field_halo: jnp.ndarray,
    layout: HaloLayout3D,
    axis: int,
    side: int,
    sample_count: int,
) -> list[jnp.ndarray]:
    values = _local_cell_halo_array(field_halo, layout, "field_halo")
    axis = int(axis)
    side = int(side)
    if sample_count < 1:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    h = layout.halo_width
    nx, ny, nz = layout.owned_shape
    if axis == 0:
        base = h if side == 0 else h + nx - 1
        step = 1 if side == 0 else -1
        return [values[base + step * i, h : h + ny, h : h + nz] for i in range(sample_count)]
    if axis == 1:
        base = h if side == 0 else h + ny - 1
        step = 1 if side == 0 else -1
        return [values[h : h + nx, base + step * i, h : h + nz] for i in range(sample_count)]
    base = h if side == 0 else h + nz - 1
    step = 1 if side == 0 else -1
    return [values[h : h + nx, h : h + ny, base + step * i] for i in range(sample_count)]


def _local_coordinate_side_values_from_array(
    values: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    layout: HaloLayout3D,
    *,
    name: str,
) -> LocalCoordinateSideValues3D:
    values = _as_float64_array(values, name)
    if values.shape != layout.cell_halo_shape:
        raise ValueError(f"{name} must have shape {layout.cell_halo_shape}, got {values.shape}")
    if geometry.layout != layout:
        raise ValueError("geometry and layout must share the same HaloLayout3D")
    nx, ny, nz = layout.owned_shape
    h = layout.halo_width
    return LocalCoordinateSideValues3D(
        x=LocalCoordinateSideValues1D(
            lower=values[h - 1 : h, h : h + ny, h : h + nz][0],
            upper=values[h + nx : h + nx + 1, h : h + ny, h : h + nz][0],
            mask_lower=jnp.ones((ny, nz), dtype=bool),
            mask_upper=jnp.ones((ny, nz), dtype=bool),
        ),
        y=LocalCoordinateSideValues1D(
            lower=values[h : h + nx, h - 1 : h, h : h + nz][:, 0, :],
            upper=values[h : h + nx, h + ny : h + ny + 1, h : h + nz][:, 0, :],
            mask_lower=jnp.ones((nx, nz), dtype=bool),
            mask_upper=jnp.ones((nx, nz), dtype=bool),
        ),
        z=LocalCoordinateSideValues1D(
            lower=values[h : h + nx, h : h + ny, h - 1 : h][:, :, 0],
            upper=values[h : h + nx, h : h + ny, h + nz : h + nz + 1][:, :, 0],
            mask_lower=jnp.ones((nx, ny), dtype=bool),
            mask_upper=jnp.ones((nx, ny), dtype=bool),
        ),
    )


@_pytree_base
@dataclass(frozen=True)
class LocalCoordinateNormalDerivativeConstructor3D(_DataclassPyTreeMixin):
    """Construct coordinate-side normal derivatives using local halo data."""

    dnormal_weights: jnp.ndarray
    d2normal_weights: jnp.ndarray

    def __post_init__(self) -> None:
        dnormal_weights = jnp.asarray(self.dnormal_weights, dtype=jnp.float64)
        d2normal_weights = jnp.asarray(self.d2normal_weights, dtype=jnp.float64)
        if dnormal_weights.ndim != 3 or d2normal_weights.ndim != 3:
            raise ValueError(
                "LocalCoordinateNormalDerivativeConstructor3D weights must have shape (3, 2, stencil_width)"
            )
        if dnormal_weights.shape != d2normal_weights.shape:
            raise ValueError(
                "LocalCoordinateNormalDerivativeConstructor3D weight tensors must have the same shape; "
                f"got {dnormal_weights.shape} and {d2normal_weights.shape}"
            )
        if dnormal_weights.shape[0] != 3 or dnormal_weights.shape[1] != 2:
            raise ValueError(
                "LocalCoordinateNormalDerivativeConstructor3D weights must have leading shape (3, 2, stencil_width); "
                f"got {dnormal_weights.shape}"
            )
        if dnormal_weights.shape[2] < 2:
            raise ValueError(
                "LocalCoordinateNormalDerivativeConstructor3D requires at least two stencil points"
            )
        object.__setattr__(self, "dnormal_weights", dnormal_weights)
        object.__setattr__(self, "d2normal_weights", d2normal_weights)

    @property
    def stencil_width(self) -> int:
        return int(self.dnormal_weights.shape[2])

    @classmethod
    def from_geometry(cls, geometry: FciGeometry3D) -> "LocalCoordinateNormalDerivativeConstructor3D":
        base = CoordinateNormalDerivativeConstructor3D.from_geometry(geometry)
        return cls(dnormal_weights=base.dnormal_weights, d2normal_weights=base.d2normal_weights)

    def _wall_side_derivatives(
        self,
        field_halo: jnp.ndarray,
        wall_value: LocalCoordinateSideValues1D,
        geometry: LocalFciGeometry3D,
        layout: HaloLayout3D,
        *,
        axis: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        sample_count = self.stencil_width - 1
        lower_samples = _local_side_samples(field_halo, layout, axis, 0, sample_count)
        upper_samples = _local_side_samples(field_halo, layout, axis, 1, sample_count)
        dnormal_lower = self.dnormal_weights[axis, 0, 0] * wall_value.lower
        d2normal_lower = self.d2normal_weights[axis, 0, 0] * wall_value.lower
        dnormal_upper = self.dnormal_weights[axis, 1, 0] * wall_value.upper
        d2normal_upper = self.d2normal_weights[axis, 1, 0] * wall_value.upper
        for idx, sample in enumerate(lower_samples, start=1):
            dnormal_lower = dnormal_lower + self.dnormal_weights[axis, 0, idx] * sample
            d2normal_lower = d2normal_lower + self.d2normal_weights[axis, 0, idx] * sample
        for idx, sample in enumerate(upper_samples, start=1):
            dnormal_upper = dnormal_upper + self.dnormal_weights[axis, 1, idx] * sample
            d2normal_upper = d2normal_upper + self.d2normal_weights[axis, 1, idx] * sample
        dnormal_lower = jnp.where(wall_value.mask_lower, dnormal_lower, 0.0)
        d2normal_lower = jnp.where(wall_value.mask_lower, d2normal_lower, 0.0)
        dnormal_upper = jnp.where(wall_value.mask_upper, dnormal_upper, 0.0)
        d2normal_upper = jnp.where(wall_value.mask_upper, d2normal_upper, 0.0)
        return dnormal_lower, dnormal_upper, d2normal_lower, d2normal_upper

    def dnormal_from_wall_value(
        self,
        field_halo: jnp.ndarray,
        wall_value: LocalCoordinateSideValues3D,
        geometry: "LocalFciGeometry3D",
        layout: "HaloLayout3D",
    ) -> "LocalCoordinateSideValues3D":
        dnormal, _d2normal = self.normal_derivatives_from_wall_value(field_halo, wall_value, geometry, layout)
        return dnormal

    def d2normal_from_wall_value(
        self,
        field_halo: jnp.ndarray,
        wall_value: LocalCoordinateSideValues3D,
        geometry: "LocalFciGeometry3D",
        layout: "HaloLayout3D",
    ) -> "LocalCoordinateSideValues3D":
        _dnormal, d2normal = self.normal_derivatives_from_wall_value(field_halo, wall_value, geometry, layout)
        return d2normal

    def normal_derivatives_from_wall_value(
        self,
        field_halo: jnp.ndarray,
        wall_value: LocalCoordinateSideValues3D,
        geometry: "LocalFciGeometry3D",
        layout: "HaloLayout3D",
    ) -> tuple["LocalCoordinateSideValues3D", "LocalCoordinateSideValues3D"]:
        values = _local_cell_halo_array(field_halo, layout, "field_halo")
        if geometry.layout != layout:
            raise ValueError("geometry and layout must share the same HaloLayout3D")
        if layout.halo_width < 1:
            raise ValueError("local normal-derivative reconstruction requires halo_width >= 1")
        if not isinstance(wall_value, LocalCoordinateSideValues3D):
            raise TypeError("wall_value must be a LocalCoordinateSideValues3D instance")

        x_dnormal_lower, x_dnormal_upper, x_d2_lower, x_d2_upper = self._wall_side_derivatives(
            values, wall_value.x, geometry, layout, axis=0
        )
        y_dnormal_lower, y_dnormal_upper, y_d2_lower, y_d2_upper = self._wall_side_derivatives(
            values, wall_value.y, geometry, layout, axis=1
        )
        z_dnormal_lower, z_dnormal_upper, z_d2_lower, z_d2_upper = self._wall_side_derivatives(
            values, wall_value.z, geometry, layout, axis=2
        )

        return (
            LocalCoordinateSideValues3D(
                x=LocalCoordinateSideValues1D(
                    lower=x_dnormal_lower,
                    upper=x_dnormal_upper,
                    mask_lower=wall_value.x.mask_lower,
                    mask_upper=wall_value.x.mask_upper,
                ),
                y=LocalCoordinateSideValues1D(
                    lower=y_dnormal_lower,
                    upper=y_dnormal_upper,
                    mask_lower=wall_value.y.mask_lower,
                    mask_upper=wall_value.y.mask_upper,
                ),
                z=LocalCoordinateSideValues1D(
                    lower=z_dnormal_lower,
                    upper=z_dnormal_upper,
                    mask_lower=wall_value.z.mask_lower,
                    mask_upper=wall_value.z.mask_upper,
                ),
            ),
            LocalCoordinateSideValues3D(
                x=LocalCoordinateSideValues1D(
                    lower=x_d2_lower,
                    upper=x_d2_upper,
                    mask_lower=wall_value.x.mask_lower,
                    mask_upper=wall_value.x.mask_upper,
                ),
                y=LocalCoordinateSideValues1D(
                    lower=y_d2_lower,
                    upper=y_d2_upper,
                    mask_lower=wall_value.y.mask_lower,
                    mask_upper=wall_value.y.mask_upper,
                ),
                z=LocalCoordinateSideValues1D(
                    lower=z_d2_lower,
                    upper=z_d2_upper,
                    mask_lower=wall_value.z.mask_lower,
                    mask_upper=wall_value.z.mask_upper,
                ),
            ),
        )

    def tree_flatten(self):
        return ((self.dnormal_weights, self.d2normal_weights), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@_pytree_base
@dataclass(frozen=True)
class CoordinateNormalDerivativeConstructor3D:
    """Dense coordinate-boundary normal derivative constructor.

    The stored weights have shape ``(3, 2, 4)``:
    axis 0 is x/y/z, axis 1 is lower/upper wall, and axis 2 multiplies
    ``(wall_value, first interior cell, second interior cell, third interior cell)``.
    These weights are the coordinate-face analogue of the cut-wall least-squares
    weights below: construction is geometry-dependent, application is just a
    linear combination of wall and cell-centered values.

    Sign convention:
        dnormal_from_wall_value(...) returns the outward normal derivative
        d/dn, not the inward coordinate derivative d/ds. The weight factory
        applies the sign flip once during construction so callers can use the
        physical outward-normal convention directly.
    """

    dnormal_weights: jnp.ndarray  # (axis=3, side=2, wall+3 interior nodes=4)
    d2normal_weights: jnp.ndarray  # (axis=3, side=2, wall+3 interior nodes=4)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dnormal_weights",
            _as_coordinate_derivative_weight_array(
                self.dnormal_weights,
                "CoordinateNormalDerivativeConstructor3D.dnormal_weights",
            ),
        )
        object.__setattr__(
            self,
            "d2normal_weights",
            _as_coordinate_derivative_weight_array(
                self.d2normal_weights,
                "CoordinateNormalDerivativeConstructor3D.d2normal_weights",
            ),
        )

    @classmethod
    def from_geometry(cls, geometry: FciGeometry3D) -> "CoordinateNormalDerivativeConstructor3D":
        """Build second-order wall-normal derivative weights from grid geometry."""

        if geometry.shape[0] < 3 or geometry.shape[1] < 3 or geometry.shape[2] < 3:
            raise ValueError(
                "CoordinateNormalDerivativeConstructor3D.from_geometry requires at least "
                f"three cells along every axis, got {geometry.shape}"
            )

        def _weights_for_side(a, b, c):
            nodes = jnp.asarray((0.0, a, b, c), dtype=jnp.float64)
            vandermonde = jnp.stack((nodes**0, nodes, nodes**2, nodes**3), axis=0)
            # Local coordinate s points inward from the wall towards computational domain. 
            #The returned first-derivative weights are already converted to the outward
            # normal convention pointing away from the computational domain, 
            #so d/dn = -d/ds is baked in once here.
            d1_weights = -jnp.linalg.solve(
                vandermonde,
                jnp.asarray((0.0, 1.0, 0.0, 0.0), dtype=jnp.float64),
            )
            # The second normal derivative has no sign flip:
            # d2/dn2 = d2/ds2.
            d2_weights = jnp.linalg.solve(
                vandermonde,
                jnp.asarray((0.0, 0.0, 2.0, 0.0), dtype=jnp.float64),
            )
            return d1_weights, d2_weights

        def _axis_weights(axis_grid):
            lower_a = jnp.asarray(axis_grid.centers[0] - axis_grid.faces[0], dtype=jnp.float64)
            lower_b = jnp.asarray(axis_grid.centers[1] - axis_grid.faces[0], dtype=jnp.float64)
            lower_c = jnp.asarray(axis_grid.centers[2] - axis_grid.faces[0], dtype=jnp.float64)
            upper_a = jnp.asarray(axis_grid.faces[-1] - axis_grid.centers[-1], dtype=jnp.float64)
            upper_b = jnp.asarray(axis_grid.faces[-1] - axis_grid.centers[-2], dtype=jnp.float64)
            upper_c = jnp.asarray(axis_grid.faces[-1] - axis_grid.centers[-3], dtype=jnp.float64)
            lower_d1, lower_d2 = _weights_for_side(lower_a, lower_b, lower_c)
            upper_d1, upper_d2 = _weights_for_side(upper_a, upper_b, upper_c)
            return jnp.stack((lower_d1, upper_d1), axis=0), jnp.stack((lower_d2, upper_d2), axis=0)

        x_d1, x_d2 = _axis_weights(geometry.grid.x)
        y_d1, y_d2 = _axis_weights(geometry.grid.y)
        z_d1, z_d2 = _axis_weights(geometry.grid.z)
        return cls(
            dnormal_weights=jnp.stack((x_d1, y_d1, z_d1), axis=0),
            d2normal_weights=jnp.stack((x_d2, y_d2, z_d2), axis=0),
        )

    def dnormal_from_wall_value(
        self,
        field: jnp.ndarray,
        wall_value: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        geometry: FciGeometry3D,
        periodic_axes: tuple[bool, bool, bool] = (False, True, True),
        axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dnormal, _d2normal = self.normal_derivatives_from_wall_value(
            field,
            wall_value,
            geometry,
            periodic_axes=periodic_axes,
            axis_regular_axes=axis_regular_axes,
        )
        return dnormal

    def d2normal_from_wall_value(
        self,
        field: jnp.ndarray,
        wall_value: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        geometry: FciGeometry3D,
        periodic_axes: tuple[bool, bool, bool] = (False, True, True),
        axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        _dnormal, d2normal = self.normal_derivatives_from_wall_value(
            field,
            wall_value,
            geometry,
            periodic_axes=periodic_axes,
            axis_regular_axes=axis_regular_axes,
        )
        return d2normal

    def normal_derivatives_from_wall_value(
        self,
        field: jnp.ndarray,
        wall_value: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        geometry: FciGeometry3D,
        periodic_axes: tuple[bool, bool, bool] = (False, True, True),
        axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    ) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ]:
        values = _as_float64_array(field, "CoordinateNormalDerivativeConstructor3D.field")
        if values.shape != geometry.shape:
            raise ValueError(f"field must have shape {geometry.shape}, got {values.shape}")
        periodic_axes = _normalize_axis_flags(periodic_axes, "periodic_axes")
        axis_regular_axes = _normalize_axis_flags(axis_regular_axes, "axis_regular_axes")
        if any(periodic and axis_regular for periodic, axis_regular in zip(periodic_axes, axis_regular_axes)):
            raise ValueError(
                "periodic_axes and axis_regular_axes cannot both be True on the same axis; "
                f"got periodic_axes={periodic_axes}, axis_regular_axes={axis_regular_axes}"
            )
        if axis_regular_axes[1] or axis_regular_axes[2]:
            raise ValueError(
                "axis_regular_axes currently only supports the lower x axis; "
                f"got axis_regular_axes={axis_regular_axes}"
            )
        fw_x, fw_y, fw_z = _as_coordinate_face_tuple(
            wall_value,
            geometry,
            "CoordinateNormalDerivativeConstructor3D.wall_value",
        )
        if axis_regular_axes[0]:
            fw_x = fw_x.at[0].set(_axis_regular_lower_x_face(values))

        dfn_x = jnp.zeros_like(fw_x, dtype=jnp.float64)
        dfn_y = jnp.zeros_like(fw_y, dtype=jnp.float64)
        dfn_z = jnp.zeros_like(fw_z, dtype=jnp.float64)
        d2fn_x = jnp.zeros_like(fw_x, dtype=jnp.float64)
        d2fn_y = jnp.zeros_like(fw_y, dtype=jnp.float64)
        d2fn_z = jnp.zeros_like(fw_z, dtype=jnp.float64)

        def _apply_weights(weights, f_wall, f0, f1, f2):
            return weights[0] * f_wall + weights[1] * f0 + weights[2] * f1 + weights[3] * f2

        def _wall_derivatives(axis: int, side: int, f_wall, f0, f1, f2):
            dnormal = _apply_weights(self.dnormal_weights[axis, side], f_wall, f0, f1, f2)
            d2normal = _apply_weights(self.d2normal_weights[axis, side], f_wall, f0, f1, f2)
            return dnormal, d2normal

        if not periodic_axes[0]:
            if values.shape[0] < 3:
                raise ValueError("x-normal derivative reconstruction requires at least three x cells")
            lower_dfn, lower_d2fn = _wall_derivatives(0, 0, fw_x[0], values[0], values[1], values[2])
            upper_dfn, upper_d2fn = _wall_derivatives(0, 1, fw_x[-1], values[-1], values[-2], values[-3])
            dfn_x = dfn_x.at[0].set(lower_dfn)
            dfn_x = dfn_x.at[-1].set(upper_dfn)
            d2fn_x = d2fn_x.at[0].set(lower_d2fn)
            d2fn_x = d2fn_x.at[-1].set(upper_d2fn)

        if not periodic_axes[1]:
            if values.shape[1] < 3:
                raise ValueError("y-normal derivative reconstruction requires at least three y cells")
            lower_dfn, lower_d2fn = _wall_derivatives(1, 0, fw_y[:, 0, :], values[:, 0, :], values[:, 1, :], values[:, 2, :])
            upper_dfn, upper_d2fn = _wall_derivatives(1, 1, fw_y[:, -1, :], values[:, -1, :], values[:, -2, :], values[:, -3, :])
            dfn_y = dfn_y.at[:, 0, :].set(lower_dfn)
            dfn_y = dfn_y.at[:, -1, :].set(upper_dfn)
            d2fn_y = d2fn_y.at[:, 0, :].set(lower_d2fn)
            d2fn_y = d2fn_y.at[:, -1, :].set(upper_d2fn)

        if not periodic_axes[2]:
            if values.shape[2] < 3:
                raise ValueError("z-normal derivative reconstruction requires at least three z cells")
            lower_dfn, lower_d2fn = _wall_derivatives(2, 0, fw_z[:, :, 0], values[:, :, 0], values[:, :, 1], values[:, :, 2])
            upper_dfn, upper_d2fn = _wall_derivatives(2, 1, fw_z[:, :, -1], values[:, :, -1], values[:, :, -2], values[:, :, -3])
            dfn_z = dfn_z.at[:, :, 0].set(lower_dfn)
            dfn_z = dfn_z.at[:, :, -1].set(upper_dfn)
            d2fn_z = d2fn_z.at[:, :, 0].set(lower_d2fn)
            d2fn_z = d2fn_z.at[:, :, -1].set(upper_d2fn)

        return (dfn_x, dfn_y, dfn_z), (d2fn_x, d2fn_y, d2fn_z)

    def tree_flatten(self):
        return ((self.dnormal_weights, self.d2normal_weights), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@_pytree_base
@dataclass(frozen=True)
class CutWallValueReconstructor3D:
    """Batched cut-wall value reconstructor for cell-centered fields."""

    cut_wall_geometry: "CutWallGeometry3D"  # geometry describing the cut-wall faces
    neighbor_i: jnp.ndarray  # (n_wall_faces, n_stencil)
    neighbor_j: jnp.ndarray  # (n_wall_faces, n_stencil)
    neighbor_k: jnp.ndarray  # (n_wall_faces, n_stencil)
    weights: jnp.ndarray  # (n_wall_faces, n_stencil)
    bc_coeff: jnp.ndarray | None = None  # (n_wall_faces,) or None for pure extrapolation

    def __post_init__(self) -> None:
        if not isinstance(self.cut_wall_geometry, CutWallGeometry3D):
            raise TypeError("CutWallValueReconstructor3D.cut_wall_geometry must be a CutWallGeometry3D")
        neighbor_i = _as_int_stencil_array(self.neighbor_i, "CutWallValueReconstructor3D.neighbor_i")
        neighbor_j = _as_int_stencil_array(self.neighbor_j, "CutWallValueReconstructor3D.neighbor_j")
        neighbor_k = _as_int_stencil_array(self.neighbor_k, "CutWallValueReconstructor3D.neighbor_k")
        weights = _as_weight_stencil_array(self.weights, "CutWallValueReconstructor3D.weights")
        if neighbor_i.shape != neighbor_j.shape or neighbor_i.shape != neighbor_k.shape or neighbor_i.shape != weights.shape:
            raise ValueError("cut-wall neighbor and weight arrays must all have the same shape")
        if neighbor_i.shape[0] != self.cut_wall_geometry.n_wall_faces:
            raise ValueError(
                "cut-wall neighbor arrays must have one row per cut-wall face; "
                f"got {neighbor_i.shape[0]} rows and {self.cut_wall_geometry.n_wall_faces} faces"
            )
        object.__setattr__(self, "neighbor_i", neighbor_i)
        object.__setattr__(self, "neighbor_j", neighbor_j)
        object.__setattr__(self, "neighbor_k", neighbor_k)
        object.__setattr__(self, "weights", weights)
        if self.bc_coeff is not None:
            bc_coeff = _as_wall_face_array(
                self.bc_coeff,
                self.cut_wall_geometry.n_wall_faces,
                "CutWallValueReconstructor3D.bc_coeff",
            )
            object.__setattr__(self, "bc_coeff", bc_coeff)

    @property
    def n_wall_faces(self) -> int:
        return self.cut_wall_geometry.n_wall_faces

    def extrapolate(self, field: jnp.ndarray) -> jnp.ndarray:
        values = _as_float64_array(field, "CutWallValueReconstructor3D.field")
        gathered = values[self.neighbor_i, self.neighbor_j, self.neighbor_k]
        return jnp.sum(self.weights * gathered, axis=1)

    def extrapolate_neumann(
        self,
        field: jnp.ndarray,
        normal_derivative: jnp.ndarray | float = 0.0,
    ) -> jnp.ndarray:
        if self.bc_coeff is None:
            raise ValueError("CutWallValueReconstructor3D.extrapolate_neumann requires bc_coeff")
        base_value = self.extrapolate(field)
        bc_value = _as_wall_face_array(
            normal_derivative,
            self.cut_wall_geometry.n_wall_faces,
            "CutWallValueReconstructor3D.normal_derivative",
        )
        return base_value + self.bc_coeff * bc_value

    def tree_flatten(self):
        children = [self.cut_wall_geometry, self.neighbor_i, self.neighbor_j, self.neighbor_k, self.weights]
        aux = self.bc_coeff is None
        if self.bc_coeff is not None:
            children.append(self.bc_coeff)
        return tuple(children), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        bc_coeff_is_none = aux_data
        if bc_coeff_is_none:
            cut_wall_geometry, neighbor_i, neighbor_j, neighbor_k, weights = children
            return cls(cut_wall_geometry, neighbor_i, neighbor_j, neighbor_k, weights, None)
        cut_wall_geometry, neighbor_i, neighbor_j, neighbor_k, weights, bc_coeff = children
        return cls(cut_wall_geometry, neighbor_i, neighbor_j, neighbor_k, weights, bc_coeff)


@_pytree_base
@dataclass(frozen=True)
class CutWallNormalDerivativeConstructor3D:
    """Batched cut-wall normal derivative constructor for cell-centered fields.

    Sign convention:
        dnormal_from_wall_value(...) returns the outward normal derivative
        d/dn, not the inward coordinate derivative d/ds. Any cut-wall
        weight construction should therefore encode the outward normal sign
        convention in the weights provided here.
    """

    cut_wall_geometry: "CutWallGeometry3D"
    neighbor_i: jnp.ndarray  # (n_wall_faces, n_stencil)
    neighbor_j: jnp.ndarray  # (n_wall_faces, n_stencil)
    neighbor_k: jnp.ndarray  # (n_wall_faces, n_stencil)
    weights_dnormal: jnp.ndarray  # (n_wall_faces, n_stencil)
    weights_d2normal: jnp.ndarray  # (n_wall_faces, n_stencil)
    wall_coeff_dnormal: jnp.ndarray  # (n_wall_faces,)
    wall_coeff_d2normal: jnp.ndarray  # (n_wall_faces,)

    def __post_init__(self) -> None:
        if not isinstance(self.cut_wall_geometry, CutWallGeometry3D):
            raise TypeError("CutWallNormalDerivativeConstructor3D.cut_wall_geometry must be a CutWallGeometry3D")
        neighbor_i = _as_int_stencil_array(self.neighbor_i, "CutWallNormalDerivativeConstructor3D.neighbor_i")
        neighbor_j = _as_int_stencil_array(self.neighbor_j, "CutWallNormalDerivativeConstructor3D.neighbor_j")
        neighbor_k = _as_int_stencil_array(self.neighbor_k, "CutWallNormalDerivativeConstructor3D.neighbor_k")
        weights_dnormal = _as_weight_stencil_array(
            self.weights_dnormal,
            "CutWallNormalDerivativeConstructor3D.weights_dnormal",
        )
        weights_d2normal = _as_weight_stencil_array(
            self.weights_d2normal,
            "CutWallNormalDerivativeConstructor3D.weights_d2normal",
        )
        expected_shape = neighbor_i.shape
        for name, value in (
            ("neighbor_j", neighbor_j),
            ("neighbor_k", neighbor_k),
            ("weights_dnormal", weights_dnormal),
            ("weights_d2normal", weights_d2normal),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    "cut-wall normal derivative neighbor and weight arrays must all have "
                    f"shape {expected_shape}; {name} has shape {value.shape}"
                )
        if expected_shape[0] != self.cut_wall_geometry.n_wall_faces:
            raise ValueError(
                "cut-wall normal derivative arrays must have one row per cut-wall face; "
                f"got {expected_shape[0]} rows and {self.cut_wall_geometry.n_wall_faces} faces"
            )
        wall_coeff_dnormal = _as_wall_face_array(
            self.wall_coeff_dnormal,
            self.cut_wall_geometry.n_wall_faces,
            "CutWallNormalDerivativeConstructor3D.wall_coeff_dnormal",
        )
        wall_coeff_d2normal = _as_wall_face_array(
            self.wall_coeff_d2normal,
            self.cut_wall_geometry.n_wall_faces,
            "CutWallNormalDerivativeConstructor3D.wall_coeff_d2normal",
        )
        object.__setattr__(self, "neighbor_i", neighbor_i)
        object.__setattr__(self, "neighbor_j", neighbor_j)
        object.__setattr__(self, "neighbor_k", neighbor_k)
        object.__setattr__(self, "weights_dnormal", weights_dnormal)
        object.__setattr__(self, "weights_d2normal", weights_d2normal)
        object.__setattr__(self, "wall_coeff_dnormal", wall_coeff_dnormal)
        object.__setattr__(self, "wall_coeff_d2normal", wall_coeff_d2normal)

    @property
    def n_wall_faces(self) -> int:
        return self.cut_wall_geometry.n_wall_faces

    def dnormal_from_wall_value(
        self,
        field: jnp.ndarray,
        wall_value: jnp.ndarray,
    ) -> jnp.ndarray:
        values = _as_float64_array(field, "CutWallNormalDerivativeConstructor3D.field")
        wall = _as_wall_face_array(
            wall_value,
            self.cut_wall_geometry.n_wall_faces,
            "CutWallNormalDerivativeConstructor3D.wall_value",
        )
        gathered = values[self.neighbor_i, self.neighbor_j, self.neighbor_k]
        return jnp.sum(self.weights_dnormal * gathered, axis=1) + self.wall_coeff_dnormal * wall

    def d2normal_from_wall_value(
        self,
        field: jnp.ndarray,
        wall_value: jnp.ndarray,
    ) -> jnp.ndarray:
        values = _as_float64_array(field, "CutWallNormalDerivativeConstructor3D.field")
        wall = _as_wall_face_array(
            wall_value,
            self.cut_wall_geometry.n_wall_faces,
            "CutWallNormalDerivativeConstructor3D.wall_value",
        )
        gathered = values[self.neighbor_i, self.neighbor_j, self.neighbor_k]
        return jnp.sum(self.weights_d2normal * gathered, axis=1) + self.wall_coeff_d2normal * wall

    def normal_derivatives_from_wall_value(
        self,
        field: jnp.ndarray,
        wall_value: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return (
            self.dnormal_from_wall_value(field, wall_value),
            self.d2normal_from_wall_value(field, wall_value),
        )

    def tree_flatten(self):
        return (
            (
                self.cut_wall_geometry,
                self.neighbor_i,
                self.neighbor_j,
                self.neighbor_k,
                self.weights_dnormal,
                self.weights_d2normal,
                self.wall_coeff_dnormal,
                self.wall_coeff_d2normal,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@_pytree_base
@dataclass(frozen=True)
class LocalStencil1D:
    """Field-dependent 1D stencil values for one coordinate direction."""

    center: jnp.ndarray
    minus: jnp.ndarray
    plus: jnp.ndarray
    dx_min: jnp.ndarray
    dx_plus: jnp.ndarray
    derivative_minus_weight: jnp.ndarray | None = None
    derivative_center_weight: jnp.ndarray | None = None
    derivative_plus_weight: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        center = _as_float64_array(self.center, "center")
        shape = center.shape

        for name in ("minus", "plus", "dx_min", "dx_plus"):
            value = _as_float64_array(getattr(self, name), f"LocalStencil1D.{name}")
            if value.shape != shape:
                raise ValueError(
                    f"LocalStencil1D.{name} must have shape {shape}, got {value.shape}"
                )
            object.__setattr__(self, name, value)

        derivative_weight_names = (
            "derivative_minus_weight",
            "derivative_center_weight",
            "derivative_plus_weight",
        )
        supplied_weights = [getattr(self, name) is not None for name in derivative_weight_names]
        if any(supplied_weights) and not all(supplied_weights):
            raise ValueError(
                "LocalStencil1D derivative weights must either all be supplied or all be omitted"
            )

        if all(supplied_weights):
            for name in derivative_weight_names:
                value = _as_float64_array(getattr(self, name), f"LocalStencil1D.{name}")
                if value.shape != shape:
                    raise ValueError(
                        f"LocalStencil1D.{name} must have shape {shape}, got {value.shape}"
                    )
                object.__setattr__(self, name, value)
        else:
            dx_min = jnp.asarray(self.dx_min, dtype=jnp.float64)
            dx_plus = jnp.asarray(self.dx_plus, dtype=jnp.float64)
            denom = jnp.maximum(dx_min * dx_plus * (dx_min + dx_plus), 1.0e-30)
            object.__setattr__(self, "derivative_minus_weight", -dx_plus * dx_plus / denom)
            object.__setattr__(
                self,
                "derivative_center_weight",
                (dx_plus * dx_plus - dx_min * dx_min) / denom,
            )
            object.__setattr__(self, "derivative_plus_weight", dx_min * dx_min / denom)

        object.__setattr__(self, "center", center)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.center.shape)

    def replace(self, **updates: object) -> "LocalStencil1D":
        """Return a new stencil with one or more fields replaced."""

        allowed = {
            "center",
            "minus",
            "plus",
            "dx_min",
            "dx_plus",
            "derivative_minus_weight",
            "derivative_center_weight",
            "derivative_plus_weight",
        }
        unknown = set(updates) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown LocalStencil1D field(s): {names}")
        return dataclass_replace_1d(self, **updates)

    def tree_flatten(self):
        return (
            (
                self.center,
                self.minus,
                self.plus,
                self.dx_min,
                self.dx_plus,
                self.derivative_minus_weight,
                self.derivative_center_weight,
                self.derivative_plus_weight,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@_pytree_base
@dataclass(frozen=True)
class LocalStencil3D:
    """Nested 3D stencil for local/pointwise operators.

    This stencil is the reconstruction object for local derivatives and
    pointwise operators. It may be boundary-aware, including embedded-wall
    aware reconstruction logic in the caller that builds it.
    """

    x: LocalStencil1D
    y: LocalStencil1D
    z: LocalStencil1D

    def __post_init__(self) -> None:
        if not isinstance(self.x, LocalStencil1D):
            raise TypeError("LocalStencil3D.x must be a LocalStencil1D")
        if not isinstance(self.y, LocalStencil1D):
            raise TypeError("LocalStencil3D.y must be a LocalStencil1D")
        if not isinstance(self.z, LocalStencil1D):
            raise TypeError("LocalStencil3D.z must be a LocalStencil1D")
        if self.x.shape != self.y.shape or self.x.shape != self.z.shape:
            raise ValueError(
                "LocalStencil3D axis stencils must all have the same shape; "
                f"got x={self.x.shape}, y={self.y.shape}, z={self.z.shape}"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.x.shape

    def replace(self, **updates: object) -> "LocalStencil3D":
        """Return a new stencil with one or more fields replaced."""

        allowed = {"x", "y", "z"}
        unknown = set(updates) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown LocalStencil3D field(s): {names}")
        return dataclass_replace_3d(self, **updates)

    def tree_flatten(self):
        return ((self.x, self.y, self.z), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@_pytree_base
@dataclass(frozen=True)
class LocalCellGradient3D:
    """Owned-cell reconstructed gradient for local derivative operators.

    Unlike ``LocalStencil3D``, this object stores the reconstructed derivative
    itself. Builders may use polynomial fits, wall samples, or other
    reconstruction strategies internally.
    """

    gradient: jnp.ndarray
    valid: jnp.ndarray
    reconstruction_mask: jnp.ndarray

    def __post_init__(self) -> None:
        gradient = jnp.asarray(self.gradient, dtype=jnp.float64)
        if gradient.ndim != 4 or gradient.shape[-1] != 3:
            raise ValueError(
                "LocalCellGradient3D.gradient must have shape "
                f"owned_shape + (3,), got {gradient.shape}"
            )
        owned_shape = gradient.shape[:-1]
        valid = jnp.asarray(self.valid, dtype=bool)
        reconstruction_mask = jnp.asarray(self.reconstruction_mask, dtype=bool)
        if valid.shape != owned_shape:
            raise ValueError(
                "LocalCellGradient3D.valid must match gradient owned shape; "
                f"got {valid.shape}, expected {owned_shape}"
            )
        if reconstruction_mask.shape != owned_shape:
            raise ValueError(
                "LocalCellGradient3D.reconstruction_mask must match gradient owned shape; "
                f"got {reconstruction_mask.shape}, expected {owned_shape}"
            )
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "reconstruction_mask", reconstruction_mask)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.gradient.shape[:-1])

    def tree_flatten(self):
        return ((self.gradient, self.valid, self.reconstruction_mask), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@_pytree_base
@dataclass(frozen=True)
class FaceGradientStencil3D:
    """Face-centered coordinate gradients for a scalar field."""

    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray

    def __post_init__(self) -> None:
        x = jnp.asarray(self.x, dtype=jnp.float64)
        y = jnp.asarray(self.y, dtype=jnp.float64)
        z = jnp.asarray(self.z, dtype=jnp.float64)

        for name, value in (("x", x), ("y", y), ("z", z)):
            if value.ndim != 4 or value.shape[-1] != 3:
                raise ValueError(
                    f"FaceGradientStencil3D.{name} must have shape (nx, ny, nz, 3), got {value.shape}"
                )

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "z", z)

    @property
    def shape(self) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        return tuple(int(v) for v in self.x.shape[:-1]), tuple(int(v) for v in self.y.shape[:-1]), tuple(int(v) for v in self.z.shape[:-1])

    def tree_flatten(self):
        return ((self.x, self.y, self.z), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@_pytree_base
@dataclass(frozen=True)
class ConservativeStencil3D:
    """Nested 3D stencil for conservative operators.

    This is intentionally separate from ``LocalStencil3D`` so conservative
    flux assembly cannot accidentally inherit embedded-wall reconstruction
    semantics. The payload is still cell-centered reconstruction data, but the
    meaning is restricted to regular control-volume flux construction.
    """

    x: LocalStencil1D
    y: LocalStencil1D
    z: LocalStencil1D
    face_grad: FaceGradientStencil3D

    def __post_init__(self) -> None:
        if not isinstance(self.x, LocalStencil1D):
            raise TypeError("ConservativeStencil3D.x must be a LocalStencil1D")
        if not isinstance(self.y, LocalStencil1D):
            raise TypeError("ConservativeStencil3D.y must be a LocalStencil1D")
        if not isinstance(self.z, LocalStencil1D):
            raise TypeError("ConservativeStencil3D.z must be a LocalStencil1D")
        if not isinstance(self.face_grad, FaceGradientStencil3D):
            raise TypeError("ConservativeStencil3D.face_grad must be a FaceGradientStencil3D")
        if self.x.shape != self.y.shape or self.x.shape != self.z.shape:
            raise ValueError(
                "ConservativeStencil3D axis stencils must all have the same shape; "
                f"got x={self.x.shape}, y={self.y.shape}, z={self.z.shape}"
            )
        expected_x = (self.x.shape[0] + 1, self.x.shape[1], self.x.shape[2])
        expected_y = (self.x.shape[0], self.x.shape[1] + 1, self.x.shape[2])
        expected_z = (self.x.shape[0], self.x.shape[1], self.x.shape[2] + 1)
        if self.face_grad.x.shape[:-1] != expected_x:
            raise ValueError(
                f"ConservativeStencil3D.face_grad.x must have shape {expected_x + (3,)}, got {self.face_grad.x.shape}"
            )
        if self.face_grad.y.shape[:-1] != expected_y:
            raise ValueError(
                f"ConservativeStencil3D.face_grad.y must have shape {expected_y + (3,)}, got {self.face_grad.y.shape}"
            )
        if self.face_grad.z.shape[:-1] != expected_z:
            raise ValueError(
                f"ConservativeStencil3D.face_grad.z must have shape {expected_z + (3,)}, got {self.face_grad.z.shape}"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.x.shape

    def replace(self, **updates: object) -> "ConservativeStencil3D":
        """Return a new stencil with one or more fields replaced."""

        allowed = {"x", "y", "z", "face_grad"}
        unknown = set(updates) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown ConservativeStencil3D field(s): {names}")
        return dataclass_replace_conservative(self, **updates)

    def tree_flatten(self):
        return ((self.x, self.y, self.z, self.face_grad), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@_pytree_base
@dataclass(frozen=True)
class FaceFluxStencil3D:
    """Face-centered flux arrays for the three coordinate directions.

    In the local domain-decomposed path, the inferred shape is the local
    owned-cell shape. These are the control faces needed to update owned
    cells, not necessarily uniquely owned global faces.
    """

    x: jnp.ndarray
    y: jnp.ndarray
    z: jnp.ndarray

    def __post_init__(self) -> None:
        x = _as_face_flux_array(self.x, "FaceFluxStencil3D.x")
        y = _as_face_flux_array(self.y, "FaceFluxStencil3D.y")
        z = _as_face_flux_array(self.z, "FaceFluxStencil3D.z")

        cell_shape = (x.shape[0] - 1, y.shape[1] - 1, z.shape[2] - 1)
        expected_x = (cell_shape[0] + 1, cell_shape[1], cell_shape[2])
        expected_y = (cell_shape[0], cell_shape[1] + 1, cell_shape[2])
        expected_z = (cell_shape[0], cell_shape[1], cell_shape[2] + 1)
        if x.shape != expected_x or y.shape != expected_y or z.shape != expected_z:
            raise ValueError(
                "FaceFluxStencil3D axis shapes must match the face-grid layout; "
                f"expected x={expected_x}, y={expected_y}, z={expected_z}, got "
                f"x={x.shape}, y={y.shape}, z={z.shape}"
            )

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "z", z)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.x.shape[0] - 1), int(self.y.shape[1] - 1), int(self.z.shape[2] - 1))

    def tree_flatten(self):
        return ((self.x, self.y, self.z), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@_pytree_base
@dataclass(frozen=True)
class BoundaryFaceBC3D:
    """Dense face-grid boundary-condition data for regular coordinate faces."""

    kind_x: jnp.ndarray  # (nx + 1, ny, nz)
    kind_y: jnp.ndarray  # (nx, ny + 1, nz)
    kind_z: jnp.ndarray  # (nx, ny, nz + 1)
    value_x: jnp.ndarray  # (nx + 1, ny, nz)
    value_y: jnp.ndarray  # (nx, ny + 1, nz)
    value_z: jnp.ndarray  # (nx, ny, nz + 1)
    mask_x: jnp.ndarray  # (nx + 1, ny, nz)
    mask_y: jnp.ndarray  # (nx, ny + 1, nz)
    mask_z: jnp.ndarray  # (nx, ny, nz + 1)

    def __post_init__(self) -> None:
        kind_x = _as_int_face_array(self.kind_x, "BoundaryFaceBC3D.kind_x")
        kind_y = _as_int_face_array(self.kind_y, "BoundaryFaceBC3D.kind_y")
        kind_z = _as_int_face_array(self.kind_z, "BoundaryFaceBC3D.kind_z")
        value_x = _as_face_flux_array(self.value_x, "BoundaryFaceBC3D.value_x")
        value_y = _as_face_flux_array(self.value_y, "BoundaryFaceBC3D.value_y")
        value_z = _as_face_flux_array(self.value_z, "BoundaryFaceBC3D.value_z")
        mask_x = _as_bool_face_array(self.mask_x, "BoundaryFaceBC3D.mask_x")
        mask_y = _as_bool_face_array(self.mask_y, "BoundaryFaceBC3D.mask_y")
        mask_z = _as_bool_face_array(self.mask_z, "BoundaryFaceBC3D.mask_z")
        if kind_x.shape != value_x.shape or kind_x.shape != mask_x.shape:
            raise ValueError("BoundaryFaceBC3D.x arrays must all have the same shape")
        if kind_y.shape != value_y.shape or kind_y.shape != mask_y.shape:
            raise ValueError("BoundaryFaceBC3D.y arrays must all have the same shape")
        if kind_z.shape != value_z.shape or kind_z.shape != mask_z.shape:
            raise ValueError("BoundaryFaceBC3D.z arrays must all have the same shape")
        object.__setattr__(self, "kind_x", kind_x)
        object.__setattr__(self, "kind_y", kind_y)
        object.__setattr__(self, "kind_z", kind_z)
        object.__setattr__(self, "value_x", value_x)
        object.__setattr__(self, "value_y", value_y)
        object.__setattr__(self, "value_z", value_z)
        object.__setattr__(self, "mask_x", mask_x)
        object.__setattr__(self, "mask_y", mask_y)
        object.__setattr__(self, "mask_z", mask_z)

    @classmethod
    def empty(cls, geometry: RegularFaceGeometry3D) -> "BoundaryFaceBC3D":
        return cls(
            kind_x=jnp.zeros_like(geometry.x_area, dtype=jnp.int32),
            kind_y=jnp.zeros_like(geometry.y_area, dtype=jnp.int32),
            kind_z=jnp.zeros_like(geometry.z_area, dtype=jnp.int32),
            value_x=jnp.zeros_like(geometry.x_area, dtype=jnp.float64),
            value_y=jnp.zeros_like(geometry.y_area, dtype=jnp.float64),
            value_z=jnp.zeros_like(geometry.z_area, dtype=jnp.float64),
            mask_x=jnp.zeros_like(geometry.x_open_mask, dtype=bool),
            mask_y=jnp.zeros_like(geometry.y_open_mask, dtype=bool),
            mask_z=jnp.zeros_like(geometry.z_open_mask, dtype=bool),
        )

    def tree_flatten(self):
        return (
            (
                self.kind_x,
                self.kind_y,
                self.kind_z,
                self.value_x,
                self.value_y,
                self.value_z,
                self.mask_x,
                self.mask_y,
                self.mask_z,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)

    def replace(self, **updates: object) -> "BoundaryFaceBC3D":
        allowed = {
            "kind_x",
            "kind_y",
            "kind_z",
            "value_x",
            "value_y",
            "value_z",
            "mask_x",
            "mask_y",
            "mask_z",
        }
        unknown = set(updates) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown BoundaryFaceBC3D field(s): {names}")
        return BoundaryFaceBC3D(
            kind_x=updates.get("kind_x", self.kind_x),
            kind_y=updates.get("kind_y", self.kind_y),
            kind_z=updates.get("kind_z", self.kind_z),
            value_x=updates.get("value_x", self.value_x),
            value_y=updates.get("value_y", self.value_y),
            value_z=updates.get("value_z", self.value_z),
            mask_x=updates.get("mask_x", self.mask_x),
            mask_y=updates.get("mask_y", self.mask_y),
            mask_z=updates.get("mask_z", self.mask_z),
        )


@_pytree_base
@dataclass(frozen=True)
class LocalBoundaryFaceBC3D(_DataclassPyTreeMixin):
    """
    Local physical regular-coordinate face boundary-condition payload.

    Arrays are dense over local owned control faces, but masks are true only on
    true physical coordinate boundary faces touched by this local shard.

    Internal shard interfaces, periodic interfaces, axis/topological fills, and
    ordinary interior faces must have mask=False.

    This object is the single source of truth for physical coordinate-face BCs.
    It is consumed by conservative flux builders and, optionally, by ghost-cell
    fillers for operators that choose ghost-materialized BC enforcement.
    """

    kind_x: jnp.ndarray
    kind_y: jnp.ndarray
    kind_z: jnp.ndarray
    value_x: jnp.ndarray
    value_y: jnp.ndarray
    value_z: jnp.ndarray
    mask_x: jnp.ndarray
    mask_y: jnp.ndarray
    mask_z: jnp.ndarray
    layout: HaloLayout3D

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")

        expected_x = self.layout.face_control_shape(axis=0)
        expected_y = self.layout.face_control_shape(axis=1)
        expected_z = self.layout.face_control_shape(axis=2)

        kind_x = jnp.asarray(self.kind_x, dtype=jnp.int32)
        kind_y = jnp.asarray(self.kind_y, dtype=jnp.int32)
        kind_z = jnp.asarray(self.kind_z, dtype=jnp.int32)
        value_x = jnp.asarray(self.value_x, dtype=jnp.float64)
        value_y = jnp.asarray(self.value_y, dtype=jnp.float64)
        value_z = jnp.asarray(self.value_z, dtype=jnp.float64)
        mask_x = jnp.asarray(self.mask_x, dtype=bool)
        mask_y = jnp.asarray(self.mask_y, dtype=bool)
        mask_z = jnp.asarray(self.mask_z, dtype=bool)

        for name, value, expected in (
            ("kind_x", kind_x, expected_x),
            ("kind_y", kind_y, expected_y),
            ("kind_z", kind_z, expected_z),
            ("value_x", value_x, expected_x),
            ("value_y", value_y, expected_y),
            ("value_z", value_z, expected_z),
            ("mask_x", mask_x, expected_x),
            ("mask_y", mask_y, expected_y),
            ("mask_z", mask_z, expected_z),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"LocalBoundaryFaceBC3D.{name} must have shape {expected}, got {value.shape}"
                )

        kind_x = jnp.where(mask_x, kind_x, BC_NONE)
        kind_y = jnp.where(mask_y, kind_y, BC_NONE)
        kind_z = jnp.where(mask_z, kind_z, BC_NONE)
        value_x = jnp.where(mask_x, value_x, 0.0)
        value_y = jnp.where(mask_y, value_y, 0.0)
        value_z = jnp.where(mask_z, value_z, 0.0)

        object.__setattr__(self, "kind_x", kind_x)
        object.__setattr__(self, "kind_y", kind_y)
        object.__setattr__(self, "kind_z", kind_z)
        object.__setattr__(self, "value_x", value_x)
        object.__setattr__(self, "value_y", value_y)
        object.__setattr__(self, "value_z", value_z)
        object.__setattr__(self, "mask_x", mask_x)
        object.__setattr__(self, "mask_y", mask_y)
        object.__setattr__(self, "mask_z", mask_z)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @classmethod
    def empty(cls, layout: HaloLayout3D) -> "LocalBoundaryFaceBC3D":
        x_shape = layout.face_control_shape(axis=0)
        y_shape = layout.face_control_shape(axis=1)
        z_shape = layout.face_control_shape(axis=2)
        return cls(
            kind_x=jnp.zeros(x_shape, dtype=jnp.int32),
            kind_y=jnp.zeros(y_shape, dtype=jnp.int32),
            kind_z=jnp.zeros(z_shape, dtype=jnp.int32),
            value_x=jnp.zeros(x_shape, dtype=jnp.float64),
            value_y=jnp.zeros(y_shape, dtype=jnp.float64),
            value_z=jnp.zeros(z_shape, dtype=jnp.float64),
            mask_x=jnp.zeros(x_shape, dtype=bool),
            mask_y=jnp.zeros(y_shape, dtype=bool),
            mask_z=jnp.zeros(z_shape, dtype=bool),
            layout=layout,
        )

    def tree_flatten(self):
        children = (
            self.kind_x,
            self.kind_y,
            self.kind_z,
            self.value_x,
            self.value_y,
            self.value_z,
            self.mask_x,
            self.mask_y,
            self.mask_z,
        )
        aux_data = self.layout
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        layout = aux_data
        (
            kind_x,
            kind_y,
            kind_z,
            value_x,
            value_y,
            value_z,
            mask_x,
            mask_y,
            mask_z,
        ) = children
        return cls(
            kind_x=kind_x,
            kind_y=kind_y,
            kind_z=kind_z,
            value_x=value_x,
            value_y=value_y,
            value_z=value_z,
            mask_x=mask_x,
            mask_y=mask_y,
            mask_z=mask_z,
            layout=layout,
        )


@_pytree_base
@dataclass(frozen=True)
class BoundaryConditionBuilder(Generic[BoundaryPayloadT]):
    """Callable adapter that delegates boundary-payload construction to an injected function."""

    build_fn: Callable[
        [
            Any,
            "FciGeometry3D",
            tuple[bool | None, bool | None, bool | None] | None,
            "CutWallGeometry3D | None",
            "CutWallBC3D | None",
        ],
        BoundaryPayloadT,
    ]

    def __call__(
        self,
        state: Any,
        geometry: "FciGeometry3D",
        periodic_axes: tuple[bool | None, bool | None, bool | None] | None,
        cut_wall_geometry: "CutWallGeometry3D | None",
        cut_wall_bc: "CutWallBC3D | None",
    ) -> BoundaryPayloadT:
        return self.build_fn(state, geometry, periodic_axes, cut_wall_geometry, cut_wall_bc)

    def tree_flatten(self):
        return (), self.build_fn

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data)


@_pytree_base
@dataclass(frozen=True)
class CutWallGeometry3D:
    """Geometry and metric data for true non-coordinate cut-wall faces."""

    owner_i: jnp.ndarray
    owner_j: jnp.ndarray
    owner_k: jnp.ndarray
    center: jnp.ndarray  # (n_wall_faces, 3)
    normal_contra: jnp.ndarray  # (n_wall_faces, 3), outward normal from the computational domain
    area_covector: jnp.ndarray  # (n_wall_faces, 3), outward-oriented area covector
    distance: jnp.ndarray  # (n_wall_faces,)
    J: jnp.ndarray  # (n_wall_faces,)
    g_contra: jnp.ndarray  # (n_wall_faces, 3, 3)
    g_cov: jnp.ndarray  # (n_wall_faces, 3, 3)
    B_contra: jnp.ndarray  # (n_wall_faces, 3)
    Bmag: jnp.ndarray  # (n_wall_faces,)
    sign: jnp.ndarray  # (n_wall_faces,), orientation sign relative to the owner cell / outward domain normal

    def __post_init__(self) -> None:
        owner_i = jnp.asarray(self.owner_i, dtype=jnp.int32)
        owner_j = jnp.asarray(self.owner_j, dtype=jnp.int32)
        owner_k = jnp.asarray(self.owner_k, dtype=jnp.int32)
        center = jnp.asarray(self.center, dtype=jnp.float64)
        normal_contra = jnp.asarray(self.normal_contra, dtype=jnp.float64)
        area_covector = jnp.asarray(self.area_covector, dtype=jnp.float64)
        distance = jnp.asarray(self.distance, dtype=jnp.float64)
        J = jnp.asarray(self.J, dtype=jnp.float64)
        g_contra = jnp.asarray(self.g_contra, dtype=jnp.float64)
        g_cov = jnp.asarray(self.g_cov, dtype=jnp.float64)
        B_contra = jnp.asarray(self.B_contra, dtype=jnp.float64)
        Bmag = jnp.asarray(self.Bmag, dtype=jnp.float64)
        sign = jnp.asarray(self.sign, dtype=jnp.float64)
        shape = owner_i.shape
        for name, value in (
            ("owner_j", owner_j),
            ("owner_k", owner_k),
            ("center", center),
            ("normal_contra", normal_contra),
            ("area_covector", area_covector),
            ("distance", distance),
            ("J", J),
            ("g_contra", g_contra),
            ("g_cov", g_cov),
            ("B_contra", B_contra),
            ("Bmag", Bmag),
            ("sign", sign),
        ):
            if name in {"center", "normal_contra", "area_covector", "B_contra"}:
                expected = shape + (3,)
            elif name in {"g_contra", "g_cov"}:
                expected = shape + (3, 3)
            else:
                expected = shape
            if value.shape != expected:
                raise ValueError(f"CutWallGeometry3D.{name} must have shape {expected}, got {value.shape}")
        object.__setattr__(self, "owner_i", owner_i)
        object.__setattr__(self, "owner_j", owner_j)
        object.__setattr__(self, "owner_k", owner_k)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "normal_contra", normal_contra)
        object.__setattr__(self, "area_covector", area_covector)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "J", J)
        object.__setattr__(self, "g_contra", g_contra)
        object.__setattr__(self, "g_cov", g_cov)
        object.__setattr__(self, "B_contra", B_contra)
        object.__setattr__(self, "Bmag", Bmag)
        object.__setattr__(self, "sign", sign)

    @property
    def n_wall_faces(self) -> int:
        return int(self.owner_i.size)

    @classmethod
    def empty(cls) -> "CutWallGeometry3D":
        return cls(
            owner_i=jnp.zeros((0,), dtype=jnp.int32),
            owner_j=jnp.zeros((0,), dtype=jnp.int32),
            owner_k=jnp.zeros((0,), dtype=jnp.int32),
            center=jnp.zeros((0, 3), dtype=jnp.float64),
            normal_contra=jnp.zeros((0, 3), dtype=jnp.float64),
            area_covector=jnp.zeros((0, 3), dtype=jnp.float64),
            distance=jnp.zeros((0,), dtype=jnp.float64),
            J=jnp.zeros((0,), dtype=jnp.float64),
            g_contra=jnp.zeros((0, 3, 3), dtype=jnp.float64),
            g_cov=jnp.zeros((0, 3, 3), dtype=jnp.float64),
            B_contra=jnp.zeros((0, 3), dtype=jnp.float64),
            Bmag=jnp.zeros((0,), dtype=jnp.float64),
            sign=jnp.zeros((0,), dtype=jnp.float64),
        )

    def tree_flatten(self):
        return (
            (
                self.owner_i,
                self.owner_j,
                self.owner_k,
                self.center,
                self.normal_contra,
                self.area_covector,
                self.distance,
                self.J,
                self.g_contra,
                self.g_cov,
                self.B_contra,
                self.Bmag,
                self.sign,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@_pytree_base
@dataclass(frozen=True)
class CutWallBC3D:
    """Future embedded-wall boundary data."""

    kind: jnp.ndarray  # (n_cut_faces,)
    value: jnp.ndarray  # (n_cut_faces,)

    def __post_init__(self) -> None:
        kind = jnp.asarray(self.kind, dtype=jnp.int32)
        value = jnp.asarray(self.value, dtype=jnp.float64)
        if kind.ndim != 1 or value.ndim != 1:
            raise ValueError("CutWallBC3D fields must be one-dimensional")
        if kind.shape != value.shape:
            raise ValueError(f"CutWallBC3D.kind and value must have the same shape, got {kind.shape} and {value.shape}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", value)

    @property
    def n_wall_faces(self) -> int:
        return int(self.kind.size)

    @classmethod
    def empty(cls) -> "CutWallBC3D":
        return cls(
            kind=jnp.zeros((0,), dtype=jnp.int32),
            value=jnp.zeros((0,), dtype=jnp.float64),
        )

    def tree_flatten(self):
        return ((self.kind, self.value), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


@_pytree_base
@dataclass(frozen=True)
class LocalCutWallGeometry3D(_DataclassPyTreeMixin):
    """Padded cut-wall geometry owned by one local shard.

    ``normal_contra`` is the outward unit normal in contravariant logical
    components, normalized with ``g_cov`` so that ``n^i g_ij n^j = 1``.
    ``distance`` is the physical distance from the owner cell center to the
    wall along that unit normal; coordinate-stencil distances belong in
    ``stencil_distance`` instead.  Conservative wall fluxes use
    ``J * area_covector`` as the signed logical wall-area covector, with
    ``sign`` selecting the owner-cell outward orientation.  ``stencil_axis``
    controls coordinate-stencil dependency construction; ``wall_axis`` records
    the physical coordinate-plane orientation for rows whose owner was moved by
    agglomeration and therefore have ``stencil_axis == -1``.
    """

    owner_i: jnp.ndarray
    owner_j: jnp.ndarray
    owner_k: jnp.ndarray
    center: jnp.ndarray
    normal_contra: jnp.ndarray
    area_covector: jnp.ndarray
    distance: jnp.ndarray
    J: jnp.ndarray
    g_contra: jnp.ndarray
    g_cov: jnp.ndarray
    B_contra: jnp.ndarray
    Bmag: jnp.ndarray
    sign: jnp.ndarray
    active: jnp.ndarray
    max_wall_faces: int
    stencil_axis: jnp.ndarray | None = None
    stencil_side: jnp.ndarray | None = None
    stencil_distance: jnp.ndarray | None = None
    wall_axis: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        max_wall_faces = int(self.max_wall_faces)
        if max_wall_faces < 0:
            raise ValueError(f"max_wall_faces must be non-negative, got {max_wall_faces}")

        owner_i = _as_local_wall_int_array(self.owner_i, max_wall_faces, "LocalCutWallGeometry3D.owner_i")
        owner_j = _as_local_wall_int_array(self.owner_j, max_wall_faces, "LocalCutWallGeometry3D.owner_j")
        owner_k = _as_local_wall_int_array(self.owner_k, max_wall_faces, "LocalCutWallGeometry3D.owner_k")
        center = _as_local_wall_array(self.center, max_wall_faces, (3,), "LocalCutWallGeometry3D.center")
        normal_contra = _as_local_wall_array(self.normal_contra, max_wall_faces, (3,), "LocalCutWallGeometry3D.normal_contra")
        area_covector = _as_local_wall_array(self.area_covector, max_wall_faces, (3,), "LocalCutWallGeometry3D.area_covector")
        distance = _as_local_wall_array(self.distance, max_wall_faces, (), "LocalCutWallGeometry3D.distance")
        J = _as_local_wall_array(self.J, max_wall_faces, (), "LocalCutWallGeometry3D.J")
        g_contra = _as_local_wall_array(self.g_contra, max_wall_faces, (3, 3), "LocalCutWallGeometry3D.g_contra")
        g_cov = _as_local_wall_array(self.g_cov, max_wall_faces, (3, 3), "LocalCutWallGeometry3D.g_cov")
        B_contra = _as_local_wall_array(self.B_contra, max_wall_faces, (3,), "LocalCutWallGeometry3D.B_contra")
        Bmag = _as_local_wall_array(self.Bmag, max_wall_faces, (), "LocalCutWallGeometry3D.Bmag")
        sign = _as_local_wall_array(self.sign, max_wall_faces, (), "LocalCutWallGeometry3D.sign")
        active = _as_local_wall_bool_array(self.active, max_wall_faces, "LocalCutWallGeometry3D.active")
        if self.stencil_axis is None:
            stencil_axis = -jnp.ones((max_wall_faces,), dtype=jnp.int32)
        else:
            stencil_axis = _as_local_wall_int_array(
                self.stencil_axis,
                max_wall_faces,
                "LocalCutWallGeometry3D.stencil_axis",
            )
        if self.stencil_side is None:
            stencil_side = jnp.zeros((max_wall_faces,), dtype=jnp.int32)
        else:
            stencil_side = _as_local_wall_int_array(
                self.stencil_side,
                max_wall_faces,
                "LocalCutWallGeometry3D.stencil_side",
            )
        if self.stencil_distance is None:
            stencil_distance = jnp.zeros((max_wall_faces,), dtype=jnp.float64)
        else:
            stencil_distance = _as_local_wall_array(
                self.stencil_distance,
                max_wall_faces,
                (),
                "LocalCutWallGeometry3D.stencil_distance",
            )
        if self.wall_axis is None:
            wall_axis = stencil_axis
        else:
            wall_axis = _as_local_wall_int_array(
                self.wall_axis,
                max_wall_faces,
                "LocalCutWallGeometry3D.wall_axis",
            )

        axis_enabled = stencil_axis >= 0
        wall_axis_enabled = wall_axis >= 0
        valid_axis = (stencil_axis == -1) | ((stencil_axis >= 0) & (stencil_axis <= 2))
        valid_wall_axis = (wall_axis == -1) | ((wall_axis >= 0) & (wall_axis <= 2))
        valid_side = (~axis_enabled) | ((stencil_side >= 0) & (stencil_side <= 1))
        valid_distance = (~(axis_enabled | wall_axis_enabled)) | (stencil_distance > 0.0)
        if not bool(jnp.all(valid_axis)):
            raise ValueError("LocalCutWallGeometry3D.stencil_axis must contain -1, 0, 1, or 2")
        if not bool(jnp.all(valid_wall_axis)):
            raise ValueError("LocalCutWallGeometry3D.wall_axis must contain -1, 0, 1, or 2")
        if not bool(jnp.all(valid_side)):
            raise ValueError(
                "LocalCutWallGeometry3D.stencil_side must contain 0 or 1 for enabled stencil rows"
            )
        if not bool(jnp.all(valid_distance)):
            raise ValueError(
                "LocalCutWallGeometry3D.stencil_distance must be positive for enabled stencil or wall-axis rows"
            )
        stencil_side = jnp.where(axis_enabled, stencil_side, 0)
        stencil_distance = jnp.where(axis_enabled | wall_axis_enabled, stencil_distance, 0.0)

        object.__setattr__(self, "owner_i", owner_i)
        object.__setattr__(self, "owner_j", owner_j)
        object.__setattr__(self, "owner_k", owner_k)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "normal_contra", normal_contra)
        object.__setattr__(self, "area_covector", area_covector)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "J", J)
        object.__setattr__(self, "g_contra", g_contra)
        object.__setattr__(self, "g_cov", g_cov)
        object.__setattr__(self, "B_contra", B_contra)
        object.__setattr__(self, "Bmag", Bmag)
        object.__setattr__(self, "sign", sign)
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "max_wall_faces", max_wall_faces)
        object.__setattr__(self, "stencil_axis", stencil_axis)
        object.__setattr__(self, "stencil_side", stencil_side)
        object.__setattr__(self, "stencil_distance", stencil_distance)
        object.__setattr__(self, "wall_axis", wall_axis)

    @property
    def n_wall_faces(self) -> int:
        return int(self.max_wall_faces)

    @classmethod
    def empty(cls, max_wall_faces: int) -> "LocalCutWallGeometry3D":
        max_wall_faces = int(max_wall_faces)
        if max_wall_faces < 0:
            raise ValueError(f"max_wall_faces must be non-negative, got {max_wall_faces}")
        zeros3 = (max_wall_faces, 3)
        zeros33 = (max_wall_faces, 3, 3)
        obj = object.__new__(cls)
        object.__setattr__(obj, "owner_i", jnp.zeros((max_wall_faces,), dtype=jnp.int32))
        object.__setattr__(obj, "owner_j", jnp.zeros((max_wall_faces,), dtype=jnp.int32))
        object.__setattr__(obj, "owner_k", jnp.zeros((max_wall_faces,), dtype=jnp.int32))
        object.__setattr__(obj, "center", jnp.zeros(zeros3, dtype=jnp.float64))
        object.__setattr__(obj, "normal_contra", jnp.zeros(zeros3, dtype=jnp.float64))
        object.__setattr__(obj, "area_covector", jnp.zeros(zeros3, dtype=jnp.float64))
        object.__setattr__(obj, "distance", jnp.zeros((max_wall_faces,), dtype=jnp.float64))
        object.__setattr__(obj, "J", jnp.zeros((max_wall_faces,), dtype=jnp.float64))
        object.__setattr__(obj, "g_contra", jnp.zeros(zeros33, dtype=jnp.float64))
        object.__setattr__(obj, "g_cov", jnp.zeros(zeros33, dtype=jnp.float64))
        object.__setattr__(obj, "B_contra", jnp.zeros(zeros3, dtype=jnp.float64))
        object.__setattr__(obj, "Bmag", jnp.zeros((max_wall_faces,), dtype=jnp.float64))
        object.__setattr__(obj, "sign", jnp.zeros((max_wall_faces,), dtype=jnp.float64))
        object.__setattr__(obj, "active", jnp.zeros((max_wall_faces,), dtype=bool))
        object.__setattr__(obj, "max_wall_faces", max_wall_faces)
        object.__setattr__(
            obj,
            "stencil_axis",
            -jnp.ones((max_wall_faces,), dtype=jnp.int32),
        )
        object.__setattr__(obj, "stencil_side", jnp.zeros((max_wall_faces,), dtype=jnp.int32))
        object.__setattr__(
            obj,
            "stencil_distance",
            jnp.zeros((max_wall_faces,), dtype=jnp.float64),
        )
        object.__setattr__(
            obj,
            "wall_axis",
            -jnp.ones((max_wall_faces,), dtype=jnp.int32),
        )
        return obj

    def tree_flatten(self):
        return (
            (
                self.owner_i,
                self.owner_j,
                self.owner_k,
                self.center,
                self.normal_contra,
                self.area_covector,
                self.distance,
                self.J,
                self.g_contra,
                self.g_cov,
                self.B_contra,
                self.Bmag,
                self.sign,
                self.active,
                self.stencil_axis,
                self.stencil_side,
                self.stencil_distance,
                self.wall_axis,
            ),
            self.max_wall_faces,
        )

    @classmethod
    def tree_unflatten(cls, max_wall_faces, children):
        (
            owner_i,
            owner_j,
            owner_k,
            center,
            normal_contra,
            area_covector,
            distance,
            J,
            g_contra,
            g_cov,
            B_contra,
            Bmag,
            sign,
            active,
            stencil_axis,
            stencil_side,
            stencil_distance,
            wall_axis,
        ) = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "owner_i", owner_i)
        object.__setattr__(obj, "owner_j", owner_j)
        object.__setattr__(obj, "owner_k", owner_k)
        object.__setattr__(obj, "center", center)
        object.__setattr__(obj, "normal_contra", normal_contra)
        object.__setattr__(obj, "area_covector", area_covector)
        object.__setattr__(obj, "distance", distance)
        object.__setattr__(obj, "J", J)
        object.__setattr__(obj, "g_contra", g_contra)
        object.__setattr__(obj, "g_cov", g_cov)
        object.__setattr__(obj, "B_contra", B_contra)
        object.__setattr__(obj, "Bmag", Bmag)
        object.__setattr__(obj, "sign", sign)
        object.__setattr__(obj, "active", active)
        object.__setattr__(obj, "max_wall_faces", int(max_wall_faces))
        object.__setattr__(obj, "stencil_axis", stencil_axis)
        object.__setattr__(obj, "stencil_side", stencil_side)
        object.__setattr__(obj, "stencil_distance", stencil_distance)
        object.__setattr__(obj, "wall_axis", wall_axis)
        return obj


@_pytree_base
@dataclass(frozen=True)
class LocalCutWallBC3D(_DataclassPyTreeMixin):
    """
    Local padded cut-wall boundary-condition payload for one shard.

    Each active entry corresponds to one active entry in LocalCutWallGeometry3D.

    The leading dimension is Wmax, where:

        Wmax = maximum number of cut-wall pieces per shard/local domain

    not the total global number of cut-wall pieces.

    The BC arrays are padded so all shards have the same static shape.
    Inactive padded entries are masked by active=False.
    """

    kind: jnp.ndarray
    value: jnp.ndarray
    active: jnp.ndarray
    max_wall_faces: int

    def __post_init__(self) -> None:
        max_wall_faces = int(self.max_wall_faces)
        if max_wall_faces < 0:
            raise ValueError(f"max_wall_faces must be non-negative, got {max_wall_faces}")

        kind = jnp.asarray(self.kind, dtype=jnp.int32)
        value = jnp.asarray(self.value, dtype=jnp.float64)
        active = jnp.asarray(self.active, dtype=bool)
        expected = (max_wall_faces,)
        if kind.shape != expected:
            raise ValueError(f"LocalCutWallBC3D.kind must have shape {expected}, got {kind.shape}")
        if value.shape != expected:
            raise ValueError(f"LocalCutWallBC3D.value must have shape {expected}, got {value.shape}")
        if active.shape != expected:
            raise ValueError(f"LocalCutWallBC3D.active must have shape {expected}, got {active.shape}")

        kind = jnp.where(active, kind, BC_NONE)
        value = jnp.where(active, value, 0.0)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "max_wall_faces", max_wall_faces)

    @property
    def shape(self) -> tuple[int]:
        return (int(self.max_wall_faces),)

    @property
    def n_wall_faces(self) -> int:
        return int(self.max_wall_faces)

    @property
    def n_active(self) -> jnp.ndarray:
        return jnp.sum(self.active)

    @classmethod
    def empty(cls, max_wall_faces: int) -> "LocalCutWallBC3D":
        max_wall_faces = int(max_wall_faces)
        return cls(
            kind=jnp.zeros((max_wall_faces,), dtype=jnp.int32),
            value=jnp.zeros((max_wall_faces,), dtype=jnp.float64),
            active=jnp.zeros((max_wall_faces,), dtype=bool),
            max_wall_faces=max_wall_faces,
        )

    def tree_flatten(self):
        return ((self.kind, self.value, self.active), self.max_wall_faces)

    @classmethod
    def tree_unflatten(cls, max_wall_faces, children):
        kind, value, active = children
        return cls(kind=kind, value=value, active=active, max_wall_faces=max_wall_faces)


@_pytree_base
@dataclass(frozen=True)
class LocalControlVolumeFaceRows3D(_DataclassPyTreeMixin):
    """Compact unique irregular interfaces for owned control volumes.

    Each active row is one physical interface.  Interior rows have both a
    minus and plus owner and are scattered with equal and opposite signs.
    Boundary rows have only a minus owner.  A face can be decomposed into
    ``max_patches`` non-overlapping rectangles, each evaluated at four tensor
    Gauss points.

    ``area_covector_weight`` contains the oriented logical area covector
    including the 2D quadrature weight, but not ``J``.  Geometry-dependent
    metric, magnetic, and projector data are collocated at the same points.
    """

    layout: HaloLayout3D
    kind: jnp.ndarray
    minus_owner_i: jnp.ndarray
    minus_owner_j: jnp.ndarray
    minus_owner_k: jnp.ndarray
    plus_owner_i: jnp.ndarray
    plus_owner_j: jnp.ndarray
    plus_owner_k: jnp.ndarray
    has_plus_owner: jnp.ndarray
    quadrature_points: jnp.ndarray
    area_covector_weight: jnp.ndarray
    J: jnp.ndarray
    g_contra: jnp.ndarray
    g_cov: jnp.ndarray
    B_contra: jnp.ndarray
    Bmag: jnp.ndarray
    projector: jnp.ndarray
    patch_active: jnp.ndarray
    active: jnp.ndarray
    max_rows: int
    max_patches: int = 4
    has_remote_owner: jnp.ndarray | None = None
    remote_halo_i: jnp.ndarray | None = None
    remote_halo_j: jnp.ndarray | None = None
    remote_halo_k: jnp.ndarray | None = None
    remote_centroid: jnp.ndarray | None = None
    remote_second_moment: jnp.ndarray | None = None
    remote_third_moment: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("LocalControlVolumeFaceRows3D.layout must be a HaloLayout3D")
        max_rows = int(self.max_rows)
        max_patches = int(self.max_patches)
        if max_rows < 0:
            raise ValueError("LocalControlVolumeFaceRows3D.max_rows must be non-negative")
        if max_patches < 1:
            raise ValueError("LocalControlVolumeFaceRows3D.max_patches must be positive")
        row_shape = (max_rows,)
        patch_shape = (max_rows, max_patches)
        quadrature_shape = (max_rows, max_patches, 4)
        point_shape = quadrature_shape + (3,)
        tensor_shape = quadrature_shape + (3, 3)

        def _row_int(value, name):
            array = jnp.asarray(value, dtype=jnp.int32)
            if array.shape != row_shape:
                raise ValueError(f"{name} must have shape {row_shape}, got {array.shape}")
            return array

        kind = _row_int(self.kind, "LocalControlVolumeFaceRows3D.kind")
        minus_owner_i = _row_int(
            self.minus_owner_i,
            "LocalControlVolumeFaceRows3D.minus_owner_i",
        )
        minus_owner_j = _row_int(
            self.minus_owner_j,
            "LocalControlVolumeFaceRows3D.minus_owner_j",
        )
        minus_owner_k = _row_int(
            self.minus_owner_k,
            "LocalControlVolumeFaceRows3D.minus_owner_k",
        )
        plus_owner_i = _row_int(
            self.plus_owner_i,
            "LocalControlVolumeFaceRows3D.plus_owner_i",
        )
        plus_owner_j = _row_int(
            self.plus_owner_j,
            "LocalControlVolumeFaceRows3D.plus_owner_j",
        )
        plus_owner_k = _row_int(
            self.plus_owner_k,
            "LocalControlVolumeFaceRows3D.plus_owner_k",
        )
        has_plus_owner = jnp.asarray(self.has_plus_owner, dtype=bool)
        has_remote_owner = jnp.asarray(
            (
                jnp.zeros(row_shape, dtype=bool)
                if self.has_remote_owner is None
                else self.has_remote_owner
            ),
            dtype=bool,
        )
        remote_halo_i = _row_int(
            (
                jnp.zeros(row_shape, dtype=jnp.int32)
                if self.remote_halo_i is None
                else self.remote_halo_i
            ),
            "LocalControlVolumeFaceRows3D.remote_halo_i",
        )
        remote_halo_j = _row_int(
            (
                jnp.zeros(row_shape, dtype=jnp.int32)
                if self.remote_halo_j is None
                else self.remote_halo_j
            ),
            "LocalControlVolumeFaceRows3D.remote_halo_j",
        )
        remote_halo_k = _row_int(
            (
                jnp.zeros(row_shape, dtype=jnp.int32)
                if self.remote_halo_k is None
                else self.remote_halo_k
            ),
            "LocalControlVolumeFaceRows3D.remote_halo_k",
        )
        remote_centroid = jnp.asarray(
            (
                jnp.zeros(row_shape + (3,), dtype=jnp.float64)
                if self.remote_centroid is None
                else self.remote_centroid
            ),
            dtype=jnp.float64,
        )
        remote_second_moment = jnp.asarray(
            (
                jnp.zeros(row_shape + (3, 3), dtype=jnp.float64)
                if self.remote_second_moment is None
                else self.remote_second_moment
            ),
            dtype=jnp.float64,
        )
        remote_third_moment = jnp.asarray(
            (
                jnp.zeros(row_shape + (3, 3, 3), dtype=jnp.float64)
                if self.remote_third_moment is None
                else self.remote_third_moment
            ),
            dtype=jnp.float64,
        )
        if remote_centroid.shape != row_shape + (3,):
            raise ValueError(
                "LocalControlVolumeFaceRows3D.remote_centroid must have shape "
                f"{row_shape + (3,)}, got {remote_centroid.shape}"
            )
        if remote_second_moment.shape != row_shape + (3, 3):
            raise ValueError(
                "LocalControlVolumeFaceRows3D.remote_second_moment must have "
                f"shape {row_shape + (3, 3)}, got {remote_second_moment.shape}"
            )
        if remote_third_moment.shape != row_shape + (3, 3, 3):
            raise ValueError(
                "LocalControlVolumeFaceRows3D.remote_third_moment must have "
                f"shape {row_shape + (3, 3, 3)}, got {remote_third_moment.shape}"
            )
        active = jnp.asarray(self.active, dtype=bool)
        patch_active = jnp.asarray(self.patch_active, dtype=bool)
        if has_plus_owner.shape != row_shape:
            raise ValueError(
                "LocalControlVolumeFaceRows3D.has_plus_owner must have shape "
                f"{row_shape}, got {has_plus_owner.shape}"
            )
        if has_remote_owner.shape != row_shape:
            raise ValueError(
                "LocalControlVolumeFaceRows3D.has_remote_owner must have shape "
                f"{row_shape}, got {has_remote_owner.shape}"
            )
        if active.shape != row_shape:
            raise ValueError(
                f"LocalControlVolumeFaceRows3D.active must have shape {row_shape}, got {active.shape}"
            )
        if patch_active.shape != patch_shape:
            raise ValueError(
                "LocalControlVolumeFaceRows3D.patch_active must have shape "
                f"{patch_shape}, got {patch_active.shape}"
            )

        def _float_shape(value, expected, name):
            array = jnp.asarray(value, dtype=jnp.float64)
            if array.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
            return array

        quadrature_points = _float_shape(
            self.quadrature_points,
            point_shape,
            "LocalControlVolumeFaceRows3D.quadrature_points",
        )
        area_covector_weight = _float_shape(
            self.area_covector_weight,
            point_shape,
            "LocalControlVolumeFaceRows3D.area_covector_weight",
        )
        J = _float_shape(self.J, quadrature_shape, "LocalControlVolumeFaceRows3D.J")
        g_contra = _float_shape(
            self.g_contra,
            tensor_shape,
            "LocalControlVolumeFaceRows3D.g_contra",
        )
        g_cov = _float_shape(
            self.g_cov,
            tensor_shape,
            "LocalControlVolumeFaceRows3D.g_cov",
        )
        B_contra = _float_shape(
            self.B_contra,
            point_shape,
            "LocalControlVolumeFaceRows3D.B_contra",
        )
        Bmag = _float_shape(
            self.Bmag,
            quadrature_shape,
            "LocalControlVolumeFaceRows3D.Bmag",
        )
        projector = _float_shape(
            self.projector,
            tensor_shape,
            "LocalControlVolumeFaceRows3D.projector",
        )

        valid_kind = (
            (~active)
            | (kind == CV_FACE_INTERIOR)
            | (kind == CV_FACE_CUT_WALL)
            | (kind == CV_FACE_PARTIAL)
            | (kind == CV_FACE_PHYSICAL_BOUNDARY)
        )
        nx, ny, nz = self.layout.owned_shape
        minus_in_bounds = (
            (minus_owner_i >= 0)
            & (minus_owner_i < nx)
            & (minus_owner_j >= 0)
            & (minus_owner_j < ny)
            & (minus_owner_k >= 0)
            & (minus_owner_k < nz)
        )
        plus_in_bounds = (
            (plus_owner_i >= 0)
            & (plus_owner_i < nx)
            & (plus_owner_j >= 0)
            & (plus_owner_j < ny)
            & (plus_owner_k >= 0)
            & (plus_owner_k < nz)
        )
        hx, hy, hz = self.layout.cell_halo_shape
        remote_in_bounds = (
            (remote_halo_i >= 0)
            & (remote_halo_i < hx)
            & (remote_halo_j >= 0)
            & (remote_halo_j < hy)
            & (remote_halo_k >= 0)
            & (remote_halo_k < hz)
        )
        valid_owners = (~active) | (
            minus_in_bounds
            & ((~has_plus_owner) | plus_in_bounds)
            & ((~has_remote_owner) | remote_in_bounds)
            & ~(has_plus_owner & has_remote_owner)
        )
        try:
            all_valid_kind = bool(jnp.all(valid_kind))
            all_valid_owners = bool(jnp.all(valid_owners))
            finite_active_geometry = bool(
                jnp.all(
                    (~(active[:, None, None] & patch_active[:, :, None]))
                    | (
                        jnp.isfinite(J)
                        & jnp.isfinite(Bmag)
                        & jnp.all(jnp.isfinite(quadrature_points), axis=-1)
                        & jnp.all(jnp.isfinite(area_covector_weight), axis=-1)
                    )
                )
            )
            finite_remote_geometry = bool(
                jnp.all(
                    (~(active & has_remote_owner))
                    | (
                        jnp.all(jnp.isfinite(remote_centroid), axis=-1)
                        & jnp.all(
                            jnp.isfinite(remote_second_moment),
                            axis=(-2, -1),
                        )
                        & jnp.all(jnp.isfinite(remote_third_moment), axis=(-3, -2, -1))
                    )
                )
            )
        except jax.errors.TracerBoolConversionError:
            all_valid_kind = True
            all_valid_owners = True
            finite_active_geometry = True
            finite_remote_geometry = True
        if not all_valid_kind:
            raise ValueError("active control-volume face rows have an invalid kind")
        if not all_valid_owners:
            raise ValueError("active control-volume face owners must be local")
        if not finite_active_geometry:
            raise ValueError("active control-volume face quadrature must be finite")
        if not finite_remote_geometry:
            raise ValueError("active remote control-volume moments must be finite")

        quadrature_active = jnp.broadcast_to(
            active[:, None, None] & patch_active[:, :, None],
            quadrature_shape,
        )
        object.__setattr__(self, "kind", jnp.where(active, kind, CV_FACE_NONE))
        object.__setattr__(self, "minus_owner_i", jnp.where(active, minus_owner_i, 0))
        object.__setattr__(self, "minus_owner_j", jnp.where(active, minus_owner_j, 0))
        object.__setattr__(self, "minus_owner_k", jnp.where(active, minus_owner_k, 0))
        object.__setattr__(self, "plus_owner_i", jnp.where(active, plus_owner_i, 0))
        object.__setattr__(self, "plus_owner_j", jnp.where(active, plus_owner_j, 0))
        object.__setattr__(self, "plus_owner_k", jnp.where(active, plus_owner_k, 0))
        object.__setattr__(self, "has_plus_owner", active & has_plus_owner)
        object.__setattr__(
            self,
            "has_remote_owner",
            active & has_remote_owner,
        )
        object.__setattr__(
            self,
            "remote_halo_i",
            jnp.where(active & has_remote_owner, remote_halo_i, 0),
        )
        object.__setattr__(
            self,
            "remote_halo_j",
            jnp.where(active & has_remote_owner, remote_halo_j, 0),
        )
        object.__setattr__(
            self,
            "remote_halo_k",
            jnp.where(active & has_remote_owner, remote_halo_k, 0),
        )
        object.__setattr__(
            self,
            "remote_centroid",
            jnp.where(
                (active & has_remote_owner)[:, None],
                remote_centroid,
                0.0,
            ),
        )
        object.__setattr__(
            self,
            "remote_second_moment",
            jnp.where(
                (active & has_remote_owner)[:, None, None],
                0.5
                * (
                    remote_second_moment
                    + jnp.swapaxes(remote_second_moment, -1, -2)
                ),
                0.0,
            ),
        )
        permutations = (
            (0, 1, 2), (0, 2, 1), (1, 0, 2),
            (1, 2, 0), (2, 0, 1), (2, 1, 0),
        )
        object.__setattr__(
            self,
            "remote_third_moment",
            jnp.where(
                (active & has_remote_owner)[:, None, None, None],
                sum(
                    jnp.transpose(
                        remote_third_moment,
                        (0,) + tuple(axis + 1 for axis in permutation),
                    )
                    for permutation in permutations
                ) / 6.0,
                0.0,
            ),
        )
        object.__setattr__(
            self,
            "quadrature_points",
            jnp.where(quadrature_active[..., None], quadrature_points, 0.0),
        )
        object.__setattr__(
            self,
            "area_covector_weight",
            jnp.where(quadrature_active[..., None], area_covector_weight, 0.0),
        )
        object.__setattr__(self, "J", jnp.where(quadrature_active, J, 0.0))
        object.__setattr__(
            self,
            "g_contra",
            jnp.where(quadrature_active[..., None, None], g_contra, 0.0),
        )
        object.__setattr__(
            self,
            "g_cov",
            jnp.where(quadrature_active[..., None, None], g_cov, 0.0),
        )
        object.__setattr__(
            self,
            "B_contra",
            jnp.where(quadrature_active[..., None], B_contra, 0.0),
        )
        object.__setattr__(self, "Bmag", jnp.where(quadrature_active, Bmag, 1.0))
        object.__setattr__(
            self,
            "projector",
            jnp.where(quadrature_active[..., None, None], projector, 0.0),
        )
        object.__setattr__(self, "patch_active", active[:, None] & patch_active)
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "max_rows", max_rows)
        object.__setattr__(self, "max_patches", max_patches)

    @property
    def quadrature_active(self) -> jnp.ndarray:
        return jnp.broadcast_to(
            self.active[:, None, None] & self.patch_active[:, :, None],
            (int(self.max_rows), int(self.max_patches), 4),
        )

    @classmethod
    def empty(
        cls,
        layout: HaloLayout3D,
        *,
        max_rows: int = 0,
        max_patches: int = 4,
    ) -> "LocalControlVolumeFaceRows3D":
        max_rows = int(max_rows)
        max_patches = int(max_patches)
        row = (max_rows,)
        patch = (max_rows, max_patches)
        quadrature = (max_rows, max_patches, 4)
        points = quadrature + (3,)
        tensors = quadrature + (3, 3)
        return cls(
            layout=layout,
            kind=jnp.zeros(row, dtype=jnp.int32),
            minus_owner_i=jnp.zeros(row, dtype=jnp.int32),
            minus_owner_j=jnp.zeros(row, dtype=jnp.int32),
            minus_owner_k=jnp.zeros(row, dtype=jnp.int32),
            plus_owner_i=jnp.zeros(row, dtype=jnp.int32),
            plus_owner_j=jnp.zeros(row, dtype=jnp.int32),
            plus_owner_k=jnp.zeros(row, dtype=jnp.int32),
            has_plus_owner=jnp.zeros(row, dtype=bool),
            quadrature_points=jnp.zeros(points, dtype=jnp.float64),
            area_covector_weight=jnp.zeros(points, dtype=jnp.float64),
            J=jnp.zeros(quadrature, dtype=jnp.float64),
            g_contra=jnp.zeros(tensors, dtype=jnp.float64),
            g_cov=jnp.zeros(tensors, dtype=jnp.float64),
            B_contra=jnp.zeros(points, dtype=jnp.float64),
            Bmag=jnp.ones(quadrature, dtype=jnp.float64),
            projector=jnp.zeros(tensors, dtype=jnp.float64),
            patch_active=jnp.zeros(patch, dtype=bool),
            active=jnp.zeros(row, dtype=bool),
            max_rows=max_rows,
            max_patches=max_patches,
            has_remote_owner=jnp.zeros(row, dtype=bool),
            remote_halo_i=jnp.zeros(row, dtype=jnp.int32),
            remote_halo_j=jnp.zeros(row, dtype=jnp.int32),
            remote_halo_k=jnp.zeros(row, dtype=jnp.int32),
            remote_centroid=jnp.zeros(row + (3,), dtype=jnp.float64),
            remote_second_moment=jnp.zeros(row + (3, 3), dtype=jnp.float64),
            remote_third_moment=jnp.zeros(row + (3, 3, 3), dtype=jnp.float64),
        )

    def tree_flatten(self):
        return (
            (
                self.kind,
                self.minus_owner_i,
                self.minus_owner_j,
                self.minus_owner_k,
                self.plus_owner_i,
                self.plus_owner_j,
                self.plus_owner_k,
                self.has_plus_owner,
                self.quadrature_points,
                self.area_covector_weight,
                self.J,
                self.g_contra,
                self.g_cov,
                self.B_contra,
                self.Bmag,
                self.projector,
                self.patch_active,
                self.active,
                self.has_remote_owner,
                self.remote_halo_i,
                self.remote_halo_j,
                self.remote_halo_k,
                self.remote_centroid,
                self.remote_second_moment,
                self.remote_third_moment,
            ),
            (self.layout, self.max_rows, self.max_patches),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        layout, max_rows, max_patches = aux_data
        names = (
            "kind",
            "minus_owner_i",
            "minus_owner_j",
            "minus_owner_k",
            "plus_owner_i",
            "plus_owner_j",
            "plus_owner_k",
            "has_plus_owner",
            "quadrature_points",
            "area_covector_weight",
            "J",
            "g_contra",
            "g_cov",
            "B_contra",
            "Bmag",
            "projector",
            "patch_active",
            "active",
            "has_remote_owner",
            "remote_halo_i",
            "remote_halo_j",
            "remote_halo_k",
            "remote_centroid",
            "remote_second_moment",
            "remote_third_moment",
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "layout", layout)
        object.__setattr__(instance, "max_rows", max_rows)
        object.__setattr__(instance, "max_patches", max_patches)
        for name, value in zip(names, children):
            object.__setattr__(instance, name, value)
        return instance


@_pytree_base
@dataclass(frozen=True)
class LocalControlVolumeBoundaryBC3D(_DataclassPyTreeMixin):
    """Field-specific BC values collocated with irregular face rows."""

    kind: jnp.ndarray
    centroid_value: jnp.ndarray
    quadrature_value: jnp.ndarray
    active: jnp.ndarray
    max_rows: int
    max_patches: int = 4

    def __post_init__(self) -> None:
        max_rows = int(self.max_rows)
        max_patches = int(self.max_patches)
        row_shape = (max_rows,)
        quadrature_shape = (max_rows, max_patches, 4)
        kind = jnp.asarray(self.kind, dtype=jnp.int32)
        centroid_value = jnp.asarray(self.centroid_value, dtype=jnp.float64)
        quadrature_value = jnp.asarray(self.quadrature_value, dtype=jnp.float64)
        active = jnp.asarray(self.active, dtype=bool)
        if kind.shape != row_shape:
            raise ValueError(f"boundary kind must have shape {row_shape}, got {kind.shape}")
        if centroid_value.shape != row_shape:
            raise ValueError(
                f"boundary centroid_value must have shape {row_shape}, got {centroid_value.shape}"
            )
        if quadrature_value.shape != quadrature_shape:
            raise ValueError(
                "boundary quadrature_value must have shape "
                f"{quadrature_shape}, got {quadrature_value.shape}"
            )
        if active.shape != row_shape:
            raise ValueError(f"boundary active must have shape {row_shape}, got {active.shape}")
        supported = (
            (kind == BC_NONE)
            | (kind == BC_DIRICHLET)
            | (kind == BC_NEUMANN)
            | (kind == BC_NORMALFLUX)
            | (kind == BC_NOFLUX)
        )
        try:
            all_supported = bool(jnp.all((~active) | supported))
        except jax.errors.TracerBoolConversionError:
            all_supported = True
        if not all_supported:
            raise ValueError("active control-volume boundary rows use an unsupported BC kind")
        object.__setattr__(self, "kind", jnp.where(active, kind, BC_NONE))
        object.__setattr__(
            self,
            "centroid_value",
            jnp.where(active, centroid_value, 0.0),
        )
        object.__setattr__(
            self,
            "quadrature_value",
            jnp.where(active[:, None, None], quadrature_value, 0.0),
        )
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "max_rows", max_rows)
        object.__setattr__(self, "max_patches", max_patches)

    @classmethod
    def empty(
        cls,
        *,
        max_rows: int = 0,
        max_patches: int = 4,
    ) -> "LocalControlVolumeBoundaryBC3D":
        return cls(
            kind=jnp.zeros((max_rows,), dtype=jnp.int32),
            centroid_value=jnp.zeros((max_rows,), dtype=jnp.float64),
            quadrature_value=jnp.zeros(
                (max_rows, max_patches, 4),
                dtype=jnp.float64,
            ),
            active=jnp.zeros((max_rows,), dtype=bool),
            max_rows=max_rows,
            max_patches=max_patches,
        )

    def tree_flatten(self):
        return (
            (self.kind, self.centroid_value, self.quadrature_value, self.active),
            (self.max_rows, self.max_patches),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        max_rows, max_patches = aux_data
        kind, centroid_value, quadrature_value, active = children
        return cls(
            kind=kind,
            centroid_value=centroid_value,
            quadrature_value=quadrature_value,
            active=active,
            max_rows=max_rows,
            max_patches=max_patches,
        )


@_pytree_base
@dataclass(frozen=True)
class LocalMomentReconstruction3D(_DataclassPyTreeMixin):
    """Precomputed moment-aware polynomial equations for irregular owners.

    ``rhs_transform`` maps equation right-hand sides directly to the nine
    physical logical-coordinate coefficients ``(g, Hsym[, Tsym])``.  Rank-revealing
    factorization is performed when this metadata is built, never in an
    operator JIT.
    """

    layout: HaloLayout3D
    target_i: jnp.ndarray
    target_j: jnp.ndarray
    target_k: jnp.ndarray
    equation_kind: jnp.ndarray
    sample_i: jnp.ndarray
    sample_j: jnp.ndarray
    sample_k: jnp.ndarray
    boundary_face_row: jnp.ndarray
    equation_active: jnp.ndarray
    rhs_transform: jnp.ndarray
    active: jnp.ndarray
    target_row_for_cell: jnp.ndarray
    polynomial_order: jnp.ndarray
    rank: jnp.ndarray
    condition_number: jnp.ndarray
    max_rows: int
    max_equations: int
    boundary_patch: jnp.ndarray | None = None
    boundary_quadrature: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("LocalMomentReconstruction3D.layout must be a HaloLayout3D")
        max_rows = int(self.max_rows)
        max_equations = int(self.max_equations)
        if max_rows < 0 or max_equations < 1:
            raise ValueError("reconstruction sizes must be non-negative/positive")
        row_shape = (max_rows,)
        equation_shape = (max_rows, max_equations)

        def _int(value, shape, name):
            array = jnp.asarray(value, dtype=jnp.int32)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
            return array

        target_i = _int(self.target_i, row_shape, "quadratic target_i")
        target_j = _int(self.target_j, row_shape, "quadratic target_j")
        target_k = _int(self.target_k, row_shape, "quadratic target_k")
        equation_kind = _int(
            self.equation_kind,
            equation_shape,
            "quadratic equation_kind",
        )
        sample_i = _int(self.sample_i, equation_shape, "quadratic sample_i")
        sample_j = _int(self.sample_j, equation_shape, "quadratic sample_j")
        sample_k = _int(self.sample_k, equation_shape, "quadratic sample_k")
        boundary_face_row = _int(
            self.boundary_face_row,
            equation_shape,
            "quadratic boundary_face_row",
        )
        boundary_patch = _int(
            (
                jnp.zeros(equation_shape, dtype=jnp.int32)
                if self.boundary_patch is None
                else self.boundary_patch
            ),
            equation_shape,
            "quadratic boundary_patch",
        )
        boundary_quadrature = _int(
            (
                jnp.zeros(equation_shape, dtype=jnp.int32)
                if self.boundary_quadrature is None
                else self.boundary_quadrature
            ),
            equation_shape,
            "quadratic boundary_quadrature",
        )
        equation_active = jnp.asarray(self.equation_active, dtype=bool)
        rhs_transform = jnp.asarray(self.rhs_transform, dtype=jnp.float64)
        active = jnp.asarray(self.active, dtype=bool)
        target_row_for_cell = jnp.asarray(self.target_row_for_cell, dtype=jnp.int32)
        polynomial_order = _int(
            self.polynomial_order,
            row_shape,
            "quadratic polynomial_order",
        )
        rank = _int(self.rank, row_shape, "quadratic rank")
        condition_number = jnp.asarray(self.condition_number, dtype=jnp.float64)
        if equation_active.shape != equation_shape:
            raise ValueError(
                f"equation_active must have shape {equation_shape}, got {equation_active.shape}"
            )
        if rhs_transform.ndim != 3 or rhs_transform.shape[0] != max_rows or rhs_transform.shape[2] != max_equations or rhs_transform.shape[1] not in (9, 19):
            raise ValueError(
                "rhs_transform must have shape "
                f"(max_rows, 9|19, max_equations), got {rhs_transform.shape}"
            )
        if active.shape != row_shape:
            raise ValueError(f"quadratic active must have shape {row_shape}, got {active.shape}")
        if target_row_for_cell.shape != self.layout.owned_shape:
            raise ValueError(
                "target_row_for_cell must match layout.owned_shape, got "
                f"{target_row_for_cell.shape}"
            )
        if condition_number.shape != row_shape:
            raise ValueError(
                f"condition_number must have shape {row_shape}, got {condition_number.shape}"
            )
        valid_kind = (
            (~equation_active)
            | (equation_kind == CV_RECONSTRUCTION_EQUATION_CELL)
            | (equation_kind == CV_RECONSTRUCTION_EQUATION_DIRICHLET)
            | (equation_kind == CV_RECONSTRUCTION_EQUATION_REMOTE_CELL)
        )
        try:
            all_valid_kind = bool(jnp.all(valid_kind))
            all_valid_order = bool(
                jnp.all((~active) | ((polynomial_order >= 1) & (polynomial_order <= 3)))
            )
        except jax.errors.TracerBoolConversionError:
            all_valid_kind = True
            all_valid_order = True
        if not all_valid_kind:
            raise ValueError("active reconstruction equation has an invalid kind")
        if not all_valid_order:
            raise ValueError("active reconstruction rows must have order one, two, or three")

        object.__setattr__(self, "target_i", jnp.where(active, target_i, 0))
        object.__setattr__(self, "target_j", jnp.where(active, target_j, 0))
        object.__setattr__(self, "target_k", jnp.where(active, target_k, 0))
        object.__setattr__(
            self,
            "equation_kind",
            jnp.where(equation_active, equation_kind, CV_RECONSTRUCTION_EQUATION_NONE),
        )
        object.__setattr__(self, "sample_i", jnp.where(equation_active, sample_i, 0))
        object.__setattr__(self, "sample_j", jnp.where(equation_active, sample_j, 0))
        object.__setattr__(self, "sample_k", jnp.where(equation_active, sample_k, 0))
        object.__setattr__(
            self,
            "boundary_face_row",
            jnp.where(equation_active, boundary_face_row, 0),
        )
        object.__setattr__(
            self,
            "boundary_patch",
            jnp.where(equation_active, boundary_patch, 0),
        )
        object.__setattr__(
            self,
            "boundary_quadrature",
            jnp.where(equation_active, boundary_quadrature, 0),
        )
        object.__setattr__(self, "equation_active", equation_active)
        object.__setattr__(
            self,
            "rhs_transform",
            jnp.where(equation_active[:, None, :], rhs_transform, 0.0),
        )
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "target_row_for_cell", target_row_for_cell)
        object.__setattr__(
            self,
            "polynomial_order",
            jnp.where(active, polynomial_order, 0),
        )
        object.__setattr__(self, "rank", jnp.where(active, rank, 0))
        object.__setattr__(
            self,
            "condition_number",
            jnp.where(active, condition_number, jnp.inf),
        )
        object.__setattr__(self, "max_rows", max_rows)
        object.__setattr__(self, "max_equations", max_equations)

    @classmethod
    def empty(
        cls,
        layout: HaloLayout3D,
        *,
        max_rows: int = 0,
        max_equations: int = 1,
        coefficient_count: int = 9,
    ) -> "LocalMomentReconstruction3D":
        coefficient_count = int(coefficient_count)
        if coefficient_count not in (9, 19):
            raise ValueError("coefficient_count must be 9 or 19")
        return cls(
            layout=layout,
            target_i=jnp.zeros((max_rows,), dtype=jnp.int32),
            target_j=jnp.zeros((max_rows,), dtype=jnp.int32),
            target_k=jnp.zeros((max_rows,), dtype=jnp.int32),
            equation_kind=jnp.zeros(
                (max_rows, max_equations),
                dtype=jnp.int32,
            ),
            sample_i=jnp.zeros((max_rows, max_equations), dtype=jnp.int32),
            sample_j=jnp.zeros((max_rows, max_equations), dtype=jnp.int32),
            sample_k=jnp.zeros((max_rows, max_equations), dtype=jnp.int32),
            boundary_face_row=jnp.zeros(
                (max_rows, max_equations),
                dtype=jnp.int32,
            ),
            equation_active=jnp.zeros((max_rows, max_equations), dtype=bool),
            rhs_transform=jnp.zeros(
                (max_rows, coefficient_count, max_equations),
                dtype=jnp.float64,
            ),
            active=jnp.zeros((max_rows,), dtype=bool),
            target_row_for_cell=-jnp.ones(layout.owned_shape, dtype=jnp.int32),
            polynomial_order=jnp.zeros((max_rows,), dtype=jnp.int32),
            rank=jnp.zeros((max_rows,), dtype=jnp.int32),
            condition_number=jnp.full((max_rows,), jnp.inf, dtype=jnp.float64),
            max_rows=max_rows,
            max_equations=max_equations,
        )

    def tree_flatten(self):
        return (
            (
                self.target_i,
                self.target_j,
                self.target_k,
                self.equation_kind,
                self.sample_i,
                self.sample_j,
                self.sample_k,
                self.boundary_face_row,
                self.boundary_patch,
                self.boundary_quadrature,
                self.equation_active,
                self.rhs_transform,
                self.active,
                self.target_row_for_cell,
                self.polynomial_order,
                self.rank,
                self.condition_number,
            ),
            (self.layout, self.max_rows, self.max_equations),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        layout, max_rows, max_equations = aux_data
        names = (
            "target_i",
            "target_j",
            "target_k",
            "equation_kind",
            "sample_i",
            "sample_j",
            "sample_k",
            "boundary_face_row",
            "boundary_patch",
            "boundary_quadrature",
            "equation_active",
            "rhs_transform",
            "active",
            "target_row_for_cell",
            "polynomial_order",
            "rank",
            "condition_number",
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "layout", layout)
        object.__setattr__(instance, "max_rows", max_rows)
        object.__setattr__(instance, "max_equations", max_equations)
        for name, value in zip(names, children):
            object.__setattr__(instance, name, value)
        return instance


@_pytree_base
@dataclass(frozen=True)
class LocalControlVolumePolynomial3D(_DataclassPyTreeMixin):
    """Runtime moment-aware polynomial coefficients for one owned scalar field."""

    gradient: jnp.ndarray
    hessian: jnp.ndarray
    valid: jnp.ndarray
    polynomial_order: jnp.ndarray
    condition_number: jnp.ndarray
    third_derivative: jnp.ndarray | None = None
    owner_values: jnp.ndarray | None = None
    remote_face_value: jnp.ndarray | None = None
    remote_face_gradient: jnp.ndarray | None = None
    remote_face_valid: jnp.ndarray | None = None
    remote_functional_value: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        gradient = jnp.asarray(self.gradient, dtype=jnp.float64)
        hessian = jnp.asarray(self.hessian, dtype=jnp.float64)
        valid = jnp.asarray(self.valid, dtype=bool)
        polynomial_order = jnp.asarray(self.polynomial_order, dtype=jnp.int32)
        condition_number = jnp.asarray(self.condition_number, dtype=jnp.float64)
        owner_values = (
            None
            if self.owner_values is None
            else jnp.asarray(self.owner_values, dtype=jnp.float64)
        )
        if self.remote_face_value is None:
            remote_face_value = jnp.zeros((0, 0, 0), dtype=jnp.float64)
            remote_face_gradient = jnp.zeros((0, 0, 0, 3), dtype=jnp.float64)
            remote_face_valid = jnp.zeros((0, 0, 0), dtype=bool)
        else:
            remote_face_value = jnp.asarray(
                self.remote_face_value,
                dtype=jnp.float64,
            )
            if self.remote_face_gradient is None or self.remote_face_valid is None:
                raise ValueError(
                    "remote_face_gradient and remote_face_valid are required "
                    "when remote_face_value is provided"
                )
            remote_face_gradient = jnp.asarray(
                self.remote_face_gradient,
                dtype=jnp.float64,
            )
            remote_face_valid = jnp.asarray(self.remote_face_valid, dtype=bool)
            if remote_face_value.ndim != 3:
                raise ValueError(
                    "remote_face_value must have shape "
                    "(face_rows, patches, quadrature_points)"
                )
            if remote_face_gradient.shape != remote_face_value.shape + (3,):
                raise ValueError(
                    "remote_face_gradient must have shape "
                    "remote_face_value.shape + (3,)"
                )
            if remote_face_valid.shape != remote_face_value.shape:
                raise ValueError(
                    "remote_face_valid must have shape remote_face_value.shape"
                )
        remote_functional_value = jnp.asarray(
            (
                jnp.zeros((0, 0), dtype=jnp.float64)
                if self.remote_functional_value is None
                else self.remote_functional_value
            ),
            dtype=jnp.float64,
        )
        if remote_functional_value.ndim != 2:
            raise ValueError(
                "remote_functional_value must have shape "
                "(face_functional_rows, equations)"
            )
        if gradient.ndim != 4 or gradient.shape[-1] != 3:
            raise ValueError("polynomial gradient must have shape owned_shape + (3,)")
        owned_shape = gradient.shape[:-1]
        third_derivative = jnp.asarray(
            jnp.zeros(owned_shape + (3, 3, 3), dtype=jnp.float64)
            if self.third_derivative is None
            else self.third_derivative,
            dtype=jnp.float64,
        )
        if hessian.shape != owned_shape + (3, 3):
            raise ValueError("polynomial hessian must have shape owned_shape + (3, 3)")
        if third_derivative.shape != owned_shape + (3, 3, 3):
            raise ValueError(
                "polynomial third_derivative must have shape owned_shape + (3, 3, 3)"
            )
        for name, array in (
            ("valid", valid),
            ("polynomial_order", polynomial_order),
            ("condition_number", condition_number),
        ):
            if array.shape != owned_shape:
                raise ValueError(f"polynomial {name} must have shape {owned_shape}")
        if owner_values is not None and owner_values.shape != owned_shape:
            raise ValueError(
                "polynomial owner_values must have shape "
                f"{owned_shape}, got {owner_values.shape}"
            )
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(
            self,
            "hessian",
            0.5 * (hessian + jnp.swapaxes(hessian, -1, -2)),
        )
        permutations = (
            (0, 1, 2), (0, 2, 1), (1, 0, 2),
            (1, 2, 0), (2, 0, 1), (2, 1, 0),
        )
        object.__setattr__(
            self,
            "third_derivative",
            sum(
                jnp.transpose(
                    third_derivative,
                    (0, 1, 2) + tuple(axis + 3 for axis in permutation),
                )
                for permutation in permutations
            ) / 6.0,
        )
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "polynomial_order", polynomial_order)
        object.__setattr__(self, "condition_number", condition_number)
        object.__setattr__(self, "owner_values", owner_values)
        object.__setattr__(self, "remote_face_value", remote_face_value)
        object.__setattr__(self, "remote_face_gradient", remote_face_gradient)
        object.__setattr__(self, "remote_face_valid", remote_face_valid)
        object.__setattr__(
            self,
            "remote_functional_value",
            remote_functional_value,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.gradient.shape[:-1])

    def as_cell_gradient(self) -> LocalCellGradient3D:
        return LocalCellGradient3D(
            gradient=self.gradient,
            valid=self.valid,
            reconstruction_mask=self.polynomial_order > 0,
        )


@_pytree_base
@dataclass(frozen=True)
class LocalRegularBoundaryMomentClosure3D(_DataclassPyTreeMixin):
    """Moment-aware Dirichlet derivative weights on regular boundaries.

    The final axis of each face/owner weight array multiplies
    ``(boundary_value, first, second, third inward cell average)``.  Weights
    return the positive-coordinate derivative either at the boundary face or
    at the first control-volume centroid. Only entries selected by the
    corresponding validity mask are used.
    """

    layout: HaloLayout3D
    x_face_weights: jnp.ndarray
    y_face_weights: jnp.ndarray
    z_face_weights: jnp.ndarray
    x_owner_weights: jnp.ndarray
    y_owner_weights: jnp.ndarray
    z_owner_weights: jnp.ndarray
    x_valid: jnp.ndarray
    y_valid: jnp.ndarray
    z_valid: jnp.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D")
        face_shapes = tuple(
            self.layout.face_control_shape(axis) for axis in range(3)
        )
        face_weights = tuple(
            jnp.asarray(value, dtype=jnp.float64)
            for value in (
                self.x_face_weights,
                self.y_face_weights,
                self.z_face_weights,
            )
        )
        owner_weights = tuple(
            jnp.asarray(value, dtype=jnp.float64)
            for value in (
                self.x_owner_weights,
                self.y_owner_weights,
                self.z_owner_weights,
            )
        )
        valid = tuple(
            jnp.asarray(value, dtype=bool)
            for value in (self.x_valid, self.y_valid, self.z_valid)
        )
        for axis, (axis_face, axis_owner, axis_valid, face_shape) in enumerate(
            zip(face_weights, owner_weights, valid, face_shapes)
        ):
            for name, axis_weights in (
                ("face", axis_face),
                ("owner", axis_owner),
            ):
                if axis_weights.shape != face_shape + (4,):
                    raise ValueError(
                        f"regular boundary {name} weights must have shape "
                        f"{face_shape + (4,)} on axis {axis}, got "
                        f"{axis_weights.shape}"
                    )
            if axis_valid.shape != face_shape:
                raise ValueError(
                    "regular boundary derivative validity must have shape "
                    f"{face_shape} on axis {axis}, got {axis_valid.shape}"
                )
            try:
                finite = bool(
                    jnp.all(
                        (~axis_valid)[..., None]
                        | (
                            jnp.isfinite(axis_face)
                            & jnp.isfinite(axis_owner)
                        )
                    )
                )
            except jax.errors.TracerBoolConversionError:
                finite = True
            if not finite:
                raise ValueError(
                    "active regular boundary derivative weights must be finite"
                )
        object.__setattr__(self, "x_face_weights", face_weights[0])
        object.__setattr__(self, "y_face_weights", face_weights[1])
        object.__setattr__(self, "z_face_weights", face_weights[2])
        object.__setattr__(self, "x_owner_weights", owner_weights[0])
        object.__setattr__(self, "y_owner_weights", owner_weights[1])
        object.__setattr__(self, "z_owner_weights", owner_weights[2])
        object.__setattr__(self, "x_valid", valid[0])
        object.__setattr__(self, "y_valid", valid[1])
        object.__setattr__(self, "z_valid", valid[2])

    @classmethod
    def empty(
        cls,
        layout: HaloLayout3D,
    ) -> "LocalRegularBoundaryMomentClosure3D":
        face_shapes = tuple(layout.face_control_shape(axis) for axis in range(3))
        zero_weights = tuple(
            jnp.zeros(shape + (4,), dtype=jnp.float64)
            for shape in face_shapes
        )
        return cls(
            layout=layout,
            x_face_weights=zero_weights[0],
            y_face_weights=zero_weights[1],
            z_face_weights=zero_weights[2],
            x_owner_weights=zero_weights[0],
            y_owner_weights=zero_weights[1],
            z_owner_weights=zero_weights[2],
            x_valid=jnp.zeros(face_shapes[0], dtype=bool),
            y_valid=jnp.zeros(face_shapes[1], dtype=bool),
            z_valid=jnp.zeros(face_shapes[2], dtype=bool),
        )

    def axis_payload(
        self,
        axis: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return (
            (
                self.x_face_weights,
                self.y_face_weights,
                self.z_face_weights,
            )[axis],
            (
                self.x_owner_weights,
                self.y_owner_weights,
                self.z_owner_weights,
            )[axis],
            (self.x_valid, self.y_valid, self.z_valid)[axis],
        )

    def tree_flatten(self):
        return (
            (
                self.x_face_weights,
                self.y_face_weights,
                self.z_face_weights,
                self.x_owner_weights,
                self.y_owner_weights,
                self.z_owner_weights,
                self.x_valid,
                self.y_valid,
                self.z_valid,
            ),
            self.layout,
        )

    @classmethod
    def tree_unflatten(cls, layout, children):
        instance = object.__new__(cls)
        object.__setattr__(instance, "layout", layout)
        for name, value in zip(
            (
                "x_face_weights",
                "y_face_weights",
                "z_face_weights",
                "x_owner_weights",
                "y_owner_weights",
                "z_owner_weights",
                "x_valid",
                "y_valid",
                "z_valid",
            ),
            children,
        ):
            object.__setattr__(instance, name, value)
        return instance


@_pytree_base
@dataclass(frozen=True)
class LocalEmbeddedControlVolumeGeometry3D(_DataclassPyTreeMixin):
    """Unified geometry used by embedded-boundary reconstruction and fluxes."""

    cells: LocalControlVolumeCellGeometry3D
    regular_faces: LocalRegularFaceGeometry3D
    irregular_faces: LocalControlVolumeFaceRows3D
    reconstruction: LocalMomentReconstruction3D
    centroid_J: jnp.ndarray | None = None
    centroid_g_cov: jnp.ndarray | None = None
    centroid_B_contra: jnp.ndarray | None = None
    centroid_Bmag: jnp.ndarray | None = None
    centroid_curvature: jnp.ndarray | None = None
    regular_boundary_closure: (
        LocalRegularBoundaryMomentClosure3D | None
    ) = None

    def __post_init__(self) -> None:
        if not isinstance(self.cells, LocalControlVolumeCellGeometry3D):
            raise TypeError("cells must be LocalControlVolumeCellGeometry3D")
        if not isinstance(self.regular_faces, LocalRegularFaceGeometry3D):
            raise TypeError("regular_faces must be LocalRegularFaceGeometry3D")
        if not isinstance(self.irregular_faces, LocalControlVolumeFaceRows3D):
            raise TypeError("irregular_faces must be LocalControlVolumeFaceRows3D")
        if not isinstance(self.reconstruction, LocalMomentReconstruction3D):
            raise TypeError("reconstruction must be LocalMomentReconstruction3D")
        layout = self.cells.layout
        for name, other_layout in (
            ("regular_faces", self.regular_faces.layout),
            ("irregular_faces", self.irregular_faces.layout),
            ("reconstruction", self.reconstruction.layout),
        ):
            if other_layout != layout:
                raise ValueError(f"{name} must share the control-volume cell layout")
        if self.regular_boundary_closure is not None:
            if not isinstance(
                self.regular_boundary_closure,
                LocalRegularBoundaryMomentClosure3D,
            ):
                raise TypeError(
                    "regular_boundary_closure must be "
                    "LocalRegularBoundaryMomentClosure3D or None"
                )
            if self.regular_boundary_closure.layout != layout:
                raise ValueError(
                    "regular_boundary_closure must share the "
                    "control-volume cell layout"
                )
        coefficient_values = (
            self.centroid_J,
            self.centroid_g_cov,
            self.centroid_B_contra,
            self.centroid_Bmag,
            self.centroid_curvature,
        )
        if any(value is not None for value in coefficient_values):
            if not all(value is not None for value in coefficient_values):
                raise ValueError(
                    "centroid operator geometry must provide J, g_cov, "
                    "B_contra, Bmag, and curvature together"
                )
            shape = self.cells.shape
            centroid_J = jnp.asarray(self.centroid_J, dtype=jnp.float64)
            centroid_g_cov = jnp.asarray(
                self.centroid_g_cov,
                dtype=jnp.float64,
            )
            centroid_B_contra = jnp.asarray(
                self.centroid_B_contra,
                dtype=jnp.float64,
            )
            centroid_Bmag = jnp.asarray(
                self.centroid_Bmag,
                dtype=jnp.float64,
            )
            centroid_curvature = jnp.asarray(
                self.centroid_curvature,
                dtype=jnp.float64,
            )
            expected_shapes = (
                (centroid_J, shape, "centroid_J"),
                (centroid_g_cov, shape + (3, 3), "centroid_g_cov"),
                (
                    centroid_B_contra,
                    shape + (3,),
                    "centroid_B_contra",
                ),
                (centroid_Bmag, shape, "centroid_Bmag"),
                (
                    centroid_curvature,
                    shape + (3,),
                    "centroid_curvature",
                ),
            )
            for value, expected, name in expected_shapes:
                if value.shape != expected:
                    raise ValueError(
                        f"{name} must have shape {expected}, got {value.shape}"
                    )
            active = self.cells.is_active_owner
            finite = (
                jnp.isfinite(centroid_J)
                & jnp.isfinite(centroid_Bmag)
                & jnp.all(jnp.isfinite(centroid_g_cov), axis=(-2, -1))
                & jnp.all(jnp.isfinite(centroid_B_contra), axis=-1)
                & jnp.all(jnp.isfinite(centroid_curvature), axis=-1)
            )
            try:
                all_finite = bool(jnp.all((~active) | finite))
            except jax.errors.TracerBoolConversionError:
                all_finite = True
            if not all_finite:
                raise ValueError(
                    "active centroid operator geometry must be finite"
                )
            object.__setattr__(self, "centroid_J", centroid_J)
            object.__setattr__(self, "centroid_g_cov", centroid_g_cov)
            object.__setattr__(
                self,
                "centroid_B_contra",
                centroid_B_contra,
            )
            object.__setattr__(self, "centroid_Bmag", centroid_Bmag)
            object.__setattr__(
                self,
                "centroid_curvature",
                centroid_curvature,
            )

    @property
    def layout(self) -> HaloLayout3D:
        return self.cells.layout

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.cells.shape

    @property
    def has_centroid_operator_geometry(self) -> bool:
        return self.centroid_J is not None

    def tree_flatten(self):
        return (
            (
                self.cells,
                self.regular_faces,
                self.irregular_faces,
                self.reconstruction,
                self.centroid_J,
                self.centroid_g_cov,
                self.centroid_B_contra,
                self.centroid_Bmag,
                self.centroid_curvature,
                self.regular_boundary_closure,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        names = (
            "cells",
            "regular_faces",
            "irregular_faces",
            "reconstruction",
            "centroid_J",
            "centroid_g_cov",
            "centroid_B_contra",
            "centroid_Bmag",
            "centroid_curvature",
            "regular_boundary_closure",
        )
        instance = object.__new__(cls)
        for name, value in zip(names, children):
            object.__setattr__(instance, name, value)
        return instance


@_pytree_base
@dataclass(frozen=True)
class LocalCutWallValueReconstructor3D(_DataclassPyTreeMixin):
    """Reconstruct field values at local cut-wall pieces from field_halo."""

    cut_wall_geometry: LocalCutWallGeometry3D
    neighbor_i: jnp.ndarray
    neighbor_j: jnp.ndarray
    neighbor_k: jnp.ndarray
    weights: jnp.ndarray
    active: jnp.ndarray
    stencil_width: int
    max_wall_faces: int

    def __post_init__(self) -> None:
        if not isinstance(self.cut_wall_geometry, LocalCutWallGeometry3D):
            raise TypeError("cut_wall_geometry must be a LocalCutWallGeometry3D")

        max_wall_faces = int(self.max_wall_faces)
        stencil_width = int(self.stencil_width)
        if max_wall_faces < 0:
            raise ValueError(f"max_wall_faces must be non-negative, got {max_wall_faces}")
        if stencil_width < 1:
            raise ValueError(f"stencil_width must be positive, got {stencil_width}")
        if self.cut_wall_geometry.max_wall_faces != max_wall_faces:
            raise ValueError(
                "LocalCutWallValueReconstructor3D.cut_wall_geometry.max_wall_faces must match max_wall_faces; "
                f"got {self.cut_wall_geometry.max_wall_faces} and {max_wall_faces}"
            )

        expected_stencil = (max_wall_faces, stencil_width)
        expected_wall = (max_wall_faces,)
        neighbor_i = _as_local_wall_stencil_index_array(
            self.neighbor_i, max_wall_faces, stencil_width, "LocalCutWallValueReconstructor3D.neighbor_i"
        )
        neighbor_j = _as_local_wall_stencil_index_array(
            self.neighbor_j, max_wall_faces, stencil_width, "LocalCutWallValueReconstructor3D.neighbor_j"
        )
        neighbor_k = _as_local_wall_stencil_index_array(
            self.neighbor_k, max_wall_faces, stencil_width, "LocalCutWallValueReconstructor3D.neighbor_k"
        )
        weights = _as_local_wall_stencil_weight_array(
            self.weights, max_wall_faces, stencil_width, "LocalCutWallValueReconstructor3D.weights"
        )
        active = _as_local_wall_bool_array(self.active, max_wall_faces, "LocalCutWallValueReconstructor3D.active")

        if active.shape != expected_wall:
            raise ValueError(
                f"LocalCutWallValueReconstructor3D.active must have shape {expected_wall}, got {active.shape}"
            )

        active2 = active[:, None]
        neighbor_i = jnp.where(active2, neighbor_i, 0)
        neighbor_j = jnp.where(active2, neighbor_j, 0)
        neighbor_k = jnp.where(active2, neighbor_k, 0)
        weights = jnp.where(active2, weights, 0.0)

        object.__setattr__(self, "neighbor_i", neighbor_i)
        object.__setattr__(self, "neighbor_j", neighbor_j)
        object.__setattr__(self, "neighbor_k", neighbor_k)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "stencil_width", stencil_width)
        object.__setattr__(self, "max_wall_faces", max_wall_faces)

    def extrapolate(self, field_halo: jnp.ndarray) -> jnp.ndarray:
        values = jnp.asarray(field_halo, dtype=jnp.float64)
        gathered = values[self.neighbor_i, self.neighbor_j, self.neighbor_k]
        wall_value = jnp.sum(self.weights * gathered, axis=-1)
        return jnp.where(self.active, wall_value, 0.0)

    def tree_flatten(self):
        return (
            (
                self.cut_wall_geometry,
                self.neighbor_i,
                self.neighbor_j,
                self.neighbor_k,
                self.weights,
                self.active,
            ),
            (self.stencil_width, self.max_wall_faces),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        stencil_width, max_wall_faces = aux_data
        (
            cut_wall_geometry,
            neighbor_i,
            neighbor_j,
            neighbor_k,
            weights,
            active,
        ) = children
        return cls(
            cut_wall_geometry=cut_wall_geometry,
            neighbor_i=neighbor_i,
            neighbor_j=neighbor_j,
            neighbor_k=neighbor_k,
            weights=weights,
            active=active,
            stencil_width=stencil_width,
            max_wall_faces=max_wall_faces,
        )


@_pytree_base
@dataclass(frozen=True)
class LocalCutWallNormalDerivativeConstructor3D(_DataclassPyTreeMixin):
    """Construct wall-normal derivatives at local cut-wall pieces from field_halo."""

    cut_wall_geometry: LocalCutWallGeometry3D
    neighbor_i: jnp.ndarray
    neighbor_j: jnp.ndarray
    neighbor_k: jnp.ndarray
    weights_dnormal: jnp.ndarray
    weights_d2normal: jnp.ndarray
    wall_coeff_dnormal: jnp.ndarray
    wall_coeff_d2normal: jnp.ndarray
    active: jnp.ndarray
    stencil_width: int
    max_wall_faces: int

    def __post_init__(self) -> None:
        if not isinstance(self.cut_wall_geometry, LocalCutWallGeometry3D):
            raise TypeError("cut_wall_geometry must be a LocalCutWallGeometry3D")

        max_wall_faces = int(self.max_wall_faces)
        stencil_width = int(self.stencil_width)
        if max_wall_faces < 0:
            raise ValueError(f"max_wall_faces must be non-negative, got {max_wall_faces}")
        if stencil_width < 1:
            raise ValueError(f"stencil_width must be positive, got {stencil_width}")
        if self.cut_wall_geometry.max_wall_faces != max_wall_faces:
            raise ValueError(
                "LocalCutWallNormalDerivativeConstructor3D.cut_wall_geometry.max_wall_faces must match max_wall_faces; "
                f"got {self.cut_wall_geometry.max_wall_faces} and {max_wall_faces}"
            )

        expected_stencil = (max_wall_faces, stencil_width)
        expected_wall = (max_wall_faces,)
        neighbor_i = _as_local_wall_stencil_index_array(
            self.neighbor_i, max_wall_faces, stencil_width, "LocalCutWallNormalDerivativeConstructor3D.neighbor_i"
        )
        neighbor_j = _as_local_wall_stencil_index_array(
            self.neighbor_j, max_wall_faces, stencil_width, "LocalCutWallNormalDerivativeConstructor3D.neighbor_j"
        )
        neighbor_k = _as_local_wall_stencil_index_array(
            self.neighbor_k, max_wall_faces, stencil_width, "LocalCutWallNormalDerivativeConstructor3D.neighbor_k"
        )
        weights_dnormal = _as_local_wall_stencil_weight_array(
            self.weights_dnormal, max_wall_faces, stencil_width, "LocalCutWallNormalDerivativeConstructor3D.weights_dnormal"
        )
        weights_d2normal = _as_local_wall_stencil_weight_array(
            self.weights_d2normal, max_wall_faces, stencil_width, "LocalCutWallNormalDerivativeConstructor3D.weights_d2normal"
        )
        wall_coeff_dnormal = _as_local_wall_array(
            self.wall_coeff_dnormal, max_wall_faces, (), "LocalCutWallNormalDerivativeConstructor3D.wall_coeff_dnormal"
        )
        wall_coeff_d2normal = _as_local_wall_array(
            self.wall_coeff_d2normal, max_wall_faces, (), "LocalCutWallNormalDerivativeConstructor3D.wall_coeff_d2normal"
        )
        active = _as_local_wall_bool_array(self.active, max_wall_faces, "LocalCutWallNormalDerivativeConstructor3D.active")

        if active.shape != expected_wall:
            raise ValueError(
                f"LocalCutWallNormalDerivativeConstructor3D.active must have shape {expected_wall}, got {active.shape}"
            )

        active2 = active[:, None]
        neighbor_i = jnp.where(active2, neighbor_i, 0)
        neighbor_j = jnp.where(active2, neighbor_j, 0)
        neighbor_k = jnp.where(active2, neighbor_k, 0)
        weights_dnormal = jnp.where(active2, weights_dnormal, 0.0)
        weights_d2normal = jnp.where(active2, weights_d2normal, 0.0)
        wall_coeff_dnormal = jnp.where(active, wall_coeff_dnormal, 0.0)
        wall_coeff_d2normal = jnp.where(active, wall_coeff_d2normal, 0.0)

        object.__setattr__(self, "neighbor_i", neighbor_i)
        object.__setattr__(self, "neighbor_j", neighbor_j)
        object.__setattr__(self, "neighbor_k", neighbor_k)
        object.__setattr__(self, "weights_dnormal", weights_dnormal)
        object.__setattr__(self, "weights_d2normal", weights_d2normal)
        object.__setattr__(self, "wall_coeff_dnormal", wall_coeff_dnormal)
        object.__setattr__(self, "wall_coeff_d2normal", wall_coeff_d2normal)
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "stencil_width", stencil_width)
        object.__setattr__(self, "max_wall_faces", max_wall_faces)

    def _gather(self, field_halo: jnp.ndarray) -> jnp.ndarray:
        values = jnp.asarray(field_halo, dtype=jnp.float64)
        return values[self.neighbor_i, self.neighbor_j, self.neighbor_k]

    def dnormal_from_wall_value(self, field_halo: jnp.ndarray, wall_value: jnp.ndarray) -> jnp.ndarray:
        wall_value = jnp.asarray(wall_value, dtype=jnp.float64)
        expected_wall = (int(self.max_wall_faces),)
        if wall_value.shape != expected_wall:
            raise ValueError(f"wall_value must have shape {expected_wall}, got {wall_value.shape}")
        gathered = self._gather(field_halo)
        dnormal = jnp.sum(self.weights_dnormal * gathered, axis=-1) + self.wall_coeff_dnormal * wall_value
        return jnp.where(self.active, dnormal, 0.0)

    def d2normal_from_wall_value(self, field_halo: jnp.ndarray, wall_value: jnp.ndarray) -> jnp.ndarray:
        wall_value = jnp.asarray(wall_value, dtype=jnp.float64)
        expected_wall = (int(self.max_wall_faces),)
        if wall_value.shape != expected_wall:
            raise ValueError(f"wall_value must have shape {expected_wall}, got {wall_value.shape}")
        gathered = self._gather(field_halo)
        d2normal = jnp.sum(self.weights_d2normal * gathered, axis=-1) + self.wall_coeff_d2normal * wall_value
        return jnp.where(self.active, d2normal, 0.0)

    def normal_derivatives_from_wall_value(
        self, field_halo: jnp.ndarray, wall_value: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return (
            self.dnormal_from_wall_value(field_halo, wall_value),
            self.d2normal_from_wall_value(field_halo, wall_value),
        )

    def tree_flatten(self):
        return (
            (
                self.cut_wall_geometry,
                self.neighbor_i,
                self.neighbor_j,
                self.neighbor_k,
                self.weights_dnormal,
                self.weights_d2normal,
                self.wall_coeff_dnormal,
                self.wall_coeff_d2normal,
                self.active,
            ),
            (self.stencil_width, self.max_wall_faces),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        stencil_width, max_wall_faces = aux_data
        (
            cut_wall_geometry,
            neighbor_i,
            neighbor_j,
            neighbor_k,
            weights_dnormal,
            weights_d2normal,
            wall_coeff_dnormal,
            wall_coeff_d2normal,
            active,
        ) = children
        return cls(
            cut_wall_geometry=cut_wall_geometry,
            neighbor_i=neighbor_i,
            neighbor_j=neighbor_j,
            neighbor_k=neighbor_k,
            weights_dnormal=weights_dnormal,
            weights_d2normal=weights_d2normal,
            wall_coeff_dnormal=wall_coeff_dnormal,
            wall_coeff_d2normal=wall_coeff_d2normal,
            active=active,
            stencil_width=stencil_width,
            max_wall_faces=max_wall_faces,
        )


@_pytree_base
@dataclass(frozen=True)
class LocalRegularFaceContributionRows3D(_DataclassPyTreeMixin):
    """Sparse regular-face flux rows for agglomerated control volumes.

    Each active row samples one already-built regular face flux, multiplies it
    by ``area`` and ``sign``, and scatters it into the row owner cell.  This is
    used when a deactivated cut/sliver cell's exterior regular face belongs to
    a neighboring agglomerated active control volume instead of the structured
    coordinate cell-difference update.
    """

    owner_i: jnp.ndarray
    owner_j: jnp.ndarray
    owner_k: jnp.ndarray
    face_axis: jnp.ndarray
    face_i: jnp.ndarray
    face_j: jnp.ndarray
    face_k: jnp.ndarray
    sign: jnp.ndarray
    area: jnp.ndarray
    active: jnp.ndarray
    max_rows: int
    minus_owner_i: jnp.ndarray | None = None
    minus_owner_j: jnp.ndarray | None = None
    minus_owner_k: jnp.ndarray | None = None
    plus_owner_i: jnp.ndarray | None = None
    plus_owner_j: jnp.ndarray | None = None
    plus_owner_k: jnp.ndarray | None = None
    use_reconstructed_flux: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        max_rows = int(self.max_rows)
        if max_rows < 0:
            raise ValueError(f"max_rows must be non-negative, got {max_rows}")
        shape = (max_rows,)
        owner_i = _as_local_wall_int_array(self.owner_i, max_rows, "LocalRegularFaceContributionRows3D.owner_i")
        owner_j = _as_local_wall_int_array(self.owner_j, max_rows, "LocalRegularFaceContributionRows3D.owner_j")
        owner_k = _as_local_wall_int_array(self.owner_k, max_rows, "LocalRegularFaceContributionRows3D.owner_k")
        face_axis = _as_local_wall_int_array(self.face_axis, max_rows, "LocalRegularFaceContributionRows3D.face_axis")
        face_i = _as_local_wall_int_array(self.face_i, max_rows, "LocalRegularFaceContributionRows3D.face_i")
        face_j = _as_local_wall_int_array(self.face_j, max_rows, "LocalRegularFaceContributionRows3D.face_j")
        face_k = _as_local_wall_int_array(self.face_k, max_rows, "LocalRegularFaceContributionRows3D.face_k")
        sign = _as_local_wall_array(self.sign, max_rows, (), "LocalRegularFaceContributionRows3D.sign")
        area = _as_local_wall_array(self.area, max_rows, (), "LocalRegularFaceContributionRows3D.area")
        active = _as_local_wall_bool_array(self.active, max_rows, "LocalRegularFaceContributionRows3D.active")
        minus_owner_i = (
            owner_i
            if self.minus_owner_i is None
            else _as_local_wall_int_array(
                self.minus_owner_i,
                max_rows,
                "LocalRegularFaceContributionRows3D.minus_owner_i",
            )
        )
        minus_owner_j = (
            owner_j
            if self.minus_owner_j is None
            else _as_local_wall_int_array(
                self.minus_owner_j,
                max_rows,
                "LocalRegularFaceContributionRows3D.minus_owner_j",
            )
        )
        minus_owner_k = (
            owner_k
            if self.minus_owner_k is None
            else _as_local_wall_int_array(
                self.minus_owner_k,
                max_rows,
                "LocalRegularFaceContributionRows3D.minus_owner_k",
            )
        )
        plus_owner_i = (
            owner_i
            if self.plus_owner_i is None
            else _as_local_wall_int_array(
                self.plus_owner_i,
                max_rows,
                "LocalRegularFaceContributionRows3D.plus_owner_i",
            )
        )
        plus_owner_j = (
            owner_j
            if self.plus_owner_j is None
            else _as_local_wall_int_array(
                self.plus_owner_j,
                max_rows,
                "LocalRegularFaceContributionRows3D.plus_owner_j",
            )
        )
        plus_owner_k = (
            owner_k
            if self.plus_owner_k is None
            else _as_local_wall_int_array(
                self.plus_owner_k,
                max_rows,
                "LocalRegularFaceContributionRows3D.plus_owner_k",
            )
        )
        use_reconstructed_flux = (
            jnp.zeros(shape, dtype=bool)
            if self.use_reconstructed_flux is None
            else _as_local_wall_bool_array(
                self.use_reconstructed_flux,
                max_rows,
                "LocalRegularFaceContributionRows3D.use_reconstructed_flux",
            )
        )

        valid_axis = (~active) | ((face_axis >= 0) & (face_axis <= 2))
        try:
            all_valid_axis = bool(jnp.all(valid_axis))
        except jax.errors.TracerBoolConversionError:
            all_valid_axis = True
        if not all_valid_axis:
            raise ValueError("active LocalRegularFaceContributionRows3D.face_axis values must be 0, 1, or 2")

        object.__setattr__(self, "owner_i", jnp.where(active, owner_i, 0))
        object.__setattr__(self, "owner_j", jnp.where(active, owner_j, 0))
        object.__setattr__(self, "owner_k", jnp.where(active, owner_k, 0))
        object.__setattr__(self, "face_axis", jnp.where(active, face_axis, 0))
        object.__setattr__(self, "face_i", jnp.where(active, face_i, 0))
        object.__setattr__(self, "face_j", jnp.where(active, face_j, 0))
        object.__setattr__(self, "face_k", jnp.where(active, face_k, 0))
        object.__setattr__(self, "sign", jnp.where(active, sign, 0.0))
        object.__setattr__(self, "area", jnp.where(active, area, 0.0))
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "max_rows", max_rows)
        object.__setattr__(self, "minus_owner_i", jnp.where(active, minus_owner_i, 0))
        object.__setattr__(self, "minus_owner_j", jnp.where(active, minus_owner_j, 0))
        object.__setattr__(self, "minus_owner_k", jnp.where(active, minus_owner_k, 0))
        object.__setattr__(self, "plus_owner_i", jnp.where(active, plus_owner_i, 0))
        object.__setattr__(self, "plus_owner_j", jnp.where(active, plus_owner_j, 0))
        object.__setattr__(self, "plus_owner_k", jnp.where(active, plus_owner_k, 0))
        object.__setattr__(
            self,
            "use_reconstructed_flux",
            active & use_reconstructed_flux,
        )

    @classmethod
    def empty(cls, max_rows: int = 0) -> "LocalRegularFaceContributionRows3D":
        max_rows = int(max_rows)
        return cls(
            owner_i=jnp.zeros((max_rows,), dtype=jnp.int32),
            owner_j=jnp.zeros((max_rows,), dtype=jnp.int32),
            owner_k=jnp.zeros((max_rows,), dtype=jnp.int32),
            face_axis=jnp.zeros((max_rows,), dtype=jnp.int32),
            face_i=jnp.zeros((max_rows,), dtype=jnp.int32),
            face_j=jnp.zeros((max_rows,), dtype=jnp.int32),
            face_k=jnp.zeros((max_rows,), dtype=jnp.int32),
            sign=jnp.zeros((max_rows,), dtype=jnp.float64),
            area=jnp.zeros((max_rows,), dtype=jnp.float64),
            active=jnp.zeros((max_rows,), dtype=bool),
            max_rows=max_rows,
            minus_owner_i=jnp.zeros((max_rows,), dtype=jnp.int32),
            minus_owner_j=jnp.zeros((max_rows,), dtype=jnp.int32),
            minus_owner_k=jnp.zeros((max_rows,), dtype=jnp.int32),
            plus_owner_i=jnp.zeros((max_rows,), dtype=jnp.int32),
            plus_owner_j=jnp.zeros((max_rows,), dtype=jnp.int32),
            plus_owner_k=jnp.zeros((max_rows,), dtype=jnp.int32),
            use_reconstructed_flux=jnp.zeros((max_rows,), dtype=bool),
        )

    @property
    def n_rows(self) -> int:
        return int(self.max_rows)

    def tree_flatten(self):
        return (
            (
                self.owner_i,
                self.owner_j,
                self.owner_k,
                self.face_axis,
                self.face_i,
                self.face_j,
                self.face_k,
                self.sign,
                self.area,
                self.active,
                self.minus_owner_i,
                self.minus_owner_j,
                self.minus_owner_k,
                self.plus_owner_i,
                self.plus_owner_j,
                self.plus_owner_k,
                self.use_reconstructed_flux,
            ),
            self.max_rows,
        )

    @classmethod
    def tree_unflatten(cls, max_rows, children):
        (
            owner_i,
            owner_j,
            owner_k,
            face_axis,
            face_i,
            face_j,
            face_k,
            sign,
            area,
            active,
            minus_owner_i,
            minus_owner_j,
            minus_owner_k,
            plus_owner_i,
            plus_owner_j,
            plus_owner_k,
            use_reconstructed_flux,
        ) = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "owner_i", owner_i)
        object.__setattr__(obj, "owner_j", owner_j)
        object.__setattr__(obj, "owner_k", owner_k)
        object.__setattr__(obj, "face_axis", face_axis)
        object.__setattr__(obj, "face_i", face_i)
        object.__setattr__(obj, "face_j", face_j)
        object.__setattr__(obj, "face_k", face_k)
        object.__setattr__(obj, "sign", sign)
        object.__setattr__(obj, "area", area)
        object.__setattr__(obj, "active", active)
        object.__setattr__(obj, "minus_owner_i", minus_owner_i)
        object.__setattr__(obj, "minus_owner_j", minus_owner_j)
        object.__setattr__(obj, "minus_owner_k", minus_owner_k)
        object.__setattr__(obj, "plus_owner_i", plus_owner_i)
        object.__setattr__(obj, "plus_owner_j", plus_owner_j)
        object.__setattr__(obj, "plus_owner_k", plus_owner_k)
        object.__setattr__(obj, "use_reconstructed_flux", use_reconstructed_flux)
        object.__setattr__(obj, "max_rows", int(max_rows))
        return obj


@_pytree_base
@dataclass(frozen=True)
class LocalControlVolumeFluxStencil3D:
    """Local control-volume flux payload consumed by conservative divergence."""

    regular_flux: FaceFluxStencil3D
    regular_face_geometry: RegularFaceGeometry3D | LocalRegularFaceGeometry3D
    cell_volume: CellVolumeGeometry3D | LocalCellVolumeGeometry3D
    cut_wall_geometry: "CutWallGeometry3D | LocalCutWallGeometry3D | None" = None
    cut_wall_flux: jnp.ndarray | None = None
    regular_face_contribution_rows: LocalRegularFaceContributionRows3D | None = None
    regular_face_contribution_flux: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.regular_flux, FaceFluxStencil3D):
            raise TypeError("LocalControlVolumeFluxStencil3D.regular_flux must be a FaceFluxStencil3D")
        if not isinstance(
            self.regular_face_geometry,
            (RegularFaceGeometry3D, LocalRegularFaceGeometry3D),
        ):
            raise TypeError(
                "LocalControlVolumeFluxStencil3D.regular_face_geometry must be a "
                "RegularFaceGeometry3D or LocalRegularFaceGeometry3D"
            )
        if not isinstance(
            self.cell_volume,
            (CellVolumeGeometry3D, LocalCellVolumeGeometry3D),
        ):
            raise TypeError(
                "LocalControlVolumeFluxStencil3D.cell_volume must be a "
                "CellVolumeGeometry3D or LocalCellVolumeGeometry3D"
            )
        cell_shape = self.cell_volume.shape
        if self.regular_flux.shape != cell_shape:
            raise ValueError(
                f"regular_flux.shape must match cell_volume.shape, got {self.regular_flux.shape} and {cell_shape}"
            )
        regular_cell_shape = (
            self.regular_face_geometry.local_owned_shape
            if isinstance(self.regular_face_geometry, LocalRegularFaceGeometry3D)
            else self.regular_face_geometry.shape
        )
        if regular_cell_shape != cell_shape:
            raise ValueError(
                "regular_face_geometry cell shape must match cell_volume.shape, "
                f"got {regular_cell_shape} and {cell_shape}"
            )
        if (
            self.regular_face_geometry.x_area.shape != self.regular_flux.x.shape
            or self.regular_face_geometry.y_area.shape != self.regular_flux.y.shape
            or self.regular_face_geometry.z_area.shape != self.regular_flux.z.shape
        ):
            raise ValueError(
                "regular_face_geometry face arrays must match regular_flux face shapes"
            )

        if self.regular_face_contribution_rows is None:
            object.__setattr__(
                self,
                "regular_face_contribution_rows",
                LocalRegularFaceContributionRows3D.empty(0),
            )
        elif not isinstance(
            self.regular_face_contribution_rows,
            LocalRegularFaceContributionRows3D,
        ):
            raise TypeError(
                "LocalControlVolumeFluxStencil3D.regular_face_contribution_rows "
                "must be a LocalRegularFaceContributionRows3D or None"
            )
        if self.regular_face_contribution_flux is not None:
            row_flux = jnp.asarray(self.regular_face_contribution_flux, dtype=jnp.float64)
            expected = (int(self.regular_face_contribution_rows.max_rows),)
            if row_flux.shape != expected:
                raise ValueError(
                    "LocalControlVolumeFluxStencil3D.regular_face_contribution_flux "
                    f"must have shape {expected}, got {row_flux.shape}"
                )
            row_flux = jnp.where(
                self.regular_face_contribution_rows.active,
                row_flux,
                0.0,
            )
            object.__setattr__(self, "regular_face_contribution_flux", row_flux)

        if self.cut_wall_geometry is None:
            if self.cut_wall_flux is not None:
                raise ValueError("cut_wall_flux must be None when cut_wall_geometry is None")
            object.__setattr__(self, "cut_wall_flux", None)
            return

        if not isinstance(
            self.cut_wall_geometry,
            (CutWallGeometry3D, LocalCutWallGeometry3D),
        ):
            raise TypeError(
                "LocalControlVolumeFluxStencil3D.cut_wall_geometry must be a "
                "CutWallGeometry3D or LocalCutWallGeometry3D"
            )
        if self.cut_wall_flux is None:
            cut_wall_flux = jnp.zeros((self.cut_wall_geometry.n_wall_faces,), dtype=jnp.float64)
        else:
            cut_wall_flux = jnp.asarray(self.cut_wall_flux, dtype=jnp.float64)
            expected = (self.cut_wall_geometry.n_wall_faces,)
            if cut_wall_flux.shape != expected:
                raise ValueError(
                    f"LocalControlVolumeFluxStencil3D.cut_wall_flux must have shape {expected}, got {cut_wall_flux.shape}"
                )

        cut_wall_active = getattr(self.cut_wall_geometry, "active", None)
        if cut_wall_active is not None:
            cut_wall_active = jnp.asarray(cut_wall_active, dtype=bool)
            if cut_wall_active.shape != cut_wall_flux.shape:
                raise ValueError(
                    "LocalControlVolumeFluxStencil3D.cut_wall_geometry.active must match cut_wall_flux shape"
                )
            cut_wall_flux = jnp.where(cut_wall_active, cut_wall_flux, 0.0)
        object.__setattr__(self, "cut_wall_flux", cut_wall_flux)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.regular_flux.shape

    def tree_flatten(self):
        return (
            (
                self.regular_flux,
                self.regular_face_geometry,
                self.cell_volume,
                self.cut_wall_geometry,
                self.cut_wall_flux,
                self.regular_face_contribution_rows,
                self.regular_face_contribution_flux,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)

def dataclass_replace_1d(instance: LocalStencil1D, **updates: object) -> LocalStencil1D:
    """Small ``dataclasses.replace`` helper that keeps this module self-contained."""

    weight_names = (
        "derivative_minus_weight",
        "derivative_center_weight",
        "derivative_plus_weight",
    )
    distance_changed = "dx_min" in updates or "dx_plus" in updates
    any_weight_updated = any(name in updates for name in weight_names)
    if distance_changed and not any_weight_updated:
        derivative_minus_weight = None
        derivative_center_weight = None
        derivative_plus_weight = None
    else:
        derivative_minus_weight = updates.get(
            "derivative_minus_weight",
            instance.derivative_minus_weight,
        )
        derivative_center_weight = updates.get(
            "derivative_center_weight",
            instance.derivative_center_weight,
        )
        derivative_plus_weight = updates.get(
            "derivative_plus_weight",
            instance.derivative_plus_weight,
        )

    return LocalStencil1D(
        center=updates.get("center", instance.center),
        minus=updates.get("minus", instance.minus),
        plus=updates.get("plus", instance.plus),
        dx_min=updates.get("dx_min", instance.dx_min),
        dx_plus=updates.get("dx_plus", instance.dx_plus),
        derivative_minus_weight=derivative_minus_weight,
        derivative_center_weight=derivative_center_weight,
        derivative_plus_weight=derivative_plus_weight,
    )


def dataclass_replace_3d(instance: LocalStencil3D, **updates: object) -> LocalStencil3D:
    """Small ``dataclasses.replace`` helper that keeps this module self-contained."""

    return LocalStencil3D(
        x=updates.get("x", instance.x),
        y=updates.get("y", instance.y),
        z=updates.get("z", instance.z),
    )


def dataclass_replace_conservative(
    instance: ConservativeStencil3D, **updates: object
) -> ConservativeStencil3D:
    """Small ``dataclasses.replace`` helper for conservative stencil payloads."""

    return ConservativeStencil3D(
        x=updates.get("x", instance.x),
        y=updates.get("y", instance.y),
        z=updates.get("z", instance.z),
        face_grad=updates.get("face_grad", instance.face_grad),
    )


__all__ = [
    "BC_DIRICHLET",
    "BC_NEUMANN",
    "BC_NONE",
    "BC_NORMALFLUX",
    "BC_NOFLUX",
    "BoundaryConditionBuilder",
    "BoundaryFaceBC3D",
    "CellVolumeGeometry3D",
    "CoordinateFaceValueReconstructor3D",
    "CoordinateNormalDerivativeConstructor3D",
    "FaceGradientStencil3D",
    "ConservativeStencil3D",
    "CutWallBC3D",
    "CutWallGeometry3D",
    "CutWallNormalDerivativeConstructor3D",
    "CutWallValueReconstructor3D",
    "FaceFluxStencil3D",
    "LocalBoundaryConditionBuilder",
    "LocalBoundaryData3D",
    "LocalBoundaryFaceBC3D",
    "LocalBoundaryPreparation3D",
    "LocalBoundaryRemoteDependencyTable",
    "LocalCoordinateFaceValueReconstructor3D",
    "LocalCoordinateNormalDerivativeConstructor3D",
    "LocalCoordinateSideValues1D",
    "LocalCoordinateSideValues3D",
    "LocalControlVolumeFluxStencil3D",
    "LocalRegularBoundaryMomentClosure3D",
    "LocalCutWallBC3D",
    "LocalCutWallGeometry3D",
    "LocalCutWallNormalDerivativeConstructor3D",
    "LocalCutWallValueReconstructor3D",
    "LocalCellGradient3D",
    "LocalStencil1D",
    "LocalStencil3D",
    "RegularFaceGeometry3D",
]
