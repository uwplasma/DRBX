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
    LocalRegularFaceGeometry3D,
    LocalCellVolumeGeometry3D,
    RegularFaceGeometry3D,
    CellVolumeGeometry3D,
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
class LocalBoundaryConditionBuilder(_DataclassPyTreeMixin):
    """Build all field boundary payloads from a complete pre-BC state."""

    build_fn: Callable[
        [
            FciModelState,
            LocalFciGeometry3D,
            LocalDomain3D,
            LocalCutWallGeometry3D | None,
        ],
        LocalBoundaryData3D,
    ]

    def __call__(
        self,
        state_halo_pre_bc: FciModelState,
        geometry: LocalFciGeometry3D,
        domain: LocalDomain3D,
        cut_wall_geometry: LocalCutWallGeometry3D | None,
    ) -> LocalBoundaryData3D:
        if not isinstance(state_halo_pre_bc, FciModelState):
            raise TypeError(
                "LocalBoundaryConditionBuilder requires an FciModelState "
                "with physical ghost cells not yet filled"
            )
        state_halo_pre_bc.assert_field_shape(domain.layout.cell_halo_shape)
        result = self.build_fn(
            state_halo_pre_bc,
            geometry,
            domain,
            cut_wall_geometry,
        )
        if not isinstance(result, LocalBoundaryData3D):
            raise TypeError(
                "LocalBoundaryConditionBuilder.build_fn must return "
                "LocalBoundaryData3D"
            )
        if result.face_bc is not None:
            assert_matching_field_names(state_halo_pre_bc, result.face_bc)
        if result.cut_wall_bc is not None:
            assert_matching_field_names(state_halo_pre_bc, result.cut_wall_bc)
        return result

    def tree_flatten(self):
        return (), self.build_fn

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data)



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
    """Padded cut-wall geometry owned by one local shard."""

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

    @property
    def n_wall_faces(self) -> int:
        return int(self.max_wall_faces)

    @classmethod
    def empty(cls, max_wall_faces: int) -> "LocalCutWallGeometry3D":
        max_wall_faces = int(max_wall_faces)
        zeros3 = (max_wall_faces, 3)
        zeros33 = (max_wall_faces, 3, 3)
        return cls(
            owner_i=jnp.zeros((max_wall_faces,), dtype=jnp.int32),
            owner_j=jnp.zeros((max_wall_faces,), dtype=jnp.int32),
            owner_k=jnp.zeros((max_wall_faces,), dtype=jnp.int32),
            center=jnp.zeros(zeros3, dtype=jnp.float64),
            normal_contra=jnp.zeros(zeros3, dtype=jnp.float64),
            area_covector=jnp.zeros(zeros3, dtype=jnp.float64),
            distance=jnp.zeros((max_wall_faces,), dtype=jnp.float64),
            J=jnp.zeros((max_wall_faces,), dtype=jnp.float64),
            g_contra=jnp.zeros(zeros33, dtype=jnp.float64),
            g_cov=jnp.zeros(zeros33, dtype=jnp.float64),
            B_contra=jnp.zeros(zeros3, dtype=jnp.float64),
            Bmag=jnp.zeros((max_wall_faces,), dtype=jnp.float64),
            sign=jnp.zeros((max_wall_faces,), dtype=jnp.float64),
            active=jnp.zeros((max_wall_faces,), dtype=bool),
            max_wall_faces=max_wall_faces,
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
                self.active,
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
        ) = children
        return cls(
            owner_i=owner_i,
            owner_j=owner_j,
            owner_k=owner_k,
            center=center,
            normal_contra=normal_contra,
            area_covector=area_covector,
            distance=distance,
            J=J,
            g_contra=g_contra,
            g_cov=g_cov,
            B_contra=B_contra,
            Bmag=Bmag,
            sign=sign,
            active=active,
            max_wall_faces=max_wall_faces,
        )


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
class LocalControlVolumeFluxStencil3D:
    """Local control-volume flux payload consumed by conservative divergence."""

    regular_flux: FaceFluxStencil3D
    regular_face_geometry: "LocalRegularFaceGeometry3D | RegularFaceGeometry3D"
    cell_volume: "LocalCellVolumeGeometry3D | CellVolumeGeometry3D"
    cut_wall_geometry: "LocalCutWallGeometry3D | CutWallGeometry3D | None" = None
    cut_wall_flux: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.regular_flux, FaceFluxStencil3D):
            raise TypeError("LocalControlVolumeFluxStencil3D.regular_flux must be a FaceFluxStencil3D")
        if not isinstance(self.regular_face_geometry, (LocalRegularFaceGeometry3D, RegularFaceGeometry3D)):
            raise TypeError(
                "LocalControlVolumeFluxStencil3D.regular_face_geometry must be a "
                "LocalRegularFaceGeometry3D or RegularFaceGeometry3D"
            )
        if not isinstance(self.cell_volume, (LocalCellVolumeGeometry3D, CellVolumeGeometry3D)):
            raise TypeError(
                "LocalControlVolumeFluxStencil3D.cell_volume must be a "
                "LocalCellVolumeGeometry3D or CellVolumeGeometry3D"
            )
        cell_shape = self.cell_volume.shape
        if self.regular_flux.shape != cell_shape:
            raise ValueError(
                f"regular_flux.shape must match cell_volume.shape, got {self.regular_flux.shape} and {cell_shape}"
            )
        face_geometry_cell_shape = (
            self.regular_face_geometry.local_owned_shape
            if isinstance(self.regular_face_geometry, LocalRegularFaceGeometry3D)
            else self.regular_face_geometry.shape
        )
        if face_geometry_cell_shape != cell_shape:
            raise ValueError(
                "regular_face_geometry.local_owned_shape must match cell_volume.shape, "
                f"got {face_geometry_cell_shape} and {cell_shape}"
            )
        if (
            self.regular_face_geometry.x_area.shape != self.regular_flux.x.shape
            or self.regular_face_geometry.y_area.shape != self.regular_flux.y.shape
            or self.regular_face_geometry.z_area.shape != self.regular_flux.z.shape
        ):
            raise ValueError(
                "regular_face_geometry face arrays must match regular_flux face shapes"
            )

        if self.cut_wall_geometry is None:
            if self.cut_wall_flux is not None:
                raise ValueError("cut_wall_flux must be None when cut_wall_geometry is None")
            object.__setattr__(self, "cut_wall_flux", None)
            return

        if not isinstance(self.cut_wall_geometry, (LocalCutWallGeometry3D, CutWallGeometry3D)):
            raise TypeError(
                "LocalControlVolumeFluxStencil3D.cut_wall_geometry must be a "
                "LocalCutWallGeometry3D or CutWallGeometry3D"
            )
        if self.cut_wall_flux is None:
            Wmax = getattr(self.cut_wall_geometry, "max_wall_faces", self.cut_wall_geometry.n_wall_faces)
            cut_wall_flux = jnp.zeros((Wmax,), dtype=jnp.float64)
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
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)

def dataclass_replace_1d(instance: LocalStencil1D, **updates: object) -> LocalStencil1D:
    """Small ``dataclasses.replace`` helper that keeps this module self-contained."""

    return LocalStencil1D(
        center=updates.get("center", instance.center),
        minus=updates.get("minus", instance.minus),
        plus=updates.get("plus", instance.plus),
        dx_min=updates.get("dx_min", instance.dx_min),
        dx_plus=updates.get("dx_plus", instance.dx_plus),
        derivative_minus_weight=updates.get(
            "derivative_minus_weight",
            instance.derivative_minus_weight,
        ),
        derivative_center_weight=updates.get(
            "derivative_center_weight",
            instance.derivative_center_weight,
        ),
        derivative_plus_weight=updates.get(
            "derivative_plus_weight",
            instance.derivative_plus_weight,
        ),
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
    "LocalCoordinateFaceValueReconstructor3D",
    "LocalCoordinateNormalDerivativeConstructor3D",
    "LocalCoordinateSideValues1D",
    "LocalCoordinateSideValues3D",
    "LocalControlVolumeFluxStencil3D",
    "LocalCutWallBC3D",
    "LocalCutWallGeometry3D",
    "LocalCutWallNormalDerivativeConstructor3D",
    "LocalCutWallValueReconstructor3D",
    "LocalStencil1D",
    "LocalStencil3D",
    "RegularFaceGeometry3D",
]
