from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Callable
import jax
import jax.numpy as jnp
from jax import lax


_pytree_base = jax.tree_util.register_pytree_node_class

def _normalize_same_shape_fields(instance, field_names: tuple[str, ...], *, expected_shape: tuple[int, ...], label: str) -> None:
    for name in field_names:
        value = jnp.asarray(getattr(instance, name), dtype=jnp.float64)
        if value.shape != expected_shape:
            raise ValueError(f"{label}.{name} must have shape {expected_shape}, got {value.shape}")
        object.__setattr__(instance, name, value)

class _DataclassPyTreeMixin:
    """Generic PyTree support for frozen dataclasses.
    All dataclass fields with init=True are treated as dynamic PyTree children.
    Computed fields with init=False are rebuilt in __post_init__.
    """
    def tree_flatten(self):
        children = tuple(getattr(self, f.name) for f in fields(self) if f.init)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        init_names = [f.name for f in fields(cls) if f.init]
        return cls(**dict(zip(init_names, children)))

def _as_float_array(value, name: str):
    return jnp.asarray(value, dtype=jnp.float64)

def _require_shape(value, expected_shape: tuple[int, ...], name: str):
    arr = jnp.asarray(value)
    if arr.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
    return arr

def _require_float_shape(value, expected_shape: tuple[int, ...], name: str):
    arr = jnp.asarray(value, dtype=jnp.float64)
    if arr.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
    return arr


def _validate_coordinate_stencil_dependency_rows(
    *,
    target_flat: jnp.ndarray,
    axis: jnp.ndarray,
    side: jnp.ndarray,
    distance: jnp.ndarray,
    active: jnp.ndarray,
    label: str,
) -> None:
    valid_axis = (~active) | ((axis >= 0) & (axis <= 2))
    valid_side = (~active) | ((side >= 0) & (side <= 1))
    valid_distance = (~active) | (distance > 0.0)
    try:
        all_valid_axis = bool(jnp.all(valid_axis))
        all_valid_side = bool(jnp.all(valid_side))
        all_valid_distance = bool(jnp.all(valid_distance))
    except jax.errors.TracerBoolConversionError:
        return
    if not all_valid_axis:
        raise ValueError(f"{label}.axis must be 0, 1, or 2 for active rows")
    if not all_valid_side:
        raise ValueError(f"{label}.side must be 0 or 1 for active rows")
    if not all_valid_distance:
        raise ValueError(f"{label}.distance must be positive for active rows")

    seen: set[tuple[int, int, int]] = set()
    for row in range(int(target_flat.size)):
        if not bool(active[row]):
            continue
        key = (int(target_flat[row]), int(axis[row]), int(side[row]))
        if key in seen:
            raise ValueError(
                f"{label} must contain at most one active row per "
                "(target_flat, axis, side)"
            )
        seen.add(key)


def _coordinate_stencil_dependency_keys(
    *,
    target_flat: jnp.ndarray,
    axis: jnp.ndarray,
    side: jnp.ndarray,
    active: jnp.ndarray,
) -> set[tuple[int, int, int]]:
    keys: set[tuple[int, int, int]] = set()
    for row in range(int(target_flat.size)):
        if bool(active[row]):
            keys.add((int(target_flat[row]), int(axis[row]), int(side[row])))
    return keys


def _normalize_periodic_axes(
    periodic_axes: tuple[bool | None, bool | None, bool | None] | None,
    *,
    default: tuple[bool, bool, bool] = (False, True, True),
) -> tuple[bool, bool, bool]:
    if periodic_axes is None:
        periodic_axes = default
    if len(periodic_axes) != 3:
        raise ValueError(f"periodic_axes must have length 3, got {periodic_axes}")
    return tuple(False if axis is None else bool(axis) for axis in periodic_axes)

def _metric_from_components(
    g11: jnp.ndarray,
    g22: jnp.ndarray,
    g33: jnp.ndarray,
    g12: jnp.ndarray,
    g13: jnp.ndarray,
    g23: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.stack(
        [
            jnp.stack([g11, g12, g13], axis=-1),
            jnp.stack([g12, g22, g23], axis=-1),
            jnp.stack([g13, g23, g33], axis=-1),
        ],
        axis=-2,
    )

def _bmag_from_contravariant_components(
    B_contra: jnp.ndarray,
    g_cov: jnp.ndarray,
) -> jnp.ndarray:
    bmag_sq = jnp.einsum("...i,...ij,...j->...", B_contra, g_cov, B_contra)
    return jnp.sqrt(jnp.maximum(bmag_sq, 0.0))

def logical_grid_from_axis_vectors(
    x_axis: jnp.ndarray,
    y_axis: jnp.ndarray,
    z_axis: jnp.ndarray,
) -> jnp.ndarray:
    x = jnp.asarray(x_axis, dtype=jnp.float64)
    y = jnp.asarray(y_axis, dtype=jnp.float64)
    z = jnp.asarray(z_axis, dtype=jnp.float64)
    xx = jnp.broadcast_to(x[:, None, None], (x.size, y.size, z.size))
    yy = jnp.broadcast_to(y[None, :, None], (x.size, y.size, z.size))
    zz = jnp.broadcast_to(z[None, None, :], (x.size, y.size, z.size))
    return jnp.stack((xx, yy, zz), axis=-1)


@_pytree_base
@dataclass(frozen=True)
class Grid1D(_DataclassPyTreeMixin):
    centers: jnp.ndarray  # (n,)
    faces: jnp.ndarray    # (n + 1,)
    def __post_init__(self) -> None:
        centers = _as_float_array(self.centers, "centers")
        faces = _as_float_array(self.faces, "faces")
        if centers.ndim != 1:
            raise ValueError(f"centers must be one-dimensional, got {centers.shape}")
        if faces.ndim != 1:
            raise ValueError(f"faces must be one-dimensional, got {faces.shape}")
        if faces.size != centers.size + 1:
            raise ValueError(
                f"faces must have length centers.size + 1; got centers={centers.shape}, faces={faces.shape}"
            )
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "faces", faces)

    @classmethod
    def from_centers(cls, centers: jnp.ndarray) -> "Grid1D":
        centers = jnp.asarray(centers, dtype=jnp.float64)
        if centers.ndim != 1:
            raise ValueError(f"centers must be one-dimensional, got {centers.shape}")
        if centers.size == 0:
            raise ValueError("centers must contain at least one point")
        if centers.size == 1:
            spacing = jnp.asarray(1.0, dtype=jnp.float64)
            faces = jnp.array([centers[0] - 0.5 * spacing, centers[0] + 0.5 * spacing], dtype=jnp.float64)
        else:
            faces = jnp.empty(centers.size + 1, dtype=jnp.float64)
            faces = faces.at[1:-1].set(0.5 * (centers[:-1] + centers[1:]))
            faces = faces.at[0].set(centers[0] - 0.5 * (centers[1] - centers[0]))
            faces = faces.at[-1].set(centers[-1] + 0.5 * (centers[-1] - centers[-2]))
        return cls(centers=centers, faces=faces)

    @property
    def n(self) -> int:
        return int(self.centers.size)

    @property
    def widths(self) -> jnp.ndarray:
        return self.faces[1:] - self.faces[:-1]

    @property
    def center_deltas(self) -> jnp.ndarray:
        return self.centers[1:] - self.centers[:-1]

    @property
    def lower_center_to_face(self):
        return self.centers[0] - self.faces[0]

    @property
    def upper_center_to_face(self):
        return self.faces[-1] - self.centers[-1]


@dataclass(frozen=True)
class HaloLayout3D:
    """Shared halo metadata for shard-local 3D geometry."""

    owned_shape: tuple[int, int, int]
    halo_width: int

    def __post_init__(self) -> None:
        owned_shape = tuple(int(v) for v in self.owned_shape)
        if len(owned_shape) != 3:
            raise ValueError(f"HaloLayout3D.owned_shape must have length 3, got {owned_shape}")
        if any(size <= 0 for size in owned_shape):
            raise ValueError(f"HaloLayout3D.owned_shape must contain positive integers, got {owned_shape}")
        halo_width = int(self.halo_width)
        if halo_width < 0:
            raise ValueError(f"HaloLayout3D.halo_width must be non-negative, got {halo_width}")
        object.__setattr__(self, "owned_shape", owned_shape)
        object.__setattr__(self, "halo_width", halo_width)

    @property
    def cell_halo_shape(self) -> tuple[int, int, int]:
        h = self.halo_width
        nx, ny, nz = self.owned_shape
        return nx + 2 * h, ny + 2 * h, nz + 2 * h

    @property
    def owned_slices_cell(self) -> tuple[slice, slice, slice]:
        h = self.halo_width
        nx, ny, nz = self.owned_shape
        return (
            slice(h, h + nx),
            slice(h, h + ny),
            slice(h, h + nz),
        )

    def face_halo_shape(self, axis: int) -> tuple[int, int, int]:
        axis = int(axis)
        if axis < 0 or axis > 2:
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        shape = list(self.cell_halo_shape)
        shape[axis] += 1
        return tuple(shape)

    def face_control_shape(self, axis: int) -> tuple[int, int, int]:
        axis = int(axis)
        if axis < 0 or axis > 2:
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        nx, ny, nz = self.owned_shape
        shape = [nx, ny, nz]
        shape[axis] += 1
        return tuple(shape)

    def face_control_slices(self, axis: int) -> tuple[slice, slice, slice]:
        axis = int(axis)
        if axis < 0 or axis > 2:
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        h = self.halo_width
        nx, ny, nz = self.owned_shape
        extents = [nx, ny, nz]
        extents[axis] += 1
        return tuple(slice(h, h + n) for n in extents)

    def location_halo_shape(self, location: str) -> tuple[int, int, int]:
        if location == "cell":
            return self.cell_halo_shape
        if location == "x_face":
            return self.face_halo_shape(0)
        if location == "y_face":
            return self.face_halo_shape(1)
        if location == "z_face":
            return self.face_halo_shape(2)
        raise ValueError(
            'location must be one of "cell", "x_face", "y_face", or "z_face", '
            f"got {location!r}"
        )

    def location_owned_slices(self, location: str) -> tuple[slice, slice, slice]:
        if location == "cell":
            return self.owned_slices_cell
        if location == "x_face":
            return self.face_control_slices(0)
        if location == "y_face":
            return self.face_control_slices(1)
        if location == "z_face":
            return self.face_control_slices(2)
        raise ValueError(
            'location must be one of "cell", "x_face", "y_face", or "z_face", '
            f"got {location!r}"
        )

    def location_owned_shape(self, location: str) -> tuple[int, int, int]:
        if location == "cell":
            return self.owned_shape
        if location == "x_face":
            return self.face_control_shape(0)
        if location == "y_face":
            return self.face_control_shape(1)
        if location == "z_face":
            return self.face_control_shape(2)
        raise ValueError(
            'location must be one of "cell", "x_face", "y_face", or "z_face", '
            f"got {location!r}"
        )


@_pytree_base
@dataclass(frozen=True)
class LocalGrid1D:
    """Shard-local 1D grid with owned and halo coordinate storage."""

    layout: HaloLayout3D
    axis: int
    centers_halo: jnp.ndarray
    faces_halo: jnp.ndarray
    owned_start_global: int
    owned_stop_global: int

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        axis = int(self.axis)
        if axis < 0 or axis > 2:
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        centers_halo = _as_float_array(self.centers_halo, "centers_halo")
        faces_halo = _as_float_array(self.faces_halo, "faces_halo")
        if centers_halo.ndim != 1:
            raise ValueError(f"centers_halo must be one-dimensional, got {centers_halo.shape}")
        if faces_halo.ndim != 1:
            raise ValueError(f"faces_halo must be one-dimensional, got {faces_halo.shape}")

        owned_start_global = int(self.owned_start_global)
        owned_stop_global = int(self.owned_stop_global)
        halo_width = self.layout.halo_width
        if owned_start_global < 0:
            raise ValueError(f"owned_start_global must be non-negative, got {owned_start_global}")
        if owned_stop_global < owned_start_global:
            raise ValueError(
                "owned_stop_global must be greater than or equal to owned_start_global, "
                f"got start={owned_start_global}, stop={owned_stop_global}"
            )
        owned_size = self.layout.owned_shape[axis]
        if owned_stop_global - owned_start_global != owned_size:
            raise ValueError(
                "owned_start_global and owned_stop_global must span the owned size from the layout; "
                f"axis={axis}, expected size={owned_size}, got start={owned_start_global}, stop={owned_stop_global}"
            )
        expected_center_size = owned_size + 2 * halo_width
        expected_face_size = expected_center_size + 1
        if centers_halo.size != expected_center_size:
            raise ValueError(
                "centers_halo must contain owned cells plus both halo layers; "
                f"expected {expected_center_size}, got {centers_halo.size}"
            )
        if faces_halo.size != expected_face_size:
            raise ValueError(
                "faces_halo must contain one more entry than centers_halo; "
                f"expected {expected_face_size}, got {faces_halo.size}"
            )

        object.__setattr__(self, "centers_halo", centers_halo)
        object.__setattr__(self, "faces_halo", faces_halo)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "owned_start_global", owned_start_global)
        object.__setattr__(self, "owned_stop_global", owned_stop_global)

    @property
    def centers(self) -> jnp.ndarray:
        return self.centers_halo

    @property
    def faces(self) -> jnp.ndarray:
        return self.faces_halo

    @property
    def n(self) -> int:
        return self.n_local

    @property
    def n_owned(self) -> int:
        return self.layout.owned_shape[self.axis]

    @property
    def n_halo(self) -> int:
        return self.layout.halo_width

    @property
    def n_local(self) -> int:
        return int(self.centers_halo.size)

    @property
    def shape(self) -> tuple[int]:
        return (self.n_local,)

    @property
    def halo_start_global(self) -> int:
        return self.owned_start_global - self.layout.halo_width

    @property
    def halo_stop_global(self) -> int:
        return self.owned_stop_global + self.layout.halo_width

    @property
    def owned_center_slice(self) -> slice:
        h = self.layout.halo_width
        return slice(h, h + self.n_owned)

    @property
    def owned_face_slice(self) -> slice:
        h = self.layout.halo_width
        return slice(h, h + self.n_owned + 1)

    @property
    def centers_owned(self) -> jnp.ndarray:
        return self.centers_halo[self.owned_center_slice]

    @property
    def faces_owned(self) -> jnp.ndarray:
        return self.faces_halo[self.owned_face_slice]

    @property
    def widths(self) -> jnp.ndarray:
        return self.faces_halo[1:] - self.faces_halo[:-1]

    @property
    def center_deltas(self) -> jnp.ndarray:
        return self.centers_halo[1:] - self.centers_halo[:-1]

    @property
    def lower_center_to_face(self):
        return self.centers_halo[0] - self.faces_halo[0]

    @property
    def upper_center_to_face(self):
        return self.faces_halo[-1] - self.centers_halo[-1]

    def tree_flatten(self):
        return (
            (self.centers_halo, self.faces_halo),
            (self.layout, self.axis, self.owned_start_global, self.owned_stop_global),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        layout, axis, owned_start_global, owned_stop_global = aux_data
        centers_halo, faces_halo = children
        return cls(
            layout=layout,
            axis=axis,
            centers_halo=centers_halo,
            faces_halo=faces_halo,
            owned_start_global=owned_start_global,
            owned_stop_global=owned_stop_global,
        )

@_pytree_base
@dataclass(frozen=True)
class CellCenteredGrid3D(_DataclassPyTreeMixin):
    x: Grid1D
    y: Grid1D
    z: Grid1D

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.x.n, self.y.n, self.z.n

    @property
    def x_centers(self) -> jnp.ndarray:
        return self.x.centers

    @property
    def y_centers(self) -> jnp.ndarray:
        return self.y.centers

    @property
    def z_centers(self) -> jnp.ndarray:
        return self.z.centers

    @property
    def x_faces(self) -> jnp.ndarray:
        return self.x.faces

    @property
    def y_faces(self) -> jnp.ndarray:
        return self.y.faces

    @property
    def z_faces(self) -> jnp.ndarray:
        return self.z.faces

    @property
    def logical_axis_vectors(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.x.centers, self.y.centers, self.z.centers

    @property
    def logical_face_vectors(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.x.faces, self.y.faces, self.z.faces


@_pytree_base
@dataclass(frozen=True)
class LocalCellCenteredGrid3D(_DataclassPyTreeMixin):
    """Shard-local cell-centered grid with halo-aware 1D axes."""

    layout: HaloLayout3D
    x: LocalGrid1D
    y: LocalGrid1D
    z: LocalGrid1D

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        if not isinstance(self.x, LocalGrid1D):
            raise TypeError("x must be a LocalGrid1D instance")
        if not isinstance(self.y, LocalGrid1D):
            raise TypeError("y must be a LocalGrid1D instance")
        if not isinstance(self.z, LocalGrid1D):
            raise TypeError("z must be a LocalGrid1D instance")
        if self.x.layout != self.layout or self.y.layout != self.layout or self.z.layout != self.layout:
            raise ValueError("LocalCellCenteredGrid3D axes must share the same HaloLayout3D")
        if self.x.axis != 0 or self.y.axis != 1 or self.z.axis != 2:
            raise ValueError("LocalCellCenteredGrid3D axes must be ordered as x=0, y=1, z=2")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.layout.cell_halo_shape

    @property
    def owned_shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def halo_shape(self) -> tuple[int, int, int]:
        return self.shape

    @property
    def local_owned_shape(self) -> tuple[int, int, int]:
        return self.owned_shape

    @property
    def owned_slices_in_halo(self) -> tuple[slice, slice, slice]:
        return self.layout.owned_slices_cell

    @property
    def local_halo_shape(self) -> tuple[int, int, int]:
        return self.shape

    @property
    def x_centers(self) -> jnp.ndarray:
        return self.x.centers

    @property
    def y_centers(self) -> jnp.ndarray:
        return self.y.centers

    @property
    def z_centers(self) -> jnp.ndarray:
        return self.z.centers

    @property
    def x_faces(self) -> jnp.ndarray:
        return self.x.faces

    @property
    def y_faces(self) -> jnp.ndarray:
        return self.y.faces

    @property
    def z_faces(self) -> jnp.ndarray:
        return self.z.faces

    @property
    def logical_axis_vectors(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.x.centers, self.y.centers, self.z.centers

    @property
    def logical_face_vectors(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.x.faces, self.y.faces, self.z.faces

    @property
    def x_centers_owned(self) -> jnp.ndarray:
        return self.x.centers_owned

    @property
    def y_centers_owned(self) -> jnp.ndarray:
        return self.y.centers_owned

    @property
    def z_centers_owned(self) -> jnp.ndarray:
        return self.z.centers_owned

@_pytree_base
@dataclass(frozen=True)
class FciMaps3D(_DataclassPyTreeMixin):
    # Fractional interpolation indices into cell-centered field arrays.
    # These are meaningful for non-boundary traces. For boundary traces,
    # use the boundary mask and endpoint coordinates instead.
    forward_x: jnp.ndarray
    forward_y: jnp.ndarray
    backward_x: jnp.ndarray
    backward_y: jnp.ndarray

    # Logical endpoint coordinates of the trace.
    # If boundary=False: endpoint is on the target toroidal plane.
    # If boundary=True: endpoint is the estimated physical boundary hit point.
    forward_endpoint_x: jnp.ndarray
    forward_endpoint_y: jnp.ndarray
    forward_endpoint_z: jnp.ndarray
    backward_endpoint_x: jnp.ndarray
    backward_endpoint_y: jnp.ndarray
    backward_endpoint_z: jnp.ndarray

    # Physical arclengths from cell center to endpoint.
    forward_length: jnp.ndarray
    backward_length: jnp.ndarray

    # True if the trace hit/exited a nonperiodic physical boundary before
    # reaching the target toroidal plane.
    forward_boundary: jnp.ndarray
    backward_boundary: jnp.ndarray

    def __post_init__(self) -> None:
        forward_x = jnp.asarray(self.forward_x, dtype=jnp.float64)
        shape = tuple(int(v) for v in forward_x.shape)

        if len(shape) != 3:
            raise ValueError(f"FciMaps3D fields must have shape (nx, ny, nz), got {shape}")

        object.__setattr__(self, "forward_x", forward_x)

        float_fields = (
            "forward_y",
            "backward_x",
            "backward_y",
            "forward_endpoint_x",
            "forward_endpoint_y",
            "forward_endpoint_z",
            "backward_endpoint_x",
            "backward_endpoint_y",
            "backward_endpoint_z",
            "forward_length",
            "backward_length",
        )
        bool_fields = (
            "forward_boundary",
            "backward_boundary",
        )
        for name in float_fields:
            value = _require_float_shape(getattr(self, name), shape, f"FciMaps3D.{name}")
            object.__setattr__(self, name, value)
        for name in bool_fields:
            value = jnp.asarray(getattr(self, name), dtype=bool)
            if value.shape != shape:
                raise ValueError(f"FciMaps3D.{name} must have shape {shape}, got {value.shape}")
            object.__setattr__(self, name, value)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.forward_x.shape)


# Dependency kinds are shared by local and remote FCI dependency metadata.
FCI_DEP_INVALID = 0
FCI_DEP_FIELD_INTERIOR = 1
FCI_DEP_PHYSICAL_BOUNDARY = 2
FCI_DEP_CUT_WALL = 3


@_pytree_base
@dataclass(frozen=True)
class LocalFciLocalDependencyTable(_DataclassPyTreeMixin):
    """Sparse interpolation rows that can be satisfied locally.

    Interior rows read from the local field halo. Boundary and cut-wall rows
    may instead use a prepared value identified by ``value_slot``. The rows
    are padded to a fixed maximum length so the object stays JAX compilation
    friendly. Only the ``active`` rows participate in interpolation.
    """

    target_flat: jnp.ndarray  # (max_entries,)
    source_i: jnp.ndarray  # (max_entries,)
    source_j: jnp.ndarray  # (max_entries,)
    source_k: jnp.ndarray  # (max_entries,)
    weight: jnp.ndarray  # (max_entries,)
    active: jnp.ndarray  # (max_entries,)
    dependency_kind: jnp.ndarray | None = None  # (max_entries,), int32
    value_slot: jnp.ndarray | None = None  # (max_entries,), int32

    def __post_init__(self) -> None:
        target_flat = jnp.asarray(self.target_flat, dtype=jnp.int32)
        shape = tuple(int(v) for v in target_flat.shape)
        if target_flat.ndim != 1:
            raise ValueError(f"LocalFciLocalDependencyTable.target_flat must be 1D, got {target_flat.shape}")
        object.__setattr__(self, "target_flat", target_flat)
        for name in ("source_i", "source_j", "source_k"):
            object.__setattr__(self, name, _require_shape(getattr(self, name), shape, f"LocalFciLocalDependencyTable.{name}"))
        object.__setattr__(self, "weight", _require_float_shape(self.weight, shape, "LocalFciLocalDependencyTable.weight"))
        active = jnp.asarray(self.active, dtype=bool)
        if active.shape != shape:
            raise ValueError(f"LocalFciLocalDependencyTable.active must have shape {shape}, got {active.shape}")
        object.__setattr__(self, "active", active)

        if self.dependency_kind is None:
            dependency_kind = jnp.full(shape, FCI_DEP_FIELD_INTERIOR, dtype=jnp.int32)
        else:
            dependency_kind = jnp.asarray(self.dependency_kind, dtype=jnp.int32)
            if dependency_kind.shape != shape:
                raise ValueError(
                    "LocalFciLocalDependencyTable.dependency_kind must have "
                    f"shape {shape}, got {dependency_kind.shape}"
                )
        object.__setattr__(self, "dependency_kind", dependency_kind)

        if self.value_slot is None:
            value_slot = jnp.zeros(shape, dtype=jnp.int32)
        else:
            value_slot = jnp.asarray(self.value_slot, dtype=jnp.int32)
            if value_slot.shape != shape:
                raise ValueError(
                    "LocalFciLocalDependencyTable.value_slot must have "
                    f"shape {shape}, got {value_slot.shape}"
                )
        object.__setattr__(self, "value_slot", value_slot)

    @property
    def max_entries(self) -> int:
        return int(self.target_flat.size)

@_pytree_base
@dataclass(frozen=True)
class LocalFciRemoteDependencyTable(_DataclassPyTreeMixin):
    """Sparse FCI interpolation rows satisfied by remote value exchange.

    This table has two logical parts:

    1. Interpolation rows, length max_entries
       These are used by LocalFciStencilBuilder.

           endpoint[target_flat[r]] += weight[r] * remote_values[receive_slot[r]]

    2. Request rows, length max_receive_values
       These are used by RemoteFciDependencyExchange.

           remote_values[q] = value requested by request row q

    Consolidation convention
    ------------------------

    The request row index is the receive slot:

        remote_values[q] contains the scalar returned for request row q

    Therefore:

        receive_slot[r] points directly to one request row q.

    This lets multiple interpolation rows reuse the same remote scalar by sharing
    the same receive_slot.

    Request dependency kinds tell the owner/source shard what kind of scalar to
    return:

        FCI_DEP_FIELD_INTERIOR:
            return a field value from the owner shard's local/halo field data.

        FCI_DEP_PHYSICAL_BOUNDARY:
            return a prepared physical-boundary value from the owner shard.

        FCI_DEP_CUT_WALL:
            return a prepared cut-wall value from the owner shard.

    The stencil builder does not interpret request_dependency_kind. It only uses
    target_flat, weight, receive_slot, and active. The exchange object uses the
    request_* arrays.
    """

    # -------------------------------------------------------------------------
    # Interpolation rows on the requesting shard.
    # Shape: (max_entries,)
    # Used by LocalFciStencilBuilder.
    # -------------------------------------------------------------------------

    target_flat: jnp.ndarray
    weight: jnp.ndarray
    receive_slot: jnp.ndarray
    active: jnp.ndarray

    # -------------------------------------------------------------------------
    # Request rows.
    # Shape: (max_receive_values,)
    # Used by RemoteFciDependencyExchange.
    #
    # Consolidation convention:
    #
    #     remote_values[q] corresponds to request row q
    #
    # Therefore active interpolation rows must satisfy:
    #
    #     0 <= receive_slot[r] < max_receive_values
    # -------------------------------------------------------------------------

    request_active: jnp.ndarray
    request_dependency_kind: jnp.ndarray

    request_source_global_i: jnp.ndarray
    request_source_global_j: jnp.ndarray
    request_source_global_k: jnp.ndarray

    request_source_shard_index: jnp.ndarray      # (max_receive_values, 3)
    request_source_shard_linear: jnp.ndarray

    request_source_owner_local_i: jnp.ndarray
    request_source_owner_local_j: jnp.ndarray
    request_source_owner_local_k: jnp.ndarray

    # Used for PHYSICAL_BOUNDARY / CUT_WALL requests.
    # Dummy zero for FIELD_INTERIOR requests.
    request_value_slot: jnp.ndarray

    def __post_init__(self) -> None:
        # ---------------------------------------------------------------------
        # Interpolation-row arrays.
        # ---------------------------------------------------------------------
        target_flat = jnp.asarray(self.target_flat, dtype=jnp.int32)
        row_shape = tuple(int(v) for v in target_flat.shape)

        if target_flat.ndim != 1:
            raise ValueError(
                "LocalFciRemoteDependencyTable.target_flat must be 1D, "
                f"got {target_flat.shape}"
            )

        object.__setattr__(self, "target_flat", target_flat)

        object.__setattr__(
            self,
            "weight",
            _require_float_shape(
                self.weight,
                row_shape,
                "LocalFciRemoteDependencyTable.weight",
            ),
        )

        object.__setattr__(
            self,
            "receive_slot",
            _require_shape(
                self.receive_slot,
                row_shape,
                "LocalFciRemoteDependencyTable.receive_slot",
            ),
        )

        active = jnp.asarray(self.active, dtype=bool)
        if active.shape != row_shape:
            raise ValueError(
                "LocalFciRemoteDependencyTable.active must have shape "
                f"{row_shape}, got {active.shape}"
            )
        object.__setattr__(self, "active", active)

        # ---------------------------------------------------------------------
        # Request-row arrays.
        # ---------------------------------------------------------------------
        request_active = jnp.asarray(self.request_active, dtype=bool)
        request_shape = tuple(int(v) for v in request_active.shape)

        if request_active.ndim != 1:
            raise ValueError(
                "LocalFciRemoteDependencyTable.request_active must be 1D, "
                f"got {request_active.shape}"
            )

        object.__setattr__(self, "request_active", request_active)

        object.__setattr__(
            self,
            "request_dependency_kind",
            _require_shape(
                self.request_dependency_kind,
                request_shape,
                "LocalFciRemoteDependencyTable.request_dependency_kind",
            ),
        )

        for name in (
            "request_source_global_i",
            "request_source_global_j",
            "request_source_global_k",
            "request_source_shard_linear",
            "request_source_owner_local_i",
            "request_source_owner_local_j",
            "request_source_owner_local_k",
            "request_value_slot",
        ):
            object.__setattr__(
                self,
                name,
                _require_shape(
                    getattr(self, name),
                    request_shape,
                    f"LocalFciRemoteDependencyTable.{name}",
                ),
            )

        request_source_shard_index = jnp.asarray(
            self.request_source_shard_index,
            dtype=jnp.int32,
        )

        if (
            request_source_shard_index.ndim != 2
            or request_source_shard_index.shape[1] != 3
        ):
            raise ValueError(
                "LocalFciRemoteDependencyTable.request_source_shard_index "
                "must have shape (max_receive_values, 3), got "
                f"{request_source_shard_index.shape}"
            )

        if int(request_source_shard_index.shape[0]) != request_shape[0]:
            raise ValueError(
                "LocalFciRemoteDependencyTable.request_source_shard_index "
                "must match request_active length; got "
                f"{request_source_shard_index.shape[0]}, "
                f"expected {request_shape[0]}"
            )

        object.__setattr__(
            self,
            "request_source_shard_index",
            request_source_shard_index,
        )

    @property
    def max_entries(self) -> int:
        """Maximum number of interpolation rows."""
        return int(self.target_flat.size)

    @property
    def max_receive_values(self) -> int:
        """Maximum number of requested/received scalar values."""
        return int(self.request_active.size)

    @property
    def has_requests(self) -> bool:
        """Static-size table may still contain no active requests at runtime."""
        return self.max_receive_values > 0



@_pytree_base
@dataclass(frozen=True)
class LocalCoordinateStencilLocalDependencyTable(_DataclassPyTreeMixin):
    """Local cut-wall replacements for coordinate stencil legs.

    Each active row patches exactly one owned target cell, coordinate axis, and
    stencil side. ``value_slot`` indexes an owner-local cut-wall value vector.
    """

    target_flat: jnp.ndarray
    axis: jnp.ndarray
    side: jnp.ndarray
    value_slot: jnp.ndarray
    distance: jnp.ndarray
    active: jnp.ndarray

    def __post_init__(self) -> None:
        target_flat = jnp.asarray(self.target_flat, dtype=jnp.int32)
        shape = tuple(int(v) for v in target_flat.shape)
        if target_flat.ndim != 1:
            raise ValueError(
                "LocalCoordinateStencilLocalDependencyTable.target_flat "
                f"must be 1D, got {target_flat.shape}"
            )
        object.__setattr__(self, "target_flat", target_flat)
        for name in ("axis", "side", "value_slot"):
            object.__setattr__(
                self,
                name,
                _require_shape(
                    getattr(self, name),
                    shape,
                    f"LocalCoordinateStencilLocalDependencyTable.{name}",
                ).astype(jnp.int32),
            )
        object.__setattr__(
            self,
            "distance",
            _require_float_shape(
                self.distance,
                shape,
                "LocalCoordinateStencilLocalDependencyTable.distance",
            ),
        )
        active = jnp.asarray(self.active, dtype=bool)
        if active.shape != shape:
            raise ValueError(
                "LocalCoordinateStencilLocalDependencyTable.active must have "
                f"shape {shape}, got {active.shape}"
            )
        object.__setattr__(self, "active", active)
        _validate_coordinate_stencil_dependency_rows(
            target_flat=target_flat,
            axis=self.axis,
            side=self.side,
            distance=self.distance,
            active=active,
            label="LocalCoordinateStencilLocalDependencyTable",
        )

    @property
    def max_entries(self) -> int:
        return int(self.target_flat.size)

    @classmethod
    def empty(cls) -> "LocalCoordinateStencilLocalDependencyTable":
        return cls(
            target_flat=jnp.zeros((0,), dtype=jnp.int32),
            axis=jnp.zeros((0,), dtype=jnp.int32),
            side=jnp.zeros((0,), dtype=jnp.int32),
            value_slot=jnp.zeros((0,), dtype=jnp.int32),
            distance=jnp.zeros((0,), dtype=jnp.float64),
            active=jnp.zeros((0,), dtype=bool),
        )


@_pytree_base
@dataclass(frozen=True)
class LocalCoordinateStencilRemoteDependencyTable(_DataclassPyTreeMixin):
    """Remote cut-wall replacements for coordinate stencil legs."""

    target_flat: jnp.ndarray
    axis: jnp.ndarray
    side: jnp.ndarray
    receive_slot: jnp.ndarray
    distance: jnp.ndarray
    active: jnp.ndarray

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
        target_flat = jnp.asarray(self.target_flat, dtype=jnp.int32)
        row_shape = tuple(int(v) for v in target_flat.shape)
        if target_flat.ndim != 1:
            raise ValueError(
                "LocalCoordinateStencilRemoteDependencyTable.target_flat "
                f"must be 1D, got {target_flat.shape}"
            )
        object.__setattr__(self, "target_flat", target_flat)
        for name in ("axis", "side", "receive_slot"):
            object.__setattr__(
                self,
                name,
                _require_shape(
                    getattr(self, name),
                    row_shape,
                    f"LocalCoordinateStencilRemoteDependencyTable.{name}",
                ).astype(jnp.int32),
            )
        object.__setattr__(
            self,
            "distance",
            _require_float_shape(
                self.distance,
                row_shape,
                "LocalCoordinateStencilRemoteDependencyTable.distance",
            ),
        )
        active = jnp.asarray(self.active, dtype=bool)
        if active.shape != row_shape:
            raise ValueError(
                "LocalCoordinateStencilRemoteDependencyTable.active must have "
                f"shape {row_shape}, got {active.shape}"
            )
        object.__setattr__(self, "active", active)
        _validate_coordinate_stencil_dependency_rows(
            target_flat=target_flat,
            axis=self.axis,
            side=self.side,
            distance=self.distance,
            active=active,
            label="LocalCoordinateStencilRemoteDependencyTable",
        )

        request_active = jnp.asarray(self.request_active, dtype=bool)
        request_shape = tuple(int(v) for v in request_active.shape)
        if request_active.ndim != 1:
            raise ValueError(
                "LocalCoordinateStencilRemoteDependencyTable.request_active "
                f"must be 1D, got {request_active.shape}"
            )
        object.__setattr__(self, "request_active", request_active)
        object.__setattr__(
            self,
            "request_dependency_kind",
            _require_shape(
                self.request_dependency_kind,
                request_shape,
                "LocalCoordinateStencilRemoteDependencyTable.request_dependency_kind",
            ).astype(jnp.int32),
        )
        for name in (
            "request_source_global_i",
            "request_source_global_j",
            "request_source_global_k",
            "request_source_shard_linear",
            "request_source_owner_local_i",
            "request_source_owner_local_j",
            "request_source_owner_local_k",
            "request_value_slot",
        ):
            object.__setattr__(
                self,
                name,
                _require_shape(
                    getattr(self, name),
                    request_shape,
                    f"LocalCoordinateStencilRemoteDependencyTable.{name}",
                ).astype(jnp.int32),
            )

        request_source_shard_index = jnp.asarray(
            self.request_source_shard_index,
            dtype=jnp.int32,
        )
        if (
            request_source_shard_index.ndim != 2
            or request_source_shard_index.shape[1] != 3
        ):
            raise ValueError(
                "LocalCoordinateStencilRemoteDependencyTable."
                "request_source_shard_index must have shape "
                f"(max_receive_values, 3), got {request_source_shard_index.shape}"
            )
        if int(request_source_shard_index.shape[0]) != request_shape[0]:
            raise ValueError(
                "LocalCoordinateStencilRemoteDependencyTable."
                "request_source_shard_index must match request_active length; "
                f"got {request_source_shard_index.shape[0]}, expected {request_shape[0]}"
            )
        object.__setattr__(
            self,
            "request_source_shard_index",
            request_source_shard_index,
        )

    @property
    def max_entries(self) -> int:
        return int(self.target_flat.size)

    @property
    def max_receive_values(self) -> int:
        return int(self.request_active.size)

    @property
    def has_requests(self) -> bool:
        return self.max_receive_values > 0

    @classmethod
    def empty(cls) -> "LocalCoordinateStencilRemoteDependencyTable":
        return cls(
            target_flat=jnp.zeros((0,), dtype=jnp.int32),
            axis=jnp.zeros((0,), dtype=jnp.int32),
            side=jnp.zeros((0,), dtype=jnp.int32),
            receive_slot=jnp.zeros((0,), dtype=jnp.int32),
            distance=jnp.zeros((0,), dtype=jnp.float64),
            active=jnp.zeros((0,), dtype=bool),
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
class LocalCoordinateStencilDependencyMap3D(_DataclassPyTreeMixin):
    """Coordinate-stencil cut-wall dependency metadata for one local shard."""

    layout: HaloLayout3D
    local: LocalCoordinateStencilLocalDependencyTable
    remote: LocalCoordinateStencilRemoteDependencyTable | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        if not isinstance(self.local, LocalCoordinateStencilLocalDependencyTable):
            raise TypeError(
                "local must be a LocalCoordinateStencilLocalDependencyTable instance"
            )
        if self.remote is not None and not isinstance(
            self.remote,
            LocalCoordinateStencilRemoteDependencyTable,
        ):
            raise TypeError(
                "remote must be a LocalCoordinateStencilRemoteDependencyTable or None"
            )
        if self.remote is not None:
            local_keys = _coordinate_stencil_dependency_keys(
                target_flat=self.local.target_flat,
                axis=self.local.axis,
                side=self.local.side,
                active=self.local.active,
            )
            remote_keys = _coordinate_stencil_dependency_keys(
                target_flat=self.remote.target_flat,
                axis=self.remote.axis,
                side=self.remote.side,
                active=self.remote.active,
            )
            if local_keys & remote_keys:
                raise ValueError(
                    "local and remote coordinate stencil dependencies must contain "
                    "at most one active row per (target_flat, axis, side)"
                )

    @classmethod
    def empty(cls, layout: HaloLayout3D) -> "LocalCoordinateStencilDependencyMap3D":
        return cls(
            layout=layout,
            local=LocalCoordinateStencilLocalDependencyTable.empty(),
            remote=None,
        )


def build_local_coordinate_stencil_dependency_map_from_cut_wall_geometry(
    layout: HaloLayout3D,
    cut_wall_geometry,
) -> LocalCoordinateStencilDependencyMap3D:
    """Build local coordinate-stencil dependencies from cut-wall leg metadata.

    This constructor treats each cut-wall entry as the value owned by the cell
    ``owner_i/j/k`` and uses the entry index as the local cut-wall value slot.
    Remote request rows require global owner/request shard information and are
    therefore left to the domain-decomposition dependency builder.
    """

    if not isinstance(layout, HaloLayout3D):
        raise TypeError("layout must be a HaloLayout3D instance")
    if cut_wall_geometry is None:
        return LocalCoordinateStencilDependencyMap3D.empty(layout)

    required_attrs = (
        "owner_i",
        "owner_j",
        "owner_k",
        "active",
        "max_wall_faces",
        "stencil_axis",
        "stencil_side",
        "stencil_distance",
    )
    for name in required_attrs:
        if not hasattr(cut_wall_geometry, name):
            raise TypeError(
                "cut_wall_geometry must provide LocalCutWallGeometry3D-style "
                f"{name!r} metadata"
            )

    max_wall_faces = int(cut_wall_geometry.max_wall_faces)
    if max_wall_faces == 0:
        return LocalCoordinateStencilDependencyMap3D.empty(layout)

    owner_i = jnp.asarray(cut_wall_geometry.owner_i, dtype=jnp.int32)
    owner_j = jnp.asarray(cut_wall_geometry.owner_j, dtype=jnp.int32)
    owner_k = jnp.asarray(cut_wall_geometry.owner_k, dtype=jnp.int32)
    stencil_axis = jnp.asarray(cut_wall_geometry.stencil_axis, dtype=jnp.int32)
    stencil_side = jnp.asarray(cut_wall_geometry.stencil_side, dtype=jnp.int32)
    stencil_distance = jnp.asarray(cut_wall_geometry.stencil_distance, dtype=jnp.float64)
    active = (
        jnp.asarray(cut_wall_geometry.active, dtype=bool)
        & (stencil_axis >= 0)
    )

    shape = (max_wall_faces,)
    for name, value in (
        ("owner_i", owner_i),
        ("owner_j", owner_j),
        ("owner_k", owner_k),
        ("stencil_axis", stencil_axis),
        ("stencil_side", stencil_side),
        ("stencil_distance", stencil_distance),
        ("active", active),
    ):
        if value.shape != shape:
            raise ValueError(
                f"cut_wall_geometry.{name} must have shape {shape}, got {value.shape}"
            )

    nx, ny, nz = layout.owned_shape
    target_in_bounds = (
        (owner_i >= 0)
        & (owner_i < nx)
        & (owner_j >= 0)
        & (owner_j < ny)
        & (owner_k >= 0)
        & (owner_k < nz)
    )
    try:
        all_targets_in_bounds = bool(jnp.all((~active) | target_in_bounds))
    except jax.errors.TracerBoolConversionError:
        all_targets_in_bounds = True
    if not all_targets_in_bounds:
        raise ValueError(
            "active coordinate-stencil cut-wall rows must use owned-local "
            "owner_i/j/k coordinates"
        )

    target_flat = (owner_i * ny + owner_j) * nz + owner_k
    local = LocalCoordinateStencilLocalDependencyTable(
        target_flat=target_flat,
        axis=stencil_axis,
        side=stencil_side,
        value_slot=jnp.arange(max_wall_faces, dtype=jnp.int32),
        distance=stencil_distance,
        active=active,
    )
    return LocalCoordinateStencilDependencyMap3D(
        layout=layout,
        local=local,
        remote=None,
    )


@_pytree_base
@dataclass(frozen=True)
class LocalFciDirectionMap(_DataclassPyTreeMixin):
    """One directional FCI dependency map for owned target cells."""

    layout: HaloLayout3D
    local: LocalFciLocalDependencyTable
    remote: LocalFciRemoteDependencyTable | None = None
    target_valid: jnp.ndarray | None = None  # (nx_owned, ny_owned, nz_owned)
    connection_length: jnp.ndarray | None = None  # (nx_owned, ny_owned, nz_owned)
    endpoint_kind: jnp.ndarray | None = None  # (nx_owned, ny_owned, nz_owned), int32

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        if not isinstance(self.local, LocalFciLocalDependencyTable):
            raise TypeError("local must be a LocalFciLocalDependencyTable instance")
        if self.remote is not None and not isinstance(self.remote, LocalFciRemoteDependencyTable):
            raise TypeError("remote must be a LocalFciRemoteDependencyTable instance or None")
        if self.target_valid is None:
            target_valid = jnp.ones(self.layout.owned_shape, dtype=bool)
        else:
            target_valid = jnp.asarray(self.target_valid, dtype=bool)
            if target_valid.shape != self.layout.owned_shape:
                raise ValueError(
                    "LocalFciDirectionMap.target_valid must match layout.owned_shape; "
                    f"got {target_valid.shape}, expected {self.layout.owned_shape}"
                )
        object.__setattr__(self, "target_valid", target_valid)

        if self.endpoint_kind is None:
            endpoint_kind = jnp.where(
                target_valid,
                FCI_DEP_FIELD_INTERIOR,
                FCI_DEP_INVALID,
            ).astype(jnp.int32)
        else:
            endpoint_kind = jnp.asarray(self.endpoint_kind, dtype=jnp.int32)
            if endpoint_kind.shape != self.layout.owned_shape:
                raise ValueError(
                    "LocalFciDirectionMap.endpoint_kind must match "
                    f"layout.owned_shape; got {endpoint_kind.shape}, expected "
                    f"{self.layout.owned_shape}"
                )
        object.__setattr__(self, "endpoint_kind", endpoint_kind)

        if self.connection_length is not None:
            connection_length = _require_float_shape(
                self.connection_length,
                self.layout.owned_shape,
                "LocalFciDirectionMap.connection_length",
            )
            object.__setattr__(self, "connection_length", connection_length)

    @property
    def owned_shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def has_remote_dependencies(self) -> bool:
        return self.remote is not None

    @property
    def has_local_dependencies(self) -> jnp.ndarray:
        return jnp.any(self.local.active)


@_pytree_base
@dataclass(frozen=True)
class LocalFciMaps3D(_DataclassPyTreeMixin):
    """Shard-local FCI dependency maps over owned target cells.

    This is a static description of interpolation dependencies only.
    It does not perform communication and does not own field data.
    """

    layout: HaloLayout3D
    forward: LocalFciDirectionMap
    backward: LocalFciDirectionMap
    mode: str = "remote_dependencies"

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        if not isinstance(self.forward, LocalFciDirectionMap):
            raise TypeError("forward must be a LocalFciDirectionMap instance")
        if not isinstance(self.backward, LocalFciDirectionMap):
            raise TypeError("backward must be a LocalFciDirectionMap instance")
        if self.forward.layout != self.layout or self.backward.layout != self.layout:
            raise ValueError("LocalFciMaps3D directions must share the same HaloLayout3D")
        mode = str(self.mode)
        if mode not in ("local_halo_only", "remote_dependencies"):
            raise ValueError(
                'mode must be either "local_halo_only" or "remote_dependencies", '
                f"got {mode!r}"
            )
        if mode == "local_halo_only" and (self.forward.remote is not None or self.backward.remote is not None):
            raise ValueError("local_halo_only mode cannot include remote dependency tables")
        object.__setattr__(self, "mode", mode)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def owned_shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def halo_shape(self) -> tuple[int, int, int]:
        return self.layout.cell_halo_shape

    @property
    def local_owned_shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def local_halo_shape(self) -> tuple[int, int, int]:
        return self.layout.cell_halo_shape

    @property
    def local_halo_only(self) -> bool:
        return self.mode == "local_halo_only"


@_pytree_base
@dataclass(frozen=True)
class LocalFciGeometry3D(_DataclassPyTreeMixin):
    """Shard-local FCI geometry bundle.

    This is the local counterpart to `FciGeometry3D`.
    It carries the shard-owned cell grid, the halo-padded local geometry
    fields, the local FCI dependency maps, and the owned-only conservative
    measures used by downstream operators.
    """

    layout: HaloLayout3D
    grid: LocalCellCenteredGrid3D
    maps: LocalFciMaps3D
    spacing: LocalSpacing3D
    cell_metric: LocalMetricGeometry
    face_metric: LocalFaceMetricGeometry
    cell_bfield: LocalBFieldGeometry
    face_bfield: LocalFaceBFieldGeometry
    regular_face_geometry: LocalRegularFaceGeometry3D
    cell_volume_geometry: LocalCellVolumeGeometry3D
    active_cell_mask: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        if not isinstance(self.grid, LocalCellCenteredGrid3D):
            raise TypeError("grid must be a LocalCellCenteredGrid3D instance")
        if not isinstance(self.maps, LocalFciMaps3D):
            raise TypeError("maps must be a LocalFciMaps3D instance")
        if not isinstance(self.spacing, LocalSpacing3D):
            raise TypeError("spacing must be a LocalSpacing3D instance")
        if not isinstance(self.cell_metric, LocalMetricGeometry):
            raise TypeError("cell_metric must be a LocalMetricGeometry instance")
        if not isinstance(self.face_metric, LocalFaceMetricGeometry):
            raise TypeError("face_metric must be a LocalFaceMetricGeometry instance")
        if not isinstance(self.cell_bfield, LocalBFieldGeometry):
            raise TypeError("cell_bfield must be a LocalBFieldGeometry instance")
        if not isinstance(self.face_bfield, LocalFaceBFieldGeometry):
            raise TypeError("face_bfield must be a LocalFaceBFieldGeometry instance")
        if not isinstance(self.regular_face_geometry, LocalRegularFaceGeometry3D):
            raise TypeError("regular_face_geometry must be a LocalRegularFaceGeometry3D instance")
        if not isinstance(self.cell_volume_geometry, LocalCellVolumeGeometry3D):
            raise TypeError("cell_volume_geometry must be a LocalCellVolumeGeometry3D instance")

        for name, value in (
            ("grid", self.grid.layout),
            ("maps", self.maps.layout),
            ("spacing", self.spacing.layout),
            ("cell_metric", self.cell_metric.layout),
            ("face_metric", self.face_metric.layout),
            ("cell_bfield", self.cell_bfield.layout),
            ("face_bfield", self.face_bfield.layout),
            ("regular_face_geometry", self.regular_face_geometry.layout),
            ("cell_volume_geometry", self.cell_volume_geometry.layout),
        ):
            if value != self.layout:
                raise ValueError(f"LocalFciGeometry3D.{name} must share the same HaloLayout3D")
        if self.active_cell_mask is not None:
            active_cell_mask = jnp.asarray(self.active_cell_mask, dtype=bool)
            if active_cell_mask.shape != self.layout.owned_shape:
                raise ValueError(
                    "LocalFciGeometry3D.active_cell_mask must have shape "
                    f"{self.layout.owned_shape}, got {active_cell_mask.shape}"
                )
            object.__setattr__(self, "active_cell_mask", active_cell_mask)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def owned_shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def halo_shape(self) -> tuple[int, int, int]:
        return self.layout.cell_halo_shape

    @property
    def local_owned_shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def local_halo_shape(self) -> tuple[int, int, int]:
        return self.layout.cell_halo_shape

    @property
    def active_cell_mask_owned(self) -> jnp.ndarray:
        """Owned-cell mask for cells that participate in solves and norms.

        Embedded solid or otherwise inactive cells are still valid storage, but
        callers that solve active-domain equations should exclude them from
        algebraic unknowns and diagnostics.  Geometries without embedded
        inactive regions default to all owned cells active.
        """

        if self.active_cell_mask is None:
            return jnp.ones(self.layout.owned_shape, dtype=bool)
        return self.active_cell_mask

    @property
    def x_centers(self) -> jnp.ndarray:
        return self.grid.x_centers

    @property
    def y_centers(self) -> jnp.ndarray:
        return self.grid.y_centers

    @property
    def z_centers(self) -> jnp.ndarray:
        return self.grid.z_centers

    @property
    def x_faces(self) -> jnp.ndarray:
        return self.grid.x_faces

    @property
    def y_faces(self) -> jnp.ndarray:
        return self.grid.y_faces

    @property
    def z_faces(self) -> jnp.ndarray:
        return self.grid.z_faces

    @property
    def logical_axis_vectors(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.grid.logical_axis_vectors

    @property
    def logical_face_vectors(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.grid.logical_face_vectors

    @property
    def x_centers_owned(self) -> jnp.ndarray:
        return self.grid.x_centers_owned

    @property
    def y_centers_owned(self) -> jnp.ndarray:
        return self.grid.y_centers_owned

    @property
    def z_centers_owned(self) -> jnp.ndarray:
        return self.grid.z_centers_owned

    @property
    def cell_volume(self) -> LocalCellVolumeGeometry3D:
        return self.cell_volume_geometry

    @property
    def regular_face(self) -> LocalRegularFaceGeometry3D:
        return self.regular_face_geometry

NeighborIndex3D = tuple[int, int, int]
OptionalNeighborIndex3D = NeighborIndex3D | None


# Meaning of a global lower/upper side in a ShardSpec3D. These values are
# metadata for later halo stages; HaloExchange3D only owns regular-neighbor
# and SIDE_SIMPLE_PERIODIC data.
SIDE_PHYSICAL = 1
SIDE_SIMPLE_PERIODIC = 2
SIDE_AXIS_REGULAR = 3
SIDE_TOPOLOGY_MAPPED = 4
SIDE_UNUSED = 5
_VALID_SIDE_KINDS = frozenset(
    {
        SIDE_PHYSICAL,
        SIDE_SIMPLE_PERIODIC,
        SIDE_AXIS_REGULAR,
        SIDE_TOPOLOGY_MAPPED,
        SIDE_UNUSED,
    }
)


@_pytree_base
@dataclass(frozen=True)
class ShardSpec3D(_DataclassPyTreeMixin):
    """Static metadata describing one shard's owned-cell block."""

    global_shape: tuple[int, int, int]
    owned_start: tuple[int, int, int]
    owned_stop: tuple[int, int, int]
    shard_index: tuple[int, int, int]
    shard_counts: tuple[int, int, int]
    periodic_axes: tuple[bool, bool, bool]
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False)
    halo_width: int = 1
    side_kind_lower: tuple[int, int, int] | None = None
    side_kind_upper: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        global_shape = tuple(int(v) for v in self.global_shape)
        owned_start = tuple(int(v) for v in self.owned_start)
        owned_stop = tuple(int(v) for v in self.owned_stop)
        shard_index = tuple(int(v) for v in self.shard_index)
        shard_counts = tuple(int(v) for v in self.shard_counts)
        periodic_axes = tuple(bool(v) for v in self.periodic_axes)
        axis_regular_axes = tuple(bool(v) for v in self.axis_regular_axes)
        halo_width = int(self.halo_width)

        if self.side_kind_lower is None:
            side_kind_lower = tuple(
                SIDE_SIMPLE_PERIODIC if periodic else SIDE_PHYSICAL
                for periodic in periodic_axes
            )
        else:
            side_kind_lower = tuple(int(v) for v in self.side_kind_lower)
        if self.side_kind_upper is None:
            side_kind_upper = tuple(
                SIDE_SIMPLE_PERIODIC if periodic else SIDE_PHYSICAL
                for periodic in periodic_axes
            )
        else:
            side_kind_upper = tuple(int(v) for v in self.side_kind_upper)

        for name, value in (
            ("global_shape", global_shape),
            ("owned_start", owned_start),
            ("owned_stop", owned_stop),
            ("shard_index", shard_index),
            ("shard_counts", shard_counts),
        ):
            if len(value) != 3:
                raise ValueError(f"ShardSpec3D.{name} must have length 3, got {value}")
        if len(periodic_axes) != 3:
            raise ValueError(f"ShardSpec3D.periodic_axes must have length 3, got {periodic_axes}")
        if len(axis_regular_axes) != 3:
            raise ValueError(f"ShardSpec3D.axis_regular_axes must have length 3, got {axis_regular_axes}")
        if len(side_kind_lower) != 3:
            raise ValueError(f"ShardSpec3D.side_kind_lower must have length 3, got {side_kind_lower}")
        if len(side_kind_upper) != 3:
            raise ValueError(f"ShardSpec3D.side_kind_upper must have length 3, got {side_kind_upper}")
        if any(kind not in _VALID_SIDE_KINDS for kind in side_kind_lower + side_kind_upper):
            raise ValueError(
                "ShardSpec3D side kinds must be one of "
                f"{sorted(_VALID_SIDE_KINDS)}, got lower={side_kind_lower}, "
                f"upper={side_kind_upper}"
            )
        for axis, periodic in enumerate(periodic_axes):
            if periodic and (
                side_kind_lower[axis] != SIDE_SIMPLE_PERIODIC
                or side_kind_upper[axis] != SIDE_SIMPLE_PERIODIC
            ):
                raise ValueError(
                    "periodic_axes requires SIDE_SIMPLE_PERIODIC on both global "
                    f"sides; axis={axis}, lower={side_kind_lower[axis]}, "
                    f"upper={side_kind_upper[axis]}"
                )
        if any(size <= 0 for size in global_shape):
            raise ValueError(f"ShardSpec3D.global_shape must contain positive integers, got {global_shape}")
        if any(start < 0 for start in owned_start):
            raise ValueError(f"ShardSpec3D.owned_start must be non-negative, got {owned_start}")
        if any(stop <= start for start, stop in zip(owned_start, owned_stop)):
            raise ValueError(
                "ShardSpec3D.owned_stop must be strictly greater than owned_start on every axis; "
                f"got start={owned_start}, stop={owned_stop}"
            )
        if any(stop > size for stop, size in zip(owned_stop, global_shape)):
            raise ValueError(
                "ShardSpec3D.owned_stop must not exceed global_shape; "
                f"got stop={owned_stop}, global_shape={global_shape}"
            )
        if any(count <= 0 for count in shard_counts):
            raise ValueError(f"ShardSpec3D.shard_counts must contain positive integers, got {shard_counts}")
        if any(index < 0 or index >= count for index, count in zip(shard_index, shard_counts)):
            raise ValueError(
                "ShardSpec3D.shard_index must lie within shard_counts; "
                f"got shard_index={shard_index}, shard_counts={shard_counts}"
            )
        if halo_width < 0:
            raise ValueError(f"ShardSpec3D.halo_width must be non-negative, got {halo_width}")

        object.__setattr__(self, "global_shape", global_shape)
        object.__setattr__(self, "owned_start", owned_start)
        object.__setattr__(self, "owned_stop", owned_stop)
        object.__setattr__(self, "shard_index", shard_index)
        object.__setattr__(self, "shard_counts", shard_counts)
        object.__setattr__(self, "periodic_axes", periodic_axes)
        object.__setattr__(self, "axis_regular_axes", axis_regular_axes)
        object.__setattr__(self, "halo_width", halo_width)
        object.__setattr__(self, "side_kind_lower", side_kind_lower)
        object.__setattr__(self, "side_kind_upper", side_kind_upper)

    @property
    def owned_shape(self) -> tuple[int, int, int]:
        return (
            self.owned_stop[0] - self.owned_start[0],
            self.owned_stop[1] - self.owned_start[1],
            self.owned_stop[2] - self.owned_start[2],
        )

    def touches_lower(self, axis: int) -> bool:
        """Static host/debug check for a per-shard domain description."""
        return self.owned_start[int(axis)] == 0

    def touches_upper(self, axis: int) -> bool:
        """Static host/debug check for a per-shard domain description."""
        return self.owned_stop[int(axis)] == self.global_shape[int(axis)]

    @staticmethod
    def _check_axis(axis: int) -> int:
        axis = int(axis)
        if axis < 0 or axis > 2:
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        return axis

    def lower_side_kind(self, axis: int) -> int:
        return int(self.side_kind_lower[self._check_axis(axis)])

    def upper_side_kind(self, axis: int) -> int:
        return int(self.side_kind_upper[self._check_axis(axis)])

    def has_physical_lower(self, axis: int) -> bool:
        """Static host/debug helper; not runtime SPMD ownership."""
        axis = int(axis)
        return self.touches_lower(axis) and self.lower_side_kind(axis) == SIDE_PHYSICAL

    def has_physical_upper(self, axis: int) -> bool:
        """Static host/debug helper; not runtime SPMD ownership."""
        axis = int(axis)
        return self.touches_upper(axis) and self.upper_side_kind(axis) == SIDE_PHYSICAL

    def allows_regular_exchange_lower(self, axis: int) -> bool:
        axis = self._check_axis(axis)
        return not self.touches_lower(axis) or self.lower_side_kind(axis) == SIDE_SIMPLE_PERIODIC

    def allows_regular_exchange_upper(self, axis: int) -> bool:
        axis = self._check_axis(axis)
        return not self.touches_upper(axis) or self.upper_side_kind(axis) == SIDE_SIMPLE_PERIODIC

    def has_topology_lower(self, axis: int) -> bool:
        return self.touches_lower(axis) and self.lower_side_kind(axis) in (
            SIDE_SIMPLE_PERIODIC,
            SIDE_AXIS_REGULAR,
            SIDE_TOPOLOGY_MAPPED,
        )

    def has_topology_upper(self, axis: int) -> bool:
        return self.touches_upper(axis) and self.upper_side_kind(axis) in (
            SIDE_SIMPLE_PERIODIC,
            SIDE_AXIS_REGULAR,
            SIDE_TOPOLOGY_MAPPED,
        )

    def tree_flatten(self):
        return (), (
            self.global_shape,
            self.owned_start,
            self.owned_stop,
            self.shard_index,
            self.shard_counts,
            self.periodic_axes,
            self.axis_regular_axes,
            self.side_kind_lower,
            self.side_kind_upper,
            self.halo_width,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        (
            global_shape,
            owned_start,
            owned_stop,
            shard_index,
            shard_counts,
            periodic_axes,
            axis_regular_axes,
            side_kind_lower,
            side_kind_upper,
            halo_width,
        ) = aux_data
        return cls(
            global_shape=global_shape,
            owned_start=owned_start,
            owned_stop=owned_stop,
            shard_index=shard_index,
            shard_counts=shard_counts,
            periodic_axes=periodic_axes,
            axis_regular_axes=axis_regular_axes,
            halo_width=halo_width,
            side_kind_lower=side_kind_lower,
            side_kind_upper=side_kind_upper,
        )


@_pytree_base
@dataclass(frozen=True)
class NeighborMap3D(_DataclassPyTreeMixin):
    """Shard-adjacency metadata for one local 3D domain."""

    minus: tuple[OptionalNeighborIndex3D, OptionalNeighborIndex3D, OptionalNeighborIndex3D]
    plus: tuple[OptionalNeighborIndex3D, OptionalNeighborIndex3D, OptionalNeighborIndex3D]

    def __post_init__(self) -> None:
        minus = tuple(self.minus)
        plus = tuple(self.plus)
        if len(minus) != 3:
            raise ValueError(f"NeighborMap3D.minus must have length 3, got {minus}")
        if len(plus) != 3:
            raise ValueError(f"NeighborMap3D.plus must have length 3, got {plus}")
        normalized_minus = []
        normalized_plus = []
        for name, side, normalized in (
            ("minus", minus, normalized_minus),
            ("plus", plus, normalized_plus),
        ):
            for entry in side:
                if entry is None:
                    normalized.append(None)
                    continue
                if len(entry) != 3:
                    raise ValueError(f"NeighborMap3D.{name} entries must be length-3 tuples or None, got {entry}")
                normalized.append(tuple(int(v) for v in entry))
        object.__setattr__(self, "minus", tuple(normalized_minus))
        object.__setattr__(self, "plus", tuple(normalized_plus))

    def tree_flatten(self):
        return (), (self.minus, self.plus)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        minus, plus = aux_data
        return cls(minus=minus, plus=plus)


@_pytree_base
@dataclass(frozen=True)
class LocalDomain3D(_DataclassPyTreeMixin):
    """Metadata for one local shard/domain.

    ``mesh_axis_names`` describes the execution mesh used by SPMD-facing
    helpers. It is deliberately kept on ``LocalDomain3D`` rather than
    ``ShardSpec3D`` because collective axis names are execution metadata, not
    geometric metadata.

    The existing ``touches_*`` and ``has_*`` methods are host/debug helpers
    based on the static per-shard metadata in ``ShardSpec3D``. Code executing
    inside ``pmap``/``shard_map`` should use the ``runtime_*`` methods below.
    """

    shard_spec: ShardSpec3D
    layout: HaloLayout3D
    neighbor_map: NeighborMap3D | None = None
    mesh_axis_names: tuple[str | None, str | None, str | None] = (
        None,
        None,
        None,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.shard_spec, ShardSpec3D):
            raise TypeError("shard_spec must be a ShardSpec3D instance")
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        if self.layout.owned_shape != self.shard_spec.owned_shape:
            raise ValueError(
                "LocalDomain3D.layout.owned_shape must match shard_spec.owned_shape; "
                f"got layout={self.layout.owned_shape}, shard_spec={self.shard_spec.owned_shape}"
            )
        if self.layout.halo_width != self.shard_spec.halo_width:
            raise ValueError(
                "LocalDomain3D.layout.halo_width must match shard_spec.halo_width; "
                f"got layout={self.layout.halo_width}, shard_spec={self.shard_spec.halo_width}"
            )
        if self.neighbor_map is not None and not isinstance(self.neighbor_map, NeighborMap3D):
            raise TypeError("neighbor_map must be a NeighborMap3D instance or None")
        mesh_axis_names = tuple(self.mesh_axis_names)
        if len(mesh_axis_names) != 3:
            raise ValueError(
                "LocalDomain3D.mesh_axis_names must have length 3, "
                f"got {mesh_axis_names}"
            )
        for axis, name in enumerate(mesh_axis_names):
            if name is not None and not isinstance(name, str):
                raise TypeError(
                    "LocalDomain3D.mesh_axis_names entries must be strings or None; "
                    f"axis={axis}, value={name!r}"
                )
        object.__setattr__(self, "mesh_axis_names", mesh_axis_names)

    @property
    def periodic_axes(self) -> tuple[bool, bool, bool]:
        return self.shard_spec.periodic_axes

    @property
    def axis_regular_axes(self) -> tuple[bool, bool, bool]:
        return self.shard_spec.axis_regular_axes

    @property
    def owned_shape(self) -> tuple[int, int, int]:
        return self.shard_spec.owned_shape

    def has_physical_lower(self, axis: int) -> bool:
        """Static host/debug helper; use ``runtime_has_physical_lower`` in SPMD."""
        return self.shard_spec.has_physical_lower(axis)

    def has_physical_upper(self, axis: int) -> bool:
        """Static host/debug helper; use ``runtime_has_physical_upper`` in SPMD."""
        return self.shard_spec.has_physical_upper(axis)

    def allows_regular_exchange_lower(self, axis: int) -> bool:
        return self.shard_spec.allows_regular_exchange_lower(axis)

    def allows_regular_exchange_upper(self, axis: int) -> bool:
        return self.shard_spec.allows_regular_exchange_upper(axis)

    def has_topology_lower(self, axis: int) -> bool:
        return self.shard_spec.has_topology_lower(axis)

    def has_topology_upper(self, axis: int) -> bool:
        return self.shard_spec.has_topology_upper(axis)

    def runtime_shard_id(self, axis: int) -> int | jnp.ndarray:
        """Return the current SPMD shard index for a logical axis.

        An axis without a configured mesh name is treated as undecomposed and
        returns the Python integer ``0``. A configured name must be valid in
        the surrounding ``pmap``/``shard_map`` context.
        """

        axis = int(axis)
        if axis < 0 or axis > 2:
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        name = self.mesh_axis_names[axis]
        if name is None:
            return 0
        return lax.axis_index(name)

    def runtime_touches_lower(self, axis: int) -> bool | jnp.ndarray:
        axis = int(axis)
        return self.runtime_shard_id(axis) == 0

    def runtime_touches_upper(self, axis: int) -> bool | jnp.ndarray:
        axis = int(axis)
        return self.runtime_shard_id(axis) == self.shard_spec.shard_counts[axis] - 1

    def runtime_has_physical_lower(self, axis: int) -> bool | jnp.ndarray:
        axis = int(axis)
        return self.runtime_touches_lower(axis) & (
            self.shard_spec.lower_side_kind(axis) == SIDE_PHYSICAL
        )

    def runtime_has_physical_upper(self, axis: int) -> bool | jnp.ndarray:
        axis = int(axis)
        return self.runtime_touches_upper(axis) & (
            self.shard_spec.upper_side_kind(axis) == SIDE_PHYSICAL
        )

    def runtime_has_axis_regular_lower(self, axis: int) -> bool | jnp.ndarray:
        axis = int(axis)
        return self.runtime_touches_lower(axis) & (
            self.shard_spec.lower_side_kind(axis) == SIDE_AXIS_REGULAR
        )

    def runtime_has_axis_regular_upper(self, axis: int) -> bool | jnp.ndarray:
        axis = int(axis)
        return self.runtime_touches_upper(axis) & (
            self.shard_spec.upper_side_kind(axis) == SIDE_AXIS_REGULAR
        )

    def runtime_has_side_kind_lower(
        self,
        axis: int,
        side_kind: int,
    ) -> bool | jnp.ndarray:
        axis = int(axis)
        return self.runtime_touches_lower(axis) & (
            self.shard_spec.lower_side_kind(axis) == int(side_kind)
        )

    def runtime_has_side_kind_upper(
        self,
        axis: int,
        side_kind: int,
    ) -> bool | jnp.ndarray:
        axis = int(axis)
        return self.runtime_touches_upper(axis) & (
            self.shard_spec.upper_side_kind(axis) == int(side_kind)
        )

    def tree_flatten(self):
        return (), (
            self.shard_spec,
            self.layout,
            self.neighbor_map,
            self.mesh_axis_names,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        shard_spec, layout, neighbor_map, mesh_axis_names = aux_data
        return cls(
            shard_spec=shard_spec,
            layout=layout,
            neighbor_map=neighbor_map,
            mesh_axis_names=mesh_axis_names,
        )


@_pytree_base
@dataclass(frozen=True)
class StencilBuilderContext(_DataclassPyTreeMixin):
    layout: HaloLayout3D
    domain: LocalDomain3D | None = None
    cut_wall_geometry: "LocalCutWallGeometry3D | None" = None
    cut_wall_bc: "LocalCutWallBC3D | None" = None
    cut_wall_value_reconstructor: "LocalCutWallValueReconstructor3D | None" = None
    cut_wall_stencil_dependencies: (
        "LocalCoordinateStencilDependencyMap3D | None"
    ) = None
    cut_wall_values: jnp.ndarray | None = None
    cut_wall_stencil_remote_values: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        if self.domain is not None and self.domain.layout != self.layout:
            raise ValueError("StencilBuilderContext.domain must share the same layout")
        if (
            self.cut_wall_stencil_dependencies is not None
            and self.cut_wall_stencil_dependencies.layout != self.layout
        ):
            raise ValueError(
                "StencilBuilderContext.cut_wall_stencil_dependencies must share "
                "the same layout"
            )


# Backward-compatible aliases. Both historical context names now refer to the
# same canonical PyTree type.
LocalStencilBuilderContext = StencilBuilderContext
ConservativeStencilBuilderContext = StencilBuilderContext


@_pytree_base
@dataclass(frozen=True)
class Spacing3D(_DataclassPyTreeMixin):
    """Logical spacings evaluated at active cell centers.
    These are usually broadcast arrays with shape (nx, ny, nz)."""
    dx: jnp.ndarray
    dy: jnp.ndarray
    dz: jnp.ndarray
    def __post_init__(self) -> None:
        dx = jnp.asarray(self.dx, dtype=jnp.float64)
        if dx.ndim != 3:
            raise ValueError(f"Spacing3D.dx must have shape (nx, ny, nz), got {dx.shape}")
        shape = tuple(int(v) for v in dx.shape)
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "dy", _require_float_shape(self.dy, shape, "Spacing3D.dy"))
        object.__setattr__(self, "dz", _require_float_shape(self.dz, shape, "Spacing3D.dz"))
    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.dx.shape)


@_pytree_base
@dataclass(frozen=True)
class LocalSpacing3D(_DataclassPyTreeMixin):
    """Shard-local logical spacings with halo and owned views."""

    layout: HaloLayout3D
    dx_halo: jnp.ndarray  # (nx + 2*h, ny + 2*h, nz + 2*h)
    dy_halo: jnp.ndarray  # (nx + 2*h, ny + 2*h, nz + 2*h)
    dz_halo: jnp.ndarray  # (nx + 2*h, ny + 2*h, nz + 2*h)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        dx_halo = jnp.asarray(self.dx_halo, dtype=jnp.float64)
        if dx_halo.ndim != 3:
            raise ValueError(f"LocalSpacing3D.dx_halo must have shape (nx, ny, nz), got {dx_halo.shape}")
        shape = tuple(int(v) for v in dx_halo.shape)
        if shape != self.layout.cell_halo_shape:
            raise ValueError(
                "LocalSpacing3D.dx_halo must match layout.cell_halo_shape; "
                f"got {shape}, expected {self.layout.cell_halo_shape}"
            )

        dy_halo = _require_float_shape(self.dy_halo, shape, "LocalSpacing3D.dy_halo")
        dz_halo = _require_float_shape(self.dz_halo, shape, "LocalSpacing3D.dz_halo")

        object.__setattr__(self, "dx_halo", dx_halo)
        object.__setattr__(self, "dy_halo", dy_halo)
        object.__setattr__(self, "dz_halo", dz_halo)

    @property
    def dx(self) -> jnp.ndarray:
        return self.dx_halo

    @property
    def dy(self) -> jnp.ndarray:
        return self.dy_halo

    @property
    def dz(self) -> jnp.ndarray:
        return self.dz_halo

    @property
    def dx_owned(self) -> jnp.ndarray:
        return self.dx_halo[self.owned_slices_in_halo]

    @property
    def dy_owned(self) -> jnp.ndarray:
        return self.dy_halo[self.owned_slices_in_halo]

    @property
    def dz_owned(self) -> jnp.ndarray:
        return self.dz_halo[self.owned_slices_in_halo]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.dx_halo.shape)

    @property
    def halo_shape(self) -> tuple[int, int, int]:
        return self.shape

    @property
    def owned_shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def local_halo_shape(self) -> tuple[int, int, int]:
        return self.shape

    @property
    def local_owned_shape(self) -> tuple[int, int, int]:
        return self.owned_shape

    @property
    def owned_slices_in_halo(self) -> tuple[slice, slice, slice]:
        return self.layout.owned_slices_cell

@_pytree_base
@dataclass(frozen=True)
class LocalMetricGeometry(_DataclassPyTreeMixin):
    """Metric coefficients on a local halo-padded cell or face grid.

    These arrays are local arrays, not views into a global MetricGeometry.
    The `location` metadata determines which local shape convention applies:
    cell-centered or one of the three face families.
    """
    #field_halo shaped arrays
    layout: HaloLayout3D
    J_halo: jnp.ndarray
    g11_halo: jnp.ndarray
    g22_halo: jnp.ndarray
    g33_halo: jnp.ndarray
    g12_halo: jnp.ndarray
    g13_halo: jnp.ndarray
    g23_halo: jnp.ndarray
    g_11_halo: jnp.ndarray
    g_22_halo: jnp.ndarray
    g_33_halo: jnp.ndarray
    g_12_halo: jnp.ndarray
    g_13_halo: jnp.ndarray
    g_23_halo: jnp.ndarray
    location: str  # "cell", "x_face", "y_face", or "z_face"

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        location = str(self.location)
        expected_shape = self.layout.location_halo_shape(location)
        J_halo = jnp.asarray(self.J_halo, dtype=jnp.float64)
        if J_halo.ndim != 3:
            raise ValueError(f"LocalMetricGeometry.J_halo must be 3D, got {J_halo.shape}")
        if tuple(int(v) for v in J_halo.shape) != expected_shape:
            raise ValueError(
                "LocalMetricGeometry.J_halo must match the expected halo shape for the location; "
                f"got {J_halo.shape}, expected {expected_shape} for location={location!r}"
            )

        object.__setattr__(self, "J_halo", J_halo)
        for name in (
            "g11_halo",
            "g22_halo",
            "g33_halo",
            "g12_halo",
            "g13_halo",
            "g23_halo",
            "g_11_halo",
            "g_22_halo",
            "g_33_halo",
            "g_12_halo",
            "g_13_halo",
            "g_23_halo",
        ):
            object.__setattr__(self, name, _require_float_shape(getattr(self, name), expected_shape, f"LocalMetricGeometry.{name}"))
        object.__setattr__(self, "location", location)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.J_halo.shape)

    @property
    def halo_shape(self) -> tuple[int, int, int]:
        return self.shape

    @property
    def local_halo_shape(self) -> tuple[int, int, int]:
        return self.shape

    @property
    def local_owned_shape(self) -> tuple[int, int, int]:
        return self.layout.location_owned_shape(self.location)

    @property
    def owned_slices_in_halo(self) -> tuple[slice, slice, slice]:
        return self.layout.location_owned_slices(self.location)

    @property
    def J(self) -> jnp.ndarray:
        return self.J_halo

    @property
    def g11(self) -> jnp.ndarray:
        return self.g11_halo

    @property
    def g22(self) -> jnp.ndarray:
        return self.g22_halo

    @property
    def g33(self) -> jnp.ndarray:
        return self.g33_halo

    @property
    def g12(self) -> jnp.ndarray:
        return self.g12_halo

    @property
    def g13(self) -> jnp.ndarray:
        return self.g13_halo

    @property
    def g23(self) -> jnp.ndarray:
        return self.g23_halo

    @property
    def g_11(self) -> jnp.ndarray:
        return self.g_11_halo

    @property
    def g_22(self) -> jnp.ndarray:
        return self.g_22_halo

    @property
    def g_33(self) -> jnp.ndarray:
        return self.g_33_halo

    @property
    def g_12(self) -> jnp.ndarray:
        return self.g_12_halo

    @property
    def g_13(self) -> jnp.ndarray:
        return self.g_13_halo

    @property
    def g_23(self) -> jnp.ndarray:
        return self.g_23_halo

    @property
    def g_contra(self) -> jnp.ndarray:
        return _metric_from_components(self.g11_halo, self.g22_halo, self.g33_halo, self.g12_halo, self.g13_halo, self.g23_halo)

    @property
    def g_cov(self) -> jnp.ndarray:
        return _metric_from_components(self.g_11_halo, self.g_22_halo, self.g_33_halo, self.g_12_halo, self.g_13_halo, self.g_23_halo)

    @property
    def g_contra_owned(self) -> jnp.ndarray:
        s = self.owned_slices_in_halo
        return _metric_from_components(
            self.g11_halo[s],
            self.g22_halo[s],
            self.g33_halo[s],
            self.g12_halo[s],
            self.g13_halo[s],
            self.g23_halo[s],
        )

    @property
    def g_cov_owned(self) -> jnp.ndarray:
        s = self.owned_slices_in_halo
        return _metric_from_components(
            self.g_11_halo[s],
            self.g_22_halo[s],
            self.g_33_halo[s],
            self.g_12_halo[s],
            self.g_13_halo[s],
            self.g_23_halo[s],
        )

    @property
    def J_owned(self) -> jnp.ndarray:
        return self.J_halo[self.owned_slices_in_halo]

    @property
    def g11_owned(self) -> jnp.ndarray:
        return self.g11_halo[self.owned_slices_in_halo]

    @property
    def g22_owned(self) -> jnp.ndarray:
        return self.g22_halo[self.owned_slices_in_halo]

    @property
    def g33_owned(self) -> jnp.ndarray:
        return self.g33_halo[self.owned_slices_in_halo]

    @property
    def g12_owned(self) -> jnp.ndarray:
        return self.g12_halo[self.owned_slices_in_halo]

    @property
    def g13_owned(self) -> jnp.ndarray:
        return self.g13_halo[self.owned_slices_in_halo]

    @property
    def g23_owned(self) -> jnp.ndarray:
        return self.g23_halo[self.owned_slices_in_halo]

    @property
    def g_11_owned(self) -> jnp.ndarray:
        return self.g_11_halo[self.owned_slices_in_halo]

    @property
    def g_22_owned(self) -> jnp.ndarray:
        return self.g_22_halo[self.owned_slices_in_halo]

    @property
    def g_33_owned(self) -> jnp.ndarray:
        return self.g_33_halo[self.owned_slices_in_halo]

    @property
    def g_12_owned(self) -> jnp.ndarray:
        return self.g_12_halo[self.owned_slices_in_halo]

    @property
    def g_13_owned(self) -> jnp.ndarray:
        return self.g_13_halo[self.owned_slices_in_halo]

    @property
    def g_23_owned(self) -> jnp.ndarray:
        return self.g_23_halo[self.owned_slices_in_halo]

@_pytree_base
@dataclass(frozen=True)
class MetricGeometry(_DataclassPyTreeMixin):
    """Metric/Jacobian data on one grid location family.
    This class is used both for cell centers and for each face family."""
    J: jnp.ndarray
    g11: jnp.ndarray
    g22: jnp.ndarray
    g33: jnp.ndarray
    g12: jnp.ndarray
    g13: jnp.ndarray
    g23: jnp.ndarray
    g_11: jnp.ndarray
    g_22: jnp.ndarray
    g_33: jnp.ndarray
    g_12: jnp.ndarray
    g_13: jnp.ndarray
    g_23: jnp.ndarray
    def __post_init__(self) -> None:
        J = jnp.asarray(self.J, dtype=jnp.float64)
        if J.ndim != 3:
            raise ValueError(f"MetricGeometry.J must have shape (a, b, c), got {J.shape}")
        shape = tuple(int(v) for v in J.shape)
        object.__setattr__(self, "J", J)
        for name in ("g11", "g22", "g33", "g12", "g13", "g23", "g_11", "g_22", "g_33", "g_12", "g_13", "g_23"):
            value = _require_float_shape(getattr(self, name), shape, f"MetricGeometry.{name}")
            object.__setattr__(self, name, value)
    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.J.shape)
    @property
    def g_contra(self) -> jnp.ndarray:
        return _metric_from_components(self.g11, self.g22, self.g33, self.g12, self.g13, self.g23)
    @property
    def g_cov(self) -> jnp.ndarray:
        return _metric_from_components(self.g_11, self.g_22, self.g_33, self.g_12, self.g_13, self.g_23)

@_pytree_base
@dataclass(frozen=True)
class FaceMetricGeometry(_DataclassPyTreeMixin):
    x: MetricGeometry  # (nx + 1, ny, nz)
    y: MetricGeometry  # (nx, ny + 1, nz)
    z: MetricGeometry  # (nx, ny, nz + 1)
    @property
    def axes(self) -> tuple[MetricGeometry, MetricGeometry, MetricGeometry]:
        return self.x, self.y, self.z


@_pytree_base
@dataclass(frozen=True)
class LocalFaceMetricGeometry(_DataclassPyTreeMixin):
    """Local metric bundles on the x/y/z face families.

    Each field stores a halo-padded local metric object for that face family.
    The shape annotations below describe the expected local array extent
    when the underlying face family is built from a shard-local owned region
    with halo width `h`.
    """

    layout: HaloLayout3D
    x: LocalMetricGeometry  # (nx_owned + 2*h + 1, ny_owned + 2*h, nz_owned + 2*h)
    y: LocalMetricGeometry  # (nx_owned + 2*h, ny_owned + 2*h + 1, nz_owned + 2*h)
    z: LocalMetricGeometry  # (nx_owned + 2*h, ny_owned + 2*h, nz_owned + 2*h + 1)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        if not isinstance(self.x, LocalMetricGeometry):
            raise TypeError("x must be a LocalMetricGeometry instance")
        if not isinstance(self.y, LocalMetricGeometry):
            raise TypeError("y must be a LocalMetricGeometry instance")
        if not isinstance(self.z, LocalMetricGeometry):
            raise TypeError("z must be a LocalMetricGeometry instance")
        if self.x.layout != self.layout or self.y.layout != self.layout or self.z.layout != self.layout:
            raise ValueError("LocalFaceMetricGeometry axes must share the same HaloLayout3D")
        if self.x.location != "x_face":
            raise ValueError(f"LocalFaceMetricGeometry.x.location must be 'x_face', got {self.x.location!r}")
        if self.y.location != "y_face":
            raise ValueError(f"LocalFaceMetricGeometry.y.location must be 'y_face', got {self.y.location!r}")
        if self.z.location != "z_face":
            raise ValueError(f"LocalFaceMetricGeometry.z.location must be 'z_face', got {self.z.location!r}")

    @property
    def axes(self) -> tuple[LocalMetricGeometry, LocalMetricGeometry, LocalMetricGeometry]:
        return self.x, self.y, self.z

    @property
    def shape(self) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        return self.x.shape, self.y.shape, self.z.shape

@_pytree_base
@dataclass(frozen=True)
class BFieldGeometry(_DataclassPyTreeMixin):
    B_contra: jnp.ndarray
    Bmag: jnp.ndarray
    def __post_init__(self) -> None:
        B_contra = jnp.asarray(self.B_contra, dtype=jnp.float64)
        if B_contra.ndim != 4 or B_contra.shape[-1] != 3:
            raise ValueError(f"BFieldGeometry.B_contra must have shape (a, b, c, 3), got {B_contra.shape}")
        shape = tuple(int(v) for v in B_contra.shape[:-1])
        Bmag = _require_float_shape(self.Bmag, shape, "BFieldGeometry.Bmag")
        object.__setattr__(self, "B_contra", B_contra)
        object.__setattr__(self, "Bmag", Bmag)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.Bmag.shape)

    @property
    def b_contra(self) -> jnp.ndarray:
        return self.B_contra / self.Bmag[..., None]

@_pytree_base
@dataclass(frozen=True)
class FaceBFieldGeometry(_DataclassPyTreeMixin):
    x: BFieldGeometry  # (nx + 1, ny, nz, 3)
    y: BFieldGeometry  # (nx, ny + 1, nz, 3)
    z: BFieldGeometry  # (nx, ny, nz + 1, 3)

    @property
    def axes(self) -> tuple[BFieldGeometry, BFieldGeometry, BFieldGeometry]:
        return self.x, self.y, self.z


@_pytree_base
@dataclass(frozen=True)
class LocalBFieldGeometry(_DataclassPyTreeMixin):
    """Local magnetic field bundle on a halo-padded cell or face grid."""

    layout: HaloLayout3D
    B_contra_halo: jnp.ndarray  # (..., 3) with leading halo_shape / face_halo_shape
    Bmag_halo: jnp.ndarray  # halo_shape / face_halo_shape
    location: str  # "cell", "x_face", "y_face", or "z_face"

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        location = str(self.location)
        expected_shape = self.layout.location_halo_shape(location)
        B_contra_halo = jnp.asarray(self.B_contra_halo, dtype=jnp.float64)
        if B_contra_halo.ndim != 4 or B_contra_halo.shape[-1] != 3:
            raise ValueError(
                "LocalBFieldGeometry.B_contra_halo must have shape "
                f"{expected_shape + (3,)}, got {B_contra_halo.shape}"
            )
        if tuple(int(v) for v in B_contra_halo.shape[:-1]) != expected_shape:
            raise ValueError(
                "LocalBFieldGeometry.B_contra_halo must match the expected halo shape for the location; "
                f"got {B_contra_halo.shape[:-1]}, expected {expected_shape} for location={location!r}"
            )

        object.__setattr__(self, "B_contra_halo", B_contra_halo)
        object.__setattr__(self, "Bmag_halo", _require_float_shape(self.Bmag_halo, expected_shape, "LocalBFieldGeometry.Bmag_halo"))
        object.__setattr__(self, "location", location)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.Bmag_halo.shape)

    @property
    def halo_shape(self) -> tuple[int, int, int]:
        return self.shape

    @property
    def local_halo_shape(self) -> tuple[int, int, int]:
        return self.shape

    @property
    def local_owned_shape(self) -> tuple[int, int, int]:
        return self.layout.location_owned_shape(self.location)

    @property
    def owned_slices_in_halo(self) -> tuple[slice, slice, slice]:
        return self.layout.location_owned_slices(self.location)

    @property
    def B_contra(self) -> jnp.ndarray:
        return self.B_contra_halo

    @property
    def Bmag(self) -> jnp.ndarray:
        return self.Bmag_halo

    @property
    def b_contra(self) -> jnp.ndarray:
        return self.B_contra_halo / self.Bmag_halo[..., None]

    @property
    def B_contra_owned(self) -> jnp.ndarray:
        return self.B_contra_halo[self.owned_slices_in_halo]

    @property
    def Bmag_owned(self) -> jnp.ndarray:
        return self.Bmag_halo[self.owned_slices_in_halo]


@_pytree_base
@dataclass(frozen=True)
class LocalFaceBFieldGeometry(_DataclassPyTreeMixin):
    """Local B-field bundles on the x/y/z face families.

    Shape annotations:
      x: (nx_owned + 2*h + 1, ny_owned + 2*h, nz_owned + 2*h, 3)
      y: (nx_owned + 2*h, ny_owned + 2*h + 1, nz_owned + 2*h, 3)
      z: (nx_owned + 2*h, ny_owned + 2*h, nz_owned + 2*h + 1, 3)
    """

    layout: HaloLayout3D
    x: LocalBFieldGeometry  # (nx_owned + 2*h + 1, ny_owned + 2*h, nz_owned + 2*h, 3)
    y: LocalBFieldGeometry  # (nx_owned + 2*h, ny_owned + 2*h + 1, nz_owned + 2*h, 3)
    z: LocalBFieldGeometry  # (nx_owned + 2*h, ny_owned + 2*h, nz_owned + 2*h + 1, 3)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        if not isinstance(self.x, LocalBFieldGeometry):
            raise TypeError("x must be a LocalBFieldGeometry instance")
        if not isinstance(self.y, LocalBFieldGeometry):
            raise TypeError("y must be a LocalBFieldGeometry instance")
        if not isinstance(self.z, LocalBFieldGeometry):
            raise TypeError("z must be a LocalBFieldGeometry instance")
        if self.x.layout != self.layout or self.y.layout != self.layout or self.z.layout != self.layout:
            raise ValueError("LocalFaceBFieldGeometry axes must share the same HaloLayout3D")
        if self.x.location != "x_face":
            raise ValueError(f"LocalFaceBFieldGeometry.x.location must be 'x_face', got {self.x.location!r}")
        if self.y.location != "y_face":
            raise ValueError(f"LocalFaceBFieldGeometry.y.location must be 'y_face', got {self.y.location!r}")
        if self.z.location != "z_face":
            raise ValueError(f"LocalFaceBFieldGeometry.z.location must be 'z_face', got {self.z.location!r}")

    @property
    def axes(self) -> tuple[LocalBFieldGeometry, LocalBFieldGeometry, LocalBFieldGeometry]:
        return self.x, self.y, self.z

    @property
    def shape(self) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        return self.x.shape, self.y.shape, self.z.shape


@_pytree_base
@dataclass(frozen=True)
class LocalRegularFaceGeometry3D(_DataclassPyTreeMixin):
    """Shard-local regular face measures for conservative fluxes.

    Design notes
    ------------
    Unlike the reconstruction and metric objects, this is intentionally
    owned-face only. Conservative flux operators only need the faces that
    belong to the shard, so we do not store halo-padded face measures here.
    The shared `HaloLayout3D` still lives on the object so the owned face
    shapes remain tied to the shard decomposition in one place.
    """

    layout: HaloLayout3D
    x_area: jnp.ndarray  # (nx_owned + 1, ny_owned, nz_owned)
    y_area: jnp.ndarray  # (nx_owned, ny_owned + 1, nz_owned)
    z_area: jnp.ndarray  # (nx_owned, ny_owned, nz_owned + 1)
    x_area_fraction: jnp.ndarray  # (nx_owned + 1, ny_owned, nz_owned)
    y_area_fraction: jnp.ndarray  # (nx_owned, ny_owned + 1, nz_owned)
    z_area_fraction: jnp.ndarray  # (nx_owned, ny_owned, nz_owned + 1)
    x_open_mask: jnp.ndarray  # (nx_owned + 1, ny_owned, nz_owned)
    y_open_mask: jnp.ndarray  # (nx_owned, ny_owned + 1, nz_owned)
    z_open_mask: jnp.ndarray  # (nx_owned, ny_owned, nz_owned + 1)
    # Offset, in logical coordinates, from the coordinate-face center to the
    # centroid of the remaining open regular face.  This is zero for ordinary
    # uncut faces.  Embedded cut cells can set it when a solid clips only part
    # of a regular face; conservative face-gradient reconstruction then uses
    # the actual open-face centroid instead of the full coordinate face center.
    x_centroid_offset: jnp.ndarray | None = None  # (nx_owned + 1, ny_owned, nz_owned, 3)
    y_centroid_offset: jnp.ndarray | None = None  # (nx_owned, ny_owned + 1, nz_owned, 3)
    z_centroid_offset: jnp.ndarray | None = None  # (nx_owned, ny_owned, nz_owned + 1, 3)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")

        expected_x = self.layout.face_control_shape(0)
        expected_y = self.layout.face_control_shape(1)
        expected_z = self.layout.face_control_shape(2)

        x_area = jnp.asarray(self.x_area, dtype=jnp.float64)
        y_area = jnp.asarray(self.y_area, dtype=jnp.float64)
        z_area = jnp.asarray(self.z_area, dtype=jnp.float64)
        if x_area.shape != expected_x or y_area.shape != expected_y or z_area.shape != expected_z:
            raise ValueError(
                "LocalRegularFaceGeometry3D face shapes must match the owned face layout; "
                f"expected x={expected_x}, y={expected_y}, z={expected_z}, got "
                f"x={x_area.shape}, y={y_area.shape}, z={z_area.shape}"
            )

        object.__setattr__(self, "x_area", x_area)
        object.__setattr__(self, "y_area", y_area)
        object.__setattr__(self, "z_area", z_area)
        object.__setattr__(self, "x_area_fraction", _require_float_shape(self.x_area_fraction, expected_x, "LocalRegularFaceGeometry3D.x_area_fraction"))
        object.__setattr__(self, "y_area_fraction", _require_float_shape(self.y_area_fraction, expected_y, "LocalRegularFaceGeometry3D.y_area_fraction"))
        object.__setattr__(self, "z_area_fraction", _require_float_shape(self.z_area_fraction, expected_z, "LocalRegularFaceGeometry3D.z_area_fraction"))
        for name, shape in (("x_open_mask", expected_x), ("y_open_mask", expected_y), ("z_open_mask", expected_z)):
            value = jnp.asarray(getattr(self, name), dtype=bool)
            if value.shape != shape:
                raise ValueError(f"LocalRegularFaceGeometry3D.{name} must have shape {shape}, got {value.shape}")
            object.__setattr__(self, name, value)
        for name, shape in (
            ("x_centroid_offset", expected_x),
            ("y_centroid_offset", expected_y),
            ("z_centroid_offset", expected_z),
        ):
            value = getattr(self, name)
            expected_offset_shape = shape + (3,)
            if value is None:
                offset = jnp.zeros(expected_offset_shape, dtype=jnp.float64)
            else:
                offset = jnp.asarray(value, dtype=jnp.float64)
                if offset.shape != expected_offset_shape:
                    raise ValueError(
                        f"LocalRegularFaceGeometry3D.{name} must have shape "
                        f"{expected_offset_shape}, got {offset.shape}"
                    )
            object.__setattr__(self, name, offset)

    @property
    def axes(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.x_area, self.y_area, self.z_area

    @property
    def centroid_offsets(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.x_centroid_offset, self.y_centroid_offset, self.z_centroid_offset

    @property
    def shape(self) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        return self.x_area.shape, self.y_area.shape, self.z_area.shape

    @property
    def local_owned_shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def local_halo_shape(self) -> tuple[int, int, int]:
        return self.layout.cell_halo_shape

    def tree_flatten(self):
        return (
            (
                self.x_area,
                self.y_area,
                self.z_area,
                self.x_area_fraction,
                self.y_area_fraction,
                self.z_area_fraction,
                self.x_open_mask,
                self.y_open_mask,
                self.z_open_mask,
                self.x_centroid_offset,
                self.y_centroid_offset,
                self.z_centroid_offset,
            ),
            self.layout,
        )

    @classmethod
    def tree_unflatten(cls, layout, children):
        names = (
            "x_area",
            "y_area",
            "z_area",
            "x_area_fraction",
            "y_area_fraction",
            "z_area_fraction",
            "x_open_mask",
            "y_open_mask",
            "z_open_mask",
            "x_centroid_offset",
            "y_centroid_offset",
            "z_centroid_offset",
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "layout", layout)
        for name, value in zip(names, children):
            object.__setattr__(instance, name, value)
        return instance


@_pytree_base
@dataclass(frozen=True)
class RegularFaceGeometry3D(_DataclassPyTreeMixin):
    """Regular coordinate-face measures for conservative fluxes."""

    x_area: jnp.ndarray
    y_area: jnp.ndarray
    z_area: jnp.ndarray
    x_area_fraction: jnp.ndarray
    y_area_fraction: jnp.ndarray
    z_area_fraction: jnp.ndarray
    x_open_mask: jnp.ndarray
    y_open_mask: jnp.ndarray
    z_open_mask: jnp.ndarray
    x_centroid_offset: jnp.ndarray | None = None
    y_centroid_offset: jnp.ndarray | None = None
    z_centroid_offset: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        x_area = jnp.asarray(self.x_area, dtype=jnp.float64)
        y_area = jnp.asarray(self.y_area, dtype=jnp.float64)
        z_area = jnp.asarray(self.z_area, dtype=jnp.float64)
        if x_area.ndim != 3 or y_area.ndim != 3 or z_area.ndim != 3:
            raise ValueError(
                "RegularFaceGeometry3D areas must be 3D arrays with face-grid shapes"
            )

        x_shape = tuple(int(v) for v in x_area.shape)
        y_shape = tuple(int(v) for v in y_area.shape)
        z_shape = tuple(int(v) for v in z_area.shape)
        cell_shape = (x_shape[0] - 1, y_shape[1] - 1, z_shape[2] - 1)
        expected_x = (cell_shape[0] + 1, cell_shape[1], cell_shape[2])
        expected_y = (cell_shape[0], cell_shape[1] + 1, cell_shape[2])
        expected_z = (cell_shape[0], cell_shape[1], cell_shape[2] + 1)
        if x_shape != expected_x or y_shape != expected_y or z_shape != expected_z:
            raise ValueError(
                "RegularFaceGeometry3D face shapes must match the face-grid layout; "
                f"expected x={expected_x}, y={expected_y}, z={expected_z}, got "
                f"x={x_shape}, y={y_shape}, z={z_shape}"
            )

        object.__setattr__(self, "x_area", x_area)
        object.__setattr__(self, "y_area", y_area)
        object.__setattr__(self, "z_area", z_area)
        object.__setattr__(self, "x_area_fraction", _require_float_shape(self.x_area_fraction, x_shape, "RegularFaceGeometry3D.x_area_fraction"))
        object.__setattr__(self, "y_area_fraction", _require_float_shape(self.y_area_fraction, y_shape, "RegularFaceGeometry3D.y_area_fraction"))
        object.__setattr__(self, "z_area_fraction", _require_float_shape(self.z_area_fraction, z_shape, "RegularFaceGeometry3D.z_area_fraction"))
        for name, shape in (("x_open_mask", x_shape), ("y_open_mask", y_shape), ("z_open_mask", z_shape)):
            value = jnp.asarray(getattr(self, name), dtype=bool)
            if value.shape != shape:
                raise ValueError(f"RegularFaceGeometry3D.{name} must have shape {shape}, got {value.shape}")
            object.__setattr__(self, name, value)
        for name, shape in (
            ("x_centroid_offset", x_shape),
            ("y_centroid_offset", y_shape),
            ("z_centroid_offset", z_shape),
        ):
            value = getattr(self, name)
            expected_offset_shape = shape + (3,)
            if value is None:
                offset = jnp.zeros(expected_offset_shape, dtype=jnp.float64)
            else:
                offset = jnp.asarray(value, dtype=jnp.float64)
                if offset.shape != expected_offset_shape:
                    raise ValueError(
                        f"RegularFaceGeometry3D.{name} must have shape "
                        f"{expected_offset_shape}, got {offset.shape}"
                    )
            object.__setattr__(self, name, offset)

    @classmethod
    def unit(cls, geometry: "FciGeometry3D") -> "RegularFaceGeometry3D":
        shape = geometry.shape
        x_shape = (shape[0] + 1, shape[1], shape[2])
        y_shape = (shape[0], shape[1] + 1, shape[2])
        z_shape = (shape[0], shape[1], shape[2] + 1)
        return cls(
            x_area=jnp.ones(x_shape, dtype=jnp.float64),
            y_area=jnp.ones(y_shape, dtype=jnp.float64),
            z_area=jnp.ones(z_shape, dtype=jnp.float64),
            x_area_fraction=jnp.ones(x_shape, dtype=jnp.float64),
            y_area_fraction=jnp.ones(y_shape, dtype=jnp.float64),
            z_area_fraction=jnp.ones(z_shape, dtype=jnp.float64),
            x_open_mask=jnp.ones(x_shape, dtype=bool),
            y_open_mask=jnp.ones(y_shape, dtype=bool),
            z_open_mask=jnp.ones(z_shape, dtype=bool),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.x_area.shape[0] - 1), int(self.y_area.shape[1] - 1), int(self.z_area.shape[2] - 1))

    @property
    def centroid_offsets(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.x_centroid_offset, self.y_centroid_offset, self.z_centroid_offset


@_pytree_base
@dataclass(frozen=True)
class CellVolumeGeometry3D(_DataclassPyTreeMixin):
    """Effective cell-volume measure for conservative operators."""

    volume: jnp.ndarray
    volume_fraction: jnp.ndarray

    def __post_init__(self) -> None:
        volume = jnp.asarray(self.volume, dtype=jnp.float64)
        if volume.ndim != 3:
            raise ValueError(f"CellVolumeGeometry3D.volume must be 3D, got {volume.shape}")
        shape = tuple(int(v) for v in volume.shape)
        object.__setattr__(self, "volume", volume)
        object.__setattr__(self, "volume_fraction", _require_float_shape(self.volume_fraction, shape, "CellVolumeGeometry3D.volume_fraction"))

    @classmethod
    def unit(cls, geometry: "FciGeometry3D") -> "CellVolumeGeometry3D":
        volume = jnp.asarray(geometry.cell_metric.J, dtype=jnp.float64)
        return cls(volume=volume, volume_fraction=jnp.ones_like(volume, dtype=jnp.float64))

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.volume.shape)


@_pytree_base
@dataclass(frozen=True)
class LocalCellVolumeGeometry3D(_DataclassPyTreeMixin):
    """Shard-local cell-volume measure for conservative operators.

    Design notes
    ------------
    This object is intentionally owned-cell only.
    Conservative operators use it to normalize or weight the cells that the
    shard owns; halo exchange is handled separately by the stencil/field
    objects, not by the volume measure itself.
    """

    layout: HaloLayout3D
    volume: jnp.ndarray  # (nx_owned, ny_owned, nz_owned)
    volume_fraction: jnp.ndarray  # (nx_owned, ny_owned, nz_owned)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        expected_shape = self.layout.owned_shape
        volume = jnp.asarray(self.volume, dtype=jnp.float64)
        if volume.ndim != 3:
            raise ValueError(f"LocalCellVolumeGeometry3D.volume must be 3D, got {volume.shape}")
        if tuple(int(v) for v in volume.shape) != expected_shape:
            raise ValueError(
                "LocalCellVolumeGeometry3D.volume must match layout.owned_shape; "
                f"got {volume.shape}, expected {expected_shape}"
            )
        object.__setattr__(self, "volume", volume)
        object.__setattr__(
            self,
            "volume_fraction",
            _require_float_shape(self.volume_fraction, expected_shape, "LocalCellVolumeGeometry3D.volume_fraction"),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def owned_shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def local_owned_shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def local_halo_shape(self) -> tuple[int, int, int]:
        return self.layout.cell_halo_shape

    @property
    def volume_owned(self) -> jnp.ndarray:
        return self.volume

    @property
    def volume_fraction_owned(self) -> jnp.ndarray:
        return self.volume_fraction


@_pytree_base
@dataclass(frozen=True)
class LocalCellAgglomeration3D(_DataclassPyTreeMixin):
    """Owned-cell agglomeration map for embedded-boundary control volumes.

    ``source_active`` marks owned storage cells whose fluid volume is merged
    into another active owned cell.  The target indices identify the active
    owner of that merged control volume.  Empty/all-false maps are a no-op.
    """

    layout: HaloLayout3D
    source_active: jnp.ndarray
    target_i: jnp.ndarray
    target_j: jnp.ndarray
    target_k: jnp.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        shape = self.layout.owned_shape
        source_active = jnp.asarray(self.source_active, dtype=bool)
        if source_active.shape != shape:
            raise ValueError(
                "LocalCellAgglomeration3D.source_active must match layout.owned_shape; "
                f"got {source_active.shape}, expected {shape}"
            )
        target_i = _require_shape(self.target_i, shape, "LocalCellAgglomeration3D.target_i").astype(jnp.int32)
        target_j = _require_shape(self.target_j, shape, "LocalCellAgglomeration3D.target_j").astype(jnp.int32)
        target_k = _require_shape(self.target_k, shape, "LocalCellAgglomeration3D.target_k").astype(jnp.int32)
        valid = (
            (~source_active)
            | (
                (target_i >= 0)
                & (target_i < shape[0])
                & (target_j >= 0)
                & (target_j < shape[1])
                & (target_k >= 0)
                & (target_k < shape[2])
            )
        )
        try:
            all_valid = bool(jnp.all(valid))
        except jax.errors.TracerBoolConversionError:
            all_valid = True
        if not all_valid:
            raise ValueError("active agglomeration sources must map to owned target cells")

        object.__setattr__(self, "source_active", source_active)
        object.__setattr__(self, "target_i", jnp.where(source_active, target_i, 0))
        object.__setattr__(self, "target_j", jnp.where(source_active, target_j, 0))
        object.__setattr__(self, "target_k", jnp.where(source_active, target_k, 0))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def has_sources(self) -> bool:
        return bool(jnp.any(self.source_active))

    @classmethod
    def empty(cls, layout: HaloLayout3D) -> "LocalCellAgglomeration3D":
        shape = layout.owned_shape
        return cls(
            layout=layout,
            source_active=jnp.zeros(shape, dtype=bool),
            target_i=jnp.zeros(shape, dtype=jnp.int32),
            target_j=jnp.zeros(shape, dtype=jnp.int32),
            target_k=jnp.zeros(shape, dtype=jnp.int32),
        )


@_pytree_base
@dataclass(frozen=True)
class LocalAggregateCellGeometry3D(_DataclassPyTreeMixin):
    """Owned aggregate-control-volume metadata for reconstruction.

    ``centroid`` is the logical fluid/control-volume centroid associated with
    each owned storage cell after local agglomeration.  Non-target source cells
    keep a finite centroid for shape consistency, but are marked inactive by
    ``source_active`` and should not receive operator outputs.
    """

    layout: HaloLayout3D
    source_active: jnp.ndarray
    target_i: jnp.ndarray
    target_j: jnp.ndarray
    target_k: jnp.ndarray
    is_agglomerated_target: jnp.ndarray
    raw_volume: jnp.ndarray
    aggregate_volume: jnp.ndarray
    centroid: jnp.ndarray
    source_count: jnp.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError("layout must be a HaloLayout3D instance")
        shape = self.layout.owned_shape
        vector_shape = shape + (3,)
        source_active = _require_shape(
            self.source_active,
            shape,
            "LocalAggregateCellGeometry3D.source_active",
        ).astype(bool)
        target_i = _require_shape(self.target_i, shape, "LocalAggregateCellGeometry3D.target_i").astype(jnp.int32)
        target_j = _require_shape(self.target_j, shape, "LocalAggregateCellGeometry3D.target_j").astype(jnp.int32)
        target_k = _require_shape(self.target_k, shape, "LocalAggregateCellGeometry3D.target_k").astype(jnp.int32)
        is_agglomerated_target = _require_shape(
            self.is_agglomerated_target,
            shape,
            "LocalAggregateCellGeometry3D.is_agglomerated_target",
        ).astype(bool)
        raw_volume = _require_float_shape(
            self.raw_volume,
            shape,
            "LocalAggregateCellGeometry3D.raw_volume",
        )
        aggregate_volume = _require_float_shape(
            self.aggregate_volume,
            shape,
            "LocalAggregateCellGeometry3D.aggregate_volume",
        )
        centroid = _require_float_shape(
            self.centroid,
            vector_shape,
            "LocalAggregateCellGeometry3D.centroid",
        )
        source_count = _require_shape(
            self.source_count,
            shape,
            "LocalAggregateCellGeometry3D.source_count",
        ).astype(jnp.int32)

        object.__setattr__(self, "source_active", source_active)
        object.__setattr__(self, "target_i", jnp.where(source_active, target_i, 0))
        object.__setattr__(self, "target_j", jnp.where(source_active, target_j, 0))
        object.__setattr__(self, "target_k", jnp.where(source_active, target_k, 0))
        object.__setattr__(self, "is_agglomerated_target", is_agglomerated_target)
        object.__setattr__(self, "raw_volume", raw_volume)
        object.__setattr__(self, "aggregate_volume", aggregate_volume)
        object.__setattr__(self, "centroid", centroid)
        object.__setattr__(self, "source_count", source_count)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def has_agglomeration(self) -> bool:
        return bool(jnp.any(self.source_active) | jnp.any(self.is_agglomerated_target))

    @classmethod
    def empty(
        cls,
        layout: HaloLayout3D,
        centroid: jnp.ndarray | None = None,
        raw_volume: jnp.ndarray | None = None,
    ) -> "LocalAggregateCellGeometry3D":
        shape = layout.owned_shape
        if centroid is None:
            centroid = jnp.zeros(shape + (3,), dtype=jnp.float64)
        if raw_volume is None:
            raw_volume = jnp.ones(shape, dtype=jnp.float64)
        return cls(
            layout=layout,
            source_active=jnp.zeros(shape, dtype=bool),
            target_i=jnp.zeros(shape, dtype=jnp.int32),
            target_j=jnp.zeros(shape, dtype=jnp.int32),
            target_k=jnp.zeros(shape, dtype=jnp.int32),
            is_agglomerated_target=jnp.zeros(shape, dtype=bool),
            raw_volume=raw_volume,
            aggregate_volume=raw_volume,
            centroid=centroid,
            source_count=jnp.ones(shape, dtype=jnp.int32),
        )


@_pytree_base
@dataclass(frozen=True)
class LocalControlVolumeCellGeometry3D(_DataclassPyTreeMixin):
    """Authoritative owned-cell geometry for embedded control volumes.

    Storage cells either own themselves or map directly to one owned active
    control volume.  ``raw_*`` fields describe the fluid portion of each
    storage cell before merging.  ``aggregate_*`` fields are meaningful on
    active owners and contain the volume-weighted union of all mapped members.

    Second moments are central logical-coordinate moments,

    ``M2 = (1 / V) integral J (xi - centroid) (xi - centroid)^T dxi``.

    Keeping the owner map and the moments in one object prevents reconstruction
    and conservative flux paths from assigning different meanings to the same
    stored finite-volume value.
    """

    layout: HaloLayout3D
    owner_i: jnp.ndarray
    owner_j: jnp.ndarray
    owner_k: jnp.ndarray
    is_merged_source: jnp.ndarray
    is_active_owner: jnp.ndarray
    is_aggregate_target: jnp.ndarray
    received_source_count: jnp.ndarray
    member_count: jnp.ndarray
    raw_volume: jnp.ndarray
    aggregate_volume: jnp.ndarray
    raw_centroid: jnp.ndarray
    centroid: jnp.ndarray
    raw_second_moment: jnp.ndarray
    second_moment: jnp.ndarray
    raw_third_moment: jnp.ndarray | None = None
    third_moment: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout, HaloLayout3D):
            raise TypeError(
                "LocalControlVolumeCellGeometry3D.layout must be a HaloLayout3D"
            )
        shape = self.layout.owned_shape
        vector_shape = shape + (3,)
        tensor_shape = shape + (3, 3)
        third_tensor_shape = shape + (3, 3, 3)
        owner_i = _require_shape(
            self.owner_i,
            shape,
            "LocalControlVolumeCellGeometry3D.owner_i",
        ).astype(jnp.int32)
        owner_j = _require_shape(
            self.owner_j,
            shape,
            "LocalControlVolumeCellGeometry3D.owner_j",
        ).astype(jnp.int32)
        owner_k = _require_shape(
            self.owner_k,
            shape,
            "LocalControlVolumeCellGeometry3D.owner_k",
        ).astype(jnp.int32)
        is_merged_source = _require_shape(
            self.is_merged_source,
            shape,
            "LocalControlVolumeCellGeometry3D.is_merged_source",
        ).astype(bool)
        is_active_owner = _require_shape(
            self.is_active_owner,
            shape,
            "LocalControlVolumeCellGeometry3D.is_active_owner",
        ).astype(bool)
        is_aggregate_target = _require_shape(
            self.is_aggregate_target,
            shape,
            "LocalControlVolumeCellGeometry3D.is_aggregate_target",
        ).astype(bool)
        received_source_count = _require_shape(
            self.received_source_count,
            shape,
            "LocalControlVolumeCellGeometry3D.received_source_count",
        ).astype(jnp.int32)
        member_count = _require_shape(
            self.member_count,
            shape,
            "LocalControlVolumeCellGeometry3D.member_count",
        ).astype(jnp.int32)
        raw_volume = _require_float_shape(
            self.raw_volume,
            shape,
            "LocalControlVolumeCellGeometry3D.raw_volume",
        )
        aggregate_volume = _require_float_shape(
            self.aggregate_volume,
            shape,
            "LocalControlVolumeCellGeometry3D.aggregate_volume",
        )
        raw_centroid = _require_float_shape(
            self.raw_centroid,
            vector_shape,
            "LocalControlVolumeCellGeometry3D.raw_centroid",
        )
        centroid = _require_float_shape(
            self.centroid,
            vector_shape,
            "LocalControlVolumeCellGeometry3D.centroid",
        )
        raw_second_moment = _require_float_shape(
            self.raw_second_moment,
            tensor_shape,
            "LocalControlVolumeCellGeometry3D.raw_second_moment",
        )
        second_moment = _require_float_shape(
            self.second_moment,
            tensor_shape,
            "LocalControlVolumeCellGeometry3D.second_moment",
        )
        raw_third_moment = _require_float_shape(
            jnp.zeros(third_tensor_shape, dtype=jnp.float64)
            if self.raw_third_moment is None
            else self.raw_third_moment,
            third_tensor_shape,
            "LocalControlVolumeCellGeometry3D.raw_third_moment",
        )
        third_moment = _require_float_shape(
            jnp.zeros(third_tensor_shape, dtype=jnp.float64)
            if self.third_moment is None
            else self.third_moment,
            third_tensor_shape,
            "LocalControlVolumeCellGeometry3D.third_moment",
        )

        in_bounds = (
            (owner_i >= 0)
            & (owner_i < shape[0])
            & (owner_j >= 0)
            & (owner_j < shape[1])
            & (owner_k >= 0)
            & (owner_k < shape[2])
        )
        try:
            all_in_bounds = bool(jnp.all(in_bounds))
            no_owner_source_overlap = bool(
                jnp.all(~(is_merged_source & is_active_owner))
            )
            target_semantics_valid = bool(
                jnp.all(is_aggregate_target == (received_source_count > 0))
            )
            positive_owner_volume = bool(
                jnp.all((~is_active_owner) | (aggregate_volume > 0.0))
            )
            finite_active_moments = bool(
                jnp.all(
                    (~is_active_owner)
                    | (
                        jnp.all(jnp.isfinite(centroid), axis=-1)
                        & jnp.all(
                            jnp.isfinite(second_moment),
                            axis=(-2, -1),
                        )
                        & jnp.all(jnp.isfinite(third_moment), axis=(-3, -2, -1))
                    )
                )
            )
        except jax.errors.TracerBoolConversionError:
            all_in_bounds = True
            no_owner_source_overlap = True
            target_semantics_valid = True
            positive_owner_volume = True
            finite_active_moments = True
        if not all_in_bounds:
            raise ValueError("all control-volume owners must be local owned cells")
        if not no_owner_source_overlap:
            raise ValueError("a merged source cannot also be an active owner")
        if not target_semantics_valid:
            raise ValueError(
                "is_aggregate_target must equal received_source_count > 0"
            )
        if not positive_owner_volume:
            raise ValueError("every active control-volume owner must have positive volume")
        if not finite_active_moments:
            raise ValueError("active control-volume moments must be finite")

        object.__setattr__(self, "owner_i", owner_i)
        object.__setattr__(self, "owner_j", owner_j)
        object.__setattr__(self, "owner_k", owner_k)
        object.__setattr__(self, "is_merged_source", is_merged_source)
        object.__setattr__(self, "is_active_owner", is_active_owner)
        object.__setattr__(self, "is_aggregate_target", is_aggregate_target)
        object.__setattr__(self, "received_source_count", received_source_count)
        object.__setattr__(self, "member_count", member_count)
        object.__setattr__(self, "raw_volume", jnp.maximum(raw_volume, 0.0))
        object.__setattr__(
            self,
            "aggregate_volume",
            jnp.where(is_active_owner, aggregate_volume, 0.0),
        )
        object.__setattr__(self, "raw_centroid", raw_centroid)
        object.__setattr__(self, "centroid", centroid)
        object.__setattr__(
            self,
            "raw_second_moment",
            0.5 * (raw_second_moment + jnp.swapaxes(raw_second_moment, -1, -2)),
        )
        object.__setattr__(
            self,
            "second_moment",
            0.5 * (second_moment + jnp.swapaxes(second_moment, -1, -2)),
        )
        # The raw quadrature and aggregation paths construct fully symmetric
        # third moments. Average all index permutations defensively here.
        permutations = (
            (0, 1, 2), (0, 2, 1), (1, 0, 2),
            (1, 2, 0), (2, 0, 1), (2, 1, 0),
        )
        object.__setattr__(
            self,
            "raw_third_moment",
            sum(jnp.transpose(raw_third_moment, (0, 1, 2) + tuple(axis + 3 for axis in perm)) for perm in permutations) / 6.0,
        )
        object.__setattr__(
            self,
            "third_moment",
            sum(jnp.transpose(third_moment, (0, 1, 2) + tuple(axis + 3 for axis in perm)) for perm in permutations) / 6.0,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.layout.owned_shape

    @property
    def active_volume(self) -> jnp.ndarray:
        return jnp.where(self.is_active_owner, self.aggregate_volume, 0.0)

    def tree_flatten(self):
        return (
            (
                self.owner_i,
                self.owner_j,
                self.owner_k,
                self.is_merged_source,
                self.is_active_owner,
                self.is_aggregate_target,
                self.received_source_count,
                self.member_count,
                self.raw_volume,
                self.aggregate_volume,
                self.raw_centroid,
                self.centroid,
                self.raw_second_moment,
                self.second_moment,
                self.raw_third_moment,
                self.third_moment,
            ),
            self.layout,
        )

    @classmethod
    def tree_unflatten(cls, layout, children):
        names = (
            "owner_i",
            "owner_j",
            "owner_k",
            "is_merged_source",
            "is_active_owner",
            "is_aggregate_target",
            "received_source_count",
            "member_count",
            "raw_volume",
            "aggregate_volume",
            "raw_centroid",
            "centroid",
            "raw_second_moment",
            "second_moment",
            "raw_third_moment",
            "third_moment",
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "layout", layout)
        for name, value in zip(names, children):
            object.__setattr__(instance, name, value)
        return instance

    @classmethod
    def identity(
        cls,
        layout: HaloLayout3D,
        *,
        volume: jnp.ndarray,
        centroid: jnp.ndarray,
        second_moment: jnp.ndarray | None = None,
        third_moment: jnp.ndarray | None = None,
        active: jnp.ndarray | None = None,
    ) -> "LocalControlVolumeCellGeometry3D":
        shape = layout.owned_shape
        i, j, k = jnp.meshgrid(
            jnp.arange(shape[0], dtype=jnp.int32),
            jnp.arange(shape[1], dtype=jnp.int32),
            jnp.arange(shape[2], dtype=jnp.int32),
            indexing="ij",
        )
        volume = _require_float_shape(
            volume,
            shape,
            "LocalControlVolumeCellGeometry3D.identity.volume",
        )
        centroid = _require_float_shape(
            centroid,
            shape + (3,),
            "LocalControlVolumeCellGeometry3D.identity.centroid",
        )
        if second_moment is None:
            second_moment = jnp.zeros(shape + (3, 3), dtype=jnp.float64)
        if third_moment is None:
            third_moment = jnp.zeros(shape + (3, 3, 3), dtype=jnp.float64)
        if active is None:
            active = volume > 0.0
        active = _require_shape(
            active,
            shape,
            "LocalControlVolumeCellGeometry3D.identity.active",
        ).astype(bool)
        return cls(
            layout=layout,
            owner_i=i,
            owner_j=j,
            owner_k=k,
            is_merged_source=jnp.zeros(shape, dtype=bool),
            is_active_owner=active,
            is_aggregate_target=jnp.zeros(shape, dtype=bool),
            received_source_count=jnp.zeros(shape, dtype=jnp.int32),
            member_count=active.astype(jnp.int32),
            raw_volume=volume,
            aggregate_volume=jnp.where(active, volume, 0.0),
            raw_centroid=centroid,
            centroid=centroid,
            raw_second_moment=second_moment,
            second_moment=second_moment,
            raw_third_moment=third_moment,
            third_moment=third_moment,
        )


def build_local_control_volume_cell_geometry(
    layout: HaloLayout3D,
    *,
    raw_volume: jnp.ndarray,
    raw_centroid: jnp.ndarray,
    raw_second_moment: jnp.ndarray,
    raw_third_moment: jnp.ndarray | None = None,
    source_active: jnp.ndarray | None = None,
    target_i: jnp.ndarray | None = None,
    target_j: jnp.ndarray | None = None,
    target_k: jnp.ndarray | None = None,
    retained_active: jnp.ndarray | None = None,
) -> LocalControlVolumeCellGeometry3D:
    """Build direct local aggregate ownership and combine fluid moments.

    The input target map is consulted only for ``source_active`` cells.  Every
    other positive-volume cell owns itself.  The function deliberately does not
    follow target chains; callers must choose non-source local targets.
    """

    if not isinstance(layout, HaloLayout3D):
        raise TypeError("layout must be a HaloLayout3D")
    shape = layout.owned_shape
    raw_volume = _require_float_shape(
        raw_volume,
        shape,
        "build_local_control_volume_cell_geometry.raw_volume",
    )
    raw_centroid = _require_float_shape(
        raw_centroid,
        shape + (3,),
        "build_local_control_volume_cell_geometry.raw_centroid",
    )
    raw_second_moment = _require_float_shape(
        raw_second_moment,
        shape + (3, 3),
        "build_local_control_volume_cell_geometry.raw_second_moment",
    )
    if raw_third_moment is None:
        raw_third_moment = jnp.zeros(shape + (3, 3, 3), dtype=jnp.float64)
    raw_third_moment = _require_float_shape(
        raw_third_moment,
        shape + (3, 3, 3),
        "build_local_control_volume_cell_geometry.raw_third_moment",
    )
    i, j, k = jnp.meshgrid(
        jnp.arange(shape[0], dtype=jnp.int32),
        jnp.arange(shape[1], dtype=jnp.int32),
        jnp.arange(shape[2], dtype=jnp.int32),
        indexing="ij",
    )
    if source_active is None:
        source_active = jnp.zeros(shape, dtype=bool)
    else:
        source_active = _require_shape(
            source_active,
            shape,
            "build_local_control_volume_cell_geometry.source_active",
        ).astype(bool)
    supplied_targets = (target_i, target_j, target_k)
    if any(value is None for value in supplied_targets):
        if not all(value is None for value in supplied_targets):
            raise ValueError("target_i, target_j, and target_k must be supplied together")
        target_i, target_j, target_k = i, j, k
    else:
        target_i = _require_shape(
            target_i,
            shape,
            "build_local_control_volume_cell_geometry.target_i",
        ).astype(jnp.int32)
        target_j = _require_shape(
            target_j,
            shape,
            "build_local_control_volume_cell_geometry.target_j",
        ).astype(jnp.int32)
        target_k = _require_shape(
            target_k,
            shape,
            "build_local_control_volume_cell_geometry.target_k",
        ).astype(jnp.int32)
    owner_i = jnp.where(source_active, target_i, i)
    owner_j = jnp.where(source_active, target_j, j)
    owner_k = jnp.where(source_active, target_k, k)
    safe_owner_i = jnp.clip(owner_i, 0, shape[0] - 1)
    safe_owner_j = jnp.clip(owner_j, 0, shape[1] - 1)
    safe_owner_k = jnp.clip(owner_k, 0, shape[2] - 1)
    target_is_source = source_active[
        safe_owner_i,
        safe_owner_j,
        safe_owner_k,
    ]
    try:
        has_chain = bool(jnp.any(source_active & target_is_source))
    except jax.errors.TracerBoolConversionError:
        has_chain = False
    if has_chain:
        raise ValueError("control-volume source targets must not be merge sources")

    positive_raw = raw_volume > 0.0
    if retained_active is None:
        retained_active = positive_raw
    else:
        retained_active = _require_shape(
            retained_active,
            shape,
            "build_local_control_volume_cell_geometry.retained_active",
        ).astype(bool)
    is_active_owner = positive_raw & retained_active & (~source_active)
    target_is_active_owner = is_active_owner[
        safe_owner_i,
        safe_owner_j,
        safe_owner_k,
    ]
    orphan_positive = positive_raw & (~source_active) & (~is_active_owner)
    invalid_source = source_active & (
        (~positive_raw) | (~target_is_active_owner)
    )
    try:
        has_orphan_positive = bool(jnp.any(orphan_positive))
        has_invalid_source = bool(jnp.any(invalid_source))
    except jax.errors.TracerBoolConversionError:
        has_orphan_positive = False
        has_invalid_source = False
    if has_orphan_positive:
        raise ValueError(
            "every positive-volume cell must be an active owner or merge source"
        )
    if has_invalid_source:
        raise ValueError(
            "every merge source must have positive volume and target a "
            "positive active owner"
        )

    moved_volume = jnp.where(source_active, raw_volume, 0.0)
    kept_volume = jnp.where(is_active_owner, raw_volume, 0.0)
    received_volume = jnp.zeros(shape, dtype=jnp.float64).at[
        safe_owner_i,
        safe_owner_j,
        safe_owner_k,
    ].add(moved_volume)
    aggregate_volume = kept_volume + received_volume

    raw_first = raw_volume[..., None] * raw_centroid
    raw_second_origin = raw_volume[..., None, None] * (
        raw_second_moment
        + raw_centroid[..., :, None] * raw_centroid[..., None, :]
    )
    raw_third_origin = raw_volume[..., None, None, None] * (
        raw_third_moment
        + raw_centroid[..., :, None, None] * raw_second_moment[..., None, :, :]
        + raw_centroid[..., None, :, None] * raw_second_moment[..., :, None, :]
        + raw_centroid[..., None, None, :] * raw_second_moment[..., :, :, None]
        + raw_centroid[..., :, None, None]
        * raw_centroid[..., None, :, None]
        * raw_centroid[..., None, None, :]
    )
    kept_first = jnp.where(is_active_owner[..., None], raw_first, 0.0)
    kept_second = jnp.where(
        is_active_owner[..., None, None],
        raw_second_origin,
        0.0,
    )
    kept_third = jnp.where(
        is_active_owner[..., None, None, None], raw_third_origin, 0.0
    )
    received_first = jnp.zeros(shape + (3,), dtype=jnp.float64).at[
        safe_owner_i,
        safe_owner_j,
        safe_owner_k,
        :,
    ].add(jnp.where(source_active[..., None], raw_first, 0.0))
    received_second = jnp.zeros(shape + (3, 3), dtype=jnp.float64).at[
        safe_owner_i,
        safe_owner_j,
        safe_owner_k,
        :,
        :,
    ].add(jnp.where(source_active[..., None, None], raw_second_origin, 0.0))
    received_third = jnp.zeros(shape + (3, 3, 3), dtype=jnp.float64).at[
        safe_owner_i, safe_owner_j, safe_owner_k, :, :, :
    ].add(jnp.where(source_active[..., None, None, None], raw_third_origin, 0.0))
    aggregate_first = kept_first + received_first
    aggregate_second_origin = kept_second + received_second
    aggregate_third_origin = kept_third + received_third
    safe_volume = jnp.maximum(aggregate_volume, 1.0e-30)
    centroid = aggregate_first / safe_volume[..., None]
    second_moment = aggregate_second_origin / safe_volume[..., None, None]
    second_moment = second_moment - (
        centroid[..., :, None] * centroid[..., None, :]
    )
    centroid = jnp.where(
        aggregate_volume[..., None] > 0.0,
        centroid,
        raw_centroid,
    )
    second_moment = jnp.where(
        aggregate_volume[..., None, None] > 0.0,
        second_moment,
        raw_second_moment,
    )
    second_origin = aggregate_second_origin / safe_volume[..., None, None]
    third_origin = aggregate_third_origin / safe_volume[..., None, None, None]
    centroid_outer_second = (
        centroid[..., :, None, None] * second_origin[..., None, :, :]
        + centroid[..., None, :, None] * second_origin[..., :, None, :]
        + centroid[..., None, None, :] * second_origin[..., :, :, None]
    )
    centroid_cubed = (
        centroid[..., :, None, None]
        * centroid[..., None, :, None]
        * centroid[..., None, None, :]
    )
    third_moment = third_origin - centroid_outer_second + 2.0 * centroid_cubed
    third_moment = jnp.where(
        aggregate_volume[..., None, None, None] > 0.0,
        third_moment,
        raw_third_moment,
    )

    received_source_count = jnp.zeros(shape, dtype=jnp.int32).at[
        safe_owner_i,
        safe_owner_j,
        safe_owner_k,
    ].add(source_active.astype(jnp.int32))
    member_count = is_active_owner.astype(jnp.int32) + received_source_count
    is_aggregate_target = received_source_count > 0
    try:
        raw_volume_sum = float(jnp.sum(jnp.where(positive_raw, raw_volume, 0.0)))
        aggregate_volume_sum = float(
            jnp.sum(jnp.where(is_active_owner, aggregate_volume, 0.0))
        )
        volume_conserved = bool(
            jnp.isclose(
                raw_volume_sum,
                aggregate_volume_sum,
                rtol=5.0e-13,
                atol=5.0e-14,
            )
        )
    except (jax.errors.ConcretizationTypeError, TypeError):
        volume_conserved = True
    if not volume_conserved:
        raise ValueError(
            "control-volume ownership must conserve local fluid volume: "
            f"raw={raw_volume_sum:.16e}, aggregate={aggregate_volume_sum:.16e}"
        )

    return LocalControlVolumeCellGeometry3D(
        layout=layout,
        owner_i=owner_i,
        owner_j=owner_j,
        owner_k=owner_k,
        is_merged_source=source_active,
        is_active_owner=is_active_owner,
        is_aggregate_target=is_aggregate_target,
        received_source_count=received_source_count,
        member_count=member_count,
        raw_volume=raw_volume,
        aggregate_volume=aggregate_volume,
        raw_centroid=raw_centroid,
        centroid=centroid,
        raw_second_moment=raw_second_moment,
        second_moment=second_moment,
        raw_third_moment=raw_third_moment,
        third_moment=third_moment,
    )


def agglomerate_local_cell_volume_geometry(
    cell_volume: LocalCellVolumeGeometry3D,
    agglomeration: LocalCellAgglomeration3D | None,
) -> LocalCellVolumeGeometry3D:
    """Scatter inactive-source fluid volume into active target cells.

    The returned ``volume_fraction`` is an effective finite-volume measure:
    target cells can legitimately exceed ``1`` when they own an agglomerated
    control volume.  It is no longer just the raw geometric fraction of the
    coordinate cell in that case.
    """

    if agglomeration is None:
        return cell_volume
    if not isinstance(cell_volume, LocalCellVolumeGeometry3D):
        raise TypeError("cell_volume must be a LocalCellVolumeGeometry3D")
    if not isinstance(agglomeration, LocalCellAgglomeration3D):
        raise TypeError("agglomeration must be a LocalCellAgglomeration3D or None")
    if cell_volume.layout != agglomeration.layout:
        raise ValueError("cell_volume and agglomeration must share the same HaloLayout3D")

    source_active = jnp.asarray(agglomeration.source_active, dtype=bool)
    effective_volume = (
        jnp.asarray(cell_volume.volume, dtype=jnp.float64)
        * jnp.asarray(cell_volume.volume_fraction, dtype=jnp.float64)
    )
    moved_volume = jnp.where(source_active, effective_volume, 0.0)
    remaining_volume = jnp.where(source_active, 0.0, effective_volume)
    target_volume = jnp.zeros_like(effective_volume).at[
        agglomeration.target_i,
        agglomeration.target_j,
        agglomeration.target_k,
    ].add(moved_volume)
    agglomerated_volume = remaining_volume + target_volume
    base_volume = jnp.maximum(jnp.asarray(cell_volume.volume, dtype=jnp.float64), 1.0e-30)
    return LocalCellVolumeGeometry3D(
        layout=cell_volume.layout,
        volume=cell_volume.volume,
        volume_fraction=agglomerated_volume / base_volume,
    )


def _local_owned_cell_logical_centroids(geometry: LocalFciGeometry3D) -> jnp.ndarray:
    x = jnp.asarray(geometry.grid.x.centers_owned, dtype=jnp.float64)
    y = jnp.asarray(geometry.grid.y.centers_owned, dtype=jnp.float64)
    z = jnp.asarray(geometry.grid.z.centers_owned, dtype=jnp.float64)
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing="ij")
    return jnp.stack((xx, yy, zz), axis=-1)


def build_local_aggregate_cell_geometry(
    geometry: LocalFciGeometry3D,
    agglomeration: LocalCellAgglomeration3D | None,
    *,
    raw_volume: jnp.ndarray | None = None,
    fluid_centroid_owned: jnp.ndarray | None = None,
) -> LocalAggregateCellGeometry3D:
    """Build aggregate-control-volume metadata for owned-cell reconstruction."""

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError("geometry must be a LocalFciGeometry3D instance")
    shape = geometry.owned_shape
    if raw_volume is None:
        raw_volume = (
            jnp.asarray(geometry.cell_volume_geometry.volume, dtype=jnp.float64)
            * jnp.asarray(geometry.cell_volume_geometry.volume_fraction, dtype=jnp.float64)
        )
    else:
        raw_volume = _require_float_shape(
            raw_volume,
            shape,
            "build_local_aggregate_cell_geometry.raw_volume",
        )
    if fluid_centroid_owned is None:
        fluid_centroid_owned = _local_owned_cell_logical_centroids(geometry)
    else:
        fluid_centroid_owned = _require_float_shape(
            fluid_centroid_owned,
            shape + (3,),
            "build_local_aggregate_cell_geometry.fluid_centroid_owned",
        )
    if agglomeration is None:
        return LocalAggregateCellGeometry3D.empty(
            geometry.layout,
            centroid=fluid_centroid_owned,
            raw_volume=raw_volume,
        )
    if not isinstance(agglomeration, LocalCellAgglomeration3D):
        raise TypeError("agglomeration must be a LocalCellAgglomeration3D or None")
    if agglomeration.layout != geometry.layout:
        raise ValueError("agglomeration must share geometry.layout")

    source_active = jnp.asarray(agglomeration.source_active, dtype=bool)
    moved_volume = jnp.where(source_active, raw_volume, 0.0)
    kept_volume = jnp.where(source_active, 0.0, raw_volume)
    moved_moment = moved_volume[..., None] * fluid_centroid_owned
    kept_moment = kept_volume[..., None] * fluid_centroid_owned

    target_volume = jnp.zeros_like(raw_volume).at[
        agglomeration.target_i,
        agglomeration.target_j,
        agglomeration.target_k,
    ].add(moved_volume)
    target_moment = jnp.zeros_like(kept_moment).at[
        agglomeration.target_i,
        agglomeration.target_j,
        agglomeration.target_k,
        :,
    ].add(moved_moment)
    moved_source_count = jnp.zeros(shape, dtype=jnp.int32).at[
        agglomeration.target_i,
        agglomeration.target_j,
        agglomeration.target_k,
    ].add(source_active.astype(jnp.int32))
    source_count = moved_source_count + (~source_active).astype(jnp.int32)
    aggregate_volume = kept_volume + target_volume
    aggregate_moment = kept_moment + target_moment
    centroid = aggregate_moment / jnp.maximum(aggregate_volume[..., None], 1.0e-30)
    centroid = jnp.where(aggregate_volume[..., None] > 0.0, centroid, fluid_centroid_owned)
    return LocalAggregateCellGeometry3D(
        layout=geometry.layout,
        source_active=source_active,
        target_i=agglomeration.target_i,
        target_j=agglomeration.target_j,
        target_k=agglomeration.target_k,
        is_agglomerated_target=moved_source_count > 0,
        raw_volume=raw_volume,
        aggregate_volume=aggregate_volume,
        centroid=centroid,
        source_count=source_count,
    )


def agglomerate_owned_cell_average(
    values_owned: jnp.ndarray,
    aggregate_geometry: LocalAggregateCellGeometry3D,
) -> jnp.ndarray:
    """Average owned cell values over aggregate-control-volume metadata."""

    if not isinstance(aggregate_geometry, LocalAggregateCellGeometry3D):
        raise TypeError("aggregate_geometry must be a LocalAggregateCellGeometry3D")
    values = _require_float_shape(
        values_owned,
        aggregate_geometry.shape,
        "agglomerate_owned_cell_average.values_owned",
    )
    source_active = jnp.asarray(aggregate_geometry.source_active, dtype=bool)
    raw_volume = jnp.asarray(aggregate_geometry.raw_volume, dtype=jnp.float64)
    weighted = raw_volume * values
    moved_weighted = jnp.where(source_active, weighted, 0.0)
    kept_weighted = jnp.where(source_active, 0.0, weighted)
    target_weighted = jnp.zeros_like(weighted).at[
        aggregate_geometry.target_i,
        aggregate_geometry.target_j,
        aggregate_geometry.target_k,
    ].add(moved_weighted)
    averaged = (kept_weighted + target_weighted) / jnp.maximum(
        aggregate_geometry.aggregate_volume,
        1.0e-30,
    )
    return jnp.where(aggregate_geometry.aggregate_volume > 0.0, averaged, 0.0)


@_pytree_base
@dataclass(frozen=True)
class FciGeometry3D(_DataclassPyTreeMixin):
    """Centralized cell-centered FCI geometry payload.
    Geometry is stored only on active cell centers and physical faces.
    Computational ghost cells are not part of this object.
    Native FCI coordinates:
        (x, y, z) = (radial, poloidal, toroidal)
    """
    grid: CellCenteredGrid3D
    maps: FciMaps3D
    spacing: Spacing3D
    cell_metric: MetricGeometry
    face_metric: FaceMetricGeometry
    cell_bfield: BFieldGeometry
    face_bfield: FaceBFieldGeometry

    def __post_init__(self) -> None:
        shape = self.grid.shape
        xface_shape = (shape[0] + 1, shape[1], shape[2])
        yface_shape = (shape[0], shape[1] + 1, shape[2])
        zface_shape = (shape[0], shape[1], shape[2] + 1)
        if self.maps.shape != shape:
            raise ValueError(f"maps shape must be {shape}, got {self.maps.shape}")
        if self.spacing.shape != shape:
            raise ValueError(f"spacing shape must be {shape}, got {self.spacing.shape}")
        if self.cell_metric.shape != shape:
            raise ValueError(f"cell_metric shape must be {shape}, got {self.cell_metric.shape}")
        if self.cell_bfield.shape != shape:
            raise ValueError(f"cell_bfield shape must be {shape}, got {self.cell_bfield.shape}")
        if self.face_metric.x.shape != xface_shape:
            raise ValueError(f"face_metric.x shape must be {xface_shape}, got {self.face_metric.x.shape}")
        if self.face_metric.y.shape != yface_shape:
            raise ValueError(f"face_metric.y shape must be {yface_shape}, got {self.face_metric.y.shape}")
        if self.face_metric.z.shape != zface_shape:
            raise ValueError(f"face_metric.z shape must be {zface_shape}, got {self.face_metric.z.shape}")
        if self.face_bfield.x.shape != xface_shape:
            raise ValueError(f"face_bfield.x shape must be {xface_shape}, got {self.face_bfield.x.shape}")
        if self.face_bfield.y.shape != yface_shape:
            raise ValueError(f"face_bfield.y shape must be {yface_shape}, got {self.face_bfield.y.shape}")
        if self.face_bfield.z.shape != zface_shape:
            raise ValueError(f"face_bfield.z shape must be {zface_shape}, got {self.face_bfield.z.shape}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.grid.shape

    @property
    def logical_axis_vectors(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.grid.logical_axis_vectors

    @property
    def logical_face_vectors(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.grid.logical_face_vectors


@lru_cache(maxsize=1)
def _stencil_types():
    from ..native.fci_boundaries import (
        FaceGradientStencil3D,
        ConservativeStencil3D,
        LocalStencil1D,
        LocalStencil3D,
    )

    return ConservativeStencil3D, FaceGradientStencil3D, LocalStencil1D, LocalStencil3D


def _shift_owned_slices(layout: HaloLayout3D, axis: int, offset: int) -> tuple[slice, slice, slice]:
    h = layout.halo_width
    nx, ny, nz = layout.owned_shape
    extents = [nx, ny, nz]
    start = h + offset
    stop = start + extents[axis]
    slices = [slice(h, h + ext) for ext in extents]
    slices[axis] = slice(start, stop)
    return tuple(slices)


def _local_axis_stencil_from_halo(
    values_halo: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    *,
    axis: int,
) -> "LocalStencil1D":
    ConservativeStencil3D, _, LocalStencil1D, _ = _stencil_types()
    del ConservativeStencil3D

    values_halo = jnp.asarray(values_halo, dtype=jnp.float64)
    if values_halo.shape != geometry.halo_shape:
        raise ValueError(
            "field_halo must have shape geometry.halo_shape; "
            f"got {values_halo.shape}, expected {geometry.halo_shape}"
        )

    h = geometry.layout.halo_width
    owned_shape = geometry.owned_shape
    if h < 1:
        raise ValueError("local stencil reconstruction requires halo_width >= 1")

    center = values_halo[geometry.layout.owned_slices_cell]
    minus = values_halo[_shift_owned_slices(geometry.layout, axis, -1)]
    plus = values_halo[_shift_owned_slices(geometry.layout, axis, +1)]

    grid_axis = (geometry.grid.x, geometry.grid.y, geometry.grid.z)[axis]
    centers_halo = jnp.asarray(grid_axis.centers_halo, dtype=jnp.float64)
    owned_slice = grid_axis.owned_center_slice
    owned_centers = centers_halo[owned_slice]
    lower_centers = centers_halo[slice(owned_slice.start - 1, owned_slice.stop - 1)]
    upper_centers = centers_halo[slice(owned_slice.start + 1, owned_slice.stop + 1)]

    lower_width_1d = owned_centers - lower_centers
    upper_width_1d = upper_centers - owned_centers

    if axis == 0:
        dx_min = jnp.broadcast_to(lower_width_1d[:, None, None], owned_shape)
        dx_plus = jnp.broadcast_to(upper_width_1d[:, None, None], owned_shape)
    elif axis == 1:
        dx_min = jnp.broadcast_to(lower_width_1d[None, :, None], owned_shape)
        dx_plus = jnp.broadcast_to(upper_width_1d[None, :, None], owned_shape)
    else:
        dx_min = jnp.broadcast_to(lower_width_1d[None, None, :], owned_shape)
        dx_plus = jnp.broadcast_to(upper_width_1d[None, None, :], owned_shape)

    return LocalStencil1D(center=center, minus=minus, plus=plus, dx_min=dx_min, dx_plus=dx_plus)


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


def _global_axis_stencil_from_field(
    field: jnp.ndarray,
    geometry: FciGeometry3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
) -> "ConservativeStencil3D":
    ConservativeStencil3D, FaceGradientStencil3D, LocalStencil1D, _ = _stencil_types()

    values = jnp.asarray(field, dtype=jnp.float64)
    if values.shape != geometry.shape:
        raise ValueError(f"field must have shape {geometry.shape}, got {values.shape}")

    periodic_axes = _normalize_periodic_axes(periodic_axes)

    def _face_spacing(field_spacing: jnp.ndarray, *, face_axis: int) -> jnp.ndarray:
        return _lift_cell_field_to_faces(field_spacing, axis=face_axis, periodic=periodic_axes[face_axis])

    def _face_gradient_for_axis(face_axis: int) -> jnp.ndarray:
        face_values = _lift_cell_field_to_faces(values, axis=face_axis, periodic=periodic_axes[face_axis])
        face_spacings = (
            _face_spacing(geometry.spacing.dx, face_axis=face_axis),
            _face_spacing(geometry.spacing.dy, face_axis=face_axis),
            _face_spacing(geometry.spacing.dz, face_axis=face_axis),
        )
        components = tuple(
            _first_derivative_3d(
                face_values,
                face_spacings[component],
                axis=component,
                periodic=periodic_axes[component],
            )
            for component in range(3)
        )
        return jnp.stack(components, axis=-1)

    def _axis_stencil(axis: int, grid_axis, periodic: bool) -> LocalStencil1D:
        axis_n = values.shape[axis]
        if axis_n == 1:
            repeated = jnp.broadcast_to(values, geometry.shape)
            width = jnp.asarray(grid_axis.faces[-1] - grid_axis.faces[0], dtype=jnp.float64)
            width = jnp.broadcast_to(width, geometry.shape)
            return LocalStencil1D(center=values, minus=repeated, plus=repeated, dx_min=width, dx_plus=width)

        if periodic:
            minus = jnp.concatenate(
                (
                    jnp.expand_dims(jnp.take(values, -1, axis=axis), axis=axis),
                    jnp.take(values, jnp.arange(axis_n - 1), axis=axis),
                ),
                axis=axis,
            )
            plus = jnp.concatenate(
                (
                    jnp.take(values, jnp.arange(1, axis_n), axis=axis),
                    jnp.expand_dims(jnp.take(values, 0, axis=axis), axis=axis),
                ),
                axis=axis,
            )
            period = jnp.asarray(grid_axis.faces[-1] - grid_axis.faces[0], dtype=jnp.float64)
            deltas = jnp.asarray(grid_axis.centers, dtype=jnp.float64)
            dx_min_1d = jnp.concatenate((jnp.asarray([deltas[0] - (deltas[-1] - period)], dtype=jnp.float64), deltas[1:] - deltas[:-1]))
            dx_plus_1d = jnp.concatenate((deltas[1:] - deltas[:-1], jnp.expand_dims((deltas[0] + period) - deltas[-1], axis=0)))
        else:
            first = jnp.take(values, 0, axis=axis)
            second = jnp.take(values, 1, axis=axis)
            last = jnp.take(values, -1, axis=axis)
            penultimate = jnp.take(values, -2, axis=axis)
            minus = jnp.concatenate((jnp.expand_dims(2.0 * first - second, axis=axis), jnp.take(values, jnp.arange(axis_n - 1), axis=axis)), axis=axis)
            plus = jnp.concatenate((jnp.take(values, jnp.arange(1, axis_n), axis=axis), jnp.expand_dims(2.0 * last - penultimate, axis=axis)), axis=axis)
            deltas = jnp.asarray(grid_axis.centers, dtype=jnp.float64)
            lower_width = 2.0 * jnp.asarray(grid_axis.lower_center_to_face, dtype=jnp.float64)
            upper_width = 2.0 * jnp.asarray(grid_axis.upper_center_to_face, dtype=jnp.float64)
            dx_min_1d = jnp.concatenate((jnp.expand_dims(lower_width, axis=0), deltas[1:] - deltas[:-1]))
            dx_plus_1d = jnp.concatenate((deltas[1:] - deltas[:-1], jnp.expand_dims(upper_width, axis=0)))

        if axis == 0:
            dx_min = jnp.broadcast_to(dx_min_1d[:, None, None], geometry.shape)
            dx_plus = jnp.broadcast_to(dx_plus_1d[:, None, None], geometry.shape)
        elif axis == 1:
            dx_min = jnp.broadcast_to(dx_min_1d[None, :, None], geometry.shape)
            dx_plus = jnp.broadcast_to(dx_plus_1d[None, :, None], geometry.shape)
        else:
            dx_min = jnp.broadcast_to(dx_min_1d[None, None, :], geometry.shape)
            dx_plus = jnp.broadcast_to(dx_plus_1d[None, None, :], geometry.shape)

        return LocalStencil1D(center=values, minus=minus, plus=plus, dx_min=dx_min, dx_plus=dx_plus)

    return ConservativeStencil3D(
        x=_axis_stencil(0, geometry.grid.x, periodic_axes[0]),
        y=_axis_stencil(1, geometry.grid.y, periodic_axes[1]),
        z=_axis_stencil(2, geometry.grid.z, periodic_axes[2]),
        face_grad=FaceGradientStencil3D(
            x=_face_gradient_for_axis(0),
            y=_face_gradient_for_axis(1),
            z=_face_gradient_for_axis(2),
        ),
    )


def _build_conservative_stencil_from_field(
    field_halo: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    context: StencilBuilderContext,
) -> "ConservativeStencil3D":
    ConservativeStencil3D, FaceGradientStencil3D, _, LocalStencil3D = _stencil_types()
    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError("geometry must be a LocalFciGeometry3D instance")
    if not isinstance(context, StencilBuilderContext):
        raise TypeError("context must be a StencilBuilderContext instance")
    if context.layout != geometry.layout:
        raise ValueError("geometry and context must share the same HaloLayout3D")
    if context.domain is None:
        raise ValueError("context.domain is required for the local conservative stencil builder")
    field_halo = jnp.asarray(field_halo, dtype=jnp.float64)
    if field_halo.shape != geometry.halo_shape:
        raise ValueError(
            "field_halo must match geometry.halo_shape; "
            f"got {field_halo.shape}, expected {geometry.halo_shape}"
        )
    coordinate_stencil = LocalStencil3D(
        x=_local_axis_stencil_from_halo(field_halo, geometry, axis=0),
        y=_local_axis_stencil_from_halo(field_halo, geometry, axis=1),
        z=_local_axis_stencil_from_halo(field_halo, geometry, axis=2),
    )

    cut_wall_values = context.cut_wall_values
    cut_wall_dependencies = context.cut_wall_stencil_dependencies
    if (
        cut_wall_dependencies is None
        and context.cut_wall_geometry is not None
        and context.cut_wall_bc is not None
    ):
        cut_wall_dependencies = build_local_coordinate_stencil_dependency_map_from_cut_wall_geometry(
            context.layout,
            context.cut_wall_geometry,
        )
        cut_wall_values = jnp.asarray(context.cut_wall_bc.value, dtype=field_halo.dtype)
        # Avoid interpreting Neumann/normal-flux data as Dirichlet wall values.
        # The native boundary module defines BC_DIRICHLET as 1; keep this
        # module independent from native imports and only auto-patch those rows.
        dirichlet_mask = jnp.asarray(context.cut_wall_bc.kind, dtype=jnp.int32) == 1
        local = cut_wall_dependencies.local
        cut_wall_dependencies = LocalCoordinateStencilDependencyMap3D(
            layout=cut_wall_dependencies.layout,
            local=LocalCoordinateStencilLocalDependencyTable(
                target_flat=local.target_flat,
                axis=local.axis,
                side=local.side,
                value_slot=local.value_slot,
                distance=local.distance,
                active=local.active & dirichlet_mask,
            ),
            remote=cut_wall_dependencies.remote,
        )

    if cut_wall_dependencies is not None:
        coordinate_stencil = _patch_local_coordinate_cut_wall_stencil(
            coordinate_stencil,
            context=StencilBuilderContext(
                layout=context.layout,
                domain=context.domain,
                cut_wall_stencil_dependencies=cut_wall_dependencies,
                cut_wall_values=cut_wall_values,
                cut_wall_stencil_remote_values=context.cut_wall_stencil_remote_values,
            ),
            dtype=field_halo.dtype,
        )

    return ConservativeStencil3D(
        x=coordinate_stencil.x,
        y=coordinate_stencil.y,
        z=coordinate_stencil.z,
        face_grad=_build_local_face_gradient_from_halo(
            field_halo,
            geometry,
            context.domain,
        ),
    )


def _build_local_stencil_from_field(
    field_halo: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    context: StencilBuilderContext,
) -> "LocalStencil3D":
    _, _, _, LocalStencil3D = _stencil_types()

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError("geometry must be a LocalFciGeometry3D instance")
    if not isinstance(context, StencilBuilderContext):
        raise TypeError("context must be a StencilBuilderContext instance")
    if context.layout != geometry.layout:
        raise ValueError("geometry and context must share the same HaloLayout3D")
    field_halo = jnp.asarray(field_halo, dtype=jnp.float64)
    if field_halo.shape != geometry.halo_shape:
        raise ValueError(
            "field_halo must match geometry.halo_shape; "
            f"got {field_halo.shape}, expected {geometry.halo_shape}"
        )
    stencil = LocalStencil3D(
        x=_local_axis_stencil_from_halo(field_halo, geometry, axis=0),
        y=_local_axis_stencil_from_halo(field_halo, geometry, axis=1),
        z=_local_axis_stencil_from_halo(field_halo, geometry, axis=2),
    )
    return _patch_local_coordinate_cut_wall_stencil(
        stencil,
        context=context,
        dtype=field_halo.dtype,
    )


def _normalize_coordinate_stencil_values(
    values: jnp.ndarray | None,
    *,
    required: bool,
    dtype: jnp.dtype,
    name: str,
) -> jnp.ndarray:
    if values is None:
        if required:
            raise ValueError(f"context.{name} is required for cut-wall stencil rows")
        return jnp.zeros((0,), dtype=dtype)
    values = jnp.asarray(values, dtype=dtype)
    if values.ndim != 1:
        raise ValueError(f"context.{name} must be 1D, got {values.shape}")
    return values


def _sample_coordinate_stencil_values(
    values: jnp.ndarray,
    slot: jnp.ndarray,
) -> jnp.ndarray:
    if int(values.size) == 0:
        return jnp.zeros(slot.shape, dtype=values.dtype)
    safe_slot = jnp.clip(slot, 0, int(values.size) - 1)
    return values[safe_slot]


def _validate_coordinate_stencil_value_slots(
    *,
    values: jnp.ndarray,
    slot: jnp.ndarray,
    active: jnp.ndarray,
    name: str,
) -> None:
    if int(slot.size) == 0:
        return
    if int(values.size) == 0:
        try:
            has_active = bool(jnp.any(active))
        except jax.errors.TracerBoolConversionError:
            return
        if has_active:
            raise ValueError(f"context.{name} is empty but active stencil rows exist")
        return
    valid_slot = (~active) | ((slot >= 0) & (slot < int(values.size)))
    try:
        all_valid = bool(jnp.all(valid_slot))
    except jax.errors.TracerBoolConversionError:
        return
    if not all_valid:
        raise ValueError(
            f"context.{name} does not contain every active cut-wall stencil slot"
        )


def _patch_flat_coordinate_stencil_side(
    current: jnp.ndarray,
    *,
    target_flat: jnp.ndarray,
    row_axis: jnp.ndarray,
    row_side: jnp.ndarray,
    row_active: jnp.ndarray,
    row_values: jnp.ndarray,
    patch_axis: int,
    patch_side: int,
) -> jnp.ndarray:
    flat = current.reshape((-1,))
    n = int(flat.size)
    if n == 0 or int(target_flat.size) == 0:
        return current
    safe_target = jnp.clip(target_flat, 0, n - 1)
    row_mask = row_active & (row_axis == patch_axis) & (row_side == patch_side)
    row_mask_f = row_mask.astype(row_values.dtype)
    replacement_sum = jnp.zeros_like(flat).at[safe_target].add(
        jnp.where(row_mask, row_values, 0.0)
    )
    replacement_count = jnp.zeros_like(flat).at[safe_target].add(row_mask_f)
    patched = jnp.where(
        replacement_count > 0.0,
        replacement_sum / jnp.maximum(replacement_count, 1.0),
        flat,
    )
    return patched.reshape(current.shape)


def _patch_coordinate_stencil_rows(
    stencil: "LocalStencil3D",
    *,
    target_flat: jnp.ndarray,
    axis: jnp.ndarray,
    side: jnp.ndarray,
    values: jnp.ndarray,
    distance: jnp.ndarray,
    active: jnp.ndarray,
) -> "LocalStencil3D":
    _, _, _, LocalStencil3D = _stencil_types()

    stencils = [stencil.x, stencil.y, stencil.z]
    distance = jnp.maximum(jnp.asarray(distance, dtype=values.dtype), 1.0e-30)
    for patch_axis in range(3):
        axis_stencil = stencils[patch_axis]
        minus = _patch_flat_coordinate_stencil_side(
            axis_stencil.minus,
            target_flat=target_flat,
            row_axis=axis,
            row_side=side,
            row_active=active,
            row_values=values,
            patch_axis=patch_axis,
            patch_side=0,
        )
        plus = _patch_flat_coordinate_stencil_side(
            axis_stencil.plus,
            target_flat=target_flat,
            row_axis=axis,
            row_side=side,
            row_active=active,
            row_values=values,
            patch_axis=patch_axis,
            patch_side=1,
        )
        dx_min = _patch_flat_coordinate_stencil_side(
            axis_stencil.dx_min,
            target_flat=target_flat,
            row_axis=axis,
            row_side=side,
            row_active=active,
            row_values=distance,
            patch_axis=patch_axis,
            patch_side=0,
        )
        dx_plus = _patch_flat_coordinate_stencil_side(
            axis_stencil.dx_plus,
            target_flat=target_flat,
            row_axis=axis,
            row_side=side,
            row_active=active,
            row_values=distance,
            patch_axis=patch_axis,
            patch_side=1,
        )
        stencils[patch_axis] = axis_stencil.replace(
            minus=minus,
            plus=plus,
            dx_min=dx_min,
            dx_plus=dx_plus,
        )
    return LocalStencil3D(x=stencils[0], y=stencils[1], z=stencils[2])


def _patch_local_coordinate_cut_wall_stencil(
    stencil: "LocalStencil3D",
    *,
    context: StencilBuilderContext,
    dtype: jnp.dtype,
) -> "LocalStencil3D":
    dependencies = context.cut_wall_stencil_dependencies
    if dependencies is None:
        return stencil

    local = dependencies.local
    cut_wall_values = _normalize_coordinate_stencil_values(
        context.cut_wall_values,
        required=local.max_entries > 0,
        dtype=dtype,
        name="cut_wall_values",
    )
    if local.max_entries:
        _validate_coordinate_stencil_value_slots(
            values=cut_wall_values,
            slot=local.value_slot,
            active=local.active,
            name="cut_wall_values",
        )
        local_values = _sample_coordinate_stencil_values(
            cut_wall_values,
            local.value_slot,
        )
        stencil = _patch_coordinate_stencil_rows(
            stencil,
            target_flat=local.target_flat,
            axis=local.axis,
            side=local.side,
            values=local_values,
            distance=local.distance,
            active=local.active,
        )

    remote = dependencies.remote
    if remote is not None and remote.max_entries:
        remote_values = _normalize_coordinate_stencil_values(
            context.cut_wall_stencil_remote_values,
            required=True,
            dtype=dtype,
            name="cut_wall_stencil_remote_values",
        )
        expected = (remote.max_receive_values,)
        if remote_values.shape != expected:
            raise ValueError(
                "context.cut_wall_stencil_remote_values must have shape "
                f"{expected}, got {remote_values.shape}"
            )
        _validate_coordinate_stencil_value_slots(
            values=remote_values,
            slot=remote.receive_slot,
            active=remote.active,
            name="cut_wall_stencil_remote_values",
        )
        remote_row_values = _sample_coordinate_stencil_values(
            remote_values,
            remote.receive_slot,
        )
        stencil = _patch_coordinate_stencil_rows(
            stencil,
            target_flat=remote.target_flat,
            axis=remote.axis,
            side=remote.side,
            values=remote_row_values,
            distance=remote.distance,
            active=remote.active,
        )
    return stencil


@_pytree_base
@dataclass(frozen=True)
class ConservativeStencilBuilder(_DataclassPyTreeMixin):
    """Callable adapter that delegates conservative-stencil construction to an injected function."""

    build_fn: Callable[
        [
            jnp.ndarray,
            "LocalFciGeometry3D",
            "StencilBuilderContext",
        ],
        "ConservativeStencil3D",
    ]

    def __call__(
        self,
        field_halo: jnp.ndarray,
        geometry: "LocalFciGeometry3D",
        context: "StencilBuilderContext",
    ) -> "ConservativeStencil3D":
        return self.build_fn(field_halo, geometry, context)

    def tree_flatten(self):
        return (), self.build_fn

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data)


build_conservative_stencil_from_field = ConservativeStencilBuilder(_build_conservative_stencil_from_field)


@_pytree_base
@dataclass(frozen=True)
class LocalStencilBuilder(_DataclassPyTreeMixin):
    """Callable adapter that delegates local-stencil construction to an injected function."""

    build_fn: Callable[
        [
            jnp.ndarray,
            "LocalFciGeometry3D",
            "StencilBuilderContext",
        ],
        "LocalStencil3D",
    ]

    def __call__(
        self,
        field_halo: jnp.ndarray,
        geometry: "LocalFciGeometry3D",
        context: "StencilBuilderContext",
    ) -> "LocalStencil3D":
        return self.build_fn(field_halo, geometry, context)

    def tree_flatten(self):
        return (), self.build_fn

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data)


build_local_stencil_from_field = LocalStencilBuilder(_build_local_stencil_from_field)


def _normalize_remote_receive_values(
    remote: LocalFciRemoteDependencyTable | None,
    values: jnp.ndarray | None,
    *,
    dtype: jnp.dtype,
    name: str,
) -> jnp.ndarray | None:
    if remote is None:
        if values is not None:
            values = jnp.asarray(values, dtype=dtype)
            if values.ndim != 1:
                raise ValueError(f"{name} must be 1D when supplied, got {values.shape}")
        return None

    expected = (remote.max_receive_values,)
    if values is None:
        if remote.max_receive_values == 0:
            return jnp.zeros(expected, dtype=dtype)
        raise ValueError(f"{name} is required when the direction has remote dependencies")
    values = jnp.asarray(values, dtype=dtype)
    if values.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {values.shape}")
    return values


def _normalize_local_cut_wall_values(
    cut_wall_values: jnp.ndarray | None,
    *,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    if cut_wall_values is None:
        return jnp.zeros((0,), dtype=dtype)
    values = jnp.asarray(cut_wall_values, dtype=dtype)
    if values.ndim != 1:
        raise ValueError(f"cut_wall_values must be 1D, got {values.shape}")
    return values


def _sample_local_fci_cut_wall_values(
    cut_wall_values: jnp.ndarray,
    value_slot: jnp.ndarray,
) -> jnp.ndarray:
    if int(cut_wall_values.size) == 0:
        return jnp.zeros(value_slot.shape, dtype=cut_wall_values.dtype)
    safe_slot = jnp.clip(value_slot, 0, int(cut_wall_values.size) - 1)
    return cut_wall_values[safe_slot]


def _sample_local_fci_table_rows(
    field_halo: jnp.ndarray,
    table: LocalFciLocalDependencyTable,
    cut_wall_values: jnp.ndarray,
) -> jnp.ndarray:
    nx, ny, nz = field_halo.shape
    safe_i = jnp.clip(table.source_i, 0, nx - 1)
    safe_j = jnp.clip(table.source_j, 0, ny - 1)
    safe_k = jnp.clip(table.source_k, 0, nz - 1)
    field_samples = field_halo[safe_i, safe_j, safe_k]
    wall_samples = _sample_local_fci_cut_wall_values(
        cut_wall_values,
        table.value_slot,
    )
    return jnp.where(
        table.dependency_kind == FCI_DEP_CUT_WALL,
        wall_samples,
        field_samples,
    )


def _evaluate_local_fci_direction_endpoint(
    *,
    field_halo: jnp.ndarray,
    direction: LocalFciDirectionMap,
    remote_values: jnp.ndarray | None,
    cut_wall_values: jnp.ndarray,
    n_owned: int,
) -> jnp.ndarray:
    table = direction.local
    safe_target = jnp.clip(table.target_flat, 0, n_owned - 1)
    samples = _sample_local_fci_table_rows(
        field_halo,
        table,
        cut_wall_values,
    )
    active = table.active & (table.dependency_kind != FCI_DEP_INVALID)
    values = jnp.zeros((n_owned,), dtype=field_halo.dtype)
    values = values.at[safe_target].add(
        jnp.where(active, table.weight * samples, 0.0)
    )

    remote = direction.remote
    if remote is not None and remote.max_entries:
        if remote_values is None:
            raise ValueError("remote_values are required for a remote FCI table")
        if remote.max_receive_values == 0:
            remote_samples = jnp.zeros(remote.receive_slot.shape, dtype=field_halo.dtype)
        else:
            safe_slot = jnp.clip(
                remote.receive_slot,
                0,
                remote.max_receive_values - 1,
            )
            remote_samples = remote_values[safe_slot]
        safe_remote_target = jnp.clip(remote.target_flat, 0, n_owned - 1)
        values = values.at[safe_remote_target].add(
            jnp.where(remote.active, remote.weight * remote_samples, 0.0)
        )

    return jnp.where(
        direction.target_valid.reshape((n_owned,)),
        values,
        field_halo[direction.layout.owned_slices_cell].reshape((n_owned,)),
    ).reshape(direction.layout.owned_shape)


def _build_local_fci_stencil_from_field(
    field_halo: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    context: StencilBuilderContext,
    *,
    forward_remote_values: jnp.ndarray | None = None,
    backward_remote_values: jnp.ndarray | None = None,
    cut_wall_values: jnp.ndarray | None = None,
) -> "LocalStencil1D":
    _, _, LocalStencil1D, _ = _stencil_types()

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError("geometry must be a LocalFciGeometry3D instance")
    if not isinstance(context, StencilBuilderContext):
        raise TypeError("context must be a StencilBuilderContext instance")
    if context.layout != geometry.layout:
        raise ValueError("geometry and context must share the same HaloLayout3D")

    field_halo = jnp.asarray(field_halo, dtype=jnp.float64)
    if field_halo.shape != geometry.halo_shape:
        raise ValueError(
            "field_halo must match geometry.halo_shape; "
            f"got {field_halo.shape}, expected {geometry.halo_shape}"
        )

    maps = geometry.maps
    if maps.forward.connection_length is None:
        raise ValueError("geometry.maps.forward.connection_length is required")
    if maps.backward.connection_length is None:
        raise ValueError("geometry.maps.backward.connection_length is required")

    forward_remote_values = _normalize_remote_receive_values(
        maps.forward.remote,
        forward_remote_values,
        dtype=field_halo.dtype,
        name="forward_remote_values",
    )
    backward_remote_values = _normalize_remote_receive_values(
        maps.backward.remote,
        backward_remote_values,
        dtype=field_halo.dtype,
        name="backward_remote_values",
    )
    cut_wall_values = _normalize_local_cut_wall_values(
        cut_wall_values,
        dtype=field_halo.dtype,
    )

    n_owned = int(geometry.owned_shape[0] * geometry.owned_shape[1] * geometry.owned_shape[2])
    center = field_halo[geometry.layout.owned_slices_cell]
    forward = _evaluate_local_fci_direction_endpoint(
        field_halo=field_halo,
        direction=maps.forward,
        remote_values=forward_remote_values,
        cut_wall_values=cut_wall_values,
        n_owned=n_owned,
    )
    backward = _evaluate_local_fci_direction_endpoint(
        field_halo=field_halo,
        direction=maps.backward,
        remote_values=backward_remote_values,
        cut_wall_values=cut_wall_values,
        n_owned=n_owned,
    )
    dx_plus = jnp.maximum(
        jnp.asarray(maps.forward.connection_length, dtype=field_halo.dtype),
        1.0e-30,
    )
    dx_min = jnp.maximum(
        jnp.asarray(maps.backward.connection_length, dtype=field_halo.dtype),
        1.0e-30,
    )
    valid = maps.forward.target_valid & maps.backward.target_valid
    return LocalStencil1D(
        center=center,
        minus=jnp.where(valid, backward, center),
        plus=jnp.where(valid, forward, center),
        dx_min=jnp.where(valid, dx_min, 1.0),
        dx_plus=jnp.where(valid, dx_plus, 1.0),
    )


@_pytree_base
@dataclass(frozen=True)
class LocalFciStencilBuilder(_DataclassPyTreeMixin):
    """Callable adapter that builds a second-order local FCI stencil."""

    build_fn: Callable[
        [
            jnp.ndarray,
            "LocalFciGeometry3D",
            "StencilBuilderContext",
        ],
        "LocalStencil1D",
    ]

    def __call__(
        self,
        field_halo: jnp.ndarray,
        geometry: "LocalFciGeometry3D",
        context: "StencilBuilderContext",
        *,
        forward_remote_values: jnp.ndarray | None = None,
        backward_remote_values: jnp.ndarray | None = None,
        cut_wall_values: jnp.ndarray | None = None,
    ) -> "LocalStencil1D":
        return self.build_fn(
            field_halo,
            geometry,
            context,
            forward_remote_values=forward_remote_values,
            backward_remote_values=backward_remote_values,
            cut_wall_values=cut_wall_values,
        )

    def tree_flatten(self):
        return (), self.build_fn

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data)


build_local_fci_stencil_from_field = LocalFciStencilBuilder(
    _build_local_fci_stencil_from_field
)


def _build_local_face_gradient_from_halo(
    field_halo: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
) -> "FaceGradientStencil3D":
    _, FaceGradientStencil3D, _, _ = _stencil_types()

    field_halo = jnp.asarray(field_halo, dtype=jnp.float64)
    if field_halo.shape != geometry.halo_shape:
        raise ValueError(
            "field_halo must match geometry.halo_shape; "
            f"got {field_halo.shape}, expected {geometry.halo_shape}"
        )
    if not isinstance(domain, LocalDomain3D):
        raise TypeError(
            "domain must be a LocalDomain3D instance, "
            f"got {type(domain).__name__}"
        )
    if domain.layout != geometry.layout:
        raise ValueError("geometry and domain must share the same HaloLayout3D")

    face_locations = ("x_face", "y_face", "z_face")
    expected_face_shapes = tuple(
        geometry.layout.location_owned_shape(location) for location in face_locations
    )
    spacing_fields = (
        geometry.spacing.dx_halo,
        geometry.spacing.dy_halo,
        geometry.spacing.dz_halo,
    )

    def _build_for_face_axis(face_axis: int) -> jnp.ndarray:
        face_values = _lift_cell_field_to_faces(
            field_halo,
            axis=face_axis,
            periodic=domain.periodic_axes[face_axis],
        )
        face_slices = geometry.layout.location_owned_slices(face_locations[face_axis])
        components = [
            _first_derivative_3d(
                face_values,
                _lift_cell_field_to_faces(
                    spacing_fields[component],
                    axis=face_axis,
                    periodic=domain.periodic_axes[face_axis],
                ),
                axis=component,
                periodic=domain.periodic_axes[component],
            )[face_slices]
            for component in range(3)
        ]

        axis_slice = face_slices[face_axis]
        if axis_slice.start is None or axis_slice.stop is None:
            raise ValueError("owned face slices must have finite bounds")
        lower_cell_slices = list(face_slices)
        upper_cell_slices = list(face_slices)
        lower_cell_slices[face_axis] = slice(axis_slice.start - 1, axis_slice.stop - 1)
        upper_cell_slices[face_axis] = slice(axis_slice.start, axis_slice.stop)
        grid_axis = (geometry.grid.x, geometry.grid.y, geometry.grid.z)[face_axis]
        centers_halo = jnp.asarray(grid_axis.centers_halo, dtype=jnp.float64)
        center_distance = (
            centers_halo[axis_slice.start:axis_slice.stop]
            - centers_halo[axis_slice.start - 1:axis_slice.stop - 1]
        )
        if face_axis == 0:
            center_distance = center_distance[:, None, None]
        elif face_axis == 1:
            center_distance = center_distance[None, :, None]
        else:
            center_distance = center_distance[None, None, :]
        components[face_axis] = (
            field_halo[tuple(upper_cell_slices)]
            - field_halo[tuple(lower_cell_slices)]
        ) / jnp.maximum(center_distance, 1.0e-30)
        return jnp.stack(components, axis=-1)

    face_grad = FaceGradientStencil3D(
        x=_build_for_face_axis(0),
        y=_build_for_face_axis(1),
        z=_build_for_face_axis(2),
    )
    if face_grad.x.shape[:-1] != expected_face_shapes[0]:
        raise ValueError(
            f"face_grad.x must have shape {expected_face_shapes[0] + (3,)}, got {face_grad.x.shape}"
        )
    if face_grad.y.shape[:-1] != expected_face_shapes[1]:
        raise ValueError(
            f"face_grad.y must have shape {expected_face_shapes[1] + (3,)}, got {face_grad.y.shape}"
        )
    if face_grad.z.shape[:-1] != expected_face_shapes[2]:
        raise ValueError(
            f"face_grad.z must have shape {expected_face_shapes[2] + (3,)}, got {face_grad.z.shape}"
        )
    return face_grad


@_pytree_base
@dataclass(frozen=True)
class LocalConservativeStencilBuilder(_DataclassPyTreeMixin):
    """Callable adapter that delegates local conservative-stencil construction."""

    build_fn: Callable[
        [
            jnp.ndarray,
            "LocalFciGeometry3D",
            "StencilBuilderContext",
        ],
        "ConservativeStencil3D",
    ]

    def __call__(
        self,
        field_halo: jnp.ndarray,
        geometry: "LocalFciGeometry3D",
        context: "StencilBuilderContext",
    ) -> "ConservativeStencil3D":
        return self.build_fn(field_halo, geometry, context)

    def tree_flatten(self):
        return (), self.build_fn

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data)


build_local_conservative_stencil_from_field = LocalConservativeStencilBuilder(
    _build_conservative_stencil_from_field
)


def _local_axis_plane_slice(axis: int, index: int | slice) -> tuple[object, object, object]:
    """Return a 3D slice tuple with ``index`` applied along one axis."""

    axis = int(axis)
    if axis == 0:
        return index, slice(None), slice(None)
    if axis == 1:
        return slice(None), index, slice(None)
    if axis == 2:
        return slice(None), slice(None), index
    raise ValueError(f"axis must be 0, 1, or 2, got {axis}")


def _local_halo_axis_slice(
    layout: HaloLayout3D,
    axis: int,
    owned_axis_offset: int,
) -> tuple[object, object, object]:
    """Slice a local halo field at an owned-axis-relative cell offset."""

    axis = int(axis)
    if axis < 0 or axis > 2:
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

    h = int(layout.halo_width)
    slices: list[object] = [
        slice(h, h + layout.owned_shape[0]),
        slice(h, h + layout.owned_shape[1]),
        slice(h, h + layout.owned_shape[2]),
    ]
    slices[axis] = h + int(owned_axis_offset)
    return tuple(slices)


def _three_point_first_derivative_weights(
    target: jnp.ndarray,
    first: jnp.ndarray,
    second: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return first-derivative weights at ``target`` for three coordinates."""

    target = jnp.asarray(target, dtype=jnp.float64)
    first = jnp.asarray(first, dtype=jnp.float64)
    second = jnp.asarray(second, dtype=jnp.float64)

    w_target = (2.0 * target - first - second) / (
        (target - first) * (target - second)
    )
    w_first = (target - second) / ((first - target) * (first - second))
    w_second = (target - first) / ((second - target) * (second - first))
    return w_target, w_first, w_second


def _patch_local_physical_one_sided_axis_stencil(
    stencil: "LocalStencil1D",
    field_halo: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    axis: int,
) -> "LocalStencil1D":
    """Patch physical side planes with nonuniform three-point formulas."""

    ConservativeStencil3D, _, LocalStencil1D, _ = _stencil_types()
    del ConservativeStencil3D

    axis = int(axis)
    layout = domain.layout
    if layout != geometry.layout:
        raise ValueError("geometry and domain must share the same HaloLayout3D")

    n_axis = int(layout.owned_shape[axis])
    if n_axis < 3:
        has_physical_side = (
            domain.shard_spec.lower_side_kind(axis) == SIDE_PHYSICAL
            or domain.shard_spec.upper_side_kind(axis) == SIDE_PHYSICAL
        )
        if has_physical_side:
            raise ValueError(
                "second-order one-sided derivative requires at least 3 owned "
                f"cells along physical axis {axis}; got {n_axis}"
            )
        return stencil

    field_halo = jnp.asarray(field_halo, dtype=jnp.float64)
    if field_halo.shape != layout.cell_halo_shape:
        raise ValueError(
            "field_halo must match domain.layout.cell_halo_shape; "
            f"got {field_halo.shape}, expected {layout.cell_halo_shape}"
        )

    minus = jnp.asarray(stencil.minus, dtype=jnp.float64)
    center = jnp.asarray(stencil.center, dtype=jnp.float64)
    plus = jnp.asarray(stencil.plus, dtype=jnp.float64)
    dx_min = jnp.asarray(stencil.dx_min, dtype=jnp.float64)
    dx_plus = jnp.asarray(stencil.dx_plus, dtype=jnp.float64)
    c_minus = jnp.asarray(stencil.derivative_minus_weight, dtype=jnp.float64)
    c_center = jnp.asarray(stencil.derivative_center_weight, dtype=jnp.float64)
    c_plus = jnp.asarray(stencil.derivative_plus_weight, dtype=jnp.float64)

    grid_axis = (geometry.grid.x, geometry.grid.y, geometry.grid.z)[axis]
    centers_halo = jnp.asarray(grid_axis.centers_halo, dtype=jnp.float64)
    h = int(layout.halo_width)

    lower_target = centers_halo[h]
    lower_first = centers_halo[h + 1]
    lower_second = centers_halo[h + 2]
    lower_weights = _three_point_first_derivative_weights(
        lower_target,
        lower_first,
        lower_second,
    )

    lower_plane = _local_axis_plane_slice(axis, 0)
    lower_f0 = field_halo[
        _local_halo_axis_slice(layout, axis, 0)
    ]
    lower_f1 = field_halo[
        _local_halo_axis_slice(layout, axis, 1)
    ]
    lower_f2 = field_halo[
        _local_halo_axis_slice(layout, axis, 2)
    ]
    do_lower = domain.runtime_has_physical_lower(axis)

    minus = minus.at[lower_plane].set(
        jnp.where(do_lower, lower_f2, minus[lower_plane])
    )
    center = center.at[lower_plane].set(
        jnp.where(do_lower, lower_f0, center[lower_plane])
    )
    plus = plus.at[lower_plane].set(
        jnp.where(do_lower, lower_f1, plus[lower_plane])
    )
    c_minus = c_minus.at[lower_plane].set(
        jnp.where(do_lower, lower_weights[2], c_minus[lower_plane])
    )
    c_center = c_center.at[lower_plane].set(
        jnp.where(do_lower, lower_weights[0], c_center[lower_plane])
    )
    c_plus = c_plus.at[lower_plane].set(
        jnp.where(do_lower, lower_weights[1], c_plus[lower_plane])
    )
    dx_min = dx_min.at[lower_plane].set(
        jnp.where(
            do_lower,
            jnp.abs(lower_second - lower_target),
            dx_min[lower_plane],
        )
    )
    dx_plus = dx_plus.at[lower_plane].set(
        jnp.where(
            do_lower,
            jnp.abs(lower_first - lower_target),
            dx_plus[lower_plane],
        )
    )

    upper_target = centers_halo[h + n_axis - 1]
    upper_first = centers_halo[h + n_axis - 2]
    upper_second = centers_halo[h + n_axis - 3]
    upper_weights = _three_point_first_derivative_weights(
        upper_target,
        upper_first,
        upper_second,
    )

    upper_plane = _local_axis_plane_slice(axis, n_axis - 1)
    upper_f0 = field_halo[
        _local_halo_axis_slice(layout, axis, n_axis - 1)
    ]
    upper_f1 = field_halo[
        _local_halo_axis_slice(layout, axis, n_axis - 2)
    ]
    upper_f2 = field_halo[
        _local_halo_axis_slice(layout, axis, n_axis - 3)
    ]
    do_upper = domain.runtime_has_physical_upper(axis)

    minus = minus.at[upper_plane].set(
        jnp.where(do_upper, upper_f1, minus[upper_plane])
    )
    center = center.at[upper_plane].set(
        jnp.where(do_upper, upper_f0, center[upper_plane])
    )
    plus = plus.at[upper_plane].set(
        jnp.where(do_upper, upper_f2, plus[upper_plane])
    )
    c_minus = c_minus.at[upper_plane].set(
        jnp.where(do_upper, upper_weights[1], c_minus[upper_plane])
    )
    c_center = c_center.at[upper_plane].set(
        jnp.where(do_upper, upper_weights[0], c_center[upper_plane])
    )
    c_plus = c_plus.at[upper_plane].set(
        jnp.where(do_upper, upper_weights[2], c_plus[upper_plane])
    )
    dx_min = dx_min.at[upper_plane].set(
        jnp.where(
            do_upper,
            jnp.abs(upper_first - upper_target),
            dx_min[upper_plane],
        )
    )
    dx_plus = dx_plus.at[upper_plane].set(
        jnp.where(
            do_upper,
            jnp.abs(upper_second - upper_target),
            dx_plus[upper_plane],
        )
    )

    return LocalStencil1D(
        center=center,
        minus=minus,
        plus=plus,
        dx_min=dx_min,
        dx_plus=dx_plus,
        derivative_minus_weight=c_minus,
        derivative_center_weight=c_center,
        derivative_plus_weight=c_plus,
    )


def build_local_direct_stencil_one_sided_physical_from_halo(
    field_halo: jnp.ndarray,
    geometry: LocalFciGeometry3D,
    context: StencilBuilderContext,
) -> "LocalStencil3D":
    """Build function for a one-sided physical-boundary local stencil.

    This follows the ``LocalStencilBuilder`` call signature so callers can
    construct the intermediate builder explicitly. The domain is taken from
    ``context.domain``.

    Interior, shard-interface, and topology-side cells use the normal local
    centered stencil. True regular-coordinate physical side planes are
    replaced by three-point one-sided formulas whose weights are computed from
    the local coordinate-center positions. This is intended for intermediate
    fields such as ``q = grad_parallel(f)`` after halo exchange and topology
    filling, but before physical ghost filling.
    """

    _, _, _, LocalStencil3D = _stencil_types()

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "build_local_direct_stencil_one_sided_physical_from_halo requires "
            f"LocalFciGeometry3D, got {type(geometry).__name__}"
        )
    if not isinstance(context, StencilBuilderContext):
        raise TypeError(
            "build_local_direct_stencil_one_sided_physical_from_halo requires "
            "StencilBuilderContext, "
            f"got {type(context).__name__}"
        )
    domain = context.domain
    if domain is None:
        raise ValueError(
            "build_local_direct_stencil_one_sided_physical_from_halo requires "
            "context.domain"
        )
    if domain.layout != geometry.layout:
        raise ValueError("geometry and domain must share the same HaloLayout3D")

    field_halo = jnp.asarray(field_halo, dtype=jnp.float64)
    if field_halo.shape != geometry.halo_shape:
        raise ValueError(
            "field_halo must match geometry.halo_shape; "
            f"got {field_halo.shape}, expected {geometry.halo_shape}"
        )

    centered = tuple(
        _local_axis_stencil_from_halo(field_halo, geometry, axis=axis)
        for axis in range(3)
    )
    patched = tuple(
        _patch_local_physical_one_sided_axis_stencil(
            centered[axis],
            field_halo,
            geometry,
            domain,
            axis=axis,
        )
        for axis in range(3)
    )
    return LocalStencil3D(x=patched[0], y=patched[1], z=patched[2])


def _axis_index_nd(axis: int, index: int, ndim: int) -> tuple[object, ...]:
    slices: list[object] = [slice(None)] * ndim
    slices[axis] = index
    return tuple(slices)


def _first_derivative_3d(
    values: jnp.ndarray,
    spacing: jnp.ndarray | float,
    *,
    axis: int,
    periodic: bool,
) -> jnp.ndarray:
    """Centered first derivative with periodic or second-order edge treatment."""

    values = jnp.asarray(values, dtype=jnp.float64)
    h = jnp.asarray(spacing, dtype=jnp.float64)
    if h.ndim == 0:
        h = jnp.ones_like(values) * h
    centered = (jnp.roll(values, -1, axis=axis) - jnp.roll(values, 1, axis=axis)) / jnp.maximum(2.0 * h, 1.0e-30)
    if periodic:
        return centered

    first = _axis_index_nd(axis, 0, values.ndim)
    second = _axis_index_nd(axis, 1, values.ndim)
    third = _axis_index_nd(axis, 2, values.ndim)
    last = _axis_index_nd(axis, -1, values.ndim)
    penultimate = _axis_index_nd(axis, -2, values.ndim)
    antepenultimate = _axis_index_nd(axis, -3, values.ndim)
    forward = (-3.0 * values[first] + 4.0 * values[second] - values[third]) / jnp.maximum(2.0 * h[first], 1.0e-30)
    backward = (3.0 * values[last] - 4.0 * values[penultimate] + values[antepenultimate]) / jnp.maximum(2.0 * h[last], 1.0e-30)
    return centered.at[first].set(forward).at[last].set(backward)


def build_curvature_coefficients(
    geometry: "FciGeometry3D",
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Build geometry-dependent curvature coefficients for a given geometry."""

    periodic_axes = tuple(bool(value) for value in periodic_axes)
    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    if any(periodic and axis_regular for periodic, axis_regular in zip(periodic_axes, axis_regular_axes)):
        raise ValueError(
            "periodic_axes and axis_regular_axes cannot both be True on the same axis; "
            f"got periodic_axes={periodic_axes}, axis_regular_axes={axis_regular_axes}"
        )
    if axis_regular_axes[1] or axis_regular_axes[2]:
        raise NotImplementedError(
            "axis_regular_axes currently only supports the lower x axis for curvature coefficients; "
            f"got axis_regular_axes={axis_regular_axes}"
        )
    if axis_regular_axes[0] and geometry.shape[1] % 2:
        raise ValueError("axis-regular lower-x curvature coefficients require an even poloidal grid")

    def _covariant_field(metric: MetricGeometry, bfield: BFieldGeometry) -> jnp.ndarray:
        b = jnp.asarray(bfield.B_contra, dtype=jnp.float64)
        bmag = jnp.maximum(jnp.asarray(bfield.Bmag, dtype=jnp.float64), float(b_floor))
        b_unit = b / bmag[..., None]
        return jnp.einsum("...ij,...j->...i", metric.g_cov, b_unit) / bmag[..., None]

    def _boundary_corrected_derivative(
        values: jnp.ndarray,
        spacing: jnp.ndarray,
        *,
        axis: int,
        component: int,
        periodic: bool,
        lower_face_value: jnp.ndarray,
        upper_face_value: jnp.ndarray,
        lower_center_to_face: float,
        upper_center_to_face: float,
        lower_center_to_center: float,
        upper_center_to_center: float,
        axis_regular_lower_parity: float | None = None,
    ) -> jnp.ndarray:
        deriv = _first_derivative_3d(values, spacing, axis=axis, periodic=periodic)
        if periodic:
            return deriv

        if values.shape[axis] < 3:
            raise ValueError("curvature coefficient construction requires at least 3 cells along each axis")

        lower_center = values[_axis_index_nd(axis, 0, values.ndim)]
        if axis == 0 and axis_regular_axes[0]:
            half_turn = values.shape[1] // 2
            component_parity = float(axis_regular_lower_parity) if axis_regular_lower_parity is not None else (-1.0 if int(component) == 0 else 1.0)
            lower_ghost = component_parity * jnp.roll(lower_center, shift=-half_turn, axis=0)
        else:
            lower_ghost = 2.0 * lower_face_value - lower_center
        upper_ghost = 2.0 * upper_face_value - values[_axis_index_nd(axis, -1, values.ndim)]

        lower_dx_min = jnp.asarray(2.0 * lower_center_to_face, dtype=jnp.float64)
        upper_dx_plus = jnp.asarray(2.0 * upper_center_to_face, dtype=jnp.float64)
        lower_dx_plus = jnp.asarray(lower_center_to_center, dtype=jnp.float64)
        upper_dx_min = jnp.asarray(upper_center_to_center, dtype=jnp.float64)

        def _fd(minus: jnp.ndarray, center: jnp.ndarray, plus: jnp.ndarray, dx_min: jnp.ndarray, dx_plus: jnp.ndarray) -> jnp.ndarray:
            denom = jnp.maximum(dx_min * dx_plus * (dx_min + dx_plus), 1.0e-30)
            c_minus = -dx_plus * dx_plus / denom
            c_center = (dx_plus * dx_plus - dx_min * dx_min) / denom
            c_plus = dx_min * dx_min / denom
            return c_minus * minus + c_center * center + c_plus * plus

        lower_deriv = _fd(
            lower_ghost,
            lower_center,
            values[_axis_index_nd(axis, 1, values.ndim)],
            lower_dx_min,
            lower_dx_plus,
        )
        upper_deriv = _fd(
            values[_axis_index_nd(axis, -2, values.ndim)],
            values[_axis_index_nd(axis, -1, values.ndim)],
            upper_ghost,
            upper_dx_min,
            upper_dx_plus,
        )
        return deriv.at[_axis_index_nd(axis, 0, values.ndim)].set(lower_deriv).at[_axis_index_nd(axis, -1, values.ndim)].set(upper_deriv)

    metric = geometry.cell_metric
    cell_bfield = geometry.cell_bfield
    bmag = jnp.maximum(jnp.asarray(cell_bfield.Bmag, dtype=jnp.float64), float(b_floor))
    covariant_field = _covariant_field(metric, cell_bfield)

    face_covariant_x = _covariant_field(geometry.face_metric.x, geometry.face_bfield.x)
    face_covariant_y = _covariant_field(geometry.face_metric.y, geometry.face_bfield.y)
    face_covariant_z = _covariant_field(geometry.face_metric.z, geometry.face_bfield.z)
    if axis_regular_axes[0]:
        # The collapsed lower-x face is topological rather than physical. It is
        # not used by the axis-regular lower derivative, but overwriting it keeps
        # singular face geometry from lingering in the traced computation graph.
        face_covariant_x = face_covariant_x.at[0].set(jnp.zeros_like(face_covariant_x[0]))

    x_lower_center_to_face = jnp.asarray(geometry.grid.x.lower_center_to_face, dtype=jnp.float64)
    x_upper_center_to_face = jnp.asarray(geometry.grid.x.upper_center_to_face, dtype=jnp.float64)
    x_lower_center_to_center = jnp.asarray(geometry.grid.x.center_deltas[0], dtype=jnp.float64)
    x_upper_center_to_center = jnp.asarray(geometry.grid.x.center_deltas[-1], dtype=jnp.float64)
    y_lower_center_to_face = jnp.asarray(geometry.grid.y.lower_center_to_face, dtype=jnp.float64)
    y_upper_center_to_face = jnp.asarray(geometry.grid.y.upper_center_to_face, dtype=jnp.float64)
    y_lower_center_to_center = jnp.asarray(geometry.grid.y.center_deltas[0], dtype=jnp.float64)
    y_upper_center_to_center = jnp.asarray(geometry.grid.y.center_deltas[-1], dtype=jnp.float64)
    z_lower_center_to_face = jnp.asarray(geometry.grid.z.lower_center_to_face, dtype=jnp.float64)
    z_upper_center_to_face = jnp.asarray(geometry.grid.z.upper_center_to_face, dtype=jnp.float64)
    z_lower_center_to_center = jnp.asarray(geometry.grid.z.center_deltas[0], dtype=jnp.float64)
    z_upper_center_to_center = jnp.asarray(geometry.grid.z.center_deltas[-1], dtype=jnp.float64)

    dcov_dx = jnp.stack(
        [
            _boundary_corrected_derivative(
                covariant_field[..., 0],
                geometry.spacing.dx,
                axis=0,
                component=0,
                periodic=periodic_axes[0],
                lower_face_value=face_covariant_x[0, ..., 0],
                upper_face_value=face_covariant_x[-1, ..., 0],
                lower_center_to_face=x_lower_center_to_face,
                upper_center_to_face=x_upper_center_to_face,
                lower_center_to_center=x_lower_center_to_center,
                upper_center_to_center=x_upper_center_to_center,
            ),
            _boundary_corrected_derivative(
                covariant_field[..., 1],
                geometry.spacing.dx,
                axis=0,
                component=1,
                periodic=periodic_axes[0],
                lower_face_value=face_covariant_x[0, ..., 1],
                upper_face_value=face_covariant_x[-1, ..., 1],
                lower_center_to_face=x_lower_center_to_face,
                upper_center_to_face=x_upper_center_to_face,
                lower_center_to_center=x_lower_center_to_center,
                upper_center_to_center=x_upper_center_to_center,
            ),
            _boundary_corrected_derivative(
                covariant_field[..., 2],
                geometry.spacing.dx,
                axis=0,
                component=2,
                periodic=periodic_axes[0],
                lower_face_value=face_covariant_x[0, ..., 2],
                upper_face_value=face_covariant_x[-1, ..., 2],
                lower_center_to_face=x_lower_center_to_face,
                upper_center_to_face=x_upper_center_to_face,
                lower_center_to_center=x_lower_center_to_center,
                upper_center_to_center=x_upper_center_to_center,
            ),
        ],
        axis=-1,
    )
    dcov_dy = jnp.stack(
        [
            _boundary_corrected_derivative(
                covariant_field[..., 0],
                geometry.spacing.dy,
                axis=1,
                component=0,
                periodic=periodic_axes[1],
                lower_face_value=face_covariant_y[:, 0, ..., 0],
                upper_face_value=face_covariant_y[:, -1, ..., 0],
                lower_center_to_face=y_lower_center_to_face,
                upper_center_to_face=y_upper_center_to_face,
                lower_center_to_center=y_lower_center_to_center,
                upper_center_to_center=y_upper_center_to_center,
            ),
            _boundary_corrected_derivative(
                covariant_field[..., 1],
                geometry.spacing.dy,
                axis=1,
                component=1,
                periodic=periodic_axes[1],
                lower_face_value=face_covariant_y[:, 0, ..., 1],
                upper_face_value=face_covariant_y[:, -1, ..., 1],
                lower_center_to_face=y_lower_center_to_face,
                upper_center_to_face=y_upper_center_to_face,
                lower_center_to_center=y_lower_center_to_center,
                upper_center_to_center=y_upper_center_to_center,
            ),
            _boundary_corrected_derivative(
                covariant_field[..., 2],
                geometry.spacing.dy,
                axis=1,
                component=2,
                periodic=periodic_axes[1],
                lower_face_value=face_covariant_y[:, 0, ..., 2],
                upper_face_value=face_covariant_y[:, -1, ..., 2],
                lower_center_to_face=y_lower_center_to_face,
                upper_center_to_face=y_upper_center_to_face,
                lower_center_to_center=y_lower_center_to_center,
                upper_center_to_center=y_upper_center_to_center,
            ),
        ],
        axis=-1,
    )
    dcov_dz = jnp.stack(
        [
            _boundary_corrected_derivative(
                covariant_field[..., 0],
                geometry.spacing.dz,
                axis=2,
                component=0,
                periodic=periodic_axes[2],
                lower_face_value=face_covariant_z[:, :, 0, 0],
                upper_face_value=face_covariant_z[:, :, -1, 0],
                lower_center_to_face=z_lower_center_to_face,
                upper_center_to_face=z_upper_center_to_face,
                lower_center_to_center=z_lower_center_to_center,
                upper_center_to_center=z_upper_center_to_center,
            ),
            _boundary_corrected_derivative(
                covariant_field[..., 1],
                geometry.spacing.dz,
                axis=2,
                component=1,
                periodic=periodic_axes[2],
                lower_face_value=face_covariant_z[:, :, 0, 1],
                upper_face_value=face_covariant_z[:, :, -1, 1],
                lower_center_to_face=z_lower_center_to_face,
                upper_center_to_face=z_upper_center_to_face,
                lower_center_to_center=z_lower_center_to_center,
                upper_center_to_center=z_upper_center_to_center,
            ),
            _boundary_corrected_derivative(
                covariant_field[..., 2],
                geometry.spacing.dz,
                axis=2,
                component=2,
                periodic=periodic_axes[2],
                lower_face_value=face_covariant_z[:, :, 0, 2],
                upper_face_value=face_covariant_z[:, :, -1, 2],
                lower_center_to_face=z_lower_center_to_face,
                upper_center_to_face=z_upper_center_to_face,
                lower_center_to_center=z_lower_center_to_center,
                upper_center_to_center=z_upper_center_to_center,
            ),
        ],
        axis=-1,
    )

    curl = jnp.stack(
        (
            dcov_dy[..., 2] - dcov_dz[..., 1],
            dcov_dz[..., 0] - dcov_dx[..., 2],
            dcov_dx[..., 1] - dcov_dy[..., 0],
        ),
        axis=-1,
    )
    coefficient = bmag / (2.0 * jnp.maximum(jnp.asarray(metric.J, dtype=jnp.float64), float(jacobian_floor)))
    curvature_coefficients = coefficient[..., None] * curl

    if axis_regular_axes[0]:
        rho = jnp.asarray(geometry.grid.x.centers, dtype=jnp.float64)
        theta = jnp.asarray(geometry.grid.y.centers, dtype=jnp.float64)
        rho_values = rho[:, None, None]
        rho_safe = jnp.maximum(rho_values, 1.0e-30)
        theta_values = theta[None, :, None]
        cos_theta = jnp.cos(theta_values)
        sin_theta = jnp.sin(theta_values)

        A_rho = covariant_field[..., 0]
        A_theta = covariant_field[..., 1]
        A_zeta = covariant_field[..., 2]

        A_X = A_rho * cos_theta - A_theta * sin_theta / rho_safe
        A_Y = A_rho * sin_theta + A_theta * cos_theta / rho_safe
        A_Z = A_zeta

        x_upper_face_rho = jnp.asarray(geometry.grid.x.faces[-1], dtype=jnp.float64)
        x_upper_face_rho_safe = jnp.maximum(x_upper_face_rho, 1.0e-30)
        x_upper_A_X = face_covariant_x[-1, ..., 0] * cos_theta[0] - face_covariant_x[-1, ..., 1] * sin_theta[0] / x_upper_face_rho_safe
        x_upper_A_Y = face_covariant_x[-1, ..., 0] * sin_theta[0] + face_covariant_x[-1, ..., 1] * cos_theta[0] / x_upper_face_rho_safe
        x_upper_A_Z = face_covariant_x[-1, ..., 2]

        def _axis_regular_cartesian_x_derivative(values: jnp.ndarray, upper_face_value: jnp.ndarray) -> jnp.ndarray:
            return _boundary_corrected_derivative(
                values,
                geometry.spacing.dx,
                axis=0,
                component=0,
                periodic=False,
                lower_face_value=jnp.zeros_like(upper_face_value),
                upper_face_value=upper_face_value,
                lower_center_to_face=x_lower_center_to_face,
                upper_center_to_face=x_upper_center_to_face,
                lower_center_to_center=x_lower_center_to_center,
                upper_center_to_center=x_upper_center_to_center,
                axis_regular_lower_parity=1.0,
            )

        dA_X_drho = _axis_regular_cartesian_x_derivative(A_X, x_upper_A_X)
        dA_Y_drho = _axis_regular_cartesian_x_derivative(A_Y, x_upper_A_Y)
        dA_Z_drho = _axis_regular_cartesian_x_derivative(A_Z, x_upper_A_Z)

        rho_xz_safe = jnp.maximum(rho[:, None], 1.0e-30)
        y_lower_theta = jnp.asarray(geometry.grid.y.faces[0], dtype=jnp.float64)
        y_upper_theta = jnp.asarray(geometry.grid.y.faces[-1], dtype=jnp.float64)
        y_lower_cos_theta = jnp.cos(y_lower_theta)
        y_lower_sin_theta = jnp.sin(y_lower_theta)
        y_upper_cos_theta = jnp.cos(y_upper_theta)
        y_upper_sin_theta = jnp.sin(y_upper_theta)
        y_lower_A_X = face_covariant_y[:, 0, :, 0] * y_lower_cos_theta - face_covariant_y[:, 0, :, 1] * y_lower_sin_theta / rho_xz_safe
        y_upper_A_X = face_covariant_y[:, -1, :, 0] * y_upper_cos_theta - face_covariant_y[:, -1, :, 1] * y_upper_sin_theta / rho_xz_safe
        y_lower_A_Y = face_covariant_y[:, 0, :, 0] * y_lower_sin_theta + face_covariant_y[:, 0, :, 1] * y_lower_cos_theta / rho_xz_safe
        y_upper_A_Y = face_covariant_y[:, -1, :, 0] * y_upper_sin_theta + face_covariant_y[:, -1, :, 1] * y_upper_cos_theta / rho_xz_safe
        y_lower_A_Z = face_covariant_y[:, 0, :, 2]
        y_upper_A_Z = face_covariant_y[:, -1, :, 2]

        dA_X_dtheta = _boundary_corrected_derivative(
            A_X,
            geometry.spacing.dy,
            axis=1,
            component=0,
            periodic=periodic_axes[1],
            lower_face_value=y_lower_A_X,
            upper_face_value=y_upper_A_X,
            lower_center_to_face=y_lower_center_to_face,
            upper_center_to_face=y_upper_center_to_face,
            lower_center_to_center=y_lower_center_to_center,
            upper_center_to_center=y_upper_center_to_center,
        )
        dA_Y_dtheta = _boundary_corrected_derivative(
            A_Y,
            geometry.spacing.dy,
            axis=1,
            component=1,
            periodic=periodic_axes[1],
            lower_face_value=y_lower_A_Y,
            upper_face_value=y_upper_A_Y,
            lower_center_to_face=y_lower_center_to_face,
            upper_center_to_face=y_upper_center_to_face,
            lower_center_to_center=y_lower_center_to_center,
            upper_center_to_center=y_upper_center_to_center,
        )
        dA_Z_dtheta = _boundary_corrected_derivative(
            A_Z,
            geometry.spacing.dy,
            axis=1,
            component=2,
            periodic=periodic_axes[1],
            lower_face_value=y_lower_A_Z,
            upper_face_value=y_upper_A_Z,
            lower_center_to_face=y_lower_center_to_face,
            upper_center_to_face=y_upper_center_to_face,
            lower_center_to_center=y_lower_center_to_center,
            upper_center_to_center=y_upper_center_to_center,
        )

        rho_xy_safe = jnp.maximum(rho[:, None], 1.0e-30)
        theta_xy = theta[None, :]
        cos_theta_xy = jnp.cos(theta_xy)
        sin_theta_xy = jnp.sin(theta_xy)
        z_lower_A_X = face_covariant_z[:, :, 0, 0] * cos_theta_xy - face_covariant_z[:, :, 0, 1] * sin_theta_xy / rho_xy_safe
        z_upper_A_X = face_covariant_z[:, :, -1, 0] * cos_theta_xy - face_covariant_z[:, :, -1, 1] * sin_theta_xy / rho_xy_safe
        z_lower_A_Y = face_covariant_z[:, :, 0, 0] * sin_theta_xy + face_covariant_z[:, :, 0, 1] * cos_theta_xy / rho_xy_safe
        z_upper_A_Y = face_covariant_z[:, :, -1, 0] * sin_theta_xy + face_covariant_z[:, :, -1, 1] * cos_theta_xy / rho_xy_safe
        z_lower_A_Z = face_covariant_z[:, :, 0, 2]
        z_upper_A_Z = face_covariant_z[:, :, -1, 2]

        dA_X_dzeta = _boundary_corrected_derivative(
            A_X,
            geometry.spacing.dz,
            axis=2,
            component=0,
            periodic=periodic_axes[2],
            lower_face_value=z_lower_A_X,
            upper_face_value=z_upper_A_X,
            lower_center_to_face=z_lower_center_to_face,
            upper_center_to_face=z_upper_center_to_face,
            lower_center_to_center=z_lower_center_to_center,
            upper_center_to_center=z_upper_center_to_center,
        )
        dA_Y_dzeta = _boundary_corrected_derivative(
            A_Y,
            geometry.spacing.dz,
            axis=2,
            component=1,
            periodic=periodic_axes[2],
            lower_face_value=z_lower_A_Y,
            upper_face_value=z_upper_A_Y,
            lower_center_to_face=z_lower_center_to_face,
            upper_center_to_face=z_upper_center_to_face,
            lower_center_to_center=z_lower_center_to_center,
            upper_center_to_center=z_upper_center_to_center,
        )
        inv_rho = 1.0 / rho_safe
        dA_X_dY = sin_theta * dA_X_drho + cos_theta * inv_rho * dA_X_dtheta
        dA_Y_dX = cos_theta * dA_Y_drho - sin_theta * inv_rho * dA_Y_dtheta
        dA_Z_dX = cos_theta * dA_Z_drho - sin_theta * inv_rho * dA_Z_dtheta
        dA_Z_dY = sin_theta * dA_Z_drho + cos_theta * inv_rho * dA_Z_dtheta

        cartesian_curl = jnp.stack(
            (
                dA_Z_dY - dA_Y_dzeta,
                dA_X_dzeta - dA_Z_dX,
                dA_Y_dX - dA_X_dY,
            ),
            axis=-1,
        )
        cartesian_coefficient = (
            bmag
            * rho_safe
            / (2.0 * jnp.maximum(jnp.asarray(metric.J, dtype=jnp.float64), float(jacobian_floor)))
        )
        C_X = cartesian_coefficient * cartesian_curl[..., 0]
        C_Y = cartesian_coefficient * cartesian_curl[..., 1]
        C_Z = cartesian_coefficient * cartesian_curl[..., 2]
        axis_regular_lower_coefficients = jnp.stack(
            (
                C_X * cos_theta + C_Y * sin_theta,
                (-C_X * sin_theta + C_Y * cos_theta) / rho_safe,
                C_Z,
            ),
            axis=-1,
        )
        curvature_coefficients = curvature_coefficients.at[0].set(axis_regular_lower_coefficients[0])

    return curvature_coefficients


def build_local_curvature_coefficients(
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    b_floor: float = 1.0e-30,
    jacobian_floor: float = 1.0e-30,
) -> jnp.ndarray:
    """Build owned-cell curvature coefficients from local halo geometry."""

    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError(
            "build_local_curvature_coefficients requires LocalFciGeometry3D, "
            f"got {type(geometry).__name__}"
        )
    if not isinstance(domain, LocalDomain3D):
        raise TypeError(
            "build_local_curvature_coefficients requires LocalDomain3D, "
            f"got {type(domain).__name__}"
        )
    if geometry.layout != domain.layout:
        raise ValueError("geometry and domain must share the same HaloLayout3D")

    periodic_axes = tuple(bool(value) for value in periodic_axes)
    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    if len(periodic_axes) != 3:
        raise ValueError(f"periodic_axes must have length 3, got {periodic_axes}")
    if len(axis_regular_axes) != 3:
        raise ValueError(f"axis_regular_axes must have length 3, got {axis_regular_axes}")
    if any(axis_regular_axes):
        raise NotImplementedError(
            "build_local_curvature_coefficients does not yet support "
            f"axis_regular_axes={axis_regular_axes}"
        )
    if any(periodic and axis_regular for periodic, axis_regular in zip(periodic_axes, axis_regular_axes)):
        raise ValueError(
            "periodic_axes and axis_regular_axes cannot both be True on the same axis; "
            f"got periodic_axes={periodic_axes}, axis_regular_axes={axis_regular_axes}"
        )

    h = int(geometry.layout.halo_width)
    if h < 1:
        raise ValueError("local curvature coefficients require at least one geometry halo cell")

    def _covariant_field(
        metric: LocalMetricGeometry,
        bfield: LocalBFieldGeometry,
    ) -> jnp.ndarray:
        b = jnp.asarray(bfield.B_contra_halo, dtype=jnp.float64)
        bmag = jnp.maximum(jnp.asarray(bfield.Bmag_halo, dtype=jnp.float64), float(b_floor))
        b_unit = b / bmag[..., None]
        return jnp.einsum("...ij,...j->...i", metric.g_cov, b_unit) / bmag[..., None]

    def _local_centered_derivative(
        values: jnp.ndarray,
        spacing: jnp.ndarray,
        *,
        axis: int,
    ) -> jnp.ndarray:
        plus = jnp.take(values, jnp.arange(2, values.shape[axis]), axis=axis)
        minus = jnp.take(values, jnp.arange(0, values.shape[axis] - 2), axis=axis)
        spacing_center = jnp.take(spacing, jnp.arange(1, spacing.shape[axis] - 1), axis=axis)
        return (plus - minus) / jnp.maximum(2.0 * spacing_center, 1.0e-30)

    def _fd(
        minus: jnp.ndarray,
        center: jnp.ndarray,
        plus: jnp.ndarray,
        dx_min: jnp.ndarray,
        dx_plus: jnp.ndarray,
    ) -> jnp.ndarray:
        denom = jnp.maximum(dx_min * dx_plus * (dx_min + dx_plus), 1.0e-30)
        c_minus = -dx_plus * dx_plus / denom
        c_center = (dx_plus * dx_plus - dx_min * dx_min) / denom
        c_plus = dx_min * dx_min / denom
        return c_minus * minus + c_center * center + c_plus * plus

    def _patch_physical_faces(
        deriv_halo_interior: jnp.ndarray,
        values_halo: jnp.ndarray,
        *,
        axis: int,
        lower_face_value: jnp.ndarray,
        upper_face_value: jnp.ndarray,
        grid_axis: LocalGrid1D,
    ) -> jnp.ndarray:
        n_axis = int(geometry.layout.owned_shape[axis])
        if n_axis < 1:
            raise ValueError(f"owned axis {axis} must contain at least one cell")

        def _halo_plane(index: int) -> tuple[object, ...]:
            slices: list[object] = [
                slice(h, h + n)
                for n in geometry.layout.owned_shape
            ]
            slices[axis] = int(index)
            return tuple(slices)

        lower_center = values_halo[_halo_plane(h)]
        lower_ghost = 2.0 * lower_face_value - lower_center
        lower_deriv = _fd(
            lower_ghost,
            lower_center,
            values_halo[_halo_plane(h + 1)],
            jnp.asarray(2.0 * (grid_axis.centers_halo[h] - grid_axis.faces_halo[h]), dtype=jnp.float64),
            jnp.asarray(grid_axis.centers_halo[h + 1] - grid_axis.centers_halo[h], dtype=jnp.float64),
        )

        upper_center_index = h + n_axis - 1
        upper_face_index = h + n_axis
        upper_center = values_halo[_halo_plane(upper_center_index)]
        upper_ghost = 2.0 * upper_face_value - upper_center
        upper_deriv = _fd(
            values_halo[_halo_plane(upper_center_index - 1)],
            upper_center,
            upper_ghost,
            jnp.asarray(grid_axis.centers_halo[upper_center_index] - grid_axis.centers_halo[upper_center_index - 1], dtype=jnp.float64),
            jnp.asarray(2.0 * (grid_axis.faces_halo[upper_face_index] - grid_axis.centers_halo[upper_center_index]), dtype=jnp.float64),
        )

        lower_plane = _axis_index_nd(axis, 0, deriv_halo_interior.ndim)
        upper_plane = _axis_index_nd(axis, -1, deriv_halo_interior.ndim)
        do_lower = domain.runtime_has_physical_lower(axis)
        do_upper = domain.runtime_has_physical_upper(axis)

        deriv_halo_interior = deriv_halo_interior.at[lower_plane].set(
            jnp.where(do_lower, lower_deriv, deriv_halo_interior[lower_plane])
        )
        deriv_halo_interior = deriv_halo_interior.at[upper_plane].set(
            jnp.where(do_upper, upper_deriv, deriv_halo_interior[upper_plane])
        )
        return deriv_halo_interior

    def _owned_derivative(
        values_halo: jnp.ndarray,
        spacing_halo: jnp.ndarray,
        *,
        axis: int,
        lower_face_value: jnp.ndarray,
        upper_face_value: jnp.ndarray,
    ) -> jnp.ndarray:
        if values_halo.shape[axis] < geometry.layout.owned_shape[axis] + 2:
            raise ValueError("local curvature coefficient construction requires halo values")

        deriv_halo_interior = _local_centered_derivative(
            values_halo,
            spacing_halo,
            axis=axis,
        )
        crop = [slice(h, h + n) for n in geometry.layout.owned_shape]
        crop[axis] = slice(
            h - 1,
            h - 1 + geometry.layout.owned_shape[axis],
        )
        owned = deriv_halo_interior[tuple(crop)]
        grid_axis = (geometry.grid.x, geometry.grid.y, geometry.grid.z)[axis]
        return _patch_physical_faces(
            owned,
            values_halo,
            axis=axis,
            lower_face_value=lower_face_value,
            upper_face_value=upper_face_value,
            grid_axis=grid_axis,
        )

    metric = geometry.cell_metric
    cell_bfield = geometry.cell_bfield
    bmag_owned = jnp.maximum(jnp.asarray(cell_bfield.Bmag_owned, dtype=jnp.float64), float(b_floor))
    covariant_field = _covariant_field(metric, cell_bfield)

    face_covariant_x = _covariant_field(geometry.face_metric.x, geometry.face_bfield.x)
    face_covariant_y = _covariant_field(geometry.face_metric.y, geometry.face_bfield.y)
    face_covariant_z = _covariant_field(geometry.face_metric.z, geometry.face_bfield.z)

    dx = jnp.stack(
        [
            _owned_derivative(
                covariant_field[..., component],
                geometry.spacing.dx_halo,
                axis=0,
                lower_face_value=face_covariant_x[h, h:-h, h:-h, component],
                upper_face_value=face_covariant_x[
                    h + geometry.layout.owned_shape[0],
                    h:-h,
                    h:-h,
                    component,
                ],
            )
            for component in range(3)
        ],
        axis=-1,
    )
    dy = jnp.stack(
        [
            _owned_derivative(
                covariant_field[..., component],
                geometry.spacing.dy_halo,
                axis=1,
                lower_face_value=face_covariant_y[h:-h, h, h:-h, component],
                upper_face_value=face_covariant_y[
                    h:-h,
                    h + geometry.layout.owned_shape[1],
                    h:-h,
                    component,
                ],
            )
            for component in range(3)
        ],
        axis=-1,
    )
    dz = jnp.stack(
        [
            _owned_derivative(
                covariant_field[..., component],
                geometry.spacing.dz_halo,
                axis=2,
                lower_face_value=face_covariant_z[h:-h, h:-h, h, component],
                upper_face_value=face_covariant_z[
                    h:-h,
                    h:-h,
                    h + geometry.layout.owned_shape[2],
                    component,
                ],
            )
            for component in range(3)
        ],
        axis=-1,
    )

    curl = jnp.stack(
        (
            dy[..., 2] - dz[..., 1],
            dz[..., 0] - dx[..., 2],
            dx[..., 1] - dy[..., 0],
        ),
        axis=-1,
    )
    coefficient = bmag_owned / (
        2.0 * jnp.maximum(jnp.asarray(metric.J_owned, dtype=jnp.float64), float(jacobian_floor))
    )
    return coefficient[..., None] * curl


def _physical_domain_valid_mask(
    grid: CellCenteredGrid3D,
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    *,
    periodic_axes: tuple[bool, bool, bool],
) -> jnp.ndarray:
    valid_x = jnp.isfinite(x) if periodic_axes[0] else (jnp.isfinite(x) & (x >= grid.x.faces[0]) & (x <= grid.x.faces[-1]))
    valid_y = jnp.isfinite(y) if periodic_axes[1] else (jnp.isfinite(y) & (y >= grid.y.faces[0]) & (y <= grid.y.faces[-1]))
    valid_z = jnp.isfinite(z) if periodic_axes[2] else (jnp.isfinite(z) & (z >= grid.z.faces[0]) & (z <= grid.z.faces[-1]))
    return valid_x & valid_y & valid_z


def _extend_axis_with_ghost_cells(grid_axis: Grid1D) -> jnp.ndarray:
    centers = grid_axis.centers
    faces = grid_axis.faces
    lower = 2.0 * faces[0] - centers[0]
    upper = 2.0 * faces[-1] - centers[-1]
    return jnp.concatenate([
        jnp.asarray([lower], dtype=jnp.float64),
        centers,
        jnp.asarray([upper], dtype=jnp.float64),
    ])

def _extend_values_x(values: jnp.ndarray, grid_axis: Grid1D) -> jnp.ndarray:
    centers = grid_axis.centers
    faces = grid_axis.faces
    xg_lower = 2.0 * faces[0] - centers[0]
    xg_upper = 2.0 * faces[-1] - centers[-1]

    if values.shape[0] == 1:
        lower = values[0:1, :, :]
        upper = values[-1:, :, :]
    else:
        lower_slope = (values[1, :, :] - values[0, :, :]) / (centers[1] - centers[0])
        upper_slope = (values[-1, :, :] - values[-2, :, :]) / (centers[-1] - centers[-2])

        lower = (values[0, :, :] + (xg_lower - centers[0]) * lower_slope)[None, :, :]
        upper = (values[-1, :, :] + (xg_upper - centers[-1]) * upper_slope)[None, :, :]

    return jnp.concatenate([lower, values, upper], axis=0)


def _extend_values_y(values: jnp.ndarray, grid_axis: Grid1D) -> jnp.ndarray:
    centers = grid_axis.centers
    faces = grid_axis.faces
    yg_lower = 2.0 * faces[0] - centers[0]
    yg_upper = 2.0 * faces[-1] - centers[-1]

    if values.shape[1] == 1:
        lower = values[:, 0:1, :]
        upper = values[:, -1:, :]
    else:
        lower_slope = (values[:, 1, :] - values[:, 0, :]) / (centers[1] - centers[0])
        upper_slope = (values[:, -1, :] - values[:, -2, :]) / (centers[-1] - centers[-2])

        lower = (values[:, 0, :] + (yg_lower - centers[0]) * lower_slope)[:, None, :]
        upper = (values[:, -1, :] + (yg_upper - centers[-1]) * upper_slope)[:, None, :]

    return jnp.concatenate([lower, values, upper], axis=1)


def _extend_values_z(values: jnp.ndarray, grid_axis: Grid1D) -> jnp.ndarray:
    centers = grid_axis.centers
    faces = grid_axis.faces
    zg_lower = 2.0 * faces[0] - centers[0]
    zg_upper = 2.0 * faces[-1] - centers[-1]

    if values.shape[2] == 1:
        lower = values[:, :, 0:1]
        upper = values[:, :, -1:]
    else:
        lower_slope = (values[:, :, 1] - values[:, :, 0]) / (centers[1] - centers[0])
        upper_slope = (values[:, :, -1] - values[:, :, -2]) / (centers[-1] - centers[-2])

        lower = (values[:, :, 0] + (zg_lower - centers[0]) * lower_slope)[:, :, None]
        upper = (values[:, :, -1] + (zg_upper - centers[-1]) * upper_slope)[:, :, None]

    return jnp.concatenate([lower, values, upper], axis=2)

def _interpolate_scalar_cell_centered(
    values: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    *,
    grid: CellCenteredGrid3D,
    periodic_axes: tuple[bool, bool, bool],
    boundary_value: float,
) -> jnp.ndarray:
    values = jnp.asarray(values, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)

    if values.shape != grid.shape:
        raise ValueError(f"values must have shape {grid.shape}, got {values.shape}")

    valid = _physical_domain_valid_mask(
        grid,
        x,
        y,
        z,
        periodic_axes=periodic_axes,
    )

    if periodic_axes[0]:
        x_axis = grid.x.centers
        x0, x1, wx, _ = _bracket_axis(x_axis, x, periodic=True)
    else:
        values = _extend_values_x(values, grid.x)
        x_axis = _extend_axis_with_ghost_cells(grid.x)
        x0, x1, wx, _ = _bracket_axis(x_axis, x, periodic=False)

    if periodic_axes[1]:
        y_axis = grid.y.centers
        y0, y1, wy, _ = _bracket_axis(y_axis, y, periodic=True)
    else:
        values = _extend_values_y(values, grid.y)
        y_axis = _extend_axis_with_ghost_cells(grid.y)
        y0, y1, wy, _ = _bracket_axis(y_axis, y, periodic=False)

    if periodic_axes[2]:
        z_axis = grid.z.centers
        z0, z1, wz, _ = _bracket_axis(z_axis, z, periodic=True)
    else:
        values = _extend_values_z(values, grid.z)
        z_axis = _extend_axis_with_ghost_cells(grid.z)
        z0, z1, wz, _ = _bracket_axis(z_axis, z, periodic=False)

    interpolated = _trilinear_sample(values, x0, x1, wx, y0, y1, wy, z0, z1, wz)
    return jnp.where(
        valid,
        interpolated,
        jnp.asarray(boundary_value, dtype=jnp.float64),
    )


def _interpolate_B_contravariant_cell_centered(
    grid: CellCenteredGrid3D,
    B_contra_cell: jnp.ndarray,
    points: jnp.ndarray,
    *,
    periodic_axes: tuple[bool, bool, bool],
    boundary_value: float,
) -> jnp.ndarray:
    sampled_points = jnp.asarray(points, dtype=jnp.float64)
    if sampled_points.shape == (3,):
        sampled_points = sampled_points[None, :]
        squeeze = True
    else:
        if sampled_points.ndim < 1 or sampled_points.shape[-1] != 3:
            raise ValueError(f"points must have shape (3,) or (..., 3), got {sampled_points.shape}")
        squeeze = False

    B_contra_cell = jnp.asarray(B_contra_cell, dtype=jnp.float64)
    if B_contra_cell.shape != grid.shape + (3,):
        raise ValueError(f"B_contra_cell must have shape {grid.shape + (3,)}, got {B_contra_cell.shape}")

    samples = []
    for component in range(3):
        samples.append(
            _interpolate_scalar_cell_centered(
                B_contra_cell[..., component],
                sampled_points[..., 0],
                sampled_points[..., 1],
                sampled_points[..., 2],
                grid=grid,
                periodic_axes=periodic_axes,
                boundary_value=boundary_value,
            )
        )
    result = jnp.stack(samples, axis=-1)
    return result[0] if squeeze else result


def _rk4_step_cell_centered(
    grid: CellCenteredGrid3D,
    B_contra_cell: jnp.ndarray,
    point: jnp.ndarray,
    step: float,
    *,
    periodic_axes: tuple[bool, bool, bool],
    min_abs_bz: float,
    boundary_value: float,
) -> jnp.ndarray:
    state = jnp.asarray(point, dtype=jnp.float64)
    if state.shape != (3,):
        raise ValueError(f"point must have shape (3,), got {state.shape}")
    h = float(step)
    
    def rhs(value: jnp.ndarray) -> jnp.ndarray:
        b = _interpolate_B_contravariant_cell_centered(
            grid,
            B_contra_cell,
            value,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )
        bz = jnp.asarray(b[2], dtype=jnp.float64)
        safe_bz = jnp.where(jnp.abs(bz) < min_abs_bz, jnp.where(bz < 0.0, -1.0, 1.0) * (min_abs_bz), bz)
        return jnp.stack((b[0] / safe_bz, b[1] / safe_bz, jnp.array(1.0, dtype=jnp.float64)))

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * h * k1)
    k3 = rhs(state + 0.5 * h * k2)
    k4 = rhs(state + h * k3)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _trace_fieldline_to_plane_cell_centered(
    grid: CellCenteredGrid3D,
    B_contra_cell: jnp.ndarray,
    Bmag_cell: jnp.ndarray,
    seed_points: jnp.ndarray,
    *,
    step: float,
    substeps: int,
    periodic_axes: tuple[bool, bool, bool],
    min_abs_bz: float,
    boundary_value: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    points = jnp.asarray(seed_points, dtype=jnp.float64)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"seed_points must have shape (n, 3), got {points.shape}")

    nseed = int(points.shape[0])
    step_size = float(step) / float(max(int(substeps), 1))
    length = jnp.zeros(nseed, dtype=jnp.float64)
    alive = jnp.ones(nseed, dtype=bool)
    state = points

    def _speed(sampled_b: jnp.ndarray, sampled_bmag: jnp.ndarray) -> jnp.ndarray:
        bz = sampled_b[..., 2]
        safe_bz = jnp.where(
            jnp.abs(bz) < min_abs_bz,
            jnp.where(bz < 0.0, -1.0, 1.0) * min_abs_bz,
            bz,
        )
        return jnp.asarray(sampled_bmag, dtype=jnp.float64) / jnp.maximum(jnp.abs(safe_bz), 1.0e-30)

    for _ in range(max(int(substeps), 1)):
        b0 = jax.vmap(
            lambda point: _interpolate_B_contravariant_cell_centered(
                grid,
                B_contra_cell,
                point,
                periodic_axes=periodic_axes,
                boundary_value=boundary_value,
            )
        )(state)
        bmag0 = _interpolate_scalar_cell_centered(
            Bmag_cell,
            state[:, 0],
            state[:, 1],
            state[:, 2],
            grid=grid,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )
        next_state = jax.vmap(
            lambda point: _rk4_step_cell_centered(
                grid,
                B_contra_cell,
                point,
                step_size,
                periodic_axes=periodic_axes,
                min_abs_bz=min_abs_bz,
                boundary_value=boundary_value,
            )
        )(state)
        bmag1 = _interpolate_scalar_cell_centered(
            Bmag_cell,
            next_state[:, 0],
            next_state[:, 1],
            next_state[:, 2],
            grid=grid,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )
        b1 = jax.vmap(
            lambda point: _interpolate_B_contravariant_cell_centered(
                grid,
                B_contra_cell,
                point,
                periodic_axes=periodic_axes,
                boundary_value=boundary_value,
            )
        )(next_state)
        finite = jnp.all(jnp.isfinite(next_state), axis=-1)
        valid = _physical_domain_valid_mask(
            grid,
            next_state[:, 0],
            next_state[:, 1],
            next_state[:, 2],
            periodic_axes=periodic_axes,
        )
        increment = 0.5 * abs(step_size) * (_speed(b0, bmag0) + _speed(b1, bmag1))
        increment = jnp.where(alive & finite & valid, increment, 0.0)
        length = length + increment
        state = jnp.where((alive & finite & valid)[..., None], next_state, state)
        alive = alive & finite & valid

    return state, length, ~alive


def interpolate_B_contravariant(
    geometry: FciGeometry3D,
    points: jnp.ndarray,
    *,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    boundary_value: float = jnp.nan,
) -> jnp.ndarray:
    """Trilinearly interpolate the cell-centered magnetic field at logical-space points."""

    return _interpolate_B_contravariant_cell_centered(
        geometry.grid,
        geometry.cell_bfield.B_contra,
        points,
        periodic_axes=periodic_axes,
        boundary_value=boundary_value,
    )


def build_fci_maps_from_b_contravariant(
    grid: CellCenteredGrid3D,
    B_contra_cell: jnp.ndarray,
    Bmag_cell: jnp.ndarray,
    *,
    substeps: int = 4,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    min_abs_bz: float = 1.0e-30,
    boundary_value: float = jnp.nan,
) -> dict[str, jnp.ndarray]:
    """JIT-compatible FCI map builder for a cell-centered grid.

    Improved boundary behavior:
      - Traces start from active cell centers.
      - If a trace reaches the target toroidal plane, normal FCI maps are stored.
      - If a trace exits a nonperiodic physical boundary first, the boundary
        mask is set and the endpoint is estimated at the boundary face.
      - forward_length/backward_length are distances to the actual endpoint:
          target plane if no boundary hit,
          boundary hit point if boundary=True.

    For jitting:

        build_maps_jit = jax.jit(
            build_fci_maps_from_b_contravariant,
            static_argnames=("substeps", "periodic_axes"),
        )
    """

    shape = grid.shape
    nx, ny, nz = shape
    nxy = nx * ny

    B_contra_cell = jnp.asarray(B_contra_cell, dtype=jnp.float64)
    Bmag_cell = jnp.asarray(Bmag_cell, dtype=jnp.float64)

    if B_contra_cell.shape != shape + (3,):
        raise ValueError(
            f"B_contra_cell must have shape {shape + (3,)}, got {B_contra_cell.shape}"
        )
    if Bmag_cell.shape != shape:
        raise ValueError(f"Bmag_cell must have shape {shape}, got {Bmag_cell.shape}")

    n_substeps = int(substeps)
    if n_substeps < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")

    x_axis = grid.x.centers
    y_axis = grid.y.centers
    z_axis = grid.z.centers

    xx = jnp.broadcast_to(x_axis[:, None], (nx, ny)).reshape(-1)
    yy = jnp.broadcast_to(y_axis[None, :], (nx, ny)).reshape(-1)

    periodic_x = bool(periodic_axes[0])
    periodic_y = bool(periodic_axes[1])
    periodic_z = bool(periodic_axes[2])

    min_bz = jnp.asarray(min_abs_bz, dtype=jnp.float64)

    def _wrap_periodic_coord(coord: jnp.ndarray, axis: Grid1D, periodic: bool) -> jnp.ndarray:
        if periodic:
            period = axis.faces[-1] - axis.faces[0]
            return jnp.mod(coord - axis.faces[0], period) + axis.faces[0]
        return coord

    def _wrap_points(points: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(
            (
                _wrap_periodic_coord(points[..., 0], grid.x, periodic_x),
                _wrap_periodic_coord(points[..., 1], grid.y, periodic_y),
                _wrap_periodic_coord(points[..., 2], grid.z, periodic_z),
            ),
            axis=-1,
        )

    def _plane_step_jit(k: jnp.ndarray, direction: int) -> jnp.ndarray:
        """Signed z step from plane k to neighboring plane."""

        if nz < 2:
            return jnp.asarray(direction, dtype=jnp.float64)

        dz_lower = z_axis[1] - z_axis[0]
        dz_upper = z_axis[-1] - z_axis[-2]
        period = (z_axis[-1] - z_axis[0]) + dz_lower

        if direction >= 0:
            k_next = jnp.minimum(k + 1, nz - 1)
            interior_step = z_axis[k_next] - z_axis[k]

            if periodic_z:
                boundary_step = (z_axis[0] + period) - z_axis[-1]
            else:
                boundary_step = dz_upper

            return jnp.where(k < nz - 1, interior_step, boundary_step)

        k_prev = jnp.maximum(k - 1, 0)
        interior_step = -(z_axis[k] - z_axis[k_prev])

        if periodic_z:
            boundary_step = (z_axis[-1] - period) - z_axis[0]
        else:
            boundary_step = -dz_lower

        return jnp.where(k > 0, interior_step, boundary_step)

    def _interp_B(points: jnp.ndarray) -> jnp.ndarray:
        return _interpolate_B_contravariant_cell_centered(
            grid,
            B_contra_cell,
            points,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )

    def _interp_Bmag(points: jnp.ndarray) -> jnp.ndarray:
        return _interpolate_scalar_cell_centered(
            Bmag_cell,
            points[..., 0],
            points[..., 1],
            points[..., 2],
            grid=grid,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )

    def _safe_bz(bz: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(
            jnp.abs(bz) < min_bz,
            jnp.where(bz < 0.0, -1.0, 1.0) * min_bz,
            bz,
        )

    def _rhs(points: jnp.ndarray) -> jnp.ndarray:
        b = _interp_B(points)
        bz = _safe_bz(b[..., 2])
        return jnp.stack(
            (
                b[..., 0] / bz,
                b[..., 1] / bz,
                jnp.ones_like(bz),
            ),
            axis=-1,
        )

    def _rk4_batch(points: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        k1 = _rhs(points)
        k2 = _rhs(points + 0.5 * h * k1)
        k3 = _rhs(points + 0.5 * h * k2)
        k4 = _rhs(points + h * k3)
        return points + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _speed(sampled_b: jnp.ndarray, sampled_bmag: jnp.ndarray) -> jnp.ndarray:
        bz = _safe_bz(sampled_b[..., 2])
        return sampled_bmag / jnp.maximum(jnp.abs(bz), 1.0e-30)

    def _axis_crossing_fraction(
        old: jnp.ndarray,
        new: jnp.ndarray,
        lower: jnp.ndarray,
        upper: jnp.ndarray,
        periodic: bool,
    ) -> jnp.ndarray:
        """Fraction along old->new where a nonperiodic axis hits a boundary.

        Returns inf if this axis does not produce a boundary crossing.
        """

        if periodic:
            return jnp.full_like(old, jnp.inf, dtype=jnp.float64)

        denom = new - old
        safe_denom = jnp.where(jnp.abs(denom) < 1.0e-300, 1.0, denom)

        crosses_lower = new < lower
        crosses_upper = new > upper

        t_lower = (lower - old) / safe_denom
        t_upper = (upper - old) / safe_denom

        t = jnp.where(crosses_lower, t_lower, jnp.inf)
        t = jnp.minimum(t, jnp.where(crosses_upper, t_upper, jnp.inf))

        valid_t = (t >= 0.0) & (t <= 1.0)
        return jnp.where(valid_t, t, jnp.inf)

    def _boundary_hit_state(
        old_state: jnp.ndarray,
        new_state: jnp.ndarray,
        finite_new: jnp.ndarray,
        valid_new: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Estimate physical boundary hit location between old_state and new_state.

        Assumes old_state is inside the domain. If new_state leaves the domain,
        compute the first face intersection along the straight RK substep chord.
        """

        tx = _axis_crossing_fraction(
            old_state[:, 0],
            new_state[:, 0],
            grid.x.faces[0],
            grid.x.faces[-1],
            periodic_x,
        )
        ty = _axis_crossing_fraction(
            old_state[:, 1],
            new_state[:, 1],
            grid.y.faces[0],
            grid.y.faces[-1],
            periodic_y,
        )
        tz = _axis_crossing_fraction(
            old_state[:, 2],
            new_state[:, 2],
            grid.z.faces[0],
            grid.z.faces[-1],
            periodic_z,
        )

        t_hit = jnp.minimum(jnp.minimum(tx, ty), tz)
        has_hit = finite_new & (~valid_new) & jnp.isfinite(t_hit)

        t_used = jnp.where(has_hit, t_hit, 1.0)
        hit_state = old_state + t_used[:, None] * (new_state - old_state)

        # Clamp nonperiodic hit coordinates exactly to the face-bounded domain.
        hx = hit_state[:, 0] if periodic_x else jnp.clip(hit_state[:, 0], grid.x.faces[0], grid.x.faces[-1])
        hy = hit_state[:, 1] if periodic_y else jnp.clip(hit_state[:, 1], grid.y.faces[0], grid.y.faces[-1])
        hz = hit_state[:, 2] if periodic_z else jnp.clip(hit_state[:, 2], grid.z.faces[0], grid.z.faces[-1])

        hit_state = jnp.stack((hx, hy, hz), axis=-1)
        hit_state = _wrap_points(hit_state)

        return hit_state, jnp.where(has_hit, t_hit, 1.0)

    def _trace_to_plane(seed_points: jnp.ndarray, step: jnp.ndarray):
        step_size = step / jnp.asarray(n_substeps, dtype=jnp.float64)

        init_state = seed_points
        init_length = jnp.zeros(seed_points.shape[0], dtype=jnp.float64)
        init_alive = jnp.ones(seed_points.shape[0], dtype=bool)
        init_boundary = jnp.zeros(seed_points.shape[0], dtype=bool)

        def substep_body(carry, _):
            state, length, alive, boundary = carry

            b0 = _interp_B(state)
            bmag0 = _interp_Bmag(state)
            speed0 = _speed(b0, bmag0)

            raw_next_state = _rk4_batch(state, step_size)
            finite_next = jnp.all(jnp.isfinite(raw_next_state), axis=-1)

            # For nonfinite results, keep the old state to avoid NaN pollution.
            next_state_finite = jnp.where(finite_next[:, None], raw_next_state, state)

            valid_next = _physical_domain_valid_mask(
                grid,
                next_state_finite[:, 0],
                next_state_finite[:, 1],
                next_state_finite[:, 2],
                periodic_axes=periodic_axes,
            )

            active_full = alive & finite_next & valid_next
            active_exit = alive & finite_next & (~valid_next)
            active_bad = alive & (~finite_next)

            hit_state, t_hit = _boundary_hit_state(
                state,
                next_state_finite,
                finite_next,
                valid_next,
            )

            b1 = _interp_B(next_state_finite)
            bmag1 = _interp_Bmag(next_state_finite)
            speed1 = _speed(b1, bmag1)

            b_hit = _interp_B(hit_state)
            bmag_hit = _interp_Bmag(hit_state)
            speed_hit = _speed(b_hit, bmag_hit)

            full_increment = 0.5 * jnp.abs(step_size) * (speed0 + speed1)
            hit_increment = 0.5 * jnp.abs(step_size) * t_hit * (speed0 + speed_hit)

            increment = jnp.where(active_full, full_increment, 0.0)
            increment = increment + jnp.where(active_exit, hit_increment, 0.0)

            new_length = length + increment

            endpoint_state = jnp.where(active_full[:, None], next_state_finite, state)
            endpoint_state = jnp.where(active_exit[:, None], hit_state, endpoint_state)
            endpoint_state = _wrap_points(endpoint_state)

            new_alive = alive & finite_next & valid_next
            new_boundary = boundary | active_exit | active_bad

            return (endpoint_state, new_length, new_alive, new_boundary), None

        final_state, final_length, final_alive, final_boundary = jax.lax.scan(
            substep_body,
            (init_state, init_length, init_alive, init_boundary),
            xs=None,
            length=n_substeps,
        )[0]

        del final_alive
        return final_state, final_length, final_boundary

    def plane_body(_carry, k):
        z_k = jnp.full((nxy,), z_axis[k], dtype=jnp.float64)
        seed_points = jnp.stack((xx, yy, z_k), axis=-1)

        forward_step = _plane_step_jit(k, direction=1)
        backward_step = _plane_step_jit(k, direction=-1)

        forward_points, forward_length, forward_boundary = _trace_to_plane(
            seed_points,
            forward_step,
        )
        backward_points, backward_length, backward_boundary = _trace_to_plane(
            seed_points,
            backward_step,
        )

        forward_x = _logical_coordinate_to_index(
            x_axis,
            forward_points[:, 0],
            periodic=periodic_x,
        )
        forward_y = _logical_coordinate_to_index(
            y_axis,
            forward_points[:, 1],
            periodic=periodic_y,
        )
        backward_x = _logical_coordinate_to_index(
            x_axis,
            backward_points[:, 0],
            periodic=periodic_x,
        )
        backward_y = _logical_coordinate_to_index(
            y_axis,
            backward_points[:, 1],
            periodic=periodic_y,
        )

        dz_plane = jnp.full((nxy,), jnp.abs(forward_step), dtype=jnp.float64)

        outputs = (
            forward_x,
            forward_y,
            backward_x,
            backward_y,
            forward_points[:, 0],
            forward_points[:, 1],
            forward_points[:, 2],
            backward_points[:, 0],
            backward_points[:, 1],
            backward_points[:, 2],
            forward_length,
            backward_length,
            forward_boundary,
            backward_boundary,
            dz_plane,
        )

        return None, outputs

    _, scanned = jax.lax.scan(
        plane_body,
        None,
        jnp.arange(nz),
    )

    (
        forward_x_k,
        forward_y_k,
        backward_x_k,
        backward_y_k,
        forward_endpoint_x_k,
        forward_endpoint_y_k,
        forward_endpoint_z_k,
        backward_endpoint_x_k,
        backward_endpoint_y_k,
        backward_endpoint_z_k,
        forward_length_k,
        backward_length_k,
        forward_boundary_k,
        backward_boundary_k,
        dz_k,
    ) = scanned

    def _planes_to_grid(arr_k_nxy: jnp.ndarray) -> jnp.ndarray:
        return jnp.swapaxes(arr_k_nxy, 0, 1).reshape(shape)

    return {
        "forward_x": _planes_to_grid(forward_x_k),
        "forward_y": _planes_to_grid(forward_y_k),
        "backward_x": _planes_to_grid(backward_x_k),
        "backward_y": _planes_to_grid(backward_y_k),
        "forward_endpoint_x": _planes_to_grid(forward_endpoint_x_k),
        "forward_endpoint_y": _planes_to_grid(forward_endpoint_y_k),
        "forward_endpoint_z": _planes_to_grid(forward_endpoint_z_k),
        "backward_endpoint_x": _planes_to_grid(backward_endpoint_x_k),
        "backward_endpoint_y": _planes_to_grid(backward_endpoint_y_k),
        "backward_endpoint_z": _planes_to_grid(backward_endpoint_z_k),
        "forward_length": _planes_to_grid(forward_length_k),
        "backward_length": _planes_to_grid(backward_length_k),
        "forward_boundary": _planes_to_grid(forward_boundary_k),
        "backward_boundary": _planes_to_grid(backward_boundary_k),
        "dz": _planes_to_grid(dz_k),
    }

def _bracket_axis(
    axis: jnp.ndarray,
    values: jnp.ndarray,
    *,
    periodic: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    axis = jnp.asarray(axis, dtype=jnp.float64)
    values = jnp.asarray(values, dtype=jnp.float64)
    if axis.size == 1:
        zero = jnp.zeros_like(values, dtype=jnp.int32)
        return zero, zero, jnp.zeros_like(values, dtype=jnp.float64), jnp.isfinite(values)
    if axis.size < 1:
        raise ValueError("Each logical axis must contain at least one point for interpolation.")

    if periodic:
        spacing = axis[1] - axis[0]
        period = (axis[-1] - axis[0]) + spacing
        wrapped = jnp.mod(values - axis[0], period) + axis[0]
        upper = jnp.searchsorted(axis, wrapped, side="right")
        lower = jnp.clip(upper - 1, 0, int(axis.size) - 1)
        next_index = jnp.mod(lower + 1, int(axis.size))
        lower_coord = axis[lower]
        upper_coord = jnp.where(lower == int(axis.size) - 1, axis[0] + period, axis[next_index])
        weight = (wrapped - lower_coord) / (upper_coord - lower_coord)
        valid = jnp.isfinite(values)
        return lower, next_index, jnp.clip(weight, 0.0, 1.0), valid

    upper = jnp.searchsorted(axis, values, side="right")
    lower = jnp.clip(upper - 1, 0, int(axis.size) - 2)
    upper = lower + 1
    lower_coord = axis[lower]
    upper_coord = axis[upper]
    weight = (values - lower_coord) / (upper_coord - lower_coord)
    valid = (values >= axis[0]) & (values <= axis[-1]) & jnp.isfinite(values)
    return lower, upper, jnp.clip(weight, 0.0, 1.0), valid


def _trilinear_sample(
    values: jnp.ndarray,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    wx: jnp.ndarray,
    y0: jnp.ndarray,
    y1: jnp.ndarray,
    wy: jnp.ndarray,
    z0: jnp.ndarray,
    z1: jnp.ndarray,
    wz: jnp.ndarray,
) -> jnp.ndarray:
    c000 = values[x0, y0, z0]
    c100 = values[x1, y0, z0]
    c010 = values[x0, y1, z0]
    c110 = values[x1, y1, z0]
    c001 = values[x0, y0, z1]
    c101 = values[x1, y0, z1]
    c011 = values[x0, y1, z1]
    c111 = values[x1, y1, z1]
    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return c0 * (1.0 - wz) + c1 * wz


def _logical_coordinate_to_index(
    axis: jnp.ndarray,
    values: jnp.ndarray,
    *,
    periodic: bool,
) -> jnp.ndarray:
    """Convert logical coordinate values to fractional grid indices."""

    axis = jnp.asarray(axis, dtype=jnp.float64)
    values = jnp.asarray(values, dtype=jnp.float64)
    if axis.size < 2:
        return jnp.zeros_like(values, dtype=jnp.float64)

    if periodic:
        spacing = axis[1] - axis[0]
        period = (axis[-1] - axis[0]) + spacing
        wrapped = jnp.mod(values - axis[0], period) + axis[0]
        upper = jnp.searchsorted(axis, wrapped, side="right")
        lower = jnp.clip(upper - 1, 0, int(axis.size) - 1)
        next_index = jnp.mod(lower + 1, int(axis.size))
        lower_coord = axis[lower]
        upper_coord = jnp.where(lower == int(axis.size) - 1, axis[0] + period, axis[next_index])
        weight = (wrapped - lower_coord) / (upper_coord - lower_coord)
        return jnp.asarray(lower, dtype=jnp.float64) + jnp.clip(weight, 0.0, 1.0)

    upper = jnp.searchsorted(axis, values, side="right")
    lower = jnp.clip(upper - 1, 0, int(axis.size) - 2)
    upper = lower + 1
    lower_coord = axis[lower]
    upper_coord = axis[upper]
    weight = (values - lower_coord) / (upper_coord - lower_coord)
    return jnp.asarray(lower, dtype=jnp.float64) + jnp.clip(weight, 0.0, 1.0)


def logical_b_contravariant_from_geometry(geometry: FciGeometry3D) -> jnp.ndarray:
    """Return the stored cell-centered contravariant magnetic field."""

    return geometry.cell_bfield.B_contra


def logical_b_contravariant_from_traced_maps(
    forward_x: jnp.ndarray,
    forward_y: jnp.ndarray,
    backward_x: jnp.ndarray,
    backward_y: jnp.ndarray,
    forward_length: jnp.ndarray,
    backward_length: jnp.ndarray,
    *,
    dz: jnp.ndarray,
) -> jnp.ndarray:
    """Reconstruct a contravariant field direction from traced field-line maps."""

    forward_x = jnp.asarray(forward_x, dtype=jnp.float64)
    forward_y = jnp.asarray(forward_y, dtype=jnp.float64)
    backward_x = jnp.asarray(backward_x, dtype=jnp.float64)
    backward_y = jnp.asarray(backward_y, dtype=jnp.float64)
    forward_length = jnp.asarray(forward_length, dtype=jnp.float64)
    backward_length = jnp.asarray(backward_length, dtype=jnp.float64)
    dz = jnp.asarray(dz, dtype=jnp.float64)

    shape = forward_x.shape
    if not (
        forward_y.shape == shape
        and backward_x.shape == shape
        and backward_y.shape == shape
        and forward_length.shape == shape
        and backward_length.shape == shape
        and dz.shape == shape
    ):
        raise ValueError("All traced-map arrays must have the same shape")
    if len(shape) != 3:
        raise ValueError(f"traced maps must have shape (nx, ny, nz), got {shape}")

    def _centered_delta(upper: jnp.ndarray, lower: jnp.ndarray, extent: int) -> jnp.ndarray:
        delta = upper - lower
        half_extent = 0.5 * float(extent)
        delta = jnp.where(delta > half_extent, delta - float(extent), delta)
        delta = jnp.where(delta < -half_extent, delta + float(extent), delta)
        return delta

    # Centered logical-space displacement between the forward and backward
    # plane intersections. The overall scale is arbitrary because only the
    # direction is used downstream.
    dx = 0.5 * _centered_delta(forward_x, backward_x, shape[0])
    dy = 0.5 * _centered_delta(forward_y, backward_y, shape[1])
    dz_safe = jnp.where(jnp.abs(dz) < 1.0e-30, 1.0, dz)

    return jnp.stack(
        (
            dx / dz_safe,
            dy / dz_safe,
            jnp.ones_like(dx),
        ),
        axis=-1,
    )




def metric_inverse_residual(geometry: FciGeometry3D) -> jnp.ndarray:
    """Return `max(abs(g^ik g_kj - delta^i_j))` over the grid."""

    product = jnp.einsum("...ik,...kj->...ij", geometry.cell_metric.g_contra, geometry.cell_metric.g_cov)
    identity = jnp.eye(3, dtype=product.dtype)
    return jnp.max(jnp.abs(product - identity))
