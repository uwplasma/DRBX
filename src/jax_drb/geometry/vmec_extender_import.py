from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from .fci_geometry import FciGeometry3D, logical_grid_from_axis_vectors


_REQUIRED_DIMENSIONS = ("nR", "nphi", "nZ")
_REQUIRED_COORDINATES = ("R", "phi", "Z")
_REQUIRED_FIELDS = ("BR", "Bphi", "BZ", "absB")
_ABSB_RTOL = 1.0e-7
_ABSB_ATOL = 1.0e-10


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VmecExtenderGrid:
    """Gridded VMEC-extender magnetic field in physical cylindrical coordinates."""

    R: jax.Array
    phi: jax.Array
    Z: jax.Array
    BR: jax.Array
    Bphi: jax.Array
    BZ: jax.Array
    absB: jax.Array
    nfp: int
    phi_period: float
    metadata: Mapping[str, Any]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.R.size), int(self.phi.size), int(self.Z.size))

    def tree_flatten(self):
        children = (self.R, self.phi, self.Z, self.BR, self.Bphi, self.BZ, self.absB)
        metadata = tuple(sorted((str(key), _static_metadata_value(value)) for key, value in self.metadata.items()))
        aux_data = (int(self.nfp), float(self.phi_period), metadata)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        nfp, phi_period, metadata = aux_data
        R, phi, Z, BR, Bphi, BZ, absB = children
        return cls(
            R=R,
            phi=phi,
            Z=Z,
            BR=BR,
            Bphi=Bphi,
            BZ=BZ,
            absB=absB,
            nfp=int(nfp),
            phi_period=float(phi_period),
            metadata=dict(metadata),
        )


def load_vmec_extender_grid_netcdf(
    path: str | Path,
    *,
    strict_metadata: bool = True,
) -> VmecExtenderGrid:
    """Load a gridded VMEC-extender magnetic field exported as NetCDF.

    The importer expects physical cylindrical coordinates `(R, phi, Z)` and
    cylindrical field components `(BR, Bphi, BZ)`. File I/O and validation are
    intentionally NumPy/NetCDF operations; the returned interpolation kernels
    operate on JAX arrays.
    """

    resolved = Path(path)
    with Dataset(resolved) as dataset:
        _validate_required_dimensions(dataset)
        metadata = {name: _normalize_metadata_value(dataset.getncattr(name)) for name in dataset.ncattrs()}
        R = _read_1d_variable(dataset, "R")
        phi = _read_1d_variable(dataset, "phi")
        Z = _read_1d_variable(dataset, "Z")
        expected_shape = (R.size, phi.size, Z.size)
        BR = _read_field_variable(dataset, "BR", expected_shape)
        Bphi = _read_field_variable(dataset, "Bphi", expected_shape)
        BZ = _read_field_variable(dataset, "BZ", expected_shape)
        absB = _read_field_variable(dataset, "absB", expected_shape)

    nfp = _metadata_nfp(metadata, strict_metadata=strict_metadata)
    _validate_metadata(metadata, strict_metadata=strict_metadata)
    _validate_axis("R", R)
    _validate_axis("phi", phi)
    _validate_axis("Z", Z)
    phi_period = _infer_phi_period(metadata, nfp=nfp)
    _validate_phi_coverage(phi, phi_period=phi_period)
    absB_error = _absB_consistency_error(BR=BR, Bphi=Bphi, BZ=BZ, absB=absB)
    if strict_metadata and absB_error > _ABSB_ATOL + _ABSB_RTOL * float(np.max(np.abs(absB))):
        raise ValueError(
            "absB is inconsistent with sqrt(BR**2 + Bphi**2 + BZ**2): "
            f"max_abs_error={absB_error:.6e}"
        )

    metadata = {
        **metadata,
        "resolved_path": str(resolved),
        "absB_consistency_max_abs_error": float(absB_error),
    }
    return VmecExtenderGrid(
        R=jnp.asarray(R, dtype=jnp.float64),
        phi=jnp.asarray(phi, dtype=jnp.float64),
        Z=jnp.asarray(Z, dtype=jnp.float64),
        BR=jnp.asarray(BR, dtype=jnp.float64),
        Bphi=jnp.asarray(Bphi, dtype=jnp.float64),
        BZ=jnp.asarray(BZ, dtype=jnp.float64),
        absB=jnp.asarray(absB, dtype=jnp.float64),
        nfp=int(nfp),
        phi_period=float(phi_period),
        metadata=metadata,
    )


def interpolate_vmec_extender_B_cyl(
    grid: VmecExtenderGrid,
    R_phi_Z: jax.Array,
    *,
    strict_bounds: bool = False,
) -> jax.Array:
    """Return `(BR, Bphi, BZ)` at target points with physical-phi wrapping.

    Nonperiodic `R` and `Z` coordinates are clamped at the loaded grid edges by
    the JAX interpolation kernel. Pass `strict_bounds=True` for an eager
    preflight check that raises when any target point lies outside the imported
    `R`/`Z` domain.
    """

    points = _as_points(R_phi_Z)
    if strict_bounds:
        validate_vmec_extender_points_in_bounds(grid, points)
    BR = _interpolate_component(grid, grid.BR, points)
    Bphi = _interpolate_component(grid, grid.Bphi, points)
    BZ = _interpolate_component(grid, grid.BZ, points)
    return jnp.stack((BR, Bphi, BZ), axis=-1)


def vmec_extender_absB(
    grid: VmecExtenderGrid,
    R_phi_Z: jax.Array,
    *,
    strict_bounds: bool = False,
) -> jax.Array:
    """Return interpolated magnetic-field magnitude at target points."""

    points = _as_points(R_phi_Z)
    if strict_bounds:
        validate_vmec_extender_points_in_bounds(grid, points)
    return _interpolate_component(grid, grid.absB, points)


def vmec_extender_fieldline_rhs_RZ_phi(
    grid: VmecExtenderGrid,
    R_phi_Z: jax.Array,
    *,
    min_abs_Bphi: float = 1.0e-12,
    strict_bounds: bool = False,
) -> jax.Array:
    """Return `(dR/dphi, dZ/dphi)` using `R * BR / Bphi` and `R * BZ / Bphi`.

    Very small `Bphi` values are bounded with a sign-preserving denominator so
    the function remains JAX-transformable instead of raising inside compiled
    code.
    """

    points = _as_points(R_phi_Z)
    if strict_bounds:
        validate_vmec_extender_points_in_bounds(grid, points)
    B = interpolate_vmec_extender_B_cyl(grid, points)
    R = points[..., 0]
    BR = B[..., 0]
    Bphi = B[..., 1]
    BZ = B[..., 2]
    floor = jnp.asarray(float(min_abs_Bphi), dtype=jnp.float64)
    sign = jnp.where(Bphi < 0.0, -1.0, 1.0)
    safe_Bphi = jnp.where(jnp.abs(Bphi) < floor, sign * floor, Bphi)
    return jnp.stack((R * BR / safe_Bphi, R * BZ / safe_Bphi), axis=-1)


def vmec_extender_points_in_bounds(grid: VmecExtenderGrid, R_phi_Z: jax.Array) -> jax.Array:
    """Return a boolean mask for points inside the imported nonperiodic `R`/`Z` domain."""

    points = _as_points(R_phi_Z)
    return (
        (points[..., 0] >= grid.R[0])
        & (points[..., 0] <= grid.R[-1])
        & (points[..., 2] >= grid.Z[0])
        & (points[..., 2] <= grid.Z[-1])
    )


def validate_vmec_extender_points_in_bounds(grid: VmecExtenderGrid, R_phi_Z: jax.Array) -> None:
    """Raise if any point lies outside the imported nonperiodic `R`/`Z` domain.

    This is an eager host-side guard intended for setup, tests, and production
    preflight checks. It is deliberately separate from the JAX interpolation
    kernel so the kernel remains usable under `jax.jit`.
    """

    points = np.asarray(R_phi_Z, dtype=np.float64)
    if points.shape == (3,):
        points = points.reshape((1, 3))
    elif points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError("R_phi_Z must have shape (3,) or (..., 3).")
    flat = points.reshape((-1, 3))
    R_min = float(np.asarray(grid.R[0]))
    R_max = float(np.asarray(grid.R[-1]))
    Z_min = float(np.asarray(grid.Z[0]))
    Z_max = float(np.asarray(grid.Z[-1]))
    mask = (
        (flat[:, 0] >= R_min)
        & (flat[:, 0] <= R_max)
        & (flat[:, 2] >= Z_min)
        & (flat[:, 2] <= Z_max)
    )
    if not bool(np.all(mask)):
        first_bad = flat[np.flatnonzero(~mask)[0]]
        raise ValueError(
            "VMEC-extender target point lies outside imported R/Z domain: "
            f"point=(R={first_bad[0]:.6g}, phi={first_bad[1]:.6g}, Z={first_bad[2]:.6g}), "
            f"R_range=({R_min:.6g}, {R_max:.6g}), Z_range=({Z_min:.6g}, {Z_max:.6g})"
        )


def build_vmec_extender_fci_geometry(
    grid: VmecExtenderGrid,
    *,
    substeps: int = 8,
) -> FciGeometry3D:
    """Build one-plane forward/backward FCI maps from the imported field.

    The imported magnetic-field grid is stored in physical cylindrical order
    `(R, phi, Z)`, but native FCI arrays use logical
    `(x, y, z) = (radial, poloidal, toroidal)`. The returned map arrays
    therefore have shape `(nR, nZ, nphi)` and contain endpoint coordinates
    `(R_index, Z_index)` on the neighboring toroidal plane.

    The physical arclength between toroidal crossings is not implemented yet
    for this importer, so `forward_length` and `backward_length` are filled
    with `NaN` as a deliberate placeholder.
    """

    dphi = float(grid.phi_period / int(grid.phi.size))
    R0, Z0, phi0 = jnp.meshgrid(grid.R, grid.Z, grid.phi, indexing="ij")
    points = jnp.stack((R0, phi0, Z0), axis=-1)
    forward = _rk4_fieldline_step(grid, points, dphi=float(dphi), substeps=int(substeps))
    backward = _rk4_fieldline_step(grid, points, dphi=-float(dphi), substeps=int(substeps))

    forward_R_index = _axis_to_logical_index(grid.R, forward[..., 0])
    forward_Z_index = _axis_to_logical_index(grid.Z, forward[..., 2])
    backward_R_index = _axis_to_logical_index(grid.R, backward[..., 0])
    backward_Z_index = _axis_to_logical_index(grid.Z, backward[..., 2])
    forward_boundary = _outside_RZ(grid, forward)
    backward_boundary = _outside_RZ(grid, backward)
    shape = forward_R_index.shape
    ones = jnp.ones(shape, dtype=jnp.float64)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    return FciGeometry3D(
        logical_grid=logical_grid_from_axis_vectors(grid.R, grid.Z, grid.phi),
        forward_x=forward_R_index,
        forward_y=forward_Z_index,
        backward_x=backward_R_index,
        backward_y=backward_Z_index,
        forward_length=jnp.full_like(forward_R_index, jnp.nan, dtype=jnp.float64),
        backward_length=jnp.full_like(backward_R_index, jnp.nan, dtype=jnp.float64),
        forward_boundary=forward_boundary,
        backward_boundary=backward_boundary,
        dx=jnp.ones(shape, dtype=jnp.float64) * float(np.mean(np.diff(np.asarray(grid.R, dtype=np.float64)))),
        dy=jnp.ones(shape, dtype=jnp.float64) * float(np.mean(np.diff(np.asarray(grid.Z, dtype=np.float64)))),
        dz=jnp.ones(shape, dtype=jnp.float64) * float(dphi),
        J=jnp.broadcast_to(jnp.asarray(grid.R, dtype=jnp.float64)[:, None, None], shape),
        B_contravariant=jnp.zeros(shape + (3,), dtype=jnp.float64).at[..., 2].set(1.0),
        g11=ones,
        g22=ones,
        g33=ones,
        g12=zeros,
        g13=zeros,
        g23=zeros,
        g_11=ones,
        g_22=ones,
        g_33=ones,
        g_12=zeros,
        g_13=zeros,
        g_23=zeros,
    )


def _rk4_fieldline_step(
    grid: VmecExtenderGrid,
    points: jax.Array,
    *,
    dphi: float,
    substeps: int,
) -> jax.Array:
    step = float(dphi) / float(max(int(substeps), 1))
    state = jnp.asarray(points, dtype=jnp.float64)
    for _ in range(max(int(substeps), 1)):
        k1 = _fieldline_state_rhs(grid, state)
        k2 = _fieldline_state_rhs(grid, state + 0.5 * step * k1)
        k3 = _fieldline_state_rhs(grid, state + 0.5 * step * k2)
        k4 = _fieldline_state_rhs(grid, state + step * k3)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def _fieldline_state_rhs(grid: VmecExtenderGrid, points: jax.Array) -> jax.Array:
    rhs_RZ = vmec_extender_fieldline_rhs_RZ_phi(grid, points)
    ones = jnp.ones_like(rhs_RZ[..., 0])
    return jnp.stack((rhs_RZ[..., 0], ones, rhs_RZ[..., 1]), axis=-1)


def _axis_to_logical_index(axis: jax.Array, values: jax.Array) -> jax.Array:
    i0, i1, weight = _bracket_nonperiodic(axis, values)
    return jnp.asarray(i0, dtype=jnp.float64) * (1.0 - weight) + jnp.asarray(i1, dtype=jnp.float64) * weight


def _outside_RZ(grid: VmecExtenderGrid, points: jax.Array) -> jax.Array:
    return (
        (points[..., 0] < grid.R[0])
        | (points[..., 0] > grid.R[-1])
        | (points[..., 2] < grid.Z[0])
        | (points[..., 2] > grid.Z[-1])
    )


def _as_points(R_phi_Z: jax.Array) -> jax.Array:
    points = jnp.asarray(R_phi_Z, dtype=jnp.float64)
    if points.shape == (3,):
        return points
    if len(points.shape) == 0 or points.shape[-1] != 3:
        raise ValueError("R_phi_Z must have shape (3,) or (..., 3).")
    return points


def _interpolate_component(grid: VmecExtenderGrid, values: jax.Array, points: jax.Array) -> jax.Array:
    R_value = points[..., 0]
    phi_value = points[..., 1]
    Z_value = points[..., 2]
    r0, r1, wr = _bracket_nonperiodic(grid.R, R_value)
    p0, p1, wp = _bracket_periodic(grid.phi, phi_value, period=float(grid.phi_period))
    z0, z1, wz = _bracket_nonperiodic(grid.Z, Z_value)
    return _trilinear(values, r0, r1, wr, p0, p1, wp, z0, z1, wz)


def _bracket_nonperiodic(axis: jax.Array, values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    upper = jnp.searchsorted(axis, values, side="right")
    lower = jnp.clip(upper - 1, 0, int(axis.size) - 2)
    upper = lower + 1
    lower_coord = axis[lower]
    upper_coord = axis[upper]
    weight = (values - lower_coord) / (upper_coord - lower_coord)
    return lower, upper, jnp.clip(weight, 0.0, 1.0)


def _bracket_periodic(axis: jax.Array, values: jax.Array, *, period: float) -> tuple[jax.Array, jax.Array, jax.Array]:
    origin = axis[0]
    wrapped = jnp.mod(values - origin, float(period)) + origin
    upper = jnp.searchsorted(axis, wrapped, side="right")
    lower = jnp.clip(upper - 1, 0, int(axis.size) - 1)
    next_index = jnp.mod(lower + 1, int(axis.size))
    lower_coord = axis[lower]
    upper_coord = jnp.where(lower == int(axis.size) - 1, axis[0] + float(period), axis[next_index])
    weight = (wrapped - lower_coord) / (upper_coord - lower_coord)
    return lower, next_index, jnp.clip(weight, 0.0, 1.0)


def _trilinear(
    values: jax.Array,
    r0: jax.Array,
    r1: jax.Array,
    wr: jax.Array,
    p0: jax.Array,
    p1: jax.Array,
    wp: jax.Array,
    z0: jax.Array,
    z1: jax.Array,
    wz: jax.Array,
) -> jax.Array:
    c000 = values[r0, p0, z0]
    c100 = values[r1, p0, z0]
    c010 = values[r0, p1, z0]
    c110 = values[r1, p1, z0]
    c001 = values[r0, p0, z1]
    c101 = values[r1, p0, z1]
    c011 = values[r0, p1, z1]
    c111 = values[r1, p1, z1]
    c00 = c000 * (1.0 - wr) + c100 * wr
    c10 = c010 * (1.0 - wr) + c110 * wr
    c01 = c001 * (1.0 - wr) + c101 * wr
    c11 = c011 * (1.0 - wr) + c111 * wr
    c0 = c00 * (1.0 - wp) + c10 * wp
    c1 = c01 * (1.0 - wp) + c11 * wp
    return c0 * (1.0 - wz) + c1 * wz


def _validate_required_dimensions(dataset: Dataset) -> None:
    for name in _REQUIRED_DIMENSIONS:
        if name not in dataset.dimensions:
            raise ValueError(f"VMEC-extender grid is missing required dimension {name!r}.")


def _read_1d_variable(dataset: Dataset, name: str) -> np.ndarray:
    if name not in dataset.variables:
        raise ValueError(f"VMEC-extender grid is missing required coordinate variable {name!r}.")
    values = np.asarray(dataset.variables[name][:], dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Coordinate variable {name!r} must be one-dimensional, got shape {values.shape}.")
    return values


def _read_field_variable(dataset: Dataset, name: str, expected_shape: tuple[int, int, int]) -> np.ndarray:
    if name not in dataset.variables:
        raise ValueError(f"VMEC-extender grid is missing required field variable {name!r}.")
    values = np.asarray(dataset.variables[name][:], dtype=np.float64)
    if values.shape != expected_shape:
        raise ValueError(
            f"Field variable {name!r} has shape {values.shape}, expected {expected_shape} for dimensions (nR, nphi, nZ)."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Field variable {name!r} contains non-finite values.")
    return values


def _validate_axis(name: str, values: np.ndarray) -> None:
    if values.size < 2:
        raise ValueError(f"Coordinate axis {name!r} must contain at least two points.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Coordinate axis {name!r} contains non-finite values.")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError(f"Coordinate axis {name!r} must be strictly increasing.")


def _validate_metadata(metadata: Mapping[str, Any], *, strict_metadata: bool) -> None:
    convention = metadata.get("coordinate_convention")
    if convention is None:
        if strict_metadata:
            raise ValueError("VMEC-extender grid is missing required coordinate_convention metadata.")
        return
    convention_text = str(convention).lower()
    if "zeta" in convention_text:
        raise ValueError("VMEC-extender grid coordinate_convention mentions zeta; expected physical phi.")
    if strict_metadata and "phi" not in convention_text:
        raise ValueError("VMEC-extender grid coordinate_convention must explicitly mention physical phi.")
    components = metadata.get("field_components")
    if components is not None:
        component_text = str(components).replace(" ", "").lower()
        for name in ("br", "bphi", "bz"):
            if name not in component_text:
                raise ValueError("VMEC-extender grid field_components must include BR,Bphi,BZ.")


def _metadata_nfp(metadata: Mapping[str, Any], *, strict_metadata: bool) -> int:
    value = metadata.get("nfp")
    if value is None:
        if strict_metadata:
            raise ValueError("VMEC-extender grid is missing required nfp metadata.")
        return 1
    try:
        nfp = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"VMEC-extender grid nfp metadata is not an integer: {value!r}") from error
    if nfp <= 0:
        raise ValueError("VMEC-extender grid nfp metadata must be positive.")
    return nfp


def _infer_phi_period(metadata: Mapping[str, Any], *, nfp: int) -> float:
    if "phi_period" in metadata:
        period = float(metadata["phi_period"])
        if period <= 0.0:
            raise ValueError("VMEC-extender grid phi_period metadata must be positive.")
        return period
    coverage = str(
        metadata.get("phi_coverage", metadata.get("toroidal_coverage", metadata.get("coverage", "")))
    ).lower()
    if "2*pi" in coverage or "2pi" in coverage or "full" in coverage:
        return float(2.0 * np.pi)
    return float(2.0 * np.pi / int(nfp))


def _validate_phi_coverage(phi: np.ndarray, *, phi_period: float) -> None:
    span = float(phi[-1] - phi[0])
    if span >= phi_period:
        raise ValueError(
            "Physical phi grid must not include a duplicated periodic endpoint; "
            f"axis span={span:.6e}, phi_period={phi_period:.6e}."
        )


def _absB_consistency_error(*, BR: np.ndarray, Bphi: np.ndarray, BZ: np.ndarray, absB: np.ndarray) -> float:
    expected = np.sqrt(BR * BR + Bphi * Bphi + BZ * BZ)
    return float(np.max(np.abs(absB - expected)))


def _normalize_metadata_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _normalize_metadata_value(value.item())
        return tuple(_normalize_metadata_value(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_metadata_value(item) for item in value)
    return value


def _static_metadata_value(value: Any) -> Any:
    value = _normalize_metadata_value(value)
    if isinstance(value, list):
        return tuple(_static_metadata_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_static_metadata_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _static_metadata_value(item)) for key, item in value.items()))
    return value
