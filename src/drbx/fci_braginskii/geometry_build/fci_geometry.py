from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Callable, Mapping, Sequence
import zipfile
import numpy as np
import jax
import jax.numpy as jnp

# Definitions identical to (and closure-equivalent with) the shared module;
# re-exported here so existing import paths keep working.
from ...geometry.fci_geometry import (  # noqa: F401
    _DataclassPyTreeMixin,
    _as_float_array,
    _require_float_shape,
    _metric_from_components,
    _bmag_from_contravariant_components,
    logical_grid_from_axis_vectors,
    Grid1D,
    LocalGrid1D,
    CellCenteredGrid3D,
    ShardSpec3D,
    NeighborMap3D,
    Spacing3D,
    MetricGeometry,
    FaceMetricGeometry,
    BFieldGeometry,
    FaceBFieldGeometry,
    _axis_index_nd,
    _physical_domain_valid_mask,
    _extend_axis_with_ghost_cells,
    _extend_values_x,
    _extend_values_y,
    _extend_values_z,
    _interpolate_scalar_cell_centered,
    _interpolate_B_contravariant_cell_centered,
    _bracket_axis,
    _trilinear_sample,
    _logical_coordinate_to_index,
)

# Definitions identical to (and closure-equivalent with) the shared module;
# re-exported here so existing import paths keep working.
from ..geometry.fci_geometry import (  # noqa: F401
    _sample_local_fci_cut_wall_values,
    _sample_local_fci_table_rows,
)
from ..geometry.fci_geometry import (
    interpolate_B_contravariant,  # noqa: F401
)
from ...geometry.fci_geometry import (
    CellVolumeGeometry3D,  # noqa: F401
    FciGeometry3D,
    build_curvature_coefficients,  # noqa: F401
)


_pytree_base = jax.tree_util.register_pytree_node_class


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

    # Continuous magnetic-field evaluation retained at each traced endpoint.
    # These values come from the same callback used by the RK tracer; physical
    # wall laws must use them instead of interpolating an already branched
    # sign(B.n)-dependent target from the regular wall grid.
    forward_endpoint_b_contra_x: jnp.ndarray | None = None
    forward_endpoint_b_contra_y: jnp.ndarray | None = None
    forward_endpoint_b_contra_z: jnp.ndarray | None = None
    forward_endpoint_bmag: jnp.ndarray | None = None
    backward_endpoint_b_contra_x: jnp.ndarray | None = None
    backward_endpoint_b_contra_y: jnp.ndarray | None = None
    backward_endpoint_b_contra_z: jnp.ndarray | None = None
    backward_endpoint_bmag: jnp.ndarray | None = None

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
        for name in (
            "forward_endpoint_b_contra_x",
            "forward_endpoint_b_contra_y",
            "forward_endpoint_b_contra_z",
            "forward_endpoint_bmag",
            "backward_endpoint_b_contra_x",
            "backward_endpoint_b_contra_y",
            "backward_endpoint_b_contra_z",
            "backward_endpoint_bmag",
        ):
            raw = getattr(self, name)
            value = (
                jnp.zeros(shape, dtype=jnp.float64)
                if raw is None
                else _require_float_shape(raw, shape, f"FciMaps3D.{name}")
            )
            object.__setattr__(self, name, value)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.forward_x.shape)


# Dependency kinds are shared by local and remote FCI dependency metadata.


@_pytree_base
@dataclass(frozen=True)
class CurvatureEdgeOneForm3D(_DataclassPyTreeMixin):
    """Owned shared-edge samples of the covariant magnetic one-form.

    The three arrays are the common edge values used by the compatible
    incidence curl: ``Az_xy`` lives on edges parallel to ``z``, ``Ay_xz`` on
    edges parallel to ``y``, and ``Ax_yz`` on edges parallel to ``x``.  They
    contain only the owned host edge set; the curvature builder places them in
    its halo work arrays before applying its existing physical-boundary and
    axis handling.
    """

    Az_xy: jnp.ndarray
    Ay_xz: jnp.ndarray
    Ax_yz: jnp.ndarray

    def __post_init__(self) -> None:
        az_shape = tuple(int(v) for v in jnp.asarray(self.Az_xy).shape)
        if len(az_shape) != 3:
            raise ValueError(
                "CurvatureEdgeOneForm3D.Az_xy must be three-dimensional, "
                f"got {az_shape}"
            )
        nx_cells = az_shape[0] - 1
        ny = az_shape[1] - 1
        nz = az_shape[2]
        if nx_cells < 1 or ny < 1 or nz < 1:
            raise ValueError(
                "CurvatureEdgeOneForm3D.Az_xy must have positive edge extents"
            )
        # Infer the cell counts from the Az family, whose first two axes are
        # one larger than the cell-centered extents.
        expected = {
            "Az_xy": (nx_cells + 1, ny + 1, nz),
            "Ay_xz": (nx_cells + 1, ny, nz + 1),
            "Ax_yz": (nx_cells, ny + 1, nz + 1),
        }
        for name, shape in expected.items():
            object.__setattr__(
                self,
                name,
                _require_float_shape(getattr(self, name), shape, f"CurvatureEdgeOneForm3D.{name}"),
            )
            try:
                finite = bool(jnp.all(jnp.isfinite(getattr(self, name))))
            except jax.errors.TracerBoolConversionError:
                finite = True
            if not finite:
                raise ValueError(f"CurvatureEdgeOneForm3D.{name} must be finite")

    @property
    def axes(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.Az_xy, self.Ay_xz, self.Ax_yz


# Meaning of a global lower/upper side in a ShardSpec3D. These values are
# metadata for later halo stages; HaloExchange3D only owns regular-neighbor
# and SIDE_SIMPLE_PERIODIC data.


# Backward-compatible aliases. Both historical context names now refer to the
# same canonical PyTree type.


ANGULAR_AGGLOMERATION_HOST_CACHE_VERSION = 4
ANGULAR_AGGLOMERATION_HOST_IDENTITY_VERSION = 2


def _metric_cache_contract_identity(metric_path: Path) -> dict[str, object]:
    """Extract the stable numerical contract needed by the RLP host cache."""

    def without_volatile_metadata(value):
        if isinstance(value, dict):
            return {
                key: without_volatile_metadata(item)
                for key, item in value.items()
                if key not in {"path", "mtime_ns"}
            }
        if isinstance(value, list):
            return [without_volatile_metadata(item) for item in value]
        return value

    if not metric_path.is_file():
        return {"missing": True}
    try:
        with np.load(metric_path, allow_pickle=False) as cached:
            cache_spec = without_volatile_metadata(
                json.loads(str(cached["cache_spec"].item()))
            )
            identity = {"cache_spec": cache_spec}
            for name in (
                "fci_maps_format_version",
                "fci_maps_source_fingerprint",
                "fci_maps_trace_substeps",
            ):
                if name in cached.files:
                    identity[name] = np.asarray(cached[name]).item()
            return identity
    except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile):
        # The metric loader will reject an unreadable payload. Keep this
        # fallback deterministic without tying valid caches to their path.
        stat = metric_path.stat()
        return {"unreadable_size": int(stat.st_size)}


def angular_agglomeration_host_geometry_cache_path(
    metric_cache_path: Path | None,
    u_faces: np.ndarray,
    theta_faces: np.ndarray,
    eta_faces: np.ndarray,
    profile: np.ndarray,
) -> Path | None:
    """Return the cache path for one metric/profile agglomeration geometry."""

    if metric_cache_path is None:
        return None
    metric_path = Path(metric_cache_path)
    identity = {
        "format_version": ANGULAR_AGGLOMERATION_HOST_CACHE_VERSION,
        "identity_version": ANGULAR_AGGLOMERATION_HOST_IDENTITY_VERSION,
        "mode": "radius-dependent-angular-agglomeration",
        "metric_contract": _metric_cache_contract_identity(metric_path),
        "u_faces": hashlib.sha256(np.asarray(u_faces, dtype=np.float64).tobytes()).hexdigest(),
        "theta_faces": hashlib.sha256(np.asarray(theta_faces, dtype=np.float64).tobytes()).hexdigest(),
        "eta_faces": hashlib.sha256(np.asarray(eta_faces, dtype=np.float64).tobytes()).hexdigest(),
        "angular_group_sizes": [int(value) for value in np.asarray(profile).ravel()],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:24]
    return metric_path.parent / f"angular_rlp_host_{digest}.npz"


def angular_agglomeration_host_geometry_payload(host) -> dict[str, np.ndarray]:
    """Serialize a polar angular-agglomeration host geometry."""

    topology = host.topology
    payload = {
        "format_version": np.asarray(
            ANGULAR_AGGLOMERATION_HOST_CACHE_VERSION, dtype=np.int64
        ),
        "mode": np.asarray("radius-dependent-angular-agglomeration"),
        "shape": np.asarray(topology.shape, dtype=np.int64),
        "topology_coordinate_periods": np.asarray(
            (1.0, host.theta_period, host.eta_period), dtype=np.float64
        ),
        "angular_group_size": np.asarray(host.angular_group_size, dtype=np.int32),
        "host_eta_period": np.asarray(host.eta_period),
        "host_theta_period": np.asarray(host.theta_period),
        "host_quadrature_order": np.asarray(host.quadrature_order, dtype=np.int64),
    }
    for name in (
        "aggregate_id", "owner_index", "is_merge_source", "is_active_owner",
        "retained_cut_cell", "aggregate_volume", "aggregate_centroid",
        "aggregate_second_moment", "aggregate_third_moment", "face_id",
        "face_axis", "face_storage_index", "face_minus_aggregate_id",
        "face_plus_aggregate_id", "face_measure",
    ):
        payload[f"topology_{name}"] = np.asarray(getattr(topology, name))
    for field in fields(host):
        if field.name in {
            "topology", "angular_group_size", "theta_period", "eta_period",
            "quadrature_order",
        }:
            continue
        payload[f"host_{field.name}"] = np.asarray(getattr(host, field.name))
    return payload


def load_angular_agglomeration_host_geometry_cache(
    path: Path | None,
    expected_profile: np.ndarray,
):
    """Load and validate a cached angular-agglomeration host geometry."""

    from .fci_control_volumes import (
        GlobalControlVolumeTopology3D,
        PolarAngularAgglomerationGeometry3D,
    )

    if path is None or not Path(path).is_file():
        return None
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as cached:
            if int(cached["format_version"].item()) != ANGULAR_AGGLOMERATION_HOST_CACHE_VERSION:
                raise ValueError("angular host cache format mismatch")
            if str(cached["mode"].item()) != "radius-dependent-angular-agglomeration":
                raise ValueError("cache is not an angular-agglomeration payload")
            profile = np.asarray(cached["angular_group_size"], dtype=np.int32)
            if not np.array_equal(profile, np.asarray(expected_profile, dtype=np.int32)):
                raise ValueError("angular host cache profile mismatch")
            shape = tuple(int(value) for value in np.asarray(cached["shape"]).tolist())
            topology = GlobalControlVolumeTopology3D(
                shape=shape,
                **{
                    name: np.array(cached[f"topology_{name}"])
                    for name in (
                        "aggregate_id", "owner_index", "is_merge_source",
                        "is_active_owner", "retained_cut_cell", "aggregate_volume",
                        "aggregate_centroid", "aggregate_second_moment",
                        "aggregate_third_moment", "face_id", "face_axis",
                        "face_storage_index", "face_minus_aggregate_id",
                        "face_plus_aggregate_id", "face_measure",
                    )
                },
            )
            host_names = [
                field.name
                for field in fields(PolarAngularAgglomerationGeometry3D)
                if field.name != "topology"
            ]
            kwargs = {
                name: np.array(cached[f"host_{name}"])
                for name in host_names
                if name not in {
                    "theta_period", "eta_period", "quadrature_order",
                    "angular_group_size",
                }
            }
            kwargs.update(
                angular_group_size=profile,
                theta_period=float(cached["host_theta_period"].item()),
                eta_period=float(cached["host_eta_period"].item()),
                quadrature_order=int(cached["host_quadrature_order"].item()),
            )
            host = PolarAngularAgglomerationGeometry3D(topology=topology, **kwargs)
            if host.quadrature_order != 2:
                raise ValueError("angular host cache has incompatible quadrature")
            return host
    except Exception as error:
        print(f"[pole-angular-host] invalid cache {path}: {error}; rebuilding", flush=True)
        return None


def write_angular_agglomeration_host_geometry_cache(path: Path, host) -> None:
    """Atomically write an angular-agglomeration host geometry cache."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez(temporary_path, **angular_agglomeration_host_geometry_payload(host))
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    print(f"[pole-angular-host] wrote cache {path}", flush=True)


def metric_aware_angular_group_profile(
    global_geometry: FciGeometry3D,
    metric_evaluator,
    *,
    explicit_profile: Sequence[int] | None = None,
) -> tuple[np.ndarray, float]:
    """Select a nested angular owner profile from the physical metric widths."""

    from .fci_control_volumes import (
        build_nested_angular_group_profile,
        build_radius_dependent_angular_group_profile,
    )

    u_faces = np.asarray(global_geometry.grid.x.faces, dtype=np.float64)
    theta_faces = np.asarray(global_geometry.grid.y.faces, dtype=np.float64)
    logical = np.stack(
        np.meshgrid(
            np.asarray(global_geometry.grid.x.centers),
            np.asarray(global_geometry.grid.y.centers),
            np.asarray(global_geometry.grid.z.centers),
            indexing="ij",
        ),
        axis=-1,
    )
    metric = metric_evaluator.evaluate(logical.reshape(-1, 3))
    g_cov = np.asarray(metric.g_cov, dtype=np.float64).reshape(
        logical.shape[:-1] + (3, 3)
    )
    if not np.all(np.isfinite(g_cov)):
        raise ValueError("radius-dependent angular profile metric is nonfinite")
    radial_width = np.sqrt(g_cov[..., 0, 0]) * np.diff(u_faces)[:, None, None]
    dtheta = (theta_faces[-1] - theta_faces[0]) / len(global_geometry.grid.y.centers)
    theta_width = np.sqrt(g_cov[..., 1, 1]) * dtheta
    if (
        np.any(~np.isfinite(radial_width))
        or np.any(~np.isfinite(theta_width))
        or np.any(radial_width <= 0.0)
        or np.any(theta_width <= 0.0)
    ):
        raise ValueError(
            "radius-dependent angular profile metric widths are not positive finite"
        )
    logical_profile = build_radius_dependent_angular_group_profile(
        u_faces, theta_faces
    )
    ntheta = len(global_geometry.grid.y.centers)
    metric_minimum = np.max(radial_width / theta_width, axis=(1, 2))
    minimum = np.maximum(metric_minimum, logical_profile.astype(np.float64))
    minimum[0] = float(ntheta)
    if explicit_profile is not None:
        profile = build_radius_dependent_angular_group_profile(
            u_faces,
            theta_faces,
            explicit_profile=tuple(int(value) for value in explicit_profile),
        )
    else:
        profile = build_nested_angular_group_profile(ntheta, minimum)
    if (
        np.any(profile.astype(np.float64) < minimum - 1.0e-14)
        or np.any(profile[1:] > profile[:-1])
        or np.any(profile[:-1] % profile[1:] != 0)
    ):
        raise ValueError("angular profile is not nested or metric-safe")
    ratio = float(
        np.min(profile[:, None, None] * theta_width / radial_width)
    )
    return profile, ratio


def build_metric_aware_polar_angular_agglomeration_geometry(
    global_geometry: FciGeometry3D,
    metric_evaluator,
    *,
    metric_cache_path: Path | None = None,
    explicit_profile: Sequence[int] | None = None,
):
    """Build/cache the production RLP owner and physical-volume geometry."""

    from .fci_control_volumes import build_polar_angular_agglomeration_geometry

    profile, safety_ratio = metric_aware_angular_group_profile(
        global_geometry,
        metric_evaluator,
        explicit_profile=explicit_profile,
    )
    u_faces = np.asarray(global_geometry.grid.x.faces, dtype=np.float64)
    theta_faces = np.asarray(global_geometry.grid.y.faces, dtype=np.float64)
    eta_faces = np.asarray(global_geometry.grid.z.faces, dtype=np.float64)
    cache_path = angular_agglomeration_host_geometry_cache_path(
        metric_cache_path, u_faces, theta_faces, eta_faces, profile
    )
    host = load_angular_agglomeration_host_geometry_cache(cache_path, profile)
    if host is None:
        host = build_polar_angular_agglomeration_geometry(
            u_faces,
            theta_faces,
            eta_faces,
            lambda points: np.asarray(
                metric_evaluator.evaluate(points).J, dtype=np.float64
            ),
            quadrature_order=2,
            angular_group_size=profile,
        )
        if cache_path is not None:
            write_angular_agglomeration_host_geometry_cache(cache_path, host)
    else:
        print(f"[pole-angular-host] loaded validated cache {cache_path}", flush=True)

    return host, safety_ratio


def build_fci_maps_from_b_contravariant(
    grid: CellCenteredGrid3D,
    B_contra_cell: jnp.ndarray,
    Bmag_cell: jnp.ndarray,
    *,
    substeps: int = 4,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
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

    ``axis_regular_axes`` enables the polar/toroidal lower-radial topology.
    The currently supported case is ``axis_regular_axes[0]`` with periodic
    theta and eta: a trial point crossing the lower x face is reflected to
    ``x -> 2*x_axis - x`` and its theta coordinate is advanced by pi.  The
    lower radial face is therefore not a physical boundary; the upper radial
    face remains physical.

    For jitting:

        build_maps_jit = jax.jit(
            build_fci_maps_from_b_contravariant,
            static_argnames=("substeps", "periodic_axes", "axis_regular_axes"),
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
    axis_regular_axes = tuple(bool(v) for v in axis_regular_axes)
    if len(axis_regular_axes) != 3:
        raise ValueError(f"axis_regular_axes must have length 3, got {axis_regular_axes}")
    if axis_regular_axes[1] or axis_regular_axes[2]:
        raise ValueError(
            "FCI axis regularity currently supports only the lower-radial x axis; "
            f"got axis_regular_axes={axis_regular_axes}"
        )
    if axis_regular_axes[0]:
        if periodic_x:
            raise ValueError("the axis-regular radial axis cannot also be periodic")
        if not periodic_y:
            raise ValueError("axis-regular radial tracing requires periodic theta")
        if not periodic_z:
            raise ValueError("axis-regular radial tracing requires periodic eta")
        if abs(float(grid.x.faces[0])) > 1.0e-12:
            raise ValueError(
                "axis-regular radial tracing requires the lower x face to be x=0; "
                f"got {float(grid.x.faces[0])}"
            )

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

    def _axis_regularize_points(points: jnp.ndarray) -> jnp.ndarray:
        """Apply the signed-radius polar identification to trial points."""

        if not axis_regular_axes[0]:
            return _wrap_points(points)

        x = points[..., 0]
        crossed_axis = x < grid.x.faces[0]
        reflected_x = jnp.where(crossed_axis, 2.0 * grid.x.faces[0] - x, x)
        reflected_theta = jnp.where(crossed_axis, points[..., 1] + jnp.pi, points[..., 1])
        return _wrap_points(jnp.stack((reflected_x, reflected_theta, points[..., 2]), axis=-1))

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
        points = _axis_regularize_points(points)
        sample_x = points[..., 0] if periodic_x else jnp.clip(points[..., 0], grid.x.faces[0], grid.x.faces[-1])
        sample_y = points[..., 1] if periodic_y else jnp.clip(points[..., 1], grid.y.faces[0], grid.y.faces[-1])
        sample_z = points[..., 2] if periodic_z else jnp.clip(points[..., 2], grid.z.faces[0], grid.z.faces[-1])
        points = _wrap_points(jnp.stack((sample_x, sample_y, sample_z), axis=-1))
        return _interpolate_B_contravariant_cell_centered(
            grid,
            B_contra_cell,
            points,
            periodic_axes=periodic_axes,
            boundary_value=boundary_value,
        )

    def _interp_Bmag(points: jnp.ndarray) -> jnp.ndarray:
        points = _axis_regularize_points(points)
        sample_x = points[..., 0] if periodic_x else jnp.clip(points[..., 0], grid.x.faces[0], grid.x.faces[-1])
        sample_y = points[..., 1] if periodic_y else jnp.clip(points[..., 1], grid.y.faces[0], grid.y.faces[-1])
        sample_z = points[..., 2] if periodic_z else jnp.clip(points[..., 2], grid.z.faces[0], grid.z.faces[-1])
        points = _wrap_points(jnp.stack((sample_x, sample_y, sample_z), axis=-1))
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
        b = _interp_B(_axis_regularize_points(points))
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
        h = jnp.asarray(h, dtype=jnp.float64)
        h_points = h if h.ndim == 0 else h[..., None]
        k1 = _rhs(points)
        k2 = _rhs(points + 0.5 * h_points * k1)
        k3 = _rhs(points + 0.5 * h_points * k2)
        k4 = _rhs(points + h_points * k3)
        return points + (h_points / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _speed(sampled_b: jnp.ndarray, sampled_bmag: jnp.ndarray) -> jnp.ndarray:
        bz = _safe_bz(sampled_b[..., 2])
        return sampled_bmag / jnp.maximum(jnp.abs(bz), 1.0e-30)

    def _axis_crossing_fraction(
        old: jnp.ndarray,
        new: jnp.ndarray,
        lower: jnp.ndarray,
        upper: jnp.ndarray,
        periodic: bool,
        axis_regular: bool = False,
    ) -> jnp.ndarray:
        """Fraction along old->new where a nonperiodic axis hits a boundary.

        Returns inf if this axis does not produce a boundary crossing.
        """

        if periodic:
            return jnp.full_like(old, jnp.inf, dtype=jnp.float64)

        denom = new - old
        safe_denom = jnp.where(jnp.abs(denom) < 1.0e-300, 1.0, denom)

        crosses_lower = jnp.where(axis_regular, jnp.zeros_like(new, dtype=bool), new < lower)
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
            axis_regular_axes[0],
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
            state = _axis_regularize_points(state)

            b0 = _interp_B(state)
            bmag0 = _interp_Bmag(state)
            speed0 = _speed(b0, bmag0)

            # The RK stages are sampled on the reflected branch by
            # ``_axis_regularize_points``. Reflect the completed endpoint
            # below after the step; splitting the step at an approximate Euler
            # event can apply the theta identification twice for fields that
            # are only cell-wise regularized.
            raw_next_state = _rk4_batch(state, step_size)
            finite_next = jnp.all(jnp.isfinite(raw_next_state), axis=-1)

            # For nonfinite results, keep the old state to avoid NaN pollution.
            next_state_finite = jnp.where(finite_next[:, None], raw_next_state, state)
            next_state_finite = _axis_regularize_points(next_state_finite)

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


def _callback_fci_field_values(
    field_evaluator: Callable[[np.ndarray], object],
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize callback results to ``(B_contravariant, Bmag)`` arrays."""

    result = field_evaluator(np.asarray(points, dtype=np.float64))
    if hasattr(result, "B_contravariant") and hasattr(result, "magnitude"):
        b_contra = result.B_contravariant
        bmag = result.magnitude
    elif isinstance(result, dict) and "B_contravariant" in result and "magnitude" in result:
        b_contra = result["B_contravariant"]
        bmag = result["magnitude"]
    elif isinstance(result, (tuple, list)) and len(result) == 2:
        b_contra, bmag = result
    else:
        raise TypeError(
            "field_evaluator must return MagneticFieldEvaluation, a "
            "{'B_contravariant', 'magnitude'} mapping, or (B_contravariant, Bmag)"
        )

    b_contra = np.asarray(b_contra, dtype=np.float64)
    bmag = np.asarray(bmag, dtype=np.float64)
    expected_b = (points.shape[0], 3)
    if b_contra.shape != expected_b:
        raise ValueError(
            f"field_evaluator B_contravariant must have shape {expected_b}, got {b_contra.shape}"
        )
    if bmag.shape not in ((points.shape[0],), (points.shape[0], 1)):
        raise ValueError(
            f"field_evaluator Bmag must have shape {(points.shape[0],)}, got {bmag.shape}"
        )
    bmag = bmag.reshape(points.shape[0])
    if not np.all(np.isfinite(b_contra)) or not np.all(np.isfinite(bmag)):
        raise ValueError("field_evaluator returned non-finite magnetic-field values")
    return b_contra, bmag


def trace_fci_points_to_plane_from_callbacks(
    grid: CellCenteredGrid3D,
    field_evaluator: Callable[[np.ndarray], object],
    seed_points: np.ndarray,
    eta_step: float,
    *,
    substeps: int = 4,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    min_abs_bz: float = 1.0e-30,
    return_numpy: bool = False,
) -> dict[str, jnp.ndarray]:
    """Trace arbitrary logical points to a neighboring eta plane.

    The callback is evaluated at every RK stage and at the returned endpoint,
    so this routine is suitable for tracing owner-face vertices as well as
    cell centers.  ``eta_step`` is signed: positive values follow increasing
    eta and negative values follow decreasing eta.  The returned ``length``
    is always nonnegative and ``boundary`` marks trajectories that terminate
    on a nonperiodic physical boundary.

    This is the point-tracing kernel used by the callback FCI map path.  It is
    intentionally NumPy-host-side (the field callback need not be JAX
    compatible), while returned arrays use the same JAX array convention as
    the rest of :mod:`fci_geometry`.
    """

    points = np.asarray(seed_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"seed_points must have shape (n, 3), got {points.shape}")
    if int(substeps) < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")
    if len(periodic_axes) != 3 or len(axis_regular_axes) != 3:
        raise ValueError("periodic_axes and axis_regular_axes must have length 3")
    periodic_axes = tuple(bool(value) for value in periodic_axes)
    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    if any(axis_regular_axes[1:]):
        raise ValueError(
            "FCI axis regularity currently supports only the lower-radial x axis; "
            f"got axis_regular_axes={axis_regular_axes}"
        )
    axis_regular_x = axis_regular_axes[0]
    if axis_regular_x and (periodic_axes[0] or not periodic_axes[1] or not periodic_axes[2]):
        raise ValueError(
            "lower-radial axis regularity requires x nonperiodic and y/z periodic; "
            f"got periodic_axes={periodic_axes}"
        )
    try:
        eta_step = float(eta_step)
    except (TypeError, ValueError) as error:
        raise ValueError(f"eta_step must be a finite scalar, got {eta_step!r}") from error
    if not np.isfinite(eta_step):
        raise ValueError(f"eta_step must be a finite scalar, got {eta_step!r}")
    min_abs_bz = float(min_abs_bz)
    if not np.isfinite(min_abs_bz) or min_abs_bz <= 0.0:
        raise ValueError(f"min_abs_bz must be positive and finite, got {min_abs_bz}")

    nx, ny, nz = grid.shape
    del nx, ny, nz
    x_lower, x_upper = float(grid.x.faces[0]), float(grid.x.faces[-1])
    y_lower, y_upper = float(grid.y.faces[0]), float(grid.y.faces[-1])
    z_lower, z_upper = float(grid.z.faces[0]), float(grid.z.faces[-1])
    y_period = y_upper - y_lower

    if axis_regular_x and abs(x_lower) > 1.0e-12:
        raise ValueError(
            "lower-radial axis regularity requires the lower x face to be x=0; "
            f"got {x_lower}"
        )

    def wrap_points(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).copy()
        for axis, lower, upper, periodic in (
            (0, x_lower, x_upper, periodic_axes[0]),
            (1, y_lower, y_upper, periodic_axes[1]),
            (2, z_lower, z_upper, periodic_axes[2]),
        ):
            if periodic:
                result[:, axis] = np.mod(result[:, axis] - lower, upper - lower) + lower
        return result

    def regularize_points(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).copy()
        if axis_regular_x:
            crossed_axis = result[:, 0] < x_lower
            result[crossed_axis, 0] = 2.0 * x_lower - result[crossed_axis, 0]
            result[crossed_axis, 1] += 0.5 * y_period
        return wrap_points(result)

    def signed_wrap_points(values: np.ndarray) -> np.ndarray:
        """Wrap periodic tangential coordinates while retaining signed radius."""

        result = np.asarray(values, dtype=np.float64).copy()
        tangential_axes = (
            ((0, x_lower, x_upper, periodic_axes[0]),) if not axis_regular_x else ()
        ) + (
            (1, y_lower, y_upper, periodic_axes[1]),
            (2, z_lower, z_upper, periodic_axes[2]),
        )
        for axis, lower, upper, periodic in tangential_axes:
            if periodic:
                result[:, axis] = np.mod(result[:, axis] - lower, upper - lower) + lower
        return result

    def callback_sample(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        signed_points = signed_wrap_points(values)
        sample_points = signed_points.copy()
        negative_branch = axis_regular_x & (sample_points[:, 0] < x_lower)
        if axis_regular_x:
            sample_points[:, 0] = np.abs(sample_points[:, 0] - x_lower) + x_lower
            sample_points[negative_branch, 1] += 0.5 * y_period
            sample_points = wrap_points(sample_points)
            axis_epsilon = max(1.0e-12, 1.0e-8 * abs(float(grid.x.widths[0])))
            sample_points[:, 0] = np.maximum(sample_points[:, 0], axis_epsilon)
        for axis, lower, upper, periodic in (
            (0, x_lower, x_upper, periodic_axes[0]),
            (1, y_lower, y_upper, periodic_axes[1]),
            (2, z_lower, z_upper, periodic_axes[2]),
        ):
            if not periodic:
                sample_points[:, axis] = np.clip(sample_points[:, axis], lower, upper)
        b, bmag = _callback_fci_field_values(field_evaluator, sample_points)
        if axis_regular_x:
            b = np.array(b, copy=True)
            b[negative_branch, 0] *= -1.0
        return b, bmag

    def rhs(values: np.ndarray) -> np.ndarray:
        b, _ = callback_sample(values)
        bz = b[:, 2].copy()
        small = np.abs(bz) < min_abs_bz
        bz[small] = np.where(bz[small] < 0.0, -min_abs_bz, min_abs_bz)
        return np.column_stack((b[:, 0] / bz, b[:, 1] / bz, np.ones(values.shape[0])))

    def speed(values: np.ndarray) -> np.ndarray:
        b, bmag = callback_sample(values)
        bz = b[:, 2].copy()
        small = np.abs(bz) < min_abs_bz
        bz[small] = np.where(bz[small] < 0.0, -min_abs_bz, min_abs_bz)
        return bmag / np.maximum(np.abs(bz), 1.0e-30)

    def valid(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        result = np.isfinite(values).all(axis=1)
        for axis, lower, upper, periodic in (
            (0, x_lower, x_upper, periodic_axes[0]),
            (1, y_lower, y_upper, periodic_axes[1]),
            (2, z_lower, z_upper, periodic_axes[2]),
        ):
            if not periodic:
                if axis == 0 and axis_regular_x:
                    result &= np.abs(values[:, axis] - lower) <= (upper - lower)
                else:
                    result &= (values[:, axis] >= lower) & (values[:, axis] <= upper)
        return result

    def boundary_hit(old: np.ndarray, new: np.ndarray, valid_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fractions = np.full(old.shape[0], np.inf, dtype=np.float64)
        for axis, lower, upper, periodic, ignore_lower in (
            (0, x_lower, x_upper, periodic_axes[0], axis_regular_x),
            (1, y_lower, y_upper, periodic_axes[1], False),
            (2, z_lower, z_upper, periodic_axes[2], False),
        ):
            if periodic:
                continue
            delta = new[:, axis] - old[:, axis]
            safe_delta = np.where(np.abs(delta) < 1.0e-300, 1.0, delta)
            candidate = np.full(old.shape[0], np.inf, dtype=np.float64)
            if axis == 0 and axis_regular_x:
                candidate = np.where(
                    new[:, axis] < (2.0 * lower - upper),
                    ((2.0 * lower - upper) - old[:, axis]) / safe_delta,
                    candidate,
                )
            elif not ignore_lower:
                candidate = np.where(new[:, axis] < lower, (lower - old[:, axis]) / safe_delta, candidate)
            candidate = np.minimum(
                candidate,
                np.where(new[:, axis] > upper, (upper - old[:, axis]) / safe_delta, np.inf),
            )
            candidate = np.where((candidate >= 0.0) & (candidate <= 1.0), candidate, np.inf)
            fractions = np.minimum(fractions, candidate)
        has_hit = (~valid_new) & np.isfinite(fractions)
        fraction = np.where(has_hit, fractions, 1.0)
        hit = old + fraction[:, None] * (new - old)
        for axis, lower, upper, periodic in (
            (0, x_lower, x_upper, periodic_axes[0]),
            (1, y_lower, y_upper, periodic_axes[1]),
            (2, z_lower, z_upper, periodic_axes[2]),
        ):
            if not periodic:
                hit[:, axis] = np.clip(
                    hit[:, axis],
                    2.0 * lower - upper if axis == 0 and axis_regular_x else lower,
                    upper,
                )
        return signed_wrap_points(hit), fraction

    state = signed_wrap_points(points)
    lengths = np.zeros(state.shape[0], dtype=np.float64)
    alive = np.ones(state.shape[0], dtype=bool)
    boundary = np.zeros(state.shape[0], dtype=bool)
    h = eta_step / int(substeps)

    def rk4_step(values: np.ndarray, step: float) -> np.ndarray:
        k1 = rhs(values)
        k2 = rhs(signed_wrap_points(values + 0.5 * step * k1))
        k3 = rhs(signed_wrap_points(values + 0.5 * step * k2))
        k4 = rhs(signed_wrap_points(values + step * k3))
        return values + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    for _ in range(int(substeps)):
        b0_speed = speed(state)
        raw_next = rk4_step(state, h)
        finite_next = np.isfinite(raw_next).all(axis=1)
        unwrapped_next = np.where(finite_next[:, None], raw_next, state)
        next_state = signed_wrap_points(unwrapped_next)
        valid_next = valid(next_state)
        active_full = alive & finite_next & valid_next
        active_exit = alive & finite_next & (~valid_next)
        hit_state, fraction = boundary_hit(state, unwrapped_next, valid_next)
        b1_speed = speed(next_state)
        hit_speed = speed(hit_state)
        lengths += np.where(active_full, 0.5 * abs(h) * (b0_speed + b1_speed), 0.0)
        lengths += np.where(active_exit, 0.5 * abs(h) * fraction * (b0_speed + hit_speed), 0.0)
        state = np.where(active_full[:, None], next_state, state)
        state = np.where(active_exit[:, None], hit_state, state)
        state = signed_wrap_points(state)
        boundary |= active_exit | (alive & ~finite_next)
        alive &= finite_next & valid_next

    endpoint = regularize_points(state)
    endpoint_b_contra, endpoint_bmag = callback_sample(endpoint)
    payload = {
        "endpoint": endpoint,
        "length": lengths,
        "boundary": boundary,
        "endpoint_b_contravariant": endpoint_b_contra,
        "endpoint_bmag": endpoint_bmag,
    }
    if return_numpy:
        return payload
    return {name: jnp.asarray(value) for name, value in payload.items()}


def build_fci_maps_from_callbacks(
    grid: CellCenteredGrid3D,
    field_evaluator: Callable[[np.ndarray], object],
    *,
    substeps: int = 4,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    min_abs_bz: float = 1.0e-30,
    endpoint_interpolation_order: int = 2,
    direction_checkpoint_path: Path | None = None,
    trace_batch: Callable[[np.ndarray, float], Mapping[str, object]] | None = None,
    trace_backend_identity: str = "numpy-callback",
    max_trace_batch_seeds: int = 2048,
) -> dict[str, jnp.ndarray]:
    """Build FCI maps by evaluating the magnetic field at arbitrary points.

    ``field_evaluator`` receives an ``(n, 3)`` NumPy array and returns either
    a ``MagneticFieldEvaluation``-like object with ``B_contravariant`` and
    ``magnitude`` attributes, a mapping with those keys, or a two-tuple
    ``(B_contravariant, Bmag)``.  This is the callback path intended for an
    HSX ``MetricEvaluator`` adapter, and avoids sampling the evaluator onto a
    materialized cell-centered magnetic-field array first.

    Endpoint indices are fractional cell-centered indices and therefore use
    bilinear, second-order endpoint interpolation.  Physical boundary hit
    coordinates and connection lengths are retained.  With
    ``axis_regular_axes[0]``, lower-radial crossings continue at reflected
    radius and ``theta + pi``; the upper radial face remains physical.  An
    optional ``direction_checkpoint_path`` atomically records each completed
    forward/backward direction so long continuous-field traces can resume
    without changing their numerical result.  Producer backends may provide a
    batch-local ``trace_batch`` implementation (for example a jitted, sharded
    executor); endpoint lowering and checkpoint assembly remain identical.
    """

    if int(substeps) < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")
    if int(endpoint_interpolation_order) != 2:
        raise ValueError(
            "only second-order endpoint interpolation is currently supported; "
            f"got {endpoint_interpolation_order}"
        )
    if int(max_trace_batch_seeds) < 1:
        raise ValueError("max_trace_batch_seeds must be positive")
    if len(periodic_axes) != 3 or len(axis_regular_axes) != 3:
        raise ValueError("periodic_axes and axis_regular_axes must have length 3")
    periodic_axes = tuple(bool(value) for value in periodic_axes)
    axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
    if any(axis_regular_axes[1:]):
        raise ValueError(
            "FCI axis regularity currently supports only the lower-radial x axis; "
            f"got axis_regular_axes={axis_regular_axes}"
        )
    axis_regular_x = axis_regular_axes[0]
    if axis_regular_x and (periodic_axes[0] or not periodic_axes[1] or not periodic_axes[2]):
        raise ValueError(
            "lower-radial axis regularity requires x nonperiodic and y/z periodic; "
            f"got periodic_axes={periodic_axes}"
        )
    nx, ny, nz = grid.shape
    x_axis = np.asarray(grid.x.centers, dtype=np.float64)
    y_axis = np.asarray(grid.y.centers, dtype=np.float64)
    z_axis = np.asarray(grid.z.centers, dtype=np.float64)
    x_lower, x_upper = float(grid.x.faces[0]), float(grid.x.faces[-1])
    y_lower, y_upper = float(grid.y.faces[0]), float(grid.y.faces[-1])
    z_lower, z_upper = float(grid.z.faces[0]), float(grid.z.faces[-1])
    y_period = y_upper - y_lower
    z_period = z_upper - z_lower
    if axis_regular_x and abs(x_lower) > 1.0e-12:
        raise ValueError(
            "lower-radial axis regularity requires the lower x face to be x=0; "
            f"got {x_lower}"
        )

    def wrap_points(points: np.ndarray) -> np.ndarray:
        result = np.asarray(points, dtype=np.float64).copy()
        for axis, lower, upper, periodic in (
            (0, x_lower, x_upper, periodic_axes[0]),
            (1, y_lower, y_upper, periodic_axes[1]),
            (2, z_lower, z_upper, periodic_axes[2]),
        ):
            if periodic:
                result[:, axis] = np.mod(result[:, axis] - lower, upper - lower) + lower
        return result

    def regularize_points(points: np.ndarray, *, shift_theta: bool = True) -> np.ndarray:
        result = np.asarray(points, dtype=np.float64).copy()
        if axis_regular_x:
            crossed_axis = result[:, 0] < x_lower
            result[crossed_axis, 0] = 2.0 * x_lower - result[crossed_axis, 0]
            if shift_theta:
                result[crossed_axis, 1] += 0.5 * y_period
        return wrap_points(result)

    def signed_wrap_points(points: np.ndarray) -> np.ndarray:
        """Wrap periodic coordinates while retaining signed radius."""

        result = np.asarray(points, dtype=np.float64).copy()
        for axis, lower, upper, periodic in (
            (1, y_lower, y_upper, periodic_axes[1]),
            (2, z_lower, z_upper, periodic_axes[2]),
        ):
            if periodic:
                result[:, axis] = np.mod(
                    result[:, axis] - lower, upper - lower
                ) + lower
        return result

    def callback_sample(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Integrate through the polar axis in a signed-radius chart.  On the
        # negative branch the same physical point is (|u|, theta+pi), while
        # the signed radial contravariant component changes sign.  This keeps
        # every RK stage on one smooth chart and avoids detecting an axis
        # crossing from a step endpoint after intermediate stages have
        # already crossed it.
        signed_points = signed_wrap_points(points)
        sample_points = signed_points.copy()
        negative_branch = axis_regular_x & (sample_points[:, 0] < x_lower)
        if axis_regular_x:
            sample_points[:, 0] = np.abs(sample_points[:, 0] - x_lower) + x_lower
            sample_points[negative_branch, 1] += 0.5 * y_period
            sample_points = wrap_points(sample_points)
            # Avoid asking a singular coordinate Jacobian for a value exactly
            # at u=0; the axis-regular fit supplies the smooth one-sided limit.
            axis_epsilon = max(1.0e-12, 1.0e-8 * abs(float(grid.x.widths[0])))
            sample_points[:, 0] = np.maximum(sample_points[:, 0], axis_epsilon)
        # A zero-width ghost/leg fill keeps RK stages inside the callback's
        # valid domain. The actual wall endpoint is still found geometrically.
        for axis, lower, upper, periodic in (
            (0, x_lower, x_upper, periodic_axes[0]),
            (1, y_lower, y_upper, periodic_axes[1]),
            (2, z_lower, z_upper, periodic_axes[2]),
        ):
            if not periodic:
                sample_points[:, axis] = np.clip(sample_points[:, axis], lower, upper)
        b, bmag = _callback_fci_field_values(field_evaluator, sample_points)
        if axis_regular_x:
            b = np.array(b, copy=True)
            b[negative_branch, 0] *= -1.0
        return b, bmag

    def rhs(points: np.ndarray) -> np.ndarray:
        b, _ = callback_sample(points)
        bz = b[:, 2].copy()
        small = np.abs(bz) < float(min_abs_bz)
        bz[small] = np.where(bz[small] < 0.0, -float(min_abs_bz), float(min_abs_bz))
        return np.column_stack((b[:, 0] / bz, b[:, 1] / bz, np.ones(points.shape[0])))

    def speed(points: np.ndarray) -> np.ndarray:
        b, bmag = callback_sample(points)
        bz = b[:, 2].copy()
        small = np.abs(bz) < float(min_abs_bz)
        bz[small] = np.where(bz[small] < 0.0, -float(min_abs_bz), float(min_abs_bz))
        return bmag / np.maximum(np.abs(bz), 1.0e-30)

    def valid(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        result = np.isfinite(points).all(axis=1)
        for axis, lower, upper, periodic in (
            (0, x_lower, x_upper, periodic_axes[0]),
            (1, y_lower, y_upper, periodic_axes[1]),
            (2, z_lower, z_upper, periodic_axes[2]),
        ):
            if not periodic:
                if axis == 0 and axis_regular_x:
                    result &= np.abs(points[:, axis] - lower) <= (upper - lower)
                else:
                    result &= (points[:, axis] >= lower) & (points[:, axis] <= upper)
        return result

    def boundary_hit(old: np.ndarray, new: np.ndarray, valid_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fractions = np.full(old.shape[0], np.inf, dtype=np.float64)
        for axis, lower, upper, periodic, ignore_lower in (
            (0, x_lower, x_upper, periodic_axes[0], axis_regular_x),
            (1, y_lower, y_upper, periodic_axes[1], False),
            (2, z_lower, z_upper, periodic_axes[2], False),
        ):
            if periodic:
                continue
            delta = new[:, axis] - old[:, axis]
            safe_delta = np.where(np.abs(delta) < 1.0e-300, 1.0, delta)
            candidate = np.full(old.shape[0], np.inf, dtype=np.float64)
            if axis == 0 and axis_regular_x:
                candidate = np.where(
                    new[:, axis] < (2.0 * lower - upper),
                    ((2.0 * lower - upper) - old[:, axis]) / safe_delta,
                    candidate,
                )
            elif not ignore_lower:
                candidate = np.where(new[:, axis] < lower, (lower - old[:, axis]) / safe_delta, candidate)
            candidate = np.minimum(candidate, np.where(new[:, axis] > upper, (upper - old[:, axis]) / safe_delta, np.inf))
            candidate = np.where((candidate >= 0.0) & (candidate <= 1.0), candidate, np.inf)
            fractions = np.minimum(fractions, candidate)
        has_hit = (~valid_new) & np.isfinite(fractions)
        fraction = np.where(has_hit, fractions, 1.0)
        hit = old + fraction[:, None] * (new - old)
        for axis, lower, upper, periodic in (
            (0, x_lower, x_upper, periodic_axes[0]),
            (1, y_lower, y_upper, periodic_axes[1]),
            (2, z_lower, z_upper, periodic_axes[2]),
        ):
            if not periodic:
                hit[:, axis] = np.clip(
                    hit[:, axis],
                    2.0 * lower - upper if axis == 0 and axis_regular_x else lower,
                    upper,
                )
        return signed_wrap_points(hit), fraction

    def trace(seed_points: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = signed_wrap_points(seed_points)
        lengths = np.zeros(state.shape[0], dtype=np.float64)
        alive = np.ones(state.shape[0], dtype=bool)
        boundary = np.zeros(state.shape[0], dtype=bool)
        h = float(step) / int(substeps)

        def rk4_step(
            points: np.ndarray,
            step_values: np.ndarray | float,
        ) -> np.ndarray:
            step_values = np.asarray(step_values, dtype=np.float64)
            step_points = step_values if step_values.ndim == 0 else step_values[:, None]
            k1 = rhs(points)
            k2 = rhs(signed_wrap_points(points + 0.5 * step_points * k1))
            k3 = rhs(signed_wrap_points(points + 0.5 * step_points * k2))
            k4 = rhs(signed_wrap_points(points + step_points * k3))
            return points + (step_points / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        for _ in range(int(substeps)):
            b0_speed = speed(state)
            raw_next = rk4_step(state, h)
            finite_next = np.isfinite(raw_next).all(axis=1)
            unwrapped_next = np.where(finite_next[:, None], raw_next, state)
            next_state = signed_wrap_points(unwrapped_next)
            valid_next = valid(next_state)
            active_full = alive & finite_next & valid_next
            active_exit = alive & finite_next & (~valid_next)
            # Intersect a physical boundary along the continuous RK chord.
            # Wrapping a periodic coordinate first can turn a short seam
            # crossing into a chord across almost the full periodic domain,
            # corrupting the retained tangential wall-hit coordinates.
            hit_state, fraction = boundary_hit(state, unwrapped_next, valid_next)
            b1_speed = speed(next_state)
            hit_speed = speed(hit_state)
            lengths += np.where(active_full, 0.5 * abs(h) * (b0_speed + b1_speed), 0.0)
            lengths += np.where(active_exit, 0.5 * abs(h) * fraction * (b0_speed + hit_speed), 0.0)
            state = np.where(active_full[:, None], next_state, state)
            state = np.where(active_exit[:, None], hit_state, state)
            state = signed_wrap_points(state)
            boundary |= active_exit | (alive & ~finite_next)
            alive &= finite_next & valid_next
        return regularize_points(state), lengths, boundary

    def plane_step(index: int, direction: int) -> float:
        if nz < 2:
            return float(direction)
        if direction > 0:
            if index < nz - 1:
                return float(z_axis[index + 1] - z_axis[index])
            return float((z_axis[0] + z_period) - z_axis[-1]) if periodic_axes[2] else float(z_axis[-1] - z_axis[-2])
        if index > 0:
            return float(-(z_axis[index] - z_axis[index - 1]))
        return float((z_axis[-1] - z_period) - z_axis[0]) if periodic_axes[2] else float(-(z_axis[1] - z_axis[0]))

    def fractional_index(
        axis: np.ndarray,
        values: np.ndarray,
        periodic: bool,
        faces: np.ndarray | None = None,
    ) -> np.ndarray:
        if axis.size < 2:
            return np.zeros_like(values, dtype=np.float64)
        if periodic:
            spacing = axis[1] - axis[0]
            period = (axis[-1] - axis[0]) + spacing
            wrapped = np.mod(values - axis[0], period) + axis[0]
            upper = np.searchsorted(axis, wrapped, side="right")
            lower = np.clip(upper - 1, 0, axis.size - 1)
            upper_index = (lower + 1) % axis.size
            upper_coord = np.where(lower == axis.size - 1, axis[0] + period, axis[upper_index])
            weight = np.clip((wrapped - axis[lower]) / (upper_coord - axis[lower]), 0.0, 1.0)
            return lower.astype(np.float64) + weight
        if faces is None:
            raise ValueError("faces are required for nonperiodic cell-centered indices")

        upper = np.searchsorted(axis, values, side="right")
        interior_lower = np.clip(upper - 1, 0, axis.size - 2)
        interior_weight = (values - axis[interior_lower]) / (
            axis[interior_lower + 1] - axis[interior_lower]
        )
        interior_index = interior_lower.astype(np.float64) + np.clip(interior_weight, 0.0, 1.0)

        lower_ghost = 2.0 * float(faces[0]) - axis[0]
        lower_weight = (values - lower_ghost) / (axis[0] - lower_ghost)
        lower_index = -1.0 + lower_weight

        upper_ghost = 2.0 * float(faces[-1]) - axis[-1]
        upper_weight = (values - axis[-1]) / (upper_ghost - axis[-1])
        upper_index = float(axis.size - 1) + upper_weight

        result = np.where(values < axis[0], lower_index, interior_index)
        result = np.where(values > axis[-1], upper_index, result)
        # Physical endpoints are at most one half-cell outside the center
        # range. Keeping this explicit makes the lowering contract clear.
        return np.clip(result, -0.5, float(axis.size) - 0.5)

    float_output_names = (
        "forward_x", "forward_y", "backward_x", "backward_y",
        "forward_endpoint_x", "forward_endpoint_y", "forward_endpoint_z",
        "backward_endpoint_x", "backward_endpoint_y", "backward_endpoint_z",
        "forward_endpoint_b_contra_x", "forward_endpoint_b_contra_y",
        "forward_endpoint_b_contra_z", "forward_endpoint_bmag",
        "backward_endpoint_b_contra_x", "backward_endpoint_b_contra_y",
        "backward_endpoint_b_contra_z", "backward_endpoint_bmag",
        "forward_length", "backward_length", "dz",
    )
    bool_output_names = ("forward_boundary", "backward_boundary")
    outputs = {
        name: np.empty((nx, ny, nz), dtype=np.float64)
        for name in float_output_names
    }
    outputs.update({
        name: np.empty((nx, ny, nz), dtype=bool)
        for name in bool_output_names
    })
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
    seeds_xy = np.column_stack((xx.reshape(-1), yy.reshape(-1)))

    checkpoint_path = (
        None
        if direction_checkpoint_path is None
        else Path(direction_checkpoint_path)
    )
    coordinate_digest = hashlib.sha256()
    for values in (
        x_axis,
        y_axis,
        z_axis,
        np.asarray(grid.x.faces, dtype=np.float64),
        np.asarray(grid.y.faces, dtype=np.float64),
        np.asarray(grid.z.faces, dtype=np.float64),
    ):
        coordinate_digest.update(np.ascontiguousarray(values).tobytes())
    checkpoint_identity = {
        "format_version": 1,
        "shape": [int(nx), int(ny), int(nz)],
        "substeps": int(substeps),
        "periodic_axes": [bool(value) for value in periodic_axes],
        "axis_regular_axes": [bool(value) for value in axis_regular_axes],
        "coordinate_sha256": coordinate_digest.hexdigest(),
        "trace_backend": str(trace_backend_identity),
        "max_trace_batch_seeds": int(max_trace_batch_seeds),
    }
    completed_directions: set[str] = set()

    def _direction_output_names(prefix: str) -> tuple[str, ...]:
        return tuple(
            name
            for name in (*float_output_names, *bool_output_names)
            if name.startswith(f"{prefix}_")
        )

    if checkpoint_path is not None and checkpoint_path.is_file():
        try:
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                stored_identity = json.loads(
                    str(checkpoint["identity_json"].item())
                )
                if stored_identity != checkpoint_identity:
                    raise ValueError("direction checkpoint identity mismatch")
                for prefix in ("forward", "backward"):
                    completed = bool(
                        np.asarray(
                            checkpoint[f"completed_{prefix}"]
                        ).item()
                    )
                    if not completed:
                        continue
                    for name in _direction_output_names(prefix):
                        value = np.asarray(checkpoint[name])
                        if value.shape != (nx, ny, nz):
                            raise ValueError(
                                f"direction checkpoint {name} shape mismatch"
                            )
                        outputs[name][...] = value
                    completed_directions.add(prefix)
            print(
                "[fci-map-trace] resumed completed directions "
                f"{sorted(completed_directions)} from {checkpoint_path}",
                flush=True,
            )
        except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
            print(
                f"[fci-map-trace] ignored invalid direction checkpoint "
                f"{checkpoint_path}: {error}",
                flush=True,
            )
            completed_directions.clear()

    def _write_direction_checkpoint() -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "identity_json": np.asarray(
                json.dumps(checkpoint_identity, sort_keys=True)
            ),
            "completed_forward": np.asarray(
                "forward" in completed_directions, dtype=bool
            ),
            "completed_backward": np.asarray(
                "backward" in completed_directions, dtype=bool
            ),
        }
        for prefix in sorted(completed_directions):
            for name in _direction_output_names(prefix):
                payload[name] = np.asarray(outputs[name])
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=checkpoint_path.parent,
                prefix=f".{checkpoint_path.stem}.",
                suffix=".npz",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            np.savez(temporary_path, **payload)
            temporary_path.replace(checkpoint_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        print(
            "[fci-map-trace] checkpointed completed directions "
            f"{sorted(completed_directions)} to {checkpoint_path}",
            flush=True,
        )

    # Group planes with the same signed eta step, then trace each group in
    # bounded batches.  A full 48^3 uniform map contains 110592 seeds; sending
    # all of them through every continuous metric/B evaluation at once uses
    # several gigabytes at higher RK substep counts.  Seed trajectories are
    # independent, so batching changes only peak memory.  Permit a batch to
    # split a logical eta plane as well: one 48x48 plane is already large
    # enough to trigger transient multi-gigabyte evaluator allocations.
    def step_groups(direction: int) -> list[tuple[float, list[int]]]:
        grouped: dict[float, list[int]] = {}
        for index in range(nz):
            step = plane_step(index, direction)
            key = round(float(step), 14)
            grouped.setdefault(key, []).append(index)
        return [(key, indices) for key, indices in grouped.items()]

    def fill_direction(direction: int) -> None:
        prefix = "forward" if direction > 0 else "backward"
        for step, indices in step_groups(direction):
            group_indices = np.asarray(indices, dtype=np.int64)
            seeds_per_plane = int(seeds_xy.shape[0])
            group_seed_count = int(group_indices.size) * seeds_per_plane
            for seed_start in range(
                0, group_seed_count, int(max_trace_batch_seeds)
            ):
                seed_stop = min(
                    seed_start + int(max_trace_batch_seeds), group_seed_count
                )
                group_flat = np.arange(seed_start, seed_stop, dtype=np.int64)
                plane_position = group_flat // seeds_per_plane
                plane_flat = group_flat % seeds_per_plane
                target_k = group_indices[plane_position]
                target_i = plane_flat // ny
                target_j = plane_flat % ny
                seeds = np.column_stack(
                    (
                        seeds_xy[plane_flat],
                        z_axis[target_k],
                    )
                )
                if trace_batch is None:
                    traced_result = trace_fci_points_to_plane_from_callbacks(
                        grid,
                        field_evaluator,
                        seeds,
                        step,
                        substeps=substeps,
                        periodic_axes=periodic_axes,
                        axis_regular_axes=axis_regular_axes,
                        min_abs_bz=min_abs_bz,
                    )
                else:
                    traced_result = trace_batch(seeds, step)
                traced = np.asarray(traced_result["endpoint"])
                lengths = np.asarray(traced_result["length"])
                boundaries = np.asarray(traced_result["boundary"])
                endpoint_b_contra = np.asarray(
                    traced_result["endpoint_b_contravariant"]
                )
                endpoint_bmag = np.asarray(traced_result["endpoint_bmag"])
                outputs[f"{prefix}_x"][target_i, target_j, target_k] = (
                    fractional_index(
                        x_axis,
                        traced[:, 0],
                        periodic_axes[0],
                        np.asarray(grid.x.faces),
                    )
                )
                outputs[f"{prefix}_y"][target_i, target_j, target_k] = (
                    fractional_index(
                        y_axis,
                        traced[:, 1],
                        periodic_axes[1],
                        np.asarray(grid.y.faces),
                    )
                )
                outputs[f"{prefix}_endpoint_x"][target_i, target_j, target_k] = traced[:, 0]
                outputs[f"{prefix}_endpoint_y"][target_i, target_j, target_k] = traced[:, 1]
                outputs[f"{prefix}_endpoint_z"][target_i, target_j, target_k] = traced[:, 2]
                outputs[f"{prefix}_endpoint_b_contra_x"][target_i, target_j, target_k] = endpoint_b_contra[:, 0]
                outputs[f"{prefix}_endpoint_b_contra_y"][target_i, target_j, target_k] = endpoint_b_contra[:, 1]
                outputs[f"{prefix}_endpoint_b_contra_z"][target_i, target_j, target_k] = endpoint_b_contra[:, 2]
                outputs[f"{prefix}_endpoint_bmag"][target_i, target_j, target_k] = endpoint_bmag
                outputs[f"{prefix}_length"][target_i, target_j, target_k] = lengths
                outputs[f"{prefix}_boundary"][target_i, target_j, target_k] = boundaries

    for direction, prefix in ((+1, "forward"), (-1, "backward")):
        if prefix in completed_directions:
            continue
        fill_direction(direction)
        completed_directions.add(prefix)
        _write_direction_checkpoint()
    for index in range(nz):
        outputs["dz"][:, :, index] = abs(plane_step(index, +1))

    return {name: jnp.asarray(values) for name, values in outputs.items()}


def trace_fci_eta_plane_from_callbacks(
    grid: CellCenteredGrid3D,
    field_evaluator: Callable[[np.ndarray], object],
    *,
    eta_index: int,
    direction: int,
    substeps: int = 4,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    min_abs_bz: float = 1.0e-30,
    endpoint_interpolation_order: int = 2,
) -> dict[str, jnp.ndarray]:
    """Trace one eta plane using an arbitrary magnetic-field callback."""

    if int(direction) not in (-1, 1):
        raise ValueError(f"direction must be +1 or -1, got {direction}")
    eta_index = int(eta_index)
    if eta_index < 0 or eta_index >= grid.z.n:
        raise ValueError(f"eta_index must be in [0, {grid.z.n}), got {eta_index}")
    maps = build_fci_maps_from_callbacks(
        grid,
        field_evaluator,
        substeps=substeps,
        periodic_axes=periodic_axes,
        axis_regular_axes=axis_regular_axes,
        min_abs_bz=min_abs_bz,
        endpoint_interpolation_order=endpoint_interpolation_order,
    )
    prefix = "forward" if int(direction) > 0 else "backward"
    return {
        "x_index": maps[f"{prefix}_x"][..., eta_index],
        "y_index": maps[f"{prefix}_y"][..., eta_index],
        "endpoint_x": maps[f"{prefix}_endpoint_x"][..., eta_index],
        "endpoint_y": maps[f"{prefix}_endpoint_y"][..., eta_index],
        "endpoint_z": maps[f"{prefix}_endpoint_z"][..., eta_index],
        "length": maps[f"{prefix}_length"][..., eta_index],
        "boundary": maps[f"{prefix}_boundary"][..., eta_index],
        "eta_index": jnp.asarray(eta_index, dtype=jnp.int32),
        "direction": jnp.asarray(int(direction), dtype=jnp.int32),
    }
