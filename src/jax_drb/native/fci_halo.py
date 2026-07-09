"""Halo exchange for shard-local 3D fields.

This stage only fills halos at interfaces between logical shards. It does not
decide how a global physical boundary, topology boundary, or cut wall should
be represented. Those operations belong to later field-preparer stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

from ..geometry.fci_geometry import (
    SIDE_AXIS_REGULAR,
    SIDE_SIMPLE_PERIODIC,
    LocalDomain3D,
    _DataclassPyTreeMixin,
)
from .fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    LocalBoundaryFaceBC3D,
)


_pytree_base = jax.tree_util.register_pytree_node_class


@_pytree_base
@dataclass(frozen=True)
class HaloExchange3D(_DataclassPyTreeMixin):
    """Exchange face halos using JAX SPMD collectives.

    This backend is intended to run inside ``shard_map``/``pmap``-style SPMD
    code where each name in ``mesh_axis_names`` is a valid collective axis
    name. It exchanges only the six face slabs of a cell-centered field:
    halo edges and corners are deliberately left for a later topology or
    ghost-cell stage.

    Global side kinds in ``ShardSpec3D`` control whether a wrapped collective
    is allowed. Internal shard interfaces are always exchanged; global sides
    are exchanged only when marked ``SIDE_SIMPLE_PERIODIC``. Physical,
    axis-regular, topology-mapped, and unused sides are left unchanged for
    their owning later stage. With one shard on an axis, no collective is
    issued; topology and physical stages own those sides.

    For halo width ``h``, one direct neighbor exchange is sufficient when the
    owned extent on every exchanged axis is at least ``h``. Wider halos that
    span multiple neighboring shards require a different communication plan
    and are rejected here rather than being silently filled incorrectly.
    """

    mesh_axis_names: tuple[str | None, str | None, str | None]
    shard_counts: tuple[int, int, int]
    periodic_axes: tuple[bool, bool, bool]
    exchange_axes: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        mesh_axis_names = tuple(self.mesh_axis_names)
        shard_counts = tuple(int(value) for value in self.shard_counts)
        periodic_axes = tuple(bool(value) for value in self.periodic_axes)
        exchange_axes = tuple(bool(value) for value in self.exchange_axes)

        if len(mesh_axis_names) != 3:
            raise ValueError(
                "HaloExchange3D.mesh_axis_names must have length 3"
            )
        if len(shard_counts) != 3:
            raise ValueError("HaloExchange3D.shard_counts must have length 3")
        if len(periodic_axes) != 3:
            raise ValueError("HaloExchange3D.periodic_axes must have length 3")
        if len(exchange_axes) != 3:
            raise ValueError("HaloExchange3D.exchange_axes must have length 3")
        if any(count <= 0 for count in shard_counts):
            raise ValueError(
                f"HaloExchange3D.shard_counts must be positive, got {shard_counts}"
            )

        for axis, (name, count, enabled) in enumerate(
            zip(mesh_axis_names, shard_counts, exchange_axes)
        ):
            if name is not None and not isinstance(name, str):
                raise TypeError(
                    "HaloExchange3D.mesh_axis_names entries must be "
                    f"strings or None, got axis={axis}, value={name!r}"
                )
            if enabled and count > 1 and not name:
                raise ValueError(
                    "HaloExchange3D needs a non-empty mesh axis name for every "
                    "enabled decomposed axis; "
                    f"axis={axis}, shard_count={count}"
                )

        object.__setattr__(self, "mesh_axis_names", mesh_axis_names)
        object.__setattr__(self, "shard_counts", shard_counts)
        object.__setattr__(self, "periodic_axes", periodic_axes)
        object.__setattr__(self, "exchange_axes", exchange_axes)

    def __call__(
        self,
        field_halo: jnp.ndarray,
        domain: LocalDomain3D,
    ) -> jnp.ndarray:
        """Fill only regular-neighbor and decomposed simple-periodic face halos."""

        if not isinstance(domain, LocalDomain3D):
            raise TypeError("HaloExchange3D.domain must be a LocalDomain3D instance")

        if tuple(domain.shard_spec.shard_counts) != self.shard_counts:
            raise ValueError(
                "HaloExchange3D.shard_counts must match "
                "domain.shard_spec.shard_counts; "
                f"got exchange={self.shard_counts}, "
                f"domain={domain.shard_spec.shard_counts}"
            )
        if tuple(domain.periodic_axes) != self.periodic_axes:
            raise ValueError(
                "HaloExchange3D.periodic_axes must match domain.periodic_axes; "
                f"got exchange={self.periodic_axes}, domain={domain.periodic_axes}"
            )
        for axis, periodic in enumerate(self.periodic_axes):
            if periodic and (
                domain.shard_spec.lower_side_kind(axis) != SIDE_SIMPLE_PERIODIC
                or domain.shard_spec.upper_side_kind(axis) != SIDE_SIMPLE_PERIODIC
            ):
                raise ValueError(
                    "HaloExchange3D.periodic_axes requires SIDE_SIMPLE_PERIODIC "
                    f"on both global sides; axis={axis}"
                )

        field_halo = jnp.asarray(field_halo)
        expected_shape = domain.layout.cell_halo_shape
        if field_halo.shape != expected_shape:
            raise ValueError(
                f"field_halo must have shape {expected_shape}, got {field_halo.shape}"
            )

        h = int(domain.layout.halo_width)
        if h == 0:
            return field_halo

        for axis, (enabled, shard_count) in enumerate(
            zip(self.exchange_axes, self.shard_counts)
        ):
            if enabled and shard_count > 1 and h > domain.owned_shape[axis]:
                raise ValueError(
                    "HaloExchange3D requires halo_width no larger than the owned "
                    "extent on each exchanged axis; "
                    f"axis={axis}, halo_width={h}, "
                    f"owned_extent={domain.owned_shape[axis]}"
                )

        result = field_halo
        for axis in range(3):
            result = self._exchange_axis(result, domain, axis=axis)
        return result

    def _exchange_axis(
        self,
        field_halo: jnp.ndarray,
        domain: LocalDomain3D,
        *,
        axis: int,
    ) -> jnp.ndarray:
        """Exchange one coordinate-direction face slab."""

        axis = int(axis)
        if axis < 0 or axis > 2:
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

        if not self.exchange_axes[axis]:
            return field_halo

        axis_name = self.mesh_axis_names[axis]
        shard_count = int(self.shard_counts[axis])

        # A one-shard axis has no internal interface. Its periodic or physical
        # outer halos are filled by later stages.
        if shard_count <= 1 or axis_name is None:
            return field_halo

        layout = domain.layout
        h = int(layout.halo_width)
        nx, ny, nz = layout.owned_shape

        i = slice(h, h + nx)
        j = slice(h, h + ny)
        k = slice(h, h + nz)

        if axis == 0:
            lower_owned_slab = field_halo[h : h + h, j, k]
            upper_owned_slab = field_halo[h + nx - h : h + nx, j, k]
            lower_halo_index = (slice(0, h), j, k)
            upper_halo_index = (slice(h + nx, h + nx + h), j, k)
        elif axis == 1:
            lower_owned_slab = field_halo[i, h : h + h, k]
            upper_owned_slab = field_halo[i, h + ny - h : h + ny, k]
            lower_halo_index = (i, slice(0, h), k)
            upper_halo_index = (i, slice(h + ny, h + ny + h), k)
        else:
            lower_owned_slab = field_halo[i, j, h : h + h]
            upper_owned_slab = field_halo[i, j, h + nz - h : h + nz]
            lower_halo_index = (i, j, slice(0, h))
            upper_halo_index = (i, j, slice(h + nz, h + nz + h))

        shard_id = lax.axis_index(axis_name)
        # The domain metadata may be closed over by an SPMD function and can
        # therefore describe one representative shard. Use the runtime
        # collective index to decide whether this rank is on a global side;
        # the side-kind metadata describes those global sides.
        lower_global_allowed = (
            domain.shard_spec.lower_side_kind(axis) == SIDE_SIMPLE_PERIODIC
        )
        upper_global_allowed = (
            domain.shard_spec.upper_side_kind(axis) == SIDE_SIMPLE_PERIODIC
        )
        fill_lower_halo = jnp.where(
            shard_id > 0,
            True,
            lower_global_allowed,
        )
        fill_upper_halo = jnp.where(
            shard_id < shard_count - 1,
            True,
            upper_global_allowed,
        )

        # ppermute maps source shard -> destination shard. Sending the upper
        # slab toward increasing shard ids supplies a lower halo; sending the
        # lower slab toward decreasing shard ids supplies an upper halo.
        recv_from_minus = lax.ppermute(
            upper_owned_slab,
            axis_name=axis_name,
            perm=[(source, (source + 1) % shard_count) for source in range(shard_count)],
        )
        recv_from_plus = lax.ppermute(
            lower_owned_slab,
            axis_name=axis_name,
            perm=[(source, (source - 1) % shard_count) for source in range(shard_count)],
        )

        old_lower_halo = field_halo[lower_halo_index]
        old_upper_halo = field_halo[upper_halo_index]
        new_lower_halo = jnp.where(
            fill_lower_halo, recv_from_minus, old_lower_halo
        )
        new_upper_halo = jnp.where(
            fill_upper_halo, recv_from_plus, old_upper_halo
        )

        field_halo = field_halo.at[lower_halo_index].set(new_lower_halo)
        field_halo = field_halo.at[upper_halo_index].set(new_upper_halo)
        return field_halo

    def tree_flatten(self):
        return (), (
            self.mesh_axis_names,
            self.shard_counts,
            self.periodic_axes,
            self.exchange_axes,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        mesh_axis_names, shard_counts, periodic_axes, exchange_axes = aux_data
        return cls(
            mesh_axis_names=mesh_axis_names,
            shard_counts=shard_counts,
            periodic_axes=periodic_axes,
            exchange_axes=exchange_axes,
        )


@_pytree_base
@dataclass(frozen=True)
class TopologyHaloFiller3D(_DataclassPyTreeMixin):
    """Apply ordered local topology rules to a halo field.

    Rules have the callable interface ``rule(field_halo, domain)`` and own
    their side-kind selection. This stage performs no communication, never
    fills physical ghost cells, and does not promise edge/corner values unless
    an individual rule explicitly provides them.
    """

    rules: tuple[object, ...]
    error_if_no_rules: bool = False

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        if self.error_if_no_rules and not rules:
            raise ValueError(
                "TopologyHaloFiller3D.rules must be nonempty when "
                "error_if_no_rules=True"
            )
        for index, rule in enumerate(rules):
            if not callable(rule):
                raise TypeError(
                    f"TopologyHaloFiller3D.rules[{index}] must be callable"
                )
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "error_if_no_rules", bool(self.error_if_no_rules))

    def __call__(self, field_halo: jnp.ndarray, domain: LocalDomain3D) -> jnp.ndarray:
        if not isinstance(domain, LocalDomain3D):
            raise TypeError("TopologyHaloFiller3D.domain must be a LocalDomain3D instance")
        field_halo = jnp.asarray(field_halo)
        expected_shape = domain.layout.cell_halo_shape
        if field_halo.shape != expected_shape:
            raise ValueError(
                f"field_halo must have shape {expected_shape}, got {field_halo.shape}"
            )
        result = field_halo
        for rule in self.rules:
            result = rule(result, domain)
            if result.shape != expected_shape:
                raise ValueError(
                    "Topology rule returned an invalid field shape; "
                    f"expected {expected_shape}, got {result.shape}"
                )
        return result

    def tree_flatten(self):
        return (self.rules,), self.error_if_no_rules

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (rules,) = children
        return cls(rules=rules, error_if_no_rules=aux_data)


@_pytree_base
@dataclass(frozen=True)
class LocalPeriodicTopologyRule3D(_DataclassPyTreeMixin):
    """Fill undecomposed ``SIDE_SIMPLE_PERIODIC`` face halos locally."""

    fill_axes: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        fill_axes = tuple(bool(value) for value in self.fill_axes)
        if len(fill_axes) != 3:
            raise ValueError(
                "LocalPeriodicTopologyRule3D.fill_axes must have length 3"
            )
        object.__setattr__(self, "fill_axes", fill_axes)

    def __call__(self, field_halo: jnp.ndarray, domain: LocalDomain3D) -> jnp.ndarray:
        result = field_halo
        for axis in range(3):
            result = self._fill_axis(result, domain, axis=axis)
        return result

    def _fill_axis(self, field_halo, domain, *, axis: int):
        if not self.fill_axes[axis]:
            return field_halo
        spec = domain.shard_spec
        if spec.shard_counts[axis] > 1:
            return field_halo

        lower = spec.touches_lower(axis) and spec.lower_side_kind(axis) == SIDE_SIMPLE_PERIODIC
        upper = spec.touches_upper(axis) and spec.upper_side_kind(axis) == SIDE_SIMPLE_PERIODIC
        if not (lower or upper) or domain.layout.halo_width == 0:
            return field_halo

        h = domain.layout.halo_width
        ext = domain.layout.owned_shape
        owned = tuple(slice(h, h + size) for size in ext)
        result = field_halo
        if axis == 0:
            if lower:
                result = result.at[0:h, owned[1], owned[2]].set(
                    result[h + ext[0] - h : h + ext[0], owned[1], owned[2]]
                )
            if upper:
                result = result.at[h + ext[0] : h + ext[0] + h, owned[1], owned[2]].set(
                    result[h : h + h, owned[1], owned[2]]
                )
        elif axis == 1:
            if lower:
                result = result.at[owned[0], 0:h, owned[2]].set(
                    result[owned[0], h + ext[1] - h : h + ext[1], owned[2]]
                )
            if upper:
                result = result.at[owned[0], h + ext[1] : h + ext[1] + h, owned[2]].set(
                    result[owned[0], h : h + h, owned[2]]
                )
        elif axis == 2:
            if lower:
                result = result.at[owned[0], owned[1], 0:h].set(
                    result[owned[0], owned[1], h + ext[2] - h : h + ext[2]]
                )
            if upper:
                result = result.at[owned[0], owned[1], h + ext[2] : h + ext[2] + h].set(
                    result[owned[0], owned[1], h : h + h]
                )
        else:
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        return result

    def tree_flatten(self):
        return (), self.fill_axes

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(fill_axes=aux_data)


@_pytree_base
@dataclass(frozen=True)
class PolarAxisRegularScalarRule3D(_DataclassPyTreeMixin):
    """Fill scalar polar-axis regularity halos with optional angle sharding."""

    angle_axis_name: str | None
    radial_axis: int = 0
    angle_axis: int = 1
    passive_axis: int = 2
    fill_lower: bool = True
    fill_upper: bool = False
    require_even_global_angle: bool = True

    def __post_init__(self) -> None:
        axes = (int(self.radial_axis), int(self.angle_axis), int(self.passive_axis))
        if axes != (0, 1, 2):
            raise NotImplementedError(
                "PolarAxisRegularScalarRule3D currently supports only "
                "(radial_axis, angle_axis, passive_axis) = (0, 1, 2)"
            )
        if self.angle_axis_name is not None and not isinstance(self.angle_axis_name, str):
            raise TypeError(
                "angle_axis_name must be a string or None, got "
                f"{self.angle_axis_name!r}"
            )
        object.__setattr__(self, "radial_axis", axes[0])
        object.__setattr__(self, "angle_axis", axes[1])
        object.__setattr__(self, "passive_axis", axes[2])
        object.__setattr__(self, "fill_lower", bool(self.fill_lower))
        object.__setattr__(self, "fill_upper", bool(self.fill_upper))
        object.__setattr__(
            self,
            "require_even_global_angle",
            bool(self.require_even_global_angle),
        )

    def __call__(self, field_halo: jnp.ndarray, domain: LocalDomain3D) -> jnp.ndarray:
        if not isinstance(domain, LocalDomain3D):
            raise TypeError(
                "PolarAxisRegularScalarRule3D.domain must be a LocalDomain3D instance"
            )
        field_halo = jnp.asarray(field_halo)
        expected_shape = domain.layout.cell_halo_shape
        if field_halo.shape != expected_shape:
            raise ValueError(
                f"field_halo must have shape {expected_shape}, got {field_halo.shape}"
            )
        result = field_halo
        if self.fill_lower:
            result = self._fill_radial_side(result, domain, side="lower")
        if self.fill_upper:
            result = self._fill_radial_side(result, domain, side="upper")
        return result

    def _fill_radial_side(self, field_halo, domain, *, side: str):
        spec = domain.shard_spec
        if side == "lower":
            if not spec.touches_lower(self.radial_axis):
                return field_halo
            if spec.lower_side_kind(self.radial_axis) != SIDE_AXIS_REGULAR:
                return field_halo
        elif side == "upper":
            if not spec.touches_upper(self.radial_axis):
                return field_halo
            if spec.upper_side_kind(self.radial_axis) != SIDE_AXIS_REGULAR:
                return field_halo
        else:
            raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")

        angle_axis = self.angle_axis
        if spec.shard_counts[angle_axis] > 1:
            if self.angle_axis_name is None:
                raise ValueError(
                    "PolarAxisRegularScalarRule3D requires angle_axis_name when "
                    "the angle axis is sharded"
                )
            if spec.lower_side_kind(angle_axis) != SIDE_SIMPLE_PERIODIC or spec.upper_side_kind(angle_axis) != SIDE_SIMPLE_PERIODIC:
                raise ValueError(
                    "PolarAxisRegularScalarRule3D requires SIDE_SIMPLE_PERIODIC "
                    "on both global angle sides when angle is sharded"
                )

        h = int(domain.layout.halo_width)
        if h == 0:
            return field_halo
        nx, ny, nz = domain.layout.owned_shape
        if h > nx:
            raise ValueError(
                "PolarAxisRegularScalarRule3D requires halo_width <= local "
                f"radial extent; got halo_width={h}, nx={nx}"
            )

        global_angle = int(spec.global_shape[angle_axis])
        local_angle = int(domain.layout.owned_shape[angle_axis])
        angle_count = int(spec.shard_counts[angle_axis])
        if self.require_even_global_angle and global_angle % 2:
            raise ValueError(
                "PolarAxisRegularScalarRule3D requires an even global angle "
                f"count; got global_angle={global_angle}"
            )
        if global_angle != angle_count * local_angle:
            raise ValueError(
                "PolarAxisRegularScalarRule3D requires equal angle sharding; "
                f"got global_angle={global_angle}, angle_count={angle_count}, "
                f"local_angle={local_angle}"
            )

        shard_shift, local_shift = divmod(global_angle // 2, local_angle)
        if angle_count == 1:
            return self._fill_local_angle(field_halo, domain, side, local_shift)
        return self._fill_distributed_angle(
            field_halo,
            domain,
            side=side,
            shard_shift=shard_shift,
            local_shift=local_shift,
        )

    def _radial_mirror_source(self, field_halo, domain, *, side: str):
        h = int(domain.layout.halo_width)
        nx, ny, nz = domain.layout.owned_shape
        j = slice(h, h + ny)
        k = slice(h, h + nz)
        if side == "lower":
            return field_halo[h : h + h, j, k]
        if side == "upper":
            return field_halo[h + nx - h : h + nx, j, k][::-1, :, :]
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")

    def _write_radial_ghost(self, field_halo, domain, *, side: str, values):
        h = int(domain.layout.halo_width)
        nx, ny, nz = domain.layout.owned_shape
        j = slice(h, h + ny)
        k = slice(h, h + nz)
        if side == "lower":
            return field_halo.at[0:h, j, k].set(values[::-1, :, :])
        if side == "upper":
            return field_halo.at[h + nx : h + nx + h, j, k].set(values)
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")

    def _fill_local_angle(self, field_halo, domain, side, local_shift):
        source = self._radial_mirror_source(field_halo, domain, side=side)
        values = jnp.roll(source, shift=-int(local_shift), axis=1)
        return self._write_radial_ghost(field_halo, domain, side=side, values=values)

    def _fill_distributed_angle(self, field_halo, domain, *, side, shard_shift, local_shift):
        count = int(domain.shard_spec.shard_counts[self.angle_axis])
        source = self._radial_mirror_source(field_halo, domain, side=side)
        shift = int(shard_shift) % count
        received = lax.ppermute(
            source,
            axis_name=self.angle_axis_name,
            perm=[
                (source_id, (source_id - shift) % count)
                for source_id in range(count)
            ],
        )
        if int(local_shift) != 0:
            received = jnp.roll(received, shift=-int(local_shift), axis=1)
        return self._write_radial_ghost(
            field_halo,
            domain,
            side=side,
            values=received,
        )

    def tree_flatten(self):
        return (), (
            self.angle_axis_name,
            self.radial_axis,
            self.angle_axis,
            self.passive_axis,
            self.fill_lower,
            self.fill_upper,
            self.require_even_global_angle,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)


def make_default_topology_halo_filler_3d(
    *,
    angle_axis_name: str | None = None,
    radial_axis: int = 0,
    theta_axis: int = 1,
    radial_axis_lower_regular: bool = True,
    radial_axis_upper_regular: bool = False,
    fill_periodic_axes: tuple[bool, bool, bool] = (True, True, True),
) -> TopologyHaloFiller3D:
    """Build the default local periodic and scalar radial topology pipeline."""

    return TopologyHaloFiller3D(
        rules=(
            LocalPeriodicTopologyRule3D(fill_axes=fill_periodic_axes),
            PolarAxisRegularScalarRule3D(
                angle_axis_name=angle_axis_name,
                radial_axis=radial_axis,
                angle_axis=theta_axis,
                fill_lower=radial_axis_lower_regular,
                fill_upper=radial_axis_upper_regular,
            ),
        )
    )


@_pytree_base
@dataclass(frozen=True)
class GhostFillWeights1D(_DataclassPyTreeMixin):
    """Weights for one coordinate direction and one ghost-fill rule.

    ``owned_weights[r, m]`` multiplies the ``m``-th owned cell inward from
    the boundary when constructing ghost layer ``r``. ``bc_weights[r]``
    multiplies the supplied boundary value. The weights are deliberately
    supplied by the caller so they can encode nonuniform spacing and any
    desired reconstruction order.
    """

    owned_weights: jnp.ndarray
    bc_weights: jnp.ndarray

    def __post_init__(self) -> None:
        owned_weights = jnp.asarray(self.owned_weights, dtype=jnp.float64)
        bc_weights = jnp.asarray(self.bc_weights, dtype=jnp.float64)
        if owned_weights.ndim != 2:
            raise ValueError(
                "GhostFillWeights1D.owned_weights must have shape "
                "(halo_width, stencil_width)"
            )
        if bc_weights.ndim != 1:
            raise ValueError(
                "GhostFillWeights1D.bc_weights must have shape (halo_width,)"
            )
        if bc_weights.shape[0] != owned_weights.shape[0]:
            raise ValueError(
                "GhostFillWeights1D.bc_weights length must match halo_width"
            )
        object.__setattr__(self, "owned_weights", owned_weights)
        object.__setattr__(self, "bc_weights", bc_weights)

    @property
    def halo_width(self) -> int:
        return int(self.owned_weights.shape[0])

    @property
    def stencil_width(self) -> int:
        return int(self.owned_weights.shape[1])


def _validate_axis_weights(
    weights: tuple[GhostFillWeights1D, GhostFillWeights1D, GhostFillWeights1D],
    name: str,
) -> tuple[GhostFillWeights1D, GhostFillWeights1D, GhostFillWeights1D]:
    weights = tuple(weights)
    if len(weights) != 3:
        raise ValueError(f"{name} must contain one weight set per axis")
    if not all(isinstance(value, GhostFillWeights1D) for value in weights):
        raise TypeError(f"{name} entries must be GhostFillWeights1D instances")
    return weights  # type: ignore[return-value]


@_pytree_base
@dataclass(frozen=True)
class PhysicalGhostCellFiller3D(_DataclassPyTreeMixin):
    """Fill regular-coordinate physical face ghost slabs.

    The BC payload is passed to ``__call__`` and is therefore dynamic: it may
    depend on the evolving state. The three weight collections are static
    reconstruction configuration, with one entry for each coordinate axis.

    This stage fills only slabs with exactly one ghost-coordinate direction.
    It intentionally does not fill halo edge or corner pieces (cells with two
    or three ghost-coordinate directions); current operators do not require
    those values. A future topology/corner stage can own that policy.

    Only ``BC_DIRICHLET`` and ``BC_NEUMANN`` are materialized. Flux-level BCs
    are not scalar ghost-cell rules and remain available to flux operators.
    """

    dirichlet: tuple[GhostFillWeights1D, GhostFillWeights1D, GhostFillWeights1D]
    neumann_lower: tuple[GhostFillWeights1D, GhostFillWeights1D, GhostFillWeights1D]
    neumann_upper: tuple[GhostFillWeights1D, GhostFillWeights1D, GhostFillWeights1D]

    def __post_init__(self) -> None:
        dirichlet = _validate_axis_weights(self.dirichlet, "dirichlet")
        neumann_lower = _validate_axis_weights(self.neumann_lower, "neumann_lower")
        neumann_upper = _validate_axis_weights(self.neumann_upper, "neumann_upper")
        for axis in range(3):
            h = dirichlet[axis].halo_width
            if neumann_lower[axis].halo_width != h:
                raise ValueError(f"neumann_lower[{axis}] halo_width must match dirichlet")
            if neumann_upper[axis].halo_width != h:
                raise ValueError(f"neumann_upper[{axis}] halo_width must match dirichlet")
        object.__setattr__(self, "dirichlet", dirichlet)
        object.__setattr__(self, "neumann_lower", neumann_lower)
        object.__setattr__(self, "neumann_upper", neumann_upper)

    def __call__(
        self,
        field_halo: jnp.ndarray,
        domain: LocalDomain3D,
        face_bc: LocalBoundaryFaceBC3D | None,
    ) -> jnp.ndarray:
        if face_bc is None:
            return field_halo
        if not isinstance(domain, LocalDomain3D):
            raise TypeError("PhysicalGhostCellFiller3D.domain must be a LocalDomain3D")
        if not isinstance(face_bc, LocalBoundaryFaceBC3D):
            raise TypeError(
                "PhysicalGhostCellFiller3D.face_bc must be a LocalBoundaryFaceBC3D"
            )
        if face_bc.layout != domain.layout:
            raise ValueError("face_bc and domain must share the same HaloLayout3D")

        field_halo = jnp.asarray(field_halo)
        expected_shape = domain.layout.cell_halo_shape
        if field_halo.shape != expected_shape:
            raise ValueError(f"field_halo must have shape {expected_shape}, got {field_halo.shape}")

        h = int(domain.layout.halo_width)
        if h == 0:
            return field_halo
        for axis in range(3):
            if self.dirichlet[axis].halo_width != h:
                raise ValueError(f"ghost-fill weights[{axis}] halo_width must match domain halo_width")
            if self.dirichlet[axis].stencil_width > domain.owned_shape[axis]:
                raise ValueError(f"dirichlet[{axis}] stencil exceeds owned extent")
            if self.neumann_lower[axis].stencil_width > domain.owned_shape[axis]:
                raise ValueError(f"neumann_lower[{axis}] stencil exceeds owned extent")
            if self.neumann_upper[axis].stencil_width > domain.owned_shape[axis]:
                raise ValueError(f"neumann_upper[{axis}] stencil exceeds owned extent")

        result = field_halo
        for axis in range(3):
            result = self._fill_axis_side(result, domain, face_bc, axis, "lower")
            result = self._fill_axis_side(result, domain, face_bc, axis, "upper")
        return result

    def _fill_axis_side(self, field_halo, domain, face_bc, axis, side):
        if side == "lower" and not domain.has_physical_lower(axis):
            return field_halo
        if side == "upper" and not domain.has_physical_upper(axis):
            return field_halo

        kind, value, mask = self._bc_plane(face_bc, axis, side)
        old = self._ghost_slab(field_halo, domain.layout, axis, side)
        stencil_width = max(
            self.dirichlet[axis].stencil_width,
            (self.neumann_lower if side == "lower" else self.neumann_upper)[axis].stencil_width,
        )
        owned = self._owned_stencil(field_halo, domain.layout, axis, side, stencil_width)
        dghost = self._apply(self.dirichlet[axis], owned[: self.dirichlet[axis].stencil_width], value)
        nweights = (self.neumann_lower if side == "lower" else self.neumann_upper)[axis]
        nghost = self._apply(nweights, owned[: nweights.stencil_width], value)
        new = jnp.where((mask & (kind == BC_DIRICHLET))[None, ...], dghost, old)
        new = jnp.where((mask & (kind == BC_NEUMANN))[None, ...], nghost, new)
        return self._set_ghost_slab(field_halo, new, domain.layout, axis, side)

    @staticmethod
    def _apply(weights, owned, value):
        ghost = jnp.tensordot(weights.owned_weights, owned, axes=((1,), (0,)))
        return ghost + weights.bc_weights.reshape((weights.halo_width,) + (1,) * value.ndim) * value[None, ...]

    @staticmethod
    def _bc_plane(face_bc, axis, side):
        index = 0 if side == "lower" else -1
        if axis == 0:
            return face_bc.kind_x[index], face_bc.value_x[index], face_bc.mask_x[index]
        if axis == 1:
            return face_bc.kind_y[:, index, :], face_bc.value_y[:, index, :], face_bc.mask_y[:, index, :]
        return face_bc.kind_z[:, :, index], face_bc.value_z[:, :, index], face_bc.mask_z[:, :, index]

    @staticmethod
    def _owned_stencil(field, layout, axis, side, width):
        h = layout.halo_width
        ext = layout.owned_shape
        slices = [slice(h, h + n) for n in ext]
        if side == "lower":
            slices[axis] = slice(h, h + width)
            slab = field[tuple(slices)]
        else:
            slices[axis] = slice(h + ext[axis] - width, h + ext[axis])
            slab = field[tuple(slices)]
            slab = jnp.flip(slab, axis=axis)
        return jnp.moveaxis(slab, axis, 0)

    @staticmethod
    def _ghost_slab(field, layout, axis, side):
        h = layout.halo_width
        ext = layout.owned_shape
        slices = [slice(h, h + n) for n in ext]
        slices[axis] = slice(0, h) if side == "lower" else slice(h + ext[axis], h + ext[axis] + h)
        slab = field[tuple(slices)]
        if side == "lower":
            slab = jnp.flip(slab, axis=axis)
        return jnp.moveaxis(slab, axis, 0)

    @staticmethod
    def _set_ghost_slab(field, slab, layout, axis, side):
        h = layout.halo_width
        ext = layout.owned_shape
        raw = jnp.moveaxis(slab, 0, axis)
        if side == "lower":
            raw = jnp.flip(raw, axis=axis)
        slices = [slice(h, h + n) for n in ext]
        slices[axis] = slice(0, h) if side == "lower" else slice(h + ext[axis], h + ext[axis] + h)
        return field.at[tuple(slices)].set(raw)

    def tree_flatten(self):
        return (self.dirichlet, self.neumann_lower, self.neumann_upper), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


__all__ = [
    "GhostFillWeights1D",
    "HaloExchange3D",
    "LocalPeriodicTopologyRule3D",
    "PhysicalGhostCellFiller3D",
    "PolarAxisRegularScalarRule3D",
    "TopologyHaloFiller3D",
    "make_default_topology_halo_filler_3d",
]
