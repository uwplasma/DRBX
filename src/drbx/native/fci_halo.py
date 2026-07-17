"""Halo exchange for shard-local 3D fields.

This stage only fills halos at interfaces between logical shards. It does not
decide how a global physical boundary, topology boundary, or cut wall should
be represented. Those operations belong to later field-preparer stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jax import lax

from ..geometry.fci_geometry import (
    FCI_DEP_CUT_WALL,
    FCI_DEP_FIELD_INTERIOR,
    FCI_DEP_INVALID,
    FCI_DEP_PHYSICAL_BOUNDARY,
    SIDE_AXIS_REGULAR,
    SIDE_SIMPLE_PERIODIC,
    LocalDomain3D,
    LocalFciDirectionMap,
    LocalFciGeometry3D,
    StencilBuilderContext,
    _DataclassPyTreeMixin,
)
from .fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    LocalBoundaryFaceBC3D,
    LocalBoundaryConditionBuilder,
    LocalBoundaryData3D,
    LocalCutWallBC3D,
    LocalCutWallGeometry3D,
    LocalCutWallValueReconstructor3D,
)
from .fci_model import (
    FciFieldBundle,
    FciModelState,
    FciModelStateT,
    assert_matching_field_names,
    inject_owned_state_to_halo,
)


_pytree_base = jax.tree_util.register_pytree_node_class


def _validate_halo_spatial_prefix(
    field_halo: jnp.ndarray,
    domain: LocalDomain3D,
    *,
    name: str = "field_halo",
) -> jnp.ndarray:
    """Validate the spatial part of a scalar or trailing-axis halo field."""

    field_halo = jnp.asarray(field_halo)
    expected_shape = tuple(int(value) for value in domain.layout.cell_halo_shape)
    if field_halo.ndim < 3 or tuple(field_halo.shape[:3]) != expected_shape:
        raise ValueError(
            f"{name} leading shape must match domain.layout.cell_halo_shape; "
            f"got {field_halo.shape}, expected prefix {expected_shape}"
        )
    return field_halo


def _trailing_slices(ndim: int) -> tuple[slice, ...]:
    """Return full slices for every non-spatial axis of a field."""

    if ndim < 3:
        raise ValueError(f"a halo field must have at least three axes, got ndim={ndim}")
    return (slice(None),) * (ndim - 3)


@_pytree_base
@dataclass(frozen=True)
class HaloExchange3D(_DataclassPyTreeMixin):
    """Exchange face halos using JAX SPMD collectives.

    This backend is intended to run inside ``shard_map``/``pmap``-style SPMD
    code where each configured ``LocalDomain3D.mesh_axis_names`` entry is a
    valid collective axis name. It exchanges only the six face slabs of a
    cell-centered field. Trailing component axes are carried unchanged, so a
    vector field can be exchanged in one call:
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

    exchange_axes: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        exchange_axes = tuple(bool(value) for value in self.exchange_axes)

        if len(exchange_axes) != 3:
            raise ValueError("HaloExchange3D.exchange_axes must have length 3")
        object.__setattr__(self, "exchange_axes", exchange_axes)

    def __call__(
        self,
        field_halo: jnp.ndarray,
        domain: LocalDomain3D,
    ) -> jnp.ndarray:
        """Fill only regular-neighbor and decomposed simple-periodic face halos."""

        if not isinstance(domain, LocalDomain3D):
            raise TypeError("HaloExchange3D.domain must be a LocalDomain3D instance")

        shard_counts = tuple(int(value) for value in domain.shard_spec.shard_counts)
        if len(domain.mesh_axis_names) != 3:
            raise ValueError("domain.mesh_axis_names must have length 3")
        for axis, (enabled, count, name) in enumerate(
            zip(self.exchange_axes, shard_counts, domain.mesh_axis_names)
        ):
            if enabled and count > 1 and not name:
                raise ValueError(
                    "HaloExchange3D requires a mesh axis name in "
                    f"domain.mesh_axis_names[{axis}] for decomposed exchange"
                )

        field_halo = _validate_halo_spatial_prefix(field_halo, domain)

        h = int(domain.layout.halo_width)
        if h == 0:
            return field_halo

        for axis, (enabled, shard_count) in enumerate(zip(self.exchange_axes, shard_counts)):
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

        axis_name = domain.mesh_axis_names[axis]
        shard_count = int(domain.shard_spec.shard_counts[axis])

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
        trailing = _trailing_slices(field_halo.ndim)

        if axis == 0:
            lower_owned_slab = field_halo[(slice(h, h + h), j, k) + trailing]
            upper_owned_slab = field_halo[(slice(h + nx - h, h + nx), j, k) + trailing]
            lower_halo_index = (slice(0, h), j, k) + trailing
            upper_halo_index = (slice(h + nx, h + nx + h), j, k) + trailing
        elif axis == 1:
            lower_owned_slab = field_halo[(i, slice(h, h + h), k) + trailing]
            upper_owned_slab = field_halo[(i, slice(h + ny - h, h + ny), k) + trailing]
            lower_halo_index = (i, slice(0, h), k) + trailing
            upper_halo_index = (i, slice(h + ny, h + ny + h), k) + trailing
        else:
            lower_owned_slab = field_halo[(i, j, slice(h, h + h)) + trailing]
            upper_owned_slab = field_halo[(i, j, slice(h + nz - h, h + nz)) + trailing]
            lower_halo_index = (i, j, slice(0, h)) + trailing
            upper_halo_index = (i, j, slice(h + nz, h + nz + h)) + trailing

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
        return (), (self.exchange_axes,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        (exchange_axes,) = aux_data
        return cls(exchange_axes=exchange_axes)


class FciCutWallValueEvaluator(Protocol):
    """Protocol for owner-local FCI cut-wall value evaluation."""

    def __call__(
        self,
        *,
        field_halo: jnp.ndarray,
        cut_wall_geometry: LocalCutWallGeometry3D | None,
        cut_wall_bc: LocalCutWallBC3D | None,
        context: StencilBuilderContext,
        value_slot: jnp.ndarray,
        active: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return one scalar cut-wall value per requested slot."""
        ...


@_pytree_base
@dataclass(frozen=True)
class LocalFciCutWallValueEvaluator(_DataclassPyTreeMixin):
    """Evaluate owner-local FCI cut-wall endpoint values.

    ``value_slot`` indexes padded owner-local cut-wall metadata. The concrete
    wall-value reconstruction is delegated to
    ``context.cut_wall_value_reconstructor`` so the exchange uses the same
    prepared-halo stencil data as the local FCI stencil builder. Dirichlet rows
    override that reconstructed value with their prescribed wall value;
    Neumann and flux-style rows use the reconstructed value directly.
    """

    def __call__(
        self,
        *,
        field_halo: jnp.ndarray,
        cut_wall_geometry: LocalCutWallGeometry3D | None,
        cut_wall_bc: LocalCutWallBC3D | None,
        context: StencilBuilderContext,
        value_slot: jnp.ndarray,
        active: jnp.ndarray,
    ) -> jnp.ndarray:
        if not isinstance(context, StencilBuilderContext):
            raise TypeError("context must be a StencilBuilderContext instance")

        field_halo = jnp.asarray(field_halo)
        if field_halo.ndim != 3:
            raise ValueError(
                "LocalFciCutWallValueEvaluator currently supports scalar "
                f"halo fields only; got shape {field_halo.shape}"
            )
        if field_halo.shape != context.layout.cell_halo_shape:
            raise ValueError(
                "field_halo must match context.layout.cell_halo_shape; "
                f"got {field_halo.shape}, expected {context.layout.cell_halo_shape}"
            )

        value_slot = jnp.asarray(value_slot, dtype=jnp.int32)
        active = jnp.asarray(active, dtype=bool)
        if active.shape != value_slot.shape:
            raise ValueError(
                "active must have the same shape as value_slot; "
                f"got active={active.shape}, value_slot={value_slot.shape}"
            )

        if cut_wall_geometry is None or cut_wall_bc is None:
            return jnp.zeros(value_slot.shape, dtype=field_halo.dtype)
        if not isinstance(cut_wall_geometry, LocalCutWallGeometry3D):
            raise TypeError(
                "cut_wall_geometry must be a LocalCutWallGeometry3D or None"
            )
        if not isinstance(cut_wall_bc, LocalCutWallBC3D):
            raise TypeError("cut_wall_bc must be a LocalCutWallBC3D or None")
        if cut_wall_geometry.max_wall_faces != cut_wall_bc.max_wall_faces:
            raise ValueError(
                "cut_wall_geometry and cut_wall_bc must have the same "
                "max_wall_faces"
            )

        max_wall_faces = int(cut_wall_geometry.max_wall_faces)
        if max_wall_faces == 0:
            return jnp.zeros(value_slot.shape, dtype=field_halo.dtype)
        value_reconstructor = context.cut_wall_value_reconstructor
        if value_reconstructor is None:
            raise ValueError(
                "context.cut_wall_value_reconstructor is required for "
                "nonempty FCI cut-wall value evaluation"
            )
        if not isinstance(value_reconstructor, LocalCutWallValueReconstructor3D):
            raise TypeError(
                "context.cut_wall_value_reconstructor must be a "
                "LocalCutWallValueReconstructor3D"
            )
        if value_reconstructor.max_wall_faces != max_wall_faces:
            raise ValueError(
                "cut_wall_value_reconstructor.max_wall_faces must match "
                "cut_wall_geometry.max_wall_faces"
            )

        safe_slot = jnp.clip(value_slot, 0, max_wall_faces - 1)
        slot_active = (
            jnp.asarray(cut_wall_geometry.active, dtype=bool)[safe_slot]
            & jnp.asarray(cut_wall_bc.active, dtype=bool)[safe_slot]
            & jnp.asarray(value_reconstructor.active, dtype=bool)[safe_slot]
        )

        wall_values = jnp.asarray(
            value_reconstructor.extrapolate(field_halo),
            dtype=field_halo.dtype,
        )
        cut_wall_kind = jnp.asarray(cut_wall_bc.kind, dtype=jnp.int32)
        cut_wall_value = jnp.asarray(cut_wall_bc.value, dtype=field_halo.dtype)
        wall_values = jnp.where(
            cut_wall_kind == BC_DIRICHLET,
            cut_wall_value,
            wall_values,
        )
        values = wall_values[safe_slot]
        return jnp.where(active & slot_active, values, jnp.zeros_like(values))

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        del children
        return cls()


@_pytree_base
@dataclass(frozen=True)
class RemoteFciDependencyExchange(_DataclassPyTreeMixin):
    """Populate remote FCI receive values for one trace direction.

    The exchange consumes the request side of ``LocalFciRemoteDependencyTable``
    and returns a local vector where request row ``q`` is receive slot ``q``.
    It communicates scalar values only; the FCI stencil builder owns endpoint
    assembly from local rows plus these returned remote values.
    """

    cut_wall_evaluator: FciCutWallValueEvaluator = LocalFciCutWallValueEvaluator()

    def __call__(
        self,
        *,
        field_halo: jnp.ndarray,
        direction: LocalFciDirectionMap,
        context: StencilBuilderContext,
    ) -> jnp.ndarray:
        if not isinstance(direction, LocalFciDirectionMap):
            raise TypeError("direction must be a LocalFciDirectionMap instance")
        if not isinstance(context, StencilBuilderContext):
            raise TypeError("context must be a StencilBuilderContext instance")
        if context.domain is None:
            raise ValueError("context.domain is required for remote FCI exchange")
        if context.layout != direction.layout:
            raise ValueError("context and direction must share the same HaloLayout3D")

        table = direction.remote
        field_halo = jnp.asarray(field_halo)
        if field_halo.ndim != 3:
            raise ValueError(
                "RemoteFciDependencyExchange currently supports scalar halo "
                f"fields only; got shape {field_halo.shape}"
            )
        if field_halo.shape != direction.layout.cell_halo_shape:
            raise ValueError(
                "field_halo must match direction.layout.cell_halo_shape; "
                f"got {field_halo.shape}, expected {direction.layout.cell_halo_shape}"
            )
        if table is None:
            return jnp.zeros((0,), dtype=field_halo.dtype)

        domain = context.domain
        self._validate_mesh_axes(domain)

        shard_counts = tuple(int(value) for value in domain.shard_spec.shard_counts)
        n_shards = shard_counts[0] * shard_counts[1] * shard_counts[2]
        n_requests = int(table.max_receive_values)

        shard_x = jnp.asarray(domain.runtime_shard_id(0), dtype=jnp.int32)
        shard_y = jnp.asarray(domain.runtime_shard_id(1), dtype=jnp.int32)
        shard_z = jnp.asarray(domain.runtime_shard_id(2), dtype=jnp.int32)
        my_shard_linear = (
            shard_z * (shard_counts[1] * shard_counts[0])
            + shard_y * shard_counts[0]
            + shard_x
        ).astype(jnp.int32)

        request_active = self._all_gather_flat(table.request_active, domain)
        request_kind = self._all_gather_flat(table.request_dependency_kind, domain)
        request_owner_linear = self._all_gather_flat(
            table.request_source_shard_linear,
            domain,
        )
        request_i = self._all_gather_flat(
            table.request_source_owner_local_i,
            domain,
        )
        request_j = self._all_gather_flat(
            table.request_source_owner_local_j,
            domain,
        )
        request_k = self._all_gather_flat(
            table.request_source_owner_local_k,
            domain,
        )
        request_value_slot = self._all_gather_flat(
            table.request_value_slot,
            domain,
        )

        owned_by_me = request_owner_linear == my_shard_linear
        supported_kind = (
            (request_kind == FCI_DEP_FIELD_INTERIOR)
            | (request_kind == FCI_DEP_PHYSICAL_BOUNDARY)
            | (request_kind == FCI_DEP_CUT_WALL)
        )
        valid_request = request_active & owned_by_me & supported_kind
        field_request = valid_request & (
            (request_kind == FCI_DEP_FIELD_INTERIOR)
            | (request_kind == FCI_DEP_PHYSICAL_BOUNDARY)
        )
        cut_wall_request = valid_request & (request_kind == FCI_DEP_CUT_WALL)

        field_values = self._sample_field_halo(
            field_halo=field_halo,
            source_i=request_i,
            source_j=request_j,
            source_k=request_k,
        )
        cut_wall_values = self.cut_wall_evaluator(
            field_halo=field_halo,
            cut_wall_geometry=context.cut_wall_geometry,
            cut_wall_bc=context.cut_wall_bc,
            context=context,
            value_slot=request_value_slot,
            active=cut_wall_request,
        )
        if cut_wall_values.shape != cut_wall_request.shape:
            raise ValueError(
                "cut_wall_evaluator must return an array with shape "
                f"{cut_wall_request.shape}, got {cut_wall_values.shape}"
            )

        owner_responses = jnp.zeros((n_shards, n_requests), dtype=field_halo.dtype)
        owner_responses = jnp.where(field_request, field_values, owner_responses)
        owner_responses = jnp.where(cut_wall_request, cut_wall_values, owner_responses)

        responses_by_requester = self._psum_over_mesh_axes(owner_responses, domain)
        remote_values = jnp.take(responses_by_requester, my_shard_linear, axis=0)
        active_remote = table.request_active & (
            table.request_dependency_kind != FCI_DEP_INVALID
        )
        return jnp.where(active_remote, remote_values, jnp.zeros_like(remote_values))

    @staticmethod
    def _validate_mesh_axes(domain: LocalDomain3D) -> None:
        for axis, (count, name) in enumerate(
            zip(domain.shard_spec.shard_counts, domain.mesh_axis_names)
        ):
            if int(count) > 1 and name is None:
                raise ValueError(
                    "RemoteFciDependencyExchange requires a mesh axis name in "
                    f"domain.mesh_axis_names[{axis}] for decomposed exchange"
                )

    @staticmethod
    def _sample_field_halo(
        *,
        field_halo: jnp.ndarray,
        source_i: jnp.ndarray,
        source_j: jnp.ndarray,
        source_k: jnp.ndarray,
    ) -> jnp.ndarray:
        nx, ny, nz = field_halo.shape
        safe_i = jnp.clip(source_i, 0, nx - 1)
        safe_j = jnp.clip(source_j, 0, ny - 1)
        safe_k = jnp.clip(source_k, 0, nz - 1)
        return field_halo[safe_i, safe_j, safe_k]

    @staticmethod
    def _all_gather_flat(value: jnp.ndarray, domain: LocalDomain3D) -> jnp.ndarray:
        gathered = value
        for shard_count, axis_name in zip(
            domain.shard_spec.shard_counts,
            domain.mesh_axis_names,
        ):
            if int(shard_count) > 1 and axis_name is not None:
                gathered = lax.all_gather(
                    gathered,
                    axis_name=axis_name,
                    axis=0,
                    tiled=False,
                )

        shard_counts = tuple(int(v) for v in domain.shard_spec.shard_counts)
        n_shards = shard_counts[0] * shard_counts[1] * shard_counts[2]
        return gathered.reshape((n_shards,) + value.shape)

    @staticmethod
    def _psum_over_mesh_axes(value: jnp.ndarray, domain: LocalDomain3D) -> jnp.ndarray:
        result = value
        for shard_count, axis_name in zip(
            domain.shard_spec.shard_counts,
            domain.mesh_axis_names,
        ):
            if int(shard_count) > 1 and axis_name is not None:
                result = lax.psum(result, axis_name=axis_name)
        return result

    def tree_flatten(self):
        return (), self.cut_wall_evaluator

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(cut_wall_evaluator=aux_data)


@_pytree_base
@dataclass(frozen=True)
class TopologyHaloFiller3D(_DataclassPyTreeMixin):
    """Apply ordered topology rules to a halo field.

    The first three axes are spatial; any trailing component axes are carried
    through the rule pipeline unchanged.

    Rules have the callable interface ``rule(field_halo, domain)``. The runner
    itself performs no communication directly; an individual rule may be
    local-only or may perform distributed collectives such as ``ppermute`` or
    ``all_to_all``. Each rule owns its side-kind checks, communication plan,
    index remaps, and write masks.

    The runner never fills physical ghost cells and does not promise
    edge/corner values unless an individual rule explicitly provides them.
    Rules are applied in order, so later rules may intentionally overwrite
    values written by earlier rules.
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
        field_halo = _validate_halo_spatial_prefix(field_halo, domain)
        expected_shape = tuple(int(value) for value in domain.layout.cell_halo_shape)
        expected_trailing_shape = tuple(field_halo.shape[3:])
        result = field_halo
        for rule in self.rules:
            result = rule(result, domain)
            if (
                result.ndim < 3
                or tuple(result.shape[:3]) != expected_shape
                or tuple(result.shape[3:]) != expected_trailing_shape
            ):
                raise ValueError(
                    "Topology rule returned an invalid field shape; expected "
                    f"prefix {expected_shape} with trailing shape "
                    f"{expected_trailing_shape}, got {result.shape}"
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
        field_halo = _validate_halo_spatial_prefix(field_halo, domain)
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

        lower = domain.runtime_touches_lower(axis) & (
            spec.lower_side_kind(axis) == SIDE_SIMPLE_PERIODIC
        )
        upper = domain.runtime_touches_upper(axis) & (
            spec.upper_side_kind(axis) == SIDE_SIMPLE_PERIODIC
        )
        if domain.layout.halo_width == 0:
            return field_halo

        h = domain.layout.halo_width
        ext = domain.layout.owned_shape
        owned = tuple(slice(h, h + size) for size in ext)
        trailing = _trailing_slices(field_halo.ndim)
        result = field_halo
        if axis == 0:
            lower_index = (slice(0, h), owned[1], owned[2]) + trailing
            upper_index = (slice(h + ext[0], h + ext[0] + h), owned[1], owned[2]) + trailing
            old_lower = result[lower_index]
            old_upper = result[upper_index]
            lower_value = result[(slice(h + ext[0] - h, h + ext[0]), owned[1], owned[2]) + trailing]
            upper_value = result[(slice(h, h + h), owned[1], owned[2]) + trailing]
            result = result.at[lower_index].set(jnp.where(lower, lower_value, old_lower))
            result = result.at[upper_index].set(jnp.where(upper, upper_value, old_upper))
        elif axis == 1:
            lower_index = (owned[0], slice(0, h), owned[2]) + trailing
            upper_index = (owned[0], slice(h + ext[1], h + ext[1] + h), owned[2]) + trailing
            old_lower = result[lower_index]
            old_upper = result[upper_index]
            lower_value = result[(owned[0], slice(h + ext[1] - h, h + ext[1]), owned[2]) + trailing]
            upper_value = result[(owned[0], slice(h, h + h), owned[2]) + trailing]
            result = result.at[lower_index].set(jnp.where(lower, lower_value, old_lower))
            result = result.at[upper_index].set(jnp.where(upper, upper_value, old_upper))
        elif axis == 2:
            lower_index = (owned[0], owned[1], slice(0, h)) + trailing
            upper_index = (owned[0], owned[1], slice(h + ext[2], h + ext[2] + h)) + trailing
            old_lower = result[lower_index]
            old_upper = result[upper_index]
            lower_value = result[(owned[0], owned[1], slice(h + ext[2] - h, h + ext[2])) + trailing]
            upper_value = result[(owned[0], owned[1], slice(h, h + h)) + trailing]
            result = result.at[lower_index].set(jnp.where(lower, lower_value, old_lower))
            result = result.at[upper_index].set(jnp.where(upper, upper_value, old_upper))
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
        # Both calls are made on every SPMD shard. Each call performs any
        # required angle collective before applying its runtime radial-side
        # write mask.
        result = self._fill_radial_side(result, domain, side="lower")
        result = self._fill_radial_side(result, domain, side="upper")
        return result

    def _fill_radial_side(self, field_halo, domain, *, side: str):
        spec = domain.shard_spec
        if side == "lower":
            do_side = bool(self.fill_lower) & domain.runtime_touches_lower(
                self.radial_axis
            ) & (
                spec.lower_side_kind(self.radial_axis) == SIDE_AXIS_REGULAR
            )
        elif side == "upper":
            do_side = bool(self.fill_upper) & domain.runtime_touches_upper(
                self.radial_axis
            ) & (
                spec.upper_side_kind(self.radial_axis) == SIDE_AXIS_REGULAR
            )
        else:
            raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")

        angle_axis = self.angle_axis
        if spec.shard_counts[self.radial_axis] > 1 and domain.mesh_axis_names[self.radial_axis] is None:
            raise ValueError(
                "PolarAxisRegularScalarRule3D requires a mesh axis name for a "
                "decomposed radial axis"
            )
        if (
            spec.lower_side_kind(angle_axis) != SIDE_SIMPLE_PERIODIC
            or spec.upper_side_kind(angle_axis) != SIDE_SIMPLE_PERIODIC
        ):
            raise ValueError(
                "PolarAxisRegularScalarRule3D requires SIDE_SIMPLE_PERIODIC "
                "on both global angle sides"
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
        if angle_count > 1 and self.angle_axis_name is None:
            raise ValueError(
                "PolarAxisRegularScalarRule3D requires angle_axis_name when "
                "the angle axis is sharded"
            )

        shard_shift, local_shift = divmod(global_angle // 2, local_angle)
        if angle_count == 1:
            values = self._fill_local_angle(field_halo, domain, side, local_shift)
        else:
            values = self._fill_distributed_angle(
                field_halo,
                domain,
                side=side,
                shard_shift=shard_shift,
                local_shift=local_shift,
            )
        return self._write_radial_ghost(
            field_halo,
            domain,
            side=side,
            values=values,
            do_side=do_side,
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

    def _write_radial_ghost(
        self,
        field_halo,
        domain,
        *,
        side: str,
        values,
        do_side,
    ):
        h = int(domain.layout.halo_width)
        nx, ny, nz = domain.layout.owned_shape
        j = slice(h, h + ny)
        k = slice(h, h + nz)
        if side == "lower":
            index = (slice(0, h), j, k)
            old = field_halo[index]
            return field_halo.at[index].set(
                jnp.where(do_side, values[::-1, :, :], old)
            )
        if side == "upper":
            index = (slice(h + nx, h + nx + h), j, k)
            old = field_halo[index]
            return field_halo.at[index].set(jnp.where(do_side, values, old))
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")

    def _fill_local_angle(self, field_halo, domain, side, local_shift):
        source = self._radial_mirror_source(field_halo, domain, side=side)
        return jnp.roll(source, shift=-int(local_shift), axis=1)

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
        return received

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


@_pytree_base
@dataclass(frozen=True)
class PolarAxisRegularVectorRule3D(_DataclassPyTreeMixin):
    """Fill polar/axis-regularity halos for contravariant 3-vectors.

    The first three axes of ``field_halo`` are spatial and the final axis has
    length three. The rule performs the same radial mirror, angular shift, and
    optional inter-shard permutation as the scalar polar rule, then applies
    ``component_transform``:

        ``V_target^i = T^i_j V_source^j``

    For contravariant logical components, ``T`` should be the Jacobian of the
    target logical coordinates with respect to the source logical coordinates.
    A constant ``(3, 3)`` transform is supported, as is a transform whose
    leading spatial dimensions broadcast to the target slab.

    This rule is intentionally separate from
    :class:`PolarAxisRegularScalarRule3D`; applying scalar polar regularity to
    a vector field would copy the components without the required transform.
    """

    axis: int
    side: str
    angular_axis: int
    mesh_axis_name: str | None
    source_shard_offset: int
    local_shift_cells: int
    component_transform: jnp.ndarray
    halo_width: int | None = None

    def __post_init__(self) -> None:
        axis = int(self.axis)
        angular_axis = int(self.angular_axis)
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        if angular_axis not in (0, 1, 2):
            raise ValueError(
                f"angular_axis must be 0, 1, or 2, got {angular_axis}"
            )
        if axis == angular_axis:
            raise ValueError(
                "angular_axis should be distinct from the polar/axis coordinate"
            )
        if self.side not in ("lower", "upper"):
            raise ValueError(
                f"side must be 'lower' or 'upper', got {self.side!r}"
            )
        if self.mesh_axis_name is not None and not isinstance(
            self.mesh_axis_name, str
        ):
            raise TypeError("mesh_axis_name must be a string or None")

        component_transform = jnp.asarray(
            self.component_transform,
            dtype=jnp.float64,
        )
        if component_transform.ndim < 2 or component_transform.shape[-2:] != (3, 3):
            raise ValueError(
                "component_transform must have shape (3, 3) or a leading-spatial "
                f"shape ending in (3, 3), got {component_transform.shape}"
            )

        halo_width = None if self.halo_width is None else int(self.halo_width)
        if halo_width is not None and halo_width < 0:
            raise ValueError(f"halo_width must be nonnegative, got {halo_width}")

        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "angular_axis", angular_axis)
        object.__setattr__(self, "source_shard_offset", int(self.source_shard_offset))
        object.__setattr__(self, "local_shift_cells", int(self.local_shift_cells))
        object.__setattr__(self, "component_transform", component_transform)
        object.__setattr__(self, "halo_width", halo_width)

    def __call__(
        self,
        field_halo: jnp.ndarray,
        domain: LocalDomain3D,
    ) -> jnp.ndarray:
        if not isinstance(domain, LocalDomain3D):
            raise TypeError(
                "PolarAxisRegularVectorRule3D requires LocalDomain3D, "
                f"got {type(domain).__name__}"
            )

        field_halo = _validate_halo_spatial_prefix(field_halo, domain)
        if field_halo.ndim != 4 or field_halo.shape[-1] != 3:
            raise ValueError(
                "PolarAxisRegularVectorRule3D requires a field with shape "
                "cell_halo_shape + (3,), "
                f"got {field_halo.shape}"
            )

        axis = self.axis
        angular_axis = self.angular_axis
        spec = domain.shard_spec
        if spec.shard_counts[axis] > 1 and domain.mesh_axis_names[axis] is None:
            raise ValueError(
                "PolarAxisRegularVectorRule3D requires a mesh axis name for a "
                "decomposed polar axis"
            )
        if (
            spec.lower_side_kind(angular_axis) != SIDE_SIMPLE_PERIODIC
            or spec.upper_side_kind(angular_axis) != SIDE_SIMPLE_PERIODIC
        ):
            raise ValueError(
                "PolarAxisRegularVectorRule3D requires SIDE_SIMPLE_PERIODIC "
                "on both global angular sides"
            )

        side_kind = (
            spec.lower_side_kind(axis)
            if self.side == "lower"
            else spec.upper_side_kind(axis)
        )
        if side_kind != SIDE_AXIS_REGULAR:
            raise ValueError(
                "PolarAxisRegularVectorRule3D requires SIDE_AXIS_REGULAR on "
                f"the selected {self.side} side of axis {axis}"
            )

        h = int(domain.layout.halo_width if self.halo_width is None else self.halo_width)
        if h == 0:
            return field_halo
        if h > domain.layout.halo_width:
            raise ValueError(
                "PolarAxisRegularVectorRule3D.halo_width cannot exceed the "
                f"field halo width; got {h}, layout halo width={domain.layout.halo_width}"
            )
        if h > domain.owned_shape[axis]:
            raise ValueError(
                "PolarAxisRegularVectorRule3D requires halo_width no larger "
                f"than the local polar extent; got h={h}, extent={domain.owned_shape[axis]}"
            )

        global_angle = int(spec.global_shape[angular_axis])
        local_angle = int(domain.owned_shape[angular_axis])
        angle_count = int(spec.shard_counts[angular_axis])
        if global_angle % 2:
            raise ValueError(
                "PolarAxisRegularVectorRule3D requires an even global angular "
                f"count; got {global_angle}"
            )
        if global_angle != angle_count * local_angle:
            raise ValueError(
                "PolarAxisRegularVectorRule3D requires equal angular sharding; "
                f"got global_angle={global_angle}, angle_count={angle_count}, "
                f"local_angle={local_angle}"
            )
        if angle_count > 1 and self.mesh_axis_name is None:
            raise ValueError(
                "PolarAxisRegularVectorRule3D requires mesh_axis_name when the "
                "angular axis is sharded"
            )
        if (
            angle_count > 1
            and self.mesh_axis_name != domain.mesh_axis_names[angular_axis]
        ):
            raise ValueError(
                "PolarAxisRegularVectorRule3D.mesh_axis_name must match the "
                f"domain mesh name for angular axis {angular_axis}; "
                f"got rule={self.mesh_axis_name!r}, "
                f"domain={domain.mesh_axis_names[angular_axis]!r}"
            )

        ext = tuple(int(value) for value in domain.layout.owned_shape)
        spatial_owned = tuple(slice(h, h + size) for size in ext)
        source_spatial = list(spatial_owned)
        target_spatial = list(spatial_owned)
        if self.side == "lower":
            do_write = domain.runtime_touches_lower(axis)
            source_spatial[axis] = slice(h, h + h)
            target_spatial[axis] = slice(0, h)
        else:
            do_write = domain.runtime_touches_upper(axis)
            source_spatial[axis] = slice(h + ext[axis] - h, h + ext[axis])
            target_spatial[axis] = slice(h + ext[axis], h + ext[axis] + h)

        trailing = _trailing_slices(field_halo.ndim)
        source_index = tuple(source_spatial) + trailing
        target_index = tuple(target_spatial) + trailing
        recv = field_halo[source_index]

        if angle_count > 1:
            recv = lax.ppermute(
                recv,
                axis_name=self.mesh_axis_name,
                perm=[
                    (
                        source,
                        (source + self.source_shard_offset) % angle_count,
                    )
                    for source in range(angle_count)
                ],
            )

        if self.local_shift_cells:
            recv = jnp.roll(
                recv,
                shift=-self.local_shift_cells,
                axis=angular_axis,
            )

        # Match the scalar rule's radial ordering. The lower and upper source
        # slabs are both reversed before being written into the corresponding
        # ghost slab; this is essential when halo_width > 1.
        recv = jnp.flip(recv, axis=axis)

        transform = jnp.asarray(self.component_transform, dtype=field_halo.dtype)
        target_transform_shape = recv.shape[:-1] + (3, 3)
        try:
            transform = jnp.broadcast_to(transform, target_transform_shape)
        except ValueError as exc:
            raise ValueError(
                "component_transform must be broadcastable to the target polar "
                f"slab matrix shape {target_transform_shape}, got {transform.shape}"
            ) from exc
        recv = jnp.einsum("...ij,...j->...i", transform, recv)

        old = field_halo[target_index]
        return field_halo.at[target_index].set(jnp.where(do_write, recv, old))

    def tree_flatten(self):
        return (self.component_transform,), (
            self.axis,
            self.side,
            self.angular_axis,
            self.mesh_axis_name,
            self.source_shard_offset,
            self.local_shift_cells,
            self.halo_width,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            axis,
            side,
            angular_axis,
            mesh_axis_name,
            source_shard_offset,
            local_shift_cells,
            halo_width,
        ) = aux_data
        (component_transform,) = children
        return cls(
            axis=axis,
            side=side,
            angular_axis=angular_axis,
            mesh_axis_name=mesh_axis_name,
            source_shard_offset=source_shard_offset,
            local_shift_cells=local_shift_cells,
            component_transform=component_transform,
            halo_width=halo_width,
        )


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

    def tree_flatten(self):
        return (self.owned_weights, self.bc_weights), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        owned_weights, bc_weights = children
        return cls(
            owned_weights=owned_weights,
            bc_weights=bc_weights,
        )


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
        if side == "lower":
            side_active = domain.runtime_has_physical_lower(axis)
        elif side == "upper":
            side_active = domain.runtime_has_physical_upper(axis)
        else:
            raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")

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
        active_mask = side_active & mask
        new = jnp.where(
            (active_mask & (kind == BC_DIRICHLET))[None, ...],
            dghost,
            old,
        )
        new = jnp.where(
            (active_mask & (kind == BC_NEUMANN))[None, ...],
            nghost,
            new,
        )
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


@_pytree_base
@dataclass(frozen=True)
class PreparedLocalState3D(_DataclassPyTreeMixin):
    """Fully prepared local state and its model-shaped boundary payloads."""

    state_halo: FciModelState
    boundary_data: LocalBoundaryData3D

    def __post_init__(self) -> None:
        if not isinstance(self.state_halo, FciModelState):
            raise TypeError("PreparedLocalState3D.state_halo must be an FciModelState")
        if not isinstance(self.boundary_data, LocalBoundaryData3D):
            raise TypeError(
                "PreparedLocalState3D.boundary_data must be a LocalBoundaryData3D"
            )


@_pytree_base
@dataclass(frozen=True)
class LocalStateAndBoundaryPreparer3D(_DataclassPyTreeMixin):
    """Prepare state halos, coupled BCs, and physical ghost cells in order.

    The pre-BC stages operate field-by-field, while the boundary builder sees
    the complete topology-prepared state so it can construct coupled payloads.
    Physical ghost filling then applies the matching face BC to each field.
    Halo edges and corners remain governed by the individual field stages;
    this orchestrator does not add corner handling.
    """

    boundary_builder: LocalBoundaryConditionBuilder
    physical_ghost_filler: PhysicalGhostCellFiller3D
    halo_exchange: HaloExchange3D | None = None
    topology_filler: TopologyHaloFiller3D | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_builder, LocalBoundaryConditionBuilder):
            raise TypeError(
                "boundary_builder must be a LocalBoundaryConditionBuilder"
            )
        if not isinstance(self.physical_ghost_filler, PhysicalGhostCellFiller3D):
            raise TypeError(
                "physical_ghost_filler must be a PhysicalGhostCellFiller3D"
            )
        if self.halo_exchange is not None and not isinstance(
            self.halo_exchange, HaloExchange3D
        ):
            raise TypeError("halo_exchange must be a HaloExchange3D or None")
        if self.topology_filler is not None and not isinstance(
            self.topology_filler, TopologyHaloFiller3D
        ):
            raise TypeError(
                "topology_filler must be a TopologyHaloFiller3D or None"
            )

    def __call__(
        self,
        state_owned: FciModelStateT,
        geometry: LocalFciGeometry3D,
        domain: LocalDomain3D,
        cut_wall_geometry: LocalCutWallGeometry3D | None = None,
    ) -> PreparedLocalState3D:
        if not isinstance(state_owned, FciModelState):
            raise TypeError("state_owned must be an FciModelState instance")
        if not isinstance(domain, LocalDomain3D):
            raise TypeError("domain must be a LocalDomain3D instance")

        state_halo = inject_owned_state_to_halo(state_owned, domain.layout)

        if self.halo_exchange is not None:
            state_halo = state_halo.map_fields(
                lambda field_halo: self.halo_exchange(field_halo, domain)
            )
        if self.topology_filler is not None:
            state_halo = state_halo.map_fields(
                lambda field_halo: self.topology_filler(field_halo, domain)
            )

        boundary_data = self.boundary_builder(
            state_halo,
            geometry,
            domain,
            cut_wall_geometry,
        )
        if boundary_data.face_bc is None:
            raise ValueError(
                "boundary_builder must return face_bc for every state field "
                "before physical ghost preparation"
            )
        face_bc_bundle = boundary_data.face_bc
        assert_matching_field_names(state_halo, face_bc_bundle)

        state_halo_full = state_halo.replace(
            **{
                name: self.physical_ghost_filler(
                    getattr(state_halo, name),
                    domain,
                    getattr(face_bc_bundle, name),
                )
                for name in state_halo.field_names()
            }
        )
        return PreparedLocalState3D(
            state_halo=state_halo_full,
            boundary_data=boundary_data,
        )

__all__ = [
    "FciCutWallValueEvaluator",
    "GhostFillWeights1D",
    "HaloExchange3D",
    "LocalFciCutWallValueEvaluator",
    "LocalPeriodicTopologyRule3D",
    "LocalStateAndBoundaryPreparer3D",
    "PreparedLocalState3D",
    "PhysicalGhostCellFiller3D",
    "PolarAxisRegularScalarRule3D",
    "PolarAxisRegularVectorRule3D",
    "RemoteFciDependencyExchange",
    "TopologyHaloFiller3D",
    "make_default_topology_halo_filler_3d",
]
