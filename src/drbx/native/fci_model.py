from __future__ import annotations

from dataclasses import fields, replace as dataclass_replace
from typing import Any, Callable, TypeVar

import jax
import jax.numpy as jnp

from ..geometry.fci_geometry import HaloLayout3D


FciFieldBundleT = TypeVar("FciFieldBundleT", bound="FciFieldBundle")
FciModelStateT = TypeVar("FciModelStateT", bound="FciModelState")


def update_halo_owned_slice(
    field_halo: jax.Array,
    field_owned: jax.Array,
    layout: HaloLayout3D,
) -> jax.Array:
    """Overwrite the owned interior of a halo field and preserve its halos."""

    if not isinstance(layout, HaloLayout3D):
        raise TypeError("layout must be a HaloLayout3D instance")
    field_halo = jnp.asarray(field_halo)
    field_owned = jnp.asarray(field_owned)
    if field_halo.shape != layout.cell_halo_shape:
        raise ValueError(
            "field_halo must have shape "
            f"{layout.cell_halo_shape}, got {field_halo.shape}"
        )
    if field_owned.shape != layout.owned_shape:
        raise ValueError(
            f"field_owned must have shape {layout.owned_shape}, "
            f"got {field_owned.shape}"
        )
    return field_halo.at[layout.owned_slices_cell].set(field_owned)


def inject_owned_field_to_halo(
    field_owned: jax.Array,
    layout: HaloLayout3D,
    fill_value: object = 0.0,
) -> jax.Array:
    """Allocate a halo field, fill it, and insert an owned field in its interior."""

    if not isinstance(layout, HaloLayout3D):
        raise TypeError("layout must be a HaloLayout3D instance")
    field_owned = jnp.asarray(field_owned)
    if field_owned.shape != layout.owned_shape:
        raise ValueError(
            f"field_owned must have shape {layout.owned_shape}, "
            f"got {field_owned.shape}"
        )
    field_halo = jnp.full(
        layout.cell_halo_shape,
        fill_value,
        dtype=field_owned.dtype,
    )
    return field_halo.at[layout.owned_slices_cell].set(field_owned)


def inject_owned_vector_field_to_halo(
    field_owned: jax.Array,
    layout: HaloLayout3D,
    fill_value: object = 0.0,
) -> jax.Array:
    """Insert an owned vector/trailing-component field into a halo field.

    The first three axes are the spatial axes and must have
    ``layout.owned_shape``. Any remaining axes are preserved verbatim. This
    lets one halo exchange carry all vector components in a single collective.
    """

    if not isinstance(layout, HaloLayout3D):
        raise TypeError("layout must be a HaloLayout3D instance")
    field_owned = jnp.asarray(field_owned)
    if field_owned.ndim < 4 or field_owned.shape[:3] != layout.owned_shape:
        raise ValueError(
            "field_owned must have spatial shape layout.owned_shape and at least "
            "one trailing component axis; "
            f"got {field_owned.shape}, expected prefix {layout.owned_shape}"
        )

    trailing_shape = tuple(field_owned.shape[3:])
    trailing = (slice(None),) * len(trailing_shape)
    field_halo = jnp.full(
        layout.cell_halo_shape + trailing_shape,
        fill_value,
        dtype=field_owned.dtype,
    )
    return field_halo.at[layout.owned_slices_cell + trailing].set(field_owned)


def extract_owned_field_from_halo(
    field_halo: jax.Array,
    layout: HaloLayout3D,
) -> jax.Array:
    """Extract the owned interior from a halo-padded field."""

    if not isinstance(layout, HaloLayout3D):
        raise TypeError("layout must be a HaloLayout3D instance")
    field_halo = jnp.asarray(field_halo)
    if field_halo.shape != layout.cell_halo_shape:
        raise ValueError(
            "field_halo must have shape "
            f"{layout.cell_halo_shape}, got {field_halo.shape}"
        )
    return field_halo[layout.owned_slices_cell]


def assert_matching_field_names(
    lhs: FciFieldBundle,
    rhs: FciFieldBundle,
) -> None:
    """Require two named field bundles to have identical field ordering."""

    if not isinstance(lhs, FciFieldBundle) or not isinstance(rhs, FciFieldBundle):
        raise TypeError("lhs and rhs must be FciFieldBundle instances")
    lhs_names = lhs.field_names()
    rhs_names = rhs.field_names()
    if lhs_names != rhs_names:
        raise ValueError(
            "field bundles must have matching field names and order; "
            f"got lhs={lhs_names}, rhs={rhs_names}"
        )


def inject_owned_state_to_halo(
    state_owned: FciModelStateT,
    layout: HaloLayout3D,
    fill_value: object = 0.0,
) -> FciModelStateT:
    """Convert an owned-shaped model state to a halo-shaped model state."""

    if not isinstance(layout, HaloLayout3D):
        raise TypeError("layout must be a HaloLayout3D instance")
    if not isinstance(state_owned, FciModelState):
        raise TypeError("state_owned must be an FciModelState instance")
    state_owned.assert_field_shape(layout.owned_shape)
    return state_owned.map_fields(
        lambda field: inject_owned_field_to_halo(field, layout, fill_value)
    )


def update_state_halo_owned_slices(
    state_halo: FciModelStateT,
    state_owned: FciModelStateT,
    layout: HaloLayout3D,
) -> FciModelStateT:
    """Refresh the owned interiors of all fields in a halo-shaped state."""

    if not isinstance(layout, HaloLayout3D):
        raise TypeError("layout must be a HaloLayout3D instance")
    if not isinstance(state_halo, FciModelState):
        raise TypeError("state_halo must be an FciModelState instance")
    if not isinstance(state_owned, FciModelState):
        raise TypeError("state_owned must be an FciModelState instance")
    if type(state_halo) is not type(state_owned):
        raise TypeError(
            "state_halo and state_owned must have matching concrete state types; "
            f"got {type(state_halo).__name__} and {type(state_owned).__name__}"
        )
    assert_matching_field_names(state_halo, state_owned)
    state_halo.assert_field_shape(layout.cell_halo_shape)
    state_owned.assert_field_shape(layout.owned_shape)
    owned_by_name = dict(state_owned.field_items())
    return state_halo.replace(
        **{
            name: update_halo_owned_slice(
                getattr(state_halo, name),
                owned_by_name[name],
                layout,
            )
            for name in state_halo.field_names()
        }
    )


def extract_owned_state_from_halo(
    state_halo: FciModelStateT,
    layout: HaloLayout3D,
) -> FciModelStateT:
    """Extract an owned-shaped model state from a halo-shaped state."""

    if not isinstance(layout, HaloLayout3D):
        raise TypeError("layout must be a HaloLayout3D instance")
    if not isinstance(state_halo, FciModelState):
        raise TypeError("state_halo must be an FciModelState instance")
    state_halo.assert_field_shape(layout.cell_halo_shape)
    return state_halo.map_fields(
        lambda field: extract_owned_field_from_halo(field, layout)
    )


@jax.tree_util.register_pytree_node_class
class FciFieldBundle:
    """Generic named-field PyTree bundle.

    Subclasses should be frozen dataclasses whose init fields are the named
    bundle leaves. Leaves may be arbitrary PyTree-compatible objects, not only
    arrays; this supports state-shaped boundary-condition bundles as well as
    model states.
    """

    def _field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in fields(self) if field.init)

    def _field_values(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in self._field_names())

    def tree_flatten(self):
        return self._field_values(), self._field_names()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(aux_data, children)))

    def field_names(self) -> tuple[str, ...]:
        return self._field_names()

    def field_values(self) -> tuple[Any, ...]:
        return self._field_values()

    def field_items(self) -> tuple[tuple[str, Any], ...]:
        return tuple((name, getattr(self, name)) for name in self._field_names())

    def replace(self: FciFieldBundleT, **updates: object) -> FciFieldBundleT:
        return dataclass_replace(self, **updates)

    def map_fields(
        self: FciFieldBundleT,
        fn: Callable[[Any], Any],
    ) -> FciFieldBundleT:
        return self.replace(**{name: fn(value) for name, value in self.field_items()})


@jax.tree_util.register_pytree_node_class
class FciModelState(FciFieldBundle):
    """Array-valued named-field base for FCI model states.

    Persistent states and RHS bundles should subclass this class. The generic
    named-field and PyTree behavior is inherited from ``FciFieldBundle``;
    these methods add state-specific array algebra.
    """

    def zeros_like(self: FciModelStateT) -> FciModelStateT:
        return self.map_fields(jnp.zeros_like)

    def assert_field_shape(
        self,
        expected_shape: tuple[int, ...],
    ) -> None:
        """Raise if any array-valued state field has the wrong shape."""

        expected_shape = tuple(int(size) for size in expected_shape)
        for name, value in self.field_items():
            shape = jnp.asarray(value).shape
            if shape != expected_shape:
                raise ValueError(
                    f"FciModelState field {name!r} must have shape "
                    f"{expected_shape}, got {shape}"
                )

    def axpy(
        self: FciModelStateT,
        other: FciModelStateT,
        *,
        scale: float,
    ) -> FciModelStateT:
        if type(self) is not type(other):
            raise TypeError(
                f"axpy requires matching state types, got {type(self).__name__} and {type(other).__name__}"
            )
        return self.replace(
            **{
                name: value + scale * getattr(other, name)
                for name, value in self.field_items()
            }
        )


__all__ = [
    "FciFieldBundle",
    "FciModelState",
    "assert_matching_field_names",
    "extract_owned_field_from_halo",
    "extract_owned_state_from_halo",
    "inject_owned_field_to_halo",
    "inject_owned_vector_field_to_halo",
    "inject_owned_state_to_halo",
    "update_halo_owned_slice",
    "update_state_halo_owned_slices",
]
