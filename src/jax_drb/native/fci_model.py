from __future__ import annotations

from dataclasses import fields, replace as dataclass_replace
from typing import Callable, TypeVar

import jax
import jax.numpy as jnp


FciModelStateT = TypeVar("FciModelStateT", bound="FciModelState")


@jax.tree_util.register_pytree_node_class
class FciModelState:
    """Generic PyTree base for FCI model state bundles.

    Subclasses declare only array-valued dataclass fields. The base class
    supplies the PyTree plumbing and a small set of field-wise update helpers
    that are shared by the different DRB model variants.
    """

    def _field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in fields(self) if field.init)

    def _field_values(self) -> tuple[jax.Array, ...]:
        return tuple(getattr(self, name) for name in self._field_names())

    def tree_flatten(self):
        return self._field_values(), self._field_names()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(aux_data, children)))

    def field_names(self) -> tuple[str, ...]:
        return self._field_names()

    def field_values(self) -> tuple[jax.Array, ...]:
        return self._field_values()

    def field_items(self) -> tuple[tuple[str, jax.Array], ...]:
        return tuple((name, getattr(self, name)) for name in self._field_names())

    def replace(self: FciModelStateT, **updates: object) -> FciModelStateT:
        return dataclass_replace(self, **updates)

    def map_fields(
        self: FciModelStateT,
        fn: Callable[[jax.Array], jax.Array],
    ) -> FciModelStateT:
        return self.replace(**{name: fn(value) for name, value in self.field_items()})

    def zeros_like(self: FciModelStateT) -> FciModelStateT:
        return self.map_fields(jnp.zeros_like)

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


__all__ = ["FciModelState"]
