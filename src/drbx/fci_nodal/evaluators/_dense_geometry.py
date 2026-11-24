"""Dense metric and field containers used by the cache checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp


_pytree_base = jax.tree_util.register_pytree_node_class

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

def _require_float_shape(value, expected_shape: tuple[int, ...], name: str):
    arr = jnp.asarray(value, dtype=jnp.float64)
    if arr.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
    return arr

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
