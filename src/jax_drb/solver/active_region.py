from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ActiveRegion:
    slices: tuple[slice, ...]
    shape: tuple[int, ...]
    size: int


def _use_jax_backend(*values: object) -> bool:
    for value in values:
        if value is None:
            continue
        module = type(value).__module__
        if hasattr(value, "aval") or module.startswith("jax") or module.startswith("jaxlib"):
            return True
    return False


def active_region_from_slices(
    template_shape: tuple[int, ...],
    active_slices: tuple[slice, ...],
) -> ActiveRegion:
    if len(template_shape) != len(active_slices):
        raise ValueError("template_shape and active_slices must have the same rank.")
    shape = tuple(
        len(range(*active_slice.indices(axis_extent)))
        for active_slice, axis_extent in zip(active_slices, template_shape, strict=True)
    )
    return ActiveRegion(
        slices=active_slices,
        shape=shape,
        size=int(np.prod(shape)),
    )


def pack_active_fields(
    fields: tuple[np.ndarray, ...],
    *,
    active_slices: tuple[slice, ...],
) -> np.ndarray:
    if not fields:
        return np.array([], dtype=np.float64)
    if _use_jax_backend(*fields):
        return jnp.concatenate(
            [jnp.asarray(field[active_slices], dtype=jnp.float64).ravel() for field in fields]
        )
    return np.concatenate(
        [np.asarray(field[active_slices], dtype=np.float64).ravel() for field in fields]
    )


def unpack_active_fields(
    packed: np.ndarray,
    *,
    templates: tuple[np.ndarray, ...],
    active_slices: tuple[slice, ...],
) -> tuple[np.ndarray, ...]:
    if not templates:
        return ()
    region = active_region_from_slices(templates[0].shape, active_slices)
    expected_size = len(templates) * region.size
    use_jax = _use_jax_backend(packed, *templates)
    packed_array = jnp.asarray(packed, dtype=jnp.float64) if use_jax else np.asarray(packed, dtype=np.float64)
    if packed_array.size != expected_size:
        raise ValueError(
            f"Packed state has size {packed_array.size}, expected {expected_size} for {len(templates)} fields."
        )

    restored: list[np.ndarray] = []
    offset = 0
    for template in templates:
        if use_jax:
            result = jnp.asarray(template, dtype=jnp.float64)
            result = result.at[active_slices].set(packed_array[offset : offset + region.size].reshape(region.shape))
        else:
            result = np.array(template, copy=True)
            result[active_slices] = packed_array[offset : offset + region.size].reshape(region.shape)
        restored.append(result)
        offset += region.size
    return tuple(restored)
