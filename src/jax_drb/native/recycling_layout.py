from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

from .array_backend import use_jax_backend
from ..solver import pack_active_fields, unpack_active_fields
from .mesh import StructuredMesh


@dataclass(frozen=True)
class RecyclingPackedStateLayout:
    """Describe how recycling fields are packed into the implicit state vector.

    The recycling implicit solvers evolve the active-domain values of several
    field arrays plus optional scalar controller integrals. This layout object
    makes that mapping explicit so the pack/unpack logic can be unit tested
    independently from the residual assembly.
    """

    field_names: tuple[str, ...]
    feedback_names: tuple[str, ...]
    active_slices: tuple[slice, slice, slice]
    active_shape: tuple[int, int, int]
    field_size: int
    field_templates: tuple[np.ndarray, ...]
    field_name_set: frozenset[str] = field(init=False, repr=False, compare=False)
    feedback_name_set: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_name_set", frozenset(self.field_names))
        object.__setattr__(self, "feedback_name_set", frozenset(self.feedback_names))


def recycling_layout_field_name_set(layout: object) -> frozenset[str]:
    """Return cached field-name membership for real or duck-typed layouts."""

    cached = getattr(layout, "field_name_set", None)
    if cached is not None:
        return cached
    return frozenset(getattr(layout, "field_names"))


def recycling_active_domain_slices(mesh: StructuredMesh) -> tuple[slice, slice, slice]:
    """Return the active-domain slices used by recycling implicit solves."""

    return (
        slice(mesh.xstart, mesh.xend + 1),
        slice(mesh.ystart, mesh.yend + 1),
        slice(None),
    )


def recycling_active_shape(mesh: StructuredMesh) -> tuple[int, int, int]:
    """Return the `(nx, ny, nz)` shape of the active recycling domain."""

    active_slices = recycling_active_domain_slices(mesh)
    return tuple(
        len(range(*active_slice.indices(axis_extent)))
        for active_slice, axis_extent in zip(active_slices, (mesh.nx, mesh.local_ny, mesh.nz), strict=True)
    )


def recycling_active_field_size(mesh: StructuredMesh) -> int:
    """Return the flattened active-domain size for a single field."""

    return int(np.prod(recycling_active_shape(mesh)))


def build_recycling_packed_state_layout(
    *,
    fields: dict[str, np.ndarray],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
) -> RecyclingPackedStateLayout:
    """Build reusable layout metadata for the recycling implicit state vector."""

    active_slices = recycling_active_domain_slices(mesh)
    active_shape = recycling_active_shape(mesh)
    field_templates = tuple(np.asarray(fields[name], dtype=np.float64) for name in field_names)
    field_size = int(np.prod(active_shape)) * len(field_names)
    return RecyclingPackedStateLayout(
        field_names=field_names,
        feedback_names=feedback_names,
        active_slices=active_slices,
        active_shape=active_shape,
        field_size=field_size,
        field_templates=field_templates,
    )


def pack_recycling_active_state(
    fields: dict[str, np.ndarray],
    *,
    feedback_integrals: dict[str, float],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    layout: RecyclingPackedStateLayout | None = None,
) -> np.ndarray:
    """Pack active-domain field values and controller integrals into one vector."""

    field_block = pack_active_fields(
        tuple(fields[name] for name in field_names),
        active_slices=(layout.active_slices if layout is not None else recycling_active_domain_slices(mesh)),
    )
    if not feedback_names:
        return field_block
    if use_jax_backend(field_block, *(feedback_integrals.get(name, 0.0) for name in feedback_names)):
        scalar_block = jnp.asarray([feedback_integrals.get(name, 0.0) for name in feedback_names], dtype=jnp.float64)
        return jnp.concatenate([field_block, scalar_block])
    scalar_block = np.asarray([feedback_integrals.get(name, 0.0) for name in feedback_names], dtype=np.float64)
    return np.concatenate([field_block, scalar_block])


def unpack_recycling_active_state(
    packed: np.ndarray,
    *,
    field_templates: dict[str, np.ndarray],
    feedback_integrals: dict[str, float],
    field_names: tuple[str, ...],
    feedback_names: tuple[str, ...],
    mesh: StructuredMesh,
    layout: RecyclingPackedStateLayout | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Restore field arrays and controller integrals from a packed state vector."""

    use_jax = use_jax_backend(packed, *(field_templates[name] for name in field_names))
    packed_array = jnp.asarray(packed, dtype=jnp.float64) if use_jax else np.asarray(packed, dtype=np.float64)
    field_size = layout.field_size if layout is not None else (recycling_active_field_size(mesh) * len(field_names))
    field_block = packed_array[:field_size]
    scalar_block = packed_array[field_size:]
    unpacked_fields = unpack_active_fields(
        field_block,
        templates=(
            layout.field_templates
            if layout is not None
            else tuple(np.asarray(field_templates[name], dtype=np.float64) for name in field_names)
        ),
        active_slices=(layout.active_slices if layout is not None else recycling_active_domain_slices(mesh)),
    )
    restored_fields = {name: value for name, value in zip(field_names, unpacked_fields, strict=True)}
    restored_integrals = {name: value if use_jax_backend(value) else float(value) for name, value in feedback_integrals.items()}
    for index, name in enumerate(feedback_names):
        restored_integrals[name] = scalar_block[index] if use_jax else float(scalar_block[index])
    return restored_fields, restored_integrals
