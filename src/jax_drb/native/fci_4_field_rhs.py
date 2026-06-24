from __future__ import annotations

from dataclasses import dataclass

import jax

from ..geometry import FciGeometry3D


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Fci4FieldState:
    density: jax.Array
    omega: jax.Array
    v_ion_parallel: jax.Array
    v_electron_parallel: jax.Array

    def tree_flatten(self):
        return ((self.density, self.omega, self.v_ion_parallel, self.v_electron_parallel), None)

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


def compute_4field_rhs(
    state: Fci4FieldState,
    *,
    geometry: FciGeometry3D,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    density_source: jax.Array | None = None,
    omega_source: jax.Array | None = None,
    v_ion_parallel_source: jax.Array | None = None,
    v_electron_parallel_source: jax.Array | None = None,
) -> Fci4FieldState:
    """Placeholder for the shifted-torus 4-field FCI RHS assembly."""

    raise NotImplementedError("4-field FCI RHS assembly is not implemented yet.")
