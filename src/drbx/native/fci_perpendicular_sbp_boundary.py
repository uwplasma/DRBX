"""Wall data for the nodal SBP perpendicular scheme.

``SatBoundaryData`` holds one entry per wall of the layout (``layout.walls`` order), each of shape
``(E, N_w, F)`` on that wall's face points (``E`` eta planes, ``N_w`` angular face nodes, ``F`` fields). The
Dirichlet ``value`` is what the inflow SAT penalises towards; ``normal_derivative`` is reserved for a later Neumann
SAT and unused so far.

Callables follow the convention of ``boundary_data_from_callables`` in
``fci_perpendicular_reconstruction_state``: ``dirichlet(points (Q, 3)) -> (value (Q, F), gradient (Q, 3, F))`` (only
``value`` is used) and ``normal(points (Q, 3)) -> g_N (Q, F)``; each is called once per wall with the wall's
points ``(u_face, theta, eta)`` flattened plane-major, and a missing callable leaves that part ``None``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from drbx.geometry.nodal_layout import NodalLayout, wall_points
from drbx.stencils.nodal_plan import NodalPlan


class SatBoundaryData(NamedTuple):
    value: tuple | None
    normal_derivative: tuple | None = None


def zero_sat_boundary_data(plan: NodalPlan, n_fields: int, dtype=np.float64) -> SatBoundaryData:
    """All-zero Dirichlet data (the linear part of the operator)."""
    E = plan.structure.n_eta
    value = tuple(jnp.zeros((E, wall[3], n_fields), dtype) for wall in plan.structure.walls)
    return SatBoundaryData(value, None)


def sat_boundary_data_from_callables(layout: NodalLayout, dirichlet: Callable | None = None,
                                     normal: Callable | None = None) -> SatBoundaryData:
    """Evaluate host callables once per wall at its face points (see the module docstring)."""
    E = layout.n_eta
    values, normals = [], []
    for wall in layout.walls:
        pts = wall_points(layout, wall)
        n_w = pts.shape[1]
        flat = pts.reshape(-1, 3)
        if dirichlet is not None:
            value, _gradient = dirichlet(flat)
            values.append(jnp.asarray(np.asarray(value).reshape(E, n_w, -1)))
        if normal is not None:
            normals.append(jnp.asarray(np.asarray(normal(flat)).reshape(E, n_w, -1)))
    return SatBoundaryData(tuple(values) if dirichlet is not None else None,
                           tuple(normals) if normal is not None else None)
