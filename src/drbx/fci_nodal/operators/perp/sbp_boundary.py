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

from typing import NamedTuple


class SatBoundaryData(NamedTuple):
    value: tuple | None
    normal_derivative: tuple | None = None


