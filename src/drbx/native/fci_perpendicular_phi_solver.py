"""Dirichlet potential (phi) inversion of the P07 perpendicular operator.

Solves ``A phi = rhs - B g`` where ``A`` is the linear part of the positive per-volume operator ``-div(P_perp grad)``
(``fci_perpendicular_p07_sparse.export_p07_sparse(plan, "dirichlet")``) and ``B g`` the boundary-data source of the
Dirichlet trace ``g``.  The solver is

* block-Jacobi over the eta planes with an exact per-plane ring-block LDU solve (float32 factors, see
  ``fci_perpendicular_plane_preconditioner``), used as a right preconditioner of
* restarted flexible GMRES (``solvax``) in the owner-volume (M) weighted inner product
  (``fci_perpendicular_p07_solve``), jitted with the preconditioner factors as an argument (not constants),
* default relative tolerance ``PHI_RTOL_DEFAULT = 1e-8`` (production choice) with warm starts from the previous
  potential (``x0``).

Typical use::

    solver = build_phi_solver(plan, raw_to_owner=raw_to_owner, n=n)      # once per geometry
    phi, info = solve_phi(solver, rhs, bc, x0=phi_previous)               # every stage

Design references: ``work/p08_step5_phi_audits_20261001/design_notes.md`` and
``work/p08_step5_solver_studies_20261001/README.md``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.fci_perpendicular_p07_solve import (
    P07LinearSystem, P07SolveConfig, p07_linear_system, solve_p07_dirichlet_with_jit)
from drbx.native.fci_perpendicular_p07_sparse import P07SparseOperator, boundary_source, export_p07_sparse
from drbx.native.fci_perpendicular_plane_preconditioner import (
    PlanePreconditioner, apply_plane_preconditioner, build_plane_preconditioner, owner_layout)
from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData

__all__ = ["PHI_RTOL_DEFAULT", "PhiSolver", "build_phi_solver", "phi_boundary_term", "phi_solver_from_operator",
           "solve_phi"]

#: production relative tolerance of the phi solve (relative to ``||rhs - B g||_M``); user decision 1 Oct 2026
PHI_RTOL_DEFAULT = 1e-8


@dataclass
class PhiSolver:
    """Everything the potential solve needs: the Dirichlet operator, its JAX system, the preconditioner and options."""

    op: P07SparseOperator
    system: P07LinearSystem
    prec: PlanePreconditioner
    config: P07SolveConfig
    setup_seconds: dict = field(default_factory=dict)


def phi_solver_from_operator(op: P07SparseOperator, raw_to_owner: np.ndarray, n: int, *,
                             factor_dtype: str = "float32", rtol: float = PHI_RTOL_DEFAULT, restart: int = 50,
                             max_restarts: int = 40) -> PhiSolver:
    """:class:`PhiSolver` for a ``"dirichlet"`` :class:`P07SparseOperator` (``raw_to_owner``: ``(n^3,)`` owner map)."""
    if op.kind != "dirichlet":
        raise ValueError(f"the phi solver needs a 'dirichlet' operator, got kind={op.kind!r}")
    seconds: dict = {}
    t0 = time.perf_counter()
    system = p07_linear_system(op)
    seconds["system"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ring, plane, theta = owner_layout(raw_to_owner, n)
    if len(ring) != op.n_owners:
        raise ValueError(f"raw_to_owner has {len(ring)} owners but the operator has {op.n_owners}")
    seconds["layout"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    prec = build_plane_preconditioner(op.matrix, ring, plane, theta, factor_dtype=factor_dtype)
    seconds["preconditioner"] = time.perf_counter() - t0
    config = P07SolveConfig(rtol=float(rtol), restart=int(restart), max_restarts=int(max_restarts))
    return PhiSolver(op=op, system=system, prec=prec, config=config, setup_seconds=seconds)


def build_phi_solver(plan: Any, *, raw_to_owner: np.ndarray, n: int, factor_dtype: str = "float32",
                     rtol: float = PHI_RTOL_DEFAULT, restart: int = 50, max_restarts: int = 40) -> PhiSolver:
    """Export the Dirichlet P07 operator of ``plan`` and build its :class:`PhiSolver` (setup seconds per phase)."""
    t0 = time.perf_counter()
    op = export_p07_sparse(plan, "dirichlet")
    t_export = time.perf_counter() - t0
    solver = phi_solver_from_operator(op, raw_to_owner, n, factor_dtype=factor_dtype, rtol=rtol, restart=restart,
                                      max_restarts=max_restarts)
    solver.setup_seconds = {"export": t_export, **solver.setup_seconds}
    return solver


def phi_boundary_term(solver: PhiSolver, bc: BoundaryData) -> np.ndarray:
    """``B g`` as ``(n_owners,)`` for the single field carried by ``bc`` (``ValueError`` if it has another count)."""
    source = np.asarray(boundary_source(solver.op, bc), dtype=np.float64)
    if source.shape[1] != 1:
        raise ValueError(f"the phi solver solves one field; bc carries {source.shape[1]}")
    return source[:, 0]


def solve_phi(solver: PhiSolver, rhs: Any, bc: BoundaryData | None = None, *,
              x0: Any = None) -> tuple[np.ndarray, dict]:
    """Solve ``A phi = rhs - B g``; ``bc`` is the Dirichlet data (``None``: ``g = 0``), ``x0`` a warm start.

    Returns ``(phi (n_owners,) float64, info)`` with ``iterations``, ``converged``, ``residual_norm``
    (``||rhs - B g - A phi||_M``, recomputed after the solve), ``relative_residual``, ``rhs_norm`` and ``seconds``
    (wall time including the device synchronisation).
    """
    n = solver.op.n_owners
    rhs_j = jnp.asarray(np.asarray(rhs, dtype=np.float64).reshape(-1))
    if rhs_j.shape != (n,):
        raise ValueError(f"rhs must have shape ({n},), got {tuple(rhs_j.shape)}")
    bt = jnp.zeros_like(rhs_j) if bc is None else jnp.asarray(phi_boundary_term(solver, bc))
    if x0 is None:
        x0_j = jnp.zeros_like(rhs_j)
    else:
        x0_j = jnp.asarray(np.asarray(x0, dtype=np.float64).reshape(-1))
        if x0_j.shape != (n,):
            raise ValueError(f"x0 must have shape ({n},), got {tuple(x0_j.shape)}")
    t0 = time.perf_counter()
    x, raw = solve_p07_dirichlet_with_jit(
        solver.system, rhs_j, solver.prec, apply=apply_plane_preconditioner, boundary_term=bt, x0=x0_j,
        config=solver.config)
    x = jax.block_until_ready(x)
    seconds = time.perf_counter() - t0
    info = {"iterations": int(raw["iterations"]), "converged": bool(raw["converged"]),
            "residual_norm": float(raw["residual_norm"]), "relative_residual": float(raw["relative_residual"]),
            "rhs_norm": float(raw["rhs_norm"]), "seconds": seconds}
    return np.asarray(x, dtype=np.float64), info
