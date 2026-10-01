"""Lean single-device Dirichlet solve on the exported P07 sparse operator (local solver studies).

Design: ``work/p08_step5_phi_audits_20261001/design_notes.md`` section D2.

The exported operator (``fci_perpendicular_p07_sparse.P07SparseOperator``, ``kind == "dirichlet"``) is the
linear part ``A`` of the per-volume positive operator ``-div(P_perp grad)`` plus a boundary-data block
``B``; the full action is ``A u + B g``. The Dirichlet potential problem for one field is
``A phi = rhs - B g``. This module solves it with ``solvax.krylov.gmres`` (restarted, right-preconditioned,
flexible GMRES) in the owner-volume weighted inner product ``<a, b> = sum(owner_volume * a * b)`` (the
production convention), with an optional Jacobi (diagonal) or user-supplied preconditioner.

* ``p07_linear_system(op)`` converts the scipy CSR matrix to a BCOO pytree (float64) plus the diagonal and
  owner volumes; the boundary block is *not* part of the system: pass ``boundary_term = B g`` (computed by
  the caller, e.g. ``boundary_source(op, bc)[:, 0]``).
* ``solve_p07_dirichlet`` is jit-compatible (``config`` and ``preconditioner`` static; use
  ``solve_p07_dirichlet_jit``). The reported residual is recomputed independently of the solver as
  ``||rhs_eff - A x||_M``.
* ``direct_solve_p07`` is the host (SuperLU) oracle for tests and tolerance studies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jax.experimental import sparse as jsparse
from solvax.krylov import gmres

__all__ = [
    "P07LinearSystem", "P07SolveConfig", "direct_solve_p07", "p07_linear_system",
    "solve_p07_dirichlet", "solve_p07_dirichlet_jit"]

_PRECONDITIONERS = ("none", "jacobi")


@dataclass(frozen=True)
class P07SolveConfig:
    """Static solve options (hashable, so usable as a ``jax.jit`` static argument)."""

    rtol: float = 1e-10
    atol: float = 0.0
    restart: int = 50
    max_restarts: int = 20
    preconditioner: str = "jacobi"      # "none" | "jacobi"

    def __post_init__(self) -> None:
        if self.preconditioner not in _PRECONDITIONERS:
            raise ValueError(
                f"preconditioner must be one of {_PRECONDITIONERS}, got {self.preconditioner!r}")


class P07LinearSystem(NamedTuple):
    """JAX pytree holding the Dirichlet linear part of the per-volume operator."""

    matrix: jsparse.BCOO       # (n, n)
    diagonal: jnp.ndarray      # (n,) diag(matrix), strictly positive
    owner_volume: jnp.ndarray  # (n,) M-weights


def p07_linear_system(op: Any) -> P07LinearSystem:
    """Build the JAX system from a ``P07SparseOperator`` of kind ``"dirichlet"``."""
    if op.kind != "dirichlet":
        raise ValueError(f"p07_linear_system requires a 'dirichlet' operator, got kind={op.kind!r}")
    csr = sp.csr_matrix(op.matrix, dtype=np.float64)
    n = csr.shape[0]
    if csr.shape != (n, n):
        raise ValueError(f"operator matrix must be square, got {csr.shape}")
    diagonal = np.asarray(csr.diagonal(), dtype=np.float64)
    if not np.all(diagonal > 0.0):
        raise ValueError(
            f"operator diagonal must be strictly positive (min {diagonal.min():.3e})")
    volume = np.asarray(op.owner_volume, dtype=np.float64).reshape(-1)
    if volume.shape != (n,):
        raise ValueError(f"owner_volume must have shape ({n},), got {volume.shape}")
    return P07LinearSystem(
        matrix=jsparse.BCOO.from_scipy_sparse(csr),
        diagonal=jnp.asarray(diagonal),
        owner_volume=jnp.asarray(volume))


def solve_p07_dirichlet(
    system: P07LinearSystem,
    rhs: jnp.ndarray,
    *,
    boundary_term: jnp.ndarray | None = None,
    config: P07SolveConfig = P07SolveConfig(),
    x0: jnp.ndarray | None = None,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Solve ``A x = rhs - boundary_term`` by right-preconditioned GMRES in the M-weighted norm.

    ``boundary_term`` is ``B g`` (``boundary_source(op, bc)[:, 0]``) or ``None``. ``preconditioner``
    (callable ``r -> M^{-1} r``) overrides ``config.preconditioner``. Returns ``(x, info)`` with
    ``iterations``, ``residual_norm`` (= ``||rhs_eff - A x||_M`` recomputed after the solve),
    ``relative_residual``, ``rhs_norm`` and ``converged`` (``residual_norm <= max(atol, rtol*rhs_norm)``).
    """
    matrix, diagonal, volume = system.matrix, system.diagonal, system.owner_volume
    rhs_eff = jnp.asarray(rhs, dtype=diagonal.dtype)
    if boundary_term is not None:
        rhs_eff = rhs_eff - jnp.asarray(boundary_term, dtype=diagonal.dtype)

    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        return matrix @ v

    def inner(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(volume * a * b)

    if preconditioner is not None:
        precond = preconditioner
    elif config.preconditioner == "jacobi":
        def precond(r: jnp.ndarray) -> jnp.ndarray:
            return r / diagonal
    else:
        precond = None

    sol = gmres(
        matvec, rhs_eff, x0=x0, precond=precond, inner_product=inner,
        restart=config.restart, rtol=config.rtol, atol=config.atol,
        max_restarts=config.max_restarts)
    x = sol.x
    residual = rhs_eff - matvec(x)
    residual_norm = jnp.sqrt(inner(residual, residual))
    rhs_norm = jnp.sqrt(inner(rhs_eff, rhs_eff))
    tolerance = jnp.maximum(config.atol, config.rtol * rhs_norm)
    info = {
        "iterations": sol.iterations,
        "residual_norm": residual_norm,
        "relative_residual": residual_norm / jnp.where(rhs_norm > 0.0, rhs_norm, 1.0),
        "converged": residual_norm <= tolerance,
        "rhs_norm": rhs_norm,
    }
    return x, info


solve_p07_dirichlet_jit = jax.jit(
    solve_p07_dirichlet, static_argnames=("config", "preconditioner"))


def direct_solve_p07(op: Any, rhs: np.ndarray, bc: Any = None) -> np.ndarray:
    """Host sparse-LU oracle for ``A x = rhs - B g`` (``bc`` is a ``BoundaryData`` or ``None``)."""
    rhs_eff = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if bc is not None:
        from drbx.native.fci_perpendicular_p07_sparse import boundary_source
        rhs_eff = rhs_eff - np.asarray(boundary_source(op, bc), dtype=np.float64)[:, 0]
    return np.asarray(spla.splu(sp.csc_matrix(op.matrix)).solve(rhs_eff))
