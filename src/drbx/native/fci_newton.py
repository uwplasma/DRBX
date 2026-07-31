"""Shard-compatible matrix-free SOLVAX Newton--FGMRES adapter.

The production EB IMEX stages are PyTrees of owned local fields.  SOLVAX's
default vector operations are per-device, which is not the nonlinear norm or
Arnoldi product of a sharded simulation.  This module supplies the collective,
physical-volume-weighted operations needed inside ``jax.shard_map``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
from solvax.implicit import newton_krylov as solvax_newton_krylov

from ..geometry import LocalDomain3D, LocalFciGeometry3D
from .fci_gmres import _local_cell_volume_weights, _mesh_axis_names, _spmd_sum


PyTree = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SolvaxNewtonConfig:
    """Static configuration for shard-compatible SOLVAX Newton--FGMRES.

    ``field_scales`` contains positive residual/unknown scales in PyTree leaf
    order.  An empty tuple means one for every leaf; one entry broadcasts to
    all leaves.  It is deliberately static because the number of Krylov
    allocations and the residual-tree structure must not change during a
    compiled advance.
    """

    rtol: float = 1.0e-8
    atol: float = 1.0e-10
    acceptance_rtol: float | None = None
    acceptance_atol: float | None = None
    max_steps: int = 12
    linear_restart: int = 20
    linear_rtol: float = 1.0e-2
    linear_atol: float = 0.0
    linear_max_restarts: int = 4
    field_scales: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if int(self.max_steps) < 0:
            raise ValueError("SolvaxNewtonConfig.max_steps must be nonnegative")
        if int(self.linear_restart) <= 0:
            raise ValueError("SolvaxNewtonConfig.linear_restart must be positive")
        if int(self.linear_max_restarts) <= 0:
            raise ValueError(
                "SolvaxNewtonConfig.linear_max_restarts must be positive"
            )
        for name, value in (
            ("rtol", self.rtol),
            ("atol", self.atol),
            ("linear_rtol", self.linear_rtol),
            ("linear_atol", self.linear_atol),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"SolvaxNewtonConfig.{name} must be finite and nonnegative")
        acceptance_rtol = self.rtol if self.acceptance_rtol is None else self.acceptance_rtol
        acceptance_atol = self.atol if self.acceptance_atol is None else self.acceptance_atol
        for name, value in (
            ("acceptance_rtol", acceptance_rtol),
            ("acceptance_atol", acceptance_atol),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"SolvaxNewtonConfig.{name} must be finite and nonnegative")
        scales = tuple(float(value) for value in self.field_scales)
        if any((not math.isfinite(value)) or value <= 0.0 for value in scales):
            raise ValueError("SolvaxNewtonConfig.field_scales must be finite and positive")
        object.__setattr__(self, "rtol", float(self.rtol))
        object.__setattr__(self, "atol", float(self.atol))
        object.__setattr__(self, "acceptance_rtol", float(acceptance_rtol))
        object.__setattr__(self, "acceptance_atol", float(acceptance_atol))
        object.__setattr__(self, "max_steps", int(self.max_steps))
        object.__setattr__(self, "linear_restart", int(self.linear_restart))
        object.__setattr__(self, "linear_rtol", float(self.linear_rtol))
        object.__setattr__(self, "linear_atol", float(self.linear_atol))
        object.__setattr__(self, "linear_max_restarts", int(self.linear_max_restarts))
        object.__setattr__(self, "field_scales", scales)

    def tree_flatten(self):
        return (), (
            self.rtol, self.atol, self.acceptance_rtol, self.acceptance_atol,
            self.max_steps, self.linear_restart, self.linear_rtol,
            self.linear_atol, self.linear_max_restarts, self.field_scales,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(*aux_data)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SolvaxNewtonInfo:
    """Array-valued diagnostics returned by :func:`solvax_newton_solve`."""

    newton_iterations: jax.Array
    linear_iterations: jax.Array
    converged: jax.Array
    linear_converged: jax.Array
    accepted: jax.Array
    failed: jax.Array
    initial_residual_l2: jax.Array
    final_residual_l2: jax.Array
    final_residual_rel_l2: jax.Array
    initial_state_is_finite: jax.Array
    final_state_is_finite: jax.Array
    initial_residual_is_finite: jax.Array
    final_residual_is_finite: jax.Array

    def tree_flatten(self):
        return (
            self.newton_iterations, self.linear_iterations, self.converged,
            self.linear_converged, self.accepted, self.failed,
            self.initial_residual_l2, self.final_residual_l2,
            self.final_residual_rel_l2, self.initial_state_is_finite,
            self.final_state_is_finite, self.initial_residual_is_finite,
            self.final_residual_is_finite,
        ), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)


def _active_mask(
    active_cell_mask: jnp.ndarray | None,
    geometry: LocalFciGeometry3D,
) -> jnp.ndarray | None:
    if active_cell_mask is None:
        return None
    mask = jnp.asarray(active_cell_mask, dtype=bool)
    if mask.shape != geometry.owned_shape:
        raise ValueError(
            "active_cell_mask must have shape "
            f"{geometry.owned_shape}, got {mask.shape}"
        )
    return mask


def _validate_tree(tree: PyTree, geometry: LocalFciGeometry3D, *, name: str) -> tuple[list[Any], jax.tree_util.PyTreeDef]:
    leaves, structure = jax.tree_util.tree_flatten(tree)
    if not leaves:
        raise ValueError(f"{name} must contain at least one owned field")
    for index, leaf in enumerate(leaves):
        if not hasattr(leaf, "shape"):
            raise TypeError(f"{name} leaf {index} must be an array, got {type(leaf).__name__}")
        if tuple(leaf.shape) != tuple(geometry.owned_shape):
            raise ValueError(
                f"{name} leaf {index} must have owned shape {geometry.owned_shape}, "
                f"got {tuple(leaf.shape)}"
            )
    return leaves, structure


def _field_scale_tuple(config: SolvaxNewtonConfig, nleaves: int) -> tuple[float, ...]:
    if not config.field_scales:
        return (1.0,) * nleaves
    if len(config.field_scales) == 1:
        return config.field_scales * nleaves
    if len(config.field_scales) != nleaves:
        raise ValueError(
            "SolvaxNewtonConfig.field_scales must be empty, contain one entry, "
            f"or match the residual leaf count ({nleaves}); got {len(config.field_scales)}"
        )
    return config.field_scales


def _tree_all_finite(
    tree: PyTree,
    domain: LocalDomain3D,
    active_mask: jnp.ndarray | None,
) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    local = jnp.asarray(True)
    for leaf in leaves:
        finite = jnp.isfinite(jnp.asarray(leaf, dtype=jnp.float64))
        if active_mask is not None:
            finite = finite | (~active_mask)
        local = local & jnp.all(finite)
    return _spmd_sum(local.astype(jnp.int32), domain) == _spmd_sum(
        jnp.asarray(1, dtype=jnp.int32), domain
    )


def solvax_newton_solve(
    residual_fn: Callable[[PyTree], PyTree],
    x0: PyTree,
    geometry: LocalFciGeometry3D,
    domain: LocalDomain3D,
    config: SolvaxNewtonConfig = SolvaxNewtonConfig(),
    *,
    active_cell_mask: jnp.ndarray | None = None,
    preconditioner: Callable[[PyTree], PyTree] | None = None,
) -> tuple[PyTree, SolvaxNewtonInfo]:
    """Solve a local owned-field nonlinear system with global SPMD norms.

    ``residual_fn`` and the optional right ``preconditioner`` operate on an
    arbitrary PyTree whose leaves are local owned arrays.  They must be safe
    under ``jax.jit(jax.shard_map(...))`` and return the same tree structure.
    """

    if not callable(residual_fn):
        raise TypeError("residual_fn must be callable")
    if preconditioner is not None and not callable(preconditioner):
        raise TypeError("preconditioner must be callable or None")
    if not isinstance(geometry, LocalFciGeometry3D):
        raise TypeError("geometry must be a LocalFciGeometry3D instance")
    if not isinstance(domain, LocalDomain3D):
        raise TypeError("domain must be a LocalDomain3D instance")
    if geometry.layout != domain.layout:
        raise ValueError("geometry and domain must share the same HaloLayout3D")
    if not isinstance(config, SolvaxNewtonConfig):
        raise TypeError("config must be a SolvaxNewtonConfig instance")

    x0_leaves, x_structure = _validate_tree(x0, geometry, name="x0")
    x0 = jax.tree_util.tree_unflatten(
        x_structure, tuple(jnp.asarray(value, dtype=jnp.float64) for value in x0_leaves)
    )
    residual0 = residual_fn(x0)
    residual_leaves, residual_structure = _validate_tree(
        residual0, geometry, name="residual_fn(x0)"
    )
    if residual_structure != x_structure:
        raise TypeError("residual_fn must return the same PyTree structure as x0")
    del residual_leaves
    scales = _field_scale_tuple(config, len(x0_leaves))
    scales_array = tuple(jnp.asarray(value, dtype=jnp.float64) for value in scales)
    mask = _active_mask(active_cell_mask, geometry)
    weights = _local_cell_volume_weights(geometry)
    if mask is not None:
        weights = jnp.where(mask, weights, 0.0)

    def global_inner_product(left: PyTree, right: PyTree) -> jax.Array:
        left_leaves = jax.tree_util.tree_leaves(left)
        right_leaves = jax.tree_util.tree_leaves(right)
        total = jnp.asarray(0.0, dtype=jnp.float64)
        for lhs, rhs, scale in zip(left_leaves, right_leaves, scales_array, strict=True):
            lhs = jnp.asarray(lhs, dtype=jnp.float64)
            rhs = jnp.asarray(rhs, dtype=jnp.float64)
            if mask is not None:
                lhs = jnp.where(mask, lhs, 0.0)
                rhs = jnp.where(mask, rhs, 0.0)
            total = total + jnp.sum(weights * lhs * rhs / (scale * scale))
        return _spmd_sum(total, domain)

    def global_norm(value: PyTree) -> jax.Array:
        return jnp.sqrt(jnp.maximum(global_inner_product(value, value), 0.0))

    initial_state_is_finite = _tree_all_finite(x0, domain, mask)
    initial_residual_is_finite = _tree_all_finite(residual0, domain, mask)
    initial_residual = global_norm(residual0)
    threshold = jnp.maximum(
        jnp.asarray(config.atol, dtype=jnp.float64),
        jnp.asarray(config.rtol, dtype=jnp.float64) * initial_residual,
    )
    acceptance_threshold = jnp.maximum(
        jnp.asarray(config.acceptance_atol, dtype=jnp.float64),
        jnp.asarray(config.acceptance_rtol, dtype=jnp.float64) * initial_residual,
    )

    solution = solvax_newton_krylov(
        residual_fn,
        x0,
        precond=preconditioner,
        inner_product=global_inner_product,
        norm=global_norm,
        rtol=config.rtol,
        atol=config.atol,
        max_steps=config.max_steps,
        linear_restart=config.linear_restart,
        linear_rtol=config.linear_rtol,
        linear_atol=config.linear_atol,
        linear_max_restarts=config.linear_max_restarts,
    )
    final_state = solution.x
    final_residual_tree = residual_fn(final_state)
    final_residual = global_norm(final_residual_tree)
    final_state_is_finite = _tree_all_finite(final_state, domain, mask)
    final_residual_is_finite = _tree_all_finite(final_residual_tree, domain, mask)
    finite = (
        initial_state_is_finite
        & initial_residual_is_finite
        & final_state_is_finite
        & final_residual_is_finite
        & jnp.isfinite(initial_residual)
        & jnp.isfinite(final_residual)
    )
    strict_converged = finite & solution.linear_converged & (final_residual <= threshold)
    accepted = finite & solution.linear_converged & (final_residual <= acceptance_threshold)
    info = SolvaxNewtonInfo(
        newton_iterations=jnp.asarray(solution.newton_iterations, dtype=jnp.int32),
        linear_iterations=jnp.asarray(solution.linear_iterations, dtype=jnp.int32),
        converged=strict_converged,
        linear_converged=jnp.asarray(solution.linear_converged),
        accepted=accepted,
        failed=~accepted,
        initial_residual_l2=initial_residual,
        final_residual_l2=final_residual,
        final_residual_rel_l2=final_residual / jnp.maximum(initial_residual, 1.0e-30),
        initial_state_is_finite=initial_state_is_finite,
        final_state_is_finite=final_state_is_finite,
        initial_residual_is_finite=initial_residual_is_finite,
        final_residual_is_finite=final_residual_is_finite,
    )
    return final_state, info


__all__ = ["SolvaxNewtonConfig", "SolvaxNewtonInfo", "solvax_newton_solve"]
