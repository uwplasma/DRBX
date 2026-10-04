"""Eta-sharded evaluation of the nodal SBP bracket.

The nodal state is plane-major ``(n_eta, P[, F])``, so sharding is a split of the leading axis over the ``"z"`` mesh
axis (``AXIS`` of ``fci_perpendicular_sharding``). The metric arrays of the plan (``jac, h, B, K, Hp``) are split the
same way; operator matrices, ``wxy`` and the static structure are replicated.

Two halo exchanges per evaluation:

1. ``phi`` with halo 2 -> the velocity flux ``F`` on the owned planes (``D_eta phi~`` reads +-2);
2. ``(g, F)`` with halo 3 -> every operator on the extended arrays -> the owned planes. Halo 3 because the eta
   dissipation reads ``g`` at +-3 and ``F`` at +-2, and ``D_eta (F3 g)`` reads +-2 (one exchange would need halo 5).

Needs ``p = n_eta / n_shards >= 3`` for more than one shard (``exchange_plane_halo`` raises otherwise); one shard is
the local periodic wrap. The result is expected to equal the single-device :func:`sbp_bracket` bitwise.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData
from drbx.native.fci_perpendicular_sbp_bracket import sbp_bracket_ext, velocity_flux_ext
from drbx.native.fci_perpendicular_sharding import AXIS, make_plane_mesh
from drbx.native.owner_plane_layout import exchange_plane_halo
from drbx.stencils.nodal_plan import NodalPlan

__all__ = ["NODAL_HALO", "PHI_HALO", "ShardedNodalPlan", "nodal_plan_specs", "shard_nodal_plan", "sharded_sbp_bracket",
           "make_plane_mesh", "AXIS"]

NODAL_HALO = 3
PHI_HALO = 2
_PLANE_FIELDS = ("jac", "h", "B", "K", "Hp")


class ShardedNodalPlan(NamedTuple):
    """A plan checked for ``n_shards`` (``plan`` itself is unchanged, optionally placed on a mesh)."""

    plan: NodalPlan
    n_shards: int


jax.tree_util.register_pytree_node(
    ShardedNodalPlan, lambda s: ((s.plan,), s.n_shards), lambda n, c: ShardedNodalPlan(c[0], n))


def nodal_plan_specs(plan: NodalPlan) -> NodalPlan:
    """``PartitionSpec`` pytree of ``plan``: metric arrays split on the leading (eta) axis, the rest replicated."""
    rep = jax.tree_util.tree_map(lambda _: P(), plan)
    return dataclasses.replace(rep, **{name: P(AXIS) for name in _PLANE_FIELDS})


def shard_nodal_plan(plan: NodalPlan, n_shards: int, mesh: Mesh | None = None) -> ShardedNodalPlan:
    """Validate the split of ``plan`` over ``n_shards`` and, given ``mesh``, place the leaves on it."""
    n_shards = int(n_shards)
    n_eta = plan.structure.n_eta
    if n_shards < 1 or n_eta % n_shards:
        raise ValueError(f"n_eta={n_eta} is not divisible by n_shards={n_shards}")
    p = n_eta // n_shards
    if n_shards > 1 and p < NODAL_HALO:
        raise ValueError(f"p = n_eta / n_shards = {p} planes per shard is below the halo {NODAL_HALO}")
    if plan.jac.shape[0] != n_eta:
        raise ValueError("shard_nodal_plan needs the global plan (all eta planes)")
    if mesh is not None:
        if int(mesh.shape[AXIS]) != n_shards:
            raise ValueError(f"the mesh has {mesh.shape[AXIS]} shards along {AXIS!r}, expected {n_shards}")
        plan = jax.tree_util.tree_map(lambda x, s: jax.device_put(x, NamedSharding(mesh, s)), plan, nodal_plan_specs(plan))
    return ShardedNodalPlan(plan, n_shards)


def sharded_sbp_bracket(sharded: ShardedNodalPlan, phi, g, bcd: SatBoundaryData | None, rho_star, mesh: Mesh):
    """``sbp_bracket`` under ``shard_map`` over the eta axis; ``phi (n_eta, P)``, ``g (n_eta, P[, F])`` global arrays."""
    plan, n_shards = sharded
    if int(mesh.shape[AXIS]) != n_shards:
        raise ValueError(f"the mesh has {mesh.shape[AXIS]} shards along {AXIS!r}, the plan {n_shards}")
    g = jnp.asarray(g)
    scalar = g.ndim == 2
    plane = jax.tree_util.tree_map(lambda _: P(AXIS), bcd)

    def body(plan_l, phi_l, g_l, bc_l, rho):
        phi_ext = exchange_plane_halo(phi_l, PHI_HALO, AXIS, n_shards)
        F = velocity_flux_ext(plan_l, phi_ext, rho)
        g3 = g_l[..., None] if scalar else g_l
        both = exchange_plane_halo(jnp.concatenate([g3, F], axis=-1), NODAL_HALO, AXIS, n_shards)
        g_ext = both[..., :g3.shape[-1]]
        F_ext = both[..., g3.shape[-1]:]
        out = sbp_bracket_ext(plan_l, F_ext, g_ext[..., 0] if scalar else g_ext, bc_l)
        return out

    fn = jax.shard_map(body, mesh=mesh, in_specs=(nodal_plan_specs(plan), P(AXIS), P(AXIS), plane, P()),
                       out_specs=P(AXIS), check_vma=False)
    return fn(plan, jnp.asarray(phi), g, bcd, jnp.asarray(rho_star))
