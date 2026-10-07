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

Curvature (:func:`sharded_sbp_curvature`): one exchange of ``[q, F, B]`` with the halo ``HALO = 3`` of ``sbp_curvature_ext``.

Laplacian and CG (:func:`sharded_laplacian_form`, :func:`sharded_laplacian_action`, :func:`sharded_solve_dirichlet`): the
``LaplacianPlan`` plane leaves are stored per shard with their halo-2 neighbours already attached (a one-time gather,
:func:`shard_laplacian_plan`), so a call only exchanges ``f`` (halo 4, ``F_HALO``) and, when given, the wall data (halo 2).
The unit coefficient only (``coeff=None``; a pointwise coefficient interpolates its face values by a global FFT along eta).
Needs ``p >= F_HALO = 4`` for more than one shard. CG runs in one ``shard_map``: the matvec is the halo-extended form
(one exchange of the search direction per iteration), the dot products and norms are ``psum`` reductions (so the iterates
agree with the single-device solve to round-off, not bitwise) and the plane-block preconditioner applies plane-locally to
its shard of the factors (:func:`shard_core_schur_preconditioner`).

Composed nodal perpendicular RHS (:func:`sharded_nodal_perpendicular_rhs`): the C1 composition of
``fci_nodal_perpendicular_rhs`` (same options, parameters, wall data, single-length ``rho_star`` scaling and ``phi_mode`` as
:func:`~drbx.native.fci_nodal_perpendicular_rhs.nodal_perpendicular_rhs`) on global arrays, every stage calling the sharded
operator above, so a stage is its own ``shard_map`` with its own halo exchanges (nothing is fused). Planes per shard
``>= 4`` for more than one shard. Per evaluation, in terms of exchanges of the owned planes with the two neighbours:

* ``phi_mode="solve"``: the CG of :func:`sharded_solve_dirichlet` (one halo-4 exchange of the search direction per
  iteration, ``psum`` dot products, one halo-2 exchange of the ``psi`` wall data at the start), then ``phi = psi - tau p``
  on the owned planes (no exchange);
* bracket: halo 2 of ``phi``, halo 3 of ``[g, F]`` (two exchanges, as in :func:`sharded_sbp_bracket`);
* curvature: halo 3 of ``[q, F]`` (one exchange);
* diffusion: halo 4 of the state and halo 2 of each wall array (``value``, and ``normal`` for Neumann data): one to three exchanges.

``rho_star`` and ``D_f`` scalings, the scatter of the curvature columns, the sum and the ``source`` act on the owned planes.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from drbx.geometry.sbp_laplacian import LaplacianPlan
from drbx.native.fci_nodal_perpendicular_rhs import (
    NodalPerpendicularContext, NodalPerpendicularOptions, NodalPerpendicularParams, NodalPerpendicularTerms, NodalWallData,
    _plan_wall_trace, assemble_terms, check_input_shapes, check_potential_arguments, curvature_operands,
    diffusion_boundary_data, polarization_rhs, prescribed_potential, pressure_variable, scatter_curvature)
from drbx.native.fci_perpendicular_reconstruction_state import DIRICHLET
from drbx.native.fci_perpendicular_plane_preconditioner import CoreSchurPreconditioner, apply_core_schur_preconditioner
from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData
from drbx.native.fci_perpendicular_sbp_bracket import sbp_bracket_ext, velocity_flux_ext
from drbx.native.fci_perpendicular_sbp_curvature import curvature_flux, sbp_curvature_ext
from drbx.native.fci_perpendicular_sbp_dissipation import HALO as CURVATURE_HALO
from drbx.native.fci_perpendicular_sbp_laplacian import (
    F_HALO, PLAN_HALO, PLANE_FIELDS, LaplacianBoundaryData, laplacian_action_ext, laplacian_form_ext, window_laplacian_plan)
from drbx.native.fci_perpendicular_sbp_laplacian_solve import _pcg
from drbx.native.fci_perpendicular_sharding import AXIS, make_plane_mesh
from drbx.native.owner_plane_layout import exchange_plane_halo
from drbx.stencils.nodal_plan import NodalPlan

__all__ = ["NODAL_HALO", "PHI_HALO", "CURVATURE_HALO", "F_HALO", "PLAN_HALO", "ShardedNodalPlan", "nodal_plan_specs",
           "shard_nodal_plan", "sharded_sbp_bracket", "sharded_sbp_curvature", "ShardedLaplacianPlan",
           "laplacian_plan_specs", "shard_laplacian_plan", "sharded_laplacian_form", "sharded_laplacian_action",
           "core_schur_specs", "shard_core_schur_preconditioner", "sharded_solve_dirichlet", "make_plane_mesh", "AXIS",
           "RHS_HALO", "ShardedNodalPerpendicularContext", "shard_nodal_perpendicular_context",
           "sharded_nodal_perpendicular_rhs", "sharded_nodal_perpendicular_rhs_jit", "sharded_bracket_rule_inflow"]

NODAL_HALO = 3
PHI_HALO = 2
_PLANE_FIELDS = ("jac", "h", "B", "K", "Hp", "core_Ginv")      # core_Ginv only when the plan has a core


class ShardedNodalPlan(NamedTuple):
    """A plan checked for ``n_shards`` (``plan`` itself is unchanged, optionally placed on a mesh)."""

    plan: NodalPlan
    n_shards: int


jax.tree_util.register_pytree_node(
    ShardedNodalPlan, lambda s: ((s.plan,), s.n_shards), lambda n, c: ShardedNodalPlan(c[0], n))


def nodal_plan_specs(plan: NodalPlan) -> NodalPlan:
    """``PartitionSpec`` pytree of ``plan``: metric arrays split on the leading (eta) axis, the rest replicated."""
    rep = jax.tree_util.tree_map(lambda _: P(), plan)
    return dataclasses.replace(rep, **{name: P(AXIS) for name in _PLANE_FIELDS if getattr(plan, name) is not None})


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


def sharded_sbp_bracket(sharded: ShardedNodalPlan, phi, g, bcd: SatBoundaryData | None, rho_star, mesh: Mesh,
                        c_kappa=1.0):
    """``sbp_bracket`` under ``shard_map`` over the eta axis; ``phi (n_eta, P)``, ``g (n_eta, P[, F])`` global arrays.

    ``c_kappa`` (core shell damping) is a replicated traced scalar.
    """
    plan, n_shards = sharded
    if int(mesh.shape[AXIS]) != n_shards:
        raise ValueError(f"the mesh has {mesh.shape[AXIS]} shards along {AXIS!r}, the plan {n_shards}")
    g = jnp.asarray(g)
    scalar = g.ndim == 2
    plane = jax.tree_util.tree_map(lambda _: P(AXIS), bcd)

    def body(plan_l, phi_l, g_l, bc_l, rho, ck):
        phi_ext = exchange_plane_halo(phi_l, PHI_HALO, AXIS, n_shards)
        F = velocity_flux_ext(plan_l, phi_ext, rho)
        g3 = g_l[..., None] if scalar else g_l
        both = exchange_plane_halo(jnp.concatenate([g3, F], axis=-1), NODAL_HALO, AXIS, n_shards)
        g_ext = both[..., :g3.shape[-1]]
        F_ext = both[..., g3.shape[-1]:]
        out = sbp_bracket_ext(plan_l, F_ext, g_ext[..., 0] if scalar else g_ext, bc_l, ck)
        return out

    fn = jax.shard_map(body, mesh=mesh, in_specs=(nodal_plan_specs(plan), P(AXIS), P(AXIS), plane, P(), P()),
                       out_specs=P(AXIS), check_vma=False)
    return fn(plan, jnp.asarray(phi), g, bcd, jnp.asarray(rho_star), jnp.asarray(c_kappa))


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------
def sharded_sbp_curvature(sharded: ShardedNodalPlan, q, bcd: SatBoundaryData | None, mesh: Mesh, *, tau=1.0,
                          psi="phi_plus_tau_ti", absolute_method="closed_form", jump_dissipation=False, c_kappa=0.0, F=None):
    """``sbp_curvature`` under ``shard_map`` over the eta axis; ``q (n_eta, P, 5)`` and the optional ``F (n_eta, P, 3)`` are global arrays.

    One halo-``CURVATURE_HALO`` exchange of ``[q, F]`` (plus ``B`` with ``jump_dissipation``, the extended ``B`` of the eta
    faces); ``F=None`` takes ``curvature_flux`` of the local plan. A Python ``c_kappa == 0`` skips the core damping as in
    ``sbp_curvature``, anything else is a replicated traced scalar. ``tau`` is a replicated scalar.
    """
    plan, n_shards = sharded
    if int(mesh.shape[AXIS]) != n_shards:
        raise ValueError(f"the mesh has {mesh.shape[AXIS]} shards along {AXIS!r}, the plan {n_shards}")
    use_ck = not (isinstance(c_kappa, (int, float)) and c_kappa == 0)
    nq = jnp.shape(q)[-1]
    plane = jax.tree_util.tree_map(lambda _: P(AXIS), bcd)
    f_spec = None if F is None else P(AXIS)

    def body(plan_l, q_l, bc_l, F_l, tau_, ck):
        F_l = curvature_flux(plan_l) if F_l is None else F_l
        parts = [q_l, F_l] + ([plan_l.B[..., None]] if jump_dissipation else [])
        ext = exchange_plane_halo(jnp.concatenate(parts, axis=-1), CURVATURE_HALO, AXIS, n_shards)
        q_ext, F_ext = ext[..., :nq], ext[..., nq:nq + 3]
        B_ext = ext[..., nq + 3] if jump_dissipation else None
        return sbp_curvature_ext(plan_l, F_ext, q_ext, bc_l, tau=tau_, psi=psi, absolute_method=absolute_method,
                                 jump_dissipation=jump_dissipation, c_kappa=ck if use_ck else 0.0, B_ext=B_ext)

    fn = jax.shard_map(body, mesh=mesh, in_specs=(nodal_plan_specs(plan), P(AXIS), plane, f_spec, P(), P()),
                       out_specs=P(AXIS), check_vma=False)
    return fn(plan, jnp.asarray(q), bcd, None if F is None else jnp.asarray(F), jnp.asarray(tau),
              jnp.asarray(c_kappa if use_ck else 0.0))


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------
class ShardedLaplacianPlan(NamedTuple):
    """A :class:`LaplacianPlan` whose plane leaves are stored per shard with ``PLAN_HALO`` neighbours on each side.

    Every plane leaf has ``n_shards * (p + 2 PLAN_HALO)`` planes (shard ``s`` holds the window of planes ``s p - PLAN_HALO
    .. (s + 1) p + PLAN_HALO``, periodic) and is split on the eta axis; the other leaves are replicated. ``structure`` still
    describes the global plan.
    """

    plan: LaplacianPlan
    n_shards: int


jax.tree_util.register_pytree_node(
    ShardedLaplacianPlan, lambda s: ((s.plan,), s.n_shards), lambda n, c: ShardedLaplacianPlan(c[0], n))


def laplacian_plan_specs(lp: LaplacianPlan) -> LaplacianPlan:
    """``PartitionSpec`` pytree of a (windowed) plan: the plane leaves split on the leading axis, the rest replicated."""
    rep = jax.tree_util.tree_map(lambda _: P(), lp)
    return dataclasses.replace(rep, **{name: P(AXIS) for name in PLANE_FIELDS if getattr(lp, name) is not None})


def _check_shards(n_eta: int, n_shards: int, halo: int, what: str) -> int:
    n_shards = int(n_shards)
    if n_shards < 1 or n_eta % n_shards:
        raise ValueError(f"n_eta={n_eta} is not divisible by n_shards={n_shards}")
    p = n_eta // n_shards
    if n_shards > 1 and p < halo:
        raise ValueError(f"p = n_eta / n_shards = {p} planes per shard is below the {what} halo {halo}")
    return p


def shard_laplacian_plan(lp: LaplacianPlan, n_shards: int, mesh: Mesh | None = None) -> ShardedLaplacianPlan:
    """Window the plane leaves of the global ``lp`` per shard (one gather) and, given ``mesh``, place the leaves on it."""
    n_eta = lp.structure.n_eta
    p = _check_shards(n_eta, n_shards, F_HALO, "field")
    if lp.Hp.shape[0] != n_eta:
        raise ValueError("shard_laplacian_plan needs the global plan (all eta planes)")
    n_shards = int(n_shards)
    index = np.concatenate([(s * p - PLAN_HALO + np.arange(p + 2 * PLAN_HALO)) % n_eta for s in range(n_shards)])
    ext = window_laplacian_plan(lp, index)
    if mesh is not None:
        if int(mesh.shape[AXIS]) != n_shards:
            raise ValueError(f"the mesh has {mesh.shape[AXIS]} shards along {AXIS!r}, expected {n_shards}")
        ext = jax.tree_util.tree_map(lambda x, sp: jax.device_put(x, NamedSharding(mesh, sp)), ext, laplacian_plan_specs(ext))
    return ShardedLaplacianPlan(ext, n_shards)


def _check_mesh(mesh: Mesh, n_shards: int) -> None:
    if int(mesh.shape[AXIS]) != n_shards:
        raise ValueError(f"the mesh has {mesh.shape[AXIS]} shards along {AXIS!r}, the plan {n_shards}")


def _no_coeff(coeff) -> None:
    if coeff is not None:
        raise ValueError("eta-sharded Laplacian needs coeff=None: the face values of a pointwise coefficient are interpolated "
                         "by a global FFT along eta")


def _wall_halo(bcd, n_shards: int):
    return jax.tree_util.tree_map(lambda a: exchange_plane_halo(jnp.asarray(a), PLAN_HALO, AXIS, n_shards), bcd)


def sharded_laplacian_form(sharded: ShardedLaplacianPlan, f, bcd, mesh: Mesh, kinds="dirichlet", coeff=None, c_kappa=1.0,
                           *, neumann_mode=None):
    """``laplacian_form`` under ``shard_map`` over the eta axis (``f (n_eta, P[, F])`` global; ``coeff`` must be ``None``)."""
    _no_coeff(coeff)
    lp, n_shards = sharded
    _check_mesh(mesh, n_shards)
    plane = jax.tree_util.tree_map(lambda _: P(AXIS), bcd)

    def body(lp_l, f_l, bc_l, ck):
        f_ext = exchange_plane_halo(f_l, F_HALO, AXIS, n_shards)
        return laplacian_form_ext(lp_l, f_ext, _wall_halo(bc_l, n_shards), kinds, ck, neumann_mode=neumann_mode)

    fn = jax.shard_map(body, mesh=mesh, in_specs=(laplacian_plan_specs(lp), P(AXIS), plane, P()), out_specs=P(AXIS),
                       check_vma=False)
    return fn(lp, jnp.asarray(f), bcd, jnp.asarray(c_kappa))


def sharded_laplacian_action(sharded: ShardedLaplacianPlan, f, bcd, mesh: Mesh, kinds="dirichlet", coeff=None, c_kappa=1.0,
                             *, neumann_mode=None):
    """``laplacian_action`` (``L f = -H^-1 (M f - b)``) under ``shard_map`` over the eta axis; ``coeff`` must be ``None``."""
    _no_coeff(coeff)
    lp, n_shards = sharded
    _check_mesh(mesh, n_shards)
    plane = jax.tree_util.tree_map(lambda _: P(AXIS), bcd)

    def body(lp_l, f_l, bc_l, ck):
        f_ext = exchange_plane_halo(f_l, F_HALO, AXIS, n_shards)
        return laplacian_action_ext(lp_l, f_ext, _wall_halo(bc_l, n_shards), kinds, ck, neumann_mode=neumann_mode)

    fn = jax.shard_map(body, mesh=mesh, in_specs=(laplacian_plan_specs(lp), P(AXIS), plane, P()), out_specs=P(AXIS),
                       check_vma=False)
    return fn(lp, jnp.asarray(f), bcd, jnp.asarray(c_kappa))


# ---------------------------------------------------------------------------
# Preconditioner shard and CG
# ---------------------------------------------------------------------------
def core_schur_specs(prec: CoreSchurPreconditioner) -> CoreSchurPreconditioner:
    """``PartitionSpec`` pytree of the core-Schur factors: ``lo (S, E, ...)`` and ``dinv (S, E, ...)`` split on axis 1, ``z`` and ``sinv`` on axis 0."""
    return CoreSchurPreconditioner(P(None, AXIS), P(None, AXIS), P(AXIS), P(AXIS), prec.meta)


def shard_core_schur_preconditioner(prec: CoreSchurPreconditioner, n_shards: int, mesh: Mesh | None = None
                                    ) -> CoreSchurPreconditioner:
    """Validate the plane split of ``prec`` and, given ``mesh``, place its leaves on it (the apply is plane-local, no halo)."""
    if not isinstance(prec, CoreSchurPreconditioner):
        raise TypeError("the eta-sharded solve needs a CoreSchurPreconditioner (method='core_schur')")
    _check_shards(prec.z.shape[0], n_shards, 1, "preconditioner")
    if mesh is None:
        return prec
    if int(mesh.shape[AXIS]) != int(n_shards):
        raise ValueError(f"the mesh has {mesh.shape[AXIS]} shards along {AXIS!r}, expected {n_shards}")
    return jax.tree_util.tree_map(lambda x, sp: jax.device_put(x, NamedSharding(mesh, sp)), prec, core_schur_specs(prec))


def sharded_solve_dirichlet(sharded: ShardedLaplacianPlan, s, bcd, prec: CoreSchurPreconditioner, mesh: Mesh, *, coeff=None,
                            c_kappa=1.0, x0=None, rtol=1e-10, maxit=200, residual_norm: str = "h_inv"):
    """Eta-sharded :func:`~drbx.native.fci_perpendicular_sbp_laplacian_solve.solve_dirichlet` (``L f = s``, Dirichlet data ``bcd``).

    The same flexible Polak-Ribiere CG, written in one ``shard_map``: the matvec is :func:`laplacian_form_ext` of the
    halo-4 extended search direction, the dot products and the residual norm are ``psum`` reductions, and the preconditioner
    (a sharded :class:`CoreSchurPreconditioner`, plane-local) applies to the local shard of the residual. ``s``, ``x0`` and the
    result are global ``(n_eta, P)`` arrays; ``coeff`` must be ``None``. Returns ``(f, info)`` like ``solve_dirichlet``.
    """
    _no_coeff(coeff)
    if residual_norm not in ("h_inv", "euclid"):
        raise ValueError(f"residual_norm must be 'h_inv' or 'euclid', got {residual_norm!r}")
    lp, n_shards = sharded
    _check_mesh(mesh, n_shards)
    if not isinstance(prec, CoreSchurPreconditioner):
        raise TypeError("the eta-sharded solve needs a CoreSchurPreconditioner (method='core_schur')")
    if prec.z.shape[0] != lp.structure.n_eta:
        raise ValueError(f"the preconditioner has {prec.z.shape[0]} planes, the plan {lp.structure.n_eta}")
    s = jnp.asarray(s)
    x0 = jnp.zeros_like(s) if x0 is None else jnp.asarray(x0)
    plane = jax.tree_util.tree_map(lambda _: P(AXIS), bcd)

    def body(lp_l, s_l, bc_l, prec_l, x0_l, ck, rtol_, maxit_):
        p_own = s_l.shape[0]
        H = lp_l.Hp[PLAN_HALO:PLAN_HALO + p_own] * lp_l.structure.deta

        def psum(x):
            return lax.psum(x, AXIS)

        def dot(a, b):
            return psum(jnp.sum(a * b))

        if residual_norm == "h_inv":
            def norm(r):
                return jnp.sqrt(psum(jnp.sum(r * r / H)))
        else:
            def norm(r):
                return jnp.sqrt(psum(jnp.sum(r * r)))

        def matvec(v):
            return laplacian_form_ext(lp_l, exchange_plane_halo(v, F_HALO, AXIS, n_shards), None, "dirichlet", ck)

        def precond(r):
            return apply_core_schur_preconditioner(prec_l, r.reshape(-1)).reshape(r.shape)

        zero_ext = jnp.zeros((p_own + 2 * F_HALO,) + s_l.shape[1:], s_l.dtype)
        rhs = -laplacian_form_ext(lp_l, zero_ext, _wall_halo(bc_l, n_shards), "dirichlet", ck) - H * s_l
        x, info = _pcg(matvec, rhs, precond, norm, x0_l, rtol_, maxit_, dot=dot)
        it, rel = info["iterations"], info["relative_residual"]
        return x, it, rel

    fn = jax.shard_map(body, mesh=mesh,
                       in_specs=(laplacian_plan_specs(lp), P(AXIS), plane, core_schur_specs(prec), P(AXIS), P(), P(), P()),
                       out_specs=(P(AXIS), P(), P()), check_vma=False)
    x, it, rel = fn(lp, s, bcd, prec, x0, jnp.asarray(c_kappa), jnp.asarray(rtol), jnp.asarray(maxit))
    return x, {"iterations": it, "relative_residual": rel, "converged": rel <= rtol}


# ---------------------------------------------------------------------------
# Composed nodal perpendicular RHS
# ---------------------------------------------------------------------------
RHS_HALO = max(NODAL_HALO, CURVATURE_HALO, F_HALO)        # the largest halo of any stage: planes per shard must reach it


class ShardedNodalPerpendicularContext(NamedTuple):
    """Sharded :class:`~drbx.native.fci_nodal_perpendicular_rhs.NodalPerpendicularContext`: the nodal plan, the windowed Laplacian
    plan, the (plane-sharded) core-Schur preconditioner or ``None``, and the curvature flux ``F (n_eta, P, 3)`` split on eta."""

    plan: ShardedNodalPlan
    lplan: ShardedLaplacianPlan
    prec: CoreSchurPreconditioner | None
    curvature_flux: jax.Array
    n_shards: int


jax.tree_util.register_pytree_node(
    ShardedNodalPerpendicularContext, lambda s: ((s.plan, s.lplan, s.prec, s.curvature_flux), s.n_shards),
    lambda n, c: ShardedNodalPerpendicularContext(*c, n))


def shard_nodal_perpendicular_context(ctx: NodalPerpendicularContext, n_shards: int, mesh: Mesh | None = None
                                      ) -> ShardedNodalPerpendicularContext:
    """Shard a global single-device context over ``n_shards`` (and place it on ``mesh`` if given).

    Validates ``n_eta % n_shards == 0`` and, for more than one shard, ``p = n_eta / n_shards >= RHS_HALO = 4``; shards the nodal
    plan, the Laplacian plan, the preconditioner (``None`` stays ``None``: prescribed-``phi`` only; a ``PlanePreconditioner``
    raises ``TypeError``, the sharded solve needs ``method="core_schur"``) and ``curvature_flux``.
    """
    n_eta = ctx.plan.structure.n_eta
    _check_shards(n_eta, n_shards, RHS_HALO, "perpendicular RHS")
    n_shards = int(n_shards)
    if jnp.shape(ctx.curvature_flux) != (n_eta, ctx.plan.structure.P, 3):
        raise ValueError(f"curvature_flux must have shape {(n_eta, ctx.plan.structure.P, 3)}, got {jnp.shape(ctx.curvature_flux)}")
    plan = shard_nodal_plan(ctx.plan, n_shards, mesh)
    lplan = shard_laplacian_plan(ctx.lplan, n_shards, mesh)
    prec = None if ctx.prec is None else shard_core_schur_preconditioner(ctx.prec, n_shards, mesh)
    flux = jnp.asarray(ctx.curvature_flux)
    if mesh is not None:
        flux = jax.device_put(flux, NamedSharding(mesh, P(AXIS)))
    return ShardedNodalPerpendicularContext(plan, lplan, prec, flux, n_shards)


def _check_context(sctx: ShardedNodalPerpendicularContext, mesh: Mesh) -> None:
    _check_mesh(mesh, sctx.n_shards)
    if sctx.plan.n_shards != sctx.n_shards or sctx.lplan.n_shards != sctx.n_shards:
        raise ValueError(f"inconsistent shard counts: context {sctx.n_shards}, nodal plan {sctx.plan.n_shards}, "
                         f"Laplacian plan {sctx.lplan.n_shards}")


def sharded_bracket_rule_inflow(sctx: ShardedNodalPerpendicularContext, opts: NodalPerpendicularOptions, state, value, mesh: Mesh):
    """Eta-sharded :func:`~drbx.native.fci_nodal_perpendicular_rhs.bracket_rule_inflow`: the wall trace of the nodal plan is plane-local,
    so the body runs on the owned planes with no exchange. ``state (E, P, F)``, ``value (E, N, F)`` global; returns ``(E, N, F)``."""
    _check_context(sctx, mesh)
    kinds = opts.diffusion_kinds
    if len(kinds) != opts.n_fields:
        raise ValueError("bracket_rule_inflow needs opts.diffusion_kinds (one kind per field)")
    state, value = jnp.asarray(state), jnp.asarray(value)
    dirichlet = np.array([k == DIRICHLET for k in kinds])
    if dirichlet.all():
        return value
    plan = sctx.plan.plan

    def body(plan_l, state_l, value_l):
        return jnp.where(dirichlet, value_l, _plan_wall_trace(plan_l, state_l))

    fn = jax.shard_map(body, mesh=mesh, in_specs=(nodal_plan_specs(plan), P(AXIS), P(AXIS)), out_specs=P(AXIS), check_vma=False)
    return fn(plan, state, value)


def sharded_nodal_perpendicular_rhs(sctx: ShardedNodalPerpendicularContext, opts: NodalPerpendicularOptions,
                                    params: NodalPerpendicularParams, state, wall: NodalWallData, mesh: Mesh, *, phi=None,
                                    psi_x0=None, sigma=None, source=None) -> NodalPerpendicularTerms:
    """Eta-sharded :func:`~drbx.native.fci_nodal_perpendicular_rhs.nodal_perpendicular_rhs` (see the module docstring).

    ``state (E, P, F)``, ``wall.value / normal (E, N, F)``, ``wall.psi (E, N)``, ``phi``, ``psi_x0``, ``sigma (E, P)``, ``source``
    are global arrays (the wall data are split along E inside the stages); the result holds global arrays and ``solve_info`` is
    replicated. Same ``phi_mode`` and ``rho_star`` semantics and the same errors as the single-device function. The CG
    iterates equal the single-device ones to round-off (``psum`` dot products), so the iteration count may differ by one only
    when the solve sits at the ``phi_rtol`` threshold.
    """
    _check_context(sctx, mesh)
    state = jnp.asarray(state)
    st, lst = sctx.plan.plan.structure, sctx.lplan.plan.structure
    check_input_shapes(st.n_eta, st.P, lst.N, opts, params, state, wall)
    if wall.psi is not None and jnp.shape(wall.psi) != (st.n_eta, lst.N):
        raise ValueError(f"wall.psi must have shape {(st.n_eta, lst.N)}, got {jnp.shape(wall.psi)}")
    n, Ti = check_potential_arguments(opts, state, wall, phi)
    if opts.phi_mode == "solve":
        if sctx.prec is None:
            raise ValueError("the context has no preconditioner (build it with build_preconditioner=True)")
        rhs = polarization_rhs(opts, params, state[..., opts.fields.index("vorticity")], sigma)
        psi, info = sharded_solve_dirichlet(sctx.lplan, rhs, LaplacianBoundaryData(value=(jnp.asarray(wall.psi),)), sctx.prec,
                                            mesh, c_kappa=opts.laplacian_c_kappa, x0=psi_x0, rtol=opts.phi_rtol,
                                            maxit=opts.phi_maxit)
        phi = psi - params.tau * pressure_variable(opts, n, Ti)
    else:
        phi, psi, info = prescribed_potential(opts, params, n, Ti, phi)
    zeros = jnp.zeros_like(state)
    terms = opts.terms
    if "bracket" in terms:
        bcd = SatBoundaryData((wall.value,))
        bracket = params.rho_star * sharded_sbp_bracket(sctx.plan, phi, state, bcd, 1.0, mesh, opts.bracket_c_kappa)
    else:
        bracket = zeros
    if "curvature" in terms:
        q, bcd = curvature_operands(opts, state, wall, phi)
        curv = sharded_sbp_curvature(sctx.plan, q, bcd, mesh, tau=params.tau, psi=opts.psi, absolute_method=opts.absolute_method,
                                     jump_dissipation=opts.curvature_jump_dissipation, c_kappa=opts.curvature_c_kappa,
                                     F=sctx.curvature_flux)
        curvature = scatter_curvature(opts, params, state, curv)
    else:
        curvature = zeros
    if "diffusion" in terms:
        lap = sharded_laplacian_action(sctx.lplan, state, diffusion_boundary_data(opts, wall), mesh, opts.diffusion_kinds, None,
                                       opts.laplacian_c_kappa, neumann_mode=opts.neumann_mode)
        diffusion = jnp.asarray(params.D) * lap
    else:
        diffusion = zeros
    return assemble_terms(bracket, curvature, diffusion, phi, psi, info, source)


sharded_nodal_perpendicular_rhs_jit = jax.jit(sharded_nodal_perpendicular_rhs, static_argnames=("opts", "mesh"))
