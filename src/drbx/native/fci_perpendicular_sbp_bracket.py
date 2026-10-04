"""Split-form SBP Poisson bracket on the nodal layout: velocity flux, transport, SATs and the full operator.

With the flux ``F = h x grad(phi~) / rho*`` (``phi~`` the level-trace-matched potential), the bracket acts on a field
``g`` as ``L g = T g + c g / 2 + wall_in(g) + iface_upwind(g)``, where

- ``T g = -1/2 (A g + K g) + iface_centred(g) + sum_walls wall_pc(g)`` is energy neutral up to the wall terms
  (``A g = F/|J| . grad g``, ``K g = div(F g) / |J|``, centred interface SAT, product-correction wall SAT),
- ``c = -2 T(1)`` (excludes the inflow and upwind terms), so ``L(1) = 0`` for matching wall data,
- ``wall_in`` penalises the inflow part of every wall towards the Dirichlet data and ``iface_upwind`` is the jump
  penalty on the finer side of every level face.

A wall with outward sign ``sigma`` (+1 at an outer side, -1 at an inner wall) contributes the energy
``-1/2 Omega sum sigma v a^2`` through the SBP boundary term and its product correction; the inflow part is
``tau = max(-sigma v, 0)``.

Array conventions (see ``fci_perpendicular_sbp_ops``): ``F (E, P, 3)``, ``phi (E, P)``, ``g (E, P[, ...])``. Functions
ending in ``_ext`` take arrays on halo-extended planes (``F`` and ``g`` with halo >= 3, ``phi`` with halo >= 2) and
return the owned planes; the plain names take owned arrays and wrap periodically (single device). Faces that touch a
non-ring block raise ``NotImplementedError`` in :func:`level_match` until the core supplies its velocity-gradient rule.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData
from drbx.native.fci_perpendicular_sbp_dissipation import HALO, core_damping_ext, dissipation, dissipation_ext
from drbx.native.fci_perpendicular_sbp_ops import (
    bc,
    crop,
    d1,
    d2,
    d_eta,
    extend_periodic,
    flux_trace,
    halo_of,
    lift,
    scatter,
    trace,
)
from drbx.stencils.nodal_plan import NodalPlan

TWO_PI = 2.0 * np.pi

__all__ = [
    "HALO", "level_match", "flux_from_potential", "velocity_flux", "velocity_flux_ext", "advect", "divergence",
    "interface_centred", "wall_correction", "wall_inflow", "interface_upwind", "transport", "transport_ext",
    "compressibility", "compressibility_ext", "linear_operator", "linear_operator_ext", "sbp_bracket_ext", "sbp_bracket",
    "dissipation", "dissipation_ext", "core_damping_ext",
]


def _apply(I, x):
    """Apply a transfer matrix along the face axis of ``x (E, N, ...)``."""
    return jnp.einsum("ij,ej...->ei...", I, x)


# ---------------------------------------------------------------------------
# Velocity flux
# ---------------------------------------------------------------------------
def level_match(plan: NodalPlan, phi):
    """Level D5c: ``phi~ = phi + sum_sides T^T (d_s / |t|^2)`` with the common face modes set to their average."""
    out = phi
    for face, (a_blk, b_blk, _na, _nb, _x) in zip(plan.faces, plan.structure.faces):
        if plan.structure.blocks[a_blk][0] == "core":
            continue                                    # core-ring face: handled by the core D5c in flux_from_potential
        if face.CAA is None:
            raise NotImplementedError("faces touching a non-ring block need the block's velocity_gradient rule (M1)")
        a = trace(plan, a_blk, "outer", phi)
        b = trace(plan, b_blk, "inner", phi)
        dA = _apply(face.CAA, a) + _apply(face.CAB, b)
        dB = _apply(face.CBA, a) + _apply(face.CBB, b)
        for blk, side, d in ((a_blk, "outer", dA), (b_blk, "inner", dB)):
            arr = plan.blocks[blk]
            t = arr.tR if side == "outer" else arr.tL
            out = out + scatter(plan, blk, side, d / jnp.sum(t * t))
    return out


def flux_from_potential(plan: NodalPlan, phi_t_ext, rho_star):
    """``F = h x (D1, D2, D_eta) phi~ / rho*`` on the owned planes of a halo-``>= 2`` extended ``phi~``."""
    h = halo_of(plan, phi_t_ext)
    own = crop(phi_t_ext, h)
    g1, g2 = d1(plan, own), d2(plan, own)
    if plan.structure.core is not None:
        cb, _p, _R, n_c, _dm, ring_blk = plan.structure.core
        arr, o0 = plan.blocks[cb], plan.structure.blocks[cb][1]
        phi_c = own[:, o0:o0 + n_c]
        tr = trace(plan, ring_blk, "inner", own)
        g1 = g1.at[:, o0:o0 + n_c].set(_apply(arr.G1_phi, phi_c) + _apply(arr.G1_tr, tr))
        g2 = g2.at[:, o0:o0 + n_c].set(_apply(arr.G2_phi, phi_c) + _apply(arr.G2_tr, tr))
    grad = jnp.stack([g1, g2, d_eta(phi_t_ext, plan.structure.deta, h)], axis=-1)
    return jnp.cross(plan.h, grad) / rho_star


def velocity_flux_ext(plan: NodalPlan, phi_ext, rho_star):
    """Velocity flux (with level D5c) on the owned planes of a halo-``>= 2`` extended ``phi``."""
    return flux_from_potential(plan, level_match(plan, phi_ext), rho_star)


def velocity_flux(plan: NodalPlan, phi, rho_star):
    return velocity_flux_ext(plan, extend_periodic(phi, 2), rho_star)


# ---------------------------------------------------------------------------
# Volume terms
# ---------------------------------------------------------------------------
def advect(plan: NodalPlan, F_ext, g_ext):
    """``A g = sum_d (F_d / |J|) D_d g`` on the owned planes."""
    h = halo_of(plan, g_ext)
    g, F = crop(g_ext, h), crop(F_ext, h)
    V = F / plan.jac[..., None]
    return (bc(V[..., 0], g) * d1(plan, g) + bc(V[..., 1], g) * d2(plan, g)
            + bc(V[..., 2], g) * d_eta(g_ext, plan.structure.deta, h))


def divergence(plan: NodalPlan, F_ext, g_ext):
    """``K g = (D1 (F1 g) + D2 (F2 g) + D_eta (F3 g)) / |J|`` on the owned planes."""
    h = halo_of(plan, g_ext)
    g, F = crop(g_ext, h), crop(F_ext, h)
    f3g = bc(F_ext[..., 2], g_ext) * g_ext
    num = (d1(plan, bc(F[..., 0], g) * g) + d2(plan, bc(F[..., 1], g) * g) + d_eta(f3g, plan.structure.deta, h))
    return num / bc(plan.jac, g)


# ---------------------------------------------------------------------------
# SATs (owned planes)
# ---------------------------------------------------------------------------
def _flux_traces(plan, F, b, side):
    return flux_trace(plan, b, side, F[..., 0], F[..., 1])


def _flux_traces_g(plan, F, g, b, side):
    return flux_trace(plan, b, side, bc(F[..., 0], g) * g, bc(F[..., 1], g) * g)


def interface_centred(plan: NodalPlan, F, g):
    """Energy-neutral centred interface SAT on every face (owned-plane ``F``, ``g``)."""
    out = jnp.zeros_like(g)
    for face, (a_blk, b_blk, _na, _nb, _x) in zip(plan.faces, plan.structure.faces):
        a, b = trace(plan, a_blk, "outer", g), trace(plan, b_blk, "inner", g)
        fA, fB = _flux_traces_g(plan, F, g, a_blk, "outer"), _flux_traces_g(plan, F, g, b_blk, "inner")
        vA, vB = _flux_traces(plan, F, a_blk, "outer"), _flux_traces(plan, F, b_blk, "inner")
        vA, vB = bc(vA, a), bc(vB, b)
        MAb = 0.5 * (vA * _apply(face.Iba, b) + _apply(face.Iba, vB * b))
        MBa = 0.5 * (_apply(face.Iab, vA * a) + vB * _apply(face.Iab, a))
        out = out + 0.5 * lift(plan, a_blk, "outer", fA - MAb) - 0.5 * lift(plan, b_blk, "inner", fB - MBa)
    return out


def wall_correction(plan: NodalPlan, F, g):
    """D7+ product correction ``sigma / 2 lift(f - v a)`` summed over the walls."""
    out = jnp.zeros_like(g)
    for blk, side, sign, _n in plan.structure.walls:
        a = trace(plan, blk, side, g)
        f = _flux_traces_g(plan, F, g, blk, side)
        v = bc(_flux_traces(plan, F, blk, side), a)
        out = out + 0.5 * sign * lift(plan, blk, side, f - v * a)
    return out


def wall_inflow(plan: NodalPlan, F, g, bcd: SatBoundaryData | None = None):
    """Inflow SAT ``-lift(tau (a - data))``, ``tau = max(-sigma v, 0)``, summed over the walls (zero data if ``bcd`` is None)."""
    out = jnp.zeros_like(g)
    for w, (blk, side, sign, _n) in enumerate(plan.structure.walls):
        a = trace(plan, blk, side, g)
        v = bc(_flux_traces(plan, F, blk, side), a)
        tau = jnp.maximum(-sign * v, 0.0)
        data = 0.0
        if bcd is not None and bcd.value is not None:
            data = jnp.asarray(bcd.value[w])
            if data.ndim == a.ndim + 1 and data.shape[-1] == 1:
                data = data[..., 0]
        out = out - lift(plan, blk, side, tau * (a - data))
    return out


def interface_upwind(plan: NodalPlan, F, g):
    """Upwind jump penalty on the finer side's face grid of every level face."""
    out = jnp.zeros_like(g)
    for face, (a_blk, b_blk, NA, NB, X) in zip(plan.faces, plan.structure.faces):
        a, b = trace(plan, a_blk, "outer", g), trace(plan, b_blk, "inner", g)
        vA = bc(_flux_traces(plan, F, a_blk, "outer"), a)
        vB = bc(_flux_traces(plan, F, b_blk, "inner"), b)
        if X == "B":
            JA, JB, NX = face.Iab, None, NB
        else:
            JA, JB, NX = None, face.Iba, NA
        JAv = lambda x: x if JA is None else _apply(JA, x)  # noqa: E731
        JBv = lambda x: x if JB is None else _apply(JB, x)  # noqa: E731
        gamma = 0.5 * jnp.abs(0.5 * (JAv(vA) + JBv(vB)))
        q = (TWO_PI / NX) * gamma * (JBv(b) - JAv(a))
        qA = q if JA is None else _apply(JA.T, q)
        qB = q if JB is None else _apply(JB.T, q)
        out = out + (scatter(plan, a_blk, "outer", qA) - scatter(plan, b_blk, "inner", qB)) / bc(plan.Hp, out)
    return out


# ---------------------------------------------------------------------------
# Operator assembly
# ---------------------------------------------------------------------------
def transport_ext(plan: NodalPlan, F_ext, g_ext):
    """``T g = -1/2 (A g + K g) + iface_centred(g) + wall_correction(g)`` on the owned planes."""
    h = halo_of(plan, g_ext)
    g, F = crop(g_ext, h), crop(F_ext, h)
    return (-0.5 * (advect(plan, F_ext, g_ext) + divergence(plan, F_ext, g_ext))
            + interface_centred(plan, F, g) + wall_correction(plan, F, g))


def compressibility_ext(plan: NodalPlan, F_ext):
    """``c = -2 T(1)`` on the owned planes, shape ``(E, P)``."""
    return -2.0 * transport_ext(plan, F_ext, jnp.ones(F_ext.shape[:2], F_ext.dtype))


def linear_operator_ext(plan: NodalPlan, F_ext, g_ext, bcd: SatBoundaryData | None = None):
    """``L g = T g + c g / 2 + wall_inflow + interface_upwind`` on the owned planes."""
    h = halo_of(plan, g_ext)
    g, F = crop(g_ext, h), crop(F_ext, h)
    c = compressibility_ext(plan, F_ext)
    return (transport_ext(plan, F_ext, g_ext) + 0.5 * bc(c, g) * g + wall_inflow(plan, F, g, bcd)
            + interface_upwind(plan, F, g))


def sbp_bracket_ext(plan: NodalPlan, F_ext, g_ext, bcd: SatBoundaryData | None = None, c_kappa=1.0):
    """``L g`` plus the dissipation (face jumps and core shell damping), from extended ``F`` and ``g`` (halo >= 3).

    ``c_kappa`` scales the core shell damping and is a traced argument (not static).
    """
    return linear_operator_ext(plan, F_ext, g_ext, bcd) + dissipation_ext(plan, F_ext, g_ext, c_kappa)


def _wrap(F, g):
    return extend_periodic(F, HALO), extend_periodic(g, HALO)


def transport(plan: NodalPlan, F, g):
    return transport_ext(plan, *_wrap(F, g))


def compressibility(plan: NodalPlan, F):
    return compressibility_ext(plan, extend_periodic(F, HALO))


def linear_operator(plan: NodalPlan, F, g, bcd: SatBoundaryData | None = None):
    return linear_operator_ext(plan, *_wrap(F, g), bcd)


def sbp_bracket(plan: NodalPlan, phi, g, bcd: SatBoundaryData | None, rho_star, c_kappa=1.0):
    """The full right-hand-side bracket ``L g + dissipation`` for owned ``phi (E, P)`` and ``g (E, P[, F])``."""
    F = velocity_flux(plan, phi, rho_star)
    return sbp_bracket_ext(plan, *_wrap(F, g), bcd, c_kappa)
