"""Face-jump dissipation of the nodal SBP perpendicular scheme: ``-H^-1 sum G^T diag(s) G``.

Each term uses the jump ``[g] = (g_{i-1} - 3 g_i + 3 g_{i+1} - g_{i+2}) / 8`` across the face between nodes ``i`` and
``i + 1`` and a face weight ``s = 1/2 w_f |U_n|`` with ``|U_n| = |1/2 (F_left + F_right)|`` (no tunable factor); it is
negative semidefinite in ``H = Hp * deta`` by construction.

- ``theta`` faces of each ring (periodic): ``w_f = w_a du deta``, ``U_n = F2``;
- ``u`` faces inside each ring level (between local rings ``a`` and ``a + 1`` for ``a = 1 .. m - 3``; no faces across
  a level end, the upwind interface SAT covers those): ``w_f = (2 pi / N) deta``, ``U_n = F1``;
- ``eta`` faces on all nodes of any block: ``w_f = wxy`` (extended planes, halo at least 3).

The core's shell damping (``core_damping_ext``) acts on the core nodes only and is part of ``dissipation_ext``.
Functions ending in ``_ext`` take halo-extended ``F, g`` (halo >= 3) and return the owned planes; the others take
owned arrays and wrap periodically.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from drbx.native.fci_perpendicular_sbp_ops import bc, crop, extend_periodic, halo_of
from drbx.stencils.nodal_plan import NodalPlan, RingBlockArrays

TWO_PI = 2.0 * np.pi
C_JUMP = (1.0 / 8.0, -3.0 / 8.0, 3.0 / 8.0, -1.0 / 8.0)
HALO = 3


def _gt(q_next, q, q_prev, q_prev2):
    """``G^T q`` of the jump stencil: ``C0 q_{k+1} + C1 q_k + C2 q_{k-1} + C3 q_{k-2}``."""
    return C_JUMP[0] * q_next + C_JUMP[1] * q + C_JUMP[2] * q_prev + C_JUMP[3] * q_prev2


def _ring_numerators(plan: NodalPlan, F, g):
    """``-sum G^T diag(s) G g`` over the theta and u faces of all ring levels, on owned planes."""
    st = plan.structure
    out = []
    for desc, arr in zip(st.blocks, plan.blocks):
        _, o, nb, m, N, _ = desc
        if not isinstance(arr, RingBlockArrays):
            out.append(jnp.zeros_like(g[:, o:o + nb]))
            continue
        E = g.shape[0]
        gl = g[:, o:o + nb].reshape((E, m, N) + g.shape[2:])
        F1 = F[:, o:o + nb, 0].reshape(E, m, N)
        F2 = F[:, o:o + nb, 1].reshape(E, m, N)
        # theta faces (periodic in j)
        jump = C_JUMP[0] * jnp.roll(gl, 1, axis=2) + C_JUMP[1] * gl + C_JUMP[2] * jnp.roll(gl, -1, axis=2) \
            + C_JUMP[3] * jnp.roll(gl, -2, axis=2)
        wf = (arr.w * st.du * st.deta)[None, :, None]
        s = 0.5 * wf * jnp.abs(0.5 * (F2 + jnp.roll(F2, -1, axis=2)))
        q = bc(s, jump) * jump
        term = -_gt(jnp.roll(q, -1, axis=2), q, jnp.roll(q, 1, axis=2), jnp.roll(q, 2, axis=2))
        # u faces inside the level (a = 1 .. m - 3)
        jump = C_JUMP[0] * gl[:, 0:m - 3] + C_JUMP[1] * gl[:, 1:m - 2] + C_JUMP[2] * gl[:, 2:m - 1] + C_JUMP[3] * gl[:, 3:m]
        s = 0.5 * (TWO_PI / N * st.deta) * jnp.abs(0.5 * (F1[:, 1:m - 2] + F1[:, 2:m - 1]))
        q = bc(s, jump) * jump
        qp = jnp.pad(q, [(0, 0), (3, 4)] + [(0, 0)] * (q.ndim - 2))   # face a (1 .. m - 3) sits at qp[:, a + 2]
        term = term - _gt(qp[:, 3:m + 3], qp[:, 2:m + 2], qp[:, 1:m + 1], qp[:, 0:m])
        out.append(term.reshape((E, nb) + g.shape[2:]))
    return jnp.concatenate(out, axis=1)


def _eta_numerator(plan: NodalPlan, F_ext, g_ext, halo: int):
    """``-G^T diag(s) G g`` over the eta faces of every node, on owned planes."""
    if halo < HALO:
        raise ValueError(f"the dissipation needs a halo of at least {HALO} planes, got {halo}")
    E_ext = g_ext.shape[0]
    e_own = E_ext - 2 * halo
    jump = (C_JUMP[0] * g_ext[0:E_ext - 3] + C_JUMP[1] * g_ext[1:E_ext - 2] + C_JUMP[2] * g_ext[2:E_ext - 1]
            + C_JUMP[3] * g_ext[3:E_ext])
    F3 = F_ext[..., 2]
    s = 0.5 * plan.wxy[None, :] * jnp.abs(0.5 * (F3[1:E_ext - 2] + F3[2:E_ext - 1]))
    q = bc(s, jump) * jump

    def at(shift):
        i = halo + shift
        return q[i:i + e_own]

    return -_gt(at(0), at(-1), at(-2), at(-3))


def dissipation_ring_ext(plan: NodalPlan, F_ext, g_ext):
    """Ring theta/u face dissipation ``/ H`` on the owned planes."""
    h = halo_of(plan, g_ext)
    H = plan.Hp * plan.structure.deta
    num = _ring_numerators(plan, crop(F_ext, h), crop(g_ext, h))
    return num / bc(H, num)


def dissipation_eta_ext(plan: NodalPlan, F_ext, g_ext):
    """Eta face dissipation ``/ H`` on the owned planes (all nodes)."""
    h = halo_of(plan, g_ext)
    H = plan.Hp * plan.structure.deta
    num = _eta_numerator(plan, F_ext, g_ext, h)
    return num / bc(H, num)


def core_damping_ext(plan: NodalPlan, F_ext, g_ext, c_kappa=1.0):
    """Core shell damping ``-kappa_k P_h g`` on the core nodes, zero elsewhere (owned planes).

    ``P_h g = g - Vm Ginv_k Vm^T (Hk g)`` removes the part of ``g`` outside ``P_{p-1}`` (``H``-orthogonally, ``Hk = wxy * jac``
    per plane) and ``kappa_k = c_kappa p / R_c max_core sqrt(V_x^2 + V_y^2)`` with ``V = F / jac`` in the block frame. The
    term is not divided by ``H`` (it is already a rate); ``c_kappa`` is traced and the per-plane max is plane-local.
    """
    out = jnp.zeros_like(crop(g_ext, halo_of(plan, g_ext)))
    if plan.structure.core is None:
        return out
    h = halo_of(plan, g_ext)
    g, F = crop(g_ext, h), crop(F_ext, h)
    cb, p, R_c, n_c, _dm, _ring = plan.structure.core
    o0 = plan.structure.blocks[cb][1]
    sl = slice(o0, o0 + n_c)
    Vm = plan.blocks[cb].Vm
    jac_c = plan.jac[:, sl]
    gc = g[:, sl]
    vxy = jnp.sqrt(jnp.sum((F[:, sl, :2] / jac_c[..., None]) ** 2, axis=-1))
    kappa = c_kappa * (p / R_c) * jnp.max(vxy, axis=1)
    Hk = plan.wxy[sl][None, :] * jac_c
    coef = jnp.einsum("qd,eq...->ed...", Vm, bc(Hk, gc) * gc)
    coef = jnp.einsum("edf,ef...->ed...", plan.core_Ginv, coef)
    ph = gc - jnp.einsum("qd,ed...->eq...", Vm, coef)
    return out.at[:, sl].set(-bc(kappa, gc) * ph)


def dissipation_ext(plan: NodalPlan, F_ext, g_ext, c_kappa=1.0):
    """Total dissipation on the owned planes of halo-``>= 3`` extended ``F`` and ``g``: face jumps plus core shell damping."""
    return (dissipation_ring_ext(plan, F_ext, g_ext) + dissipation_eta_ext(plan, F_ext, g_ext)
            + core_damping_ext(plan, F_ext, g_ext, c_kappa))


def _wrapped(fn, plan, F, g, *args):
    return fn(plan, extend_periodic(F, HALO), extend_periodic(g, HALO), *args)


def dissipation_ring(plan: NodalPlan, F, g):
    return _wrapped(dissipation_ring_ext, plan, F, g)


def dissipation_eta(plan: NodalPlan, F, g):
    return _wrapped(dissipation_eta_ext, plan, F, g)


def dissipation(plan: NodalPlan, F, g, c_kappa=1.0):
    """Dissipation (face jumps plus core shell damping) of owned ``F (E, P, 3)``, ``g (E, P[, ...])`` (periodic eta)."""
    return _wrapped(dissipation_ext, plan, F, g, c_kappa)


def core_damping(plan: NodalPlan, F, g, c_kappa=1.0):
    return _wrapped(core_damping_ext, plan, F, g, c_kappa)
