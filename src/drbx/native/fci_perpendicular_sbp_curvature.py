"""P06 curvature operator on the nodal SBP layout: split-form derivative along the curvature flux, matrix wall SAT and options.

Continuum form (``p06_midpoint_material_remainder``), fields ``(n, Te, Ti, omega, phi)``::

    rhs[0:4] = M_psi(q) C(q)[0:4] + coeff C(psi),   coeff = (-2 n, -4 Te / 3, -4 Ti / 3, 0),   C(f) = (K / B) . grad f,

with ``M_psi = curvature_principal_matrix(n, Te, Ti, B, tau, psi=psi)`` and ``C(psi) = C(phi) + tau C(Ti)`` (legacy,
``"phi_plus_tau_ti"``) or ``C(phi) + tau (Ti C(n) + n C(Ti))`` (``"phi_plus_tau_pi"``, the chain rule at the node).

Discretisation. ``F = |J| K / B`` (``curvature_flux``) is divergence free in the continuum (``|J| K / B = curl(b_cov / B) / 2``)
and plays the role of the bracket's E x B flux. The scalar derivative along it is the split form of the bracket::

    C_h g = -(T_F g + c g / 2) -> + (F / |J|) . grad g,       c = -2 T_F(1),

with ``T_F`` the transport of ``fci_perpendicular_sbp_bracket`` (centred interface SAT and wall product correction included), so
``C_h(1) = 0`` and ``sum_H [f C_h g + g C_h f] = sum_walls Omega sigma v a_f a_g - sum_H c f g`` exactly (``v`` the normal trace
of ``F``, ``a`` the wall traces). ``sbp_curvature`` applies the pointwise matrix ``M_psi(q)`` to the columns ``C_h q`` and adds

* the **wall inflow SAT** ``-lift(X+ (a - data))``, ``X = sigma v M_psi(a)``, ``X+ = (X + |X|) / 2``, ``|X| = |sigma v| |M_psi|``
  through the P06 absolute action (closed form or LAPACK), which makes every wall term ``<= 0`` in ``H (x) W`` for a frozen
  ``M`` (``W = R^-T R^-1`` for the eigenvector matrix ``R`` of ``M``);
* optionally (off by default, measured, not adopted) the matrix generalisation of the face-jump dissipation and of the level-face
  upwind with the weight ``w |U_n| / 2`` replaced by ``w |U_n| |M_psi(q_face)| / 2``, and the core shell damping with
  ``kappa = c_kappa (p / R_c) max_core(|V_xy| rho(M))``.

Array conventions are those of ``fci_perpendicular_sbp_ops``: ``F (E, P, 3)``, ``q (E, P, 5)``, outputs ``(E, P, 4)``;
functions ending in ``_ext`` take halo-extended ``F`` and fields (halo >= 3) and return the owned planes.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from drbx.native.fci_curvature_production_flux import curvature_principal_matrix
from drbx.native.fci_perpendicular_face_corrections import (
    ABSOLUTE_METHODS,
    _absolute_action_closed_form,
    _p06_absolute_action,
    _validated_absolute_method,
)
from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData
from drbx.native.fci_perpendicular_sbp_bracket import compressibility_ext, transport_ext
from drbx.native.fci_perpendicular_sbp_dissipation import C_JUMP, HALO, _gt
from drbx.native.fci_perpendicular_sbp_ops import bc, crop, extend_periodic, flux_trace, halo_of, lift, scatter, trace
from drbx.stencils.nodal_plan import NodalPlan, RingBlockArrays

TWO_PI = 2.0 * np.pi
FLOOR = 1.0e-12
PSI_VARIANTS = ("phi_plus_tau_ti", "phi_plus_tau_pi")

__all__ = [
    "ABSOLUTE_METHODS", "PSI_VARIANTS", "curvature_flux", "curvature_derivative", "curvature_derivative_ext",
    "compressibility_flux", "absolute_action", "spectral_radius", "wall_inflow_matrix", "jump_dissipation_ext",
    "interface_upwind_matrix", "core_damping_matrix_ext", "sbp_curvature_ext", "sbp_curvature", "continuum_rhs",
]


def _check_psi(psi: str) -> None:
    if psi not in PSI_VARIANTS:
        raise ValueError(f"psi must be one of {PSI_VARIANTS}, got {psi!r}")


def _apply(I, x):
    return jnp.einsum("ij,ej...->ei...", I, x)


# ---------------------------------------------------------------------------
# Flux and scalar derivative
# ---------------------------------------------------------------------------
def curvature_flux(plan: NodalPlan):
    """``F = |J| K / B`` in each block's frame, ``(E, P, 3)`` (divergence free in the continuum)."""
    return plan.jac[..., None] * plan.K / plan.B[..., None]


def compressibility_flux(plan: NodalPlan, F_ext):
    """``c = -2 T_F(1)``: the discrete divergence of ``F`` (``(E, P)``, owned planes)."""
    return compressibility_ext(plan, F_ext)


def curvature_derivative_ext(plan: NodalPlan, F_ext, g_ext, c=None):
    """``C_h g = -(T_F g + c g / 2)`` on the owned planes; ``g (E, P[, ...])`` carries any trailing field axes."""
    h = halo_of(plan, g_ext)
    if c is None:
        c = compressibility_ext(plan, F_ext)
    T = transport_ext(plan, F_ext, g_ext)
    return -(T + 0.5 * bc(c, T) * crop(g_ext, h))


def curvature_derivative(plan: NodalPlan, F, g):
    """``C_h g`` of owned ``F (E, P, 3)`` and ``g`` (periodic eta)."""
    return curvature_derivative_ext(plan, extend_periodic(F, HALO), extend_periodic(g, HALO))


# ---------------------------------------------------------------------------
# Matrix pieces: |scale * M_psi| action, spectral radius
# ---------------------------------------------------------------------------
def absolute_action(method: str, n, te, ti, b, tau, scale, jump, psi: str = "phi_plus_tau_ti", floor: float = FLOOR):
    """``|scale M_psi(n, Te, Ti, B, tau)| jump`` with the P06 absolute action (``"closed_form"`` or ``"lapack4"``).

    ``n, Te, Ti, B, scale`` broadcast to the leading shape of ``jump (..., 4)``.
    """
    _validated_absolute_method(method)
    _check_psi(psi)
    n, te, ti, b, scale = (jnp.broadcast_to(jnp.asarray(x, dtype=jump.dtype), jump.shape[:-1]) for x in (n, te, ti, b, scale))
    matrix = scale[..., None, None] * curvature_principal_matrix(n, te, ti, b, tau, psi=psi)
    if method == "closed_form":
        action, _invalid = _absolute_action_closed_form(n, te, ti, b, tau, scale, matrix, jump, floor, psi=psi)
    else:
        action, _invalid = _p06_absolute_action(matrix, jump)
    return action


def _monic_cubic(n, te, ti, tau, psi):
    r = tau * ti / te
    if psi == "phi_plus_tau_pi":
        s = n * r
        return (30 * (r - s) - 60) / 9, (60 + 100 * s - 200 * r - 60 * r * s) / 9, 200 * r * (1 + s) / 9
    return (18 * r - 60) / 9, (60 - 160 * r) / 9, 200 * r / 9


def spectral_radius(n, te, ti, tau, psi: str = "phi_plus_tau_ti"):
    """Spectral radius of ``M_psi``: ``Te max |mu|`` over the three real roots of the (n, Te, Ti) block cubic (Viete)."""
    _check_psi(psi)
    a2, a1, a0 = _monic_cubic(n, te, ti, tau, psi)
    shift = a2 / 3
    p = a1 - a2 * a2 / 3
    q = 2 * a2**3 / 27 - a2 * a1 / 3 + a0
    radius = jnp.sqrt(jnp.maximum(-p / 3, 1e-300))
    angle = jnp.arccos(jnp.clip(-q / (2 * radius**3), -1.0 + 1e-15, 1.0 - 1e-15)) / 3
    mu_max = 2 * radius * jnp.cos(angle) - shift
    mu_min = 2 * radius * jnp.cos(angle - 4 * jnp.pi / 3) - shift
    return te * jnp.maximum(jnp.abs(mu_max), jnp.abs(mu_min))


def _material(n, te, ti, b, tau, psi, x):
    """``M_psi(n, Te, Ti, B) x`` for ``x (..., 4)`` (the matrix is not assembled for the caller)."""
    M = curvature_principal_matrix(n, te, ti, b, tau, psi=psi)
    return jnp.einsum("...ij,...j->...i", M, x)


# ---------------------------------------------------------------------------
# Wall inflow SAT
# ---------------------------------------------------------------------------
def _data_of(bcd, w, like):
    if bcd is None or bcd.value is None:
        return jnp.zeros_like(like)
    data = jnp.asarray(bcd.value[w])
    return data[..., : like.shape[-1]]


def wall_inflow_matrix(plan: NodalPlan, F, state, g, bcd: SatBoundaryData | None = None, *, tau=1.0,
                       psi="phi_plus_tau_ti", absolute_method="closed_form"):
    """``-sum_walls lift(X+ (a - data))`` with ``X = sigma v M_psi(a_state)``, ``X+ = (X + |X|) / 2`` (owned planes).

    ``state (E, P, >=3)`` supplies ``(n, Te, Ti)`` of ``M_psi`` (its wall traces), ``g (E, P, 4)`` the penalised field (the
    state itself in ``sbp_curvature``), ``F`` the owned curvature flux; ``data`` is the Dirichlet wall value of ``g``.
    """
    out = jnp.zeros_like(g)
    for w, (blk, side, sign, _n) in enumerate(plan.structure.walls):
        a = trace(plan, blk, side, g)
        s = trace(plan, blk, side, state[..., :3])
        b = trace(plan, blk, side, plan.B)
        v = flux_trace(plan, blk, side, F[..., 0], F[..., 1])
        jump = a - _data_of(bcd, w, a)
        scale = sign * v
        n, te, ti = s[..., 0], s[..., 1], s[..., 2]
        material = scale[..., None] * _material(n, te, ti, b, tau, psi, jump)
        absolute = absolute_action(absolute_method, n, te, ti, b, tau, scale, jump, psi)
        out = out - lift(plan, blk, side, 0.5 * (material + absolute))
    return out


# ---------------------------------------------------------------------------
# Matrix face-jump dissipation and level-face upwind
# ---------------------------------------------------------------------------
def _face_absolute(method, psi, tau, sbar, bbar, scale, jump):
    return absolute_action(method, sbar[..., 0], sbar[..., 1], sbar[..., 2], bbar, tau, scale, jump, psi)


def _ring_numerators(plan: NodalPlan, F, state, g, tau, psi, method):
    """``-sum G^T s |M(q_face)| G g`` over the theta and u faces of all ring levels on owned planes."""
    st = plan.structure
    out = []
    for desc, arr in zip(st.blocks, plan.blocks):
        _, o, nb, m, N, _ = desc
        if not isinstance(arr, RingBlockArrays):
            out.append(jnp.zeros_like(g[:, o:o + nb]))
            continue
        E = g.shape[0]
        gl = g[:, o:o + nb].reshape((E, m, N, g.shape[-1]))
        sl = state[:, o:o + nb, :3].reshape((E, m, N, 3))
        Bl = plan.B[:, o:o + nb].reshape(E, m, N)
        F1 = F[:, o:o + nb, 0].reshape(E, m, N)
        F2 = F[:, o:o + nb, 1].reshape(E, m, N)
        # theta faces (periodic in j)
        jump = C_JUMP[0] * jnp.roll(gl, 1, axis=2) + C_JUMP[1] * gl + C_JUMP[2] * jnp.roll(gl, -1, axis=2) \
            + C_JUMP[3] * jnp.roll(gl, -2, axis=2)
        wf = (arr.w * st.du * st.deta)[None, :, None]
        scale = 0.5 * wf * jnp.abs(0.5 * (F2 + jnp.roll(F2, -1, axis=2)))
        q = _face_absolute(method, psi, tau, 0.5 * (sl + jnp.roll(sl, -1, axis=2)), 0.5 * (Bl + jnp.roll(Bl, -1, axis=2)),
                           scale, jump)
        term = -_gt(jnp.roll(q, -1, axis=2), q, jnp.roll(q, 1, axis=2), jnp.roll(q, 2, axis=2))
        # u faces inside the level (a = 1 .. m - 3)
        jump = C_JUMP[0] * gl[:, 0:m - 3] + C_JUMP[1] * gl[:, 1:m - 2] + C_JUMP[2] * gl[:, 2:m - 1] + C_JUMP[3] * gl[:, 3:m]
        scale = 0.5 * (TWO_PI / N * st.deta) * jnp.abs(0.5 * (F1[:, 1:m - 2] + F1[:, 2:m - 1]))
        q = _face_absolute(method, psi, tau, 0.5 * (sl[:, 1:m - 2] + sl[:, 2:m - 1]), 0.5 * (Bl[:, 1:m - 2] + Bl[:, 2:m - 1]),
                           scale, jump)
        qp = jnp.pad(q, [(0, 0), (3, 4)] + [(0, 0)] * (q.ndim - 2))
        term = term - _gt(qp[:, 3:m + 3], qp[:, 2:m + 2], qp[:, 1:m + 1], qp[:, 0:m])
        out.append(term.reshape((E, nb, g.shape[-1])))
    return jnp.concatenate(out, axis=1)


def _eta_numerator(plan: NodalPlan, F_ext, state_ext, g_ext, halo, tau, psi, method, B_ext=None):
    if halo < HALO:
        raise ValueError(f"the dissipation needs a halo of at least {HALO} planes, got {halo}")
    E_ext = g_ext.shape[0]
    e_own = E_ext - 2 * halo
    jump = (C_JUMP[0] * g_ext[0:E_ext - 3] + C_JUMP[1] * g_ext[1:E_ext - 2] + C_JUMP[2] * g_ext[2:E_ext - 1]
            + C_JUMP[3] * g_ext[3:E_ext])
    F3 = F_ext[..., 2]
    scale = 0.5 * plan.wxy[None, :] * jnp.abs(0.5 * (F3[1:E_ext - 2] + F3[2:E_ext - 1]))
    B_ext = extend_periodic(plan.B, halo) if B_ext is None else B_ext
    sbar = 0.5 * (state_ext[1:E_ext - 2, :, :3] + state_ext[2:E_ext - 1, :, :3])
    bbar = 0.5 * (B_ext[1:E_ext - 2] + B_ext[2:E_ext - 1])
    q = _face_absolute(method, psi, tau, sbar, bbar, scale, jump)

    def at(shift):
        i = halo + shift
        return q[i:i + e_own]

    return -_gt(at(0), at(-1), at(-2), at(-3))


def jump_dissipation_ext(plan: NodalPlan, F_ext, state_ext, g_ext, *, tau=1.0, psi="phi_plus_tau_ti",
                         absolute_method="closed_form", B_ext=None):
    """``-H^-1 sum G^T (w |U_n| / 2) |M_psi(q_face)| G g`` over the theta, u (inside levels) and eta faces, owned planes.

    ``state_ext (E_ext, P, >=3)`` gives ``(n, Te, Ti)`` of the face matrices (face average of the two centre nodes), ``g_ext
    (E_ext, P, 4)`` is the dissipated field, ``F_ext`` the extended curvature flux (all halo >= 3); ``B_ext`` the extended ``B``
    (default: the periodic wrap of ``plan.B``; give it when the planes of ``plan`` are a chunk of a larger array).
    """
    _validated_absolute_method(absolute_method)
    _check_psi(psi)
    h = halo_of(plan, g_ext)
    H = plan.Hp * plan.structure.deta
    ring = _ring_numerators(plan, crop(F_ext, h), crop(state_ext, h), crop(g_ext, h), tau, psi, absolute_method)
    eta = _eta_numerator(plan, F_ext, state_ext, g_ext, h, tau, psi, absolute_method, B_ext)
    return (ring + eta) / bc(H, ring)


def interface_upwind_matrix(plan: NodalPlan, F, state, g, *, tau=1.0, psi="phi_plus_tau_ti", absolute_method="closed_form"):
    """Level-face upwind jump penalty on the finer side's grid with ``gamma -> |v_bar| |M_psi(q_bar)| / 2`` (owned planes)."""
    out = jnp.zeros_like(g)
    for face, (a_blk, b_blk, NA, NB, X) in zip(plan.faces, plan.structure.faces):
        a, b = trace(plan, a_blk, "outer", g), trace(plan, b_blk, "inner", g)
        sa, sb = trace(plan, a_blk, "outer", state[..., :3]), trace(plan, b_blk, "inner", state[..., :3])
        Ba, Bb = trace(plan, a_blk, "outer", plan.B), trace(plan, b_blk, "inner", plan.B)
        vA = flux_trace(plan, a_blk, "outer", F[..., 0], F[..., 1])
        vB = flux_trace(plan, b_blk, "inner", F[..., 0], F[..., 1])
        if X == "B":
            JA, JB, NX = face.Iab, None, NB
        else:
            JA, JB, NX = None, face.Iba, NA
        JAv = lambda x: x if JA is None else _apply(JA, x)  # noqa: E731
        JBv = lambda x: x if JB is None else _apply(JB, x)  # noqa: E731
        gamma = 0.5 * jnp.abs(0.5 * (JAv(vA) + JBv(vB)))
        jump = JBv(b) - JAv(a)
        sbar = 0.5 * (JAv(sa) + JBv(sb))
        bbar = 0.5 * (JAv(Ba) + JBv(Bb))
        q = _face_absolute(absolute_method, psi, tau, sbar, bbar, (TWO_PI / NX) * gamma, jump)
        qA = q if JA is None else _apply(JA.T, q)
        qB = q if JB is None else _apply(JB.T, q)
        out = out + (scatter(plan, a_blk, "outer", qA) - scatter(plan, b_blk, "inner", qB)) / bc(plan.Hp, out)
    return out


def core_damping_matrix_ext(plan: NodalPlan, F_ext, state_ext, g_ext, c_kappa=1.0, *, tau=1.0, psi="phi_plus_tau_ti"):
    """Core shell damping ``-kappa P_h g`` on the core nodes (zero elsewhere), ``kappa = c_kappa (p / R_c) max_core(|V_xy| rho(M))``.

    ``V = F / |J|`` in the block frame, ``rho`` the spectral radius of ``M_psi`` at the node, ``P_h`` the shell projector of
    ``core_damping_ext``; ``kappa`` is plane local and ``c_kappa`` may be traced.
    """
    h = halo_of(plan, g_ext)
    g, F, s = crop(g_ext, h), crop(F_ext, h), crop(state_ext, h)
    out = jnp.zeros_like(g)
    if plan.structure.core is None:
        return out
    cb, p, R_c, n_c, _dm, _ring = plan.structure.core
    o0 = plan.structure.blocks[cb][1]
    sl = slice(o0, o0 + n_c)
    Vm = plan.blocks[cb].Vm
    jac_c = plan.jac[:, sl]
    gc = g[:, sl]
    rho = spectral_radius(s[:, sl, 0], s[:, sl, 1], s[:, sl, 2], tau, psi)
    vxy = jnp.sqrt(jnp.sum((F[:, sl, :2] / jac_c[..., None]) ** 2, axis=-1))
    kappa = c_kappa * (p / R_c) * jnp.max(vxy * rho, axis=1)
    Hk = plan.wxy[sl][None, :] * jac_c
    coef = jnp.einsum("qd,eq...->ed...", Vm, bc(Hk, gc) * gc)
    coef = jnp.einsum("edf,ef...->ed...", plan.core_Ginv, coef)
    ph = gc - jnp.einsum("qd,ed...->eq...", Vm, coef)
    return out.at[:, sl].set(-bc(kappa, gc) * ph)


# ---------------------------------------------------------------------------
# Full operator
# ---------------------------------------------------------------------------
def _remainder_coefficients(n, te, ti):
    return jnp.stack((-2.0 * n, -4.0 * te / 3.0, -4.0 * ti / 3.0, jnp.zeros_like(n)), axis=-1)


def _psi_derivative(Cq, n, ti, tau, psi):
    """``C_h(psi)`` from the scalar columns ``Cq (E, P, 5)``: ``C(phi) + tau C(Ti)`` or ``C(phi) + tau (Ti C(n) + n C(Ti))``."""
    if psi == "phi_plus_tau_pi":
        return Cq[..., 4] + tau * (ti * Cq[..., 0] + n * Cq[..., 2])
    return Cq[..., 4] + tau * Cq[..., 2]


def sbp_curvature_ext(plan: NodalPlan, F_ext, q_ext, bcd: SatBoundaryData | None = None, *, tau=1.0,
                      psi="phi_plus_tau_ti", absolute_method="closed_form", jump_dissipation=False, c_kappa=0.0,
                      B_ext=None):
    """The curvature right-hand side ``(E, P, 4)`` of the extended state ``q_ext (E_ext, P, 5)`` (halo >= 3).

    Centred ``M_psi(q) C_h q[0:4] + coeff C_h(psi)`` plus the matrix wall SAT; ``jump_dissipation=True`` adds the matrix
    face-jump dissipation and the level-face upwind; ``c_kappa`` (a Python 0 skips it, otherwise traced) the core shell damping.
    ``B_ext`` is the extended ``B`` of the eta faces (see ``jump_dissipation_ext``).
    """
    _validated_absolute_method(absolute_method)
    _check_psi(psi)
    h = halo_of(plan, q_ext)
    q, F = crop(q_ext, h), crop(F_ext, h)
    n, te, ti = q[..., 0], q[..., 1], q[..., 2]
    Cq = curvature_derivative_ext(plan, F_ext, q_ext)
    out = _material(n, te, ti, plan.B, tau, psi, Cq[..., :4])
    out = out + _remainder_coefficients(n, te, ti) * _psi_derivative(Cq, n, ti, tau, psi)[..., None]
    out = out + wall_inflow_matrix(plan, F, q, q[..., :4], bcd, tau=tau, psi=psi, absolute_method=absolute_method)
    if jump_dissipation:
        g_ext = q_ext[..., :4]
        out = out + jump_dissipation_ext(plan, F_ext, q_ext, g_ext, tau=tau, psi=psi, absolute_method=absolute_method,
                                                B_ext=B_ext)
        out = out + interface_upwind_matrix(plan, F, q, q[..., :4], tau=tau, psi=psi, absolute_method=absolute_method)
    if not (isinstance(c_kappa, (int, float)) and c_kappa == 0):
        out = out + core_damping_matrix_ext(plan, F_ext, q_ext, q_ext[..., :4], c_kappa, tau=tau, psi=psi)
    return out


def sbp_curvature(plan: NodalPlan, q, bcd: SatBoundaryData | None = None, *, tau=1.0, psi="phi_plus_tau_ti",
                  absolute_method="closed_form", jump_dissipation=False, c_kappa=0.0, F=None):
    """``sbp_curvature_ext`` of owned ``q (E, P, 5)`` (periodic eta); ``F`` defaults to ``curvature_flux(plan)``."""
    F = curvature_flux(plan) if F is None else F
    return sbp_curvature_ext(plan, extend_periodic(F, HALO), extend_periodic(q, HALO), bcd, tau=tau, psi=psi,
                             absolute_method=absolute_method, jump_dissipation=jump_dissipation, c_kappa=c_kappa)


# ---------------------------------------------------------------------------
# Continuum reference (host, exact gradients)
# ---------------------------------------------------------------------------
def continuum_rhs(values, gradients, K, B, tau=1.0, psi="phi_plus_tau_pi"):
    """The continuum RHS ``(..., 4)`` at nodes: ``values (5, ...)``, logical ``gradients (5, ..., 3)``, logical ``K (..., 3)``.

    With the chain rule ``C(psi)`` the result is the same for both ``psi`` to rounding; ``psi`` only selects the split.
    """
    _check_psi(psi)
    values, gradients, K, B = (np.asarray(x, dtype=np.float64) for x in (values, gradients, K, B))
    C = np.einsum("...d,f...d->f...", K, gradients) / B
    n, te, ti = values[0], values[1], values[2]
    M = np.asarray(curvature_principal_matrix(n, te, ti, B, tau, psi=psi))
    material = np.einsum("...ij,j...->...i", M, C[:4])
    if psi == "phi_plus_tau_pi":
        cpsi = C[4] + tau * (ti * C[0] + n * C[2])
    else:
        cpsi = C[4] + tau * C[2]
    coeff = np.stack((-2.0 * n, -4.0 * te / 3.0, -4.0 * ti / 3.0, np.zeros_like(n)), axis=-1)
    return material + coeff * cpsi[..., None]
