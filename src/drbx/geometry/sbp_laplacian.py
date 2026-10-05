"""Host-side builders of the nodal SBP perpendicular Laplacian (diffusion and polarization) on the family-A layout.

The operator acts on nodal point values in the norm ``H = Hp * deta`` shared with the bracket and is the energy form

    ``L f = -H^-1 (M f - b(data))``,  ``v^T M f = a(f, v)``  (symmetric for Dirichlet and conormal-Neumann data),

with the narrow face-flux discretisation chosen in the P09 Laplacian study (``A = J P_perp`` in the logical frame
``(u, theta, eta)``, block frame ``(x, y, eta)`` in the core, ``W = wxy * deta``):

- rings: ``M = sum_{i != j} D_i^T W A_ij D_j + Gu^T Wf a_uu,f Gu + Gt^T W a_tt,h Gt + Ge^T W a_ee,h Ge``. ``D_u`` is the
  collocated radial derivative with the cubic-trace ``t3`` closure, ``Gu`` the staggered radial pair (centres to faces,
  :data:`STAG_CLOSURE`), ``Gt`` the spectral derivative from nodes to half nodes (Nyquist mode as a cosine) and ``Ge``
  the periodic staggered ``(1, -27, 27, -1) / 24`` derivative. Face coefficients are 4-point Lagrange interpolants in
  ``u`` and Fourier interpolants in ``theta`` and ``eta``;
- core: ``sum_{a, b in {x, y}} D_a^T W A_ab D_b`` plus the collocated ``eta`` cross terms, the staggered ``eta eta`` term and
  the shell penalty ``(I - Pi)^T W kappa (I - Pi)``, ``kappa = c_kappa * lambda_max(A_xy) * (p / R_c)^2``;
- core | ring interface: symmetric interior penalty with ``tau``; wall: Nitsche (Dirichlet, ``tau_w``) or the conormal
  flux (Neumann), with a physical-normal to conormal conversion through the wall trace.

Everything here is host NumPy, built once (see the JAX boundary in ``docs/code_structure.md``): stencils, closures, face
interpolation data, the penalty rule, ``kappa``, traces, and the :class:`LaplacianPlan` pytree the JAX apply
(:mod:`drbx.native.fci_perpendicular_sbp_laplacian`) takes as an argument. Host sparse assembly for audits and
preconditioners lives in :mod:`drbx.validation.sbp_laplacian_audit`.

Supported layouts: ``inner="core"`` with exactly one ring level of ``N = n`` nodes (family A) and one outer wall.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields as dc_fields
from fractions import Fraction as _F
from typing import NamedTuple

import jax
import numpy as np

from drbx.geometry.nodal_layout import NodalLayout, RingLevelBlock
from drbx.geometry.sbp_core import CoreBlock
from drbx.geometry.sbp_operators import TWO_PI, ring_basis, radial_block

__all__ = [
    "STAG_CLOSURE", "StaggeredPair", "staggered_pair", "stag_fourier", "fourier_half_interp", "stag_eta",
    "radial_face_interp", "core_frame_A", "LaplacianMetric", "nodal_laplacian_metric_from_callable",
    "LaplacianStructure", "LaplacianPlan", "build_laplacian_plan", "face_coefficients", "penalty_rule",
    "PENALTY_GAMMA", "plane_keys",
]

# ---------------------------------------------------------------------------------------------------------------------
# Staggered radial pair (centres -> faces), closure key (r, s, st, bf) = (4, 6, 4, 5) of the P09 Laplacian study
# ---------------------------------------------------------------------------------------------------------------------
#: ``wf`` leading face weights, ``D+`` closure block (``r = 4`` rows, ``s = 6`` columns; nonzero entries below 1e-13 are
#: round-off of the optimiser that tuned the free parameters for the smallest spectral radius of ``D- D+``) and the
#: staggered boundary extrapolation ``t`` (the unique cubic one, equal to the collocated ``t3`` trace).
STAG_CLOSURE = {
    "wf": (_F(397, 1152), _F(161, 128), _F(111, 128), _F(1187, 1152), _F(1)),
    "D": (
        (-3.0264166472733738e+00, 6.0792499418200405e+00, -4.0792499418199553e+00, 1.0264166472733010e+00,
         -4.9028525224368829e-14, 3.3621385926358095e-14),
        (-9.3687907046064267e-01, 8.1063721138195810e-01, 1.8936278861801392e-01, -6.3120929539321691e-02,
         -4.3466039183903555e-15, -1.8250024264385764e-15),
        (5.8596369466956622e-02, -1.1757891084009140e+00, 1.1757891084009406e+00, -5.8596369466986153e-02,
         -3.2385452269694281e-15, 3.1440986364997431e-15),
        (-1.6438577112468752e-02, 8.9753810528633904e-02, -1.1706299689111181e+00, 1.1377528146861848e+00,
         -4.0438079191237944e-02, -6.5009124182249584e-17),
    ),
    "t": (_F(35, 16), _F(-35, 16), _F(21, 16), _F(-5, 16)),
}
_STAG_R, _STAG_S = 4, 6
_WC = (_F(433, 384), _F(95, 128), _F(451, 384), _F(367, 384))        # the shared radial norm (interior weight 1)
_DINT = np.array([1.0, -27.0, 27.0, -1.0]) / 24.0                     # D+ row k: centres k-2 .. k+1


class StaggeredPair(NamedTuple):
    """Unit-spacing staggered pair on ``m`` centres: ``Dp (m + 1, m)`` centres to faces, ``Dm (m, m + 1)`` faces to centres.

    ``wc (m,)`` is the shared centre norm, ``wf (m + 1,)`` the face weights, ``tL, tR (m,)`` the centre extrapolations to
    the end faces. By construction ``diag(wc) Dm = -Dp^T diag(wf) + tR e_R^T - tL e_L^T``.
    """

    Dp: np.ndarray
    Dm: np.ndarray
    wc: np.ndarray
    wf: np.ndarray
    tL: np.ndarray
    tR: np.ndarray


def staggered_pair(m: int) -> StaggeredPair:
    m = int(m)
    r, s = _STAG_R, _STAG_S
    if m < 2 * r:
        raise ValueError(f"the staggered pair needs m >= {2 * r} centres, got {m}")
    Dc = np.array(STAG_CLOSURE["D"])
    Dp = np.zeros((m + 1, m))
    for k in range(2, m - 1):
        Dp[k, k - 2:k + 2] = _DINT
    for k in range(r):
        Dp[k, :s] = Dc[k]
        Dp[m - k, m - 1 - np.arange(s)] = -Dc[k]
    wfh = np.array([float(v) for v in STAG_CLOSURE["wf"]])
    wf = np.ones(m + 1)
    wf[:wfh.size] = wfh
    wf[m + 1 - wfh.size:] = wfh[::-1]
    wc = np.ones(m)
    wcl = np.array([float(v) for v in _WC])
    wc[:wcl.size] = wcl
    wc[m - wcl.size:] = wcl[::-1]
    tl = np.array([float(v) for v in STAG_CLOSURE["t"]])
    tL = np.zeros(m)
    tL[:tl.size] = tl
    tR = tL[::-1].copy()
    eL, eR = np.zeros(m + 1), np.zeros(m + 1)
    eL[0], eR[-1] = 1.0, 1.0
    Dm = (-(Dp.T * wf[None, :]) + np.outer(tR, eR) - np.outer(tL, eL)) / wc[:, None]
    return StaggeredPair(Dp, Dm, wc, wf, tL, tR)


# ---------------------------------------------------------------------------------------------------------------------
# Angular and eta operators (dense, host)
# ---------------------------------------------------------------------------------------------------------------------
def stag_fourier(N: int, delta: float) -> np.ndarray:
    """Spectral derivative ``(N, N)`` from the nodes ``theta_j = delta + 2 pi j / N`` to the half nodes ``theta_j + pi / N``.

    The Nyquist mode is the cosine ``cos(N / 2 (theta - delta))``; unlike the node derivative (zero on it), its half-node
    derivative is nonzero.
    """
    _B, Binv, keys, _om = ring_basis(N, delta)
    th = delta + TWO_PI * np.arange(N) / N
    thh = th + np.pi / N
    cols = []
    for m, typ in keys:
        if typ == "c":
            cols.append(-m * np.sin(m * thh))
        elif typ == "s":
            cols.append(m * np.cos(m * thh))
        else:
            cols.append(-m * np.sin(m * (thh - delta)))
    return np.column_stack(cols) @ Binv


def fourier_half_interp(N: int, delta: float) -> np.ndarray:
    """Trigonometric interpolation ``(N, N)`` from the nodes to the half nodes (the Nyquist mode is dropped)."""
    _B, Binv, keys, _om = ring_basis(N, delta)
    th = delta + TWO_PI * np.arange(N) / N
    thh = th + np.pi / N
    cols = []
    for m, typ in keys:
        cols.append(np.cos(m * thh) if typ == "c" else (np.sin(m * thh) if typ == "s" else 0.0 * thh))
    return np.column_stack(cols) @ Binv


def stag_eta(n_eta: int, deta: float) -> np.ndarray:
    """Periodic fourth-order staggered derivative ``(n_eta, n_eta)``: plane ``k`` to the half plane ``k + 1/2``."""
    G = np.zeros((n_eta, n_eta))
    for k in range(n_eta):
        for o, c in zip(range(-1, 3), _DINT):
            G[k, (k + o) % n_eta] += c / deta
    return G


def radial_face_interp(m: int) -> np.ndarray:
    """``(m + 1, m)`` 4-point Lagrange interpolation from the centres ``i + 1/2`` to the faces ``k`` (one-sided at the ends)."""
    c = np.arange(m) + 0.5
    I = np.zeros((m + 1, m))
    for k in range(m + 1):
        j0 = min(max(k - 2, 0), m - 4)
        idx = np.arange(j0, j0 + 4)
        for ja in idx:
            w = 1.0
            for jb in idx:
                if jb != ja:
                    w *= (k - c[jb]) / (c[ja] - c[jb])
            I[k, ja] = w
    return I


def core_frame_A(A_log: np.ndarray, J_log: np.ndarray, u: np.ndarray, theta: np.ndarray):
    """Logical ``A = J P_perp`` ``(..., 3, 3)`` and ``J`` at the core nodes to the block frame ``(x, y, eta)``.

    ``A_xy = Lam A Lam^T / u`` and ``J_xy = J / u`` with ``Lam = d(x, y, eta) / d(u, theta, eta)``.
    """
    u = np.asarray(u, dtype=np.float64)
    c, s = np.cos(theta), np.sin(theta)
    Lam = np.zeros(u.shape + (3, 3))
    Lam[..., 0, 0], Lam[..., 0, 1] = c, -u * s
    Lam[..., 1, 0], Lam[..., 1, 1] = s, u * c
    Lam[..., 2, 2] = 1.0
    A = np.einsum("nai,...nij,nbj->...nab", Lam, A_log, Lam) / u[:, None, None]
    return A, J_log / u


def face_coefficients(A, Iu: np.ndarray, Ith: np.ndarray, Ie: np.ndarray, Nc: int, m: int, N: int):
    """Face coefficients ``(auu (E, m + 1, N), att (E, m, N), aee (E, P))`` of the diagonal entries of ``A (E, P, 3, 3)``.

    ``auu`` interpolates ``A_uu`` of the ring nodes radially (``Iu``), ``att`` ``A_theta theta`` to the half nodes of every ring
    (``Ith``), and ``aee`` ``A_eta eta`` of every node to the half planes (``Ie``, a periodic ``(E, E)`` matrix).
    """
    A = np.asarray(A)
    E = A.shape[0]
    ring = A[:, Nc:]
    auu = np.einsum("km,emj->ekj", Iu, ring[..., 0, 0].reshape(E, m, N))
    att = np.einsum("hj,emj->emh", Ith, ring[..., 1, 1].reshape(E, m, N))
    aee = np.einsum("kl,lp->kp", Ie, A[..., 2, 2])
    return auu, att, aee


# ---------------------------------------------------------------------------------------------------------------------
# Metric input
# ---------------------------------------------------------------------------------------------------------------------
class LaplacianMetric(NamedTuple):
    """Nodal metric data of the Laplacian in the logical frame ``(u, theta, eta)``.

    ``A_log (E, P, 3, 3)`` is ``J P_perp`` (``J`` the logical jacobian determinant; the sign is dropped), ``J_log (E, P)`` and
    ``ginv_u (E, P, 3)`` the first row ``g^{u j}`` of the inverse metric (needed for the physical-normal Neumann data;
    ``None`` disables it). Core nodes carry the logical values at their polar ``(u, theta)`` (the plan builder applies the
    frame transform).
    """

    A_log: np.ndarray
    J_log: np.ndarray
    ginv_u: np.ndarray | None = None


def nodal_laplacian_metric_from_callable(layout: NodalLayout, fn: Callable) -> LaplacianMetric:
    """Evaluate ``fn(points (E*P, 3)) -> (A_log (Q, 3, 3), J (Q,), ginv_u (Q, 3) | None)`` at every node (index ``k * P + p``)."""
    E, P = layout.n_eta, layout.P
    eta = (np.arange(E) + 0.5) * layout.deta
    pts = np.stack([np.broadcast_to(layout.node_u, (E, P)), np.broadcast_to(layout.node_theta, (E, P)),
                    np.broadcast_to(eta[:, None], (E, P))], axis=-1).reshape(-1, 3)
    A, J, gu = fn(pts)
    return LaplacianMetric(np.asarray(A, dtype=np.float64).reshape(E, P, 3, 3), np.asarray(J, dtype=np.float64).reshape(E, P),
                           None if gu is None else np.asarray(gu, dtype=np.float64).reshape(E, P, 3))


# ---------------------------------------------------------------------------------------------------------------------
# Penalty rule
# ---------------------------------------------------------------------------------------------------------------------
#: ``gamma`` of the trace-constant rule ``C_X = gamma_X * ratio_X``: 1.25 times the largest ratio of the exact constant
#: ``C_X = sup |F_X f|^2_Omega / E_block(f)`` to the local ratio measured on the analytic testbed (P09 Laplacian study).
PENALTY_GAMMA = {"r": 1.0, "w": 1.1, "c": 0.11}


def penalty_rule(auu_f: np.ndarray, A_core: np.ndarray, wf0: float, du: float, p: int, R_c: float,
                 tau_mult: float = 1.0, gamma: dict | None = None):
    """``(tau, tau_w, C)`` of the trace-constant rule ``tau = C_c / 4 + C_r / 2``, ``tau_w = 2 C_w``.

    ``C_r = gamma_r max|a_uu,f(face 0)| / (wf0 du)``, ``C_w = gamma_w max|a_uu,f(wall face)| / (wf0 du)`` and
    ``C_c = gamma_c max|eig A_xy| p^2 / R_c`` with the largest absolute eigenvalue of the core ``xy`` block.
    """
    g = PENALTY_GAMMA if gamma is None else gamma
    ar, aw = np.abs(auu_f[:, 0]).max(), np.abs(auu_f[:, -1]).max()
    axy = A_core[..., :2, :2]
    acm = np.abs(np.linalg.eigvalsh(0.5 * (axy + np.swapaxes(axy, -1, -2)))).max()
    C = {"C_c": g["c"] * acm * p ** 2 / R_c, "C_r": g["r"] * ar / (wf0 * du), "C_w": g["w"] * aw / (wf0 * du)}
    return tau_mult * (C["C_c"] / 4 + C["C_r"] / 2), tau_mult * 2 * C["C_w"], C


# ---------------------------------------------------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LaplacianStructure:
    """Static, hashable description of a :class:`LaplacianPlan` (one core, one ring level, one outer wall)."""

    n: int
    n_eta: int
    P: int
    Nc: int
    m: int
    N: int
    i0: int
    deta: float
    du: float
    R_c: float
    p: int


@dataclass(frozen=True)
class LaplacianPlan:
    """Pytree of the arrays the JAX Laplacian needs (host float64 NumPy arrays; ``structure`` is a meta field).

    ``Hp (E, P)`` is the per-plane norm ``wxy |J_block|``; ``A (E, P, 3, 3)`` the block-frame coefficient ``J P_perp``;
    ``auu_f, att_h, aee_h`` its face coefficients (unit polarization coefficient); ``Du`` the ``t3`` collocated radial
    derivative and ``Dp`` the staggered pair (both with ``1 / du``), ``wf`` the face weights times ``du``, ``Iu`` the radial
    face interpolation, ``t_in, t_out`` the cubic ring traces at the core and wall sides (zero beyond four rows),
    ``core_D1, core_D2, core_IP`` the core derivatives and ``I - Pi``, ``Rx, cos_g, sin_g`` the core trace and the
    interface directions, ``kappa`` the shell rate at ``c_kappa = 1`` and ``tau, tau_w`` the SIPG and Nitsche penalties.
    ``wall_alpha, wall_beta_th, wall_beta_eta (E, N)`` convert a physical-normal Neumann datum ``g_n`` to the conormal flux
    ``alpha g_n + beta_th D_theta T_w f + beta_eta D_eta T_w f`` (``None`` without ``ginv_u``).
    """

    Hp: np.ndarray
    wxy: np.ndarray
    A: np.ndarray
    auu_f: np.ndarray
    att_h: np.ndarray
    aee_h: np.ndarray
    Du: np.ndarray
    Dp: np.ndarray
    wf: np.ndarray
    Iu: np.ndarray
    t_in: np.ndarray
    t_out: np.ndarray
    core_D1: np.ndarray
    core_D2: np.ndarray
    core_IP: np.ndarray
    Rx: np.ndarray
    cos_g: np.ndarray
    sin_g: np.ndarray
    kappa: np.ndarray
    tau: np.ndarray
    tau_w: np.ndarray
    wall_alpha: np.ndarray | None
    wall_beta_th: np.ndarray | None
    wall_beta_eta: np.ndarray | None
    structure: LaplacianStructure


jax.tree_util.register_dataclass(
    LaplacianPlan,
    data_fields=[f.name for f in dc_fields(LaplacianPlan) if f.name != "structure"],
    meta_fields=["structure"],
)


def plane_keys(st: LaplacianStructure) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ring, plane, theta)`` keys of the ``E * P`` flat nodes ``k * P + p`` for the plane-block preconditioner.

    Ring nodes carry their absolute ring ``i0 + a`` and angular index ``j``; core nodes ring ``-1`` and their in-block index.
    """
    ring = np.concatenate([np.full(st.Nc, -1), np.repeat(np.arange(st.i0, st.i0 + st.m), st.N)])
    theta = np.concatenate([np.arange(st.Nc), np.tile(np.arange(st.N), st.m)])
    return (np.tile(ring, st.n_eta).astype(np.int64), np.repeat(np.arange(st.n_eta), st.P).astype(np.int64),
            np.tile(theta, st.n_eta).astype(np.int64))


def _check_layout(layout: NodalLayout) -> tuple[CoreBlock, RingLevelBlock]:
    if layout.inner != "core" or len(layout.levels) != 1 or len(layout.blocks) != 2:
        raise NotImplementedError("the nodal Laplacian supports the family-A layout (one core, one ring level)")
    core, ring = layout.blocks
    if not isinstance(core, CoreBlock) or not isinstance(ring, RingLevelBlock):
        raise NotImplementedError("the nodal Laplacian needs a Zernike core followed by a ring level")
    if ring.N != core.N_g or len(layout.walls) != 1:
        raise NotImplementedError("the ring level must have N = n nodes (matching the core trace) and one outer wall")
    return core, ring


def build_laplacian_plan(layout: NodalLayout, metric: LaplacianMetric, *, tau_mult: float = 1.0,
                         gamma: dict | None = None) -> LaplacianPlan:
    """Build the plan from a family-A layout and the nodal Laplacian metric (see the module docstring)."""
    core, ring = _check_layout(layout)
    E, P, Nc, m, N = layout.n_eta, layout.P, core.n_nodes, ring.m, ring.N
    A = np.array(metric.A_log, dtype=np.float64)
    J = np.abs(np.asarray(metric.J_log, dtype=np.float64))
    if A.shape != (E, P, 3, 3) or J.shape != (E, P):
        raise ValueError(f"metric needs A_log {(E, P, 3, 3)} and J_log {(E, P)}, got {A.shape} and {J.shape}")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(J)) and np.all(J > 0.0)):
        raise ValueError("the Laplacian metric must be finite with a nonzero jacobian at every node")
    Jb = J.copy()
    A[:, :Nc], Jb[:, :Nc] = core_frame_A(A[:, :Nc], J[:, :Nc], core.u, core.theta)
    du, deta = layout.du, layout.deta
    rb = radial_block(m, "t3")
    sp_ = staggered_pair(m)
    Iu = radial_face_interp(m)
    Ith = fourier_half_interp(N, layout.delta)
    Ie = fourier_half_interp(E, 0.0)
    auu, att, aee = face_coefficients(A, Iu, Ith, Ie, Nc, m, N)
    # shell rate and penalties
    axy = A[:, :Nc, :2, :2]
    a_core = np.linalg.eigvalsh(0.5 * (axy + np.swapaxes(axy, -1, -2)))[..., -1]
    kappa = a_core * (core.p / core.R_c) ** 2
    tau, tau_w, _C = penalty_rule(auu, A[:, :Nc], float(sp_.wf[0]), du, core.p, core.R_c, tau_mult, gamma)
    # wall conversion: traces of A^{u j} and g^{u j} at the wall (cubic extrapolation of the last rings)
    alpha = beta_th = beta_eta = None
    if metric.ginv_u is not None:
        gu = np.asarray(metric.ginv_u, dtype=np.float64)
        if gu.shape != (E, P, 3):
            raise ValueError(f"ginv_u must have shape {(E, P, 3)}, got {gu.shape}")
        tw = rb.tR
        Aw = np.einsum("r,erjc->ejc", tw, A[:, Nc:, 0, :].reshape(E, m, N, 3))
        gw = np.einsum("r,erjc->ejc", tw, gu[:, Nc:].reshape(E, m, N, 3))
        alpha = Aw[..., 0] / np.sqrt(gw[..., 0])
        beta_th = Aw[..., 1] - Aw[..., 0] * gw[..., 1] / gw[..., 0]
        beta_eta = Aw[..., 2] - Aw[..., 0] * gw[..., 2] / gw[..., 0]
    struct = LaplacianStructure(n=layout.n, n_eta=E, P=P, Nc=Nc, m=m, N=N, i0=ring.i0, deta=float(deta), du=float(du),
                                R_c=float(core.R_c), p=int(core.p))
    return LaplacianPlan(
        Hp=layout.wxy[None, :] * Jb, wxy=np.asarray(layout.wxy, dtype=np.float64), A=A, auu_f=auu, att_h=att, aee_h=aee,
        Du=rb.D_unit / du, Dp=sp_.Dp / du, wf=sp_.wf * du, Iu=Iu, t_in=rb.tL.copy(), t_out=rb.tR.copy(),
        core_D1=np.asarray(core.D1), core_D2=np.asarray(core.D2), core_IP=np.eye(Nc) - core.Pi,
        Rx=np.asarray(core.Rx), cos_g=np.cos(core.theta_g), sin_g=np.sin(core.theta_g), kappa=kappa,
        tau=np.asarray(tau, dtype=np.float64), tau_w=np.asarray(tau_w, dtype=np.float64),
        wall_alpha=alpha, wall_beta_th=beta_th, wall_beta_eta=beta_eta, structure=struct)
