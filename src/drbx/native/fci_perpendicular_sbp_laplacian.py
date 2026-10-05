"""JAX apply of the nodal SBP perpendicular Laplacian (P07 diffusion and polarization), narrow face-flux energy form.

``L f = -H^-1 (M f - b(data))`` with ``H = Hp * deta`` and ``v^T M f = a(f, v)``; the discretisation and its host builders
are described in :mod:`drbx.geometry.sbp_laplacian`. The :class:`~drbx.geometry.sbp_laplacian.LaplacianPlan` is an
ordinary (jit) argument; the polarization coefficient ``coeff (E, P)``, the shell rate ``c_kappa`` and the boundary data are
traced, only the per-field ``kinds`` (``"dirichlet"`` / ``"neumann"``) and the Neumann datum type are static.

Energy-form structure. Every term of ``M`` is ``G^T Q G`` for a linear feature map ``G`` (derivatives, face derivatives,
traces and flux traces of ``f``) and a pointwise Gram ``Q``. :func:`laplacian_form` evaluates the features and their
Gram weights and applies the exact transpose of ``G`` by ``jax.vjp``, so the symmetry of ``M`` holds by construction and
the boundary data enter as cotangents on the trace features:

- core | ring interface: SIPG, ``Om [f] {F(v)} + Om {F(f)} [v] + tau Om [f] [v]``;
- wall, Dirichlet (Nitsche): ``-Om T v F f - Om F v T f + tau_w Om T v T f`` with data ``g`` replacing ``T f`` in the
  data-carrying slots; Neumann: ``-Om T v g_conormal``;
- Neumann datum: ``bcd.conormal`` is the conormal flux ``(A grad f)^u`` per ``d theta d eta`` used directly;
  ``bcd.normal_derivative`` is the physical-normal derivative ``n . grad f``, converted through the wall trace
  ``T_w f``: ``(A grad f)^u = alpha g_n + beta_theta D_theta T_w f + beta_eta D_eta T_w f`` with centred skew-symmetric
  tangential derivatives, so the operator then carries the (non-symmetric) oblique term.

Arrays are ``(E, P)`` or ``(E, P, F)``; wall data ``(E, N, F)`` (or ``(E, N)`` for one field), one entry per wall as a tuple
like :class:`~drbx.native.fci_perpendicular_sbp_boundary.SatBoundaryData`. ``coeff`` multiplies ``A`` pointwise
(``n``, ``n / B^2``, ...); its face values are interpolated from the nodal product ``coeff * A`` at call time (for a face
family the plan evaluated at the faces, ``plan.structure.evaluated``: the evaluated tensor times the interpolated ``coeff``), the
penalties scale with the largest coefficient on their surface and the shell rate with the local value (``coeff >= 0``).
Single device (the ``eta`` operators wrap periodically), except the halo-extended forms at the end of the module
(:func:`laplacian_form_ext`, :func:`laplacian_action_ext`: unit coefficient only, owned planes from extended inputs) used by the
eta-sharded Laplacian and CG in :mod:`drbx.native.fci_perpendicular_sbp_sharding`.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry.sbp_laplacian import LaplacianPlan
from drbx.native.fci_perpendicular_reconstruction_state import DIRICHLET, normalize_kinds
from drbx.native.fci_perpendicular_sbp_ops import _ring_d1, _ring_d2, d_eta, extend_periodic

TWO_PI = 2.0 * np.pi
N_TRACE = 4                       # rows of the cubic ring trace

__all__ = ["LaplacianBoundaryData", "laplacian_form", "laplacian_action", "laplacian_linear", "positive_action",
           "conormal_from_normal", "laplacian_form_jit", "laplacian_action_jit", "laplacian_linear_jit",
           "F_HALO", "PLAN_HALO", "PLANE_FIELDS", "window_laplacian_plan", "extend_laplacian_plan", "laplacian_form_ext",
           "laplacian_action_ext"]

#: eta halo of the field ``f`` and of the plan arrays / wall data in the halo-extended forms (:func:`laplacian_form_ext`).
#: The features read ``f`` at ``k - 2 .. k + 2`` (``D_eta``) and ``k - 1 .. k + 2`` (staggered), the transpose collects the
#: cotangents of the planes ``k - 2 .. k + 2`` (``PLAN_HALO``), whose features read ``f`` out to ``k +- 4`` (``F_HALO``).
PLAN_HALO = 2
F_HALO = 4


class LaplacianBoundaryData(NamedTuple):
    """Wall data, each ``None`` or a tuple with one ``(E, N, F)`` array per wall.

    ``value``: Dirichlet values; ``conormal``: conormal flux ``(A grad f)^u``; ``normal_derivative``: physical-normal
    derivative ``n . grad f`` (converted to the conormal flux in the operator). ``conormal`` and ``normal_derivative``
    are exclusive.
    """

    value: tuple | None = None
    normal_derivative: tuple | None = None
    conormal: tuple | None = None


# ---------------------------------------------------------------------------
# 1-D pieces
# ---------------------------------------------------------------------------
def _apply_axis(x, mult, axis: int, n: int):
    """``irfft(mult * rfft(x, axis), n)`` with ``mult`` a host vector of the ``n // 2 + 1`` frequencies."""
    xh = jnp.fft.rfft(x, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = mult.size
    return jnp.fft.irfft(jnp.asarray(mult).reshape(shape) * xh, n=n, axis=axis)


def _freq(n: int) -> np.ndarray:
    return np.arange(n // 2 + 1, dtype=float)


def _half_derivative(x, axis: int, n: int):
    """Spectral derivative from nodes to half nodes (shift ``pi / n``); the Nyquist mode is a cosine."""
    k = _freq(n)
    return _apply_axis(x, 1j * k * np.exp(1j * k * np.pi / n), axis, n)


def _half_interp(x, axis: int, n: int):
    """Trigonometric interpolation to the half nodes with the Nyquist mode dropped."""
    k = _freq(n)
    mult = np.exp(1j * k * np.pi / n)
    if n % 2 == 0:
        mult[-1] = 0.0
    return _apply_axis(x, mult, axis, n)


def _node_derivative(x, axis: int, n: int):
    """Spectral derivative on the nodes (Nyquist mode zeroed)."""
    k = _freq(n)
    if n % 2 == 0:
        k[-1] = 0.0
    return _apply_axis(x, 1j * k, axis, n)


def _stag_d1(Dp, gl, m: int):
    """Staggered radial derivative of ``gl (E, m, N, F)`` to the ``m + 1`` faces with the banded ``Dp (m + 1, m)``."""
    b, w = 4, 6
    left = jnp.einsum("ab,ebj...->eaj...", Dp[:b, :w], gl[:, :w])
    right = jnp.einsum("ab,ebj...->eaj...", Dp[m + 1 - b:, m - w:], gl[:, m - w:])
    n_mid = m + 1 - 2 * b
    mid = sum(Dp[b, b - 2 + o] * gl[:, b - 2 + o:b - 2 + o + n_mid] for o in range(4))
    return jnp.concatenate([left, mid, right], axis=1)


def _stag_eta(x, deta):
    """Staggered ``(f_{k-1} - 27 f_k + 27 f_{k+1} - f_{k+2}) / (24 deta)`` along the leading axis (periodic)."""
    return (jnp.roll(x, 1, axis=0) - 27.0 * x + 27.0 * jnp.roll(x, -1, axis=0) - jnp.roll(x, -2, axis=0)) / (24.0 * deta)


def _eta_derivative(x, deta):
    """Central fourth-order ``eta`` derivative along the leading axis (periodic)."""
    return d_eta(extend_periodic(x, 2), deta, 2)


def _bc(a, like):
    return jnp.reshape(a, a.shape + (1,) * (like.ndim - a.ndim))


# ---------------------------------------------------------------------------
# Coefficients (nodal A, face coefficients, penalty scalings) for a pointwise polarization coefficient
# ---------------------------------------------------------------------------
class _Coef(NamedTuple):
    A: object            # (E, P, 3, 3)
    auu: object          # (E, m + 1, N)
    att: object          # (E, m, N)
    aee: object          # (E, P)
    c_core: object       # (E, Nc) or 1.0
    c_wall: object       # (E, N) wall trace of the coefficient or 1.0
    s_in: object         # scalar scaling of tau
    s_wall: object       # scalar scaling of tau_w


def _coefficients(lp: LaplacianPlan, coeff) -> _Coef:
    if coeff is None:
        return _Coef(lp.A, lp.auu_f, lp.att_h, lp.aee_h, 1.0, 1.0, 1.0, 1.0)
    st = lp.structure
    E, Nc, m, N = st.n_eta, st.Nc, st.m, st.N
    coeff = jnp.asarray(coeff)
    A = coeff[..., None, None] * lp.A
    ring = A[:, Nc:]
    cr = coeff[:, Nc:].reshape(E, m, N)
    ev = st.evaluated                      # static: families whose face tensor was evaluated at the faces by the plan
    if "uu" in ev:
        auu = lp.auu_f * jnp.einsum("km,emj->ekj", lp.Iu, cr)
    else:
        auu = jnp.einsum("km,emj->ekj", lp.Iu, ring[..., 0, 0].reshape(E, m, N))
    att = lp.att_h * _half_interp(cr, 2, N) if "tt" in ev else _half_interp(ring[..., 1, 1].reshape(E, m, N), 2, N)
    aee = lp.aee_h * _half_interp(coeff, 0, E) if "ee" in ev else _half_interp(A[..., 2, 2], 0, E)
    c_in = jnp.einsum("r,erj->ej", lp.t_in[:N_TRACE], cr[:, :N_TRACE])
    c_wall = jnp.einsum("r,erj->ej", lp.t_out[-N_TRACE:], cr[:, -N_TRACE:])
    c_core_tr = jnp.einsum("jp,ep->ej", lp.Rx, coeff[:, :Nc])
    s_in = jnp.maximum(jnp.max(jnp.abs(c_in)), jnp.max(jnp.abs(c_core_tr)))
    return _Coef(A, auu, att, aee, coeff[:, :Nc], c_wall, s_in, jnp.max(jnp.abs(c_wall)))


# ---------------------------------------------------------------------------
# Feature map G
# ---------------------------------------------------------------------------
def _features(lp: LaplacianPlan, f, Cf: _Coef):
    """Linear features of ``f (E, P, F)``: collocated derivatives, face derivatives, traces and flux traces."""
    st = lp.structure
    E, Nc, m, N = st.n_eta, st.Nc, st.m, st.N
    F = f.shape[-1]
    core = f[:, :Nc]
    rg = f[:, Nc:].reshape(E, m, N, F)
    d1c = jnp.einsum("pq,eqf->epf", lp.core_D1, core)
    d2c = jnp.einsum("pq,eqf->epf", lp.core_D2, core)
    d1 = jnp.concatenate([d1c, _ring_d1(lp.Du, rg, m).reshape(E, m * N, F)], axis=1)
    d2 = jnp.concatenate([d2c, _ring_d2(rg, N).reshape(E, m * N, F)], axis=1)
    d3 = _eta_derivative(f, st.deta)
    fu = _stag_d1(lp.Dp, rg, m)
    ft = _half_derivative(rg, 2, N)
    fe = _stag_eta(f, st.deta)
    A = Cf.A
    # traces
    Tc = jnp.einsum("jp,epf->ejf", lp.Rx, core)
    Tr = jnp.einsum("r,erjf->ejf", lp.t_in[:N_TRACE], rg[:, :N_TRACE])
    Tw = jnp.einsum("r,erjf->ejf", lp.t_out[-N_TRACE:], rg[:, -N_TRACE:])
    # fluxes (+u) per radian: ring = face flux of the staggered uu part + the collocated cross terms
    Fcol = (_bc(A[:, Nc:, 0, 1], d2) * d2[:, Nc:] + _bc(A[:, Nc:, 0, 2], d3) * d3[:, Nc:]).reshape(E, m, N, F)
    Fr = _bc(Cf.auu[:, 0], Tr) * fu[:, 0] + jnp.einsum("r,erjf->ejf", lp.t_in[:N_TRACE], Fcol[:, :N_TRACE])
    Fw = _bc(Cf.auu[:, m], Tw) * fu[:, m] + jnp.einsum("r,erjf->ejf", lp.t_out[-N_TRACE:], Fcol[:, -N_TRACE:])
    Ax = (_bc(A[:, :Nc, 0, 0], d1c) * d1c + _bc(A[:, :Nc, 0, 1], d2c) * d2c + _bc(A[:, :Nc, 0, 2], d3[:, :Nc]) * d3[:, :Nc])
    Ay = (_bc(A[:, :Nc, 1, 0], d1c) * d1c + _bc(A[:, :Nc, 1, 1], d2c) * d2c + _bc(A[:, :Nc, 1, 2], d3[:, :Nc]) * d3[:, :Nc])
    cs, sn = lp.cos_g[None, :, None], lp.sin_g[None, :, None]
    Fc = st.R_c * (cs * jnp.einsum("jp,epf->ejf", lp.Rx, Ax) + sn * jnp.einsum("jp,epf->ejf", lp.Rx, Ay))
    return {"d1": d1, "d2": d2, "d3": d3, "fu": fu, "ft": ft, "fe": fe, "sh": jnp.einsum("pq,eqf->epf", lp.core_IP, core),
            "Tc": Tc, "Tr": Tr, "Tw": Tw, "Fc": Fc, "Fr": Fr, "Fw": Fw}


def _colloc_mask(Nc: int, P: int) -> np.ndarray:
    """``(P, 3, 3)`` mask of the collocated ``D_i^T W A_ij D_j`` terms: off-diagonal on rings, all but ``eta eta`` in the core."""
    mask = np.ones((P, 3, 3)) - np.eye(3)[None]
    mask[:Nc] = 1.0
    mask[:Nc, 2, 2] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Boundary data helpers
# ---------------------------------------------------------------------------
def _wall_array(entry, F: int, like):
    """First (only) wall's data as ``(E, N, F)``; ``None`` gives zeros like ``like``."""
    if entry is None:
        return jnp.zeros_like(like)
    a = jnp.asarray(entry[0] if isinstance(entry, (tuple, list)) else entry, dtype=like.dtype)
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[-1] == 1 and F > 1:
        a = jnp.broadcast_to(a, like.shape)
    if a.shape != like.shape:
        raise ValueError(f"wall data must have shape {like.shape}, got {a.shape}")
    return a


def _neumann_mode(bcd) -> str:
    conormal = getattr(bcd, "conormal", None)
    normal = getattr(bcd, "normal_derivative", None)
    if conormal is not None and normal is not None:
        raise ValueError("pass either the conormal flux or the physical-normal derivative, not both")
    return "physical" if normal is not None else "conormal"


# ---------------------------------------------------------------------------
# Physical-normal to conormal conversion
# ---------------------------------------------------------------------------
def _conormal_from_trace(lp: LaplacianPlan, Tw, gn, c_wall=1.0):
    """Conormal flux ``(A grad f)^u`` on the wall grid from the wall trace ``Tw (E, N, F)`` and ``g_n = n . grad f``.

    ``d_u f = (|grad u| g_n - g^{u theta} d_theta f - g^{u eta} d_eta f) / g^{uu}`` with the tangential derivatives of the
    trace (spectral in ``theta``, fourth-order centred in ``eta``), then
    ``A^{uu} d_u f + A^{u theta} d_theta f + A^{u eta} d_eta f = alpha g_n + beta_theta d_theta f + beta_eta d_eta f``.
    """
    if lp.wall_alpha is None:
        raise ValueError("the plan has no wall metric (ginv_u) for physical-normal Neumann data")
    st = lp.structure
    return (_bc(c_wall * lp.wall_alpha, Tw) * gn + _bc(c_wall * lp.wall_beta_th, Tw) * _node_derivative(Tw, 1, st.N)
            + _bc(c_wall * lp.wall_beta_eta, Tw) * _eta_derivative(Tw, st.deta))


def conormal_from_normal(lp: LaplacianPlan, f, g_n, coeff=None):
    """Conormal flux ``(coeff A grad f)^u`` ``(E, N[, F])`` at the wall grid from the nodal ``f (E, P[, F])`` and the
    physical-normal derivative ``g_n (E, N[, F])`` (the conversion used by ``bcd.normal_derivative``)."""
    f = jnp.asarray(f)
    scalar = f.ndim == 2
    f3 = f[..., None] if scalar else f
    st = lp.structure
    g3 = jnp.asarray(g_n)
    g3 = g3[..., None] if g3.ndim == 2 else g3
    Cf = _coefficients(lp, coeff)
    Tw = jnp.einsum("r,erjf->ejf", lp.t_out[-N_TRACE:], f3[:, st.Nc:].reshape(st.n_eta, st.m, st.N, -1)[:, -N_TRACE:])
    out = _conormal_from_trace(lp, Tw, g3, Cf.c_wall)
    return out[..., 0] if scalar else out


# ---------------------------------------------------------------------------
# The energy form
# ---------------------------------------------------------------------------
def _gram(lp: LaplacianPlan, feats: dict, Cf: _Coef, c_kappa, kinds, mode: str, gD, gN):
    """Cotangents ``Q G f - q_data`` on the features."""
    st = lp.structure
    E, P, Nc, m, N = st.n_eta, st.P, st.Nc, st.m, st.N
    deta = st.deta
    W = lp.wxy * deta                                           # (P,)
    om = TWO_PI / N * deta
    F = feats["d1"].shape[-1]
    like = feats["Tw"]
    cot = {}
    # staggered face terms
    wfw = (lp.wf * (TWO_PI / N * deta))[None, :, None]
    cot["fu"] = _bc(wfw * Cf.auu, feats["fu"]) * feats["fu"]
    cot["ft"] = _bc(W[Nc:].reshape(m, N)[None] * Cf.att, feats["ft"]) * feats["ft"]
    cot["fe"] = _bc(W[None, :] * Cf.aee, feats["fe"]) * feats["fe"]
    # collocated cross terms
    Cm = (W[None, :, None, None] * Cf.A) * _colloc_mask(Nc, P)[None]
    D = jnp.stack([feats["d1"], feats["d2"], feats["d3"]], axis=2)          # (E, P, 3, F)
    cD = jnp.einsum("epij,epjf->epif", Cm, D)
    cot["d1"], cot["d2"], cot["d3"] = cD[:, :, 0], cD[:, :, 1], cD[:, :, 2]
    # core shell penalty
    kap = c_kappa * lp.kappa * Cf.c_core
    cot["sh"] = _bc(W[None, :Nc] * kap, feats["sh"]) * feats["sh"]
    # core | ring interface (SIPG)
    jump = feats["Tr"] - feats["Tc"]
    cot_j = om * (lp.tau * Cf.s_in * jump + 0.5 * (feats["Fc"] + feats["Fr"]))
    cot["Tr"], cot["Tc"] = cot_j, -cot_j
    cot["Fc"] = cot["Fr"] = 0.5 * om * jump
    # wall
    isD = jnp.asarray([1.0 if k == DIRICHLET else 0.0 for k in kinds], dtype=like.dtype)
    isN = 1.0 - isD
    aT = feats["Tw"] - gD
    cot_T = isD * om * (lp.tau_w * Cf.s_wall * aT - feats["Fw"])
    cot["Fw"] = -isD * om * aT
    g = _conormal_from_trace(lp, feats["Tw"], gN, Cf.c_wall) if mode == "physical" else gN
    cot["Tw"] = cot_T - isN * om * g
    return cot


def laplacian_form(lp: LaplacianPlan, f, bcd=None, kinds="dirichlet", coeff=None, c_kappa=1.0, *, neumann_mode=None):
    """``M f - b(data)`` of shape ``f.shape`` (the numerator of ``L f = -H^-1 (M f - b)``); see the module docstring."""
    f = jnp.asarray(f)
    scalar = f.ndim == 2
    f3 = f[..., None] if scalar else f
    F = f3.shape[-1]
    kinds = normalize_kinds(kinds, F)
    mode = neumann_mode or _neumann_mode(bcd)
    st = lp.structure
    like = jnp.zeros((st.n_eta, st.N, F), f3.dtype)
    gD = _wall_array(getattr(bcd, "value", None), F, like)
    gN = _wall_array(getattr(bcd, "conormal", None) if mode == "conormal" else getattr(bcd, "normal_derivative", None),
                     F, like)
    Cf = _coefficients(lp, coeff)
    feats, vjp = jax.vjp(lambda x: _features(lp, x, Cf), f3)
    out = vjp(_gram(lp, feats, Cf, c_kappa, kinds, mode, gD, gN))[0]
    return out[..., 0] if scalar else out


def _H(lp: LaplacianPlan, like):
    return _bc(lp.Hp * lp.structure.deta, like)


def laplacian_action(lp: LaplacianPlan, f, bcd=None, kinds="dirichlet", coeff=None, c_kappa=1.0, *, neumann_mode=None):
    """``L f = -H^-1 (M f - b)`` (``f (E, P[, F])``; the perpendicular Laplacian ``div(coeff A grad f) / J`` per volume)."""
    out = laplacian_form(lp, f, bcd, kinds, coeff, c_kappa, neumann_mode=neumann_mode)
    return -out / _H(lp, out)


def positive_action(lp: LaplacianPlan, f, bcd=None, kinds="dirichlet", coeff=None, c_kappa=1.0, *, neumann_mode=None):
    """The positive polarization action ``-L f = H^-1 (M f - b)`` (the P07 ``A(q) = -L_perp(q)`` convention)."""
    return -laplacian_action(lp, f, bcd, kinds, coeff, c_kappa, neumann_mode=neumann_mode)


def laplacian_linear(lp: LaplacianPlan, f, kinds="dirichlet", coeff=None, c_kappa=1.0, *, neumann_mode="conormal"):
    """The linear part ``-H^-1 M f`` (zero wall data): the map for JVPs, assembly and solves.

    ``neumann_mode="physical"`` includes the oblique tangential-derivative term of the physical-normal conversion.
    """
    out = laplacian_form(lp, f, None, kinds, coeff, c_kappa, neumann_mode=neumann_mode)
    return -out / _H(lp, out)


laplacian_form_jit = jax.jit(laplacian_form, static_argnames=("kinds", "neumann_mode"))
laplacian_action_jit = jax.jit(laplacian_action, static_argnames=("kinds", "neumann_mode"))
laplacian_linear_jit = jax.jit(laplacian_linear, static_argnames=("kinds", "neumann_mode"))


# ---------------------------------------------------------------------------
# Halo-extended form (eta sharding): owned rows from extended f and plan arrays, no wrap, no global FFT
# ---------------------------------------------------------------------------
#: ``LaplacianPlan`` leaves with a leading eta (plane) axis; the other leaves are operator matrices / scalars (replicated).
PLANE_FIELDS = ("Hp", "A", "auu_f", "att_h", "aee_h", "kappa", "wall_alpha", "wall_beta_th", "wall_beta_eta")


def window_laplacian_plan(lp: LaplacianPlan, index) -> LaplacianPlan:
    """``lp`` with every plane leaf gathered at the plane indices ``index`` (``None`` leaves stay ``None``)."""
    return dataclasses.replace(lp, **{name: getattr(lp, name)[index] for name in PLANE_FIELDS
                                      if getattr(lp, name) is not None})


def extend_laplacian_plan(lp: LaplacianPlan, halo: int = PLAN_HALO) -> LaplacianPlan:
    """Plan arrays of ``E + 2 halo`` planes, the periodic wrap of the plane leaves (single device; ``structure`` unchanged)."""
    E = lp.Hp.shape[0]
    return window_laplacian_plan(lp, np.arange(-halo, E + halo) % E)


def _stag_eta_ext(x_ext, deta, halo: int):
    """``_stag_eta`` on the owned planes of a halo-``halo`` extended array (``halo >= 2``): the same arithmetic, no wrap."""
    n = x_ext.shape[0] - 2 * halo

    def at(offset):
        return x_ext[halo + offset:halo + offset + n]

    return (at(-1) - 27.0 * at(0) + 27.0 * at(1) - at(2)) / (24.0 * deta)


def _features_ext(lp: LaplacianPlan, f_ext, Cf: _Coef):
    """:func:`_features` on the centre ``E_c`` planes (those with plan arrays) of ``f_ext (E_c + 2 h, P, F)``, ``h = F_HALO - PLAN_HALO``.

    The plane-local features are taken from the centre planes; the ``eta`` derivative and the staggered ``eta`` difference
    read the ``h`` extra planes on each side instead of wrapping.
    """
    st = lp.structure
    Nc, m, N = st.Nc, st.m, st.N
    h = F_HALO - PLAN_HALO
    F = f_ext.shape[-1]
    f = f_ext[h:f_ext.shape[0] - h]
    E = f.shape[0]
    core = f[:, :Nc]
    rg = f[:, Nc:].reshape(E, m, N, F)
    d1c = jnp.einsum("pq,eqf->epf", lp.core_D1, core)
    d2c = jnp.einsum("pq,eqf->epf", lp.core_D2, core)
    d1 = jnp.concatenate([d1c, _ring_d1(lp.Du, rg, m).reshape(E, m * N, F)], axis=1)
    d2 = jnp.concatenate([d2c, _ring_d2(rg, N).reshape(E, m * N, F)], axis=1)
    d3 = d_eta(f_ext, st.deta, h)
    fu = _stag_d1(lp.Dp, rg, m)
    ft = _half_derivative(rg, 2, N)
    fe = _stag_eta_ext(f_ext, st.deta, h)
    A = Cf.A
    Tc = jnp.einsum("jp,epf->ejf", lp.Rx, core)
    Tr = jnp.einsum("r,erjf->ejf", lp.t_in[:N_TRACE], rg[:, :N_TRACE])
    Tw = jnp.einsum("r,erjf->ejf", lp.t_out[-N_TRACE:], rg[:, -N_TRACE:])
    Fcol = (_bc(A[:, Nc:, 0, 1], d2) * d2[:, Nc:] + _bc(A[:, Nc:, 0, 2], d3) * d3[:, Nc:]).reshape(E, m, N, F)
    Fr = _bc(Cf.auu[:, 0], Tr) * fu[:, 0] + jnp.einsum("r,erjf->ejf", lp.t_in[:N_TRACE], Fcol[:, :N_TRACE])
    Fw = _bc(Cf.auu[:, m], Tw) * fu[:, m] + jnp.einsum("r,erjf->ejf", lp.t_out[-N_TRACE:], Fcol[:, -N_TRACE:])
    Ax = (_bc(A[:, :Nc, 0, 0], d1c) * d1c + _bc(A[:, :Nc, 0, 1], d2c) * d2c + _bc(A[:, :Nc, 0, 2], d3[:, :Nc]) * d3[:, :Nc])
    Ay = (_bc(A[:, :Nc, 1, 0], d1c) * d1c + _bc(A[:, :Nc, 1, 1], d2c) * d2c + _bc(A[:, :Nc, 1, 2], d3[:, :Nc]) * d3[:, :Nc])
    cs, sn = lp.cos_g[None, :, None], lp.sin_g[None, :, None]
    Fc = st.R_c * (cs * jnp.einsum("jp,epf->ejf", lp.Rx, Ax) + sn * jnp.einsum("jp,epf->ejf", lp.Rx, Ay))
    return {"d1": d1, "d2": d2, "d3": d3, "fu": fu, "ft": ft, "fe": fe, "sh": jnp.einsum("pq,eqf->epf", lp.core_IP, core),
            "Tc": Tc, "Tr": Tr, "Tw": Tw, "Fc": Fc, "Fr": Fr, "Fw": Fw}


def _conormal_from_trace_ext(lp: LaplacianPlan, Tw_ext, gn):
    """:func:`_conormal_from_trace` (unit coefficient) on the centre planes of the wall trace ``Tw_ext`` with ``h`` extra planes."""
    if lp.wall_alpha is None:
        raise ValueError("the plan has no wall metric (ginv_u) for physical-normal Neumann data")
    st = lp.structure
    h = F_HALO - PLAN_HALO
    Tw = Tw_ext[h:Tw_ext.shape[0] - h]
    return (_bc(lp.wall_alpha, Tw) * gn + _bc(lp.wall_beta_th, Tw) * _node_derivative(Tw, 1, st.N)
            + _bc(lp.wall_beta_eta, Tw) * d_eta(Tw_ext, st.deta, h))


def laplacian_form_ext(lp: LaplacianPlan, f_ext, bcd=None, kinds="dirichlet", c_kappa=1.0, *, neumann_mode=None):
    """:func:`laplacian_form` of the owned planes from the halo-extended ``f_ext (p + 2 F_HALO, P[, F])`` (eta sharding).

    ``lp`` carries its plane leaves with ``p + 2 PLAN_HALO`` planes (:func:`extend_laplacian_plan`, or a shard of
    ``shard_laplacian_plan``), and the wall data ``bcd`` (when given) the same ``p + 2 PLAN_HALO`` planes. Returns ``(p, P[, F])``.
    The coefficient is the unit one (``coeff=None``): its face values would need a global FFT along eta. ``structure.n_eta`` is
    not read. The owned rows are those of the single-device form: the features of the planes ``k - 2 .. k + 2`` need ``f`` at
    ``k +- 4`` and the transpose (a ``jax.vjp`` over the extended block, cropped) is exact for rows at least 2 planes inside.
    """
    f_ext = jnp.asarray(f_ext)
    scalar = f_ext.ndim == 2
    f3 = f_ext[..., None] if scalar else f_ext
    F = f3.shape[-1]
    Ec = lp.Hp.shape[0]
    h = F_HALO - PLAN_HALO
    if f3.shape[0] != Ec + 2 * h or Ec <= 2 * PLAN_HALO:
        raise ValueError(f"f_ext needs {Ec + 2 * h} planes (plan arrays carry {Ec}, halo {PLAN_HALO}), got {f3.shape[0]}")
    kinds = normalize_kinds(kinds, F)
    mode = neumann_mode or _neumann_mode(bcd)
    st = lp.structure
    like = jnp.zeros((Ec, st.N, F), f3.dtype)
    gD = _wall_array(getattr(bcd, "value", None), F, like)
    gN = _wall_array(getattr(bcd, "conormal", None) if mode == "conormal" else getattr(bcd, "normal_derivative", None),
                     F, like)
    Cf = _coefficients(lp, None)
    feats, vjp = jax.vjp(lambda x: _features_ext(lp, x, Cf), f3)
    if mode == "physical":
        rg = f3[:, st.Nc:].reshape(f3.shape[0], st.m, st.N, F)
        Tw_ext = jnp.einsum("r,erjf->ejf", lp.t_out[-N_TRACE:], rg[:, -N_TRACE:])
        gN = _conormal_from_trace_ext(lp, Tw_ext, gN)
    out = vjp(_gram(lp, feats, Cf, c_kappa, kinds, "conormal", gD, gN))[0]
    out = out[F_HALO:out.shape[0] - F_HALO]
    return out[..., 0] if scalar else out


def laplacian_action_ext(lp: LaplacianPlan, f_ext, bcd=None, kinds="dirichlet", c_kappa=1.0, *, neumann_mode=None):
    """``L f = -H^-1 (M f - b)`` on the owned planes of the halo-extended ``f_ext`` (see :func:`laplacian_form_ext`)."""
    out = laplacian_form_ext(lp, f_ext, bcd, kinds, c_kappa, neumann_mode=neumann_mode)
    H = lp.Hp[PLAN_HALO:lp.Hp.shape[0] - PLAN_HALO] * lp.structure.deta
    return -out / _bc(H, out)
