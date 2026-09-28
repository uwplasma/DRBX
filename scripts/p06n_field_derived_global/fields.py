"""P06N field-derived catalogue: reuses P05N's (and, through it, P07N's)
analytic fields verbatim by path import.

Do not copy-modify ``scripts/p05n_field_derived_global/fields.py`` (another
agent is concurrently developing that package). This module loads it by file
path (a distinct module identity, so this package's own ``fields`` module
name never collides with it) and re-exports ``evaluate``/``normal``/
``normal_data`` unchanged for the six P05N-inherited fields
(``field_b1``, ``field_e3``, ``field_e12``, ``heldout_field_b2``, ``constant``,
``zero_trace_generator``). It adds only two shifted variants, frozen by the
provisional bounded assignment in
``work/p_neumann_p05n_p06n_design_20260927/design.md`` section "Provisional
bounded field assignment":

- ``field_e3_minus1`` = field_e3 - 1: the Neumann-phi generator (pairing a).
- ``field_b1_minus1`` = field_b1 - 1: the omega field (any smooth field --
  omega does not affect the P06 curvature action; see the design doc and
  ``operator.py``'s ``check_omega_independence``) and also usable as a second
  Dirichlet-phi generator (pairing b2), per the design doc's "or, separately,
  field_b1-1 with its trace".

Per the task spec: never evaluate any N (numerical) action on
``heldout_field_b2`` here -- it is reserved, unseen, for the P06N catalogue
freeze. This module exposes it (inherited from P05N/P07N) but nothing in the
P06N driver below uses it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_P05N_FIELDS_PATH = REPO / "scripts/p05n_field_derived_global/fields.py"


def _load_p05n_fields():
    spec = importlib.util.spec_from_file_location("p06n_field_derived_global._p05n_fields_verbatim", _P05N_FIELDS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_p05n = _load_p05n_fields()

# Verbatim re-exports; do not modify.
_P05N_NAMES = _p05n.NAMES  # ('field_b1','field_e3','field_e12','heldout_field_b2','constant','zero_trace_generator')
normal = _p05n.normal  # (ref, q) -> physical-normal a = g^{u.}/sqrt(g^{uu}), field-independent

# The two shifted variants this package adds. Both are simple additive shifts
# of an inherited field, so their value/gradient/Hessian and normal datum are
# those of the base field with the value shifted (gradients, Hessians and the
# normal derivative g_N = a.grad are shift-invariant).
_SHIFTED = {
    "field_e3_minus1": "field_e3",
    "field_b1_minus1": "field_b1",
    "rich_a_minus1": "rich_a",
    "rich_f_minus1": "rich_f",
}

# Rich fields for the P06N catalogue. rich_a and rich_f come from P05N (fields
# outside the reconstruction's exactness space). rich_c and the P06N held-out
# heldout_rich_h are defined here; their code was generated once with sympy and
# pasted as plain numpy, so no sympy is needed at run time.
P05N_RICH = ("rich_a", "rich_f")
P06N_RICH = ("rich_c", "heldout_rich_h")
NAMES = _P05N_NAMES + P05N_RICH + tuple(_SHIFTED) + P06N_RICH

import numpy  # noqa: E402  (the generated code uses fully qualified numpy names)

def _rich_c_raw(u, th, z):
    """u**6*sin(6*th)*cos(2*z)/25 + u**4*(1 - 3*u**2/10)*cos(z)*cos(4*th + 3/5)/20 + 3*u*sin(th)*cos(z)/50 + 1 + 2*exp(-2*u**2)*sin(z + 3/10)/25"""
    v = (1/25)*u**6*numpy.sin(6*th)*numpy.cos(2*z) + (1/20)*u**4*(1 - 3/10*u**2)*numpy.cos(z)*numpy.cos(4*th + 3/5) + (3/50)*u*numpy.sin(th)*numpy.cos(z) + 1 + (2/25)*numpy.exp(-2*u**2)*numpy.sin(z + 3/10)
    gu = (6/25)*u**5*numpy.sin(6*th)*numpy.cos(2*z) - 3/100*u**5*numpy.cos(z)*numpy.cos(4*th + 3/5) + (1/5)*u**3*(1 - 3/10*u**2)*numpy.cos(z)*numpy.cos(4*th + 3/5) - 8/25*u*numpy.exp(-2*u**2)*numpy.sin(z + 3/10) + (3/50)*numpy.sin(th)*numpy.cos(z)
    gt = (6/25)*u**6*numpy.cos(6*th)*numpy.cos(2*z) - 1/5*u**4*(1 - 3/10*u**2)*numpy.sin(4*th + 3/5)*numpy.cos(z) + (3/50)*u*numpy.cos(th)*numpy.cos(z)
    gz = -2/25*u**6*numpy.sin(6*th)*numpy.sin(2*z) - 1/20*u**4*(1 - 3/10*u**2)*numpy.sin(z)*numpy.cos(4*th + 3/5) - 3/50*u*numpy.sin(th)*numpy.sin(z) + (2/25)*numpy.exp(-2*u**2)*numpy.cos(z + 3/10)
    huu = (1/100)*(120*u**4*numpy.sin(6*th)*numpy.cos(2*z) - 27*u**4*numpy.cos(z)*numpy.cos(4*th + 3/5) - 6*u**2*(3*u**2 - 10)*numpy.cos(z)*numpy.cos(4*th + 3/5) + 128*u**2*numpy.exp(-2*u**2)*numpy.sin(z + 3/10) - 32*numpy.exp(-2*u**2)*numpy.sin(z + 3/10))
    hut = (1/50)*(6*u**5*numpy.sin(4*th + 3/5)*numpy.cos(z) + 72*u**5*numpy.cos(6*th)*numpy.cos(2*z) + 4*u**3*(3*u**2 - 10)*numpy.sin(4*th + 3/5)*numpy.cos(z) + 3*numpy.cos(th)*numpy.cos(z))
    huz = (1/100)*(-48*u**5*numpy.sin(6*th)*numpy.sin(2*z) + 3*u**5*numpy.sin(z)*numpy.cos(4*th + 3/5) + 2*u**3*(3*u**2 - 10)*numpy.sin(z)*numpy.cos(4*th + 3/5) - 32*u*numpy.exp(-2*u**2)*numpy.cos(z + 3/10) - 6*numpy.sin(th)*numpy.sin(z))
    htt = (1/50)*u*(-72*u**5*numpy.sin(6*th)*numpy.cos(2*z) + 4*u**3*(3*u**2 - 10)*numpy.cos(z)*numpy.cos(4*th + 3/5) - 3*numpy.sin(th)*numpy.cos(z))
    htz = -1/50*u*(24*u**5*numpy.sin(2*z)*numpy.cos(6*th) + u**3*(3*u**2 - 10)*numpy.sin(z)*numpy.sin(4*th + 3/5) + 3*numpy.sin(z)*numpy.cos(th))
    hzz = (1/100)*(-16*u**6*numpy.sin(6*th)*numpy.cos(2*z) + (1/2)*u**4*(3*u**2 - 10)*numpy.cos(z)*numpy.cos(4*th + 3/5) - 6*u*numpy.sin(th)*numpy.cos(z) - 8*numpy.exp(-2*u**2)*numpy.sin(z + 3/10))
    return v, (gu, gt, gz), (huu, hut, huz, htt, htz, hzz)

def _heldout_rich_h_raw(u, th, z):
    """u**5*sin(z)*sin(5*th + 1/5)/20 + u**4*(u**2/5 + 1)*sin(2*z + 1/10)*cos(4*th)/25 + u*sin(z)*cos(th)/20 + 7*cos(pi*u**2/3)*cos(2*z + 1/2)/100 + 1"""
    v = (1/20)*u**5*numpy.sin(z)*numpy.sin(5*th + 1/5) + (1/25)*u**4*((1/5)*u**2 + 1)*numpy.sin(2*z + 1/10)*numpy.cos(4*th) + (1/20)*u*numpy.sin(z)*numpy.cos(th) + (7/100)*numpy.cos((1/3)*numpy.pi*u**2)*numpy.cos(2*z + 1/2) + 1
    gu = (2/125)*u**5*numpy.sin(2*z + 1/10)*numpy.cos(4*th) + (1/4)*u**4*numpy.sin(z)*numpy.sin(5*th + 1/5) + (4/25)*u**3*((1/5)*u**2 + 1)*numpy.sin(2*z + 1/10)*numpy.cos(4*th) - 7/150*numpy.pi*u*numpy.sin((1/3)*numpy.pi*u**2)*numpy.cos(2*z + 1/2) + (1/20)*numpy.sin(z)*numpy.cos(th)
    gt = (1/4)*u**5*numpy.sin(z)*numpy.cos(5*th + 1/5) - 4/25*u**4*((1/5)*u**2 + 1)*numpy.sin(4*th)*numpy.sin(2*z + 1/10) - 1/20*u*numpy.sin(th)*numpy.sin(z)
    gz = (1/20)*u**5*numpy.sin(5*th + 1/5)*numpy.cos(z) + (2/25)*u**4*((1/5)*u**2 + 1)*numpy.cos(4*th)*numpy.cos(2*z + 1/10) + (1/20)*u*numpy.cos(th)*numpy.cos(z) - 7/50*numpy.sin(2*z + 1/2)*numpy.cos((1/3)*numpy.pi*u**2)
    huu = (18/125)*u**4*numpy.sin(2*z + 1/10)*numpy.cos(4*th) + u**3*numpy.sin(z)*numpy.sin(5*th + 1/5) + (12/125)*u**2*(u**2 + 5)*numpy.sin(2*z + 1/10)*numpy.cos(4*th) - 7/225*numpy.pi**2*u**2*numpy.cos((1/3)*numpy.pi*u**2)*numpy.cos(2*z + 1/2) - 7/150*numpy.pi*numpy.sin((1/3)*numpy.pi*u**2)*numpy.cos(2*z + 1/2)
    hut = (1/500)*(-32*u**5*numpy.sin(4*th)*numpy.sin(2*z + 1/10) + 625*u**4*numpy.sin(z)*numpy.cos(5*th + 1/5) - 64*u**3*(u**2 + 5)*numpy.sin(4*th)*numpy.sin(2*z + 1/10) - 25*numpy.sin(th)*numpy.sin(z))
    huz = (1/1500)*(48*u**5*numpy.cos(4*th)*numpy.cos(2*z + 1/10) + 375*u**4*numpy.sin(5*th + 1/5)*numpy.cos(z) + 96*u**3*(u**2 + 5)*numpy.cos(4*th)*numpy.cos(2*z + 1/10) + 140*numpy.pi*u*numpy.sin((1/3)*numpy.pi*u**2)*numpy.sin(2*z + 1/2) + 75*numpy.cos(th)*numpy.cos(z))
    htt = -1/100*u*(125*u**4*numpy.sin(z)*numpy.sin(5*th + 1/5) + (64/5)*u**3*(u**2 + 5)*numpy.sin(2*z + 1/10)*numpy.cos(4*th) + 5*numpy.sin(z)*numpy.cos(th))
    htz = (1/100)*u*(25*u**4*numpy.cos(z)*numpy.cos(5*th + 1/5) - 32/5*u**3*(u**2 + 5)*numpy.sin(4*th)*numpy.cos(2*z + 1/10) - 5*numpy.sin(th)*numpy.cos(z))
    hzz = -1/100*(5*u**5*numpy.sin(z)*numpy.sin(5*th + 1/5) + (16/5)*u**4*(u**2 + 5)*numpy.sin(2*z + 1/10)*numpy.cos(4*th) + 5*u*numpy.sin(z)*numpy.cos(th) + 28*numpy.cos((1/3)*numpy.pi*u**2)*numpy.cos(2*z + 1/2))
    return v, (gu, gt, gz), (huu, hut, huz, htt, htz, hzz)


_P06N_RAW = {"rich_c": _rich_c_raw, "heldout_rich_h": _heldout_rich_h_raw}


def _p06n_rich(q, name, period):
    q = np.atleast_2d(np.asarray(q, dtype=float))
    om = 2 * np.pi / period
    ones = np.ones(len(q))
    v, (gu, gt, gz), (huu, hut, huz, htt, htz, hzz) = _P06N_RAW[name](q[:, 0], q[:, 1], om * q[:, 2])
    g = np.stack([gu * ones, gt * ones, om * gz * ones], axis=1)
    H = np.empty((len(q), 3, 3))
    H[:, 0, 0] = huu * ones; H[:, 0, 1] = H[:, 1, 0] = hut * ones; H[:, 0, 2] = H[:, 2, 0] = om * huz * ones
    H[:, 1, 1] = htt * ones; H[:, 1, 2] = H[:, 2, 1] = om * htz * ones; H[:, 2, 2] = om * om * hzz * ones
    return v * ones, g, H

assert "heldout_field_b2" in NAMES  # inherited, present, but never driven below


def evaluate(ref, q, name, period, step=None, derivatives=True):
    """(v, g, H) -- p05n's fields verbatim, the shifted variants and the P06N rich fields."""
    if name in P06N_RICH:
        return _p06n_rich(q, name, period)
    if name in _SHIFTED:
        v, g, H = _p05n.evaluate(ref, q, _SHIFTED[name], period, step=step, derivatives=derivatives)
        return v - 1.0, g, H
    return _p05n.evaluate(ref, q, name, period, step=step, derivatives=derivatives)


def normal_data(ref, q, name, period):
    """Physical-normal wall datum g_N = a.grad_x f (shift-invariant)."""
    if name in P06N_RICH:
        q = np.asarray(q, dtype=float)
        return np.einsum("qa,qa->q", normal(ref, q), _p06n_rich(q, name, period)[1])
    if name in _SHIFTED:
        return _p05n.normal_data(ref, q, _SHIFTED[name], period)
    return _p05n.normal_data(ref, q, name, period)


def dirichlet_trace(ref, q, name, period):
    """(value, gradient) at points q, in the boundary_trace(ref, p) format."""
    v, g, _ = evaluate(ref, q, name, period)
    return v, g
