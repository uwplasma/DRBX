"""P05N field-derived catalogue: reuses P07N's analytic fields verbatim by path import.

Do not copy-modify scripts/p07n_field_derived_global/fields.py. This module loads it
by file path (a distinct module identity, so this package's own ``fields`` module
name never collides with it) and re-exports ``NAMES``, ``evaluate``, ``normal`` and
``normal_data`` unchanged for the five P07N-inherited fields. It adds only a
Dirichlet trace adapter in the same (value, gradient) format as
``p05_structured_global.numerics.boundary_trace``, for the pairing-(b)
Dirichlet-generator rows and the all-Dirichlet replay check -- plus one new
field, ``zero_trace_generator``, added here (not in p07n) for the frozen P05N
catalogue (vendored byte-identically as p05n_catalogue.json in this package).

``normal_data(ref, q, name, period)`` is the physical-normal Neumann datum g_N =
n.grad_x f, with n = gcontra[:,0,:]/sqrt(gcontra[:,0,0]) (the ripple-free physical
normal), used unchanged for both the wall-node elimination inside
``prepare_neumann_point_rows`` and for scoring/exact-input checks here.

``evaluate``/``normal_data`` below are P05N-level dispatchers: they delegate to
the verbatim p07n functions for the five inherited names, and compute
``zero_trace_generator`` locally. ``dirichlet_trace`` calls this module's
``evaluate`` (not p07n's), so it works unchanged for every P05N field.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_P07N_FIELDS_PATH = REPO / "scripts/p07n_field_derived_global/fields.py"


def _load_p07n_fields():
    spec = importlib.util.spec_from_file_location("p05n_field_derived_global._p07n_fields_verbatim", _P07N_FIELDS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_p07n = _load_p07n_fields()

# Verbatim re-exports; do not modify.
_P07N_NAMES = _p07n.NAMES  # ('field_b1','field_e3','field_e12','heldout_field_b2','constant')
base = _p07n.base
normal = _p07n.normal

# P05N-extended catalogue: the five p07n fields plus zero_trace_generator.
NAMES = _P07N_NAMES + ("zero_trace_generator",)

# actual_vorticity is excluded from this catalogue: its wall normal derivative
# is unavailable at r=1 (P07N does not define a Neumann datum for it either).
assert "actual_vorticity" not in NAMES


def _zero_trace_generator(q, period):
    """(v, g, H) for f = (1-u^2)*(0.12*x*cos(z) + 0.08*(x^2-y^2)*sin(2z)).

    x = u*cos(theta), y = u*sin(theta), z = (2*pi/period)*eta. Zero value at
    the wall (u=1, the (1-u^2) prefactor) with nonzero normal (u) derivative
    there; axis-regular (polynomial in x,y times eta harmonics). Frozen by
    p05n_catalogue.json (vendored in this package directory).
    """
    q = np.asarray(q, dtype=float)
    u, th, eta = q[:, 0], q[:, 1], q[:, 2]
    om = 2 * np.pi / period
    z = om * eta
    c, s = np.cos(th), np.sin(th)
    c2, s2 = np.cos(2 * th), np.sin(2 * th)
    cz, sz = np.cos(z), np.sin(z)
    c2z, s2z = np.cos(2 * z), np.sin(2 * z)

    A = 0.12 * u * c * cz + 0.08 * u * u * c2 * s2z
    dA_du = 0.12 * c * cz + 0.16 * u * c2 * s2z
    dA_dth = -0.12 * u * s * cz - 0.16 * u * u * s2 * s2z
    dA_dz = -0.12 * u * c * sz + 0.16 * u * u * c2 * c2z

    d2A_duu = 0.16 * c2 * s2z
    d2A_duth = -0.12 * s * cz - 0.32 * u * s2 * s2z
    d2A_duz = -0.12 * c * sz + 0.32 * u * c2 * c2z
    d2A_thth = -0.12 * u * c * cz - 0.32 * u * u * c2 * s2z
    d2A_thz = 0.12 * u * s * sz - 0.32 * u * u * s2 * c2z
    d2A_zz = -0.12 * u * c * cz - 0.32 * u * u * c2 * s2z

    sfac = 1 - u * u
    n = len(q)
    v = sfac * A
    g = np.empty((n, 3))
    g[:, 0] = -2 * u * A + sfac * dA_du
    g[:, 1] = sfac * dA_dth
    g[:, 2] = om * sfac * dA_dz

    H = np.empty((n, 3, 3))
    H[:, 0, 0] = -2 * A - 4 * u * dA_du + sfac * d2A_duu
    H[:, 0, 1] = H[:, 1, 0] = -2 * u * dA_dth + sfac * d2A_duth
    H[:, 0, 2] = H[:, 2, 0] = om * (-2 * u * dA_dz + sfac * d2A_duz)
    H[:, 1, 1] = sfac * d2A_thth
    H[:, 1, 2] = H[:, 2, 1] = om * sfac * d2A_thz
    H[:, 2, 2] = om * om * sfac * d2A_zz
    return v, g, H


# ---------------------------------------------------------------------------
# Rich fields for the upwind_v1 catalogue (p05n_upwind_catalogue.json).
#
# The frozen_v1 fields are at most cubic in u with theta harmonics <= 3, which
# the interior reconstruction stencils reproduce exactly, so the upwind jump
# vanishes on interior faces. These fields are outside that exactness space.
# The code below was generated once with sympy (value, logical gradient and
# Hessian in (u, theta, z), z = 2*pi*eta/period) and pasted as plain numpy, so
# no sympy is needed at run time; tests check it against finite differences.
# ---------------------------------------------------------------------------
import numpy  # noqa: E402  (the generated code uses fully qualified numpy names)

RICH_NAMES = ("rich_a", "rich_f", "heldout_rich_g")
ALL_NAMES = NAMES + RICH_NAMES

def _rich_f_raw(u, th, z):
    """u**5*sin(5*th)*sin(2*z)/25 + 3*u**4*(u**2/2 + 1)*cos(4*th)*cos(z + 2/5)/50 + u*cos(th)*cos(z)/20 + 1 + exp(-u**2)*cos(z)/10"""
    v = (1/25)*u**5*numpy.sin(5*th)*numpy.sin(2*z) + (3/50)*u**4*((1/2)*u**2 + 1)*numpy.cos(4*th)*numpy.cos(z + 2/5) + (1/20)*u*numpy.cos(th)*numpy.cos(z) + 1 + (1/10)*numpy.exp(-u**2)*numpy.cos(z)
    gu = (3/50)*u**5*numpy.cos(4*th)*numpy.cos(z + 2/5) + (1/5)*u**4*numpy.sin(5*th)*numpy.sin(2*z) + (6/25)*u**3*((1/2)*u**2 + 1)*numpy.cos(4*th)*numpy.cos(z + 2/5) - 1/5*u*numpy.exp(-u**2)*numpy.cos(z) + (1/20)*numpy.cos(th)*numpy.cos(z)
    gt = (1/5)*u**5*numpy.sin(2*z)*numpy.cos(5*th) - 6/25*u**4*((1/2)*u**2 + 1)*numpy.sin(4*th)*numpy.cos(z + 2/5) - 1/20*u*numpy.sin(th)*numpy.cos(z)
    gz = (2/25)*u**5*numpy.sin(5*th)*numpy.cos(2*z) - 3/50*u**4*((1/2)*u**2 + 1)*numpy.sin(z + 2/5)*numpy.cos(4*th) - 1/20*u*numpy.sin(z)*numpy.cos(th) - 1/10*numpy.exp(-u**2)*numpy.sin(z)
    huu = (1/50)*(27*u**4*numpy.cos(4*th)*numpy.cos(z + 2/5) + 40*u**3*numpy.sin(5*th)*numpy.sin(2*z) + 18*u**2*(u**2 + 2)*numpy.cos(4*th)*numpy.cos(z + 2/5) + 20*u**2*numpy.exp(-u**2)*numpy.cos(z) - 10*numpy.exp(-u**2)*numpy.cos(z))
    hut = -6/25*u**5*numpy.sin(4*th)*numpy.cos(z + 2/5) + u**4*numpy.sin(2*z)*numpy.cos(5*th) - 12/25*u**3*(u**2 + 2)*numpy.sin(4*th)*numpy.cos(z + 2/5) - 1/20*numpy.sin(th)*numpy.cos(z)
    huz = (1/100)*(-6*u**5*numpy.sin(z + 2/5)*numpy.cos(4*th) + 40*u**4*numpy.sin(5*th)*numpy.cos(2*z) - 12*u**3*(u**2 + 2)*numpy.sin(z + 2/5)*numpy.cos(4*th) + 20*u*numpy.exp(-u**2)*numpy.sin(z) - 5*numpy.sin(z)*numpy.cos(th))
    htt = -u*(u**4*numpy.sin(5*th)*numpy.sin(2*z) + (12/25)*u**3*(u**2 + 2)*numpy.cos(4*th)*numpy.cos(z + 2/5) + (1/20)*numpy.cos(th)*numpy.cos(z))
    htz = (1/100)*u*(40*u**4*numpy.cos(5*th)*numpy.cos(2*z) + 12*u**3*(u**2 + 2)*numpy.sin(4*th)*numpy.sin(z + 2/5) + 5*numpy.sin(th)*numpy.sin(z))
    hzz = -1/100*(16*u**5*numpy.sin(5*th)*numpy.sin(2*z) + 3*u**4*(u**2 + 2)*numpy.cos(4*th)*numpy.cos(z + 2/5) + 5*u*numpy.cos(th)*numpy.cos(z) + 10*numpy.exp(-u**2)*numpy.cos(z))
    return v, (gu, gt, gz), (huu, hut, huz, htt, htz, hzz)

def _rich_a_raw(u, th, z):
    """3*u**6*sin(z)*cos(6*th)/100 + u**4*sin(4*th)*cos(2*z + 1/5)/20 + u*cos(th)*cos(z)/10 + 2*sin(z)*sin(pi*u**2/2)/25 + 1"""
    v = (3/100)*u**6*numpy.sin(z)*numpy.cos(6*th) + (1/20)*u**4*numpy.sin(4*th)*numpy.cos(2*z + 1/5) + (1/10)*u*numpy.cos(th)*numpy.cos(z) + (2/25)*numpy.sin(z)*numpy.sin((1/2)*numpy.pi*u**2) + 1
    gu = (9/50)*u**5*numpy.sin(z)*numpy.cos(6*th) + (1/5)*u**3*numpy.sin(4*th)*numpy.cos(2*z + 1/5) + (2/25)*numpy.pi*u*numpy.sin(z)*numpy.cos((1/2)*numpy.pi*u**2) + (1/10)*numpy.cos(th)*numpy.cos(z)
    gt = -9/50*u**6*numpy.sin(6*th)*numpy.sin(z) + (1/5)*u**4*numpy.cos(4*th)*numpy.cos(2*z + 1/5) - 1/10*u*numpy.sin(th)*numpy.cos(z)
    gz = (3/100)*u**6*numpy.cos(6*th)*numpy.cos(z) - 1/10*u**4*numpy.sin(4*th)*numpy.sin(2*z + 1/5) - 1/10*u*numpy.sin(z)*numpy.cos(th) + (2/25)*numpy.sin((1/2)*numpy.pi*u**2)*numpy.cos(z)
    huu = (1/50)*(45*u**4*numpy.sin(z)*numpy.cos(6*th) + 30*u**2*numpy.sin(4*th)*numpy.cos(2*z + 1/5) - 4*numpy.pi**2*u**2*numpy.sin(z)*numpy.sin((1/2)*numpy.pi*u**2) + 4*numpy.pi*numpy.sin(z)*numpy.cos((1/2)*numpy.pi*u**2))
    hut = (1/50)*(-54*u**5*numpy.sin(6*th)*numpy.sin(z) + 40*u**3*numpy.cos(4*th)*numpy.cos(2*z + 1/5) - 5*numpy.sin(th)*numpy.cos(z))
    huz = (1/50)*(9*u**5*numpy.cos(6*th)*numpy.cos(z) - 20*u**3*numpy.sin(4*th)*numpy.sin(2*z + 1/5) + 4*numpy.pi*u*numpy.cos(z)*numpy.cos((1/2)*numpy.pi*u**2) - 5*numpy.sin(z)*numpy.cos(th))
    htt = -1/50*u*(54*u**5*numpy.sin(z)*numpy.cos(6*th) + 40*u**3*numpy.sin(4*th)*numpy.cos(2*z + 1/5) + 5*numpy.cos(th)*numpy.cos(z))
    htz = (1/50)*u*(-9*u**5*numpy.sin(6*th)*numpy.cos(z) - 20*u**3*numpy.sin(2*z + 1/5)*numpy.cos(4*th) + 5*numpy.sin(th)*numpy.sin(z))
    hzz = -1/100*(3*u**6*numpy.sin(z)*numpy.cos(6*th) + 20*u**4*numpy.sin(4*th)*numpy.cos(2*z + 1/5) + 10*u*numpy.cos(th)*numpy.cos(z) + 8*numpy.sin(z)*numpy.sin((1/2)*numpy.pi*u**2))
    return v, (gu, gt, gz), (huu, hut, huz, htt, htz, hzz)

def _heldout_rich_g_raw(u, th, z):
    """u**5*cos(z)*cos(5*th + 3/10)/20 + u**4*(1 - 2*u**2/5)*sin(2*z)*sin(4*th + 1/2)/20 + 3*u*sin(th)*sin(z)/50 + 9*cos(pi*u**2/2)*cos(z + 7/10)/100 + 1"""
    v = (1/20)*u**5*numpy.cos(z)*numpy.cos(5*th + 3/10) + (1/20)*u**4*(1 - 2/5*u**2)*numpy.sin(2*z)*numpy.sin(4*th + 1/2) + (3/50)*u*numpy.sin(th)*numpy.sin(z) + (9/100)*numpy.cos((1/2)*numpy.pi*u**2)*numpy.cos(z + 7/10) + 1
    gu = -1/25*u**5*numpy.sin(2*z)*numpy.sin(4*th + 1/2) + (1/4)*u**4*numpy.cos(z)*numpy.cos(5*th + 3/10) + (1/5)*u**3*(1 - 2/5*u**2)*numpy.sin(2*z)*numpy.sin(4*th + 1/2) - 9/100*numpy.pi*u*numpy.sin((1/2)*numpy.pi*u**2)*numpy.cos(z + 7/10) + (3/50)*numpy.sin(th)*numpy.sin(z)
    gt = -1/4*u**5*numpy.sin(5*th + 3/10)*numpy.cos(z) + (1/5)*u**4*(1 - 2/5*u**2)*numpy.sin(2*z)*numpy.cos(4*th + 1/2) + (3/50)*u*numpy.sin(z)*numpy.cos(th)
    gz = -1/20*u**5*numpy.sin(z)*numpy.cos(5*th + 3/10) + (1/10)*u**4*(1 - 2/5*u**2)*numpy.sin(4*th + 1/2)*numpy.cos(2*z) + (3/50)*u*numpy.sin(th)*numpy.cos(z) - 9/100*numpy.sin(z + 7/10)*numpy.cos((1/2)*numpy.pi*u**2)
    huu = -9/25*u**4*numpy.sin(2*z)*numpy.sin(4*th + 1/2) + u**3*numpy.cos(z)*numpy.cos(5*th + 3/10) - 3/25*u**2*(2*u**2 - 5)*numpy.sin(2*z)*numpy.sin(4*th + 1/2) - 9/100*numpy.pi**2*u**2*numpy.cos((1/2)*numpy.pi*u**2)*numpy.cos(z + 7/10) - 9/100*numpy.pi*numpy.sin((1/2)*numpy.pi*u**2)*numpy.cos(z + 7/10)
    hut = (1/100)*(-16*u**5*numpy.sin(2*z)*numpy.cos(4*th + 1/2) - 125*u**4*numpy.sin(5*th + 3/10)*numpy.cos(z) - 16*u**3*(2*u**2 - 5)*numpy.sin(2*z)*numpy.cos(4*th + 1/2) + 6*numpy.sin(z)*numpy.cos(th))
    huz = (1/100)*(-8*u**5*numpy.sin(4*th + 1/2)*numpy.cos(2*z) - 25*u**4*numpy.sin(z)*numpy.cos(5*th + 3/10) - 8*u**3*(2*u**2 - 5)*numpy.sin(4*th + 1/2)*numpy.cos(2*z) + 9*numpy.pi*u*numpy.sin((1/2)*numpy.pi*u**2)*numpy.sin(z + 7/10) + 6*numpy.sin(th)*numpy.cos(z))
    htt = (1/100)*u*(-125*u**4*numpy.cos(z)*numpy.cos(5*th + 3/10) + 16*u**3*(2*u**2 - 5)*numpy.sin(2*z)*numpy.sin(4*th + 1/2) - 6*numpy.sin(th)*numpy.sin(z))
    htz = (1/100)*u*(25*u**4*numpy.sin(z)*numpy.sin(5*th + 3/10) - 8*u**3*(2*u**2 - 5)*numpy.cos(2*z)*numpy.cos(4*th + 1/2) + 6*numpy.cos(th)*numpy.cos(z))
    hzz = (1/100)*(-5*u**5*numpy.cos(z)*numpy.cos(5*th + 3/10) + 4*u**4*(2*u**2 - 5)*numpy.sin(2*z)*numpy.sin(4*th + 1/2) - 6*u*numpy.sin(th)*numpy.sin(z) - 9*numpy.cos((1/2)*numpy.pi*u**2)*numpy.cos(z + 7/10))
    return v, (gu, gt, gz), (huu, hut, huz, htt, htz, hzz)


_RICH_RAW = {"rich_a": _rich_a_raw, "rich_f": _rich_f_raw, "heldout_rich_g": _heldout_rich_g_raw}


def _rich(q, name, period):
    q = np.atleast_2d(np.asarray(q, dtype=float))
    om = 2 * np.pi / period
    ones = np.ones(len(q))
    v, (gu, gt, gz), (huu, hut, huz, htt, htz, hzz) = _RICH_RAW[name](q[:, 0], q[:, 1], om * q[:, 2])
    g = np.stack([gu * ones, gt * ones, om * gz * ones], axis=1)
    H = np.empty((len(q), 3, 3))
    H[:, 0, 0] = huu * ones; H[:, 0, 1] = H[:, 1, 0] = hut * ones; H[:, 0, 2] = H[:, 2, 0] = om * huz * ones
    H[:, 1, 1] = htt * ones; H[:, 1, 2] = H[:, 2, 1] = om * htz * ones; H[:, 2, 2] = om * om * hzz * ones
    return v * ones, g, H


def evaluate(ref, q, name, period, step=None, derivatives=True):
    """P05N dispatcher: p07n's five fields verbatim, zero_trace_generator and the rich fields."""
    if name in RICH_NAMES:
        return _rich(q, name, period)
    if name == "zero_trace_generator":
        return _zero_trace_generator(np.atleast_2d(np.asarray(q, dtype=float)), period)
    return _p07n.evaluate(ref, q, name, period, step=step, derivatives=derivatives)


def normal_data(ref, q, name, period):
    """P05N dispatcher: physical-normal wall datum g_N = a.grad_x f, a = normal(ref,q)."""
    if name in RICH_NAMES:
        q = np.asarray(q, dtype=float)
        return np.einsum("qa,qa->q", normal(ref, q), _rich(q, name, period)[1])
    if name == "zero_trace_generator":
        q = np.asarray(q, dtype=float)
        a = normal(ref, q)
        g = _zero_trace_generator(q, period)[1]
        return np.einsum("qa,qa->q", a, g)
    return _p07n.normal_data(ref, q, name, period)


def dirichlet_trace(ref, q, name, period):
    """(value, gradient) at points ``q``, in the ``boundary_trace(ref, p)`` format.

    Only the tangential (theta, eta) slots of the returned gradient are used by
    ``PointRows.apply``'s boundary-conditioned lift (``gradient[:, 1:] += dgt[:, 1:]``);
    the radial slot is present for shape compatibility only and is never read there.
    """
    v, g, _ = evaluate(ref, q, name, period)
    return v, g
