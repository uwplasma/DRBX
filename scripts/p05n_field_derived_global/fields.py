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


def evaluate(ref, q, name, period, step=None, derivatives=True):
    """P05N dispatcher: p07n's five fields verbatim, plus zero_trace_generator."""
    if name == "zero_trace_generator":
        return _zero_trace_generator(np.atleast_2d(np.asarray(q, dtype=float)), period)
    return _p07n.evaluate(ref, q, name, period, step=step, derivatives=derivatives)


def normal_data(ref, q, name, period):
    """P05N dispatcher: physical-normal wall datum g_N = a.grad_x f, a = normal(ref,q)."""
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
