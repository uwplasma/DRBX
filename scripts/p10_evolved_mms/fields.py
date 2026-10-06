"""Manufactured fields of the P10 evolved MMS: ``f(p, t)`` of ``(n, Te, Ti, Omega, phi)`` and their derivatives (chunk C2).

The time-dependent step-6 transverse family of :mod:`p09_evolved_mms.fields` (``TimeState``, ``TIME_PARAMS``) with three
harness parameters (:class:`MmsParams`):

* ``a_phi``       scales the potential ``phi`` (the advective strength knob);
* ``time_scale``  multiplies the time: the fields are the P09 fields at ``time_scale * t``, i.e. every manufactured frequency
                  (the phase drifts ``Omega_t`` and the amplitude modulations ``nu``) is multiplied by it, and ``d/dt``
                  carries the factor. ``time_scale = 0`` freezes the fields at the static step-6 set (no time dependence at
                  any ``t``);
* ``w1``          the W1 variant flag (reported-only case): ``w1 = 1`` adds the eta-harmonic-2 switch term :data:`W1_SPEC` to
                  ``Omega``; ``w1 = 0`` is the base family.

Python-float ``a_phi == 1``, ``time_scale == 1`` and ``w1 == 0`` add no operation (so the defaults trace to the P09 fields
exactly); traced values are fine inside ``jit``. Values, gradients, Hessians (``jax.jacfwd`` of ``jax.jacfwd``) and ``d/dt``
are exact autodiff of the analytic definitions, evaluated on arrays of points ``(..., 3) = (u, theta, eta)`` with the field
axis last: values ``(..., 5)``, gradients ``(..., 5, 3)``, Hessians ``(..., 5, 3, 3)``, ``d/dt`` ``(..., 5)``.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p09_evolved_mms.fields import (FIELD_SET, TERM_OMEGA, TIME_PARAMS, TimeState, build_function,    # noqa: E402,F401
                                    time_fields)

CONFIG = json.loads((HERE / "configuration.json").read_text())
SLOTS = ("n", "Te", "Ti", "omega", "phi")
FIELD_NAMES = SLOTS[:4]
PERIOD = 2.0 * np.pi
#: the eta-harmonic-2 term of the W1 variant: the vorticity's switch structure at twice the eta wavenumber, with its own phase
#: drift and modulation (a time-dependent field with the eta-parity content of harmonic 2)
W1_SPEC = {"kind": "switch_harmonic", "harmonic": "cos4", "u_s": 0.21, "w2": 0.06, "eta_phase": 0.5, "amplitude": 0.5,
           "Omega": -30.0, "a_mod": 0.2, "nu": 40.0}


class MmsParams(NamedTuple):
    """Parameters of the manufactured problem (a pytree of scalars; ``D (4,)`` for ``(n, Te, Ti, Omega)``)."""

    rho_star: float | jax.Array
    tau: float | jax.Array
    D: np.ndarray | jax.Array
    a_phi: float | jax.Array = 1.0
    time_scale: float | jax.Array = 1.0
    w1: float | jax.Array = 0.0

    def nodal(self):
        """The :class:`~drbx.native.fci_nodal_perpendicular_rhs.NodalPerpendicularParams` of the RHS."""
        from drbx.native.fci_nodal_perpendicular_rhs import NodalPerpendicularParams

        return NodalPerpendicularParams(jnp.asarray(self.rho_star), jnp.asarray(self.tau), jnp.asarray(self.D))


def default_params(**override) -> MmsParams:
    """The provisional ``configuration.json`` parameters (overridable by keyword)."""
    c = CONFIG
    base = dict(rho_star=float(c["rho_star"]), tau=float(c["tau"]), D=np.asarray(c["D"], dtype=np.float64),
                a_phi=float(c["a_phi"]), time_scale=float(c["time_scale"]), w1=float(c["w1"]))
    base.update(override)
    return MmsParams(**base)


def _is_const(x, value) -> bool:
    return isinstance(x, (int, float)) and float(x) == value


class MmsFields:
    """The five manufactured fields with exact derivatives; ``om = 2 pi / eta_period`` (``1`` for the period ``2 pi``)."""

    def __init__(self, om: float = 1.0, post=None):
        """``post(p, t, v, mp) -> v`` (optional) rewrites the stacked values ``v (5,)`` of a point (a test hook: a derived field
        such as ``Omega := rho_star^2 L psi``); derivatives are taken through it."""
        self.om = float(om)
        self.base = TimeState(om=self.om)
        self._w1 = build_function(W1_SPEC, 2.0 * self.om)

        def f(p, t, mp):
            ts = mp.time_scale
            tt = t if _is_const(ts, 1.0) else (0.0 if _is_const(ts, 0.0) else ts * t)
            v = self.base.f(p, tt)
            n, te, ti, om_, ph = v[0], v[1], v[2], v[3], v[4]
            if not _is_const(mp.a_phi, 1.0):
                ph = mp.a_phi * ph
            if not _is_const(mp.w1, 0.0):
                om_ = om_ + mp.w1 * self._w1(p, tt)
            v = jnp.stack([n, te, ti, om_, ph])
            return v if post is None else post(p, t, v, mp)

        self.f = f
        d1 = jax.jacfwd(f, argnums=0)
        d2 = jax.jacfwd(d1, argnums=0)
        dt = jax.jacfwd(f, argnums=1)
        self.point_all = lambda p, t, mp: (f(p, t, mp), d1(p, t, mp), d2(p, t, mp), dt(p, t, mp))
        self._jit = {}
        self._k = {"v": jax.vmap(f, in_axes=(0, None, None)),
                   "vdt": jax.vmap(lambda p, t, mp: (f(p, t, mp), dt(p, t, mp)), in_axes=(0, None, None)),
                   "vg": jax.vmap(lambda p, t, mp: (f(p, t, mp), d1(p, t, mp)), in_axes=(0, None, None)),
                   "vgh": jax.vmap(lambda p, t, mp: (f(p, t, mp), d1(p, t, mp), d2(p, t, mp)), in_axes=(0, None, None)),
                   "all": jax.vmap(self.point_all, in_axes=(0, None, None))}

    def _apply(self, kind, pts, t, mp):
        """Evaluate a kernel on ``pts (..., 3)``. With Python-float field parameters (``a_phi``, ``time_scale``, ``w1``) the kernel is
        jitted once per parameter triple (fused, like the step-6 kernels; the Python decisions of :func:`_is_const` are kept); with
        traced parameters the unjitted kernel is traced into the caller's ``jit``."""
        pts = jnp.asarray(pts)
        lead = pts.shape[:-1]
        flags = (mp.a_phi, mp.time_scale, mp.w1)
        if all(isinstance(x, (int, float)) for x in flags):
            key = (kind,) + tuple(float(x) for x in flags)
            fn = self._jit.get(key)
            if fn is None:
                a, ts, w1 = flags
                fn = self._jit[key] = jax.jit(lambda pts_, t_: self._k[kind](pts_, t_, MmsParams(None, None, None, a, ts, w1)))
            out = fn(pts.reshape(-1, 3), jnp.asarray(t))
        else:
            out = self._k[kind](pts.reshape(-1, 3), jnp.asarray(t), mp)
        if isinstance(out, tuple):
            return tuple(o.reshape(lead + o.shape[1:]) for o in out)
        return out.reshape(lead + out.shape[1:])

    def values(self, pts, t, mp):
        """``(..., 5)``."""
        return self._apply("v", pts, t, mp)

    def values_dt(self, pts, t, mp):
        """``(values (..., 5), d/dt (..., 5))``."""
        return self._apply("vdt", pts, t, mp)

    def values_grad(self, pts, t, mp):
        """``(values (..., 5), logical gradients (..., 5, 3))``."""
        return self._apply("vg", pts, t, mp)

    def values_grad_hess(self, pts, t, mp):
        """``(values, gradients (..., 5, 3), Hessians (..., 5, 3, 3))``."""
        return self._apply("vgh", pts, t, mp)

    def all(self, pts, t, mp):
        """``(values, gradients, Hessians, d/dt)`` in one pass."""
        return self._apply("all", pts, t, mp)


FIELDS = MmsFields()
