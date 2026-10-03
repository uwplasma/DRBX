"""Time-dependent transverse manufactured fields of the P09 evolved MMS: ``f(p, t)`` with ``p = (u, theta, eta)``.

The step-6 fields (:data:`p08_step6_global.fields.PINNED_FIELDS`) with three optional time parameters per field spec
(missing or zero means static):

* ``wave``            ``offset + amp (1 + a_mod sin(nu t)) {cos | sin}(k x_a + om eta + eta_phase - Omega t)``;
* ``switch_harmonic`` ``amp (1 + a_mod sin(nu t)) B(u^2) P_4(x, y) / u_s^4 cos(om eta + eta_phase - Omega t)``;
* ``sum``             ``amp (1 + a_mod sin(nu t)) sum_i coef_i f_i(p, t)`` (the terms carry their own ``Omega``).

``Omega`` / ``a_mod`` are decided in Python (a zero parameter adds no operation), so a spec without time parameters
traces to exactly the step-6 expression: bitwise the pinned fields at every ``t``, and the full spec at ``t = 0`` differs
from them only in the (unit) amplitude factor. Derivatives are ``jax.jacfwd`` of the definition (no finite differences);
``n, Te, Ti`` stay ``>= 1 - 0.5 (1 + a_mod) = 0.4``. Everything is traceable (``t`` may be a tracer inside ``jit``).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import copy
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.dont_write_bytecode = True

from p08_step6_global.fields import PINNED_FIELD_SETS, PINNED_FIELDS, SLOTS      # noqa: E402,F401

#: the five physical columns of the evolved MMS, in slot order ``n, Te, Ti, omega, phi``
FIELD_SET = PINNED_FIELD_SETS["transverse"]
#: time parameters per field (top-level of the spec; ``phi_switch`` carries ``a_mod, nu`` on the sum and ``Omega`` per term)
TIME_PARAMS = {"n": {"Omega": 30.0, "a_mod": 0.2, "nu": 40.0}, "Te": {"Omega": -25.0, "a_mod": 0.2, "nu": 35.0},
               "Ti": {"Omega": 20.0, "a_mod": 0.2, "nu": 45.0}, "omega": {"Omega": 35.0, "a_mod": 0.2, "nu": 30.0},
               "phi_switch": {"a_mod": 0.2, "nu": 50.0}}
TERM_OMEGA = {"phi_switch": (28.0, -22.0)}


def time_fields() -> dict:
    """The pinned step-6 field specs with the P09 time parameters added (a deep copy)."""
    out = copy.deepcopy(PINNED_FIELDS)
    for name, par in TIME_PARAMS.items():
        out[name].update(par)
    for name, omegas in TERM_OMEGA.items():
        for term, om_t in zip(out[name]["terms"], omegas):
            term["field"]["Omega"] = om_t
    return out


def build_function(spec: dict, om: float):
    """``(p (3,), t) -> scalar`` ``jnp`` function of a field spec (``om = 2 pi / eta_period``)."""
    kind = spec["kind"]
    amp = float(spec.get("amplitude", 1.0))
    eta_phase = float(spec.get("eta_phase", 0.0))
    a_mod, nu, big_om = float(spec.get("a_mod", 0.0)), float(spec.get("nu", 0.0)), float(spec.get("Omega", 0.0))

    def amplitude(t):
        return amp * (1.0 + a_mod * jnp.sin(nu * t)) if a_mod else amp

    if kind == "wave":
        angle = math.radians(float(spec["direction_deg"]))
        ca, sa = math.cos(angle), math.sin(angle)
        k = 2.0 * math.pi / float(spec["wavelength"])
        part = spec["part"]
        if part not in ("re", "im"):
            raise ValueError(f"unknown wave part {part!r}")
        offset = float(spec.get("offset", 0.0))

        def wave(p, t):
            u, th, eta = p[0], p[1], p[2]
            x, y = u * jnp.cos(th), u * jnp.sin(th)
            phase = k * (x * ca + y * sa) + om * eta + eta_phase
            if big_om:
                phase = phase - big_om * t
            return offset + amplitude(t) * (jnp.cos(phase) if part == "re" else jnp.sin(phase))
        return wave
    if kind == "switch_harmonic":
        us, w2, harmonic = float(spec["u_s"]), float(spec["w2"]), spec["harmonic"]
        if harmonic not in ("cos4", "sin4"):
            raise ValueError(f"unknown harmonic {harmonic!r}")

        def switch(p, t):
            u, th, eta = p[0], p[1], p[2]
            x, y = u * jnp.cos(th), u * jnp.sin(th)
            s = x * x + y * y
            bump = jnp.exp(-((s - us * us) / w2) ** 2)
            poly = (x ** 4 - 6.0 * x * x * y * y + y ** 4) if harmonic == "cos4" else 4.0 * (x ** 3 * y - x * y ** 3)
            phase = om * eta + eta_phase
            if big_om:
                phase = phase - big_om * t
            return amplitude(t) * bump * poly / us ** 4 * jnp.cos(phase)
        return switch
    if kind == "sum":
        terms = [(float(t["coef"]), build_function(t["field"], om)) for t in spec["terms"]]

        def total(p, t):
            out = 0.0
            for coef, fn in terms:
                out = out + coef * fn(p, t)
            return amplitude(t) * out
        return total
    raise ValueError(f"unknown field kind {kind!r}")


class TimeState:
    """Batched kernels of an ordered spec list (default: the P09 ``[n, Te, Ti, omega, phi_switch]``).

    ``f(p, t) -> (F,)``; per-point ``value_grad_dt(p, t) -> (f (F,), df/dp (F, 3), df/dt (F,))`` and
    ``grad_dt(p, t) -> d/dt df/dp (F, 3)``; the ``batch_*`` functions ``vmap`` them over ``P (Q, 3)`` with a scalar ``t``
    (traceable, to be used inside the caller's ``jit``); the ``jit_*`` ones are jitted wrappers for host use."""

    def __init__(self, specs=None, om: float = 1.0):
        if specs is None:
            specs = [time_fields()[n] for n in FIELD_SET]
        self.om = float(om)
        fns = [build_function(s, self.om) for s in specs]
        self.f = lambda p, t: jnp.stack([fn(p, t) for fn in fns])
        jac = jax.jacfwd(self.f, argnums=0)
        dt = jax.jacfwd(self.f, argnums=1)
        self.value_grad_dt = lambda p, t: (self.f(p, t), jac(p, t), dt(p, t))
        self.grad_dt = jax.jacfwd(jac, argnums=1)
        self.batch_value = jax.vmap(self.f, in_axes=(0, None))
        self.batch_value_grad = jax.vmap(lambda p, t: (self.f(p, t), jac(p, t)), in_axes=(0, None))
        self.batch_value_grad_dt = jax.vmap(self.value_grad_dt, in_axes=(0, None))
        self.batch_grad_dt = jax.vmap(self.grad_dt, in_axes=(0, None))
        self.jit_value = jax.jit(self.batch_value)
        self.jit_value_grad = jax.jit(self.batch_value_grad)
        self.jit_value_grad_dt = jax.jit(self.batch_value_grad_dt)
        self.jit_grad_dt = jax.jit(self.batch_grad_dt)


def blocked(fn, points, t, block: int = 8192):
    """Host helper: apply a jitted batch kernel ``fn(P, t)`` over ``points (Q, 3)`` in fixed-size blocks (one compilation);
    returns a tuple of ``np`` arrays (or one array) with leading axis ``Q``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    outs = None
    for start in range(0, len(pts), block):
        chunk = pts[start:start + block]
        pad = block - len(chunk)
        if pad:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)])
        res = fn(chunk, t)
        res = res if isinstance(res, tuple) else (res,)
        res = [np.asarray(r)[:block - pad] for r in res]
        outs = [[r] for r in res] if outs is None else [o + [r] for o, r in zip(outs, res)]
    merged = [np.concatenate(o) for o in outs]
    return tuple(merged) if len(merged) > 1 else merged[0]
