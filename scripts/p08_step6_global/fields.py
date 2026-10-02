"""The transverse manufactured fields of the P08 step-6 check: values, gradients and Hessians in the computational
disk coordinates ``(u, theta, eta)`` (the coordinates of the P06N state interface), and their owner averages.

Definitions (``x = u cos(theta)``, ``y = u sin(theta)``, ``eta`` the third logical coordinate; the third coordinate enters
as ``om * eta`` with ``om = 2 pi / eta_period``, which is exactly 1 for the pinned period ``2 pi``):

* ``wave``  ``offset + amplitude * {cos | sin}(2 pi x_a / lambda + om eta + eta_phase)`` with ``x_a = x cos(a) + y sin(a)`` the
  coordinate along the direction ``a`` (``cos`` = real part, ``sin`` = imaginary part of ``exp(i(...))``); this is Q's
  plane wave of ``p08_inner_support_eval/q_fields.py`` (``k u cos(theta - a)`` there, the same function) in the same
  ``(u, theta, eta)`` convention as ``P06NState`` (gradient components ``d/du, d/dtheta, d/deta``). The transported
  fields ``n``, ``Te``, ``Ti`` carry ``offset = 1`` and ``amplitude = 0.5``: the production curvature matrix divides by
  ``n`` (``curvature_principal_matrix``: ``4 Te^2 / (3 n)``, ``4 Ti Te / (3 n)``, ``2 B^2 (Te + tau Ti) / n`` with
  ``n_safe = max(n, 1e-30)``), so a zero-mean wave ``n`` (which vanishes on a surface) makes the reference and the
  operator blow up by ~1e30 -- it is not a state of the model; the catalogue fields are likewise ``1 +`` small;
* ``switch_harmonic``  ``amplitude * B(x^2 + y^2) * P_4(x, y) / u_s^4 * cos(om eta + eta_phase)`` with the switch
  ``B(s) = exp(-((s - u_s^2) / w2)^2)`` and the degree-4 harmonic ``P_4 = x^4 - 6 x^2 y^2 + y^4`` (``= u^4 cos 4 theta``,
  ``harmonic = "cos4"``) or ``P_4 = 4 (x^3 y - x y^3)`` (``= u^4 sin 4 theta``, ``"sin4"``). The field is a smooth
  function of ``(x, y)`` (``B`` depends on ``u^2`` only, the harmonic is a polynomial), hence smooth at the axis;
* ``sum``  ``sum_i coef_i * field_i``.

Values, gradients and Hessians are ``jax.jacfwd`` / ``jax.jacfwd(jax.jacfwd)`` of the ``jnp`` definition above in
float64 (exact derivatives of the definition, no finite differences). The ``(u, theta)`` chain rule is applied by the
tracing itself, so ``x = u cos(theta)`` is differentiated, never ``u = sqrt(x^2 + y^2)``: the derivatives are finite at
``u = 0``.

**Owner averages** are computed exactly like the frozen P06N ``owner_values``
(``p06n_field_derived_global.core.observation_chunk`` + ``reduce_observations`` =
``p05n_field_derived_global.operator.live_observations``): ``sum_{raw cells of the owner} raw_volume * f(raw midpoint)
/ owner_volume`` -- the raw-volume-weighted average of the field at the raw-cell midpoints ``t.pts`` (for an owner that
is a single raw cell this is the midpoint value). :func:`owner_average_chunk` accumulates in the same order as the full
``np.add.at`` (ascending raw id within an owner), so a chunked run is bit-identical to the one-shot routine.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import math
import sys

import numpy as np

sys.dont_write_bytecode = True

#: slots of the five-field state, in the order of ``perpendicular_reference_rhs.SLOTS``
SLOTS = ("n", "Te", "Ti", "omega", "phi")

#: the pinned definitions (``configuration.json`` must say exactly this; hashed into the identity)
PINNED_FIELDS = {
    "n": {"kind": "wave", "direction_deg": 0.0, "wavelength": 2.0, "part": "re", "offset": 1.0, "amplitude": 0.5},
    "Te": {"kind": "wave", "direction_deg": 60.0, "wavelength": 2.0, "part": "im", "offset": 1.0, "amplitude": 0.5},
    "Ti": {"kind": "wave", "direction_deg": 120.0, "wavelength": 4.0, "part": "re", "offset": 1.0, "amplitude": 0.5},
    "omega": {"kind": "switch_harmonic", "harmonic": "cos4", "u_s": 0.21, "w2": 0.06, "eta_phase": 0.0},
    "phi_switch": {"kind": "sum", "terms": [
        {"coef": 1.0, "field": {"kind": "switch_harmonic", "harmonic": "sin4", "u_s": 0.21, "w2": 0.06,
                                "eta_phase": 0.3}},
        {"coef": 0.5, "field": {"kind": "wave", "direction_deg": 30.0, "wavelength": 2.0, "part": "re"}}]},
    "phi_wave": {"kind": "wave", "direction_deg": 150.0, "wavelength": 2.0, "part": "re"},
}
#: the pinned field sets: the five physical fields of each slot, in slot order
PINNED_FIELD_SETS = {"transverse": ["n", "Te", "Ti", "omega", "phi_switch"],
                     "transverse_phi_wave": ["n", "Te", "Ti", "omega", "phi_wave"]}

#: points per jitted block (a fixed block size means one compilation per kernel, not one per call shape)
BLOCK = 8192


def _jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return jax, jnp


# ---------------------------------------------------------------------------
# The jnp definitions
# ---------------------------------------------------------------------------
def build_function(spec: dict, om: float):
    """``p = (u, theta, eta) -> scalar`` (a ``jnp`` function of one point) for a field spec."""
    _jax_mod, jnp = _jax()
    kind = spec["kind"]
    amp = float(spec.get("amplitude", 1.0))
    eta_phase = float(spec.get("eta_phase", 0.0))
    if kind == "wave":
        angle = math.radians(float(spec["direction_deg"]))
        ca, sa = math.cos(angle), math.sin(angle)
        k = 2.0 * math.pi / float(spec["wavelength"])
        part = spec["part"]
        if part not in ("re", "im"):
            raise ValueError(f"unknown wave part {part!r}")

        offset = float(spec.get("offset", 0.0))

        def wave(p):
            u, th, eta = p[0], p[1], p[2]
            x, y = u * jnp.cos(th), u * jnp.sin(th)
            phase = k * (x * ca + y * sa) + om * eta + eta_phase
            return offset + amp * (jnp.cos(phase) if part == "re" else jnp.sin(phase))
        return wave
    if kind == "switch_harmonic":
        us, w2, harmonic = float(spec["u_s"]), float(spec["w2"]), spec["harmonic"]
        if harmonic not in ("cos4", "sin4"):
            raise ValueError(f"unknown harmonic {harmonic!r}")

        def switch(p):
            u, th, eta = p[0], p[1], p[2]
            x, y = u * jnp.cos(th), u * jnp.sin(th)
            s = x * x + y * y                                        # u^2, without a square root
            bump = jnp.exp(-((s - us * us) / w2) ** 2)
            poly = (x ** 4 - 6.0 * x * x * y * y + y ** 4) if harmonic == "cos4" else 4.0 * (x ** 3 * y - x * y ** 3)
            return amp * bump * poly / us ** 4 * jnp.cos(om * eta + eta_phase)
        return switch
    if kind == "sum":
        terms = [(float(t["coef"]), build_function(t["field"], om)) for t in spec["terms"]]

        def total(p):
            out = 0.0
            for coef, fn in terms:
                out = out + coef * fn(p)
            return amp * out
        return total
    raise ValueError(f"unknown field kind {kind!r}")


def switch_function(s, u_s: float = 0.21, w2: float = 0.06):
    """``B(s) = exp(-((s - u_s^2) / w2)^2)`` (``s = u^2``) as a ``jnp`` function, for the smoothness tests."""
    _jax_mod, jnp = _jax()
    return jnp.exp(-((s - u_s * u_s) / w2) ** 2)


_KERNELS: dict = {}


def _kernels(specs: tuple, om: float):
    """Jitted ``vmap`` kernels (value / value+gradient / value+gradient+Hessian) of the ordered spec list."""
    key = (json.dumps(specs, sort_keys=True), float(om))
    if key not in _KERNELS:
        jax, jnp = _jax()
        fns = [build_function(s, om) for s in specs]

        def f(p):
            return jnp.stack([fn(p) for fn in fns])

        jac = jax.jacfwd(f)
        hes = jax.jacfwd(jac)
        _KERNELS[key] = {
            "v": jax.jit(jax.vmap(f)),
            "vg": jax.jit(jax.vmap(lambda p: (f(p), jac(p)))),
            "vgh": jax.jit(jax.vmap(lambda p: (f(p), jac(p), hes(p))))}
    return _KERNELS[key]


def _blocked(fn, points: np.ndarray, block: int = BLOCK):
    """Apply a jitted per-block kernel to ``points (Q, 3)``; the last block is padded with its last point."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    q = len(pts)
    if q == 0:
        raise ValueError("no points")
    outs = None
    for start in range(0, q, block):
        chunk = pts[start:start + block]
        pad = block - len(chunk)
        if pad:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)])
        res = fn(chunk)
        res = res if isinstance(res, tuple) else (res,)
        res = [np.asarray(r)[:block - pad] for r in res]
        outs = [[r] for r in res] if outs is None else [o + [r] for o, r in zip(outs, res)]
    merged = [np.concatenate(o) for o in outs]
    return merged if len(merged) > 1 else merged[0]


class TransverseState:
    """A five-field manufactured state (slots ``n, Te, Ti, omega, phi``) with the ``P06NState`` interface:
    ``values_gradients(points) -> (values (5, Q), gradients (5, Q, 3))`` and ``values_gradients_hessians(points) ->
    (..., hessians (5, Q, 3, 3))``; ``points`` are ``(Q, 3)`` in ``(u, theta, eta)``."""

    def __init__(self, names, specs, om: float):
        self.fields = tuple(names)
        self.slots = SLOTS
        self.om = float(om)
        self._kernels = _kernels(tuple(specs), self.om)

    def values(self, points):
        return np.asarray(_blocked(self._kernels["v"], points)).T

    def values_gradients(self, points):
        v, g = _blocked(self._kernels["vg"], points)
        return np.asarray(v).T, np.moveaxis(np.asarray(g), 0, 1)

    def values_gradients_hessians(self, points):
        v, g, h = _blocked(self._kernels["vgh"], points)
        return np.asarray(v).T, np.moveaxis(np.asarray(g), 0, 1), np.moveaxis(np.asarray(h), 0, 1)


class ColumnSet:
    """The physical columns of the campaign (``configuration.json["fields"]`` in order) and the five-slot states of the
    field sets built from them. ``period`` is the eta period of the environment's grid."""

    def __init__(self, fields: dict, field_sets: dict, period: float):
        self.names = tuple(fields)
        self.specs = {name: fields[name] for name in self.names}
        self.field_sets = {fs: tuple(names) for fs, names in field_sets.items()}
        for fs, names in self.field_sets.items():
            if len(names) != len(SLOTS) or any(n not in self.specs for n in names):
                raise ValueError(f"field set {fs!r} must name {len(SLOTS)} known fields, got {names}")
        self.period = float(period)
        self.om = 2.0 * math.pi / self.period
        self._all = TransverseState(self.names, [self.specs[n] for n in self.names], self.om)

    def column_index(self, field_set: str) -> list:
        return [self.names.index(n) for n in self.field_sets[field_set]]

    def values(self, points) -> np.ndarray:
        """``(Q, n_columns)`` values of every physical column."""
        return self._all.values(points).T

    def state(self, field_set: str) -> TransverseState:
        names = self.field_sets[field_set]
        return TransverseState(names, [self.specs[n] for n in names], self.om)

    def states(self) -> dict:
        return {fs: self.state(fs) for fs in self.field_sets}


# ---------------------------------------------------------------------------
# Owner averages (the frozen P06N routine, chunked)
# ---------------------------------------------------------------------------
def owner_average_chunk(t, values_fn, start: int, stop: int) -> np.ndarray:
    """``(stop - start, n_columns)`` owner averages of the owners ``start .. stop - 1``: ``sum raw_volume * f(raw
    midpoint) / owner_volume`` over their raw cells (``t.order[t.starts[start]:t.starts[stop]]`` -- stable by raw id).
    ``values_fn(points (Q, 3)) -> (Q, n_columns)`` (``ColumnSet.values``, or the P06N catalogue for the oracle check).
    Identical, bit for bit, to the rows ``start:stop`` of ``live_observations(t, values at t.pts)``."""
    raw_ids = np.asarray(t.order[int(t.starts[start]):int(t.starts[stop])], dtype=np.int64)
    values = np.asarray(values_fn(t.pts[raw_ids]), dtype=np.float64)
    sums = np.zeros((stop - start, values.shape[1]))
    np.add.at(sums, np.asarray(t.ro[raw_ids], dtype=np.int64) - start, t.rv[raw_ids, None] * values)
    return sums / np.asarray(t.vol[start:stop])[:, None]


def owner_average(t, values_fn) -> np.ndarray:
    """All owners at once with ``live_observations``' formula (reference for tests / the bounded preflight)."""
    values = np.asarray(values_fn(t.pts), dtype=np.float64)
    sums = np.zeros((len(t.vol), values.shape[1]))
    np.add.at(sums, t.ro, t.rv[:, None] * values)
    return sums / t.vol[:, None]


def owner_average_exact_sum(t, values_fn, owners) -> np.ndarray:
    """Independent check of :func:`owner_average_chunk` for a few owners: a per-owner loop over the members with
    ``math.fsum`` (exactly rounded sums) instead of ``np.add.at``."""
    out = None
    for i, o in enumerate(owners):
        members = t.order[int(t.starts[o]):int(t.starts[o + 1])]
        values = np.asarray(values_fn(t.pts[members]), dtype=np.float64)
        if out is None:
            out = np.empty((len(owners), values.shape[1]))
        for c in range(values.shape[1]):
            out[i, c] = math.fsum(float(w) * float(v) for w, v in zip(t.rv[members], values[:, c])) / float(t.vol[o])
    return out


# ---------------------------------------------------------------------------
# Derivative checks (bounded preflight / tests)
# ---------------------------------------------------------------------------
def fd_defects(state: TransverseState, points, step: float = 1e-5) -> dict:
    """Central-difference defects at ``points (Q, 3)``: the gradient against differences of the values, the Hessian
    against differences of the gradients, and the Hessian's asymmetry; each ``max |fd - exact| / (1 + |exact|)``
    (``hess[slot, q, a, b] = d_a d_b f``, so ``fd[a, b] = d_a (grad_b)``)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    _v, g, h = state.values_gradients_hessians(pts)
    g_fd, h_fd = [], []
    for a in range(3):
        d = np.zeros(3)
        d[a] = step
        g_fd.append((state.values(pts + d) - state.values(pts - d)) / (2.0 * step))
        h_fd.append((state.values_gradients(pts + d)[1] - state.values_gradients(pts - d)[1]) / (2.0 * step))
    g_fd = np.stack(g_fd, axis=-1)                                   # (5, Q, 3)
    h_fd = np.stack(h_fd, axis=2)                                    # (5, Q, 3 [d_a], 3 [grad_b])
    return {"gradient_fd_rel": float(np.max(np.abs(g_fd - g) / (1.0 + np.abs(g)))),
            "hessian_fd_rel": float(np.max(np.abs(h_fd - h) / (1.0 + np.abs(h)))),
            "hessian_asymmetry": float(np.max(np.abs(h - np.swapaxes(h, 2, 3)) / (1.0 + np.abs(h)))),
            "max_abs_value": float(np.max(np.abs(_v))), "max_abs_gradient": float(np.max(np.abs(g))),
            "max_abs_hessian": float(np.max(np.abs(h)))}
