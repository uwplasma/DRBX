"""Exact field catalogues for the nodal bracket gates, as one adapter: values and logical gradients at points.

Every catalogue is a :class:`FieldSet`: ``evaluate(points (Q, 3)) -> (values (F, Q), gradients (F, Q, 3))`` with the
gradients in the logical frame ``(d/du, d/dtheta, d/deta)``, plus the list of bracket pairs ``(a, b)`` (``a`` the potential
like field, ``b`` the advected one; orientation of ``fci_perpendicular_midpoint_bracket``).

* ``p05``: ``p05_structured_global.numerics.fields(ref, p, full_omega_gradient=True)`` (needs the frozen reference ``ref``
  for the actual vorticity and its finite-difference gradient), pairs ``PAIRS`` labelled by ``CASES``;
* ``p05n``: the analytic fields of ``p05n_field_derived_global`` (no reference needed) with the pairings of
  ``p05n_catalogue.json`` (Neumann- and Dirichlet-structured generators; both kinds use the manufactured trace here);
* ``switch`` (gating) / ``wave`` (control): the transverse manufactured fields of ``p08_step6_global.fields``.
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
CONFIG = json.loads((HERE / "configuration.json").read_text())
PERIOD = float(CONFIG["eta_period"])
SET_NAMES = ("p05", "p05w", "p05n", "switch", "wave")


@dataclass(frozen=True)
class Pair:
    label: str
    group: str            # "gate" or a catalogue group ("main", "heldout", "controls", "report")
    a: str
    b: str
    gate: bool
    constant: bool        # the bracket is identically zero (constant control)


@dataclass
class FieldSet:
    name: str
    names: tuple
    pairs: tuple
    evaluator: Callable
    chunk: int = 8192
    wall_evaluator: Callable | None = None       # values/gradients usable at u = 1 (no radial derivative of omega)
    progress: bool = False
    value_evaluator: Callable | None = None      # cheap values-only evaluation (all planes) when gradients are subsampled
    subset_planes: int | None = None

    def evaluate(self, points, wall: bool = False, values_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        fn = self.wall_evaluator if (wall and self.wall_evaluator is not None) else self.evaluator
        if values_only and self.value_evaluator is not None:
            fn = self.value_evaluator
        vs, gs = [], []
        t0 = time.perf_counter()
        for s in range(0, len(pts), self.chunk):
            v, g = fn(pts[s:s + self.chunk])
            vs.append(np.asarray(v)); gs.append(np.asarray(g))
            if self.progress:
                done = min(s + self.chunk, len(pts))
                print(f"[fields {self.name}] {done}/{len(pts)} points, {time.perf_counter() - t0:.0f} s", flush=True)
        return np.concatenate(vs, axis=1), np.concatenate(gs, axis=1)

    def pairs_json(self) -> str:
        return json.dumps([p.__dict__ for p in self.pairs])


class _OmegaOff:
    """Context manager replacing ``p05_structured_global.numerics.omega`` by zeros (fields that do not use it)."""

    def __init__(self, num):
        self.num = num

    def __enter__(self):
        self.saved = self.num.omega
        self.num.omega = lambda ref, p: np.zeros(len(p))

    def __exit__(self, *exc):
        self.num.omega = self.saved


def p05_set(ref, step: float | None = None, which: str = "analytic") -> FieldSet:
    """The P05 catalogue on the frozen reference ``ref`` (``step`` overrides ``ref.finite_difference_step``).

    ``which="analytic"``: the seven pairs that do not involve the actual vorticity; omega is switched off (zeros), so the
    evaluation is cheap. ``which="omega"``: the single pair ``(phi_mms, actual_vorticity)``; ``evaluator`` includes the
    finite-difference omega gradient (12 extra omega evaluations per point), ``value_evaluator`` only the omega value;
    ``subset_planes`` says how many eta planes the gradient (hence the reference bracket) is evaluated on.
    """
    from p05_structured_global import numerics as num

    if step is not None:
        ref.finite_difference_step = step

    def conv(v, g):
        return v.T, np.moveaxis(g, 2, 0)

    def ev_omega(p):
        return conv(*num.fields(ref, p, full_omega_gradient=True))

    def ev_value(p):
        return conv(*num.fields(ref, p, omega_gradient=False))

    def ev_wall(p):
        return conv(*num.fields(ref, p, omega_gradient=True))        # tangential omega gradient only (valid at u = 1)

    def ev_analytic(p):
        with _OmegaOff(num):
            return conv(*num.fields(ref, p))

    const = {(0, 6)}
    entries = list(zip(num.CASES, num.PAIRS))
    if which == "analytic":
        pairs = tuple(Pair(case, "gate", num.FIELDS[a], num.FIELDS[b], (a, b) not in const, (a, b) in const)
                      for case, (a, b) in entries if (a, b) != (0, 1))
        return FieldSet("p05", tuple(num.FIELDS), pairs, ev_analytic, chunk=8192, wall_evaluator=ev_analytic)
    if which == "omega":
        case, (a, b) = entries[0]
        pair = Pair(case, "gate", num.FIELDS[a], num.FIELDS[b], True, False)
        return FieldSet("p05w", tuple(num.FIELDS), (pair,), ev_omega, chunk=2048, wall_evaluator=ev_wall,
                        progress=True, value_evaluator=ev_value, subset_planes=int(CONFIG["p05w_planes"]))
    raise ValueError(which)


def p05n_set() -> FieldSet:
    import p05n_field_derived_global.fields as pf

    cat = json.loads((SCRIPTS / "p05n_field_derived_global" / "p05n_catalogue.json").read_text())
    pairs, names = [], []
    for kind, groups in cat["pairings"].items():
        for group, plist in groups.items():
            for a, b in plist:
                a0, b0 = a.split(":")[0], b.split(":")[0]
                const = "constant" in (a0, b0)
                pairs.append(Pair(f"{kind}/{group}/{a}|{b}", group, a0, b0, group in ("main", "heldout") and not const, const))
                names += [a0, b0]
    names = tuple(dict.fromkeys(names))

    def ev(q):
        vs, gs = [], []
        for name in names:
            v, g, _H = pf.evaluate(None, q, name, PERIOD)
            vs.append(np.asarray(v, dtype=np.float64).reshape(len(q))); gs.append(np.asarray(g, dtype=np.float64).reshape(len(q), 3))
        return np.stack(vs), np.stack(gs)

    return FieldSet("p05n", names, tuple(pairs), ev, chunk=16384)


def transverse_set(which: str) -> FieldSet:
    """``switch`` (gating) or ``wave`` (control): the potential against ``n, Te, Ti, omega``."""
    from p08_step6_global import fields as t6

    key = {"switch": "transverse", "wave": "transverse_phi_wave"}[which]
    names = tuple(t6.PINNED_FIELD_SETS[key])
    state = t6.TransverseState(names, [t6.PINNED_FIELDS[n] for n in names], 2.0 * np.pi / PERIOD)
    phi = names[4]
    gate = which == "switch"
    pairs = tuple(Pair(f"{which}/{phi}|{b}", "gate" if gate else "report", phi, b, gate, False) for b in names[:4])
    return FieldSet(which, names, pairs, state.values_gradients, chunk=8192)


def make_set(name: str, ref=None) -> FieldSet:
    if name in ("p05", "p05w"):
        if ref is None:
            raise ValueError("the p05 catalogue needs the frozen reference")
        return p05_set(ref, which="analytic" if name == "p05" else "omega")
    if name == "p05n":
        return p05n_set()
    if name in ("switch", "wave"):
        return transverse_set(name)
    raise ValueError(f"unknown field set {name!r}")
