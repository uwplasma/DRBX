"""Exact field catalogues of the nodal Laplacian gates: values, logical gradients and Hessians, and the exact Laplacian.

Every catalogue is evaluated at points ``(Q, 3)`` in the logical frame ``(u, theta, eta)`` and returns
``values (F, Q)``, ``gradients (F, Q, 3)``, ``hessians (F, Q, 3, 3)``:

* ``transverse`` (gating, Dirichlet) / ``transverse_wave`` (control): the five pinned fields ``n, Te, Ti, omega, phi`` of
  ``p08_step6_global.fields`` (``phi_switch`` / ``phi_wave``) plus the derived ``q = n Ti`` and ``psi = phi + tau q``: the
  catalogue of the Laplacian prototype (``work/p09_laplacian_20261004/hsx_run.py``: ``n, Te, Ti, omega, psi``);
* ``p07n``: the P07N catalogue ``field_b1, field_e3, field_e12, heldout_field_b2, constant``
  (``p07n_field_derived_global.fields``), Dirichlet and Neumann (conormal and physical-normal) data;
* ``p06n:<variant>``: the five slots of a P06N case plus ``q`` and ``psi`` (``p06n_field_derived_global.fields``).

The exact Laplacian at the nodes is ``R = (divA_j d_j f + A_ij d_i d_j f) / |J|`` with the extracted ``A = |J| P_perp``
and ``divA_j = d_i A^{ij}`` (:class:`common.MetricData`), i.e. ``div(P_perp grad f)`` per unit volume.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import common as C                                                                      # noqa: E402

PERIOD = C.PERIOD
TAU = float(C.CONFIG["tau_polarization"])
SLOTS = ("n", "Te", "Ti", "omega", "phi")
DERIVED = ("q", "psi")
P07N_NAMES = tuple(C.CONFIG["catalogues"]["p07n"]["fields"])
GATE_TRANSVERSE = tuple(C.CONFIG["catalogues"]["transverse"]["fields"])           # n, Te, Ti, omega, psi
CATALOGUE_PATH = SCRIPTS / "p06n_field_derived_global" / "p06n_catalogue.json"


def add_derived(V, G, H, tau: float = TAU):
    """Append ``q = n Ti`` and ``psi = phi + tau q`` to the stacks of the slots ``(n, Te, Ti, omega, phi)``."""
    n_, Ti, ph = 0, 2, 4
    q_v = V[n_] * V[Ti]
    q_g = V[Ti][:, None] * G[n_] + V[n_][:, None] * G[Ti]
    q_h = (V[Ti][:, None, None] * H[n_] + V[n_][:, None, None] * H[Ti]
           + np.einsum("qa,qb->qab", G[n_], G[Ti]) + np.einsum("qa,qb->qab", G[Ti], G[n_]))
    psi_v, psi_g, psi_h = V[ph] + tau * q_v, G[ph] + tau * q_g, H[ph] + tau * q_h
    return (np.concatenate([V, q_v[None], psi_v[None]]), np.concatenate([G, q_g[None], psi_g[None]]),
            np.concatenate([H, q_h[None], psi_h[None]]))


class Catalogue:
    """``names`` (F), per-field boundary ``kinds`` and ``evaluate(points) -> (V, G, H)`` (chunked)."""

    def __init__(self, name, names, evaluate, kinds=None, chunk=32768):
        self.name, self.names, self._eval, self.chunk = name, tuple(names), evaluate, chunk
        self.kinds = tuple(kinds) if kinds is not None else ("dirichlet",) * len(self.names)

    def evaluate(self, points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        vs, gs, hs = [], [], []
        for s in range(0, len(pts), self.chunk):
            v, g, h = self._eval(pts[s:s + self.chunk])
            vs.append(v); gs.append(g); hs.append(h)
        return np.concatenate(vs, 1), np.concatenate(gs, 1), np.concatenate(hs, 1)


def _transverse(which: str) -> Catalogue:
    from p08_step6_global import fields as t6

    key = {"transverse": "transverse", "transverse_wave": "transverse_phi_wave"}[which]
    names = t6.PINNED_FIELD_SETS[key]
    state = t6.TransverseState(names, [t6.PINNED_FIELDS[k] for k in names], 2.0 * np.pi / PERIOD)

    def ev(q):
        v, g, h = state.values_gradients_hessians(q)
        return add_derived(np.asarray(v), np.asarray(g), np.asarray(h))

    return Catalogue(which, SLOTS + DERIVED, ev)


def _p07n(names=P07N_NAMES) -> Catalogue:
    import p07n_field_derived_global.fields as pf

    def ev(q):
        vs, gs, hs = [], [], []
        for name in names:
            v, g, h = pf.evaluate(None, q, name, PERIOD)
            vs.append(np.asarray(v, dtype=np.float64).reshape(len(q))); gs.append(np.asarray(g, dtype=np.float64).reshape(len(q), 3))
            hs.append(np.asarray(h, dtype=np.float64).reshape(len(q), 3, 3))
        return np.stack(vs), np.stack(gs), np.stack(hs)

    return Catalogue("p07n", names, ev)


def p06n_variant(variant: str):
    """``(Catalogue, spec)`` of a P06N case: slots + ``q``, ``psi``; ``kinds`` from the ``:dirichlet`` suffix of the spec
    (a slot without the suffix is a Neumann state; the derived ``q`` and ``psi`` are Dirichlet)."""
    import p06n_field_derived_global.fields as p06f

    cat = json.loads(CATALOGUE_PATH.read_text())
    spec = cat["cases"][variant]
    fields = [spec[s].split(":")[0] for s in SLOTS]
    kinds = ["dirichlet" if spec[s].endswith(":dirichlet") else "neumann" for s in SLOTS] + ["dirichlet", "dirichlet"]

    def ev(q):
        vs, gs, hs = [], [], []
        for name in fields:
            v, g, h = p06f.evaluate(None, q, name, PERIOD)
            vs.append(np.asarray(v, dtype=np.float64).reshape(len(q))); gs.append(np.asarray(g, dtype=np.float64).reshape(len(q), 3))
            hs.append(np.asarray(h, dtype=np.float64).reshape(len(q), 3, 3))
        return add_derived(np.stack(vs), np.stack(gs), np.stack(hs))

    return Catalogue(f"p06n:{variant}", SLOTS + DERIVED, ev, kinds), spec


def make_catalogue(name: str) -> Catalogue:
    if name in ("transverse", "transverse_wave"):
        return _transverse(name)
    if name == "p07n":
        return _p07n()
    raise ValueError(f"unknown catalogue {name!r}")


# ---------------------------------------------------------------------------------------------------------------------
# exact data at the nodes and the wall
# ---------------------------------------------------------------------------------------------------------------------
def exact_laplacian(md, V, G, H):
    """``R (F, E, P) = (divA_j d_j f + A_ij d_i d_j f) / |J|`` from the extracted metric (``V`` etc. in flat point order)."""
    E, P = md.layout.n_eta, md.layout.P
    F = V.shape[0]
    g = G.reshape(F, E, P, 3)
    h = H.reshape(F, E, P, 3, 3)
    num = np.einsum("epj,fepj->fep", md.divA, g) + np.einsum("epij,fepij->fep", md.A, h)
    return num / md.J[None]


class NodalData:
    """Exact data of a catalogue on the nodes and the wall: ``V (F, E, P)``, ``G``, ``R`` and the wall datum arrays
    ``wall_value (E, N, F)``, ``wall_conormal (E, N, F)`` (``(A grad f)^u``), ``wall_normal (E, N, F)`` (``n . grad f``)."""

    def __init__(self, md, cat: Catalogue):
        self.cat, self.names, self.md = cat, cat.names, md
        E, P = md.layout.n_eta, md.layout.P
        V, G, H = cat.evaluate(md.points.reshape(-1, 3))
        F = len(cat.names)
        self.V = V.reshape(F, E, P)
        self.R = exact_laplacian(md, V, G, H)
        wp = md.wall_points.reshape(-1, 3)
        Vw, Gw, _ = cat.evaluate(wp)
        Nw = md.wall_points.shape[1]
        Gw = Gw.reshape(F, E, Nw, 3)
        self.wall_value = np.moveaxis(Vw.reshape(F, E, Nw), 0, -1)
        self.wall_conormal = np.moveaxis(np.einsum("enj,fenj->fen", md.wall_A[..., 0, :], Gw), 0, -1)
        a = md.wall_ginv_u / np.sqrt(md.wall_ginv_u[..., 0:1])
        self.wall_normal = np.moveaxis(np.einsum("enj,fenj->fen", a, Gw), 0, -1)

    def field(self, name):
        return self.names.index(name)
