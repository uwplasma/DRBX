"""Geometry bundle of the nodal SBP perpendicular stack: the two metric extractions joined into one verified, saveable object.

The nodal stack takes two separate logical-frame metric extractions of one geometry (family-A layout):

* the bracket / curvature extraction ``nodal`` (:class:`~drbx.stencils.nodal_plan.NodalMetric`: ``h, |J|, B, K``);
* the Laplacian extraction ``lap`` (``A = |J| P_perp``, ``d_i A^{ij}``, ``|J|``, ``g^{u j}`` at the nodes and at the wall points).

:func:`assemble_nodal_bundle` joins them: it asserts that the arm identity, ``P`` / ``n_eta``, the nodes, the wall points, ``|J|`` and
the norm ``Hp`` of the two plans agree (to ``1e-13`` for the points, ``1e-12`` relative for ``|J|`` and ``Hp``), builds the
:class:`~drbx.stencils.nodal_plan.NodalPlan`, the :class:`~drbx.geometry.sbp_laplacian.LaplacianPlan` and the
:class:`~drbx.native.fci_nodal_perpendicular_rhs.NodalPerpendicularContext` (core-Schur preconditioner, float32 factors by
default: :data:`~drbx.native.fci_nodal_perpendicular_model.NODAL_PERPENDICULAR_DEFAULTS`), and records everything in an identity.

:class:`NodalGeometryBundle` is a pytree (a ``jit`` argument): ``ctx`` and the reference arrays :class:`NodalReferenceArrays` are
the traced data, the identity and the layout are static (identity-compared, so a bundle is one compilation key).

**Identity** (``bundle.identity``, also ``identity.json`` of a saved bundle): ``schema`` (caller supplied, so every campaign keeps its
own), ``arm`` (an opaque label), ``n``, ``n_eta``, ``P``, ``N_wall``, ``family`` (``"A"``), ``arm_identity`` (the ``eta_filter``
option of the metric extractions, ``None`` for an unfiltered one), the layout sha, both plan shas, the ``|J|`` / ``Hp`` / layout
``agreement`` report, ``arrays_sha256`` (the stored reference arrays), the caller's provenance entries, ``bundle_sha256`` (a hash of
everything but the volatile entries :data:`VOLATILE_IDENTITY_KEYS`: the git commit, the preconditioner timings, file paths) and
``git_commit``.

**Portable identity.** The layout and plan shas hash float bytes whose Gauss weights come from LAPACK, so they differ by ulps between
platforms (a bundle built on Linux does not reproduce them on macOS). They are provenance, never a gate: a stored layout sha that
differs from the rebuilt one is recorded in ``agreement["layout"]``, the layout agreement is checked to tolerance (nodes, wall
points, ``|J|``, ``Hp``), and :func:`load_nodal_bundle` keeps the stored identity of a bundle whose rebuilt plan / layout shas
differ and records the rebuilt ones under ``identity["rebuilt"]``. What is verified bit for bit is the stored data:
``arrays_sha256`` of ``bundle.npz`` and the consistency of ``identity.json`` with its ``bundle_sha256``.

``save_nodal_bundle`` writes ``<out>/bundle.npz`` (the reference arrays) and ``<out>/identity.json``; ``load_nodal_bundle`` rebuilds
both plans (deterministic given the arrays) and the preconditioner.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from drbx.native.fci_nodal_perpendicular_model import NODAL_PERPENDICULAR_DEFAULTS

__all__ = [
    "NODAL_BUNDLE_SCHEMA", "VOLATILE_IDENTITY_KEYS", "NodalReferenceArrays", "NodalBundleStatic", "NodalGeometryBundle",
    "sha256_arrays", "layout_identity", "laplacian_plan_identity", "bundle_sha", "jsonable", "reference_arrays_numpy",
    "assemble_nodal_bundle", "save_nodal_bundle", "load_nodal_bundle",
]

NODAL_BUNDLE_SCHEMA = "drbx.nodal-geometry-bundle-v1"
#: identity entries excluded from ``bundle_sha256`` (they do not change the geometry)
VOLATILE_IDENTITY_KEYS = ("git_commit", "preconditioner", "bundle_sha256", "created", "nodal_metric_path", "laplacian_metric_path",
                          "rebuilt")
_PLAN_SHA_KEYS = ("layout_sha256", "nodal_plan_sha256", "laplacian_plan_sha256")


# ---------------------------------------------------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------------------------------------------------
def sha256_arrays(arrays: dict, extra: str = "") -> str:
    """sha256 over ``(name, dtype, shape, bytes)`` of every array (sorted by name) and an extra string."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype}|{a.shape}|".encode())
        h.update(a.tobytes())
    h.update(extra.encode())
    return h.hexdigest()


def layout_identity(layout) -> str:
    """sha256 over the family-A layout (``n``, planes, node coordinates, ``H`` weights). Platform dependent (see the module
    docstring)."""
    return sha256_arrays({"node_u": layout.node_u, "node_theta": layout.node_theta, "wxy": layout.wxy,
                          "node_ring": np.asarray(layout.node_ring)},
                         extra=json.dumps(dict(n=layout.n, E=layout.n_eta, P=layout.P, deta=float(layout.deta),
                                               du=float(layout.du), family="A"), sort_keys=True))


def laplacian_plan_identity(plan) -> str:
    """sha256 over every array of a :class:`~drbx.geometry.sbp_laplacian.LaplacianPlan` (name, dtype, shape, bytes) and its
    structure (a plan without evaluated faces keeps its pre-faces identity)."""
    arrays = {f.name: np.asarray(getattr(plan, f.name)) for f in dataclasses.fields(plan)
              if f.name != "structure" and getattr(plan, f.name) is not None}
    st = dataclasses.asdict(plan.structure)
    if not st.get("evaluated"):
        st.pop("evaluated", None)
    return sha256_arrays(arrays, json.dumps(st, sort_keys=True))


def bundle_sha(identity: dict) -> str:
    """The ``bundle_sha256`` of an identity: a hash of every entry except :data:`VOLATILE_IDENTITY_KEYS`."""
    keep = {k: v for k, v in identity.items() if k not in VOLATILE_IDENTITY_KEYS}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def jsonable(o):
    """``o`` with numpy scalars / arrays converted for ``json`` (unknown objects become ``str``)."""
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------------------------------------------------
# the bundle
# ---------------------------------------------------------------------------------------------------------------------
class NodalReferenceArrays(NamedTuple):
    """Nodal reference arrays (logical frame ``(u, theta, eta)``, ``E`` planes, ``P`` nodes, ``N`` wall points per plane)."""

    points: jax.Array          # (E, P, 3) node coordinates
    wall_points: jax.Array     # (E, N, 3) wall lattice (``wall_points(layout, layout.walls[0])``)
    h: jax.Array               # (E, P, 3) b_cov / B
    jac: jax.Array             # (E, P) |J| (one array for the bracket, the curvature and the Laplacian)
    B: jax.Array               # (E, P)
    K: jax.Array               # (E, P, 3) logical curvature
    A: jax.Array               # (E, P, 3, 3) |J| P_perp
    divA: jax.Array            # (E, P, 3) d_i A^{ij}
    ginv_u: jax.Array          # (E, P, 3) g^{u j} at the nodes
    wall_A_row: jax.Array      # (E, N, 3) first row A^{u j} at the wall points (conormal flux ``(A grad f)^u``)
    wall_ginv_u: jax.Array     # (E, N, 3) g^{u j} at the wall points (physical normal ``g^{u j} d_j f / sqrt(g^{uu})``)
    wall_J: jax.Array          # (E, N) |J| at the wall points


class NodalBundleStatic:
    """Identity-compared holder of the static (non-traced) part of a bundle: identity dict, layout, arm label, ``n``."""

    def __init__(self, identity: dict, layout, arm: str, n: int):
        self.identity, self.layout, self.arm, self.n = identity, layout, arm, n

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass(frozen=True)
class NodalGeometryBundle:
    ctx: object                # NodalPerpendicularContext (plan, lplan, prec, curvature_flux)
    ref: NodalReferenceArrays
    static: NodalBundleStatic

    @property
    def identity(self) -> dict:
        return self.static.identity

    @property
    def layout(self):
        return self.static.layout

    @property
    def arm(self) -> str:
        return self.static.arm

    @property
    def n(self) -> int:
        return self.static.n

    @property
    def E(self) -> int:
        return self.ctx.plan.structure.n_eta

    @property
    def P(self) -> int:
        return self.ctx.plan.structure.P

    @property
    def N(self) -> int:
        return self.ctx.lplan.structure.N

    @property
    def H(self):
        """The SBP norm weights ``Hp * deta`` ``(E, P)``."""
        return jnp.asarray(self.ctx.plan.Hp) * self.ctx.plan.structure.deta


jax.tree_util.register_dataclass(NodalGeometryBundle, data_fields=["ctx", "ref"], meta_fields=["static"])


def reference_arrays_numpy(ref: NodalReferenceArrays) -> dict:
    """The reference arrays as float64 numpy arrays by field name."""
    return {k: np.asarray(v, dtype=np.float64) for k, v in ref._asdict().items()}


# ---------------------------------------------------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------------------------------------------------
def assemble_nodal_bundle(arm: str, layout, nodal, nodal_meta: dict, lap: dict, lap_meta: dict, provenance: dict, *,
                          schema: str = NODAL_BUNDLE_SCHEMA, build_preconditioner: bool = True, prec_kw: dict | None = None,
                          expected_plans: dict | None = None, git_commit: str = "unknown") -> NodalGeometryBundle:
    """Assemble a bundle from in-memory metrics (shared by every builder and by :func:`load_nodal_bundle`).

    ``nodal`` is a :class:`~drbx.stencils.nodal_plan.NodalMetric`; ``lap`` holds ``points, A, divA, J, ginv_u, wall_points,
    wall_A, wall_ginv_u, wall_J`` (and ``sign``) as the Laplacian extraction; ``provenance`` entries (file shas, metric
    identities, sidecar, ...) are copied into the identity; ``arm`` is an opaque label and ``schema`` / ``git_commit`` are
    recorded in the identity. ``prec_kw`` overrides the preconditioner options of
    :data:`~drbx.native.fci_nodal_perpendicular_model.NODAL_PERPENDICULAR_DEFAULTS` (method, factor dtype).

    Raises ``ValueError`` on an arm / P / n_eta / node / wall / Hp / |J| mismatch; a differing stored ``layout_sha256`` is only
    recorded (``agreement["layout"]``), since float shas are not portable. ``expected_plans`` (a dict with ``nodal_plan_sha256`` and
    ``laplacian_plan_sha256``) makes a differing rebuilt plan sha an error (same-platform strictness; not used by
    :func:`load_nodal_bundle` unless asked).
    """
    from drbx.geometry.nodal_layout import wall_points as layout_wall_points
    from drbx.geometry.sbp_laplacian import LaplacianMetric, build_laplacian_plan
    from drbx.native.fci_nodal_perpendicular_rhs import build_nodal_perpendicular_context
    from drbx.stencils.nodal_plan import build_nodal_plan, plan_identity as nodal_plan_identity

    prec_opts = {**NODAL_PERPENDICULAR_DEFAULTS.preconditioner, **(prec_kw or {})}
    t0 = time.perf_counter()
    arm_nodal, arm_lap = nodal_meta.get("eta_filter"), lap_meta.get("eta_filter")
    if arm_nodal != arm_lap:
        raise ValueError(f"arm identity mismatch: nodal metric eta_filter {arm_nodal!r} != Laplacian metric {arm_lap!r}")
    if lap_meta.get("arm", arm) != arm:
        raise ValueError(f"the Laplacian metric is of arm {lap_meta.get('arm')!r}, not {arm!r}")
    # the layout sha hashes float bytes (core Gauss weights come from LAPACK), so it is not portable across platforms: a
    # stored sha that differs is provenance only; the layout match is checked below to tolerance (P, n_eta, nodes, wall, |J|)
    layout_sha = layout_identity(layout)
    metric_layout_sha = lap_meta.get("layout_sha256")
    for key, want in (("P", layout.P), ("n_eta", layout.n_eta)):
        for who, meta in (("nodal", nodal_meta), ("Laplacian", lap_meta)):
            if key in meta and int(meta[key]) != int(want):
                raise ValueError(f"the {who} metric has {key} = {meta[key]}, the layout {want}")
    pts = np.asarray(lap["points"])
    want_pts = np.stack([np.broadcast_to(layout.node_u, (layout.n_eta, layout.P)),
                         np.broadcast_to(layout.node_theta, (layout.n_eta, layout.P)),
                         np.broadcast_to(((np.arange(layout.n_eta) + 0.5) * layout.deta)[:, None], (layout.n_eta, layout.P))], -1)
    if pts.shape != want_pts.shape or np.abs(pts - want_pts).max() > 1e-13:
        raise ValueError("the Laplacian metric nodes differ from the layout nodes")
    wpts = layout_wall_points(layout, layout.walls[0])
    lw = np.asarray(lap["wall_points"])
    if lw.shape != wpts.shape or np.abs(lw - wpts).max() > 1e-13:
        raise ValueError("the Laplacian metric wall points differ from wall_points(layout, wall)")

    jac_n, jac_l = np.asarray(nodal.jac, dtype=np.float64), np.asarray(lap["J"], dtype=np.float64)
    if jac_n.shape != jac_l.shape:
        raise ValueError(f"|J| shapes differ: nodal {jac_n.shape}, Laplacian {jac_l.shape}")
    jac_rel = float(np.abs(jac_n - jac_l).max() / np.abs(jac_n).max())
    if jac_rel > 1e-12:
        raise ValueError(f"|J| of the two extractions differs by {jac_rel:.3e} (> 1e-12)")
    agreement = {"jac": "bitwise" if jac_rel == 0.0 else f"max_rel {jac_rel:.3e}"}
    if metric_layout_sha is not None:
        agreement["layout"] = ("sha equal" if metric_layout_sha == layout_sha else
                               f"metric layout sha {metric_layout_sha} differs; nodes and wall points agree to 1e-13")
    lmetric = LaplacianMetric(np.asarray(lap["A"]), jac_l, np.asarray(lap["ginv_u"]))
    nplan = build_nodal_plan(layout, nodal)
    lplan = build_laplacian_plan(layout, lmetric)
    hp_n, hp_l = np.asarray(nplan.Hp), np.asarray(lplan.Hp)
    if np.array_equal(hp_n, hp_l):
        agreement["Hp"] = "bitwise"
    else:
        hp_rel = float(np.abs(hp_n - hp_l).max() / np.abs(hp_n).max())
        if hp_rel > 1e-12:
            raise ValueError(f"Hp of the nodal and the Laplacian plan differ by {hp_rel:.3e} (> 1e-12)")
        # the Laplacian plan takes the nodal |J| (so that the context's bitwise Hp check holds); the change is <= 1e-12
        lmetric = LaplacianMetric(np.asarray(lap["A"]), jac_n, np.asarray(lap["ginv_u"]))
        lplan = build_laplacian_plan(layout, lmetric)
        agreement["Hp"] = f"max_rel {hp_rel:.3e}; Laplacian plan rebuilt with the nodal |J|"
        agreement["jac"] += " (replaced by the nodal |J| in the Laplacian plan)"
    t_plans = time.perf_counter() - t0
    ctx = build_nodal_perpendicular_context(nplan, lplan, build_preconditioner=build_preconditioner,
                                            **(prec_opts if build_preconditioner else {}))
    t_prec = time.perf_counter() - t0 - t_plans
    ref = NodalReferenceArrays(points=lap["points"], wall_points=wpts, h=nodal.h, jac=jac_n, B=nodal.B, K=nodal.K, A=lap["A"],
                               divA=lap["divA"], ginv_u=lap["ginv_u"], wall_A_row=np.asarray(lap["wall_A"])[..., 0, :],
                               wall_ginv_u=lap["wall_ginv_u"], wall_J=lap["wall_J"])
    ref = NodalReferenceArrays(*(jnp.asarray(np.asarray(a, dtype=np.float64)) for a in ref))
    n_plan_sha, l_plan_sha = nodal_plan_identity(nplan), laplacian_plan_identity(lplan)
    if expected_plans is not None:
        for key, got in (("nodal_plan_sha256", n_plan_sha), ("laplacian_plan_sha256", l_plan_sha)):
            if expected_plans.get(key) != got:
                raise ValueError(f"rebuilt {key} {got} != stored {expected_plans.get(key)}")
    identity = dict(schema=schema, arm=arm, n_eta=int(layout.n_eta), P=int(layout.P), N_wall=int(lplan.structure.N),
                    family=NODAL_PERPENDICULAR_DEFAULTS.layout_family, arm_identity=arm_nodal, layout_sha256=layout_sha,
                    nodal_plan_sha256=n_plan_sha, laplacian_plan_sha256=l_plan_sha, agreement=agreement,
                    sign=float(lap.get("sign", 1.0)), arrays_sha256=sha256_arrays(reference_arrays_numpy(ref)))
    identity.update(provenance)
    identity["n"] = int(provenance.get("n", layout.n))
    identity["bundle_sha256"] = bundle_sha(identity)
    identity["git_commit"] = git_commit
    identity["preconditioner"] = dict(**(prec_opts if build_preconditioner else {"method": None}),
                                      info=jsonable(getattr(ctx.prec, "info", None)), plans_seconds=t_plans,
                                      build_seconds=t_prec)
    return NodalGeometryBundle(ctx, ref, NodalBundleStatic(identity, layout, arm, int(identity["n"])))


# ---------------------------------------------------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------------------------------------------------
def save_nodal_bundle(bundle: NodalGeometryBundle, out) -> Path:
    """Write ``<out>/bundle.npz`` (the reference arrays) and ``<out>/identity.json``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = reference_arrays_numpy(bundle.ref)
    arrays["nodal_meta_eta_filter"] = np.array(json.dumps(bundle.identity["arm_identity"], sort_keys=True))
    tmp = out / "bundle.npz.tmp"
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(out / "bundle.npz")
    (out / "identity.json").write_text(json.dumps(jsonable(bundle.identity), indent=1, sort_keys=True))
    return out


def load_nodal_bundle(path, expected_identity=None, *, schema: str = NODAL_BUNDLE_SCHEMA, build_preconditioner: bool = True,
                      prec_kw: dict | None = None, strict_plan_shas: bool = False, git_commit: str = "unknown"
                      ) -> NodalGeometryBundle:
    """Load a bundle directory written by :func:`save_nodal_bundle`; rebuilds the plans and the preconditioner.

    ``expected_identity``: ``None``; a ``str`` (the expected ``bundle_sha256``); or a ``dict`` of identity entries that must equal
    the stored ones (for example ``{"arm": "filtered", "n": 32, "arm_identity": {...}, "nodal_metric_identity": "..."}``).
    Raises ``ValueError`` on a schema or identity mismatch, if ``identity.json`` is inconsistent with its ``bundle_sha256`` or if
    the stored arrays do not match their recorded ``arrays_sha256``.

    The rebuilt layout / plan shas are compared with the stored ones. Equal: the rebuilt identity must reproduce the stored
    ``bundle_sha256``. Different (the bundle was built on another platform, whose LAPACK differs by ulps; see the module
    docstring): the stored identity is kept, so the bundle keeps the identity of its build, and the rebuilt shas are recorded in
    ``identity["rebuilt"]``; the layout, norm and ``|J|`` agreement was checked to tolerance by the assembly.
    ``strict_plan_shas=True`` makes a differing plan sha an error (same-platform strictness).
    """
    from drbx.geometry.nodal_families import build_family_a_layout
    from drbx.stencils.nodal_plan import NodalMetric

    path = Path(path)
    identity = json.loads((path / "identity.json").read_text())
    if identity.get("schema") != schema:
        raise ValueError(f"unsupported bundle schema {identity.get('schema')!r}")
    if expected_identity is not None:
        if isinstance(expected_identity, str):
            if identity["bundle_sha256"] != expected_identity:
                raise ValueError(f"bundle identity {identity['bundle_sha256']} != expected {expected_identity}")
        else:
            bad = {k: (identity.get(k), v) for k, v in expected_identity.items() if identity.get(k) != v}
            if bad:
                raise ValueError(f"bundle identity mismatch (stored, expected): {bad}")
    if bundle_sha(identity) != identity["bundle_sha256"]:
        raise ValueError("bundle identity.json is inconsistent with its bundle_sha256")
    with np.load(path / "bundle.npz", allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in NodalReferenceArrays._fields}
    if sha256_arrays(arrays) != identity["arrays_sha256"]:
        raise ValueError("bundle.npz does not match its recorded arrays_sha256")

    layout = build_family_a_layout(identity["n"], n_eta=identity["n_eta"])
    nodal = NodalMetric(arrays["h"], arrays["jac"], arrays["B"], arrays["K"])
    lap = dict(points=arrays["points"], A=arrays["A"], divA=arrays["divA"], J=arrays["jac"], ginv_u=arrays["ginv_u"],
               wall_points=arrays["wall_points"], wall_A=np.stack([arrays["wall_A_row"]] * 3, axis=-2),
               wall_ginv_u=arrays["wall_ginv_u"], wall_J=arrays["wall_J"], sign=identity.get("sign", 1.0))
    lap_meta = dict(arm=identity["arm"], eta_filter=identity["arm_identity"], layout_sha256=identity["layout_sha256"])
    prov = {k: v for k, v in identity.items() if k in ("n", "nodal_metric_path", "nodal_metric_identity", "nodal_metric_file_sha256",
                                                      "laplacian_metric_path", "laplacian_metric_identity",
                                                      "laplacian_metric_file_sha256", "sidecar_sha256", "synthetic")}
    plan_shas = {k: identity[k] for k in ("nodal_plan_sha256", "laplacian_plan_sha256")}
    b = assemble_nodal_bundle(identity["arm"], layout, nodal, dict(eta_filter=identity["arm_identity"]), lap, lap_meta, prov,
                              schema=schema, build_preconditioner=build_preconditioner, prec_kw=prec_kw,
                              expected_plans=plan_shas if strict_plan_shas else None, git_commit=git_commit)
    b.identity["agreement"] = identity["agreement"]                  # what the build measured (the rebuild sees equal arrays)
    differs = sorted(k for k in _PLAN_SHA_KEYS if b.identity[k] != identity[k])
    if differs:
        rebuilt = {k: b.identity[k] for k in _PLAN_SHA_KEYS}
        for key in _PLAN_SHA_KEYS:
            b.identity[key] = identity[key]
        b.identity["rebuilt"] = dict(rebuilt, differs=differs)
    b.identity["bundle_sha256"] = bundle_sha(b.identity)
    if b.identity["bundle_sha256"] != identity["bundle_sha256"]:
        raise ValueError("the rebuilt bundle's identity differs from the stored one")
    return b
