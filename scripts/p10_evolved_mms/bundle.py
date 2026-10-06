"""Geometry bundle of the P10 evolved MMS (chunk C1): one per (arm, N).

Joins the two separate metric extractions of the nodal SBP perpendicular stack:

* the bracket / curvature extraction ``nodal_metric.npz`` (``h, |J|, B, K``; raw ``work/p09_m3_campaign_20261004/N<n>``,
  filtered ``work/p09_m3_filtered_20261004/N<n>``), read by ``drbx.stencils.nodal_plan.load_nodal_metric`` (the loader of
  ``scripts/p09_sbp_bracket/references.py``);
* the Laplacian extraction ``laplacian_metric.npz`` (``A = |J| P_perp``, ``d_i A^{ij}``, ``|J|``, ``g^{u j}`` at the nodes and
  at the wall points; ``work/p09_m5_laplacian_20261004/<arm>/N<n>``), read by ``scripts/p09_sbp_laplacian/common.MetricData``;

builds the :class:`~drbx.stencils.nodal_plan.NodalPlan` and the :class:`~drbx.geometry.sbp_laplacian.LaplacianPlan`, asserts
that the layout, the arm identity, the nodes and ``Hp`` / ``|J|`` agree, and builds the
:class:`~drbx.native.fci_nodal_perpendicular_rhs.NodalPerpendicularContext` (core-Schur preconditioner, float32 factors). The
:class:`Bundle` is a pytree (a ``jit`` argument): ``ctx`` and the nodal reference arrays :class:`RefArrays` are the traced
data, the identity and the layout are static.

``Bundle.identity`` (also ``identity.json`` of a saved bundle): arm identity (``eta_filter`` option incl. its arm sha, ``None``
for the raw arm), both metric identities and file shas, layout sha, both plan shas, the ``Hp`` / ``|J|`` agreement report
(``"bitwise"`` or the measured maximum relative difference), the git commit and ``bundle_sha256`` (everything but the commit and
the preconditioner timings). Saving writes ``<out>/bundle.npz`` (the reference arrays) and ``<out>/identity.json``; loading
rebuilds both plans (deterministic: the plan shas must match) and the preconditioner.

    python bundle.py ARM N OUT_DIR          # build_bundle("raw" | "filtered", N, OUT_DIR)
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("NPROC", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parent
WORKSPACE = REPO.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SCHEMA = "drbx.p10-bundle-v1"
CONFIG = json.loads((HERE / "configuration.json").read_text())
ARMS = tuple(CONFIG["arms"])
#: identity entries excluded from ``bundle_sha256`` (they do not change the geometry)
_VOLATILE = ("git_commit", "preconditioner", "bundle_sha256", "created", "nodal_metric_path", "laplacian_metric_path")
PRECONDITIONER = dict(method="core_schur", factor_dtype="float32")


# ---------------------------------------------------------------------------------------------------------------------
# the existing P09 loaders
# ---------------------------------------------------------------------------------------------------------------------
def laplacian_common():
    """``scripts/p09_sbp_laplacian/common.py`` (``MetricData``, ``layout_identity``, ``sha256_file``) under a private module name
    (the file shares its basename with other campaign helpers)."""
    name = "p09_sbp_laplacian_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / "p09_sbp_laplacian" / "common.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def metric_paths(arm: str, n: int, nodal_root=None, laplacian_root=None) -> tuple[Path, Path]:
    """``(nodal_metric.npz, laplacian_metric.npz)`` of ``arm`` at ``N = n``."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    roots = CONFIG["roots"]
    nodal_root = WORKSPACE / roots["nodal_metric"][arm] if nodal_root is None else Path(nodal_root)
    laplacian_root = WORKSPACE / roots["laplacian_metric"] if laplacian_root is None else Path(laplacian_root)
    return nodal_root / f"N{n}" / "nodal_metric.npz", laplacian_root / arm / f"N{n}" / "laplacian_metric.npz"


def git_commit(repo: Path = REPO) -> str:
    """The HEAD commit read from the ``.git`` directory (no git command); ``"unknown"`` if it cannot be found."""
    try:
        git = repo / ".git"
        if git.is_file():                                                           # a worktree: ``gitdir: <path>``
            git = (repo / git.read_text().split(":", 1)[1].strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(":", 1)[1].strip()
        common = git
        if (git / "commondir").exists():
            common = (git / (git / "commondir").read_text().strip()).resolve()
        for base in (git, common):
            if (base / ref).exists():
                return (base / ref).read_text().strip()
        packed = common / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
    except Exception:                                                               # noqa: BLE001
        pass
    return "unknown"


# ---------------------------------------------------------------------------------------------------------------------
# the bundle
# ---------------------------------------------------------------------------------------------------------------------
class RefArrays(NamedTuple):
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


class _Static:
    """Identity-compared holder of the static (non-traced) part of a bundle: identity dict, layout, arm, sizes."""

    def __init__(self, identity: dict, layout, arm: str, n: int):
        self.identity, self.layout, self.arm, self.n = identity, layout, arm, n

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass(frozen=True)
class Bundle:
    ctx: object                # NodalPerpendicularContext (plan, lplan, prec, curvature_flux)
    ref: RefArrays
    static: _Static

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


jax.tree_util.register_dataclass(Bundle, data_fields=["ctx", "ref"], meta_fields=["static"])


def _sha_arrays(arrays: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _bundle_sha(identity: dict) -> str:
    keep = {k: v for k, v in identity.items() if k not in _VOLATILE}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def _np_ref(ref: RefArrays) -> dict:
    return {k: np.asarray(v, dtype=np.float64) for k, v in ref._asdict().items()}


def assemble_bundle(arm: str, layout, nodal, nodal_meta: dict, lap: dict, lap_meta: dict, provenance: dict, *,
                    build_preconditioner: bool = True, prec_kw: dict | None = None, expected_plans: dict | None = None) -> Bundle:
    """Assemble a bundle from in-memory metrics (shared by :func:`build_bundle`, :func:`load_bundle` and the synthetic bundle).

    ``nodal`` is a :class:`~drbx.stencils.nodal_plan.NodalMetric`; ``lap`` holds ``points, A, divA, J, ginv_u, wall_points,
    wall_A, wall_ginv_u, wall_J`` (and ``sign``) as the Laplacian extraction; ``provenance`` entries (file shas, metric
    identities, sidecar, ...) are copied into the identity. Raises ``ValueError`` on a layout / arm / node / Hp / |J| mismatch.
    """
    from drbx.geometry.sbp_laplacian import LaplacianMetric, build_laplacian_plan
    from drbx.native.fci_nodal_perpendicular_rhs import build_nodal_perpendicular_context
    from drbx.stencils.nodal_plan import build_nodal_plan, plan_identity as nodal_plan_identity

    C = laplacian_common()
    t0 = time.perf_counter()
    arm_nodal, arm_lap = nodal_meta.get("eta_filter"), lap_meta.get("eta_filter")
    if arm_nodal != arm_lap:
        raise ValueError(f"arm identity mismatch: nodal metric eta_filter {arm_nodal!r} != Laplacian metric {arm_lap!r}")
    if lap_meta.get("arm", arm) != arm:
        raise ValueError(f"the Laplacian metric is of arm {lap_meta.get('arm')!r}, not {arm!r}")
    layout_sha = C.layout_identity(layout)
    if "layout_sha256" in lap_meta and lap_meta["layout_sha256"] != layout_sha:
        raise ValueError("the Laplacian metric was extracted for a different layout")
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
    from drbx.geometry.nodal_layout import wall_points as layout_wall_points
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
                                            **({**PRECONDITIONER, **(prec_kw or {})} if build_preconditioner else {}))
    t_prec = time.perf_counter() - t0 - t_plans
    ref = RefArrays(points=lap["points"], wall_points=wpts, h=nodal.h, jac=jac_n, B=nodal.B, K=nodal.K, A=lap["A"],
                    divA=lap["divA"], ginv_u=lap["ginv_u"], wall_A_row=np.asarray(lap["wall_A"])[..., 0, :],
                    wall_ginv_u=lap["wall_ginv_u"], wall_J=lap["wall_J"])
    ref = RefArrays(*(jnp.asarray(np.asarray(a, dtype=np.float64)) for a in ref))
    n_plan_sha, l_plan_sha = nodal_plan_identity(nplan), C.plan_identity(lplan)
    if expected_plans is not None:
        for key, got in (("nodal_plan_sha256", n_plan_sha), ("laplacian_plan_sha256", l_plan_sha)):
            if expected_plans.get(key) != got:
                raise ValueError(f"rebuilt {key} {got} != stored {expected_plans.get(key)}")
    identity = dict(schema=SCHEMA, arm=arm, n_eta=int(layout.n_eta), P=int(layout.P), N_wall=int(lplan.structure.N), family="A",
                    arm_identity=arm_nodal, layout_sha256=layout_sha, nodal_plan_sha256=n_plan_sha,
                    laplacian_plan_sha256=l_plan_sha, agreement=agreement, sign=float(lap.get("sign", 1.0)),
                    arrays_sha256=_sha_arrays(_np_ref(ref)))
    identity.update(provenance)
    identity["n"] = int(provenance.get("n", layout.n))
    identity["bundle_sha256"] = _bundle_sha(identity)
    identity["git_commit"] = git_commit()
    identity["preconditioner"] = dict(**({**PRECONDITIONER, **(prec_kw or {})} if build_preconditioner else {"method": None}),
                                      info=_jsonable(getattr(ctx.prec, "info", None)), plans_seconds=t_plans,
                                      build_seconds=t_prec)
    return Bundle(ctx, ref, _Static(identity, layout, arm, int(identity["n"])))


def _jsonable(o):
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _file_identity(path: Path) -> str:
    with np.load(path, allow_pickle=False) as z:
        return str(z["identity"])


def build_bundle(arm: str, n: int, out=None, *, nodal_root=None, laplacian_root=None, build_preconditioner: bool = True,
                 prec_kw: dict | None = None) -> Bundle:
    """Load both extractions of ``(arm, n)``, build the plans and the context and (if ``out`` is given) save the bundle.

    ``out`` is a directory (``bundle.npz`` + ``identity.json``); ``nodal_root`` / ``laplacian_root`` override the campaign
    roots of ``configuration.json``.
    """
    from drbx.stencils.nodal_plan import load_nodal_metric

    C = laplacian_common()
    nodal_path, lap_path = metric_paths(arm, n, nodal_root, laplacian_root)
    layout = C.build_layout(n)
    nodal, nodal_meta = load_nodal_metric(nodal_path, layout)                       # checks schema, identity, nodes
    md = C.MetricData(lap_path, layout)                                             # checks schema, identity, nodes
    lap = dict(md.arrays)
    prov = dict(n=int(n), nodal_metric_path=str(nodal_path), nodal_metric_identity=_file_identity(nodal_path),
                nodal_metric_file_sha256=C.sha256_file(nodal_path), laplacian_metric_path=str(lap_path),
                laplacian_metric_identity=md.identity, laplacian_metric_file_sha256=C.sha256_file(lap_path),
                sidecar_sha256=nodal_meta.get("sidecar_sha256"))
    if md.meta.get("sidecar_sha256") != nodal_meta.get("sidecar_sha256"):
        raise ValueError("the two metric extractions used different sidecars")
    bundle = assemble_bundle(arm, layout, nodal, nodal_meta, lap, md.meta, prov, build_preconditioner=build_preconditioner,
                             prec_kw=prec_kw)
    if out is not None:
        save_bundle(bundle, out)
    return bundle


def save_bundle(bundle: Bundle, out) -> Path:
    """Write ``<out>/bundle.npz`` (the reference arrays) and ``<out>/identity.json``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = _np_ref(bundle.ref)
    arrays["nodal_meta_eta_filter"] = np.array(json.dumps(bundle.identity["arm_identity"], sort_keys=True))
    tmp = out / "bundle.npz.tmp"
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(out / "bundle.npz")
    (out / "identity.json").write_text(json.dumps(_jsonable(bundle.identity), indent=1, sort_keys=True))
    return out


def load_bundle(path, expected_identity=None, *, build_preconditioner: bool = True, prec_kw: dict | None = None) -> Bundle:
    """Load a bundle directory written by :func:`save_bundle`; rebuilds the plans (their shas must equal the stored ones) and the
    preconditioner.

    ``expected_identity``: ``None``; a ``str`` (the expected ``bundle_sha256``); or a ``dict`` of identity entries that must equal
    the stored ones (for example ``{"arm": "filtered", "n": 32, "arm_identity": {...}, "nodal_metric_identity": "..."}``).
    Raises ``ValueError`` on a mismatch or if the stored arrays do not match their recorded sha.
    """
    from drbx.stencils.nodal_plan import NodalMetric

    path = Path(path)
    identity = json.loads((path / "identity.json").read_text())
    if identity.get("schema") != SCHEMA:
        raise ValueError(f"unsupported bundle schema {identity.get('schema')!r}")
    if expected_identity is not None:
        if isinstance(expected_identity, str):
            if identity["bundle_sha256"] != expected_identity:
                raise ValueError(f"bundle identity {identity['bundle_sha256']} != expected {expected_identity}")
        else:
            bad = {k: (identity.get(k), v) for k, v in expected_identity.items() if identity.get(k) != v}
            if bad:
                raise ValueError(f"bundle identity mismatch (stored, expected): {bad}")
    if _bundle_sha(identity) != identity["bundle_sha256"]:
        raise ValueError("bundle identity.json is inconsistent with its bundle_sha256")
    with np.load(path / "bundle.npz", allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in RefArrays._fields}
    if _sha_arrays(arrays) != identity["arrays_sha256"]:
        raise ValueError("bundle.npz does not match its recorded arrays_sha256")
    from drbx.geometry.nodal_families import build_family_a_layout

    layout = build_family_a_layout(identity["n"], n_eta=identity["n_eta"])
    nodal = NodalMetric(arrays["h"], arrays["jac"], arrays["B"], arrays["K"])
    lap = dict(points=arrays["points"], A=arrays["A"], divA=arrays["divA"], J=arrays["jac"], ginv_u=arrays["ginv_u"],
               wall_points=arrays["wall_points"], wall_A=np.stack([arrays["wall_A_row"]] * 3, axis=-2),
               wall_ginv_u=arrays["wall_ginv_u"], wall_J=arrays["wall_J"], sign=identity.get("sign", 1.0))
    lap_meta = dict(arm=identity["arm"], eta_filter=identity["arm_identity"], layout_sha256=identity["layout_sha256"])
    prov = {k: v for k, v in identity.items() if k in ("n", "nodal_metric_path", "nodal_metric_identity", "nodal_metric_file_sha256",
                                                      "laplacian_metric_path", "laplacian_metric_identity",
                                                      "laplacian_metric_file_sha256", "sidecar_sha256", "synthetic")}
    b = assemble_bundle(identity["arm"], layout, nodal, dict(eta_filter=identity["arm_identity"]), lap, lap_meta, prov,
                        build_preconditioner=build_preconditioner, prec_kw=prec_kw,
                        expected_plans={k: identity[k] for k in ("nodal_plan_sha256", "laplacian_plan_sha256")})
    b.identity["agreement"] = identity["agreement"]                  # what the build measured (the rebuild sees equal arrays)
    b.identity["bundle_sha256"] = _bundle_sha(b.identity)
    if b.identity["bundle_sha256"] != identity["bundle_sha256"]:
        raise ValueError("the rebuilt bundle's identity differs from the stored one")
    return b


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=ARMS)
    ap.add_argument("n", type=int)
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)
    b = build_bundle(args.arm, args.n, args.out)
    print(json.dumps(_jsonable(b.identity), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
