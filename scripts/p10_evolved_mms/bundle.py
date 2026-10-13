"""Geometry bundle of the P10 evolved MMS (chunk C1): one per (arm, N). A thin layer over the package module
:mod:`drbx.native.fci_nodal_geometry_bundle` (the arm-agnostic assembly, identity, save and load).

Joins the two separate metric extractions of the nodal SBP perpendicular stack:

* the bracket / curvature extraction ``nodal_metric.npz`` (``h, |J|, B, K``; raw ``work/p09_m3_campaign_20261004/N<n>``,
  filtered ``work/p09_m3_filtered_20261004/N<n>``), read by ``drbx.stencils.nodal_plan.load_nodal_metric`` (the loader of
  ``scripts/p09_sbp_bracket/references.py``);
* the Laplacian extraction ``laplacian_metric.npz`` (``A = |J| P_perp``, ``d_i A^{ij}``, ``|J|``, ``g^{u j}`` at the nodes and
  at the wall points; ``work/p09_m5_laplacian_20261004/<arm>/N<n>``), read by ``scripts/p09_sbp_laplacian/common.MetricData``;

and assembles the bundle with the package (:func:`~drbx.native.fci_nodal_geometry_bundle.assemble_nodal_bundle`). What stays
here is P10-specific: the ``CONFIG`` arms and roots, :func:`metric_paths`, the P10 schema ``drbx.p10-bundle-v1`` (so the identity and
``bundle_sha256`` of every bundle are the ones the campaign recorded), the preconditioner choice and the builders that read the
campaign's metric files. ``Bundle``, ``RefArrays`` and ``_Static`` are the package's :class:`NodalGeometryBundle`,
:class:`NodalReferenceArrays` and :class:`NodalBundleStatic`.

``Bundle.identity`` (also ``identity.json`` of a saved bundle): arm identity (``eta_filter`` option incl. its arm sha, ``None``
for the raw arm), both metric identities and file shas, layout sha, both plan shas, the ``Hp`` / ``|J|`` agreement report
(``"bitwise"`` or the measured maximum relative difference), the git commit and ``bundle_sha256`` (everything but the commit and
the preconditioner timings). Saving writes ``<out>/bundle.npz`` (the reference arrays) and ``<out>/identity.json``; loading
rebuilds both plans and the preconditioner; layout / plan shas of a bundle built on another platform are recorded, not rejected
(see the package module).

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

import importlib.util
import json
import sys
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parent
WORKSPACE = REPO.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native import fci_nodal_geometry_bundle as pkg                                              # noqa: E402

SCHEMA = "drbx.p10-bundle-v1"
CONFIG = json.loads((HERE / "configuration.json").read_text())
ARMS = tuple(CONFIG["arms"])
#: identity entries excluded from ``bundle_sha256`` (they do not change the geometry)
_VOLATILE = pkg.VOLATILE_IDENTITY_KEYS
PRECONDITIONER = dict(method="core_schur", factor_dtype="float32")
RefArrays = pkg.NodalReferenceArrays
_Static = pkg.NodalBundleStatic
Bundle = pkg.NodalGeometryBundle
_sha_arrays = pkg.sha256_arrays
_bundle_sha = pkg.bundle_sha
_jsonable = pkg.jsonable


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
# assembly (the package, with the P10 schema, preconditioner and commit)
# ---------------------------------------------------------------------------------------------------------------------
def _np_ref(ref: RefArrays) -> dict:
    return pkg.reference_arrays_numpy(ref)


def assemble_bundle(arm: str, layout, nodal, nodal_meta: dict, lap: dict, lap_meta: dict, provenance: dict, *,
                    build_preconditioner: bool = True, prec_kw: dict | None = None, expected_plans: dict | None = None) -> Bundle:
    """Assemble a bundle from in-memory metrics (shared by :func:`build_bundle`, :func:`load_bundle` and the synthetic bundle):
    :func:`~drbx.native.fci_nodal_geometry_bundle.assemble_nodal_bundle` with the P10 schema, the core-Schur float32
    preconditioner (:data:`PRECONDITIONER`) and the commit of this repository. Raises ``ValueError`` on an arm / P / n_eta / node /
    wall / Hp / |J| mismatch; a differing stored ``layout_sha256`` is only recorded (``agreement["layout"]``), since float shas
    are not portable."""
    return pkg.assemble_nodal_bundle(arm, layout, nodal, nodal_meta, lap, lap_meta, provenance, schema=SCHEMA,
                                     build_preconditioner=build_preconditioner, prec_kw={**PRECONDITIONER, **(prec_kw or {})},
                                     expected_plans=expected_plans, git_commit=git_commit())


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
    return pkg.save_nodal_bundle(bundle, out)


def load_bundle(path, expected_identity=None, *, build_preconditioner: bool = True, prec_kw: dict | None = None,
                strict_plan_shas: bool = False) -> Bundle:
    """Load a bundle directory written by :func:`save_bundle`; rebuilds the plans and the preconditioner
    (:func:`~drbx.native.fci_nodal_geometry_bundle.load_nodal_bundle` with the P10 schema).

    ``expected_identity``: ``None``; a ``str`` (the expected ``bundle_sha256``); or a ``dict`` of identity entries that must equal
    the stored ones (for example ``{"arm": "filtered", "n": 32, "arm_identity": {...}, "nodal_metric_identity": "..."}``).
    Raises ``ValueError`` on a mismatch or if the stored arrays do not match their recorded sha. Plan / layout shas of a bundle
    built on another platform are recorded in ``identity["rebuilt"]`` instead of rejected; ``strict_plan_shas=True`` rejects them.
    """
    return pkg.load_nodal_bundle(path, expected_identity, schema=SCHEMA, build_preconditioner=build_preconditioner,
                                 prec_kw={**PRECONDITIONER, **(prec_kw or {})}, strict_plan_shas=strict_plan_shas,
                                 git_commit=git_commit())


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
