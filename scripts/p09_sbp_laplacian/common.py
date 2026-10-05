"""Shared helpers of the P09 nodal SBP Laplacian campaign (M5b): environment, paths, identities, metric loading.

Layout of the output root ``ROOT`` (``work/p09_m5_laplacian_20261004``)::

    ROOT/<arm>/N<n>/laplacian_metric.npz      extracted logical-frame metric (extract_metric.py)
    ROOT/<arm>/N<n>/extract_receipt.json
    ROOT/<arm>/N<n>/audit.json                coefficients + definiteness (audit.py)
    ROOT/<arm>/N<n>/static.json               N - R tables (static.py)
    ROOT/<arm>/N<n>/solve.json                phi controls + CG benchmark (solve_bench.py)
    ROOT/<arm>/N<n>/nodal_laplacian_reference.npz + manifest   (refreeze.py)

Evaluated face tensors (``extract_metric.py --faces``; the face-coefficient variants of :data:`VARIANTS`)::

    FACES_ROOT/<arm>/N<n>/laplacian_faces.npz            A^{ee} at the eta half planes, A^{tt} at the theta half nodes, A^{uu} at the radial faces
    FACES_ROOT/<arm>/N<n>/faces_receipt.json
    FACES_ROOT/<arm>/N<n>/<variant>/{audit,static,solve}.json   the scripts' ``--variant`` runs (the metric still comes from ROOT)

``arm`` is ``raw`` (the production field) or ``filtered`` (the eta-filtered verification arm of ``p_shared.eta_filter``).
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
import json
import resource
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
for _p in (str(SCRIPTS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                                      # noqa: E402

CONFIG = json.loads((HERE / "configuration.json").read_text())
PERIOD = float(CONFIG["eta_period"])
ARMS = tuple(CONFIG["arms"])
WORK = Path("/Users/yxie/Desktop/HSX drbx/work")
DEFAULT_ROOT = WORK / "p09_m5_laplacian_20261004"
METRIC_FILE = "laplacian_metric.npz"
METRIC_SCHEMA = "drbx.p09-laplacian-metric-v1"
FACES_ROOT = DEFAULT_ROOT / "faces"
FACES_FILE = "laplacian_faces.npz"
FACES_SCHEMA = "drbx.p09-laplacian-faces-v1"
#: face-coefficient variants: the face families taking the tensor evaluated at the faces (the rest interpolated from the nodes)
VARIANTS = {"interp": (), "ee": ("ee",), "ee_tt": ("ee", "tt"), "all": ("ee", "tt", "uu")}
M3_METRIC = {"raw": WORK / "p09_m3_campaign_20261004", "filtered": WORK / "p09_m3_filtered_20261004"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] rss={peak_rss_gib():.2f}GiB {msg}", flush=True)


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30        # bytes on macOS


def arm_dir(root: Path, arm: str, n: int) -> Path:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    return Path(root) / arm / f"N{n}"


def faces_dir(root: Path, arm: str, n: int) -> Path:
    return arm_dir(root, arm, n)


def variant_dir(root: Path, arm: str, n: int, variant: str) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {tuple(VARIANTS)}")
    return faces_dir(root, arm, n) / variant


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_arrays(arrays: dict, extra: str = "") -> str:
    """sha256 over (name, dtype, shape, bytes) of every array (sorted by name) and an extra string."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype}|{a.shape}|".encode())
        h.update(a.tobytes())
    h.update(extra.encode())
    return h.hexdigest()


def layout_identity(layout) -> str:
    """sha256 over the family-A layout (n, planes, node coordinates, H weights)."""
    return sha256_arrays({"node_u": layout.node_u, "node_theta": layout.node_theta, "wxy": layout.wxy,
                          "node_ring": np.asarray(layout.node_ring)},
                         extra=json.dumps(dict(n=layout.n, E=layout.n_eta, P=layout.P, deta=float(layout.deta),
                                               du=float(layout.du), family="A"), sort_keys=True))


def layout_points(layout) -> np.ndarray:
    E, P = layout.n_eta, layout.P
    eta = (np.arange(E) + 0.5) * layout.deta
    return np.stack([np.broadcast_to(layout.node_u, (E, P)), np.broadcast_to(layout.node_theta, (E, P)),
                     np.broadcast_to(eta[:, None], (E, P))], axis=-1)


def build_layout(n: int):
    from drbx.geometry.nodal_families import build_family_a_layout

    return build_family_a_layout(n, n_eta=n)


class MetricData:
    """The extracted metric of one arm and resolution: ``metric`` (the ``LaplacianMetric``), exact quantities and wall data.

    All tensors are in the positive-jacobian convention (``A = |J| P_perp``, ``divA = d_i A^{ij}``); ``wall_*`` are at the
    wall grid points ``(u = 1, theta_j, eta_k)`` of ``wall_points(layout, layout.walls[0])``.
    """

    def __init__(self, path: Path, layout):
        from drbx.geometry.sbp_laplacian import LaplacianMetric

        with np.load(path, allow_pickle=False) as z:
            if str(z["schema"]) != METRIC_SCHEMA:
                raise ValueError(f"unsupported metric schema in {path}")
            self.arrays = {k: np.asarray(z[k]) for k in z.files if k not in ("schema", "meta", "identity")}
            self.meta = json.loads(str(z["meta"]))
            self.identity = str(z["identity"])
        a = self.arrays
        if sha256_arrays(a, json.dumps(self.meta, sort_keys=True)) != self.identity:
            raise ValueError(f"metric identity mismatch in {path}")
        pts = layout_points(layout)
        if a["points"].shape != pts.shape or np.abs(a["points"] - pts).max() > 1e-13:
            raise ValueError("stored points differ from the layout nodes")
        self.points = a["points"]
        self.A, self.divA, self.J, self.ginv_u = a["A"], a["divA"], a["J"], a["ginv_u"]
        self.metric = LaplacianMetric(self.A, self.J, self.ginv_u)
        self.wall_points = a["wall_points"]
        self.wall_A, self.wall_ginv_u, self.wall_J = a["wall_A"], a["wall_ginv_u"], a["wall_J"]
        self.sign = float(a["sign"])
        self.layout = layout


def load_metric(root: Path, arm: str, n: int, layout=None) -> MetricData:
    layout = build_layout(n) if layout is None else layout
    return MetricData(arm_dir(root, arm, n) / METRIC_FILE, layout)


def build_plan(md: MetricData, **kw):
    from drbx.geometry.sbp_laplacian import build_laplacian_plan

    return build_laplacian_plan(md.layout, md.metric, **kw)


class FaceData:
    """The tensors evaluated at the faces (``laplacian_faces.npz`` of ``extract_metric.py --faces``) of one arm and resolution."""

    def __init__(self, path: Path, md: MetricData):
        with np.load(path, allow_pickle=False) as z:
            if str(z["schema"]) != FACES_SCHEMA:
                raise ValueError(f"unsupported faces schema in {path}")
            self.arrays = {k: np.asarray(z[k]) for k in z.files if k not in ("schema", "meta", "identity")}
            self.meta = json.loads(str(z["meta"]))
            self.identity = str(z["identity"])
        if sha256_arrays(self.arrays, json.dumps(self.meta, sort_keys=True)) != self.identity:
            raise ValueError(f"faces identity mismatch in {path}")
        if self.meta["metric_identity"] != md.identity:
            raise ValueError("the faces were extracted for a different nodal metric")
        self.A_ee_h, self.A_tt_h, self.A_uu_f = (self.arrays[k] for k in ("A_ee_h", "A_tt_h", "A_uu_f"))

    def face_metric(self, variant: str):
        """``LaplacianFaceMetric`` of the variant (``None`` for ``"interp"``)."""
        from drbx.geometry.sbp_laplacian import LaplacianFaceMetric

        ev = VARIANTS[variant]
        if not ev:
            return None
        return LaplacianFaceMetric(self.A_ee_h, self.A_tt_h if "tt" in ev else None, self.A_uu_f if "uu" in ev else None)


def load_faces(root: Path, arm: str, n: int, md: MetricData) -> FaceData:
    return FaceData(faces_dir(root, arm, n) / FACES_FILE, md)


def variant_plan(md: MetricData, variant: str | None, faces_root: Path = FACES_ROOT, **kw):
    """Plan of a face-coefficient variant (``None``: the plain interpolated plan, the M5b default)."""
    if variant is None or variant == "interp":
        return build_plan(md, **kw)
    return build_plan(md, faces=load_faces(faces_root, md.meta["arm"], md.meta["n"], md).face_metric(variant), **kw)


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=conv))
    tmp.replace(path)


def swap_free_mib() -> float:
    import subprocess

    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    for tok in out.split("  "):
        if tok.strip().startswith("free"):
            return float(tok.split("=")[1].strip().rstrip("M"))
    return float("nan")


def plan_identity(plan) -> str:
    """sha256 over every array of a ``LaplacianPlan`` (name, dtype, shape, bytes) and its structure."""
    import dataclasses

    arrays = {f.name: np.asarray(getattr(plan, f.name)) for f in dataclasses.fields(plan)
              if f.name != "structure" and getattr(plan, f.name) is not None}
    st = dataclasses.asdict(plan.structure)
    if not st.get("evaluated"):
        st.pop("evaluated", None)                       # an interpolated plan keeps its pre-faces identity
    return sha256_arrays(arrays, json.dumps(st, sort_keys=True))
