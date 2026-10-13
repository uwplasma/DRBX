"""Input identities and the metric loader of the P GPU benchmark.

The campaign reads exactly one file per resolution, the filtered-arm Laplacian metric of the P10 campaign:
``<inputs>/laplacian/filtered/N<n>/laplacian_metric.npz`` (the layout of ``p10_evolved_mms.inputs``; ``<inputs>/inputs_manifest.json``
is that packed set's manifest). The loader mirrors ``p09_sbp_laplacian.common.MetricData`` but does not import it: that module sets
``JAX_PLATFORMS=cpu`` and ``CUDA_VISIBLE_DEVICES=""`` at import, which would hide the GPUs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import pgb_cases as K

METRIC_SCHEMA = "drbx.p09-laplacian-metric-v1"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sha256_arrays(arrays: dict, extra: str = "") -> str:
    """sha256 over (name, dtype, shape, bytes) of every array (sorted by name) and an extra string (as p09 ``common``)."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype}|{a.shape}|".encode())
        h.update(a.tobytes())
    h.update(extra.encode())
    return h.hexdigest()


def metric_path(inputs, n: int) -> Path:
    return Path(inputs) / K.INPUT_FILES[n]["path"]


def verify_inputs(inputs, ns, campaign: dict | None = None) -> dict:
    """Hash the files of ``ns`` against the campaign identities (and the manifest, when present).

    Returns ``{"ok", "problems", "files": {n: {...}}, "manifest": {...}}``; no JAX.
    """
    inputs = Path(inputs)
    expected = (campaign["inputs"]["files"] if campaign else {str(n): K.INPUT_FILES[n] for n in ns})
    problems, files = [], {}
    manifest_entries = {}
    mpath = inputs / K.MANIFEST_NAME
    manifest = {"present": mpath.is_file()}
    if mpath.is_file():
        manifest["sha256"] = sha256_file(mpath)
        manifest["matches_campaign"] = manifest["sha256"] == K.MANIFEST_SHA256
        try:
            manifest_entries = {e["path"]: e for e in json.loads(mpath.read_text())["files"]}
        except Exception as exc:                                                  # noqa: BLE001
            problems.append(f"unreadable manifest: {exc}")
    for n in ns:
        exp = expected[str(n)]
        p = inputs / exp["path"]
        rec = {"path": str(p), "expected_sha256": exp["sha256"], "expected_bytes": exp["bytes"]}
        if not p.is_file():
            rec["ok"] = False
            problems.append(f"N{n}: missing {p}")
        else:
            rec["bytes"] = p.stat().st_size
            rec["sha256"] = sha256_file(p)
            rec["ok"] = rec["sha256"] == exp["sha256"] and rec["bytes"] == exp["bytes"]
            if not rec["ok"]:
                problems.append(f"N{n}: sha256/size mismatch for {p}")
            me = manifest_entries.get(exp["path"])
            if me is not None and me.get("sha256") != rec["sha256"]:
                problems.append(f"N{n}: file differs from the manifest entry")
                rec["ok"] = False
            rec["in_manifest"] = me is not None
        files[str(n)] = rec
    return {"ok": not problems, "problems": problems, "files": files, "manifest": manifest}


def layout_points(layout) -> np.ndarray:
    E, P = layout.n_eta, layout.P
    eta = (np.arange(E) + 0.5) * layout.deta
    return np.stack([np.broadcast_to(layout.node_u, (E, P)), np.broadcast_to(layout.node_theta, (E, P)),
                     np.broadcast_to(eta[:, None], (E, P))], axis=-1)


def build_layout(n: int):
    from drbx.geometry.nodal_families import build_family_a_layout

    return build_family_a_layout(n, n_eta=n)


def layout_structure(n: int) -> dict:
    """``{E, P, Nc, m, N}`` of the family-A layout at ``n`` (no metric needed; cheap up to n = 256)."""
    lay = build_layout(n)
    core, ring = lay.blocks
    return dict(n=n, E=int(lay.n_eta), P=int(lay.P), Nc=int(core.n_nodes), m=int(ring.m), N=int(ring.N))


class Metric:
    """The extracted Laplacian metric of one resolution (identity-checked), ``layout`` and ``metric`` (``LaplacianMetric``)."""

    def __init__(self, path, n: int):
        from drbx.geometry.sbp_laplacian import LaplacianMetric

        self.n = n
        self.layout = build_layout(n)
        with np.load(path, allow_pickle=False) as z:
            if str(z["schema"]) != METRIC_SCHEMA:
                raise ValueError(f"unsupported metric schema in {path}")
            self.arrays = {k: np.asarray(z[k]) for k in z.files if k not in ("schema", "meta", "identity")}
            self.meta = json.loads(str(z["meta"]))
            self.identity = str(z["identity"])
        if sha256_arrays(self.arrays, json.dumps(self.meta, sort_keys=True)) != self.identity:
            raise ValueError(f"metric identity mismatch in {path}")
        pts = layout_points(self.layout)
        if self.arrays["points"].shape != pts.shape or np.abs(self.arrays["points"] - pts).max() > 1e-13:
            raise ValueError("stored points differ from the layout nodes")
        self.points = self.arrays["points"]
        a = self.arrays
        self.metric = LaplacianMetric(a["A"], a["J"], a["ginv_u"])
