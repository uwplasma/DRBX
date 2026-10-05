#!/usr/bin/env python3
"""Extract the HSX logical-frame nodal metric of the family-A layout (core + ring level) at resolution N.

One process per N. Evaluates ``h = b_cov / B``, ``|J|``, ``B`` and the autodiff curvature ``K`` at every node
``(u, theta, eta_k)`` of ``build_family_a_layout(N)`` (flat index ``k * P + p``) through the scripts provider
(``compact_c3`` B-field options of ``p08_step5_compact_c3``) and writes ``<out>/N{N}/nodal_metric.npz``
(``drbx.stencils.nodal_plan.save_nodal_metric``) plus a receipt with seconds and peak RSS.

    python extract_metric.py N --out ROOT [--check-m1 hsx_m1_N32.npz]
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("NPROC", "2")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import numpy as np                                                                      # noqa: E402

CONFIG = json.loads((HERE / "configuration.json").read_text())


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30        # bytes on macOS


def layout_points(layout) -> np.ndarray:
    E, P = layout.n_eta, layout.P
    eta = (np.arange(E) + 0.5) * layout.deta
    return np.stack([np.broadcast_to(layout.node_u, (E, P)), np.broadcast_to(layout.node_theta, (E, P)),
                     np.broadcast_to(eta[:, None], (E, P))], axis=-1)


def extract(n: int, out_root: Path, check_m1: Path | None = None) -> dict:
    t0 = time.perf_counter()
    from drbx.geometry.nodal_families import build_family_a_layout, family_a
    from drbx.stencils.nodal_plan import NodalMetric, metric_from_geometry_provider, save_nodal_metric
    from p08_step5_compact_c3 import campaign as cc
    from p_shared import provider as pshared_provider
    from p_shared import replay_support as rs

    K, p, _levels = family_a(n)
    layout = build_family_a_layout(n, n_eta=n)
    options = cc.operator_options(cc.config())
    sidecar = Path(rs.DEFAULT_SIDECAR)
    prov = pshared_provider.ScriptsGeometryProvider.from_sidecar(
        str(sidecar), verify_hashes=False, curvature=options.get("curvature", "autodiff"),
        bfield_toroidal=options.get("bfield_toroidal", "spline"))
    log(f"N{n}: provider built in {time.perf_counter() - t0:.1f} s; layout P={layout.P} (core {layout.blocks[0].n_nodes}, "
        f"K={K}, p={p}), {n * layout.P} points")
    adapter = metric_from_geometry_provider(prov)
    points = layout_points(layout)
    flat = points.reshape(-1, 3)
    chunk = int(CONFIG["extraction_chunk"])
    hs, js, Bs, Ks = [], [], [], []
    for s in range(0, len(flat), chunk):
        h, jac, B, Kc = adapter.nodal_metric(flat[s:s + chunk])
        hs.append(h); js.append(jac); Bs.append(B); Ks.append(Kc)
    E, P = layout.n_eta, layout.P
    metric = NodalMetric(np.concatenate(hs).reshape(E, P, 3), np.concatenate(js).reshape(E, P),
                         np.concatenate(Bs).reshape(E, P), np.concatenate(Ks).reshape(E, P, 3))
    seconds_eval = time.perf_counter() - t0
    meta = dict(N=n, K=K, p=p, P=P, n_eta=E, core_nodes=int(layout.blocks[0].n_nodes), family="A",
                sidecar=str(sidecar), sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                options=options, provider=type(prov).__name__, jacobian="abs(p05_metric)")
    path = out_root / f"N{n}" / "nodal_metric.npz"
    identity = save_nodal_metric(path, metric, points, meta)
    check = {}
    if check_m1 is not None:
        z = np.load(check_m1)
        if int(z["n"]) != n:
            raise ValueError("--check-m1 file is for a different N")
        ring = layout.node_ring >= 0
        i, j = layout.node_ring[ring], layout.node_raw_theta[ring]
        for name, mine in (("h", metric.h), ("jac", metric.jac), ("B", metric.B), ("K", metric.K)):
            ref = np.moveaxis(z[f"raw_{name}"][i, j], 1, 0)                      # (E, P_ring[, 3])
            got = mine[:, ring]
            check[name] = float(np.abs(got - ref).max() / np.abs(ref).max())
        pts_ref = np.moveaxis(z["raw_pts"][i, j], 1, 0)
        check["points"] = float(np.abs(points[:, ring] - pts_ref).max())
        log("ring nodes vs M1 raw extraction: " + json.dumps(check))
    receipt = dict(n=n, identity=identity, path=str(path), seconds=time.perf_counter() - t0,
                   seconds_evaluation=seconds_eval, peak_rss_gib=peak_rss_gib(), bytes=path.stat().st_size,
                   check_vs_m1=check, meta=meta)
    (out_root / f"N{n}" / "extract_receipt.json").write_text(json.dumps(receipt, indent=1))
    log(f"N{n}: saved {path} ({receipt['bytes'] / 1e6:.1f} MB, {receipt['seconds']:.1f} s, peak {receipt['peak_rss_gib']:.2f} GiB)")
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check-m1", type=Path, default=None)
    args = ap.parse_args(argv)
    extract(args.n, args.out, args.check_m1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
