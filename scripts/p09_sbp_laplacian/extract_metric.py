#!/usr/bin/env python3
"""Extract the logical-frame metric of the nodal Laplacian (``LaplacianMetric``) at the family-A nodes, one arm, one N.

For every node ``(u, theta, eta_k)`` of ``build_family_a_layout(N)`` (flat index ``k * P + p``) and every wall point
``(1, theta_j, eta_k)``: the perpendicular flux tensor ``A = J (g^ij - b^i b^j)`` and its logical divergence
``d_i A^{ij}`` by autodiff (``curvature_autodiff.autodiff_perpendicular_geometry``; for the filtered arm through the
``logical_field`` table twin of the eta-filtered field), the jacobian and the first row ``g^{u j}`` of the inverse metric
(both independent of the magnetic field). Stored in the positive-jacobian convention (``A = |J| P_perp``).

    python extract_metric.py N --arm raw|filtered [--out ROOT]

Receipt: ``<out>/<arm>/N<N>/extract_receipt.json`` with the identities (layout, arm sha, sidecar sha, metric sha), the
comparison of ``|J|`` with the M3 nodal metric of the same arm, and a bounded cross-check of the autodiff tensor and
divergence against the reference's own ``_perpendicular_flux_tensor`` / finite-difference ``_perpendicular_geometry``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                                      # noqa: E402
from common import CONFIG, log                                                          # noqa: E402

import numpy as np                                                                      # noqa: E402


def extract(n: int, arm: str, out_root: Path) -> dict:
    t0 = time.perf_counter()
    if arm == "filtered":
        os.environ.setdefault("DRBX_ETA_FILTER_TABLE_CACHE", str(C.WORK.parent / CONFIG["eta_filter_table_cache"]))
    from drbx.geometry.nodal_layout import wall_points
    from drbx.geometry.sbp_laplacian import nodal_laplacian_metric_from_callable
    from p08_step5_compact_c3 import campaign as cc
    from p_shared import provider as pshared_provider
    from p_shared import replay_support as rs

    layout = C.build_layout(n)
    E, P = layout.n_eta, layout.P
    options = cc.operator_options(cc.config())
    sidecar = Path(rs.DEFAULT_SIDECAR)
    eta_filter = CONFIG["eta_filter_arm"] if arm == "filtered" else None
    prov = pshared_provider.ScriptsGeometryProvider.from_sidecar(
        str(sidecar), verify_hashes=False, curvature=options.get("curvature", "autodiff"),
        bfield_toroidal=options.get("bfield_toroidal", "spline"), eta_filter=eta_filter)
    ref = prov.reference                                   # AutodiffCurvatureReference (delegating wrapper)
    perp = ref.autodiff_perpendicular()                    # points -> (tensor (Q, 3, 3), divergence (Q, 3))
    mev = ref.metric_evaluator
    log(f"{arm} N{n}: provider {time.perf_counter() - t0:.1f}s; layout P={P}, {E * P} nodes + {E * layout.blocks[1].N} wall points")
    chunk = int(CONFIG["extraction_chunk"])
    divs = []

    def geometry(q):
        q = np.asarray(q, dtype=np.float64)
        A, d, J, gu = [], [], [], []
        for s in range(0, len(q), chunk):
            qq = q[s:s + chunk]
            T, D = perp(qq)
            m = mev.evaluate(qq, reject_nonpositive_J=False)
            Js = np.asarray(m.signed_J, dtype=np.float64)
            sg = np.sign(Js)
            A.append(T * sg[:, None, None]); d.append(D * sg[:, None]); J.append(Js)
            gu.append(np.asarray(m.contravariant_metric, dtype=np.float64)[:, 0, :])
        return np.concatenate(A), np.concatenate(d), np.concatenate(J), np.concatenate(gu)

    stash = {}

    def fn_nodes(q):
        A, d, J, gu = geometry(q)
        stash["divA"], stash["Js"] = d, J
        return A, J, gu

    met = nodal_laplacian_metric_from_callable(layout, fn_nodes)
    t_nodes = time.perf_counter() - t0
    log(f"{arm} N{n}: nodes done {t_nodes:.1f}s")
    Js = stash["Js"].reshape(E, P)
    if not (np.all(Js > 0) or np.all(Js < 0)):
        raise ValueError("the jacobian changes sign over the nodes")
    sign = float(np.sign(Js.flat[0]))
    wp = wall_points(layout, layout.walls[0])
    Aw, dw, Jw, gw = geometry(wp.reshape(-1, 3))
    Nw = wp.shape[1]
    arrays = dict(points=C.layout_points(layout), A=met.A_log, divA=stash["divA"].reshape(E, P, 3), J=np.abs(Js),
                  ginv_u=met.ginv_u, sign=np.array(sign), wall_points=wp, wall_A=Aw.reshape(E, Nw, 3, 3),
                  wall_divA=dw.reshape(E, Nw, 3), wall_J=np.abs(Jw).reshape(E, Nw), wall_ginv_u=gw.reshape(E, Nw, 3))
    log(f"{arm} N{n}: wall done {time.perf_counter() - t0:.1f}s")

    # --- checks: |J| against the M3 nodal metric of the same arm; tensor/divergence against the reference's own routines
    check = {}
    m3 = C.M3_METRIC[arm] / f"N{n}" / "nodal_metric.npz"
    if m3.exists():
        with np.load(m3, allow_pickle=False) as z:
            check["jac_vs_m3_max_rel"] = float(np.abs(z["jac"] - arrays["J"]).max() / np.abs(z["jac"]).max())
            check["points_vs_m3_max_abs"] = float(np.abs(z["points"] - arrays["points"]).max())
            check["m3_metric_identity"] = str(z["identity"])
        check["m3_metric_file"] = str(m3)
    kc = int(CONFIG["extraction_check_points"])
    flat = arrays["points"].reshape(-1, 3)
    sel = np.linspace(0, len(flat) - 1, kc).astype(np.int64)
    tc = time.perf_counter()
    T_ref = np.asarray(ref._perpendicular_flux_tensor(flat[sel])) * np.sign(Js.reshape(-1)[sel])[:, None, None]
    T_ad = arrays["A"].reshape(-1, 3, 3)[sel]
    check["tensor_vs_reference_max_rel"] = float(np.abs(T_ad - T_ref).max() / np.abs(T_ref).max())
    _, d_fd = ref._perpendicular_geometry(flat[sel])
    d_fd = np.asarray(d_fd) * np.sign(Js.reshape(-1)[sel])[:, None]
    d_ad = arrays["divA"].reshape(-1, 3)[sel]
    check["divergence_vs_fd_max_rel"] = float(np.abs(d_ad - d_fd).max() / np.abs(d_fd).max())
    check["check_points"] = kc
    check["check_seconds"] = time.perf_counter() - tc
    log(f"{arm} N{n}: checks {json.dumps({k: v for k, v in check.items() if isinstance(v, float)})}")

    arm_meta = None
    a = getattr(ref, "eta_filter_arm", None)
    if a is not None:
        arm_meta = a.meta()
    meta = dict(n=n, arm=arm, P=P, n_eta=E, family="A", sign=sign, convention="A = |J| P_perp (positive jacobian)",
                sidecar=str(sidecar), sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(), options=options,
                provider=type(prov).__name__, eta_filter=arm_meta, layout_sha256=C.layout_identity(layout),
                perpendicular_geometry="autodiff_perpendicular_geometry" + (" (logical_field = eta-filter table twin)" if arm_meta else ""))
    meta_json = json.dumps(meta, sort_keys=True)
    identity = C.sha256_arrays(arrays, meta_json)
    path = C.arm_dir(out_root, arm, n) / C.METRIC_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, schema=np.array(C.METRIC_SCHEMA), meta=np.array(meta_json), identity=np.array(identity), **arrays)
    tmp.replace(path)
    receipt = dict(n=n, arm=arm, identity=identity, path=str(path), bytes=path.stat().st_size, file_sha256=C.sha256_file(path),
                   seconds=time.perf_counter() - t0, seconds_nodes=t_nodes, peak_rss_gib=C.peak_rss_gib(),
                   layout_sha256=meta["layout_sha256"], sidecar_sha256=meta["sidecar_sha256"], eta_filter=arm_meta,
                   checks=check, sign=sign, J_range=[float(arrays["J"].min()), float(arrays["J"].max())],
                   A_range=dict(Auu=[float(arrays["A"][..., 0, 0].min()), float(arrays["A"][..., 0, 0].max())],
                                Att=[float(arrays["A"][..., 1, 1].min()), float(arrays["A"][..., 1, 1].max())],
                                Aee=[float(arrays["A"][..., 2, 2].min()), float(arrays["A"][..., 2, 2].max())]))
    if a is not None:
        receipt["eta_filter_table"] = dict(table_seconds=a.table_seconds, table_from_cache=a.table_from_cache,
                                           equivalence_vs_columns=a.table_check)
    C.write_json(path.parent / "extract_receipt.json", receipt)
    log(f"{arm} N{n}: saved {path} ({receipt['bytes'] / 1e6:.1f} MB, {receipt['seconds']:.1f}s)")
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--arm", choices=C.ARMS, required=True)
    ap.add_argument("--out", type=Path, default=C.DEFAULT_ROOT)
    args = ap.parse_args(argv)
    extract(args.n, args.arm, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
