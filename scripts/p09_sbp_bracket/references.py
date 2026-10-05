"""Nodal exact fields and brackets for the gates: ``<out>/N{n}/references/<set>.npz``.

For each catalogue: values of every field at all nodes ``(F, E, P)``, at the outer-wall face points ``(F, E, N_w)``, and the
exact nodal bracket ``R = point_bracket(h, |J|, grad a, grad b) / rho*`` ``(n_pairs, E, P)`` with the logical metric and the
logical exact gradients (frame invariant; no owner projection). For ``p05`` a reference-budget check re-evaluates the
brackets with the finite-difference step of the actual vorticity halved on a bounded node subset (``budget_*`` arrays).
Needs the provider only for ``p05`` (the frozen reference ``ref``).

    python references.py N --out ROOT [--sets p05n,switch,wave,p05]
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("NPROC", "2")

import argparse
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np                                                                      # noqa: E402

import fields as F                                                                      # noqa: E402

CONFIG = F.CONFIG
RHO = float(CONFIG["rho_star"])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def layout_points(layout) -> np.ndarray:
    E, P = layout.n_eta, layout.P
    eta = (np.arange(E) + 0.5) * layout.deta
    return np.stack([np.broadcast_to(layout.node_u, (E, P)), np.broadcast_to(layout.node_theta, (E, P)),
                     np.broadcast_to(eta[:, None], (E, P))], axis=-1)


def point_bracket(h, jac, ga, gb, rho=RHO):
    """``-(h x grad a) . grad b / (|J| rho*)`` (orientation of ``fci_perpendicular_midpoint_bracket``)."""
    return -np.sum(np.cross(h, ga) * gb, axis=-1) / np.abs(jac) / rho


def load_layout_metric(n: int, out_root: Path):
    from drbx.geometry.nodal_families import build_family_a_layout
    from drbx.stencils.nodal_plan import load_nodal_metric

    layout = build_family_a_layout(n, n_eta=n)
    metric, meta = load_nodal_metric(out_root / f"N{n}" / "nodal_metric.npz", layout)
    return layout, metric, meta


def build_reference():
    """The frozen reference ``ref`` of the P05 field adapter (``compact_c3`` options, as the extraction)."""
    from p08_step5_compact_c3 import campaign as cc
    from p_shared import provider as pshared_provider
    from p_shared import replay_support as rs

    options = cc.operator_options(cc.config())
    prov = pshared_provider.ScriptsGeometryProvider.from_sidecar(
        str(rs.DEFAULT_SIDECAR), verify_hashes=False, curvature=options.get("curvature", "autodiff"),
        bfield_toroidal=options.get("bfield_toroidal", "spline"))
    return prov.reference


def brackets(fs: F.FieldSet, vals_grads, h, jac):
    """``(n_pairs, ...)`` nodal brackets from ``(values, gradients)`` of the field set."""
    _vals, grads = vals_grads
    idx = {name: i for i, name in enumerate(fs.names)}
    return np.stack([point_bracket(h, jac, grads[idx[p.a]], grads[idx[p.b]]) for p in fs.pairs])


def subset_planes(E: int, count: int) -> np.ndarray:
    """Planes ``k = j * E / count`` (same eta fractions at every resolution)."""
    return (np.arange(count) * (E // count)).astype(np.int64)


def compute_set(n: int, out_root: Path, name: str, layout, metric, ref=None) -> dict:
    from drbx.geometry.nodal_layout import wall_points

    t0 = time.perf_counter()
    fs = F.make_set(name, ref)
    E, P = layout.n_eta, layout.P
    pts = layout_points(layout)
    path = out_root / f"N{n}" / "references"
    path.mkdir(parents=True, exist_ok=True)
    planes = subset_planes(E, fs.subset_planes) if fs.subset_planes else None
    nodes_file = path / f"{name}_nodes.npz"
    if nodes_file.exists():                                    # resume: node values/gradients saved before the wall step
        zn = np.load(nodes_file)
        vals, grads = zn["vals"], zn["grads"]
        log(f"N{n} {name}: resumed node values from {nodes_file.name}")
    else:
        if planes is None:
            vals, grads = fs.evaluate(pts)
            vals, grads = vals.reshape(len(fs.names), E, P), grads.reshape(len(fs.names), E, P, 3)
        else:                                                  # values on all planes (cheap), gradients on the subset
            vals, _g = fs.evaluate(pts, values_only=True)
            vals = vals.reshape(len(fs.names), E, P)
            _v, grads = fs.evaluate(pts[planes])
            grads = grads.reshape(len(fs.names), len(planes), P, 3)
        np.savez_compressed(nodes_file, vals=vals, grads=grads, planes=planes if planes is not None else np.arange(E))
    t_eval = time.perf_counter() - t0
    wv, _wg = fs.evaluate(wall_points(layout, layout.walls[0]), wall=True)
    wv = wv.reshape(len(fs.names), E, -1)
    sel = slice(None) if planes is None else planes
    R = brackets(fs, (None, grads), metric.h[sel], metric.jac[sel])
    payload = dict(vals=vals, wall=wv, R=R, names=np.array(fs.names), pairs=np.array(fs.pairs_json()),
                   planes=planes if planes is not None else np.arange(E))
    receipt = dict(set=name, n=n, seconds_eval=t_eval, fields=len(fs.names), pairs=len(fs.pairs),
                   planes=None if planes is None else planes.tolist())
    if name == "p05w":
        bplanes = planes[::2][: int(CONFIG["reference_budget_planes"])]
        step = ref.finite_difference_step
        fs_half = F.p05_set(ref, step=0.5 * step, which="omega")
        try:
            _v2, g2 = fs_half.evaluate(pts[bplanes])
            g2 = g2.reshape(len(fs.names), len(bplanes), P, 3)
            payload["budget_planes"] = bplanes
            payload["budget_R_half"] = brackets(fs_half, (None, g2), metric.h[bplanes], metric.jac[bplanes])
            payload["budget_step"] = np.array([step, 0.5 * step])
        finally:
            ref.finite_difference_step = step
        receipt["budget_planes"] = bplanes.tolist()
    np.savez_compressed(path / f"{name}.npz", **payload)
    nodes_file.unlink()
    receipt["seconds"] = time.perf_counter() - t0
    log(f"N{n} references {name}: {receipt['seconds']:.1f} s (eval {t_eval:.1f} s), max|R| per pair "
        + " ".join(f"{np.abs(r).max():.2e}" for r in R))
    return receipt


def available_sets(out_root: Path, n: int, static: bool = False) -> tuple:
    """Catalogues with a references (or static) file; ``p05full`` is the N32 all-plane omega-pair check."""
    sub = "static" if static else "references"
    return tuple(name for name in F.SET_NAMES + ("p05full",) if (out_root / f"N{n}" / sub / f"{name}.npz").exists())


def run(n: int, out_root: Path, sets=None) -> dict:
    t0 = time.perf_counter()
    sets = tuple(sets or F.SET_NAMES)
    layout, metric, meta = load_layout_metric(n, out_root)
    ref = build_reference() if any(x in ("p05", "p05w") for x in sets) else None
    if ref is not None:
        log(f"N{n}: frozen reference built ({time.perf_counter() - t0:.1f} s)")
    receipts = [compute_set(n, out_root, name, layout, metric, ref) for name in sets]
    out = dict(n=n, seconds=time.perf_counter() - t0, peak_rss_gib=peak_rss_gib(), sets=receipts)
    (out_root / f"N{n}" / "references_receipt.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", default=",".join(F.SET_NAMES))
    args = ap.parse_args(argv)
    run(args.n, args.out, args.sets.split(","))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
