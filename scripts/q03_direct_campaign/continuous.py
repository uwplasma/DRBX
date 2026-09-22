"""Frozen analytical HSX face integration; no historical workspace imports.

Quadrature copied unchanged from run_global_shared_flux.py (20260919).
"""
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from .frozen_mms import ManufacturedField


def context(seeds, meta, metric_cache, makegrid):
    from drbx.geometry.MetricEvaluator import MetricEvaluator
    from drbx.geometry.Bfield_evaluator import bfield_evaluator_from_makegrid
    with np.load(metric_cache, allow_pickle=False) as payload:
        evaluator = MetricEvaluator.from_cache_payload(payload, prefix="metric_evaluator_")
    bfield = bfield_evaluator_from_makegrid(makegrid, currents=np.asarray(meta["currents"], dtype=float), method="cubic")
    grid = SimpleNamespace(**{axis: SimpleNamespace(faces=seeds["grid."+axis]) for axis in "xyz"})
    fields = {}
    for name, manifest in meta["field_manifests"].items():
        m = manifest["field"]
        fields[name] = {"field": ManufacturedField(m=int(m["m"]), amplitude=float(m["amplitude"]), lam=float(m["lambda"]), chi=float(m["chi"]), eta_period=float(m["eta_period"]), radial_profile=m.get("radial_profile", "smooth_axis"), field_kind=m["field_kind"])}
    return {"artifact": SimpleNamespace(geometry=SimpleNamespace(grid=grid)), "evaluator": evaluator, "bfield": bfield, "fields": fields}


def face_quadrature_batch(context, faces, indices, order=6):
    grid = context["artifact"].geometry.grid
    nodes, weights = np.polynomial.legendre.leggauss(order)
    all_points = []
    all_weights = []
    for face_index in indices:
        axis = int(faces["axis"][face_index])
        i, j, k = map(int, faces["storage"][face_index])
        intervals = [
            (float(grid.x.faces[i]), float(grid.x.faces[i + 1])) if axis != 0 else None,
            (float(grid.y.faces[j]), float(grid.y.faces[j + 1])) if axis != 1 else None,
            (float(grid.z.faces[k]), float(grid.z.faces[k + 1])) if axis != 2 else None,
        ]
        coordinate = (
            float(grid.x.faces[i]) if axis == 0 else float(grid.y.faces[j]) if axis == 1 else float(grid.z.faces[k])
        )
        other = [q for q in range(3) if q != axis]
        values = []
        quadrature_weights = []
        for q in other:
            lo, hi = intervals[q]
            values.append(0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes)
            quadrature_weights.append(0.5 * (hi - lo) * weights)
        mesh = np.stack(np.meshgrid(values[0], values[1], indexing="ij"), axis=-1).reshape(-1, 2)
        point = np.empty((order * order, 3))
        point[:, axis] = coordinate
        point[:, other[0]] = mesh[:, 0]
        point[:, other[1]] = mesh[:, 1]
        weight2 = np.einsum("i,j->ij", quadrature_weights[0], quadrature_weights[1]).reshape(-1)
        all_points.append(point)
        all_weights.append(weight2)
    points = np.concatenate(all_points)
    quadrature_weight = np.concatenate(all_weights)
    metric = context["evaluator"].evaluate(points, reject_nonpositive_J=False)
    magnetic = context["evaluator"].evaluate_magnetic_field(points, context["bfield"], reject_nonpositive_J=False)
    J = np.abs(np.asarray(getattr(metric, "signed_J", metric.J)))
    B = np.asarray(magnetic.B_contravariant)
    Bmag = np.maximum(np.asarray(magnetic.magnitude), 1e-30)
    return points.reshape(len(indices), order * order, 3), quadrature_weight.reshape(len(indices), -1), J.reshape(len(indices), -1), B.reshape(len(indices), order * order, 3), Bmag.reshape(len(indices), -1)
