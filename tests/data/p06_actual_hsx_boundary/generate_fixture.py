#!/usr/bin/env python3
"""Regenerate the bounded N32 real-HSX wall reconstruction fixture."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WORKSPACE = REPO.parent
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]

import audit_p06_curvature_bounded as p  # noqa: E402


def main() -> None:
    resolution = 32
    parent = WORKSPACE / "work/p06_wall_followup_20260922/N32.npz"
    corrected = WORKSPACE / "work/p06_curvature_corrected_20260922"
    with np.load(parent) as source:
        raw_keys = np.asarray(source["raw_keys"], dtype=np.int64)
    raw_key = raw_keys[np.flatnonzero(raw_keys[:, 0] == resolution - 1)[0]]
    context = p.cubic._load_context(p.GEOMETRY, p.BASELINE, resolution)
    reference = p.build_continuum_reference_from_sidecar(
        corrected / "localized_reference_sidecar.json", verify_hashes=True
    )
    point = np.asarray(
        (
            context.x_centers[raw_key[0]],
            context.y_centers[raw_key[1]],
            context.z_centers[raw_key[2]],
        ),
        dtype=np.float64,
    )
    fit = p.numerics._fit_entity(
        context,
        point,
        axis=0,
        eta_index=int(raw_key[2]),
        owner_values=np.asarray(context.arrays["owner_values"]),
    )
    policy = p.cubic.POLICY["deficient_row_expansion_schedule"][fit.diagnostics["fallback_level"]]
    donors, _weights, _diagnostics, _ties = p.cubic._row_batch(
        context,
        0,
        int(raw_key[2]),
        point[None, :],
        exact_query=False,
        count=int(policy["donors_per_plane"]),
        pool_count=int(policy["candidate_pool_per_plane"]),
    )
    donors = donors[0]
    observation = p.cubic._centered_observations(
        context, donors[None, :], fit.center_regular[None, :], fit.scale[None, :]
    )[0]
    eta = p.cubic.base._unwrap_periodic(
        context.owner_eta[donors], fit.center_regular[2], context.eta_period
    )
    distance2 = np.sum(
        (
            (context.arrays["owner_centroid_xy"][donors] - fit.center_regular[:2])
            / fit.scale[:2]
        )
        ** 2,
        axis=1,
    ) + ((eta - fit.center_regular[2]) / fit.scale[2]) ** 2
    wall_key = np.asarray([[0, resolution, raw_key[1], raw_key[2]]], dtype=np.int64)
    wall_points, wall_weights = p._face_quadrature(context, wall_key, order=3)
    wall_points = wall_points[0]
    wall_weights = wall_weights[0]
    basis_fit = replace(fit, coefficients=np.eye(fit.coefficients.shape[1]))
    value, gradient = p.numerics._evaluate_fit(
        basis_fit, wall_points, context.eta_period
    )
    metric = reference._metric(wall_points)
    owner_values = np.asarray(context.arrays["owner_values"][:, donors], dtype=np.float64)
    payload = {
        "observation": observation,
        "observation_weight": (1.0 + distance2) ** -2,
        "value_rows": value.T,
        "gradient_rows": gradient.transpose(1, 2, 0),
        "g_contravariant": np.asarray(metric["gcontra"]),
        "jacobian": np.asarray(metric["J"]),
        "logical_quadrature_weight": wall_weights,
        "tangential_coordinates": wall_points[:, 1:],
        "owner_values": owner_values,
        "unconstrained_coefficients": fit.coefficients,
        "raw_key": raw_key,
        "wall_points": wall_points,
        "donors": donors,
    }
    path = HERE / "p06_actual_hsx_boundary.npz"
    np.savez_compressed(path, **payload)
    metadata = {
        "schema": "drbx.p06-actual-hsx-boundary-fixture-v1",
        "resolution": resolution,
        "parent": "work/p06_wall_followup_20260922/N32.npz",
        "parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "corrected_reference": "work/p06_curvature_corrected_20260922/localized_reference_sidecar.json",
        "raw_key": raw_key.tolist(),
        "donor_sha256": hashlib.sha256(np.asarray(donors, dtype="<i8").tobytes()).hexdigest(),
        "contents": "real-HSX observation/metric/wall-q3 geometry plus dynamic owner fixture values",
    }
    (HERE / "p06_actual_hsx_boundary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
