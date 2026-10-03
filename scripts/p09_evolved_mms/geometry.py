"""One-time host precompute of the time-independent geometry of the P09 evolved MMS (per grid, all owners).

Everything the reference functionals of ``p_shared.perpendicular_reference_rhs`` read is computed ONCE with the existing
host code and saved as ``N{n}.geometry.npz`` (:func:`precompute`; :func:`load` returns the :class:`Geometry` pytree used by
``source.py`` as device inputs):

* raw midpoints of every raw cell: ``points``, ``raw_owner``, ``raw_volume``, ``h = bcov / B``, ``jac = |J|``, ``B``,
  ``K`` and the evolution weight ``q1_weight J / max(B, 1e-30)`` (``OwnerSupport.raw_geometry``, chunked);
* the P07 census faces that are not collapsed (family 0): the 9-node q3 face points, the integrand
  ``contract_face_tensor(weight, tensor, axis)`` (``diffusion_reference``), the lower / upper owner;
* ``owner_volume``;
* the plan's ``dirichlet_points`` / ``neumann_points`` (from the lowered step-4 plan; the plan is dropped afterwards) and
  the physical-normal coefficients ``a = env.normal_coefficients(neumann_points)``.

Operator options of the step-5 compact_c3 campaign (autodiff K, compact_c3 B). Run as ``python -m p09_evolved_mms.geometry
--n 32 --out DIR`` from ``DRBX/scripts``.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.dont_write_bytecode = True

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

WORKSPACE = Path("/Users/yxie/Desktop/HSX drbx")
#: the step-5 compact_c3 campaign folder (``artifact/N{n}``)
STEP4 = WORKSPACE / "work/p08-step5-compact-c3-6c4de657-20261002T150857Z-27485"
RAW_CHUNK, FACE_CHUNK = 4096, 8192


class Geometry(NamedTuple):
    """Device inputs of ``source.py`` (a pytree; ``Q`` raw cells, ``F`` non-collapsed P07 faces, ``N`` owners)."""
    points: object            # (Q, 3) raw midpoints (u, theta, eta)
    raw_owner: object         # (Q,) int
    raw_volume: object        # (Q,)
    h: object                 # (Q, 3) bcov / B
    jac: object               # (Q,) |J|
    B: object                 # (Q,)
    K: object                 # (Q, 3) curvature vector
    evolution_weight: object  # (Q,)
    owner_volume: object      # (N,)
    face_points: object       # (F, 9, 3)
    face_integrand: object    # (F, 9, 3)
    face_lower: object        # (F,) int, -1 = none
    face_upper: object        # (F,) int, -1 = none
    dirichlet_points: object  # (Qd, 3)
    neumann_points: object    # (Qn, 3)
    neumann_a: object         # (Qn, 3) physical-normal coefficients


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_env(n: int, workspace=WORKSPACE):
    """The step-5 compact_c3 environment (``cc.operator_options``: autodiff K, compact_c3 B)."""
    from p_shared import replay_support as rs
    from p08_step5_compact_c3 import campaign as cc
    return rs.build_environment(n=n, input_root=Path(workspace), sidecar_path=rs.DEFAULT_SIDECAR,
                                **cc.operator_options(cc.config()))


def plan_points(env, n: int, step4=STEP4):
    """``(dirichlet_points, neumann_points)`` of the lowered full-grid plan (the plan is freed)."""
    from p08_step2_global import replay
    artifact = replay.load_artifact(Path(step4) / "artifact", n)
    plan, _ = replay.lower_plan(env, artifact, Path(step4) / "artifact", n, device_put=False)
    out = np.array(plan.dirichlet_points, dtype=np.float64), np.array(plan.neumann_points, dtype=np.float64)
    del plan, artifact
    gc.collect()
    return out


def raw_arrays(env, support) -> dict:
    """``OwnerSupport.raw_geometry`` of the whole raw set, in chunks (same per-point calls)."""
    from perpendicular_structured.reference_geometry import curvature_geometry
    from p07_diffusion_global.numerics import quadrature
    pts = support.points
    cols = {k: [] for k in ("h", "jac", "B", "K", "J")}
    for s in range(0, len(pts), RAW_CHUNK):
        p = pts[s:s + RAW_CHUNK]
        metric = env.ref._metric(p)
        prepared = curvature_geometry(env.ref, p)
        cols["h"].append(np.asarray(metric["bcov"] / metric["B"][:, None]))
        cols["jac"].append(np.abs(np.asarray(metric["J"])))
        cols["B"].append(np.asarray(prepared.B))
        cols["K"].append(np.asarray(prepared.K))
        cols["J"].append(np.asarray(prepared.J))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    _p, q1_weight = quadrature(env.t.faces, support.raw_keys, 1, face=False)
    out["evolution_weight"] = q1_weight.reshape(-1) * out.pop("J") / np.maximum(out["B"], 1.0e-30)
    return out


def face_arrays(env, support) -> dict:
    """Non-collapsed P07 faces: q3 points, integrand and owners (``diffusion_reference``'s construction, chunked)."""
    from drbx.geometry.fci_perpendicular_integrated_rows import contract_face_tensor
    from p_shared import provider
    sel = np.flatnonzero(support.p07_family != 0)
    keys = support.p07_keys[sel]
    pts, integ = [], []
    for s in range(0, len(sel), FACE_CHUNK):
        k = keys[s:s + FACE_CHUNK]
        points, weight = provider._quadrature(env.t.faces, k, 3, face=True)
        tensor = env.ref._perpendicular_flux_tensor(points.reshape(-1, 3)).reshape(len(k), 9, 3, 3)
        pts.append(points)
        integ.append(np.asarray(contract_face_tensor(weight, tensor, k[:, 0])))
    return {"face_points": np.concatenate(pts), "face_integrand": np.concatenate(integ),
            "face_lower": support.p07_lower[sel].astype(np.int64), "face_upper": support.p07_upper[sel].astype(np.int64)}


def precompute(n: int = 32, out_dir=None, *, step4=STEP4, workspace=WORKSPACE, log=_log) -> dict:
    """Compute (and save to ``out_dir/N{n}.geometry.npz`` when given) the geometry of all owners of grid ``n``.
    Returns ``{"arrays": {name: np.ndarray}, "eta_period": float, "seconds": {...}}``."""
    from p_shared import perpendicular_reference_rhs as prr
    sec, t0 = {}, time.perf_counter()

    def lap(name):
        nonlocal t0
        sec[name] = time.perf_counter() - t0
        t0 = time.perf_counter()
        log(f"N{n}: {name} {sec[name]:.1f} s")

    env = build_env(n, workspace)
    lap("environment")
    support = prr.owner_support(env, np.arange(len(env.t.vol)))
    arrays = {"points": support.points, "raw_owner": support.raw_owner, "raw_volume": support.raw_volume,
              "owner_volume": support.owner_volume, "raw_ids": support.raw_ids}
    arrays.update(raw_arrays(env, support))
    lap("raw_geometry")
    arrays.update(face_arrays(env, support))
    lap("faces")
    arrays["dirichlet_points"], arrays["neumann_points"] = plan_points(env, n, step4)
    lap("plan_points")
    arrays["neumann_a"] = np.asarray(env.normal_coefficients(arrays["neumann_points"]), dtype=np.float64)
    lap("normal_coefficients")
    eta_period = float(env.t.g.eta_period)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(out_dir) / f"N{n}.geometry.npz", eta_period=eta_period, **arrays)
        lap("save")
    return {"arrays": arrays, "eta_period": eta_period, "seconds": sec, "env": env, "support": support}


def load(path, device: bool = True) -> tuple:
    """``(Geometry, eta_period)`` from ``N{n}.geometry.npz`` (``jax`` device arrays when ``device``)."""
    import jax.numpy as jnp
    with np.load(path, allow_pickle=False) as z:
        eta_period = float(z["eta_period"])
        data = {k: (jnp.asarray(z[k]) if device else z[k].copy()) for k in Geometry._fields}
    return Geometry(**data), eta_period


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    res = precompute(args.n, args.out)
    print({k: v.shape for k, v in res["arrays"].items()}, res["seconds"])


if __name__ == "__main__":
    main()
