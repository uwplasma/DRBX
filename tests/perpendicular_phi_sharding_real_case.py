"""Single-device vs eta-sharded potential solve on a real P07 Dirichlet export (subprocess case).

Executed by ``tests/test_perpendicular_phi_sharding_real.py`` with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``
(``python perpendicular_phi_sharding_real_case.py EXPORT_DIR N SZ [SZ ...]``). The manufactured problem has a smooth
``x_true`` (low Fourier modes in ring / theta / plane), smooth Dirichlet data ``g`` and ``rhs = A x_true + B g``. The
single-device ``solve_phi`` and the sharded solve (each run twice: the first call includes the compilation) are
compared; peak RSS is the process maximum after each stage. One JSON object on the last line.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax                                                                                       # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                                                          # noqa: E402

from drbx.native.fci_perpendicular_p07_sparse import boundary_source, load_p07_sparse            # noqa: E402
from drbx.native.fci_perpendicular_phi_sharding import (                                          # noqa: E402
    shard_phi_solver, sharded_matvec, sharded_solve_phi)
from drbx.native.fci_perpendicular_phi_solver import phi_solver_from_operator, solve_phi          # noqa: E402
from drbx.native.fci_perpendicular_plane_preconditioner import owner_layout                       # noqa: E402
from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData                       # noqa: E402
from drbx.native.fci_perpendicular_sharding import (                                              # noqa: E402
    from_plane_major, make_plane_mesh, to_plane_major)

RTOL = 1e-10


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2 ** 20 if sys.platform == "darwin" else peak / 2 ** 10


def manufactured(op, raw_to_owner, n):
    """Smooth ``x_true`` per owner and smooth Dirichlet data for the points of ``op``."""
    ring, plane, theta = owner_layout(raw_to_owner, n)
    s, th, ph = ring / (n - 1.0), 2 * np.pi * theta / n, 2 * np.pi * plane / n
    x_true = (np.cos(th + 2 * ph + 0.3) * (0.2 + s * (1 - s)) + 0.5 * np.sin(ph) * s
              + 0.3 * np.cos(2 * th - ph) * s ** 2 + 0.1)
    pts = np.asarray(op.dirichlet_points, dtype=np.float64)                  # (rho, theta, eta) of the wall points
    t, e = pts[:, 1], pts[:, 2]
    g_val = 0.3 * np.sin(t + 2 * e + 0.5) + 0.2 * np.cos(e - 0.7) + 0.1 * np.sin(2 * t)
    g_tan = 0.05 * np.stack([np.cos(t - e), np.sin(2 * e + t)], axis=1)
    bc = BoundaryData(g_val[:, None], g_tan[:, :, None], np.zeros((op.neumann_normal.shape[1], 1)))
    return x_true, bc


def m_norm(volume, x) -> float:
    return float(np.sqrt(np.sum(volume * np.asarray(x) ** 2)))


def run(export: Path, n: int, shard_counts) -> dict:
    out: dict = {"n": n, "devices": len(jax.devices()), "rtol": RTOL, "rss_mb": {}}
    gdir = export / f"N{n}"
    op = load_p07_sparse(gdir / "p07_dirichlet.npz")
    with np.load(gdir / "owner_map.npz") as z:
        raw = np.asarray(z["raw_to_owner"])
    out["owners"], out["nnz"] = op.n_owners, int(op.matrix.nnz)
    x_true, bc = manufactured(op, raw, n)
    bt = boundary_source(op, bc)[:, 0]
    rhs = op.matrix @ x_true + bt
    vol = op.owner_volume
    solver = phi_solver_from_operator(op, raw, n, rtol=RTOL)
    out["rss_mb"]["setup"] = rss_mb()
    out["setup_seconds"] = dict(solver.setup_seconds)
    phi_ref, first = solve_phi(solver, rhs, bc)
    _, second = solve_phi(solver, rhs, bc)
    out["single"] = {"iterations": first["iterations"], "converged": first["converged"],
                     "relative_residual": first["relative_residual"], "seconds_first": first["seconds"],
                     "seconds": second["seconds"], "compile_seconds": first["seconds"] - second["seconds"],
                     "true_error_rel": float(np.linalg.norm(phi_ref - x_true) / np.linalg.norm(x_true)),
                     "phi_m": m_norm(vol, phi_ref), "rhs_m": m_norm(vol, rhs - bt)}
    out["rss_mb"]["single"] = rss_mb()
    rng = np.random.default_rng(1)
    rough_rhs = op.matrix @ rng.normal(size=op.n_owners)                        # white-noise solution: many more iterations
    phi_rough_ref, rough_ref = solve_phi(solver, rough_rhs, bc)
    out["single_rough"] = {"iterations": rough_ref["iterations"], "converged": rough_ref["converged"],
                           "relative_residual": rough_ref["relative_residual"], "phi_m": m_norm(vol, phi_rough_ref)}
    v = rng.normal(size=op.n_owners)
    av = op.matrix @ v
    for sz in shard_counts:
        mesh = make_plane_mesh(sz)
        t0 = time.perf_counter()
        sharded = shard_phi_solver(solver, raw, n, sz, mesh=mesh)
        shard_seconds = time.perf_counter() - t0
        perm, inverse = sharded.perm, sharded.inverse
        factor = 0.0
        p = n // sz
        for name in ("lo", "up", "dinv"):
            got, ref = np.asarray(getattr(sharded.prec, name)), np.asarray(getattr(solver.prec, name))
            factor = max([factor] + [float(np.max(np.abs(got[s] - ref[:, s * p:(s + 1) * p]))) for s in range(sz)])
        got = from_plane_major(np.asarray(sharded_matvec(sharded, to_plane_major(v, inverse), mesh)), perm)
        rhs_pm = to_plane_major(rhs, inverse)
        phi_pm, one = sharded_solve_phi(sharded, rhs_pm, bc, mesh=mesh)
        _, two = sharded_solve_phi(sharded, rhs_pm, bc, mesh=mesh)
        phi = from_plane_major(np.asarray(phi_pm), perm)
        res = rhs - bt - op.matrix @ phi
        phi_rough_pm, rough = sharded_solve_phi(sharded, to_plane_major(rough_rhs, inverse), bc, mesh=mesh)
        phi_rough = from_plane_major(np.asarray(phi_rough_pm), perm)
        out[f"Sz{sz}"] = {
            "iterations": one["iterations"], "converged": one["converged"],
            "relative_residual": one["relative_residual"],
            "host_residual_rel": m_norm(vol, res) / m_norm(vol, rhs - bt),
            "dphi_m": m_norm(vol, phi - phi_ref), "phi_m": m_norm(vol, phi_ref),
            "true_error_rel": float(np.linalg.norm(phi - x_true) / np.linalg.norm(x_true)),
            "seconds_first": one["seconds"], "seconds": two["seconds"],
            "compile_seconds": one["seconds"] - two["seconds"], "lowering_seconds": shard_seconds,
            "lowering_breakdown": dict(sharded.setup_seconds), "factor_max_diff": factor,
            "matvec_rel": float(np.max(np.abs(got - av)) / np.max(np.abs(av))),
            "rough": {"iterations": rough["iterations"], "converged": rough["converged"],
                      "relative_residual": rough["relative_residual"], "seconds": rough["seconds"],
                      "dphi_m": m_norm(vol, phi_rough - phi_rough_ref), "phi_m": m_norm(vol, phi_rough_ref)},
            "nnz_padded": sharded.info["nnz_padded"], "nnz_per_shard": sharded.info["nnz_per_shard"]}
        out["rss_mb"][f"Sz{sz}"] = rss_mb()
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    export, n, counts = Path(argv[0]), int(argv[1]), tuple(int(x) for x in argv[2:])
    print(json.dumps(run(export, n, counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
