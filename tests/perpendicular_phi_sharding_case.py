"""Multi-device cases of the eta-sharded potential solve (run in a subprocess with forced host devices).

``tests/test_perpendicular_phi_sharding.py`` executes this module as a script with
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` (set before JAX is imported) and reads the JSON on the last stdout
line. The toy P07 operator is the anisotropic flux-form operator of ``tests/test_perpendicular_phi_solver.py`` on an
``n x n x n`` (ring, theta, plane) grid with ``raw_to_owner = identity`` (owner ids interleave the planes), weak coupling to
the planes ``+-1`` *and* ``+-2`` (the halo of the real operator), a skew perturbation, wall conductances and a nonzero
Dirichlet boundary block. For 1, 2 and 4 shards it compares the preconditioner (factors and apply), one matvec, the
boundary term and the solve against the single-device path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax                                                                                       # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                                                          # noqa: E402

from drbx.native.fci_perpendicular_p07_sparse import P07SparseOperator, boundary_source          # noqa: E402
from drbx.native.fci_perpendicular_phi_sharding import (                                          # noqa: E402
    shard_phi_solver, sharded_boundary_term, sharded_matvec, sharded_preconditioner_apply, sharded_solve_phi)
from drbx.native.fci_perpendicular_phi_solver import phi_solver_from_operator, solve_phi          # noqa: E402
from drbx.native.fci_perpendicular_plane_preconditioner import apply_plane_preconditioner         # noqa: E402
from drbx.native.fci_perpendicular_reconstruction_state import BoundaryData                       # noqa: E402
from drbx.native.fci_perpendicular_sharding import (                                              # noqa: E402
    from_plane_major, make_plane_mesh, to_plane_major)

NG = 8
QD = 7                       # Dirichlet trace points
RTOL = 1e-10


def build_operator(n=NG, eps1=2e-2, eps2=5e-3, skew=0.15, seed=0):
    rng = np.random.default_rng(seed)
    nn = n ** 3
    cell = lambda i, j, k: (i * n + j) * n + k                     # raw cell == owner id (ring i, theta j, plane k)
    vol = rng.uniform(0.6, 1.4, nn)
    kr, kc, kv = [], [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for d, coef in (((1, 0, 0), 1.0), ((0, 1, 0), 0.7), ((0, 0, 1), eps1), ((0, 0, 2), eps2)):
                    i2, j2, k2 = i + d[0], j + d[1], (k + d[2]) % n                      # eta is periodic
                    if i2 >= n or j2 >= n:
                        continue
                    a, b = cell(i, j, k), cell(i2, j2, k2)
                    c = coef * rng.uniform(0.8, 1.2)
                    kr += [a, a, b, b, a, b]
                    kc += [a, b, a, b, b, a]
                    kv += [c, -c, -c, c, skew * c, -skew * c]
    wall = [cell(i, j, k) for i in range(n) for j in range(n) for k in range(n) if i in (0, n - 1) or j in (0, n - 1)]
    kr += wall
    kc += wall
    kv += list(rng.uniform(0.8, 1.2, len(wall)))
    kmat = sp.csr_matrix((kv, (kr, kc)), shape=(nn, nn))
    a = (sp.diags(1.0 / vol) @ kmat).tocsr()
    a.sort_indices()
    rows = rng.choice(wall, size=3 * QD, replace=False)
    bval = sp.csr_matrix((rng.uniform(0.2, 0.8, 3 * QD), (rows, np.tile(np.arange(QD), 3))), shape=(nn, QD))
    btan = sp.csr_matrix((rng.uniform(-0.2, 0.2, 2 * QD), (rows[:2 * QD], np.arange(2 * QD))), shape=(nn, 2 * QD))
    op = P07SparseOperator("dirichlet", a, bval, btan, sp.csr_matrix((nn, 0)), vol, np.zeros((QD, 3)),
                           np.zeros((0, 3)))
    return op, np.arange(nn, dtype=np.int64)


def make_bc(seed=3):
    rng = np.random.default_rng(seed)
    return BoundaryData(rng.normal(size=(QD, 1)), rng.normal(size=(QD, 2, 1)), np.zeros((0, 1)))


def m_norm(op, x) -> float:
    return float(np.sqrt(np.sum(op.owner_volume * np.asarray(x) ** 2)))


def run(shard_counts=(1, 2, 4)) -> dict:
    op, raw = build_operator()
    nn = op.n_owners
    solver = phi_solver_from_operator(op, raw, NG, rtol=RTOL)
    rng = np.random.default_rng(5)
    bc = make_bc()
    x_true = rng.normal(size=nn)
    bt = boundary_source(op, bc)[:, 0]
    rhs = op.matrix @ x_true + bt
    phi_ref, info_ref = solve_phi(solver, rhs, bc)
    v = rng.normal(size=nn)
    av_ref = op.matrix @ v
    prec_ref = np.asarray(apply_plane_preconditioner(solver.prec, jnp.asarray(v)))
    out = {"single": {k: info_ref[k] for k in ("iterations", "converged", "relative_residual")},
           "coupling_planes": 2, "n": NG, "owners": nn}
    for sz in shard_counts:
        mesh = make_plane_mesh(sz)
        sharded = shard_phi_solver(solver, raw, NG, sz, mesh=mesh)
        perm, inverse = sharded.perm, sharded.inverse
        p, m = NG // sz, sharded.meta.m
        entry = {}
        # factors: the stacked local factors against the global plane slices
        glob = solver.prec
        diffs = []
        for name in ("lo", "up", "dinv"):
            got, ref = np.asarray(getattr(sharded.prec, name)), np.asarray(getattr(glob, name))
            for s in range(sz):
                diffs.append(float(np.max(np.abs(got[s] - ref[:, s * p:(s + 1) * p]))))
        entry["factor_max_diff"] = max(diffs)
        entry["prec_shapes"] = {"S": sharded.prec.S, "B": sharded.prec.B, "w": sharded.prec.w, "P": sharded.prec.P,
                                "n_owners": sharded.prec.n_owners}
        # preconditioner apply
        got = np.asarray(sharded_preconditioner_apply(sharded, to_plane_major(v, inverse), mesh))
        entry["prec_apply_rel"] = float(np.max(np.abs(from_plane_major(got, perm) - prec_ref)) / np.max(np.abs(prec_ref)))
        # matvec and boundary term
        got = np.asarray(sharded_matvec(sharded, to_plane_major(v, inverse), mesh))
        diff = np.abs(from_plane_major(got, perm) - av_ref)
        entry["matvec_rel"] = float(diff.max() / np.abs(av_ref).max())
        entry["matvec_bitwise"] = bool(np.array_equal(from_plane_major(got, perm), av_ref))
        got = np.asarray(sharded_boundary_term(sharded, bc, mesh))
        entry["boundary_rel"] = float(np.max(np.abs(from_plane_major(got, perm) - bt)) / np.abs(bt).max())
        # solve
        phi_pm, info = sharded_solve_phi(sharded, to_plane_major(rhs, inverse), bc, mesh=mesh)
        phi = from_plane_major(np.asarray(phi_pm), perm)
        entry["iterations"] = info["iterations"]
        entry["converged"] = info["converged"]
        entry["relative_residual"] = info["relative_residual"]
        entry["residual_norm"], entry["rhs_norm"] = info["residual_norm"], info["rhs_norm"]
        entry["dphi_m"] = m_norm(op, phi - phi_ref)
        entry["phi_m"] = m_norm(op, phi_ref)
        entry["true_error_rel"] = float(np.linalg.norm(phi - x_true) / np.linalg.norm(x_true))
        res = rhs - bt - op.matrix @ phi
        entry["host_residual_rel"] = m_norm(op, res) / m_norm(op, rhs - bt)
        # no boundary data, warm start from the solution (zero iterations)
        _, warm = sharded_solve_phi(sharded, to_plane_major(rhs, inverse), bc, x0_pm=phi_pm, mesh=mesh)
        entry["warm_iterations"] = warm["iterations"]
        # no bc: same problem as the single-device path without bc
        ref_nb, info_nb = solve_phi(solver, rhs)
        phi_nb, inf = sharded_solve_phi(sharded, to_plane_major(rhs, inverse), mesh=mesh)
        entry["nobc_iterations"] = [info_nb["iterations"], inf["iterations"]]
        entry["nobc_dphi_m"] = m_norm(op, from_plane_major(np.asarray(phi_nb), perm) - ref_nb)
        entry["nobc_phi_m"] = m_norm(op, ref_nb)
        entry["local_rows"] = [int(x.data.shape[0]) for x in phi_pm.addressable_shards]
        out[f"Sz{sz}"] = entry
    # the halo window is validated: halo 1 cannot hold the planes +-2 of the toy operator
    try:
        shard_phi_solver(solver, raw, NG, 2, halo=1)
        out["halo1_error"] = None
    except ValueError as error:
        out["halo1_error"] = str(error)
    return out


def main(argv=None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    result = run()
    result["devices"] = len(jax.devices())
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
