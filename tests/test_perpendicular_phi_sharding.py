"""eta-sharded potential solve (P08 step 6, stage B): lowering, preconditioner, matvec and solve (fast).

A toy plane-structured P07 operator (``tests/perpendicular_phi_sharding_case.py``: periodic eta, coupling to the planes
+-1 and +-2, owner ids interleaving the planes) is solved with the single-device path and eta-sharded over 1, 2 and 4
shards (multi-device cases run in a subprocess with forced host devices: the device count must be set before JAX is
imported). Checked per shard count: the stacked preconditioner factors equal the plane slices of the global ones and
apply to the same vector; one matvec and the boundary term equal the global ones; the solve takes the same number of
iterations (+-1) and agrees to ``10 rtol ||phi||_M``. Host-side lowering checks (halo window, divisibility) run
in-process. The real N32 / N48 exports are in ``tests/test_perpendicular_phi_sharding_real.py`` (removed 4 October 2026: it needed the oracle arrays retired on 2 October) (slow).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from drbx.native.fci_perpendicular_phi_sharding import PHI_HALO, shard_phi_solver
from drbx.native.fci_perpendicular_phi_solver import phi_solver_from_operator
from tests import perpendicular_phi_sharding_case as case

CASE = Path(case.__file__).resolve()


def _subprocess(devices: int = 4, timeout: int = 900) -> dict:
    env = dict(os.environ)
    env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={devices}".strip()
    env["JAX_PLATFORMS"] = "cpu"
    done = subprocess.run([sys.executable, str(CASE)], capture_output=True, text=True, env=env, timeout=timeout,
                          check=False)
    assert done.returncode == 0, f"subprocess failed ({done.returncode}):\n{done.stderr[-4000:]}"
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def report():
    return _subprocess()


def test_phi_halo_is_two():
    assert PHI_HALO == 2


@pytest.mark.parametrize("sz", (1, 2, 4))
def test_preconditioner_matches_global_planes(report, sz):
    e = report[f"Sz{sz}"]
    assert report["devices"] == 4
    assert e["factor_max_diff"] == 0.0                                       # the plane slices of the global factors
    assert e["prec_shapes"]["P"] == case.NG // sz and e["prec_shapes"]["n_owners"] == case.NG ** 3 // sz
    assert e["prec_apply_rel"] <= 1e-6                                       # float32 factors, same arithmetic per plane


@pytest.mark.parametrize("sz", (1, 2, 4))
def test_matvec_and_boundary_term_equal_global(report, sz):
    e = report[f"Sz{sz}"]
    assert e["matvec_rel"] <= 1e-15, e["matvec_rel"]
    assert e["boundary_rel"] <= 1e-15, e["boundary_rel"]


@pytest.mark.parametrize("sz", (1, 2, 4))
def test_sharded_solve_matches_single_device(report, sz):
    single, e = report["single"], report[f"Sz{sz}"]
    assert single["converged"] and e["converged"]
    assert abs(e["iterations"] - single["iterations"]) <= 1, (e["iterations"], single["iterations"])
    assert e["dphi_m"] <= 10 * case.RTOL * e["phi_m"], (e["dphi_m"], e["phi_m"])
    assert e["relative_residual"] <= case.RTOL and e["host_residual_rel"] <= case.RTOL * (1 + 1e-6)
    assert abs(e["residual_norm"] / e["rhs_norm"] - e["relative_residual"]) <= 1e-12
    assert e["true_error_rel"] < 1e-5
    assert e["warm_iterations"] == 0                                         # warm start from the solution
    assert abs(e["nobc_iterations"][0] - e["nobc_iterations"][1]) <= 1       # bc = None: g = 0
    assert e["nobc_dphi_m"] <= 10 * case.RTOL * e["nobc_phi_m"]
    assert e["local_rows"] == [case.NG ** 3 // sz] * sz                      # the potential stays sharded


def test_halo_window_is_validated(report):
    assert report["halo1_error"] is not None and "outside the halo window" in report["halo1_error"]
    assert "[-2, 2]" in report["halo1_error"]                                # the offending plane offsets are reported


def test_lowering_errors_and_metadata():
    op, raw = case.build_operator()
    solver = phi_solver_from_operator(op, raw, case.NG, rtol=case.RTOL)
    with pytest.raises(ValueError, match="divisible"):
        shard_phi_solver(solver, raw, case.NG, 3)
    with pytest.raises(ValueError, match="halo"):
        shard_phi_solver(solver, raw, case.NG, 8)                            # one plane per shard < halo 2
    sharded = shard_phi_solver(solver, raw, case.NG, 4)
    assert sharded.n_shards == 4 and sharded.n_owners == case.NG ** 3 and sharded.meta.halo == 2
    assert sharded.meta.window == (2 + 4) * sharded.meta.m and sharded.meta.rows == 2 * sharded.meta.m
    assert sharded.a[0].shape[0] == 4 and sharded.volume.shape == (4, 2 * sharded.meta.m)
    assert {"global_solver", "layout", "matrix", "preconditioner"} <= set(sharded.setup_seconds)
    # the operator input builds the global solver itself
    from_op = shard_phi_solver(op, raw, case.NG, 2, rtol=1e-6, restart=30)
    assert from_op.config.rtol == 1e-6 and from_op.config.restart == 30
    leaves = jax.tree_util.tree_leaves(sharded)
    assert all(x.shape[0] == 4 for x in leaves)                              # every leaf has the shard axis
