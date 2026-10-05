"""Slow checks of the nodal SBP Laplacian on the smooth analytic testbed: static convergence orders (Dirichlet, conormal and
physical-normal Neumann), the PCG iteration count and the lowest-eigenvalue audit at the larger family-A sizes."""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from drbx.native import fci_perpendicular_sbp_laplacian as lap
from drbx.native import fci_perpendicular_sbp_laplacian_solve as sol
from drbx.validation.sbp_laplacian_audit import LaplacianAssembly, audit_laplacian_plan
from tests import sbp_laplacian_testbed as tb

pytestmark = pytest.mark.slow

NS = (16, 24, 32)
N_ETA = 16


def _orders(errs):
    errs = np.asarray(errs)
    return [np.log(errs[i] / errs[i + 1]) / np.log(NS[i + 1] / NS[i]) for i in range(len(NS) - 1)]


def test_static_orders_of_dirichlet_and_neumann_actions_on_the_smooth_testbed():
    act = jax.jit(lap.laplacian_action, static_argnames=("kinds", "neumann_mode"))
    errs = {k: [] for k in ("D", "N", "Np", "Dc")}
    for n in NS:
        c = tb.case(n, N_ETA)
        runs = {"D": (lap.LaplacianBoundaryData(value=(c.wall_val,)), "dirichlet", None, c.lap),
                "N": (lap.LaplacianBoundaryData(conormal=(c.wall_conormal,)), "neumann", None, c.lap),
                "Np": (lap.LaplacianBoundaryData(normal_derivative=(c.wall_normal,)), "neumann", None, c.lap),
                "Dc": (lap.LaplacianBoundaryData(value=(c.wall_val,)), "dirichlet", c.coeff, c.lap_c)}
        for key, (bcd, kind, coeff, ref) in runs.items():
            out = np.asarray(act(c.plan, c.vals, bcd, (kind,) * 3, coeff))
            errs[key].append([c.h_rel_error(out[..., i] - ref[..., i], ref[..., i]) for i in (0, 1)])    # n and Ti
    for key, e in errs.items():
        e = np.asarray(e)
        assert (np.diff(e, axis=0) < 0).all(), key
    for key in ("D", "Dc"):                                       # Dirichlet: about 2.3 or better (prototype 2.3 - 4.3)
        o = np.asarray(_orders(errs[key]))
        assert (o >= 2.1).all(), (key, o)
    for key in ("N", "Np"):                                       # Neumann: 1.6 - 2.8 (the degree-2 norm at block boundaries)
        o = np.asarray(_orders(errs[key]))
        assert (o >= 1.4).all(), (key, o)


def test_pcg_iterations_and_definiteness_audit_at_n32():
    c = tb.case(32, N_ETA)
    prec = sol.build_dirichlet_preconditioner(c.plan)
    bcd = lap.LaplacianBoundaryData(value=(c.wall_val[..., 0],))
    x, info = sol.solve_dirichlet_jit(c.plan, c.lap[..., 0], bcd, prec, rtol=1e-10, maxit=100)
    assert bool(info["converged"]) and int(info["iterations"]) <= 12
    assert c.h_rel_error(np.asarray(x) - c.vals[..., 0], c.vals[..., 0]) <= 0.02
    res = audit_laplacian_plan(c.plan, k=3)                                   # 15 232 unknowns: lobpcg path
    assert res["flags"] == [] and res["dirichlet"]["method"] == "lobpcg" and res["dirichlet"]["lowest"][0] > 1.0
    assert res["neumann"]["null_defect"] <= 1e-14 and res["neumann"]["lowest"][0] > 0.1
    assert res["seconds"] < 120
    asm = LaplacianAssembly(c.plan)
    assert asm.matrix("dirichlet").nnz < 5e7


def test_static_orders_with_faces_evaluated_are_not_worse_than_interpolated():
    """Evaluating the tensor at the eta half planes (and at every face) keeps the orders of the interpolated faces (analytic testbed)."""
    act = jax.jit(lap.laplacian_action, static_argnames=("kinds", "neumann_mode"))
    errs = {(k, v): [] for k in ("D", "N") for v in ("interp", "ee", "all")}
    for n in NS:
        c = tb.case(n, N_ETA)
        plans = {"interp": c.plan, "ee": c.faces_plan(False, False), "all": c.faces_plan(True, True)}
        runs = {"D": (lap.LaplacianBoundaryData(value=(c.wall_val,)), "dirichlet"),
                "N": (lap.LaplacianBoundaryData(conormal=(c.wall_conormal,)), "neumann")}
        for (key, (bcd, kind)), (vname, plan) in ((a, b) for a in runs.items() for b in plans.items()):
            out = np.asarray(act(plan, c.vals, bcd, (kind,) * 3, None))
            errs[key, vname].append([c.h_rel_error(out[..., i] - c.lap[..., i], c.lap[..., i]) for i in (0, 1)])
    for key in ("D", "N"):
        base = np.asarray(_orders(errs[key, "interp"]))
        for vname in ("ee", "all"):
            e = np.asarray(errs[key, vname])
            assert (np.diff(e, axis=0) < 0).all(), (key, vname)
            assert (e[-1] <= 1.05 * np.asarray(errs[key, "interp"])[-1]).all(), (key, vname)
            assert (np.asarray(_orders(e)) >= base - 0.15).all(), (key, vname, _orders(e), base)
