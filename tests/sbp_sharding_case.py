"""Multi-device cases of the eta-sharded nodal SBP bracket, curvature, Laplacian and CG (run in a subprocess with forced host devices).

``tests/test_fci_perpendicular_sbp_sharding.py`` runs this module as a script with
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` (set before JAX is imported) and reads the JSON on the last
stdout line. Modes: ``agree`` (sharded against single-device ``sbp_bracket`` for 1/2/4 shards), ``family_a`` (the same
for the family-A n = 16 layout with a Zernike core) and ``short`` (a plan with fewer than 3 planes per shard is rejected).
Further modes: ``curvature`` (``sharded_sbp_curvature`` against ``sbp_curvature``), ``laplacian`` (``sharded_laplacian_form`` /
``action``, Dirichlet and both Neumann data types, against the single-device form), ``cg`` (``sharded_solve_dirichlet`` against
``solve_dirichlet``) and ``laplacian_short`` (fewer than 4 planes per shard is rejected).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jax                                                                                       # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                                                          # noqa: E402

from drbx.geometry.nodal_families import build_family_a_layout                                    # noqa: E402
from drbx.geometry.nodal_layout import build_nodal_layout                                        # noqa: E402
from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData                           # noqa: E402
from drbx.native.fci_perpendicular_sbp_bracket import sbp_bracket                                # noqa: E402
from drbx.native.fci_perpendicular_sbp_curvature import sbp_curvature                            # noqa: E402
from drbx.native.fci_perpendicular_sbp_laplacian import (                                        # noqa: E402
    LaplacianBoundaryData, laplacian_action, laplacian_form)
from drbx.native.fci_perpendicular_sbp_laplacian_solve import (                                  # noqa: E402
    build_dirichlet_preconditioner, solve_dirichlet)
from drbx.native.fci_perpendicular_sbp_sharding import (                                         # noqa: E402
    make_plane_mesh, shard_core_schur_preconditioner, shard_laplacian_plan, shard_nodal_plan, sharded_laplacian_action,
    sharded_laplacian_form, sharded_sbp_bracket, sharded_sbp_curvature, sharded_solve_dirichlet)
from drbx.geometry.sbp_laplacian import build_laplacian_plan, nodal_laplacian_metric_from_callable  # noqa: E402
from drbx.stencils.nodal_plan import build_nodal_plan, nodal_metric_from_callable                # noqa: E402
from tests import sbp_laplacian_testbed as tb                                                    # noqa: E402

N = 32
LEVELS = [(8, 16, 16), (16, 32, 32)]
RHO = 0.05


def metric(points):
    u, th, et = points.T
    h = np.stack([0.1 * np.sin(th), 0.2 * np.cos(et), 1.0 + 0.1 * u], -1)
    return h, u * (1.0 + 0.1 * np.cos(th) * np.cos(et)), 1.0 + 0 * u, np.zeros((len(u), 3))


def build(n_eta):
    lay = build_nodal_layout(N, LEVELS, n_eta=n_eta, inner="wall")
    return lay, build_nodal_plan(lay, nodal_metric_from_callable(lay, metric))


def run_agree(n_eta=16, shard_counts=(1, 2, 4)) -> dict:
    lay, plan = build(n_eta)
    rng = np.random.default_rng(11)
    phi = rng.standard_normal((n_eta, lay.P))
    g = rng.standard_normal((n_eta, lay.P, 2))
    bcd = SatBoundaryData(tuple(jnp.asarray(rng.standard_normal((n_eta, w[3], 2))) for w in plan.structure.walls), None)
    ref = np.asarray(jax.jit(sbp_bracket)(plan, phi, g, bcd, RHO))
    out = {}
    for s in shard_counts:
        mesh = make_plane_mesh(s)
        sharded = shard_nodal_plan(plan, s, mesh)
        got = np.asarray(jax.jit(lambda sp, ph, gg, b, r: sharded_sbp_bracket(sp, ph, gg, b, r, mesh))(
            sharded, phi, g, bcd, RHO))
        out[f"S{s}"] = float(np.abs(got - ref).max() / np.abs(ref).max())
        out[f"S{s}_bitwise"] = bool(np.array_equal(got, ref))
    out["ref_max"] = float(np.abs(ref).max())
    return out


def run_family_a(n_eta=16, shard_counts=(1, 2, 4)) -> dict:
    """Family A, n = 16 (core K = 2, p = 4): core D5c, core shell damping (traced ``c_kappa``) and ring levels."""
    lay = build_family_a_layout(16, n_eta=n_eta)
    plan = build_nodal_plan(lay, nodal_metric_from_callable(lay, metric))
    rng = np.random.default_rng(13)
    phi = rng.standard_normal((n_eta, lay.P))
    g = rng.standard_normal((n_eta, lay.P, 2))
    bcd = SatBoundaryData(tuple(jnp.asarray(rng.standard_normal((n_eta, w[3], 2))) for w in plan.structure.walls), None)
    ref = np.asarray(jax.jit(sbp_bracket)(plan, phi, g, bcd, RHO, 0.7))
    out = {"P": lay.P}
    for s in shard_counts:
        mesh = make_plane_mesh(s)
        sharded = shard_nodal_plan(plan, s, mesh)
        got = np.asarray(jax.jit(lambda sp, ph, gg, b, r, ck: sharded_sbp_bracket(sp, ph, gg, b, r, mesh, ck))(
            sharded, phi, g, bcd, RHO, 0.7))
        out[f"S{s}"] = float(np.abs(got - ref).max() / np.abs(ref).max())
    out["ref_max"] = float(np.abs(ref).max())
    return out


def _compare(out: dict, key: str, got, ref) -> None:
    got, ref = np.asarray(got), np.asarray(ref)
    out[key] = float(np.abs(got - ref).max() / np.abs(ref).max())
    out[key + "_bitwise"] = bool(np.array_equal(got, ref))


def metric_k(points):
    """Metric with a nonzero curvature vector ``K`` (the curvature flux ``|J| K / B`` is nonzero)."""
    u, th, et = points.T
    h = np.stack([0.1 * np.sin(th), 0.2 * np.cos(et), 1.0 + 0.1 * u], -1)
    K = np.stack([0.3 * np.cos(th), 0.1 * u + 0.05, 0.2 * np.sin(et) + 0.1], -1)
    return h, u * (1.0 + 0.1 * np.cos(th) * np.cos(et)), 1.0 + 0.2 * u, K


def run_curvature(n_eta=16, shard_counts=(1, 2, 4)) -> dict:
    """Family A n = 16 (core): default, ``phi_plus_tau_pi`` with ``tau != 1``, and jump dissipation with the core damping."""
    lay = build_family_a_layout(16, n_eta=n_eta)
    plan = build_nodal_plan(lay, nodal_metric_from_callable(lay, metric_k))
    rng = np.random.default_rng(17)
    q = np.concatenate([1.0 + 0.3 * rng.random((n_eta, lay.P, 3)), rng.standard_normal((n_eta, lay.P, 2))], axis=-1)
    bcd = SatBoundaryData(tuple(jnp.asarray(rng.standard_normal((n_eta, w[3], 4))) for w in plan.structure.walls), None)
    options = {"default": {}, "pi": {"psi": "phi_plus_tau_pi", "tau": 0.8},
               "jump_core": {"jump_dissipation": True, "c_kappa": 0.7}}
    out = {"P": lay.P}
    for name, kw in options.items():
        ref = jax.jit(lambda p, qq, b, kw=kw: sbp_curvature(p, qq, b, **kw))(plan, q, bcd)
        out[f"{name}_ref_max"] = float(np.abs(np.asarray(ref)).max())
        for s in shard_counts:
            mesh = make_plane_mesh(s)
            sharded = shard_nodal_plan(plan, s, mesh)
            got = jax.jit(lambda sp, qq, b, kw=kw, mesh=mesh: sharded_sbp_curvature(sp, qq, b, mesh, **kw))(sharded, q, bcd)
            _compare(out, f"{name}_S{s}", got, ref)
    # op-by-op (no XLA fusion): the sharded and the single-device curvature agree bitwise
    ref = np.asarray(sbp_curvature(plan, q, bcd))
    for s in shard_counts[-1:]:                      # eager shard_map is slow (about 12 s per call): the largest count only
        mesh = make_plane_mesh(s)
        _compare(out, f"eager_S{s}", sharded_sbp_curvature(shard_nodal_plan(plan, s, mesh), q, bcd, mesh), ref)
    return out


def laplacian_case(n_eta=16):
    lay = build_family_a_layout(16, n_eta=n_eta)
    return lay, build_laplacian_plan(lay, nodal_laplacian_metric_from_callable(lay, tb.synthetic_geometry))


def run_laplacian(n_eta=16, shard_counts=(1, 2, 4)) -> dict:
    """Form and action: Dirichlet, mixed Dirichlet / conormal-Neumann, and physical-normal Neumann (two fields)."""
    lay, lp = laplacian_case(n_eta)
    N = lp.structure.N
    rng = np.random.default_rng(19)
    f = rng.standard_normal((n_eta, lay.P, 2))
    gD, gN = (rng.standard_normal((n_eta, N, 2)) for _ in range(2))
    cases = {"dirichlet": (LaplacianBoundaryData(value=(gD,)), "dirichlet", None),
             "conormal": (LaplacianBoundaryData(value=(gD,), conormal=(gN,)), ("dirichlet", "neumann"), None),
             "physical": (LaplacianBoundaryData(value=(gD,), normal_derivative=(gN,)), ("neumann", "dirichlet"), None),
             "nodata": (None, "dirichlet", 2.0)}
    out = {"P": lay.P}
    for name, (bcd, kinds, ck) in cases.items():
        ck = 1.0 if ck is None else ck
        first = None
        ref_form = jax.jit(lambda l, ff, b, kinds=kinds, ck=ck: laplacian_form(l, ff, b, kinds, None, ck))(lp, f, bcd)
        ref_act = jax.jit(lambda l, ff, b, kinds=kinds, ck=ck: laplacian_action(l, ff, b, kinds, None, ck))(lp, f, bcd)
        out[f"{name}_ref_max"] = float(np.abs(np.asarray(ref_form)).max())
        for s in shard_counts:
            mesh = make_plane_mesh(s)
            sharded = shard_laplacian_plan(lp, s, mesh)
            got = jax.jit(lambda sp, ff, b, mesh=mesh, kinds=kinds, ck=ck: sharded_laplacian_form(sp, ff, b, mesh, kinds, None, ck))(
                sharded, f, bcd)
            _compare(out, f"{name}_form_S{s}", got, ref_form)
            first = np.asarray(got) if first is None else first
            out[f"{name}_form_S{s}_same_as_S1"] = bool(np.array_equal(np.asarray(got), first))
            got = jax.jit(lambda sp, ff, b, mesh=mesh, kinds=kinds, ck=ck: sharded_laplacian_action(sp, ff, b, mesh, kinds, None, ck))(
                sharded, f, bcd)
            _compare(out, f"{name}_action_S{s}", got, ref_act)
    return out


def run_cg(n_eta=16, shard_counts=(1, 2, 4)) -> dict:
    """``sharded_solve_dirichlet`` against ``solve_dirichlet`` (random right-hand side and wall data, cold and warm start)."""
    lay, lp = laplacian_case(n_eta)
    N = lp.structure.N
    rng = np.random.default_rng(23)
    s = rng.standard_normal((n_eta, lay.P))
    g = rng.standard_normal((n_eta, N))
    bcd = LaplacianBoundaryData(value=(g,))
    prec = build_dirichlet_preconditioner(lp, rings_per_block=2, group_planes=4)
    out = {"P": lay.P}
    x0 = 0.1 * rng.standard_normal((n_eta, lay.P))
    for tag, start in (("cold", None), ("warm", x0)):
        ref, ri = jax.jit(lambda l, ss, b, pr, xs: solve_dirichlet(l, ss, b, pr, x0=xs, rtol=1e-11))(lp, s, bcd, prec, start)
        ref = np.asarray(ref)
        out[f"{tag}_ref_iterations"] = int(ri["iterations"])
        out[f"{tag}_ref_converged"] = bool(ri["converged"])
        for sh in shard_counts:
            mesh = make_plane_mesh(sh)
            sharded = shard_laplacian_plan(lp, sh, mesh)
            pr = shard_core_schur_preconditioner(prec, sh, mesh)
            x, info = jax.jit(lambda sp, ss, b, p_, xs, mesh=mesh: sharded_solve_dirichlet(sp, ss, b, p_, mesh, x0=xs, rtol=1e-11))(
                sharded, s, bcd, pr, start)
            x = np.asarray(x)
            out[f"{tag}_S{sh}_iterations"] = int(info["iterations"])
            out[f"{tag}_S{sh}_converged"] = bool(info["converged"])
            out[f"{tag}_S{sh}_rel_res"] = float(info["relative_residual"])
            out[f"{tag}_S{sh}"] = float(np.abs(x - ref).max() / np.abs(ref).max())
    return out


def run_laplacian_short() -> dict:
    errors = {}
    for n_eta, s in ((8, 4), (12, 4), (16, 8)):
        _, lp = laplacian_case(n_eta)
        try:
            shard_laplacian_plan(lp, s)
            errors[f"E{n_eta}_S{s}"] = None
        except ValueError as exc:
            errors[f"E{n_eta}_S{s}"] = str(exc)
    return errors


def run_short() -> dict:
    _, plan = build(8)
    errors = {}
    for s in (4, 8):
        try:
            shard_nodal_plan(plan, s)
            errors[f"S{s}"] = None
        except ValueError as exc:
            errors[f"S{s}"] = str(exc)
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    modes = {"agree": run_agree, "short": run_short, "family_a": run_family_a, "curvature": run_curvature,
             "laplacian": run_laplacian, "cg": run_cg, "laplacian_short": run_laplacian_short}
    parser.add_argument("mode", choices=tuple(modes))
    args = parser.parse_args(argv)
    result = modes[args.mode]()
    result["devices"] = len(jax.devices())
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
