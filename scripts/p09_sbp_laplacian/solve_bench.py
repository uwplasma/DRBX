#!/usr/bin/env python3
"""phi elliptic controls and the CG benchmark of the nodal Laplacian at one arm and resolution.

1. **Controls.** Dirichlet solve ``L phi = s`` by the package's plane-preconditioned CG (``solve_dirichlet``) for the manufactured
   potentials ``phi_switch`` (gating) and ``phi_wave`` (control) with the exact wall trace:
   ``discrete``: ``s = L_h phi_nodal`` (the solver must return the nodal ``phi`` -- solver error);
   ``continuum``: ``s = R_phi`` the exact Laplacian (solution error = discretisation error of the operator).
   Reported: iterations, the solver's relative ``H^-1`` residual, the recomputed true residual, the H-weighted relative error
   and its regions.
2. **CG benchmark.** Warm per-solve time and iterations at ``rtol`` (compile time separated), the per-iteration split into the
   operator apply and the preconditioner apply (jitted, warm), the preconditioner build time and memory.
   Reference: P07 FGMRES + plane solve (``work/p08_step5_solver_studies_20261001/README.md``).

The preconditioner is built per group of planes and merged into the stock ``PlanePreconditioner`` (``tools.py``) to stay inside
the host memory budget; with ``--groups 1`` it is the stock ``build_dirichlet_preconditioner``.

    python solve_bench.py N --arm raw|filtered [--out ROOT] [--groups G] [--factor-dtype float64|float32]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                                      # noqa: E402
from common import CONFIG, log                                                          # noqa: E402
import fields as Fl                                                                     # noqa: E402
import tools as T                                                                       # noqa: E402

import numpy as np                                                                      # noqa: E402

DEFAULT_GROUPS = {32: 1, 48: 8, 64: 16}


def timeit(fn, reps: int):
    """Warm timing: ``fn()`` must block; returns (min, median) seconds over ``reps`` calls (after one warm-up call)."""
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.min(ts)), float(np.median(ts))


def run(n: int, arm: str, out_root: Path, groups: int, factor_dtype: str, tag: str) -> dict:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from drbx.native import fci_perpendicular_sbp_laplacian_solve as sol
    from drbx.native.fci_perpendicular_plane_preconditioner import apply_plane_preconditioner
    from drbx.native.fci_perpendicular_sbp_laplacian import (LaplacianBoundaryData, laplacian_action_jit, laplacian_form_jit)
    from drbx.native.fci_perpendicular_sbp_norms import region_errors, ring_region_masks

    t0 = time.perf_counter()
    cfg = CONFIG["cg"]
    rtol, maxit, reps = float(cfg["rtol"]), int(cfg["maxit"]), int(cfg["warm_repeats"])
    md = C.load_metric(out_root, arm, n)
    plan = C.build_plan(md)
    st = plan.structure
    E, P = st.n_eta, st.P
    H = np.asarray(plan.Hp) * st.deta
    plan_np = plan
    plan = jax.tree_util.tree_map(jnp.asarray, plan)           # device-resident plan (as in production), not copied per call
    masks = ring_region_masks(md.layout, n)
    res = dict(n=n, arm=arm, groups=groups, factor_dtype=factor_dtype, rtol=rtol, unknowns=E * P, metric_identity=md.identity)

    # --- preconditioner build
    rss0 = C.peak_rss_gib()
    tb = time.perf_counter()
    prec = T.build_merged_preconditioner(plan_np, groups, factor_dtype, log=log)
    t_build = time.perf_counter() - tb
    info = {k: v for k, v in prec.info.items() if k != "per_group"}
    res["preconditioner"] = dict(build_seconds=t_build, rss_before_gib=rss0, peak_rss_after_build_gib=C.peak_rss_gib(),
                                 storage_gib=prec.nbytes / 2 ** 30, info=info, per_group=prec.info.get("per_group"))
    log(f"{arm} N{n}: preconditioner built {t_build:.1f}s, storage {prec.nbytes / 2 ** 30:.2f} GiB ({factor_dtype}), "
        f"S={prec.meta.S} B={prec.meta.B} w={prec.meta.w}, peak rss {C.peak_rss_gib():.2f}")

    # --- controls
    controls = {}
    phis = {}
    for cname in ("transverse", "transverse_wave"):
        cat = Fl.make_catalogue(cname)
        data = Fl.NodalData(md, cat)
        f = cat.names.index("phi")
        phis[cname] = dict(V=data.V[f], R=data.R[f], wall=data.wall_value[..., f])
    solve = jax.jit(lambda lp, s, g, pr, x0: sol.solve_dirichlet(lp, s, LaplacianBoundaryData(value=(g,)), pr, rtol=rtol,
                                                                 maxit=maxit, x0=x0))
    zero = jnp.zeros((E, P))
    for cname, d in phis.items():
        phi, g = jnp.asarray(d["V"]), jnp.asarray(d["wall"])
        for case in ("discrete", "continuum"):
            if case == "discrete":
                s = laplacian_action_jit(plan, phi, LaplacianBoundaryData(value=(g,)), "dirichlet", None, 1.0)
            else:
                s = jnp.asarray(d["R"])
            ts = time.perf_counter()
            x, inf = solve(plan, s, g, prec, zero)
            x = np.asarray(x)
            sec = time.perf_counter() - ts
            # true residual of the discrete problem L x = s (the package's own apply), relative to |s|_H
            Lx = np.asarray(laplacian_action_jit(plan, jnp.asarray(x), LaplacianBoundaryData(value=(g,)), "dirichlet", None, 1.0))
            sn = np.asarray(s)
            err = x - d["V"]
            controls[f"{cname}/{case}"] = dict(
                iterations=int(inf["iterations"]), solver_relative_residual_hinv=float(inf["relative_residual"]),
                converged=bool(inf["converged"]), true_residual_L2_rel=float(np.sqrt(np.sum(H * (Lx - sn) ** 2) / np.sum(H * sn ** 2))),
                solution_error_rel=float(np.sqrt(np.sum(H * err ** 2) / np.sum(H * d["V"] ** 2))),
                solution_error_max=float(np.abs(err).max()), seconds_incl_compile_first=sec,
                regions=region_errors(err, d["V"], H, masks))
            c = controls[f"{cname}/{case}"]
            log(f"  control {cname}/{case}: it={c['iterations']} res={c['solver_relative_residual_hinv']:.2e} "
                f"true_res={c['true_residual_L2_rel']:.2e} err={c['solution_error_rel']:.3e} ({sec:.1f}s)")
    res["controls"] = controls

    # --- CG benchmark (rhs: continuum rhs of phi_switch, zero initial guess)
    d = phis["transverse"]
    phi, g, s = jnp.asarray(d["V"]), jnp.asarray(d["wall"]), jnp.asarray(d["R"])
    jax.clear_caches()                                   # the first call below traces and compiles the solve afresh
    solve = jax.jit(lambda lp, s, g, pr, x0: sol.solve_dirichlet(lp, s, LaplacianBoundaryData(value=(g,)), pr, rtol=rtol,
                                                                 maxit=maxit, x0=x0))
    tc = time.perf_counter()
    x, inf = solve(plan, s, g, prec, zero)
    jax.block_until_ready(x)
    t_first = time.perf_counter() - tc

    def one():
        xx, ii = solve(plan, s, g, prec, zero)
        jax.block_until_ready(xx)
        return ii

    tmin, tmed = timeit(one, reps)
    its = int(one()["iterations"])
    res["cg"] = dict(first_call_seconds=t_first, warm_min_seconds=tmin, warm_median_seconds=tmed, compile_seconds_estimate=t_first - tmed,
                     iterations=its, rtol=rtol, seconds_per_iteration=tmed / max(its, 1))
    # per-iteration split: operator apply (the jitted energy form), preconditioner apply, vector updates
    Mv = jax.jit(lambda lp, v: laplacian_form_jit(lp, v, None, "dirichlet", None, 1.0))
    Pa = jax.jit(lambda pr, r: apply_plane_preconditioner(pr, r.reshape(-1)).reshape(E, P))
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.standard_normal((E, P)))
    Hj = jnp.asarray(H)

    def vec_ops(x, r, p, z, Ap):
        alpha = jnp.sum(r * z) / jnp.sum(p * Ap)
        x = x + alpha * p
        rn = r - alpha * Ap
        zn = z * 1.0
        rz = jnp.sum(rn * zn)
        beta = jnp.maximum(jnp.sum(zn * (rn - r)) / jnp.sum(r * z), 0.0)
        nrm = jnp.sqrt(jnp.sum(rn * rn / Hj))
        return x, rn, zn + beta * p, rz, nrm

    Vo = jax.jit(vec_ops)
    t_mv = timeit(lambda: jax.block_until_ready(Mv(plan, v)), 20)
    t_pa = timeit(lambda: jax.block_until_ready(Pa(prec, v)), 20)
    t_vo = timeit(lambda: jax.block_until_ready(Vo(v, v, v, v, v)), 20)
    res["cg"]["per_iteration"] = dict(operator_apply_min=t_mv[0], operator_apply_median=t_mv[1], preconditioner_apply_min=t_pa[0],
                                      preconditioner_apply_median=t_pa[1], vector_ops_median=t_vo[1],
                                      sum_median=t_mv[1] + t_pa[1] + t_vo[1])
    res["cg"]["setup_ops_estimate_seconds"] = 2 * t_mv[1] + t_pa[1]
    res["baseline"] = dict(fgmres_plane_seconds=CONFIG["cg"]["baseline_fgmres_plane_seconds"][str(n)],
                           fgmres_plane_iterations=CONFIG["cg"]["baseline_fgmres_plane_iterations"][str(n)],
                           source="work/p08_step5_solver_studies_20261001/README.md (owner-based P07, rtol 1e-10, plane_jax)")
    res["cg"]["ratio_vs_fgmres_seconds"] = tmed / res["baseline"]["fgmres_plane_seconds"]
    log(f"{arm} N{n}: CG it={its} warm {tmin:.3f}/{tmed:.3f}s (first {t_first:.1f}s); per iteration: apply {t_mv[1] * 1e3:.1f} ms, "
        f"prec {t_pa[1] * 1e3:.1f} ms, vec {t_vo[1] * 1e3:.2f} ms; baseline {res['baseline']['fgmres_plane_seconds']}s / "
        f"{res['baseline']['fgmres_plane_iterations']} it")
    res["seconds"] = time.perf_counter() - t0
    res["peak_rss_gib"] = C.peak_rss_gib()
    C.write_json(C.arm_dir(out_root, arm, n) / f"solve{tag}.json", res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--arm", choices=C.ARMS, required=True)
    ap.add_argument("--out", type=Path, default=C.DEFAULT_ROOT)
    ap.add_argument("--groups", type=int, default=0)
    ap.add_argument("--factor-dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    groups = args.groups or DEFAULT_GROUPS[args.n]
    run(args.n, args.arm, args.out, groups, args.factor_dtype, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
