"""Audits of the nodal curvature operator: scalar energy identity, discrete divergence, W2 interchange, linearised spectrum.

Parts (written to ``<out>/<arm>/N{n}/audit/<part>.json``; a finished part is skipped unless ``--force``):

- ``energy``     the scalar identity ``(g, C_h g)_Hp = 1/2 Omega sum sigma v a^2 - 1/2 (c, g^2)_Hp`` for random ``g`` (two seeds);
- ``divergence`` ``max |c|`` (the discrete ``div F`` of the curvature flux, ``c = -2 T_F(1)``) per region, rms and max of ``|V|``;
- ``w2``         ``sum_Hp [phi C_h(p) + p C_h(phi)] - (wall - (c, phi p))`` with ``p = n (Te + tau Ti)`` for every case;
- ``spectrum``   Jacobian of the full curvature RHS wrt ``(n, Te, Ti, omega)`` (``jax.linearize``, matrix free via ``jax.jvp``/``jax.vjp``, ``phi`` and the wall data
  fixed) about a manufactured state and about the uniform background, for the three variants: ARPACK 'LR' and 'LM' from two start
  vectors each, RK4 ``dt`` from the 'LM' set (``validation.sbp_audit.rk4_dt``). ``--state``, ``--variant`` and ``--budget`` select/limit.

    python curv_audit.py N --arm raw|filtered --parts energy,divergence,w2
    python curv_audit.py 32 --arm filtered --parts spectrum --state manufactured --variant centered --budget 400
"""
from __future__ import annotations

import curv_common as C

import argparse
import json
import math
import time

import numpy as np
import scipy.sparse.linalg as spla

import spectral as M3S                                                                  # noqa: E402  (M3 harness: rk4_dt)

TWO_PI = 2.0 * np.pi
HALO = 3


def _setup(arm, n):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from drbx.native import fci_perpendicular_sbp_curvature as cv
    from drbx.native import fci_perpendicular_sbp_ops as ops
    from drbx.native.fci_perpendicular_sbp_norms import h_weights, ring_region_masks

    layout, metric, meta, plan = C.load_plan(arm, n)
    plan_d = jax.tree_util.tree_map(jnp.asarray, plan)
    F = cv.curvature_flux(plan_d)
    return dict(jax=jax, jnp=jnp, cv=cv, ops=ops, layout=layout, metric=metric, plan=plan, plan_d=plan_d, F=F,
                Fe=ops.extend_periodic(F, HALO), H=np.asarray(h_weights(plan)), masks=ring_region_masks(layout, n))


def part_energy(ctx):
    jax, jnp, cv, ops, plan = ctx["jax"], ctx["jnp"], ctx["cv"], ctx["ops"], ctx["plan_d"]
    Hp = np.asarray(ctx["plan"].Hp)
    F = np.asarray(ctx["F"])
    c = np.asarray(jax.jit(cv.compressibility_flux)(plan, ctx["Fe"]))
    deriv = jax.jit(lambda p, Fe, ge: cv.curvature_derivative_ext(p, Fe, ge))
    res = []
    for seed in (0, 1):
        g = np.random.default_rng(seed).standard_normal(Hp.shape)
        Cg = np.asarray(deriv(plan, ctx["Fe"], ops.extend_periodic(jnp.asarray(g), HALO)))
        wall = 0.0
        for blk, side, sign, N in ctx["plan"].structure.walls:
            a = np.asarray(ops.trace(plan, blk, side, jnp.asarray(g)))
            v = np.asarray(ops.flux_trace(plan, blk, side, F[..., 0], F[..., 1]))
            wall += 0.5 * (TWO_PI / N) * sign * np.sum(v * a * a)
        lhs = np.sum(Hp * g * Cg)
        res.append(float(abs(lhs - (wall - 0.5 * np.sum(Hp * c * g * g))) / (np.sqrt(np.sum(Hp * g * g)) * np.sqrt(np.sum(Hp * Cg * Cg)))))
    return {"residuals": res, "max": max(res), "gate": 1e-13, "pass": bool(max(res) <= 1e-13)}


def part_divergence(ctx):
    jax, cv, plan = ctx["jax"], ctx["cv"], ctx["plan_d"]
    c = np.asarray(jax.jit(cv.compressibility_flux)(plan, ctx["Fe"]))
    F = np.asarray(ctx["F"])
    V = F / np.asarray(ctx["plan"].jac)[..., None]
    H = ctx["H"]
    out = {"regions": {}, "max_abs_c": float(np.abs(c).max()), "max_abs_V": float(np.abs(V).max())}
    for name, m in ctx["masks"].items():
        out["regions"][name] = {"max_abs_c": float(np.abs(c[m]).max()), "rms_c": float(np.sqrt(np.sum(H[m] * c[m] ** 2) / H[m].sum())),
                                "max_abs_V": float(np.abs(V[m]).max())}
    return out


def part_w2(ctx, arm, n):
    jax, jnp, cv, ops, plan = ctx["jax"], ctx["jnp"], ctx["cv"], ctx["ops"], ctx["plan_d"]
    Hp = np.asarray(ctx["plan"].Hp)
    F = np.asarray(ctx["F"])
    c = np.asarray(jax.jit(cv.compressibility_flux)(plan, ctx["Fe"]))
    deriv = jax.jit(lambda p, Fe, ge: cv.curvature_derivative_ext(p, Fe, ge))
    out = {}
    for f in sorted((C.arm_dir(arm, n) / "references").glob("*.npz")):
        z = np.load(f)
        vals = z["vals"]
        phi = vals[4]
        pres = vals[0] * (vals[1] + C.TAU * vals[2])
        g = np.stack([phi, pres], axis=-1)
        Cg = np.asarray(deriv(plan, ctx["Fe"], ops.extend_periodic(jnp.asarray(g), HALO)))
        lhs = np.sum(Hp * (phi * Cg[..., 1] + pres * Cg[..., 0]))
        wall = 0.0
        for blk, side, sign, N in ctx["plan"].structure.walls:
            a = np.asarray(ops.trace(plan, blk, side, jnp.asarray(g)))
            v = np.asarray(ops.flux_trace(plan, blk, side, F[..., 0], F[..., 1]))
            wall += (TWO_PI / N) * sign * np.sum(v * a[..., 0] * a[..., 1])
        ctm = float(np.sum(Hp * c * phi * pres))
        scale = np.sqrt(np.sum(Hp * phi**2) * np.sum(Hp * Cg[..., 1] ** 2)) + np.sqrt(np.sum(Hp * pres**2) * np.sum(Hp * Cg[..., 0] ** 2))
        out[str(z["label"])] = {"lhs": float(lhs), "wall_term": float(wall), "c_term": ctm,
                                "residual_abs": float(abs(lhs - (wall - ctm))), "scale": float(scale),
                                "residual_rel": float(abs(lhs - (wall - ctm)) / scale) if scale > 0 else None}
    return out


# ---------------------------------------------------------------------------------------------- spectrum
def _state(ctx, arm, n, which):
    if which == "uniform":
        E, P = ctx["plan"].jac.shape
        q = np.zeros((E, P, 5)) + np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        w = np.broadcast_to(q[0, 0, :4], (E, ctx["plan"].structure.walls[0][3], 4)).copy()
        return q, w, "uniform"
    for name in ("p06__corrected_frozen_mms", "p06__regular_chart_heldout"):
        f = C.arm_dir(arm, n) / "references" / f"{name}.npz"
        if f.exists():
            z = np.load(f)
            return np.moveaxis(z["vals"], 0, -1), np.moveaxis(z["wall"][:4], 0, -1), str(z["label"])
    raise FileNotFoundError("no P06 reference file")


def _eigs(op, which, seed, k, ncv, tol, maxiter):
    t0 = time.perf_counter()
    v0 = np.random.default_rng(seed).standard_normal(op.shape[0])
    try:
        ev = spla.eigs(op, k=k, which=which, ncv=ncv, tol=tol, maxiter=maxiter, v0=v0, return_eigenvectors=False)
        conv = True
    except spla.ArpackNoConvergence as exc:
        ev, conv = exc.eigenvalues, False
    ev = ev[np.argsort(-np.abs(ev))] if which == "LM" else ev[np.argsort(-ev.real)]
    return {"eig": [[float(e.real), float(e.imag)] for e in ev], "converged": conv, "seconds": time.perf_counter() - t0}


def part_spectrum(ctx, arm, n, which, variant, budget, k=6, ncv=40, tol=1e-4, ncv_lr=60, tol_lr=1e-3):
    jax, jnp, cv = ctx["jax"], ctx["jnp"], ctx["cv"]
    from drbx.native.fci_perpendicular_sbp_boundary import SatBoundaryData
    spec = C.VARIANTS[variant]
    q, w, label = _state(ctx, arm, n, which)
    q, w = jnp.asarray(q), jnp.asarray(w)
    plan = ctx["plan_d"]
    shape = q.shape[:2] + (4,)
    phi = q[..., 4:5]
    ck, jd = float(spec["c_kappa"]), bool(spec["jump_dissipation"])

    def rhs(q4):
        return cv.sbp_curvature(plan, jnp.concatenate([q4, phi], axis=-1), SatBoundaryData((w,), None), tau=C.TAU, psi=C.PSI,
                                absolute_method=C.METHOD, jump_dissipation=jd, c_kappa=ck, F=ctx["F"])

    q4 = q[..., :4]
    t0 = time.perf_counter()
    # no stored linearisation (the dissipative variant's residuals reach ~3 GB): the tangent is recomputed per matvec
    jvp = jax.jit(lambda v: jax.jvp(rhs, (q4,), (v,))[1])
    x0 = jnp.asarray(np.random.default_rng(0).standard_normal(shape))
    np.asarray(jvp(x0))                                              # compile
    tc = time.perf_counter() - t0
    t1 = time.perf_counter()
    for _ in range(5):
        np.asarray(jvp(x0))
    per = (time.perf_counter() - t1) / 5
    count = {"n": 0}

    def mv(x):
        count["n"] += 1
        return np.asarray(jvp(jnp.asarray(np.asarray(x, dtype=np.float64).reshape(shape)))).reshape(-1)

    vjp = jax.jit(lambda y: jax.vjp(rhs, q4)[1](y)[0])
    np.asarray(vjp(x0))

    def rmv(y):
        count["n"] += 1
        return np.asarray(vjp(jnp.asarray(np.asarray(y, dtype=np.float64).reshape(shape)))).reshape(-1)

    op = spla.LinearOperator((int(np.prod(shape)),) * 2, matvec=mv, rmatvec=rmv, dtype=np.float64)
    # matvec budget split over four ARPACK runs (two starts of LM and of LR), ARPACK restarts cost ~ (ncv - k) matvecs each
    per_run = max(budget / 4.0, 4 * per)
    maxiter = max(3, int(per_run / per / (ncv - k)))
    res = {"state": which, "state_label": label, "variant": variant, "n_unknowns": op.shape[0], "seconds_matvec": per,
           "seconds_compile": tc, "ncv": ncv, "tol": tol, "maxiter": maxiter, "budget_seconds": budget}
    allv = []
    for kind in ("LM", "LR"):
        for seed in (1, 2):
            r = _eigs(op, kind, seed, k, ncv if kind == "LM" else ncv_lr, tol if kind == "LM" else tol_lr,
                      maxiter if kind == "LM" else max(3, int(per_run / per / (ncv_lr - k))))
            res[f"{kind}_start{seed}"] = r
            if kind == "LM":
                allv += [complex(a, b) for a, b in r["eig"]]
    lm = [max(abs(complex(a, b)) for a, b in res[f"LM_start{s}"]["eig"]) for s in (1, 2) if res[f"LM_start{s}"]["eig"]]
    lr = [max(a for a, _b in res[f"LR_start{s}"]["eig"]) for s in (1, 2) if res[f"LR_start{s}"]["eig"]]
    res["absmax"] = max(lm) if lm else None
    res["absmax_rel_diff_starts"] = (abs(lm[0] - lm[1]) / max(lm)) if len(lm) == 2 else None
    res["max_re"] = max(lr) if lr else None
    res["max_re_rel_diff_starts"] = (abs(lr[0] - lr[1]) / max(abs(lr[0]), abs(lr[1]), 1e-300)) if len(lr) == 2 else None
    # numerical abscissa (H-norm, unit field weights): lambda_max of sym(H^1/2 J H^-1/2), an upper bound of max Re(lambda)
    d = np.sqrt(np.repeat(ctx["H"].reshape(-1), 4))
    sym = spla.LinearOperator(op.shape, matvec=lambda x: 0.5 * (d * op.matvec(x / d) + op.rmatvec(d * x) / d), dtype=np.float64)
    ta = time.perf_counter()
    try:
        ev = spla.eigsh(sym, k=1, which="LA", ncv=40, tol=1e-4, maxiter=max(5, int(per_run / per / 34)),
                        v0=np.random.default_rng(0).standard_normal(op.shape[0]), return_eigenvectors=False)
        ac = True
    except spla.ArpackNoConvergence as exc:
        ev, ac = exc.eigenvalues, False
    res["abscissa_H"] = {"omega": float(ev.max()) if len(ev) else None, "converged": ac, "seconds": time.perf_counter() - ta}
    res["converged"] = all(res[f"{k_}_start{s}"]["converged"] for k_ in ("LM", "LR") for s in (1, 2))
    res["dt_rk4"] = float(M3S.rk4_dt(allv)) if allv else None
    res["matvecs"] = count["n"]
    res["seconds"] = time.perf_counter() - t0
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--arm", required=True, choices=list(C.ARM_ROOT))
    ap.add_argument("--parts", default="energy,divergence,w2")
    ap.add_argument("--state", default="manufactured,uniform")
    ap.add_argument("--variant", default=",".join(C.VARIANTS))
    ap.add_argument("--budget", type=float, default=400.0, help="seconds of matvecs per spectrum run (all four ARPACK calls)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    d = C.arm_dir(a.arm, a.n) / "audit"
    d.mkdir(parents=True, exist_ok=True)
    ctx = _setup(a.arm, a.n)
    for part in a.parts.split(","):
        if part == "spectrum":
            for which in a.state.split(","):
                for variant in a.variant.split(","):
                    path = d / f"spectrum_{which}_{variant}.json"
                    if path.exists() and not a.force:
                        C.log(f"skip {path.name}")
                        continue
                    res = part_spectrum(ctx, a.arm, a.n, which, variant, a.budget)
                    res["peak_rss_gib"] = C.peak_rss_gib()
                    path.write_text(json.dumps(res, indent=1))
                    fmt = lambda x: "none" if x is None else f"{x:.4g}"  # noqa: E731
                    C.log(f"{a.arm} N{a.n} spectrum {which}/{variant}: |lam|max {fmt(res['absmax'])} maxRe {fmt(res['max_re'])} "
                          f"dt {fmt(res['dt_rk4'])} conv {res['converged']} matvecs {res['matvecs']} {res['seconds']:.0f} s")
            continue
        path = d / f"{part}.json"
        if path.exists() and not a.force:
            C.log(f"skip {path.name}")
            continue
        t0 = time.perf_counter()
        res = {"energy": lambda: part_energy(ctx), "divergence": lambda: part_divergence(ctx),
               "w2": lambda: part_w2(ctx, a.arm, a.n)}[part]()
        path.write_text(json.dumps(res, indent=1))
        C.log(f"{a.arm} N{a.n} {part}: {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
