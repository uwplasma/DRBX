"""Analytic-eigensolve investigation (roadmap P09 Step 5): the closed-form P06 ``|M|`` action against ``lapack4``.

Parts (JSON in ``<out>/eigensolve/``; ``--parts accuracy,symmetry,cost``):

- ``accuracy``  ``|M(n, Te, Ti, B, tau)| j`` by the closed form and by ``lapack4`` (the campaign's batched 4x4 ``eig``) over (a) the node
  states of every catalogue case, both arms and every resolution, with two random jump vectors per node and both selectors;
  (b) a synthetic sweep ``n, Te, Ti in [0.02, 10]`` (and ``Ti = 0``), ``tau in {0, 1e-3, 0.1, 1, 5}``, ``B in {0.5, 1, 2}``, both
  selectors, plus a fine sweep of ``r = tau Ti / Te`` from 1e-14 to 1e6. The error is ``|cf - lp| / |lp|`` (and normalised by
  ``|M|_F |j|``); the maximum and its location are reported, lapack4 fallbacks (real-spectrum / cond <= 1e8 test) are counted and
  excluded. The states of smallest relative eigenvalue separation of the sweep and a random subset are also compared with a
  60-digit ``mpmath`` eigendecomposition (the true error of each method).
- ``symmetry``  ``W |M|`` with ``W = R^-T R^-1`` (``R`` the eigenvectors of ``M``) must be symmetric positive semidefinite: the relative
  asymmetry and the smallest eigenvalue of its symmetric part for the closed-form ``|M|`` (and for lapack4's, for comparison).
- ``cost``      wall clock per 1e6 evaluations (``M`` build + action), jitted, ``NPROC=2``, closed form against batched lapack4.

    python curv_eigensolve.py --parts accuracy,symmetry,cost
"""
from __future__ import annotations

import curv_common as C

import argparse
import functools
import json
import time

import numpy as np

FLOOR = 1e-12
PSIS = ("phi_plus_tau_ti", "phi_plus_tau_pi")


def kernels():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from drbx.native.fci_curvature_production_flux import curvature_principal_matrix
    from drbx.native.fci_perpendicular_face_corrections import _absolute_action_closed_form, _p06_absolute_action

    @functools.partial(jax.jit, static_argnames=("psi",))
    def both(n, te, ti, b, tau, jump, psi):
        M = curvature_principal_matrix(n, te, ti, b, tau, psi=psi)
        cf, bad_cf = _absolute_action_closed_form(n, te, ti, b, tau, jnp.ones_like(n), M, jump, FLOOR, psi=psi)
        lp, bad_lp = _p06_absolute_action(M, jump)
        return cf, lp, bad_cf, bad_lp, jnp.sqrt(jnp.sum(M * M, axis=(-2, -1)))

    @functools.partial(jax.jit, static_argnames=("psi",))
    def only_cf(n, te, ti, b, tau, jump, psi):
        M = curvature_principal_matrix(n, te, ti, b, tau, psi=psi)
        return _absolute_action_closed_form(n, te, ti, b, tau, jnp.ones_like(n), M, jump, FLOOR, psi=psi)[0]

    @functools.partial(jax.jit, static_argnames=("psi",))
    def only_lp(n, te, ti, b, tau, jump, psi):
        M = curvature_principal_matrix(n, te, ti, b, tau, psi=psi)
        return _p06_absolute_action(M, jump)[0]

    return both, only_cf, only_lp


class Tracker:
    """Running maxima of the action error with the location of the worst state."""

    def __init__(self, name):
        self.name = name
        self.count = self.valid = self.fallback_lp = self.invalid_cf = 0
        self.max_rel = self.max_norm = 0.0
        self.worst = None
        self.worst_norm = None

    def update(self, n, te, ti, b, tau, psi, cf, lp, bad_cf, bad_lp, normM, jump, tag):
        ok = ~(np.asarray(bad_lp) | np.asarray(bad_cf))
        self.count += len(n)
        self.fallback_lp += int(np.sum(np.asarray(bad_lp)))
        self.invalid_cf += int(np.sum(np.asarray(bad_cf)))
        self.valid += int(ok.sum())
        if not ok.any():
            return
        d = np.linalg.norm(cf - lp, axis=-1)
        nl = np.linalg.norm(lp, axis=-1)
        rel = np.where(ok, d / np.maximum(nl, 1e-300), 0.0)
        nrm = np.where(ok, d / np.maximum(normM * np.linalg.norm(jump, axis=-1), 1e-300), 0.0)
        for arr, key in ((rel, "max_rel"), (nrm, "max_norm")):
            i = int(np.argmax(arr))
            if arr[i] > getattr(self, key):
                setattr(self, key, float(arr[i]))
                info = dict(value=float(arr[i]), n=float(n[i]), Te=float(te[i]), Ti=float(ti[i]), B=float(b[i]), tau=float(tau[i]),
                            psi=psi, tag=tag, r=float(tau[i] * ti[i] / te[i]), abs_err=float(d[i]), norm_lp=float(nl[i]))
                if key == "max_rel":
                    self.worst = info
                else:
                    self.worst_norm = info

    def result(self):
        return dict(states=self.count, compared=self.valid, lapack4_fallback=self.fallback_lp, closed_form_nonphysical=self.invalid_cf,
                    max_rel_action_error=self.max_rel, max_norm_action_error=self.max_norm, worst=self.worst, worst_normalised=self.worst_norm)


def feed(tracker, both, n, te, ti, b, tau, psi, tag, seed, chunk=200_000, jumps=2):
    rng = np.random.default_rng(seed)
    n, te, ti, b, tau = (np.asarray(x, dtype=np.float64).reshape(-1) for x in np.broadcast_arrays(n, te, ti, b, tau))
    for s in range(0, len(n), chunk):
        sl = slice(s, s + chunk)
        for _ in range(jumps):
            j = rng.standard_normal((len(n[sl]), 4))
            cf, lp, bcf, blp, nm = both(n[sl], te[sl], ti[sl], b[sl], tau[sl], j, psi)
            tracker.update(n[sl], te[sl], ti[sl], b[sl], tau[sl], psi, np.asarray(cf), np.asarray(lp), bcf, blp, np.asarray(nm), j, tag)


def principal_matrix_np(n, te, ti, b, tau, psi):
    M = np.zeros((4, 4))
    M[0] = [2 * te, 2 * n, 2 * n * tau, 0]
    M[1] = [4 * te * te / (3 * n), 14 * te / 3, 4 * tau * te / 3, 0]
    M[2] = [4 * ti * te / (3 * n), 4 * ti / 3, -2 * tau * ti, 0]
    M[3] = [2 * b * b * (te + tau * ti) / n, 2 * b * b, 2 * tau * b * b, 0]
    if psi == "phi_plus_tau_pi":
        c = np.array([2 * n, 4 * te / 3, 4 * ti / 3])
        M[:3, 0] += c * tau * ti
        M[:3, 2] += c * tau * (n - 1)
    return M


def mp_abs_action(n, te, ti, b, tau, psi, jump, dps=60):
    import mpmath as mp
    mp.mp.dps = dps
    n, te, ti, b, tau = (mp.mpf(float(x)) for x in (n, te, ti, b, tau))
    M = mp.matrix(4, 4)
    rows = [[2 * te, 2 * n, 2 * n * tau, 0], [4 * te * te / (3 * n), 14 * te / 3, 4 * tau * te / 3, 0],
            [4 * ti * te / (3 * n), 4 * ti / 3, -2 * tau * ti, 0], [2 * b * b * (te + tau * ti) / n, 2 * b * b, 2 * tau * b * b, 0]]
    for i in range(4):
        for k in range(4):
            M[i, k] = rows[i][k]
    if psi == "phi_plus_tau_pi":
        cvec = [2 * n, 4 * te / 3, 4 * ti / 3]
        for i in range(3):
            M[i, 0] += cvec[i] * tau * ti
            M[i, 2] += cvec[i] * tau * (n - 1)
    E, ER = mp.eig(M)
    absM = ER * mp.diag([abs(mp.re(e)) for e in E]) * mp.inverse(ER)
    j = mp.matrix([mp.mpf(float(x)) for x in jump])
    out = absM * j
    return np.array([float(mp.re(out[i])) for i in range(4)])


def sweep_states():
    v = np.geomspace(0.02, 10.0, 13)
    tis = np.concatenate([[0.0], v])
    taus = np.array([0.0, 1e-3, 0.1, 1.0, 5.0])
    bs = np.array([0.5, 1.0, 2.0])
    n, te, ti, tau, b = (a.reshape(-1) for a in np.meshgrid(v, v, tis, taus, bs, indexing="ij"))
    main = (n, te, ti, b, tau)
    r = np.geomspace(1e-14, 1e6, 41)
    nn, tt, rr, bb = (a.reshape(-1) for a in np.meshgrid([0.5, 1.0, 2.0], [0.5, 1.0, 2.0], r, [1.0], indexing="ij"))
    fine = (nn, tt, rr * tt, bb, np.full_like(nn, 1.0))               # tau = 1: Ti = r Te
    fine5 = (nn, tt, rr * tt / 5.0, bb, np.full_like(nn, 5.0))         # tau = 5, same r
    return main, fine, fine5


def separation(n, te, ti, b, tau, psi):
    lam = np.linalg.eigvals(principal_matrix_np(n, te, ti, b, tau, psi)).real
    d = np.abs(lam[:, None] - lam[None, :])[np.triu_indices(4, 1)]
    return float(d.min() / max(np.abs(lam).max(), 1e-300))


def part_accuracy(args):
    both, _cf, _lp = kernels()
    out = {"catalogue": {}, "sweep": {}}
    # (a) catalogue nodes (duplicate field sets across cases are skipped by hash)
    seen = {}
    for arm in C.ARM_ROOT:
        for n in C.GRIDS:
            base = C.arm_dir(arm, n) / "references"
            if not base.exists():
                continue
            _l, metric, _m = C.load_layout_metric(n, C.ARM_ROOT[arm])
            Bn = np.asarray(metric.B).reshape(-1)
            for f in sorted(base.glob("*.npz")):
                z = np.load(f)
                vals = np.asarray(z["vals"])[:3].reshape(3, -1)
                key = (arm, n, hash(vals.tobytes()))
                if key in seen:
                    continue
                seen[key] = str(z["label"])
                for psi in PSIS:
                    tr = out["catalogue"].setdefault(f"{arm}/N{n}/{psi}", Tracker(f"{arm}/N{n}/{psi}"))
                    feed(tr, both, vals[0], vals[1], vals[2], Bn, np.full(Bn.shape, C.TAU), psi, f"{arm}/N{n}/{z['label']}", 1)
    out["catalogue"] = {k: t.result() for k, t in out["catalogue"].items()}
    allc = [v for v in out["catalogue"].values()]
    out["catalogue_overall"] = dict(max_rel_action_error=max(v["max_rel_action_error"] for v in allc),
                                    max_norm_action_error=max(v["max_norm_action_error"] for v in allc),
                                    lapack4_fallback=sum(v["lapack4_fallback"] for v in allc),
                                    closed_form_nonphysical=sum(v["closed_form_nonphysical"] for v in allc),
                                    states=sum(v["states"] for v in allc),
                                    worst=max((v["worst"] for v in allc if v["worst"]), key=lambda w: w["value"]))
    C.log(f"catalogue: {out['catalogue_overall']['states']} nodes, max rel {out['catalogue_overall']['max_rel_action_error']:.2e}")
    # (b) synthetic sweep
    main, fine, fine5 = sweep_states()
    sweeps = {"main": main, "fine_tau1": fine, "fine_tau5": fine5}
    cand = []
    for name, st in sweeps.items():
        for psi in PSIS:
            tr = Tracker(f"{name}/{psi}")
            feed(tr, both, *st, psi, name, 2, jumps=2)
            out["sweep"][f"{name}/{psi}"] = tr.result()
            C.log(f"sweep {name}/{psi}: {tr.count} states ({tr.valid} compared, lapack4 fallback {tr.fallback_lp}) max rel {tr.max_rel:.2e}")
    # (c) smallest relative separation and mpmath truth
    n_, te_, ti_, b_, tau_ = main
    sel = np.flatnonzero((ti_ > 0) & (tau_ > 0))
    rng = np.random.default_rng(3)
    sub = rng.choice(sel, size=min(len(sel), 6000), replace=False)
    fine_idx = np.arange(len(fine[0]))
    pool = [("main", i, psi) for i in sub for psi in PSIS] + [("fine_tau1", i, psi) for i in fine_idx for psi in PSIS]
    seps = []
    for name, i, psi in pool:
        st = sweeps[name]
        seps.append(separation(st[0][i], st[1][i], st[2][i], st[3][i], st[4][i], psi))
    order = np.argsort(seps)
    worst = [pool[k] for k in order[:40]]
    rand = [pool[k] for k in rng.choice(len(pool), size=40, replace=False)]
    mp_rows = []
    cf_only, lp_only = kernels()[1:]
    for group, items in (("smallest_separation", worst), ("random", rand)):
        for name, i, psi in items:
            st = sweeps[name]
            args5 = [np.array([float(a[i])]) for a in st]
            j = np.random.default_rng(int(i)).standard_normal((1, 4))
            truth = mp_abs_action(*(float(a[i]) for a in st), psi, j[0])
            cfv = np.asarray(cf_only(*args5, j, psi))[0]
            lpv = np.asarray(lp_only(*args5, j, psi))[0]
            sep = separation(*(float(a[i]) for a in st), psi)
            mp_rows.append(dict(group=group, set=name, psi=psi, n=float(st[0][i]), Te=float(st[1][i]), Ti=float(st[2][i]), B=float(st[3][i]),
                                tau=float(st[4][i]), separation=sep,
                                err_closed_form=float(np.linalg.norm(cfv - truth) / np.linalg.norm(truth)),
                                err_lapack4=float(np.linalg.norm(lpv - truth) / np.linalg.norm(truth))))
    out["mpmath"] = {"rows": mp_rows,
                     "smallest_separation_max_err_closed_form": max(r["err_closed_form"] for r in mp_rows if r["group"] == "smallest_separation"),
                     "smallest_separation_max_err_lapack4": max(r["err_lapack4"] for r in mp_rows if r["group"] == "smallest_separation"),
                     "random_max_err_closed_form": max(r["err_closed_form"] for r in mp_rows if r["group"] == "random"),
                     "random_max_err_lapack4": max(r["err_lapack4"] for r in mp_rows if r["group"] == "random"),
                     "smallest_relative_separation_found": float(seps[order[0]]), "pool": len(pool)}
    C.log(f"mpmath: smallest separation {out['mpmath']['smallest_relative_separation_found']:.2e}; closed-form worst "
          f"{out['mpmath']['smallest_separation_max_err_closed_form']:.2e}, lapack4 worst {out['mpmath']['smallest_separation_max_err_lapack4']:.2e}")
    return out


def part_symmetry(args):
    both, cf_only, _lp = kernels()
    import jax.numpy as jnp
    eye = np.eye(4)
    out = {}

    def check(n, te, ti, b, tau, psi, cap=20000):
        n, te, ti, b, tau = (np.asarray(x, dtype=np.float64).reshape(-1) for x in np.broadcast_arrays(n, te, ti, b, tau))
        keep = (ti > 0) & (tau > 0)
        idx = np.flatnonzero(keep)[:cap]
        n, te, ti, b, tau = n[idx], te[idx], ti[idx], b[idx], tau[idx]
        absM = np.zeros((len(n), 4, 4))
        absL = np.zeros_like(absM)
        for k in range(4):
            j = np.broadcast_to(eye[k], (len(n), 4)).copy()
            cf, lp, *_ = both(n, te, ti, b, tau, j, psi)
            absM[:, :, k], absL[:, :, k] = np.asarray(cf), np.asarray(lp)
        res = dict(states=int(len(n)), excluded_ill_conditioned=0, max_asym_cf=0.0, max_asym_lp=0.0, min_eig_ratio_cf=0.0, min_eig_ratio_lp=0.0)
        worst = None
        for i in range(len(n)):
            M = principal_matrix_np(n[i], te[i], ti[i], b[i], tau[i], psi)
            lam, R = np.linalg.eig(M)
            if np.abs(lam.imag).max() > 1e-10 * np.abs(lam).max() or np.linalg.cond(R) > 1e7:
                res["excluded_ill_conditioned"] += 1
                continue
            R = R.real
            Ri = np.linalg.inv(R)
            W = Ri.T @ Ri
            for key, A in (("cf", absM[i]), ("lp", absL[i])):
                S = W @ A
                asym = np.linalg.norm(S - S.T) / np.linalg.norm(S)
                sym = 0.5 * (S + S.T)
                ev = np.linalg.eigvalsh(sym)
                ratio = min(ev.min(), 0.0) / max(abs(ev).max(), 1e-300)
                res[f"max_asym_{key}"] = max(res[f"max_asym_{key}"], float(asym))
                if ratio < res[f"min_eig_ratio_{key}"]:
                    res[f"min_eig_ratio_{key}"] = float(ratio)
                if key == "cf" and (worst is None or asym > worst["asym"]):
                    worst = dict(asym=float(asym), n=float(n[i]), Te=float(te[i]), Ti=float(ti[i]), B=float(b[i]), tau=float(tau[i]),
                                 condR=float(np.linalg.cond(R)))
        res["worst_closed_form"] = worst
        return res

    main, fine, fine5 = sweep_states()
    for psi in PSIS:
        n_, te_, ti_, b_, tau_ = main
        rng = np.random.default_rng(11)
        pick = rng.choice(len(n_), size=min(len(n_), 30000), replace=False)
        out[f"sweep_main/{psi}"] = check(n_[pick], te_[pick], ti_[pick], b_[pick], tau_[pick], psi)
        out[f"sweep_fine_tau1/{psi}"] = check(*fine, psi)
        C.log(f"symmetry {psi}: main asym cf {out[f'sweep_main/{psi}']['max_asym_cf']:.2e} lp {out[f'sweep_main/{psi}']['max_asym_lp']:.2e}; "
              f"min eig ratio cf {out[f'sweep_main/{psi}']['min_eig_ratio_cf']:.2e}")
    # catalogue nodes of the largest resolution available, both arms
    for arm in C.ARM_ROOT:
        for n in C.GRIDS[:2]:
            base = C.arm_dir(arm, n) / "references"
            if not base.exists():
                continue
            _l, metric, _m = C.load_layout_metric(n, C.ARM_ROOT[arm])
            z = np.load(base / "p06n__main_phi_neumann.npz")
            vals = z["vals"][:3].reshape(3, -1)
            Bn = np.asarray(metric.B).reshape(-1)
            sel = np.random.default_rng(5).choice(len(Bn), size=min(len(Bn), 8000), replace=False)
            out[f"catalogue/{arm}/N{n}/phi_plus_tau_pi"] = check(vals[0][sel], vals[1][sel], vals[2][sel], Bn[sel], np.full(len(sel), C.TAU),
                                                                 "phi_plus_tau_pi")
    return out


def part_cost(args):
    import jax
    both, cf_only, lp_only = kernels()
    rng = np.random.default_rng(0)
    chunk, reps = 250_000, 4
    st = [rng.uniform(0.5, 2.0, chunk) for _ in range(3)] + [np.ones(chunk), np.ones(chunk)]
    jump = rng.standard_normal((chunk, 4))
    out = {}
    for psi in PSIS:
        for name, fn in (("closed_form", cf_only), ("lapack4", lp_only)):
            jax.block_until_ready(fn(*st, jump, psi))                  # compile + warm up
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                for _r in range(reps):
                    jax.block_until_ready(fn(*st, jump, psi))
                times.append(time.perf_counter() - t0)
            out[f"{name}/{psi}"] = dict(seconds_per_1e6=min(times) * 1e6 / (reps * chunk), repeats=times, chunk=chunk,
                                        threads=dict(NPROC=C.os.environ.get("NPROC"), OMP=C.os.environ.get("OMP_NUM_THREADS")))
        out[f"speedup/{psi}"] = out[f"lapack4/{psi}"]["seconds_per_1e6"] / out[f"closed_form/{psi}"]["seconds_per_1e6"]
        C.log(f"cost {psi}: closed form {out[f'closed_form/{psi}']['seconds_per_1e6']:.2f} s / 1e6, lapack4 {out[f'lapack4/{psi}']['seconds_per_1e6']:.2f} s / 1e6")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="accuracy,symmetry,cost")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    d = C.OUT / "eigensolve"
    d.mkdir(parents=True, exist_ok=True)
    for part in a.parts.split(","):
        path = d / f"{part}.json"
        if path.exists() and not a.force:
            C.log(f"skip {path.name}")
            continue
        t0 = time.perf_counter()
        res = {"accuracy": part_accuracy, "symmetry": part_symmetry, "cost": part_cost}[part](a)
        res["seconds"], res["peak_rss_gib"] = time.perf_counter() - t0, C.peak_rss_gib()
        path.write_text(json.dumps(res, indent=1))
        C.log(f"{part}: {res['seconds']:.0f} s, peak {res['peak_rss_gib']:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
