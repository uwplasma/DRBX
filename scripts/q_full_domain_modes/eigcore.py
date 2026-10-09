"""Shared time-stepped Arnoldi + analysis (numpy/scipy only), N-generic copy of p09_eigenmodes_n32_20261007/eigcore.py.
Owner-grid localization removed; localization is the nodal one supplied by the driver (info["nodal_loc"])."""
import json, resource, time
from pathlib import Path
import numpy as np
import scipy.linalg as sl
import scipy.sparse.linalg as sla

HERE = Path(__file__).resolve().parent
K_WANT, NCV, TOL = 8, 48, 1e-9
MAX_SECONDS = 1500.0          # default Krylov-Schur wall budget per operator (restart boundary); driver overrides per op


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss_peak={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f}GiB {msg}",
          flush=True)


def krylov_schur(P, n, v0, k=K_WANT, m=NCV, tol=TOL, max_seconds=MAX_SECONDS):
    """Restarted (Krylov-Schur) Arnoldi for the k largest-|mu| eigenpairs of P; returns all k Ritz pairs at exit."""
    t0 = time.perf_counter()
    V = np.zeros((n, m + 1)); H = np.zeros((m + 1, m))
    V[:, 0] = v0 / np.linalg.norm(v0); j0 = 0; apps = 0; restarts = 0
    while True:
        for j in range(j0, m):
            w = P(V[:, j]); apps += 1
            for _ in range(2):
                h = V[:, :j + 1].T @ w; w = w - V[:, :j + 1] @ h; H[:j + 1, j] += h
            H[j + 1, j] = np.linalg.norm(w); V[:, j + 1] = w / H[j + 1, j]
        Hm, beta = H[:m, :m], H[m, m - 1]
        ev, Y = np.linalg.eig(Hm)
        order = np.argsort(-np.abs(ev))
        est = np.abs(beta * Y[m - 1, :]) / np.abs(ev)
        top = order[:k]
        conv = bool(np.all(est[top] <= tol))
        log(f"  KS restart {restarts} apps {apps} max_est_top {est[top].max():.2e} |mu|top {np.abs(ev[top[0]]):.6f}")
        if conv or time.perf_counter() - t0 > max_seconds:
            break
        thr = np.abs(ev[order[m // 2 - 1]])
        T, Z, kk = sl.schur(Hm, output="real", sort=lambda re, im: np.hypot(re, im) >= thr * (1 - 1e-12))
        kk = int(min(kk, m - 2))
        V[:, :kk] = V[:, :m] @ Z[:, :kk]; V[:, kk] = V[:, m]
        H = np.zeros((m + 1, m)); H[:kk, :kk] = T[:kk, :kk]; H[kk, :kk] = beta * Z[m - 1, :kk]
        j0 = kk; restarts += 1
    X = V[:, :m] @ Y[:, top]
    return ev[top], X, est[top], dict(propagator_apps=apps, restarts=restarts, converged_tol=conv,
                                      seconds=round(time.perf_counter() - t0, 1), max_seconds=max_seconds,
                                      ritz_est_top=[float(e) for e in est[top]])


def phase_fix(v):
    v = v / np.linalg.norm(v)
    return v * np.exp(-1j * np.angle(v.ravel()[np.argmax(np.abs(v.ravel()))]))


def run(op_id, L, n, check, info=None, seed=7, extra_npz=None, nodal_shape=None, tag="", max_seconds=MAX_SECONDS,
        ks_v0=None):
    """L: real (n,) -> (n,) numpy; check(results) -> dict. ks_v0: optional Krylov-Schur start vector (default rng(seed+1))."""
    out_npz, out_json = HERE / f"modes_{op_id}{tag}.npz", HERE / f"modes_{op_id}{tag}.json"
    if out_json.exists():
        log(f"{op_id}: done already, skipping"); return
    t0 = time.perf_counter(); cnt = [0]
    def Lc(v):
        cnt[0] += 1; return np.asarray(L(np.ascontiguousarray(v)), dtype=np.float64)
    tl = time.perf_counter(); Lc(np.random.default_rng(0).normal(size=n)); t_first = time.perf_counter() - tl
    tl = time.perf_counter(); Lc(np.random.default_rng(1).normal(size=n)); t_prod = time.perf_counter() - tl
    log(f"{op_id}: n {n}, first product {t_first:.1f}s, next {t_prod:.3f}s")
    # 1. spectral radius
    A = sla.LinearOperator((n, n), matvec=Lc, dtype=np.float64)
    c0 = cnt[0]; tl = time.perf_counter()
    lm = sla.eigs(A, k=2, which="LM", tol=1e-3, v0=np.random.default_rng(seed).normal(size=n), return_eigenvectors=False)
    rho_max = float(np.max(np.abs(lm))); n_lm = cnt[0] - c0; t_lm = time.perf_counter() - tl
    dt = 0.3 / rho_max
    log(f"{op_id}: LM {[complex(x) for x in lm]} rho_max {rho_max:.6e} dt {dt:.4e} ({n_lm} products, {t_lm:.1f}s)")
    # 2-3. propagator Arnoldi
    def P(v):
        k1 = Lc(v); k2 = Lc(v + .5 * dt * k1); k3 = Lc(v + .5 * dt * k2); k4 = Lc(v + dt * k3)
        return v + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    c0 = cnt[0]
    v0 = np.random.default_rng(seed + 1).normal(size=n) if ks_v0 is None else np.asarray(ks_v0, dtype=np.float64)
    mu, X, est, ks = krylov_schur(P, n, v0, max_seconds=max_seconds)
    n_ks = cnt[0] - c0
    log(f"{op_id}: Krylov-Schur exit {json.dumps(ks)}")
    # 4. refine on L, conjugate pairs
    pairs = []
    for q in range(len(mu)):
        m_, x = complex(mu[q]), X[:, q]
        if m_.imag < -1e-14 * abs(m_):
            if any(abs(m_.conjugate() - complex(mu[p])) <= 1e-10 * abs(m_) for p in range(len(mu)) if p != q):
                continue
            m_, x = m_.conjugate(), np.conj(x)
        if abs(m_.imag) <= 1e-14 * abs(m_):
            x = np.real(x * np.exp(-1j * np.angle(x[np.argmax(np.abs(x))]))).astype(complex)
        x = x / np.linalg.norm(x)
        Lx = Lc(x.real) + (1j * Lc(x.imag) if np.abs(x.imag).max() > 0 else 0)
        lam = complex(np.vdot(x, Lx) / np.vdot(x, x))
        r = float(np.linalg.norm(Lx - lam * x) / (abs(lam) * np.linalg.norm(x)))
        pairs.append(dict(mu=m_, lam_prop=complex(np.log(m_) / dt), lam=lam, resid=r, ritz_est_prop=float(est[q]), x=x))
    pairs.sort(key=lambda p: -p["lam"].real)
    rec, nre, nim = [], [], []
    for p in pairs:
        vn = phase_fix(p["x"])
        if nodal_shape is not None:
            nre.append(vn.real.reshape(nodal_shape)); nim.append(vn.imag.reshape(nodal_shape))
        d = dict(lam=[p["lam"].real, p["lam"].imag], lam_prop=[p["lam_prop"].real, p["lam_prop"].imag],
                 lam_prop_minus_lam=abs(p["lam_prop"] - p["lam"]), mu=[p["mu"].real, p["mu"].imag], resid=p["resid"],
                 accepted=bool(p["resid"] <= 1e-3), ritz_est_propagator=p["ritz_est_prop"])
        if info and "nodal_loc" in info:
            d["nodal_loc"] = info["nodal_loc"](vn)
        rec.append(d)
        nl = d.get("nodal_loc", {})
        log(f"{op_id}: lam {p['lam']:.6e} prop {p['lam_prop']:.6e} resid {p['resid']:.2e} peak ring {nl.get('peak_ring')} "
            f"u {nl.get('peak_ring_u')} plane {nl.get('peak_plane')} fields {nl.get('field_share')}")
    res = dict(id=op_id, rho_max=rho_max, LM=[[complex(x).real, complex(x).imag] for x in lm], dt=dt, pairs=rec,
               n_accepted=sum(r["accepted"] for r in rec), products=dict(lm=n_lm, arnoldi=n_ks, total=cnt[0], probe=2),
               krylov_schur=ks, product_seconds=t_prod, first_product_seconds=t_first, lm_seconds=round(t_lm, 1),
               wall_seconds=round(time.perf_counter() - t0, 1),
               peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
    res["check"] = check(res)
    if info:
        res.update({k: v for k, v in info.items() if k != "nodal_loc"})
    arrs = dict(lam=np.array([p["lam"] for p in pairs]), lam_prop=np.array([p["lam_prop"] for p in pairs]),
                resid=np.array([p["resid"] for p in pairs]), accepted=np.array([r["accepted"] for r in rec]),
                dt=dt, rho_max=rho_max)
    if nodal_shape is not None:
        arrs.update(vec_nodal_re=np.array(nre), vec_nodal_im=np.array(nim))
    arrs.update(extra_npz or {})
    np.savez(out_npz, **arrs)
    out_json.write_text(json.dumps(res, indent=1, default=str))
    log(f"{op_id}: done in {res['wall_seconds']}s, {cnt[0]} products, check {json.dumps(res['check'])}")
    return res


def rightmost(res, accepted_only=True):
    ps = [p for p in res["pairs"] if p["accepted"] or not accepted_only]
    return max(ps, key=lambda p: p["lam"][0]) if ps else None
