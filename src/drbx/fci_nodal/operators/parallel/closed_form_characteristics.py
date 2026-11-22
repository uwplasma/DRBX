"""Eigensolver-free |A| for DRBX parallel_production_principal_matrix(tau, mu, psi).

Matrix (2D_fci fci_parallel_production_flux.py:380-441), q = (n, Te, Ti, Vi, Ve):
  row0: Ve, 0, 0, 0, n
  row1: -1.42 Te dV/(3n), Ve, 0, -1.42 Te/3, 3.42 Te/3
  row2: -2 Ti dV/(3n), 0, Vi, 0, 2 Ti/3
  row3: (Te + tau Ti)/n, 1, tau, Vi, 0
  row4: mu Te/n + mu tau Ti, 1.71 mu, mu tau n, 0, Ve

det(A - lam) = -(lam - Vi) p(lam - Ve),  d = Vi - Ve.
The quartic uses the coupled ion-pressure convention.

Roots: the legacy seed (biquadratic + Newton + Vieta deflation) is used first;
faces whose roots are not strictly ordered f1 < s1 < s2 < f2 or not converged
are recomputed (robust=True) with a bracketed solve between the analytic
critical points of p (cubic, Viete), 80 bisections + 2 Newton.  `valid`
requires four distinct real roots, p(d) != 0 (contact not coalesced; l0.r0 is
proportional to p(d)), n > 0, Te > 0, Ti >= 0, tau >= 0, mu > 0.
"""
import numpy as np

PI = 'phi_plus_tau_pi'


def coefficients(n, t, i, d, tau, mu, psi, xp=np):
    ct = (44241.0 * mu - 7100.0) / 15000.0
    bt = 28191.0 * mu / 7500.0
    At = 15123.0 * mu / 12.0
    Ad = 36423.0 * mu / 12.0
    if psi == PI:
        cs = 5.0 * mu * tau / 3.0
        c = -(cs * (n * i) + ct * t)
        b = d * (cs * (n * i) + bt * t)
        Ai = (60705.0 * mu * tau / 12.0 - 35500.0 * mu * tau / 12.0 * n)
        a = t * (At * t + Ai * i - Ad * d * d) / 3750.0
    else:
        raise ValueError(psi)
    return c, b, a


def _p(x, d, c, b, a):
    return (((x - d) * x + c) * x + b) * x + a


def _dp(x, d, c, b):
    return ((4.0 * x - 3.0 * d) * x + 2.0 * c) * x + b


def _legacy_roots(d, c, b, a, xp, newton=3):
    """Verbatim legacy algorithm (closed_form.quartic_roots) on given coefficients."""
    disc0 = xp.sqrt(xp.maximum(c * c - 4.0 * a, 0.0))
    ff = xp.sqrt((-c + disc0) / 2.0)
    f1, f2 = -ff, ff
    for _ in range(newton + 1):
        f1 = f1 - _p(f1, d, c, b, a) / _dp(f1, d, c, b)
        f2 = f2 - _p(f2, d, c, b, a) / _dp(f2, d, c, b)
    g = f1 + f2 - d
    h = a / (f1 * f2)
    disc = g * g - 4.0 * h
    sq = xp.sqrt(xp.maximum(disc, 0.0))
    qq = -0.5 * (g + xp.where(g >= 0, sq, -sq))
    r1 = qq
    r2 = h / qq
    s1 = xp.minimum(r1, r2)
    s2 = xp.maximum(r1, r2)
    for _ in range(newton):
        s1 = s1 - _p(s1, d, c, b, a) / _dp(s1, d, c, b)
        s2 = s2 - _p(s2, d, c, b, a) / _dp(s2, d, c, b)
    f1 = f1 - _p(f1, d, c, b, a) / _dp(f1, d, c, b)
    f2 = f2 - _p(f2, d, c, b, a) / _dp(f2, d, c, b)
    return (f1, s1, s2, f2), disc


def _bracketed_roots(d, c, b, a, xp, steps=80):
    """Roots between the critical points of p (3 real critical points required)."""
    aa, bb, cc = -3.0 * d / 4.0, c / 2.0, b / 4.0
    pp = bb - aa * aa / 3.0
    qv = 2.0 * aa ** 3 / 27.0 - aa * bb / 3.0 + cc
    rad = xp.sqrt(xp.maximum(-pp / 3.0, 1e-300))
    arg = xp.clip(-qv / (2.0 * rad ** 3), -1.0, 1.0)
    ang = xp.arccos(arg) / 3.0
    m = [2.0 * rad * xp.cos(ang - 2.0 * np.pi * k / 3.0) - aa / 3.0 for k in range(3)]
    m1 = xp.minimum(xp.minimum(m[0], m[1]), m[2]); m3 = xp.maximum(xp.maximum(m[0], m[1]), m[2]); m2 = m[0] + m[1] + m[2] - m1 - m3
    bound = 1.0 + xp.maximum(xp.maximum(xp.abs(d), xp.abs(c)), xp.maximum(xp.abs(b), xp.abs(a)))
    los = [-bound, m1, m2, m3]; his = [m1, m2, m3, bound]
    leftpos = [True, False, True, False]   # sign of p at the left bracket end
    roots = []
    for lo, hi, lp in zip(los, his, leftpos):
        for _ in range(steps):
            mid = lo + (hi - lo) / 2.0
            val = _p(mid, d, c, b, a)
            same = (val > 0) if lp else (val < 0)
            lo, hi = xp.where(same, mid, lo), xp.where(same, hi, mid)
        x = lo + (hi - lo) / 2.0
        for _ in range(2):
            dpx = _dp(x, d, c, b)
            x = xp.where(dpx != 0, x - _p(x, d, c, b, a) / xp.where(dpx != 0, dpx, 1.0), x)
        roots.append(x)
    three_real = (pp < 0) & (xp.abs(-qv / (2.0 * rad ** 3)) < 1.0)
    # four distinct real roots <=> p(m1) < 0 < p(m2) and p(m3) < 0
    three_real = three_real & (_p(m1, d, c, b, a) < 0) & (_p(m2, d, c, b, a) > 0) & (_p(m3, d, c, b, a) < 0)
    return tuple(roots), three_real


def _ordered(r, xp):
    f1, s1, s2, f2 = r
    ok = (f1 < s1) & (s1 < s2) & (s2 < f2)
    for x in r:
        ok = ok & xp.isfinite(x)
    return ok


def quartic_roots(n, t, i, d, tau, mu, psi, xp=np, robust=True):
    """Resolve quartic roots from legacy seeds with bracketed recovery."""
    c, b, a = coefficients(n, t, i, d, tau, mu, psi, xp)
    leg, disc = _legacy_roots(d, c, b, a, xp)
    scale = xp.maximum(xp.abs(leg[0]), xp.abs(leg[3]))
    conv = xp.ones_like(d, dtype=bool)
    for x in leg:   # converged: one more Newton step moves by < 64 ulp of the speed scale
        step = _p(x, d, c, b, a) / _dp(x, d, c, b)
        conv = conv & (xp.abs(step) <= 64 * 2.2e-16 * xp.maximum(scale, 1.0))
    good = (disc > 0) & _ordered(leg, xp) & conv
    pd = _p(d, d, c, b, a)
    if not robust:
        raise ValueError("Only the robust quartic solver is supported")
    if xp is np:
        roots = [np.array(x, copy=True) for x in leg]
        bad = ~good
        if np.any(bad):
            br, tr = _bracketed_roots(d[bad], c[bad], b[bad], a[bad], np)
            for k in range(4):
                roots[k][bad] = br[k]
            three = np.ones_like(good); three[bad] = tr
        else:
            three = np.ones_like(good)
    else:
        import jax
        def _resolve(args):
            br, tr = _bracketed_roots(*args, xp)
            return tuple(xp.where(good, l, r_) for l, r_ in zip(leg, br)), tr | good
        roots, three = jax.lax.cond(xp.any(~good), _resolve, lambda args: (tuple(leg), xp.ones_like(good)), (d, c, b, a))
        roots = list(roots)
    # Valid: four distinct real quartic roots, contact not on a root (p(d) != 0; l0.r0 is
    # proportional to p(d)), physical inputs.  Near-coalescence is left to the
    # production condition-number test (production_spectral).
    valid = (three & _ordered(roots, xp) & (pd != 0) & (n > 0) & (t > 0) & (i >= 0)
             & (xp.asarray(tau) >= 0) & (xp.asarray(mu) > 0))
    return tuple(roots), valid, good


def contact_shifted(n, t, i, d, tau, mu, psi):
    """Coefficients of q(e) = p(d + e) = e^4 + 3d e^3 + q2 e^2 + q1 e + q0, e = lam - Vi.
    q0 = p(d) in its exact factored form (no cancellation beyond the data's own)."""
    ct = (44241.0 * mu - 7100.0) / 15000.0
    bt = 28191.0 * mu / 7500.0
    s_i = (5.0 * mu * tau / 3.0) * (n * i)
    c = -(s_i + ct * t)
    q2 = 3.0 * d * d + c
    q1 = d * (d * d - s_i + (bt - 2.0 * ct) * t)
    q0 = 71.0 * t * (213.0 * mu * t + mu * tau * i * (855.0 - 500.0 * n) + 300.0 * d * d) / 45000.0
    return q2, q1, q0


def _dq(e, d, q2, q1):
    return ((4.0 * e + 9.0 * d) * e + 2.0 * q2) * e + q1


def eigensystem(u, tau, mu, psi, xp=np, robust=True):
    """lam[k], right r[k], left l[k] (l.r = 1); k=0 contact (Vi), 1..4 = quartic roots.

    Roots are re-polished in the contact-shifted variable e = x - d = lam - Vi
                  (2 Newton steps on q(e)), lam = Vi + e, and the projector normalisations
                  taken from their exact forms  l_k.r_k = (3/mu) e^2 p'(x)  and
                  l0.r0 = 150 p(d)/(71 Te mu), avoiding the O(e) -> O(e^2)
                  cancellation of the dot products near contact coalescence.
    """
    n, t, i, v, w = (u[:, j] for j in range(5))
    d = v - w
    roots, valid, legacy_ok = quartic_roots(n, t, i, d, tau, mu, psi, xp, robust)
    q2, q1, q0 = contact_shifted(n, t, i, d, tau, mu, psi)
    es = []
    for x in roots:
        e = x - d
        for _ in range(2):
            de = _dq(e, d, q2, q1)
            e = e - ((((e + 3.0 * d) * e + q2) * e + q1) * e + q0) / xp.where(de != 0, de, 1.0)
        es.append(e)
    roots = [d + e for e in es]
    lam = [v] + [v + e for e in es]
    one = xp.ones_like(n)
    zero = 0.0 * one
    sig = tau * n
    un = sig - 1.71 * tau
    ut = d * d / mu + 0.71 * t + i * (1.71 * tau - sig)
    ue = -(t + tau * i) * un - tau * ut
    r0 = (n * un, ue, ut, d * (2.0 * t * un - 3.0 * ue) / (1.42 * t), d * un)
    l0 = (-2.0 * i / (3.0 * n), zero, one, zero, zero)
    rs, ls = [r0], [l0]
    for x, e in zip(roots, es):
        T = (x * x / mu - t - 5.0 * (tau * n) * i / 3.0)
        V = x * x / mu + 0.71 * t + 5.0 / 3.0 * i * (1.71 * tau - tau * n)
        r = (1.71 * n * e, T * e, 1.14 * i * e, V, 1.71 * x * e)
        beta = 3.0 * e * e
        delta = -1.42 * t * e
        q = 3.0 * x * e + 1.42 * t
        eps = q * e / (1.71 * mu)
        gam = tau * (-1.42 * t + n * q / 1.71)
        alpha = (x * eps - 1.14 * t * beta - 2.0 * i * gam / 3.0) / n
        rs.append(r)
        ls.append((alpha, beta, gam, delta, eps))
    norms = [150.0 * q0 / (71.0 * t * mu)]
    norms += [3.0 / mu * e * e * _dq(e, d, q2, q1) for e in es]
    out_l = [tuple(a_ / s for a_ in l) for l, s in zip(ls, norms)]
    return lam, rs, out_l, valid


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4]


def material(u, tau, mu, psi, xp=np):
    n, t, i, v, w = (u[:, j] for j in range(5))
    dv = v - w
    z = 0.0 * n
    a40 = mu * t / n + (mu * tau * i)
    a42 = mu * tau * n
    rows = [(w, z, z, z, n),
            (-1.42 * t * dv / (3 * n), w, z, -1.42 * t / 3, 3.42 * t / 3),
            (-2 * i * dv / (3 * n), z, v, z, 2 * i / 3),
            ((t + tau * i) / n, z + 1, z + tau, v, z),
            (a40, z + 1.71 * mu, a42, z, w)]
    return xp.stack([xp.stack(r, axis=1) for r in rows], axis=1)


def jac_action(u, g, tau, mu, psi):
    """J(g)[j] = (dA/du_j) g as tuples."""
    n, t, i, v, w = (u[:, j] for j in range(5))
    dv = v - w
    g0, g1, g2, g3, g4 = g
    z = 0.0 * n
    c0_4 = -mu * t * g0 / (n * n) + (mu * tau * g2)
    c0 = (g4, 1.42 * t * dv * g0 / (3 * n * n), 2 * i * dv * g0 / (3 * n * n), -(t + tau * i) * g0 / (n * n), c0_4)
    c1 = (z, -1.42 * dv * g0 / (3 * n) - 1.42 / 3 * g3 + 3.42 / 3 * g4, z, g0 / n, mu * g0 / n)
    c2 = (z, z, -2 * dv * g0 / (3 * n) + 2 / 3 * g4, tau * g0 / n, (mu * tau * g0))
    c3 = (z, -1.42 * t * g0 / (3 * n), -2 * i * g0 / (3 * n) + g2, g3, z)
    c4 = (g0, 1.42 * t * g0 / (3 * n) + g1, 2 * i * g0 / (3 * n), z, g4)
    return (c0, c1, c2, c3, c4)


def divided_differences(lam, xp=np):
    dd = [[None] * 5 for _ in range(5)]
    for a_ in range(5):
        for b_ in range(5):
            den = lam[a_] - lam[b_]
            num = xp.abs(lam[a_]) - xp.abs(lam[b_])
            safe = xp.where(xp.abs(den) > 1e-12, den, 1.0)
            q = xp.where(xp.abs(den) > 1e-12, num / safe, 0.0)
            same = xp.sign(lam[a_]) == xp.sign(lam[b_])
            dd[a_][b_] = xp.where(same, xp.sign(lam[a_]), q)
    return dd


# ---------------- Robust closed-form characteristic API (zero-speed derivative convention) ----------------
def _contact_free(rs, ls, cI, cA, A, wts, xp):
    """cI*I + cA*A + sum_{k>=1} w_k r_k l_k^T.  Used with cA = 0: F(A) = f0 I + sum_{k>=1}
    (f_k - f0) P_k, i.e. the contact projector P0 is eliminated with sum_k P_k = I, so a
    near-coalescing contact/sound pair (gap -> 0) only enters as (f_k - f0) P_k = O(1)."""
    rows = []
    for a_ in range(5):
        row = []
        for b_ in range(5):
            s_ = cA * A[:, a_, b_] + (cI if a_ == b_ else 0.0)
            for k in range(1, 5):
                s_ = s_ + wts[k] * rs[k][a_] * ls[k][b_]
            row.append(s_)
        rows.append(xp.stack(row, axis=1))
    return xp.stack(rows, axis=1)


NEAR = 1e-4   # contact/sound gap below NEAR * max|lam| uses the pair-interpolated form


def _spectral_weights(lam, f, same, beta, xp):
    """Coefficients (cI, cA, w) of F = cI I + cA A + sum_{k>=1} w_k P_k representing
    sum_k f_k P_k.  Default: P0 eliminated (cI = f0, w_k = f_k - f0).  Where the nearest
    quartic root s is within NEAR*max|lam| of the contact and in the same class
    (`same[k]`), both P0 and P_s are eliminated with the interpolant a + b lam through
    (lam0, f0), (lam_s, f_s): cI = a, cA = b (b = `beta[k]`, given exactly), w_s = 0."""
    gaps = [xp.abs(l_ - lam[0]) for l_ in lam[1:]]
    gmin = xp.minimum(xp.minimum(gaps[0], gaps[1]), xp.minimum(gaps[2], gaps[3]))
    scale = xp.maximum(xp.maximum(xp.abs(lam[1]), xp.abs(lam[4])), 1.0)
    cI = f[0]; cA = 0.0 * f[0]
    w = [None] + [f[k] - f[0] for k in range(1, 5)]
    for k in range(1, 5):
        sel = (gaps[k - 1] == gmin) & (gmin < NEAR * scale) & same[k]
        a = f[0] - beta[k] * lam[0]
        cI = xp.where(sel, a, cI); cA = xp.where(sel, beta[k], cA)
        w = [w[0]] + [xp.where(sel, (0.0 * f[j] if j == k else f[j] - a - beta[k] * lam[j]), w[j]) for j in range(1, 5)]
    return cI, cA, w


def absolute(u, tau=1.0, mu=1836.0, psi=PI, xp=np, sysm=None, robust=True):
    lam, rs, ls, _ = eigensystem(u, tau, mu, psi, xp, robust) if sysm is None else sysm
    if not robust:
        raise ValueError("Only the robust spectral absolute value is supported")
    f = [xp.abs(l_) for l_ in lam]
    sg = [xp.sign(l_) for l_ in lam]
    cI, cA, w = _spectral_weights(lam, f, [None] + [sg[k] == sg[0] for k in range(1, 5)], [None] + [sg[0]] * 4, xp)
    return _contact_free(rs, ls, cI, cA, material(u, tau, mu, psi, xp), w, xp)


def absolute_apply(u, x, tau=1.0, mu=1836.0, psi=PI, xp=np, sysm=None, robust=True):
    lam, rs, ls, _ = eigensystem(u, tau, mu, psi, xp, robust) if sysm is None else sysm
    xs = tuple(x[:, j] for j in range(5))
    if not robust:
        raise ValueError("Only the robust spectral action is supported")
    f = [xp.abs(l_) for l_ in lam]
    sg = [xp.sign(l_) for l_ in lam]
    cI, cA, w = _spectral_weights(lam, f, [None] + [sg[k] == sg[0] for k in range(1, 5)], [None] + [sg[0]] * 4, xp)
    A = material(u, tau, mu, psi, xp)
    out = [cI * xs[a_] + cA * sum(A[:, a_, j] * xs[j] for j in range(5)) for a_ in range(5)]
    for k in range(1, 5):
        cf = w[k] * _dot(ls[k], xs)
        out = [o + cf * rk for o, rk in zip(out, rs[k])]
    return xp.stack(out, axis=1)


def abs_derivative_contraction(u, z, tau=1.0, mu=1836.0, psi=PI, xp=np, sysm=None, need_abs=True, robust=True):
    sysm = eigensystem(u, tau, mu, psi, xp, robust) if sysm is None else sysm
    lam, rs, ls, _ = sysm
    dd = divided_differences(lam, xp)
    zs = tuple(z[:, j] for j in range(5))
    zeta = [_dot(ls[m], zs) for m in range(5)]
    E = [[0.0] * 5 for _ in range(5)]
    for k in range(5):
        gk = [0.0] * 5
        for m in range(5):
            cf = dd[k][m] * zeta[m]
            gk = [gg + cf * rr for gg, rr in zip(gk, rs[m])]
        J = jac_action(u, gk, tau, mu, psi)
        qk = [_dot(ls[k], J[j]) for j in range(5)]
        for a_ in range(5):
            for j in range(5):
                E[a_][j] = E[a_][j] + rs[k][a_] * qk[j]
    Em = xp.stack([xp.stack(E[a_], axis=1) for a_ in range(5)], axis=1)
    return (absolute(u, tau, mu, psi, xp, sysm, robust) if need_abs else None), Em


# ---------------- production-semantics API ----------------
def production_condition(rs, ls, xp=np):
    """Frobenius cond of the unit-column eigenvector matrix (LAPACK eig convention)."""
    tot = 0.0
    for r, l in zip(rs, ls):          # l is scaled so l.r = 1
        tot = tot + _dot(r, r) * _dot(l, l)
    return xp.sqrt(5.0 * tot)


# Production fallback wrappers are deliberately excluded from this Q package.
