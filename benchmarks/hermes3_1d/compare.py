"""Hermes-3 vs DRBX (SD1D model), benchmark ladder L3/L4: per-code order, Richardson, GCI, verdict.

Usage: python compare.py HERMES_DIR DRBX_DIR [--variant base]

Both codes see the same SD1D deck, but they hold a different control fixed:
- Hermes-3 runs with a fixed particle source S (no feedback controller). At steady state with 99 %
  recycling the target flow is Gamma_t A_t = S L_xpt / 0.01 exactly, so n_up is the output.
- DRBX holds n_up fixed (SD1D's PI controller, algebraic) and returns Gamma_t A_t.
So the comparison is made at equal Gamma_t A_t. For each DRBX n_up the grid-extrapolated value is
computed; the DRBX extrapolated curves are interpolated (in log n_up) at the Hermes flow.

Observables (fixed before comparing): n_up at equal flow, T_t (last cell; Hermes (Te+Ti)/2),
T_up (first cell), L_iz (ionisation-weighted distance from the target, same post-processing).
Uncertainty per code = GCI (three finest grids, ratio 2, Fs 1.25 when the observed order is in
[0.5, 2.5], else Fs 3). Tolerance = U_Hermes + U_DRBX + measured model-difference bound (see main).
"""

import glob
import json
import sys

import numpy as np

# Model-difference bound (relative), per observable, measured rather than assumed:
#  - recombination: Hermes-3 uses AMJUEL H.4 2.1.8, SD1D/DRBX its own fit. Bound = |DRBX(amjuel_rec)
#    - DRBX(base)| / DRBX(base) on the finest grid, interpolated at the same flow (drbx_amjuel_rec_*).
#  - two temperatures: Hermes-3 keeps T_e and T_i (coupled with ei_multiplier = 1e5); SD1D has one T.
#    Bound on T_t = |T_e - T_i| / (T_e + T_i) at the target on the finest Hermes grid.
# Discretisation differences (MC vs MinMod limiter, cell-centre vs Simpson rates) vanish in the
# extrapolated values and are covered by the GCIs, not by this bound.
KEYS = ("n_up", "T_t", "T_up", "L_iz")


def willett(t):
    c = (
        -3.271397e1,
        1.353656e1,
        -5.739329,
        1.563155,
        -2.877056e-1,
        3.482560e-2,
        -2.631976e-3,
        1.119544e-4,
        -2.039150e-6,
    )
    x = np.log(np.maximum(t, 0.025))
    return np.exp(sum(ci * x**k for k, ci in enumerate(c))) * 1e-6


def metrics(r):
    y, dl, a = (np.asarray(r[k]) for k in ("y", "dl", "area"))
    ne, te, nn = (np.asarray(r[k]) for k in ("ne", "te", "nn"))
    w = ne * nn * willett(te) * dl * a
    return dict(
        n_up=float(ne[0]),
        GA=float(r["flux_area"]),
        T_t=float(te[-1]),
        T_up=float(te[0]),
        L_iz=float(np.sum(w * (y[-1] + dl[-1] / 2 - y)) / np.sum(w)),
    )


def gci(f):
    """Three grids coarse -> fine, ratio 2. Returns (p, f_ext, U), U = GCI * |f_fine| (absolute)."""
    f3, f2, f1 = f
    e21, e32 = f1 - f2, f2 - f3
    if e21 == 0:
        return np.inf, f1, 0.0
    ratio = e32 / e21
    if ratio <= 0:  # oscillatory convergence: not asymptotic, conservative band
        return np.nan, f1, 3.0 * max(abs(e21), abs(e32))
    p = np.log(ratio) / np.log(2.0)
    pp = min(max(p, 0.5), 2.0)
    fs = 1.25 if 0.5 <= p <= 2.5 else 3.0
    return p, f1 + e21 / (2**pp - 1), fs * abs(e21) / (2**pp - 1)


def extrap(runs):
    grids = sorted(runs)[-3:]
    m = [metrics(runs[g]) for g in grids]
    return {k: (grids, [x[k] for x in m], *gci([x[k] for x in m])) for k in m[0]}


def main():
    hdir, ddir = sys.argv[1:3]
    variant = (
        sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "base"
    )
    # DRBX extrapolated curves vs n_up
    D = {}
    for f in glob.glob(f"{ddir}/drbx_{variant}_*.json"):
        r = json.load(open(f))
        D.setdefault(r["n_up"], {})[r["ny"]] = r
    nups = sorted(n for n in D if len(D[n]) >= 3)
    dx = {n: extrap(D[n]) for n in nups}
    lga = np.log([dx[n]["GA"][3] for n in nups])
    assert np.all(np.diff(lga) > 0), (
        "DRBX Gamma_t A_t must increase with n_up over the scan"
    )

    def curve(k, j):
        return np.array([dx[n][k][j] for n in nups])

    A = {}
    for f in glob.glob(f"{ddir}/drbx_amjuel_rec_*.json"):
        r = json.load(open(f))
        A.setdefault(r["n_up"], {})[r["ny"]] = r
    rec = {}
    for n in sorted(A):
        g = max(A[n])
        if n in D and g in D[n]:
            mb, ma = metrics(D[n][g]), metrics(A[n][g])
            rec[np.log(mb["GA"])] = {k: abs(ma[k] / mb[k] - 1) for k in KEYS}
            # DRBX holds n_up: map its flow change to n_up at fixed flow through d ln n / d ln GA (~1)
            rec[np.log(mb["GA"])]["n_up"] = abs(ma["GA"] / mb["GA"] - 1)

    H = {}
    for f in glob.glob(f"{hdir}/h_*.json"):
        r = json.load(open(f))
        H.setdefault(f.split("/")[-1].split("_")[1], {})[r["ny"]] = r
    out = []
    for src in sorted(H, key=float):
        if len(H[src]) < 3:
            print(f"source {src}: only grids {sorted(H[src])}, skipped")
            continue
        hx = extrap(H[src])
        ga = hx["GA"][1][-1]
        x = np.log(ga)
        if not lga[0] <= x <= lga[-1]:
            print(f"source {src}: Gamma_t A_t {ga:.4e} outside the DRBX scan, skipped")
            continue
        led = H[src][max(H[src])]["ledger"]
        print(
            f"\n== source {src} m^-3 s^-1 -> Gamma_t A_t = {ga:.5e} s^-1 per upstream m^2 "
            f"(Hermes grids {hx['T_t'][0]}, finest ledger particle {led['particle_imbalance']:.1e} "
            f"power {led['power_imbalance']:.1e}, drift {H[src][max(H[src])]['drift']:.1e})"
        )
        fin = H[src][max(H[src])]
        bound = {
            k: (
                float(np.interp(x, sorted(rec), [rec[q][k] for q in sorted(rec)]))
                if rec
                else 0.0
            )
            for k in KEYS
        }
        bound["T_t"] += abs(fin["te_e"][-1] - fin["ti"][-1]) / (
            fin["te_e"][-1] + fin["ti"][-1]
        )
        for k in KEYS:
            if k == "n_up":
                fd = float(np.exp(np.interp(x, lga, np.log(nups))))
                # DRBX GA uncertainty mapped to n_up through the local slope d ln n / d ln GA
                s = np.gradient(np.log(nups), lga)
                ud = float(np.interp(x, lga, s * curve("GA", 4) / curve("GA", 3))) * fd
                pd = float(np.interp(x, lga, curve("GA", 2)))
            else:
                fd, ud, pd = (float(np.interp(x, lga, curve(k, j))) for j in (3, 4, 2))
            _, vh, ph, fh, uh = hx[k]
            tol = uh + ud + bound[k] * abs(fd)
            diff = fh - fd
            ok = bool(abs(diff) <= tol)
            print(
                f" {k:5s} Hermes {' '.join(f'{v:.4g}' for v in vh)} p={ph:.2f} ext={fh:.4g}+-{uh:.2g} | "
                f"DRBX p={pd:.2f} ext={fd:.4g}+-{ud:.2g} | diff={diff:+.3g} ({diff / fd:+.1%}) "
                f"bound={bound[k]:.1e} tol={tol:.2g} {'AGREE' if ok else 'DIFFER'}"
            )
            out.append(
                dict(
                    source=float(src),
                    flow=ga,
                    key=k,
                    hermes=fh,
                    u_hermes=uh,
                    p_hermes=ph,
                    drbx=fd,
                    u_drbx=ud,
                    p_drbx=pd,
                    model_bound=bound[k],
                    diff=diff,
                    tol=tol,
                    agree=ok,
                )
            )
    json.dump(out, open(f"{hdir}/compare_{variant}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
