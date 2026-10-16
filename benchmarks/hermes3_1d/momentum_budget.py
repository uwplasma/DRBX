"""Upstream-to-target plasma momentum budget for one Hermes-3 run and one DRBX run (same grid).

Usage: python momentum_budget.py HERMES_RUN_DIR DRBX_JSON
(DRBX_JSON from run_drbx.py, ideally at the n_up that gives the Hermes-3 Gamma_t A_t on that grid;
needs drbx importable, with the same src/ that produced the JSON.)

Each code's discrete steady ion momentum equation, summed over all cells (A dl weights), gives
    P_up + sum P dA - F_fric = A_t P_f + A_t (m n v^2)_f
P_up: upstream P_e + P_i. sum P dA: flux-tube expansion force. F_fric: momentum given to the
neutrals (CX friction minus the ionisation momentum gain). A_t P_f: target-face pressure as seen by
the central pressure gradient (mean of last cell and guard). A_t (m n v^2)_f: momentum flux through
the sheath face, from the closure of the sum (the scheme's own boundary flux). All terms in Pa per
upstream m^2 of flux tube. Also printed: the CX reaction count and the CX-weighted slip
<v_i - v_n>, which isolate how the neutral velocity sets the friction.
"""

import json
import sys

import netCDF4
import numpy as np

import drbx.native.neutrals.detachment_sol_model as m

QE, MP = 1.602176634e-19, 1.67262192e-27


def hermes(path):
    d = netCDF4.Dataset(f"{path}/BOUT.dmp.0.nc")

    def g(k):
        a = np.squeeze(np.asarray(d[k][:]))
        return a[-1] if a.ndim == 2 else a

    nn, tn = float(d["Nnorm"][:]), float(d["Tnorm"][:])
    pa, cs0, rho = nn * tn * QE, float(d["Cs0"][:]), float(d["rho_s0"][:])
    area = g("J") / g("J")[2]
    dl = g("dy") * np.sqrt(g("g_22"))
    i = slice(2, -2)
    vol = area[i] * dl[i]

    def grad(f):
        return (f[3:-1] - f[1:-3]) / (2 * dl[i])

    pe, pt = g("Pe"), g("Pe") + g("Ph+")
    fric = (
        -np.sum((g("SNVh+")[i] + grad(pe)) * vol) * pa
    )  # ion sources without the E-field force
    minus_gradp = -np.sum(grad(pt) * vol) * pa
    expansion = np.sum(pt[i] * 0.5 * (area[3:-1] - area[1:-3])) * pa
    ne, nh, te = g("Ne")[i] * nn, g("Nh")[i] * nn, g("Te")[i] * tn
    slip = (g("Vh+")[i] - g("Vh")[i]) * cs0
    rcx = ne * nh * np.asarray(m.sd1d_charge_exchange_rate(te)) * vol * rho
    return dict(
        P_up=pt[2] * pa,
        expansion=expansion,
        friction=fric,
        face_pressure=pt[2] * pa + expansion - minus_gradp,
        face_momentum_flux=minus_gradp - fric,
        cx_reactions=float(np.sum(rcx)),
        cx_slip=float(np.sum(rcx * slip) / np.sum(rcx)),
        T_t=0.5 * (g("Te")[-3] + g("Th+")[-3]) * tn,
    )


def drbx(path):
    r = json.load(open(path))
    p = m.DetachmentSolParameters(ny=r["ny"], upstream_density=r["n_up"])
    grid = m._grid(p)
    st = np.stack(
        [np.asarray(r["state"][k]) for k in m.DetachmentSolState._fields], axis=1
    )
    t = m._terms(st, 1.0, m._theta(p), p, grid)
    pa = p.Nnorm * p.Tnorm * QE
    area, dl = np.asarray(grid.area), np.asarray(grid.dl)
    vol = area[2:-2] * dl[2:-2]
    pt = st[:, 2]
    px = np.concatenate([[pt[0]] * 2, pt, [2 * float(t.Pout) - pt[-1]] * 2])
    minus_gradp = -np.sum((px[3:-1] - px[1:-3]) / (2 * dl[2:-2]) * vol) * pa
    expansion = np.sum(pt * 0.5 * (area[3:-1] - area[1:-3])) * pa
    fric = np.sum(np.asarray(t.F) * vol) * pa
    ne, nh, te = st[:, 0] * p.Nnorm, st[:, 3] * p.Nnorm, np.asarray(t.Te) * p.Tnorm
    slip = (st[:, 1] / st[:, 0] - st[:, 4] / np.maximum(st[:, 3], 1e-5)) * p.cs0
    rcx = ne * nh * np.asarray(m.sd1d_charge_exchange_rate(te)) * vol * p.rho_s0
    return dict(
        P_up=pt[0] * pa,
        expansion=expansion,
        friction=fric,
        face_pressure=pt[0] * pa + expansion - minus_gradp,
        face_momentum_flux=minus_gradp - fric,
        cx_reactions=float(np.sum(rcx)),
        cx_slip=float(np.sum(rcx * slip) / np.sum(rcx)),
        T_t=float(t.Te[-1]) * p.Tnorm,
    )


h, d = hermes(sys.argv[1]), drbx(sys.argv[2])
print(
    f"{'term':20s} {'Hermes-3':>12s} {'DRBX':>12s} {'H - D':>10s}   (Pa per upstream m^2; cx_reactions s^-1 m^-2; cx_slip m/s)"
)
for k in h:
    print(f"{k:20s} {h[k]:12.5g} {d[k]:12.5g} {h[k] - d[k]:+10.4g}")
print(
    "closure: P_up + expansion - friction = face_pressure + face_momentum_flux holds by construction"
)
