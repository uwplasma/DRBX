"""Hermes-3 run directory -> JSON record with the same fields as run_drbx.py.

Usage: python extract_hermes.py RUNDIR OUT.json
Values are the last output; `drift` is the relative change of T_t and Gamma_t A_t
over the last 10 outputs (steady-state check).
"""

import json
import sys

import numpy as np
from boutdata import collect

d, out = sys.argv[1:3]


def g(name, guards=False):
    return np.squeeze(collect(name, path=d, info=False, xguards=False, yguards=guards))


Nn, Tn, cs, rho = (
    float(collect(k, path=d, info=False)) for k in ("Nnorm", "Tnorm", "Cs0", "rho_s0")
)
om = float(collect("Omega_ci", path=d, info=False))
dy, J, g22 = g("dy"), g("J"), g("g_22")
J0 = float(J[0])
JG = g("J", True) / J[0]  # area relative to upstream
J = J / J[0]
dl = dy * np.sqrt(g22) * rho  # m
y = np.cumsum(dl) - 0.5 * dl
vol = dl * J  # m per unit upstream area (J = 1 upstream)
ne, te, ti, nn = g("Ne") * Nn, g("Te") * Tn, g("Th+") * Tn, g("Nh") * Nn
neG, teG, tiG, viG = (g(k, True) for k in ("Ne", "Te", "Th+", "Vh+"))


def face(
    f, k=-1
):  # value at the sheath face: mean of last cell and first guard (one guard used)
    return 0.5 * (f[k, -3] + f[k, -2])


def ga(k):  # sheath-face particle flow per upstream m^2 [s^-1]
    return face(neG, k) * Nn * face(viG, k) * cs * 0.5 * (JG[-3] + JG[-2])


tbar = 0.5 * (te + ti)
src = g("Sh+_src") * Nn * om  # external (feedback) particle source
p_in = (
    1.5
    * np.sum((g("Pe_src") + g("Ph+_src"))[-1] * vol)
    * Nn
    * Tn
    * 1.602176634e-19
    * om
)
q_sheath = 6.0 * face(0.5 * (teG + tiG), -1) * Tn * 1.602176634e-19 * ga(-1)
pw = Nn * Tn * 1.602176634e-19 * om  # W m^-3 per normalised energy rate


def vint(f):  # volume integral per upstream m^2, last output
    return float(np.sum(f[-1] * vol))


def willett(t):  # SD1D/Hermes `ionisation` rate [m^3/s]
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


riz = ne * nn * willett(te)  # cell-centre ionisation rate [m^-3 s^-1]
A0 = float(
    g("dx", True)[2] * g("dz", True)[2] * JG[2] * J0 / np.sqrt(g("g_22", True)[2])
)  # flow -> per upstream m^2


def flow(k):  # normalised flow through the sheath face
    return float(g(k, True)[-1, -2]) / A0


sp = ("e", "h+", "h")
fw = Nn * Tn * 1.602176634e-19 * cs
e_src = (
    1.5 * sum(vint(g(f"SP{s}")) for s in sp) * pw
)  # all energy sources incl. sheath, conduction, atomic
e_work = (
    sum(vint(g(f"E{s}_VgradP")) for s in sp) * pw
)  # V grad P exchange with the flow (kinetic) energy
e_adv = sum(flow(f"ef{s}_adv_ylow") for s in sp) * fw  # 5/2 P V through the sheath face
ledger = dict(
    particle_source=vint(src),
    target_flux=flow("pfh+_tot_ylow") * Nn * cs,
    power_input=float(p_in),
    energy_sources_total=e_src,
    vgradp_work=e_work,
    target_advected_energy=e_adv,
    sheath_power_6Tnv=float(q_sheath),
    ionisation_radiation=13.6 * 1.602176634e-19 * vint(riz),
    recycled_neutral_energy=vint(g("Eh_target_recycle")) * pw,
)
ledger["particle_imbalance"] = (
    ledger["particle_source"] - 0.01 * ledger["target_flux"]
) / ledger["target_flux"]
ledger["power_imbalance"] = (e_src + e_work - e_adv) / ledger["power_input"]
kd = -min(11, tbar.shape[0])  # drift over the last 10 outputs (or the whole run)
rec = dict(
    code="hermes3",
    ny=len(dl),
    y=y.tolist(),
    dl=dl.tolist(),
    area=J.tolist(),
    n_up=float(ne[-1, 0]),
    ne=ne[-1].tolist(),
    te=tbar[-1].tolist(),
    te_e=te[-1].tolist(),
    ti=ti[-1].tolist(),
    nn=nn[-1].tolist(),
    flux_area=float(ga(-1)),
    drift=max(abs(tbar[-1, -1] / tbar[kd, -1] - 1), abs(ga(-1) / ga(kd) - 1)),
    ledger=ledger,
)
json.dump(rec, open(out, "w"))
print(
    out,
    f"n_up={rec['n_up']:.4e} T_t={tbar[-1, -1]:.3f} (Te {te[-1, -1]:.3f} Ti {ti[-1, -1]:.3f}) "
    f"T_up={tbar[-1, 0]:.2f} GA={rec['flux_area']:.4e} drift={rec['drift']:.1e}",
)
print({k: f"{v:.4g}" for k, v in ledger.items()})
