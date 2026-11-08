"""Gap of one DRBX variant against an aligned Hermes-3 attached run, per grid and extrapolated.

Usage: python swap_gap.py HERMES_DIR SOURCE DRBX_DIR [DRBX_DIR ...]
HERMES_DIR holds h_<SOURCE>_<ny>_*.json (aligned runs); each DRBX_DIR holds drbx_*_<n_up>_<ny>.json
for several n_up bracketing the Hermes flow. For every grid DRBX n_up and T_t are interpolated (log)
at the Hermes flow on the same grid; the extrapolated row uses the 400/800/1600 triplet.
"""
import glob, json, sys
import numpy as np
from compare import metrics, gci

hdir, src = sys.argv[1:3]
H = {}
for f in glob.glob(f"{hdir}/h_{src}_*.json"):
    r = json.load(open(f)); H[r["ny"]] = metrics(r)
grids = sorted(H)

def row(ddir):
    D = {}
    for f in glob.glob(f"{ddir}/drbx_*.json"):
        r = json.load(open(f)); D.setdefault(r["ny"], {})[r["n_up"]] = metrics(r)
    res = {}
    for g in grids:
        if g not in D or len(D[g]) < 2: continue
        n = sorted(D[g]); ga = np.log([D[g][k]["GA"] for k in n])
        o = np.argsort(ga); x = np.log(H[g]["GA"])
        nup = float(np.exp(np.interp(x, ga[o], np.log(n)[o]))); tt = float(np.interp(x, ga[o], np.array([D[g][k]["T_t"] for k in n])[o]))
        res[g] = (nup, tt)
    return res

def ext(vals):  # vals at 400/800/1600
    p, f, u = gci(vals)[0], gci(vals)[1], gci(vals)[2]
    return f, u, p

print(f"Hermes flow {H[grids[-1]]['GA']:.4e}")
print("grid | Hermes n_up T_t")
for g in grids: print(g, f"{H[g]['n_up']/1e19:.4f} {H[g]['T_t']:.3f}")
hx = [ext([H[g][k] for g in (400, 800, 1600)]) for k in ("n_up", "T_t")]
print(f"Hermes extrapolated n_up {hx[0][0]/1e19:.4f}+-{hx[0][1]/1e19:.4f} (p={hx[0][2]:.2f}), T_t {hx[1][0]:.3f}+-{hx[1][1]:.3f}")
for d in sys.argv[3:]:
    r = row(d)
    line = [d.split("/")[-1]]
    for g in grids:
        if g in r: line.append(f"{g}:{H[g]['n_up']/r[g][0]-1:+.1%}/{H[g]['T_t']/r[g][1]-1:+.1%}")
    if all(g in r for g in (400, 800, 1600)):
        en, eu, ep = ext([r[g][0] for g in (400, 800, 1600)]); et, etu, etp = ext([r[g][1] for g in (400, 800, 1600)])
        line.append(f"EXT n_up {en/1e19:.4f}+-{eu/1e19:.4f}(p={ep:.2f}) gap {hx[0][0]/en-1:+.2%} | T_t {et:.3f}+-{etu:.3f}(p={etp:.2f}) gap {hx[1][0]/et-1:+.2%}")
    print(" ".join(line))
