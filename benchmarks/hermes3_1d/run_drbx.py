"""DRBX side of the Hermes-3 L3/L4 comparison: SD1D-matched steady states on 100-1600 cells.

Usage: python run_drbx.py OUTDIR VARIANT N_UP [N_UP ...]
VARIANT: base | key=value[,key=value] (DetachmentSolParameters fields) | amjuel_rec (recombination rate replaced by the AMJUEL H.4 2.1.8 fit that
Hermes-3 uses; needs HERMES_JSON=<hermes-3>/json_database) | no_rec (recombination off).
Writes one JSON per (n_up, ny) with the profiles, ledger and normalised state that compare.py and
momentum_budget.py read. The consistent sheath momentum flux is the option sheath_momentum_flux=face.
"""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import drbx.native.neutrals.detachment_sol_model as m

out, variant = Path(sys.argv[1]), sys.argv[2]
out.mkdir(parents=True, exist_ok=True)
kw = {"no_rec": {"recombination": False}}.get(variant, {})
if "=" in variant:  # e.g. sheath_momentum_flux=face,sheath_gamma=5.5 (any DetachmentSolParameters field)
    kw = {
        k: (float(v) if v.replace(".", "", 1).replace("e", "", 1).isdigit() else v)
        for k, v in (a.split("=") for a in variant.split(","))
    }
if variant == "amjuel_rec":
    c = np.array(
        json.load(open(Path(os.environ["HERMES_JSON"]) / "AMJUEL_H.4_2.1.8.json"))[
            "coeffs"
        ]
    )

    def amjuel_rec(
        n, t
    ):  # Hermes-3 eval_amjuel_nT_fit: rows = ln T power, columns = ln(n/1e14) power
        lt, ln = jnp.log(jnp.clip(t, 0.1, 1e4)), jnp.log(jnp.clip(n, 1e14, 1e22) / 1e14)
        return (
            jnp.exp(sum(c[i, j] * lt**i * ln**j for i in range(9) for j in range(9)))
            * 1e-6
        )

    m.sd1d_recombination_rate = amjuel_rec

for n_up in map(float, sys.argv[3:]):
    prev = None
    for ny in (100, 200, 400, 800, 1600):
        p = m.DetachmentSolParameters(ny=ny, upstream_density=n_up, **kw)
        dl, area, y = m.sd1d_grid(p)
        # grid-independent integrated power: the discrete H() source mask covers 9.86-10.09 m
        p = replace(
            p,
            power_flux=p.power_flux
            * p.length_xpt
            / float(np.sum(dl[np.asarray(m._grid(p).source_mask) > 0])),
        )
        r = m.detachment_sol_run(
            p,
            None if prev is None else prev[0].state,
            source_scale=0.1 if prev is None else prev[0].source_scale,
            source_params=None if prev is None else prev[1],
            tol=1e-9,
            max_iter=800,
        )
        prev = (r, p)
        t = m._terms(
            jnp.stack(r.state, axis=1), r.source_scale, m._theta(p), p, m._grid(p)
        )
        te = np.asarray(t.Te) * p.Tnorm
        rec = dict(
            code="drbx",
            variant=variant,
            n_up=n_up,
            ny=ny,
            converged=bool(r.converged),
            residual=r.residual,
            y=y.tolist(),
            dl=dl.tolist(),
            area=area.tolist(),
            ne=(np.asarray(r.state.ion_density) * p.Nnorm).tolist(),
            te=te.tolist(),
            nn=(np.asarray(r.state.neutral_density) * p.Nnorm).tolist(),
            flux_area=float(t.target_flux * t.target_area) * p.Nnorm * p.cs0,
            ledger={k: float(v) for k, v in m.detachment_ledger(r, p).items()},
            state={
                k: np.asarray(v).tolist() for k, v in r.state._asdict().items()
            },  # normalised (momentum_budget.py)
        )
        f = out / f"drbx_{variant.replace('=', '-').replace(',', '_')}_{n_up:.5e}_{ny}.json"
        f.write_text(json.dumps(rec))
        print(
            f.name,
            r.converged,
            f"{r.residual:.1e}",
            f"Tt={te[-1]:.4f}",
            f"GA={rec['flux_area']:.5e}",
            flush=True,
        )
