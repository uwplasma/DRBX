# Literature reproduction guide

This section explains how to use `jaxdrb` to reproduce **common linear SOL/edge analysis workflows**
seen in the literature, using the cold-ion drift-reduced Braginskii-like model and the
matrix-free eigen/initial-value solvers.

## Scope and caveats (important)

The papers in this repo include:

- Mosetto et al. (2012): low-frequency linear-mode regimes (drift waves and ballooning modes),
- Halpern et al. (2013): transition to ideal ballooning and SOL-width estimation via gradient removal,
- Jorge et al. (2016): SOL turbulence context and diagnostics (ISTTOK),
- Jorge & Landreman (2021): near-axis stellarator geometry for turbulence simulations.

The default `jaxdrb` model is **electrostatic** and intentionally simplified:

- periodic field-line boundary conditions,
- Boussinesq polarization closure in Fourier form,
- open-field-line MPSE/sheath boundary physics is essential in many SOL studies; `jaxdrb` includes
  both a simplified velocity-only MPSE enforcement mode and a Loizu-2012-inspired linearized
  “full set” for open geometries,
- the electrostatic model omits magnetic induction / $A_\parallel$ (an electromagnetic extension
  model exists, but does not aim to reproduce every published closure term),
- reduced closure set (no gyroviscosity, no full heat flux closures).

Even with these limitations, `jaxdrb` can reproduce many **methodological** steps used in the
papers:

- scanning $\gamma(k_y)$ (and optionally $k_x$),
- identifying resistive-like vs inertial-like branches by toggling parameters,
- evaluating the proxy $(\gamma/k_y)_{\max}$,
- computing a self-consistent SOL width $L_p$ using the fixed-point rule from Halpern (2013),
- swapping magnetic geometry by changing only a geometry provider or a `.npz` file.

## What to run

See the scripts in:

- `examples/scripts/06_literature_tokamak_sol/mosetto2012_driftwave_branches.py`
- `examples/scripts/06_literature_tokamak_sol/mosetto2012_ballooning_branches.py`
- `examples/scripts/06_literature_tokamak_sol/halpern2013_gradient_removal_lp.py`
- `examples/scripts/07_essos_geometries/stellarator_nearaxis_essos.py` (stellarator near-axis / ESSOS-driven workflow).
- `examples/scripts/06_literature_tokamak_sol/mosetto2012_regime_map.py` (Mosetto-like regime map: InDW/RDW/InBM/RBM).
- `examples/scripts/06_literature_tokamak_sol/halpern2013_salpha_ideal_ballooning_map.py` (s–alpha growth-rate map).
- `examples/scripts/07_essos_geometries/essos_vmec_edge_s09.py` (VMEC field line via ESSOS at s=0.9).
- `examples/scripts/07_essos_geometries/essos_biotsavart_r14.py` (Biot–Savart field line via ESSOS).
- `examples/scripts/03_sheath_mpse/loizu2012_full_mpse_bc.py` (MPSE BC comparison: simple vs Loizu-2012 mode).

Each script writes an `out/...` folder with:

- a machine-readable `.npz` results file,
- one or more plots (`.png`),
- a `params.json` / `summary.json` describing the run.

## Where to look next

- SOL width estimation: `literature/sol_width.md`
- Mosetto (2012) regimes and branch toggles: `literature/mosetto2012.md`
- Halpern (2013) gradient removal workflow: `literature/halpern2013.md`
- Jorge et al. (2016) ISTTOK context (and `jaxdrb` mapping): `literature/jorge2016_isttok.md`
- Near-axis stellarator geometry (ESSOS): `literature/jorge2021_stellarator.md`
