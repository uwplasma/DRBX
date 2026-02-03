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
- no *full* sheath boundary conditions (line-tied/sheath physics is essential in some SOL studies),
  though a lightweight volumetric sheath-loss closure exists for open-field-line geometries,
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

- `examples/3_advanced/01_mosetto2012_driftwave_branches.py`
- `examples/3_advanced/02_mosetto2012_ballooning_branches.py`
- `examples/3_advanced/03_halpern2013_gradient_removal_lp.py`
- `examples/3_advanced/04_stellarator_nearaxis_pyqsc.py` (stellarator near-axis / pyQSC-driven workflow).

Each script writes an `out/3_advanced/...` folder with:

- a machine-readable `.npz` results file,
- one or more plots (`.png`),
- a `params.json` / `summary.json` describing the run.

## Where to look next

- SOL width estimation: `literature/sol_width.md`
- Mosetto (2012) regimes and branch toggles: `literature/mosetto2012.md`
- Halpern (2013) gradient removal workflow: `literature/halpern2013.md`
- Jorge et al. (2016) ISTTOK context (and `jaxdrb` mapping): `literature/jorge2016_isttok.md`
- Near-axis stellarator geometry (pyQSC): `literature/jorge2021_stellarator.md`
