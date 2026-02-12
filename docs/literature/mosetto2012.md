# Mosetto (2012): drift-wave and ballooning branches

Mosetto et al. (Phys. Plasmas 19, 112103 (2012)) analyze low-frequency linear modes in the tokamak
scrape-off layer and identify distinct regimes:

- drift-wave branches (often separated into resistive vs inertial),
- ballooning-mode branches (resistive vs inertial, and an ideal branch at finite beta).

The default `jaxdrb` model is electrostatic, so it does not reproduce the ideal electromagnetic
branch quantitatively. An electromagnetic extension model is available (`--model em`) and is covered
by unit tests and 2D milestone benchmarks. Quantitative paper-to-paper agreement still requires
matching the paper’s normalization, closure set, and boundary conditions; `jaxdrb` provides a
transparent, reproducible workflow for branch separation and scanning, plus calibrated gates for
the reduced transition formulas used in Mosetto (2012).

## What `jaxdrb` maps to in the paper

The paper uses drift-reduced Braginskii equations in a flux-tube/field-line-following representation.
In `jaxdrb`:

- curvature is toggled by `params.curvature_on` and controlled by a geometry's curvature operator,
- resistivity vs inertia is controlled by `eta` and `me_hat`,
- the background gradient drive is controlled by `omega_n` (and optionally `omega_Te`).

## Regime-map workflow (InDW / RDW / InBM / RBM)

Mosetto (2012) discuss how low-frequency instabilities change character across parameter space.
A common workflow is to scan a grid of parameters (collisionality and gradient strength) and label
each point by which branch dominates.

Run:

```bash
python examples/06_literature_tokamak_sol/mosetto2012_regime_map.py
```

This example now provides a **calibrated 4-regime map** based on the transition formulas
discussed in Mosetto et al. (Sec. V), with explicit thresholds for:

- RDW ↔ RBM,
- InDW ↔ InBM,
- RDW ↔ InDW (through a calibrated `d` threshold fit).

You can also request an additional (slow) solver-ablation map for comparison:

```bash
python examples/06_literature_tokamak_sol/mosetto2012_regime_map.py --classifier both
```

![Mosetto-style regime map (calibrated 4-regime)](../assets/images/mosetto2012_regime_map.png)

## Drift-wave-like scan (curvature off)

Run:

```bash
python examples/06_literature_tokamak_sol/mosetto2012_driftwave_branches.py
```

Outputs in `out/mosetto2012_driftwave_branches/` include:

- `branches_overlay.png`: $\gamma(k_y)$ and $\max(\gamma,0)/k_y$ for RDW-like vs IDW-like,
- `scan_panel_*.png`: a compact scan diagnostic for each branch,
- `eigenfunctions_*.png` and `spectrum_*.png`: mode structure and Ritz spectrum at $k_{y,*}$.

### Interpreting the branches

In this demo, we label:

- **RDW-like**: small electron inertia (`me_hat` small) and finite resistivity (`eta` moderate),
- **IDW-like**: finite inertia (`me_hat` larger) and weak resistivity (`eta` small).

This matches the branch-separation picture emphasized in Mosetto (2012), where different closures
dominate depending on collisionality/inertia ordering.

## Ballooning-like scan (curvature on)

Run:

```bash
python examples/06_literature_tokamak_sol/mosetto2012_ballooning_branches.py
```

This script:

- turns on curvature,
- varies magnetic shear (`shat`) in a simple slab model,
- compares a resistive-like vs inertial-like ballooning branch.

The results show the trend that increasing shear can reduce growth in curvature-driven
modes (depending on parameter choices), consistent with many ballooning discussions.

## Notes on quantitative comparisons

To match Mosetto (2012) figures quantitatively you will generally need:

- electromagnetic effects (finite beta, $A_\parallel$),
- the same normalization and operator definitions used in their code,
- matching boundary conditions and field-line connection length,
- inclusion of additional closure terms that are omitted in the current default model.

The point of the `examples/06_literature_tokamak_sol/` scripts is to provide a *transparent, hackable reference*
for these workflows within a JAX-based matrix-free linear solver.
