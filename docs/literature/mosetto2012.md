# Mosetto (2012): drift-wave and ballooning branches

Mosetto et al. (Phys. Plasmas 19, 112103 (2012)) analyze low-frequency linear modes in the tokamak
scrape-off layer and identify distinct regimes:

- drift-wave branches (often separated into resistive vs inertial),
- ballooning-mode branches (resistive vs inertial, and an ideal branch at finite beta).

The default `jaxdrb` model is electrostatic, so it does not reproduce the ideal electromagnetic
branch quantitatively. An electromagnetic extension model is available (`--model em`), but it is
still intended for qualitative trends and methodology. `jaxdrb` can reproduce the **workflow** of
branch separation and scanning.

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
python examples/scripts/06_literature_tokamak_sol/mosetto2012_regime_map.py
```

This example computes a **proxy regime map** by evaluating four branch candidates and selecting
the one with the largest growth rate over a ky scan, then classifies it using two simple
ablations:

- **DW vs BM:** compare growth with curvature on/off,
- **inertial vs resistive:** compare growth with electron inertia on/off (`me_hat=0`).

![Mosetto-style regime map (proxy)](../assets/images/mosetto2012_regime_map.png)

## Drift-wave-like scan (curvature off)

Run:

```bash
python examples/scripts/06_literature_tokamak_sol/mosetto2012_driftwave_branches.py
```

Outputs in `out/mosetto2012_driftwave_branches/` include:

- `branches_overlay.png`: $\gamma(k_y)$ and $\max(\gamma,0)/k_y$ for RDW-like vs IDW-like,
- `scan_panel_*.png`: a compact scan diagnostic for each branch,
- `eigenfunctions_*.png` and `spectrum_*.png`: mode structure and Ritz spectrum at $k_{y,*}$.

### Interpreting the branches

In this demo, we label:

- **RDW-like**: small electron inertia (`me_hat` small) and finite resistivity (`eta` moderate),
- **IDW-like**: finite inertia (`me_hat` larger) and weak resistivity (`eta` small).

This matches the *qualitative* idea in Mosetto (2012) that different closures dominate depending on
collisionality/inertia ordering.

## Ballooning-like scan (curvature on)

Run:

```bash
python examples/scripts/06_literature_tokamak_sol/mosetto2012_ballooning_branches.py
```

This script:

- turns on curvature,
- varies magnetic shear (`shat`) in a simple slab model,
- compares a resistive-like vs inertial-like ballooning branch.

The results show the qualitative trend that increasing shear can reduce growth in curvature-driven
modes (depending on parameter choices), consistent with many ballooning discussions.

## Notes on quantitative comparisons

To match Mosetto (2012) figures quantitatively you will generally need:

- electromagnetic effects (finite beta, $A_\parallel$),
- the same normalization and operator definitions used in their code,
- matching boundary conditions and field-line connection length,
- inclusion of additional closure terms that are omitted in the current default model.

The point of the `examples/scripts/06_literature_tokamak_sol/` scripts is to provide a *transparent, hackable reference*
for these workflows within a JAX-based matrix-free linear solver.
