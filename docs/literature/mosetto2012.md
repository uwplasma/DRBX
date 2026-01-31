# Mosetto (2012): drift-wave and ballooning branches

Mosetto et al. (Phys. Plasmas 19, 112103 (2012)) analyze low-frequency linear modes in the tokamak
scrape-off layer and identify distinct regimes:

- drift-wave branches (often separated into resistive vs inertial),
- ballooning-mode branches (resistive vs inertial, and an ideal branch at finite beta).

`jaxdrb` v1 is electrostatic, so it does not reproduce the ideal electromagnetic branch
quantitatively. However, it can reproduce the **workflow** of branch separation and scanning.

## What `jaxdrb` maps to in the paper

The paper uses drift-reduced Braginskii equations in a flux-tube/field-line-following representation.
In `jaxdrb`:

- curvature is toggled by `params.curvature_on` and controlled by a geometry's curvature operator,
- resistivity vs inertia is controlled by `eta` and `me_hat`,
- the background gradient drive is controlled by `omega_n` (and optionally `omega_Te`).

## Drift-wave-like scan (curvature off)

Run:

```bash
python examples/literature/mosetto2012_driftwaves.py
```

Outputs in `out_mosetto2012_driftwaves/` include:

- `gamma_ky.png`: leading growth rate vs $k_y$ for two branches,
- `gamma_over_ky.png`: $\gamma/k_y$ vs $k_y$ and the maximizing $k_{y,*}$,
- `summary.json`: the maximizing $k_{y,*}$ and $(\gamma/k_y)_{\max}$ for each branch.

### Interpreting the branches

In this v1 demo, we label:

- **RDW-like**: small electron inertia (`me_hat` small) and finite resistivity (`eta` moderate),
- **IDW-like**: finite inertia (`me_hat` larger) and weak resistivity (`eta` small).

This matches the *qualitative* idea in Mosetto (2012) that different closures dominate depending on
collisionality/inertia ordering.

## Ballooning-like scan (curvature on)

Run:

```bash
python examples/literature/mosetto2012_ballooning.py
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
- inclusion of additional closure terms that are omitted in `jaxdrb` v1.

The point of the `examples/literature/` scripts is to provide a *transparent, hackable reference*
for these workflows within a JAX-based matrix-free linear solver.

