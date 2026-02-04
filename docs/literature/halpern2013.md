# Halpern (2013): ideal ballooning and gradient removal

Halpern et al. (Phys. Plasmas 20, 052306 (2013)) focus on how electromagnetic effects modify SOL
ballooning instabilities and show how non-linear saturation can be interpreted through a gradient
removal paradigm.

`jaxdrb` supports two complementary pieces of that workflow:

- scanning $\gamma(k_y)$,
- computing $(\gamma/k_y)_{\max}$,
- solving the fixed-point relation for $L_p$.

In addition, Halpern et al. derive a reduced **ideal ballooning** eigenproblem (their Eq. (16))
that produces an s–alpha stability diagram. `jaxdrb` implements this eigenproblem directly so the
diagram can be reproduced cheaply and transparently, without conflating it with the drift-reduced
Braginskii closure set.

## The analysis pipeline

The key sequence used in the paper (around Eq. (20)) is:

1. Choose an equilibrium gradient scale length $L_p$ (or equivalently $R/L_p$).
2. Compute linear growth rates $\gamma(k_y)$ from a reduced model.
3. Use the gradient-removal proxy to estimate transport:
   $$\Gamma \propto (\gamma/k_y)_{\max}.$$
4. Determine $L_p$ self-consistently by solving:
   $$L_p = q\,(\gamma/k_y)_{\max}(L_p).$$

## `jaxdrb` implementation

The fixed-point solver is implemented in:

- `jaxdrb.analysis.lp.solve_lp_fixed_point`

and exercised in:

- `examples/scripts/06_literature_tokamak_sol/halpern2013_gradient_removal_lp.py`

## Running the example

```bash
python examples/scripts/06_literature_tokamak_sol/halpern2013_gradient_removal_lp.py
```

Outputs:

- `out/halpern2013_gradient_removal/gamma_over_ky_curves.png`
- `out/halpern2013_gradient_removal/lp_fixed_point_history.png`
- `out/halpern2013_gradient_removal/lp_scaling_curvature0.png`
- `out/halpern2013_gradient_removal/results.npz`

In the script we vary `curvature0` as a surrogate "drive knob" (since the electrostatic model
lacks beta/induction).
When adding electromagnetic physics later, this example is a natural place to connect the workflow
to a true $\beta$-scan.

## s–alpha “ideal ballooning” map (qualitative)

Halpern et al. (2013) present an ideal-ballooning s–alpha diagram obtained from a reduced
Sturm–Liouville eigenproblem (their Eq. (16)). In `jaxdrb` this is implemented in:

- `jaxdrb.linear.ideal_ballooning.ideal_ballooning_gamma_hat`

Run:

```bash
python examples/scripts/06_literature_tokamak_sol/halpern2013_salpha_ideal_ballooning_map.py
```

This example scans $(\hat{s},\alpha)$ and plots the resulting $\hat{c}(\hat{s},\alpha)$ map, along
with a coarse marginal stability curve $\alpha_{\mathrm{crit}}(\hat{s})$ extracted from the grid.

![s–alpha growth-rate map (qualitative)](../assets/images/halpern2013_salpha_gamma_map.png)
