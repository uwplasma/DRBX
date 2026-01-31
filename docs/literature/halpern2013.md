# Halpern (2013): gradient removal and SOL width

Halpern et al. (Phys. Plasmas 20, 052306 (2013)) focus on how electromagnetic effects modify SOL
ballooning instabilities and show how non-linear saturation can be interpreted through a gradient
removal paradigm.

`jaxdrb` v1 does not include magnetic induction, so it does not reproduce the **ideal** ballooning
threshold quantitatively. It *does* reproduce the analysis steps:

- scanning $\gamma(k_y)$,
- computing $(\gamma/k_y)_{\max}$,
- solving the fixed-point relation for $L_p$.

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

- `examples/literature/halpern2013_gradient_removal.py`

## Running the example

```bash
python examples/literature/halpern2013_gradient_removal.py
```

Outputs:

- `out_halpern2013_gradient_removal/gamma_over_ky_two_gradients.png`
- `out_halpern2013_gradient_removal/Lp_vs_curvature0.png`
- `out_halpern2013_gradient_removal/results.npz`

In the script we vary `curvature0` as a surrogate "drive knob" (since v1 lacks beta/induction).
When adding electromagnetic physics later, this example is a natural place to connect the workflow
to a true $\beta$-scan.

