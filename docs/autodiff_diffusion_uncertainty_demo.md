# Autodiff Diffusion Uncertainty Demo

This demo adds a standard uncertainty-quantification artifact on the same compact
native differentiable diffusion lane already used for sensitivity, inverse
design, and strong scaling.

The package is intentionally narrow:

- the uncertain inputs are the four physical diffusion-design parameters;
- the scalar quantity of interest is the final active-domain density variance;
- the field quantity of interest is the radial mean of the final active-domain density;
- the comparison is between a first-order autodiff covariance pushforward and a vectorized Monte Carlo estimate on the same native solve path.

Run:

```bash
PYTHONPATH=src .venv/bin/python examples/autodiff_diffusion_uncertainty_demo.py
```

Outputs:

- `docs/data/autodiff_diffusion_uncertainty_artifacts/data/autodiff_diffusion_uncertainty_analysis.json`
- `docs/data/autodiff_diffusion_uncertainty_artifacts/data/autodiff_diffusion_uncertainty_arrays.npz`
- `docs/data/autodiff_diffusion_uncertainty_artifacts/images/autodiff_diffusion_uncertainty.png`

Interpretation:

- the left panel shows the assumed Gaussian uncertainty model on the four input parameters;
- the middle panel compares the Monte Carlo scalar QoI distribution with the Gaussian prediction from the autodiff linearization;
- the right panel compares the linearized and Monte Carlo uncertainty bands for the final radial profile.

This is the right manuscript-facing scope for now: it demonstrates that the
native differentiable lane can propagate uncertainty in both scalar and field
observables without claiming that the full 2D or 3D parity matrix is already
end-to-end differentiable.

![Autodiff diffusion uncertainty](data/autodiff_diffusion_uncertainty_artifacts/images/autodiff_diffusion_uncertainty.png)
