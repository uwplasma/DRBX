# Autodiff And Scaling Examples

This page records the current focused differentiable examples and the current fixed-workload scaling artifact on the strongest compact native diffusion lane.

## Why These Examples

Differentiable PDE and differentiable simulation workflows usually show three things first:

- direct sensitivity of a scalar objective to physical parameters
- uncertainty propagation from uncertain inputs to scalar and field quantities of interest
- inverse recovery of hidden parameters from a target state
- fixed-workload scaling of the gradient-enabled kernel

That is the surface used here too. It is the same general pattern used by projects such as [JAX-FEM](https://github.com/deepmodeling/jax-fem), [JAX-MD](https://github.com/jax-md/jax-md), and the device-parallel mapping model documented in [JAX `pmap`](https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html).

## Scripts

- [examples/autodiff_diffusion_sensitivity_demo.py](../examples/autodiff_diffusion_sensitivity_demo.py)
- [examples/autodiff_diffusion_uncertainty_demo.py](../examples/autodiff_diffusion_uncertainty_demo.py)
- [examples/autodiff_diffusion_inverse_design_demo.py](../examples/autodiff_diffusion_inverse_design_demo.py)
- [examples/strong_scaling_diffusion_demo.py](../examples/strong_scaling_diffusion_demo.py)
- shared helper module: [src/jax_drb/validation/autodiff_diffusion.py](../src/jax_drb/validation/autodiff_diffusion.py)
- uncertainty helper module: [src/jax_drb/validation/autodiff_diffusion_uncertainty.py](../src/jax_drb/validation/autodiff_diffusion_uncertainty.py)

## Sensitivity Analysis

Run:

```bash
PYTHONPATH=src .venv/bin/python examples/autodiff_diffusion_sensitivity_demo.py
```

Outputs:

- analysis JSON: [docs/data/autodiff_diffusion_sensitivity_artifacts/data/autodiff_diffusion_sensitivity_analysis.json](data/autodiff_diffusion_sensitivity_artifacts/data/autodiff_diffusion_sensitivity_analysis.json)
- figure: [docs/data/autodiff_diffusion_sensitivity_artifacts/images/autodiff_diffusion_sensitivity.png](data/autodiff_diffusion_sensitivity_artifacts/images/autodiff_diffusion_sensitivity.png)

![Autodiff sensitivity](data/autodiff_diffusion_sensitivity_artifacts/images/autodiff_diffusion_sensitivity.png)

Current committed result:

- autodiff and finite-difference gradients agree closely on all four design parameters
- the center parameter is the dominant sensitivity on this setup
- the diffusivity tangent matches the explicit sweep in the local neighborhood

## Uncertainty Quantification

Run:

```bash
PYTHONPATH=src .venv/bin/python examples/autodiff_diffusion_uncertainty_demo.py
```

Outputs:

- analysis JSON: [docs/data/autodiff_diffusion_uncertainty_artifacts/data/autodiff_diffusion_uncertainty_analysis.json](data/autodiff_diffusion_uncertainty_artifacts/data/autodiff_diffusion_uncertainty_analysis.json)
- arrays NPZ: [docs/data/autodiff_diffusion_uncertainty_artifacts/data/autodiff_diffusion_uncertainty_arrays.npz](data/autodiff_diffusion_uncertainty_artifacts/data/autodiff_diffusion_uncertainty_arrays.npz)
- figure: [docs/data/autodiff_diffusion_uncertainty_artifacts/images/autodiff_diffusion_uncertainty.png](data/autodiff_diffusion_uncertainty_artifacts/images/autodiff_diffusion_uncertainty.png)

![Autodiff uncertainty](data/autodiff_diffusion_uncertainty_artifacts/images/autodiff_diffusion_uncertainty.png)

Current committed result:

- the scalar quantity of interest uses the final active-domain density variance on the same compact native diffusion lane;
- the field quantity of interest uses the radial mean of the final active-domain density;
- first-order autodiff covariance pushforward and vectorized Monte Carlo stay close on both the scalar QoI and the profile uncertainty band;
- this gives the differentiable lane a standard UQ example rather than stopping at gradients and inverse design only.

## Inverse Design

Run:

```bash
PYTHONPATH=src .venv/bin/python examples/autodiff_diffusion_inverse_design_demo.py
```

Outputs:

- analysis JSON: [docs/data/autodiff_diffusion_inverse_design_artifacts/data/autodiff_diffusion_inverse_design_analysis.json](data/autodiff_diffusion_inverse_design_artifacts/data/autodiff_diffusion_inverse_design_analysis.json)
- figure: [docs/data/autodiff_diffusion_inverse_design_artifacts/images/autodiff_diffusion_inverse_design.png](data/autodiff_diffusion_inverse_design_artifacts/images/autodiff_diffusion_inverse_design.png)

![Autodiff inverse design](data/autodiff_diffusion_inverse_design_artifacts/images/autodiff_diffusion_inverse_design.png)

Current committed result:

- the objective drops from about `2.95e-3` to about `5.52e-5`
- the optimized final-state profile nearly overlays the target profile
- the optimized design parameters recover the dominant target structure cleanly

## Strong Scaling

Run locally on CPU only:

```bash
PYTHONPATH=src .venv/bin/python examples/strong_scaling_diffusion_demo.py \
  --skip-gpu \
  --cpu-device-counts 1,2,4,8
```

The CPU artifact now measures two distinct local modes:

- `process_group`: one Python worker per CPU partition
- `host_pmap`: one process with `JAX_DRB_HOST_DEVICE_COUNT=N` and device-parallel `pmap`

To inspect the runtime mode directly in a fresh process:

```bash
JAX_DRB_HOST_DEVICE_COUNT=4 PYTHONPATH=src .venv/bin/python - <<'PY'
from jax_drb.runtime import runtime_parallel_summary
import json
print(json.dumps(runtime_parallel_summary(), indent=2, sort_keys=True))
PY
```

Run the optional remote GPU benchmark:

```bash
PYTHONPATH=src .venv/bin/python examples/strong_scaling_diffusion_demo.py \
  --cpu-device-counts 1,2,4,8 \
  --gpu-device-counts 1,2 \
  --remote-host office
```

Outputs:

- analysis JSON: [docs/data/strong_scaling_diffusion_artifacts/data/strong_scaling_diffusion_analysis.json](data/strong_scaling_diffusion_artifacts/data/strong_scaling_diffusion_analysis.json)
- figure: [docs/data/strong_scaling_diffusion_artifacts/images/strong_scaling_diffusion.png](data/strong_scaling_diffusion_artifacts/images/strong_scaling_diffusion.png)

![Strong scaling](data/strong_scaling_diffusion_artifacts/images/strong_scaling_diffusion.png)

Current committed result:

- local CPU process-group reference: about `1.25x` speedup from `1 -> 8`
- local CPU host-device `pmap`: about `1.08x` from `1 -> 2`, `1.27x` from `1 -> 4`, and `1.25x` from `1 -> 8`
- the currently committed artifact was regenerated locally with `--skip-gpu`, so the figure emphasizes the two CPU modes on this MacBook rather than repeating the earlier remote GPU line

Interpretation:

- the host-device `pmap` curve is the cleanest demonstration that several CPU cores can be used explicitly from JAX on this machine
- the process-group curve remains a useful reference for Python-level task parallelism
- both CPU curves are still modest strong-scaling results on a small fixed workload, not a claim that CPU splitting replaces accelerator execution
- all curves are measured on a differentiable objective, not just a forward solve

## Notes On Method

- the objective is evaluated on the compact native diffusion lane because it is already JAX-native and differentiable end to end
- the CPU benchmark uses one JAX worker process per local worker to avoid oversubscribing host threads
- the explicit host-device CPU benchmark uses `JAX_DRB_HOST_DEVICE_COUNT=N` to expose multiple CPU devices before import, then maps the objective with `pmap`
- the optional GPU benchmark uses `pmap` on the remote two-GPU machine
- the total workload is held fixed, so the figure is a strong-scaling plot rather than a throughput plot

## Follow-On Work

The next higher-value differentiable examples should be:

- vorticity or drift-wave sensitivity/UQ of a scalar QoI
- an inverse-design example with boundary/source controls rather than only initial-condition controls
- memory and compilation-cache measurements alongside the current timing plot
- the first promoted differentiable recycling/open-field transient lane once the native transient backbone is closed
