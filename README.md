# jax_drb

`jax_drb` is a fresh JAX plasma codebase for edge and scrape-off-layer modeling. The active tree is being built from a clean implementation plan: differentiable solver kernels, CPU/GPU portability, a Python API, and a CLI that can run curated validation cases end to end.

The current validated slices are small on purpose. Each one is locked to committed baselines before the next layer of physics is added:

- density-only `one_rhs`;
- anomalous diffusion `one_step` and `short_window`;
- periodic 1D manufactured fluid `one_rhs`, `one_step`, and `short_window`;
- standalone electrostatic vorticity `one_rhs`, `one_step`, and `short_window`;
- blob2d curvature-driven `one_rhs`, `one_step`, and `short_window`;
- coupled 2D drift-wave `one_rhs`, `one_step`, and `short_window`.

## Validation Snapshots

The figures below come from the committed validation ladder. They compare native `jax_drb` outputs against the stored baseline artifacts used by the regression harness.

![Diffusion short-window parity](docs/images/diffusion_short_window_parity.png)

![Vorticity short-window parity](docs/images/vorticity_short_window_parity.png)

![Drift-wave one-step parity](docs/images/drift_wave_one_step_parity.png)

![Drift-wave short-window parity](docs/images/drift_wave_short_window_parity.png)

![Blob2d short-window parity](docs/images/blob2d_short_window_parity.png)

The current benchmark diagnostics page also includes a short-window drift-wave validation figure with measured growth/frequency extraction against the analytic dispersion target:

![Drift-wave short-window diagnostics](docs/images/drift_wave_short_window_diagnostics.png)

The staged neutral branch now also has a compact short-window benchmark target that locks the reference transient before the native stiff integrator is exposed:

![Neutral mixed short-window diagnostics](docs/images/neutral_mixed_short_window_diagnostics.png)

## Running Cases

Editable install:

```bash
pip install -e .[dev,validation]
```

Run a curated native case:

```bash
PYTHONPATH=src python -m jax_drb run-case diffusion_short_window --reference-root /path/to/reference-checkout
```

Inspect the curated ladder:

```bash
PYTHONPATH=src python -m jax_drb reference-cases --reference-root /path/to/reference-checkout
```

Generate the drift-wave short-window parity report and figure:

```bash
PYTHONPATH=src python -m jax_drb compare-drift-wave \
  /path/to/curated/drift_wave/BOUT.inp \
  references/baselines/reference_arrays/drift_wave_short_window.npz \
  /tmp/jax_drb_drift_wave_short_window_native.npz \
  --json-out docs/data/drift_wave_short_window_parity.json \
  --plot-out docs/images/drift_wave_short_window_parity.png
```

Generate the blob2d short-window parity report and figure:

```bash
PYTHONPATH=src python -m jax_drb compare-blob2d \
  references/baselines/reference_metrics/blob2d_short_window_metrics.json \
  /tmp/jax_drb_blob2d_short_window_native.npz \
  --json-out docs/data/blob2d_short_window_parity.json \
  --plot-out docs/images/blob2d_short_window_parity.png
```

Generate the compact neutral short-window benchmark report and figure:

```bash
PYTHONPATH=src python -m jax_drb analyze-neutral-mixed \
  references/baselines/reference_arrays/neutral_mixed_short_window.npz \
  --x-index 5 \
  --y-index 3 \
  --z-index 5 \
  --json-out references/baselines/reference_metrics/neutral_mixed_short_window_metrics.json \
  --plot-out docs/images/neutral_mixed_short_window_diagnostics.png
```

Re-run committed reference baselines as a smoke check:

```bash
PYTHONPATH=src python -m jax_drb validate-reference-baselines \
  --reference-root /path/to/reference-checkout \
  --case evolve_density_rhs \
  --case diffusion_one_step \
  --case vorticity_rhs
```

Run the regression suite:

```bash
pytest -q
```

## Docs Map

- Validation gallery: [docs/validation_gallery.md](/Users/rogerio/local/jax_drb/docs/validation_gallery.md)
- Drift-wave benchmark: [docs/drift_wave_benchmark.md](/Users/rogerio/local/jax_drb/docs/drift_wave_benchmark.md)
- Neutral mixed benchmark: [docs/neutral_mixed_benchmark.md](/Users/rogerio/local/jax_drb/docs/neutral_mixed_benchmark.md)
- Parity harness: [docs/parity_harness.md](/Users/rogerio/local/jax_drb/docs/parity_harness.md)
- Parity matrix: [docs/parity_matrix.md](/Users/rogerio/local/jax_drb/docs/parity_matrix.md)
- Implementation inventory: [docs/implementation_inventory.md](/Users/rogerio/local/jax_drb/docs/implementation_inventory.md)
- Full staged roadmap: [PLAN.md](/Users/rogerio/local/jax_drb/PLAN.md)
