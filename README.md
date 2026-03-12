# jax_drb

`jax_drb` is a fresh JAX plasma codebase for edge and scrape-off-layer modeling. The active tree is being built from a clean implementation plan: differentiable solver kernels, CPU/GPU portability, a Python API, and a CLI that can run curated validation cases end to end.

The current validated slices are small on purpose. Each one is locked to committed baselines before the next layer of physics is added:

- density-only `one_rhs`;
- anomalous diffusion `one_step` and `short_window`;
- periodic 1D manufactured fluid `one_rhs`, `one_step`, and `short_window`;
- standalone electrostatic vorticity `one_rhs`, `one_step`, and `short_window`;
- coupled 2D drift-wave `one_rhs` and `one_step`.

## Validation Snapshots

The figures below come from the committed validation ladder. They compare native `jax_drb` outputs against the stored baseline artifacts used by the regression harness.

![Diffusion short-window parity](docs/images/diffusion_short_window_parity.png)

![Vorticity short-window parity](docs/images/vorticity_short_window_parity.png)

![Drift-wave one-step parity](docs/images/drift_wave_one_step_parity.png)

## Running Cases

Editable install:

```bash
pip install -e .[dev]
```

Run a curated native case:

```bash
PYTHONPATH=src python -m jax_drb run-case diffusion_short_window --reference-root /path/to/reference-checkout
```

Inspect the curated ladder:

```bash
PYTHONPATH=src python -m jax_drb reference-cases --reference-root /path/to/reference-checkout
```

Run the regression suite:

```bash
pytest -q
```

## Docs Map

- Validation gallery: [docs/validation_gallery.md](/Users/rogerio/local/jax_drb/docs/validation_gallery.md)
- Parity harness: [docs/parity_harness.md](/Users/rogerio/local/jax_drb/docs/parity_harness.md)
- Parity matrix: [docs/parity_matrix.md](/Users/rogerio/local/jax_drb/docs/parity_matrix.md)
- Implementation inventory: [docs/implementation_inventory.md](/Users/rogerio/local/jax_drb/docs/implementation_inventory.md)
- Full staged roadmap: [PLAN.md](/Users/rogerio/local/jax_drb/PLAN.md)
