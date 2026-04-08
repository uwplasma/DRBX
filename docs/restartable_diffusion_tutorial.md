# Restartable Diffusion Tutorial

This example is the first end-to-end runtime tutorial for `jax_drb` as a standalone code path, not just a parity harness utility.

Entry point:

- [examples/restartable_diffusion_tutorial.py](/Users/rogerio/local/jax_drb/examples/restartable_diffusion_tutorial.py)

What it demonstrates:

- how to define a small TOML input deck directly in Python;
- how to choose mesh resolution, timestep, `nout`, diffusion coefficient, and initial conditions explicitly;
- how to choose runtime precision explicitly (`float64` or `float32`);
- how to call the bare `jax_drb input.toml` CLI programmatically;
- how to write summary JSON, full-result `.npz`, restart `.npz`, and verbose run-log JSON artifacts;
- how to resume from a saved restart bundle;
- how to read the saved `.npz` files back in and make Matplotlib 2D, 3D, and movie outputs.

## Run It

```bash
PYTHONPATH=src .venv/bin/python examples/restartable_diffusion_tutorial.py
```

Quiet mode:

```bash
PYTHONPATH=src .venv/bin/python examples/restartable_diffusion_tutorial.py --quiet
```

Choose a different output location:

```bash
PYTHONPATH=src .venv/bin/python examples/restartable_diffusion_tutorial.py \
  --output-root /tmp/jax_drb_restart_demo
```

Skip the GIF and only write static figures:

```bash
PYTHONPATH=src .venv/bin/python examples/restartable_diffusion_tutorial.py \
  --output-root /tmp/jax_drb_restart_demo \
  --skip-movie
```

Run the same tutorial in `float32`:

```bash
PYTHONPATH=src .venv/bin/python examples/restartable_diffusion_tutorial.py \
  --precision float32
```

Keep `float64` in the input deck but force the CLI/runtime override path explicitly:

```bash
PYTHONPATH=src .venv/bin/python examples/restartable_diffusion_tutorial.py \
  --cli-precision-override float32
```

Keep `float64` in the input deck but force a driver-side CLI override:

```bash
PYTHONPATH=src .venv/bin/python examples/restartable_diffusion_tutorial.py \
  --precision float64 \
  --cli-precision-override float32
```

## Generated Artifacts

The script writes:

- `input/input.toml`
- `run_first/<case>_summary.json`
- `run_first/<case>_arrays.npz`
- `run_first/<case>_restart.npz`
- `run_first/<case>_run_log.json`
- `run_resumed/<case>_resumed_summary.json`
- `run_resumed/<case>_resumed_arrays.npz`
- `run_resumed/<case>_resumed_restart.npz`
- `run_resumed/<case>_resumed_run_log.json`
- `run_full/<case>_full_arrays.npz`
- `data/<case>_combined_history.npz`
- `data/<case>_analysis.json`
- `images/<case>_density_snapshots.png`
- `images/<case>_restart_consistency.png`
- `images/<case>_density_surface.png`
- `movies/<case>_density.gif`

A QA-checked example output package from a local run is currently staged under:

- [docs/data/restartable_diffusion_demo_artifacts](/Users/rogerio/local/jax_drb/docs/data/restartable_diffusion_demo_artifacts)
- [docs/data/restartable_diffusion_demo_artifacts/input/input.toml](/Users/rogerio/local/jax_drb/docs/data/restartable_diffusion_demo_artifacts/input/input.toml)
- [docs/data/restartable_diffusion_demo_artifacts/images/restartable_diffusion_density_surface.png](/Users/rogerio/local/jax_drb/docs/data/restartable_diffusion_demo_artifacts/images/restartable_diffusion_density_surface.png)
- [docs/data/restartable_diffusion_demo_artifacts/movies/restartable_diffusion_density.gif](/Users/rogerio/local/jax_drb/docs/data/restartable_diffusion_demo_artifacts/movies/restartable_diffusion_density.gif)

The companion precision benchmark is:

- [examples/diffusion_precision_benchmark.py](/Users/rogerio/local/jax_drb/examples/diffusion_precision_benchmark.py)
- [docs/runtime_precision_benchmark/data/diffusion_precision_analysis.json](/Users/rogerio/local/jax_drb/docs/runtime_precision_benchmark/data/diffusion_precision_analysis.json)
- [docs/runtime_precision_benchmark/images/diffusion_precision_elapsed.png](/Users/rogerio/local/jax_drb/docs/runtime_precision_benchmark/images/diffusion_precision_elapsed.png)

## What To Edit First

The tutorial is meant to be modified by users directly. The most important functions are:

- `build_settings(...)`
- `build_input_text(...)`
- `write_input_file(...)`
- `run_segment(...)`
- `stitch_histories(...)`
- `plot_density_snapshots(...)`
- `plot_restart_consistency(...)`
- `plot_density_surface(...)`
- `render_density_movie(...)`

That is the intended learning surface for custom cases: change the TOML deck text, rerun, inspect the saved `.npz` files and run-log JSON, then adapt the plotting functions to your own fields.
