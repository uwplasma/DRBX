# Restartable Diffusion Tutorial

This example is the first end-to-end runtime tutorial for `jax_drb` as a standalone code path, not just a parity harness utility.

Entry point:

- [examples/restartable_diffusion_tutorial.py](/Users/rogerio/local/jax_drb/examples/restartable_diffusion_tutorial.py)

What it demonstrates:

- how to define a small BOUT-style input deck directly in Python;
- how to choose mesh resolution, timestep, `nout`, diffusion coefficient, and initial conditions explicitly;
- how to call `jax_drb run` programmatically;
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

## Generated Artifacts

The script writes:

- `input/BOUT.inp`
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
- `images/<case>_density_surface.png`
- `movies/<case>_density.gif`

## What To Edit First

The tutorial is meant to be modified by users directly. The most important functions are:

- `build_settings(...)`
- `build_input_text(...)`
- `run_segment(...)`
- `stitch_histories(...)`
- `plot_density_snapshots(...)`
- `plot_restart_consistency(...)`
- `plot_density_surface(...)`
- `render_density_movie(...)`

That is the intended learning surface for custom cases: change the input-deck text, rerun, inspect the saved `.npz` files, then adapt the plotting functions to your own fields.
